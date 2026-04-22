from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
import pdfplumber
import nltk
import os

app = Flask(__name__)
CORS(app)

# ─── NLTK SETUP ──────────────────────────────────────────────────
# Download punkt tokenizer model — needed for sent_tokenize
# Downloads once, cached after that
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize

# ─── CONFIG ──────────────────────────────────────────────────────

LABELS = ["cognitive", "affective", "psychomotor"]

LABEL_TEXTS = [
    "thinking analyzing evaluating problem solving reasoning understanding concepts learning knowledge",
    "feelings emotions attitude values empathy motivation behavior social interaction response",
    "physical activity performing tasks motor skills coordination hands-on practice movement execution"
]

BATCH_SIZE = 64

# ─── LOAD MODEL AT STARTUP ───────────────────────────────────────

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
LABEL_EMBEDDINGS = model.encode(LABEL_TEXTS, convert_to_numpy=True)
print(f"Model ready. Label embeddings: {LABEL_EMBEDDINGS.shape}")


# ─── PDF EXTRACTION ──────────────────────────────────────────────

def extract_pdf_text(file):
    pages = []
    with pdfplumber.open(file) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)

    text = "\n".join(pages)

    # Fix hyphenated line breaks: "connec-\ntion" → "connection"
    import re
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Fix mid-paragraph line breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

    # Clean multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    print(f"[extract_pdf] Pages: {total_pages}, Characters: {len(text)}")
    return text.strip()


# ─── SENTENCE SPLITTING ──────────────────────────────────────────

def split_sentences(text):
    """
    Uses NLTK Punkt tokenizer.
    Handles abbreviations, decimals, academic text automatically.
    No regex — no sentences dropped incorrectly.
    """
    sentences = sent_tokenize(text)
    result = [s.strip() for s in sentences if len(s.strip()) >= 5]
    print(f"[split_sentences] Found {len(result)} sentences")
    return result


# ─── SCORING ─────────────────────────────────────────────────────

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def get_scores(sentences):
    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=False
    )

    s_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    l_norm = LABEL_EMBEDDINGS / (np.linalg.norm(LABEL_EMBEDDINGS, axis=1, keepdims=True) + 1e-9)

    sims = np.dot(s_norm, l_norm.T)
    probs = softmax(sims * 10)

    print(f"[get_scores] Scored {len(sentences)} sentences")
    return probs


# ─── CORE ANALYSIS ───────────────────────────────────────────────

def analyze_text(text):
    sentences = split_sentences(text)

    if not sentences:
        return {"error": "No valid sentences found."}

    print(f"[analyze_text] Processing {len(sentences)} sentences")

    probs = get_scores(sentences)

    results = []
    domain_counts = {"cognitive": 0, "affective": 0, "psychomotor": 0}

    for i, sentence in enumerate(sentences):
        idx = int(np.argmax(probs[i]))
        domain = LABELS[idx]
        confidence = float(round(probs[i][idx], 3))
        domain_counts[domain] += 1

        results.append({
            "text": sentence,
            "domain": domain,
            "confidence": confidence,
            "scores": {
                "cognitive":   float(round(probs[i][0], 3)),
                "affective":   float(round(probs[i][1], 3)),
                "psychomotor": float(round(probs[i][2], 3))
            }
        })

    total = len(sentences)

    # Largest remainder — always exactly 100
    exact = {d: (domain_counts[d] / total) * 100 for d in domain_counts}
    floored = {d: int(v) for d, v in exact.items()}
    rem = 100 - sum(floored.values())
    for key in sorted(exact.keys(), key=lambda x: exact[x] - floored[x], reverse=True)[:rem]:
        floored[key] += 1

    print(f"[analyze_text] Done. Counts: {domain_counts}, Percentages: {floored}")

    return {
        "sentence_count": total,
        "domain_counts": domain_counts,
        "domain_percentages": floored,
        "sentences": results
    }


# ─── ROUTES ──────────────────────────────────────────────────────

@app.route('/')
def home():
    return jsonify({"status": "Evalio ML Online", "model": "all-MiniLM-L6-v2"})


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get("text", "") if data else ""

    if not text or len(text.strip()) < 10:
        return jsonify({"error": "Text too short"}), 400

    try:
        return jsonify(analyze_text(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files accepted'}), 400

    try:
        text = extract_pdf_text(file)

        if not text or len(text.strip()) < 10:
            return jsonify({'error': 'Could not extract text from PDF'}), 400

        result = analyze_text(text)
        result["source"] = "pdf"
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── RUN ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)