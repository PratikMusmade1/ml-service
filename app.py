from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
import pdfplumber
import re
import os

app = Flask(__name__)
CORS(app)

# ─── CONFIG ──────────────────────────────────────────────────────

LABELS = ["cognitive", "affective", "psychomotor"]

LABEL_TEXTS = [
    "thinking analyzing evaluating problem solving reasoning understanding concepts learning knowledge",
    "feelings emotions attitude values empathy motivation behavior social interaction response",
    "physical activity performing tasks motor skills coordination hands-on practice movement execution"
]

# Render free tier has 512MB RAM and 30s request timeout
# These limits keep us well within those bounds
MAX_SENTENCES = 300   # enough for a full research paper
BATCH_SIZE = 32       # process 32 sentences at once — fast on CPU

# ─── LOAD MODEL AT MODULE LEVEL (not inside a function) ──────────
# This runs ONCE when Render starts the server
# All requests share the same loaded model — no re-loading
print("Loading MiniLM model...")
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
LABEL_EMBEDDINGS = model.encode(LABEL_TEXTS, convert_to_numpy=True)
# LABEL_EMBEDDINGS shape: (3, 384) — one vector per domain label
print("Model loaded. Server ready.")

# ─── UTILS ───────────────────────────────────────────────────────

def softmax(x):
    """
    Stable softmax — subtracts max to prevent overflow.
    * 10 is temperature scaling — sharpens the probability distribution.
    Without it all three domains get ~33% because similarities are close.
    """
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cosine_sim_matrix(sentence_embeddings, label_embeddings):
    """
    Computes cosine similarity between every sentence and every label.
    Returns matrix of shape (num_sentences, num_labels).
    
    sentence_embeddings: (N, 384)
    label_embeddings:    (3, 384)
    output:              (N, 3)
    """
    # Ensure 2D — handles both single sentence and batch
    if sentence_embeddings.ndim == 1:
        sentence_embeddings = sentence_embeddings.reshape(1, -1)

    # L2 normalize each row
    s_norm = sentence_embeddings / (np.linalg.norm(sentence_embeddings, axis=1, keepdims=True) + 1e-9)
    l_norm = label_embeddings / (np.linalg.norm(label_embeddings, axis=1, keepdims=True) + 1e-9)

    # Matrix multiply: (N, 384) x (384, 3) = (N, 3)
    return np.dot(s_norm, l_norm.T)


def split_sentences(text):
    """Split text into sentences on .!? boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def extract_pdf_text(file):
    """Extract all text from PDF using pdfplumber page by page."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text.strip()


def sample_sentences(sentences, max_count):
    """
    If too many sentences, sample evenly across the document.
    This preserves distribution across intro/body/conclusion
    instead of just taking first N sentences.
    """
    if len(sentences) <= max_count:
        return sentences
    
    step = len(sentences) / max_count
    indices = [int(i * step) for i in range(max_count)]
    return [sentences[i] for i in indices]


# ─── SCORING ─────────────────────────────────────────────────────

def get_model_scores(sentences):
    """
    Encodes all sentences in one batched call.
    Returns list of score dicts — one per sentence.
    """
    # Encode all sentences at once — batch_size controls memory usage
    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=False
    )
    # embeddings shape: (N, 384)

    # Compute similarity to all 3 domain labels at once — fully vectorized
    sims = cosine_sim_matrix(embeddings, LABEL_EMBEDDINGS)
    # sims shape: (N, 3)

    # Convert to probabilities with temperature scaling
    probs = softmax(sims * 10)
    # probs shape: (N, 3) — each row sums to 1.0

    results = []
    for i in range(len(sentences)):
        results.append({
            "cognitive":    float(round(probs[i][0], 4)),
            "affective":    float(round(probs[i][1], 4)),
            "psychomotor":  float(round(probs[i][2], 4))
        })

    return results


# ─── CORE ANALYSIS ───────────────────────────────────────────────

def analyze_text(text):
    """
    Main pipeline:
    1. Split text into sentences
    2. Sample if too many (preserves distribution)
    3. Score all sentences with MiniLM in one batched call
    4. Count domain distribution
    5. Return exact 100% percentages using largest remainder method
    """
    # Split into sentences
    all_sentences = split_sentences(text)

    if not all_sentences:
        return {"error": "No valid sentences found in the text."}

    # Sample evenly if over limit — keeps analysis fast on Render free tier
    sampled = sample_sentences(all_sentences, MAX_SENTENCES)
    was_sampled = len(all_sentences) > MAX_SENTENCES

    # Score all sentences in one batched call
    model_scores = get_model_scores(sampled)

    results = []
    domain_counts = {"cognitive": 0, "affective": 0, "psychomotor": 0}

    for i, sentence in enumerate(sampled):
        scores = model_scores[i]
        domain = max(scores, key=scores.get)
        confidence = round(scores[domain], 3)
        domain_counts[domain] += 1

        results.append({
            "text": sentence,
            "domain": domain,
            "confidence": confidence,
            "scores": {k: round(v, 3) for k, v in scores.items()}
        })

    total = len(sampled)

    # Largest remainder method — guarantees sum = exactly 100
    exact = {d: (domain_counts[d] / total) * 100 for d in domain_counts}
    floored = {d: int(v) for d, v in exact.items()}
    rem = 100 - sum(floored.values())
    sorted_keys = sorted(exact.keys(), key=lambda x: exact[x] - floored[x], reverse=True)
    for i in range(rem):
        floored[sorted_keys[i]] += 1

    response = {
        "sentence_count": total,
        "total_sentences_in_doc": len(all_sentences),
        "domain_counts": domain_counts,
        "domain_percentages": floored,
        "sentences": results
    }

    # Tell frontend if we sampled
    if was_sampled:
        response["warning"] = f"Document had {len(all_sentences)} sentences. Analyzed a representative sample of {MAX_SENTENCES}."

    return response


# ─── ROUTES ──────────────────────────────────────────────────────

@app.route('/')
def home():
    return jsonify({"status": "Evalio ML running", "model": "paraphrase-MiniLM-L3-v2"})


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


@app.route('/analyze', methods=['POST'])
def analyze():
    """Accepts plain text in JSON body."""
    data = request.get_json()
    text = data.get("text", "") if data else ""

    if not text or len(text.strip()) < 10:
        return jsonify({"error": "No text provided"}), 400

    try:
        return jsonify(analyze_text(text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    """Accepts PDF file upload, extracts text, runs analysis."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are accepted'}), 400

    try:
        text = extract_pdf_text(file)

        if not text or len(text.strip()) < 10:
            return jsonify({'error': 'Could not extract text. PDF may be scanned or image-based.'}), 400

        result = analyze_text(text)
        result["source"] = "pdf"
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─── RUN ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)