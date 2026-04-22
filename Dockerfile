# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for PDF processing
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to use Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app.py into the container
COPY . .

# Hugging Face Spaces use port 7860
EXPOSE 7860

# Start the app using Gunicorn
# Timeout is set to 600 seconds to handle large 1000-sentence PDFs
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "600", "--workers", "1", "app:app"]