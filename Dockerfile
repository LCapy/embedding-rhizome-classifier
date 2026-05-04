FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt huggingface_hub

# Pre-download LaBSE into the image so cold starts don't need to re-fetch it.
# This makes the image ~1.5 GB but startup is much faster.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/LaBSE')"

# Copy application code
COPY api/     api/
COPY scripts/ scripts/

# HF Spaces runs as a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Startup script: fetch coredrill JSON from HF Dataset repo, then launch API
COPY --chown=appuser deploy/hf_spaces/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# HF Spaces requires port 7860
EXPOSE 7860

ENV PORT=7860
ENV COREDRILL_PATH=/app/coredrill/coredrill_hierarchical.json
ENV MODEL_NAME=sentence-transformers/LaBSE

CMD ["/app/start.sh"]
