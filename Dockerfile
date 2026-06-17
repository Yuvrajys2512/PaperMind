# PaperMind backend (FastAPI + ML deps). Host-agnostic — runs on HF Spaces
# (Docker), Render, Fly, etc. The frontend is built/hosted separately (Vercel).

FROM python:3.12-slim

# Minimal build deps some wheels fall back to; dropped from the final layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so this layer caches across code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the two models the hot path loads, so the first request after a deploy
# doesn't wait ~30s downloading weights from Hugging Face. These MUST match
# ingestion/models.py (embeddings) and ingestion/reranker.py (reranker).
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-small-en-v1.5'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy every package api.main imports: api, ingestion, discovery.
COPY api ./api
COPY ingestion ./ingestion
COPY discovery ./discovery

# HF Spaces serves on 7860 by default; ${PORT} keeps this portable to Render/Fly.
# Single worker: ChromaDB and the in-memory models are not multi-worker safe —
# scale by getting a bigger box, not more workers.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
