import sys
import json
sys.path.insert(0, r"c:\Users\Yuvraj Srivastava\Desktop\Projects\PaperMind")

from ingestion.embedder import embed_and_store

paper_name = "rag"

with open("scratch/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks. Embedding...")
embed_and_store(chunks, paper_name)
print("Done embedding!")
