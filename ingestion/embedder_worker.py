import sys
import os
import json
from pathlib import Path

# Ensure the root directory is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.embedder import embed_and_store

def main():
    if len(sys.argv) != 3:
        print("Usage: python embedder_worker.py <json_path> <paper_name>")
        sys.exit(1)
        
    json_path = sys.argv[1]
    paper_name = sys.argv[2]
    
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
        
    print(f"[embedder_worker] Loaded {len(chunks)} chunks from {json_path}")
    embed_and_store(chunks, paper_name)
    print("[embedder_worker] Done.")

if __name__ == "__main__":
    main()
