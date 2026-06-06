"""
test_parser_v3.py  (place this in the root directory)

Structure expected:
  root/
  ├── test_parser_v3.py
  ├── data/
  │   └── BERT paper.pdf
  └── ingestion/
      └── pdf_parser.py
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── paths ──────────────────────────────────────────────────────────────────────
INPUT_PDF  = os.path.join(ROOT_DIR, "data", "BERT paper.pdf")
OUTPUT_TXT = os.path.join(ROOT_DIR, "output", "bert_extracted.txt")

# ── point Python at the ingestion folder so pdf_parser is importable ──────────
sys.path.insert(0, os.path.join(ROOT_DIR, "ingestion"))
from pdf_parser import extract_text_from_pdf, remove_credits_block, remove_references_section


def main():
    # ── sanity check ──────────────────────────────────────────────────────────
    if not os.path.exists(INPUT_PDF):
        print(f"[ERROR] PDF not found at: {INPUT_PDF}")
        print("Make sure 'BERT paper.pdf' is inside a 'data/' folder next to this script.")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

    # ── extract ───────────────────────────────────────────────────────────────
    print(f"Reading : {INPUT_PDF}")
    result = extract_text_from_pdf(INPUT_PDF)
    print(f"Pages   : {result['total_pages']}")

    # ── clean ─────────────────────────────────────────────────────────────────
    text = result["full_text"]
    text = remove_credits_block(text)
    text = remove_references_section(text)

    # ── write ─────────────────────────────────────────────────────────────────
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Written : {OUTPUT_TXT}")
    print(f"Chars   : {len(text):,}")
    print("\n── Preview (first 500 chars) ──────────────────────────────────────")
    print(text[:500])


if __name__ == "__main__":
    main()