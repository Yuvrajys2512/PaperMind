"""
step2_collect.py — Phase 3, Step 2 (Part 1)

Runs all 20 test queries through the Phase 2 pipeline, evaluates each
answer with evaluate_answer(), and saves everything to step2_results.json.

Run this FIRST. It takes 2-4 minutes.
Then run step2_label.py to score and calibrate.

Run from project root:
    python step2_collect.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.query_router import route_query
from ingestion.generator    import generate_answer
from ingestion.evaluator    import evaluate_answer, compute_confidence

PAPER_NAME = "attention-is-all-you-need"

# ── 20 test queries ───────────────────────────────────────────────────────────
# Mix of: factual, mechanistic, comparative, hypothetical, specific numbers.
# Drawn from topics covered across the full paper.
QUERIES = [
    # Factual / specific
    "What optimizer was used to train the Transformer?",
    "How many attention heads does the base Transformer model use?",
    "How many layers does the encoder have in the base Transformer model?",
    "What is the dimensionality of the model in the base Transformer?",
    "What dropout rate was used during training?",
    "How many training steps were used for the base model?",
    "What hardware was used to train the Transformer?",
    "What is the BLEU score of the Transformer on WMT 2014 English-to-German translation?",

    # Mechanistic / how
    "How does multi-head attention work?",
    "How does scaled dot-product attention work?",
    "How does the Transformer handle the order of tokens in a sequence?",
    "How does the encoder pass information to the decoder?",
    "What is the purpose of the feed-forward network in each Transformer layer?",
    "Why is the dot product scaled by the square root of the key dimension?",

    # Comparative / conceptual
    "What are the advantages of self-attention over recurrent layers?",
    "How does the Transformer differ from sequence-to-sequence models with attention?",
    "What is the difference between encoder self-attention and decoder self-attention?",

    # Hypothetical / analytical
    "What would happen to the model's performance if positional encoding was removed?",
    "Why did the authors use label smoothing during training?",

    # The known hard query
    "How is relevance computed between tokens?",
]

# ─────────────────────────────────────────────────────────────────────────────

SEP = "─" * 70

print(SEP)
print("PaperMind Phase 3 — Step 2 Collection")
print(f"Running {len(QUERIES)} queries through the pipeline...")
print(SEP)

results = []

for i, query in enumerate(QUERIES, 1):
    print(f"\n[{i:02d}/{len(QUERIES)}] {query}")
    try:
        routed    = route_query(query, PAPER_NAME)
        generated = generate_answer(routed["query"], routed["chunks"], routed["intents"])
        answer    = generated["answer"]
        chunks    = routed["chunks"]

        scores = evaluate_answer(query, answer, chunks)
        conf   = compute_confidence(scores["faithfulness"], scores["answer_relevancy"])

        results.append({
            "id":               i,
            "query":            query,
            "answer":           answer,
            "faithfulness":     round(scores["faithfulness"], 4),
            "answer_relevancy": round(scores["answer_relevancy"], 4),
            "confidence":       conf,
            "method":           scores["method"],
            "label":            None,   # filled in by step2_label.py
        })

        print(f"         Confidence: {conf:.1f}/100  "
              f"Faith: {scores['faithfulness']:.3f}  "
              f"Relev: {scores['answer_relevancy']:.3f}")

    except Exception as e:
        print(f"         ERROR: {e}")
        results.append({
            "id": i, "query": query, "answer": f"ERROR: {e}",
            "faithfulness": 0.0, "answer_relevancy": 0.0,
            "confidence": 0.0, "method": "error", "label": None,
        })

# Save
out_path = "step2_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{SEP}")
print(f"Done. Results saved to {out_path}")
print(f"Errors: {sum(1 for r in results if r['method'] == 'error')}")
print(f"\nNext: run  python step2_label.py")
print(SEP)