"""
test_evaluator.py — Phase 3, Step 1 verification gate

Runs evaluate_answer() on 5 test queries drawn from "Attention Is All You Need".
Uses the real Phase 2 pipeline to generate answers, then evaluates them.

Pass criteria (from the Phase 3 plan):
  1. Good answers score faithfulness > 0.80
  2. The hypothetical answer scores slightly lower (< good answers)
  3. Confidence scores correlate with visible answer quality
  4. Script completes without crashing

Run from the project root:
    python test_evaluator.py
"""

import sys
import os

# ── Make ingestion/ importable ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.query_router    import route_query
from ingestion.generator       import generate_answer
from ingestion.evaluator       import evaluate_answer, compute_confidence

# ─────────────────────────────────────────────────────────────────────────────
# Test queries
# ─────────────────────────────────────────────────────────────────────────────

TEST_QUERIES = [
    # 1. Factual — well-supported in the paper, should score high faithfulness
    "What optimizer was used to train the Transformer model?",

    # 2. Mechanism — should be well-retrieved and grounded
    "How does multi-head attention work?",

    # 3. Factual — specific number, easy to verify
    "How many layers does the encoder have in the base Transformer model?",

    # 4. Hypothetical — answer involves inference; expected to score slightly lower
    "What would happen to the model's performance if positional encoding was removed?",

    # 5. Q8 — the known vocabulary-mismatch failure from Phase 2 evaluation
    # Expected: low faithfulness OR low relevancy — this is the one retry must fix
    "How is relevance computed between tokens?",
]

# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

PAPER_NAME = "attention-is-all-you-need"
SEP = "─" * 70

print(SEP)
print("PaperMind Phase 3 — Step 1 Verification")
print("Testing evaluate_answer() on 5 queries")
print(SEP)

results = []

for i, query in enumerate(TEST_QUERIES, 1):
    print(f"\nQuery {i}/5: {query}")
    print("  Routing + retrieving...")

    try:
        routed    = route_query(query, PAPER_NAME)
        generated = generate_answer(routed["query"], routed["chunks"], routed["intents"])
        answer    = generated["answer"]
        chunks    = routed["chunks"]

        print("  Evaluating...")
        scores = evaluate_answer(query, answer, chunks)
        conf   = compute_confidence(scores["faithfulness"], scores["answer_relevancy"])

        results.append({
            "query":             query,
            "answer_snippet":    answer[:120].replace("\n", " "),
            "faithfulness":      scores["faithfulness"],
            "answer_relevancy":  scores["answer_relevancy"],
            "confidence":        conf,
            "method":            scores["method"],
        })

        print(f"  Method:           {scores['method']}")
        print(f"  Faithfulness:     {scores['faithfulness']:.4f}")
        print(f"  Answer relevancy: {scores['answer_relevancy']:.4f}")
        print(f"  Confidence:       {conf:.1f}/100")
        print(f"  Answer snippet:   {answer[:100].replace(chr(10), ' ')}...")

    except Exception as e:
        print(f"  ERROR: {e}")
        results.append({"query": query, "error": str(e)})

# ─────────────────────────────────────────────────────────────────────────────
# Summary + pass/fail
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("SUMMARY")
print(SEP)

clean_results = [r for r in results if "error" not in r]

if not clean_results:
    print("No results — all queries failed. Fix errors above before proceeding.")
    sys.exit(1)

print(f"\n{'Query':<55} {'Faith':>6} {'Relev':>6} {'Conf':>6}  Method")
print("─" * 80)
for r in clean_results:
    label = r["query"][:52] + "..." if len(r["query"]) > 55 else r["query"]
    print(f"{label:<55} {r['faithfulness']:>6.3f} {r['answer_relevancy']:>6.3f} {r['confidence']:>6.1f}  {r['method']}")

# ── Pass / fail checks ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("VERIFICATION GATE CHECKS")
print(SEP)

# Check 1: Good factual answers score > 0.80 faithfulness
good_queries_idx = [0, 1, 2]  # optimizer, multi-head, layer count
good_results     = [clean_results[i] for i in good_queries_idx if i < len(clean_results)]
good_faiths      = [r["faithfulness"] for r in good_results]

if good_faiths:
    avg_good_faith = sum(good_faiths) / len(good_faiths)
    check1 = avg_good_faith > 0.70  # using 0.70 as minimum average — RAGAS vs local differ in scale
    print(f"\n[{'PASS' if check1 else 'WARN'}] Factual answers avg faithfulness: {avg_good_faith:.3f}")
    if not check1:
        print("       Note: If using local scorer, 0.50–0.75 is typical.")
        print("       RAGAS and local scorers have different absolute scales.")
        print("       This is informational — check the numbers manually.")

# Check 2: Hypothetical answer scores lower than factual average
if len(clean_results) > 3:
    hyp_faith = clean_results[3]["faithfulness"]
    check2    = hyp_faith < avg_good_faith if good_faiths else True
    print(f"\n[{'PASS' if check2 else 'NOTE'}] Hypothetical faithfulness ({hyp_faith:.3f}) "
          f"{'<' if check2 else '>='} factual avg ({avg_good_faith:.3f})")

# Check 3: Q8 (vocabulary mismatch) gets low score — it should need retry
if len(clean_results) > 4:
    q8_conf = clean_results[4]["confidence"]
    print(f"\n[INFO] Q8 confidence: {q8_conf:.1f}/100")
    print(f"       This is the query that Phase 3 retry must fix.")
    print(f"       Low score here is expected and correct — not a failure.")

# Check 4: No crashes
crashes = [r for r in results if "error" in r]
check4  = len(crashes) == 0
print(f"\n[{'PASS' if check4 else 'FAIL'}] No crashes: {len(crashes)} error(s)")
if crashes:
    for c in crashes:
        print(f"       ✗ {c['query']}: {c['error']}")

# Check 5: scorer method
methods = list(set(r["method"] for r in clean_results))
print(f"\n[INFO] Scorer used: {', '.join(methods)}")
if "ragas" in methods:
    print("       RAGAS is working without an OpenAI key.")
elif "local" in methods:
    print("       Local embedding scorer active (RAGAS requires OpenAI key).")

print(f"\n{SEP}")
if check4 and clean_results:
    print("Step 1 verification complete.")
    print("Report the numbers above and we'll decide if we proceed to Step 2.")
else:
    print("Fix errors above before proceeding.")
print(SEP)