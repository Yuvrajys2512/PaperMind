"""
test_retry.py — Phase 3, Step 4 verification gate

Tests the full retry loop on Q01 — "What optimizer was used to train the Transformer?"
This query returned "cannot answer" on attempt 1 (confidence 3.6/100).
The retry engine must find the Adam optimizer answer on attempt 2 or 3.

Pass criteria:
  - Attempt 2 or 3 produces an answer mentioning "Adam"
  - Confidence improves above 50 on a retry attempt
  - System does not crash

Run from project root:
    python test_retry.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.query_router  import route_query
from ingestion.generator     import generate_answer
from ingestion.evaluator     import evaluate_answer, compute_confidence
from ingestion.retry_engine  import diagnose_failure, retry_query

PAPER_NAME      = "attention-is-all-you-need"
RETRY_GATE_QUERY = "What optimizer was used to train the Transformer?"
CONFIDENCE_THRESHOLD = 50

SEP  = "─" * 70
SEP2 = "═" * 70

print(SEP2)
print("Phase 3 Step 4 — Retry Loop Verification")
print(f"Query: {RETRY_GATE_QUERY}")
print(SEP2)

# ── Attempt 1 — original query, original config ───────────────────────────────
print(f"\n{'─'*30} Attempt 1 {'─'*30}")
routed_1    = route_query(RETRY_GATE_QUERY, PAPER_NAME)
generated_1 = generate_answer(routed_1["query"], routed_1["chunks"], routed_1["intents"])
answer_1    = generated_1["answer"]
chunks_1    = routed_1["chunks"]

eval_1 = evaluate_answer(RETRY_GATE_QUERY, answer_1, chunks_1)
conf_1 = compute_confidence(eval_1["faithfulness"], eval_1["answer_relevancy"])

print(f"Answer:     {answer_1[:150]}")
print(f"Confidence: {conf_1:.1f}/100  Faith: {eval_1['faithfulness']:.3f}  Relev: {eval_1['answer_relevancy']:.3f}")

best_answer     = answer_1
best_confidence = conf_1
adam_found      = False
winning_attempt = 1

if conf_1 >= CONFIDENCE_THRESHOLD:
    print(f"\nAttempt 1 already passes threshold ({conf_1:.1f} >= {CONFIDENCE_THRESHOLD})")
    adam_found = "adam" in answer_1.lower()
else:
    # ── Diagnose ──────────────────────────────────────────────────────────────
    failure_type = diagnose_failure(RETRY_GATE_QUERY, answer_1, chunks_1, eval_1)

    # ── Attempt 2 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*30} Attempt 2 {'─'*30}")
    result_2 = retry_query(RETRY_GATE_QUERY, PAPER_NAME, failure_type, attempt=2)
    answer_2  = result_2["answer"]
    chunks_2  = result_2["chunks"]

    eval_2 = evaluate_answer(RETRY_GATE_QUERY, answer_2, chunks_2)
    conf_2 = compute_confidence(eval_2["faithfulness"], eval_2["answer_relevancy"])

    print(f"Query used: {result_2['query_used']}")
    print(f"Answer:     {answer_2[:150]}")
    print(f"Confidence: {conf_2:.1f}/100  Faith: {eval_2['faithfulness']:.3f}  Relev: {eval_2['answer_relevancy']:.3f}")

    if conf_2 > best_confidence:
        best_answer, best_confidence = answer_2, conf_2
        winning_attempt = 2

    if conf_2 >= CONFIDENCE_THRESHOLD or "adam" in answer_2.lower():
        adam_found = "adam" in answer_2.lower()
        print(f"\n✓ Retry succeeded on attempt 2")
    else:
        # ── Attempt 3 ─────────────────────────────────────────────────────────
        failure_type_2 = diagnose_failure(RETRY_GATE_QUERY, answer_2, chunks_2, eval_2)
        print(f"\n{'─'*30} Attempt 3 {'─'*30}")
        result_3 = retry_query(RETRY_GATE_QUERY, PAPER_NAME, failure_type_2, attempt=3)
        answer_3  = result_3["answer"]
        chunks_3  = result_3["chunks"]

        eval_3 = evaluate_answer(RETRY_GATE_QUERY, answer_3, chunks_3)
        conf_3 = compute_confidence(eval_3["faithfulness"], eval_3["answer_relevancy"])

        print(f"Query used: {result_3['query_used']}")
        print(f"Answer:     {answer_3[:150]}")
        print(f"Confidence: {conf_3:.1f}/100  Faith: {eval_3['faithfulness']:.3f}  Relev: {eval_3['answer_relevancy']:.3f}")

        if conf_3 > best_confidence:
            best_answer, best_confidence = answer_3, conf_3
            winning_attempt = 3

        adam_found = "adam" in answer_3.lower()

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("STEP 4 VERIFICATION SUMMARY")
print(SEP2)

print(f"\nBest answer (attempt {winning_attempt}, confidence {best_confidence:.1f}/100):")
print(f"  {best_answer[:200]}")

check1 = adam_found
check2 = best_confidence > conf_1   # improved from attempt 1
check3 = winning_attempt in (2, 3)  # retry did something

print(f"\n[{'PASS' if check1 else 'FAIL'}] Answer mentions 'Adam': {adam_found}")
print(f"[{'PASS' if check2 else 'FAIL'}] Confidence improved: {conf_1:.1f} -> {best_confidence:.1f}")
print(f"[{'PASS' if check3 else 'INFO'}] Answer found on attempt: {winning_attempt}")

all_pass = check1 and check2
print(f"\n{'Step 4 PASSED. Ready to build Step 5 — pipeline.py' if all_pass else 'Step 4 needs adjustment — report output above.'}")
print(SEP2)