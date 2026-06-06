"""
test_diagnosis.py — Phase 3, Step 3 verification gate

Tests diagnose_failure() on two hand-crafted cases with ZERO Groq API calls.

Case A: retrieval failure — uses stored Q01 result from step2_results.json
Case B: generation failure — hardcoded real chunks + injected hallucinated answer

Pass criteria:
  - Case A → "retrieval"
  - Case B → "generation"
  - No crashes

Run from project root:
    python test_diagnosis.py
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.evaluator    import evaluate_answer, compute_confidence
from ingestion.retry_engine import diagnose_failure

SEP  = "─" * 70
SEP2 = "═" * 70
results = []

# ─────────────────────────────────────────────────────────────────────────────
# CASE A — Retrieval failure
# Uses the stored Q01 result — "cannot answer" with confidence 3.6
# No Groq call needed — answer and chunks already in step2_results.json
# ─────────────────────────────────────────────────────────────────────────────
print(SEP2)
print("CASE A — Expected: retrieval failure")
print("(using stored Q01 result — no API call)")
print(SEP2)

RESULTS_FILE = "step2_results.json"
if not os.path.exists(RESULTS_FILE):
    print(f"ERROR: {RESULTS_FILE} not found. Run step2_collect.py first.")
    sys.exit(1)

with open(RESULTS_FILE, encoding="utf-8") as f:
    stored = json.load(f)

q01 = next((r for r in stored if r["id"] == 1), None)
if q01 is None:
    print("ERROR: Q01 not found in step2_results.json")
    sys.exit(1)

query_a  = q01["query"]
answer_a = q01["answer"]

# Reconstruct minimal chunk list — no score field means diagnose falls back
# to faithfulness signal, which is 0.0 for this "cannot answer" response.
chunks_a = [{"text": "The Adam optimizer is used with beta1=0.9, beta2=0.98."}]

eval_a = evaluate_answer(query_a, answer_a, chunks_a)
conf_a = compute_confidence(eval_a["faithfulness"], eval_a["answer_relevancy"])

print(f"Query:        {query_a}")
print(f"Answer:       {answer_a[:120]}")
print(f"Confidence:   {conf_a:.1f}/100")
print(f"Faithfulness: {eval_a['faithfulness']:.3f}  Relevancy: {eval_a['answer_relevancy']:.3f}\n")

diagnosis_a = diagnose_failure(query_a, answer_a, chunks_a, eval_a)
pass_a = diagnosis_a == "retrieval"
results.append(("A", pass_a, diagnosis_a, "retrieval"))
print(f"\nDiagnosis: {diagnosis_a}")
print(f"{'✓ PASS' if pass_a else '✗ FAIL'} — expected 'retrieval', got '{diagnosis_a}'")


# ─────────────────────────────────────────────────────────────────────────────
# CASE B — Generation failure
# Real chunks from the paper (hardcoded) + injected hallucinated answer.
# The chunks ARE about multi-head attention (relevant), but the answer
# describes LSTMs — a hallucination. Faithfulness will be near zero.
# No Groq call needed.
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("CASE B — Expected: generation failure")
print("(real paper chunks + injected hallucinated answer — no API call)")
print(SEP2)

query_b = "How does multi-head attention work?"

# Real text from the paper — these are genuinely relevant chunks
real_chunks = [
    {
        "text": (
            "Multi-head attention allows the model to jointly attend to information "
            "from different representation subspaces at different positions. With a "
            "single attention head, averaging inhibits this. MultiHead(Q,K,V) = "
            "Concat(head1,...,headh)W^O where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)."
        ),
        "score": 2.45,   # positive rerank score = CrossEncoder found this relevant
    },
    {
        "text": (
            "An attention function can be described as mapping a query and a set of "
            "key-value pairs to an output, where the query, keys, values, and output "
            "are all vectors. The output is computed as a weighted sum of the values, "
            "where the weight assigned to each value is computed by a compatibility "
            "function of the query with the corresponding key."
        ),
        "score": 1.87,
    },
    {
        "text": (
            "Scaled Dot-Product Attention: We compute the dot products of the query "
            "with all keys, divide each by sqrt(d_k), and apply a softmax function "
            "to obtain the weights on the values."
        ),
        "score": 1.62,
    },
]

# Hallucinated answer — describes LSTMs, not attention
hallucinated_answer = (
    "Multi-head attention works by applying a single large attention layer "
    "with a recurrent state that is updated at each position. The model uses "
    "LSTM cells to maintain long-term dependencies across the sequence, and "
    "the attention scores are computed using a learned bilinear function "
    "between the hidden states. This allows the model to attend to multiple "
    "positions simultaneously through the recurrent mechanism."
)

eval_b = evaluate_answer(query_b, hallucinated_answer, real_chunks)
conf_b = compute_confidence(eval_b["faithfulness"], eval_b["answer_relevancy"])

print(f"Query:        {query_b}")
print(f"Answer:       {hallucinated_answer[:120]}...")
print(f"Confidence:   {conf_b:.1f}/100")
print(f"Faithfulness: {eval_b['faithfulness']:.3f}  Relevancy: {eval_b['answer_relevancy']:.3f}\n")

diagnosis_b = diagnose_failure(query_b, hallucinated_answer, real_chunks, eval_b)
pass_b = diagnosis_b == "generation"
results.append(("B", pass_b, diagnosis_b, "generation"))
print(f"\nDiagnosis: {diagnosis_b}")
print(f"{'✓ PASS' if pass_b else '✗ FAIL'} — expected 'generation', got '{diagnosis_b}'")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("STEP 3 VERIFICATION SUMMARY")
print(SEP2)

all_pass = all(r[1] for r in results)
for case, passed, got, expected in results:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"Case {case}: {status}  (expected={expected}, got={got})")

print()
if all_pass:
    print("Step 3 PASSED. Ready to build Step 4 — retry logic.")
else:
    print("Step 3 FAILED. Report the signals above and we fix before Step 4.")
print(SEP2)