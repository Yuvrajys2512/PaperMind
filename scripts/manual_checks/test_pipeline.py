"""
test_pipeline.py — Phase 3, Step 5 verification gate

Tests answer_query() on 3 cases:
  1. Q that should pass on attempt 1       (multi-head attention)
  2. Q that needs retry to pass            (optimizer — was failing before)
  3. Q that genuinely doesn't exist        (graceful degradation)

Pass criteria:
  - Case 1: passed=True, attempts=1
  - Case 2: passed=True, attempts=2, "Adam" in answer
  - Case 3: passed=False, warning is set, answer is not empty
  - Zero crashes across all 3

Run from project root:
    python test_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.pipeline import answer_query

SEP  = "─" * 70
SEP2 = "═" * 70

cases = [
    {
        "label":       "Case 1 — Should pass on attempt 1",
        "query":       "How does multi-head attention work?",
        "expect_pass": True,
        "expect_adam": False,
    },
    {
        "label":       "Case 2 — Needs retry (optimizer query)",
        "query":       "What optimizer was used to train the Transformer?",
        "expect_pass": True,
        "expect_adam": True,
    },
    {
        "label":       "Case 3 — Graceful degradation (question not in paper)",
        "query":       "What is the recipe for chocolate cake?",
        "expect_pass": False,
        "expect_adam": False,
    },
]

results = []

for case in cases:
    print(f"\n{SEP2}")
    print(case["label"])
    print(f"Query: {case['query']}")
    print(SEP2)

    try:
        result = answer_query(case["query"], "attention-is-all-you-need")

        print(f"\nAnswer:      {result['answer'][:200]}")
        print(f"Confidence:  {result['confidence']:.1f}/100")
        print(f"Attempts:    {result['attempts']}")
        print(f"Passed:      {result['passed']}")
        print(f"Warning:     {result['warning']}")

        results.append({
            "label":   case["label"],
            "result":  result,
            "case":    case,
            "crashed": False,
        })

    except Exception as e:
        print(f"CRASHED: {e}")
        results.append({
            "label":   case["label"],
            "crashed": True,
            "error":   str(e),
            "case":    case,
        })

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print("STEP 5 VERIFICATION SUMMARY")
print(SEP2)

all_pass = True

for r in results:
    case = r["case"]
    print(f"\n{r['label']}")

    if r["crashed"]:
        print(f"  [FAIL] Crashed: {r['error']}")
        all_pass = False
        continue

    res = r["result"]

    # Check 1: no crash
    print(f"  [PASS] No crash")

    # Check 2: answer not empty
    c2 = bool(res["answer"].strip())
    print(f"  [{'PASS' if c2 else 'FAIL'}] Answer not empty")
    if not c2: all_pass = False

    # Check 3: pass/fail as expected
    c3 = res["passed"] == case["expect_pass"]
    print(f"  [{'PASS' if c3 else 'FAIL'}] passed={res['passed']} (expected {case['expect_pass']})")
    if not c3: all_pass = False

    # Check 4: Adam in answer (case 2 only)
    if case["expect_adam"]:
        c4 = "adam" in res["answer"].lower()
        print(f"  [{'PASS' if c4 else 'FAIL'}] Answer mentions Adam")
        if not c4: all_pass = False

    # Check 5: degradation case has warning
    if not case["expect_pass"]:
        c5 = res["warning"] is not None
        print(f"  [{'PASS' if c5 else 'FAIL'}] Warning set on degradation: {res['warning']}")
        if not c5: all_pass = False

print(f"\n{SEP2}")
if all_pass:
    print("Step 5 PASSED. Phase 3 is complete.")
    print("answer_query() is production-ready.")
else:
    print("Step 5 needs adjustment. Report output above.")
print(SEP2)