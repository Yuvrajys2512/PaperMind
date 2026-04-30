"""
step2_label.py — Phase 3, Step 2 (Part 2)

Loads step2_results.json, shows you each answer, and asks you to label
it GOOD or BAD. Then tests confidence thresholds 0.50, 0.60, 0.70
against your labels to find the best cutoff.

Run AFTER step2_collect.py:
    python step2_label.py

Your labels are saved back to step2_results.json so you can re-run
the threshold analysis any time without re-labelling.
"""

import json
import os
import sys

RESULTS_FILE = "step2_results.json"
SEP  = "─" * 70
SEP2 = "═" * 70

# ── Load results ──────────────────────────────────────────────────────────────
if not os.path.exists(RESULTS_FILE):
    print(f"ERROR: {RESULTS_FILE} not found.")
    print("Run step2_collect.py first.")
    sys.exit(1)

with open(RESULTS_FILE, encoding="utf-8") as f:
    results = json.load(f)

# ── Labelling ─────────────────────────────────────────────────────────────────
unlabelled = [r for r in results if r["label"] is None and r["method"] != "error"]

if unlabelled:
    print(SEP2)
    print("STEP 2 — Answer Labelling")
    print("For each answer, press G (good) or B (bad), then Enter.")
    print("Good = answer is correct, specific, and cited from the paper.")
    print("Bad  = answer is wrong, vague, hallucinated, or 'cannot answer'.")
    print(SEP2)

    for r in unlabelled:
        print(f"\n[{r['id']:02d}/20] Query:  {r['query']}")
        print(f"       Confidence: {r['confidence']:.1f}/100  "
              f"Faith: {r['faithfulness']:.3f}  "
              f"Relev: {r['answer_relevancy']:.3f}")
        print(f"\n       Answer:\n")

        # Print answer wrapped at 80 chars
        answer = r["answer"]
        words  = answer.split()
        line   = "       "
        for word in words:
            if len(line) + len(word) + 1 > 88:
                print(line)
                line = "       " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

        print()
        while True:
            raw = input("       Label this answer — G (good) / B (bad): ").strip().upper()
            if raw in ("G", "GOOD"):
                r["label"] = "GOOD"
                break
            elif raw in ("B", "BAD"):
                r["label"] = "BAD"
                break
            else:
                print("       Please type G or B.")

    # Save labels
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nLabels saved to {RESULTS_FILE}")

else:
    print("All answers already labelled. Running threshold analysis...")

# ── Threshold calibration ─────────────────────────────────────────────────────
labelled = [r for r in results if r["label"] in ("GOOD", "BAD")]

if not labelled:
    print("No labelled results found. Something went wrong.")
    sys.exit(1)

print(f"\n{SEP2}")
print("THRESHOLD CALIBRATION")
print(f"Labelled answers: {len(labelled)}  "
      f"(GOOD: {sum(1 for r in labelled if r['label']=='GOOD')}  "
      f"BAD: {sum(1 for r in labelled if r['label']=='BAD')})")
print(SEP2)

# Show full table sorted by confidence
print(f"\n{'#':>3}  {'Conf':>6}  {'Faith':>6}  {'Relev':>6}  {'Label':>5}  Query")
print("─" * 80)
for r in sorted(labelled, key=lambda x: x["confidence"], reverse=True):
    q = r["query"][:45] + "..." if len(r["query"]) > 48 else r["query"]
    print(f"{r['id']:>3}  {r['confidence']:>6.1f}  {r['faithfulness']:>6.3f}  "
          f"{r['answer_relevancy']:>6.3f}  {r['label']:>5}  {q}")

# Test thresholds
print(f"\n{SEP}")
print("THRESHOLD ANALYSIS")
print(SEP)
print("For each threshold — answers ABOVE = predicted GOOD, BELOW = predicted BAD")
print()

best_threshold   = None
best_accuracy    = -1
best_stats       = None

for threshold in [50, 55, 60, 65, 70]:
    tp = sum(1 for r in labelled if r["confidence"] >= threshold and r["label"] == "GOOD")
    tn = sum(1 for r in labelled if r["confidence"] <  threshold and r["label"] == "BAD")
    fp = sum(1 for r in labelled if r["confidence"] >= threshold and r["label"] == "BAD")
    fn = sum(1 for r in labelled if r["confidence"] <  threshold and r["label"] == "GOOD")

    accuracy  = (tp + tn) / len(labelled) if labelled else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Misclassified queries
    misses = []
    for r in labelled:
        pred = "GOOD" if r["confidence"] >= threshold else "BAD"
        if pred != r["label"]:
            misses.append(f"  Q{r['id']:02d} ({r['confidence']:.1f}) predicted {pred}, actually {r['label']}: {r['query'][:50]}")

    stats = {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "misses": misses,
    }

    marker = ""
    if accuracy > best_accuracy:
        best_accuracy  = accuracy
        best_threshold = threshold
        best_stats     = stats
        marker = "  ← BEST SO FAR"

    print(f"Threshold {threshold}:  accuracy={accuracy:.0%}  precision={precision:.0%}  recall={recall:.0%}  "
          f"(TP={tp} TN={tn} FP={fp} FN={fn}){marker}")
    for m in misses:
        print(m)
    print()

# ── Verdict ───────────────────────────────────────────────────────────────────
print(SEP2)
print(f"RECOMMENDED THRESHOLD: {best_threshold}")
print(f"Accuracy: {best_stats['accuracy']:.0%}  "
      f"Precision: {best_stats['precision']:.0%}  "
      f"Recall: {best_stats['recall']:.0%}")
print()

# Find the lowest-scoring BAD answer — that's our retry gate query
bad_answers = sorted(
    [r for r in labelled if r["label"] == "BAD"],
    key=lambda x: x["confidence"]
)
if bad_answers:
    worst = bad_answers[0]
    print(f"RETRY GATE QUERY (lowest-confidence BAD answer):")
    print(f"  Q{worst['id']:02d}: \"{worst['query']}\"")
    print(f"  Confidence: {worst['confidence']:.1f}/100")
    print(f"  This is the query the Step 4 retry engine must fix.")

print(SEP2)
print("Step 2 complete.")
print("Report the recommended threshold and retry gate query.")
print("We proceed to Step 3 when you confirm those two values.")
print(SEP2)