"""
step2_inject_labels.py

Injects manually assigned labels into step2_results.json
and runs the full threshold calibration immediately.

Run from project root:
    python step2_inject_labels.py
"""

import json
import sys

RESULTS_FILE = "step2_results.json"

# ── Labels (by query ID, 1-indexed) ───────────────────────────────────────────
LABELS = {
    1:  "BAD",   # Cannot answer — Adam optimizer info exists in paper
    2:  "GOOD",  # 8 attention heads, cited
    3:  "GOOD",  # 6 layers, cited
    4:  "GOOD",  # d_model=512, FFN=2048, cited
    5:  "GOOD",  # Dropout 0.1, cited
    6:  "BAD",   # Incorrect — training steps vary, not single fixed value
    7:  "BAD",   # Hallucinated extra GPUs — paper uses P100s
    8:  "GOOD",  # BLEU 28.4, cited
    9:  "BAD",   # Garbled dimensionality expression
    10: "GOOD",  # Correct scaling formula and reasoning
    11: "GOOD",  # Positional encoding, correct + cited
    12: "GOOD",  # Encoder-decoder attention, correct + cited
    13: "GOOD",  # Feed-forward purpose, correct + cited
    14: "GOOD",  # Dot product scaling, correct reasoning
    15: "GOOD",  # Self-attention advantages, grounded in Section 4
    16: "GOOD",  # Transformer vs seq2seq, correct
    17: "GOOD",  # Encoder vs decoder self-attention, cited
    18: "BAD",   # Speculative — paper doesn't analyze removing pos encoding
    # Q19 and Q20 were rate-limit errors — excluded from calibration
}

# ── Load and inject ───────────────────────────────────────────────────────────
with open(RESULTS_FILE, encoding="utf-8") as f:
    results = json.load(f)

for r in results:
    if r["id"] in LABELS:
        r["label"] = LABELS[r["id"]]

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Labels injected for {len(LABELS)} queries.")

# ── Threshold calibration ─────────────────────────────────────────────────────
labelled = [r for r in results if r["label"] in ("GOOD", "BAD")]

SEP  = "─" * 70
SEP2 = "═" * 70

n_good = sum(1 for r in labelled if r["label"] == "GOOD")
n_bad  = sum(1 for r in labelled if r["label"] == "BAD")

print(f"\n{SEP2}")
print("THRESHOLD CALIBRATION")
print(f"Labelled: {len(labelled)}  GOOD: {n_good}  BAD: {n_bad}")
print(SEP2)

# Full table sorted by confidence
print(f"\n{'#':>3}  {'Conf':>6}  {'Faith':>6}  {'Relev':>6}  {'Label':>5}  Query")
print("─" * 80)
for r in sorted(labelled, key=lambda x: x["confidence"], reverse=True):
    q = r["query"][:45] + "..." if len(r["query"]) > 48 else r["query"]
    print(f"{r['id']:>3}  {r['confidence']:>6.1f}  {r['faithfulness']:>6.3f}  "
          f"{r['answer_relevancy']:>6.3f}  {r['label']:>5}  {q}")

print(f"\n{SEP}")
print("THRESHOLD ANALYSIS")
print(SEP)

best_threshold = None
best_accuracy  = -1
best_stats     = None

for threshold in [50, 55, 60, 65, 70]:
    tp = sum(1 for r in labelled if r["confidence"] >= threshold and r["label"] == "GOOD")
    tn = sum(1 for r in labelled if r["confidence"] <  threshold and r["label"] == "BAD")
    fp = sum(1 for r in labelled if r["confidence"] >= threshold and r["label"] == "BAD")
    fn = sum(1 for r in labelled if r["confidence"] <  threshold and r["label"] == "GOOD")

    accuracy  = (tp + tn) / len(labelled) if labelled else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    misses = []
    for r in labelled:
        pred = "GOOD" if r["confidence"] >= threshold else "BAD"
        if pred != r["label"]:
            misses.append(
                f"    Q{r['id']:02d} conf={r['confidence']:.1f} → predicted {pred}, "
                f"actually {r['label']}: {r['query'][:55]}"
            )

    marker = ""
    if accuracy > best_accuracy:
        best_accuracy  = accuracy
        best_threshold = threshold
        best_stats     = {"tp":tp,"tn":tn,"fp":fp,"fn":fn,
                          "accuracy":accuracy,"precision":precision,
                          "recall":recall,"misses":misses}
        marker = "  ← BEST"

    print(f"\nThreshold {threshold}:  accuracy={accuracy:.0%}  "
          f"precision={precision:.0%}  recall={recall:.0%}  "
          f"TP={tp} TN={tn} FP={fp} FN={fn}{marker}")
    for m in misses:
        print(m)

# ── Verdict ───────────────────────────────────────────────────────────────────
print(f"\n{SEP2}")
print(f"RECOMMENDED THRESHOLD: {best_threshold}")
s = best_stats
print(f"Accuracy: {s['accuracy']:.0%}  Precision: {s['precision']:.0%}  Recall: {s['recall']:.0%}")
print(f"TP={s['tp']}  TN={s['tn']}  FP={s['fp']}  FN={s['fn']}")

if s["misses"]:
    print(f"\nMisclassified at threshold {best_threshold}:")
    for m in s["misses"]:
        print(m)

# Retry gate — lowest confidence BAD answer
bad_answers = sorted(
    [r for r in labelled if r["label"] == "BAD"],
    key=lambda x: x["confidence"]
)
if bad_answers:
    worst = bad_answers[0]
    print(f"\nRETRY GATE QUERY (lowest-confidence BAD answer that retry must fix):")
    print(f"  Q{worst['id']:02d}: \"{worst['query']}\"")
    print(f"  Confidence: {worst['confidence']:.1f}/100")

print(f"\n{SEP2}")
print("Step 2 complete. Report threshold + retry gate query to proceed to Step 3.")
print(SEP2)