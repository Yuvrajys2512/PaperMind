"""
Step 1 diagnostic — run this FIRST before building evaluator.py.

What it does:
  - Checks if ragas is installed
  - Checks if ragas + datasets are importable
  - Runs a tiny RAGAS evaluation with dummy data
  - Tells you exactly which scorer to use

Run with:
    python check_ragas.py
"""
from dotenv import load_dotenv
load_dotenv()



print("=" * 60)
print("PaperMind Phase 3 — Step 1 Diagnostic")
print("=" * 60)

# ── 1. Check ragas is installed ──────────────────────────────
print("\n[1/4] Checking ragas install...")
try:
    import ragas
    print(f"      ✓ ragas installed (version: {ragas.__version__})")
except ImportError:
    print("      ✗ ragas not installed.")
    print("\n      Fix: pip install ragas datasets")
    print("      Then re-run this script.\n")
    exit(1)

# ── 2. Check datasets is installed ───────────────────────────
print("\n[2/4] Checking datasets install...")
try:
    from datasets import Dataset
    print("      ✓ datasets installed")
except ImportError:
    print("      ✗ datasets not installed.")
    print("\n      Fix: pip install datasets")
    print("      Then re-run this script.\n")
    exit(1)

# ── 3. Check for OpenAI key ───────────────────────────────────
import os
print("\n[3/4] Checking for OpenAI API key...")
openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
if openai_key:
    print("      ✓ OPENAI_API_KEY found in environment")
else:
    print("      ✗ OPENAI_API_KEY not set — this is expected for this project")

# ── 4. Attempt a tiny RAGAS evaluation ───────────────────────
print("\n[4/4] Attempting RAGAS evaluation with dummy data...")
print("      (this will reveal whether RAGAS needs an OpenAI key)")

try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy

    dummy = Dataset.from_dict({
        "question": ["What is attention?"],
        "answer":   ["Attention is a mechanism that allows the model to focus on relevant parts of the input."],
        "contexts": [["Attention mechanisms allow models to focus on relevant parts of input sequences."]],
    })

    result = evaluate(dummy, metrics=[faithfulness, answer_relevancy])
    f  = float(result["faithfulness"])
    ar = float(result["answer_relevancy"])

    print(f"\n      ✓ RAGAS ran successfully without OpenAI key!")
    print(f"        faithfulness:     {f:.4f}")
    print(f"        answer_relevancy: {ar:.4f}")

    print("\n" + "=" * 60)
    print("RESULT: Use RAGAS scorer in evaluator.py")
    print("evaluator.py will use RAGAS as the primary scorer.")
    print("=" * 60)

except Exception as e:
    err = str(e)
    print(f"\n      ✗ RAGAS failed: {err[:120]}")

    if "openai" in err.lower() or "api_key" in err.lower() or "key" in err.lower():
        print("\n" + "=" * 60)
        print("RESULT: Use LOCAL embedding-similarity scorer in evaluator.py")
        print("RAGAS requires an OpenAI key. evaluator.py will use the local fallback.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print(f"RESULT: RAGAS failed for an unexpected reason.")
        print("Share the full error above and we'll fix it before proceeding.")
        print("=" * 60)