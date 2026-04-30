# PaperMind Phase 3 — Progress Summary & Technical Overview

## The Big Picture: What is Phase 3?
In Phases 1 & 2, you built a pipeline that ingests a paper and answers questions using hybrid search and LLM generation. However, LLMs hallucinate. **Phase 3 introduces "Layer 5 — Self-Evaluation."** 
Instead of blindly returning an answer to the user, the system now:
1. Generates an answer.
2. Evaluates its own answer for accuracy and relevance.
3. If the answer is poor, it diagnoses *why* it failed.
4. It retries the process with a corrected strategy (e.g., re-phrasing the question or changing the search parameters).

---

## What is Built So Far

### Step 1: The Evaluation Engine
**Files:** `evaluator.py`, `test_evaluator.py`

*   **What it does:** It takes the user's query, the LLM's generated answer, and the retrieved chunks, and scores the answer on two metrics:
    *   **Faithfulness (0 to 1):** Splits the answer into sentences and checks if each sentence is semantically supported by the source chunks. (Did the LLM make things up?)
    *   **Answer Relevancy (0 to 1):** Checks the similarity between the original query and the generated answer. (Did the LLM actually answer the user's question?)
    *   **Confidence Score (0 to 100):** A weighted combination of the above (70% Faithfulness + 30% Relevancy).
*   **How it works (The Clever Part):** The Phase 3 handoff initially suggested using the `ragas` library, which requires an expensive OpenAI API key. `evaluator.py` brilliantly bypasses this by using a **local embedding model** (`all-MiniLM-L6-v2`) and cosine similarity to do the scoring entirely locally, for free.
*   **What works:** `test_evaluator.py` successfully verifies this logic against 5 distinct test queries to ensure the scoring behaves predictably (e.g., scoring factual answers highly and appropriately penalizing queries with vocabulary mismatches like Q8).

### Step 2: Data-Driven Threshold Calibration
**Files:** `step2_collect.py`, `step2_label.py`, `step2_inject_labels.py`

*   **What it does:** To trigger a "retry", the system needs a cutoff line (e.g., "retry if confidence < 60"). Instead of guessing this number, you built a scientific calibration suite.
*   **How it works:** 
    *   `step2_collect.py` runs 20 diverse test questions through the Phase 2 pipeline and scores them with your new `evaluator.py`, saving the raw data to `step2_results.json`.
    *   `step2_label.py` / `step2_inject_labels.py` takes those answers and asks for a human ground-truth label (GOOD or BAD). It then tests various confidence thresholds (50, 55, 60, 65, 70) to see which threshold yields the best accuracy, precision, and recall.
*   **Why it's significant:** This ensures your system isn't retrying needlessly on good answers, nor is it letting bad answers slip through. It also outputs the **"Retry Gate Query"**—the worst-performing question that your Retry Engine must be able to fix later.

### Step 3: Failure Diagnosis
**Files:** `test_diagnosis.py`, (and `diagnose_failure` inside `retry_engine.py`)

*   **What it does:** If an answer falls below the confidence threshold, the system must know *why* it failed before it can fix it.
*   **How it works:** `diagnose_failure` looks at the evaluation scores and the CrossEncoder reranker scores.
    *   **"Retrieval Failure":** If the reranker scored all chunks negatively, or if faithfulness is `0.0`, it means the system simply didn't find the right text in the PDF.
    *   **"Generation Failure":** If the reranker scored the chunks highly, but faithfulness is low, it means the right text *was* retrieved, but the LLM hallucinated or ignored it.
*   **What works:** `test_diagnosis.py` effectively proves this logic works by feeding it a mock "cannot answer" scenario and a mock "LSTM hallucination" scenario. It accurately identifies which is which without needing to make any Groq API calls.

### Step 4: The Retry Engine (Currently in Progress)
**File:** `retry_engine.py`

*   **What it does:** This is the recovery mechanism (`retry_query`). It allows the system a maximum of 3 attempts to get the answer right.
*   **How it works:** Logic is written to execute different strategies based on the diagnosis:
    *   **If Retrieval Failed:** It uses a secondary LLM call (`_expand_query`) to rewrite the user's query into dense academic vocabulary (e.g., changing "tokens" to "scaled dot-product attention") and increases the number of chunks retrieved (`top_k`).
    *   **If Generation Failed:** It keeps the same query but forces the LLM to be stricter by lowering the `temperature` (making it less creative/hallucinatory) and reducing the `llm_k` (giving it fewer chunks so it doesn't get confused).
*   **What doesn't work / What's left:** The logic is written, but it still needs to be **tested**. Specifically, a test script (`test_retry.py`) is needed to pass the notorious "Retry Gate Query" (Q8: "How is relevance computed between tokens?") into `retry_query` and prove that the query expansion actually fetches the correct chunks on the 2nd attempt.

---

## Summary of Next Steps
The infrastructure for evaluating answers, finding the statistical threshold for failure, and diagnosing why failures happen is successfully laid down. 

To complete Phase 3:
1. **Finish Step 4:** Verify that `retry_engine.py` successfully recovers a failed query (like Q8) by finishing `test_retry.py`.
2. **Build Step 5 (Graceful Degradation):** Ensure that if all 3 retry attempts fail, the system returns a polite warning instead of crashing.
3. **Build Step 6 (Full Pipeline Integration):** Create a final `pipeline.py` script that neatly wraps the intent router, generator, evaluator, and retry engine into one clean function.
