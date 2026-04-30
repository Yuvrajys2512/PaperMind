"""
ingestion/retry_engine.py — Phase 3, Steps 3 & 4

Step 3 (this file, initial version): failure diagnosis only.
Step 4 will add retry_query() on top of this.

Diagnosis logic
---------------
Two failure modes:

  RETRIEVAL FAILURE — the right chunks were never found.
    Signals:
      - Top rerank score is negative (CrossEncoder found nothing relevant)
      - Faithfulness near zero ("cannot answer" responses)
    Retry strategy (Step 4): expand retrieval_k, rewrite query.

  GENERATION FAILURE — right chunks retrieved, LLM went off-script.
    Signals:
      - Top rerank score is positive (chunks were relevant)
      - Faithfulness is below 0.80 (LLM not grounded despite good chunks)
    Retry strategy (Step 4): stricter prompt, lower temp, reduce llm_k.

Public API (Step 3)
-------------------
diagnose_failure(query, answer, chunks, eval_scores) -> "retrieval" | "generation"
"""

from __future__ import annotations

# ── Diagnosis thresholds ──────────────────────────────────────────────────────
# Rerank score below this = CrossEncoder found nothing relevant.
_RERANK_RELEVANCE_THRESHOLD = 0.0

# When retrieval FAILED: faithfulness below this confirms it.
_FAITHFULNESS_LOW = 0.40

# When retrieval SUCCEEDED (positive rerank): faithfulness below this means
# LLM went off-script despite good chunks → generation failure.
# Set at 0.80 — a well-grounded answer on good chunks should score high.
_FAITHFULNESS_GENERATION_THRESHOLD = 0.80

# Answer relevancy below this = answer is off-topic.
_RELEVANCY_LOW = 0.40


def _extract_rerank_scores(chunks: list) -> list[float]:
    """Extract CrossEncoder rerank scores from chunk dicts."""
    scores = []
    for c in chunks:
        if isinstance(c, dict) and "score" in c:
            scores.append(float(c["score"]))
    return scores


def diagnose_failure(
    query: str,
    answer: str,
    chunks: list,
    eval_scores: dict,
) -> str:
    """
    Diagnose why an answer failed to meet the confidence threshold.

    Parameters
    ----------
    query       : original user query
    answer      : LLM-generated answer
    chunks      : retrieved+reranked chunks (dicts with 'text', optionally 'score')
    eval_scores : output of evaluate_answer(): {faithfulness, answer_relevancy}

    Returns
    -------
    "retrieval"  — retry should change the query / expand k
    "generation" — retry should tighten the prompt / reduce k
    """
    faithfulness     = eval_scores.get("faithfulness", 0.0)
    answer_relevancy = eval_scores.get("answer_relevancy", 0.0)

    # ── Rerank signal ─────────────────────────────────────────────────────────
    rerank_scores = _extract_rerank_scores(chunks)

    if rerank_scores:
        top_score = max(rerank_scores)
        retrieval_ok = top_score >= _RERANK_RELEVANCE_THRESHOLD
    else:
        # No rerank scores — use faithfulness as proxy.
        # Near-zero faithfulness = generator found nothing to ground on.
        retrieval_ok = faithfulness >= 0.05

    # ── Decision ──────────────────────────────────────────────────────────────
    #
    # If retrieval failed (no relevant chunks found) → retrieval failure.
    # If retrieval succeeded but faithfulness is still low → generation failure
    #   (LLM had good chunks but went off-script).
    # If both are ambiguous → default to retrieval (cheaper fix first).

    if not retrieval_ok:
        failure_type = "retrieval"
    elif faithfulness < _FAITHFULNESS_GENERATION_THRESHOLD:
        # Retrieval was fine but answer not well-grounded → generation failure
        failure_type = "generation"
    elif answer_relevancy < _RELEVANCY_LOW:
        failure_type = "generation"
    else:
        # Ambiguous — default to retrieval
        failure_type = "retrieval"

    # ── Log ───────────────────────────────────────────────────────────────────
    top_str = f"{max(rerank_scores):.3f}" if rerank_scores else "N/A"
    print(f"[diagnose] faith={faithfulness:.3f}  relev={answer_relevancy:.3f}  "
          f"top_rerank={top_str}  retrieval_ok={retrieval_ok}  -> {failure_type} failure")

    return failure_type