"""
ingestion/pipeline.py — Phase 3, Step 5

Single entry point for the entire PaperMind query pipeline.

Public API
----------
answer_query(query, paper_name) -> dict
"""

from __future__ import annotations

from ingestion.query_router  import route_query
from ingestion.generator     import generate_answer
from ingestion.evaluator     import evaluate_answer, compute_confidence
from ingestion.retry_engine  import diagnose_failure, retry_query, MAX_ATTEMPTS

CONFIDENCE_THRESHOLD = 50

# If attempt 1 answer_relevancy is below this, the query is out-of-domain.
# Retrying an out-of-domain query only makes it worse (query expander maps
# nonsense into paper vocabulary). Skip retries and degrade immediately.
_OUT_OF_DOMAIN_RELEVANCY = 0.05


def answer_query(query: str, paper_name: str) -> dict:
    """
    Run the full PaperMind pipeline with self-evaluation and retry.

    Never crashes. Never returns empty.

    Returns
    -------
    {
        query, answer, confidence, attempts,
        passed, warning, failure_type, sources, query_used
    }
    """
    best_result     = None
    best_confidence = -1
    failure_type    = None
    out_of_domain   = False

    for attempt in range(1, MAX_ATTEMPTS + 1):

        # Skip retries if query is out-of-domain
        if out_of_domain and attempt > 1:
            break

        try:
            if attempt == 1:
                routed     = route_query(query, paper_name)
                generated  = generate_answer(routed["query"], routed["chunks"], routed["intents"])
                answer     = generated["answer"]
                chunks     = routed["chunks"]
                sources    = generated.get("sources", [])
                query_used = query
            else:
                retry_result = retry_query(
                    query        = query,
                    paper_name   = paper_name,
                    failure_type = failure_type,
                    attempt      = attempt,
                )
                answer     = retry_result["answer"]
                chunks     = retry_result["chunks"]
                query_used = retry_result["query_used"]
                sources    = []

            eval_scores = evaluate_answer(query, answer, chunks)
            confidence  = compute_confidence(
                eval_scores["faithfulness"],
                eval_scores["answer_relevancy"]
            )

            print(f"[pipeline] Attempt {attempt}: confidence={confidence:.1f}  "
                  f"faith={eval_scores['faithfulness']:.3f}  "
                  f"relev={eval_scores['answer_relevancy']:.3f}")

            # Detect out-of-domain on attempt 1
            if attempt == 1 and eval_scores["answer_relevancy"] < _OUT_OF_DOMAIN_RELEVANCY:
                out_of_domain = True
                print(f"[pipeline] Out-of-domain query detected "
                      f"(relevancy={eval_scores['answer_relevancy']:.3f} < {_OUT_OF_DOMAIN_RELEVANCY}). "
                      f"Skipping retries.")

            if confidence > best_confidence:
                best_confidence = confidence
                best_result = {
                    "query":        query,
                    "answer":       answer,
                    "confidence":   confidence,
                    "attempts":     attempt,
                    "passed":       confidence >= CONFIDENCE_THRESHOLD,
                    "warning":      None,
                    "failure_type": None,
                    "sources":      sources,
                    "query_used":   query_used,
                }

            if confidence >= CONFIDENCE_THRESHOLD and not out_of_domain:
                return best_result

            if attempt < MAX_ATTEMPTS and not out_of_domain:
                failure_type = diagnose_failure(query, answer, chunks, eval_scores)

        except Exception as e:
            print(f"[pipeline] Attempt {attempt} error: {e}")
            continue

    # ── Graceful degradation ──────────────────────────────────────────────────
    if best_result is None:
        best_result = {
            "query":        query,
            "answer":       "Unable to answer this question from the provided paper.",
            "confidence":   0.0,
            "attempts":     MAX_ATTEMPTS,
            "passed":       False,
            "warning":      "All pipeline attempts failed with errors.",
            "failure_type": "retrieval",
            "sources":      [],
            "query_used":   query,
        }
    else:
        if out_of_domain:
            warning = (
                "This question does not appear to be related to the paper's content. "
                "Please ask a question about the paper's topics."
            )
        else:
            warning = (
                f"Low confidence answer ({best_confidence:.1f}/100 below threshold {CONFIDENCE_THRESHOLD}). "
                f"The paper may not contain sufficient information to answer this question."
            )

        best_result["passed"]       = False
        best_result["warning"]      = warning
        best_result["failure_type"] = failure_type if not out_of_domain else "out_of_domain"

    return best_result