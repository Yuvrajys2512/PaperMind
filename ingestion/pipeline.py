"""
ingestion/pipeline.py — Phase 3 + Upgrades 1, 2, 3

Single entry point for the entire PaperMind query pipeline.

Changes in this version
-----------------------
Upgrade 1 fix : generate_answer() now receives plan (dict) not intents (list).
                route_query() now returns "plan" key, not "intents".

Upgrade 3     : Evidence grading is inserted between generation and evaluation.
                The CLEANED answer (UNSUPPORTED sentences removed) is what gets:
                  - returned to the user
                  - scored by the evaluator
                Grading metadata is surfaced in the response dict.

Public API
----------
answer_query(query, paper_name) -> dict
"""

from __future__ import annotations

from ingestion.query_router    import route_query
from ingestion.generator       import generate_answer
from ingestion.evaluator       import evaluate_answer, compute_confidence
from ingestion.retry_engine    import diagnose_failure, retry_query, MAX_ATTEMPTS
from ingestion.evidence_grader import grade_answer

CONFIDENCE_THRESHOLD = 50

# If attempt 1 answer_relevancy is below this, the query is out-of-domain.
# Retrying an out-of-domain query only makes it worse. Skip retries immediately.
_OUT_OF_DOMAIN_RELEVANCY = 0.05


def answer_query(query: str, paper_name: str) -> dict:
    """
    Run the full PaperMind pipeline:

    For each attempt (up to MAX_ATTEMPTS):
      1. Route  — plan query, retrieve chunks (query_router)
      2. Generate — CoT reasoning + structured answer (generator)
      3. Grade  — remove UNSUPPORTED sentences (evidence_grader)   ← NEW
      4. Evaluate — score the CLEANED answer (evaluator)
      5. Retry  — if confidence < threshold, diagnose & retry

    Never crashes. Never returns empty.

    Returns
    -------
    dict:
        query            : str
        answer           : str    Cleaned answer (UNSUPPORTED sentences removed)
        reasoning_chain  : str    CoT scratchpad from generator (for debug)
        confidence       : float
        faithfulness     : float
        answer_relevancy : float
        attempts         : int
        passed           : bool
        warning          : str | None
        failure_type     : str | None
        sources          : list
        query_used       : str
        grading          : dict   Per-sentence evidence grades + removed count
        plan             : dict   Query Plan from Upgrade 1
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
            # ── Step 1 & 2: Route + Generate ─────────────────────────────
            if attempt == 1:
                routed    = route_query(query, paper_name)
                generated = generate_answer(
                    query  = routed["query"],
                    chunks = routed["chunks"],
                    plan   = routed["plan"],    # ← Upgrade 1 fix: was routed["intents"]
                )
                raw_answer      = generated["answer"]
                reasoning_chain = generated.get("reasoning_chain", "")
                chunks          = routed["chunks"]
                sources         = generated.get("sources", [])
                query_used      = query
                plan            = routed["plan"]

            else:
                retry_result = retry_query(
                    query        = query,
                    paper_name   = paper_name,
                    failure_type = failure_type,
                    attempt      = attempt,
                )
                raw_answer      = retry_result["answer"]
                reasoning_chain = retry_result.get("reasoning_chain", "")
                chunks          = retry_result["chunks"]
                query_used      = retry_result["query_used"]
                sources         = []
                plan            = retry_result.get("plan", {})

            # ── Step 3: Evidence Grading ──────────────────────────────────────────
            # Always run grading so the frontend can render per-sentence
            # evidence indicators regardless of faithfulness level.
            # Only re-evaluate when grading actually removes sentences.
            _pre_eval      = evaluate_answer(query, raw_answer, chunks)
            grading_result = grade_answer(raw_answer, chunks)
            answer         = grading_result["cleaned_answer"]

            if _pre_eval["faithfulness"] > 0.75 or grading_result["removed_count"] == 0:
                # Faithfulness already strong, or answer unchanged — reuse pre-eval
                eval_scores = _pre_eval
            else:
                # Sentences were removed — re-evaluate the cleaned text
                eval_scores = evaluate_answer(query, answer, chunks)

            # Use the cleaned answer everywhere downstream
            answer = grading_result["cleaned_answer"]

            if grading_result["removed_count"] > 0:
                print(
                    f"[pipeline] Attempt {attempt}: evidence grader removed "
                    f"{grading_result['removed_count']} unsupported sentence(s)."
                )
            confidence  = compute_confidence(
                eval_scores["faithfulness"],
                eval_scores["answer_relevancy"],
            )

            print(
                f"[pipeline] Attempt {attempt}: confidence={confidence:.1f}  "
                f"faith={eval_scores['faithfulness']:.3f}  "
                f"relev={eval_scores['answer_relevancy']:.3f}"
            )

            # ── Out-of-domain detection (attempt 1 only) ──────────────────
            if attempt == 1 and eval_scores["answer_relevancy"] < _OUT_OF_DOMAIN_RELEVANCY:
                out_of_domain = True
                print(
                    f"[pipeline] Out-of-domain detected "
                    f"(relevancy={eval_scores['answer_relevancy']:.3f} "
                    f"< {_OUT_OF_DOMAIN_RELEVANCY}). Skipping retries."
                )

            # ── Deduplicate sources by (section, page) ────────────────────
            seen = set()
            deduped_sources = []
            for s in sources:
                key = (s["section"], s["page"])
                if key not in seen:
                    seen.add(key)
                    deduped_sources.append(s)

            # ── Track best result ─────────────────────────────────────────
            if confidence > best_confidence:
                best_confidence = confidence
                best_result = {
                    "query":            query,
                    "answer":           answer,
                    "reasoning_chain":  reasoning_chain,
                    "confidence":       confidence,
                    "faithfulness":     eval_scores["faithfulness"],
                    "answer_relevancy": eval_scores["answer_relevancy"],
                    "attempts":         attempt,
                    "passed":           confidence >= CONFIDENCE_THRESHOLD,
                    "warning":          None,
                    "failure_type":     None,
                    "sources":          deduped_sources,
                    "query_used":       query_used,
                    "plan":             plan,
                    # Grading metadata — useful for the frontend to render
                    # per-sentence confidence indicators
                    "grading": {
                        "grades":         grading_result["grades"],
                        "removed_count":  grading_result["removed_count"],
                        "grading_failed": grading_result["grading_failed"],
                    },
                }

            # ── Early exit on pass ────────────────────────────────────────
            if confidence >= CONFIDENCE_THRESHOLD and not out_of_domain:
                return best_result

            # ── Diagnose for next retry ───────────────────────────────────
            if attempt < MAX_ATTEMPTS and not out_of_domain:
                failure_type = diagnose_failure(query, answer, chunks, eval_scores)

        except Exception as e:
            print(f"[pipeline] Attempt {attempt} error: {e}")
            continue

    # ── Graceful degradation ──────────────────────────────────────────────
    if best_result is None:
        best_result = {
            "query":            query,
            "answer":           "Unable to answer this question from the provided paper.",
            "reasoning_chain":  "",
            "confidence":       0.0,
            "faithfulness":     0.0,
            "answer_relevancy": 0.0,
            "attempts":         MAX_ATTEMPTS,
            "passed":           False,
            "warning":          "All pipeline attempts failed with errors.",
            "failure_type":     "retrieval",
            "sources":          [],
            "query_used":       query,
            "plan":             {},
            "grading":          {"grades": [], "removed_count": 0, "grading_failed": True},
        }
    else:
        warning = (
            "This question does not appear to be related to the paper's content. "
            "Please ask a question about the paper's topics."
            if out_of_domain else
            f"Low confidence answer ({best_confidence:.1f}/100 below threshold "
            f"{CONFIDENCE_THRESHOLD}). The paper may not contain sufficient "
            f"information to answer this question."
        )
        best_result["passed"]       = False
        best_result["warning"]      = warning
        best_result["failure_type"] = "out_of_domain" if out_of_domain else failure_type

    return best_result