"""
ingestion/retry_engine.py — Phase 3, Steps 3 & 4

Public API
----------
diagnose_failure(query, answer, chunks, eval_scores) -> "retrieval" | "generation"
retry_query(query, paper_name, failure_type, attempt) -> dict
"""

from __future__ import annotations
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Diagnosis thresholds ──────────────────────────────────────────────────────
_RERANK_RELEVANCE_THRESHOLD        = 0.0
_FAITHFULNESS_LOW                  = 0.40
_FAITHFULNESS_GENERATION_THRESHOLD = 0.80
_RELEVANCY_LOW                     = 0.40

MAX_ATTEMPTS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Failure diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def _extract_rerank_scores(chunks: list) -> list[float]:
    return [float(c["score"]) for c in chunks
            if isinstance(c, dict) and "score" in c]


def diagnose_failure(
    query: str,
    answer: str,
    chunks: list,
    eval_scores: dict,
) -> str:
    """
    Diagnose why an answer failed. Returns "retrieval" or "generation".
    """
    faithfulness     = eval_scores.get("faithfulness", 0.0)
    answer_relevancy = eval_scores.get("answer_relevancy", 0.0)

    rerank_scores = _extract_rerank_scores(chunks)
    if rerank_scores:
        top_score    = max(rerank_scores)
        retrieval_ok = top_score >= _RERANK_RELEVANCE_THRESHOLD
    else:
        # No rerank scores in chunks — use faithfulness as proxy
        retrieval_ok = faithfulness >= 0.05

    if not retrieval_ok:
        failure_type = "retrieval"
    elif faithfulness < _FAITHFULNESS_GENERATION_THRESHOLD:
        failure_type = "generation"
    elif answer_relevancy < _RELEVANCY_LOW:
        failure_type = "generation"
    else:
        failure_type = "retrieval"

    top_str = f"{max(rerank_scores):.3f}" if rerank_scores else "N/A"
    print(f"[diagnose] faith={faithfulness:.3f}  relev={answer_relevancy:.3f}  "
          f"top_rerank={top_str}  retrieval_ok={retrieval_ok}  -> {failure_type} failure")

    return failure_type


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Query expansion
# ─────────────────────────────────────────────────────────────────────────────

def _expand_query(query: str) -> str:
    """
    Rewrite query using technical vocabulary from the paper.
    Helps when everyday language doesn't match the paper's terminology.
    """
    prompt = (
        "You are helping a research paper question-answering system improve retrieval.\n"
        "Rewrite the following question using precise academic and technical vocabulary "
        "that would appear in the paper 'Attention Is All You Need' by Vaswani et al.\n"
        "Focus on the specific technical terms, algorithm names, and section topics "
        "used in the paper. Return ONLY the rewritten question, nothing else.\n\n"
        f"Original question: {query}\n"
        "Rewritten question:"
    )
    try:
        client   = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.3,
            max_tokens  = 80,
        )
        expanded = response.choices[0].message.content.strip()
        print(f"[retry] Query expanded:\n  Original: {query}\n  Expanded: {expanded}")
        return expanded
    except Exception as e:
        print(f"[retry] Query expansion failed ({e}) — using original query.")
        return query


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Core pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_attempt(
    query: str,
    paper_name: str,
    llm_k_slice: int | None = None,
) -> dict:
    """
    Run one full pipeline attempt via route_query (which handles
    routing + retrieval + reranking internally).

    Parameters
    ----------
    query       : query to use (may be expanded)
    paper_name  : paper identifier
    llm_k_slice : if set, take only the top-N chunks before generation
                  (simulates reducing llm_k for generation retries)

    Returns
    -------
    dict with: answer, chunks, query_used
    """
    from ingestion.query_router import route_query
    from ingestion.generator    import generate_answer

    routed  = route_query(query, paper_name)
    chunks  = routed["chunks"]
    intents = routed["intents"]

    # Slice chunks if we want to tighten the generation context
    if llm_k_slice is not None:
        chunks = chunks[:llm_k_slice]

    generated = generate_answer(query, chunks, intents)

    return {
        "answer":     generated["answer"],
        "chunks":     chunks,
        "query_used": query,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Retry strategies
# ─────────────────────────────────────────────────────────────────────────────

def retry_query(
    query: str,
    paper_name: str,
    failure_type: str,
    attempt: int,
) -> dict:
    """
    Execute one retry attempt. Each attempt MUST change something from the previous.

    Retrieval failure strategy
    --------------------------
    attempt 2: LLM expands query to paper vocabulary → new retrieval run
    attempt 3: different LLM expansion + keep top 3 chunks only (tighten focus)

    Generation failure strategy
    ---------------------------
    attempt 2: same query, slice to top 4 chunks (force focus on best chunks)
    attempt 3: expand query + slice to top 3 chunks

    Parameters
    ----------
    query        : original user query
    paper_name   : paper identifier
    failure_type : "retrieval" or "generation"
    attempt      : 2 or 3

    Returns
    -------
    dict: answer, chunks, query_used, attempt, failure_type
    """
    if attempt not in (2, 3):
        raise ValueError(f"attempt must be 2 or 3, got {attempt}")

    print(f"\n[retry] Attempt {attempt} | failure_type={failure_type}")

    if failure_type == "retrieval":
        # Both attempts expand the query — this is the core fix for vocab mismatch.
        # The expansion call gives a different result each time (temperature=0.3).
        expanded = _expand_query(query)

        if attempt == 2:
            result = _run_attempt(expanded, paper_name, llm_k_slice=None)
        else:
            # attempt 3: use expanded query but tighten to top 3 chunks
            # so the generator has less noise to sift through
            result = _run_attempt(expanded, paper_name, llm_k_slice=3)

    else:  # generation failure
        if attempt == 2:
            # Same query, but reduce context to top 4 chunks
            # Forces generator to use only the most relevant retrieved content
            result = _run_attempt(query, paper_name, llm_k_slice=4)
        else:
            # attempt 3: expand query AND tighten context
            expanded = _expand_query(query)
            result   = _run_attempt(expanded, paper_name, llm_k_slice=3)

    result["attempt"]      = attempt
    result["failure_type"] = failure_type
    return result