"""
ingestion/retry_engine.py — Phase 3, Steps 3 & 4

Public API
----------
diagnose_failure(query, answer, chunks, eval_scores) -> "retrieval" | "generation"
retry_query(query, paper_name, failure_type, attempt)  -> dict with keys:
    answer, chunks, eval_scores, confidence, query_used, attempt
"""

from __future__ import annotations
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Diagnosis thresholds ──────────────────────────────────────────────────────
_RERANK_RELEVANCE_THRESHOLD       = 0.0
_FAITHFULNESS_LOW                 = 0.40
_FAITHFULNESS_GENERATION_THRESHOLD = 0.80
_RELEVANCY_LOW                    = 0.40

# ── Retry config ──────────────────────────────────────────────────────────────
MAX_ATTEMPTS = 3

# Retrieval retry: expand these k values on each attempt
_RETRIEVAL_K_EXPANSIONS = [None, 20, 30]   # None = use route_query default
_LLM_K_EXPANSIONS       = [None,  8, 12]

# Generation retry: tighten these on each attempt
_GEN_TEMPERATURES  = [0.1, 0.05, 0.0]
_GEN_LLM_K_VALUES  = [None, 4, 3]          # None = use route_query default


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
    Diagnose why an answer failed to meet the confidence threshold.
    Returns "retrieval" or "generation".
    """
    faithfulness     = eval_scores.get("faithfulness", 0.0)
    answer_relevancy = eval_scores.get("answer_relevancy", 0.0)

    rerank_scores = _extract_rerank_scores(chunks)
    if rerank_scores:
        top_score    = max(rerank_scores)
        retrieval_ok = top_score >= _RERANK_RELEVANCE_THRESHOLD
    else:
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
# Step 4 — Query expansion (LLM rewrites query to paper vocabulary)
# ─────────────────────────────────────────────────────────────────────────────

def _expand_query(query: str) -> str:
    """
    Use LLM to rewrite the query using vocabulary closer to academic paper language.
    This helps when the user's query uses everyday words that don't match the paper's
    technical terminology — the vocabulary mismatch problem.

    Example:
      "How is relevance computed between tokens?"
      → "How does the attention mechanism compute compatibility scores between
         query and key vectors using scaled dot-product attention?"
    """
    prompt = (
        "You are helping a research paper question-answering system improve retrieval.\n"
        "Rewrite the following question using precise academic and technical vocabulary "
        "that would appear in the paper 'Attention Is All You Need' by Vaswani et al.\n"
        "Keep the same meaning. Return ONLY the rewritten question, nothing else.\n\n"
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
# Step 4 — Per-attempt retry logic
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_attempt(
    query: str,
    paper_name: str,
    retrieval_k_override: int | None = None,
    llm_k_override: int | None = None,
    temperature_override: float | None = None,
) -> dict:
    """
    Run one full pipeline attempt (retrieve → rerank → generate).
    Accepts optional overrides for retrieval_k, llm_k, and temperature.
    Falls back to route_query defaults when overrides are None.
    """
    from ingestion.query_router import route_query
    from ingestion.retriever    import retrieve
    from ingestion.reranker     import rerank
    from ingestion.generator    import generate_answer

    # ── Route (intent detection + default config) ─────────────────────────────
    routed  = route_query(query, paper_name)
    intents = routed["intents"]

    # ── Retrieval with optional k override ────────────────────────────────────
    ret_k = retrieval_k_override or routed.get("retrieval_k", 15)
    lm_k  = llm_k_override       or routed.get("llm_k", 5)

    try:
        raw_chunks = retrieve(query, paper_name, top_k=ret_k)
    except TypeError:
        # Fallback: some retrieve() signatures don't take top_k as kwarg
        raw_chunks = routed["chunks"]

    # ── Rerank ────────────────────────────────────────────────────────────────
    try:
        chunks = rerank(query, raw_chunks, top_k=lm_k)
    except (TypeError, ImportError):
        chunks = raw_chunks[:lm_k]

    # ── Generate with optional temperature override ───────────────────────────
    try:
        generated = generate_answer(query, chunks, intents,
                                    temperature=temperature_override)
    except TypeError:
        # generator.py doesn't accept temperature kwarg — use default
        generated = generate_answer(query, chunks, intents)

    return {
        "answer":     generated["answer"],
        "chunks":     chunks,
        "query_used": query,
    }


def retry_query(
    query: str,
    paper_name: str,
    failure_type: str,
    attempt: int,
) -> dict:
    """
    Execute one retry attempt with a strategy appropriate to the failure type
    and attempt number. Each attempt MUST change something from the previous one.

    Attempt numbering
    -----------------
    attempt=2 → first retry  (something changes from attempt 1)
    attempt=3 → second retry (different change from attempt 2)

    Retrieval failure strategy
    --------------------------
    attempt 2: expand query to paper vocabulary + increase retrieval_k to 20
    attempt 3: use expanded query + increase retrieval_k further to 30 + increase llm_k

    Generation failure strategy
    ---------------------------
    attempt 2: same query + reduce llm_k to 4 + lower temperature to 0.05
    attempt 3: expand query + reduce llm_k to 3 + temperature to 0.0 (deterministic)

    Parameters
    ----------
    query        : original user query
    paper_name   : paper identifier
    failure_type : "retrieval" or "generation"
    attempt      : 2 or 3 (attempt 1 is the original, handled by pipeline.py)

    Returns
    -------
    dict with: answer, chunks, query_used, attempt, failure_type
    """
    if attempt not in (2, 3):
        raise ValueError(f"retry_query: attempt must be 2 or 3, got {attempt}")

    print(f"\n[retry] Attempt {attempt} | failure_type={failure_type}")

    if failure_type == "retrieval":
        # ── Retrieval retry: change the query and expand k ────────────────────
        if attempt == 2:
            expanded_query = _expand_query(query)
            result = _run_pipeline_attempt(
                query              = expanded_query,
                paper_name         = paper_name,
                retrieval_k_override = 20,
                llm_k_override     = 8,
            )
        else:  # attempt == 3
            # Use a different expansion + even larger k
            expanded_query = _expand_query(query)
            result = _run_pipeline_attempt(
                query              = expanded_query,
                paper_name         = paper_name,
                retrieval_k_override = 30,
                llm_k_override     = 12,
            )

    else:  # generation failure
        # ── Generation retry: same query, tighter generation ──────────────────
        if attempt == 2:
            result = _run_pipeline_attempt(
                query                = query,
                paper_name           = paper_name,
                llm_k_override       = 4,
                temperature_override = 0.05,
            )
        else:  # attempt == 3
            # Also expand query in case retrieval was partially at fault
            expanded_query = _expand_query(query)
            result = _run_pipeline_attempt(
                query                = expanded_query,
                paper_name           = paper_name,
                llm_k_override       = 3,
                temperature_override = 0.0,
            )

    result["attempt"]      = attempt
    result["failure_type"] = failure_type
    return result