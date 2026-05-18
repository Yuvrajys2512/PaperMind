"""
ingestion/query_router.py

Upgrade 1 — Query Understanding Layer (modified)

Changes from previous version:
  - REMOVED: ingestion.intent_detector  (detect_intent)
  - REMOVED: separate multi_hop sub-question generation
  - ADDED:   ingestion.query_planner    (plan_query)

The router now calls plan_query() once and uses the resulting Query Plan to:
  1. Look up retrieval config (via answer_type)
  2. Decide retrieval strategy (via complexity)
  3. Pass the plan's sub_questions directly into multi_hop_retrieve
     so the retriever uses the plan's decomposition instead of re-deriving it

The returned dict now carries "plan" instead of "intents" so generator.py
can consume the full plan (including answer_structure).
"""

import os

from ingestion.query_planner    import plan_query
from ingestion.hybrid_retriever import hybrid_retrieve
from ingestion.reranker         import rerank
from ingestion.multi_hop        import multi_hop_retrieve
from ingestion.hyde             import generate_hypothetical

# When set, skip CrossEncoder reranking entirely and take the top llm_k
# chunks by raw retrieval rank. Used for Phase 6 "no-rerank" ablation.
_DISABLE_RERANK = os.getenv("PAPERMIND_DISABLE_RERANK", "").lower() in ("1", "true", "yes")

# When set, skip the HyDE pseudo-passage generation and use the raw user
# query for dense retrieval. Used for ablation / latency-sensitive runs.
_DISABLE_HYDE = os.getenv("PAPERMIND_DISABLE_HYDE", "").lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Retrieval config — keyed by answer_type from the Query Plan
# ---------------------------------------------------------------------------

ANSWER_TYPE_CONFIG: dict[str, dict] = {
    "factual":            {"retrieval_k": 10, "llm_k": 3},
    "summarization":      {"retrieval_k": 15, "llm_k": 7},
    "critique":           {"retrieval_k": 10, "llm_k": 5},
    "comparison":         {"retrieval_k": 12, "llm_k": 6},
    "mechanism":          {"retrieval_k": 10, "llm_k": 5},
    "causal_explanation": {"retrieval_k": 10, "llm_k": 5},
    "hypothetical":       {"retrieval_k": 12, "llm_k": 6},
    "analysis":           {"retrieval_k": 12, "llm_k": 7},
}

DEFAULT_CONFIG: dict = {"retrieval_k": 10, "llm_k": 5}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_query(query: str, paper_name: str, on_progress=None) -> dict:
    """
    Full routing pipeline:

    1. Call plan_query() — single LLM pass that replaces intent detection
       AND multi-hop decomposition.
    2. Resolve retrieval config from plan["answer_type"].
    3. Run retrieval:
         - multi_hop if plan["complexity"] == "multi_hop"
         - hybrid single-pass otherwise
       In the multi-hop branch the plan's sub_questions are passed directly
       to multi_hop_retrieve, so no second LLM decomposition call is needed.
    4. Rerank with CrossEncoder, keep top llm_k chunks.
    5. Return chunks + the full plan for generator.py.

    Parameters
    ----------
    query      : str   User's raw question.
    paper_name : str   Paper ID used to scope the vector-store lookup.

    Returns
    -------
    dict with keys:
        query   : str   Original question.
        plan    : dict  Full Query Plan (answer_type, key_concepts,
                        sub_questions, answer_structure, complexity).
        config  : dict  Resolved retrieval config (retrieval_k, llm_k).
        chunks  : list  Reranked chunks ready for the generator.
    """

    def _emit(stage: str, message: str, **kwargs):
        if on_progress:
            try:
                on_progress({"stage": stage, "message": message, **kwargs})
            except Exception:
                pass  # never let progress reporting break the pipeline

    # ── Step 1: unified planning ──────────────────────────────────────────
    _emit("planning", "Planning your question…")
    print(f"[router] Planning query: {query[:80]}...")
    plan = plan_query(query)

    # ── Step 2: resolve retrieval config from answer_type ─────────────────
    config = ANSWER_TYPE_CONFIG.get(plan["answer_type"], DEFAULT_CONFIG)
    print(
        f"[router] Config -> retrieval_k={config['retrieval_k']}, "
        f"llm_k={config['llm_k']}"
    )

    # ── Step 3: retrieval strategy driven by plan complexity ───────────────
    if plan["complexity"] == "multi_hop":
        _emit(
            "retrieving",
            f"Multi-hop search across {len(plan['sub_questions'])} angles…",
        )
        print(
            f"[router] Multi-hop retrieval | "
            f"sub_questions={plan['sub_questions']}"
        )
        # Pass the plan's sub_questions so multi_hop_retrieve uses them
        # directly instead of running its own decomposition LLM call.
        # HyDE is intentionally skipped here: sub-questions already target
        # specific aspects, and one HyDE call per sub-question would
        # blow the LLM budget.
        raw_chunks = multi_hop_retrieve(
            query         = query,
            paper_name    = paper_name,
            retrieval_k   = config["retrieval_k"],
            sub_questions = plan["sub_questions"],   # ← from plan
            boost_terms   = plan.get("key_concepts", []),
        )
    else:
        # HyDE: generate a passage-shaped pseudo-answer to use as the
        # dense-retrieval seed. BM25 still uses the original query.
        if _DISABLE_HYDE:
            hyde_text = None
        else:
            _emit("hyde", "Drafting a search seed…")
            hyde_text = generate_hypothetical(query, plan, paper_name)
            if hyde_text == query:
                hyde_text = None  # treat fallback as "no hyde"

        _emit("retrieving", "Searching the paper…")
        print("[router] Single-pass retrieval" + (" (with HyDE)" if hyde_text else ""))
        raw_chunks = hybrid_retrieve(
            query,
            paper_name,
            top_k=config["retrieval_k"],
            boost_terms=plan.get("key_concepts", []),
            hyde_text=hyde_text,
        )

    _emit("reviewing", f"Reviewing {len(raw_chunks)} relevant passages…")

    # ── Step 4: rerank ────────────────────────────────────────────────────
    if _DISABLE_RERANK:
        chunks = raw_chunks[: config["llm_k"]]
        print(
            f"[router] Reranker DISABLED via env -> taking top "
            f"{len(chunks)} of {len(raw_chunks)} chunks by retrieval rank"
        )
    else:
        chunks = rerank(query, raw_chunks, top_k=config["llm_k"])
        print(f"[router] {len(raw_chunks)} raw chunks -> {len(chunks)} after rerank")

    return {
        "query":  query,
        "plan":   plan,     # full plan — generator reads answer_structure from here
        "config": config,
        "chunks": chunks,
    }


# ---------------------------------------------------------------------------
# NOTE — multi_hop_retrieve compatibility
# ---------------------------------------------------------------------------
# The new router passes sub_questions=plan["sub_questions"] to
# multi_hop_retrieve.  If your current multi_hop.py signature is:
#
#   def multi_hop_retrieve(query, paper_name, retrieval_k):
#
# Add sub_questions as an optional kwarg:
#
#   def multi_hop_retrieve(query, paper_name, retrieval_k, sub_questions=None):
#       if sub_questions is None:
#           sub_questions = decompose_query(query)   # old behaviour as fallback
#       ...
#
# This keeps the function backward-compatible with any direct callers while
# letting the router supply pre-computed sub-questions from the plan.