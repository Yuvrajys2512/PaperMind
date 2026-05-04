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

from ingestion.query_planner    import plan_query
from ingestion.hybrid_retriever import hybrid_retrieve
from ingestion.reranker         import rerank
from ingestion.multi_hop        import multi_hop_retrieve


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

def route_query(query: str, paper_name: str) -> dict:
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

    # ── Step 1: unified planning ──────────────────────────────────────────
    print(f"[router] Planning query: {query[:80]}...")
    plan = plan_query(query)

    # ── Step 2: resolve retrieval config from answer_type ─────────────────
    config = ANSWER_TYPE_CONFIG.get(plan["answer_type"], DEFAULT_CONFIG)
    print(
        f"[router] Config → retrieval_k={config['retrieval_k']}, "
        f"llm_k={config['llm_k']}"
    )

    # ── Step 3: retrieval strategy driven by plan complexity ───────────────
    if plan["complexity"] == "multi_hop":
        print(
            f"[router] Multi-hop retrieval | "
            f"sub_questions={plan['sub_questions']}"
        )
        # Pass the plan's sub_questions so multi_hop_retrieve uses them
        # directly instead of running its own decomposition LLM call.
        # NOTE: multi_hop_retrieve must accept a sub_questions kwarg.
        #       If your current version does not, see the note below.
        raw_chunks = multi_hop_retrieve(
            query         = query,
            paper_name    = paper_name,
            retrieval_k   = config["retrieval_k"],
            sub_questions = plan["sub_questions"],   # ← from plan
        )
    else:
        print("[router] Single-pass retrieval")
        raw_chunks = hybrid_retrieve(
            query,
            paper_name,
            top_k=config["retrieval_k"],
        )

    # ── Step 4: rerank ────────────────────────────────────────────────────
    chunks = rerank(query, raw_chunks, top_k=config["llm_k"])
    print(f"[router] {len(raw_chunks)} raw chunks → {len(chunks)} after rerank")

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