"""
ingestion/query_planner.py

Upgrade 1 — Query Understanding Layer

Replaces the separate intent_detector + multi_hop decomposer calls with a
single LLM planning pass that produces a structured Query Plan.  The plan is
consumed by:
  - query_router.py  → drives retrieval strategy and config
  - generator.py     → drives answer structure and tone
"""

import json
from ingestion.llm_client import chat_completion

# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

PLANNING_SYSTEM_PROMPT = """You are a query planning engine for a research paper Q&A system.

Given a user question about an academic paper, produce a structured JSON query plan
that will guide both retrieval and answer generation downstream.

Return ONLY valid JSON — no markdown fences, no preamble, no explanation.

Schema (all fields required):

{
  "answer_type": "<one of: factual | causal_explanation | mechanism | comparison | critique | summarization | analysis | hypothetical>",
  "key_concepts": ["<concept1>", "<concept2>"],
  "sub_questions": ["<sub_q1>", "<sub_q2>"],
  "answer_structure": [
    "<step 1: concrete thing the answer must cover>",
    "<step 2: next thing>",
    "..."
  ],
  "complexity": "<simple | multi_hop>"
}

Field rules:
- answer_type   : pick the single best-fitting type.
- key_concepts  : 2-5 core technical terms the retriever should locate in the paper.
- sub_questions : 1-2 for simple queries; 2-4 for multi_hop. Write them as
                  search queries the retriever will run, not rhetorical questions.
- answer_structure: 2-5 ordered steps describing what a complete answer must cover.
                  Be concrete — e.g. "State the RNN bottleneck", not "explain background".
- complexity    : "simple"    if one passage can answer the question.
                  "multi_hop" if the answer requires connecting evidence from
                  multiple sections or concepts.

Examples of good answer_structure steps:
  "State the core claim and the paper section where it appears"
  "Explain the mechanism using the paper's own terminology"
  "Cite the empirical result or ablation that supports the claim"
  "Acknowledge what the paper does NOT address"
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_query(query: str) -> dict:
    """
    Makes a single LLM call to produce a structured Query Plan.

    Parameters
    ----------
    query : str
        The user's raw question.

    Returns
    -------
    dict
        A Query Plan with keys:
          answer_type, key_concepts, sub_questions, answer_structure, complexity

    Notes
    -----
    On JSON parse failure a safe fallback plan is returned so the pipeline
    never hard-crashes at the planning stage.
    """
    raw = chat_completion(
        messages=[
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Question: {query}"}
        ],
        max_tokens=200,
        temperature=0.0,   # deterministic — planning should be stable
    )

    try:
        plan = json.loads(raw)
        _validate_plan(plan)   # fills missing keys with defaults
    except (json.JSONDecodeError, Exception) as exc:
        print(f"[query_planner] WARNING: plan parse failed ({exc}). "
              f"Raw output: {raw[:300]}")
        plan = _fallback_plan(query)

    print(
        f"[query_planner] type={plan['answer_type']} | "
        f"complexity={plan['complexity']} | "
        f"sub_qs={len(plan['sub_questions'])}"
    )
    return plan


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_VALID_ANSWER_TYPES = {
    "factual", "causal_explanation", "mechanism", "comparison",
    "critique", "summarization", "analysis", "hypothetical",
}

_VALID_COMPLEXITY = {"simple", "multi_hop"}


def _validate_plan(plan: dict) -> None:
    """
    Mutates the plan in-place to guarantee all required keys exist
    and have sensible types/values.  Does NOT raise — always recovers.
    """
    if plan.get("answer_type") not in _VALID_ANSWER_TYPES:
        plan["answer_type"] = "factual"

    if plan.get("complexity") not in _VALID_COMPLEXITY:
        plan["complexity"] = "simple"

    if not isinstance(plan.get("key_concepts"), list) or not plan["key_concepts"]:
        plan["key_concepts"] = []

    if not isinstance(plan.get("sub_questions"), list) or not plan["sub_questions"]:
        plan["sub_questions"] = [plan.get("_original_query", "")]

    if not isinstance(plan.get("answer_structure"), list) or not plan["answer_structure"]:
        plan["answer_structure"] = ["Answer the question directly using the provided context."]


def _fallback_plan(query: str) -> dict:
    """Returns a minimal safe plan when the LLM output cannot be parsed."""
    return {
        "answer_type":     "factual",
        "key_concepts":    [],
        "sub_questions":   [query],
        "answer_structure": [
            "Answer the question directly using the provided context.",
            "Cite the most relevant chunk.",
        ],
        "complexity": "simple",
    }