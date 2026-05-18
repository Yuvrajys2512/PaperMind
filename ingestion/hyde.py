"""
ingestion/hyde.py

Hypothetical Document Embeddings — pre-retrieval query expansion.

Idea
----
Users phrase questions in everyday language ("how was it trained?"), but
the relevant passages in a research paper use precise technical
vocabulary ("trained on 8 NVIDIA P100 GPUs over 12 hours"). The semantic
distance between the two phrasings can be wider than the distance
between two unrelated passages, which hurts top-k recall.

HyDE solves this by asking an LLM to *write* a plausible answer first,
then embedding *that* — a passage-shaped text in the paper's vocabulary
— and using it as the dense-retrieval query. BM25 still uses the
original wording, so lexical signal is preserved.

Cost: one extra LLM call per query (small, fast model). Latency is
dominated by the generation call already in the pipeline, so the added
hop is in the 300-600 ms range.

Public API
----------
generate_hypothetical(query, plan, paper_name=None) -> str
    Returns a 3-4 sentence pseudo-passage. On any LLM error returns the
    original query so retrieval still runs (HyDE is best-effort).
"""

from __future__ import annotations

from ingestion.llm_client import chat_completion

_HYDE_SYSTEM_PROMPT = """You write short, plausible passages that look like they were lifted from a research paper.

Given a question and a few key technical concepts, write a 3-4 sentence passage that:
- uses precise academic/technical vocabulary (no everyday paraphrase),
- reads like prose from a methods, results, or discussion section,
- commits to specific-sounding details (numbers, mechanism names, section terminology)
  even if invented — this is a retrieval seed, not an answer.

Do NOT write a meta-comment. Do NOT say "Here is...". Just output the passage."""


def generate_hypothetical(query: str, plan: dict | None = None, paper_name: str | None = None) -> str:
    """
    Produce a passage-shaped pseudo-answer to use as the dense-retrieval query.

    Parameters
    ----------
    query      : User's original question.
    plan       : Optional Query Plan from query_planner. If provided, the
                 plan's key_concepts seed the prompt with technical anchors.
    paper_name : Optional paper title — used only as a soft contextual hint.

    Returns
    -------
    str: A 3-4 sentence pseudo-passage. Returns the original query on
         any failure so the retriever still has something to embed.
    """
    key_concepts = (plan or {}).get("key_concepts", []) or []
    concepts_line = (
        f"Key concepts to weave in: {', '.join(key_concepts)}"
        if key_concepts else
        "No concept hints — infer from the question."
    )
    paper_line = f"Paper context: {paper_name}" if paper_name else ""

    user_prompt = "\n".join(p for p in [
        paper_line,
        concepts_line,
        f"Question: {query}",
        "",
        "Passage:",
    ] if p)

    try:
        passage = chat_completion(
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=180,
            temperature=0.3,   # tiny bit of variety so we don't always
                               # generate the same hypothetical
        )
        passage = passage.strip()
        if not passage:
            return query
        print(f"[hyde] generated {len(passage)} chars")
        return passage

    except Exception as exc:
        # Never let HyDE failure block retrieval — fall back to the user's query.
        print(f"[hyde] generation failed ({exc}); falling back to original query")
        return query
