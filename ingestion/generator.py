"""
ingestion/generator.py

Upgrade 1 — Query Understanding Layer (modified)

Changes from previous version:
  - generate_answer() now accepts plan: dict instead of intents: list
  - The plan's answer_structure is injected into the prompt as a
    numbered checklist the model must follow
  - The plan's answer_type drives the tone instruction (replaces INTENT_INSTRUCTIONS)
  - Return dict now includes the full plan instead of just intents list

Upgrade 2 — Chain of Thought Reasoning (applied here)
  - The prompt now enforces a 6-step reasoning scratchpad BEFORE the
    ESSENCE + DETAIL answer is written
  - The reasoning chain is extracted and returned as "reasoning_chain"
    in the response dict for debugging / audit purposes
"""

from ingestion.llm_client import chat_completion

# ---------------------------------------------------------------------------
# Tone instructions keyed by answer_type (replaces old INTENT_INSTRUCTIONS)
# ---------------------------------------------------------------------------

ANSWER_TYPE_INSTRUCTIONS: dict[str, str] = {
    "factual": (
        "Answer precisely and concisely. "
        "State the exact fact, number, or definition asked for. "
        "Do not elaborate beyond what was asked."
    ),
    "summarization": (
        "Provide a comprehensive overview covering all major points "
        "found across the provided context. "
        "Structure your answer with clear logical flow."
    ),
    "critique": (
        "Identify limitations, weaknesses, and problems explicitly. "
        "Distinguish between what the paper claims and what it empirically demonstrates. "
        "Be specific about what evidence supports each weakness."
    ),
    "comparison": (
        "Explicitly identify agreements, differences, and contrasts. "
        "Structure your answer to address both sides being compared. "
        "Be specific about what distinguishes each approach."
    ),
    "mechanism": (
        "Explain the internal workings step by step. "
        "Be precise about the sequence of operations and how components interact. "
        "Use technical detail from the context."
    ),
    "causal_explanation": (
        "Explain the reasoning and motivation behind the decision or phenomenon. "
        "Address the 'why' directly with supporting evidence from the context."
    ),
    "hypothetical": (
        "Reason carefully about the hypothetical scenario using evidence from the context. "
        "Clearly distinguish between what the paper states and what can be reasonably inferred. "
        "Acknowledge uncertainty where it exists."
    ),
    "analysis": (
        "Analyze trade-offs, implications, and performance characteristics thoroughly. "
        "Consider multiple angles and be specific about complexity or impact. "
        "Support every claim with evidence from the context."
    ),
}

DEFAULT_INSTRUCTION = (
    "Answer accurately using only the provided context. "
    "Support every claim with evidence from the chunks."
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are PaperMind, a precise research paper Q&A assistant.

## YOUR TASK

Work through a structured reasoning process, then write a final answer.

---

## STEP 1 — REASONING SCRATCHPAD

Before writing your answer, complete all six reasoning steps below.
Label each step exactly as shown.

[INVENTORY]
List what each numbered chunk explicitly states — no inference yet.
One bullet per distinct fact.

[GAPS]
List what the question asks for that NO chunk directly addresses.

[INFERENCE]
List what can be reasonably inferred from the stated facts.
Distinguish from direct statements.

[UNCERTAINTY]
Flag anything that must be labelled as inferred (not stated) in the answer.

[STRUCTURE]
Map the answer_structure steps to your evidence:
For each step in the answer_structure, note which chunk(s) support it.

[WRITE]
Now write the final answer using the ESSENCE + DETAIL format below.

---

## FINAL ANSWER FORMAT

**ESSENCE:** 2-3 sentences capturing the single most important insight.
Sharp, direct, standalone — someone should grasp the core answer from this alone.

**DETAIL:** Expand using ONLY what the context chunks explicitly state.
Follow the answer_structure steps in order.
Do not infer, connect, or editorialize beyond the source text.
Every sentence must be traceable to a specific chunk.
Maximum 2 paragraphs.

---

## RULES

1. Use ONLY the provided context chunks. Never use outside knowledge.

2. CITATIONS — Quality over quantity:
   - Maximum 3 citations per answer.
   - Cite only the most directly relevant sources.
   - Format: [Section: <section_name>, Page: <page_num>]
   - Place citations at the END of the sentence they support.
   - Never stack multiple citations on one sentence.

3. UNCERTAINTY — Be specific, never vague:
   - "This is not explicitly stated in the paper."
   - "The paper does not contain this specific detail — the closest relevant section is [X]."
   - "This can be inferred from [Section], but is not directly stated."
   - "This question asks about something outside the scope of this paper."
   Never say a generic "I don't know."

4. Never fabricate facts, numbers, or claims not present in the context.

5. Complete ALL six reasoning steps before writing the answer."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_context_block(chunks: list) -> str:
    lines = []
    for i, chunk in enumerate(chunks):
        section = chunk["metadata"]["section"]
        page    = chunk["metadata"]["page_num"]
        text    = chunk["text"]
        lines.append(f"[Chunk {i+1} | Section: {section} | Page: {page}]\n{text}")
    return "\n\n---\n\n".join(lines)


def _format_answer_structure(steps: list) -> str:
    """Converts answer_structure list into a numbered prompt block."""
    return "\n".join(f"  {i+1}. {step}" for i, step in enumerate(steps))


def _extract_reasoning_and_answer(full_response: str) -> tuple[str, str]:
    """
    Splits the model's output into reasoning_chain and answer.
    Handles both the canonical [WRITE] marker and the model's common
    variant "STEP 2 — WRITE" (with em/en/hyphen dash variants).
    """
    import re
    pattern = re.compile(r'\[WRITE\]|STEP\s+\d+\s*[—\-–]+\s*WRITE', re.IGNORECASE)
    m = pattern.search(full_response)
    if m:
        reasoning_chain = full_response[: m.end()].strip()
        answer          = full_response[m.end() :].strip()
        return reasoning_chain, answer
    return "", full_response.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(query: str, chunks: list, plan: dict) -> dict:
    """
    Generates an answer using the Query Plan to guide structure and tone.

    Parameters
    ----------
    query  : str   The user's original question.
    chunks : list  Reranked context chunks from the retriever.
    plan   : dict  Query Plan produced by query_planner.plan_query().
                   Must contain: answer_type, answer_structure.

    Returns
    -------
    dict with keys:
        query           : str
        answer          : str   Final ESSENCE + DETAIL answer.
        reasoning_chain : str   The 5-step scratchpad (for debugging/audit).
        plan            : dict  Full Query Plan.
        sources         : list
        model           : str
        chunk_count     : int
    """
    answer_type      = plan.get("answer_type", "factual")
    answer_structure = plan.get("answer_structure", [
        "Answer the question directly using the provided context."
    ])
    tone_instruction = ANSWER_TYPE_INSTRUCTIONS.get(answer_type, DEFAULT_INSTRUCTION)
    structure_block  = _format_answer_structure(answer_structure)

    context = build_context_block(chunks)

    user_prompt = f"""Answer Type: {answer_type}
Tone Instruction: {tone_instruction}

Answer Structure (follow these steps in the DETAIL section, in order):
{structure_block}

Context:
{context}

Question: {query}

Work through all six reasoning steps [INVENTORY] → [GAPS] → [INFERENCE] → \
[UNCERTAINTY] → [STRUCTURE] → [WRITE], then write the final answer."""

    full_output = chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    reasoning_chain, answer = _extract_reasoning_and_answer(full_output)

    sources = [
        {
            "section":     c["metadata"]["section"],
            "page":        c["metadata"]["page_num"],
            "chunk_index": c["metadata"].get("chunk_index", 0),
        }
        for c in chunks
    ]

    return {
        "query":           query,
        "answer":          answer,
        "reasoning_chain": reasoning_chain,
        "plan":            plan,
        "sources":         sources,
        "model":           "multi-provider",
        "chunk_count":     len(chunks),
    }