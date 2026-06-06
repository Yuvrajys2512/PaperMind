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
Follow the answer_structure steps in order as a silent writing guide — do NOT print them as headers or label them (no "Step 1:", no bold titles).
Write flowing prose that covers each step invisibly.
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
   - NEVER reference chunks by number in any form. Do not write "Chunk 1", "Chunk 5", "as described in Chunk 1", "according to Chunk 3", "in Chunk 6", or any variation. Always use [Section: <section_name>, Page: <page_num>].

3. UNCERTAINTY — Be specific, never vague:
   - "This is not explicitly stated in the paper."
   - "The paper does not contain this specific detail — the closest relevant section is [X]."
   - "This can be inferred from [Section], but is not directly stated."
   - "This question asks about something outside the scope of this paper."
   Never say a generic "I don't know."

4. Never fabricate facts, numbers, or claims not present in the context.

5. ANSWER STRUCTURE STEPS are a private writing guide only.
   Never print them as headers, numbered labels, or bold titles.
   Correct: flowing prose that naturally covers each point.
   Wrong: "Step 1: ..." / "Paper A:" / "Point 1:" as a visible header.

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


def _strip_chunk_refs(text: str) -> str:
    """Remove raw chunk-number references the model inserts despite instructions."""
    import re
    # "Chunk 1", "(Chunk 1)", "as described in Chunk 1,", "in Chunk 6", etc.
    text = re.sub(r'\(\s*Chunk\s+\d+\s*\)', '', text, flags=re.IGNORECASE)
    text = re.sub(
        r'\b(as described in|according to|per|in|from|see)\s+Chunk\s+\d+\b[,]?',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(r'\bChunk\s+\d+\b', '', text, flags=re.IGNORECASE)
    # Clean up spacing artifacts
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\s+([,\.])', r'\1', text)
    return text.strip()


# Verbatim descriptive text from SYSTEM_PROMPT (FINAL ANSWER FORMAT + RULES)
# that weak models (e.g. Llama-3.1-8B) echo into the answer body instead of
# following. Stored normalized (lowercase, single spaces, plain dashes) and
# matched as substrings so an echoed line is dropped wherever it appears —
# even when it trails an ESSENCE/DETAIL marker the anchor would otherwise keep.
_SCAFFOLDING_SENTENCES = (
    "2-3 sentences capturing the single most important insight",
    "someone should grasp the core answer from this alone",
    "expand using only what the context chunks explicitly state",
    "follow the answer_structure steps in order as a silent writing guide",
    "write flowing prose that covers each step invisibly",
    "write flowing prose that covers each point invisibly",
    "do not infer, connect, or editorialize beyond the source text",
    "every sentence must be traceable to a specific chunk",
    "maximum 2 paragraphs",
    "use only the provided context chunks",
    "never use outside knowledge",
    "do not print them as headers or label them",
)


def _normalize_scaffolding(s: str) -> str:
    """Lowercase, unify dashes/quotes, collapse whitespace — so prompt-template
    echoes match regardless of the model's punctuation/spacing variants."""
    import re
    s = s.lower().replace("—", "-").replace("–", "-")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


def _drop_scaffolding_sentences(answer: str) -> str:
    """Remove prompt-template sentences echoed into the answer body.

    Splits each line on the same boundaries as the grader's sentence splitter,
    drops any unit whose normalized text contains a known scaffolding phrase,
    and rejoins. A line left empty after scrubbing (e.g. a bare '**ESSENCE:**'
    whose only content was template text) is dropped entirely.
    """
    import re
    out_lines: list[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        kept: list[str] = []
        parts = stripped.split(". ")
        for i, part in enumerate(parts):
            p = part.strip()
            if not p:
                continue
            unit = p + "." if i < len(parts) - 1 else p
            norm = _normalize_scaffolding(unit)
            if any(phrase in norm for phrase in _SCAFFOLDING_SENTENCES):
                continue
            kept.append(unit)
        rejoined = " ".join(kept).strip()
        if rejoined:
            out_lines.append(rejoined)
    result = "\n".join(out_lines)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


def _strip_scaffolding(answer: str) -> str:
    """Remove leaked prompt scaffolding that models echo into the answer.

    Two failure modes:
      1. A preamble before the real answer — '## FINAL ANSWER FORMAT' headers,
         'Now writing the final answer.' filler. Strong models (Llama-3.3-70B)
         do this; we anchor at the ESSENCE marker or peel known top lines.
      2. The template DESCRIPTIONS echoed verbatim as the answer body — weak
         models (Llama-3.1-8B) copy 'Maximum 2 paragraphs.', 'Every sentence
         must be traceable…' etc. instead of filling the template in. Anchoring
         on ESSENCE alone KEEPS these (they follow the marker), so we then drop
         them by phrase in _drop_scaffolding_sentences.
    """
    import re
    m = re.search(r'\*{0,2}ESSENCE', answer)
    if m:
        answer = answer[m.start():].strip()
    else:
        lines = answer.splitlines()
        while lines:
            head = lines[0].strip().lower()
            if (not head
                    or "final answer format" in head
                    or head.startswith(("now writing", "now i will write",
                                        "here is the final answer",
                                        "here is my final answer"))):
                lines.pop(0)
                continue
            break
        answer = "\n".join(lines).strip()

    return _drop_scaffolding_sentences(answer)


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
        return reasoning_chain, _strip_scaffolding(answer)
    return "", _strip_scaffolding(full_response.strip())


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

    # Experiments can pin the generation model (only) via PAPERMIND_GEN_MODEL,
    # e.g. "llama-3.1-8b-instant" (weak) vs "llama-3.3-70b-versatile" (strong),
    # to study how generator strength affects the evidence grader. Grading and
    # judging stay on the normal provider chain.
    import os
    # Pin generation to Groq-2 (separate TPM quota from Groq-1) so generation
    # calls don't compete with planning/HyDE/judging that run on the main chain.
    # Falls back to Groq-1 if Groq-2 is not configured.
    _gen_model = os.getenv("PAPERMIND_GEN_MODEL")
    if _gen_model:
        from ingestion.llm_client import _PROVIDERS as _all_providers
        _groq2 = next((p["name"] for p in _all_providers if p["name"] == "Groq-2"), None)
        _pin = (_groq2 if _groq2 else "Groq-1", _gen_model)
    else:
        _pin = None

    full_output = chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=2048,
        temperature=0.1,
        pin=_pin,
    )

    reasoning_chain, answer = _extract_reasoning_and_answer(full_output)
    answer = _strip_chunk_refs(answer)

    sources = [
        {
            "section":      c["metadata"]["section"],
            "section_type": c["metadata"].get("section_type", "text"),
            "page":         c["metadata"]["page_num"],
            "chunk_index":  c["metadata"].get("chunk_index", 0),
            "text":         (c.get("text") or "")[:600],
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