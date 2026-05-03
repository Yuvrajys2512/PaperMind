import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INTENT_INSTRUCTIONS = {
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
    "explanation": (
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

SYSTEM_PROMPT = """You are PaperMind, a precise research paper Q&A assistant.

## ANSWER FORMAT — ALWAYS FOLLOW THIS STRUCTURE:

**ESSENCE:** Write 2-3 sentences capturing the single most important insight.
This must be sharp, direct, and standalone — someone should understand the core answer from this alone.

**DETAIL:** Expand using ONLY what the context chunks explicitly state.
Do not infer, connect, or editorialize beyond the source text.
Every sentence in DETAIL must be traceable to a specific chunk.
Keep to 1-2 paragraphs maximum.

## YOUR RULES:

1. Answer ONLY using the provided context chunks. Never use outside knowledge.

2. CITATIONS — Quality over quantity:
   - Use a MAXIMUM of 3 citations per answer.
   - Cite only the most directly relevant sources.
   - Format: [Section: <section_name>, Page: <page_num>]
   - Place citations at the END of the specific sentence they support.
   - Never stack multiple citations on one sentence.

3. UNCERTAINTY — Be specific, never vague:
   If the context does not fully support the answer, say exactly which applies:
   - "This is not explicitly stated in the paper."
   - "The paper does not contain this specific detail — the closest relevant section is [X]."
   - "This can be inferred from [Section], but is not directly stated."
   - "This question asks about something outside the scope of this paper."
   Never say a generic "I don't know" or "unable to answer."

4. Never fabricate facts, numbers, or claims not present in the context.

5. Be precise. Research answers must be verifiable."""


def build_context_block(chunks: list) -> str:
    lines = []
    for i, chunk in enumerate(chunks):
        section = chunk["metadata"]["section"]
        page    = chunk["metadata"]["page_num"]
        text    = chunk["text"]
        lines.append(f"[Chunk {i+1} | Section: {section} | Page: {page}]\n{text}")
    return "\n\n---\n\n".join(lines)


def generate_answer(query: str, chunks: list, intents: list) -> dict:
    """
    Takes query, retrieved chunks, and detected intents.
    Assembles intent-aware prompt.
    Calls LLM.
    Returns structured response dict.
    """
    intent_instruction = " ".join(
        INTENT_INSTRUCTIONS.get(i, "") for i in intents
    ).strip()

    context = build_context_block(chunks)

    user_prompt = f"""Intent: {', '.join(intents)}
Instruction: {intent_instruction}

Context:
{context}

Question: {query}

Respond using the ESSENCE + DETAIL format. Maximum 3 citations total."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=1024,
        temperature=0.1
    )

    answer = response.choices[0].message.content.strip()

    sources = [
        {
            "section":     c["metadata"]["section"],
            "page":        c["metadata"]["page_num"],
            "chunk_index": c["metadata"].get("chunk_index", 0)
        }
        for c in chunks
    ]

    return {
        "query":       query,
        "answer":      answer,
        "intents":     intents,
        "sources":     sources,
        "model":       "llama-3.3-70b-versatile",
        "chunk_count": len(chunks)
    }
    