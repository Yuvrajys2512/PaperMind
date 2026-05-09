"""
ingestion/evidence_grader.py

Upgrade 3 — Evidence Grading

Post-generation verification pass.  After the answer is written, each sentence
is classified against the retrieved chunks:

    DIRECT      — explicitly stated in a chunk (near-verbatim support exists)
    INFERRED    — logically follows from chunk content, but not directly stated
    UNSUPPORTED — no chunk supports this; model used outside knowledge or hallucinated

UNSUPPORTED sentences are stripped from the answer before it reaches the user
or the evaluator.  This is what actually fixes faithfulness — not better metrics,
but removing bad content before it ships.

Public API
----------
grade_answer(answer, chunks) -> dict
    {
        "cleaned_answer"  : str,          # answer with UNSUPPORTED sentences removed
        "original_answer" : str,          # unmodified answer
        "grades"          : list[dict],   # per-sentence classification
        "removed_count"   : int,
        "grading_failed"  : bool,         # True if LLM call failed; original returned
    }
"""

from __future__ import annotations

import json
from ingestion.llm_client import chat_completion

# ---------------------------------------------------------------------------
# Grader prompt
# ---------------------------------------------------------------------------

GRADER_SYSTEM_PROMPT = """You are a fact-checker for a research paper Q&A system.

You will be given:
  1. A set of numbered source chunks retrieved from a paper.
  2. An answer generated from those chunks.

Your job: classify each sentence in the answer into one of three grades.

GRADES:
  DIRECT      — The chunk contains this information explicitly.
                Word-for-word or near-verbatim support exists.
  INFERRED    — The chunk does not state this directly, but it logically
                follows from what is stated. The inference is reasonable.
  UNSUPPORTED — No chunk supports this sentence. The model generated it
                from general knowledge or hallucinated it.

RULES:
  - Classify EVERY sentence in the answer — do not skip any.
  - For preamble lines like "**ESSENCE:**" or "**DETAIL:**" (section headers),
    set grade to "DIRECT" and chunk_ref to "header" so they are preserved.
  - Be strict: if a sentence makes a specific factual claim (number, name,
    date, result) that does not appear in any chunk, mark it UNSUPPORTED.
  - Hedged inferences ("This suggests...", "This implies...") are INFERRED
    only if the underlying fact IS in the chunks.

Return ONLY a JSON array — no markdown, no preamble:

[
  {
    "sentence"  : "<exact sentence text>",
    "grade"     : "DIRECT | INFERRED | UNSUPPORTED",
    "chunk_ref" : "Chunk N | none | header"
  },
  ...
]
"""

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Splits answer text into gradeable units.
    Preserves section headers (**ESSENCE:**, **DETAIL:**) as their own units.
    Skips blank lines.
    """
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("**") and line.endswith("**"):
            sentences.append(line)
            continue
        parts = line.split(". ")
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if i < len(parts) - 1:
                part = part + "."
            if len(part) > 15:
                sentences.append(part)
    return sentences


# ---------------------------------------------------------------------------
# Context block builder
# ---------------------------------------------------------------------------

def _build_grader_context(chunks: list) -> str:
    lines = []
    for i, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            text    = chunk.get("text", "")
            section = chunk.get("metadata", {}).get("section", "?")
            page    = chunk.get("metadata", {}).get("page_num", "?")
            lines.append(f"[Chunk {i+1} | Section: {section} | Page: {page}]\n{text}")
        else:
            lines.append(f"[Chunk {i+1}]\n{str(chunk)}")
    return "\n\n---\n\n".join(lines)


# ---------------------------------------------------------------------------
# LLM grading call
# ---------------------------------------------------------------------------

def _call_grader(sentences: list[str], chunks: list) -> list[dict] | None:
    """
    Calls the LLM to classify each sentence.
    Returns a list of grade dicts, or None on failure.
    """
    context  = _build_grader_context(chunks)
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))

    user_prompt = f"""Source Chunks:
{context}

Answer Sentences to Grade:
{numbered}

Classify every sentence. Return only the JSON array."""

    try:
        raw = chat_completion(
            messages=[
                {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        grades = json.loads(raw)
        return grades if isinstance(grades, list) else None

    except Exception as exc:
        print(f"[evidence_grader] Grading LLM call failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Answer reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_answer(original: str, sentences: list[str], grades: list[dict]) -> tuple[str, list[dict]]:
    """
    Removes UNSUPPORTED sentences from the original answer text directly,
    preserving all newlines, section headers, and formatting.
    """
    import re as _re

    grade_map: dict[str, str] = {}
    for g in grades:
        key = g.get("sentence", "")[:60].strip()
        grade_map[key] = g.get("grade", "INFERRED")

    enriched: list[dict] = []
    to_remove: list[str] = []

    for sent in sentences:
        key   = sent[:60].strip()
        grade = grade_map.get(key, "INFERRED")

        chunk_ref = "none"
        for g in grades:
            if g.get("sentence", "")[:60].strip() == key:
                chunk_ref = g.get("chunk_ref", "none")
                break

        keep = grade != "UNSUPPORTED"
        if not keep:
            to_remove.append(sent)

        enriched.append({
            "sentence":  sent,
            "grade":     grade,
            "chunk_ref": chunk_ref,
            "kept":      keep,
        })

    # Remove UNSUPPORTED sentences in-place to preserve structure
    cleaned = original
    for sent in to_remove:
        cleaned = cleaned.replace(sent, "")

    cleaned = _re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = _re.sub(r'[ \t]{2,}', ' ', cleaned)
    cleaned = cleaned.strip()

    return cleaned, enriched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade_answer(answer: str, chunks: list) -> dict:
    """
    Grade every sentence in the answer against the source chunks,
    and return a cleaned answer with UNSUPPORTED sentences removed.
    """
    sentences = _split_sentences(answer)

    if not sentences:
        return {
            "cleaned_answer":  answer,
            "original_answer": answer,
            "grades":          [],
            "removed_count":   0,
            "grading_failed":  False,
        }

    grades = _call_grader(sentences, chunks)

    if grades is None:
        print("[evidence_grader] Grading failed — returning original answer unchanged.")
        return {
            "cleaned_answer":  answer,
            "original_answer": answer,
            "grades":          [],
            "removed_count":   0,
            "grading_failed":  True,
        }

    cleaned_text, enriched_grades = _reconstruct_answer(answer, sentences, grades)
    removed = sum(1 for g in enriched_grades if not g["kept"])

    print(
        f"[evidence_grader] {len(sentences)} sentences graded | "
        f"{removed} UNSUPPORTED removed | "
        f"{len(sentences) - removed} kept"
    )

    return {
        "cleaned_answer":  cleaned_text,
        "original_answer": answer,
        "grades":          enriched_grades,
        "removed_count":   removed,
        "grading_failed":  False,
    }