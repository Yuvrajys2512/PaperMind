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

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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
        # Keep section headers as single units
        if line.startswith("**") and line.endswith("**"):
            sentences.append(line)
            continue
        # Split on ". " but avoid splitting on abbreviations like "Fig. 3"
        parts = line.split(". ")
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            # Re-attach the period (except for the last part if line ended cleanly)
            if i < len(parts) - 1:
                part = part + "."
            if len(part) > 15:   # skip very short fragments
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
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.0,   # deterministic — grading should be stable
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if model adds them despite instructions
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

def _reconstruct_answer(sentences: list[str], grades: list[dict]) -> tuple[str, list[dict]]:
    """
    Rebuilds the answer, dropping UNSUPPORTED sentences.

    Returns
    -------
    cleaned_text : str
    enriched_grades : list[dict]  — grades with 'kept' boolean added
    """
    # Build a grade map keyed by sentence text (first 60 chars for robustness)
    grade_map: dict[str, str] = {}
    for g in grades:
        key = g.get("sentence", "")[:60].strip()
        grade_map[key] = g.get("grade", "INFERRED")

    enriched: list[dict] = []
    kept_sentences: list[str] = []

    for i, sent in enumerate(sentences):
        key   = sent[:60].strip()
        grade = grade_map.get(key, "INFERRED")   # default INFERRED if not found

        # Find the matching grade entry for chunk_ref
        chunk_ref = "none"
        for g in grades:
            if g.get("sentence", "")[:60].strip() == key:
                chunk_ref = g.get("chunk_ref", "none")
                break

        keep = grade != "UNSUPPORTED"
        if keep:
            kept_sentences.append(sent)

        enriched.append({
            "sentence":  sent,
            "grade":     grade,
            "chunk_ref": chunk_ref,
            "kept":      keep,
        })

    cleaned_text = " ".join(kept_sentences)
    return cleaned_text, enriched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade_answer(answer: str, chunks: list) -> dict:
    """
    Grade every sentence in the answer against the source chunks,
    and return a cleaned answer with UNSUPPORTED sentences removed.

    Parameters
    ----------
    answer : str   The generated answer (ESSENCE + DETAIL format).
    chunks : list  Retrieved chunks used to generate the answer.

    Returns
    -------
    dict:
        cleaned_answer  : str         Answer with UNSUPPORTED sentences removed.
        original_answer : str         Unmodified original answer.
        grades          : list[dict]  Per-sentence: sentence, grade, chunk_ref, kept.
        removed_count   : int         Number of sentences removed.
        grading_failed  : bool        True if LLM call failed; original returned as-is.
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
        # Grading failed — return original answer unchanged so the pipeline
        # doesn't crash. Flag it so the caller can log/surface the failure.
        print("[evidence_grader] Grading failed — returning original answer unchanged.")
        return {
            "cleaned_answer":  answer,
            "original_answer": answer,
            "grades":          [],
            "removed_count":   0,
            "grading_failed":  True,
        }

    cleaned_text, enriched_grades = _reconstruct_answer(sentences, grades)

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