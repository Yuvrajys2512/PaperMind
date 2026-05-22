"""
ingestion/off_topic_guard.py

Cheap pre-flight check before the RAG pipeline runs.
Makes a single tiny LLM call (max_tokens=5) to classify whether the
user's question is related to academic / scientific research.

Returns (is_off_topic: bool, friendly_message: str).
On any LLM failure, returns (False, "") so the pipeline always runs as
a safe fallback — never blocks the user because of a guard error.
"""

from __future__ import annotations
from ingestion.llm_client import chat_completion

_SYSTEM = (
    "You are a one-word classifier. "
    "Decide whether the user's question is asking about something that "
    "could plausibly be covered in an academic research paper — methodology, "
    "results, concepts, comparisons, summaries, or related scientific topics. "
    "Reply with exactly one word: RESEARCH or OFFTOPIC. No punctuation."
)

_OFF_TOPIC_REPLY = (
    "Haha, I wish I could help with that! But I'm a research paper nerd at heart — "
    "I live and breathe methodology, results, and scientific contributions. "
    "For scores, weather, and the rest of life's important questions, you'll need "
    "to look elsewhere. Now, got anything you want to dig into from the paper? I'm all yours!"
)


def check(question: str) -> tuple[bool, str]:
    """
    Returns (is_off_topic, message).
    is_off_topic=True  → caller should short-circuit with message.
    is_off_topic=False → proceed normally.
    """
    try:
        verdict = chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": question},
            ],
            max_tokens=5,
            temperature=0.0,
        ).strip().upper()
        if verdict.startswith("OFFTOPIC"):
            return True, _OFF_TOPIC_REPLY
        return False, ""
    except Exception:
        # Guard failure → let the pipeline run rather than blocking the user
        return False, ""
