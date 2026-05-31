"""
eval/metrics.py — Scoring primitives for the QASPER evaluation harness.

Everything here is a pure function with no I/O and no LLM calls, so it can be
unit-tested instantly. Three families:

  * Answer-F1   — SQuAD-style token F1, taken as the max over a question's
                  annotator reference answers (the official QASPER convention).
  * Answerable  — did the system correctly decide whether the paper answers the
                  question at all (vs. abstaining)?
  * Evidence    — token-level recall / F1 of retrieved context against the gold
                  evidence paragraphs. (Adaptation: PaperMind retrieves chunks,
                  not QASPER's original paragraphs, so we score on text overlap
                  rather than exact paragraph-set membership.)
"""

from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(s: str) -> str:
    """SQuAD/QASPER normalization: lowercase, drop punctuation, articles, and
    collapse whitespace. The standard pre-step before token overlap."""
    s = (s or "").lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _tokens(s: str) -> list[str]:
    return normalize_text(s).split()


def token_f1(prediction: str, ground_truth: str) -> float:
    """SQuAD-style token-overlap F1 between two strings."""
    pred_toks = _tokens(prediction)
    gold_toks = _tokens(ground_truth)

    # Mirrors the official handling of empty strings: F1 is 1.0 only if both
    # sides are empty, else 0.0.
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)

    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# --- abstention -------------------------------------------------------------

# Phrases PaperMind emits when it declines to answer from the paper.
_ABSTAIN_MARKERS = (
    "unable to answer",
    "does not appear to be related",
    "not contain sufficient information",
    "cannot answer",
    "no answer",
)


def looks_like_abstention(answer: str) -> bool:
    """True if the answer text reads as a refusal / 'I can't answer this'."""
    a = (answer or "").lower()
    return any(m in a for m in _ABSTAIN_MARKERS)


def is_no_answer(answer: str, passed: bool) -> bool:
    """The system's effective 'unanswerable' decision: it either failed the
    confidence gate or explicitly abstained in the text."""
    return (not passed) or looks_like_abstention(answer)


# --- answer scoring ---------------------------------------------------------

def answer_f1(
    prediction: str,
    predicted_no_answer: bool,
    gold_answers: list[dict],
) -> float:
    """Best Answer-F1 across a question's annotator references.

    ``gold_answers`` is the list produced by ``qasper_loader.normalize_answer``
    (each has ``answerable``, ``type``, ``text``). For an *unanswerable*
    reference, the system scores 1.0 iff it also produced no answer, else 0.0 —
    token overlap is meaningless there. For answerable references we use
    SQuAD token-F1 on the reference text.
    """
    if not gold_answers:
        return 0.0

    best = 0.0
    for gold in gold_answers:
        if not gold.get("answerable", True):
            score = 1.0 if predicted_no_answer else 0.0
        elif predicted_no_answer:
            score = 0.0
        else:
            score = token_f1(prediction, gold.get("text", ""))
        best = max(best, score)
    return best


def gold_is_answerable(gold_answers: list[dict]) -> bool:
    """Majority vote over annotators: is the question answerable from the paper?"""
    if not gold_answers:
        return False
    answerable = sum(1 for g in gold_answers if g.get("answerable", True))
    return answerable >= (len(gold_answers) / 2)


def answerable_correct(predicted_no_answer: bool, gold_answers: list[dict]) -> bool:
    """Did the system make the right answerable-vs-abstain call?"""
    return (not predicted_no_answer) == gold_is_answerable(gold_answers)


# --- evidence scoring -------------------------------------------------------

def evidence_recall(
    retrieved_texts: list[str],
    gold_evidence: list[str],
    coverage_threshold: float = 0.5,
) -> float | None:
    """Fraction of gold evidence paragraphs 'covered' by the retrieved context.

    A gold paragraph counts as covered if at least ``coverage_threshold`` of its
    (normalized, unique) tokens appear in some single retrieved chunk. Returns
    ``None`` when the question has no gold evidence (e.g. unanswerable), so the
    caller can exclude it from the average.
    """
    gold_evidence = [g for g in gold_evidence if g and g.strip()]
    if not gold_evidence:
        return None
    if not retrieved_texts:
        return 0.0

    retrieved_token_sets = [set(_tokens(t)) for t in retrieved_texts]

    covered = 0
    for paragraph in gold_evidence:
        gold_toks = set(_tokens(paragraph))
        if not gold_toks:
            continue
        best = max(
            (len(gold_toks & chunk) / len(gold_toks) for chunk in retrieved_token_sets),
            default=0.0,
        )
        if best >= coverage_threshold:
            covered += 1
    return covered / len(gold_evidence)


def evidence_token_f1(retrieved_texts: list[str], gold_evidence: list[str]) -> float | None:
    """Token-F1 between all retrieved context and all gold evidence concatenated.
    ``None`` when there is no gold evidence."""
    gold_evidence = [g for g in gold_evidence if g and g.strip()]
    if not gold_evidence:
        return None
    return token_f1(" ".join(retrieved_texts), " ".join(gold_evidence))
