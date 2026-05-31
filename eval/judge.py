"""
eval/judge.py — LLM-as-judge for QASPER answers (majority-vote).

Token-F1 punishes semantically correct answers phrased differently from the
terse human reference (e.g. prose vs. the gold citation "BIBREF19"). The judge
compares the system answer to the reference(s) on meaning: CORRECT / PARTIAL /
INCORRECT.

A single judge call is stable on clear-cut answers but can waver on borderline
ones, and that wobble swamps small effects at low n. So `judge_answer` runs N
independent calls and returns the MEAN score (a less noisy estimator than one
discrete verdict), bucketed back to a verdict for display.

Runs through the project's unified rotating LLM client.
"""

from __future__ import annotations

from ingestion.llm_client import chat_completion

_SCORES = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}

_SYSTEM = (
    "You are a strict grader for question answering over a single research "
    "paper. You are given a question, one or more REFERENCE answers written by "
    "human annotators (these may be very terse), and a SYSTEM answer. Decide "
    "whether the SYSTEM answer conveys the same information as any reference "
    "answer. Ignore differences in wording, length, or formatting; judge meaning "
    "only. Reply with exactly one word: CORRECT, PARTIAL, or INCORRECT."
)


def _single_judge(question: str, gold_texts: list[str], system_answer: str) -> dict:
    """One judge call → {verdict, score, judged}."""
    references = "\n".join(f"  - {t}" for t in gold_texts if t and t.strip())
    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER(S):\n{references}\n\n"
        f"SYSTEM ANSWER:\n{system_answer}\n\n"
        "Verdict (CORRECT, PARTIAL, or INCORRECT):"
    )
    try:
        raw = chat_completion(
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": user}],
            max_tokens=8,
            temperature=0.0,
        )
    except Exception as e:  # never abort a run on a judge failure
        return {"verdict": "ERROR", "score": None, "judged": False, "error": str(e)}

    upper = (raw or "").upper()
    # Order matters: "INCORRECT" contains "CORRECT", so test it first.
    for verdict in ("INCORRECT", "PARTIAL", "CORRECT"):
        if verdict in upper:
            return {"verdict": verdict, "score": _SCORES[verdict], "judged": True}
    return {"verdict": "UNPARSED", "score": None, "judged": False, "raw": raw}


def _verdict_from_score(score: float) -> str:
    """Bucket a continuous mean score back into a discrete verdict for display."""
    if score >= 0.75:
        return "CORRECT"
    if score >= 0.25:
        return "PARTIAL"
    return "INCORRECT"


def judge_answer(question: str, gold_texts: list[str], system_answer: str,
                 votes: int = 3) -> dict:
    """Majority-of-N judge.

    Runs ``votes`` independent judge calls and returns the MEAN score — a less
    noisy estimator of answer quality than a single discrete verdict — with the
    verdict derived by bucketing that mean. ``judged`` is False only if every
    call failed/was unparseable, so a long run never crashes on the judge.

    Returns ``{verdict, score, judged, n_votes, vote_verdicts}``.
    """
    results = [_single_judge(question, gold_texts, system_answer)
               for _ in range(max(1, votes))]
    valid = [r for r in results if r["judged"]]

    if not valid:
        return {"verdict": results[0]["verdict"], "score": None, "judged": False,
                "n_votes": 0, "vote_verdicts": [r["verdict"] for r in results]}

    mean_score = sum(r["score"] for r in valid) / len(valid)
    return {
        "verdict": _verdict_from_score(mean_score),
        "score": mean_score,
        "judged": True,
        "n_votes": len(valid),
        "vote_verdicts": [r["verdict"] for r in valid],
    }
