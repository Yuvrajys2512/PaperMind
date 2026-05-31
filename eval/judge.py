"""
eval/judge.py — LLM-as-judge for QASPER answers.

Token-F1 punishes semantically correct answers that happen to be phrased
differently from the terse human reference (e.g. a prose answer vs. the gold
citation "BIBREF19"). The judge catches those: it compares the system answer to
the annotator reference(s) on meaning and returns CORRECT / PARTIAL / INCORRECT.

Runs through the project's unified rotating LLM client, so it shares the same
provider-fallback behavior as the rest of the pipeline.
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


def judge_answer(question: str, gold_texts: list[str], system_answer: str) -> dict:
    """Grade ``system_answer`` against the reference answers.

    Returns ``{verdict, score, judged}`` where ``judged`` is False if the LLM
    call failed or returned something unparseable (so the caller can exclude it
    from judged-accuracy without crashing a long run).
    """
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
