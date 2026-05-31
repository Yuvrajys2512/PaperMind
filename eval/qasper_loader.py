"""
eval/qasper_loader.py — Download, cache, and parse the QASPER benchmark.

QASPER (Dasigi et al., NAACL 2021) is a dataset of information-seeking
questions over NLP papers. Each question carries one or more annotator answers
that are extractive / abstractive / yes-no / unanswerable, plus gold *evidence*
paragraphs. Those evidence spans are what make QASPER the right fit for
measuring PaperMind's evidence-grading contribution.

We use the official v0.3 train/dev release (one ~10 MB tarball). `dev` is our
development split; the separate test tarball is reserved for final numbers.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import httpx

# Official AllenAI release (train + dev in one gzipped tar).
QASPER_TRAIN_DEV_URL = (
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "qasper"
_MEMBER = {"dev": "qasper-dev-v0.3.json", "train": "qasper-train-v0.3.json"}


def ensure_split(split: str = "dev") -> Path:
    """Return the local path to the QASPER ``split`` JSON.

    On first use this downloads the official tarball and extracts both train
    and dev JSON files into ``data/qasper/``. Subsequent calls hit the cache.
    """
    if split not in _MEMBER:
        raise ValueError(f"split must be one of {list(_MEMBER)}, got {split!r}")

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = _DATA_DIR / _MEMBER[split]
    if out.exists():
        return out

    print("[qasper] downloading train/dev tarball (~10 MB) ...")
    with httpx.stream(
        "GET", QASPER_TRAIN_DEV_URL, follow_redirects=True, timeout=120
    ) as resp:
        resp.raise_for_status()
        buf = io.BytesIO(resp.read())

    print("[qasper] extracting ...")
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        for member_file in _MEMBER.values():
            try:
                member = tar.getmember(member_file)
            except KeyError:
                # Some tarballs prefix entries with a folder name; match by suffix.
                member = next(
                    m for m in tar.getmembers() if m.name.endswith(member_file)
                )
            extracted = tar.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"could not read {member_file} from tarball")
            (_DATA_DIR / member_file).write_bytes(extracted.read())

    print(f"[qasper] cached -> {out}")
    return out


def load_papers(split: str = "dev") -> dict:
    """Return ``{paper_id: paper_dict}`` for the given split."""
    path = ensure_split(split)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_answer(answer_entry: dict) -> dict:
    """Collapse one annotator's QASPER answer object into a flat record.

    QASPER stores four mutually exclusive answer kinds in a single object. We
    resolve them in QASPER's own priority order into:

        {answerable: bool, type: str, text: str, evidence: list[str]}

    where ``type`` is one of ``extractive | abstractive | boolean | none`` and
    ``text`` is a single reference string suitable for display and token-F1.
    """
    a = answer_entry.get("answer", {})
    evidence = [e for e in a.get("evidence", []) if e and e.strip()]

    if a.get("unanswerable"):
        return {"answerable": False, "type": "none",
                "text": "Unanswerable", "evidence": evidence}

    if a.get("yes_no") is not None:
        return {"answerable": True, "type": "boolean",
                "text": "Yes" if a["yes_no"] else "No", "evidence": evidence}

    spans = [s for s in a.get("extractive_spans", []) if s and s.strip()]
    if spans:
        return {"answerable": True, "type": "extractive",
                "text": " ; ".join(spans), "evidence": evidence}

    free_form = (a.get("free_form_answer") or "").strip()
    if free_form:
        return {"answerable": True, "type": "abstractive",
                "text": free_form, "evidence": evidence}

    # No content of any kind — treat as unanswerable.
    return {"answerable": False, "type": "none",
            "text": "Unanswerable", "evidence": evidence}


def iter_questions(paper: dict):
    """Yield ``(question_text, question_id, [normalized answers])`` per question."""
    for qa in paper.get("qas", []):
        answers = [normalize_answer(a) for a in qa.get("answers", [])]
        yield qa.get("question", ""), qa.get("question_id", ""), answers
