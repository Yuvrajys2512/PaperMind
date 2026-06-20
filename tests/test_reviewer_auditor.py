"""
Unit tests for the reviewer / weakness auditor (ingestion/reviewer_auditor.py).

Pure-function tests — no network, no LLM — covering the deterministic glue that
guards against the module's one real failure mode (a false "MISSING" accusation):
  - the section map / present-section inventory the model is grounded on
  - _attach_verdict's evidence resolution, validation, and the conservative
    MISSING -> WEAK downgrade when a section on-topic actually exists.

The LLM-driven verdict itself is exercised manually against real papers.
"""

from ingestion.reviewer_auditor import (
    DIMENSIONS,
    build_section_map,
    _present_section_hints,
    _attach_verdict,
)


def _chunk(text, section, section_type="text", page=1):
    return {"text": text, "metadata": {
        "section": section, "section_type": section_type, "page_num": page,
    }}


def _dim(key):
    return next(d for d in DIMENSIONS if d["key"] == key)


# --- section map ------------------------------------------------------------

def test_section_map_lists_distinct_sections_and_counts_tables():
    chunks = [
        _chunk("...", "Introduction"),
        _chunk("...", "Introduction"),          # dup collapses
        _chunk("BERT — F1: 0.79", "Results", section_type="table"),
        _chunk("...", "Related Work"),
    ]
    out = build_section_map(chunks)
    assert "Introduction, Results, Related Work" in out
    assert "Tables detected: 1" in out


def test_section_map_handles_no_labels():
    out = build_section_map([_chunk("body", "")])
    assert "no section labels detected" in out
    assert "Tables detected: 0" in out


def test_present_section_hints_are_lowercased():
    hints = _present_section_hints([_chunk("...", "Related Work"), _chunk("...", "Ablation")])
    assert hints == {"related work", "ablation"}


# --- verdict attachment -----------------------------------------------------

def test_attach_resolves_cited_evidence_chunk():
    evidence = [_chunk("we compare against BERT and RoBERTa", "Experiments", page=4)]
    verdict = {"verdict": "OK", "severity": "low", "why": "compares vs BERT/RoBERTa",
               "reviewer_question": "why not GPT?", "evidence_ref": 1}
    out = _attach_verdict(_dim("baselines"), evidence, verdict, present_sections=set())
    assert out["verdict"] == "OK"
    assert out["evidence"]["section"] == "Experiments"
    assert out["evidence"]["page"] == 4
    assert out["reviewer_question"] == "why not GPT?"


def test_attach_falls_back_to_dim_question_when_model_omits_one():
    out = _attach_verdict(_dim("ablations"), [], {"verdict": "MISSING", "evidence_ref": None},
                          present_sections=set())
    assert out["reviewer_question"] == _dim("ablations")["ask"]


def test_invalid_verdict_and_severity_default_safely():
    out = _attach_verdict(_dim("related_work"), [], {"verdict": "BOGUS", "severity": "nope"},
                          present_sections=set())
    assert out["verdict"] == "MISSING"
    assert out["severity"] == "med"


def test_out_of_range_evidence_ref_yields_no_citation():
    evidence = [_chunk("body", "Results")]
    out = _attach_verdict(_dim("baselines"), evidence,
                          {"verdict": "OK", "evidence_ref": 5}, present_sections=set())
    assert out["evidence"] is None


# --- the conservative guard: never accuse "MISSING" when the section exists --

def test_missing_downgraded_to_weak_when_section_on_topic_exists():
    # The paper HAS a Related Work section, so related-work can't be "missing".
    out = _attach_verdict(
        _dim("related_work"), [], {"verdict": "MISSING", "evidence_ref": None},
        present_sections={"related work"},
    )
    assert out["verdict"] == "WEAK"


def test_missing_stays_missing_when_no_matching_section():
    out = _attach_verdict(
        _dim("ablations"), [], {"verdict": "MISSING", "evidence_ref": None},
        present_sections={"introduction", "results"},
    )
    assert out["verdict"] == "MISSING"


def test_statistical_rigor_has_no_section_hint_so_never_auto_downgraded():
    # Rigor lives inside results, not its own section — a MISSING here must
    # survive even when other sections exist.
    assert _dim("statistical_rigor")["section_hints"] == ()
    out = _attach_verdict(
        _dim("statistical_rigor"), [], {"verdict": "MISSING", "evidence_ref": None},
        present_sections={"results", "experiments"},
    )
    assert out["verdict"] == "MISSING"
