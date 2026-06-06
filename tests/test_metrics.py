"""
Unit tests for eval/metrics.py — the scoring primitives the QASPER evaluation
harness (and the paper's numbers) are built on. Pure functions, no network,
no LLM calls: this suite runs in milliseconds and is safe for CI.

If any of these break, every reported Answer-F1 / evidence / answerable number
is suspect — which is exactly why they're worth pinning down.
"""

from eval.metrics import (
    answer_f1,
    answerable_correct,
    evidence_recall,
    evidence_token_f1,
    gold_is_answerable,
    is_no_answer,
    looks_like_abstention,
    normalize_text,
    token_f1,
)

# --- normalize_text ---------------------------------------------------------

def test_normalize_lowercases_strips_punct_and_articles():
    assert normalize_text("The Quick, Brown Fox!") == "quick brown fox"


def test_normalize_collapses_whitespace():
    assert normalize_text("  a   b\tc\n") == "b c"  # 'a' is an article, dropped


def test_normalize_handles_none_and_empty():
    assert normalize_text(None) == ""
    assert normalize_text("") == ""


# --- token_f1 ---------------------------------------------------------------

def test_token_f1_exact_match_is_one():
    assert token_f1("transformer model", "transformer model") == 1.0


def test_token_f1_no_overlap_is_zero():
    assert token_f1("apples oranges", "transformer model") == 0.0


def test_token_f1_partial_overlap():
    # pred has 3 toks, gold has 2; 2 shared -> P=2/3, R=2/2=1 -> F1=0.8
    score = token_f1("alpha beta gamma", "alpha beta")
    assert abs(score - 0.8) < 1e-9


def test_token_f1_both_empty_is_one_else_zero():
    assert token_f1("", "") == 1.0
    assert token_f1("the a an", "") == 1.0       # normalizes to empty on both? no:
    assert token_f1("hello", "") == 0.0


def test_token_f1_is_order_independent():
    assert token_f1("brown fox quick", "quick brown fox") == 1.0


# --- abstention detection ---------------------------------------------------

def test_looks_like_abstention_true_cases():
    assert looks_like_abstention("Unable to answer this from the paper.")
    assert looks_like_abstention("This does not appear to be related to the paper.")


def test_looks_like_abstention_false_for_real_answer():
    assert not looks_like_abstention("The model uses multi-head self-attention.")


def test_is_no_answer_combines_gate_and_text():
    assert is_no_answer("anything", passed=False) is True          # failed the gate
    assert is_no_answer("cannot answer", passed=True) is True       # abstained in text
    assert is_no_answer("It achieves 91% F1.", passed=True) is False


# --- answer_f1 over annotator references ------------------------------------

def test_answer_f1_takes_best_over_references():
    gold = [
        {"answerable": True, "text": "self attention"},
        {"answerable": True, "text": "multi head attention mechanism"},
    ]
    # Matches the second reference better.
    score = answer_f1("multi head attention mechanism", False, gold)
    assert score == 1.0


def test_answer_f1_unanswerable_reference_rewards_abstention():
    gold = [{"answerable": False, "text": ""}]
    assert answer_f1("", True, gold) == 1.0      # correctly abstained
    assert answer_f1("some answer", False, gold) == 0.0


def test_answer_f1_answerable_but_system_abstained_scores_zero():
    gold = [{"answerable": True, "text": "the answer"}]
    assert answer_f1("", True, gold) == 0.0


def test_answer_f1_empty_gold_is_zero():
    assert answer_f1("anything", False, []) == 0.0


# --- answerable decision ----------------------------------------------------

def test_gold_is_answerable_majority_vote():
    assert gold_is_answerable([{"answerable": True}, {"answerable": True},
                               {"answerable": False}]) is True
    assert gold_is_answerable([{"answerable": False}, {"answerable": False}]) is False


def test_gold_is_answerable_ties_count_as_answerable():
    # 1 of 2 answerable -> 1 >= 1.0 -> True (boundary behavior)
    assert gold_is_answerable([{"answerable": True}, {"answerable": False}]) is True


def test_answerable_correct():
    gold = [{"answerable": True}]
    assert answerable_correct(predicted_no_answer=False, gold_answers=gold) is True
    assert answerable_correct(predicted_no_answer=True, gold_answers=gold) is False


# --- evidence scoring -------------------------------------------------------

def test_evidence_recall_full_coverage():
    gold = ["transformer self attention mechanism"]
    retrieved = ["the transformer uses a self attention mechanism throughout"]
    assert evidence_recall(retrieved, gold) == 1.0


def test_evidence_recall_no_coverage():
    gold = ["completely unrelated evidence paragraph words"]
    retrieved = ["the transformer uses self attention"]
    assert evidence_recall(retrieved, gold) == 0.0


def test_evidence_recall_none_when_no_gold():
    assert evidence_recall(["anything"], []) is None
    assert evidence_recall(["anything"], ["   "]) is None


def test_evidence_recall_zero_when_nothing_retrieved():
    assert evidence_recall([], ["some gold evidence"]) == 0.0


def test_evidence_token_f1_none_without_gold():
    assert evidence_token_f1(["retrieved"], []) is None


def test_evidence_token_f1_perfect_overlap():
    assert evidence_token_f1(["alpha beta gamma"], ["alpha beta gamma"]) == 1.0
