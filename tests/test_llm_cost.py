"""
Unit tests for ingestion/llm_client._cost — the per-call USD cost projection.

This is billing-critical math: it's what tells us "what would this query have
cost on paid keys" before the freemium tier is priced. We pin the formula
(per-1M-token list price), the zero-token base case, and the deliberate
fail-soft behavior for an unpriced (provider, model) pair: cost 0.0, not a crash.
"""

import pytest

from ingestion import llm_client
from ingestion.llm_client import _cost


def test_known_pair_prices_input_tokens():
    # Gemini flash-lite input is $0.10 / 1M tokens.
    assert _cost("Gemini", "gemini-2.5-flash-lite", 1_000_000, 0) == pytest.approx(0.10)


def test_known_pair_prices_output_tokens():
    # ...and $0.40 / 1M output tokens.
    assert _cost("Gemini", "gemini-2.5-flash-lite", 0, 1_000_000) == pytest.approx(0.40)


def test_known_pair_sums_input_and_output():
    # 500k in + 500k out = half of each rate.
    expected = (500_000 * 0.10 + 500_000 * 0.40) / 1_000_000
    assert _cost("Gemini", "gemini-2.5-flash-lite", 500_000, 500_000) == pytest.approx(expected)


def test_zero_tokens_costs_nothing():
    assert _cost("Groq-1", "llama-3.3-70b-versatile", 0, 0) == 0.0


def test_unpriced_pair_is_free_not_an_error():
    # A newly added (provider, model) that isn't in _PRICING must under-count
    # to $0 rather than raise — a missing price should never crash a request.
    assert _cost("BrandNewProvider", "some-future-model", 1_000_000, 1_000_000) == 0.0


def test_every_priced_pair_has_two_positive_rates():
    # Guard the pricing table itself: each entry is (price_in, price_out), both > 0.
    for (provider, model), rates in llm_client._PRICING.items():
        assert len(rates) == 2, f"{provider}/{model} must be (in, out)"
        assert all(r > 0 for r in rates), f"{provider}/{model} has a non-positive rate"
