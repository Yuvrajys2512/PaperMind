"""
ingestion/llm_client.py

Unified LLM client with automatic provider rotation and fallback.

Priority order (fastest / most reliable first):
  1. Groq key pool     — LPU, ~150ms; GROQ_API_KEY + GROQ_API_KEY_2.._8, each with
                         its own budget, auto-discovered as Groq-1, Groq-2, …
  2. Gemini Flash Lite — free quota, ~1s, 2.5-flash-lite
  3. Mistral Small     — reliable fallback, 1B tokens/month
  4. Cerebras          — last: gpt-oss-120b returns null content intermittently

On rate limit / quota errors the client automatically tries the next provider.
If a whole pass exhausts every provider AND at least one was hit by a
self-clearing per-minute rate limit, the client sleeps (escalating backoff) and
retries the chain — so transient TPM/RPM contention recovers instead of
skipping the call. Daily-quota / auth / not-found errors do NOT trigger backoff
(they can't clear in time); the call fails fast.
On unexpected errors (network, auth) the error is re-raised immediately.

Public API
----------
chat_completion(messages, max_tokens, temperature) -> str
"""

from __future__ import annotations
import os
import time
import threading
from openai import OpenAI, APIConnectionError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

# Per-thread call stats. FastAPI runs sync pipeline work in a thread-pool
# executor, so thread-local isolates one request's stats from another's.
# The pipeline calls reset_stats() at the start of a query and get_stats()
# at the end to surface llm_calls / providers_used / cost on the response.
_stats_local = threading.local()


def reset_stats() -> None:
    """Zero the per-thread LLM call counter. Call at the start of a request."""
    _stats_local.stats = {
        "call_count": 0, "providers": [],
        "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
    }


def get_stats() -> dict:
    """Read per-thread LLM call stats accumulated since the last reset."""
    s = getattr(_stats_local, "stats", None)
    if s is None:
        return {
            "call_count": 0, "providers": [],
            "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
        }
    return {
        "call_count":  s["call_count"],
        "providers":   list(s["providers"]),
        "tokens_in":   s["tokens_in"],
        "tokens_out":  s["tokens_out"],
        "cost_usd":    s["cost_usd"],
    }


# Public list price per 1M tokens, USD (verified Jun 2026 — see provider pricing
# pages). The keys we run on today are free-tier, so this is purely a cost
# *projection*: "what would this have cost on paid keys" — exactly what's needed
# before pricing a freemium tier. Unpriced (provider, model) pairs cost $0
# rather than crashing, so a newly added model just under-counts until priced.
_PRICING: dict[tuple[str, str], tuple[float, float]] = {
    ("Groq-1",   "llama-3.3-70b-versatile"): (0.59, 0.79),
    ("Groq-2",   "llama-3.3-70b-versatile"): (0.59, 0.79),
    ("Gemini",   "gemini-2.5-flash-lite"):   (0.10, 0.40),
    ("Mistral",  "mistral-small-latest"):    (0.20, 0.60),
    ("Cerebras", "gpt-oss-120b"):            (0.35, 0.75),
}


def _cost(provider_name: str, model: str, tokens_in: int, tokens_out: int) -> float:
    """USD cost of one call, or 0.0 if the (provider, model) pair isn't priced."""
    # Every Groq key (Groq-1, Groq-2, … Groq-N) runs the same model at the same
    # price; collapse to the canonical "Groq-1" entry so overflow keys are still
    # priced instead of silently projecting $0.
    canon = "Groq-1" if provider_name.startswith("Groq-") else provider_name
    price_in, price_out = _PRICING.get((canon, model), (0.0, 0.0))
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


_PROVIDERS: list[dict] = []

# Provider order = throughput priority. Every client uses max_retries=0 so a
# rate-limited/failing provider fails over to the next one INSTANTLY instead of
# spending the OpenAI SDK's internal retry/backoff (~2s+ per 429). We own the
# fallback loop in chat_completion(), so SDK-level retries only slow it down.
#
# Order rationale:
#   Groq    — fastest (LPU, ~150ms) + strong 70B model + fresh quota; workhorse.
#   Gemini  — fast and free (2.5-flash-lite); see model note below.
#   Mistral — reliable fallback.
#   Cerebras— last: gpt-oss-120b returns null content intermittently.

# Groq key pool: GROQ_API_KEY (primary) plus GROQ_API_KEY_2, _3, … up to _8.
# Each key carries its own independent RPM/TPM/daily budget on the same fast LPU
# model, so they all sit at the front of the chain as the workhorse pool — under
# concurrent eval load the fallback loop rotates across them before ever reaching
# Gemini/Mistral. Auto-discovered so adding a key to .env is the only step
# needed; gaps are fine (a missing _3 doesn't stop _4 being picked up).
_GROQ_KEY_ENVS = ["GROQ_API_KEY"] + [f"GROQ_API_KEY_{i}" for i in range(2, 9)]
for _idx, _env in enumerate(_GROQ_KEY_ENVS, start=1):
    _key = os.getenv(_env)
    if _key:
        _PROVIDERS.append({
            "name":   f"Groq-{_idx}",
            "client": OpenAI(
                api_key     = _key,
                base_url    = "https://api.groq.com/openai/v1",
                max_retries = 0,
            ),
            "model": "llama-3.3-70b-versatile",
        })

if os.getenv("GEMINI_API_KEY"):
    _PROVIDERS.append({
        "name":   "Gemini",
        "client": OpenAI(
            api_key     = os.getenv("GEMINI_API_KEY"),
            base_url    = "https://generativelanguage.googleapis.com/v1beta/openai/",
            max_retries = 0,
        ),
        # gemini-2.0-flash has free-tier quota 0 (every call 429s with
        # RESOURCE_EXHAUSTED limit:0). 2.5-flash-lite has real free quota, is
        # fast (~1s), and — unlike 2.5-flash — does not burn the token budget on
        # "thinking", so it won't truncate planning JSON / answers.
        "model": "gemini-2.5-flash-lite",
    })

if os.getenv("MISTRAL_API_KEY"):
    _PROVIDERS.append({
        "name":   "Mistral",
        "client": OpenAI(
            api_key     = os.getenv("MISTRAL_API_KEY"),
            base_url    = "https://api.mistral.ai/v1",
            max_retries = 0,
        ),
        "model": "mistral-small-latest",
    })

if os.getenv("CEREBRAS_API_KEY"):
    _PROVIDERS.append({
        "name":   "Cerebras",
        "client": OpenAI(
            api_key     = os.getenv("CEREBRAS_API_KEY"),
            base_url    = "https://api.cerebras.ai/v1",
            max_retries = 0,
        ),
        "model": "gpt-oss-120b",
    })

if not _PROVIDERS:
    raise RuntimeError(
        "[llm_client] No API keys found in .env. "
        "Add at least one of: GEMINI_API_KEY, CEREBRAS_API_KEY, "
        "MISTRAL_API_KEY, GROQ_API_KEY, GROQ_API_KEY_2"
    )


_SKIP_KEYWORDS = [
    "rate", "quota", "429", "limit", "exhausted",
    "invalid_argument", "api_key_invalid", "invalid api key",
    "resource_exhausted", "too many requests",
    "not_found", "404", "does not exist",
]

# Rate-limit errors that recover ON THEIR OWN after a short wait — i.e. a
# per-MINUTE TPM/RPM window (Groq's "Rate limit reached … please try again in
# 2s"). These are worth sleeping on and retrying.
#
# Deliberately EXCLUDES daily-quota exhaustion (Gemini's "You exceeded your
# current quota") and auth/not-found errors: those won't clear within a backoff
# window (the quota resets in 24h), so sleeping on them only wastes wall-clock.
# Such errors still match _SKIP_KEYWORDS → we skip that provider, but they do
# NOT trigger a backoff-and-retry pass.
_RETRYABLE_RATE_LIMIT = [
    "rate limit reached",     # Groq, per-minute window
    "please try again in",    # Groq appends a short retry hint
    "requests per minute",
    "tokens per minute",
    "rpm", "tpm",
]

# Whole-loop backoff: when EVERY provider in one pass was momentarily
# rate-limited, sleep and retry the whole chain. One entry per extra pass;
# values are seconds. Bounded so a provider that is actually day-exhausted
# (but reports via a retryable-looking message) costs at most ~sum(schedule).
_BACKOFF_SCHEDULE = (5, 15, 30)


def _is_retryable_rate_limit(err_str: str) -> bool:
    """True if the error is a self-clearing per-minute rate limit (worth a wait)."""
    return any(kw in err_str for kw in _RETRYABLE_RATE_LIMIT)


def chat_completion(
    messages:    list[dict],
    max_tokens:  int   = 1000,
    temperature: float = 0.1,
    pin:         tuple[str, str] | None = None,
) -> str:
    """
    Send a chat completion request, automatically falling back through
    providers on rate limit / quota errors.

    Unexpected errors (network failures, bad auth) are re-raised immediately
    rather than silently swallowed.

    pin : optional (provider_name, model_id). When set, the call is forced onto
          that single provider+model with NO cross-provider fallback — so the
          model is held fixed. Used by experiments (e.g. weak-vs-strong
          generator) that must control which model produced the output.

    Returns the model's response text.
    Raises RuntimeError if ALL providers are rate-limited / exhausted.
    """
    last_error = None

    providers = _PROVIDERS
    if pin is not None:
        name, model = pin
        base = next((p for p in _PROVIDERS if p["name"] == name), None)
        if base is None:
            raise RuntimeError(f"[llm_client] pinned provider '{name}' is not configured")
        providers = [{**base, "model": model}]

    # Outer loop = backoff passes. Pass 0 is the normal fast path (no sleep);
    # each later pass is entered only if every provider in the previous pass was
    # momentarily rate-limited, and is preceded by an escalating sleep. Note:
    # time.sleep blocks the calling thread — fine for the eval CLI and bounded
    # for live requests (max ~sum(_BACKOFF_SCHEDULE)).
    for pass_idx in range(len(_BACKOFF_SCHEDULE) + 1):
        if pass_idx > 0:
            wait = _BACKOFF_SCHEDULE[pass_idx - 1]
            print(f"[llm_client] all providers rate-limited — backing off {wait}s "
                  f"(retry pass {pass_idx}/{len(_BACKOFF_SCHEDULE)})...")
            time.sleep(wait)

        saw_retryable = False

        for provider in providers:
            try:
                response = provider["client"].chat.completions.create(
                    model       = provider["model"],
                    messages    = messages,
                    max_tokens  = max_tokens,
                    temperature = temperature,
                )
                content = response.choices[0].message.content
                if content is None:
                    print(f"[llm_client] {provider['name']} returned null content — trying next provider...")
                    last_error = ValueError("null content")
                    continue
                text = content.strip()
                print(f"[llm_client] Used: {provider['name']}")
                usage = getattr(response, "usage", None)
                tokens_in  = getattr(usage, "prompt_tokens", 0) or 0
                tokens_out = getattr(usage, "completion_tokens", 0) or 0
                s = getattr(_stats_local, "stats", None)
                if s is not None:
                    s["call_count"] += 1
                    s["providers"].append(provider["name"])
                    s["tokens_in"]  += tokens_in
                    s["tokens_out"] += tokens_out
                    s["cost_usd"]   += _cost(provider["name"], provider["model"], tokens_in, tokens_out)
                return text

            except Exception as e:
                err_str = str(e).lower()
                # Network/timeout errors: this one provider is unreachable right
                # now (DNS, TCP, TLS, read timeout). Don't kill the whole request
                # on it — fail over to the next provider, and treat it as
                # retryable so a transient blip recovers on a backoff pass.
                conn_err = isinstance(e, (APIConnectionError, APITimeoutError))
                retryable = _is_retryable_rate_limit(err_str) or conn_err
                if retryable or any(kw in err_str for kw in _SKIP_KEYWORDS):
                    saw_retryable = saw_retryable or retryable
                    if conn_err:
                        kind = "unreachable (retryable)"
                    elif retryable:
                        kind = "rate-limited (retryable)"
                    else:
                        kind = "skipped"
                    print(f"[llm_client] {provider['name']} {kind} "
                          f"({type(e).__name__}: {str(e)[:120]}) — trying next provider...")
                    last_error = e
                    continue  # try the next provider
                raise  # unexpected error — surface it immediately

        # A full pass completed with no success. Only sleep & retry if at least
        # one provider was retryably rate-limited; if all failures were daily
        # quota / auth / not-found, waiting cannot help — fail fast.
        if not saw_retryable:
            break

    raise RuntimeError(
        f"[llm_client] All {len(providers)} provider(s) exhausted"
        f"{' after backoff' if _BACKOFF_SCHEDULE else ''}. "
        f"Last error: {last_error}"
    )
