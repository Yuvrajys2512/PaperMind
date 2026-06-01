"""
ingestion/llm_client.py

Unified LLM client with automatic provider rotation and fallback.

Priority order (most generous → least):
  1. Gemini Flash      — 1M tokens/day
  2. Cerebras          — ~1M tokens/day
  3. Mistral Small     — 1B tokens/month
  4. Groq key 1        — 100k tokens/day
  5. Groq key 2        — 100k tokens/day

On rate limit / quota errors the client automatically tries the next provider.
On unexpected errors (network, auth) the error is re-raised immediately.

Public API
----------
chat_completion(messages, max_tokens, temperature) -> str
"""

from __future__ import annotations
import os
import threading
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Per-thread call stats. FastAPI runs sync pipeline work in a thread-pool
# executor, so thread-local isolates one request's stats from another's.
# The pipeline calls reset_stats() at the start of a query and get_stats()
# at the end to surface llm_calls / providers_used on the response.
_stats_local = threading.local()


def reset_stats() -> None:
    """Zero the per-thread LLM call counter. Call at the start of a request."""
    _stats_local.stats = {"call_count": 0, "providers": []}


def get_stats() -> dict:
    """Read per-thread LLM call stats accumulated since the last reset."""
    s = getattr(_stats_local, "stats", None)
    if s is None:
        return {"call_count": 0, "providers": []}
    return {"call_count": s["call_count"], "providers": list(s["providers"])}


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

if os.getenv("GROQ_API_KEY"):
    _PROVIDERS.append({
        "name":   "Groq-1",
        "client": OpenAI(
            api_key     = os.getenv("GROQ_API_KEY"),
            base_url    = "https://api.groq.com/openai/v1",
            max_retries = 0,
        ),
        "model": "llama-3.3-70b-versatile",
    })

if os.getenv("GROQ_API_KEY_2"):
    _PROVIDERS.append({
        "name":   "Groq-2",
        "client": OpenAI(
            api_key     = os.getenv("GROQ_API_KEY_2"),
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
            s = getattr(_stats_local, "stats", None)
            if s is not None:
                s["call_count"] += 1
                s["providers"].append(provider["name"])
            return text

        except Exception as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in _SKIP_KEYWORDS):
                print(f"[llm_client] {provider['name']} skipped "
                      f"({type(e).__name__}: {str(e)[:120]}) — trying next provider...")
                last_error = e
                continue  # try the next provider
            raise  # unexpected error — surface it immediately

    raise RuntimeError(
        f"[llm_client] All {len(_PROVIDERS)} provider(s) exhausted. "
        f"Last error: {last_error}"
    )
