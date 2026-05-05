"""
ingestion/llm_client.py

Unified LLM client with automatic provider rotation and fallback.

Priority order (most generous → least):
  1. Gemini Flash      — 1M tokens/day
  2. Cerebras          — ~1M tokens/day  
  3. Mistral Small     — 1B tokens/month
  4. Groq key 1        — 100k tokens/day
  5. Groq key 2        — 100k tokens/day

On RateLimitError the client automatically tries the next provider.

Public API
----------
chat_completion(messages, max_tokens, temperature) -> str
"""

from __future__ import annotations
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_PROVIDERS: list[dict] = []

if os.getenv("GEMINI_API_KEY"):
    _PROVIDERS.append({
        "name":   "Gemini Flash",
        "client": OpenAI(
            api_key  = os.getenv("GEMINI_API_KEY"),
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        "model": "gemini-2.0-flash",
    })

if os.getenv("CEREBRAS_API_KEY"):
    _PROVIDERS.append({
        "name":   "Cerebras",
        "client": OpenAI(
            api_key  = os.getenv("CEREBRAS_API_KEY"),
            base_url = "https://api.cerebras.ai/v1",
        ),
        "model": "llama-3.3-70b",
    })

if os.getenv("MISTRAL_API_KEY"):
    _PROVIDERS.append({
        "name":   "Mistral",
        "client": OpenAI(
            api_key  = os.getenv("MISTRAL_API_KEY"),
            base_url = "https://api.mistral.ai/v1",
        ),
        "model": "mistral-small-latest",
    })

if os.getenv("GROQ_API_KEY"):
    _PROVIDERS.append({
        "name":   "Groq-1",
        "client": OpenAI(
            api_key  = os.getenv("GROQ_API_KEY"),
            base_url = "https://api.groq.com/openai/v1",
        ),
        "model": "llama-3.3-70b-versatile",
    })

if os.getenv("GROQ_API_KEY_2"):
    _PROVIDERS.append({
        "name":   "Groq-2",
        "client": OpenAI(
            api_key  = os.getenv("GROQ_API_KEY_2"),
            base_url = "https://api.groq.com/openai/v1",
        ),
        "model": "llama-3.3-70b-versatile",
    })

if not _PROVIDERS:
    raise RuntimeError(
        "[llm_client] No API keys found in .env. "
        "Add at least one of: GEMINI_API_KEY, CEREBRAS_API_KEY, "
        "MISTRAL_API_KEY, GROQ_API_KEY, GROQ_API_KEY_2"
    )

print(f"[llm_client] {len(_PROVIDERS)} provider(s) loaded: "
      f"{', '.join(p['name'] for p in _PROVIDERS)}")


def chat_completion(
    messages:    list[dict],
    max_tokens:  int   = 1000,
    temperature: float = 0.1,
) -> str:
    """
    Send a chat completion request, automatically falling back through
    providers on RateLimitError.

    Returns the model's response text.
    Raises RuntimeError if ALL providers are exhausted.
    """
    last_error = None

    for provider in _PROVIDERS:
        try:
            response = provider["client"].chat.completions.create(
                model       = provider["model"],
                messages    = messages,
                max_tokens  = max_tokens,
                temperature = temperature,
            )
            text = response.choices[0].message.content.strip()
            print(f"[llm_client] Used: {provider['name']}")
            return text

        except Exception as e:
            err_str = str(e).lower()
            skip_keywords = ["rate", "quota", "429", "limit", "exhausted",
                             "invalid_argument", "api_key_invalid", "400", "invalid api key"]
            if any(word in err_str for word in skip_keywords):
                print(f"[llm_client] {provider['name']} skipping ({type(e).__name__}) — trying next...")
                last_error = e
        continue
    else:
        raise

    raise RuntimeError(
        f"[llm_client] All {len(_PROVIDERS)} providers exhausted. "
        f"Last error: {last_error}"
    )