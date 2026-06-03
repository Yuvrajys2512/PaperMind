from ingestion.llm_client import chat_completion

_SYSTEM = "You are a writing assistant. Follow the instruction exactly. Return only the rewritten text — no preamble, no commentary, no quotes around the output."

_PROMPTS = {
    "academic": (
        "Rewrite the following text to improve academic clarity and precision. "
        "Maintain all claims and meaning exactly. Use formal tone, precise vocabulary, "
        "and proper hedging where appropriate (e.g. 'suggests', 'indicates', 'demonstrates'). "
        "Do not add new information, arguments, or citations."
    ),
    "plain": (
        "Rewrite the following text in plain, accessible language. "
        "Replace jargon and technical terms with everyday equivalents. "
        "Keep every key point — do not drop information. "
        "Write for a smart reader with no specialist background in this field."
    ),
    "concise": (
        "Rewrite the following text to be more concise. "
        "Cut redundant phrases, filler words, and unnecessary hedging. "
        "Preserve all key information and claims. "
        "Aim to reduce length by roughly 30–50% without losing substance."
    ),
}


def rewrite_text(text: str, mode: str) -> str:
    if mode not in _PROMPTS:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {list(_PROMPTS)}")
    if not text.strip():
        raise ValueError("Text cannot be empty.")

    instruction = _PROMPTS[mode]
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",   "content": f"{instruction}\n\n---\n\n{text.strip()}"},
    ]
    return chat_completion(messages, max_tokens=2048, temperature=0.3)
