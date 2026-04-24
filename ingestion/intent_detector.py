import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INTENT_PROMPT = """You are a query intent classifier for a research paper Q&A system.

Classify the following query into exactly one of these intents:
- factual: asking for a specific fact, number, name, or definition
- summarization: asking for an overview or summary of a topic or section
- critique: asking about limitations, weaknesses, or problems
- comparison: asking how something compares to or differs from something else

Query: {query}

Respond with exactly one word — the intent label. Nothing else."""


def detect_intent(query: str) -> str:
    """
    Classifies query into: factual | summarization | critique | comparison
    Falls back to 'factual' if LLM returns unexpected output.
    """
    prompt = INTENT_PROMPT.format(query=query)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0
    )

    raw = response.choices[0].message.content.strip().lower()

    valid_intents = {"factual", "summarization", "critique", "comparison"}
    if raw in valid_intents:
        return raw

    # fallback — try to find intent word anywhere in response
    for intent in valid_intents:
        if intent in raw:
            return intent

    return "factual"