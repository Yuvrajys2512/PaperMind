import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INTENT_PROMPT = """You are a query intent classifier for a research paper Q&A system.

Classify the following query into up to TWO most relevant intents from the list below.

Intents:
- factual: asking for a specific fact, number, name, or definition
- summarization: asking for an overview or summary
- critique: asking about limitations, weaknesses, or problems
- comparison: comparing two or more things
- mechanism: asking how something works or operates internally
- explanation: asking why something is the case (reasoning or motivation)
- hypothetical: asking what would happen in a changed scenario
- analysis: asking about trade-offs, complexity, implications, or performance

Rules:
- Return 1 or 2 intents only
- If multiple intents exist, return the TOP 2 most relevant
- Output must be comma-separated (no extra text)

Examples:

Query: Why does the Transformer remove recurrence?
Answer: explanation

Query: How does attention work step by step?
Answer: mechanism

Query: What happens if positional encoding is removed?
Answer: hypothetical

Query: Why is O(n^2) complexity a problem for long sequences?
Answer: analysis

Query: How is multi-head attention different from single-head?
Answer: comparison

Query: How does attention work and why is scaling needed?
Answer: mechanism, explanation

---

Query: {query}

Answer:
"""


def detect_intent(query: str):
    """
    Returns 1 or 2 intents as a list.
    Falls back safely if parsing fails.
    """

    prompt = INTENT_PROMPT.format(query=query)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0
    )

    raw = response.choices[0].message.content.strip().lower()

    valid_intents = {
        "factual",
        "summarization",
        "critique",
        "comparison",
        "mechanism",
        "explanation",
        "hypothetical",
        "analysis"
    }

    # Split by comma
    predicted = [i.strip() for i in raw.split(",")]

    # Keep only valid intents
    filtered = [i for i in predicted if i in valid_intents]

    if len(filtered) == 0:
        return ["factual"]

    # Return max 2 intents
    return filtered[:2]