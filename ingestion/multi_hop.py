"""
ingestion/multi_hop.py

Decomposes a complex query into 2-3 focused sub-questions,
then runs retrieval for each and merges the results.

This ensures the retriever pulls chunks from multiple relevant
sections of the paper rather than clustering around one.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv
from ingestion.hybrid_retriever import hybrid_retrieve

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DECOMPOSE_SYSTEM_PROMPT = """You are a query decomposition assistant for a research paper Q&A system.

Your job: break a complex question into 2-3 focused sub-questions that together cover everything needed to answer the original question.

Rules:
- Each sub-question must target a DIFFERENT aspect or concept
- Sub-questions must be specific enough to retrieve precise passages
- Use technical terms from the domain — don't paraphrase into vague language
- Return ONLY a JSON array of strings, nothing else
- Maximum 3 sub-questions. Minimum 2.

Example:
Question: "Why can the Transformer be trained faster than RNN-based models?"
Output: ["How do RNNs process sequences sequentially and what limits their parallelization?", "How does self-attention connect positions with constant operations?", "What training times and hardware did the Transformer use?"]"""


def decompose_query(query: str) -> list[str]:
    """
    Calls the LLM to break a complex query into 2-3 sub-questions.
    Falls back to [query] if decomposition fails for any reason.
    """
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}"}
            ],
            max_tokens=200,
            temperature=0.1
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()

        sub_questions = json.loads(raw)

        # Validate — must be a list of strings
        if (
            isinstance(sub_questions, list)
            and len(sub_questions) >= 2
            and all(isinstance(q, str) for q in sub_questions)
        ):
            print(f"[multi_hop] Decomposed into {len(sub_questions)} sub-questions:")
            for i, q in enumerate(sub_questions, 1):
                print(f"  {i}. {q}")
            return sub_questions

        # If structure is wrong, fall back
        print("[multi_hop] Decomposition returned unexpected structure, falling back.")
        return [query]

    except Exception as e:
        print(f"[multi_hop] Decomposition failed ({e}), falling back to original query.")
        return [query]


def multi_hop_retrieve(query: str, paper_name: str, retrieval_k: int) -> list:
    """
    Decomposes the query, retrieves for each sub-question,
    merges results, and deduplicates by chunk id.

    Returns a merged list of unique chunks ready for reranking.
    The original query is always included as one retrieval pass
    to ensure we don't miss direct matches.
    """
    sub_questions = decompose_query(query)

    # Always include the original query as a retrieval pass
    all_queries = [query] + [q for q in sub_questions if q != query]

    seen_ids = set()
    merged_chunks = []

    for q in all_queries:
        results = hybrid_retrieve(q, paper_name, top_k=retrieval_k)
        for chunk in results:
            # Use chunk_id from metadata as dedup key
            chunk_id = chunk["metadata"].get("chunk_id")
            if chunk_id is None:
                # Fall back to text hash if no chunk_id
                chunk_id = hash(chunk["text"])
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged_chunks.append(chunk)

    print(f"[multi_hop] Retrieved {len(merged_chunks)} unique chunks across {len(all_queries)} queries.")
    return merged_chunks