import sys
import os
sys.path.insert(0, r"c:\Users\Yuvraj Srivastava\Desktop\Projects\PaperMind")

from ingestion.pipeline import answer_query

PAPER = "rag"

questions = [
    "What is the difference between RAG-Sequence and RAG-Token models?",
    "How is the retriever component DPR initialized?",
    "What is the effect of the number of retrieved documents on Jeopardy question generation?",
    "How do RAG models prevent hallucination compared to standard parametric models?",
    "What are the two types of knowledge representation combined in RAG?"
]

print("=" * 80)
print(f"Testing Phase 2 & 3 on Paper: {PAPER}")
print("=" * 80)

for i, q in enumerate(questions):
    print(f"\n[Q{i+1}] {q}")
    print("-" * 80)
    
    result = answer_query(q, PAPER)
    
    print(f"Passed:      {result['passed']}")
    print(f"Confidence:  {result['confidence']:.1f}/100")
    print(f"Attempts:    {result['attempts']}")
    if result.get("warning"):
        print(f"Warning:     {result['warning']}")
    print(f"\nAnswer:\n{result['answer']}")
    print("-" * 80)
