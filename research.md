# PaperMind — Research Paper Potential

## Honest Assessment

There is a real kernel of publishable work here, but not in its current form.

---

## Genuine Research Contributions

**1. Sentence-level evidence grading + confidence-gated retry with failure diagnosis**

This is the most novel piece. It is not just engineering — it is a testable claim:
> *"Grading and removing unsupported sentences before returning the answer improves faithfulness without sacrificing relevance."*

If supported with numbers, this is publishable.

**2. Query-type-aware adaptive retrieval**

Different retrieval strategies for factual, causal, comparative, hypothetical, and summarization queries. This is a solid contribution if ablation numbers prove each adaptation matters independently.

---

## What Is Missing for Publication

| Gap | What Is Needed |
|-----|---------------|
| Evaluation against baselines | Run vanilla RAG, full PaperMind, and ablated variants on a standard benchmark (QuALITY, QASPER, or NarrativeQA) |
| A sharp research question | "Sentence-level evidence grading reduces hallucination in academic paper QA by X% while retaining Y% answer completeness" — not "we built a good RAG system" |
| Ablation study | Ablation flags already exist in the codebase — need to actually run them and report numbers |
| Dataset or annotation | 200–300 human-annotated QA pairs on real papers would significantly strengthen evaluation |

---

## Realistic Venue Targets

- **Workshop papers at ACL / EMNLP / NAACL** — 6–8 pages, lower bar, good first submission
- **ECIR or SIGIR** — if framed as an IR paper about retrieval quality
- **arXiv preprint first** — establish priority while polishing for a full venue

---

## Sharp Research Question (Draft)

> *"Does sentence-level evidence grading with confidence-gated retry improve answer faithfulness in academic paper question answering, and at what cost to completeness and latency?"*

---

## Next Steps

1. Define evaluation benchmarks (QASPER is the most directly relevant for academic paper QA)
2. Run ablation experiments using existing flags — at minimum: no evidence grading, no reranking, no HyDE, no retry
3. Annotate a small eval set of 200–300 QA pairs across 10–15 real papers
4. Write up results and target a workshop at ACL 2026 or EMNLP 2026

---

## Status

~60% of a paper exists. The system is built and the ideas are defensible. What is missing is the evaluation that turns "we think this works" into "here is evidence it works."
