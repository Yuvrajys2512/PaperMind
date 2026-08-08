/* Plain-language explanation for each metric, surfaced on hover by
   MetricRing and reused by ChatPage to build per-answer variants.
   Keyed by the lowercased label so both the live and replayed renders share
   one source of truth. Mirrors the math in ingestion/evaluator.py.

   Lives apart from MetricRing.jsx so that file only exports a component —
   react-refresh cannot hot-reload a module that mixes the two. */
export const METRIC_TOOLTIPS = {
  confidence: {
    title: 'Confidence',
    body: 'Overall trust score for this answer. Blends the two metrics below: faithfulness × 0.7 + relevancy × 0.3. Weighted toward grounding, so a well-sourced answer scores high even if relevancy is modest.',
  },
  faithfulness: {
    title: 'Faithfulness',
    body: 'How much of the answer is actually backed by the retrieved paper text. Each answer sentence is matched against the source chunks; the score is the fraction that closely match. Higher = lower hallucination risk.',
  },
  relevancy: {
    title: 'Relevancy',
    body: 'How directly the answer addresses your question (similarity between question and answer). Naturally lands lower than faithfulness — a short question and a long answer rarely match perfectly — so read it relative to other answers, not as a pass/fail.',
  },
  retrieval: {
    title: 'Retrieval',
    body: 'How relevant the evidence pulled from the paper was, before any answer was written — how closely the retrieved passages match your question. The missing piece the others can\'t see: a high-faithfulness answer can still be faithful to weak evidence, so a low score here (especially with few strongly-relevant passages) is your warning sign.',
  },
  numbers: {
    title: 'Numbers',
    body: 'Of the figures in the answer (scores, p-values, dataset sizes), how many actually appear in the paper. Numbers are where hallucination hurts most and where the other metrics are weakest — a 0.9 faithfulness won\'t notice 28.4 quietly becoming 82.4.',
  },
}
