from ingestion.intent_detector import detect_intent

test_queries = [
    "What optimizer was used for training?",
    "What are the limitations of this model?",
    "Summarize the attention mechanism section",
    "How does the Transformer compare to RNNs?",
    "What is the BLEU score on WMT 2014?",
    "What are the weaknesses of this approach?",
    "Give me an overview of the model architecture",
    "How is multi-head attention different from single-head attention?"
]

print(f"{'Query':<55} {'Intent'}")
print("-" * 70)
for q in test_queries:
    intent = detect_intent(q)
    print(f"{q:<55} {intent}")