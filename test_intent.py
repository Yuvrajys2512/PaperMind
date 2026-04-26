from ingestion.intent_detector import detect_intent

test_queries = [
    "Why did the authors choose to remove recurrence and convolution entirely in the Transformer?",
    "How exactly is scaled dot-product attention computed, and why is the scaling factor necessary?",
    "What would happen if positional encoding was removed from the Transformer?",
    "Why does self-attention reduce the path length between tokens, and why does that matter?",
    "How is multi-head attention different from single-head attention in terms of learning capability?",
    "Why does the Transformer have O(n²) complexity, and when could this become a problem?",
    "Why does label smoothing improve BLEU score but worsen perplexity?",
    "Why does the decoder use masking in self-attention, and what would break without it?"
]

print(f"{'Query':<55} {'Intent'}")
print("-" * 70)
for q in test_queries:
    intent = detect_intent(q)
    print(q, intent)