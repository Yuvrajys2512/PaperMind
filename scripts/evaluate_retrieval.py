from ingestion.retriever import retrieve
from ingestion.hybrid_retriever import hybrid_retrieve
from ingestion.reranker import rerank

PAPER = "attention-is-all-you-need"

EVAL_QUESTIONS = [
    # EXACT TERM queries — BM25 should help here
    {"query": "What is the value of dropout rate used?", "expected_section": "5.4 Regularization", "expected_page": 8},
    {"query": "What optimizer was used with beta1 beta2 epsilon?", "expected_section": "5.3 Optimizer", "expected_page": 7},
    {"query": "What is the BLEU score on WMT 2014 English to German?", "expected_section": "6.1 Machine Translation", "expected_page": 8},
    {"query": "How many attention heads are used?", "expected_section": "3.2.2 Multi-Head Attention", "expected_page": 5},
    {"query": "What is the dimensionality of the model dmodel?", "expected_section": "3.2.2 Multi-Head Attention", "expected_page": 4},

    # SEMANTIC queries — vector should help here
    {"query": "How does the model handle word order?", "expected_section": "3.5 Positional Encoding", "expected_page": 6},
    {"query": "Why did the authors move away from recurrent models?", "expected_section": "1 Introduction", "expected_page": 2},
    {"query": "How is relevance computed between tokens?", "expected_section": "3.2.1 Scaled Dot-Product Attention", "expected_page": 4},
    {"query": "What prevents the model from being too confident?", "expected_section": "5.4 Regularization", "expected_page": 8},
    {"query": "How does the decoder prevent attending to future positions?", "expected_section": "3.2.3 Applications of Attention", "expected_page": 5},

    # BOTH needed — hybrid advantage
    {"query": "What warmup steps were used in learning rate scheduling?", "expected_section": "5.3 Optimizer", "expected_page": 7},
    {"query": "How does multi-head attention differ from single attention?", "expected_section": "3.2.2 Multi-Head Attention", "expected_page": 5},
    {"query": "What is label smoothing and what value was used?", "expected_section": "5.4 Regularization", "expected_page": 8},
    {"query": "What feed-forward network size was used in each layer?", "expected_section": "3.3 Position-wise Feed-Forward Networks", "expected_page": 5},
    {"query": "How were the English to French translation results?", "expected_section": "6.1 Machine Translation", "expected_page": 8},

    # HARD queries — vague or indirect
    {"query": "What makes this architecture faster to train?", "expected_section": "4 Why Self-Attention", "expected_page": 6},
    {"query": "How does the model generalize beyond translation?", "expected_section": "6.3 English Constituency Parsing", "expected_page": 9},
    {"query": "What are the encoder and decoder made of?", "expected_section": "3.1 Encoder and Decoder Stacks", "expected_page": 3},
    {"query": "How is attention scaled and why?", "expected_section": "3.2.1 Scaled Dot-Product Attention", "expected_page": 4},
    {"query": "What training data was used?", "expected_section": "5.1 Training Data and Batching", "expected_page": 7},
]


def is_hit(results, expected_section, expected_page):
    for r in results:
        section_match = expected_section.lower() in r["metadata"]["section"].lower()
        page_match = r["metadata"]["page_num"] == expected_page
        if section_match or page_match:
            return True
    return False


def get_reciprocal_rank(results, expected_section, expected_page):
    for i, r in enumerate(results):
        section_match = expected_section.lower() in r["metadata"]["section"].lower()
        page_match = r["metadata"]["page_num"] == expected_page
        if section_match or page_match:
            return 1 / (i + 1)
    return 0


def evaluate(pipeline_name, retrieve_fn, questions):
    hits = 0
    rr_sum = 0

    print(f"\n{'='*70}")
    print(f"PIPELINE: {pipeline_name}")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Hit':<6} {'RR':<8} {'Query':<45} Expected Section")
    print("-" * 70)

    for i, q in enumerate(questions):
        results = retrieve_fn(q["query"], PAPER, top_k=5)
        hit = is_hit(results, q["expected_section"], q["expected_page"])
        rr = get_reciprocal_rank(results, q["expected_section"], q["expected_page"])

        hits += int(hit)
        rr_sum += rr

        hit_str = "✓" if hit else "✗"
        print(f"{i+1:<4} {hit_str:<6} {rr:.3f}   {q['query'][:43]:<45} {q['expected_section']}")

    hit_rate = hits / len(questions)
    mrr = rr_sum / len(questions)

    print(f"\nHit@5 : {hits}/{len(questions)} = {hit_rate:.2%}")
    print(f"MRR   : {mrr:.4f}")

    return hit_rate, mrr


def hybrid_retrieve_with_rerank(query, paper_name, top_k=5):
    rrf_results = hybrid_retrieve(query, paper_name, top_k=10)
    reranked = rerank(query, rrf_results, top_k=top_k)
    return reranked


# Run evaluation
naive_hit, naive_mrr = evaluate(
    "NAIVE VECTOR ONLY",
    retrieve,
    EVAL_QUESTIONS
)

hybrid_hit, hybrid_mrr = evaluate(
    "HYBRID (BM25 + RRF + CrossEncoder)",
    hybrid_retrieve_with_rerank,
    EVAL_QUESTIONS
)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"{'Metric':<15} {'Naive':>10} {'Hybrid':>10} {'Delta':>10}")
print("-" * 45)
print(f"{'Hit@5':<15} {naive_hit:>10.2%} {hybrid_hit:>10.2%} {hybrid_hit - naive_hit:>+10.2%}")
print(f"{'MRR':<15} {naive_mrr:>10.4f} {hybrid_mrr:>10.4f} {hybrid_mrr - naive_mrr:>+10.4f}")