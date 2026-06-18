# PaperMind

### Ask questions about any research paper. Get answers you can verify.

A research assistant that reads papers *with* you — and shows its proof.

---

## The Problem

- Research papers are long, dense, and full of jargon.
- Students and researchers waste hours hunting for a single answer.
- People turn to ChatGPT — but it **makes things up** and even **invents citations**.
- The real challenge isn't getting *an* answer. It's getting an answer you can **trust**.

---

## What PaperMind Is

- **Upload a PDF → Ask anything → Get a cited, verifiable answer.**
- Every claim links to the **exact section** it came from (e.g. "§6.2").
- Each answer shows a **confidence score** and a **grounding check**.
- Works on preprints, published papers, and technical reports.

---

## Why It's Better Than ChatGPT

| | General chatbot (ChatGPT) | PaperMind |
|---|---|---|
| Source of the answer | Its general memory | **Your actual uploaded paper** |
| Citations | Often invented | **Real — links to the exact section** |
| Made-up answers | Common | **Checked and graded for grounding** |
| Long papers | Can lose track | **Indexed and searched precisely** |
| Trust | "Take my word for it" | **"Here's the proof"** |

---

## How It Works

1. **Ingest** — split the paper into chunks and turn each into a searchable "fingerprint."
2. **Retrieve** — find the most relevant chunks for your question (keyword + meaning-based search, combined).
3. **Re-rank** — a second model re-sorts them so the best passages come first.
4. **Answer** — the AI writes a response using *only* those passages.
5. **Grade** — check every sentence is actually supported, and score the confidence.

---

## What Makes It Trustworthy

- **Evidence grading** — each sentence is checked against the source passages.
- **Faithfulness score** — how much of the answer is genuinely backed by the paper.
- **Confidence %** — shown on every single answer.
- **Retry engine** — weak answers are automatically re-attempted with better evidence.
- **No invented citations — ever.**

---

## Beyond Basic Q&A

- **Multi-hop questions** — answers that connect several parts of the paper.
- **Compare two papers** side by side.
- **Auto-glossary** — plain-English definitions of the paper's jargon.
- **Related-paper recommendations.**
- **Rewrite tool** — rephrase text as academic, plain, or concise.

---

## Proving It Actually Works

- We don't just *claim* accuracy — we **measure** it.
- Built an evaluation harness on **QASPER**, a standard benchmark of expert Q&A over research papers.
- We score answers automatically and run **ablations** — turning features on and off to see what truly helps.
- **"Answers benchmarked on QASPER"** — a credibility line most competitors don't have.

---

## From Demo to Real Product

- **Accounts & login** — every user sees only their own papers.
- **Cloud storage** — papers stay safe across restarts.
- **Cost tracking & free-tier limits** — we measure what each query actually costs.
- **Payments** — a $9/month Pro plan via Stripe.
- **Deployment & monitoring** — packaged to go live, with error tracking and analytics built in.

---

## Progress & What's Next

**Done**
- Core verifiable-answer engine
- Advanced features (multi-hop, compare, glossary, rewrite)
- Evaluation harness
- Accounts, storage, cost tracking, billing, deployment prep, monitoring

**Next**
- Sample papers for instant first use
- Landing page
- Terms & privacy → then **go live on our own domain**

> A research assistant you can actually trust — because it shows its proof.
