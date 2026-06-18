# PaperMind — Mentor Presentation (10 slides)

> **How to use this:** Paste this whole file into Gamma (gamma.app) → "Create with AI / Import text" → it will turn each `---`-separated block into a slide. Each slide has **🖥️ On the slide** (the text/bullets to show) and **🗣️ What to say** (your spoken explanation — don't put this on the slide, just say it). Suggested look in Gamma: dark theme, one accent colour (teal/cyan), clean sans-serif. Where it says *[suggest a visual]*, add a screenshot or simple diagram.

---

## Slide 1 — Title

🖥️ **On the slide:**
- **PaperMind**
- *Ask questions about any research paper. Get answers you can verify.*
- Your name(s) · Mentor presentation · [date]
- *[suggest a visual: the app's upload screen or logo]*

🗣️ **What to say:**
"Good morning. Our project is called PaperMind. In one sentence: it lets you upload any research paper as a PDF and ask questions about it in plain English — and unlike a normal chatbot, every answer points back to the exact place in the paper it came from, so you can trust it. Over the next few minutes I'll explain the idea, why it's different from something like ChatGPT, how it works under the hood, how we proved it actually works, and how far we've taken it toward being a real product."

---

## Slide 2 — The problem & our inspiration

🖥️ **On the slide:**
- Research papers are long, dense, and full of jargon.
- Students & researchers waste hours hunting for one answer.
- People already ask ChatGPT — but it **makes things up** ("hallucinates") and **invents citations**.
- The real problem isn't getting *an* answer — it's getting an answer you can **trust**.
- *[suggest a visual: a stack of papers / a confused student icon]*

🗣️ **What to say:**
"The inspiration came from a frustration we all share. Reading research papers is slow and hard — they're long, technical, and full of jargon. Naturally, people now paste them into ChatGPT and ask questions. But that creates a worse problem: general chatbots confidently make up facts, and even invent fake citations and page numbers. For a student or researcher, a confident wrong answer is dangerous. So we realised the goal isn't just to *answer* questions about a paper — it's to answer them in a way the reader can actually verify. That trust gap is the problem PaperMind solves."

---

## Slide 3 — What PaperMind is

🖥️ **On the slide:**
- **Upload a PDF → Ask anything → Get a cited, verifiable answer.**
- Every claim links to the exact section it came from (e.g. "§6.2").
- Each answer shows a **confidence score** and a **grounding check**.
- Works on preprints, published papers, and reports.
- *[suggest a visual: screenshot of a chat answer with a "§6.2" citation chip + confidence %]*

🗣️ **What to say:**
"Here's what it actually does. You drag in a paper, and then you just chat with it. When it answers, it doesn't only give you the text — it attaches the exact section the answer is based on, and a confidence score telling you how sure it is. So if it says 'the model reached 28.4 BLEU,' you can click straight to section 6.2 and check. The whole product is built around that idea: an answer you can verify in one click, not one you have to take on faith."

---

## Slide 4 — Why it's better than ChatGPT

🖥️ **On the slide:** *(make this a 2-column comparison table)*

| | **General chatbot (ChatGPT)** | **PaperMind** |
|---|---|---|
| Source of answer | Its general memory | **Your actual uploaded paper** |
| Citations | Often invented | **Real — links to the exact section** |
| Made-up answers | Common | **Checked & graded for grounding** |
| Long papers | Can lose track | **Indexed & searched precisely** |
| Trust | "Take my word" | **"Here's the proof"** |

🗣️ **What to say:**
"This is the heart of the pitch. A general chatbot answers from its memory of the whole internet, so it can drift, hallucinate, and cite things that don't exist. PaperMind only answers from the specific paper you gave it — and it shows its work. It searches the actual document, grounds each answer in real passages, checks itself, and links to the exact section. So it's not competing with ChatGPT on being a know-it-all; it's better at the one thing that matters here: being trustworthy about *this* paper."

---

## Slide 5 — How it works (the simple version)

🖥️ **On the slide:**
- **1. Ingest** — split the paper into chunks, turn each into a searchable "fingerprint."
- **2. Retrieve** — for your question, find the most relevant chunks (keyword search + meaning-based search, combined).
- **3. Re-rank** — a second model re-sorts them so the *best* passages come first.
- **4. Answer** — the AI writes an answer using only those passages.
- **5. Grade** — check every sentence is actually supported, score the confidence.
- *[suggest a visual: a 5-step left-to-right pipeline diagram]*

🗣️ **What to say:**
"Here's the engine, in plain terms — it's a technique called RAG, retrieval-augmented generation. First we break the paper into small chunks and convert each into a numeric 'fingerprint' that captures its meaning. When you ask a question, we search those chunks two ways at once — classic keyword matching *and* meaning-based matching — and combine the results. Then a second AI model re-ranks them so the most relevant passages float to the top. Only then does the language model write an answer, and it's only allowed to use those retrieved passages. Finally we grade the answer — checking each sentence is really supported by the source. That last step is what makes it verifiable instead of just plausible. We don't need to go deeper than that today, but that's the full loop."

---

## Slide 6 — The part that makes it trustworthy

🖥️ **On the slide:**
- **Evidence grading** — each sentence is checked against the source passages.
- **Faithfulness score** — how much of the answer is actually backed by the paper.
- **Confidence %** — shown on every answer.
- **Retry engine** — if an answer is weak, the system tries again with better evidence.
- *No invented citations — ever.*
- *[suggest a visual: an answer card showing "grounded in source ✓ · confidence 94%"]*

🗣️ **What to say:**
"This slide is the differentiator, so I'll dwell on it. After the AI writes an answer, we don't just hand it over. We re-check it against the source: does each sentence actually have evidence behind it? We turn that into a faithfulness score and a confidence percentage that the user sees. And if an answer comes back weak or poorly supported, the system automatically retries with better evidence before showing it. The result is that PaperMind would rather say 'I'm not confident' than make something up — which is exactly the behaviour you want from a research tool."

---

## Slide 7 — Beyond basic Q&A

🖥️ **On the slide:**
- **Multi-hop questions** — answers that need connecting several parts of the paper.
- **Compare two papers** side by side.
- **Auto-glossary** — plain-English definitions of the paper's jargon.
- **Related-paper recommendations.**
- **Rewrite tool** — rephrase text as academic / plain / concise.
- *[suggest a visual: small icons for each feature]*

🗣️ **What to say:**
"On top of the core Q&A, we built features that go beyond what a simple chatbot does. It can answer 'multi-hop' questions — ones where the answer is spread across several parts of the paper and has to be stitched together. It can compare two papers side by side. It auto-generates a glossary that explains the paper's jargon in plain English, recommends related papers, and can rewrite passages in a simpler or more formal style. These make it a genuine research companion, not just a search box."

---

## Slide 8 — Proving it actually works (our eval harness)

🖥️ **On the slide:**
- We don't just *claim* it's accurate — we **measure** it.
- Built an evaluation harness on **QASPER** — a standard benchmark of expert Q&A over research papers.
- We score our answers automatically and run **ablations** (turn features on/off to see what helps).
- *"Answers benchmarked on QASPER"* — a credibility line most competitors don't have.
- *[suggest a visual: a simple bar chart / "benchmark" badge]*

🗣️ **What to say:**
"A big part of our work was making sure we could *prove* quality, not just assert it. We built an evaluation harness around QASPER, which is a well-known academic benchmark — a large set of real questions about research papers with expert-written answers. We run PaperMind against it and score the answers automatically, and we do 'ablation' experiments — switching parts of our pipeline on and off to measure which ones actually improve answers. This is genuinely rare for student projects and even for commercial 'chat-with-PDF' tools, and it's something we're proud of: we can put a number on how good the system is."

---

## Slide 9 — From demo to real product

🖥️ **On the slide:**
- Turned a single-user demo into a **multi-user product**:
  - **Accounts & login** (so users only see their own papers)
  - **Real cloud storage** (papers safe across restarts)
  - **Cost tracking & free-tier limits** (we measure what each query costs)
  - **Payments** — a $9/month Pro plan (Stripe)
  - **Deployment + monitoring** (error tracking & analytics)
- *[suggest a visual: simple architecture diagram — Frontend ↔ Backend ↔ AI + Database]*

🗣️ **What to say:**
"We didn't stop at a tech demo. We've been turning it into something that could actually run as a real product. That meant adding proper user accounts so everyone's papers are private, moving storage to the cloud so nothing gets lost, and — importantly — tracking exactly what each question costs in AI usage, so we can offer a free tier and a paid plan sensibly. We added a $9-a-month Pro subscription through Stripe, packaged the whole thing to deploy on free hosting, and wired in error-tracking and analytics so we can run it without flying blind. So it's not just a clever algorithm — it's most of the way to a launchable service."

---

## Slide 10 — Progress & what's next

🖥️ **On the slide:**
- **Done:** core verifiable-answer engine, advanced features, evaluation harness, accounts, storage, cost tracking, billing, deployment prep, monitoring.
- **Next:** sample papers for instant first-use, a landing page, terms & privacy, then **go live on our own domain**.
- **The one-line takeaway:** *A research assistant you can actually trust — because it shows its proof.*
- *[suggest a visual: a simple roadmap / checkmarks]*

🗣️ **What to say:**
"To wrap up — here's where we are. The core engine, the advanced features, the evaluation harness, and almost all the 'real product' plumbing — accounts, storage, cost control, payments, deployment, monitoring — are built and tested. What's left is mostly polish: a few sample papers so new users get value instantly, a landing page, and the legal terms, and then we go live on our own domain. If you remember one thing, let it be this: PaperMind is a research assistant you can actually trust, because instead of just giving you an answer, it shows you the proof. Thank you — happy to take questions or give a quick demo."

---

## Optional: likely mentor questions (prep, not a slide)

- **"How is this different from ChatGPT's PDF upload?"** → We search the document with specialised retrieval, grade every answer for grounding, and cite exact sections — and we benchmark accuracy on QASPER. It's built for trust, not general chat.
- **"What if it doesn't know?"** → It's designed to show low confidence / decline rather than make something up.
- **"What did *you* build vs. use libraries?"** → We use standard building blocks (embeddings, a vector database, language-model APIs) but the pipeline, the evidence-grading/retry logic, the evaluation harness, and the whole product around it are ours.
- **"Does it cost money to run?"** → We track cost per query; it runs on free AI tiers today, with a paid plan designed in for when it scales.
- **"Is it live?"** → Code-complete and tested; final deployment to our own domain is the last step.
