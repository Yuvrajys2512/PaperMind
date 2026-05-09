Sub tast 1.1
PDF Parsing — pdfplumber
Used pdfplumber for text extraction. Key decisions: filter rotated characters using the upright property to remove arXiv watermarks. Reconstruct word spacing manually by measuring gaps between character x-coordinates — PDFs don't store space characters, only positional gaps. Remove credits block by finding "Abstract" heading. Remove references by finding "References" heading. Known limitations: mathematical formulas, superscripts, and tables are not cleanly extracted — to be addressed with specialist modules later.



API setup
LLM Layer — Unified rotating client (llm_client.py)
All LLM calls go through ingestion/llm_client.py which rotates across providers in priority order: Gemini Flash (1M tokens/day, OpenAI-compatible via generativelanguage.googleapis.com/v1beta/openai/) → Cerebras (llama3.3-70b) → Mistral Small → Groq-1 (llama-3.3-70b-versatile). Rate-limit and quota errors trigger automatic fallback to the next provider. Unexpected errors surface immediately. This means no single provider outage can crash the pipeline. All providers use the standard OpenAI chat completions interface — switching models requires one line change per provider entry.

Chunk size: 512 tokens, overlap: 100 tokens. Tested all six combinations on "Attention Is All You Need". 256 over-fragments short sections. 1024 under-splits large sections reducing retrieval precision. 512/100 produces 27 chunks — short sections fit cleanly in one chunk, large sections get 2-3 chunks. 100 token overlap chosen over 50 because academic sentences average 20-30 tokens — 100 tokens captures 3-4 complete sentences of shared context at boundaries.