Sub tast 1.1
PDF Parsing — pdfplumber
Used pdfplumber for text extraction. Key decisions: filter rotated characters using the upright property to remove arXiv watermarks. Reconstruct word spacing manually by measuring gaps between character x-coordinates — PDFs don't store space characters, only positional gaps. Remove credits block by finding "Abstract" heading. Remove references by finding "References" heading. Known limitations: mathematical formulas, superscripts, and tables are not cleanly extracted — to be addressed with specialist modules later.



API setup
LLM API — Groq
Using Groq API with llama-3.3-70b-versatile model for the section detection layer. Chose Groq over Gemini because Gemini's free tier quota is too restrictive for active development (exhausted within a few retry attempts). Groq's free tier is generous enough for the entire build phase. The API uses the OpenAI-compatible chat completions format — a messages list with role and content. This is the industry standard pattern, meaning switching to OpenAI or another provider later requires changing one line. Will revisit model choice when moving to production.

Chunk size: 512 tokens, overlap: 100 tokens. Tested all six combinations on "Attention Is All You Need". 256 over-fragments short sections. 1024 under-splits large sections reducing retrieval precision. 512/100 produces 27 chunks — short sections fit cleanly in one chunk, large sections get 2-3 chunks. 100 token overlap chosen over 50 because academic sentences average 20-30 tokens — 100 tokens captures 3-4 complete sentences of shared context at boundaries.