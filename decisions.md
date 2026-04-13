Sub tast 1.1
PDF Parsing — pdfplumber
Used pdfplumber for text extraction. Key decisions: filter rotated characters using the upright property to remove arXiv watermarks. Reconstruct word spacing manually by measuring gaps between character x-coordinates — PDFs don't store space characters, only positional gaps. Remove credits block by finding "Abstract" heading. Remove references by finding "References" heading. Known limitations: mathematical formulas, superscripts, and tables are not cleanly extracted — to be addressed with specialist modules later.



API setup
LLM API — Groq
Using Groq API with llama-3.3-70b-versatile model for the section detection layer. Chose Groq over Gemini because Gemini's free tier quota is too restrictive for active development (exhausted within a few retry attempts). Groq's free tier is generous enough for the entire build phase. The API uses the OpenAI-compatible chat completions format — a messages list with role and content. This is the industry standard pattern, meaning switching to OpenAI or another provider later requires changing one line. Will revisit model choice when moving to production.

