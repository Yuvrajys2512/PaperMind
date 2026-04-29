# import re

# def score_candidate(line_text: str, line_chars: list, 
#                     body_size: float, prev_line_y: float) -> int:
#     """
#     Scores a line against all heading signals.
#     Returns total score — if >= 5, treat as candidate.
#     """
#     score = 0

#     # -------- Signal 1 — font size --------
#     if line_chars:
#         avg_size = sum(c["size"] for c in line_chars) / len(line_chars)
#         if avg_size > body_size + 1.5:
#             score += 3

#     # -------- Signal 2 — word count --------
#     words = line_text.strip().split()
#     if 0 < len(words) <= 8:
#         score += 2

#     # -------- Signal 3 — does not end with period --------
#     if not line_text.strip().endswith((".", "!", "?")):
#         score += 2

#     # -------- Signal 4 — numeric prefix --------
#     # Matches: 1., 1.1, 2.3.4, etc.
#     if re.match(r'^\d+(\.\d+)*\s+', line_text.strip()):
#         score += 2

#     # -------- Signal 5 — capitalisation --------
#     stripped = line_text.strip()
#     if stripped.istitle() or stripped.isupper():
#         score += 1

#     # -------- Signal 6 — vertical gap --------
#     if prev_line_y is not None and line_chars:
#         current_y = max(c["y1"] for c in line_chars)  # top of line
#         gap = prev_line_y - current_y  # pdf: higher y = higher on page

#         if gap > 10:
#             score += 2

#     return score


import re
import os
from collections import Counter
from dotenv import load_dotenv
from groq import Groq

from ingestion.pdf_parser import build_page_lines

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chars_to_text(chars: list) -> str:
    if not chars:
        return ""
    chars = sorted(chars, key=lambda c: c["x0"])
    result = chars[0]["text"]
    for i in range(1, len(chars)):
        gap = chars[i]["x0"] - chars[i - 1]["x1"]
        if gap > 2:
            result += " "
        result += chars[i]["text"]
    return result


def get_body_font_size(chars: list) -> float:
    sizes = [round(c["size"]) for c in chars if c.get("size")]
    if not sizes:
        return 10.0
    return Counter(sizes).most_common(1)[0][0]


def score_candidate(line_text: str, line_chars: list,
                    body_size: float, prev_line_y: float) -> int:
    score = 0

    if line_chars:
        avg_size = sum(c["size"] for c in line_chars) / len(line_chars)
        if avg_size > body_size + 1.5:
            score += 3

    words = line_text.strip().split()
    if 0 < len(words) <= 8:
        score += 2

    if not line_text.strip().endswith((".", ",", ":", ";", "!", "?")):
        score += 2

    if re.match(r'^\d+(\.\d+)*\s+', line_text.strip()):
        score += 2

    stripped = line_text.strip()
    if stripped.istitle() or stripped.isupper():
        score += 1

    if prev_line_y is not None and line_chars:
        current_y = max(c["y1"] for c in line_chars)
        gap = prev_line_y - current_y
        if gap > 10:
            score += 2

    return score


def build_candidates(pages: list) -> list:
    candidates = []
    candidate_id = 1

    for page in pages:
        chars = page["chars"]
        page_num = page["page_num"]
        page_width = page.get("page_width", 600)  # fallback just in case
        body_size = get_body_font_size(chars)

        lines = build_page_lines(chars, page_width)

        prev_line_y = None

        for i, (line_text, line_chars) in enumerate(lines):
            score = score_candidate(
                line_text, line_chars, body_size, prev_line_y
            )

            if score >= 8:
                context_lines = [
                    lines[j][0]
                    for j in range(i + 1, min(i + 4, len(lines)))
                ]
                candidates.append({
                    "id": candidate_id,
                    "candidate_line": line_text,
                    "context": " ".join(context_lines),
                    "page_num": page_num,
                    "line_index": i,
                    "score": score
                })
                candidate_id += 1

            if line_chars:
                prev_line_y = min(c["y0"] for c in line_chars)

    return candidates\

#CONFIRM HEADING WITH LLM CODE BLOCK

def confirm_headings_with_llm(candidates: list) -> dict:
    if not candidates:
        return {}

    # Build the prompt
    prompt_lines = []
    for c in candidates:
        line = f"{c['id']} | Candidate: \"{c['candidate_line']}\" | Context: \"{c['context']}\""
        prompt_lines.append(line)

    candidates_text = "\n".join(prompt_lines)

    prompt = f"""You are analyzing candidate section headings from an academic research paper.

For each numbered candidate below, I provide the candidate line and the
lines immediately following it as context.

Respond with ONLY this format, one line per candidate:
ID | SECTION or SUBSECTION or NONE

Rules:
- SECTION = a major section heading (Introduction, Methodology, Results etc.)
- SUBSECTION = a numbered or titled subsection (3.1 Encoder, 5.2 Hardware etc.)
- NONE = not a heading (figure label, table content, math expression etc.)

Candidates:
{candidates_text}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.choices[0].message.content

    results = {}
    for line in raw.strip().split("\n"):
        parts = line.strip().split("|")
        if len(parts) >= 2:
            try:
                candidate_id = int(parts[0].strip())
                verdict = parts[1].strip().upper()
                if verdict in ("SECTION", "SUBSECTION", "NONE"):
                    results[candidate_id] = verdict
            except ValueError:
                continue

    return results




#ASSEMBLER SECTION
def assemble_sections(pages: list, confirmed: dict, candidates: list) -> list:
    """
    Walk every line of every page in order. When a line matches a confirmed
    heading (SECTION or SUBSECTION), open a new bucket. Pour subsequent lines
    into the active bucket until the next heading or "References".

    Returns a list of dicts:
        {
            "heading":  str,            # the heading line text
            "type":     "SECTION" | "SUBSECTION",
            "page_num": int,
            "body":     str,            # all content lines joined by newline
        }
    """
    # Build a fast lookup: (page_num, line_index) → (verdict, heading_text)
    # Only keep confirmed headings — NONE entries are ignored.
    heading_lookup: dict[tuple, tuple] = {}
    for c in candidates:
        verdict = confirmed.get(c["id"])
        if verdict in ("SECTION", "SUBSECTION"):
            key = (c["page_num"], c["line_index"])
            heading_lookup[key] = (verdict, c["candidate_line"])

    sections = []
    current_heading: str | None = None
    current_type: str | None = None
    current_page: int | None = None
    current_body: list[str] = []

    def flush():
        """Save the active bucket to sections if there's an open heading."""
        if current_heading is not None:
            sections.append({
                "heading": current_heading,
                "type": current_type,
                "page_num": current_page,
                "body": "\n".join(current_body).strip(),
            })

    for page in pages:
        chars = page["chars"]
        page_num = page["page_num"]
        page_width = page.get("page_width", 600)

        # Rebuild the same line list build_candidates produced so line_index matches.
        page_lines = build_page_lines(chars, page_width)
        lines = [text for text, _ in page_lines]

        for line_index, line_text in enumerate(lines):
            stripped = line_text.strip()

            # Stop at References — we don't want bibliography content.
            if stripped.lower().startswith("references"):
                flush()
                return sections

            key = (page_num, line_index)
            if key in heading_lookup:
                flush()
                current_heading = stripped
                current_type, _ = heading_lookup[key]
                current_page = page_num
                current_body = []
            elif current_heading is not None:
                current_body.append(stripped)

    flush()
    return sections