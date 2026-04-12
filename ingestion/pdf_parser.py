import pdfplumber
import re


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


def extract_text_from_pdf(pdf_path: str) -> dict:
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, start=1):

            # Keep only upright characters — filters rotated watermarks
            upright_chars = [c for c in page.chars if c.get("upright")]

            if not upright_chars:
                print(f"Warning: page {page_num} had no upright text")
                continue

            # Group characters into lines by their y-coordinate
            line_map = {}
            for c in upright_chars:
                y = round(c["y0"])
                if y not in line_map:
                    line_map[y] = []
                line_map[y].append(c)

            # Sort lines top to bottom, reconstruct text with proper spacing
            sorted_ys = sorted(line_map.keys(), reverse=True)
            page_lines = []

            for y in sorted_ys:
                line_text = chars_to_text(line_map[y])
                if line_text.strip():
                    page_lines.append(line_text.strip())

            page_text = "\n".join(page_lines)
            pages.append({"page_num": page_num, "text": page_text})

    full_text = "\n\n".join([p["text"] for p in pages])

    return {
        "pages": pages,
        "full_text": full_text,
        "total_pages": total_pages
    }


def remove_credits_block(full_text: str) -> str:
    """
    Everything before 'Abstract' is author credits — noise for retrieval.
    Find the word Abstract as a standalone line and cut there.
    """
    match = re.search(r'\nAbstract\n', full_text, re.IGNORECASE)
    if match:
        return full_text[match.start():].strip()
    return full_text


def remove_references_section(full_text: str) -> str:
    match = re.search(r'\nReferences\b', full_text, re.IGNORECASE)
    if match:
        return full_text[:match.start()].strip()
    return full_text