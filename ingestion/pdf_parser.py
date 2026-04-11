import pdfplumber
import re


def extract_text_from_pdf(pdf_path: str) -> dict:
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text()

            if raw_text is None:
                print(f"Warning: page {page_num} returned no text (possibly scanned image)")
                continue

            cleaned = clean_page_text(raw_text, page_num, total_pages)

            if cleaned.strip():
                pages.append({
                    "page_num": page_num,
                    "text": cleaned
                })

    full_text = "\n\n".join([p["text"] for p in pages])

    return {
        "pages": pages,
        "full_text": full_text,
        "total_pages": total_pages
    }


def clean_page_text(text: str, page_num: int, total_pages: int) -> str:
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if re.match(r'^\d{1,3}$', stripped):
            continue

        words_in_line = stripped.split()
        if len(words_in_line) <= 3 and not stripped.endswith(('.', ':', ',')):
            continue

        cleaned_lines.append(stripped)

    return '\n'.join(cleaned_lines)


def remove_references_section(full_text: str) -> str:
    ref_patterns = [
        r'\nReferences\n',
        r'\nBibliography\n',
        r'\nREFERENCES\n',
        r'\nWorks Cited\n'
    ]

    cut_position = len(full_text)

    for pattern in ref_patterns:
        match = re.search(pattern, full_text)
        if match:
            cut_position = min(cut_position, match.start())

    return full_text[:cut_position]