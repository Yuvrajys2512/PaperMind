"""
pdf_parser.py — robust PDF text extractor for research papers.

Handles both single-column AND two-column layouts by detecting
column structure per-page using character x-coordinate gap analysis.

Public API:
    extract_text_from_pdf(pdf_path)  → dict with pages, full_text, total_pages
    remove_credits_block(text)       → str
    remove_references_section(text)  → str
"""

import pdfplumber
import re
from statistics import median


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def chars_to_text(chars: list) -> str:
    """Convert a list of character dicts on the same line into a text string,
    inserting spaces where there is a visible horizontal gap."""
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


def _group_chars_into_lines(chars: list, y_tolerance: float = 2.0) -> list:
    """Group characters into lines based on their vertical position.

    Returns a list of (y_key, [chars]) sorted top-to-bottom.
    Characters on approximately the same y are merged into one line.
    """
    if not chars:
        return []

    # Sort by y position (top of char), then left-to-right
    chars = sorted(chars, key=lambda c: (c["top"], c["x0"]))

    lines = []
    current_y = chars[0]["top"]
    current_line = [chars[0]]

    for c in chars[1:]:
        if abs(c["top"] - current_y) <= y_tolerance:
            current_line.append(c)
        else:
            lines.append((current_y, current_line))
            current_y = c["top"]
            current_line = [c]
    lines.append((current_y, current_line))

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Column detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_column_boundary(chars: list, page_width: float) -> float | None:
    """Detect whether the page has a two-column layout by looking for a
    vertical gutter (a strip of empty space near the page center).

    Returns the x-coordinate of the column split, or None if single-column.

    Strategy:
      1. Exclude characters from the top portion of the page (title area)
         because full-width titles/authors obscure the gutter.
      2. Build a histogram of horizontal character positions from the
         remaining body text.
      3. Look for a gap (valley) in the histogram near the page center.
      4. If a clear gap exists, that's the column boundary.
    """
    if not chars:
        return None

    # Determine page height from character positions
    page_top = min(c["top"] for c in chars)
    page_bottom = max(c["top"] for c in chars)
    page_height = page_bottom - page_top

    if page_height < 50:
        return None

    # Only use characters from the lower 70% of the page for detection.
    # This skips titles, author names, and conference headers that
    # typically span the full width on page 1.
    body_cutoff = page_top + page_height * 0.30
    body_chars = [c for c in chars if c["top"] >= body_cutoff]

    if len(body_chars) < 20:
        return None

    page_mid = page_width / 2
    # Only look for a gutter in the middle 30% of the page
    search_left = page_mid - page_width * 0.15
    search_right = page_mid + page_width * 0.15

    # Build coverage histogram using only body characters
    bin_count = int(page_width) + 1
    coverage = [0] * bin_count

    for c in body_chars:
        x_start = max(0, int(c["x0"]))
        x_end = min(bin_count - 1, int(c["x1"]))
        for b in range(x_start, x_end + 1):
            coverage[b] += 1

    # Look for a gap in the search zone
    search_start = max(0, int(search_left))
    search_end = min(bin_count - 1, int(search_right))

    # Find the bin with minimum coverage in the search zone
    min_coverage = float("inf")
    min_bin = None
    for b in range(search_start, search_end + 1):
        if coverage[b] < min_coverage:
            min_coverage = coverage[b]
            min_bin = b

    if min_bin is None:
        return None

    # To qualify as a gutter, the minimum must be significantly less
    # than the median coverage of the body text area
    nonzero_bins = [c for c in coverage if c > 0]
    if not nonzero_bins:
        return None
    med_coverage = median(nonzero_bins)

    # A gutter should have very low coverage (< 15% of the median)
    gap_threshold = med_coverage * 0.15

    # Find the extent of the gap around min_bin
    gap_left = min_bin
    while gap_left > search_start and coverage[gap_left - 1] <= gap_threshold:
        gap_left -= 1
    gap_right = min_bin
    while gap_right < search_end and coverage[gap_right + 1] <= gap_threshold:
        gap_right += 1

    gap_width = gap_right - gap_left
    if gap_width < 5:  # gap must be at least ~5 points wide
        return None

    # Return the midpoint of the gap as the column boundary
    return (gap_left + gap_right) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Page processing
# ─────────────────────────────────────────────────────────────────────────────

def _process_single_column(chars: list) -> str:
    """Process characters as a single-column page."""
    lines = _group_chars_into_lines(chars)
    page_lines = []
    for _, line_chars in lines:
        text = chars_to_text(line_chars)
        if text.strip():
            page_lines.append(text.strip())
    return "\n".join(page_lines)


def _process_two_column(chars: list, col_boundary: float,
                        page_width: float) -> str:
    """Process characters in a two-column layout, respecting reading order.

    Strategy: split characters into left-column, right-column, and
    full-width buckets FIRST (by x-position), then group each bucket
    into lines independently. This avoids merging characters from both
    columns that share the same y-coordinate into one garbled line.

    Reading order:
      1. Full-width lines above column content (title, authors, etc.)
      2. Left column, top to bottom
      3. Right column, top to bottom
      4. Full-width lines below column content (footnotes, page numbers)
    """
    # ── Step 1: Partition characters into column buckets ──────────────
    left_chars = []
    right_chars = []
    # Characters that sit squarely on the gutter are ambiguous;
    # we'll assign them to the nearest column.
    for c in chars:
        char_center = (c["x0"] + c["x1"]) / 2
        if char_center < col_boundary:
            left_chars.append(c)
        else:
            right_chars.append(c)

    # ── Step 2: Group each column's chars into lines independently ────
    left_lines = _group_chars_into_lines(left_chars)
    right_lines = _group_chars_into_lines(right_chars)

    # ── Step 3: Identify full-width lines ─────────────────────────────
    # A "full-width" line is one where the text actually flows THROUGH
    # the gutter (no gap at the column boundary). Two-column lines will
    # have a clear empty gap at the gutter even though they span the
    # full page width.
    # We check: does the left-side text extend into the gutter zone,
    # or does the right-side text start before the gutter zone?
    gutter_margin = 8  # points of tolerance around the column boundary

    full_width_lines = []  # (y_key, chars)
    matched_left_ys = set()
    matched_right_ys = set()

    for ly, lchars in left_lines:
        for ry, rchars in right_lines:
            if abs(ly - ry) <= 2:  # same vertical position
                # Check if left text extends close to (or past) the gutter
                left_rightmost = max(c["x1"] for c in lchars)
                # Check if right text starts close to (or before) the gutter
                right_leftmost = min(c["x0"] for c in rchars)

                # For a full-width line, the gap between left and right
                # text should be small (characters flow continuously).
                # For two-column text, the gap equals the gutter width.
                gap = right_leftmost - left_rightmost

                if gap < gutter_margin:
                    # Text flows through the gutter — full-width line
                    full_width_lines.append((min(ly, ry), lchars + rchars))
                    matched_left_ys.add(ly)
                    matched_right_ys.add(ry)

    # Remove matched lines from column lists
    left_only = [(y, c) for y, c in left_lines if y not in matched_left_ys]
    right_only = [(y, c) for y, c in right_lines if y not in matched_right_ys]

    # ── Step 4: Determine y-range of columnar content ─────────────────
    col_lines = left_only + right_only
    if not col_lines:
        # No columnar content — everything is full-width
        all_lines = sorted(full_width_lines, key=lambda x: x[0])
        return "\n".join(
            chars_to_text(c).strip() for _, c in all_lines
            if chars_to_text(c).strip()
        )

    col_top = min(y for y, _ in col_lines)
    col_bottom = max(y for y, _ in col_lines)

    # Split full-width lines into header / footer / mid-column
    header_lines = [(y, c) for y, c in full_width_lines if y < col_top]
    footer_lines = [(y, c) for y, c in full_width_lines if y > col_bottom]
    mid_lines = [(y, c) for y, c in full_width_lines
                 if col_top <= y <= col_bottom]

    result = []

    # 1. Header (full-width lines above columns)
    for _, line_chars in sorted(header_lines, key=lambda x: x[0]):
        text = chars_to_text(line_chars)
        if text.strip():
            result.append(text.strip())

    # 2. Left column (top to bottom), interleaving mid-page full-width lines
    left_and_mid = left_only + mid_lines
    left_and_mid.sort(key=lambda x: x[0])
    for _, line_chars in left_and_mid:
        text = chars_to_text(line_chars)
        if text.strip():
            result.append(text.strip())

    # 3. Right column (top to bottom)
    right_only.sort(key=lambda x: x[0])
    for _, line_chars in right_only:
        text = chars_to_text(line_chars)
        if text.strip():
            result.append(text.strip())

    # 4. Footer (full-width lines below columns)
    for _, line_chars in sorted(footer_lines, key=lambda x: x[0]):
        text = chars_to_text(line_chars)
        if text.strip():
            result.append(text.strip())

    return "\n".join(result)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_page_lines(chars: list, page_width: float) -> list:
    """Build an ordered list of (line_text, line_chars) for a page,
    respecting column layout.

    For single-column pages: lines are sorted top-to-bottom.
    For two-column pages: left column lines first (top-to-bottom),
    then right column lines (top-to-bottom), with full-width lines
    placed at their vertical position.

    This is the same ordering used by the parser, so section_detector
    and assemble_sections can reuse it to stay in sync.

    Returns:
        [(line_text, line_chars), ...]
    """
    col_boundary = _detect_column_boundary(chars, page_width)

    if col_boundary is None:
        # Single-column: group all chars into lines
        raw_lines = _group_chars_into_lines(chars)
        result = []
        for _, line_chars in raw_lines:
            text = chars_to_text(line_chars)
            if text.strip():
                result.append((text.strip(), line_chars))
        return result

    # Two-column: split chars by column, group each independently
    left_chars = [c for c in chars if (c["x0"] + c["x1"]) / 2 < col_boundary]
    right_chars = [c for c in chars if (c["x0"] + c["x1"]) / 2 >= col_boundary]

    left_lines = _group_chars_into_lines(left_chars)
    right_lines = _group_chars_into_lines(right_chars)

    # Detect full-width lines (text flows through gutter, gap < 8pt)
    gutter_margin = 8
    full_width_lines = []
    matched_left_ys = set()
    matched_right_ys = set()

    for ly, lchars in left_lines:
        for ry, rchars in right_lines:
            if abs(ly - ry) <= 2:
                left_rightmost = max(c["x1"] for c in lchars)
                right_leftmost = min(c["x0"] for c in rchars)
                if (right_leftmost - left_rightmost) < gutter_margin:
                    full_width_lines.append((min(ly, ry), lchars + rchars))
                    matched_left_ys.add(ly)
                    matched_right_ys.add(ry)

    left_only = [(y, c) for y, c in left_lines if y not in matched_left_ys]
    right_only = [(y, c) for y, c in right_lines if y not in matched_right_ys]

    # Determine column content y-range
    col_lines_all = left_only + right_only
    if not col_lines_all:
        all_lines = sorted(full_width_lines, key=lambda x: x[0])
        return [
            (chars_to_text(c).strip(), c)
            for _, c in all_lines if chars_to_text(c).strip()
        ]

    col_top = min(y for y, _ in col_lines_all)
    col_bottom = max(y for y, _ in col_lines_all)

    header_fw = [(y, c) for y, c in full_width_lines if y < col_top]
    footer_fw = [(y, c) for y, c in full_width_lines if y > col_bottom]
    mid_fw = [(y, c) for y, c in full_width_lines if col_top <= y <= col_bottom]

    result = []

    # 1. Header full-width
    for _, lc in sorted(header_fw, key=lambda x: x[0]):
        text = chars_to_text(lc)
        if text.strip():
            result.append((text.strip(), lc))

    # 2. Left column + mid full-width
    left_and_mid = left_only + mid_fw
    left_and_mid.sort(key=lambda x: x[0])
    for _, lc in left_and_mid:
        text = chars_to_text(lc)
        if text.strip():
            result.append((text.strip(), lc))

    # 3. Right column
    right_only.sort(key=lambda x: x[0])
    for _, lc in right_only:
        text = chars_to_text(lc)
        if text.strip():
            result.append((text.strip(), lc))

    # 4. Footer full-width
    for _, lc in sorted(footer_fw, key=lambda x: x[0]):
        text = chars_to_text(lc)
        if text.strip():
            result.append((text.strip(), lc))

    return result

def extract_text_from_pdf(pdf_path: str) -> dict:
    """Extract clean text from a PDF, handling both 1-column and 2-column
    research paper layouts.

    Returns:
        {
            "pages": [{"page_num": int, "text": str, "chars": list}, ...],
            "full_text": str,
            "total_pages": int,
        }
    """
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            # Keep only upright characters — filters rotated watermarks
            upright_chars = [c for c in page.chars if c.get("upright")]

            if not upright_chars:
                continue

            page_width = page.width

            # Detect column layout for this specific page
            col_boundary = _detect_column_boundary(upright_chars, page_width)

            if col_boundary is not None:
                page_text = _process_two_column(
                    upright_chars, col_boundary, page_width
                )
            else:
                page_text = _process_single_column(upright_chars)

            pages.append({
                "page_num": page_num,
                "text": page_text,
                "chars": upright_chars,
                "page_width": page_width,
            })

    full_text = "\n\n".join(p["text"] for p in pages)

    return {
        "pages": pages,
        "full_text": full_text,
        "total_pages": total_pages,
    }


def remove_credits_block(full_text: str) -> str:
    """Everything before 'Abstract' is author/affiliation noise.
    Find the word Abstract as a standalone line and cut there."""
    match = re.search(r'\nAbstract\n', full_text, re.IGNORECASE)
    if match:
        return full_text[match.start():].strip()
    return full_text


def remove_references_section(full_text: str) -> str:
    """Cut everything from the 'References' heading onwards."""
    match = re.search(r'\nReferences\b', full_text, re.IGNORECASE)
    if match:
        return full_text[:match.start()].strip()
    return full_text