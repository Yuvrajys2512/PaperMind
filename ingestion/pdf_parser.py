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
# Table region handling — caption-anchored detection
# ─────────────────────────────────────────────────────────────────────────────
#
# Pure-geometry table detection is unreliable on real papers: the default
# "lines" strategy misses borderless (booktabs) tables and mis-fires on
# figure grids, while the "text" strategy flags every justified-prose page as a
# table. So we invert the mechanism: a "Table N:" caption is a reliable TEXT
# anchor, so we use captions to LOCATE tables, then run text-based extraction
# confined to the region just below each caption. Confining extraction to a
# known table region is what makes the text strategy safe.

# A caption looks like "Table 1:" / "Table 2." — we require the trailing colon
# or period so in-body cross-references ("Table 2 summarizes our results ...")
# are NOT mistaken for captions.
_TABLE_CAPTION_RE = re.compile(r'^\s*table\s+\d+\s*[:.]', re.IGNORECASE)

# How far below a caption we look for its table grid (points). Generous — the
# detector returns the table's own tight bbox, so over-cropping is harmless.
_TABLE_CROP_DEPTH = 340.0

# Text-alignment table extraction, confined to the crop below a caption.
_TABLE_SETTINGS = {"horizontal_strategy": "text", "vertical_strategy": "text"}


def _char_center_in_bbox(c: dict, bbox: tuple) -> bool:
    """True if a character's center point falls inside a table bounding box.

    bbox is pdfplumber's (x0, top, x1, bottom) in the same top-based coordinate
    space as char['x0']/'top']/'x1']/'bottom']. Using the center (rather than any
    overlap) avoids dropping body-text chars that merely graze a table's edge.
    """
    x0, top, x1, bottom = bbox
    cx = (c["x0"] + c["x1"]) / 2
    cy = (c["top"] + c["bottom"]) / 2
    return x0 <= cx <= x1 and top <= cy <= bottom


def _caption_x_range(line_chars: list, col_boundary, page_width: float) -> tuple:
    """Horizontal crop range for a caption's table.

    On two-column pages a full-width crop would merge the table with prose from
    the other column, so we confine the crop to the caption's own column. A
    caption whose text spans the gutter belongs to a full-width table and keeps
    the whole width.
    """
    if col_boundary is None or not line_chars:
        return (0.0, page_width)
    left = min(c["x0"] for c in line_chars)
    right = max(c["x1"] for c in line_chars)
    margin = 8.0
    spans_gutter = left < col_boundary - margin and right > col_boundary + margin
    if spans_gutter:
        return (0.0, page_width)
    center = (left + right) / 2
    if center < col_boundary:
        return (0.0, col_boundary)
    return (col_boundary, page_width)


def _extract_table_records(page, all_lines: list, page_num: int,
                           col_boundary, page_width: float) -> list:
    """Caption-anchored table extraction for one page.

    1. Find caption lines ("Table N:") via text.
    2. For each, crop the region below it (bounded by the next caption, a fixed
       depth, and — on two-column pages — the caption's own column) and run
       text-strategy extraction confined to that crop.
    3. Keep the densest detected grid; record its rows, caption, and tight bbox.

    Returns a list of {"rows", "page_num", "caption", "bbox"} records. The bbox
    is in page coordinates so the caller can carve those chars out of the prose.
    """
    captions = []  # (y, text, line_chars)
    for y, line_chars in all_lines:
        text = chars_to_text(line_chars).strip()
        if _TABLE_CAPTION_RE.match(text):
            captions.append((y, text, line_chars))
    if not captions:
        return []

    captions.sort(key=lambda c: c[0])
    cap_ys = [y for y, _, _ in captions]
    records = []

    for idx, (cap_y, cap_text, cap_chars) in enumerate(captions):
        top = cap_y + 12  # skip the caption line itself
        # Bottom bound: the next caption on this page, else a fixed depth.
        next_cap = cap_ys[idx + 1] if idx + 1 < len(cap_ys) else None
        bottom = min(
            page.height,
            cap_y + _TABLE_CROP_DEPTH,
            next_cap if next_cap is not None else page.height,
        )
        if bottom - top < 10:
            continue

        x_left, x_right = _caption_x_range(cap_chars, col_boundary, page_width)

        try:
            crop = page.crop((x_left, top, x_right, bottom))
            found = crop.find_tables(table_settings=_TABLE_SETTINGS)
        except Exception:
            found = []
        if not found:
            continue

        # Densest grid = most cells; that's the table, not stray aligned prose.
        best = max(found, key=lambda t: sum(len(r) for r in (t.extract() or [])))
        rows = best.extract()
        if not rows:
            continue

        records.append({
            "rows": rows,
            "page_num": page_num,
            "caption": cap_text,
            "bbox": best.bbox,
        })

    return records


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
            "tables": [{"rows": list, "page_num": int, "caption": str,
                        "bbox": tuple}, ...],
        }

    Table regions are detected once here and their characters are carved OUT of
    each page's prose stream (and out of page['chars']). This stops table cells
    from being mangled into prose lines or scored as heading candidates, and it
    means the table grid is represented exactly once — as a structured record in
    'tables' — instead of being duplicated as both garbled prose and a chunk.
    """
    pages = []
    tables = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages, start=1):
            # Keep only upright characters — filters rotated watermarks
            upright_chars = [c for c in page.chars if c.get("upright")]

            if not upright_chars:
                continue

            page_width = page.width

            # ── Detect tables (caption-anchored); carve them out of prose ───
            # Column boundary is computed from the full char set (before
            # carving) so two-column table crops can be confined to one column.
            all_lines = _group_chars_into_lines(upright_chars)
            page_col_boundary = _detect_column_boundary(upright_chars, page_width)
            page_records = _extract_table_records(
                page, all_lines, page_num, page_col_boundary, page_width
            )
            tables.extend(page_records)
            table_bboxes = [r["bbox"] for r in page_records]

            if table_bboxes:
                prose_chars = [
                    c for c in upright_chars
                    if not any(_char_center_in_bbox(c, b) for b in table_bboxes)
                ]
            else:
                prose_chars = upright_chars

            if not prose_chars:
                # Page was entirely table — nothing to add to the prose stream.
                continue

            # Detect column layout for this specific page
            col_boundary = _detect_column_boundary(prose_chars, page_width)

            if col_boundary is not None:
                page_text = _process_two_column(
                    prose_chars, col_boundary, page_width
                )
            else:
                page_text = _process_single_column(prose_chars)

            pages.append({
                "page_num": page_num,
                "text": page_text,
                "chars": prose_chars,
                "page_width": page_width,
            })

    full_text = "\n\n".join(p["text"] for p in pages)

    return {
        "pages": pages,
        "full_text": full_text,
        "total_pages": total_pages,
        "tables": tables,
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