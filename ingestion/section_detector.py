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
from collections import Counter


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
        body_size = get_body_font_size(chars)

        line_map = {}
        for c in chars:
            y = round(c["y0"])
            if y not in line_map:
                line_map[y] = []
            line_map[y].append(c)

        sorted_ys = sorted(line_map.keys(), reverse=True)

        lines = []
        for y in sorted_ys:
            line_chars = sorted(line_map[y], key=lambda c: c["x0"])
            line_text = chars_to_text(line_chars)
            if line_text.strip():
                lines.append((line_text, line_chars))

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
                    "score": score
                })
                candidate_id += 1

            if line_chars:
                prev_line_y = min(c["y0"] for c in line_chars)

    return candidates