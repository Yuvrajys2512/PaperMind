import asyncio
import re
import unicodedata
from discovery.sources.arxiv import search_arxiv
from discovery.sources.semantic_scholar import search_semantic_scholar


def _normalize(title: str) -> str:
    t = unicodedata.normalize("NFKD", title.lower())
    return re.sub(r"[^a-z0-9 ]", "", t).strip()


def _similar(a: str, b: str) -> bool:
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return True
    shorter = na if len(na) <= len(nb) else nb
    longer  = nb if len(na) <= len(nb) else na
    return len(shorter) > 12 and longer.startswith(shorter[:35])


async def search_papers(query: str, limit: int = 20) -> list[dict]:
    per_source = min(limit, 15)

    arxiv_res, s2_res = await asyncio.gather(
        search_arxiv(query, limit=per_source),
        search_semantic_scholar(query, limit=per_source),
        return_exceptions=True,
    )

    combined: list[dict] = []
    if not isinstance(arxiv_res, Exception):
        combined.extend(arxiv_res)
    if not isinstance(s2_res, Exception):
        combined.extend(s2_res)

    # Deduplicate by title similarity — keep first occurrence
    seen: list[str] = []
    unique: list[dict] = []
    for r in combined:
        t = r.get("title", "")
        if t and not any(_similar(t, s) for s in seen):
            seen.append(t)
            unique.append(r)

    # Sort: papers with a PDF first, then by citation count desc, then year desc
    unique.sort(
        key=lambda x: (
            1 if x.get("pdf_url") else 0,
            x.get("citations") or 0,
            x.get("year") or 0,
        ),
        reverse=True,
    )

    return unique[:limit]
