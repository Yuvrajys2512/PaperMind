import asyncio
import logging
import os
import httpx

log = logging.getLogger(__name__)

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,authors,year,abstract,openAccessPdf,citationCount,venue,externalIds"


def _headers() -> dict:
    h = {"User-Agent": "PaperMind/1.0 (research tool)"}
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        h["x-api-key"] = key
    return h


async def search_semantic_scholar(query: str, limit: int = 12) -> list[dict]:
    params = {"query": query, "fields": _FIELDS, "limit": limit}

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(S2_API, params=params, headers=_headers())

        if resp.status_code == 429:
            log.warning("Semantic Scholar rate-limited (429) — retrying after 5s")
            await asyncio.sleep(5)
            resp = await client.get(S2_API, params=params, headers=_headers())

        if resp.status_code != 200:
            log.error("Semantic Scholar returned HTTP %d: %s", resp.status_code, resp.text[:200])
            resp.raise_for_status()

    raw = resp.json().get("data", [])
    log.info("Semantic Scholar returned %d results for %r", len(raw), query)

    results = []
    for paper in raw:
        oa = paper.get("openAccessPdf") or {}
        pdf_url = oa.get("url") or None

        s2_id = paper.get("paperId", "")
        authors = [a.get("name", "") for a in (paper.get("authors") or [])]
        abstract = paper.get("abstract") or ""

        # Extract arXiv ID so the aggregator can merge instead of duplicate
        external = paper.get("externalIds") or {}
        arxiv_id = external.get("ArXiv") or None

        results.append({
            "id": f"s2:{s2_id}",
            "arxiv_id": arxiv_id,          # used for merge-dedup in search.py
            "title": paper.get("title") or "",
            "authors": authors[:5],
            "year": paper.get("year"),
            "abstract": abstract[:600] + ("…" if len(abstract) > 600 else ""),
            "source": "Semantic Scholar",
            "source_url": f"https://www.semanticscholar.org/paper/{s2_id}",
            "pdf_url": pdf_url,
            "citations": paper.get("citationCount"),
            "venue": paper.get("venue") or None,
        })

    return results
