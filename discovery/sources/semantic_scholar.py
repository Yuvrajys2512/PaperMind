import httpx

S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,authors,year,abstract,openAccessPdf,citationCount,venue,externalIds"
_HEADERS = {"User-Agent": "PaperMind/1.0 (research tool)"}


async def search_semantic_scholar(query: str, limit: int = 12) -> list[dict]:
    params = {"query": query, "fields": _FIELDS, "limit": limit}

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(S2_API, params=params, headers=_HEADERS)
        resp.raise_for_status()

    results = []
    for paper in resp.json().get("data", []):
        oa = paper.get("openAccessPdf") or {}
        pdf_url = oa.get("url") or None

        s2_id = paper.get("paperId", "")
        authors = [a.get("name", "") for a in (paper.get("authors") or [])]
        abstract = paper.get("abstract") or ""

        results.append({
            "id": f"s2:{s2_id}",
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
