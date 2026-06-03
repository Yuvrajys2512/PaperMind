import asyncio
import logging
import re
import unicodedata
from discovery.sources.arxiv import search_arxiv
from discovery.sources.semantic_scholar import search_semantic_scholar

log = logging.getLogger(__name__)


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

    if isinstance(arxiv_res, Exception):
        log.error("arXiv search failed: %s", arxiv_res)
        arxiv_res = []
    if isinstance(s2_res, Exception):
        log.error("Semantic Scholar search failed: %s", s2_res)
        s2_res = []

    log.info("Search sources: arXiv=%d, S2=%d", len(arxiv_res), len(s2_res))

    # Build a lookup from arXiv ID → arXiv result so we can merge S2 metadata in.
    arxiv_by_id: dict[str, dict] = {}
    for r in arxiv_res:
        aid = r["id"].removeprefix("arxiv:")
        if aid:
            arxiv_by_id[aid] = r

    # Merge S2 citation/venue data into matched arXiv entries; collect S2-only papers.
    s2_only: list[dict] = []
    for paper in s2_res:
        aid = paper.get("arxiv_id")
        if aid and aid in arxiv_by_id:
            # Same paper — enrich the arXiv entry with S2 metadata
            target = arxiv_by_id[aid]
            if paper.get("citations") is not None:
                target["citations"] = paper["citations"]
            if paper.get("venue"):
                target["venue"] = paper["venue"]
            target["source_url_s2"] = paper["source_url"]
        else:
            # No arXiv counterpart — check by title similarity against all arXiv results
            title = paper.get("title", "")
            matched = any(_similar(title, r["title"]) for r in arxiv_res)
            if matched:
                # Same paper, no arXiv ID on S2 side — enrich the matching arXiv entry
                for r in arxiv_res:
                    if _similar(title, r["title"]):
                        if paper.get("citations") is not None:
                            r["citations"] = paper["citations"]
                        if paper.get("venue"):
                            r["venue"] = paper["venue"]
                        break
            else:
                s2_only.append(paper)

    # Strip the internal arxiv_id field before returning
    for p in s2_only:
        p.pop("arxiv_id", None)

    combined = list(arxiv_res) + s2_only
    log.info("After merge: %d arXiv + %d S2-only = %d total", len(arxiv_res), len(s2_only), len(combined))

    # Sort: papers with a PDF first, then by citation count desc, then year desc
    combined.sort(
        key=lambda x: (
            1 if x.get("pdf_url") else 0,
            x.get("citations") or 0,
            x.get("year") or 0,
        ),
        reverse=True,
    )

    return combined[:limit]
