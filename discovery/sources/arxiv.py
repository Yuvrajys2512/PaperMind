import httpx
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

ARXIV_API = "https://export.arxiv.org/api/query"
_NS = "http://www.w3.org/2005/Atom"


async def search_arxiv(query: str, limit: int = 12) -> list[dict]:
    params = {
        "search_query": f"all:{quote_plus(query)}",
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(ARXIV_API, params=params)
        resp.raise_for_status()

    root = ET.fromstring(resp.text)
    results = []

    for entry in root.findall(f"{{{_NS}}}entry"):
        id_url = entry.findtext(f"{{{_NS}}}id", "")
        arxiv_id = id_url.split("/abs/")[-1].split("v")[0] if "/abs/" in id_url else ""

        title = entry.findtext(f"{{{_NS}}}title", "").strip().replace("\n", " ")
        abstract = entry.findtext(f"{{{_NS}}}summary", "").strip().replace("\n", " ")
        published = entry.findtext(f"{{{_NS}}}published", "")
        year = int(published[:4]) if len(published) >= 4 else None

        authors = [
            a.findtext(f"{{{_NS}}}name", "")
            for a in entry.findall(f"{{{_NS}}}author")
        ]

        pdf_url = None
        for link in entry.findall(f"{{{_NS}}}link"):
            if link.get("title") == "pdf":
                href = link.get("href", "").replace("http://", "https://")
                pdf_url = href if href.endswith(".pdf") else href + ".pdf"
                break
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

        results.append({
            "id": f"arxiv:{arxiv_id}",
            "title": title,
            "authors": authors[:5],
            "year": year,
            "abstract": abstract[:600] + ("…" if len(abstract) > 600 else ""),
            "source": "arXiv",
            "source_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": pdf_url,
            "citations": None,
            "venue": None,
        })

    return results
