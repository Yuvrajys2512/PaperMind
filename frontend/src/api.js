const BASE = '/api'

export async function uploadPaper(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error('Upload failed')
  return res.json()
}

export async function getPaperStatus(paperId) {
  const res = await fetch(`${BASE}/status/${paperId}`)
  if (!res.ok) throw new Error('Status check failed')
  return res.json()
}

export async function queryPaper(paperId, question) {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paper_id: paperId, question })
  })
  if (!res.ok) throw new Error('Query failed')
  return res.json()
}

export async function listPapers() {
  const res = await fetch(`${BASE}/papers`)
  if (!res.ok) throw new Error('Failed to fetch papers')
  return res.json()
}

export async function deletePaper(paperId) {
  const res = await fetch(`${BASE}/papers/${paperId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Delete failed')
  return res.json()
}

export async function comparePapers(paperIdA, paperIdB, question) {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ paper_ids: [paperIdA, paperIdB], question })
  })
  if (!res.ok) throw new Error('Compare failed')
  return res.json()
}

/* ─────────────────────────────────────────────────────────────────
   Streaming variants — Server-Sent Events
   onEvent receives { type, data } where type ∈ {open, progress, done, error}.
───────────────────────────────────────────────────────────────── */
async function streamQuery(body, onEvent) {
  const res = await fetch(`${BASE}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
    body: JSON.stringify(body),
  })
  if (!res.ok || !res.body) {
    throw new Error(`Stream failed: HTTP ${res.status}`)
  }

  const reader  = res.body.getReader()
  const decoder = new TextDecoder()
  let   buffer  = ''
  let   finalResult = null

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // SSE frames are separated by blank lines. Parse complete frames
    // and leave any partial frame in the buffer for the next chunk.
    let idx
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const rawFrame = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)

      let eventType = 'message'
      let dataLine  = ''
      for (const line of rawFrame.split('\n')) {
        if (line.startsWith('event:')) eventType = line.slice(6).trim()
        else if (line.startsWith('data:')) dataLine += line.slice(5).trim()
      }
      if (!dataLine) continue

      let parsed
      try { parsed = JSON.parse(dataLine) } catch { continue }

      onEvent({ type: eventType, data: parsed })

      if (eventType === 'done')  finalResult = parsed
      if (eventType === 'error') throw new Error(parsed.message || 'Pipeline error')
    }
  }

  if (!finalResult) throw new Error('Stream ended without a final result')
  return finalResult
}

export function queryPaperStream(paperId, question, onEvent) {
  return streamQuery({ paper_id: paperId, question }, onEvent)
}

export function comparePapersStream(paperIdA, paperIdB, question, onEvent) {
  return streamQuery({ paper_ids: [paperIdA, paperIdB], question }, onEvent)
}

/* ─────────────────────────────────────────────────────────────────
   Discovery — live paper search + import
───────────────────────────────────────────────────────────────── */
export async function getGlossary(paperId) {
  const res = await fetch(`${BASE}/papers/${paperId}/glossary`)
  if (!res.ok) throw new Error(`Glossary failed: HTTP ${res.status}`)
  return res.json()
}

export async function getRecommendations(paperId) {
  const res = await fetch(`${BASE}/papers/${paperId}/recommendations`)
  if (!res.ok) throw new Error(`Recommendations failed: HTTP ${res.status}`)
  return res.json()
}

export async function searchPapers(query, limit = 20) {
  const res = await fetch(`${BASE}/discovery/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, limit }),
  })
  if (!res.ok) throw new Error(`Search failed: HTTP ${res.status}`)
  return res.json()
}

export async function importPaper(result) {
  const res = await fetch(`${BASE}/discovery/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      title: result.title,
      pdf_url: result.pdf_url,
      source_id: result.id,
      authors: result.authors || [],
      year: result.year || null,
      venue: result.venue || null,
    }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Import failed: HTTP ${res.status}`)
  }
  return res.json()
}