// Dev: unset → '/api', served via Vite's proxy (vite.config.js).
// Prod: set VITE_API_URL to the backend's URL at build time, since the
// frontend (your domain) and backend (e.g. *.hf.space) are different origins.
const BASE = import.meta.env.VITE_API_URL || '/api'

// api.js is plain functions, not components, so it can't use Clerk's
// useAuth() hook — window.Clerk is set globally by @clerk/clerk-react
// and is the documented way to grab a fresh token from non-component code.
async function authHeaders() {
  const token = await window.Clerk?.session?.getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

// Build an Error carrying the HTTP status, so callers can tell a quota 429
// apart from other failures without string-matching the message.
async function httpError(res, fallback) {
  const body = await res.json().catch(() => ({}))
  const err = new Error(body.detail || fallback)
  err.status = res.status
  return err
}

export async function uploadPaper(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/upload`, {
    method: 'POST',
    headers: await authHeaders(),
    body: form,
  })
  if (!res.ok) throw await httpError(res, 'Upload failed')
  return res.json()
}

export async function getPaperStatus(paperId) {
  const res = await fetch(`${BASE}/status/${paperId}`, { headers: await authHeaders() })
  if (!res.ok) throw new Error('Status check failed')
  return res.json()
}

export async function queryPaper(paperId, question) {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...await authHeaders() },
    body: JSON.stringify({ paper_id: paperId, question })
  })
  if (!res.ok) throw await httpError(res, 'Query failed')
  return res.json()
}

export async function listPapers() {
  const res = await fetch(`${BASE}/papers`, { headers: await authHeaders() })
  if (!res.ok) throw new Error('Failed to fetch papers')
  return res.json()
}

export async function deletePaper(paperId) {
  const res = await fetch(`${BASE}/papers/${paperId}`, { method: 'DELETE', headers: await authHeaders() })
  if (!res.ok) throw new Error('Delete failed')
  return res.json()
}

export async function comparePapers(paperIdA, paperIdB, question) {
  const res = await fetch(`${BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...await authHeaders() },
    body: JSON.stringify({ paper_ids: [paperIdA, paperIdB], question })
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Compare failed')
  }
  return res.json()
}

/* ─────────────────────────────────────────────────────────────────
   Streaming variants — Server-Sent Events
   onEvent receives { type, data } where type ∈ {open, progress, done, error}.
───────────────────────────────────────────────────────────────── */
// Drain a fetch Response body as Server-Sent Events, forwarding each frame to
// onEvent and returning the payload of the final `done` frame. Shared by every
// streaming endpoint (query, compare, audit).
async function consumeSSE(res, onEvent) {
  if (!res.ok || !res.body) {
    throw await httpError(res, `Stream failed: HTTP ${res.status}`)
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

async function streamQuery(body, onEvent) {
  const res = await fetch(`${BASE}/query/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream', ...await authHeaders() },
    body: JSON.stringify(body),
  })
  return consumeSSE(res, onEvent)
}

export function queryPaperStream(paperId, question, onEvent) {
  return streamQuery({ paper_id: paperId, question }, onEvent)
}

export function comparePapersStream(paperIdA, paperIdB, question, onEvent) {
  return streamQuery({ paper_ids: [paperIdA, paperIdB], question }, onEvent)
}

// Claim audit — streams progress while the paper's claims are checked against
// its own evidence. Pass force=true to recompute and bypass the cached report.
export async function auditPaperStream(paperId, onEvent, force = false) {
  const res = await fetch(`${BASE}/papers/${paperId}/audit/stream${force ? '?force=1' : ''}`, {
    method: 'POST',
    headers: { 'Accept': 'text/event-stream', ...await authHeaders() },
  })
  return consumeSSE(res, onEvent)
}

// Reviewer / weakness audit — streams progress while the paper is graded against
// venue methodological norms (baselines, ablations, error bars, N, threats,
// related work). Pass force=true to recompute and bypass the cached report.
export async function reviewPaperStream(paperId, onEvent, force = false) {
  const res = await fetch(`${BASE}/papers/${paperId}/review/stream${force ? '?force=1' : ''}`, {
    method: 'POST',
    headers: { 'Accept': 'text/event-stream', ...await authHeaders() },
  })
  return consumeSSE(res, onEvent)
}

/* ─────────────────────────────────────────────────────────────────
   Discovery — live paper search + import
───────────────────────────────────────────────────────────────── */
export async function getGlossary(paperId) {
  const res = await fetch(`${BASE}/papers/${paperId}/glossary`, { headers: await authHeaders() })
  if (!res.ok) throw new Error(`Glossary failed: HTTP ${res.status}`)
  return res.json()
}

export async function getRecommendations(paperId) {
  const res = await fetch(`${BASE}/papers/${paperId}/recommendations`, { headers: await authHeaders() })
  if (!res.ok) throw new Error(`Recommendations failed: HTTP ${res.status}`)
  return res.json()
}

export async function rewriteText(text, mode) {
  const res = await fetch(`${BASE}/rewrite`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...await authHeaders() },
    body: JSON.stringify({ text, mode }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Rewrite failed: HTTP ${res.status}`)
  }
  return res.json()
}

export async function searchPapers(query, limit = 20) {
  const res = await fetch(`${BASE}/discovery/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...await authHeaders() },
    body: JSON.stringify({ query, limit }),
  })
  if (!res.ok) throw new Error(`Search failed: HTTP ${res.status}`)
  return res.json()
}

export async function importPaper(result) {
  const res = await fetch(`${BASE}/discovery/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...await authHeaders() },
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

/* ─────────────────────────────────────────────────────────────────
   Billing — usage/plan + Stripe Checkout / customer portal
───────────────────────────────────────────────────────────────── */
export async function getUsage() {
  const res = await fetch(`${BASE}/usage`, { headers: await authHeaders() })
  if (!res.ok) throw await httpError(res, 'Failed to load plan')
  return res.json()
}

// Admin-only: aggregate cross-user usage + upstream-capacity projection.
// The backend gates this to PAPERMIND_ADMIN_USER_IDS, so a non-admin gets a 403
// (surfaced here as an Error with .status === 403).
export async function getAdminUsage() {
  const res = await fetch(`${BASE}/admin/usage`, { headers: await authHeaders() })
  if (!res.ok) throw await httpError(res, 'Failed to load admin usage')
  return res.json()
}

export async function startCheckout() {
  const res = await fetch(`${BASE}/billing/checkout`, {
    method: 'POST',
    headers: await authHeaders(),
  })
  if (!res.ok) throw await httpError(res, 'Could not start checkout')
  return res.json() // { url }
}

export async function openBillingPortal() {
  const res = await fetch(`${BASE}/billing/portal`, {
    method: 'POST',
    headers: await authHeaders(),
  })
  if (!res.ok) throw await httpError(res, 'Could not open billing portal')
  return res.json() // { url }
}