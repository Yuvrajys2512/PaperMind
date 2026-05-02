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