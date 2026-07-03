import { useState, useEffect, useCallback, useRef } from 'react'
import { listPapers, uploadPaper, deletePaper } from '../api'
import { track } from '../analytics'

const ACCENT = '#a78bfa'

function CosmicOrbs() {
  return (
    <>
      <div className="cosmic-orb-cyan" />
      <div className="cosmic-orb-violet" />
      <div className="cosmic-orb-pink" />
      <div className="scan-line" />
    </>
  )
}

function StatusBadge({ status }) {
  const styles = {
    ready: { bg: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.2)', color: '#6ee7b7' },
    processing: { bg: 'rgba(167,139,250,0.06)', border: '1px solid rgba(167,139,250,0.15)', color: '#c4b5fd' },
    failed: { bg: 'rgba(248,113,113,0.06)', border: '1px solid rgba(248,113,113,0.2)', color: '#fca5a5' },
  }
  const s = styles[status] || styles.failed
  return (
    <span className="text-[9px] font-bold uppercase tracking-[0.12em] px-2 py-0.5 rounded-full"
      style={{ background: s.bg, border: s.border, color: s.color }}>
      {status === 'processing' ? 'Ingesting…' : status}
    </span>
  )
}

function DraftRow({ draft, onOpen, onDelete }) {
  const [confirming, setConfirming] = useState(false)
  const [deleting, setDeleting]     = useState(false)
  const [deleteError, setDeleteError] = useState('')

  const displayName = draft.filename.replace(/\.pdf$/i, '')
  const date = draft.completed_at
    ? new Date(draft.completed_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
    : new Date(draft.uploaded_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })

  const handleDelete = async () => {
    if (!confirming) { setConfirming(true); return }
    setDeleting(true)
    setDeleteError('')
    try { await deletePaper(draft.paper_id); onDelete(draft.paper_id) }
    catch (err) {
      setDeleting(false)
      setConfirming(false)
      setDeleteError(err.message || 'Delete failed. Try again.')
    }
  }

  return (
    <div
      className="flex items-center gap-4 px-5 py-4 rounded-2xl group transition-all duration-200"
      style={{
        background: 'linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01))',
        border: '1px solid rgba(255,255,255,0.05)',
      }}
    >
      <div
        className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
        style={{ background: 'rgba(167,139,250,0.08)', border: '1px solid rgba(167,139,250,0.15)' }}
      >
        <svg className="w-4 h-4 text-violet-400/70" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75"
            d="M11 4h6a2 2 0 012 2v12a2 2 0 01-2 2H7a2 2 0 01-2-2V6a2 2 0 012-2h1m3-2v4m0-4L9 4m3-2l3 2" />
        </svg>
      </div>

      <div className="flex-1 min-w-0">
        <p className="text-sm text-white font-medium truncate" style={{ fontFamily: 'var(--font-display)' }}>
          {displayName}
        </p>
        <div className="flex items-center gap-2 mt-1">
          <StatusBadge status={draft.status} />
          <span className="text-[9px] text-gray-700 font-mono">{date}</span>
        </div>
        {draft.status === 'failed' && draft.error && (
          <p className="text-[10px] text-red-400/60 mt-0.5 truncate">{draft.error}</p>
        )}
        {deleteError && (
          <p className="text-[10px] text-red-400/60 mt-0.5 truncate">{deleteError}</p>
        )}
      </div>

      <div className="flex items-center gap-2 flex-shrink-0">
        {draft.status === 'ready' && (
          <button
            onClick={() => onOpen(draft)}
            className="px-3 py-1.5 rounded-xl text-xs font-semibold transition-all duration-200"
            style={{ background: 'rgba(167,139,250,0.1)', border: '1px solid rgba(167,139,250,0.25)', color: '#c4b5fd' }}
          >
            Review
          </button>
        )}
        {draft.status === 'processing' && (
          <div className="flex items-center gap-1.5 px-3 py-1.5">
            <span className="w-3 h-3 border border-violet-400/40 border-t-violet-400 rounded-full"
              style={{ animation: 'spin 0.9s linear infinite' }} />
            <span className="text-[10px] text-violet-400/50">Processing</span>
          </div>
        )}

        <button
          onClick={handleDelete}
          disabled={deleting}
          className="px-3 py-1.5 rounded-xl text-xs font-semibold transition-all duration-200"
          style={
            confirming
              ? { background: 'rgba(248,113,113,0.12)', border: '1px solid rgba(248,113,113,0.3)', color: '#fca5a5' }
              : { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', color: '#4b5563' }
          }
          onBlur={() => setTimeout(() => setConfirming(false), 200)}
        >
          {deleting ? '…' : confirming ? 'Confirm?' : 'Delete'}
        </button>
      </div>
    </div>
  )
}

export default function DraftsPage({ onOpen, onBack }) {
  const [drafts, setDrafts]   = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState('')
  const [dragging, setDragging] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const [quotaHit, setQuotaHit] = useState(false)
  const inputRef = useRef()

  const fetchDrafts = useCallback(async () => {
    try {
      const data = await listPapers('draft')
      setDrafts(data)
    } catch {
      setError('Could not load your drafts.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    ;(async () => {
      try {
        const data = await listPapers('draft')
        if (active) setDrafts(data)
      } catch {
        if (active) setError('Could not load your drafts.')
      } finally {
        if (active) setLoading(false)
      }
    })()
    return () => { active = false }
  }, [])

  // Auto-refresh while any draft is still processing
  useEffect(() => {
    const anyProcessing = drafts.some(d => d.status === 'processing')
    if (!anyProcessing) return
    const id = setInterval(fetchDrafts, 3000)
    return () => clearInterval(id)
  }, [drafts, fetchDrafts])

  const handleDelete = (paperId) =>
    setDrafts(prev => prev.filter(d => d.paper_id !== paperId))

  const handleFile = useCallback(async (file) => {
    if (!file || !file.name.endsWith('.pdf')) {
      setUploadError('Please upload a PDF file.')
      return
    }
    setUploadError('')
    setQuotaHit(false)
    try {
      await uploadPaper(file, 'draft')
      track('draft_uploaded')
      fetchDrafts()
    } catch (err) {
      setQuotaHit(err.status === 429)
      setUploadError(err.message || 'Upload failed.')
    }
  }, [fetchDrafts])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [handleFile])

  const onDragOver  = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const ready      = drafts.filter(d => d.status === 'ready')
  const processing = drafts.filter(d => d.status === 'processing')
  const failed     = drafts.filter(d => d.status === 'failed')

  return (
    <div className="min-h-screen cosmic-bg flex flex-col items-center px-4 pb-20" style={{ position: 'relative' }}>
      <CosmicOrbs />

      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(167,139,250,0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(167,139,250,0.015) 1px, transparent 1px)
          `,
          backgroundSize: '60px 60px',
          zIndex: 0,
        }}
      />

      {/* Header */}
      <div className="relative z-10 w-full max-w-3xl flex items-center pt-8 mb-10">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 text-gray-600 hover:text-gray-400 transition-colors text-xs mr-auto"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
          </svg>
          Back
        </button>

        <div className="flex flex-col items-center">
          <div
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-3"
            style={{ background: 'rgba(167,139,250,0.06)', border: '1px solid rgba(167,139,250,0.14)' }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: ACCENT, boxShadow: `0 0 6px ${ACCENT}` }} />
            <span className="text-[10px] font-bold uppercase tracking-[0.2em]" style={{ color: '#c4b5fd' }}>
              Write Mode
            </span>
          </div>
          <h1 className="text-3xl font-bold text-white" style={{ fontFamily: 'var(--font-display)' }}>
            My Drafts
          </h1>
          <p className="text-gray-600 text-sm mt-1">
            {loading ? 'Loading…' : `${ready.length} ready · ${processing.length} processing · ${failed.length} failed`}
          </p>
        </div>

        <div className="ml-auto w-[52px]" />
      </div>

      {/* Upload dropzone */}
      <div className="relative z-10 w-full max-w-3xl mb-8">
        <div
          onClick={() => inputRef.current.click()}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className="group relative overflow-hidden rounded-2xl cursor-pointer transition-all duration-300"
          style={{
            background: dragging ? `${ACCENT}0d` : 'rgba(255,255,255,0.02)',
            border: `1px ${dragging ? 'solid' : 'dashed'} ${dragging ? `${ACCENT}88` : 'rgba(255,255,255,0.14)'}`,
            boxShadow: dragging ? `0 0 0 4px ${ACCENT}1a` : 'none',
          }}
        >
          <div className="px-6 py-6 flex items-center gap-5">
            <div
              className="w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors duration-300"
              style={{
                background: dragging ? `${ACCENT}22` : 'rgba(255,255,255,0.05)',
                border: `1px solid ${dragging ? `${ACCENT}55` : 'rgba(255,255,255,0.08)'}`,
              }}
            >
              <svg className="w-5 h-5 transition-colors duration-300"
                style={{ color: dragging ? ACCENT : '#9ca3af' }}
                fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75"
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-[14px] font-medium text-white mb-0.5">
                {dragging ? 'Release to upload' : 'Upload your draft'}
              </div>
              <div className="text-[12px] text-gray-500">
                or <span className="text-gray-300 underline underline-offset-2">browse files</span> — get a pre-submission review before you send it out
              </div>
            </div>
          </div>
          <input ref={inputRef} type="file" accept=".pdf" className="hidden"
            onChange={e => handleFile(e.target.files[0])} />
        </div>

        {uploadError && (
          <div
            className="mt-3 px-4 py-3 rounded-xl flex items-center gap-3"
            style={{ background: 'rgba(248,113,113,0.07)', border: '1px solid rgba(248,113,113,0.22)' }}
          >
            <span className="text-red-400 text-sm font-bold flex-shrink-0">!</span>
            <p className="text-red-200/80 text-[13px] flex-1">{uploadError}</p>
            {quotaHit && (
              <span className="text-[11px] text-red-300/60 flex-shrink-0">Free tier limit reached</span>
            )}
          </div>
        )}
      </div>

      {/* Body */}
      <div className="relative z-10 w-full max-w-3xl flex flex-col gap-2">
        {loading && (
          <div className="flex justify-center py-16">
            <div className="relative w-10 h-10">
              <div className="absolute inset-0 rounded-full border-2 border-transparent"
                style={{ borderTopColor: '#a78bfa', animation: 'spin 1.2s linear infinite' }} />
              <div className="absolute inset-1.5 rounded-full border-2 border-transparent"
                style={{ borderTopColor: '#00f5ff', animation: 'spin 1.8s linear infinite reverse' }} />
            </div>
          </div>
        )}

        {error && (
          <p className="text-center text-red-400/60 text-sm py-10">{error}</p>
        )}

        {!loading && !error && drafts.length === 0 && (
          <div className="flex flex-col items-center py-12 text-center">
            <h3 className="text-white font-semibold text-base mb-2" style={{ fontFamily: 'var(--font-display)' }}>
              No drafts yet
            </h3>
            <p className="text-gray-600 text-sm max-w-xs leading-relaxed">
              Upload an unpublished draft above to get a weakness review and a claim↔evidence audit before you submit.
            </p>
          </div>
        )}

        {!loading && drafts.map(draft => (
          <DraftRow key={draft.paper_id} draft={draft} onOpen={onOpen} onDelete={handleDelete} />
        ))}
      </div>

      <div className="fixed bottom-0 left-0 right-0 text-center py-4 pointer-events-none z-10">
        <p className="text-gray-700 text-[10px] uppercase tracking-[0.2em]" style={{ fontFamily: 'var(--font-mono)' }}>
          PaperMind · Write Mode
        </p>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
