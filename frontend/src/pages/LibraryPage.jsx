import { useState, useEffect, useCallback } from 'react'
import { listPapers, deletePaper } from '../api'

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
    ready: {
      bg: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.2)', color: '#6ee7b7',
    },
    processing: {
      bg: 'rgba(0,245,255,0.06)', border: '1px solid rgba(0,245,255,0.15)', color: '#67e8f9',
    },
    failed: {
      bg: 'rgba(248,113,113,0.06)', border: '1px solid rgba(248,113,113,0.2)', color: '#fca5a5',
    },
  }
  const s = styles[status] || styles.failed
  return (
    <span
      className="text-[9px] font-bold uppercase tracking-[0.12em] px-2 py-0.5 rounded-full"
      style={{ background: s.bg, border: s.border, color: s.color }}
    >
      {status === 'processing' ? 'Ingesting…' : status}
    </span>
  )
}

function PaperRow({ paper, onOpen, onDelete }) {
  const [confirming, setConfirming] = useState(false)
  const [deleting, setDeleting]     = useState(false)

  const displayName = paper.filename.replace(/\.pdf$/i, '')
  const date = paper.completed_at
    ? new Date(paper.completed_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
    : new Date(paper.uploaded_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })

  const handleDelete = async () => {
    if (!confirming) { setConfirming(true); return }
    setDeleting(true)
    try { await deletePaper(paper.paper_id); onDelete(paper.paper_id) }
    catch { setDeleting(false); setConfirming(false) }
  }

  return (
    <div
      className="flex items-center gap-4 px-5 py-4 rounded-2xl group transition-all duration-200"
      style={{
        background: 'linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01))',
        border: '1px solid rgba(255,255,255,0.05)',
      }}
    >
      {/* Icon */}
      <div
        className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
        style={{ background: 'rgba(0,245,255,0.06)', border: '1px solid rgba(0,245,255,0.1)' }}
      >
        <svg className="w-4 h-4 text-cyan-500/60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75"
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      </div>

      {/* Name + meta */}
      <div className="flex-1 min-w-0">
        <p className="text-sm text-white font-medium truncate" style={{ fontFamily: 'var(--font-display)' }}>
          {displayName}
        </p>
        <div className="flex items-center gap-2 mt-1">
          <StatusBadge status={paper.status} />
          <span className="text-[9px] text-gray-700 font-mono">{date}</span>
          {paper.source_id && (
            <span className="text-[9px] text-gray-700 truncate max-w-[120px]">{paper.source_id}</span>
          )}
        </div>
        {paper.status === 'failed' && paper.error && (
          <p className="text-[10px] text-red-400/60 mt-0.5 truncate">{paper.error}</p>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2 flex-shrink-0">
        {paper.status === 'ready' && (
          <button
            onClick={() => onOpen(paper)}
            className="px-3 py-1.5 rounded-xl text-xs font-semibold transition-all duration-200"
            style={{
              background: 'rgba(0,245,255,0.08)',
              border: '1px solid rgba(0,245,255,0.18)',
              color: '#22d3ee',
            }}
          >
            Open in Chat
          </button>
        )}
        {paper.status === 'processing' && (
          <div className="flex items-center gap-1.5 px-3 py-1.5">
            <span className="w-3 h-3 border border-cyan-400/40 border-t-cyan-400 rounded-full"
              style={{ animation: 'spin 0.9s linear infinite' }} />
            <span className="text-[10px] text-cyan-500/50">Processing</span>
          </div>
        )}

        {/* Delete */}
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

export default function LibraryPage({ onOpen, onBack, onDiscover }) {
  const [papers, setPapers]   = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState('')

  const fetchPapers = useCallback(async () => {
    try {
      const data = await listPapers()
      setPapers(data)
    } catch {
      setError('Could not load library.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchPapers()
  }, [fetchPapers])

  // Auto-refresh while any paper is still processing
  useEffect(() => {
    const anyProcessing = papers.some(p => p.status === 'processing')
    if (!anyProcessing) return
    const id = setInterval(fetchPapers, 3000)
    return () => clearInterval(id)
  }, [papers, fetchPapers])

  const handleDelete = (paperId) =>
    setPapers(prev => prev.filter(p => p.paper_id !== paperId))

  const ready      = papers.filter(p => p.status === 'ready')
  const processing = papers.filter(p => p.status === 'processing')
  const failed     = papers.filter(p => p.status === 'failed')

  return (
    <div
      className="min-h-screen cosmic-bg flex flex-col items-center px-4 pb-20"
      style={{ position: 'relative' }}
    >
      <CosmicOrbs />

      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0,245,255,0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,245,255,0.015) 1px, transparent 1px)
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
            style={{ background: 'rgba(0,245,255,0.06)', border: '1px solid rgba(0,245,255,0.14)' }}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"
              style={{ boxShadow: '0 0 6px rgba(0,245,255,0.7)' }} />
            <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-400/70">
              Research Library
            </span>
          </div>
          <h1 className="text-3xl font-bold text-white" style={{ fontFamily: 'var(--font-display)' }}>
            My Papers
          </h1>
          <p className="text-gray-600 text-sm mt-1">
            {loading ? 'Loading…' : `${ready.length} ready · ${processing.length} processing · ${failed.length} failed`}
          </p>
        </div>

        <button
          onClick={onDiscover}
          className="ml-auto flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full transition-all"
          style={{
            background: 'rgba(167,139,250,0.06)',
            border: '1px solid rgba(167,139,250,0.14)',
            color: '#8b5cf6',
          }}
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          Discover
        </button>
      </div>

      {/* Body */}
      <div className="relative z-10 w-full max-w-3xl flex flex-col gap-2">

        {loading && (
          <div className="flex justify-center py-16">
            <div className="relative w-10 h-10">
              <div className="absolute inset-0 rounded-full border-2 border-transparent"
                style={{ borderTopColor: '#00f5ff', animation: 'spin 1.2s linear infinite' }} />
              <div className="absolute inset-1.5 rounded-full border-2 border-transparent"
                style={{ borderTopColor: '#a78bfa', animation: 'spin 1.8s linear infinite reverse' }} />
            </div>
          </div>
        )}

        {error && (
          <p className="text-center text-red-400/60 text-sm py-10">{error}</p>
        )}

        {!loading && !error && papers.length === 0 && (
          <div className="flex flex-col items-center py-16 gap-4">
            <svg className="w-10 h-10 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5"
                d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
            <p className="text-gray-600 text-sm">No papers yet</p>
            <div className="flex gap-3">
              <button onClick={onBack}
                className="px-4 py-2 rounded-xl text-xs font-semibold"
                style={{ background: 'rgba(0,245,255,0.08)', border: '1px solid rgba(0,245,255,0.18)', color: '#22d3ee' }}>
                Upload a PDF
              </button>
              <button onClick={onDiscover}
                className="px-4 py-2 rounded-xl text-xs font-semibold"
                style={{ background: 'rgba(167,139,250,0.08)', border: '1px solid rgba(167,139,250,0.2)', color: '#c4b5fd' }}>
                Discover Papers
              </button>
            </div>
          </div>
        )}

        {!loading && papers.map(paper => (
          <PaperRow
            key={paper.paper_id}
            paper={paper}
            onOpen={onOpen}
            onDelete={handleDelete}
          />
        ))}
      </div>

      <div className="fixed bottom-0 left-0 right-0 text-center py-4 pointer-events-none z-10">
        <p className="text-gray-700 text-[10px] uppercase tracking-[0.2em]"
          style={{ fontFamily: 'var(--font-mono)' }}>
          PaperMind · Research Library
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
