import { useState, useRef, useCallback } from 'react'
import { searchPapers, importPaper, getPaperStatus } from '../api'

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

/* ── Source badge ─────────────────────────────────────────────── */
function SourceBadge({ source }) {
  const isArxiv = source === 'arXiv'
  return (
    <span
      className="text-[9px] font-bold uppercase tracking-[0.15em] px-2 py-0.5 rounded-full"
      style={{
        background: isArxiv
          ? 'rgba(0,245,255,0.08)'
          : 'rgba(167,139,250,0.10)',
        border: isArxiv
          ? '1px solid rgba(0,245,255,0.2)'
          : '1px solid rgba(167,139,250,0.25)',
        color: isArxiv ? '#67e8f9' : '#c4b5fd',
      }}
    >
      {source}
    </span>
  )
}

/* ── Single result card ───────────────────────────────────────── */
function PaperCard({ result, importState, onImport, onGoToChat }) {
  const hasPdf = Boolean(result.pdf_url)
  const state  = importState || 'idle'

  const buttonContent = () => {
    if (state === 'downloading') return (
      <span className="flex items-center gap-1.5">
        <span className="w-3 h-3 border border-cyan-400/60 border-t-cyan-400 rounded-full"
          style={{ animation: 'spin 0.8s linear infinite' }} />
        Downloading…
      </span>
    )
    if (state === 'processing') return (
      <span className="flex items-center gap-1.5">
        <span className="w-3 h-3 border border-violet-400/60 border-t-violet-400 rounded-full"
          style={{ animation: 'spin 0.8s linear infinite' }} />
        Ingesting…
      </span>
    )
    if (state === 'ready') return (
      <span className="flex items-center gap-1.5">
        <svg className="w-3 h-3 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M5 13l4 4L19 7" />
        </svg>
        Open in Chat
      </span>
    )
    if (state === 'failed') return 'Failed — retry?'
    if (!hasPdf) return 'No PDF available'
    return 'Import to Library'
  }

  const buttonDisabled = !hasPdf && state === 'idle'
  const buttonActive   = hasPdf && (state === 'idle' || state === 'failed')

  return (
    <div
      className="flex flex-col rounded-2xl p-5 transition-all duration-300 group"
      style={{
        background: 'linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01))',
        border: state === 'ready'
          ? '1px solid rgba(52,211,153,0.25)'
          : '1px solid rgba(255,255,255,0.06)',
        backdropFilter: 'blur(16px)',
        boxShadow: state === 'ready'
          ? '0 0 20px rgba(52,211,153,0.06)'
          : 'none',
      }}
    >
      {/* Top row: source + year + venue */}
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <SourceBadge source={result.source} />
        {result.year && (
          <span className="text-[9px] font-mono text-gray-600">{result.year}</span>
        )}
        {result.venue && (
          <span className="text-[9px] text-gray-700 truncate max-w-[160px]">{result.venue}</span>
        )}
        {result.citations != null && (
          <span className="ml-auto text-[9px] font-mono text-gray-600 flex items-center gap-1 flex-shrink-0">
            <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V12a1 1 0 11-2 0V4.804z" />
            </svg>
            {result.citations.toLocaleString()}
          </span>
        )}
      </div>

      {/* Title */}
      <h3
        className="text-sm font-semibold text-white mb-1.5 leading-snug line-clamp-2"
        style={{ fontFamily: 'var(--font-display)' }}
      >
        <a
          href={result.source_url}
          target="_blank"
          rel="noopener noreferrer"
          className="hover:text-cyan-300 transition-colors"
          onClick={e => e.stopPropagation()}
        >
          {result.title}
        </a>
      </h3>

      {/* Authors */}
      {result.authors?.length > 0 && (
        <p className="text-[10px] text-gray-600 mb-2 truncate">
          {result.authors.join(', ')}
          {result.authors.length >= 5 && ' et al.'}
        </p>
      )}

      {/* Abstract */}
      {result.abstract && (
        <p className="text-xs text-gray-500 leading-relaxed mb-4 line-clamp-3 flex-1">
          {result.abstract}
        </p>
      )}

      {/* Import button */}
      <button
        onClick={() => {
          if (state === 'ready') { onGoToChat(); return }
          if (buttonActive) onImport()
        }}
        disabled={buttonDisabled || state === 'downloading' || state === 'processing'}
        title={!hasPdf ? 'No open-access PDF available for this paper' : undefined}
        className="mt-auto w-full py-2 rounded-xl text-xs font-semibold transition-all duration-200"
        style={
          state === 'ready'
            ? {
                background: 'rgba(52,211,153,0.12)',
                border: '1px solid rgba(52,211,153,0.3)',
                color: '#6ee7b7',
                cursor: 'pointer',
              }
            : buttonDisabled
            ? {
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.06)',
                color: '#374151',
                cursor: 'not-allowed',
              }
            : state === 'failed'
            ? {
                background: 'rgba(248,113,113,0.08)',
                border: '1px solid rgba(248,113,113,0.2)',
                color: '#fca5a5',
                cursor: 'pointer',
              }
            : state === 'downloading' || state === 'processing'
            ? {
                background: 'rgba(0,245,255,0.06)',
                border: '1px solid rgba(0,245,255,0.15)',
                color: '#67e8f9',
                cursor: 'default',
              }
            : {
                background: 'rgba(0,245,255,0.08)',
                border: '1px solid rgba(0,245,255,0.2)',
                color: '#22d3ee',
                cursor: 'pointer',
              }
        }
      >
        {buttonContent()}
      </button>
    </div>
  )
}

/* ── Main page ────────────────────────────────────────────────── */
export default function DiscoverPage({ onPaperReady, onBack }) {
  const [query, setQuery]         = useState('')
  const [phase, setPhase]         = useState('idle')  // idle | searching | results | empty | error
  const [results, setResults]     = useState([])
  const [searchError, setSearchError] = useState('')
  // Map of result.id → { state: 'idle'|'downloading'|'processing'|'ready'|'failed', paperId }
  const [importStates, setImportStates] = useState({})
  const inputRef = useRef()

  const setImportState = (id, patch) =>
    setImportStates(prev => ({ ...prev, [id]: { ...(prev[id] || {}), ...patch } }))

  const handleSearch = useCallback(async () => {
    const q = query.trim()
    if (!q) return
    setPhase('searching')
    setResults([])
    setImportStates({})
    setSearchError('')
    try {
      const data = await searchPapers(q, 24)
      if (!data.results?.length) {
        setPhase('empty')
      } else {
        setResults(data.results)
        setPhase('results')
      }
    } catch (err) {
      setSearchError(err.message || 'Search failed.')
      setPhase('error')
    }
  }, [query])

  const handleImport = useCallback(async (result) => {
    const id = result.id
    setImportState(id, { state: 'downloading' })
    try {
      const { paper_id } = await importPaper(result)
      setImportState(id, { state: 'processing', paperId: paper_id })

      const interval = setInterval(async () => {
        try {
          const status = await getPaperStatus(paper_id)
          if (status.status === 'ready') {
            clearInterval(interval)
            setImportState(id, { state: 'ready', paperId: paper_id, paper: status })
          } else if (status.status === 'failed') {
            clearInterval(interval)
            setImportState(id, { state: 'failed' })
          }
        } catch {
          clearInterval(interval)
          setImportState(id, { state: 'failed' })
        }
      }, 2500)
    } catch {
      setImportState(id, { state: 'failed' })
    }
  }, [])

  const handleGoToChat = useCallback((id) => {
    const entry = importStates[id]
    if (entry?.paper) onPaperReady(entry.paper)
  }, [importStates, onPaperReady])

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

      {/* ── HEADER ── */}
      <div className="relative z-10 w-full max-w-4xl flex items-center pt-8 mb-10">
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
            style={{
              background: 'rgba(167,139,250,0.06)',
              border: '1px solid rgba(167,139,250,0.14)',
            }}
          >
            <span
              className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse"
              style={{ boxShadow: '0 0 6px rgba(167,139,250,0.7)' }}
            />
            <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-violet-400/70">
              Live Research Discovery
            </span>
          </div>
          <h1
            className="text-3xl font-bold text-white text-center"
            style={{ fontFamily: 'var(--font-display)' }}
          >
            Discover Papers
          </h1>
          <p className="text-gray-600 text-sm mt-1.5 text-center">
            Search arXiv and Semantic Scholar · import directly into your library
          </p>
        </div>

        <div className="ml-auto w-12" /> {/* balance flex */}
      </div>

      {/* ── SEARCH BAR ── */}
      <div className="relative z-10 w-full max-w-2xl mb-10">
        <div
          className="flex items-center rounded-2xl overflow-hidden"
          style={{
            background: 'linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02))',
            border: '1px solid rgba(0,245,255,0.15)',
            backdropFilter: 'blur(24px)',
            boxShadow: '0 0 30px rgba(0,245,255,0.05)',
          }}
        >
          <svg
            className="w-4 h-4 text-gray-600 ml-4 flex-shrink-0"
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>

          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSearch()}
            placeholder="e.g. attention mechanism transformers, RLHF alignment…"
            className="flex-1 bg-transparent px-3 py-3.5 text-sm text-white placeholder-gray-700 outline-none"
            autoFocus
          />

          <button
            onClick={handleSearch}
            disabled={!query.trim() || phase === 'searching'}
            className="m-1.5 px-5 py-2 rounded-xl text-xs font-semibold transition-all duration-200"
            style={
              !query.trim() || phase === 'searching'
                ? {
                    background: 'rgba(255,255,255,0.04)',
                    color: '#374151',
                    cursor: 'not-allowed',
                  }
                : {
                    background: 'linear-gradient(135deg, rgba(0,245,255,0.15), rgba(99,102,241,0.12))',
                    border: '1px solid rgba(0,245,255,0.2)',
                    color: '#22d3ee',
                    cursor: 'pointer',
                    boxShadow: '0 0 12px rgba(0,245,255,0.08)',
                  }
            }
          >
            {phase === 'searching' ? 'Searching…' : 'Search'}
          </button>
        </div>
      </div>

      {/* ── BODY ── */}
      <div className="relative z-10 w-full max-w-4xl">

        {/* Searching spinner */}
        {phase === 'searching' && (
          <div className="flex flex-col items-center py-20 gap-5">
            <div className="relative w-16 h-16">
              <div className="absolute inset-0 rounded-full border-2 border-transparent"
                style={{ borderTopColor: '#00f5ff', animation: 'spin 1.2s linear infinite' }} />
              <div className="absolute inset-2 rounded-full border-2 border-transparent"
                style={{ borderTopColor: '#a78bfa', animation: 'spin 1.8s linear infinite reverse' }} />
              <div
                className="absolute inset-4 rounded-full animate-pulse"
                style={{ background: 'linear-gradient(135deg, rgba(0,245,255,0.3), rgba(99,102,241,0.3))' }}
              />
            </div>
            <p className="text-gray-600 text-sm">
              Searching arXiv and Semantic Scholar in parallel…
            </p>
          </div>
        )}

        {/* Empty */}
        {phase === 'empty' && (
          <div className="flex flex-col items-center py-20 gap-3">
            <svg className="w-10 h-10 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5"
                d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-gray-600 text-sm">No results found for &ldquo;{query}&rdquo;</p>
            <button onClick={() => { setPhase('idle'); inputRef.current?.focus() }}
              className="text-cyan-500/60 hover:text-cyan-400 text-xs transition-colors mt-1">
              Try a different query
            </button>
          </div>
        )}

        {/* Error */}
        {phase === 'error' && (
          <div className="flex flex-col items-center py-16 gap-3">
            <div
              className="px-5 py-3 rounded-2xl text-sm text-red-300/80"
              style={{ background: 'rgba(248,113,113,0.06)', border: '1px solid rgba(248,113,113,0.15)' }}
            >
              {searchError}
            </div>
            <button onClick={() => setPhase('idle')}
              className="text-gray-600 hover:text-gray-400 text-xs transition-colors">
              Try again
            </button>
          </div>
        )}

        {/* Idle hint */}
        {phase === 'idle' && (
          <div className="flex flex-col items-center py-16 gap-4 text-center">
            <div className="flex gap-3 flex-wrap justify-center">
              {[
                'attention mechanism transformers',
                'RAG retrieval augmented generation',
                'diffusion models image synthesis',
                'RLHF alignment language models',
              ].map(hint => (
                <button
                  key={hint}
                  onClick={() => { setQuery(hint); setTimeout(handleSearch, 0) }}
                  className="px-3 py-1.5 rounded-full text-[10px] text-gray-600 transition-all duration-200 hover:text-gray-400"
                  style={{
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.06)',
                  }}
                >
                  {hint}
                </button>
              ))}
            </div>
            <p className="text-gray-700 text-xs">Type a topic or click a suggestion to start</p>
          </div>
        )}

        {/* Results grid */}
        {phase === 'results' && (
          <>
            <p className="text-gray-700 text-xs mb-4">
              {results.length} result{results.length !== 1 ? 's' : ''} for &ldquo;{query}&rdquo;
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {results.map(result => (
                <PaperCard
                  key={result.id}
                  result={result}
                  importState={importStates[result.id]?.state}
                  onImport={() => handleImport(result)}
                  onGoToChat={() => handleGoToChat(result.id)}
                />
              ))}
            </div>
          </>
        )}
      </div>

      {/* Footer */}
      <div className="fixed bottom-0 left-0 right-0 text-center py-4 pointer-events-none z-10">
        <p
          className="text-gray-700 text-[10px] uppercase tracking-[0.2em]"
          style={{ fontFamily: 'var(--font-mono)' }}
        >
          PaperMind · Live Research Discovery
        </p>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
        .line-clamp-3 {
          display: -webkit-box;
          -webkit-line-clamp: 3;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>
    </div>
  )
}
