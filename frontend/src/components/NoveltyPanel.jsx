import { useState, useEffect, useCallback } from 'react'
import { noveltyScanStream } from '../api'

/* ── NOVELTY / RELATED-WORK PANEL ────────────────────────────────────
   Write-mode literature scan: distils the draft's abstract into search
   queries, matches against Semantic Scholar, and rates each neighbour for
   how closely it overlaps the draft. Unlike the audits, results link OUT to
   real papers rather than back into the draft's own PDF. */

const ACCENT = '#f472b6' // fuchsia — distinct from review (violet) / audit (cyan)

const NOVELTY_STAGES = [
  { key: 'reading',      label: 'Reading draft'        },
  { key: 'querying',     label: 'Distilling queries'   },
  { key: 'searching',    label: 'Searching literature' },
  { key: 'comparing',    label: 'Comparing neighbours' },
  { key: 'synthesizing', label: 'Summarizing'          },
]

const CLOSENESS_STYLE = {
  HIGH:   { label: 'Direct comparison', color: 'rgba(251,113,133,0.97)', bg: 'rgba(251,113,133,0.09)', border: 'rgba(251,113,133,0.3)',  dot: '#fb7185' },
  MEDIUM: { label: 'Related',           color: 'rgba(251,191,36,0.97)',  bg: 'rgba(251,191,36,0.09)',  border: 'rgba(251,191,36,0.3)',   dot: '#fbbf24' },
  LOW:    { label: 'Background',         color: 'rgba(156,163,175,0.95)', bg: 'rgba(156,163,175,0.06)', border: 'rgba(255,255,255,0.12)', dot: '#9ca3af' },
}

function RelatedCard({ paper }) {
  const s = CLOSENESS_STYLE[paper.closeness] || CLOSENESS_STYLE.LOW
  const meta = [
    paper.authors?.length ? paper.authors.slice(0, 3).join(', ') + (paper.authors.length > 3 ? ' et al.' : '') : null,
    paper.year || null,
    paper.venue || null,
    paper.citations != null ? `${paper.citations} citations` : null,
  ].filter(Boolean).join(' · ')

  return (
    <div className="rounded-xl p-4" style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.05)' }}>
      <div className="flex items-center justify-between mb-2.5 gap-3">
        <span className="text-[8px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full flex items-center gap-1.5 flex-shrink-0"
          style={{ background: s.bg, border: `1px solid ${s.border}`, color: s.color }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: s.dot, boxShadow: `0 0 5px ${s.dot}` }} />
          {s.label}
        </span>
      </div>

      {paper.url ? (
        <a href={paper.url} target="_blank" rel="noopener noreferrer"
          className="text-gray-100 text-xs font-semibold leading-snug hover:text-white transition-colors inline-flex items-start gap-1.5">
          {paper.title}
          <svg className="w-3 h-3 mt-0.5 flex-shrink-0 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
          </svg>
        </a>
      ) : (
        <p className="text-gray-100 text-xs font-semibold leading-snug">{paper.title}</p>
      )}

      {meta && <p className="text-[10px] text-gray-600 mt-1" style={{ fontFamily: 'var(--font-mono)' }}>{meta}</p>}

      {paper.overlap && (
        <p className="text-[11px] leading-relaxed mt-2.5">
          <span className="text-gray-600 uppercase tracking-wider text-[8px] font-bold mr-1.5">Overlap</span>
          <span className="text-gray-300">{paper.overlap}</span>
        </p>
      )}
      {paper.differentiator && (
        <p className="text-[11px] leading-relaxed mt-1.5">
          <span className="uppercase tracking-wider text-[8px] font-bold mr-1.5" style={{ color: ACCENT }}>Your angle</span>
          <span className="text-gray-400">{paper.differentiator}</span>
        </p>
      )}
    </div>
  )
}

export default function NoveltyPanel({ paperId, onClose, embedded = false }) {
  const [report,   setReport]   = useState(null)
  const [loading,  setLoading]  = useState(true)
  const [progress, setProgress] = useState([])
  const [error,    setError]    = useState('')

  // Manual re-run (button / retry). setState in an event handler is fine.
  const run = useCallback((force = false) => {
    setLoading(true); setError(''); setProgress([]); setReport(null)
    noveltyScanStream(paperId, ({ type, data }) => {
      if (type === 'progress') setProgress(prev => [...prev, data.stage])
    }, force)
      .then(setReport)
      .catch(e => setError(e.message || 'Novelty scan failed'))
      .finally(() => setLoading(false))
  }, [paperId])

  // Initial load. Kept inline (not via run()) so no setState fires synchronously
  // in the effect body. A cancel flag drops late results if the panel closes or
  // the paper switches mid-scan.
  useEffect(() => {
    let cancelled = false
    noveltyScanStream(paperId, ({ type, data }) => {
      if (type === 'progress' && !cancelled) setProgress(prev => [...prev, data.stage])
    }, false)
      .then(r => { if (!cancelled) setReport(r) })
      .catch(e => { if (!cancelled) setError(e.message || 'Novelty scan failed') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [paperId])

  const latestStage = progress[progress.length - 1]
  const related = report?.related || []
  const counts  = report?.closeness_counts || {}

  return (
    <div className={embedded ? 'flex flex-col w-full max-w-3xl mx-auto' : 'panel-slide-in fixed top-0 right-0 bottom-0 z-[80] flex flex-col'}
      style={embedded
        ? { minHeight: '60vh' }
        : { width: 420, background: 'rgba(4,4,16,0.97)', borderLeft: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(32px)' }}>
      <div className="flex items-center justify-between px-6 py-5 border-b" style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
        <div>
          <p className="text-[9px] uppercase tracking-[0.22em] font-bold mb-0.5" style={{ fontFamily: 'var(--font-mono)', color: 'rgba(244,114,182,0.6)' }}>Positioning</p>
          <h2 className="text-white font-semibold text-sm" style={{ fontFamily: 'var(--font-display)' }}>Novelty Scan</h2>
        </div>
        <div className="flex items-center gap-2">
          {report && !loading && (
            <button onClick={() => run(true)} title="Re-run scan"
              className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-600 hover:text-pink-300 hover:bg-white/[0.06] transition-all">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          )}
          <button onClick={onClose} className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-600 hover:text-white hover:bg-white/[0.06] transition-all">×</button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-5 custom-scrollbar">
        {/* Loading — staged checklist */}
        {loading && (
          <div className="py-4">
            <div className="space-y-3">
              {NOVELTY_STAGES.map(st => {
                const seenIdx = progress.indexOf(st.key)
                const isActive = latestStage === st.key
                const isDone   = seenIdx !== -1 && !isActive
                return (
                  <div key={st.key} className="flex items-center gap-3">
                    <div className="w-4 h-4 flex items-center justify-center flex-shrink-0">
                      {isDone   && <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'rgba(244,114,182,0.6)' }} />}
                      {isActive && <span className="w-3 h-3 rounded-full animate-pulse" style={{ background: ACCENT, boxShadow: '0 0 8px rgba(244,114,182,0.8)' }} />}
                      {!isDone && !isActive && <span className="w-2 h-2 rounded-full border border-white/10" />}
                    </div>
                    <span className={`text-xs ${isActive ? 'text-pink-300' : isDone ? 'text-pink-400/55' : 'text-gray-700'}`}
                      style={{ fontFamily: 'var(--font-mono)' }}>{st.label}</span>
                  </div>
                )
              })}
            </div>
            <p className="text-[10px] text-gray-700 mt-5 leading-relaxed">
              Searching the literature for the prior work closest to your draft. This can take up to a minute.
            </p>
          </div>
        )}

        {error && !loading && (
          <div className="py-10 text-center">
            <p className="text-red-400/60 text-xs mb-4">{error}</p>
            <button onClick={() => run(true)}
              className="text-[10px] uppercase tracking-wider font-bold px-4 py-2 rounded-lg"
              style={{ background: 'rgba(244,114,182,0.08)', border: '1px solid rgba(244,114,182,0.25)', color: 'rgba(244,114,182,0.85)' }}>
              Try again
            </button>
          </div>
        )}

        {report && !loading && report.scan_failed && (
          <p className="text-gray-600 text-xs text-center py-10">{report.reason || 'Could not scan this draft.'}</p>
        )}

        {report && !loading && !report.scan_failed && related.length === 0 && (
          <p className="text-gray-500 text-xs text-center py-10 leading-relaxed">
            {report.reason || 'No closely related papers surfaced — the space around this specific angle looks relatively open.'}
          </p>
        )}

        {report && !loading && !report.scan_failed && related.length > 0 && (
          <>
            {/* Headline */}
            <div className="rounded-2xl p-4 mb-5" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
              <p className="text-white text-sm font-semibold mb-0.5" style={{ fontFamily: 'var(--font-display)' }}>
                {related.length} related paper{related.length === 1 ? '' : 's'} found
              </p>
              <p className="text-[11px] text-gray-500">
                {counts.high > 0 && <span style={{ color: 'rgba(251,113,133,0.85)' }}>{counts.high} direct</span>}
                {counts.high > 0 && (counts.medium > 0 || counts.low > 0) && ' · '}
                {counts.medium > 0 && <span className="text-amber-400/80">{counts.medium} related</span>}
                {counts.medium > 0 && counts.low > 0 && ' · '}
                {counts.low > 0 && <span className="text-gray-500">{counts.low} background</span>}
              </p>
              {report.cached && <p className="text-[8px] uppercase tracking-wider text-gray-700 mt-1.5" style={{ fontFamily: 'var(--font-mono)' }}>cached · re-run to refresh</p>}
            </div>

            {/* Novelty synthesis */}
            {report.summary && (
              <div className="rounded-xl p-4 mb-5" style={{ background: 'rgba(244,114,182,0.05)', border: '1px solid rgba(244,114,182,0.15)' }}>
                <p className="text-[9px] uppercase tracking-wider font-bold mb-2" style={{ color: ACCENT, fontFamily: 'var(--font-mono)' }}>Where your draft sits</p>
                <p className="text-gray-200 text-xs leading-relaxed">{report.summary}</p>
                {report.positioning?.length > 0 && (
                  <ul className="mt-3 space-y-1.5">
                    {report.positioning.map((p, i) => (
                      <li key={i} className="text-[11px] text-gray-400 leading-relaxed flex gap-2">
                        <span style={{ color: ACCENT }}>→</span>
                        <span>{p}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            <p className="text-[10px] text-gray-700 leading-relaxed mb-4">
              Closeness reflects abstract overlap only — open each paper and judge novelty for yourself.
            </p>

            <div className="space-y-3">
              {related.map((p, i) => <RelatedCard key={i} paper={p} />)}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
