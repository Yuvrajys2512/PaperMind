import { useState, useEffect, useCallback } from 'react'
import { numbersCheckStream } from '../api'
import { MetricRing } from './MetricRing'
import PDFPreviewPanel from './PDFPreviewPanel'

/* ── NUMBERS CONSISTENCY PANEL ───────────────────────────────────────
   Reconciles the figures stated in the abstract/intro against the results
   tables. MISMATCH is the money verdict (claimed ≠ found); NOT_FOUND is a
   nudge to check manually; MATCH is reassurance. */

const ACCENT = '#34d399' // emerald

const NUMBERS_STAGES = [
  { key: 'reading',    label: 'Reading paper'      },
  { key: 'extracting', label: 'Extracting numbers' },
  { key: 'gathering',  label: 'Locating results'   },
  { key: 'checking',   label: 'Reconciling'        },
]

const VERDICT_STYLE = {
  MATCH:     { label: 'Match',      color: 'rgba(52,211,153,0.95)',  bg: 'rgba(52,211,153,0.08)',  border: 'rgba(52,211,153,0.25)',  dot: '#34d399' },
  MISMATCH:  { label: 'Mismatch',   color: 'rgba(248,113,113,0.97)', bg: 'rgba(248,113,113,0.09)', border: 'rgba(248,113,113,0.3)',  dot: '#f87171' },
  NOT_FOUND: { label: 'Not found',  color: 'rgba(156,163,175,0.95)', bg: 'rgba(156,163,175,0.06)', border: 'rgba(255,255,255,0.12)', dot: '#9ca3af' },
}

function NumberLabel({ n }) {
  const parts = [n.value, n.metric, n.dataset ? `on ${n.dataset}` : ''].filter(Boolean).join(' ')
  return <span className="text-gray-100 text-xs font-semibold" style={{ fontFamily: 'var(--font-mono)' }}>{parts}</span>
}

function NumberCard({ n, onViewEvidence }) {
  const s = VERDICT_STYLE[n.verdict] || VERDICT_STYLE.NOT_FOUND
  return (
    <div className="rounded-xl p-4" style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.05)' }}>
      <div className="flex items-center justify-between mb-2.5 gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-[8px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full flex items-center gap-1.5 flex-shrink-0"
            style={{ background: s.bg, border: `1px solid ${s.border}`, color: s.color }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: s.dot, boxShadow: `0 0 5px ${s.dot}` }} />
            {s.label}
          </span>
          <NumberLabel n={n} />
        </div>
        <span className="text-[8px] uppercase tracking-wider text-gray-700 flex-shrink-0" style={{ fontFamily: 'var(--font-mono)' }}>
          {n.source_section}
        </span>
      </div>

      {/* The discrepancy, front and center for a mismatch */}
      {n.verdict === 'MISMATCH' && n.found_value && (
        <div className="flex items-center gap-2 mb-2.5 text-xs" style={{ fontFamily: 'var(--font-mono)' }}>
          <span className="px-2 py-1 rounded-lg" style={{ background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)', color: 'rgba(248,113,113,0.95)' }}>
            abstract: {n.value}
          </span>
          <svg className="w-3.5 h-3.5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
          </svg>
          <span className="px-2 py-1 rounded-lg" style={{ background: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.2)', color: 'rgba(52,211,153,0.9)' }}>
            results: {n.found_value}
          </span>
        </div>
      )}

      <p className="text-gray-400 text-[11px] leading-relaxed mb-2">“{n.claim}”</p>
      {n.why && <p className="text-[11px] leading-relaxed mb-2.5" style={{ color: s.color }}>{n.why}</p>}

      {n.evidence && (
        <button onClick={() => onViewEvidence(n)}
          className="flex items-center gap-1.5 text-[9px] uppercase tracking-wider font-bold px-2.5 py-1 rounded-lg transition-all"
          style={{ background: 'rgba(52,211,153,0.06)', border: '1px solid rgba(52,211,153,0.2)', color: 'rgba(52,211,153,0.85)' }}>
          <span className="w-1 h-1 rounded-full" style={{ background: ACCENT }} />
          {n.evidence.section_type === 'table' ? 'TABLE · ' : ''}{n.evidence.section} · p.{n.evidence.page}
        </button>
      )}
    </div>
  )
}

export default function NumbersPanel({ paperId, onClose, embedded = false }) {
  const [report,   setReport]   = useState(null)
  const [loading,  setLoading]  = useState(true)
  const [progress, setProgress] = useState([])
  const [error,    setError]    = useState('')
  const [showMatched, setShowMatched] = useState(false)
  const [evidence, setEvidence] = useState(null)

  const run = useCallback((force = false) => {
    setLoading(true); setError(''); setProgress([]); setReport(null); setShowMatched(false)
    numbersCheckStream(paperId, ({ type, data }) => {
      if (type === 'progress') setProgress(prev => [...prev, data.stage])
    }, force)
      .then(setReport)
      .catch(e => setError(e.message || 'Numbers check failed'))
      .finally(() => setLoading(false))
  }, [paperId])

  useEffect(() => {
    let cancelled = false
    numbersCheckStream(paperId, ({ type, data }) => {
      if (type === 'progress' && !cancelled) setProgress(prev => [...prev, data.stage])
    }, false)
      .then(r => { if (!cancelled) setReport(r) })
      .catch(e => { if (!cancelled) setError(e.message || 'Numbers check failed') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [paperId])

  const latestStage = progress[progress.length - 1]
  const numbers   = report?.numbers || []
  const mismatch  = numbers.filter(n => n.verdict === 'MISMATCH')
  const notFound  = numbers.filter(n => n.verdict === 'NOT_FOUND')
  const matched   = numbers.filter(n => n.verdict === 'MATCH')
  const pct = report?.consistency_score != null ? Math.round(report.consistency_score * 100) : null

  return (
    <>
      <div className={embedded ? 'flex flex-col w-full max-w-3xl mx-auto' : 'panel-slide-in fixed top-0 right-0 bottom-0 z-[80] flex flex-col'}
        style={embedded
          ? { minHeight: '60vh' }
          : { width: 420, background: 'rgba(4,4,16,0.97)', borderLeft: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(32px)' }}>
        <div className="flex items-center justify-between px-6 py-5 border-b" style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
          <div>
            <p className="text-[9px] uppercase tracking-[0.22em] font-bold mb-0.5" style={{ fontFamily: 'var(--font-mono)', color: 'rgba(52,211,153,0.6)' }}>Consistency</p>
            <h2 className="text-white font-semibold text-sm" style={{ fontFamily: 'var(--font-display)' }}>Numbers Check</h2>
          </div>
          <div className="flex items-center gap-2">
            {report && !loading && (
              <button onClick={() => run(true)} title="Re-run check"
                className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-600 hover:text-emerald-300 hover:bg-white/[0.06] transition-all">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            )}
            <button onClick={onClose} className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-600 hover:text-white hover:bg-white/[0.06] transition-all">×</button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-5 custom-scrollbar">
          {loading && (
            <div className="py-4">
              <div className="space-y-3">
                {NUMBERS_STAGES.map(st => {
                  const seenIdx = progress.indexOf(st.key)
                  const isActive = latestStage === st.key
                  const isDone   = seenIdx !== -1 && !isActive
                  return (
                    <div key={st.key} className="flex items-center gap-3">
                      <div className="w-4 h-4 flex items-center justify-center flex-shrink-0">
                        {isDone   && <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'rgba(52,211,153,0.6)' }} />}
                        {isActive && <span className="w-3 h-3 rounded-full animate-pulse" style={{ background: ACCENT, boxShadow: '0 0 8px rgba(52,211,153,0.8)' }} />}
                        {!isDone && !isActive && <span className="w-2 h-2 rounded-full border border-white/10" />}
                      </div>
                      <span className={`text-xs ${isActive ? 'text-emerald-300' : isDone ? 'text-emerald-400/55' : 'text-gray-700'}`}
                        style={{ fontFamily: 'var(--font-mono)' }}>{st.label}</span>
                    </div>
                  )
                })}
              </div>
              <p className="text-[10px] text-gray-700 mt-5 leading-relaxed">
                Matching each number in your abstract against your results tables. This can take up to a minute.
              </p>
            </div>
          )}

          {error && !loading && (
            <div className="py-10 text-center">
              <p className="text-red-400/60 text-xs mb-4">{error}</p>
              <button onClick={() => run(true)}
                className="text-[10px] uppercase tracking-wider font-bold px-4 py-2 rounded-lg"
                style={{ background: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.25)', color: 'rgba(52,211,153,0.85)' }}>
                Try again
              </button>
            </div>
          )}

          {report && !loading && report.audit_failed && (
            <p className="text-gray-600 text-xs text-center py-10">{report.reason || 'Could not check this paper.'}</p>
          )}

          {report && !loading && !report.audit_failed && numbers.length === 0 && (
            <p className="text-gray-600 text-xs text-center py-10">{report.reason || 'No checkable numbers found in the abstract/intro.'}</p>
          )}

          {report && !loading && !report.audit_failed && numbers.length > 0 && (
            <>
              {/* Headline */}
              <div className="flex items-center gap-4 mb-5 rounded-2xl p-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                {pct != null && <MetricRing label="consistent" value={pct} isPercentage accent="emerald" />}
                <div className="flex-1">
                  <p className="text-white text-sm font-semibold mb-0.5" style={{ fontFamily: 'var(--font-display)' }}>
                    {report.numbers_checked} number{report.numbers_checked === 1 ? '' : 's'} checked
                  </p>
                  <p className="text-[11px] text-gray-500">
                    {report.mismatched > 0 && <span className="text-red-400/80">{report.mismatched} mismatch{report.mismatched === 1 ? '' : 'es'}</span>}
                    {report.mismatched > 0 && (report.matched > 0 || report.not_found > 0) && ' · '}
                    {report.matched > 0 && <span className="text-emerald-400/80">{report.matched} match</span>}
                    {report.matched > 0 && report.not_found > 0 && ' · '}
                    {report.not_found > 0 && <span className="text-gray-500">{report.not_found} not found</span>}
                  </p>
                  {report.cached && <p className="text-[8px] uppercase tracking-wider text-gray-700 mt-1.5" style={{ fontFamily: 'var(--font-mono)' }}>cached · re-run to refresh</p>}
                </div>
              </div>

              {/* Mismatch banner — the headline risk */}
              {mismatch.length > 0 && (
                <div className="rounded-xl p-3.5 mb-5 flex items-start gap-2.5" style={{ background: 'rgba(248,113,113,0.07)', border: '1px solid rgba(248,113,113,0.22)' }}>
                  <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="none" stroke="rgba(248,113,113,0.9)" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01M5.07 19h13.86a2 2 0 001.71-3l-6.93-12a2 2 0 00-3.42 0l-6.93 12a2 2 0 001.71 3z" />
                  </svg>
                  <p className="text-[11px] leading-relaxed" style={{ color: 'rgba(248,113,113,0.92)' }}>
                    {mismatch.length} number{mismatch.length === 1 ? '' : 's'} in your abstract don't match the results — verify before submitting.
                  </p>
                </div>
              )}

              <p className="text-[10px] text-gray-700 leading-relaxed mb-4">
                Flags are prompts to double-check, not verdicts — always open the evidence and confirm for yourself.
              </p>

              {/* Mismatches first */}
              {mismatch.length > 0 && (
                <div className="space-y-3 mb-4">
                  {mismatch.map((n, i) => <NumberCard key={i} n={n} onViewEvidence={setEvidence} />)}
                </div>
              )}

              {/* Not-found (couldn't locate in results — worth a manual check) */}
              {notFound.length > 0 && (
                <div className="space-y-3 mb-4">
                  {notFound.map((n, i) => <NumberCard key={i} n={n} onViewEvidence={setEvidence} />)}
                </div>
              )}

              {mismatch.length === 0 && notFound.length === 0 && (
                <p className="text-emerald-400/70 text-xs text-center py-4">Every number matches the results.</p>
              )}

              {/* Matched behind a toggle */}
              {matched.length > 0 && (
                <div className="mt-2">
                  <button onClick={() => setShowMatched(v => !v)}
                    className="text-[9px] uppercase tracking-wider text-gray-600 hover:text-gray-400 transition-colors font-bold">
                    {showMatched ? 'Hide' : 'Show'} {matched.length} matched number{matched.length === 1 ? '' : 's'}
                  </button>
                  {showMatched && (
                    <div className="space-y-3 mt-3 animate-slide-down">
                      {matched.map((n, i) => <NumberCard key={i} n={n} onViewEvidence={setEvidence} />)}
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {evidence?.evidence && (
        <PDFPreviewPanel
          pdfUrl={`/api/papers/${paperId}/pdf#page=${evidence.evidence.page}`}
          title={`${evidence.evidence.section_type === 'table' ? 'TABLE · ' : ''}${evidence.evidence.section}`}
          page={evidence.evidence.page}
          highlight={evidence.evidence.quote}
          onClose={() => setEvidence(null)}
        />
      )}
    </>
  )
}
