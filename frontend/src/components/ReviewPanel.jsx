import { useState, useEffect, useCallback } from 'react'
import { reviewPaperStream } from '../api'
import { MetricRing } from './MetricRing'
import PDFPreviewPanel from './PDFPreviewPanel'

/* ── REVIEWER / WEAKNESS AUDIT PANEL ─────────────────────────────────
   Complement of AuditPanel: instead of "do the claims match the paper's own
   evidence?", this grades the paper against a generic ML/NLP venue's
   methodological norms (baselines, ablations, error bars, N, threats, related
   work). Same SSE + R2-cache architecture; OK / WEAK / MISSING per dimension. */
const REVIEW_STAGES = [
  { key: 'reading',   label: 'Reading paper'     },
  { key: 'gathering', label: 'Tracing evidence'  },
  { key: 'reviewing', label: 'Reviewing vs norms' },
]

const REVIEW_VERDICT_STYLE = {
  OK:      { label: 'Adequate', color: 'rgba(52,211,153,0.95)',  bg: 'rgba(52,211,153,0.08)',  border: 'rgba(52,211,153,0.25)',  dot: '#34d399' },
  WEAK:    { label: 'Weak',     color: 'rgba(251,191,36,0.97)',  bg: 'rgba(251,191,36,0.09)',  border: 'rgba(251,191,36,0.3)',   dot: '#fbbf24' },
  MISSING: { label: 'Missing',  color: 'rgba(248,113,113,0.97)', bg: 'rgba(248,113,113,0.09)', border: 'rgba(248,113,113,0.3)',  dot: '#f87171' },
}

function ReviewDimensionCard({ dim, onViewEvidence }) {
  const s = REVIEW_VERDICT_STYLE[dim.verdict] || REVIEW_VERDICT_STYLE.MISSING
  return (
    <div className="rounded-xl p-4" style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.05)' }}>
      <div className="flex items-center justify-between mb-2.5">
        <span className="text-[8px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full flex items-center gap-1.5"
          style={{ background: s.bg, border: `1px solid ${s.border}`, color: s.color }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: s.dot, boxShadow: `0 0 5px ${s.dot}` }} />
          {s.label}
        </span>
        <span className="text-[8px] uppercase tracking-wider text-gray-700" style={{ fontFamily: 'var(--font-mono)' }}>
          {dim.label}
        </span>
      </div>
      {dim.why && (
        <p className="text-gray-200 text-xs leading-relaxed mb-2">{dim.why}</p>
      )}
      {dim.reviewer_question && (
        <p className="text-[11px] leading-relaxed mb-2.5 italic" style={{ color: s.color }}>“{dim.reviewer_question}”</p>
      )}
      {dim.evidence && (
        <button onClick={() => onViewEvidence(dim)}
          className="flex items-center gap-1.5 text-[9px] uppercase tracking-wider font-bold px-2.5 py-1 rounded-lg transition-all"
          style={{ background: 'rgba(167,139,250,0.06)', border: '1px solid rgba(167,139,250,0.2)', color: 'rgba(167,139,250,0.85)' }}>
          <span className="w-1 h-1 rounded-full bg-violet-400" />
          {dim.evidence.section_type === 'table' ? 'TABLE · ' : ''}{dim.evidence.section} · p.{dim.evidence.page}
        </button>
      )}
    </div>
  )
}

export default function ReviewPanel({ paperId, onClose, embedded = false }) {
  const [report,   setReport]   = useState(null)
  const [loading,  setLoading]  = useState(true)
  const [progress, setProgress] = useState([])
  const [error,    setError]    = useState('')
  const [showOk,   setShowOk]   = useState(false)
  const [evidence, setEvidence] = useState(null)  // dimension whose evidence is open

  // Manual re-run (button / retry). setState in an event handler is fine.
  const run = useCallback((force = false) => {
    setLoading(true); setError(''); setProgress([]); setReport(null); setShowOk(false)
    reviewPaperStream(paperId, ({ type, data }) => {
      if (type === 'progress') setProgress(prev => [...prev, data.stage])
    }, force)
      .then(setReport)
      .catch(e => setError(e.message || 'Review failed'))
      .finally(() => setLoading(false))
  }, [paperId])

  // Initial load. Kept inline (not via run()) so no setState fires synchronously
  // in the effect body — initial state already is loading/empty. A cancel flag
  // drops late results if the panel closes or the paper switches mid-review.
  useEffect(() => {
    let cancelled = false
    reviewPaperStream(paperId, ({ type, data }) => {
      if (type === 'progress' && !cancelled) setProgress(prev => [...prev, data.stage])
    }, false)
      .then(r => { if (!cancelled) setReport(r) })
      .catch(e => { if (!cancelled) setError(e.message || 'Review failed') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [paperId])

  const latestStage = progress[progress.length - 1]
  const dims    = report?.dimensions || []
  const flagged = dims.filter(d => d.verdict !== 'OK')
  const passes  = dims.filter(d => d.verdict === 'OK')
  const pct = report?.review_score != null ? Math.round(report.review_score * 100) : null

  return (
    <>
      <div className={embedded ? 'flex flex-col w-full max-w-3xl mx-auto' : 'panel-slide-in fixed top-0 right-0 bottom-0 z-[80] flex flex-col'}
        style={embedded
          ? { minHeight: '60vh' }
          : { width: 420, background: 'rgba(4,4,16,0.97)', borderLeft: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(32px)' }}>
        <div className="flex items-center justify-between px-6 py-5 border-b" style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
          <div>
            <p className="text-[9px] uppercase tracking-[0.22em] font-bold mb-0.5" style={{ fontFamily: 'var(--font-mono)', color: 'rgba(167,139,250,0.6)' }}>Peer review</p>
            <h2 className="text-white font-semibold text-sm" style={{ fontFamily: 'var(--font-display)' }}>Weakness Audit</h2>
          </div>
          <div className="flex items-center gap-2">
            {report && !loading && (
              <button onClick={() => run(true)} title="Re-run review"
                className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-600 hover:text-violet-300 hover:bg-white/[0.06] transition-all">
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
                {REVIEW_STAGES.map(st => {
                  const seenIdx = progress.indexOf(st.key)
                  const isActive = latestStage === st.key
                  const isDone   = seenIdx !== -1 && !isActive
                  return (
                    <div key={st.key} className="flex items-center gap-3">
                      <div className="w-4 h-4 flex items-center justify-center flex-shrink-0">
                        {isDone   && <span className="w-2.5 h-2.5 rounded-full bg-violet-400/60" />}
                        {isActive && <span className="w-3 h-3 rounded-full bg-violet-400 animate-pulse" style={{ boxShadow: '0 0 8px rgba(167,139,250,0.8)' }} />}
                        {!isDone && !isActive && <span className="w-2 h-2 rounded-full border border-white/10" />}
                      </div>
                      <span className={`text-xs ${isActive ? 'text-violet-300' : isDone ? 'text-violet-400/55' : 'text-gray-700'}`}
                        style={{ fontFamily: 'var(--font-mono)' }}>{st.label}</span>
                    </div>
                  )
                })}
              </div>
              <p className="text-[10px] text-gray-700 mt-5 leading-relaxed">
                Grading the paper against a reviewer's methodological checklist — baselines, ablations, error bars, sample size, threats to validity, related work. This can take up to a minute.
              </p>
            </div>
          )}

          {error && !loading && (
            <div className="py-10 text-center">
              <p className="text-red-400/60 text-xs mb-4">{error}</p>
              <button onClick={() => run(true)}
                className="text-[10px] uppercase tracking-wider font-bold px-4 py-2 rounded-lg"
                style={{ background: 'rgba(167,139,250,0.08)', border: '1px solid rgba(167,139,250,0.25)', color: 'rgba(167,139,250,0.85)' }}>
                Try again
              </button>
            </div>
          )}

          {report && !loading && report.review_failed && (
            <p className="text-gray-600 text-xs text-center py-10">{report.reason || 'Could not review this paper.'}</p>
          )}

          {report && !loading && !report.review_failed && dims.length === 0 && (
            <p className="text-gray-600 text-xs text-center py-10">{report.reason || 'No review dimensions could be assessed for this paper.'}</p>
          )}

          {report && !loading && !report.review_failed && dims.length > 0 && (
            <>
              {/* Headline */}
              <div className="flex items-center gap-4 mb-6 rounded-2xl p-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                {pct != null && <MetricRing label="meets bar" value={pct} isPercentage accent="violet" />}
                <div className="flex-1">
                  <p className="text-white text-sm font-semibold mb-0.5" style={{ fontFamily: 'var(--font-display)' }}>
                    {report.dimensions_checked} dimension{report.dimensions_checked === 1 ? '' : 's'} reviewed
                  </p>
                  <p className="text-[11px] text-gray-500">
                    <span className="text-emerald-400/80">{report.ok} adequate</span>
                    {report.weak > 0 && <> · <span className="text-amber-400/80">{report.weak} weak</span></>}
                    {report.missing > 0 && <> · <span className="text-red-400/80">{report.missing} missing</span></>}
                  </p>
                  {report.cached && <p className="text-[8px] uppercase tracking-wider text-gray-700 mt-1.5" style={{ fontFamily: 'var(--font-mono)' }}>cached · re-run to refresh</p>}
                </div>
              </div>

              <p className="text-[10px] text-gray-700 leading-relaxed mb-5">
                A reviewer's-eye critique, not a verdict on the paper — retrieval can miss things, so open the evidence and judge for yourself.
              </p>

              {/* Flagged dimensions (the interesting ones) */}
              {flagged.length > 0 ? (
                <div className="space-y-3">
                  {flagged.map((d, i) => (
                    <ReviewDimensionCard key={i} dim={d} onViewEvidence={setEvidence} />
                  ))}
                </div>
              ) : (
                <p className="text-emerald-400/70 text-xs text-center py-4">The paper meets the bar on every reviewed dimension.</p>
              )}

              {/* Adequate dimensions behind a toggle */}
              {passes.length > 0 && (
                <div className="mt-5">
                  <button onClick={() => setShowOk(v => !v)}
                    className="text-[9px] uppercase tracking-wider text-gray-600 hover:text-gray-400 transition-colors font-bold">
                    {showOk ? 'Hide' : 'Show'} {passes.length} adequate dimension{passes.length === 1 ? '' : 's'}
                  </button>
                  {showOk && (
                    <div className="space-y-3 mt-3 animate-slide-down">
                      {passes.map((d, i) => (
                        <ReviewDimensionCard key={i} dim={d} onViewEvidence={setEvidence} />
                      ))}
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
