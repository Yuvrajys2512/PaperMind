import { useState, useEffect, useCallback } from 'react'
import { citationGapStream } from '../api'
import { MetricRing } from './MetricRing'
import PDFPreviewPanel from './PDFPreviewPanel'

/* ── CITATION GAP PANEL ──────────────────────────────────────────────
   Write-mode "[citation needed]" pass: statements that assert something
   citable but carry no citation marker. Each gap jumps back into the draft's
   own PDF (like the audits); a confirmed Semantic Scholar suggestion, when we
   have one, links OUT to the paper that may be the missing reference. */

const ACCENT = '#fb7185' // rose — distinct from review-violet / audit-cyan /
                         // novelty-fuchsia / venue-amber / numbers-emerald

const GAP_STAGES = [
  { key: 'reading',     label: 'Reading draft'      },
  { key: 'scanning',    label: 'Scanning citations' },
  { key: 'classifying', label: 'Judging statements' },
  { key: 'searching',   label: 'Finding references' },
]

const KIND_LABEL = {
  prior_work:        'Prior work',
  external_stat:     'External figure',
  attributed_method: 'Attributed method',
  comparison:        'External comparison',
  established_fact:  'Established fact',
}

const SEV_STYLE = {
  high: { color: 'rgba(248,113,113,0.97)', bg: 'rgba(248,113,113,0.09)', border: 'rgba(248,113,113,0.3)',  dot: '#f87171' },
  med:  { color: 'rgba(251,113,133,0.95)', bg: 'rgba(251,113,133,0.08)', border: 'rgba(251,113,133,0.26)', dot: '#fb7185' },
  low:  { color: 'rgba(156,163,175,0.95)', bg: 'rgba(156,163,175,0.06)', border: 'rgba(255,255,255,0.12)', dot: '#9ca3af' },
}

function GapCard({ gap, onViewEvidence }) {
  const s = SEV_STYLE[gap.severity] || SEV_STYLE.med
  const sug = gap.suggestion
  const sugMeta = sug ? [
    sug.authors?.length ? sug.authors.slice(0, 3).join(', ') + (sug.authors.length > 3 ? ' et al.' : '') : null,
    sug.year || null,
    sug.venue || null,
    sug.citations != null ? `${sug.citations} citations` : null,
  ].filter(Boolean).join(' · ') : ''

  return (
    <div className="rounded-xl p-4" style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.05)' }}>
      <div className="flex items-center justify-between mb-2.5 gap-2">
        <span className="text-[8px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full flex items-center gap-1.5 flex-shrink-0"
          style={{ background: s.bg, border: `1px solid ${s.border}`, color: s.color }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: s.dot, boxShadow: `0 0 5px ${s.dot}` }} />
          {KIND_LABEL[gap.kind] || 'Citation needed'}
        </span>
        <span className="text-[8px] uppercase tracking-wider text-gray-700 flex-shrink-0" style={{ fontFamily: 'var(--font-mono)' }}>
          {gap.section}
        </span>
      </div>

      <p className="text-gray-300 text-[11px] leading-relaxed mb-2">“{gap.statement}”</p>
      {gap.why && <p className="text-[11px] leading-relaxed mb-2.5" style={{ color: s.color }}>{gap.why}</p>}

      {/* A suggestion is only ever shown after a verification pass confirmed the
          paper actually supports the statement. */}
      {sug && (
        <div className="rounded-lg p-3 mb-2.5" style={{ background: 'rgba(251,113,133,0.05)', border: '1px solid rgba(251,113,133,0.16)' }}>
          <p className="text-[8px] uppercase tracking-wider font-bold mb-1.5" style={{ color: ACCENT, fontFamily: 'var(--font-mono)' }}>
            Possible reference
          </p>
          {sug.url ? (
            <a href={sug.url} target="_blank" rel="noopener noreferrer"
              className="text-gray-100 text-[11px] font-semibold leading-snug hover:text-white transition-colors inline-flex items-start gap-1.5">
              {sug.title}
              <svg className="w-3 h-3 mt-0.5 flex-shrink-0 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
              </svg>
            </a>
          ) : (
            <p className="text-gray-100 text-[11px] font-semibold leading-snug">{sug.title}</p>
          )}
          {sugMeta && <p className="text-[10px] text-gray-600 mt-1" style={{ fontFamily: 'var(--font-mono)' }}>{sugMeta}</p>}
          {sug.note && <p className="text-[10px] text-gray-500 leading-relaxed mt-1.5">{sug.note}</p>}
        </div>
      )}

      {gap.evidence && (
        <button onClick={() => onViewEvidence(gap)}
          className="flex items-center gap-1.5 text-[9px] uppercase tracking-wider font-bold px-2.5 py-1 rounded-lg transition-all"
          style={{ background: 'rgba(251,113,133,0.06)', border: '1px solid rgba(251,113,133,0.2)', color: 'rgba(251,113,133,0.85)' }}>
          <span className="w-1 h-1 rounded-full" style={{ background: ACCENT }} />
          {gap.evidence.section} · p.{gap.evidence.page}
        </button>
      )}
    </div>
  )
}

export default function CitationGapPanel({ paperId, onClose, embedded = false }) {
  const [report,   setReport]   = useState(null)
  const [loading,  setLoading]  = useState(true)
  const [progress, setProgress] = useState([])
  const [error,    setError]    = useState('')
  const [evidence, setEvidence] = useState(null)

  // Manual re-run (button / retry). setState in an event handler is fine.
  const run = useCallback((force = false) => {
    setLoading(true); setError(''); setProgress([]); setReport(null)
    citationGapStream(paperId, ({ type, data }) => {
      if (type === 'progress') setProgress(prev => [...prev, data.stage])
    }, force)
      .then(setReport)
      .catch(e => setError(e.message || 'Citation gap check failed'))
      .finally(() => setLoading(false))
  }, [paperId])

  // Initial load. Kept inline (not via run()) so no setState fires synchronously
  // in the effect body. A cancel flag drops late results if the panel closes or
  // the paper switches mid-check.
  useEffect(() => {
    let cancelled = false
    citationGapStream(paperId, ({ type, data }) => {
      if (type === 'progress' && !cancelled) setProgress(prev => [...prev, data.stage])
    }, false)
      .then(r => { if (!cancelled) setReport(r) })
      .catch(e => { if (!cancelled) setError(e.message || 'Citation gap check failed') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [paperId])

  const latestStage = progress[progress.length - 1]
  const gaps = report?.gaps || []
  const pct  = report?.coverage_score != null ? Math.round(report.coverage_score * 100) : null

  return (
    <>
      <div className={embedded ? 'flex flex-col w-full max-w-3xl mx-auto' : 'panel-slide-in fixed top-0 right-0 bottom-0 z-[80] flex flex-col'}
        style={embedded
          ? { minHeight: '60vh' }
          : { width: 420, background: 'rgba(4,4,16,0.97)', borderLeft: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(32px)' }}>
        <div className="flex items-center justify-between px-6 py-5 border-b" style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
          <div>
            <p className="text-[9px] uppercase tracking-[0.22em] font-bold mb-0.5" style={{ fontFamily: 'var(--font-mono)', color: 'rgba(251,113,133,0.6)' }}>Sourcing</p>
            <h2 className="text-white font-semibold text-sm" style={{ fontFamily: 'var(--font-display)' }}>Citation Gap Check</h2>
          </div>
          <div className="flex items-center gap-2">
            {report && !loading && (
              <button onClick={() => run(true)} title="Re-run check"
                className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-600 hover:text-rose-300 hover:bg-white/[0.06] transition-all">
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
                {GAP_STAGES.map(st => {
                  const seenIdx = progress.indexOf(st.key)
                  const isActive = latestStage === st.key
                  const isDone   = seenIdx !== -1 && !isActive
                  return (
                    <div key={st.key} className="flex items-center gap-3">
                      <div className="w-4 h-4 flex items-center justify-center flex-shrink-0">
                        {isDone   && <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'rgba(251,113,133,0.6)' }} />}
                        {isActive && <span className="w-3 h-3 rounded-full animate-pulse" style={{ background: ACCENT, boxShadow: '0 0 8px rgba(251,113,133,0.8)' }} />}
                        {!isDone && !isActive && <span className="w-2 h-2 rounded-full border border-white/10" />}
                      </div>
                      <span className={`text-xs ${isActive ? 'text-rose-300' : isDone ? 'text-rose-400/55' : 'text-gray-700'}`}
                        style={{ fontFamily: 'var(--font-mono)' }}>{st.label}</span>
                    </div>
                  )
                })}
              </div>
              <p className="text-[10px] text-gray-700 mt-5 leading-relaxed">
                Looking for statements that need a source, then searching for the references you may be missing. This can take up to a minute.
              </p>
            </div>
          )}

          {error && !loading && (
            <div className="py-10 text-center">
              <p className="text-red-400/60 text-xs mb-4">{error}</p>
              <button onClick={() => run(true)}
                className="text-[10px] uppercase tracking-wider font-bold px-4 py-2 rounded-lg"
                style={{ background: 'rgba(251,113,133,0.08)', border: '1px solid rgba(251,113,133,0.25)', color: 'rgba(251,113,133,0.85)' }}>
                Try again
              </button>
            </div>
          )}

          {report && !loading && report.audit_failed && (
            <p className="text-gray-600 text-xs text-center py-10">{report.reason || 'Could not check this draft.'}</p>
          )}

          {report && !loading && !report.audit_failed && (
            <>
              {/* Headline */}
              <div className="flex items-center gap-4 mb-5 rounded-2xl p-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                {pct != null && <MetricRing label="sourced" value={pct} isPercentage accent="rose"
                  tooltip={{ title: 'Sourced', body: 'Of the statements that appear to need a source, how many already carry a citation. Sentences we judged to be common knowledge are excluded from both sides — they never needed one.' }} />}
                <div className="flex-1">
                  <p className="text-white text-sm font-semibold mb-0.5" style={{ fontFamily: 'var(--font-display)' }}>
                    {report.gaps_found} citation gap{report.gaps_found === 1 ? '' : 's'}
                  </p>
                  <p className="text-[11px] text-gray-500">
                    {report.cited_sentences} cited · {report.statements_checked} uncited statement{report.statements_checked === 1 ? '' : 's'} checked
                    {report.suggestions_found > 0 && <span style={{ color: ACCENT }}> · {report.suggestions_found} reference{report.suggestions_found === 1 ? '' : 's'} suggested</span>}
                  </p>
                  {report.cached && <p className="text-[8px] uppercase tracking-wider text-gray-700 mt-1.5" style={{ fontFamily: 'var(--font-mono)' }}>cached · re-run to refresh</p>}
                </div>
              </div>

              {gaps.length === 0 && (
                <p className="text-rose-300/70 text-xs text-center py-6 leading-relaxed">
                  {report.reason || 'No uncited statements looked like they needed a source.'}
                </p>
              )}

              {gaps.length > 0 && (
                <>
                  <p className="text-[10px] text-gray-700 leading-relaxed mb-4">
                    Flags are prompts to double-check, not verdicts — statements about your own contribution and common knowledge are skipped, but only you know your field's conventions.
                  </p>
                  <div className="space-y-3">
                    {gaps.map((g, i) => <GapCard key={i} gap={g} onViewEvidence={setEvidence} />)}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>

      {evidence?.evidence && (
        <PDFPreviewPanel
          pdfUrl={`/api/papers/${paperId}/pdf#page=${evidence.evidence.page}`}
          title={evidence.evidence.section}
          page={evidence.evidence.page}
          highlight={evidence.statement}
          onClose={() => setEvidence(null)}
        />
      )}
    </>
  )
}
