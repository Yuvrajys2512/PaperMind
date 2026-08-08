import { useState, useEffect, useCallback } from 'react'
import { overlapStream } from '../api'
import { MetricRing } from './MetricRing'
import PDFPreviewPanel from './PDFPreviewPanel'

/* ── OVERLAP PANEL ───────────────────────────────────────────────────
   Write-mode overlap check: passages of the draft that also appear in the
   other papers in the user's library. Each match shows BOTH sides — the
   draft's text and the source's text — because the whole value of this tab is
   that the author can see the shared string and judge it themselves.

   UI stance, and it matters more here than on any other tab: this is not an
   accusation panel. The copy never says "plagiarism", matches the author
   already quoted or cited are shown as a separate, calm "attributed" group
   rather than as findings, and the paraphrase tier is explicitly labelled as
   the weaker signal. */

const ACCENT = '#60a5fa' // blue — distinct from review-violet / audit-cyan /
                         // novelty-fuchsia / venue-amber / numbers-emerald /
                         // citation-rose

const OVERLAP_STAGES = [
  { key: 'reading',    label: 'Reading draft'     },
  { key: 'indexing',   label: 'Indexing library'  },
  { key: 'matching',   label: 'Comparing text'    },
  { key: 'paraphrase', label: 'Checking rewrites' },
]

const VERDICT_STYLE = {
  VERBATIM:      { label: 'Verbatim',    color: 'rgba(248,113,113,0.97)', bg: 'rgba(248,113,113,0.09)', border: 'rgba(248,113,113,0.3)',  dot: '#f87171' },
  NEAR_VERBATIM: { label: 'Near-verbatim', color: 'rgba(251,191,36,0.95)', bg: 'rgba(251,191,36,0.08)',  border: 'rgba(251,191,36,0.26)', dot: '#fbbf24' },
  PARAPHRASE:    { label: 'Possible rewrite', color: 'rgba(96,165,250,0.95)', bg: 'rgba(96,165,250,0.07)', border: 'rgba(96,165,250,0.24)', dot: '#60a5fa' },
  ATTRIBUTED:    { label: 'Quoted / cited', color: 'rgba(156,163,175,0.95)', bg: 'rgba(156,163,175,0.06)', border: 'rgba(255,255,255,0.12)', dot: '#9ca3af' },
}

function MatchCard({ match, onViewEvidence }) {
  const s = VERDICT_STYLE[match.verdict] || VERDICT_STYLE.NEAR_VERBATIM
  const src = match.source || {}

  return (
    <div className="rounded-xl p-4" style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.05)' }}>
      <div className="flex items-center justify-between mb-2.5 gap-2">
        <span className="text-[8px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full flex items-center gap-1.5 flex-shrink-0"
          style={{ background: s.bg, border: `1px solid ${s.border}`, color: s.color }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: s.dot, boxShadow: `0 0 5px ${s.dot}` }} />
          {s.label}
          {match.words > 0 && <span className="opacity-70">· {match.words} words</span>}
          {match.verdict === 'PARAPHRASE' && match.similarity != null && (
            <span className="opacity-70">· {Math.round(match.similarity * 100)}% similar</span>
          )}
        </span>
        <span className="text-[8px] uppercase tracking-wider text-gray-700 flex-shrink-0" style={{ fontFamily: 'var(--font-mono)' }}>
          {match.section}
        </span>
      </div>

      {/* Your text */}
      <p className="text-[8px] uppercase tracking-wider font-bold mb-1" style={{ color: 'rgba(255,255,255,0.28)', fontFamily: 'var(--font-mono)' }}>
        In your draft
      </p>
      <p className="text-gray-300 text-[11px] leading-relaxed mb-3">“{match.excerpt}”</p>

      {/* Their text */}
      <div className="rounded-lg p-3 mb-2.5" style={{ background: 'rgba(96,165,250,0.05)', border: '1px solid rgba(96,165,250,0.16)' }}>
        <p className="text-[8px] uppercase tracking-wider font-bold mb-1.5" style={{ color: ACCENT, fontFamily: 'var(--font-mono)' }}>
          In “{src.title}”
        </p>
        <p className="text-gray-400 text-[11px] leading-relaxed">“{src.excerpt}”</p>
        <p className="text-[10px] text-gray-600 mt-1.5" style={{ fontFamily: 'var(--font-mono)' }}>
          {src.section}{src.page != null ? ` · p.${src.page}` : ''}
        </p>
      </div>

      {match.note && <p className="text-[11px] leading-relaxed mb-2.5" style={{ color: s.color }}>{match.note}</p>}

      {match.evidence && (
        <button onClick={() => onViewEvidence(match)}
          className="flex items-center gap-1.5 text-[9px] uppercase tracking-wider font-bold px-2.5 py-1 rounded-lg transition-all"
          style={{ background: 'rgba(96,165,250,0.06)', border: '1px solid rgba(96,165,250,0.2)', color: 'rgba(96,165,250,0.85)' }}>
          <span className="w-1 h-1 rounded-full" style={{ background: ACCENT }} />
          {match.evidence.section} · p.{match.evidence.page}
        </button>
      )}
    </div>
  )
}

export default function OverlapPanel({ paperId, onClose, embedded = false }) {
  const [report,   setReport]   = useState(null)
  const [loading,  setLoading]  = useState(true)
  const [progress, setProgress] = useState([])
  const [error,    setError]    = useState('')
  const [evidence, setEvidence] = useState(null)

  // Manual re-run (button / retry). setState in an event handler is fine.
  const run = useCallback((force = false) => {
    setLoading(true); setError(''); setProgress([]); setReport(null)
    overlapStream(paperId, ({ type, data }) => {
      if (type === 'progress') setProgress(prev => [...prev, data.stage])
    }, force)
      .then(setReport)
      .catch(e => setError(e.message || 'Overlap check failed'))
      .finally(() => setLoading(false))
  }, [paperId])

  // Initial load. Kept inline (not via run()) so no setState fires synchronously
  // in the effect body. A cancel flag drops late results if the panel closes or
  // the paper switches mid-check.
  useEffect(() => {
    let cancelled = false
    overlapStream(paperId, ({ type, data }) => {
      if (type === 'progress' && !cancelled) setProgress(prev => [...prev, data.stage])
    }, false)
      .then(r => { if (!cancelled) setReport(r) })
      .catch(e => { if (!cancelled) setError(e.message || 'Overlap check failed') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [paperId])

  const latestStage = progress[progress.length - 1]
  const matches = report?.matches || []
  const flagged    = matches.filter(m => m.verdict !== 'ATTRIBUTED')
  const attributed = matches.filter(m => m.verdict === 'ATTRIBUTED')
  const pct = report?.originality_score != null ? Math.round(report.originality_score * 100) : null

  return (
    <>
      <div className={embedded ? 'flex flex-col w-full max-w-3xl mx-auto' : 'panel-slide-in fixed top-0 right-0 bottom-0 z-[80] flex flex-col'}
        style={embedded
          ? { minHeight: '60vh' }
          : { width: 420, background: 'rgba(4,4,16,0.97)', borderLeft: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(32px)' }}>
        <div className="flex items-center justify-between px-6 py-5 border-b" style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
          <div>
            <p className="text-[9px] uppercase tracking-[0.22em] font-bold mb-0.5" style={{ fontFamily: 'var(--font-mono)', color: 'rgba(96,165,250,0.6)' }}>Originality</p>
            <h2 className="text-white font-semibold text-sm" style={{ fontFamily: 'var(--font-display)' }}>Overlap Check</h2>
          </div>
          <div className="flex items-center gap-2">
            {report && !loading && (
              <button onClick={() => run(true)} title="Re-run check"
                className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-600 hover:text-blue-300 hover:bg-white/[0.06] transition-all">
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
                {OVERLAP_STAGES.map(st => {
                  const seenIdx = progress.indexOf(st.key)
                  const isActive = latestStage === st.key
                  const isDone   = seenIdx !== -1 && !isActive
                  return (
                    <div key={st.key} className="flex items-center gap-3">
                      <div className="w-4 h-4 flex items-center justify-center flex-shrink-0">
                        {isDone   && <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'rgba(96,165,250,0.6)' }} />}
                        {isActive && <span className="w-3 h-3 rounded-full animate-pulse" style={{ background: ACCENT, boxShadow: '0 0 8px rgba(96,165,250,0.8)' }} />}
                        {!isDone && !isActive && <span className="w-2 h-2 rounded-full border border-white/10" />}
                      </div>
                      <span className={`text-xs ${isActive ? 'text-blue-300' : isDone ? 'text-blue-400/55' : 'text-gray-700'}`}
                        style={{ fontFamily: 'var(--font-mono)' }}>{st.label}</span>
                    </div>
                  )
                })}
              </div>
              <p className="text-[10px] text-gray-700 mt-5 leading-relaxed">
                Comparing your draft against the other papers in your library. The bigger your library, the longer this takes.
              </p>
            </div>
          )}

          {error && !loading && (
            <div className="py-10 text-center">
              <p className="text-red-400/60 text-xs mb-4">{error}</p>
              <button onClick={() => run(true)}
                className="text-[10px] uppercase tracking-wider font-bold px-4 py-2 rounded-lg"
                style={{ background: 'rgba(96,165,250,0.08)', border: '1px solid rgba(96,165,250,0.25)', color: 'rgba(96,165,250,0.85)' }}>
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
                {pct != null && report.corpus_size > 0 && (
                  <MetricRing label="original" value={pct} isPercentage accent="blue"
                    tooltip={{ title: 'Original', body: 'The share of your draft that does NOT appear word-for-word in the other papers in your library. Text you quoted or cited is not counted against you. This only compares against your own library — it is not a web-wide plagiarism scan.' }} />
                )}
                <div className="flex-1">
                  <p className="text-white text-sm font-semibold mb-0.5" style={{ fontFamily: 'var(--font-display)' }}>
                    {report.matches_found} unattributed match{report.matches_found === 1 ? '' : 'es'}
                  </p>
                  <p className="text-[11px] text-gray-500">
                    {report.corpus_size} paper{report.corpus_size === 1 ? '' : 's'} compared
                    {report.sources_matched > 0 && ` · ${report.sources_matched} with overlap`}
                    {report.attributed_count > 0 && ` · ${report.attributed_count} quoted or cited`}
                  </p>
                  {report.cached && <p className="text-[8px] uppercase tracking-wider text-gray-700 mt-1.5" style={{ fontFamily: 'var(--font-mono)' }}>cached · re-run to refresh</p>}
                </div>
              </div>

              {/* Papers excluded as the same document — explained, not hidden,
                  otherwise the numbers look inexplicably clean. */}
              {report.excluded_sources?.length > 0 && (
                <div className="rounded-xl p-3 mb-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                  {report.excluded_sources.map((ex, i) => (
                    <p key={i} className="text-[10px] text-gray-600 leading-relaxed">
                      Skipped <span className="text-gray-400">“{ex.title}”</span> — {ex.reason}
                    </p>
                  ))}
                </div>
              )}

              {report.corpus_size === 0 && (
                <p className="text-blue-300/70 text-xs text-center py-6 leading-relaxed">
                  {report.reason || 'No other papers in your library to compare against.'}
                </p>
              )}

              {report.corpus_size > 0 && flagged.length === 0 && (
                <p className="text-blue-300/70 text-xs text-center py-6 leading-relaxed">
                  No unattributed overlap with the papers in your library.
                </p>
              )}

              {flagged.length > 0 && (
                <>
                  <p className="text-[10px] text-gray-700 leading-relaxed mb-4">
                    Matches are prompts to check, not verdicts. Shared text can be a standard definition, your own earlier writing, or a quote whose citation we couldn't see — only you know which.
                  </p>
                  <div className="space-y-3">
                    {flagged.map((m, i) => <MatchCard key={i} match={m} onViewEvidence={setEvidence} />)}
                  </div>
                </>
              )}

              {attributed.length > 0 && (
                <>
                  <p className="text-[9px] uppercase tracking-wider font-bold text-gray-700 mt-6 mb-3" style={{ fontFamily: 'var(--font-mono)' }}>
                    Already attributed
                  </p>
                  <div className="space-y-3">
                    {attributed.map((m, i) => <MatchCard key={i} match={m} onViewEvidence={setEvidence} />)}
                  </div>
                </>
              )}

              <p className="text-[10px] text-gray-700 leading-relaxed mt-6 pt-4" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                This compares your draft only against the papers in your PaperMind library — not against the web or published literature at large. A clean result here is not a plagiarism clearance.
              </p>
            </>
          )}
        </div>
      </div>

      {evidence?.evidence && (
        <PDFPreviewPanel
          pdfUrl={`/api/papers/${paperId}/pdf#page=${evidence.evidence.page}`}
          title={evidence.evidence.section}
          page={evidence.evidence.page}
          highlight={evidence.excerpt}
          onClose={() => setEvidence(null)}
        />
      )}
    </>
  )
}
