import { useState, useEffect, useCallback } from 'react'
import { structureCheckStream, getVenues } from '../api'
import { MetricRing } from './MetricRing'
import PDFPreviewPanel from './PDFPreviewPanel'

/* ── VENUE-FIT / STRUCTURE PANEL ─────────────────────────────────────
   Presence checklist: does the draft contain the structural components a
   target venue expects (Limitations, Ethics, Reproducibility, …)? Venue is
   selectable; a required component that's missing is the headline gap. */

const ACCENT = '#fbbf24' // amber/gold — distinct from review/audit/novelty

const STRUCTURE_STAGES = [
  { key: 'reading',   label: 'Reading draft'        },
  { key: 'gathering', label: 'Locating components'  },
  { key: 'checking',  label: 'Checking structure'   },
]

// Fallback if /venues can't be fetched — keeps the selector usable offline.
const FALLBACK_VENUES = [
  { slug: 'generic', label: 'Generic ML/CS' },
  { slug: 'neurips', label: 'NeurIPS' },
  { slug: 'icml',    label: 'ICML' },
  { slug: 'iclr',    label: 'ICLR' },
  { slug: 'acl',     label: 'ACL / ARR' },
  { slug: 'emnlp',   label: 'EMNLP' },
  { slug: 'cvpr',    label: 'CVPR / ICCV' },
]

// Style is a function of (verdict, required): a required component that's absent
// is a hard gap (red); an optional one is a soft nudge (muted).
function styleFor(c) {
  if (c.verdict === 'PRESENT') return { label: 'Present', color: 'rgba(52,211,153,0.95)', bg: 'rgba(52,211,153,0.08)', border: 'rgba(52,211,153,0.25)', dot: '#34d399' }
  if (c.verdict === 'THIN')    return { label: 'Thin',    color: 'rgba(251,191,36,0.97)', bg: 'rgba(251,191,36,0.09)', border: 'rgba(251,191,36,0.3)',  dot: '#fbbf24' }
  // MISSING
  return c.required
    ? { label: 'Missing', color: 'rgba(248,113,113,0.97)', bg: 'rgba(248,113,113,0.09)', border: 'rgba(248,113,113,0.3)',  dot: '#f87171' }
    : { label: 'Absent',  color: 'rgba(156,163,175,0.95)', bg: 'rgba(156,163,175,0.06)', border: 'rgba(255,255,255,0.12)', dot: '#9ca3af' }
}

function ComponentCard({ comp, onViewEvidence }) {
  const s = styleFor(comp)
  return (
    <div className="rounded-xl p-4" style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.05)' }}>
      <div className="flex items-center justify-between mb-2.5 gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-[8px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full flex items-center gap-1.5 flex-shrink-0"
            style={{ background: s.bg, border: `1px solid ${s.border}`, color: s.color }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: s.dot, boxShadow: `0 0 5px ${s.dot}` }} />
            {s.label}
          </span>
          <span className="text-gray-200 text-xs font-semibold truncate">{comp.label}</span>
        </div>
        {comp.required && (
          <span className="text-[8px] uppercase tracking-wider text-gray-500 font-bold flex-shrink-0" style={{ fontFamily: 'var(--font-mono)' }}>
            required
          </span>
        )}
      </div>

      {comp.why && <p className="text-[11px] leading-relaxed text-gray-400 mb-2">{comp.why}</p>}

      {comp.fix && comp.verdict !== 'PRESENT' && (
        <p className="text-[11px] leading-relaxed mb-2.5">
          <span className="uppercase tracking-wider text-[8px] font-bold mr-1.5" style={{ color: ACCENT }}>Fix</span>
          <span className="text-gray-300">{comp.fix}</span>
        </p>
      )}

      {comp.evidence && (
        <button onClick={() => onViewEvidence(comp)}
          className="flex items-center gap-1.5 text-[9px] uppercase tracking-wider font-bold px-2.5 py-1 rounded-lg transition-all"
          style={{ background: 'rgba(251,191,36,0.06)', border: '1px solid rgba(251,191,36,0.2)', color: 'rgba(251,191,36,0.85)' }}>
          <span className="w-1 h-1 rounded-full" style={{ background: ACCENT }} />
          {comp.evidence.section} · p.{comp.evidence.page}
        </button>
      )}
    </div>
  )
}

export default function StructurePanel({ paperId, onClose, embedded = false }) {
  const [venues,   setVenues]   = useState(FALLBACK_VENUES)
  const [venue,    setVenue]    = useState('generic')
  const [report,   setReport]   = useState(null)
  const [loading,  setLoading]  = useState(true)
  const [progress, setProgress] = useState([])
  const [error,    setError]    = useState('')
  const [evidence, setEvidence] = useState(null)

  // Load the venue list once (best-effort — falls back to the hardcoded list).
  useEffect(() => {
    let cancelled = false
    getVenues().then(v => { if (!cancelled && v?.length) setVenues(v) }).catch(() => {})
    return () => { cancelled = true }
  }, [])

  // Manual force re-run (bypasses the cache for the current venue).
  const run = useCallback((force = false) => {
    setLoading(true); setError(''); setProgress([]); setReport(null)
    structureCheckStream(paperId, venue, ({ type, data }) => {
      if (type === 'progress') setProgress(prev => [...prev, data.stage])
    }, force)
      .then(setReport)
      .catch(e => setError(e.message || 'Structure check failed'))
      .finally(() => setLoading(false))
  }, [paperId, venue])

  // Runs on mount AND whenever the venue changes (venue drives the rubric, so a
  // new venue means a fresh report — served from cache when it matches). Inline,
  // not via run(), so no setState fires synchronously in the effect body.
  useEffect(() => {
    let cancelled = false
    setLoading(true); setError(''); setProgress([]); setReport(null)
    structureCheckStream(paperId, venue, ({ type, data }) => {
      if (type === 'progress' && !cancelled) setProgress(prev => [...prev, data.stage])
    }, false)
      .then(r => { if (!cancelled) setReport(r) })
      .catch(e => { if (!cancelled) setError(e.message || 'Structure check failed') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [paperId, venue])

  const latestStage = progress[progress.length - 1]
  const comps = report?.components || []
  const pct   = report?.completeness_score != null ? Math.round(report.completeness_score * 100) : null

  return (
    <>
      <div className={embedded ? 'flex flex-col w-full max-w-3xl mx-auto' : 'panel-slide-in fixed top-0 right-0 bottom-0 z-[80] flex flex-col'}
        style={embedded
          ? { minHeight: '60vh' }
          : { width: 420, background: 'rgba(4,4,16,0.97)', borderLeft: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(32px)' }}>
        <div className="flex items-center justify-between px-6 py-5 border-b" style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
          <div>
            <p className="text-[9px] uppercase tracking-[0.22em] font-bold mb-0.5" style={{ fontFamily: 'var(--font-mono)', color: 'rgba(251,191,36,0.6)' }}>Completeness</p>
            <h2 className="text-white font-semibold text-sm" style={{ fontFamily: 'var(--font-display)' }}>Venue Fit</h2>
          </div>
          <div className="flex items-center gap-2">
            {/* Target-venue selector — changing it re-runs the check. */}
            <select
              value={venue}
              onChange={e => setVenue(e.target.value)}
              disabled={loading}
              className="text-[10px] font-semibold rounded-lg px-2 py-1.5 outline-none cursor-pointer disabled:opacity-40"
              style={{ background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.25)', color: 'rgba(251,191,36,0.95)', fontFamily: 'var(--font-mono)' }}
            >
              {venues.map(v => (
                <option key={v.slug} value={v.slug} style={{ background: '#0a0a1a', color: '#e5e7eb' }}>{v.label}</option>
              ))}
            </select>
            {report && !loading && (
              <button onClick={() => run(true)} title="Re-run check"
                className="w-7 h-7 rounded-lg flex items-center justify-center text-gray-600 hover:text-amber-300 hover:bg-white/[0.06] transition-all">
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
                {STRUCTURE_STAGES.map(st => {
                  const seenIdx = progress.indexOf(st.key)
                  const isActive = latestStage === st.key
                  const isDone   = seenIdx !== -1 && !isActive
                  return (
                    <div key={st.key} className="flex items-center gap-3">
                      <div className="w-4 h-4 flex items-center justify-center flex-shrink-0">
                        {isDone   && <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'rgba(251,191,36,0.6)' }} />}
                        {isActive && <span className="w-3 h-3 rounded-full animate-pulse" style={{ background: ACCENT, boxShadow: '0 0 8px rgba(251,191,36,0.8)' }} />}
                        {!isDone && !isActive && <span className="w-2 h-2 rounded-full border border-white/10" />}
                      </div>
                      <span className={`text-xs ${isActive ? 'text-amber-300' : isDone ? 'text-amber-400/55' : 'text-gray-700'}`}
                        style={{ fontFamily: 'var(--font-mono)' }}>{st.label}</span>
                    </div>
                  )
                })}
              </div>
              <p className="text-[10px] text-gray-700 mt-5 leading-relaxed">
                Checking your draft for the sections a {venues.find(v => v.slug === venue)?.label || venue} reviewer expects.
              </p>
            </div>
          )}

          {error && !loading && (
            <div className="py-10 text-center">
              <p className="text-red-400/60 text-xs mb-4">{error}</p>
              <button onClick={() => run(true)}
                className="text-[10px] uppercase tracking-wider font-bold px-4 py-2 rounded-lg"
                style={{ background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.25)', color: 'rgba(251,191,36,0.85)' }}>
                Try again
              </button>
            </div>
          )}

          {report && !loading && report.structure_failed && (
            <p className="text-gray-600 text-xs text-center py-10">{report.reason || 'Could not check this draft.'}</p>
          )}

          {report && !loading && !report.structure_failed && comps.length > 0 && (
            <>
              {/* Headline */}
              <div className="flex items-center gap-4 mb-5 rounded-2xl p-4" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                {pct != null && <MetricRing label="present" value={pct} isPercentage accent="amber" />}
                <div className="flex-1">
                  <p className="text-white text-sm font-semibold mb-0.5" style={{ fontFamily: 'var(--font-display)' }}>
                    {report.present}/{report.components_checked} present · {report.venue_label}
                  </p>
                  <p className="text-[11px] text-gray-500">
                    {report.thin > 0 && <span className="text-amber-400/80">{report.thin} thin</span>}
                    {report.thin > 0 && report.missing > 0 && ' · '}
                    {report.missing > 0 && <span className="text-gray-500">{report.missing} absent</span>}
                    {report.thin === 0 && report.missing === 0 && <span className="text-emerald-400/80">all components present</span>}
                  </p>
                  {report.cached && <p className="text-[8px] uppercase tracking-wider text-gray-700 mt-1.5" style={{ fontFamily: 'var(--font-mono)' }}>cached · re-run to refresh</p>}
                </div>
              </div>

              {/* Required-missing banner — the headline risk for this venue */}
              {report.required_missing > 0 && (
                <div className="rounded-xl p-3.5 mb-5 flex items-start gap-2.5" style={{ background: 'rgba(248,113,113,0.07)', border: '1px solid rgba(248,113,113,0.22)' }}>
                  <svg className="w-4 h-4 flex-shrink-0 mt-0.5" fill="none" stroke="rgba(248,113,113,0.9)" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01M5.07 19h13.86a2 2 0 001.71-3l-6.93-12a2 2 0 00-3.42 0l-6.93 12a2 2 0 001.71 3z" />
                  </svg>
                  <p className="text-[11px] leading-relaxed" style={{ color: 'rgba(248,113,113,0.92)' }}>
                    {report.required_missing} section{report.required_missing === 1 ? '' : 's'} that {report.venue_label} expects {report.required_missing === 1 ? 'is' : 'are'} missing — add {report.required_missing === 1 ? 'it' : 'them'} before submitting.
                  </p>
                </div>
              )}

              <p className="text-[10px] text-gray-700 leading-relaxed mb-4">
                A presence checklist, not a quality judgement — it flags what's absent, not whether your results are good.
              </p>

              <div className="space-y-3">
                {comps.map((c, i) => <ComponentCard key={i} comp={c} onViewEvidence={setEvidence} />)}
              </div>
            </>
          )}
        </div>
      </div>

      {evidence?.evidence && (
        <PDFPreviewPanel
          pdfUrl={`/api/papers/${paperId}/pdf#page=${evidence.evidence.page}`}
          title={evidence.evidence.section}
          page={evidence.evidence.page}
          highlight={evidence.evidence.quote}
          onClose={() => setEvidence(null)}
        />
      )}
    </>
  )
}
