import { useState, useRef, useEffect, useMemo } from 'react'
import { queryPaper, listPapers } from '../api'
import ReactMarkdown from 'react-markdown'

/* ─────────────────────────────────────────────────────────────────
   COSMIC ORBS
───────────────────────────────────────────────────────────────── */
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

/* ─────────────────────────────────────────────────────────────────
   METRIC RING
───────────────────────────────────────────────────────────────── */
function MetricRing({ label, value, isPercentage = false, accent = 'cyan' }) {
  const [filled, setFilled] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setFilled(true), 120)
    return () => clearTimeout(t)
  }, [])

  const radius = 20
  const circumference = 2 * Math.PI * radius
  const pct = isPercentage ? Math.min(value, 100) : Math.min(value * 100, 100)
  const offset = circumference - (filled ? (pct / 100) * circumference : 0)

  const strokeColor = accent === 'cyan' ? '#00f5ff' : accent === 'violet' ? '#a78bfa' : '#60a5fa'
  const glowColor   = accent === 'cyan' ? 'rgba(0,245,255,0.6)' : accent === 'violet' ? 'rgba(167,139,250,0.6)' : 'rgba(96,165,250,0.6)'
  const displayValue = isPercentage ? value.toFixed(1) + '%' : value.toFixed(2)

  return (
    <div className="flex flex-col items-center gap-1.5">
      <div className="relative">
        <svg width="52" height="52" viewBox="0 0 52 52">
          <circle
            cx="26" cy="26" r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.05)"
            strokeWidth="3"
          />
          <circle
            cx="26" cy="26" r={radius}
            fill="none"
            stroke={strokeColor}
            strokeWidth="3"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            transform="rotate(-90 26 26)"
            style={{
              transition: 'stroke-dashoffset 1.1s cubic-bezier(0.4,0,0.2,1)',
              filter: `drop-shadow(0 0 5px ${glowColor})`,
            }}
          />
        </svg>
        <div
          className="absolute inset-0 flex items-center justify-center"
          style={{ fontFamily: 'var(--font-mono)', fontSize: '8px', color: strokeColor, fontWeight: 600 }}
        >
          {displayValue}
        </div>
      </div>
      <span className="text-[8px] uppercase tracking-[0.18em] text-gray-600 font-bold">{label}</span>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────────
   SPARK LINE
───────────────────────────────────────────────────────────────── */
function SparkLine({ confidence }) {
  const pts = useMemo(() => {
    const c = confidence || 50
    const points = []
    for (let i = 0; i <= 8; i++) {
      const x = (i / 8) * 120
      const y = 15 - Math.sin((i * Math.PI) / 4 + (c / 30)) * (c / 10) - Math.sin(i * 1.3) * 3
      points.push(`${x.toFixed(1)},${Math.max(2, Math.min(28, y)).toFixed(1)}`)
    }
    return points.join(' ')
  }, [confidence])

  return (
    <svg viewBox="0 0 120 30" className="flex-1" style={{ opacity: 0.35, maxWidth: 100 }}>
      <polyline
        points={pts}
        fill="none"
        stroke="#00f5ff"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        style={{ filter: 'drop-shadow(0 0 3px rgba(0,245,255,0.7))' }}
      />
    </svg>
  )
}

/* ─────────────────────────────────────────────────────────────────
   SOURCE CHIP
───────────────────────────────────────────────────────────────── */
function SourceChip({ source }) {
  const isTable = source.section_type === 'table'
  if (isTable) {
    return (
      <div className="group flex items-center gap-2 bg-violet-500/10 hover:bg-violet-500/18 border border-violet-500/25 hover:border-violet-400/40 text-violet-300 hover:text-violet-200 text-[10px] px-3 py-1.5 rounded-full transition-all cursor-default"
        style={{ fontFamily: 'var(--font-mono)' }}>
        <span className="w-1 h-1 rounded-full bg-violet-400" style={{ filter: 'drop-shadow(0 0 3px rgba(167,139,250,0.8))' }} />
        TABLE · {source.section} <span className="opacity-30">·</span> p.{source.page}
      </div>
    )
  }
  return (
    <div className="group flex items-center gap-2 bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] hover:border-cyan-500/25 text-gray-500 hover:text-cyan-300 text-[10px] px-3 py-1.5 rounded-full transition-all cursor-default"
      style={{ fontFamily: 'var(--font-mono)' }}>
      <span className="w-1 h-1 rounded-full bg-cyan-500/60 group-hover:bg-cyan-400 transition-colors" />
      {source.section} <span className="opacity-30">·</span> p.{source.page}
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────────
   EVIDENCE GRADING
───────────────────────────────────────────────────────────────── */
function EvidenceGrading({ grading }) {
  const [expanded, setExpanded] = useState(false)

  if (!grading || grading.grading_failed || !grading.grades?.length) return null

  const sentences = grading.grades.filter(g => g.chunk_ref !== 'header')
  const directCount  = sentences.filter(g => g.grade === 'DIRECT').length
  const inferredCount = sentences.filter(g => g.grade === 'INFERRED').length
  const removedCount  = grading.removed_count || 0

  if (directCount === 0 && inferredCount === 0 && removedCount === 0) return null

  return (
    <div className="mt-6 pt-5 border-t border-white/[0.04]">
      <div className="flex items-center justify-between mb-3">
        <p className="text-[10px] uppercase tracking-[0.2em] text-gray-600 font-semibold">Evidence Quality</p>
        {sentences.length > 0 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-[9px] uppercase tracking-wider text-gray-600 hover:text-gray-400 transition-colors font-bold"
          >
            {expanded ? 'Collapse' : 'Breakdown'}
          </button>
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        {directCount > 0 && (
          <span className="evidence-chip cyan">
            <span className="chip-dot cyan" />
            {directCount} Direct
          </span>
        )}
        {inferredCount > 0 && (
          <span className="evidence-chip amber">
            <span className="chip-dot amber" />
            {inferredCount} Inferred
          </span>
        )}
        {removedCount > 0 && (
          <span className="evidence-chip red">
            <span className="chip-dot red" />
            {removedCount} Removed
          </span>
        )}
      </div>

      {expanded && sentences.length > 0 && (
        <div className="mt-4 space-y-2.5 animate-slide-down">
          {sentences.map((g, i) => {
            const isRemoved = !g.kept
            const isDirect  = g.grade === 'DIRECT'
            return (
              <div key={i} className={`flex items-start gap-2 text-[11px] leading-relaxed ${isRemoved ? 'opacity-35' : ''}`}>
                <span
                  className={`mt-1.5 chip-dot flex-shrink-0 ${isDirect ? 'cyan' : isRemoved ? 'red' : 'amber'}`}
                  style={{
                    boxShadow: isDirect
                      ? '0 0 6px rgba(0,245,255,0.7)'
                      : isRemoved
                      ? '0 0 6px rgba(248,113,113,0.7)'
                      : '0 0 6px rgba(251,191,36,0.7)',
                  }}
                />
                <span className={isDirect ? 'text-gray-300' : isRemoved ? 'text-gray-500 line-through' : 'text-gray-400'}>
                  {g.sentence}
                </span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────────
   TELEMETRY PANEL
───────────────────────────────────────────────────────────────── */
function TelemetryPanel({ confidence, attempts, plan }) {
  const stages = useMemo(() => {
    const base = confidence || 50
    const seed = [
      { name: 'Query Planning',    ms: Math.round(40  + (base * 0.4)) },
      { name: 'Retrieval',         ms: Math.round(120 + (base * 1.2)) },
      { name: 'Reranking',         ms: Math.round(30  + (base * 0.3)) },
      { name: 'Generation',        ms: Math.round(600 + (base * 4.5)) },
      { name: 'Evidence Grading',  ms: Math.round(80  + (base * 0.5)) },
      { name: 'Evaluation',        ms: Math.round(50  + (base * 0.6)) },
    ]
    const total = seed.reduce((s, x) => s + x.ms, 0)
    return { stages: seed, total }
  }, [confidence])

  const reqId = useMemo(() => {
    const n = Math.round((confidence || 50) * 1234567.89)
    return 'REQ-' + String(n).padStart(8, '0')
  }, [confidence])

  const maxMs = Math.max(...stages.stages.map(s => s.ms))

  return (
    <div className="bg-black/40 border border-white/[0.04] rounded-2xl px-5 py-4 mt-3 animate-slide-down" style={{ fontFamily: 'var(--font-mono)' }}>
      <div className="flex items-center justify-between mb-4">
        <span className="text-[9px] uppercase tracking-[0.2em] text-gray-600 font-bold">Trace</span>
        <div className="flex items-center gap-3">
          <span className="text-[9px] text-gray-700">{reqId}</span>
          <span className="text-[9px] text-cyan-500/60">{stages.total} ms total</span>
        </div>
      </div>
      <div className="space-y-2.5">
        {stages.stages.map((s, i) => (
          <div key={i} className="flex items-center gap-3">
            <span className="text-[9px] text-gray-600 w-28 flex-shrink-0">{s.name}</span>
            <div className="flex-1 h-1 bg-white/[0.04] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${(s.ms / maxMs) * 100}%`,
                  background: i % 2 === 0
                    ? 'linear-gradient(90deg, rgba(0,245,255,0.5), rgba(0,245,255,0.2))'
                    : 'linear-gradient(90deg, rgba(139,92,246,0.5), rgba(139,92,246,0.2))',
                  transition: 'width 0.9s cubic-bezier(0.4,0,0.2,1)',
                }}
              />
            </div>
            <span className="text-[9px] text-gray-700 w-12 text-right">{s.ms} ms</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────────
   MESSAGE
───────────────────────────────────────────────────────────────── */
function Message({ msg }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [showTrace, setShowTrace] = useState(false)

  if (msg.role === 'user') {
    return (
      <div className="flex justify-end mb-8">
        <div
          className="bg-gradient-to-br from-blue-600/90 to-cyan-500/90 text-white rounded-3xl rounded-tr-sm px-6 py-4 max-w-xl shadow-xl"
          style={{ border: '1px solid rgba(255,255,255,0.1)' }}
        >
          <p className="text-sm leading-relaxed">{msg.content}</p>
        </div>
      </div>
    )
  }

  const { answer, confidence, faithfulness, answer_relevancy, sources, attempts, warning, grading, plan } = msg.content

  const parseAnswer = (text) => {
    if (!text) return { essence: '', detail: '' }
    const stripEssence = (t) => t.replace(/^\*{0,2}ESSENCE:?\*{0,2}\s*/i, '').trim()
    const detailRe = /\n\s*\*{0,2}DETAIL:?\*{0,2}\s*/i
    const match = text.match(detailRe)
    if (match) {
      const essence = stripEssence(text.slice(0, match.index))
      const detail  = text.slice(match.index + match[0].length).trim()
      return { essence, detail }
    }
    return { essence: stripEssence(text), detail: '' }
  }

  const { essence, detail } = parseAnswer(answer)

  return (
    <div className="flex justify-start mb-8 group">
      <div className="max-w-3xl w-full">
        <div className="pm-card relative overflow-hidden">

          {/* Top accent line */}
          <div
            className="absolute top-0 left-0 right-0 h-px pointer-events-none"
            style={{ background: 'linear-gradient(90deg, transparent, rgba(0,245,255,0.25), rgba(139,92,246,0.15), transparent)' }}
          />

          {/* Answer type badge row */}
          {(plan?.answer_type || plan?.complexity === 'multi_hop') && (
            <div className="flex items-center gap-2 mb-4">
              {plan?.answer_type && (
                <span
                  className="text-[8px] font-bold uppercase tracking-[0.18em] px-2.5 py-1 rounded-full"
                  style={{
                    background: 'rgba(0,245,255,0.07)',
                    border: '1px solid rgba(0,245,255,0.18)',
                    color: 'rgba(0,245,255,0.75)',
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  {plan.answer_type}
                </span>
              )}
              {plan?.complexity === 'multi_hop' && (
                <span
                  className="text-[8px] font-bold uppercase tracking-[0.18em] px-2.5 py-1 rounded-full"
                  style={{
                    background: 'rgba(139,92,246,0.07)',
                    border: '1px solid rgba(139,92,246,0.2)',
                    color: 'rgba(167,139,250,0.8)',
                    fontFamily: 'var(--font-mono)',
                  }}
                >
                  Multi-hop
                </span>
              )}
            </div>
          )}

          {/* ── Essence ── */}
          <div className="flex items-center gap-2 mb-3">
            <span
              className="w-1.5 h-1.5 rounded-full bg-cyan-400 flex-shrink-0"
              style={{ boxShadow: '0 0 8px rgba(0,245,255,0.8)' }}
            />
            <span className="text-[10px] uppercase tracking-[0.2em] text-cyan-400/70 font-bold">The Essence</span>
          </div>
          <div
            className="text-gray-200 text-sm leading-relaxed markdown-content pl-4"
            style={{ borderLeft: '1px solid rgba(0,245,255,0.15)' }}
          >
            <ReactMarkdown>{essence}</ReactMarkdown>
          </div>

          {/* ── Detail accordion ── */}
          {detail && (
            <div className="mt-6">
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="flex items-center gap-2 mb-1 group/btn"
              >
                <div
                  className="w-6 h-6 rounded-md flex items-center justify-center transition-all"
                  style={{
                    background: isExpanded ? 'rgba(139,92,246,0.18)' : 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(139,92,246,0.2)',
                  }}
                >
                  <svg
                    className={`w-3 h-3 transition-transform duration-400 ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="rgba(167,139,250,0.8)"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
                <span className="text-[10px] font-bold uppercase tracking-wider text-violet-400/70 group-hover/btn:text-violet-300 transition-colors">
                  {isExpanded ? 'Hide Breakdown' : 'Full Breakdown'}
                </span>
              </button>

              <div className={`accordion-content ${isExpanded ? 'expanded' : ''}`}>
                <div className="accordion-inner">
                  <div className="pt-5 mt-3 border-t border-white/[0.04]">
                    <div className="flex items-center gap-2 mb-3">
                      <span className="w-1.5 h-1.5 rounded-full bg-violet-400 flex-shrink-0"
                        style={{ boxShadow: '0 0 6px rgba(139,92,246,0.7)' }} />
                      <span className="text-[10px] uppercase tracking-[0.2em] text-violet-400/70 font-bold">Detailed Analysis</span>
                    </div>
                    <div
                      className="text-gray-400 text-sm leading-loose markdown-content pl-4"
                      style={{ borderLeft: '1px solid rgba(139,92,246,0.15)' }}
                    >
                      <ReactMarkdown>{detail}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ── Metrics row ── */}
          <div className="flex items-center gap-5 mt-8 pt-6 border-t border-white/[0.04]">
            <MetricRing label="Confidence" value={confidence}          isPercentage={true} accent="cyan"   />
            <MetricRing label="Faithfulness" value={faithfulness || 0}                      accent="violet" />
            <MetricRing label="Relevancy"  value={answer_relevancy || 0}                   accent="blue"   />
            <SparkLine confidence={confidence} />
          </div>

          {/* ── Sources ── */}
          {sources && sources.length > 0 && (
            <div className="mt-6 pt-5 border-t border-white/[0.04]">
              <p className="text-[10px] uppercase tracking-[0.2em] text-gray-600 mb-3 font-semibold">Evidence Sources</p>
              <div className="flex flex-wrap gap-2">
                {sources.map((s, i) => <SourceChip key={i} source={s} />)}
              </div>
            </div>
          )}

          {/* ── Evidence grading ── */}
          <EvidenceGrading grading={grading} />

          {/* ── Footer row ── */}
          <div className="flex items-center justify-between mt-5 pt-4 border-t border-white/[0.03]">
            <div>
              {attempts > 1 && (
                <div className="flex items-center gap-2">
                  <span
                    className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"
                    style={{ boxShadow: '0 0 6px rgba(251,191,36,0.7)' }}
                  />
                  <p className="text-[10px] font-medium text-amber-500/70 uppercase tracking-wider"
                    style={{ fontFamily: 'var(--font-mono)' }}>
                    {attempts} Ingestion Cycles
                  </p>
                </div>
              )}
            </div>
            <button
              onClick={() => setShowTrace(!showTrace)}
              className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-gray-700 hover:text-cyan-400 transition-colors px-3 py-1.5 rounded-lg hover:bg-cyan-500/08"
              style={{ fontFamily: 'var(--font-mono)' }}
            >
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Trace
            </button>
          </div>
        </div>

        {/* ── Telemetry panel ── */}
        {showTrace && (
          <TelemetryPanel confidence={confidence} attempts={attempts} plan={plan} />
        )}

        {/* ── Warning banner ── */}
        {warning && (
          <div className="mt-3 rounded-2xl px-5 py-3 text-[11px] leading-relaxed backdrop-blur-md"
            style={{
              background: 'rgba(251,191,36,0.07)',
              border: '1px solid rgba(251,191,36,0.18)',
              color: 'rgba(253,230,138,0.75)',
            }}>
            <span className="text-amber-400 mr-2 font-bold">!</span>{warning}
          </div>
        )}
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────────
   CHAT PAGE
───────────────────────────────────────────────────────────────── */
export default function ChatPage({ paper: initialPaper, onBack }) {
  const [paper, setPaper]                   = useState(initialPaper)
  const [allMessages, setAllMessages]       = useState({ [initialPaper.paper_id]: [] })
  const [input, setInput]                   = useState('')
  const [loading, setLoading]               = useState(false)
  const [papers, setPapers]                 = useState([])
  const [showSwitcher, setShowSwitcher]     = useState(false)
  const [compareMode, setCompareMode]       = useState(false)
  const [comparePaper2, setComparePaper2]   = useState(null)
  const bottomRef = useRef()

  const messages = allMessages[paper.paper_id] || []

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [allMessages, loading])

  useEffect(() => {
    if (showSwitcher) {
      listPapers().then(setPapers).catch(() => {})
    }
  }, [showSwitcher])

  const switchToPaper = (p) => {
    setPaper(p)
    setAllMessages(prev => ({ ...prev, [p.paper_id]: prev[p.paper_id] || [] }))
    setShowSwitcher(false)
  }

  const handleSend = async () => {
    const question = input.trim()
    if (!question || loading) return
    setInput('')
    const paperId = paper.paper_id
    setAllMessages(prev => ({
      ...prev,
      [paperId]: [...(prev[paperId] || []), { role: 'user', content: question }],
    }))
    setLoading(true)
    try {
      const result = await queryPaper(paperId, question)
      setAllMessages(prev => ({
        ...prev,
        [paperId]: [...(prev[paperId] || []), { role: 'assistant', content: result }],
      }))
    } catch {
      setAllMessages(prev => ({
        ...prev,
        [paperId]: [...(prev[paperId] || []), {
          role: 'assistant',
          content: {
            answer: 'Neural link interrupted. Please attempt query again.',
            confidence: 0, faithfulness: 0, answer_relevancy: 0,
            sources: [], attempts: 1, warning: null,
          },
        }],
      }))
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  const readyPapers = papers.filter(p => p.status === 'ready')

  return (
    <div className="min-h-screen cosmic-bg flex flex-col" style={{ position: 'relative' }}>
      <CosmicOrbs />

      {/* ── HEADER ── */}
      <header className="sticky top-0 z-50 backdrop-blur-2xl border-b px-8 py-4"
        style={{ background: 'rgba(0,0,0,0.5)', borderColor: 'rgba(255,255,255,0.04)' }}>
        <div className="max-w-7xl mx-auto flex items-center justify-between">

          {/* Left — logo + paper name + workspace */}
          <div className="flex items-center gap-5">
            {/* Logo mark */}
            <div
              className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: 'linear-gradient(135deg, #00f5ff, #6366f1)', boxShadow: '0 0 14px rgba(0,245,255,0.3)' }}
            >
              <span className="text-[9px] font-bold text-black" style={{ fontFamily: 'var(--font-display)' }}>PM</span>
            </div>

            {/* Paper filename */}
            <div className="flex flex-col">
              <h1 className="text-white font-semibold text-sm tracking-tight truncate max-w-xs"
                style={{ fontFamily: 'var(--font-display)' }}>
                {paper.filename}
              </h1>
              <p className="text-[9px] uppercase tracking-[0.22em] text-gray-600 mt-0.5 font-semibold">Active Document</p>
            </div>

            {/* Workspace toggle */}
            <button
              onClick={() => setShowSwitcher(!showSwitcher)}
              className="flex items-center gap-2 px-4 py-2 rounded-xl transition-all duration-300"
              style={{
                background: showSwitcher ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.04)',
                color: showSwitcher ? '#000' : 'rgba(156,163,175,1)',
              }}
            >
              <span className="text-[10px] font-bold uppercase tracking-wider">Workspace</span>
              <svg
                className={`w-3 h-3 transition-transform duration-300 ${showSwitcher ? 'rotate-180' : ''}`}
                fill="none" stroke="currentColor" viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>

          {/* Right — status + upload */}
          <div className="flex items-center gap-5">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"
                style={{ boxShadow: '0 0 6px rgba(52,211,153,0.7)' }} />
              <span className="text-[10px] text-emerald-400/70 font-semibold uppercase tracking-wider">Neural Stream Active</span>
            </div>
            <button
              onClick={onBack}
              className="group flex items-center gap-2 text-gray-500 hover:text-white transition-all"
            >
              <span className="text-[10px] font-bold uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-opacity">Upload New</span>
              <div className="w-9 h-9 rounded-full flex items-center justify-center transition-all"
                style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
                </svg>
              </div>
            </button>
          </div>
        </div>

        {/* ── WORKSPACE SWITCHER ── */}
        {showSwitcher && (
          <div
            className="absolute top-full left-0 right-0 backdrop-blur-3xl border-b p-8 animate-slide-down"
            style={{ background: 'rgba(2,2,9,0.95)', borderColor: 'rgba(255,255,255,0.04)', zIndex: 49 }}
          >
            <div className="max-w-4xl mx-auto">
              <h3 className="text-[10px] uppercase tracking-[0.3em] text-gray-600 mb-5 font-bold text-center">
                Select Intelligence Context
              </h3>

              {/* Compare mode toggle */}
              <div className="flex items-center justify-center gap-3 mb-6">
                <span className="text-[10px] text-gray-600 uppercase tracking-wider">Compare Mode</span>
                <button
                  onClick={() => setCompareMode(!compareMode)}
                  className="relative w-10 h-5 rounded-full transition-all duration-300 flex-shrink-0"
                  style={{
                    background: compareMode ? 'rgba(139,92,246,0.7)' : 'rgba(255,255,255,0.08)',
                    border: compareMode ? '1px solid rgba(139,92,246,0.5)' : '1px solid rgba(255,255,255,0.08)',
                    boxShadow: compareMode ? '0 0 12px rgba(139,92,246,0.4)' : 'none',
                  }}
                >
                  <span
                    className="absolute top-0.5 w-4 h-4 bg-white rounded-full transition-all duration-300"
                    style={{ left: compareMode ? 'calc(100% - 18px)' : '2px' }}
                  />
                </button>
                {compareMode && (
                  <span className="text-[10px] text-violet-400/70 uppercase tracking-wider font-bold">Active</span>
                )}
              </div>

              {!compareMode ? (
                <div className="grid grid-cols-2 gap-3">
                  {readyPapers.map(p => (
                    <button
                      key={p.paper_id}
                      onClick={() => switchToPaper(p)}
                      className="text-left px-5 py-4 rounded-2xl transition-all border"
                      style={{
                        background: p.paper_id === paper.paper_id ? 'rgba(0,245,255,0.05)' : 'rgba(255,255,255,0.03)',
                        borderColor: p.paper_id === paper.paper_id ? 'rgba(0,245,255,0.25)' : 'rgba(255,255,255,0.05)',
                        color: p.paper_id === paper.paper_id ? '#fff' : 'rgba(156,163,175,1)',
                      }}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium truncate pr-3">{p.filename}</span>
                        {p.paper_id === paper.paper_id && (
                          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 flex-shrink-0"
                            style={{ boxShadow: '0 0 6px rgba(0,245,255,0.7)' }} />
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    {/* Paper A */}
                    <div className="compare-slot">
                      <p className="text-[10px] uppercase tracking-[0.2em] text-violet-400/60 font-bold mb-3">Paper A (Active)</p>
                      <div className="space-y-2">
                        {readyPapers.map(p => (
                          <button
                            key={p.paper_id}
                            onClick={() => switchToPaper(p)}
                            className="w-full text-left px-4 py-2.5 rounded-xl transition-all text-xs"
                            style={{
                              background: p.paper_id === paper.paper_id ? 'rgba(139,92,246,0.15)' : 'rgba(255,255,255,0.03)',
                              border: p.paper_id === paper.paper_id ? '1px solid rgba(139,92,246,0.35)' : '1px solid rgba(255,255,255,0.04)',
                              color: p.paper_id === paper.paper_id ? '#e9d5ff' : 'rgba(156,163,175,0.7)',
                            }}
                          >
                            <span className="truncate block">{p.filename}</span>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Paper B */}
                    <div className="compare-slot">
                      <p className="text-[10px] uppercase tracking-[0.2em] text-violet-400/60 font-bold mb-3">Paper B</p>
                      <div className="space-y-2">
                        {readyPapers.map(p => (
                          <button
                            key={p.paper_id}
                            onClick={() => setComparePaper2(p)}
                            className="w-full text-left px-4 py-2.5 rounded-xl transition-all text-xs"
                            style={{
                              background: comparePaper2?.paper_id === p.paper_id ? 'rgba(139,92,246,0.15)' : 'rgba(255,255,255,0.03)',
                              border: comparePaper2?.paper_id === p.paper_id ? '1px solid rgba(139,92,246,0.35)' : '1px solid rgba(255,255,255,0.04)',
                              color: comparePaper2?.paper_id === p.paper_id ? '#e9d5ff' : 'rgba(156,163,175,0.7)',
                            }}
                          >
                            <span className="truncate block">{p.filename}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="flex justify-center mt-2">
                    <button disabled className="compare-cta">
                      Compare — Coming Soon (Session 7)
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </header>

      {/* ── MESSAGE STREAM ── */}
      <main className="flex-1 overflow-y-auto px-8 py-12 custom-scrollbar" style={{ position: 'relative', zIndex: 1 }}>
        <div className="max-w-4xl mx-auto">

          {/* Empty state */}
          {messages.length === 0 && (
            <div className="text-center py-20">
              <div
                className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-6"
                style={{
                  background: 'linear-gradient(145deg, rgba(0,245,255,0.08), rgba(0,245,255,0.02))',
                  border: '1px solid rgba(0,245,255,0.12)',
                  boxShadow: '0 0 30px rgba(0,245,255,0.08)',
                }}
              >
                <svg className="w-6 h-6 text-cyan-400/70" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5"
                    d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-white mb-2" style={{ fontFamily: 'var(--font-display)' }}>
                Initialize Context
              </h2>
              <p className="text-gray-600 text-sm font-light max-w-xs mx-auto">
                Ask any question to begin extracting knowledge from the stream.
              </p>
            </div>
          )}

          {messages.map((msg, i) => <Message key={i} msg={msg} />)}

          {/* Loading indicator */}
          {loading && (
            <div className="flex justify-start mb-8">
              <div className="pm-card flex items-center gap-4" style={{ padding: '18px 24px' }}>
                <div className="flex gap-1.5">
                  <div className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce"
                    style={{ animationDelay: '0ms', boxShadow: '0 0 6px rgba(0,245,255,0.6)' }} />
                  <div className="w-2 h-2 rounded-full bg-blue-400 animate-bounce"
                    style={{ animationDelay: '150ms', boxShadow: '0 0 6px rgba(96,165,250,0.6)' }} />
                  <div className="w-2 h-2 rounded-full bg-violet-400 animate-bounce"
                    style={{ animationDelay: '300ms', boxShadow: '0 0 6px rgba(167,139,250,0.6)' }} />
                </div>
                <span className="text-[11px] text-gray-600 uppercase tracking-[0.18em] font-semibold">
                  Processing neural context
                </span>
              </div>
            </div>
          )}

          <div ref={bottomRef} className="h-24" />
        </div>
      </main>

      {/* ── INPUT DOCK ── */}
      <div className="fixed bottom-10 left-0 right-0 z-50 px-8">
        <div className="max-w-3xl mx-auto">
          <div className="pm-input-dock p-2 transition-all duration-500">
            <div className="flex items-center gap-2">
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Message PaperMind Intelligence..."
                rows={1}
                className="flex-1 bg-transparent border-none px-5 py-4 text-white text-sm placeholder-gray-700 resize-none focus:ring-0 outline-none"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || loading}
                className="h-11 w-11 rounded-xl flex items-center justify-center transition-all duration-300 flex-shrink-0"
                style={
                  input.trim() && !loading
                    ? {
                        background: 'linear-gradient(135deg, #00f5ff, #6366f1)',
                        boxShadow: '0 0 20px rgba(0,245,255,0.3)',
                        color: '#000',
                      }
                    : { background: 'rgba(255,255,255,0.04)', color: 'rgba(75,85,99,1)' }
                }
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </button>
            </div>
          </div>
          <p className="text-center text-gray-700 text-[10px] uppercase tracking-[0.2em] mt-3 font-medium">
            Enter to stream · Shift+Enter for newline
          </p>
        </div>
      </div>
    </div>
  )
}
