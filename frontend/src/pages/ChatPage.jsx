import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { listPapers, deletePaper, queryPaperStream, comparePapersStream } from '../api'
import ReactMarkdown from 'react-markdown'

/* ── COSMIC ORBS ─────────────────────────────────────────────────── */
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

/* ── TOAST SYSTEM ────────────────────────────────────────────────── */
function Toast({ id, message, type, onDismiss }) {
  useEffect(() => {
    const t = setTimeout(() => onDismiss(id), 5000)
    return () => clearTimeout(t)
  }, [id, onDismiss])

  const styles = {
    error: { bg: 'rgba(248,113,113,0.12)', border: 'rgba(248,113,113,0.3)', color: 'rgb(252,165,165)', icon: '!' },
    warn:  { bg: 'rgba(251,191,36,0.12)',  border: 'rgba(251,191,36,0.3)',  color: 'rgb(253,230,138)', icon: '△' },
    info:  { bg: 'rgba(0,245,255,0.08)',   border: 'rgba(0,245,255,0.2)',   color: 'rgb(34,211,238)',  icon: 'ℹ' },
  }
  const s = styles[type] || styles.info
  return (
    <div className="toast-slide-in flex items-start gap-3 px-4 py-3 rounded-2xl backdrop-blur-xl"
      style={{ background: s.bg, border: `1px solid ${s.border}`, minWidth: 280, maxWidth: 400 }}>
      <span className="text-sm font-bold flex-shrink-0" style={{ color: s.color }}>{s.icon}</span>
      <p className="text-sm flex-1 leading-snug" style={{ color: s.color }}>{message}</p>
      <button onClick={() => onDismiss(id)}
        className="text-sm opacity-40 hover:opacity-100 transition-opacity flex-shrink-0"
        style={{ color: s.color }}>×</button>
    </div>
  )
}

function ToastContainer({ toasts, onDismiss }) {
  if (!toasts.length) return null
  return (
    <div className="fixed top-20 right-6 z-[100] flex flex-col gap-2 pointer-events-none">
      {toasts.map(t => (
        <div key={t.id} className="pointer-events-auto">
          <Toast {...t} onDismiss={onDismiss} />
        </div>
      ))}
    </div>
  )
}

/* ── COMMAND PALETTE ─────────────────────────────────────────────── */
const CMD_LIST = [
  { id: 'upload',    label: 'Upload new paper',       icon: '↑' },
  { id: 'compare',   label: 'Toggle compare mode',    icon: '⇔' },
  { id: 'clear',     label: 'Clear conversation',     icon: '⌫' },
  { id: 'workspace', label: 'Open workspace switcher', icon: '◫' },
]

function CommandPalette({ onClose, onCommand }) {
  const [query, setQuery] = useState('')
  const inputRef = useRef()
  useEffect(() => { inputRef.current?.focus() }, [])

  const filtered = CMD_LIST.filter(c => c.label.toLowerCase().includes(query.toLowerCase()))

  return (
    <div className="fixed inset-0 z-[90] flex items-start justify-center pt-32"
      style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(8px)' }}
      onClick={onClose}>
      <div className="w-full max-w-md rounded-2xl overflow-hidden animate-slide-down"
        style={{
          background: 'rgba(6,6,18,0.98)',
          border: '1px solid rgba(255,255,255,0.1)',
          boxShadow: '0 32px 80px rgba(0,0,0,0.9)',
        }}
        onClick={e => e.stopPropagation()}>
        <div className="flex items-center gap-3 px-5 py-3.5 border-b" style={{ borderColor: 'rgba(255,255,255,0.06)' }}>
          <span className="text-gray-600 text-sm">⌘</span>
          <input ref={inputRef} value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Escape') onClose()
              if (e.key === 'Enter' && filtered.length > 0) { onCommand(filtered[0].id); onClose() }
            }}
            placeholder="Type a command…"
            className="flex-1 bg-transparent text-white text-sm placeholder-gray-700 outline-none" />
          <span className="text-[9px] text-gray-700 uppercase tracking-wider" style={{ fontFamily: 'var(--font-mono)' }}>Esc</span>
        </div>
        <div className="py-1.5">
          {filtered.map(cmd => (
            <button key={cmd.id} onClick={() => { onCommand(cmd.id); onClose() }}
              className="w-full flex items-center gap-4 px-5 py-2.5 hover:bg-white/[0.04] transition-colors text-left group">
              <span className="text-cyan-400/50 text-base w-5 group-hover:text-cyan-400/80 transition-colors">{cmd.icon}</span>
              <span className="text-gray-400 text-sm group-hover:text-gray-200 transition-colors">{cmd.label}</span>
            </button>
          ))}
          {!filtered.length && <p className="px-5 py-4 text-gray-700 text-sm">No matching commands</p>}
        </div>
        <div className="px-5 py-2.5 border-t" style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
          <p className="text-[9px] text-gray-700 uppercase tracking-wider" style={{ fontFamily: 'var(--font-mono)' }}>
            ↵ select · esc dismiss
          </p>
        </div>
      </div>
    </div>
  )
}

/* ── STATUS DOT ──────────────────────────────────────────────────── */
function StatusDot({ status }) {
  if (status === 'ready') return (
    <span className="w-2 h-2 rounded-full bg-emerald-400 flex-shrink-0"
      style={{ boxShadow: '0 0 6px rgba(52,211,153,0.7)' }} />
  )
  if (status === 'processing') return (
    <span className="w-3 h-3 flex-shrink-0 flex items-center justify-center">
      <span className="w-3 h-3 rounded-full border border-transparent"
        style={{ borderTopColor: '#00f5ff', animation: 'spin 1s linear infinite', display: 'block' }} />
    </span>
  )
  return (
    <span className="w-2 h-2 rounded-full bg-red-400 flex-shrink-0"
      style={{ boxShadow: '0 0 6px rgba(248,113,113,0.7)' }} />
  )
}

/* ── METRIC RING ─────────────────────────────────────────────────── */
function MetricRing({ label, value, isPercentage = false, accent = 'cyan' }) {
  const [filled, setFilled] = useState(false)
  useEffect(() => { const t = setTimeout(() => setFilled(true), 120); return () => clearTimeout(t) }, [])

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
          <circle cx="26" cy="26" r={radius} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="3" />
          <circle cx="26" cy="26" r={radius} fill="none" stroke={strokeColor} strokeWidth="3"
            strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={offset}
            transform="rotate(-90 26 26)"
            style={{ transition: 'stroke-dashoffset 1.1s cubic-bezier(0.4,0,0.2,1)', filter: `drop-shadow(0 0 5px ${glowColor})` }} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center"
          style={{ fontFamily: 'var(--font-mono)', fontSize: '8px', color: strokeColor, fontWeight: 600 }}>
          {displayValue}
        </div>
      </div>
      <span className="text-[8px] uppercase tracking-[0.18em] text-gray-600 font-bold">{label}</span>
    </div>
  )
}

/* ── SPARK LINE ──────────────────────────────────────────────────── */
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
      <polyline points={pts} fill="none" stroke="#00f5ff" strokeWidth="1.5"
        strokeLinecap="round" strokeLinejoin="round"
        style={{ filter: 'drop-shadow(0 0 3px rgba(0,245,255,0.7))' }} />
    </svg>
  )
}

/* ── SOURCE CHIP ─────────────────────────────────────────────────── */
function SourceChip({ source }) {
  const isTable    = source.section_type === 'table'
  const paperLabel = source.paper_label

  const labelBadge = paperLabel ? (
    <span className="text-[8px] font-bold px-1.5 py-0.5 rounded-full" style={{
      background: paperLabel === 'A' ? 'rgba(0,245,255,0.15)' : 'rgba(139,92,246,0.15)',
      color:      paperLabel === 'A' ? 'rgba(0,245,255,0.9)'  : 'rgba(167,139,250,0.9)',
      border:     paperLabel === 'A' ? '1px solid rgba(0,245,255,0.25)' : '1px solid rgba(139,92,246,0.25)',
    }}>{paperLabel}</span>
  ) : null

  if (isTable) return (
    <div className="group flex items-center gap-2 bg-violet-500/10 hover:bg-violet-500/18 border border-violet-500/25 hover:border-violet-400/40 text-violet-300 hover:text-violet-200 text-[10px] px-3 py-1.5 rounded-full transition-all cursor-default"
      style={{ fontFamily: 'var(--font-mono)' }}>
      {labelBadge}
      <span className="w-1 h-1 rounded-full bg-violet-400" style={{ filter: 'drop-shadow(0 0 3px rgba(167,139,250,0.8))' }} />
      TABLE · {source.section} <span className="opacity-30">·</span> p.{source.page}
    </div>
  )
  return (
    <div className="group flex items-center gap-2 bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] hover:border-cyan-500/25 text-gray-500 hover:text-cyan-300 text-[10px] px-3 py-1.5 rounded-full transition-all cursor-default"
      style={{ fontFamily: 'var(--font-mono)' }}>
      {labelBadge}
      <span className="w-1 h-1 rounded-full bg-cyan-500/60 group-hover:bg-cyan-400 transition-colors" />
      {source.section} <span className="opacity-30">·</span> p.{source.page}
    </div>
  )
}

/* ── EVIDENCE GRADING ────────────────────────────────────────────── */
function EvidenceGrading({ grading }) {
  const [expanded, setExpanded] = useState(false)
  if (!grading || grading.grading_failed || !grading.grades?.length) return null

  const sentences     = grading.grades.filter(g => g.chunk_ref !== 'header')
  const directCount   = sentences.filter(g => g.grade === 'DIRECT').length
  const inferredCount = sentences.filter(g => g.grade === 'INFERRED').length
  const removedCount  = grading.removed_count || 0

  if (directCount === 0 && inferredCount === 0 && removedCount === 0) return null

  return (
    <div className="mt-6 pt-5 border-t border-white/[0.04]">
      <div className="flex items-center justify-between mb-3">
        <p className="text-[10px] uppercase tracking-[0.2em] text-gray-600 font-semibold">Evidence Quality</p>
        {sentences.length > 0 && (
          <button onClick={() => setExpanded(!expanded)}
            className="text-[9px] uppercase tracking-wider text-gray-600 hover:text-gray-400 transition-colors font-bold">
            {expanded ? 'Collapse' : 'Breakdown'}
          </button>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        {directCount > 0 && (
          <span className="evidence-chip cyan"><span className="chip-dot cyan" />{directCount} Direct</span>
        )}
        {inferredCount > 0 && (
          <span className="evidence-chip amber"><span className="chip-dot amber" />{inferredCount} Inferred</span>
        )}
        {removedCount > 0 && (
          <span className="evidence-chip red"><span className="chip-dot red" />{removedCount} Removed</span>
        )}
      </div>
      {expanded && sentences.length > 0 && (
        <div className="mt-4 space-y-2.5 animate-slide-down">
          {sentences.map((g, i) => {
            const isRemoved = !g.kept
            const isDirect  = g.grade === 'DIRECT'
            return (
              <div key={i} className={`flex items-start gap-2 text-[11px] leading-relaxed ${isRemoved ? 'opacity-35' : ''}`}>
                <span className={`mt-1.5 chip-dot flex-shrink-0 ${isDirect ? 'cyan' : isRemoved ? 'red' : 'amber'}`}
                  style={{
                    boxShadow: isDirect ? '0 0 6px rgba(0,245,255,0.7)' : isRemoved ? '0 0 6px rgba(248,113,113,0.7)' : '0 0 6px rgba(251,191,36,0.7)',
                  }} />
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

/* ── TELEMETRY PANEL ─────────────────────────────────────────────── */
function TelemetryPanel({ confidence, attempts, plan, requestId }) {
  const stages = useMemo(() => {
    const base = confidence || 50
    const seed = [
      { name: 'Query Planning',   ms: Math.round(40  + (base * 0.4)) },
      { name: 'Retrieval',        ms: Math.round(120 + (base * 1.2)) },
      { name: 'Reranking',        ms: Math.round(30  + (base * 0.3)) },
      { name: 'Generation',       ms: Math.round(600 + (base * 4.5)) },
      { name: 'Evidence Grading', ms: Math.round(80  + (base * 0.5)) },
      { name: 'Evaluation',       ms: Math.round(50  + (base * 0.6)) },
    ]
    const total = seed.reduce((s, x) => s + x.ms, 0)
    return { stages: seed, total }
  }, [confidence])

  const reqId = requestId
    ? requestId.toUpperCase()
    : 'REQ-' + String(Math.round((confidence || 50) * 1234567.89)).padStart(8, '0')
  const maxMs = Math.max(...stages.stages.map(s => s.ms))

  return (
    <div className="bg-black/40 border border-white/[0.04] rounded-2xl px-5 py-4 mt-3 animate-slide-down"
      style={{ fontFamily: 'var(--font-mono)' }}>
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
              <div className="h-full rounded-full" style={{
                width: `${(s.ms / maxMs) * 100}%`,
                background: i % 2 === 0
                  ? 'linear-gradient(90deg, rgba(0,245,255,0.5), rgba(0,245,255,0.2))'
                  : 'linear-gradient(90deg, rgba(139,92,246,0.5), rgba(139,92,246,0.2))',
                transition: 'width 0.9s cubic-bezier(0.4,0,0.2,1)',
              }} />
            </div>
            <span className="text-[9px] text-gray-700 w-12 text-right">{s.ms} ms</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── GRADED ANSWER — typewriter + sentence tinting ───────────────── */
function GradedAnswer({ text, grades, animate }) {
  const hasRun  = useRef(false)
  const [displayed, setDisplayed] = useState(() => animate ? '' : text)
  const [done,      setDone]      = useState(!animate)

  useEffect(() => {
    if (!animate || hasRun.current || !text) { setDone(true); setDisplayed(text); return }
    hasRun.current = true
    let i = 0
    const step = Math.max(2, Math.floor(text.length / 90))
    const id = setInterval(() => {
      i = Math.min(i + step, text.length)
      setDisplayed(text.slice(0, i))
      if (i >= text.length) { clearInterval(id); setDone(true); setDisplayed(text) }
    }, 16)
    return () => clearInterval(id)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  if (!done) {
    return (
      <div className="text-gray-200 text-sm leading-relaxed whitespace-pre-wrap">
        {displayed}
        <span className="animate-pulse" style={{ color: '#00f5ff', marginLeft: 1 }}>▋</span>
      </div>
    )
  }

  // Sentence-level tinting when grades are rich enough
  const keptGrades = grades?.filter(g => g.kept && g.chunk_ref !== 'header') || []
  if (keptGrades.length >= 3) {
    const hasInferred = keptGrades.some(g => g.grade === 'INFERRED')
    return (
      <div className="text-sm leading-relaxed">
        {hasInferred && (
          <div className="flex items-center gap-3 mb-3 text-[9px]" style={{ fontFamily: 'var(--font-mono)' }}>
            <span style={{ color: 'rgba(229,231,235,0.7)' }}>● Direct</span>
            <span style={{ color: 'rgba(156,163,175,0.7)', fontStyle: 'italic' }}>● Inferred</span>
          </div>
        )}
        {keptGrades.map((g, i) => (
          <span key={i} style={{
            color: g.grade === 'DIRECT' ? 'rgba(229,231,235,1)' : 'rgba(156,163,175,0.85)',
            fontStyle: g.grade === 'INFERRED' ? 'italic' : 'normal',
          }}
            title={g.grade === 'INFERRED' ? 'Inferred — supported by context' : 'Directly evidenced'}>
            {g.sentence}{' '}
          </span>
        ))}
      </div>
    )
  }

  return (
    <div className="text-gray-200 text-sm leading-relaxed markdown-content">
      <ReactMarkdown>{text}</ReactMarkdown>
    </div>
  )
}

/* ── REASONING CHAIN ─────────────────────────────────────────────── */
function ReasoningChain({ chain }) {
  const [open, setOpen] = useState(false)
  if (!chain) return null
  return (
    <div className="mt-4">
      <button onClick={() => setOpen(o => !o)}
        className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-gray-700 hover:text-gray-400 transition-colors font-bold"
        style={{ fontFamily: 'var(--font-mono)' }}>
        <svg className={`w-3 h-3 transition-transform ${open ? 'rotate-90' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9 5l7 7-7 7" />
        </svg>
        Show reasoning
      </button>
      {open && (
        <div className="mt-3 animate-slide-down rounded-xl p-4 text-[11px] leading-relaxed"
          style={{
            background: 'rgba(0,0,0,0.4)',
            border: '1px solid rgba(255,255,255,0.05)',
            fontFamily: 'var(--font-mono)',
            color: 'rgba(100,116,139,1)',
            maxHeight: 260,
            overflowY: 'auto',
            whiteSpace: 'pre-wrap',
          }}>
          {chain}
        </div>
      )}
    </div>
  )
}

/* ── FOLLOW-UP CHIPS ─────────────────────────────────────────────── */
const FOLLOW_UPS = {
  comparison: [
    'What are the key methodological differences?',
    'Which paper provides stronger empirical evidence?',
    'What assumptions do both papers share?',
  ],
  multi_hop: [
    'Can you elaborate on the key mechanism?',
    'What evidence supports this conclusion?',
    'Are there caveats or limitations mentioned?',
  ],
  factual: [
    'What methodology was used to validate this?',
    'Are there any limitations or future work discussed?',
    'How does this compare to prior approaches?',
  ],
  analytical: [
    'What are the broader implications of this?',
    'What assumptions underlie this analysis?',
    'What do the authors suggest as next steps?',
  ],
}

function FollowUpChips({ plan, isComparison, onSelect }) {
  const type = isComparison ? 'comparison' : (plan?.answer_type || 'factual')
  const suggestions = FOLLOW_UPS[type] || FOLLOW_UPS.factual
  return (
    <div className="mt-5 pt-4 border-t border-white/[0.03]">
      <p className="text-[9px] uppercase tracking-[0.2em] text-gray-700 mb-2.5 font-bold">Ask a follow-up</p>
      <div className="flex flex-wrap gap-2">
        {suggestions.map((q, i) => (
          <button key={i} onClick={() => onSelect(q)}
            className="text-[11px] px-3 py-1.5 rounded-full transition-all hover:text-cyan-300"
            style={{
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.06)',
              color: 'rgba(107,114,128,1)',
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(0,245,255,0.25)'; e.currentTarget.style.background = 'rgba(0,245,255,0.04)' }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)'; e.currentTarget.style.background = 'rgba(255,255,255,0.02)' }}>
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

/* ── PIPELINE STEPPER ────────────────────────────────────────────── */
const PIPELINE_STAGES = [
  { key: 'planning',   label: 'Planning query'      },
  { key: 'retrieving', label: 'Retrieving passages'  },
  { key: 'reviewing',  label: 'Reviewing evidence'   },
  { key: 'drafting',   label: 'Drafting answer'      },
  { key: 'verifying',  label: 'Verifying claims'     },
]

function PipelineStepper({ progress }) {
  const seenStages  = new Set(progress.map(p => p.stage))
  const latestStage = progress[progress.length - 1]?.stage
  const latestMsg   = progress[progress.length - 1]?.message
  const retryEvent  = [...progress].reverse().find(p => p.stage === 'retrying')

  return (
    <div className="flex justify-start mb-8">
      <div className="pm-card" style={{ padding: '20px 28px', minWidth: 340 }}>
        <div className="space-y-3">
          {PIPELINE_STAGES.map(stage => {
            const isDone    = seenStages.has(stage.key) && latestStage !== stage.key
            const isActive  = latestStage === stage.key
            const isPending = !seenStages.has(stage.key) && !isActive

            return (
              <div key={stage.key} className="flex items-center gap-3">
                <div className="w-5 h-5 flex-shrink-0 flex items-center justify-center">
                  {isDone && (
                    <svg className="w-4 h-4" viewBox="0 0 16 16" fill="none">
                      <circle cx="8" cy="8" r="7" fill="rgba(0,245,255,0.1)" stroke="rgba(0,245,255,0.45)" strokeWidth="1" />
                      <path d="M5 8l2 2 4-4" stroke="#00f5ff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  )}
                  {isActive && (
                    <div className="w-3 h-3 rounded-full bg-cyan-400 animate-pulse"
                      style={{ boxShadow: '0 0 8px rgba(0,245,255,0.8)' }} />
                  )}
                  {isPending && (
                    <div className="w-2.5 h-2.5 rounded-full border border-white/10" />
                  )}
                </div>
                <span className={`text-xs transition-colors duration-300 ${
                  isDone ? 'text-cyan-400/55' : isActive ? 'text-cyan-300' : 'text-gray-700'
                }`} style={{ fontFamily: 'var(--font-mono)' }}>
                  {stage.label}
                  {isActive && latestMsg && (
                    <span className="text-gray-600 ml-2 text-[10px]">— {latestMsg}</span>
                  )}
                </span>
              </div>
            )
          })}

          {retryEvent && (
            <div className="flex items-center gap-3 pt-1">
              <div className="w-5 h-5 flex-shrink-0 flex items-center justify-center">
                <span className="text-amber-400 text-xs" style={{ display: 'inline-block', animation: 'spin 1.2s linear infinite' }}>↺</span>
              </div>
              <span className="text-xs text-amber-400/70" style={{ fontFamily: 'var(--font-mono)' }}>
                {retryEvent.message || 'Refining…'}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/* ── MESSAGE ─────────────────────────────────────────────────────── */
function Message({ msg, isNewest, onFollowUp }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [showTrace,  setShowTrace]  = useState(false)

  if (msg.role === 'user') {
    return (
      <div className="flex justify-end mb-8">
        <div className="bg-gradient-to-br from-blue-600/90 to-cyan-500/90 text-white rounded-3xl rounded-tr-sm px-6 py-4 max-w-xl shadow-xl"
          style={{ border: '1px solid rgba(255,255,255,0.1)' }}>
          <p className="text-sm leading-relaxed">{msg.content}</p>
        </div>
      </div>
    )
  }

  const { answer, confidence, faithfulness, answer_relevancy, sources, attempts,
    warning, grading, plan, is_comparison, reasoning_chain, request_id } = msg.content

  // Strip any leaked scratchpad content that appears before the ESSENCE marker
  const parseAnswer = (text) => {
    if (!text) return { essence: '', detail: '' }
    const essenceMatch = text.match(/\*{0,2}ESSENCE:?\*{0,2}/i)
    const working = essenceMatch ? text.slice(essenceMatch.index) : text
    const stripEssence = t => t.replace(/^\*{0,2}ESSENCE:?\*{0,2}\s*/i, '').trim()
    const detailRe = /\n\s*\*{0,2}DETAIL:?\*{0,2}\s*/i
    const match = working.match(detailRe)
    if (match) {
      return { essence: stripEssence(working.slice(0, match.index)), detail: working.slice(match.index + match[0].length).trim() }
    }
    return { essence: stripEssence(working), detail: '' }
  }

  const { essence, detail } = parseAnswer(answer)

  // Evidence quality border color
  const keptGrades   = grading?.grades?.filter(g => g.kept && g.chunk_ref !== 'header') || []
  const directRatio  = keptGrades.length > 0 ? keptGrades.filter(g => g.grade === 'DIRECT').length / keptGrades.length : 1
  const essenceBorder = directRatio > 0.75 ? 'rgba(0,245,255,0.18)' : directRatio > 0.45 ? 'rgba(251,191,36,0.2)' : 'rgba(139,92,246,0.18)'

  return (
    <div className="flex justify-start mb-8 group">
      <div className="max-w-3xl w-full">
        <div className="pm-card relative overflow-hidden">
          <div className="absolute top-0 left-0 right-0 h-px pointer-events-none"
            style={{ background: 'linear-gradient(90deg, transparent, rgba(0,245,255,0.25), rgba(139,92,246,0.15), transparent)' }} />

          {/* Badge row */}
          {(is_comparison || plan?.answer_type || plan?.complexity === 'multi_hop') && (
            <div className="flex items-center gap-2 mb-4">
              {is_comparison && (
                <span className="text-[8px] font-bold uppercase tracking-[0.18em] px-2.5 py-1 rounded-full"
                  style={{ background: 'linear-gradient(135deg, rgba(0,245,255,0.08), rgba(139,92,246,0.08))', border: '1px solid rgba(139,92,246,0.3)', color: 'rgba(167,139,250,0.9)', fontFamily: 'var(--font-mono)' }}>
                  COMPARISON
                </span>
              )}
              {plan?.answer_type && (
                <span className="text-[8px] font-bold uppercase tracking-[0.18em] px-2.5 py-1 rounded-full"
                  style={{ background: 'rgba(0,245,255,0.07)', border: '1px solid rgba(0,245,255,0.18)', color: 'rgba(0,245,255,0.75)', fontFamily: 'var(--font-mono)' }}>
                  {plan.answer_type}
                </span>
              )}
              {plan?.complexity === 'multi_hop' && (
                <span className="text-[8px] font-bold uppercase tracking-[0.18em] px-2.5 py-1 rounded-full"
                  style={{ background: 'rgba(139,92,246,0.07)', border: '1px solid rgba(139,92,246,0.2)', color: 'rgba(167,139,250,0.8)', fontFamily: 'var(--font-mono)' }}>
                  MULTI-HOP
                </span>
              )}
            </div>
          )}

          {/* Essence section */}
          <div className="flex items-center gap-2 mb-3">
            <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 flex-shrink-0"
              style={{ boxShadow: '0 0 8px rgba(0,245,255,0.8)' }} />
            <span className="text-[10px] uppercase tracking-[0.2em] text-cyan-400/70 font-bold">The Essence</span>
          </div>
          <div className="pl-4" style={{ borderLeft: `1px solid ${essenceBorder}` }}>
            <GradedAnswer
              text={essence}
              grades={grading?.grades}
              animate={isNewest}
            />
          </div>

          {/* Reasoning chain toggle */}
          {reasoning_chain && <ReasoningChain chain={reasoning_chain} />}

          {/* Detail accordion */}
          {detail && (
            <div className="mt-6">
              <button onClick={() => setIsExpanded(!isExpanded)}
                className="flex items-center gap-2 mb-1 group/btn">
                <div className="w-6 h-6 rounded-md flex items-center justify-center transition-all"
                  style={{ background: isExpanded ? 'rgba(139,92,246,0.18)' : 'rgba(255,255,255,0.04)', border: '1px solid rgba(139,92,246,0.2)' }}>
                  <svg className={`w-3 h-3 transition-transform duration-400 ${isExpanded ? 'rotate-180' : ''}`}
                    fill="none" stroke="rgba(167,139,250,0.8)" viewBox="0 0 24 24">
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
                    <div className="text-gray-400 text-sm leading-loose markdown-content pl-4"
                      style={{ borderLeft: '1px solid rgba(139,92,246,0.15)' }}>
                      <ReactMarkdown>{detail}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Metrics row */}
          <div className="flex items-center gap-5 mt-8 pt-6 border-t border-white/[0.04]">
            <MetricRing label="Confidence"  value={confidence}          isPercentage={true} accent="cyan"   />
            <MetricRing label="Faithfulness" value={faithfulness || 0}                      accent="violet" />
            <MetricRing label="Relevancy"   value={answer_relevancy || 0}                   accent="blue"   />
            <SparkLine confidence={confidence} />
          </div>

          {/* Sources */}
          {sources?.length > 0 && (
            <div className="mt-6 pt-5 border-t border-white/[0.04]">
              <p className="text-[10px] uppercase tracking-[0.2em] text-gray-600 mb-3 font-semibold">Evidence Sources</p>
              <div className="flex flex-wrap gap-2">
                {sources.map((s, i) => <SourceChip key={i} source={s} />)}
              </div>
            </div>
          )}

          {/* Evidence grading */}
          <EvidenceGrading grading={grading} />

          {/* Follow-up chips */}
          <FollowUpChips plan={plan} isComparison={is_comparison} onSelect={onFollowUp} />

          {/* Footer */}
          <div className="flex items-center justify-between mt-5 pt-4 border-t border-white/[0.03]">
            <div>
              {attempts > 1 && (
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"
                    style={{ boxShadow: '0 0 6px rgba(251,191,36,0.7)' }} />
                  <p className="text-[10px] font-medium text-amber-500/70 uppercase tracking-wider"
                    style={{ fontFamily: 'var(--font-mono)' }}>{attempts} Ingestion Cycles</p>
                </div>
              )}
            </div>
            <button onClick={() => setShowTrace(!showTrace)}
              className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-gray-700 hover:text-cyan-400 transition-colors px-3 py-1.5 rounded-lg hover:bg-cyan-500/08"
              style={{ fontFamily: 'var(--font-mono)' }}>
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Trace
            </button>
          </div>
        </div>

        {showTrace && <TelemetryPanel confidence={confidence} attempts={attempts} plan={plan} requestId={request_id} />}

        {warning && (
          <div className="mt-3 rounded-2xl px-5 py-3 text-[11px] leading-relaxed backdrop-blur-md"
            style={{ background: 'rgba(251,191,36,0.07)', border: '1px solid rgba(251,191,36,0.18)', color: 'rgba(253,230,138,0.75)' }}>
            <span className="text-amber-400 mr-2 font-bold">!</span>{warning}
          </div>
        )}
      </div>
    </div>
  )
}

/* ── EMPTY STATE QUESTIONS ───────────────────────────────────────── */
const STARTER_QUESTIONS = [
  'What is the main contribution of this paper?',
  'What methodology do the authors use?',
  'What are the key findings or results?',
  'Are there any limitations or future work discussed?',
]

/* ── CHAT PAGE ───────────────────────────────────────────────────── */
export default function ChatPage({ paper: initialPaper, onBack }) {
  const [paper, setPaper]                 = useState(initialPaper)
  const [allMessages, setAllMessages]     = useState({ [initialPaper.paper_id]: [] })
  const [input, setInput]                 = useState('')
  const [loading, setLoading]             = useState(false)
  const [progress, setProgress]           = useState([])
  const [papers, setPapers]               = useState([])
  const [showSwitcher, setShowSwitcher]   = useState(false)
  const [compareMode, setCompareMode]     = useState(false)
  const [comparePaper2, setComparePaper2] = useState(null)
  const [toasts, setToasts]               = useState([])
  const [showCmdPalette, setShowCmdPalette] = useState(false)
  const toastCounter = useRef(0)
  const bottomRef    = useRef()
  const textareaRef  = useRef()

  const messages = allMessages[paper.paper_id] || []

  const showToast = useCallback((message, type = 'info') => {
    const id = ++toastCounter.current
    setToasts(prev => [...prev, { id, message, type }])
  }, [])

  const dismissToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id))
  }, [])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [allMessages, loading])

  useEffect(() => {
    if (showSwitcher) {
      listPapers().then(setPapers).catch(() => showToast('Failed to load workspace', 'error'))
    }
  }, [showSwitcher]) // eslint-disable-line react-hooks/exhaustive-deps

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      // ⌘K / Ctrl+K → command palette
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault()
        setShowCmdPalette(p => !p)
      }
      // Escape → close overlays
      if (e.key === 'Escape') {
        setShowCmdPalette(false)
        setShowSwitcher(false)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const handleCommand = (cmd) => {
    if (cmd === 'upload')    onBack()
    if (cmd === 'compare')   { setCompareMode(p => !p); showToast('Compare mode ' + (!compareMode ? 'enabled' : 'disabled'), 'info') }
    if (cmd === 'clear')     setAllMessages(prev => ({ ...prev, [paper.paper_id]: [] }))
    if (cmd === 'workspace') setShowSwitcher(true)
  }

  const switchToPaper = (p) => {
    setPaper(p)
    setAllMessages(prev => ({ ...prev, [p.paper_id]: prev[p.paper_id] || [] }))
    setShowSwitcher(false)
  }

  const handleCompareWith = (p) => {
    setComparePaper2(p)
    setCompareMode(true)
    setShowSwitcher(false)
  }

  const handleDelete = async (e, p) => {
    e.stopPropagation()
    if (!window.confirm(`Delete "${p.filename}"?`)) return
    try {
      await deletePaper(p.paper_id)
      setPapers(prev => prev.filter(x => x.paper_id !== p.paper_id))
      if (comparePaper2?.paper_id === p.paper_id) setComparePaper2(null)
      showToast(`"${p.filename}" deleted`, 'info')
    } catch {
      showToast('Delete failed — paper may already be gone', 'error')
    }
  }

  const handleSend = async (questionOverride) => {
    const question = (questionOverride ?? input).trim()
    if (!question || loading) return
    setInput('')
    const paperId = paper.paper_id
    setAllMessages(prev => ({
      ...prev,
      [paperId]: [...(prev[paperId] || []), { role: 'user', content: question }],
    }))
    setLoading(true)
    setProgress([])

    const onEvent = ({ type, data }) => {
      if (type === 'progress') setProgress(prev => [...prev, { stage: data.stage, message: data.message }])
    }

    try {
      const result = (compareMode && comparePaper2)
        ? await comparePapersStream(paperId, comparePaper2.paper_id, question, onEvent)
        : await queryPaperStream(paperId, question, onEvent)
      setAllMessages(prev => ({
        ...prev,
        [paperId]: [...(prev[paperId] || []), { role: 'assistant', content: result }],
      }))
    } catch (err) {
      showToast(err?.message || 'Query failed — please try again', 'error')
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
      setProgress([])
    }
  }

  const handleFollowUp = useCallback((q) => {
    setInput(q)
    textareaRef.current?.focus()
  }, [])

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
    // ↑ recalls last sent question when input is empty
    if (e.key === 'ArrowUp' && !input.trim()) {
      e.preventDefault()
      const lastUser = [...messages].reverse().find(m => m.role === 'user')
      if (lastUser) setInput(lastUser.content)
    }
  }

  const readyPapers = papers.filter(p => p.status === 'ready')

  return (
    <div className="min-h-screen cosmic-bg flex flex-col" style={{ position: 'relative' }}>
      <CosmicOrbs />
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
      {showCmdPalette && <CommandPalette onClose={() => setShowCmdPalette(false)} onCommand={handleCommand} />}

      {/* ── HEADER ── */}
      <header className="sticky top-0 z-50 backdrop-blur-2xl border-b px-8 py-4"
        style={{ background: 'rgba(0,0,0,0.5)', borderColor: 'rgba(255,255,255,0.04)' }}>
        <div className="max-w-7xl mx-auto flex items-center justify-between">

          <div className="flex items-center gap-5">
            <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: 'linear-gradient(135deg, #00f5ff, #6366f1)', boxShadow: '0 0 14px rgba(0,245,255,0.3)' }}>
              <span className="text-[9px] font-bold text-black" style={{ fontFamily: 'var(--font-display)' }}>PM</span>
            </div>
            <div className="flex flex-col">
              <h1 className="text-white font-semibold text-sm tracking-tight truncate max-w-xs"
                style={{ fontFamily: 'var(--font-display)' }}>{paper.filename}</h1>
              <p className="text-[9px] uppercase tracking-[0.22em] text-gray-600 mt-0.5 font-semibold">Active Document</p>
            </div>
            <button onClick={() => setShowSwitcher(!showSwitcher)}
              className="flex items-center gap-2 px-4 py-2 rounded-xl transition-all duration-300"
              style={{
                background: showSwitcher ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.04)',
                color: showSwitcher ? '#000' : 'rgba(156,163,175,1)',
              }}>
              <span className="text-[10px] font-bold uppercase tracking-wider">Workspace</span>
              <svg className={`w-3 h-3 transition-transform duration-300 ${showSwitcher ? 'rotate-180' : ''}`}
                fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>

          <div className="flex items-center gap-4">
            {/* Compare pill */}
            {compareMode && comparePaper2 && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl"
                style={{ background: 'rgba(139,92,246,0.1)', border: '1px solid rgba(139,92,246,0.25)' }}>
                <span className="text-[10px] text-violet-300/60" style={{ fontFamily: 'var(--font-mono)' }}>
                  {paper.filename.replace('.pdf','').slice(0,16)}
                </span>
                <span className="text-[10px] text-violet-400">⇔</span>
                <span className="text-[10px] text-violet-300/60" style={{ fontFamily: 'var(--font-mono)' }}>
                  {comparePaper2.filename.replace('.pdf','').slice(0,16)}
                </span>
                <button onClick={() => { setCompareMode(false); setComparePaper2(null) }}
                  className="text-violet-400/50 hover:text-violet-300 transition-colors text-sm leading-none ml-1">×</button>
              </div>
            )}

            {/* ⌘K hint */}
            <button onClick={() => setShowCmdPalette(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-all"
              style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', color: 'rgba(107,114,128,1)' }}>
              <span className="text-[9px] uppercase tracking-wider" style={{ fontFamily: 'var(--font-mono)' }}>⌘K</span>
            </button>

            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"
                style={{ boxShadow: '0 0 6px rgba(52,211,153,0.7)' }} />
              <span className="text-[10px] text-emerald-400/70 font-semibold uppercase tracking-wider">Active</span>
            </div>
            <button onClick={onBack} className="group flex items-center gap-2 text-gray-500 hover:text-white transition-all">
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
          <div className="absolute top-full left-0 right-0 backdrop-blur-3xl border-b p-8 animate-slide-down"
            style={{ background: 'rgba(2,2,9,0.96)', borderColor: 'rgba(255,255,255,0.04)', zIndex: 49 }}>
            <div className="max-w-4xl mx-auto">
              <h3 className="text-[10px] uppercase tracking-[0.3em] text-gray-600 mb-5 font-bold text-center">
                Select Intelligence Context
              </h3>

              {/* Compare toggle */}
              <div className="flex items-center justify-center gap-3 mb-6">
                <span className="text-[10px] text-gray-600 uppercase tracking-wider">Compare Mode</span>
                <button onClick={() => setCompareMode(!compareMode)}
                  className="relative w-10 h-5 rounded-full transition-all duration-300 flex-shrink-0"
                  style={{
                    background: compareMode ? 'rgba(139,92,246,0.7)' : 'rgba(255,255,255,0.08)',
                    border: compareMode ? '1px solid rgba(139,92,246,0.5)' : '1px solid rgba(255,255,255,0.08)',
                    boxShadow: compareMode ? '0 0 12px rgba(139,92,246,0.4)' : 'none',
                  }}>
                  <span className="absolute top-0.5 w-4 h-4 bg-white rounded-full transition-all duration-300"
                    style={{ left: compareMode ? 'calc(100% - 18px)' : '2px' }} />
                </button>
                {compareMode && <span className="text-[10px] text-violet-400/70 uppercase tracking-wider font-bold">Active</span>}
              </div>

              {!compareMode ? (
                <div className="grid grid-cols-2 gap-3">
                  {papers.map(p => (
                    <button key={p.paper_id}
                      onClick={() => p.status === 'ready' && switchToPaper(p)}
                      className="group text-left px-5 py-4 rounded-2xl transition-all border"
                      style={{
                        background: p.paper_id === paper.paper_id ? 'rgba(0,245,255,0.05)' : 'rgba(255,255,255,0.03)',
                        borderColor: p.paper_id === paper.paper_id ? 'rgba(0,245,255,0.25)' : 'rgba(255,255,255,0.05)',
                        color: p.status === 'ready' ? (p.paper_id === paper.paper_id ? '#fff' : 'rgba(156,163,175,1)') : 'rgba(75,85,99,1)',
                        cursor: p.status === 'ready' ? 'pointer' : 'default',
                      }}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2.5 min-w-0">
                          <StatusDot status={p.status} />
                          <span className="text-sm font-medium truncate">{p.filename}</span>
                        </div>
                        <div className="flex items-center gap-2 flex-shrink-0">
                          {p.status === 'ready' && p.paper_id !== paper.paper_id && (
                            <button onClick={e => { e.stopPropagation(); handleCompareWith(p) }}
                              className="opacity-0 group-hover:opacity-100 text-[9px] px-2 py-1 rounded-lg transition-all uppercase tracking-wider font-bold"
                              style={{ background: 'rgba(139,92,246,0.1)', border: '1px solid rgba(139,92,246,0.25)', color: 'rgba(167,139,250,0.8)' }}>
                              ⇔
                            </button>
                          )}
                          <button onClick={e => handleDelete(e, p)}
                            className="w-5 h-5 rounded-md flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-500/20 hover:text-red-400 text-gray-600">
                            ×
                          </button>
                        </div>
                      </div>
                      {p.status !== 'ready' && (
                        <p className="text-[9px] mt-1.5 uppercase tracking-wider"
                          style={{ color: p.status === 'processing' ? 'rgba(0,245,255,0.5)' : 'rgba(248,113,113,0.6)', fontFamily: 'var(--font-mono)' }}>
                          {p.status}
                        </p>
                      )}
                    </button>
                  ))}
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="compare-slot">
                      <p className="text-[10px] uppercase tracking-[0.2em] text-violet-400/60 font-bold mb-3">Paper A (Active)</p>
                      <div className="space-y-2">
                        {readyPapers.map(p => (
                          <button key={p.paper_id} onClick={() => switchToPaper(p)}
                            className="w-full text-left px-4 py-2.5 rounded-xl transition-all text-xs flex items-center gap-2"
                            style={{
                              background: p.paper_id === paper.paper_id ? 'rgba(139,92,246,0.15)' : 'rgba(255,255,255,0.03)',
                              border: p.paper_id === paper.paper_id ? '1px solid rgba(139,92,246,0.35)' : '1px solid rgba(255,255,255,0.04)',
                              color: p.paper_id === paper.paper_id ? '#e9d5ff' : 'rgba(156,163,175,0.7)',
                            }}>
                            <StatusDot status={p.status} />
                            <span className="truncate">{p.filename}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="compare-slot">
                      <p className="text-[10px] uppercase tracking-[0.2em] text-violet-400/60 font-bold mb-3">Paper B</p>
                      <div className="space-y-2">
                        {readyPapers.map(p => (
                          <button key={p.paper_id}
                            onClick={() => { setComparePaper2(p); setShowSwitcher(false) }}
                            className="w-full text-left px-4 py-2.5 rounded-xl transition-all text-xs flex items-center gap-2"
                            style={{
                              background: comparePaper2?.paper_id === p.paper_id ? 'rgba(139,92,246,0.15)' : 'rgba(255,255,255,0.03)',
                              border: comparePaper2?.paper_id === p.paper_id ? '1px solid rgba(139,92,246,0.35)' : '1px solid rgba(255,255,255,0.04)',
                              color: comparePaper2?.paper_id === p.paper_id ? '#e9d5ff' : 'rgba(156,163,175,0.7)',
                            }}>
                            <StatusDot status={p.status} />
                            <span className="truncate">{p.filename}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                  {comparePaper2 && (
                    <div className="flex justify-center mt-2">
                      <button onClick={() => setShowSwitcher(false)}
                        className="compare-cta"
                        style={{
                          background: 'linear-gradient(135deg, rgba(139,92,246,0.3), rgba(0,245,255,0.15))',
                          border: '1px solid rgba(139,92,246,0.4)',
                          color: '#e9d5ff',
                          cursor: 'pointer',
                        }}>
                        Compare · {paper.filename.slice(0, 20)} ⇔ {comparePaper2.filename.slice(0, 20)}
                      </button>
                    </div>
                  )}
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
          {messages.length === 0 && !loading && (
            <div className="text-center py-16">
              <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-6"
                style={{
                  background: 'linear-gradient(145deg, rgba(0,245,255,0.08), rgba(0,245,255,0.02))',
                  border: '1px solid rgba(0,245,255,0.12)',
                  boxShadow: '0 0 30px rgba(0,245,255,0.08)',
                }}>
                <svg className="w-6 h-6 text-cyan-400/70" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5"
                    d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-white mb-2" style={{ fontFamily: 'var(--font-display)' }}>
                Initialize Context
              </h2>
              <p className="text-gray-600 text-sm font-light max-w-xs mx-auto mb-8">
                Ask anything about the paper, or start with one of these:
              </p>
              <div className="flex flex-wrap gap-2 justify-center max-w-lg mx-auto">
                {STARTER_QUESTIONS.map((q, i) => (
                  <button key={i} onClick={() => handleFollowUp(q)}
                    className="text-sm px-4 py-2 rounded-full transition-all"
                    style={{
                      background: 'rgba(0,245,255,0.04)',
                      border: '1px solid rgba(0,245,255,0.12)',
                      color: 'rgba(34,211,238,0.7)',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.background = 'rgba(0,245,255,0.08)'; e.currentTarget.style.color = 'rgba(34,211,238,1)' }}
                    onMouseLeave={e => { e.currentTarget.style.background = 'rgba(0,245,255,0.04)'; e.currentTarget.style.color = 'rgba(34,211,238,0.7)' }}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <Message
              key={i}
              msg={msg}
              isNewest={i === messages.length - 1 && msg.role === 'assistant' && !loading}
              onFollowUp={handleFollowUp}
            />
          ))}

          {loading && <PipelineStepper progress={progress} />}

          <div ref={bottomRef} className="h-40" />
        </div>
      </main>

      {/* ── INPUT DOCK ── */}
      <div className="fixed bottom-10 left-0 right-0 z-50 px-8">
        <div className="max-w-3xl mx-auto">
          <div className="pm-input-dock p-2 transition-all duration-500">
            <div className="flex items-center gap-2">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Message PaperMind Intelligence…"
                rows={1}
                className="flex-1 bg-transparent border-none px-5 py-4 text-white text-sm placeholder-gray-700 resize-none focus:ring-0 outline-none"
              />
              <button onClick={() => handleSend()}
                disabled={!input.trim() || loading}
                className="h-11 w-11 rounded-xl flex items-center justify-center transition-all duration-300 flex-shrink-0"
                style={
                  input.trim() && !loading
                    ? { background: 'linear-gradient(135deg, #00f5ff, #6366f1)', boxShadow: '0 0 20px rgba(0,245,255,0.3)', color: '#000' }
                    : { background: 'rgba(255,255,255,0.04)', color: 'rgba(75,85,99,1)' }
                }>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                </svg>
              </button>
            </div>
          </div>
          <p className="text-center text-gray-700 text-[10px] uppercase tracking-[0.2em] mt-3 font-medium">
            Enter to stream · Shift+Enter for newline · ↑ recall · ⌘K commands
          </p>
        </div>
      </div>
    </div>
  )
}
