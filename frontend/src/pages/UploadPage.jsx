import { useState, useRef, useCallback, useEffect } from 'react'
import { uploadPaper, getPaperStatus, listPapers } from '../api'
import { track } from '../analytics'
import Wordmark from '../components/Wordmark'
import NavButton from '../components/NavButton'
import CitedAnswer from '../components/CitedAnswer'

/* Single restrained accent for the whole page. */
const ACCENT = '#22d3ee'

/* ─────────────────────────────────────────────────────────────────
   BACKDROP — quiet dot grid, faded at the edges. No orbs, no glass.
   Editorial: the type carries the page, the background stays out of it.
───────────────────────────────────────────────────────────────── */
function Backdrop() {
  return (
    <div
      className="fixed inset-0 pointer-events-none"
      style={{
        backgroundImage: 'radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px)',
        backgroundSize: '26px 26px',
        maskImage: 'radial-gradient(ellipse 90% 70% at 50% 25%, #000 35%, transparent 100%)',
        WebkitMaskImage: 'radial-gradient(ellipse 90% 70% at 50% 25%, #000 35%, transparent 100%)',
        zIndex: 0,
      }}
    />
  )
}

/* Derive home-page stats from saved chat history (read once, at mount). */
function computeSessionStats() {
  try {
    const saved = localStorage.getItem('papermind_chats')
    if (!saved) return null
    const msgs = Object.values(JSON.parse(saved))
      .flat()
      .filter(m => m.role === 'assistant' && m.content?.confidence != null)
    if (msgs.length === 0) return null
    const avgConf  = msgs.reduce((s, m) => s + (m.content.confidence || 0), 0) / msgs.length
    const avgFaith = msgs.reduce((s, m) => s + (m.content.faithfulness || 0), 0) / msgs.length
    const multiHop = msgs.filter(m => m.content.plan?.complexity === 'multi_hop').length
    const n = msgs.length
    return [
      { label: 'avg confidence', value: avgConf.toFixed(1) + '%' },
      { label: 'faithfulness',   value: (avgFaith * 100).toFixed(1) + '%' },
      { label: 'multi-hop',      value: ((multiHop / n) * 100).toFixed(0) + '%' },
    ]
  } catch {
    return null
  }
}

/* ─────────────────────────────────────────────────────────────────
   CAPABILITY ROW — an editorial, clickable entry point. Replaces the
   "nothing shows your power" emptiness: each of the real surfaces
   (Discover, Drafts/write-mode, Rewrite, Library) is a real link, laid
   out as a typographic row with a mono index and a hairline divider —
   not a glossy feature card.
───────────────────────────────────────────────────────────────── */
function CapabilityRow({ index, label, desc, onClick, badge, last }) {
  return (
    <button
      onClick={onClick}
      className="group w-full flex items-center gap-5 py-5 text-left transition-colors"
      style={{ borderBottom: last ? 'none' : '1px solid rgba(255,255,255,0.07)' }}
    >
      <span className="text-[11px] text-gray-600 w-6 flex-shrink-0" style={{ fontFamily: 'var(--font-mono)' }}>
        {index}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-[15px] font-semibold text-gray-200 group-hover:text-white transition-colors" style={{ fontFamily: 'var(--font-display)' }}>
            {label}
          </span>
          {badge != null && badge > 0 && (
            <span className="text-[10px] font-semibold px-1.5 py-px rounded-full" style={{ background: `${ACCENT}22`, color: ACCENT, fontFamily: 'var(--font-mono)' }}>
              {badge}
            </span>
          )}
        </div>
        <p className="text-[13px] text-gray-500 mt-0.5 leading-relaxed">{desc}</p>
      </div>
      <svg
        className="w-4 h-4 text-gray-700 group-hover:text-gray-300 group-hover:translate-x-0.5 transition-all flex-shrink-0"
        fill="none" stroke="currentColor" viewBox="0 0 24 24"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
      </svg>
    </button>
  )
}

/* ─────────────────────────────────────────────────────────────────
   HOME (post-sign-in)
───────────────────────────────────────────────────────────────── */
export default function UploadPage({ onPaperReady, onDiscover, onLibrary, onDrafts, onRewrite, onBilling, onAdmin }) {
  const [dragging, setDragging]             = useState(false)
  const [phase, setPhase]                   = useState('idle') // idle | uploading | processing | error
  const [filename, setFilename]             = useState('')
  const [error, setError]                   = useState('')
  const [quotaHit, setQuotaHit]             = useState(false) // last error was a 429 quota cap
  const [uploadProgress, setUploadProgress] = useState(0)
  const [paperCount, setPaperCount]         = useState(null)
  const inputRef = useRef()

  useEffect(() => {
    listPapers().then(papers => setPaperCount(papers.filter(p => p.status === 'ready').length)).catch(() => {})
  }, [])

  const [stats] = useState(computeSessionStats)

  const handleFile = useCallback(async (file) => {
    if (!file || !file.name.endsWith('.pdf')) {
      setError('Please upload a PDF file.')
      return
    }
    setError('')
    setFilename(file.name)
    setPhase('uploading')
    setUploadProgress(20)

    try {
      const { paper_id } = await uploadPaper(file)
      track('paper_uploaded')
      setUploadProgress(100)
      setTimeout(() => { setPhase('processing') }, 500)

      const interval = setInterval(async () => {
        try {
          const status = await getPaperStatus(paper_id)
          if (status.status === 'ready') {
            clearInterval(interval)
            onPaperReady(status)
          } else if (status.status === 'failed') {
            clearInterval(interval)
            setPhase('error')
            setError(status.error || 'Ingestion failed.')
          }
        } catch {
          clearInterval(interval)
          setPhase('error')
          setError('Lost connection while processing.')
        }
      }, 2000)
    } catch (err) {
      setPhase('error')
      setQuotaHit(err.status === 429)
      setError(err.message || 'Upload failed. Is the server running?')
    }
  }, [onPaperReady])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }, [handleFile])

  const onDragOver  = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)
  const openPicker  = () => inputRef.current && inputRef.current.click()

  return (
    <div className="min-h-screen flex flex-col" style={{ background: '#0a0b0e', position: 'relative' }}>
      <Backdrop />

      {/* ── TOP NAV — sticky, solid, unmistakably visible ── */}
      <nav
        className="sticky top-0 z-20 flex items-center justify-between px-6 md:px-10 py-4"
        style={{ background: 'rgba(10,11,14,0.85)', borderBottom: '1px solid rgba(255,255,255,0.07)' }}
      >
        <Wordmark accent={ACCENT} />
        <div className="flex items-center gap-2">
          {onDiscover && <NavButton onClick={onDiscover}>Discover</NavButton>}
          {onLibrary  && <NavButton onClick={onLibrary} badge={paperCount}>Library</NavButton>}
          {onDrafts   && <NavButton onClick={onDrafts}>Drafts</NavButton>}
          {onRewrite  && <NavButton onClick={onRewrite}>Rewrite</NavButton>}
          {onBilling  && <NavButton onClick={onBilling}>Plan</NavButton>}
          {onAdmin    && <NavButton onClick={onAdmin}>Admin</NavButton>}
          <div className="w-px h-5 mx-1" style={{ background: 'rgba(255,255,255,0.1)' }} />
          <NavButton primary onClick={openPicker}>New paper</NavButton>
        </div>
      </nav>

      {/* ── HERO ── */}
      <section className="relative z-10 px-6 md:px-10 pt-16 md:pt-24 pb-20">
        <div className="w-full max-w-6xl mx-auto grid md:grid-cols-2 gap-12 lg:gap-16 items-center">

          {/* LEFT — copy + dropzone */}
          <div>
            <div
              className="ed-reveal inline-flex items-center gap-2 mb-7 px-3 py-1 rounded-full"
              style={{ animationDelay: '0ms', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: ACCENT, boxShadow: `0 0 6px ${ACCENT}` }} />
              <span className="text-[11px] text-gray-400 tracking-wide" style={{ fontFamily: 'var(--font-mono)' }}>
                evidence-graded · benchmarked on QASPER
              </span>
            </div>

            <h1
              className="ed-reveal text-5xl md:text-6xl font-semibold mb-6 tracking-[-0.02em] leading-[1.03] text-white"
              style={{ animationDelay: '80ms', fontFamily: 'var(--font-display)' }}
            >
              Ask any paper.<br />
              Get answers you can{' '}
              <span style={{ position: 'relative', whiteSpace: 'nowrap' }}>
                verify
                <span
                  className="ed-underline"
                  style={{ position: 'absolute', left: 0, right: 0, bottom: 4, height: 10, background: `${ACCENT}55`, borderRadius: 2, zIndex: -1 }}
                />
              </span>
              <span className="ed-caret" style={{ color: ACCENT }}>.</span>
            </h1>

            <p
              className="ed-reveal text-gray-400 text-[15px] max-w-md leading-relaxed mb-8"
              style={{ animationDelay: '160ms' }}
            >
              Drop a research PDF and ask anything. Every claim links to the exact
              line it came from — no invented citations.
            </p>

            {/* ── DROPZONE ── */}
            {(phase === 'idle' || phase === 'error') && (
              <div className="ed-reveal" style={{ animationDelay: '240ms' }}>
                <div
                  onClick={openPicker}
                  onDrop={onDrop}
                  onDragOver={onDragOver}
                  onDragLeave={onDragLeave}
                  className="group relative overflow-hidden rounded-2xl cursor-pointer transition-all duration-300"
                  style={{
                    background: dragging ? `${ACCENT}0d` : 'rgba(255,255,255,0.02)',
                    border: `1px ${dragging ? 'solid' : 'dashed'} ${dragging ? `${ACCENT}88` : 'rgba(255,255,255,0.16)'}`,
                    boxShadow: dragging ? `0 0 0 4px ${ACCENT}1a` : 'none',
                  }}
                >
                  <div className="px-6 py-7 flex items-center gap-5">
                    <div
                      className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors duration-300"
                      style={{
                        background: dragging ? `${ACCENT}22` : 'rgba(255,255,255,0.05)',
                        border: `1px solid ${dragging ? `${ACCENT}55` : 'rgba(255,255,255,0.09)'}`,
                      }}
                    >
                      <svg className="w-6 h-6 transition-colors duration-300"
                        style={{ color: dragging ? ACCENT : '#9ca3af' }}
                        fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75"
                          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-[15px] font-medium text-white mb-0.5">
                        {dragging ? 'Release to upload' : 'Drop a paper PDF'}
                      </div>
                      <div className="text-[13px] text-gray-500">
                        or <span className="text-gray-300 underline underline-offset-2">browse files</span> — preprints, published papers, reports
                      </div>
                    </div>
                  </div>

                  <input
                    ref={inputRef}
                    type="file"
                    accept=".pdf"
                    className="hidden"
                    onChange={e => handleFile(e.target.files[0])}
                  />
                </div>

                {/* ── ERROR ── */}
                {phase === 'error' && (
                  <div
                    className="mt-4 px-4 py-3 rounded-xl flex items-center gap-3"
                    style={{ background: 'rgba(248,113,113,0.07)', border: '1px solid rgba(248,113,113,0.22)' }}
                  >
                    <span className="text-red-400 text-sm font-bold flex-shrink-0">!</span>
                    <p className="text-red-200/80 text-[13px] flex-1">{error}</p>
                    {quotaHit && onBilling ? (
                      <button
                        onClick={onBilling}
                        className="text-xs font-semibold px-2.5 py-1 rounded-lg transition-opacity flex-shrink-0"
                        style={{ background: ACCENT, color: '#0a0b0e' }}
                      >
                        Upgrade
                      </button>
                    ) : (
                      <button
                        onClick={() => setPhase('idle')}
                        className="text-red-400/70 hover:text-red-300 text-xs transition-colors flex-shrink-0"
                      >
                        Try again
                      </button>
                    )}
                  </div>
                )}

                {/* ── session stats — quiet inline row ── */}
                {stats && (
                  <div className="mt-6 flex items-center gap-5 flex-wrap">
                    {stats.map((s, i) => (
                      <div key={i} className="flex items-baseline gap-1.5">
                        <span className="text-sm font-semibold text-white" style={{ fontFamily: 'var(--font-mono)' }}>
                          {s.value}
                        </span>
                        <span className="text-[11px] text-gray-600">{s.label}</span>
                        {i < stats.length - 1 && <span className="text-gray-800 ml-3">·</span>}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* ── UPLOADING / PROCESSING ── */}
            {(phase === 'uploading' || phase === 'processing') && (
              <div
                className="rounded-2xl px-6 py-7"
                style={{ background: 'rgba(255,255,255,0.025)', border: `1px solid ${ACCENT}22` }}
              >
                <div className="flex items-center gap-4 mb-5">
                  <div className="relative w-10 h-10 flex-shrink-0">
                    <div className="absolute inset-0 rounded-full border-2"
                      style={{ borderColor: 'rgba(255,255,255,0.08)' }} />
                    <div className="absolute inset-0 rounded-full border-2 border-transparent"
                      style={{ borderTopColor: ACCENT, animation: 'spin 0.9s linear infinite' }} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-[15px] font-medium text-white">
                      {phase === 'uploading' ? 'Uploading…' : 'Processing paper'}
                    </div>
                    <div className="text-[13px] text-gray-500 truncate">
                      {phase === 'uploading'
                        ? `Sending ${filename}`
                        : `Parsing, chunking & indexing ${filename}`}
                    </div>
                  </div>
                </div>

                {phase === 'processing' && (
                  <div className="flex items-center gap-2 mb-5">
                    {['Parse', 'Chunk', 'Embed', 'Index'].map((s, i) => (
                      <div key={s} className="flex items-center gap-2 flex-1">
                        <span className="text-[10px] text-gray-500 uppercase tracking-wider"
                          style={{ fontFamily: 'var(--font-mono)' }}>{s}</span>
                        {i < 3 && <div className="flex-1 h-px" style={{ background: 'rgba(255,255,255,0.08)' }} />}
                      </div>
                    ))}
                  </div>
                )}

                <div className="w-full h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${phase === 'processing' ? 'animate-pulse' : ''}`}
                    style={{
                      width: phase === 'uploading' ? `${uploadProgress}%` : '100%',
                      background: ACCENT,
                    }}
                  />
                </div>
              </div>
            )}
          </div>

          {/* RIGHT — product proof */}
          <div className="hidden md:block ed-reveal" style={{ animationDelay: '200ms' }}>
            <CitedAnswer />
          </div>
        </div>
      </section>

      {/* ── CAPABILITIES — the real surfaces, as editorial entry points ── */}
      <section className="relative z-10 px-6 md:px-10 pb-24">
        <div className="w-full max-w-4xl mx-auto">
          <div className="flex items-baseline justify-between mb-2">
            <h2 className="text-[13px] uppercase tracking-[0.18em] text-gray-500" style={{ fontFamily: 'var(--font-mono)' }}>
              Everything you can do
            </h2>
            <span className="text-[11px] text-gray-700" style={{ fontFamily: 'var(--font-mono)' }}>PaperMind</span>
          </div>
          <div style={{ borderTop: '1px solid rgba(255,255,255,0.09)' }}>
            {(() => {
              const rows = [
                onDiscover && { label: 'Discover', desc: 'Search across the literature and import papers by topic.', onClick: onDiscover },
                onLibrary  && { label: 'Library',  desc: 'Everything you\'ve processed, ready to reopen and query.', onClick: onLibrary, badge: paperCount },
                onDrafts   && { label: 'Write mode', desc: 'Pre-submission audits on your own draft — claim grounding, numbers, venue fit, and novelty.', onClick: onDrafts },
                onRewrite  && { label: 'Rewrite',  desc: 'Tighten a passage without losing its meaning or citations.', onClick: onRewrite },
              ].filter(Boolean)
              return rows.map((r, i) => (
                <CapabilityRow
                  key={r.label}
                  index={String(i + 1).padStart(2, '0')}
                  label={r.label}
                  desc={r.desc}
                  onClick={r.onClick}
                  badge={r.badge}
                  last={i === rows.length - 1}
                />
              ))
            })()}
          </div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <div className="relative z-10 mt-auto px-6 md:px-10 py-5 flex items-center justify-between"
        style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
        <span className="text-[11px] text-gray-700" style={{ fontFamily: 'var(--font-mono)' }}>
          PaperMind
        </span>
        <div className="flex items-center gap-4">
          <a href="#/terms" className="text-[11px] text-gray-600 hover:text-gray-400 transition-colors">Terms</a>
          <a href="#/privacy" className="text-[11px] text-gray-600 hover:text-gray-400 transition-colors">Privacy</a>
          <span className="text-[11px] text-gray-700 hidden sm:inline">
            Cites the exact section · evidence-graded
          </span>
        </div>
      </div>
    </div>
  )
}
