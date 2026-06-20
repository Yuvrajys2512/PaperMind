import { useState, useRef, useCallback, useEffect } from 'react'
import { uploadPaper, getPaperStatus, listPapers } from '../api'
import { track } from '../analytics'

/* Single restrained accent for the whole page. */
const ACCENT = '#22d3ee'

/* ─────────────────────────────────────────────────────────────────
   BACKGROUND — quiet dot grid + one soft top glow. No orbs, no scan.
───────────────────────────────────────────────────────────────── */
function Backdrop() {
  return (
    <>
      {/* dot grid */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: 'radial-gradient(rgba(255,255,255,0.045) 1px, transparent 1px)',
          backgroundSize: '24px 24px',
          maskImage: 'radial-gradient(ellipse 80% 60% at 50% 30%, #000 40%, transparent 100%)',
          WebkitMaskImage: 'radial-gradient(ellipse 80% 60% at 50% 30%, #000 40%, transparent 100%)',
          zIndex: 0,
        }}
      />
      {/* one soft accent glow, top-center — subtle, not a floating orb */}
      <div
        className="fixed pointer-events-none"
        style={{
          top: '-220px', left: '50%', transform: 'translateX(-50%)',
          width: 900, height: 500,
          background: `radial-gradient(ellipse at center, ${ACCENT}14 0%, transparent 65%)`,
          filter: 'blur(40px)',
          zIndex: 0,
        }}
      />
    </>
  )
}

/* Derive landing-page stats from saved chat history (read once, at mount). */
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
   TOP NAV — replaces the floating pill buttons at the bottom.
───────────────────────────────────────────────────────────────── */
function NavLink({ children, onClick, badge }) {
  return (
    <button
      onClick={onClick}
      className="group relative flex items-center gap-1.5 text-sm text-gray-400 hover:text-white transition-colors duration-200"
    >
      {children}
      {badge != null && badge > 0 && (
        <span
          className="text-[10px] font-semibold px-1.5 py-px rounded-full"
          style={{ background: `${ACCENT}22`, color: ACCENT, fontFamily: 'var(--font-mono)' }}
        >
          {badge}
        </span>
      )}
      <span
        className="absolute -bottom-1 left-0 h-px w-0 group-hover:w-full transition-all duration-300"
        style={{ background: ACCENT }}
      />
    </button>
  )
}

/* ─────────────────────────────────────────────────────────────────
   PRODUCT PROOF — a static, real-looking cited answer. The actual
   reason to trust the product, instead of three floating stat cards.
───────────────────────────────────────────────────────────────── */
function CitationPreview() {
  return (
    <div
      className="rounded-2xl p-5 w-full"
      style={{
        background: 'linear-gradient(160deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012))',
        border: '1px solid rgba(255,255,255,0.07)',
        backdropFilter: 'blur(20px)',
      }}
    >
      {/* fake window chrome — signals "this is the product" */}
      <div className="flex items-center gap-1.5 mb-4">
        <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'rgba(255,255,255,0.12)' }} />
        <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'rgba(255,255,255,0.12)' }} />
        <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'rgba(255,255,255,0.12)' }} />
        <span className="ml-2 text-[10px] text-gray-600 truncate" style={{ fontFamily: 'var(--font-mono)' }}>
          attention-is-all-you-need.pdf
        </span>
      </div>

      {/* question */}
      <div className="flex justify-end mb-3">
        <div className="text-sm text-gray-200 px-3.5 py-2 rounded-2xl rounded-br-md max-w-[80%]"
          style={{ background: 'rgba(255,255,255,0.06)' }}>
          What was the main result?
        </div>
      </div>

      {/* answer */}
      <div className="text-sm text-gray-300 leading-relaxed mb-3">
        The Transformer reached <span className="text-white font-medium">28.4 BLEU</span> on
        WMT14 EN→DE, improving over the best prior result by more than 2 BLEU
        <button
          className="align-super text-[10px] font-semibold px-1.5 py-0.5 rounded-md mx-0.5 transition-colors"
          style={{ background: `${ACCENT}1a`, color: ACCENT, fontFamily: 'var(--font-mono)' }}
        >
          §6.2
        </button>
        while training in a fraction of the time.
      </div>

      {/* evidence footer */}
      <div className="flex items-center gap-3 pt-3" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
        <span className="flex items-center gap-1.5 text-[11px] text-gray-500" style={{ fontFamily: 'var(--font-mono)' }}>
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: ACCENT, boxShadow: `0 0 6px ${ACCENT}` }} />
          grounded in source
        </span>
        <span className="text-[11px] text-gray-600" style={{ fontFamily: 'var(--font-mono)' }}>
          confidence 94%
        </span>
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────────────────────────
   UPLOAD PAGE
───────────────────────────────────────────────────────────────── */
export default function UploadPage({ onPaperReady, onDiscover, onLibrary, onRewrite, onBilling, onAdmin }) {
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

  return (
    <div className="min-h-screen flex flex-col" style={{ background: '#0a0b0e', position: 'relative' }}>
      <Backdrop />

      {/* ── TOP NAV ── */}
      <nav className="relative z-10 flex items-center justify-between px-6 md:px-10 py-5">
        <div className="flex items-center gap-2.5">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center"
            style={{ background: `${ACCENT}1a`, border: `1px solid ${ACCENT}33` }}
          >
            <svg className="w-4 h-4" fill="none" stroke={ACCENT} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
          </div>
          <span className="text-[15px] font-semibold text-white tracking-tight"
            style={{ fontFamily: 'var(--font-display)' }}>
            PaperMind
          </span>
        </div>

        <div className="flex items-center gap-6">
          {onDiscover && <NavLink onClick={onDiscover}>Discover</NavLink>}
          {onLibrary  && <NavLink onClick={onLibrary} badge={paperCount}>Library</NavLink>}
          {onRewrite  && <NavLink onClick={onRewrite}>Rewrite</NavLink>}
          {onBilling  && <NavLink onClick={onBilling}>Plan</NavLink>}
          {onAdmin    && <NavLink onClick={onAdmin}>Admin</NavLink>}
        </div>
      </nav>

      {/* ── HERO ── */}
      <div className="relative z-10 flex-1 flex items-center px-6 md:px-10 pb-16">
        <div className="w-full max-w-6xl mx-auto grid md:grid-cols-2 gap-10 lg:gap-16 items-center">

          {/* LEFT — copy + dropzone */}
          <div>
            {/* eyebrow */}
            <div className="inline-flex items-center gap-2 mb-6 px-3 py-1 rounded-full"
              style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)' }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: ACCENT, boxShadow: `0 0 6px ${ACCENT}` }} />
              <span className="text-[11px] text-gray-400 tracking-wide" style={{ fontFamily: 'var(--font-mono)' }}>
                evidence-graded answers
              </span>
            </div>

            <h1
              className="text-4xl md:text-5xl font-semibold mb-5 tracking-tight leading-[1.08] text-white"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              Ask questions.<br />
              Get answers you can&nbsp;
              <span style={{ position: 'relative', whiteSpace: 'nowrap' }}>
                verify
                <span style={{
                  position: 'absolute', left: 0, right: 0, bottom: 2, height: 8,
                  background: `${ACCENT}55`, borderRadius: 2, zIndex: -1,
                }} />
              </span>.
            </h1>

            <p className="text-gray-400 text-[15px] max-w-md leading-relaxed mb-8">
              Drop any research PDF and ask anything. Every claim links to the
              exact section it came from — no invented citations.
            </p>

            {/* ── DROPZONE ── */}
            {(phase === 'idle' || phase === 'error') && (
              <>
                <div
                  onClick={() => inputRef.current.click()}
                  onDrop={onDrop}
                  onDragOver={onDragOver}
                  onDragLeave={onDragLeave}
                  className="group relative overflow-hidden rounded-2xl cursor-pointer transition-all duration-300"
                  style={{
                    background: dragging ? `${ACCENT}0d` : 'rgba(255,255,255,0.02)',
                    border: `1px ${dragging ? 'solid' : 'dashed'} ${dragging ? `${ACCENT}88` : 'rgba(255,255,255,0.14)'}`,
                    boxShadow: dragging ? `0 0 0 4px ${ACCENT}1a` : 'none',
                  }}
                >
                  <div className="px-6 py-8 flex items-center gap-5">
                    <div
                      className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors duration-300"
                      style={{
                        background: dragging ? `${ACCENT}22` : 'rgba(255,255,255,0.05)',
                        border: `1px solid ${dragging ? `${ACCENT}55` : 'rgba(255,255,255,0.08)'}`,
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

                {/* ── session stats — quiet inline row, not big cards ── */}
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
              </>
            )}

            {/* ── UPLOADING / PROCESSING ── */}
            {(phase === 'uploading' || phase === 'processing') && (
              <div
                className="rounded-2xl px-6 py-7"
                style={{ background: 'rgba(255,255,255,0.025)', border: `1px solid ${ACCENT}22` }}
              >
                <div className="flex items-center gap-4 mb-5">
                  {/* single refined ring */}
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

                {/* stages */}
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

                {/* progress bar */}
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
          <div className="hidden md:block">
            <CitationPreview />
          </div>
        </div>
      </div>

      {/* ── FOOTER ── */}
      <div className="relative z-10 px-6 md:px-10 py-5 flex items-center justify-between"
        style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
        <span className="text-[11px] text-gray-700" style={{ fontFamily: 'var(--font-mono)' }}>
          PaperMind
        </span>
        <div className="flex items-center gap-4">
          <a href="#/terms" className="text-[11px] text-gray-700 hover:text-gray-500 transition-colors">Terms</a>
          <a href="#/privacy" className="text-[11px] text-gray-700 hover:text-gray-500 transition-colors">Privacy</a>
          <span className="text-[11px] text-gray-700 hidden sm:inline">
            Cites the exact section · evidence-graded
          </span>
        </div>
      </div>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}
