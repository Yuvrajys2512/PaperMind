import { useState, useRef, useCallback } from 'react'
import { uploadPaper, getPaperStatus } from '../api'

/* ─────────────────────────────────────────────────────────────────
   COSMIC ORBS  (same as ChatPage)
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
   UPLOAD PAGE
───────────────────────────────────────────────────────────────── */
export default function UploadPage({ onPaperReady, onDiscover }) {
  const [dragging, setDragging]             = useState(false)
  const [phase, setPhase]                   = useState('idle') // idle | uploading | processing | error
  const [filename, setFilename]             = useState('')
  const [error, setError]                   = useState('')
  const [uploadProgress, setUploadProgress] = useState(0)
  const inputRef = useRef()

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
    } catch {
      setPhase('error')
      setError('Upload failed. Is the server running?')
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

  /* static stats */
  const stats = [
    { label: 'Avg Confidence', value: '73.4%',  sub: 'across 1.2k queries' },
    { label: 'Evidence Grade', value: '94.1%',  sub: 'sentences kept' },
    { label: 'Multi-hop Rate', value: '38%',    sub: 'of complex queries' },
  ]

  return (
    <div className="min-h-screen cosmic-bg flex flex-col items-center justify-center px-4 pb-20"
      style={{ position: 'relative' }}>
      <CosmicOrbs />

      {/* Decorative grid overlay */}
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0,245,255,0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,245,255,0.015) 1px, transparent 1px)
          `,
          backgroundSize: '60px 60px',
          zIndex: 0,
        }}
      />

      {/* ── HERO ── */}
      <div className="relative z-10 mb-12 text-center">
        {/* Badge pill */}
        <div
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-7"
          style={{
            background: 'rgba(0,245,255,0.06)',
            border: '1px solid rgba(0,245,255,0.14)',
          }}
        >
          <span
            className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse"
            style={{ boxShadow: '0 0 6px rgba(0,245,255,0.7)' }}
          />
          <span className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-400/70">
            Powered by PaperMind AI
          </span>
        </div>

        {/* Heading */}
        <h1
          className="text-5xl md:text-6xl font-bold mb-5 tracking-tight leading-tight"
          style={{ fontFamily: 'var(--font-display)' }}
        >
          The Future of Research
          <br />
          <span
            className="text-transparent bg-clip-text"
            style={{
              backgroundImage: 'linear-gradient(90deg, #22d3ee, #60a5fa, #a78bfa)',
              WebkitBackgroundClip: 'text',
            }}
          >
            Starts with PaperMind
          </span>
        </h1>

        {/* Subtitle */}
        <p className="text-gray-500 text-base max-w-lg mx-auto font-light leading-relaxed">
          Upload your research papers and interact with them like never before.
          Context-aware, grounded, and incredibly fast.
        </p>
      </div>

      {/* ── STATS ROW ── */}
      <div className="relative z-10 flex gap-4 mb-10 flex-wrap justify-center">
        {stats.map((s, i) => (
          <div
            key={i}
            className="flex flex-col items-center px-5 py-4 rounded-2xl"
            style={{
              background: 'linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01))',
              border: '1px solid rgba(255,255,255,0.06)',
              backdropFilter: 'blur(16px)',
              minWidth: 120,
            }}
          >
            <span
              className="text-2xl font-bold text-white mb-0.5"
              style={{ fontFamily: 'var(--font-mono)' }}
            >
              {s.value}
            </span>
            <span className="text-[9px] font-bold uppercase tracking-[0.18em] text-gray-500 mb-0.5">{s.label}</span>
            <span className="text-[9px] text-gray-700">{s.sub}</span>
          </div>
        ))}
      </div>

      {/* ── MAIN ACTION AREA ── */}
      <div className="relative z-10 w-full max-w-2xl">

        {/* ── DROPZONE ── */}
        {(phase === 'idle' || phase === 'error') && (
          <div
            onClick={() => inputRef.current.click()}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            className="relative overflow-hidden rounded-3xl cursor-pointer transition-all duration-500"
            style={{
              background: 'linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01))',
              border: dragging
                ? '1px solid rgba(0,245,255,0.45)'
                : '1px solid rgba(255,255,255,0.07)',
              backdropFilter: 'blur(24px)',
              boxShadow: dragging
                ? '0 0 40px rgba(0,245,255,0.15), inset 0 0 40px rgba(0,245,255,0.03)'
                : 'none',
              transform: dragging ? 'scale(1.015)' : 'scale(1)',
            }}
          >
            {/* Gradient glow border overlay on hover/drag */}
            <div
              className="absolute inset-0 rounded-3xl pointer-events-none transition-opacity duration-500"
              style={{
                background: 'linear-gradient(135deg, rgba(0,245,255,0.05), rgba(139,92,246,0.03), rgba(236,72,153,0.02))',
                opacity: dragging ? 1 : 0,
              }}
            />

            {/* Corner accent decorations */}
            {/* top-left */}
            <div className="absolute top-4 left-4 pointer-events-none">
              <div style={{ width: 16, height: 2, background: 'rgba(0,245,255,0.4)', borderRadius: 1 }} />
              <div style={{ width: 2, height: 14, background: 'rgba(0,245,255,0.4)', borderRadius: 1, marginTop: 0 }} />
            </div>
            {/* top-right */}
            <div className="absolute top-4 right-4 pointer-events-none flex flex-col items-end">
              <div style={{ width: 16, height: 2, background: 'rgba(0,245,255,0.4)', borderRadius: 1 }} />
              <div style={{ width: 2, height: 14, background: 'rgba(0,245,255,0.4)', borderRadius: 1 }} />
            </div>
            {/* bottom-left */}
            <div className="absolute bottom-4 left-4 pointer-events-none flex flex-col justify-end">
              <div style={{ width: 2, height: 14, background: 'rgba(0,245,255,0.4)', borderRadius: 1 }} />
              <div style={{ width: 16, height: 2, background: 'rgba(0,245,255,0.4)', borderRadius: 1 }} />
            </div>
            {/* bottom-right */}
            <div className="absolute bottom-4 right-4 pointer-events-none flex flex-col items-end justify-end">
              <div style={{ width: 2, height: 14, background: 'rgba(0,245,255,0.4)', borderRadius: 1 }} />
              <div style={{ width: 16, height: 2, background: 'rgba(0,245,255,0.4)', borderRadius: 1 }} />
            </div>

            {/* Content */}
            <div className="p-12 py-16 flex flex-col items-center justify-center">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6"
                style={{
                  background: 'linear-gradient(135deg, rgba(0,245,255,0.15), rgba(99,102,241,0.12))',
                  border: '1px solid rgba(0,245,255,0.2)',
                  boxShadow: '0 0 24px rgba(0,245,255,0.1)',
                }}
              >
                <svg className="w-8 h-8 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75"
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>

              <h2
                className="text-2xl font-semibold text-white mb-2"
                style={{ fontFamily: 'var(--font-display)' }}
              >
                {dragging ? 'Drop to Ingest' : 'Ingest Research Paper'}
              </h2>
              <p className="text-gray-600 text-center font-light mb-8 max-w-xs text-sm">
                Drag and drop your PDF here, or click to browse your workspace files.
              </p>

              <button
                className="px-6 py-2.5 rounded-full bg-white text-black text-sm font-semibold hover:bg-gray-100 transition-colors shadow-lg shadow-white/10 pointer-events-none"
              >
                Select PDF
              </button>
            </div>

            <input
              ref={inputRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={e => handleFile(e.target.files[0])}
            />
          </div>
        )}

        {/* ── UPLOADING / PROCESSING ── */}
        {(phase === 'uploading' || phase === 'processing') && (
          <div
            className="rounded-3xl p-12 py-14 flex flex-col items-center justify-center"
            style={{
              background: 'linear-gradient(145deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01))',
              border: '1px solid rgba(0,245,255,0.14)',
              backdropFilter: 'blur(24px)',
              boxShadow: '0 0 50px rgba(0,245,255,0.06)',
            }}
          >
            {/* Triple-ring spinner */}
            <div className="relative w-24 h-24 mb-8">
              {/* Outer ring — cyan */}
              <div className="absolute inset-0 rounded-full border-2 border-transparent"
                style={{ borderTopColor: '#00f5ff', animation: 'spin 1.2s linear infinite' }} />
              {/* Middle ring — violet, reverse */}
              <div className="absolute inset-2 rounded-full border-2 border-transparent"
                style={{ borderTopColor: '#a78bfa', animation: 'spin 1.8s linear infinite reverse' }} />
              {/* Inner gradient circle */}
              <div
                className="absolute inset-5 rounded-full flex items-center justify-center animate-pulse"
                style={{ background: 'linear-gradient(135deg, rgba(0,245,255,0.3), rgba(99,102,241,0.3))' }}
              >
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                    d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
            </div>

            <h3
              className="text-xl font-semibold text-white mb-2"
              style={{ fontFamily: 'var(--font-display)' }}
            >
              {phase === 'uploading' ? 'Uploading Knowledge' : 'Synthesizing Content'}
            </h3>
            <p className="text-gray-600 text-center font-light mb-8 text-sm">
              {phase === 'uploading'
                ? `Moving ${filename} to our secure AI workspace…`
                : `Applying neural parsing to ${filename}. This usually takes 30s.`}
            </p>

            {/* Stage indicators */}
            {phase === 'processing' && (
              <div className="flex items-center gap-2 mb-7">
                {['Parse', 'Chunk', 'Embed', 'Index'].map((s, i) => (
                  <div key={s} className="flex items-center gap-2">
                    <div className="flex flex-col items-center gap-1">
                      <div
                        className="w-2 h-2 rounded-full"
                        style={{
                          background: 'rgba(0,245,255,0.7)',
                          boxShadow: '0 0 6px rgba(0,245,255,0.6)',
                          animation: `pulse ${1 + i * 0.3}s ease-in-out infinite`,
                        }}
                      />
                      <span className="text-[8px] text-gray-600 uppercase tracking-wider"
                        style={{ fontFamily: 'var(--font-mono)' }}>{s}</span>
                    </div>
                    {i < 3 && (
                      <div className="w-6 h-px mb-3" style={{ background: 'rgba(0,245,255,0.2)' }} />
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Progress bar */}
            <div className="w-full max-w-xs h-0.5 rounded-full overflow-hidden"
              style={{ background: 'rgba(255,255,255,0.05)' }}>
              <div
                className={`h-full rounded-full transition-all duration-500 ${phase === 'processing' ? 'animate-pulse' : ''}`}
                style={{
                  width: phase === 'uploading' ? `${uploadProgress}%` : '100%',
                  background: 'linear-gradient(90deg, #00f5ff, #a78bfa)',
                  boxShadow: '0 0 8px rgba(0,245,255,0.4)',
                }}
              />
            </div>
          </div>
        )}

        {/* ── ERROR ── */}
        {phase === 'error' && (
          <div
            className="mt-8 p-5 rounded-2xl flex items-center gap-4"
            style={{
              background: 'rgba(248,113,113,0.06)',
              border: '1px solid rgba(248,113,113,0.2)',
            }}
          >
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ background: 'rgba(248,113,113,0.12)' }}
            >
              <span
                className="text-red-400 text-base font-bold"
                style={{ filter: 'drop-shadow(0 0 4px rgba(248,113,113,0.7))' }}
              >!</span>
            </div>
            <div className="flex-1">
              <p className="text-red-200/80 text-sm font-medium">{error}</p>
              <button
                onClick={() => setPhase('idle')}
                className="text-red-400/70 hover:text-red-300 text-xs transition-colors mt-1"
              >
                Try again
              </button>
            </div>
          </div>
        )}
      </div>

      {/* ── DISCOVER LINK ── */}
      {(phase === 'idle' || phase === 'error') && onDiscover && (
        <div className="relative z-10 mt-5 flex items-center gap-3">
          <div className="flex-1 h-px" style={{ background: 'rgba(255,255,255,0.04)' }} />
          <button
            onClick={onDiscover}
            className="flex items-center gap-2 px-4 py-2 rounded-full text-xs transition-all duration-200 hover:text-violet-300"
            style={{
              background: 'rgba(167,139,250,0.05)',
              border: '1px solid rgba(167,139,250,0.12)',
              color: '#8b5cf6',
            }}
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            Discover papers from arXiv &amp; Semantic Scholar
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
            </svg>
          </button>
          <div className="flex-1 h-px" style={{ background: 'rgba(255,255,255,0.04)' }} />
        </div>
      )}

      {/* ── FOOTER ── */}
      <div className="fixed bottom-0 left-0 right-0 text-center py-4 pointer-events-none z-10">
        <p
          className="text-gray-700 text-[10px] uppercase tracking-[0.2em]"
          style={{ fontFamily: 'var(--font-mono)' }}
        >
          PaperMind · Context-Grounded Research Intelligence
        </p>
      </div>

      {/* Keyframe for triple-ring spin (injected inline since Tailwind's animate-spin goes one direction) */}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to   { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
