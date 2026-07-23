import { useState, useRef, useCallback, useEffect } from 'react'
import { useUser } from '@clerk/clerk-react'
import { uploadPaper, getPaperStatus, listPapers } from '../api'
import { track } from '../analytics'
import AppShell from '../components/AppShell'
import Backdrop from '../components/Backdrop'

/* Single restrained accent for the whole page. */
const ACCENT = '#22d3ee'

const FOCUS_RING =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#22d3ee] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0b0e]'

/* Small hand-rolled icons for the capability grid — same Heroicons-outline
   conventions used across the app, no icon package. */
const CAP_ICONS = {
  discover: <svg className="w-full h-full" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" d="M21 21l-4.35-4.35M18 10.5a7.5 7.5 0 11-15 0 7.5 7.5 0 0115 0z" /></svg>,
  library:  <svg className="w-full h-full" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" d="M4 7l8-4 8 4-8 4-8-4zm0 5l8 4 8-4M4 17l8 4 8-4" /></svg>,
  write:    <svg className="w-full h-full" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" d="M11 4H6a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2v-5m-1.5-9.5a2.121 2.121 0 013 3L12 16l-4 1 1-4 9.5-9.5z" /></svg>,
  rewrite:  <svg className="w-full h-full" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" d="M8 7h12m0 0l-4-4m4 4l-4 4M16 17H4m0 0l4 4m-4-4l4-4" /></svg>,
}

/* Status-color convention shared with LibraryPage's StatusBadge, kept local
   here since the recent-papers list is a lighter-weight view. */
const STATUS_STYLE = {
  ready:      { color: '#6ee7b7', label: 'Ready' },
  processing: { color: '#67e8f9', label: 'Processing…' },
  failed:     { color: '#fca5a5', label: 'Failed' },
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

/* ── RECENT PAPERS — a real, personal front door back into work, replacing
   the static marketing screenshot that used to sit here. ── */
function RecentPaperCard({ paper, onOpen }) {
  const displayName = paper.filename.replace(/\.pdf$/i, '')
  const date = new Date(paper.completed_at || paper.uploaded_at)
    .toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  const status = STATUS_STYLE[paper.status] || STATUS_STYLE.failed
  const openable = paper.status === 'ready'

  const body = (
    <>
      <div className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
        style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
        <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75"
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      </div>
      <div className="flex-1 min-w-0 text-left">
        <p className="text-[13px] font-medium text-white truncate">{displayName}</p>
        <div className="flex items-center gap-1.5 mt-0.5">
          {paper.status === 'processing' && (
            <span className="w-2.5 h-2.5 border border-cyan-400/40 border-t-cyan-400 rounded-full flex-shrink-0"
              style={{ animation: 'spin 0.9s linear infinite' }} />
          )}
          <span className="text-[11px]" style={{ color: status.color }}>{status.label}</span>
          <span className="text-[11px] text-gray-700">· {date}</span>
        </div>
      </div>
    </>
  )

  if (!openable) {
    return (
      <div className="flex items-center gap-3 rounded-xl px-3 py-2.5 opacity-70"
        style={{ background: 'rgba(255,255,255,0.015)', border: '1px solid rgba(255,255,255,0.06)' }}>
        {body}
      </div>
    )
  }

  return (
    <button
      onClick={() => onOpen(paper)}
      className={`flex items-center gap-3 rounded-xl px-3 py-2.5 transition-colors ${FOCUS_RING}`}
      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.08)' }}
      onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.05)' }}
      onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.02)' }}
    >
      {body}
    </button>
  )
}

/* ── CAPABILITY CARD — bento entry point into a real product surface. ── */
function CapabilityCard({ icon, label, desc, onClick, badge }) {
  return (
    <button
      onClick={onClick}
      className={`group text-left rounded-2xl p-6 transition-colors ${FOCUS_RING}`}
      style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.09)' }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.18)' }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.09)' }}
    >
      <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-4 p-2.5"
        style={{ background: `${ACCENT}14`, border: `1px solid ${ACCENT}33`, color: ACCENT }}>
        {icon}
      </div>
      <div className="flex items-center gap-2 mb-1">
        <h3 className="text-[15px] font-semibold text-white" style={{ fontFamily: 'var(--font-display)' }}>{label}</h3>
        {badge != null && badge > 0 && (
          <span className="text-[10px] font-semibold px-1.5 py-px rounded-full" style={{ background: `${ACCENT}22`, color: ACCENT, fontFamily: 'var(--font-mono)' }}>
            {badge}
          </span>
        )}
      </div>
      <p className="text-[13px] text-gray-500 leading-relaxed">{desc}</p>
    </button>
  )
}

/* ─────────────────────────────────────────────────────────────────
   HOME (post-sign-in)
───────────────────────────────────────────────────────────────── */
export default function UploadPage({ onPaperReady, onDiscover, onLibrary, onDrafts, onRewrite, onBilling, onAdmin }) {
  const { user } = useUser()
  const [dragging, setDragging]             = useState(false)
  const [phase, setPhase]                   = useState('idle') // idle | uploading | processing | error
  const [filename, setFilename]             = useState('')
  const [error, setError]                   = useState('')
  const [quotaHit, setQuotaHit]             = useState(false) // last error was a 429 quota cap
  const [uploadProgress, setUploadProgress] = useState(0)
  const [papers, setPapers]                 = useState(null) // null = loading, [] = confirmed empty
  const inputRef = useRef()

  useEffect(() => {
    listPapers().then(setPapers).catch(() => setPapers([]))
  }, [])

  const [stats] = useState(computeSessionStats)

  const paperCount   = papers?.filter(p => p.status === 'ready').length ?? null
  const recentPapers = papers
    ? [...papers]
        .sort((a, b) => new Date(b.completed_at || b.uploaded_at) - new Date(a.completed_at || a.uploaded_at))
        .slice(0, 5)
    : []

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

  const capabilities = [
    onDiscover && { key: 'discover', icon: CAP_ICONS.discover, label: 'Discover', desc: 'Search across the literature and import papers by topic.', onClick: onDiscover },
    onLibrary  && { key: 'library',  icon: CAP_ICONS.library,  label: 'Library',  desc: "Everything you've processed, ready to reopen and query.", onClick: onLibrary, badge: paperCount },
    onDrafts   && { key: 'drafts',   icon: CAP_ICONS.write,    label: 'Write Mode', desc: 'Pre-submission audits on your own draft — claim grounding, numbers, venue fit, and novelty.', onClick: onDrafts },
    onRewrite  && { key: 'rewrite',  icon: CAP_ICONS.rewrite,  label: 'Rewrite',  desc: 'Tighten a passage without losing its meaning or citations.', onClick: onRewrite },
  ].filter(Boolean)

  return (
    <div className="min-h-screen" style={{ background: '#0a0b0e', position: 'relative' }}>
      <Backdrop />

      <AppShell
        active="home"
        onDiscover={onDiscover}
        onLibrary={onLibrary}
        onDrafts={onDrafts}
        onRewrite={onRewrite}
        onBilling={onBilling}
        onAdmin={onAdmin}
        onNewPaper={openPicker}
        libraryBadge={paperCount}
      >
        <div className="min-h-screen flex flex-col">
          {/* ── HERO — single column: greeting + dropzone. No marketing
              copy, no demo screenshot; this is a returning user's home. ── */}
          <section className="relative z-10 px-6 md:px-10 pt-12 md:pt-16 pb-16">
            <div className="w-full max-w-2xl mx-auto">
              <div className="ed-reveal inline-flex items-center gap-2 mb-7 px-3 py-1 rounded-full"
                style={{ animationDelay: '0ms', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: ACCENT, boxShadow: `0 0 6px ${ACCENT}` }} />
                <span className="text-[11px] text-gray-400 tracking-wide" style={{ fontFamily: 'var(--font-mono)' }}>
                  evidence-graded · benchmarked on QASPER
                </span>
              </div>

              <h1 className="ed-reveal text-balance text-4xl md:text-5xl font-semibold mb-3 tracking-[-0.02em] leading-[1.05] text-white"
                style={{ animationDelay: '80ms', fontFamily: 'var(--font-display)' }}>
                {user?.firstName ? `Welcome back, ${user.firstName}.` : 'Welcome back.'}
              </h1>

              <p className="ed-reveal text-gray-400 text-[15px] leading-relaxed mb-8" style={{ animationDelay: '160ms' }}>
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
                          fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
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
                          className={`text-xs font-semibold px-2.5 py-1 rounded-lg transition-opacity flex-shrink-0 ${FOCUS_RING}`}
                          style={{ background: ACCENT, color: '#0a0b0e' }}
                        >
                          Upgrade
                        </button>
                      ) : (
                        <button
                          onClick={() => setPhase('idle')}
                          className={`text-red-400/70 hover:text-red-300 text-xs transition-colors flex-shrink-0 rounded ${FOCUS_RING}`}
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

                  {/* ── empty-library nudge — one quiet line, not a second empty state ── */}
                  {papers?.length === 0 && (
                    <p className="text-[12px] text-gray-600 mt-4">
                      Your library is empty — the paper you upload above will show up here.
                    </p>
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

              {/* ── RECENT PAPERS — real, personal, replaces the old demo screenshot ── */}
              {recentPapers.length > 0 && (
                <div className="mt-10">
                  <div className="flex items-baseline justify-between mb-3">
                    <h2 className="text-[13px] uppercase tracking-[0.18em] text-gray-500" style={{ fontFamily: 'var(--font-mono)' }}>
                      Recent Papers
                    </h2>
                    {onLibrary && (
                      <button onClick={onLibrary} className={`text-[12px] text-gray-500 hover:text-gray-300 transition-colors rounded ${FOCUS_RING}`}>
                        View all in Library →
                      </button>
                    )}
                  </div>
                  <div className="flex flex-col gap-2">
                    {recentPapers.map(paper => (
                      <RecentPaperCard key={paper.paper_id} paper={paper} onOpen={onPaperReady} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>

          {/* ── CAPABILITIES — bento grid, the real surfaces ── */}
          {capabilities.length > 0 && (
            <section className="relative z-10 px-6 md:px-10 pb-24">
              <div className="w-full max-w-4xl mx-auto">
                <h2 className="text-[13px] uppercase tracking-[0.18em] text-gray-500 mb-4" style={{ fontFamily: 'var(--font-mono)' }}>
                  Everything You Can Do
                </h2>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {capabilities.map(c => (
                    <CapabilityCard key={c.key} icon={c.icon} label={c.label} desc={c.desc} onClick={c.onClick} badge={c.badge} />
                  ))}
                </div>
              </div>
            </section>
          )}

          {/* ── FOOTER ── */}
          <div className="relative z-10 mt-auto px-6 md:px-10 py-5 flex items-center justify-between"
            style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
            <span className="text-[11px] text-gray-700" style={{ fontFamily: 'var(--font-mono)' }}>
              PaperMind
            </span>
            <div className="flex items-center gap-4">
              <a href="#/terms" className={`text-[11px] text-gray-600 hover:text-gray-400 transition-colors rounded-sm ${FOCUS_RING}`}>Terms</a>
              <a href="#/privacy" className={`text-[11px] text-gray-600 hover:text-gray-400 transition-colors rounded-sm ${FOCUS_RING}`}>Privacy</a>
              <span className="text-[11px] text-gray-700 hidden sm:inline">
                Cites the exact section · evidence-graded
              </span>
            </div>
          </div>
        </div>
      </AppShell>
    </div>
  )
}
