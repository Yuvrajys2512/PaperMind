import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { createPortal } from 'react-dom'
import { useAuth } from '@clerk/clerk-react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/TextLayer.css'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import { escapeHtml } from '../textUtils'

// pdf.js worker — Vite resolves this to a hashed asset URL at build time.
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

/* ── PDF PREVIEW PANEL ───────────────────────────────────────────── */
// Collapse whitespace + lowercase so PDF text-layer fragments can be matched
// against the cited evidence chunk regardless of line-wrapping differences.
function normalizeForMatch(s) {
  return String(s || '').toLowerCase().replace(/\s+/g, ' ').trim()
}

export default function PDFPreviewPanel({ pdfUrl, title, page, highlight, onClose }) {
  const panelRef  = useRef()
  const scrollRef = useRef()
  const { getToken } = useAuth()

  // The Document loader wants a clean URL; the `#page=` hash is only useful
  // for the native "open in new tab" fallback.
  const fileUrl = useMemo(() => (pdfUrl || '').split('#')[0], [pdfUrl])

  const initialPage = useMemo(() => {
    const p = parseInt(page, 10)
    return Number.isFinite(p) && p > 0 ? p : 1
  }, [page])

  const [numPages,   setNumPages]   = useState(null)
  const [pageNumber, setPageNumber] = useState(initialPage)
  const [width,      setWidth]      = useState(0)
  const [loadError,  setLoadError]  = useState(false)
  // /papers/{id}/pdf is auth-gated; pdf.js's internal fetch for <Document file>
  // doesn't go through api.js, so it needs its own Authorization header.
  const [docFile,    setDocFile]    = useState(null)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      const token = await getToken()
      if (!cancelled) {
        setDocFile({ url: fileUrl, httpHeaders: token ? { Authorization: `Bearer ${token}` } : {} })
      }
    })()
    return () => { cancelled = true }
  }, [fileUrl, getToken])

  // Evidence text to highlight, normalized once.
  const needle = useMemo(() => normalizeForMatch(highlight), [highlight])

  // Close on Escape
  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // Track the scroll container's width so the page renders to fit.
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const update = () => setWidth(Math.max(0, el.clientWidth - 32))
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  // Close on backdrop click (not panel click)
  const handleBackdrop = e => {
    if (e.target === e.currentTarget) onClose()
  }

  // Wrap text-layer fragments that belong to the cited chunk in <mark>.
  // The chunk was extracted from this same PDF, so its line fragments are
  // substrings of the evidence text — matching them reproduces the span.
  const textRenderer = useCallback((item) => {
    const str = item?.str || ''
    const safe = escapeHtml(str)
    if (!needle) return safe
    const frag = normalizeForMatch(str)
    if (frag.length >= 5 && needle.includes(frag)) {
      return `<mark class="pm-pdf-mark">${safe}</mark>`
    }
    return safe
  }, [needle])

  // After the text layer renders, scroll the first highlight into view.
  const handleTextLayer = useCallback(() => {
    const root = scrollRef.current
    if (!root) return
    const first = root.querySelector('.pm-pdf-mark')
    if (first) first.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }, [])

  const goPrev = () => setPageNumber(n => Math.max(1, n - 1))
  const goNext = () => setPageNumber(n => Math.min(numPages || n, n + 1))

  const tabUrl = `${fileUrl}#page=${pageNumber}`

  // Plain <a href> navigation can't carry an Authorization header, and the
  // PDF route is auth-gated — fetch it as a blob with the token instead and
  // open that. Falls back to the direct URL if the fetch fails.
  const handleOpenInTab = useCallback(async (e) => {
    e.preventDefault()
    e.stopPropagation()
    try {
      const token = await getToken()
      const res = await fetch(fileUrl, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
      if (!res.ok) throw new Error('Failed to fetch PDF')
      const blobUrl = URL.createObjectURL(await res.blob())
      window.open(`${blobUrl}#page=${pageNumber}`, '_blank', 'noopener,noreferrer')
    } catch {
      window.open(tabUrl, '_blank', 'noopener,noreferrer')
    }
  }, [fileUrl, pageNumber, tabUrl, getToken])

  const navBtnStyle = (disabled) => ({
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    width: 22, height: 22, borderRadius: 5, flexShrink: 0,
    color: disabled ? 'rgba(156,163,175,0.2)' : 'rgba(156,163,175,0.6)',
    border: '1px solid rgba(255,255,255,0.06)',
    background: 'transparent',
    cursor: disabled ? 'default' : 'pointer',
  })

  // Render through a portal to <body> so the fixed overlay is positioned
  // against the viewport, not a transformed/blurred ancestor (the message
  // tree uses backdrop-blur + slide-down transforms, both of which would
  // otherwise become the containing block and trap/clip this modal).
  return createPortal((
    <div
      onClick={handleBackdrop}
      style={{
        position: 'fixed', inset: 0,
        background: 'rgba(0,0,0,0.55)',
        backdropFilter: 'blur(6px)',
        zIndex: 10000,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
    >
      <style>{`
        .pm-pdf-mark {
          background: rgba(0,245,255,0.38);
          color: transparent;
          border-radius: 2px;
          box-shadow: 0 0 0 1px rgba(0,245,255,0.3);
        }
      `}</style>
      <div
        ref={panelRef}
        style={{
          width: 'min(720px, 92vw)',
          height: '85vh',
          background: 'rgba(6,6,22,0.97)',
          border: '1px solid rgba(0,245,255,0.18)',
          borderRadius: 16,
          boxShadow: '0 32px 80px rgba(0,0,0,0.9), 0 0 0 1px rgba(0,245,255,0.06)',
          display: 'flex', flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 10,
          padding: '10px 16px',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
          flexShrink: 0,
        }}>
          <span style={{
            width: 6, height: 6, borderRadius: '50%',
            background: 'rgba(0,245,255,0.8)',
            boxShadow: '0 0 8px rgba(0,245,255,0.7)',
            flexShrink: 0,
          }} />
          <span style={{
            flex: 1, fontSize: 11, fontFamily: 'var(--font-mono)',
            color: 'rgba(0,245,255,0.7)',
            textTransform: 'uppercase', letterSpacing: '0.15em',
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          }}>
            {title}
          </span>
          {/* Page navigation */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginRight: 4 }}>
            <button onClick={goPrev} disabled={pageNumber <= 1} title="Previous page" style={navBtnStyle(pageNumber <= 1)}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="15 18 9 12 15 6" />
              </svg>
            </button>
            <span style={{
              fontSize: 10, fontFamily: 'var(--font-mono)',
              color: 'rgba(156,163,175,0.55)', minWidth: 48, textAlign: 'center',
            }}>
              p.{pageNumber}{numPages ? ` / ${numPages}` : ''}
            </span>
            <button onClick={goNext} disabled={!!numPages && pageNumber >= numPages} title="Next page" style={navBtnStyle(!!numPages && pageNumber >= numPages)}>
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="9 18 15 12 9 6" />
              </svg>
            </button>
          </div>
          {/* Open in tab button */}
          <a
            href={tabUrl} target="_blank" rel="noopener noreferrer"
            onClick={handleOpenInTab}
            title="Open in new tab"
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              width: 26, height: 26, borderRadius: 6, flexShrink: 0,
              color: 'rgba(156,163,175,0.5)',
              border: '1px solid rgba(255,255,255,0.06)',
              background: 'transparent',
              transition: 'color 0.15s, border-color 0.15s',
              textDecoration: 'none',
            }}
            onMouseEnter={e => { e.currentTarget.style.color = 'rgba(0,245,255,0.8)'; e.currentTarget.style.borderColor = 'rgba(0,245,255,0.25)' }}
            onMouseLeave={e => { e.currentTarget.style.color = 'rgba(156,163,175,0.5)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.06)' }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
              <polyline points="15 3 21 3 21 9" />
              <line x1="10" y1="14" x2="21" y2="3" />
            </svg>
          </a>
          {/* Close button */}
          <button
            onClick={onClose}
            title="Close (Esc)"
            aria-label="Close preview"
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              width: 26, height: 26, borderRadius: 6, flexShrink: 0,
              color: 'rgba(229,231,235,0.85)',
              border: '1px solid rgba(255,255,255,0.16)',
              background: 'rgba(255,255,255,0.05)',
              cursor: 'pointer',
              transition: 'color 0.15s, border-color 0.15s, background 0.15s',
            }}
            onMouseEnter={e => { e.currentTarget.style.color = 'rgba(239,68,68,0.95)'; e.currentTarget.style.borderColor = 'rgba(239,68,68,0.45)'; e.currentTarget.style.background = 'rgba(239,68,68,0.12)' }}
            onMouseLeave={e => { e.currentTarget.style.color = 'rgba(229,231,235,0.85)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.16)'; e.currentTarget.style.background = 'rgba(255,255,255,0.05)' }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* PDF render surface */}
        <div
          ref={scrollRef}
          style={{
            flex: 1, overflow: 'auto', padding: 16,
            display: 'flex', justifyContent: 'center', alignItems: 'flex-start',
            background: 'rgba(0,0,0,0.35)',
          }}
        >
          {loadError ? (
            <div style={{
              margin: 'auto', textAlign: 'center', maxWidth: 320,
              color: 'rgba(156,163,175,0.7)', fontSize: 13, lineHeight: 1.6,
            }}>
              Couldn't render the PDF inline.{' '}
              <a href={tabUrl} target="_blank" rel="noopener noreferrer"
                onClick={handleOpenInTab}
                style={{ color: 'rgba(0,245,255,0.8)' }}>
                Open it in a new tab
              </a>.
            </div>
          ) : !docFile ? (
            <div style={{ margin: 'auto', color: 'rgba(156,163,175,0.6)', fontSize: 12, fontFamily: 'var(--font-mono)' }}>Loading PDF…</div>
          ) : (
            <Document
              file={docFile}
              onLoadSuccess={({ numPages }) => setNumPages(numPages)}
              onLoadError={() => setLoadError(true)}
              loading={<div style={{ margin: 'auto', color: 'rgba(156,163,175,0.6)', fontSize: 12, fontFamily: 'var(--font-mono)' }}>Loading PDF…</div>}
              error={<div style={{ margin: 'auto', color: 'rgba(248,113,113,0.7)', fontSize: 12 }}>Failed to load PDF.</div>}
            >
              <Page
                pageNumber={pageNumber}
                width={width || undefined}
                customTextRenderer={textRenderer}
                onRenderTextLayerSuccess={handleTextLayer}
                renderAnnotationLayer={false}
                loading={<div style={{ margin: 'auto', color: 'rgba(156,163,175,0.6)', fontSize: 12, fontFamily: 'var(--font-mono)' }}>Rendering page…</div>}
              />
            </Document>
          )}
        </div>
      </div>
    </div>
  ), document.body)
}
