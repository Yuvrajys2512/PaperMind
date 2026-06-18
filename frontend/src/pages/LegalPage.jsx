import ReactMarkdown from 'react-markdown'
import { TERMS_MD, PRIVACY_MD } from '../legal/content'

const ACCENT = '#22d3ee'

const DOCS = {
  terms: TERMS_MD,
  privacy: PRIVACY_MD,
}

/* Standalone, auth-agnostic legal page (reached via the #/terms and #/privacy
   hash routes in App.jsx so the links work signed-out and are directly
   shareable). */
export default function LegalPage({ doc }) {
  const md = DOCS[doc] || `# Not found\n\nNo such page.`

  const goHome = () => { window.location.hash = '' }

  return (
    <div className="min-h-screen" style={{ background: '#0a0b0e' }}>
      <nav className="flex items-center justify-between px-6 md:px-10 py-5"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
        <button onClick={goHome} className="flex items-center gap-2.5 group">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center"
            style={{ background: `${ACCENT}1a`, border: `1px solid ${ACCENT}33` }}>
            <svg className="w-4 h-4" fill="none" stroke={ACCENT} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
          </div>
          <span className="text-[15px] font-semibold text-white tracking-tight"
            style={{ fontFamily: 'var(--font-display)' }}>PaperMind</span>
        </button>
        <button onClick={goHome}
          className="text-sm text-gray-400 hover:text-white transition-colors">
          ← Back
        </button>
      </nav>

      <article className="legal-doc max-w-3xl mx-auto px-6 md:px-10 py-12 text-gray-300">
        <ReactMarkdown>{md}</ReactMarkdown>
      </article>

      <style>{`
        .legal-doc h1 { font-size: 2rem; font-weight: 600; color: #fff; margin-bottom: 1.25rem; font-family: var(--font-display); }
        .legal-doc h2 { font-size: 1.15rem; font-weight: 600; color: #fff; margin: 2rem 0 0.75rem; font-family: var(--font-display); }
        .legal-doc p { line-height: 1.75; margin-bottom: 1rem; font-size: 0.95rem; }
        .legal-doc strong { color: rgba(255,255,255,0.92); font-weight: 600; }
        .legal-doc ul { list-style: disc; padding-left: 1.4em; margin-bottom: 1rem; }
        .legal-doc li { margin-bottom: 0.4rem; line-height: 1.65; }
        .legal-doc a { color: ${ACCENT}; text-decoration: underline; text-underline-offset: 2px; }
        .legal-doc table { width: 100%; border-collapse: collapse; margin: 1rem 0 1.5rem; font-size: 0.85rem; }
        .legal-doc th, .legal-doc td { text-align: left; padding: 8px 12px; border: 1px solid rgba(255,255,255,0.08); vertical-align: top; }
        .legal-doc th { background: rgba(255,255,255,0.03); color: #fff; font-weight: 600; }
      `}</style>
    </div>
  )
}
