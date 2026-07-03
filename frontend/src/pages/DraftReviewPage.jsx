import { useState } from 'react'
import ReviewPanel from '../components/ReviewPanel'
import AuditPanel from '../components/AuditPanel'

const ACCENT = '#a78bfa'

function TabButton({ active, onClick, color, children }) {
  return (
    <button
      onClick={onClick}
      className="px-4 py-2 rounded-xl text-xs font-semibold transition-all duration-200"
      style={active
        ? { background: `${color}1a`, border: `1px solid ${color}55`, color }
        : { background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', color: '#6b7280' }}
    >
      {children}
    </button>
  )
}

export default function DraftReviewPage({ draft, onBack }) {
  const [activeTab, setActiveTab] = useState('review') // 'review' | 'audit' | null

  const displayName = draft?.filename?.replace(/\.pdf$/i, '') || 'Draft'

  return (
    <div className="min-h-screen cosmic-bg" style={{ position: 'relative' }}>
      <div
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(167,139,250,0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(167,139,250,0.015) 1px, transparent 1px)
          `,
          backgroundSize: '60px 60px',
          zIndex: 0,
        }}
      />

      {/* Header */}
      <div className="relative z-10 flex items-center gap-5 px-8 py-6" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 text-gray-600 hover:text-gray-400 transition-colors text-xs flex-shrink-0"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
          </svg>
          Back
        </button>

        <div className="min-w-0 flex-1">
          <p className="text-white text-sm font-semibold truncate" style={{ fontFamily: 'var(--font-display)' }}>
            {displayName}
          </p>
          <p className="text-[10px] uppercase tracking-[0.16em] text-gray-600 mt-0.5" style={{ fontFamily: 'var(--font-mono)' }}>
            Pre-submission review
          </p>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          <TabButton active={activeTab === 'review'} onClick={() => setActiveTab('review')} color={ACCENT}>
            Weakness Review
          </TabButton>
          <TabButton active={activeTab === 'audit'} onClick={() => setActiveTab('audit')} color="#00f5ff">
            Claim Audit
          </TabButton>
        </div>
      </div>

      {/* Empty state when both tabs are closed */}
      {activeTab === null && (
        <div className="relative z-10 flex flex-col items-center justify-center text-center px-6" style={{ minHeight: '60vh' }}>
          <h3 className="text-white font-semibold text-lg mb-2" style={{ fontFamily: 'var(--font-display)' }}>
            Pick a check to run
          </h3>
          <p className="text-gray-600 text-sm max-w-sm leading-relaxed mb-6">
            Weakness Review grades your draft against methodological norms a reviewer would check.
            Claim Audit checks that your abstract/intro/conclusion claims are backed by your own results.
          </p>
          <div className="flex items-center gap-3">
            <TabButton active={false} onClick={() => setActiveTab('review')} color={ACCENT}>Weakness Review</TabButton>
            <TabButton active={false} onClick={() => setActiveTab('audit')} color="#00f5ff">Claim Audit</TabButton>
          </div>
        </div>
      )}

      {activeTab === 'review' && (
        <div className="relative z-10 px-8 py-8">
          <ReviewPanel paperId={draft.paper_id} onClose={() => setActiveTab(null)} embedded />
        </div>
      )}
      {activeTab === 'audit' && (
        <div className="relative z-10 px-8 py-8">
          <AuditPanel paperId={draft.paper_id} onClose={() => setActiveTab(null)} embedded />
        </div>
      )}
    </div>
  )
}
