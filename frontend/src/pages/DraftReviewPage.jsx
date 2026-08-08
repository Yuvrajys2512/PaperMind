import { useState } from 'react'
import ReviewPanel from '../components/ReviewPanel'
import AuditPanel from '../components/AuditPanel'
import NoveltyPanel from '../components/NoveltyPanel'
import StructurePanel from '../components/StructurePanel'
import NumbersPanel from '../components/NumbersPanel'
import CitationGapPanel from '../components/CitationGapPanel'
import OverlapPanel from '../components/OverlapPanel'

const ACCENT = '#a78bfa'
const NOVELTY_ACCENT = '#f472b6'
const STRUCTURE_ACCENT = '#fbbf24'
const NUMBERS_ACCENT = '#34d399'
const CITATION_ACCENT = '#fb7185'
const OVERLAP_ACCENT = '#60a5fa'

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
  const [activeTab, setActiveTab] = useState('review') // 'review' | 'audit' | 'novelty' | 'structure' | 'numbers' | 'citations' | 'overlap' | null

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

        {/* Seven tabs — wraps rather than overflowing on narrower viewports. */}
        <div className="flex flex-wrap items-center justify-end gap-2 flex-shrink-0">
          <TabButton active={activeTab === 'review'} onClick={() => setActiveTab('review')} color={ACCENT}>
            Weakness Review
          </TabButton>
          <TabButton active={activeTab === 'audit'} onClick={() => setActiveTab('audit')} color="#00f5ff">
            Claim Audit
          </TabButton>
          <TabButton active={activeTab === 'novelty'} onClick={() => setActiveTab('novelty')} color={NOVELTY_ACCENT}>
            Novelty Scan
          </TabButton>
          <TabButton active={activeTab === 'structure'} onClick={() => setActiveTab('structure')} color={STRUCTURE_ACCENT}>
            Venue Fit
          </TabButton>
          <TabButton active={activeTab === 'numbers'} onClick={() => setActiveTab('numbers')} color={NUMBERS_ACCENT}>
            Numbers Check
          </TabButton>
          <TabButton active={activeTab === 'citations'} onClick={() => setActiveTab('citations')} color={CITATION_ACCENT}>
            Citation Gaps
          </TabButton>
          <TabButton active={activeTab === 'overlap'} onClick={() => setActiveTab('overlap')} color={OVERLAP_ACCENT}>
            Overlap Check
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
            Novelty Scan searches the literature for the closest prior work and how to position against it.
            Venue Fit checks your draft has the sections a target venue expects.
            Numbers Check reconciles the figures in your abstract against your results.
            Citation Gaps finds statements that need a source but have no citation.
            Overlap Check compares your text against the other papers in your library.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <TabButton active={false} onClick={() => setActiveTab('review')} color={ACCENT}>Weakness Review</TabButton>
            <TabButton active={false} onClick={() => setActiveTab('audit')} color="#00f5ff">Claim Audit</TabButton>
            <TabButton active={false} onClick={() => setActiveTab('novelty')} color={NOVELTY_ACCENT}>Novelty Scan</TabButton>
            <TabButton active={false} onClick={() => setActiveTab('structure')} color={STRUCTURE_ACCENT}>Venue Fit</TabButton>
            <TabButton active={false} onClick={() => setActiveTab('numbers')} color={NUMBERS_ACCENT}>Numbers Check</TabButton>
            <TabButton active={false} onClick={() => setActiveTab('citations')} color={CITATION_ACCENT}>Citation Gaps</TabButton>
            <TabButton active={false} onClick={() => setActiveTab('overlap')} color={OVERLAP_ACCENT}>Overlap Check</TabButton>
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
      {activeTab === 'novelty' && (
        <div className="relative z-10 px-8 py-8">
          <NoveltyPanel paperId={draft.paper_id} onClose={() => setActiveTab(null)} embedded />
        </div>
      )}
      {activeTab === 'structure' && (
        <div className="relative z-10 px-8 py-8">
          <StructurePanel paperId={draft.paper_id} onClose={() => setActiveTab(null)} embedded />
        </div>
      )}
      {activeTab === 'numbers' && (
        <div className="relative z-10 px-8 py-8">
          <NumbersPanel paperId={draft.paper_id} onClose={() => setActiveTab(null)} embedded />
        </div>
      )}
      {activeTab === 'citations' && (
        <div className="relative z-10 px-8 py-8">
          <CitationGapPanel paperId={draft.paper_id} onClose={() => setActiveTab(null)} embedded />
        </div>
      )}
      {activeTab === 'overlap' && (
        <div className="relative z-10 px-8 py-8">
          <OverlapPanel paperId={draft.paper_id} onClose={() => setActiveTab(null)} embedded />
        </div>
      )}
    </div>
  )
}
