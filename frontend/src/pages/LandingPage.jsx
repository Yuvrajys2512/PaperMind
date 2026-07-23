import { SignInButton, SignUpButton } from '@clerk/clerk-react'
import { track } from '../analytics'
import Wordmark from '../components/Wordmark'
import CitedAnswer from '../components/CitedAnswer'
import Backdrop from '../components/Backdrop'

const ACCENT = '#22d3ee'

/* One accented call-to-action, wrapping Clerk's sign-up modal. */
function PrimaryCTA({ children, from }) {
  return (
    <SignUpButton mode="modal">
      <button onClick={() => track('upgrade_clicked', { from })}
        className="text-sm font-semibold px-5 py-2.5 rounded-xl transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#22d3ee] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0b0e]"
        style={{ background: ACCENT, color: '#0a0b0e' }}>
        {children}
      </button>
    </SignUpButton>
  )
}

/* Editorial capability row — a typographic line item, not a glossy card. */
function Capability({ index, title, body, last }) {
  return (
    <div
      className="flex items-start gap-4 py-5"
      style={{ borderBottom: last ? 'none' : '1px solid rgba(255,255,255,0.07)' }}
    >
      <span className="text-[11px] text-gray-600 w-6 flex-shrink-0 mt-1" style={{ fontFamily: 'var(--font-mono)' }}>
        {index}
      </span>
      <div className="min-w-0">
        <h3 className="text-[15px] font-semibold text-white mb-1" style={{ fontFamily: 'var(--font-display)' }}>{title}</h3>
        <p className="text-[13px] text-gray-400 leading-relaxed">{body}</p>
      </div>
    </div>
  )
}

/* De-glassed pricing plan — solid surface + hairline; accent only marks Pro. */
function PlanCard({ name, price, period, features, cta, highlight }) {
  return (
    <div className="rounded-2xl p-6 flex flex-col" style={{
      background: 'rgba(255,255,255,0.02)',
      border: `1px solid ${highlight ? `${ACCENT}55` : 'rgba(255,255,255,0.09)'}`,
    }}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm text-gray-400">{name}</span>
        {highlight && (
          <span className="text-[9px] uppercase tracking-[0.16em] font-bold px-2 py-0.5 rounded-full"
            style={{ background: `${ACCENT}1a`, color: ACCENT, fontFamily: 'var(--font-mono)' }}>
            Recommended
          </span>
        )}
      </div>
      <div className="flex items-baseline gap-1 mb-5">
        <span className="text-3xl font-semibold text-white" style={{ fontFamily: 'var(--font-display)' }}>{price}</span>
        {period && <span className="text-sm text-gray-500">{period}</span>}
      </div>
      <ul className="flex-1 space-y-2.5 mb-6">
        {features.map((f, i) => (
          <li key={i} className="flex items-start gap-2 text-[13px] text-gray-300">
            <span className="mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: ACCENT }} />
            {f}
          </li>
        ))}
      </ul>
      {cta}
    </div>
  )
}

export default function LandingPage() {
  return (
    <div className="min-h-screen flex flex-col" style={{ background: '#0a0b0e', position: 'relative' }}>
      <Backdrop maskY={22} />

      {/* ── NAV ── */}
      <nav
        className="sticky top-0 z-20 flex items-center justify-between px-6 md:px-10 py-4"
        style={{ background: 'rgba(10,11,14,0.85)', borderBottom: '1px solid rgba(255,255,255,0.07)' }}
      >
        <Wordmark accent={ACCENT} />
        <div className="flex items-center gap-4">
          <SignInButton mode="modal">
            <button className="text-sm font-medium text-gray-200 hover:text-white transition-colors rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#22d3ee] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0b0e]">Sign in</button>
          </SignInButton>
          <PrimaryCTA from="landing_nav">Get started</PrimaryCTA>
        </div>
      </nav>

      {/* ── HERO ── */}
      <section className="relative z-10 px-6 md:px-10 pt-16 md:pt-24 pb-20">
        <div className="w-full max-w-6xl mx-auto grid md:grid-cols-2 gap-12 lg:gap-16 items-center">
          <div>
            <div className="ed-reveal inline-flex items-center gap-2 mb-7 px-3 py-1 rounded-full"
              style={{ animationDelay: '0ms', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: ACCENT, boxShadow: `0 0 6px ${ACCENT}` }} />
              <span className="text-[11px] text-gray-400 tracking-wide" style={{ fontFamily: 'var(--font-mono)' }}>
                evidence-graded · benchmarked on QASPER
              </span>
            </div>

            <h1 className="ed-reveal text-balance text-5xl md:text-6xl font-semibold mb-6 tracking-[-0.02em] leading-[1.03] text-white"
              style={{ animationDelay: '80ms', fontFamily: 'var(--font-display)' }}>
              Read papers<br />
              you can{' '}
              <span style={{ position: 'relative', whiteSpace: 'nowrap' }}>
                verify
                <span className="ed-underline"
                  style={{ position: 'absolute', left: 0, right: 0, bottom: 4, height: 10, background: `${ACCENT}55`, borderRadius: 2, zIndex: -1 }} />
              </span>
              <span className="ed-caret" style={{ color: ACCENT }}>.</span>
            </h1>

            <p className="ed-reveal text-gray-400 text-[15px] max-w-md leading-relaxed mb-8" style={{ animationDelay: '160ms' }}>
              Upload a research PDF and ask anything. Every claim links to the exact
              line it came from — evidence-graded, with no invented citations.
            </p>

            <div className="ed-reveal flex items-center gap-5" style={{ animationDelay: '240ms' }}>
              <PrimaryCTA from="landing_hero">Start free</PrimaryCTA>
              <SignInButton mode="modal">
                <button className="text-sm text-gray-300 hover:text-white transition-colors inline-flex items-center gap-1.5 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#22d3ee] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0b0e]">
                  I already have an account
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </SignInButton>
            </div>
            <p className="ed-reveal text-[12px] text-gray-600 mt-4" style={{ animationDelay: '300ms' }}>
              Free plan: 3 papers + 20 queries / month. No card required.
            </p>
          </div>

          <div className="hidden md:block ed-reveal" style={{ animationDelay: '200ms' }}>
            <CitedAnswer />
          </div>
        </div>
      </section>

      {/* ── CAPABILITIES ── */}
      <section className="relative z-10 px-6 md:px-10 pb-20">
        <div className="w-full max-w-4xl mx-auto">
          <h2 className="text-[13px] uppercase tracking-[0.18em] text-gray-500 mb-2" style={{ fontFamily: 'var(--font-mono)' }}>
            What It Does
          </h2>
          <div className="grid sm:grid-cols-2 gap-x-12" style={{ borderTop: '1px solid rgba(255,255,255,0.09)' }}>
            <div>
              <Capability index="01" title="Exact-section citations"
                body="Every answer points to the specific line it was drawn from — open the source and check it yourself." />
              <Capability index="02" title="Evidence-graded"
                body="Answers are scored for how well the source actually supports them, so you see confidence, not just text." />
              <Capability index="03" title="Multi-hop reasoning" last
                body="Questions that need several passages stitched together are planned and answered across the whole paper." />
            </div>
            <div>
              <Capability index="04" title="Write mode"
                body="Pre-submission audits on your own draft: claim grounding, numbers consistency, venue fit, and novelty." />
              <Capability index="05" title="Discover"
                body="Search across the literature and import the papers you need by topic." />
              <Capability index="06" title="Benchmarked on QASPER" last
                body="Quality is measured against the QASPER research-QA benchmark — not just vibes." />
            </div>
          </div>
        </div>
      </section>

      {/* ── PRICING ── */}
      <section className="relative z-10 px-6 md:px-10 pb-20">
        <div className="w-full max-w-3xl mx-auto">
          <h2 className="text-2xl font-semibold text-white text-center mb-2 text-balance" style={{ fontFamily: 'var(--font-display)' }}>
            Simple Pricing
          </h2>
          <p className="text-gray-500 text-sm text-center mb-8">Start free. Upgrade when you need more.</p>
          <div className="grid sm:grid-cols-2 gap-4">
            <PlanCard name="Free" price="$0"
              features={['3 papers', '20 queries / month', 'Exact-section citations', 'Evidence-graded answers']}
              cta={<SignUpButton mode="modal">
                <button className="text-sm font-semibold px-4 py-2.5 rounded-xl w-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#22d3ee] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0b0e]"
                  style={{ background: 'rgba(255,255,255,0.06)', color: '#fff', border: '1px solid rgba(255,255,255,0.1)' }}>
                  Start free
                </button>
              </SignUpButton>} />
            <PlanCard name="Pro" price="$9" period="/ month" highlight
              features={['Unlimited papers', 'Unlimited queries', 'Write-mode audits', 'Priority processing']}
              cta={<SignUpButton mode="modal">
                <button onClick={() => track('upgrade_clicked', { from: 'landing_pricing' })}
                  className="text-sm font-semibold px-4 py-2.5 rounded-xl w-full transition-opacity hover:opacity-90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#22d3ee] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0b0e]"
                  style={{ background: ACCENT, color: '#0a0b0e' }}>
                  Get Pro
                </button>
              </SignUpButton>} />
          </div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="relative z-10 mt-auto px-6 md:px-10 py-6 flex flex-col sm:flex-row items-center justify-between gap-3"
        style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
        <span className="text-[11px] text-gray-700" style={{ fontFamily: 'var(--font-mono)' }}>
          PaperMind · cites the exact section
        </span>
        <div className="flex items-center gap-5 text-[12px] text-gray-500">
          <a href="#/terms" className="hover:text-gray-300 transition-colors rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#22d3ee] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0b0e]">Terms</a>
          <a href="#/privacy" className="hover:text-gray-300 transition-colors rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#22d3ee] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0b0e]">Privacy</a>
        </div>
      </footer>
    </div>
  )
}
