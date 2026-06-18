import { useState } from 'react'
import { SignInButton, SignUpButton } from '@clerk/clerk-react'
import { track } from '../analytics'

const ACCENT = '#22d3ee'

/* Quiet dot-grid + soft top glow, matching UploadPage's backdrop. */
function Backdrop() {
  return (
    <>
      <div className="fixed inset-0 pointer-events-none" style={{
        backgroundImage: 'radial-gradient(rgba(255,255,255,0.045) 1px, transparent 1px)',
        backgroundSize: '24px 24px',
        maskImage: 'radial-gradient(ellipse 80% 60% at 50% 25%, #000 40%, transparent 100%)',
        WebkitMaskImage: 'radial-gradient(ellipse 80% 60% at 50% 25%, #000 40%, transparent 100%)',
        zIndex: 0,
      }} />
      <div className="fixed pointer-events-none" style={{
        top: '-220px', left: '50%', transform: 'translateX(-50%)',
        width: 900, height: 500,
        background: `radial-gradient(ellipse at center, ${ACCENT}14 0%, transparent 65%)`,
        filter: 'blur(40px)', zIndex: 0,
      }} />
    </>
  )
}

function Logo() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="w-7 h-7 rounded-lg flex items-center justify-center"
        style={{ background: `${ACCENT}1a`, border: `1px solid ${ACCENT}33` }}>
        <svg className="w-4 h-4" fill="none" stroke={ACCENT} viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
            d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      </div>
      <span className="text-[15px] font-semibold text-white tracking-tight"
        style={{ fontFamily: 'var(--font-display)' }}>PaperMind</span>
    </div>
  )
}

function PrimaryCTA({ children }) {
  return (
    <SignUpButton mode="modal">
      <button onClick={() => track('upgrade_clicked', { from: 'landing' })}
        className="text-sm font-semibold px-5 py-2.5 rounded-xl transition-opacity hover:opacity-90"
        style={{ background: ACCENT, color: '#0a0b0e' }}>
        {children}
      </button>
    </SignUpButton>
  )
}

/* The product demo. Tries /demo.gif (drop it in frontend/public/); until that
   asset exists it shows a labelled placeholder rather than a broken image. */
function DemoMedia() {
  const [failed, setFailed] = useState(false)
  const src = `${import.meta.env.BASE_URL}demo.gif`
  return (
    <div className="rounded-2xl overflow-hidden w-full" style={{
      background: 'linear-gradient(160deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012))',
      border: '1px solid rgba(255,255,255,0.07)', backdropFilter: 'blur(20px)',
    }}>
      <div className="flex items-center gap-1.5 px-4 py-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
        <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'rgba(255,255,255,0.12)' }} />
        <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'rgba(255,255,255,0.12)' }} />
        <span className="w-2.5 h-2.5 rounded-full" style={{ background: 'rgba(255,255,255,0.12)' }} />
        <span className="ml-2 text-[10px] text-gray-600" style={{ fontFamily: 'var(--font-mono)' }}>papermind — demo</span>
      </div>
      {failed ? (
        <div className="aspect-[16/10] flex flex-col items-center justify-center gap-2 text-center px-6">
          <span className="text-[11px] uppercase tracking-widest text-gray-600" style={{ fontFamily: 'var(--font-mono)' }}>
            demo
          </span>
          <span className="text-sm text-gray-500">Product walkthrough coming soon.</span>
        </div>
      ) : (
        <img src={src} alt="PaperMind demo" className="w-full block" onError={() => setFailed(true)} />
      )}
    </div>
  )
}

function Feature({ title, body }) {
  return (
    <div className="rounded-2xl p-5" style={{
      background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)',
    }}>
      <div className="w-8 h-8 rounded-lg flex items-center justify-center mb-3"
        style={{ background: `${ACCENT}14`, border: `1px solid ${ACCENT}2a` }}>
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: ACCENT, boxShadow: `0 0 6px ${ACCENT}` }} />
      </div>
      <h3 className="text-[15px] font-semibold text-white mb-1.5">{title}</h3>
      <p className="text-[13px] text-gray-400 leading-relaxed">{body}</p>
    </div>
  )
}

function PlanCard({ name, price, period, features, cta, highlight }) {
  return (
    <div className="rounded-2xl p-6 flex flex-col" style={{
      background: highlight
        ? `linear-gradient(160deg, ${ACCENT}12, rgba(255,255,255,0.012))`
        : 'rgba(255,255,255,0.02)',
      border: `1px solid ${highlight ? `${ACCENT}44` : 'rgba(255,255,255,0.07)'}`,
    }}>
      <div className="text-sm text-gray-400 mb-1">{name}</div>
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
      <Backdrop />

      {/* ── NAV ── */}
      <nav className="relative z-10 flex items-center justify-between px-6 md:px-10 py-5">
        <Logo />
        <div className="flex items-center gap-4">
          <SignInButton mode="modal">
            <button className="text-sm text-gray-400 hover:text-white transition-colors">Sign in</button>
          </SignInButton>
          <PrimaryCTA>Get started</PrimaryCTA>
        </div>
      </nav>

      {/* ── HERO ── */}
      <section className="relative z-10 px-6 md:px-10 pt-10 pb-20">
        <div className="w-full max-w-6xl mx-auto grid md:grid-cols-2 gap-10 lg:gap-16 items-center">
          <div>
            <div className="inline-flex items-center gap-2 mb-6 px-3 py-1 rounded-full"
              style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)' }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: ACCENT, boxShadow: `0 0 6px ${ACCENT}` }} />
              <span className="text-[11px] text-gray-400 tracking-wide" style={{ fontFamily: 'var(--font-mono)' }}>
                benchmarked on QASPER
              </span>
            </div>

            <h1 className="text-4xl md:text-5xl font-semibold mb-5 tracking-tight leading-[1.08] text-white"
              style={{ fontFamily: 'var(--font-display)' }}>
              Ask any paper.<br />
              Get answers you can&nbsp;
              <span style={{ position: 'relative', whiteSpace: 'nowrap' }}>
                verify
                <span style={{ position: 'absolute', left: 0, right: 0, bottom: 2, height: 8, background: `${ACCENT}55`, borderRadius: 2, zIndex: -1 }} />
              </span>.
            </h1>

            <p className="text-gray-400 text-[15px] max-w-md leading-relaxed mb-8">
              Upload a research PDF and ask anything. Every claim links to the exact
              section it came from — evidence-graded, with no invented citations.
            </p>

            <div className="flex items-center gap-4">
              <PrimaryCTA>Start free</PrimaryCTA>
              <SignInButton mode="modal">
                <button className="text-sm text-gray-300 hover:text-white transition-colors">
                  I already have an account
                </button>
              </SignInButton>
            </div>
            <p className="text-[12px] text-gray-600 mt-4">
              Free plan: 3 papers + 20 queries / month. No card required.
            </p>
          </div>

          <div className="hidden md:block">
            <DemoMedia />
          </div>
        </div>
      </section>

      {/* ── FEATURES ── */}
      <section className="relative z-10 px-6 md:px-10 pb-20">
        <div className="w-full max-w-6xl mx-auto grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Feature title="Exact-section citations"
            body="Every answer points to the specific section it was drawn from — open the source and check it yourself." />
          <Feature title="Evidence-graded"
            body="Answers are scored for how well the source actually supports them, so you see confidence, not just text." />
          <Feature title="Multi-hop reasoning"
            body="Questions that need several passages stitched together are planned and answered across the whole paper." />
          <Feature title="Benchmarked on QASPER"
            body="Quality is measured against the QASPER research-QA benchmark — not just vibes." />
        </div>
      </section>

      {/* ── PRICING ── */}
      <section className="relative z-10 px-6 md:px-10 pb-20">
        <div className="w-full max-w-3xl mx-auto">
          <h2 className="text-2xl font-semibold text-white text-center mb-2" style={{ fontFamily: 'var(--font-display)' }}>
            Simple pricing
          </h2>
          <p className="text-gray-500 text-sm text-center mb-8">Start free. Upgrade when you need more.</p>
          <div className="grid sm:grid-cols-2 gap-4">
            <PlanCard name="Free" price="$0"
              features={['3 papers', '20 queries / month', 'Exact-section citations', 'Evidence-graded answers']}
              cta={<SignUpButton mode="modal">
                <button className="text-sm font-semibold px-4 py-2.5 rounded-xl w-full transition-colors"
                  style={{ background: 'rgba(255,255,255,0.06)', color: '#fff', border: '1px solid rgba(255,255,255,0.1)' }}>
                  Start free
                </button>
              </SignUpButton>} />
            <PlanCard name="Pro" price="$9" period="/ month" highlight
              features={['Unlimited papers', 'Unlimited queries', 'Everything in Free', 'Priority processing']}
              cta={<SignUpButton mode="modal">
                <button onClick={() => track('upgrade_clicked', { from: 'landing_pricing' })}
                  className="text-sm font-semibold px-4 py-2.5 rounded-xl w-full transition-opacity hover:opacity-90"
                  style={{ background: ACCENT, color: '#0a0b0e' }}>
                  Get Pro
                </button>
              </SignUpButton>} />
          </div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="relative z-10 mt-auto px-6 md:px-10 py-6 flex flex-col sm:flex-row items-center justify-between gap-3"
        style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
        <span className="text-[11px] text-gray-700" style={{ fontFamily: 'var(--font-mono)' }}>
          PaperMind · cites the exact section
        </span>
        <div className="flex items-center gap-5 text-[12px] text-gray-500">
          <a href="#/terms" className="hover:text-gray-300 transition-colors">Terms</a>
          <a href="#/privacy" className="hover:text-gray-300 transition-colors">Privacy</a>
        </div>
      </footer>
    </div>
  )
}
