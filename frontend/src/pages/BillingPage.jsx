import { useState, useEffect, useCallback } from 'react'
import { getUsage, startCheckout, openBillingPortal } from '../api'
import { track } from '../analytics'

const ACCENT = '#22d3ee'

/* One metered row — "Papers   2 / 3" with a thin progress bar. */
function MeterRow({ label, used, limit }) {
  const unlimited = limit == null
  const pct = unlimited ? 0 : Math.min(100, Math.round((used / limit) * 100))
  return (
    <div className="mb-5 last:mb-0">
      <div className="flex items-baseline justify-between mb-1.5">
        <span className="text-[13px] text-gray-400">{label}</span>
        <span className="text-[13px] text-white" style={{ fontFamily: 'var(--font-mono)' }}>
          {used}{unlimited ? '' : ` / ${limit}`}
          {unlimited && <span className="text-gray-500"> · unlimited</span>}
        </span>
      </div>
      {!unlimited && (
        <div className="w-full h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
          <div className="h-full rounded-full transition-all duration-500"
            style={{ width: `${pct}%`, background: pct >= 100 ? '#f87171' : ACCENT }} />
        </div>
      )}
    </div>
  )
}

export default function BillingPage({ onBack }) {
  const [usage, setUsage]     = useState(null)
  const [error, setError]     = useState('')
  const [redirecting, setRedirecting] = useState(false)

  useEffect(() => {
    track('plan_viewed')
    getUsage().then(setUsage).catch(e => setError(e.message || 'Failed to load plan'))
  }, [])

  const isPro = usage?.tier === 'pro'

  const handleAction = useCallback(async () => {
    setError('')
    setRedirecting(true)
    if (!isPro) track('upgrade_clicked')
    try {
      const { url } = isPro ? await openBillingPortal() : await startCheckout()
      window.location.href = url // hand off to Stripe-hosted page
    } catch (e) {
      setError(e.message || 'Something went wrong')
      setRedirecting(false)
    }
  }, [isPro])

  return (
    <div className="min-h-screen flex flex-col items-center" style={{ background: '#0a0b0e' }}>
      <div className="w-full max-w-lg px-6 py-10">
        {/* header */}
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-2xl font-semibold text-white tracking-tight"
            style={{ fontFamily: 'var(--font-display)' }}>
            Plan &amp; Billing
          </h1>
          <button onClick={onBack}
            className="text-sm text-gray-400 hover:text-white transition-colors">
            ← Back
          </button>
        </div>

        {error && (
          <div className="mb-6 px-4 py-3 rounded-xl text-[13px] text-red-200/80"
            style={{ background: 'rgba(248,113,113,0.07)', border: '1px solid rgba(248,113,113,0.22)' }}>
            {error}
          </div>
        )}

        {!usage && !error && (
          <div className="text-sm text-gray-500">Loading…</div>
        )}

        {usage && (
          <div className="rounded-2xl p-6"
            style={{
              background: 'linear-gradient(160deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012))',
              border: '1px solid rgba(255,255,255,0.07)',
            }}>
            {/* current tier */}
            <div className="flex items-center gap-2.5 mb-6">
              <span className="text-[11px] uppercase tracking-wider text-gray-500"
                style={{ fontFamily: 'var(--font-mono)' }}>Current plan</span>
              <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full"
                style={{ background: `${ACCENT}1a`, color: ACCENT, fontFamily: 'var(--font-mono)' }}>
                {isPro ? 'PRO' : 'FREE'}
              </span>
            </div>

            {/* usage meters */}
            <MeterRow label="Papers"           used={usage.papers_used}  limit={usage.papers_limit} />
            <MeterRow label="Queries (month)"  used={usage.queries_used} limit={usage.queries_limit} />
            <MeterRow label="Analyses (month)" used={usage.audits_used}  limit={usage.audits_limit} />

            {/* call to action */}
            <div className="mt-7 pt-6" style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
              {isPro ? (
                <>
                  <p className="text-[13px] text-gray-400 mb-4">
                    You're on <span className="text-white">Pro</span> — generous monthly limits, well above free.
                  </p>
                  <button onClick={handleAction} disabled={redirecting}
                    className="w-full py-2.5 rounded-xl text-sm font-medium text-white transition-colors disabled:opacity-50"
                    style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)' }}>
                    {redirecting ? 'Opening…' : 'Manage billing'}
                  </button>
                </>
              ) : (
                <>
                  <div className="flex items-baseline gap-1.5 mb-1">
                    <span className="text-3xl font-semibold text-white"
                      style={{ fontFamily: 'var(--font-display)' }}>$9</span>
                    <span className="text-sm text-gray-500">/ month</span>
                  </div>
                  <p className="text-[13px] text-gray-400 mb-4">
                    100 papers, 2,000 queries, 300 analyses per month. Cancel anytime.
                  </p>
                  <button onClick={handleAction} disabled={redirecting}
                    className="w-full py-2.5 rounded-xl text-sm font-semibold transition-opacity disabled:opacity-50"
                    style={{ background: ACCENT, color: '#0a0b0e' }}>
                    {redirecting ? 'Redirecting…' : 'Upgrade to Pro'}
                  </button>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
