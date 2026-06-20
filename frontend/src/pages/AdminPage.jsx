import { useState, useEffect } from 'react'
import { getAdminUsage } from '../api'

const ACCENT = '#22d3ee'

/* One labelled metric — big number with a caption. */
function Stat({ label, value, sub }) {
  return (
    <div className="rounded-xl px-4 py-3.5"
      style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.06)' }}>
      <div className="text-[11px] uppercase tracking-wider text-gray-500 mb-1.5"
        style={{ fontFamily: 'var(--font-mono)' }}>{label}</div>
      <div className="text-xl text-white" style={{ fontFamily: 'var(--font-mono)' }}>{value}</div>
      {sub && <div className="text-[11px] text-gray-500 mt-0.5">{sub}</div>}
    </div>
  )
}

function Section({ title, children }) {
  return (
    <div className="rounded-2xl p-6 mb-5"
      style={{
        background: 'linear-gradient(160deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012))',
        border: '1px solid rgba(255,255,255,0.07)',
      }}>
      <h2 className="text-[11px] uppercase tracking-wider text-gray-500 mb-4"
        style={{ fontFamily: 'var(--font-mono)' }}>{title}</h2>
      {children}
    </div>
  )
}

const num = (n) => (n == null ? '—' : n.toLocaleString())

export default function AdminPage({ onBack, previewData }) {
  const [data, setData]   = useState(previewData || null)
  const [error, setError] = useState('')

  useEffect(() => {
    if (previewData) return // preview mode: render injected sample data, no fetch
    getAdminUsage()
      .then(setData)
      .catch(e => setError(
        e.status === 403
          ? "You don't have admin access. Ask the owner to add your user id to PAPERMIND_ADMIN_USER_IDS."
          : (e.message || 'Failed to load admin usage')
      ))
  }, [previewData])

  const cap = data?.capacity
  const pq  = data?.per_query

  return (
    <div className="min-h-screen flex flex-col items-center" style={{ background: '#0a0b0e' }}>
      <div className="w-full max-w-2xl px-6 py-10">
        {/* header */}
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-2xl font-semibold text-white tracking-tight"
            style={{ fontFamily: 'var(--font-display)' }}>
            Admin · Usage
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

        {!data && !error && <div className="text-sm text-gray-500">Loading…</div>}

        {data && (
          <>
            <Section title="Users & events">
              <div className="grid grid-cols-3 gap-3">
                <Stat label="Total users" value={num(data.users?.total)}
                  sub={Object.entries(data.users?.by_tier || {}).map(([t, c]) => `${c} ${t}`).join(' · ')} />
                <Stat label="Query events" value={num(data.events_by_kind?.query || 0)} />
                <Stat label="Upload events" value={num(data.events_by_kind?.upload || 0)} />
              </div>
            </Section>

            <Section title="Per query">
              {pq?.n ? (
                <div className="grid grid-cols-3 gap-3">
                  <Stat label="Avg LLM calls" value={pq.avg_llm_calls} sub={`p90 ${pq.p90_llm_calls}`} />
                  <Stat label="Avg tokens"    value={num(pq.avg_tokens)} sub={`p90 ${num(pq.p90_tokens)}`} />
                  <Stat label="Avg cost"      value={`$${pq.avg_cost_usd?.toFixed(5)}`} />
                </div>
              ) : (
                <p className="text-[13px] text-gray-400">
                  No real query events recorded yet — capacity below uses the modeled estimate.
                </p>
              )}
            </Section>

            <Section title="Upstream capacity (free provider quotas)">
              <div className="grid grid-cols-3 gap-3 mb-4">
                <Stat label="Groq / day"        value={num(cap?.groq_queries_per_day)}     sub="fast LPU path" />
                <Stat label="Mistral / month"   value={num(cap?.mistral_queries_per_month)} sub="real backstop" />
                <Stat label="Free users"        value={num(cap?.free_users_supportable)}    sub="@ free cap/mo" />
              </div>
              <div className="flex items-center gap-2 text-[12px] text-gray-500">
                <span style={{ fontFamily: 'var(--font-mono)' }}>
                  {num(cap?.tokens_per_query)} tokens/query
                </span>
                <span className="px-2 py-0.5 rounded-full text-[10px] uppercase tracking-wide"
                  style={{
                    background: cap?.tokens_source === 'measured' ? `${ACCENT}1a` : 'rgba(255,255,255,0.06)',
                    color: cap?.tokens_source === 'measured' ? ACCENT : '#9ca3af',
                    fontFamily: 'var(--font-mono)',
                  }}>
                  {cap?.tokens_source}
                </span>
              </div>
            </Section>

            <Section title="Lifetime">
              <div className="grid grid-cols-2 gap-3">
                <Stat label="Query tokens" value={num(data.lifetime?.query_tokens)} />
                <Stat label="Projected cost" value={`$${(data.lifetime?.projected_cost_usd || 0).toFixed(4)}`}
                  sub="on paid keys" />
              </div>
            </Section>
          </>
        )}
      </div>
    </div>
  )
}
