import { useState, useEffect } from 'react'

import { METRIC_TOOLTIPS } from './metricTooltips'

/* ── METRIC RING ─────────────────────────────────────────────────── */

export function MetricRing({ label, value, isPercentage = false, accent = 'cyan', tooltip = null }) {
  const [filled, setFilled] = useState(false)
  const [hovered, setHovered] = useState(false)
  useEffect(() => { const t = setTimeout(() => setFilled(true), 120); return () => clearTimeout(t) }, [])

  const radius = 20
  const circumference = 2 * Math.PI * radius
  const pct = isPercentage ? Math.min(value, 100) : Math.min(value * 100, 100)
  const offset = circumference - (filled ? (pct / 100) * circumference : 0)
  const strokeColor = accent === 'cyan' ? '#00f5ff' : accent === 'violet' ? '#a78bfa' : accent === 'amber' ? '#fbbf24' : accent === 'emerald' ? '#34d399' : accent === 'rose' ? '#fb7185' : accent === 'blue' ? '#60a5fa' : '#60a5fa'
  const glowColor   = accent === 'cyan' ? 'rgba(0,245,255,0.6)' : accent === 'violet' ? 'rgba(167,139,250,0.6)' : accent === 'amber' ? 'rgba(251,191,36,0.6)' : accent === 'emerald' ? 'rgba(52,211,153,0.6)' : accent === 'rose' ? 'rgba(251,113,133,0.6)' : 'rgba(96,165,250,0.6)'
  const displayValue = isPercentage ? value.toFixed(1) + '%' : value.toFixed(2)
  const tip = tooltip || METRIC_TOOLTIPS[String(label).toLowerCase()]

  return (
    <div
      className="relative flex flex-col items-center gap-1.5"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div className="relative cursor-help">
        <svg width="52" height="52" viewBox="0 0 52 52">
          <circle cx="26" cy="26" r={radius} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="3" />
          <circle cx="26" cy="26" r={radius} fill="none" stroke={strokeColor} strokeWidth="3"
            strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={offset}
            transform="rotate(-90 26 26)"
            style={{ transition: 'stroke-dashoffset 1.1s cubic-bezier(0.4,0,0.2,1)', filter: `drop-shadow(0 0 5px ${glowColor})` }} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center"
          style={{ fontFamily: 'var(--font-mono)', fontSize: '8px', color: strokeColor, fontWeight: 600 }}>
          {displayValue}
        </div>
      </div>
      <span className="text-[8px] uppercase tracking-[0.18em] text-gray-600 font-bold cursor-help">{label}</span>

      {tip && hovered && (
        <div
          className="absolute bottom-full mb-2 left-1/2 z-50 w-60 panel-slide-in"
          style={{ pointerEvents: 'none', transform: 'translateX(-26px)' }}
        >
          <div
            className="rounded-xl px-3.5 py-3 text-left"
            style={{
              background: 'rgba(8,8,18,0.96)',
              border: `1px solid ${strokeColor}33`,
              boxShadow: `0 12px 40px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.03)`,
              backdropFilter: 'blur(20px)',
            }}
          >
            <div className="text-[10px] font-bold uppercase tracking-[0.14em] mb-1" style={{ color: strokeColor }}>
              {tip.title}
            </div>
            <div className="text-[11px] leading-[1.5] text-gray-300">{tip.body}</div>
          </div>
          {/* Arrow sits under the ring center (26px in from the tooltip's left edge). */}
          <div
            className="absolute -bottom-1 w-2 h-2 rotate-45"
            style={{ left: '20px', background: 'rgba(8,8,18,0.96)', borderRight: `1px solid ${strokeColor}33`, borderBottom: `1px solid ${strokeColor}33` }}
          />
        </div>
      )}
    </div>
  )
}
