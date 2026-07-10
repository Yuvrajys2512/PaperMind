const ACCENT = '#22d3ee'

/* Shared nav item. The old landing nav used bare `text-gray-400` links with no
   weight, background, or border — invisible on black. This gives every item a
   persistent faint surface + hairline so it reads unmistakably as a control,
   brightening on hover. `primary` renders the one accented call-to-action. */
export default function NavButton({ children, onClick, badge, icon, primary = false }) {
  if (primary) {
    return (
      <button
        onClick={onClick}
        className="inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-[13px] font-semibold transition-opacity hover:opacity-90"
        style={{ background: ACCENT, color: '#0a0b0e' }}
      >
        {icon}
        {children}
      </button>
    )
  }
  return (
    <button
      onClick={onClick}
      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[13px] font-medium text-gray-200 hover:text-white transition-colors"
      style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}
      onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.08)' }}
      onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.03)' }}
    >
      {icon}
      {children}
      {badge != null && badge > 0 && (
        <span
          className="text-[10px] font-semibold px-1.5 py-px rounded-full"
          style={{ background: `${ACCENT}22`, color: ACCENT, fontFamily: 'var(--font-mono)' }}
        >
          {badge}
        </span>
      )}
    </button>
  )
}
