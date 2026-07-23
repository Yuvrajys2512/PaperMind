const ACCENT = '#22d3ee'

const FOCUS_RING =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#22d3ee] focus-visible:ring-offset-2 focus-visible:ring-offset-[#0a0b0e]'

/* Shared nav item, used by AppShell in two shapes:
   - "pill": horizontal, icon optional + label + badge — the mobile top bar
     uses it icon-only, so callers MUST pass `label` there for its aria-label.
   - "sidebar": full-width row, icon + label + badge, with a persistent
     tinted state (not just hover) when `active` — "you are here", not a
     hover flourish. */
export default function NavButton({
  children, label, onClick, badge, icon, active = false, primary = false, variant = 'pill', iconOnly = false,
}) {
  const content = children ?? label

  if (variant === 'sidebar') {
    return (
      <button
        onClick={onClick}
        aria-current={active ? 'page' : undefined}
        className={`group w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-[13px] font-medium transition-colors ${FOCUS_RING}`}
        style={
          primary
            ? { background: ACCENT, color: '#0a0b0e' }
            : active
            ? { background: 'rgba(34,211,238,0.08)', color: '#fff', borderLeft: `2px solid ${ACCENT}` }
            : { background: 'transparent', color: 'rgba(229,231,235,0.85)', borderLeft: '2px solid transparent' }
        }
      >
        {icon && <span className="w-4 h-4 flex-shrink-0" aria-hidden="true">{icon}</span>}
        <span className="flex-1 min-w-0 truncate text-left">{content}</span>
        {badge != null && badge > 0 && (
          <span
            className="text-[10px] font-semibold px-1.5 py-px rounded-full flex-shrink-0"
            style={
              primary
                ? { background: 'rgba(10,11,14,0.2)', color: '#0a0b0e', fontFamily: 'var(--font-mono)' }
                : { background: `${ACCENT}22`, color: ACCENT, fontFamily: 'var(--font-mono)' }
            }
          >
            {badge}
          </span>
        )}
      </button>
    )
  }

  if (primary) {
    return (
      <button
        onClick={onClick}
        aria-label={iconOnly ? label : undefined}
        className={`inline-flex items-center justify-center gap-1.5 rounded-lg text-[13px] font-semibold transition-opacity hover:opacity-90 ${iconOnly ? 'w-10 h-10' : 'px-4 py-1.5'} ${FOCUS_RING}`}
        style={{ background: ACCENT, color: '#0a0b0e' }}
      >
        {icon && <span className="w-4 h-4 flex-shrink-0" aria-hidden="true">{icon}</span>}
        {!iconOnly && content}
      </button>
    )
  }

  return (
    <button
      onClick={onClick}
      aria-label={iconOnly ? label : undefined}
      className={`relative inline-flex items-center justify-center gap-1.5 rounded-lg text-[13px] font-medium text-gray-200 hover:text-white transition-colors ${iconOnly ? 'w-10 h-10' : 'px-3 py-1.5'} ${FOCUS_RING}`}
      style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)' }}
      onMouseEnter={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.08)' }}
      onMouseLeave={e => { e.currentTarget.style.background = 'rgba(255,255,255,0.03)' }}
    >
      {icon && <span className="w-4 h-4 flex-shrink-0" aria-hidden="true">{icon}</span>}
      {!iconOnly && content}
      {badge != null && badge > 0 && (
        <span
          className={iconOnly
            ? 'absolute -top-1 -right-1 text-[9px] font-semibold w-4 h-4 flex items-center justify-center rounded-full'
            : 'text-[10px] font-semibold px-1.5 py-px rounded-full'}
          style={{ background: `${ACCENT}22`, color: ACCENT, fontFamily: 'var(--font-mono)' }}
        >
          {badge}
        </span>
      )}
    </button>
  )
}
