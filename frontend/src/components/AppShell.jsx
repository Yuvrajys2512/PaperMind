import Wordmark from './Wordmark'
import NavButton from './NavButton'

const ACCENT = '#22d3ee'

/* Hand-rolled, Heroicons-outline-style icons — matches the stroke
   conventions already used everywhere else in this app (no icon package). */
const ICONS = {
  home:   <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" d="M3 12l9-9 9 9M5 10v10a1 1 0 001 1h4a1 1 0 001-1v-4a1 1 0 011-1h0a1 1 0 011 1v4a1 1 0 001 1h4a1 1 0 001-1V10" /></svg>,
  search: <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" d="M21 21l-4.35-4.35M18 10.5a7.5 7.5 0 11-15 0 7.5 7.5 0 0115 0z" /></svg>,
  stack:  <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" d="M4 7l8-4 8 4-8 4-8-4zm0 5l8 4 8-4M4 17l8 4 8-4" /></svg>,
  pen:    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" d="M11 4H6a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2v-5m-1.5-9.5a2.121 2.121 0 013 3L12 16l-4 1 1-4 9.5-9.5z" /></svg>,
  swap:   <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" d="M8 7h12m0 0l-4-4m4 4l-4 4M16 17H4m0 0l4 4m-4-4l4-4" /></svg>,
  card:   <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" d="M3 6a2 2 0 012-2h14a2 2 0 012 2v12a2 2 0 01-2 2H5a2 2 0 01-2-2V6zm0 4h18" /></svg>,
  shield: <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" d="M12 3l8 3.5v5c0 4.6-3.2 8.5-8 9.5-4.8-1-8-4.9-8-9.5v-5L12 3z" /></svg>,
  plus:   <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.75" d="M12 5v14m-7-7h14" /></svg>,
}

/* Shared chrome for the signed-in app: a persistent left sidebar on md+
   screens, a slim icon-only top bar below md. Pure CSS breakpoint swap —
   no JS width measurement, no drawer/open-close state. Backdrop stays a
   page-level concern; this component owns nav only. */
export default function AppShell({
  active = 'home', onDiscover, onLibrary, onDrafts, onRewrite, onBilling, onAdmin, onNewPaper,
  libraryBadge, children,
}) {
  const navItems = [
    { key: 'home',     label: 'Home',       icon: ICONS.home,   onClick: undefined },
    onDiscover && { key: 'discover', label: 'Discover',   icon: ICONS.search, onClick: onDiscover },
    onLibrary  && { key: 'library',  label: 'Library',    icon: ICONS.stack,  onClick: onLibrary, badge: libraryBadge },
    onDrafts   && { key: 'drafts',   label: 'Write mode',  icon: ICONS.pen,    onClick: onDrafts },
    onRewrite  && { key: 'rewrite',  label: 'Rewrite',    icon: ICONS.swap,   onClick: onRewrite },
    onBilling  && { key: 'billing',  label: 'Plan',       icon: ICONS.card,   onClick: onBilling },
    onAdmin    && { key: 'admin',    label: 'Admin',      icon: ICONS.shield, onClick: onAdmin },
  ].filter(Boolean)

  const mobileItems = navItems.filter(i => i.key !== 'home' && i.key !== 'billing' && i.key !== 'admin')

  return (
    <>
      {/* ── DESKTOP / TABLET SIDEBAR ── */}
      <aside
        className="hidden md:flex md:flex-col md:fixed md:inset-y-0 md:left-0 md:w-60 z-30 px-4 py-5"
        style={{ background: 'rgba(10,11,14,0.9)', borderRight: '1px solid rgba(255,255,255,0.07)' }}
      >
        <div className="px-1 mb-8">
          <Wordmark accent={ACCENT} />
        </div>

        <nav className="flex-1 flex flex-col gap-1">
          {navItems.map(item => (
            <NavButton
              key={item.key}
              variant="sidebar"
              icon={item.icon}
              label={item.label}
              badge={item.badge}
              active={active === item.key}
              onClick={item.onClick}
            />
          ))}
        </nav>

        <NavButton variant="sidebar" primary icon={ICONS.plus} label="New paper" onClick={onNewPaper} />
      </aside>

      {/* ── MOBILE TOP BAR ── */}
      <nav
        className="md:hidden sticky top-0 z-20 flex items-center justify-between gap-2 px-4 py-3 pr-16"
        style={{ background: 'rgba(10,11,14,0.9)', borderBottom: '1px solid rgba(255,255,255,0.07)' }}
      >
        <Wordmark accent={ACCENT} compact />
        <div className="flex items-center gap-1.5">
          {mobileItems.map(item => (
            <NavButton
              key={item.key}
              variant="pill"
              iconOnly
              icon={item.icon}
              label={item.label}
              badge={item.badge}
              onClick={item.onClick}
            />
          ))}
          <NavButton variant="pill" iconOnly primary icon={ICONS.plus} label="New paper" onClick={onNewPaper} />
        </div>
      </nav>

      {/* ── CONTENT (offset for the fixed sidebar) ── */}
      <div className="md:pl-60">{children}</div>
    </>
  )
}
