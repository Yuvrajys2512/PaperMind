import { useState, useEffect } from 'react'
import { SignedIn, SignedOut, UserButton, useUser } from '@clerk/clerk-react'
import { identify, resetAnalytics } from './analytics'
import UploadPage from './pages/UploadPage'
import ChatPage from './pages/ChatPage'
import DiscoverPage from './pages/DiscoverPage'
import LibraryPage from './pages/LibraryPage'
import RewritePage from './pages/RewritePage'
import BillingPage from './pages/BillingPage'
import AdminPage from './pages/AdminPage'
import LandingPage from './pages/LandingPage'
import LegalPage from './pages/LegalPage'

// Clerk user ids allowed to see the Admin nav link, mirroring the backend's
// PAPERMIND_ADMIN_USER_IDS. This only hides the menu entry — the real gate is
// server-side (/admin/usage 403s anyone not on the backend list), so a
// mismatch or a poked-around UI never leaks data.
const ADMIN_USER_IDS = new Set(
  (import.meta.env.VITE_ADMIN_USER_IDS || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)
)

// Tiny hash-based route layer for the standalone legal pages. Hash routing
// works on static hosting with no rewrite config and stays reachable whether
// or not the visitor is signed in (footer links, Stripe, Clerk all need them).
function useHashRoute() {
  const [hash, setHash] = useState(() => window.location.hash)
  useEffect(() => {
    const onChange = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return hash
}

function AppPages() {
  const { user } = useUser()
  const isAdmin = !!user && ADMIN_USER_IDS.has(user.id)

  // Land on the billing page after returning from Stripe Checkout/portal
  // (success_url / cancel_url carry ?billing=…), then strip the param so a
  // refresh doesn't keep forcing it.
  const [page, setPage] = useState(() => {
    const hasBillingReturn = new URLSearchParams(window.location.search).has('billing')
    return hasBillingReturn ? 'billing' : 'upload'
  })
  const [currentPaper, setCurrentPaper] = useState(null)

  useEffect(() => {
    if (new URLSearchParams(window.location.search).has('billing')) {
      window.history.replaceState({}, '', window.location.pathname)
    }
  }, [])

  const handlePaperReady = (paper) => {
    setCurrentPaper(paper)
    setPage('chat')
  }

  if (page === 'chat') {
    return <ChatPage paper={currentPaper} onBack={() => setPage('upload')} onRewrite={() => setPage('rewrite')} />
  }
  if (page === 'discover') {
    return (
      <DiscoverPage
        onPaperReady={handlePaperReady}
        onBack={() => setPage('upload')}
        onLibrary={() => setPage('library')}
      />
    )
  }
  if (page === 'library') {
    return (
      <LibraryPage
        onOpen={handlePaperReady}
        onBack={() => setPage('upload')}
        onDiscover={() => setPage('discover')}
      />
    )
  }
  if (page === 'rewrite') {
    return <RewritePage onBack={() => setPage('upload')} />
  }
  if (page === 'billing') {
    return <BillingPage onBack={() => setPage('upload')} />
  }
  if (page === 'admin') {
    return <AdminPage onBack={() => setPage('upload')} />
  }
  return (
    <UploadPage
      onPaperReady={handlePaperReady}
      onDiscover={() => setPage('discover')}
      onLibrary={() => setPage('library')}
      onRewrite={() => setPage('rewrite')}
      onBilling={() => setPage('billing')}
      onAdmin={isAdmin ? () => setPage('admin') : undefined}
    />
  )
}

// Tie PostHog events to the signed-in user (pseudonymous — Clerk ID only).
function PostHogIdentify() {
  const { user, isLoaded } = useUser()
  useEffect(() => {
    if (isLoaded && user) identify(user.id)
  }, [isLoaded, user])
  return null
}

// Clear the PostHog identity when signed out, so events aren't attributed to a
// previous user on a shared device.
function PostHogReset() {
  useEffect(() => { resetAnalytics() }, [])
  return null
}

export default function App() {
  const hash = useHashRoute()

  // Standalone legal pages — rendered regardless of auth state.
  if (hash === '#/terms') return <LegalPage doc="terms" />
  if (hash === '#/privacy') return <LegalPage doc="privacy" />

  // TEMP preview of the admin panel — REAL values from get_aggregate_usage()
  // against the live DB (no auth/backend needed to render). Mostly zeros because
  // no real query events exist yet, so capacity falls back to the modeled
  // estimate. Remove this route + the import below once viewed.
  if (hash === '#/admin-preview') {
    return <AdminPage onBack={() => { window.location.hash = '' }} previewData={{
      users: { total: 1, by_tier: { free: 1 } },
      events_by_kind: { seed: 4 },
      per_query: { n: 0, avg_llm_calls: 0, p90_llm_calls: 0, avg_tokens: 0, p90_tokens: 0, avg_cost_usd: 0, distinct_users: 0 },
      lifetime: { query_tokens: 0, projected_cost_usd: 0 },
      capacity: { tokens_per_query: 6000, tokens_source: 'modeled', groq_queries_per_day: 33, mistral_queries_per_month: 166667, free_users_supportable: 8333 },
    }} />
  }

  return (
    <>
      <SignedOut>
        <PostHogReset />
        <LandingPage />
      </SignedOut>
      <SignedIn>
        <PostHogIdentify />
        <div className="fixed top-4 right-4 z-50">
          <UserButton />
        </div>
        <AppPages />
      </SignedIn>
    </>
  )
}
