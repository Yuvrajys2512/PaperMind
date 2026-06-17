// Thin PostHog wrapper. Every export is a no-op unless VITE_POSTHOG_KEY is set,
// so call sites stay clean and local dev / CI run fine without analytics keys.
import posthog from 'posthog-js'

const KEY = import.meta.env.VITE_POSTHOG_KEY
let enabled = false

export function initAnalytics() {
  if (!KEY) return
  posthog.init(KEY, {
    api_host: import.meta.env.VITE_POSTHOG_HOST || 'https://us.i.posthog.com',
  })
  enabled = true
}

export const track = (event, props) => { if (enabled) posthog.capture(event, props) }
export const identify = (id) => { if (enabled) posthog.identify(id) }
export const resetAnalytics = () => { if (enabled) posthog.reset() }
