// Vitest setup, loaded before every test file (see `test.setupFiles` in
// vite.config.js).
import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// React Testing Library does not auto-clean under Vitest's globals, so mounted
// trees would leak between tests and duplicate-match queries.
afterEach(cleanup)

// jsdom implements no layout, so these are simply absent. Components that pin a
// conversation to the newest message call scrollIntoView on every render, which
// would otherwise throw before any assertion ran. Stubbing them here rather than
// per-file keeps the gap in one place — they are jsdom limitations, not app
// behaviour worth asserting.
window.HTMLElement.prototype.scrollIntoView = () => {}
window.HTMLElement.prototype.scrollTo = () => {}
