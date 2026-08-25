/**
 * Tests for AppShell's free-plan quota strip.
 *
 * This exists because the sidebar used to contradict the server: the Library
 * badge counted the user's papers PLUS the shared demo samples, while the quota
 * counts the user's papers PLUS their drafts and ignores samples. A user
 * reading "Library 5" who was then refused at "limit of 3" had no way to
 * reconcile the two.
 *
 * The strip's whole job is showing the number the server actually enforces, so
 * the cases below pin: a genuinely uncapped tier (null papers_limit) shows
 * nothing, pro shows its own real limit under a "Pro plan" label (Launch
 * Checklist 2.10 gave pro a real ceiling instead of unlimited), the drafts
 * breakdown is correct and correctly pluralised, and the at-limit state is
 * visually distinct.
 */
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

import AppShell from './AppShell'

const shell = (usage) =>
  render(<AppShell usage={usage} libraryBadge={2} onLibrary={() => {}} onDrafts={() => {}} />)

describe('when the tier is genuinely uncapped', () => {
  it('renders nothing when papers_limit is null (e.g. an unrecognized future tier)', () => {
    shell({ tier: 'enterprise', papers_used: 40, papers_limit: null, drafts_used: 3 })
    expect(screen.queryByText(/plan/i)).not.toBeInTheDocument()
  })

  it('renders nothing when usage has not loaded yet', () => {
    shell(null)
    expect(screen.queryByText(/plan/i)).not.toBeInTheDocument()
  })
})

describe('pro plan counts', () => {
  it('shows the "Pro plan" label with its own real limit, not "unlimited"', () => {
    shell({ tier: 'pro', papers_used: 40, papers_limit: 100, drafts_used: 3 })
    expect(screen.getByText(/pro plan/i)).toBeInTheDocument()
    expect(screen.getByText('40/100')).toBeInTheDocument()
  })
})

describe('free plan counts', () => {
  it('shows used/limit using the number the server enforces', () => {
    // 2 papers + 1 draft = 3, even though Library only displays 2.
    shell({ tier: 'free', papers_used: 3, papers_limit: 3, drafts_used: 1 })
    expect(screen.getByText('3/3')).toBeInTheDocument()
  })

  it('breaks out drafts, because Library never shows them', () => {
    shell({ tier: 'free', papers_used: 3, papers_limit: 3, drafts_used: 1 })
    expect(screen.getByText(/2 papers \+ 1 draft/i)).toBeInTheDocument()
  })

  it('says samples do not count, since they inflate the Library badge', () => {
    shell({ tier: 'free', papers_used: 3, papers_limit: 3, drafts_used: 1 })
    expect(screen.getByText(/samples don't count/i)).toBeInTheDocument()
  })

  it('pluralises a single paper correctly', () => {
    shell({ tier: 'free', papers_used: 2, papers_limit: 3, drafts_used: 1 })
    expect(screen.getByText(/1 paper \+ 1 draft\./i)).toBeInTheDocument()
  })

  it('pluralises multiple drafts correctly', () => {
    shell({ tier: 'free', papers_used: 3, papers_limit: 3, drafts_used: 2 })
    expect(screen.getByText(/1 paper \+ 2 drafts/i)).toBeInTheDocument()
  })

  it('omits the breakdown when there are no drafts', () => {
    shell({ tier: 'free', papers_used: 1, papers_limit: 3, drafts_used: 0 })
    expect(screen.getByText('1/3')).toBeInTheDocument()
    expect(screen.queryByText(/\+ 0 drafts/i)).not.toBeInTheDocument()
    expect(screen.getByText(/drafts count too/i)).toBeInTheDocument()
  })

  it('treats a missing drafts_used as zero rather than rendering NaN', () => {
    shell({ tier: 'free', papers_used: 1, papers_limit: 3 })
    expect(screen.getByText('1/3')).toBeInTheDocument()
    expect(screen.queryByText(/NaN/)).not.toBeInTheDocument()
  })
})

describe('at-limit state', () => {
  const red = 'rgba(248, 113, 113, 0.9)'

  it('colours the count red once the limit is reached', () => {
    shell({ tier: 'free', papers_used: 3, papers_limit: 3, drafts_used: 1 })
    expect(screen.getByText('3/3')).toHaveStyle({ color: red })
  })

  it('stays neutral below the limit', () => {
    shell({ tier: 'free', papers_used: 1, papers_limit: 3, drafts_used: 0 })
    expect(screen.getByText('1/3')).not.toHaveStyle({ color: red })
  })

  it('still reads as at-limit if the count somehow exceeds it', () => {
    // Pre-existing accounts can sit above a limit introduced later.
    shell({ tier: 'free', papers_used: 5, papers_limit: 3, drafts_used: 1 })
    expect(screen.getByText('5/3')).toHaveStyle({ color: red })
  })
})

describe('library badge', () => {
  it('shows the count it is given', () => {
    shell({ tier: 'free', papers_used: 3, papers_limit: 3, drafts_used: 1 })
    // Rendered twice — desktop sidebar and mobile top bar.
    expect(screen.getAllByText('2').length).toBeGreaterThan(0)
  })
})
