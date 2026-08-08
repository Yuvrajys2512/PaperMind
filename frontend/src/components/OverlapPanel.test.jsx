/**
 * Tests for OverlapPanel — the write-mode overlap ("plagiarism") tab.
 *
 * The important behaviour is not the layout, it is the *separation*: matches
 * the author already quoted or cited must be shown apart from unattributed
 * ones and must not be counted against them. This panel is the one place in
 * the product that can imply someone plagiarised, so the copy guarding that —
 * "prompts to check, not verdicts", and the note that a clean result is not a
 * clearance — is treated as behaviour and asserted, not decoration.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

vi.mock('../api', () => ({ overlapStream: vi.fn() }))
vi.mock('./PDFPreviewPanel', () => ({
  default: ({ page, highlight }) => (
    <div data-testid="pdf-preview">{`page ${page}: ${highlight}`}</div>
  ),
}))

import { overlapStream } from '../api'
import OverlapPanel from './OverlapPanel'

const match = (over = {}) => ({
  verdict: 'VERBATIM',
  severity: 'high',
  words: 30,
  excerpt: 'a reused passage of text',
  section: '2 Background',
  page: 2,
  source: {
    paper_id: 'src-1',
    title: 'Source Paper',
    section: '2.4 Training',
    page: 3,
    excerpt: 'a reused passage of text',
  },
  note: '',
  evidence: { section: '2 Background', page: 2, section_type: 'text' },
  ...over,
})

const report = (over = {}) => ({
  paper_id: 'p1',
  corpus_size: 2,
  sources_matched: 1,
  words_scanned: 700,
  overlap_words: 30,
  overlap_score: 0.04,
  originality_score: 0.96,
  matches_found: 1,
  attributed_count: 0,
  paraphrase_count: 0,
  matches: [match()],
  excluded_sources: [],
  audit_failed: false,
  reason: '',
  cached: false,
  ...over,
})

/** Resolve the stream after emitting the given progress stages. */
function resolveWith(result, stages = []) {
  overlapStream.mockImplementation(async (_paperId, onEvent) => {
    for (const stage of stages) onEvent({ type: 'progress', data: { stage } })
    return result
  })
}

beforeEach(() => {
  vi.clearAllMocks()
})

describe('loading', () => {
  it('shows the staged checklist while the check runs', async () => {
    overlapStream.mockImplementation(() => new Promise(() => {}))  // never settles
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)

    expect(screen.getByText('Reading draft')).toBeInTheDocument()
    expect(screen.getByText('Indexing library')).toBeInTheDocument()
    expect(screen.getByText('Comparing text')).toBeInTheDocument()
    expect(screen.getByText('Checking rewrites')).toBeInTheDocument()
  })
})

describe('reporting matches', () => {
  it('renders both sides of a match, and the source it came from', async () => {
    resolveWith(report())
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)

    await screen.findByText(/1 unattributed match/i)
    expect(screen.getByText(/In your draft/i)).toBeInTheDocument()
    expect(screen.getByText(/In “Source Paper”/)).toBeInTheDocument()
    expect(screen.getByText(/2\.4 Training · p\.3/)).toBeInTheDocument()
  })

  it('labels the verdict and the run length', async () => {
    resolveWith(report())
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)

    await screen.findByText(/Verbatim/i)
    expect(screen.getByText(/30 words/i)).toBeInTheDocument()
  })

  it('shows the originality ring as a percentage', async () => {
    resolveWith(report({ originality_score: 0.93 }))
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)
    expect(await screen.findByText('93.0%')).toBeInTheDocument()
  })

  it('always carries the "not a clearance" disclaimer', async () => {
    resolveWith(report())
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)
    expect(await screen.findByText(/not a plagiarism clearance/i)).toBeInTheDocument()
  })

  it('frames matches as prompts rather than verdicts', async () => {
    resolveWith(report())
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)
    expect(await screen.findByText(/prompts to check, not verdicts/i)).toBeInTheDocument()
  })
})

describe('attributed matches are separated', () => {
  it('groups quoted/cited reuse under its own heading', async () => {
    resolveWith(report({
      matches_found: 1,
      attributed_count: 1,
      matches: [match(), match({ verdict: 'ATTRIBUTED', severity: 'low' })],
    }))
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)

    await screen.findByText(/Already attributed/i)
    expect(screen.getByText(/Quoted \/ cited/i)).toBeInTheDocument()
  })

  it('does not count attributed matches as findings', async () => {
    resolveWith(report({
      matches_found: 0,
      attributed_count: 2,
      matches: [
        match({ verdict: 'ATTRIBUTED' }),
        match({ verdict: 'ATTRIBUTED' }),
      ],
    }))
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)

    await screen.findByText(/0 unattributed matches/i)
    // The reassuring line must still appear even though matches exist.
    expect(screen.getByText(/No unattributed overlap/i)).toBeInTheDocument()
  })
})

describe('same-document exclusions', () => {
  it('explains a skipped source instead of silently dropping it', async () => {
    resolveWith(report({
      matches_found: 0,
      matches: [],
      excluded_sources: [{
        paper_id: 'dupe',
        title: 'My Paper',
        reason: 'Looks like the same document as this draft — excluded so it doesn\'t swamp the report.',
      }],
    }))
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)

    await screen.findByText(/Skipped/i)
    expect(screen.getByText(/My Paper/)).toBeInTheDocument()
    expect(screen.getByText(/same document as this draft/i)).toBeInTheDocument()
  })
})

describe('empty and failure states', () => {
  it('tells the user when there is nothing to compare against', async () => {
    resolveWith(report({
      corpus_size: 0,
      matches: [],
      matches_found: 0,
      originality_score: null,
      reason: 'There are no other papers in your library to compare against.',
    }))
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)
    expect(await screen.findByText(/no other papers in your library/i)).toBeInTheDocument()
  })

  it('shows the reason when the audit hard-failed', async () => {
    resolveWith(report({ audit_failed: true, reason: 'No readable body text found.' }))
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)
    expect(await screen.findByText('No readable body text found.')).toBeInTheDocument()
  })

  it('offers a retry when the stream rejects', async () => {
    overlapStream.mockRejectedValue(new Error('Overlap check timed out'))
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)

    expect(await screen.findByText('Overlap check timed out')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument()
  })
})

describe('caching', () => {
  it('badges a cached report', async () => {
    resolveWith(report({ cached: true }))
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)
    expect(await screen.findByText(/cached · re-run to refresh/i)).toBeInTheDocument()
  })

  it('re-run forces a recompute', async () => {
    resolveWith(report())
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)
    await screen.findByText(/1 unattributed match/i)

    // Initial load must not force; the re-run button must.
    expect(overlapStream).toHaveBeenCalledWith('p1', expect.any(Function), false)

    await userEvent.click(screen.getByTitle('Re-run check'))
    await waitFor(() =>
      expect(overlapStream).toHaveBeenCalledWith('p1', expect.any(Function), true),
    )
  })
})

describe('evidence', () => {
  it('opens the draft PDF at the matched page', async () => {
    resolveWith(report())
    render(<OverlapPanel paperId="p1" onClose={() => {}} />)

    await userEvent.click(await screen.findByRole('button', { name: /2 Background · p\.2/ }))
    expect(await screen.findByTestId('pdf-preview')).toHaveTextContent(
      'page 2: a reused passage of text',
    )
  })
})
