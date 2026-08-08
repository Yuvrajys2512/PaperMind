/**
 * Tests for DraftsPage — the "My Drafts" list and its upload/delete flows.
 *
 * Scope note: this file sticks to behaviour that is unambiguously intended —
 * things the code comments call out, or that exist to protect the user (the
 * two-step delete confirmation, the quota banner keyed on HTTP status, the
 * PDF-only guard). Styling and layout are left alone deliberately; they are the
 * parts most likely to change for good reasons.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

vi.mock('../api', () => ({
  listPapers: vi.fn(),
  uploadPaper: vi.fn(),
  deletePaper: vi.fn(),
}))
vi.mock('../analytics', () => ({ track: vi.fn() }))

import { listPapers, uploadPaper, deletePaper } from '../api'
import DraftsPage from './DraftsPage'

const draft = (over = {}) => ({
  paper_id: 'd1',
  filename: 'my-draft.pdf',
  status: 'ready',
  uploaded_at: '2026-08-01T10:00:00Z',
  completed_at: '2026-08-01T10:05:00Z',
  ...over,
})

const pdf = (name = 'draft.pdf') =>
  new File(['%PDF-1.4 content'], name, { type: 'application/pdf' })

/** The file input is visually hidden, so it has no accessible label to query. */
const fileInput = (container) => container.querySelector('input[type="file"]')

beforeEach(() => {
  vi.clearAllMocks()
  listPapers.mockResolvedValue([])
})

afterEach(() => {
  vi.useRealTimers()
})

describe('loading the list', () => {
  it('asks for drafts, not reference papers', async () => {
    render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    await waitFor(() => expect(listPapers).toHaveBeenCalledWith('draft'))
  })

  it('summarises drafts by status', async () => {
    listPapers.mockResolvedValue([
      draft({ paper_id: 'a', status: 'ready' }),
      draft({ paper_id: 'b', status: 'ready' }),
      draft({ paper_id: 'c', status: 'processing' }),
      draft({ paper_id: 'd', status: 'failed' }),
    ])
    render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    expect(await screen.findByText('2 ready · 1 processing · 1 failed')).toBeInTheDocument()
  })

  it('strips the .pdf extension from the displayed name', async () => {
    listPapers.mockResolvedValue([draft({ filename: 'Corvid Scheduler.pdf' })])
    render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    expect(await screen.findByText('Corvid Scheduler')).toBeInTheDocument()
  })

  it('reports a failed fetch instead of rendering an empty list as success', async () => {
    listPapers.mockRejectedValue(new Error('network down'))
    render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    expect(await screen.findByText('Could not load your drafts.')).toBeInTheDocument()
  })

  it('surfaces a per-draft ingestion error', async () => {
    listPapers.mockResolvedValue([
      draft({ status: 'failed', error: 'Parsing failed: encrypted PDF' }),
    ])
    render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    expect(await screen.findByText('Parsing failed: encrypted PDF')).toBeInTheDocument()
  })
})

describe('per-status controls', () => {
  it('offers Review only once a draft is ready', async () => {
    listPapers.mockResolvedValue([draft({ status: 'ready' })])
    render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    expect(await screen.findByRole('button', { name: 'Review' })).toBeInTheDocument()
  })

  it('shows progress instead of Review while ingesting', async () => {
    listPapers.mockResolvedValue([draft({ status: 'processing' })])
    render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)

    expect(await screen.findByText('Processing')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Review' })).not.toBeInTheDocument()
    expect(screen.getByText('Ingesting…')).toBeInTheDocument()
  })

  it('hands the whole draft to onOpen so the review page can use its metadata', async () => {
    const d = draft()
    const onOpen = vi.fn()
    listPapers.mockResolvedValue([d])
    render(<DraftsPage onOpen={onOpen} onBack={() => {}} />)

    await userEvent.click(await screen.findByRole('button', { name: 'Review' }))
    expect(onOpen).toHaveBeenCalledWith(d)
  })
})

describe('upload', () => {
  it('rejects a non-PDF before hitting the network', async () => {
    const { container } = render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    await waitFor(() => expect(listPapers).toHaveBeenCalled())

    // applyAccept:false so the browser-level accept=".pdf" filter does not
    // swallow the file first — the guard under test is the component's own.
    await userEvent.upload(
      fileInput(container),
      new File(['x'], 'notes.txt', { type: 'text/plain' }),
      { applyAccept: false },
    )

    expect(await screen.findByText('Please upload a PDF file.')).toBeInTheDocument()
    expect(uploadPaper).not.toHaveBeenCalled()
  })

  it('uploads with paper_type "draft" and refreshes the list', async () => {
    uploadPaper.mockResolvedValue({ paper_id: 'new' })
    const { container } = render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    await waitFor(() => expect(listPapers).toHaveBeenCalledTimes(1))

    await userEvent.upload(fileInput(container), pdf())

    await waitFor(() => expect(uploadPaper).toHaveBeenCalledWith(expect.any(File), 'draft'))
    await waitFor(() => expect(listPapers).toHaveBeenCalledTimes(2))
  })

  it('shows the quota banner for a 429, keyed on status not message text', async () => {
    const err = new Error('Free tier limit of 3 reached — you have 2 papers and 1 draft.')
    err.status = 429
    uploadPaper.mockRejectedValue(err)

    const { container } = render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    await waitFor(() => expect(listPapers).toHaveBeenCalled())
    await userEvent.upload(fileInput(container), pdf())

    expect(await screen.findByText(/Free tier limit of 3 reached/)).toBeInTheDocument()
    expect(screen.getByText('Free tier limit reached')).toBeInTheDocument()
  })

  it('shows a plain error for a non-quota failure', async () => {
    const err = new Error('Upload failed')
    err.status = 500
    uploadPaper.mockRejectedValue(err)

    const { container } = render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    await waitFor(() => expect(listPapers).toHaveBeenCalled())
    await userEvent.upload(fileInput(container), pdf())

    expect(await screen.findByText('Upload failed')).toBeInTheDocument()
    expect(screen.queryByText('Free tier limit reached')).not.toBeInTheDocument()
  })

  it('a failed upload does not blank the existing list', async () => {
    // Regression guard: upload errors and list errors are separate state, so a
    // rejected upload must not make the page claim it cannot load your drafts.
    listPapers.mockResolvedValue([draft({ filename: 'existing.pdf' })])
    const err = new Error('Upload failed'); err.status = 500
    uploadPaper.mockRejectedValue(err)

    const { container } = render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    await screen.findByText('existing')
    await userEvent.upload(fileInput(container), pdf())

    await screen.findByText('Upload failed')
    expect(screen.getByText('existing')).toBeInTheDocument()
    expect(screen.queryByText('Could not load your drafts.')).not.toBeInTheDocument()
  })
})

describe('delete', () => {
  const row = async () => {
    const name = await screen.findByText('my-draft')
    return name.closest('div.flex.items-center.gap-4')
  }

  it('requires a second click to confirm', async () => {
    listPapers.mockResolvedValue([draft()])
    render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    const r = await row()

    await userEvent.click(within(r).getByRole('button', { name: 'Delete' }))
    expect(deletePaper).not.toHaveBeenCalled()
    expect(within(r).getByRole('button', { name: 'Confirm?' })).toBeInTheDocument()

    await userEvent.click(within(r).getByRole('button', { name: 'Confirm?' }))
    await waitFor(() => expect(deletePaper).toHaveBeenCalledWith('d1'))
  })

  it('removes the row without refetching once the delete succeeds', async () => {
    listPapers.mockResolvedValue([draft()])
    deletePaper.mockResolvedValue({})
    render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    const r = await row()

    await userEvent.click(within(r).getByRole('button', { name: 'Delete' }))
    await userEvent.click(within(r).getByRole('button', { name: 'Confirm?' }))

    await waitFor(() => expect(screen.queryByText('my-draft')).not.toBeInTheDocument())
    expect(listPapers).toHaveBeenCalledTimes(1)
  })

  it('keeps the row and explains when the delete fails', async () => {
    listPapers.mockResolvedValue([draft()])
    deletePaper.mockRejectedValue(new Error('Delete failed'))
    render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    const r = await row()

    await userEvent.click(within(r).getByRole('button', { name: 'Delete' }))
    await userEvent.click(within(r).getByRole('button', { name: 'Confirm?' }))

    expect(await screen.findByText('Delete failed')).toBeInTheDocument()
    expect(screen.getByText('my-draft')).toBeInTheDocument()
    // Back to the un-armed state, so a stray click cannot delete.
    expect(within(r).getByRole('button', { name: 'Delete' })).toBeInTheDocument()
  })
})

describe('polling while a draft ingests', () => {
  /** Advance fake timers and let the resulting state updates settle in act(). */
  const tick = (ms) => act(async () => { await vi.advanceTimersByTimeAsync(ms) })

  it('refetches on an interval until nothing is processing', async () => {
    vi.useFakeTimers()
    listPapers.mockResolvedValue([draft({ status: 'processing' })])
    render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)

    // Wait for the row to render, not just for the call: the polling effect
    // only starts once `drafts` state has committed with a processing entry.
    await vi.waitFor(() => expect(screen.getByText('Processing')).toBeInTheDocument())
    const afterMount = listPapers.mock.calls.length

    await tick(3000)
    await vi.waitFor(() =>
      expect(listPapers.mock.calls.length).toBeGreaterThan(afterMount))

    // Once everything reports ready the interval is torn down.
    listPapers.mockResolvedValue([draft({ status: 'ready' })])
    await tick(3000)
    await vi.waitFor(() => expect(screen.getByText('ready')).toBeInTheDocument())

    const settled = listPapers.mock.calls.length
    await tick(12000)
    expect(listPapers).toHaveBeenCalledTimes(settled)
  })

  it('does not poll when every draft is already ready', async () => {
    vi.useFakeTimers()
    listPapers.mockResolvedValue([draft({ status: 'ready' })])
    render(<DraftsPage onOpen={() => {}} onBack={() => {}} />)
    await vi.waitFor(() => expect(screen.getByText('my-draft')).toBeInTheDocument())

    await tick(15000)
    expect(listPapers).toHaveBeenCalledTimes(1)
  })
})
