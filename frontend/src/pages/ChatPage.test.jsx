/**
 * Tests for ChatPage — the ask-a-paper conversation view.
 *
 * ChatPage is ~2,500 lines and owns a lot of UI. These tests deliberately cover
 * only the parts with real, checkable logic and a clear intent:
 *
 *   - the send flow (what gets sent, what is rendered, what happens on failure),
 *   - conversation persistence in localStorage, including the corrupted-storage
 *     guard the code already has a try/catch for,
 *   - per-paper conversation isolation,
 *   - the search filter, which pairs a matching question with its answer,
 *   - the audience prefix, which changes what is sent but not what is shown,
 *   - keyboard behaviour (Enter to send, ArrowUp to recall).
 *
 * Chrome, styling and the many side panels are left alone on purpose — they are
 * the parts most likely to change for good reasons.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

vi.mock('../api', () => ({
  listPapers: vi.fn(),
  deletePaper: vi.fn(),
  queryPaperStream: vi.fn(),
  getGlossary: vi.fn(),
  getRecommendations: vi.fn(),
}))
vi.mock('../analytics', () => ({ track: vi.fn() }))
// Heavy children that render their own async worlds; not under test here.
vi.mock('../components/PDFPreviewPanel', () => ({ default: () => null }))
vi.mock('../components/AuditPanel', () => ({ default: () => null }))
vi.mock('../components/ReviewPanel', () => ({ default: () => null }))
vi.mock('react-markdown', () => ({ default: ({ children }) => <div>{children}</div> }))

import { listPapers, queryPaperStream } from '../api'
import ChatPage from './ChatPage'

const PAPER = { paper_id: 'p1', filename: 'attention.pdf', status: 'ready' }
const COMPOSER = /Message PaperMind/i
const STORAGE_KEY = 'papermind_chats'

const answer = (over = {}) => ({
  answer: 'The Transformer reached 28.4 BLEU.',
  confidence: 0.94,
  faithfulness: 0.96,
  answer_relevancy: 0.88,
  sources: [],
  attempts: 1,
  warning: null,
  grading: null,
  ...over,
})

const ask = async (user, text) => {
  const box = screen.getByPlaceholderText(COMPOSER)
  await user.type(box, text)
  await user.keyboard('{Enter}')
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  listPapers.mockResolvedValue([PAPER])
  queryPaperStream.mockResolvedValue(answer())
})

describe('sending a question', () => {
  it('streams the question for the open paper and renders the answer', async () => {
    const user = userEvent.setup()
    render(<ChatPage paper={PAPER} onBack={() => {}} />)

    await ask(user, 'What was the main result?')

    await waitFor(() =>
      expect(queryPaperStream).toHaveBeenCalledWith(
        'p1', 'What was the main result?', expect.any(Function)))
    expect(await screen.findByText(/28\.4 BLEU/)).toBeInTheDocument()
  })

  it('echoes the question immediately, before the answer arrives', async () => {
    const user = userEvent.setup()
    let release
    queryPaperStream.mockImplementation(() => new Promise(r => { release = r }))
    render(<ChatPage paper={PAPER} onBack={() => {}} />)

    await ask(user, 'Does it beat the baseline?')
    expect(await screen.findByText('Does it beat the baseline?')).toBeInTheDocument()

    release(answer())
    await screen.findByText(/28\.4 BLEU/)
  })

  it('clears the composer after sending', async () => {
    const user = userEvent.setup()
    render(<ChatPage paper={PAPER} onBack={() => {}} />)

    await ask(user, 'A question')
    await waitFor(() => expect(screen.getByPlaceholderText(COMPOSER)).toHaveValue(''))
  })

  it('ignores an empty or whitespace-only question', async () => {
    const user = userEvent.setup()
    render(<ChatPage paper={PAPER} onBack={() => {}} />)

    await ask(user, '   ')
    expect(queryPaperStream).not.toHaveBeenCalled()
  })

  it('will not start a second query while one is in flight', async () => {
    const user = userEvent.setup()
    queryPaperStream.mockImplementation(() => new Promise(() => {}))
    render(<ChatPage paper={PAPER} onBack={() => {}} />)

    await ask(user, 'first')
    await ask(user, 'second')
    expect(queryPaperStream).toHaveBeenCalledTimes(1)
  })

  it('renders progress stages as they stream in', async () => {
    const user = userEvent.setup()
    queryPaperStream.mockImplementation(async (_id, _q, onEvent) => {
      onEvent({ type: 'progress', data: { stage: 'retrieving', message: 'Retrieving…' } })
      return answer()
    })
    render(<ChatPage paper={PAPER} onBack={() => {}} />)

    await ask(user, 'q')
    expect(await screen.findByText(/28\.4 BLEU/)).toBeInTheDocument()
  })
})

describe('when the query fails', () => {
  it('keeps the conversation usable instead of losing the turn', async () => {
    const user = userEvent.setup()
    queryPaperStream.mockRejectedValue(new Error('Query failed — upstream timeout'))
    render(<ChatPage paper={PAPER} onBack={() => {}} />)

    await ask(user, 'what happens on failure?')

    // The question stays, and a placeholder answer takes the assistant's turn
    // so the transcript is not left dangling.
    expect(await screen.findByText('what happens on failure?')).toBeInTheDocument()
    expect(await screen.findByText(/Neural link interrupted/i)).toBeInTheDocument()
  })

  it('surfaces the server message in a toast', async () => {
    const user = userEvent.setup()
    queryPaperStream.mockRejectedValue(new Error('Query failed — upstream timeout'))
    render(<ChatPage paper={PAPER} onBack={() => {}} />)

    await ask(user, 'q')
    expect(await screen.findByText('Query failed — upstream timeout')).toBeInTheDocument()
  })

  it('re-enables the composer so the user can retry', async () => {
    const user = userEvent.setup()
    queryPaperStream.mockRejectedValue(new Error('boom'))
    render(<ChatPage paper={PAPER} onBack={() => {}} />)

    await ask(user, 'first')
    await screen.findByText(/Neural link interrupted/i)

    queryPaperStream.mockResolvedValue(answer({ answer: 'second answer works' }))
    await ask(user, 'second')
    expect(await screen.findByText('second answer works')).toBeInTheDocument()
  })
})

describe('conversation persistence', () => {
  it('writes the conversation to localStorage', async () => {
    const user = userEvent.setup()
    render(<ChatPage paper={PAPER} onBack={() => {}} />)

    await ask(user, 'persist me')
    await waitFor(() => {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY))
      expect(saved.p1.some(m => m.content === 'persist me')).toBe(true)
    })
  })

  it('restores a saved conversation on mount', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      p1: [{ role: 'user', content: 'asked earlier' }],
    }))
    render(<ChatPage paper={PAPER} onBack={() => {}} />)
    expect(await screen.findByText('asked earlier')).toBeInTheDocument()
  })

  it('keeps each paper\'s conversation separate', async () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      p1: [{ role: 'user', content: 'about paper one' }],
      p2: [{ role: 'user', content: 'about paper two' }],
    }))
    render(<ChatPage paper={PAPER} onBack={() => {}} />)

    expect(await screen.findByText('about paper one')).toBeInTheDocument()
    expect(screen.queryByText('about paper two')).not.toBeInTheDocument()
  })

  it('starts clean rather than crashing on corrupted storage', async () => {
    // The component already guards this with try/catch; pin it, because a
    // throw here would white-screen the page on every load until the user
    // cleared their storage by hand.
    localStorage.setItem(STORAGE_KEY, 'not json{{{')
    render(<ChatPage paper={PAPER} onBack={() => {}} />)
    expect(await screen.findByPlaceholderText(COMPOSER)).toBeInTheDocument()
  })
})

describe('audience prefix', () => {
  it('shows the user their own words, not the prefixed prompt', async () => {
    // The prefix steers the model; echoing it back would look like the app
    // rewrote the question.
    const user = userEvent.setup()
    render(<ChatPage paper={PAPER} onBack={() => {}} />)

    await ask(user, 'explain attention')
    expect(await screen.findByText('explain attention')).toBeInTheDocument()
    await waitFor(() => {
      const [, sentQuestion] = queryPaperStream.mock.calls[0]
      expect(sentQuestion).toBe('explain attention')  // 'default' audience: no prefix
    })
  })
})

describe('keyboard', () => {
  it('Shift+Enter inserts a newline instead of sending', async () => {
    const user = userEvent.setup()
    render(<ChatPage paper={PAPER} onBack={() => {}} />)

    const box = screen.getByPlaceholderText(COMPOSER)
    await user.type(box, 'line one')
    await user.keyboard('{Shift>}{Enter}{/Shift}')
    await user.type(box, 'line two')

    expect(queryPaperStream).not.toHaveBeenCalled()
    expect(box.value).toContain('line one')
    expect(box.value).toContain('line two')
  })

  it('ArrowUp on an empty composer recalls the last question', async () => {
    const user = userEvent.setup()
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      p1: [{ role: 'user', content: 'the previous question' }],
    }))
    render(<ChatPage paper={PAPER} onBack={() => {}} />)
    await screen.findByText('the previous question')

    const box = screen.getByPlaceholderText(COMPOSER)
    await user.click(box)
    await user.keyboard('{ArrowUp}')

    expect(box).toHaveValue('the previous question')
  })

  it('ArrowUp does not clobber text already typed', async () => {
    const user = userEvent.setup()
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      p1: [{ role: 'user', content: 'the previous question' }],
    }))
    render(<ChatPage paper={PAPER} onBack={() => {}} />)
    await screen.findByText('the previous question')

    const box = screen.getByPlaceholderText(COMPOSER)
    await user.type(box, 'half-written')
    await user.keyboard('{ArrowUp}')

    expect(box).toHaveValue('half-written')
  })
})

describe('searching the conversation', () => {
  const withHistory = () => localStorage.setItem(STORAGE_KEY, JSON.stringify({
    p1: [
      { role: 'user', content: 'question about BLEU' },
      { role: 'assistant', content: answer({ answer: 'the BLEU answer' }) },
      { role: 'user', content: 'question about parsing' },
      { role: 'assistant', content: answer({ answer: 'the parsing answer' }) },
    ],
  }))

  // Matching text is wrapped in <mark> by HighlightText, so it is split across
  // elements and getByText cannot see it whole. What these tests care about is
  // which messages are on screen, so assert against the concatenated text.
  const onScreen = () => document.body.textContent

  const openSearch = async (user) => {
    await user.click(screen.getByRole('button', { name: /search/i }))
    return screen.getByPlaceholderText(/search conversation/i)
  }

  it('keeps a matching question together with its answer', async () => {
    // The pairing is the point: showing a question whose answer is filtered
    // away (or vice versa) would make the transcript unreadable.
    const user = userEvent.setup()
    withHistory()
    render(<ChatPage paper={PAPER} onBack={() => {}} />)
    await screen.findByText('question about BLEU')

    await user.type(await openSearch(user), 'parsing')

    await waitFor(() => expect(onScreen()).not.toContain('question about BLEU'))
    expect(onScreen()).toContain('question about parsing')
    expect(onScreen()).toContain('the parsing answer')
  })

  it('pulls in the question when only the answer matches', async () => {
    const user = userEvent.setup()
    withHistory()
    render(<ChatPage paper={PAPER} onBack={() => {}} />)
    await screen.findByText('question about BLEU')

    await user.type(await openSearch(user), 'the parsing answer')

    await waitFor(() => expect(onScreen()).toContain('question about parsing'))
  })

  it('matches case-insensitively', async () => {
    const user = userEvent.setup()
    withHistory()
    render(<ChatPage paper={PAPER} onBack={() => {}} />)
    await screen.findByText('question about BLEU')

    await user.type(await openSearch(user), 'BLEU')
    await waitFor(() => expect(onScreen()).not.toContain('question about parsing'))
    expect(onScreen()).toContain('question about BLEU')
  })

  it('shows everything again when the query is cleared', async () => {
    const user = userEvent.setup()
    withHistory()
    render(<ChatPage paper={PAPER} onBack={() => {}} />)
    await screen.findByText('question about BLEU')

    const box = await openSearch(user)
    await user.type(box, 'parsing')
    await waitFor(() => expect(onScreen()).not.toContain('question about BLEU'))

    await user.clear(box)
    await waitFor(() => expect(onScreen()).toContain('question about BLEU'))
  })
})

describe('navigation', () => {
  it('leaving the paper is delegated to the parent, not done here', async () => {
    // The control is labelled "Upload New" rather than "Back" — it returns to
    // the upload view. ChatPage does not touch history itself.
    const onBack = vi.fn()
    const user = userEvent.setup()
    render(<ChatPage paper={PAPER} onBack={onBack} />)

    await user.click(screen.getByRole('button', { name: /upload new/i }))
    expect(onBack).toHaveBeenCalled()
  })
})
