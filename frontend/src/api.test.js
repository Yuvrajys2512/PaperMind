/**
 * Tests for src/api.js — the SSE stream parser and HTTP error shaping.
 *
 * `consumeSSE` is the single most load-bearing piece of client logic in the
 * app: every audit tab, the query view and the compare view all stream through
 * it. It is module-private, so it is exercised through the exported streaming
 * functions rather than being exported just for tests.
 *
 * The cases that matter are the ones a happy-path manual click-through never
 * produces: a frame split across two network chunks, a `data:` field spread
 * over multiple lines, a malformed payload, and a stream that ends without a
 * terminal `done` frame.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

import {
  queryPaperStream,
  overlapStream,
  uploadPaper,
  listPapers,
} from './api'

/** Build a fetch Response whose body streams the given string pieces. */
function sseResponse(pieces, { ok = true, status = 200 } = {}) {
  const encoder = new TextEncoder()
  return {
    ok,
    status,
    body: new ReadableStream({
      start(controller) {
        for (const p of pieces) controller.enqueue(encoder.encode(p))
        controller.close()
      },
    }),
  }
}

const frame = (event, data) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`

beforeEach(() => {
  // api.js reads a Clerk token from window.Clerk; absent it just sends no
  // Authorization header, which is fine here.
  vi.stubGlobal('fetch', vi.fn())
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('consumeSSE', () => {
  it('returns the payload of the done frame and reports every event', async () => {
    fetch.mockResolvedValue(sseResponse([
      frame('open', { req_id: 'r1' }),
      frame('progress', { stage: 'reading' }),
      frame('done', { answer: '42', cached: false }),
    ]))

    const seen = []
    const result = await queryPaperStream('p1', 'q', e => seen.push(e))

    expect(result).toEqual({ answer: '42', cached: false })
    expect(seen.map(e => e.type)).toEqual(['open', 'progress', 'done'])
    expect(seen[1].data.stage).toBe('reading')
  })

  it('reassembles a frame split across two network chunks', async () => {
    // The realistic failure: TCP does not respect frame boundaries.
    const whole = frame('done', { answer: 'split-safe' })
    const cut = Math.floor(whole.length / 2)
    fetch.mockResolvedValue(sseResponse([whole.slice(0, cut), whole.slice(cut)]))

    await expect(queryPaperStream('p1', 'q', () => {})).resolves.toEqual({
      answer: 'split-safe',
    })
  })

  it('handles several frames arriving in one chunk', async () => {
    fetch.mockResolvedValue(sseResponse([
      frame('progress', { stage: 'a' }) + frame('progress', { stage: 'b' }) + frame('done', { ok: 1 }),
    ]))

    const seen = []
    await queryPaperStream('p1', 'q', e => seen.push(e))
    expect(seen.map(e => e.type)).toEqual(['progress', 'progress', 'done'])
  })

  it('concatenates a data payload spread over multiple lines', async () => {
    // SSE permits repeated `data:` lines; the parser joins them before parsing.
    const payload = JSON.stringify({ answer: 'multiline' })
    const half = Math.floor(payload.length / 2)
    fetch.mockResolvedValue(sseResponse([
      `event: done\ndata: ${payload.slice(0, half)}\ndata: ${payload.slice(half)}\n\n`,
    ]))

    await expect(queryPaperStream('p1', 'q', () => {})).resolves.toEqual({
      answer: 'multiline',
    })
  })

  it('rejects with the server message when an error frame arrives', async () => {
    fetch.mockResolvedValue(sseResponse([
      frame('progress', { stage: 'reading' }),
      frame('error', { message: 'Overlap check timed out' }),
    ]))

    await expect(overlapStream('p1', () => {})).rejects.toThrow('Overlap check timed out')
  })

  it('rejects when the stream ends without a done frame', async () => {
    // A dropped connection mid-audit must not resolve with a half-built report.
    fetch.mockResolvedValue(sseResponse([frame('progress', { stage: 'reading' })]))
    await expect(queryPaperStream('p1', 'q', () => {})).rejects.toThrow(
      /without a final result/i,
    )
  })

  it('skips malformed frames rather than failing the whole stream', async () => {
    fetch.mockResolvedValue(sseResponse([
      'event: progress\ndata: {not json\n\n',
      'event: progress\n\n',              // no data line at all
      frame('done', { answer: 'survived' }),
    ]))

    const seen = []
    await expect(queryPaperStream('p1', 'q', e => seen.push(e))).resolves.toEqual({
      answer: 'survived',
    })
    expect(seen.map(e => e.type)).toEqual(['done'])
  })

  it('throws with the HTTP status when the response is not ok', async () => {
    fetch.mockResolvedValue({
      ok: false,
      status: 429,
      body: null,
      json: async () => ({ detail: 'Free tier limit reached' }),
    })

    await expect(overlapStream('p1', () => {})).rejects.toMatchObject({
      message: 'Free tier limit reached',
      status: 429,
    })
  })
})

describe('force flag', () => {
  it('is off by default and adds ?force=1 when set', async () => {
    // A fresh body per call: a ReadableStream can only be read once, so
    // reusing one Response object locks the stream on the second request.
    fetch.mockImplementation(async () => sseResponse([frame('done', {})]))

    await overlapStream('p1', () => {})
    expect(fetch.mock.calls[0][0]).not.toContain('force')

    await overlapStream('p1', () => {}, true)
    expect(fetch.mock.calls[1][0]).toContain('force=1')
  })
})

describe('httpError', () => {
  it('attaches the status so callers can detect a quota 429 without string matching', async () => {
    // DraftsPage relies on err.status === 429 to show the quota banner.
    fetch.mockResolvedValue({
      ok: false,
      status: 429,
      json: async () => ({ detail: 'Free tier limit of 3 reached — you have 2 papers and 1 draft.' }),
    })

    const err = await uploadPaper(new File(['x'], 'a.pdf'), 'draft').catch(e => e)
    expect(err.status).toBe(429)
    expect(err.message).toMatch(/Free tier limit/)
  })

  it('falls back to a generic message when the body has no detail', async () => {
    fetch.mockResolvedValue({ ok: false, status: 500, json: async () => ({}) })
    const err = await uploadPaper(new File(['x'], 'a.pdf')).catch(e => e)
    expect(err.message).toBe('Upload failed')
    expect(err.status).toBe(500)
  })

  it('survives a non-JSON error body', async () => {
    fetch.mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => { throw new SyntaxError('not json') },
    })
    const err = await uploadPaper(new File(['x'], 'a.pdf')).catch(e => e)
    expect(err.message).toBe('Upload failed')
    expect(err.status).toBe(502)
  })
})

describe('listPapers', () => {
  it('requests drafts separately from reference papers', async () => {
    fetch.mockResolvedValue({ ok: true, status: 200, json: async () => [] })

    await listPapers()
    expect(fetch.mock.calls[0][0]).not.toContain('paper_type')

    await listPapers('draft')
    expect(fetch.mock.calls[1][0]).toContain('paper_type=draft')
  })
})
