/**
 * Tests for escapeHtml.
 *
 * This is a security boundary, not a formatting helper. Its output is
 * interpolated into raw HTML in two places:
 *   - ChatPage's "export conversation" builds an HTML document out of question
 *     text, answer text, section names and the filename;
 *   - PDFPreviewPanel builds a highlight overlay.
 * Both take content that ultimately comes from an uploaded PDF or from what the
 * user typed, so an unescaped `<script>` would execute in the exported file.
 */
import { describe, it, expect } from 'vitest'

import { escapeHtml } from './textUtils'

describe('escaping', () => {
  it('neutralises a script tag', () => {
    expect(escapeHtml('<script>alert(1)</script>'))
      .toBe('&lt;script&gt;alert(1)&lt;/script&gt;')
  })

  it('escapes angle brackets, ampersands and double quotes', () => {
    expect(escapeHtml('<')).toBe('&lt;')
    expect(escapeHtml('>')).toBe('&gt;')
    expect(escapeHtml('&')).toBe('&amp;')
    expect(escapeHtml('"')).toBe('&quot;')
  })

  it('escapes the ampersand first, so entities are not double-broken', () => {
    // If `<` were replaced before `&`, this would come out as `&amp;lt;`.
    expect(escapeHtml('&lt;')).toBe('&amp;lt;')
  })

  it('handles every special character in one string', () => {
    expect(escapeHtml('a & b < c > d "e"'))
      .toBe('a &amp; b &lt; c &gt; d &quot;e&quot;')
  })

  it('escapes repeated occurrences, not just the first', () => {
    expect(escapeHtml('<<>>')).toBe('&lt;&lt;&gt;&gt;')
  })

  it('leaves ordinary prose untouched', () => {
    const s = 'The Transformer reached 28.4 BLEU on WMT14 EN-DE.'
    expect(escapeHtml(s)).toBe(s)
  })
})

describe('non-string input', () => {
  // Callers pass metadata straight through (section names, page numbers,
  // filenames), any of which can be absent.
  it('renders null and undefined as an empty string, not "null"', () => {
    expect(escapeHtml(null)).toBe('')
    expect(escapeHtml(undefined)).toBe('')
    expect(escapeHtml('')).toBe('')
  })

  it('stringifies numbers', () => {
    expect(escapeHtml(42)).toBe('42')
  })

  it('coerces 0 and false to their text, not to empty', () => {
    // `String(str || '')` turns these falsy values into ''. Pinning the current
    // behaviour: page 0 would render blank rather than "0".
    expect(escapeHtml(0)).toBe('')
    expect(escapeHtml(false)).toBe('')
  })
})

describe('known limits', () => {
  it('does NOT escape single quotes', () => {
    // Safe only while every interpolation site is a text node or a
    // double-quoted attribute. If a single-quoted attribute is ever added,
    // this must be extended to &#39;.
    expect(escapeHtml("it's")).toBe("it's")
  })
})
