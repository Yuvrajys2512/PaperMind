import { useState, useCallback } from 'react'
import { rewriteText } from '../api'

function CosmicOrbs() {
  return (
    <>
      <div className="cosmic-orb-cyan" />
      <div className="cosmic-orb-violet" />
      <div className="cosmic-orb-pink" />
      <div className="scan-line" />
    </>
  )
}

const MODES = [
  {
    id: 'academic',
    label: 'Academic',
    description: 'Formal tone, precise vocabulary, proper hedging',
  },
  {
    id: 'plain',
    label: 'Plain language',
    description: 'Jargon replaced, accessible to non-specialists',
  },
  {
    id: 'concise',
    label: 'Concise',
    description: 'Cut filler, same substance, ~30–50% shorter',
  },
]

const MAX_CHARS = 4000

export default function RewritePage({ onBack }) {
  const [text, setText]       = useState('')
  const [mode, setMode]       = useState('academic')
  const [output, setOutput]   = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')
  const [copied, setCopied]   = useState(false)

  const handleRewrite = useCallback(async () => {
    if (!text.trim() || loading) return
    setError('')
    setOutput('')
    setLoading(true)
    try {
      const res = await rewriteText(text, mode)
      setOutput(res.result)
    } catch (e) {
      setError(e.message || 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }, [text, mode, loading])

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(output).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }, [output])

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0

  return (
    <div className="min-h-screen cosmic-bg flex flex-col" style={{ position: 'relative' }}>
      <CosmicOrbs />

      {/* Grid overlay */}
      <div className="fixed inset-0 pointer-events-none" style={{
        backgroundImage: `linear-gradient(rgba(0,245,255,0.015) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(0,245,255,0.015) 1px, transparent 1px)`,
        backgroundSize: '60px 60px',
        zIndex: 0,
      }} />

      {/* Header */}
      <div className="relative z-10 flex items-center justify-between px-6 py-4 border-b"
        style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
        <button onClick={onBack}
          className="flex items-center gap-2 text-gray-500 hover:text-gray-300 transition-colors text-sm">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7" />
          </svg>
          Back
        </button>

        <div className="text-center">
          <h1 className="text-white font-semibold text-sm">Writing assistant</h1>
          <p className="text-gray-600 text-xs">Improve tone and clarity — not a bypass tool</p>
        </div>

        <div className="w-16" /> {/* spacer */}
      </div>

      {/* Body */}
      <div className="relative z-10 flex-1 flex flex-col lg:flex-row gap-5 p-6 max-w-6xl mx-auto w-full">

        {/* Left — input */}
        <div className="flex-1 flex flex-col gap-4">
          {/* Mode selector */}
          <div className="flex gap-2 flex-wrap">
            {MODES.map(m => (
              <button key={m.id} onClick={() => setMode(m.id)}
                className="flex flex-col px-4 py-2.5 rounded-xl transition-all duration-200 text-left"
                style={{
                  background: mode === m.id
                    ? 'rgba(0,245,255,0.1)'
                    : 'rgba(255,255,255,0.03)',
                  border: mode === m.id
                    ? '1px solid rgba(0,245,255,0.35)'
                    : '1px solid rgba(255,255,255,0.07)',
                }}>
                <span className="text-sm font-medium"
                  style={{ color: mode === m.id ? '#22d3ee' : 'rgba(255,255,255,0.6)' }}>
                  {m.label}
                </span>
                <span className="text-[10px] text-gray-600 mt-0.5 max-w-[150px]">{m.description}</span>
              </button>
            ))}
          </div>

          {/* Textarea */}
          <div className="flex-1 flex flex-col rounded-2xl overflow-hidden"
            style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.07)',
            }}>
            <textarea
              value={text}
              onChange={e => setText(e.target.value.slice(0, MAX_CHARS))}
              placeholder="Paste your text here…"
              className="flex-1 w-full bg-transparent text-gray-200 text-sm leading-relaxed p-5 outline-none resize-none placeholder-gray-700"
              style={{ minHeight: 260, fontFamily: 'inherit' }}
            />
            <div className="flex items-center justify-between px-5 py-2.5 border-t"
              style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
              <span className="text-[10px] text-gray-700">
                {wordCount} word{wordCount !== 1 ? 's' : ''}
              </span>
              <span className="text-[10px]"
                style={{ color: text.length > MAX_CHARS * 0.9 ? 'rgba(251,191,36,0.6)' : 'rgba(75,85,99,1)' }}>
                {text.length} / {MAX_CHARS}
              </span>
            </div>
          </div>

          {/* Rewrite button */}
          <button
            onClick={handleRewrite}
            disabled={!text.trim() || loading}
            className="w-full py-3 rounded-2xl font-semibold text-sm transition-all duration-200"
            style={{
              background: !text.trim() || loading
                ? 'rgba(255,255,255,0.05)'
                : 'linear-gradient(135deg, rgba(0,245,255,0.2), rgba(99,102,241,0.2))',
              border: !text.trim() || loading
                ? '1px solid rgba(255,255,255,0.07)'
                : '1px solid rgba(0,245,255,0.3)',
              color: !text.trim() || loading ? 'rgba(107,114,128,1)' : '#e2e8f0',
              cursor: !text.trim() || loading ? 'not-allowed' : 'pointer',
            }}>
            {loading ? 'Rewriting…' : 'Rewrite'}
          </button>
        </div>

        {/* Right — output */}
        <div className="flex-1 flex flex-col gap-4">
          <div className="flex-1 rounded-2xl overflow-hidden flex flex-col"
            style={{
              background: 'rgba(255,255,255,0.03)',
              border: output
                ? '1px solid rgba(0,245,255,0.15)'
                : '1px solid rgba(255,255,255,0.07)',
              minHeight: 320,
            }}>
            {loading && (
              <div className="flex-1 flex flex-col items-center justify-center gap-3">
                <div className="w-8 h-8 rounded-full border-2 border-transparent"
                  style={{ borderTopColor: '#00f5ff', animation: 'spin 1s linear infinite' }} />
                <p className="text-gray-600 text-sm">Rewriting…</p>
              </div>
            )}

            {!loading && !output && !error && (
              <div className="flex-1 flex items-center justify-center">
                <p className="text-gray-700 text-sm">Output appears here</p>
              </div>
            )}

            {!loading && error && (
              <div className="flex-1 flex items-center justify-center p-6">
                <p className="text-red-400/70 text-sm text-center">{error}</p>
              </div>
            )}

            {!loading && output && (
              <p className="flex-1 text-gray-200 text-sm leading-relaxed p-5 whitespace-pre-wrap">
                {output}
              </p>
            )}

            {output && !loading && (
              <div className="flex items-center justify-between px-5 py-2.5 border-t"
                style={{ borderColor: 'rgba(255,255,255,0.05)' }}>
                <span className="text-[10px] text-gray-700">
                  {output.trim().split(/\s+/).length} words
                </span>
                <button onClick={handleCopy}
                  className="flex items-center gap-1.5 text-[10px] transition-colors"
                  style={{ color: copied ? '#22d3ee' : 'rgba(107,114,128,1)' }}>
                  {copied ? (
                    <>
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                      </svg>
                      Copied
                    </>
                  ) : (
                    <>
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                          d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      Copy
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}
