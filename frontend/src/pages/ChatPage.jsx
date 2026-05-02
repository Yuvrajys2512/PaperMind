import { useState, useRef, useEffect } from 'react'
import { queryPaper, listPapers } from '../api'

function ConfidenceBar({ score }) {
  const color = score >= 70 ? 'bg-green-500' : score >= 40 ? 'bg-yellow-400' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2 mt-2">
      <div className="flex-1 bg-gray-800 rounded-full h-1.5">
        <div className={`${color} h-1.5 rounded-full transition-all`} style={{ width: `${score}%` }} />
      </div>
      <span className="text-xs text-gray-500 w-16">{score.toFixed(1)}% conf.</span>
    </div>
  )
}

function SourceChip({ source }) {
  return (
    <span className="inline-block bg-gray-800 text-gray-400 text-xs px-2 py-0.5 rounded mr-1 mb-1">
      {source.section} · p.{source.page}
    </span>
  )
}

function Message({ msg }) {
  if (msg.role === 'user') {
    return (
      <div className="flex justify-end mb-6">
        <div className="bg-gray-800 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-xl text-sm">
          {msg.content}
        </div>
      </div>
    )
  }

  const { answer, confidence, sources, attempts, warning } = msg.content
  return (
    <div className="flex justify-start mb-6">
      <div className="max-w-2xl w-full">
        <div className="bg-gray-900 border border-gray-800 rounded-2xl rounded-tl-sm px-5 py-4">
          <p className="text-gray-100 text-sm leading-relaxed">{answer}</p>
          <ConfidenceBar score={confidence} />
          {sources && sources.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-800">
              <p className="text-xs text-gray-600 mb-1">Sources</p>
              <div>{sources.map((s, i) => <SourceChip key={i} source={s} />)}</div>
            </div>
          )}
          {attempts > 1 && (
            <p className="text-xs text-amber-600 mt-2">{attempts} attempts</p>
          )}
        </div>
        {warning && (
          <div className="mt-2 bg-yellow-950/40 border border-yellow-800/50 rounded-xl px-4 py-2 text-yellow-400 text-xs">
            ⚠ {warning}
          </div>
        )}
      </div>
    </div>
  )
}

export default function ChatPage({ paper: initialPaper, onBack }) {
  const [paper, setPaper] = useState(initialPaper)
  // Each paper gets its own message history: { [paper_id]: messages[] }
  const [allMessages, setAllMessages] = useState({ [initialPaper.paper_id]: [] })
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [papers, setPapers] = useState([])
  const [showSwitcher, setShowSwitcher] = useState(false)
  const bottomRef = useRef()

  const messages = allMessages[paper.paper_id] || []

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [allMessages, loading])

  // Load all papers when switcher is opened
  useEffect(() => {
    if (showSwitcher) {
      listPapers().then(setPapers).catch(() => {})
    }
  }, [showSwitcher])

  const switchToPaper = (p) => {
    setPaper(p)
    // Initialize empty history for this paper if we haven't seen it
    setAllMessages(prev => ({
      ...prev,
      [p.paper_id]: prev[p.paper_id] || []
    }))
    setShowSwitcher(false)
  }

  const handleSend = async () => {
    const question = input.trim()
    if (!question || loading) return

    setInput('')
    const paperId = paper.paper_id
    setAllMessages(prev => ({
      ...prev,
      [paperId]: [...(prev[paperId] || []), { role: 'user', content: question }]
    }))
    setLoading(true)

    try {
      const result = await queryPaper(paperId, question)
      setAllMessages(prev => ({
        ...prev,
        [paperId]: [...(prev[paperId] || []), { role: 'assistant', content: result }]
      }))
    } catch {
      setAllMessages(prev => ({
        ...prev,
        [paperId]: [...(prev[paperId] || []), {
          role: 'assistant',
          content: { answer: 'Something went wrong. Please try again.', confidence: 0, sources: [], attempts: 1, warning: null }
        }]
      }))
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
  }

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">

      {/* Header */}
      <div className="border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div>
            <h1 className="text-white font-semibold text-sm">{paper.filename}</h1>
            <p className="text-gray-600 text-xs mt-0.5">Ask anything about this paper</p>
          </div>
          <button
            onClick={() => setShowSwitcher(!showSwitcher)}
            className="text-gray-600 hover:text-gray-300 text-xs border border-gray-800 hover:border-gray-600 rounded-lg px-2 py-1 transition-colors ml-2"
          >
            Switch paper ↓
          </button>
        </div>
        <button onClick={onBack} className="text-gray-600 hover:text-gray-300 text-sm transition-colors">
          ← Upload new
        </button>
      </div>

      {/* Paper switcher dropdown */}
      {showSwitcher && (
        <div className="border-b border-gray-800 bg-gray-900 px-6 py-3">
          <p className="text-gray-600 text-xs mb-2">Your papers</p>
          <div className="flex flex-col gap-1">
            {papers.filter(p => p.status === 'ready').map(p => (
              <button
                key={p.paper_id}
                onClick={() => switchToPaper(p)}
                className={`text-left text-sm px-3 py-2 rounded-lg transition-colors ${
                  p.paper_id === paper.paper_id
                    ? 'bg-gray-800 text-white'
                    : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                }`}
              >
                {p.filename}
                {p.paper_id === paper.paper_id && (
                  <span className="ml-2 text-xs text-amber-500">current</span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-8 max-w-3xl w-full mx-auto">
        {messages.length === 0 && (
          <div className="text-center text-gray-700 text-sm mt-20">
            Ask your first question about the paper.
          </div>
        )}
        {messages.map((msg, i) => <Message key={i} msg={msg} />)}
        {loading && (
          <div className="flex justify-start mb-6">
            <div className="bg-gray-900 border border-gray-800 rounded-2xl px-5 py-4">
              <div className="flex gap-1.5 items-center">
                <div className="w-1.5 h-1.5 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-1.5 h-1.5 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-1.5 h-1.5 bg-gray-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-800 px-6 py-4">
        <div className="max-w-3xl mx-auto flex gap-3">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask a question about the paper..."
            rows={1}
            className="flex-1 bg-gray-900 border border-gray-800 rounded-xl px-4 py-3 text-white text-sm placeholder-gray-600 resize-none focus:outline-none focus:border-gray-600 transition-colors"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="bg-white text-gray-950 rounded-xl px-5 py-3 text-sm font-medium disabled:opacity-30 hover:bg-gray-100 transition-colors"
          >
            Send
          </button>
        </div>
        <p className="text-center text-gray-800 text-xs mt-2">Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  )
}