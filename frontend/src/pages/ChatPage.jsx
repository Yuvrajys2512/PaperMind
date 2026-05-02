import { useState, useRef, useEffect } from 'react'
import { queryPaper, listPapers } from '../api'

function ConfidenceBar({ score }) {
  const color = score >= 70 ? 'bg-cyan-400' : score >= 40 ? 'bg-blue-500' : 'bg-red-500'
  const glow = score >= 70 ? 'shadow-[0_0_10px_rgba(34,211,238,0.4)]' : ''
  
  return (
    <div className="flex items-center gap-3 mt-4">
      <div className="flex-1 bg-white/5 rounded-full h-1.5 overflow-hidden">
        <div 
          className={`${color} ${glow} h-1.5 rounded-full transition-all duration-1000 ease-out`} 
          style={{ width: `${score}%` }} 
        />
      </div>
      <span className="text-[10px] font-medium text-gray-500 w-16 uppercase tracking-wider">
        {score.toFixed(1)}% Match
      </span>
    </div>
  )
}

function SourceChip({ source }) {
  return (
    <div className="group flex items-center gap-2 bg-white/5 hover:bg-white/10 border border-white/5 hover:border-blue-500/30 text-gray-400 hover:text-blue-300 text-[10px] px-3 py-1.5 rounded-full transition-all cursor-default">
      <span className="w-1 h-1 rounded-full bg-blue-500 group-hover:bg-blue-300" />
      {source.section} <span className="opacity-30">·</span> p.{source.page}
    </div>
  )
}

function Message({ msg }) {
  if (msg.role === 'user') {
    return (
      <div className="flex justify-end mb-8">
        <div className="bg-gradient-to-br from-blue-600 to-cyan-500 text-white rounded-3xl rounded-tr-sm px-6 py-4 max-w-xl shadow-xl shadow-blue-500/10 text-sm leading-relaxed">
          {msg.content}
        </div>
      </div>
    )
  }

  const { answer, confidence, sources, attempts, warning } = msg.content
  return (
    <div className="flex justify-start mb-8 group">
      <div className="max-w-3xl w-full">
        <div className="glass-dark rounded-3xl rounded-tl-sm px-7 py-6 border-white/5 group-hover:border-blue-500/20 transition-all duration-300 shadow-2xl">
          <p className="text-gray-200 text-sm leading-loose whitespace-pre-wrap">{answer}</p>
          
          <ConfidenceBar score={confidence} />
          
          {sources && sources.length > 0 && (
            <div className="mt-6 pt-5 border-t border-white/5">
              <p className="text-[10px] uppercase tracking-[0.2em] text-gray-600 mb-3 font-semibold">Evidence Sources</p>
              <div className="flex flex-wrap gap-2">
                {sources.map((s, i) => <SourceChip key={i} source={s} />)}
              </div>
            </div>
          )}
          
          {attempts > 1 && (
            <div className="mt-4 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
              <p className="text-[10px] font-medium text-amber-500/70 uppercase tracking-wider">{attempts} Ingestion Cycles</p>
            </div>
          )}
        </div>
        
        {warning && (
          <div className="mt-3 bg-amber-500/10 border border-amber-500/20 rounded-2xl px-5 py-3 text-amber-200/70 text-[11px] leading-relaxed backdrop-blur-md animate-float">
            <span className="text-amber-500 mr-2 font-bold">!</span> {warning}
          </div>
        )}
      </div>
    </div>
  )
}

export default function ChatPage({ paper: initialPaper, onBack }) {
  const [paper, setPaper] = useState(initialPaper)
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

  useEffect(() => {
    if (showSwitcher) {
      listPapers().then(setPapers).catch(() => {})
    }
  }, [showSwitcher])

  const switchToPaper = (p) => {
    setPaper(p)
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
          content: { answer: 'Neural link interrupted. Please attempt query again.', confidence: 0, sources: [], attempts: 1, warning: null }
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
    <div className="min-h-screen cosmic-bg flex flex-col">
      
      {/* Background Orbs */}
      <div className="fixed top-[-20%] right-[-10%] w-[600px] h-[600px] bg-blue-600/5 rounded-full blur-[150px] pointer-events-none" />
      <div className="fixed bottom-[-10%] left-[-10%] w-[400px] h-[400px] bg-cyan-600/5 rounded-full blur-[120px] pointer-events-none" />

      {/* Premium Header */}
      <header className="sticky top-0 z-50 backdrop-blur-2xl bg-black/40 border-b border-white/5 px-8 py-5">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-6">
            <div className="flex flex-col">
              <h1 className="text-white font-bold text-lg tracking-tight flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cyan-400 neon-glow" />
                {paper.filename}
              </h1>
              <p className="text-gray-500 text-[10px] uppercase tracking-[0.2em] mt-1 font-semibold">Active Document Stream</p>
            </div>
            
            <button
              onClick={() => setShowSwitcher(!showSwitcher)}
              className={`
                flex items-center gap-2 px-4 py-2 rounded-xl transition-all duration-300
                ${showSwitcher ? 'bg-white text-black' : 'bg-white/5 text-gray-400 hover:bg-white/10'}
              `}
            >
              <span className="text-[11px] font-bold uppercase tracking-wider">Workspace</span>
              <svg className={`w-3 h-3 transition-transform duration-300 ${showSwitcher ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="3" d="M19 9l-7 7-7-7" /></svg>
            </button>
          </div>
          
          <button 
            onClick={onBack}
            className="group flex items-center gap-3 text-gray-500 hover:text-white transition-all"
          >
            <span className="text-[11px] font-bold uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-opacity">Upload New</span>
            <div className="w-10 h-10 rounded-full bg-white/5 border border-white/10 flex items-center justify-center group-hover:border-white/30 transition-all">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" /></svg>
            </div>
          </button>
        </div>

        {/* Workspace Switcher Modal Style */}
        {showSwitcher && (
          <div className="absolute top-full left-0 right-0 bg-black/90 backdrop-blur-3xl border-b border-white/5 p-8 animate-in fade-in slide-in-from-top-4 duration-300">
            <div className="max-w-3xl mx-auto">
              <h3 className="text-[10px] uppercase tracking-[0.3em] text-gray-600 mb-6 font-bold text-center">Select Intelligence Context</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {papers.filter(p => p.status === 'ready').map(p => (
                  <button
                    key={p.paper_id}
                    onClick={() => switchToPaper(p)}
                    className={`
                      text-left px-5 py-4 rounded-2xl transition-all border
                      ${p.paper_id === paper.paper_id 
                        ? 'bg-blue-600/10 border-blue-500/30 text-white' 
                        : 'bg-white/5 border-transparent text-gray-400 hover:bg-white/10 hover:border-white/10'}
                    `}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium truncate pr-4">{p.filename}</span>
                      {p.paper_id === paper.paper_id && <span className="w-2 h-2 rounded-full bg-cyan-400 neon-glow" />}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </header>

      {/* Message Stream */}
      <main className="flex-1 overflow-y-auto px-8 py-12">
        <div className="max-w-4xl mx-auto">
          {messages.length === 0 && (
            <div className="text-center py-24 animate-float">
              <div className="w-20 h-20 bg-blue-600/10 rounded-full flex items-center justify-center mx-auto mb-8 border border-blue-500/20">
                <svg className="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-white mb-2">Initialize Context</h2>
              <p className="text-gray-500 text-sm font-light">Ask any question to begin extracting knowledge from the stream.</p>
            </div>
          )}
          
          {messages.map((msg, i) => <Message key={i} msg={msg} />)}
          
          {loading && (
            <div className="flex justify-start mb-8">
              <div className="glass-dark rounded-3xl rounded-tl-sm px-6 py-4 border-white/10">
                <div className="flex gap-2 items-center">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}
          <div ref={bottomRef} className="h-24" />
        </div>
      </main>

      {/* Floating Input Dock */}
      <div className="fixed bottom-10 left-0 right-0 z-50 px-8">
        <div className="max-w-3xl mx-auto">
          <div className="relative glass-dark rounded-[2.5rem] p-2 border-white/10 shadow-[0_20px_50px_rgba(0,0,0,0.5)] focus-within:border-blue-500/40 transition-all duration-500">
            <div className="flex items-center gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Message PaperMind Intelligence..."
                rows={1}
                className="flex-1 bg-transparent border-none px-6 py-4 text-white text-sm placeholder-gray-600 resize-none focus:ring-0 outline-none"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || loading}
                className={`
                  h-12 w-12 rounded-full flex items-center justify-center transition-all duration-300
                  ${!input.trim() || loading 
                    ? 'bg-white/5 text-gray-700' 
                    : 'bg-white text-black hover:scale-105 active:scale-95 shadow-xl shadow-white/10'}
                `}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M14 5l7 7m0 0l-7 7m7-7H3" /></svg>
              </button>
            </div>
          </div>
          <p className="text-center text-gray-600 text-[10px] uppercase tracking-[0.2em] mt-4 font-medium opacity-50">
            Enter to stream · Shift+Enter for newline
          </p>
        </div>
      </div>
    </div>
  )
}