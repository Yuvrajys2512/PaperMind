import { useState, useRef, useCallback } from 'react'
import { uploadPaper, getPaperStatus } from '../api'

export default function UploadPage({ onPaperReady }) {
  const [dragging, setDragging] = useState(false)
  const [phase, setPhase] = useState('idle') // idle | uploading | processing | error
  const [filename, setFilename] = useState('')
  const [error, setError] = useState('')
  const [uploadProgress, setUploadProgress] = useState(0)
  const inputRef = useRef()

  const handleFile = useCallback(async (file) => {
    if (!file || !file.name.endsWith('.pdf')) {
      setError('Please upload a PDF file.')
      return
    }

    setError('')
    setFilename(file.name)
    setPhase('uploading')
    setUploadProgress(20)

    try {
      const { paper_id } = await uploadPaper(file)
      setUploadProgress(100)
      
      setTimeout(() => {
        setPhase('processing')
      }, 500)

      // Poll until ready or failed
      const interval = setInterval(async () => {
        try {
          const status = await getPaperStatus(paper_id)
          if (status.status === 'ready') {
            clearInterval(interval)
            onPaperReady(status)
          } else if (status.status === 'failed') {
            clearInterval(interval)
            setPhase('error')
            setError(status.error || 'Ingestion failed.')
          }
        } catch {
          clearInterval(interval)
          setPhase('error')
          setError('Lost connection while processing.')
        }
      }, 2000)
    } catch (err) {
      setPhase('error')
      setError('Upload failed. Is the server running?')
    }
  }, [onPaperReady])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }, [handleFile])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  return (
    <div className="min-h-screen cosmic-bg flex flex-col items-center justify-center px-4">
      
      {/* Background Glows */}
      <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-blue-600/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[10%] right-[10%] w-[400px] h-[400px] bg-purple-600/10 rounded-full blur-[100px] pointer-events-none" />

      {/* Hero Content */}
      <div className="relative z-10 mb-16 text-center animate-float">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-[10px] text-gray-400 uppercase tracking-widest mb-6">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
          Powered by PaperMind AI
        </div>
        <h1 className="text-6xl md:text-7xl font-bold text-white mb-6 tracking-tight leading-tight">
          The Future of Research <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-cyan-300 text-glow">
            Starts with PaperMind
          </span>
        </h1>
        <p className="text-gray-400 text-lg max-w-xl mx-auto font-light leading-relaxed">
          Upload your research papers and interact with them like never before. 
          Context-aware, grounded, and incredibly fast.
        </p>
      </div>

      {/* Main Action Area */}
      <div className="relative z-10 w-full max-w-2xl">
        {phase === 'idle' || phase === 'error' ? (
          <div
            onClick={() => inputRef.current.click()}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            className={`
              relative group overflow-hidden glass rounded-3xl p-1 px-1
              transition-all duration-500 ease-out cursor-pointer
              ${dragging ? 'scale-[1.02] neon-glow' : 'hover:neon-glow hover:scale-[1.01]'}
            `}
          >
            <div className="bg-black/40 rounded-[22px] p-12 py-16 flex flex-col items-center justify-center border border-white/5">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-600 to-cyan-400 flex items-center justify-center mb-6 shadow-lg shadow-blue-500/20">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              
              <h2 className="text-2xl font-semibold text-white mb-2">
                {dragging ? 'Drop to Ingest' : 'Ingest Research Paper'}
              </h2>
              <p className="text-gray-500 text-center font-light mb-8 max-w-xs">
                Drag and drop your PDF here or click to browse your workspace files.
              </p>
              
              <div className="px-6 py-2 rounded-full bg-white text-black text-sm font-semibold hover:bg-gray-100 transition-colors">
                Select PDF
              </div>
            </div>

            <input
              ref={inputRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={(e) => handleFile(e.target.files[0])}
            />
          </div>
        ) : null}

        {/* Uploading/Processing State */}
        {(phase === 'uploading' || phase === 'processing') && (
          <div className="glass rounded-3xl p-12 py-16 flex flex-col items-center justify-center border border-white/10 neon-glow">
            <div className="relative w-24 h-24 mb-8">
              <div className="absolute inset-0 border-4 border-blue-500/20 rounded-full" />
              <div className="absolute inset-0 border-4 border-t-cyan-400 rounded-full animate-spin" />
              <div className="absolute inset-4 bg-gradient-to-br from-blue-600 to-cyan-400 rounded-full flex items-center justify-center animate-pulse">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
            </div>
            
            <h3 className="text-2xl font-semibold text-white mb-2">
              {phase === 'uploading' ? 'Uploading Knowledge' : 'Synthesizing Content'}
            </h3>
            <p className="text-gray-500 text-center font-light mb-8">
              {phase === 'uploading' 
                ? `Moving ${filename} to our secure AI workspace...` 
                : `Applying neural parsing to ${filename}. This usually takes 30s.`}
            </p>
            
            <div className="w-full max-w-xs h-1.5 bg-white/5 rounded-full overflow-hidden">
              <div 
                className={`h-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all duration-500 ${phase === 'processing' ? 'animate-pulse w-full' : ''}`}
                style={{ width: phase === 'uploading' ? `${uploadProgress}%` : '100%' }}
              />
            </div>
          </div>
        )}

        {/* Error Handling */}
        {phase === 'error' && (
          <div className="mt-8 p-4 rounded-2xl bg-red-500/10 border border-red-500/20 flex items-center gap-4 animate-float">
            <div className="w-10 h-10 rounded-xl bg-red-500/20 flex items-center justify-center flex-shrink-0">
              <span className="text-red-500 text-xl font-bold">!</span>
            </div>
            <div className="flex-1">
              <p className="text-red-200 text-sm font-medium">{error}</p>
              <button 
                onClick={() => setPhase('idle')}
                className="text-red-400/80 hover:text-red-300 text-xs transition-colors mt-0.5"
              >
                Reset and try again
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="absolute bottom-8 left-0 right-0 text-center opacity-30 pointer-events-none">
        <p className="text-gray-500 text-[10px] uppercase tracking-[0.2em]">
          End-to-end encrypted · ISO 27001 Certified Infrastructure
        </p>
      </div>
    </div>
  )
}