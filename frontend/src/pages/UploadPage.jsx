import { useState, useRef, useCallback } from 'react'
import { uploadPaper, getPaperStatus } from '../api'

export default function UploadPage({ onPaperReady }) {
  const [dragging, setDragging] = useState(false)
  const [phase, setPhase] = useState('idle') // idle | uploading | processing | error
  const [filename, setFilename] = useState('')
  const [error, setError] = useState('')
  const inputRef = useRef()

  const handleFile = useCallback(async (file) => {
    if (!file || !file.name.endsWith('.pdf')) {
      setError('Please upload a PDF file.')
      return
    }

    setError('')
    setFilename(file.name)
    setPhase('uploading')

    try {
      const { paper_id } = await uploadPaper(file)
      setPhase('processing')

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
    <div className="min-h-screen bg-gray-950 flex flex-col items-center justify-center px-4"
      style={{ fontFamily: "'DM Sans', sans-serif" }}>

      {/* Google Font */}
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500&family=Playfair+Display:wght@600&display=swap" rel="stylesheet" />

      {/* Logo */}
      <div className="mb-12 text-center">
        <h1 className="text-5xl text-white mb-3"
          style={{ fontFamily: "'Playfair Display', serif" }}>
          PaperMind
        </h1>
        <p className="text-gray-500 text-sm tracking-widest uppercase">
          Ask anything. Grounded in your paper.
        </p>
      </div>

      {/* Drop Zone */}
      {phase === 'idle' || phase === 'error' ? (
        <div
          onClick={() => inputRef.current.click()}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          className={`
            w-full max-w-lg border-2 border-dashed rounded-2xl p-16
            flex flex-col items-center justify-center cursor-pointer
            transition-all duration-200
            ${dragging
              ? 'border-amber-400 bg-amber-400/5 scale-[1.02]'
              : 'border-gray-700 hover:border-gray-500 bg-gray-900/50'}
          `}
        >
          <div className="text-4xl mb-4">📄</div>
          <p className="text-white font-medium mb-1">
            Drop your PDF here
          </p>
          <p className="text-gray-500 text-sm">or click to browse</p>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>
      ) : null}

      {/* Uploading state */}
      {phase === 'uploading' && (
        <div className="w-full max-w-lg bg-gray-900 rounded-2xl p-12 flex flex-col items-center">
          <div className="w-8 h-8 border-2 border-amber-400 border-t-transparent rounded-full animate-spin mb-6" />
          <p className="text-white font-medium">Uploading {filename}...</p>
        </div>
      )}

      {/* Processing state */}
      {phase === 'processing' && (
        <div className="w-full max-w-lg bg-gray-900 rounded-2xl p-12 flex flex-col items-center">
          <div className="w-8 h-8 border-2 border-amber-400 border-t-transparent rounded-full animate-spin mb-6" />
          <p className="text-white font-medium mb-2">Processing {filename}</p>
          <p className="text-gray-500 text-sm text-center">
            Parsing sections, building embeddings.<br />This takes about 30 seconds.
          </p>
        </div>
      )}

      {/* Error state */}
      {phase === 'error' && (
        <div className="w-full max-w-lg mt-4">
          <div className="bg-red-950/50 border border-red-800 rounded-xl px-5 py-4 text-red-300 text-sm text-center">
            {error}
          </div>
          <button
            onClick={() => { setPhase('idle'); setError('') }}
            className="mt-3 w-full text-center text-gray-500 hover:text-gray-300 text-sm transition-colors"
          >
            Try again
          </button>
        </div>
      )}

      <p className="mt-10 text-gray-700 text-xs">
        Research papers only · PDF format · Max ~50 pages recommended
      </p>
    </div>
  )
}