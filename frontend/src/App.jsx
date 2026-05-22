import { useState } from 'react'
import UploadPage from './pages/UploadPage'
import ChatPage from './pages/ChatPage'
import DiscoverPage from './pages/DiscoverPage'

export default function App() {
  const [page, setPage]               = useState('upload')
  const [currentPaper, setCurrentPaper] = useState(null)

  const handlePaperReady = (paper) => {
    setCurrentPaper(paper)
    setPage('chat')
  }

  if (page === 'chat') {
    return <ChatPage paper={currentPaper} onBack={() => setPage('upload')} />
  }
  if (page === 'discover') {
    return <DiscoverPage onPaperReady={handlePaperReady} onBack={() => setPage('upload')} />
  }
  return <UploadPage onPaperReady={handlePaperReady} onDiscover={() => setPage('discover')} />
}
