import { useState } from 'react'
import UploadPage from './pages/UploadPage'
import ChatPage from './pages/ChatPage'

export default function App() {
  const [currentPaper, setCurrentPaper] = useState(null)

  return (
    <div className="font-sans">
      {currentPaper
        ? <ChatPage paper={currentPaper} onBack={() => setCurrentPaper(null)} />
        : <UploadPage onPaperReady={setCurrentPaper} />
      }
    </div>
  )
}