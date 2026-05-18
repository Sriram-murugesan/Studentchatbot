import { useState, useRef } from "react"
import ReactMarkdown from "react-markdown"

export default function App() {
  const [sessionId, setSessionId] = useState(null)
  const [chunks, setChunks] = useState(null)
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState("")
  const [uploading, setUploading] = useState(false)
  const [asking, setAsking] = useState(false)
  const [error, setError] = useState(null)
  const fileRef = useRef(null)
  const bottomRef = useRef(null)

  const handleUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return

    if (!file.name.endsWith(".pdf")) {
      setError("Only PDF files allowed")
      return
    }
    if (file.size > 50 * 1024 * 1024) {
      setError("File too large. Max 50MB")
      return
    }

    setError(null)
    setUploading(true)
    setMessages([])
    setSessionId(null)

    const formData = new FormData()
    formData.append("file", file)

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      })
      const data = await res.json()

      if (!res.ok) throw new Error(data.detail)

      setSessionId(data.session_id)
      setChunks(data.chunks)
    } catch (err) {
      setError(err.message || "Upload failed. Is the backend running?")
    } finally {
      setUploading(false)
    }
  }

  const handleAsk = async () => {
    if (!question.trim() || !sessionId) return

    const userMessage = { role: "user", content: question }
    setMessages(prev => [...prev, userMessage])
    setQuestion("")
    setAsking(true)
    setError(null)

    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, question }),
      })
      const data = await res.json()

      if (!res.ok) throw new Error(data.detail)

      setMessages(prev => [...prev, { role: "bot", content: data.answer }])
    } catch (err) {
      setError(err.message || "Something went wrong")
    } finally {
      setAsking(false)
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 100)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleAsk()
    }
  }

  const handleClear = () => {
    setMessages([])
    setSessionId(null)
    setChunks(null)
    setError(null)
    if (fileRef.current) fileRef.current.value = ""
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white flex flex-col items-center py-10 px-4">

      {/* Header */}
      <div className="w-full max-w-2xl mb-8">
        <h1 className="text-3xl font-bold text-white">📚 StudyBot</h1>
        <p className="text-gray-400 mt-1 text-sm">Upload your textbook and ask anything</p>
      </div>

      {/* Upload Section */}
      <div className="w-full max-w-2xl bg-gray-900 rounded-2xl p-6 mb-6 border border-gray-800">
        <p className="text-sm font-medium text-gray-300 mb-3">Upload your PDF</p>

        <div className="flex items-center gap-3">
          <label className="flex-1 cursor-pointer border-2 border-dashed border-gray-700 rounded-xl p-4 text-center hover:border-blue-500 transition">
            <input
              ref={fileRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={handleUpload}
              disabled={uploading}
            />
            <span className="text-gray-400 text-sm">
              {uploading ? "Processing..." : "Click to upload PDF (max 50MB)"}
            </span>
          </label>

          {sessionId && (
            <button
              onClick={handleClear}
              className="text-sm text-red-400 hover:text-red-300 border border-red-800 rounded-xl px-4 py-3 transition"
            >
              Clear
            </button>
          )}
        </div>

        {/* Success state */}
        {sessionId && (
          <div className="mt-3 flex items-center gap-2 text-green-400 text-sm">
            <span>✓</span>
            <span>Ready — {chunks} chunks indexed. Start asking questions.</span>
          </div>
        )}

        {/* Error state */}
        {error && (
          <div className="mt-3 text-red-400 text-sm">
            ⚠ {error}
          </div>
        )}
      </div>

      {/* Chat Section */}
      {sessionId && (
        <div className="w-full max-w-2xl flex flex-col bg-gray-900 rounded-2xl border border-gray-800 overflow-hidden">

          {/* Messages */}
          <div className="flex-1 p-6 space-y-4 max-h-96 overflow-y-auto">
            {messages.length === 0 && (
              <p className="text-gray-600 text-sm text-center">Ask your first question...</p>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`max-w-sm px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-blue-600 text-white rounded-br-sm"
                    : "bg-gray-800 text-gray-100 rounded-bl-sm"
                }`}>
                  {msg.role === "user" ? (
                    msg.content
                  ) : (
                    <ReactMarkdown
                      components={{
                        h1: ({node, ...props}) => <h1 className="text-base font-bold text-white mt-3 mb-1" {...props} />,
                        h2: ({node, ...props}) => <h2 className="text-sm font-bold text-blue-300 mt-3 mb-1" {...props} />,
                        h3: ({node, ...props}) => <h3 className="text-sm font-semibold text-blue-200 mt-2 mb-1" {...props} />,
                        p: ({node, ...props}) => <p className="mb-2 leading-relaxed" {...props} />,
                        strong: ({node, ...props}) => <strong className="text-white font-semibold" {...props} />,
                        ul: ({node, ...props}) => <ul className="list-disc list-inside space-y-1 mb-2" {...props} />,
                        ol: ({node, ...props}) => <ol className="list-decimal list-inside space-y-1 mb-2" {...props} />,
                        li: ({node, ...props}) => <li className="text-gray-200" {...props} />,
                        hr: ({node, ...props}) => <hr className="border-gray-600 my-3" {...props} />,
                        code: ({node, ...props}) => <code className="bg-gray-900 text-green-400 px-1 rounded text-xs" {...props} />,
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                  )}
                </div>
              </div>
            ))}

            {/* Loading bubble */}
            {asking && (
              <div className="flex justify-start">
                <div className="bg-gray-800 px-4 py-3 rounded-2xl rounded-bl-sm">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay:"0ms"}}></span>
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay:"150ms"}}></span>
                    <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay:"300ms"}}></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          {/* Input */}
          <div className="border-t border-gray-800 p-4 flex gap-3">
            <input
              type="text"
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question about your book..."
              disabled={asking}
              className="flex-1 bg-gray-800 text-white text-sm rounded-xl px-4 py-3 outline-none border border-gray-700 focus:border-blue-500 transition placeholder-gray-500"
            />
            <button
              onClick={handleAsk}
              disabled={asking || !question.trim()}
              className="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed text-white text-sm font-medium px-5 py-3 rounded-xl transition"
            >
              Send
            </button>
          </div>
        </div>
      )}
    </div>
  )
}