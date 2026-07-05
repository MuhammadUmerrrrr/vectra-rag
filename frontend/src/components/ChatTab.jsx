import { useState, useRef, useEffect } from 'react'
import { Bot, Send, Loader2 } from 'lucide-react'

export default function ChatTab({ onAsk }) {
  const [question, setQuestion] = useState('')
  const [k, setK] = useState(3)
  const [busy, setBusy] = useState(false)
  const [thread, setThread] = useState(null) // { question, answer, contexts, model, error }
  const [openCtx, setOpenCtx] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), [thread, busy])

  async function handleAsk() {
    if (!question.trim()) return
    const q = question
    setThread({ question: q, loading: true })
    setQuestion('')
    setBusy(true)
    try {
      const res = await onAsk(q, k)
      setThread({ question: q, ...res })
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex h-full flex-col gap-4 overflow-y-auto p-5">
      <section>
        <Label>Ask a question</Label>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && e.ctrlKey && handleAsk()}
          placeholder="What is dynamic programming? How does HNSW work? …"
          rows={3}
          className="input resize-y"
        />
        <div className="mt-2 flex gap-2">
          <select value={k} onChange={(e) => setK(+e.target.value)} className="input w-24">
            <option value={2}>Top 2</option>
            <option value={3}>Top 3</option>
            <option value={5}>Top 5</option>
          </select>
          <button onClick={handleAsk} disabled={busy} className="btn-secondary flex-1 disabled:opacity-50">
            {busy ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
            Ask AI
          </button>
        </div>
        <div className="mt-1 text-[10px] text-slate-500">Uses your inserted documents as context. Answers come from the local LLM.</div>
      </section>

      <section className="flex-1">
        <Label>Conversation</Label>
        {!thread && <div className="text-[11px] text-slate-500">Ask a question about your inserted documents…</div>}

        {thread && (
          <div className="flex flex-col gap-3">
            <div className="rounded-lg border border-brand/25 bg-brand/10 p-3 text-[13px]">{thread.question}</div>

            {thread.loading && (
              <div className="flex items-center gap-2 text-[11px] text-slate-500">
                <Loader2 size={13} className="animate-spin" /> Retrieving context & generating answer…
              </div>
            )}

            {thread.error && (
              <div className="card border-rose-500/30 text-[12px] text-rose-300">{thread.error}</div>
            )}

            {thread.answer !== undefined && (
              <div className="card">
                <div className="mb-2 flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-sport">
                  <Bot size={12} /> {thread.model || 'llm'}
                </div>
                <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-slate-200">{thread.answer}</div>

                {thread.contexts?.length > 0 && (
                  <div className="mt-3 border-t border-line pt-2">
                    <div className="mb-1.5 text-[9px] uppercase tracking-widest text-slate-500">
                      Retrieved context ({thread.contexts.length} chunks)
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {thread.contexts.map((c, i) => (
                        <button
                          key={i}
                          onClick={() => setOpenCtx(openCtx === i ? null : i)}
                          className="rounded-full border border-brand/20 bg-brand/10 px-2 py-0.5 text-[9px] text-brand"
                        >
                          #{i + 1} {c.title} · {c.distance.toFixed(3)}
                        </button>
                      ))}
                    </div>
                    {openCtx !== null && thread.contexts[openCtx] && (
                      <div className="mt-2 rounded-md bg-ink p-2 text-[10px] leading-relaxed text-slate-500">
                        {thread.contexts[openCtx].text}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
        <div ref={bottomRef} />
      </section>
    </div>
  )
}

function Label({ children }) {
  return <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-500">{children}</div>
}
