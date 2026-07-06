import { useState } from 'react'
import { Sparkles, Trash2, CheckCircle2, XCircle } from 'lucide-react'

export default function DocumentsTab({ status, docs, onInsert, onDelete }) {
  const [title, setTitle] = useState('')
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState('')

  async function handleInsert() {
    if (!title.trim() || !text.trim()) {
      setMsg('⚠ Need both a title and text.')
      return
    }
    setBusy(true)
    setMsg('Calling Hugging Face Embeddings…')
    try {
      const res = await onInsert(title, text)
      if (res.error) {
        setMsg(res.error)
      } else {
        setMsg(`✓ Inserted ${res.chunks} chunk(s) · ${res.dims}D embeddings`)
        setTitle('')
        setText('')
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="flex flex-col gap-5 overflow-y-auto p-5">
      <section>
        <Label>AI Status</Label>
        <div className={`card flex items-start gap-2 ${status?.aiAvailable ? 'border-sport/30' : 'border-rose-500/30'}`}>
          {status?.aiAvailable ? (
            <CheckCircle2 size={16} className="mt-0.5 shrink-0 text-sport" />
          ) : (
            <XCircle size={16} className="mt-0.5 shrink-0 text-rose-400" />
          )}
          <div className="text-[11px] leading-relaxed text-slate-400">
            {status?.aiAvailable ? (
              <>
                <div className="text-sport">Connected</div>
                Embed: <span className="text-brand">Hugging Face Embeddings</span><br />
                Generate: <span className="text-brand">Groq Llama 3</span><br />
                Documents: <span className="text-slate-200">{status.docCount}</span>
              </>
            ) : (
              <>
                <div className="text-rose-400">Disconnected</div>
                Please set GROQ_API_KEY and HF_API_KEY environment variables.
              </>
            )}
          </div>
        </div>
      </section>

      <section>
        <Label>Insert document</Label>
        <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Document title / topic…" className="input" />
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste your notes, textbook excerpt, lecture content…&#10;Long text is automatically split into overlapping chunks and embedded with Hugging Face Embeddings."
          rows={6}
          className="input mt-2 resize-y"
        />
        <button onClick={handleInsert} disabled={busy} className="btn-secondary mt-2 disabled:opacity-50">
          <Sparkles size={14} /> {busy ? 'Embedding…' : 'Embed & insert'}
        </button>
        {msg && <div className="mt-1 text-[11px] text-slate-500">{msg}</div>}
      </section>

      <section>
        <Label>Stored documents ({docs.length})</Label>
        <div className="flex flex-col gap-2">
          {docs.length === 0 && <div className="text-[11px] text-slate-500">No documents yet.</div>}
          {docs.map((d) => (
            <div key={d.id} className="card">
              <div className="mb-1 text-[13px] font-medium text-sport">{d.title}</div>
              <div className="mb-2 text-[11px] leading-relaxed text-slate-500">{d.preview}</div>
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-slate-500">{d.words} words</span>
                <button onClick={() => onDelete(d.id)} className="btn-ghost">
                  <Trash2 size={11} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

function Label({ children }) {
  return <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-500">{children}</div>
}
