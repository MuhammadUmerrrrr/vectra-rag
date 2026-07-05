import { useEffect, useState, useCallback } from 'react'
import { Boxes } from 'lucide-react'
import { api, textToEmbedding, pca2D } from './api'
import Sidebar from './components/Sidebar'
import ScatterPlot from './components/ScatterPlot'
import SearchTabReal from './components/SearchTab'
import DocumentsTab from './components/DocumentsTab'
import ChatTab from './components/ChatTab'

const TABS = [
  { id: 'search', label: 'Search' },
  { id: 'docs', label: 'Documents' },
  { id: 'chat', label: 'Ask AI' },
]

export default function App() {
  // demo vector state
  const [items, setItems] = useState([])
  const [pcaPoints, setPcaPoints] = useState([])
  const [hitIds, setHitIds] = useState(new Set())
  const [queryPoint, setQueryPoint] = useState(null)

  // controls
  const [query, setQuery] = useState('')
  const [algo, setAlgo] = useState('hnsw')
  const [metric, setMetric] = useState('cosine')
  const [k, setK] = useState(5)
  const [addMeta, setAddMeta] = useState('')
  const [addCat, setAddCat] = useState('cs')

  // results
  const [results, setResults] = useState([])
  const [latency, setLatency] = useState(null)
  const [benchmark, setBenchmark] = useState(null)
  const [hnswInfo, setHnswInfo] = useState(null)

  // docs + status
  const [status, setStatus] = useState(null)
  const [docs, setDocs] = useState([])
  const [tab, setTab] = useState('search')

  const refreshItems = useCallback(async () => {
    const data = await api.items()
    setItems(data)
    if (data.length >= 2) {
      const coords = pca2D(data.map((v) => v.embedding))
      setPcaPoints(data.map((item, i) => ({ x: coords[i][0], y: coords[i][1], item })))
    }
  }, [])

  const refreshHnsw = useCallback(async () => setHnswInfo(await api.hnswInfo()), [])
  const refreshStatus = useCallback(async () => setStatus(await api.status()), [])
  const refreshDocs = useCallback(async () => setDocs(await api.docList()), [])

  useEffect(() => {
    refreshItems()
    refreshHnsw()
    refreshStatus()
  }, [refreshItems, refreshHnsw, refreshStatus])

  useEffect(() => {
    if (tab === 'docs') refreshDocs()
  }, [tab, refreshDocs])

  async function handleSearch() {
    if (!query.trim()) return
    const emb = textToEmbedding(query)
    const data = await api.search(emb, k, metric, algo)
    setResults(data.results || [])
    setLatency(data.latencyUs || 0)
    setHitIds(new Set((data.results || []).map((r) => r.id)))

    // Position the query "star" near the weighted centroid of its top-3 hits.
    let sx = 0, sy = 0, sw = 0
    ;(data.results || []).slice(0, 3).forEach((r, i) => {
      const pt = pcaPoints.find((p) => p.item.id === r.id)
      if (pt) { const w = 1 / (i + 1); sx += pt.x * w; sy += pt.y * w; sw += w }
    })
    if (sw > 0) setQueryPoint({ x: sx / sw, y: sy / sw })
  }

  async function handleInsert() {
    if (!addMeta.trim()) return
    const emb = textToEmbedding(`${addMeta} ${addCat}`)
    await api.insert(addMeta, addCat, emb)
    setAddMeta('')
    await refreshItems()
    await refreshHnsw()
  }

  async function handleDelete(id) {
    await api.remove(id)
    setResults((r) => r.filter((x) => x.id !== id))
    await refreshItems()
    await refreshHnsw()
  }

  async function handleBenchmark() {
    const emb = textToEmbedding(query || 'binary tree algorithm')
    setBenchmark(await api.benchmark(emb, k, metric))
  }

  async function handleDocInsert(title, text) {
    const res = await api.docInsert(title, text)
    await refreshDocs()
    await refreshStatus()
    return res
  }

  async function handleDocDelete(id) {
    await api.docDelete(id)
    await refreshDocs()
    await refreshStatus()
  }

  async function handleAsk(question, k) {
    return api.docAsk(question, k)
  }

  return (
    <div className="flex h-screen flex-col bg-ink">
      <header className="flex items-center gap-3 border-b border-line bg-panel/60 px-5 py-3">
        <Boxes size={18} className="text-brand" />
        <h1 className="bg-gradient-to-r from-cs via-brand to-math bg-clip-text text-sm font-bold tracking-widest text-transparent">
          VECTRA
        </h1>
        <span className="rounded-full border border-brand/40 bg-brand/10 px-2 py-0.5 text-[10px] text-brand">HNSW</span>
        <span className="rounded-full border border-line px-2 py-0.5 text-[10px] text-slate-500">KD-TREE</span>
        <span className="rounded-full border border-line px-2 py-0.5 text-[10px] text-slate-500">BRUTE FORCE</span>
        <span
          className={`rounded-full border px-2 py-0.5 text-[10px] ${
            status?.ollamaAvailable ? 'border-sport/40 text-sport' : 'border-rose-500/40 text-rose-400'
          }`}
        >
          {status ? (status.ollamaAvailable ? 'OLLAMA ✓' : 'OLLAMA ✗') : 'OLLAMA…'}
        </span>
        <span className="ml-auto text-[11px] text-slate-500">{items.length} vectors · 16 dims</span>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          query={query} setQuery={setQuery} onSearch={handleSearch}
          algo={algo} setAlgo={setAlgo} metric={metric} setMetric={setMetric} k={k} setK={setK}
          addMeta={addMeta} setAddMeta={setAddMeta} addCat={addCat} setAddCat={setAddCat} onInsert={handleInsert}
          onBenchmark={handleBenchmark}
        />

        <main className="flex-1 p-4">
          <ScatterPlot points={pcaPoints} hitIds={hitIds} queryPoint={queryPoint} />
        </main>

        <aside className="flex w-96 flex-shrink-0 flex-col border-l border-line bg-panel/60">
          <div className="flex border-b border-line">
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`flex-1 py-2.5 text-[11px] font-medium tracking-wide transition ${
                  tab === t.id ? 'border-b-2 border-brand text-brand' : 'text-slate-500 hover:text-slate-300'
                }`}
              >
                {t.label.toUpperCase()}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-hidden">
            {tab === 'search' && (
              <SearchTabReal
                latency={latency} algo={algo} metric={metric} k={k}
                results={results} onDelete={handleDelete} benchmark={benchmark} hnswInfo={hnswInfo}
              />
            )}
            {tab === 'docs' && (
              <DocumentsTab status={status} docs={docs} onInsert={handleDocInsert} onDelete={handleDocDelete} />
            )}
            {tab === 'chat' && <ChatTab onAsk={handleAsk} />}
          </div>
        </aside>
      </div>
    </div>
  )
}
