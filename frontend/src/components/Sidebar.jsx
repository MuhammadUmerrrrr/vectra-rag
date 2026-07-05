import { Zap, Plus, BarChart3 } from 'lucide-react'

const ALGOS = [
  { id: 'hnsw', label: 'HNSW' },
  { id: 'kdtree', label: 'KD-Tree' },
  { id: 'bruteforce', label: 'Brute Force' },
]
const CATS = [
  { id: 'cs', label: 'CS / Algorithms', dot: 'bg-cs' },
  { id: 'math', label: 'Mathematics', dot: 'bg-math' },
  { id: 'food', label: 'Food & Cooking', dot: 'bg-food' },
  { id: 'sports', label: 'Sports & Games', dot: 'bg-sport' },
]

export default function Sidebar({
  query, setQuery, onSearch,
  algo, setAlgo, metric, setMetric, k, setK,
  addMeta, setAddMeta, addCat, setAddCat, onInsert,
  onBenchmark,
}) {
  return (
    <aside className="flex w-72 flex-shrink-0 flex-col gap-6 overflow-y-auto border-r border-line bg-panel/60 p-5">
      <section>
        <Label>Query (demo vectors)</Label>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && onSearch()}
          placeholder="binary tree, sushi, basketball…"
          className="input"
        />
        <button onClick={onSearch} className="btn-primary mt-2">
          <Zap size={14} /> Search
        </button>
      </section>

      <section>
        <Label>Algorithm</Label>
        <div className="flex gap-1.5">
          {ALGOS.map((a) => (
            <button
              key={a.id}
              onClick={() => setAlgo(a.id)}
              className={`flex-1 rounded-lg border px-2 py-1.5 text-[11px] font-medium transition ${
                algo === a.id ? 'border-brand/50 bg-brand/10 text-brand' : 'border-line text-slate-500 hover:text-slate-300'
              }`}
            >
              {a.label}
            </button>
          ))}
        </div>
      </section>

      <section>
        <Label>Distance metric</Label>
        <select value={metric} onChange={(e) => setMetric(e.target.value)} className="input">
          <option value="cosine">Cosine Similarity</option>
          <option value="euclidean">Euclidean Distance</option>
          <option value="manhattan">Manhattan Distance</option>
        </select>
      </section>

      <section>
        <Label>Top-K: <span className="text-brand">{k}</span></Label>
        <input type="range" min={1} max={10} value={k} onChange={(e) => setK(+e.target.value)} className="w-full accent-brand" />
      </section>

      <section>
        <Label>Category legend</Label>
        <div className="flex flex-col gap-1.5 text-[11px] text-slate-400">
          {CATS.map((c) => (
            <div key={c.id} className="flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${c.dot}`} /> {c.label}
            </div>
          ))}
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-doc" /> Documents (RAG)
          </div>
        </div>
      </section>

      <section>
        <Label>Insert demo vector</Label>
        <input
          value={addMeta}
          onChange={(e) => setAddMeta(e.target.value)}
          placeholder="Description…"
          className="input"
        />
        <select value={addCat} onChange={(e) => setAddCat(e.target.value)} className="input mt-2">
          {CATS.map((c) => (
            <option key={c.id} value={c.id}>{c.label}</option>
          ))}
        </select>
        <button onClick={onInsert} className="btn-secondary mt-2">
          <Plus size={14} /> Insert
        </button>
      </section>

      <section>
        <Label>Benchmark</Label>
        <button onClick={onBenchmark} className="btn-secondary">
          <BarChart3 size={14} /> Compare all algorithms
        </button>
      </section>
    </aside>
  )
}

function Label({ children }) {
  return <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-500">{children}</div>
}
