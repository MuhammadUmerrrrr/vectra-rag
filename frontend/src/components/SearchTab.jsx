import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts'
import { Trash2 } from 'lucide-react'

const CAT_COLOR = { cs: '#3ec6ff', math: '#c58bff', food: '#ffb15c', sports: '#5fe6a5', doc: '#5fe6a5' }

export default function SearchTab({ latency, algo, metric, k, results, onDelete, benchmark, hnswInfo }) {
  return (
    <div className="flex flex-col gap-5 overflow-y-auto p-5">
      <section>
        <Label>Search latency</Label>
        <div className="text-3xl font-semibold text-cs">{latency ? fmt(latency) : '—'}</div>
        <div className="mt-1 text-[11px] text-slate-500">
          {latency ? `${algo.toUpperCase()} · ${metric} · k=${k}` : 'No query yet'}
        </div>
      </section>

      <section>
        <Label>Top matches</Label>
        <div className="flex flex-col gap-2">
          {results.length === 0 && <Empty>Run a search to see results…</Empty>}
          {results.map((r, i) => (
            <div key={r.id} className="card">
              <div className="mb-1 text-[10px] uppercase tracking-widest text-slate-500">#{i + 1} nearest</div>
              <div className="mb-2 text-[13px] text-slate-200">{r.metadata}</div>
              <div className="flex items-center justify-between">
                <span
                  className="rounded-full px-2 py-0.5 text-[10px] font-medium"
                  style={{ color: CAT_COLOR[r.category], background: `${CAT_COLOR[r.category]}22` }}
                >
                  {r.category.toUpperCase()}
                </span>
                <span className="text-[10px] text-slate-500">dist: {r.distance.toFixed(5)}</span>
                <button onClick={() => onDelete(r.id)} className="btn-ghost">
                  <Trash2 size={11} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </section>

      {benchmark && (
        <section>
          <Label>Algorithm comparison</Label>
          <div className="h-40">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={[
                  { name: 'Brute Force', us: benchmark.bruteforceUs, color: '#fb7185' },
                  { name: 'KD-Tree', us: benchmark.kdtreeUs, color: '#38bdf8' },
                  { name: 'HNSW', us: benchmark.hnswUs, color: '#c58bff' },
                ]}
                layout="vertical"
                margin={{ left: 10, right: 20 }}
              >
                <XAxis type="number" hide />
                <YAxis type="category" dataKey="name" width={80} tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: '#11141d', border: '1px solid #1e2230', borderRadius: 8, fontSize: 12 }}
                  formatter={(v) => fmt(v)}
                />
                <Bar dataKey="us" radius={[0, 6, 6, 0]}>
                  {[0, 1, 2].map((i) => (
                    <Cell key={i} fill={['#fb7185', '#38bdf8', '#c58bff'][i]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      {hnswInfo && (
        <section>
          <Label>HNSW graph layers</Label>
          <div className="flex flex-col gap-1.5">
            {hnswInfo.nodesPerLayer.map((cnt, i) => {
              const max = hnswInfo.nodesPerLayer[0] || 1
              const pct = Math.max((cnt / max) * 100, 3)
              return (
                <div key={i} className="flex items-center gap-2 text-[10px]">
                  <span className="w-6 text-right text-brand">L{i}</span>
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink">
                    <div className="h-full rounded-full bg-brand/70" style={{ width: `${pct}%` }} />
                  </div>
                  <span className="w-20 text-right text-slate-500">
                    {cnt}n · {hnswInfo.edgesPerLayer[i] || 0}e
                  </span>
                </div>
              )
            })}
          </div>
        </section>
      )}
    </div>
  )
}

function Label({ children }) {
  return <div className="mb-2 text-[10px] font-semibold uppercase tracking-widest text-slate-500">{children}</div>
}
function Empty({ children }) {
  return <div className="text-[11px] text-slate-500">{children}</div>
}
function fmt(us) {
  return us < 1000 ? `${us} μs` : `${(us / 1000).toFixed(2)} ms`
}
