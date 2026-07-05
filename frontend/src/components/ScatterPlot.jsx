import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts'

const CAT_COLOR = { cs: '#3ec6ff', math: '#c58bff', food: '#ffb15c', sports: '#5fe6a5', doc: '#5fe6a5' }

export default function ScatterPlot({ points, hitIds, queryPoint }) {
  const data = points.map((p) => ({
    x: p.x,
    y: p.y,
    id: p.item.id,
    metadata: p.item.metadata,
    category: p.item.category,
    isHit: hitIds.has(p.item.id),
  }))

  return (
    <div className="relative h-full w-full rounded-xl border border-line bg-panel/60">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
          <XAxis type="number" dataKey="x" hide />
          <YAxis type="number" dataKey="y" hide />
          <ZAxis range={[60, 200]} />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            contentStyle={{ background: '#11141d', border: '1px solid #1e2230', borderRadius: 8, fontSize: 12 }}
            formatter={(_, __, entry) => [entry.payload.metadata, entry.payload.category]}
          />
          <Scatter data={data} shape="circle">
            {data.map((d, i) => (
              <Cell
                key={i}
                fill={CAT_COLOR[d.category] || '#94a3b8'}
                opacity={d.isHit ? 1 : 0.55}
                r={d.isHit ? 8 : 5}
              />
            ))}
          </Scatter>
          {queryPoint && (
            <Scatter data={[{ x: queryPoint.x, y: queryPoint.y, metadata: 'query', category: '★' }]} shape="star">
              <Cell fill="#ffffff" />
            </Scatter>
          )}
        </ScatterChart>
      </ResponsiveContainer>
      <div className="pointer-events-none absolute left-3 top-3 text-[10px] uppercase tracking-widest text-slate-500">
        2D PCA projection
      </div>
    </div>
  )
}
