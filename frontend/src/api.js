const BASE = '/api'

async function j(res) {
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export const api = {
  items: () => fetch(`${BASE}/items`).then(j),
  search: (v, k, metric, algo) =>
    fetch(`${BASE}/search?v=${v.join(',')}&k=${k}&metric=${metric}&algo=${algo}`).then(j),
  insert: (metadata, category, embedding) =>
    fetch(`${BASE}/insert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ metadata, category, embedding }),
    }).then(j),
  remove: (id) => fetch(`${BASE}/delete/${id}`, { method: 'DELETE' }).then(j),
  benchmark: (v, k, metric) =>
    fetch(`${BASE}/benchmark?v=${v.join(',')}&k=${k}&metric=${metric}`).then(j),
  hnswInfo: () => fetch(`${BASE}/hnsw-info`).then(j),
  status: () => fetch(`${BASE}/status`).then(j),
  docInsert: (title, text) =>
    fetch(`${BASE}/doc/insert`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, text }),
    }).then(j),
  docList: () => fetch(`${BASE}/doc/list`).then(j),
  docDelete: (id) => fetch(`${BASE}/doc/delete/${id}`, { method: 'DELETE' }).then(j),
  docSearch: (question, k) =>
    fetch(`${BASE}/doc/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, k }),
    }).then(j),
  docAsk: (question, k) =>
    fetch(`${BASE}/doc/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, k }),
    }).then(j),
}

// Same keyword-bucket embedding trick as the backend demo, kept on the client
// so search feels instant and the query dot can be plotted immediately.
const KEYWORDS = {
  cs: ['algorithm','data','tree','graph','array','linked','hash','stack','queue','sort','binary','dynamic','programming','recursion','complexity','pointer','node','search','insert','bfs','dfs','heap','trie'],
  math: ['calculus','matrix','probability','theorem','integral','derivative','linear','algebra','equation','function','prime','modular','combinatorics','permutation','eigenvalue','statistics','proof'],
  food: ['food','pizza','sushi','ramen','pasta','recipe','cook','eat','restaurant','dish','ingredient','flavor','spice','noodle','bread','croissant','taco','fish','rice','soup'],
  sports: ['sport','basketball','football','tennis','chess','swim','game','play','score','team','athlete','competition','match','tournament','olympic','dribble','tackle','serve'],
}
const ORDER = ['cs', 'math', 'food', 'sports']

export function textToEmbedding(text) {
  const words = text.toLowerCase().split(/\s+/)
  const scores = { cs: 0, math: 0, food: 0, sports: 0 }
  for (const w of words) {
    for (const cat of ORDER) {
      if (KEYWORDS[cat].some((kw) => w.includes(kw) || kw.startsWith(w))) {
        scores[cat] += 0.35
        break
      }
    }
  }
  const mx = Math.max(...Object.values(scores), 0.01)
  const emb = new Array(16).fill(0.08)
  ORDER.forEach((cat, i) => {
    if (scores[cat] < 0.01) return
    const base = Math.min((scores[cat] / mx) * 0.88, 0.94)
    const off = i * 4
    emb[off] = Math.max(0.05, base)
    emb[off + 1] = Math.max(0.05, base)
    emb[off + 2] = Math.max(0.05, base * 0.92)
    emb[off + 3] = Math.max(0.05, base * 0.87)
  })
  return emb
}

// Simple 2-component PCA via power iteration, used purely for the scatter plot.
export function pca2D(vectors) {
  const n = vectors.length
  if (n < 2) return vectors.map(() => [0, 0])
  const d = vectors[0].length
  const mean = new Array(d).fill(0)
  vectors.forEach((v) => v.forEach((val, i) => (mean[i] += val / n)))
  const X = vectors.map((v) => v.map((val, i) => val - mean[i]))

  function powerIter(excl) {
    let v = new Array(d).fill(0).map(() => Math.random() - 0.5)
    if (excl) {
      const dot = v.reduce((s, vi, i) => s + vi * excl[i], 0)
      v = v.map((vi, i) => vi - dot * excl[i])
    }
    let norm = Math.sqrt(v.reduce((s, vi) => s + vi * vi, 0))
    v = v.map((vi) => vi / norm)
    for (let it = 0; it < 200; it++) {
      const Xv = X.map((xi) => xi.reduce((s, xij, j) => s + xij * v[j], 0))
      const nv = new Array(d).fill(0)
      for (let k = 0; k < n; k++) for (let j = 0; j < d; j++) nv[j] += X[k][j] * Xv[k]
      if (excl) {
        const dot = nv.reduce((s, vi, i) => s + vi * excl[i], 0)
        for (let i = 0; i < d; i++) nv[i] -= dot * excl[i]
      }
      norm = Math.sqrt(nv.reduce((s, vi) => s + vi * vi, 0))
      if (norm < 1e-10) break
      v = nv.map((vi) => vi / norm)
    }
    return v
  }

  const pc1 = powerIter(null)
  const pc2 = powerIter(pc1)
  return X.map((x) => [
    x.reduce((s, v, i) => s + v * pc1[i], 0),
    x.reduce((s, v, i) => s + v * pc2[i], 0),
  ])
}
