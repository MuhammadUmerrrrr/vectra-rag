# Vectra — Vector Search & RAG Playground

A from-scratch vector database with three nearest-neighbour algorithms
(Brute Force, KD-Tree, HNSW), plus a document-grounded RAG chat powered
by a local LLM through Ollama.

Rebuilt in **Python (FastAPI)** + **React (Vite + Tailwind + Recharts)**
so every line is something you can explain and defend.

## Why these algorithms

| Algorithm  | Complexity      | Use here                                   |
|------------|-----------------|---------------------------------------------|
| Brute Force| O(n)            | Ground truth / baseline for benchmarking     |
| KD-Tree    | O(log n) avg     | Classic spatial index, degrades in high-D    |
| HNSW       | O(log n) approx  | What real vector DBs (Qdrant, Weaviate, pgvector) use for ANN search |

`backend/vector_index.py` implements all three from scratch: distance
metrics (cosine / euclidean / manhattan), a k-d tree with backtracking
search, and a layered HNSW graph (random level assignment, greedy
layer-descent search, neighbor pruning) — see Malkov & Yashunin, 2016.

## Architecture

```
frontend (React)  <--REST-->  backend (FastAPI)
                                  ├── VectorDB        16D demo vectors (BruteForce/KD-Tree/HNSW)
                                  ├── DocumentDB       real embeddings, HNSW index
                                  └── OllamaClient     nomic-embed-text + llama3.2 (local LLM)
```

- **Demo search tab**: a simple keyword-bucket "embedding" (16D, 4
  categories) so search results are instant and explainable without a
  GPU. PCA-projected to 2D on the frontend for the scatter plot.
- **Documents tab**: real text is chunked (250 words, 30-word overlap)
  and embedded via Ollama's `nomic-embed-text`, stored in a second HNSW
  index.
- **Ask AI tab**: full RAG pipeline — embed the question, retrieve top-k
  chunks by cosine similarity, stuff them into a prompt, generate with
  `llama3.2`.

## Run it

### 1. Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

### 2. Ollama (for the Documents / Ask AI tabs)
```bash
# https://ollama.com
ollama pull nomic-embed-text
ollama pull llama3.2
ollama serve
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 — the Vite dev server proxies `/api/*` to
the FastAPI backend on port 8080 (see `vite.config.js`).

## What to say about it on your resume / in an interview

- Implemented HNSW (the ANN algorithm behind production vector DBs)
  from scratch in Python: layer assignment, greedy search, neighbor
  pruning.
- Benchmarked it against brute force and k-d tree to show the
  logarithmic vs. linear search-time tradeoff as data grows.
- Built a full RAG pipeline (chunk → embed → retrieve → generate)
  against a local LLM via Ollama, with visual retrieval feedback
  (which chunks fed the answer, and why).
- Clean separation: FastAPI for the algorithmic core, React for an
  interactive, chart-driven UI (Recharts for benchmarks/scatter,
  Tailwind for styling).
