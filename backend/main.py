"""
main.py
=======
FastAPI backend for the VectorDB + RAG demo.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8080

Endpoints mirror the original demo 1:1 so the frontend contract stays stable:
    GET  /items                demo vectors (16D)
    GET  /search                run knn search over demo vectors
    POST /insert                add a demo vector
    DELETE /delete/{id}         remove a demo vector
    GET  /benchmark              compare bruteforce vs kdtree vs hnsw timing
    GET  /hnsw-info              graph layer/edge stats for visualization
    GET  /status                  Ollama availability + doc count
    POST /doc/insert              chunk + embed + store a document (real embeddings)
    GET  /doc/list                 list stored documents
    DELETE /doc/delete/{id}       remove a document chunk
    POST /doc/search              retrieval only (for the map visualizer)
    POST /doc/ask                  full RAG: embed -> retrieve -> generate
"""
import itertools
import re
from typing import Literal

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from demo_data import load_demo
from ollama_client import OllamaClient
from vector_index import VectorDB, VectorItem, cosine, HNSW, BruteForce

DIMS = 16

app = FastAPI(title="VectorDB + RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = VectorDB(DIMS)
load_demo(db)
ollama = OllamaClient()

# --------------------------------------------------------------------------- #
# Demo text -> 16D embedding (same keyword-bucket trick as the original demo:
# 4 categories x 4 dims each; this is intentionally simple/explainable, not a
# real embedding model -- the real embeddings live in the /doc/* RAG path)
# --------------------------------------------------------------------------- #
KEYWORDS = {
    "cs": ["algorithm", "data", "tree", "graph", "array", "linked", "hash", "stack", "queue", "sort",
           "binary", "dynamic", "programming", "recursion", "complexity", "pointer", "node", "search",
           "insert", "bfs", "dfs", "heap", "trie"],
    "math": ["calculus", "matrix", "probability", "theorem", "integral", "derivative", "linear", "algebra",
             "equation", "function", "prime", "modular", "combinatorics", "permutation", "eigenvalue",
             "statistics", "proof"],
    "food": ["food", "pizza", "sushi", "ramen", "pasta", "recipe", "cook", "eat", "restaurant", "dish",
             "ingredient", "flavor", "spice", "noodle", "bread", "croissant", "taco", "fish", "rice", "soup"],
    "sports": ["sport", "basketball", "football", "tennis", "chess", "swim", "game", "play", "score",
               "team", "athlete", "competition", "match", "tournament", "olympic", "dribble", "tackle", "serve"],
}
CATEGORY_ORDER = ["cs", "math", "food", "sports"]


def text_to_embedding(text: str) -> list[float]:
    words = text.lower().split()
    scores = {c: 0.0 for c in CATEGORY_ORDER}
    for w in words:
        for cat, kws in KEYWORDS.items():
            if any(w in kw or kw.startswith(w) for kw in kws):
                scores[cat] += 0.35
                break
    mx = max(max(scores.values()), 0.01)
    emb = [0.08] * DIMS
    for i, cat in enumerate(CATEGORY_ORDER):
        base = min(scores[cat] / mx * 0.88, 0.94)
        if scores[cat] < 0.01:
            continue
        off = i * 4
        emb[off] = max(0.05, base)
        emb[off + 1] = max(0.05, base)
        emb[off + 2] = max(0.05, base * 0.92)
        emb[off + 3] = max(0.05, base * 0.87)
    return emb


# --------------------------------------------------------------------------- #
# Request bodies
# --------------------------------------------------------------------------- #
class InsertBody(BaseModel):
    metadata: str
    category: str
    embedding: list[float]


class DocInsertBody(BaseModel):
    title: str
    text: str


class DocSearchBody(BaseModel):
    question: str
    k: int = 3


class DocAskBody(BaseModel):
    question: str
    k: int = 3


# --------------------------------------------------------------------------- #
# Demo vector endpoints
# --------------------------------------------------------------------------- #
@app.get("/items")
def get_items():
    return db.all_items()


@app.get("/search")
def search(v: str, k: int = 5, metric: str = "cosine", algo: str = "hnsw"):
    query = [float(x) for x in v.split(",") if x]
    if len(query) != DIMS:
        return {"error": f"need {DIMS}D vector"}
    return db.search(query, k, metric, algo)


@app.post("/insert")
def insert_vector(body: InsertBody):
    if len(body.embedding) != DIMS:
        return {"error": "invalid body"}
    item_id = db.insert(body.metadata, body.category, body.embedding)
    return {"id": item_id}


@app.delete("/delete/{item_id}")
def delete_vector(item_id: int):
    return {"ok": db.remove(item_id)}


@app.get("/benchmark")
def benchmark(v: str, k: int = 5, metric: str = "cosine"):
    query = [float(x) for x in v.split(",") if x]
    if len(query) != DIMS:
        return {"error": f"need {DIMS}D vector"}
    return db.benchmark(query, k, metric)


@app.get("/hnsw-info")
def hnsw_info():
    return db.hnsw.graph_info()


@app.get("/stats")
def stats():
    return {
        "count": len(db.store),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    }


# --------------------------------------------------------------------------- #
# Document + RAG (real embeddings, via Ollama)
# --------------------------------------------------------------------------- #
class DocItem:
    __slots__ = ("id", "title", "text", "embedding")

    def __init__(self, id_, title, text, embedding):
        self.id, self.title, self.text, self.embedding = id_, title, text, embedding


class DocumentDB:
    def __init__(self):
        self.store: dict[int, DocItem] = {}
        self.hnsw = HNSW()
        self.bf = BruteForce()
        self._next_id = itertools.count(1)
        self.dims = 0

    def insert(self, title: str, text: str, embedding: list[float]) -> int:
        if not self.dims:
            self.dims = len(embedding)
        doc_id = next(self._next_id)
        emb = np.array(embedding, dtype=np.float32)
        item = DocItem(doc_id, title, text, emb)
        self.store[doc_id] = item
        vi = VectorItem(doc_id, title, "doc", emb)
        self.hnsw.insert(vi, cosine)
        self.bf.insert(vi)
        return doc_id

    def search(self, query: list[float], k: int, max_dist: float = 0.7):
        if not self.store:
            return []
        q = np.array(query, dtype=np.float32)
        raw = self.bf.knn(q, k, cosine) if len(self.store) < 10 else self.hnsw.knn(q, k, 50, cosine)
        return [(d, self.store[i]) for d, i in raw if i in self.store and d <= max_dist]

    def remove(self, doc_id: int) -> bool:
        if doc_id not in self.store:
            return False
        del self.store[doc_id]
        self.hnsw.remove(doc_id)
        self.bf.remove(doc_id)
        return True

    def all(self):
        return list(self.store.values())


doc_db = DocumentDB()


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    words = text.split()
    if len(words) <= chunk_words:
        return [text] if words else []
    step = chunk_words - overlap_words
    chunks = []
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
    return chunks


@app.get("/status")
def status():
    up = ollama.is_available()
    return {
        "ollamaAvailable": up,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": len(doc_db.store),
        "docDims": doc_db.dims,
        "demoDims": DIMS,
        "demoCount": len(db.store),
    }


@app.post("/doc/insert")
def doc_insert(body: DocInsertBody):
    if not body.title or not body.text:
        return {"error": "need title and text"}

    chunks = chunk_text(body.text)
    ids = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if emb is None:
            return {
                "error": "Ollama unavailable. Install from https://ollama.com then run: "
                         "ollama pull nomic-embed-text && ollama pull llama3.2"
            }
        title = f"{body.title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else body.title
        ids.append(doc_db.insert(title, chunk, emb))

    return {"ids": ids, "chunks": len(chunks), "dims": doc_db.dims}


@app.delete("/doc/delete/{doc_id}")
def doc_delete(doc_id: int):
    return {"ok": doc_db.remove(doc_id)}


@app.get("/doc/list")
def doc_list():
    out = []
    for d in doc_db.all():
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        out.append({"id": d.id, "title": d.title, "preview": preview, "words": len(d.text.split())})
    return out


@app.post("/doc/search")
def doc_search(body: DocSearchBody):
    q_emb = ollama.embed(body.question)
    if q_emb is None:
        return {"error": "Ollama unavailable"}
    hits = doc_db.search(q_emb, body.k)
    return {"contexts": [{"id": d.id, "title": d.title, "distance": dist} for dist, d in hits]}


RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question directly. "
    "Use the provided context if it contains relevant information. "
    "If it doesn't, just use your own general knowledge. "
    "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
    "'the context doesn't mention'. Just answer the question naturally."
)


@app.post("/doc/ask")
def doc_ask(body: DocAskBody):
    q_emb = ollama.embed(body.question)
    if q_emb is None:
        return {"error": "Ollama unavailable"}

    hits = doc_db.search(q_emb, body.k)
    ctx_block = "\n\n".join(f"[{i + 1}] {d.title}:\n{d.text}" for i, (_, d) in enumerate(hits))
    prompt = f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{ctx_block}\n\nQuestion: {body.question}\n\nAnswer:"

    answer = ollama.generate(prompt)
    return {
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": [{"id": d.id, "title": d.title, "text": d.text, "distance": dist} for dist, d in hits],
        "docCount": len(doc_db.store),
    }
