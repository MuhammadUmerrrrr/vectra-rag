"""
vector_index.py
================
Three from-scratch nearest-neighbour search structures over N-dimensional
vectors, plus the distance metrics they share.

- BruteForce : O(n) linear scan. Baseline for correctness + benchmarking.
- KDTree     : binary space partition, good on low dimensions.
- HNSW       : Hierarchical Navigable Small World graph -- the same
               approximate-nearest-neighbour algorithm used by real vector
               databases (Qdrant, Weaviate, Milvus, pgvector's HNSW index).

This file has no framework dependencies -- it's plain Python + numpy so it
can be unit tested on its own.
"""
from __future__ import annotations

import heapq
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

Metric = Literal["cosine", "euclidean", "manhattan"]


# --------------------------------------------------------------------------- #
# Distance metrics
# --------------------------------------------------------------------------- #
def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


DIST_FNS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "cosine": cosine,
    "euclidean": euclidean,
    "manhattan": manhattan,
}


def get_dist_fn(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    return DIST_FNS.get(metric, cosine)


@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    embedding: np.ndarray


# --------------------------------------------------------------------------- #
# Brute force
# --------------------------------------------------------------------------- #
class BruteForce:
    def __init__(self) -> None:
        self.items: dict[int, VectorItem] = {}

    def insert(self, item: VectorItem) -> None:
        self.items[item.id] = item

    def remove(self, item_id: int) -> None:
        self.items.pop(item_id, None)

    def knn(self, query: np.ndarray, k: int, dist_fn) -> list[tuple[float, int]]:
        scored = [(dist_fn(query, it.embedding), it.id) for it in self.items.values()]
        scored.sort(key=lambda x: x[0])
        return scored[:k]


# --------------------------------------------------------------------------- #
# KD-Tree
# --------------------------------------------------------------------------- #
class _KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem):
        self.item = item
        self.left: "_KDNode | None" = None
        self.right: "_KDNode | None" = None


class KDTree:
    """Simple (unbalanced, insert-order-dependent) k-d tree. Rebuilt on delete."""

    def __init__(self, dims: int):
        self.dims = dims
        self.root: _KDNode | None = None

    def insert(self, item: VectorItem) -> None:
        self.root = self._insert(self.root, item, 0)

    def _insert(self, node, item, depth):
        if node is None:
            return _KDNode(item)
        axis = depth % self.dims
        if item.embedding[axis] < node.item.embedding[axis]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def rebuild(self, items: list[VectorItem]) -> None:
        self.root = None
        for it in items:
            self.insert(it)

    def knn(self, query: np.ndarray, k: int, dist_fn) -> list[tuple[float, int]]:
        # max-heap of size k, storing (-dist, id) so heap[0] is the farthest kept
        heap: list[tuple[float, int]] = []

        def visit(node, depth):
            if node is None:
                return
            d = dist_fn(query, node.item.embedding)
            if len(heap) < k:
                heapq.heappush(heap, (-d, node.item.id))
            elif d < -heap[0][0]:
                heapq.heapreplace(heap, (-d, node.item.id))

            axis = depth % self.dims
            diff = query[axis] - node.item.embedding[axis]
            closer, farther = (node.left, node.right) if diff < 0 else (node.right, node.left)
            visit(closer, depth + 1)
            if len(heap) < k or abs(diff) < -heap[0][0]:
                visit(farther, depth + 1)

        visit(self.root, 0)
        heap.sort(key=lambda x: -x[0])
        return [(-d, i) for d, i in heap]


# --------------------------------------------------------------------------- #
# HNSW -- Hierarchical Navigable Small World
# --------------------------------------------------------------------------- #
@dataclass
class _HNSWNode:
    item: VectorItem
    max_layer: int
    neighbors: list[list[int]] = field(default_factory=list)  # per-layer adjacency


class HNSW:
    """
    A didactic but functionally faithful HNSW implementation:
      - each inserted point is assigned a random top layer (exponential decay)
      - greedy search narrows down from the top layer to layer 0
      - at each layer, up to M neighbors are kept (M0 = 2*M at layer 0)
      - neighbor lists are pruned back to the closest M when they overflow

    This mirrors the structure described in Malkov & Yashunin (2016).
    """

    def __init__(self, m: int = 16, ef_construction: int = 200, seed: int = 42):
        self.M = m
        self.M0 = 2 * m
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(m)
        self.rng = random.Random(seed)

        self.nodes: dict[int, _HNSWNode] = {}
        self.entry_point: int | None = None
        self.top_layer = -1

    def _random_level(self) -> int:
        return int(math.floor(-math.log(self.rng.random()) * self.mL))

    def _search_layer(self, query, entry_id, ef, layer, dist_fn):
        visited = {entry_id}
        d0 = dist_fn(query, self.nodes[entry_id].item.embedding)
        candidates = [(d0, entry_id)]
        found = [(-d0, entry_id)]  # max-heap via negation
        heapq.heapify(candidates)
        heapq.heapify(found)

        while candidates:
            cd, cid = heapq.heappop(candidates)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = self.nodes.get(cid)
            if node is None or layer >= len(node.neighbors):
                continue
            for nid in node.neighbors[layer]:
                if nid in visited or nid not in self.nodes:
                    continue
                visited.add(nid)
                nd = dist_fn(query, self.nodes[nid].item.embedding)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(candidates, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = sorted([(-d, i) for d, i in found])
        return result

    def insert(self, item: VectorItem, dist_fn) -> None:
        level = self._random_level()
        node = _HNSWNode(item=item, max_layer=level, neighbors=[[] for _ in range(level + 1)])
        self.nodes[item.id] = node

        if self.entry_point is None:
            self.entry_point = item.id
            self.top_layer = level
            return

        ep = self.entry_point
        for lc in range(self.top_layer, level, -1):
            if lc < len(self.nodes[ep].neighbors):
                w = self._search_layer(item.embedding, ep, 1, lc, dist_fn)
                if w:
                    ep = w[0][1]

        for lc in range(min(self.top_layer, level), -1, -1):
            w = self._search_layer(item.embedding, ep, self.ef_construction, lc, dist_fn)
            max_m = self.M0 if lc == 0 else self.M
            selected = [i for _, i in w[:max_m]]
            node.neighbors[lc] = selected

            for nid in selected:
                nnode = self.nodes.get(nid)
                if nnode is None:
                    continue
                if len(nnode.neighbors) <= lc:
                    nnode.neighbors.extend([[] for _ in range(lc - len(nnode.neighbors) + 1)])
                conn = nnode.neighbors[lc]
                conn.append(item.id)
                if len(conn) > max_m:
                    scored = sorted(
                        (dist_fn(nnode.item.embedding, self.nodes[c].item.embedding), c)
                        for c in conn
                        if c in self.nodes
                    )
                    nnode.neighbors[lc] = [c for _, c in scored[:max_m]]

            if w:
                ep = w[0][1]

        if level > self.top_layer:
            self.top_layer = level
            self.entry_point = item.id

    def knn(self, query, k, ef, dist_fn) -> list[tuple[float, int]]:
        if self.entry_point is None:
            return []
        ep = self.entry_point
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.nodes[ep].neighbors):
                w = self._search_layer(query, ep, 1, lc, dist_fn)
                if w:
                    ep = w[0][1]
        w = self._search_layer(query, ep, max(ef, k), 0, dist_fn)
        return w[:k]

    def remove(self, item_id: int) -> None:
        if item_id not in self.nodes:
            return
        for node in self.nodes.values():
            for layer in node.neighbors:
                if item_id in layer:
                    layer.remove(item_id)
        if self.entry_point == item_id:
            remaining = [i for i in self.nodes if i != item_id]
            self.entry_point = remaining[0] if remaining else None
        del self.nodes[item_id]

    def graph_info(self) -> dict:
        max_layer = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_layer
        edges_per_layer = [0] * max_layer
        nodes_out = []
        edges_out = []
        for nid, node in self.nodes.items():
            nodes_out.append(
                {"id": nid, "metadata": node.item.metadata, "category": node.item.category, "maxLyr": node.max_layer}
            )
            for lc in range(min(node.max_layer, max_layer - 1) + 1):
                nodes_per_layer[lc] += 1
                if lc < len(node.neighbors):
                    for nnid in node.neighbors[lc]:
                        if nid < nnid:
                            edges_per_layer[lc] += 1
                            edges_out.append({"src": nid, "dst": nnid, "lyr": lc})
        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.nodes),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes_out,
            "edges": edges_out,
        }


# --------------------------------------------------------------------------- #
# VectorDB -- ties the three structures + timing together
# --------------------------------------------------------------------------- #
class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self.store: dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW()
        self._next_id = 1

    def insert(self, metadata: str, category: str, embedding: list[float]) -> int:
        item = VectorItem(self._next_id, metadata, category, np.array(embedding, dtype=np.float32))
        self.store[item.id] = item
        self.bf.insert(item)
        self.kdt.insert(item)
        self.hnsw.insert(item, cosine)
        self._next_id += 1
        return item.id

    def remove(self, item_id: int) -> bool:
        if item_id not in self.store:
            return False
        del self.store[item_id]
        self.bf.remove(item_id)
        self.hnsw.remove(item_id)
        self.kdt.rebuild(list(self.store.values()))
        return True

    def search(self, query: list[float], k: int, metric: str, algo: str):
        q = np.array(query, dtype=np.float32)
        dist_fn = get_dist_fn(metric)

        t0 = time.perf_counter()
        if algo == "bruteforce":
            raw = self.bf.knn(q, k, dist_fn)
        elif algo == "kdtree":
            raw = self.kdt.knn(q, k, dist_fn)
        else:
            raw = self.hnsw.knn(q, k, 50, dist_fn)
        us = int((time.perf_counter() - t0) * 1_000_000)

        hits = []
        for dist, iid in raw:
            it = self.store.get(iid)
            if it:
                hits.append({"id": it.id, "metadata": it.metadata, "category": it.category,
                            "distance": dist, "embedding": it.embedding.tolist()})
        return {"results": hits, "latencyUs": us, "algo": algo, "metric": metric}

    def benchmark(self, query: list[float], k: int, metric: str):
        q = np.array(query, dtype=np.float32)
        dist_fn = get_dist_fn(metric)

        def timed(fn):
            t0 = time.perf_counter()
            fn()
            return int((time.perf_counter() - t0) * 1_000_000)

        bf_us = timed(lambda: self.bf.knn(q, k, dist_fn))
        kd_us = timed(lambda: self.kdt.knn(q, k, dist_fn))
        hnsw_us = timed(lambda: self.hnsw.knn(q, k, 50, dist_fn))
        return {"bruteforceUs": bf_us, "kdtreeUs": kd_us, "hnswUs": hnsw_us, "itemCount": len(self.store)}

    def all_items(self):
        return [
            {"id": v.id, "metadata": v.metadata, "category": v.category, "embedding": v.embedding.tolist()}
            for v in self.store.values()
        ]
