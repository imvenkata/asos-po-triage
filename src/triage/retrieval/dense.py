"""Dense retrieval over an in-memory matrix.

An in-memory numpy index is the right call for 28 chunks: FAISS/Chroma would add
a dependency and an index-lifecycle problem to solve a scale problem that does
not exist. The `DenseIndex` interface is the seam - swapping in Azure AI Search
for the real corpus means replacing this class, nothing else.
"""
from __future__ import annotations

import numpy as np


class DenseIndex:
    def __init__(self, vectors: list[list[float]]) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
        q = np.asarray(query_vector, dtype=np.float32)
        norm = float(np.linalg.norm(q)) or 1.0
        sims = self._matrix @ (q / norm)
        top = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in top]
