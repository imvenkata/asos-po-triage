"""Dense retrieval over an in-memory matrix.

An in-memory numpy index is the right call for 28 chunks: FAISS/Chroma would add
a dependency and an index-lifecycle problem to solve a scale problem that does
not exist. The `DenseIndex` interface is the seam - swapping in Azure AI Search
for the real corpus means replacing this class, nothing else.

Honest note: `langchain_core.vectorstores.InMemoryVectorStore` is already a
transitive dependency and does the same thing - cosine over a matrix, argsort,
take k - and returns the score. Using it here would be a fair swap rather than
an improvement; this class exists because it indexes positionally against the
chunk list, which keeps the fusion code simple. At real corpus size neither
survives: both are replaced by Azure AI Search.
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
