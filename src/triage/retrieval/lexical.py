"""Okapi BM25.

Implemented directly rather than pulled in as a dependency: it is ~40 lines, and
for this corpus the lexical ranker is doing the load-bearing work. The SOPs are
dense with exact tokens that must match exactly - "15%", "£50,000", "wholesale",
"parent_po_id", "§4". Dense retrievers are known to blur precisely these, which
is why hybrid fusion is the retrieval strategy here rather than an upgrade.
"""
from __future__ import annotations

import math
import re
from collections import Counter

K1 = 1.5
B = 0.75
# Keep %, £ and digits as part of tokens - "15%" and "50,000" carry the meaning.
_TOKEN_RE = re.compile(r"[a-z0-9£%_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace(",", ""))


class BM25:
    def __init__(self, documents: list[str]) -> None:
        self._docs = [tokenize(d) for d in documents]
        self._lens = [len(d) for d in self._docs]
        self._avg_len = (sum(self._lens) / len(self._lens)) if self._lens else 0.0
        self._tfs = [Counter(d) for d in self._docs]

        n_docs = len(self._docs)
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        self._idf = {
            term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        q_terms = tokenize(query)
        scores: list[tuple[int, float]] = []
        for idx, tf in enumerate(self._tfs):
            doc_len = self._lens[idx] or 1
            score = 0.0
            for term in q_terms:
                freq = tf.get(term)
                if not freq:
                    continue
                denom = freq + K1 * (1 - B + B * doc_len / (self._avg_len or 1))
                score += self._idf.get(term, 0.0) * (freq * (K1 + 1)) / denom
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda x: (-x[1], x[0]))
        return scores[:top_k]
