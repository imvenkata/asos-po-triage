"""Deterministic local embedder used by the `scripted` provider.

This is a hashed bag-of-words projection, NOT a semantic model. It exists so the
retrieval pipeline, fusion, and guardrails are testable in CI with no network and
no credentials. Semantic quality claims must be measured against the real
embedding deployment - see WRITEUP.md.
"""

from __future__ import annotations

import hashlib
import math
import re

DIMS = 256
_TOKEN = re.compile(r"[a-z0-9£%\.]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _bucket(token: str) -> int:
    return int(hashlib.blake2b(token.encode(), digest_size=4).hexdigest(), 16) % DIMS


def embed_texts(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for text in texts:
        vec = [0.0] * DIMS
        toks = _tokens(text)
        for tok in toks:
            vec[_bucket(tok)] += 1.0
        # Include bigrams so short queries retain a little word-order signal.
        for a, b in zip(toks, toks[1:]):
            vec[_bucket(f"{a}_{b}")] += 0.5
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out.append([v / norm for v in vec])
    return out
