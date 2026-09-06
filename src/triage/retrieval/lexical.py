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

# Small closed-class stopword list. Used only for the grounding check, never for
# ranking - BM25's idf already discounts common terms, but the grounding check
# asks a different question ("does this query share any CONTENT word with the
# corpus?") and needs function words removed to answer it.
STOPWORDS = frozenset("""
a an the and or but if is are was were be been being am of to in on at for with
by from as it its this that these those i you we they he she what which who whom
how do does did done can could should would may might will shall must about into
over under then than so such no not only own same too very just now our your
their my me him her them there here when where why all any both each few more
most other some need needs got get
""".split())

# PO / SKU identifiers and bare numerics. Present in almost every real question
# and absent from the policy corpus by design, so counting them as "unmatched
# content words" would make every legitimate query look ungrounded.
IDENTIFIER_RE = re.compile(r"^(?:po|sku)[-_]?[a-z0-9-]*$|^\d+$|^\d+%$|^£\d+$")
# Keep %, £ and digits as part of tokens - "15%" and "50,000" carry the meaning.
_TOKEN_RE = re.compile(r"[a-z0-9£%_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace(",", ""))


def informative_terms(text: str) -> list[str]:
    """Content words only: no function words, no PO/SKU ids, no bare numbers."""
    return [
        t for t in tokenize(text) if t not in STOPWORDS and not IDENTIFIER_RE.match(t)
    ]


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
        self.vocabulary: frozenset[str] = frozenset(df)

    def informative_overlap(self, query: str) -> tuple[int, int]:
        """(content words in the query that exist in the corpus, total content words).

        A necessary condition for grounding, not a relevance score: if a question
        shares NO content word with the corpus, the corpus cannot be about it.
        Deliberately not expressed as a ratio - measured on this corpus, natural
        planner phrasing ("came back slightly short from the supplier") scores a
        LOWER ratio than an off-topic question, because most of the words in a
        real question are ordinary English rather than policy vocabulary. The
        count of zero is the signal; the ratio is not.
        """
        terms = informative_terms(query)
        return sum(1 for t in terms if t in self.vocabulary), len(terms)

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
