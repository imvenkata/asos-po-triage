"""Heading-aware chunking.

Fixed-size sliding windows would give chunk ids with no meaning, which forces
either fuzzy citation matching or trusting the model to cite honestly. Splitting
on the `## §N.` section headings instead makes each chunk's identity the same
string a planner would cite - "po_amendment_policy.md §3" - so citations become
verifiable by set membership.

Corpus sections here are short enough to index whole (largest is well under a
typical 512-token target), so there is no sub-splitting step; the guard below
    rejects oversized sections rather than silently truncating them.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..guardrails.pii import PiiRegistry, redact
from ..logging_setup import get_logger
from ..models import Chunk

log = get_logger("ingest.chunker")

SECTION_RE = re.compile(r"^##\s*§(?P<num>\d+)\.?\s*(?P<title>.*)$", re.MULTILINE)
# Roughly 4 chars/token; warn well before a typical embedding context limit.
MAX_CHUNK_CHARS = 6000


def chunk_document(path: Path, registry: PiiRegistry | None = None) -> list[Chunk]:
    doc = path.name
    text = path.read_text(encoding="utf-8")

    matches = list(SECTION_RE.finditer(text))
    if not matches:
        log.warning("%s has no '## §N.' sections; indexing whole document as §0", doc)
        matches = []

    spans: list[tuple[str, str, str]] = []  # (section, heading, body)
    preamble = text[: matches[0].start()].strip() if matches else text.strip()
    if preamble:
        title = preamble.splitlines()[0].lstrip("# ").strip() or doc
        spans.append(("0", title, preamble))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans.append((m.group("num"), m.group("title").strip(), text[m.start() : end].strip()))

    chunks: list[Chunk] = []
    for section, heading, body in spans:
        if len(body) > MAX_CHUNK_CHARS:
            raise ValueError(f"{doc} §{section} exceeds the chunk size limit; split it before indexing.")
        was_redacted = False
        if registry is not None:
            body, was_redacted = redact(body, registry)
            heading, _ = redact(heading, registry)
        chunks.append(
            Chunk(
                chunk_id=f"{doc} §{section}",
                doc=doc,
                section=section,
                heading=heading,
                text=body,
                redacted=was_redacted,
            )
        )
    return chunks


def chunk_corpus(corpus_dir: Path, registry: PiiRegistry | None = None) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(corpus_dir).glob("*.md")):
        chunks.extend(chunk_document(path, registry))
    if not chunks:
        raise FileNotFoundError(f"No markdown documents found in {corpus_dir}")
    log.info(
        "Indexed %d chunks from %d documents (%d redacted)",
        len(chunks),
        len({c.doc for c in chunks}),
        sum(c.redacted for c in chunks),
    )
    return chunks
