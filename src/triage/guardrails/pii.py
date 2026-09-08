"""PII guardrail.

Design note - the load-bearing control is redaction at INDEX time, not filtering
at output time. Contact details from the escalation matrix are stripped before a
chunk can ever enter the context window, so there is no prompt-injection or
jailbreak path to them: the model is never shown the data. The output scan in
`scan_for_pii` is defence in depth for the case where PII reaches the answer by
some other route (e.g. pasted into the user's question).

Emails are the anchor for detection. Corporate addresses are firstname.lastname,
so the local part yields the person's name without needing an NER model, and the
same registry then powers the output scan.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
# "Priya Raman — priya.raman@asos.com" / "James Fenwick, james.fenwick@asos.com"
ADJACENT_NAME_RE = re.compile(
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*[—\-,–:]\s*(?=[\w.+-]+@)"
)
PHONE_RE = re.compile(r"(?<![\w£])(?:\+\d[\d ().-]{7,}\d|0\d[\d ().-]{7,}\d)(?!\w)")

EMAIL_TOKEN = "[REDACTED_EMAIL]"
NAME_TOKEN = "[REDACTED_NAME]"
PHONE_TOKEN = "[REDACTED_PHONE]"


@dataclass
class PiiRegistry:
    emails: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)

    @property
    def is_empty(self) -> bool:
        return not self.emails and not self.names

    def name_patterns(self) -> list[re.Pattern[str]]:
        # Longest first so "Priya Raman" is matched before "Priya".
        return [
            re.compile(rf"\b{re.escape(n)}\b", re.IGNORECASE)
            for n in sorted(self.names, key=len, reverse=True)
        ]


def _names_from_email(address: str) -> set[str]:
    local = address.split("@", 1)[0]
    parts = [p for p in re.split(r"[._\-+]", local) if p.isalpha() and len(p) > 1]
    if len(parts) < 2:
        return set()
    full = " ".join(p.capitalize() for p in parts)
    return {full}


def build_pii_registry(corpus_dir: Path) -> PiiRegistry:
    """Scan the corpus once and record every contact identity it contains."""
    reg = PiiRegistry()
    for path in sorted(Path(corpus_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for address in EMAIL_RE.findall(text):
            reg.emails.add(address)
            reg.names |= _names_from_email(address)
        reg.names |= {m.strip() for m in ADJACENT_NAME_RE.findall(text)}
    return reg


def redact(text: str, registry: PiiRegistry) -> tuple[str, bool]:
    """Return (redacted_text, was_modified). Roles and structure are preserved."""
    original = text
    text = EMAIL_RE.sub(EMAIL_TOKEN, text)
    text = PHONE_RE.sub(PHONE_TOKEN, text)
    for pattern in registry.name_patterns():
        text = pattern.sub(NAME_TOKEN, text)
    # Collapse the "[REDACTED_NAME] — [REDACTED_EMAIL]" pairs left in tables.
    text = re.sub(
        rf"{re.escape(NAME_TOKEN)}\s*[—\-,–]\s*{re.escape(EMAIL_TOKEN)}",
        "[CONTACT WITHHELD - USE ROLE]",
        text,
    )
    return text, text != original


def scan_for_pii(text: str, registry: PiiRegistry) -> list[str]:
    """Output-side detector. Returns human-readable findings; empty means clean."""
    findings: list[str] = []
    for address in EMAIL_RE.findall(text or ""):
        if address != EMAIL_TOKEN:
            findings.append("email address detected")
    for name in sorted(registry.names):
        if re.search(rf"\b{re.escape(name)}\b", text or "", re.IGNORECASE):
            findings.append("registered personal name detected")
    if PHONE_RE.search(text or ""):
        findings.append("phone number detected")
    return findings


def sanitize(value, registry: PiiRegistry):
    """Scrub every string (including dictionary keys) in a public JSON tree."""
    if isinstance(value, str):
        return redact(value, registry)[0]
    if isinstance(value, dict):
        return {redact(str(k), registry)[0]: sanitize(v, registry) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, registry) for v in value]
    return value
