"""Structured-ish logging. One place to configure it, so the agent trace and the
library logs interleave predictably when demoing."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)-7s %(name)-28s %(message)s"))
    root = logging.getLogger("triage")
    root.setLevel(level.upper())
    root.addHandler(handler)
    root.propagate = False
    # The openai client is chatty at DEBUG and drowns the demo.
    logging.getLogger("httpx").setLevel("WARNING")
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"triage.{name}")
