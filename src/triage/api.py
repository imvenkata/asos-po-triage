"""Single /triage endpoint.

The agent is built once at startup, not per request: index construction embeds
the whole corpus, and doing that per request would dominate latency and cost.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import TriageAgent, build_agent
from .config import get_settings
from .llm.base import LLMError
from .logging_setup import get_logger, setup_logging
from .models import TriageResult

log = get_logger("api")
_state: dict[str, TriageAgent] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    _state["agent"] = build_agent(settings)
    log.info("Agent ready (provider=%s)", settings.triage_llm_provider)
    yield
    _state.clear()


app = FastAPI(title="PO Exception Triage Agent", version="0.1.0", lifespan=lifespan)


class TriageRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok" if "agent" in _state else "starting"}


@app.post("/triage", response_model=TriageResult)
def triage(request: TriageRequest) -> TriageResult:
    agent = _state.get("agent")
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not ready")
    try:
        return agent.triage(request.question)
    except LLMError as exc:
        # 502: the upstream model failed. Never substitute a fabricated answer.
        log.error("LLM failure: %s", exc)
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {exc}") from exc
