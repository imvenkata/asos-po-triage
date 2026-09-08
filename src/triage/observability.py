"""Request-local, content-free measurements; no exporter or global request state."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import contextmanager
from uuid import uuid4

from .logging_setup import get_logger
from .models import RequestTelemetry, StageMeasurement

log = get_logger("telemetry")


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class RunTelemetry:
    def __init__(self, prompt_sha256: str, policy_sha256: str):
        self.request_id = "run_" + uuid4().hex
        self.prompt_sha256 = prompt_sha256
        self.policy_sha256 = policy_sha256
        self.started = time.perf_counter()
        self.stages: list[StageMeasurement] = []

    @contextmanager
    def measure(self, stage, operation, **fields):
        # Pydantic rejects arbitrary operation labels. Never pass prompt text,
        # model tool arguments, results or exception bodies into this schema.
        record = StageMeasurement(stage=stage, operation=operation, **fields)
        self.stages.append(record)
        started = time.perf_counter()
        try:
            yield record
        except asyncio.CancelledError:
            record.status = "cancelled"
            raise
        except BaseException:
            record.status = "error"
            raise
        finally:
            record.duration_ms = round((time.perf_counter() - started) * 1000, 3)
            log.info(
                "%s",
                json.dumps(
                    {
                        "event": "triage_stage",
                        "request_id": self.request_id,
                        **record.model_dump(exclude_none=True),
                    }
                ),
            )

    def finish(self, outcome, budget) -> RequestTelemetry:
        result = RequestTelemetry(
            request_id=self.request_id,
            outcome=outcome,
            duration_ms=round((time.perf_counter() - self.started) * 1000, 3),
            prompt_sha256=self.prompt_sha256,
            policy_sha256=self.policy_sha256,
            chat_token_budget=budget.limit,
            chat_tokens_charged=budget.charged,
            usage_complete=budget.usage_complete,
            stages=self.stages,
        )
        log.info(
            "%s",
            json.dumps(
                {
                    "event": "triage_request",
                    **result.model_dump(exclude={"stages"}),
                }
            ),
        )
        return result
