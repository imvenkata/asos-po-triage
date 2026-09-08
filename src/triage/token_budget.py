"""Small chat-call admission budget, separate from provider token reporting.

Count the full messages and tool schemas using a UTF-8 size heuristic with
overhead. This works offline without downloading a tokenizer; it is NOT an exact
count or guaranteed upper bound for a provider's message serialisation.
Reserve input estimate + maximum completion before every call. Keep the whole
reservation on failure/missing usage; otherwise settle with provider totals.
Embeddings and provider billing reconciliation are outside this chat budget.
"""

from __future__ import annotations

import json
import math

from langchain_core.utils.function_calling import convert_to_openai_tool

from .config import Settings


def estimate_input_tokens(messages, tool_schemas) -> int:
    payload = {
        "messages": [
            {
                "role": m.type,
                "content": m.content,
                "name": m.name,
                "tool_calls": getattr(m, "tool_calls", []),
                "tool_call_id": getattr(m, "tool_call_id", None),
            }
            for m in messages
        ],
        "tools": tool_schemas,
    }
    size = len(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))
    return math.ceil(size / 3) + 256 + 24 * len(messages)


class TokenBudget:
    def __init__(self, settings: Settings):
        self.limit = settings.triage_max_total_chat_tokens
        self.max_input = settings.triage_max_input_tokens
        self.max_output = settings.triage_max_output_tokens
        self.charged = 0
        self.usage_complete = True

    @staticmethod
    def schemas(tools) -> list[dict]:
        return [convert_to_openai_tool(tool) for tool in tools]

    def reserve(self, estimated_input: int) -> bool:
        amount = estimated_input + self.max_output
        if estimated_input > self.max_input or self.charged + amount > self.limit:
            return False
        self.charged += amount
        return True

    def settle(self, estimated_input: int, usage, measurement) -> bool:
        """Return False if reported usage exceeds a configured boundary."""
        keys = ("input_tokens", "output_tokens", "total_tokens")
        valid = isinstance(usage, dict) and all(
            type(usage.get(k)) is int and usage[k] >= 0 for k in keys
        )
        valid = valid and usage["total_tokens"] >= usage["input_tokens"] + usage["output_tokens"]
        if not valid:
            self.usage_complete = False
            measurement.usage_source = "reservation"
            return True
        for key in keys:
            setattr(measurement, key, usage[key])
        measurement.usage_source = "provider"
        self.charged += usage["total_tokens"] - (estimated_input + self.max_output)
        return (
            self.charged <= self.limit
            and usage["input_tokens"] <= self.max_input
            and usage["output_tokens"] <= self.max_output
        )
