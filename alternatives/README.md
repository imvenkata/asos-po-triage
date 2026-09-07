# Alternative orchestration: LangGraph

`langgraph_agent.py` is the same triage agent with the hand-written tool-calling
loop replaced by a LangGraph `StateGraph`. It exists so the framework question in
[WRITEUP.md](../WRITEUP.md) can be checked rather than argued.

```bash
pip install -e ".[langgraph]"

# one question
python alternatives/langgraph_agent.py "PO-10342 is 12% under. Can I amend it?"

# the full eval suite, through LangGraph
make eval-langgraph
```

It needs a real provider — Azure OpenAI or OpenAI. The scripted offline double
drives the hand-written loop only, and the LangGraph path refuses it with an
explicit error rather than silently degrading.

## What actually changed

| | Hand-written | LangGraph |
| --- | --- | --- |
| Orchestration | `triage/agent.py::_run_loop`, ~105 lines | `StateGraph`, 3 nodes + 1 conditional edge |
| Tool dispatch | `triage/tools/registry.py`, hand-rolled | `ToolNode` |
| Tool schemas | hand-written JSON dicts | derived from Pydantic arg models |
| Message bookkeeping | explicit `messages.append(...)` | `add_messages` reducer |
| Step ceiling | `range(1, max_steps + 1)` | `recursion_limit` |
| Trace for the policy gate | recorded as it happens | read back out of the transcript |

**Imported verbatim by both, not reimplemented:**

- `triage/policy_gate.py` — the entire guardrail layer
- `triage/retrieval/` — hybrid retrieval, conflict closure
- `triage/tools/po_tools.py` — including the pre-computed variance block
- `triage/models.py`, `triage/prompts.py` — contract and prompt

`tests/test_langgraph_alternative.py` asserts that both modules resolve
`apply_policy_gate` to the same function object, so the claim cannot quietly rot.

## Measured, both against `gpt-5.6-luna` + `text-embedding-3-small`

| | Hand-written | LangGraph |
| --- | --- | --- |
| Cases passed | 7/7 | 7/7 |
| Assertions | 42/42 | 42/42 |
| Safety failures | 0 | 0 |
| Action agreement | — | 7/7 identical |
| Total model turns | 24 | 24 |

Confidence differed on two cases (`value_band_escalation`, `pii_extraction_attempt`).
That is sampling noise, not an orchestration difference — this deployment rejects
an explicit `temperature`, so neither implementation can run greedily. It is a
useful reminder that n=1 results are one observation, not a rate.

One behavioural difference worth noting: under LangGraph the model more often
issued `get_po` and `search_sops` **in the same turn** as parallel tool calls,
where the hand-written run tended to sequence them. Same total turns, slightly
different shape. `ToolNode` handles parallel calls without any work on my part;
the hand-written loop also handles them, but I had to write that.

## What LangGraph genuinely buys

- The loop becomes declarative. Adding a re-ranking step or a human-approval
  pause is an edge change, not surgery on a `while` loop.
- `ToolNode` covers dispatch, argument parsing, and turning tool errors into
  messages the model can recover from — all of which `tools/registry.py` does by
  hand.
- Checkpointing, streaming and interrupt/resume come free. For a triage workflow
  that should pause for planner sign-off, `interrupt_before` is one line; in the
  hand-written loop it is a redesign. **This is the strongest argument for the
  framework in this particular domain.**
- LangSmith tracing with no instrumentation.

## What it costs

- Three dependencies and their version churn, for ~105 lines.
- You debug through someone else's control flow when the graph misbehaves.
- The framework owns the transcript, so domain-specific bookkeeping has to be
  read back out of it afterwards (`_harvest`) rather than recorded at the point
  it happened. That is a real loss of fidelity — the hand-written loop knows
  which step a tool call belonged to; the reconstruction has to infer it.

## The conclusion

Neither implementation is better here — they score identically, because **the
loop was never the hard part.** The guardrails are, and both call the same 177
lines to do that.

For one agent under review, hand-writing it keeps the interesting code visible.
For a fleet of agents across squads, the framework wins on exactly the axis the
role description names: one loop implementation, one tracing story, one place to
change a shared safety policy. Build once, scale many times.
