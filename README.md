# PO Exception Triage Agent

A prototype agent for ASOS Merchandising. A Merch Planner asks about a problem
purchase order in natural language; the agent grounds itself in the merch SOPs,
pulls live PO and forecast data through tool calls, and returns a **structured**
recommendation — `amend`, `split_child_po`, `firm_planned_order`,
`raise_backorder` or `escalate` — with citations, a confidence level, and a
role-only escalation target.

Design rationale, trade-offs and limitations are in **[WRITEUP.md](WRITEUP.md)**.

---

## Quick start

```bash
make install
cp .env.example .env      # fill in your Azure OpenAI details
make doctor               # verifies config and reaches the model
```

No credentials to hand? Everything below runs offline:

```bash
TRIAGE_LLM_PROVIDER=scripted make eval-offline
```

## Usage

```bash
# One-shot question
triage ask "PO-10342 is about 12% under the original order. Can I amend it in place?"

# Show the agent trace (which tools it chose, in what order)
triage ask "What should happen with PO-10777?" --trace

# Full result envelope as JSON
triage ask "Can I backorder the shortfall on PO-10600?" --json

# Interactive planner loop
triage repl

# Corpus hygiene: every threshold conflict, independent of any PO
triage audit

# Check provider configuration and connectivity before a demo
triage doctor
```

HTTP interface:

```bash
make api
curl -s localhost:8000/triage -H 'content-type: application/json' \
  -d '{"question":"PO-10342 is 12% under. Can I amend it?"}' | jq .recommendation
```

## Output contract

`POST /triage` returns an envelope; `recommendation` is the contract the brief
specifies, unchanged:

```json
{
  "po_id": "PO-10342",
  "recommended_action": "escalate",
  "rationale": "po_amendment_policy.md §2 permits in-place amendment up to 15% cumulative variance, while §4 requires cancel-and-re-raise above 10%. This PO sits at 12%, between the two, so the sections disagree...",
  "citations": ["po_amendment_policy.md §2", "po_amendment_policy.md §4", "variance_detection_sop.md §5"],
  "confidence": "low",
  "escalation_target_role": "Senior Merch Planner"
}
```

Everything operational — guardrail flags, retrieved chunk ids, the tool trace,
token usage — sits alongside it on the envelope rather than polluting the
contract.

## Layout

```
corpus/          5 merch SOPs (~2,100 words). Contains a deliberate contradiction.
data/            20 POs + forecasts. Values engineered to hit specific policy boundaries.
src/triage/
  llm/           provider protocol; Azure/OpenAI adapter; offline test double
  ingest/        heading-aware chunker (chunk id == citation string)
  retrieval/     BM25 · dense · RRF fusion · conflict-closure invariant
  guardrails/    pii · contradiction · grounding
  tools/         get_po · get_forecast · search_sops · submit_recommendation
  agent.py       bounded tool-calling loop + deterministic policy gate
  cli.py api.py  the two interfaces
evals/           7 behavioural cases, scored runner, threshold calibrator
tests/           39 unit tests
```

## Testing

```bash
make test            # 39 unit tests, no credentials required
make eval-offline    # 6 cases offline; 1 skipped (needs semantic retrieval)
make eval            # the same cases against the configured live provider

# Re-derive the grounding threshold for your embedding model
python evals/calibrate_threshold.py
```

Measured against Azure OpenAI (`gpt-5.6-luna` + `text-embedding-3-small`):
**7/7 cases, 42/42 assertions, 0 safety failures.** The first live run was 4/7 —
see [WRITEUP.md](WRITEUP.md#what-live-evaluation-changed).

One case (`out_of_scope_question`) exercises the grounding guardrail, which
requires a real embedding model. Offline it is reported as **skipped**, never as
passed — see [WRITEUP.md](WRITEUP.md#the-guardrail-that-did-not-work).

`make eval` exits non-zero on any failure, so it can gate CI. Safety assertions
(no PII leakage, no fabricated citations, escalations always name a role) are
applied to every case regardless of what that case asserts, and fail the run
independently of the quality assertions.

## Configuration

All configuration is typed in `src/triage/config.py`; nothing reads `os.environ`
directly. See `.env.example` for the full list. The ones that matter:

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRIAGE_LLM_PROVIDER` | `azure` | `azure` · `openai` · `scripted` |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | `gpt-4o` | Chat deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | unset | Omit and retrieval runs lexical-only, loudly |
| `TRIAGE_RETRIEVAL_TOP_K` | `6` | Before conflict closure |
| `TRIAGE_MIN_SEMANTIC_SIMILARITY` | `0.38` | Cosine floor for grounding. Needs a real embedding model; calibrated via `evals/calibrate_threshold.py` |
| `TRIAGE_MAX_AGENT_STEPS` | `6` | Hard ceiling on the tool-calling loop |
