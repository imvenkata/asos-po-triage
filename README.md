# PO Exception Triage Agent

A prototype agent for ASOS Merchandising. A Merch Planner asks about a problem
purchase order in natural language; the agent grounds itself in the merch SOPs,
pulls mock PO and forecast records through model-selected tool calls, and returns a **structured**
recommendation — `amend`, `split_child_po`, `firm_planned_order`,
`raise_backorder` or `escalate` — with citations, a confidence level, and a
role-only escalation target.

Design rationale, trade-offs and limitations are in **[WRITEUP.md](WRITEUP.md)**.
The service recommends actions only: it never modifies an order or records an
approval. A detailed rehearsal walkthrough is in **[INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md)**.

---

## Quick start

Requires Python 3.11 or newer.

```bash
make install
source .venv/bin/activate
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

# Show tool choices, stage timings and per-call token measurements
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
  "rationale": "The 12% variance sits between the conflicting amendment thresholds: up to 15% is permitted [po_amendment_policy.md §2], while above 10% requires re-raise [po_amendment_policy.md §4]. Escalate for adjudication [variance_detection_sop.md §5].",
  "citations": ["po_amendment_policy.md §2", "po_amendment_policy.md §4", "variance_detection_sop.md §5"],
  "confidence": "low",
  "escalation_target_role": "Senior Merch Planner"
}
```

Everything operational — guardrail flags, retrieved chunk ids, the tool trace,
token usage, request telemetry and review-required metadata — sits alongside it on the envelope.
This is a sanitised diagnostic summary, not a full replayable trace. The example
above illustrates the contract; live wording can differ.

## Layout

```
corpus/          5 merch SOPs (~2,100 words). Contains a deliberate contradiction.
data/            20 POs + forecasts. Values engineered to hit specific policy boundaries.
src/triage/
  llm/           chat + embedding clients; deterministic offline double
  ingest/        heading-aware chunker (chunk id == citation string)
  retrieval/     BM25 · dense · RRF fusion · conflict-closure invariant
  guardrails/    pii · contradiction · grounding
  tools/         get_po · get_forecast · search_sops · submit_recommendation
  agent.py       LangGraph tool-calling loop
  observability.py request IDs, stage timing and content-free log events
  token_budget.py chat-call admission estimates and usage accounting
  policy_gate.py deterministic guardrail enforcement
  policy_rules.py explicit business-rule checks over the model's chosen action
  cli.py api.py  the two interfaces
evals/           12 behavioural cases, scored runner, threshold calibrator
tests/           111 unit/integration regression tests
```

## Testing

```bash
make test            # 111 tests, no credentials required
make eval-offline    # 10 cases offline; 2 skipped (need a real model)
make eval            # the same cases against the configured live provider
.venv/bin/python evals/run_evals.py --repeat 2 --json eval_report.json

# Re-derive the grounding threshold for your embedding model
python evals/calibrate_threshold.py
```

Measured against Azure OpenAI (`gpt-5.6-luna` chat + `text-embedding-3-small`
embeddings): **12 scenarios run twice; 24/24 passed, 210/210 assertions,
0 safety failures observed.** The report records source fingerprints, settings,
dependencies and returned model metadata. This is a small synthetic regression
set, not a production reliability estimate. The previous single-run report is
preserved in `evals/reports/2026-09-08-single-run.json`.

This deployment rejects an explicit `temperature`, so runs are sampled rather
than greedy and any single number here is one observation, not a stable rate.

Two cases (`out_of_scope_question` and `sops_silent_on_specification`) require
semantic retrieval and live model behaviour. Offline they are **skipped**, never
counted as passed. The offline suite passed 10/10 with 88/88 assertions.

A fresh temporary Python 3.11 installation using the exact dependency constraints
also passed all 111 tests and `pip check`. One third-party test-client deprecation
warning remains.

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
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | `gpt-5.6-luna` | Chat deployment **name**, not model name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | unset | Required for normal Azure operation; missing or failed semantic indexing refuses startup |
| `TRIAGE_ALLOW_LEXICAL_ONLY` | `false` | Explicit degraded demonstration: starts but blocks real-provider decisions |
| `TRIAGE_RETRIEVAL_TOP_K` | `6` | Before conflict closure |
| `TRIAGE_MIN_SEMANTIC_SIMILARITY` | `0.38` | Cosine floor for grounding. Needs a real embedding model; calibrated via `evals/calibrate_threshold.py` |
| `TRIAGE_MAX_AGENT_STEPS` | `6` | Hard ceiling on the tool-calling loop |
| `TRIAGE_REQUEST_TIMEOUT_S` | `60` | Individual provider-call timeout |
| `TRIAGE_TOTAL_TIMEOUT_S` | `120` | Whole asynchronous triage deadline |
| `TRIAGE_MAX_CONTEXT_CHUNKS` | `28` | Unique evidence budget per request |
| `TRIAGE_MAX_TOTAL_CHAT_TOKENS` | `50000` | Cumulative chat-token admission budget per investigation |
| `TRIAGE_MAX_INPUT_TOKENS` | `16000` | Per-call input estimate limit; reported input is checked too |
| `TRIAGE_MAX_OUTPUT_TOKENS` | `2048` | Reserved per call and sent as the provider's completion limit |

## Request measurements and token limits

Use `--trace` for a timing table, or inspect `.telemetry` in the JSON response.
It records a request ID, seed retrieval, model calls, tool execution and final
validation, with durations and per-call input/output tokens where reported.
The API returns the same ID in `X-Request-ID`, including on triage errors.
Matching INFO log events contain fixed operation labels, counters and prompt/
corpus fingerprints, not raw questions, tool arguments or provider error bodies.
These fingerprints identify content; they do not provide durable policy history.

Before each model call, the budget estimates the complete messages and tool
schemas and reserves the maximum output allowance. Complete provider usage
replaces that reservation; missing usage or a failed call keeps it and marks
`usage_complete=false`. `chat_tokens_charged` is this accounting total, not a
bill. Embeddings and startup work are excluded, and the input heuristic is not
an exact token count or guaranteed upper bound. Chat-client automatic retries
are disabled; adding retries would require reserving each attempt separately.

If another call cannot fit, the result is a low-confidence escalation with a
`token_budget_exceeded` flag. Evidence is not silently trimmed. Truncated model
output is rejected without executing its proposed tools.

An offline demonstration that refuses before the first chat call:

```bash
TRIAGE_LLM_PROVIDER=scripted TRIAGE_MAX_TOTAL_CHAT_TOKENS=100 \
  .venv/bin/triage ask "Can I amend PO-10001?" --trace
```

Keep the unauthenticated API local. Production authentication, durable approvals,
write tools, persistent policy versions and complete policy enforcement are not
implemented.
