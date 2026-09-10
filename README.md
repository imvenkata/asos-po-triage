# PO Exception Triage Agent

A take-home prototype for ASOS Merchandising. A planner asks about a purchase
order (PO); the agent looks up mock order data and standard operating procedures
(SOPs), then recommends an action with citations and an escalation role.

Built with Python, Azure OpenAI, LangGraph and FastAPI. It recommends actions
only—it does not change orders or record approvals. Design choices and
limitations are in [WRITEUP.md](WRITEUP.md).

## Setup

Requires Python 3.11 or newer.

~~~bash
make install
source .venv/bin/activate
cp .env.example .env
~~~

Set the Azure endpoint, API key, chat deployment and embedding deployment in
.env, then run:

~~~bash
make doctor
~~~

The tested deployments were gpt-5.6-luna for chat and text-embedding-3-small for
embeddings. Normal Azure operation requires a working semantic index.

Without credentials, use the scripted test double:

~~~bash
make eval-offline
~~~

This checks the application flow and guardrails, not real model reasoning.

## Run it

~~~bash
# Clean amendment
triage ask "PO-10001 has a minor variance. Can I amend it in place?" --trace

# Conflicting policy thresholds
triage ask "PO-10342 is about 12% under the original order. Can I amend it?" --json

triage repl    # interactive questions; each starts a new investigation
triage audit   # list known policy conflicts
~~~

For the local HTTP endpoint, run make api, then in another terminal:

~~~bash
curl -s http://127.0.0.1:8000/triage \
  -H 'Content-Type: application/json' \
  -d '{"question":"PO-10342 is 12% under. Can I amend it?"}'
~~~

The API has no authentication; keep it local.

## Output

The recommendation has six fields. For example:

~~~json
{
  "po_id": "PO-10342",
  "recommended_action": "escalate",
  "rationale": "The 12% variance falls between conflicting rules: amendment up to 15% is allowed [po_amendment_policy.md §2], while above 10% requires re-raise [po_amendment_policy.md §4]. A policy decision is needed.",
  "citations": ["po_amendment_policy.md §2", "po_amendment_policy.md §4"],
  "confidence": "low",
  "escalation_target_role": "Senior Merch Planner"
}
~~~

Allowed actions are amend, split_child_po, firm_planned_order, raise_backorder
and escalate. The full response places this object under recommendation, with
guardrail flags, evidence IDs, tool summaries and telemetry alongside it. Live
wording can differ from the example.

## Project structure

~~~text
corpus/                  Five mock SOPs, including a deliberate contradiction
data/                    Twenty mock purchase orders and forecasts
src/triage/
  agent.py               Two-node model/tool loop
  models.py              Data and recommendation schemas
  tools/                 Order, forecast, policy search and submission tools
  ingest/ retrieval/     Section chunking and hybrid search
  guardrails/            Privacy, citation, relevance and conflict checks
  policy_gate.py         Final validation and enforcement
  policy_rules.py        Selected business-rule checks
  llm/                   Provider clients and the offline double
  observability.py       Request IDs, timings and per-call usage
  token_budget.py        Chat-token accounting
  cli.py api.py          Terminal and HTTP interfaces
evals/                   Behavioural cases, runner and calibration script
tests/                   Unit and integration tests
~~~

## Tests and evaluations

~~~bash
make test
make eval-offline
# Requires Azure credentials; makes paid model calls
python evals/run_evals.py --repeat 2 --json eval_report.json
~~~

Recorded checks from 8 September 2026:

- 111 unit/integration tests passed, including in a fresh environment.
- Offline: 10/10 scenarios passed; two semantic cases were skipped.
- Azure: 12 scenarios run twice, 24/24 passed, 210/210 assertions and no safety
  failures observed.

[eval_report.json](eval_report.json) records the tested settings and source
fingerprints. Earlier runs, including failures, are retained in evals/reports/.
This is a small synthetic regression set, not a production reliability estimate.

## Request limits

Settings are in [config.py](src/triage/config.py) and [.env.example](.env.example).

| Setting | Default |
| --- | --- |
| TRIAGE_MAX_AGENT_STEPS | 6 model turns |
| TRIAGE_TOTAL_TIMEOUT_S | 120 seconds |
| TRIAGE_MAX_TOTAL_CHAT_TOKENS | 50,000 per investigation |
| TRIAGE_MAX_INPUT_TOKENS | 16,000 per model call |
| TRIAGE_MAX_OUTPUT_TOKENS | 2,048 per model call |

The budget estimates the complete input and reserves output space before each
call. Reported usage replaces that reservation; missing usage keeps it.
Insufficient room causes escalation, not silent removal of evidence. Automatic
chat retries are disabled. Embeddings are excluded, so this is not an exact
billing cap.

Use --trace for timings and token usage. The API also returns X-Request-ID for
correlation with local logs. The new telemetry events omit raw questions, tool
contents and provider error bodies.
