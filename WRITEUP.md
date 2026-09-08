# PO Exception Triage Agent

## What I built

This is a prototype to help a merchandising planner decide what to do with a
problem purchase order (PO). The planner asks a question, the agent looks up the
order and relevant standard operating procedures (SOPs), and returns an action
with a reason and citations.

It can recommend an amendment, a child-order split, firming a planned order,
a backorder or escalation to a human role. It does not make any changes to orders.

I used Python, Azure OpenAI and LangGraph, with five mock policy documents,
twenty purchase orders and forecast data. The configured chat deployment is
`gpt-5.6-luna`, with `text-embedding-3-small` for embeddings. Order and forecast
lookups use local JSON files. There is a CLI and a FastAPI endpoint; setup and
commands are in the README.

## How I approached it

I wanted the model to handle the investigation and explanation, while keeping
calculations and clear business limits in Python.

For example, `get_po` returns the order with its quantity and value variance
already calculated. The model can then decide which policy to look for, rather
than working out percentages itself. It can also fetch the forecast or search
again when it discovers something relevant, such as a wholesale channel or a
known later delivery date.

The LangGraph loop has two nodes: one calls the model, and the other validates
and runs its tool requests. I chose this to make the flow easy to follow. A
plain loop would also work at this size; I did not see a reason to add multiple
agents.

The final answer comes through `submit_recommendation`. Its schema is the same
Pydantic model used by the application, so there is one definition of the allowed
actions and roles. An invalid submission gets one repair attempt within a
six-turn limit. There is also an overall request deadline.

After the model submits, `policy_gate.py` checks the proposal against the
evidence and explicit rules. A valid JSON object is not enough on its own.

## Retrieval choices

I split documents at their numbered section headings. This keeps a rule and its
explanation together, and gives each chunk a readable citation such as
`po_amendment_policy.md §3`.

Hybrid retrieval was my main stretch item. BM25 handles exact policy wording,
while embeddings help with questions phrased differently from the documents.
Reciprocal rank fusion (RRF) combines the two ranked lists without adding scores
that have different scales.

The index is an in-memory NumPy matrix. There are only 28 chunks, so a separate
search service was not necessary for the exercise. Putting all five documents
into the prompt would also be a reasonable baseline at this size. I would compare
that with the retrieval approach before claiming hybrid search is better.

One detail that mattered was retrieving both sides of a known contradiction.
If a search returns a section from a policy containing conflicting thresholds,
the index adds the conflicting sections too. Otherwise, the model could give a
well-cited answer based on only half the guidance.

## Handling unsafe or unsupported answers

The planted contradiction is a good example of why I added checks after the
model's answer. One section permits amendment up to 15% variance; another
requires cancel-and-re-raise above 10%.

For PO-10342, the variance is 12%, so the two rules give different answers. The
agent must escalate to a Senior Merch Planner with low confidence. For the clean
4.5% case, both thresholds agree. Escalating that order just because the documents
contain a contradiction would be unnecessary.

The other checks cover:

- **Order identity:** a recommendation must be supported by a successful lookup
  of the requested PO. The model cannot substitute a different order ID.
- **Citations:** references must exist and their text must have been shown to
  the model. Inline references must match the citation list.
- **Numbers:** figures in the explanation are checked against cited policy and
  structured tool results. This catches invented numbers, but not every wrong
  interpretation of a real number.
- **Personal details:** known contacts are removed before indexing. The full
  response, including diagnostic fields, is checked and redacted. Escalation
  targets are restricted to roles.
- **Action limits:** code checks conditions such as the £50,000 escalation
  ceiling, the wholesale backorder ban and the need for a known delivery window
  before recommending a split.

If a blocking check fires, the result becomes an escalation. When that changes
the model's proposed action, the explanation is rebuilt so it does not keep
telling the planner to take the rejected action.

## Keeping requests bounded and visible

I added timing for retrieval, model calls, tools and final validation, with a
request ID linking the response to local logs. Each model call records reported
input and output tokens. The new log events contain labels and counts, not the
planner's question or tool contents.

Before calling the model, I estimate the full input, including tool schemas,
and reserve output space. The default investigation budget is 50,000 chat tokens,
with 16,000 input and 2,048 output per call. Reported usage replaces the estimate;
missing usage keeps the reservation. If another call cannot fit, the agent
escalates rather than dropping evidence. Automatic chat retries are disabled so
attempts cannot bypass that accounting. This is a practical application limit,
not an exact billing cap, and it excludes embeddings.

## What testing changed

The earlier implementation used the fused RRF score to judge whether retrieval
was relevant. That did not work: rank one gets the same contribution whether the
best result is useful or unrelated. The relevance check now uses raw cosine
similarity from the original question's retrieval, not the model's later searches.

The current cutoff is 0.38, based on a small set of 25 development questions.
Some legitimate short questions still fall below it. I would treat this as a
starting point, not a general measure of answer correctness.

Review also caught problems that ordinary happy-path cases missed: contact
details could leak through diagnostics, a proposed tool call could be mistaken
for a completed lookup, and an existing citation could be accepted without its
text having been read. These now have regression tests.

The latest checks were:

| Check | Result |
| --- | --- |
| Unit and integration tests | 111 passed, also verified in a fresh environment |
| Offline scenarios | 10/10 passed; two semantic cases skipped |
| Live Azure scenarios | Twelve cases run twice; 24/24 passed |
| Live assertions | 210/210 passed; no safety failures observed |

The live suite includes successful examples of all four non-escalation actions,
as well as contradictions, high-value orders, contact extraction and uncovered
conditions. The saved `eval_report.json` records the tested configuration and
source fingerprints.

The offline model is a scripted test double. It checks the application flow,
not language-model reasoning. Two live samples per case are still a small test,
and the model's confidence labels are not calibrated probabilities.

## What I would improve next

The biggest remaining gap is checking whether an explanation genuinely follows
from its sources. A real citation and a matching number can still be used
incorrectly. I would start with planner-labelled examples and a separate test
set for retrieval and explanation quality.

The rule checks are also incomplete. The mock data does not cover full amendment
history, supplier contract minimums or an executable child-order plan. Some
thresholds exist in both the SOPs and Python, so changing policy currently means
reviewing both together. A versioned, business-owned rule source would be a
better long-term approach.

Before using real business data, I would add authentication, order and document
permissions, and versioned evidence. If order execution came into scope, it
would need durable approval, checks that the order had not changed, and
protection against duplicate writes. Those are not implemented in this prototype.

## AI tools used

I used Claude Code for initial drafting and implementation, and Codex for
review, hardening and documentation. The changes were checked through code
review, regression tests and live evaluations.
