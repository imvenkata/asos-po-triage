# PO Exception Triage Agent

## Approach

I built a small assistant to help a merchandising planner decide what to do with
a problem purchase order. It uses five mock standard operating procedures (SOPs),
twenty purchase orders and forecast data. It recommends an amendment, split,
backorder, firming a planned order or escalation. It never changes an order.

I wanted the model to handle the investigation and explanation, with arithmetic
and clear business constraints in Python. For example, get_po returns calculated
quantity, value and date variances. The model can then choose whether to fetch a
forecast or search for a more specific policy.

The LangGraph loop has two nodes: one calls the model and the other validates
and runs its tool requests. One agent was enough for this task. A plain Python
loop would also work; I used LangGraph to make the state and stopping conditions
explicit.

The final answer comes through submit_recommendation, using the same Pydantic
schema as the application. An invalid submission gets one repair opportunity
within the six-turn limit. A separate policy gate checks the proposal before it
is returned.

## Retrieval

Hybrid retrieval is my stretch item. BM25 finds exact policy wording, embeddings
help with paraphrases, and reciprocal rank fusion combines their ranked results.
I split the documents at numbered headings so citations refer to readable
sections.

There are only 28 chunks, so I kept the index in memory using NumPy. A hosted
search service was unnecessary at this size. Providing all five documents in
the prompt would be a useful baseline to compare against.

The important retrieval detail is keeping conflicting rules together. If a
result comes from a policy with a known threshold conflict, the index adds both
conflicting sections. Otherwise the model could give a plausible answer after
seeing only half the guidance.

## Checks on the answer

The planted contradiction allows amendment up to 15% in one section but requires
cancel-and-re-raise above 10% in another. PO-10342 has a 12% variance, so the agent
must escalate to a Senior Merch Planner with low confidence. The conflict does
not change the outcome for the clean 4.5% example.

Other checks require a successful lookup of the requested PO, citations whose
text was shown to the model, and agreement between inline references and the
citation list. Numeric claims are checked against cited policy and structured
tool results. Selected business rules cover the £50,000 escalation ceiling,
wholesale backorders and delivery-window requirements.

Known contacts are removed before indexing. The outward response is checked for
personal details, including in diagnostic fields, and escalation targets are
roles rather than named people. These checks are not a general-purpose privacy
guarantee.

Blocking failures produce an escalation. If the model's proposed action is
rejected, the explanation is rebuilt so it does not keep recommending that
action.

## Request limits and logging

Requests have a deadline, a turn limit and a configurable chat-token budget.
Before each model call, I estimate the full input, including tool schemas, and
reserve output space. Reported usage replaces the reservation; missing usage
keeps it. The agent escalates if another call cannot fit, rather than trimming
evidence. This excludes embeddings and is not an exact billing cap.

Request IDs, stage timings and per-call token usage help explain where time and
tokens went. The new log events do not contain raw questions or tool contents.
Automatic chat retries are disabled so attempts cannot bypass the accounting.

## Testing and what I learned

The recorded checks passed 111 unit/integration tests, 10 offline scenarios and
24 live Azure runs: twelve cases tested twice, with 210 assertions and no safety
failures observed. Two semantic cases are skipped offline because the scripted
double cannot test model reasoning. Reports, including earlier failures, are
saved in the repository.

Testing changed the relevance check. I originally used the fused retrieval
score, but a high rank does not establish relevance. The gate now uses raw
cosine similarity from the original question. The 0.38 cutoff comes from 25
development questions and still rejects some legitimate short queries.

Disabling chat retries also exposed reuse of a client across closed event loops.
The interactive CLI and evaluation runner now keep one loop alive per session.

## Limitations and next steps

A real citation and a matching number can still support a misleading explanation.
I would start next with planner-labelled examples to test whether claims genuinely
follow from their sources. The current confidence labels are not calibrated
probabilities.

Policy coverage is incomplete, and some thresholds exist in both the documents
and Python. The mock data also lacks full amendment history and supplier
constraints. I would improve those before adding more agents or fine-tuning.

Real deployment would need authentication, source permissions and versioned
evidence. Order execution would additionally need durable approval, stale-data
checks and duplicate-write protection. Those are outside this take-home.

## AI tools used

I used Claude Code for initial drafting and implementation, and Codex for review,
hardening and documentation. I checked the changes through code review,
regression tests and live evaluations.
