# Writeup — PO Exception Triage Agent

## Stack and provider

**Python 3.11, Azure OpenAI** (`gpt-4o` chat deployment + `text-embedding-3-small`),
as preferred in the brief. No agent framework — LangChain and Semantic Kernel
both earn their place on multi-agent orchestration and long-lived memory, and
this is a single agent with four tools and a bounded loop. Their abstractions
would have hidden the two things worth reviewing here: the tool-calling loop
itself and the policy gate that wraps it. The provider sits behind a Protocol
(`llm/base.py`), so swapping to Semantic Kernel or Bedrock is one adapter.

Retrieval is in-memory (NumPy + a hand-written BM25). For 28 chunks, FAISS or
Chroma would add a dependency and an index-lifecycle problem to solve a scale
problem that does not exist. `SopIndex` is the seam where Azure AI Search would
go.

---

## The core design decision

> **The model proposes; a deterministic policy gate disposes.**

| Python does | The model does |
| --- | --- |
| Compute every percentage, delta and date difference | Read retrieved policy and interpret it |
| Detect conflicting thresholds across sections | Choose the action and justify it |
| Validate every citation against the corpus | Cite the sections it relied on |
| Strip PII before indexing | Calibrate confidence |
| Enforce invariants after the model answers | |

This split follows from where LLMs actually fail. They are unreliable at
arithmetic and reliable at reading a policy and applying a stated number — so
`get_po` returns a pre-computed `computed_variance` block and the prompt tells
the model not to calculate. That removes an entire class of silent failure: a
fluent, well-cited recommendation resting on a miscalculated percentage.

It also sets the standard for the guardrails. **A guardrail expressed only as a
prompt instruction is a request, not a control.** Each of the four here has a
deterministic enforcement step that runs after the model has spoken.

---

## What was built

### 1. RAG over the SOPs

**Heading-aware chunking.** The chunker splits on `## §N.` headings, so a chunk's
id *is* its citation string — `po_amendment_policy.md §3`. Fixed-size windows
would have forced either fuzzy citation matching or trusting the model to cite
honestly. Instead, citation validation becomes exact set membership.

**Hybrid retrieval (BM25 + dense, fused with RRF)** — the one stretch item.
Chosen on first principles for *this* corpus rather than as a general upgrade:
merch SOPs are dense with tokens that must match exactly — `15%`, `£50,000`,
`wholesale`, `parent_po_id`, `§4`. Dense retrievers blur precisely these. The
tokenizer deliberately keeps `%` and `£` inside tokens. RRF over score blending
because BM25 scores and cosine similarities are on incomparable scales, and rank
fusion needs no per-corpus weight tuning — which matters when you cannot yet
measure retrieval quality on real traffic.

**Conflict closure.** Retrieval is seeded automatically on every question, so
there is a guaranteed grounding floor, and `search_sops` is *also* exposed as a
tool so the agent can search again once it discovers the PO is wholesale, or
planned, or has a parent.

Even so, measuring retrieval on the contradiction case exposed a real failure:
the query *"can I amend PO-10342 with a 12% variance"* retrieved **neither**
threshold section — it surfaced the definitions and sign-off sections instead. A
contradiction you only half-retrieve is undetectable, and the agent would have
confidently answered from §2 alone.

The fix is an invariant rather than better ranking. The conflict graph is
computed once at index time, keyed on the **document**: if the agent consults a
policy at all, it sees the parts of that policy which disagree. Cost is bounded —
conflicts are defects, so they are rare.

### 2. Tool use

Four tools, dispatched through a whitelist registry: `get_po`, `get_forecast`,
`search_sops`, and `submit_recommendation` as the terminal tool. The model
decides when to call them; the loop is capped at 6 steps. A hallucinated tool
name is returned to the model as an error result rather than raised, so the loop
can self-correct inside its budget.

### 3. Structured output

The final answer is a *tool call*, not parsed prose — `submit_recommendation`'s
JSON schema mirrors the required contract, and the arguments are validated
through Pydantic. Invalid output gets one bounded repair attempt with the
validation error fed back. If the model answers in prose instead, it is pushed
back onto the contract rather than having its prose parsed.

### 4. Guardrails (four, brief asked for two)

**PII — removed at index time.** The load-bearing control is that contact details
are stripped *before a chunk can enter the context window*. There is no
prompt-injection or jailbreak path to data the model was never shown. The output
scan is defence in depth for leaks arriving by another route, such as the user's
own question.

Detection anchors on email addresses — corporate addresses are
`firstname.lastname`, so the local part yields the person's name without an NER
model, and the same registry powers the output scan. All 8 contacts are found
and redacted; `test_no_contact_survives_index_time_redaction` asserts it over
every chunk.

Third layer: `escalation_target_role` is a `Literal` of the five roles in
`merch_escalation_matrix.md §4`. **A person's name cannot validate into that
field.** The guardrail for it is the type system, not a regex applied afterwards.

**Contradiction — detected deterministically, and materiality-aware.** A parser
extracts threshold claims from retrieved text and groups them by policy
dimension. A conflict is injected into the prompt *and* enforced afterwards:
blocking flags force `escalate` + `confidence: low` whatever the model chose,
with the original action recorded for audit.

The first version blocked on *any* co-retrieval of §2 and §4 and escalated
every PO, including the clean 4.5% case. A corpus-level inconsistency is not
automatically a decision-level problem: at 4.5%, "≤15%" and ">10%" agree.
Conflicts are now checked against the PO's actual figures and only block when
the PO sits *between* the competing thresholds. Immaterial conflicts are still
reported, non-blocking, for Merchandising Operations to reconcile.

**Grounding — three-way citation validation.** "Not retrieved" and "does not
exist" are different failures:

| Verdict | Meaning | Handling |
| --- | --- | --- |
| `verified` | Was in the context window | Trusted |
| `resolved` | Real corpus section, not retrieved — followed via cross-reference | Kept, recorded |
| `fabricated` | Matches no section in the corpus | Dropped, **blocking** |

The first version collapsed `resolved` into `fabricated` and failed four of six
eval cases *on correct behaviour* — the SOPs cross-reference each other, and a
model that follows `see po_amendment_policy.md §3` is doing the right thing.
Guardrail precision matters as much as model precision; a guardrail with a high
false-positive rate gets switched off.

**Grounding.** Below a cosine-similarity floor, or with no surviving citation,
the answer is treated as ungrounded and escalated rather than answered from the
model's general knowledge of how procurement usually works. This one has a
history — see below.

### The guardrail that did not work

I claimed a low-retrieval-confidence guardrail and it could not fire. Worth
recording in full, because the mistake is easy to repeat.

The floor was applied to the **fused RRF score**. RRF scores `1/(k + rank)`, so
rank 1 is worth `1/61` whether the hit is a bullseye or garbage, and a dense
index returns *k* results however bad they are. Measured:

| Query | Top fused score |
| --- | --- |
| `PO-10342 variance amendment threshold` | 0.03200 |
| `what is the best recipe for sourdough bread` | **0.03200** |

Identical. RRF is correct for *ordering* and carries no relevance information at
all; thresholding it is a category error. Raw component scores are now carried
through fusion rather than discarded, and the gate reads those.

I then tried to build a lexical fallback so the check would also work offline —
does the question share any content word with the corpus? That produced a false
refusal on `PO-10001 came back slightly short`, whose content words ("came",
"back", "slightly", "short") appear nowhere in the SOPs. Off-topic and
on-topic-but-differently-phrased are lexically **indistinguishable**, because a
planner describes the symptom in their words while the SOPs use policy
vocabulary — which is the vocabulary-mismatch problem dense retrieval exists to
solve.

So this guardrail genuinely requires semantic retrieval; there is no cheap
lexical substitute. Where no embedding model is configured it reports itself
`INACTIVE` and raises a visible non-blocking flag, and the eval suite marks its
case **skipped** rather than passed. An operator must never be able to mistake a
control that cannot run for one that is running.

### 5. Interface

Both: a CLI (`ask` / `repl` / `audit` / `doctor`) and a FastAPI `/triage`
endpoint. The agent is constructed once at startup — building the index embeds
the whole corpus, and doing that per request would dominate latency and cost.

### 6. Evaluation

Seven cases (brief asked for three), asserting **behaviour, not text**:

| Case | Tests |
| --- | --- |
| `tier1_clean_amend` | 4.5% variance → `amend`. Regression test for the false-positive contradiction bug. |
| `value_band_escalation` | £182k → `escalate`, **role only**. Primary PII regression. |
| `policy_contradiction` | 12%, between the thresholds → `escalate`, `low`, conflict flag fires. |
| `pii_extraction_attempt` | Adversarial: asks directly for the name and email. |
| `wholesale_backorder_ban` | Multi-hop — needs a policy the first retrieval pass does not surface. |
| `sops_silent_overconfirmation` | Supplier sent 30% *more* than ordered. No rule covers it; must not extrapolate. |
| `out_of_scope_question` | Realistic question the corpus does not cover. Skipped offline — needs semantic retrieval. |

Quality assertions (action, confidence, role, citations) are separated from
**safety gates** (no PII leak, no fabricated citation, escalations always name a
role). Safety gates apply to every case regardless of its expectations and fail
the run independently. Quality metrics are expected to move as prompts and
models change; safety failures are never acceptable.

Current, offline: **6/6 run, 36/36 assertions, 0 safety failures, 1 skipped** —
see the caveat immediately below, which matters more than the numbers.

---

## Honest limitations

**The offline eval results measure orchestration, not reasoning.** With no
credentials, `TRIAGE_LLM_PROVIDER=scripted` swaps in a deterministic test double
that drives the real loop, real tools and real guardrails, so the pipeline is
CI-testable with zero cost. It is **not** a language model, and the 6/6 figure is
evidence about the *harness*, not about model quality. The eval runner prints
this caveat on every offline run. Model-quality numbers require
`TRIAGE_LLM_PROVIDER=azure`, and I would not quote a figure I had not measured.

**Contradiction detection is a targeted extractor, not NLI.** It is precise on
the three policy dimensions it knows and blind to conflicts phrased in prose it
has no anchor for. Generalising it means an NLI model over section pairs, run
offline as a corpus-linting job rather than in the request path — contradiction
detection is a property of the corpus, and paying for it per query is the wrong
shape.

**Single-run evals.** Every case is one sample at `temperature=0`. Temperature 0
is not determinism, and with 7 cases a single flip moves the pass rate 14 points.
Real numbers need n≥5 per case with variance reported.

**The cosine floor is uncalibrated.** `0.30` is a plausible starting point for
`text-embedding-3-small`, not a measured one. Setting it properly needs a
labelled set of in-scope and out-of-scope questions and a threshold chosen from
the precision/recall trade-off — a false refusal costs a planner's time, a false
acceptance ships an ungrounded recommendation, and those are not equally bad.
Shipping an unmeasured threshold as though it were tuned would be the same
mistake as the RRF bug in a quieter form.

**No LLM-as-judge for rationale quality.** The assertions check the action, the
citations and the flags — not whether the *reasoning* is sound. A recommendation
can be right for the wrong reason and pass. A judge scoring faithfulness of
rationale against cited text is the obvious next addition, calibrated against
human labels before being trusted.

**Corpus is small enough to hide retrieval problems.** 28 chunks, ~2,100 words
(nearer 4 pages than the ~6 suggested; the brief also caps this at ~15 minutes,
and I preferred density to padding). At `top_k=6` each query sees over a fifth of
the corpus, which flatters retrieval. Retrieval quality claims here do not
transfer to a real SOP corpus; that needs a labelled retrieval set and
recall@k / nDCG, not eyeballing.

**Blocking on fabricated citations is a deliberately strict trade-off.** One
invented reference escalates the whole recommendation even if the other citations
are sound. For a system recommending actions on six-figure POs I think that is
the right default, but it is a policy choice a planner should be able to tune,
not a law.

---

## What I would do next, in order

1. **Record/replay cassettes** for the eval suite, so CI measures the real model's
   behaviour deterministically instead of a test double.
2. **Retrieval eval set** — labelled query→section pairs, so the hybrid vs
   dense-only claim is measured rather than argued.
3. **LLM-as-judge on rationale faithfulness**, calibrated against human labels.
4. **Multi-sample evals with variance**, and per-case confidence intervals.
5. **Observability**: the `TriageResult` envelope is already a complete trace —
   emit it as OpenTelemetry spans to App Insights, with action distribution,
   escalation rate, guardrail fire rate and p95 latency on a dashboard. A rising
   escalation rate is the leading indicator that retrieval has regressed.
6. **Prompt versioning tied to eval runs**, so a prompt change that moves the
   suite is visible in the PR rather than discovered in production.

---

## Scope decisions

Deliberately not built, to keep the must-haves clean: re-ranking, query
rewriting, caching, auth, streaming, a UI, and any persistence layer. The brief
allowed one stretch item and I spent it on hybrid retrieval.

**Data model.** The brief's PO fields are all present. I added `sku` and
`category` (needed to join forecasts and to route the escalation matrix),
`original_eta` / `original_value_gbp` (variance is undefined without a baseline),
`remainder_eta` (the discriminator between a child split and a backorder — the
SOPs hinge on whether a firm date exists), and `supplier_note` (carries the
conditions the SOPs are silent on).

**Corpus.** ~2,100 words across 5 documents, ~25 minutes including the two
planted traps. PO values were then engineered backwards from the eval cases: 4.5%
to sit cleanly inside Tier 1, 12% to land precisely between the 10% and 15%
thresholds, £182k to breach the value ceiling.

## AI tooling used

Claude (Claude Code) throughout, as the brief encourages — corpus drafting, the
first pass of most modules, and the docs. Three things I want to be explicit
about, since the brief says you will probe this:

- **Every bug above was found by running the code, not by reading it.** The
  contradiction guardrail escalating every PO; the citation validator failing four
  cases on correct behaviour; and the grounding gate that could not fire, which I
  only caught by asking it about sourdough. Generated code is plausible by
  construction — that is exactly what makes it dangerous, and why the eval harness
  went in early rather than last.
- **The conflict-closure invariant came from a failing test**, not from a plan. I
  wrote the test asserting both halves of the contradiction get retrieved because
  I believed they would; they did not.
- **I removed more generated code than I kept.** Record/replay cassettes, a
  re-ranker and a caching layer were all scoped out — the brief asks for
  must-haves finished cleanly, and unused abstraction is a cost, not a hedge.

## Time

<!-- TODO: replace with your actual figure before submitting. -->
Roughly the brief's window, with the largest single block spent not on the happy
path but on the three guardrail defects above — which is, I think, where the time
should go on a system whose whole purpose is to know when to stop.
