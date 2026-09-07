# Variance Detection SOP

Owner: Merchandising Operations. Version 2.8. Read together with `po_amendment_policy.md`.

## §1. Definition of Variance

Three variance types are tracked independently for every live PO:

- **Quantity variance** — `(ordered_qty - confirmed_qty) / ordered_qty`, expressed as a percentage. Positive values indicate under-confirmation by the supplier.
- **Value variance** — the change in `value_gbp` against the original committed value, expressed as a percentage.
- **ETA variance** — the slip in calendar days between the original ETA and the latest supplier-confirmed ETA.

A PO is flagged as an exception when any one of the three variance types breaches its Tier 1 boundary in §2. Variance types are never averaged or netted against each other; the PO takes the severity of its **worst** breaching type.

## §2. Severity Tiers

| Tier | Name | Quantity or value variance | ETA slip | Handling |
| --- | --- | --- | --- | --- |
| 1 | Minor | Up to 5% | Up to 3 days | Auto-action permitted, no human review |
| 2 | Moderate | Above 5% up to 15% | 4 to 10 days | Human review by Merch Planner |
| 3 | Severe | Above 15% up to 30% | 11 to 20 days | Human review by Senior Merch Planner |
| 4 | Critical | Above 30% | Above 20 days | Escalate, no auto-action |

Any PO with a value above £50,000 is treated as **Tier 4 Critical regardless of variance size** and must be escalated.

## §3. Auto-action Boundary

Auto-action is permitted **only** for Tier 1 exceptions. The agent or planner may apply the amendment without human review where all of the following hold:

- The PO is in status `open` or `partially_received`.
- Quantity, value, and ETA variance are all within Tier 1 bounds.
- The PO has no `parent_po_id` (child POs always require review).

Everything at Tier 2 and above requires a human in the loop. The recommendation may be produced automatically, but it must be presented for approval rather than applied.

## §4. Silent and Unknown Conditions

Where a PO exhibits a condition not described in this SOP or in any related policy — including supplier insolvency, force majeure, customs seizure, or a quality-hold that has no defined variance signature — **no auto-action or recommendation may be inferred by analogy**. These cases must be escalated for human judgement with the unknown condition stated explicitly.

Planners must not extrapolate a threshold that is not written down. If the applicable threshold cannot be located, the correct outcome is escalation, not a best guess.

## §5. Conflicting Guidance

Where two policy documents, or two sections of the same document, specify different thresholds for the same decision, the exception must be escalated to a Senior Merch Planner for adjudication **where the conflict is material to this PO**. Planners must not select whichever threshold is more permissive.

A conflict is material only where the PO's measured figures fall **between** the competing thresholds, so that the two sections would produce different outcomes. Where the PO's figures sit below all of the competing thresholds, or above all of them, both sections agree on the outcome and the exception is handled normally under the applicable tier.

Every conflict is recorded for Merchandising Operations to reconcile, whether or not it was material to the PO in hand.
