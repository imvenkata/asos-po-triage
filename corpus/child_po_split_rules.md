# Child PO Split Rules

Owner: Merchandising Operations. Version 3.1.

## §1. When to Sub-divide a Parent PO

A parent PO should be sub-divided into one or more child POs where the ordered quantity will be fulfilled across materially different delivery windows or channels, and the parent record can no longer represent both. Sub-division is the correct action where:

- The supplier has confirmed a partial quantity for the original ETA and the remainder for a later, **known** delivery date.
- A single PO spans both Retail and Wholesale demand and must be separated for channel reporting.
- A quality hold applies to part of the ordered quantity only.

Where the remainder has **no known delivery date**, do not split. Raise a backorder under `backorder_reconciliation.md` instead.

## §2. Sizing Rules

Every child PO must independently satisfy:

- Minimum 500 units, **or** minimum value of £5,000. A residual below both floors must be absorbed into a sibling child PO or written off, not raised as its own child.
- Maximum of 3 child POs per parent PO. A parent requiring a fourth split must instead be cancelled and re-raised under `po_amendment_policy.md §4`.
- The sum of child quantities must equal the parent `ordered_qty`. Child POs never create new demand.

A child PO inherits the parent's supplier and category, and carries `parent_po_id` referencing the parent. Child POs are always subject to human review under `variance_detection_sop.md §3`, regardless of tier.

## §3. Retail and Wholesale Split Logic

Retail and Wholesale demand must never share a PO beyond the point at which the split is identified. Where a PO spans both channels:

- The Retail child PO takes the original ETA and the original `po_id` lineage.
- The Wholesale child PO takes a separate ETA agreed with the Wholesale planning team and must not be netted against Retail forecast.
- Wholesale child POs above £25,000 require Senior Merch Planner sign-off in addition to any sign-off required by `po_amendment_policy.md §3`.

Channel reallocation of an entire PO — as opposed to a split — is not permitted. The PO must be cancelled and re-raised against the correct channel.

## §4. Parent PO Status After Split

Once child POs are raised, the parent moves to status `split` and becomes read-only. Variance is thereafter measured against each child independently. Re-opening a split parent is not supported and requires Merchandising Operations intervention.
