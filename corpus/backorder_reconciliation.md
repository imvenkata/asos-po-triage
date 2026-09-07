# Backorder Reconciliation

Owner: Merchandising Operations. Version 2.4.

## §1. When to Raise a Backorder

Raise a backorder where the supplier has confirmed less than the ordered quantity and the shortfall is still expected to be delivered, but **no firm delivery date has been agreed**. Where a firm date for the remainder exists, sub-divide into a child PO under `child_po_split_rules.md §1` instead.

Backorders are permitted only where all of the following hold:

- PO status is `open` or `partially_received`.
- The expected delay for the shortfall does not exceed the maximum in §3.
- The PO `channel` is `retail`.
- The shortfall is **above the Tier 1 minor band** defined in
  `variance_detection_sop.md §2`. A shortfall at or below 5% of ordered quantity
  is not backordered; it is absorbed as a routine variance and the PO is amended
  in place under `po_amendment_policy.md §2`.
- The exception is **not Tier 4 Critical**. Where the PO breaches the critical
  tier on variance or value, it is escalated under `variance_detection_sop.md §2`
  and no backorder is raised.

## §2. Channel Restriction

**Backorders are not permitted on Wholesale POs.** Wholesale commitments are contractual against a fixed delivery window and cannot carry an open shortfall. A Wholesale PO with an unfulfilled quantity must be split under `child_po_split_rules.md §3` if a later window is agreed, and escalated if it is not.

## §3. Maximum Permissible Delay

The maximum permissible backorder delay is **28 calendar days** from the original ETA. Where the expected shortfall delay exceeds 28 days, the remaining quantity must be cancelled from the PO and re-forecast rather than held on backorder.

Backorders are reconciled weekly. A backorder that has been open for more than 28 days without a delivery date is closed automatically and the quantity returned to forecast.

## §4. Customer Communications Thresholds

Where a backorder affects units already allocated to customer orders, customer communications are required when the expected delay **exceeds 7 calendar days** from the original promised date. Communications below this threshold are suppressed to avoid unnecessary contact.

Delays exceeding 14 days additionally require a proactive refund-or-wait option to be offered. Delays exceeding 21 days require automatic cancellation and refund of the affected customer orders.

## §5. Interaction with Forecast

A backorder does not release forecast demand. The forecast for the affected SKU remains committed until the backorder is either fulfilled or closed under §3. Planners must not re-forecast against a quantity that is still held on an open backorder, as this double-counts demand.
