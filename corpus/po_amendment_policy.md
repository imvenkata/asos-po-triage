# PO Amendment Policy

Owner: Merchandising Operations. Version 4.2. Applies to all Retail and Wholesale purchase orders raised in the Merch Planning system.

## §1. Scope and Definitions

This policy governs when an existing purchase order (PO) may be amended in place versus cancelled and re-raised as a new PO.

"Cumulative variance" means the absolute difference between the original committed order value and the latest supplier-confirmed order value, expressed as a percentage of the original committed order value. Quantity variance is measured separately under `variance_detection_sop.md`.

"Amendment in place" means mutating the existing PO record while retaining the original `po_id`, audit trail, and supplier commitment date. "Cancel and re-raise" means voiding the PO and issuing a new `po_id`, which resets the supplier commitment date and requires fresh supplier acknowledgement.

An amendment must never change the `channel` of a PO. A PO that needs to move between Retail and Wholesale must be split under `child_po_split_rules.md`.

## §2. Amendment Thresholds

A PO may be amended in place where **the cumulative variance does not exceed 15% of the original committed order value**. Amendments within this band are considered routine and do not reset the supplier commitment date.

Amendments in place are permitted for the following mutations only:

- Adjustment of `confirmed_qty` downward or upward against the original `ordered_qty`.
- Revision of the ETA by up to 20 calendar days from the original ETA.
- Correction of unit cost where the resulting value change remains inside the variance band above.

A PO in status `cancelled`, `closed`, or `invoiced` may not be amended under any circumstances and must be re-raised.

## §3. Sign-off Requirements

Sign-off scales with the absolute value of the amended PO, not the size of the variance:

| Amended PO value (GBP) | Required sign-off |
| --- | --- |
| Up to 50,000 | Merch Planner (self-approve) |
| Above 50,000 | Head of Buying |

Any PO with an amended value **above £50,000 must be escalated for Head of Buying sign-off before the amendment is applied**. Planners must not self-approve above this ceiling regardless of how small the variance is.

Where an amendment would breach a supplier's contractual minimum order quantity, sign-off escalates one level above the band in the table.

## §4. Cancel and Re-raise Conditions

A PO must be cancelled and re-raised, rather than amended, where any of the following hold:

- **The cumulative variance exceeds 10% of the original committed order value.** In-place amendment is not permitted beyond this point because the original supplier commitment is no longer considered representative.
- The supplier of record changes.
- The delivery ETA moves by more than 20 calendar days from the original ETA.
- The PO has already been amended three times.

Cancel-and-re-raise always requires Senior Merch Planner sign-off in addition to the value-band sign-off in §3.

## §5. Firm Planned Orders

Where a PO is still in status `planned` and has not been transmitted to the supplier, no amendment is required. The planner should firm the planned order at the corrected quantity and value, which supersedes the planned record without generating an amendment event.

Firming a planned order is not permitted once supplier acknowledgement has been received; from that point the PO is live and §2 to §4 apply.
