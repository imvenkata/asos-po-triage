"""PO and forecast lookup tools.

Deliberate design choice: `get_po` returns pre-computed variance alongside the
raw record. LLMs are unreliable at arithmetic and highly reliable at reading a
policy and applying a stated number, so the split is - Python computes the
figures, the model interprets the policy. This removes an entire class of silent
failure (a plausible-looking recommendation resting on a miscalculated
percentage) without weakening the model's actual job.
"""
from __future__ import annotations

from typing import Any
from decimal import Decimal

from ..data_access import PoRepository
from ..models import PurchaseOrder


def compute_variance(po: PurchaseOrder) -> dict[str, Any]:
    qty_var = (
        float(Decimal(po.ordered_qty - po.confirmed_qty) / Decimal(po.ordered_qty) * 100) if po.ordered_qty else None
    )
    val_var = (
        float((Decimal(str(po.original_value_gbp)) - Decimal(str(po.value_gbp))) / Decimal(str(po.original_value_gbp)) * 100)
        if po.original_value_gbp
        else None
    )
    eta_slip = (po.eta - po.original_eta).days
    return {
        "qty_variance_pct": qty_var,
        "value_variance_pct": val_var,
        "baseline_valid": qty_var is not None and val_var is not None,
        "eta_slip_days": eta_slip,
        "unfulfilled_qty": po.ordered_qty - po.confirmed_qty,
        "supplier_over_confirmed": po.confirmed_qty > po.ordered_qty,
        "remainder_has_firm_date": po.remainder_eta is not None,
        "note": (
            "Percentages are computed from the PO record, not estimated. "
            "Positive variance means the supplier confirmed less than ordered."
        ),
    }


def get_po(repo: PoRepository, po_id: str) -> dict[str, Any]:
    po = repo.get_po(po_id)
    if po is None:
        return {
            "po_id": po_id,
            "error": f"No purchase order found with id '{po_id}'.",
        }
    payload = po.model_dump(mode="json")
    payload["computed_variance"] = compute_variance(po)
    children = repo.children_of(po.po_id)
    payload["child_po_ids"] = [c.po_id for c in children]
    payload["child_po_count"] = len(children)
    return payload


def get_forecast(repo: PoRepository, sku: str) -> dict[str, Any]:
    forecast = repo.get_forecast(sku)
    if forecast is None:
        return {"sku": sku, "error": f"No forecast found for SKU '{sku}'."}
    return forecast.model_dump(mode="json")
