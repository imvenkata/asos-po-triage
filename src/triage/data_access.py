"""Repository over the mock PO / forecast files.

Kept behind a small interface so the tools do not care whether the rows come
from JSON on disk or the Merch Planning API. Loaded once and cached.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .config import Settings, get_settings
from .models import Forecast, PurchaseOrder


class PoRepository:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        po_path = self._data_dir / "purchase_orders.json"
        fc_path = self._data_dir / "forecasts.json"
        for path in (po_path, fc_path):
            if not path.exists():
                raise FileNotFoundError(f"Required data file missing: {path}")

        self._pos: dict[str, PurchaseOrder] = {
            row["po_id"]: PurchaseOrder(**row)
            for row in json.loads(po_path.read_text(encoding="utf-8"))
        }
        self._forecasts: dict[str, Forecast] = {
            row["sku"]: Forecast(**row) for row in json.loads(fc_path.read_text(encoding="utf-8"))
        }

    def get_po(self, po_id: str) -> PurchaseOrder | None:
        return self._pos.get((po_id or "").strip().upper())

    def get_forecast(self, sku: str) -> Forecast | None:
        return self._forecasts.get((sku or "").strip().upper())

    def children_of(self, po_id: str) -> list[PurchaseOrder]:
        return [p for p in self._pos.values() if p.parent_po_id == po_id]

    @property
    def po_ids(self) -> list[str]:
        return sorted(self._pos)


@lru_cache(maxsize=4)
def _repository_for(data_dir: Path) -> PoRepository:
    return PoRepository(data_dir)


def get_repository(settings: Settings | None = None) -> PoRepository:
    """Cached on the data directory, not on Settings (which is not hashable)."""
    return _repository_for((settings or get_settings()).data_dir)
