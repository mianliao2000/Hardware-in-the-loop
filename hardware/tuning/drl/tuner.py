"""Candidate tuners used by the DRL collection and hardware validation workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models import HardwarePidCandidate, IterationRecord
from .common import candidate_from_mapping, candidate_key


class PlannedCandidateTuner:
    """Replay a persisted, safety-screened hardware collection plan serially."""

    def __init__(self, plan_path: Path, history: list[IterationRecord] | None = None):
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"DRL collection plan could not be loaded: {plan_path}") from exc
        items = plan.get("candidates") if isinstance(plan, dict) else None
        if not isinstance(items, list) or not items:
            raise RuntimeError(f"DRL collection plan '{plan_path}' contains no candidates.")
        self.plan_id = str(plan.get("plan_id") or plan_path.parent.name)
        self._items = [item for item in items if isinstance(item, dict) and isinstance(item.get("candidate"), dict)]
        self._index = min(len(history or []), len(self._items))
        self._last_metadata: dict[tuple[Any, ...], dict[str, Any]] = {}

    def next_candidate(
        self,
        history: list[IterationRecord],
        best: IterationRecord | None,
    ) -> HardwarePidCandidate | None:
        _ = best
        self._index = max(self._index, min(len(history), len(self._items)))
        if self._index >= len(self._items):
            return None
        item = self._items[self._index]
        self._index += 1
        candidate = candidate_from_mapping(item["candidate"], phase=str(item["candidate"].get("phase") or "drl_collection"))
        metadata = dict(item.get("optimizer_metadata") or {})
        metadata.update(
            {
                "algorithm": "drl-collection",
                "collection_plan_id": self.plan_id,
                "plan_index": int(item.get("index") or self._index),
            }
        )
        self._last_metadata[candidate_key(candidate)] = metadata
        return candidate

    def metadata_for(self, candidate: HardwarePidCandidate) -> dict[str, Any]:
        return dict(
            self._last_metadata.get(
                candidate_key(candidate),
                {"algorithm": "drl-collection", "collection_plan_id": self.plan_id},
            )
        )
