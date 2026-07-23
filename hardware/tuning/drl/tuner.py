"""Candidate tuners used by the DRL collection and hardware validation workflows."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from ..models import HardwarePidCandidate, IterationRecord, SearchSpace
from ..search import _next_higher_bandwidth_candidate
from .common import candidate_from_mapping, candidate_key


class PlannedCandidateTuner:
    """Replay a persisted, safety-screened hardware collection plan serially."""

    def __init__(
        self,
        plan_path: Path,
        history: list[IterationRecord] | None = None,
        search: SearchSpace | None = None,
    ):
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"DRL collection plan could not be loaded: {plan_path}") from exc
        items = plan.get("candidates") if isinstance(plan, dict) else None
        if not isinstance(items, list) or not items:
            raise RuntimeError(f"DRL collection plan '{plan_path}' contains no candidates.")
        self.plan_id = str(plan.get("plan_id") or plan_path.parent.name)
        self._items = [item for item in items if isinstance(item, dict) and isinstance(item.get("candidate"), dict)]
        self._dynamic_confirmation = bool(plan.get("dynamic_confirmation"))
        self._confirmation_count = max(1, int(plan.get("confirmation_count") or 3))
        self._bandwidth_climb = bool(plan.get("bandwidth_climb_after_confirmation"))
        self._search = search
        consumed_plan_indexes = [
            int(record.optimizer_metadata.get("plan_index") or 0)
            for record in (history or [])
            if str(record.optimizer_metadata.get("collection_plan_id") or "") == self.plan_id
        ]
        legacy_offset = min(len(history or []), len(self._items)) if not consumed_plan_indexes else 0
        self._index = min(max(consumed_plan_indexes, default=legacy_offset), len(self._items))
        self._last_metadata: dict[tuple[Any, ...], dict[str, Any]] = {}

    def next_candidate(
        self,
        history: list[IterationRecord],
        best: IterationRecord | None,
    ) -> HardwarePidCandidate | None:
        _ = best
        if self._dynamic_confirmation and history:
            last = history[-1]
            confirmation = _confirmation_streak(history)
            if last.metrics.passed and last.candidate is not None and confirmation < self._confirmation_count:
                candidate = replace(last.candidate, phase="drl_targeted_confirm")
                self._remember_dynamic(
                    candidate,
                    "pass_confirmation",
                    {
                        "confirmation_attempt": confirmation + 1,
                        "confirmation_required": self._confirmation_count,
                    },
                )
                return candidate
            if confirmation >= self._confirmation_count and self._bandwidth_climb and self._search is not None:
                seen = {candidate_key(record.candidate) for record in history if record.candidate is not None}
                candidate = _next_higher_bandwidth_candidate(history, self._search, seen)
                if candidate is not None:
                    candidate = replace(candidate, phase="drl_targeted_bandwidth_climb")
                    self._remember_dynamic(
                        candidate,
                        "bandwidth_climb",
                        {
                            "confirmation_required": self._confirmation_count,
                            "bandwidth_climb_from": last.candidate.mod0_ll_bw if last.candidate else None,
                        },
                    )
                    return candidate
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

    def _remember_dynamic(self, candidate: HardwarePidCandidate, source: str, extra: dict[str, Any]) -> None:
        self._last_metadata[candidate_key(candidate)] = {
            "algorithm": "drl-collection",
            "collection_plan_id": self.plan_id,
            "proposal_source": source,
            "dynamic_insertion": True,
            **extra,
        }

    def metadata_for(self, candidate: HardwarePidCandidate) -> dict[str, Any]:
        return dict(
            self._last_metadata.get(
                candidate_key(candidate),
                {"algorithm": "drl-collection", "collection_plan_id": self.plan_id},
            )
        )


def _confirmation_streak(history: list[IterationRecord]) -> int:
    if not history or not history[-1].metrics.passed or history[-1].candidate is None:
        return 0
    key = candidate_key(history[-1].candidate)
    streak = 0
    for record in reversed(history):
        if not record.metrics.passed or record.candidate is None or candidate_key(record.candidate) != key:
            break
        streak += 1
    return streak
