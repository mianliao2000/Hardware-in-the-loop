"""Grid-refine search for PID autotuning."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import IterationRecord, SearchSpace


@dataclass(frozen=True)
class TuningCandidate:
    phase: str
    wc_rad_s: float
    phi_deg: float


class GridRefinePidTuner:
    """Generate candidates using the reference project's multi-stage pattern."""

    def __init__(self, search: SearchSpace):
        self.search = search
        self._queue: list[TuningCandidate] = [
            TuningCandidate("bootstrap", search.initial_wc_rad_s, search.initial_phi_deg)
        ]
        self._coarse_loaded = False
        self._local_loaded = False
        self._post_loaded = False

    def next_candidate(self, history: list[IterationRecord], best: IterationRecord | None) -> TuningCandidate | None:
        if len(history) >= self.search.max_iterations:
            return None
        if self._queue:
            return self._queue.pop(0)

        if not self._coarse_loaded:
            self._coarse_loaded = True
            self._queue.extend(self._coarse_grid())
            return self.next_candidate(history, best)

        if not self._local_loaded and best is not None:
            self._local_loaded = True
            self._queue.extend(self._local_grid(best, scale=0.12, phi_span=8.0, phase="local_refine"))
            return self.next_candidate(history, best)

        if not self._post_loaded and any(item.metrics.passed for item in history):
            self._post_loaded = True
            passing_best = select_best_result([item for item in history if item.metrics.passed])
            self._queue.extend(self._local_grid(passing_best, scale=0.06, phi_span=4.0, phase="post_pass_fine"))
            return self.next_candidate(history, best)

        return None

    def _coarse_grid(self) -> list[TuningCandidate]:
        wc_values = _logspace(self.search.wc_min_rad_s, self.search.wc_max_rad_s, 5)
        phi_values = _linspace(self.search.phi_min_deg, self.search.phi_max_deg, 4)
        return [TuningCandidate("coarse_grid", wc, phi) for wc in wc_values for phi in phi_values]

    def _local_grid(self, center: IterationRecord, scale: float, phi_span: float, phase: str) -> list[TuningCandidate]:
        wc_values = _linspace(center.wc_rad_s * (1.0 - scale), center.wc_rad_s * (1.0 + scale), 5)
        phi_values = _linspace(center.phi_deg - phi_span, center.phi_deg + phi_span, 5)
        candidates: list[TuningCandidate] = []
        for wc in wc_values:
            clamped_wc = min(max(wc, self.search.wc_min_rad_s), self.search.wc_max_rad_s)
            for phi in phi_values:
                clamped_phi = min(max(phi, self.search.phi_min_deg), self.search.phi_max_deg)
                candidates.append(TuningCandidate(phase, clamped_wc, clamped_phi))
        return candidates


def select_best_result(records: list[IterationRecord]) -> IterationRecord | None:
    if not records:
        return None
    passing = [record for record in records if record.metrics.passed]
    pool = passing or records
    return min(pool, key=_record_priority)


def _record_priority(record: IterationRecord) -> tuple[float, float, float, int, float, int]:
    metrics = record.metrics
    if metrics.passed:
        balance = metrics.overshoot_pct**2 + metrics.undershoot_pct**2
        worst = max(metrics.overshoot_pct, metrics.undershoot_pct)
        return (0.0, balance, worst, metrics.oscillations, metrics.settling_time_s, record.iteration)
    return (1.0, metrics.score, max(metrics.overshoot_pct, metrics.undershoot_pct), metrics.oscillations, metrics.settling_time_s, record.iteration)


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [start + step * index for index in range(count)]


def _logspace(start: float, stop: float, count: int) -> list[float]:
    log_start = math.log(start)
    log_stop = math.log(stop)
    return [math.exp(value) for value in _linspace(log_start, log_stop, count)]
