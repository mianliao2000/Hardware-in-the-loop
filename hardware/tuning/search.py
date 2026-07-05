"""Grid-refine search for PID autotuning."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import HARDWARE_TUNING_FIELD_NAMES, HardwarePidCandidate, IterationRecord, SearchParameter, SearchSpace


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


class HardwareGridHeuristicTuner:
    """Generate raw hardware candidates with local grid + heuristic refinement."""

    def __init__(self, search: SearchSpace):
        self.search = search
        self._queue: list[HardwarePidCandidate] = [self._center_candidate("baseline")]
        self._coordinate_loaded = False
        self._pairwise_loaded = False
        self._local_loaded = False
        self._seen: set[tuple] = set()

    def next_candidate(self, history: list[IterationRecord], best: IterationRecord | None) -> HardwarePidCandidate | None:
        if len(history) >= self.search.max_iterations:
            return None
        for record in history:
            if record.candidate is not None:
                self._seen.add(hardware_candidate_key(record.candidate))

        while True:
            if self._queue:
                candidate = self._queue.pop(0)
                key = hardware_candidate_key(candidate)
                if key in self._seen:
                    continue
                self._seen.add(key)
                return candidate

            if not self._coordinate_loaded:
                self._coordinate_loaded = True
                self._queue.extend(self._coordinate_grid())
                continue

            if not self._pairwise_loaded and best is not None:
                self._pairwise_loaded = True
                self._queue.extend(self._pairwise_refine(history, best))
                continue

            if not self._local_loaded and best is not None:
                self._local_loaded = True
                self._queue.extend(self._local_refine(best))
                continue

            return None

    def _center_candidate(self, phase: str) -> HardwarePidCandidate:
        values = {
            name: _typed_value(name, parameter.clamped(parameter.center))
            for name in HARDWARE_TUNING_FIELD_NAMES
            for parameter in [_range_for(self.search, name)]
        }
        return HardwarePidCandidate(**values, phase=phase)

    def _coordinate_grid(self) -> list[HardwarePidCandidate]:
        center = self._center_candidate("coordinate")
        candidates: list[HardwarePidCandidate] = []
        for name in HARDWARE_TUNING_FIELD_NAMES:
            for value in _range_values(_range_for(self.search, name), name):
                if _typed_value(name, value) == getattr(center, name):
                    continue
                candidates.append(_candidate_with(center, name, value, "coordinate"))
        return candidates

    def _pairwise_refine(self, history: list[IterationRecord], best: IterationRecord) -> list[HardwarePidCandidate]:
        base = best.candidate or self._center_candidate("pairwise")
        ranked_fields = _rank_coordinate_fields(history)
        if not ranked_fields:
            ranked_fields = list(HARDWARE_TUNING_FIELD_NAMES[:4])
        fields = ranked_fields[:4]
        candidates: list[HardwarePidCandidate] = []
        for index, first in enumerate(fields):
            for second in fields[index + 1:]:
                for first_value in _neighbor_values(_range_for(self.search, first), getattr(base, first), first):
                    for second_value in _neighbor_values(_range_for(self.search, second), getattr(base, second), second):
                        candidate = _candidate_with(base, first, first_value, "pairwise_refine")
                        candidates.append(_candidate_with(candidate, second, second_value, "pairwise_refine"))
        return candidates

    def _local_refine(self, best: IterationRecord) -> list[HardwarePidCandidate]:
        base = best.candidate or self._center_candidate("local")
        candidates: list[HardwarePidCandidate] = []
        for name in HARDWARE_TUNING_FIELD_NAMES:
            parameter = _range_for(self.search, name)
            step = _parameter_step(parameter, name) / 2.0
            if _is_int_field(name):
                step = max(1.0, step)
            local_parameter = SearchParameter(
                center=getattr(base, name),
                min=parameter.min,
                max=parameter.max,
                step=step,
                points=3,
            )
            for value in _neighbor_values(local_parameter, getattr(base, name), name):
                candidates.append(_candidate_with(base, name, value, "local_refine"))
        return candidates


def select_best_result(records: list[IterationRecord]) -> IterationRecord | None:
    if not records:
        return None
    passing = [record for record in records if record.metrics.passed]
    pool = passing or records
    return min(pool, key=_record_priority)


def hardware_candidate_key(candidate: HardwarePidCandidate) -> tuple:
    return (
        int(candidate.mod0_kp),
        int(candidate.mod0_ki),
        int(candidate.mod0_kd),
        int(candidate.mod0_kpole1),
        int(candidate.mod0_kpole2),
        round(float(candidate.output_inductance_nh), 6),
        round(float(candidate.effective_lc_inductance_nh), 6),
    )


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


def _range_for(search: SearchSpace, name: str) -> SearchParameter:
    value = getattr(search, name)
    if not isinstance(value, SearchParameter):
        raise ValueError(f"Search field {name} is not a SearchParameter.")
    return value


def _range_values(parameter: SearchParameter, name: str) -> list[float]:
    points = max(1, int(round(getattr(parameter, "points", 0) or 0)))
    if points <= 1:
        return [_typed_value(name, parameter.center)]
    values: list[float] = [parameter.center]
    if points == 2:
        values.extend([parameter.min, parameter.max])
    else:
        step = (parameter.max - parameter.min) / (points - 1)
        values.extend(parameter.min + step * index for index in range(points))
    deduped: list[float] = []
    seen: set[float | int] = set()
    for value in values:
        typed = _typed_value(name, parameter.clamped(value))
        if typed in seen:
            continue
        seen.add(typed)
        deduped.append(typed)
    return deduped


def _neighbor_values(parameter: SearchParameter, center: float, name: str) -> list[float]:
    step = _parameter_step(parameter, name)
    return [
        _typed_value(name, parameter.clamped(center - step)),
        _typed_value(name, parameter.clamped(center + step)),
    ]


def _parameter_step(parameter: SearchParameter, name: str) -> float:
    points = max(2, int(round(getattr(parameter, "points", 0) or 0)))
    span = max(0.0, parameter.max - parameter.min)
    if span > 0:
        step = span / (points - 1)
    else:
        step = parameter.step
    if _is_int_field(name):
        return max(1.0, round(step))
    return max(float(step), 1e-12)


def _candidate_with(base: HardwarePidCandidate, name: str, value: float, phase: str) -> HardwarePidCandidate:
    values = {field: getattr(base, field) for field in HARDWARE_TUNING_FIELD_NAMES}
    values[name] = _typed_value(name, _range_value_for_base(base, name, value))
    return HardwarePidCandidate(**values, phase=phase)


def _range_value_for_base(base: HardwarePidCandidate, name: str, value: float) -> float:
    # Candidate helpers already clamp through SearchParameter in normal paths.
    # This fallback keeps direct candidate construction conservative.
    _ = base
    if name in {"mod0_kp", "mod0_ki", "mod0_kd"}:
        return min(max(value, 0), 255)
    if name in {"mod0_kpole1", "mod0_kpole2"}:
        return min(max(value, 0), 15)
    return value


def _typed_value(name: str, value: float) -> int | float:
    if _is_int_field(name):
        return int(round(value))
    return float(value)


def _is_int_field(name: str) -> bool:
    return name.startswith("mod0_")


def _rank_coordinate_fields(history: list[IterationRecord]) -> list[str]:
    center_record = next((item for item in history if item.candidate and item.candidate.phase == "baseline"), None)
    if center_record is None or center_record.candidate is None:
        return []
    center = center_record.candidate
    best_by_field: dict[str, float] = {}
    for record in history:
        if record.candidate is None or record.candidate.phase != "coordinate":
            continue
        changed = [name for name in HARDWARE_TUNING_FIELD_NAMES if getattr(record.candidate, name) != getattr(center, name)]
        if len(changed) != 1:
            continue
        field = changed[0]
        best_by_field[field] = min(best_by_field.get(field, float("inf")), record.metrics.score)
    return [name for name, _ in sorted(best_by_field.items(), key=lambda item: item[1])]
