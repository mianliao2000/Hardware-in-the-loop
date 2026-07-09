"""Grid-refine search for PID autotuning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

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
        if len(history) >= _total_iteration_budget(self.search):
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

    COORDINATE_COARSE_FRACTION = 0.60

    def __init__(self, search: SearchSpace):
        self.search = search
        self._queue: list[HardwarePidCandidate] = [self._center_candidate("baseline")]
        self._coordinate_loaded = False
        self._pairwise_loaded = False
        self._local_pass_index = 0
        self._seen: set[tuple] = set()

    def next_candidate(self, history: list[IterationRecord], best: IterationRecord | None) -> HardwarePidCandidate | None:
        if len(history) >= _total_iteration_budget(self.search):
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
                remaining = self._remaining_coordinate_budget(history)
                if remaining > 0:
                    self._queue.extend(self._coordinate_grid()[:remaining])
                continue

            if not self._pairwise_loaded and best is not None:
                self._pairwise_loaded = True
                remaining = self._remaining_pairwise_coarse_budget(history)
                if remaining > 0:
                    fresh = _dedupe_candidates(self._pairwise_coarse(history, best), self._seen)
                    self._queue.extend(fresh[:remaining])
                continue

            if best is not None and self._remaining_refined_budget(history) > 0:
                local_loaded = self._load_next_local_pass(history, best)
                if local_loaded:
                    continue

            return None

    def next_candidate_batch(
        self,
        history: list[IterationRecord],
        best: IterationRecord | None,
        max_count: int,
    ) -> list[HardwarePidCandidate]:
        """Return a small batch of compatible candidates for coarse hardware scans.

        The first candidate decides the batch phase. Baseline, pairwise, and local
        refine stay single-candidate so adaptive tuning still waits for the latest
        result before choosing the next point. Coordinate candidates are independent
        enough to be measured in small batches.
        """

        remaining = max(0, _total_iteration_budget(self.search) - len(history))
        limit = max(1, min(int(max_count), remaining))
        first = self.next_candidate(history, best)
        if first is None:
            return []
        if first.phase != "coordinate" or limit == 1:
            return [first]

        candidates = [first]
        while len(candidates) < limit:
            candidate = self.next_candidate(history, best)
            if candidate is None:
                break
            if candidate.phase != first.phase:
                self._queue.insert(0, candidate)
                self._seen.discard(hardware_candidate_key(candidate))
                break
            candidates.append(candidate)
        return candidates

    def _center_candidate(self, phase: str) -> HardwarePidCandidate:
        values = {
            name: _typed_value(name, parameter.clamped(parameter.center))
            for name in HARDWARE_TUNING_FIELD_NAMES
            for parameter in [_range_for(self.search, name)]
        }
        return HardwarePidCandidate(**values, phase=phase)

    def _coordinate_grid(self) -> list[HardwarePidCandidate]:
        center = self._center_candidate("coordinate")
        value_sets = {
            name: _range_values(_range_for(self.search, name), name)
            for name in HARDWARE_TUNING_FIELD_NAMES
        }
        return _sample_coordinate_combinations(center, value_sets, max(0, _coarse_iteration_budget(self.search) - 1))

    def _pairwise_coarse(self, history: list[IterationRecord], best: IterationRecord) -> list[HardwarePidCandidate]:
        base = best.candidate or self._center_candidate("pairwise_coarse")
        ranked_fields = _rank_coordinate_fields(history)
        if not ranked_fields:
            ranked_fields = list(HARDWARE_TUNING_FIELD_NAMES)
        fields = list(dict.fromkeys([*ranked_fields, *HARDWARE_TUNING_FIELD_NAMES]))
        candidates: list[HardwarePidCandidate] = []
        for index, first in enumerate(fields):
            for second in fields[index + 1:]:
                for first_value in _range_values(_range_for(self.search, first), first):
                    for second_value in _range_values(_range_for(self.search, second), second):
                        candidate = _candidate_with(base, first, first_value, "pairwise_coarse")
                        candidates.append(_candidate_with(candidate, second, second_value, "pairwise_coarse"))
        return candidates

    def _load_next_local_pass(self, history: list[IterationRecord], best: IterationRecord) -> bool:
        max_passes = max(1, _refined_iteration_budget(self.search) or _total_iteration_budget(self.search))
        while self._local_pass_index < max_passes:
            remaining = self._remaining_refined_budget(history)
            if remaining <= 0:
                return False
            candidates = self._local_refine(best, self._local_pass_index)
            self._local_pass_index += 1
            fresh = _dedupe_candidates(candidates, self._seen)
            if fresh:
                self._queue.extend(fresh[:remaining])
                return True
        return False

    def _remaining_coordinate_budget(self, history: list[IterationRecord]) -> int:
        coordinate_budget, _ = _coarse_phase_budgets(self.search)
        used = sum(1 for record in history if record.phase == "coordinate")
        queued = sum(1 for candidate in self._queue if candidate.phase == "coordinate")
        return max(0, coordinate_budget - used - queued)

    def _remaining_pairwise_coarse_budget(self, history: list[IterationRecord]) -> int:
        _, pairwise_budget = _coarse_phase_budgets(self.search)
        used = sum(1 for record in history if record.phase == "pairwise_coarse")
        queued = sum(1 for candidate in self._queue if candidate.phase == "pairwise_coarse")
        return max(0, pairwise_budget - used - queued)

    def _remaining_refined_budget(self, history: list[IterationRecord]) -> int:
        local_budget = _refined_iteration_budget(self.search)
        used = sum(1 for record in history if record.phase == "local_refine")
        queued = sum(1 for candidate in self._queue if candidate.phase == "local_refine")
        return max(0, local_budget - used - queued)

    def _local_refine(self, best: IterationRecord, pass_index: int = 0) -> list[HardwarePidCandidate]:
        base = best.candidate or self._center_candidate("local")
        candidates: list[HardwarePidCandidate] = []
        divisor = 2.0 ** (pass_index + 1)
        for name in HARDWARE_TUNING_FIELD_NAMES:
            parameter = _range_for(self.search, name)
            step = _parameter_step(parameter, name) / divisor
            if _is_int_field(name):
                step = max(1.0, step)
            center = getattr(base, name)
            for value in (
                _typed_value(name, parameter.clamped(center - step)),
                _typed_value(name, parameter.clamped(center + step)),
            ):
                candidates.append(_candidate_with(base, name, value, "local_refine"))
        return candidates


def select_best_result(records: list[IterationRecord]) -> IterationRecord | None:
    if not records:
        return None
    valid = [record for record in records if not _is_invalid_hardware_record(record)]
    passing = [record for record in valid if record.metrics.passed]
    pool = passing or valid or records
    return min(pool, key=_record_priority)


def _total_iteration_budget(search: SearchSpace) -> int:
    total = getattr(search, "total_iteration_budget", None)
    if callable(total):
        return total()
    return max(1, int(getattr(search, "max_iterations", 40)))


def _coarse_iteration_budget(search: SearchSpace) -> int:
    coarse = getattr(search, "coarse_iteration_budget", None)
    if callable(coarse):
        return coarse()
    return max(1, int(getattr(search, "max_iterations", 40)))


def _refined_iteration_budget(search: SearchSpace) -> int:
    refined = getattr(search, "refined_iteration_budget", None)
    if callable(refined):
        return refined()
    return max(0, int(getattr(search, "max_iterations", 40)) - _coarse_iteration_budget(search))


def _coarse_phase_budgets(search: SearchSpace) -> tuple[int, int]:
    # Baseline consumes one coarse slot. The remaining coarse budget is split
    # between independent coordinate combinations and best-based pairwise coarse.
    coarse_remaining = max(0, _coarse_iteration_budget(search) - 1)
    if coarse_remaining <= 0:
        return (0, 0)
    coordinate_budget = int(math.ceil(coarse_remaining * HardwareGridHeuristicTuner.COORDINATE_COARSE_FRACTION))
    coordinate_budget = min(max(0, coordinate_budget), coarse_remaining)
    return (coordinate_budget, coarse_remaining - coordinate_budget)


def hardware_candidate_key(candidate: HardwarePidCandidate) -> tuple:
    return (
        int(candidate.mod0_kp),
        int(candidate.mod0_ki),
        int(candidate.mod0_kd),
        int(candidate.mod0_kpole1),
        int(candidate.mod0_kpole2),
        int(candidate.mod0_cm_gain),
        round(float(candidate.output_inductance_nh), 6),
        round(float(candidate.effective_lc_inductance_nh), 6),
    )


def _dedupe_candidates(candidates: list[HardwarePidCandidate], seen: set[tuple]) -> list[HardwarePidCandidate]:
    fresh: list[HardwarePidCandidate] = []
    local_seen: set[tuple] = set()
    for candidate in candidates:
        key = hardware_candidate_key(candidate)
        if key in seen or key in local_seen:
            continue
        local_seen.add(key)
        fresh.append(candidate)
    return fresh


def _sample_coordinate_combinations(
    center: HardwarePidCandidate,
    value_sets: dict[str, list[float]],
    target_count: int,
) -> list[HardwarePidCandidate]:
    """Sample multi-parameter coarse-grid combinations without exploding runtime."""

    if target_count <= 0:
        return []
    fields = [name for name in HARDWARE_TUNING_FIELD_NAMES if len(value_sets.get(name, [])) > 1]
    if not fields:
        return []

    total = 1
    for name in fields:
        total *= len(value_sets[name])

    candidates: list[HardwarePidCandidate] = []
    local_seen: set[tuple] = set()

    def add_from_values(values_by_field: dict[str, float]) -> None:
        nonlocal candidates
        values = {field: getattr(center, field) for field in HARDWARE_TUNING_FIELD_NAMES}
        for field, value in values_by_field.items():
            if field in {"mod0_kpole1", "mod0_kpole2"}:
                kpole_value = _kpole_pair_value(float(value))
                values["mod0_kpole1"] = kpole_value
                values["mod0_kpole2"] = kpole_value
            else:
                values[field] = _typed_value(field, value)
        candidate = HardwarePidCandidate(**values, phase="coordinate")
        key = hardware_candidate_key(candidate)
        if key == hardware_candidate_key(center) or key in local_seen:
            return
        local_seen.add(key)
        candidates.append(candidate)

    lengths = [len(value_sets[name]) for name in fields]
    target = min(target_count, total - 1)
    if target <= 0:
        return []

    stride = max(1, total // target)
    while math.gcd(stride, total) != 1:
        stride += 1

    attempts = 0
    index = 0
    max_attempts = min(total, max(target_count * 100, 500))
    while len(candidates) < target_count and attempts < max_attempts:
        combo_indexes = _mixed_radix_indexes(index, lengths)
        values = {
            name: value_sets[name][combo_indexes[field_index]]
            for field_index, name in enumerate(fields)
        }
        add_from_values(values)
        index = (index + stride) % total
        attempts += 1

    return candidates


def _mixed_radix_indexes(index: int, lengths: list[int]) -> list[int]:
    indexes = [0] * len(lengths)
    for position in range(len(lengths) - 1, -1, -1):
        length = max(1, lengths[position])
        indexes[position] = index % length
        index //= length
    return indexes


def _record_priority(record: IterationRecord) -> tuple[float, float, float, int, float, int]:
    metrics = record.metrics
    if _is_invalid_hardware_record(record):
        return (2.0, metrics.score, max(metrics.overshoot_pct, metrics.undershoot_pct), metrics.oscillations, metrics.settling_time_s, record.iteration)
    if metrics.passed:
        balance = metrics.overshoot_pct**2 + metrics.undershoot_pct**2
        worst = max(metrics.overshoot_pct, metrics.undershoot_pct)
        return (0.0, balance, worst, metrics.oscillations, metrics.settling_time_s, record.iteration)
    return (1.0, metrics.score, max(metrics.overshoot_pct, metrics.undershoot_pct), metrics.oscillations, metrics.settling_time_s, record.iteration)


def _is_invalid_hardware_record(record: IterationRecord) -> bool:
    reasons = " ".join(str(item).lower() for item in (record.metrics.pass_reasons or []))
    if (
        "transient protection skipped" in reasons
        or "protection skipped" in reasons
        or "invalid bode" in reasons
        or "duplicate 0 db crossover" in reasons
        or "second 0 db crossover" in reasons
    ):
        return True
    return math.isfinite(record.metrics.score) and record.metrics.score >= 1.0e6


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
    typed_value = _typed_value(name, _range_value_for_base(base, name, value))
    if name in {"mod0_kpole1", "mod0_kpole2"}:
        kpole_value = _kpole_pair_value(float(typed_value))
        values["mod0_kpole1"] = kpole_value
        values["mod0_kpole2"] = kpole_value
    else:
        values[name] = typed_value
    return HardwarePidCandidate(**values, phase=phase)


def _kpole_pair_value(value: float) -> int:
    return 3 if abs(value - 3) <= abs(value - 6) else 6


def _range_value_for_base(base: HardwarePidCandidate, name: str, value: float) -> float:
    # Candidate helpers already clamp through SearchParameter in normal paths.
    # This fallback keeps direct candidate construction conservative.
    _ = base
    if name in {"mod0_kp", "mod0_ki", "mod0_kd"}:
        return min(max(value, 0), 255)
    if name in {"mod0_kpole1", "mod0_kpole2"}:
        return min(max(value, 0), 15)
    if name == "mod0_cm_gain":
        return min(max(value, 0), 127)
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
