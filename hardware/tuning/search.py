"""Grid-refine search for PID autotuning."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from dataclasses import replace

from .models import (
    HARDWARE_TUNING_FIELD_NAMES,
    HardwarePidCandidate,
    IterationRecord,
    SearchParameter,
    SearchSpace,
    hardware_candidate_key as model_hardware_candidate_key,
)


BASIN_QUALITY_POOL_MULTIPLIER = 6
CONFIRMED_PASS_COUNT = 3


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

        confirmation = _confirmation_streak_for_last(history)
        if history and history[-1].metrics.passed and 0 < confirmation < CONFIRMED_PASS_COUNT:
            candidate = history[-1].candidate
            if candidate is not None:
                return replace(candidate, phase="grid_confirm")
        if confirmation >= CONFIRMED_PASS_COUNT:
            climb = _next_higher_bandwidth_candidate(history, self.search, self._seen)
            if climb is not None:
                self._seen.add(hardware_candidate_key(climb))
                return climb

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
                    self._queue.extend(self._coordinate_grid(remaining))
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

    def _center_candidate(self, phase: str) -> HardwarePidCandidate:
        values = {
            name: _typed_value(name, parameter.clamped(parameter.center))
            for name in HARDWARE_TUNING_FIELD_NAMES
            for parameter in [_range_for(self.search, name)]
        }
        return HardwarePidCandidate(**values, phase=phase)

    def _coordinate_grid(self, target_count: int | None = None) -> list[HardwarePidCandidate]:
        center = self._center_candidate("coordinate")
        value_sets = {
            name: _range_values(_range_for(self.search, name), name)
            for name in HARDWARE_TUNING_FIELD_NAMES
        }
        value_sets["mod0_ll_bw"] = sorted(value_sets["mod0_ll_bw"], reverse=True)
        if target_count is None:
            target_count = max(0, _coarse_iteration_budget(self.search) - 1)
        return _sample_coordinate_combinations(center, value_sets, max(0, target_count))

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
            values = (
                (_typed_value(name, parameter.clamped(center + step)),)
                if name == "mod0_ll_bw"
                else (
                    _typed_value(name, parameter.clamped(center - step)),
                    _typed_value(name, parameter.clamped(center + step)),
                )
            )
            for value in values:
                candidates.append(_candidate_with(base, name, value, "local_refine"))
        return candidates


def select_best_result(records: list[IterationRecord]) -> IterationRecord | None:
    if not records:
        return None
    valid = [record for record in records if not _is_invalid_hardware_record(record)]
    objective_eligible = [record for record in valid if _has_bandwidth_objective(record)]
    if objective_eligible:
        valid = objective_eligible
    confirmed_best = select_confirmed_best_result(valid)
    if confirmed_best is not None:
        return confirmed_best
    passing = [record for record in valid if record.metrics.passed]
    pool = passing or valid or records
    return min(pool, key=_record_priority)


def select_confirmed_best_result(records: list[IterationRecord]) -> IterationRecord | None:
    valid = [record for record in records if not _is_invalid_hardware_record(record)]
    objective_eligible = [record for record in valid if _has_bandwidth_objective(record)]
    if objective_eligible:
        valid = objective_eligible
    confirmed = _confirmed_candidate_groups(valid)
    if not confirmed:
        return None
    _, measurements = min(
        confirmed.items(),
        key=lambda item: _aggregate_candidate_priority(item[1]),
    )
    return _aggregate_representative(measurements)


def select_diverse_results(records: list[IterationRecord], limit: int = 5) -> list[IterationRecord]:
    """Return low-penalty representatives from distinct hardware-search basins."""

    valid = [record for record in records if record.candidate is not None and not _is_invalid_hardware_record(record)]
    # A partially migrated archive can contain both first-entry V1 labels and
    # final-entry V2 labels. Once V2 measurements exist, never let an old
    # false-fast Ts displace them in the Quality Basin panel.
    latest_settling_version = max(
        (int(record.metrics.settling_analysis_version or 0) for record in valid),
        default=0,
    )
    if latest_settling_version >= 2:
        valid = [
            record
            for record in valid
            if int(record.metrics.settling_analysis_version or 0) == latest_settling_version
        ]
    objective_eligible = [record for record in valid if _has_bandwidth_objective(record)]
    if objective_eligible:
        valid = objective_eligible
    confirmed = _confirmed_candidate_groups(valid)
    aggregate_priorities: dict[tuple, tuple] = {}
    if confirmed:
        ranked = []
        for key, measurements in confirmed.items():
            aggregate_priorities[key] = _aggregate_candidate_priority(measurements)
            ranked.append(_aggregate_representative(measurements))
        ranked.sort(key=lambda record: aggregate_priorities[hardware_candidate_key(record.candidate)])
        # Confirmed candidates own the highest ranking tier, but they should
        # not make a "Top 5" panel contain fewer than five entries. Fill the
        # remaining slots with the best single-pass candidates and then failed
        # candidates, while keeping duplicate hardware keys out.
        confirmed_keys = set(confirmed)
        ranked.extend(
            sorted(
                (
                    record
                    for record in valid
                    if hardware_candidate_key(record.candidate) not in confirmed_keys
                ),
                key=_basin_priority,
            )
        )
    else:
        ranked = sorted(valid, key=_basin_priority)
    unique_ranked: list[IterationRecord] = []
    ranked_keys: set[tuple] = set()
    for record in ranked:
        candidate = record.candidate
        if candidate is None:
            continue
        key = hardware_candidate_key(candidate)
        if key in ranked_keys:
            continue
        ranked_keys.add(key)
        unique_ranked.append(record)

    safe_limit = max(0, int(limit))
    # Diversity is useful only among reasonably good results. Restricting the
    # search to the best N penalties prevents a distant but poor region from
    # displacing a much better nearby basin.
    quality_pool_size = max(safe_limit, safe_limit * BASIN_QUALITY_POOL_MULTIPLIER)
    quality_pool = unique_ranked[:quality_pool_size]
    selected: list[IterationRecord] = []
    selected_keys: set[tuple] = set()
    for minimum_distance in (0.18, 0.10, 0.0):
        for record in quality_pool:
            if len(selected) >= safe_limit:
                return _sort_selected_basins(selected, aggregate_priorities)
            candidate = record.candidate
            if candidate is None:
                continue
            key = hardware_candidate_key(candidate)
            if key in selected_keys:
                continue
            vector = _diversity_vector(candidate)
            if minimum_distance > 0 and any(
                _euclidean_distance(vector, _diversity_vector(item.candidate)) < minimum_distance
                for item in selected
                if item.candidate is not None
            ):
                continue
            selected.append(record)
            selected_keys.add(key)
    return _sort_selected_basins(selected, aggregate_priorities)


def _confirmed_candidate_groups(
    records: list[IterationRecord],
    required: int = CONFIRMED_PASS_COUNT,
) -> dict[tuple, list[IterationRecord]]:
    """Return candidates with at least ``required`` consecutive passing measurements."""

    confirmed_keys: set[tuple] = set()
    previous_key: tuple | None = None
    streak = 0
    for record in records:
        candidate = record.candidate
        key = hardware_candidate_key(candidate) if candidate is not None else None
        if record.metrics.passed and key is not None and key == previous_key:
            streak += 1
        elif record.metrics.passed and key is not None:
            streak = 1
        else:
            streak = 0
        previous_key = key
        if key is not None and streak >= max(1, int(required)):
            confirmed_keys.add(key)

    grouped: dict[tuple, list[IterationRecord]] = {key: [] for key in confirmed_keys}
    for record in records:
        if record.candidate is None:
            continue
        key = hardware_candidate_key(record.candidate)
        if key in grouped:
            grouped[key].append(record)
    return grouped


def _aggregate_candidate_priority(records: list[IterationRecord]) -> tuple:
    scores = [float(record.metrics.score) for record in records]
    objectives = [_record_objective_score(record) for record in records]
    failure_rate = sum(not record.metrics.passed for record in records) / max(1, len(records))
    worst_transient = [max(record.metrics.overshoot_pct, record.metrics.undershoot_pct) for record in records]
    settling = [float(record.metrics.settling_time_s) for record in records]
    return (
        float(statistics.median(objectives)),
        -float(records[0].candidate.mod0_ll_bw) if records[0].candidate is not None else 0.0,
        float(statistics.median(scores)),
        failure_rate,
        max(scores),
        float(statistics.median(worst_transient)),
        float(statistics.median(settling)),
        min(record.iteration for record in records),
    )


def _aggregate_representative(records: list[IterationRecord]) -> IterationRecord:
    median_score = float(statistics.median(_record_objective_score(record) for record in records))
    return min(
        records,
        key=lambda record: (
            abs(_record_objective_score(record) - median_score),
            0 if record.metrics.passed else 1,
            _record_priority(record),
        ),
    )


def _sort_selected_basins(
    records: list[IterationRecord],
    aggregate_priorities: dict[tuple, tuple],
) -> list[IterationRecord]:
    if not aggregate_priorities:
        return sorted(records, key=_basin_priority)

    def priority(record: IterationRecord) -> tuple:
        aggregate = aggregate_priorities.get(hardware_candidate_key(record.candidate))
        if aggregate is not None:
            return (0, *aggregate)
        basin = _basin_priority(record)
        return (1 if record.metrics.passed else 2, *basin[1:])

    return sorted(
        records,
        key=priority,
    )


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
    # Baseline consumes one coarse slot. Every remaining coarse iteration samples
    # the complete multidimensional search space. Heuristic refinement has its own
    # explicit refined-iteration budget and must not truncate global coverage.
    coarse_remaining = max(0, _coarse_iteration_budget(search) - 1)
    return (coarse_remaining, 0)


def hardware_candidate_key(candidate: HardwarePidCandidate) -> tuple:
    return model_hardware_candidate_key(candidate)


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

    candidates: list[HardwarePidCandidate] = []
    local_seen: set[tuple] = set()

    def add_from_values(values_by_field: dict[str, float]) -> None:
        nonlocal candidates
        values = {field: getattr(center, field) for field in HARDWARE_TUNING_FIELD_NAMES}
        for field, value in values_by_field.items():
            values[field] = _typed_value(field, value)
        candidate = HardwarePidCandidate(**values, phase="coordinate")
        key = hardware_candidate_key(candidate)
        if key == hardware_candidate_key(center) or key in local_seen:
            return
        local_seen.add(key)
        candidates.append(candidate)

    normalized_sets = {
        name: _dedupe_search_values(value_sets[name], name)
        for name in fields
    }
    lengths = [len(normalized_sets[name]) for name in fields]
    total = math.prod(lengths)
    target = min(target_count, total - 1)
    if target <= 0:
        return []

    # Seed both opposite corners so every field's configured min/max is always
    # represented, even with a small coarse budget.
    add_from_values({name: normalized_sets[name][0] for name in fields})
    if len(candidates) < target:
        add_from_values({name: normalized_sets[name][-1] for name in fields})

    # A deterministic stratified pass covers every discrete level in every
    # dimension before adding more combinations. Different coprime strides and
    # offsets prevent equal-length dimensions from moving in lockstep.
    max_levels = max(lengths)
    coverage_attempts = max(max_levels, target)
    for sample_index in range(coverage_attempts):
        if len(candidates) >= target:
            break
        values: dict[str, float] = {}
        for field_index, name in enumerate(fields):
            length = len(normalized_sets[name])
            stride = _coprime_stride(length, field_index + 1)
            offset = (field_index * field_index + field_index) % length
            level_index = (sample_index * stride + offset) % length
            values[name] = normalized_sets[name][level_index]
        add_from_values(values)

    # Fill the remaining budget with a low-discrepancy sequence. Unlike the old
    # mixed-radix prefix, each prefix spans every axis instead of exhausting low
    # kp values first.
    sample_index = 1
    max_attempts = max(target * 200, 1000)
    while len(candidates) < target and sample_index <= max_attempts:
        values = {}
        for field_index, name in enumerate(fields):
            levels = normalized_sets[name]
            fraction = _radical_inverse(sample_index, _prime_for_dimension(field_index))
            level_index = min(len(levels) - 1, int(fraction * len(levels)))
            values[name] = levels[level_index]
        add_from_values(values)
        sample_index += 1

    return candidates


def _dedupe_search_values(values: list[float], name: str) -> list[float]:
    deduped: list[float] = []
    seen: set[float | int] = set()
    for value in values:
        typed = _typed_value(name, value)
        key: float | int = typed if _is_int_field(name) else round(float(typed), 9)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(typed)
    return sorted(deduped)


def _coprime_stride(length: int, seed: int) -> int:
    if length <= 1:
        return 1
    stride = 1 + (2 * seed) % length
    while math.gcd(stride, length) != 1:
        stride = stride % length + 1
    return stride


def _prime_for_dimension(index: int) -> int:
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
    return primes[index % len(primes)]


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index > 0:
        index, digit = divmod(index, base)
        result += digit * factor
        factor /= base
    return result

def _record_priority(record: IterationRecord) -> tuple:
    metrics = record.metrics
    if _is_invalid_hardware_record(record):
        return (2.0, metrics.score, max(metrics.overshoot_pct, metrics.undershoot_pct), metrics.oscillations, metrics.settling_time_s, record.iteration)
    if metrics.passed:
        bandwidth = record.candidate.mod0_ll_bw if record.candidate is not None else 0
        return (0.0, _record_objective_score(record), -bandwidth, metrics.score, record.iteration)
    return (1.0, metrics.score, max(metrics.overshoot_pct, metrics.undershoot_pct), metrics.oscillations, metrics.settling_time_s, record.iteration)


def _basin_priority(record: IterationRecord) -> tuple:
    """Rank basin representatives by measured penalty before tie-break metrics."""

    metrics = record.metrics
    return (
        0 if metrics.passed else 1,
        _record_objective_score(record) if metrics.passed else metrics.score,
        -(record.candidate.mod0_ll_bw if record.candidate is not None else 0),
        metrics.score,
        max(metrics.overshoot_pct, metrics.undershoot_pct),
        metrics.oscillations,
        metrics.settling_time_s,
        record.iteration,
    )


def _has_bandwidth_objective(record: IterationRecord) -> bool:
    try:
        return math.isfinite(float(record.objective_score)) and math.isfinite(float(record.bandwidth_bonus))
    except (TypeError, ValueError):
        return False


def _record_objective_score(record: IterationRecord) -> float:
    value = record.objective_score
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(record.metrics.score)
    return parsed if math.isfinite(parsed) else float(record.metrics.score)


def _confirmation_streak_for_last(records: list[IterationRecord]) -> int:
    if not records or not records[-1].metrics.passed or records[-1].candidate is None:
        return 0
    key = hardware_candidate_key(records[-1].candidate)
    streak = 0
    for record in reversed(records):
        if record.candidate is None or not record.metrics.passed or hardware_candidate_key(record.candidate) != key:
            break
        streak += 1
    return streak


def _candidate_key_without_bandwidth(candidate: HardwarePidCandidate) -> tuple:
    key = list(hardware_candidate_key(candidate))
    del key[6]
    return tuple(key)


def _next_higher_bandwidth_candidate(
    records: list[IterationRecord],
    search: SearchSpace,
    seen: set[tuple],
) -> HardwarePidCandidate | None:
    current_record = records[-1]
    current = current_record.candidate
    if current is None or not _has_bandwidth_objective(current_record):
        return None
    base_key = _candidate_key_without_bandwidth(current)
    lower_confirmed = [
        record
        for record in records[:-CONFIRMED_PASS_COUNT]
        if record.candidate is not None
        and record.candidate.mod0_ll_bw < current.mod0_ll_bw
        and _candidate_key_without_bandwidth(record.candidate) == base_key
        and _confirmation_streak_ending_at(records, record.iteration) >= CONFIRMED_PASS_COUNT
    ]
    if lower_confirmed and _record_objective_score(current_record) >= min(
        _record_objective_score(record) for record in lower_confirmed
    ):
        return None
    # LS/LR bandwidth is a monotonic constrained objective, not a generic
    # coarse-grid coordinate. Probe every raw register code so a coarse GUI
    # resolution cannot skip a feasible boundary (for example 76 -> 79).
    higher = int(current.mod0_ll_bw) + 1
    if higher > int(math.floor(search.mod0_ll_bw.max)):
        return None
    candidate = replace(current, mod0_ll_bw=higher, phase="bandwidth_climb")
    return None if hardware_candidate_key(candidate) in seen else candidate


def _confirmation_streak_ending_at(records: list[IterationRecord], iteration: int) -> int:
    index = next((i for i, record in enumerate(records) if record.iteration == iteration), -1)
    if index < 0 or records[index].candidate is None or not records[index].metrics.passed:
        return 0
    key = hardware_candidate_key(records[index].candidate)
    streak = 0
    for record in reversed(records[: index + 1]):
        if record.candidate is None or not record.metrics.passed or hardware_candidate_key(record.candidate) != key:
            break
        streak += 1
    return streak


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
    score = float(record.metrics.score)
    return not math.isfinite(score) or score >= 300.0


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
    if name in {"mod0_kpole1", "mod0_kpole2"}:
        values.extend(
            value
            for value in (2, 3, 4, 5, 6)
            if parameter.min <= value <= parameter.max
        )
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
    values[name] = typed_value
    return HardwarePidCandidate(**values, phase=phase)


def _kpole_value(value: float) -> int:
    return min((2, 3, 4, 5, 6), key=lambda option: abs(float(value) - option))


def _range_value_for_base(base: HardwarePidCandidate, name: str, value: float) -> float:
    # Candidate helpers already clamp through SearchParameter in normal paths.
    # This fallback keeps direct candidate construction conservative.
    _ = base
    if name in {"mod0_kp", "mod0_ki", "mod0_kd"}:
        return min(max(value, 0), 255)
    if name in {"mod0_kpole1", "mod0_kpole2"}:
        return min(max(value, 0), 15)
    if name == "mod0_cm_gain":
        return min(max(value, 0), 9)
    if name == "mod0_ll_bw":
        return min(max(value, 0), 127)
    return value


def _typed_value(name: str, value: float) -> int | float:
    if name in {"mod0_kpole1", "mod0_kpole2"}:
        return _kpole_value(value)
    if name == "mod0_cm_gain":
        return int(round(min(max(value, 0), 9)))
    if name == "mod0_ll_bw":
        return int(round(min(max(value, 0), 127)))
    if _is_int_field(name):
        return int(round(value))
    return float(value)


def _is_int_field(name: str) -> bool:
    return name.startswith("mod0_")


def _diversity_vector(candidate: HardwarePidCandidate) -> tuple[float, ...]:
    return (
        candidate.mod0_kp / 255.0,
        candidate.mod0_ki / 255.0,
        candidate.mod0_kd / 255.0,
        (candidate.mod0_kpole1 - 2.0) / 4.0,
        (candidate.mod0_kpole2 - 2.0) / 4.0,
        candidate.mod0_cm_gain / 9.0,
        (candidate.mod0_ll_bw - 47.0) / 32.0,
        candidate.output_inductance_nh / 40.0,
        candidate.effective_lc_inductance_nh / 150.0,
    )


def _euclidean_distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


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
