"""Build fixed-operating-point DRL datasets and guarded collection plans."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Protocol

import numpy as np

from ..models import (
    AutotuneExperimentConfig,
    HardwarePidCandidate,
    SearchSpace,
    TuningConfig,
    effective_lc_inductance_from_raw,
    effective_lc_inductance_raw,
    output_inductance_from_raw,
    output_inductance_raw,
)
from .common import (
    ACTION_FIELDS,
    KPOLE_PAIRS,
    METRIC_FIELDS,
    artifact_id,
    atomic_write_json,
    candidate_from_mapping,
    candidate_from_normalized,
    candidate_key,
    candidate_to_mapping,
    candidate_to_normalized,
    candidate_with_delta,
    invalid_labels,
    metric_vector,
    relabeled_score,
    vector_to_metric_mapping,
)


class CandidatePredictor(Protocol):
    def predict_features(self, features: np.ndarray) -> dict[str, np.ndarray]:
        ...


@dataclass
class DrlDataset:
    features: np.ndarray
    actions: np.ndarray
    metrics: np.ndarray
    metric_mask: np.ndarray
    invalid_labels: np.ndarray
    scores: np.ndarray
    passed: np.ndarray
    groups: np.ndarray
    candidates: list[HardwarePidCandidate]
    records: list[dict[str, Any]]
    baseline_values: np.ndarray
    baseline_mask: np.ndarray

    @property
    def size(self) -> int:
        return int(self.features.shape[0])

    def features_for_candidates(self, candidates: list[HardwarePidCandidate], search: SearchSpace) -> np.ndarray:
        if not candidates:
            return np.zeros((0, self.features.shape[1]), dtype=np.float32)
        baseline_values = _masked_column_median(self.baseline_values, self.baseline_mask)
        baseline_mask = np.ones(len(METRIC_FIELDS), dtype=np.float32)
        rows = []
        for candidate in candidates:
            action = candidate_to_normalized(candidate, search).astype(np.float32)
            rows.append(np.concatenate([action, baseline_values, baseline_mask]))
        return np.asarray(rows, dtype=np.float32)

    def save(self, dataset_dir: Path, manifest: dict[str, Any]) -> None:
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = dataset_dir / "dataset.npz"
        np.savez_compressed(
            dataset_path,
            features=self.features,
            actions=self.actions,
            metrics=self.metrics,
            metric_mask=self.metric_mask,
            invalid_labels=self.invalid_labels,
            scores=self.scores,
            passed=self.passed,
            groups=self.groups.astype(str),
            baseline_values=self.baseline_values,
            baseline_mask=self.baseline_mask,
        )
        manifest_payload = dict(manifest)
        manifest_payload["files_sha256"] = {
            "dataset.npz": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        }
        atomic_write_json(dataset_dir / "manifest.json", manifest_payload)


def load_autotune_dataset(
    run_roots: Iterable[Path],
    config: TuningConfig,
    experiment: AutotuneExperimentConfig | None = None,
    allow_legacy_inferred: bool = True,
    include_run_ids: set[str] | None = None,
) -> tuple[DrlDataset, dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    source_record_count = 0
    excluded_action_count = 0
    excluded_missing_bandwidth_count = 0
    excluded_out_of_search_space_count = 0
    excluded_measurement_integrity_count = 0
    excluded_legacy_settling_count = 0
    for root in run_roots:
        if not root.exists():
            continue
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if include_run_ids is not None and run_dir.name not in include_run_ids:
                continue
            rows = _load_run_rows(run_dir)
            if not rows:
                continue
            compatible, compatibility = _run_compatibility(run_dir, config, experiment)
            source_record_count += len(rows)
            if not compatible or (not allow_legacy_inferred and compatibility != "exact_fixed_condition"):
                source_runs.append(
                    {
                        "run_id": run_dir.name,
                        "iterations": len(rows),
                        "path": str(run_dir),
                        "compatibility": compatibility,
                        "included": False,
                        "exclusion_reason": (
                            "legacy_metadata_not_allowed"
                            if compatible and compatibility != "exact_fixed_condition"
                            else "incompatible_operating_condition"
                        ),
                    }
                )
                continue
            baseline_values, baseline_mask = _baseline_for_rows(rows)
            source_runs.append(
                {
                    "run_id": run_dir.name,
                    "iterations": len(rows),
                    "path": str(run_dir),
                    "compatibility": compatibility,
                    "included": True,
                }
            )
            for row in rows:
                optimizer_metadata = row.get("optimizer_metadata")
                if isinstance(optimizer_metadata, dict) and bool(
                    optimizer_metadata.get("exclude_from_surrogate")
                ):
                    excluded_measurement_integrity_count += 1
                    continue
                candidate_payload = row.get("candidate")
                metrics_payload = row.get("metrics")
                if not isinstance(candidate_payload, dict) or not isinstance(metrics_payload, dict):
                    continue
                # Settling V2 changes the semantic meaning of both Ts labels.
                # Mixing first-entry V1 measurements with final-entry V2
                # measurements would teach the surrogate the old false-fast
                # behavior, so old records remain browsable but never train a
                # model using the current settling-label schema.
                if int(metrics_payload.get("settling_analysis_version") or 0) < 15:
                    excluded_legacy_settling_count += 1
                    continue
                # Runs recorded before the unified LS/LR field was introduced
                # were measured with two independent register values. Treating
                # them as an inferred value of 74 would create false 8-D labels.
                if "mod0_ll_bw" not in candidate_payload:
                    excluded_missing_bandwidth_count += 1
                    continue
                candidate = candidate_from_mapping(candidate_payload, phase=str(row.get("phase") or "loaded"))
                if not _candidate_is_representable(candidate):
                    excluded_action_count += 1
                    continue
                # Historical runs can contain values outside the search space
                # configured for the policy being trained. Those samples are
                # useful for archival analysis, but they distort a normalized
                # action model that can never propose them at runtime.
                if not _candidate_is_legal(candidate, config.search):
                    excluded_out_of_search_space_count += 1
                    continue
                metric_values, metric_mask = metric_vector(metrics_payload)
                score, passed = relabeled_score(row, config.targets)
                samples.append(
                    {
                        "candidate": candidate,
                        "metrics": metric_values,
                        "metric_mask": metric_mask,
                        "invalid_labels": np.asarray(invalid_labels(row), dtype=np.float32),
                        "score": score,
                        "passed": passed,
                        "group": f"{root.name}:{run_dir.name}",
                        "record": row,
                        "baseline_values": baseline_values,
                        "baseline_mask": baseline_mask,
                    }
                )
    aggregation = _apply_raw_key_robust_targets(samples, config)
    dataset = _dataset_from_samples(samples, config.search)
    unique_hardware_keys = {candidate_key(candidate) for candidate in dataset.candidates}
    manifest = {
        "dataset_id": artifact_id("dataset"),
        "schema_version": 1,
        "sample_count": dataset.size,
        "unique_hardware_candidate_count": len(unique_hardware_keys),
        "repeat_measurement_count": dataset.size - len(unique_hardware_keys),
        "included_run_ids": sorted(include_run_ids) if include_run_ids is not None else None,
        "source_record_count": source_record_count,
        "excluded_incompatible_action_count": excluded_action_count,
        "excluded_missing_bandwidth_count": excluded_missing_bandwidth_count,
        "excluded_out_of_search_space_count": excluded_out_of_search_space_count,
        "excluded_measurement_integrity_count": excluded_measurement_integrity_count,
        "excluded_legacy_settling_count": excluded_legacy_settling_count,
        "source_runs": source_runs,
        "allow_legacy_inferred": allow_legacy_inferred,
        "metric_fields": list(METRIC_FIELDS),
        "settling_analysis_version": 15,
        "feature_layout": [
            f"normalized_action[{len(ACTION_FIELDS)}]",
            f"session_baseline_metrics[{len(METRIC_FIELDS)}]",
            f"session_baseline_mask[{len(METRIC_FIELDS)}]",
        ],
        "invalid_label_layout": [
            "protection",
            "invalid_transient",
            "invalid_bode",
            "robust_pass_probability",
        ],
        "raw_key_robust_aggregation": aggregation,
        "target_relabel": {
            "overshoot_pct": config.targets.overshoot_pct,
            "undershoot_pct": config.targets.undershoot_pct,
            "settling_time_s": config.targets.settling_time_s,
            "phase_margin_deg": config.targets.phase_margin_deg,
            "crossover_frequency_hz": config.targets.crossover_frequency_hz,
            "gain_margin_constrained": False,
        },
    }
    return dataset, manifest


def build_collection_plan(
    dataset: DrlDataset,
    config: TuningConfig,
    predictor: CandidatePredictor,
    repeat_count: int = 60,
    local_count: int = 120,
    uncertainty_count: int = 60,
    pool_size: int = 20_000,
    seed: int = 20260709,
) -> dict[str, Any]:
    if dataset.size < 20:
        raise RuntimeError("At least 20 valid historical DRL samples are required to build a collection plan.")
    rng = random.Random(seed)
    existing = {candidate_key(candidate) for candidate in dataset.candidates}
    anchors = select_anchor_candidates(dataset, config.search, count=10)
    plan_items: list[dict[str, Any]] = []

    repeats_per_anchor = max(1, repeat_count // max(1, len(anchors)))
    repeated: list[HardwarePidCandidate] = []
    for anchor in anchors:
        repeated.extend([_with_phase(anchor, "drl_repeat") for _ in range(repeats_per_anchor)])
    while len(repeated) < repeat_count:
        repeated.append(_with_phase(anchors[len(repeated) % len(anchors)], "drl_repeat"))
    rng.shuffle(repeated)
    for index, candidate in enumerate(repeated[:repeat_count]):
        plan_candidate = _with_phase(candidate, "baseline") if index == 0 else candidate
        plan_items.append(_plan_item(plan_candidate, "repeat_anchor", None))

    top_candidates = _top_distinct_candidates(dataset, config.search, count=20)
    local_candidates = _generate_local_candidates(
        top_candidates,
        config.search,
        local_count,
        existing,
        seed + 1,
        phase="drl_local_sobol",
    )
    rng.shuffle(local_candidates)
    for candidate in local_candidates:
        existing.add(candidate_key(candidate))
        prediction = _single_prediction(predictor, dataset, config.search, candidate)
        plan_items.append(_plan_item(candidate, "local_sobol", prediction))

    safe_samples = [
        candidate
        for candidate, labels in zip(dataset.candidates, dataset.invalid_labels)
        if float(np.max(labels)) <= 0.0 and _candidate_is_legal(candidate, config.search)
    ]
    pool = _generate_local_candidates(
        safe_samples or top_candidates,
        config.search,
        pool_size,
        existing,
        seed + 2,
        phase="drl_uncertainty",
        allow_shortfall=False,
    )
    features = dataset.features_for_candidates(pool, config.search)
    predictions = predictor.predict_features(features)
    safety = np.asarray(predictions["safety_probability"], dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(predictions["uncertainty"], dtype=np.float64).reshape(-1)
    order = np.argsort(-uncertainty)
    uncertainty_candidates: list[HardwarePidCandidate] = []
    for index in order:
        if safety[index] < 0.995:
            continue
        candidate = pool[int(index)]
        key = candidate_key(candidate)
        if key in existing:
            continue
        existing.add(key)
        uncertainty_candidates.append(candidate)
        prediction = _prediction_at(predictions, int(index))
        plan_items.append(_plan_item(candidate, "ensemble_uncertainty", prediction))
        if len(uncertainty_candidates) >= uncertainty_count:
            break
    if len(uncertainty_candidates) < uncertainty_count:
        raise RuntimeError(
            f"Only {len(uncertainty_candidates)} uncertainty candidates passed the 0.995 safety gate; "
            f"{uncertainty_count} are required."
        )

    for index, item in enumerate(plan_items, 1):
        item["index"] = index
    plan_id = artifact_id("collection")
    return {
        "plan_id": plan_id,
        "schema_version": 1,
        "seed": seed,
        "budget": len(plan_items),
        "allocation": {
            "repeat": repeat_count,
            "local_sobol": local_count,
            "ensemble_uncertainty": uncertainty_count,
        },
        "candidates": plan_items,
    }


def build_targeted_collection_plan(
    dataset: DrlDataset,
    config: TuningConfig,
    predictor: CandidatePredictor,
    *,
    near_pass_basin_count: int = 3,
    repeats_per_basin: int = 5,
    directional_basin_count: int = 2,
    local_sobol_count: int = 19,
    guarded_global_count: int = 10,
    seed: int = 20260720,
) -> dict[str, Any]:
    """Build the focused 100-point follow-up after the 9-D bootstrap run.

    The allocation is intentionally explicit: 15 near-pass repeats, 56
    two-sided/two-scale raw-hardware perturbations, 19 basin-local Sobol
    points, and 10 surrogate-guarded global points. Confirmation and BW climb
    candidates are inserted online by ``PlannedCandidateTuner`` and consume
    the fixed outer hardware budget rather than extending it silently.
    """

    if dataset.size < 20:
        raise RuntimeError("At least 20 compatible measurements are required for a targeted collection plan.")
    anchors = select_near_pass_candidates(dataset, config.search, near_pass_basin_count)
    if len(anchors) < near_pass_basin_count:
        raise RuntimeError(f"Only {len(anchors)} distinct near-pass basins were found.")
    directional_anchors = [
        candidate for candidate in anchors if _supports_directional_design(candidate, config.search)
    ][:directional_basin_count]
    if len(directional_anchors) < directional_basin_count:
        raise RuntimeError(
            f"Only {len(directional_anchors)} near-pass basins have room for both small-step directions; "
            f"{directional_basin_count} are required."
        )
    existing = {candidate_key(candidate) for candidate in dataset.candidates}
    plan_items: list[dict[str, Any]] = []

    for repeat_index in range(repeats_per_basin):
        for basin_index, anchor in enumerate(anchors):
            phase = "baseline" if not plan_items else "drl_targeted_near_pass_repeat"
            candidate = _with_phase(anchor, phase)
            item = _plan_item(candidate, "near_pass_repeat", _single_prediction(predictor, dataset, config.search, candidate))
            item["optimizer_metadata"].update(
                {"basin_rank": basin_index + 1, "repeat_index": repeat_index + 1, "intentional_repeat": True}
            )
            plan_items.append(item)

    directional = _generate_directional_raw_candidates(
        directional_anchors,
        config.search,
        existing,
    )
    if len(directional) != 56:
        raise RuntimeError(f"Targeted directional design produced {len(directional)} points; exactly 56 are required.")
    for candidate, metadata in directional:
        existing.add(candidate_key(candidate))
        item = _plan_item(candidate, "two_sided_local_step", _single_prediction(predictor, dataset, config.search, candidate))
        item["optimizer_metadata"].update(metadata)
        plan_items.append(item)

    local_candidates = _generate_local_candidates(
        anchors,
        config.search,
        local_sobol_count,
        existing,
        seed + 1,
        phase="drl_targeted_local_sobol",
    )
    for candidate in local_candidates:
        existing.add(candidate_key(candidate))
        plan_items.append(
            _plan_item(candidate, "basin_local_sobol", _single_prediction(predictor, dataset, config.search, candidate))
        )

    global_candidates, global_predictions = _guarded_global_candidates(
        dataset,
        config,
        predictor,
        guarded_global_count,
        existing,
        seed + 2,
    )
    for candidate, prediction in zip(global_candidates, global_predictions):
        existing.add(candidate_key(candidate))
        plan_items.append(_plan_item(candidate, "guarded_global", prediction))

    expected_repeat_count = near_pass_basin_count * repeats_per_basin
    expected_budget = expected_repeat_count + 56 + local_sobol_count + guarded_global_count
    if len(plan_items) != expected_budget:
        raise RuntimeError(f"Targeted plan has {len(plan_items)} points; expected {expected_budget}.")
    for index, item in enumerate(plan_items, 1):
        item["index"] = index
    plan_id = artifact_id("targeted_collection")
    return {
        "plan_id": plan_id,
        "schema_version": 2,
        "seed": int(seed),
        "targeted": True,
        "budget": len(plan_items),
        "dynamic_confirmation": True,
        "confirmation_count": 3,
        "bandwidth_climb_after_confirmation": True,
        "allocation": {
            "near_pass_repeat": expected_repeat_count,
            "two_sided_local_step": 56,
            "basin_local_sobol": local_sobol_count,
            "guarded_global": guarded_global_count,
        },
        "near_pass_anchors": [candidate_to_mapping(candidate) for candidate in anchors],
        "candidates": plan_items,
    }


def select_near_pass_candidates(
    dataset: DrlDataset,
    search: SearchSpace,
    count: int = 3,
) -> list[HardwarePidCandidate]:
    """Select low-score, valid, raw-key-distinct basins with action-space separation."""

    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, candidate in enumerate(dataset.candidates):
        if (
            np.max(dataset.invalid_labels[index, :3]) > 0
            or not np.all(dataset.metric_mask[index, :6] > 0)
            or not _candidate_is_legal(candidate, search)
        ):
            continue
        grouped.setdefault(candidate_key(candidate), []).append(index)
    ranked: list[tuple[float, int]] = []
    for indexes in grouped.values():
        representative = min(indexes, key=lambda index: float(dataset.scores[index]))
        median_score = float(np.median(dataset.scores[indexes]))
        ranked.append((median_score, representative))
    ranked.sort(key=lambda item: (item[0], float(dataset.scores[item[1]])))
    if not ranked:
        return []

    selected: list[int] = [ranked[0][1]]
    pool = ranked[1 : max(40, count * 12)]
    while len(selected) < count and pool:
        separated = [
            item
            for item in pool
            if min(float(np.linalg.norm(dataset.actions[item[1]] - dataset.actions[index])) for index in selected) >= 0.35
        ]
        if separated:
            chosen = separated[0]
        else:
            chosen = max(
                pool,
                key=lambda item: min(
                    float(np.linalg.norm(dataset.actions[item[1]] - dataset.actions[index])) for index in selected
                ),
            )
        selected.append(chosen[1])
        pool.remove(chosen)
    return [_with_phase(dataset.candidates[index], "drl_targeted_near_pass") for index in selected[:count]]


def _generate_directional_raw_candidates(
    anchors: list[HardwarePidCandidate],
    search: SearchSpace,
    excluded: set[tuple[Any, ...]],
) -> list[tuple[HardwarePidCandidate, dict[str, Any]]]:
    specs: tuple[tuple[str, tuple[int, int]], ...] = (
        ("mod0_kp", (2, 5)),
        ("mod0_ki", (2, 5)),
        ("mod0_kd", (2, 5)),
        ("mod0_cm_gain", (1, 2)),
        ("mod0_ll_bw", (1, 2)),
        ("output_inductance_raw", (1, 2)),
        ("effective_lc_inductance_raw", (1, 2)),
    )
    generated: list[tuple[HardwarePidCandidate, dict[str, Any]]] = []
    local_seen: set[tuple[Any, ...]] = set()
    for basin_index, anchor in enumerate(anchors):
        for field, steps in specs:
            for direction in (-1, 1):
                for scale_index, step in enumerate(steps, 1):
                    if field == "output_inductance_raw":
                        raw = output_inductance_raw(anchor.output_inductance_nh) + direction * step
                        candidate = replace(
                            anchor,
                            output_inductance_nh=output_inductance_from_raw(raw),
                            phase="drl_targeted_directional",
                        )
                    elif field == "effective_lc_inductance_raw":
                        raw = effective_lc_inductance_raw(anchor.effective_lc_inductance_nh) + direction * step
                        candidate = replace(
                            anchor,
                            effective_lc_inductance_nh=effective_lc_inductance_from_raw(raw),
                            phase="drl_targeted_directional",
                        )
                    else:
                        candidate = replace(
                            anchor,
                            **{field: int(getattr(anchor, field)) + direction * step},
                            phase="drl_targeted_directional",
                        )
                    key = candidate_key(candidate)
                    if not _candidate_is_legal(candidate, search):
                        raise RuntimeError(
                            f"Near-pass basin {basin_index + 1} cannot support {field} direction {direction:+d} step {step}."
                        )
                    if key in excluded or key in local_seen:
                        raise RuntimeError(
                            f"Raw-key collision in directional design for basin {basin_index + 1}, {field}, "
                            f"direction {direction:+d}, step {step}."
                        )
                    local_seen.add(key)
                    generated.append(
                        (
                            candidate,
                            {
                                "basin_rank": basin_index + 1,
                                "perturbed_field": field,
                                "direction": direction,
                                "step_scale": scale_index,
                                "step": step,
                            },
                        )
                    )
    return generated


def _supports_directional_design(candidate: HardwarePidCandidate, search: SearchSpace) -> bool:
    return bool(
        search.mod0_kp.min <= candidate.mod0_kp - 5
        and candidate.mod0_kp + 5 <= search.mod0_kp.max
        and search.mod0_ki.min <= candidate.mod0_ki - 5
        and candidate.mod0_ki + 5 <= search.mod0_ki.max
        and search.mod0_kd.min <= candidate.mod0_kd - 5
        and candidate.mod0_kd + 5 <= search.mod0_kd.max
        and search.mod0_cm_gain.min <= candidate.mod0_cm_gain - 2
        and candidate.mod0_cm_gain + 2 <= search.mod0_cm_gain.max
        and search.mod0_ll_bw.min <= candidate.mod0_ll_bw - 2
        and candidate.mod0_ll_bw + 2 <= search.mod0_ll_bw.max
    )


def _guarded_global_candidates(
    dataset: DrlDataset,
    config: TuningConfig,
    predictor: CandidatePredictor,
    count: int,
    excluded: set[tuple[Any, ...]],
    seed: int,
) -> tuple[list[HardwarePidCandidate], list[dict[str, Any]]]:
    pool: list[HardwarePidCandidate] = []
    local_seen: set[tuple[Any, ...]] = set()
    for point in _sobol_points(max(2000, count * 200), len(ACTION_FIELDS), seed):
        candidate = candidate_from_normalized(point * 2.0 - 1.0, config.search, "drl_targeted_guarded_global")
        key = candidate_key(candidate)
        if key in excluded or key in local_seen or not _candidate_is_legal(candidate, config.search):
            continue
        local_seen.add(key)
        pool.append(candidate)
    predictions = predictor.predict_features(dataset.features_for_candidates(pool, config.search))
    invalid_probability = np.asarray(predictions["invalid_probability"], dtype=np.float64)
    protection_probability = invalid_probability[:, 0]
    safety = 1.0 - protection_probability
    uncertainty = np.asarray(predictions["uncertainty"], dtype=np.float64).reshape(-1)
    calibration = predictor.predict_features(dataset.features)
    calibration_invalid = np.asarray(calibration["invalid_probability"], dtype=np.float64)
    observed_safe = dataset.invalid_labels[:, 0] <= 0
    calibrated_protection_threshold = (
        float(np.quantile(calibration_invalid[observed_safe, 0], 0.95))
        if np.any(observed_safe)
        else 0.25
    )
    protection_threshold = max(
        float(getattr(predictor, "invalid_probability_threshold", 0.25)),
        calibrated_protection_threshold,
    )
    uncertainty_threshold = float(getattr(predictor, "uncertainty_threshold", float("inf")))
    eligible = [
        index
        for index in range(len(pool))
        if protection_probability[index] <= protection_threshold
        and uncertainty[index] <= uncertainty_threshold
    ]
    eligible.sort(key=lambda index: (-float(safety[index]), -float(uncertainty[index])))
    if len(eligible) < count:
        raise RuntimeError(
            f"Only {len(eligible)} global candidates passed the calibrated protection/uncertainty guard; "
            f"{count} are required."
        )
    selected = eligible[:count]
    selected_predictions = []
    for index in selected:
        prediction = _prediction_at(predictions, index)
        prediction.update(
            {
                "global_guard": "calibrated_protection_plus_uncertainty",
                "protection_probability": float(protection_probability[index]),
                "protection_probability_threshold": protection_threshold,
                "uncertainty_threshold": uncertainty_threshold,
            }
        )
        selected_predictions.append(prediction)
    return [pool[index] for index in selected], selected_predictions


def build_bootstrap_collection_plan(
    config: TuningConfig,
    repeat_count: int = 40,
    global_count: int = 160,
    local_count: int = 40,
    seed: int = 20260720,
) -> dict[str, Any]:
    """Build the first exact-schema plan without a surrogate or seed dataset.

    A regular Cartesian grid is intractable in nine dimensions. The bootstrap
    therefore combines an explicit stable center anchor, a scrambled global
    Sobol design, and a denser local Sobol design around that anchor. Pole
    pairs are stratified across all 25 independent 2--6 combinations.
    """

    repeat_count = max(1, int(repeat_count))
    global_count = max(1, int(global_count))
    local_count = max(0, int(local_count))
    search = config.search
    anchor_unquantized = HardwarePidCandidate(
        mod0_kp=int(round(search.mod0_kp.center)),
        mod0_ki=int(round(search.mod0_ki.center)),
        mod0_kd=int(round(search.mod0_kd.center)),
        mod0_kpole1=int(round(search.mod0_kpole1.center)),
        mod0_kpole2=int(round(search.mod0_kpole2.center)),
        mod0_cm_gain=int(round(search.mod0_cm_gain.center)),
        mod0_ll_bw=int(round(search.mod0_ll_bw.center)),
        output_inductance_nh=float(search.output_inductance_nh.center),
        effective_lc_inductance_nh=float(search.effective_lc_inductance_nh.center),
        phase="drl_bootstrap_anchor",
    )
    # Use the same normalization/quantization path as SAC inference so the
    # persisted plan cannot contain a value the 9-D runtime cannot reproduce.
    anchor = candidate_from_normalized(
        candidate_to_normalized(anchor_unquantized, search),
        search,
        "drl_bootstrap_anchor",
    )

    excluded = {candidate_key(anchor)}
    global_candidates: list[HardwarePidCandidate] = []
    global_seen: set[tuple[Any, ...]] = set()
    points = _sobol_points(max(global_count * 3, global_count), len(ACTION_FIELDS), seed)
    for sequence_index, point in enumerate(points):
        candidate = candidate_from_normalized(
            point * 2.0 - 1.0,
            search,
            "drl_bootstrap_global_sobol",
        )
        kpole1, kpole2 = KPOLE_PAIRS[sequence_index % len(KPOLE_PAIRS)]
        candidate = replace(candidate, mod0_kpole1=kpole1, mod0_kpole2=kpole2)
        key = candidate_key(candidate)
        if key in excluded or key in global_seen:
            continue
        global_seen.add(key)
        global_candidates.append(candidate)
        if len(global_candidates) >= global_count:
            break
    if len(global_candidates) < global_count:
        raise RuntimeError(
            f"Only {len(global_candidates)} unique global 9-D bootstrap candidates were generated; "
            f"{global_count} are required."
        )
    excluded.update(global_seen)

    local_candidates = _generate_local_candidates(
        [anchor],
        search,
        local_count,
        excluded,
        seed + 1,
        phase="drl_bootstrap_local_sobol",
    )
    exploration = [
        _plan_item(candidate, "global_9d_sobol", None)
        for candidate in global_candidates
    ] + [
        _plan_item(candidate, "local_anchor_sobol", None)
        for candidate in local_candidates
    ]
    random.Random(seed).shuffle(exploration)

    # Put the first anchor at index 1 so it becomes the explicit session
    # baseline. Interleave the remaining repeats to estimate drift/noise over
    # the whole run instead of measuring all repeats back-to-back.
    plan_items = [_plan_item(_with_phase(anchor, "baseline"), "repeat_anchor", None)]
    repeats_remaining = repeat_count - 1
    repeat_interval = max(1, len(exploration) // max(1, repeats_remaining))
    for item in exploration:
        plan_items.append(item)
        explored = sum(1 for current in plan_items if current.get("source") != "repeat_anchor")
        if repeats_remaining > 0 and explored % repeat_interval == 0:
            plan_items.append(_plan_item(anchor, "repeat_anchor", None))
            repeats_remaining -= 1
    while repeats_remaining > 0:
        plan_items.append(_plan_item(anchor, "repeat_anchor", None))
        repeats_remaining -= 1

    for index, item in enumerate(plan_items, 1):
        item["index"] = index
    plan_id = artifact_id("bootstrap_collection")
    return {
        "plan_id": plan_id,
        "schema_version": 1,
        "seed": int(seed),
        "bootstrap": True,
        "requires_provisional_surrogate": False,
        "budget": len(plan_items),
        "allocation": {
            "repeat": repeat_count,
            "global_9d_sobol": global_count,
            "local_anchor_sobol": local_count,
        },
        "anchor": candidate_to_mapping(anchor),
        "candidates": plan_items,
    }


def save_collection_plan(plan: dict[str, Any], plan_root: Path) -> Path:
    plan_id = str(plan.get("plan_id") or artifact_id("collection"))
    target = plan_root / plan_id / "plan.json"
    atomic_write_json(target, plan)
    return target


def select_anchor_candidates(
    dataset: DrlDataset,
    search: SearchSpace,
    count: int = 10,
) -> list[HardwarePidCandidate]:
    valid_indexes = [
        index
        for index in range(dataset.size)
        if np.max(dataset.invalid_labels[index, :3]) <= 0
        and np.all(dataset.metric_mask[index, :6] > 0)
        and _candidate_is_legal(dataset.candidates[index], search)
    ]
    if len(valid_indexes) < count:
        raise RuntimeError(f"Only {len(valid_indexes)} complete safe samples are available for {count} anchors.")
    unique: dict[tuple[Any, ...], int] = {}
    for index in valid_indexes:
        key = candidate_key(dataset.candidates[index])
        previous = unique.get(key)
        if previous is None or dataset.scores[index] < dataset.scores[previous]:
            unique[key] = index
    ranked = sorted(unique.values(), key=lambda index: float(dataset.scores[index]))
    selected: list[int] = []
    target_scores = [float(dataset.scores[ranked[index]]) for index in range(min(4, len(ranked)))]
    target_scores.extend([0.0, 0.0])
    target_scores.extend(
        float(dataset.scores[ranked[int(round((len(ranked) - 1) * quantile))]])
        for quantile in (0.50, 0.60, 0.85, 0.95)
    )
    for target_score in target_scores[:count]:
        selected.append(
            _nearest_score_index(ranked, target_score, selected, dataset.scores, dataset.actions)
        )
    for index in ranked:
        if len(selected) >= count:
            break
        if index not in selected:
            selected.append(index)
    return [_with_phase(dataset.candidates[index], "drl_repeat") for index in selected]


def _dataset_from_samples(samples: list[dict[str, Any]], search: SearchSpace) -> DrlDataset:
    if not samples:
        width = len(ACTION_FIELDS) + len(METRIC_FIELDS) * 2
        return DrlDataset(
            features=np.zeros((0, width), dtype=np.float32),
            actions=np.zeros((0, len(ACTION_FIELDS)), dtype=np.float32),
            metrics=np.zeros((0, len(METRIC_FIELDS)), dtype=np.float32),
            metric_mask=np.zeros((0, len(METRIC_FIELDS)), dtype=np.float32),
            invalid_labels=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros(0, dtype=np.float32),
            passed=np.zeros(0, dtype=np.float32),
            groups=np.asarray([], dtype=str),
            candidates=[],
            records=[],
            baseline_values=np.zeros((0, len(METRIC_FIELDS)), dtype=np.float32),
            baseline_mask=np.zeros((0, len(METRIC_FIELDS)), dtype=np.float32),
        )
    actions = np.asarray(
        [candidate_to_normalized(sample["candidate"], search, clip=False) for sample in samples],
        dtype=np.float32,
    )
    baseline_values = np.asarray([sample["baseline_values"] for sample in samples], dtype=np.float32)
    baseline_mask = np.asarray([sample["baseline_mask"] for sample in samples], dtype=np.float32)
    filled_baseline = baseline_values.copy()
    medians = _masked_column_median(baseline_values, baseline_mask)
    for column in range(filled_baseline.shape[1]):
        missing = baseline_mask[:, column] <= 0
        filled_baseline[missing, column] = medians[column]
    features = np.concatenate([actions, filled_baseline, baseline_mask], axis=1).astype(np.float32)
    return DrlDataset(
        features=features,
        actions=actions,
        metrics=np.asarray([sample["metrics"] for sample in samples], dtype=np.float32),
        metric_mask=np.asarray([sample["metric_mask"] for sample in samples], dtype=np.float32),
        invalid_labels=np.asarray([sample["invalid_labels"] for sample in samples], dtype=np.float32),
        scores=np.asarray([sample["score"] for sample in samples], dtype=np.float32),
        passed=np.asarray([sample["passed"] for sample in samples], dtype=np.float32),
        groups=np.asarray([sample["group"] for sample in samples], dtype=str),
        candidates=[sample["candidate"] for sample in samples],
        records=[sample["record"] for sample in samples],
        baseline_values=baseline_values,
        baseline_mask=baseline_mask,
    )


def _load_run_rows(run_dir: Path) -> list[dict[str, Any]]:
    jsonl = run_dir / "iterations.jsonl"
    if jsonl.exists():
        rows = []
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
            except Exception:
                continue
        if rows:
            return rows
    status_path = run_dir / "run_status.json"
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        history = payload.get("history") if isinstance(payload, dict) else None
        return [item for item in history if isinstance(item, dict)] if isinstance(history, list) else []
    except Exception:
        return []


def _run_compatibility(
    run_dir: Path,
    config: TuningConfig,
    experiment: AutotuneExperimentConfig | None,
) -> tuple[bool, str]:
    if experiment is None:
        return True, "not_checked"
    try:
        status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    except Exception:
        return False, "missing_run_status"
    if not isinstance(status, dict):
        return False, "invalid_run_status"
    run_config = status.get("config") if isinstance(status.get("config"), dict) else {}
    targets = run_config.get("targets") if isinstance(run_config.get("targets"), dict) else {}
    run_vout = _finite_float(targets.get("vout_target_v"))
    if run_vout is not None and abs(run_vout - config.targets.vout_target_v) > 0.01:
        return False, "vout_mismatch"

    raw_experiment = status.get("experiment") if isinstance(status.get("experiment"), dict) else {}
    inferred = run_vout is None
    for key, expected in (
        ("board_address", experiment.board_address),
        ("board_page", experiment.board_page),
        ("board_adapter", experiment.board_adapter),
        ("response_channel", experiment.response_channel),
    ):
        if key not in raw_experiment:
            inferred = True
            continue
        if str(raw_experiment.get(key)).strip().lower() != str(expected).strip().lower():
            return False, f"{key}_mismatch"

    run_fg = raw_experiment.get("function_generator_config")
    run_fg = run_fg if isinstance(run_fg, dict) else {}
    expected_fg = experiment.function_generator_config or {}
    if "mode" in run_fg and str(run_fg.get("mode", "")).strip().lower() != str(
        expected_fg.get("mode", "square")
    ).strip().lower():
        return False, "function_generator_mode_mismatch"
    if "mode" not in run_fg:
        inferred = True
    compatible, used_inference = _optional_numeric_settings_match(
        run_fg,
        expected_fg,
        (("frequency_hz",), ("low_v", "low_level"), ("high_v", "high_level")),
        (1.0, 1e-6, 1e-6),
    )
    if not compatible:
        return False, "function_generator_mismatch"
    inferred = inferred or used_inference

    run_bode = raw_experiment.get("bode_config")
    run_bode = run_bode if isinstance(run_bode, dict) else {}
    expected_bode = experiment.bode_config or {}
    compatible, used_inference = _optional_numeric_settings_match(
        run_bode,
        expected_bode,
        (("start_hz",), ("stop_hz",), ("points",), ("bandwidth_hz",), ("source_vpp",)),
        (1.0, 1.0, 0.0, 1.0, 1e-6),
    )
    if not compatible:
        return False, "bode_config_mismatch"
    inferred = inferred or used_inference
    return True, "legacy_inferred" if inferred else "exact_fixed_condition"


def _optional_numeric_settings_match(
    actual: dict[str, Any],
    expected: dict[str, Any],
    key_groups: tuple[tuple[str, ...], ...],
    tolerances: tuple[float, ...],
) -> tuple[bool, bool]:
    inferred = False
    for keys, tolerance in zip(key_groups, tolerances):
        expected_value = _first_numeric(expected, keys)
        actual_value = _first_numeric(actual, keys)
        if actual_value is None:
            inferred = True
            continue
        if expected_value is None or abs(actual_value - expected_value) > tolerance:
            return False, inferred
    return True, inferred


def _first_numeric(payload: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key not in payload:
            continue
        value = _finite_float(payload.get(key))
        if value is not None:
            return value
    return None


def _candidate_is_representable(candidate: HardwarePidCandidate) -> bool:
    return bool(
        0 <= candidate.mod0_cm_gain <= 9
        and 0 <= candidate.mod0_ll_bw <= 127
        and candidate.mod0_kpole1 in {2, 3, 4, 5, 6}
        and candidate.mod0_kpole2 in {2, 3, 4, 5, 6}
    )


def _apply_raw_key_robust_targets(
    samples: list[dict[str, Any]],
    config: TuningConfig,
) -> dict[str, Any]:
    """Replace noisy repeats with conservative raw-key aggregate targets.

    A single hardware capture is not a deterministic label near the 2 us
    settling boundary.  Every occurrence of one exact register key therefore
    receives the same robust metric target: p90 for transient magnitude/Ts
    and Bode shape penalty, p10 for phase margin, and the median for
    crossover/gain margin.  The fourth
    classifier target is a Jeffreys-smoothed robust-pass probability.
    This makes three confirmed passes materially different from one lucky pass.
    """

    if not samples:
        return {
            "enabled": True,
            "candidate_group_count": 0,
            "repeat_group_count": 0,
            "max_repeat_count": 0,
            "quantiles": {},
        }
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, sample in enumerate(samples):
        grouped.setdefault(candidate_key(sample["candidate"]), []).append(index)
    confirmed_windows = _latest_confirmed_windows(samples)
    quantiles = (0.90, 0.90, 0.90, 0.90, 0.10, 0.50, 0.50, 0.90)
    repeat_counts: list[int] = []
    confirmed_group_count = 0
    for key, indexes in grouped.items():
        repeat_counts.append(len(indexes))
        confirmed_indexes = confirmed_windows.get(key)
        target_indexes = confirmed_indexes or indexes
        if confirmed_indexes:
            confirmed_group_count += 1
        pass_count = sum(float(samples[index]["passed"]) >= 0.5 for index in target_indexes)
        pass_probability = (float(pass_count) + 0.5) / (float(len(target_indexes)) + 1.0)
        robust_metrics = np.zeros(len(METRIC_FIELDS), dtype=np.float32)
        robust_mask = np.zeros(len(METRIC_FIELDS), dtype=np.float32)
        for column, quantile in enumerate(quantiles):
            values = [
                float(samples[index]["metrics"][column])
                for index in target_indexes
                if float(samples[index]["metric_mask"][column]) > 0.0
                and np.isfinite(float(samples[index]["metrics"][column]))
            ]
            if values:
                robust_metrics[column] = float(np.quantile(np.asarray(values, dtype=np.float64), quantile))
                robust_mask[column] = 1.0
        robust_payload = vector_to_metric_mapping(robust_metrics)
        robust_score, _ = relabeled_score(robust_payload, config.targets)
        safety_labels = np.asarray(
            [samples[index]["invalid_labels"][:3] for index in target_indexes],
            dtype=np.float32,
        )
        safety_probability_targets = np.mean(safety_labels, axis=0)
        requirement_failure_probability = 1.0 - pass_probability
        combined_labels = np.concatenate(
            [safety_probability_targets, np.asarray([pass_probability], dtype=np.float32)]
        ).astype(np.float32)
        for index in indexes:
            samples[index]["metrics"] = robust_metrics.copy()
            samples[index]["metric_mask"] = robust_mask.copy()
            samples[index]["invalid_labels"] = combined_labels.copy()
            samples[index]["score"] = float(robust_score)
            samples[index]["passed"] = float(pass_probability)
            samples[index]["record"].setdefault("drl_robust_target", {})
            samples[index]["record"]["drl_robust_target"].update(
                {
                    "raw_key_repeat_count": len(indexes),
                    "robust_target_measurement_count": len(target_indexes),
                    "pass_count": int(pass_count),
                    "pass_probability": float(pass_probability),
                    "confirmed_streak_target": bool(confirmed_indexes),
                    "requirement_failure_probability": float(requirement_failure_probability),
                }
            )
    return {
        "enabled": True,
        "candidate_group_count": len(grouped),
        "repeat_group_count": sum(count > 1 for count in repeat_counts),
        "max_repeat_count": max(repeat_counts, default=0),
        "confirmed_group_count": confirmed_group_count,
        "quantiles": {
            "overshoot_pct": 0.90,
            "undershoot_pct": 0.90,
            "overshoot_settling_time_us": 0.90,
            "undershoot_settling_time_us": 0.90,
            "phase_margin_deg": 0.10,
            "crossover_frequency_khz": 0.50,
            "gain_margin_db": 0.50,
            "bode_gain_shape_penalty": 0.90,
        },
        "pass_probability_prior": "Jeffreys Beta(0.5, 0.5)",
        "robust_pass_threshold": 0.80,
    }


def _latest_confirmed_windows(
    samples: list[dict[str, Any]],
    required: int = 3,
) -> dict[tuple[Any, ...], list[int]]:
    """Return the latest truly consecutive passing window for each raw key.

    Confirmation is a temporal hardware property: three measurements must be
    adjacent iterations in the same run.  Grouping all historical repeats can
    otherwise let an old failure permanently invalidate a point that was later
    re-measured and confirmed under stable conditions.
    """

    windows: dict[tuple[Any, ...], list[int]] = {}
    streak: list[int] = []
    previous_group: str | None = None
    previous_iteration: int | None = None
    previous_key: tuple[Any, ...] | None = None
    for index, sample in enumerate(samples):
        group = str(sample.get("group") or "")
        record = sample.get("record") if isinstance(sample.get("record"), dict) else {}
        try:
            iteration = int(record.get("iteration"))
        except (TypeError, ValueError):
            iteration = None
        key = candidate_key(sample["candidate"])
        adjacent = bool(
            group
            and group == previous_group
            and iteration is not None
            and previous_iteration is not None
            and iteration == previous_iteration + 1
            and key == previous_key
        )
        if float(sample.get("passed", 0.0)) >= 0.5:
            streak = [*streak, index] if adjacent else [index]
        else:
            streak = []
        if len(streak) >= max(1, int(required)):
            windows[key] = streak[-max(1, int(required)):]
        previous_group = group
        previous_iteration = iteration
        previous_key = key
    return windows


def _candidate_is_legal(candidate: HardwarePidCandidate, search: SearchSpace) -> bool:
    tolerance = 1e-3
    return bool(
        search.mod0_kp.min <= candidate.mod0_kp <= search.mod0_kp.max
        and search.mod0_ki.min <= candidate.mod0_ki <= search.mod0_ki.max
        and search.mod0_kd.min <= candidate.mod0_kd <= search.mod0_kd.max
        and search.mod0_ll_bw.min <= candidate.mod0_ll_bw <= search.mod0_ll_bw.max
        and search.output_inductance_nh.min - tolerance
        <= candidate.output_inductance_nh
        <= search.output_inductance_nh.max + tolerance
        and search.effective_lc_inductance_nh.min - tolerance
        <= candidate.effective_lc_inductance_nh
        <= search.effective_lc_inductance_nh.max + tolerance
        and _candidate_is_representable(candidate)
    )


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None
    except Exception:
        return None


def _baseline_for_rows(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    baseline = next((row for row in rows if str(row.get("phase", "")).lower() == "baseline"), None)
    # Never infer session context from the first measured candidate. That row
    # is also a supervised target, so feeding its metrics back as a baseline
    # leaks the answer into every feature row from the run. Runs without a
    # dedicated baseline remain explicitly missing and are imputed from the
    # training partition by the offline benchmark.
    if baseline is None:
        baseline = {}
    metrics = baseline.get("metrics") if isinstance(baseline, dict) else None
    return metric_vector(metrics if isinstance(metrics, dict) else {})


def _top_distinct_candidates(
    dataset: DrlDataset,
    search: SearchSpace,
    count: int,
) -> list[HardwarePidCandidate]:
    indexes = np.argsort(dataset.scores)
    result: list[HardwarePidCandidate] = []
    seen: set[tuple[Any, ...]] = set()
    for raw_index in indexes:
        index = int(raw_index)
        if (
            np.max(dataset.invalid_labels[index, :3]) > 0
            or not np.all(dataset.metric_mask[index, :6] > 0)
            or not _candidate_is_legal(dataset.candidates[index], search)
        ):
            continue
        candidate = dataset.candidates[index]
        key = candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
        if len(result) >= count:
            break
    if not result:
        raise RuntimeError("No complete safe candidates are available for local collection.")
    return result


def _generate_local_candidates(
    bases: list[HardwarePidCandidate],
    search: SearchSpace,
    count: int,
    excluded: set[tuple[Any, ...]],
    seed: int,
    phase: str,
    allow_shortfall: bool = False,
) -> list[HardwarePidCandidate]:
    if not bases or count <= 0:
        return []
    points = _sobol_points(max(count * 5, count), len(ACTION_FIELDS), seed)
    candidates: list[HardwarePidCandidate] = []
    local_seen: set[tuple[Any, ...]] = set()
    for index, point in enumerate(points):
        base = bases[index % len(bases)]
        candidate = candidate_with_delta(base, point * 2.0 - 1.0, search, phase, trust_fraction=0.10)
        key = candidate_key(candidate)
        if key in excluded or key in local_seen:
            continue
        local_seen.add(key)
        candidates.append(candidate)
        if len(candidates) >= count:
            break
    if len(candidates) < count and not allow_shortfall:
        raise RuntimeError(f"Only {len(candidates)} fresh {phase} candidates could be generated; {count} are required.")
    return candidates


def _sobol_points(count: int, dimensions: int, seed: int) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to generate the DRL Sobol collection plan.") from exc
    engine = torch.quasirandom.SobolEngine(dimension=dimensions, scramble=True, seed=seed)
    return engine.draw(count).cpu().numpy().astype(np.float64)


def _single_prediction(
    predictor: CandidatePredictor,
    dataset: DrlDataset,
    search: SearchSpace,
    candidate: HardwarePidCandidate,
) -> dict[str, Any]:
    predictions = predictor.predict_features(dataset.features_for_candidates([candidate], search))
    return _prediction_at(predictions, 0)


def _prediction_at(predictions: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    metric_mean = np.asarray(predictions["metric_mean"])[index]
    metric_std = np.asarray(predictions["metric_std"])[index]
    safety_values = np.asarray(predictions["safety_probability"])
    invalid_values = np.asarray(
        predictions.get("invalid_probability", np.zeros((len(safety_values), 3), dtype=float))
    )
    return {
        "predicted_metrics": {name: float(metric_mean[position]) for position, name in enumerate(METRIC_FIELDS)},
        "metric_std": {name: float(metric_std[position]) for position, name in enumerate(METRIC_FIELDS)},
        "safety_probability": float(safety_values[index]),
        "validity_probability": float(
            np.asarray(predictions.get("validity_probability", predictions["safety_probability"]))[index]
        ),
        "invalid_probability": invalid_values[index].tolist(),
        "uncertainty": float(np.asarray(predictions["uncertainty"])[index]),
    }


def _plan_item(candidate: HardwarePidCandidate, source: str, prediction: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "source": source,
        "candidate": candidate_to_mapping(candidate),
        "optimizer_metadata": {
            "algorithm": "drl-collection",
            "proposal_source": source,
            **(prediction or {}),
        },
    }


def _nearest_score_index(
    ranked: list[int],
    target_score: float,
    selected: list[int],
    scores: np.ndarray,
    actions: np.ndarray,
) -> int:
    available = [index for index in ranked if index not in selected]
    closest_distance = min(abs(float(scores[index]) - target_score) for index in available)
    tied = [
        index
        for index in available
        if abs(abs(float(scores[index]) - target_score) - closest_distance) <= 1e-9
    ]
    if not selected:
        return tied[0]
    return max(
        tied,
        key=lambda index: min(float(np.linalg.norm(actions[index] - actions[other])) for other in selected),
    )


def _masked_column_median(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(len(METRIC_FIELDS), dtype=np.float32)
    medians = np.zeros(values.shape[1], dtype=np.float32)
    for column in range(values.shape[1]):
        observed = values[mask[:, column] > 0, column]
        medians[column] = float(np.median(observed)) if observed.size else 0.0
    return medians


def _with_phase(candidate: HardwarePidCandidate, phase: str) -> HardwarePidCandidate:
    payload = candidate_to_mapping(candidate)
    payload["phase"] = phase
    return candidate_from_mapping(payload, phase=phase)
