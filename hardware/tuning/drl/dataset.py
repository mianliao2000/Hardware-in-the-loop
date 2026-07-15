"""Build fixed-operating-point DRL datasets and guarded collection plans."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Protocol

import numpy as np

from ..models import AutotuneExperimentConfig, HardwarePidCandidate, SearchSpace, TuningConfig
from .common import (
    METRIC_FIELDS,
    artifact_id,
    atomic_write_json,
    candidate_from_mapping,
    candidate_key,
    candidate_to_mapping,
    candidate_to_normalized,
    candidate_with_delta,
    invalid_labels,
    metric_vector,
    relabeled_score,
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
) -> tuple[DrlDataset, dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    source_record_count = 0
    excluded_action_count = 0
    excluded_out_of_search_space_count = 0
    for root in run_roots:
        if not root.exists():
            continue
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
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
                candidate_payload = row.get("candidate")
                metrics_payload = row.get("metrics")
                if not isinstance(candidate_payload, dict) or not isinstance(metrics_payload, dict):
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
    dataset = _dataset_from_samples(samples, config.search)
    manifest = {
        "dataset_id": artifact_id("dataset"),
        "schema_version": 1,
        "sample_count": dataset.size,
        "source_record_count": source_record_count,
        "excluded_incompatible_action_count": excluded_action_count,
        "excluded_out_of_search_space_count": excluded_out_of_search_space_count,
        "source_runs": source_runs,
        "allow_legacy_inferred": allow_legacy_inferred,
        "metric_fields": list(METRIC_FIELDS),
        "feature_layout": [
            "normalized_action[6]",
            "session_baseline_metrics[7]",
            "session_baseline_mask[7]",
        ],
        "invalid_label_layout": ["protection", "invalid_transient", "invalid_bode"],
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
        if np.max(dataset.invalid_labels[index]) <= 0
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
        width = 6 + len(METRIC_FIELDS) * 2
        return DrlDataset(
            features=np.zeros((0, width), dtype=np.float32),
            actions=np.zeros((0, 6), dtype=np.float32),
            metrics=np.zeros((0, len(METRIC_FIELDS)), dtype=np.float32),
            metric_mask=np.zeros((0, len(METRIC_FIELDS)), dtype=np.float32),
            invalid_labels=np.zeros((0, 3), dtype=np.float32),
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
        candidate.mod0_cm_gain == 2
        and candidate.mod0_kpole1 == candidate.mod0_kpole2
        and candidate.mod0_kpole1 in {3, 6}
    )


def _candidate_is_legal(candidate: HardwarePidCandidate, search: SearchSpace) -> bool:
    tolerance = 1e-3
    return bool(
        search.mod0_kp.min <= candidate.mod0_kp <= search.mod0_kp.max
        and search.mod0_ki.min <= candidate.mod0_ki <= search.mod0_ki.max
        and search.mod0_kd.min <= candidate.mod0_kd <= search.mod0_kd.max
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
    if baseline is None:
        baseline = next((row for row in rows if isinstance(row.get("metrics"), dict)), {})
    return metric_vector(baseline.get("metrics") if isinstance(baseline, dict) else {})


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
            np.max(dataset.invalid_labels[index]) > 0
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
    points = _sobol_points(max(count * 5, count), 6, seed)
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
