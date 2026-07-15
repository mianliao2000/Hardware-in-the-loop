"""Bootstrap neural surrogate ensemble for fixed-operating-point autotuning."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from ..models import TuningConfig
from .common import METRIC_FIELDS, atomic_write_json, relabeled_score, vector_to_metric_mapping
from .dataset import DrlDataset


class DrlDependencyError(RuntimeError):
    pass


def dependency_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "ok": False,
        "torch": None,
        "gymnasium": None,
        "stable_baselines3": None,
        "device": "cpu",
        "error": None,
    }
    try:
        import gymnasium
        import stable_baselines3
        import torch

        status.update(
            {
                "torch": torch.__version__,
                "gymnasium": gymnasium.__version__,
                "stable_baselines3": stable_baselines3.__version__,
            }
        )
        status["ok"] = True
    except Exception as exc:
        status["error"] = str(exc)
    return status


def require_ml_dependencies() -> dict[str, Any]:
    status = dependency_status()
    if not status["ok"]:
        raise DrlDependencyError(
            "DRL dependencies are unavailable. Install requirements-ml.txt before training or inference. "
            f"{status.get('error') or ''}".strip()
        )
    return status


class SurrogateEnsemble:
    def __init__(
        self,
        artifact_dir: Path,
        manifest: dict[str, Any],
        members: list[Any],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        metric_mean: np.ndarray,
        metric_std: np.ndarray,
        device: str,
    ):
        self.artifact_dir = artifact_dir
        self.manifest = manifest
        self.members = members
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
        self.metric_mean = metric_mean.astype(np.float32)
        self.metric_std = metric_std.astype(np.float32)
        self.device = device

    @property
    def model_id(self) -> str:
        return str(self.manifest.get("model_id") or self.artifact_dir.name)

    @property
    def accepted(self) -> bool:
        return bool(self.manifest.get("accepted", False))

    @property
    def uncertainty_threshold(self) -> float:
        return float(self.manifest.get("uncertainty_threshold", 1.0))

    @property
    def invalid_probability_threshold(self) -> float:
        return float(self.manifest.get("invalid_probability_threshold", 0.25))

    def predict_features(self, features: np.ndarray) -> dict[str, np.ndarray]:
        import torch

        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        # Sparse session-level baseline features can place a new power cycle
        # outside the small training hull. Bound standardized inputs so an
        # unseen session cannot drive an unconstrained neural extrapolation.
        normalized = np.clip((values - self.feature_mean) / self.feature_std, -6.0, 6.0)
        tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        metric_predictions: list[np.ndarray] = []
        invalid_probabilities: list[np.ndarray] = []
        with torch.no_grad():
            for member in self.members:
                member.eval()
                metric_scaled, invalid_logits = member(tensor)
                metric_scaled_np = metric_scaled.cpu().numpy()
                if self.manifest.get("prediction_target") == "baseline_residual":
                    metric = metric_scaled_np * self.metric_std + values[:, 6:13]
                else:
                    metric = metric_scaled_np * self.metric_std + self.metric_mean
                invalid = torch.sigmoid(invalid_logits).cpu().numpy()
                metric_predictions.append(metric)
                invalid_probabilities.append(invalid)
        metric_stack = np.stack(metric_predictions, axis=0)
        invalid_stack = np.stack(invalid_probabilities, axis=0)
        metric_mean = np.mean(metric_stack, axis=0)
        metric_std = np.std(metric_stack, axis=0)
        invalid_mean = np.mean(invalid_stack, axis=0)
        positive_counts = self.manifest.get("invalid_positive_counts") or []
        if positive_counts and int(positive_counts[0]) == 0:
            # The historical fixed-condition data contains no protection
            # events. A classifier head with only negative labels is not
            # calibrated, so use the Jeffreys-prior posterior mean instead of
            # its arbitrary logit. Any observed protection disables this path.
            sample_count = max(1, int(self.manifest.get("sample_count", 1)))
            invalid_mean[:, 0] = 0.5 / (sample_count + 1.0)
        safety_probability = 1.0 - invalid_mean[:, 0]
        validity_probability = 1.0 - np.max(invalid_mean, axis=1)
        normalized_uncertainty = metric_std / np.maximum(np.abs(self.metric_std), 1e-6)
        uncertainty = np.mean(normalized_uncertainty, axis=1)
        return {
            "metric_mean": metric_mean.astype(np.float64),
            "metric_std": metric_std.astype(np.float64),
            "invalid_probability": invalid_mean.astype(np.float64),
            "safety_probability": safety_probability.astype(np.float64),
            "validity_probability": validity_probability.astype(np.float64),
            "uncertainty": uncertainty.astype(np.float64),
        }

    @classmethod
    def load(cls, artifact_dir: Path, device: str | None = None) -> "SurrogateEnsemble":
        require_ml_dependencies()
        import torch

        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"DRL model manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _verify_artifact_hashes(artifact_dir, manifest.get("files_sha256"))
        scaler = np.load(artifact_dir / "scalers.npz")
        selected_device = device or "cpu"
        network_type = _surrogate_network_type(torch)
        members = []
        member_files = manifest.get("member_files") or []
        for filename in member_files:
            member = network_type(
                int(manifest["feature_count"]),
                int(manifest["metric_count"]),
                int(manifest.get("invalid_label_count", 3)),
            ).to(selected_device)
            state = torch.load(artifact_dir / str(filename), map_location=selected_device, weights_only=True)
            member.load_state_dict(state)
            member.eval()
            members.append(member)
        if not members:
            raise RuntimeError(f"DRL model '{artifact_dir.name}' contains no ensemble members.")
        return cls(
            artifact_dir=artifact_dir,
            manifest=manifest,
            members=members,
            feature_mean=scaler["feature_mean"],
            feature_std=scaler["feature_std"],
            metric_mean=scaler["metric_mean"],
            metric_std=scaler["metric_std"],
            device=selected_device,
        )


def train_surrogate_ensemble(
    dataset: DrlDataset,
    config: TuningConfig,
    artifact_dir: Path,
    operating_signature: dict[str, Any],
    members: int = 5,
    epochs: int = 300,
    batch_size: int = 128,
    seed: int = 20260709,
    progress: Callable[[float, str], None] | None = None,
) -> SurrogateEnsemble:
    dependency = require_ml_dependencies()
    import torch
    from torch import nn

    if dataset.size < 20:
        raise RuntimeError(f"At least 20 DRL samples are required; found {dataset.size}.")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    cpu_threads = _cpu_thread_count()
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_indexes, validation_indexes = _group_candidate_split(dataset, seed)
    feature_mean = np.mean(dataset.features[train_indexes], axis=0)
    feature_std = np.std(dataset.features[train_indexes], axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    metric_mean, metric_std = _masked_mean_std(dataset.metrics[train_indexes], dataset.metric_mask[train_indexes])
    x = np.clip((dataset.features - feature_mean) / feature_std, -6.0, 6.0).astype(np.float32)
    # Session baseline metrics remain explicit input context. Predict absolute
    # metrics so incomplete or legacy baseline rows cannot force a biased
    # residual target.
    y = ((dataset.metrics - metric_mean) / metric_std).astype(np.float32)
    mask = dataset.metric_mask.astype(np.float32)
    invalid = dataset.invalid_labels.astype(np.float32)

    network_type = _surrogate_network_type(torch)
    trained_members: list[Any] = []
    member_files: list[str] = []
    member_validation: list[float] = []
    rng = np.random.default_rng(seed)
    for member_index in range(max(1, members)):
        if progress:
            progress(member_index / max(1, members), f"Training surrogate {member_index + 1}/{members}")
        member = network_type(x.shape[1], y.shape[1], invalid.shape[1]).to(device)
        optimizer = torch.optim.AdamW(member.parameters(), lr=1e-3, weight_decay=1e-4)
        positive = np.sum(invalid[train_indexes], axis=0)
        negative = len(train_indexes) - positive
        pos_weight = torch.as_tensor(
            np.clip(negative / np.maximum(positive, 1.0), 1.0, 30.0),
            dtype=torch.float32,
            device=device,
        )
        bootstrap = rng.choice(train_indexes, size=len(train_indexes), replace=True)
        best_state: dict[str, Any] | None = None
        best_loss = float("inf")
        patience = 0
        for epoch in range(max(1, epochs)):
            if progress and epoch % 5 == 0:
                progress(
                    (member_index + epoch / max(1, epochs)) / max(1, members),
                    f"Training surrogate {member_index + 1}/{members}, epoch {epoch + 1}/{epochs}",
                )
            rng.shuffle(bootstrap)
            member.train()
            for start in range(0, len(bootstrap), max(8, batch_size)):
                indexes = bootstrap[start: start + max(8, batch_size)]
                xb = torch.as_tensor(x[indexes], dtype=torch.float32, device=device)
                yb = torch.as_tensor(y[indexes], dtype=torch.float32, device=device)
                mb = torch.as_tensor(mask[indexes], dtype=torch.float32, device=device)
                ib = torch.as_tensor(invalid[indexes], dtype=torch.float32, device=device)
                prediction, invalid_logits = member(xb)
                regression = nn.functional.smooth_l1_loss(prediction, yb, reduction="none")
                regression_loss = (regression * mb).sum() / mb.sum().clamp_min(1.0)
                invalid_loss = nn.functional.binary_cross_entropy_with_logits(
                    invalid_logits,
                    ib,
                    pos_weight=pos_weight,
                )
                loss = regression_loss + 0.5 * invalid_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(member.parameters(), max_norm=5.0)
                optimizer.step()
            validation_loss = _validation_loss(member, x, y, mask, invalid, validation_indexes, device)
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = copy.deepcopy(member.state_dict())
                patience = 0
            else:
                patience += 1
            if patience >= 35:
                break
        if best_state is not None:
            member.load_state_dict(best_state)
        member.eval()
        filename = f"surrogate_{member_index + 1}.pt"
        torch.save(member.state_dict(), artifact_dir / filename)
        trained_members.append(member)
        member_files.append(filename)
        member_validation.append(best_loss)

    np.savez_compressed(
        artifact_dir / "scalers.npz",
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        metric_mean=metric_mean.astype(np.float32),
        metric_std=metric_std.astype(np.float32),
    )
    model_id = artifact_dir.name
    initial_manifest = {
        "model_id": model_id,
        "schema_version": 1,
        "created_at": time.time(),
        "feature_count": int(x.shape[1]),
        "metric_count": int(y.shape[1]),
        "invalid_label_count": int(invalid.shape[1]),
        "invalid_label_layout": ["protection", "invalid_transient", "invalid_bode"],
        "invalid_positive_counts": [int(value) for value in np.sum(dataset.invalid_labels > 0, axis=0)],
        "zero_protection_calibration": "Jeffreys-prior posterior mean when no protection event is observed",
        "member_files": member_files,
        "member_validation_loss": member_validation,
        "sample_count": dataset.size,
        "training_sample_count": int(len(train_indexes)),
        "validation_sample_count": int(len(validation_indexes)),
        "validation_split": "held-out run with candidate-key purge",
        "training_groups": sorted(set(dataset.groups[train_indexes].tolist())),
        "validation_groups": sorted(set(dataset.groups[validation_indexes].tolist())),
        "operating_signature": operating_signature,
        "dependency": dependency,
        "training_device": device,
        "torch_cpu_threads": cpu_threads,
        "prediction_target": "absolute_metrics",
        "invalid_probability_threshold": 0.25,
        "accepted": False,
    }
    atomic_write_json(artifact_dir / "manifest.json", initial_manifest)
    ensemble = SurrogateEnsemble(
        artifact_dir,
        initial_manifest,
        trained_members,
        feature_mean,
        feature_std,
        metric_mean,
        metric_std,
        device,
    )
    evaluation = evaluate_surrogate(ensemble, dataset, config, validation_indexes)
    manifest = {**initial_manifest, **evaluation}
    manifest["files_sha256"] = _artifact_hashes(artifact_dir, [*member_files, "scalers.npz"])
    atomic_write_json(artifact_dir / "manifest.json", manifest)
    ensemble.manifest = manifest
    if progress:
        progress(1.0, "Surrogate training complete")
    return ensemble


def evaluate_surrogate(
    ensemble: SurrogateEnsemble,
    dataset: DrlDataset,
    config: TuningConfig,
    indexes: np.ndarray | list[int] | None = None,
) -> dict[str, Any]:
    selected = np.asarray(indexes if indexes is not None else np.arange(dataset.size), dtype=int)
    predictions = ensemble.predict_features(dataset.features[selected])
    mean = predictions["metric_mean"]
    std = predictions["metric_std"]
    truth = dataset.metrics[selected]
    mask = dataset.metric_mask[selected]
    repeat_mad = _repeat_metric_mad(dataset)
    interval_std = np.sqrt(std ** 2 + (1.4826 * repeat_mad.reshape(1, -1)) ** 2)
    mae: dict[str, float | None] = {}
    coverage_values: list[float] = []
    for metric_index, name in enumerate(METRIC_FIELDS):
        observed = mask[:, metric_index] > 0
        if not np.any(observed):
            mae[name] = None
            continue
        absolute = np.abs(mean[observed, metric_index] - truth[observed, metric_index])
        mae[name] = float(np.mean(absolute))
        lower = mean[observed, metric_index] - 1.645 * interval_std[observed, metric_index]
        upper = mean[observed, metric_index] + 1.645 * interval_std[observed, metric_index]
        coverage_values.extend(((truth[observed, metric_index] >= lower) & (truth[observed, metric_index] <= upper)).tolist())

    predicted_scores = []
    true_scores = dataset.scores[selected]
    for row in mean:
        payload = _metric_payload(row)
        predicted_scores.append(relabeled_score(payload, config.targets)[0])
    rank_correlation = _spearman(np.asarray(predicted_scores), np.asarray(true_scores))
    uncertainty = np.asarray(predictions["uncertainty"])
    actual_unsafe = np.max(dataset.invalid_labels[selected], axis=1) > 0
    safe_uncertainty = uncertainty[~actual_unsafe]
    uncertainty_threshold = float(np.quantile(safe_uncertainty, 0.95)) if safe_uncertainty.size else 1.0
    invalid_probability = np.asarray(predictions["invalid_probability"])
    invalid_threshold = ensemble.invalid_probability_threshold
    predicted_unsafe = (
        np.max(invalid_probability, axis=1) >= invalid_threshold
    ) | (uncertainty > uncertainty_threshold)
    safety_recall = float(np.mean(predicted_unsafe[actual_unsafe])) if np.any(actual_unsafe) else 1.0
    validity_specificity = float(np.mean(~predicted_unsafe[~actual_unsafe])) if np.any(~actual_unsafe) else 0.0
    interval_coverage = float(np.mean(coverage_values)) if coverage_values else 0.0
    fixed_thresholds = np.asarray([0.25, 0.25, 0.75, 0.75, 5.0, 20.0, float("inf")])
    thresholds = np.maximum(fixed_thresholds, repeat_mad * 1.25)
    mae_values = np.asarray([mae[name] if mae[name] is not None else float("inf") for name in METRIC_FIELDS])
    metric_gate = bool(np.all(mae_values[:6] <= thresholds[:6]))
    accepted = bool(
        rank_correlation >= 0.65
        and interval_coverage >= 0.85
        and safety_recall >= 1.0
        and validity_specificity >= 0.30
        and metric_gate
    )
    return {
        "accepted": accepted,
        "acceptance": {
            "penalty_spearman": rank_correlation,
            "interval_coverage_90": interval_coverage,
            "safety_recall": safety_recall,
            "validity_specificity": validity_specificity,
            "invalid_probability_threshold": invalid_threshold,
            "metric_gate": metric_gate,
            "metric_mean_absolute_error": mae,
            "metric_error_thresholds": {
                name: None if not math.isfinite(float(thresholds[index])) else float(thresholds[index])
                for index, name in enumerate(METRIC_FIELDS)
            },
            "repeat_measurement_mad": {
                name: float(repeat_mad[index]) for index, name in enumerate(METRIC_FIELDS)
            },
        },
        "uncertainty_threshold": uncertainty_threshold,
    }


def _surrogate_network_type(torch_module: Any) -> type:
    nn = torch_module.nn

    class SurrogateNetwork(nn.Module):
        def __init__(self, feature_count: int, metric_count: int, invalid_count: int):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(feature_count, 128),
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.SiLU(),
            )
            self.metric_head = nn.Linear(64, metric_count)
            self.invalid_head = nn.Linear(64, invalid_count)

        def forward(self, value: Any) -> tuple[Any, Any]:
            hidden = self.backbone(value)
            return self.metric_head(hidden), self.invalid_head(hidden)

    return SurrogateNetwork


def _validation_loss(
    member: Any,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    invalid: np.ndarray,
    indexes: np.ndarray,
    device: str,
) -> float:
    import torch
    from torch import nn

    member.eval()
    with torch.no_grad():
        xb = torch.as_tensor(x[indexes], dtype=torch.float32, device=device)
        yb = torch.as_tensor(y[indexes], dtype=torch.float32, device=device)
        mb = torch.as_tensor(mask[indexes], dtype=torch.float32, device=device)
        ib = torch.as_tensor(invalid[indexes], dtype=torch.float32, device=device)
        prediction, invalid_logits = member(xb)
        regression = nn.functional.smooth_l1_loss(prediction, yb, reduction="none")
        regression_loss = (regression * mb).sum() / mb.sum().clamp_min(1.0)
        invalid_loss = nn.functional.binary_cross_entropy_with_logits(invalid_logits, ib)
        return float((regression_loss + 0.5 * invalid_loss).item())


def _group_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(groups.tolist()))
    rng = np.random.default_rng(seed)
    if len(unique) >= 2:
        shuffled = list(unique)
        rng.shuffle(shuffled)
        validation_count = max(1, int(round(len(shuffled) * 0.20)))
        validation_groups = set(shuffled[-validation_count:])
        validation = np.asarray([index for index, group in enumerate(groups) if group in validation_groups], dtype=int)
        training = np.asarray([index for index, group in enumerate(groups) if group not in validation_groups], dtype=int)
    else:
        indexes = rng.permutation(len(groups))
        split = max(1, int(round(len(indexes) * 0.20)))
        validation = np.asarray(indexes[:split], dtype=int)
        training = np.asarray(indexes[split:], dtype=int)
    if training.size == 0 or validation.size == 0:
        raise RuntimeError("DRL dataset cannot be split into non-empty training and validation sets.")
    return training, validation


def _group_candidate_split(dataset: DrlDataset, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from .common import candidate_key

    unique_groups = sorted(set(dataset.groups.tolist()))
    if len(unique_groups) < 2:
        training, validation = _group_split(dataset.groups, seed)
    else:
        rng = np.random.default_rng(seed)
        shuffled = list(unique_groups)
        rng.shuffle(shuffled)
        required_metrics = {
            index for index in range(6) if np.any(dataset.metric_mask[:, index] > 0)
        }
        required_invalid = {
            index for index in range(dataset.invalid_labels.shape[1]) if np.any(dataset.invalid_labels[:, index] > 0)
        }
        covered_metrics: set[int] = set()
        covered_invalid: set[int] = set()
        validation_groups: list[str] = []
        while covered_metrics != required_metrics or covered_invalid != required_invalid:
            best_group = None
            best_gain = 0
            for group in shuffled:
                if group in validation_groups:
                    continue
                indexes = np.where(dataset.groups == group)[0]
                metric_gain = sum(
                    index not in covered_metrics and np.any(dataset.metric_mask[indexes, index] > 0)
                    for index in required_metrics
                )
                invalid_gain = sum(
                    index not in covered_invalid and np.any(dataset.invalid_labels[indexes, index] > 0)
                    for index in required_invalid
                )
                gain = metric_gain + invalid_gain * 2
                if gain > best_gain:
                    best_group = group
                    best_gain = gain
            if best_group is None or best_gain <= 0 or len(validation_groups) >= len(unique_groups) - 1:
                break
            validation_groups.append(best_group)
            indexes = np.where(dataset.groups == best_group)[0]
            covered_metrics.update(
                index for index in required_metrics if np.any(dataset.metric_mask[indexes, index] > 0)
            )
            covered_invalid.update(
                index for index in required_invalid if np.any(dataset.invalid_labels[indexes, index] > 0)
            )
        minimum_groups = max(1, int(round(len(unique_groups) * 0.20)))
        for group in shuffled:
            if len(validation_groups) >= minimum_groups:
                break
            if group not in validation_groups:
                validation_groups.append(group)
        validation_group_set = set(validation_groups)
        validation = np.asarray(
            [index for index, group in enumerate(dataset.groups) if group in validation_group_set],
            dtype=int,
        )
        training = np.asarray(
            [index for index, group in enumerate(dataset.groups) if group not in validation_group_set],
            dtype=int,
        )
        if covered_metrics != required_metrics or covered_invalid != required_invalid:
            raise RuntimeError(
                "Run-grouped validation cannot cover every available metric and invalid class."
            )
    validation_candidates = {candidate_key(dataset.candidates[index]) for index in validation}
    purged_training = np.asarray(
        [index for index in training if candidate_key(dataset.candidates[index]) not in validation_candidates],
        dtype=int,
    )
    if purged_training.size < 20:
        raise RuntimeError(
            "Run/candidate grouped validation left fewer than 20 training samples; more independent runs are required."
        )
    return purged_training, validation


def _masked_mean_std(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(values.shape[1], dtype=np.float32)
    stds = np.ones(values.shape[1], dtype=np.float32)
    for column in range(values.shape[1]):
        observed = values[mask[:, column] > 0, column]
        if observed.size:
            means[column] = float(np.mean(observed))
            stds[column] = max(float(np.std(observed)), 1e-3)
    return means, stds


def _repeat_metric_mad(dataset: DrlDataset) -> np.ndarray:
    groups: dict[tuple[Any, ...], list[int]] = {}
    from .common import candidate_key

    for index, candidate in enumerate(dataset.candidates):
        groups.setdefault(candidate_key(candidate), []).append(index)
    deviations: list[list[float]] = [[] for _ in METRIC_FIELDS]
    for indexes in groups.values():
        if len(indexes) < 2:
            continue
        for metric_index in range(len(METRIC_FIELDS)):
            observed = [
                float(dataset.metrics[index, metric_index])
                for index in indexes
                if dataset.metric_mask[index, metric_index] > 0
            ]
            if len(observed) < 2:
                continue
            median = float(np.median(observed))
            deviations[metric_index].extend(abs(value - median) for value in observed)
    return np.asarray(
        [float(np.median(items)) if items else 0.0 for items in deviations],
        dtype=np.float64,
    )


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.size < 2 or second.size != first.size:
        return 0.0
    first_rank = _rank(first)
    second_rank = _rank(second)
    if np.std(first_rank) <= 1e-12 or np.std(second_rank) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop - 1) / 2.0
        ranks[order[start:stop]] = rank
        start = stop
    return ranks


def _metric_payload(values: np.ndarray) -> dict[str, Any]:
    mapping = vector_to_metric_mapping(values)
    return {
        "overshoot_pct": mapping["overshoot_pct"],
        "undershoot_pct": mapping["undershoot_pct"],
        "overshoot_settling_time_s": mapping["overshoot_settling_time_us"] * 1e-6,
        "undershoot_settling_time_s": mapping["undershoot_settling_time_us"] * 1e-6,
        "phase_margin_deg": mapping["phase_margin_deg"],
        "crossover_frequency_hz": mapping["crossover_frequency_khz"] * 1e3,
        "gain_margin_db": mapping["gain_margin_db"],
    }


def _artifact_hashes(root: Path, filenames: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename in filenames:
        path = root / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[filename] = digest
    return hashes


def _verify_artifact_hashes(root: Path, expected: Any) -> None:
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError(f"DRL model '{root.name}' has no artifact hashes and is not safe to load.")
    for filename, expected_hash in expected.items():
        path = root / str(filename)
        if not path.is_file():
            raise RuntimeError(f"DRL model artifact is missing: {path}")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != str(expected_hash):
            raise RuntimeError(f"DRL model artifact hash mismatch: {path.name}")


def _cpu_thread_count() -> int:
    default = max(1, (os.cpu_count() or 2) - 1)
    try:
        requested = int(os.environ.get("DRL_CPU_THREADS", default))
    except (TypeError, ValueError):
        requested = default
    return max(1, min(8, requested))
