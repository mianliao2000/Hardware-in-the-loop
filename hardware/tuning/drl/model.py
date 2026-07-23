"""Bootstrap neural surrogate ensemble for fixed-operating-point autotuning."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np

from ..models import TuningConfig
from .common import (
    ACTION_FIELDS,
    METRIC_FIELDS,
    SCHEMA_VERSION,
    atomic_write_json,
    candidate_key,
    relabeled_score,
    vector_to_metric_mapping,
)
from .dataset import DrlDataset


# Capacity sweep 2026-07-16: this topology was the strongest research
# candidate while using substantially fewer parameters than the legacy model.
# Existing artifacts without architecture metadata must still load with the
# exact topology they were originally trained with.
DEFAULT_SURROGATE_HIDDEN_SIZES = (96, 64, 32)
LEGACY_SURROGATE_HIDDEN_SIZES = (128, 128, 64)
SAFETY_LABEL_COUNT = 3


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
        feasibility_calibrator: dict[str, np.ndarray] | None = None,
    ):
        self.artifact_dir = artifact_dir
        self.manifest = manifest
        self.members = members
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
        self.metric_mean = metric_mean.astype(np.float32)
        self.metric_std = metric_std.astype(np.float32)
        self.device = device
        self.feasibility_calibrator = feasibility_calibrator or {}

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

    @property
    def robust_pass_probability_threshold(self) -> float:
        return float(self.manifest.get("robust_pass_probability_threshold", 0.80))

    def predict_features(self, features: np.ndarray) -> dict[str, np.ndarray]:
        import torch

        values = np.asarray(features, dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        values = values.copy()
        baseline_imputation = np.asarray(
            self.manifest.get("baseline_imputation_values", []),
            dtype=np.float32,
        )
        action_count = len(ACTION_FIELDS)
        baseline_start = action_count
        baseline_stop = baseline_start + len(METRIC_FIELDS)
        mask_stop = baseline_stop + len(METRIC_FIELDS)
        if values.shape[1] >= mask_stop and baseline_imputation.shape == (len(METRIC_FIELDS),):
            baseline_mask = values[:, baseline_stop:mask_stop]
            baseline_values = values[:, baseline_start:baseline_stop]
            for column in range(7):
                missing = baseline_mask[:, column] <= 0
                baseline_values[missing, column] = baseline_imputation[column]
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
                    metric = metric_scaled_np * self.metric_std + values[:, baseline_start:baseline_stop]
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
            sample_count = max(
                1,
                int(self.manifest.get("training_sample_count", self.manifest.get("sample_count", 1))),
            )
            invalid_mean[:, 0] = 0.5 / (sample_count + 1.0)
        if invalid_mean.shape[1] > SAFETY_LABEL_COUNT:
            pass_probability = invalid_mean[:, SAFETY_LABEL_COUNT]
        else:
            # Old three-label artifacts do not contain a repeatability head.
            # Keep them fail-closed for robust-pass selection.
            pass_probability = np.zeros(values.shape[0], dtype=np.float32)
        local = self._local_feasibility_prediction(values[:, :action_count])
        if local is not None:
            confidence = local["confidence"].reshape(-1, 1)
            metric_mean = confidence * local["metric_mean"] + (1.0 - confidence) * metric_mean
            if "invalid_probability" in local:
                invalid_mean[:, :SAFETY_LABEL_COUNT] = (
                    confidence * local["invalid_probability"]
                    + (1.0 - confidence) * invalid_mean[:, :SAFETY_LABEL_COUNT]
                )
            pass_probability = (
                local["confidence"] * local["pass_probability"]
                + (1.0 - local["confidence"]) * pass_probability
            )
        safety_invalid = invalid_mean[:, :SAFETY_LABEL_COUNT]
        safety_probability = 1.0 - safety_invalid[:, 0]
        validity_probability = 1.0 - np.max(safety_invalid, axis=1)
        normalized_uncertainty = metric_std / np.maximum(np.abs(self.metric_std), 1e-6)
        uncertainty = np.mean(normalized_uncertainty, axis=1)
        if local is not None:
            uncertainty = np.maximum(
                uncertainty,
                np.clip(local["nearest_distance"] / 0.30, 0.0, 1.0) * 0.25,
            )
        return {
            "metric_mean": metric_mean.astype(np.float64),
            "metric_std": metric_std.astype(np.float64),
            "invalid_probability": invalid_mean.astype(np.float64),
            "safety_probability": safety_probability.astype(np.float64),
            "validity_probability": validity_probability.astype(np.float64),
            "pass_probability": pass_probability.astype(np.float64),
            "local_feasibility_confidence": (
                local["confidence"].astype(np.float64)
                if local is not None
                else np.zeros(values.shape[0], dtype=np.float64)
            ),
            "nearest_calibration_distance": (
                local["nearest_distance"].astype(np.float64)
                if local is not None
                else np.full(values.shape[0], np.inf, dtype=np.float64)
            ),
            "uncertainty": uncertainty.astype(np.float64),
        }

    def _local_feasibility_prediction(
        self,
        actions: np.ndarray,
    ) -> dict[str, np.ndarray] | None:
        calibration_actions = np.asarray(
            self.feasibility_calibrator.get("actions", []), dtype=np.float64
        )
        if calibration_actions.ndim != 2 or calibration_actions.shape[0] == 0:
            return None
        pass_probability = np.asarray(
            self.feasibility_calibrator.get("pass_probability", []), dtype=np.float64
        )
        metrics = np.asarray(
            self.feasibility_calibrator.get("metrics", []), dtype=np.float64
        )
        support = np.asarray(
            self.feasibility_calibrator.get("support", np.ones(len(calibration_actions))),
            dtype=np.float64,
        )
        if pass_probability.shape != (len(calibration_actions),) or metrics.shape != (
            len(calibration_actions),
            len(METRIC_FIELDS),
        ):
            return None
        invalid_probability = np.asarray(
            self.feasibility_calibrator.get("invalid_probability", []), dtype=np.float64
        )
        has_invalid_probability = invalid_probability.shape == (
            len(calibration_actions),
            SAFETY_LABEL_COUNT,
        )
        query = np.asarray(actions, dtype=np.float64)
        distances = np.sqrt(
            np.mean((query[:, None, :] - calibration_actions[None, :, :]) ** 2, axis=2)
        )
        neighbor_count = min(7, calibration_actions.shape[0])
        nearest_indexes = np.argpartition(distances, neighbor_count - 1, axis=1)[:, :neighbor_count]
        nearest_distances = np.take_along_axis(distances, nearest_indexes, axis=1)
        kernel = np.exp(-0.5 * (nearest_distances / 0.12) ** 2)
        kernel *= np.sqrt(np.maximum(1.0, support[nearest_indexes]))
        # An exact raw-key measurement is stronger evidence than nearby
        # continuous-space interpolation.  L/Lc round trips can introduce a
        # tiny normalized float difference, hence the small nonzero tolerance.
        exact_rows = np.min(nearest_distances, axis=1) <= 1e-3
        if np.any(exact_rows):
            for row in np.flatnonzero(exact_rows):
                nearest_position = int(np.argmin(nearest_distances[row]))
                kernel[row, :] = 0.0
                kernel[row, nearest_position] = max(
                    1.0, math.sqrt(max(1.0, support[nearest_indexes[row, nearest_position]]))
                )
        kernel_sum = np.maximum(np.sum(kernel, axis=1, keepdims=True), 1e-12)
        weights = kernel / kernel_sum
        local_pass = np.sum(weights * pass_probability[nearest_indexes], axis=1)
        local_metrics = np.sum(weights[:, :, None] * metrics[nearest_indexes], axis=1)
        nearest = np.min(distances, axis=1)
        confidence = np.exp(-0.5 * (nearest / 0.18) ** 2)
        result = {
            "pass_probability": local_pass,
            "metric_mean": local_metrics,
            "nearest_distance": nearest,
            "confidence": confidence,
        }
        if has_invalid_probability:
            result["invalid_probability"] = np.sum(
                weights[:, :, None] * invalid_probability[nearest_indexes], axis=1
            )
        return result

    @classmethod
    def load(cls, artifact_dir: Path, device: str | None = None) -> "SurrogateEnsemble":
        require_ml_dependencies()
        import torch

        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"DRL model manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _verify_artifact_hashes(artifact_dir, manifest.get("files_sha256"))
        manifest_action_fields = manifest.get("action_fields")
        if manifest_action_fields is not None and tuple(manifest_action_fields) != ACTION_FIELDS:
            raise RuntimeError(
                f"DRL model '{artifact_dir.name}' uses an incompatible action schema: "
                f"expected {list(ACTION_FIELDS)}, found {list(manifest_action_fields)}."
            )
        manifest_action_count = manifest.get("action_count")
        if manifest_action_count is not None and int(manifest_action_count) != len(ACTION_FIELDS):
            raise RuntimeError(
                f"DRL model '{artifact_dir.name}' has {manifest_action_count} action dimensions; "
                f"the current runtime requires {len(ACTION_FIELDS)}."
            )
        manifest_metric_fields = manifest.get("metric_fields")
        if manifest_metric_fields is not None and tuple(manifest_metric_fields) != METRIC_FIELDS:
            raise RuntimeError(
                f"DRL model '{artifact_dir.name}' uses an incompatible metric schema: "
                f"expected {list(METRIC_FIELDS)}, found {list(manifest_metric_fields)}."
            )
        manifest_metric_count = manifest.get("metric_count")
        if manifest_metric_count is not None and int(manifest_metric_count) != len(METRIC_FIELDS):
            raise RuntimeError(
                f"DRL model '{artifact_dir.name}' predicts {manifest_metric_count} metrics; "
                f"the current runtime requires {len(METRIC_FIELDS)}."
            )
        scaler = np.load(artifact_dir / "scalers.npz")
        calibrator_path = artifact_dir / "feasibility_calibrator.npz"
        feasibility_calibrator = (
            {key: value for key, value in np.load(calibrator_path).items()}
            if calibrator_path.exists()
            else None
        )
        selected_device = device or "cpu"
        network_type = _surrogate_network_type(torch)
        hidden_sizes = _manifest_hidden_sizes(manifest)
        members = []
        member_files = manifest.get("member_files") or []
        for filename in member_files:
            member = network_type(
                int(manifest["feature_count"]),
                int(manifest["metric_count"]),
                int(manifest.get("invalid_label_count", 3)),
                hidden_sizes=hidden_sizes,
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
            feasibility_calibrator=feasibility_calibrator,
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
    hidden_sizes: Sequence[int] | None = None,
    train_indexes: np.ndarray | Sequence[int] | None = None,
    validation_indexes: np.ndarray | Sequence[int] | None = None,
    evaluation_indexes: np.ndarray | Sequence[int] | None = None,
    early_stopping_patience: int = 35,
) -> SurrogateEnsemble:
    dependency = require_ml_dependencies()
    import torch
    from torch import nn

    if dataset.size < 20:
        raise RuntimeError(f"At least 20 DRL samples are required; found {dataset.size}.")
    resolved_hidden_sizes = _normalize_hidden_sizes(hidden_sizes)
    training_started = time.perf_counter()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu"
    cpu_threads = _cpu_thread_count()
    torch.set_num_threads(cpu_threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    explicit_split = train_indexes is not None or validation_indexes is not None or evaluation_indexes is not None
    train_indexes, validation_indexes, evaluation_indexes = _resolve_training_splits(
        dataset,
        seed,
        train_indexes,
        validation_indexes,
        evaluation_indexes,
    )
    training_features, baseline_imputation = _training_fold_features(dataset, train_indexes)
    feature_mean = np.mean(training_features[train_indexes], axis=0)
    feature_std = np.std(training_features[train_indexes], axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    metric_mean, metric_std = _masked_mean_std(dataset.metrics[train_indexes], dataset.metric_mask[train_indexes])
    x = np.clip((training_features - feature_mean) / feature_std, -6.0, 6.0).astype(np.float32)
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
    member_epochs: list[int] = []
    rng = np.random.default_rng(seed)
    for member_index in range(max(1, members)):
        if progress:
            progress(member_index / max(1, members), f"Training surrogate {member_index + 1}/{members}")
        member = network_type(
            x.shape[1],
            y.shape[1],
            invalid.shape[1],
            hidden_sizes=resolved_hidden_sizes,
        ).to(device)
        optimizer = torch.optim.AdamW(member.parameters(), lr=1e-3, weight_decay=1e-4)
        # Runs can differ by an order of magnitude in length (for example a
        # 500-point exploration run beside a 29-point boundary-validation
        # run).  Sample complete run groups equally so the long run cannot
        # erase the scarce confirmed/high-BW examples from short runs.
        bootstrap = _group_balanced_bootstrap(train_indexes, dataset.groups, rng)
        positive = np.sum(invalid[bootstrap], axis=0)
        negative = len(bootstrap) - positive
        pos_weight = torch.as_tensor(
            np.clip(negative / np.maximum(positive, 1.0), 1.0, 30.0),
            dtype=torch.float32,
            device=device,
        )
        best_state: dict[str, Any] | None = None
        best_loss = float("inf")
        patience = 0
        epochs_ran = 0
        for epoch in range(max(1, epochs)):
            epochs_ran = epoch + 1
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
            if patience >= max(1, int(early_stopping_patience)):
                break
        if best_state is not None:
            member.load_state_dict(best_state)
        member.eval()
        filename = f"surrogate_{member_index + 1}.pt"
        torch.save(member.state_dict(), artifact_dir / filename)
        trained_members.append(member)
        member_files.append(filename)
        member_validation.append(best_loss)
        member_epochs.append(epochs_ran)

    np.savez_compressed(
        artifact_dir / "scalers.npz",
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        metric_mean=metric_mean.astype(np.float32),
        metric_std=metric_std.astype(np.float32),
    )
    training_calibrator = _build_feasibility_calibrator(dataset, train_indexes)
    np.savez_compressed(
        artifact_dir / "feasibility_calibrator.npz",
        **training_calibrator,
    )
    model_id = artifact_dir.name
    parameter_count_per_member = _trainable_parameter_count(trained_members[0])
    dataset_hash = _dataset_hash(dataset)
    training_indexes_hash = _indexes_hash(train_indexes)
    validation_indexes_hash = _indexes_hash(validation_indexes)
    evaluation_indexes_hash = _indexes_hash(evaluation_indexes)
    split_hash = _split_hash(train_indexes, validation_indexes, evaluation_indexes)
    initial_manifest = {
        "model_id": model_id,
        "schema_version": 1,
        "drl_schema_version": SCHEMA_VERSION,
        "action_fields": list(ACTION_FIELDS),
        "action_count": len(ACTION_FIELDS),
        "metric_fields": list(METRIC_FIELDS),
        "created_at": time.time(),
        "feature_count": int(x.shape[1]),
        "metric_count": int(y.shape[1]),
        "invalid_label_count": int(invalid.shape[1]),
        "invalid_label_layout": [
            "protection",
            "invalid_transient",
            "invalid_bode",
            "robust_pass_probability",
        ],
        "invalid_positive_counts": [
            int(value) for value in np.sum(dataset.invalid_labels[train_indexes] > 0, axis=0)
        ],
        "zero_protection_calibration": "Jeffreys-prior posterior mean when no protection event is observed",
        "member_files": member_files,
        "member_validation_loss": member_validation,
        "member_epochs_completed": member_epochs,
        "ensemble_members": len(trained_members),
        "hidden_sizes": list(resolved_hidden_sizes),
        "activation": "SiLU",
        "parameter_count_per_member": parameter_count_per_member,
        "trainable_parameter_count": parameter_count_per_member * len(trained_members),
        "sample_count": dataset.size,
        "training_sample_count": int(len(train_indexes)),
        "validation_sample_count": int(len(validation_indexes)),
        "evaluation_sample_count": int(len(evaluation_indexes)),
        "validation_split": "explicit grouped indexes" if explicit_split else "held-out run with candidate-key purge",
        "training_sampling": "run_group_balanced_bootstrap",
        "training_groups": sorted(set(dataset.groups[train_indexes].tolist())),
        "validation_groups": sorted(set(dataset.groups[validation_indexes].tolist())),
        "evaluation_groups": sorted(set(dataset.groups[evaluation_indexes].tolist())),
        "evaluation_reuses_validation": bool(np.array_equal(validation_indexes, evaluation_indexes)),
        "baseline_imputation_values": baseline_imputation.tolist(),
        "baseline_imputation_policy": "training-partition median with missing mask preserved",
        "dataset_hash": dataset_hash,
        "training_indexes_hash": training_indexes_hash,
        "validation_indexes_hash": validation_indexes_hash,
        "evaluation_indexes_hash": evaluation_indexes_hash,
        "split_hash": split_hash,
        "training_seed": int(seed),
        "epochs_requested": max(1, int(epochs)),
        "batch_size": max(8, int(batch_size)),
        "early_stopping_patience": max(1, int(early_stopping_patience)),
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "operating_signature": operating_signature,
        "dependency": dependency,
        "training_device": device,
        "torch_cpu_threads": cpu_threads,
        "invalid_probability_threshold": 0.25,
        "robust_pass_probability_threshold": 0.80,
        "prediction_target": "raw_key_robust_metrics",
        "feasibility_calibrator": "normalized-action Gaussian kNN",
        "feasibility_calibration_training_keys": int(len(training_calibrator["actions"])),
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
        feasibility_calibrator=training_calibrator,
    )
    evaluation = evaluate_surrogate(
        ensemble,
        dataset,
        config,
        evaluation_indexes,
        calibration_indexes=train_indexes,
    )
    manifest = {**initial_manifest, **evaluation}
    deployment_calibrator = _build_feasibility_calibrator(
        dataset,
        np.arange(dataset.size, dtype=int),
    )
    np.savez_compressed(
        artifact_dir / "feasibility_calibrator.npz",
        **deployment_calibrator,
    )
    # Persist the all-data calibrator for a separately loaded deployment
    # artifact, but keep the returned in-memory ensemble on the training-fold
    # calibrator.  ``train_safe_sac_policy`` is called with this object in the
    # same process; switching it to the deployment calibrator here would let
    # SAC indirectly see held-out run labels before policy validation.
    manifest["feasibility_calibration_deployment_keys"] = int(
        len(deployment_calibrator["actions"])
    )
    manifest["feasibility_calibration_scope"] = (
        "training partition in-memory for policy training/evaluation; all exact-condition data after artifact reload"
    )
    manifest["training_wall_time_seconds"] = float(time.perf_counter() - training_started)
    manifest["inference_latency_p95_ms"] = _inference_latency_p95_ms(
        ensemble,
        dataset.features[evaluation_indexes],
    )
    manifest["files_sha256"] = _artifact_hashes(
        artifact_dir,
        [*member_files, "scalers.npz", "feasibility_calibrator.npz"],
    )
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
    calibration_indexes: np.ndarray | Sequence[int] | None = None,
) -> dict[str, Any]:
    selected = np.asarray(indexes if indexes is not None else np.arange(dataset.size), dtype=int)
    calibration_selected = np.asarray(
        calibration_indexes if calibration_indexes is not None else selected,
        dtype=int,
    )
    predictions = ensemble.predict_features(dataset.features[selected])
    mean = predictions["metric_mean"]
    std = predictions["metric_std"]
    truth = dataset.metrics[selected]
    mask = dataset.metric_mask[selected]
    repeat_mad = _repeat_metric_mad(dataset, calibration_selected)
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
    actual_unsafe = np.max(dataset.invalid_labels[selected, :SAFETY_LABEL_COUNT], axis=1) > 0
    calibration_predictions = ensemble.predict_features(dataset.features[calibration_selected])
    calibration_uncertainty = np.asarray(calibration_predictions["uncertainty"])
    calibration_unsafe = np.max(
        dataset.invalid_labels[calibration_selected, :SAFETY_LABEL_COUNT], axis=1
    ) > 0
    safe_uncertainty = calibration_uncertainty[~calibration_unsafe]
    uncertainty_threshold = float(np.quantile(safe_uncertainty, 0.95)) if safe_uncertainty.size else 1.0
    invalid_probability = np.asarray(predictions["invalid_probability"])
    calibration_invalid_probability = np.asarray(
        calibration_predictions["invalid_probability"]
    )
    calibration_invalid_score = np.max(
        calibration_invalid_probability[:, :SAFETY_LABEL_COUNT], axis=1
    )
    if np.any(calibration_unsafe):
        # Largest threshold that still catches every observed unsafe training
        # example. Pos-weighted BCE logits are not calibrated probabilities,
        # so a fixed 0.25 threshold otherwise rejects nearly every safe point.
        invalid_threshold = float(
            np.clip(np.min(calibration_invalid_score[calibration_unsafe]) - 1e-6, 0.25, 0.95)
        )
    else:
        invalid_threshold = float(
            np.clip(np.quantile(calibration_invalid_score, 0.99), 0.25, 0.95)
        )
    predicted_unsafe = (
        np.max(invalid_probability[:, :SAFETY_LABEL_COUNT], axis=1) >= invalid_threshold
    ) | (uncertainty > uncertainty_threshold)
    safety_recall = float(np.mean(predicted_unsafe[actual_unsafe])) if np.any(actual_unsafe) else 1.0
    validity_specificity = float(np.mean(~predicted_unsafe[~actual_unsafe])) if np.any(~actual_unsafe) else 0.0
    interval_coverage = float(np.mean(coverage_values)) if coverage_values else 0.0
    fixed_thresholds = np.asarray(
        [0.25, 0.25, 0.75, 0.75, 5.0, 20.0, float("inf"), 10.0]
    )
    thresholds = np.maximum(fixed_thresholds, repeat_mad * 1.25)
    mae_values = np.asarray([mae[name] if mae[name] is not None else float("inf") for name in METRIC_FIELDS])
    # Gain margin remains informational; the new shape penalty is a gated
    # optimization target because proposal ranking must distinguish a smooth
    # roll-off from a rebound even when both candidates fail the hard gate.
    gated_metric_indexes = (0, 1, 2, 3, 4, 5, 7)
    metric_gate = bool(
        all(mae_values[index] <= thresholds[index] for index in gated_metric_indexes)
    )
    pass_probability = np.asarray(predictions.get("pass_probability", np.zeros(len(selected))))
    true_pass_probability = np.asarray(dataset.passed[selected], dtype=np.float64)
    pass_brier = float(np.mean((pass_probability - true_pass_probability) ** 2))
    calibration_pass_probability = np.asarray(
        calibration_predictions.get("pass_probability", np.zeros(len(calibration_selected)))
    )
    calibration_truth = np.asarray(dataset.passed[calibration_selected], dtype=np.float64) >= 0.80
    robust_pass_threshold = _high_precision_pass_threshold(
        calibration_pass_probability,
        calibration_truth,
    )
    predicted_robust_pass = pass_probability >= robust_pass_threshold
    true_robust_pass = true_pass_probability >= 0.80
    robust_pass_precision = float(
        np.mean(true_robust_pass[predicted_robust_pass])
    ) if np.any(predicted_robust_pass) else 0.0
    robust_pass_recall = float(
        np.mean(predicted_robust_pass[true_robust_pass])
    ) if np.any(true_robust_pass) else 1.0
    accepted = bool(
        rank_correlation >= 0.65
        and interval_coverage >= 0.85
        and safety_recall >= 1.0
        and validity_specificity >= 0.30
        and metric_gate
        and pass_brier <= 0.20
        and robust_pass_precision >= 0.80
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
            "robust_pass_brier": pass_brier,
            "robust_pass_precision": robust_pass_precision,
            "robust_pass_recall": robust_pass_recall,
            "robust_pass_probability_threshold": robust_pass_threshold,
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
        "robust_pass_probability_threshold": robust_pass_threshold,
    }


def _high_precision_pass_threshold(probability: np.ndarray, truth: np.ndarray) -> float:
    """Choose a conservative threshold with at least 80% calibration precision."""

    values = np.asarray(probability, dtype=np.float64).reshape(-1)
    labels = np.asarray(truth, dtype=bool).reshape(-1)
    for threshold in np.linspace(0.50, 0.95, 46):
        selected = values >= threshold
        if int(np.sum(selected)) >= 3 and float(np.mean(labels[selected])) >= 0.80:
            return float(threshold)
    if np.any(labels):
        # Research/protection mode still needs a learnable feasibility signal.
        # The acceptance precision gate remains false, but the synthetic SAC
        # environment can use a conservative lower-quartile positive cutoff.
        return float(np.clip(np.quantile(values[labels], 0.25), 0.60, 0.90))
    return 0.80


def _surrogate_network_type(torch_module: Any) -> type:
    nn = torch_module.nn

    class SurrogateNetwork(nn.Module):
        def __init__(
            self,
            feature_count: int,
            metric_count: int,
            invalid_count: int,
            hidden_sizes: Sequence[int] | None = None,
        ):
            super().__init__()
            resolved_hidden_sizes = _normalize_hidden_sizes(hidden_sizes)
            layers: list[Any] = []
            previous_size = feature_count
            for hidden_size in resolved_hidden_sizes:
                layers.extend((nn.Linear(previous_size, hidden_size), nn.SiLU()))
                previous_size = hidden_size
            self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
            self.metric_head = nn.Linear(previous_size, metric_count)
            self.invalid_head = nn.Linear(previous_size, invalid_count)

        def forward(self, value: Any) -> tuple[Any, Any]:
            hidden = self.backbone(value)
            return self.metric_head(hidden), self.invalid_head(hidden)

    return SurrogateNetwork


def _normalize_hidden_sizes(hidden_sizes: Sequence[int] | None) -> tuple[int, ...]:
    if hidden_sizes is None:
        return DEFAULT_SURROGATE_HIDDEN_SIZES
    try:
        resolved = tuple(int(value) for value in hidden_sizes)
    except (TypeError, ValueError) as exc:
        raise ValueError("Surrogate hidden_sizes must be a sequence of positive integers.") from exc
    if any(value <= 0 for value in resolved):
        raise ValueError("Surrogate hidden_sizes must contain only positive integers; use an empty sequence for linear.")
    return resolved


def _training_fold_features(
    dataset: DrlDataset,
    train_indexes: np.ndarray | Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build feature rows without borrowing baseline values from held-out runs."""

    features = np.asarray(dataset.features, dtype=np.float32).copy()
    action_count = len(ACTION_FIELDS)
    metric_count = len(METRIC_FIELDS)
    baseline_start = action_count
    baseline_stop = baseline_start + metric_count
    mask_stop = baseline_stop + metric_count
    medians = np.zeros(metric_count, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] < mask_stop:
        return features, medians
    values = np.asarray(dataset.baseline_values, dtype=np.float32)
    masks = np.asarray(dataset.baseline_mask, dtype=np.float32)
    if values.shape != (dataset.size, metric_count) or masks.shape != values.shape:
        return features, medians
    training = np.asarray(train_indexes, dtype=int).reshape(-1)
    filled = values.copy()
    for column in range(metric_count):
        observed = training[masks[training, column] > 0]
        medians[column] = float(np.median(values[observed, column])) if observed.size else 0.0
        filled[masks[:, column] <= 0, column] = medians[column]
    features[:, baseline_start:baseline_stop] = filled
    features[:, baseline_stop:mask_stop] = masks
    return features, medians


def _manifest_hidden_sizes(manifest: dict[str, Any]) -> tuple[int, ...]:
    # Artifacts created before configurable architectures used this exact
    # topology and did not record it in their manifests.
    if "hidden_sizes" not in manifest or manifest.get("hidden_sizes") is None:
        return LEGACY_SURROGATE_HIDDEN_SIZES
    return _normalize_hidden_sizes(manifest["hidden_sizes"])


def _resolve_training_splits(
    dataset: DrlDataset,
    seed: int,
    train_indexes: np.ndarray | Sequence[int] | None,
    validation_indexes: np.ndarray | Sequence[int] | None,
    evaluation_indexes: np.ndarray | Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    explicit = train_indexes is not None or validation_indexes is not None or evaluation_indexes is not None
    if not explicit:
        training, validation = _group_candidate_split(dataset, seed)
        return training, validation, validation.copy()
    if train_indexes is None or validation_indexes is None:
        raise ValueError("Explicit surrogate splits require both train_indexes and validation_indexes.")

    training = _validated_indexes(train_indexes, dataset.size, "train_indexes")
    validation = _validated_indexes(validation_indexes, dataset.size, "validation_indexes")
    evaluation_is_explicit = evaluation_indexes is not None
    evaluation = (
        validation.copy()
        if not evaluation_is_explicit
        else _validated_indexes(evaluation_indexes, dataset.size, "evaluation_indexes")
    )
    if training.size < 20:
        raise RuntimeError(
            f"Explicit surrogate split requires at least 20 training samples; found {training.size}."
        )
    if np.intersect1d(training, validation).size or np.intersect1d(training, evaluation).size:
        raise ValueError("Surrogate train indexes must not overlap validation or evaluation indexes.")
    if evaluation_is_explicit and np.intersect1d(validation, evaluation).size:
        raise ValueError("Surrogate validation and evaluation indexes must not overlap when both are explicit.")

    training_groups = set(dataset.groups[training].tolist())
    validation_groups = set(dataset.groups[validation].tolist())
    evaluation_groups = set(dataset.groups[evaluation].tolist())
    held_out_groups = validation_groups | evaluation_groups
    overlapping_groups = training_groups & held_out_groups
    if overlapping_groups:
        raise ValueError(
            "Surrogate explicit splits must keep complete run groups isolated; "
            f"overlap: {sorted(str(value) for value in overlapping_groups)}"
        )
    if evaluation_is_explicit and validation_groups & evaluation_groups:
        raise ValueError(
            "Surrogate explicit validation and evaluation splits must use different complete run groups."
        )

    from .common import candidate_key

    training_candidates = {candidate_key(dataset.candidates[index]) for index in training}
    validation_candidates = {
        candidate_key(dataset.candidates[index])
        for index in validation
    }
    evaluation_candidates = {
        candidate_key(dataset.candidates[index])
        for index in evaluation
    }
    held_out_candidates = validation_candidates | evaluation_candidates
    if training_candidates & held_out_candidates:
        raise ValueError(
            "Surrogate explicit splits contain candidate keys in both training and held-out data; "
            "purge repeated candidates before training."
        )
    if evaluation_is_explicit and validation_candidates & evaluation_candidates:
        raise ValueError(
            "Surrogate validation and evaluation splits share candidate keys; purge repeated candidates."
        )
    return training, validation, evaluation


def _validated_indexes(indexes: np.ndarray | Sequence[int], size: int, name: str) -> np.ndarray:
    values = np.asarray(indexes, dtype=int).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} contains duplicate indexes.")
    if np.any(values < 0) or np.any(values >= size):
        raise IndexError(f"{name} contains an index outside [0, {size}).")
    return values


def _trainable_parameter_count(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def _indexes_hash(indexes: np.ndarray | Sequence[int]) -> str:
    values = np.asarray(indexes, dtype=np.int64).reshape(-1)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _split_hash(
    train_indexes: np.ndarray,
    validation_indexes: np.ndarray,
    evaluation_indexes: np.ndarray,
) -> str:
    payload = {
        "train": _indexes_hash(train_indexes),
        "validation": _indexes_hash(validation_indexes),
        "evaluation": _indexes_hash(evaluation_indexes),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _dataset_hash(dataset: DrlDataset) -> str:
    digest = hashlib.sha256()
    arrays = (
        dataset.features,
        dataset.actions,
        dataset.metrics,
        dataset.metric_mask,
        dataset.invalid_labels,
        dataset.scores,
        dataset.passed,
        dataset.baseline_values,
        dataset.baseline_mask,
    )
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    digest.update(json.dumps(dataset.groups.astype(str).tolist(), separators=(",", ":")).encode("utf-8"))

    from .common import candidate_key

    candidate_keys = [list(candidate_key(candidate)) for candidate in dataset.candidates]
    digest.update(json.dumps(candidate_keys, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _build_feasibility_calibrator(
    dataset: DrlDataset,
    indexes: np.ndarray | Sequence[int],
) -> dict[str, np.ndarray]:
    """Collapse exact raw keys into the compact local feasibility memory."""

    selected = np.asarray(indexes, dtype=int).reshape(-1)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index in selected:
        groups.setdefault(candidate_key(dataset.candidates[int(index)]), []).append(int(index))
    actions: list[np.ndarray] = []
    pass_probability: list[float] = []
    metrics: list[np.ndarray] = []
    support: list[float] = []
    invalid_probability: list[np.ndarray] = []
    for group_indexes in groups.values():
        first = group_indexes[0]
        actions.append(np.asarray(dataset.actions[first], dtype=np.float32))
        pass_probability.append(float(np.median(dataset.passed[group_indexes])))
        metrics.append(np.median(dataset.metrics[group_indexes], axis=0).astype(np.float32))
        invalid_probability.append(
            np.median(
                dataset.invalid_labels[group_indexes, :SAFETY_LABEL_COUNT], axis=0
            ).astype(np.float32)
        )
        support.append(float(len(group_indexes)))
    return {
        "actions": np.asarray(actions, dtype=np.float32).reshape(-1, len(ACTION_FIELDS)),
        "pass_probability": np.asarray(pass_probability, dtype=np.float32),
        "metrics": np.asarray(metrics, dtype=np.float32).reshape(-1, len(METRIC_FIELDS)),
        "invalid_probability": np.asarray(invalid_probability, dtype=np.float32).reshape(
            -1, SAFETY_LABEL_COUNT
        ),
        "support": np.asarray(support, dtype=np.float32),
    }


def _inference_latency_p95_ms(ensemble: SurrogateEnsemble, features: np.ndarray) -> float:
    values = np.asarray(features, dtype=np.float32)
    if values.size == 0:
        return 0.0
    probes = values[: min(len(values), 32)]
    ensemble.predict_features(probes)
    durations: list[float] = []
    for _ in range(10):
        started = time.perf_counter()
        ensemble.predict_features(probes)
        durations.append((time.perf_counter() - started) * 1000.0)
    return float(np.percentile(np.asarray(durations, dtype=np.float64), 95))


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


def _group_balanced_bootstrap(
    indexes: np.ndarray,
    groups: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap equal expected rows per complete run group."""

    selected = np.asarray(indexes, dtype=int)
    if selected.size == 0:
        return selected
    unique_groups = sorted(set(groups[selected].tolist()))
    if len(unique_groups) <= 1:
        return rng.choice(selected, size=len(selected), replace=True).astype(int)
    base_count, remainder = divmod(len(selected), len(unique_groups))
    extra_order = list(rng.permutation(len(unique_groups)))
    extra_groups = set(extra_order[:remainder])
    samples: list[np.ndarray] = []
    for group_index, group in enumerate(unique_groups):
        members = selected[groups[selected] == group]
        draw_count = base_count + (1 if group_index in extra_groups else 0)
        if draw_count > 0:
            samples.append(rng.choice(members, size=draw_count, replace=True).astype(int))
    result = np.concatenate(samples) if samples else selected.copy()
    rng.shuffle(result)
    return result.astype(int)


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
            index
            for index in range(SAFETY_LABEL_COUNT)
            if np.any(dataset.invalid_labels[:, index] > 0)
        }
        required_pass_classes = {
            value
            for value in (False, True)
            if np.any((dataset.passed >= 0.80) == value)
        }
        covered_metrics: set[int] = set()
        covered_invalid: set[int] = set()
        covered_pass_classes: set[bool] = set()
        validation_groups: list[str] = []
        while (
            covered_metrics != required_metrics
            or covered_invalid != required_invalid
            or covered_pass_classes != required_pass_classes
        ):
            best_group = None
            best_gain = 0
            best_size = float("inf")
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
                group_pass_classes = set(
                    ((dataset.passed[indexes] >= 0.80).astype(bool)).tolist()
                )
                pass_gain = len((required_pass_classes - covered_pass_classes) & group_pass_classes)
                gain = metric_gain + invalid_gain * 2 + pass_gain * 3
                # Several runs often cover the same metric/safety classes.
                # Holding out the first shuffled run could therefore put a
                # 500-point run in validation and leave fewer than 200 points
                # for training.  For equal coverage, prefer the smaller whole
                # run; the seed remains the final deterministic tie-break.
                if gain > best_gain or (gain == best_gain and gain > 0 and len(indexes) < best_size):
                    best_group = group
                    best_gain = gain
                    best_size = len(indexes)
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
            covered_pass_classes.update(
                ((dataset.passed[indexes] >= 0.80).astype(bool)).tolist()
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
        if (
            covered_metrics != required_metrics
            or covered_invalid != required_invalid
            or covered_pass_classes != required_pass_classes
        ):
            raise RuntimeError(
                "Run-grouped validation cannot cover every metric, safety class, and robust-pass class."
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


def _repeat_metric_mad(
    dataset: DrlDataset,
    indexes: np.ndarray | Sequence[int] | None = None,
) -> np.ndarray:
    groups: dict[tuple[Any, ...], list[int]] = {}
    from .common import candidate_key

    selected = np.asarray(indexes if indexes is not None else np.arange(dataset.size), dtype=int)
    for index in selected:
        candidate = dataset.candidates[int(index)]
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
        "bode_gain_shape_penalty": max(0.0, mapping["bode_gain_shape_penalty"]),
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
