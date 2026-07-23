"""Synthetic Safe SAC environment, policy training, and guarded online tuner."""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # The heuristic server remains usable without ML extras.
    gym = None
    spaces = None

from ..models import (
    HardwarePidCandidate,
    IterationRecord,
    TuningConfig,
    bandwidth_objective,
    effective_lc_inductance_from_raw,
    effective_lc_inductance_raw,
    output_inductance_from_raw,
    output_inductance_raw,
)
from ..search import _next_higher_bandwidth_candidate, select_best_result, select_diverse_results
from .common import (
    ACTION_FIELDS,
    KPOLE_PAIRS,
    METRIC_FIELDS,
    SCHEMA_VERSION,
    atomic_write_json,
    candidate_from_mapping,
    candidate_from_normalized,
    candidate_key,
    candidate_to_mapping,
    candidate_to_normalized,
    candidate_with_delta,
    metric_vector,
    read_json,
    relabeled_score,
    vector_to_metric_mapping,
)
from .dataset import DrlDataset
from .model import SurrogateEnsemble, require_ml_dependencies


OBSERVATION_SIZE = 3 * len(METRIC_FIELDS) + 2 * len(ACTION_FIELDS) + 2
GLOBAL_EXPLORATION_INTERVAL = 10
GLOBAL_EXPLORATION_INTERVAL_TWO_BASINS = 25
GLOBAL_EXPLORATION_INTERVAL_MANY_BASINS = 50
MEDIUM_EXPLORATION_INTERVAL = 3
KPOLE_DIVERSITY_INTERVAL = 8
TRUST_REGION_EXPAND_AFTER = 8
SOBOL_RESTART_AFTER = 12
ONLINE_UPDATE_BATCH = 20
ONLINE_GRADIENT_STEPS = 64
GLOBAL_REPLAY_WEIGHT = 0.25
SURROGATE_CALIBRATION_INTERVAL = 32
SURROGATE_CALIBRATION_WINDOW = 128
POLICY_VALIDATION_MIN_SAMPLES = 8
POLICY_VALIDATION_MAX_SAMPLES = 32
GOOD_BASIN_SCORE = 10.0
GOOD_BASIN_TRUST_FRACTION = 0.10
GOOD_BASIN_MEDIUM_TRUST_FRACTION = 0.40
HARDWARE_NEIGHBOR_COUNT = 5
SAFETY_LABEL_COUNT = 3
BOUNDARY_REPAIR_CONFIDENCE_Z = 1.2815515655446004
BOUNDARY_REPAIR_ATTEMPT_LIMIT = 12
FAST_SETTLING_TARGET_US = 1.0
FAST_SETTLING_PASS_LIMIT_US = 2.0
FAST_SETTLING_REWARD_SCALE = 2.0
RELIABILITY_RETEST_INTERVAL = 100
POLICY_PROPOSAL_COUNT = 24
LOCAL_SOBOL_PROPOSAL_COUNT = 12
LOCAL_DIRECTIONAL_PROPOSAL_COUNT = 12
BOUNDARY_REPAIR_SPECS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("mod0_kp", (2, 5)),
    ("mod0_ki", (2, 5)),
    ("mod0_kd", (2, 5)),
    ("mod0_kpole1", (1,)),
    ("mod0_kpole2", (1,)),
    ("mod0_cm_gain", (1,)),
    ("output_inductance_raw", (1, 2)),
    ("effective_lc_inductance_raw", (1, 2)),
)


def _robust_pass_threshold(ensemble: Any) -> float:
    value = getattr(ensemble, "robust_pass_probability_threshold", 0.80)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.80


def _fast_settling_quality(overshoot_us: float, undershoot_us: float) -> float:
    """Return 0 at the 2-us limit and 1 at/below the 1-us research target."""

    worst_us = max(float(overshoot_us), float(undershoot_us))
    span_us = max(1e-9, FAST_SETTLING_PASS_LIMIT_US - FAST_SETTLING_TARGET_US)
    return float(np.clip((FAST_SETTLING_PASS_LIMIT_US - worst_us) / span_us, 0.0, 1.0))


class SurrogateTuningEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    metadata = {"render_modes": []}

    def __init__(
        self,
        ensemble: SurrogateEnsemble,
        dataset: DrlDataset,
        config: TuningConfig,
        max_steps: int = 15,
        seed: int = 20260709,
    ):
        if gym is None or spaces is None:
            raise RuntimeError("Gymnasium is required for the Safe SAC synthetic environment.")
        super().__init__()
        self.ensemble = ensemble
        self.dataset = dataset
        self.config = config
        self.max_steps = max(1, int(max_steps))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(ACTION_FIELDS),), dtype=np.float32)
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._steps = 0
        self._baseline = np.zeros(len(METRIC_FIELDS), dtype=np.float32)
        self._baseline_mask = np.zeros(len(METRIC_FIELDS), dtype=np.float32)
        self._last_action = np.zeros(len(ACTION_FIELDS), dtype=np.float32)
        self._last_metrics = np.zeros(len(METRIC_FIELDS), dtype=np.float32)
        self._best_action = np.zeros(len(ACTION_FIELDS), dtype=np.float32)
        self._best_metrics = np.zeros(len(METRIC_FIELDS), dtype=np.float32)
        self._best_score = 250.0
        self._best_objective = 250.0
        self._best_passed = False
        self._best_pass_probability = 0.0
        self._uncertainty = 0.0
        self._repeat_noise = _repeat_measurement_noise(dataset)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        legal_actions = np.all(np.abs(self.dataset.actions) <= 1.0 + 1e-6, axis=1)
        confirmed_starts, local_recovery_starts, global_starts = _episode_start_pools(
            self.dataset,
            threshold=_robust_pass_threshold(self.ensemble),
        )
        valid = np.concatenate([confirmed_starts, local_recovery_starts, global_starts])
        if valid.size == 0:
            valid = np.where(
                (np.max(self.dataset.invalid_labels[:, :SAFETY_LABEL_COUNT], axis=1) <= 0)
                & legal_actions
            )[0]
        if valid.size == 0:
            raise RuntimeError("The DRL dataset has no valid episode starting points.")
        validation_start = (options or {}).get("validation_start")
        if validation_start is None:
            draw = self._rng.random()
            pool = (
                confirmed_starts
                if draw < 0.30
                else local_recovery_starts
                if draw < 0.85
                else global_starts
            )
            if pool.size == 0:
                pool = valid
            index = int(self._rng.choice(pool if pool.size else valid))
        else:
            index = _resolve_validation_start_index(self.dataset, valid, validation_start, self.config)
        self._steps = 0
        baseline_start = len(ACTION_FIELDS)
        baseline_stop = baseline_start + len(METRIC_FIELDS)
        self._baseline = self.dataset.features[index, baseline_start:baseline_stop].astype(np.float32)
        self._baseline_mask = self.dataset.features[
            index, baseline_stop:baseline_stop + len(METRIC_FIELDS)
        ].astype(np.float32)
        self._last_action = self.dataset.actions[index].astype(np.float32)
        self._last_metrics = self.dataset.metrics[index].astype(np.float32)
        self._best_action = self._last_action.copy()
        self._best_metrics = self._last_metrics.copy()
        self._best_score = float(self.dataset.scores[index])
        start_candidate = self.dataset.candidates[index]
        self._best_pass_probability = float(self.dataset.passed[index])
        self._best_passed = bool(
            self._best_pass_probability >= _robust_pass_threshold(self.ensemble)
        )
        self._best_objective = bandwidth_objective(
            self._best_score,
            start_candidate,
            passed=self._best_passed,
        )[0]
        self._uncertainty = 0.0
        return self._observation(), {
            "score": self._best_score,
            "objective_score": self._best_objective,
            "bandwidth": start_candidate.mod0_ll_bw,
            "start_index": index,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._steps += 1
        base = candidate_from_normalized(self._best_action, self.config.search, "drl_sim")
        started_from_confirmed = bool(self._best_passed)
        if self._best_passed:
            # Once feasibility is known, the next decision is one-dimensional:
            # preserve the confirmed raw key and probe exactly one BW code up.
            # The hardware tuner applies the same hierarchy after confirmation.
            candidate = replace(
                base,
                mod0_ll_bw=min(
                    int(self.config.search.mod0_ll_bw.max),
                    int(base.mod0_ll_bw) + 1,
                ),
                phase="drl_sim_bandwidth_climb",
            )
        else:
            candidate = candidate_with_delta(base, action, self.config.search, "drl_sim", trust_fraction=0.10)
        prediction: dict[str, np.ndarray]
        shield_rejected = True
        trust_fractions = (0.10,) if started_from_confirmed else (0.10, 0.05, 0.025)
        for trust_fraction in trust_fractions:
            if not started_from_confirmed:
                candidate = candidate_with_delta(
                    base,
                    action,
                    self.config.search,
                    "drl_sim",
                    trust_fraction=trust_fraction,
                )
            normalized = candidate_to_normalized(candidate, self.config.search).astype(np.float32)
            feature = np.concatenate(
                [normalized, self._baseline, self._baseline_mask]
            ).reshape(1, -1)
            prediction = self.ensemble.predict_features(feature)
            safety_probability = float(prediction["safety_probability"][0])
            validity_probability = float(
                prediction.get("validity_probability", prediction["safety_probability"])[0]
            )
            invalid_probability = np.asarray(
                prediction.get("invalid_probability", np.zeros((1, 3), dtype=np.float64))[0],
                dtype=np.float64,
            )
            self._uncertainty = float(prediction["uncertainty"][0])
            shield_rejected = bool(
                safety_probability < 0.995
                or float(np.max(invalid_probability[:SAFETY_LABEL_COUNT]))
                >= self.ensemble.invalid_probability_threshold
                or self._uncertainty > self.ensemble.uncertainty_threshold
            )
            if not shield_rejected:
                break
        if shield_rejected:
            if started_from_confirmed:
                return self._observation(), 1.0, True, False, {
                    "passed": True,
                    "probe_passed": False,
                    "bandwidth_climb_stopped": "shield_rejected",
                    "candidate": candidate_to_mapping(base),
                    "unsafe": False,
                    "protection": False,
                    "invalid": False,
                    "shield_rejected": True,
                    "safety_probability": safety_probability,
                    "validity_probability": validity_probability,
                    "uncertainty": self._uncertainty,
                    "score": self._best_score,
                    "best_score": self._best_score,
                    "objective_score": self._best_objective,
                    "best_objective_score": self._best_objective,
                }
            return self._observation(), -5.0, True, False, {
                "unsafe": False,
                "protection": False,
                "invalid": False,
                "shield_rejected": True,
                "safety_probability": safety_probability,
                "validity_probability": validity_probability,
                "uncertainty": self._uncertainty,
                "score": self._best_score,
                "best_score": self._best_score,
            }
        protection_probability = float(np.clip(invalid_probability[0] if invalid_probability.size else 0.0, 0.0, 1.0))
        other_invalid_probability = float(
            np.clip(
                np.max(invalid_probability[1:SAFETY_LABEL_COUNT])
                if invalid_probability.size > 1
                else 0.0,
                0.0,
                1.0,
            )
        )
        protection_event = bool(self._rng.random() < protection_probability)
        invalid_event = bool(protection_event or self._rng.random() < other_invalid_probability)
        if invalid_event:
            if started_from_confirmed:
                return self._observation(), 0.5, True, False, {
                    "passed": True,
                    "probe_passed": False,
                    "bandwidth_climb_stopped": "predicted_invalid",
                    "candidate": candidate_to_mapping(base),
                    "unsafe": protection_event,
                    "protection": protection_event,
                    "invalid": True,
                    "shield_rejected": False,
                    "safety_probability": safety_probability,
                    "validity_probability": validity_probability,
                    "uncertainty": self._uncertainty,
                    "score": self._best_score,
                    "best_score": self._best_score,
                    "objective_score": self._best_objective,
                    "best_objective_score": self._best_objective,
                }
            return self._observation(), -5.0, True, False, {
                "unsafe": protection_event,
                "protection": protection_event,
                "invalid": True,
                "shield_rejected": False,
                "safety_probability": safety_probability,
                "validity_probability": validity_probability,
                "uncertainty": self._uncertainty,
                "score": self._best_score,
                "best_score": self._best_score,
            }

        mean = np.asarray(prediction["metric_mean"][0], dtype=np.float64)
        epistemic_std = np.asarray(prediction["metric_std"][0], dtype=np.float64)
        std = np.maximum(np.sqrt(epistemic_std ** 2 + self._repeat_noise ** 2), 1e-6)
        measured = self._rng.normal(mean, std)
        measured[:4] = np.maximum(measured[:4], 0.0)
        measured[5] = max(measured[5], 1e-3)
        payload = _metric_payload(measured)
        score, metric_passed = relabeled_score(payload, self.config.targets)
        pass_probability = float(np.clip(prediction.get("pass_probability", [0.0])[0], 0.0, 1.0))
        passed = bool(
            metric_passed
            and pass_probability >= _robust_pass_threshold(self.ensemble)
        )
        objective_score, objective_bonus = bandwidth_objective(score, candidate, passed=passed)
        previous_best = self._best_objective
        previous_best_pass_probability = self._best_pass_probability
        self._last_action = normalized
        self._last_metrics = measured.astype(np.float32)
        improves = bool(
            (passed and not self._best_passed)
            or (
                passed == self._best_passed
                and (
                    objective_score < self._best_objective - 0.10
                    or (
                        abs(objective_score - self._best_objective) <= 0.10
                        and pass_probability > self._best_pass_probability
                    )
                )
            )
        )
        if improves:
            self._best_objective = objective_score
            self._best_score = score
            self._best_passed = passed
            self._best_pass_probability = pass_probability
            self._best_action = normalized.copy()
            self._best_metrics = measured.astype(np.float32)
        bandwidth_fraction = float(
            np.clip(
                (candidate.mod0_ll_bw - self.config.search.mod0_ll_bw.min)
                / max(1.0, self.config.search.mod0_ll_bw.max - self.config.search.mod0_ll_bw.min),
                0.0,
                1.0,
            )
        )
        reward = (
            (previous_best - self._best_objective) / 20.0
            + 0.75 * (pass_probability - previous_best_pass_probability)
            - 0.02
        )
        if started_from_confirmed and not passed:
            retained_bandwidth_fraction = float(
                np.clip(
                    (base.mod0_ll_bw - self.config.search.mod0_ll_bw.min)
                    / max(1.0, self.config.search.mod0_ll_bw.max - self.config.search.mod0_ll_bw.min),
                    0.0,
                    1.0,
                )
            )
            return self._observation(), 1.0 + retained_bandwidth_fraction, True, False, {
                "score": self._best_score,
                "best_score": self._best_score,
                "objective_score": self._best_objective,
                "best_objective_score": self._best_objective,
                "bandwidth_bonus": bandwidth_objective(
                    self._best_score, base, passed=True
                )[1],
                "passed": True,
                "probe_passed": False,
                "metric_passed": metric_passed,
                "pass_probability": pass_probability,
                "bandwidth_climb_stopped": "probe_failed",
                "unsafe": False,
                "protection": False,
                "invalid": False,
                "shield_rejected": False,
                "candidate": candidate_to_mapping(base),
            }
        terminated = bool(passed)
        if terminated:
            fast_quality = _fast_settling_quality(measured[2], measured[3])
            reward += 2.0 + 2.0 * bandwidth_fraction + FAST_SETTLING_REWARD_SCALE * fast_quality
        truncated = self._steps >= self.max_steps and not terminated
        return self._observation(), float(reward), terminated, truncated, {
            "score": score,
            "best_score": self._best_score,
            "objective_score": objective_score,
            "best_objective_score": self._best_objective,
            "bandwidth_bonus": objective_bonus,
            "passed": passed,
            "metric_passed": metric_passed,
            "pass_probability": pass_probability,
            "fast_settling_quality": _fast_settling_quality(measured[2], measured[3]),
            "unsafe": False,
            "protection": False,
            "invalid": False,
            "shield_rejected": False,
            "candidate": candidate_to_mapping(candidate),
        }

    def _observation(self) -> np.ndarray:
        metric_mean = self.ensemble.metric_mean
        metric_std = self.ensemble.metric_std
        last_metrics = (self._last_metrics - metric_mean) / metric_std
        best_metrics = (self._best_metrics - metric_mean) / metric_std
        progress = np.asarray([self._steps / self.max_steps], dtype=np.float32)
        uncertainty = np.asarray([self._uncertainty], dtype=np.float32)
        value = np.concatenate(
            [
                (self._baseline - metric_mean) / metric_std,
                self._last_action,
                last_metrics,
                self._best_action,
                best_metrics,
                progress,
                uncertainty,
            ]
        )
        return np.clip(value, -20.0, 20.0).astype(np.float32)


def _normalize_policy_arch(
    architecture: tuple[int, ...] | list[int] | None,
    name: str,
) -> list[int]:
    """Validate an SB3 MLP architecture while preserving an empty linear net."""

    if architecture is None:
        return []
    result: list[int] = []
    for value in architecture:
        width = int(value)
        if width <= 0:
            raise ValueError(f"{name} widths must be positive integers; received {value!r}.")
        result.append(width)
    return result


def _valid_validation_start_indices(dataset: DrlDataset) -> np.ndarray:
    legal_actions = np.all(np.abs(dataset.actions) <= 1.0 + 1e-6, axis=1)
    valid = np.where(
        (np.max(dataset.invalid_labels[:, :SAFETY_LABEL_COUNT], axis=1) <= 0)
        & legal_actions
    )[0]
    if valid.size == 0:
        valid = np.where(
            (np.max(dataset.invalid_labels[:, :SAFETY_LABEL_COUNT], axis=1) <= 0)
            & legal_actions
        )[0]
    if valid.size == 0:
        raise RuntimeError("The DRL dataset has no valid policy-validation starting points.")
    return valid.astype(np.int64)


def _episode_start_pools(
    dataset: DrlDataset,
    *,
    threshold: float = 0.80,
    local_radius: float = 0.18,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split safe starts into confirmed, local recovery, and guarded global."""

    valid = _valid_validation_start_indices(dataset)
    confirmed = valid[dataset.passed[valid] >= float(threshold)]
    failed = valid[dataset.passed[valid] < float(threshold)]
    if confirmed.size == 0 or failed.size == 0:
        return confirmed, failed, np.zeros(0, dtype=np.int64)
    distances = np.sqrt(
        np.mean(
            (
                dataset.actions[failed, None, :]
                - dataset.actions[confirmed, :][None, :, :]
            )
            ** 2,
            axis=2,
        )
    )
    nearest = np.min(distances, axis=1)
    # Exclude high-penalty points from the local task even if one normalized
    # coordinate happens to be close; they belong to guarded global recovery.
    local_mask = (nearest <= float(local_radius)) & (dataset.scores[failed] <= 50.0)
    return (
        confirmed.astype(np.int64),
        failed[local_mask].astype(np.int64),
        failed[~local_mask].astype(np.int64),
    )


def _warm_start_actor_from_confirmed_basins(
    policy: Any,
    environment: SurrogateTuningEnv,
    dataset: DrlDataset,
    *,
    epochs: int = 200,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    """Behavior-clone local recovery directions measured by the hardware.

    Sparse 9-D pass labels make random SAC exploration collapse against the
    safety shield before it receives useful rewards.  For each local near-pass
    row, this warm start points the actor toward the nearest confirmed raw-key;
    subsequent SAC updates still decide whether that direction improves the
    objective and continue to train on guarded global failures.
    """

    import torch

    confirmed, local, _ = _episode_start_pools(dataset)
    if confirmed.size == 0 or local.size == 0 or int(epochs) <= 0:
        return {"enabled": False, "sample_count": 0, "epochs": 0, "final_loss": None}
    # Candidate-key duplicates carry the same action target. Keep one row per
    # session baseline and action pair without letting deliberate repeats
    # dominate the supervised warm start.
    selected: list[int] = []
    seen: set[tuple[Any, ...]] = set()
    for index in local:
        baseline_start = len(ACTION_FIELDS)
        baseline_stop = baseline_start + len(METRIC_FIELDS) * 2
        signature = (
            candidate_key(dataset.candidates[int(index)]),
            tuple(np.round(dataset.features[int(index), baseline_start:baseline_stop], 5)),
        )
        if signature not in seen:
            seen.add(signature)
            selected.append(int(index))
    confirmed_actions = np.asarray(dataset.actions[confirmed], dtype=np.float32)
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index in selected:
        observation, _ = environment.reset(options={"validation_start": {"index": index}})
        start = np.asarray(dataset.actions[index], dtype=np.float32)
        distances = np.sqrt(np.mean((confirmed_actions - start[None, :]) ** 2, axis=1))
        target = confirmed_actions[int(np.argmin(distances))]
        # candidate_with_delta uses normalized_delta * (2 * trust_fraction).
        action = np.clip((target - start) / 0.20, -1.0, 1.0)
        observations.append(observation.astype(np.float32))
        targets.append(action.astype(np.float32))
    observation_tensor = torch.as_tensor(
        np.asarray(observations), dtype=torch.float32, device=policy.device
    )
    target_tensor = torch.as_tensor(
        np.asarray(targets), dtype=torch.float32, device=policy.device
    )
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=float(learning_rate))
    rng = np.random.default_rng(20260720)
    final_loss = 0.0
    policy.actor.train()
    for _ in range(max(1, int(epochs))):
        order = rng.permutation(len(observations))
        for start in range(0, len(order), 64):
            indexes = torch.as_tensor(order[start:start + 64], dtype=torch.long, device=policy.device)
            predicted = policy.actor(observation_tensor[indexes], deterministic=True)
            loss = torch.nn.functional.smooth_l1_loss(predicted, target_tensor[indexes])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), 5.0)
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    policy.actor.eval()
    return {
        "enabled": True,
        "sample_count": len(observations),
        "epochs": max(1, int(epochs)),
        "final_loss": final_loss,
        "target": "normalized delta to nearest confirmed raw-key",
    }


def _serialize_validation_start(value: Any) -> dict[str, Any]:
    if isinstance(value, (int, np.integer)):
        return {"index": int(value)}
    if isinstance(value, HardwarePidCandidate):
        return {"candidate": candidate_to_mapping(value)}
    if isinstance(value, dict):
        if "index" in value:
            return {"index": int(value["index"])}
        if "action" in value:
            action = np.asarray(value["action"], dtype=np.float64).reshape(-1)
            if action.size != len(ACTION_FIELDS):
                raise ValueError(f"A policy validation action must contain {len(ACTION_FIELDS)} values.")
            return {"action": action.tolist()}
        candidate_payload = value.get("candidate", value)
        if isinstance(candidate_payload, HardwarePidCandidate):
            return {"candidate": candidate_to_mapping(candidate_payload)}
        if isinstance(candidate_payload, dict) and any(
            str(key).startswith("mod0_") for key in candidate_payload
        ):
            return {"candidate": dict(candidate_payload)}
    action = np.asarray(value, dtype=np.float64).reshape(-1)
    if action.size == len(ACTION_FIELDS):
        return {"action": action.tolist()}
    raise ValueError(
        f"A policy validation start must be a dataset index, PID candidate, candidate mapping, or {len(ACTION_FIELDS)}-value action."
    )


def _resolve_validation_start_index(
    dataset: DrlDataset,
    valid_indices: np.ndarray,
    value: Any,
    config: TuningConfig,
) -> int:
    serialized = _serialize_validation_start(value)
    valid_set = {int(index) for index in valid_indices}
    if "index" in serialized:
        index = int(serialized["index"])
        if index not in valid_set:
            raise ValueError(f"Policy validation start index {index} is not a valid safe dataset row.")
        return index
    if "candidate" in serialized:
        from .common import candidate_from_mapping

        candidate = candidate_from_mapping(serialized["candidate"], phase="drl_validation")
        target_action = candidate_to_normalized(candidate, config.search)
    else:
        target_action = np.asarray(serialized["action"], dtype=np.float64)
    valid_actions = np.asarray(dataset.actions[valid_indices], dtype=np.float64)
    distances = np.linalg.norm(valid_actions - target_action.reshape(1, -1), axis=1)
    return int(valid_indices[int(np.argmin(distances))])


def _build_policy_validation_pack(
    dataset: DrlDataset,
    config: TuningConfig,
    *,
    count: int,
    seed: int,
    validation_seeds: list[int] | tuple[int, ...] | None,
    validation_starts: list[Any] | tuple[Any, ...] | None,
) -> dict[str, Any]:
    episode_count = max(1, int(count))
    seeds = (
        [int(value) for value in validation_seeds]
        if validation_seeds
        else [int(seed + index) for index in range(episode_count)]
    )
    if validation_starts:
        starts = [_serialize_validation_start(value) for value in validation_starts]
    else:
        valid = _valid_validation_start_indices(dataset)
        confirmed, recovery, guarded_global = _episode_start_pools(dataset)
        starts = []
        for index, value in enumerate(seeds):
            rng = np.random.default_rng(value)
            # Fixed validation mirrors the training mixture: 30% confirmed BW
            # climb, 50% local recovery, and 20% guarded global recovery.
            bucket = index % 10
            pool = confirmed if bucket < 3 else recovery if bucket < 8 else guarded_global
            if pool.size == 0:
                pool = valid
            starts.append({"index": int(rng.choice(pool))})
    # Validate injected values before spending any time training.
    valid = _valid_validation_start_indices(dataset)
    for start in starts:
        _resolve_validation_start_index(dataset, valid, start, config)
    return {
        "version": 1,
        "seeds": seeds,
        "starts": starts,
        "dataset_size": int(dataset.size),
    }


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def train_safe_sac_policy(
    ensemble: SurrogateEnsemble,
    dataset: DrlDataset,
    config: TuningConfig,
    total_steps: int = 1_000_000,
    evaluation_episodes: int = 10_000,
    max_episode_steps: int = 15,
    seed: int = 20260709,
    progress: Callable[[float, str], None] | None = None,
    allow_unaccepted_surrogate: bool = False,
    policy_net_arch: tuple[int, ...] | list[int] | None = (64, 64),
    batch_size: int = 256,
    train_frequency: int = 1,
    actor_net_arch: tuple[int, ...] | list[int] | None = None,
    critic_net_arch: tuple[int, ...] | list[int] | None = None,
    validation_seeds: list[int] | tuple[int, ...] | None = None,
    validation_starts: list[Any] | tuple[Any, ...] | None = None,
    checkpoint_interval: int = 25_000,
    checkpoint_evaluation_episodes: int | None = None,
    actor_warm_start_epochs: int = 200,
) -> dict[str, Any]:
    require_ml_dependencies()
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    if not ensemble.accepted and not allow_unaccepted_surrogate:
        raise RuntimeError("The surrogate acceptance gates failed; Safe SAC training is blocked.")
    legacy_arch = _normalize_policy_arch(policy_net_arch, "policy_net_arch")
    actor_arch = (
        _normalize_policy_arch(actor_net_arch, "actor_net_arch")
        if actor_net_arch is not None
        else list(legacy_arch)
    )
    critic_arch = (
        _normalize_policy_arch(critic_net_arch, "critic_net_arch")
        if critic_net_arch is not None
        else list(legacy_arch)
    )
    requested_steps = int(total_steps)
    learning_steps = max(100, requested_steps)
    checkpoint_every = max(0, int(checkpoint_interval))
    checkpoint_episodes = max(
        1,
        int(
            checkpoint_evaluation_episodes
            if checkpoint_evaluation_episodes is not None
            else min(max(1, int(evaluation_episodes)), 2_000)
        ),
    )
    validation_count = max(max(1, int(evaluation_episodes)), checkpoint_episodes)
    validation_pack = _build_policy_validation_pack(
        dataset,
        config,
        count=validation_count,
        seed=seed + 1,
        validation_seeds=validation_seeds,
        validation_starts=validation_starts,
    )
    ensemble.artifact_dir.mkdir(parents=True, exist_ok=True)
    validation_filename = "policy_validation.json"
    atomic_write_json(ensemble.artifact_dir / validation_filename, validation_pack)
    environment = SurrogateTuningEnv(ensemble, dataset, config, max_steps=max_episode_steps, seed=seed)
    checkpoint_dir = ensemble.artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "safe_sac_best.zip"
    checkpoint_history: list[dict[str, Any]] = []
    best_rank: tuple[float, ...] | None = None
    best_step: int | None = None
    training_started_at = time.perf_counter()
    training_started_cpu = time.process_time()

    def evaluate_checkpoint(model: Any, step: int) -> None:
        nonlocal best_rank, best_step
        evaluation = evaluate_safe_sac_policy(
            model,
            ensemble,
            dataset,
            config,
            episodes=checkpoint_episodes,
            max_episode_steps=max_episode_steps,
            seed=seed + 1,
            validation_seeds=validation_pack["seeds"],
            validation_starts=validation_pack["starts"],
        )
        rank = _policy_evaluation_rank(evaluation)
        is_best = best_rank is None or rank < best_rank
        if is_best:
            _atomic_save_policy(model, best_checkpoint_path)
            best_rank = rank
            best_step = int(step)
        checkpoint_history.append(
            {
                "step": int(step),
                "is_best": bool(is_best),
                "rank": [float(value) for value in rank],
                "evaluation": evaluation,
            }
        )

    class ProgressCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self._next_checkpoint = checkpoint_every if checkpoint_every > 0 else None
            self._last_evaluated_step = -1

        def _on_step(self) -> bool:
            if progress and self.num_timesteps % max(1000, learning_steps // 100) == 0:
                progress(
                    min(0.90, self.num_timesteps / max(1, learning_steps) * 0.90),
                    "Training Safe SAC policy",
                )
            if self._next_checkpoint is not None and self.num_timesteps >= self._next_checkpoint:
                evaluate_checkpoint(self.model, self.num_timesteps)
                self._last_evaluated_step = int(self.num_timesteps)
                while self._next_checkpoint <= self.num_timesteps:
                    self._next_checkpoint += checkpoint_every
            return True

        def _on_training_end(self) -> None:
            if self._last_evaluated_step != int(self.num_timesteps):
                evaluate_checkpoint(self.model, self.num_timesteps)
                self._last_evaluated_step = int(self.num_timesteps)

    device = "cpu"
    policy = SAC(
        "MlpPolicy",
        environment,
        learning_rate=3e-4,
        buffer_size=min(max(100_000, learning_steps), 1_000_000),
        learning_starts=min(10_000, max(100, learning_steps // 20)),
        batch_size=max(16, int(batch_size)),
        gamma=0.98,
        tau=0.005,
        train_freq=max(1, int(train_frequency)),
        gradient_steps=1,
        policy_kwargs={"net_arch": {"pi": actor_arch, "qf": critic_arch}},
        verbose=0,
        seed=seed,
        device=device,
    )
    warm_start = _warm_start_actor_from_confirmed_basins(
        policy,
        environment,
        dataset,
        epochs=max(0, int(actor_warm_start_epochs)),
    )
    # The fixed pack decides whether later SAC updates improve on behavior
    # cloning; a collapsed final actor cannot overwrite a better warm start.
    if warm_start.get("enabled"):
        evaluate_checkpoint(policy, 0)
    callback = ProgressCallback()
    policy.learn(total_timesteps=learning_steps, callback=callback)
    completed_training_steps = int(policy.num_timesteps)
    if not best_checkpoint_path.is_file():
        evaluate_checkpoint(policy, int(policy.num_timesteps))
    policy = SAC.load(best_checkpoint_path, env=environment, device=device)
    policy_path = ensemble.artifact_dir / "safe_sac_policy"
    _atomic_save_policy(policy, policy_path.with_suffix(".zip"))
    evaluation = evaluate_safe_sac_policy(
        policy,
        ensemble,
        dataset,
        config,
        episodes=evaluation_episodes,
        max_episode_steps=max_episode_steps,
        seed=seed + 1,
        progress=progress,
        validation_seeds=validation_pack["seeds"],
        validation_starts=validation_pack["starts"],
    )
    policy_accepted = evaluation["success_rate"] >= 0.90 and evaluation["protection_rate"] < 0.005
    parameter_counts = _sac_parameter_counts(policy)
    inference_latency_p95_ms = _policy_inference_latency_p95_ms(policy)
    manifest = dict(ensemble.manifest)
    manifest.update(
        {
            "policy_file": "safe_sac_policy.zip",
            "drl_schema_version": SCHEMA_VERSION,
            "action_fields": list(ACTION_FIELDS),
            "action_count": len(ACTION_FIELDS),
            "policy_training_steps": completed_training_steps,
            "actor_warm_start": warm_start,
            "policy_requested_training_steps": requested_steps,
            # Keep the legacy field list-shaped for older readers. The two
            # explicit fields below are authoritative when the nets differ.
            "policy_net_arch": list(actor_arch),
            "policy_actor_net_arch": list(actor_arch),
            "policy_critic_net_arch": list(critic_arch),
            "policy_parameter_counts": parameter_counts,
            "policy_seed": int(seed),
            "policy_batch_size": max(16, int(batch_size)),
            "policy_train_frequency": max(1, int(train_frequency)),
            "policy_checkpoint_interval": checkpoint_every,
            "policy_checkpoint_evaluation_episodes": checkpoint_episodes,
            "policy_best_checkpoint": "checkpoints/safe_sac_best.zip",
            "policy_best_checkpoint_step": best_step,
            "policy_checkpoint_history": checkpoint_history,
            "policy_validation_file": validation_filename,
            "policy_validation_pack_sha256": _json_sha256(validation_pack),
            "policy_evaluation": evaluation,
            "policy_accepted": policy_accepted,
            "ready": bool(ensemble.accepted and policy_accepted),
            "hardware_protection_policy": bool(allow_unaccepted_surrogate),
            "policy_training_cpu_seconds": float(time.process_time() - training_started_cpu),
            "policy_training_wall_seconds": float(time.perf_counter() - training_started_at),
            "policy_inference_latency_p95_ms": inference_latency_p95_ms,
            "policy_created_at": time.time(),
        }
    )
    artifact_files = [
        *[str(item) for item in manifest.get("member_files", [])],
        "scalers.npz",
        "safe_sac_policy.zip",
        "checkpoints/safe_sac_best.zip",
        validation_filename,
    ]
    if (ensemble.artifact_dir / "validation_starts.json").is_file():
        artifact_files.append("validation_starts.json")
    manifest["files_sha256"] = {
        filename: hashlib.sha256((ensemble.artifact_dir / filename).read_bytes()).hexdigest()
        for filename in artifact_files
        if (ensemble.artifact_dir / filename).is_file()
    }
    atomic_write_json(ensemble.artifact_dir / "manifest.json", manifest)
    ensemble.manifest = manifest
    if progress:
        progress(1.0, "Safe SAC training complete")
    return manifest


def evaluate_safe_sac_policy(
    policy: Any,
    ensemble: SurrogateEnsemble,
    dataset: DrlDataset,
    config: TuningConfig,
    episodes: int,
    max_episode_steps: int,
    seed: int,
    progress: Callable[[float, str], None] | None = None,
    validation_seeds: list[int] | tuple[int, ...] | None = None,
    validation_starts: list[Any] | tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    environment = SurrogateTuningEnv(ensemble, dataset, config, max_steps=max_episode_steps, seed=seed)
    episode_count = max(1, int(episodes))
    seed_values = [int(value) for value in validation_seeds] if validation_seeds else []
    start_values = list(validation_starts) if validation_starts else []
    successes = 0
    protections = 0
    invalid_events = 0
    shield_rejections = 0
    steps_to_success: list[int] = []
    initial_penalties: list[float] = []
    best_penalties: list[float] = []
    final_penalties: list[float] = []
    best_objectives: list[float] = []
    final_objectives: list[float] = []
    successful_bandwidths: list[float] = []
    for episode in range(episode_count):
        episode_seed = seed_values[episode % len(seed_values)] if seed_values else seed + episode
        options = (
            {"validation_start": start_values[episode % len(start_values)]}
            if start_values
            else None
        )
        observation, reset_info = environment.reset(seed=episode_seed, options=options)
        initial_penalty = float(reset_info["score"])
        episode_best = initial_penalty
        episode_final = initial_penalty
        episode_best_objective = float(reset_info.get("objective_score", initial_penalty))
        episode_final_objective = episode_best_objective
        for step in range(max_episode_steps):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            episode_final = float(info.get("score", episode_final))
            episode_best = min(episode_best, float(info.get("best_score", episode_best)))
            episode_final_objective = float(info.get("objective_score", episode_final_objective))
            episode_best_objective = min(
                episode_best_objective,
                float(info.get("best_objective_score", episode_best_objective)),
            )
            if info.get("protection"):
                protections += 1
            if info.get("invalid"):
                invalid_events += 1
            if info.get("shield_rejected"):
                shield_rejections += 1
            if terminated and info.get("passed"):
                successes += 1
                steps_to_success.append(step + 1)
                candidate_payload = info.get("candidate") if isinstance(info.get("candidate"), dict) else {}
                bandwidth = candidate_payload.get("mod0_ll_bw")
                if isinstance(bandwidth, (int, float)):
                    successful_bandwidths.append(float(bandwidth))
            if terminated or truncated:
                break
        initial_penalties.append(initial_penalty)
        best_penalties.append(episode_best)
        final_penalties.append(episode_final)
        best_objectives.append(episode_best_objective)
        final_objectives.append(episode_final_objective)
        if progress and episode % max(10, episodes // 100) == 0:
            progress(0.90 + episode / max(1, episodes) * 0.10, "Evaluating Safe SAC policy")
    success_rate = float(successes / episode_count)
    protection_rate = float(protections / episode_count)
    success_ci = _wilson_interval(successes, episode_count)
    protection_ci = _wilson_interval(protections, episode_count)
    return {
        "episodes": episode_count,
        "success_rate": success_rate,
        "success_rate_ci95": success_ci,
        "protection_rate": protection_rate,
        "protection_rate_ci95": protection_ci,
        "unsafe_rate": protection_rate,
        "invalid_rate": float(invalid_events / episode_count),
        "shield_rejection_rate": float(shield_rejections / episode_count),
        "median_steps_to_success": float(np.median(steps_to_success)) if steps_to_success else None,
        "p90_steps_to_success": float(np.percentile(steps_to_success, 90)) if steps_to_success else None,
        "mean_initial_penalty": float(np.mean(initial_penalties)),
        "mean_best_penalty": float(np.mean(best_penalties)),
        "mean_final_penalty": float(np.mean(final_penalties)),
        "median_best_penalty": float(np.median(best_penalties)),
        "median_final_penalty": float(np.median(final_penalties)),
        "mean_best_penalty_ci95": _mean_interval(best_penalties),
        "mean_final_penalty_ci95": _mean_interval(final_penalties),
        "mean_best_objective": float(np.mean(best_objectives)),
        "mean_final_objective": float(np.mean(final_objectives)),
        "mean_successful_bandwidth": (
            float(np.mean(successful_bandwidths)) if successful_bandwidths else None
        ),
        "validation_seed_count": len(seed_values),
        "validation_start_count": len(start_values),
    }


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 1.0]
    proportion = float(successes) / float(total)
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * np.sqrt(
        (proportion * (1.0 - proportion) / total) + (z * z / (4.0 * total * total))
    ) / denominator
    return [float(max(0.0, center - margin)), float(min(1.0, center + margin))]


def _mean_interval(values: list[float], z: float = 1.959963984540054) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return [float("nan"), float("nan")]
    mean = float(np.mean(array))
    if array.size == 1:
        return [mean, mean]
    margin = float(z * np.std(array, ddof=1) / np.sqrt(array.size))
    return [mean - margin, mean + margin]


def _policy_evaluation_rank(evaluation: dict[str, Any]) -> tuple[float, ...]:
    p90_steps = evaluation.get("p90_steps_to_success")
    return (
        -float(evaluation.get("success_rate", 0.0)),
        float(evaluation.get("protection_rate", 1.0)),
        float(evaluation.get("invalid_rate", 1.0)),
        -float(evaluation.get("mean_successful_bandwidth") or 0.0),
        float(evaluation.get("mean_best_objective", 300.0)),
        float(evaluation.get("mean_final_objective", 300.0)),
        float(evaluation.get("mean_best_penalty", 300.0)),
        float(evaluation.get("mean_final_penalty", 300.0)),
        float(p90_steps) if p90_steps is not None else 1.0e9,
    )


def _atomic_save_policy(policy: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.zip")
    policy.save(temporary)
    temporary.replace(destination)


def _module_parameter_count(module: Any, *, trainable_only: bool = False) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if not trainable_only or bool(parameter.requires_grad)
        )
    )


def _sac_parameter_counts(policy: Any) -> dict[str, int]:
    actor = policy.policy.actor
    critic = policy.policy.critic
    critic_target = policy.policy.critic_target
    return {
        "actor": _module_parameter_count(actor),
        "actor_trainable": _module_parameter_count(actor, trainable_only=True),
        "critic": _module_parameter_count(critic),
        "critic_trainable": _module_parameter_count(critic, trainable_only=True),
        "critic_target": _module_parameter_count(critic_target),
        "critic_target_trainable": _module_parameter_count(critic_target, trainable_only=True),
        "total": _module_parameter_count(policy.policy),
        "total_trainable": _module_parameter_count(policy.policy, trainable_only=True),
        "optimized_total": (
            _module_parameter_count(actor, trainable_only=True)
            + _module_parameter_count(critic, trainable_only=True)
        ),
    }


def _policy_inference_latency_p95_ms(policy: Any, samples: int = 64) -> float:
    observations = np.zeros((max(1, int(samples)), OBSERVATION_SIZE), dtype=np.float32)
    policy.predict(observations[:1], deterministic=True)
    timings: list[float] = []
    for observation in observations:
        started = time.perf_counter()
        policy.predict(observation, deterministic=True)
        timings.append((time.perf_counter() - started) * 1_000.0)
    return float(np.percentile(timings, 95))


class SafeSacTuner:
    """Run shielded SAC with global restarts, multiple basins, and online replay."""

    def __init__(
        self,
        ensemble: SurrogateEnsemble,
        policy_path: Path,
        config: TuningConfig,
        history: list[IterationRecord] | None = None,
        validation_starts: list[HardwarePidCandidate] | None = None,
        exploration_starts: list[HardwarePidCandidate] | None = None,
        episode_budget: int = 15,
        confirmation_count: int = 3,
        validation_episodes: int = 1,
        hardware_protection_mode: bool = False,
        run_full_budget: bool = False,
        seed: int = 20260709,
    ):
        require_ml_dependencies()
        from stable_baselines3 import SAC

        if not hardware_protection_mode and not bool(ensemble.manifest.get("ready", False)):
            raise RuntimeError(f"DRL model '{ensemble.model_id}' is not accepted and ready for hardware use.")
        if not policy_path.exists():
            raise RuntimeError(f"Safe SAC policy file is missing: {policy_path}")
        self.ensemble = ensemble
        self._policy_path = Path(policy_path)
        self._online_dir = self._policy_path.parent / "online"
        self._online_policy_path = self._policy_path.parent / "safe_sac_online_latest.zip"
        self._online_replay_path = self._online_dir / "replay_buffer.pkl"
        self._online_state_path = self._online_dir / "state.json"
        self._online_transitions_path = self._online_dir / "hardware_transitions.jsonl"
        self._online_dir.mkdir(parents=True, exist_ok=True)
        load_path = self._online_policy_path if self._online_policy_path.exists() else self._policy_path
        self.policy = SAC.load(load_path, device=ensemble.device)
        self.policy.set_random_seed(seed)
        try:
            from stable_baselines3.common.logger import configure

            self.policy.set_logger(configure(str(self._online_dir / "sb3_logs"), []))
        except Exception:
            pass
        if self._online_replay_path.exists():
            try:
                self.policy.load_replay_buffer(self._online_replay_path)
            except Exception:
                pass
        self.config = config
        self.episode_budget = max(1, int(episode_budget))
        self.confirmation_count = max(1, int(confirmation_count))
        self.validation_episodes = max(1, int(validation_episodes))
        self.is_validation = self.validation_episodes > 1
        self.validation_starts = _ordered_hardware_starts(
            list(validation_starts or [_center_candidate(config)]),
            is_validation=self.is_validation,
        )
        self.exploration_starts = _ordered_hardware_starts(
            [
                _with_phase(candidate, "drl_shape_retest")
                for candidate in (exploration_starts or [])
                if np.all(
                    np.abs(candidate_to_normalized(candidate, config.search, clip=False))
                    <= 1.0 + 1e-9
                )
            ],
            is_validation=True,
        )
        self.hardware_protection_mode = bool(hardware_protection_mode)
        self.run_full_budget = bool(run_full_budget)
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._last_metadata: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._pending_transitions: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._initial_history = list(history or [])
        online_state = read_json(self._online_state_path) or {}
        self._online_total_samples = max(0, int(online_state.get("total_samples", 0) or 0))
        self._online_total_hardware_samples = max(
            self._online_total_samples,
            int(online_state.get("total_hardware_samples", self._online_total_samples) or 0),
        )
        self._online_updates = max(0, int(online_state.get("updates", 0) or 0))
        self._online_new_samples = 0
        self._online_status = str(online_state.get("status") or "ready")
        self._surrogate_score_offset = float(online_state.get("surrogate_score_offset", 0.0) or 0.0)
        confirmed_payload, confirmed_objective = _confirmed_start_from_online_artifacts(
            online_state,
            self._online_transitions_path,
        )
        self._confirmed_best_candidate = dict(confirmed_payload) if confirmed_payload is not None else None
        self._confirmed_best_objective = float(confirmed_objective)
        if confirmed_payload is not None and not self.is_validation:
            persisted_start = candidate_from_mapping(confirmed_payload, phase="drl_start")
            normalized_start = candidate_to_normalized(persisted_start, self.config.search, clip=False)
            if np.all(np.abs(normalized_start) <= 1.0 + 1e-9):
                starts_by_key = {candidate_key(persisted_start): persisted_start}
                for candidate in self.validation_starts:
                    starts_by_key.setdefault(candidate_key(candidate), candidate)
                self.validation_starts = _ordered_hardware_starts(
                    list(starts_by_key.values()),
                    is_validation=False,
                )
        self._surrogate_residuals: list[float] = []
        self._validation_observations: list[np.ndarray] = []
        self._validation_frozen = False
        self._best_validation_score = float(online_state.get("best_validation_score", float("inf")))
        self._best_policy_path = self._policy_path.parent / "safe_sac_online_best.zip"
        # Continue the low-discrepancy sequence across independent hardware
        # runs. Restarting Sobol at zero repeated the same global candidates
        # (including previously measured protection regions) every run.
        self._global_sequence_offset = self._online_total_hardware_samples

    def next_candidate(
        self,
        history: list[IterationRecord],
        best: IterationRecord | None,
    ) -> HardwarePidCandidate | None:
        episode, episode_records = self._episode_state(history)
        if episode >= self.validation_episodes:
            return None
        if not episode_records:
            start = (
                _with_phase(self.exploration_starts[0], "drl_shape_retest")
                if self.exploration_starts and not self.is_validation
                else _with_phase(self.validation_starts[episode % len(self.validation_starts)], "drl_start")
            )
            source = "shape_fast_seed_retest" if self.exploration_starts and not self.is_validation else "episode_start"
            self._remember(start, episode, 1, source, {"exploration_seed_rank": 1} if source != "episode_start" else None)
            return start

        last = episode_records[-1]
        confirmation = _confirmation_streak(episode_records)
        if self.is_validation:
            source = str(last.optimizer_metadata.get("proposal_source") or "")
            if source == "bandwidth_climb" and not last.metrics.passed:
                return self._next_validation_episode_start(episode, history)
            if (
                last.metrics.passed
                and confirmation < self.confirmation_count
                and last.candidate is not None
            ):
                candidate = _with_phase(last.candidate, "drl_confirm")
                self._remember(
                    candidate,
                    episode,
                    len(episode_records) + 1,
                    "pass_confirmation",
                    None,
                )
                return candidate
            if confirmation >= self.confirmation_count:
                seen = {candidate_key(record.candidate) for record in history if record.candidate is not None}
                climb = _next_higher_bandwidth_candidate(episode_records, self.config.search, seen)
                if climb is not None:
                    base_record = _best_record(episode_records) or last
                    observation = self._observation(episode_records, base_record)
                    action, _ = _relative_replay_action(
                        base_record.candidate or last.candidate,
                        climb,
                        self.config,
                    )
                    self._remember(
                        climb,
                        episode,
                        len(episode_records) + 1,
                        "bandwidth_climb",
                        {"bandwidth_climb_from": last.candidate.mod0_ll_bw if last.candidate else None},
                        observation=observation,
                        action=action,
                        previous_best_score=_record_objective_score(base_record),
                    )
                    return climb
                return self._next_validation_episode_start(episode, history)
            # A historically confirmed near-boundary point may fail one noisy
            # capture. Re-measure the exact raw key before allowing any policy
            # perturbation; validation never explores unrelated 9-D actions.
            if len(episode_records) < min(self.episode_budget, 5):
                anchor = self.validation_starts[episode % len(self.validation_starts)]
                candidate = _with_phase(anchor, "drl_near_pass_retest")
                self._remember(
                    candidate,
                    episode,
                    len(episode_records) + 1,
                    "near_pass_retest",
                    None,
                )
                return candidate
            return self._next_validation_episode_start(episode, history)
        if (
            last.metrics.passed
            and confirmation < self.confirmation_count
            and last.candidate is not None
        ):
            candidate = _with_phase(last.candidate, "drl_confirm")
            self._remember(candidate, episode, len(episode_records) + 1, "pass_confirmation", None)
            return candidate
        if confirmation >= self.confirmation_count:
            seen = {candidate_key(record.candidate) for record in history if record.candidate is not None}
            climb = _next_higher_bandwidth_candidate(episode_records, self.config.search, seen)
            if climb is not None:
                base_record = _best_record(episode_records) or last
                observation = self._observation(episode_records, base_record)
                action, _ = _relative_replay_action(
                    base_record.candidate or last.candidate,
                    climb,
                    self.config,
                )
                self._remember(
                    climb,
                    episode,
                    len(episode_records) + 1,
                    "bandwidth_climb",
                    {"bandwidth_climb_from": last.candidate.mod0_ll_bw if last.candidate else None},
                    observation=observation,
                    action=action,
                    previous_best_score=_record_objective_score(base_record),
                )
                return climb
        confirmation_complete = not self.run_full_budget and confirmation >= self.confirmation_count
        if confirmation_complete or len(episode_records) >= self.episode_budget:
            next_episode = episode + 1
            if next_episode >= self.validation_episodes:
                return None
            start = _with_phase(self.validation_starts[next_episode % len(self.validation_starts)], "drl_start")
            self._remember(start, next_episode, 1, "episode_start", None)
            return start

        next_step = len(episode_records) + 1
        seen = {candidate_key(record.candidate) for record in history if record.candidate is not None}
        if self.exploration_starts:
            for rank, seed_candidate in enumerate(self.exploration_starts, start=1):
                if candidate_key(seed_candidate) in seen:
                    continue
                candidate = _with_phase(seed_candidate, "drl_shape_retest")
                self._remember(
                    candidate,
                    episode,
                    next_step,
                    "shape_fast_seed_retest",
                    {"exploration_seed_rank": rank},
                    replay_eligible=False,
                )
                return candidate
        if next_step % RELIABILITY_RETEST_INTERVAL == 0:
            confirmed_best = _best_record(episode_records)
            if confirmed_best is not None and confirmed_best.candidate is not None:
                retest = _with_phase(confirmed_best.candidate, "drl_reliability_retest")
                self._remember(
                    retest,
                    episode,
                    next_step,
                    "best_reliability_retest",
                    {"reliability_retest_of_iteration": _iteration_number(confirmed_best)},
                )
                return retest
        boundary_repair = self._boundary_repair_proposal(
            episode_records,
            history,
            seen,
            episode=episode,
            next_step=next_step,
        )
        if boundary_repair is not None:
            return boundary_repair
        basins = select_diverse_results(episode_records, 5)
        observed_best = _best_record(episode_records) or best or last
        observed_best_score = float(observed_best.metrics.score)
        basin_schedule = (
            (0, 0, 0, 1, 0, 0, 0, 2)
            if observed_best_score <= GOOD_BASIN_SCORE
            else (0, 1, 0, 2, 0, 3, 0, 4)
        )
        basin_rank = min(basin_schedule[(next_step - 1) % len(basin_schedule)], max(0, len(basins) - 1))
        base_record = (basins[basin_rank] if basins else None) or _best_record(episode_records) or best or last
        if base_record.candidate is None:
            raise RuntimeError("Safe SAC cannot propose a candidate because the episode has no hardware candidate.")
        observation = self._observation(episode_records, base_record)
        stagnation = _stagnation_count(episode_records)
        trust_fraction, global_reason = _exploration_schedule(
            next_step,
            stagnation,
            best_score=observed_best_score,
            confirmed_basins=_confirmed_candidate_count(episode_records, self.confirmation_count),
        )
        if self.hardware_protection_mode:
            # Hardware protection is a last line of defense, not permission
            # for the policy to jump across the sparse 9-D action space.
            trust_fraction = min(trust_fraction, 0.05)
        if global_reason:
            global_candidate, absolute_action, metadata = self._sobol_proposal(
                history,
                seen,
                baseline=_baseline_vector(history),
                reason=global_reason,
            )
            # SAC actions are deltas from the candidate encoded in the
            # observation. A Sobol restart is sampled in absolute normalized
            # coordinates, so writing that absolute vector to replay teaches
            # a different transition from the one the hardware executed.
            # Store the direction as a clipped relative delta instead. Global
            # transitions retain their deliberately low replay weight.
            replay_action, replay_action_clipped = _relative_replay_action(
                base_record.candidate,
                global_candidate,
                self.config,
                trust_fraction=GOOD_BASIN_TRUST_FRACTION,
            )
            metadata.update(
                {
                    "trust_fraction": trust_fraction,
                    "stagnation_count": stagnation,
                    "basin_rank": basin_rank + 1,
                    "basin_count": len(basins),
                    "global_normalized_action": np.asarray(absolute_action, dtype=float).tolist(),
                    "replay_action_semantics": "relative_delta_from_observation_base",
                    "replay_action_clipped": bool(replay_action_clipped),
                }
            )
            self._remember(
                global_candidate,
                episode,
                next_step,
                global_reason,
                metadata,
                observation=observation,
                action=replay_action,
                previous_best_score=_record_objective_score(_best_record(episode_records) or last),
                replay_eligible=True,
                replay_weight=GLOBAL_REPLAY_WEIGHT,
            )
            return global_candidate

        forced_kpole_pair = (
            _least_used_kpole_pair(episode_records)
            if next_step % KPOLE_DIVERSITY_INTERVAL == 0
            else None
        )
        proposals: list[
            tuple[int, float, float, float, float, HardwarePidCandidate, dict[str, Any], np.ndarray]
        ] = []
        rejections = {"duplicate": 0, "protection_probability": 0, "invalid_probability": 0, "uncertainty": 0}
        baseline = _baseline_vector(history)
        for action_array, proposal_family in _mixed_local_proposal_actions(
            self.policy,
            observation,
            self._rng,
            seed=self.seed,
            next_step=next_step,
        ):
            candidate = candidate_with_delta(
                base_record.candidate,
                action_array,
                self.config.search,
                "drl_policy",
                trust_fraction=trust_fraction,
            )
            if forced_kpole_pair is not None:
                candidate = replace(
                    candidate,
                    mod0_kpole1=forced_kpole_pair[0],
                    mod0_kpole2=forced_kpole_pair[1],
                )
                # The scheduler overrides two policy outputs. Replay must use
                # the relative action that produced the candidate the hardware
                # actually ran, not the pre-override SAC sample.
                action_array, diversity_action_clipped = _relative_replay_action(
                    base_record.candidate,
                    candidate,
                    self.config,
                    trust_fraction=trust_fraction,
                )
            else:
                diversity_action_clipped = False
            if candidate_key(candidate) in seen:
                rejections["duplicate"] += 1
                continue
            feature = np.concatenate(
                [
                    candidate_to_normalized(candidate, self.config.search),
                    baseline,
                    np.ones(len(METRIC_FIELDS), dtype=np.float64),
                ]
            ).reshape(1, -1)
            prediction = self.ensemble.predict_features(feature)
            safety_probability = float(prediction["safety_probability"][0])
            validity_probability = float(
                prediction.get("validity_probability", prediction["safety_probability"])[0]
            )
            invalid_probability = np.asarray(
                prediction.get("invalid_probability", np.zeros((1, 3), dtype=np.float64))[0],
                dtype=np.float64,
            )
            uncertainty = float(prediction["uncertainty"][0])
            if not self.hardware_protection_mode:
                if safety_probability < 0.995:
                    rejections["protection_probability"] += 1
                    continue
                if (
                    float(np.max(invalid_probability[:SAFETY_LABEL_COUNT]))
                    >= self.ensemble.invalid_probability_threshold
                ):
                    rejections["invalid_probability"] += 1
                    continue
                if uncertainty > self.ensemble.uncertainty_threshold:
                    rejections["uncertainty"] += 1
                    continue
            metrics = np.asarray(prediction["metric_mean"][0], dtype=np.float64)
            metric_std = np.asarray(prediction["metric_std"][0], dtype=np.float64)
            raw_score, metric_predicted_pass = relabeled_score(_metric_payload(metrics), self.config.targets)
            pass_probability = float(
                np.clip(prediction.get("pass_probability", [0.0])[0], 0.0, 1.0)
            )
            predicted_pass = bool(
                metric_predicted_pass
                and pass_probability >= _robust_pass_threshold(self.ensemble)
            )
            score = self._calibrated_predicted_score(raw_score)
            predicted_objective, predicted_bonus = bandwidth_objective(
                score,
                candidate,
                passed=predicted_pass,
            )
            selection_score, empirical_score, nearest_distance = _calibrated_hardware_score(
                candidate,
                predicted_objective,
                episode_records,
                self.config,
            )
            settling_ucb_us = float(
                max(
                    metrics[2] + BOUNDARY_REPAIR_CONFIDENCE_Z * max(0.0, metric_std[2]),
                    metrics[3] + BOUNDARY_REPAIR_CONFIDENCE_Z * max(0.0, metric_std[3]),
                )
            )
            fast_selection_us, empirical_settling_us, settling_neighbor_distance = (
                _calibrated_hardware_settling_us(
                    candidate,
                    settling_ucb_us,
                    episode_records,
                    self.config,
                )
            )
            metadata = {
                "predicted_metrics": vector_to_metric_mapping(metrics),
                "metric_std": vector_to_metric_mapping(metric_std),
                "predicted_score": score,
                "predicted_objective": predicted_objective,
                "predicted_bandwidth_bonus": predicted_bonus,
                "raw_predicted_score": raw_score,
                "surrogate_score_offset": self._surrogate_score_offset,
                "selection_score": selection_score,
                "predicted_settling_ucb_us": settling_ucb_us,
                "fast_settling_selection_us": fast_selection_us,
                "empirical_neighbor_settling_us": empirical_settling_us,
                "settling_neighbor_distance": settling_neighbor_distance,
                "empirical_neighbor_score": empirical_score,
                "nearest_hardware_distance": nearest_distance,
                "predicted_pass": predicted_pass,
                "metric_predicted_pass": metric_predicted_pass,
                "predicted_pass_probability": pass_probability,
                "robust_pass_probability_threshold": _robust_pass_threshold(self.ensemble),
                "safety_probability": safety_probability,
                "validity_probability": validity_probability,
                "uncertainty": uncertainty,
                "hardware_protection_mode": self.hardware_protection_mode,
                "rejected_proposals": dict(rejections),
                "trust_fraction": trust_fraction,
                "stagnation_count": stagnation,
                "basin_rank": basin_rank + 1,
                "basin_iteration": _iteration_number(base_record),
                "basin_count": len(basins),
                "forced_kpole_pair": list(forced_kpole_pair) if forced_kpole_pair is not None else None,
                "replay_action_semantics": "relative_delta_from_observation_base",
                "replay_action_clipped": bool(diversity_action_clipped),
                "proposal_family": proposal_family,
            }
            proposals.append(
                (
                    0 if predicted_pass else 1,
                    fast_selection_us if predicted_pass else selection_score,
                    selection_score if predicted_pass else fast_selection_us,
                    -pass_probability,
                    -float(candidate.mod0_ll_bw),
                    candidate,
                    metadata,
                    action_array,
                )
            )
        if not proposals and self.hardware_protection_mode:
            candidate, action, metadata = self._sobol_proposal(
                history,
                seen,
                baseline=baseline,
                reason="global_legal_fallback",
            )
            metadata.update(
                {
                    "proposal_fallback": "sobol_global_legal_search",
                    "rejected_proposals": dict(rejections),
                    "trust_fraction": trust_fraction,
                    "stagnation_count": stagnation,
                    "basin_rank": basin_rank + 1,
                    "basin_iteration": _iteration_number(base_record),
                    "basin_count": len(basins),
                }
            )
            selection_score, empirical_score, nearest_distance = _calibrated_hardware_score(
                candidate,
                float(metadata.get("predicted_objective", metadata["predicted_score"])),
                episode_records,
                self.config,
            )
            predicted_metrics = metadata.get("predicted_metrics") or {}
            predicted_metric_std = metadata.get("metric_std") or {}
            settling_ucb_us = max(
                float(predicted_metrics.get("overshoot_settling_time_us", FAST_SETTLING_PASS_LIMIT_US))
                + BOUNDARY_REPAIR_CONFIDENCE_Z
                * max(0.0, float(predicted_metric_std.get("overshoot_settling_time_us", 0.0))),
                float(predicted_metrics.get("undershoot_settling_time_us", FAST_SETTLING_PASS_LIMIT_US))
                + BOUNDARY_REPAIR_CONFIDENCE_Z
                * max(0.0, float(predicted_metric_std.get("undershoot_settling_time_us", 0.0))),
            )
            fast_selection_us, empirical_settling_us, settling_neighbor_distance = (
                _calibrated_hardware_settling_us(
                    candidate,
                    settling_ucb_us,
                    episode_records,
                    self.config,
                )
            )
            metadata.update(
                {
                    "selection_score": selection_score,
                    "empirical_neighbor_score": empirical_score,
                    "nearest_hardware_distance": nearest_distance,
                    "predicted_settling_ucb_us": settling_ucb_us,
                    "fast_settling_selection_us": fast_selection_us,
                    "empirical_neighbor_settling_us": empirical_settling_us,
                    "settling_neighbor_distance": settling_neighbor_distance,
                    "proposal_family": "global_fallback",
                }
            )
            proposals.append(
                (
                    0 if bool(metadata.get("predicted_pass")) else 1,
                    fast_selection_us,
                    selection_score,
                    -float(metadata.get("predicted_pass_probability") or 0.0),
                    -float(candidate.mod0_ll_bw),
                    candidate,
                    metadata,
                    action,
                )
            )
        if not proposals:
            gate_description = (
                "duplicate filtering"
                if self.hardware_protection_mode
                else "the 0.995 protection, validity, and uncertainty gates"
            )
            raise RuntimeError(f"Safe SAC found no fresh proposal after {gate_description} (rejections={rejections}).")
        _, _, _, _, _, selected, metadata, selected_action = min(
            proposals,
            key=lambda item: item[:5],
        )
        metadata["rejected_proposals"] = dict(rejections)
        self._remember(
            selected,
            episode,
            next_step,
            "safe_sac_kpole_diversity" if forced_kpole_pair is not None else "safe_sac",
            metadata,
            observation=observation,
            action=selected_action,
            previous_best_score=_record_objective_score(_best_record(episode_records) or last),
            replay_eligible=True,
            replay_weight=GLOBAL_REPLAY_WEIGHT if metadata.get("proposal_fallback") else 1.0,
        )
        return selected

    def _boundary_repair_proposal(
        self,
        episode_records: list[IterationRecord],
        history: list[IterationRecord],
        seen: set[tuple[Any, ...]],
        *,
        episode: int,
        next_step: int,
    ) -> HardwarePidCandidate | None:
        """Repair a failed BW climb locally before resuming broad exploration.

        A single-code BW climb identifies a useful constraint boundary. Throwing
        that information away and sampling another unrestricted 9-D action is
        both inefficient and prone to trips. Keep the failed BW fixed, perturb
        the confirmed lower-BW raw key by small hardware-native steps, and rank
        fresh candidates by robust pass probability and an upper confidence
        bound on the two settling times.
        """

        context = _bandwidth_repair_context(episode_records, self.confirmation_count)
        if context is None:
            return None
        failed_climb, anchor = context
        if failed_climb.candidate is None or anchor.candidate is None:
            return None
        if _boundary_repair_attempt_count(episode_records, failed_climb) >= BOUNDARY_REPAIR_ATTEMPT_LIMIT:
            # Hardware evidence from the completed run shows that an exhausted
            # one-coordinate pool is much less productive than the learned
            # local policy (roughly 19% versus 67% pass rate).  Preserve a
            # bounded repair window, then return the remaining budget to SAC.
            return None
        target_bw = int(failed_climb.candidate.mod0_ll_bw)
        candidates = _boundary_repair_candidates(anchor.candidate, target_bw, self.config, seen)
        if not candidates:
            return None

        baseline = _baseline_vector(history)
        ranked: list[
            tuple[int, float, float, float, HardwarePidCandidate, dict[str, Any]]
        ] = []
        rejected = {"protection_probability": 0, "invalid_probability": 0, "uncertainty": 0}
        target_us = float(self.config.targets.settling_time_s) * 1e6
        for candidate, perturbation in candidates:
            metadata = self._prediction_metadata(candidate, baseline)
            if not self.hardware_protection_mode and not self._prediction_is_allowed(metadata):
                invalid = [float(value) for value in metadata.get("invalid_probability", [])[:SAFETY_LABEL_COUNT]]
                if float(metadata.get("safety_probability", 0.0)) < 0.995:
                    rejected["protection_probability"] += 1
                elif invalid and max(invalid) >= self.ensemble.invalid_probability_threshold:
                    rejected["invalid_probability"] += 1
                else:
                    rejected["uncertainty"] += 1
                continue
            metrics = dict(metadata.get("predicted_metrics") or {})
            metric_std = dict(metadata.get("metric_std") or {})
            os_ts = float(metrics.get("overshoot_settling_time_us", 300.0))
            us_ts = float(metrics.get("undershoot_settling_time_us", 300.0))
            os_std = max(0.0, float(metric_std.get("overshoot_settling_time_us", 0.0)))
            us_std = max(0.0, float(metric_std.get("undershoot_settling_time_us", 0.0)))
            settling_ucb_us = max(
                os_ts + BOUNDARY_REPAIR_CONFIDENCE_Z * os_std,
                us_ts + BOUNDARY_REPAIR_CONFIDENCE_Z * us_std,
            )
            pass_probability = float(metadata.get("predicted_pass_probability", 0.0))
            selection_score, empirical_score, nearest_distance = _calibrated_hardware_score(
                candidate,
                float(metadata.get("predicted_objective", metadata.get("predicted_score", 300.0))),
                episode_records,
                self.config,
            )
            metadata.update(
                {
                    "boundary_repair_from_iteration": _iteration_number(failed_climb),
                    "boundary_repair_anchor_iteration": _iteration_number(anchor),
                    "boundary_repair_target_bw": target_bw,
                    "boundary_repair_perturbation": perturbation,
                    "predicted_settling_ucb_us": settling_ucb_us,
                    "predicted_settling_margin_us": target_us - settling_ucb_us,
                    "selection_score": selection_score,
                    "empirical_neighbor_score": empirical_score,
                    "nearest_hardware_distance": nearest_distance,
                    "rejected_proposals": dict(rejected),
                }
            )
            ranked.append(
                (
                    0 if bool(metadata.get("predicted_pass")) else 1,
                    -pass_probability,
                    settling_ucb_us,
                    selection_score,
                    candidate,
                    metadata,
                )
            )
        if not ranked:
            return None
        _, _, _, _, selected, metadata = min(ranked, key=lambda item: item[:4])
        metadata["rejected_proposals"] = dict(rejected)
        action, action_clipped = _relative_replay_action(
            anchor.candidate,
            selected,
            self.config,
            trust_fraction=0.05 if self.hardware_protection_mode else GOOD_BASIN_TRUST_FRACTION,
        )
        metadata.update(
            {
                "replay_action_semantics": "relative_delta_from_confirmed_boundary_anchor",
                "replay_action_clipped": bool(action_clipped),
            }
        )
        self._remember(
            selected,
            episode,
            next_step,
            "boundary_repair",
            metadata,
            observation=self._observation(episode_records, anchor),
            action=action,
            previous_best_score=_record_objective_score(anchor),
            replay_eligible=True,
        )
        return selected

    def _next_validation_episode_start(
        self,
        episode: int,
        history: list[IterationRecord],
    ) -> HardwarePidCandidate | None:
        next_episode = episode + 1
        if next_episode >= self.validation_episodes:
            return None
        start = _with_phase(
            self.validation_starts[next_episode % len(self.validation_starts)],
            "drl_start",
        )
        self._remember(start, next_episode, 1, "episode_start", None)
        return start

    def observe_result(
        self,
        record: IterationRecord,
        history: list[IterationRecord],
        best: IterationRecord | None,
    ) -> None:
        """Append a real hardware transition and update SAC every 20 samples."""

        try:
            if record.candidate is None:
                return
            pending = self._pending_transitions.pop(candidate_key(record.candidate), None)
            if pending is None:
                return
            next_best = best or record
            next_observation = self._observation(history, next_best)
            score = float(record.metrics.score)
            objective_score = (
                float(record.objective_score)
                if record.objective_score is not None
                else score
            )
            previous_best = float(pending["previous_best_score"])
            invalid = _record_is_invalid(record)
            confirmation_streak = _confirmation_streak(history)
            confirmed_pass = bool(
                record.metrics.passed
                and not invalid
                and confirmation_streak >= max(1, int(self.confirmation_count))
            )
            reward = -5.0 if invalid else float(
                np.clip((previous_best - objective_score) / 50.0 - 0.02, -2.0, 2.0)
            )
            if record.metrics.passed and not invalid:
                # A lucky single capture is only weak evidence.  The large
                # reward is reserved for the third consecutive pass, then
                # scaled by BW so RL learns the lexicographic objective:
                # robust feasibility first, maximum BW second.
                reward += 0.5
                if confirmed_pass:
                    bandwidth_fraction = float(
                        np.clip(
                            (record.candidate.mod0_ll_bw - self.config.search.mod0_ll_bw.min)
                            / max(
                                1.0,
                                self.config.search.mod0_ll_bw.max
                                - self.config.search.mod0_ll_bw.min,
                            ),
                            0.0,
                            1.0,
                        )
                    )
                    reward += 3.5 + 2.0 * bandwidth_fraction
                    fast_quality = _fast_settling_quality(
                        float(record.metrics.overshoot_settling_time_s) * 1e6,
                        float(record.metrics.undershoot_settling_time_s) * 1e6,
                    )
                    reward += 2.0 * FAST_SETTLING_REWARD_SCALE * fast_quality
                    if record.candidate is not None and objective_score < self._confirmed_best_objective:
                        self._confirmed_best_candidate = candidate_to_mapping(record.candidate)
                        self._confirmed_best_objective = objective_score
                else:
                    fast_quality = _fast_settling_quality(
                        float(record.metrics.overshoot_settling_time_s) * 1e6,
                        float(record.metrics.undershoot_settling_time_s) * 1e6,
                    )
                    reward += FAST_SETTLING_REWARD_SCALE * fast_quality
            done = bool(
                confirmed_pass
                and not self.run_full_budget
                and record.candidate.mod0_ll_bw >= int(round(self.config.search.mod0_ll_bw.max))
            )
            replay_eligible = bool(pending.get("replay_eligible", True))
            replay_weight = float(np.clip(pending.get("replay_weight", 1.0), 0.0, 1.0))
            if record.metrics.passed:
                replay_weight = 1.0
            stride = max(1, int(round(1.0 / replay_weight))) if replay_weight > 0 else 0
            replay_admitted = bool(
                replay_eligible
                and replay_weight > 0
                and (replay_weight >= 1.0 or int(record.iteration) % stride == 0)
            )
            if replay_admitted:
                self.policy.replay_buffer.add(
                    np.asarray(pending["observation"], dtype=np.float32).reshape(1, -1),
                    np.asarray(next_observation, dtype=np.float32).reshape(1, -1),
                    np.asarray(pending["action"], dtype=np.float32).reshape(1, -1),
                    np.asarray([reward], dtype=np.float32),
                    np.asarray([done], dtype=np.float32),
                    [{"hardware": True, "iteration": int(record.iteration), "invalid": invalid}],
                )
            transition = {
                "iteration": int(record.iteration),
                "candidate": candidate_to_mapping(record.candidate),
                "reward": reward,
                "objective_score": objective_score,
                "bandwidth_bonus": record.bandwidth_bonus,
                "score": score,
                "passed": bool(record.metrics.passed),
                "confirmation_streak": int(confirmation_streak),
                "confirmed_pass": confirmed_pass,
                "invalid": invalid,
                "replay_eligible": replay_eligible,
                "replay_weight": replay_weight,
                "replay_admitted": replay_admitted,
                "proposal_source": pending.get("proposal_source", ""),
                "observation": np.asarray(pending["observation"], dtype=np.float32).tolist(),
                "action": np.asarray(pending["action"], dtype=np.float32).tolist(),
                "next_observation": next_observation.tolist(),
                "timestamp": time.time(),
            }
            with self._online_transitions_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(transition, separators=(",", ":")) + "\n")
            self._online_total_hardware_samples += 1
            predicted_score = pending.get("predicted_score")
            if isinstance(predicted_score, (int, float)) and np.isfinite(predicted_score) and not invalid:
                self._surrogate_residuals.append(score - float(predicted_score))
                self._surrogate_residuals = self._surrogate_residuals[-SURROGATE_CALIBRATION_WINDOW:]
                if self._online_total_hardware_samples % SURROGATE_CALIBRATION_INTERVAL == 0:
                    self._surrogate_score_offset = float(np.median(self._surrogate_residuals))
                    self._online_status = f"surrogate calibrated ({self._surrogate_score_offset:+.2f})"
            if not self._validation_frozen and len(self._validation_observations) < POLICY_VALIDATION_MAX_SAMPLES:
                self._validation_observations.append(np.asarray(pending["observation"], dtype=np.float32).copy())
            if replay_admitted:
                self._online_total_samples += 1
                self._online_new_samples += 1
                self._online_status = "replay updated"
            else:
                self._online_status = "low-weight global sample logged"
            record.optimizer_metadata["online_learning"] = {
                "replay_samples": self._online_total_samples,
                "hardware_samples": self._online_total_hardware_samples,
                "replay_eligible": replay_eligible,
                "replay_weight": replay_weight,
                "replay_admitted": replay_admitted,
                "updates": self._online_updates,
                "status": self._online_status,
            }
            if replay_admitted and self._online_new_samples >= ONLINE_UPDATE_BATCH:
                self._train_online_batch()
            else:
                self._write_online_state()
        except Exception as exc:
            self._online_status = f"online update skipped: {exc}"
            record.optimizer_metadata["online_learning"] = {"status": self._online_status}
            self._write_online_state()

    def metadata_for(self, candidate: HardwarePidCandidate) -> dict[str, Any]:
        metadata = dict(
            self._last_metadata.get(
                candidate_key(candidate),
                {"algorithm": "deep-reinforcement", "model_id": self.ensemble.model_id},
            )
        )
        metadata["online_learning_status"] = self._online_status
        metadata["online_replay_samples"] = self._online_total_samples
        metadata["online_hardware_samples"] = self._online_total_hardware_samples
        metadata["online_updates"] = self._online_updates
        return metadata

    def _sobol_proposal(
        self,
        history: list[IterationRecord],
        seen: set[tuple[Any, ...]],
        *,
        baseline: np.ndarray,
        reason: str,
    ) -> tuple[HardwarePidCandidate, np.ndarray, dict[str, Any]]:
        import torch

        global_index = sum(
            1
            for record in history
            if "global" in str(record.optimizer_metadata.get("proposal_source", ""))
            or "sobol" in str(record.optimizer_metadata.get("proposal_source", ""))
        )
        sequence_index = max(0, int(getattr(self, "_global_sequence_offset", 0))) + global_index
        engine = torch.quasirandom.SobolEngine(len(ACTION_FIELDS), scramble=True, seed=self.seed)
        if sequence_index:
            engine.fast_forward(sequence_index)
        for attempt in range(1024):
            action = engine.draw(1).cpu().numpy()[0].astype(np.float64) * 2.0 - 1.0
            pair_index = (sequence_index + attempt) % len(KPOLE_PAIRS)
            kpole1, kpole2 = KPOLE_PAIRS[pair_index]
            kpole1_span = max(1e-12, self.config.search.mod0_kpole1.max - self.config.search.mod0_kpole1.min)
            kpole2_span = max(1e-12, self.config.search.mod0_kpole2.max - self.config.search.mod0_kpole2.min)
            action[3] = (kpole1 - self.config.search.mod0_kpole1.min) / kpole1_span * 2.0 - 1.0
            action[4] = (kpole2 - self.config.search.mod0_kpole2.min) / kpole2_span * 2.0 - 1.0
            candidate = candidate_from_normalized(action, self.config.search, "drl_global")
            if candidate_key(candidate) in seen:
                continue
            metadata = self._prediction_metadata(candidate, baseline)
            if not self.hardware_protection_mode and not self._prediction_is_allowed(metadata):
                continue
            metadata.update(
                {
                    "global_exploration_reason": reason,
                    "sobol_index": sequence_index + attempt,
                    "kpole_pair": list(KPOLE_PAIRS[pair_index]),
                }
            )
            return candidate, action, metadata
        raise RuntimeError("Safe SAC could not find a fresh Sobol candidate after 1024 attempts.")

    def _prediction_metadata(self, candidate: HardwarePidCandidate, baseline: np.ndarray) -> dict[str, Any]:
        feature = np.concatenate(
            [
                candidate_to_normalized(candidate, self.config.search),
                baseline,
                np.ones(len(METRIC_FIELDS), dtype=np.float64),
            ]
        ).reshape(1, -1)
        prediction = self.ensemble.predict_features(feature)
        metrics = np.asarray(prediction["metric_mean"][0], dtype=np.float64)
        raw_score, metric_predicted_pass = relabeled_score(_metric_payload(metrics), self.config.targets)
        pass_probability = float(
            np.clip(prediction.get("pass_probability", [0.0])[0], 0.0, 1.0)
        )
        predicted_pass = bool(
            metric_predicted_pass
            and pass_probability >= _robust_pass_threshold(self.ensemble)
        )
        score = self._calibrated_predicted_score(raw_score)
        predicted_objective, predicted_bonus = bandwidth_objective(
            score,
            candidate,
            passed=predicted_pass,
        )
        invalid_probability = np.asarray(
            prediction.get("invalid_probability", np.zeros((1, 3), dtype=np.float64))[0],
            dtype=np.float64,
        )
        return {
            "predicted_metrics": vector_to_metric_mapping(metrics),
            "metric_std": vector_to_metric_mapping(np.asarray(prediction["metric_std"][0])),
            "predicted_score": score,
            "predicted_objective": predicted_objective,
            "predicted_bandwidth_bonus": predicted_bonus,
            "raw_predicted_score": raw_score,
            "surrogate_score_offset": self._surrogate_score_offset,
            "predicted_pass": predicted_pass,
            "metric_predicted_pass": metric_predicted_pass,
            "predicted_pass_probability": pass_probability,
            "robust_pass_probability_threshold": _robust_pass_threshold(self.ensemble),
            "safety_probability": float(prediction["safety_probability"][0]),
            "validity_probability": float(
                prediction.get("validity_probability", prediction["safety_probability"])[0]
            ),
            "invalid_probability": invalid_probability.tolist(),
            "uncertainty": float(prediction["uncertainty"][0]),
            "hardware_protection_mode": self.hardware_protection_mode,
        }

    def _calibrated_predicted_score(self, score: float) -> float:
        return float(min(300.0, float(score) + self._surrogate_score_offset))

    def _prediction_is_allowed(self, metadata: dict[str, Any]) -> bool:
        return bool(
            float(metadata["safety_probability"]) >= 0.995
            and max(
                float(value)
                for value in metadata.get("invalid_probability", [0.0])[:SAFETY_LABEL_COUNT]
            )
            < self.ensemble.invalid_probability_threshold
            and float(metadata["uncertainty"]) <= self.ensemble.uncertainty_threshold
        )

    def _train_online_batch(self) -> None:
        replay_size = int(self.policy.replay_buffer.size())
        batch_size = min(64, replay_size)
        if batch_size <= 0:
            return
        if len(self._validation_observations) >= POLICY_VALIDATION_MIN_SAMPLES:
            self._validation_frozen = True
        before_validation = self._policy_validation_score()
        before_parameters = copy.deepcopy(self.policy.get_parameters())
        self._online_status = "training"
        self.policy.train(gradient_steps=ONLINE_GRADIENT_STEPS, batch_size=batch_size)
        after_validation = self._policy_validation_score()
        accepted = bool(
            before_validation is None
            or after_validation is None
            or after_validation <= before_validation + 1.0
        )
        if not accepted:
            self.policy.set_parameters(before_parameters, exact_match=True)
            self._online_status = (
                f"policy rollback (validation {before_validation:.2f} -> {after_validation:.2f})"
            )
        else:
            self._online_updates += 1
            self._online_status = (
                "trained"
                if after_validation is None
                else f"trained (validation {after_validation:.2f})"
            )
            if after_validation is not None and after_validation < self._best_validation_score:
                self._best_validation_score = after_validation
                temporary_best = self._online_dir / "safe_sac_online_best.tmp.zip"
                self.policy.save(temporary_best)
                temporary_best.replace(self._best_policy_path)
        self._online_new_samples = 0
        self._persist_online_artifacts()
        self._write_online_state()

    def _policy_validation_score(self) -> float | None:
        if len(self._validation_observations) < POLICY_VALIDATION_MIN_SAMPLES:
            return None
        observations = np.stack(self._validation_observations).astype(np.float32)
        actions, _ = self.policy.predict(observations, deterministic=True)
        scores: list[float] = []
        for observation, action in zip(observations, np.asarray(actions)):
            baseline = observation[: len(METRIC_FIELDS)] * self.ensemble.metric_std + self.ensemble.metric_mean
            candidate = _candidate_from_policy_observation(
                observation,
                action,
                self.config,
                trust_fraction=0.05 if self.hardware_protection_mode else GOOD_BASIN_TRUST_FRACTION,
            )
            feature = np.concatenate(
                [candidate_to_normalized(candidate, self.config.search), baseline, np.ones(len(METRIC_FIELDS))]
            ).reshape(1, -1)
            prediction = self.ensemble.predict_features(feature)
            invalid_probability = np.asarray(
                prediction.get("invalid_probability", np.zeros((1, 3), dtype=np.float64))[0]
            )
            if (
                float(prediction["safety_probability"][0]) < 0.995
                or float(np.max(invalid_probability[:SAFETY_LABEL_COUNT]))
                >= self.ensemble.invalid_probability_threshold
            ):
                scores.append(300.0)
                continue
            metrics = np.asarray(prediction["metric_mean"][0], dtype=np.float64)
            score, metric_predicted_pass = relabeled_score(_metric_payload(metrics), self.config.targets)
            pass_probability = float(
                np.clip(prediction.get("pass_probability", [0.0])[0], 0.0, 1.0)
            )
            predicted_pass = bool(
                metric_predicted_pass
                and pass_probability >= _robust_pass_threshold(self.ensemble)
            )
            calibrated = self._calibrated_predicted_score(score)
            if not predicted_pass:
                scores.append(300.0 + 100.0 * (1.0 - pass_probability))
            else:
                scores.append(bandwidth_objective(calibrated, candidate, passed=True)[0])
        return float(np.median(scores)) if scores else None

    def _persist_online_artifacts(self) -> None:
        # Stable-Baselines3 only auto-appends .zip when the supplied path has
        # no suffix. A `.tmp` suffix therefore produced a valid ZIP named
        # exactly `.tmp`, while the old code tried to rename `.tmp.zip`.
        temporary_policy_zip = self._online_dir / "safe_sac_online_latest.tmp.zip"
        self.policy.save(temporary_policy_zip)
        temporary_policy_zip.replace(self._online_policy_path)
        temporary_replay = self._online_dir / "replay_buffer.tmp.pkl"
        self.policy.save_replay_buffer(temporary_replay)
        temporary_replay.replace(self._online_replay_path)

    def _write_online_state(self) -> None:
        atomic_write_json(
            self._online_state_path,
            {
                "total_samples": self._online_total_samples,
                "total_hardware_samples": self._online_total_hardware_samples,
                "updates": self._online_updates,
                "status": self._online_status,
                "surrogate_score_offset": self._surrogate_score_offset,
                "best_validation_score": self._best_validation_score,
                "confirmed_best_candidate": self._confirmed_best_candidate,
                "confirmed_best_objective": self._confirmed_best_objective,
                "updated_at": time.time(),
            },
        )

    def _episode_state(self, history: list[IterationRecord]) -> tuple[int, list[IterationRecord]]:
        drl_records = [
            record
            for record in history
            if str(record.optimizer_metadata.get("algorithm", "")).lower() == "deep-reinforcement"
        ]
        if not drl_records:
            return 0, []
        episode = max(int(record.optimizer_metadata.get("episode", 0)) for record in drl_records)
        records = [record for record in drl_records if int(record.optimizer_metadata.get("episode", 0)) == episode]
        return episode, records

    def _observation(self, records: list[IterationRecord], best: IterationRecord) -> np.ndarray:
        last = records[-1]
        baseline = _baseline_vector(records)
        last_action = candidate_to_normalized(last.candidate or _center_candidate(self.config), self.config.search)
        best_action = candidate_to_normalized(best.candidate or _center_candidate(self.config), self.config.search)
        last_metrics = _record_metric_vector(last)
        best_metrics = _record_metric_vector(best)
        normalized_last = (last_metrics - self.ensemble.metric_mean) / self.ensemble.metric_std
        normalized_best = (best_metrics - self.ensemble.metric_mean) / self.ensemble.metric_std
        value = np.concatenate(
            [
                (baseline - self.ensemble.metric_mean) / self.ensemble.metric_std,
                last_action,
                normalized_last,
                best_action,
                normalized_best,
                np.asarray([len(records) / self.episode_budget, 0.0]),
            ]
        )
        return np.clip(value, -20.0, 20.0).astype(np.float32)

    def _remember(
        self,
        candidate: HardwarePidCandidate,
        episode: int,
        step: int,
        source: str,
        prediction: dict[str, Any] | None,
        *,
        observation: np.ndarray | None = None,
        action: np.ndarray | None = None,
        previous_best_score: float | None = None,
        replay_eligible: bool = True,
        replay_weight: float = 1.0,
    ) -> None:
        self._last_metadata[candidate_key(candidate)] = {
            "algorithm": "deep-reinforcement",
            "model_id": self.ensemble.model_id,
            "episode": episode,
            "episode_step": step,
            "proposal_source": source,
            **(prediction or {}),
        }
        if observation is not None and action is not None and previous_best_score is not None:
            self._pending_transitions[candidate_key(candidate)] = {
                "observation": np.asarray(observation, dtype=np.float32),
                "action": np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0),
                "previous_best_score": float(previous_best_score),
                "proposal_source": source,
                "replay_eligible": bool(replay_eligible),
                "replay_weight": float(np.clip(replay_weight, 0.0, 1.0)),
                "predicted_score": (prediction or {}).get(
                    "raw_predicted_score",
                    (prediction or {}).get("predicted_score"),
                ),
            }


def validation_start_candidates(dataset: DrlDataset, count: int = 4) -> list[HardwarePidCandidate]:
    valid = [
        index
        for index in range(dataset.size)
        if np.max(dataset.invalid_labels[index, :SAFETY_LABEL_COUNT]) <= 0
        and np.all(dataset.metric_mask[index, :6] > 0)
        and np.all(np.abs(dataset.actions[index]) <= 1.0 + 1e-6)
    ]
    if not valid:
        return []
    confirmed_valid = [index for index in valid if float(dataset.passed[index]) >= 0.80]
    confirmed_keys = {candidate_key(dataset.candidates[index]) for index in confirmed_valid}
    if len(confirmed_keys) >= max(1, int(count)):
        # Hardware validation begins from independently measured, repeatably
        # passing raw keys. It validates retention and BW climbing without
        # deliberately programming an unrelated high-penalty global point.
        # Recovery/generalization remains a separate reported experiment.
        ranked_confirmed = sorted(
            confirmed_valid,
            key=lambda index: (
                float(dataset.scores[index]),
                -int(dataset.candidates[index].mod0_ll_bw),
            ),
        )
        selected = [ranked_confirmed[0]]
        selected_keys = {candidate_key(dataset.candidates[selected[0]])}
        while len(selected) < max(1, int(count)):
            available = [
                index
                for index in ranked_confirmed
                if candidate_key(dataset.candidates[index]) not in selected_keys
            ]
            selected_actions = np.asarray(dataset.actions[selected], dtype=np.float64)
            next_index = max(
                available,
                key=lambda index: (
                    float(np.min(np.linalg.norm(selected_actions - dataset.actions[index], axis=1))),
                    -float(dataset.scores[index]),
                    int(dataset.candidates[index].mod0_ll_bw),
                ),
            )
            selected.append(next_index)
            selected_keys.add(candidate_key(dataset.candidates[next_index]))
        return [_with_phase(dataset.candidates[index], "drl_start") for index in selected]
    ranked = sorted(valid, key=lambda index: float(dataset.scores[index]))
    baseline_index = next(
        (
            index
            for index in reversed(valid)
            if str(dataset.records[index].get("phase", "")).lower() == "baseline"
        ),
        ranked[len(ranked) // 2],
    )
    selected_indexes = [baseline_index]
    selected_keys = {candidate_key(dataset.candidates[baseline_index])}

    # Always include the best valid measured point when it is distinct from
    # the baseline, then fill with max-min action-space diversity. The old
    # fixed 25/50/75-percentile selector could choose the repeated baseline at
    # every percentile and incorrectly report fewer than four starts despite
    # dozens of distinct valid candidates.
    for index in ranked:
        key = candidate_key(dataset.candidates[index])
        if key not in selected_keys:
            selected_indexes.append(index)
            selected_keys.add(key)
            break

    unique_valid = []
    seen_valid: set[tuple[Any, ...]] = set()
    for index in valid:
        key = candidate_key(dataset.candidates[index])
        if key in seen_valid:
            continue
        seen_valid.add(key)
        unique_valid.append(index)
    while len(selected_indexes) < max(1, int(count)):
        available = [index for index in unique_valid if candidate_key(dataset.candidates[index]) not in selected_keys]
        if not available:
            break
        selected_actions = np.asarray(dataset.actions[selected_indexes], dtype=np.float64)
        next_index = max(
            available,
            key=lambda index: (
                float(np.min(np.linalg.norm(selected_actions - dataset.actions[index], axis=1))),
                -float(dataset.scores[index]),
            ),
        )
        selected_indexes.append(next_index)
        selected_keys.add(candidate_key(dataset.candidates[next_index]))

    result = [
        _with_phase(dataset.candidates[index], "drl_start")
        for index in selected_indexes[: max(1, int(count))]
    ]
    return result


def _confirmed_start_from_online_artifacts(
    online_state: dict[str, Any],
    transitions_path: Path,
) -> tuple[dict[str, Any] | None, float]:
    """Recover the best confirmed raw key, including artifacts made before it was in state.json."""

    payload = online_state.get("confirmed_best_candidate")
    try:
        objective = float(online_state.get("confirmed_best_objective", float("inf")))
    except (TypeError, ValueError):
        objective = float("inf")
    best = dict(payload) if isinstance(payload, dict) else None
    if not transitions_path.is_file():
        return best, objective
    try:
        with transitions_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    transition = json.loads(line)
                    if not bool(transition.get("confirmed_pass")):
                        continue
                    candidate = transition.get("candidate")
                    candidate_objective = float(transition.get("objective_score", float("inf")))
                    if isinstance(candidate, dict) and np.isfinite(candidate_objective) and candidate_objective < objective:
                        best = dict(candidate)
                        objective = candidate_objective
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
    except OSError:
        pass
    return best, objective


def _ordered_hardware_starts(
    starts: list[HardwarePidCandidate],
    *,
    is_validation: bool,
) -> list[HardwarePidCandidate]:
    """Keep diverse validation order, but start optimization at max confirmed BW."""

    result = list(starts)
    if not is_validation:
        # Persisted starts are ordered for diverse four-episode hardware
        # validation, where the first point may be a low-BW outlying basin.
        # A normal optimization run has only one continuing episode and should
        # begin at the highest already-confirmed BW boundary.
        result.sort(
            key=lambda candidate: (
                -int(candidate.mod0_ll_bw),
                int(candidate.mod0_kp),
                int(candidate.mod0_ki),
                int(candidate.mod0_kd),
            )
        )
    return result


def _baseline_vector(records: list[IterationRecord]) -> np.ndarray:
    baseline = next((record for record in records if record.phase in {"baseline", "drl_start"}), None)
    baseline = baseline or (records[0] if records else None)
    if baseline is None:
        return np.zeros(len(METRIC_FIELDS), dtype=np.float64)
    return _record_metric_vector(baseline)


def _record_metric_vector(record: IterationRecord) -> np.ndarray:
    values, mask = metric_vector(
        {
            "overshoot_pct": record.metrics.overshoot_pct,
            "undershoot_pct": record.metrics.undershoot_pct,
            "overshoot_settling_time_s": record.metrics.overshoot_settling_time_s,
            "undershoot_settling_time_s": record.metrics.undershoot_settling_time_s,
            "phase_margin_deg": record.metrics.phase_margin_deg,
            "crossover_frequency_hz": record.metrics.crossover_frequency_hz,
            "gain_margin_db": record.metrics.gain_margin_db,
            "bode_gain_shape_penalty": record.metrics.bode_gain_shape_penalty,
        }
    )
    values[mask <= 0] = 0.0
    return values


def _best_record(records: list[IterationRecord]) -> IterationRecord | None:
    return select_best_result([record for record in records if record.candidate is not None])


def _relative_replay_action(
    base: HardwarePidCandidate,
    target: HardwarePidCandidate,
    config: TuningConfig,
    *,
    trust_fraction: float = GOOD_BASIN_TRUST_FRACTION,
) -> tuple[np.ndarray, bool]:
    """Encode an executed candidate using the SAC environment's delta semantics."""

    scale = max(1e-9, 2.0 * float(trust_fraction))
    base_action = candidate_to_normalized(base, config.search)
    target_action = candidate_to_normalized(target, config.search)
    raw_delta = (target_action - base_action) / scale
    clipped = bool(np.any(np.abs(raw_delta) > 1.0 + 1e-9))
    return np.clip(raw_delta, -1.0, 1.0).astype(np.float32), clipped


def _candidate_from_policy_observation(
    observation: np.ndarray,
    action: np.ndarray,
    config: TuningConfig,
    *,
    trust_fraction: float,
) -> HardwarePidCandidate:
    """Decode the best raw key in an observation and apply a SAC delta."""

    best_action_start = 2 * len(METRIC_FIELDS) + len(ACTION_FIELDS)
    best_action_stop = best_action_start + len(ACTION_FIELDS)
    normalized_best = np.asarray(observation, dtype=np.float64)[best_action_start:best_action_stop]
    if normalized_best.size != len(ACTION_FIELDS):
        raise ValueError("Policy observation does not contain a complete best-action vector.")
    base = candidate_from_normalized(normalized_best, config.search, "drl_validation_base")
    return candidate_with_delta(
        base,
        np.asarray(action, dtype=np.float64),
        config.search,
        "drl_validation",
        trust_fraction=trust_fraction,
    )


def _bandwidth_repair_context(
    records: list[IterationRecord],
    confirmation_count: int,
) -> tuple[IterationRecord, IterationRecord] | None:
    """Return the latest unresolved failed climb and its confirmed anchor."""

    required = max(1, int(confirmation_count))
    for failed_index in range(len(records) - 1, -1, -1):
        failed = records[failed_index]
        if (
            failed.metrics.passed
            or failed.candidate is None
            or str(failed.optimizer_metadata.get("proposal_source") or "") != "bandwidth_climb"
        ):
            continue
        target_bw = int(failed.candidate.mod0_ll_bw)
        # A later confirmed key at this BW (or above) resolved the boundary.
        if any(
            record.candidate is not None
            and int(record.candidate.mod0_ll_bw) >= target_bw
            and _confirmation_streak(records[: index + 1]) >= required
            for index, record in enumerate(records[failed_index + 1 :], start=failed_index + 1)
        ):
            return None
        confirmed: list[IterationRecord] = []
        for index, record in enumerate(records[: failed_index + 1]):
            if (
                record.candidate is not None
                and record.metrics.passed
                and int(record.candidate.mod0_ll_bw) < target_bw
                and _confirmation_streak(records[: index + 1]) >= required
            ):
                confirmed.append(record)
        if not confirmed:
            return None
        anchor = min(
            confirmed,
            key=lambda record: (
                -int(record.candidate.mod0_ll_bw) if record.candidate is not None else 0,
                _record_objective_score(record),
                -_iteration_number(record),
            ),
        )
        return failed, anchor
    return None


def _boundary_repair_candidates(
    anchor: HardwarePidCandidate,
    target_bw: int,
    config: TuningConfig,
    seen: set[tuple[Any, ...]],
) -> list[tuple[HardwarePidCandidate, dict[str, Any]]]:
    """Generate small, hardware-quantized repairs while holding BW fixed."""

    generated: list[tuple[HardwarePidCandidate, dict[str, Any]]] = []
    local_seen: set[tuple[Any, ...]] = set()
    for field, steps in BOUNDARY_REPAIR_SPECS:
        for direction in (-1, 1):
            for step in steps:
                if field == "output_inductance_raw":
                    raw_value = output_inductance_raw(anchor.output_inductance_nh) + direction * step
                    candidate = replace(
                        anchor,
                        output_inductance_nh=output_inductance_from_raw(raw_value),
                        mod0_ll_bw=target_bw,
                        phase="drl_boundary_repair",
                    )
                elif field == "effective_lc_inductance_raw":
                    raw_value = effective_lc_inductance_raw(anchor.effective_lc_inductance_nh) + direction * step
                    candidate = replace(
                        anchor,
                        effective_lc_inductance_nh=effective_lc_inductance_from_raw(raw_value),
                        mod0_ll_bw=target_bw,
                        phase="drl_boundary_repair",
                    )
                else:
                    candidate = replace(
                        anchor,
                        **{field: int(getattr(anchor, field)) + direction * step},
                        mod0_ll_bw=target_bw,
                        phase="drl_boundary_repair",
                    )
                # Do not clip here: clipping would turn an out-of-range repair
                # (for example kpole=1 when the configured minimum is 2) into
                # an apparently valid boundary point.  Proposal decoding may
                # clip actions, but candidate generation must reject values
                # outside the active hardware search space.
                normalized = candidate_to_normalized(candidate, config.search, clip=False)
                key = candidate_key(candidate)
                if np.any(np.abs(normalized) > 1.0 + 1e-9) or key in seen or key in local_seen:
                    continue
                local_seen.add(key)
                generated.append(
                    (
                        candidate,
                        {"field": field, "direction": direction, "step": step},
                    )
                )
    return generated


def _boundary_repair_attempt_count(
    records: list[IterationRecord],
    failed_climb: IterationRecord,
) -> int:
    """Count hardware repair proposals made for one unresolved BW climb."""

    if failed_climb.candidate is None:
        return 0
    failed_iteration = _iteration_number(failed_climb)
    target_bw = int(failed_climb.candidate.mod0_ll_bw)
    return sum(
        1
        for record in records
        if _iteration_number(record) > failed_iteration
        and record.candidate is not None
        and int(record.candidate.mod0_ll_bw) == target_bw
        and str(record.optimizer_metadata.get("proposal_source") or "") == "boundary_repair"
    )


def _iteration_number(record: IterationRecord | Any) -> int:
    value = getattr(record, "iteration", 0)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and np.isfinite(value):
        return int(value)
    return 0


def _exploration_schedule(
    next_step: int,
    stagnation: int,
    best_score: float = float("inf"),
    confirmed_basins: int = 0,
) -> tuple[float, str]:
    if confirmed_basins >= 3:
        global_interval = GLOBAL_EXPLORATION_INTERVAL_MANY_BASINS
    elif confirmed_basins >= 1:
        global_interval = GLOBAL_EXPLORATION_INTERVAL_TWO_BASINS
    else:
        global_interval = GLOBAL_EXPLORATION_INTERVAL
    if np.isfinite(best_score) and best_score <= GOOD_BASIN_SCORE:
        if next_step % global_interval == 0:
            return GOOD_BASIN_TRUST_FRACTION, "periodic_global_exploration"
        if next_step % MEDIUM_EXPLORATION_INTERVAL == 0:
            return GOOD_BASIN_MEDIUM_TRUST_FRACTION, ""
        return GOOD_BASIN_TRUST_FRACTION, ""
    if stagnation >= SOBOL_RESTART_AFTER:
        return 0.50, "stagnation_sobol_restart"
    if next_step % global_interval == 0:
        trust_fraction = 0.35 if stagnation >= TRUST_REGION_EXPAND_AFTER else 0.20
        return trust_fraction, "periodic_global_exploration"
    if stagnation >= TRUST_REGION_EXPAND_AFTER:
        return 0.35, ""
    return 0.20, ""


def _confirmed_candidate_count(records: list[IterationRecord], required: int = 3) -> int:
    confirmed: set[tuple[Any, ...]] = set()
    previous_key: tuple[Any, ...] | None = None
    streak = 0
    for record in records:
        key = candidate_key(record.candidate) if record.candidate is not None else None
        if record.metrics.passed and key is not None and key == previous_key:
            streak += 1
        elif record.metrics.passed and key is not None:
            streak = 1
        else:
            streak = 0
        previous_key = key
        if key is not None and streak >= max(1, int(required)):
            confirmed.add(key)
    return len(confirmed)


def _least_used_kpole_pair(records: list[IterationRecord]) -> tuple[int, int]:
    """Choose the least-measured legal pole pair for explicit diversity steps."""

    counts = {pair: 0 for pair in KPOLE_PAIRS}
    for record in records:
        if record.candidate is None:
            continue
        pair = (int(record.candidate.mod0_kpole1), int(record.candidate.mod0_kpole2))
        if pair in counts:
            counts[pair] += 1
    return min(KPOLE_PAIRS, key=lambda pair: (counts[pair], KPOLE_PAIRS.index(pair)))


def _mixed_local_proposal_actions(
    policy: Any,
    observation: np.ndarray,
    rng: np.random.Generator,
    *,
    seed: int,
    next_step: int,
) -> list[tuple[np.ndarray, str]]:
    """Mix learned, low-discrepancy, and coordinate actions around one basin."""

    actions: list[tuple[np.ndarray, str]] = []
    for _ in range(POLICY_PROPOSAL_COUNT):
        action, _ = policy.predict(observation, deterministic=False)
        actions.append((np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0), "sac"))

    import torch

    sobol = torch.quasirandom.SobolEngine(
        len(ACTION_FIELDS),
        scramble=True,
        seed=(int(seed) + 104729) % (2**31 - 1),
    )
    sobol_offset = max(0, int(next_step) - 1) * LOCAL_SOBOL_PROPOSAL_COUNT
    if sobol_offset:
        sobol.fast_forward(sobol_offset)
    for action in sobol.draw(LOCAL_SOBOL_PROPOSAL_COUNT).cpu().numpy():
        actions.append((np.asarray(action, dtype=np.float64) * 2.0 - 1.0, "basin_local_sobol"))

    axis_order = np.asarray(rng.permutation(len(ACTION_FIELDS)), dtype=int)
    for index in range(LOCAL_DIRECTIONAL_PROPOSAL_COUNT):
        axis = int(axis_order[(index // 2) % len(axis_order)])
        direction = -1.0 if index % 2 == 0 else 1.0
        magnitude = 0.45 if index < 2 * len(ACTION_FIELDS) else 0.85
        action = np.zeros(len(ACTION_FIELDS), dtype=np.float64)
        action[axis] = direction * magnitude
        actions.append((action, "two_sided_directional"))
    return actions


def _calibrated_hardware_score(
    candidate: HardwarePidCandidate,
    predicted_score: float,
    records: list[IterationRecord],
    config: TuningConfig,
) -> tuple[float, float | None, float | None]:
    """Blend surrogate performance with nearby real hardware observations.

    The surrogate remains the safety model. This score only ranks proposals
    that already passed the safety gates, preventing a poorly calibrated
    penalty prediction from repeatedly steering away from a measured good
    basin.
    """

    candidate_vector = candidate_to_normalized(candidate, config.search)
    neighbors: list[tuple[float, float]] = []
    for record in records:
        if record.candidate is None or _record_is_invalid(record):
            continue
        actual_score = _record_objective_score(record)
        if not np.isfinite(actual_score):
            continue
        distance = float(
            np.linalg.norm(candidate_vector - candidate_to_normalized(record.candidate, config.search))
        )
        neighbors.append((distance, actual_score))
    if not neighbors:
        return float(predicted_score), None, None

    nearest = sorted(neighbors, key=lambda item: item[0])[:HARDWARE_NEIGHBOR_COUNT]
    distances = np.asarray([item[0] for item in nearest], dtype=np.float64)
    scores = np.asarray([item[1] for item in nearest], dtype=np.float64)
    weights = 1.0 / np.square(distances + 0.02)
    empirical_score = float(np.average(scores, weights=weights))
    nearest_distance = float(distances[0])
    # Trust hardware most inside the local 20% normalized neighborhood, then
    # smoothly fall back to the surrogate for genuinely unexplored regions.
    local_confidence = float(np.clip((0.30 - nearest_distance) / 0.20, 0.0, 1.0))
    hardware_weight = 0.85 * local_confidence
    selection_score = (1.0 - hardware_weight) * float(predicted_score) + hardware_weight * empirical_score
    return float(selection_score), empirical_score, nearest_distance


def _calibrated_hardware_settling_us(
    candidate: HardwarePidCandidate,
    predicted_settling_us: float,
    records: list[IterationRecord],
    config: TuningConfig,
) -> tuple[float, float | None, float | None]:
    """Rank fast-response proposals with local hardware Ts before surrogate Ts."""

    candidate_vector = candidate_to_normalized(candidate, config.search)
    neighbors: list[tuple[float, float]] = []
    for record in records:
        if record.candidate is None or _record_is_invalid(record):
            continue
        settling_us = max(
            float(record.metrics.overshoot_settling_time_s) * 1e6,
            float(record.metrics.undershoot_settling_time_s) * 1e6,
        )
        if not np.isfinite(settling_us) or settling_us <= 0:
            continue
        distance = float(
            np.linalg.norm(candidate_vector - candidate_to_normalized(record.candidate, config.search))
        )
        neighbors.append((distance, settling_us))
    if not neighbors:
        return float(predicted_settling_us), None, None

    nearest = sorted(neighbors, key=lambda item: item[0])[:HARDWARE_NEIGHBOR_COUNT]
    distances = np.asarray([item[0] for item in nearest], dtype=np.float64)
    settling = np.asarray([item[1] for item in nearest], dtype=np.float64)
    weights = 1.0 / np.square(distances + 0.02)
    empirical_settling = float(np.average(settling, weights=weights))
    nearest_distance = float(distances[0])
    local_confidence = float(np.clip((0.30 - nearest_distance) / 0.20, 0.0, 1.0))
    hardware_weight = 0.85 * local_confidence
    selection = (1.0 - hardware_weight) * float(predicted_settling_us) + hardware_weight * empirical_settling
    return float(selection), empirical_settling, nearest_distance


def _stagnation_count(records: list[IterationRecord], improvement_epsilon: float = 1e-6) -> int:
    restart_index = -1
    for index, record in enumerate(records):
        if str(record.optimizer_metadata.get("proposal_source", "")) == "stagnation_sobol_restart":
            restart_index = index
    segment = records[restart_index + 1:]
    best_score = float("inf")
    last_improvement = -1
    for index, record in enumerate(segment):
        if _record_is_invalid(record):
            continue
        score = _record_objective_score(record)
        if np.isfinite(score) and score < best_score - improvement_epsilon:
            best_score = score
            last_improvement = index
    if last_improvement < 0:
        return len(segment)
    return max(0, len(segment) - 1 - last_improvement)


def _record_is_invalid(record: IterationRecord) -> bool:
    reasons = " ".join(str(reason).lower() for reason in (record.metrics.pass_reasons or []))
    return bool(
        not np.isfinite(float(record.metrics.score))
        or float(record.metrics.score) >= 300.0
        or "protection" in reasons
        or "invalid" in reasons
        or "duplicate 0 db crossover" in reasons
        or "second 0 db crossover" in reasons
    )


def _record_objective_score(record: IterationRecord) -> float:
    value = record.objective_score
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(record.metrics.score)
    return parsed if np.isfinite(parsed) else float(record.metrics.score)


def _confirmation_streak(records: list[IterationRecord]) -> int:
    if not records or not records[-1].metrics.passed or records[-1].candidate is None:
        return 0
    key = candidate_key(records[-1].candidate)
    streak = 0
    for record in reversed(records):
        if not record.metrics.passed or record.candidate is None or candidate_key(record.candidate) != key:
            break
        streak += 1
    return streak


def _repeat_measurement_noise(dataset: DrlDataset) -> np.ndarray:
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, candidate in enumerate(dataset.candidates):
        grouped.setdefault(candidate_key(candidate), []).append(index)
    residuals: list[list[float]] = [[] for _ in METRIC_FIELDS]
    for indexes in grouped.values():
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
            residuals[metric_index].extend(value - median for value in observed)
    noise = np.zeros(len(METRIC_FIELDS), dtype=np.float64)
    for index, values in enumerate(residuals):
        if values:
            noise[index] = 1.4826 * float(np.median(np.abs(np.asarray(values, dtype=np.float64))))
    return noise


def _metric_payload(values: np.ndarray) -> dict[str, Any]:
    mapping = vector_to_metric_mapping(values)
    return {
        "overshoot_pct": mapping["overshoot_pct"],
        "undershoot_pct": mapping["undershoot_pct"],
        "overshoot_settling_time_s": max(0.0, mapping["overshoot_settling_time_us"]) * 1e-6,
        "undershoot_settling_time_s": max(0.0, mapping["undershoot_settling_time_us"]) * 1e-6,
        "phase_margin_deg": mapping["phase_margin_deg"],
        "crossover_frequency_hz": max(1e-3, mapping["crossover_frequency_khz"]) * 1e3,
        "gain_margin_db": mapping["gain_margin_db"],
        "bode_gain_shape_penalty": max(0.0, mapping["bode_gain_shape_penalty"]),
    }


def _center_candidate(config: TuningConfig) -> HardwarePidCandidate:
    search = config.search
    kpole1 = min((2, 3, 4, 5, 6), key=lambda value: abs(search.mod0_kpole1.center - value))
    kpole2 = min((2, 3, 4, 5, 6), key=lambda value: abs(search.mod0_kpole2.center - value))
    return HardwarePidCandidate(
        mod0_kp=int(round(search.mod0_kp.center)),
        mod0_ki=int(round(search.mod0_ki.center)),
        mod0_kd=int(round(search.mod0_kd.center)),
        mod0_kpole1=kpole1,
        mod0_kpole2=kpole2,
        mod0_cm_gain=int(round(search.mod0_cm_gain.clamped(search.mod0_cm_gain.center))),
        mod0_ll_bw=int(round(search.mod0_ll_bw.clamped(search.mod0_ll_bw.center))),
        output_inductance_nh=float(search.output_inductance_nh.center),
        effective_lc_inductance_nh=float(search.effective_lc_inductance_nh.center),
        phase="drl_start",
    )


def _with_phase(candidate: HardwarePidCandidate, phase: str) -> HardwarePidCandidate:
    payload = candidate_to_mapping(candidate)
    payload["phase"] = phase
    from .common import candidate_from_mapping

    return candidate_from_mapping(payload, phase=phase)
