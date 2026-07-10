"""Synthetic Safe SAC environment, policy training, and guarded online tuner."""

from __future__ import annotations

import hashlib
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

from ..models import HardwarePidCandidate, IterationRecord, TuningConfig
from .common import (
    METRIC_FIELDS,
    atomic_write_json,
    candidate_from_normalized,
    candidate_key,
    candidate_to_mapping,
    candidate_to_normalized,
    candidate_with_delta,
    metric_vector,
    relabeled_score,
    vector_to_metric_mapping,
)
from .dataset import DrlDataset
from .model import SurrogateEnsemble, require_ml_dependencies


OBSERVATION_SIZE = 35


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
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._steps = 0
        self._baseline = np.zeros(len(METRIC_FIELDS), dtype=np.float32)
        self._last_action = np.zeros(6, dtype=np.float32)
        self._last_metrics = np.zeros(len(METRIC_FIELDS), dtype=np.float32)
        self._best_action = np.zeros(6, dtype=np.float32)
        self._best_metrics = np.zeros(len(METRIC_FIELDS), dtype=np.float32)
        self._best_score = 250.0
        self._uncertainty = 0.0
        self._repeat_noise = _repeat_measurement_noise(dataset)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        _ = options
        legal_actions = np.all(np.abs(self.dataset.actions) <= 1.0 + 1e-6, axis=1)
        valid = np.where(
            (np.max(self.dataset.invalid_labels, axis=1) <= 0)
            & (self.dataset.passed <= 0)
            & legal_actions
        )[0]
        if valid.size == 0:
            valid = np.where((np.max(self.dataset.invalid_labels, axis=1) <= 0) & legal_actions)[0]
        if valid.size == 0:
            raise RuntimeError("The DRL dataset has no valid episode starting points.")
        index = int(self._rng.choice(valid))
        self._steps = 0
        self._baseline = self.dataset.features[index, 6:13].astype(np.float32)
        self._last_action = self.dataset.actions[index].astype(np.float32)
        self._last_metrics = self.dataset.metrics[index].astype(np.float32)
        self._best_action = self._last_action.copy()
        self._best_metrics = self._last_metrics.copy()
        self._best_score = float(self.dataset.scores[index])
        self._uncertainty = 0.0
        return self._observation(), {"score": self._best_score}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._steps += 1
        base = candidate_from_normalized(self._best_action, self.config.search, "drl_sim")
        candidate = candidate_with_delta(base, action, self.config.search, "drl_sim", trust_fraction=0.10)
        normalized = candidate_to_normalized(candidate, self.config.search).astype(np.float32)
        feature = np.concatenate(
            [normalized, self._baseline, np.ones(len(METRIC_FIELDS), dtype=np.float32)]
        ).reshape(1, -1)
        prediction = self.ensemble.predict_features(feature)
        safety_probability = float(prediction["safety_probability"][0])
        validity_probability = float(prediction.get("validity_probability", prediction["safety_probability"])[0])
        invalid_probability = np.asarray(
            prediction.get("invalid_probability", np.zeros((1, 3), dtype=np.float64))[0],
            dtype=np.float64,
        )
        self._uncertainty = float(prediction["uncertainty"][0])
        if (
            safety_probability < 0.995
            or validity_probability < 0.50
            or self._uncertainty > self.ensemble.uncertainty_threshold
        ):
            return self._observation(), -5.0, True, False, {
                "unsafe": False,
                "protection": False,
                "invalid": False,
                "shield_rejected": True,
                "safety_probability": safety_probability,
                "validity_probability": validity_probability,
                "uncertainty": self._uncertainty,
            }
        protection_probability = float(np.clip(invalid_probability[0] if invalid_probability.size else 0.0, 0.0, 1.0))
        other_invalid_probability = float(
            np.clip(np.max(invalid_probability[1:]) if invalid_probability.size > 1 else 0.0, 0.0, 1.0)
        )
        protection_event = bool(self._rng.random() < protection_probability)
        invalid_event = bool(protection_event or self._rng.random() < other_invalid_probability)
        if invalid_event:
            return self._observation(), -5.0, True, False, {
                "unsafe": protection_event,
                "protection": protection_event,
                "invalid": True,
                "shield_rejected": False,
                "safety_probability": safety_probability,
                "validity_probability": validity_probability,
                "uncertainty": self._uncertainty,
            }

        mean = np.asarray(prediction["metric_mean"][0], dtype=np.float64)
        epistemic_std = np.asarray(prediction["metric_std"][0], dtype=np.float64)
        std = np.maximum(np.sqrt(epistemic_std ** 2 + self._repeat_noise ** 2), 1e-6)
        measured = self._rng.normal(mean, std)
        measured[:4] = np.maximum(measured[:4], 0.0)
        measured[5] = max(measured[5], 1e-3)
        payload = _metric_payload(measured)
        score, passed = relabeled_score(payload, self.config.targets)
        previous_best = self._best_score
        self._last_action = normalized
        self._last_metrics = measured.astype(np.float32)
        if score < self._best_score:
            self._best_score = score
            self._best_action = normalized.copy()
            self._best_metrics = measured.astype(np.float32)
        reward = (previous_best - self._best_score) / 50.0 - 0.02
        terminated = bool(passed)
        if terminated:
            reward += 2.0
        truncated = self._steps >= self.max_steps and not terminated
        return self._observation(), float(reward), terminated, truncated, {
            "score": score,
            "best_score": self._best_score,
            "passed": passed,
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


def train_safe_sac_policy(
    ensemble: SurrogateEnsemble,
    dataset: DrlDataset,
    config: TuningConfig,
    total_steps: int = 1_000_000,
    evaluation_episodes: int = 10_000,
    max_episode_steps: int = 15,
    seed: int = 20260709,
    progress: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    require_ml_dependencies()
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    if not ensemble.accepted:
        raise RuntimeError("The surrogate acceptance gates failed; Safe SAC training is blocked.")
    environment = SurrogateTuningEnv(ensemble, dataset, config, max_steps=max_episode_steps, seed=seed)

    class ProgressCallback(BaseCallback):
        def _on_step(self) -> bool:
            if progress and self.num_timesteps % max(1000, total_steps // 100) == 0:
                progress(min(0.90, self.num_timesteps / max(1, total_steps) * 0.90), "Training Safe SAC policy")
            return True

    device = "cpu"
    policy = SAC(
        "MlpPolicy",
        environment,
        learning_rate=3e-4,
        buffer_size=min(max(100_000, total_steps), 1_000_000),
        learning_starts=min(10_000, max(100, total_steps // 20)),
        batch_size=256,
        gamma=0.98,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=0,
        seed=seed,
        device=device,
    )
    policy.learn(total_timesteps=max(100, total_steps), callback=ProgressCallback())
    policy_path = ensemble.artifact_dir / "safe_sac_policy"
    policy.save(policy_path)
    evaluation = evaluate_safe_sac_policy(
        policy,
        ensemble,
        dataset,
        config,
        episodes=evaluation_episodes,
        max_episode_steps=max_episode_steps,
        seed=seed + 1,
        progress=progress,
    )
    policy_accepted = evaluation["success_rate"] >= 0.90 and evaluation["protection_rate"] < 0.005
    manifest = dict(ensemble.manifest)
    manifest.update(
        {
            "policy_file": "safe_sac_policy.zip",
            "policy_training_steps": int(total_steps),
            "policy_evaluation": evaluation,
            "policy_accepted": policy_accepted,
            "ready": bool(ensemble.accepted and policy_accepted),
            "policy_created_at": time.time(),
        }
    )
    artifact_files = [
        *[str(item) for item in manifest.get("member_files", [])],
        "scalers.npz",
        "safe_sac_policy.zip",
    ]
    if (ensemble.artifact_dir / "validation_starts.json").is_file():
        artifact_files.append("validation_starts.json")
    manifest["files_sha256"] = {
        filename: hashlib.sha256((ensemble.artifact_dir / filename).read_bytes()).hexdigest()
        for filename in artifact_files
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
) -> dict[str, Any]:
    environment = SurrogateTuningEnv(ensemble, dataset, config, max_steps=max_episode_steps, seed=seed)
    successes = 0
    protections = 0
    invalid_events = 0
    shield_rejections = 0
    steps_to_success: list[int] = []
    for episode in range(max(1, episodes)):
        observation, _ = environment.reset(seed=seed + episode)
        for step in range(max_episode_steps):
            action, _ = policy.predict(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            if info.get("protection"):
                protections += 1
            if info.get("invalid"):
                invalid_events += 1
            if info.get("shield_rejected"):
                shield_rejections += 1
            if terminated and info.get("passed"):
                successes += 1
                steps_to_success.append(step + 1)
            if terminated or truncated:
                break
        if progress and episode % max(10, episodes // 100) == 0:
            progress(0.90 + episode / max(1, episodes) * 0.10, "Evaluating Safe SAC policy")
    return {
        "episodes": int(max(1, episodes)),
        "success_rate": float(successes / max(1, episodes)),
        "protection_rate": float(protections / max(1, episodes)),
        "unsafe_rate": float(protections / max(1, episodes)),
        "invalid_rate": float(invalid_events / max(1, episodes)),
        "shield_rejection_rate": float(shield_rejections / max(1, episodes)),
        "median_steps_to_success": float(np.median(steps_to_success)) if steps_to_success else None,
    }


class SafeSacTuner:
    """Use a frozen SAC policy with an ensemble safety and uncertainty shield."""

    def __init__(
        self,
        ensemble: SurrogateEnsemble,
        policy_path: Path,
        config: TuningConfig,
        history: list[IterationRecord] | None = None,
        validation_starts: list[HardwarePidCandidate] | None = None,
        episode_budget: int = 15,
        confirmation_count: int = 3,
        validation_episodes: int = 1,
        seed: int = 20260709,
    ):
        require_ml_dependencies()
        from stable_baselines3 import SAC

        if not bool(ensemble.manifest.get("ready", False)):
            raise RuntimeError(f"DRL model '{ensemble.model_id}' is not accepted and ready for hardware use.")
        if not policy_path.exists():
            raise RuntimeError(f"Safe SAC policy file is missing: {policy_path}")
        self.ensemble = ensemble
        self.policy = SAC.load(policy_path, device=ensemble.device)
        self.policy.set_random_seed(seed)
        self.config = config
        self.episode_budget = max(1, int(episode_budget))
        self.confirmation_count = max(1, int(confirmation_count))
        self.validation_starts = validation_starts or [_center_candidate(config)]
        self.validation_episodes = max(1, int(validation_episodes))
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._last_metadata: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._initial_history = list(history or [])

    def next_candidate(
        self,
        history: list[IterationRecord],
        best: IterationRecord | None,
    ) -> HardwarePidCandidate | None:
        episode, episode_records = self._episode_state(history)
        if episode >= self.validation_episodes:
            return None
        if not episode_records:
            start = _with_phase(self.validation_starts[episode % len(self.validation_starts)], "drl_start")
            self._remember(start, episode, 1, "episode_start", None)
            return start

        last = episode_records[-1]
        confirmation = _confirmation_streak(episode_records)
        if last.metrics.passed and confirmation < self.confirmation_count and last.candidate is not None:
            candidate = _with_phase(last.candidate, "drl_confirm")
            self._remember(candidate, episode, len(episode_records) + 1, "pass_confirmation", None)
            return candidate
        if confirmation >= self.confirmation_count or len(episode_records) >= self.episode_budget:
            next_episode = episode + 1
            if next_episode >= self.validation_episodes:
                return None
            start = _with_phase(self.validation_starts[next_episode % len(self.validation_starts)], "drl_start")
            self._remember(start, next_episode, 1, "episode_start", None)
            return start

        base_record = _best_record(episode_records) or best or last
        if base_record.candidate is None:
            raise RuntimeError("Safe SAC cannot propose a candidate because the episode has no hardware candidate.")
        observation = self._observation(episode_records, base_record)
        seen = {candidate_key(record.candidate) for record in history if record.candidate is not None}
        proposals: list[tuple[float, HardwarePidCandidate, dict[str, Any]]] = []
        rejections = {"duplicate": 0, "protection_probability": 0, "invalid_probability": 0, "uncertainty": 0}
        baseline = _baseline_vector(history)
        for _ in range(32):
            action, _ = self.policy.predict(observation, deterministic=False)
            candidate = candidate_with_delta(
                base_record.candidate,
                np.asarray(action, dtype=np.float64),
                self.config.search,
                "drl_policy",
                trust_fraction=0.10,
            )
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
            uncertainty = float(prediction["uncertainty"][0])
            if safety_probability < 0.995:
                rejections["protection_probability"] += 1
                continue
            if validity_probability < 0.50:
                rejections["invalid_probability"] += 1
                continue
            if uncertainty > self.ensemble.uncertainty_threshold:
                rejections["uncertainty"] += 1
                continue
            metrics = np.asarray(prediction["metric_mean"][0], dtype=np.float64)
            score, predicted_pass = relabeled_score(_metric_payload(metrics), self.config.targets)
            metadata = {
                "predicted_metrics": vector_to_metric_mapping(metrics),
                "metric_std": vector_to_metric_mapping(np.asarray(prediction["metric_std"][0])),
                "predicted_score": score,
                "predicted_pass": predicted_pass,
                "safety_probability": safety_probability,
                "validity_probability": validity_probability,
                "uncertainty": uncertainty,
                "rejected_proposals": dict(rejections),
            }
            proposals.append((score, candidate, metadata))
        if not proposals:
            raise RuntimeError(
                "Safe SAC fail-closed: no fresh proposal passed the 0.995 protection, validity, and uncertainty "
                f"gates (rejections={rejections})."
            )
        _, selected, metadata = min(proposals, key=lambda item: item[0])
        metadata["rejected_proposals"] = dict(rejections)
        self._remember(selected, episode, len(episode_records) + 1, "safe_sac", metadata)
        return selected

    def metadata_for(self, candidate: HardwarePidCandidate) -> dict[str, Any]:
        return dict(
            self._last_metadata.get(
                candidate_key(candidate),
                {"algorithm": "deep-reinforcement", "model_id": self.ensemble.model_id},
            )
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
    ) -> None:
        self._last_metadata[candidate_key(candidate)] = {
            "algorithm": "deep-reinforcement",
            "model_id": self.ensemble.model_id,
            "episode": episode,
            "episode_step": step,
            "proposal_source": source,
            **(prediction or {}),
        }


def validation_start_candidates(dataset: DrlDataset, count: int = 4) -> list[HardwarePidCandidate]:
    valid = [
        index
        for index in range(dataset.size)
        if np.max(dataset.invalid_labels[index]) <= 0
        and np.all(dataset.metric_mask[index, :6] > 0)
        and np.all(np.abs(dataset.actions[index]) <= 1.0 + 1e-6)
    ]
    if not valid:
        return []
    ranked = sorted(valid, key=lambda index: float(dataset.scores[index]))
    baseline_index = next(
        (
            index
            for index in reversed(valid)
            if str(dataset.records[index].get("phase", "")).lower() == "baseline"
        ),
        ranked[len(ranked) // 2],
    )
    result = [_with_phase(dataset.candidates[baseline_index], "drl_start")]
    for position in (0.25, 0.50, 0.75):
        if len(result) >= count:
            break
        index = ranked[int(round((len(ranked) - 1) * position))]
        candidate = _with_phase(dataset.candidates[index], "drl_start")
        if candidate_key(candidate) not in {candidate_key(item) for item in result}:
            result.append(candidate)
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
        }
    )
    values[mask <= 0] = 0.0
    return values


def _best_record(records: list[IterationRecord]) -> IterationRecord | None:
    valid = [record for record in records if record.candidate is not None]
    return min(valid, key=lambda record: float(record.metrics.score)) if valid else None


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
    }


def _center_candidate(config: TuningConfig) -> HardwarePidCandidate:
    search = config.search
    kpole = 3 if abs(search.mod0_kpole1.center - 3) <= abs(search.mod0_kpole1.center - 6) else 6
    return HardwarePidCandidate(
        mod0_kp=int(round(search.mod0_kp.center)),
        mod0_ki=int(round(search.mod0_ki.center)),
        mod0_kd=int(round(search.mod0_kd.center)),
        mod0_kpole1=kpole,
        mod0_kpole2=kpole,
        mod0_cm_gain=2,
        output_inductance_nh=float(search.output_inductance_nh.center),
        effective_lc_inductance_nh=float(search.effective_lc_inductance_nh.center),
        phase="drl_start",
    )


def _with_phase(candidate: HardwarePidCandidate, phase: str) -> HardwarePidCandidate:
    payload = candidate_to_mapping(candidate)
    payload["phase"] = phase
    from .common import candidate_from_mapping

    return candidate_from_mapping(payload, phase=phase)
