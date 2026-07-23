from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hardware.tuning.drl.common import ACTION_FIELDS, METRIC_FIELDS, candidate_to_normalized
from hardware.tuning.drl.dataset import DrlDataset
from hardware.tuning.drl.model import dependency_status
from hardware.tuning.drl.policy import (
    OBSERVATION_SIZE,
    SurrogateTuningEnv,
    _candidate_from_policy_observation,
    _relative_replay_action,
    evaluate_safe_sac_policy,
    train_safe_sac_policy,
)
from hardware.tuning.models import HardwarePidCandidate, TuningConfig


def _dataset(size: int = 24) -> DrlDataset:
    config = TuningConfig()
    candidates = [
        HardwarePidCandidate(
            mod0_kp=110 + index,
            mod0_ki=170 + index,
            mod0_kd=110 + index,
            mod0_kpole1=3 if index % 2 == 0 else 6,
            mod0_kpole2=3 if index % 2 == 0 else 6,
            mod0_cm_gain=2,
            output_inductance_nh=82.0 + index,
            effective_lc_inductance_nh=300.0 + index,
            phase="historical",
        )
        for index in range(size)
    ]
    actions = np.asarray(
        [candidate_to_normalized(candidate, config.search) for candidate in candidates],
        dtype=np.float32,
    )
    baseline = np.tile(
        np.asarray([1.0, 1.0, 1.5, 1.5, 60.0, 150.0, 10.0, 0.0], dtype=np.float32),
        (size, 1),
    )
    features = np.concatenate([actions, baseline, np.ones_like(baseline)], axis=1)
    return DrlDataset(
        features=features,
        actions=actions,
        metrics=baseline.copy(),
        metric_mask=np.ones_like(baseline),
        invalid_labels=np.zeros((size, 3), dtype=np.float32),
        scores=np.arange(10, 10 + size, dtype=np.float32),
        passed=np.zeros(size, dtype=np.float32),
        groups=np.asarray([f"run-{index // 4}" for index in range(size)]),
        candidates=candidates,
        records=[{"phase": "historical"} for _ in range(size)],
        baseline_values=baseline.copy(),
        baseline_mask=np.ones_like(baseline),
    )


class _SafeEnsemble:
    accepted = True
    invalid_probability_threshold = 0.25
    uncertainty_threshold = 1.0
    device = "cpu"
    metric_mean = np.asarray([1.0, 1.0, 1.5, 1.5, 60.0, 150.0, 10.0, 0.0], dtype=np.float32)
    metric_std = np.ones(len(METRIC_FIELDS), dtype=np.float32)

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir
        self.manifest = {"accepted": True, "model_id": "policy-architecture-test", "member_files": []}
        self.last_features: np.ndarray | None = None

    def predict_features(self, features: np.ndarray) -> dict[str, np.ndarray]:
        self.last_features = np.asarray(features).copy()
        count = len(features)
        return {
            "metric_mean": np.tile(self.metric_mean, (count, 1)),
            "metric_std": np.zeros((count, len(METRIC_FIELDS)), dtype=np.float64),
            "invalid_probability": np.zeros((count, 3), dtype=np.float64),
            "safety_probability": np.ones(count, dtype=np.float64),
            "validity_probability": np.ones(count, dtype=np.float64),
            "uncertainty": np.zeros(count, dtype=np.float64),
        }


class _ZeroPolicy:
    def predict(self, observation, deterministic=True):
        if np.asarray(observation).ndim > 1:
            return np.zeros((len(observation), len(ACTION_FIELDS)), dtype=np.float32), None
        return np.zeros(len(ACTION_FIELDS), dtype=np.float32), None


@unittest.skipUnless(dependency_status()["ok"], "Optional DRL dependencies are not installed.")
class SafeSacArchitectureTest(unittest.TestCase):
    def test_global_candidate_is_encoded_as_relative_replay_delta(self) -> None:
        config = TuningConfig()
        base = HardwarePidCandidate(mod0_kp=150, mod0_ki=200, mod0_kd=140)
        nearby = HardwarePidCandidate(mod0_kp=155, mod0_ki=205, mod0_kd=145)
        distant = HardwarePidCandidate(mod0_kp=255, mod0_ki=255, mod0_kd=255)

        nearby_action, nearby_clipped = _relative_replay_action(base, nearby, config)
        distant_action, distant_clipped = _relative_replay_action(base, distant, config)

        reconstructed = candidate_to_normalized(base, config.search) + nearby_action * 0.2
        np.testing.assert_allclose(
            reconstructed,
            candidate_to_normalized(nearby, config.search),
            atol=1e-6,
        )
        self.assertFalse(nearby_clipped)
        self.assertTrue(distant_clipped)
        self.assertTrue(np.all(np.abs(distant_action) <= 1.0))

    def test_online_validation_decodes_policy_output_as_delta_from_observation_best(self) -> None:
        config = TuningConfig()
        base = HardwarePidCandidate(
            mod0_kp=135,
            mod0_ki=243,
            mod0_kd=175,
            mod0_kpole1=2,
            mod0_kpole2=2,
            mod0_cm_gain=2,
            mod0_ll_bw=76,
            output_inductance_nh=93.586,
            effective_lc_inductance_nh=382.118,
        )
        observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        best_start = 2 * len(METRIC_FIELDS) + len(ACTION_FIELDS)
        observation[best_start : best_start + len(ACTION_FIELDS)] = candidate_to_normalized(
            base, config.search
        )
        action = np.zeros(len(ACTION_FIELDS), dtype=np.float32)
        action[0] = 0.5
        action[2] = -0.25

        decoded = _candidate_from_policy_observation(
            observation,
            action,
            config,
            trust_fraction=0.05,
        )

        self.assertGreater(decoded.mod0_kp, base.mod0_kp)
        self.assertLess(decoded.mod0_kd, base.mod0_kd)
        self.assertEqual(decoded.mod0_ki, base.mod0_ki)
        self.assertEqual(decoded.mod0_ll_bw, base.mod0_ll_bw)

    def test_synthetic_environment_preserves_the_baseline_mask(self) -> None:
        dataset = _dataset()
        expected_mask = np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        dataset.features[0, 14:21] = expected_mask
        with tempfile.TemporaryDirectory() as temporary:
            ensemble = _SafeEnsemble(Path(temporary))
            environment = SurrogateTuningEnv(ensemble, dataset, TuningConfig(), max_steps=1)
            environment.reset(seed=1, options={"validation_start": {"index": 0}})
            environment.step(np.zeros(len(ACTION_FIELDS), dtype=np.float32))

        self.assertIsNotNone(ensemble.last_features)
        np.testing.assert_array_equal(ensemble.last_features[0, 14:21], expected_mask)

    def test_linear_actor_and_critic_are_preserved_and_best_checkpoint_is_restored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = train_safe_sac_policy(
                ensemble=_SafeEnsemble(root),
                dataset=_dataset(),
                config=TuningConfig(),
                total_steps=100,
                evaluation_episodes=4,
                max_episode_steps=2,
                actor_net_arch=(),
                critic_net_arch=(),
                validation_seeds=[101, 102, 103, 104],
                validation_starts=[0, 1, 2, 3],
                checkpoint_interval=50,
                checkpoint_evaluation_episodes=2,
            )

            self.assertEqual(manifest["policy_actor_net_arch"], [])
            self.assertEqual(manifest["policy_critic_net_arch"], [])
            self.assertEqual(manifest["policy_net_arch"], [])
            self.assertEqual(manifest["policy_parameter_counts"]["actor"], 810)
            self.assertEqual(manifest["policy_parameter_counts"]["critic"], 108)
            self.assertEqual(manifest["policy_parameter_counts"]["optimized_total"], 918)
            self.assertEqual(manifest["policy_training_steps"], 100)
            self.assertEqual(manifest["policy_best_checkpoint_step"], 50)
            self.assertEqual(
                [item["step"] for item in manifest["policy_checkpoint_history"]],
                [50, 100],
            )
            self.assertTrue((root / "safe_sac_policy.zip").is_file())
            self.assertTrue((root / "checkpoints" / "safe_sac_best.zip").is_file())
            from stable_baselines3 import SAC

            restored = SAC.load(root / "safe_sac_policy.zip", device="cpu")
            self.assertEqual(restored.num_timesteps, 50)
            validation = json.loads((root / "policy_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(validation["seeds"], [101, 102, 103, 104])
            self.assertEqual(validation["starts"], [{"index": 0}, {"index": 1}, {"index": 2}, {"index": 3}])

    def test_legacy_policy_architecture_populates_actor_and_critic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = train_safe_sac_policy(
                ensemble=_SafeEnsemble(Path(temporary)),
                dataset=_dataset(),
                config=TuningConfig(),
                total_steps=100,
                evaluation_episodes=1,
                max_episode_steps=1,
                policy_net_arch=(8,),
                checkpoint_interval=0,
                checkpoint_evaluation_episodes=1,
            )
        self.assertEqual(manifest["policy_net_arch"], [8])
        self.assertEqual(manifest["policy_actor_net_arch"], [8])
        self.assertEqual(manifest["policy_critic_net_arch"], [8])

    def test_evaluation_uses_fixed_starts_and_reports_penalty_and_confidence_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = evaluate_safe_sac_policy(
                _ZeroPolicy(),
                _SafeEnsemble(Path(temporary)),
                _dataset(),
                TuningConfig(),
                episodes=4,
                max_episode_steps=2,
                seed=1,
                validation_seeds=[11, 12],
                validation_starts=[0, 1],
            )

        self.assertEqual(evaluation["validation_seed_count"], 2)
        self.assertEqual(evaluation["validation_start_count"], 2)
        self.assertAlmostEqual(evaluation["mean_initial_penalty"], 10.5)
        self.assertIn("mean_best_penalty", evaluation)
        self.assertIn("mean_final_penalty", evaluation)
        self.assertIn("p90_steps_to_success", evaluation)
        self.assertEqual(len(evaluation["success_rate_ci95"]), 2)
        self.assertEqual(len(evaluation["protection_rate_ci95"]), 2)


if __name__ == "__main__":
    unittest.main()
