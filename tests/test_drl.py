from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from hardware.tuning.drl.common import (
    candidate_from_mapping,
    candidate_from_normalized,
    candidate_key,
    candidate_to_normalized,
    candidate_with_delta,
    metric_vector,
    operating_signature,
    relabeled_score,
    signatures_compatible,
)
from hardware.tuning.drl.dataset import DrlDataset, build_collection_plan, load_autotune_dataset
from hardware.tuning.drl.model import SurrogateEnsemble, dependency_status
from hardware.tuning.drl.tuner import PlannedCandidateTuner
from hardware.tuning.drl.workflow import DrlWorkflowManager
from hardware.tuning.models import (
    AutotuneExperimentConfig,
    ExperimentResult,
    HardwarePidCandidate,
    ResponseMetrics,
    SearchSpace,
    TuningConfig,
    Waveform,
)
from hardware.tuning.runner import PidAutotuneSession, default_candidate_tuner_factory


def fixed_experiment(**updates) -> AutotuneExperimentConfig:
    experiment = AutotuneExperimentConfig(
        enable_transient_analysis=True,
        enable_bode_analysis=True,
        function_generator_config={
            "mode": "square",
            "frequency_hz": 10_000,
            "low_v": 0,
            "high_v": 1,
        },
        bode_config={
            "start_hz": 1_000,
            "stop_hz": 1_000_000,
            "points": 201,
            "bandwidth_hz": 300,
            "source_vpp": 0.1,
        },
    )
    return replace(experiment, **updates)


def synthetic_dataset(size: int = 32) -> DrlDataset:
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
            effective_lc_inductance_nh=300.0 + index * 3.0,
            phase="baseline" if index == 0 else "historical",
        )
        for index in range(size)
    ]
    actions = np.asarray([candidate_to_normalized(candidate, config.search) for candidate in candidates], dtype=np.float32)
    baseline = np.tile(np.asarray([1.0, 1.0, 1.5, 1.5, 60.0, 150.0, 10.0], dtype=np.float32), (size, 1))
    metrics = baseline.copy()
    metrics[:, 0] += np.arange(size, dtype=np.float32) * 0.04
    features = np.concatenate([actions, baseline, np.ones_like(baseline)], axis=1)
    return DrlDataset(
        features=features,
        actions=actions,
        metrics=metrics,
        metric_mask=np.ones_like(metrics),
        invalid_labels=np.zeros((size, 3), dtype=np.float32),
        scores=np.linspace(-2.0, 100.0, size, dtype=np.float32),
        passed=np.zeros(size, dtype=np.float32),
        groups=np.asarray([f"run-{index // 4}" for index in range(size)]),
        candidates=candidates,
        records=[{"phase": candidate.phase} for candidate in candidates],
        baseline_values=baseline.copy(),
        baseline_mask=np.ones_like(baseline),
    )


class SafePredictor:
    def __init__(self, safety: float = 1.0):
        self.safety = safety

    def predict_features(self, features: np.ndarray) -> dict[str, np.ndarray]:
        count = len(features)
        return {
            "metric_mean": np.tile(np.asarray([1, 1, 1.5, 1.5, 60, 150, 10], dtype=float), (count, 1)),
            "metric_std": np.tile(np.asarray([0.1] * 7, dtype=float), (count, 1)),
            "safety_probability": np.full(count, self.safety, dtype=float),
            "uncertainty": np.linspace(0.0, 1.0, count, dtype=float),
        }


class DrlCoreTest(unittest.TestCase):
    def test_saved_dataset_excludes_actions_outside_current_search_space(self) -> None:
        config = TuningConfig()
        inside = HardwarePidCandidate(mod0_kp=int(config.search.mod0_kp.min))
        outside = HardwarePidCandidate(mod0_kp=int(config.search.mod0_kp.min) - 1)
        metrics = {
            "overshoot_pct": 1.0,
            "undershoot_pct": 1.0,
            "overshoot_settling_time_s": 1e-6,
            "undershoot_settling_time_s": 1e-6,
        }
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "saved" / "run-1"
            run.mkdir(parents=True)
            rows = [
                {"candidate": inside.__dict__, "metrics": metrics},
                {"candidate": outside.__dict__, "metrics": metrics},
            ]
            (run / "iterations.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            (run / "run_status.json").write_text(json.dumps({"history": rows}), encoding="utf-8")
            dataset, manifest = load_autotune_dataset([run.parent], config)
        self.assertEqual(dataset.size, 1)
        self.assertEqual(manifest["excluded_out_of_search_space_count"], 1)

    def test_old_candidate_without_cm_gain_defaults_to_two(self) -> None:
        candidate = candidate_from_mapping({"mod0_kp": 151, "mod0_ki": 201, "mod0_kd": 111})
        self.assertEqual(candidate.mod0_cm_gain, 2)

    def test_missing_bode_metrics_are_masked_not_filled(self) -> None:
        values, mask = metric_vector(
            {
                "overshoot_pct": 1.0,
                "undershoot_pct": 1.0,
                "overshoot_settling_time_s": 1e-6,
                "undershoot_settling_time_s": 1.5e-6,
            }
        )
        self.assertTrue(np.all(mask[:4] == 1))
        self.assertTrue(np.all(mask[4:] == 0))
        self.assertTrue(np.all(values[4:] == 0))

    def test_reward_is_relabelled_with_two_microsecond_target(self) -> None:
        config = TuningConfig()
        passing = {
            "overshoot_pct": 2.0,
            "undershoot_pct": 2.0,
            "overshoot_settling_time_s": 1.9e-6,
            "undershoot_settling_time_s": 2.0e-6,
            "phase_margin_deg": 50.0,
            "crossover_frequency_hz": 190_000.0,
            "gain_margin_db": -20.0,
        }
        score, passed = relabeled_score({"metrics": passing}, config.targets)
        self.assertTrue(passed)
        self.assertLessEqual(score, 0.0)
        failing = dict(passing, undershoot_settling_time_s=2.1e-6)
        self.assertFalse(relabeled_score({"metrics": failing}, config.targets)[1])

    def test_relabelled_settling_penalty_uses_ten_per_microsecond(self) -> None:
        metrics = {
            "overshoot_pct": 3.0,
            "undershoot_pct": 3.0,
            "overshoot_settling_time_s": 3e-6,
            "undershoot_settling_time_s": 4e-6,
            "phase_margin_deg": 45.0,
            "crossover_frequency_hz": 200_000.0,
            "gain_margin_db": 6.0,
        }
        score, passed = relabeled_score({"metrics": metrics}, TuningConfig().targets)
        self.assertFalse(passed)
        self.assertAlmostEqual(score, 30.0)

    def test_action_quantization_and_trust_region(self) -> None:
        search = TuningConfig().search
        candidate = candidate_from_normalized([1, -1, 0, 0.9, 0, 0], search, "test")
        self.assertEqual(candidate.mod0_kp, int(search.mod0_kp.max))
        self.assertEqual(candidate.mod0_ki, int(search.mod0_ki.min))
        self.assertEqual(candidate.mod0_kpole1, 6)
        self.assertEqual(candidate.mod0_kpole1, candidate.mod0_kpole2)
        self.assertEqual(candidate.mod0_cm_gain, 2)

        base = candidate_from_normalized([0, 0, 0, -1, 0, 0], search, "base")
        moved = candidate_with_delta(base, np.ones(6), search, "moved", trust_fraction=0.10)
        self.assertLessEqual(moved.mod0_kp - base.mod0_kp, np.ceil((search.mod0_kp.max - search.mod0_kp.min) * 0.10))
        self.assertLessEqual(
            moved.output_inductance_nh - base.output_inductance_nh,
            (search.output_inductance_nh.max - search.output_inductance_nh.min) * 0.10 + 1e-9,
        )

    def test_collection_plan_deduplicates_and_enforces_safety_gate(self) -> None:
        if not dependency_status()["ok"]:
            self.skipTest("Optional ML dependencies are not installed.")
        dataset = synthetic_dataset()
        plan = build_collection_plan(
            dataset,
            TuningConfig(),
            SafePredictor(),
            repeat_count=10,
            local_count=5,
            uncertainty_count=5,
            pool_size=100,
            seed=42,
        )
        fresh = [item for item in plan["candidates"] if item["source"] != "repeat_anchor"]
        keys = [candidate_key(candidate_from_mapping(item["candidate"])) for item in fresh]
        self.assertEqual(len(keys), len(set(keys)))
        with self.assertRaisesRegex(RuntimeError, "0.995 safety gate"):
            build_collection_plan(
                dataset,
                TuningConfig(),
                SafePredictor(safety=0.994),
                repeat_count=10,
                local_count=2,
                uncertainty_count=2,
                pool_size=50,
                seed=43,
            )

    def test_signature_and_default_factory_fail_closed(self) -> None:
        config = TuningConfig()
        first = operating_signature(config, fixed_experiment())
        second = operating_signature(config, replace(fixed_experiment(), board_page=1))
        self.assertFalse(signatures_compatible(first, second))
        with self.assertRaisesRegex(RuntimeError, "no DRL tuner provider"):
            default_candidate_tuner_factory(
                config,
                replace(fixed_experiment(), optimization_algorithm="deep-reinforcement"),
                [],
            )

    def test_corrupt_model_artifact_fails_closed_before_loading(self) -> None:
        if not dependency_status()["ok"]:
            self.skipTest("Optional ML dependencies are not installed.")
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary)
            (artifact / "scalers.npz").write_bytes(b"damaged")
            (artifact / "manifest.json").write_text(
                json.dumps({"model_id": "damaged", "files_sha256": {"scalers.npz": "not-the-file-hash"}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                SurrogateEnsemble.load(artifact, device="cpu")

    def test_collection_tuner_resumes_at_history_offset(self) -> None:
        candidate = HardwarePidCandidate(mod0_kp=150)
        plan = {
            "plan_id": "plan-test",
            "candidates": [
                {"index": index + 1, "candidate": {**candidate.__dict__, "mod0_kp": 150 + index}}
                for index in range(3)
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            tuner = PlannedCandidateTuner(path, history=[mock.Mock()])
            selected = tuner.next_candidate([mock.Mock()], None)
        self.assertEqual(selected.mod0_kp, 151)
        self.assertEqual(tuner.metadata_for(selected)["plan_index"], 2)

    def test_transient_protection_skips_bode_recovers_and_completes_candidate(self) -> None:
        candidate = HardwarePidCandidate(phase="drl_policy")

        class OneCandidateTuner:
            def __init__(self):
                self.used = False

            def next_candidate(self, history, best):
                if self.used:
                    return None
                self.used = True
                return candidate

            def metadata_for(self, value):
                return {"algorithm": "deep-reinforcement", "model_id": "test-model"}

        class ProtectedRunner:
            supports_split_analysis = True

            def __init__(self):
                self.transient_calls = 0
                self.bode_calls = 0
                self.recoveries = 0

            def evaluate(self, value, config, experiment):
                if experiment.enable_transient_analysis:
                    self.transient_calls += 1
                    raise RuntimeError("Scope safety check failed: protected waveform")
                self.bode_calls += 1
                raise AssertionError("Bode must not run after transient protection")

            def recover_after_transient_protection(self, experiment, config):
                self.recoveries += 1
                return {"ok": True, "steps": [{"ok": True, "name": "safe_baseline"}]}

        runner = ProtectedRunner()
        config = TuningConfig(search=replace(SearchSpace(), max_iterations=1, max_coarse_iterations=1, max_refined_iterations=0))
        experiment = replace(fixed_experiment(), optimization_algorithm="deep-reinforcement")
        session = PidAutotuneSession(
            config=config,
            experiment_runner=runner,
            tuner_factory=lambda *_: OneCandidateTuner(),
        )
        status = session.step(config, experiment)
        self.assertEqual(status["state"], "complete")
        self.assertEqual(runner.transient_calls, 1)
        self.assertEqual(runner.bode_calls, 0)
        self.assertEqual(runner.recoveries, 1)
        self.assertEqual(status["history"][0]["optimizer_metadata"]["model_id"], "test-model")

    def test_background_drl_continues_with_next_candidate_after_recovered_trip(self) -> None:
        candidates = [
            HardwarePidCandidate(mod0_kp=150, phase="drl_policy"),
            HardwarePidCandidate(mod0_kp=151, phase="drl_policy"),
        ]

        class TwoCandidateTuner:
            def next_candidate(self, history, best):
                return candidates[len(history)] if len(history) < len(candidates) else None

            def metadata_for(self, value):
                return {"algorithm": "deep-reinforcement", "model_id": "test-model"}

        class RecoveringRunner:
            supports_split_analysis = False

            def __init__(self):
                self.calls = 0
                self.recoveries = 0

            def evaluate(self, value, config, experiment):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("Scope safety check failed: protected waveform")
                return ExperimentResult(
                    waveform=Waveform(time_s=[0.0, 1e-6], vout_v=[0.9, 0.9], input_v=[0.0, 1.0]),
                    metrics=ResponseMetrics(
                        overshoot_pct=0.0,
                        undershoot_pct=0.0,
                        settling_time_s=0.0,
                        oscillations=0,
                        score=1.0,
                        passed=False,
                    ),
                    write_results={"ok": True},
                    bode_result={"ok": True},
                    scope_result={"ok": True},
                )

            def recover_after_transient_protection(self, experiment, config):
                self.recoveries += 1
                return {"ok": True, "steps": [{"ok": True, "name": "power_cycle"}]}

        runner = RecoveringRunner()
        config = TuningConfig(
            search=replace(
                SearchSpace(),
                max_iterations=2,
                max_coarse_iterations=2,
                max_refined_iterations=0,
            )
        )
        experiment = replace(fixed_experiment(), optimization_algorithm="deep-reinforcement")
        session = PidAutotuneSession(
            config=config,
            experiment_runner=runner,
            tuner_factory=lambda *_: TwoCandidateTuner(),
        )
        session.start(config, experiment)
        deadline = time.monotonic() + 2.0
        status = session.status()
        while status["state"] == "running" and time.monotonic() < deadline:
            time.sleep(0.01)
            status = session.status()

        self.assertEqual(status["state"], "complete")
        self.assertEqual(len(status["history"]), 2)
        self.assertEqual(status["history"][0]["metrics"]["score"], 300.0)
        self.assertEqual(status["history"][1]["candidate"]["mod0_kp"], 151)
        self.assertEqual(runner.calls, 2)
        self.assertEqual(runner.recoveries, 1)

    def test_hardware_protection_mode_runs_full_episode_budget_after_pass(self) -> None:
        if not dependency_status()["ok"]:
            self.skipTest("Optional ML dependencies are not installed.")
        ensemble = mock.Mock()
        ensemble.manifest = {"ready": False}
        ensemble.model_id = "test-model"
        ensemble.device = "cpu"
        with mock.patch("stable_baselines3.SAC.load") as load:
            load.return_value = mock.Mock()
            with tempfile.TemporaryDirectory() as temporary:
                policy_path = Path(temporary) / "safe_sac_policy.zip"
                policy_path.write_bytes(b"policy")
                from hardware.tuning.drl.policy import SafeSacTuner

                tuner = SafeSacTuner(
                    ensemble=ensemble,
                    policy_path=policy_path,
                    config=TuningConfig(),
                    episode_budget=2,
                    confirmation_count=1,
                    hardware_protection_mode=True,
                    run_full_budget=True,
                )
        passed_record = mock.Mock()
        passed_record.candidate = HardwarePidCandidate(mod0_kp=150)
        passed_record.metrics = ResponseMetrics(
            overshoot_pct=1.0,
            undershoot_pct=1.0,
            settling_time_s=1e-6,
            oscillations=0,
            score=1.0,
            passed=True,
            overshoot_settling_time_s=1e-6,
            undershoot_settling_time_s=1e-6,
            phase_margin_deg=60.0,
            crossover_frequency_hz=150_000.0,
            gain_margin_db=10.0,
        )
        passed_record.optimizer_metadata = {"algorithm": "deep-reinforcement", "episode": 0}
        with mock.patch.object(tuner, "_observation", return_value=mock.Mock()):
            tuner.policy.predict.return_value = ([0.1] * 6, None)
            tuner.ensemble.predict_features.return_value = {
                "safety_probability": [0.0],
                "validity_probability": [0.0],
                "invalid_probability": [[1.0, 1.0, 1.0]],
                "uncertainty": [999.0],
                "metric_mean": [[1.0, 1.0, 1.0, 1.0, 50.0, 150.0, 10.0]],
                "metric_std": [[1.0] * 7],
            }
            proposed = tuner.next_candidate([passed_record], passed_record)
        self.assertIsNotNone(proposed)
        self.assertEqual(proposed.phase, "drl_policy")


class DrlWorkflowStateMachineTest(unittest.TestCase):
    def test_fake_hardware_collect_train_validate_and_restart_recovery(self) -> None:
        dataset = synthetic_dataset()

        class FakeEnsemble:
            def __init__(self, artifact_dir: Path):
                self.artifact_dir = artifact_dir
                self.artifact_dir.mkdir(parents=True, exist_ok=True)
                self.manifest = {"accepted": True, "acceptance": {"penalty_spearman": 0.9}}
                self.accepted = True

        hardware_status = {"state": "idle", "history": [], "message": "Ready"}

        def start_hardware(config, experiment):
            if experiment.drl_workflow_mode == "collect":
                history = [{} for _ in range(240)]
            else:
                history = []
                for episode in range(4):
                    for step in range(3):
                        history.append(
                            {
                                "optimizer_metadata": {"algorithm": "deep-reinforcement", "episode": episode},
                                "candidate": {"mod0_kp": 150 + episode, "mod0_ki": 200, "mod0_kd": 110},
                                "metrics": {"passed": episode < 3},
                            }
                        )
            hardware_status.update(
                {
                    "state": "complete",
                    "history": history,
                    "message": "Complete",
                    "run": {"run_id": f"fake-{experiment.drl_workflow_mode}", "kind": "recent"},
                }
            )
            return dict(hardware_status)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = DrlWorkflowManager(root / "ml", [root / "runs"])
            manager.bind_session(
                status=lambda: dict(hardware_status),
                stop=lambda: dict(hardware_status),
                start_hardware=start_hardware,
                resume_hardware=lambda run_id, kind: dict(hardware_status),
                persist_hardware=lambda status: status,
            )
            experiment = fixed_experiment()
            config = TuningConfig()

            def fake_train_surrogate(dataset, config, artifact_dir, **kwargs):
                return FakeEnsemble(artifact_dir)

            def fake_plan(*args, **kwargs):
                return {"plan_id": "fake-plan", "budget": 240, "candidates": [{"candidate": {"mod0_kp": 150}}]}

            with mock.patch("hardware.tuning.drl.workflow.load_autotune_dataset", return_value=(dataset, {"dataset_id": "fake-data"})), mock.patch(
                "hardware.tuning.drl.workflow.train_surrogate_ensemble", side_effect=fake_train_surrogate
            ), mock.patch("hardware.tuning.drl.workflow.build_collection_plan", side_effect=fake_plan), mock.patch(
                "hardware.tuning.drl.workflow.validation_start_candidates",
                return_value=[HardwarePidCandidate(mod0_kp=150 + index) for index in range(4)],
            ), mock.patch(
                "hardware.tuning.drl.workflow.train_safe_sac_policy",
                return_value={"accepted": True, "ready": True, "policy_accepted": True},
            ):
                manager.start_collection(config, experiment)
                manager._worker.join(timeout=5)
                self.assertEqual(manager.status()["state"], "collection_complete")
                self.assertEqual(manager.status()["collection_completed"], 240)

                manager.start_training(config, experiment)
                manager._worker.join(timeout=5)
                trained = manager.status()
                self.assertEqual(trained["state"], "ready_for_validation")
                self.assertTrue(trained["model_compatible"])

                manager.start_validation(config, experiment)
                manager._worker.join(timeout=5)
                validated = manager.status()
                self.assertEqual(validated["state"], "hardware_ready")
                self.assertEqual(validated["validation_result"]["episodes_succeeded"], 3)

            manifest_path = root / "ml" / "workflow_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update({"state": "collecting", "workflow": "collection", "run_id": "fake-collect"})
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            restored = DrlWorkflowManager(root / "ml", [root / "runs"])
            self.assertEqual(restored.status()["state"], "paused")
            self.assertTrue(restored.status()["resume_available"])


if __name__ == "__main__":
    unittest.main()
