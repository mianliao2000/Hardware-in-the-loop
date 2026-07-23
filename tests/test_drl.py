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
    ACTION_FIELDS,
    KPOLE_PAIRS,
    METRIC_FIELDS,
    candidate_to_action,
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
from hardware.tuning.drl.dataset import (
    DrlDataset,
    build_bootstrap_collection_plan,
    build_collection_plan,
    build_targeted_collection_plan,
    load_autotune_dataset,
    _apply_raw_key_robust_targets,
)
from hardware.tuning.drl.model import SurrogateEnsemble, dependency_status
from hardware.tuning.drl.tuner import PlannedCandidateTuner
from hardware.tuning.drl.workflow import (
    DrlWorkflowManager,
    _collection_signature_covers,
    _hardware_episode_budget,
)
from hardware.tuning.models import (
    AutotuneExperimentConfig,
    ExperimentResult,
    HardwarePidCandidate,
    IterationRecord,
    PidParameters,
    ResponseMetrics,
    SearchSpace,
    TuningConfig,
    Waveform,
    output_inductance_from_raw,
)
from hardware.tuning.runner import PidAutotuneSession, default_candidate_tuner_factory


def fixed_experiment(**updates) -> AutotuneExperimentConfig:
    experiment = AutotuneExperimentConfig(
        enable_transient_analysis=True,
        enable_bode_analysis=True,
        function_generator_config={
            "mode": "square",
            "frequency_hz": 10_000,
            "low_v": 0.1,
            "high_v": 1.1,
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
    baseline = np.tile(np.asarray([1.0, 1.0, 1.5, 1.5, 60.0, 150.0, 10.0, 0.0], dtype=np.float32), (size, 1))
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
            "metric_mean": np.tile(np.asarray([1, 1, 1.5, 1.5, 60, 150, 10, 0], dtype=float), (count, 1)),
            "metric_std": np.tile(np.asarray([0.1] * len(METRIC_FIELDS), dtype=float), (count, 1)),
            "safety_probability": np.full(count, self.safety, dtype=float),
            "uncertainty": np.linspace(0.0, 1.0, count, dtype=float),
        }


class DrlCoreTest(unittest.TestCase):
    def test_relabeled_score_accepts_canonical_surrogate_units(self) -> None:
        from hardware.tuning.drl.common import relabeled_score

        score, passed = relabeled_score(
            {
                "overshoot_pct": 0.7,
                "undershoot_pct": 0.8,
                "overshoot_settling_time_us": 1.91,
                "undershoot_settling_time_us": 1.91,
                "phase_margin_deg": 85.0,
                "crossover_frequency_khz": 128.0,
                "gain_margin_db": 10.0,
            },
            TuningConfig().targets,
        )

        self.assertTrue(passed)
        self.assertLess(score, 0.0)

    def test_relabeled_score_blocks_measured_bode_shape_failure(self) -> None:
        score, passed = relabeled_score(
            {
                "metrics": {
                    "overshoot_pct": 0.7,
                    "undershoot_pct": 0.8,
                    "overshoot_settling_time_s": 1.0e-6,
                    "undershoot_settling_time_s": 1.0e-6,
                    "phase_margin_deg": 85.0,
                    "crossover_frequency_hz": 128_000.0,
                    "gain_margin_db": 10.0,
                    "bode_gain_shape_penalty": 60.0,
                    "pass_reasons": ["bode gain shape failed: gain rebound 5.0 dB"],
                }
            },
            TuningConfig().targets,
        )

        self.assertFalse(passed)
        self.assertGreater(score, 50.0)

    def test_relabeled_score_uses_shape_penalty_proxy_for_synthetic_rollouts(self) -> None:
        payload = {
            "overshoot_pct": 0.7,
            "undershoot_pct": 0.8,
            "overshoot_settling_time_us": 1.0,
            "undershoot_settling_time_us": 1.0,
            "phase_margin_deg": 85.0,
            "crossover_frequency_khz": 128.0,
            "gain_margin_db": 10.0,
            "bode_gain_shape_penalty": 30.0,
        }

        _, passed = relabeled_score(payload, TuningConfig().targets)

        self.assertFalse(passed)

    def test_raw_key_repeats_produce_conservative_probability_and_p90_ts(self) -> None:
        candidate = HardwarePidCandidate(
            mod0_kp=135,
            mod0_ki=243,
            mod0_kd=170,
            mod0_kpole1=2,
            mod0_kpole2=2,
            mod0_cm_gain=2,
            mod0_ll_bw=75,
            output_inductance_nh=93.586,
            effective_lc_inductance_nh=382.118,
        )
        settling = [1.90, 1.91, 1.92, 4.50, 5.00]
        samples = []
        for index, settling_us in enumerate(settling):
            samples.append(
                {
                    "candidate": candidate,
                    "metrics": np.asarray(
                        [0.7, 0.8, settling_us, settling_us, 85.0, 125.0, 10.0, 0.0],
                        dtype=np.float32,
                    ),
                    "metric_mask": np.ones(len(METRIC_FIELDS), dtype=np.float32),
                    "invalid_labels": np.zeros(3, dtype=np.float32),
                    "score": 0.0,
                    "passed": index < 3,
                    "record": {},
                }
            )

        manifest = _apply_raw_key_robust_targets(samples, TuningConfig())

        self.assertEqual(manifest["candidate_group_count"], 1)
        self.assertEqual(manifest["repeat_group_count"], 1)
        self.assertAlmostEqual(samples[0]["passed"], 3.5 / 6.0)
        self.assertAlmostEqual(samples[0]["invalid_labels"][3], 3.5 / 6.0)
        self.assertAlmostEqual(samples[0]["metrics"][2], np.quantile(settling, 0.9), places=5)
        for sample in samples[1:]:
            np.testing.assert_allclose(sample["metrics"], samples[0]["metrics"])
            np.testing.assert_allclose(sample["invalid_labels"], samples[0]["invalid_labels"])

    def test_latest_consecutive_three_passes_replace_old_raw_key_failures(self) -> None:
        candidate = HardwarePidCandidate(mod0_kp=130, mod0_ki=243, mod0_kd=170, mod0_ll_bw=75)
        samples = []
        settling = [6.0, 5.0, 1.91, 1.92, 1.90]
        for index, settling_us in enumerate(settling, start=1):
            passed = index >= 3
            samples.append(
                {
                    "candidate": candidate,
                    "metrics": np.asarray(
                        [0.7, 0.8, settling_us, settling_us, 85.0, 125.0, 10.0, 0.0],
                        dtype=np.float32,
                    ),
                    "metric_mask": np.ones(len(METRIC_FIELDS), dtype=np.float32),
                    "invalid_labels": np.zeros(3, dtype=np.float32),
                    "score": 0.0,
                    "passed": passed,
                    "group": "recent:run-a",
                    "record": {"iteration": index},
                }
            )

        manifest = _apply_raw_key_robust_targets(samples, TuningConfig())

        self.assertEqual(manifest["confirmed_group_count"], 1)
        self.assertAlmostEqual(samples[0]["passed"], 3.5 / 4.0)
        self.assertLess(samples[0]["metrics"][2], 2.0)
        self.assertTrue(samples[0]["record"]["drl_robust_target"]["confirmed_streak_target"])

    def test_full_budget_hardware_run_extends_policy_episode_to_outer_budget(self) -> None:
        config = TuningConfig(
            search=replace(
                SearchSpace(),
                max_iterations=500,
                max_coarse_iterations=500,
                max_refined_iterations=0,
            )
        )
        full_run = fixed_experiment(
            drl_episode_budget=15,
            ignore_pass_until_max_iterations=True,
        )
        self.assertEqual(
            _hardware_episode_budget(config, full_run, is_validation=False),
            500,
        )
        self.assertEqual(
            _hardware_episode_budget(config, full_run, is_validation=True),
            15,
        )
        short_run = replace(full_run, ignore_pass_until_max_iterations=False)
        self.assertEqual(
            _hardware_episode_budget(config, short_run, is_validation=False),
            15,
        )

    def test_saved_dataset_excludes_actions_outside_current_search_space(self) -> None:
        config = TuningConfig()
        inside = HardwarePidCandidate(mod0_kp=int(config.search.mod0_kp.min))
        outside = HardwarePidCandidate(mod0_kp=int(config.search.mod0_kp.min) - 1)
        metrics = {
            "overshoot_pct": 1.0,
            "undershoot_pct": 1.0,
            "overshoot_settling_time_s": 1e-6,
            "undershoot_settling_time_s": 1e-6,
            "settling_analysis_version": 15,
        }
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "saved" / "run-1"
            run.mkdir(parents=True)
            rows = [
                {"candidate": inside.__dict__, "metrics": metrics},
                {
                    "candidate": inside.__dict__,
                    "metrics": {**metrics, "settling_analysis_version": 1},
                },
                {"candidate": outside.__dict__, "metrics": metrics},
                {
                    "candidate": {key: value for key, value in inside.__dict__.items() if key != "mod0_ll_bw"},
                    "metrics": metrics,
                },
            ]
            (run / "iterations.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows),
                encoding="utf-8",
            )
            (run / "run_status.json").write_text(json.dumps({"history": rows}), encoding="utf-8")
            dataset, manifest = load_autotune_dataset([run.parent], config)
        self.assertEqual(dataset.size, 1)
        self.assertEqual(manifest["excluded_out_of_search_space_count"], 1)
        self.assertEqual(manifest["excluded_missing_bandwidth_count"], 1)
        self.assertEqual(manifest["excluded_legacy_settling_count"], 1)

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

    def test_relabelled_reward_has_no_hard_minimum(self) -> None:
        config = TuningConfig()
        passing = {
            "overshoot_pct": 0.0,
            "undershoot_pct": 0.0,
            "overshoot_settling_time_s": 0.0,
            "undershoot_settling_time_s": 0.0,
            "phase_margin_deg": 90.0,
            "crossover_frequency_hz": 100_000.0,
            "gain_margin_db": 20.0,
        }

        score, passed = relabeled_score({"metrics": passing}, config.targets)

        self.assertTrue(passed)
        self.assertLess(score, -3.0)

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

    def test_relabelled_penalty_is_capped_at_three_hundred(self) -> None:
        metrics = {
            "overshoot_pct": 100.0,
            "undershoot_pct": 100.0,
            "overshoot_settling_time_s": 100e-6,
            "undershoot_settling_time_s": 100e-6,
            "phase_margin_deg": -100.0,
            "crossover_frequency_hz": 1_000_000.0,
            "gain_margin_db": -20.0,
        }
        score, passed = relabeled_score({"metrics": metrics}, TuningConfig().targets)
        self.assertEqual(score, 300.0)
        self.assertFalse(passed)

    def test_action_quantization_and_trust_region(self) -> None:
        search = TuningConfig().search
        candidate = candidate_from_normalized([1, -1, 0, 1, 0.5, 1, 0, 0, 0], search, "test")
        self.assertEqual(candidate.mod0_kp, int(search.mod0_kp.max))
        self.assertEqual(candidate.mod0_ki, int(search.mod0_ki.min))
        self.assertEqual(candidate.mod0_kpole1, 6)
        self.assertEqual(candidate.mod0_kpole2, 5)
        self.assertEqual(candidate.mod0_cm_gain, 9)

        base = candidate_from_normalized([0, 0, 0, -1, -1, 0, 0, 0, 0], search, "base")
        moved = candidate_with_delta(base, np.ones(len(ACTION_FIELDS)), search, "moved", trust_fraction=0.10)
        self.assertLessEqual(moved.mod0_kp - base.mod0_kp, np.ceil((search.mod0_kp.max - search.mod0_kp.min) * 0.10))
        self.assertLessEqual(
            moved.output_inductance_nh - base.output_inductance_nh,
            (search.output_inductance_nh.max - search.output_inductance_nh.min) * 0.10 + 1e-9,
        )

    def test_all_kpole_pairs_round_trip_independently(self) -> None:
        search = TuningConfig().search
        for kpole1, kpole2 in KPOLE_PAIRS:
            candidate = HardwarePidCandidate(mod0_kpole1=kpole1, mod0_kpole2=kpole2)
            normalized = candidate_to_normalized(candidate, search)
            restored = candidate_from_normalized(normalized, search, "round_trip")
            self.assertEqual((restored.mod0_kpole1, restored.mod0_kpole2), (kpole1, kpole2))
            self.assertEqual(int(candidate_to_action(restored)[3]), kpole1)
            self.assertEqual(int(candidate_to_action(restored)[4]), kpole2)

    def test_current_mode_gain_covers_every_integer_from_zero_through_nine(self) -> None:
        search = TuningConfig().search
        self.assertEqual(
            (search.mod0_cm_gain.min, search.mod0_cm_gain.max, search.mod0_cm_gain.points),
            (0, 9, 10),
        )
        restored = {
            candidate_from_normalized(
                [0, 0, 0, 0, 0, normalized, 0, 0, 0],
                search,
                "cm_gain_round_trip",
            ).mod0_cm_gain
            for normalized in np.linspace(-1.0, 1.0, 10)
        }
        self.assertEqual(restored, set(range(10)))

    def test_shared_loop_a_bandwidth_covers_search_range(self) -> None:
        search = TuningConfig().search
        self.assertEqual((search.mod0_ll_bw.min, search.mod0_ll_bw.max), (47, 79))
        low = candidate_from_normalized([0, 0, 0, 0, 0, 0, -1, 0, 0], search, "bw_low")
        high = candidate_from_normalized([0, 0, 0, 0, 0, 0, 1, 0, 0], search, "bw_high")
        self.assertEqual(low.mod0_ll_bw, 47)
        self.assertEqual(high.mod0_ll_bw, 79)

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

    def test_candidate_key_uses_hardware_inductance_raw_code(self) -> None:
        first = HardwarePidCandidate(output_inductance_nh=output_inductance_from_raw(1200))
        second = replace(first, output_inductance_nh=first.output_inductance_nh + 1e-5)
        self.assertNotEqual(first.output_inductance_nh, second.output_inductance_nh)
        self.assertEqual(candidate_key(first), candidate_key(second))

    def test_zero_history_bootstrap_plan_covers_independent_nine_dimensional_actions(self) -> None:
        if not dependency_status()["ok"]:
            self.skipTest("Optional ML dependencies are not installed.")
        plan = build_bootstrap_collection_plan(
            TuningConfig(),
            repeat_count=4,
            global_count=25,
            local_count=4,
            seed=44,
        )
        self.assertTrue(plan["bootstrap"])
        self.assertEqual(plan["budget"], 33)
        self.assertEqual(plan["allocation"]["repeat"], 4)
        self.assertEqual(plan["candidates"][0]["candidate"]["phase"], "baseline")
        global_candidates = [
            candidate_from_mapping(item["candidate"])
            for item in plan["candidates"]
            if item["source"] == "global_9d_sobol"
        ]
        self.assertEqual(
            {(candidate.mod0_kpole1, candidate.mod0_kpole2) for candidate in global_candidates},
            set(KPOLE_PAIRS),
        )
        self.assertTrue(all(2 <= candidate.mod0_kpole1 <= 6 for candidate in global_candidates))
        self.assertTrue(all(2 <= candidate.mod0_kpole2 <= 6 for candidate in global_candidates))
        self.assertEqual(
            len({candidate_key(candidate) for candidate in global_candidates}),
            len(global_candidates),
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

    def test_completed_collection_can_train_a_strictly_narrower_bandwidth_range(self) -> None:
        broad = TuningConfig()
        broad = replace(
            broad,
            search=replace(
                broad.search,
                mod0_ll_bw=replace(broad.search.mod0_ll_bw, center=74, min=47, max=127),
            ),
        )
        narrow = TuningConfig()
        self.assertTrue(
            _collection_signature_covers(
                operating_signature(broad, fixed_experiment()),
                operating_signature(narrow, fixed_experiment()),
            )
        )
        self.assertFalse(
            _collection_signature_covers(
                operating_signature(narrow, fixed_experiment()),
                operating_signature(broad, fixed_experiment()),
            )
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

    def test_targeted_collection_confirms_pass_before_bandwidth_climb(self) -> None:
        anchor = HardwarePidCandidate(mod0_ll_bw=66, phase="drl_targeted_near_pass_repeat")
        plan = {
            "plan_id": "targeted-test",
            "dynamic_confirmation": True,
            "confirmation_count": 3,
            "bandwidth_climb_after_confirmation": True,
            "candidates": [
                {"index": 1, "candidate": anchor.__dict__, "optimizer_metadata": {"proposal_source": "near_pass_repeat"}},
                {"index": 2, "candidate": replace(anchor, mod0_kp=anchor.mod0_kp + 5).__dict__},
            ],
        }

        def passed(iteration: int) -> mock.Mock:
            record = mock.Mock()
            record.iteration = iteration
            record.candidate = anchor
            record.metrics = ResponseMetrics(
                overshoot_pct=1.0,
                undershoot_pct=1.0,
                settling_time_s=1e-6,
                oscillations=0,
                score=0.0,
                passed=True,
                overshoot_settling_time_s=1e-6,
                undershoot_settling_time_s=1e-6,
                phase_margin_deg=60.0,
                crossover_frequency_hz=150_000.0,
                gain_margin_db=10.0,
            )
            record.objective_score = 0.0
            record.bandwidth_bonus = 0.0
            record.optimizer_metadata = {"collection_plan_id": "targeted-test", "plan_index": 1}
            return record

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            tuner = PlannedCandidateTuner(path, search=TuningConfig().search)
            first = tuner.next_candidate([], None)
            self.assertEqual(first.mod0_ll_bw, 66)
            confirmation_two = tuner.next_candidate([passed(1)], None)
            self.assertEqual(confirmation_two.phase, "drl_targeted_confirm")
            confirmation_three = tuner.next_candidate([passed(1), passed(2)], None)
            self.assertEqual(confirmation_three.phase, "drl_targeted_confirm")
            climb = tuner.next_candidate([passed(1), passed(2), passed(3)], None)
            self.assertEqual(climb.phase, "drl_targeted_bandwidth_climb")
            self.assertGreater(climb.mod0_ll_bw, anchor.mod0_ll_bw)

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
        with mock.patch.object(tuner, "_observation", return_value=np.zeros(37, dtype=np.float32)):
            tuner.policy.predict.return_value = ([0.1] * len(ACTION_FIELDS), None)
            tuner.ensemble.predict_features.return_value = {
                "safety_probability": [0.0],
                "validity_probability": [0.0],
                "invalid_probability": [[1.0, 1.0, 1.0]],
                "uncertainty": [999.0],
                "metric_mean": [[1.0, 1.0, 1.0, 1.0, 50.0, 150.0, 10.0, 0.0]],
                "metric_std": [[1.0] * len(METRIC_FIELDS)],
            }
            proposed = tuner.next_candidate([passed_record], passed_record)
        self.assertIsNotNone(proposed)
        self.assertEqual(proposed.phase, "drl_policy")

    def test_drl_exploration_schedule_expands_and_restarts(self) -> None:
        from hardware.tuning.drl.policy import _exploration_schedule, _stagnation_count

        self.assertEqual(_exploration_schedule(7, 7), (0.20, ""))
        self.assertEqual(_exploration_schedule(8, 8), (0.35, ""))
        self.assertEqual(_exploration_schedule(13, 12), (0.50, "stagnation_sobol_restart"))
        self.assertEqual(_exploration_schedule(10, 4), (0.20, "periodic_global_exploration"))
        self.assertEqual(_exploration_schedule(15, 12), (0.50, "stagnation_sobol_restart"))
        self.assertEqual(_exploration_schedule(11, 15, best_score=5.0), (0.10, ""))
        self.assertEqual(_exploration_schedule(12, 15, best_score=5.0), (0.40, ""))
        self.assertEqual(
            _exploration_schedule(15, 15, best_score=5.0),
            (0.40, ""),
        )
        self.assertEqual(_exploration_schedule(5, 0, best_score=5.0, confirmed_basins=1), (0.10, ""))
        self.assertEqual(
            _exploration_schedule(25, 0, best_score=5.0, confirmed_basins=1),
            (0.10, "periodic_global_exploration"),
        )
        self.assertEqual(_exploration_schedule(10, 0, best_score=5.0, confirmed_basins=3), (0.10, ""))
        self.assertEqual(
            _exploration_schedule(50, 0, best_score=5.0, confirmed_basins=3),
            (0.10, "periodic_global_exploration"),
        )

        def record(score: float, source: str = "safe_sac"):
            value = mock.Mock()
            value.metrics = mock.Mock(score=score, pass_reasons=[])
            value.optimizer_metadata = {"proposal_source": source}
            return value

        before_restart = [record(10.0), record(9.0), record(9.0), record(9.0)]
        self.assertEqual(_stagnation_count(before_restart), 2)
        after_restart = before_restart + [record(8.0, "stagnation_sobol_restart")]
        self.assertEqual(_stagnation_count(after_restart), 0)
        after_restart.extend([record(7.0), record(7.0), record(7.0)])
        self.assertEqual(_stagnation_count(after_restart), 2)

    def test_kpole_diversity_prefers_the_least_measured_pair(self) -> None:
        from hardware.tuning.drl.policy import _least_used_kpole_pair

        records = []
        for pair in KPOLE_PAIRS[:3]:
            repeats = 5 if pair == (3, 3) else 1
            for _ in range(repeats):
                record = mock.Mock()
                record.candidate = HardwarePidCandidate(mod0_kpole1=pair[0], mod0_kpole2=pair[1])
                records.append(record)

        self.assertEqual(_least_used_kpole_pair(records), KPOLE_PAIRS[3])

    def test_hardware_neighbor_score_overrides_bad_local_surrogate_ranking(self) -> None:
        from hardware.tuning.drl.policy import _calibrated_hardware_score

        config = TuningConfig()

        def record(candidate: HardwarePidCandidate, score: float) -> IterationRecord:
            return IterationRecord(
                iteration=1,
                phase="drl_policy",
                wc_rad_s=0.0,
                phi_deg=0.0,
                pid=PidParameters(kp=0.0, ki=0.0, kd=0.0, kf=0.0),
                metrics=ResponseMetrics(
                    overshoot_pct=1.0,
                    undershoot_pct=1.0,
                    settling_time_s=1e-6,
                    oscillations=0,
                    score=score,
                    passed=score <= 0.0,
                ),
                waveform=Waveform(time_s=[], vout_v=[]),
                timestamp=0.0,
                candidate=candidate,
            )

        good = HardwarePidCandidate(mod0_kp=155, mod0_ki=204, mod0_kd=149)
        bad = HardwarePidCandidate(mod0_kp=220, mod0_ki=240, mod0_kd=190)
        near_good = HardwarePidCandidate(mod0_kp=156, mod0_ki=204, mod0_kd=149)
        near_bad = HardwarePidCandidate(mod0_kp=219, mod0_ki=240, mod0_kd=190)
        history = [record(good, 2.0), record(bad, 90.0)]

        good_rank, good_empirical, _ = _calibrated_hardware_score(near_good, 80.0, history, config)
        bad_rank, bad_empirical, _ = _calibrated_hardware_score(near_bad, 75.0, history, config)

        self.assertLess(good_rank, bad_rank)
        self.assertLess(good_empirical, bad_empirical)

    def test_failed_bandwidth_climb_creates_fixed_bw_local_repairs_until_resolved(self) -> None:
        from hardware.tuning.drl.policy import (
            BOUNDARY_REPAIR_ATTEMPT_LIMIT,
            _bandwidth_repair_context,
            _boundary_repair_attempt_count,
            _boundary_repair_candidates,
        )

        config = TuningConfig()
        anchor_candidate = HardwarePidCandidate(
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

        def record(
            iteration: int,
            candidate: HardwarePidCandidate,
            passed: bool,
            source: str,
        ) -> IterationRecord:
            return IterationRecord(
                iteration=iteration,
                phase=candidate.phase,
                wc_rad_s=0.0,
                phi_deg=0.0,
                pid=PidParameters(kp=0.0, ki=0.0, kd=0.0, kf=0.0),
                metrics=ResponseMetrics(
                    overshoot_pct=0.6,
                    undershoot_pct=0.9,
                    settling_time_s=1.91e-6 if passed else 2.87e-6,
                    overshoot_settling_time_s=1.91e-6,
                    undershoot_settling_time_s=1.91e-6 if passed else 2.87e-6,
                    oscillations=0,
                    score=-2.3 if passed else 8.7,
                    passed=passed,
                ),
                waveform=Waveform(time_s=[], vout_v=[]),
                timestamp=float(iteration),
                candidate=candidate,
                objective_score=-11.4 if passed else 8.7,
                bandwidth_bonus=9.0 if passed else 0.0,
                optimizer_metadata={"proposal_source": source},
            )

        records = [
            record(1, anchor_candidate, True, "episode_start"),
            record(2, anchor_candidate, True, "pass_confirmation"),
            record(3, anchor_candidate, True, "pass_confirmation"),
        ]
        failed_candidate = replace(anchor_candidate, mod0_ll_bw=77, phase="bandwidth_climb")
        failed = record(4, failed_candidate, False, "bandwidth_climb")
        records.append(failed)

        context = _bandwidth_repair_context(records, 3)
        self.assertIsNotNone(context)
        self.assertIs(context[0], failed)
        self.assertEqual(context[1].candidate, anchor_candidate)
        repairs = _boundary_repair_candidates(
            anchor_candidate,
            77,
            config,
            {candidate_key(item.candidate) for item in records if item.candidate is not None},
        )
        self.assertGreaterEqual(len(repairs), 15)
        self.assertTrue(all(candidate.mod0_ll_bw == 77 for candidate, _ in repairs))
        self.assertTrue(all(candidate_key(candidate) != candidate_key(failed_candidate) for candidate, _ in repairs))
        self.assertTrue(
            all(
                config.search.mod0_kpole1.min
                <= candidate.mod0_kpole1
                <= config.search.mod0_kpole1.max
                and config.search.mod0_kpole2.min
                <= candidate.mod0_kpole2
                <= config.search.mod0_kpole2.max
                for candidate, _ in repairs
            )
        )
        self.assertFalse(
            any(
                metadata["field"] in {"mod0_kpole1", "mod0_kpole2"}
                and metadata["direction"] < 0
                for _, metadata in repairs
            )
        )

        attempted_records = list(records)
        for offset, (candidate, _) in enumerate(
            repairs[:BOUNDARY_REPAIR_ATTEMPT_LIMIT],
            start=5,
        ):
            attempted_records.append(record(offset, candidate, False, "boundary_repair"))
        self.assertEqual(
            _boundary_repair_attempt_count(attempted_records, failed),
            BOUNDARY_REPAIR_ATTEMPT_LIMIT,
        )

        repaired = repairs[0][0]
        records.extend(
            [
                record(5, repaired, True, "boundary_repair"),
                record(6, repaired, True, "pass_confirmation"),
                record(7, repaired, True, "pass_confirmation"),
            ]
        )
        self.assertIsNone(_bandwidth_repair_context(records, 3))

    def test_normal_drl_starts_at_highest_confirmed_bandwidth(self) -> None:
        from hardware.tuning.drl.policy import _ordered_hardware_starts

        starts = [
            HardwarePidCandidate(mod0_kp=142, mod0_ll_bw=66),
            HardwarePidCandidate(mod0_kp=135, mod0_ll_bw=76),
            HardwarePidCandidate(mod0_kp=126, mod0_ll_bw=74),
        ]

        self.assertEqual(
            [candidate.mod0_ll_bw for candidate in _ordered_hardware_starts(starts, is_validation=False)],
            [76, 74, 66],
        )
        self.assertEqual(
            _ordered_hardware_starts(starts, is_validation=True),
            starts,
        )

    def test_global_exploration_enters_replay_with_low_weight(self) -> None:
        from hardware.tuning.drl.policy import SafeSacTuner

        tuner = object.__new__(SafeSacTuner)
        tuner.ensemble = mock.Mock(model_id="test-model")
        tuner._last_metadata = {}
        tuner._pending_transitions = {}
        candidate = HardwarePidCandidate(mod0_kp=255, mod0_kpole1=4, mod0_kpole2=6)
        tuner._remember(
            candidate,
            0,
            20,
            "periodic_global_exploration",
            None,
            observation=np.zeros(37, dtype=np.float32),
            action=np.ones(len(ACTION_FIELDS), dtype=np.float32),
            previous_best_score=10.0,
            replay_eligible=True,
            replay_weight=0.25,
        )
        pending = tuner._pending_transitions[candidate_key(candidate)]
        self.assertTrue(pending["replay_eligible"])
        self.assertEqual(pending["replay_weight"], 0.25)
        self.assertEqual(pending["proposal_source"], "periodic_global_exploration")

    def test_online_policy_persistence_uses_an_actual_zip_temp_path(self) -> None:
        from hardware.tuning.drl.policy import SafeSacTuner

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tuner = object.__new__(SafeSacTuner)
            tuner._online_dir = root / "online"
            tuner._online_dir.mkdir()
            tuner._online_policy_path = root / "safe_sac_online_latest.zip"
            tuner._online_replay_path = tuner._online_dir / "replay_buffer.pkl"
            tuner.policy = mock.Mock()

            def save_policy(path: Path) -> None:
                self.assertTrue(str(path).endswith(".tmp.zip"))
                Path(path).write_bytes(b"updated-policy")

            def save_replay(path: Path) -> None:
                Path(path).write_bytes(b"updated-replay")

            tuner.policy.save.side_effect = save_policy
            tuner.policy.save_replay_buffer.side_effect = save_replay
            tuner._persist_online_artifacts()

            self.assertEqual(tuner._online_policy_path.read_bytes(), b"updated-policy")
            self.assertEqual(tuner._online_replay_path.read_bytes(), b"updated-replay")


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
