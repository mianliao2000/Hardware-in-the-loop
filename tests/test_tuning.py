from __future__ import annotations

import math
from pathlib import Path
import time
import unittest

import numpy as np

from hardware.tuning import (
    CompensatorDesign,
    ExperimentResult,
    HardwarePidCandidate,
    HardwareGridHeuristicTuner,
    IterationRecord,
    PidAutotuneSession,
    PidParameters,
    PlantParams,
    ResponseAnalyzer,
    ResponseMetrics,
    SearchParameter,
    SearchSpace,
    StubPidProgrammer,
    TuningConfig,
    TuningTargets,
    Waveform,
)
from hardware.tuning.search import hardware_candidate_key, select_best_result, select_diverse_results
from hardware.tuning.analyzer import _passed_reward, score_metrics


class TuningFrameworkTest(unittest.TestCase):
    def test_settling_penalty_and_reward_use_ten_per_microsecond(self) -> None:
        targets = TuningTargets(settling_time_s=2e-6)
        penalty = score_metrics(3.0, 3.0, 0, 3e-6, 4e-6, targets)
        self.assertAlmostEqual(penalty, 30.0)

        transient = ResponseMetrics(
            overshoot_pct=3.0,
            undershoot_pct=3.0,
            settling_time_s=1.95e-6,
            oscillations=0,
            score=0.0,
            passed=True,
            overshoot_settling_time_s=1.95e-6,
            undershoot_settling_time_s=1.95e-6,
        )
        reward = _passed_reward(targets, transient, None, None, None, True, False)
        self.assertAlmostEqual(reward, 1.0)

    def test_penalty_is_capped_at_three_hundred(self) -> None:
        targets = TuningTargets(settling_time_s=2e-6)
        penalty = score_metrics(100.0, 100.0, 50, 100e-6, 100e-6, targets)
        self.assertEqual(penalty, 300.0)

    def test_passing_reward_has_no_hard_minimum(self) -> None:
        targets = TuningTargets(settling_time_s=2e-6)
        analyzer = ResponseAnalyzer(targets)
        transient = ResponseMetrics(
            overshoot_pct=0.0,
            undershoot_pct=0.0,
            settling_time_s=0.0,
            oscillations=0,
            score=0.0,
            passed=True,
            overshoot_settling_time_s=0.0,
            undershoot_settling_time_s=0.0,
        )
        analyzer.analyze = lambda waveform: transient

        metrics = analyzer.analyze_hardware(
            Waveform(time_s=[], vout_v=[]),
            {"phase_margin_deg": 90.0, "phase_crossover_hz": 100_000.0},
        )

        self.assertTrue(metrics.passed)
        self.assertLess(metrics.score, -3.0)

    def test_bode_gain_shape_failure_adds_penalty_and_blocks_pass(self) -> None:
        analyzer = ResponseAnalyzer(TuningTargets())
        transient = ResponseMetrics(
            overshoot_pct=0.5,
            undershoot_pct=0.5,
            settling_time_s=1e-6,
            oscillations=0,
            score=0.0,
            passed=True,
            overshoot_settling_time_s=1e-6,
            undershoot_settling_time_s=1e-6,
        )

        metrics = analyzer.analyze_hardware(
            None,
            {
                "phase_margin_deg": 80.0,
                "phase_crossover_hz": 120_000.0,
                "gain_shape_valid": False,
                "gain_rebound_db": 5.0,
                "gain_flat_span_decades": 0.38,
                "gain_shape_penalty": 60.0,
            },
            precomputed_transient=transient,
        )

        self.assertFalse(metrics.passed)
        self.assertEqual(metrics.score, 60.0)
        self.assertEqual(metrics.bode_gain_shape_penalty, 60.0)
        self.assertTrue(any("bode gain shape failed" in reason for reason in metrics.pass_reasons))

    def test_compensator_generates_positive_pid_terms(self) -> None:
        pid = CompensatorDesign(PlantParams()).compute(157_080.0, 60.0)

        self.assertGreater(pid.kp, 0.0)
        self.assertGreater(pid.ki, 0.0)
        self.assertGreaterEqual(pid.kd, 0.0)
        self.assertGreater(pid.kf, 0.0)

    def test_session_step_records_iteration_without_hardware_pid_write(self) -> None:
        programmer = StubPidProgrammer()
        config = TuningConfig(search=SearchSpace(max_iterations=3))
        session = PidAutotuneSession(config=config, pid_programmer=programmer)

        status = session.step()

        self.assertEqual(status["state"], "stopped")
        self.assertEqual(len(status["history"]), 1)
        self.assertEqual(status["pid_programming"]["write_attempts"], 0)
        self.assertIsNotNone(status["current"]["pid"])

    def test_session_completes_at_max_iterations(self) -> None:
        session = PidAutotuneSession(config=TuningConfig(search=SearchSpace(max_iterations=1)))

        first = session.step()
        second = session.step()

        self.assertEqual(len(first["history"]), 1)
        self.assertEqual(second["state"], "complete")
        self.assertEqual(len(second["history"]), 1)

    def test_hardware_search_clamps_center_candidate(self) -> None:
        search = SearchSpace(
            max_iterations=4,
            mod0_kp=SearchParameter(center=300, min=0, max=255, step=8),
            mod0_kpole1=SearchParameter(center=-3, min=0, max=15, step=1),
        )
        tuner = HardwareGridHeuristicTuner(search)

        candidate = tuner.next_candidate([], None)

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.mod0_kp, 255)
        self.assertEqual(candidate.mod0_kpole1, 2)

    def test_hardware_search_does_not_repeat_baseline(self) -> None:
        search = SearchSpace(max_iterations=10)
        tuner = HardwareGridHeuristicTuner(search)
        seen = set()
        history = []

        for _ in range(8):
            candidate = tuner.next_candidate(history, None)
            self.assertIsNotNone(candidate)
            key = (
                candidate.mod0_kp,
                candidate.mod0_ki,
                candidate.mod0_kd,
                candidate.mod0_kpole1,
                candidate.mod0_kpole2,
                candidate.output_inductance_nh,
                candidate.effective_lc_inductance_nh,
            )
            self.assertNotIn(key, seen)
            seen.add(key)

    def test_hardware_search_continues_local_refine_to_max_iterations(self) -> None:
        search = SearchSpace(max_iterations=100, max_coarse_iterations=80, max_refined_iterations=20)
        tuner = HardwareGridHeuristicTuner(search)
        history: list[IterationRecord] = []
        best: IterationRecord | None = None

        for iteration in range(1, search.max_iterations + 1):
            candidate = tuner.next_candidate(history, best)
            self.assertIsNotNone(candidate)
            metrics = ResponseMetrics(
                overshoot_pct=0.0,
                undershoot_pct=0.0,
                settling_time_s=1e-6,
                oscillations=0,
                score=float(search.max_iterations - iteration),
                passed=False,
            )
            record = IterationRecord(
                iteration=iteration,
                phase=candidate.phase,
                wc_rad_s=0.0,
                phi_deg=0.0,
                pid=PidParameters(
                    kp=float(candidate.mod0_kp),
                    ki=float(candidate.mod0_ki),
                    kd=float(candidate.mod0_kd),
                    kf=float(candidate.mod0_kpole1),
                ),
                metrics=metrics,
                waveform=Waveform(time_s=[], vout_v=[]),
                timestamp=float(iteration),
                candidate=candidate,
            )
            history.append(record)
            best = record if best is None or record.metrics.score < best.metrics.score else best

        self.assertIsNone(tuner.next_candidate(history, best))

    def test_hardware_search_uses_full_coarse_budget_for_global_sampling(self) -> None:
        search = SearchSpace(max_coarse_iterations=11, max_refined_iterations=4)
        tuner = HardwareGridHeuristicTuner(search)
        history: list[IterationRecord] = []
        best: IterationRecord | None = None

        for iteration in range(1, search.total_iteration_budget() + 1):
            candidate = tuner.next_candidate(history, best)
            self.assertIsNotNone(candidate)
            metrics = ResponseMetrics(
                overshoot_pct=0.0,
                undershoot_pct=0.0,
                settling_time_s=1e-6,
                oscillations=0,
                score=float(search.total_iteration_budget() - iteration),
                passed=False,
            )
            record = IterationRecord(
                iteration=iteration,
                phase=candidate.phase,
                wc_rad_s=0.0,
                phi_deg=0.0,
                pid=PidParameters(
                    kp=float(candidate.mod0_kp),
                    ki=float(candidate.mod0_ki),
                    kd=float(candidate.mod0_kd),
                    kf=float(candidate.mod0_kpole1),
                ),
                metrics=metrics,
                waveform=Waveform(time_s=[], vout_v=[]),
                timestamp=float(iteration),
                candidate=candidate,
            )
            history.append(record)
            best = record if best is None or record.metrics.score < best.metrics.score else best

        self.assertEqual(sum(1 for record in history if record.phase == "baseline"), 1)
        self.assertEqual(sum(1 for record in history if record.phase == "coordinate"), 10)
        self.assertEqual(sum(1 for record in history if record.phase == "pairwise_coarse"), 0)
        self.assertEqual(sum(1 for record in history if record.phase == "local_refine"), 4)
        self.assertIsNone(tuner.next_candidate(history, best))

    def test_coarse_global_sampling_covers_pid_endpoints(self) -> None:
        search = SearchSpace(max_coarse_iterations=90, max_refined_iterations=0)
        tuner = HardwareGridHeuristicTuner(search)
        history: list[IterationRecord] = []
        best: IterationRecord | None = None

        for iteration in range(1, search.total_iteration_budget() + 1):
            candidate = tuner.next_candidate(history, best)
            self.assertIsNotNone(candidate)
            metrics = ResponseMetrics(
                overshoot_pct=0.0,
                undershoot_pct=0.0,
                settling_time_s=1e-6,
                oscillations=0,
                score=float(iteration),
                passed=False,
            )
            record = IterationRecord(
                iteration=iteration,
                phase=candidate.phase,
                wc_rad_s=0.0,
                phi_deg=0.0,
                pid=PidParameters(kp=0.0, ki=0.0, kd=0.0, kf=0.0),
                metrics=metrics,
                waveform=Waveform(time_s=[], vout_v=[]),
                timestamp=float(iteration),
                candidate=candidate,
            )
            history.append(record)
            best = best or record

        coordinates = [record.candidate for record in history if record.phase == "coordinate"]
        self.assertEqual(len(coordinates), 89)
        self.assertEqual({candidate.mod0_kp for candidate in coordinates} & {100, 255}, {100, 255})
        self.assertEqual({candidate.mod0_ki for candidate in coordinates} & {150, 255}, {150, 255})
        self.assertEqual({candidate.mod0_kd for candidate in coordinates} & {100, 200}, {100, 200})
        self.assertEqual({candidate.mod0_kpole1 for candidate in coordinates}, {2, 3, 4, 5, 6})
        self.assertEqual({candidate.mod0_kpole2 for candidate in coordinates}, {2, 3, 4, 5, 6})
        self.assertTrue(any(candidate.mod0_kpole1 != candidate.mod0_kpole2 for candidate in coordinates))
        self.assertEqual(len({hardware_candidate_key(candidate) for candidate in coordinates}), len(coordinates))

    def test_recommendations_return_five_distinct_valid_basins(self) -> None:
        candidates = [
            HardwarePidCandidate(mod0_kp=100, mod0_ki=150, mod0_kd=100, mod0_kpole1=3, mod0_kpole2=3),
            HardwarePidCandidate(mod0_kp=255, mod0_ki=255, mod0_kd=200, mod0_kpole1=6, mod0_kpole2=6),
            HardwarePidCandidate(mod0_kp=100, mod0_ki=255, mod0_kd=200, mod0_kpole1=3, mod0_kpole2=6),
            HardwarePidCandidate(mod0_kp=255, mod0_ki=150, mod0_kd=100, mod0_kpole1=6, mod0_kpole2=3),
            HardwarePidCandidate(mod0_kp=178, mod0_ki=202, mod0_kd=150, mod0_kpole1=4, mod0_kpole2=5),
        ]

        def record(iteration: int, candidate: HardwarePidCandidate, score: float) -> IterationRecord:
            return IterationRecord(
                iteration=iteration,
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
                    passed=False,
                ),
                waveform=Waveform(time_s=[], vout_v=[]),
                timestamp=float(iteration),
                candidate=candidate,
            )

        records = [record(index + 1, candidate, float(index + 1)) for index, candidate in enumerate(candidates)]
        records.append(record(6, candidates[0], 0.5))
        records.append(record(7, HardwarePidCandidate(mod0_kp=220), 300.0))
        records.append(record(8, HardwarePidCandidate(mod0_kp=221), float("inf")))
        selected = select_diverse_results(records, 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(len({hardware_candidate_key(item.candidate) for item in selected}), 5)
        self.assertNotIn(7, {item.iteration for item in selected})
        self.assertNotIn(8, {item.iteration for item in selected})
        self.assertIn(6, {item.iteration for item in selected})

    def test_recommendations_prefer_low_penalty_across_distinct_basins(self) -> None:
        candidates = [
            HardwarePidCandidate(mod0_kp=100, mod0_ki=150, mod0_kd=100, mod0_kpole1=3, mod0_kpole2=3),
            HardwarePidCandidate(mod0_kp=255, mod0_ki=255, mod0_kd=200, mod0_kpole1=6, mod0_kpole2=6),
            HardwarePidCandidate(mod0_kp=100, mod0_ki=255, mod0_kd=200, mod0_kpole1=3, mod0_kpole2=6),
            HardwarePidCandidate(mod0_kp=255, mod0_ki=150, mod0_kd=100, mod0_kpole1=6, mod0_kpole2=3),
            HardwarePidCandidate(mod0_kp=178, mod0_ki=202, mod0_kd=150, mod0_kpole1=4, mod0_kpole2=5),
            HardwarePidCandidate(mod0_kp=178, mod0_ki=202, mod0_kd=150, mod0_kpole1=5, mod0_kpole2=4),
        ]

        records = []
        for index, candidate in enumerate(candidates, start=1):
            high_penalty = index == len(candidates)
            records.append(IterationRecord(
                iteration=index,
                phase="drl_policy",
                wc_rad_s=0.0,
                phi_deg=0.0,
                pid=PidParameters(kp=0.0, ki=0.0, kd=0.0, kf=0.0),
                metrics=ResponseMetrics(
                    overshoot_pct=0.1 if high_penalty else 1.0,
                    undershoot_pct=0.1 if high_penalty else 1.0,
                    settling_time_s=1e-6,
                    oscillations=0,
                    score=80.0 if high_penalty else float(index),
                    passed=True,
                ),
                waveform=Waveform(time_s=[], vout_v=[]),
                timestamp=float(index),
                candidate=candidate,
            ))

        selected = select_diverse_results(records, 5)

        self.assertEqual([record.metrics.score for record in selected], [1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertNotIn(6, {record.iteration for record in selected})

    def test_best_prefers_confirmed_and_basins_fill_after_confirmed(self) -> None:
        lucky = HardwarePidCandidate(mod0_kp=200)
        stable = HardwarePidCandidate(mod0_kp=150)
        second_stable = HardwarePidCandidate(mod0_kp=160)

        def record(iteration: int, candidate: HardwarePidCandidate, score: float, passed: bool = True) -> IterationRecord:
            return IterationRecord(
                iteration=iteration,
                phase="drl_confirm",
                wc_rad_s=0.0,
                phi_deg=0.0,
                pid=PidParameters(kp=0.0, ki=0.0, kd=0.0, kf=0.0),
                metrics=ResponseMetrics(0.5, 0.5, 1e-6, 0, score, passed),
                waveform=Waveform(time_s=[], vout_v=[]),
                timestamp=float(iteration),
                candidate=candidate,
            )

        records = [record(1, lucky, -100.0)]
        records.extend(record(index, stable, score) for index, score in zip((2, 3, 4), (-10.0, -12.0, -11.0)))
        records.extend(record(index, second_stable, score) for index, score in zip((5, 6, 7), (-8.0, -9.0, -10.0)))

        best = select_best_result(records)
        basins = select_diverse_results(records, 5)

        self.assertIsNotNone(best)
        self.assertEqual(hardware_candidate_key(best.candidate), hardware_candidate_key(stable))
        self.assertEqual(
            [hardware_candidate_key(item.candidate) for item in basins],
            [
                hardware_candidate_key(stable),
                hardware_candidate_key(second_stable),
                hardware_candidate_key(lucky),
            ],
        )

    def test_completed_run_restores_confirmed_best(self) -> None:
        candidate = HardwarePidCandidate(mod0_kp=151, mod0_ki=199, mod0_kd=112)

        class RepeatingTuner:
            def next_candidate(self, history, best):
                return candidate if len(history) < 3 else None

        class RestoringRunner:
            def __init__(self):
                self.restored = None

            def evaluate(self, proposed, config, experiment):
                return ExperimentResult(
                    waveform=Waveform(time_s=[0.0], vout_v=[config.targets.vout_target_v]),
                    metrics=ResponseMetrics(0.5, 0.5, 1e-6, 0, -10.0, True),
                )

            def restore_candidate(self, restored, config, experiment):
                self.restored = restored
                return {"ok": True}

        runner = RestoringRunner()
        config = TuningConfig(
            search=SearchSpace(max_iterations=3, max_coarse_iterations=3, max_refined_iterations=0)
        )
        session = PidAutotuneSession(
            config=config,
            experiment_runner=runner,
            tuner_factory=lambda config, experiment, history: RepeatingTuner(),
        )

        session.step()
        session.step()
        status = session.step()

        self.assertEqual(status["state"], "complete")
        self.assertEqual(runner.restored, candidate)
        self.assertIn("Restored confirmed Best", status["message"])

    def test_no_fresh_candidate_completion_restores_confirmed_best(self) -> None:
        candidate = HardwarePidCandidate(mod0_kp=151, mod0_ki=199, mod0_kd=112, mod0_ll_bw=76)

        class ExhaustedTuner:
            def next_candidate(self, history, best):
                return candidate if len(history) < 3 else None

        class RestoringRunner:
            def __init__(self):
                self.restored = None

            def evaluate(self, proposed, config, experiment):
                return ExperimentResult(
                    waveform=Waveform(time_s=[0.0], vout_v=[config.targets.vout_target_v]),
                    metrics=ResponseMetrics(0.5, 0.5, 1e-6, 0, -10.0, True),
                )

            def restore_candidate(self, restored, config, experiment):
                self.restored = restored
                return {"ok": True}

        runner = RestoringRunner()
        config = TuningConfig(
            search=SearchSpace(max_iterations=10, max_coarse_iterations=10, max_refined_iterations=0)
        )
        session = PidAutotuneSession(
            config=config,
            experiment_runner=runner,
            tuner_factory=lambda config, experiment, history: ExhaustedTuner(),
        )

        session.step()
        session.step()
        session.step()
        status = session.step()

        self.assertEqual(status["state"], "complete")
        self.assertEqual(runner.restored, candidate)
        self.assertIn("Restored confirmed Best", status["message"])

    def test_session_stops_when_local_refine_penalty_saturates(self) -> None:
        class FlatRunner:
            def evaluate(self, candidate, config, experiment):
                metrics = ResponseMetrics(
                    overshoot_pct=1.0,
                    undershoot_pct=1.0,
                    settling_time_s=5e-6,
                    oscillations=1,
                    score=10.0,
                    passed=False,
                )
                return ExperimentResult(
                    waveform=Waveform(time_s=[0.0], vout_v=[config.targets.vout_target_v]),
                    metrics=metrics,
                )

        session = PidAutotuneSession(
            config=TuningConfig(search=SearchSpace(max_iterations=100)),
            experiment_runner=FlatRunner(),
        )
        status = {}
        for _ in range(100):
            status = session.step()
            if status["state"] == "complete":
                break

        self.assertEqual(status["state"], "complete")
        self.assertIn("Fine tune saturated", status["message"])
        self.assertLess(len(status["history"]), 100)

    def test_hardware_score_penalizes_missing_margins(self) -> None:
        waveform = Waveform(time_s=[0, 1e-6, 2e-6], vout_v=[0.9, 0.91, 0.9])
        metrics = ResponseAnalyzer(TuningTargets()).analyze_hardware(waveform, {})

        self.assertFalse(metrics.passed)
        self.assertIn("missing phase margin", metrics.pass_reasons)
        self.assertNotIn("missing gain margin", metrics.pass_reasons)

    def test_dynamic_baseline_uses_rising_edge_for_undershoot(self) -> None:
        time_s = [index * 1e-6 for index in range(50)]
        input_v = [0.0 if index < 15 else 2.0 for index in range(50)]
        vout_v = [0.95] * 15 + [0.90, 0.92, 0.93] + [0.935] * 32

        metrics = ResponseAnalyzer(TuningTargets()).analyze(Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v))

        self.assertAlmostEqual(metrics.undershoot_pct, (0.935 - 0.90) / 0.935 * 100.0, places=3)
        self.assertEqual(metrics.overshoot_pct, 0.0)
        self.assertAlmostEqual(metrics.low_load_steady_v, 0.95, places=6)
        self.assertAlmostEqual(metrics.high_load_steady_v, 0.935, places=6)

    def test_dynamic_baseline_uses_falling_edge_for_overshoot(self) -> None:
        time_s = [index * 1e-6 for index in range(50)]
        input_v = [2.0 if index < 15 else 0.0 for index in range(50)]
        vout_v = [0.85] * 15 + [0.91, 0.88, 0.87] + [0.865] * 32

        metrics = ResponseAnalyzer(TuningTargets()).analyze(Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v))

        self.assertAlmostEqual(metrics.overshoot_pct, (0.91 - 0.865) / 0.865 * 100.0, places=3)
        self.assertEqual(metrics.undershoot_pct, 0.0)
        self.assertAlmostEqual(metrics.high_load_steady_v, 0.85, places=6)
        self.assertAlmostEqual(metrics.low_load_steady_v, 0.865, places=6)

    def test_dynamic_steady_voltage_averages_multiple_load_segments(self) -> None:
        time_s = [index * 1e-6 for index in range(80)]
        input_v = []
        for index in range(80):
            input_v.append(0.0 if index < 15 or 35 <= index < 55 else 2.0)
        vout_v = (
            [0.96] * 15
            + [0.90, 0.92, 0.93]
            + [0.94] * 17
            + [1.00, 0.98, 0.97]
            + [0.965] * 17
            + [0.91, 0.92, 0.925]
            + [0.93] * 22
        )

        metrics = ResponseAnalyzer(TuningTargets()).analyze(Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v))

        self.assertAlmostEqual(metrics.low_load_steady_v, 0.965, places=6)
        self.assertAlmostEqual(metrics.high_load_steady_v, 0.94, places=6)

    def test_steady_voltage_uses_last_rising_before_window_when_two_rising_edges(self) -> None:
        time_s = [index * 1e-6 for index in range(110)]
        input_v = []
        for index in range(110):
            input_v.append(2.0 if 3 <= index < 52 or index >= 102 else 0.0)
        vout_v = (
            [0.80] * 3
            + [0.70] * 49
            + [0.93] * 50
            + [0.88] * 8
        )

        metrics = ResponseAnalyzer(TuningTargets()).analyze(Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v))

        self.assertAlmostEqual(metrics.low_load_steady_v, 0.93, places=6)
        self.assertAlmostEqual(metrics.high_load_steady_v, 0.70, places=6)

    def test_transient_extremes_use_first_full_high_and_low_load_segments(self) -> None:
        time_s = [index * 1e-6 for index in range(110)]
        input_v = []
        for index in range(110):
            input_v.append(2.0 if 3 <= index < 52 or index >= 102 else 0.0)
        vout_v = (
            [0.93] * 3
            + [0.66, 0.68, 0.70]
            + [0.70] * 46
            + [0.98, 0.95, 0.93]
            + [0.93] * 47
            + [0.80] * 8
        )

        metrics = ResponseAnalyzer(TuningTargets()).analyze(Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v))

        self.assertAlmostEqual(metrics.undershoot_pct, (0.70 - 0.66) / 0.70 * 100.0, places=3)
        self.assertAlmostEqual(metrics.overshoot_pct, (0.98 - 0.93) / 0.93 * 100.0, places=3)
        self.assertGreater(metrics.undershoot_settling_time_s, 0.0)
        self.assertGreater(metrics.overshoot_settling_time_s, 0.0)

    def test_input_edge_debounce_ignores_threshold_chatter(self) -> None:
        time_s = [index * 1e-6 for index in range(120)]
        input_v = [0.0] * 120
        for index in range(3, 52):
            input_v[index] = 2.0
        for index in range(102, 120):
            input_v[index] = 2.0
        input_v[103] = 0.0
        input_v[104] = 2.0
        vout_v = (
            [0.93] * 3
            + [0.826, 0.835, 0.837]
            + [0.837] * 46
            + [0.948, 0.94, 0.937]
            + [0.937] * 47
            + [0.85] * 18
        )

        metrics = ResponseAnalyzer(TuningTargets()).analyze(Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v))

        self.assertAlmostEqual(metrics.high_load_steady_v, 0.837, places=6)
        self.assertAlmostEqual(metrics.low_load_steady_v, 0.937, places=6)
        self.assertAlmostEqual(metrics.undershoot_pct, (0.837 - 0.826) / 0.837 * 100.0, places=3)

    def test_high_load_steady_ignores_trailing_falling_edge(self) -> None:
        dt = 0.1e-6
        count = 1100
        time_s = [index * dt for index in range(count)]
        input_v = []
        vout_v = []
        for index in range(count):
            t_us = time_s[index] * 1e6
            if t_us < 2.0:
                input_v.append(0.0)
                vout_v.append(0.93)
            elif t_us < 50.0:
                input_v.append(2.0)
                vout_v.append(0.83)
            elif t_us < 100.0:
                input_v.append(0.0)
                vout_v.append(0.93)
            else:
                input_v.append(-0.2)
                vout_v.append(0.84)

        metrics = ResponseAnalyzer(TuningTargets()).analyze(Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v))

        self.assertAlmostEqual(metrics.high_load_steady_v, 0.83, places=3)
        self.assertLess(metrics.undershoot_pct, 1.0)
        self.assertLess(metrics.settling_time_s, 10e-6)

    def test_settling_uses_smoothed_waveform_so_ripple_does_not_extend_time(self) -> None:
        dt = 0.05e-6
        count = 1200
        time_s = [index * dt for index in range(count)]
        input_v = [2.0 if 200 <= index < 800 else 0.0 for index in range(count)]
        vout_v = []
        for index in range(count):
            if index < 200:
                vout_v.append(0.94)
            elif index < 260:
                vout_v.append(0.76 + (index - 200) / 60.0 * 0.08)
            elif index < 800:
                ripple = 0.018 if index % 2 else -0.018
                vout_v.append(0.84 + ripple)
            else:
                vout_v.append(0.94)

        metrics = ResponseAnalyzer(TuningTargets()).analyze(Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v))

        self.assertLess(metrics.undershoot_settling_time_s, 4e-6)

    def test_undershoot_settling_includes_later_dip_after_first_recovery(self) -> None:
        dt = 0.1e-6
        count = 700
        time_s = [index * dt for index in range(count)]
        input_v = [2.0 if 20 <= index < 500 else 0.0 for index in range(count)]
        vout_v = []
        for index in range(count):
            t_us = time_s[index] * 1e6
            if t_us < 2.0:
                vout_v.append(0.94)
            elif t_us < 5.0:
                vout_v.append(0.83 + (t_us - 2.0) / 3.0 * 0.03)
            elif t_us < 15.0:
                vout_v.append(0.85)
            elif t_us < 25.0:
                vout_v.append(0.80)
            else:
                vout_v.append(0.84)

        metrics = ResponseAnalyzer(TuningTargets()).analyze(Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v))

        self.assertFalse(metrics.undershoot_settling_valid)
        self.assertEqual(metrics.score, 300.0)

    def test_overshoot_settling_includes_later_dip_after_falling_edge(self) -> None:
        dt = 0.1e-6
        count = 700
        time_s = [index * dt for index in range(count)]
        input_v = [2.0 if index < 200 or index >= 600 else 0.0 for index in range(count)]
        vout_v = []
        for index in range(count):
            t_us = time_s[index] * 1e6
            if t_us < 20.0:
                vout_v.append(0.84)
            elif t_us < 23.0:
                vout_v.append(0.91 + (t_us - 20.0) / 3.0 * 0.02)
            elif t_us < 27.0:
                vout_v.append(0.925)
            elif t_us < 34.0:
                vout_v.append(0.912)
            else:
                vout_v.append(0.93)

        metrics = ResponseAnalyzer(TuningTargets()).analyze(Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v))

        self.assertFalse(metrics.overshoot_settling_valid)
        self.assertEqual(metrics.score, 300.0)

    def test_overshoot_settling_catches_shallow_dip_after_first_recovery(self) -> None:
        dt = 0.05e-6
        count = 1200
        time_s = [index * dt for index in range(count)]
        input_v = [2.0 if index < 200 or index >= 900 else 0.0 for index in range(count)]
        vout_v = []
        for index in range(count):
            t_us = time_s[index] * 1e6
            if t_us < 10.0:
                vout_v.append(0.84)
            elif t_us < 10.75:
                vout_v.append(0.86 + (t_us - 10.0) / 0.75 * 0.06)
            elif t_us < 11.75:
                vout_v.append(0.925)
            elif t_us < 12.75:
                vout_v.append(0.920)
            else:
                vout_v.append(0.928)

        metrics = ResponseAnalyzer(TuningTargets()).analyze(Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v))

        self.assertGreater(metrics.overshoot_settling_time_s, 2e-6)
        self.assertLess(metrics.overshoot_settling_time_s, 8e-6)

    def test_overshoot_settling_includes_shallow_post_rise_rollback(self) -> None:
        dt = 0.05e-6
        count = 1400
        time_s = [index * dt for index in range(count)]
        input_v = [2.0 if index < 1000 else 0.0 for index in range(count)]
        vout_v = []
        for index in range(count):
            t_us = time_s[index] * 1e6
            if t_us < 50.0:
                vout_v.append(0.84)
            elif t_us < 51.0:
                vout_v.append(0.88 + (t_us - 50.0) * 0.06)
            elif t_us < 54.0:
                vout_v.append(0.94)
            elif t_us < 58.0:
                vout_v.append(0.932)
            else:
                vout_v.append(0.94)

        metrics = ResponseAnalyzer(TuningTargets()).analyze(Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v))

        self.assertGreater(metrics.overshoot_settling_time_s, 7e-6)

    def test_settling_ignores_late_brief_noise_after_recovery(self) -> None:
        dt = 0.1e-6
        count = 700
        time_s = [index * dt for index in range(count)]
        input_v = [2.0 if 20 <= index < 500 else 0.0 for index in range(count)]
        vout_v = []
        for index in range(count):
            t_us = time_s[index] * 1e6
            if t_us < 2.0:
                vout_v.append(0.94)
            elif t_us < 5.0:
                vout_v.append(0.82 + (t_us - 2.0) / 3.0 * 0.02)
            else:
                vout_v.append(0.84)
        # A tiny late glitch should not make the settling time look like the
        # whole high-load segment.
        for index in range(300, 301):
            vout_v[index] = 0.79

        metrics = ResponseAnalyzer(TuningTargets()).analyze(Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v))

        self.assertLess(metrics.undershoot_settling_time_s, 8e-6)

    def test_settling_v15_uses_direction_aware_asymmetric_band(self) -> None:
        dt = 0.05e-6
        time_s = [(index - 40) * dt for index in range(2140)]
        input_v: list[float] = []
        vout_v: list[float] = []
        for time_value in time_s:
            t_us = time_value * 1e6
            input_v.append(2.0 if 0.0 <= t_us < 50.0 or t_us >= 100.0 else 0.0)
            if t_us < 0.0:
                value = 0.92
            elif t_us < 0.8:
                value = 0.92 - t_us / 0.8 * 0.10
            elif t_us < 1.8:
                value = 0.82
            elif t_us < 2.8:
                value = 0.82 + (t_us - 1.8) * 0.006
            elif t_us < 4.0:
                value = 0.826 - (t_us - 2.8) / 1.2 * 0.006
            elif t_us < 50.0:
                value = 0.82
            elif t_us < 50.8:
                value = 0.82 + (t_us - 50.0) / 0.8 * 0.10
            elif t_us < 52.0:
                value = 0.92
            elif t_us < 56.0:
                value = 0.92 + (t_us - 52.0) / 4.0 * 0.006
            elif t_us < 60.0:
                value = 0.926 - (t_us - 56.0) / 4.0 * 0.006
            elif t_us < 100.0:
                value = 0.92
            else:
                value = 0.82
            vout_v.append(value)

        metrics = ResponseAnalyzer(TuningTargets()).analyze(
            Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v)
        )

        self.assertEqual(metrics.settling_analysis_version, 15)
        self.assertGreater(metrics.undershoot_settling_time_s, 3.3e-6)
        self.assertLess(metrics.undershoot_settling_time_s, 3.5e-6)
        self.assertGreater(metrics.overshoot_settling_time_s, 6.4e-6)
        self.assertLess(metrics.overshoot_settling_time_s, 6.8e-6)
        self.assertEqual(metrics.settling_diagnostics["undershoot"]["lower_tolerance_mv"], 5.0)
        self.assertEqual(metrics.settling_diagnostics["undershoot"]["upper_tolerance_mv"], 3.0)
        self.assertEqual(metrics.settling_diagnostics["overshoot"]["lower_tolerance_mv"], 3.0)
        self.assertEqual(metrics.settling_diagnostics["overshoot"]["upper_tolerance_mv"], 5.0)
        self.assertIsNone(metrics.settling_diagnostics["undershoot"]["band_ramp_start_us"])
        self.assertIsNone(metrics.settling_diagnostics["undershoot"]["band_ramp_stop_us"])
        self.assertEqual(metrics.settling_diagnostics["undershoot"]["decision_filter_hz"], 600_000.0)
        self.assertEqual(metrics.settling_diagnostics["undershoot"]["band_schedule"], "voltage falling: -5/+3 mV")
        self.assertEqual(metrics.settling_diagnostics["overshoot"]["band_schedule"], "voltage rising: -3/+5 mV")

    def test_iteration_1728_uses_final_entry_not_first_entry(self) -> None:
        artifact = Path(
            "results/autotune_runs/saved/Permanent_2026-07-21_02/files/iteration_1728_scope.npz"
        )
        if not artifact.is_file():
            self.skipTest("iteration 1728 scope artifact is not available")
        with np.load(artifact, allow_pickle=False) as payload:
            points = int(payload["points"])
            time_s = (
                float(payload["x_start"])
                + np.arange(points, dtype=np.float64) * float(payload["x_increment"])
            )
            waveform = Waveform(
                time_s=time_s.tolist(),
                vout_v=np.asarray(payload["y_CH3"], dtype=np.float64).tolist(),
                input_v=np.asarray(payload["y_CH1"], dtype=np.float64).tolist(),
            )

        metrics = ResponseAnalyzer(TuningTargets()).analyze(waveform)

        self.assertTrue(metrics.undershoot_settling_valid)
        self.assertTrue(metrics.overshoot_settling_valid)
        self.assertGreater(metrics.undershoot_settling_time_s, 2.6e-6)
        self.assertLess(metrics.undershoot_settling_time_s, 2.8e-6)
        self.assertGreater(metrics.overshoot_settling_time_s, 7.3e-6)
        self.assertLess(metrics.overshoot_settling_time_s, 7.6e-6)

    def test_iteration_298_uses_six_hundred_khz_asymmetric_band(self) -> None:
        artifact = Path(
            "results/autotune_runs/saved/Permanent_2026-07-21_02/files/iteration_298_scope.npz"
        )
        if not artifact.is_file():
            self.skipTest("iteration 298 scope artifact is not available")
        with np.load(artifact, allow_pickle=False) as payload:
            points = int(payload["points"])
            time_s = (
                float(payload["x_start"])
                + np.arange(points, dtype=np.float64) * float(payload["x_increment"])
            )
            waveform = Waveform(
                time_s=time_s.tolist(),
                vout_v=np.asarray(payload["y_CH3"], dtype=np.float64).tolist(),
                input_v=np.asarray(payload["y_CH1"], dtype=np.float64).tolist(),
            )

        metrics = ResponseAnalyzer(TuningTargets()).analyze(waveform)

        self.assertTrue(metrics.undershoot_settling_valid)
        self.assertTrue(metrics.overshoot_settling_valid)
        self.assertGreater(metrics.undershoot_settling_time_s, 1.0e-6)
        self.assertLess(metrics.undershoot_settling_time_s, 1.2e-6)
        self.assertGreater(metrics.overshoot_settling_time_s, 1.0e-6)
        self.assertLess(metrics.overshoot_settling_time_s, 1.3e-6)

    def test_iteration_124_respects_five_mv_rising_upper_band(self) -> None:
        artifact = Path(
            "results/autotune_runs/saved/Permanent_2026-07-21_02/files/iteration_124_scope.npz"
        )
        if not artifact.is_file():
            self.skipTest("iteration 124 scope artifact is not available")
        with np.load(artifact, allow_pickle=False) as payload:
            points = int(payload["points"])
            time_s = (
                float(payload["x_start"])
                + np.arange(points, dtype=np.float64) * float(payload["x_increment"])
            )
            waveform = Waveform(
                time_s=time_s.tolist(),
                vout_v=np.asarray(payload["y_CH3"], dtype=np.float64).tolist(),
                input_v=np.asarray(payload["y_CH1"], dtype=np.float64).tolist(),
            )

        metrics = ResponseAnalyzer(TuningTargets()).analyze(waveform)
        diagnostics = metrics.settling_diagnostics["overshoot"]

        self.assertTrue(metrics.overshoot_settling_valid)
        self.assertGreater(metrics.overshoot_settling_time_s, 1.1e-6)
        self.assertLess(metrics.overshoot_settling_time_s, 1.4e-6)
        self.assertEqual(diagnostics["secondary_excursion_count"], 0)

    def test_iteration_1511_uses_six_hundred_khz_asymmetric_band(self) -> None:
        artifact = Path(
            "results/autotune_runs/saved/Permanent_2026-07-21_02/files/iteration_1511_scope.npz"
        )
        if not artifact.is_file():
            self.skipTest("iteration 1511 scope artifact is not available")
        with np.load(artifact, allow_pickle=False) as payload:
            points = int(payload["points"])
            time_s = (
                float(payload["x_start"])
                + np.arange(points, dtype=np.float64) * float(payload["x_increment"])
            )
            waveform = Waveform(
                time_s=time_s.tolist(),
                vout_v=np.asarray(payload["y_CH3"], dtype=np.float64).tolist(),
                input_v=np.asarray(payload["y_CH1"], dtype=np.float64).tolist(),
            )

        metrics = ResponseAnalyzer(TuningTargets()).analyze(waveform)
        diagnostics = metrics.settling_diagnostics["undershoot"]

        self.assertTrue(metrics.undershoot_settling_valid)
        self.assertGreater(metrics.undershoot_settling_time_s, 1.0e-6)
        self.assertLess(metrics.undershoot_settling_time_s, 1.2e-6)
        self.assertGreater(metrics.overshoot_settling_time_s, 1.0e-6)
        self.assertLess(metrics.overshoot_settling_time_s, 1.2e-6)
        self.assertGreater(diagnostics["first_entry_us"], 1.0)
        self.assertLess(diagnostics["first_entry_us"], 1.2)
        self.assertFalse(diagnostics["uses_time_bins"])
        self.assertEqual(diagnostics["lower_tolerance_mv"], 5.0)
        self.assertEqual(diagnostics["upper_tolerance_mv"], 3.0)
        self.assertEqual(diagnostics["prominent_reversal_count"], 0)
        self.assertEqual(diagnostics["secondary_excursion_count"], 0)

    def test_settling_v15_filters_high_frequency_tail_without_false_invalid(self) -> None:
        dt = 0.1e-6
        time_s = [(index - 20) * dt for index in range(1070)]
        input_v: list[float] = []
        vout_v: list[float] = []
        for time_value in time_s:
            t_us = time_value * 1e6
            high_load = 0.0 <= t_us < 50.0 or t_us >= 100.0
            input_v.append(2.0 if high_load else 0.0)
            target = 0.82 if high_load else 0.92
            if t_us < 0.0:
                target = 0.92
            ripple = 0.006 if int(max(t_us, 0.0) / 0.6) % 2 else -0.006
            vout_v.append(target + ripple)

        metrics = ResponseAnalyzer(TuningTargets()).analyze(
            Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v)
        )

        self.assertTrue(metrics.passed)
        self.assertLess(metrics.score, 300.0)
        self.assertTrue(metrics.undershoot_settling_valid)
        self.assertTrue(metrics.overshoot_settling_valid)

    def test_settling_does_not_turn_slow_steady_drift_into_watch_window_limit(self) -> None:
        dt = 0.1e-6
        count = 1070
        time_s = [(index - 20) * dt for index in range(count)]
        input_v = []
        vout_v = []
        for time_value in time_s:
            t_us = time_value * 1e6
            input_v.append(2.0 if 0.0 <= t_us < 50.0 or t_us >= 100.0 else 0.0)
            if t_us < 0.0:
                vout_v.append(0.935)
            elif t_us < 50.0:
                elapsed = t_us
                vout_v.append(0.837 - 0.020 * math.exp(-elapsed / 0.7))
            elif t_us < 58.0:
                elapsed = t_us - 50.0
                vout_v.append(0.935 - 0.095 * math.exp(-elapsed / 1.2))
            elif t_us < 100.0:
                # A few millivolts of slow steady-state drift is not another
                # transient and must not pin Ts to the 30 us reset-watch cap.
                vout_v.append(0.938 - 0.003 * (t_us - 58.0) / 42.0)
            else:
                vout_v.append(0.84)

        metrics = ResponseAnalyzer(TuningTargets()).analyze(
            Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v)
        )

        self.assertLess(metrics.overshoot_settling_time_s, 15e-6)
        self.assertNotAlmostEqual(metrics.overshoot_settling_time_s, 30e-6, places=7)

    def test_ch1_spike_and_threshold_chatter_do_not_create_extra_edges(self) -> None:
        input_v = [0.0] * 100 + [2.0] * 100 + [0.0] * 100
        input_v[20] = 20.0
        input_v[99] = 0.95
        input_v[100] = 1.05
        input_v[101] = 0.98
        input_v[102] = 2.0

        edges = ResponseAnalyzer(TuningTargets()).input_edge_indices(input_v)

        self.assertEqual([kind for _, kind in edges], ["rising", "falling"])
        self.assertGreaterEqual(edges[0][0], 100)

    def test_sustained_large_oscillation_has_no_settling_time_and_max_penalty(self) -> None:
        dt = 0.05e-6
        count = 2200
        time_s = [(index - 40) * dt for index in range(count)]
        input_v = []
        vout_v = []
        for time_value in time_s:
            t_us = time_value * 1e6
            high_load = 0.0 <= t_us < 50.0 or t_us >= 100.0
            input_v.append(2.0 if high_load else 0.0)
            if t_us < 0.0:
                vout_v.append(0.935)
            else:
                center = 0.837 if high_load else 0.935
                vout_v.append(center + 0.055 * math.sin(2.0 * math.pi * t_us / 2.0))

        waveform = Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v)
        analyzer = ResponseAnalyzer(TuningTargets())
        transient = analyzer.analyze(waveform)
        metrics = analyzer.analyze_hardware(
            waveform,
            {
                "phase_margin_deg": 60.0,
                "phase_crossover_hz": 150_000.0,
                "gain_margin_db": 10.0,
            },
            precomputed_transient=transient,
        )

        self.assertFalse(metrics.passed)
        self.assertEqual(metrics.score, 300.0)
        self.assertEqual(metrics.overshoot_settling_time_s, 0.0)
        self.assertEqual(metrics.undershoot_settling_time_s, 0.0)
        self.assertGreaterEqual(metrics.oscillations, 8)
        self.assertTrue(any("sustained oscillation" in reason for reason in metrics.pass_reasons))

    def test_damped_ringing_is_not_classified_as_sustained_oscillation(self) -> None:
        dt = 0.05e-6
        count = 2200
        time_s = [(index - 40) * dt for index in range(count)]
        input_v = []
        vout_v = []
        for time_value in time_s:
            t_us = time_value * 1e6
            high_load = 0.0 <= t_us < 50.0 or t_us >= 100.0
            input_v.append(2.0 if high_load else 0.0)
            if t_us < 0.0:
                vout_v.append(0.935)
                continue
            edge_time = 100.0 if t_us >= 100.0 else 50.0 if t_us >= 50.0 else 0.0
            elapsed = t_us - edge_time
            center = 0.837 if high_load else 0.935
            vout_v.append(center + 0.055 * math.exp(-elapsed / 1.5) * math.sin(2.0 * math.pi * elapsed / 1.2))

        metrics = ResponseAnalyzer(TuningTargets()).analyze(
            Waveform(time_s=time_s, vout_v=vout_v, input_v=input_v)
        )

        self.assertLess(metrics.score, 300.0)
        self.assertFalse(any("invalid transient waveform" in reason for reason in metrics.pass_reasons))

    def test_hardware_analysis_can_reuse_synchronous_transient_snapshot(self) -> None:
        analyzer = ResponseAnalyzer(TuningTargets())
        transient = ResponseMetrics(
            overshoot_pct=0.4,
            undershoot_pct=0.5,
            settling_time_s=3.2e-6,
            oscillations=0,
            score=12.0,
            passed=False,
            overshoot_settling_time_s=3.2e-6,
            undershoot_settling_time_s=2.8e-6,
        )

        metrics = analyzer.analyze_hardware(
            waveform=None,
            bode_margins={
                "phase_margin_deg": 60.0,
                "phase_crossover_hz": 150_000.0,
                "gain_margin_db": 10.0,
            },
            precomputed_transient=transient,
        )

        self.assertEqual(metrics.overshoot_settling_time_s, 3.2e-6)
        self.assertEqual(metrics.undershoot_settling_time_s, 2.8e-6)
        self.assertIn("transient limits not met", metrics.pass_reasons)

    def test_steady_voltage_fields_are_none_without_input_edges(self) -> None:
        waveform = Waveform(time_s=[0, 1e-6, 2e-6], vout_v=[0.9, 0.91, 0.9])
        metrics = ResponseAnalyzer(TuningTargets()).analyze(waveform)

        self.assertIsNone(metrics.low_load_steady_v)
        self.assertIsNone(metrics.high_load_steady_v)

    def test_session_uses_hardware_experiment_runner_candidate(self) -> None:
        class Runner:
            def __init__(self) -> None:
                self.candidates = []

            def evaluate(self, candidate, config, experiment):
                self.candidates.append(candidate)
                waveform = Waveform(time_s=[0, 1e-6, 2e-6], vout_v=[0.9, 0.9, 0.9])
                metrics = ResponseAnalyzer(config.targets).analyze_hardware(
                    waveform,
                    {"phase_margin_deg": 60, "phase_crossover_hz": 100_000, "gain_margin_db": 10},
                )
                return ExperimentResult(waveform=waveform, metrics=metrics, duration_s=0.1)

        runner = Runner()
        session = PidAutotuneSession(experiment_runner=runner)

        status = session.step()

        self.assertEqual(len(runner.candidates), 1)
        self.assertEqual(status["history"][0]["candidate"]["mod0_kp"], runner.candidates[0].mod0_kp)

    def test_serial_split_skips_recoverable_transient_protection_points(self) -> None:
        class Runner:
            supports_split_analysis = True

            def __init__(self) -> None:
                self.transient_calls = 0
                self.coordinate_transient_calls = 0
                self.recoveries = 0

            def evaluate(self, candidate, config, experiment):
                if experiment.enable_transient_analysis and not experiment.enable_bode_analysis:
                    self.transient_calls += 1
                    if candidate.phase == "coordinate":
                        self.coordinate_transient_calls += 1
                        if self.coordinate_transient_calls <= 2:
                            raise RuntimeError("Scope safety check failed: response range tripped protection.")
                metrics = ResponseMetrics(
                    overshoot_pct=1.0,
                    undershoot_pct=1.0,
                    settling_time_s=5e-6,
                    oscillations=0,
                    score=20.0,
                    passed=False,
                )
                return ExperimentResult(
                    waveform=Waveform(time_s=[0.0, 1e-6], vout_v=[0.9, 0.9]),
                    metrics=metrics,
                    bode_result={"margins": {"phase_margin_deg": 50.0, "phase_crossover_hz": 120_000.0}},
                    scope_result={"ok": True},
                    duration_s=0.01,
                )

            def recover_after_transient_protection(self, experiment):
                self.recoveries += 1
                return {"ok": True, "steps": [{"ok": True, "name": "mock_recovery"}]}

        runner = Runner()
        session = PidAutotuneSession(
            config=TuningConfig(search=SearchSpace(max_coarse_iterations=4, max_refined_iterations=0)),
            experiment_runner=runner,
        )

        status = session.start()
        deadline = time.time() + 5.0
        while status["state"] == "running" and time.time() < deadline:
            time.sleep(0.05)
            status = session.status()

        self.assertEqual(status["state"], "complete")
        self.assertEqual(runner.recoveries, 2)
        self.assertEqual(len(status["history"]), 4)
        skipped = [record for record in status["history"] if record["write_results"].get("skipped")]
        self.assertEqual(len(skipped), 2)
        self.assertTrue(all("transient protection skipped" in record["metrics"]["pass_reasons"][0] for record in skipped))

    def test_stop_pauses_after_current_serial_candidate(self) -> None:
        class Runner:
            supports_split_analysis = True

            def __init__(self) -> None:
                self.session = None
                self.transient_calls = 0
                self.bode_calls = 0
                self.stop_requested = False

            def evaluate(self, candidate, config, experiment):
                if experiment.enable_transient_analysis:
                    self.transient_calls += 1
                    # The first call is the baseline. Stop during the first
                    # coordinate transient, before its Bode action begins.
                    if candidate.phase == "coordinate" and not self.stop_requested:
                        self.stop_requested = True
                        self.session.stop()
                if experiment.enable_bode_analysis:
                    self.bode_calls += 1
                metrics = ResponseMetrics(
                    overshoot_pct=1.0,
                    undershoot_pct=1.0,
                    settling_time_s=1e-6,
                    oscillations=0,
                    score=20.0,
                    passed=False,
                )
                return ExperimentResult(
                    waveform=Waveform(time_s=[0.0, 1e-6], vout_v=[0.9, 0.9]),
                    metrics=metrics,
                    bode_result={"margins": {"phase_margin_deg": 50.0, "phase_crossover_hz": 120_000.0}},
                    scope_result={"ok": True},
                    duration_s=0.01,
                )

        runner = Runner()
        session = PidAutotuneSession(
            config=TuningConfig(search=SearchSpace(max_coarse_iterations=4, max_refined_iterations=0)),
            experiment_runner=runner,
        )
        runner.session = session

        # Complete the baseline first. The fake runner requests Stop during the
        # next transient; that candidate still completes its Bode measurement
        # before the session pauses between candidates.
        session.step()
        session.start()
        deadline = time.time() + 5.0
        status = session.status()
        while status["state"] == "running" and time.time() < deadline:
            time.sleep(0.02)
            status = session.status()

        self.assertEqual(status["state"], "paused")
        self.assertEqual(len(status["history"]), 2)
        self.assertEqual(runner.transient_calls, 2)
        self.assertEqual(runner.bode_calls, 2)

        # Resume continues with the following candidate.
        resumed = session.resume()
        deadline = time.time() + 5.0
        while resumed["state"] == "running" and time.time() < deadline:
            time.sleep(0.02)
            resumed = session.status()
        self.assertGreaterEqual(len(resumed["history"]), 2)

    def test_serial_coarse_candidate_skips_recoverable_transient_protection_point(self) -> None:
        class Runner:
            def __init__(self) -> None:
                self.recoveries = 0
                self.did_raise = False

            def evaluate(self, candidate, config, experiment):
                if candidate.phase == "coordinate" and not self.did_raise:
                    self.did_raise = True
                    raise RuntimeError(
                        "Scope safety check failed: response range 0.6560 V to 0.6560 V exceeds 0.6797 V to 1.1797 V."
                    )
                metrics = ResponseMetrics(
                    overshoot_pct=1.0,
                    undershoot_pct=1.0,
                    settling_time_s=5e-6,
                    oscillations=0,
                    score=20.0,
                    passed=False,
                )
                return ExperimentResult(
                    waveform=Waveform(time_s=[0.0, 1e-6], vout_v=[0.9, 0.9]),
                    metrics=metrics,
                    duration_s=0.01,
                )

            def recover_after_transient_protection(self, experiment):
                self.recoveries += 1
                return {"ok": True, "steps": [{"ok": True, "name": "mock_recovery"}]}

        runner = Runner()
        session = PidAutotuneSession(
            config=TuningConfig(search=SearchSpace(max_coarse_iterations=4, max_refined_iterations=0)),
            experiment_runner=runner,
        )
        status = {}
        for _ in range(4):
            status = session.step()

        self.assertNotEqual(status["state"], "error")
        self.assertEqual(runner.recoveries, 1)
        skipped = [record for record in status["history"] if record["write_results"].get("skipped")]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["phase"], "coordinate")

    def test_scope_timeout_pauses_run_without_recording_a_false_penalty(self) -> None:
        class Runner:
            def evaluate(self, candidate, config, experiment):
                raise RuntimeError(
                    "Scope capture failed: Scope single acquisition did not complete within 8.0 s."
                )

        session = PidAutotuneSession(
            config=TuningConfig(search=SearchSpace(max_coarse_iterations=4, max_refined_iterations=0)),
            experiment_runner=Runner(),
        )

        status = session.step()

        self.assertEqual(status["state"], "paused")
        self.assertEqual(status["history"], [])
        self.assertIn("Scope recovery exhausted", status["message"])
        self.assertIn("Resume", status["message"])


if __name__ == "__main__":
    unittest.main()
