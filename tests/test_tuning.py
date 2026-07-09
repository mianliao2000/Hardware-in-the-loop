from __future__ import annotations

import time
import unittest

from hardware.tuning import (
    CompensatorDesign,
    ExperimentResult,
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


class TuningFrameworkTest(unittest.TestCase):
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
        self.assertEqual(candidate.mod0_kpole1, 0)

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
        search = SearchSpace(max_iterations=100)
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

    def test_hardware_search_splits_coarse_between_coordinate_and_pairwise(self) -> None:
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
        self.assertEqual(sum(1 for record in history if record.phase == "coordinate"), 6)
        self.assertEqual(sum(1 for record in history if record.phase == "pairwise_coarse"), 4)
        self.assertEqual(sum(1 for record in history if record.phase == "local_refine"), 4)
        self.assertIsNone(tuner.next_candidate(history, best))

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

        self.assertGreater(metrics.undershoot_settling_time_s, 10e-6)

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

        self.assertGreater(metrics.overshoot_settling_time_s, 10e-6)

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

    def test_coordinate_batch_skips_recoverable_transient_protection_points(self) -> None:
        class Runner:
            def __init__(self) -> None:
                self.transient_calls = 0
                self.recoveries = 0

            def evaluate(self, candidate, config, experiment):
                if experiment.enable_transient_analysis and not experiment.enable_bode_analysis:
                    self.transient_calls += 1
                    if candidate.phase == "coordinate" and self.transient_calls <= 2:
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
            config=TuningConfig(search=SearchSpace(max_iterations=4)),
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

    def test_refine_candidate_skips_recoverable_transient_protection_point(self) -> None:
        class Runner:
            def __init__(self) -> None:
                self.recoveries = 0
                self.did_raise = False

            def evaluate(self, candidate, config, experiment):
                if candidate.phase == "pairwise_coarse" and not self.did_raise:
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
        self.assertEqual(skipped[0]["phase"], "pairwise_coarse")


if __name__ == "__main__":
    unittest.main()
