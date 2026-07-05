from __future__ import annotations

import unittest

from hardware.tuning import (
    CompensatorDesign,
    ExperimentResult,
    HardwareGridHeuristicTuner,
    PidAutotuneSession,
    PlantParams,
    ResponseAnalyzer,
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

    def test_hardware_score_penalizes_missing_margins(self) -> None:
        waveform = Waveform(time_s=[0, 1e-6, 2e-6], vout_v=[0.9, 0.91, 0.9])
        metrics = ResponseAnalyzer(TuningTargets()).analyze_hardware(waveform, {})

        self.assertFalse(metrics.passed)
        self.assertIn("missing phase margin", metrics.pass_reasons)
        self.assertIn("missing gain margin", metrics.pass_reasons)

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


if __name__ == "__main__":
    unittest.main()
