from __future__ import annotations

import unittest

from hardware.tuning.models import (
    HardwarePidCandidate,
    IterationRecord,
    PidParameters,
    ResponseMetrics,
    SearchParameter,
    SearchSpace,
    Waveform,
    bandwidth_objective,
    automatic_search_parameter,
)
from hardware.tuning.search import HardwareGridHeuristicTuner, select_best_result


def record(
    iteration: int,
    candidate: HardwarePidCandidate,
    *,
    penalty: float,
    passed: bool,
) -> IterationRecord:
    objective, bonus = bandwidth_objective(penalty, candidate, passed=passed)
    return IterationRecord(
        iteration=iteration,
        phase=candidate.phase,
        wc_rad_s=0.0,
        phi_deg=0.0,
        pid=PidParameters(0.0, 0.0, 0.0, 0.0),
        metrics=ResponseMetrics(0.5, 0.5, 1e-6, 0, penalty, passed),
        waveform=Waveform([], []),
        timestamp=float(iteration),
        candidate=candidate,
        objective_score=objective,
        bandwidth_bonus=bonus,
        optimizer_metadata={"bandwidth_objective_eligible": True},
    )


class BandwidthObjectiveTest(unittest.TestCase):
    def test_search_resolution_is_derived_from_budget_not_submitted_points(self) -> None:
        submitted = SearchParameter(center=100, min=47, max=127, step=1, points=99)
        automatic = automatic_search_parameter(submitted, 100, integer=True)
        self.assertEqual(automatic.points, 12)
        self.assertNotEqual(automatic.points, submitted.points)
        self.assertEqual(automatic_search_parameter(submitted, 20, integer=True).points, 9)

        small_integer_range = automatic_search_parameter(
            SearchParameter(center=3, min=3, max=6, step=1, points=99),
            500,
            integer=True,
        )
        self.assertEqual(small_integer_range.points, 4)

    def test_bonus_is_bounded_and_only_applies_to_passes(self) -> None:
        low = HardwarePidCandidate(mod0_ll_bw=47)
        high = HardwarePidCandidate(mod0_ll_bw=79)
        middle = HardwarePidCandidate(mod0_ll_bw=55)
        self.assertEqual(bandwidth_objective(20.0, low, passed=True), (20.0, 0.0))
        self.assertEqual(bandwidth_objective(20.0, high, passed=True), (10.0, 10.0))
        objective, bonus = bandwidth_objective(20.0, middle, passed=True)
        self.assertAlmostEqual(bonus, 10.0 * 8.0 / 32.0)
        self.assertAlmostEqual(objective, 20.0 - bonus)
        self.assertEqual(bandwidth_objective(2.0, high, passed=False), (2.0, 0.0))
        self.assertEqual(
            bandwidth_objective(2.0, high, passed=True, both_analyses_enabled=False),
            (2.0, 0.0),
        )

    def test_confirmed_pass_beats_unconfirmed_and_failed_high_bandwidth(self) -> None:
        confirmed = HardwarePidCandidate(mod0_ll_bw=87, phase="grid_confirm")
        unconfirmed = HardwarePidCandidate(mod0_kp=166, mod0_ll_bw=79, phase="coordinate")
        failed = HardwarePidCandidate(mod0_kp=167, mod0_ll_bw=79, phase="coordinate")
        history = [
            record(1, confirmed, penalty=5.0, passed=True),
            record(2, confirmed, penalty=5.2, passed=True),
            record(3, confirmed, penalty=5.1, passed=True),
            record(4, unconfirmed, penalty=-10.0, passed=True),
            record(5, failed, penalty=-100.0, passed=False),
        ]
        best = select_best_result(history)
        self.assertIsNotNone(best)
        self.assertEqual(best.candidate.mod0_ll_bw, 87)  # type: ignore[union-attr]

    def test_grid_confirms_then_climbs_one_discrete_bandwidth_level(self) -> None:
        tuner = HardwareGridHeuristicTuner(SearchSpace(max_coarse_iterations=20, max_refined_iterations=20))
        candidate = HardwarePidCandidate(mod0_ll_bw=74, phase="coordinate")
        first = record(1, candidate, penalty=2.0, passed=True)
        confirmation = tuner.next_candidate([first], first)
        self.assertEqual(confirmation.mod0_ll_bw, 74)  # type: ignore[union-attr]
        self.assertEqual(confirmation.phase, "grid_confirm")  # type: ignore[union-attr]

        history = [
            first,
            record(2, candidate, penalty=2.1, passed=True),
            record(3, candidate, penalty=2.0, passed=True),
        ]
        climb = tuner.next_candidate(history, select_best_result(history))
        self.assertEqual(climb.mod0_ll_bw, 75)  # type: ignore[union-attr]
        self.assertEqual(climb.phase, "bandwidth_climb")  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
