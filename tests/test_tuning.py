from __future__ import annotations

import unittest

from hardware.tuning import (
    CompensatorDesign,
    PidAutotuneSession,
    PlantParams,
    SearchSpace,
    StubPidProgrammer,
    TuningConfig,
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


if __name__ == "__main__":
    unittest.main()
