from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hardware.instruments.bode_analyzer import calculate_stability_margins
from hardware.instruments.bode100 import Bode100Driver, Bode100Error


class _FakeSocket:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class Bode100Test(unittest.TestCase):
    def test_scpi_port_check_success(self) -> None:
        calls = []

        def fake_create_connection(address, timeout):
            calls.append((address, timeout))
            return _FakeSocket()

        with mock.patch("hardware.instruments.bode100.socket.create_connection", fake_create_connection):
            driver = Bode100Driver(serial_number="Bode100R2-TEST", host="127.0.0.1", port=5025)

            self.assertTrue(driver.is_scpi_server_running())

        self.assertEqual(calls, [(("127.0.0.1", 5025), 0.5)])

    def test_scpi_port_check_failure(self) -> None:
        def fake_create_connection(address, timeout):
            raise OSError("closed")

        with mock.patch("hardware.instruments.bode100.socket.create_connection", fake_create_connection):
            driver = Bode100Driver(serial_number="Bode100R2-TEST")

            self.assertFalse(driver.is_scpi_server_running())

    def test_scpi_runner_command_uses_dash_s_serial(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = Path(temp_dir) / "ScpiRunner.exe"
            driver = Bode100Driver(serial_number="Bode100R2-TEST", scpi_runner_path=str(runner))

            self.assertEqual(driver.build_scpi_runner_command(), [str(runner), "-s", "Bode100R2-TEST"])

    def test_missing_serial_has_actionable_error(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BODE100_SERIAL", None)
            driver = Bode100Driver(serial_number="", scpi_runner_path="runner.exe")

            with self.assertRaisesRegex(Bode100Error, "BODE100_SERIAL"):
                driver.build_scpi_runner_command()

    def test_missing_runner_path_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing = Path(temp_dir) / "missing-runner.exe"
            driver = Bode100Driver(serial_number="Bode100R2-TEST", scpi_runner_path=str(missing))

            with mock.patch.object(driver, "is_scpi_server_running", return_value=False):
                with self.assertRaisesRegex(Bode100Error, "BODE100_SCPI_RUNNER_PATH"):
                    driver.ensure_scpi_server()

    def test_phase_margin_uses_positive_bode100_phase_directly(self) -> None:
        margins = calculate_stability_margins(
            [1e3, 1e4, 1e5, 1e6],
            [20.0, 10.0, 0.0, -20.0],
            [100.0, 99.0, 98.0, 20.0],
        )

        self.assertAlmostEqual(margins.phase_crossover_hz, 1e5)
        self.assertAlmostEqual(margins.phase_margin_deg, 98.0)

    def test_phase_margin_supports_classical_negative_phase(self) -> None:
        margins = calculate_stability_margins(
            [1e3, 1e4, 1e5],
            [10.0, 0.0, -10.0],
            [-80.0, -90.0, -100.0],
        )

        self.assertAlmostEqual(margins.phase_crossover_hz, 1e4)
        self.assertAlmostEqual(margins.phase_margin_deg, 90.0)


if __name__ == "__main__":
    unittest.main()
