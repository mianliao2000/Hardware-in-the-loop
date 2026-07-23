from __future__ import annotations

import os
import math
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock

from hardware.instruments.bode_analyzer import calculate_gain_shape, calculate_stability_margins
from hardware.instruments.bode100 import Bode100Driver, Bode100Error


class _FakeSocket:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class Bode100Test(unittest.TestCase):
    def test_scpi_port_check_success(self) -> None:
        listener = SimpleNamespace(status="LISTEN", laddr=SimpleNamespace(port=5025))
        with mock.patch("psutil.net_connections", return_value=[listener]):
            driver = Bode100Driver(serial_number="Bode100R2-TEST", host="127.0.0.1", port=5025)

            self.assertTrue(driver.is_scpi_server_running())

    def test_scpi_port_check_failure(self) -> None:
        def fake_create_connection(address, timeout):
            raise OSError("closed")

        with mock.patch("psutil.net_connections", return_value=[]), mock.patch(
            "hardware.instruments.bode100.socket.create_connection", side_effect=fake_create_connection
        ) as socket_probe:
            driver = Bode100Driver(serial_number="Bode100R2-TEST")

            self.assertFalse(driver.is_scpi_server_running())
            socket_probe.assert_not_called()

    def test_scpi_runner_command_uses_dash_s_serial(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = Path(temp_dir) / "ScpiRunner.exe"
            driver = Bode100Driver(serial_number="Bode100R2-TEST", scpi_runner_path=str(runner))

            self.assertEqual(
                driver.build_scpi_runner_command(),
                [str(runner), "-s", "Bode100R2-TEST", "-i", "127.0.0.1", "-p", "5025"],
            )

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

    def test_phase_margin_is_invariant_to_unwrap_branch_offset(self) -> None:
        from hardware.instruments.bode_analyzer import _phase_margin_from_gain_crossover_phase

        self.assertAlmostEqual(_phase_margin_from_gain_crossover_phase(84.25), 84.25)
        self.assertAlmostEqual(_phase_margin_from_gain_crossover_phase(84.25 - 360.0), 84.25)
        self.assertAlmostEqual(_phase_margin_from_gain_crossover_phase(84.25 + 720.0), 84.25)

    def test_gain_shape_accepts_steady_log_frequency_descent(self) -> None:
        frequencies = [10 ** (3.0 + 3.0 * index / 200.0) for index in range(201)]
        gain = [40.0 - 20.0 * (math.log10(frequency) - 3.0) for frequency in frequencies]

        shape = calculate_gain_shape(frequencies, gain)

        self.assertTrue(shape["gain_shape_valid"])
        self.assertAlmostEqual(shape["gain_rebound_db"], 0.0, places=6)
        self.assertAlmostEqual(shape["gain_shape_penalty"], 0.0, places=6)

    def test_gain_shape_rejects_sustained_high_frequency_rebound(self) -> None:
        frequencies = [10 ** (3.0 + 3.0 * index / 200.0) for index in range(201)]
        gain = []
        for frequency in frequencies:
            log_frequency = math.log10(frequency)
            if log_frequency <= 5.2:
                value = 40.0 - 20.0 * (log_frequency - 3.0)
            elif log_frequency <= 5.6:
                value = -4.0 + 18.0 * (log_frequency - 5.2)
            else:
                value = 3.2 - 20.0 * (log_frequency - 5.6)
            gain.append(value)

        shape = calculate_gain_shape(frequencies, gain)

        self.assertFalse(shape["gain_shape_valid"])
        self.assertGreater(shape["gain_rebound_db"], 5.0)
        self.assertGreater(shape["gain_shape_penalty"], 40.0)

    def test_gain_shape_rejects_a_broad_flat_platform(self) -> None:
        frequencies = [10 ** (3.0 + 3.0 * index / 200.0) for index in range(201)]
        gain = []
        for frequency in frequencies:
            log_frequency = math.log10(frequency)
            if log_frequency <= 5.0:
                value = 40.0 - 20.0 * (log_frequency - 3.0)
            elif log_frequency <= 5.55:
                value = -1.0 * (log_frequency - 5.0)
            else:
                value = -0.55 - 20.0 * (log_frequency - 5.55)
            gain.append(value)

        shape = calculate_gain_shape(frequencies, gain)

        self.assertFalse(shape["gain_shape_valid"])
        self.assertGreater(shape["gain_flat_span_decades"], 0.30)
        self.assertGreater(shape["gain_shape_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
