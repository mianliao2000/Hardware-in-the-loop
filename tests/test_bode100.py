from __future__ import annotations

import pytest

from hardware.instruments.bode_analyzer import calculate_stability_margins
from hardware.instruments.bode100 import Bode100Driver, Bode100Error


class _FakeSocket:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_scpi_port_check_success(monkeypatch):
    calls = []

    def fake_create_connection(address, timeout):
        calls.append((address, timeout))
        return _FakeSocket()

    monkeypatch.setattr("hardware.instruments.bode100.socket.create_connection", fake_create_connection)
    driver = Bode100Driver(serial_number="Bode100R2-TEST", host="127.0.0.1", port=5025)

    assert driver.is_scpi_server_running() is True
    assert calls == [(("127.0.0.1", 5025), 0.5)]


def test_scpi_port_check_failure(monkeypatch):
    def fake_create_connection(address, timeout):
        raise OSError("closed")

    monkeypatch.setattr("hardware.instruments.bode100.socket.create_connection", fake_create_connection)
    driver = Bode100Driver(serial_number="Bode100R2-TEST")

    assert driver.is_scpi_server_running() is False


def test_scpi_runner_command_uses_dash_s_serial(tmp_path):
    runner = tmp_path / "ScpiRunner.exe"
    driver = Bode100Driver(serial_number="Bode100R2-TEST", scpi_runner_path=str(runner))

    assert driver.build_scpi_runner_command() == [str(runner), "-s", "Bode100R2-TEST"]


def test_missing_serial_has_actionable_error(monkeypatch):
    monkeypatch.delenv("BODE100_SERIAL", raising=False)
    driver = Bode100Driver(serial_number="", scpi_runner_path="runner.exe")

    with pytest.raises(Bode100Error, match="BODE100_SERIAL"):
        driver.build_scpi_runner_command()


def test_missing_runner_path_has_actionable_error(monkeypatch, tmp_path):
    missing = tmp_path / "missing-runner.exe"
    driver = Bode100Driver(serial_number="Bode100R2-TEST", scpi_runner_path=str(missing))
    monkeypatch.setattr(driver, "is_scpi_server_running", lambda: False)

    with pytest.raises(Bode100Error, match="BODE100_SCPI_RUNNER_PATH"):
        driver.ensure_scpi_server()


def test_phase_margin_uses_positive_bode100_phase_directly():
    margins = calculate_stability_margins(
        [1e3, 1e4, 1e5, 1e6],
        [20.0, 10.0, 0.0, -20.0],
        [100.0, 99.0, 98.0, 20.0],
    )

    assert margins.phase_crossover_hz == pytest.approx(1e5)
    assert margins.phase_margin_deg == pytest.approx(98.0)


def test_phase_margin_supports_classical_negative_phase():
    margins = calculate_stability_margins(
        [1e3, 1e4, 1e5],
        [10.0, 0.0, -10.0],
        [-80.0, -90.0, -100.0],
    )

    assert margins.phase_crossover_hz == pytest.approx(1e4)
    assert margins.phase_margin_deg == pytest.approx(90.0)
