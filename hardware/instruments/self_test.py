"""Conservative reversible self-tests for lab instruments.

The tests intentionally never toggle instrument output state. When a writable
setting is exercised, its original value is restored before the test completes.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from .board_controller import BoardControllerConfig, create_board_controller
from .bode_analyzer import BodeScpiClient
from .function_generator import FunctionGenerator
from .i2c_adapters import create_i2c_adapter
from .oscilloscope import TektronixOscilloscope
from .power_supply import KeysightN5700PowerSupply
from .visa_resource import list_visa_resources


DEFAULT_AFG_RESOURCE = "USB0::0x0699::0x0356::B010652::INSTR"
DEFAULT_SCOPE_RESOURCE = "USB0::0x0699::0x0522::B010536::INSTR"
DEFAULT_POWER_SUPPLY_RESOURCE = "USB0::0x0957::0xA407::US17M5136P::INSTR"
DEFAULT_BODE_HOST = os.environ.get("BODE100_HOST", "127.0.0.1")
DEFAULT_BODE_PORT = int(os.environ.get("BODE100_PORT", "5025"))
DEFAULT_BODE_SCPI_RUNNER = os.environ.get(
    "BODE100_SCPI_RUNNER_PATH",
    r"C:\Program Files\OMICRON\BodeAnalyzerSuite\OmicronLab.VectorNetworkAnalysis.ScpiRunner.exe",
)
DEFAULT_BODE_SERIAL = os.environ.get("BODE100_SERIAL", "")
DEFAULT_BOARD_ADDRESS = "0x5E"
DEFAULT_BOARD_PAGE = 0
DEFAULT_BOARD_ADAPTER = "xdp"
SELF_TEST_DEVICE_ORDER = ("afg", "bode", "power_supply", "scope", "board_i2c")


@dataclass(frozen=True)
class InstrumentSelfTestConfig:
    afg_resource: str = DEFAULT_AFG_RESOURCE
    scope_resource: str = DEFAULT_SCOPE_RESOURCE
    power_supply_resource: str = DEFAULT_POWER_SUPPLY_RESOURCE
    bode_host: str = DEFAULT_BODE_HOST
    bode_port: int = DEFAULT_BODE_PORT
    bode_runner_path: str = DEFAULT_BODE_SCPI_RUNNER
    bode_serial: str = DEFAULT_BODE_SERIAL
    board_address: str = DEFAULT_BOARD_ADDRESS
    board_page: int = DEFAULT_BOARD_PAGE
    board_adapter: str = DEFAULT_BOARD_ADAPTER
    timeout_ms: int = 5000


def run_instrument_self_test(config: InstrumentSelfTestConfig | None = None) -> dict:
    config = config or InstrumentSelfTestConfig()
    started = time.time()
    resources, resource_error = _list_visa_resources()

    tests = [
        _test_afg(config, resources),
        _test_bode(config),
        _test_power_supply(config, resources),
        _test_scope(config, resources),
        _test_board_i2c(config),
    ]
    return {
        "ok": True,
        "timestamp": time.time(),
        "duration_s": time.time() - started,
        "visa_resources": list(resources),
        "visa_resource_error": resource_error,
        "tests": tests,
        "all_passed": all(item["status"] == "passed" for item in tests),
    }


def run_single_instrument_self_test(device_key: str, config: InstrumentSelfTestConfig | None = None) -> dict:
    config = config or InstrumentSelfTestConfig()
    device_key = device_key.strip().lower()
    if device_key not in SELF_TEST_DEVICE_ORDER:
        raise ValueError(f"Unsupported self-test device: {device_key}")

    started = time.time()
    resources, resource_error = _list_visa_resources()
    if device_key == "afg":
        tests = [_test_afg(config, resources)]
    elif device_key == "bode":
        tests = [_test_bode(config)]
    elif device_key == "power_supply":
        tests = [_test_power_supply(config, resources)]
    elif device_key == "scope":
        tests = [_test_scope(config, resources)]
    else:
        tests = [_test_board_i2c(config)]

    return {
        "ok": True,
        "timestamp": time.time(),
        "duration_s": time.time() - started,
        "visa_resources": list(resources),
        "visa_resource_error": resource_error,
        "tests": tests,
        "all_passed": all(item["status"] == "passed" for item in tests),
    }


def _list_visa_resources() -> tuple[tuple[str, ...], str | None]:
    try:
        return list_visa_resources(), None
    except Exception as exc:
        return (), str(exc)


def _new_result(key: str, label: str, resource: str, resources: tuple[str, ...] = ()) -> dict:
    return {
        "key": key,
        "label": label,
        "status": "failed",
        "resource": resource,
        "resource_present": not resources or resource in resources,
        "identity": "",
        "details": {},
        "actions": [],
        "restored": False,
        "error": "",
        "duration_s": 0.0,
    }


def _test_afg(config: InstrumentSelfTestConfig, resources: tuple[str, ...]) -> dict:
    started = time.time()
    result = _new_result("afg", "Tektronix AFG31000", config.afg_resource, resources)
    afg = None
    original_phase = None
    try:
        afg = FunctionGenerator(config.afg_resource, timeout_ms=config.timeout_ms)
        afg.connect()
        result["identity"] = afg.idn()
        result["details"]["frequency_ch1_hz"] = afg.get_frequency(1)
        original_phase = _afg_phase_query_deg(afg, 1)
        result["details"]["phase_ch1_before_deg"] = f"{original_phase:.6g}"

        test_phase = (original_phase + 1.0) % 360.0
        _record_action(result, "set CH1 phase test value", f"{test_phase:.6g} deg")
        afg.set_phase(test_phase, 1)
        readback = _afg_phase_query_deg(afg, 1)
        result["details"]["phase_ch1_test_deg"] = f"{readback:.6g}"
        if abs(_phase_error_deg(readback, test_phase)) > 0.25:
            raise RuntimeError(f"AFG phase write/readback mismatch: requested {test_phase}, read {readback}")

        _record_action(result, "restore CH1 phase", f"{original_phase:.6g} deg")
        afg.set_phase(original_phase, 1)
        restored = _afg_phase_query_deg(afg, 1)
        result["details"]["phase_ch1_restored_deg"] = f"{restored:.6g}"
        result["restored"] = abs(_phase_error_deg(restored, original_phase)) <= 0.25
        if not result["restored"]:
            raise RuntimeError(f"AFG phase restore mismatch: expected {original_phase}, read {restored}")
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = str(exc)
        if afg is not None and original_phase is not None:
            try:
                afg.set_phase(original_phase, 1)
                result["restored"] = True
                _record_action(result, "restore CH1 phase after error", f"{original_phase:.6g} deg")
            except Exception as restore_exc:
                result["actions"].append({"name": "restore CH1 phase after error", "status": "failed", "value": str(restore_exc)})
    finally:
        if afg is not None:
            try:
                afg.close()
            except Exception:
                pass
        result["duration_s"] = time.time() - started
    return result


def _test_bode(config: InstrumentSelfTestConfig) -> dict:
    started = time.time()
    result = _new_result(
        "bode",
        "OMICRON Bode 100",
        BodeScpiClient.tcpip_resource(config.bode_host, config.bode_port),
    )
    if not _ensure_bode_scpi_runner(config, result):
        result["resource_present"] = False
        result["error"] = (
            f"No TCP SCPI listener is available at {config.bode_host}:{config.bode_port}. "
            "Open Bode Analyzer Suite's SCPI runner/server, or verify that the runner path "
            f"and serial are correct: {config.bode_runner_path}, {config.bode_serial}."
        )
        result["duration_s"] = time.time() - started
        return result
    result["resource_present"] = True
    bode = None
    original_points = None
    try:
        bode = BodeScpiClient(host=config.bode_host, port=config.bode_port, timeout_ms=config.timeout_ms)
        bode.connect()
        result["identity"] = bode.idn()
        original_points = int(float(bode.query(":SENS:SWE:POIN?")))
        result["details"]["sweep_points_before"] = str(original_points)
        test_points = max(2, min(401, original_points + 1 if original_points < 401 else original_points - 1))
        _record_action(result, "set sweep points test value", str(test_points))
        bode.write(f":SENS:SWE:POIN {test_points}")
        readback = int(float(bode.query(":SENS:SWE:POIN?")))
        result["details"]["sweep_points_test"] = str(readback)
        if readback != test_points:
            raise RuntimeError(f"Bode sweep point write/readback mismatch: requested {test_points}, read {readback}")

        _record_action(result, "restore sweep points", str(original_points))
        bode.write(f":SENS:SWE:POIN {original_points}")
        restored = int(float(bode.query(":SENS:SWE:POIN?")))
        result["details"]["sweep_points_restored"] = str(restored)
        result["restored"] = restored == original_points
        if not result["restored"]:
            raise RuntimeError(f"Bode sweep point restore mismatch: expected {original_points}, read {restored}")
        try:
            result["details"]["system_error"] = bode.get_error()
        except Exception as exc:
            result["details"]["system_error"] = f"probe failed: {exc}"
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = str(exc)
        if bode is not None and original_points is not None:
            try:
                bode.write(f":SENS:SWE:POIN {original_points}")
                result["restored"] = True
                _record_action(result, "restore sweep points after error", str(original_points))
            except Exception as restore_exc:
                result["actions"].append({"name": "restore sweep points after error", "status": "failed", "value": str(restore_exc)})
    finally:
        if bode is not None:
            try:
                bode.close()
            except Exception:
                pass
        result["duration_s"] = time.time() - started
    return result


def _ensure_bode_scpi_runner(config: InstrumentSelfTestConfig, result: dict) -> bool:
    """Ensure the local Bode SCPI TCP server exists without opening a probe socket.

    The OMICRON SCPI runner can be sensitive to throwaway TCP connections, so the
    listener check uses Windows' TCP table instead of connecting to the port.
    """
    if _tcp_listener_present(config.bode_port):
        result["details"]["tcp_listener"] = "yes"
        result["details"]["scpi_runner"] = "already running"
        return True

    runner = Path(config.bode_runner_path)
    result["details"]["tcp_listener"] = "no"
    if not config.bode_serial:
        result["details"]["scpi_runner"] = "serial number missing"
        return False
    if not runner.exists():
        result["details"]["scpi_runner"] = "runner executable not found"
        return False

    try:
        _start_bode_scpi_runner(config)
        result["details"]["scpi_runner"] = "started"
    except Exception as exc:
        result["details"]["scpi_runner"] = f"start failed: {exc}"
        return False

    deadline = time.time() + max(8.0, config.timeout_ms / 1000.0)
    while time.time() < deadline:
        if _tcp_listener_present(config.bode_port):
            result["details"]["tcp_listener"] = "yes"
            return True
        time.sleep(0.25)
    result["details"]["tcp_listener"] = "no"
    return False


def _start_bode_scpi_runner(config: InstrumentSelfTestConfig) -> None:
    args = [
        "--ip-address",
        config.bode_host,
        "--port",
        str(config.bode_port),
        "--serial",
        config.bode_serial,
        "--logging-level",
        "Warning",
    ]
    command = (
        "Start-Process "
        f"-FilePath {_ps_quote(config.bode_runner_path)} "
        f"-ArgumentList @({', '.join(_ps_quote(arg) for arg in args)}) "
        "-WindowStyle Hidden"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or f"exit code {completed.returncode}").strip()
        raise RuntimeError(message)


def _tcp_listener_present(port: int) -> bool:
    command = (
        "$c = Get-NetTCPConnection "
        f"-LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue; "
        "if ($c) { 'yes' } else { 'no' }"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip().lower().startswith("yes")


def _ps_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _test_power_supply(config: InstrumentSelfTestConfig, resources: tuple[str, ...]) -> dict:
    started = time.time()
    result = _new_result("power_supply", "Keysight N5767A Power Supply", config.power_supply_resource, resources)
    supply = None
    original_voltage = None
    try:
        supply = KeysightN5700PowerSupply(config.power_supply_resource, timeout_ms=config.timeout_ms)
        supply.connect()
        result["identity"] = supply.idn()
        output_enabled = supply.get_output_enabled()
        original_voltage = _parse_float(supply.query("VOLT?"))
        original_current = _parse_float(supply.query("CURR?"))
        result["details"]["output_enabled"] = str(output_enabled)
        result["details"]["voltage_setpoint_before_v"] = f"{original_voltage:.6g}"
        result["details"]["current_limit_before_a"] = f"{original_current:.6g}"
        result["details"]["measured_voltage_v"] = _safe_float_string(supply.query("MEAS:VOLT?"))
        result["details"]["measured_current_a"] = _safe_float_string(supply.query("MEAS:CURR?"))

        if output_enabled:
            _record_action(result, "safe same-value voltage write while output is on", f"{original_voltage:.6g} V")
            supply.set_voltage(original_voltage)
            _record_action(result, "safe same-value current-limit write while output is on", f"{original_current:.6g} A")
            supply.set_current_limit(original_current)
            result["restored"] = True
        else:
            test_voltage = _small_voltage_step(original_voltage, supply.model_voltage_limit_v)
            _record_action(result, "set voltage test value with output off", f"{test_voltage:.6g} V")
            supply.set_voltage(test_voltage)
            readback = _parse_float(supply.query("VOLT?"))
            result["details"]["voltage_setpoint_test_v"] = f"{readback:.6g}"
            if abs(readback - test_voltage) > 0.02:
                raise RuntimeError(f"Power supply voltage write/readback mismatch: requested {test_voltage}, read {readback}")
            _record_action(result, "restore voltage setpoint", f"{original_voltage:.6g} V")
            supply.set_voltage(original_voltage)
            restored = _parse_float(supply.query("VOLT?"))
            result["details"]["voltage_setpoint_restored_v"] = f"{restored:.6g}"
            result["restored"] = abs(restored - original_voltage) <= 0.02
            if not result["restored"]:
                raise RuntimeError(f"Power supply voltage restore mismatch: expected {original_voltage}, read {restored}")
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = str(exc)
        if supply is not None and original_voltage is not None:
            try:
                supply.set_voltage(original_voltage)
                result["restored"] = True
                _record_action(result, "restore voltage setpoint after error", f"{original_voltage:.6g} V")
            except Exception as restore_exc:
                result["actions"].append({"name": "restore voltage setpoint after error", "status": "failed", "value": str(restore_exc)})
    finally:
        if supply is not None:
            try:
                supply.close()
            except Exception:
                pass
        result["duration_s"] = time.time() - started
    return result


def _test_scope(config: InstrumentSelfTestConfig, resources: tuple[str, ...]) -> dict:
    started = time.time()
    result = _new_result("scope", "Tektronix MSO58 Scope", config.scope_resource, resources)
    scope = None
    original_source = None
    try:
        scope = TektronixOscilloscope(config.scope_resource, timeout_ms=config.timeout_ms)
        scope.connect()
        result["identity"] = scope.idn()
        original_source = _normalize_scope_source(scope.query("DATA:SOURCE?"))
        result["details"]["data_source_before"] = original_source
        test_source = "CH2" if original_source == "CH1" else "CH1"
        _record_action(result, "set waveform data source test value", test_source)
        scope.set_waveform_source(test_source)
        readback = _normalize_scope_source(scope.query("DATA:SOURCE?"))
        result["details"]["data_source_test"] = readback
        if readback != test_source:
            raise RuntimeError(f"Scope data source write/readback mismatch: requested {test_source}, read {readback}")

        _record_action(result, "restore waveform data source", original_source)
        scope.set_waveform_source(original_source)
        restored = _normalize_scope_source(scope.query("DATA:SOURCE?"))
        result["details"]["data_source_restored"] = restored
        result["restored"] = restored == original_source
        if not result["restored"]:
            raise RuntimeError(f"Scope data source restore mismatch: expected {original_source}, read {restored}")
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = str(exc)
        if scope is not None and original_source is not None:
            try:
                scope.set_waveform_source(original_source)
                result["restored"] = True
                _record_action(result, "restore waveform data source after error", original_source)
            except Exception as restore_exc:
                result["actions"].append({"name": "restore waveform data source after error", "status": "failed", "value": str(restore_exc)})
    finally:
        if scope is not None:
            try:
                scope.close()
            except Exception:
                pass
        result["duration_s"] = time.time() - started
    return result


def _test_board_i2c(config: InstrumentSelfTestConfig) -> dict:
    started = time.time()
    resource = f"{config.board_adapter}:address={config.board_address}:page={config.board_page}"
    result = _new_result("board_i2c", "Board I2C / XDPE1A2G5C", resource)
    result["resource_present"] = False
    board = None
    original_page = None
    original_vout = None
    try:
        adapter = create_i2c_adapter(config.board_adapter, timeout_ms=config.timeout_ms)
        board = create_board_controller(
            "infineon_xdp",
            adapter,
            BoardControllerConfig(address=config.board_address, name="XDPE1A2G5C"),
        )
        board.connect()
        result["resource_present"] = True

        try:
            original_page = board.read_page()
            result["details"]["page_before"] = str(original_page)
        except Exception as exc:
            result["details"]["page_before"] = f"read failed: {exc}"

        identity = board.identify()
        result["identity"] = _format_board_identity(identity)
        if identity.status_word is not None:
            result["details"]["status_word"] = f"0x{identity.status_word:04X}"

        raw_mode, mode_name, exponent = board.read_vout_mode(config.board_page)
        result["details"]["vout_mode_raw"] = f"0x{raw_mode:02X}"
        result["details"]["vout_mode"] = mode_name
        result["details"]["vout_exponent"] = str(exponent)

        original_vout = board.read_vout_command(config.board_page)
        read_vout = board.read_vout(config.board_page)
        read_iout = board.read_iout(config.board_page)
        result["details"]["vout_command_before_v"] = f"{original_vout:.6g}"
        result["details"]["read_vout_v"] = f"{read_vout:.6g}"
        result["details"]["read_iout_a"] = f"{read_iout:.6g}"

        test_vout = _small_board_vout_step(original_vout)
        _record_action(result, "set VOUT_COMMAND test value", f"{test_vout:.6g} V")
        board.set_vout_command(test_vout, page=config.board_page)
        vout_test = board.read_vout_command(config.board_page)
        result["details"]["vout_command_test_v"] = f"{vout_test:.6g}"
        if abs(vout_test - test_vout) > 0.002:
            raise RuntimeError(f"Board VOUT_COMMAND mismatch: requested {test_vout}, read {vout_test}")

        _record_action(result, "restore VOUT_COMMAND", f"{original_vout:.6g} V")
        board.set_vout_command(original_vout, page=config.board_page)
        restored = board.read_vout_command(config.board_page)
        result["details"]["vout_command_restored_v"] = f"{restored:.6g}"

        _record_action(
            result,
            "set IOUT test value",
            "skipped: READ_IOUT is telemetry; no safe IOUT setpoint register is confirmed",
        )
        result["details"]["iout_write"] = "skipped_safe_until_register_map_confirmed"

        result["restored"] = abs(restored - original_vout) <= 0.002
        if not result["restored"]:
            raise RuntimeError(f"Board VOUT_COMMAND restore mismatch: expected {original_vout}, read {restored}")
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = str(exc)
        if board is not None and original_vout is not None:
            try:
                board.set_vout_command(original_vout, page=config.board_page)
                result["restored"] = True
                _record_action(result, "restore VOUT_COMMAND after error", f"{original_vout:.6g} V")
            except Exception as restore_exc:
                result["actions"].append(
                    {"name": "restore VOUT_COMMAND after error", "status": "failed", "value": str(restore_exc)}
                )
    finally:
        if board is not None:
            try:
                if original_page is not None:
                    board.set_page(original_page)
            except Exception:
                pass
            try:
                board.close()
            except Exception:
                pass
        result["duration_s"] = time.time() - started
    return result


def _record_action(result: dict, name: str, value: Any) -> None:
    result["actions"].append({"name": name, "status": "ok", "value": str(value)})


def _safe_float_string(value: str) -> str:
    try:
        return f"{float(value):.6g}"
    except Exception:
        return value.strip()


def _parse_float(value: str) -> float:
    return float(str(value).strip().strip('"').split()[-1])


def _phase_error_deg(actual: float, expected: float) -> float:
    return ((actual - expected + 180.0) % 360.0) - 180.0


def _afg_phase_query_deg(afg: FunctionGenerator, channel: int) -> float:
    import math

    raw = _parse_float(afg.get_phase(channel))
    if abs(raw) <= (2.0 * math.pi + 1e-6):
        return math.degrees(raw)
    return raw


def _small_voltage_step(value: float, limit: float) -> float:
    if value + 0.01 <= limit:
        return value + 0.01
    return max(0.0, value - 0.01)


def _small_board_vout_step(value: float) -> float:
    delta = 0.001
    if 0.0 <= value + delta <= 2.0:
        return value + delta
    return max(0.0, value - delta)


def _format_board_identity(identity: Any) -> str:
    parts = [f"address=0x{identity.address:02X}"]
    if identity.manufacturer:
        parts.append(identity.manufacturer)
    if identity.model:
        parts.append(identity.model)
    if identity.revision:
        parts.append(identity.revision)
    return ",".join(parts)


def _normalize_scope_source(value: str) -> str:
    text = str(value).strip().strip('"').upper()
    if " " in text:
        text = text.split()[-1].strip('"').upper()
    return text
