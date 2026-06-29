"""Local web GUI and API server for the hardware PID autotuner."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import mimetypes
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_ADDRESS = "0x5E"
DEFAULT_PAGE = 0

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.instruments.board_controller import BoardControllerConfig, create_board_controller
from hardware.instruments.bode_analyzer import BodeScpiClient
from hardware.instruments.function_generator import FunctionGenerator
from hardware.instruments.i2c_adapters import create_i2c_adapter
from hardware.instruments.oscilloscope import TektronixOscilloscope
from hardware.instruments.power_supply import KeysightN5700PowerSupply
from hardware.instruments.self_test import (
    DEFAULT_AFG_RESOURCE,
    DEFAULT_BODE_SCPI_RUNNER,
    DEFAULT_BODE_SERIAL,
    DEFAULT_POWER_SUPPLY_RESOURCE,
    DEFAULT_SCOPE_RESOURCE,
    InstrumentSelfTestConfig,
    run_instrument_self_test,
    run_single_instrument_self_test,
)
from hardware.tuning import (
    PidAutotuneSession,
    PlantParams,
    SearchSpace,
    TuningConfig,
    TuningTargets,
)


DEVICE_LOCK = threading.Lock()
TUNING_SESSION = PidAutotuneSession()
FRONTEND_DIST_DIR = Path(__file__).resolve().parent / "frontend" / "dist"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), GuiHandler)
    print(f"PID autotuner GUI listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


class GuiHandler(BaseHTTPRequestHandler):
    server_version = "PidAutotunerGui/0.2"
    _log_lock = threading.Lock()
    _log_window_start = time.monotonic()
    _log_counts: dict[str, int] = {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "time": time.time(),
                    "frontend": "react-vite",
                    "pid_programming": TUNING_SESSION.status()["pid_programming"],
                }
            )
            return
        if parsed.path == "/api/read":
            self._handle_read(parsed.query)
            return
        if parsed.path == "/api/inductance":
            self._handle_read_inductance(parsed.query)
            return
        if parsed.path == "/api/xdp-pid":
            self._handle_read_xdp_pid(parsed.query)
            return
        if parsed.path == "/api/pmbus-output":
            self._handle_read_pmbus_output(parsed.query)
            return
        if parsed.path == "/api/xdp-output":
            self._handle_read_xdp_output(parsed.query)
            return
        if parsed.path == "/api/power-supply":
            self._handle_read_power_supply(parsed.query)
            return
        if parsed.path == "/api/function-generator":
            self._handle_read_function_generator(parsed.query)
            return
        if parsed.path == "/api/scope":
            self._handle_scope_capture(parsed.query)
            return
        if parsed.path == "/api/tuning/status":
            self._send_json({"ok": True, **TUNING_SESSION.status()})
            return
        if parsed.path == "/api/tuning/config":
            self._send_json({"ok": True, "config": TUNING_SESSION.status()["config"]})
            return
        if parsed.path == "/api/self-test":
            self._handle_self_test(parsed.query)
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/vout":
            self._handle_set_vout()
            return
        if parsed.path == "/api/inductance":
            self._handle_set_inductance()
            return
        if parsed.path == "/api/xdp-pid":
            self._handle_set_xdp_pid()
            return
        if parsed.path == "/api/pmbus-output":
            self._handle_set_pmbus_output()
            return
        if parsed.path == "/api/xdp-output":
            self._handle_set_xdp_output()
            return
        if parsed.path == "/api/power-supply":
            self._handle_set_power_supply()
            return
        if parsed.path == "/api/function-generator":
            self._handle_set_function_generator()
            return
        if parsed.path == "/api/scope":
            self._handle_scope_capture_post()
            return
        if parsed.path == "/api/tuning/start":
            self._handle_tuning_start()
            return
        if parsed.path == "/api/tuning/stop":
            self._send_json({"ok": True, **TUNING_SESSION.stop()})
            return
        if parsed.path == "/api/tuning/step":
            self._handle_tuning_step()
            return
        if parsed.path == "/api/self-test":
            self._handle_self_test(parsed.query)
            return
        if parsed.path == "/api/bode/sweep":
            self._handle_bode_sweep()
            return
        self._send_json({"ok": False, "error": "Unknown endpoint."}, status=404)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        now = time.monotonic()
        parsed = urlparse(self.path)
        key = f"{self.command} {parsed.path}"
        with self._log_lock:
            self._log_counts[key] = self._log_counts.get(key, 0) + 1
            elapsed = now - self._log_window_start
            if elapsed < 5.0:
                return
            total = sum(self._log_counts.values())
            summary = ", ".join(f"{name}: {count}" for name, count in sorted(self._log_counts.items()))
            print(f"{self.address_string()} - {total} requests in {elapsed:.1f}s ({summary})", flush=True)
            self._log_counts = {}
            self._log_window_start = now

    def _handle_read(self, query: str) -> None:
        params = parse_qs(query)
        address = params.get("address", [DEFAULT_ADDRESS])[0]
        page = _int_param(params, "page", DEFAULT_PAGE)
        adapter = params.get("adapter", ["xdp"])[0]
        self._send_json(_read_status(address=address, page=page, adapter_kind=adapter))

    def _handle_set_vout(self) -> None:
        try:
            payload = self._read_json_body()
            voltage = float(payload["voltage"])
            address = str(payload.get("address", DEFAULT_ADDRESS))
            page = int(payload.get("page", DEFAULT_PAGE))
            adapter = str(payload.get("adapter", "xdp"))
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return
        self._send_json(_set_vout(address=address, page=page, voltage=voltage, adapter_kind=adapter))

    def _handle_read_inductance(self, query: str) -> None:
        params = parse_qs(query)
        address = params.get("address", [DEFAULT_ADDRESS])[0]
        page = _int_param(params, "page", DEFAULT_PAGE)
        adapter = params.get("adapter", ["xdp"])[0]
        self._send_json(_read_inductance(address=address, page=page, adapter_kind=adapter))

    def _handle_set_inductance(self) -> None:
        try:
            payload = self._read_json_body()
            address = str(payload.get("address", DEFAULT_ADDRESS))
            page = int(payload.get("page", DEFAULT_PAGE))
            adapter = str(payload.get("adapter", "xdp"))
            output = payload.get("output_inductance_nh")
            effective_lc = payload.get("effective_lc_inductance_nh")
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return
        self._send_json(
            _set_inductance(
                address=address,
                page=page,
                adapter_kind=adapter,
                output_inductance_nh=None if output is None else float(output),
                effective_lc_inductance_nh=None if effective_lc is None else float(effective_lc),
            )
        )

    def _handle_read_xdp_pid(self, query: str) -> None:
        params = parse_qs(query)
        address = params.get("address", [DEFAULT_ADDRESS])[0]
        page = _int_param(params, "page", DEFAULT_PAGE)
        adapter = params.get("adapter", ["xdp"])[0]
        self._send_json(_read_xdp_pid(address=address, page=page, adapter_kind=adapter))

    def _handle_set_xdp_pid(self) -> None:
        try:
            payload = self._read_json_body()
            address = str(payload.get("address", DEFAULT_ADDRESS))
            page = int(payload.get("page", DEFAULT_PAGE))
            adapter = str(payload.get("adapter", "xdp"))
            values = payload.get("values", {})
            if not isinstance(values, dict):
                raise ValueError("values must be an object")
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return
        self._send_json(_set_xdp_pid(address=address, page=page, adapter_kind=adapter, values=values))

    def _handle_read_pmbus_output(self, query: str) -> None:
        params = parse_qs(query)
        address = params.get("address", [DEFAULT_ADDRESS])[0]
        page = _int_param(params, "page", DEFAULT_PAGE)
        adapter = params.get("adapter", ["xdp"])[0]
        self._send_json(_read_pmbus_output(address=address, page=page, adapter_kind=adapter))

    def _handle_set_pmbus_output(self) -> None:
        try:
            payload = self._read_json_body()
            address = str(payload.get("address", DEFAULT_ADDRESS))
            page = int(payload.get("page", DEFAULT_PAGE))
            adapter = str(payload.get("adapter", "xdp"))
            action = str(payload["action"]).strip().lower()
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return
        self._send_json(_set_pmbus_output(address=address, page=page, adapter_kind=adapter, action=action))

    def _handle_read_xdp_output(self, query: str) -> None:
        params = parse_qs(query)
        address = params.get("address", [DEFAULT_ADDRESS])[0]
        page = _int_param(params, "page", DEFAULT_PAGE)
        adapter = params.get("adapter", ["xdp"])[0]
        self._send_json(_read_xdp_output(address=address, page=page, adapter_kind=adapter))

    def _handle_set_xdp_output(self) -> None:
        try:
            payload = self._read_json_body()
            address = str(payload.get("address", DEFAULT_ADDRESS))
            page = int(payload.get("page", DEFAULT_PAGE))
            adapter = str(payload.get("adapter", "xdp"))
            action = str(payload["action"]).strip().lower()
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return
        self._send_json(_set_xdp_output(address=address, page=page, adapter_kind=adapter, action=action))

    def _handle_tuning_start(self) -> None:
        try:
            payload = self._read_json_body()
            config = _config_from_payload(payload.get("config", payload))
            self._send_json({"ok": True, **TUNING_SESSION.start(config)})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_tuning_step(self) -> None:
        try:
            payload = self._read_json_body()
            config_payload = payload.get("config")
            config = _config_from_payload(config_payload) if config_payload else None
            self._send_json({"ok": True, **TUNING_SESSION.step(config)})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_self_test(self, query: str) -> None:
        params = parse_qs(query)
        config = InstrumentSelfTestConfig(
            afg_resource=params.get("afg", [InstrumentSelfTestConfig.afg_resource])[0],
            scope_resource=params.get("scope", [InstrumentSelfTestConfig.scope_resource])[0],
            power_supply_resource=params.get("power", [InstrumentSelfTestConfig.power_supply_resource])[0],
            bode_host=params.get("bode_host", [InstrumentSelfTestConfig.bode_host])[0],
            bode_port=_int_param(params, "bode_port", InstrumentSelfTestConfig.bode_port),
            bode_runner_path=params.get("bode_runner", [InstrumentSelfTestConfig.bode_runner_path])[0],
            bode_serial=params.get("bode_serial", [InstrumentSelfTestConfig.bode_serial])[0],
            board_address=params.get("board_address", [InstrumentSelfTestConfig.board_address])[0],
            board_page=_int_param(params, "board_page", InstrumentSelfTestConfig.board_page),
            board_adapter=params.get("board_adapter", [InstrumentSelfTestConfig.board_adapter])[0],
            timeout_ms=_int_param(params, "timeout_ms", InstrumentSelfTestConfig.timeout_ms),
        )
        device = params.get("device", [""])[0].strip()
        try:
            if device:
                self._send_json(run_single_instrument_self_test(device, config))
            else:
                self._send_json(run_instrument_self_test(config))
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_bode_sweep(self) -> None:
        try:
            payload = self._read_json_body()
            host = str(payload.get("host", "127.0.0.1"))
            port = int(payload.get("port", 5025))
            start_hz = float(payload.get("start_hz", 1000.0))
            stop_hz = float(payload.get("stop_hz", 1_000_000.0))
            points = int(payload.get("points", 201))
            bandwidth_hz = float(payload.get("bandwidth_hz", 300.0))
            source_dbm_raw = payload.get("source_dbm", 0.0)
            source_dbm = None if source_dbm_raw is None else float(source_dbm_raw)
            timeout_ms = int(payload.get("timeout_ms", 30000))
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return
        result = _run_bode_sweep(
            host=host,
            port=port,
            start_hz=start_hz,
            stop_hz=stop_hz,
            points=points,
            bandwidth_hz=bandwidth_hz,
            source_dbm=source_dbm,
            timeout_ms=timeout_ms,
        )
        self._send_json(result, status=200 if result.get("ok", False) else 202)

    def _handle_read_power_supply(self, query: str) -> None:
        params = parse_qs(query)
        resource = params.get("resource", [DEFAULT_POWER_SUPPLY_RESOURCE])[0]
        self._send_json(_read_power_supply(resource))

    def _handle_set_power_supply(self) -> None:
        try:
            payload = self._read_json_body()
            resource = str(payload.get("resource", DEFAULT_POWER_SUPPLY_RESOURCE))
            voltage = payload.get("voltage_v")
            current = payload.get("current_limit_a")
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return
        self._send_json(_set_power_supply(resource, voltage, current))

    def _handle_read_function_generator(self, query: str) -> None:
        params = parse_qs(query)
        resource = params.get("resource", [DEFAULT_AFG_RESOURCE])[0]
        channel = _int_param(params, "channel", 1)
        self._send_json(_read_function_generator(resource, channel))

    def _handle_set_function_generator(self) -> None:
        try:
            payload = self._read_json_body()
            resource = str(payload.get("resource", DEFAULT_AFG_RESOURCE))
            channel = int(payload.get("channel", 1))
            mode = str(payload.get("mode", "square"))
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return
        self._send_json(_set_function_generator(resource, channel, mode, payload))

    def _handle_scope_capture(self, query: str) -> None:
        params = parse_qs(query)
        resource = params.get("resource", [DEFAULT_SCOPE_RESOURCE])[0]
        channels = _list_param(params, "channels", ["CH1"])
        measurements = _list_param(params, "measurements", ["MEAN", "PK2PK"])
        points = _int_param(params, "points", 2000)
        self._send_json(_capture_scope(resource, channels, measurements, points))

    def _handle_scope_capture_post(self) -> None:
        try:
            payload = self._read_json_body()
            resource = str(payload.get("resource", DEFAULT_SCOPE_RESOURCE))
            channels = [str(item).upper() for item in payload.get("channels", ["CH1"])]
            measurements = [str(item).upper() for item in payload.get("measurements", ["MEAN", "PK2PK"])]
            points = int(payload.get("points", 2000))
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return
        self._send_json(_capture_scope(resource, channels, measurements, points))

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        return json.loads(body.decode("utf-8"))

    def _serve_static(self, path: str) -> None:
        static_dir = FRONTEND_DIST_DIR
        if not static_dir.exists():
            self._send_json(
                {
                    "ok": False,
                    "error": "React frontend build is missing. Run `cd gui/frontend && npm install && npm run build`.",
                },
                status=503,
            )
            return
        if path in {"", "/"}:
            target = static_dir / "index.html"
        else:
            target = (static_dir / path.lstrip("/")).resolve()
            if static_dir.resolve() not in target.parents:
                self.send_error(403)
                return
            if not target.exists():
                target = self._fallback_static_asset(static_dir, path) or static_dir / "index.html"
        if not target.exists() or not target.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _fallback_static_asset(self, static_dir: Path, path: str) -> Path | None:
        requested = Path(path.lstrip("/"))
        if requested.parts[:1] != ("assets",):
            return None
        suffix = requested.suffix.lower()
        if suffix not in {".js", ".css"}:
            return None
        assets_dir = static_dir / "assets"
        candidates = sorted(
            assets_dir.glob(f"*{suffix}"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        response_status = status if payload.get("ok", True) or status != 200 else 500
        self.send_response(response_status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def _read_status(address: str, page: int, adapter_kind: str) -> dict:
    with DEVICE_LOCK:
        board = None
        try:
            board = _connect_board(address, adapter_kind)
            telemetry = board.read_vout_telemetry(page)
            operation = _safe_read_operation(board, page)
            try:
                status_word = board.read_status_word()
            except Exception:
                status_word = None
            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "operation": _format_optional_byte(operation),
                "status_word": _format_optional_word(status_word),
                "vout_mode_raw": f"0x{telemetry['vout_mode_raw']:02X}",
                "vout_mode": telemetry["vout_mode"],
                "exponent": telemetry["exponent"],
                "vout_command_v": telemetry["vout_command_v"],
                "read_vout_v": telemetry["read_vout_v"],
                "read_iout_a": telemetry["read_iout_a"],
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page}
        finally:
            if board is not None:
                board.close()


def _set_vout(address: str, page: int, voltage: float, adapter_kind: str) -> dict:
    if page != 0:
        return {
            "ok": False,
            "error": "Refusing to apply XDP-style Vout write to nonzero page. Confirm Loop B safety first.",
            "address": address,
            "page": page,
            "requested_v": voltage,
        }
    with DEVICE_LOCK:
        board = None
        try:
            board = _connect_board(address, adapter_kind)
            operation_before = _safe_read_operation(board, page)
            raw_written = board.set_vout_command(voltage, page=page)
            board.set_operation(0x80, page=page)
            settled = _wait_for_vout(board, page=page, target_v=voltage)
            operation_after = _safe_read_operation(board, page)
            raw_mode, mode_name, exponent = board.read_vout_mode(page)
            vout_command = board.read_vout_command(page)
            read_vout = board.read_vout(page)
            read_iout = board.read_iout(page)
            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "requested_v": voltage,
                "raw_written": f"0x{raw_written:04X}",
                "operation_before": _format_optional_byte(operation_before),
                "operation_after": _format_optional_byte(operation_after),
                "operation_set": "0x80",
                "settled": settled,
                "vout_mode_raw": f"0x{raw_mode:02X}",
                "vout_mode": mode_name,
                "exponent": exponent,
                "vout_command_v": vout_command,
                "read_vout_v": read_vout,
                "read_iout_a": read_iout,
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page, "requested_v": voltage}
        finally:
            if board is not None:
                board.close()


def _read_inductance(address: str, page: int, adapter_kind: str) -> dict:
    with DEVICE_LOCK:
        board = None
        try:
            board = _connect_board(address, adapter_kind)
            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "output_inductance": board.read_output_inductance_nh(page),
                "effective_lc_inductance": board.read_effective_lc_inductance_nh(page),
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page}
        finally:
            if board is not None:
                board.close()


def _set_inductance(
    *,
    address: str,
    page: int,
    adapter_kind: str,
    output_inductance_nh: float | None,
    effective_lc_inductance_nh: float | None,
) -> dict:
    if output_inductance_nh is None and effective_lc_inductance_nh is None:
        return {"ok": False, "error": "No inductance value was provided.", "address": address, "page": page}
    with DEVICE_LOCK:
        board = None
        try:
            board = _connect_board(address, adapter_kind)
            writes = {}
            if output_inductance_nh is not None:
                writes["output_inductance"] = board.set_output_inductance_nh(output_inductance_nh, page)
            if effective_lc_inductance_nh is not None:
                writes["effective_lc_inductance"] = board.set_effective_lc_inductance_nh(
                    effective_lc_inductance_nh,
                    page,
                )
            readback = {
                "output_inductance": board.read_output_inductance_nh(page),
                "effective_lc_inductance": board.read_effective_lc_inductance_nh(page),
            }
            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "writes": writes,
                **readback,
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page}
        finally:
            if board is not None:
                board.close()


def _read_xdp_pid(address: str, page: int, adapter_kind: str) -> dict:
    with DEVICE_LOCK:
        board = None
        try:
            board = _connect_board(address, adapter_kind)
            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "pid_registers": board.read_mod0_pid_registers(page),
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page}
        finally:
            if board is not None:
                board.close()


def _set_xdp_pid(address: str, page: int, adapter_kind: str, values: dict) -> dict:
    if not values:
        return {"ok": False, "error": "No XDP PID register value was provided.", "address": address, "page": page}
    with DEVICE_LOCK:
        board = None
        try:
            board = _connect_board(address, adapter_kind)
            int_values = {str(name): int(value) for name, value in values.items()}
            write = board.set_mod0_pid_registers(int_values, page)
            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "write": write,
                "pid_registers": write["readback"],
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page, "requested": values}
        finally:
            if board is not None:
                board.close()


def _read_pmbus_output(address: str, page: int, adapter_kind: str) -> dict:
    with DEVICE_LOCK:
        board = None
        try:
            board = _connect_board(address, adapter_kind)
            operation = board.read_operation(page)
            on_off_config = board.read_on_off_config(page)
            status_word = board.read_status_word()
            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "operation": _format_optional_byte(operation),
                "on_off_config": _format_optional_byte(on_off_config),
                "status_word": _format_optional_word(status_word),
                "standard_commands": {
                    "OPERATION": "0x01",
                    "ON_OFF_CONFIG": "0x02",
                },
                "note": "PMBus OPERATION controls output on/off. ON_OFF_CONFIG is read-only in this GUI for safety.",
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page}
        finally:
            if board is not None:
                board.close()


def _set_pmbus_output(address: str, page: int, adapter_kind: str, action: str) -> dict:
    turn_on_actions = {"on", "high", "enable", "set", "set_bit7"}
    turn_off_actions = {"off", "low", "disable", "clear", "clear_bit7"}
    if action not in turn_on_actions | turn_off_actions:
        return {
            "ok": False,
            "error": "Unsupported PMBus output action. Use 'on' or 'off'.",
            "address": address,
            "page": page,
            "requested": action,
        }
    with DEVICE_LOCK:
        board = None
        try:
            board = _connect_board(address, adapter_kind)
            operation_before = board.read_operation(page)
            on_off_config = board.read_on_off_config(page)
            if action in turn_on_actions:
                operation_written = 0x80
            else:
                operation_written = 0x00
            board.set_operation(operation_written, page)
            operation_after = board.read_operation(page)
            status_word = board.read_status_word()
            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "requested": action,
                "operation_before": _format_optional_byte(operation_before),
                "operation_after": _format_optional_byte(operation_after),
                "operation_written": _format_optional_byte(operation_written),
                "operation_bit7_before": (operation_before >> 7) & 1,
                "operation_bit7_after": (operation_after >> 7) & 1,
                "on_off_config": _format_optional_byte(on_off_config),
                "status_word": _format_optional_word(status_word),
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page, "requested": action}
        finally:
            if board is not None:
                board.close()


def _read_xdp_output(address: str, page: int, adapter_kind: str) -> dict:
    with DEVICE_LOCK:
        board = None
        try:
            board = _connect_board(address, adapter_kind)
            readback = board.read_vren_state(page)
            operation = board.read_operation(page)
            status_word = board.read_status_word()
            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "method": "xdp_xv_en",
                "state": readback.get("state"),
                "readback": readback,
                "operation": _format_optional_byte(operation),
                "status_word": _format_optional_word(status_word),
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page}
        finally:
            if board is not None:
                board.close()


def _set_xdp_output(address: str, page: int, adapter_kind: str, action: str) -> dict:
    action_map = {
        "on": "high",
        "enable": "high",
        "high": "high",
        "off": "low",
        "disable": "low",
        "low": "low",
        "release": "release",
    }
    state = action_map.get(action)
    if state is None:
        return {
            "ok": False,
            "error": "Unsupported XDP output action. Use 'enable', 'disable', or 'release'.",
            "address": address,
            "page": page,
            "requested": action,
        }
    with DEVICE_LOCK:
        board = None
        try:
            board = _connect_board(address, adapter_kind)
            write = board.set_vren_state(state, page)
            operation = board.read_operation(page)
            status_word = board.read_status_word()
            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "method": "xdp_xv_en",
                "requested": action,
                "state_written": state,
                "state": write.get("readback", {}).get("state"),
                "write": write,
                "readback": write.get("readback"),
                "operation": _format_optional_byte(operation),
                "status_word": _format_optional_word(status_word),
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page, "requested": action}
        finally:
            if board is not None:
                board.close()


def _run_bode_sweep(
    *,
    host: str,
    port: int,
    start_hz: float,
    stop_hz: float,
    points: int,
    bandwidth_hz: float,
    source_dbm: float | None,
    timeout_ms: int,
) -> dict:
    started = time.perf_counter()
    if start_hz <= 0 or stop_hz <= start_hz:
        return {"ok": False, "error": "Bode sweep requires 0 < start_hz < stop_hz."}
    if not 2 <= points <= 2001:
        return {"ok": False, "error": "Bode sweep points must be between 2 and 2001."}

    if not _ensure_bode_listener(host, port):
        return {
            "ok": False,
            "error": (
                f"No Bode SCPI TCP listener is reachable at {host}:{port}. "
                "Open Bode Analyzer Suite or verify the SCPI runner path/serial."
            ),
            "host": host,
            "port": port,
            "config": {
                "start_hz": start_hz,
                "stop_hz": stop_hz,
                "points": points,
                "bandwidth_hz": bandwidth_hz,
                "source_dbm": source_dbm,
            },
            "duration_s": round(time.perf_counter() - started, 3),
            "timestamp": time.time(),
        }

    bode = None
    try:
        bode = BodeScpiClient(host=host, port=port, timeout_ms=timeout_ms)
        bode.connect()
        identity = bode.idn()
        try:
            bode.lock()
        except Exception:
            pass
        bode.configure_gain_phase(
            start_hz=start_hz,
            stop_hz=stop_hz,
            points=points,
            bandwidth_hz=bandwidth_hz,
            source_dbm=source_dbm,
        )
        data = bode.run_sweep()
        system_error = ""
        try:
            system_error = bode.get_error()
        except Exception:
            pass
        return {
            "ok": True,
            "identity": identity,
            "host": host,
            "port": port,
            "config": {
                "start_hz": start_hz,
                "stop_hz": stop_hz,
                "points": points,
                "bandwidth_hz": bandwidth_hz,
                "source_dbm": source_dbm,
            },
            "frequency_hz": data.frequency_hz,
            "magnitude_db": data.magnitude_db,
            "phase_deg": data.phase_deg,
            "system_error": system_error,
            "duration_s": round(time.perf_counter() - started, 3),
            "timestamp": time.time(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "host": host,
            "port": port,
            "config": {
                "start_hz": start_hz,
                "stop_hz": stop_hz,
                "points": points,
                "bandwidth_hz": bandwidth_hz,
                "source_dbm": source_dbm,
            },
            "duration_s": round(time.perf_counter() - started, 3),
            "timestamp": time.time(),
        }
    finally:
        if bode is not None:
            try:
                bode.unlock()
            except Exception:
                pass
            try:
                bode.close()
            except Exception:
                pass


def _ensure_bode_listener(host: str, port: int) -> bool:
    if _tcp_listener_present(port):
        return True
    runner = Path(DEFAULT_BODE_SCPI_RUNNER)
    if not runner.exists():
        return False
    try:
        _start_bode_scpi_runner(host, port)
    except Exception:
        return False
    deadline = time.time() + 8.0
    while time.time() < deadline:
        if _tcp_listener_present(port):
            return True
        time.sleep(0.25)
    return False


def _tcp_listener_present(port: int) -> bool:
    command = (
        "Get-NetTCPConnection "
        f"-LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue | "
        "Select-Object -First 1 -ExpandProperty LocalPort"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return completed.returncode == 0 and completed.stdout.strip() == str(int(port))


def _start_bode_scpi_runner(host: str, port: int) -> None:
    args = [
        "--ip-address",
        host,
        "--port",
        str(int(port)),
        "--serial",
        DEFAULT_BODE_SERIAL,
        "--logging-level",
        "Warning",
    ]
    command = (
        "Start-Process "
        f"-FilePath {_ps_quote(DEFAULT_BODE_SCPI_RUNNER)} "
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


def _ps_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _read_power_supply(resource: str) -> dict:
    with DEVICE_LOCK:
        supply = None
        try:
            supply = KeysightN5700PowerSupply(resource, timeout_ms=5000)
            supply.connect()
            readback = supply.readback()
            error = _visible_instrument_error(readback.error)
            return {
                "ok": True,
                "resource": resource,
                "identity": readback.identity,
                "output_enabled": readback.output_enabled,
                "voltage_setpoint_v": readback.voltage_setpoint_v,
                "current_limit_a": readback.current_limit_a,
                "measured_voltage_v": readback.measured_voltage_v,
                "measured_current_a": readback.measured_current_a,
                "error": error,
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "resource": resource, "error": str(exc), "timestamp": time.time()}
        finally:
            if supply is not None:
                supply.close()


def _set_power_supply(resource: str, voltage: object, current: object) -> dict:
    if voltage is None and current is None:
        return {"ok": False, "resource": resource, "error": "No voltage or current limit was provided."}
    with DEVICE_LOCK:
        supply = None
        try:
            supply = KeysightN5700PowerSupply(resource, timeout_ms=5000)
            supply.connect()
            if voltage is not None:
                supply.set_voltage(float(voltage))
            if current is not None:
                supply.set_current_limit(float(current))
            readback = supply.readback()
            error = _visible_instrument_error(readback.error)
            return {
                "ok": True,
                "resource": resource,
                "identity": readback.identity,
                "output_enabled": readback.output_enabled,
                "voltage_setpoint_v": readback.voltage_setpoint_v,
                "current_limit_a": readback.current_limit_a,
                "measured_voltage_v": readback.measured_voltage_v,
                "measured_current_a": readback.measured_current_a,
                "error": error,
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "resource": resource, "error": str(exc), "timestamp": time.time()}
        finally:
            if supply is not None:
                supply.close()


def _read_function_generator(resource: str, channel: int) -> dict:
    with DEVICE_LOCK:
        fg = None
        try:
            fg = FunctionGenerator(resource, timeout_ms=5000, output_channel=channel)
            fg.connect()
            return _function_generator_snapshot(fg, resource, channel)
        except Exception as exc:
            return {"ok": False, "resource": resource, "channel": channel, "error": str(exc), "timestamp": time.time()}
        finally:
            if fg is not None:
                fg.close()


def _set_function_generator(resource: str, channel: int, mode: str, payload: dict) -> dict:
    mode_name = mode.strip().lower()
    with DEVICE_LOCK:
        fg = None
        try:
            fg = FunctionGenerator(resource, timeout_ms=5000, output_channel=channel)
            fg.connect()
            if mode_name == "square":
                fg.configure_square_levels(
                    frequency_hz=float(payload.get("frequency_hz", 100000.0)),
                    low_v=float(payload.get("low_v", 0.365)),
                    high_v=float(payload.get("high_v", 2.0)),
                    duty_percent=float(payload.get("duty_percent", 50.0)),
                    channel=channel,
                )
            elif mode_name == "pulse":
                fg.configure_pulse_levels(
                    frequency_hz=float(payload.get("frequency_hz", 100000.0)),
                    low_v=float(payload.get("low_v", 0.365)),
                    high_v=float(payload.get("high_v", 2.0)),
                    width_s=_optional_float(payload.get("pulse_width_s")),
                    duty_percent=_optional_float(payload.get("duty_percent")),
                    channel=channel,
                )
            elif mode_name == "dc":
                fg.configure_dc(float(payload.get("dc_level_v", payload.get("offset_v", 0.0))), channel=channel)
            elif mode_name == "sine":
                fg.configure_sine(
                    frequency_hz=float(payload.get("frequency_hz", 100000.0)),
                    amplitude_vpp=float(payload.get("amplitude_vpp", 1.0)),
                    offset_v=float(payload.get("offset_v", 0.0)),
                    phase_deg=_optional_float(payload.get("phase_deg")),
                    channel=channel,
                )
            else:
                return {"ok": False, "resource": resource, "channel": channel, "error": f"Unsupported AFG mode: {mode}"}
            return _function_generator_snapshot(fg, resource, channel)
        except Exception as exc:
            return {"ok": False, "resource": resource, "channel": channel, "mode": mode, "error": str(exc), "timestamp": time.time()}
        finally:
            if fg is not None:
                fg.close()


def _function_generator_snapshot(fg: FunctionGenerator, resource: str, channel: int) -> dict:
    system_error = _visible_instrument_error(_safe_query(fg, "SYST:ERR?"))
    return {
        "ok": True,
        "resource": resource,
        "channel": channel,
        "identity": fg.idn(),
        "function": _safe_query(fg, f"SOUR{channel}:FUNC?"),
        "frequency_hz": _safe_float_query(fg, f"SOUR{channel}:FREQ?"),
        "amplitude_vpp": _safe_float_query(fg, f"SOUR{channel}:VOLT?"),
        "offset_v": _safe_float_query(fg, f"SOUR{channel}:VOLT:OFFS?"),
        "high_v": _safe_float_query(fg, f"SOUR{channel}:VOLT:HIGH?"),
        "low_v": _safe_float_query(fg, f"SOUR{channel}:VOLT:LOW?"),
        "phase_deg": _safe_float_query(fg, f"SOUR{channel}:PHAS?"),
        "duty_percent": _safe_float_query(fg, f"SOUR{channel}:FUNC:SQU:DCYC?"),
        "pulse_width_s": _safe_float_query(fg, f"SOUR{channel}:PULS:WIDT?"),
        "output": _safe_query(fg, f"OUTP{channel}?"),
        "system_error": system_error,
        "timestamp": time.time(),
    }


def _visible_instrument_error(message: str | None) -> str | None:
    if not message:
        return None
    normalized = message.strip().lower()
    if "no error" in normalized or normalized in {"0", "+0"}:
        return None
    return message


def _capture_scope(resource: str, channels: list[str], measurements: list[str], points: int) -> dict:
    safe_channels = [ch for ch in (item.strip().upper() for item in channels) if ch.startswith("CH")][:8] or ["CH1"]
    safe_measurements = [item.strip().upper() for item in measurements if item.strip()][:8]
    stop = max(10, min(100000, int(points)))
    with DEVICE_LOCK:
        scope = None
        try:
            scope = TektronixOscilloscope(resource, timeout_ms=15000)
            scope.connect()
            identity = scope.idn()
            waveforms = []
            measurement_rows = []
            for channel in safe_channels:
                capture = scope.capture_ascii_waveform(channel, start=1, stop=stop)
                waveforms.append(
                    {
                        "source": capture.source,
                        "x": capture.x,
                        "y": capture.y,
                        "x_unit": capture.x_unit,
                        "y_unit": capture.y_unit,
                    }
                )
                for measurement in safe_measurements:
                    try:
                        value = scope.read_immediate_measurement(channel, measurement)
                        measurement_rows.append({"source": channel, "measurement": measurement, "value": value, "ok": True})
                    except Exception as exc:
                        measurement_rows.append(
                            {"source": channel, "measurement": measurement, "value": None, "ok": False, "error": str(exc)}
                        )
            return {
                "ok": True,
                "resource": resource,
                "identity": identity,
                "channels": safe_channels,
                "measurements": safe_measurements,
                "waveforms": waveforms,
                "measurement_values": measurement_rows,
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "resource": resource, "error": str(exc), "timestamp": time.time()}
        finally:
            if scope is not None:
                scope.close()


def _safe_query(instrument, command: str) -> str | None:
    try:
        return instrument.query(command)
    except Exception:
        return None


def _safe_float_query(instrument, command: str) -> float | None:
    try:
        value = instrument.query(command)
        return float(str(value).strip().strip('"'))
    except Exception:
        return None


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _safe_read_operation(board, page: int) -> int | None:
    try:
        return board.read_operation(page)
    except Exception:
        return None


def _wait_for_vout(board, page: int, target_v: float, tolerance_v: float = 0.006, timeout_s: float = 2.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if abs(board.read_vout(page) - target_v) <= tolerance_v:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def _format_optional_byte(value: int | None) -> str | None:
    if value is None:
        return None
    return f"0x{value & 0xFF:02X}"


def _format_optional_word(value: int | None) -> str | None:
    if value is None:
        return None
    return f"0x{value & 0xFFFF:04X}"


def _connect_board(address: str, adapter_kind: str):
    adapter = create_i2c_adapter(adapter_kind, timeout_ms=3000)
    board = create_board_controller(
        "infineon_xdp",
        adapter,
        BoardControllerConfig(address=address, name="XDPE1A2G5C"),
    )
    board.connect()
    return board


def _int_param(params: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int(params.get(name, [str(default)])[0])
    except Exception:
        return default


def _list_param(params: dict[str, list[str]], name: str, default: list[str]) -> list[str]:
    raw_values = params.get(name)
    if not raw_values:
        return default
    values: list[str] = []
    for raw in raw_values:
        values.extend(part.strip() for part in raw.split(",") if part.strip())
    return values or default


def _loop_name(page: int) -> str:
    if page == 0:
        return "Loop A"
    if page == 1:
        return "Loop B"
    return f"Page {page}"


def _config_from_payload(payload: dict | None) -> TuningConfig:
    payload = payload or {}
    plant_payload = payload.get("plant", {})
    targets_payload = payload.get("targets", {})
    search_payload = payload.get("search", {})
    return TuningConfig(
        plant=PlantParams(
            vdc=_float_field(plant_payload, "vdc", 12.0),
            inductance_h=_float_field(plant_payload, "inductance_h", 30e-6),
            capacitance_f=_float_field(plant_payload, "capacitance_f", 15e-6),
            capacitor_esr_ohm=_float_field(plant_payload, "capacitor_esr_ohm", 7.5e-3),
            inductor_dcr_ohm=_float_field(plant_payload, "inductor_dcr_ohm", 50e-3),
        ),
        targets=TuningTargets(
            vout_target_v=_float_field(targets_payload, "vout_target_v", 0.9),
            overshoot_pct=_float_field(targets_payload, "overshoot_pct", 4.0),
            undershoot_pct=_float_field(targets_payload, "undershoot_pct", 4.0),
            settling_time_s=_float_field(targets_payload, "settling_time_s", 100e-6),
            oscillations=_int_field(targets_payload, "oscillations", 0),
        ),
        search=SearchSpace(
            wc_min_rad_s=_float_field(search_payload, "wc_min_rad_s", 94_248.0),
            wc_max_rad_s=_float_field(search_payload, "wc_max_rad_s", 314_159.0),
            phi_min_deg=_float_field(search_payload, "phi_min_deg", 30.0),
            phi_max_deg=_float_field(search_payload, "phi_max_deg", 80.0),
            initial_wc_rad_s=_float_field(search_payload, "initial_wc_rad_s", 157_080.0),
            initial_phi_deg=_float_field(search_payload, "initial_phi_deg", 60.0),
            max_iterations=_int_field(search_payload, "max_iterations", 40),
        ),
    )


def _float_field(payload: dict, name: str, default: float) -> float:
    try:
        return float(payload.get(name, default))
    except Exception:
        return default


def _int_field(payload: dict, name: str, default: int) -> int:
    try:
        return int(payload.get(name, default))
    except Exception:
        return default


if __name__ == "__main__":
    raise SystemExit(main())
