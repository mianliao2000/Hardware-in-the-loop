"""Local web GUI and API server for the hardware PID autotuner."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import mimetypes
from pathlib import Path
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
from hardware.instruments.i2c_adapters import create_i2c_adapter
from hardware.instruments.self_test import (
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
        self._send_json({"ok": False, "error": "Unknown endpoint."}, status=404)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

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
                target = static_dir / "index.html"
        if not target.exists() or not target.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        if target.name == "index.html":
            self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
        board = _connect_board(address, adapter_kind)
        try:
            raw_mode, mode_name, exponent = board.read_vout_mode(page)
            vout_command = board.read_vout_command(page)
            read_vout = board.read_vout(page)
            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "vout_mode_raw": f"0x{raw_mode:02X}",
                "vout_mode": mode_name,
                "exponent": exponent,
                "vout_command_v": vout_command,
                "read_vout_v": read_vout,
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page}
        finally:
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
        board = _connect_board(address, adapter_kind)
        try:
            operation_before = _safe_read_operation(board, page)
            raw_written = board.set_vout_command(voltage, page=page)
            board.set_operation(0x80, page=page)
            settled = _wait_for_vout(board, page=page, target_v=voltage)
            operation_after = _safe_read_operation(board, page)
            raw_mode, mode_name, exponent = board.read_vout_mode(page)
            vout_command = board.read_vout_command(page)
            read_vout = board.read_vout(page)
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
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page, "requested_v": voltage}
        finally:
            board.close()


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
