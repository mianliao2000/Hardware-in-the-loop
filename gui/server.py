"""Local web GUI and API server for the hardware PID autotuner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import copy
import json
import mimetypes
from pathlib import Path
import shutil
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_ADDRESS = "0x5E"
DEFAULT_PAGE = 0

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.instruments.board_controller import BoardControllerConfig, create_board_controller
from hardware.instruments.bode_analyzer import BodeScpiClient
from hardware.instruments.bode100 import Bode100Driver, Bode100Error, DEFAULT_BODE100_SCPI_RUNNER_PATH
from hardware.instruments.function_generator import FunctionGenerator
from hardware.instruments.i2c_adapters import create_i2c_adapter, reset_xdp_usb_bridges
from hardware.instruments.oscilloscope import TektronixOscilloscope
from hardware.instruments.power_supply import KeysightN5700PowerSupply
from hardware.instruments.self_test import (
    DEFAULT_AFG_RESOURCE,
    DEFAULT_POWER_SUPPLY_RESOURCE,
    DEFAULT_SCOPE_RESOURCE,
    InstrumentSelfTestConfig,
    run_instrument_self_test,
    run_single_instrument_self_test,
)
from hardware.instruments.visa_resource import VisaConnectionError
from hardware.tuning import (
    AutotuneExperimentConfig,
    ExperimentResult,
    HardwarePidCandidate,
    PidAutotuneSession,
    PlantParams,
    ResponseAnalyzer,
    SearchSpace,
    SearchParameter,
    TuningConfig,
    TuningTargets,
    Waveform,
)


DEVICE_LOCK = threading.Lock()
ARTIFACT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="autotune-artifact")
FRONTEND_DIST_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
SCOPE_CONNECTIONS: dict[str, TektronixOscilloscope] = {}
SCOPE_CAPTURE_DIR = ROOT / "data" / "scope_captures"
BODE_SWEEP_DIR = ROOT / "data" / "bode_sweeps"
RESULTS_DIR = ROOT / "results"
LATEST_SCOPE_PNG = RESULTS_DIR / "latest_scope_capture.png"
LATEST_BODE_PNG = RESULTS_DIR / "latest_bode_sweep.png"
AUTOTUNE_RUN_DIR = RESULTS_DIR / "autotune_runs"
AUTOTUNE_RECENT_DIR = AUTOTUNE_RUN_DIR / "recent"
AUTOTUNE_SAVED_DIR = AUTOTUNE_RUN_DIR / "saved"
AUTOTUNE_RECENT_LIMIT = 5
SCOPE_CAPTURE_CACHE: dict[str, dict] = {}
SCOPE_CAPTURE_CACHE_LIMIT = 1
SCOPE_DISPLAY_MAX_POINTS = 200_000
SCOPE_TRIGGER_OFFSET_FROM_LEFT_S = 2e-6
DEFAULT_SCOPE_AXIS_SETTINGS = {
    "leftMin": -0.5,
    "leftMax": 3.0,
    "rightMin": 0.7,
    "rightMax": 1.1,
    "channelAxes": {
        "CH1": "left",
        "CH2": "left",
        "CH3": "right",
        "CH4": "right",
        "CH5": "left",
        "CH6": "left",
        "CH7": "right",
        "CH8": "right",
    },
}


class ServerHardwareExperimentRunner:
    """Run one real hardware tuning candidate through the existing bench APIs."""

    def evaluate(
        self,
        candidate: HardwarePidCandidate,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
    ) -> ExperimentResult:
        started = time.perf_counter()
        stage_started = started
        stage_durations: dict[str, float] = {}

        def mark_stage(name: str) -> None:
            nonlocal stage_started
            now = time.perf_counter()
            stage_durations[name] = round(now - stage_started, 3)
            stage_started = now

        if experiment.board_page != 0:
            raise RuntimeError("Auto-tune writes are only enabled for Loop A / page 0.")
        if not experiment.enable_bode_analysis and not experiment.enable_transient_analysis:
            raise RuntimeError("Select at least one analysis mode before starting Auto-Tune.")

        write_results: dict[str, object] = {}
        bode_result: dict[str, object] = {}
        scope_result: dict[str, object] = {}
        waveform: Waveform | None = None
        async_artifacts = bool(getattr(experiment, "async_artifacts", False))
        afg_enabled = False
        fg_config = dict(experiment.function_generator_config)
        fg_resource = str(fg_config.get("resource", DEFAULT_AFG_RESOURCE))
        fg_channel = int(fg_config.get("channel", 1))
        fg_mode = str(fg_config.get("mode", "square"))
        try:
            vout_write = _prepare_vout_for_autotune(
                address=experiment.board_address,
                page=experiment.board_page,
                adapter_kind=experiment.board_adapter,
                voltage=config.targets.vout_target_v,
                tolerance_v=experiment.vout_tolerance_v,
            )
            write_results["vout"] = vout_write
            mark_stage("write_vout")
            if not vout_write.get("ok"):
                raise RuntimeError(f"VOUT write failed: {vout_write.get('error')}")
            if str(vout_write.get("vout_mode", "")).lower() != "linear":
                raise RuntimeError(f"Vout safety check failed: invalid VOUT_MODE {vout_write.get('vout_mode')}.")
            read_vout = _optional_float_value(vout_write.get("read_vout_v"))
            if read_vout is not None and abs(read_vout - config.targets.vout_target_v) > experiment.vout_tolerance_v:
                raise RuntimeError(
                    f"Vout safety check failed: read {read_vout:.4f} V, target {config.targets.vout_target_v:.4f} V."
                )

            pid_write = _set_xdp_pid(
                address=experiment.board_address,
                page=experiment.board_page,
                adapter_kind=experiment.board_adapter,
                values=candidate.pid_values(),
            )
            write_results["xdp_pid"] = pid_write
            mark_stage("write_pid")
            if not pid_write.get("ok"):
                raise RuntimeError(f"PID write failed: {pid_write.get('error')}")

            inductance_write = _set_inductance(
                address=experiment.board_address,
                page=experiment.board_page,
                adapter_kind=experiment.board_adapter,
                output_inductance_nh=candidate.output_inductance_nh,
                effective_lc_inductance_nh=candidate.effective_lc_inductance_nh,
            )
            write_results["inductance"] = inductance_write
            mark_stage("write_inductance")
            if not inductance_write.get("ok"):
                raise RuntimeError(f"Inductance write failed: {inductance_write.get('error')}")

            if experiment.enable_bode_analysis:
                bode_cfg = dict(experiment.bode_config)
                bode_result = _run_bode_sweep(
                    host=str(bode_cfg.get("host", "127.0.0.1")),
                    port=int(bode_cfg.get("port", 5025)),
                    start_hz=float(bode_cfg.get("start_hz", 1000.0)),
                    stop_hz=float(bode_cfg.get("stop_hz", 1_000_000.0)),
                    points=int(bode_cfg.get("points", 201)),
                    bandwidth_hz=float(bode_cfg.get("bandwidth_hz", 300.0)),
                    source_dbm=None if bode_cfg.get("source_dbm") is None else float(bode_cfg.get("source_dbm", 0.0)),
                    timeout_ms=int(bode_cfg.get("timeout_ms", 60000)),
                    async_artifacts=async_artifacts,
                )
                mark_stage("bode_sweep")
                if not bode_result.get("ok"):
                    raise RuntimeError(f"Bode sweep failed: {bode_result.get('error')}")

            if experiment.enable_transient_analysis:
                fg_apply = _set_function_generator(fg_resource, fg_channel, fg_mode, fg_config)
                write_results["function_generator_apply"] = fg_apply
                mark_stage("function_generator_apply")
                if not fg_apply.get("ok"):
                    raise RuntimeError(f"Function generator setup failed: {fg_apply.get('error')}")
                fg_on = _set_function_generator(fg_resource, fg_channel, fg_mode, {"output_enabled": True})
                write_results["function_generator_on"] = fg_on
                mark_stage("function_generator_enable")
                if not fg_on.get("ok"):
                    raise RuntimeError(f"Function generator enable failed: {fg_on.get('error')}")
                afg_enabled = True
                time.sleep(0.12)
                mark_stage("transient_settle_delay")

                scope_cfg = dict(experiment.scope_config)
                response_channel = experiment.response_channel.strip().upper() or "CH3"
                scope_channels = [str(ch).upper() for ch in scope_cfg.get("channels", ["CH1", response_channel])]
                for required in ("CH1", response_channel):
                    if required not in scope_channels:
                        scope_channels.append(required)
                scope_result = _capture_scope(
                    resource=str(scope_cfg.get("resource", DEFAULT_SCOPE_RESOURCE)),
                    channels=scope_channels,
                    measurements=[str(item).upper() for item in scope_cfg.get("measurements", [])],
                    points=None,
                    function_generator_frequency_hz=_optional_float_value(fg_config.get("frequency_hz")),
                    scope_axis_settings=_normalize_scope_axis_settings(scope_cfg.get("scope_axis_settings")),
                    async_artifacts=async_artifacts,
                )
                mark_stage("scope_capture")
                if not scope_result.get("ok"):
                    raise RuntimeError(f"Scope capture failed: {scope_result.get('error')}")

                waveform = _response_waveform_from_scope(scope_result, response_channel)
                _enforce_scope_response_safety(waveform, config.targets.vout_target_v, experiment.response_abs_limit_v)

            metrics = ResponseAnalyzer(config.targets).analyze_hardware(
                waveform,
                bode_result.get("margins") if bode_result else None,
                enable_transient=experiment.enable_transient_analysis,
                enable_bode=experiment.enable_bode_analysis,
            )
            mark_stage("metrics")
            write_results["stage_durations_s"] = dict(stage_durations)
            return ExperimentResult(
                waveform=waveform or Waveform(time_s=[], vout_v=[]),
                metrics=metrics,
                write_results=write_results,
                bode_result=_compact_bode_result(bode_result) if bode_result else {"skipped": True},
                scope_result=_compact_scope_result(scope_result) if scope_result else {"skipped": True},
                duration_s=time.perf_counter() - started,
            )
        finally:
            if afg_enabled:
                off_started = time.perf_counter()
                off_result = _set_function_generator(fg_resource, fg_channel, fg_mode, {"output_enabled": False})
                write_results["function_generator_off"] = off_result
                stage_durations["function_generator_disable"] = round(time.perf_counter() - off_started, 3)
            write_results["stage_durations_s"] = dict(stage_durations)


TUNING_SESSION = PidAutotuneSession(experiment_runner=ServerHardwareExperimentRunner())


class AutotuneRunStore:
    """Keep a rolling local history of hardware auto-tune runs."""

    def __init__(self, recent_dir: Path, saved_dir: Path, recent_limit: int):
        self.recent_dir = recent_dir
        self.saved_dir = saved_dir
        self.recent_limit = recent_limit
        self._lock = threading.RLock()
        self._current_run_id: str | None = None
        self._persisted_iterations = 0

    def start_new(self, status: dict | None = None) -> dict:
        with self._lock:
            self.recent_dir.mkdir(parents=True, exist_ok=True)
            run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            run_dir = self.recent_dir / run_id
            (run_dir / "files").mkdir(parents=True, exist_ok=True)
            self._current_run_id = run_id
            self._persisted_iterations = 0
            initial_status = copy.deepcopy(status) if isinstance(status, dict) else {}
            initial_status["run"] = self._run_payload(run_id, "recent", run_dir)
            self._write_json(run_dir / "run_status.json", initial_status)
            self._write_summary(run_dir, initial_status)
            self._enforce_recent_limit()
            return initial_status

    def persist_status(self, status: dict) -> dict:
        with self._lock:
            next_status = copy.deepcopy(status)
            history = next_status.get("history")
            if not isinstance(history, list):
                history = []
                next_status["history"] = history
            if self._current_run_id is None:
                self.start_new(next_status)
            run_id = self._current_run_id
            if run_id is None:
                return next_status
            run_dir = self.recent_dir / run_id
            (run_dir / "files").mkdir(parents=True, exist_ok=True)
            next_status["run"] = self._run_payload(run_id, "recent", run_dir)
            axis_settings = _scope_axis_settings_from_status(next_status)
            for record in history:
                if isinstance(record, dict):
                    self._copy_record_assets(record, run_dir, scope_axis_settings=axis_settings)
            for key in ("current", "best"):
                record = next_status.get(key)
                if isinstance(record, dict):
                    self._copy_record_assets(record, run_dir, scope_axis_settings=axis_settings)
            if len(history) > self._persisted_iterations:
                with (run_dir / "iterations.jsonl").open("a", encoding="utf-8") as handle:
                    for record in history[self._persisted_iterations :]:
                        handle.write(json.dumps(record, indent=None, ensure_ascii=False) + "\n")
                self._persisted_iterations = len(history)
            self._write_json(run_dir / "run_status.json", next_status)
            self._write_summary(run_dir, next_status)
            self._auto_save_completion_gif(run_dir, next_status)
            self._enforce_recent_limit()
            return next_status

    def stop_current(self) -> None:
        with self._lock:
            self._current_run_id = None
            self._persisted_iterations = 0

    def archive_current(self, name: str | None = None) -> dict:
        with self._lock:
            if self._current_run_id is None:
                status = TUNING_SESSION.status()
                self.persist_status(status)
            if self._current_run_id is None:
                raise RuntimeError("No auto-tune run is available to save.")
            source = self.recent_dir / self._current_run_id
            if not source.exists():
                raise RuntimeError("Current auto-tune result folder does not exist yet.")
            safe_name = _safe_file_stem(name or self._current_run_id)
            target = self.saved_dir / safe_name
            if target.exists():
                target = self.saved_dir / f"{safe_name}_{uuid.uuid4().hex[:6]}"
            self.saved_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
            summary = self._read_json(target / "summary.json") or {}
            summary.update({
                "run_id": target.name,
                "source_run_id": self._current_run_id,
                "kind": "saved",
                "archived_at": time.time(),
                "path": _path_label(target),
            })
            self._write_json(target / "summary.json", summary)
            return {"ok": True, "saved_run": summary}

    def archive_run(self, run_id: str, kind: str = "recent", name: str | None = None) -> dict:
        with self._lock:
            source = self._run_dir(kind, run_id)
            if not source.exists():
                raise RuntimeError("Selected auto-tune result folder does not exist.")
            status = self._read_json(source / "run_status.json")
            if not isinstance(status, dict):
                raise RuntimeError(f"No saved status was found for run '{run_id}'.")
            safe_name = _safe_file_stem(name or run_id)
            target = self.saved_dir / safe_name
            if target.exists():
                target = self.saved_dir / f"{safe_name}_{uuid.uuid4().hex[:6]}"
            self.saved_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
            status["run"] = self._run_payload(target.name, "saved", target)
            self._write_json(target / "run_status.json", status)
            self._write_summary(target, status)
            summary = self._read_json(target / "summary.json") or {}
            summary.update({
                "run_id": target.name,
                "source_run_id": run_id,
                "source_kind": kind,
                "kind": "saved",
                "archived_at": time.time(),
                "path": _path_label(target),
            })
            self._write_json(target / "summary.json", summary)
            return {"ok": True, "saved_run": summary}

    def delete_run(self, run_id: str, kind: str = "recent") -> dict:
        with self._lock:
            run_dir = self._run_dir(kind, run_id)
            if kind == "recent" and run_dir.name == self._current_run_id:
                self.stop_current()
            shutil.rmtree(run_dir, ignore_errors=True)
            return {"ok": True, "deleted_run_id": run_id, "kind": kind}

    def list_runs(self) -> dict:
        with self._lock:
            return {
                "ok": True,
                "current_run_id": self._current_run_id,
                "recent": self._list_kind(self.recent_dir, "recent"),
                "saved": self._list_kind(self.saved_dir, "saved"),
            }

    def load_run(self, run_id: str, kind: str = "recent") -> dict:
        with self._lock:
            run_dir = self._run_dir(kind, run_id)
            status = self._read_json(run_dir / "run_status.json")
            if not isinstance(status, dict):
                raise RuntimeError(f"No saved status was found for run '{run_id}'.")
            saved_state = str(status.get("state") or "")
            if saved_state in {"running", "paused"}:
                status = copy.deepcopy(status)
                status["loaded_state"] = saved_state
                status["state"] = "stopped"
                status["can_resume_saved_run"] = True
                status["message"] = "Loaded saved result snapshot. Press Resume to continue from the next candidate, or load another result."
            status["run"] = self._run_payload(run_dir.name, kind, run_dir)
            return {"ok": True, **status}

    def resume_run(self, run_id: str, kind: str = "recent") -> dict:
        with self._lock:
            source = self._run_dir(kind, run_id)
            status = self._read_json(source / "run_status.json")
            if not isinstance(status, dict):
                raise RuntimeError(f"No saved status was found for run '{run_id}'.")
            if str(status.get("state") or "") == "complete":
                raise RuntimeError("Selected auto-tune result is already complete.")
            if kind == "saved":
                self.recent_dir.mkdir(parents=True, exist_ok=True)
                target = self.recent_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_resume_{uuid.uuid4().hex[:6]}"
                shutil.copytree(source, target)
                run_dir = target
            else:
                run_dir = source
            self._current_run_id = run_dir.name
            self._persisted_iterations = len(status.get("history") if isinstance(status.get("history"), list) else [])
            status = copy.deepcopy(status)
            status["run"] = self._run_payload(run_dir.name, "recent", run_dir)
            restored = TUNING_SESSION.restore(status)
            resumed = TUNING_SESSION.resume()
            resumed = self.persist_status(resumed)
            self._enforce_recent_limit()
            return {"ok": True, "restored": restored, **resumed}

    def save_animation_gif(self, run_id: str | None = None, kind: str = "recent", duration_ms: int = 100) -> dict:
        with self._lock:
            if run_id is None:
                if self._current_run_id is None:
                    status = TUNING_SESSION.status()
                    self.persist_status(status)
                run_id = self._current_run_id
                kind = "recent"
            if run_id is None:
                raise RuntimeError("No auto-tune run is available for GIF export.")
            run_dir = self._run_dir(kind, run_id)
            status = self._read_json(run_dir / "run_status.json")
            if not isinstance(status, dict):
                raise RuntimeError(f"No saved status was found for run '{run_id}'.")
            history = status.get("history") if isinstance(status.get("history"), list) else []
            output_dir = run_dir / "animations"
            output_dir.mkdir(parents=True, exist_ok=True)
            duration_ms = max(50, min(5000, int(duration_ms)))
            combined = self._make_combined_gif(history, target=output_dir / "combined_response_bode.gif", duration_ms=duration_ms)
            generated_at = time.time()
            return {
                "ok": True,
                "run_id": run_id,
                "kind": kind,
                "generated_at": generated_at,
                "duration_ms": duration_ms,
                "animation_dir": str(output_dir.relative_to(ROOT)).replace("\\", "/"),
                "combined_gif": _scope_png_public_path(combined) if combined else None,
                "transient_gif": None,
                "bode_gif": None,
            }

    def _auto_save_completion_gif(self, run_dir: Path, status: dict) -> None:
        if status.get("state") != "complete":
            return
        history = status.get("history") if isinstance(status.get("history"), list) else []
        if len(history) < 2:
            return
        output_dir = run_dir / "animations"
        target = output_dir / "combined_response_bode.gif"
        if target.exists():
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        self._make_combined_gif(history, target=target)

    def rebuild_all_run_images(self) -> dict:
        """Rebuild stored run PNG/GIF artifacts from saved numeric data."""

        rebuilt_runs = 0
        rebuilt_images = 0
        with self._lock:
            for folder in (self.recent_dir, self.saved_dir):
                if not folder.exists():
                    continue
                for run_dir in folder.iterdir():
                    if not run_dir.is_dir():
                        continue
                    status = self._read_json(run_dir / "run_status.json")
                    if not isinstance(status, dict):
                        continue
                    axis_settings = _scope_axis_settings_from_status(status)
                    changed = False
                    history = status.get("history") if isinstance(status.get("history"), list) else []
                    for record in history:
                        if isinstance(record, dict):
                            if self._copy_record_assets(
                                record,
                                run_dir,
                                scope_axis_settings=axis_settings,
                                force_rebuild=True,
                            ):
                                changed = True
                                rebuilt_images += 1
                    for key in ("current", "best"):
                        record = status.get(key)
                        if isinstance(record, dict):
                            if self._copy_record_assets(
                                record,
                                run_dir,
                                scope_axis_settings=axis_settings,
                                force_rebuild=True,
                            ):
                                changed = True
                    if history:
                        output_dir = run_dir / "animations"
                        output_dir.mkdir(parents=True, exist_ok=True)
                        if self._make_combined_gif(history, target=output_dir / "combined_response_bode.gif", duration_ms=100):
                            changed = True
                    if changed:
                        self._write_json(run_dir / "run_status.json", status)
                        self._write_summary(run_dir, status)
                        rebuilt_runs += 1
        return {"runs": rebuilt_runs, "images": rebuilt_images}

    def _make_gif(self, history: list, *, result_key: str, image_key: str, target: Path, duration_ms: int = 800) -> Path | None:
        from PIL import Image

        image_paths: list[Path] = []
        for record in history:
            if not isinstance(record, dict):
                continue
            result = record.get(result_key)
            if not isinstance(result, dict):
                continue
            path = _path_from_result_reference(result.get(image_key))
            if path and path.exists():
                image_paths.append(path)
        unique_paths = []
        seen = set()
        for path in image_paths:
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                unique_paths.append(path)
        if not unique_paths:
            return None
        frames = [Image.open(path).convert("RGB") for path in unique_paths]
        try:
            frames[0].save(
                target,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=True,
            )
        finally:
            for frame in frames:
                frame.close()
        return target

    def _make_combined_gif(self, history: list, *, target: Path, duration_ms: int = 100) -> Path | None:
        from PIL import Image

        frames: list[Image.Image] = []
        try:
            for record in history:
                if not isinstance(record, dict):
                    continue
                scope_path = _path_from_result_reference((record.get("scope_result") or {}).get("scope_png") if isinstance(record.get("scope_result"), dict) else None)
                bode_path = _path_from_result_reference((record.get("bode_result") or {}).get("bode_png") if isinstance(record.get("bode_result"), dict) else None)
                if not scope_path or not bode_path or not scope_path.exists() or not bode_path.exists():
                    continue
                with Image.open(scope_path) as scope_image, Image.open(bode_path) as bode_image:
                    scope = scope_image.convert("RGB")
                    bode = bode_image.convert("RGB")
                    width = max(scope.width, bode.width)
                    if scope.width != width:
                        scope = scope.resize((width, max(1, round(scope.height * width / scope.width))))
                    if bode.width != width:
                        bode = bode.resize((width, max(1, round(bode.height * width / bode.width))))
                    combined = Image.new("RGB", (width, scope.height + bode.height), "white")
                    combined.paste(scope, (0, 0))
                    combined.paste(bode, (0, scope.height))
                    frames.append(combined)
            if not frames:
                return None
            if target.exists():
                target.unlink()
            frames[0].save(
                target,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=True,
            )
            return target
        finally:
            for frame in frames:
                frame.close()

    def _copy_record_assets(
        self,
        record: dict,
        run_dir: Path,
        *,
        scope_axis_settings: dict | None = None,
        force_rebuild: bool = False,
    ) -> bool:
        iteration = int(record.get("iteration") or 0)
        if iteration <= 0:
            return False
        files_dir = run_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)
        changed = False
        scope_result = record.get("scope_result")
        if isinstance(scope_result, dict):
            target = files_dir / f"iteration_{iteration:03d}_scope.png"
            if self._copy_scope_channel_data_files(scope_result, files_dir, iteration):
                changed = True
            if self._rebuild_scope_png_from_record(scope_result, target, iteration, scope_axis_settings):
                changed = True
            else:
                source = _path_from_result_reference(scope_result.get("scope_png"))
                if source and source.exists():
                    if source.resolve() != target.resolve():
                        shutil.copy2(source, target)
                        changed = True
                    scope_result["scope_png"] = _scope_png_public_path(target)
            record["scope_result"] = scope_result

        bode_result = record.get("bode_result")
        if isinstance(bode_result, dict):
            target_png = files_dir / f"iteration_{iteration:03d}_bode.png"
            data_source = _path_from_result_reference(bode_result.get("data_file"))
            target_data = None
            if data_source and data_source.exists():
                target_data = files_dir / f"iteration_{iteration:03d}_bode{data_source.suffix}"
                if data_source.resolve() != target_data.resolve():
                    shutil.copy2(data_source, target_data)
                    changed = True
                bode_result["data_file"] = str(target_data.relative_to(ROOT)).replace("\\", "/")
            if self._rebuild_bode_png_from_data_file(
                target_data or _path_from_result_reference(bode_result.get("data_file")),
                target_png,
                bode_result.get("margins") if isinstance(bode_result.get("margins"), dict) else None,
                iteration,
                force_rebuild=force_rebuild,
            ):
                changed = True
                bode_result["bode_png"] = _scope_png_public_path(target_png)
                bode_result["bode_png_pending"] = False
                bode_result["bode_png_error"] = None
            else:
                source = _path_from_result_reference(bode_result.get("bode_png"))
                if source and source.exists():
                    if source.resolve() != target_png.resolve():
                        shutil.copy2(source, target_png)
                        changed = True
                    bode_result["bode_png"] = _scope_png_public_path(target_png)
            record["bode_result"] = bode_result
        return changed

    def _copy_scope_channel_data_files(self, scope_result: dict, files_dir: Path, iteration: int) -> bool:
        changed = False
        waveforms = scope_result.get("waveforms") if isinstance(scope_result.get("waveforms"), list) else []
        for waveform in waveforms:
            if not isinstance(waveform, dict):
                continue
            source_label = str(waveform.get("source") or "CH").upper()
            data_source = _path_from_result_reference(waveform.get("data_file"))
            if not data_source or not data_source.exists():
                continue
            target = files_dir / f"iteration_{iteration:03d}_scope_{_safe_file_stem(source_label)}{data_source.suffix}"
            if data_source.resolve() != target.resolve():
                shutil.copy2(data_source, target)
                changed = True
            waveform["data_file"] = str(target.relative_to(ROOT)).replace("\\", "/")
        return changed

    def _rebuild_scope_png_from_record(
        self,
        scope_result: dict,
        target: Path,
        iteration: int,
        axis_settings: dict | None,
    ) -> bool:
        entry = _scope_capture_entry_from_result(scope_result)
        if entry is None:
            return False
        _plot_full_scope_capture_png(
            entry,
            axis_settings,
            path=target,
            title=f"Iteration {iteration} - Scope Capture - Full Data",
        )
        scope_result["scope_png"] = _scope_png_public_path(target)
        scope_result["scope_png_pending"] = False
        scope_result["scope_png_error"] = None
        return True

    def _rebuild_bode_png_from_data_file(
        self,
        data_file: Path | None,
        target: Path,
        margins: dict | None,
        iteration: int,
        *,
        force_rebuild: bool = False,
    ) -> bool:
        if not data_file or not data_file.exists():
            return False
        with np.load(data_file, allow_pickle=True) as payload:
            _plot_full_bode_sweep_png(
                frequency_hz=np.asarray(payload["frequency_hz"], dtype=np.float64).tolist(),
                magnitude_db=np.asarray(payload["magnitude_db"], dtype=np.float64).tolist(),
                phase_deg=np.asarray(payload["phase_deg"], dtype=np.float64).tolist(),
                margins=margins,
                path=target,
                title=f"Iteration {iteration} - Bode 100 Sweep - Full Data",
            )
        return True

    def _write_summary(self, run_dir: Path, status: dict) -> None:
        history = status.get("history") if isinstance(status.get("history"), list) else []
        best = status.get("best") if isinstance(status.get("best"), dict) else None
        current = status.get("current") if isinstance(status.get("current"), dict) else None
        summary = {
            "run_id": run_dir.name,
            "kind": "recent" if run_dir.parent == self.recent_dir else "saved",
            "path": _path_label(run_dir),
            "updated_at": time.time(),
            "state": status.get("state"),
            "message": status.get("message"),
            "iteration_count": len(history),
            "current_iteration": current.get("iteration") if current else None,
            "best_iteration": best.get("iteration") if best else None,
            "best_penalty": (best.get("metrics") or {}).get("score") if best else None,
        }
        existing = self._read_json(run_dir / "summary.json") or {}
        if "created_at" not in existing:
            existing["created_at"] = time.time()
        existing.update(summary)
        self._write_json(run_dir / "summary.json", existing)

    def _list_kind(self, folder: Path, kind: str) -> list[dict]:
        if not folder.exists():
            return []
        runs = []
        for run_dir in folder.iterdir():
            if not run_dir.is_dir():
                continue
            summary = self._read_json(run_dir / "summary.json") or {"run_id": run_dir.name}
            status = self._read_json(run_dir / "run_status.json")
            history = status.get("history") if isinstance(status, dict) and isinstance(status.get("history"), list) else None
            if history is not None and int(summary.get("iteration_count") or 0) != len(history):
                summary["iteration_count"] = len(history)
                self._write_json(run_dir / "summary.json", summary)
            summary["run_id"] = run_dir.name
            summary["kind"] = kind
            runs.append(summary)
        runs.sort(key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0), reverse=True)
        return runs

    def _enforce_recent_limit(self) -> None:
        runs = self._list_kind(self.recent_dir, "recent")
        for item in runs[self.recent_limit :]:
            run_id = str(item.get("run_id", ""))
            if run_id and run_id != self._current_run_id:
                shutil.rmtree(self.recent_dir / run_id, ignore_errors=True)

    def _run_dir(self, kind: str, run_id: str) -> Path:
        folder = self.saved_dir if kind == "saved" else self.recent_dir
        safe_id = Path(str(run_id)).name
        run_dir = folder / safe_id
        if not run_dir.exists() or not run_dir.is_dir():
            raise RuntimeError(f"Auto-tune result '{run_id}' was not found.")
        return run_dir

    def _run_payload(self, run_id: str, kind: str, run_dir: Path) -> dict:
        return {
            "run_id": run_id,
            "kind": kind,
            "path": _path_label(run_dir),
            "recent_limit": self.recent_limit,
        }

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None


AUTOTUNE_RUN_STORE = AutotuneRunStore(AUTOTUNE_RECENT_DIR, AUTOTUNE_SAVED_DIR, AUTOTUNE_RECENT_LIMIT)


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
        if parsed.path.startswith("/api/scope/capture/") and parsed.path.endswith("/full"):
            self._handle_scope_capture_full(parsed.path, parsed.query)
            return
        if parsed.path == "/api/instruments/bode100/idn":
            self._handle_bode100_idn(parsed.query)
            return
        if parsed.path == "/api/tuning/status":
            status = TUNING_SESSION.status()
            self._send_json({"ok": True, **status})
            return
        if parsed.path == "/api/tuning/config":
            self._send_json({"ok": True, "config": TUNING_SESSION.status()["config"]})
            return
        if parsed.path == "/api/tuning/runs":
            self._send_json(AUTOTUNE_RUN_STORE.list_runs())
            return
        if parsed.path == "/api/tuning/run":
            self._handle_tuning_load(parsed.query)
            return
        if parsed.path == "/api/self-test":
            self._handle_self_test(parsed.query)
            return
        if parsed.path.startswith("/results/"):
            self._serve_result_file(parsed.path)
            return
        if parsed.path.startswith("/api/"):
            self._send_json({"ok": False, "error": f"Unknown endpoint: {parsed.path}"}, status=404)
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
        if parsed.path == "/api/scope/acquisition":
            self._handle_scope_acquisition()
            return
        if parsed.path == "/api/scope/warmup":
            self._handle_scope_warmup()
            return
        if parsed.path == "/api/tuning/start":
            self._handle_tuning_start()
            return
        if parsed.path == "/api/tuning/pause":
            status = TUNING_SESSION.pause()
            if status.get("history") or status.get("state") == "running" or AUTOTUNE_RUN_STORE._current_run_id:
                status = AUTOTUNE_RUN_STORE.persist_status(status)
            self._send_json({"ok": True, **status})
            return
        if parsed.path == "/api/tuning/resume":
            try:
                payload = self._read_json_body()
            except Exception:
                payload = {}
            try:
                run_id = str(payload.get("run_id", "")).strip()
                kind = str(payload.get("kind", "recent")).strip() or "recent"
                if run_id:
                    status = AUTOTUNE_RUN_STORE.resume_run(run_id, kind)
                else:
                    status = TUNING_SESSION.resume()
                    if status.get("history") or status.get("state") == "running" or AUTOTUNE_RUN_STORE._current_run_id:
                        status = AUTOTUNE_RUN_STORE.persist_status(status)
                    status = {"ok": True, **status}
                self._send_json(status)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
        if parsed.path == "/api/tuning/stop":
            status = TUNING_SESSION.stop()
            if status.get("history") or AUTOTUNE_RUN_STORE._current_run_id:
                status = AUTOTUNE_RUN_STORE.persist_status(status)
            self._send_json({"ok": True, **status})
            return
        if parsed.path == "/api/tuning/reset":
            self._handle_tuning_reset()
            return
        if parsed.path == "/api/tuning/step":
            self._handle_tuning_step()
            return
        if parsed.path == "/api/tuning/archive":
            self._handle_tuning_archive()
            return
        if parsed.path == "/api/tuning/delete":
            self._handle_tuning_delete()
            return
        if parsed.path == "/api/tuning/load":
            self._handle_tuning_load_post()
            return
        if parsed.path == "/api/tuning/gif":
            self._handle_tuning_gif()
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
            experiment = _experiment_from_payload(payload.get("experiment", {}))
            status = TUNING_SESSION.start(config, experiment)
            status = AUTOTUNE_RUN_STORE.start_new(status)
            self._send_json({"ok": True, **status})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_tuning_step(self) -> None:
        try:
            payload = self._read_json_body()
            config_payload = payload.get("config")
            config = _config_from_payload(config_payload) if config_payload else None
            experiment = _experiment_from_payload(payload.get("experiment", {})) if payload.get("experiment") else None
            status_before = TUNING_SESSION.status()
            if not status_before.get("history") and AUTOTUNE_RUN_STORE._current_run_id is None:
                AUTOTUNE_RUN_STORE.start_new(status_before)
            status = TUNING_SESSION.step(config, experiment)
            status = AUTOTUNE_RUN_STORE.persist_status(status)
            self._send_json({"ok": True, **status})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_tuning_reset(self) -> None:
        try:
            payload = self._read_json_body()
        except Exception:
            payload = {}
        try:
            config = _config_from_payload(payload.get("config", {}))
            experiment = _experiment_from_payload(payload.get("experiment", {})) if payload.get("experiment") else None
            status = TUNING_SESSION.configure(config, experiment)
            AUTOTUNE_RUN_STORE.stop_current()
            self._send_json({"ok": True, **status})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_tuning_archive(self) -> None:
        try:
            payload = self._read_json_body()
        except Exception:
            payload = {}
        try:
            status = TUNING_SESSION.status()
            if status.get("history") or status.get("state") == "running" or AUTOTUNE_RUN_STORE._current_run_id:
                AUTOTUNE_RUN_STORE.persist_status(status)
            run_id = payload.get("run_id")
            kind = str(payload.get("kind", "recent"))
            if run_id not in (None, ""):
                self._send_json(AUTOTUNE_RUN_STORE.archive_run(str(run_id), kind, payload.get("name")))
            else:
                self._send_json(AUTOTUNE_RUN_STORE.archive_current(payload.get("name")))
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_tuning_delete(self) -> None:
        try:
            payload = self._read_json_body()
            run_id = str(payload.get("run_id", ""))
            kind = str(payload.get("kind", "recent"))
            if not run_id:
                raise RuntimeError("Select an auto-tune result to delete.")
            self._send_json(AUTOTUNE_RUN_STORE.delete_run(run_id, kind))
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_tuning_load(self, query: str) -> None:
        params = parse_qs(query)
        run_id = params.get("run_id", [""])[0]
        kind = params.get("kind", ["recent"])[0]
        try:
            self._send_json(AUTOTUNE_RUN_STORE.load_run(run_id, kind))
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=404)

    def _handle_tuning_load_post(self) -> None:
        try:
            payload = self._read_json_body()
            run_id = str(payload.get("run_id", ""))
            kind = str(payload.get("kind", "recent"))
            self._send_json(AUTOTUNE_RUN_STORE.load_run(run_id, kind))
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_tuning_gif(self) -> None:
        try:
            payload = self._read_json_body()
        except Exception:
            payload = {}
        try:
            status = TUNING_SESSION.status()
            if status.get("history") or status.get("state") == "running" or AUTOTUNE_RUN_STORE._current_run_id:
                AUTOTUNE_RUN_STORE.persist_status(status)
            run_id = payload.get("run_id")
            kind = str(payload.get("kind", "recent"))
            try:
                duration_ms = int(payload.get("duration_ms", 100))
            except (TypeError, ValueError):
                duration_ms = 100
            self._send_json(AUTOTUNE_RUN_STORE.save_animation_gif(None if run_id in (None, "") else str(run_id), kind, duration_ms))
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
            if device == "board_i2c":
                with DEVICE_LOCK:
                    self._send_json(run_single_instrument_self_test(device, config))
            elif device:
                self._send_json(run_single_instrument_self_test(device, config))
            else:
                with DEVICE_LOCK:
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

    def _handle_bode100_idn(self, query: str) -> None:
        params = parse_qs(query)
        host = params.get("host", [None])[0]
        port = _optional_int_param(params, "port")
        serial = params.get("serial", [None])[0]
        runner_path = params.get("scpi_runner_path", [None])[0]
        visa_resource = params.get("visa_resource", [None])[0]
        timeout = _optional_float_param(params, "timeout")
        self._send_json(_read_bode100_idn(host, port, serial, runner_path, visa_resource, timeout))

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
            output_enabled = payload.get("output_enabled")
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return
        self._send_json(_set_power_supply(resource, voltage, current, output_enabled))

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
        measurements = _list_param(params, "measurements", [])
        points = _optional_int_param(params, "points")
        fg_frequency_hz = _optional_float_param(params, "function_generator_frequency_hz")
        self._send_json(_capture_scope(resource, channels, measurements, points, fg_frequency_hz))

    def _handle_scope_capture_post(self) -> None:
        try:
            payload = self._read_json_body()
            resource = str(payload.get("resource", DEFAULT_SCOPE_RESOURCE))
            channels = [str(item).upper() for item in payload.get("channels", ["CH1"])]
            measurements = [str(item).upper() for item in payload.get("measurements", [])]
            points = None if payload.get("points") is None else int(payload.get("points"))
            fg_frequency_hz = None if payload.get("function_generator_frequency_hz") is None else float(payload.get("function_generator_frequency_hz"))
            scope_axis_settings = _normalize_scope_axis_settings(payload.get("scope_axis_settings"))
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return
        self._send_json(_capture_scope(resource, channels, measurements, points, fg_frequency_hz, scope_axis_settings))

    def _handle_scope_capture_full(self, path: str, query: str) -> None:
        capture_id = path.removeprefix("/api/scope/capture/").removesuffix("/full").strip("/")
        params = parse_qs(query)
        channel = params.get("channel", [None])[0]
        inline = params.get("inline", ["0"])[0].lower() in {"1", "true", "yes"}
        self._send_json(_get_full_scope_capture(capture_id, channel=channel, inline=inline))

    def _handle_scope_acquisition(self) -> None:
        try:
            payload = self._read_json_body()
            resource = str(payload.get("resource", DEFAULT_SCOPE_RESOURCE))
            running = bool(payload.get("running"))
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return
        self._send_json(_set_scope_acquisition(resource, running))

    def _handle_scope_warmup(self) -> None:
        try:
            payload = self._read_json_body()
            resource = str(payload.get("resource", DEFAULT_SCOPE_RESOURCE))
        except Exception as exc:
            self._send_json({"ok": False, "error": f"Invalid request: {exc}"}, status=400)
            return
        self._send_json(_warm_scope_connection(resource))

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

    def _serve_result_file(self, path: str) -> None:
        target = (RESULTS_DIR / path.removeprefix("/results/")).resolve()
        if RESULTS_DIR.resolve() not in target.parents:
            self.send_error(403)
            return
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


def _prepare_vout_for_autotune(
    *,
    address: str,
    page: int,
    voltage: float,
    adapter_kind: str,
    tolerance_v: float,
) -> dict:
    """Prepare VOUT for Auto-Tune without changing PMBus OPERATION.

    Manual Vout writes intentionally mirror XDP Designer and write OPERATION=0x80.
    During Auto-Tune that extra OPERATION write can disturb an already-running
    board, so this path only writes VOUT_COMMAND when the command is not already
    close to the target.
    """

    if page != 0:
        return {
            "ok": False,
            "error": "Refusing Auto-Tune Vout preparation on nonzero page.",
            "address": address,
            "page": page,
            "requested_v": voltage,
        }
    with DEVICE_LOCK:
        board = None
        try:
            board = _connect_board(address, adapter_kind)
            operation_before = _safe_read_operation(board, page)
            raw_mode, mode_name, exponent = board.read_vout_mode(page)
            if mode_name != "linear":
                return {
                    "ok": False,
                    "error": f"Vout safety check failed: invalid VOUT_MODE {mode_name}.",
                    "address": f"0x{board.device.address:02X}",
                    "page": page,
                    "requested_v": voltage,
                    "vout_mode_raw": f"0x{raw_mode:02X}",
                    "vout_mode": mode_name,
                    "operation_before": _format_optional_byte(operation_before),
                }

            command_before = board.read_vout_command(page)
            read_before = board.read_vout(page)
            command_step = abs(2.0 ** exponent)
            command_tolerance = max(command_step * 1.25, 0.0025)
            raw_written = None
            skipped = abs(command_before - voltage) <= command_tolerance
            if not skipped:
                raw_written = board.set_vout_command(voltage, page=page)
                _wait_for_vout(board, page=page, target_v=voltage)

            operation_after = _safe_read_operation(board, page)
            vout_command = board.read_vout_command(page)
            read_vout = board.read_vout(page)
            read_iout = board.read_iout(page)
            if abs(read_vout - voltage) > tolerance_v:
                return {
                    "ok": False,
                    "error": f"Vout safety check failed: read {read_vout:.4f} V, target {voltage:.4f} V.",
                    "address": f"0x{board.device.address:02X}",
                    "page": page,
                    "requested_v": voltage,
                    "operation_before": _format_optional_byte(operation_before),
                    "operation_after": _format_optional_byte(operation_after),
                    "operation_preserved": operation_before == operation_after,
                    "vout_mode_raw": f"0x{raw_mode:02X}",
                    "vout_mode": mode_name,
                    "exponent": exponent,
                    "vout_command_before_v": command_before,
                    "vout_command_v": vout_command,
                    "read_vout_v": read_vout,
                    "read_iout_a": read_iout,
                    "skipped_vout_write": skipped,
                }

            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "requested_v": voltage,
                "raw_written": None if raw_written is None else f"0x{raw_written:04X}",
                "operation_before": _format_optional_byte(operation_before),
                "operation_after": _format_optional_byte(operation_after),
                "operation_preserved": operation_before == operation_after,
                "operation_set": "preserved",
                "vout_mode_raw": f"0x{raw_mode:02X}",
                "vout_mode": mode_name,
                "exponent": exponent,
                "vout_command_before_v": command_before,
                "vout_command_v": vout_command,
                "read_vout_before_v": read_before,
                "read_vout_v": read_vout,
                "read_iout_a": read_iout,
                "skipped_vout_write": skipped,
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
    async_artifacts: bool = False,
) -> dict:
    started = time.perf_counter()
    if start_hz <= 0 or stop_hz <= start_hz:
        return {"ok": False, "error": "Bode sweep requires 0 < start_hz < stop_hz."}
    if not 2 <= points <= 2001:
        return {"ok": False, "error": "Bode sweep points must be between 2 and 2001."}

    bode_driver = Bode100Driver(host=host, port=port, startup_timeout_s=max(timeout_ms / 1000.0, 5.0))
    try:
        bode_driver.ensure_scpi_server()
    except Exception as exc:
        return {
            "ok": False,
            "error": (
                f"No Bode SCPI TCP listener is reachable at {host}:{port}: {exc}"
            ),
            "host": host,
            "port": port,
            "resource": bode_driver.resource_name,
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
        bode = BodeScpiClient(resource_name=bode_driver.resource_name, timeout_ms=timeout_ms)
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
        margins = data.stability_margins.as_dict()
        timestamp = time.time()
        sweep_id = uuid.uuid4().hex[:12]
        BODE_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
        data_file_path = BODE_SWEEP_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{sweep_id}.npz"
        data_file_pending = False
        if async_artifacts:
            data_file_pending = True
            _schedule_bode_data_artifact(
                frequency_hz=data.frequency_hz,
                magnitude_db=data.magnitude_db,
                phase_deg=data.phase_deg,
                path=data_file_path,
                metadata={
                    "start_hz": float(start_hz),
                    "stop_hz": float(stop_hz),
                    "points": int(points),
                    "bandwidth_hz": float(bandwidth_hz),
                    "source_dbm": np.nan if source_dbm is None else float(source_dbm),
                    "identity": str(identity),
                    "timestamp": float(timestamp),
                },
            )
        else:
            _write_bode_data_artifact(
                np.asarray(data.frequency_hz, dtype=np.float64),
                np.asarray(data.magnitude_db, dtype=np.float64),
                np.asarray(data.phase_deg, dtype=np.float64),
                data_file_path,
                {
                    "start_hz": float(start_hz),
                    "stop_hz": float(stop_hz),
                    "points": int(points),
                    "bandwidth_hz": float(bandwidth_hz),
                    "source_dbm": np.nan if source_dbm is None else float(source_dbm),
                    "identity": str(identity),
                    "timestamp": float(timestamp),
                },
            )
        bode_png = None
        bode_png_error = None
        bode_png_pending = False
        try:
            bode_png_path = RESULTS_DIR / "bode_sweeps" / f"{time.strftime('%Y%m%d_%H%M%S')}_{sweep_id}.png"
            if async_artifacts:
                bode_png = _scope_png_public_path(bode_png_path)
                bode_png_pending = True
                _schedule_bode_png_artifact(
                    frequency_hz=data.frequency_hz,
                    magnitude_db=data.magnitude_db,
                    phase_deg=data.phase_deg,
                    margins=margins,
                    path=bode_png_path,
                )
            else:
                bode_png = _plot_full_bode_sweep_png(
                    frequency_hz=data.frequency_hz,
                    magnitude_db=data.magnitude_db,
                    phase_deg=data.phase_deg,
                    margins=margins,
                    path=bode_png_path,
                )
                LATEST_BODE_PNG.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bode_png_path, LATEST_BODE_PNG)
        except Exception as exc:
            bode_png_error = str(exc)
            bode_png_pending = False
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
            "resource": bode_driver.resource_name,
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
            "margins": margins,
            "sweep_id": sweep_id,
            "data_file": str(data_file_path.relative_to(ROOT)),
            "data_file_pending": data_file_pending,
            "original_points": len(data.frequency_hz),
            "display_points": len(data.frequency_hz),
            "bode_png": bode_png,
            "bode_png_error": bode_png_error,
            "bode_png_pending": bode_png_pending,
            "system_error": system_error,
            "duration_s": round(time.perf_counter() - started, 3),
            "timestamp": timestamp,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "host": host,
            "port": port,
            "resource": bode_driver.resource_name,
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


def _read_bode100_idn(
    host: str | None,
    port: int | None,
    serial: str | None,
    runner_path: str | None,
    visa_resource: str | None,
    timeout_s: float | None,
) -> dict:
    driver = Bode100Driver(
        serial_number=serial,
        host=host or "127.0.0.1",
        port=port or 5025,
        scpi_runner_path=runner_path or DEFAULT_BODE100_SCPI_RUNNER_PATH,
        startup_timeout_s=timeout_s or 30.0,
        visa_resource=visa_resource,
    )
    try:
        driver.connect()
        identity = driver.identify()
        return {
            "ok": True,
            "status": "connected",
            "idn": identity,
            "host": driver.host,
            "port": driver.port,
            "resource": driver.resource_name,
            "timestamp": time.time(),
        }
    except (Bode100Error, VisaConnectionError) as exc:
        return {
            "ok": False,
            "status": "failed",
            "error": str(exc),
            "host": driver.host,
            "port": driver.port,
            "resource": driver.resource_name,
            "timestamp": time.time(),
        }
    finally:
        driver.close()


def _read_power_supply(resource: str) -> dict:
    with DEVICE_LOCK:
        supply = None
        try:
            supply = KeysightN5700PowerSupply(resource, timeout_ms=1500)
            supply.connect()
            return _power_supply_snapshot(supply, resource)
        except Exception as exc:
            return {"ok": False, "resource": resource, "error": str(exc), "timestamp": time.time()}
        finally:
            if supply is not None:
                supply.close()


def _set_power_supply(resource: str, voltage: object, current: object, output_enabled: object = None) -> dict:
    if voltage is None and current is None and output_enabled is None:
        return {"ok": False, "resource": resource, "error": "No voltage, current limit, or output state was provided."}
    with DEVICE_LOCK:
        supply = None
        try:
            supply = KeysightN5700PowerSupply(resource, timeout_ms=1500)
            supply.connect()
            if output_enabled is not None:
                if bool(output_enabled):
                    supply.output_on()
                else:
                    supply.output_off()
            if voltage is not None:
                supply.set_voltage(float(voltage))
            if current is not None:
                supply.set_current_limit(float(current))
            return _power_supply_snapshot(supply, resource, include_error=True)
        except Exception as exc:
            return {"ok": False, "resource": resource, "error": str(exc), "timestamp": time.time()}
        finally:
            if supply is not None:
                supply.close()


def _power_supply_snapshot(supply: KeysightN5700PowerSupply, resource: str, include_error: bool = False) -> dict:
    output_raw = _safe_query(supply, "OUTP?")
    output_enabled = None
    if output_raw is not None:
        normalized_output = output_raw.strip().upper()
        if normalized_output in {"1", "ON"}:
            output_enabled = True
        elif normalized_output in {"0", "OFF"}:
            output_enabled = False
    error = _visible_instrument_error(_safe_query(supply, "SYST:ERR?")) if include_error else None
    return {
        "ok": True,
        "resource": resource,
        "identity": None,
        "output_enabled": output_enabled,
        "voltage_setpoint_v": _safe_float_query(supply, "VOLT?"),
        "current_limit_a": _safe_float_query(supply, "CURR?"),
        "measured_voltage_v": _safe_float_query(supply, "MEAS:VOLT?"),
        "measured_current_a": _safe_float_query(supply, "MEAS:CURR?"),
        "error": error,
        "timestamp": time.time(),
    }


def _read_function_generator(resource: str, channel: int) -> dict:
    with DEVICE_LOCK:
        fg = None
        try:
            fg = FunctionGenerator(resource, timeout_ms=1500, output_channel=channel)
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
            fg = FunctionGenerator(resource, timeout_ms=1500, output_channel=channel)
            fg.connect()
            if "output_enabled" in payload:
                if bool(payload.get("output_enabled")):
                    fg.output_on(channel=channel)
                else:
                    fg.output_off(channel=channel)
                return _function_generator_snapshot(fg, resource, channel)
            voltage_unit = str(payload.get("voltage_unit", "VPP"))
            if voltage_unit:
                fg.set_voltage_unit(voltage_unit, channel=channel)
            if mode_name == "square":
                fg.configure_square_levels(
                    frequency_hz=float(payload.get("frequency_hz", 10000.0)),
                    low_v=float(payload.get("low_v", 0.365)),
                    high_v=float(payload.get("high_v", 2.0)),
                    channel=channel,
                )
            elif mode_name == "pulse":
                fg.configure_pulse_levels(
                    frequency_hz=float(payload.get("frequency_hz", 10000.0)),
                    low_v=float(payload.get("low_v", 0.365)),
                    high_v=float(payload.get("high_v", 2.0)),
                    width_s=_optional_float(payload.get("pulse_width_s")),
                    channel=channel,
                )
            elif mode_name == "dc":
                fg.configure_dc(float(payload.get("dc_level_v", payload.get("offset_v", 0.0))), channel=channel)
            elif mode_name == "sine":
                fg.configure_sine(
                    frequency_hz=float(payload.get("frequency_hz", 10000.0)),
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
    fg.clear_status()
    function = _safe_query(fg, f"SOUR{channel}:FUNC?")
    frequency_hz = _safe_float_query(fg, f"SOUR{channel}:FREQ?")
    voltage_unit = _safe_query(fg, f"SOUR{channel}:VOLT:UNIT?")
    high_v = _safe_float_query(fg, f"SOUR{channel}:VOLT:HIGH?")
    low_v = _safe_float_query(fg, f"SOUR{channel}:VOLT:LOW?")
    output = _safe_query(fg, f"OUTP{channel}?")
    system_error = _visible_instrument_error(_safe_query(fg, "SYST:ERR?"))
    return {
        "ok": True,
        "resource": resource,
        "channel": channel,
        "identity": None,
        "function": function,
        "frequency_hz": frequency_hz,
        "voltage_unit": voltage_unit,
        "amplitude_vpp": None,
        "offset_v": None,
        "high_v": high_v,
        "low_v": low_v,
        "phase_deg": None,
        "duty_percent": None,
        "pulse_width_s": None,
        "output": output,
        "system_error": system_error,
        "timestamp": time.time(),
    }


def _visible_instrument_error(message: str | None) -> str | None:
    if not message:
        return None
    normalized = message.strip().lower()
    if "no error" in normalized or normalized in {"0", "+0"}:
        return None
    if "query unterminated" in normalized:
        return None
    return message


def _get_scope_connection(resource: str, timeout_ms: int) -> tuple[TektronixOscilloscope, bool]:
    scope = SCOPE_CONNECTIONS.get(resource)
    if scope is not None and scope.is_connected:
        if getattr(scope, "_inst", None) is not None:
            scope._inst.timeout = timeout_ms
        return scope, False
    if scope is not None:
        _drop_scope_connection(resource)
    scope = TektronixOscilloscope(resource, timeout_ms=timeout_ms)
    scope.connect()
    SCOPE_CONNECTIONS[resource] = scope
    return scope, True


def _drop_scope_connection(resource: str) -> None:
    scope = SCOPE_CONNECTIONS.pop(resource, None)
    if scope is not None:
        scope.close()


def _is_stale_scope_session_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "invalid session" in message
        or "session handle" in message
        or "resource might be closed" in message
    )


def _edge_focused_scope_display(
    x_values: list[float],
    y_values: list[float],
    max_points: int = SCOPE_DISPLAY_MAX_POINTS,
) -> tuple[list[float], list[float], str]:
    total = min(len(x_values), len(y_values))
    if total == 0:
        return [], [], "edge-focused"
    if total <= max_points or max_points < 2:
        return x_values[:total], y_values[:total], "edge-focused"

    x_array = np.asarray(x_values[:total], dtype=np.float64)
    y_array = np.asarray(y_values[:total], dtype=np.float64)
    diff = np.abs(np.diff(y_array))
    finite_diff = diff[np.isfinite(diff)]
    if finite_diff.size == 0 or float(np.max(finite_diff)) <= 0.0:
        indices = np.rint(np.linspace(0, total - 1, max_points)).astype(np.int64)
        return x_array[indices].tolist(), y_array[indices].tolist(), "edge-focused"

    median = float(np.median(finite_diff))
    mad = float(np.median(np.abs(finite_diff - median)))
    percentile = float(np.percentile(finite_diff, 99.7))
    threshold = max(percentile, median + 10.0 * mad, float(np.max(finite_diff)) * 0.05)
    edge_locations = np.flatnonzero(diff >= threshold) + 1
    if edge_locations.size == 0:
        indices = np.rint(np.linspace(0, total - 1, max_points)).astype(np.int64)
        return x_array[indices].tolist(), y_array[indices].tolist(), "edge-focused"

    window = max(20, min(2_000, total // 200))
    edge_mask = np.zeros(total, dtype=bool)
    for location in edge_locations:
        start = max(0, int(location) - window)
        stop = min(total, int(location) + window + 1)
        edge_mask[start:stop] = True

    edge_indices = np.flatnonzero(edge_mask)
    max_edge_points = max(2, int(max_points * 0.7))
    if edge_indices.size > max_edge_points:
        sample = np.rint(np.linspace(0, edge_indices.size - 1, max_edge_points)).astype(np.int64)
        edge_indices = edge_indices[sample]

    remaining = max(0, max_points - int(edge_indices.size) - 2)
    steady_indices = np.flatnonzero(~edge_mask)
    if remaining > 0 and steady_indices.size > 0:
        steady_count = min(remaining, int(steady_indices.size))
        sample = np.rint(np.linspace(0, steady_indices.size - 1, steady_count)).astype(np.int64)
        steady_indices = steady_indices[sample]
    else:
        steady_indices = np.array([], dtype=np.int64)

    indices = np.unique(np.concatenate((np.array([0, total - 1], dtype=np.int64), edge_indices, steady_indices)))
    if indices.size > max_points:
        sample = np.rint(np.linspace(0, indices.size - 1, max_points)).astype(np.int64)
        indices = indices[sample]
    indices.sort()
    return x_array[indices].tolist(), y_array[indices].tolist(), "edge-focused"


def _scope_capture_file_path(capture_id: str, source: str, timestamp: float) -> Path:
    safe_source = "".join(char for char in source.upper() if char.isalnum() or char in {"_", "-"})
    time_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
    return SCOPE_CAPTURE_DIR / f"{time_tag}_{capture_id}_{safe_source}.npz"


def _schedule_scope_data_artifact(x: np.ndarray, y: np.ndarray, path: Path, metadata: dict) -> None:
    ARTIFACT_EXECUTOR.submit(
        _write_scope_data_artifact,
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        path,
        copy.deepcopy(metadata),
    )


def _write_scope_data_artifact(x: np.ndarray, y: np.ndarray, path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, y=y, **metadata)


def _store_full_scope_waveform(capture_id: str, capture, timestamp: float, async_save: bool = False) -> tuple[str, dict]:
    x_array = np.asarray(capture.x, dtype=np.float64)
    y_array = np.asarray(capture.y, dtype=np.float64)
    file_path = _scope_capture_file_path(capture_id, capture.source, timestamp)
    metadata = {
        "source": capture.source,
        "x_unit": capture.x_unit,
        "y_unit": capture.y_unit,
        "original_points": int(capture.original_points or len(capture.y)),
        "transfer_encoding": capture.transfer_encoding or "",
        "capture_id": capture_id,
        "timestamp": timestamp,
    }
    if async_save:
        _schedule_scope_data_artifact(x_array, y_array, file_path, metadata)
    else:
        _write_scope_data_artifact(x_array, y_array, file_path, metadata)
    try:
        data_file = str(file_path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        data_file = str(file_path)
    channel_record = {
        "source": capture.source,
        "x": x_array,
        "y": y_array,
        "x_unit": capture.x_unit,
        "y_unit": capture.y_unit,
        "original_points": int(capture.original_points or len(capture.y)),
        "transfer_encoding": capture.transfer_encoding,
        "data_file": data_file,
        "data_file_pending": bool(async_save),
    }
    return data_file, channel_record


def _remember_scope_capture(capture_id: str, entry: dict) -> None:
    SCOPE_CAPTURE_CACHE[capture_id] = entry
    while len(SCOPE_CAPTURE_CACHE) > SCOPE_CAPTURE_CACHE_LIMIT:
        oldest_key = next(iter(SCOPE_CAPTURE_CACHE))
        SCOPE_CAPTURE_CACHE.pop(oldest_key, None)


def _scope_png_public_path(path: Path) -> str:
    try:
        return "/" + str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _safe_file_stem(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value).strip())
    return safe.strip("_") or uuid.uuid4().hex[:8]


def _path_label(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _path_from_result_reference(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    text = value.strip().replace("\\", "/")
    if text.startswith("/"):
        text = text[1:]
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    try:
        path.resolve().relative_to(ROOT.resolve())
    except Exception:
        return None
    return path


def _normalize_scope_axis_settings(raw: object) -> dict:
    if not isinstance(raw, dict):
        return DEFAULT_SCOPE_AXIS_SETTINGS
    channel_axes = raw.get("channelAxes")
    if not isinstance(channel_axes, dict):
        channel_axes = {}
    normalized_axes = dict(DEFAULT_SCOPE_AXIS_SETTINGS["channelAxes"])
    for key, value in channel_axes.items():
        channel = str(key).upper()
        if channel in normalized_axes:
            normalized_axes[channel] = "right" if str(value).lower() == "right" else "left"
    try:
        left_min = float(raw.get("leftMin", DEFAULT_SCOPE_AXIS_SETTINGS["leftMin"]))
        left_max = float(raw.get("leftMax", DEFAULT_SCOPE_AXIS_SETTINGS["leftMax"]))
        right_min = float(raw.get("rightMin", DEFAULT_SCOPE_AXIS_SETTINGS["rightMin"]))
        right_max = float(raw.get("rightMax", DEFAULT_SCOPE_AXIS_SETTINGS["rightMax"]))
    except Exception:
        left_min = float(DEFAULT_SCOPE_AXIS_SETTINGS["leftMin"])
        left_max = float(DEFAULT_SCOPE_AXIS_SETTINGS["leftMax"])
        right_min = float(DEFAULT_SCOPE_AXIS_SETTINGS["rightMin"])
        right_max = float(DEFAULT_SCOPE_AXIS_SETTINGS["rightMax"])
    if left_min >= left_max:
        left_min, left_max = float(DEFAULT_SCOPE_AXIS_SETTINGS["leftMin"]), float(DEFAULT_SCOPE_AXIS_SETTINGS["leftMax"])
    if right_min >= right_max:
        right_min, right_max = float(DEFAULT_SCOPE_AXIS_SETTINGS["rightMin"]), float(DEFAULT_SCOPE_AXIS_SETTINGS["rightMax"])
    return {
        "leftMin": left_min,
        "leftMax": left_max,
        "rightMin": right_min,
        "rightMax": right_max,
        "channelAxes": normalized_axes,
    }


def _scope_axis_settings_from_status(status: dict) -> dict:
    experiment = status.get("experiment") if isinstance(status, dict) else None
    if not isinstance(experiment, dict):
        return DEFAULT_SCOPE_AXIS_SETTINGS
    scope_config = experiment.get("scope_config")
    if not isinstance(scope_config, dict):
        return DEFAULT_SCOPE_AXIS_SETTINGS
    return _normalize_scope_axis_settings(scope_config.get("scope_axis_settings"))


def _scope_capture_entry_from_result(scope_result: dict) -> dict | None:
    waveforms = scope_result.get("waveforms") if isinstance(scope_result.get("waveforms"), list) else []
    channels: dict[str, dict] = {}
    capture_id = scope_result.get("capture_id")
    created_at = time.time()
    for waveform in waveforms:
        if not isinstance(waveform, dict):
            continue
        data_file = _path_from_result_reference(waveform.get("data_file"))
        if not data_file or not data_file.exists():
            continue
        try:
            with np.load(data_file, allow_pickle=True) as payload:
                source = str(payload["source"].item() if getattr(payload["source"], "shape", None) == () else waveform.get("source"))
                x_values = np.asarray(payload["x"], dtype=np.float64)
                y_values = np.asarray(payload["y"], dtype=np.float64)
                x_unit = str(payload["x_unit"].item()) if "x_unit" in payload.files else str(waveform.get("x_unit") or "s")
                y_unit = str(payload["y_unit"].item()) if "y_unit" in payload.files else str(waveform.get("y_unit") or "V")
                original_points = int(payload["original_points"].item()) if "original_points" in payload.files else int(len(y_values))
                transfer_encoding = (
                    str(payload["transfer_encoding"].item()) if "transfer_encoding" in payload.files else str(waveform.get("transfer_encoding") or "")
                )
                if "capture_id" in payload.files:
                    capture_id = str(payload["capture_id"].item())
                if "timestamp" in payload.files:
                    created_at = float(payload["timestamp"].item())
        except Exception:
            continue
        channels[source.upper()] = {
            "source": source.upper(),
            "x": x_values,
            "y": y_values,
            "x_unit": x_unit,
            "y_unit": y_unit,
            "original_points": original_points,
            "transfer_encoding": transfer_encoding,
            "data_file": waveform.get("data_file"),
            "data_file_pending": False,
        }
    if not channels:
        return None
    return {
        "created_at": created_at,
        "capture_id": capture_id,
        "resource": scope_result.get("resource"),
        "channels": channels,
    }


def _plot_full_bode_sweep_png(
    *,
    frequency_hz: list[float],
    magnitude_db: list[float],
    phase_deg: list[float],
    margins: dict | None = None,
    path: Path = LATEST_BODE_PNG,
    title: str = "Latest Bode 100 Sweep - Full Data",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    magnitude = np.asarray(magnitude_db, dtype=np.float64)
    phase = np.asarray(phase_deg, dtype=np.float64)
    total = min(frequency.size, magnitude.size, phase.size)
    if total == 0:
        raise ValueError("No Bode sweep points are available to plot.")
    frequency = frequency[:total]
    magnitude = magnitude[:total]
    phase = phase[:total]
    mask = np.isfinite(frequency) & np.isfinite(magnitude) & np.isfinite(phase) & (frequency > 0)
    if not np.any(mask):
        raise ValueError("No finite Bode sweep points are available to plot.")
    frequency = frequency[mask]
    magnitude = magnitude[mask]
    phase = phase[mask]

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    fig, gain_ax = plt.subplots(figsize=(16, 6), dpi=150)
    phase_ax = gain_ax.twinx()
    gain_line, = gain_ax.semilogx(frequency, magnitude, color="#ea4335", linewidth=1.2, label="Gain")
    phase_line, = phase_ax.semilogx(frequency, phase, color="#1a73e8", linewidth=1.2, label="Phase")
    gain_ax.axhline(0, color="#9aa0a6", linewidth=0.8)
    phase_ax.axhline(0, color="#9aa0a6", linewidth=0.8, alpha=0.6)

    margins = margins or {}
    phase_crossover = margins.get("phase_crossover_hz")
    gain_crossover = margins.get("gain_crossover_hz")
    if phase_crossover:
        gain_ax.axvline(float(phase_crossover), color="#1a73e8", linestyle="--", linewidth=0.9, alpha=0.7)
    if gain_crossover:
        gain_ax.axvline(float(gain_crossover), color="#ea4335", linestyle="--", linewidth=0.9, alpha=0.7)

    gain_ax.set_xlim(float(np.min(frequency)), float(np.max(frequency)))
    gain_ax.set_ylim(-100, 100)
    phase_ax.set_ylim(-200, 200)
    gain_ax.set_title(title)
    gain_ax.set_xlabel("Frequency (Hz)")
    gain_ax.set_ylabel("Gain (dB)", color="#ea4335")
    phase_ax.set_ylabel("Phase (deg)", color="#1a73e8")
    gain_ax.tick_params(axis="y", colors="#ea4335")
    phase_ax.tick_params(axis="y", colors="#1a73e8")
    gain_ax.spines["left"].set_color("#ea4335")
    phase_ax.spines["right"].set_color("#1a73e8")
    gain_ax.grid(True, which="both", color="#d9dee7", linewidth=0.7, alpha=0.8)
    gain_ax.legend([gain_line, phase_line], ["Gain", "Phase"], loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return _scope_png_public_path(path)


def _schedule_bode_data_artifact(
    *,
    frequency_hz: list[float],
    magnitude_db: list[float],
    phase_deg: list[float],
    path: Path,
    metadata: dict,
) -> None:
    ARTIFACT_EXECUTOR.submit(
        _write_bode_data_artifact,
        np.asarray(frequency_hz, dtype=np.float64).copy(),
        np.asarray(magnitude_db, dtype=np.float64).copy(),
        np.asarray(phase_deg, dtype=np.float64).copy(),
        path,
        copy.deepcopy(metadata),
    )


def _write_bode_data_artifact(
    frequency_hz: np.ndarray,
    magnitude_db: np.ndarray,
    phase_deg: np.ndarray,
    path: Path,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frequency_hz=frequency_hz,
        magnitude_db=magnitude_db,
        phase_deg=phase_deg,
        **metadata,
    )


def _plot_full_scope_capture_png(
    entry: dict,
    axis_settings: dict | None = None,
    path: Path = LATEST_SCOPE_PNG,
    title: str = "Latest Scope Capture - Full Data",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = entry.get("channels", {})
    if not channels:
        raise ValueError("No scope channels are available to plot.")
    axis_settings = _normalize_scope_axis_settings(axis_settings)

    x_values_all = []
    for record in channels.values():
        x_array = np.asarray(record.get("x", []), dtype=np.float64)
        if x_array.size:
            x_values_all.append(x_array[np.isfinite(x_array)])
    if not x_values_all:
        raise ValueError("No finite scope time values are available to plot.")

    finite_x = np.concatenate([values for values in x_values_all if values.size])
    x0 = float(np.min(finite_x))
    x1 = float(np.max(finite_x))
    span_s = max(0.0, x1 - x0)
    x_unit = "s"
    x_scale = 1.0
    if span_s < 1e-3:
        x_unit = "us"
        x_scale = 1e6
    elif span_s < 1.0:
        x_unit = "ms"
        x_scale = 1e3

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")
    fig, left_ax = plt.subplots(figsize=(16, 6), dpi=150)
    right_ax = left_ax.twinx()
    colors = ["#1a73e8", "#ea4335", "#34a853", "#fbbc04", "#8ab4f8", "#f28b82", "#81c995", "#fde293"]
    lines = []
    labels = []
    for index, (source, record) in enumerate(channels.items()):
        x_array = np.asarray(record.get("x", []), dtype=np.float64)
        y_array = np.asarray(record.get("y", []), dtype=np.float64)
        total = min(x_array.size, y_array.size)
        if total == 0:
            continue
        axis_side = axis_settings["channelAxes"].get(source.upper(), "left")
        axis = right_ax if axis_side == "right" else left_ax
        line, = axis.plot(
            (x_array[:total] - x0) * x_scale,
            y_array[:total],
            label=source,
            linewidth=0.6,
            color=colors[index % len(colors)],
            rasterized=True,
        )
        lines.append(line)
        labels.append(source)

    left_color = "#1a73e8"
    right_color = "#ea4335"
    left_ax.set_xlim(0, (x1 - x0) * x_scale)
    left_ax.set_ylim(axis_settings["leftMin"], axis_settings["leftMax"])
    right_ax.set_ylim(axis_settings["rightMin"], axis_settings["rightMax"])
    left_ax.set_title(title)
    left_ax.set_xlabel(f"Time ({x_unit})")
    left_ax.set_ylabel("Voltage (V)", color=left_color)
    right_ax.set_ylabel("Voltage (V)", color=right_color)
    left_ax.tick_params(axis="y", colors=left_color)
    right_ax.tick_params(axis="y", colors=right_color)
    left_ax.spines["left"].set_color(left_color)
    right_ax.spines["right"].set_color(right_color)
    left_ax.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.8)
    if lines:
        left_ax.legend(lines, labels, loc="upper center", ncol=min(4, max(1, len(labels))))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return _scope_png_public_path(path)


def _schedule_bode_png_artifact(
    *,
    frequency_hz: list[float],
    magnitude_db: list[float],
    phase_deg: list[float],
    margins: dict | None,
    path: Path,
) -> None:
    frequency_copy = np.asarray(frequency_hz, dtype=np.float64).copy()
    magnitude_copy = np.asarray(magnitude_db, dtype=np.float64).copy()
    phase_copy = np.asarray(phase_deg, dtype=np.float64).copy()
    margins_copy = copy.deepcopy(margins or {})
    ARTIFACT_EXECUTOR.submit(
        _write_bode_png_artifact,
        frequency_copy,
        magnitude_copy,
        phase_copy,
        margins_copy,
        path,
    )


def _write_bode_png_artifact(
    frequency_hz: np.ndarray,
    magnitude_db: np.ndarray,
    phase_deg: np.ndarray,
    margins: dict,
    path: Path,
) -> None:
    _plot_full_bode_sweep_png(
        frequency_hz=frequency_hz.tolist(),
        magnitude_db=magnitude_db.tolist(),
        phase_deg=phase_deg.tolist(),
        margins=margins,
        path=path,
    )
    LATEST_BODE_PNG.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, LATEST_BODE_PNG)


def _schedule_scope_png_artifact(
    *,
    capture_entry: dict,
    scope_axis_settings: dict | None,
    path: Path,
) -> None:
    capture_copy = _copy_scope_capture_for_plot(capture_entry)
    axis_copy = copy.deepcopy(scope_axis_settings)
    ARTIFACT_EXECUTOR.submit(_write_scope_png_artifact, capture_copy, axis_copy, path)


def _write_scope_png_artifact(capture_entry: dict, scope_axis_settings: dict | None, path: Path) -> None:
    _plot_full_scope_capture_png(capture_entry, scope_axis_settings, path=path)
    LATEST_SCOPE_PNG.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, LATEST_SCOPE_PNG)


def _copy_scope_capture_for_plot(capture_entry: dict) -> dict:
    """Shallow-copy scope metadata without duplicating full waveform arrays."""

    channels = capture_entry.get("channels", {})
    copied_channels = {}
    if isinstance(channels, dict):
        for key, record in channels.items():
            if not isinstance(record, dict):
                continue
            copied_channels[key] = {
                "source": record.get("source"),
                "x": record.get("x"),
                "y": record.get("y"),
                "x_unit": record.get("x_unit"),
                "y_unit": record.get("y_unit"),
                "original_points": record.get("original_points"),
                "transfer_encoding": record.get("transfer_encoding"),
                "data_file": record.get("data_file"),
                "data_file_pending": record.get("data_file_pending"),
            }
    return {
        "created_at": capture_entry.get("created_at"),
        "resource": capture_entry.get("resource"),
        "channels": copied_channels,
    }


def _get_full_scope_capture(capture_id: str, channel: str | None = None, inline: bool = False) -> dict:
    entry = SCOPE_CAPTURE_CACHE.get(capture_id)
    if entry is None:
        return {"ok": False, "error": f"Scope capture '{capture_id}' is not in the in-memory cache."}
    channels = entry.get("channels", {})
    if channel:
        channel_key = channel.upper()
        record = channels.get(channel_key)
        if record is None:
            return {"ok": False, "error": f"Channel '{channel_key}' is not available in capture '{capture_id}'."}
        payload = {
            "ok": True,
            "capture_id": capture_id,
            "channel": channel_key,
            "source": record.get("source"),
            "x_unit": record.get("x_unit"),
            "y_unit": record.get("y_unit"),
            "original_points": record.get("original_points"),
            "transfer_encoding": record.get("transfer_encoding"),
            "data_file": record.get("data_file"),
        }
        if inline:
            x_values = record.get("x", [])
            y_values = record.get("y", [])
            payload["x"] = x_values.tolist() if hasattr(x_values, "tolist") else x_values
            payload["y"] = y_values.tolist() if hasattr(y_values, "tolist") else y_values
        return payload
    return {
        "ok": True,
        "capture_id": capture_id,
        "created_at": entry.get("created_at"),
        "channels": {
            key: {
                "source": record.get("source"),
                "x_unit": record.get("x_unit"),
                "y_unit": record.get("y_unit"),
                "original_points": record.get("original_points"),
                "transfer_encoding": record.get("transfer_encoding"),
                "data_file": record.get("data_file"),
            }
            for key, record in channels.items()
        },
    }


def _capture_scope(
    resource: str,
    channels: list[str],
    measurements: list[str],
    points: int | None,
    function_generator_frequency_hz: float | None = None,
    scope_axis_settings: dict | None = None,
    async_artifacts: bool = False,
) -> dict:
    started = time.perf_counter()
    timestamp = time.time()
    capture_id = uuid.uuid4().hex[:12]
    safe_channels = [
        ch
        for ch in (item.strip().upper() for item in channels)
        if ch in {f"CH{idx}" for idx in range(1, 9)}
    ][:8] or ["CH1"]
    safe_measurements = [item.strip().upper() for item in measurements if item.strip()][:8]
    stop = None if points is None else max(10, min(1_000_000, int(points)))
    scope_window_s = None
    scope_actual_window_s = None
    scope_scale_s_per_div = None
    scope_trigger_position_percent = None
    if function_generator_frequency_hz is not None and function_generator_frequency_hz > 0:
        function_generator_period_s = 1.0 / float(function_generator_frequency_hz)
        scope_window_s = max(
            1e-9,
            min(10.0, SCOPE_TRIGGER_OFFSET_FROM_LEFT_S + function_generator_period_s + 3e-6),
        )
    with DEVICE_LOCK:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                scope, opened = _get_scope_connection(resource, timeout_ms=6000)
                waveforms = []
                measurement_rows = []
                capture_cache_entry = {
                    "created_at": timestamp,
                    "resource": resource,
                    "channels": {},
                }
                selected_channels = set(safe_channels)
                for index in range(1, 9):
                    channel = f"CH{index}"
                    scope.set_channel_display(channel, channel in selected_channels)
                scope.set_edge_trigger("CH1", "RISE")
                if scope_window_s is not None:
                    scope_scale_s_per_div = scope.set_horizontal_window(scope_window_s)
                    scope_actual_window_s = 10.0 * scope_scale_s_per_div
                    scope_trigger_position_percent = scope.set_trigger_position_from_left(
                        SCOPE_TRIGGER_OFFSET_FROM_LEFT_S,
                        scope_actual_window_s,
                    )
                force_after_s = 0.5 if scope_window_s is None else max(0.25, min(2.0, scope_window_s * 1.5))
                scope.single_acquisition(timeout_s=8.0, force_after_s=force_after_s)
                for channel in safe_channels:
                    capture = scope.capture_waveform(channel, start=1, stop=stop, max_plot_points=None)
                    data_file, full_record = _store_full_scope_waveform(
                        capture_id,
                        capture,
                        timestamp,
                        async_save=async_artifacts,
                    )
                    capture_cache_entry["channels"][channel] = full_record
                    display_x, display_y, display_strategy = _edge_focused_scope_display(
                        capture.x,
                        capture.y,
                        max_points=SCOPE_DISPLAY_MAX_POINTS,
                    )
                    time_span_s = max(capture.x) - min(capture.x) if capture.x else None
                    waveforms.append(
                        {
                            "source": capture.source,
                            "x": display_x,
                            "y": display_y,
                            "x_unit": capture.x_unit,
                            "y_unit": capture.y_unit,
                            "time_span_s": time_span_s,
                            "original_points": capture.original_points,
                            "plotted_points": len(display_y),
                            "display_points": len(display_y),
                            "display_strategy": display_strategy,
                            "capture_id": capture_id,
                            "data_file": data_file,
                            "data_file_pending": bool(async_artifacts),
                            "transfer_encoding": capture.transfer_encoding,
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
                _remember_scope_capture(capture_id, capture_cache_entry)
                scope_png = None
                scope_png_error = None
                scope_png_pending = False
                try:
                    scope_png_path = RESULTS_DIR / "scope_captures" / f"{time.strftime('%Y%m%d_%H%M%S')}_{capture_id}.png"
                    if async_artifacts:
                        scope_png = _scope_png_public_path(scope_png_path)
                        scope_png_pending = True
                        _schedule_scope_png_artifact(
                            capture_entry=capture_cache_entry,
                            scope_axis_settings=scope_axis_settings,
                            path=scope_png_path,
                        )
                    else:
                        scope_png = _plot_full_scope_capture_png(capture_cache_entry, scope_axis_settings, path=scope_png_path)
                        LATEST_SCOPE_PNG.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(scope_png_path, LATEST_SCOPE_PNG)
                except Exception as exc:
                    scope_png_error = str(exc)
                    scope_png_pending = False
                return {
                    "ok": True,
                    "resource": resource,
                    "capture_id": capture_id,
                    "scope_png": scope_png,
                    "scope_png_error": scope_png_error,
                    "scope_png_pending": scope_png_pending,
                    "identity": None,
                    "channels": safe_channels,
                    "measurements": safe_measurements,
                    "waveforms": waveforms,
                    "measurement_values": measurement_rows,
                    "acquisition_mode": "single",
                    "function_generator_frequency_hz": function_generator_frequency_hz,
                    "scope_window_s": scope_window_s,
                    "scope_actual_window_s": scope_actual_window_s,
                    "scope_scale_s_per_div": scope_scale_s_per_div,
                    "scope_trigger_source": "CH1",
                    "scope_trigger_slope": "RISE",
                    "scope_trigger_offset_from_left_s": SCOPE_TRIGGER_OFFSET_FROM_LEFT_S if scope_window_s is not None else None,
                    "scope_trigger_position_percent": scope_trigger_position_percent,
                    "acquisition_started": True,
                    "acquisition_stopped": True,
                    "session_reused": not opened,
                    "session_retry": attempt,
                    "duration_s": round(time.perf_counter() - started, 3),
                    "timestamp": time.time(),
                }
            except Exception as exc:
                last_error = exc
                _drop_scope_connection(resource)
                if attempt == 0 and _is_stale_scope_session_error(exc):
                    continue
                break
        return {
            "ok": False,
            "resource": resource,
            "error": str(last_error) if last_error is not None else "Scope capture failed.",
            "duration_s": round(time.perf_counter() - started, 3),
            "timestamp": time.time(),
        }


def _set_scope_acquisition(resource: str, running: bool) -> dict:
    started = time.perf_counter()
    with DEVICE_LOCK:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                scope, opened = _get_scope_connection(resource, timeout_ms=3000)
                if running:
                    scope.start_acquisition()
                else:
                    scope.stop_acquisition()
                return {
                    "ok": True,
                    "resource": resource,
                    "running": running,
                    "session_reused": not opened,
                    "session_retry": attempt,
                    "duration_s": round(time.perf_counter() - started, 3),
                    "timestamp": time.time(),
                }
            except Exception as exc:
                last_error = exc
                _drop_scope_connection(resource)
                if attempt == 0 and _is_stale_scope_session_error(exc):
                    continue
                break
        return {
            "ok": False,
            "resource": resource,
            "running": None,
            "error": str(last_error) if last_error is not None else "Scope acquisition command failed.",
            "duration_s": round(time.perf_counter() - started, 3),
            "timestamp": time.time(),
        }


def _warm_scope_connection(resource: str) -> dict:
    started = time.perf_counter()
    with DEVICE_LOCK:
        try:
            _, opened = _get_scope_connection(resource, timeout_ms=3000)
            return {
                "ok": True,
                "resource": resource,
                "session_reused": not opened,
                "duration_s": round(time.perf_counter() - started, 3),
                "timestamp": time.time(),
            }
        except Exception as exc:
            _drop_scope_connection(resource)
            return {
                "ok": False,
                "resource": resource,
                "error": str(exc),
                "duration_s": round(time.perf_counter() - started, 3),
                "timestamp": time.time(),
            }


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
    adapter_name = adapter_kind.strip().lower()
    can_reset_xdp = adapter_name in {"xdp", "xdp_usb"}
    last_error: Exception | None = None
    for attempt in range(2 if can_reset_xdp else 1):
        board = None
        try:
            adapter = create_i2c_adapter(adapter_kind, timeout_ms=3000)
            board = create_board_controller(
                "infineon_xdp",
                adapter,
                BoardControllerConfig(address=address, name="XDPE1A2G5C"),
            )
            board.connect()
            return board
        except Exception as exc:
            last_error = exc
            if board is not None:
                try:
                    board.close()
                except Exception:
                    pass
            if not can_reset_xdp or "LIBUSB_ERROR_ACCESS" not in str(exc) or attempt == 1:
                raise
            reset_xdp_usb_bridges(include_external_processes=True)
            time.sleep(0.75)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not connect to board.")


def _int_param(params: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int(params.get(name, [str(default)])[0])
    except Exception:
        return default


def _optional_int_param(params: dict[str, list[str]], name: str) -> int | None:
    try:
        raw = params.get(name, [None])[0]
        return None if raw in {None, ""} else int(raw)
    except Exception:
        return None


def _optional_float_param(params: dict[str, list[str]], name: str) -> float | None:
    try:
        raw = params.get(name, [None])[0]
        return None if raw in {None, ""} else float(raw)
    except Exception:
        return None


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
    default_search = SearchSpace()
    return TuningConfig(
        plant=PlantParams(
            vdc=_float_field(plant_payload, "vdc", 12.0),
            inductance_h=_float_field(plant_payload, "inductance_h", 30e-6),
            capacitance_f=_float_field(plant_payload, "capacitance_f", 15e-6),
            capacitor_esr_ohm=_float_field(plant_payload, "capacitor_esr_ohm", 7.5e-3),
            inductor_dcr_ohm=_float_field(plant_payload, "inductor_dcr_ohm", 50e-3),
        ),
        targets=TuningTargets(
            vout_target_v=_float_field(targets_payload, "vout_target_v", 0.9296875),
            overshoot_pct=_float_field(targets_payload, "overshoot_pct", 3.0),
            undershoot_pct=_float_field(targets_payload, "undershoot_pct", 3.0),
            settling_time_s=_float_field(targets_payload, "settling_time_s", 4e-6),
            oscillations=_int_field(targets_payload, "oscillations", 0),
            phase_margin_deg=_float_field(targets_payload, "phase_margin_deg", 45.0),
            crossover_frequency_hz=_float_field(targets_payload, "crossover_frequency_hz", 200_000.0),
            gain_margin_db=_float_field(targets_payload, "gain_margin_db", 6.0),
            phase_margin_tolerance_deg=_float_field(targets_payload, "phase_margin_tolerance_deg", 5.0),
            crossover_tolerance_pct=_float_field(targets_payload, "crossover_tolerance_pct", 20.0),
        ),
        search=SearchSpace(
            wc_min_rad_s=_float_field(search_payload, "wc_min_rad_s", 94_248.0),
            wc_max_rad_s=_float_field(search_payload, "wc_max_rad_s", 314_159.0),
            phi_min_deg=_float_field(search_payload, "phi_min_deg", 30.0),
            phi_max_deg=_float_field(search_payload, "phi_max_deg", 80.0),
            initial_wc_rad_s=_float_field(search_payload, "initial_wc_rad_s", 157_080.0),
            initial_phi_deg=_float_field(search_payload, "initial_phi_deg", 60.0),
            max_iterations=_int_field(search_payload, "max_iterations", 40),
            mod0_kp=_search_parameter_from_payload(search_payload, "mod0_kp", default_search.mod0_kp, integer=True),
            mod0_ki=_search_parameter_from_payload(search_payload, "mod0_ki", default_search.mod0_ki, integer=True),
            mod0_kd=_search_parameter_from_payload(search_payload, "mod0_kd", default_search.mod0_kd, integer=True),
            mod0_kpole1=_search_parameter_from_payload(search_payload, "mod0_kpole1", default_search.mod0_kpole1, integer=True),
            mod0_kpole2=_search_parameter_from_payload(search_payload, "mod0_kpole2", default_search.mod0_kpole2, integer=True),
            output_inductance_nh=_search_parameter_from_payload(
                search_payload,
                "output_inductance_nh",
                default_search.output_inductance_nh,
            ),
            effective_lc_inductance_nh=_search_parameter_from_payload(
                search_payload,
                "effective_lc_inductance_nh",
                default_search.effective_lc_inductance_nh,
            ),
        ),
    )


def _experiment_from_payload(payload: dict | None) -> AutotuneExperimentConfig:
    payload = payload or {}
    return AutotuneExperimentConfig(
        board_address=str(payload.get("board_address", DEFAULT_ADDRESS)),
        board_page=_int_field(payload, "board_page", DEFAULT_PAGE),
        board_adapter=str(payload.get("board_adapter", "xdp")),
        response_channel=str(payload.get("response_channel", "CH3")).strip().upper() or "CH3",
        enable_bode_analysis=bool(payload.get("enable_bode_analysis", True)),
        enable_transient_analysis=bool(payload.get("enable_transient_analysis", True)),
        bode_config=dict(payload.get("bode_config", {}) or {}),
        function_generator_config=dict(payload.get("function_generator_config", {}) or {}),
        scope_config=dict(payload.get("scope_config", {}) or {}),
        vout_tolerance_v=_float_field(payload, "vout_tolerance_v", 0.15),
        response_abs_limit_v=_float_field(payload, "response_abs_limit_v", 0.25),
    )


def _search_parameter_from_payload(payload: dict, name: str, default: SearchParameter, integer: bool = False) -> SearchParameter:
    raw = payload.get(name)
    if isinstance(raw, dict):
        center = _float_field(raw, "center", default.center)
        minimum = _float_field(raw, "min", default.min)
        maximum = _float_field(raw, "max", default.max)
        points = _int_field(raw, "points", default.points)
        step = _float_field(raw, "step", default.step)
    else:
        center = _float_field(payload, name, default.center)
        radius = max(abs(default.max - default.center), abs(default.center - default.min))
        minimum = max(default.min, center - radius)
        maximum = min(default.max, center + radius)
        points = default.points
        step = default.step
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    points = max(1, min(101, int(round(points))))
    if points > 1 and maximum > minimum:
        step = (maximum - minimum) / (points - 1)
    elif step <= 0:
        step = default.step if default.step > 0 else 1.0
    center = min(max(center, minimum), maximum)
    if integer:
        center = round(center)
        minimum = round(minimum)
        maximum = round(maximum)
        step = max(1, round(step))
    return SearchParameter(center=float(center), min=float(minimum), max=float(maximum), step=float(step), points=points)


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


def _optional_float_value(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _response_waveform_from_scope(scope_result: dict, response_channel: str) -> Waveform:
    response = response_channel.strip().upper()
    capture_id = str(scope_result.get("capture_id", "") or "")
    cached_entry = SCOPE_CAPTURE_CACHE.get(capture_id) if capture_id else None
    if isinstance(cached_entry, dict):
        channels = cached_entry.get("channels", {})
        if isinstance(channels, dict) and response in channels:
            response_time, response_y = _scope_record_time_and_y(channels[response])
            input_time, input_y = _scope_record_time_and_y(channels.get("CH1", {}))
            if response_time and len(response_time) == len(response_y):
                input_aligned = _align_input_to_response_time(response_time, input_time, input_y)
                if input_aligned:
                    return Waveform(time_s=response_time, vout_v=response_y, input_v=input_aligned)
                return Waveform(time_s=response_time, vout_v=response_y)

    waveforms = scope_result.get("waveforms", [])
    if not isinstance(waveforms, list):
        raise RuntimeError("Scope response did not include waveform data.")

    display_records: dict[str, tuple[list[float], list[float]]] = {}
    for waveform in waveforms:
        if not isinstance(waveform, dict):
            continue
        source = str(waveform.get("source", "")).upper()
        x_values = _float_list(waveform.get("x", []))
        y_values = _float_list(waveform.get("y", []))
        if not x_values or len(x_values) != len(y_values):
            continue
        x_unit = str(waveform.get("x_unit", "s")).lower()
        if x_unit in {"us", "microsecond", "microseconds"}:
            time_s = [value * 1e-6 for value in x_values]
        elif x_unit in {"ms", "millisecond", "milliseconds"}:
            time_s = [value * 1e-3 for value in x_values]
        else:
            time_s = x_values
        display_records[source] = (time_s, y_values)

    if response in display_records:
        response_time, response_y = display_records[response]
        input_time, input_y = display_records.get("CH1", ([], []))
        input_aligned = _align_input_to_response_time(response_time, input_time, input_y)
        if input_aligned:
            return Waveform(time_s=response_time, vout_v=response_y, input_v=input_aligned)
        return Waveform(time_s=response_time, vout_v=response_y)
    raise RuntimeError(f"Scope response channel {response} was not captured.")


def _align_input_to_response_time(
    response_time: list[float],
    input_time: list[float],
    input_y: list[float],
) -> list[float]:
    if not response_time or not input_time or len(input_time) != len(input_y):
        return []
    try:
        response_time_array = np.asarray(response_time, dtype=np.float64)
        input_time_array = np.asarray(input_time, dtype=np.float64)
        input_y_array = np.asarray(input_y, dtype=np.float64)
        valid = np.isfinite(input_time_array) & np.isfinite(input_y_array)
        input_time_array = input_time_array[valid]
        input_y_array = input_y_array[valid]
        if input_time_array.size < 2:
            return []

        order = np.argsort(input_time_array)
        input_time_array = input_time_array[order]
        input_y_array = input_y_array[order]
        unique_time, unique_indices = np.unique(input_time_array, return_index=True)
        unique_y = input_y_array[unique_indices]
        if unique_time.size < 2:
            return []

        aligned = np.interp(response_time_array, unique_time, unique_y)
        return aligned.astype(float).tolist()
    except Exception:
        return []


def _scope_record_time_and_y(record: object) -> tuple[list[float], list[float]]:
    if not isinstance(record, dict):
        return [], []
    x_values = _float_list(record.get("x", []))
    y_values = _float_list(record.get("y", []))
    if not x_values or len(x_values) != len(y_values):
        return [], []
    x_unit = str(record.get("x_unit", "s")).lower()
    if x_unit in {"us", "microsecond", "microseconds"}:
        x_values = [value * 1e-6 for value in x_values]
    elif x_unit in {"ms", "millisecond", "milliseconds"}:
        x_values = [value * 1e-3 for value in x_values]
    return x_values, y_values


def _float_list(values: object) -> list[float]:
    if not isinstance(values, list):
        return []
    result: list[float] = []
    for value in values:
        try:
            result.append(float(value))
        except Exception:
            continue
    return result


def _enforce_scope_response_safety(waveform: Waveform, target_v: float, abs_limit_v: float) -> None:
    if not waveform.vout_v:
        raise RuntimeError("Scope safety check failed: response waveform is empty.")
    low = target_v - abs_limit_v
    high = target_v + abs_limit_v
    min_v = min(waveform.vout_v)
    max_v = max(waveform.vout_v)
    if min_v < low or max_v > high:
        raise RuntimeError(
            f"Scope safety check failed: response range {min_v:.4f} V to {max_v:.4f} V exceeds "
            f"{low:.4f} V to {high:.4f} V."
        )


def _compact_bode_result(result: dict) -> dict:
    return {
        "ok": bool(result.get("ok")),
        "identity": result.get("identity"),
        "resource": result.get("resource"),
        "config": result.get("config"),
        "margins": result.get("margins"),
        "sweep_id": result.get("sweep_id"),
        "data_file": result.get("data_file"),
        "data_file_pending": result.get("data_file_pending"),
        "bode_png": result.get("bode_png"),
        "original_points": result.get("original_points"),
        "display_points": result.get("display_points"),
        "bode_png_pending": result.get("bode_png_pending"),
        "duration_s": result.get("duration_s"),
        "error": result.get("error"),
    }


def _compact_scope_result(result: dict) -> dict:
    compact_waveforms = []
    waveforms = result.get("waveforms", [])
    if not isinstance(waveforms, list):
        waveforms = []
    for waveform in waveforms:
        if not isinstance(waveform, dict):
            continue
        compact_waveforms.append(
            {
                "source": waveform.get("source"),
                "x_unit": waveform.get("x_unit"),
                "y_unit": waveform.get("y_unit"),
                "time_span_s": waveform.get("time_span_s"),
                "original_points": waveform.get("original_points"),
                "display_points": waveform.get("display_points"),
                "display_strategy": waveform.get("display_strategy"),
                "capture_id": waveform.get("capture_id"),
                "data_file": waveform.get("data_file"),
                "data_file_pending": waveform.get("data_file_pending"),
                "transfer_encoding": waveform.get("transfer_encoding"),
            }
        )
    return {
        "ok": bool(result.get("ok")),
        "resource": result.get("resource"),
        "capture_id": result.get("capture_id"),
        "scope_png": result.get("scope_png"),
        "scope_png_error": result.get("scope_png_error"),
        "scope_png_pending": result.get("scope_png_pending"),
        "channels": result.get("channels"),
        "measurements": result.get("measurements"),
        "measurement_values": result.get("measurement_values"),
        "waveforms": compact_waveforms,
        "function_generator_frequency_hz": result.get("function_generator_frequency_hz"),
        "scope_window_s": result.get("scope_window_s"),
        "scope_actual_window_s": result.get("scope_actual_window_s"),
        "scope_trigger_source": result.get("scope_trigger_source"),
        "scope_trigger_slope": result.get("scope_trigger_slope"),
        "scope_trigger_offset_from_left_s": result.get("scope_trigger_offset_from_left_s"),
        "duration_s": result.get("duration_s"),
        "error": result.get("error"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
