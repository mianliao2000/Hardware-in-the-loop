"""Local web GUI and API server for the hardware PID autotuner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import copy
import json
import math
import mimetypes
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import TYPE_CHECKING
from urllib import error as urllib_error
from urllib.parse import parse_qs, quote_plus, urlparse
from urllib import request as urllib_request
import uuid
import re

import numpy as np

if TYPE_CHECKING:
    from PIL import Image, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADDRESS = "0x5E"
DEFAULT_PAGE = 0


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")

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
    ResponseMetrics,
    SearchSpace,
    SearchParameter,
    TuningConfig,
    TuningTargets,
    Waveform,
    automatic_search_parameter,
)
from hardware.tuning.drl import DrlWorkflowManager


DEVICE_LOCK = threading.Lock()
# Matplotlib is not reliably thread-safe. A single bounded worker keeps file
# saving/rendering off the hardware path without letting work accumulate over a
# long auto-tune run.
ARTIFACT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autotune-artifact")
ARTIFACT_QUEUE_SLOTS = threading.BoundedSemaphore(2)
FRONTEND_DIST_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
SCOPE_CONNECTIONS: dict[str, TektronixOscilloscope] = {}
BODE_CONNECTIONS: dict[str, dict[str, object]] = {}
BODE_CONNECTION_LOCK = threading.Lock()
SCOPE_CAPTURE_DIR = ROOT / "data" / "scope_captures"
BODE_SWEEP_DIR = ROOT / "data" / "bode_sweeps"
RESULTS_DIR = ROOT / "results"
LATEST_SCOPE_PNG = RESULTS_DIR / "latest_scope_capture.png"
LATEST_BODE_PNG = RESULTS_DIR / "latest_bode_sweep.png"
AUTOTUNE_RUN_DIR = RESULTS_DIR / "autotune_runs"
AUTOTUNE_RECENT_DIR = AUTOTUNE_RUN_DIR / "recent"
AUTOTUNE_SAVED_DIR = AUTOTUNE_RUN_DIR / "saved"
# Targeted experiments can span a base run plus one or more recovery runs.
# Keep enough Recent entries that starting a recovery cannot evict the source
# measurements before they are aggregated or explicitly saved.
AUTOTUNE_RECENT_LIMIT = 10
AUTOTUNE_RECENT_MIN_ITERATIONS = 10
SCOPE_CAPTURE_CACHE: dict[str, dict] = {}
SCOPE_CAPTURE_CACHE_LIMIT = 1
SCOPE_DISPLAY_MAX_POINTS = 140_000
SCOPE_TRIGGER_OFFSET_FROM_LEFT_S = 2e-6
SCOPE_AUTOTUNE_MAX_RECORD_LENGTH = 250_000
SCOPE_AUTOTUNE_MAINTENANCE_INTERVAL = 100
SCOPE_AUTOTUNE_CAPTURE_ATTEMPTS = 3
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


def _submit_artifact(function, *args) -> bool:
    # Plot generation is optional and can be rebuilt from NPZ data. Never let
    # a saturated renderer stall the hardware measurement thread.
    if not ARTIFACT_QUEUE_SLOTS.acquire(blocking=False):
        return False

    def run_artifact() -> None:
        try:
            function(*args)
        except Exception as exc:
            print(f"Artifact task failed: {exc}", file=sys.stderr)
        finally:
            ARTIFACT_QUEUE_SLOTS.release()

    try:
        ARTIFACT_EXECUTOR.submit(run_artifact)
    except Exception:
        ARTIFACT_QUEUE_SLOTS.release()
        raise
    return True


class ServerHardwareExperimentRunner:
    """Run one real hardware tuning candidate through the existing bench APIs."""

    supports_split_analysis = True

    def __init__(self) -> None:
        self._iteration_context: int | None = None

    def set_iteration_context(self, iteration: int) -> None:
        self._iteration_context = max(1, int(iteration))

    def restore_candidate(
        self,
        candidate: HardwarePidCandidate,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
    ) -> dict:
        """Program a confirmed candidate without launching another measurement."""

        started = time.perf_counter()
        vout = _prepare_vout_for_autotune(
            address=experiment.board_address,
            page=experiment.board_page,
            adapter_kind=experiment.board_adapter,
            voltage=config.targets.vout_target_v,
            tolerance_v=experiment.vout_tolerance_v,
        )
        if not vout.get("ok"):
            return {"ok": False, "error": f"VOUT restore check failed: {vout.get('error')}"}
        pid = _set_xdp_pid(
            address=experiment.board_address,
            page=experiment.board_page,
            adapter_kind=experiment.board_adapter,
            values={**candidate.pid_values(), **candidate.current_mode_values()},
        )
        if not pid.get("ok"):
            return {"ok": False, "error": f"Confirmed PID restore failed: {pid.get('error')}"}
        bandwidth = _set_mod0_ll_bandwidth(
            address=experiment.board_address,
            page=experiment.board_page,
            adapter_kind=experiment.board_adapter,
            value=candidate.avp_bandwidth_value(),
        )
        if not bandwidth.get("ok"):
            return {"ok": False, "error": f"Confirmed AVP bandwidth restore failed: {bandwidth.get('error')}"}
        inductance = _set_inductance(
            address=experiment.board_address,
            page=experiment.board_page,
            adapter_kind=experiment.board_adapter,
            output_inductance_nh=candidate.output_inductance_nh,
            effective_lc_inductance_nh=candidate.effective_lc_inductance_nh,
        )
        if not inductance.get("ok"):
            return {"ok": False, "error": f"Confirmed inductance restore failed: {inductance.get('error')}"}
        return {
            "ok": True,
            "candidate": {
                **candidate.pid_values(),
                **candidate.current_mode_values(),
                "mod0_ll_bw": candidate.avp_bandwidth_value(),
            },
            "duration_s": round(time.perf_counter() - started, 3),
        }

    def recover_after_transient_protection(
        self,
        experiment: AutotuneExperimentConfig,
        config: TuningConfig | None = None,
    ) -> dict:
        started = time.perf_counter()
        steps: list[dict] = []

        def run_step(name: str, action) -> dict:
            step_started = time.perf_counter()
            result = action()
            if isinstance(result, dict):
                step = dict(result)
            else:
                step = {"ok": False, "error": "Recovery action returned a non-dict result."}
            step["name"] = name
            step["duration_s"] = round(time.perf_counter() - step_started, 3)
            steps.append(step)
            return step

        try:
            run_step(
                "pmbus_output_disable",
                lambda: _set_pmbus_output(
                    address=experiment.board_address,
                    page=experiment.board_page,
                    adapter_kind=experiment.board_adapter,
                    action="disable",
                ),
            )
            run_step(
                "xdp_output_disable",
                lambda: _set_xdp_output(
                    address=experiment.board_address,
                    page=experiment.board_page,
                    adapter_kind=experiment.board_adapter,
                    action="disable",
                ),
            )
            disable_failures = [step for step in steps[-2:] if not step.get("ok")]
            if disable_failures:
                return {
                    "ok": False,
                    "error": "Could not confirm both outputs disabled; recovery stopped fail-closed.",
                    "steps": steps,
                    "duration_s": round(time.perf_counter() - started, 3),
                }
            time.sleep(0.25)
            if config is not None:
                search = config.search
                baseline = HardwarePidCandidate(
                    mod0_kp=int(round(search.mod0_kp.center)),
                    mod0_ki=int(round(search.mod0_ki.center)),
                    mod0_kd=int(round(search.mod0_kd.center)),
                    mod0_kpole1=3 if abs(search.mod0_kpole1.center - 3) <= abs(search.mod0_kpole1.center - 6) else 6,
                    mod0_kpole2=3 if abs(search.mod0_kpole2.center - 3) <= abs(search.mod0_kpole2.center - 6) else 6,
                    mod0_cm_gain=int(round(search.mod0_cm_gain.clamped(search.mod0_cm_gain.center))),
                    mod0_ll_bw=int(round(search.mod0_ll_bw.clamped(search.mod0_ll_bw.center))),
                    output_inductance_nh=float(search.output_inductance_nh.center),
                    effective_lc_inductance_nh=float(search.effective_lc_inductance_nh.center),
                    phase="safe_recovery_baseline",
                )
                run_step(
                    "restore_safe_pid_baseline",
                    lambda: _set_xdp_pid(
                        address=experiment.board_address,
                        page=experiment.board_page,
                        adapter_kind=experiment.board_adapter,
                        values={**baseline.pid_values(), **baseline.current_mode_values()},
                    ),
                )
                run_step(
                    "restore_safe_avp_bandwidth_baseline",
                    lambda: _set_mod0_ll_bandwidth(
                        address=experiment.board_address,
                        page=experiment.board_page,
                        adapter_kind=experiment.board_adapter,
                        value=baseline.avp_bandwidth_value(),
                    ),
                )
                run_step(
                    "restore_safe_inductance_baseline",
                    lambda: _set_inductance(
                        address=experiment.board_address,
                        page=experiment.board_page,
                        adapter_kind=experiment.board_adapter,
                        output_inductance_nh=baseline.output_inductance_nh,
                        effective_lc_inductance_nh=baseline.effective_lc_inductance_nh,
                    ),
                )
                baseline_failures = [step for step in steps[-3:] if not step.get("ok")]
                if baseline_failures:
                    return {
                        "ok": False,
                        "error": "Safe baseline restore failed; outputs remain disabled.",
                        "steps": steps,
                        "duration_s": round(time.perf_counter() - started, 3),
                    }
            run_step(
                "pmbus_output_enable",
                lambda: _set_pmbus_output(
                    address=experiment.board_address,
                    page=experiment.board_page,
                    adapter_kind=experiment.board_adapter,
                    action="enable",
                ),
            )
            run_step(
                "xdp_output_enable",
                lambda: _set_xdp_output(
                    address=experiment.board_address,
                    page=experiment.board_page,
                    adapter_kind=experiment.board_adapter,
                    action="enable",
                ),
            )
            failed = [step for step in steps if not step.get("ok")]
            return {
                "ok": not failed,
                "error": "; ".join(str(step.get("error")) for step in failed if step.get("error")) or None,
                "steps": steps,
                "duration_s": round(time.perf_counter() - started, 3),
            }
        except Exception as exc:
            return {
                "ok": False,
                "error": str(exc),
                "steps": steps,
                "duration_s": round(time.perf_counter() - started, 3),
            }

    def evaluate(
        self,
        candidate: HardwarePidCandidate,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
    ) -> ExperimentResult:
        started = time.perf_counter()
        iteration_number = self._iteration_context
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
        # Metrics remain synchronous because adaptive search needs them before
        # selecting the next candidate. Only independent NPZ/PNG artifacts may
        # use the bounded background pipeline during continuous auto-tune.
        async_artifacts = bool(experiment.async_artifacts)
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

            pid_values = candidate.pid_values()
            pid_values.update(candidate.current_mode_values())
            pid_write = _set_xdp_pid(
                address=experiment.board_address,
                page=experiment.board_page,
                adapter_kind=experiment.board_adapter,
                values=pid_values,
            )
            write_results["xdp_pid"] = pid_write
            mark_stage("write_pid")
            if not pid_write.get("ok"):
                raise RuntimeError(f"PID write failed: {pid_write.get('error')}")

            bandwidth_write = _set_mod0_ll_bandwidth(
                address=experiment.board_address,
                page=experiment.board_page,
                adapter_kind=experiment.board_adapter,
                value=candidate.avp_bandwidth_value(),
            )
            write_results["avp_bandwidth"] = bandwidth_write
            mark_stage("write_avp_bandwidth")
            if not bandwidth_write.get("ok"):
                raise RuntimeError(f"AVP bandwidth write failed: {bandwidth_write.get('error')}")

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
                source_vpp = _optional_bode_source_vpp(bode_cfg, default=0.1)
                source_dbm = _optional_bode_source_dbm(bode_cfg)
                bode_result = _run_bode_sweep(
                    host=str(bode_cfg.get("host", "127.0.0.1")),
                    port=int(bode_cfg.get("port", 5025)),
                    start_hz=float(bode_cfg.get("start_hz", 1000.0)),
                    stop_hz=float(bode_cfg.get("stop_hz", 1_000_000.0)),
                    points=int(bode_cfg.get("points", 201)),
                    bandwidth_hz=float(bode_cfg.get("bandwidth_hz", 300.0)),
                    source_vpp=source_vpp,
                    source_dbm=source_dbm,
                    timeout_ms=int(bode_cfg.get("timeout_ms", 60000)),
                    async_artifacts=async_artifacts,
                    reuse_session=True,
                    iteration_number=iteration_number,
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
                    iteration_number=iteration_number,
                    response_channel=response_channel,
                    response_targets=config.targets,
                )
                mark_stage("scope_capture")
                if not scope_result.get("ok"):
                    raise RuntimeError(f"Scope capture failed: {scope_result.get('error')}")

                waveform = _response_waveform_from_scope(scope_result, response_channel)
                _enforce_scope_response_safety(waveform, config.targets.vout_target_v, experiment.response_abs_limit_v)

            capture_metrics = None
            if scope_result:
                capture_id = str(scope_result.get("capture_id", "") or "")
                capture_entry = SCOPE_CAPTURE_CACHE.get(capture_id) if capture_id else None
                if isinstance(capture_entry, dict):
                    cached_metrics = capture_entry.get("settling_metrics")
                    if isinstance(cached_metrics, ResponseMetrics):
                        capture_metrics = cached_metrics

            metrics = ResponseAnalyzer(config.targets).analyze_hardware(
                waveform,
                bode_result.get("margins") if bode_result else None,
                enable_transient=experiment.enable_transient_analysis,
                enable_bode=experiment.enable_bode_analysis,
                precomputed_transient=capture_metrics,
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

class AutotuneRunStore:
    """Keep a rolling local history of hardware auto-tune runs."""

    def __init__(self, recent_dir: Path, saved_dir: Path, recent_limit: int, recent_min_iterations: int = 10):
        self.recent_dir = recent_dir
        self.saved_dir = saved_dir
        self.recent_limit = recent_limit
        self.recent_min_iterations = recent_min_iterations
        self._lock = threading.RLock()
        self._current_run_id: str | None = None
        self._current_run_kind: str | None = None
        self._persisted_iterations = 0
        self._history_generation = 0
        # Status polling continues after a run reaches a terminal state. Keep
        # the final write idempotent so each poll does not rewrite the full run.
        self._last_persisted_signature: tuple[object, ...] | None = None

    def start_new(self, status: dict | None = None) -> dict:
        with self._lock:
            self._history_generation += 1
            self.recent_dir.mkdir(parents=True, exist_ok=True)
            previous_run_id = self._current_run_id
            previous_kind = self._current_run_kind
            if previous_run_id and previous_kind == "recent":
                previous_dir = self.recent_dir / previous_run_id
                previous_status = self._read_json(previous_dir / "run_status.json") or {}
                previous_history = previous_status.get("history")
                previous_iterations = len(previous_history) if isinstance(previous_history, list) else 0
                # Starting a new search supersedes a paused short trial, so it
                # should not remain in Recent as an unusable partial result.
                if previous_iterations < self.recent_min_iterations:
                    shutil.rmtree(previous_dir, ignore_errors=True)
            run_dir = self._next_friendly_run_dir("recent")
            run_id = run_dir.name
            (run_dir / "files").mkdir(parents=True, exist_ok=True)
            self._current_run_id = run_id
            self._current_run_kind = "recent"
            self._persisted_iterations = 0
            self._last_persisted_signature = None
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
            run_kind = self._current_run_kind or "recent"
            run_dir = (self.saved_dir if run_kind == "saved" else self.recent_dir) / run_id
            (run_dir / "files").mkdir(parents=True, exist_ok=True)
            next_status["run"] = self._run_payload(run_id, run_kind, run_dir)
            axis_settings = _scope_axis_settings_from_status(next_status)
            previous_status = self._read_json(run_dir / "run_status.json") or {}
            previous_history = previous_status.get("history")
            if not isinstance(previous_history, list):
                previous_history = []
            previous_by_iteration: dict[int, dict] = {}
            for old_record in previous_history:
                if not isinstance(old_record, dict):
                    continue
                try:
                    old_iteration = int(old_record.get("iteration") or 0)
                except (TypeError, ValueError):
                    old_iteration = 0
                if old_iteration > 0:
                    previous_by_iteration[old_iteration] = old_record

            persisted_cutoff = min(self._persisted_iterations, len(history))
            merged_history: list[dict] = []
            new_records_for_log: list[dict] = []
            for index, record in enumerate(history):
                if not isinstance(record, dict):
                    continue
                try:
                    iteration = int(record.get("iteration") or index + 1)
                except (TypeError, ValueError):
                    iteration = index + 1
                if index < persisted_cutoff and iteration in previous_by_iteration:
                    merged_history.append(copy.deepcopy(previous_by_iteration[iteration]))
                    continue
                # Scope/Bode capture already creates its raw data and PNG
                # artifacts during the iteration. Rebuilding every plot here
                # made the last status update replay an entire 100+ point run.
                # Artifact materialization is reserved for explicit rebuilds
                # and archive operations, where the user expects that work.
                merged_history.append(record)
                if index >= self._persisted_iterations:
                    new_records_for_log.append(record)
            next_status["history"] = merged_history

            history_by_iteration: dict[int, dict] = {}
            for record in merged_history:
                try:
                    iteration = int(record.get("iteration") or 0)
                except (TypeError, ValueError):
                    iteration = 0
                if iteration > 0:
                    history_by_iteration[iteration] = record
            for key in ("current", "best"):
                record = next_status.get(key)
                if not isinstance(record, dict):
                    continue
                try:
                    iteration = int(record.get("iteration") or 0)
                except (TypeError, ValueError):
                    iteration = 0
                if iteration in history_by_iteration:
                    next_status[key] = history_by_iteration[iteration]

            if next_status.get("state") != "running":
                _refresh_status_artifact_readiness(next_status)

            if new_records_for_log:
                with (run_dir / "iterations.jsonl").open("a", encoding="utf-8") as handle:
                    for record in new_records_for_log:
                        handle.write(json.dumps(record, indent=None, ensure_ascii=False) + "\n")
                self._persisted_iterations = len(history)
            self._write_json(run_dir / "run_status.json", next_status)
            self._write_summary(run_dir, next_status)
            self._last_persisted_signature = self._status_persistence_signature(next_status)
            if self._discard_terminal_short_recent_run(run_dir, run_kind, next_status):
                return next_status
            if run_kind == "recent":
                self._enforce_recent_limit()
            return next_status

    def persist_iteration_record(self, record: dict, terminal: bool = False) -> None:
        """Durably append one iteration without rewriting the full run on every step."""

        with self._lock:
            if self._current_run_id is None or not isinstance(record, dict):
                return
            try:
                iteration = int(record.get("iteration") or 0)
            except (TypeError, ValueError):
                return
            if iteration <= self._persisted_iterations:
                return
            run_kind = self._current_run_kind or "recent"
            run_dir = (self.saved_dir if run_kind == "saved" else self.recent_dir) / self._current_run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            payload = (json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
            descriptor = os.open(str(run_dir / "iterations.jsonl"), os.O_APPEND | os.O_CREAT | os.O_WRONLY)
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._persisted_iterations = iteration

        # JSONL is the per-iteration crash-safe journal. Refresh the larger
        # status snapshot periodically and once at the terminal state.
        if terminal or iteration % 10 == 0:
            self.persist_status(TUNING_SESSION.status())

    def needs_persist(self, status: dict) -> bool:
        """Return whether a status poll contains unsaved run state."""

        with self._lock:
            if self._current_run_id is None:
                return False
            return self._last_persisted_signature != self._status_persistence_signature(status)

    @staticmethod
    def _status_persistence_signature(status: dict) -> tuple[object, ...]:
        history = status.get("history") if isinstance(status.get("history"), list) else []
        current = status.get("current") if isinstance(status.get("current"), dict) else {}
        best = status.get("best") if isinstance(status.get("best"), dict) else {}
        return (
            status.get("state"),
            status.get("message"),
            len(history),
            current.get("iteration"),
            best.get("iteration"),
        )

    def stop_current(self) -> None:
        with self._lock:
            self._history_generation += 1
            self._current_run_id = None
            self._current_run_kind = None
            self._persisted_iterations = 0
            self._last_persisted_signature = None

    def archive_current(self, name: str | None = None) -> dict:
        with self._lock:
            if self._current_run_id is None:
                status = TUNING_SESSION.status()
                self.persist_status(status)
            if self._current_run_id is None:
                raise RuntimeError("No auto-tune run is available to save.")
            source_kind = self._current_run_kind or "recent"
            source = (self.saved_dir if source_kind == "saved" else self.recent_dir) / self._current_run_id
            if not source.exists():
                raise RuntimeError("Current auto-tune result folder does not exist yet.")
            if source_kind == "saved":
                status = self._read_json(source / "run_status.json")
                if isinstance(status, dict):
                    status = copy.deepcopy(status)
                    status["run"] = self._run_payload(source.name, "saved", source)
                    self._write_json(source / "run_status.json", status)
                    self._write_summary(source, status)
                summary = self._read_json(source / "summary.json") or {}
                summary.update({
                    "run_id": source.name,
                    "display_name": self._display_name_for_run(source.name, "saved"),
                    "kind": "saved",
                    "archived_at": summary.get("archived_at") or time.time(),
                    "path": _path_label(source),
                })
                self._write_json(source / "summary.json", summary)
                return {"ok": True, "saved_run": summary}
            self.saved_dir.mkdir(parents=True, exist_ok=True)
            # Recent runs retain references to the fast live artifacts. Make
            # a self-contained copy only when the user explicitly archives it.
            source_run_id = self._current_run_id
            self._materialize_run_assets(source)
            target = self._next_friendly_run_dir("saved", name)
            # Recent and saved runs live on the same volume. Moving the
            # self-contained run avoids copying every waveform a second time.
            shutil.move(str(source), str(target))
            status = self._read_json(target / "run_status.json")
            if isinstance(status, dict):
                self._repair_run_status_artifact_paths(target, status)
                status["run"] = self._run_payload(target.name, "saved", target)
                self._write_json(target / "run_status.json", status)
                self._write_summary(target, status)
            summary = self._read_json(target / "summary.json") or {}
            summary.update({
                "run_id": target.name,
                "display_name": self._display_name_for_run(target.name, "saved"),
                "source_run_id": source_run_id,
                "kind": "saved",
                "archived_at": time.time(),
                "path": _path_label(target),
            })
            self._write_json(target / "summary.json", summary)
            self._current_run_id = target.name
            self._current_run_kind = "saved"
            if isinstance(status, dict) and isinstance(status.get("history"), list):
                self._persisted_iterations = len(status["history"])
            return {"ok": True, "saved_run": summary}

    def _materialize_run_assets(self, run_dir: Path) -> None:
        status = self._read_json(run_dir / "run_status.json")
        if not isinstance(status, dict):
            return
        axis_settings = _scope_axis_settings_from_status(status)
        history = status.get("history") if isinstance(status.get("history"), list) else []
        for record in history:
            if isinstance(record, dict):
                self._copy_record_assets(record, run_dir, scope_axis_settings=axis_settings)
        by_iteration = {
            int(record.get("iteration") or 0): record
            for record in history
            if isinstance(record, dict)
        }
        for key in ("current", "best"):
            record = status.get(key)
            if not isinstance(record, dict):
                continue
            iteration = int(record.get("iteration") or 0)
            if iteration in by_iteration:
                status[key] = by_iteration[iteration]
            else:
                self._copy_record_assets(record, run_dir, scope_axis_settings=axis_settings)
        self._write_json(run_dir / "run_status.json", status)
        self._write_summary(run_dir, status)

    def archive_run(self, run_id: str, kind: str = "recent", name: str | None = None) -> dict:
        with self._lock:
            source = self._run_dir(kind, run_id)
            if not source.exists():
                raise RuntimeError("Selected auto-tune result folder does not exist.")
            status = self._read_json(source / "run_status.json")
            if not isinstance(status, dict):
                raise RuntimeError(f"No saved status was found for run '{run_id}'.")
            if kind == "saved":
                summary = self._read_json(source / "summary.json") or {}
                summary.update({
                    "run_id": source.name,
                    "display_name": self._display_name_for_run(source.name, "saved"),
                    "kind": "saved",
                    "archived_at": summary.get("archived_at") or time.time(),
                    "path": _path_label(source),
                })
                self._write_json(source / "summary.json", summary)
                return {"ok": True, "saved_run": summary}
            self.saved_dir.mkdir(parents=True, exist_ok=True)
            self._materialize_run_assets(source)
            status = self._read_json(source / "run_status.json") or status
            target = self._next_friendly_run_dir("saved", name)
            shutil.move(str(source), str(target))
            self._repair_run_status_artifact_paths(target, status)
            status["run"] = self._run_payload(target.name, "saved", target)
            self._write_json(target / "run_status.json", status)
            self._write_summary(target, status)
            summary = self._read_json(target / "summary.json") or {}
            summary.update({
                "run_id": target.name,
                "display_name": self._display_name_for_run(target.name, "saved"),
                "source_run_id": run_id,
                "source_kind": kind,
                "kind": "saved",
                "archived_at": time.time(),
                "path": _path_label(target),
            })
            self._write_json(target / "summary.json", summary)
            if run_id == self._current_run_id and (self._current_run_kind or "recent") == "recent":
                self._current_run_id = target.name
                self._current_run_kind = "saved"
                self._persisted_iterations = len(status.get("history") if isinstance(status.get("history"), list) else [])
            return {"ok": True, "saved_run": summary}

    def delete_run(self, run_id: str, kind: str = "recent") -> dict:
        with self._lock:
            run_dir = self._run_dir(kind, run_id)
            if run_dir.name == self._current_run_id and kind == (self._current_run_kind or "recent"):
                self.stop_current()
            shutil.rmtree(run_dir, ignore_errors=True)
            return {"ok": True, "deleted_run_id": run_id, "kind": kind}

    def list_runs(self) -> dict:
        with self._lock:
            return {
                "ok": True,
                "current_run_id": self._current_run_id,
                "current_run_kind": self._current_run_kind,
                "recent": self._list_kind(self.recent_dir, "recent"),
                "saved": self._list_kind(self.saved_dir, "saved"),
            }

    def load_run(self, run_id: str, kind: str = "recent") -> dict:
        with self._lock:
            run_dir = self._run_dir(kind, run_id)
            status = self._read_json(run_dir / "run_status.json")
            if not isinstance(status, dict):
                raise RuntimeError(f"No saved status was found for run '{run_id}'.")
            status = copy.deepcopy(status)
            repaired = self._repair_run_status_artifact_paths(run_dir, status)
            repaired = _refresh_status_artifact_readiness(status) or repaired
            saved_state = str(status.get("state") or "")
            if saved_state in {"running", "paused"}:
                status["loaded_state"] = saved_state
                status["state"] = "stopped"
                status["can_resume_saved_run"] = True
                status["message"] = "Loaded saved result snapshot. Press Resume to continue from the next candidate, or load another result."
            status["run"] = self._run_payload(run_dir.name, kind, run_dir)
            if repaired:
                self._write_json(run_dir / "run_status.json", status)
                self._write_summary(run_dir, status)
            return {"ok": True, **_compact_loaded_status_for_client(status)}

    def resume_run(self, run_id: str, kind: str = "recent") -> dict:
        with self._lock:
            source = self._run_dir(kind, run_id)
            status = self._read_json(source / "run_status.json")
            if not isinstance(status, dict):
                raise RuntimeError(f"No saved status was found for run '{run_id}'.")
            if str(status.get("state") or "") == "complete":
                raise RuntimeError("Selected auto-tune result is already complete.")
            run_kind = "saved" if kind == "saved" else "recent"
            run_dir = source
            self._current_run_id = run_dir.name
            self._current_run_kind = run_kind
            self._history_generation += 1
            self._persisted_iterations = len(status.get("history") if isinstance(status.get("history"), list) else [])
            status = copy.deepcopy(status)
            status["run"] = self._run_payload(run_dir.name, run_kind, run_dir)
            restored = TUNING_SESSION.restore(status)
            resumed = TUNING_SESSION.resume()
            resumed = self.persist_status(resumed)
            if run_kind == "recent":
                self._enforce_recent_limit()
            return {"ok": True, "restored": restored, **resumed}

    def history_token(self) -> str:
        """Identify one in-memory history so clients never merge different runs."""

        with self._lock:
            run_id = self._current_run_id or "none"
            run_kind = self._current_run_kind or "session"
            return f"{os.getpid()}:{self._history_generation}:{run_kind}:{run_id}"

    def save_animation_gif(self, run_id: str | None = None, kind: str = "recent", duration_ms: int = 100) -> dict:
        with self._lock:
            if run_id is None:
                if self._current_run_id is None:
                    status = TUNING_SESSION.status()
                    self.persist_status(status)
                run_id = self._current_run_id
                kind = self._current_run_kind or "recent"
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
            combined = self._make_combined_gif(
                history,
                target=output_dir / "combined_response_bode.gif",
                duration_ms=duration_ms,
                scope_axis_settings=_scope_axis_settings_from_status(status),
            )
            generated_at = time.time()
            if combined is None:
                raise RuntimeError("No GIF frames were generated. This run may not contain scope or Bode images.")
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

    def open_animation_gif(self, run_id: str | None = None, kind: str = "recent", duration_ms: int = 100) -> dict:
        with self._lock:
            if run_id is None:
                if self._current_run_id is None:
                    status = TUNING_SESSION.status()
                    self.persist_status(status)
                run_id = self._current_run_id
                kind = self._current_run_kind or "recent"
            if run_id is None:
                raise RuntimeError("No auto-tune run is available for GIF export.")
            run_dir = self._run_dir(kind, run_id)
            output_dir = run_dir / "animations"
            gif_path = output_dir / "combined_response_bode.gif"
            existing_gif = gif_path.exists()

        if existing_gif:
            result = {
                "ok": True,
                "run_id": run_id,
                "kind": kind,
                "generated_at": gif_path.stat().st_mtime,
                "duration_ms": max(50, min(5000, int(duration_ms))),
                "animation_dir": str(output_dir.relative_to(ROOT)).replace("\\", "/"),
                "combined_gif": _scope_png_public_path(gif_path),
                "transient_gif": None,
                "bode_gif": None,
                "used_existing": True,
            }
        else:
            result = self.save_animation_gif(run_id, kind, duration_ms)
            gif_public_path = result.get("combined_gif") or result.get("transient_gif") or result.get("bode_gif")
            resolved = _path_from_result_reference(gif_public_path)
            if resolved is None:
                raise RuntimeError("GIF was generated, but the GIF file could not be found.")
            gif_path = resolved
            result["used_existing"] = False
        if not gif_path.exists():
            raise RuntimeError("GIF file could not be found.")
        os.startfile(str(gif_path))
        result["opened"] = True
        result["opened_path"] = str(gif_path.relative_to(ROOT)).replace("\\", "/")
        return result

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
                        if self._make_combined_gif(
                            history,
                            target=output_dir / "combined_response_bode.gif",
                            duration_ms=100,
                            scope_axis_settings=axis_settings,
                        ):
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

    def _make_combined_gif(
        self,
        history: list,
        *,
        target: Path,
        duration_ms: int = 100,
        scope_axis_settings: dict | None = None,
    ) -> Path | None:
        from PIL import Image

        frames: list[Image.Image] = []
        trend_records = self._penalty_trend_records(history)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame_cache_dir = target.parent / "frame_cache"
        frame_cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = frame_cache_dir / "manifest.json"
        cached_manifest = self._read_json(manifest_path) or {}
        cache_context = {
            "version": 5,
            "layout": "scope_bode_left_penalty_right",
            "trend": trend_records,
            "scope_axis_settings": scope_axis_settings or {},
        }
        cached_frames = (
            cached_manifest.get("frames")
            if cached_manifest.get("context") == cache_context and isinstance(cached_manifest.get("frames"), dict)
            else {}
        )
        next_manifest_frames: dict[str, dict] = {}
        used_cache_files: set[Path] = set()
        try:
            for record in history:
                if not isinstance(record, dict):
                    continue
                if self._record_is_skipped_trip(record) or self._record_is_invalid_bode(record):
                    continue
                scope_result = record.get("scope_result") if isinstance(record.get("scope_result"), dict) else None
                bode_result = record.get("bode_result") if isinstance(record.get("bode_result"), dict) else None
                scope_path = _path_from_result_reference(scope_result.get("scope_png")) if isinstance(scope_result, dict) else None
                bode_path = _path_from_result_reference(bode_result.get("bode_png")) if isinstance(bode_result, dict) else None
                has_scope = (scope_path is not None and scope_path.exists()) or isinstance(scope_result, dict)
                bode_data_file = _path_from_result_reference(bode_result.get("data_file")) if isinstance(bode_result, dict) else None
                has_bode = (
                    (bode_path is not None and bode_path.exists())
                    or (bode_data_file is not None and bode_data_file.exists())
                )
                if not has_scope and not has_bode:
                    continue
                try:
                    iteration = int(record.get("iteration") or 0)
                    width = self._gif_frame_width(scope_path, bode_path)
                    height = max(900, min(1400, round(width * 0.58)))
                    frame_signature = {
                        "width": width,
                        "height": height,
                        "scope": self._gif_result_signature(scope_result, scope_path),
                        "bode": self._gif_result_signature(bode_result, bode_path),
                    }
                    cache_key = str(iteration)
                    cache_path = frame_cache_dir / f"iteration_{iteration:04d}.png"
                    used_cache_files.add(cache_path)
                    next_manifest_frames[cache_key] = frame_signature
                    if cached_frames.get(cache_key) == frame_signature and cache_path.exists():
                        with Image.open(cache_path) as cached:
                            frames.append(cached.copy())
                        continue

                    combined = Image.new("RGB", (width, height), "white")
                    if has_scope and has_bode:
                        left_width = max(1, round(width * 0.62))
                        right_width = max(1, width - left_width)
                        plot_gap = 6
                        available_left_height = max(2, height - plot_gap)
                        top_height = max(1, available_left_height // 2)
                        bottom_height = max(1, available_left_height - top_height)
                        scope = self._render_scope_for_gif(
                            scope_result,
                            scope_path,
                            left_width,
                            top_height,
                            scope_axis_settings,
                            target.parent,
                            iteration,
                        )
                        bode = self._render_bode_for_gif(
                            bode_result,
                            bode_path,
                            left_width,
                            bottom_height,
                            target.parent,
                            iteration,
                        )
                        compact_scope = self._resize_gif_plot_to_width(scope, left_width)
                        compact_bode = self._resize_gif_plot_to_width(bode, left_width)
                        stack_height = compact_scope.height + plot_gap + compact_bode.height
                        if stack_height > height:
                            stack_scale = height / stack_height
                            compact_scope = self._resize_gif_plot(compact_scope, stack_scale)
                            compact_bode = self._resize_gif_plot(compact_bode, stack_scale)
                            stack_height = compact_scope.height + plot_gap + compact_bode.height
                        stack_top = max(0, (height - stack_height) // 2)
                        trend = self._make_penalty_trend_frame(
                            trend_records,
                            record,
                            right_width,
                            stack_height,
                        )
                        combined.paste(compact_scope, (0, stack_top))
                        combined.paste(compact_bode, (0, stack_top + compact_scope.height + plot_gap))
                        combined.paste(trend, (left_width, stack_top))
                        compact_scope.close()
                        compact_bode.close()
                        bode.close()
                        scope.close()
                        trend.close()
                    else:
                        plot_height = max(1, round(height * 0.68))
                        trend_height = max(1, height - plot_height)
                        if has_scope:
                            plot = self._render_scope_for_gif(
                                scope_result,
                                scope_path,
                                width,
                                plot_height,
                                scope_axis_settings,
                                target.parent,
                                iteration,
                            )
                        else:
                            plot = self._render_bode_for_gif(
                                bode_result,
                                bode_path,
                                width,
                                plot_height,
                                target.parent,
                                iteration,
                            )
                        trend = self._make_penalty_trend_frame(trend_records, record, width, trend_height)
                        combined.paste(plot, (0, 0))
                        combined.paste(trend, (0, plot_height))
                        plot.close()
                        trend.close()
                    palette_frame = self._quantize_gif_frame(combined)
                    combined.close()
                    palette_frame.save(cache_path, format="PNG", optimize=False)
                    frames.append(palette_frame)
                except Exception:
                    continue
            if not frames:
                return None
            self._write_json(manifest_path, {"context": cache_context, "frames": next_manifest_frames})
            for cached_path in frame_cache_dir.glob("iteration_*.png"):
                if cached_path not in used_cache_files:
                    cached_path.unlink(missing_ok=True)
            if target.exists():
                target.unlink()
            frames[0].save(
                target,
                save_all=True,
                append_images=frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
            )
            return target
        finally:
            for frame in frames:
                frame.close()

    @staticmethod
    def _gif_frame_width(scope_path: Path | None, bode_path: Path | None) -> int:
        from PIL import Image

        widths: list[int] = []
        for path in (scope_path, bode_path):
            if path and path.exists():
                try:
                    with Image.open(path) as image:
                        widths.append(int(image.width))
                except Exception:
                    pass
        return max(widths) if widths else 2400

    @staticmethod
    def _gif_result_signature(result: dict | None, image_path: Path | None) -> list[list[object]]:
        paths: list[Path] = []
        if image_path is not None:
            paths.append(image_path)
        if isinstance(result, dict):
            data_file = _path_from_result_reference(result.get("data_file"))
            if data_file is not None:
                paths.append(data_file)
            waveforms = result.get("waveforms") if isinstance(result.get("waveforms"), list) else []
            for waveform in waveforms:
                if not isinstance(waveform, dict):
                    continue
                waveform_file = _path_from_result_reference(waveform.get("data_file"))
                if waveform_file is not None:
                    paths.append(waveform_file)
        signature: list[list[object]] = []
        seen: set[str] = set()
        for path in paths:
            try:
                resolved = path.resolve()
                key = str(resolved)
                if key in seen or not resolved.exists():
                    continue
                seen.add(key)
                stat = resolved.stat()
                signature.append([key, int(stat.st_size), int(stat.st_mtime_ns)])
            except OSError:
                continue
        return signature

    @staticmethod
    def _quantize_gif_frame(image: "Image.Image") -> "Image.Image":
        from PIL import Image

        try:
            return image.quantize(
                colors=256,
                method=Image.Quantize.FASTOCTREE,
                dither=Image.Dither.NONE,
            )
        except AttributeError:
            return image.convert("P", palette=Image.ADAPTIVE, colors=256)

    @staticmethod
    def _fit_image_no_stretch(image: "Image.Image", width: int, height: int) -> "Image.Image":
        from PIL import Image

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        cropped = AutotuneRunStore._trim_near_white_image(image)
        scale = min(width / max(1, cropped.width), height / max(1, cropped.height))
        scaled = cropped.resize(
            (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale))),
            resampling,
        )
        cropped.close()
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(scaled, ((width - scaled.width) // 2, (height - scaled.height) // 2))
        scaled.close()
        return canvas

    @staticmethod
    def _trim_near_white_image(image: "Image.Image") -> "Image.Image":
        from PIL import Image, ImageChops

        rgb = image.convert("RGB")
        white = Image.new("RGB", rgb.size, "white")
        difference = ImageChops.difference(rgb, white).convert("L")
        content_mask = difference.point(lambda value: 255 if value > 12 else 0)
        bounds = content_mask.getbbox()
        if bounds is not None:
            padding = 8
            left = max(0, bounds[0] - padding)
            top = max(0, bounds[1] - padding)
            right = min(rgb.width, bounds[2] + padding)
            bottom = min(rgb.height, bounds[3] + padding)
            cropped = rgb.crop((left, top, right, bottom))
        else:
            cropped = rgb.copy()
        white.close()
        difference.close()
        content_mask.close()
        if rgb is not image:
            rgb.close()
        return cropped

    @staticmethod
    def _resize_gif_plot_to_width(image: "Image.Image", width: int) -> "Image.Image":
        from PIL import Image

        cropped = AutotuneRunStore._trim_near_white_image(image)
        scale = width / max(1, cropped.width)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = cropped.resize(
            (width, max(1, round(cropped.height * scale))),
            resampling,
        )
        cropped.close()
        return resized

    @staticmethod
    def _resize_gif_plot(image: "Image.Image", scale: float) -> "Image.Image":
        from PIL import Image

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        resized = image.resize(
            (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ),
            resampling,
        )
        image.close()
        return resized

    @staticmethod
    def _render_scope_for_gif(
        scope_result: dict | None,
        fallback_path: Path | None,
        width: int,
        height: int,
        axis_settings: dict | None,
        temp_dir: Path,
        iteration: int,
    ) -> "Image.Image":
        from PIL import Image

        # Auto-Tune already renders an iteration PNG when acquisition finishes.
        # Reuse it for GIF export; raw-data plotting is only a repair fallback.
        if fallback_path and fallback_path.exists() and fallback_path.stat().st_size > 0:
            with Image.open(fallback_path) as image:
                return AutotuneRunStore._fit_image_no_stretch(image.convert("RGB"), width, height)
        entry = _scope_capture_entry_from_result(scope_result) if isinstance(scope_result, dict) else None
        if entry is not None:
            temp_path = temp_dir / f".gif_scope_{uuid.uuid4().hex}.png"
            try:
                _plot_full_scope_capture_png(
                    entry,
                    axis_settings,
                    path=temp_path,
                    title=f"Iteration {iteration} - Scope Capture - Full Data",
                    figsize=(width / 150.0, height / 150.0),
                    dpi=150,
                )
                with Image.open(temp_path) as image:
                    return image.convert("RGB")
            finally:
                temp_path.unlink(missing_ok=True)
        raise RuntimeError("No scope image data is available for GIF rendering.")

    @staticmethod
    def _render_bode_for_gif(
        bode_result: dict | None,
        fallback_path: Path | None,
        width: int,
        height: int,
        temp_dir: Path,
        iteration: int,
    ) -> "Image.Image":
        from PIL import Image

        if fallback_path and fallback_path.exists() and fallback_path.stat().st_size > 0:
            with Image.open(fallback_path) as image:
                return AutotuneRunStore._fit_image_no_stretch(image.convert("RGB"), width, height)
        data_file = _path_from_result_reference(bode_result.get("data_file")) if isinstance(bode_result, dict) else None
        if data_file and data_file.exists():
            temp_path = temp_dir / f".gif_bode_{uuid.uuid4().hex}.png"
            try:
                with np.load(data_file, allow_pickle=True) as payload:
                    _plot_full_bode_sweep_png(
                        frequency_hz=np.asarray(payload["frequency_hz"], dtype=np.float64).tolist(),
                        magnitude_db=np.asarray(payload["magnitude_db"], dtype=np.float64).tolist(),
                        phase_deg=np.asarray(payload["phase_deg"], dtype=np.float64).tolist(),
                        margins=bode_result.get("margins") if isinstance(bode_result.get("margins"), dict) else None,
                        path=temp_path,
                        title=f"Iteration {iteration} - Bode Sweep - Full Data",
                        figsize=(width / 150.0, height / 150.0),
                        dpi=150,
                    )
                with Image.open(temp_path) as image:
                    return image.convert("RGB")
            finally:
                temp_path.unlink(missing_ok=True)
        raise RuntimeError("No Bode image data is available for GIF rendering.")

    @staticmethod
    def _penalty_trend_records(history: list) -> list[dict]:
        trend: list[dict] = []
        for record in history:
            if not isinstance(record, dict):
                continue
            if AutotuneRunStore._record_is_skipped_trip(record):
                normalized_penalty = 300.0
            elif AutotuneRunStore._record_is_invalid_bode(record):
                normalized_penalty = 250.0
            else:
                normalized_penalty = None
            metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
            penalty = metrics.get("score") if isinstance(metrics, dict) else None
            iteration = record.get("iteration")
            try:
                penalty_f = float(normalized_penalty if normalized_penalty is not None else penalty)
                iteration_i = int(iteration)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(penalty_f) or iteration_i <= 0:
                continue
            trend.append({"iteration": iteration_i, "penalty": penalty_f})
        trend.sort(key=lambda item: item["iteration"])
        return trend

    @staticmethod
    def _record_is_skipped_trip(record: dict) -> bool:
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        reasons = metrics.get("pass_reasons") if isinstance(metrics.get("pass_reasons"), list) else []
        reason_text = " ".join(str(item).lower() for item in reasons)
        if "protection skipped" in reason_text or "transient protection skipped" in reason_text:
            return True
        scope_result = record.get("scope_result") if isinstance(record.get("scope_result"), dict) else {}
        scope_error = str(scope_result.get("error") or "").lower()
        if scope_result.get("skipped") or "protection" in scope_error or "trip" in scope_error:
            return True
        try:
            penalty = float(metrics.get("score"))
        except (TypeError, ValueError):
            penalty = None
        return penalty is not None and math.isfinite(penalty) and penalty >= 1.0e6

    @staticmethod
    def _record_is_invalid_bode(record: dict) -> bool:
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        reasons = metrics.get("pass_reasons") if isinstance(metrics.get("pass_reasons"), list) else []
        reason_text = " ".join(str(item).lower() for item in reasons)
        return (
            "invalid bode" in reason_text
            or "duplicate 0 db crossover" in reason_text
            or "second 0 db crossover" in reason_text
        )

    @staticmethod
    def _make_penalty_trend_frame(
        trend_records: list[dict],
        current_record: dict,
        width: int,
        height: int | None = None,
    ) -> "Image.Image":
        from PIL import Image, ImageDraw

        height = height or max(280, min(460, int(width * 0.36)))
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = AutotuneRunStore._gif_font(22)
        small_font = AutotuneRunStore._gif_font(18)
        title_font = AutotuneRunStore._gif_font(28, bold=True)

        left = 92
        right = 36
        top = 62
        bottom = 62
        plot_w = max(10, width - left - right)
        plot_h = max(10, height - top - bottom)

        draw.text((left, 18), "Penalty Trend", fill=(32, 33, 36), font=title_font)
        draw.text((left, height - 34), "Iteration", fill=(95, 99, 104), font=font)
        draw.text((14, top + 4), "Penalty", fill=(95, 99, 104), font=small_font)

        # Axes and light grid.
        axis = (180, 186, 195)
        grid = (230, 234, 240)
        draw.line((left, top, left, top + plot_h), fill=axis, width=2)
        draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill=axis, width=2)
        for i in range(1, 5):
            y = top + round(plot_h * i / 5)
            draw.line((left, y, left + plot_w, y), fill=grid, width=2)
        for i in range(1, 6):
            x = left + round(plot_w * i / 6)
            draw.line((x, top, x, top + plot_h), fill=grid, width=2)

        if not trend_records:
            draw.text((left + 10, top + 20), "No penalty data", fill=(95, 99, 104), font=font)
            return image

        iterations = [item["iteration"] for item in trend_records]
        penalties = [item["penalty"] for item in trend_records]
        min_iter = min(iterations)
        max_iter = max(iterations)
        min_penalty = min(penalties)
        max_penalty = max(penalties)
        if max_iter == min_iter:
            max_iter = min_iter + 1
        if math.isclose(max_penalty, min_penalty):
            pad = max(1.0, abs(max_penalty) * 0.05)
            min_penalty -= pad
            max_penalty += pad
        else:
            pad = (max_penalty - min_penalty) * 0.08
            min_penalty -= pad
            max_penalty += pad

        def point(iteration: int, penalty: float) -> tuple[int, int]:
            x = left + round((iteration - min_iter) / (max_iter - min_iter) * plot_w)
            y = top + round((max_penalty - penalty) / (max_penalty - min_penalty) * plot_h)
            return x, y

        points = [point(item["iteration"], item["penalty"]) for item in trend_records]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(26, 115, 232))
        else:
            draw.line(points, fill=(66, 133, 244), width=5)
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(66, 133, 244))

        current_iteration = current_record.get("iteration")
        current_penalty = None
        metrics = current_record.get("metrics") if isinstance(current_record.get("metrics"), dict) else {}
        if isinstance(metrics, dict):
            current_penalty = metrics.get("score")
        try:
            current_iteration_i = int(current_iteration)
            current_penalty_f = float(current_penalty)
        except (TypeError, ValueError):
            current_iteration_i = None
            current_penalty_f = None
        if current_iteration_i is not None and current_penalty_f is not None and math.isfinite(current_penalty_f):
            cx, cy = point(current_iteration_i, current_penalty_f)
            draw.line((cx, top, cx, top + plot_h), fill=(234, 67, 53), width=4)
            draw.ellipse((cx - 12, cy - 12, cx + 12, cy + 12), fill=(234, 67, 53), outline=(255, 255, 255), width=3)
            label = f"Iteration {current_iteration_i}   Penalty {current_penalty_f:.3f}"
            draw.rectangle((left, height - 58, min(width - right, left + 430), height - 30), fill=(255, 255, 255))
            draw.text((left, height - 58), label, fill=(32, 33, 36), font=font)

        # Endpoint labels after plotting so they remain legible.
        draw.text((left - 76, top - 10), f"{max_penalty:.2f}", fill=(95, 99, 104), font=small_font)
        draw.text((left - 76, top + plot_h - 10), f"{min_penalty:.2f}", fill=(95, 99, 104), font=small_font)
        draw.text((left - 8, top + plot_h + 12), str(min(iterations)), fill=(95, 99, 104), font=small_font)
        max_text = str(max(iterations))
        draw.text((left + plot_w - 12 * len(max_text), top + plot_h + 12), max_text, fill=(95, 99, 104), font=small_font)
        return image

    @staticmethod
    def _gif_font(size: int, bold: bool = False) -> "ImageFont.ImageFont":
        from PIL import ImageFont

        candidates = (
            ["arialbd.ttf", "DejaVuSans-Bold.ttf", "segoeuib.ttf"]
            if bold
            else ["arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"]
        )
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _repair_run_status_artifact_paths(self, run_dir: Path, status: dict) -> bool:
        """Make archived run records point at files inside their own run folder."""
        changed = False
        files_dir = run_dir / "files"
        history = status.get("history") if isinstance(status.get("history"), list) else []
        by_iteration: dict[int, dict] = {}

        for record in history:
            if not isinstance(record, dict):
                continue
            if self._repair_record_artifact_paths(record, files_dir):
                changed = True
            try:
                iteration = int(record.get("iteration") or 0)
            except (TypeError, ValueError):
                iteration = 0
            if iteration > 0:
                by_iteration[iteration] = record

        for key in ("current", "best"):
            record = status.get(key)
            if not isinstance(record, dict):
                continue
            try:
                iteration = int(record.get("iteration") or 0)
            except (TypeError, ValueError):
                iteration = 0
            if iteration in by_iteration:
                if record is not by_iteration[iteration]:
                    status[key] = by_iteration[iteration]
                    changed = True
            elif self._repair_record_artifact_paths(record, files_dir):
                changed = True

        status["run"] = self._run_payload(
            run_dir.name,
            "saved" if run_dir.parent == self.saved_dir else "recent",
            run_dir,
        )
        return changed

    def _repair_record_artifact_paths(self, record: dict, files_dir: Path) -> bool:
        changed = False
        try:
            iteration = int(record.get("iteration") or 0)
        except (TypeError, ValueError):
            iteration = 0
        if iteration <= 0:
            return False

        scope_result = record.get("scope_result")
        if isinstance(scope_result, dict):
            if self._repair_result_file_reference(
                scope_result,
                "scope_png",
                files_dir / f"iteration_{iteration:03d}_scope.png",
                public=True,
            ):
                scope_result["scope_png_pending"] = False
                scope_result["scope_png_error"] = None
                changed = True
            waveforms = scope_result.get("waveforms")
            if isinstance(waveforms, list):
                for waveform in waveforms:
                    if not isinstance(waveform, dict):
                        continue
                    source = str(waveform.get("source") or "CH").upper()
                    compact = files_dir / f"iteration_{iteration:03d}_scope.npz"
                    per_channel = files_dir / f"iteration_{iteration:03d}_scope_{_safe_file_stem(source)}.npz"
                    target = compact if compact.exists() else per_channel
                    if self._repair_result_file_reference(waveform, "data_file", target, public=False):
                        changed = True

        bode_result = record.get("bode_result")
        if isinstance(bode_result, dict):
            if self._repair_result_file_reference(
                bode_result,
                "bode_png",
                files_dir / f"iteration_{iteration:03d}_bode.png",
                public=True,
            ):
                bode_result["bode_png_pending"] = False
                bode_result["bode_png_error"] = None
                changed = True
            if self._repair_result_file_reference(
                bode_result,
                "data_file",
                files_dir / f"iteration_{iteration:03d}_bode.npz",
                public=False,
            ):
                changed = True

        return changed

    @staticmethod
    def _repair_result_file_reference(result: dict, key: str, target: Path, *, public: bool) -> bool:
        if not target.exists():
            return False
        expected = _scope_png_public_path(target) if public else _path_label(target)
        current = result.get(key)
        current_path = _path_from_result_reference(current)
        if current == expected:
            return False
        if current_path is not None:
            try:
                if current_path.resolve() == target.resolve():
                    if current != expected:
                        result[key] = expected
                        return True
                    return False
            except OSError:
                pass
        result[key] = expected
        return True

    @staticmethod
    def _copy_asset_file(source: Path, target: Path) -> bool:
        """Materialize an immutable run artifact without an avoidable byte copy."""

        source = source.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                if os.path.samefile(source, target):
                    return False
            except OSError:
                pass
            target.unlink()
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
        return True

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
            scope_png_ready = False
            source = _path_from_result_reference(scope_result.get("scope_png"))
            if not force_rebuild and source and source.exists() and source.stat().st_size > 0:
                if self._copy_asset_file(source, target):
                    changed = True
                scope_result["scope_png"] = _scope_png_public_path(target)
                scope_result["scope_png_pending"] = False
                scope_result["scope_png_error"] = None
                scope_png_ready = True
            if not scope_png_ready and self._rebuild_scope_png_from_record(
                scope_result,
                target,
                iteration,
                scope_axis_settings,
                record.get("metrics") if isinstance(record.get("metrics"), dict) else None,
            ):
                changed = True
            record["scope_result"] = scope_result

        bode_result = record.get("bode_result")
        if isinstance(bode_result, dict):
            target_png = files_dir / f"iteration_{iteration:03d}_bode.png"
            data_source = _path_from_result_reference(bode_result.get("data_file"))
            target_data = None
            if data_source and data_source.exists():
                target_data = files_dir / f"iteration_{iteration:03d}_bode{data_source.suffix}"
                if self._copy_asset_file(data_source, target_data):
                    changed = True
                bode_result["data_file"] = str(target_data.relative_to(ROOT)).replace("\\", "/")
            bode_png_ready = False
            source = _path_from_result_reference(bode_result.get("bode_png"))
            if not force_rebuild and source and source.exists() and source.stat().st_size > 0:
                if self._copy_asset_file(source, target_png):
                    changed = True
                bode_result["bode_png"] = _scope_png_public_path(target_png)
                bode_result["bode_png_pending"] = False
                bode_result["bode_png_error"] = None
                bode_png_ready = True
            if not bode_png_ready and self._rebuild_bode_png_from_data_file(
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
            record["bode_result"] = bode_result
        return changed

    def _copy_scope_channel_data_files(self, scope_result: dict, files_dir: Path, iteration: int) -> bool:
        changed = False
        waveforms = scope_result.get("waveforms") if isinstance(scope_result.get("waveforms"), list) else []
        copied: dict[Path, Path] = {}
        compact_files: dict[Path, bool] = {}
        for waveform in waveforms:
            if not isinstance(waveform, dict):
                continue
            source_label = str(waveform.get("source") or "CH").upper()
            data_source = _path_from_result_reference(waveform.get("data_file"))
            if not data_source or not data_source.exists():
                continue
            data_source = data_source.resolve()
            if data_source in copied:
                waveform["data_file"] = str(copied[data_source].relative_to(ROOT)).replace("\\", "/")
                continue
            if data_source not in compact_files:
                try:
                    with np.load(data_source, allow_pickle=False) as payload:
                        compact_files[data_source] = (
                            "format_version" in payload.files
                            and int(np.asarray(payload["format_version"]).item()) >= 2
                        )
                except Exception:
                    compact_files[data_source] = False
            is_compact = compact_files[data_source]
            target_name = f"iteration_{iteration:03d}_scope{data_source.suffix}" if is_compact else f"iteration_{iteration:03d}_scope_{_safe_file_stem(source_label)}{data_source.suffix}"
            target = files_dir / target_name
            if self._copy_asset_file(data_source, target):
                changed = True
            copied[data_source] = target
            waveform["data_file"] = str(target.relative_to(ROOT)).replace("\\", "/")
        return changed

    def _rebuild_scope_png_from_record(
        self,
        scope_result: dict,
        target: Path,
        iteration: int,
        axis_settings: dict | None,
        metrics_payload: dict | None = None,
    ) -> bool:
        entry = _scope_capture_entry_from_result(scope_result)
        if entry is None:
            return False
        if isinstance(metrics_payload, dict):
            try:
                entry["settling_metrics"] = ResponseMetrics(**metrics_payload)
            except (TypeError, ValueError):
                # Legacy records may not contain every current metrics field;
                # those can still be rebuilt by analyzing the saved waveform.
                pass
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
                title=f"Iteration {iteration} - Bode Sweep - Full Data",
            )
        return True

    def _write_summary(self, run_dir: Path, status: dict) -> None:
        history = status.get("history") if isinstance(status.get("history"), list) else []
        best = status.get("best") if isinstance(status.get("best"), dict) else None
        current = status.get("current") if isinstance(status.get("current"), dict) else None
        summary = {
            "run_id": run_dir.name,
            "display_name": self._display_name_for_run(run_dir.name, "recent" if run_dir.parent == self.recent_dir else "saved"),
            "algorithm": self._algorithm_label_for_status(status),
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
        """List lightweight run descriptors without loading full histories.

        ``run_status.json`` can contain thousands of iterations and tens of
        megabytes of diagnostics. Reading every status just to populate the
        Result Library made a page refresh scale with the total archive size.
        ``summary.json`` is the index for this endpoint; the complete status is
        loaded only by ``load_run`` after the user presses Load.
        """

        if not folder.exists():
            return []
        runs = []
        for run_dir in folder.iterdir():
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "summary.json"
            summary = self._read_json(summary_path)
            if not isinstance(summary, dict):
                # One-time compatibility path for a legacy run without an
                # index. Future listings use the summary written here.
                status = self._read_json(run_dir / "run_status.json")
                if isinstance(status, dict):
                    self._write_summary(run_dir, status)
                    summary = self._read_json(summary_path)
            if not isinstance(summary, dict):
                summary = {"run_id": run_dir.name}
            else:
                summary = dict(summary)
            summary["run_id"] = run_dir.name
            summary["display_name"] = self._display_name_for_run(run_dir.name, kind)
            summary["kind"] = kind
            summary.setdefault("algorithm", "Grid")
            runs.append(summary)
        runs.sort(key=lambda item: float(item.get("updated_at") or item.get("created_at") or 0), reverse=True)
        return runs

    def _enforce_recent_limit(self) -> None:
        runs = self._list_kind(self.recent_dir, "recent")
        for item in runs:
            run_id = str(item.get("run_id", ""))
            if not run_id or run_id == self._current_run_id:
                continue
            if self._is_terminal_short_run(item):
                shutil.rmtree(self.recent_dir / run_id, ignore_errors=True)
        runs = self._list_kind(self.recent_dir, "recent")
        for item in runs[self.recent_limit :]:
            run_id = str(item.get("run_id", ""))
            if run_id and run_id != self._current_run_id:
                shutil.rmtree(self.recent_dir / run_id, ignore_errors=True)

    def _discard_terminal_short_recent_run(self, run_dir: Path, run_kind: str, status: dict) -> bool:
        """Discard completed/error short trial runs instead of filling Recent."""

        if run_kind != "recent" or run_dir.name != self._current_run_id:
            return False
        if not self._is_terminal_short_run(status):
            return False
        shutil.rmtree(run_dir, ignore_errors=True)
        self.stop_current()
        return True

    def _is_terminal_short_run(self, payload: dict) -> bool:
        state = str(payload.get("state") or "").lower()
        if state not in {"complete", "error", "stopped"}:
            return False
        history = payload.get("history")
        iteration_count = len(history) if isinstance(history, list) else int(payload.get("iteration_count") or 0)
        return iteration_count < self.recent_min_iterations

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
            "display_name": self._display_name_for_run(run_id, kind),
            "kind": kind,
            "path": _path_label(run_dir),
            "recent_limit": self.recent_limit,
            "recent_min_iterations": self.recent_min_iterations,
        }

    def _next_friendly_run_dir(self, kind: str, requested_name: str | None = None) -> Path:
        folder = self.saved_dir if kind == "saved" else self.recent_dir
        folder.mkdir(parents=True, exist_ok=True)
        if requested_name:
            base = _safe_file_stem(requested_name)
            if base:
                candidate = folder / base
                if not candidate.exists():
                    return candidate
                index = 2
                while True:
                    candidate = folder / f"{base}_{index:02d}"
                    if not candidate.exists():
                        return candidate
                    index += 1

        prefix = "Permanent" if kind == "saved" else "Recent"
        date = time.strftime("%Y-%m-%d")
        next_index = self._next_friendly_index(folder, prefix, date)
        while True:
            candidate = folder / f"{prefix}_{date}_{next_index:02d}"
            if not candidate.exists():
                return candidate
            next_index += 1

    @staticmethod
    def _next_friendly_index(folder: Path, prefix: str, date: str) -> int:
        pattern = re.compile(rf"^{re.escape(prefix)}_{re.escape(date)}_(\d+)(?:_.*)?$")
        highest = 0
        if folder.exists():
            for child in folder.iterdir():
                if not child.is_dir():
                    continue
                match = pattern.match(child.name)
                if match:
                    highest = max(highest, int(match.group(1)))
        return highest + 1

    @staticmethod
    def _display_name_for_run(run_id: str, kind: str) -> str:
        text = str(run_id)
        prefix = "Permanent" if kind == "saved" else "Recent"
        friendly = re.match(r"^(Recent|Permanent)_(\d{4}-\d{2}-\d{2})_(\d+)", text)
        if friendly:
            return f"{friendly.group(1)} / {friendly.group(2)} #{int(friendly.group(3))}"
        legacy = re.match(r"^(\d{4})(\d{2})(\d{2})", text)
        if legacy:
            return f"{prefix} / {legacy.group(1)}-{legacy.group(2)}-{legacy.group(3)}"
        return f"{prefix} / {text or 'Unknown'}"

    @staticmethod
    def _algorithm_label_for_status(status: dict) -> str:
        experiment = status.get("experiment") if isinstance(status.get("experiment"), dict) else {}
        algorithm = str(experiment.get("optimization_algorithm") or "").strip().lower()
        if any(token in algorithm for token in ("drl", "reinforcement", "safe-sac", "sac")):
            return "DRL"

        # Older summaries may predate the experiment algorithm field. Their
        # iteration phases still make DRL runs unambiguous.
        history = status.get("history") if isinstance(status.get("history"), list) else []
        for record in history:
            phase = str(record.get("phase") or "").strip().lower() if isinstance(record, dict) else ""
            if phase.startswith("drl_") or phase.startswith("drl-"):
                return "DRL"
        return "Grid"

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None


DRL_WORKFLOW = DrlWorkflowManager(
    RESULTS_DIR / "autotune_ml",
    [AUTOTUNE_SAVED_DIR, AUTOTUNE_RECENT_DIR],
)
TUNING_SESSION = PidAutotuneSession(
    experiment_runner=ServerHardwareExperimentRunner(),
    tuner_factory=DRL_WORKFLOW.candidate_tuner_factory,
)
AUTOTUNE_RUN_STORE = AutotuneRunStore(
    AUTOTUNE_RECENT_DIR,
    AUTOTUNE_SAVED_DIR,
    AUTOTUNE_RECENT_LIMIT,
    AUTOTUNE_RECENT_MIN_ITERATIONS,
)
TUNING_SESSION.set_iteration_callback(AUTOTUNE_RUN_STORE.persist_iteration_record)


def _compact_loaded_status_for_client(status: dict) -> dict:
    """Drop per-iteration diagnostics from an archived status response.

    The run_status.json file remains authoritative and untouched, so Resume,
    relabeling, and artifact repair still have the original diagnostics.  The
    browser only needs candidate, metrics, and artifact references for old
    history rows.
    """

    history = status.get("history")
    if isinstance(history, list):
        compact_history: list[dict] = []
        for record in history:
            if not isinstance(record, dict):
                continue
            compact = dict(record)
            compact["waveform"] = {"time_s": [], "vout_v": [], "input_v": []}
            compact["write_results"] = {}
            compact["optimizer_metadata"] = {}
            compact["bode_result"] = _compact_bode_result(compact.get("bode_result") or {})
            compact["scope_result"] = _compact_scope_result(compact.get("scope_result") or {})
            compact_history.append(compact)
        status["history"] = compact_history
    recommendations = status.get("recommendations")
    if isinstance(recommendations, list):
        status["recommendations"] = [
            {
                **record,
                "waveform": {"time_s": [], "vout_v": [], "input_v": []},
                "write_results": {},
                "optimizer_metadata": {},
                "bode_result": _compact_bode_result(record.get("bode_result") or {}),
                "scope_result": _compact_scope_result(record.get("scope_result") or {}),
            }
            for record in recommendations
            if isinstance(record, dict)
        ]
    return status


def _requested_history_cursor(query: str, history_token: str) -> int | None:
    params = parse_qs(query)
    requested_token = str(params.get("history_token", [""])[0] or "")
    raw_after = params.get("after_iteration", [None])[0]
    if requested_token != history_token or raw_after in (None, ""):
        return None
    try:
        return max(0, int(str(raw_after)))
    except (TypeError, ValueError):
        return None


def _incremental_status_for_client(
    status: dict,
    query: str,
    history_token: str,
    *,
    history_is_filtered: bool = False,
) -> dict:
    """Return either a complete status or history records newer than the cursor."""

    history = status.get("history") if isinstance(status.get("history"), list) else []
    after_iteration = _requested_history_cursor(query, history_token)

    numbered_history: list[tuple[dict, int]] = []
    for index, record in enumerate(history):
        if not isinstance(record, dict):
            continue
        try:
            iteration = int(record.get("iteration") or index + 1)
        except (TypeError, ValueError):
            iteration = index + 1
        numbered_history.append((record, iteration))
    history_total = int(status.get("history_total", len(history)) or 0)
    last_iteration = int(
        status.get(
            "history_last_iteration",
            max((iteration for _, iteration in numbered_history), default=0),
        )
        or 0
    )
    delta_allowed = bool(
        after_iteration is not None
        and after_iteration <= last_iteration
    )
    payload = dict(status)
    if delta_allowed and not history_is_filtered:
        payload["history"] = [
            record for record, iteration in numbered_history if iteration > int(after_iteration)
        ]
    payload["history_delta"] = delta_allowed
    payload["history_token"] = history_token
    payload["history_after_iteration"] = int(after_iteration or 0) if delta_allowed else None
    payload["history_total"] = history_total
    payload["history_last_iteration"] = last_iteration
    return payload


def _start_drl_hardware(config: TuningConfig, experiment: AutotuneExperimentConfig) -> dict:
    configured = TUNING_SESSION.configure(config, experiment)
    AUTOTUNE_RUN_STORE.start_new(configured)
    return AUTOTUNE_RUN_STORE.persist_status(TUNING_SESSION.start())


DRL_WORKFLOW.bind_session(
    status=TUNING_SESSION.status,
    stop=TUNING_SESSION.stop,
    start_hardware=_start_drl_hardware,
    resume_hardware=AUTOTUNE_RUN_STORE.resume_run,
    persist_hardware=AUTOTUNE_RUN_STORE.persist_status,
)


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


def _coerce_llm_messages(messages: object) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    clean: list[dict[str, str]] = []
    for item in messages[-16:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant", "system"} or not content:
            continue
        clean.append({"role": role, "content": content[:6000]})
    if not any(message["role"] == "user" for message in clean):
        raise ValueError("at least one user message is required")
    return clean


def _llm_chat_endpoint(base_url: str, explicit_endpoint: str = "") -> str:
    explicit = explicit_endpoint.strip()
    if explicit:
        return explicit
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return f"{endpoint}/chat/completions"


LLM_MODEL_CHOICES = {
    "gemini-3.5-flash": {
        "label": "Gemini 3.5 Flash",
        "api_key_env": ("OPENROUTER_API_KEY",),
        "base_url_env": ("OPENROUTER_BASE_URL",),
        "model_env": ("OPENROUTER_MODEL",),
        "endpoint_env": ("OPENROUTER_CHAT_ENDPOINT",),
        "default_base_url": "https://openrouter.ai/api/v1",
        "default_model": "google/gemini-3.5-flash",
    },
    "minimax-m3": {
        "label": "Minimax 3",
        "api_key_env": ("MINIMAX_API_KEY",),
        "base_url_env": ("MINIMAX_BASE_URL",),
        "model_env": ("MINIMAX_MODEL",),
        "endpoint_env": ("MINIMAX_CHAT_ENDPOINT",),
        "default_base_url": "https://api.minimax.io/v1",
        "default_model": "MiniMax-M3",
    },
}


def _first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _resolve_llm_choice(model_choice: object) -> dict[str, object]:
    choice_key = str(model_choice or "gemini-3.5-flash").strip().lower()
    if choice_key in {"minimax", "minimax-3", "minimax3", "minimax-m3"}:
        choice_key = "minimax-m3"
    if choice_key in {"gemini", "gemini-3.5", "gemini-3.5-flash", "openrouter-gemini"}:
        choice_key = "gemini-3.5-flash"
    return dict(LLM_MODEL_CHOICES.get(choice_key, LLM_MODEL_CHOICES["gemini-3.5-flash"]))


def _sanitize_llm_reply(reply: str) -> str:
    return re.sub(r"(?is)<think>.*?</think>\s*", "", reply).strip()


def _truthy_env(name: str, default: str = "false") -> bool:
    return (os.environ.get(name, default) or "").strip().lower() in {"1", "true", "yes", "on"}


def _latest_user_message(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"]
    return ""


def _should_web_search(query: str) -> bool:
    if not _truthy_env("WEB_SEARCH_ENABLED", "true"):
        return False
    lowered = query.lower()
    triggers = [
        "latest",
        "newest",
        "today",
        "current",
        "recent",
        "news",
        "web",
        "internet",
        "search",
        "lookup",
        "google",
    ]
    return any(trigger in lowered for trigger in triggers)


def _compact_search_query(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:200]


def _search_with_tavily(query: str, max_results: int, timeout_s: float) -> list[dict[str, str]]:
    api_key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if not api_key:
        return []
    payload = json.dumps(
        {
            "api_key": api_key,
            "query": query,
            "search_depth": os.environ.get("TAVILY_SEARCH_DEPTH", "basic"),
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }
    ).encode("utf-8")
    request = urllib_request.Request(
        "https://api.tavily.com/search",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib_request.urlopen(request, timeout=timeout_s) as response:
        data = json.loads(response.read().decode("utf-8"))
    results = data.get("results") if isinstance(data, dict) else []
    clean: list[dict[str, str]] = []
    for item in results or []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        content = str(item.get("content", "")).strip()
        if title and url:
            clean.append({"title": title[:160], "url": url[:500], "snippet": content[:450]})
    return clean


def _search_with_duckduckgo(query: str, max_results: int, timeout_s: float) -> list[dict[str, str]]:
    url = (
        "https://api.duckduckgo.com/"
        f"?q={quote_plus(query)}&format=json&no_redirect=1&no_html=1&skip_disambig=1"
    )
    request = urllib_request.Request(url, headers={"User-Agent": "PidAutotunerGui/0.2"})
    with urllib_request.urlopen(request, timeout=timeout_s) as response:
        data = json.loads(response.read().decode("utf-8"))
    clean: list[dict[str, str]] = []

    abstract = str(data.get("AbstractText", "") if isinstance(data, dict) else "").strip()
    abstract_url = str(data.get("AbstractURL", "") if isinstance(data, dict) else "").strip()
    heading = str(data.get("Heading", "") if isinstance(data, dict) else "").strip()
    if abstract and abstract_url:
        clean.append({"title": heading or "DuckDuckGo result", "url": abstract_url, "snippet": abstract[:450]})

    def add_related(items: object) -> None:
        if len(clean) >= max_results or not isinstance(items, list):
            return
        for item in items:
            if len(clean) >= max_results:
                return
            if not isinstance(item, dict):
                continue
            nested = item.get("Topics")
            if nested:
                add_related(nested)
                continue
            title = str(item.get("Text", "")).strip()
            first_url = str(item.get("FirstURL", "")).strip()
            if title and first_url:
                clean.append({"title": title[:160], "url": first_url[:500], "snippet": title[:450]})

    if isinstance(data, dict):
        add_related(data.get("RelatedTopics"))
    return clean[:max_results]


def _web_search_context(query: str) -> str:
    query = _compact_search_query(query)
    if not query:
        return ""
    max_results = max(1, min(5, int(os.environ.get("WEB_SEARCH_MAX_RESULTS", "4") or "4")))
    timeout_s = max(2.0, min(15.0, float(os.environ.get("WEB_SEARCH_TIMEOUT_S", "6") or "6")))
    results: list[dict[str, str]] = []
    errors: list[str] = []

    for provider in ("tavily", "duckduckgo"):
        try:
            results = _search_with_tavily(query, max_results, timeout_s) if provider == "tavily" else _search_with_duckduckgo(query, max_results, timeout_s)
            if results:
                break
        except Exception as exc:
            errors.append(f"{provider}: {exc}")

    if not results:
        if errors:
            return "Web search was attempted but returned no usable results. Search errors: " + "; ".join(errors)[:700]
        return ""

    lines = [f"Web search results for: {query}"]
    for index, item in enumerate(results, start=1):
        lines.append(
            f"{index}. {item['title']}\n"
            f"   URL: {item['url']}\n"
            f"   Snippet: {item['snippet']}"
        )
    lines.append("Use these search results only when relevant, cite URLs in the answer, and say when information comes from search results.")
    return "\n".join(lines)[:3500]


LLM_COMPLETION_MARKER = "[[AI_COPILOT_RESPONSE_COMPLETE]]"


def _call_llm_chat(messages: object, context: object, model_choice: object = None) -> tuple[str, str, dict[str, object]]:
    choice = _resolve_llm_choice(model_choice)
    api_key = _first_env(choice["api_key_env"])  # type: ignore[arg-type]
    if not api_key:
        env_names = ", ".join(choice["api_key_env"])  # type: ignore[arg-type]
        raise RuntimeError(f"{env_names} is not set for {choice['label']}. Add it to .env and restart the GUI server.")

    base_url = _first_env(choice["base_url_env"]) or str(choice["default_base_url"])  # type: ignore[arg-type]
    model = _first_env(choice["model_env"]) or str(choice["default_model"])  # type: ignore[arg-type]
    explicit_endpoint = _first_env(choice["endpoint_env"])  # type: ignore[arg-type]
    timeout_s = float(os.environ.get("LLM_TIMEOUT_S", "120") or "120")
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.2") or "0.2")
    max_tokens = max(256, min(100000, int(os.environ.get("LLM_MAX_TOKENS", "10000") or "10000")))
    max_continuations = max(1, min(8, int(os.environ.get("LLM_MAX_CONTINUATIONS", "4") or "4")))

    clean_messages = _coerce_llm_messages(messages)
    user_query = _latest_user_message(clean_messages)
    search_context = _web_search_context(user_query) if _should_web_search(user_query) else ""
    gui_context = json.dumps(context if isinstance(context, dict) else {}, ensure_ascii=False, default=str)[:4000]
    system_prompt = (
        "You are a concise assistant embedded inside the Google Power Auto-Tuner GUI. "
        "Help users understand this hardware-in-the-loop PID tuning interface: PID Auto-Tune, "
        "Manual Tuning, Self Testing, XDP/PMBus board control, Bode 100, oscilloscope, "
        "function generator, and power supply panels. Respond in the same language as the user. "
        "Do not claim that you can directly control hardware; tell the user which GUI control "
        "or workflow to use. Be safety-aware and practical. "
        f"A response is complete only when it ends with the exact marker {LLM_COMPLETION_MARKER}. "
        "Always put that marker once at the very end, after the complete answer; never stop before it.\n\n"
        f"Current GUI context JSON:\n{gui_context}"
    )
    if search_context:
        system_prompt += f"\n\nExternal web search context:\n{search_context}"
    outgoing = [{"role": "system", "content": system_prompt}, *clean_messages]
    reply_parts: list[str] = []
    returned_model = model
    final_finish_reason = ""
    continuation_count = 0
    for continuation_index in range(max_continuations + 1):
        body = json.dumps(
            {
                "model": model,
                "messages": outgoing,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        request = urllib_request.Request(
            _llm_chat_endpoint(base_url, explicit_endpoint),
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib_request.urlopen(request, timeout=timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:700]
            raise RuntimeError(f"LLM API error {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Could not reach LLM API: {exc.reason}") from exc

        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not choices:
            raise RuntimeError("LLM API returned no choices.")
        choice_result = choices[0] if isinstance(choices[0], dict) else {}
        message = choice_result.get("message", {}) if isinstance(choice_result, dict) else {}
        raw_reply = str(message.get("content", "")).strip()
        marker_present = LLM_COMPLETION_MARKER in raw_reply
        visible_reply = raw_reply.split(LLM_COMPLETION_MARKER, 1)[0] if marker_present else raw_reply
        clean_reply = _sanitize_llm_reply(visible_reply)
        if clean_reply:
            reply_parts.append(clean_reply)
        returned_model = str(payload.get("model", returned_model))

        finish_reason = str(choice_result.get("finish_reason", "")).strip().lower()
        final_finish_reason = finish_reason
        if marker_present:
            break
        if continuation_index >= max_continuations:
            raise RuntimeError(
                "LLM provider repeatedly returned an incomplete response without the completion marker "
                f"(last finish_reason={finish_reason or 'missing'}). Try another model or ask again."
            )
        if not raw_reply:
            raise RuntimeError("LLM used its output budget without returning visible text.")
        continuation_count += 1
        outgoing.extend(
            [
                {"role": "assistant", "content": raw_reply},
                {
                    "role": "user",
                    "content": (
                        "Your previous response was incomplete because it did not include the required "
                        f"ending marker {LLM_COMPLETION_MARKER}. Continue exactly where it stopped, "
                        "do not repeat earlier text, finish every unfinished sentence/list, and put the "
                        "marker once at the very end."
                    ),
                },
            ]
        )

    reply = "\n".join(part.strip() for part in reply_parts if part.strip()).strip()
    if not reply:
        raise RuntimeError("LLM API returned an empty reply.")
    return reply, returned_model, {
        "complete": True,
        "completion_protocol": "sentinel-v1",
        "continuations": continuation_count,
        "finish_reason": final_finish_reason,
        "max_tokens": max_tokens,
    }


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
            history_token = AUTOTUNE_RUN_STORE.history_token()
            requested_cursor = _requested_history_cursor(parsed.query, history_token)
            status = TUNING_SESSION.status(after_iteration=requested_cursor)
            is_running = status.get("state") == "running"
            # A matching token should never move backwards, but fail closed to
            # a complete snapshot for malformed/stale cursors.
            if (
                requested_cursor is not None
                and requested_cursor > int(status.get("history_last_iteration", 0) or 0)
            ):
                requested_cursor = None
                status = TUNING_SESSION.status()
                is_running = status.get("state") == "running"
            # Persistence and artifact reconciliation require the complete
            # history. During an active run, keep the cursor-filtered payload.
            if not is_running and requested_cursor is not None:
                status = TUNING_SESSION.status()
            artifacts_changed = _refresh_status_artifact_readiness(status, include_history=not is_running)
            if not is_running and (artifacts_changed or AUTOTUNE_RUN_STORE.needs_persist(status)):
                status = AUTOTUNE_RUN_STORE.persist_status(status)
            status = _incremental_status_for_client(
                status,
                parsed.query,
                history_token,
                history_is_filtered=is_running and requested_cursor is not None,
            )
            self._send_json({"ok": True, **status})
            return
        if parsed.path == "/api/tuning/config":
            self._send_json({"ok": True, "config": TUNING_SESSION.status()["config"]})
            return
        if parsed.path == "/api/tuning/drl/status":
            self._send_json(DRL_WORKFLOW.status())
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
        if parsed.path == "/api/tuning/drl/collect":
            self._handle_drl_action("collect")
            return
        if parsed.path == "/api/tuning/drl/train":
            self._handle_drl_action("train")
            return
        if parsed.path == "/api/tuning/drl/validate":
            self._handle_drl_action("validate")
            return
        if parsed.path == "/api/tuning/drl/targeted":
            self._handle_drl_action("targeted")
            return
        if parsed.path == "/api/tuning/drl/targeted-recovery":
            self._handle_drl_action("targeted-recovery")
            return
        if parsed.path == "/api/tuning/drl/stop":
            self._handle_drl_action("stop")
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
            status = TUNING_SESSION.pause()
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
        if parsed.path == "/api/tuning/gif/open":
            self._handle_tuning_gif(open_after_save=True)
            return
        if parsed.path == "/api/llm/chat":
            self._handle_llm_chat()
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
            DRL_WORKFLOW.assert_tuning_available()
            payload = self._read_json_body()
            config = _config_from_payload(payload.get("config", payload))
            experiment = _experiment_from_payload(payload.get("experiment", {}))
            configured = TUNING_SESSION.configure(config, experiment)
            AUTOTUNE_RUN_STORE.start_new(configured)
            status = AUTOTUNE_RUN_STORE.persist_status(TUNING_SESSION.start())
            self._send_json({"ok": True, **status})
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_drl_action(self, action: str) -> None:
        try:
            payload = self._read_json_body()
            if action == "stop":
                result = DRL_WORKFLOW.stop()
            else:
                config = _config_from_payload(payload.get("config", payload))
                experiment = _experiment_from_payload(payload.get("experiment", {}))
                if action == "collect":
                    result = DRL_WORKFLOW.start_collection(config, experiment)
                elif action == "train":
                    result = DRL_WORKFLOW.start_training(config, experiment)
                elif action == "validate":
                    result = DRL_WORKFLOW.start_validation(config, experiment)
                elif action == "targeted":
                    source_run_ids = payload.get("source_run_ids")
                    if not isinstance(source_run_ids, list):
                        raise RuntimeError("Targeted collection requires a source_run_ids list.")
                    result = DRL_WORKFLOW.start_targeted_collection(
                        config,
                        experiment,
                        [str(value) for value in source_run_ids],
                    )
                elif action == "targeted-recovery":
                    source_plan_id = str(payload.get("source_plan_id") or "")
                    source_plan_indexes = payload.get("source_plan_indexes")
                    if not isinstance(source_plan_indexes, list):
                        raise RuntimeError("Targeted recovery requires a source_plan_indexes list.")
                    result = DRL_WORKFLOW.start_targeted_recovery(
                        config,
                        experiment,
                        source_plan_id,
                        [int(value) for value in source_plan_indexes],
                    )
                else:
                    raise RuntimeError(f"Unsupported DRL action: {action}")
            self._send_json(result)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _handle_tuning_step(self) -> None:
        try:
            DRL_WORKFLOW.assert_tuning_available()
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
            DRL_WORKFLOW.assert_tuning_available()
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return
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

    def _handle_tuning_gif(self, open_after_save: bool = False) -> None:
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
            if open_after_save:
                self._send_json(AUTOTUNE_RUN_STORE.open_animation_gif(None if run_id in (None, "") else str(run_id), kind, duration_ms))
            else:
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
            source_vpp = _optional_bode_source_vpp(payload, default=0.1)
            source_dbm = _optional_bode_source_dbm(payload)
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
            source_vpp=source_vpp,
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

    def _handle_llm_chat(self) -> None:
        try:
            payload = self._read_json_body()
            messages = payload.get("messages", [])
            context = payload.get("context", {})
            model_choice = payload.get("model_choice") or payload.get("model")
            reply, model, completion = _call_llm_chat(messages, context, model_choice)
            self._send_json({"ok": True, "reply": reply, "model": model, "completion": completion})
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
                "current_mode_registers": board.read_mod0_current_mode_registers(page),
                "ll_bandwidth": board.read_mod0_ll_bandwidth(page),
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
            supported_pid_fields = {"mod0_kp", "mod0_ki", "mod0_kd", "mod0_kpole1", "mod0_kpole2"}
            supported_current_mode_fields = {"mod0_cm_gain"}
            unknown = sorted(set(int_values) - supported_pid_fields - supported_current_mode_fields)
            if unknown:
                raise ValueError(f"Unsupported XDP field(s): {', '.join(unknown)}")
            pid_values = {name: value for name, value in int_values.items() if name in supported_pid_fields}
            current_mode_values = {name: value for name, value in int_values.items() if name == "mod0_cm_gain"}
            if "mod0_cm_gain" in current_mode_values and not 0 <= current_mode_values["mod0_cm_gain"] <= 9:
                raise ValueError("mod0_cm_gain must be an integer from 0 through 9.")
            write = board.set_mod0_pid_registers(pid_values, page) if pid_values else None
            current_mode_write = (
                board.set_mod0_current_mode_registers(current_mode_values, page)
                if current_mode_values
                else None
            )
            pid_readback = write["readback"] if write else board.read_mod0_pid_registers(page)
            current_mode_readback = (
                current_mode_write["readback"]
                if current_mode_write
                else board.read_mod0_current_mode_registers(page)
            )
            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "write": write,
                "current_mode_write": current_mode_write,
                "pid_registers": pid_readback,
                "current_mode_registers": current_mode_readback,
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page, "requested": values}
        finally:
            if board is not None:
                board.close()


def _set_mod0_ll_bandwidth(*, address: str, page: int, adapter_kind: str, value: int) -> dict:
    with DEVICE_LOCK:
        board = None
        try:
            if not 47 <= int(value) <= 127:
                raise ValueError("mod0_ll_bw search value must be an integer from 47 through 127.")
            board = _connect_board(address, adapter_kind)
            write = board.set_mod0_ll_bandwidth(int(value), page)
            return {
                "ok": True,
                "address": f"0x{board.device.address:02X}",
                "page": page,
                "loop": _loop_name(page),
                "value": int(value),
                "write": write,
                "timestamp": time.time(),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "address": address, "page": page}
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
    source_vpp: float | None = None,
    source_dbm: float | None = None,
    timeout_ms: int,
    async_artifacts: bool = False,
    reuse_session: bool = False,
    iteration_number: int | None = None,
    _retry_stale_session: bool = True,
) -> dict:
    started = time.perf_counter()
    stage_started = started
    stage_durations: dict[str, float] = {}

    def mark_stage(name: str) -> None:
        nonlocal stage_started
        now = time.perf_counter()
        stage_durations[name] = round(now - stage_started, 3)
        stage_started = now
    if start_hz <= 0 or stop_hz <= start_hz:
        return {"ok": False, "error": "Bode sweep requires 0 < start_hz < stop_hz."}
    if not 2 <= points <= 2001:
        return {"ok": False, "error": "Bode sweep points must be between 2 and 2001."}
    if source_vpp is not None and source_vpp <= 0:
        return {"ok": False, "error": "Bode source level must be greater than 0 Vpp."}
    effective_source_dbm = _vpp_to_dbm(source_vpp) if source_vpp is not None else source_dbm

    bode_driver = Bode100Driver(host=host, port=port, startup_timeout_s=max(timeout_ms / 1000.0, 5.0))
    cached_session_available = reuse_session and _has_bode_connection(host, port)
    if not cached_session_available:
        try:
            bode_driver.ensure_scpi_server()
            mark_stage("ensure_scpi_server")
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
                    "source_vpp": source_vpp,
                    "source_dbm": effective_source_dbm,
                },
                "duration_s": round(time.perf_counter() - started, 3),
                "timestamp": time.time(),
            }
    else:
        mark_stage("ensure_scpi_server")

    bode = None
    close_when_done = not reuse_session
    session_reused = False
    retried_stale_session = not _retry_stale_session
    try:
        if reuse_session:
            bode, identity, session_reused = _get_bode_connection(
                bode_driver.resource_name,
                host=host,
                port=port,
                timeout_ms=timeout_ms,
                runner=bode_driver,
            )
        else:
            bode = BodeScpiClient(resource_name=bode_driver.resource_name, timeout_ms=timeout_ms)
            bode.connect()
            identity = bode.idn()
            try:
                bode.lock()
            except Exception:
                pass
        mark_stage("session")
        config_signature = (
            float(start_hz),
            float(stop_hz),
            int(points),
            float(bandwidth_hz),
            None if effective_source_dbm is None else float(effective_source_dbm),
        )
        config_reused = bool(
            reuse_session and getattr(bode, "_autotune_config_signature", None) == config_signature
        )
        # Re-define the gain/phase measurement before every hardware sweep.
        # Bode Analyzer Suite can keep a stale trace/format state on a reused
        # SCPI session even when the requested numeric config is unchanged;
        # that produced low-frequency phase noise and many false 0 dB
        # crossovers in long runs. The configuration commands are cheap
        # compared with acquisition and make each iteration self-contained.
        bode.configure_gain_phase(
            start_hz=start_hz,
            stop_hz=stop_hz,
            points=points,
            bandwidth_hz=bandwidth_hz,
            source_dbm=effective_source_dbm,
        )
        bode._autotune_config_signature = config_signature
        mark_stage("configure")
        data = bode.run_sweep()
        mark_stage("acquire_transfer")
        margins = data.stability_margins.as_dict()
        timestamp = time.time()
        sweep_id = uuid.uuid4().hex[:12]
        BODE_SWEEP_DIR.mkdir(parents=True, exist_ok=True)
        data_file_path = BODE_SWEEP_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{sweep_id}.npz"
        data_file_pending = False
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
                "source_vpp": np.nan if source_vpp is None else float(source_vpp),
                "source_dbm": np.nan if effective_source_dbm is None else float(effective_source_dbm),
                "identity": str(identity),
                "timestamp": float(timestamp),
            },
        )
        mark_stage("data_artifact")
        bode_png = None
        bode_png_error = None
        bode_png_pending = False
        try:
            bode_png_path = RESULTS_DIR / "bode_sweeps" / f"{time.strftime('%Y%m%d_%H%M%S')}_{sweep_id}.png"
            bode_title = (
                f"Iteration {iteration_number} - Bode Sweep - Full Data"
                if iteration_number is not None
                else "Latest Bode Sweep - Full Data"
            )
            if async_artifacts:
                bode_png = _scope_png_public_path(bode_png_path)
                bode_png_pending = _schedule_bode_png_artifact(
                    frequency_hz=data.frequency_hz,
                    magnitude_db=data.magnitude_db,
                    phase_deg=data.phase_deg,
                    margins=margins,
                    path=bode_png_path,
                    title=bode_title,
                )
                if not bode_png_pending:
                    bode_png = None
                    bode_png_error = "PNG rendering deferred because the artifact queue is busy."
            else:
                bode_png = _plot_full_bode_sweep_png(
                    frequency_hz=data.frequency_hz,
                    magnitude_db=data.magnitude_db,
                    phase_deg=data.phase_deg,
                    margins=margins,
                    path=bode_png_path,
                    title=bode_title,
                )
                LATEST_BODE_PNG.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bode_png_path, LATEST_BODE_PNG)
        except Exception as exc:
            bode_png_error = str(exc)
            bode_png_pending = False
        mark_stage("png_artifact")
        system_error = ""
        try:
            system_error = bode.get_error()
        except Exception:
            pass
        mark_stage("error_query")
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
                "source_vpp": source_vpp,
                "source_dbm": effective_source_dbm,
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
            "session_reused": session_reused,
            "config_reused": config_reused,
            "retried_stale_session": retried_stale_session,
            "stage_durations_s": stage_durations,
            "system_error": system_error,
            "duration_s": round(time.perf_counter() - started, 3),
            "timestamp": timestamp,
        }
    except Exception as exc:
        if reuse_session:
            _drop_bode_connection(host, port)
            if _retry_stale_session and _is_stale_bode_session_error(exc):
                return _run_bode_sweep(
                    host=host,
                    port=port,
                    start_hz=start_hz,
                    stop_hz=stop_hz,
                    points=points,
                    bandwidth_hz=bandwidth_hz,
                    source_vpp=source_vpp,
                    source_dbm=source_dbm,
                    timeout_ms=timeout_ms,
                    async_artifacts=async_artifacts,
                    reuse_session=reuse_session,
                    iteration_number=iteration_number,
                    _retry_stale_session=False,
                )
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
                "source_vpp": source_vpp,
                "source_dbm": effective_source_dbm,
            },
            "duration_s": round(time.perf_counter() - started, 3),
            "timestamp": time.time(),
        }
    finally:
        if bode is not None and close_when_done:
            try:
                bode.unlock()
            except Exception:
                pass
            try:
                bode.close()
            except Exception:
                pass


def _bode_connection_key(host: str, port: int) -> str:
    return f"{host}:{int(port)}"


def _has_bode_connection(host: str, port: int) -> bool:
    key = _bode_connection_key(host, port)
    with BODE_CONNECTION_LOCK:
        cached = BODE_CONNECTIONS.get(key)
        client = cached.get("client") if cached else None
        return isinstance(client, BodeScpiClient) and getattr(client, "_inst", None) is not None


def _get_bode_connection(
    resource_name: str,
    *,
    host: str,
    port: int,
    timeout_ms: int,
    runner: Bode100Driver | None = None,
) -> tuple[BodeScpiClient, str, bool]:
    key = _bode_connection_key(host, port)
    with BODE_CONNECTION_LOCK:
        cached = BODE_CONNECTIONS.get(key)
        if cached is not None:
            client = cached.get("client")
            identity = str(cached.get("identity") or "")
            if isinstance(client, BodeScpiClient):
                try:
                    client.timeout_ms = timeout_ms
                    if getattr(client, "_inst", None) is not None:
                        client._inst.timeout = timeout_ms
                except Exception:
                    pass
                return client, identity, True

        client = BodeScpiClient(resource_name=resource_name, timeout_ms=timeout_ms)
        try:
            client.connect()
            identity = client.idn()
            try:
                client.lock()
            except Exception:
                pass
            # Retain the driver as long as the cached VISA session. Otherwise
            # its stdin pipe is garbage-collected and the console ScpiRunner
            # interprets EOF as "press enter to stop".
            BODE_CONNECTIONS[key] = {"client": client, "identity": identity, "runner": runner}
            return client, identity, False
        except Exception:
            try:
                client.close()
            except Exception:
                pass
            raise


def _drop_bode_connection(host: str, port: int) -> None:
    key = _bode_connection_key(host, port)
    with BODE_CONNECTION_LOCK:
        cached = BODE_CONNECTIONS.pop(key, None)
    client = cached.get("client") if cached else None
    runner = cached.get("runner") if cached else None
    if isinstance(client, BodeScpiClient):
        try:
            client.unlock()
        except Exception:
            pass
        try:
            client.close()
        except Exception:
            pass
    if isinstance(runner, Bode100Driver):
        runner.stop_scpi_server()


def _is_stale_bode_session_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "invalid session" in message
        or "session handle" in message
        or "resource might be closed" in message
        or "connection is closed" in message
        or "connection for the given session has been lost" in message
        or "vi_error_conn_lost" in message
    )


def _optional_bode_source_vpp(payload: dict, default: float | None = None) -> float | None:
    raw = payload.get("source_vpp")
    if raw is None or raw == "":
        return default
    return float(raw)


def _optional_bode_source_dbm(payload: dict) -> float | None:
    # Legacy compatibility for old saved runs and external callers. New UI sends source_vpp.
    if payload.get("source_vpp") is not None:
        return None
    raw = payload.get("source_dbm")
    if raw is None or raw == "":
        return None
    return float(raw)


def _vpp_to_dbm(vpp: float, impedance_ohm: float = 50.0) -> float:
    vrms = float(vpp) / (2.0 * math.sqrt(2.0))
    power_w = (vrms * vrms) / impedance_ohm
    return 10.0 * math.log10(power_w / 1e-3)


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
                    low_v=float(payload.get("low_v", 0.1)),
                    high_v=float(payload.get("high_v", 1.1)),
                    channel=channel,
                )
            elif mode_name == "pulse":
                fg.configure_pulse_levels(
                    frequency_hz=float(payload.get("frequency_hz", 10000.0)),
                    low_v=float(payload.get("low_v", 0.1)),
                    high_v=float(payload.get("high_v", 1.1)),
                    width_s=_optional_float(payload.get("pulse_width_s")),
                    channel=channel,
                )
            elif mode_name == "dc":
                fg.configure_dc(float(payload.get("dc_level_v", payload.get("offset_v", 0.0))), channel=channel)
            elif mode_name == "sine":
                fg.configure_sine(
                    frequency_hz=float(payload.get("frequency_hz", 10000.0)),
                    amplitude_vpp=float(payload.get("amplitude_vpp", 1.0)),
                    offset_v=float(payload.get("offset_v", 0.6)),
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


def _scope_capture_compact_file_path(capture_id: str, timestamp: float) -> Path:
    time_tag = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
    return SCOPE_CAPTURE_DIR / f"{time_tag}_{capture_id}_scope.npz"


def _linear_x_metadata(x: np.ndarray) -> tuple[float, float, int]:
    x_array = np.asarray(x, dtype=np.float64)
    points = int(x_array.size)
    if points <= 0:
        return 0.0, 0.0, 0
    x_start = float(x_array[0])
    if points > 1:
        x_increment = float((x_array[-1] - x_array[0]) / float(points - 1))
    else:
        x_increment = 0.0
    return x_start, x_increment, points


def _schedule_scope_capture_artifact(channels: dict[str, dict], path: Path, metadata: dict) -> bool:
    compact_channels = {}
    for source, record in channels.items():
        compact_channels[source] = {
            "x": np.asarray(record["x"], dtype=np.float64),
            "y": np.asarray(record["y"], dtype=np.float32),
            "x_unit": record.get("x_unit") or "s",
            "y_unit": record.get("y_unit") or "V",
            "original_points": int(record.get("original_points") or len(record["y"])),
            "transfer_encoding": record.get("transfer_encoding") or "",
        }
    return _submit_artifact(_write_scope_capture_artifact, compact_channels, path, copy.deepcopy(metadata))


def _write_scope_capture_artifact(channels: dict[str, dict], path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sources = [str(source).upper() for source in channels]
    first = channels[sources[0]] if sources else None
    x_start, x_increment, points = _linear_x_metadata(first["x"] if first else np.array([], dtype=np.float64))
    payload: dict[str, object] = {
        "format_version": np.array(2, dtype=np.int16),
        "sources": np.asarray(sources, dtype="U8"),
        "x_start": np.array(x_start, dtype=np.float64),
        "x_increment": np.array(x_increment, dtype=np.float64),
        "points": np.array(points, dtype=np.int64),
        **metadata,
    }
    if first:
        payload["x_unit"] = np.asarray(first.get("x_unit") or "s")
    for source in sources:
        record = channels[source]
        safe_source = "".join(char for char in source.upper() if char.isalnum() or char in {"_", "-"})
        payload[f"y_{safe_source}"] = np.asarray(record["y"], dtype=np.float32)
        payload[f"y_unit_{safe_source}"] = np.asarray(record.get("y_unit") or "V")
        payload[f"original_points_{safe_source}"] = np.array(
            int(record.get("original_points") or len(record["y"])),
            dtype=np.int64,
        )
        payload[f"transfer_encoding_{safe_source}"] = np.asarray(record.get("transfer_encoding") or "")
    np.savez_compressed(path, **payload)


def _load_scope_waveform_npz(data_file: Path, source: str | None = None) -> dict | None:
    if not data_file.exists():
        return None
    with np.load(data_file, allow_pickle=False) as payload:
        requested = str(source or "").upper()
        if "format_version" in payload.files and int(np.asarray(payload["format_version"]).item()) >= 2:
            sources = [str(item).upper() for item in np.asarray(payload["sources"]).tolist()] if "sources" in payload.files else []
            selected = requested if requested in sources else (sources[0] if sources else "")
            if not selected:
                return None
            safe_source = "".join(char for char in selected.upper() if char.isalnum() or char in {"_", "-"})
            y_key = f"y_{safe_source}"
            if y_key not in payload.files:
                return None
            points = int(np.asarray(payload["points"]).item()) if "points" in payload.files else int(len(payload[y_key]))
            x_start = float(np.asarray(payload["x_start"]).item()) if "x_start" in payload.files else 0.0
            x_increment = float(np.asarray(payload["x_increment"]).item()) if "x_increment" in payload.files else 0.0
            x_values = x_start + np.arange(points, dtype=np.float64) * x_increment
            y_values = np.asarray(payload[y_key], dtype=np.float64)
            if y_values.size != x_values.size:
                count = min(int(y_values.size), int(x_values.size))
                x_values = x_values[:count]
                y_values = y_values[:count]
            return {
                "source": selected,
                "x": x_values,
                "y": y_values,
                "x_unit": str(np.asarray(payload["x_unit"]).item()) if "x_unit" in payload.files else "s",
                "y_unit": str(np.asarray(payload[f"y_unit_{safe_source}"]).item()) if f"y_unit_{safe_source}" in payload.files else "V",
                "original_points": int(np.asarray(payload[f"original_points_{safe_source}"]).item())
                if f"original_points_{safe_source}" in payload.files
                else int(len(y_values)),
                "transfer_encoding": str(np.asarray(payload[f"transfer_encoding_{safe_source}"]).item())
                if f"transfer_encoding_{safe_source}" in payload.files
                else "",
                "capture_id": str(np.asarray(payload["capture_id"]).item()) if "capture_id" in payload.files else None,
                "timestamp": float(np.asarray(payload["timestamp"]).item()) if "timestamp" in payload.files else None,
            }

        x_values = np.asarray(payload["x"], dtype=np.float64)
        y_values = np.asarray(payload["y"], dtype=np.float64)
        payload_source = str(np.asarray(payload["source"]).item()) if "source" in payload.files else requested
        return {
            "source": payload_source.upper() if payload_source else requested,
            "x": x_values,
            "y": y_values,
            "x_unit": str(np.asarray(payload["x_unit"]).item()) if "x_unit" in payload.files else "s",
            "y_unit": str(np.asarray(payload["y_unit"]).item()) if "y_unit" in payload.files else "V",
            "original_points": int(np.asarray(payload["original_points"]).item()) if "original_points" in payload.files else int(len(y_values)),
            "transfer_encoding": str(np.asarray(payload["transfer_encoding"]).item()) if "transfer_encoding" in payload.files else "",
            "capture_id": str(np.asarray(payload["capture_id"]).item()) if "capture_id" in payload.files else None,
            "timestamp": float(np.asarray(payload["timestamp"]).item()) if "timestamp" in payload.files else None,
        }


def _store_full_scope_capture(capture_id: str, captures: list, timestamp: float, async_save: bool = False) -> tuple[str, dict[str, dict]]:
    file_path = _scope_capture_compact_file_path(capture_id, timestamp)
    channel_records: dict[str, dict] = {}
    for capture in captures:
        source = str(capture.source).upper()
        x_array = np.asarray(capture.x, dtype=np.float64)
        y_array = np.asarray(capture.y, dtype=np.float64)
        channel_records[source] = {
            "source": source,
            "x": x_array,
            "y": y_array,
            "x_unit": capture.x_unit,
            "y_unit": capture.y_unit,
            "original_points": int(capture.original_points or len(capture.y)),
            "transfer_encoding": capture.transfer_encoding,
            "data_file": "",
            "data_file_pending": False,
        }
    metadata = {
        "capture_id": capture_id,
        "timestamp": timestamp,
    }
    # NPZ is the source of truth and is much cheaper than rendering. Write it
    # immediately so large waveform arrays can be released before returning.
    _write_scope_capture_artifact(channel_records, file_path, metadata)
    try:
        data_file = str(file_path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        data_file = str(file_path)
    for record in channel_records.values():
        record["data_file"] = data_file
    return data_file, channel_records


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


def _refresh_status_artifact_readiness(status: dict, *, include_history: bool = True) -> bool:
    """Clear stale async-artifact flags once their files reach disk."""

    changed = False
    seen: set[int] = set()
    records: list[dict] = []
    if include_history:
        history = status.get("history")
        if isinstance(history, list):
            records.extend(record for record in history if isinstance(record, dict))
    for key in ("current", "best"):
        record = status.get(key)
        if isinstance(record, dict):
            records.append(record)

    for record in records:
        identity = id(record)
        if identity in seen:
            continue
        seen.add(identity)
        for result_key, file_key, pending_key, error_key in (
            ("scope_result", "scope_png", "scope_png_pending", "scope_png_error"),
            ("bode_result", "bode_png", "bode_png_pending", "bode_png_error"),
        ):
            result = record.get(result_key)
            if not isinstance(result, dict) or result.get(pending_key) is not True:
                continue
            path = _path_from_result_reference(result.get(file_key))
            if path is None or not path.is_file() or path.stat().st_size <= 0:
                continue
            result[pending_key] = False
            result[error_key] = None
            changed = True
        bode_result = record.get("bode_result")
        if isinstance(bode_result, dict) and bode_result.get("data_file_pending") is True:
            path = _path_from_result_reference(bode_result.get("data_file"))
            if path is not None and path.is_file() and path.stat().st_size > 0:
                bode_result["data_file_pending"] = False
                changed = True
        scope_result = record.get("scope_result")
        waveforms = scope_result.get("waveforms") if isinstance(scope_result, dict) else None
        if isinstance(waveforms, list):
            for waveform in waveforms:
                if not isinstance(waveform, dict) or waveform.get("data_file_pending") is not True:
                    continue
                path = _path_from_result_reference(waveform.get("data_file"))
                if path is not None and path.is_file() and path.stat().st_size > 0:
                    waveform["data_file_pending"] = False
                    changed = True
    return changed


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
            loaded = _load_scope_waveform_npz(data_file, str(waveform.get("source") or ""))
            if not loaded:
                continue
            source = str(loaded["source"])
            x_values = np.asarray(loaded["x"], dtype=np.float64)
            y_values = np.asarray(loaded["y"], dtype=np.float64)
            x_unit = str(loaded.get("x_unit") or waveform.get("x_unit") or "s")
            y_unit = str(loaded.get("y_unit") or waveform.get("y_unit") or "V")
            original_points = int(loaded.get("original_points") or len(y_values))
            transfer_encoding = str(loaded.get("transfer_encoding") or waveform.get("transfer_encoding") or "")
            if loaded.get("capture_id"):
                capture_id = str(loaded["capture_id"])
            if loaded.get("timestamp") is not None:
                created_at = float(loaded["timestamp"])
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
    title: str = "Latest Bode Sweep - Full Data",
    figsize: tuple[float, float] = (16, 6),
    dpi: int = 150,
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
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 15,
        }
    )
    fig, gain_ax = plt.subplots(figsize=figsize, dpi=dpi)
    phase_ax = gain_ax.twinx()
    gain_line, = gain_ax.semilogx(frequency, magnitude, color="#ea4335", linewidth=1.6, label="Gain")
    phase_line, = phase_ax.semilogx(frequency, phase, color="#1a73e8", linewidth=1.6, label="Phase")
    gain_ax.axhline(0, color="#9aa0a6", linewidth=1.0)
    phase_ax.axhline(0, color="#9aa0a6", linewidth=1.0, alpha=0.6)

    margins = margins or {}
    phase_crossover = margins.get("phase_crossover_hz")
    gain_crossover = margins.get("gain_crossover_hz")
    phase_margin = margins.get("phase_margin_deg")
    gain_margin = margins.get("gain_margin_db")

    def _fmt_freq(value: object) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "--"
        if not math.isfinite(number) or number <= 0:
            return "--"
        if number >= 1e6:
            return f"{number / 1e6:.3g} MHz"
        if number >= 1e3:
            return f"{number / 1e3:.3g} kHz"
        return f"{number:.3g} Hz"

    def _fmt_metric(value: object, unit: str) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return f"-- {unit}".strip()
        if not math.isfinite(number):
            return f"-- {unit}".strip()
        return f"{number:.2f} {unit}".strip()

    def _interp_at_frequency(x_value: object, values: np.ndarray) -> float | None:
        try:
            freq = float(x_value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(freq) or freq <= 0:
            return None
        if freq < float(np.min(frequency)) or freq > float(np.max(frequency)):
            return None
        return float(np.interp(np.log10(freq), np.log10(frequency), values))

    if phase_crossover:
        crossover_hz = float(phase_crossover)
        phase_at_cross = _interp_at_frequency(crossover_hz, phase)
        gain_ax.axvline(crossover_hz, color="#ea4335", linestyle="--", linewidth=1.5, alpha=0.85)
        gain_ax.scatter([crossover_hz], [0.0], color="#ea4335", s=46, zorder=5)
        if phase_at_cross is not None:
            phase_ax.scatter([crossover_hz], [phase_at_cross], color="#1a73e8", s=46, zorder=5)
        gain_ax.annotate(
            f"fc = {_fmt_freq(crossover_hz)}\nPM = {_fmt_metric(phase_margin, 'deg')}",
            xy=(crossover_hz, 0.0),
            xytext=(10, 34),
            textcoords="offset points",
            color="#202124",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#ea4335", "alpha": 0.92},
            arrowprops={"arrowstyle": "->", "color": "#ea4335", "lw": 1.1},
        )
    if gain_crossover:
        phase_cross_hz = float(gain_crossover)
        gain_at_phase_cross = _interp_at_frequency(phase_cross_hz, magnitude)
        gain_ax.axvline(phase_cross_hz, color="#1a73e8", linestyle="--", linewidth=1.5, alpha=0.85)
        phase_ax.scatter([phase_cross_hz], [0.0], color="#1a73e8", s=46, zorder=5)
        if gain_at_phase_cross is not None:
            gain_ax.scatter([phase_cross_hz], [gain_at_phase_cross], color="#ea4335", s=46, zorder=5)
        gain_ax.annotate(
            f"phase = 0 deg\nGM = {_fmt_metric(gain_margin, 'dB')}",
            xy=(phase_cross_hz, gain_at_phase_cross if gain_at_phase_cross is not None else 0.0),
            xytext=(-118, -48),
            textcoords="offset points",
            color="#202124",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#1a73e8", "alpha": 0.92},
            arrowprops={"arrowstyle": "->", "color": "#1a73e8", "lw": 1.1},
        )
    else:
        endpoint_hz = float(frequency[-1])
        endpoint_gain = float(magnitude[-1])
        endpoint_phase = float(phase[-1])
        gain_ax.axvline(endpoint_hz, color="#1a73e8", linestyle="--", linewidth=1.5, alpha=0.85)
        gain_ax.scatter([endpoint_hz], [endpoint_gain], color="#ea4335", s=46, zorder=5)
        phase_ax.scatter([endpoint_hz], [endpoint_phase], color="#1a73e8", s=46, zorder=5)
        gain_ax.annotate(
            f"{_fmt_freq(endpoint_hz)} phase = {endpoint_phase:.2f} deg\nGM < {endpoint_gain:.2f} dB",
            xy=(endpoint_hz, endpoint_gain),
            xytext=(-168, -48),
            textcoords="offset points",
            color="#202124",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#1a73e8", "alpha": 0.92},
            arrowprops={"arrowstyle": "->", "color": "#1a73e8", "lw": 1.1},
        )

    gain_ax.set_xlim(float(np.min(frequency)), float(np.max(frequency)))
    gain_ax.set_ylim(-100, 100)
    phase_ax.set_ylim(-200, 200)
    gain_ax.set_title(title, pad=14)
    gain_ax.set_xlabel("Frequency (Hz)")
    gain_ax.set_ylabel("Gain (dB)", color="#ea4335")
    phase_ax.set_ylabel("Phase (deg)", color="#1a73e8")
    gain_ax.tick_params(axis="y", colors="#ea4335")
    phase_ax.tick_params(axis="y", colors="#1a73e8")
    gain_ax.spines["left"].set_color("#ea4335")
    phase_ax.spines["right"].set_color("#1a73e8")
    gain_ax.grid(True, which="both", color="#d9dee7", linewidth=0.8, alpha=0.8)
    gain_ax.legend(
        [gain_line, phase_line],
        ["Gain", "Phase"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        framealpha=0.92,
        borderpad=0.45,
        handlelength=2.2,
    )
    fig.subplots_adjust(top=0.86, left=0.075, right=0.925, bottom=0.14)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return _scope_png_public_path(path)


def _schedule_bode_data_artifact(
    *,
    frequency_hz: list[float],
    magnitude_db: list[float],
    phase_deg: list[float],
    path: Path,
    metadata: dict,
) -> bool:
    return _submit_artifact(
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


def _scope_input_edge_times_s(channels: dict) -> list[tuple[float, str]]:
    record = channels.get("CH1") or channels.get("ch1")
    if not isinstance(record, dict):
        return []
    x_array = np.asarray(record.get("x", []), dtype=np.float64)
    y_array = np.asarray(record.get("y", []), dtype=np.float64)
    total = min(x_array.size, y_array.size)
    if total < 2:
        return []
    x_array = x_array[:total]
    y_array = y_array[:total]
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    if not np.any(finite):
        return []
    x_array = x_array[finite]
    y_array = y_array[finite]
    if x_array.size < 2:
        return []
    # Use exactly the same robust threshold, hysteresis and debounce as the
    # analyzer. Separate plot-only edge logic previously allowed annotations
    # to disagree with the values used for scoring.
    edge_indices = ResponseAnalyzer(TuningTargets()).input_edge_indices(y_array.tolist())
    return [
        (float(x_array[index]), kind)
        for index, kind in edge_indices
        if 0 <= index < x_array.size
    ]


def _draw_scope_settling_markers(
    axis,
    *,
    channels: dict,
    x0: float,
    x1: float,
    x_scale: float,
    metrics: ResponseMetrics | None = None,
) -> None:
    # Old records only had a shared invalid-waveform reason. V2+ records carry
    # independent OS/US validity flags, so one failed direction must not hide
    # the valid settling marker for the other direction.
    legacy_invalid = (
        metrics is not None
        and int(getattr(metrics, "settling_analysis_version", 1) or 1) < 2
        and any(
        "invalid transient waveform" in str(reason).lower()
        for reason in (metrics.pass_reasons or [])
        )
    )
    edge_times = _selected_scope_settling_marker_edges(_scope_input_edge_times_s(channels))
    if not edge_times:
        return

    x_min = 0.0
    x_max = max(0.0, (x1 - x0) * x_scale)
    edge_colors = {"rising": "#5f6368", "falling": "#5f6368"}
    settle_color = "#ea4335"
    label_slots: list[float] = []

    for edge_time_s, kind in edge_times:
        settling_time_s = None
        settling_valid = not legacy_invalid
        if metrics is not None:
            if kind == "rising":
                settling_time_s = float(metrics.undershoot_settling_time_s)
                settling_valid = bool(getattr(metrics, "undershoot_settling_valid", True)) and not legacy_invalid
            else:
                settling_time_s = float(metrics.overshoot_settling_time_s)
                settling_valid = bool(getattr(metrics, "overshoot_settling_valid", True)) and not legacy_invalid
        if not settling_valid:
            settling_time_s = None
        elif settling_time_s is None or not math.isfinite(settling_time_s) or settling_time_s < 0:
            continue
        edge_x = (edge_time_s - x0) * x_scale
        settle_x = (edge_time_s + settling_time_s - x0) * x_scale if settling_time_s is not None else edge_x
        edge_visible = x_min <= edge_x <= x_max
        settle_visible = x_min <= settle_x <= x_max

        if edge_visible:
            axis.axvline(
                edge_x,
                color=edge_colors.get(kind, "#5f6368"),
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
                zorder=0,
            )
        if not settling_valid and edge_visible:
            label_x = min(edge_x + 0.035 * (x_max - x_min), x_max - 0.04 * (x_max - x_min))
            axis.text(
                label_x,
                0.805,
                "Ts --",
                transform=axis.get_xaxis_transform(),
                color=settle_color,
                fontsize=13,
                fontweight="bold",
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#f3b7b0", "alpha": 0.94},
                clip_on=True,
            )
            continue
        if settle_visible:
            axis.axvline(
                settle_x,
                color=settle_color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.85,
                zorder=0,
            )
        if edge_visible and settle_visible and abs(settle_x - edge_x) > 0:
            label_x = (edge_x + settle_x) / 2.0
            label_x = min(max(label_x, x_min + 0.06 * (x_max - x_min)), x_max - 0.06 * (x_max - x_min))
            y_arrow = 0.78
            y_text = 0.805
            for used_x in label_slots:
                if abs(label_x - used_x) < max(0.08 * (x_max - x_min), 1e-12):
                    y_text = 0.69
                    y_arrow = 0.665
                    break
            label_slots.append(label_x)
            axis.annotate(
                "",
                xy=(edge_x, y_arrow),
                xytext=(settle_x, y_arrow),
                xycoords=axis.get_xaxis_transform(),
                textcoords=axis.get_xaxis_transform(),
                arrowprops={"arrowstyle": "<->", "color": settle_color, "linewidth": 1.4},
                annotation_clip=True,
            )
            axis.text(
                label_x,
                y_text,
                f"Ts {settling_time_s * 1e6:.1f} us",
                transform=axis.get_xaxis_transform(),
                color=settle_color,
                fontsize=13,
                fontweight="bold",
                ha="center",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#f3b7b0", "alpha": 0.94},
                clip_on=True,
            )


def _is_recoverable_scope_capture_error(exc: Exception) -> bool:
    """Return whether reconnecting and flushing the scope may fix the error."""

    if isinstance(exc, TimeoutError):
        return True
    message = str(exc).lower()
    return _is_stale_scope_session_error(exc) or any(
        marker in message
        for marker in (
            "single acquisition did not complete",
            "timeout",
            "timed out",
            "vi_error_tmo",
            "low on memory",
            "out of memory",
            "insufficient memory",
        )
    )


def _scope_response_metrics_from_plot_channels(
    channels: dict,
    *,
    response_channel: str = "CH3",
    targets: TuningTargets | None = None,
) -> ResponseMetrics | None:
    ch1 = channels.get("CH1") or channels.get("ch1")
    response_key = str(response_channel or "CH3").upper()
    response = channels.get(response_key) or channels.get(response_key.lower())
    if not ch1 or not response:
        return None

    time_array = np.asarray(response.get("x", []), dtype=np.float64)
    response_array = np.asarray(response.get("y", []), dtype=np.float64)
    input_time = np.asarray(ch1.get("x", []), dtype=np.float64)
    input_array = np.asarray(ch1.get("y", []), dtype=np.float64)
    total = min(time_array.size, response_array.size)
    input_total = min(input_time.size, input_array.size)
    if total < 2 or input_total < 2:
        return None

    time_array = time_array[:total]
    response_array = response_array[:total]
    finite = np.isfinite(time_array) & np.isfinite(response_array)
    if not np.any(finite):
        return None
    time_array = time_array[finite]
    response_array = response_array[finite]
    if time_array.size < 2:
        return None

    input_time = input_time[:input_total]
    input_array = input_array[:input_total]
    input_finite = np.isfinite(input_time) & np.isfinite(input_array)
    if not np.any(input_finite):
        return None
    input_time = input_time[input_finite]
    input_array = input_array[input_finite]
    if input_time.size < 2:
        return None

    order = np.argsort(input_time)
    input_time = input_time[order]
    input_array = input_array[order]
    if input_time.size == time_array.size and np.allclose(input_time, time_array, rtol=0.0, atol=1e-15):
        aligned_input = input_array
    else:
        aligned_input = np.interp(time_array, input_time, input_array)

    try:
        return ResponseAnalyzer(targets or TuningTargets()).analyze(
            Waveform(
                time_s=time_array.tolist(),
                vout_v=response_array.tolist(),
                input_v=aligned_input.tolist(),
            )
        )
    except Exception:
        return None


def _selected_scope_settling_marker_edges(edges: list[tuple[float, str]]) -> list[tuple[float, str]]:
    """Show only the first load step and final return step markers."""

    selected: list[tuple[float, str]] = []
    rising = [edge for edge in edges if edge[1] == "rising"]
    falling = [edge for edge in edges if edge[1] == "falling"]
    if rising:
        selected.append(rising[0])
    if len(falling) >= 2:
        selected.append(falling[1])
    elif falling:
        selected.append(falling[-1])
    return sorted(selected, key=lambda item: item[0])


def _plot_full_scope_capture_png(
    entry: dict,
    axis_settings: dict | None = None,
    path: Path = LATEST_SCOPE_PNG,
    title: str = "Latest Scope Capture - Full Data",
    figsize: tuple[float, float] = (16, 6),
    dpi: int = 150,
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
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    fig, left_ax = plt.subplots(figsize=figsize, dpi=dpi)
    right_ax = left_ax.twinx()
    colors = ["#1a73e8", "#ea4335", "#34a853", "#fbbc04", "#8ab4f8", "#f28b82", "#81c995", "#fde293"]
    lines = []
    labels = []
    response_filter = ResponseAnalyzer(TuningTargets())
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
            linewidth=0.9,
            color="#111111" if source.upper() == "CH3" else colors[index % len(colors)],
            alpha=0.3 if source.upper() == "CH3" else 1.0,
            rasterized=True,
        )
        lines.append(line)
        labels.append(source)
        if source.upper() == "CH3":
            decision_cutoff_hz = ResponseAnalyzer.RESPONSE_LOWPASS_CUTOFF_HZ
            decision_label = f"CH3 {decision_cutoff_hz / 1e6:g}MHz LPF"
            filtered_y = response_filter._zero_phase_lowpass(
                x_array[:total].tolist(),
                y_array[:total].tolist(),
                cutoff_hz=decision_cutoff_hz,
            )
            if len(filtered_y) == total:
                filtered_line, = axis.plot(
                    (x_array[:total] - x0) * x_scale,
                    filtered_y,
                    label=decision_label,
                    linewidth=1.8,
                    color="#ea4335",
                    alpha=1.0,
                    zorder=10,
                    rasterized=True,
                )
                lines.append(filtered_line)
                labels.append(decision_label)

    left_color = "#1a73e8"
    right_color = "#ea4335"
    left_ax.set_xlim(0, (x1 - x0) * x_scale)
    left_ax.set_ylim(axis_settings["leftMin"], axis_settings["leftMax"])
    right_ax.set_ylim(axis_settings["rightMin"], axis_settings["rightMax"])
    left_ax.set_title(title, pad=16)
    left_ax.set_xlabel(f"Time ({x_unit})")
    left_ax.set_ylabel("Voltage (V)", color=left_color)
    right_ax.set_ylabel("Voltage (V)", color=right_color)
    left_ax.tick_params(axis="y", colors=left_color)
    right_ax.tick_params(axis="y", colors=right_color)
    left_ax.spines["left"].set_color(left_color)
    right_ax.spines["right"].set_color(right_color)
    left_ax.grid(True, color="#d9dee7", linewidth=0.8, alpha=0.8)
    settling_metrics = entry.get("settling_metrics")
    if not isinstance(settling_metrics, ResponseMetrics):
        settling_metrics = _scope_response_metrics_from_plot_channels(channels)
    _draw_scope_settling_markers(
        left_ax,
        channels=channels,
        x0=x0,
        x1=x1,
        x_scale=x_scale,
        metrics=settling_metrics,
    )
    if lines:
        left_ax.legend(
            lines,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(4, max(1, len(labels))),
            framealpha=0.92,
        )
    fig.subplots_adjust(top=0.78, left=0.075, right=0.925, bottom=0.14)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return _scope_png_public_path(path)


def _schedule_bode_png_artifact(
    *,
    frequency_hz: list[float],
    magnitude_db: list[float],
    phase_deg: list[float],
    margins: dict | None,
    path: Path,
    title: str,
) -> bool:
    frequency_copy = np.asarray(frequency_hz, dtype=np.float64).copy()
    magnitude_copy = np.asarray(magnitude_db, dtype=np.float64).copy()
    phase_copy = np.asarray(phase_deg, dtype=np.float64).copy()
    margins_copy = copy.deepcopy(margins or {})
    return _submit_artifact(
        _write_bode_png_artifact,
        frequency_copy,
        magnitude_copy,
        phase_copy,
        margins_copy,
        path,
        title,
    )


def _write_bode_png_artifact(
    frequency_hz: np.ndarray,
    magnitude_db: np.ndarray,
    phase_deg: np.ndarray,
    margins: dict,
    path: Path,
    title: str,
) -> None:
    _plot_full_bode_sweep_png(
        frequency_hz=frequency_hz.tolist(),
        magnitude_db=magnitude_db.tolist(),
        phase_deg=phase_deg.tolist(),
        margins=margins,
        path=path,
        title=title,
    )
    LATEST_BODE_PNG.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, LATEST_BODE_PNG)


def _schedule_scope_png_artifact(
    *,
    capture_entry: dict,
    scope_axis_settings: dict | None,
    path: Path,
    title: str,
) -> bool:
    capture_copy = _copy_scope_capture_for_plot(capture_entry)
    axis_copy = copy.deepcopy(scope_axis_settings)
    return _submit_artifact(_write_scope_png_artifact, capture_copy, axis_copy, path, title)


def _write_scope_png_artifact(
    capture_entry: dict,
    scope_axis_settings: dict | None,
    path: Path,
    title: str,
) -> None:
    _plot_full_scope_capture_png(capture_entry, scope_axis_settings, path=path, title=title)
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
        # Freeze the synchronous analysis result for the background renderer.
        # Recomputing it later can produce a PNG label that disagrees with the
        # score/history when the backend code changes or an artifact is queued.
        "settling_metrics": capture_entry.get("settling_metrics"),
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
    iteration_number: int | None = None,
    response_channel: str = "CH3",
    response_targets: TuningTargets | None = None,
) -> dict:
    started = time.perf_counter()
    stage_started = started
    stage_durations: dict[str, float] = {}

    def mark_stage(name: str) -> None:
        nonlocal stage_started
        now = time.perf_counter()
        stage_durations[name] = round(now - stage_started, 3)
        stage_started = now
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
    scope_memory_guard: dict[str, object] = {}
    scope_recovery_attempts: list[dict[str, object]] = []
    autotune_memory_guard_enabled = iteration_number is not None
    if function_generator_frequency_hz is not None and function_generator_frequency_hz > 0:
        function_generator_period_s = 1.0 / float(function_generator_frequency_hz)
        scope_window_s = max(
            1e-9,
            min(10.0, SCOPE_TRIGGER_OFFSET_FROM_LEFT_S + function_generator_period_s + 3e-6),
        )
    with DEVICE_LOCK:
        last_error: Exception | None = None
        for attempt in range(SCOPE_AUTOTUNE_CAPTURE_ATTEMPTS):
            scope = None
            try:
                scope, opened = _get_scope_connection(resource, timeout_ms=6000)
                mark_stage("session")
                waveforms = []
                measurement_rows = []
                capture_cache_entry = {
                    "created_at": timestamp,
                    "resource": resource,
                    "channels": {},
                }
                config_signature = (
                    tuple(safe_channels),
                    scope_window_s,
                    SCOPE_AUTOTUNE_MAX_RECORD_LENGTH if autotune_memory_guard_enabled else None,
                )
                config_reused = bool(
                    not opened and getattr(scope, "_autotune_config_signature", None) == config_signature
                )
                if config_reused:
                    scope_scale_s_per_div = getattr(scope, "_autotune_scale_s_per_div", None)
                    scope_actual_window_s = getattr(scope, "_autotune_actual_window_s", None)
                    scope_trigger_position_percent = getattr(scope, "_autotune_trigger_position_percent", None)
                else:
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
                    scope._autotune_config_signature = config_signature
                    scope._autotune_scale_s_per_div = scope_scale_s_per_div
                    scope._autotune_actual_window_s = scope_actual_window_s
                    scope._autotune_trigger_position_percent = scope_trigger_position_percent
                mark_stage("configure")
                if autotune_memory_guard_enabled:
                    scope_memory_guard = scope.prepare_autotune_acquisition(
                        max_record_length=SCOPE_AUTOTUNE_MAX_RECORD_LENGTH,
                        maintenance_interval=SCOPE_AUTOTUNE_MAINTENANCE_INTERVAL,
                    )
                mark_stage("scope_memory_guard")
                force_after_s = 0.5 if scope_window_s is None else max(0.25, min(2.0, scope_window_s * 1.5))
                scope.single_acquisition(timeout_s=8.0, force_after_s=force_after_s)
                mark_stage("acquisition")
                captures = []
                for channel in safe_channels:
                    captures.append(scope.capture_waveform(channel, start=1, stop=stop, max_plot_points=None))
                mark_stage("waveform_transfer")
                data_file, full_records = _store_full_scope_capture(
                    capture_id,
                    captures,
                    timestamp,
                    async_save=async_artifacts,
                )
                mark_stage("data_artifact")
                for channel, full_record in full_records.items():
                    capture_cache_entry["channels"][channel] = full_record
                for capture in captures:
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
                            "data_file_pending": False,
                            "transfer_encoding": capture.transfer_encoding,
                        }
                    )
                    source_channel = str(capture.source).upper()
                    for measurement in safe_measurements:
                        try:
                            value = scope.read_immediate_measurement(source_channel, measurement)
                            measurement_rows.append({"source": source_channel, "measurement": measurement, "value": value, "ok": True})
                        except Exception as exc:
                            measurement_rows.append(
                                {"source": source_channel, "measurement": measurement, "value": None, "ok": False, "error": str(exc)}
                            )
                mark_stage("display_data")
                # Calculate the marker values while this capture is still the
                # active synchronous result. The PNG task may run later, but it
                # must display exactly this immutable metrics snapshot rather
                # than re-analyzing the waveform in a background thread.
                capture_cache_entry["settling_metrics"] = _scope_response_metrics_from_plot_channels(
                    capture_cache_entry["channels"],
                    response_channel=response_channel,
                    targets=response_targets,
                )
                _remember_scope_capture(capture_id, capture_cache_entry)
                scope_png = None
                scope_png_error = None
                scope_png_pending = False
                try:
                    scope_png_path = RESULTS_DIR / "scope_captures" / f"{time.strftime('%Y%m%d_%H%M%S')}_{capture_id}.png"
                    scope_title = (
                        f"Iteration {iteration_number} - Scope Capture - Full Data"
                        if iteration_number is not None
                        else "Latest Scope Capture - Full Data"
                    )
                    if async_artifacts:
                        scope_png = _scope_png_public_path(scope_png_path)
                        scope_png_pending = _schedule_scope_png_artifact(
                            capture_entry=capture_cache_entry,
                            scope_axis_settings=scope_axis_settings,
                            path=scope_png_path,
                            title=scope_title,
                        )
                        if not scope_png_pending:
                            scope_png = None
                            scope_png_error = "PNG rendering deferred because the artifact queue is busy."
                    else:
                        scope_png = _plot_full_scope_capture_png(
                            capture_cache_entry,
                            scope_axis_settings,
                            path=scope_png_path,
                            title=scope_title,
                        )
                        LATEST_SCOPE_PNG.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(scope_png_path, LATEST_SCOPE_PNG)
                except Exception as exc:
                    scope_png_error = str(exc)
                    scope_png_pending = False
                mark_stage("png_artifact")
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
                    "config_reused": config_reused,
                    "session_retry": attempt,
                    "capture_attempts": attempt + 1,
                    "scope_memory_guard": scope_memory_guard,
                    "scope_recovery_attempts": scope_recovery_attempts,
                    "stage_durations_s": stage_durations,
                    "duration_s": round(time.perf_counter() - started, 3),
                    "timestamp": time.time(),
                }
            except Exception as exc:
                last_error = exc
                recoverable = _is_recoverable_scope_capture_error(exc)
                recovery: dict[str, object] = {
                    "attempt": attempt + 1,
                    "error": str(exc),
                    "recoverable": recoverable,
                    "forced_flush": False,
                }
                if recoverable and scope is not None and autotune_memory_guard_enabled:
                    try:
                        recovery["memory_guard"] = scope.prepare_autotune_acquisition(
                            max_record_length=SCOPE_AUTOTUNE_MAX_RECORD_LENGTH,
                            maintenance_interval=SCOPE_AUTOTUNE_MAINTENANCE_INTERVAL,
                            force_flush=True,
                        )
                        recovery["forced_flush"] = True
                    except Exception as recovery_exc:
                        recovery["recovery_error"] = str(recovery_exc)
                scope_recovery_attempts.append(recovery)
                _drop_scope_connection(resource)
                if recoverable and attempt + 1 < SCOPE_AUTOTUNE_CAPTURE_ATTEMPTS:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                break
        return {
            "ok": False,
            "resource": resource,
            "error": str(last_error) if last_error is not None else "Scope capture failed.",
            "capture_attempts": len(scope_recovery_attempts),
            "scope_recovery_attempts": scope_recovery_attempts,
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
    legacy_max_iterations = _int_field(search_payload, "max_iterations", default_search.max_iterations)
    default_coarse = getattr(default_search, "max_coarse_iterations", max(1, legacy_max_iterations // 2))
    default_refined = getattr(default_search, "max_refined_iterations", max(0, legacy_max_iterations - default_coarse))
    max_coarse_iterations = _int_field(search_payload, "max_coarse_iterations", default_coarse)
    max_refined_iterations = _int_field(search_payload, "max_refined_iterations", default_refined)
    explicit_phase_budgets = "max_coarse_iterations" in search_payload and "max_refined_iterations" in search_payload
    if not explicit_phase_budgets and legacy_max_iterations > max_coarse_iterations + max_refined_iterations:
        max_coarse_iterations += legacy_max_iterations - (max_coarse_iterations + max_refined_iterations)
    total_iterations = max(1, max_coarse_iterations + max_refined_iterations)
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
            settling_time_s=_float_field(targets_payload, "settling_time_s", 2e-6),
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
            max_iterations=total_iterations,
            max_coarse_iterations=max_coarse_iterations,
            max_refined_iterations=max_refined_iterations,
            mod0_kp=_search_parameter_from_payload(search_payload, "mod0_kp", default_search.mod0_kp, integer=True, coarse_iteration_budget=max_coarse_iterations),
            mod0_ki=_search_parameter_from_payload(search_payload, "mod0_ki", default_search.mod0_ki, integer=True, coarse_iteration_budget=max_coarse_iterations),
            mod0_kd=_search_parameter_from_payload(search_payload, "mod0_kd", default_search.mod0_kd, integer=True, coarse_iteration_budget=max_coarse_iterations),
            mod0_kpole1=_search_parameter_from_payload(search_payload, "mod0_kpole1", default_search.mod0_kpole1, integer=True, coarse_iteration_budget=max_coarse_iterations),
            mod0_kpole2=_search_parameter_from_payload(search_payload, "mod0_kpole2", default_search.mod0_kpole2, integer=True, coarse_iteration_budget=max_coarse_iterations),
            mod0_cm_gain=_search_parameter_from_payload(search_payload, "mod0_cm_gain", default_search.mod0_cm_gain, integer=True, coarse_iteration_budget=max_coarse_iterations),
            mod0_ll_bw=_search_parameter_from_payload(search_payload, "mod0_ll_bw", default_search.mod0_ll_bw, integer=True, coarse_iteration_budget=max_coarse_iterations),
            output_inductance_nh=_search_parameter_from_payload(
                search_payload,
                "output_inductance_nh",
                default_search.output_inductance_nh,
                coarse_iteration_budget=max_coarse_iterations,
            ),
            effective_lc_inductance_nh=_search_parameter_from_payload(
                search_payload,
                "effective_lc_inductance_nh",
                default_search.effective_lc_inductance_nh,
                coarse_iteration_budget=max_coarse_iterations,
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
        optimization_algorithm=str(payload.get("optimization_algorithm", "heuristic")),
        bode_config=dict(payload.get("bode_config", {}) or {}),
        function_generator_config=dict(payload.get("function_generator_config", {}) or {}),
        scope_config=dict(payload.get("scope_config", {}) or {}),
        vout_tolerance_v=_float_field(payload, "vout_tolerance_v", 0.15),
        response_abs_limit_v=_float_field(payload, "response_abs_limit_v", 0.25),
        ignore_pass_until_max_iterations=bool(payload.get("ignore_pass_until_max_iterations", True)),
        drl_workflow_mode=str(payload.get("drl_workflow_mode", "")),
        drl_model_id=str(payload.get("drl_model_id", "")),
        drl_collection_plan_id=str(payload.get("drl_collection_plan_id", "")),
        drl_episode_budget=max(1, _int_field(payload, "drl_episode_budget", 15)),
        drl_confirmation_count=max(1, _int_field(payload, "drl_confirmation_count", 3)),
        drl_hardware_protection_mode=bool(payload.get("drl_hardware_protection_mode", True)),
    )


def _search_parameter_from_payload(
    payload: dict,
    name: str,
    default: SearchParameter,
    integer: bool = False,
    coarse_iteration_budget: int | None = None,
) -> SearchParameter:
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
    parameter = SearchParameter(center=float(center), min=float(minimum), max=float(maximum), step=float(step), points=points)
    if coarse_iteration_budget is not None:
        return automatic_search_parameter(parameter, coarse_iteration_budget, integer=integer)
    return parameter


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
        "session_reused": result.get("session_reused"),
        "config_reused": result.get("config_reused"),
        "retried_stale_session": result.get("retried_stale_session"),
        "stage_durations_s": result.get("stage_durations_s"),
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
        "session_reused": result.get("session_reused"),
        "config_reused": result.get("config_reused"),
        "session_retry": result.get("session_retry"),
        "capture_attempts": result.get("capture_attempts"),
        "scope_memory_guard": result.get("scope_memory_guard"),
        "scope_recovery_attempts": result.get("scope_recovery_attempts"),
        "stage_durations_s": result.get("stage_durations_s"),
        "duration_s": result.get("duration_s"),
        "error": result.get("error"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
