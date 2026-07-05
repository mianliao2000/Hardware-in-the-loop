"""Session runner for the PID autotuning framework."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import replace
from typing import Any, Protocol

from .analyzer import ResponseAnalyzer
from .compensator import CompensatorDesign
from .models import (
    AutotuneExperimentConfig,
    ExperimentResult,
    HardwarePidCandidate,
    IterationRecord,
    PidParameters,
    PlantParams,
    ResponseMetrics,
    SearchParameter,
    SearchSpace,
    TuningConfig,
    TuningRunSnapshot,
    TuningTargets,
    Waveform,
    to_jsonable,
)
from .pid_programmer import PidProgrammer, StubPidProgrammer
from .search import GridRefinePidTuner, HardwareGridHeuristicTuner, TuningCandidate, select_best_result


class AutotuneExperimentRunner(Protocol):
    def evaluate(
        self,
        candidate: HardwarePidCandidate,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
    ) -> ExperimentResult:
        ...


class PlaceholderExperimentRunner:
    """Deterministic fake experiment used before hardware PID writes are safe."""

    def evaluate(
        self,
        candidate: HardwarePidCandidate,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
    ) -> ExperimentResult:
        waveform = self.capture_response(
            TuningCandidate(candidate.phase, config.search.initial_wc_rad_s, config.search.initial_phi_deg),
            config,
        )
        kp_error = abs(candidate.mod0_kp - 170) / 255.0
        ki_error = abs(candidate.mod0_ki - 220) / 255.0
        kd_error = abs(candidate.mod0_kd - 175) / 255.0
        fake_phase_margin = config.targets.phase_margin_deg + (candidate.mod0_kpole1 - 3) * 3.0 - kp_error * 12.0
        fake_crossover = config.targets.crossover_frequency_hz * (1.0 + (candidate.mod0_kp - 165) / 500.0)
        fake_gain_margin = config.targets.gain_margin_db + 4.0 - (kp_error + ki_error + kd_error) * 5.0
        metrics = ResponseAnalyzer(config.targets).analyze_hardware(
            waveform,
            {
                "phase_margin_deg": fake_phase_margin,
                "phase_crossover_hz": fake_crossover,
                "gain_margin_db": fake_gain_margin,
            },
        )
        return ExperimentResult(
            waveform=waveform,
            metrics=metrics,
            write_results={"placeholder": True},
            bode_result={"margins": {
                "phase_margin_deg": fake_phase_margin,
                "phase_crossover_hz": fake_crossover,
                "gain_margin_db": fake_gain_margin,
            }},
            scope_result={"placeholder": True},
        )

    def capture_response(self, candidate: TuningCandidate, config: TuningConfig) -> Waveform:
        target = config.targets.vout_target_v
        wc_mid = (config.search.wc_min_rad_s + config.search.wc_max_rad_s) / 2.0
        wc_norm = min(1.0, abs(candidate.wc_rad_s - wc_mid) / max(1.0, wc_mid))
        phi_norm = min(1.0, abs(candidate.phi_deg - 62.0) / 40.0)
        damping = max(0.08, 0.55 - phi_norm * 0.35)
        frequency = 35_000.0 + wc_norm * 65_000.0
        amplitude = target * (0.015 + wc_norm * 0.035 + phi_norm * 0.06)

        time_s: list[float] = []
        vout_v: list[float] = []
        for index in range(240):
            t = index * 2.0e-6
            envelope = math.exp(-t * frequency * damping)
            ripple = math.sin(2.0 * math.pi * frequency * t)
            load_step = -0.018 * target * math.exp(-max(0.0, t - 35e-6) * 22_000.0) if t >= 35e-6 else 0.0
            value = target + amplitude * envelope * ripple + load_step
            time_s.append(t)
            vout_v.append(value)
        return Waveform(time_s=time_s, vout_v=vout_v)


class PidAutotuneSession:
    def __init__(
        self,
        config: TuningConfig | None = None,
        pid_programmer: PidProgrammer | None = None,
        experiment_runner: AutotuneExperimentRunner | None = None,
    ):
        self._lock = threading.RLock()
        self._snapshot = TuningRunSnapshot(config=config or TuningConfig())
        self._pid_programmer = pid_programmer or StubPidProgrammer()
        self._experiment_runner = experiment_runner or PlaceholderExperimentRunner()
        self._tuner = HardwareGridHeuristicTuner(self._snapshot.config.search)
        self._stop_requested = False
        self._worker: threading.Thread | None = None

    def configure(self, config: TuningConfig, experiment: AutotuneExperimentConfig | None = None) -> dict:
        with self._lock:
            if self._snapshot.state == "running":
                raise RuntimeError("Cannot reconfigure while tuning is running.")
            self._snapshot = TuningRunSnapshot(
                state="idle",
                message="Configured",
                config=config,
                experiment=experiment or self._snapshot.experiment,
            )
            self._tuner = HardwareGridHeuristicTuner(config.search)
            return self.status()

    def update_context(self, config: TuningConfig | None = None, experiment: AutotuneExperimentConfig | None = None) -> None:
        """Update config/experiment without resetting history or search state."""

        if config is None and experiment is None:
            return
        if self._snapshot.state == "running":
            raise RuntimeError("Cannot update tuning context while tuning is running.")
        current = self._snapshot
        self._snapshot = TuningRunSnapshot(
            state=current.state,
            message=current.message,
            config=config or current.config,
            experiment=experiment or current.experiment,
            current=current.current,
            best=current.best,
            history=current.history,
            pid_programming=current.pid_programming,
        )

    def start(self, config: TuningConfig | None = None, experiment: AutotuneExperimentConfig | None = None) -> dict:
        with self._lock:
            if config is not None or experiment is not None:
                self.configure(config or self._snapshot.config, experiment)
            if self._snapshot.state == "running":
                return self.status()
            self._stop_requested = False
            self._snapshot.experiment = replace(self._snapshot.experiment, async_artifacts=True)
            self._snapshot.state = "running"
            self._snapshot.message = "Hardware auto-tune started."
            self._worker = threading.Thread(target=self._run_loop, name="pid-autotune", daemon=True)
            self._worker.start()
            return self.status()

    def pause(self) -> dict:
        with self._lock:
            self._stop_requested = True
            if self._snapshot.state == "running":
                self._snapshot.message = "Pause requested. Current hardware action will finish first."
            return self.status()

    def resume(self) -> dict:
        with self._lock:
            if self._snapshot.state == "running":
                return self.status()
            if self._snapshot.state not in {"paused", "stopped", "idle"}:
                raise RuntimeError(f"Cannot resume from state '{self._snapshot.state}'.")
            self._stop_requested = False
            self._snapshot.experiment = replace(self._snapshot.experiment, async_artifacts=True)
            self._snapshot.state = "running"
            self._snapshot.message = "Hardware auto-tune resumed."
            self._worker = threading.Thread(target=self._run_loop, name="pid-autotune", daemon=True)
            self._worker.start()
            return self.status()

    def restore(self, status: dict[str, Any]) -> dict:
        """Restore a saved run snapshot for continued hardware tuning."""

        with self._lock:
            if self._snapshot.state == "running":
                raise RuntimeError("Cannot restore a run while tuning is running.")
            snapshot = _snapshot_from_status(status)
            if snapshot.state == "complete":
                raise RuntimeError("This auto-tune run is already complete.")
            snapshot.state = "stopped"
            snapshot.message = "Loaded saved result. Ready to resume from the next candidate."
            self._snapshot = snapshot
            self._tuner = HardwareGridHeuristicTuner(snapshot.config.search)
            self._stop_requested = False
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            self._stop_requested = True
            if self._snapshot.state == "running":
                self._snapshot.state = "stopped"
                self._snapshot.message = "Stop requested."
            return self.status()

    def step(self, config: TuningConfig | None = None, experiment: AutotuneExperimentConfig | None = None) -> dict:
        with self._lock:
            if self._snapshot.state == "running":
                raise RuntimeError("Cannot step while background tuning is running.")
            if self._snapshot.history:
                self.update_context(config, experiment)
            elif config is not None or experiment is not None:
                self.configure(config or self._snapshot.config, experiment)
            self._snapshot.experiment = replace(self._snapshot.experiment, async_artifacts=False)
            self._snapshot.state = "running"
            self._snapshot.message = "Running one tuning iteration."
        self._run_one_iteration()
        with self._lock:
            if self._snapshot.state == "running":
                self._snapshot.state = "stopped"
                self._snapshot.message = "Single step complete."
            return self.status()

    def status(self) -> dict:
        with self._lock:
            self._snapshot.pid_programming = self._pid_programmer.status()
            return _compact_status(to_jsonable(self._snapshot))

    def _run_loop(self) -> None:
        while True:
            with self._lock:
                if self._stop_requested or self._snapshot.state != "running":
                    if self._stop_requested and self._snapshot.state == "running":
                        self._snapshot.state = "paused"
                        self._snapshot.message = "Paused."
                    return
            did_run = self._run_one_iteration()
            if not did_run:
                return
            with self._lock:
                if self._stop_requested and self._snapshot.state == "running":
                    self._snapshot.state = "paused"
                    self._snapshot.message = "Paused."
                    return
            time.sleep(0.2)

    def _run_one_iteration(self) -> bool:
        with self._lock:
            config = self._snapshot.config
            experiment = self._snapshot.experiment
            best = self._snapshot.best
            candidate = self._tuner.next_candidate(self._snapshot.history, best)
            if candidate is None:
                self._snapshot.state = "complete"
                self._snapshot.message = "Tuning search complete."
                return False

        try:
            started = time.perf_counter()
            result = self._experiment_runner.evaluate(candidate, config, experiment)
            pid = PidParameters(
                kp=float(candidate.mod0_kp),
                ki=float(candidate.mod0_ki),
                kd=float(candidate.mod0_kd),
                kf=float(candidate.mod0_kpole1),
            )
            with self._lock:
                record = IterationRecord(
                    iteration=len(self._snapshot.history) + 1,
                    phase=candidate.phase,
                    wc_rad_s=0.0,
                    phi_deg=0.0,
                    pid=pid,
                    metrics=result.metrics,
                    waveform=result.waveform,
                    timestamp=time.time(),
                    candidate=candidate,
                    write_results=result.write_results,
                    bode_result=result.bode_result,
                    scope_result=result.scope_result,
                    duration_s=result.duration_s or (time.perf_counter() - started),
                )
                self._snapshot.history.append(record)
                self._snapshot.current = record
                self._snapshot.best = select_best_result(self._snapshot.history)
                self._snapshot.message = f"Iteration {record.iteration} complete ({record.phase})."
                if record.metrics.passed:
                    self._snapshot.state = "complete"
                    self._snapshot.message = f"Auto-tune passed at iteration {record.iteration}."
                elif len(self._snapshot.history) >= config.search.max_iterations:
                    self._snapshot.state = "complete"
                    self._snapshot.message = "Reached max iterations."
            return True
        except Exception as exc:
            with self._lock:
                self._snapshot.state = "error"
                self._snapshot.message = str(exc)
            return False


def _compact_status(status: dict[str, Any]) -> dict[str, Any]:
    """Keep tuning status lightweight enough for frequent GUI polling."""

    for key in ("current", "best"):
        if isinstance(status.get(key), dict):
            status[key] = _compact_record(status[key])
    history = status.get("history")
    if isinstance(history, list):
        status["history"] = [_compact_record(record) if isinstance(record, dict) else record for record in history]
    return status


def _compact_record(record: dict[str, Any]) -> dict[str, Any]:
    compact = dict(record)
    compact["waveform"] = {"time_s": [], "vout_v": []}
    compact["bode_result"] = _compact_bode_result(compact.get("bode_result"))
    compact["scope_result"] = _compact_scope_result(compact.get("scope_result"))
    return compact


def _compact_bode_result(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    keep = {
        "ok",
        "error",
        "identity",
        "host",
        "port",
        "config",
        "sweep_id",
        "data_file",
        "data_file_pending",
        "original_points",
        "display_points",
        "bode_png",
        "bode_png_error",
        "bode_png_pending",
        "margins",
        "duration_s",
        "timestamp",
    }
    return {key: value.get(key) for key in keep if key in value}


def _compact_scope_result(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    keep = {
        "ok",
        "error",
        "resource",
        "identity",
        "channels",
        "measurements",
        "measurement_values",
        "capture_id",
        "scope_png",
        "scope_png_error",
        "scope_png_pending",
        "function_generator_frequency_hz",
        "scope_window_s",
        "scope_actual_window_s",
        "scope_scale_s_per_div",
        "scope_trigger_source",
        "scope_trigger_slope",
        "scope_trigger_offset_from_left_s",
        "scope_trigger_position_percent",
        "duration_s",
        "timestamp",
    }
    compact = {key: value.get(key) for key in keep if key in value}
    waveforms = value.get("waveforms")
    if isinstance(waveforms, list):
        compact["waveforms"] = [_compact_scope_waveform(item) for item in waveforms if isinstance(item, dict)]
    return compact


def _compact_scope_waveform(value: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "source",
        "x_unit",
        "y_unit",
        "time_span_s",
        "original_points",
        "plotted_points",
        "display_points",
        "display_strategy",
        "capture_id",
        "data_file",
        "data_file_pending",
        "transfer_encoding",
    }
    compact = {key: value.get(key) for key in keep if key in value}
    compact["x"] = []
    compact["y"] = []
    return compact


def _snapshot_from_status(status: dict[str, Any]) -> TuningRunSnapshot:
    history = [_record_from_payload(item) for item in _list_payload(status.get("history"))]
    current = _record_by_iteration(history, status.get("current")) or (history[-1] if history else None)
    best = _record_by_iteration(history, status.get("best")) or select_best_result(history)
    state = str(status.get("state") or "stopped")
    if state not in {"idle", "running", "paused", "stopped", "complete", "error"}:
        state = "stopped"
    return TuningRunSnapshot(
        state=state,  # type: ignore[arg-type]
        message=str(status.get("message") or "Loaded saved result."),
        config=_config_from_payload(status.get("config")),
        experiment=_experiment_from_payload(status.get("experiment")),
        current=current,
        best=best,
        history=history,
        pid_programming=_dict_payload(status.get("pid_programming")),
    )


def _config_from_payload(payload: Any) -> TuningConfig:
    payload = _dict_payload(payload)
    plant = _dict_payload(payload.get("plant"))
    targets = _dict_payload(payload.get("targets"))
    search = _dict_payload(payload.get("search"))
    default_search = SearchSpace()
    return TuningConfig(
        plant=PlantParams(
            vdc=_float_payload(plant, "vdc", 12.0),
            inductance_h=_float_payload(plant, "inductance_h", 30e-6),
            capacitance_f=_float_payload(plant, "capacitance_f", 15e-6),
            capacitor_esr_ohm=_float_payload(plant, "capacitor_esr_ohm", 7.5e-3),
            inductor_dcr_ohm=_float_payload(plant, "inductor_dcr_ohm", 50e-3),
        ),
        targets=TuningTargets(
            vout_target_v=_float_payload(targets, "vout_target_v", 0.9296875),
            overshoot_pct=_float_payload(targets, "overshoot_pct", 3.0),
            undershoot_pct=_float_payload(targets, "undershoot_pct", 3.0),
            settling_time_s=_float_payload(targets, "settling_time_s", 4e-6),
            oscillations=_int_payload(targets, "oscillations", 0),
            phase_margin_deg=_float_payload(targets, "phase_margin_deg", 45.0),
            crossover_frequency_hz=_float_payload(targets, "crossover_frequency_hz", 200_000.0),
            gain_margin_db=_float_payload(targets, "gain_margin_db", 6.0),
            phase_margin_tolerance_deg=_float_payload(targets, "phase_margin_tolerance_deg", 5.0),
            crossover_tolerance_pct=_float_payload(targets, "crossover_tolerance_pct", 20.0),
        ),
        search=SearchSpace(
            wc_min_rad_s=_float_payload(search, "wc_min_rad_s", default_search.wc_min_rad_s),
            wc_max_rad_s=_float_payload(search, "wc_max_rad_s", default_search.wc_max_rad_s),
            phi_min_deg=_float_payload(search, "phi_min_deg", default_search.phi_min_deg),
            phi_max_deg=_float_payload(search, "phi_max_deg", default_search.phi_max_deg),
            initial_wc_rad_s=_float_payload(search, "initial_wc_rad_s", default_search.initial_wc_rad_s),
            initial_phi_deg=_float_payload(search, "initial_phi_deg", default_search.initial_phi_deg),
            max_iterations=_int_payload(search, "max_iterations", default_search.max_iterations),
            mod0_kp=_search_parameter_from_payload(search.get("mod0_kp"), default_search.mod0_kp),
            mod0_ki=_search_parameter_from_payload(search.get("mod0_ki"), default_search.mod0_ki),
            mod0_kd=_search_parameter_from_payload(search.get("mod0_kd"), default_search.mod0_kd),
            mod0_kpole1=_search_parameter_from_payload(search.get("mod0_kpole1"), default_search.mod0_kpole1),
            mod0_kpole2=_search_parameter_from_payload(search.get("mod0_kpole2"), default_search.mod0_kpole2),
            output_inductance_nh=_search_parameter_from_payload(search.get("output_inductance_nh"), default_search.output_inductance_nh),
            effective_lc_inductance_nh=_search_parameter_from_payload(search.get("effective_lc_inductance_nh"), default_search.effective_lc_inductance_nh),
        ),
    )


def _experiment_from_payload(payload: Any) -> AutotuneExperimentConfig:
    payload = _dict_payload(payload)
    return AutotuneExperimentConfig(
        board_address=str(payload.get("board_address", "0x5E")),
        board_page=_int_payload(payload, "board_page", 0),
        board_adapter=str(payload.get("board_adapter", "xdp")),
        response_channel=str(payload.get("response_channel", "CH3")),
        enable_bode_analysis=bool(payload.get("enable_bode_analysis", True)),
        enable_transient_analysis=bool(payload.get("enable_transient_analysis", True)),
        bode_config=_dict_payload(payload.get("bode_config")),
        function_generator_config=_dict_payload(payload.get("function_generator_config")),
        scope_config=_dict_payload(payload.get("scope_config")),
        vout_tolerance_v=_float_payload(payload, "vout_tolerance_v", 0.15),
        response_abs_limit_v=_float_payload(payload, "response_abs_limit_v", 0.25),
        async_artifacts=bool(payload.get("async_artifacts", False)),
    )


def _record_from_payload(payload: Any) -> IterationRecord:
    payload = _dict_payload(payload)
    return IterationRecord(
        iteration=_int_payload(payload, "iteration", 0),
        phase=str(payload.get("phase") or "loaded"),
        wc_rad_s=_float_payload(payload, "wc_rad_s", 0.0),
        phi_deg=_float_payload(payload, "phi_deg", 0.0),
        pid=_pid_from_payload(payload.get("pid")),
        metrics=_metrics_from_payload(payload.get("metrics")),
        waveform=_waveform_from_payload(payload.get("waveform")),
        timestamp=_float_payload(payload, "timestamp", time.time()),
        candidate=_candidate_from_payload(payload.get("candidate")),
        write_results=_dict_payload(payload.get("write_results")),
        bode_result=_dict_payload(payload.get("bode_result")),
        scope_result=_dict_payload(payload.get("scope_result")),
        duration_s=_float_payload(payload, "duration_s", 0.0),
    )


def _pid_from_payload(payload: Any) -> PidParameters:
    payload = _dict_payload(payload)
    return PidParameters(
        kp=_float_payload(payload, "kp", 0.0),
        ki=_float_payload(payload, "ki", 0.0),
        kd=_float_payload(payload, "kd", 0.0),
        kf=_float_payload(payload, "kf", 0.0),
    )


def _metrics_from_payload(payload: Any) -> ResponseMetrics:
    payload = _dict_payload(payload)
    return ResponseMetrics(
        overshoot_pct=_float_payload(payload, "overshoot_pct", 0.0),
        undershoot_pct=_float_payload(payload, "undershoot_pct", 0.0),
        settling_time_s=_float_payload(payload, "settling_time_s", 0.0),
        oscillations=_int_payload(payload, "oscillations", 0),
        score=_float_payload(payload, "score", float("inf")),
        passed=bool(payload.get("passed", False)),
        overshoot_settling_time_s=_float_payload(payload, "overshoot_settling_time_s", 0.0),
        undershoot_settling_time_s=_float_payload(payload, "undershoot_settling_time_s", 0.0),
        low_load_steady_v=_optional_float_payload(payload.get("low_load_steady_v")),
        high_load_steady_v=_optional_float_payload(payload.get("high_load_steady_v")),
        phase_margin_deg=_optional_float_payload(payload.get("phase_margin_deg")),
        crossover_frequency_hz=_optional_float_payload(payload.get("crossover_frequency_hz")),
        gain_margin_db=_optional_float_payload(payload.get("gain_margin_db")),
        pass_reasons=[str(item) for item in _list_payload(payload.get("pass_reasons"))],
    )


def _waveform_from_payload(payload: Any) -> Waveform:
    payload = _dict_payload(payload)
    return Waveform(
        time_s=[float(item) for item in _list_payload(payload.get("time_s"))],
        vout_v=[float(item) for item in _list_payload(payload.get("vout_v"))],
        input_v=[float(item) for item in _list_payload(payload.get("input_v"))],
    )


def _candidate_from_payload(payload: Any) -> HardwarePidCandidate | None:
    if not isinstance(payload, dict):
        return None
    return HardwarePidCandidate(
        mod0_kp=_int_payload(payload, "mod0_kp", 165),
        mod0_ki=_int_payload(payload, "mod0_ki", 220),
        mod0_kd=_int_payload(payload, "mod0_kd", 175),
        mod0_kpole1=_int_payload(payload, "mod0_kpole1", 3),
        mod0_kpole2=_int_payload(payload, "mod0_kpole2", 3),
        output_inductance_nh=_float_payload(payload, "output_inductance_nh", 100.024),
        effective_lc_inductance_nh=_float_payload(payload, "effective_lc_inductance_nh", 369.276),
        phase=str(payload.get("phase") or "loaded"),
    )


def _record_by_iteration(history: list[IterationRecord], payload: Any) -> IterationRecord | None:
    if not isinstance(payload, dict):
        return None
    iteration = _int_payload(payload, "iteration", -1)
    return next((record for record in history if record.iteration == iteration), None)


def _search_parameter_from_payload(payload: Any, default: SearchParameter) -> SearchParameter:
    payload = _dict_payload(payload)
    return SearchParameter(
        center=_float_payload(payload, "center", default.center),
        min=_float_payload(payload, "min", default.min),
        max=_float_payload(payload, "max", default.max),
        step=_float_payload(payload, "step", default.step),
        points=_int_payload(payload, "points", default.points),
    )


def _dict_payload(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_payload(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _float_payload(payload: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(payload.get(key, default))
    except Exception:
        return default


def _int_payload(payload: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(payload.get(key, default))
    except Exception:
        return default


def _optional_float_payload(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None
