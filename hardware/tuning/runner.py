"""Session runner for the PID autotuning framework."""

from __future__ import annotations

import math
import threading
import time

from .analyzer import ResponseAnalyzer
from .compensator import CompensatorDesign
from .models import (
    IterationRecord,
    TuningConfig,
    TuningRunSnapshot,
    Waveform,
    to_jsonable,
)
from .pid_programmer import PidProgrammer, StubPidProgrammer
from .search import GridRefinePidTuner, TuningCandidate, select_best_result


class PlaceholderExperimentRunner:
    """Deterministic fake experiment used before hardware PID writes are safe."""

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
        experiment_runner: PlaceholderExperimentRunner | None = None,
    ):
        self._lock = threading.RLock()
        self._snapshot = TuningRunSnapshot(config=config or TuningConfig())
        self._pid_programmer = pid_programmer or StubPidProgrammer()
        self._experiment_runner = experiment_runner or PlaceholderExperimentRunner()
        self._tuner = GridRefinePidTuner(self._snapshot.config.search)
        self._stop_requested = False
        self._worker: threading.Thread | None = None

    def configure(self, config: TuningConfig) -> dict:
        with self._lock:
            if self._snapshot.state == "running":
                raise RuntimeError("Cannot reconfigure while tuning is running.")
            self._snapshot = TuningRunSnapshot(state="idle", message="Configured", config=config)
            self._tuner = GridRefinePidTuner(config.search)
            return self.status()

    def start(self, config: TuningConfig | None = None) -> dict:
        with self._lock:
            if config is not None:
                self.configure(config)
            if self._snapshot.state == "running":
                return self.status()
            self._stop_requested = False
            self._snapshot.state = "running"
            self._snapshot.message = "Tuning started in stub experiment mode."
            self._worker = threading.Thread(target=self._run_loop, name="pid-autotune", daemon=True)
            self._worker.start()
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            self._stop_requested = True
            if self._snapshot.state == "running":
                self._snapshot.state = "stopped"
                self._snapshot.message = "Stop requested."
            return self.status()

    def step(self, config: TuningConfig | None = None) -> dict:
        with self._lock:
            if config is not None:
                self.configure(config)
            if self._snapshot.state == "running":
                raise RuntimeError("Cannot step while background tuning is running.")
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
            return to_jsonable(self._snapshot)

    def _run_loop(self) -> None:
        while True:
            with self._lock:
                if self._stop_requested or self._snapshot.state != "running":
                    return
            did_run = self._run_one_iteration()
            if not did_run:
                return
            time.sleep(0.2)

    def _run_one_iteration(self) -> bool:
        with self._lock:
            config = self._snapshot.config
            best = self._snapshot.best
            candidate = self._tuner.next_candidate(self._snapshot.history, best)
            if candidate is None:
                self._snapshot.state = "complete"
                self._snapshot.message = "Tuning search complete."
                return False

        try:
            pid = CompensatorDesign(config.plant).compute(candidate.wc_rad_s, candidate.phi_deg)
            waveform = self._experiment_runner.capture_response(candidate, config)
            metrics = ResponseAnalyzer(config.targets).analyze(waveform)
            with self._lock:
                record = IterationRecord(
                    iteration=len(self._snapshot.history) + 1,
                    phase=candidate.phase,
                    wc_rad_s=candidate.wc_rad_s,
                    phi_deg=candidate.phi_deg,
                    pid=pid,
                    metrics=metrics,
                    waveform=waveform,
                    timestamp=time.time(),
                )
                self._snapshot.history.append(record)
                self._snapshot.current = record
                self._snapshot.best = select_best_result(self._snapshot.history)
                self._snapshot.message = f"Iteration {record.iteration} complete ({record.phase})."
                if len(self._snapshot.history) >= config.search.max_iterations:
                    self._snapshot.state = "complete"
                    self._snapshot.message = "Reached max iterations."
            return True
        except Exception as exc:
            with self._lock:
                self._snapshot.state = "error"
                self._snapshot.message = str(exc)
            return False
