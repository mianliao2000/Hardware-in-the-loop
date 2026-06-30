"""Data models for the hardware PID autotuning workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class PidParameters:
    kp: float
    ki: float
    kd: float
    kf: float


@dataclass(frozen=True)
class PlantParams:
    vdc: float = 12.0
    inductance_h: float = 30e-6
    capacitance_f: float = 15e-6
    capacitor_esr_ohm: float = 7.5e-3
    inductor_dcr_ohm: float = 50e-3


@dataclass(frozen=True)
class TuningTargets:
    vout_target_v: float = 0.9
    overshoot_pct: float = 4.0
    undershoot_pct: float = 4.0
    settling_time_s: float = 100e-6
    oscillations: int = 0
    phase_margin_deg: float = 60.0
    crossover_frequency_hz: float = 100_000.0


@dataclass(frozen=True)
class SearchSpace:
    wc_min_rad_s: float = 94_248.0
    wc_max_rad_s: float = 314_159.0
    phi_min_deg: float = 30.0
    phi_max_deg: float = 80.0
    initial_wc_rad_s: float = 157_080.0
    initial_phi_deg: float = 60.0
    max_iterations: int = 40


@dataclass(frozen=True)
class TuningConfig:
    plant: PlantParams = field(default_factory=PlantParams)
    targets: TuningTargets = field(default_factory=TuningTargets)
    search: SearchSpace = field(default_factory=SearchSpace)


@dataclass(frozen=True)
class ResponseMetrics:
    overshoot_pct: float
    undershoot_pct: float
    settling_time_s: float
    oscillations: int
    score: float
    passed: bool


@dataclass(frozen=True)
class Waveform:
    time_s: list[float]
    vout_v: list[float]


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    phase: str
    wc_rad_s: float
    phi_deg: float
    pid: PidParameters
    metrics: ResponseMetrics
    waveform: Waveform
    timestamp: float


RunState = Literal["idle", "running", "stopped", "complete", "error"]


@dataclass
class TuningRunSnapshot:
    state: RunState = "idle"
    message: str = "Ready"
    config: TuningConfig = field(default_factory=TuningConfig)
    current: IterationRecord | None = None
    best: IterationRecord | None = None
    history: list[IterationRecord] = field(default_factory=list)
    pid_programming: dict[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    """Convert nested tuning dataclasses into JSON-safe primitives."""

    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value
