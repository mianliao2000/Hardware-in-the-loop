"""Data models for the hardware PID autotuning workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Literal


PID_FIELD_NAMES = ("mod0_kp", "mod0_ki", "mod0_kd", "mod0_kpole1", "mod0_kpole2")
HARDWARE_TUNING_FIELD_NAMES = (
    "mod0_kp",
    "mod0_ki",
    "mod0_kd",
    "mod0_kpole1",
    "mod0_kpole2",
    "mod0_cm_gain",
    "output_inductance_nh",
    "effective_lc_inductance_nh",
)


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
    vout_target_v: float = 0.9296875
    overshoot_pct: float = 3.0
    undershoot_pct: float = 3.0
    settling_time_s: float = 2e-6
    oscillations: int = 0
    phase_margin_deg: float = 45.0
    crossover_frequency_hz: float = 200_000.0
    gain_margin_db: float = 6.0
    phase_margin_tolerance_deg: float = 5.0
    crossover_tolerance_pct: float = 20.0


@dataclass(frozen=True)
class SearchParameter:
    center: float
    min: float
    max: float
    step: float
    points: int = 7

    def clamped(self, value: float) -> float:
        return min(max(value, self.min), self.max)


@dataclass(frozen=True)
class HardwarePidCandidate:
    mod0_kp: int = 165
    mod0_ki: int = 220
    mod0_kd: int = 175
    mod0_kpole1: int = 3
    mod0_kpole2: int = 3
    mod0_cm_gain: int = 2
    output_inductance_nh: float = 100.024
    effective_lc_inductance_nh: float = 369.276
    phase: str = "baseline"

    def pid_values(self) -> dict[str, int]:
        return {
            "mod0_kp": int(self.mod0_kp),
            "mod0_ki": int(self.mod0_ki),
            "mod0_kd": int(self.mod0_kd),
            "mod0_kpole1": int(self.mod0_kpole1),
            "mod0_kpole2": int(self.mod0_kpole2),
        }

    def current_mode_values(self) -> dict[str, int]:
        return {
            "mod0_cm_gain": int(self.mod0_cm_gain),
        }


@dataclass(frozen=True)
class SearchSpace:
    wc_min_rad_s: float = 94_248.0
    wc_max_rad_s: float = 314_159.0
    phi_min_deg: float = 30.0
    phi_max_deg: float = 80.0
    initial_wc_rad_s: float = 157_080.0
    initial_phi_deg: float = 60.0
    max_iterations: int = 40
    max_coarse_iterations: int = 20
    max_refined_iterations: int = 20
    mod0_kp: SearchParameter = field(default_factory=lambda: SearchParameter(165, 100, 255, 19.375, 9))
    mod0_ki: SearchParameter = field(default_factory=lambda: SearchParameter(220, 150, 255, 13.125, 9))
    mod0_kd: SearchParameter = field(default_factory=lambda: SearchParameter(175, 100, 200, 12.5, 9))
    mod0_kpole1: SearchParameter = field(default_factory=lambda: SearchParameter(3, 3, 6, 3, 2))
    mod0_kpole2: SearchParameter = field(default_factory=lambda: SearchParameter(3, 3, 6, 3, 2))
    mod0_cm_gain: SearchParameter = field(default_factory=lambda: SearchParameter(2, 2, 2, 1, 1))
    output_inductance_nh: SearchParameter = field(default_factory=lambda: SearchParameter(100.024, 80.019, 120.029, 10.0025, 5))
    effective_lc_inductance_nh: SearchParameter = field(default_factory=lambda: SearchParameter(369.276, 295.421, 443.131, 36.9275, 5))

    def total_iteration_budget(self) -> int:
        return max(1, int(self.max_coarse_iterations) + int(self.max_refined_iterations))

    def coarse_iteration_budget(self) -> int:
        return max(1, int(self.max_coarse_iterations))

    def refined_iteration_budget(self) -> int:
        return max(0, int(self.max_refined_iterations))


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
    overshoot_settling_time_s: float = 0.0
    undershoot_settling_time_s: float = 0.0
    low_load_steady_v: float | None = None
    high_load_steady_v: float | None = None
    phase_margin_deg: float | None = None
    crossover_frequency_hz: float | None = None
    gain_margin_db: float | None = None
    pass_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Waveform:
    time_s: list[float]
    vout_v: list[float]
    input_v: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class AutotuneExperimentConfig:
    board_address: str = "0x5E"
    board_page: int = 0
    board_adapter: str = "xdp"
    response_channel: str = "CH3"
    enable_bode_analysis: bool = True
    enable_transient_analysis: bool = True
    optimization_algorithm: str = "heuristic"
    bode_config: dict[str, Any] = field(default_factory=dict)
    function_generator_config: dict[str, Any] = field(default_factory=dict)
    scope_config: dict[str, Any] = field(default_factory=dict)
    vout_tolerance_v: float = 0.15
    response_abs_limit_v: float = 0.25
    async_artifacts: bool = False
    ignore_pass_until_max_iterations: bool = True
    drl_workflow_mode: str = ""
    drl_model_id: str = ""
    drl_collection_plan_id: str = ""
    drl_episode_budget: int = 15
    drl_confirmation_count: int = 3


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
    candidate: HardwarePidCandidate | None = None
    write_results: dict[str, Any] = field(default_factory=dict)
    bode_result: dict[str, Any] = field(default_factory=dict)
    scope_result: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0
    optimizer_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    waveform: Waveform
    metrics: ResponseMetrics
    write_results: dict[str, Any] = field(default_factory=dict)
    bode_result: dict[str, Any] = field(default_factory=dict)
    scope_result: dict[str, Any] = field(default_factory=dict)
    duration_s: float = 0.0


RunState = Literal["idle", "running", "paused", "stopped", "complete", "error"]


@dataclass
class TuningRunSnapshot:
    state: RunState = "idle"
    message: str = "Ready"
    config: TuningConfig = field(default_factory=TuningConfig)
    experiment: AutotuneExperimentConfig = field(default_factory=AutotuneExperimentConfig)
    current: IterationRecord | None = None
    best: IterationRecord | None = None
    history: list[IterationRecord] = field(default_factory=list)
    pid_programming: dict[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    """Convert nested tuning dataclasses into JSON-safe primitives."""

    if is_dataclass(value):
        return {item.name: to_jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    return value
