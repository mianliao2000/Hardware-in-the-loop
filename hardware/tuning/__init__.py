"""PID autotuning framework for the hardware test bench."""

from .analyzer import ResponseAnalyzer, score_metrics
from .compensator import CompensatorDesign
from .models import (
    AutotuneExperimentConfig,
    ExperimentResult,
    HARDWARE_TUNING_FIELD_NAMES,
    HardwarePidCandidate,
    IterationRecord,
    PID_FIELD_NAMES,
    PidParameters,
    PlantParams,
    ResponseMetrics,
    SearchSpace,
    SearchParameter,
    TuningConfig,
    TuningRunSnapshot,
    TuningTargets,
    Waveform,
)
from .pid_programmer import PidProgrammer, StubPidProgrammer
from .runner import AutotuneExperimentRunner, PidAutotuneSession, PlaceholderExperimentRunner
from .search import GridRefinePidTuner, HardwareGridHeuristicTuner, TuningCandidate, select_best_result

__all__ = [
    "AutotuneExperimentConfig",
    "AutotuneExperimentRunner",
    "CompensatorDesign",
    "ExperimentResult",
    "GridRefinePidTuner",
    "HARDWARE_TUNING_FIELD_NAMES",
    "HardwareGridHeuristicTuner",
    "HardwarePidCandidate",
    "IterationRecord",
    "PID_FIELD_NAMES",
    "PidAutotuneSession",
    "PidParameters",
    "PidProgrammer",
    "PlaceholderExperimentRunner",
    "PlantParams",
    "ResponseAnalyzer",
    "ResponseMetrics",
    "SearchSpace",
    "SearchParameter",
    "StubPidProgrammer",
    "TuningCandidate",
    "TuningConfig",
    "TuningRunSnapshot",
    "TuningTargets",
    "Waveform",
    "score_metrics",
    "select_best_result",
]
