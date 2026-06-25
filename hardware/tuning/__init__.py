"""PID autotuning framework for the hardware test bench."""

from .analyzer import ResponseAnalyzer, score_metrics
from .compensator import CompensatorDesign
from .models import (
    IterationRecord,
    PidParameters,
    PlantParams,
    ResponseMetrics,
    SearchSpace,
    TuningConfig,
    TuningRunSnapshot,
    TuningTargets,
    Waveform,
)
from .pid_programmer import PidProgrammer, StubPidProgrammer
from .runner import PidAutotuneSession, PlaceholderExperimentRunner
from .search import GridRefinePidTuner, TuningCandidate, select_best_result

__all__ = [
    "CompensatorDesign",
    "GridRefinePidTuner",
    "IterationRecord",
    "PidAutotuneSession",
    "PidParameters",
    "PidProgrammer",
    "PlaceholderExperimentRunner",
    "PlantParams",
    "ResponseAnalyzer",
    "ResponseMetrics",
    "SearchSpace",
    "StubPidProgrammer",
    "TuningCandidate",
    "TuningConfig",
    "TuningRunSnapshot",
    "TuningTargets",
    "Waveform",
    "score_metrics",
    "select_best_result",
]
