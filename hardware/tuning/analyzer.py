"""Response analysis helpers for PID tuning iterations."""

from __future__ import annotations

from .models import ResponseMetrics, TuningTargets, Waveform


class ResponseAnalyzer:
    def __init__(self, targets: TuningTargets):
        self.targets = targets

    def analyze(self, waveform: Waveform) -> ResponseMetrics:
        if not waveform.time_s or not waveform.vout_v or len(waveform.time_s) != len(waveform.vout_v):
            raise ValueError("Waveform must contain matching time and Vout arrays.")

        target = self.targets.vout_target_v
        if target <= 0:
            raise ValueError("Vout target must be positive.")

        max_v = max(waveform.vout_v)
        min_v = min(waveform.vout_v)
        overshoot = max(0.0, (max_v - target) / target * 100.0)
        undershoot = max(0.0, (target - min_v) / target * 100.0)
        settling_time = self._settling_time(waveform, target)
        oscillations = self._oscillation_count(waveform, target)
        score = score_metrics(overshoot, undershoot, oscillations, settling_time, self.targets)
        passed = (
            overshoot <= self.targets.overshoot_pct
            and undershoot <= self.targets.undershoot_pct
            and oscillations <= self.targets.oscillations
            and settling_time <= self.targets.settling_time_s
        )
        return ResponseMetrics(
            overshoot_pct=overshoot,
            undershoot_pct=undershoot,
            settling_time_s=settling_time,
            oscillations=oscillations,
            score=score,
            passed=passed,
        )

    def _settling_time(self, waveform: Waveform, target: float) -> float:
        tolerance = target * 0.02
        last_outside = 0
        for index, value in enumerate(waveform.vout_v):
            if abs(value - target) > tolerance:
                last_outside = index
        return waveform.time_s[min(last_outside, len(waveform.time_s) - 1)]

    def _oscillation_count(self, waveform: Waveform, target: float) -> int:
        tolerance = target * 0.01
        signs: list[int] = []
        for value in waveform.vout_v:
            delta = value - target
            if abs(delta) <= tolerance:
                continue
            sign = 1 if delta > 0 else -1
            if not signs or signs[-1] != sign:
                signs.append(sign)
        return max(0, len(signs) - 1)


def score_metrics(
    overshoot_pct: float,
    undershoot_pct: float,
    oscillations: int,
    settling_time_s: float,
    targets: TuningTargets,
) -> float:
    excess_os = max(0.0, overshoot_pct - targets.overshoot_pct)
    excess_us = max(0.0, undershoot_pct - targets.undershoot_pct)
    excess_osc = max(0, oscillations - targets.oscillations)
    excess_ts = max(0.0, settling_time_s - targets.settling_time_s)
    return excess_os + excess_us + excess_osc * 3.0 + excess_ts * 10_000.0
