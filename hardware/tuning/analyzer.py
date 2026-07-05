"""Response analysis helpers for PID tuning iterations."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ResponseMetrics, TuningTargets, Waveform


@dataclass(frozen=True)
class _StepEvent:
    index: int
    edge: str


@dataclass(frozen=True)
class _DynamicStepMetrics:
    overshoot_pct: float
    undershoot_pct: float
    settling_time_s: float
    overshoot_settling_time_s: float
    undershoot_settling_time_s: float
    oscillations: int
    low_load_steady_v: float | None
    high_load_steady_v: float | None


class ResponseAnalyzer:
    def __init__(self, targets: TuningTargets):
        self.targets = targets

    def analyze(self, waveform: Waveform) -> ResponseMetrics:
        if not waveform.time_s or not waveform.vout_v or len(waveform.time_s) != len(waveform.vout_v):
            raise ValueError("Waveform must contain matching time and Vout arrays.")

        target = self.targets.vout_target_v
        if target <= 0:
            raise ValueError("Vout target must be positive.")

        dynamic_metrics = self._dynamic_step_metrics(waveform)
        if dynamic_metrics is None:
            max_v = max(waveform.vout_v)
            min_v = min(waveform.vout_v)
            overshoot = max(0.0, (max_v - target) / target * 100.0)
            undershoot = max(0.0, (target - min_v) / target * 100.0)
            settling_time = self._settling_time(waveform, target)
            overshoot_settling_time = settling_time
            undershoot_settling_time = settling_time
            oscillations = self._oscillation_count(waveform, target)
            low_load_steady_v = None
            high_load_steady_v = None
        else:
            overshoot = dynamic_metrics.overshoot_pct
            undershoot = dynamic_metrics.undershoot_pct
            settling_time = dynamic_metrics.settling_time_s
            overshoot_settling_time = dynamic_metrics.overshoot_settling_time_s
            undershoot_settling_time = dynamic_metrics.undershoot_settling_time_s
            oscillations = dynamic_metrics.oscillations
            low_load_steady_v = dynamic_metrics.low_load_steady_v
            high_load_steady_v = dynamic_metrics.high_load_steady_v
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
            overshoot_settling_time_s=overshoot_settling_time,
            undershoot_settling_time_s=undershoot_settling_time,
            low_load_steady_v=low_load_steady_v,
            high_load_steady_v=high_load_steady_v,
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

    def _dynamic_step_metrics(self, waveform: Waveform) -> _DynamicStepMetrics | None:
        if not waveform.input_v or len(waveform.input_v) != len(waveform.time_s):
            return None
        events = self._input_edges(waveform.input_v)
        if not events:
            return None

        overshoot_pct = 0.0
        undershoot_pct = 0.0
        settling_time_s = 0.0
        overshoot_settling_time_s = 0.0
        undershoot_settling_time_s = 0.0
        oscillations = 0
        low_load_steady_values, high_load_steady_values = self._load_steady_values(waveform, events)
        low_load_steady_v = _mean(low_load_steady_values)
        high_load_steady_v = _mean(high_load_steady_values)
        overshoot_pct, undershoot_pct = self._load_step_extremes(waveform, events, low_load_steady_v, high_load_steady_v)
        for event_index, event in enumerate(events):
            next_index = events[event_index + 1].index if event_index + 1 < len(events) else len(waveform.time_s)
            if next_index <= event.index:
                continue

            segment = waveform.vout_v[event.index:next_index]
            if not segment:
                continue
            final_value = self._post_step_steady_value(waveform, event.index, next_index)
            event_settling_time = self._dynamic_settling_time(waveform, event.index, next_index, final_value)
            settling_time_s = max(settling_time_s, event_settling_time)
            if event.edge == "falling":
                overshoot_settling_time_s = max(overshoot_settling_time_s, event_settling_time)
            elif event.edge == "rising":
                undershoot_settling_time_s = max(undershoot_settling_time_s, event_settling_time)
            oscillations = max(oscillations, self._dynamic_oscillation_count(waveform, event.index, next_index, final_value))

        return _DynamicStepMetrics(
            overshoot_pct=overshoot_pct,
            undershoot_pct=undershoot_pct,
            settling_time_s=settling_time_s,
            overshoot_settling_time_s=overshoot_settling_time_s,
            undershoot_settling_time_s=undershoot_settling_time_s,
            oscillations=oscillations,
            low_load_steady_v=low_load_steady_v,
            high_load_steady_v=high_load_steady_v,
        )

    def _load_step_extremes(
        self,
        waveform: Waveform,
        events: list[_StepEvent],
        low_load_steady_v: float | None,
        high_load_steady_v: float | None,
    ) -> tuple[float, float]:
        rising_events = [event for event in events if event.edge == "rising"]
        falling_events = [event for event in events if event.edge == "falling"]
        overshoot_pct = 0.0
        undershoot_pct = 0.0

        if rising_events and high_load_steady_v is not None:
            first_rising = rising_events[0]
            next_falling = next((event for event in falling_events if event.index > first_rising.index), None)
            end_index = next_falling.index if next_falling is not None else len(waveform.vout_v)
            if end_index > first_rising.index:
                high_load_segment = waveform.vout_v[first_rising.index:end_index]
                if high_load_segment:
                    minimum = min(high_load_segment)
                    denominator = max(abs(high_load_steady_v), 1e-9)
                    undershoot_pct = max(0.0, (high_load_steady_v - minimum) / denominator * 100.0)

        if falling_events and low_load_steady_v is not None:
            first_falling = falling_events[0]
            next_rising = next((event for event in rising_events if event.index > first_falling.index), None)
            end_index = next_rising.index if next_rising is not None else len(waveform.vout_v)
            if end_index > first_falling.index:
                low_load_segment = waveform.vout_v[first_falling.index:end_index]
                if low_load_segment:
                    maximum = max(low_load_segment)
                    denominator = max(abs(low_load_steady_v), 1e-9)
                    overshoot_pct = max(0.0, (maximum - low_load_steady_v) / denominator * 100.0)

        return overshoot_pct, undershoot_pct

    def _load_steady_values(self, waveform: Waveform, events: list[_StepEvent]) -> tuple[list[float], list[float]]:
        rising_events = [event for event in events if event.edge == "rising"]
        falling_events = [event for event in events if event.edge == "falling"]
        low_load_values: list[float] = []
        high_load_values: list[float] = []

        # With a full-period capture plus a little margin, CH1 can contain two
        # rising edges. The first one is near the left trigger boundary, so use
        # the last rising-edge pre-window as the representative light-load value.
        selected_rising = rising_events[-1:] if len(rising_events) > 1 else rising_events
        for event in selected_rising:
            baseline = self._mean_before(waveform, event.index, window_s=10e-6)
            if baseline is not None:
                low_load_values.append(baseline)

        # In the normal one-period window there is one falling edge; its
        # pre-window is the representative heavy-load steady-state value.
        for event in falling_events:
            baseline = self._mean_before(waveform, event.index, window_s=10e-6)
            if baseline is not None:
                high_load_values.append(baseline)

        # Fallback for partial captures that only include a single edge.
        if not high_load_values and len(rising_events) == 1:
            event = rising_events[0]
            final_value = self._post_step_steady_value(waveform, event.index, len(waveform.time_s))
            high_load_values.append(final_value)
        if not low_load_values and len(falling_events) == 1:
            event = falling_events[0]
            final_value = self._post_step_steady_value(waveform, event.index, len(waveform.time_s))
            low_load_values.append(final_value)

        return low_load_values, high_load_values

    def _input_edges(self, input_v: list[float]) -> list[_StepEvent]:
        if len(input_v) < 2:
            return []
        low = min(input_v)
        high = max(input_v)
        span = high - low
        if span <= 1e-9:
            return []
        threshold = low + span * 0.5
        events: list[_StepEvent] = []
        for index in range(1, len(input_v)):
            previous = input_v[index - 1]
            current = input_v[index]
            if previous <= threshold < current:
                events.append(_StepEvent(index=index, edge="rising"))
            elif previous >= threshold > current:
                events.append(_StepEvent(index=index, edge="falling"))
        events = self._debounce_edges(events, len(input_v))
        return events

    def _debounce_edges(self, events: list[_StepEvent], sample_count: int) -> list[_StepEvent]:
        if len(events) < 2:
            return events
        # CH1 can ring around the threshold at an edge. Treat crossings inside a
        # tiny part of the record as the same physical transition.
        min_separation = max(3, int(round(sample_count * 0.002)))
        debounced: list[_StepEvent] = []
        for event in events:
            if debounced and event.index - debounced[-1].index <= min_separation:
                continue
            debounced.append(event)
        return debounced

    def _mean_before(self, waveform: Waveform, index: int, window_s: float) -> float | None:
        edge_time = waveform.time_s[index]
        start_time = edge_time - window_s
        values = [
            value
            for time_s, value in zip(waveform.time_s[:index], waveform.vout_v[:index])
            if start_time <= time_s < edge_time
        ]
        if values:
            return sum(values) / len(values)
        if index > 0:
            return waveform.vout_v[index - 1]
        return None

    def _post_step_steady_value(self, waveform: Waveform, start_index: int, end_index: int) -> float:
        segment_time = waveform.time_s[end_index - 1] - waveform.time_s[start_index]
        window_s = max(0.0, min(10e-6, segment_time * 0.25))
        end_time = waveform.time_s[end_index - 1]
        start_time = end_time - window_s
        values = [
            value
            for time_s, value in zip(waveform.time_s[start_index:end_index], waveform.vout_v[start_index:end_index])
            if time_s >= start_time
        ]
        if not values:
            values = waveform.vout_v[max(start_index, end_index - 5):end_index]
        return sum(values) / len(values)

    def _dynamic_settling_time(self, waveform: Waveform, start_index: int, end_index: int, final_value: float) -> float:
        tolerance = max(abs(final_value) * 0.02, 1e-6)
        last_outside = start_index
        for index in range(start_index, end_index):
            if abs(waveform.vout_v[index] - final_value) > tolerance:
                last_outside = index
        return max(0.0, waveform.time_s[last_outside] - waveform.time_s[start_index])

    def _dynamic_oscillation_count(self, waveform: Waveform, start_index: int, end_index: int, final_value: float) -> int:
        tolerance = max(abs(final_value) * 0.01, 1e-6)
        signs: list[int] = []
        for value in waveform.vout_v[start_index:end_index]:
            delta = value - final_value
            if abs(delta) <= tolerance:
                continue
            sign = 1 if delta > 0 else -1
            if not signs or signs[-1] != sign:
                signs.append(sign)
        return max(0, len(signs) - 1)

    def analyze_hardware(
        self,
        waveform: Waveform | None,
        bode_margins: dict | None,
        enable_transient: bool = True,
        enable_bode: bool = True,
    ) -> ResponseMetrics:
        if not enable_transient and not enable_bode:
            raise ValueError("At least one analysis mode must be enabled.")

        transient = self.analyze(waveform) if enable_transient and waveform is not None else None
        margins = bode_margins or {}
        phase_margin = _optional_float(margins.get("phase_margin_deg"))
        crossover = _optional_float(margins.get("phase_crossover_hz"))
        gain_margin = _optional_float(margins.get("gain_margin_db"))

        reasons: list[str] = []
        score = transient.score if transient is not None else 0.0
        phase_error: float | None = None
        crossover_error_pct: float | None = None

        if enable_bode:
            if phase_margin is None:
                score += 100.0
                reasons.append("missing phase margin")
                phase_ok = False
            else:
                phase_error = max(0.0, self.targets.phase_margin_deg - phase_margin)
                phase_ok = phase_margin >= self.targets.phase_margin_deg
                score += phase_error * 1.5
                if not phase_ok:
                    reasons.append(f"phase margin below target by {phase_error:.2f} deg")

            if crossover is None or crossover <= 0 or self.targets.crossover_frequency_hz <= 0:
                score += 100.0
                reasons.append("missing crossover frequency")
                crossover_ok = False
            else:
                crossover_error_pct = abs(crossover - self.targets.crossover_frequency_hz) / self.targets.crossover_frequency_hz * 100.0
                crossover_ok = crossover_error_pct <= self.targets.crossover_tolerance_pct
                score += max(0.0, crossover_error_pct - self.targets.crossover_tolerance_pct) * 0.5
                if not crossover_ok:
                    reasons.append(f"crossover error {crossover_error_pct:.1f}%")

            if gain_margin is None:
                score += 80.0
                reasons.append("missing gain margin")
                gain_ok = False
            else:
                gain_ok = gain_margin >= self.targets.gain_margin_db
                score += max(0.0, self.targets.gain_margin_db - gain_margin) * 2.0
                if not gain_ok:
                    reasons.append(f"gain margin {gain_margin:.2f} dB")
        else:
            phase_ok = True
            crossover_ok = True
            gain_ok = True

        transient_ok = transient.passed if transient is not None else True
        if enable_transient and not transient_ok:
            reasons.append("transient limits not met")
        passed = transient_ok and phase_ok and crossover_ok and gain_ok
        if passed:
            reward = _passed_reward(
                targets=self.targets,
                transient=transient,
                phase_error=phase_error,
                crossover_error_pct=crossover_error_pct,
                gain_margin=gain_margin,
                enable_transient=enable_transient,
                enable_bode=enable_bode,
            )
            if reward > 0:
                score = max(-3.0, score - reward)
                reasons.append(f"passed reward {reward:.3f}")
            reasons.append("passed")

        return ResponseMetrics(
            overshoot_pct=transient.overshoot_pct if transient is not None else 0.0,
            undershoot_pct=transient.undershoot_pct if transient is not None else 0.0,
            settling_time_s=transient.settling_time_s if transient is not None else 0.0,
            oscillations=transient.oscillations if transient is not None else 0,
            score=score,
            passed=passed,
            overshoot_settling_time_s=transient.overshoot_settling_time_s if transient is not None else 0.0,
            undershoot_settling_time_s=transient.undershoot_settling_time_s if transient is not None else 0.0,
            low_load_steady_v=transient.low_load_steady_v if transient is not None else None,
            high_load_steady_v=transient.high_load_steady_v if transient is not None else None,
            phase_margin_deg=phase_margin,
            crossover_frequency_hz=crossover,
            gain_margin_db=gain_margin,
            pass_reasons=reasons,
        )


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


def _passed_reward(
    targets: TuningTargets,
    transient: ResponseMetrics | None,
    phase_error: float | None,
    crossover_error_pct: float | None,
    gain_margin: float | None,
    enable_transient: bool,
    enable_bode: bool,
) -> float:
    """Small tie-break reward applied only after every enabled target passes."""

    reward = 0.0
    if enable_transient and transient is not None:
        reward += 0.15 * _headroom(targets.overshoot_pct, transient.overshoot_pct)
        reward += 0.15 * _headroom(targets.undershoot_pct, transient.undershoot_pct)
        reward += 0.25 * _headroom(targets.settling_time_s, transient.settling_time_s)
        if targets.oscillations <= 0 and transient.oscillations == 0:
            reward += 0.10
        elif targets.oscillations > 0:
            reward += 0.10 * _headroom(float(targets.oscillations), float(transient.oscillations))

    if enable_bode:
        if phase_error is not None:
            reward += min(max(targets.phase_margin_tolerance_deg - phase_error, 0.0), targets.phase_margin_tolerance_deg) * 0.05
        if crossover_error_pct is not None:
            reward += 0.25 * _headroom(targets.crossover_tolerance_pct, crossover_error_pct)
        if gain_margin is not None:
            reward += min(max(gain_margin - targets.gain_margin_db, 0.0), 12.0) * 0.04

    return min(reward, 1.5)


def _headroom(limit: float, value: float) -> float:
    if limit <= 0:
        return 0.0
    return max(0.0, min(1.0, (limit - value) / limit))


def _optional_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
