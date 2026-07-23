"""Response analysis helpers for PID tuning iterations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .models import ResponseMetrics, TuningTargets, Waveform


SETTLING_WEIGHT_PER_US = 10.0
MAX_PENALTY = 300.0
INVALID_TRANSIENT_PENALTY = MAX_PENALTY
SUSTAINED_OSCILLATION_MIN_SIGN_CHANGES = 8
SETTLING_ANALYSIS_VERSION = 19


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
    overshoot_settling_valid: bool
    undershoot_settling_valid: bool
    settling_diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _SettlingResult:
    time_s: float
    valid: bool
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _EnvelopeBin:
    start_s: float
    stop_s: float
    q05: float
    q10: float
    q25: float
    q50: float
    q75: float
    q90: float
    q95: float


@dataclass(frozen=True)
class _TrendBin:
    start_s: float
    stop_s: float
    center: float


class ResponseAnalyzer:
    RESPONSE_LOWPASS_CUTOFF_HZ = 600_000.0

    def __init__(self, targets: TuningTargets):
        self.targets = targets

    def analyze(self, waveform: Waveform) -> ResponseMetrics:
        if not waveform.time_s or not waveform.vout_v or len(waveform.time_s) != len(waveform.vout_v):
            raise ValueError("Waveform must contain matching time and Vout arrays.")

        target = self.targets.vout_target_v
        if target <= 0:
            raise ValueError("Vout target must be positive.")

        analysis_waveform = self._lowpass_response_waveform(waveform)
        dynamic_metrics = self._dynamic_step_metrics(analysis_waveform)
        if dynamic_metrics is None:
            # Without a debounced CH1 edge this remains the legacy whole-trace
            # fallback.  Do not label it as V2, otherwise it could leak into
            # the V2-only DRL training set despite never running the final-entry
            # step analysis below.
            settling_analysis_version = 1
            max_v = max(analysis_waveform.vout_v)
            min_v = min(analysis_waveform.vout_v)
            overshoot = max(0.0, (max_v - target) / target * 100.0)
            undershoot = max(0.0, (target - min_v) / target * 100.0)
            settling_time = self._settling_time(analysis_waveform, target)
            overshoot_settling_time = settling_time
            undershoot_settling_time = settling_time
            oscillations = self._oscillation_count(analysis_waveform, target)
            low_load_steady_v = None
            high_load_steady_v = None
            overshoot_settling_valid = True
            undershoot_settling_valid = True
            settling_diagnostics: dict[str, Any] = {}
        else:
            settling_analysis_version = SETTLING_ANALYSIS_VERSION
            overshoot = dynamic_metrics.overshoot_pct
            undershoot = dynamic_metrics.undershoot_pct
            settling_time = dynamic_metrics.settling_time_s
            overshoot_settling_time = dynamic_metrics.overshoot_settling_time_s
            undershoot_settling_time = dynamic_metrics.undershoot_settling_time_s
            oscillations = dynamic_metrics.oscillations
            low_load_steady_v = dynamic_metrics.low_load_steady_v
            high_load_steady_v = dynamic_metrics.high_load_steady_v
            overshoot_settling_valid = dynamic_metrics.overshoot_settling_valid
            undershoot_settling_valid = dynamic_metrics.undershoot_settling_valid
            settling_diagnostics = dynamic_metrics.settling_diagnostics
        sustained_oscillation = (
            dynamic_metrics is not None
            and oscillations >= SUSTAINED_OSCILLATION_MIN_SIGN_CHANGES
            and self._has_sustained_oscillation(analysis_waveform)
        )
        if sustained_oscillation:
            # Invalid responses have no meaningful settling time. Keep numeric
            # zeros for backward-compatible serialization; pass_reasons marks
            # both Ts values as unavailable to the plot and GUI.
            settling_time = 0.0
            overshoot_settling_time = 0.0
            undershoot_settling_time = 0.0
            score = INVALID_TRANSIENT_PENALTY
            passed = False
            overshoot_settling_valid = False
            undershoot_settling_valid = False
            pass_reasons = [f"invalid transient waveform: sustained oscillation ({oscillations} crossings)"]
        elif not overshoot_settling_valid or not undershoot_settling_valid:
            score = INVALID_TRANSIENT_PENALTY
            passed = False
            invalid_edges = []
            if not overshoot_settling_valid:
                invalid_edges.append("OS")
            if not undershoot_settling_valid:
                invalid_edges.append("US")
            pass_reasons = [
                "invalid transient waveform: no reliable final settling dwell "
                f"for {'/'.join(invalid_edges)}"
            ]
        else:
            score = score_metrics(
                overshoot,
                undershoot,
                oscillations,
                overshoot_settling_time,
                undershoot_settling_time,
                self.targets,
            )
            passed = (
                overshoot <= self.targets.overshoot_pct
                and undershoot <= self.targets.undershoot_pct
                and overshoot_settling_time <= self.targets.settling_time_s
                and undershoot_settling_time <= self.targets.settling_time_s
            )
            pass_reasons = []
        return ResponseMetrics(
            overshoot_pct=overshoot,
            undershoot_pct=undershoot,
            settling_time_s=settling_time,
            oscillations=oscillations,
            score=min(MAX_PENALTY, score),
            passed=passed,
            overshoot_settling_time_s=overshoot_settling_time,
            undershoot_settling_time_s=undershoot_settling_time,
            low_load_steady_v=low_load_steady_v,
            high_load_steady_v=high_load_steady_v,
            settling_analysis_version=settling_analysis_version,
            overshoot_settling_valid=overshoot_settling_valid,
            undershoot_settling_valid=undershoot_settling_valid,
            settling_diagnostics=settling_diagnostics,
            pass_reasons=pass_reasons,
        )

    def _lowpass_response_waveform(self, waveform: Waveform) -> Waveform:
        filtered = self._zero_phase_lowpass(
            waveform.time_s,
            waveform.vout_v,
            cutoff_hz=self.RESPONSE_LOWPASS_CUTOFF_HZ,
        )
        if filtered is waveform.vout_v:
            return waveform
        return Waveform(time_s=waveform.time_s, vout_v=filtered, input_v=waveform.input_v)

    def _zero_phase_lowpass(self, time_s: list[float], values: list[float], cutoff_hz: float) -> list[float]:
        if cutoff_hz <= 0 or len(values) < 4 or len(time_s) != len(values):
            return values
        sample_dt = self._median_sample_dt(time_s)
        if sample_dt <= 0:
            return values
        sample_rate_hz = 1.0 / sample_dt
        # If the requested cutoff is at/above Nyquist, filtering cannot remove
        # anything meaningful and can only distort coarse synthetic waveforms.
        if cutoff_hz >= sample_rate_hz * 0.45:
            return values
        rc = 1.0 / (2.0 * math.pi * cutoff_hz)
        alpha = sample_dt / (rc + sample_dt)
        if not 0.0 < alpha < 1.0:
            return values
        forward = self._one_pole_lowpass(values, alpha)
        backward = self._one_pole_lowpass(list(reversed(forward)), alpha)
        return list(reversed(backward))

    def _one_pole_lowpass(self, values: list[float], alpha: float) -> list[float]:
        if not values:
            return []
        output = [values[0]]
        previous = values[0]
        for value in values[1:]:
            previous = previous + alpha * (value - previous)
            output.append(previous)
        return output

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
        analysis_events = self._analysis_edges(events)

        overshoot_pct = 0.0
        undershoot_pct = 0.0
        settling_time_s = 0.0
        overshoot_settling_time_s = 0.0
        undershoot_settling_time_s = 0.0
        overshoot_settling_valid = True
        undershoot_settling_valid = True
        settling_diagnostics: dict[str, Any] = {}
        oscillations = 0
        low_load_steady_values, high_load_steady_values = self._load_steady_values(waveform, events)
        low_load_steady_v = _mean(low_load_steady_values)
        high_load_steady_v = _mean(high_load_steady_values)
        overshoot_pct, undershoot_pct = self._load_step_extremes(waveform, events, low_load_steady_v, high_load_steady_v)
        for event in analysis_events:
            next_event = next((candidate for candidate in events if candidate.index > event.index), None)
            next_index = next_event.index if next_event is not None else len(waveform.time_s)
            if next_index <= event.index:
                continue

            segment = waveform.vout_v[event.index:next_index]
            if not segment:
                continue
            # Settling is relative to the steady value reached by this exact
            # hardware step. OS/US magnitude intentionally continues to use
            # the shared load-state values above so those percentages remain
            # comparable across the capture.
            final_value = self._local_step_steady_value(waveform, event.index, next_index)
            shared_final_value = self._event_target_steady_value(
                waveform,
                event.index,
                next_index,
                event.edge,
                low_load_steady_v,
                high_load_steady_v,
            )
            # An incomplete trailing generator transition can occur near the
            # capture boundary without becoming a debounced analysis edge. Do
            # not let that unrelated tail replace the known load-state target.
            reference_locked = False
            if abs(final_value - shared_final_value) > max(12e-3, abs(shared_final_value) * 0.015):
                final_value = shared_final_value
                reference_locked = True
            settling = self._dynamic_settling_time(
                waveform,
                event.index,
                next_index,
                final_value,
                event.edge,
                reference_locked=reference_locked,
            )
            settling_time_s = max(settling_time_s, settling.time_s)
            if event.edge == "falling":
                overshoot_settling_time_s = max(overshoot_settling_time_s, settling.time_s)
                overshoot_settling_valid = overshoot_settling_valid and settling.valid
                settling_diagnostics["overshoot"] = settling.diagnostics
            elif event.edge == "rising":
                undershoot_settling_time_s = max(undershoot_settling_time_s, settling.time_s)
                undershoot_settling_valid = undershoot_settling_valid and settling.valid
                settling_diagnostics["undershoot"] = settling.diagnostics
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
            overshoot_settling_valid=overshoot_settling_valid,
            undershoot_settling_valid=undershoot_settling_valid,
            settling_diagnostics=settling_diagnostics,
        )

    def _analysis_edges(self, events: list[_StepEvent]) -> list[_StepEvent]:
        first_rising = next((event for event in events if event.edge == "rising"), None)
        if first_rising is None:
            return events
        first_falling = next((event for event in events if event.edge == "falling" and event.index > first_rising.index), None)
        if first_falling is None:
            return events
        return [first_rising, first_falling]

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
                envelopes = self._binned_envelope(waveform, first_rising.index, end_index, bin_s=0.25e-6)
                if envelopes:
                    minimum = min(item.q05 for item in envelopes)
                    denominator = max(abs(high_load_steady_v), 1e-9)
                    undershoot_pct = max(0.0, (high_load_steady_v - minimum) / denominator * 100.0)

        if falling_events and low_load_steady_v is not None:
            first_falling = falling_events[0]
            next_rising = next((event for event in rising_events if event.index > first_falling.index), None)
            end_index = next_rising.index if next_rising is not None else len(waveform.vout_v)
            if end_index > first_falling.index:
                envelopes = self._binned_envelope(waveform, first_falling.index, end_index, bin_s=0.25e-6)
                if envelopes:
                    maximum = max(item.q95 for item in envelopes)
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

        # Use falling edges that close a real high-load interval. Some captures
        # include an extra trailing falling edge whose pre-window is already
        # back at low load; including it would bias the high-load steady value
        # upward and make the whole high-load plateau look like undershoot.
        selected_falling = [
            event
            for index, event in enumerate(events)
            if event.edge == "falling" and index > 0 and events[index - 1].edge == "rising"
        ]
        if not selected_falling:
            selected_falling = falling_events
        for event in selected_falling:
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

    def _event_target_steady_value(
        self,
        waveform: Waveform,
        start_index: int,
        end_index: int,
        edge: str,
        low_load_steady_v: float | None,
        high_load_steady_v: float | None,
    ) -> float:
        if edge == "falling" and low_load_steady_v is not None:
            return low_load_steady_v
        if edge == "rising" and high_load_steady_v is not None:
            return high_load_steady_v
        return self._post_step_steady_value(waveform, start_index, end_index)

    def _input_edges(self, input_v: list[float]) -> list[_StepEvent]:
        if len(input_v) < 2:
            return []
        # Estimate CH1 plateaus robustly. A single acquisition spike must not
        # move the switching threshold enough to create or hide a load edge.
        stride = max(1, len(input_v) // 4096)
        sampled = [float(value) for value in input_v[::stride] if math.isfinite(float(value))]
        if len(sampled) < 2:
            return []
        sampled.sort()
        low = _quantile(sampled, 0.10)
        high = _quantile(sampled, 0.90)
        span = high - low
        if span <= 1e-9:
            return []
        low_threshold = low + span * 0.35
        high_threshold = low + span * 0.65
        events: list[_StepEvent] = []
        first = float(input_v[0])
        if first >= high_threshold:
            state = "high"
        elif first <= low_threshold:
            state = "low"
        else:
            state = "high" if first >= (low + high) * 0.5 else "low"
        for index in range(1, len(input_v)):
            current = float(input_v[index])
            if not math.isfinite(current):
                continue
            if state == "low" and current >= high_threshold:
                events.append(_StepEvent(index=index, edge="rising"))
                state = "high"
            elif state == "high" and current <= low_threshold:
                events.append(_StepEvent(index=index, edge="falling"))
                state = "low"
        events = self._debounce_edges(events, len(input_v))
        return events

    def input_edge_indices(self, input_v: list[float]) -> list[tuple[int, str]]:
        """Return the same debounced CH1 edges used by transient analysis."""

        return [(event.index, event.edge) for event in self._input_edges(input_v)]

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
            if debounced and event.edge == debounced[-1].edge:
                # The earlier same-direction edge was a short spike whose
                # return crossing was swallowed by the debounce window.
                debounced[-1] = event
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

    def _local_step_steady_value(self, waveform: Waveform, start_index: int, end_index: int) -> float:
        """Return a robust steady reference for one physical load step."""

        segment_time = waveform.time_s[end_index - 1] - waveform.time_s[start_index]
        window_s = max(0.0, min(10e-6, segment_time * 0.25))
        end_time = waveform.time_s[end_index - 1]
        values = sorted(
            value
            for time_s, value in zip(
                waveform.time_s[start_index:end_index],
                waveform.vout_v[start_index:end_index],
            )
            if time_s >= end_time - window_s
        )
        if not values:
            values = sorted(waveform.vout_v[max(start_index, end_index - 5) : end_index])
        return _quantile(values, 0.50)

    def _dynamic_settling_time(
        self,
        waveform: Waveform,
        start_index: int,
        end_index: int,
        final_value: float,
        edge: str,
        *,
        reference_locked: bool = False,
    ) -> _SettlingResult:
        if end_index <= start_index:
            return _SettlingResult(0.0, False, {"valid": False, "reason": "empty step segment"})
        sample_dt = self._median_sample_dt(waveform.time_s[start_index:end_index])
        segment_time = waveform.time_s[end_index - 1] - waveform.time_s[start_index]
        ripple_tolerance = self._steady_ripple_tolerance(waveform, start_index, end_index)
        main_watch_s = min(segment_time, max(8e-6, min(15e-6, segment_time * 0.25)))
        tail_window_s = min(10e-6, max(0.0, segment_time * 0.25))
        origin_s = waveform.time_s[start_index]
        tail_start_s = origin_s + max(0.0, segment_time - tail_window_s)
        if reference_locked:
            noise_stop_s = min(segment_time, max(main_watch_s + 5e-6, segment_time * 0.40)) + origin_s
            tail_values = [
                value
                for time_s, value in zip(
                    waveform.time_s[start_index:end_index],
                    waveform.vout_v[start_index:end_index],
                )
                if origin_s + main_watch_s <= time_s <= noise_stop_s
            ]
        else:
            tail_values = [
                value
                for time_s, value in zip(
                    waveform.time_s[start_index:end_index],
                    waveform.vout_v[start_index:end_index],
                )
                if time_s >= tail_start_s
            ]
        if not tail_values:
            tail_values = waveform.vout_v[max(start_index, end_index - 8) : end_index]
        sorted_tail = sorted(tail_values)
        tail_center = _quantile(sorted_tail, 0.50)
        # V19 makes the displayed 600 kHz zero-phase LPF the complete Ts decision
        # signal. There is no 0.25 us binning and no second 5/9-bin median
        # filter. This keeps a real post-entry dip visible to the settling
        # detector and makes the plotted LPF directly explain the reported Ts.
        if not reference_locked:
            final_value = tail_center
        # V19 uses a direction-aware asymmetric band. For a voltage-falling
        # response (CH1 rising), the band is -4/+3 mV. For a voltage-rising
        # response (CH1 falling), it is -3/+5 mV. The 600 kHz trace can
        # graze this band because of normal switching ripple, so the temporal
        # duration/depth qualification below remains responsible for rejecting
        # non-physical threshold chatter.
        if edge == "rising":
            lower_tolerance = 4.0e-3
            upper_tolerance = 3.0e-3
            minimum_exit_depth = 0.0
            band_schedule = "voltage falling: -4/+3 mV"
        else:
            lower_tolerance = 3.0e-3
            upper_tolerance = 5.0e-3
            minimum_exit_depth = 0.5e-3
            band_schedule = "voltage rising: -3/+5 mV"

        def band_limits(elapsed_s: float) -> tuple[float, float]:
            del elapsed_s
            return lower_tolerance, upper_tolerance

        def outside_band(elapsed_s: float, value: float) -> bool:
            lower, upper = band_limits(elapsed_s)
            return value < final_value - lower or value > final_value + upper

        def depth_beyond_band(elapsed_s: float, value: float) -> tuple[float, float]:
            lower, upper = band_limits(elapsed_s)
            if value >= final_value:
                return max(0.0, value - final_value - upper), upper
            return max(0.0, final_value - value - lower), lower

        diagnostics: dict[str, Any] = {
            "method": "six_hundred_khz_asymmetric_band_reentry_v19",
            "edge": edge,
            "valid": False,
            "local_steady_v": float(final_value),
            "outer_tolerance_mv": float(max(lower_tolerance, upper_tolerance) * 1e3),
            "core_tolerance_mv": float(min(lower_tolerance, upper_tolerance) * 1e3),
            "final_tolerance_mv": float(max(lower_tolerance, upper_tolerance) * 1e3),
            "lower_tolerance_mv": float(lower_tolerance * 1e3),
            "upper_tolerance_mv": float(upper_tolerance * 1e3),
            "band_ramp_start_us": None,
            "band_ramp_stop_us": None,
            "band_schedule": band_schedule,
            "measured_ripple_tolerance_mv": float(ripple_tolerance * 1e3),
            "decision_filter_hz": float(self.RESPONSE_LOWPASS_CUTOFF_HZ),
            "uses_time_bins": False,
            "stable_dwell_us": 1.0,
            "minimum_exit_duration_us": 0.08,
            "late_exit_after_us": 5.0,
            "late_minimum_exit_duration_us": 3.0,
            "minimum_exit_depth_mv": float(minimum_exit_depth * 1e3),
            "exit_merge_gap_us": 0.05,
            "prominent_reversal_count": 0,
            "main_watch_us": float(main_watch_s * 1e6),
            "first_entry_us": None,
            "final_entry_us": None,
            "secondary_excursion_count": 0,
            "secondary_excursions": [],
            "rejected_band_graze_count": 0,
        }

        watched: list[tuple[float, float]] = [
            (time_s - origin_s, value)
            for time_s, value in zip(
                waveform.time_s[start_index:end_index],
                waveform.vout_v[start_index:end_index],
            )
        ]
        if not watched:
            diagnostics["reason"] = "no 600 kHz LPF samples in the transient window"
            return _SettlingResult(0.0, False, diagnostics)

        # The first band crossing belongs to the primary transition. Every
        # later exit is a real US/OS/dip candidate on the same 600 kHz trace. Ts
        # is moved to the re-entry after the final such excursion.
        saw_initial_outside = False
        first_entry_index: int | None = None
        for index, (elapsed_s, value) in enumerate(watched):
            if elapsed_s > main_watch_s + 1e-12:
                break
            if outside_band(elapsed_s, value):
                saw_initial_outside = True
            elif saw_initial_outside:
                first_entry_index = index
                break
        if first_entry_index is None:
            diagnostics["reason"] = "600 kHz LPF never enters the steady band"
            return _SettlingResult(0.0, False, diagnostics)

        first_entry_s = watched[first_entry_index][0]
        diagnostics["first_entry_us"] = float(first_entry_s * 1e6)
        final_entry_index = first_entry_index
        raw_excursions: list[tuple[int, int]] = []
        index = first_entry_index + 1
        while index < len(watched) and watched[index][0] <= main_watch_s + 1e-12:
            if not outside_band(watched[index][0], watched[index][1]):
                index += 1
                continue
            exit_index = index
            while (
                index < len(watched)
                and outside_band(watched[index][0], watched[index][1])
            ):
                index += 1
            if index >= len(watched):
                reentry_index = len(watched) - 1
            else:
                reentry_index = index
            raw_excursions.append((exit_index, reentry_index))

        # A 600 kHz trace can graze a numeric band edge for a few nanoseconds.
        # That is neither a physical second lobe nor a useful definition of
        # settling. Merge threshold chatter, then require meaningful time
        # outside the band. For the tighter voltage-falling band, any sustained
        # excursion counts regardless of depth so a visibly out-of-band
        # undershoot cannot be hidden by a second amplitude threshold. The
        # voltage-rising path retains its 0.5 mV noise allowance. This is
        # temporal qualification of the original 600 kHz samples, not
        # binning/filtering.
        merged_excursions: list[tuple[int, int]] = []
        for exit_index, reentry_index in raw_excursions:
            if (
                merged_excursions
                and watched[exit_index][0] - watched[merged_excursions[-1][1]][0]
                <= 0.05e-6 + sample_dt
            ):
                merged_excursions[-1] = (merged_excursions[-1][0], reentry_index)
            else:
                merged_excursions.append((exit_index, reentry_index))

        excursions: list[dict[str, Any]] = []
        rejected_band_grazes = 0
        for exit_index, reentry_index in merged_excursions:
            cluster = watched[exit_index : reentry_index + 1]
            extreme_offset = max(
                range(len(cluster)),
                key=lambda item: (
                    depth_beyond_band(cluster[item][0], cluster[item][1])[0]
                ),
            )
            extreme_time_s, extreme_value = cluster[extreme_offset]
            duration_s = watched[reentry_index][0] - watched[exit_index][0]
            depth_v, extreme_tolerance = depth_beyond_band(extreme_time_s, extreme_value)
            required_duration_s = (
                3.0e-6 if watched[exit_index][0] >= 5.0e-6 - sample_dt else 0.08e-6
            )
            qualified = (
                duration_s + sample_dt >= required_duration_s
                and depth_v >= minimum_exit_depth - 1e-12
            )
            detail = {
                "exit_us": float(watched[exit_index][0] * 1e6),
                "reentry_us": float(watched[reentry_index][0] * 1e6),
                "duration_us": float(duration_s * 1e6),
                "required_duration_us": float(required_duration_s * 1e6),
                "extreme_us": float(extreme_time_s * 1e6),
                "extreme_mv_from_steady": float((extreme_value - final_value) * 1e3),
                "depth_beyond_band_mv": float(depth_v * 1e3),
                "band_tolerance_mv": float(extreme_tolerance * 1e3),
                "direction": "high" if extreme_value > final_value else "low",
            }
            if not qualified:
                rejected_band_grazes += 1
                continue
            if reentry_index >= len(watched) - 1:
                diagnostics["reason"] = "qualified 600 kHz excursion has no steady-band re-entry"
                diagnostics["secondary_excursion_count"] = len(excursions) + 1
                diagnostics["secondary_excursions"] = (excursions + [detail])[-8:]
                diagnostics["rejected_band_graze_count"] = rejected_band_grazes
                return _SettlingResult(0.0, False, diagnostics)
            excursions.append(detail)
            final_entry_index = reentry_index

        final_entry_s = watched[final_entry_index][0]
        dwell_stop_s = final_entry_s + 1.0e-6
        # Dwell means no *qualified* excursion for 1 us. Rejected nanosecond
        # band grazes must not make an otherwise stable response invalid.
        if watched[-1][0] + sample_dt < dwell_stop_s:
            diagnostics["reason"] = "insufficient samples for 1.0 us final dwell"
            diagnostics["secondary_excursion_count"] = len(excursions)
            diagnostics["secondary_excursions"] = excursions[-8:]
            diagnostics["rejected_band_graze_count"] = rejected_band_grazes
            return _SettlingResult(0.0, False, diagnostics)

        if edge == "rising":
            primary_index = min(range(len(watched)), key=lambda item: watched[item][1])
        else:
            primary_index = max(range(len(watched)), key=lambda item: watched[item][1])
        diagnostics["primary_extreme_us"] = float(watched[primary_index][0] * 1e6)
        diagnostics["primary_extreme_v"] = float(watched[primary_index][1])
        diagnostics["secondary_excursion_count"] = len(excursions)
        diagnostics["prominent_reversal_count"] = len(excursions)
        diagnostics["secondary_excursions"] = excursions[-8:]
        diagnostics["rejected_band_graze_count"] = rejected_band_grazes
        final_entry_s = max(sample_dt, final_entry_s)
        diagnostics["valid"] = True
        diagnostics["final_entry_us"] = float(final_entry_s * 1e6)
        diagnostics["reason"] = "settled at final 600 kHz steady-band re-entry"
        return _SettlingResult(final_entry_s, True, diagnostics)

    def _settling_center_trend(
        self,
        envelopes: list[_EnvelopeBin],
        *,
        span_bins: int = 5,
    ) -> list[_TrendBin]:
        """Return an odd-span centered median trend used by settling V5."""

        centers = [item.q50 for item in envelopes]
        span_bins = max(1, int(span_bins))
        if span_bins % 2 == 0:
            span_bins += 1
        radius = span_bins // 2
        trend: list[_TrendBin] = []
        for index, envelope in enumerate(envelopes):
            window = sorted(
                centers[
                    max(0, index - radius) : min(len(centers), index + radius + 1)
                ]
            )
            trend.append(
                _TrendBin(
                    start_s=envelope.start_s,
                    stop_s=envelope.stop_s,
                    center=_quantile(window, 0.50),
                )
            )
        return trend

    @staticmethod
    def _merge_time_intervals(
        intervals: list[tuple[float, float]],
        *,
        merge_gap_s: float,
    ) -> list[tuple[float, float]]:
        """Merge already-qualified time intervals without changing duration."""

        merged: list[tuple[float, float]] = []
        for start_s, stop_s in sorted(intervals):
            if merged and start_s - merged[-1][1] <= merge_gap_s + 1e-15:
                merged[-1] = (merged[-1][0], max(merged[-1][1], stop_s))
            else:
                merged.append((start_s, stop_s))
        return merged

    def _prominent_reversal_clusters(
        self,
        trend: list[_TrendBin],
        final_value: float,
        *,
        first_entry_s: float | None,
        watch_stop_s: float,
        prominence_swing: float,
        min_duration_s: float,
        reversal_confirmation: float,
        recovery_tolerance: float,
        recovery_trend: list[_TrendBin] | None = None,
    ) -> list[tuple[float, float]]:
        """Find coherent post-entry peak/valley swings missed by absolute bands.

        A plain final-value threshold cannot see a response that moves from one
        side of steady state to the other while both endpoints remain inside
        the late core band.  This detector therefore measures peak-to-valley
        prominence, but accepts an extremum only after the trend moves back by
        ``reversal_confirmation``.  That confirmation prevents an ordinary
        monotonic approach to steady state from being classified as rebound.
        """

        if first_entry_s is None:
            return []
        watched = [
            item
            for item in trend
            if first_entry_s - 1e-12 <= item.start_s <= watch_stop_s + 1e-12
        ]
        if len(watched) < 3:
            return []

        values = [item.center for item in watched]
        # The nine-bin trend is spaced at 0.25 us. A two-bin local window and
        # a 3 us look-ahead retain broad hardware lobes while rejecting single
        # switching spikes and tiny median-filter plateaus.
        local_radius = 2
        confirmation_stop_s = 10.0e-6
        anchor = 0
        intervals: list[tuple[float, float]] = []
        index = 1
        while index < len(watched) - 1:
            item = watched[index]
            local = values[
                max(anchor, index - local_radius) : min(len(values), index + local_radius + 1)
            ]
            if not local:
                index += 1
                continue
            is_peak = values[index] >= max(local) - 1e-12
            is_valley = values[index] <= min(local) + 1e-12
            prior = values[anchor : index + 1]
            prior_min_offset = min(range(len(prior)), key=prior.__getitem__)
            prior_max_offset = max(range(len(prior)), key=prior.__getitem__)
            prior_min_index = anchor + prior_min_offset
            prior_max_index = anchor + prior_max_offset

            future_stop = index + 1
            while (
                future_stop < len(watched)
                and watched[future_stop].start_s - item.start_s <= confirmation_stop_s + 1e-12
            ):
                future_stop += 1
            future = values[index + 1 : future_stop]
            peak_retreat = (
                max((values[index] - value for value in future), default=0.0)
                >= reversal_confirmation
            )
            valley_retreat = (
                max((value - values[index] for value in future), default=0.0)
                >= reversal_confirmation
            )

            start_index: int | None = None
            if (
                is_peak
                and peak_retreat
                and values[index] - values[prior_min_index] >= prominence_swing
                and item.start_s - watched[prior_min_index].start_s >= min_duration_s - 1e-12
            ):
                start_index = prior_min_index
            elif (
                is_valley
                and valley_retreat
                and values[prior_max_index] - values[index] >= prominence_swing
                and item.start_s - watched[prior_max_index].start_s >= min_duration_s - 1e-12
            ):
                start_index = prior_max_index

            if start_index is None:
                index += 1
                continue

            # Include the confirmed turn and, when the extremum itself is
            # outside the stricter early band, extend through its return to
            # that band. The normal 1 us final-dwell check still runs after
            # this interval and determines the reported Ts start.
            confirmation_index = index
            for candidate in range(index + 1, future_stop):
                if (
                    abs(values[candidate] - values[index]) >= reversal_confirmation
                    and (values[candidate] - values[index]) * (values[index] - values[start_index]) < 0
                ):
                    confirmation_index = candidate
                    break
            # The look-ahead only proves that this extremum is a real reversal;
            # it must not be added to Ts. Back-date the excursion end to the
            # extremum itself, or to the first subsequent return inside the
            # stricter recovery band when the extremum is outside that band.
            stop_s = item.stop_s
            if abs(values[index] - final_value) > recovery_tolerance:
                recovery_bins = recovery_trend or watched
                recovery_stop_s = stop_s
                for candidate in recovery_bins:
                    if candidate.start_s < item.start_s - 1e-12:
                        continue
                    recovery_stop_s = candidate.stop_s
                    if abs(candidate.center - final_value) <= recovery_tolerance:
                        break
                stop_s = recovery_stop_s
            intervals.append(
                (watched[start_index].start_s, stop_s)
            )
            anchor = index
            index = max(index + 1, confirmation_index)

        return self._merge_time_intervals(intervals, merge_gap_s=0.50e-6)

    def _merged_outside_clusters(
        self,
        outside: list[bool],
        bins: list[_TrendBin],
        *,
        merge_gap_s: float,
        min_duration_s: float,
    ) -> list[tuple[float, float]]:
        primitive: list[tuple[int, int]] = []
        index = 0
        while index < len(outside):
            if not outside[index]:
                index += 1
                continue
            start = index
            while index + 1 < len(outside) and outside[index + 1]:
                index += 1
            primitive.append((start, index))
            index += 1

        merged: list[tuple[int, int]] = []
        for start, stop in primitive:
            if merged and bins[start].start_s - bins[merged[-1][1]].stop_s <= merge_gap_s:
                merged[-1] = (merged[-1][0], stop)
            else:
                merged.append((start, stop))
        return [
            (bins[start].start_s, bins[stop].stop_s)
            for start, stop in merged
            if bins[stop].stop_s - bins[start].start_s + 1e-12 >= min_duration_s
        ]

    def _first_stable_trend_start(
        self,
        trend: list[_TrendBin],
        final_value: float,
        *,
        tolerance: float,
        stable_duration_s: float,
        not_before_s: float,
    ) -> float | None:
        index = 0
        while index < len(trend):
            item = trend[index]
            if item.start_s + 1e-12 < not_before_s or abs(item.center - final_value) > tolerance:
                index += 1
                continue
            start = index
            while index < len(trend) and abs(trend[index].center - final_value) <= tolerance:
                index += 1
            stop = index
            duration_s = trend[stop - 1].stop_s - trend[start].start_s
            if duration_s + 1e-12 >= stable_duration_s:
                return trend[start].start_s
        return None

    def _first_stable_window_start(
        self,
        envelopes: list[_EnvelopeBin],
        final_value: float,
        tolerance: float,
        stable_duration_s: float,
    ) -> float | None:
        if not envelopes:
            return None
        outside = [abs(envelope.q50 - final_value) > tolerance for envelope in envelopes]
        saw_outside = False
        index = 0
        while index < len(outside):
            if outside[index]:
                saw_outside = True
                index += 1
                continue
            start = index
            while index < len(outside) and not outside[index]:
                index += 1
            stop = index
            duration_s = envelopes[stop - 1].stop_s - envelopes[start].start_s
            if saw_outside and duration_s >= stable_duration_s:
                return envelopes[start].start_s
        return None

    def _last_trend_outside_stop(
        self,
        waveform: Waveform,
        start_index: int,
        end_index: int,
        final_value: float,
        tolerance: float,
        min_duration_s: float,
    ) -> float | None:
        bins = self._binned_center_trend(waveform, start_index, end_index, bin_s=0.25e-6)
        if not bins:
            return None
        outside = [abs(item.center - final_value) > tolerance for item in bins]

        last_stop_s: float | None = None
        index = 0
        while index < len(outside):
            if not outside[index]:
                index += 1
                continue
            start = index
            while index < len(outside) and outside[index]:
                index += 1
            stop = index
            duration_s = bins[stop - 1].stop_s - bins[start].start_s
            duration_epsilon_s = max(1e-12, min_duration_s * 0.02)
            if duration_s + duration_epsilon_s >= min_duration_s:
                last_stop_s = bins[stop - 1].stop_s
        return last_stop_s

    def _last_directional_extreme_stop(
        self,
        envelopes: list[_EnvelopeBin],
        final_value: float,
        tolerance: float,
        edge: str,
        min_duration_s: float,
        merge_gap_s: float,
    ) -> float | None:
        if edge == "falling":
            threshold = final_value + tolerance
            hits = [envelope.q95 > threshold for envelope in envelopes]
        elif edge == "rising":
            threshold = final_value - tolerance
            hits = [envelope.q05 < threshold for envelope in envelopes]
        else:
            return None

        # Directional extrema can appear as sparse q95/q05 bins because the
        # waveform is noisy. Merge nearby extrema into a cluster, but ignore a
        # single isolated bin so late one-off noise does not reset settling.
        clusters: list[tuple[float, float]] = []
        index = 0
        while index < len(hits):
            if not hits[index]:
                index += 1
                continue
            start = index
            stop = index
            index += 1
            while index < len(hits):
                if hits[index]:
                    stop = index
                    index += 1
                    continue
                gap_start = envelopes[stop].stop_s
                gap_index = index
                while gap_index < len(hits) and not hits[gap_index]:
                    gap_index += 1
                if gap_index >= len(hits):
                    index = gap_index
                    break
                gap_s = envelopes[gap_index].start_s - gap_start
                if gap_s > merge_gap_s:
                    break
                stop = gap_index
                index = gap_index + 1
            clusters.append((envelopes[start].start_s, envelopes[stop].stop_s))

        last_stop_s: float | None = None
        for start_s, stop_s in clusters:
            if stop_s - start_s >= min_duration_s:
                last_stop_s = stop_s
        return last_stop_s

    def _last_iqr_outside_stop(
        self,
        envelopes: list[_EnvelopeBin],
        final_value: float,
        tolerance: float,
        min_duration_s: float,
    ) -> float | None:
        outside = [
            envelope.q75 < final_value - tolerance or envelope.q25 > final_value + tolerance
            for envelope in envelopes
        ]
        return self._last_significant_outside_stop(outside, envelopes, min_duration_s=min_duration_s)

    def _last_significant_outside_stop(
        self,
        outside: list[bool],
        envelopes: list[_EnvelopeBin],
        min_duration_s: float,
    ) -> float | None:
        last_stop_s: float | None = None
        index = 0
        while index < len(outside):
            if not outside[index]:
                index += 1
                continue
            start = index
            while index < len(outside) and outside[index]:
                index += 1
            stop = index
            duration_s = envelopes[stop - 1].stop_s - envelopes[start].start_s
            if duration_s >= min_duration_s:
                last_stop_s = envelopes[stop - 1].stop_s
        return last_stop_s

    def _steady_ripple_tolerance(self, waveform: Waveform, start_index: int, end_index: int) -> float:
        segment_time = waveform.time_s[end_index - 1] - waveform.time_s[start_index]
        window_s = max(0.0, min(10e-6, segment_time * 0.25))
        end_time = waveform.time_s[end_index - 1]
        values = [
            value
            for time_s, value in zip(waveform.time_s[start_index:end_index], waveform.vout_v[start_index:end_index])
            if time_s >= end_time - window_s
        ]
        if len(values) < 8:
            return 0.0
        sorted_values = sorted(values)
        ripple_half = (_quantile(sorted_values, 0.90) - _quantile(sorted_values, 0.10)) / 2.0
        return max(0.0, ripple_half * 1.5)

    def _binned_envelope(
        self,
        waveform: Waveform,
        start_index: int,
        end_index: int,
        bin_s: float,
    ) -> list[_EnvelopeBin]:
        if end_index <= start_index:
            return []
        sample_dt = self._median_sample_dt(waveform.time_s[start_index:end_index])
        if sample_dt <= 0:
            values = sorted(waveform.vout_v[start_index:end_index])
            if not values:
                return []
            return [
                _EnvelopeBin(
                    0.0,
                    0.0,
                    _quantile(values, 0.05),
                    _quantile(values, 0.10),
                    _quantile(values, 0.25),
                    _quantile(values, 0.50),
                    _quantile(values, 0.75),
                    _quantile(values, 0.90),
                    _quantile(values, 0.95),
                )
            ]
        points_per_bin = max(8, int(round(bin_s / sample_dt)))
        points_per_bin = min(points_per_bin, max(8, end_index - start_index))
        origin = waveform.time_s[start_index]
        bins: list[_EnvelopeBin] = []
        index = start_index
        while index < end_index:
            stop = min(end_index, index + points_per_bin)
            values = sorted(waveform.vout_v[index:stop])
            if values:
                bins.append(
                    _EnvelopeBin(
                        start_s=waveform.time_s[index] - origin,
                        stop_s=waveform.time_s[stop - 1] - origin,
                        q05=_quantile(values, 0.05),
                        q10=_quantile(values, 0.10),
                        q25=_quantile(values, 0.25),
                        q50=_quantile(values, 0.50),
                        q75=_quantile(values, 0.75),
                        q90=_quantile(values, 0.90),
                        q95=_quantile(values, 0.95),
                    )
                )
            index = stop
        return bins

    def _binned_center_trend(
        self,
        waveform: Waveform,
        start_index: int,
        end_index: int,
        bin_s: float,
    ) -> list[_TrendBin]:
        if end_index <= start_index:
            return []
        sample_dt = self._median_sample_dt(waveform.time_s[start_index:end_index])
        if sample_dt <= 0:
            values = waveform.vout_v[start_index:end_index]
            if not values:
                return []
            return [_TrendBin(0.0, 0.0, _quantile(sorted(values), 0.50))]

        points_per_bin = max(8, int(round(bin_s / sample_dt)))
        points_per_bin = min(points_per_bin, max(8, end_index - start_index))
        origin = waveform.time_s[start_index]
        values = waveform.vout_v[start_index:end_index]

        bins: list[_TrendBin] = []
        relative = 0
        total = end_index - start_index
        while relative < total:
            stop_relative = min(total, relative + points_per_bin)
            count = stop_relative - relative
            if count > 0:
                center = _quantile(sorted(values[relative:stop_relative]), 0.50)
                bins.append(
                    _TrendBin(
                        start_s=waveform.time_s[start_index + relative] - origin,
                        stop_s=waveform.time_s[start_index + stop_relative - 1] - origin,
                        center=center,
                    )
                )
            relative = stop_relative
        return bins

    def _smoothed_segment(self, waveform: Waveform, start_index: int, end_index: int, window_s: float) -> list[float]:
        values = waveform.vout_v[start_index:end_index]
        if len(values) < 3:
            return values
        sample_dt = self._median_sample_dt(waveform.time_s[start_index:end_index])
        if sample_dt <= 0:
            return values
        window_points = max(3, int(round(window_s / sample_dt)))
        if window_points <= 3:
            return values
        window_points = min(window_points, max(3, len(values) // 4))
        if window_points <= 3:
            return values
        half = window_points // 2
        prefix = [0.0]
        for value in values:
            prefix.append(prefix[-1] + value)
        smoothed: list[float] = []
        for index in range(len(values)):
            start = max(0, index - half)
            stop = min(len(values), index + half + 1)
            smoothed.append((prefix[stop] - prefix[start]) / (stop - start))
        return smoothed

    def _median_sample_dt(self, time_s: list[float]) -> float:
        if len(time_s) < 2:
            return 0.0
        diffs = [b - a for a, b in zip(time_s, time_s[1:]) if b > a]
        if not diffs:
            return 0.0
        diffs.sort()
        middle = len(diffs) // 2
        if len(diffs) % 2:
            return diffs[middle]
        return (diffs[middle - 1] + diffs[middle]) / 2.0

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

    def _has_sustained_oscillation(self, waveform: Waveform) -> bool:
        """Return whether a large oscillation persists late in a load segment."""

        if not waveform.input_v or len(waveform.input_v) != len(waveform.time_s):
            return False
        events = self._input_edges(waveform.input_v)
        for event in self._analysis_edges(events):
            next_event = next((candidate for candidate in events if candidate.index > event.index), None)
            end_index = next_event.index if next_event is not None else len(waveform.time_s)
            if end_index - event.index < 8:
                continue
            start_time = waveform.time_s[event.index]
            stop_time = waveform.time_s[end_index - 1]
            segment_duration = max(0.0, stop_time - start_time)
            tail_start_time = start_time + segment_duration * 0.55
            tail = [
                value
                for time_s, value in zip(
                    waveform.time_s[event.index:end_index],
                    waveform.vout_v[event.index:end_index],
                )
                if time_s >= tail_start_time
            ]
            if len(tail) < 8:
                continue

            ordered = sorted(tail)
            center = _quantile(ordered, 0.50)
            robust_span = _quantile(ordered, 0.95) - _quantile(ordered, 0.05)
            minimum_span = max(25e-3, abs(center) * 0.04)
            if robust_span < minimum_span:
                continue

            tolerance = max(4e-3, abs(center) * 0.01)
            signs: list[int] = []
            for value in tail:
                delta = value - center
                if abs(delta) <= tolerance:
                    continue
                sign = 1 if delta > 0 else -1
                if not signs or signs[-1] != sign:
                    signs.append(sign)
            if len(signs) - 1 >= SUSTAINED_OSCILLATION_MIN_SIGN_CHANGES:
                return True
        return False

    def analyze_hardware(
        self,
        waveform: Waveform | None,
        bode_margins: dict | None,
        enable_transient: bool = True,
        enable_bode: bool = True,
        precomputed_transient: ResponseMetrics | None = None,
    ) -> ResponseMetrics:
        if not enable_transient and not enable_bode:
            raise ValueError("At least one analysis mode must be enabled.")

        transient = precomputed_transient if enable_transient else None
        if transient is None and enable_transient and waveform is not None:
            transient = self.analyze(waveform)
        margins = bode_margins or {}
        phase_margin = _optional_float(margins.get("phase_margin_deg"))
        crossover = _optional_float(margins.get("phase_crossover_hz"))
        gain_margin = _optional_float(margins.get("gain_margin_db"))
        gain_rebound_db = _optional_float(margins.get("gain_rebound_db"))
        gain_flat_span_decades = _optional_float(margins.get("gain_flat_span_decades"))
        gain_slope_p90 = _optional_float(margins.get("gain_slope_p90_db_per_decade"))
        gain_shape_penalty = max(0.0, _optional_float(margins.get("gain_shape_penalty")) or 0.0)
        gain_shape_valid = margins.get("gain_shape_valid")
        gain_crossover_count = _optional_int(margins.get("gain_crossover_count")) or 0
        duplicate_gain_crossover = bool(margins.get("duplicate_gain_crossover")) or gain_crossover_count > 1

        reasons: list[str] = list(transient.pass_reasons) if transient is not None else []
        transient_invalid = any("invalid transient waveform" in str(reason).lower() for reason in reasons)
        score = transient.score if transient is not None else 0.0
        phase_error: float | None = None
        crossover_error_pct: float | None = None
        crossover_headroom_pct: float | None = None
        bode_invalid = False

        if enable_bode:
            if duplicate_gain_crossover:
                bode_invalid = True
                second = _optional_float(margins.get("second_phase_crossover_hz"))
                if second is not None:
                    reasons.append(f"invalid bode: second 0 dB crossover at {second:.3g} Hz")
                else:
                    reasons.append("invalid bode: duplicate 0 dB crossover")

            score += gain_shape_penalty
            gain_shape_ok = gain_shape_valid is not False
            if not gain_shape_ok:
                shape_parts: list[str] = []
                if gain_rebound_db is not None:
                    shape_parts.append(f"gain rebound {gain_rebound_db:.2f} dB")
                if gain_flat_span_decades is not None:
                    shape_parts.append(f"flat span {gain_flat_span_decades:.2f} decade")
                reasons.append(
                    "bode gain shape failed: " + (", ".join(shape_parts) if shape_parts else "non-descending gain")
                )

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
                crossover_error_pct = max(
                    0.0,
                    (crossover - self.targets.crossover_frequency_hz) / self.targets.crossover_frequency_hz * 100.0,
                )
                crossover_headroom_pct = max(
                    0.0,
                    (self.targets.crossover_frequency_hz - crossover) / self.targets.crossover_frequency_hz * 100.0,
                )
                crossover_ok = crossover <= self.targets.crossover_frequency_hz
                score += crossover_error_pct * 0.5
                if not crossover_ok:
                    reasons.append(f"crossover above upper limit by {crossover_error_pct:.1f}%")

            gain_ok = gain_shape_ok
        else:
            phase_ok = True
            crossover_ok = True
            gain_ok = True

        transient_ok = transient.passed if transient is not None else True
        if enable_transient and not transient_ok:
            reasons.append("transient limits not met")
        passed = transient_ok and phase_ok and crossover_ok and gain_ok
        if transient_invalid:
            score = INVALID_TRANSIENT_PENALTY
            passed = False
        elif bode_invalid:
            score = 250.0
            passed = False
        if passed:
            reward = _passed_reward(
                targets=self.targets,
                transient=transient,
                phase_error=phase_error,
                crossover_error_pct=crossover_error_pct,
                crossover_headroom_pct=crossover_headroom_pct,
                enable_transient=enable_transient,
                enable_bode=enable_bode,
            )
            if reward > 0:
                score -= reward
                reasons.append(f"passed reward {reward:.3f}")
            reasons.append("passed")

        return ResponseMetrics(
            overshoot_pct=transient.overshoot_pct if transient is not None else 0.0,
            undershoot_pct=transient.undershoot_pct if transient is not None else 0.0,
            settling_time_s=transient.settling_time_s if transient is not None else 0.0,
            oscillations=transient.oscillations if transient is not None else 0,
            score=min(MAX_PENALTY, score),
            passed=passed,
            overshoot_settling_time_s=transient.overshoot_settling_time_s if transient is not None else 0.0,
            undershoot_settling_time_s=transient.undershoot_settling_time_s if transient is not None else 0.0,
            low_load_steady_v=transient.low_load_steady_v if transient is not None else None,
            high_load_steady_v=transient.high_load_steady_v if transient is not None else None,
            phase_margin_deg=phase_margin,
            crossover_frequency_hz=crossover,
            gain_margin_db=gain_margin,
            bode_gain_rebound_db=gain_rebound_db,
            bode_gain_flat_span_decades=gain_flat_span_decades,
            bode_gain_slope_p90_db_per_decade=gain_slope_p90,
            bode_gain_shape_penalty=gain_shape_penalty,
            settling_analysis_version=(
                transient.settling_analysis_version if transient is not None else SETTLING_ANALYSIS_VERSION
            ),
            overshoot_settling_valid=(
                transient.overshoot_settling_valid if transient is not None else True
            ),
            undershoot_settling_valid=(
                transient.undershoot_settling_valid if transient is not None else True
            ),
            settling_diagnostics=(transient.settling_diagnostics if transient is not None else {}),
            pass_reasons=reasons,
        )


def score_metrics(
    overshoot_pct: float,
    undershoot_pct: float,
    oscillations: int,
    overshoot_settling_time_s: float,
    undershoot_settling_time_s: float,
    targets: TuningTargets,
) -> float:
    excess_os = max(0.0, overshoot_pct - targets.overshoot_pct)
    excess_us = max(0.0, undershoot_pct - targets.undershoot_pct)
    excess_os_settling_us = max(0.0, (overshoot_settling_time_s - targets.settling_time_s) * 1e6)
    excess_us_settling_us = max(0.0, (undershoot_settling_time_s - targets.settling_time_s) * 1e6)
    return min(
        MAX_PENALTY,
        excess_os
        + excess_us
        + SETTLING_WEIGHT_PER_US * excess_os_settling_us
        + SETTLING_WEIGHT_PER_US * excess_us_settling_us
    )


def _passed_reward(
    targets: TuningTargets,
    transient: ResponseMetrics | None,
    phase_error: float | None,
    crossover_error_pct: float | None,
    crossover_headroom_pct: float | None,
    enable_transient: bool,
    enable_bode: bool,
) -> float:
    """Tie-break reward applied only after every enabled target passes.

    OS/US amplitude headroom stays a small tie-breaker. Settling-time
    headroom mirrors the transient penalty units and coefficients: microseconds
    with the same 10x coefficient used for excess settling time.
    """

    reward = 0.0
    if enable_transient and transient is not None:
        reward += 0.15 * _headroom(targets.overshoot_pct, transient.overshoot_pct)
        reward += 0.15 * _headroom(targets.undershoot_pct, transient.undershoot_pct)
        reward += SETTLING_WEIGHT_PER_US * max(
            0.0,
            (targets.settling_time_s - transient.overshoot_settling_time_s) * 1e6,
        )
        reward += SETTLING_WEIGHT_PER_US * max(
            0.0,
            (targets.settling_time_s - transient.undershoot_settling_time_s) * 1e6,
        )

    if enable_bode:
        if phase_error is not None:
            reward += min(max(targets.phase_margin_tolerance_deg - phase_error, 0.0), targets.phase_margin_tolerance_deg) * 0.05
        if crossover_error_pct is not None and crossover_error_pct <= 0.0 and crossover_headroom_pct is not None:
            reward += min(crossover_headroom_pct, 100.0) / 100.0 * 0.25

    return reward


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


def _optional_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = max(0.0, min(1.0, fraction)) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
