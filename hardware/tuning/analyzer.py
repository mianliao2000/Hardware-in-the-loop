"""Response analysis helpers for PID tuning iterations."""

from __future__ import annotations

from dataclasses import dataclass
import math

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
    RESPONSE_LOWPASS_CUTOFF_HZ = 5_000_000.0

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
        else:
            overshoot = dynamic_metrics.overshoot_pct
            undershoot = dynamic_metrics.undershoot_pct
            settling_time = dynamic_metrics.settling_time_s
            overshoot_settling_time = dynamic_metrics.overshoot_settling_time_s
            undershoot_settling_time = dynamic_metrics.undershoot_settling_time_s
            oscillations = dynamic_metrics.oscillations
            low_load_steady_v = dynamic_metrics.low_load_steady_v
            high_load_steady_v = dynamic_metrics.high_load_steady_v
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
            final_value = self._event_target_steady_value(
                waveform,
                event.index,
                next_index,
                event.edge,
                low_load_steady_v,
                high_load_steady_v,
            )
            event_settling_time = self._dynamic_settling_time(waveform, event.index, next_index, final_value, event.edge)
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

    def _dynamic_settling_time(
        self,
        waveform: Waveform,
        start_index: int,
        end_index: int,
        final_value: float,
        edge: str,
    ) -> float:
        if end_index <= start_index:
            return 0.0
        if edge == "rising":
            tolerance_pct = self.targets.undershoot_pct
        elif edge == "falling":
            tolerance_pct = self.targets.overshoot_pct
        else:
            tolerance_pct = max(self.targets.overshoot_pct, self.targets.undershoot_pct)
        sample_dt = self._median_sample_dt(waveform.time_s[start_index:end_index])
        segment_time = waveform.time_s[end_index - 1] - waveform.time_s[start_index]
        # Settling uses a tighter, ripple-aware band than the OS/US pass limit.
        # A full 3% transient limit is too wide for settling: later load-line
        # dips can hide inside that band even though the response is visibly not
        # settled. The 0.65% cap keeps shallow sustained rollback visible while
        # the ripple tolerance below still suppresses ordinary switching ripple.
        settling_pct = min(max(tolerance_pct, 0.0), 0.65)
        percentage_tolerance = max(abs(final_value) * settling_pct / 100.0, 1e-6)
        ripple_tolerance = self._steady_ripple_tolerance(waveform, start_index, end_index)
        ripple_floor = max(6e-3, abs(final_value) * 0.005)
        tolerance = max(percentage_tolerance, ripple_tolerance, ripple_floor)
        envelopes = self._binned_envelope(waveform, start_index, end_index, bin_s=0.25e-6)
        if not envelopes:
            return max(sample_dt, 0.05e-6)

        # Settling is decided by the main transient after the edge. Looking all
        # the way to the next load edge makes slow DC drift or switching ripple
        # look like "not settled" and can stretch Ts to the whole segment. Still
        # keep enough of the segment to catch the common overdamped rollback/dip
        # after a first apparent recovery.
        reset_watch_s = min(segment_time, max(12e-6, segment_time * 0.60))
        watched_envelopes = [envelope for envelope in envelopes if envelope.start_s <= reset_watch_s]
        if not watched_envelopes:
            watched_envelopes = envelopes

        # Use median/interquartile envelopes instead of raw or q10/q90 samples.
        # q10/q90 is still too sensitive for the TPU load-step captures because
        # normal switching ripple can live there for the entire steady state.
        stable_tolerance = max(tolerance, ripple_tolerance * 0.85, 5.0e-3, abs(final_value) * 0.006)
        reset_tolerance = max(stable_tolerance, ripple_tolerance * 1.10)
        outside = [
            abs(envelope.q50 - final_value) > reset_tolerance
            or envelope.q75 < final_value - reset_tolerance
            or envelope.q25 > final_value + reset_tolerance
            for envelope in watched_envelopes
        ]
        min_reset_duration_s = min(1.2e-6, max(0.45e-6, segment_time * 0.012))
        last_significant_outside_stop = self._last_significant_outside_stop(
            outside,
            watched_envelopes,
            min_duration_s=min_reset_duration_s,
        )
        # q05/q95 directional extrema are useful for OS/US magnitude, but they
        # are too sensitive for settling because one late narrow spike can reset
        # Ts. Settling uses the median/IQR sustained-outside checks below.
        # The core settling answer should come from a stable dwell, not from
        # "last outside until the next edge". This catches the common
        # overdamped rollback shape: the waveform reaches the steady value,
        # dips back out, then settles again. A two-microsecond dwell rejects
        # that false first crossing while ignoring slow drift much later.
        stable_window_stop = self._first_stable_window_start(
            envelopes,
            final_value,
            tolerance=stable_tolerance,
            stable_duration_s=min(2.0e-6, max(0.8e-6, segment_time * 0.025)),
        )
        # The trend median can miss a shallow but persistent rollback when the
        # bin is partly settled and partly dipping. Use the interquartile band as
        # a second, still spike-resistant guard. This is bidirectional on
        # purpose: after a falling edge, a later dip below the low-load steady
        # value must reset OS settling, and after a rising edge a later rebound
        # above the high-load steady value must reset US settling.
        iqr_stop = self._last_iqr_outside_stop(
            watched_envelopes,
            final_value,
            tolerance=reset_tolerance,
            min_duration_s=min_reset_duration_s,
        )
        candidates = [
            value
            for value in (last_significant_outside_stop, iqr_stop)
            if value is not None
        ]
        if stable_window_stop is not None:
            candidates.append(stable_window_stop)
        if not candidates:
            return max(sample_dt, min(0.25e-6, max(segment_time, sample_dt)))
        return max(sample_dt, max(candidates))

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
        gain_crossover_count = _optional_int(margins.get("gain_crossover_count")) or 0
        duplicate_gain_crossover = bool(margins.get("duplicate_gain_crossover")) or gain_crossover_count > 1

        reasons: list[str] = []
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

            gain_ok = True
        else:
            phase_ok = True
            crossover_ok = True
            gain_ok = True

        transient_ok = transient.passed if transient is not None else True
        if enable_transient and not transient_ok:
            reasons.append("transient limits not met")
        passed = transient_ok and phase_ok and crossover_ok and gain_ok
        if bode_invalid:
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
    overshoot_settling_time_s: float,
    undershoot_settling_time_s: float,
    targets: TuningTargets,
) -> float:
    excess_os = max(0.0, overshoot_pct - targets.overshoot_pct)
    excess_us = max(0.0, undershoot_pct - targets.undershoot_pct)
    excess_os_settling_us = max(0.0, (overshoot_settling_time_s - targets.settling_time_s) * 1e6)
    excess_us_settling_us = max(0.0, (undershoot_settling_time_s - targets.settling_time_s) * 1e6)
    return excess_os + excess_us + 3.0 * excess_os_settling_us + 3.0 * excess_us_settling_us


def _passed_reward(
    targets: TuningTargets,
    transient: ResponseMetrics | None,
    phase_error: float | None,
    crossover_error_pct: float | None,
    crossover_headroom_pct: float | None,
    enable_transient: bool,
    enable_bode: bool,
) -> float:
    """Small tie-break reward applied only after every enabled target passes."""

    reward = 0.0
    if enable_transient and transient is not None:
        reward += 0.15 * _headroom(targets.overshoot_pct, transient.overshoot_pct)
        reward += 0.15 * _headroom(targets.undershoot_pct, transient.undershoot_pct)
        reward += 0.125 * _headroom(targets.settling_time_s, transient.overshoot_settling_time_s)
        reward += 0.125 * _headroom(targets.settling_time_s, transient.undershoot_settling_time_s)

    if enable_bode:
        if phase_error is not None:
            reward += min(max(targets.phase_margin_tolerance_deg - phase_error, 0.0), targets.phase_margin_tolerance_deg) * 0.05
        if crossover_error_pct is not None and crossover_error_pct <= 0.0 and crossover_headroom_pct is not None:
            reward += min(crossover_headroom_pct, 100.0) / 100.0 * 0.25

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
