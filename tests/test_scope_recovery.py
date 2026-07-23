from __future__ import annotations

import unittest
from unittest import mock

from gui import server
from hardware.instruments.oscilloscope import WaveformCapture
from hardware.tuning.models import ResponseMetrics


class _RetryingScope:
    def __init__(self) -> None:
        self.acquire_calls = 0
        self.prepare_calls: list[bool] = []

    def set_channel_display(self, source: str, enabled: bool) -> None:
        return None

    def set_edge_trigger(self, source: str, slope: str) -> None:
        return None

    def set_horizontal_window(self, duration_s: float) -> float:
        return duration_s / 10.0

    def set_trigger_position_from_left(self, offset_s: float, window_s: float) -> float:
        return 100.0 * offset_s / window_s

    def prepare_autotune_acquisition(self, **kwargs):
        forced = bool(kwargs.get("force_flush", False))
        self.prepare_calls.append(forced)
        return {
            "history_after": False,
            "record_length_after": 250_000,
            "force_flush": forced,
        }

    def single_acquisition(self, **kwargs) -> None:
        self.acquire_calls += 1
        if self.acquire_calls < 3:
            raise TimeoutError("Scope single acquisition did not complete within 8.0 s.")

    def capture_waveform(self, source: str, **kwargs) -> WaveformCapture:
        y = [2.0, 2.0, 0.0, 0.0] if source == "CH1" else [0.92, 0.82, 0.92, 0.82]
        return WaveformCapture(
            source=source,
            x=[0.0, 1e-6, 2e-6, 3e-6],
            y=y,
            original_points=4,
            plotted_points=4,
            transfer_encoding="binary",
        )


class _MarkerAxis:
    def __init__(self) -> None:
        self.lines: list[float] = []
        self.labels: list[str] = []
        self.arrows = 0

    def axvline(self, x: float, **kwargs) -> None:
        self.lines.append(x)

    def get_xaxis_transform(self):
        return object()

    def annotate(self, *args, **kwargs) -> None:
        self.arrows += 1

    def text(self, x, y, label: str, **kwargs) -> None:
        self.labels.append(label)


class ScopeCaptureRecoveryTest(unittest.TestCase):
    def test_scope_plot_draws_valid_settling_markers(self) -> None:
        axis = _MarkerAxis()
        metrics = ResponseMetrics(
            overshoot_pct=0.5,
            undershoot_pct=0.7,
            settling_time_s=3e-6,
            oscillations=0,
            score=10.0,
            passed=False,
            overshoot_settling_time_s=2.75e-6,
            undershoot_settling_time_s=3.0e-6,
            settling_analysis_version=15,
        )
        edges = [(2e-6, "rising"), (52e-6, "falling")]

        with mock.patch.object(server, "_scope_input_edge_times_s", return_value=edges):
            server._draw_scope_settling_markers(
                axis,
                channels={},
                x0=0.0,
                x1=105e-6,
                x_scale=1e6,
                metrics=metrics,
            )

        self.assertIn("Ts 3.0 us", axis.labels)
        self.assertIn("Ts 2.8 us", axis.labels)
        self.assertEqual(axis.arrows, 2)
        self.assertEqual(len(axis.lines), 4)

    def test_scope_plot_keeps_valid_os_marker_when_only_us_is_invalid(self) -> None:
        axis = _MarkerAxis()
        metrics = ResponseMetrics(
            overshoot_pct=0.5,
            undershoot_pct=0.7,
            settling_time_s=7.25e-6,
            oscillations=0,
            score=300.0,
            passed=False,
            overshoot_settling_time_s=7.25e-6,
            undershoot_settling_time_s=0.0,
            overshoot_settling_valid=True,
            undershoot_settling_valid=False,
            settling_analysis_version=15,
            pass_reasons=["invalid transient waveform: no reliable final settling dwell for US"],
        )
        edges = [(2e-6, "rising"), (52e-6, "falling")]

        with mock.patch.object(server, "_scope_input_edge_times_s", return_value=edges):
            server._draw_scope_settling_markers(
                axis,
                channels={},
                x0=0.0,
                x1=105e-6,
                x_scale=1e6,
                metrics=metrics,
            )

        self.assertIn("Ts --", axis.labels)
        self.assertIn("Ts 7.2 us", axis.labels)
        self.assertEqual(axis.arrows, 1)

    def test_timeout_flushes_reconnects_and_retries_same_capture(self) -> None:
        scope = _RetryingScope()

        def store_capture(capture_id, captures, timestamp, async_save=False):
            records = {
                capture.source: {
                    "source": capture.source,
                    "x": capture.x,
                    "y": capture.y,
                    "x_unit": "s",
                    "y_unit": "V",
                    "original_points": len(capture.y),
                    "transfer_encoding": "binary",
                    "data_file": "fake_scope.npz",
                }
                for capture in captures
            }
            return "fake_scope.npz", records

        with mock.patch.object(server, "_get_scope_connection", return_value=(scope, False)), mock.patch.object(
            server, "_drop_scope_connection"
        ) as drop, mock.patch.object(
            server, "_store_full_scope_capture", side_effect=store_capture
        ), mock.patch.object(
            server, "_scope_response_metrics_from_plot_channels", return_value=None
        ), mock.patch.object(
            server, "_remember_scope_capture"
        ), mock.patch.object(
            server, "_schedule_scope_png_artifact", return_value=False
        ):
            result = server._capture_scope(
                resource="FAKE::SCOPE",
                channels=["CH1", "CH3"],
                measurements=[],
                points=None,
                function_generator_frequency_hz=10_000.0,
                async_artifacts=True,
                iteration_number=2569,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(scope.acquire_calls, 3)
        self.assertEqual(result["session_retry"], 2)
        self.assertEqual(result["capture_attempts"], 3)
        self.assertEqual(len(result["scope_recovery_attempts"]), 2)
        self.assertEqual(scope.prepare_calls.count(True), 2)
        self.assertEqual(drop.call_count, 2)


if __name__ == "__main__":
    unittest.main()
