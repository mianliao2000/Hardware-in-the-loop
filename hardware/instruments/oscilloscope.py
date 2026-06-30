"""Generic Tektronix oscilloscope helpers over SCPI/VISA."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable

import numpy as np

from .visa_resource import VisaInstrument


@dataclass
class WaveformCapture:
    source: str
    x: list[float]
    y: list[float]
    x_unit: str = "s"
    y_unit: str = "V"
    original_points: int | None = None
    plotted_points: int | None = None
    transfer_encoding: str | None = None

    def save_csv(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Time_s", f"{self.source}_{self.y_unit}"])
            writer.writerows(zip(self.x, self.y))
        return out


@dataclass
class TektronixOscilloscope(VisaInstrument):
    """Minimal Tektronix MSO/DPO oscilloscope driver."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # MSO5 USBTMC reports VI_ERROR_INP_PROT_VIOL on reads when PyVISA
        # appends LF/CRLF to queries. Let USBTMC EOM terminate writes instead.
        self.write_termination = ""
        self.read_termination = "\n"

    @staticmethod
    def _parse_numeric_response(response: str) -> float:
        """Parse either bare numbers or HEADER ON responses."""

        text = response.strip()
        if " " in text:
            text = text.split()[-1]
        return float(text.strip('"'))

    @staticmethod
    def _parse_curve_ascii(response: str) -> list[float]:
        text = response.strip()
        if " " in text and text.upper().startswith(":CURVE"):
            text = text.split(" ", 1)[1]
        return [float(part) for part in text.replace("\n", "").split(",") if part.strip()]

    @staticmethod
    def _time_axis(num_points: int, xincr: float, xzero: float, pt_off: float = 0.0) -> list[float]:
        return [xzero + (idx - pt_off) * xincr for idx in range(num_points)]

    @staticmethod
    def _scale_y(raw_vals: Iterable[float], ymult: float, yoff: float, yzero: float) -> list[float]:
        return [(val - yoff) * ymult + yzero for val in raw_vals]

    @staticmethod
    def _evenly_downsample(x_vals: list[float], y_vals: list[float], max_points: int) -> tuple[list[float], list[float]]:
        total = min(len(x_vals), len(y_vals))
        if total <= max_points or max_points < 2:
            return x_vals[:total], y_vals[:total]
        sampled_x = []
        sampled_y = []
        last_index = -1
        for idx in range(max_points):
            source_index = round(idx * (total - 1) / (max_points - 1))
            if source_index == last_index:
                continue
            sampled_x.append(x_vals[source_index])
            sampled_y.append(y_vals[source_index])
            last_index = source_index
        return sampled_x, sampled_y

    def set_waveform_source(self, source: str = "CH1") -> None:
        self.write(f"DATA:SOURCE {source}")

    def set_channel_display(self, source: str = "CH1", enabled: bool = True) -> None:
        self.write(f"DISPLAY:WAVEVIEW1:{source}:STATE {1 if enabled else 0}")

    def start_acquisition(self) -> None:
        try:
            self.write("ACQUIRE:STOPAFTER RUNSTOP")
        except Exception:
            pass
        self.write("ACQUIRE:STATE RUN")

    def stop_acquisition(self) -> None:
        self.write("ACQUIRE:STATE STOP")

    def force_trigger(self) -> None:
        last_error: Exception | None = None
        for command in ("TRIGGER FORCE", "TRIGger FORCe", "TRIG FORC"):
            try:
                self.write(command)
                return
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not force oscilloscope trigger: {last_error}")

    def set_edge_trigger(self, source: str = "CH1", slope: str = "RISE") -> None:
        safe_source = source.strip().upper()
        safe_slope = slope.strip().upper()
        if safe_slope not in {"RISE", "FALL", "EITHER"}:
            raise ValueError("Trigger slope must be RISE, FALL, or EITHER.")
        commands = (
            "TRIGGER:A:TYPE EDGE",
            f"TRIGGER:A:EDGE:SOURCE {safe_source}",
            f"TRIGGER:A:EDGE:SLOPE {safe_slope}",
        )
        for command in commands:
            self.write(command)

    def set_horizontal_window(self, duration_s: float) -> float:
        """Set the time span shown across the scope grid.

        Tektronix scopes expose horizontal scale as seconds/division. The
        display has ten major horizontal divisions, so one full window is
        approximately 10 * scale.
        """

        if duration_s <= 0:
            raise ValueError("Horizontal window duration must be positive.")
        scale_s_per_div = duration_s / 10.0
        last_error: Exception | None = None
        for command in ("HORIZONTAL:MODE MANUAL", "HOR:MODE MANUAL"):
            try:
                self.write(command)
                break
            except Exception:
                pass

        wrote_scale = False
        for command in ("HORIZONTAL:MODE:SCALE", "HORIZONTAL:SCALE", "HOR:SCA"):
            try:
                self.write(f"{command} {scale_s_per_div:.12g}")
                wrote_scale = True
            except Exception as exc:
                last_error = exc
        if not wrote_scale:
            raise RuntimeError(f"Could not set oscilloscope horizontal scale: {last_error}")

        for query in ("HORIZONTAL:MODE:SCALE?", "HORIZONTAL:SCALE?", "HOR:SCA?"):
            try:
                actual = self._parse_numeric_response(self.query(query))
                if actual > 0:
                    return actual
            except Exception:
                pass
        return scale_s_per_div

    def set_trigger_position_from_left(self, offset_s: float, window_s: float) -> float:
        if offset_s < 0:
            raise ValueError("Trigger offset from left must be non-negative.")
        if window_s <= 0:
            raise ValueError("Horizontal window duration must be positive.")
        position_percent = max(0.0, min(100.0, 100.0 * offset_s / window_s))
        last_error: Exception | None = None
        for command in ("HORIZONTAL:POSITION", "HOR:POS"):
            try:
                self.write(f"{command} {position_percent:.9g}")
                return position_percent
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Could not set oscilloscope trigger position: {last_error}")

    def single_acquisition(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.05,
        force_after_s: float | None = 0.5,
    ) -> None:
        """Run one acquisition sequence and wait until the scope stops."""

        previous_stop_after = None
        try:
            previous_stop_after = self.query("ACQUIRE:STOPAFTER?").strip().strip('"')
        except Exception:
            previous_stop_after = None

        self.write("ACQUIRE:STOPAFTER SEQUENCE")
        self.write("ACQUIRE:STATE RUN")
        started = time.monotonic()
        deadline = started + max(0.1, timeout_s)
        force_deadline = None if force_after_s is None else started + max(0.05, force_after_s)
        forced = False
        try:
            while time.monotonic() < deadline:
                try:
                    state = self.query("ACQUIRE:STATE?").strip().strip('"').upper()
                    if state in {"0", "OFF", "STOP", "STOPPED"}:
                        break
                except Exception:
                    pass
                if not forced and force_deadline is not None and time.monotonic() >= force_deadline:
                    self.force_trigger()
                    forced = True
                time.sleep(max(0.01, poll_interval_s))
            else:
                self.write("ACQUIRE:STATE STOP")
                raise TimeoutError(f"Scope single acquisition did not complete within {timeout_s:.1f} s.")
        finally:
            if previous_stop_after:
                try:
                    self.write(f"ACQUIRE:STOPAFTER {previous_stop_after}")
                except Exception:
                    self.write("ACQUIRE:STOPAFTER RUNSTOP")
            else:
                self.write("ACQUIRE:STOPAFTER RUNSTOP")

    def read_immediate_measurement(self, source: str = "CH1", measurement: str = "MEAN") -> float:
        self.write("HEADER OFF")
        self.write("VERBOSE OFF")
        self.write(f"MEASU:IMM:SOU1 {source}")
        self.write(f"MEASU:IMM:TYP {measurement}")
        return self._parse_numeric_response(self.query("MEASU:IMM:VAL?"))

    def waveform_record_points(self, source: str = "CH1") -> int:
        self.set_waveform_source(source)
        for command in ("HORIZONTAL:RECORDLENGTH?", "HOR:RECO?", "WFMOutpre:NR_Pt?"):
            try:
                return max(1, int(round(self._parse_numeric_response(self.query(command)))))
            except Exception:
                continue
        return 10000

    def capture_ascii_waveform(
        self,
        source: str = "CH1",
        start: int = 1,
        stop: int | None = None,
        max_plot_points: int | None = 1_000_000,
    ) -> WaveformCapture:
        """Capture waveform data as scaled ASCII points.

        Tektronix scopes expose waveform preamble values as:
        XINCR, XZERO, YMULT, YOFF, YZERO. Raw curve points are converted with:
        y = (raw - YOFF) * YMULT + YZERO
        x = XZERO + index * XINCR
        """

        self.set_waveform_source(source)
        self.write("HEADER OFF")
        self.write("VERBOSE OFF")
        self.set_channel_display(source, True)
        self.write("DATA:ENCdg ASCii")
        self.write("DATA:WIDTH 1")
        if stop is None:
            stop = self.waveform_record_points(source)
        self.write(f"DATA:START {int(start)}")
        self.write(f"DATA:STOP {int(stop)}")

        xincr = self._parse_numeric_response(self.query("WFMOutpre:XINcr?"))
        xzero = self._parse_numeric_response(self.query("WFMOutpre:XZEro?"))
        pt_off = self._parse_numeric_response(self.query("WFMOutpre:PT_Off?"))
        ymult = self._parse_numeric_response(self.query("WFMOutpre:YMUlt?"))
        yoff = self._parse_numeric_response(self.query("WFMOutpre:YOFf?"))
        yzero = self._parse_numeric_response(self.query("WFMOutpre:YZEro?"))
        xunit = self.query("WFMOutpre:XUNit?").strip('"')
        yunit = self.query("WFMOutpre:YUNit?").strip('"')

        raw_vals = self._parse_curve_ascii(self.query("CURVE?"))
        x_vals = self._time_axis(len(raw_vals), xincr, xzero, pt_off)
        y_vals = self._scale_y(raw_vals, ymult, yoff, yzero)
        original_points = len(y_vals)
        if max_plot_points is not None:
            x_vals, y_vals = self._evenly_downsample(x_vals, y_vals, max_plot_points)
        return WaveformCapture(
            source=source,
            x=x_vals,
            y=y_vals,
            x_unit=xunit,
            y_unit=yunit,
            original_points=original_points,
            plotted_points=len(y_vals),
            transfer_encoding="ascii",
        )

    def capture_binary_waveform(
        self,
        source: str = "CH1",
        start: int = 1,
        stop: int | None = None,
        max_plot_points: int | None = 1_000_000,
        width: int = 1,
    ) -> WaveformCapture:
        """Capture waveform data with Tektronix binary block transfer.

        Binary transfer avoids the large ASCII string returned by CURVE? and
        lets numpy convert the raw byte block directly into scaled voltages.
        """

        if self._inst is None:
            raise RuntimeError("Instrument is not connected.")
        if width not in {1, 2}:
            raise ValueError("Binary waveform width must be 1 or 2 bytes.")

        self.set_waveform_source(source)
        self.write("HEADER OFF")
        self.write("VERBOSE OFF")
        self.set_channel_display(source, True)
        self.write("DATA:ENCdg RIBinary")
        self.write(f"DATA:WIDTH {width}")
        if width == 2:
            self.write("WFMOutpre:BYT_Or MSB")
        if stop is None:
            stop = self.waveform_record_points(source)
        self.write(f"DATA:START {int(start)}")
        self.write(f"DATA:STOP {int(stop)}")

        xincr = self._parse_numeric_response(self.query("WFMOutpre:XINcr?"))
        xzero = self._parse_numeric_response(self.query("WFMOutpre:XZEro?"))
        pt_off = self._parse_numeric_response(self.query("WFMOutpre:PT_Off?"))
        ymult = self._parse_numeric_response(self.query("WFMOutpre:YMUlt?"))
        yoff = self._parse_numeric_response(self.query("WFMOutpre:YOFf?"))
        yzero = self._parse_numeric_response(self.query("WFMOutpre:YZEro?"))
        xunit = self.query("WFMOutpre:XUNit?").strip('"')
        yunit = self.query("WFMOutpre:YUNit?").strip('"')

        datatype = "b" if width == 1 else "h"
        raw_vals = self._inst.query_binary_values(
            "CURVE?",
            datatype=datatype,
            is_big_endian=True,
            container=np.array,
            expect_termination=False,
        )
        raw_vals = raw_vals.astype(np.float64, copy=False)
        total = int(raw_vals.size)
        if total == 0:
            return WaveformCapture(
                source=source,
                x=[],
                y=[],
                x_unit=xunit,
                y_unit=yunit,
                original_points=0,
                plotted_points=0,
                transfer_encoding="binary",
            )

        if max_plot_points is not None and total > max_plot_points >= 2:
            sample_indices = np.rint(np.linspace(0, total - 1, max_plot_points)).astype(np.int64)
            raw_vals = raw_vals[sample_indices]
            indices = sample_indices.astype(np.float64, copy=False)
        else:
            indices = np.arange(total, dtype=np.float64)
        x_vals = xzero + (indices - pt_off) * xincr
        y_vals = (raw_vals - yoff) * ymult + yzero
        x_list = x_vals.tolist()
        y_list = y_vals.tolist()
        return WaveformCapture(
            source=source,
            x=x_list,
            y=y_list,
            x_unit=xunit,
            y_unit=yunit,
            original_points=total,
            plotted_points=len(y_list),
            transfer_encoding="binary",
        )

    def capture_waveform(
        self,
        source: str = "CH1",
        start: int = 1,
        stop: int | None = None,
        max_plot_points: int | None = 1_000_000,
    ) -> WaveformCapture:
        try:
            return self.capture_binary_waveform(source, start=start, stop=stop, max_plot_points=max_plot_points)
        except Exception:
            return self.capture_ascii_waveform(source, start=start, stop=stop, max_plot_points=max_plot_points)
