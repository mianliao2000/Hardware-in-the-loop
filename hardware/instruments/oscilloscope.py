"""Generic Tektronix oscilloscope helpers over SCPI/VISA."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .visa_resource import VisaInstrument


@dataclass
class WaveformCapture:
    source: str
    x: list[float]
    y: list[float]
    x_unit: str = "s"
    y_unit: str = "V"

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

    def set_waveform_source(self, source: str = "CH1") -> None:
        self.write(f"DATA:SOURCE {source}")

    def set_channel_display(self, source: str = "CH1", enabled: bool = True) -> None:
        self.write(f"DISPLAY:WAVEVIEW1:{source}:STATE {1 if enabled else 0}")

    def read_immediate_measurement(self, source: str = "CH1", measurement: str = "MEAN") -> float:
        self.write("HEADER OFF")
        self.write("VERBOSE OFF")
        self.write(f"MEASU:IMM:SOU1 {source}")
        self.write(f"MEASU:IMM:TYP {measurement}")
        return self._parse_numeric_response(self.query("MEASU:IMM:VAL?"))

    def capture_ascii_waveform(self, source: str = "CH1", start: int = 1, stop: int = 10000) -> WaveformCapture:
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
        return WaveformCapture(source=source, x=x_vals, y=y_vals, x_unit=xunit, y_unit=yunit)
