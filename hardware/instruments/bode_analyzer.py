"""OMICRON Lab Bode 100/500 SCPI client.

Bode 100 automation does not expose the USB device as a normal VISA USBTMC
instrument. The Bode Analyzer Suite starts a SCPI server, usually on localhost
port 5025, and Python talks to that server through a VISA TCPIP socket.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .visa_resource import VisaConnectionError, VisaInstrument


@dataclass
class BodeStabilityMargins:
    phase_margin_deg: float | None
    phase_crossover_hz: float | None
    gain_margin_db: float | None
    gain_crossover_hz: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "phase_margin_deg": self.phase_margin_deg,
            "phase_crossover_hz": self.phase_crossover_hz,
            "gain_margin_db": self.gain_margin_db,
            "gain_crossover_hz": self.gain_crossover_hz,
        }


@dataclass
class BodeSweepData:
    frequency_hz: list[float]
    real: list[float]
    imag: list[float]

    @property
    def magnitude(self) -> list[float]:
        return [(re * re + im * im) ** 0.5 for re, im in zip(self.real, self.imag)]

    @property
    def phase_rad(self) -> list[float]:
        import math

        return [math.atan2(im, re) for re, im in zip(self.real, self.imag)]

    @property
    def magnitude_db(self) -> list[float]:
        import math

        return [20.0 * math.log10(max(value, 1e-30)) for value in self.magnitude]

    @property
    def phase_deg(self) -> list[float]:
        import math

        return [math.degrees(value) for value in self.phase_rad]

    @property
    def stability_margins(self) -> BodeStabilityMargins:
        return calculate_stability_margins(self.frequency_hz, self.magnitude_db, self.phase_deg)

    def save_csv(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Frequency_Hz", "Real", "Imag", "Magnitude_dB", "Phase_deg"])
            for row in zip(self.frequency_hz, self.real, self.imag, self.magnitude_db, self.phase_deg):
                writer.writerow(row)
        return out


class BodeScpiClient(VisaInstrument):
    """SCPI client for a running Bode Analyzer Suite SCPI server."""

    @classmethod
    def tcpip_resource(cls, host: str = "127.0.0.1", port: int = 5025) -> str:
        return f"TCPIP::{host}::{int(port)}::SOCKET"

    def __init__(
        self,
        resource_name: str | None = None,
        host: str = "127.0.0.1",
        port: int = 5025,
        timeout_ms: int = 20000,
    ):
        super().__init__(
            resource_name or self.tcpip_resource(host, port),
            timeout_ms=timeout_ms,
            read_termination="\n",
            write_termination="\n",
        )

    def connect(self) -> "BodeScpiClient":
        super().connect()
        return self

    def idn(self) -> str:
        return self.query("*IDN?")

    def lock(self) -> bool:
        return self.query(":SYST:LOCK:REQ?").strip().startswith("1")

    def unlock(self) -> None:
        self.write(":SYST:LOCK:REL")

    def reset_scpi_server(self) -> None:
        self.write("*CLS")
        self.write("*RST")

    def configure_gain_phase(
        self,
        start_hz: float,
        stop_hz: float,
        points: int = 201,
        bandwidth_hz: float = 1000.0,
        source_dbm: float | None = None,
    ) -> None:
        # Gain/phase mode must be created before sweep settings; defining a
        # measurement resets the suite to its defaults.
        self.write(":CALC:PAR:DEF GAIN, DEF")
        self.write(f":SENS:FREQ:STAR {start_hz:.12g}")
        self.write(f":SENS:FREQ:STOP {stop_hz:.12g}")
        self.write(f":SENS:SWE:POIN {int(points)}")
        self.write(":SENS:SWE:TYPE LOG")
        self.write(f":SENS:BAND {bandwidth_hz:.12g}")
        self.write(":CALC:FORM MLOG")
        if source_dbm is not None:
            self.write(f":SOUR:POW {source_dbm:.12g}")

    def run_sweep(self) -> BodeSweepData:
        self.write(":TRIG:SOUR BUS")
        self.write(":INIT:CONT ON")
        self.write(":TRIG:SING")
        self.query("*OPC?")
        freqs = _parse_float_list(self.query(":SENS:FREQ:DATA?"))
        magnitude_db = self._read_formatted_trace("MLOG", len(freqs))
        phase_deg = self._read_formatted_trace("PHAS", len(freqs))
        real, imag = _db_phase_to_complex(magnitude_db, phase_deg)
        return BodeSweepData(freqs, real, imag)

    def _read_formatted_trace(self, form: str, points: int) -> list[float]:
        """Read a formatted trace.

        OMICRON's SCPI examples return formatted data from SDAT as a trace
        vector, not interleaved real/imaginary pairs. Some formats return two
        vectors, but the first vector is the selected display format.
        """

        self.write(f":CALC:FORM {form}")
        data = _parse_float_list(self.query(":CALC:DATA:SDAT?"))
        if len(data) < points:
            raise ValueError(f"Bode returned {len(data)} values for {form}, expected at least {points}.")
        return data[:points]

    def get_error(self) -> str:
        return self.query(":SYST:ERR?")


def _parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.replace("\n", "").split(",") if part.strip()]


def _db_phase_to_complex(magnitude_db: list[float], phase_deg: list[float]) -> tuple[list[float], list[float]]:
    import math

    real = []
    imag = []
    for mag_db, phase in zip(magnitude_db, phase_deg):
        magnitude = 10.0 ** (mag_db / 20.0)
        radians = math.radians(phase)
        real.append(magnitude * math.cos(radians))
        imag.append(magnitude * math.sin(radians))
    return real, imag


def calculate_stability_margins(
    frequency_hz: list[float],
    magnitude_db: list[float],
    phase_deg: list[float],
) -> BodeStabilityMargins:
    """Calculate classical gain and phase margins from sampled Bode data.

    Phase margin is evaluated at the 0 dB gain crossover. Classical loop-gain
    data usually reports phase as a negative angle, so PM = 180 + phase(f_gc).
    Bode Analyzer Suite can report gain/phase phase as a positive margin-like
    angle, in which case the displayed phase at 0 dB is already the PM.

    In this lab setup Bode Analyzer Suite reports phase in the positive
    phase-margin style used by the GUI. Gain margin is therefore evaluated at
    the 0 deg phase crossover:
        GM = -gain(f_phase_0)

    Linear interpolation is performed in log-frequency space, which matches
    logarithmic Bode sweeps better than linear-Hz interpolation.
    """

    points = [
        (freq, mag, phase)
        for freq, mag, phase in zip(frequency_hz, magnitude_db, _unwrap_phase_deg(phase_deg))
        if freq > 0
    ]
    if len(points) < 2:
        return BodeStabilityMargins(None, None, None, None)

    gain_crossover_hz = None
    phase_at_gain_crossover = None
    phase_crossover_hz = None
    gain_at_phase_crossover = None

    for (f0, m0, p0), (f1, m1, p1) in zip(points, points[1:]):
        if gain_crossover_hz is None and _crosses(m0, m1, 0.0):
            gain_crossover_hz, phase_at_gain_crossover = _interpolate_log_frequency(f0, p0, m0, f1, p1, m1, 0.0)
        if phase_crossover_hz is None and _crosses(p0, p1, 0.0):
            phase_crossover_hz, gain_at_phase_crossover = _interpolate_log_frequency(f0, m0, p0, f1, m1, p1, 0.0)
        if gain_crossover_hz is not None and phase_crossover_hz is not None:
            break

    phase_margin = (
        _phase_margin_from_gain_crossover_phase(phase_at_gain_crossover)
        if phase_at_gain_crossover is not None
        else None
    )
    gain_margin = -gain_at_phase_crossover if gain_at_phase_crossover is not None else None
    return BodeStabilityMargins(
        phase_margin_deg=phase_margin,
        phase_crossover_hz=gain_crossover_hz,
        gain_margin_db=gain_margin,
        gain_crossover_hz=phase_crossover_hz,
    )


def _phase_margin_from_gain_crossover_phase(phase_deg: float) -> float:
    if phase_deg >= 0.0:
        return phase_deg
    return 180.0 + phase_deg


def _unwrap_phase_deg(values: list[float]) -> list[float]:
    if not values:
        return []
    unwrapped = [values[0]]
    offset = 0.0
    previous = values[0]
    for value in values[1:]:
        delta = value - previous
        if delta > 180.0:
            offset -= 360.0
        elif delta < -180.0:
            offset += 360.0
        unwrapped.append(value + offset)
        previous = value
    return unwrapped


def _crosses(a: float, b: float, target: float) -> bool:
    return (a - target) == 0.0 or (b - target) == 0.0 or (a - target) * (b - target) < 0.0


def _interpolate_log_frequency(
    f0: float,
    y0: float,
    x0: float,
    f1: float,
    y1: float,
    x1: float,
    target_x: float,
) -> tuple[float, float]:
    import math

    if x1 == x0:
        ratio = 0.0
    else:
        ratio = (target_x - x0) / (x1 - x0)
    log_f = math.log10(f0) + ratio * (math.log10(f1) - math.log10(f0))
    y = y0 + ratio * (y1 - y0)
    return 10.0 ** log_f, y


def bode_usb_driver_status(instance_id: str = r"USB\VID_156D&PID_0010\PN287H") -> dict[str, str]:
    """Return a small PnP status dict for the Bode USB device on Windows."""

    import subprocess

    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            f"$d = Get-PnpDevice -InstanceId '{instance_id}' -ErrorAction SilentlyContinue; "
            "if ($null -eq $d) { 'Present=False' } else { "
            "'Present=True'; 'Status=' + $d.Status; 'Class=' + $d.Class; "
            "'FriendlyName=' + $d.FriendlyName; 'InstanceId=' + $d.InstanceId; "
            "$p = Get-PnpDeviceProperty -InstanceId $d.InstanceId -ErrorAction SilentlyContinue; "
            "$pc = ($p | Where-Object KeyName -eq 'DEVPKEY_Device_ProblemCode').Data; "
            "'ProblemCode=' + $pc }"
        ),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    result: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result
