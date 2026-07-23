"""OMICRON Lab Bode 100/500 SCPI client.

Bode 100 automation does not expose the USB device as a normal VISA USBTMC
instrument. The Bode Analyzer Suite starts a SCPI server, usually on localhost
port 5025, and Python talks to that server through a VISA TCPIP socket.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics


GAIN_SHAPE_START_HZ = 2_000.0
GAIN_SHAPE_STOP_HZ = 900_000.0
GAIN_SHAPE_WINDOW_DECADES = 0.12
GAIN_SHAPE_FLAT_SLOPE_DB_PER_DECADE = -4.0
GAIN_SHAPE_ALLOWED_REBOUND_DB = 0.25
GAIN_SHAPE_ALLOWED_FLAT_SPAN_DECADES = 0.12
GAIN_SHAPE_MAX_REBOUND_DB = 1.00
GAIN_SHAPE_MAX_FLAT_SPAN_DECADES = 0.18
GAIN_SHAPE_REBOUND_PENALTY_PER_DB = 12.0
GAIN_SHAPE_FLAT_PENALTY_PER_DECADE = 60.0

from .visa_resource import VisaInstrument


@dataclass
class BodeStabilityMargins:
    phase_margin_deg: float | None
    phase_crossover_hz: float | None
    gain_margin_db: float | None
    gain_crossover_hz: float | None
    gain_crossover_count: int = 0
    duplicate_gain_crossover: bool = False
    second_phase_crossover_hz: float | None = None
    gain_rebound_db: float | None = None
    gain_rebound_start_hz: float | None = None
    gain_rebound_stop_hz: float | None = None
    gain_flat_span_decades: float | None = None
    gain_slope_p90_db_per_decade: float | None = None
    gain_shape_penalty: float = 0.0
    gain_shape_valid: bool | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "phase_margin_deg": self.phase_margin_deg,
            "phase_crossover_hz": self.phase_crossover_hz,
            "gain_margin_db": self.gain_margin_db,
            "gain_crossover_hz": self.gain_crossover_hz,
            "gain_crossover_count": self.gain_crossover_count,
            "duplicate_gain_crossover": self.duplicate_gain_crossover,
            "second_phase_crossover_hz": self.second_phase_crossover_hz,
            "gain_rebound_db": self.gain_rebound_db,
            "gain_rebound_start_hz": self.gain_rebound_start_hz,
            "gain_rebound_stop_hz": self.gain_rebound_stop_hz,
            "gain_flat_span_decades": self.gain_flat_span_decades,
            "gain_slope_p90_db_per_decade": self.gain_slope_p90_db_per_decade,
            "gain_shape_penalty": self.gain_shape_penalty,
            "gain_shape_valid": self.gain_shape_valid,
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

    gain_crossovers: list[tuple[float, float]] = []
    phase_crossovers: list[tuple[float, float]] = []

    for (f0, m0, p0), (f1, m1, p1) in zip(points, points[1:]):
        if _crosses(m0, m1, 0.0):
            gain_crossovers.append(_interpolate_log_frequency(f0, p0, m0, f1, p1, m1, 0.0))
        if _crosses(p0, p1, 0.0):
            phase_crossovers.append(_interpolate_log_frequency(f0, m0, p0, f1, m1, p1, 0.0))

    gain_crossovers = _dedupe_crossovers(gain_crossovers)
    phase_crossovers = _dedupe_crossovers(phase_crossovers)

    gain_crossover_hz = gain_crossovers[0][0] if gain_crossovers else None
    phase_at_gain_crossover = gain_crossovers[0][1] if gain_crossovers else None
    phase_crossover_hz = phase_crossovers[0][0] if phase_crossovers else None
    gain_at_phase_crossover = phase_crossovers[0][1] if phase_crossovers else None

    phase_margin = (
        _phase_margin_from_gain_crossover_phase(phase_at_gain_crossover)
        if phase_at_gain_crossover is not None
        else None
    )
    gain_margin = -gain_at_phase_crossover if gain_at_phase_crossover is not None else None
    gain_shape = calculate_gain_shape(frequency_hz, magnitude_db)
    return BodeStabilityMargins(
        phase_margin_deg=phase_margin,
        phase_crossover_hz=gain_crossover_hz,
        gain_margin_db=gain_margin,
        gain_crossover_hz=phase_crossover_hz,
        gain_crossover_count=len(gain_crossovers),
        duplicate_gain_crossover=len(gain_crossovers) > 1,
        second_phase_crossover_hz=gain_crossovers[1][0] if len(gain_crossovers) > 1 else None,
        **gain_shape,
    )


def calculate_gain_shape(
    frequency_hz: list[float],
    magnitude_db: list[float],
) -> dict[str, float | bool | None]:
    """Measure sustained gain rebound and flat spans on a logarithmic axis.

    A seven-point median followed by a three-point mean suppresses isolated
    Bode100 noise without hiding the broad 150--500 kHz rebound seen on the
    bench. Slopes use 0.12-decade windows, so a single sample can neither fail
    nor rescue the shape test.
    """

    points = sorted(
        (float(frequency), float(gain))
        for frequency, gain in zip(frequency_hz, magnitude_db)
        if math.isfinite(float(frequency))
        and math.isfinite(float(gain))
        and GAIN_SHAPE_START_HZ <= float(frequency) <= GAIN_SHAPE_STOP_HZ
    )
    if len(points) < 20:
        return {
            "gain_rebound_db": None,
            "gain_rebound_start_hz": None,
            "gain_rebound_stop_hz": None,
            "gain_flat_span_decades": None,
            "gain_slope_p90_db_per_decade": None,
            "gain_shape_penalty": 0.0,
            "gain_shape_valid": None,
        }
    log_frequency = [math.log10(frequency) for frequency, _ in points]
    gain = [value for _, value in points]
    median_gain: list[float] = []
    half_window = 3
    for index in range(len(gain)):
        start = max(0, index - half_window)
        stop = min(len(gain), index + half_window + 1)
        median_gain.append(float(statistics.median(gain[start:stop])))
    smoothed_gain = [
        sum(median_gain[max(0, index - 1) : min(len(median_gain), index + 2)])
        / len(median_gain[max(0, index - 1) : min(len(median_gain), index + 2)])
        for index in range(len(median_gain))
    ]

    running_minimum = smoothed_gain[0]
    running_minimum_index = 0
    rebound_db = 0.0
    rebound_start_index = 0
    rebound_stop_index = 0
    for index, value in enumerate(smoothed_gain):
        if value < running_minimum:
            running_minimum = value
            running_minimum_index = index
        drawup = value - running_minimum
        if drawup > rebound_db:
            rebound_db = drawup
            rebound_start_index = running_minimum_index
            rebound_stop_index = index

    slopes: list[tuple[float, float, float]] = []
    for start_index, start_log_frequency in enumerate(log_frequency):
        stop_index = start_index + 1
        while (
            stop_index < len(log_frequency)
            and log_frequency[stop_index] - start_log_frequency < GAIN_SHAPE_WINDOW_DECADES
        ):
            stop_index += 1
        if stop_index >= len(log_frequency):
            break
        width = log_frequency[stop_index] - start_log_frequency
        slopes.append(
            (
                start_log_frequency,
                log_frequency[stop_index],
                (smoothed_gain[stop_index] - smoothed_gain[start_index]) / max(width, 1e-12),
            )
        )

    flat_span_decades = 0.0
    flat_start: float | None = None
    for start_log_frequency, stop_log_frequency, slope in slopes:
        if slope > GAIN_SHAPE_FLAT_SLOPE_DB_PER_DECADE:
            if flat_start is None:
                flat_start = start_log_frequency
        elif flat_start is not None:
            flat_span_decades = max(flat_span_decades, start_log_frequency - flat_start)
            flat_start = None
    if flat_start is not None and slopes:
        flat_span_decades = max(flat_span_decades, slopes[-1][1] - flat_start)
    sorted_slopes = sorted(slope for _, _, slope in slopes)
    if sorted_slopes:
        percentile_index = min(len(sorted_slopes) - 1, int(math.ceil(0.90 * len(sorted_slopes))) - 1)
        slope_p90 = sorted_slopes[percentile_index]
    else:
        slope_p90 = None

    shape_penalty = (
        GAIN_SHAPE_REBOUND_PENALTY_PER_DB
        * max(0.0, rebound_db - GAIN_SHAPE_ALLOWED_REBOUND_DB)
        + GAIN_SHAPE_FLAT_PENALTY_PER_DECADE
        * max(0.0, flat_span_decades - GAIN_SHAPE_ALLOWED_FLAT_SPAN_DECADES)
    )
    shape_valid = bool(
        rebound_db <= GAIN_SHAPE_MAX_REBOUND_DB
        and flat_span_decades <= GAIN_SHAPE_MAX_FLAT_SPAN_DECADES
    )
    return {
        "gain_rebound_db": float(rebound_db),
        "gain_rebound_start_hz": float(points[rebound_start_index][0]),
        "gain_rebound_stop_hz": float(points[rebound_stop_index][0]),
        "gain_flat_span_decades": float(flat_span_decades),
        "gain_slope_p90_db_per_decade": float(slope_p90) if slope_p90 is not None else None,
        "gain_shape_penalty": float(shape_penalty),
        "gain_shape_valid": shape_valid,
    }


def _phase_margin_from_gain_crossover_phase(phase_deg: float) -> float:
    # A single low-SNR point near the start of a sweep can make sequential
    # unwrapping choose a branch offset by one or more full turns. Phase
    # margin is periodic in 360 degrees, so canonicalize the interpolated
    # crossover phase before distinguishing Bode100's positive margin-style
    # phase from a classical negative loop phase.
    canonical = (float(phase_deg) + 180.0) % 360.0 - 180.0
    if canonical >= 0.0:
        return canonical
    return 180.0 + canonical


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


def _dedupe_crossovers(crossovers: list[tuple[float, float]]) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for frequency, value in crossovers:
        if not deduped:
            deduped.append((frequency, value))
            continue
        previous_frequency = deduped[-1][0]
        if previous_frequency > 0 and abs(frequency - previous_frequency) / previous_frequency < 1e-6:
            continue
        deduped.append((frequency, value))
    return deduped


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
