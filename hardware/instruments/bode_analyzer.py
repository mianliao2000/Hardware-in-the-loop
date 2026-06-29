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
