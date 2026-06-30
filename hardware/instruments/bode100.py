"""Bode 100 SCPI server launcher and VISA connection helper.

The OMICRON Lab Bode 100 is controlled through Bode Analyzer Suite's SCPI
server, not as a raw USBTMC VISA instrument. This module starts the SCPI runner
when needed, waits for the localhost TCP socket, and then opens the VISA socket
resource.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import socket
import subprocess
import time

from .bode_analyzer import BodeScpiClient
from .visa_resource import VisaConnectionError


DEFAULT_BODE100_HOST = "127.0.0.1"
DEFAULT_BODE100_PORT = 5025
DEFAULT_BODE100_SCPI_RUNNER_PATH = (
    r"C:\Program Files\OMICRON\BodeAnalyzerSuite\OmicronLab.VectorNetworkAnalysis.ScpiRunner.exe"
)


class Bode100Error(RuntimeError):
    """Raised when the Bode 100 SCPI server or VISA socket cannot be reached."""


@dataclass
class Bode100Driver:
    serial_number: str | None = None
    host: str | None = None
    port: int | None = None
    scpi_runner_path: str | None = None
    startup_timeout_s: float = 30.0
    visa_resource: str | None = None
    timeout_ms: int = 20000

    def __post_init__(self) -> None:
        self.serial_number = self.serial_number or os.environ.get("BODE100_SERIAL")
        self.host = self.host or os.environ.get("BODE100_HOST", DEFAULT_BODE100_HOST)
        self.port = int(self.port or os.environ.get("BODE100_PORT", DEFAULT_BODE100_PORT))
        self.scpi_runner_path = self.scpi_runner_path or os.environ.get(
            "BODE100_SCPI_RUNNER_PATH", DEFAULT_BODE100_SCPI_RUNNER_PATH
        )
        self.visa_resource = self.visa_resource or os.environ.get("BODE100_VISA_RESOURCE")
        self._client: BodeScpiClient | None = None
        self._process: subprocess.Popen | None = None

    @property
    def resource_name(self) -> str:
        return self.visa_resource or BodeScpiClient.tcpip_resource(self.host, self.port)

    def is_scpi_server_running(self) -> bool:
        try:
            with socket.create_connection((self.host, int(self.port)), timeout=0.5):
                return True
        except OSError:
            return False

    def build_scpi_runner_command(self) -> list[str]:
        if not self.serial_number:
            raise Bode100Error(
                "Bode 100 serial number is required to start ScpiRunner. "
                "Pass --serial or set BODE100_SERIAL."
            )
        return [self.scpi_runner_path, "-s", self.serial_number]

    def ensure_scpi_server(self) -> None:
        if self.is_scpi_server_running():
            return
        if not self.serial_number:
            raise Bode100Error(
                "Bode 100 serial number is required to start ScpiRunner. "
                "Pass --serial or set BODE100_SERIAL."
            )
        runner = Path(self.scpi_runner_path)
        if not runner.exists():
            raise Bode100Error(
                "Bode Analyzer Suite ScpiRunner executable was not found: "
                f"{runner}. Set BODE100_SCPI_RUNNER_PATH or pass --scpi-runner-path."
            )
        if os.name != "nt":
            raise Bode100Error(
                "Automatic Bode Analyzer Suite ScpiRunner startup is only supported on Windows. "
                f"Start the SCPI server manually at {self.host}:{self.port}, then retry."
            )
        command = self.build_scpi_runner_command()
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except Exception as exc:
            raise Bode100Error(f"Failed to start Bode Analyzer Suite ScpiRunner: {exc}") from exc

        deadline = time.monotonic() + float(self.startup_timeout_s)
        while time.monotonic() < deadline:
            if self.is_scpi_server_running():
                return
            time.sleep(0.25)
        raise Bode100Error(
            "Bode Analyzer Suite ScpiRunner was started but no SCPI listener became "
            f"available at {self.host}:{self.port} within {self.startup_timeout_s:.1f} s. "
            "Check the serial number, USB connection, and whether another program owns the device."
        )

    def connect(self) -> "Bode100Driver":
        self.ensure_scpi_server()
        try:
            self._client = BodeScpiClient(resource_name=self.resource_name, timeout_ms=self.timeout_ms)
            self._client.connect()
        except VisaConnectionError:
            raise
        except Exception as exc:
            raise VisaConnectionError(f"Failed to open Bode 100 VISA resource {self.resource_name!r}: {exc}") from exc
        return self

    def query(self, command: str) -> str:
        if self._client is None:
            raise Bode100Error("Bode 100 is not connected. Call connect() first.")
        return self._client.query(command)

    def write(self, command: str) -> None:
        if self._client is None:
            raise Bode100Error("Bode 100 is not connected. Call connect() first.")
        self._client.write(command)

    def identify(self) -> str:
        try:
            return self.query("*IDN?")
        except Exception as exc:
            raise Bode100Error(f"Bode 100 *IDN? query failed on {self.resource_name!r}: {exc}") from exc

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


def bode100_from_environment(**overrides) -> Bode100Driver:
    """Create a Bode100Driver using BODE100_* environment variables."""

    return Bode100Driver(**{key: value for key, value in overrides.items() if value is not None})
