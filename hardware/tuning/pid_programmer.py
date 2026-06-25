"""PID programming interfaces.

The real XDP/I2C register path is intentionally disabled until the controller
PID register map is verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import PidParameters


class PidProgrammer(Protocol):
    def is_available(self) -> bool:
        ...

    def read_pid(self) -> PidParameters | None:
        ...

    def write_pid(self, pid: PidParameters) -> None:
        ...

    def status(self) -> dict:
        ...


@dataclass
class StubPidProgrammer:
    reason: str = "XDP/I2C PID register map has not been verified yet."
    write_attempts: int = 0

    def is_available(self) -> bool:
        return False

    def read_pid(self) -> PidParameters | None:
        return None

    def write_pid(self, pid: PidParameters) -> None:
        self.write_attempts += 1
        raise RuntimeError("PID programming is disabled in stub mode.")

    def status(self) -> dict:
        return {
            "available": False,
            "mode": "stub",
            "disabled": True,
            "message": self.reason,
            "write_attempts": self.write_attempts,
        }
