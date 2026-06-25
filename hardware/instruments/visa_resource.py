"""Small PyVISA wrapper used by all USB/GPIB/LAN instruments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


class VisaConnectionError(RuntimeError):
    """Raised when PyVISA or a VISA instrument cannot be reached."""


def _load_pyvisa():
    try:
        import pyvisa  # type: ignore
    except ImportError as exc:
        raise VisaConnectionError(
            "PyVISA is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc
    return pyvisa


def list_visa_resources() -> tuple[str, ...]:
    """Return all VISA resources visible to any available VISA backend."""

    pyvisa = _load_pyvisa()
    last_exc: Exception | None = None
    for rm in _resource_managers(pyvisa):
        try:
            resources = tuple(rm.list_resources())
            _safe_close(rm)
            return resources
        except Exception as exc:
            last_exc = exc
            _safe_close(rm)
    raise VisaConnectionError(
        "Could not list VISA resources. Check that NI-VISA, Keysight IO "
        "Libraries, or another VISA backend is installed."
    ) from last_exc


def _resource_managers(pyvisa):
    """Yield native VISA first, then pyvisa-py as a fallback."""

    try:
        yield pyvisa.ResourceManager()
    except Exception:
        pass
    try:
        yield pyvisa.ResourceManager("@py")
    except Exception:
        pass


def _safe_close(resource) -> None:
    try:
        resource.close()
    except Exception:
        pass


def find_first_usb_resource(resources: Iterable[str]) -> Optional[str]:
    """Return the first USB instrument resource from a resource list."""

    for resource in resources:
        if resource.upper().startswith("USB"):
            return resource
    return None


@dataclass
class VisaInstrument:
    """Base class for simple SCPI instruments."""

    resource_name: str
    timeout_ms: int = 5000
    read_termination: str = "\n"
    write_termination: str = "\n"

    def __post_init__(self) -> None:
        self._rm = None
        self._inst = None

    @property
    def is_connected(self) -> bool:
        return self._inst is not None

    def connect(self) -> "VisaInstrument":
        pyvisa = _load_pyvisa()
        last_exc: Exception | None = None
        try:
            for rm in _resource_managers(pyvisa):
                try:
                    inst = rm.open_resource(self.resource_name)
                    inst.timeout = self.timeout_ms
                    inst.read_termination = self.read_termination
                    inst.write_termination = self.write_termination
                    self._rm = rm
                    self._inst = inst
                    return self
                except Exception as exc:
                    last_exc = exc
                    _safe_close(rm)
            raise VisaConnectionError(f"Could not open VISA resource {self.resource_name!r}") from last_exc
        except Exception as exc:
            raise VisaConnectionError(f"Could not open VISA resource {self.resource_name!r}") from exc
        return self

    def close(self) -> None:
        if self._inst is not None:
            try:
                self._inst.close()
            except Exception:
                pass
            self._inst = None
        if self._rm is not None:
            _safe_close(self._rm)
            self._rm = None

    def write(self, command: str) -> None:
        if self._inst is None:
            raise VisaConnectionError("Instrument is not connected.")
        self._inst.write(command)

    def query(self, command: str) -> str:
        if self._inst is None:
            raise VisaConnectionError("Instrument is not connected.")
        return str(self._inst.query(command)).strip()

    def idn(self) -> str:
        return self.query("*IDN?")

    def reset(self) -> None:
        self.write("*RST")

    def clear_status(self) -> None:
        self.write("*CLS")
