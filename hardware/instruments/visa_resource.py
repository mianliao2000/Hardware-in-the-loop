"""Small PyVISA wrapper used by all USB/GPIB/LAN instruments."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import threading
from typing import Iterable, Optional


_RESOURCE_MANAGER_LOCK = threading.RLock()
_RESOURCE_MANAGERS: dict[str, object] = {}


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
            return tuple(rm.list_resources())
        except Exception as exc:
            last_exc = exc
    raise VisaConnectionError(
        "Could not list VISA resources. Check that NI-VISA, Keysight IO "
        "Libraries, or another VISA backend is installed."
    ) from last_exc


def _resource_managers(pyvisa):
    """Yield shared native VISA first, then pyvisa-py as a fallback."""

    for backend in (None, "@py"):
        try:
            yield _shared_resource_manager(pyvisa, backend)
        except Exception:
            continue


def _shared_resource_manager(pyvisa, backend: str | None):
    """Return one process-wide manager per VISA backend.

    NI-VISA ResourceManager objects can share the same underlying default
    session. Closing a short-lived manager may therefore invalidate unrelated
    cached instrument handles. Keep the managers alive for the process and
    close only individual instrument resources during normal operation.
    """

    key = backend or "native"
    with _RESOURCE_MANAGER_LOCK:
        manager = _RESOURCE_MANAGERS.get(key)
        if manager is not None and _resource_manager_is_open(manager):
            return manager

        manager = pyvisa.ResourceManager() if backend is None else pyvisa.ResourceManager(backend)
        _RESOURCE_MANAGERS[key] = manager
        return manager


def _resource_manager_is_open(manager) -> bool:
    try:
        return manager.session is not None
    except Exception:
        return False


def close_shared_resource_managers() -> None:
    """Close process-wide VISA managers during interpreter shutdown."""

    with _RESOURCE_MANAGER_LOCK:
        managers = tuple(_RESOURCE_MANAGERS.values())
        _RESOURCE_MANAGERS.clear()
    for manager in managers:
        _safe_close(manager)


atexit.register(close_shared_resource_managers)


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
        # The ResourceManager is process-wide. Closing it here can invalidate
        # cached Bode/Scope sessions when a short-lived AFG object is released.
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
