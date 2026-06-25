"""Board-controller abstractions for Infineon XDP and ADI PMBus devices."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .pmbus import (
    PmbusDevice,
    VoutModeError,
    decode_vout_mode,
    float_to_linear16,
    linear11_to_float,
    linear16_to_float,
    parse_i2c_address,
)


PMBUS_OPERATION = 0x01
PMBUS_CLEAR_FAULTS = 0x03
PMBUS_STATUS_WORD = 0x79
PMBUS_MFR_ID = 0x99
PMBUS_MFR_MODEL = 0x9A
PMBUS_MFR_REVISION = 0x9B
PMBUS_PAGE = 0x00
PMBUS_VOUT_MODE = 0x20
PMBUS_VOUT_COMMAND = 0x21
PMBUS_READ_VOUT = 0x8B
PMBUS_READ_IOUT = 0x8C


@dataclass
class BoardIdentity:
    address: int
    manufacturer: str = ""
    model: str = ""
    revision: str = ""
    status_word: Optional[int] = None


@dataclass
class PidRegisterMap:
    """Placeholder for board-specific PID register addresses.

    These are intentionally optional until we confirm the exact Infineon/ADI
    controller part number and register/PMBus command map.
    """

    kp_command: Optional[int] = None
    ki_command: Optional[int] = None
    kd_command: Optional[int] = None
    kf_command: Optional[int] = None
    encoding: str = "linear11"


@dataclass
class BoardControllerConfig:
    address: int | str
    name: str = "board"
    pid_map: PidRegisterMap = field(default_factory=PidRegisterMap)

    def resolved_address(self) -> int:
        return parse_i2c_address(self.address)


class BoardController:
    """Safe PMBus board controller base class."""

    def __init__(self, adapter, config: BoardControllerConfig):
        self.adapter = adapter
        self.config = config
        self.device = PmbusDevice(adapter, config.resolved_address())

    def connect(self) -> "BoardController":
        self.adapter.open()
        return self

    def close(self) -> None:
        self.adapter.close()

    def identify(self) -> BoardIdentity:
        identity = BoardIdentity(address=self.device.address)
        identity.manufacturer = _safe_block_ascii(self.device, PMBUS_MFR_ID)
        identity.model = _safe_block_ascii(self.device, PMBUS_MFR_MODEL)
        identity.revision = _safe_block_ascii(self.device, PMBUS_MFR_REVISION)
        try:
            identity.status_word = self.device.read_word(PMBUS_STATUS_WORD)
        except Exception:
            identity.status_word = None
        return identity

    def clear_faults(self) -> None:
        self.device.send_byte(PMBUS_CLEAR_FAULTS)

    def set_page(self, page: int) -> None:
        if not 0 <= page <= 0xFF:
            raise ValueError(f"PMBus page out of byte range: {page}")
        self.device.write_byte(PMBUS_PAGE, page)

    def read_page(self) -> int:
        return self.device.read_byte(PMBUS_PAGE)

    def read_status_word(self) -> int:
        return self.device.read_word(PMBUS_STATUS_WORD)

    def read_operation(self, page: int = 0) -> int:
        self.set_page(page)
        return self.device.read_byte(PMBUS_OPERATION)

    def set_operation(self, value: int, page: int = 0) -> None:
        if not 0 <= value <= 0xFF:
            raise ValueError(f"PMBus OPERATION value out of byte range: {value}")
        self.set_page(page)
        self.device.write_byte(PMBUS_OPERATION, value)

    def read_vout_mode(self, page: int = 0) -> tuple[int, str, int]:
        self.set_page(page)
        raw = self.device.read_byte(PMBUS_VOUT_MODE)
        mode_name, parameter = decode_vout_mode(raw)
        return raw, mode_name, parameter

    def read_vout_command(self, page: int = 0) -> float:
        raw_mode, mode_name, parameter = self.read_vout_mode(page)
        raw = self.device.read_word(PMBUS_VOUT_COMMAND)
        if mode_name != "linear":
            raise VoutModeError(f"VOUT_MODE 0x{raw_mode:02X} is {mode_name}; only LINEAR16 is supported.")
        return linear16_to_float(raw, parameter)

    def read_vout(self, page: int = 0) -> float:
        raw_mode, mode_name, parameter = self.read_vout_mode(page)
        raw = self.device.read_word(PMBUS_READ_VOUT)
        if mode_name != "linear":
            raise VoutModeError(f"VOUT_MODE 0x{raw_mode:02X} is {mode_name}; only LINEAR16 is supported.")
        return linear16_to_float(raw, parameter)

    def read_iout(self, page: int = 0) -> float:
        self.set_page(page)
        return linear11_to_float(self.device.read_word(PMBUS_READ_IOUT))

    def set_vout_command(self, voltage_v: float, page: int = 0) -> int:
        if not 0.0 <= voltage_v <= 2.0:
            raise ValueError(f"Refusing VOUT_COMMAND outside 0-2 V bring-up range: {voltage_v}")
        raw_mode, mode_name, parameter = self.read_vout_mode(page)
        if mode_name != "linear":
            raise VoutModeError(f"VOUT_MODE 0x{raw_mode:02X} is {mode_name}; only LINEAR16 is supported.")
        raw = float_to_linear16(voltage_v, parameter)
        self.device.write_word(PMBUS_VOUT_COMMAND, raw)
        return raw

    def enable_output(self) -> None:
        self.device.write_byte(PMBUS_OPERATION, 0x80)

    def disable_output(self) -> None:
        self.device.write_byte(PMBUS_OPERATION, 0x00)

    def set_pid(self, kp: float, ki: float, kd: float, kf: float) -> None:
        raise NotImplementedError(
            "PID register map is board-specific. Confirm controller part number "
            "and PMBus commands before enabling writes."
        )

    def read_pid(self) -> tuple[float, float, float, float]:
        raise NotImplementedError("PID readback is board-specific.")


class InfineonXdpController(BoardController):
    """Infineon XDP PMBus controller placeholder."""


class AdiPowerController(BoardController):
    """ADI PMBus controller placeholder."""


def create_board_controller(kind: str, adapter, config: BoardControllerConfig) -> BoardController:
    name = kind.strip().lower()
    if name in {"generic", "pmbus", "board"}:
        return BoardController(adapter, config)
    if name in {"infineon", "infineon_xdp", "xdp"}:
        return InfineonXdpController(adapter, config)
    if name in {"adi", "adi_power", "power_studio"}:
        return AdiPowerController(adapter, config)
    raise ValueError(f"Unsupported board controller kind: {kind}")


def _safe_block_ascii(device: PmbusDevice, command: int) -> str:
    try:
        return device.read_block_ascii(command).replace("\x00", "").strip()
    except Exception:
        return ""
