"""PMBus helpers built on top of a small I2C adapter interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class I2cAdapter(Protocol):
    """Minimal I2C adapter contract used by board controllers."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def write(self, address: int, data: bytes) -> None: ...

    def read(self, address: int, length: int) -> bytes: ...

    def write_read(self, address: int, write_data: bytes, read_length: int) -> bytes: ...


class PmbusError(RuntimeError):
    """Raised when a PMBus operation fails."""


class VoutModeError(PmbusError):
    """Raised when a PMBus VOUT_MODE byte cannot be handled safely."""


@dataclass
class PmbusDevice:
    adapter: I2cAdapter
    address: int

    def write_command(self, command: int) -> None:
        self.adapter.write(self.address, bytes([command & 0xFF]))

    def send_byte(self, command: int) -> None:
        self.write_command(command)

    def write_byte(self, command: int, value: int) -> None:
        self.adapter.write(self.address, bytes([command & 0xFF, value & 0xFF]))

    def read_byte(self, command: int) -> int:
        return self.adapter.write_read(self.address, bytes([command & 0xFF]), 1)[0]

    def write_word(self, command: int, value: int) -> None:
        self.adapter.write(
            self.address,
            bytes([command & 0xFF, value & 0xFF, (value >> 8) & 0xFF]),
        )

    def read_word(self, command: int) -> int:
        data = self.adapter.write_read(self.address, bytes([command & 0xFF]), 2)
        return data[0] | (data[1] << 8)

    def read_block(self, command: int, max_length: int = 255) -> bytes:
        data = self.adapter.write_read(self.address, bytes([command & 0xFF]), max_length + 1)
        if not data:
            return b""
        count = min(data[0], max_length, len(data) - 1)
        return data[1 : 1 + count]

    def read_block_ascii(self, command: int, max_length: int = 64) -> str:
        return self.read_block(command, max_length=max_length).decode("ascii", errors="replace").strip()


def linear11_to_float(raw: int) -> float:
    """Decode PMBus LINEAR11 two-byte value."""

    raw &= 0xFFFF
    mantissa = raw & 0x07FF
    exponent = (raw >> 11) & 0x1F
    if mantissa & 0x0400:
        mantissa -= 0x0800
    if exponent & 0x10:
        exponent -= 0x20
    return float(mantissa) * (2.0 ** exponent)


def float_to_linear11(value: float) -> int:
    """Encode a float as PMBus LINEAR11 with a compact exponent search."""

    if value == 0:
        return 0
    best_raw = 0
    best_error = float("inf")
    for exponent in range(-16, 16):
        mantissa = round(value / (2.0 ** exponent))
        if -1024 <= mantissa <= 1023:
            decoded = mantissa * (2.0 ** exponent)
            error = abs(decoded - value)
            if error < best_error:
                encoded_mantissa = mantissa & 0x07FF
                encoded_exponent = exponent & 0x1F
                best_raw = encoded_mantissa | (encoded_exponent << 11)
                best_error = error
    return best_raw


def decode_vout_mode(mode: int) -> tuple[str, int]:
    """Decode PMBus VOUT_MODE into mode name and signed exponent/parameter."""

    mode &= 0xFF
    mode_type = (mode >> 5) & 0x07
    parameter = mode & 0x1F
    if parameter & 0x10:
        parameter -= 0x20
    if mode_type == 0:
        return "linear", parameter
    if mode_type == 1:
        return "vid", parameter
    if mode_type == 2:
        return "direct", parameter
    raise VoutModeError(f"Unsupported VOUT_MODE 0x{mode:02X}.")


def linear16_to_float(raw: int, exponent: int) -> float:
    """Decode PMBus VOUT LINEAR16 using exponent from VOUT_MODE."""

    return float(raw & 0xFFFF) * (2.0 ** exponent)


def float_to_linear16(value: float, exponent: int) -> int:
    """Encode a voltage as PMBus VOUT LINEAR16 using exponent from VOUT_MODE."""

    raw = round(value / (2.0 ** exponent))
    if not 0 <= raw <= 0xFFFF:
        raise VoutModeError(f"Value {value} cannot be encoded as LINEAR16 with exponent {exponent}.")
    return int(raw)


def parse_i2c_address(value: str | int) -> int:
    if isinstance(value, int):
        return value
    text = value.strip().lower()
    base = 16 if text.startswith("0x") else 10
    address = int(text, base)
    if not 0 <= address <= 0x7F:
        raise ValueError(f"I2C address out of 7-bit range: {value}")
    return address
