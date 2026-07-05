"""I2C adapter implementations for PMBus board control."""

from __future__ import annotations

import atexit
from ctypes import POINTER, WinDLL, c_int, c_ubyte, c_ushort
from dataclasses import dataclass, field
from importlib.util import find_spec
import json
from pathlib import Path
import subprocess
import threading
from typing import Literal


class I2cAdapterError(RuntimeError):
    """Raised when an I2C adapter cannot perform an operation."""


@dataclass
class MockI2cAdapter:
    """In-memory adapter for dry runs and script validation."""

    responses: dict[tuple[int, int, int], bytes] = field(default_factory=dict)
    writes: list[tuple[int, bytes]] = field(default_factory=list)
    opened: bool = False

    def open(self) -> None:
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def write(self, address: int, data: bytes) -> None:
        self._require_open()
        self.writes.append((address, bytes(data)))
        if len(data) >= 2:
            command = data[0]
            payload = bytes(data[1:])
            self.responses[(address, command, len(payload))] = payload

    def read(self, address: int, length: int) -> bytes:
        self._require_open()
        return bytes([0x00] * length)

    def write_read(self, address: int, write_data: bytes, read_length: int) -> bytes:
        self._require_open()
        command = write_data[0] if write_data else 0
        return self.responses.get((address, command, read_length), bytes([0x00] * read_length))

    def _require_open(self) -> None:
        if not self.opened:
            raise I2cAdapterError("Mock I2C adapter is not open.")


@dataclass
class AardvarkI2cAdapter:
    """Total Phase Aardvark I2C adapter using the vendor DLL via ctypes."""

    port: int = 0
    bitrate_khz: int = 100
    enable_pullups: bool = False
    handle: int | None = None
    dll_path: str | None = None

    def __post_init__(self) -> None:
        self._aa = None

    def open(self) -> None:
        aa = _AardvarkDll.load(self.dll_path)
        handle = aa.open(self.port)
        if handle <= 0:
            raise I2cAdapterError(f"Could not open Aardvark on port {self.port}: {handle}")
        self.handle = handle
        self._aa = aa
        aa.configure(handle, _AardvarkDll.AA_CONFIG_SPI_I2C)
        aa.i2c_pullup(handle, _AardvarkDll.AA_I2C_PULLUP_BOTH if self.enable_pullups else 0)
        aa.i2c_bitrate(handle, int(self.bitrate_khz))

    def close(self) -> None:
        if self.handle is not None and self._aa is not None:
            self._aa.close(self.handle)
        self.handle = None

    def write(self, address: int, data: bytes) -> None:
        self._require_open()
        count = self._aa.i2c_write(self.handle, address, bytes(data))
        if count < 0:
            raise I2cAdapterError(f"Aardvark I2C write failed: {count}")
        if count != len(data):
            raise I2cAdapterError(f"Aardvark I2C write incomplete: {count}/{len(data)} bytes")

    def read(self, address: int, length: int) -> bytes:
        self._require_open()
        count, data = self._aa.i2c_read(self.handle, address, int(length))
        if count < 0:
            raise I2cAdapterError(f"Aardvark I2C read failed: {count}")
        return bytes(data[:count])

    def write_read(self, address: int, write_data: bytes, read_length: int) -> bytes:
        self.write(address, write_data)
        return self.read(address, read_length)

    def _require_open(self) -> None:
        if self.handle is None or self._aa is None:
            raise I2cAdapterError("Aardvark I2C adapter is not open.")


def create_i2c_adapter(kind: str, **kwargs):
    name = kind.strip().lower()
    if name == "mock":
        return MockI2cAdapter(**kwargs)
    if name == "aardvark":
        return AardvarkI2cAdapter(**kwargs)
    if name in {"xdp", "xdp_usb"}:
        return XdpNodeUsbI2cAdapter(**kwargs)
    if name in {"xdp_pyusb", "xdp_usb_pyusb"}:
        return XdpUsbI2cAdapter(**kwargs)
    raise ValueError(f"Unsupported I2C adapter kind: {kind}")


@dataclass
class XdpNodeUsbI2cAdapter:
    """Infineon XDP USB dongle adapter using XDP Designer's Node USB stack."""

    timeout_ms: int = 1000
    address_mode: Literal["xdp_8bit", "7bit"] = "xdp_8bit"
    node_exe: str = "node"
    bridge_path: str | None = None
    persistent: bool = True
    opened: bool = False

    def open(self) -> None:
        self.identify_dongle()
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def write(self, address: int, data: bytes) -> None:
        self._require_open()
        self._transfer(address, data, 0)

    def read(self, address: int, length: int) -> bytes:
        self._require_open()
        return self._transfer(address, b"", int(length))

    def write_read(self, address: int, write_data: bytes, read_length: int) -> bytes:
        self._require_open()
        return self._transfer(address, bytes(write_data), int(read_length))

    def read_memory_word(self, address: int, memory_address: int, word_size: int = 4) -> int:
        self._require_open()
        result = self._run_bridge(
            [
                "memory-read",
                "--address",
                str(address),
                "--memory-address",
                str(memory_address),
                "--word-size",
                str(word_size),
                "--timeout-ms",
                str(self.timeout_ms),
            ],
            {
                "command": "memory-read",
                "address": address,
                "memoryAddress": memory_address,
                "wordSize": word_size,
            },
        )
        value = result.get("value")
        if value is None:
            data = _parse_hex_bytes(result.get("data", ""))
            return _little_endian_to_int(data)
        return int(value)

    def write_memory_word(self, address: int, memory_address: int, value: int, word_size: int = 4) -> None:
        self._require_open()
        self._run_bridge(
            [
                "memory-write",
                "--address",
                str(address),
                "--memory-address",
                str(memory_address),
                "--value",
                str(value),
                "--word-size",
                str(word_size),
                "--timeout-ms",
                str(self.timeout_ms),
            ],
            {
                "command": "memory-write",
                "address": address,
                "memoryAddress": memory_address,
                "value": value,
                "wordSize": word_size,
            },
        )

    def update_memory_bits(
        self,
        address: int,
        memory_address: int,
        start: int,
        length: int,
        field_value: int,
        word_size: int = 4,
    ) -> tuple[int, int]:
        if start < 0 or length <= 0 or start + length > 8 * word_size:
            raise I2cAdapterError(f"Invalid memory bitfield start={start}, length={length}.")
        max_field = (1 << length) - 1
        if not 0 <= field_value <= max_field:
            raise I2cAdapterError(f"Field value 0x{field_value:X} does not fit in {length} bits.")
        before = self.read_memory_word(address, memory_address, word_size=word_size)
        mask = max_field << start
        after = (before & ~mask) | ((field_value & max_field) << start)
        if after != before:
            self.write_memory_word(address, memory_address, after, word_size=word_size)
        return before, after

    def identify_dongle(self) -> bytes:
        result = self._run_bridge(["identify"], {"command": "identify"})
        return _parse_hex_bytes(result.get("data", ""))

    def _transfer(self, address: int, write_data: bytes, read_length: int) -> bytes:
        result = self._run_bridge(
            [
                "transfer",
                "--address",
                str(address),
                "--write",
                write_data.hex(),
                "--read",
                str(read_length),
                "--address-mode",
                self.address_mode,
                "--timeout-ms",
                str(self.timeout_ms),
            ],
            {
                "command": "transfer",
                "address": address,
                "write": write_data.hex(),
                "read": int(read_length),
                "addressMode": self.address_mode,
            },
        )
        if not result.get("ok", False):
            status = result.get("status")
            raise I2cAdapterError(f"XDP USB I2C transfer failed with status {status}.")
        return _parse_hex_bytes(result.get("data", ""))

    def _run_bridge(self, args: list[str], persistent_payload: dict | None = None) -> dict:
        bridge = Path(self.bridge_path) if self.bridge_path else Path(__file__).with_name("xdp_usb_bridge.js")
        if self.persistent and persistent_payload is not None:
            return _get_persistent_xdp_bridge(self.node_exe, bridge, self.timeout_ms).request(persistent_payload)
        command = [self.node_exe, str(bridge), *args]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(5.0, self.timeout_ms / 1000.0 + 3.0),
        )
        output = completed.stdout.strip() or completed.stderr.strip()
        try:
            result = json.loads(output.splitlines()[-1])
        except Exception as exc:
            raise I2cAdapterError(f"XDP USB bridge returned invalid output: {output}") from exc
        if completed.returncode != 0 or not result.get("ok", False):
            raise I2cAdapterError(result.get("error", output))
        return result

    def _require_open(self) -> None:
        if not self.opened:
            raise I2cAdapterError("XDP USB adapter is not open.")


class _PersistentXdpBridge:
    def __init__(self, node_exe: str, bridge_path: Path, timeout_ms: int):
        self.node_exe = node_exe
        self.bridge_path = bridge_path
        self.timeout_ms = int(timeout_ms)
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None

    def request(self, payload: dict) -> dict:
        retryable = _is_retryable_xdp_payload(payload)
        with self._lock:
            try:
                return self._request_once_locked(payload)
            except I2cAdapterError:
                if not retryable:
                    raise
                return self._request_once_locked(payload)

    def _request_once_locked(self, payload: dict) -> dict:
        try:
            self._ensure_started()
            assert self._process is not None
            self._process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        except Exception as exc:
            self.stop()
            raise I2cAdapterError(f"XDP persistent bridge request failed: {exc}") from exc
        if not line:
            error = self._collect_process_error()
            self.stop()
            raise I2cAdapterError(f"XDP persistent bridge exited unexpectedly. {error}".strip())
        try:
            result = json.loads(line)
        except Exception as exc:
            self.stop()
            raise I2cAdapterError(f"XDP persistent bridge returned invalid output: {line.strip()}") from exc
        if not result.get("ok", False):
            self.stop()
            raise I2cAdapterError(result.get("error", line.strip()))
        return result

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except Exception:
            pass
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self.stop()
        command = [
            self.node_exe,
            str(self.bridge_path),
            "serve",
            "--timeout-ms",
            str(self.timeout_ms),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def _collect_process_error(self) -> str:
        process = self._process
        if process is None or process.stdout is None:
            return ""
        try:
            return process.stdout.read().strip()
        except Exception:
            return ""


def _is_retryable_xdp_payload(payload: dict) -> bool:
    """Allow reconnect retries only for read-only bridge commands."""
    command = payload.get("command")
    if command in {"identify", "memory-read"}:
        return True
    if command == "transfer":
        return int(payload.get("read") or 0) > 0
    return False


_PERSISTENT_XDP_BRIDGES: dict[tuple[str, str, int], _PersistentXdpBridge] = {}
_PERSISTENT_XDP_BRIDGES_LOCK = threading.Lock()


def _get_persistent_xdp_bridge(node_exe: str, bridge_path: Path, timeout_ms: int) -> _PersistentXdpBridge:
    key = (node_exe, str(bridge_path.resolve()), int(timeout_ms))
    with _PERSISTENT_XDP_BRIDGES_LOCK:
        bridge = _PERSISTENT_XDP_BRIDGES.get(key)
        if bridge is None:
            bridge = _PersistentXdpBridge(node_exe, bridge_path, timeout_ms)
            _PERSISTENT_XDP_BRIDGES[key] = bridge
        return bridge


def _stop_persistent_xdp_bridges() -> None:
    with _PERSISTENT_XDP_BRIDGES_LOCK:
        bridges = list(_PERSISTENT_XDP_BRIDGES.values())
        _PERSISTENT_XDP_BRIDGES.clear()
    for bridge in bridges:
        bridge.stop()


def reset_xdp_usb_bridges(*, include_external_processes: bool = False) -> None:
    """Close XDP bridge processes so the next request opens a fresh USB handle."""
    _stop_persistent_xdp_bridges()
    if include_external_processes:
        _stop_external_xdp_bridge_processes()


def _stop_external_xdp_bridge_processes() -> None:
    powershell = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -eq 'node.exe' -and $_.CommandLine -like '*xdp_usb_bridge.js*' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", powershell],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except Exception:
        pass


atexit.register(_stop_persistent_xdp_bridges)


@dataclass
class XdpUsbI2cAdapter:
    """Infineon XDP Designer USB dongle adapter.

    This follows the packet shape used by XDP Designer:
    [payload_len_lo, payload_len_hi, 9, mode, count, addr, write_len, write..., read_len]
    and expects one status byte followed by read data.
    """

    vendor_id: int = 0x10C4
    product_id: int = 0xEA61
    timeout_ms: int = 1000
    address_mode: Literal["xdp_8bit", "7bit"] = "xdp_8bit"

    def __post_init__(self) -> None:
        self.device = None
        self.out_endpoint = None
        self.in_endpoint = None

    def open(self) -> None:
        import usb.core
        import usb.util

        backend = _get_libusb_backend()
        self.device = usb.core.find(idVendor=self.vendor_id, idProduct=self.product_id, backend=backend)
        if self.device is None:
            raise I2cAdapterError(
                f"XDP USB dongle not found at VID:PID={self.vendor_id:04X}:{self.product_id:04X}."
            )

        try:
            self.device.set_configuration()
        except usb.core.USBError:
            pass

        config = self.device.get_active_configuration()
        interface = config[(0, 0)]
        try:
            if self.device.is_kernel_driver_active(interface.bInterfaceNumber):
                self.device.detach_kernel_driver(interface.bInterfaceNumber)
        except (NotImplementedError, usb.core.USBError):
            pass

        try:
            usb.util.claim_interface(self.device, interface.bInterfaceNumber)
        except usb.core.USBError as exc:
            raise I2cAdapterError(
                "Could not claim XDP USB interface. Close XDP Designer and try again."
            ) from exc

        self.out_endpoint = usb.util.find_descriptor(
            interface,
            custom_match=lambda endpoint: usb.util.endpoint_direction(endpoint.bEndpointAddress)
            == usb.util.ENDPOINT_OUT,
        )
        self.in_endpoint = usb.util.find_descriptor(
            interface,
            custom_match=lambda endpoint: usb.util.endpoint_direction(endpoint.bEndpointAddress)
            == usb.util.ENDPOINT_IN,
        )
        if self.out_endpoint is None or self.in_endpoint is None:
            raise I2cAdapterError("XDP USB endpoints were not found.")

        if self.vendor_id == 0x10C4:
            self._silabs_init()

    def close(self) -> None:
        if self.device is not None:
            try:
                import usb.util

                usb.util.dispose_resources(self.device)
            except Exception:
                pass
        self.device = None
        self.out_endpoint = None
        self.in_endpoint = None

    def write(self, address: int, data: bytes) -> None:
        self._read_write([_XdpI2cTransfer(address, bytes(data), 0)])

    def read(self, address: int, length: int) -> bytes:
        return self._read_write([_XdpI2cTransfer(address, b"", int(length))])[0]

    def write_read(self, address: int, write_data: bytes, read_length: int) -> bytes:
        return self._read_write([_XdpI2cTransfer(address, bytes(write_data), int(read_length))])[0]

    def identify_dongle(self) -> bytes:
        self._require_open()
        return self._request(bytes([1, 0, 1]), 6)

    def _read_write(self, transfers: list["_XdpI2cTransfer"]) -> list[bytes]:
        payload = [9, 0, len(transfers)]
        total_read_length = 0
        for transfer in transfers:
            payload.append(self._format_address(transfer.address, transfer.read_length > 0))
            payload.append(len(transfer.write_data))
            payload.extend(transfer.write_data)
            if transfer.read_length:
                payload.append(transfer.read_length)
                total_read_length += transfer.read_length

        response = self._request(bytes(payload), 1 + total_read_length)
        if not response:
            raise I2cAdapterError("XDP USB returned an empty response.")
        status = response[0]
        if status != 0:
            raise I2cAdapterError(f"XDP USB I2C transfer failed with status 0x{status:02X}.")

        chunks = []
        offset = 1
        for transfer in transfers:
            if transfer.read_length:
                end = offset + transfer.read_length
                chunks.append(bytes(response[offset:end]))
                offset = end
            else:
                chunks.append(b"")
        return chunks

    def _request(self, payload: bytes, read_length: int) -> bytes:
        self._require_open()
        packet = bytes([len(payload) & 0xFF, (len(payload) >> 8) & 0xFF]) + payload
        self.out_endpoint.write(packet, timeout=self.timeout_ms)
        if read_length == 0:
            return b""
        return bytes(self.in_endpoint.read(read_length, timeout=self.timeout_ms))

    def _format_address(self, address: int, read: bool) -> int:
        if self.address_mode == "7bit":
            return address & 0x7F
        return ((address & 0x7F) << 1) | (1 if read else 0)

    def _silabs_init(self) -> None:
        data = bytes([0])
        for request_type, request, value, index in ((0, 9, 1, 0), (65, 2, 2, 0), (65, 2, 1, 0)):
            try:
                self.device.ctrl_transfer(request_type, request, value, index, data, timeout=self.timeout_ms)
            except Exception:
                pass

    def _require_open(self) -> None:
        if self.device is None or self.out_endpoint is None or self.in_endpoint is None:
            raise I2cAdapterError("XDP USB adapter is not open.")


@dataclass(frozen=True)
class _XdpI2cTransfer:
    address: int
    write_data: bytes
    read_length: int


def _get_libusb_backend():
    try:
        import libusb_package

        return libusb_package.get_libusb1_backend()
    except Exception:
        return None


def _parse_hex_bytes(text: str) -> bytes:
    if not text:
        return b""
    return bytes(int(part, 16) for part in text.replace(",", " ").split())


def _little_endian_to_int(data: bytes) -> int:
    value = 0
    for index, byte in enumerate(data):
        value |= int(byte) << (8 * index)
    return value


class _AardvarkDll:
    AA_CONFIG_SPI_I2C = 0x03
    AA_I2C_NO_FLAGS = 0x00
    AA_I2C_PULLUP_BOTH = 0x03

    def __init__(self, dll):
        self.dll = dll
        self._bind()

    @classmethod
    def load(cls, dll_path: str | None = None) -> "_AardvarkDll":
        if dll_path is None:
            spec = find_spec("aardvark_py")
            if spec is None or not spec.submodule_search_locations:
                raise I2cAdapterError("aardvark_py package is not installed. Run: python -m pip install aardvark_py")
            dll_path = str(Path(next(iter(spec.submodule_search_locations))) / "aardvark.dll")
        return cls(WinDLL(dll_path))

    def _bind(self) -> None:
        self.dll.c_aa_open.argtypes = [c_int]
        self.dll.c_aa_open.restype = c_int
        self.dll.c_aa_close.argtypes = [c_int]
        self.dll.c_aa_close.restype = c_int
        self.dll.c_aa_configure.argtypes = [c_int, c_int]
        self.dll.c_aa_configure.restype = c_int
        self.dll.c_aa_i2c_bitrate.argtypes = [c_int, c_int]
        self.dll.c_aa_i2c_bitrate.restype = c_int
        self.dll.c_aa_i2c_pullup.argtypes = [c_int, c_ubyte]
        self.dll.c_aa_i2c_pullup.restype = c_int
        self.dll.c_aa_i2c_write.argtypes = [c_int, c_ushort, c_ushort, c_ushort, POINTER(c_ubyte)]
        self.dll.c_aa_i2c_write.restype = c_int
        self.dll.c_aa_i2c_read.argtypes = [c_int, c_ushort, c_ushort, c_ushort, POINTER(c_ubyte)]
        self.dll.c_aa_i2c_read.restype = c_int

    def open(self, port: int) -> int:
        return self.dll.c_aa_open(int(port))

    def close(self, handle: int) -> int:
        return self.dll.c_aa_close(int(handle))

    def configure(self, handle: int, config: int) -> int:
        return self.dll.c_aa_configure(int(handle), int(config))

    def i2c_bitrate(self, handle: int, bitrate_khz: int) -> int:
        return self.dll.c_aa_i2c_bitrate(int(handle), int(bitrate_khz))

    def i2c_pullup(self, handle: int, pullup_mask: int) -> int:
        return self.dll.c_aa_i2c_pullup(int(handle), int(pullup_mask))

    def i2c_write(self, handle: int, address: int, data: bytes) -> int:
        buffer = (c_ubyte * len(data)).from_buffer_copy(data)
        return self.dll.c_aa_i2c_write(
            int(handle),
            int(address),
            self.AA_I2C_NO_FLAGS,
            len(data),
            buffer,
        )

    def i2c_read(self, handle: int, address: int, length: int) -> tuple[int, bytes]:
        buffer = (c_ubyte * length)()
        count = self.dll.c_aa_i2c_read(
            int(handle),
            int(address),
            self.AA_I2C_NO_FLAGS,
            int(length),
            buffer,
        )
        return count, bytes(buffer)
