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
PMBUS_ON_OFF_CONFIG = 0x02
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
PMBUS_MFR_AHB_ADDRESS = 0xCE
PMBUS_MFR_REG_WRITE = 0xDE
PMBUS_MFR_REG_READ = 0xDF

XDPE_MOD0_CE_DT_L_ADDRESS = 0x70000C3C
XDPE_MOD0_CE_DT_L_START = 12
XDPE_MOD0_CE_DT_L_LENGTH = 13
XDPE_MOD0_CE_LC_LM_ADDRESS = 0x70000C48
XDPE_MOD0_CE_LC_LM_START = 9
XDPE_MOD0_CE_LC_LM_LENGTH = 9
XDPE_MOD0_PID_ADDRESS = 0x70006000
XDPE_MOD0_PID_FIELDS = {
    "mod0_kp": (0, 8),
    "mod0_ki": (8, 8),
    "mod0_kd": (16, 8),
    "mod0_kpole1": (24, 4),
    "mod0_kpole2": (28, 4),
}
XDPE_MOD0_CM_GAIN_ADDRESS = 0x70006004
XDPE_MOD0_CURRENT_MODE_FIELDS = {
    "mod0_cm_gain": (0, 7),
    "mod0_cm_pole": (7, 5),
    "mod0_cm_gain_fatr": (12, 7),
    "mod0_cm_pole_fatr": (19, 5),
}
XDPE_MOD0_CURRENT_MODE_WRITABLE_FIELDS = {"mod0_cm_gain"}
XDPE_MOD0_LL_BW_ADDRESS = 0x70000C14
XDPE_MOD0_LL_BW_FIELDS = {
    "mod0_ll_ls_bw": (0, 7),
    "mod0_ll_lr_bw": (7, 7),
}
XDPE_XV_EN_ADDRESS = 0x2005D8E0
XDPE_XV_EN_VREN_START = 4
XDPE_XV_EN_VREN_LENGTH = 2
XDPE_XV_EN_VREN_VALUES = {
    "release": 0b00,
    "low": 0b01,
    "high": 0b11,
}
XDPE_MEMORY_WORD_SIZE = 4


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

    def read_on_off_config(self, page: int = 0) -> int:
        self.set_page(page)
        return self.device.read_byte(PMBUS_ON_OFF_CONFIG)

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

    def read_vout_telemetry(self, page: int = 0) -> dict:
        self.set_page(page)
        raw_mode = self.device.read_byte(PMBUS_VOUT_MODE)
        mode_name, parameter = decode_vout_mode(raw_mode)
        if mode_name != "linear":
            raise VoutModeError(f"VOUT_MODE 0x{raw_mode:02X} is {mode_name}; only LINEAR16 is supported.")
        return {
            "vout_mode_raw": raw_mode,
            "vout_mode": mode_name,
            "exponent": parameter,
            "vout_command_v": linear16_to_float(self.device.read_word(PMBUS_VOUT_COMMAND), parameter),
            "read_vout_v": linear16_to_float(self.device.read_word(PMBUS_READ_VOUT), parameter),
            "read_iout_a": linear11_to_float(self.device.read_word(PMBUS_READ_IOUT)),
        }

    def set_vout_command(self, voltage_v: float, page: int = 0) -> int:
        if not 0.0 <= voltage_v <= 2.0:
            raise ValueError(f"Refusing VOUT_COMMAND outside 0-2 V bring-up range: {voltage_v}")
        raw_mode, mode_name, parameter = self.read_vout_mode(page)
        if mode_name != "linear":
            raise VoutModeError(f"VOUT_MODE 0x{raw_mode:02X} is {mode_name}; only LINEAR16 is supported.")
        raw = float_to_linear16(voltage_v, parameter)
        self.device.write_word(PMBUS_VOUT_COMMAND, raw)
        return raw

    def read_output_inductance_nh(self, page: int = 0) -> dict:
        self._require_loop_a_register(page, "Output Inductance")
        word, raw = self._read_xdpe_memory_field(
            XDPE_MOD0_CE_DT_L_ADDRESS,
            XDPE_MOD0_CE_DT_L_START,
            XDPE_MOD0_CE_DT_L_LENGTH,
        )
        value_nh = _output_inductance_from_raw(raw)
        return _memory_field_result(
            name="mod0_ce_dt_l",
            address=XDPE_MOD0_CE_DT_L_ADDRESS,
            start=XDPE_MOD0_CE_DT_L_START,
            length=XDPE_MOD0_CE_DT_L_LENGTH,
            word=word,
            raw=raw,
            value_nh=value_nh,
        )

    def set_output_inductance_nh(self, value_nh: float, page: int = 0) -> dict:
        self._require_loop_a_register(page, "Output Inductance")
        raw = _raw_from_output_inductance(value_nh)
        before, after = self._update_xdpe_memory_field(
            XDPE_MOD0_CE_DT_L_ADDRESS,
            XDPE_MOD0_CE_DT_L_START,
            XDPE_MOD0_CE_DT_L_LENGTH,
            raw,
        )
        actual = _output_inductance_from_raw(raw)
        return _memory_field_write_result(
            name="mod0_ce_dt_l",
            address=XDPE_MOD0_CE_DT_L_ADDRESS,
            start=XDPE_MOD0_CE_DT_L_START,
            length=XDPE_MOD0_CE_DT_L_LENGTH,
            word_before=before,
            word_after=after,
            raw=raw,
            requested_nh=value_nh,
            actual_nh=actual,
        )

    def read_effective_lc_inductance_nh(self, page: int = 0) -> dict:
        self._require_loop_a_register(page, "Effective Lc Inductance")
        word, raw = self._read_xdpe_memory_field(
            XDPE_MOD0_CE_LC_LM_ADDRESS,
            XDPE_MOD0_CE_LC_LM_START,
            XDPE_MOD0_CE_LC_LM_LENGTH,
        )
        value_nh = _effective_lc_from_raw(raw)
        return _memory_field_result(
            name="mod0_ce_lc_lm",
            address=XDPE_MOD0_CE_LC_LM_ADDRESS,
            start=XDPE_MOD0_CE_LC_LM_START,
            length=XDPE_MOD0_CE_LC_LM_LENGTH,
            word=word,
            raw=raw,
            value_nh=value_nh,
        )

    def set_effective_lc_inductance_nh(self, value_nh: float, page: int = 0) -> dict:
        self._require_loop_a_register(page, "Effective Lc Inductance")
        raw = _raw_from_effective_lc(value_nh)
        before, after = self._update_xdpe_memory_field(
            XDPE_MOD0_CE_LC_LM_ADDRESS,
            XDPE_MOD0_CE_LC_LM_START,
            XDPE_MOD0_CE_LC_LM_LENGTH,
            raw,
        )
        actual = _effective_lc_from_raw(raw)
        return _memory_field_write_result(
            name="mod0_ce_lc_lm",
            address=XDPE_MOD0_CE_LC_LM_ADDRESS,
            start=XDPE_MOD0_CE_LC_LM_START,
            length=XDPE_MOD0_CE_LC_LM_LENGTH,
            word_before=before,
            word_after=after,
            raw=raw,
            requested_nh=value_nh,
            actual_nh=actual,
        )

    def read_mod0_pid_registers(self, page: int = 0) -> dict:
        self._require_loop_a_register(page, "mod0 PID registers")
        word = self._read_xdpe_ahb_word(XDPE_MOD0_PID_ADDRESS)
        fields = {}
        for name, (start, length) in XDPE_MOD0_PID_FIELDS.items():
            raw = _extract_bits(word, start, length)
            fields[name] = _xdp_register_field_result(
                name=name,
                address=XDPE_MOD0_PID_ADDRESS,
                start=start,
                length=length,
                word=word,
                raw=raw,
            )
        return {
            "name": "modulator.mod0_pid",
            "memory_address": f"0x{XDPE_MOD0_PID_ADDRESS:08X}",
            "word": f"0x{word:08X}",
            "fields": fields,
        }

    def set_mod0_pid_registers(self, values: dict[str, int], page: int = 0) -> dict:
        self._require_loop_a_register(page, "mod0 PID registers")
        unknown = sorted(set(values) - set(XDPE_MOD0_PID_FIELDS))
        if unknown:
            raise ValueError(f"Unsupported mod0 PID field(s): {', '.join(unknown)}")
        before = self._read_xdpe_ahb_word(XDPE_MOD0_PID_ADDRESS)
        after = before
        writes = {}
        for name, value in values.items():
            start, length = XDPE_MOD0_PID_FIELDS[name]
            raw = int(value)
            max_field = (1 << length) - 1
            if not 0 <= raw <= max_field:
                raise ValueError(f"{name} value {raw} does not fit in {length} bits.")
            mask = max_field << start
            after = (after & ~mask) | ((raw & max_field) << start)
            writes[name] = _xdp_register_field_result(
                name=name,
                address=XDPE_MOD0_PID_ADDRESS,
                start=start,
                length=length,
                word=after,
                raw=raw,
            )
        if after != before:
            self._write_xdpe_ahb_word(XDPE_MOD0_PID_ADDRESS, after)
        readback = self.read_mod0_pid_registers(page)
        return {
            "name": "modulator.mod0_pid",
            "memory_address": f"0x{XDPE_MOD0_PID_ADDRESS:08X}",
            "word_before": f"0x{before:08X}",
            "word_after": f"0x{after:08X}",
            "changed": before != after,
            "writes": writes,
            "readback": readback,
        }

    def read_mod0_current_mode_registers(self, page: int = 0) -> dict:
        self._require_loop_a_register(page, "mod0 current-mode registers")
        word = self._read_xdpe_ahb_word(XDPE_MOD0_CM_GAIN_ADDRESS)
        fields = {}
        for name, (start, length) in XDPE_MOD0_CURRENT_MODE_FIELDS.items():
            raw = _extract_bits(word, start, length)
            fields[name] = _xdp_register_field_result(
                name=name,
                address=XDPE_MOD0_CM_GAIN_ADDRESS,
                start=start,
                length=length,
                word=word,
                raw=raw,
            )
        return {
            "name": "modulator.mod0_current_mode",
            "memory_address": f"0x{XDPE_MOD0_CM_GAIN_ADDRESS:08X}",
            "word": f"0x{word:08X}",
            "fields": fields,
        }

    def set_mod0_current_mode_registers(self, values: dict[str, int], page: int = 0) -> dict:
        self._require_loop_a_register(page, "mod0 current-mode registers")
        unknown = sorted(set(values) - XDPE_MOD0_CURRENT_MODE_WRITABLE_FIELDS)
        if unknown:
            raise ValueError(f"Unsupported mod0 current-mode field(s): {', '.join(unknown)}")
        before = self._read_xdpe_ahb_word(XDPE_MOD0_CM_GAIN_ADDRESS)
        after = before
        writes = {}
        for name, value in values.items():
            start, length = XDPE_MOD0_CURRENT_MODE_FIELDS[name]
            raw = int(value)
            max_field = (1 << length) - 1
            if not 0 <= raw <= max_field:
                raise ValueError(f"{name} value {raw} does not fit in {length} bits.")
            mask = max_field << start
            after = (after & ~mask) | ((raw & max_field) << start)
            writes[name] = _xdp_register_field_result(
                name=name,
                address=XDPE_MOD0_CM_GAIN_ADDRESS,
                start=start,
                length=length,
                word=after,
                raw=raw,
            )
        if after != before:
            self._write_xdpe_ahb_word(XDPE_MOD0_CM_GAIN_ADDRESS, after)
        readback = self.read_mod0_current_mode_registers(page)
        return {
            "name": "modulator.mod0_current_mode",
            "memory_address": f"0x{XDPE_MOD0_CM_GAIN_ADDRESS:08X}",
            "word_before": f"0x{before:08X}",
            "word_after": f"0x{after:08X}",
            "changed": before != after,
            "writes": writes,
            "readback": readback,
        }

    def read_mod0_ll_bandwidth(self, page: int = 0) -> dict:
        """Read the two Loop-A low-load AVP bandwidth fields from one word."""

        self._require_loop_a_register(page, "mod0 low-load AVP bandwidth")
        word = self._read_xdpe_ahb_word(XDPE_MOD0_LL_BW_ADDRESS)
        fields = {
            name: _xdp_register_field_result(
                name=name,
                address=XDPE_MOD0_LL_BW_ADDRESS,
                start=start,
                length=length,
                word=word,
                raw=_extract_bits(word, start, length),
            )
            for name, (start, length) in XDPE_MOD0_LL_BW_FIELDS.items()
        }
        ls_value = int(fields["mod0_ll_ls_bw"]["raw"])
        lr_value = int(fields["mod0_ll_lr_bw"]["raw"])
        return {
            "name": "loop_a.mod0_ll_bandwidth",
            "memory_address": f"0x{XDPE_MOD0_LL_BW_ADDRESS:08X}",
            "word": f"0x{word:08X}",
            "fields": fields,
            "equal": ls_value == lr_value,
            "value": ls_value if ls_value == lr_value else None,
        }

    def set_mod0_ll_bandwidth(self, value: int, page: int = 0) -> dict:
        """Atomically set Loop-A LS and LR bandwidth to the same 7-bit value."""

        self._require_loop_a_register(page, "mod0 low-load AVP bandwidth")
        raw = int(value)
        if not 0 <= raw <= 0x7F:
            raise ValueError("mod0_ll_bw must be an integer from 0 through 127.")
        before = self._read_xdpe_ahb_word(XDPE_MOD0_LL_BW_ADDRESS)
        after = before
        writes = {}
        for name, (start, length) in XDPE_MOD0_LL_BW_FIELDS.items():
            mask = ((1 << length) - 1) << start
            after = (after & ~mask) | (raw << start)
            writes[name] = _xdp_register_field_result(
                name=name,
                address=XDPE_MOD0_LL_BW_ADDRESS,
                start=start,
                length=length,
                word=after,
                raw=raw,
            )
        if after != before:
            self._write_xdpe_ahb_word(XDPE_MOD0_LL_BW_ADDRESS, after)
        readback = self.read_mod0_ll_bandwidth(page)
        if not readback["equal"] or readback["value"] != raw:
            raise RuntimeError("Loop-A LS/LR bandwidth readback does not match the requested shared value.")
        return {
            "name": "loop_a.mod0_ll_bandwidth",
            "memory_address": f"0x{XDPE_MOD0_LL_BW_ADDRESS:08X}",
            "word_before": f"0x{before:08X}",
            "word_after": f"0x{after:08X}",
            "changed": before != after,
            "writes": writes,
            "readback": readback,
        }

    def read_vren_state(self, page: int = 0) -> dict:
        word, raw = self._read_xdpe_memory_field(
            XDPE_XV_EN_ADDRESS,
            XDPE_XV_EN_VREN_START,
            XDPE_XV_EN_VREN_LENGTH,
        )
        return _vren_state_result(word=word, raw=raw, page=page)

    def set_vren_state(self, state: str, page: int = 0) -> dict:
        normalized = state.strip().lower()
        if normalized not in XDPE_XV_EN_VREN_VALUES:
            allowed = ", ".join(sorted(XDPE_XV_EN_VREN_VALUES))
            raise ValueError(f"Unsupported VREN state {state!r}; expected one of: {allowed}.")
        operation_before = self.read_operation(page)
        before, after = self._update_xdpe_memory_field(
            XDPE_XV_EN_ADDRESS,
            XDPE_XV_EN_VREN_START,
            XDPE_XV_EN_VREN_LENGTH,
            XDPE_XV_EN_VREN_VALUES[normalized],
        )
        # XDP Designer re-writes OPERATION with the same value after changing xv_en.
        self.set_operation(operation_before, page)
        operation_after = self.read_operation(page)
        readback = self.read_vren_state(page)
        return {
            "requested": normalized,
            "memory_address": f"0x{XDPE_XV_EN_ADDRESS:08X}",
            "bitfield": f"[{XDPE_XV_EN_VREN_START + XDPE_XV_EN_VREN_LENGTH - 1}:{XDPE_XV_EN_VREN_START}]",
            "word_before": f"0x{before:08X}",
            "word_after": f"0x{after:08X}",
            "changed": before != after,
            "operation_before": f"0x{operation_before & 0xFF:02X}",
            "operation_after": f"0x{operation_after & 0xFF:02X}",
            "readback": readback,
        }

    def _read_xdpe_memory_field(self, memory_address: int, start: int, length: int) -> tuple[int, int]:
        word = self._read_xdpe_ahb_word(memory_address)
        return word, _extract_bits(word, start, length)

    def _update_xdpe_memory_field(
        self,
        memory_address: int,
        start: int,
        length: int,
        raw: int,
    ) -> tuple[int, int]:
        max_field = (1 << length) - 1
        if not 0 <= raw <= max_field:
            raise ValueError(f"Field value 0x{raw:X} does not fit in {length} bits.")
        before = self._read_xdpe_ahb_word(memory_address)
        mask = max_field << start
        after = (before & ~mask) | ((raw & max_field) << start)
        if after != before:
            self._write_xdpe_ahb_word(memory_address, after)
        return before, after

    def _read_xdpe_ahb_word(self, memory_address: int) -> int:
        self.device.write_block(PMBUS_MFR_AHB_ADDRESS, _int_to_le_bytes(memory_address, XDPE_MEMORY_WORD_SIZE))
        data = self.device.read_block(PMBUS_MFR_REG_READ, max_length=XDPE_MEMORY_WORD_SIZE)
        if len(data) != XDPE_MEMORY_WORD_SIZE:
            raise RuntimeError(
                f"XDPE AHB read returned {len(data)} bytes at 0x{memory_address:08X}; expected {XDPE_MEMORY_WORD_SIZE}."
            )
        return _le_bytes_to_int(data)

    def _write_xdpe_ahb_word(self, memory_address: int, value: int) -> None:
        self.device.write_block(PMBUS_MFR_AHB_ADDRESS, _int_to_le_bytes(memory_address, XDPE_MEMORY_WORD_SIZE))
        self.device.write_block(PMBUS_MFR_REG_WRITE, _int_to_le_bytes(value, XDPE_MEMORY_WORD_SIZE))

    def _require_loop_a_register(self, page: int, field_name: str) -> None:
        if page != 0:
            raise ValueError(f"{field_name} memory-register mapping is only confirmed for Loop A / page 0.")

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


def _extract_bits(word: int, start: int, length: int) -> int:
    return (word >> start) & ((1 << length) - 1)


def _int_to_le_bytes(value: int, length: int) -> bytes:
    return int(value).to_bytes(length, byteorder="little", signed=False)


def _le_bytes_to_int(data: bytes) -> int:
    return int.from_bytes(data, byteorder="little", signed=False)


def _output_inductance_from_raw(raw: int) -> float | None:
    if raw <= 0:
        return None
    return 10.0 * 4096.0 / (raw * 0.35)


def _raw_from_output_inductance(value_nh: float) -> int:
    if value_nh <= 0:
        raise ValueError("Output Inductance must be positive.")
    raw = round(10.0 * 4096.0 / (value_nh * 0.35))
    if not 1 <= raw <= (1 << XDPE_MOD0_CE_DT_L_LENGTH) - 1:
        raise ValueError(
            f"Output Inductance {value_nh} nH encodes to raw 0x{raw:X}, outside the confirmed 13-bit field."
        )
    return raw


def _effective_lc_from_raw(raw: int) -> float | None:
    if raw <= 0:
        return None
    return 4096.0 / (0.035 * raw)


def _raw_from_effective_lc(value_nh: float) -> int:
    if value_nh <= 0:
        raise ValueError("Effective Lc Inductance must be positive.")
    raw = round(4096.0 / (0.035 * value_nh))
    if not 1 <= raw <= (1 << XDPE_MOD0_CE_LC_LM_LENGTH) - 1:
        raise ValueError(
            f"Effective Lc Inductance {value_nh} nH encodes to raw 0x{raw:X}, outside the confirmed 9-bit field."
        )
    return raw


def _memory_field_result(
    *,
    name: str,
    address: int,
    start: int,
    length: int,
    word: int,
    raw: int,
    value_nh: float | None,
) -> dict:
    return {
        "name": name,
        "memory_address": f"0x{address:08X}",
        "bitfield": f"[{start + length - 1}:{start}]",
        "word": f"0x{word:08X}",
        "raw": raw,
        "raw_hex": f"0x{raw:X}",
        "value_nh": value_nh,
    }


def _memory_field_write_result(
    *,
    name: str,
    address: int,
    start: int,
    length: int,
    word_before: int,
    word_after: int,
    raw: int,
    requested_nh: float,
    actual_nh: float | None,
) -> dict:
    result = _memory_field_result(
        name=name,
        address=address,
        start=start,
        length=length,
        word=word_after,
        raw=raw,
        value_nh=actual_nh,
    )
    result.update(
        {
            "requested_nh": requested_nh,
            "actual_nh": actual_nh,
            "word_before": f"0x{word_before:08X}",
            "word_after": f"0x{word_after:08X}",
            "changed": word_before != word_after,
        }
    )
    return result


def _xdp_register_field_result(
    *,
    name: str,
    address: int,
    start: int,
    length: int,
    word: int,
    raw: int,
) -> dict:
    return {
        "name": name,
        "memory_address": f"0x{address:08X}",
        "bitfield": f"[{start + length - 1}:{start}]",
        "word": f"0x{word:08X}",
        "raw": raw,
        "raw_hex": f"0x{raw:X}",
        "min": 0,
        "max": (1 << length) - 1,
        "step": 1,
    }


def _vren_state_result(word: int, raw: int, page: int) -> dict:
    state = "unknown"
    for name, value in XDPE_XV_EN_VREN_VALUES.items():
        if value == raw:
            state = name
            break
    xv_en_byte = word & 0xFF
    return {
        "name": "fw_config_data.xv_en",
        "memory_address": f"0x{XDPE_XV_EN_ADDRESS:08X}",
        "bitfield": f"[{XDPE_XV_EN_VREN_START + XDPE_XV_EN_VREN_LENGTH - 1}:{XDPE_XV_EN_VREN_START}]",
        "page": page,
        "word": f"0x{word:08X}",
        "byte": f"0x{xv_en_byte:02X}",
        "raw": raw,
        "raw_binary": f"0b{raw:02b}",
        "state": state,
        "bit5_sw_enable_pin_value": (xv_en_byte >> 5) & 1,
        "bit4_enable_sw_enable_pin": (xv_en_byte >> 4) & 1,
    }
