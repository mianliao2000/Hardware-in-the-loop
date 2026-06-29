"""Read XDPE VREN-related state without changing the board.

For XDPE1A2G5C / Yosemite RevC, XDP Designer's VREN High/Low/Release menu
updates the firmware configuration byte ``fw_config_data.xv_en`` at
0x2005D8E0.  This helper decodes that byte and also prints basic PMBus state
for correlation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import localtime, strftime
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.instruments.board_controller import (  # noqa: E402
    PMBUS_OPERATION,
    PMBUS_PAGE,
    PMBUS_READ_IOUT,
    PMBUS_READ_VOUT,
    PMBUS_STATUS_WORD,
    PMBUS_VOUT_COMMAND,
    PMBUS_VOUT_MODE,
    BoardControllerConfig,
    create_board_controller,
)
from hardware.instruments.i2c_adapters import create_i2c_adapter  # noqa: E402
from hardware.instruments.pmbus import decode_vout_mode, linear11_to_float, linear16_to_float  # noqa: E402


XDPE_XV_EN_ADDRESS = 0x2005D8E0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="xdp", choices=["xdp", "xdp_pyusb"])
    parser.add_argument("--address", default="0x5E")
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    args = parser.parse_args()

    adapter = create_i2c_adapter(args.adapter, timeout_ms=args.timeout_ms)
    board = create_board_controller(
        "infineon_xdp",
        adapter,
        BoardControllerConfig(address=args.address, name="XDPE1A2G5C"),
    )
    board.connect()
    try:
        result = read_state(board, args.page)
    finally:
        board.close()

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def read_state(board, page: int) -> dict[str, Any]:
    board.set_page(page)
    page_readback = board.device.read_byte(PMBUS_PAGE)
    operation = board.device.read_byte(PMBUS_OPERATION)
    status_word = board.device.read_word(PMBUS_STATUS_WORD)
    vout_mode_raw = board.device.read_byte(PMBUS_VOUT_MODE)
    mode_name, exponent = decode_vout_mode(vout_mode_raw)

    vout_command_v = None
    read_vout_v = None
    if mode_name == "linear":
        vout_command_v = linear16_to_float(board.device.read_word(PMBUS_VOUT_COMMAND), exponent)
        read_vout_v = linear16_to_float(board.device.read_word(PMBUS_READ_VOUT), exponent)
    read_iout_a = linear11_to_float(board.device.read_word(PMBUS_READ_IOUT))

    xv_en_word = board._read_xdpe_ahb_word(XDPE_XV_EN_ADDRESS)
    xv_en = xv_en_word & 0xFF
    vren_bits = (xv_en >> 4) & 0x03
    return {
        "captured_at": strftime("%Y-%m-%d %H:%M:%S", localtime()),
        "pmbus": {
            "address": f"0x{board.device.address:02X}",
            "page": page_readback,
            "operation": f"0x{operation:02X}",
            "status_word": f"0x{status_word:04X}",
            "vout_mode": f"0x{vout_mode_raw:02X}",
            "vout_mode_decoded": mode_name,
            "vout_exponent": exponent,
            "vout_command_v": vout_command_v,
            "read_vout_v": read_vout_v,
            "read_iout_a": read_iout_a,
        },
        "xv_en": {
            "address": f"0x{XDPE_XV_EN_ADDRESS:08X}",
            "word": f"0x{xv_en_word & 0xFFFFFFFF:08X}",
            "byte": f"0x{xv_en:02X}",
            "bit5_sw_enable_pin_value": (xv_en >> 5) & 1,
            "bit4_enable_sw_enable_pin": (xv_en >> 4) & 1,
            "bits_5_4": f"0b{vren_bits:02b}",
            "decoded_vren": _decode_vren(vren_bits),
        },
    }


def _decode_vren(bits_5_4: int) -> str:
    if bits_5_4 == 0b11:
        return "VREN High"
    if bits_5_4 == 0b01:
        return "VREN Low"
    if bits_5_4 == 0b00:
        return "VREN Release"
    return "reserved/unknown"


if __name__ == "__main__":
    raise SystemExit(main())
