"""Read or set XDPE1A2G5C VOUT_COMMAND over PMBus/I2C."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.instruments.board_controller import BoardControllerConfig, create_board_controller
from hardware.instruments.i2c_adapters import MockI2cAdapter, create_i2c_adapter
from hardware.instruments.pmbus import float_to_linear16


DEFAULT_ADDRESS = "0x5E"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="mock", choices=["mock", "aardvark", "xdp"])
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--page", type=int, default=0, help="0 = Loop A, 1 = Loop B")
    parser.add_argument("--voltage", type=float, help="Target VOUT_COMMAND voltage in V.")
    parser.add_argument("--read", action="store_true", help="Read current VOUT_MODE/VOUT_COMMAND/READ_VOUT.")
    parser.add_argument("--write", action="store_true", help="Actually write VOUT_COMMAND.")
    parser.add_argument("--dry-run", action="store_true", help="Compute encoding but do not write.")
    parser.add_argument("--aardvark-port", type=int, default=0)
    parser.add_argument("--bitrate-khz", type=int, default=100)
    parser.add_argument(
        "--xdp-address-mode",
        default="xdp_8bit",
        choices=["xdp_8bit", "7bit"],
        help="How to encode the I2C address inside the XDP dongle packet.",
    )
    args = parser.parse_args()

    if args.write and args.dry_run:
        raise SystemExit("--write and --dry-run cannot be used together.")
    if args.write and args.voltage is None:
        raise SystemExit("--write requires --voltage.")

    adapter_kwargs = {}
    if args.adapter == "aardvark":
        adapter_kwargs = {"port": args.aardvark_port, "bitrate_khz": args.bitrate_khz}
    elif args.adapter == "xdp":
        adapter_kwargs = {"address_mode": args.xdp_address_mode}
    elif args.adapter == "mock":
        adapter_kwargs = {"responses": _mock_xdpe_responses(args.address, args.page)}

    adapter = create_i2c_adapter(args.adapter, **adapter_kwargs)
    board = create_board_controller(
        "infineon_xdp",
        adapter,
        BoardControllerConfig(address=args.address, name="XDPE1A2G5C"),
    )

    board.connect()
    try:
        raw_mode, mode_name, exponent = board.read_vout_mode(args.page)
        print(f"Connected XDPE1A2G5C at 0x{board.device.address:02X}, page {args.page}")
        print(f"VOUT_MODE: 0x{raw_mode:02X} ({mode_name}, exponent/parameter={exponent})")

        if args.read:
            _print_readback(board, args.page)

        if args.voltage is not None:
            if mode_name != "linear":
                raise SystemExit(f"Unsupported VOUT_MODE {mode_name}; refusing to encode automatically.")
            raw = float_to_linear16(args.voltage, exponent)
            print(f"Requested VOUT_COMMAND: {args.voltage:.6g} V -> raw 0x{raw:04X}")

        if args.dry_run:
            print("Dry run only; no VOUT_COMMAND write sent.")

        if args.write:
            raw = board.set_vout_command(args.voltage, page=args.page)
            print(f"Wrote VOUT_COMMAND raw 0x{raw:04X} to page {args.page}.")
            _print_readback(board, args.page)

        if isinstance(adapter, MockI2cAdapter):
            for address, data in adapter.writes:
                print(f"MOCK write 0x{address:02X}: {data.hex(' ')}")
    finally:
        board.close()

    return 0


def _print_readback(board, page: int) -> None:
    try:
        print(f"VOUT_COMMAND readback: {board.read_vout_command(page):.6g} V")
    except Exception as exc:
        print(f"VOUT_COMMAND readback unavailable: {exc}")
    try:
        print(f"READ_VOUT telemetry: {board.read_vout(page):.6g} V")
    except Exception as exc:
        print(f"READ_VOUT telemetry unavailable: {exc}")


def _mock_xdpe_responses(address: str, page: int) -> dict[tuple[int, int, int], bytes]:
    addr = int(address, 16) if address.lower().startswith("0x") else int(address)
    exponent = -9
    mode = exponent & 0x1F
    vout_command = float_to_linear16(0.93, exponent)
    read_vout = float_to_linear16(0.93, exponent)
    return {
        (addr, 0x00, 1): bytes([page & 0xFF]),
        (addr, 0x20, 1): bytes([mode]),
        (addr, 0x21, 2): bytes([vout_command & 0xFF, (vout_command >> 8) & 0xFF]),
        (addr, 0x8B, 2): bytes([read_vout & 0xFF, (read_vout >> 8) & 0xFF]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
