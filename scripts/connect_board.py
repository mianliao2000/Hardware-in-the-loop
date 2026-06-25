"""Connect to a PMBus/I2C board controller and run safe checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.instruments.board_controller import BoardControllerConfig, create_board_controller
from hardware.instruments.i2c_adapters import create_i2c_adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="mock", choices=["mock", "aardvark"])
    parser.add_argument("--address", default="0x40", help="7-bit I2C address, e.g. 0x40")
    parser.add_argument("--kind", default="generic", choices=["generic", "infineon_xdp", "adi_power"])
    parser.add_argument("--name", default="board")
    parser.add_argument("--aardvark-port", type=int, default=0)
    parser.add_argument("--bitrate-khz", type=int, default=100)
    parser.add_argument("--identify", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--clear-faults",
        action="store_true",
        help="Write PMBus CLEAR_FAULTS. This changes device state.",
    )
    args = parser.parse_args()

    adapter_kwargs = {}
    if args.adapter == "aardvark":
        adapter_kwargs = {"port": args.aardvark_port, "bitrate_khz": args.bitrate_khz}

    adapter = create_i2c_adapter(args.adapter, **adapter_kwargs)
    board = create_board_controller(
        args.kind,
        adapter,
        BoardControllerConfig(address=args.address, name=args.name),
    )

    board.connect()
    try:
        print(f"Connected {args.kind} board at I2C address 0x{board.device.address:02X}")

        if args.identify:
            identity = board.identify()
            print(f"Manufacturer: {identity.manufacturer or '<unavailable>'}")
            print(f"Model: {identity.model or '<unavailable>'}")
            print(f"Revision: {identity.revision or '<unavailable>'}")
            if identity.status_word is not None:
                print(f"Status word: 0x{identity.status_word:04X}")
            else:
                print("Status word: <unavailable>")

        if args.status:
            status = board.read_status_word()
            print(f"Status word: 0x{status:04X}")

        if args.clear_faults:
            board.clear_faults()
            print("CLEAR_FAULTS sent.")
    finally:
        board.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
