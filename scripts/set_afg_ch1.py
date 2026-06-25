from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware.instruments import FunctionGenerator, VisaConnectionError


DEFAULT_AFG_RESOURCE = "USB0::0x0699::0x0356::B010652::INSTR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set Tektronix AFG31000 CH1 frequency and phase.")
    parser.add_argument("--resource", default=DEFAULT_AFG_RESOURCE)
    parser.add_argument("--frequency", type=float, help="CH1 frequency in Hz.")
    parser.add_argument("--phase", type=float, help="CH1 phase in degrees.")
    parser.add_argument("--output", choices=["on", "off", "keep"], default="keep")
    parser.add_argument("--query", action="store_true", help="Query CH1 settings after applying changes.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fg = FunctionGenerator(args.resource, timeout_ms=5000, output_channel=1)
    try:
        fg.connect()
        print(f"Connected: {fg.idn()}")

        if args.frequency is not None:
            fg.set_frequency(args.frequency, 1)
            print(f"CH1 frequency set to {args.frequency:g} Hz")
        if args.phase is not None:
            fg.set_phase(args.phase, 1)
            print(f"CH1 phase set to {args.phase:g} deg")

        if args.output == "on":
            fg.output_on(1)
            print("CH1 output ON")
        elif args.output == "off":
            fg.output_off(1)
            print("CH1 output OFF")

        if args.query:
            print(f"CH1 frequency? {fg.get_frequency(1)}")
            print(f"CH1 phase? {fg.get_phase(1)}")
            try:
                print(f"SYST:ERR? {fg.get_error()}")
            except Exception as exc:
                print(f"SYST:ERR? unavailable: {exc}")
    except VisaConnectionError as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        fg.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
