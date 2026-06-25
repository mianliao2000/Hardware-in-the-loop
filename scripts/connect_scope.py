from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware.instruments import TektronixOscilloscope, VisaConnectionError


DEFAULT_MSO58_RESOURCE = "USB0::0x0699::0x0522::B010536::INSTR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Connect to a Tektronix MSO scope and capture waveform data.")
    parser.add_argument("--resource", default=DEFAULT_MSO58_RESOURCE)
    parser.add_argument("--source", default="CH1")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=10000)
    parser.add_argument("--csv", default="results/scope_ch1.csv")
    parser.add_argument("--capture", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scope = TektronixOscilloscope(args.resource, timeout_ms=10000)
    try:
        scope.connect()
        try:
            scope.write("*CLS")
        except Exception:
            pass
        print(f"Connected resource: {scope.resource_name}")
        print(f"*IDN?: {scope.idn()}")

        if args.capture:
            waveform = scope.capture_ascii_waveform(args.source, args.start, args.stop)
            path = waveform.save_csv(args.csv)
            print(f"Captured {len(waveform.x)} points from {args.source}")
            print(f"Saved CSV: {path}")
            if waveform.x:
                print(f"First point: t={waveform.x[0]:.6g} {waveform.x_unit}, y={waveform.y[0]:.6g} {waveform.y_unit}")
    except VisaConnectionError as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        scope.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
