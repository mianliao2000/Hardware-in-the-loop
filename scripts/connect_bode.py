from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware.instruments import BodeScpiClient, VisaConnectionError, bode_usb_driver_status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Connect to Bode 100 through the Bode Analyzer Suite SCPI server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5025)
    parser.add_argument("--timeout-ms", type=int, default=20000)
    parser.add_argument("--status", action="store_true", help="Print Windows USB driver status for Bode 100.")
    parser.add_argument("--status-only", action="store_true", help="Print USB status and do not connect to SCPI.")
    parser.add_argument("--sweep", action="store_true", help="Run a small Gain/Phase sweep.")
    parser.add_argument("--start", type=float, default=100.0)
    parser.add_argument("--stop", type=float, default=100000.0)
    parser.add_argument("--points", type=int, default=51)
    parser.add_argument("--bandwidth", type=float, default=1000.0)
    parser.add_argument("--csv", default="results/bode_gain_phase.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status:
        print("Bode USB status:")
        for key, value in bode_usb_driver_status().items():
            print(f"  {key}: {value}")
    if args.status_only:
        return 0

    client = BodeScpiClient(host=args.host, port=args.port, timeout_ms=args.timeout_ms)
    try:
        client.connect()
        print(f"Connected resource: {client.resource_name}")
        print(f"*IDN?: {client.idn()}")
        try:
            print(f"Lock: {client.lock()}")
        except Exception as exc:
            print(f"Lock warning: {exc}")

        if args.sweep:
            client.configure_gain_phase(args.start, args.stop, args.points, args.bandwidth)
            data = client.run_sweep()
            path = data.save_csv(args.csv)
            print(f"Captured {len(data.frequency_hz)} Bode points")
            print(f"CSV: {path}")
        try:
            print(f"SYST:ERR?: {client.get_error()}")
        except Exception as exc:
            print(f"SYST:ERR? warning: {exc}")
        try:
            client.unlock()
        except Exception:
            pass
    except VisaConnectionError as exc:
        print(f"ERROR: {exc}")
        print()
        print("For Bode 100, start Bode Analyzer Suite's SCPI server first.")
        print("BAS: Advanced -> Select SCPI Server -> host 127.0.0.1 -> port 5025 -> Start")
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
