from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware.instruments import Bode100Driver, Bode100Error, VisaConnectionError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start/connect to Bode 100 through Bode Analyzer Suite ScpiRunner.")
    parser.add_argument("--serial", default=None, help="Bode 100 serial number, or BODE100_SERIAL.")
    parser.add_argument("--host", default=None, help="SCPI server host, or BODE100_HOST.")
    parser.add_argument("--port", type=int, default=None, help="SCPI server port, or BODE100_PORT.")
    parser.add_argument("--scpi-runner-path", default=None)
    parser.add_argument("--visa-resource", default=None)
    parser.add_argument("--timeout", type=float, default=30.0, help="SCPI server startup timeout in seconds.")
    parser.add_argument("--visa-timeout-ms", type=int, default=20000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    driver = Bode100Driver(
        serial_number=args.serial,
        host=args.host,
        port=args.port,
        scpi_runner_path=args.scpi_runner_path,
        startup_timeout_s=args.timeout,
        visa_resource=args.visa_resource,
        timeout_ms=args.visa_timeout_ms,
    )
    try:
        driver.ensure_scpi_server()
        driver.connect()
        print(f"Connected resource: {driver.resource_name}")
        print(f"*IDN?: {driver.identify()}")
    except (Bode100Error, VisaConnectionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
