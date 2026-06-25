from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware.instruments import VisaConnectionError
from hardware.testbench import HardwareTestbench, HardwareTestbenchConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Connect to a USB function generator over VISA.")
    parser.add_argument("--resource", help="VISA resource name. Defaults to first USB resource.")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--configure-sine", action="store_true")
    parser.add_argument("--frequency", type=float, default=1000.0)
    parser.add_argument("--phase", type=float, help="Phase in degrees.")
    parser.add_argument("--amplitude", type=float, default=0.2, help="Amplitude in Vpp.")
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--enable-output", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bench = HardwareTestbench(
        HardwareTestbenchConfig(
            function_generator_resource=args.resource,
            visa_timeout_ms=args.timeout_ms,
        )
    )

    try:
        fg = bench.connect_function_generator()
        fg.output_channel = args.channel
        print(f"Connected resource: {fg.resource_name}")
        print(f"*IDN?: {fg.idn()}")

        if args.configure_sine:
            fg.configure_sine(args.frequency, args.amplitude, args.offset, args.phase, args.channel)
            print(
                "Configured sine: "
                f"{args.frequency:g} Hz, {args.amplitude:g} Vpp, offset {args.offset:g} V"
                + (f", phase {args.phase:g} deg" if args.phase is not None else "")
            )
        elif args.phase is not None:
            fg.set_phase(args.phase, args.channel)
            print(f"Set CH{args.channel} phase: {args.phase:g} deg")

        if args.enable_output:
            fg.output_on(args.channel)
            print(f"Output channel {args.channel}: ON")
        else:
            fg.output_off(args.channel)
            print(f"Output channel {args.channel}: OFF")

        try:
            print(f"SYST:ERR?: {fg.get_error()}")
        except Exception as exc:
            print(f"SYST:ERR? unavailable: {exc}")

    except (VisaConnectionError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        bench.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
