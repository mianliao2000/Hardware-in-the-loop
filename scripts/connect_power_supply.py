"""Connect to a Keysight N5700/N5767A power supply safely."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.instruments.power_supply import KeysightN5700PowerSupply
from hardware.instruments.visa_resource import list_visa_resources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource", help="VISA resource for the master supply.")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument("--voltage", type=float, help="Set voltage in V; output state is unchanged.")
    parser.add_argument("--current-limit", type=float, help="Set master current-limit command in A.")
    parser.add_argument("--single-unit-current-limit", type=float, default=25.0)
    parser.add_argument("--voltage-limit", type=float, default=60.0)
    parser.add_argument("--parallel-units", type=int, default=2)
    parser.add_argument("--output-on", action="store_true", help="Enable output after setting limits.")
    parser.add_argument("--output-off", action="store_true", help="Disable output.")
    args = parser.parse_args()

    if args.output_on and args.output_off:
        raise SystemExit("--output-on and --output-off cannot be used together.")

    resource = args.resource or _find_keysight_resource(list_visa_resources())
    if resource is None:
        raise SystemExit("No Keysight VISA resource found. Pass --resource explicitly.")

    supply = KeysightN5700PowerSupply(
        resource,
        timeout_ms=args.timeout_ms,
        model_voltage_limit_v=args.voltage_limit,
        model_current_limit_a=args.single_unit_current_limit,
        configured_parallel_units=args.parallel_units,
    )
    supply.connect()
    try:
        print(f"Connected resource: {resource}")
        _print_readback("Initial", supply)

        if args.voltage is not None:
            supply.set_voltage(args.voltage)
            print(f"Voltage setpoint requested: {args.voltage:g} V")

        if args.current_limit is not None:
            supply.set_current_limit(args.current_limit)
            print(f"Current limit requested: {args.current_limit:g} A")

        if args.output_off:
            supply.output_off()
            print("Output OFF requested.")

        if args.output_on:
            if args.voltage is None or args.current_limit is None:
                raise SystemExit("--output-on requires both --voltage and --current-limit.")
            supply.output_on()
            print("Output ON requested.")

        if any([args.voltage is not None, args.current_limit is not None, args.output_off, args.output_on]):
            _print_readback("Final", supply)
    finally:
        supply.close()

    return 0


def _find_keysight_resource(resources: tuple[str, ...]) -> str | None:
    for resource in resources:
        if "0X0957" in resource.upper():
            return resource
    return None


def _print_readback(label: str, supply: KeysightN5700PowerSupply) -> None:
    readback = supply.readback()
    print(f"{label} IDN: {readback.identity}")
    print(f"{label} output: {_format_output(readback.output_enabled)}")
    print(f"{label} Vset: {_format_float(readback.voltage_setpoint_v)} V")
    print(f"{label} Ilimit: {_format_float(readback.current_limit_a)} A")
    print(f"{label} Vmeas: {_format_float(readback.measured_voltage_v)} V")
    print(f"{label} Imeas: {_format_float(readback.measured_current_a)} A")
    print(f"{label} error: {readback.error}")


def _format_output(value: bool | None) -> str:
    if value is True:
        return "ON"
    if value is False:
        return "OFF"
    return "<unknown>"


def _format_float(value: float | None) -> str:
    if value is None:
        return "<unavailable>"
    return f"{value:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
