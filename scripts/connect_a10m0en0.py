"""Probe the FTDI serial device with serial number A10M0EN0.

The device enumerates as an FT232R USB UART, so communication happens through
the Windows COM port / VISA ASRL resource instead of USBTMC.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass

import pyvisa


SERIAL_NUMBER = "A10M0EN0"
DEFAULT_QUERY = "*IDN?"
BAUD_RATES = (9600, 115200, 57600, 38400, 19200)


@dataclass
class SerialDevice:
    com_port: str
    visa_resource: str


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Query to send after opening the serial port.")
    parser.add_argument("--baud", type=int, default=0, help="Baud rate to try. Default tries common rates.")
    args = parser.parse_args()

    device = find_a10m0en0()
    if device is None:
        print(f"Could not find FTDI device serial {SERIAL_NUMBER}.")
        return 1

    print(f"Found {SERIAL_NUMBER}: {device.com_port} / {device.visa_resource}")
    rates = (args.baud,) if args.baud else BAUD_RATES
    for baud in rates:
        probe_visa_serial(device.visa_resource, baud, args.query)
    return 0


def find_a10m0en0() -> SerialDevice | None:
    command = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-PnpDevice -PresentOnly | "
            f"Where-Object {{ $_.InstanceId -like '*{SERIAL_NUMBER}*' -and $_.Class -eq 'Ports' }} | "
            "Select-Object -First 1 -ExpandProperty FriendlyName"
        ),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    friendly = completed.stdout.strip()
    if not friendly:
        return None
    start = friendly.rfind("(COM")
    end = friendly.rfind(")")
    if start < 0 or end < 0:
        return None
    com_port = friendly[start + 1 : end]
    digits = "".join(ch for ch in com_port if ch.isdigit())
    return SerialDevice(com_port=com_port, visa_resource=f"ASRL{digits}::INSTR")


def probe_visa_serial(resource: str, baud: int, query: str) -> None:
    print(f"\n--- {resource} baud={baud} ---")
    rm = pyvisa.ResourceManager()
    inst = None
    try:
        inst = rm.open_resource(resource)
        inst.baud_rate = baud
        inst.data_bits = 8
        inst.stop_bits = pyvisa.constants.StopBits.one
        inst.parity = pyvisa.constants.Parity.none
        inst.timeout = 800
        inst.read_termination = "\n"
        inst.write_termination = "\n"
        try:
            inst.clear()
        except Exception as exc:
            print(f"clear: {type(exc).__name__}: {exc}")
        try:
            data = inst.read_bytes(256, break_on_termchar=False)
            print(f"initial_read: {data.hex(' ') if data else '<none>'}")
        except Exception as exc:
            print(f"initial_read: {type(exc).__name__}: {exc}")
        try:
            inst.write(query)
            response = inst.read()
            print(f"query_response: {response.strip() if response else '<none>'}")
        except Exception as exc:
            print(f"query_response: {type(exc).__name__}: {exc}")
    except Exception as exc:
        print(f"open_error: {type(exc).__name__}: {exc}")
    finally:
        if inst is not None:
            inst.close()


if __name__ == "__main__":
    raise SystemExit(main())
