from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware.instruments import VisaConnectionError, list_visa_resources


def main() -> int:
    try:
        resources = list_visa_resources()
    except VisaConnectionError as exc:
        print(f"ERROR: {exc}")
        return 1

    if not resources:
        print("No VISA resources found.")
        return 0

    print("VISA resources:")
    for resource in resources:
        print(f"  {resource}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
