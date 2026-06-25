from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    try:
        import usb.core  # type: ignore
        import usb.util  # type: ignore
        import libusb_package  # type: ignore
    except ImportError:
        print("ERROR: USB dependencies are not installed. Run: python -m pip install -r requirements.txt")
        return 1

    backend = libusb_package.get_libusb1_backend()
    devices = list(usb.core.find(find_all=True, backend=backend))
    if not devices:
        print("No USB devices visible through PyUSB.")
        return 0

    print("USB devices visible through PyUSB:")
    for dev in devices:
        manufacturer = _safe_usb_string(usb.util, dev, dev.iManufacturer)
        product = _safe_usb_string(usb.util, dev, dev.iProduct)
        serial = _safe_usb_string(usb.util, dev, dev.iSerialNumber)
        print(
            f"  VID:PID={dev.idVendor:04x}:{dev.idProduct:04x} "
            f"bus={getattr(dev, 'bus', '?')} address={getattr(dev, 'address', '?')} "
            f"manufacturer={manufacturer or '-'} product={product or '-'} serial={serial or '-'}"
        )
    return 0


def _safe_usb_string(usb_util, dev, index: int | None) -> str:
    if not index:
        return ""
    try:
        return str(usb_util.get_string(dev, index))
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
