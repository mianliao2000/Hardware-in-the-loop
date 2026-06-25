"""List Total Phase Aardvark adapters visible to the Aardvark DLL."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ctypes import POINTER, WinDLL, c_int, c_ushort
from importlib.util import find_spec


def main() -> int:
    dll = _load_aardvark_dll()
    dll.c_aa_find_devices.argtypes = [c_int, POINTER(c_ushort)]
    dll.c_aa_find_devices.restype = c_int
    dll.c_aa_open.argtypes = [c_int]
    dll.c_aa_open.restype = c_int
    dll.c_aa_close.argtypes = [c_int]
    dll.c_aa_close.restype = c_int

    devices = (c_ushort * 16)()
    count = dll.c_aa_find_devices(16, devices)
    print(f"Aardvark devices found/status: {count}")
    for index in range(max(0, count)):
        port = devices[index]
        in_use = bool(port & 0x8000)
        port_number = port & ~0x8000
        open_status = dll.c_aa_open(port_number)
        print(f"  port={port_number} in_use={in_use} open_status={open_status}")
        if open_status > 0:
            dll.c_aa_close(open_status)
    return 0


def _load_aardvark_dll():
    spec = find_spec("aardvark_py")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("aardvark_py package is not installed. Run: python -m pip install aardvark_py")
    dll_path = Path(next(iter(spec.submodule_search_locations))) / "aardvark.dll"
    if not dll_path.exists():
        raise SystemExit(f"Could not find aardvark.dll at {dll_path}")
    return WinDLL(str(dll_path))


if __name__ == "__main__":
    raise SystemExit(main())
