"""Capture and diff XDPE1A2G5C PMBus register snapshots.

This is intended for comparing what XDP Designer changes when a Vout setting
is applied. Snapshot mode is read-only and restores the original PAGE.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import localtime, strftime, time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.instruments.board_controller import BoardControllerConfig, create_board_controller
from hardware.instruments.i2c_adapters import create_i2c_adapter
from hardware.instruments.pmbus import decode_vout_mode, linear11_to_float, linear16_to_float


DEFAULT_ADDRESS = "0x5E"


@dataclass(frozen=True)
class CommandSpec:
    name: str
    code: int
    access: str
    decode: str = "raw"


BYTE_COMMANDS = [
    CommandSpec("PAGE", 0x00, "byte"),
    CommandSpec("OPERATION", 0x01, "byte"),
    CommandSpec("ON_OFF_CONFIG", 0x02, "byte"),
    CommandSpec("WRITE_PROTECT", 0x10, "byte"),
    CommandSpec("CAPABILITY", 0x19, "byte"),
    CommandSpec("VOUT_MODE", 0x20, "byte", "vout_mode"),
    CommandSpec("STATUS_VOUT", 0x7A, "byte"),
    CommandSpec("STATUS_IOUT", 0x7B, "byte"),
    CommandSpec("STATUS_INPUT", 0x7C, "byte"),
    CommandSpec("STATUS_CML", 0x7E, "byte"),
]

WORD_COMMANDS = [
    CommandSpec("VOUT_COMMAND", 0x21, "word", "vout"),
    CommandSpec("VOUT_TRIM", 0x22, "word", "vout_signed"),
    CommandSpec("VOUT_CAL_OFFSET", 0x23, "word", "vout_signed"),
    CommandSpec("VOUT_MAX", 0x24, "word", "vout"),
    CommandSpec("VOUT_MARGIN_HIGH", 0x25, "word", "vout"),
    CommandSpec("VOUT_MARGIN_LOW", 0x26, "word", "vout"),
    CommandSpec("VOUT_TRANSITION_RATE", 0x27, "word", "linear11"),
    CommandSpec("VOUT_DROOP", 0x28, "word", "linear11"),
    CommandSpec("VOUT_MIN", 0x2B, "word", "vout"),
    CommandSpec("STATUS_WORD", 0x79, "word"),
    CommandSpec("READ_VIN", 0x88, "word", "linear11"),
    CommandSpec("READ_VOUT", 0x8B, "word", "vout"),
    CommandSpec("READ_IOUT", 0x8C, "word", "linear11"),
    CommandSpec("READ_TEMPERATURE_1", 0x8D, "word", "linear11"),
    CommandSpec("READ_TEMPERATURE_2", 0x8E, "word", "linear11"),
    CommandSpec("READ_DUTY_CYCLE", 0x94, "word", "linear11"),
]

BLOCK_COMMANDS = [
    CommandSpec("MFR_ID", 0x99, "block_ascii"),
    CommandSpec("MFR_MODEL", 0x9A, "block_ascii"),
    CommandSpec("MFR_REVISION", 0x9B, "block_ascii"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="xdp", choices=["xdp", "xdp_pyusb", "aardvark", "mock"])
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--pages", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--output", type=Path, help="Where to save the snapshot JSON.")
    parser.add_argument("--aardvark-port", type=int, default=0)
    parser.add_argument("--bitrate-khz", type=int, default=100)
    parser.add_argument("--timeout-ms", type=int, default=3000)
    parser.add_argument("--xdp-address-mode", default="xdp_8bit", choices=["xdp_8bit", "7bit"])
    parser.add_argument(
        "--vendor-scan",
        action="store_true",
        help="Also try read-only byte/word probes for 0xA0-0xDF. This may take longer.",
    )
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.diff:
        before = _load_json(args.diff[0])
        after = _load_json(args.diff[1])
        _print_diff(before, after)
        return 0

    snapshot = capture_snapshot(args)
    text = json.dumps(snapshot, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Saved snapshot: {args.output}")
    else:
        print(text)
    _print_summary(snapshot)
    return 0


def capture_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    adapter_kwargs: dict[str, Any] = {}
    if args.adapter == "aardvark":
        adapter_kwargs = {"port": args.aardvark_port, "bitrate_khz": args.bitrate_khz}
    elif args.adapter in {"xdp", "xdp_pyusb"}:
        adapter_kwargs = {"address_mode": args.xdp_address_mode, "timeout_ms": args.timeout_ms}

    adapter = create_i2c_adapter(args.adapter, **adapter_kwargs)
    board = create_board_controller(
        "infineon_xdp",
        adapter,
        BoardControllerConfig(address=args.address, name="XDPE1A2G5C"),
    )

    board.connect()
    original_page: int | None = None
    try:
        try:
            original_page = board.read_page()
        except Exception:
            original_page = None

        identity = board.identify()
        pages = {}
        for page in args.pages:
            board.set_page(page)
            pages[str(page)] = _read_page(board, page, include_vendor_scan=args.vendor_scan)

        return {
            "adapter": args.adapter,
            "address": f"0x{board.device.address:02X}",
            "captured_at": strftime("%Y-%m-%d %H:%M:%S", localtime()),
            "captured_at_epoch": time(),
            "identity": {
                "manufacturer": _clean_text(identity.manufacturer),
                "model": _clean_text(identity.model),
                "revision": _clean_text(identity.revision),
                "status_word": _format_word(identity.status_word) if identity.status_word is not None else None,
            },
            "original_page": original_page,
            "pages": pages,
        }
    finally:
        if original_page is not None:
            try:
                board.set_page(original_page)
            except Exception:
                pass
        board.close()


def _read_page(board, page: int, include_vendor_scan: bool) -> dict[str, Any]:
    commands: dict[str, Any] = {}

    vout_exponent = None
    try:
        raw_mode = board.device.read_byte(0x20)
        mode_name, parameter = decode_vout_mode(raw_mode)
        if mode_name == "linear":
            vout_exponent = parameter
    except Exception:
        pass

    for spec in [*BYTE_COMMANDS, *WORD_COMMANDS, *BLOCK_COMMANDS]:
        commands[spec.name] = _read_command(board, spec, vout_exponent)

    if include_vendor_scan:
        commands["VENDOR_SCAN"] = _read_vendor_scan(board, vout_exponent)

    return {"page": page, "commands": commands}


def _read_command(board, spec: CommandSpec, vout_exponent: int | None) -> dict[str, Any]:
    item: dict[str, Any] = {"code": _format_byte(spec.code), "access": spec.access, "decode": spec.decode}
    try:
        if spec.access == "byte":
            raw = board.device.read_byte(spec.code)
            item["raw"] = _format_byte(raw)
            item["decoded"] = _decode_value(raw, spec.decode, vout_exponent)
        elif spec.access == "word":
            raw = board.device.read_word(spec.code)
            item["raw"] = _format_word(raw)
            item["decoded"] = _decode_value(raw, spec.decode, vout_exponent)
        elif spec.access == "block_ascii":
            raw = board.device.read_block(spec.code, max_length=64)
            item["raw"] = raw.hex(" ")
            item["decoded"] = _clean_text(raw.decode("ascii", errors="replace"))
        else:
            raise ValueError(f"Unsupported access type: {spec.access}")
        item["ok"] = True
    except Exception as exc:
        item["ok"] = False
        item["error"] = str(exc)
    return item


def _read_vendor_scan(board, vout_exponent: int | None) -> dict[str, Any]:
    scan: dict[str, Any] = {}
    for code in range(0xA0, 0xE0):
        entry: dict[str, Any] = {"code": _format_byte(code)}
        byte_value = None
        word_value = None
        try:
            byte_value = board.device.read_byte(code)
            entry["byte_raw"] = _format_byte(byte_value)
        except Exception as exc:
            entry["byte_error"] = str(exc)
        try:
            word_value = board.device.read_word(code)
            entry["word_raw"] = _format_word(word_value)
            entry["word_linear11"] = linear11_to_float(word_value)
            if vout_exponent is not None:
                entry["word_vout"] = linear16_to_float(word_value, vout_exponent)
        except Exception as exc:
            entry["word_error"] = str(exc)

        if byte_value is not None or word_value is not None:
            scan[_format_byte(code)] = entry
    return scan


def _decode_value(raw: int, decode: str, vout_exponent: int | None) -> Any:
    if decode == "raw":
        return raw
    if decode == "vout_mode":
        mode_name, parameter = decode_vout_mode(raw)
        return {"mode": mode_name, "parameter": parameter}
    if decode == "linear11":
        return linear11_to_float(raw)
    if decode == "vout":
        if vout_exponent is None:
            return None
        return linear16_to_float(raw, vout_exponent)
    if decode == "vout_signed":
        if vout_exponent is None:
            return None
        signed = raw if raw < 0x8000 else raw - 0x10000
        return signed * (2.0**vout_exponent)
    return raw


def _print_summary(snapshot: dict[str, Any]) -> None:
    print(f"Device: {snapshot.get('identity', {}).get('manufacturer')} {snapshot.get('identity', {}).get('model')}")
    for page_name, page in snapshot.get("pages", {}).items():
        commands = page.get("commands", {})
        vcmd = commands.get("VOUT_COMMAND", {})
        rvout = commands.get("READ_VOUT", {})
        riout = commands.get("READ_IOUT", {})
        op = commands.get("OPERATION", {})
        status = commands.get("STATUS_WORD", {})
        print(
            f"Page {page_name}: OP={op.get('raw')} STATUS={status.get('raw')} "
            f"VOUT_COMMAND={vcmd.get('decoded')} READ_VOUT={rvout.get('decoded')} READ_IOUT={riout.get('decoded')}"
        )


def _print_diff(before: dict[str, Any], after: dict[str, Any]) -> None:
    print(f"Before: {before.get('captured_at')}  After: {after.get('captured_at')}")
    changes = []
    before_pages = before.get("pages", {})
    after_pages = after.get("pages", {})
    for page_name in sorted(set(before_pages) | set(after_pages), key=_sort_page_name):
        before_commands = before_pages.get(page_name, {}).get("commands", {})
        after_commands = after_pages.get(page_name, {}).get("commands", {})
        for name in sorted(set(before_commands) | set(after_commands)):
            b = before_commands.get(name)
            a = after_commands.get(name)
            if name == "VENDOR_SCAN":
                changes.extend(_diff_vendor_scan(page_name, b or {}, a or {}))
                continue
            b_sig = _signature(b)
            a_sig = _signature(a)
            if b_sig != a_sig:
                changes.append((page_name, name, b_sig, a_sig))

    if not changes:
        print("No PMBus differences found in the captured commands.")
        return

    for page_name, name, b_sig, a_sig in changes:
        print(f"Page {page_name} {name}: {b_sig} -> {a_sig}")


def _diff_vendor_scan(page_name: str, before: dict[str, Any], after: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    changes = []
    for code in sorted(set(before) | set(after)):
        b_sig = _signature(before.get(code))
        a_sig = _signature(after.get(code))
        if b_sig != a_sig:
            changes.append((page_name, f"VENDOR_SCAN {code}", b_sig, a_sig))
    return changes


def _signature(item: Any) -> str:
    if item is None:
        return "<missing>"
    if isinstance(item, dict):
        if not item.get("ok", True) and "error" in item:
            return f"error={item.get('error')}"
        raw = item.get("raw")
        decoded = item.get("decoded")
        if raw is not None or decoded is not None:
            return f"raw={raw}, decoded={decoded}"
        parts = []
        for key in ("byte_raw", "word_raw", "word_linear11", "word_vout"):
            if key in item:
                parts.append(f"{key}={item[key]}")
        if parts:
            return ", ".join(parts)
    return json.dumps(item, sort_keys=True)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_byte(value: int) -> str:
    return f"0x{value & 0xFF:02X}"


def _format_word(value: int) -> str:
    return f"0x{value & 0xFFFF:04X}"


def _sort_page_name(value: str) -> tuple[int, str]:
    try:
        return (int(value), value)
    except ValueError:
        return (999, value)


def _clean_text(text: str) -> str:
    return text.replace("\x00", "").strip()


if __name__ == "__main__":
    raise SystemExit(main())
