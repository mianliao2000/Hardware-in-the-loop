"""Capture and diff XDPE internal AHB memory snapshots.

This is read-only unless another script explicitly writes memory. It is used
to compare XDP Designer actions such as VREN High/Low/Release, which may not
show up as standard PMBus command changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import localtime, perf_counter, strftime, time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.instruments.board_controller import BoardControllerConfig, create_board_controller
from hardware.instruments.i2c_adapters import create_i2c_adapter


DEFAULT_ADDRESS = "0x5E"
DEFAULT_RANGES = ["0x70000000:0x70002000"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="xdp", choices=["xdp", "xdp_pyusb"])
    parser.add_argument("--address", default=DEFAULT_ADDRESS)
    parser.add_argument("--range", dest="ranges", action="append", default=[], help="Scan range START:END[:STEP]. END is exclusive.")
    parser.add_argument("--output", type=Path, help="Where to save the snapshot JSON.")
    parser.add_argument("--timeout-ms", type=int, default=3000)
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
        print(f"Saved memory snapshot: {args.output}")
    else:
        print(text)
    _print_summary(snapshot)
    return 0


def capture_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    ranges = [_parse_range(item) for item in (args.ranges or DEFAULT_RANGES)]
    adapter = create_i2c_adapter(args.adapter, timeout_ms=args.timeout_ms)
    board = create_board_controller(
        "infineon_xdp",
        adapter,
        BoardControllerConfig(address=args.address, name="XDPE1A2G5C"),
    )
    started = perf_counter()
    board.connect()
    try:
        blocks = []
        for start, end, step in ranges:
            values = {}
            errors = {}
            for address in range(start, end, step):
                try:
                    values[_format_address(address)] = _format_word32(board._read_xdpe_ahb_word(address))
                except Exception as exc:
                    errors[_format_address(address)] = str(exc)
            blocks.append(
                {
                    "start": _format_address(start),
                    "end": _format_address(end),
                    "step": step,
                    "count": len(values),
                    "error_count": len(errors),
                    "values": values,
                    "errors": errors,
                }
            )
        return {
            "adapter": args.adapter,
            "address": f"0x{board.device.address:02X}",
            "captured_at": strftime("%Y-%m-%d %H:%M:%S", localtime()),
            "captured_at_epoch": time(),
            "duration_s": perf_counter() - started,
            "ranges": blocks,
        }
    finally:
        board.close()


def _parse_range(text: str) -> tuple[int, int, int]:
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Range must be START:END[:STEP], got {text!r}")
    start = int(parts[0], 0)
    end = int(parts[1], 0)
    step = int(parts[2], 0) if len(parts) == 3 else 4
    if start < 0 or end <= start or step <= 0:
        raise ValueError(f"Invalid scan range: {text!r}")
    return start, end, step


def _print_summary(snapshot: dict[str, Any]) -> None:
    total = sum(block.get("count", 0) for block in snapshot.get("ranges", []))
    errors = sum(block.get("error_count", 0) for block in snapshot.get("ranges", []))
    print(f"Read {total} words with {errors} errors in {snapshot.get('duration_s', 0):.2f} s")
    for block in snapshot.get("ranges", []):
        print(f"{block['start']}:{block['end']} step={block['step']} count={block['count']} errors={block['error_count']}")


def _print_diff(before: dict[str, Any], after: dict[str, Any]) -> None:
    before_values = _flatten_values(before)
    after_values = _flatten_values(after)
    changes = []
    for address in sorted(set(before_values) | set(after_values), key=lambda item: int(item, 16)):
        before_value = before_values.get(address)
        after_value = after_values.get(address)
        if before_value != after_value:
            changes.append((address, before_value, after_value))
    print(f"Before: {before.get('captured_at')}  After: {after.get('captured_at')}")
    if not changes:
        print("No memory differences found in scanned ranges.")
        return
    for address, before_value, after_value in changes:
        print(f"{address}: {before_value} -> {after_value}")


def _flatten_values(snapshot: dict[str, Any]) -> dict[str, str]:
    values = {}
    for block in snapshot.get("ranges", []):
        values.update(block.get("values", {}))
    return values


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_address(value: int) -> str:
    return f"0x{value:08X}"


def _format_word32(value: int) -> str:
    return f"0x{value & 0xFFFFFFFF:08X}"


if __name__ == "__main__":
    raise SystemExit(main())
