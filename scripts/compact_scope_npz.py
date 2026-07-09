"""Compact saved scope waveform NPZ files for an autotune run.

Old format:
    iteration_001_scope_CH1.npz -> x float64 array + y float64 array
    iteration_001_scope_CH3.npz -> x float64 array + y float64 array

New format:
    iteration_001_scope.npz -> x_start/x_increment/points + y_CH1/y_CH3 float32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Autotune run directory to compact.")
    parser.add_argument("--keep-old", action="store_true", help="Keep old per-channel NPZ files.")
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    before = _npz_size(run_dir)
    run_status_path = run_dir / "run_status.json"
    iterations_path = run_dir / "iterations.jsonl"

    run_status = _read_json(run_status_path)
    history = run_status.get("history") if isinstance(run_status.get("history"), list) else []
    changed_records = 0
    compact_files: set[Path] = set()
    old_files: set[Path] = set()

    for record in history:
        if _compact_record(record, run_dir, compact_files, old_files):
            changed_records += 1

    if changed_records:
        _write_json(run_status_path, run_status)
        _write_iterations_jsonl(iterations_path, history)
        if not args.keep_old:
            for old_file in sorted(old_files):
                if old_file.exists() and old_file not in compact_files:
                    old_file.unlink()

    after = _npz_size(run_dir)
    print(f"run_dir={run_dir}")
    print(f"records_compacted={changed_records}")
    print(f"compact_files={len(compact_files)}")
    print(f"old_files_removed={0 if args.keep_old else len([p for p in old_files if not p.exists()])}")
    print(f"size_before_mb={before / 1024 / 1024:.2f}")
    print(f"size_after_mb={after / 1024 / 1024:.2f}")
    print(f"saved_mb={(before - after) / 1024 / 1024:.2f}")
    return 0


def _resolve_run_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_iterations_jsonl(path: Path, history: list[dict[str, Any]]) -> None:
    lines = [json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in history]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _npz_size(run_dir: Path) -> int:
    return sum(path.stat().st_size for path in run_dir.rglob("*.npz"))


def _compact_record(record: dict[str, Any], run_dir: Path, compact_files: set[Path], old_files: set[Path]) -> bool:
    iteration = int(record.get("iteration") or 0)
    scope_result = record.get("scope_result") if isinstance(record.get("scope_result"), dict) else None
    if iteration <= 0 or not scope_result:
        return False
    waveforms = scope_result.get("waveforms") if isinstance(scope_result.get("waveforms"), list) else []
    source_payloads: dict[str, dict[str, Any]] = {}
    for waveform in waveforms:
        if not isinstance(waveform, dict):
            continue
        source = str(waveform.get("source") or "").upper()
        if not source:
            continue
        data_file = _resolve_data_file(waveform.get("data_file"), run_dir)
        if not data_file or not data_file.exists():
            continue
        loaded = _load_scope_npz(data_file, source)
        if loaded is None:
            continue
        source_payloads[source] = loaded
        old_files.add(data_file)

    if not source_payloads:
        return False

    compact_path = run_dir / "files" / f"iteration_{iteration:03d}_scope.npz"
    _write_compact_npz(compact_path, record, scope_result, source_payloads)
    compact_files.add(compact_path)
    rel = str(compact_path.relative_to(ROOT)).replace("\\", "/")
    changed = False
    for waveform in waveforms:
        if not isinstance(waveform, dict):
            continue
        source = str(waveform.get("source") or "").upper()
        if source in source_payloads and waveform.get("data_file") != rel:
            waveform["data_file"] = rel
            waveform["data_file_pending"] = False
            changed = True
    return changed


def _resolve_data_file(value: Any, run_dir: Path) -> Path | None:
    if not value:
        return None
    text = str(value).replace("\\", "/").lstrip("/")
    path = Path(text)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([ROOT / path, run_dir / path.name, run_dir / "files" / path.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else None


def _load_scope_npz(path: Path, source: str) -> dict[str, Any] | None:
    with np.load(path, allow_pickle=False) as payload:
        if "format_version" in payload.files and int(np.asarray(payload["format_version"]).item()) >= 2:
            safe = _safe_source(source)
            y_key = f"y_{safe}"
            if y_key not in payload.files:
                return None
            points = int(np.asarray(payload["points"]).item())
            x_start = float(np.asarray(payload["x_start"]).item())
            x_increment = float(np.asarray(payload["x_increment"]).item())
            x = x_start + np.arange(points, dtype=np.float64) * x_increment
            y = np.asarray(payload[y_key], dtype=np.float32)
            return {
                "x": x,
                "y": y,
                "x_unit": str(np.asarray(payload["x_unit"]).item()) if "x_unit" in payload.files else "s",
                "y_unit": str(np.asarray(payload[f"y_unit_{safe}"]).item()) if f"y_unit_{safe}" in payload.files else "V",
                "original_points": int(np.asarray(payload[f"original_points_{safe}"]).item())
                if f"original_points_{safe}" in payload.files
                else int(len(y)),
                "transfer_encoding": str(np.asarray(payload[f"transfer_encoding_{safe}"]).item())
                if f"transfer_encoding_{safe}" in payload.files
                else "",
            }
        if "x" not in payload.files or "y" not in payload.files:
            return None
        return {
            "x": np.asarray(payload["x"], dtype=np.float64),
            "y": np.asarray(payload["y"], dtype=np.float32),
            "x_unit": str(np.asarray(payload["x_unit"]).item()) if "x_unit" in payload.files else "s",
            "y_unit": str(np.asarray(payload["y_unit"]).item()) if "y_unit" in payload.files else "V",
            "original_points": int(np.asarray(payload["original_points"]).item()) if "original_points" in payload.files else int(len(payload["y"])),
            "transfer_encoding": str(np.asarray(payload["transfer_encoding"]).item()) if "transfer_encoding" in payload.files else "",
        }


def _write_compact_npz(path: Path, record: dict[str, Any], scope_result: dict[str, Any], sources: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_sources = sorted(sources)
    first = sources[ordered_sources[0]]
    x = np.asarray(first["x"], dtype=np.float64)
    points = int(x.size)
    x_start = float(x[0]) if points else 0.0
    x_increment = float((x[-1] - x[0]) / float(points - 1)) if points > 1 else 0.0
    payload: dict[str, Any] = {
        "format_version": np.array(2, dtype=np.int16),
        "sources": np.asarray(ordered_sources, dtype="U8"),
        "x_start": np.array(x_start, dtype=np.float64),
        "x_increment": np.array(x_increment, dtype=np.float64),
        "points": np.array(points, dtype=np.int64),
        "x_unit": np.asarray(first.get("x_unit") or "s"),
        "capture_id": np.asarray(str(scope_result.get("capture_id") or "")),
        "timestamp": np.array(float(record.get("timestamp") or 0.0), dtype=np.float64),
    }
    for source in ordered_sources:
        safe = _safe_source(source)
        item = sources[source]
        y = np.asarray(item["y"], dtype=np.float32)
        if int(y.size) != points:
            count = min(points, int(y.size))
            y = y[:count]
            payload["points"] = np.array(count, dtype=np.int64)
        payload[f"y_{safe}"] = y
        payload[f"y_unit_{safe}"] = np.asarray(item.get("y_unit") or "V")
        payload[f"original_points_{safe}"] = np.array(int(item.get("original_points") or len(y)), dtype=np.int64)
        payload[f"transfer_encoding_{safe}"] = np.asarray(item.get("transfer_encoding") or "")
    np.savez(path, **payload)


def _safe_source(source: str) -> str:
    return "".join(char for char in source.upper() if char.isalnum() or char in {"_", "-"})


if __name__ == "__main__":
    raise SystemExit(main())
