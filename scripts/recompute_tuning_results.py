"""Recompute saved auto-tune metrics from captured waveform files.

This intentionally overwrites the run files in place. It does not create backup
copies of old metrics/results.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.tuning.analyzer import ResponseAnalyzer
from hardware.tuning.models import ResponseMetrics, Waveform, to_jsonable
from hardware.tuning.runner import _config_from_payload, _record_from_payload
from hardware.tuning.search import select_best_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/autotune_runs")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--run", action="append", default=[])
    args = parser.parse_args()

    root = Path(args.root)
    runs = [Path(item) for item in args.run] if args.run else _find_runs(root, args.iterations)
    if not runs:
        print(f"No runs with {args.iterations} iterations found under {root}.")
        return 1

    updated = 0
    for run_dir in runs:
        count = _recompute_run(run_dir)
        updated += count
        print(f"updated {count:3d} iterations: {run_dir}")
    print(f"done: recomputed {updated} iteration records across {len(runs)} run(s).")
    return 0


def _find_runs(root: Path, iteration_count: int) -> list[Path]:
    runs: list[Path] = []
    for status_file in sorted(root.rglob("run_status.json")):
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        history = status.get("history")
        if isinstance(history, list) and len(history) == iteration_count:
            runs.append(status_file.parent)
    return runs


def _recompute_run(run_dir: Path) -> int:
    status_path = run_dir / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    history = status.get("history")
    if not isinstance(history, list):
        raise RuntimeError(f"{run_dir} has no history list.")

    config = _config_from_payload(status.get("config"))
    experiment = status.get("experiment") if isinstance(status.get("experiment"), dict) else {}
    enable_transient = bool(experiment.get("enable_transient_analysis", True))
    enable_bode = bool(experiment.get("enable_bode_analysis", True))

    analyzer = ResponseAnalyzer(config.targets)
    new_history: list[dict[str, Any]] = []
    for record in history:
        if not isinstance(record, dict):
            continue
        waveform = _waveform_from_scope_files(record, run_dir)
        margins = _bode_margins(record)
        if enable_transient and not _waveform_is_valid(waveform):
            metrics = _invalid_transient_metrics(record, margins)
        else:
            metrics = analyzer.analyze_hardware(
                waveform,
                margins,
                enable_transient=enable_transient,
                enable_bode=enable_bode,
            )
        next_record = dict(record)
        next_record["metrics"] = to_jsonable(metrics)
        new_history.append(next_record)

    status["history"] = new_history
    if new_history:
        status["current"] = new_history[-1]
        records = [_record_from_payload(item) for item in new_history]
        best = select_best_result(records)
        status["best"] = _record_by_iteration(new_history, best.iteration) if best else None

    _write_json(status_path, status)
    _write_iterations_jsonl(run_dir / "iterations.jsonl", new_history)
    _update_summary(run_dir, status)
    return len(new_history)


def _waveform_from_scope_files(record: dict[str, Any], run_dir: Path) -> Waveform:
    scope_result = record.get("scope_result") if isinstance(record.get("scope_result"), dict) else {}
    waveforms = scope_result.get("waveforms") if isinstance(scope_result.get("waveforms"), list) else []
    input_x: np.ndarray | None = None
    input_y: np.ndarray | None = None
    output_x: np.ndarray | None = None
    output_y: np.ndarray | None = None

    for item in waveforms:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").upper()
        data_file = _resolve_data_file(item.get("data_file"), run_dir)
        if not data_file or not data_file.exists():
            continue
        loaded = _load_scope_waveform_npz(data_file, source)
        if not loaded:
            continue
        x = loaded["x"]
        y = loaded["y"]
        if source == "CH1":
            input_x = x
            input_y = y
        elif source == "CH3":
            output_x = x
            output_y = y

    if output_x is None or output_y is None:
        payload_waveform = record.get("waveform") if isinstance(record.get("waveform"), dict) else {}
        return Waveform(
            time_s=[float(item) for item in payload_waveform.get("time_s", [])],
            vout_v=[float(item) for item in payload_waveform.get("vout_v", [])],
            input_v=[float(item) for item in payload_waveform.get("input_v", [])],
        )

    if input_y is None or input_x is None:
        input_interp = []
    elif len(input_x) == len(output_x) and np.allclose(input_x[[0, -1]], output_x[[0, -1]]):
        input_interp = input_y.tolist()
    else:
        input_interp = np.interp(output_x, input_x, input_y).tolist()

    return Waveform(time_s=output_x.tolist(), vout_v=output_y.tolist(), input_v=input_interp)


def _resolve_data_file(value: Any, run_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.exists():
        return path
    local = run_dir / path.name
    if local.exists():
        return local
    files_local = run_dir / "files" / path.name
    if files_local.exists():
        return files_local
    if "_scope_CH" in path.name:
        combined_name = path.name.split("_scope_CH", 1)[0] + "_scope.npz"
        combined_local = run_dir / "files" / combined_name
        if combined_local.exists():
            return combined_local
    return path


def _load_scope_waveform_npz(data_file: Path, source: str) -> dict[str, np.ndarray] | None:
    with np.load(data_file, allow_pickle=False) as payload:
        requested = str(source or "").upper()
        if "format_version" in payload.files and int(np.asarray(payload["format_version"]).item()) >= 2:
            sources = [str(item).upper() for item in np.asarray(payload["sources"]).tolist()] if "sources" in payload.files else []
            selected = requested if requested in sources else (sources[0] if sources else "")
            if not selected:
                return None
            safe_source = "".join(char for char in selected if char.isalnum() or char in {"_", "-"})
            y_key = f"y_{safe_source}"
            if y_key not in payload.files:
                return None
            points = int(np.asarray(payload["points"]).item()) if "points" in payload.files else int(len(payload[y_key]))
            x_start = float(np.asarray(payload["x_start"]).item()) if "x_start" in payload.files else 0.0
            x_increment = float(np.asarray(payload["x_increment"]).item()) if "x_increment" in payload.files else 0.0
            x = x_start + np.arange(points, dtype=np.float64) * x_increment
            y = np.asarray(payload[y_key], dtype=np.float64)
            count = min(int(x.size), int(y.size))
            return {"x": x[:count], "y": y[:count]}
        return {
            "x": np.asarray(payload["x"], dtype=np.float64),
            "y": np.asarray(payload["y"], dtype=np.float64),
        }


def _bode_margins(record: dict[str, Any]) -> dict[str, float]:
    bode_result = record.get("bode_result") if isinstance(record.get("bode_result"), dict) else {}
    margins = bode_result.get("margins") if isinstance(bode_result.get("margins"), dict) else {}
    return {str(key): float(value) for key, value in margins.items() if _is_number(value)}


def _waveform_is_valid(waveform: Waveform | None) -> bool:
    if waveform is None:
        return False
    return bool(waveform.time_s) and bool(waveform.vout_v) and len(waveform.time_s) == len(waveform.vout_v)


def _invalid_transient_metrics(record: dict[str, Any], margins: dict[str, float]) -> ResponseMetrics:
    reason = "invalid transient waveform"
    scope_result = record.get("scope_result") if isinstance(record.get("scope_result"), dict) else {}
    if scope_result.get("skipped"):
        reason = str(scope_result.get("reason") or scope_result.get("error") or "transient protection skipped")
    elif scope_result.get("error"):
        reason = str(scope_result.get("error"))
    return ResponseMetrics(
        overshoot_pct=0.0,
        undershoot_pct=0.0,
        settling_time_s=0.0,
        oscillations=0,
        score=300.0,
        passed=False,
        overshoot_settling_time_s=0.0,
        undershoot_settling_time_s=0.0,
        phase_margin_deg=_optional_float(margins.get("phase_margin_deg")),
        crossover_frequency_hz=_optional_float(margins.get("phase_crossover_hz")),
        gain_margin_db=_optional_float(margins.get("gain_margin_db")),
        pass_reasons=[reason],
    )


def _record_by_iteration(history: list[dict[str, Any]], iteration: int) -> dict[str, Any] | None:
    return next((item for item in history if int(item.get("iteration") or -1) == iteration), None)


def _write_iterations_jsonl(path: Path, history: list[dict[str, Any]]) -> None:
    lines = [json.dumps(item, ensure_ascii=False, separators=(",", ":")) for item in history]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _update_summary(run_dir: Path, status: dict[str, Any]) -> None:
    summary_path = run_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        summary = {}
    history = status.get("history") if isinstance(status.get("history"), list) else []
    best = status.get("best") if isinstance(status.get("best"), dict) else None
    current = status.get("current") if isinstance(status.get("current"), dict) else None
    summary.update(
        {
            "run_id": run_dir.name,
            "updated_at": time.time(),
            "state": status.get("state"),
            "message": status.get("message"),
            "iteration_count": len(history),
            "current_iteration": current.get("iteration") if current else None,
            "best_iteration": best.get("iteration") if best else None,
            "best_penalty": (best.get("metrics") or {}).get("score") if best else None,
        }
    )
    _write_json(summary_path, summary)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
