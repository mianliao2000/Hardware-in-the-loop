"""Recompute saved auto-tune metrics from captured waveform files.

This intentionally rewrites the run files in place after creating a one-time,
versioned backup of the previous metrics/results.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.tuning.analyzer import (
    SETTLING_ANALYSIS_VERSION,
    ResponseAnalyzer,
    score_metrics,
)
from hardware.instruments.bode_analyzer import calculate_stability_margins
from hardware.tuning.models import ResponseMetrics, Waveform, bandwidth_objective, to_jsonable
from hardware.tuning.runner import _config_from_payload, _record_from_payload
from hardware.tuning.search import select_best_result, select_diverse_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/autotune_runs")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument(
        "--iteration",
        action="append",
        type=int,
        default=[],
        help="Recompute only this iteration number; repeat for multiple records.",
    )
    parser.add_argument("--all", action="store_true", help="Recompute every run below --root.")
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Skip metrics and only force-regenerate stored scope and Bode PNGs.",
    )
    parser.add_argument(
        "--rebuild-images",
        action="store_true",
        help="Force-regenerate stored scope and Bode PNGs (implied when metrics are recomputed).",
    )
    parser.add_argument(
        "--scope-images-only",
        action="store_true",
        help="When rebuilding images, regenerate Scope PNGs only and preserve existing Bode PNGs.",
    )
    parser.add_argument(
        "--no-image-status-write",
        action="store_true",
        help="Do not rewrite run metadata after image generation; useful for disjoint image workers.",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Recompute metrics without rebuilding PNGs; annotations may then be stale.",
    )
    parser.add_argument(
        "--bode-shape-only",
        action="store_true",
        help=(
            "Recompute Bode margins/shape and the combined score while reusing stored transient "
            "metrics. This avoids loading large scope captures."
        ),
    )
    args = parser.parse_args()
    if args.images_only and args.skip_images:
        parser.error("--images-only and --skip-images cannot be used together")

    root = Path(args.root).resolve()
    runs = (
        [Path(item).resolve() for item in args.run]
        if args.run
        else _find_runs(root, None if args.all else args.iterations)
    )
    if not runs:
        selection = "runs" if args.all else f"runs with {args.iterations} iterations"
        print(f"No {selection} found under {root}.")
        return 1

    updated = 0
    for run_dir in runs:
        if not args.images_only:
            print(f"recomputing metrics: {run_dir}", flush=True)
            count = _recompute_run(
                run_dir,
                bode_shape_only=bool(args.bode_shape_only),
                selected_iterations=set(args.iteration) or None,
                create_backup=True,
            )
            updated += count
            print(f"updated {count:3d} iterations: {run_dir}", flush=True)
        rebuild_images = args.images_only or args.rebuild_images or (
            not args.images_only and not args.skip_images
        )
        if rebuild_images:
            print(f"rebuilding images:   {run_dir}", flush=True)
            images, errors = _rebuild_run_images(
                run_dir,
                selected_iterations=set(args.iteration) or None,
                scope_only=bool(args.scope_images_only),
                persist_status=not bool(args.no_image_status_write),
            )
            print(f"rebuilt {images:3d} images:     {run_dir}", flush=True)
            for error in errors:
                print(f"  warning: {error}")
    if args.images_only:
        print(f"done: rebuilt stored images across {len(runs)} run(s).")
    else:
        print(f"done: recomputed {updated} iteration records across {len(runs)} run(s).")
    return 0


def _find_runs(root: Path, iteration_count: int | None) -> list[Path]:
    runs: list[Path] = []
    status_files = list(root.glob("*/run_status.json"))
    status_files.extend(root.glob("recent/*/run_status.json"))
    status_files.extend(root.glob("saved/*/run_status.json"))
    if (root / "run_status.json").is_file():
        status_files.append(root / "run_status.json")
    for status_file in sorted(set(status_files)):
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        history = status.get("history")
        if isinstance(history, list) and (iteration_count is None or len(history) == iteration_count):
            runs.append(status_file.parent)
    return runs


def _recompute_run(
    run_dir: Path,
    *,
    bode_shape_only: bool = False,
    selected_iterations: set[int] | None = None,
    create_backup: bool = False,
) -> int:
    status_path = run_dir / "run_status.json"
    if create_backup:
        _backup_run_files(run_dir)
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
    total = len(history)
    updated = 0
    found_iterations: set[int] = set()
    for index, record in enumerate(history, start=1):
        if not isinstance(record, dict):
            continue
        iteration = int(record.get("iteration") or 0)
        if selected_iterations is not None and iteration not in selected_iterations:
            new_history.append(record)
            continue
        found_iterations.add(iteration)
        waveform = None if bode_shape_only else _waveform_from_scope_files(record, run_dir)
        margins = _bode_margins(record, run_dir)
        if bode_shape_only and enable_transient:
            transient = _stored_transient_metrics(record, config.targets)
            metrics = analyzer.analyze_hardware(
                None,
                margins,
                enable_transient=True,
                enable_bode=enable_bode,
                precomputed_transient=transient,
            )
        elif enable_transient and not _waveform_is_valid(waveform):
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
        parsed_record = _record_from_payload(next_record)
        both_analyses = enable_transient and enable_bode
        objective, bonus = bandwidth_objective(
            metrics.score,
            parsed_record.candidate,
            passed=metrics.passed,
            both_analyses_enabled=both_analyses,
        )
        next_record["objective_score"] = objective
        next_record["bandwidth_bonus"] = bonus
        if margins and isinstance(next_record.get("bode_result"), dict):
            next_record["bode_result"] = dict(next_record["bode_result"])
            next_record["bode_result"]["margins"] = margins
        new_history.append(next_record)
        updated += 1
        if index == total or index % 10 == 0:
            print(f"  metrics {index:4d}/{total}", flush=True)

    if selected_iterations is not None:
        missing = sorted(selected_iterations - found_iterations)
        if missing:
            raise RuntimeError(f"Iterations not found in {run_dir}: {missing}")

    status["history"] = new_history
    if new_history:
        status["current"] = new_history[-1]
        records = [_record_from_payload(item) for item in new_history]
        best = select_best_result(records)
        status["best"] = _record_by_iteration(new_history, best.iteration) if best else None
        status["recommendations"] = [
            _record_by_iteration(new_history, record.iteration)
            for record in select_diverse_results(records, 5)
            if _record_by_iteration(new_history, record.iteration) is not None
        ]

    _write_json(status_path, status)
    _write_iterations_jsonl(run_dir / "iterations.jsonl", new_history)
    _update_summary(run_dir, status)
    return updated


def _stored_transient_metrics(record: dict[str, Any], targets: Any) -> ResponseMetrics:
    """Rebuild the transient-only portion from stored scalar measurements."""

    payload = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    reasons = [str(item) for item in (payload.get("pass_reasons") or [])]
    invalid_reasons = [item for item in reasons if "invalid transient waveform" in item.lower()]
    overshoot = float(payload.get("overshoot_pct") or 0.0)
    undershoot = float(payload.get("undershoot_pct") or 0.0)
    overshoot_ts = float(payload.get("overshoot_settling_time_s") or 0.0)
    undershoot_ts = float(payload.get("undershoot_settling_time_s") or 0.0)
    oscillations = int(payload.get("oscillations") or 0)
    invalid = bool(invalid_reasons)
    score = 300.0 if invalid else score_metrics(
        overshoot,
        undershoot,
        oscillations,
        overshoot_ts,
        undershoot_ts,
        targets,
    )
    passed = bool(
        not invalid
        and overshoot <= targets.overshoot_pct
        and undershoot <= targets.undershoot_pct
        and overshoot_ts <= targets.settling_time_s
        and undershoot_ts <= targets.settling_time_s
    )
    return ResponseMetrics(
        overshoot_pct=overshoot,
        undershoot_pct=undershoot,
        settling_time_s=max(overshoot_ts, undershoot_ts),
        oscillations=oscillations,
        score=score,
        passed=passed,
        overshoot_settling_time_s=overshoot_ts,
        undershoot_settling_time_s=undershoot_ts,
        low_load_steady_v=_optional_float(payload.get("low_load_steady_v")),
        high_load_steady_v=_optional_float(payload.get("high_load_steady_v")),
        pass_reasons=invalid_reasons,
    )


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


def _bode_margins(record: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    bode_result = record.get("bode_result") if isinstance(record.get("bode_result"), dict) else {}
    data_file = _resolve_data_file(bode_result.get("data_file"), run_dir)
    if data_file and data_file.exists():
        try:
            with np.load(data_file, allow_pickle=False) as payload:
                calculated = calculate_stability_margins(
                    np.asarray(payload["frequency_hz"], dtype=np.float64).tolist(),
                    np.asarray(payload["magnitude_db"], dtype=np.float64).tolist(),
                    np.asarray(payload["phase_deg"], dtype=np.float64).tolist(),
                )
            return calculated.as_dict()
        except Exception:
            pass
    margins = bode_result.get("margins") if isinstance(bode_result.get("margins"), dict) else {}
    return dict(margins)


def _rebuild_run_images(
    run_dir: Path,
    selected_iterations: set[int] | None = None,
    *,
    scope_only: bool = False,
    persist_status: bool = True,
) -> tuple[int, list[str]]:
    # Importing the GUI plotters here keeps metrics-only use of this script light.
    from gui.server import AutotuneRunStore, _scope_axis_settings_from_status

    status_path = run_dir / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    history = status.get("history") if isinstance(status.get("history"), list) else []
    store = AutotuneRunStore(
        PROJECT_ROOT / "results" / "autotune_runs" / "recent",
        PROJECT_ROOT / "results" / "autotune_runs" / "saved",
        recent_limit=5,
    )
    axis_settings = _scope_axis_settings_from_status(status)
    rebuilt = 0
    errors: list[str] = []
    total = len(history)
    for index, record in enumerate(history, start=1):
        if not isinstance(record, dict):
            continue
        iteration = int(record.get("iteration") or 0)
        if selected_iterations is not None and iteration not in selected_iterations:
            continue
        try:
            if scope_only:
                scope_result = (
                    record.get("scope_result")
                    if isinstance(record.get("scope_result"), dict)
                    else None
                )
                if scope_result is not None:
                    files_dir = run_dir / "files"
                    files_dir.mkdir(parents=True, exist_ok=True)
                    store._copy_scope_channel_data_files(scope_result, files_dir, iteration)
                    store._rebuild_scope_png_from_record(
                        scope_result,
                        files_dir / f"iteration_{iteration:03d}_scope.png",
                        iteration,
                        axis_settings,
                        record.get("metrics") if isinstance(record.get("metrics"), dict) else None,
                    )
                    record["scope_result"] = scope_result
            else:
                store._copy_record_assets(
                    record,
                    run_dir,
                    scope_axis_settings=axis_settings,
                    force_rebuild=True,
                )
            scope_result = record.get("scope_result") if isinstance(record.get("scope_result"), dict) else {}
            bode_result = record.get("bode_result") if isinstance(record.get("bode_result"), dict) else {}
            rebuilt += int(bool(scope_result.get("scope_png")))
            if not scope_only:
                rebuilt += int(bool(bode_result.get("bode_png")))
        except Exception as exc:
            errors.append(f"iteration {iteration}: {exc}")
        if index == total or index % 10 == 0:
            print(f"  images  {index:4d}/{total}", flush=True)

    by_iteration = {
        int(record.get("iteration") or 0): record
        for record in history
        if isinstance(record, dict)
    }
    for key in ("current", "best"):
        record = status.get(key)
        if isinstance(record, dict):
            replacement = by_iteration.get(int(record.get("iteration") or 0))
            if replacement is not None:
                status[key] = replacement
    if persist_status:
        _write_json(status_path, status)
        # Keep the append-log representation consistent with run_status. Without
        # this write, a later repair/load path could resurrect pre-rebuild PNG
        # references even though the visible status had already been updated.
        _write_iterations_jsonl(run_dir / "iterations.jsonl", history)
        store._write_summary(run_dir, status)
    return rebuilt, errors


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
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


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
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _backup_run_files(run_dir: Path) -> None:
    suffix = f"pre_settling_v{SETTLING_ANALYSIS_VERSION}"
    for source_name, backup_name in (
        ("run_status.json", f"run_status.{suffix}.json"),
        ("iterations.jsonl", f"iterations.{suffix}.jsonl"),
        ("summary.json", f"summary.{suffix}.json"),
    ):
        source = run_dir / source_name
        backup = run_dir / backup_name
        if source.is_file() and not backup.exists():
            shutil.copy2(source, backup)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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
