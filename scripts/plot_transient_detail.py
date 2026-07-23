from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.tuning.analyzer import ResponseAnalyzer
from hardware.tuning.models import TuningTargets, Waveform


COLORS = {
    "raw": "#9aa0a6",
    "filtered": "#d93025",
    "band": "#1a73e8",
    "settled": "#188038",
    "edge": "#5f6368",
    "grid": "#d9e2ef",
}


def _downsample_envelope(
    time_us: np.ndarray, values: np.ndarray, max_points: int = 7000
) -> tuple[np.ndarray, np.ndarray]:
    if values.size <= max_points:
        return time_us, values
    bin_size = int(np.ceil(values.size / max_points))
    usable = values.size - values.size % bin_size
    if usable <= 0:
        return time_us, values
    grouped_t = time_us[:usable].reshape(-1, bin_size)
    grouped_y = values[:usable].reshape(-1, bin_size)
    x = np.repeat(grouped_t[:, grouped_t.shape[1] // 2], 2)
    y = np.column_stack((grouped_y.min(axis=1), grouped_y.max(axis=1))).reshape(-1)
    return x, y


def _final_band_entry_us(
    time_us: np.ndarray,
    values: np.ndarray,
    *,
    steady_v: float,
    lower_mv: float,
    upper_mv: float,
    main_watch_us: float,
    minimum_exit_depth_mv: float = 0.5,
    minimum_exit_duration_us: float = 0.08,
    late_exit_after_us: float = 5.0,
    late_minimum_exit_duration_us: float = 3.0,
) -> float | None:
    decision = (time_us >= 0.0) & (time_us <= main_watch_us + 1.0)
    x = time_us[decision]
    y = values[decision]
    if x.size < 2:
        return None
    outside = (y < steady_v - lower_mv * 1e-3) | (y > steady_v + upper_mv * 1e-3)
    initial_outside = np.flatnonzero(outside)
    if initial_outside.size == 0:
        return None
    entries = np.flatnonzero((~outside) & (np.arange(outside.size) > initial_outside[0]))
    if entries.size == 0:
        return None
    first_entry = int(entries[0])
    final_entry = first_entry
    sample_dt_us = float(np.median(np.diff(x)))
    index = first_entry + 1
    excursions: list[tuple[int, int]] = []
    while index < x.size and x[index] <= main_watch_us:
        if not outside[index]:
            index += 1
            continue
        exit_index = index
        while index < x.size and outside[index]:
            index += 1
        reentry_index = min(index, x.size - 1)
        if excursions and x[exit_index] - x[excursions[-1][1]] <= 0.05 + sample_dt_us:
            excursions[-1] = (excursions[-1][0], reentry_index)
        else:
            excursions.append((exit_index, reentry_index))
    for exit_index, reentry_index in excursions:
        cluster = y[exit_index : reentry_index + 1]
        below_depth = steady_v - cluster - lower_mv * 1e-3
        above_depth = cluster - steady_v - upper_mv * 1e-3
        depth_v = float(np.maximum(below_depth, above_depth).max())
        duration_us = float(x[reentry_index] - x[exit_index])
        required_duration_us = (
            late_minimum_exit_duration_us
            if float(x[exit_index]) >= late_exit_after_us - sample_dt_us
            else minimum_exit_duration_us
        )
        if (
            duration_us + sample_dt_us >= required_duration_us
            and depth_v >= minimum_exit_depth_mv * 1e-3 - 1e-12
        ):
            final_entry = reentry_index
    return max(sample_dt_us, float(x[final_entry]))


def _plot_event(
    axis,
    *,
    time_s: np.ndarray,
    raw_v: np.ndarray,
    filtered_v: np.ndarray,
    edge_time_s: float,
    diagnostics: dict[str, Any],
    title: str,
    lower_tolerance_mv: float | None = None,
) -> None:
    relative_us = (time_s - edge_time_s) * 1e6
    display_us = 30.0
    mask = (relative_us >= -1.0) & (relative_us <= display_us)
    x = relative_us[mask]
    raw = raw_v[mask]
    filtered = filtered_v[mask]
    steady_v = float(diagnostics["local_steady_v"])
    lower_mv = (
        float(lower_tolerance_mv)
        if lower_tolerance_mv is not None
        else float(diagnostics["lower_tolerance_mv"])
    )
    upper_mv = float(diagnostics["upper_tolerance_mv"])
    lower_v = steady_v - lower_mv * 1e-3
    upper_v = steady_v + upper_mv * 1e-3
    valid_ts = bool(diagnostics.get("valid", True))
    settled_us = (
        _final_band_entry_us(
            relative_us,
            filtered_v,
            steady_v=steady_v,
            lower_mv=lower_mv,
            upper_mv=upper_mv,
            main_watch_us=float(diagnostics["main_watch_us"]),
            # A manually overridden display band should report its literal
            # final re-entry. The analyzer's amplitude allowance belongs to
            # the original metric definition, not to this detail overlay.
            minimum_exit_depth_mv=0.0 if lower_tolerance_mv is not None else 0.5,
            minimum_exit_duration_us=float(diagnostics.get("minimum_exit_duration_us", 0.08)),
            late_exit_after_us=float(diagnostics.get("late_exit_after_us", 5.0)),
            late_minimum_exit_duration_us=float(
                diagnostics.get("late_minimum_exit_duration_us", 3.0)
            ),
        )
        if valid_ts
        else None
    )

    raw_x, raw_y = _downsample_envelope(x, raw)
    axis.plot(raw_x, raw_y, color=COLORS["raw"], linewidth=0.55, alpha=0.48, label="Raw Vout")
    axis.fill_between(
        [0.0, display_us],
        lower_v,
        upper_v,
        color=COLORS["band"],
        alpha=0.14,
        label=(
            f"Settling band "
            f"(-{lower_mv:.0f}/+{upper_mv:.0f} mV)"
        ),
        zorder=0,
    )
    axis.axhline(lower_v, color=COLORS["band"], linewidth=1.0, linestyle="--", alpha=0.8)
    axis.axhline(upper_v, color=COLORS["band"], linewidth=1.0, linestyle="--", alpha=0.8)
    axis.axhline(steady_v, color=COLORS["band"], linewidth=0.9, alpha=0.65)
    axis.plot(x, filtered, color=COLORS["filtered"], linewidth=1.8, label="Vout, 600 kHz LPF")
    axis.axvline(0.0, color=COLORS["edge"], linewidth=1.2, linestyle="--", label="Load edge")
    if settled_us is not None:
        axis.axvline(
            settled_us,
            color=COLORS["settled"],
            linewidth=1.4,
            linestyle="--",
            label=f"Final band entry: {settled_us:.2f} µs",
        )
    else:
        excursions = diagnostics.get("secondary_excursions") or []
        if excursions:
            last_excursion = excursions[-1]
            exit_us = float(last_excursion["exit_us"])
            reentry_us = float(last_excursion["reentry_us"])
            axis.axvspan(exit_us, reentry_us, color="#f9ab00", alpha=0.15, zorder=1)
            axis.axvline(
                reentry_us,
                color="#f29900",
                linewidth=1.4,
                linestyle="--",
                label=f"Last re-entry: {reentry_us:.2f} µs — no 1 µs dwell",
            )
        axis.text(
            0.985,
            0.03,
            f"No valid Ts\n{diagnostics.get('reason', 'invalid settling')}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="#b3261e",
            fontsize=9,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#fce8e6",
                "edgecolor": "#ea4335",
                "alpha": 0.95,
            },
        )

    if "primary_extreme_us" in diagnostics and "primary_extreme_v" in diagnostics:
        extreme_us = float(diagnostics["primary_extreme_us"])
        extreme_v = float(diagnostics["primary_extreme_v"])
    else:
        transient = (x >= 0.0) & (x <= float(diagnostics["main_watch_us"]))
        candidates = np.flatnonzero(transient)
        if diagnostics.get("edge") == "rising":
            extreme_index = candidates[int(np.argmin(filtered[candidates]))]
        else:
            extreme_index = candidates[int(np.argmax(filtered[candidates]))]
        extreme_us = float(x[extreme_index])
        extreme_v = float(filtered[extreme_index])
    axis.scatter(
        [extreme_us],
        [extreme_v],
        color=COLORS["filtered"],
        edgecolor="white",
        linewidth=0.8,
        s=42,
        zorder=5,
    )

    # Keep the detail view centered on the post-step steady state so that the
    # millivolt-scale settling band and secondary excursions remain visible.
    y_span_v = 0.012
    axis.set_xlim(-1.0, display_us)
    axis.set_ylim(steady_v - y_span_v, steady_v + y_span_v)
    axis.set_title(title, fontsize=13, fontweight="bold")
    axis.set_xlabel("Time from load edge (µs)")
    axis.grid(True, color=COLORS["grid"], linewidth=0.8, alpha=0.85)
    axis.legend(loc="best", fontsize=8.6, framealpha=0.95)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot falling/rising Vout transient details and asymmetric settling bands."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files_dir = args.run_dir / "files"
    scope_path = files_dir / f"iteration_{args.iteration}_scope.npz"
    if not scope_path.is_file():
        scope_path = files_dir / f"iteration_{args.iteration:03d}_scope.npz"
    output_path = args.output or files_dir / f"iteration_{args.iteration}_transient_detail.png"

    with np.load(scope_path, allow_pickle=False) as data:
        count = int(data["points"])
        time_s = float(data["x_start"]) + np.arange(count, dtype=np.float64) * float(data["x_increment"])
        load_v = np.asarray(data["y_CH1"], dtype=np.float64)
        raw_v = np.asarray(data["y_CH3"], dtype=np.float64)

    analyzer = ResponseAnalyzer(TuningTargets())
    filtered_v = np.asarray(
        analyzer._zero_phase_lowpass(
            time_s.tolist(),
            raw_v.tolist(),
            cutoff_hz=analyzer.RESPONSE_LOWPASS_CUTOFF_HZ,
        ),
        dtype=np.float64,
    )
    events = analyzer.input_edge_indices(load_v.tolist())
    current_metrics = analyzer.analyze(
        Waveform(
            time_s=time_s.tolist(),
            vout_v=raw_v.tolist(),
            input_v=load_v.tolist(),
        )
    )
    diagnostics = current_metrics.settling_diagnostics
    rising = next((index for index, edge in events if edge == "rising"), None)
    falling = next(
        (index for index, edge in events if edge == "falling" and rising is not None and index > rising),
        None,
    )
    if rising is None or falling is None:
        raise RuntimeError(f"expected rising and falling load edges, found: {events}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#3c4043",
            "axes.labelcolor": "#202124",
            "xtick.color": "#3c4043",
            "ytick.color": "#3c4043",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0), sharey=False)
    _plot_event(
        axes[0],
        time_s=time_s,
        raw_v=raw_v,
        filtered_v=filtered_v,
        edge_time_s=float(time_s[rising]),
        diagnostics=diagnostics["undershoot"],
        title="Load Step-Up — Undershoot",
        lower_tolerance_mv=4.0,
    )
    _plot_event(
        axes[1],
        time_s=time_s,
        raw_v=raw_v,
        filtered_v=filtered_v,
        edge_time_s=float(time_s[falling]),
        diagnostics=diagnostics["overshoot"],
        title="Load Step-Down — Overshoot",
    )
    axes[0].set_ylabel("Output voltage (V)")
    fig.suptitle(
        f"Iteration {args.iteration} — Transient Waveforms and Settling Bands",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), w_pad=2.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
