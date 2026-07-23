"""Run the offline, resumable DRL network-capacity study.

Examples:
  python scripts/sweep_drl_networks.py --dry-run
  python scripts/sweep_drl_networks.py --quick --output-dir results/autotune_ml/network_sweeps/smoke
  python scripts/sweep_drl_networks.py --output-dir results/autotune_ml/network_sweeps/overnight

The command never opens a hardware adapter and never changes the active model.
Reusing the same output directory resumes completed trials after verifying the
frozen dataset hash.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.tuning.drl.sweep import OfflineNetworkSweep, SweepSettings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare surrogate and Safe SAC network capacities using archived exact-condition data only. "
            "This is an offline research command; it cannot activate a model or access hardware."
        )
    )
    parser.add_argument(
        "--run-root",
        action="append",
        type=Path,
        default=None,
        help=(
            "Archived run root. Repeat to add roots. Defaults to both "
            "results/autotune_runs/saved and results/autotune_runs/recent."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Study directory. Reuse the same path to resume. The default creates a timestamped directory "
            "below results/autotune_ml/network_sweeps."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="STUDY_DIR",
        help="Explicit alias for resuming an existing --output-dir; completed trial JSON files are skipped.",
    )
    parser.add_argument(
        "--phase",
        choices=("surrogate", "sac", "all"),
        default="all",
        help="Run the surrogate stage, SAC stage, or both in order.",
    )
    parser.add_argument("--max-hours", type=float, default=12.0, help="Stop safely between trials after this many hours.")
    parser.add_argument("--threads", type=int, default=4, help="CPU thread cap (hard limited to four).")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a tiny three-architecture smoke study with minimal epochs/steps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Freeze and validate the dataset/splits and write the study plan without training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.resume is not None and args.output_dir is not None:
        raise SystemExit("Use either --resume STUDY_DIR or --output-dir STUDY_DIR, not both.")
    roots = tuple(
        Path(path).resolve()
        for path in (
            args.run_root
            or [
                REPO_ROOT / "results" / "autotune_runs" / "saved",
                REPO_ROOT / "results" / "autotune_runs" / "recent",
            ]
        )
    )
    output = args.resume or args.output_dir
    if output is None:
        output = REPO_ROOT / "results" / "autotune_ml" / "network_sweeps" / time.strftime("sweep_%Y%m%d_%H%M%S")
    settings = SweepSettings(
        output_dir=Path(output).resolve(),
        run_roots=roots,
        max_wall_time_s=max(1.0, float(args.max_hours) * 60.0 * 60.0),
        cpu_threads=min(4, max(1, int(args.threads))),
        seed=int(args.seed),
        batch_size=max(8, int(args.batch_size)),
        phase=str(args.phase),
        quick=bool(args.quick),
        dry_run=bool(args.dry_run),
    )
    print("DRL capacity sweep: OFFLINE ONLY (hardware and active model are untouched)", flush=True)
    print(f"Output: {settings.output_dir}", flush=True)
    result = OfflineNetworkSweep(settings).run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") in {"complete", "dry_run_complete", "pending"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
