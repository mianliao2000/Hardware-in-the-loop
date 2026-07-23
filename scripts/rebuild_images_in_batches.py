"""Rebuild stored tuning images in disposable, memory-bounded batches."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()

    run_dir = Path(args.run).resolve()
    if not (run_dir / "run_status.json").is_file():
        parser.error(f"run_status.json not found below {run_dir}")
    if args.start <= 0 or args.stop < args.start or args.batch_size <= 0:
        parser.error("expected 0 < start <= stop and batch-size > 0")

    for batch_start in range(args.start, args.stop + 1, args.batch_size):
        batch_stop = min(args.stop, batch_start + args.batch_size - 1)
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "recompute_tuning_results.py"),
            "--run",
            str(run_dir),
            "--images-only",
            "--rebuild-images",
        ]
        for iteration in range(batch_start, batch_stop + 1):
            command.extend(("--iteration", str(iteration)))
        print(f"batch {batch_start}-{batch_stop}: starting", flush=True)
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            print(
                f"batch {batch_start}-{batch_stop}: failed with exit code {completed.returncode}",
                file=sys.stderr,
                flush=True,
            )
            return completed.returncode
        print(f"batch {batch_start}-{batch_stop}: complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
