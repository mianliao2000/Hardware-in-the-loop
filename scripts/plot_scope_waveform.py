from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hardware.instruments import TektronixOscilloscope, VisaConnectionError


DEFAULT_MSO58_RESOURCE = "USB0::0x0699::0x0522::B010536::INSTR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture MSO58 waveform data and plot it with matplotlib.")
    parser.add_argument("--resource", default=DEFAULT_MSO58_RESOURCE)
    parser.add_argument("--source", default="CH1")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--csv", default="results/mso58_ch1.csv")
    parser.add_argument("--png", default="results/mso58_ch1.png")
    parser.add_argument("--timeout-ms", type=int, default=5000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scope = TektronixOscilloscope(args.resource, timeout_ms=args.timeout_ms)
    try:
        scope.connect()
        try:
            scope.write("*CLS")
        except Exception:
            pass
        print(f"Connected: {scope.idn()}")
        waveform = scope.capture_ascii_waveform(args.source, args.start, args.stop)
        csv_path = waveform.save_csv(args.csv)
        png_path = plot_waveform(waveform.x, waveform.y, args.source, args.png)
        print(f"Captured {len(waveform.x)} points")
        print(f"CSV: {csv_path}")
        print(f"PNG: {png_path}")
    except VisaConnectionError as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        scope.close()
    return 0


def plot_waveform(x_vals: list[float], y_vals: list[float], source: str, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x_vals, y_vals, linewidth=1.2)
    ax.set_title(f"MSO58 {source} Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
