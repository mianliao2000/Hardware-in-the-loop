"""Build a research-only DRL artifact for hardware-protection experiments.

This intentionally permits archived measurements from earlier fixed operating
conditions to bootstrap a structurally compatible policy.  It never marks the
artifact accepted, ready, hardware-ready, or deployable.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.tuning.drl.common import (  # noqa: E402
    artifact_id,
    atomic_write_json,
    candidate_to_mapping,
    operating_signature,
)
from hardware.tuning.drl.dataset import load_autotune_dataset  # noqa: E402
from hardware.tuning.drl.model import train_surrogate_ensemble  # noqa: E402
from hardware.tuning.drl.policy import (  # noqa: E402
    train_safe_sac_policy,
    validation_start_candidates,
)
from hardware.tuning.models import AutotuneExperimentConfig, TuningConfig  # noqa: E402


def experiment_config() -> AutotuneExperimentConfig:
    return AutotuneExperimentConfig(
        board_address="0x5E",
        board_page=0,
        board_adapter="xdp",
        response_channel="CH3",
        enable_bode_analysis=True,
        enable_transient_analysis=True,
        optimization_algorithm="deep-reinforcement",
        function_generator_config={
            "mode": "square",
            "frequency_hz": 10_000.0,
            "low_v": 0.1,
            "high_v": 1.1,
        },
        bode_config={
            "start_hz": 1_000.0,
            "stop_hz": 1_000_000.0,
            "points": 201,
            "bandwidth_hz": 300.0,
            "source_vpp": 0.1,
        },
        drl_hardware_protection_mode=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="")
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--policy-steps", type=int, default=5_000)
    parser.add_argument("--evaluation-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def directory_hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)).replace("\\", "/"): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.name != "manifest.json"
    }


def main() -> int:
    args = parse_args()
    model_id = args.model_id.strip() or artifact_id("exploratory_9d")
    model_dir = REPO_ROOT / "results" / "autotune_ml" / "models" / model_id
    if model_dir.exists():
        raise SystemExit(f"Artifact already exists: {model_dir}")

    config = TuningConfig()
    experiment = experiment_config()
    signature = operating_signature(config, experiment)
    roots = [
        REPO_ROOT / "results" / "autotune_runs" / "saved",
        REPO_ROOT / "results" / "autotune_runs" / "recent",
    ]
    # Deliberately pass experiment=None.  This is a research bootstrap from
    # archived measurements, not evidence that conditions are equivalent.
    dataset, dataset_manifest = load_autotune_dataset(
        roots,
        config,
        experiment=None,
        allow_legacy_inferred=True,
    )
    if dataset.size < 20:
        raise SystemExit(f"Only {dataset.size} in-range archived samples are available.")

    dataset_manifest.update(
        {
            "purpose": "hardware_protection_only_bootstrap",
            "source_condition_compatibility_not_enforced": True,
            "target_operating_signature": signature,
            "created_at": time.time(),
        }
    )
    dataset.save(model_dir / "dataset", dataset_manifest)

    def progress(value: float, message: str) -> None:
        print(f"[{value * 100:6.2f}%] {message}", flush=True)

    ensemble = train_surrogate_ensemble(
        dataset=dataset,
        config=config,
        artifact_dir=model_dir,
        operating_signature=signature,
        members=max(1, args.members),
        epochs=max(1, args.epochs),
        batch_size=128,
        seed=args.seed,
        progress=progress,
        hidden_sizes=(96, 64, 32),
        early_stopping_patience=20,
    )
    starts = validation_start_candidates(dataset, count=4)
    if not starts:
        raise SystemExit("No validation start candidate could be constructed.")
    atomic_write_json(
        model_dir / "validation_starts.json",
        {"candidates": [candidate_to_mapping(candidate) for candidate in starts]},
    )
    manifest = train_safe_sac_policy(
        ensemble=ensemble,
        dataset=dataset,
        config=config,
        total_steps=max(100, args.policy_steps),
        evaluation_episodes=max(1, args.evaluation_episodes),
        max_episode_steps=15,
        seed=args.seed + 1,
        progress=progress,
        allow_unaccepted_surrogate=True,
        actor_net_arch=(64, 64),
        critic_net_arch=(64, 64),
        checkpoint_interval=max(100, args.policy_steps),
        checkpoint_evaluation_episodes=min(100, max(1, args.evaluation_episodes)),
    )
    manifest.update(
        {
            "operating_signature": signature,
            "research_only": True,
            "offline_only": True,
            "accepted": False,
            "ready": False,
            "hardware_ready": False,
            "hardware_protection_policy": True,
            "source_condition_compatibility_not_enforced": True,
            "warning": (
                "Archived data only bootstraps proposals. Current-gain and 0.1-1.1 V behavior "
                "must be learned from hardware; this artifact is not deployable."
            ),
            "files_sha256": directory_hashes(model_dir),
        }
    )
    atomic_write_json(model_dir / "manifest.json", manifest)
    print(f"MODEL_ID={model_id}", flush=True)
    print(f"DATASET_SIZE={dataset.size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
