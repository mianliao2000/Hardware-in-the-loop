"""Pretrain and evaluate the offline DRL stack from saved hardware runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hardware.tuning.drl.common import artifact_id, atomic_write_json, candidate_key, operating_signature
from hardware.tuning.drl.dataset import DrlDataset, load_autotune_dataset
from hardware.tuning.drl.model import dependency_status, train_surrogate_ensemble
from hardware.tuning.drl.policy import train_safe_sac_policy
from hardware.tuning.models import AutotuneExperimentConfig, TuningConfig


def fixed_condition_experiment() -> AutotuneExperimentConfig:
    return AutotuneExperimentConfig(
        board_address="0x5E",
        board_page=0,
        board_adapter="xdp",
        response_channel="CH3",
        enable_bode_analysis=True,
        enable_transient_analysis=True,
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
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the surrogate on archived fixed-condition measurements and optionally "
            "pretrain Safe SAC. This command does not enable hardware inference or bypass validation."
        )
    )
    parser.add_argument(
        "--saved-root",
        action="append",
        type=Path,
        default=None,
        help="Saved run root. Repeat to combine roots. Defaults to results/autotune_runs/saved.",
    )
    parser.add_argument(
        "--include-recent",
        action="store_true",
        help="Also include non-permanent recent runs.",
    )
    parser.add_argument(
        "--include-run-id",
        action="append",
        default=None,
        help=(
            "Restrict loading to these exact run IDs. Repeat the option for a frozen run set; "
            "useful for auditable bootstrap + follow-up retraining."
        ),
    )
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help=(
            "Use only runs with complete fixed-condition metadata. By default, compatible legacy runs "
            "whose condition can be inferred are included to improve operating-point coverage."
        ),
    )
    parser.add_argument(
        "--validation-run-id",
        action="append",
        default=None,
        help=(
            "Use these complete run IDs as the explicit validation/evaluation partition. "
            "Candidate keys present in validation are purged from training."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "results" / "autotune_ml" / "pretraining",
    )
    parser.add_argument("--members", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--policy-steps",
        type=int,
        default=0,
        help="Safe SAC synthetic pretraining steps. Zero trains only the surrogate.",
    )
    parser.add_argument("--evaluation-episodes", type=int, default=1_000)
    parser.add_argument(
        "--hardware-protection-mode",
        action="store_true",
        help=(
            "Train a proposal policy even when the surrogate misses strict offline acceptance gates. "
            "The resulting policy is marked unverified and may only run with hardware trip recovery enabled."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    dependency = dependency_status()
    if not dependency.get("ok"):
        raise SystemExit(
            "ML dependencies are missing. Run: python -m pip install -r requirements-ml.txt\n"
            f"Details: {dependency.get('error')}"
        )

    roots = list(args.saved_root or [REPO_ROOT / "results" / "autotune_runs" / "saved"])
    if args.include_recent:
        roots.append(REPO_ROOT / "results" / "autotune_runs" / "recent")
    roots = [path.resolve() for path in roots]

    config = TuningConfig()
    experiment = fixed_condition_experiment()
    signature = operating_signature(config, experiment)
    dataset, dataset_manifest = load_autotune_dataset(
        roots,
        config,
        experiment,
        allow_legacy_inferred=not args.exact_only,
        include_run_ids=set(args.include_run_id) if args.include_run_id else None,
    )
    if dataset.size < 20:
        raise SystemExit(f"Only {dataset.size} compatible in-range samples were found; at least 20 are required.")
    train_indexes: np.ndarray | None = None
    validation_indexes: np.ndarray | None = None
    evaluation_indexes: np.ndarray | None = None
    if args.validation_run_id:
        train_indexes, validation_indexes = _explicit_validation_split(
            dataset,
            set(args.validation_run_id),
        )
        # Omit an explicit evaluation partition so the training API reuses the
        # complete validation runs, matching the default grouped workflow.
        evaluation_indexes = None

    run_id = artifact_id("offline_pretrain")
    artifact_dir = args.output_root.resolve() / run_id
    dataset_dir = artifact_dir / "dataset"
    model_dir = artifact_dir / "model"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    dataset_manifest.update(
        {
            "purpose": "offline_pretraining",
            "operating_signature": signature,
            "source_roots": [str(path) for path in roots],
        }
    )
    dataset.save(dataset_dir, dataset_manifest)

    started = time.perf_counter()

    def progress(value: float, message: str) -> None:
        print(f"[{value * 100:6.2f}%] {message}", flush=True)

    ensemble = train_surrogate_ensemble(
        dataset=dataset,
        config=config,
        artifact_dir=model_dir,
        operating_signature=signature,
        members=max(1, args.members),
        epochs=max(1, args.epochs),
        batch_size=max(8, args.batch_size),
        seed=args.seed,
        progress=progress,
        train_indexes=train_indexes,
        validation_indexes=validation_indexes,
        evaluation_indexes=evaluation_indexes,
    )
    policy_manifest: dict[str, Any] | None = None
    policy_message = "not requested"
    if args.policy_steps > 0:
        if ensemble.accepted or args.hardware_protection_mode:
            policy_manifest = train_safe_sac_policy(
                ensemble=ensemble,
                dataset=dataset,
                config=config,
                total_steps=args.policy_steps,
                evaluation_episodes=max(1, args.evaluation_episodes),
                seed=args.seed + 1,
                progress=progress,
                allow_unaccepted_surrogate=bool(args.hardware_protection_mode),
            )
            policy_message = "trained for hardware-protection mode" if args.hardware_protection_mode else "trained"
        else:
            policy_message = "blocked because surrogate acceptance gates failed"

    elapsed = time.perf_counter() - started
    report = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "elapsed_s": elapsed,
        "dependency": dependency,
        "dataset": {
            "sample_count": dataset.size,
            "exact_only": bool(args.exact_only),
            "included_run_ids": sorted(args.include_run_id or []),
            "explicit_validation_run_ids": sorted(args.validation_run_id or []),
            "source_record_count": dataset_manifest.get("source_record_count"),
            "excluded_incompatible_action_count": dataset_manifest.get("excluded_incompatible_action_count"),
            "excluded_out_of_search_space_count": dataset_manifest.get("excluded_out_of_search_space_count"),
            "source_runs": dataset_manifest.get("source_runs"),
        },
        "fixed_condition": asdict(experiment),
        "surrogate": {
            "accepted": ensemble.accepted,
            "model_id": ensemble.model_id,
            "training_sample_count": ensemble.manifest.get("training_sample_count"),
            "validation_sample_count": ensemble.manifest.get("validation_sample_count"),
            "training_groups": ensemble.manifest.get("training_groups"),
            "validation_groups": ensemble.manifest.get("validation_groups"),
            "acceptance": ensemble.manifest.get("acceptance"),
        },
        "policy": {
            "status": policy_message,
            "steps": int(args.policy_steps),
            "evaluation": (policy_manifest or {}).get("policy_evaluation"),
            "accepted": (policy_manifest or {}).get("policy_accepted"),
        },
        "hardware_ready": False,
        "hardware_protection_mode": bool(args.hardware_protection_mode),
        "note": "Offline pretraining never marks a policy hardware-ready; guarded hardware validation is still required.",
    }
    atomic_write_json(artifact_dir / "pretrain_report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if ensemble.accepted or args.hardware_protection_mode else 2


def _explicit_validation_split(
    dataset: DrlDataset,
    validation_run_ids: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a run-isolated split and purge repeated hardware raw keys."""

    validation_ids = {str(value) for value in validation_run_ids if str(value)}
    validation = np.asarray(
        [
            index
            for index, group in enumerate(dataset.groups.tolist())
            if str(group).split(":", 1)[-1] in validation_ids
        ],
        dtype=int,
    )
    if validation.size == 0:
        available = sorted({str(group).split(":", 1)[-1] for group in dataset.groups.tolist()})
        raise SystemExit(
            f"No rows matched --validation-run-id {sorted(validation_ids)}; available runs: {available}."
        )
    validation_keys = {candidate_key(dataset.candidates[index]) for index in validation}
    training = np.asarray(
        [
            index
            for index, group in enumerate(dataset.groups.tolist())
            if str(group).split(":", 1)[-1] not in validation_ids
            and candidate_key(dataset.candidates[index]) not in validation_keys
        ],
        dtype=int,
    )
    if training.size < 20 or validation.size < 5:
        raise SystemExit(
            "Explicit grouped split is too small after candidate-key purge: "
            f"training={training.size}, validation={validation.size}."
        )
    return training, validation


if __name__ == "__main__":
    raise SystemExit(main())
