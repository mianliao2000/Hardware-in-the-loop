"""Offline, resumable capacity sweeps for the DRL surrogate and Safe SAC.

This module is deliberately isolated from the hardware workflow.  It only reads
archived run files and writes research artifacts below a caller supplied output
directory.  In particular, it never imports the hardware runner and never
updates the active-model pointer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import csv
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..models import AutotuneExperimentConfig, HardwarePidCandidate, TuningConfig
from .common import (
    ACTION_FIELDS,
    METRIC_FIELDS,
    atomic_write_json,
    candidate_key,
    candidate_to_mapping,
    metric_vector,
    operating_signature,
    relabeled_score,
)
from .dataset import DrlDataset, load_autotune_dataset


SURROGATE_ARCHITECTURES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("linear", ()),
    ("h8", (8,)),
    ("h16", (16,)),
    ("h32", (32,)),
    ("h16x16", (16, 16)),
    ("h32x16", (32, 16)),
    ("h32x32", (32, 32)),
    ("h64x32", (64, 32)),
    ("h64x64", (64, 64)),
    ("h64x64x32", (64, 64, 32)),
    ("h96x64x32", (96, 64, 32)),
    ("legacy_h128x128x64", (128, 128, 64)),
)

SAC_ARCHITECTURES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("linear", ()),
    ("h16", (16,)),
    ("h16x16", (16, 16)),
    ("h32", (32,)),
    ("h32x32", (32, 32)),
    ("h64", (64,)),
    ("h64x32", (64, 32)),
    ("current_h64x64", (64, 64)),
    ("h128x128", (128, 128)),
    ("legacy_h256x256", (256, 256)),
)


@dataclass(frozen=True)
class SweepSettings:
    """Resource and fidelity settings for one offline study."""

    output_dir: Path
    run_roots: tuple[Path, ...]
    max_wall_time_s: float = 12.0 * 60.0 * 60.0
    cpu_threads: int = 4
    seed: int = 20260715
    batch_size: int = 128
    phase: str = "all"
    quick: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class SplitSpec:
    outer_fold: int
    inner_fold: int
    train_indexes: np.ndarray
    early_stop_indexes: np.ndarray
    evaluation_indexes: np.ndarray
    train_groups: tuple[str, ...]
    early_stop_groups: tuple[str, ...]
    evaluation_groups: tuple[str, ...]
    split_hash: str


@dataclass(frozen=True)
class TrialSpec:
    kind: str
    stage: str
    architecture_name: str
    hidden_sizes: tuple[int, ...]
    seed: int
    members: int = 0
    epochs_or_steps: int = 0
    evaluation_episodes: int = 0
    outer_fold: int | None = None
    train_fraction: float = 1.0

    @property
    def trial_id(self) -> str:
        payload = {
            "kind": self.kind,
            "stage": self.stage,
            "architecture_name": self.architecture_name,
            "hidden_sizes": list(self.hidden_sizes),
            "seed": self.seed,
            "members": self.members,
            "epochs_or_steps": self.epochs_or_steps,
            "evaluation_episodes": self.evaluation_episodes,
            "outer_fold": self.outer_fold,
            "train_fraction": self.train_fraction,
        }
        digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]
        return f"{self.kind}_{self.stage}_{self.architecture_name}_{digest}"


def fixed_condition_experiment() -> AutotuneExperimentConfig:
    """Return the fixed operating point used by the archived DRL data."""

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


def build_exact_snapshot(
    run_roots: Iterable[Path],
    config: TuningConfig,
    experiment: AutotuneExperimentConfig,
    snapshot_dir: Path | None = None,
) -> tuple[DrlDataset, dict[str, Any]]:
    """Load exact-condition records and freeze leakage-safe baseline inputs.

    Only rows explicitly marked ``baseline`` provide baseline context. Baseline
    rows themselves are removed from supervised targets. Missing baseline values
    remain masked here and are filled from each trial's training fold later.
    """

    roots = tuple(Path(root).resolve() for root in run_roots)
    loaded, source_manifest = load_autotune_dataset(
        roots,
        config,
        experiment,
        allow_legacy_inferred=False,
    )
    run_hashes = _included_run_hashes(source_manifest)
    canonical_groups, duplicate_groups = _canonical_run_groups(loaded.groups, run_hashes)
    explicit_baselines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for group, record in zip(loaded.groups.tolist(), loaded.records):
        if str(record.get("phase") or "").strip().lower() != "baseline":
            continue
        metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
        explicit_baselines[str(group)] = metric_vector(metrics)

    keep = np.asarray(
        [
            index
            for index, record in enumerate(loaded.records)
            if str(record.get("phase") or "").strip().lower() != "baseline"
            and str(loaded.groups[index]) not in duplicate_groups
        ],
        dtype=int,
    )
    metric_count = len(METRIC_FIELDS)
    if keep.size:
        baseline_values = np.asarray(
            [explicit_baselines.get(str(loaded.groups[index]), (np.zeros(metric_count), np.zeros(metric_count)))[0]
             for index in keep],
            dtype=np.float32,
        )
        baseline_mask = np.asarray(
            [explicit_baselines.get(str(loaded.groups[index]), (np.zeros(metric_count), np.zeros(metric_count)))[1]
             for index in keep],
            dtype=np.float32,
        )
        actions = loaded.actions[keep].astype(np.float32, copy=True)
        # Zero is only a placeholder. The mask makes it unavailable, and every
        # trial replaces it with a median learned solely from its training fold.
        features = np.concatenate([actions, baseline_values * baseline_mask, baseline_mask], axis=1).astype(np.float32)
        records = [loaded.records[index] for index in keep]
        scores_and_passed = [relabeled_score(record, config.targets) for record in records]
        dataset = DrlDataset(
            features=features,
            actions=actions,
            metrics=loaded.metrics[keep].astype(np.float32, copy=True),
            metric_mask=loaded.metric_mask[keep].astype(np.float32, copy=True),
            invalid_labels=loaded.invalid_labels[keep].astype(np.float32, copy=True),
            scores=np.asarray([item[0] for item in scores_and_passed], dtype=np.float32),
            passed=np.asarray([item[1] for item in scores_and_passed], dtype=np.float32),
            groups=np.asarray([canonical_groups[str(loaded.groups[index])] for index in keep], dtype=str),
            candidates=[loaded.candidates[index] for index in keep],
            records=records,
            baseline_values=baseline_values,
            baseline_mask=baseline_mask,
        )
    else:
        dataset = _empty_dataset(
            loaded.features.shape[1]
            if loaded.features.ndim == 2
            else len(ACTION_FIELDS) + 2 * len(METRIC_FIELDS)
        )

    dataset_hash = hash_dataset(dataset)
    manifest: dict[str, Any] = {
        **source_manifest,
        "schema_version": 2,
        "purpose": "offline_network_capacity_sweep",
        "exact_condition_only": True,
        "allow_legacy_inferred": False,
        "sample_count": dataset.size,
        "excluded_explicit_baseline_target_count": int(loaded.size - dataset.size),
        "missing_explicit_baseline_sample_count": int(np.sum(np.max(dataset.baseline_mask, axis=1) <= 0))
        if dataset.size
        else 0,
        "baseline_policy": "explicit-only; target row excluded; train-fold median imputation",
        "penalty_policy": "recomputed with current relabeled_score; max=300; no hard minimum",
        "objective_policy": "passing-only LS/LR bandwidth bonus; max bonus=10 over capped range 47-79",
        "source_roots": [str(root) for root in roots],
        "run_content_hashes": run_hashes,
        "run_group_policy": "content-hash identity; duplicate archives removed",
        "duplicate_source_groups": sorted(duplicate_groups),
        "deduplicated_run_count": int(len(duplicate_groups)),
        "dataset_hash": dataset_hash,
        "group_count": int(len(set(dataset.groups.tolist()))),
        "groups": sorted(set(dataset.groups.tolist())),
        "operating_signature": operating_signature(config, experiment),
    }
    if snapshot_dir is not None:
        _save_snapshot(dataset, manifest, Path(snapshot_dir))
    return dataset, manifest


def hash_dataset(dataset: DrlDataset) -> str:
    digest = hashlib.sha256()
    for array in (
        dataset.actions,
        dataset.metrics,
        dataset.metric_mask,
        dataset.invalid_labels,
        dataset.scores,
        dataset.passed,
        dataset.baseline_values,
        dataset.baseline_mask,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    digest.update(_canonical_json(dataset.groups.tolist()).encode("utf-8"))
    digest.update(
        _canonical_json([list(candidate_key(candidate)) for candidate in dataset.candidates]).encode("utf-8")
    )
    return digest.hexdigest()


def make_grouped_splits(
    dataset: DrlDataset,
    outer_folds: int = 4,
    seed: int = 20260715,
    inner_fold_offset: int = 0,
) -> list[SplitSpec]:
    """Create outer run folds and candidate-purged inner early-stop folds."""

    unique_groups = sorted(set(str(value) for value in dataset.groups.tolist()))
    if len(unique_groups) < outer_folds:
        raise ValueError(f"Need at least {outer_folds} independent run groups; found {len(unique_groups)}.")
    outer_group_folds = _balanced_group_folds(dataset.groups, outer_folds, seed)
    keys = np.asarray([_key_text(candidate_key(candidate)) for candidate in dataset.candidates], dtype=str)
    splits: list[SplitSpec] = []
    for outer_index, evaluation_groups_list in enumerate(outer_group_folds):
        evaluation_groups = set(evaluation_groups_list)
        evaluation = np.asarray(
            [index for index, group in enumerate(dataset.groups) if str(group) in evaluation_groups],
            dtype=int,
        )
        evaluation_keys = set(keys[evaluation].tolist())
        outer_pool = np.asarray(
            [
                index
                for index, group in enumerate(dataset.groups)
                if str(group) not in evaluation_groups and keys[index] not in evaluation_keys
            ],
            dtype=int,
        )
        pool_groups = dataset.groups[outer_pool]
        if len(set(pool_groups.tolist())) < 3:
            raise ValueError(f"Outer fold {outer_index} has fewer than three groups for inner splitting.")
        inner_group_folds = _balanced_group_folds(pool_groups, 3, seed + 101 + outer_index)
        # Rotate the held-out inner fold across outer folds. All three inner
        # folds are persisted in the plan; seeds rotate it again during trials.
        inner_index = (outer_index + int(inner_fold_offset)) % len(inner_group_folds)
        early_groups_set = set(inner_group_folds[inner_index])
        early = np.asarray(
            [index for index in outer_pool if str(dataset.groups[index]) in early_groups_set],
            dtype=int,
        )
        early_keys = set(keys[early].tolist())
        train = np.asarray(
            [
                index
                for index in outer_pool
                if str(dataset.groups[index]) not in early_groups_set and keys[index] not in early_keys
            ],
            dtype=int,
        )
        if min(len(train), len(early), len(evaluation)) <= 0:
            raise ValueError(f"Candidate purge emptied a partition in outer fold {outer_index}.")
        _assert_disjoint_split(dataset, train, early, evaluation)
        payload = {
            "outer_fold": outer_index,
            "inner_fold": inner_index,
            "train": train.tolist(),
            "early_stop": early.tolist(),
            "evaluation": evaluation.tolist(),
        }
        splits.append(
            SplitSpec(
                outer_fold=outer_index,
                inner_fold=inner_index,
                train_indexes=train,
                early_stop_indexes=early,
                evaluation_indexes=evaluation,
                train_groups=tuple(sorted(set(dataset.groups[train].tolist()))),
                early_stop_groups=tuple(sorted(set(dataset.groups[early].tolist()))),
                evaluation_groups=tuple(sorted(set(dataset.groups[evaluation].tolist()))),
                split_hash=hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
            )
        )
    return splits


def prepare_fold_dataset(dataset: DrlDataset, train_indexes: Sequence[int]) -> tuple[DrlDataset, np.ndarray]:
    """Fill missing baselines using medians computed from the training fold only."""

    train = np.asarray(train_indexes, dtype=int)
    values = dataset.baseline_values.astype(np.float32, copy=True)
    masks = dataset.baseline_mask.astype(np.float32, copy=True)
    medians = np.zeros(values.shape[1], dtype=np.float32)
    for column in range(values.shape[1]):
        observed = train[masks[train, column] > 0]
        medians[column] = float(np.median(values[observed, column])) if observed.size else 0.0
        values[masks[:, column] <= 0, column] = medians[column]
    features = np.concatenate([dataset.actions, values, masks], axis=1).astype(np.float32)
    return replace(dataset, features=features), medians


def surrogate_parameter_count(
    feature_count: int,
    metric_count: int,
    invalid_count: int,
    hidden_sizes: Sequence[int],
) -> int:
    widths = [int(value) for value in hidden_sizes]
    if not widths:
        return (feature_count + 1) * (metric_count + invalid_count)
    count = 0
    previous = feature_count
    for width in widths:
        count += (previous + 1) * width
        previous = width
    count += (previous + 1) * metric_count
    count += (previous + 1) * invalid_count
    return int(count)


class TrialStore:
    """Atomic per-trial persistence with JSON files as the source of truth."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.trial_dir = self.root / "trials"
        self.model_dir = self.root / "models"
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def result_path(self, trial_id: str) -> Path:
        return self.trial_dir / f"{trial_id}.json"

    def completed(self, trial_id: str) -> bool:
        payload = self.load(trial_id)
        return bool(payload and payload.get("status") == "complete")

    def load(self, trial_id: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.result_path(trial_id).read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def all_results(self) -> list[dict[str, Any]]:
        results = []
        for path in sorted(self.trial_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    results.append(payload)
            except Exception:
                continue
        return results

    def write(self, result: Mapping[str, Any]) -> None:
        trial_id = str(result["trial_id"])
        payload = dict(result)
        payload.setdefault("schema_version", 1)
        atomic_write_json(self.result_path(trial_id), payload)
        # The append-only audit log is secondary; each append is one OS-level
        # O_APPEND write, while the atomic JSON above remains authoritative.
        encoded = (_canonical_json(payload) + "\n").encode("utf-8")
        descriptor = os.open(str(self.root / "trials.jsonl"), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        # Rewriting a growing CSV on every trial becomes quadratic and was a
        # noticeable source of desktop stalls in long studies. JSON files and
        # JSONL are durable immediately; refresh the convenience CSV in small
        # batches and once more on every orderly exit.
        trial_count = sum(1 for _ in self.trial_dir.glob("*.json"))
        if trial_count == 1 or trial_count % 10 == 0:
            self.sync_csv()

    def sync_csv(self) -> None:
        rows = [_flatten_trial(item) for item in self.all_results()]
        fields = sorted({key for row in rows for key in row}) or ["trial_id"]
        target = self.root / "trials.csv"
        temporary = target.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)


class OfflineNetworkSweep:
    """Sequential successive-halving runner with bounded CPU and wall time."""

    def __init__(
        self,
        settings: SweepSettings,
        config: TuningConfig | None = None,
        experiment: AutotuneExperimentConfig | None = None,
        surrogate_trainer: Callable[..., Any] | None = None,
        sac_trainer: Callable[..., Mapping[str, Any]] | None = None,
    ):
        self.settings = settings
        self.config = config or TuningConfig()
        self.experiment = experiment or fixed_condition_experiment()
        self.output_dir = Path(settings.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.store = TrialStore(self.output_dir)
        self._surrogate_trainer = surrogate_trainer
        self._sac_trainer = sac_trainer
        self.started = time.monotonic()
        self.deadline = self.started + max(1.0, float(settings.max_wall_time_s))
        self.dataset: DrlDataset | None = None
        self.dataset_manifest: dict[str, Any] = {}
        self.splits: list[SplitSpec] = []

    def run(self) -> dict[str, Any]:
        _limit_cpu_threads(self.settings.cpu_threads)
        dataset, manifest = build_exact_snapshot(
            self.settings.run_roots,
            self.config,
            self.experiment,
            snapshot_dir=None,
        )
        if dataset.size < 20:
            raise RuntimeError(f"Only {dataset.size} exact-condition non-baseline samples were found; need at least 20.")
        self.dataset = dataset
        self.dataset_manifest = manifest
        self._verify_or_record_study_identity()
        _save_snapshot(dataset, manifest, self.output_dir / "snapshot")
        self.splits = make_grouped_splits(dataset, outer_folds=4, seed=self.settings.seed)
        atomic_write_json(self.output_dir / "splits.json", {"splits": [_split_payload(item) for item in self.splits]})
        plan = self.plan()
        atomic_write_json(self.output_dir / "plan.json", plan)
        if self.settings.dry_run:
            state = self._state("dry_run_complete", complete=False)
            atomic_write_json(self.output_dir / "sweep_state.json", state)
            return state

        surrogate_summary: dict[str, Any] | None = None
        sac_summary: dict[str, Any] | None = None
        if self.settings.phase in {"all", "surrogate"}:
            surrogate_summary = self._run_surrogate_sweep()
            if self._expired():
                return self._pause("wall_clock_limit", surrogate_summary, sac_summary)
        else:
            surrogate_summary = self._load_surrogate_summary()
        if self.settings.phase in {"all", "sac"}:
            if not surrogate_summary or not surrogate_summary.get("research_leader"):
                raise RuntimeError("SAC sweep requires a completed surrogate leaderboard in this study directory.")
            sac_summary = self._run_sac_sweep(surrogate_summary)
            if self._expired():
                return self._pause("wall_clock_limit", surrogate_summary, sac_summary)

        complete = bool(
            (self.settings.phase == "surrogate" and surrogate_summary and surrogate_summary.get("complete"))
            or (self.settings.phase in {"all", "sac"} and sac_summary and sac_summary.get("complete"))
        )
        leaderboard = self._write_leaderboard(surrogate_summary, sac_summary, complete=complete)
        self.store.sync_csv()
        _write_pareto_plot(self.store.all_results(), self.output_dir / "parameter_performance_pareto.png")
        state = self._state("complete" if complete else "pending", complete=complete)
        state["leaderboard_path"] = str(self.output_dir / "leaderboard.json")
        state["research_leader"] = _compact_leader(leaderboard.get("research_leader"))
        state["accepted_winner"] = _compact_leader(leaderboard.get("accepted_winner"))
        atomic_write_json(self.output_dir / "sweep_state.json", state)
        return state

    def plan(self) -> dict[str, Any]:
        quick = self.settings.quick
        surrogate_arches = list(SURROGATE_ARCHITECTURES)
        sac_arches = list(SAC_ARCHITECTURES)
        if quick:
            surrogate_arches = [surrogate_arches[0], surrogate_arches[6], surrogate_arches[-1]]
            sac_arches = [sac_arches[0], sac_arches[4], sac_arches[7]]
        return {
            "schema_version": 1,
            "offline_only": True,
            "hardware_access": False,
            "active_artifact_mutation": False,
            "dataset_hash": self.dataset_manifest.get("dataset_hash"),
            "max_wall_time_s": self.settings.max_wall_time_s,
            "cpu_threads": min(4, max(1, self.settings.cpu_threads)),
            "quick": quick,
            "surrogate": {
                "architectures": [{"name": name, "hidden_sizes": list(widths)} for name, widths in surrogate_arches],
                "coarse": {"members": 1 if quick else 3, "folds": 4, "seeds": 1, "epochs": 2 if quick else 120, "keep": min(4, len(surrogate_arches))},
                "refine": {"members": 2 if quick else 5, "folds": 4, "seeds": 1 if quick else 3, "epochs": 3 if quick else 300, "keep": min(2, len(surrogate_arches))},
                "ensemble": {"members": [1, 2] if quick else [3, 5, 7], "folds": 4, "seeds": 1 if quick else 3},
                "inner_early_stopping_folds": 3,
                "inner_fold_rotation": "the three refinement seeds rotate across all inner run folds",
                "learning_curve_fractions": [0.4, 0.6, 0.8, 1.0],
            },
            "sac": {
                "architectures": [{"name": name, "actor": list(widths), "critic": list(widths)} for name, widths in sac_arches],
                "coarse": {"seeds": 1 if quick else 2, "steps": 100 if quick else 25_000, "episodes": 10 if quick else 2_000, "keep": min(4, len(sac_arches))},
                "refine": {"seeds": 1 if quick else 3, "steps": 200 if quick else 100_000, "episodes": 20 if quick else 5_000, "keep": min(2, len(sac_arches))},
                "confirm": {"seeds": 1 if quick else 5, "steps": 300 if quick else 300_000, "episodes": 25 if quick else 10_000, "checkpoint_interval": 100 if quick else 25_000},
            },
            "ranking": {
                "surrogate": ["all_safety_gates", "normalized_top5_regret", "top5_basin_recall", "penalty_spearman", "penalty_p90_absolute_error", "parameter_count"],
                "sac": ["policy_gate", "success_rate", "final_penalty", "steps_to_success_p90", "seed_variance"],
            },
        }

    def _run_surrogate_sweep(self) -> dict[str, Any]:
        plan = self.plan()["surrogate"]
        architectures = [(item["name"], tuple(item["hidden_sizes"])) for item in plan["architectures"]]
        coarse_cfg = plan["coarse"]
        self._run_surrogate_stage(
            "coarse",
            architectures,
            members=coarse_cfg["members"],
            epochs=coarse_cfg["epochs"],
            seeds=[self.settings.seed],
        )
        if self._expired():
            return self._surrogate_summary(complete=False)
        coarse = self._rank_surrogates("coarse")
        finalists = [(item["architecture_name"], tuple(item["hidden_sizes"])) for item in coarse[: coarse_cfg["keep"]]]

        refine_cfg = plan["refine"]
        refine_seeds = [self.settings.seed + 1_000 + index for index in range(refine_cfg["seeds"])]
        self._run_surrogate_stage(
            "refine",
            finalists,
            members=refine_cfg["members"],
            epochs=refine_cfg["epochs"],
            seeds=refine_seeds,
        )
        if self._expired():
            return self._surrogate_summary(complete=False)
        refined = self._rank_surrogates("refine")
        top_two = [(item["architecture_name"], tuple(item["hidden_sizes"])) for item in refined[: refine_cfg["keep"]]]

        ensemble_members = plan["ensemble"]["members"]
        for members in ensemble_members:
            self._run_surrogate_stage(
                f"ensemble_m{members}",
                top_two,
                members=members,
                epochs=refine_cfg["epochs"],
                seeds=refine_seeds,
            )
            if self._expired():
                return self._surrogate_summary(complete=False)

        ensemble_results = []
        for members in ensemble_members:
            ensemble_results.extend(self._rank_surrogates(f"ensemble_m{members}"))
        ensemble_results.sort(key=_surrogate_rank_key)
        if not ensemble_results:
            return self._surrogate_summary(complete=False)
        winner = ensemble_results[0]
        equivalent = _smallest_statistically_equivalent(ensemble_results, winner)
        curve_models = {
            (winner["architecture_name"], tuple(winner["hidden_sizes"]), int(winner["members"])),
            (equivalent["architecture_name"], tuple(equivalent["hidden_sizes"]), int(equivalent["members"])),
        }
        for fraction in plan["learning_curve_fractions"]:
            for name, widths, members in sorted(curve_models):
                self._run_surrogate_stage(
                    "learning_curve",
                    [(name, widths)],
                    members=members,
                    epochs=refine_cfg["epochs"],
                    seeds=[refine_seeds[0]],
                    train_fraction=float(fraction),
                )
                if self._expired():
                    return self._surrogate_summary(complete=False)
        summary = self._surrogate_summary(complete=True)
        atomic_write_json(self.output_dir / "surrogate_leaderboard.json", summary)
        _write_learning_curves(self.store.all_results(), self.output_dir)
        return summary

    def _run_surrogate_stage(
        self,
        stage: str,
        architectures: Sequence[tuple[str, tuple[int, ...]]],
        members: int,
        epochs: int,
        seeds: Sequence[int],
        train_fraction: float = 1.0,
    ) -> None:
        assert self.dataset is not None
        for architecture_name, hidden_sizes in architectures:
            for seed_position, seed in enumerate(seeds):
                stage_splits = make_grouped_splits(
                    self.dataset,
                    outer_folds=4,
                    seed=self.settings.seed,
                    inner_fold_offset=seed_position,
                )
                for split in stage_splits:
                    spec = TrialSpec(
                        kind="surrogate",
                        stage=stage,
                        architecture_name=architecture_name,
                        hidden_sizes=tuple(hidden_sizes),
                        seed=int(seed),
                        members=int(members),
                        epochs_or_steps=int(epochs),
                        outer_fold=split.outer_fold,
                        train_fraction=float(train_fraction),
                    )
                    if self.store.completed(spec.trial_id):
                        continue
                    if self._expired():
                        return
                    effective_split = _fractional_split(self.dataset, split, train_fraction, seed)
                    result = self._execute_surrogate(spec, effective_split)
                    self.store.write(result)

    def _execute_surrogate(self, spec: TrialSpec, split: SplitSpec) -> dict[str, Any]:
        assert self.dataset is not None
        started = time.perf_counter()
        rss_before, peak_before = _process_memory_mb()
        fold_dataset, medians = prepare_fold_dataset(self.dataset, split.train_indexes)
        artifact_dir = self.store.model_dir / spec.trial_id
        _rotate_incomplete_artifact(artifact_dir)
        trainer = self._surrogate_trainer
        if trainer is None:
            from .model import train_surrogate_ensemble

            trainer = train_surrogate_ensemble
        kwargs: dict[str, Any] = {
            "dataset": fold_dataset,
            "config": self.config,
            "artifact_dir": artifact_dir,
            "operating_signature": operating_signature(self.config, self.experiment),
            "members": spec.members,
            "epochs": spec.epochs_or_steps,
            "batch_size": self.settings.batch_size,
            "seed": spec.seed,
            "hidden_sizes": spec.hidden_sizes,
            "train_indexes": split.train_indexes,
            "validation_indexes": split.early_stop_indexes,
            "evaluation_indexes": split.evaluation_indexes,
            "early_stopping_patience": 2 if self.settings.quick else (20 if spec.stage == "coarse" else 35),
            "progress": self._progress(spec),
        }
        ensemble = _call_supported(trainer, kwargs)
        metrics = evaluate_surrogate_ranking(
            ensemble,
            fold_dataset,
            self.config,
            split.evaluation_indexes,
            calibration_indexes=split.train_indexes,
        )
        manifest = dict(getattr(ensemble, "manifest", {}) or {})
        parameter_count = int(
            manifest.get("trainable_parameter_count")
            or surrogate_parameter_count(
                fold_dataset.features.shape[1],
                fold_dataset.metrics.shape[1],
                fold_dataset.invalid_labels.shape[1],
                spec.hidden_sizes,
            ) * spec.members
        )
        result = {
            "trial_id": spec.trial_id,
            "status": "complete",
            "kind": spec.kind,
            "stage": spec.stage,
            "architecture_name": spec.architecture_name,
            "hidden_sizes": list(spec.hidden_sizes),
            "members": spec.members,
            "seed": spec.seed,
            "outer_fold": spec.outer_fold,
            "inner_fold": split.inner_fold,
            "train_fraction": spec.train_fraction,
            "epochs_or_steps": spec.epochs_or_steps,
            "dataset_hash": self.dataset_manifest["dataset_hash"],
            "split_hash": split.split_hash,
            "train_count": int(len(split.train_indexes)),
            "early_stop_count": int(len(split.early_stop_indexes)),
            "evaluation_count": int(len(split.evaluation_indexes)),
            "train_indexes": split.train_indexes.tolist(),
            "early_stop_indexes": split.early_stop_indexes.tolist(),
            "evaluation_indexes": split.evaluation_indexes.tolist(),
            "train_groups": list(split.train_groups),
            "early_stop_groups": list(split.early_stop_groups),
            "evaluation_groups": list(split.evaluation_groups),
            "baseline_train_medians": medians.tolist(),
            "parameter_count": parameter_count,
            "artifact_dir": str(artifact_dir),
            # Sweep acceptance is stricter than a single aggregate model
            # manifest: every held-out run must pass, so a 500-row run cannot
            # hide a failure in a smaller run.
            "accepted": bool(metrics["all_safety_gates"]),
            "metrics": metrics,
            "wall_time_s": time.perf_counter() - started,
            "completed_at": time.time(),
            "inference_latency_p95_ms": manifest.get("inference_latency_p95_ms"),
            "offline_only": True,
            "hardware_ready": False,
        }
        del ensemble
        gc.collect()
        rss_after, peak_after = _process_memory_mb()
        result["rss_before_mb"] = rss_before
        result["rss_after_release_mb"] = rss_after
        result["peak_rss_mb"] = max(peak_before, peak_after)
        return result

    def _run_sac_sweep(self, surrogate_summary: Mapping[str, Any]) -> dict[str, Any]:
        plan = self.plan()["sac"]
        architectures = [(item["name"], tuple(item["actor"])) for item in plan["architectures"]]
        surrogate_artifacts = self._surrogate_artifacts_for_leader(surrogate_summary["research_leader"])
        if not surrogate_artifacts:
            raise RuntimeError("No completed cross-fit surrogate artifacts are available for the SAC sweep.")
        validation_pack = self._validation_pack()
        coarse = plan["coarse"]
        self._run_sac_stage(
            "coarse",
            architectures,
            seeds=[self.settings.seed + 20_000 + index for index in range(coarse["seeds"])],
            steps=coarse["steps"],
            episodes=coarse["episodes"],
            surrogate_artifacts=surrogate_artifacts,
            validation_pack=validation_pack,
        )
        if self._expired():
            return self._sac_summary(complete=False)
        coarse_ranked = self._rank_sac("coarse")
        finalists = [(item["architecture_name"], tuple(item["hidden_sizes"])) for item in coarse_ranked[: coarse["keep"]]]
        refine = plan["refine"]
        self._run_sac_stage(
            "refine",
            finalists,
            seeds=[self.settings.seed + 30_000 + index for index in range(refine["seeds"])],
            steps=refine["steps"],
            episodes=refine["episodes"],
            surrogate_artifacts=surrogate_artifacts,
            validation_pack=validation_pack,
        )
        if self._expired():
            return self._sac_summary(complete=False)
        refined = self._rank_sac("refine")
        top_two = [(item["architecture_name"], tuple(item["hidden_sizes"])) for item in refined[: refine["keep"]]]
        confirm = plan["confirm"]
        self._run_sac_stage(
            "confirm",
            top_two,
            seeds=[self.settings.seed + 40_000 + index for index in range(confirm["seeds"])],
            steps=confirm["steps"],
            episodes=confirm["episodes"],
            surrogate_artifacts=surrogate_artifacts,
            validation_pack=validation_pack,
            checkpoint_interval=confirm["checkpoint_interval"],
        )
        summary = self._sac_summary(complete=not self._expired())
        if not surrogate_summary.get("accepted_winner"):
            summary["accepted_winner"] = None
            summary["blocked_by_surrogate_gates"] = True
        atomic_write_json(self.output_dir / "sac_leaderboard.json", summary)
        return summary

    def _run_sac_stage(
        self,
        stage: str,
        architectures: Sequence[tuple[str, tuple[int, ...]]],
        seeds: Sequence[int],
        steps: int,
        episodes: int,
        surrogate_artifacts: Sequence[Path],
        validation_pack: Mapping[str, Any],
        checkpoint_interval: int = 25_000,
    ) -> None:
        for architecture_name, hidden_sizes in architectures:
            for seed_index, seed in enumerate(seeds):
                spec = TrialSpec(
                    kind="sac",
                    stage=stage,
                    architecture_name=architecture_name,
                    hidden_sizes=tuple(hidden_sizes),
                    seed=int(seed),
                    epochs_or_steps=int(steps),
                    evaluation_episodes=int(episodes),
                )
                if self.store.completed(spec.trial_id):
                    continue
                if self._expired():
                    return
                source = surrogate_artifacts[seed_index % len(surrogate_artifacts)]
                result = self._execute_sac(spec, source, surrogate_artifacts, validation_pack, checkpoint_interval)
                self.store.write(result)

    def _execute_sac(
        self,
        spec: TrialSpec,
        source_surrogate: Path,
        crossfit_surrogates: Sequence[Path],
        validation_pack: Mapping[str, Any],
        checkpoint_interval: int,
    ) -> dict[str, Any]:
        assert self.dataset is not None
        started = time.perf_counter()
        rss_before, peak_before = _process_memory_mb()
        artifact_dir = self.store.model_dir / spec.trial_id
        _rotate_incomplete_artifact(artifact_dir)
        _copy_surrogate_artifact(source_surrogate, artifact_dir)
        from .model import SurrogateEnsemble

        ensemble = SurrogateEnsemble.load(artifact_dir)
        training_dataset = self._dataset_for_surrogate_artifact(source_surrogate, partition="train")
        trainer = self._sac_trainer
        if trainer is None:
            from .policy import train_safe_sac_policy

            trainer = train_safe_sac_policy
        starts = [self.dataset.candidates[int(index)] for index in validation_pack["sample_indexes"]]
        kwargs: dict[str, Any] = {
            "ensemble": ensemble,
            "dataset": training_dataset,
            "config": self.config,
            "total_steps": spec.epochs_or_steps,
            "evaluation_episodes": spec.evaluation_episodes,
            "seed": spec.seed,
            "progress": self._progress(spec),
            "allow_unaccepted_surrogate": True,
            "policy_net_arch": spec.hidden_sizes,
            "actor_net_arch": spec.hidden_sizes,
            "critic_net_arch": spec.hidden_sizes,
            "validation_seeds": validation_pack["seeds"],
            "validation_starts": starts,
            "checkpoint_interval": checkpoint_interval,
            # The full fixed evaluation pack decides the final result. Use a
            # stable subset at intermediate checkpoints so a 300k-step trial
            # does not multiply 10k episodes by every checkpoint.
            "checkpoint_evaluation_episodes": min(spec.evaluation_episodes, 1_000),
        }
        manifest = dict(_call_supported(trainer, kwargs))
        crossfit = _crossfit_policy_evaluation(
            artifact_dir,
            crossfit_surrogates,
            [
                self._dataset_for_surrogate_artifact(path, partition="evaluation")
                for path in crossfit_surrogates
            ],
            self.config,
            spec.evaluation_episodes,
            spec.seed + 100_000,
            validation_seeds=validation_pack["seeds"],
            validation_starts=starts,
        )
        native = dict(manifest.get("policy_evaluation") or {})
        metrics = _merge_policy_metrics(native, crossfit)
        accepted = bool(
            metrics.get("success_rate", 0.0) >= 0.90
            and metrics.get("protection_rate", 1.0) < 0.005
            and metrics.get("crossfit_policy_gate", True)
        )
        parameter_counts = manifest.get("policy_parameter_counts") or {}
        if isinstance(parameter_counts, Mapping):
            parameter_count = int(
                parameter_counts.get("optimized_total")
                or parameter_counts.get("total")
                or sum(int(value) for value in parameter_counts.values())
            )
        else:
            parameter_count = int(parameter_counts or 0)
        result = {
            "trial_id": spec.trial_id,
            "status": "complete",
            "kind": spec.kind,
            "stage": spec.stage,
            "architecture_name": spec.architecture_name,
            "hidden_sizes": list(spec.hidden_sizes),
            "actor_net_arch": list(spec.hidden_sizes),
            "critic_net_arch": list(spec.hidden_sizes),
            "seed": spec.seed,
            "epochs_or_steps": spec.epochs_or_steps,
            "evaluation_episodes": spec.evaluation_episodes,
            "dataset_hash": self.dataset_manifest["dataset_hash"],
            "source_surrogate": str(source_surrogate),
            "crossfit_surrogates": [str(path) for path in crossfit_surrogates],
            "parameter_count": parameter_count,
            "policy_parameter_counts": parameter_counts,
            "artifact_dir": str(artifact_dir),
            "accepted": accepted,
            "metrics": metrics,
            "best_checkpoint": manifest.get("policy_best_checkpoint"),
            "best_checkpoint_step": manifest.get("policy_best_checkpoint_step"),
            "wall_time_s": time.perf_counter() - started,
            "completed_at": time.time(),
            "offline_only": True,
            "hardware_ready": False,
        }
        del ensemble
        gc.collect()
        rss_after, peak_after = _process_memory_mb()
        result["rss_before_mb"] = rss_before
        result["rss_after_release_mb"] = rss_after
        result["peak_rss_mb"] = max(peak_before, peak_after)
        return result

    def _dataset_for_surrogate_artifact(
        self,
        artifact: Path,
        *,
        partition: str = "all",
    ) -> DrlDataset:
        """Recreate the exact fold baseline imputation used by an artifact."""

        assert self.dataset is not None
        resolved = str(Path(artifact).resolve())
        result = next(
            (
                item for item in self.store.all_results()
                if str(Path(str(item.get("artifact_dir", ""))).resolve()) == resolved
            ),
            None,
        )
        medians = np.asarray((result or {}).get("baseline_train_medians", []), dtype=np.float32)
        if medians.shape != (len(METRIC_FIELDS),):
            # This path is only a compatibility fallback for artifacts created
            # before the sweep persisted fold medians. It still avoids target
            # information by using the artifact's training indexes when known.
            train_indexes = np.asarray((result or {}).get("train_indexes", []), dtype=int)
            if train_indexes.size:
                prepared = prepare_fold_dataset(self.dataset, train_indexes)[0]
                return _subset_dataset(prepared, train_indexes) if partition == "train" else prepared
            medians = np.zeros(len(METRIC_FIELDS), dtype=np.float32)
        prepared = _dataset_with_baseline_medians(self.dataset, medians)
        if partition == "all":
            return prepared
        index_field = {
            "train": "train_indexes",
            "early_stop": "early_stop_indexes",
            "evaluation": "evaluation_indexes",
        }.get(partition)
        if index_field is None:
            raise ValueError(f"Unknown surrogate artifact dataset partition: {partition}")
        indexes = np.asarray((result or {}).get(index_field, []), dtype=int)
        if indexes.size == 0:
            group_field = {
                "train": "train_groups",
                "early_stop": "early_stop_groups",
                "evaluation": "evaluation_groups",
            }[partition]
            groups = set(str(value) for value in (result or {}).get(group_field, []))
            indexes = np.asarray(
                [index for index, group in enumerate(prepared.groups) if str(group) in groups],
                dtype=int,
            )
        if indexes.size == 0:
            raise RuntimeError(f"Surrogate artifact {artifact} has no persisted {partition} partition.")
        return _subset_dataset(prepared, indexes)

    def _surrogate_summary(self, complete: bool) -> dict[str, Any]:
        stages = sorted(
            set(
                str(item.get("stage"))
                for item in self.store.all_results()
                if item.get("kind") == "surrogate" and str(item.get("stage", "")).startswith("ensemble_m")
            )
        )
        ranked = []
        for stage in stages:
            ranked.extend(self._rank_surrogates(stage))
        ranked.sort(key=_surrogate_rank_key)
        if not ranked:
            fallback_stage = "refine" if self._rank_surrogates("refine") else "coarse"
            ranked = self._rank_surrogates(fallback_stage)
        leader = ranked[0] if ranked else None
        smallest = _smallest_statistically_equivalent(ranked, leader) if leader else None
        accepted_winner = leader if complete and leader and leader.get("all_safety_gates") else None
        return {
            "complete": complete,
            "research_leader": leader,
            "accepted_winner": accepted_winner,
            "smallest_statistically_equivalent": smallest,
            "ranking": ranked,
            "note": "No accepted winner is emitted unless every required stage completes and all outer folds pass.",
        }

    def _sac_summary(self, complete: bool) -> dict[str, Any]:
        ranked = self._rank_sac("confirm") or self._rank_sac("refine") or self._rank_sac("coarse")
        leader = ranked[0] if ranked else None
        accepted_winner = leader if complete and leader and leader.get("policy_gate") else None
        return {
            "complete": complete,
            "research_leader": leader,
            "accepted_winner": accepted_winner,
            "ranking": ranked,
            "offline_only": True,
            "hardware_ready": False,
        }

    def _rank_surrogates(self, stage: str) -> list[dict[str, Any]]:
        results = [
            item for item in self.store.all_results()
            if item.get("status") == "complete" and item.get("kind") == "surrogate" and item.get("stage") == stage
        ]
        grouped: dict[tuple[str, tuple[int, ...], int, float], list[dict[str, Any]]] = {}
        for item in results:
            key = (
                str(item["architecture_name"]),
                tuple(int(value) for value in item.get("hidden_sizes", [])),
                int(item.get("members", 0)),
                float(item.get("train_fraction", 1.0)),
            )
            grouped.setdefault(key, []).append(item)
        ranked = [_aggregate_surrogate_group(key, values) for key, values in grouped.items()]
        ranked.sort(key=_surrogate_rank_key)
        return ranked

    def _rank_sac(self, stage: str) -> list[dict[str, Any]]:
        results = [
            item for item in self.store.all_results()
            if item.get("status") == "complete" and item.get("kind") == "sac" and item.get("stage") == stage
        ]
        grouped: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = {}
        for item in results:
            key = (str(item["architecture_name"]), tuple(int(value) for value in item.get("hidden_sizes", [])))
            grouped.setdefault(key, []).append(item)
        ranked = [_aggregate_sac_group(key, values) for key, values in grouped.items()]
        ranked.sort(key=_sac_rank_key)
        return ranked

    def _surrogate_artifacts_for_leader(self, leader: Mapping[str, Any]) -> list[Path]:
        result_ids = set(str(value) for value in leader.get("trial_ids", []))
        candidates = sorted(
            (
                item for item in self.store.all_results()
                if item.get("trial_id") in result_ids and Path(str(item.get("artifact_dir", ""))).is_dir()
            ),
            key=lambda item: (int(item.get("outer_fold", 999)), int(item.get("seed", 0))),
        )
        artifacts: list[Path] = []
        seen_folds: set[int] = set()
        for item in candidates:
            fold = int(item.get("outer_fold", -1))
            if fold in seen_folds:
                continue
            seen_folds.add(fold)
            artifacts.append(Path(item["artifact_dir"]))
            if len(artifacts) >= 4:
                break
        return artifacts

    def _validation_pack(self) -> dict[str, Any]:
        assert self.dataset is not None
        path = self.output_dir / "policy_validation_pack.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("dataset_hash") != self.dataset_manifest["dataset_hash"]:
                raise RuntimeError("Existing policy validation pack belongs to a different dataset snapshot.")
            return payload
        indexes = _stratified_validation_indexes(self.dataset, count=min(64, self.dataset.size))
        payload = {
            "schema_version": 1,
            "dataset_hash": self.dataset_manifest["dataset_hash"],
            "sample_indexes": indexes,
            "candidate_keys": [_key_text(candidate_key(self.dataset.candidates[index])) for index in indexes],
            "candidates": [candidate_to_mapping(self.dataset.candidates[index]) for index in indexes],
            "seeds": [
                self.settings.seed + 50_000 + index
                for index in range(max(100, len(indexes)) if self.settings.quick else 10_000)
            ],
        }
        atomic_write_json(path, payload)
        return payload

    def _load_surrogate_summary(self) -> dict[str, Any] | None:
        try:
            payload = json.loads((self.output_dir / "surrogate_leaderboard.json").read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _verify_or_record_study_identity(self) -> None:
        identity_path = self.output_dir / "study.json"
        identity = {
            "schema_version": 1,
            "dataset_hash": self.dataset_manifest["dataset_hash"],
            "implementation_hash": _implementation_hash(),
            "operating_signature": self.dataset_manifest["operating_signature"],
            "offline_only": True,
            "hardware_ready": False,
            "sweep_identity": {
                "seed": self.settings.seed,
                "batch_size": self.settings.batch_size,
                "quick": self.settings.quick,
                "surrogate_architectures": [[name, list(widths)] for name, widths in SURROGATE_ARCHITECTURES],
                "sac_architectures": [[name, list(widths)] for name, widths in SAC_ARCHITECTURES],
            },
            "created_at": time.time(),
        }
        if identity_path.exists():
            existing = json.loads(identity_path.read_text(encoding="utf-8"))
            if existing.get("dataset_hash") != identity["dataset_hash"]:
                raise RuntimeError(
                    "The archived data changed after this sweep started. Use a new output directory to preserve reproducibility."
                )
            if existing.get("implementation_hash") not in (None, identity["implementation_hash"]):
                raise RuntimeError(
                    "The DRL sweep/model implementation changed after this study started. "
                    "Use a new output directory so architecture trials remain comparable."
                )
            if existing.get("sweep_identity") not in (None, identity["sweep_identity"]):
                raise RuntimeError(
                    "This study was created with different seed, batch-size, quick-mode, or architecture settings. "
                    "Resume with the original settings or use a new output directory."
                )
            return
        atomic_write_json(identity_path, identity)

    def _write_leaderboard(
        self,
        surrogate: Mapping[str, Any] | None,
        sac: Mapping[str, Any] | None,
        complete: bool,
    ) -> dict[str, Any]:
        all_results = self.store.all_results()
        payload = {
            "schema_version": 1,
            "complete": complete,
            "dataset_hash": self.dataset_manifest.get("dataset_hash"),
            "surrogate": surrogate,
            "sac": sac,
            "accepted_winner": (
                (sac or {}).get("accepted_winner")
                if complete and (surrogate or {}).get("accepted_winner") and sac is not None
                else ((surrogate or {}).get("accepted_winner") if complete and sac is None else None)
            ),
            "research_leader": (sac or {}).get("research_leader") or (surrogate or {}).get("research_leader"),
            "offline_only": True,
            "hardware_ready": False,
            "resource_summary": _resource_summary(all_results),
        }
        atomic_write_json(self.output_dir / "leaderboard.json", payload)
        return payload

    def _expired(self) -> bool:
        return time.monotonic() >= self.deadline

    def _pause(
        self,
        reason: str,
        surrogate: Mapping[str, Any] | None,
        sac: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        self._write_leaderboard(surrogate, sac, complete=False)
        self.store.sync_csv()
        state = self._state("pending", complete=False)
        state["pending_reason"] = reason
        atomic_write_json(self.output_dir / "sweep_state.json", state)
        return state

    def _state(self, status: str, complete: bool) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": status,
            "complete": complete,
            "dataset_hash": self.dataset_manifest.get("dataset_hash"),
            "elapsed_s": time.monotonic() - self.started,
            "max_wall_time_s": self.settings.max_wall_time_s,
            "completed_trials": sum(item.get("status") == "complete" for item in self.store.all_results()),
            "output_dir": str(self.output_dir),
            "offline_only": True,
            "hardware_ready": False,
        }

    @staticmethod
    def _progress(spec: TrialSpec) -> Callable[[float, str], None]:
        last_bucket = -1
        last_printed_at = 0.0

        def report(value: float, message: str) -> None:
            nonlocal last_bucket, last_printed_at
            bounded = min(1.0, max(0.0, float(value)))
            bucket = int(math.floor(bounded * 20.0 + 1e-9))
            now = time.monotonic()
            if bucket == last_bucket and bounded < 1.0 and now - last_printed_at < 30.0:
                return
            last_bucket = bucket
            last_printed_at = now
            print(f"[{spec.trial_id}] {value * 100:6.2f}% {message}", flush=True)

        return report


def evaluate_surrogate_ranking(
    ensemble: Any,
    dataset: DrlDataset,
    config: TuningConfig,
    indexes: Sequence[int],
    calibration_indexes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compute run-balanced downstream-ranking metrics and existing gates."""

    from .model import evaluate_surrogate

    selected = np.asarray(indexes, dtype=int)
    predictions = ensemble.predict_features(dataset.features[selected])
    predicted_scores = np.asarray(
        [relabeled_score(_predicted_metric_payload(row), config.targets)[0] for row in predictions["metric_mean"]],
        dtype=float,
    )
    true_scores = np.asarray(dataset.scores[selected], dtype=float)
    selected_groups = sorted(set(dataset.groups[selected].tolist()))
    per_run: dict[str, dict[str, Any]] = {}
    for group in selected_groups:
        local = np.flatnonzero(dataset.groups[selected] == group)
        per_run[str(group)] = _ranking_metrics(
                predicted_scores[local],
                true_scores[local],
                dataset.actions[selected][local],
            )
    ranking = {
        key: float(np.mean([item[key] for item in per_run.values()]))
        for key in ("normalized_top5_regret", "top5_basin_recall", "penalty_spearman", "penalty_p90_absolute_error")
    }
    calibration = (
        np.asarray(calibration_indexes, dtype=int)
        if calibration_indexes is not None
        else selected
    )
    run_gates = []
    for group in selected_groups:
        run_indexes = selected[dataset.groups[selected] == group]
        run_gates.append(
            _call_supported(
                evaluate_surrogate,
                {
                    "ensemble": ensemble,
                    "dataset": dataset,
                    "config": config,
                    "indexes": run_indexes,
                    "calibration_indexes": calibration,
                },
            )
        )
    acceptances = [dict(item.get("acceptance") or {}) for item in run_gates]
    for group, gate, acceptance in zip(selected_groups, run_gates, acceptances):
        per_run[str(group)].update(
            {
                "all_safety_gates": bool(gate.get("accepted", False)),
                "interval_coverage_90": float(acceptance.get("interval_coverage_90", 0.0) or 0.0),
                "safety_recall": float(acceptance.get("safety_recall", 0.0) or 0.0),
                "validity_specificity": float(acceptance.get("validity_specificity", 0.0) or 0.0),
                "metric_gate": bool(acceptance.get("metric_gate", False)),
            }
        )
    return {
        **ranking,
        "all_safety_gates": bool(run_gates and all(bool(item.get("accepted", False)) for item in run_gates)),
        "interval_coverage_90": _mean_acceptance(acceptances, "interval_coverage_90"),
        "safety_recall": _mean_acceptance(acceptances, "safety_recall"),
        "validity_specificity": _mean_acceptance(acceptances, "validity_specificity"),
        "metric_gate": bool(acceptances and all(bool(item.get("metric_gate", False)) for item in acceptances)),
        "held_out_run_count": len(run_gates),
        "run_metrics": per_run,
    }


def _ranking_metrics(predicted: np.ndarray, truth: np.ndarray, actions: np.ndarray | None = None) -> dict[str, float]:
    if predicted.size == 0:
        return {
            "normalized_top5_regret": float("inf"),
            "top5_basin_recall": 0.0,
            "penalty_spearman": 0.0,
            "penalty_p90_absolute_error": float("inf"),
        }
    count = min(5, len(truth))
    if actions is None:
        predicted_top = np.argsort(predicted)[:count]
        truth_top = np.argsort(truth)[:count]
        basin_matches = len(set(predicted_top.tolist()) & set(truth_top.tolist()))
    else:
        action_rows = np.asarray(actions, dtype=float)
        predicted_top = _distinct_basin_indexes(predicted, action_rows, count)
        truth_top = _distinct_basin_indexes(truth, action_rows, count)
        basin_matches = sum(
            any(float(np.linalg.norm(action_rows[truth_index] - action_rows[predicted_index])) <= 0.20
                for predicted_index in predicted_top)
            for truth_index in truth_top
        )
    selected_best = float(np.min(truth[predicted_top]))
    true_best = float(np.min(truth))
    scale = max(1.0, float(np.percentile(truth, 90) - np.percentile(truth, 10)))
    return {
        "normalized_top5_regret": max(0.0, (selected_best - true_best) / scale),
        "top5_basin_recall": float(basin_matches / max(1, len(truth_top))),
        "penalty_spearman": _spearman(predicted, truth),
        "penalty_p90_absolute_error": float(np.percentile(np.abs(predicted - truth), 90)),
    }


def _distinct_basin_indexes(scores: np.ndarray, actions: np.ndarray, count: int, distance: float = 0.20) -> np.ndarray:
    selected: list[int] = []
    for raw_index in np.argsort(scores).tolist():
        index = int(raw_index)
        if all(float(np.linalg.norm(actions[index] - actions[previous])) > distance for previous in selected):
            selected.append(index)
        if len(selected) >= count:
            break
    # Small/sparse groups may not contain five separated regions. Do not fill
    # with near-duplicates: the denominator reflects the number of real basins.
    return np.asarray(selected, dtype=int)


def _aggregate_surrogate_group(
    key: tuple[str, tuple[int, ...], int, float],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    name, hidden, members, fraction = key
    metric_names = (
        "normalized_top5_regret",
        "top5_basin_recall",
        "penalty_spearman",
        "penalty_p90_absolute_error",
        "interval_coverage_90",
        "safety_recall",
        "validity_specificity",
    )
    payload: dict[str, Any] = {
        "architecture_name": name,
        "hidden_sizes": list(hidden),
        "members": members,
        "train_fraction": fraction,
        "trial_count": len(results),
        "trial_ids": [str(item["trial_id"]) for item in results],
        "all_safety_gates": bool(all(bool(item.get("metrics", {}).get("all_safety_gates")) for item in results)),
        "parameter_count": int(statistics.median(int(item.get("parameter_count", 0)) for item in results)),
    }
    for metric in metric_names:
        run_values = [
            float(run_metric.get(metric, 0.0))
            for item in results
            for run_metric in (item.get("metrics", {}).get("run_metrics", {}) or {}).values()
        ]
        values = run_values or [float(item.get("metrics", {}).get(metric, 0.0)) for item in results]
        payload[metric] = float(np.mean(values))
        payload[f"{metric}_std"] = float(np.std(values))
        payload[f"{metric}_se"] = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return payload


def _aggregate_sac_group(
    key: tuple[str, tuple[int, ...]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    name, hidden = key
    success = [float(item.get("metrics", {}).get("success_rate", 0.0)) for item in results]
    protection = [float(item.get("metrics", {}).get("protection_rate", 1.0)) for item in results]
    final_penalty = [_optional_finite(item.get("metrics", {}).get("mean_final_penalty")) for item in results]
    final_objective = [_optional_finite(item.get("metrics", {}).get("mean_final_objective")) for item in results]
    successful_bandwidth = [
        _optional_finite(item.get("metrics", {}).get("mean_successful_bandwidth")) for item in results
    ]
    p90_steps = [_optional_finite(item.get("metrics", {}).get("p90_steps_to_success")) for item in results]
    return {
        "architecture_name": name,
        "hidden_sizes": list(hidden),
        "trial_count": len(results),
        "trial_ids": [str(item["trial_id"]) for item in results],
        "policy_gate": bool(
            all(bool(item.get("accepted", False)) for item in results)
            and all(value >= 0.90 for value in success)
            and all(value < 0.005 for value in protection)
        ),
        "success_rate": float(np.mean(success)),
        "success_rate_seed_variance": float(np.var(success)),
        "protection_rate": float(np.mean(protection)),
        "mean_final_penalty": _mean_optional(final_penalty),
        "mean_final_objective": _mean_optional(final_objective),
        "mean_successful_bandwidth": _mean_optional(successful_bandwidth),
        "p90_steps_to_success": _mean_optional(p90_steps),
        "parameter_count": int(statistics.median(int(item.get("parameter_count", 0)) for item in results)),
    }


def _surrogate_rank_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(item.get("all_safety_gates")),
        _finite_or(item.get("normalized_top5_regret"), float("inf")),
        -_finite_or(item.get("top5_basin_recall"), 0.0),
        -_finite_or(item.get("penalty_spearman"), -1.0),
        _finite_or(item.get("penalty_p90_absolute_error"), float("inf")),
        int(item.get("parameter_count", 0)),
    )


def _sac_rank_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(item.get("policy_gate")),
        -_finite_or(item.get("success_rate"), 0.0),
        _finite_or(item.get("mean_final_objective"), float("inf")),
        -_finite_or(item.get("mean_successful_bandwidth"), 0.0),
        _finite_or(item.get("mean_final_penalty"), float("inf")),
        _finite_or(item.get("p90_steps_to_success"), float("inf")),
        _finite_or(item.get("success_rate_seed_variance"), float("inf")),
        int(item.get("parameter_count", 0)),
    )


def _smallest_statistically_equivalent(
    ranked: Sequence[Mapping[str, Any]],
    winner: Mapping[str, Any],
) -> dict[str, Any]:
    threshold = _finite_or(winner.get("normalized_top5_regret"), float("inf")) + _finite_or(
        winner.get("normalized_top5_regret_se"), 0.0
    )
    eligible = [
        item for item in ranked
        if bool(item.get("all_safety_gates")) == bool(winner.get("all_safety_gates"))
        and _finite_or(item.get("normalized_top5_regret"), float("inf")) <= threshold
    ]
    return dict(min(eligible or [winner], key=lambda item: int(item.get("parameter_count", 0))))


def _crossfit_policy_evaluation(
    policy_artifact: Path,
    surrogate_artifacts: Sequence[Path],
    datasets: Sequence[DrlDataset],
    config: TuningConfig,
    episodes: int,
    seed: int,
    validation_seeds: Sequence[int] | None = None,
    validation_starts: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    if not surrogate_artifacts:
        return []
    try:
        from stable_baselines3 import SAC
        from .model import SurrogateEnsemble
        from .policy import evaluate_safe_sac_policy
    except Exception:
        return []
    policy_files = [
        policy_artifact / "safe_sac_policy.zip",
        policy_artifact / "safe_sac_policy",
    ]
    policy_path = next((path for path in policy_files if path.exists()), None)
    if policy_path is None:
        return []
    policy = SAC.load(policy_path, device="cpu")
    per_model_episodes = max(1, int(episodes) // max(1, len(surrogate_artifacts)))
    evaluations = []
    for index, (artifact, dataset) in enumerate(zip(surrogate_artifacts, datasets)):
        try:
            ensemble = SurrogateEnsemble.load(artifact)
            evaluations.append(
                evaluate_safe_sac_policy(
                    policy,
                    ensemble,
                    dataset,
                    config,
                    episodes=per_model_episodes,
                    max_episode_steps=15,
                    seed=seed + index * 10_000,
                    validation_seeds=list(validation_seeds or ()),
                    validation_starts=list(validation_starts or ()),
                )
            )
            del ensemble
        except Exception as exc:
            evaluations.append({"error": str(exc), "success_rate": 0.0, "protection_rate": 1.0})
    del policy
    gc.collect()
    return evaluations


def _merge_policy_metrics(native: Mapping[str, Any], crossfit: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [item for item in crossfit if "success_rate" in item]
    if not valid:
        return dict(native)
    merged = dict(native)
    for name in (
        "success_rate",
        "protection_rate",
        "unsafe_rate",
        "invalid_rate",
        "shield_rejection_rate",
        "mean_best_penalty",
        "mean_final_penalty",
        "p90_steps_to_success",
    ):
        values = [_finite_or(item.get(name), float("nan")) for item in valid]
        values = [value for value in values if math.isfinite(value)]
        if values:
            merged[name] = float(np.mean(values))
    merged["crossfit_model_count"] = len(valid)
    merged["crossfit"] = list(valid)
    merged["crossfit_policy_gate"] = bool(
        valid
        and all(float(item.get("success_rate", 0.0)) >= 0.90 for item in valid)
        and all(float(item.get("protection_rate", 1.0)) < 0.005 for item in valid)
    )
    return merged


def _balanced_group_folds(groups: np.ndarray, folds: int, seed: int) -> list[list[str]]:
    counts: dict[str, int] = {}
    for value in groups.tolist():
        counts[str(value)] = counts.get(str(value), 0) + 1
    rng = np.random.default_rng(seed)
    tie_break = {group: float(rng.random()) for group in counts}
    ordered = sorted(counts, key=lambda group: (-counts[group], tie_break[group], group))
    result: list[list[str]] = [[] for _ in range(folds)]
    totals = [0] * folds
    for group in ordered:
        target = min(range(folds), key=lambda index: (totals[index], len(result[index]), index))
        result[target].append(group)
        totals[target] += counts[group]
    return result


def _assert_disjoint_split(dataset: DrlDataset, train: np.ndarray, early: np.ndarray, evaluation: np.ndarray) -> None:
    partitions = [set(train.tolist()), set(early.tolist()), set(evaluation.tolist())]
    if any(partitions[left] & partitions[right] for left in range(3) for right in range(left + 1, 3)):
        raise AssertionError("Split indexes overlap.")
    group_sets = [set(dataset.groups[indexes].tolist()) for indexes in (train, early, evaluation)]
    if any(group_sets[left] & group_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise AssertionError("Run groups overlap across train/early-stop/evaluation.")
    key_sets = [set(_key_text(candidate_key(dataset.candidates[index])) for index in indexes) for indexes in (train, early, evaluation)]
    if any(key_sets[left] & key_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise AssertionError("Candidate keys overlap across train/early-stop/evaluation.")


def _fractional_split(dataset: DrlDataset, split: SplitSpec, fraction: float, seed: int) -> SplitSpec:
    if fraction >= 0.999:
        return split
    groups = sorted(set(dataset.groups[split.train_indexes].tolist()))
    ordered = sorted(groups, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest())
    keep_count = max(1, int(math.ceil(len(ordered) * max(0.01, fraction))))
    keep = set(ordered[:keep_count])
    train = np.asarray([index for index in split.train_indexes if dataset.groups[index] in keep], dtype=int)
    if len(train) < 20:
        # Add complete groups until the trainer's minimum sample count is met.
        for group in ordered[keep_count:]:
            keep.add(group)
            train = np.asarray([index for index in split.train_indexes if dataset.groups[index] in keep], dtype=int)
            if len(train) >= 20:
                break
    payload = {
        "parent": split.split_hash,
        "fraction": fraction,
        "seed": seed,
        "train": train.tolist(),
    }
    return replace(
        split,
        train_indexes=train,
        train_groups=tuple(sorted(keep)),
        split_hash=hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest(),
    )


def _dataset_with_baseline_medians(dataset: DrlDataset, medians: np.ndarray) -> DrlDataset:
    values = dataset.baseline_values.astype(np.float32, copy=True)
    masks = dataset.baseline_mask.astype(np.float32, copy=True)
    for column in range(values.shape[1]):
        values[masks[:, column] <= 0, column] = float(medians[column])
    features = np.concatenate([dataset.actions, values, masks], axis=1).astype(np.float32)
    return replace(dataset, features=features)


def _subset_dataset(dataset: DrlDataset, indexes: Sequence[int]) -> DrlDataset:
    selected = np.asarray(indexes, dtype=int).reshape(-1)
    return DrlDataset(
        features=dataset.features[selected].astype(np.float32, copy=True),
        actions=dataset.actions[selected].astype(np.float32, copy=True),
        metrics=dataset.metrics[selected].astype(np.float32, copy=True),
        metric_mask=dataset.metric_mask[selected].astype(np.float32, copy=True),
        invalid_labels=dataset.invalid_labels[selected].astype(np.float32, copy=True),
        scores=dataset.scores[selected].astype(np.float32, copy=True),
        passed=dataset.passed[selected].astype(np.float32, copy=True),
        groups=dataset.groups[selected].astype(str),
        candidates=[dataset.candidates[int(index)] for index in selected],
        records=[dataset.records[int(index)] for index in selected],
        baseline_values=dataset.baseline_values[selected].astype(np.float32, copy=True),
        baseline_mask=dataset.baseline_mask[selected].astype(np.float32, copy=True),
    )


def _save_snapshot(dataset: DrlDataset, manifest: Mapping[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "dataset.npz"
    temporary = directory / "dataset.npz.tmp.npz"
    np.savez_compressed(
        temporary,
        features=dataset.features,
        actions=dataset.actions,
        metrics=dataset.metrics,
        metric_mask=dataset.metric_mask,
        invalid_labels=dataset.invalid_labels,
        scores=dataset.scores,
        passed=dataset.passed,
        groups=dataset.groups.astype(str),
        baseline_values=dataset.baseline_values,
        baseline_mask=dataset.baseline_mask,
    )
    temporary.replace(target)
    index_payload = {
        "dataset_hash": manifest["dataset_hash"],
        "samples": [
            {
                "index": index,
                "group": str(dataset.groups[index]),
                "candidate_key": _key_text(candidate_key(dataset.candidates[index])),
                "phase": str(dataset.records[index].get("phase") or ""),
            }
            for index in range(dataset.size)
        ],
    }
    atomic_write_json(directory / "sample_index.json", index_payload)
    snapshot_manifest = dict(manifest)
    snapshot_manifest["files_sha256"] = {
        "dataset.npz": _file_hash(target),
        "sample_index.json": _file_hash(directory / "sample_index.json"),
    }
    atomic_write_json(directory / "manifest.json", snapshot_manifest)


def _included_run_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for item in manifest.get("source_runs", []) or []:
        if not isinstance(item, Mapping) or not item.get("included"):
            continue
        path = Path(str(item.get("path") or ""))
        digest = hashlib.sha256()
        found = False
        # iterations.jsonl is the immutable sample stream. run_status.json is
        # only a fallback and may change after a run is archived, which would
        # otherwise give identical data different group identities.
        filenames = ("iterations.jsonl",) if (path / "iterations.jsonl").is_file() else ("run_status.json",)
        for filename in filenames:
            file_path = path / filename
            if not file_path.is_file():
                continue
            found = True
            digest.update(filename.encode("utf-8"))
            with file_path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        run_id = str(item.get("run_id") or path.name)
        key = f"{path.parent.name}:{run_id}"
        hashes[key] = digest.hexdigest() if found else "missing"
    return hashes


def _canonical_run_groups(
    groups: np.ndarray,
    run_hashes: Mapping[str, str],
) -> tuple[dict[str, str], set[str]]:
    """Map archive names to stable content identities and remove duplicates."""

    owners: dict[str, str] = {}
    canonical: dict[str, str] = {}
    duplicates: set[str] = set()
    sources = set(str(value) for value in groups.tolist())
    for source in sorted(sources, key=lambda value: (not value.startswith("saved:"), value)):
        digest = str(run_hashes.get(source) or "")
        if not digest or digest == "missing":
            canonical[source] = source
            continue
        owner = owners.get(digest)
        if owner is not None:
            duplicates.add(source)
            canonical[source] = canonical[owner]
            continue
        owners[digest] = source
        canonical[source] = f"run:{digest[:24]}"
    return canonical, duplicates


def _copy_surrogate_artifact(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    filenames = ["manifest.json", "scalers.npz", *[str(value) for value in manifest.get("member_files", [])]]
    for filename in filenames:
        source_path = source / filename
        if source_path.is_file():
            shutil.copy2(source_path, target / filename)


def _rotate_incomplete_artifact(path: Path) -> None:
    if not path.exists():
        return
    suffix = time.strftime("%Y%m%d_%H%M%S")
    target = path.with_name(f"{path.name}.incomplete_{suffix}_{time.time_ns() % 1_000_000}")
    path.replace(target)


def _call_supported(function: Callable[..., Any], kwargs: Mapping[str, Any]) -> Any:
    """Call a concurrently evolving training API without hiding real errors."""

    parameters = inspect.signature(function).parameters
    if any(value.kind == inspect.Parameter.VAR_KEYWORD for value in parameters.values()):
        return function(**dict(kwargs))
    return function(**{name: value for name, value in kwargs.items() if name in parameters})


def _limit_cpu_threads(requested: int) -> None:
    threads = str(min(4, max(1, int(requested))))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = threads
    try:
        import torch

        torch.set_num_threads(int(threads))
        torch.set_num_interop_threads(1)
    except (ImportError, RuntimeError):
        pass


def _process_memory_mb() -> tuple[float, float]:
    """Return current and peak RSS without adding a psutil dependency."""

    scale = 1024.0 * 1024.0
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            process = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
                return counters.WorkingSetSize / scale, counters.PeakWorkingSetSize / scale
        except Exception:
            pass
    else:
        try:
            import resource

            peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            # Linux reports KiB; macOS reports bytes.
            peak_mb = peak / (scale if sys.platform == "darwin" else 1024.0)
            return peak_mb, peak_mb
        except Exception:
            pass
    return 0.0, 0.0


def _write_learning_curves(results: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    rows = [
        item for item in results
        if item.get("kind") == "surrogate" and item.get("stage") == "learning_curve" and item.get("status") == "complete"
    ]
    grouped: dict[tuple[str, int, float], list[Mapping[str, Any]]] = {}
    for item in rows:
        grouped.setdefault(
            (str(item["architecture_name"]), int(item.get("members", 0)), float(item.get("train_fraction", 1.0))),
            [],
        ).append(item)
    payload = {
        "curves": [
            _aggregate_surrogate_group(
                (name, tuple(values[0].get("hidden_sizes", [])), members, fraction),
                values,
            )
            for (name, members, fraction), values in sorted(grouped.items())
        ]
    }
    atomic_write_json(output_dir / "learning_curves.json", payload)


def _write_pareto_plot(results: Sequence[Mapping[str, Any]], target: Path) -> None:
    points = [item for item in results if item.get("status") == "complete" and int(item.get("parameter_count", 0)) > 0]
    if not points:
        return
    atomic_write_json(
        target.with_suffix(".json"),
        {
            "points": [
                {
                    "trial_id": item.get("trial_id"),
                    "kind": item.get("kind"),
                    "stage": item.get("stage"),
                    "architecture_name": item.get("architecture_name"),
                    "parameter_count": item.get("parameter_count"),
                    "normalized_top5_regret": (item.get("metrics") or {}).get("normalized_top5_regret"),
                    "success_rate": (item.get("metrics") or {}).get("success_rate"),
                }
                for item in points
            ]
        },
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    surrogate = [item for item in points if item.get("kind") == "surrogate"]
    sac = [item for item in points if item.get("kind") == "sac"]
    if surrogate:
        axes[0].scatter(
            [item["parameter_count"] for item in surrogate],
            [item.get("metrics", {}).get("normalized_top5_regret", np.nan) for item in surrogate],
            alpha=0.65,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("surrogate parameters")
    axes[0].set_ylabel("normalized Top-5 regret (lower is better)")
    axes[0].grid(alpha=0.2)
    if sac:
        axes[1].scatter(
            [item["parameter_count"] for item in sac],
            [item.get("metrics", {}).get("success_rate", np.nan) for item in sac],
            alpha=0.65,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("SAC parameters")
    axes[1].set_ylabel("offline success rate (higher is better)")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    temporary = target.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=150)
    plt.close(figure)
    temporary.replace(target)


def _flatten_trial(item: Mapping[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in item.items() if not isinstance(value, (dict, list, tuple))}
    row["hidden_sizes"] = json.dumps(item.get("hidden_sizes", []), separators=(",", ":"))
    for name, value in (item.get("metrics") or {}).items():
        if not isinstance(value, (dict, list, tuple)):
            row[f"metric_{name}"] = value
    return row


def _compact_leader(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {
        key: value.get(key)
        for key in (
            "architecture_name",
            "hidden_sizes",
            "members",
            "parameter_count",
            "all_safety_gates",
            "policy_gate",
            "normalized_top5_regret",
            "success_rate",
        )
        if key in value
    }


def _resource_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    measured = [
        item for item in results
        if _optional_finite(item.get("rss_before_mb")) is not None
        and _optional_finite(item.get("rss_after_release_mb")) is not None
    ]
    if not measured:
        return {"measured_trials": 0}
    ordered = sorted(measured, key=lambda item: _finite_or(item.get("completed_at"), 0.0))
    first = _finite_or(ordered[0].get("rss_before_mb"), 0.0)
    last = _finite_or(ordered[-1].get("rss_after_release_mb"), first)
    peaks = [_finite_or(item.get("peak_rss_mb"), 0.0) for item in measured]
    return {
        "measured_trials": len(measured),
        "rss_first_mb": first,
        "rss_last_after_release_mb": last,
        "rss_net_growth_mb": last - first,
        "peak_rss_mb": max(peaks),
        "note": "Framework allocators may retain arenas; net growth should stabilize instead of increasing per trial.",
    }


def _split_payload(split: SplitSpec) -> dict[str, Any]:
    return {
        "outer_fold": split.outer_fold,
        "inner_fold": split.inner_fold,
        "train_indexes": split.train_indexes.tolist(),
        "early_stop_indexes": split.early_stop_indexes.tolist(),
        "evaluation_indexes": split.evaluation_indexes.tolist(),
        "train_groups": list(split.train_groups),
        "early_stop_groups": list(split.early_stop_groups),
        "evaluation_groups": list(split.evaluation_groups),
        "split_hash": split.split_hash,
    }


def _stratified_validation_indexes(dataset: DrlDataset, count: int) -> list[int]:
    if count <= 0:
        return []
    order = np.argsort(dataset.scores)
    positions = np.linspace(0, max(0, len(order) - 1), count)
    selected: list[int] = []
    seen: set[str] = set()
    for position in positions:
        index = int(order[int(round(float(position)))])
        key = _key_text(candidate_key(dataset.candidates[index]))
        if key not in seen:
            seen.add(key)
            selected.append(index)
    if len(selected) < count:
        for index in order.tolist():
            key = _key_text(candidate_key(dataset.candidates[index]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(int(index))
            if len(selected) >= count:
                break
    return selected


def _predicted_metric_payload(values: Sequence[float]) -> dict[str, Any]:
    row = np.asarray(values, dtype=float)
    return {
        "overshoot_pct": float(row[0]),
        "undershoot_pct": float(row[1]),
        "overshoot_settling_time_s": float(row[2]) * 1e-6,
        "undershoot_settling_time_s": float(row[3]) * 1e-6,
        "phase_margin_deg": float(row[4]),
        "crossover_frequency_hz": float(row[5]) * 1e3,
        "gain_margin_db": float(row[6]),
        "bode_gain_shape_penalty": max(0.0, float(row[7])),
    }


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if np.std(left_rank) < 1e-12 or np.std(right_rank) < 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _mean_acceptance(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    values = [_finite_or(row.get(name), float("nan")) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else 0.0


def _empty_dataset(feature_count: int) -> DrlDataset:
    metric_count = len(METRIC_FIELDS)
    return DrlDataset(
        features=np.zeros((0, feature_count), dtype=np.float32),
        actions=np.zeros((0, len(ACTION_FIELDS)), dtype=np.float32),
        metrics=np.zeros((0, metric_count), dtype=np.float32),
        metric_mask=np.zeros((0, metric_count), dtype=np.float32),
        invalid_labels=np.zeros((0, 3), dtype=np.float32),
        scores=np.zeros(0, dtype=np.float32),
        passed=np.zeros(0, dtype=np.float32),
        groups=np.asarray([], dtype=str),
        candidates=[],
        records=[],
        baseline_values=np.zeros((0, metric_count), dtype=np.float32),
        baseline_mask=np.zeros((0, metric_count), dtype=np.float32),
    )


def _key_text(key: Any) -> str:
    return _canonical_json(list(key) if isinstance(key, tuple) else key)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    directory = Path(__file__).resolve().parent
    for filename in ("common.py", "dataset.py", "model.py", "policy.py", "sweep.py"):
        path = directory / filename
        digest.update(filename.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _finite_or(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else fallback
    except (TypeError, ValueError):
        return fallback


def _optional_finite(value: Any) -> float | None:
    parsed = _finite_or(value, float("nan"))
    return parsed if math.isfinite(parsed) else None


def _mean_optional(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


__all__ = [
    "OfflineNetworkSweep",
    "SAC_ARCHITECTURES",
    "SURROGATE_ARCHITECTURES",
    "SplitSpec",
    "SweepSettings",
    "TrialSpec",
    "TrialStore",
    "build_exact_snapshot",
    "evaluate_surrogate_ranking",
    "fixed_condition_experiment",
    "hash_dataset",
    "make_grouped_splits",
    "prepare_fold_dataset",
    "surrogate_parameter_count",
]
