from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from hardware.tuning.drl.common import ACTION_FIELDS, METRIC_FIELDS, candidate_key, candidate_to_normalized
from hardware.tuning.drl.dataset import DrlDataset, _baseline_for_rows
from hardware.tuning.drl.sweep import (
    OfflineNetworkSweep,
    SAC_ARCHITECTURES,
    SURROGATE_ARCHITECTURES,
    SweepSettings,
    TrialStore,
    _canonical_run_groups,
    _merge_policy_metrics,
    _subset_dataset,
    build_exact_snapshot,
    fixed_condition_experiment,
    make_grouped_splits,
    prepare_fold_dataset,
    surrogate_parameter_count,
)
from hardware.tuning.models import HardwarePidCandidate, TuningConfig


def synthetic_grouped_dataset(groups: int = 8, samples_per_group: int = 6) -> DrlDataset:
    config = TuningConfig()
    candidates = []
    group_values = []
    records = []
    for group_index in range(groups):
        for local_index in range(samples_per_group):
            index = group_index * samples_per_group + local_index
            candidate = HardwarePidCandidate(
                mod0_kp=100 + index,
                mod0_ki=150 + (index * 3) % 105,
                mod0_kd=100 + (index * 7) % 100,
                mod0_kpole1=3 + index % 4,
                mod0_kpole2=3 + (index // 4) % 4,
                output_inductance_nh=80.019 + index * 0.2,
                effective_lc_inductance_nh=295.421 + index * 0.5,
                phase="historical",
            )
            candidates.append(candidate)
            group_values.append(f"run-{group_index}")
            records.append({"phase": "historical", "metrics": {}})
    actions = np.asarray([candidate_to_normalized(item, config.search) for item in candidates], dtype=np.float32)
    size = len(candidates)
    baseline = np.tile(np.asarray([1, 1, 1.5, 1.5, 60, 150, 10, 0], dtype=np.float32), (size, 1))
    baseline_mask = np.ones_like(baseline)
    metrics = baseline.copy()
    metrics[:, 0] += np.linspace(0, 2, size, dtype=np.float32)
    features = np.concatenate([actions, baseline, baseline_mask], axis=1)
    return DrlDataset(
        features=features,
        actions=actions,
        metrics=metrics,
        metric_mask=np.ones_like(metrics),
        invalid_labels=np.zeros((size, 3), dtype=np.float32),
        scores=np.linspace(-3, 100, size, dtype=np.float32),
        passed=np.zeros(size, dtype=np.float32),
        groups=np.asarray(group_values, dtype=str),
        candidates=candidates,
        records=records,
        baseline_values=baseline,
        baseline_mask=baseline_mask,
    )


class DrlNetworkSweepTest(unittest.TestCase):
    def test_dataset_never_uses_first_target_as_an_implicit_baseline(self) -> None:
        values, mask = _baseline_for_rows(
            [{"phase": "drl_policy", "metrics": {"overshoot_pct": 123.0}}]
        )
        np.testing.assert_array_equal(values, np.zeros_like(values))
        np.testing.assert_array_equal(mask, np.zeros_like(mask))

    def test_content_hash_groups_deduplicate_renamed_archives(self) -> None:
        mapping, duplicates = _canonical_run_groups(
            np.asarray(["saved:a", "recent:renamed", "saved:b"]),
            {"saved:a": "a" * 64, "recent:renamed": "a" * 64, "saved:b": "b" * 64},
        )
        self.assertEqual(mapping["saved:a"], mapping["recent:renamed"])
        self.assertEqual(duplicates, {"recent:renamed"})
        self.assertNotEqual(mapping["saved:a"], mapping["saved:b"])

    def test_architecture_grids_and_parameter_count(self) -> None:
        self.assertEqual(len(SURROGATE_ARCHITECTURES), 12)
        self.assertEqual(len(SAC_ARCHITECTURES), 10)
        self.assertEqual(SURROGATE_ARCHITECTURES[0], ("linear", ()))
        self.assertEqual(SURROGATE_ARCHITECTURES[-1][1], (128, 128, 64))
        # 20->128->128->64 plus 64->7 and 64->3 heads.
        self.assertEqual(surrogate_parameter_count(20, 7, 3, (128, 128, 64)), 28_106)
        self.assertEqual(surrogate_parameter_count(20, 7, 3, ()), 210)

    def test_group_folds_and_candidate_keys_are_purged(self) -> None:
        dataset = synthetic_grouped_dataset()
        # Deliberately repeat one candidate in a different run. It must not leak
        # into train/early-stop when its twin is in outer evaluation.
        dataset.candidates[-1] = dataset.candidates[0]
        dataset.actions[-1] = dataset.actions[0]
        splits = make_grouped_splits(dataset, outer_folds=4, seed=19)
        self.assertEqual(len(splits), 4)
        for split in splits:
            partitions = [split.train_indexes, split.early_stop_indexes, split.evaluation_indexes]
            group_sets = [set(dataset.groups[indexes]) for indexes in partitions]
            key_sets = [set(candidate_key(dataset.candidates[index]) for index in indexes) for indexes in partitions]
            for left in range(3):
                for right in range(left + 1, 3):
                    self.assertFalse(group_sets[left] & group_sets[right])
                    self.assertFalse(key_sets[left] & key_sets[right])

        rotated = make_grouped_splits(dataset, outer_folds=4, seed=19, inner_fold_offset=1)
        for original, alternate in zip(splits, rotated):
            self.assertEqual(original.evaluation_groups, alternate.evaluation_groups)
            self.assertNotEqual(original.inner_fold, alternate.inner_fold)
            self.assertNotEqual(original.early_stop_groups, alternate.early_stop_groups)

    def test_baseline_imputation_uses_training_fold_only(self) -> None:
        dataset = synthetic_grouped_dataset(groups=4, samples_per_group=6)
        dataset.baseline_values[:] = 0
        dataset.baseline_mask[:] = 0
        dataset.baseline_values[0, 0] = 2.0
        dataset.baseline_mask[0, 0] = 1.0
        dataset.baseline_values[-1, 0] = 999.0
        dataset.baseline_mask[-1, 0] = 1.0
        prepared, medians = prepare_fold_dataset(dataset, np.arange(0, 12))
        self.assertEqual(float(medians[0]), 2.0)
        # A missing held-out baseline receives 2, not the held-out value 999.
        self.assertEqual(float(prepared.features[-2, len(ACTION_FIELDS)]), 2.0)
        self.assertEqual(float(prepared.features[-2, len(ACTION_FIELDS) + len(METRIC_FIELDS)]), 0.0)

        subset = _subset_dataset(prepared, np.arange(0, 12))
        self.assertEqual(subset.size, 12)
        self.assertEqual(set(subset.groups), {"run-0", "run-1"})

    def test_snapshot_requires_explicit_baseline_and_excludes_target_row(self) -> None:
        loaded = synthetic_grouped_dataset(groups=2, samples_per_group=4)
        loaded.records[0] = {
            "phase": "baseline",
            "metrics": {
                "overshoot_pct": 1.0,
                "undershoot_pct": 1.0,
                "overshoot_settling_time_s": 1e-6,
                "undershoot_settling_time_s": 1e-6,
                "phase_margin_deg": 60.0,
                "crossover_frequency_hz": 150_000.0,
                "gain_margin_db": 10.0,
            },
        }
        source = {
            "source_runs": [
                {"run_id": "run-0", "path": "does-not-exist", "included": True},
                {"run_id": "run-1", "path": "does-not-exist", "included": True},
            ],
            "source_record_count": loaded.size,
        }
        with mock.patch("hardware.tuning.drl.sweep.load_autotune_dataset", return_value=(loaded, source)):
            snapshot, manifest = build_exact_snapshot([], TuningConfig(), fixed_condition_experiment())
        self.assertEqual(snapshot.size, loaded.size - 1)
        self.assertNotIn("baseline", [item.get("phase") for item in snapshot.records])
        first_group = snapshot.groups == "run-0"
        second_group = snapshot.groups == "run-1"
        self.assertTrue(np.all(snapshot.baseline_mask[first_group, :4] == 1))
        self.assertTrue(np.all(snapshot.baseline_mask[second_group] == 0))
        self.assertEqual(manifest["excluded_explicit_baseline_target_count"], 1)
        self.assertTrue(manifest["exact_condition_only"])

    def test_trial_store_resume_uses_atomic_result_as_source_of_truth(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = TrialStore(Path(temporary))
            store.write({"trial_id": "trial-1", "status": "complete", "metrics": {"score": 1.0}})
            self.assertTrue(store.completed("trial-1"))
            self.assertEqual(len(store.all_results()), 1)
            self.assertTrue((Path(temporary) / "trials.csv").is_file())
            self.assertEqual(json.loads(store.result_path("trial-1").read_text())["status"], "complete")

    def test_dry_run_writes_reproducible_plan_without_trainers(self) -> None:
        dataset = synthetic_grouped_dataset()
        manifest = {
            "dataset_hash": "frozen-test-hash",
            "operating_signature": {"signature": "fixed"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            settings = SweepSettings(
                output_dir=Path(temporary),
                run_roots=(),
                dry_run=True,
                quick=True,
            )
            with mock.patch(
                "hardware.tuning.drl.sweep.build_exact_snapshot",
                return_value=(dataset, manifest),
            ):
                result = OfflineNetworkSweep(settings).run()
            self.assertEqual(result["status"], "dry_run_complete")
            plan = json.loads((Path(temporary) / "plan.json").read_text())
            self.assertTrue(plan["offline_only"])
            self.assertFalse(plan["hardware_access"])
            self.assertEqual(len(plan["surrogate"]["architectures"]), 3)
            self.assertTrue((Path(temporary) / "splits.json").is_file())

    def test_policy_cannot_be_accepted_when_surrogate_gates_failed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            runner = OfflineNetworkSweep(SweepSettings(output_dir=Path(temporary), run_roots=()))
            runner.dataset_manifest = {"dataset_hash": "frozen"}
            leaderboard = runner._write_leaderboard(
                {"accepted_winner": None, "research_leader": {"architecture_name": "surrogate"}},
                {"accepted_winner": {"architecture_name": "policy"}, "research_leader": {"architecture_name": "policy"}},
                complete=True,
            )
            self.assertIsNone(leaderboard["accepted_winner"])
            self.assertEqual(leaderboard["research_leader"]["architecture_name"], "policy")

    def test_crossfit_policy_gate_requires_every_surrogate_model_to_pass(self) -> None:
        metrics = _merge_policy_metrics(
            {"success_rate": 1.0, "protection_rate": 0.0},
            [
                {"success_rate": 1.0, "protection_rate": 0.0},
                {"success_rate": 0.89, "protection_rate": 0.0},
            ],
        )
        self.assertFalse(metrics["crossfit_policy_gate"])


if __name__ == "__main__":
    unittest.main()
