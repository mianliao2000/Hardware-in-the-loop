from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hardware.tuning.drl.common import ACTION_FIELDS, METRIC_FIELDS, candidate_to_normalized
from hardware.tuning.drl.dataset import DrlDataset
from hardware.tuning.drl.model import (
    DEFAULT_SURROGATE_HIDDEN_SIZES,
    LEGACY_SURROGATE_HIDDEN_SIZES,
    SurrogateEnsemble,
    _group_balanced_bootstrap,
    _resolve_training_splits,
    _surrogate_network_type,
    _trainable_parameter_count,
    dependency_status,
    train_surrogate_ensemble,
)
from hardware.tuning.models import HardwarePidCandidate, TuningConfig
from scripts.pretrain_drl import _explicit_validation_split


def _dataset(size: int = 36) -> DrlDataset:
    config = TuningConfig()
    candidates = [
        HardwarePidCandidate(
            mod0_kp=110 + index,
            mod0_ki=160 + index,
            mod0_kd=105 + index,
            mod0_kpole1=3 + index % 4,
            mod0_kpole2=3 + index % 4,
            mod0_cm_gain=2,
            output_inductance_nh=82.0 + index,
            effective_lc_inductance_nh=290.0 + 2.0 * index,
            phase="historical",
        )
        for index in range(size)
    ]
    actions = np.asarray(
        [candidate_to_normalized(candidate, config.search) for candidate in candidates],
        dtype=np.float32,
    )
    baseline_row = np.asarray([0.8, 0.7, 2.0, 2.2, 75.0, 140.0, 12.0, 0.0], dtype=np.float32)
    baseline = np.tile(baseline_row, (size, 1))
    metrics = baseline.copy()
    ramp = np.arange(size, dtype=np.float32)
    metrics[:, 0] += ramp * 0.01
    metrics[:, 1] += ramp * 0.005
    metrics[:, 2] += ramp * 0.02
    metrics[:, 4] += ramp * 0.1
    features = np.concatenate((actions, baseline, np.ones_like(baseline)), axis=1)
    return DrlDataset(
        features=features,
        actions=actions,
        metrics=metrics,
        metric_mask=np.ones_like(metrics),
        invalid_labels=np.zeros((size, 3), dtype=np.float32),
        scores=np.linspace(0.0, 20.0, size, dtype=np.float32),
        passed=np.zeros(size, dtype=np.float32),
        groups=np.asarray([f"run-{index // 4}" for index in range(size)]),
        candidates=candidates,
        records=[{"phase": "historical"} for _ in range(size)],
        baseline_values=baseline,
        baseline_mask=np.ones_like(baseline),
    )


class SurrogateArchitectureTest(unittest.TestCase):
    def setUp(self) -> None:
        if not dependency_status()["ok"]:
            self.skipTest("Optional ML dependencies are not installed.")

    def test_network_supports_linear_and_custom_hidden_sizes(self) -> None:
        import torch

        network_type = _surrogate_network_type(torch)
        linear = network_type(20, 7, 3, hidden_sizes=())
        custom = network_type(20, 7, 3, hidden_sizes=(8,))
        default = network_type(20, 7, 3)

        metric, invalid = linear(torch.zeros((2, 20), dtype=torch.float32))
        self.assertEqual(tuple(metric.shape), (2, 7))
        self.assertEqual(tuple(invalid.shape), (2, 3))
        self.assertEqual(_trainable_parameter_count(linear), 210)
        self.assertEqual(_trainable_parameter_count(custom), 258)
        self.assertEqual(DEFAULT_SURROGATE_HIDDEN_SIZES, (96, 64, 32))
        self.assertEqual(default.metric_head.in_features, 32)
        with self.assertRaisesRegex(ValueError, "positive integers"):
            network_type(20, 7, 3, hidden_sizes=(8, 0))

    def test_training_bootstrap_gives_each_run_equal_weight(self) -> None:
        indexes = np.arange(100, dtype=int)
        groups = np.asarray(["long"] * 90 + ["short"] * 10)
        sampled = _group_balanced_bootstrap(indexes, groups, np.random.default_rng(7))

        self.assertEqual(len(sampled), len(indexes))
        self.assertEqual(int(np.sum(groups[sampled] == "long")), 50)
        self.assertEqual(int(np.sum(groups[sampled] == "short")), 50)

    def test_pretraining_explicit_validation_split_keeps_complete_runs_isolated(self) -> None:
        dataset = _dataset()
        training, validation = _explicit_validation_split(dataset, {"run-7", "run-8"})

        self.assertEqual(set(dataset.groups[validation].tolist()), {"run-7", "run-8"})
        self.assertNotIn("run-8", set(dataset.groups[training].tolist()))
        self.assertGreaterEqual(len(training), 20)

    def test_training_records_architecture_splits_and_loads_custom_model(self) -> None:
        dataset = _dataset()
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "custom"
            ensemble = train_surrogate_ensemble(
                dataset=dataset,
                config=TuningConfig(),
                artifact_dir=artifact,
                operating_signature={"test": True},
                members=1,
                epochs=2,
                batch_size=32,
                seed=17,
                hidden_sizes=(8,),
                train_indexes=np.arange(0, 24),
                validation_indexes=np.arange(24, 28),
                evaluation_indexes=np.arange(28, 36),
                early_stopping_patience=1,
            )

            manifest = ensemble.manifest
            self.assertEqual(manifest["hidden_sizes"], [8])
            self.assertEqual(manifest["ensemble_members"], 1)
            self.assertEqual(manifest["parameter_count_per_member"], 307)
            self.assertEqual(manifest["trainable_parameter_count"], 307)
            self.assertEqual(manifest["training_seed"], 17)
            self.assertEqual(manifest["epochs_requested"], 2)
            self.assertEqual(manifest["batch_size"], 32)
            self.assertEqual(manifest["evaluation_sample_count"], 8)
            self.assertEqual(len(manifest["dataset_hash"]), 64)
            self.assertEqual(len(manifest["split_hash"]), 64)
            self.assertFalse(manifest["evaluation_reuses_validation"])

            loaded = SurrogateEnsemble.load(artifact, device="cpu")
            self.assertEqual(loaded.manifest["hidden_sizes"], [8])
            self.assertEqual(loaded.members[0].metric_head.in_features, 8)
            self.assertEqual(len(ensemble.feasibility_calibrator["actions"]), 24)
            self.assertEqual(len(loaded.feasibility_calibrator["actions"]), 36)
            prediction = loaded.predict_features(dataset.features[28])
            self.assertEqual(prediction["metric_mean"].shape, (1, len(METRIC_FIELDS)))

    def test_training_imputes_missing_baseline_from_training_partition_only(self) -> None:
        dataset = _dataset()
        dataset.baseline_values[:] = 0.0
        dataset.baseline_mask[:] = 0.0
        dataset.baseline_values[0, 0] = 2.0
        dataset.baseline_mask[0, 0] = 1.0
        dataset.baseline_values[-1, 0] = 999.0
        dataset.baseline_mask[-1, 0] = 1.0
        with tempfile.TemporaryDirectory() as temporary:
            ensemble = train_surrogate_ensemble(
                dataset=dataset,
                config=TuningConfig(),
                artifact_dir=Path(temporary),
                operating_signature={"test": True},
                members=1,
                epochs=1,
                seed=23,
                hidden_sizes=(),
                train_indexes=np.arange(0, 24),
                validation_indexes=np.arange(24, 28),
                evaluation_indexes=np.arange(28, 36),
                early_stopping_patience=1,
            )

            self.assertEqual(ensemble.manifest["baseline_imputation_values"][0], 2.0)
            probe = dataset.features[-2].copy()
            baseline_index = len(ACTION_FIELDS)
            baseline_mask_index = baseline_index + len(METRIC_FIELDS)
            probe[baseline_index] = 999.0
            probe[baseline_mask_index] = 0.0
            captured: dict[str, np.ndarray] = {}

            def hook(_module, inputs):
                captured["normalized"] = inputs[0].detach().cpu().numpy().copy()

            handle = ensemble.members[0].register_forward_pre_hook(hook)
            try:
                ensemble.predict_features(probe)
            finally:
                handle.remove()
            expected = np.clip(
                (2.0 - ensemble.feature_mean[baseline_index]) / ensemble.feature_std[baseline_index],
                -6.0,
                6.0,
            )
            self.assertAlmostEqual(
                float(captured["normalized"][0, baseline_index]),
                float(expected),
                places=5,
            )

    def test_loader_defaults_old_manifest_to_legacy_architecture(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary)
            member_path = artifact / "surrogate_1.pt"
            scaler_path = artifact / "scalers.npz"
            member = _surrogate_network_type(torch)(
                25,
                len(METRIC_FIELDS),
                3,
                hidden_sizes=LEGACY_SURROGATE_HIDDEN_SIZES,
            )
            torch.save(member.state_dict(), member_path)
            np.savez_compressed(
                scaler_path,
                feature_mean=np.zeros(25, dtype=np.float32),
                feature_std=np.ones(25, dtype=np.float32),
                metric_mean=np.zeros(len(METRIC_FIELDS), dtype=np.float32),
                metric_std=np.ones(len(METRIC_FIELDS), dtype=np.float32),
            )
            manifest = {
                "model_id": "legacy",
                "feature_count": 25,
                "metric_count": len(METRIC_FIELDS),
                "invalid_label_count": 3,
                "member_files": [member_path.name],
                "files_sha256": {
                    member_path.name: hashlib.sha256(member_path.read_bytes()).hexdigest(),
                    scaler_path.name: hashlib.sha256(scaler_path.read_bytes()).hexdigest(),
                },
            }
            (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            loaded = SurrogateEnsemble.load(artifact, device="cpu")
            self.assertEqual(loaded.members[0].metric_head.in_features, 64)
            self.assertEqual(len(loaded.members[0].backbone), 6)

    def test_explicit_splits_reject_partial_run_group_leakage(self) -> None:
        dataset = _dataset()
        with self.assertRaisesRegex(ValueError, "complete run groups"):
            _resolve_training_splits(
                dataset,
                seed=1,
                train_indexes=np.arange(0, 21),
                validation_indexes=np.arange(21, 24),
                evaluation_indexes=np.arange(24, 28),
            )


if __name__ == "__main__":
    unittest.main()
