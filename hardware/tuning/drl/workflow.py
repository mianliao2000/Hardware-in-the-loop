"""Persistent orchestration for fixed-operating-point Safe SAC workflows."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable

from ..models import AutotuneExperimentConfig, SearchSpace, TuningConfig
from ..search import HardwareGridHeuristicTuner
from .common import (
    artifact_id,
    atomic_write_json,
    candidate_to_mapping,
    operating_signature,
    read_json,
    signatures_compatible,
)
from .dataset import (
    build_bootstrap_collection_plan,
    build_collection_plan,
    build_targeted_collection_plan,
    load_autotune_dataset,
    save_collection_plan,
)
from .model import SurrogateEnsemble, dependency_status, require_ml_dependencies, train_surrogate_ensemble
from .policy import SafeSacTuner, train_safe_sac_policy, validation_start_candidates
from .tuner import PlannedCandidateTuner


COLLECTION_BUDGET = 240
TARGETED_COLLECTION_BUDGET = 100
VALIDATION_BUDGET = 60
VALIDATION_EPISODES = 4
VALIDATION_EPISODE_BUDGET = 15
VALIDATION_CONFIRMATIONS = 3


def _hardware_episode_budget(
    config: TuningConfig,
    experiment: AutotuneExperimentConfig,
    *,
    is_validation: bool,
) -> int:
    configured = max(1, int(experiment.drl_episode_budget))
    if is_validation or not experiment.ignore_pass_until_max_iterations:
        return configured
    return max(configured, config.search.total_iteration_budget())


class WorkflowStopped(RuntimeError):
    pass


class DrlWorkflowManager:
    """Coordinate collection, training, and frozen-policy hardware validation."""

    def __init__(self, artifact_root: Path, run_roots: list[Path]):
        self.artifact_root = artifact_root
        self.dataset_root = artifact_root / "datasets"
        self.model_root = artifact_root / "models"
        self.plan_root = artifact_root / "plans"
        self.status_path = artifact_root / "workflow_manifest.json"
        self.run_roots = list(run_roots)
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._session_status: Callable[[], dict[str, Any]] | None = None
        self._session_stop: Callable[[], dict[str, Any]] | None = None
        self._start_hardware: Callable[[TuningConfig, AutotuneExperimentConfig], dict[str, Any]] | None = None
        self._resume_hardware: Callable[[str, str], dict[str, Any]] | None = None
        self._persist_hardware: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._state = self._load_state()

    def bind_session(
        self,
        *,
        status: Callable[[], dict[str, Any]],
        stop: Callable[[], dict[str, Any]],
        start_hardware: Callable[[TuningConfig, AutotuneExperimentConfig], dict[str, Any]],
        resume_hardware: Callable[[str, str], dict[str, Any]],
        persist_hardware: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._session_status = status
        self._session_stop = stop
        self._start_hardware = start_hardware
        self._resume_hardware = resume_hardware
        self._persist_hardware = persist_hardware

    def candidate_tuner_factory(
        self,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
        history: list[Any],
    ) -> Any:
        algorithm = str(experiment.optimization_algorithm or "heuristic").strip().lower()
        if algorithm in {"", "heuristic"}:
            return HardwareGridHeuristicTuner(config.search)
        _validate_fixed_condition(config, experiment)
        signature = operating_signature(config, experiment)
        if algorithm == "drl-collection":
            plan_id = _safe_artifact_name(experiment.drl_collection_plan_id, "collection plan")
            with self._lock:
                authorized = (
                    self._state.get("workflow") == "collection"
                    and str(self._state.get("collection_plan_id") or "") == plan_id
                    and self._state.get("state") in {"collecting", "paused", "stopped"}
                )
            if not authorized:
                raise RuntimeError("DRL collection is fail-closed because this plan is not the active workflow.")
            plan_path = self.plan_root / plan_id / "plan.json"
            plan = read_json(plan_path)
            if not plan_id or plan is None:
                raise RuntimeError("DRL collection is fail-closed because its persisted plan is missing.")
            if not signatures_compatible(plan.get("operating_signature") or {}, signature):
                raise RuntimeError("DRL collection plan is incompatible with the current board configuration.")
            plan_manifest = read_json(plan_path.parent / "manifest.json") or {}
            expected_hash = (plan_manifest.get("files_sha256") or {}).get("plan.json")
            if not expected_hash or _sha256(plan_path) != expected_hash:
                raise RuntimeError("DRL collection plan failed its persisted SHA-256 integrity check.")
            return PlannedCandidateTuner(plan_path, history, search=config.search)
        if algorithm not in {"deep-reinforcement", "safe-sac"}:
            raise RuntimeError(f"Unsupported optimization algorithm: {experiment.optimization_algorithm}")

        model_id = _safe_artifact_name(
            experiment.drl_model_id or self._state.get("model_id"),
            "model",
        )
        model_dir = self.model_root / model_id
        manifest = read_json(model_dir / "manifest.json")
        if not model_id or manifest is None:
            raise RuntimeError("Safe SAC is fail-closed because no persisted model is available.")
        if not signatures_compatible(manifest.get("operating_signature") or {}, signature):
            raise RuntimeError(f"Safe SAC model '{model_id}' is incompatible with the current configuration.")
        hardware_protection_mode = bool(experiment.drl_hardware_protection_mode)
        if not hardware_protection_mode and not bool(manifest.get("ready")):
            raise RuntimeError(f"Safe SAC model '{model_id}' did not pass synthetic acceptance gates.")
        is_validation = str(experiment.drl_workflow_mode).strip().lower() == "validate"
        if is_validation:
            with self._lock:
                authorized = (
                    self._state.get("workflow") == "validation"
                    and str(self._state.get("model_id") or "") == model_id
                    and self._state.get("state") in {"validating", "paused", "stopped"}
                )
            if not authorized:
                raise RuntimeError("Safe SAC validation is fail-closed because it is not the active workflow.")
        if not hardware_protection_mode and not is_validation and not bool(manifest.get("hardware_ready")):
            raise RuntimeError(f"Safe SAC model '{model_id}' has not passed hardware validation.")

        ensemble = SurrogateEnsemble.load(model_dir)
        starts_payload = read_json(model_dir / "validation_starts.json") or {}
        starts = [
            _candidate_from_payload(item)
            for item in starts_payload.get("candidates", [])
            if isinstance(item, dict)
        ]
        exploration_payload = read_json(model_dir / "exploration_starts.json") or {}
        exploration_starts = [
            _candidate_from_payload(item)
            for item in exploration_payload.get("candidates", [])
            if isinstance(item, dict)
        ]
        required_starts = VALIDATION_EPISODES if is_validation else 1
        if len(starts) < required_starts:
            raise RuntimeError(
                f"Safe SAC model '{model_id}' is missing its {required_starts} persisted validation start(s)."
            )
        run_full_budget = bool(experiment.ignore_pass_until_max_iterations)
        # A normal DRL run has one continuing hardware episode.  Previously
        # its 15-step policy horizon was also used as the candidate-source
        # lifetime, so a requested 500-iteration run stopped after 15 points
        # with "no fresh candidates remain".  Keep the short horizon for
        # validation/early-stop runs, but let full-budget hardware learning
        # span the outer tuning budget.
        episode_budget = _hardware_episode_budget(
            config,
            experiment,
            is_validation=is_validation,
        )
        return SafeSacTuner(
            ensemble=ensemble,
            policy_path=model_dir / "safe_sac_policy.zip",
            config=config,
            history=history,
            validation_starts=starts,
            exploration_starts=exploration_starts,
            episode_budget=episode_budget,
            confirmation_count=experiment.drl_confirmation_count,
            validation_episodes=VALIDATION_EPISODES if is_validation else 1,
            hardware_protection_mode=hardware_protection_mode,
            run_full_budget=run_full_budget,
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._reconcile_hardware_locked()
            self._reconcile_model_compatibility_locked()
            return {"ok": True, **self._state}

    def assert_tuning_available(self) -> None:
        with self._lock:
            if bool(self._state.get("busy")) or (self._worker is not None and self._worker.is_alive()):
                raise RuntimeError(
                    f"The DRL {self._state.get('workflow') or 'workflow'} task is active; tuning actions are mutually exclusive."
                )

    def start_collection(
        self,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
    ) -> dict[str, Any]:
        require_ml_dependencies()
        _validate_fixed_condition(config, experiment)
        signature = operating_signature(config, experiment)
        with self._lock:
            self._ensure_can_start_locked("collection")
            if self._can_resume_locked("collection", signature):
                collection_total = int(self._state.get("collection_total") or COLLECTION_BUDGET)
                self._set_locked(state="collecting", busy=True, message="Resuming DRL hardware collection.")
                self._spawn_locked(self._resume_and_monitor, "collection", collection_total, None)
                return {"ok": True, **self._state}
            self._set_locked(
                state="preparing_collection",
                workflow="collection",
                busy=True,
                message="Preparing the guarded 240-point collection plan.",
                error=None,
                progress=0.0,
                operating_signature=signature,
                collection_completed=0,
                collection_total=COLLECTION_BUDGET,
                collection_finished=False,
                resume_available=False,
                run_id=None,
                run_kind=None,
            )
            self._spawn_locked(self._prepare_collection, config, experiment, signature)
            return {"ok": True, **self._state}

    def start_training(
        self,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
    ) -> dict[str, Any]:
        require_ml_dependencies()
        _validate_fixed_condition(config, experiment)
        signature = operating_signature(config, experiment)
        with self._lock:
            self._ensure_can_start_locked("training")
            if (
                int(self._state.get("collection_completed") or 0) < COLLECTION_BUDGET
                or not bool(self._state.get("collection_finished"))
            ):
                raise RuntimeError("Complete and persist the guarded 240-point collection before training Safe SAC.")
            if not _collection_signature_covers(
                self._state.get("operating_signature") or {},
                signature,
            ):
                raise RuntimeError("The completed collection is incompatible with the current configuration.")
            self._set_locked(
                state="training",
                workflow="training",
                busy=True,
                message="Building the fixed-condition dataset.",
                error=None,
                progress=0.0,
                operating_signature=signature,
                model_status="training",
                model_compatible=True,
                resume_available=False,
            )
            self._spawn_locked(self._train, config, experiment, signature)
            return {"ok": True, **self._state}

    def start_targeted_collection(
        self,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
        source_run_ids: list[str],
    ) -> dict[str, Any]:
        """Retrain a focused surrogate and launch the persisted 100-point plan."""

        require_ml_dependencies()
        _validate_fixed_condition(config, experiment)
        signature = operating_signature(config, experiment)
        selected_runs = sorted({_safe_artifact_name(value, "source run") for value in source_run_ids})
        if len(selected_runs) < 2:
            raise RuntimeError("Targeted collection requires both the bootstrap run and the 500-point run.")
        with self._lock:
            self._ensure_can_start_locked("collection")
            reusable_surrogate_id = self._state.get("targeted_surrogate_model_id")
            self._set_locked(
                state="preparing_collection",
                workflow="collection",
                busy=True,
                message="Retraining the surrogate from the selected bootstrap and 500-point runs.",
                error=None,
                progress=0.0,
                operating_signature=signature,
                collection_completed=0,
                collection_total=TARGETED_COLLECTION_BUDGET,
                collection_finished=False,
                resume_available=False,
                run_id=None,
                run_kind=None,
                targeted_source_run_ids=selected_runs,
                targeted_surrogate_model_id=reusable_surrogate_id,
            )
            self._spawn_locked(
                self._prepare_targeted_collection,
                config,
                experiment,
                signature,
                selected_runs,
            )
            return {"ok": True, **self._state}

    def start_targeted_recovery(
        self,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
        source_plan_id: str,
        source_plan_indexes: list[int],
    ) -> dict[str, Any]:
        """Run selected base proposals that were invalid or truncated by dynamic confirmations."""

        require_ml_dependencies()
        _validate_fixed_condition(config, experiment)
        signature = operating_signature(config, experiment)
        source_plan_id = _safe_artifact_name(source_plan_id, "collection plan")
        source_plan = read_json(self.plan_root / source_plan_id / "plan.json")
        if not isinstance(source_plan, dict):
            raise RuntimeError(f"Targeted source plan '{source_plan_id}' is missing.")
        requested = sorted({int(value) for value in source_plan_indexes if int(value) > 0})
        source_items = source_plan.get("candidates")
        if not requested or not isinstance(source_items, list):
            raise RuntimeError("Targeted recovery requires at least one valid source plan index.")
        by_index = {
            int(item.get("index") or position): item
            for position, item in enumerate(source_items, 1)
            if isinstance(item, dict)
        }
        missing = [index for index in requested if index not in by_index]
        if missing:
            raise RuntimeError(f"Targeted recovery plan indexes are missing: {missing}")

        recovery_id = artifact_id("targeted_recovery")
        recovery_items: list[dict[str, Any]] = []
        for recovery_index, source_index in enumerate(requested, 1):
            source_item = by_index[source_index]
            item = dict(source_item)
            item["candidate"] = dict(source_item.get("candidate") or {})
            metadata = dict(source_item.get("optimizer_metadata") or {})
            metadata.update(
                {
                    "recovery_source_plan_id": source_plan_id,
                    "recovery_source_plan_index": source_index,
                }
            )
            item["optimizer_metadata"] = metadata
            item["index"] = recovery_index
            recovery_items.append(item)
        recovery_plan = {
            **{key: value for key, value in source_plan.items() if key != "candidates"},
            "plan_id": recovery_id,
            "recovery": True,
            "source_plan_id": source_plan_id,
            "source_plan_indexes": requested,
            "budget": len(recovery_items),
            "candidates": recovery_items,
            **_artifact_context(signature),
        }
        plan_path = save_collection_plan(recovery_plan, self.plan_root)
        atomic_write_json(
            plan_path.parent / "manifest.json",
            {
                "plan_id": recovery_id,
                "source_plan_id": source_plan_id,
                "operating_signature": signature,
                "files_sha256": {"plan.json": _sha256(plan_path)},
            },
        )
        # Confirmation and BW-climb measurements are deliberately overhead,
        # not replacements for the selected base proposals. This cap lets the
        # tuner exhaust the recovery plan naturally while still bounding a
        # pathological sequence of successful climbs.
        measurement_cap = max(len(recovery_items), len(recovery_items) * 4)
        model_id = str(source_plan.get("provisional_model_id") or experiment.drl_model_id or "")
        with self._lock:
            self._ensure_can_start_locked("collection")
            self._set_locked(
                state="collecting",
                workflow="collection",
                busy=True,
                message=f"Running {len(recovery_items)} targeted recovery proposals plus confirmation overhead.",
                error=None,
                progress=0.0,
                operating_signature=signature,
                collection_completed=0,
                collection_total=measurement_cap,
                collection_finished=False,
                resume_available=False,
                run_id=None,
                run_kind=None,
                collection_plan_id=recovery_id,
                targeted_recovery_base_count=len(recovery_items),
                targeted_recovery_source_plan_id=source_plan_id,
            )
            recovery_config = _with_budget(config, measurement_cap)
            recovery_experiment = replace(
                experiment,
                optimization_algorithm="drl-collection",
                drl_workflow_mode="collect",
                drl_model_id=model_id,
                drl_collection_plan_id=recovery_id,
                drl_confirmation_count=3,
                ignore_pass_until_max_iterations=True,
            )
            self._spawn_locked(
                self._start_and_monitor,
                recovery_config,
                recovery_experiment,
                "collection",
                measurement_cap,
                None,
            )
            return {"ok": True, **self._state}

    def start_validation(
        self,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
    ) -> dict[str, Any]:
        require_ml_dependencies()
        _validate_fixed_condition(config, experiment)
        signature = operating_signature(config, experiment)
        with self._lock:
            self._ensure_can_start_locked("validation")
            if self._can_resume_locked("validation", signature):
                self._set_locked(state="validating", busy=True, message="Resuming Safe SAC hardware validation.")
                self._spawn_locked(
                    self._resume_and_monitor,
                    "validation",
                    VALIDATION_BUDGET,
                    str(self._state.get("model_id") or ""),
                )
                return {"ok": True, **self._state}
            model_id = _safe_artifact_name(
                experiment.drl_model_id or self._state.get("model_id"),
                "model",
            )
            model_manifest = read_json(self.model_root / model_id / "manifest.json")
            if not model_id or model_manifest is None:
                raise RuntimeError("Train and accept a Safe SAC model before hardware validation.")
            if not signatures_compatible(model_manifest.get("operating_signature") or {}, signature):
                raise RuntimeError(f"Safe SAC model '{model_id}' is incompatible with the current configuration.")
            protection_candidate = bool(
                experiment.drl_hardware_protection_mode
                and model_manifest.get("hardware_protection_policy")
                and model_manifest.get("policy_file")
            )
            if not bool(model_manifest.get("ready")) and not protection_candidate:
                raise RuntimeError(f"Safe SAC model '{model_id}' did not pass synthetic acceptance gates.")

            validation_config = _with_budget(config, VALIDATION_BUDGET)
            validation_experiment = replace(
                experiment,
                optimization_algorithm="deep-reinforcement",
                drl_workflow_mode="validate",
                drl_model_id=model_id,
                drl_episode_budget=VALIDATION_EPISODE_BUDGET,
                drl_confirmation_count=VALIDATION_CONFIRMATIONS,
                ignore_pass_until_max_iterations=True,
            )
            self._set_locked(
                state="validating",
                workflow="validation",
                busy=True,
                message=(
                    "Starting four guarded hardware-protection Safe SAC episodes."
                    if protection_candidate and not bool(model_manifest.get("ready"))
                    else "Starting four guarded Safe SAC hardware episodes."
                ),
                error=None,
                progress=0.0,
                model_id=model_id,
                model_status="validating",
                model_compatible=True,
                operating_signature=signature,
                validation_completed=0,
                validation_total=VALIDATION_BUDGET,
                validation_result=None,
                resume_available=False,
            )
            self._spawn_locked(
                self._start_and_monitor,
                validation_config,
                validation_experiment,
                "validation",
                VALIDATION_BUDGET,
                model_id,
            )
            return {"ok": True, **self._state}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._stop_event.set()
            if self._session_stop is not None and self._state.get("workflow") in {"collection", "validation"}:
                try:
                    self._session_stop()
                except Exception:
                    pass
            self._set_locked(
                state="stopped",
                busy=False,
                message="DRL workflow stopped.",
                resume_available=False,
            )
            return {"ok": True, **self._state}

    def _prepare_collection(
        self,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
        signature: dict[str, Any],
    ) -> None:
        try:
            dataset, dataset_manifest = load_autotune_dataset(self.run_roots, config, experiment)
            self._raise_if_stopped()
            dataset_manifest.update(_artifact_context(signature))
            dataset_id = str(dataset_manifest["dataset_id"])
            dataset.save(self.dataset_root / dataset_id, dataset_manifest)
            self._progress(
                0.05,
                f"Loaded {dataset.size} usable measurements from {dataset_manifest.get('source_record_count', dataset.size)} records.",
                dataset_count=dataset.size,
                dataset_source_count=int(dataset_manifest.get("source_record_count", dataset.size)),
                dataset_id=dataset_id,
            )

            provisional_id = ""
            if dataset.size >= 20:
                provisional_id = artifact_id("surrogate_precollect")
                ensemble = train_surrogate_ensemble(
                    dataset=dataset,
                    config=config,
                    artifact_dir=self.model_root / provisional_id,
                    operating_signature=signature,
                    members=5,
                    epochs=_env_int("DRL_SURROGATE_EPOCHS", 300),
                    progress=lambda value, message: self._progress(0.05 + value * 0.70, message),
                )
                provisional_manifest = dict(ensemble.manifest)
                provisional_manifest.update(_artifact_context(signature))
                atomic_write_json(ensemble.artifact_dir / "manifest.json", provisional_manifest)
                ensemble.manifest = provisional_manifest
                self._raise_if_stopped()
                plan = build_collection_plan(dataset, config, ensemble)
            else:
                self._progress(
                    0.20,
                    "No compatible 9-D seed data; building the hardware-protected bootstrap Sobol plan.",
                )
                plan = build_bootstrap_collection_plan(config)
            plan.update(_artifact_context(signature))
            plan["provisional_model_id"] = provisional_id or None
            for item in plan.get("candidates", []):
                if isinstance(item, dict):
                    metadata = item.setdefault("optimizer_metadata", {})
                    if isinstance(metadata, dict) and provisional_id:
                        metadata["model_id"] = provisional_id
            plan_path = save_collection_plan(plan, self.plan_root)
            atomic_write_json(
                plan_path.parent / "manifest.json",
                {
                    "plan_id": plan["plan_id"],
                    "operating_signature": signature,
                    "files_sha256": {"plan.json": _sha256(plan_path)},
                },
            )
            with self._lock:
                self._set_locked(
                    state="collecting",
                    busy=True,
                    progress=0.80,
                    message=(
                        "Collection plan passed model safety screening; starting hardware collection."
                        if provisional_id
                        else "Starting the hardware-protected 9-D bootstrap collection."
                    ),
                    collection_plan_id=plan["plan_id"],
                )

            collection_config = _with_budget(config, COLLECTION_BUDGET)
            collection_experiment = replace(
                experiment,
                optimization_algorithm="drl-collection",
                drl_workflow_mode="collect",
                drl_model_id=provisional_id,
                drl_collection_plan_id=str(plan["plan_id"]),
                ignore_pass_until_max_iterations=True,
            )
            self._start_and_monitor(
                collection_config,
                collection_experiment,
                "collection",
                COLLECTION_BUDGET,
                None,
            )
        except WorkflowStopped:
            return
        except Exception as exc:
            self._fail(exc)

    def _prepare_targeted_collection(
        self,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
        signature: dict[str, Any],
        source_run_ids: list[str],
    ) -> None:
        try:
            dataset, dataset_manifest = load_autotune_dataset(
                self.run_roots,
                config,
                experiment,
                allow_legacy_inferred=False,
                include_run_ids=set(source_run_ids),
            )
            self._raise_if_stopped()
            dataset_manifest.update(
                {
                    **_artifact_context(signature),
                    "purpose": "targeted_100_point_followup",
                    "source_run_ids": source_run_ids,
                }
            )
            dataset_id = str(dataset_manifest["dataset_id"])
            dataset.save(self.dataset_root / dataset_id, dataset_manifest)
            self._progress(
                0.05,
                f"Loaded {dataset.size} compatible samples from the selected runs; retraining [96,64,32].",
                dataset_count=dataset.size,
                dataset_source_count=int(dataset_manifest.get("source_record_count", dataset.size)),
                dataset_id=dataset_id,
            )

            with self._lock:
                reusable_surrogate_id = str(self._state.get("targeted_surrogate_model_id") or "")
            reusable_dir = self.model_root / reusable_surrogate_id
            reusable_manifest = read_json(reusable_dir / "manifest.json") if reusable_surrogate_id else None
            can_reuse = bool(
                reusable_manifest
                and reusable_manifest.get("purpose") == "targeted_100_point_proposal_surrogate"
                and sorted(reusable_manifest.get("source_run_ids") or []) == sorted(source_run_ids)
                and signatures_compatible(reusable_manifest.get("operating_signature") or {}, signature)
            )
            if can_reuse:
                surrogate_id = reusable_surrogate_id
                model_dir = reusable_dir
                ensemble = SurrogateEnsemble.load(model_dir)
                self._progress(0.70, f"Reusing completed targeted surrogate {surrogate_id}.")
            else:
                surrogate_id = artifact_id("surrogate_targeted")
                model_dir = self.model_root / surrogate_id
                ensemble = train_surrogate_ensemble(
                    dataset=dataset,
                    config=config,
                    artifact_dir=model_dir,
                    operating_signature=signature,
                    members=_env_int("DRL_TARGETED_SURROGATE_MEMBERS", 5),
                    epochs=_env_int("DRL_TARGETED_SURROGATE_EPOCHS", 120),
                    hidden_sizes=(96, 64, 32),
                    progress=lambda value, message: self._progress(0.05 + value * 0.65, message),
                )
                self._raise_if_stopped()
                surrogate_manifest = dict(ensemble.manifest)
                surrogate_manifest.update(
                    {
                        **_artifact_context(signature),
                        "purpose": "targeted_100_point_proposal_surrogate",
                        "dataset_id": dataset_id,
                        "source_run_ids": source_run_ids,
                        "hardware_ready": False,
                        "offline_only": True,
                        "files_sha256": _directory_hashes(model_dir),
                    }
                )
                atomic_write_json(model_dir / "manifest.json", surrogate_manifest)
                ensemble.manifest = surrogate_manifest
            self._progress(
                0.72,
                "Surrogate retraining complete; building the 15/56/19/10 mixed proposal pool.",
                targeted_surrogate_model_id=surrogate_id,
            )

            plan = build_targeted_collection_plan(dataset, config, ensemble)
            plan.update(_artifact_context(signature))
            plan["provisional_model_id"] = surrogate_id
            plan["source_run_ids"] = source_run_ids
            for item in plan.get("candidates", []):
                if isinstance(item, dict):
                    metadata = item.setdefault("optimizer_metadata", {})
                    if isinstance(metadata, dict):
                        metadata["model_id"] = surrogate_id
            plan_path = save_collection_plan(plan, self.plan_root)
            atomic_write_json(
                plan_path.parent / "manifest.json",
                {
                    "plan_id": plan["plan_id"],
                    "operating_signature": signature,
                    "files_sha256": {"plan.json": _sha256(plan_path)},
                },
            )
            with self._lock:
                self._set_locked(
                    state="collecting",
                    busy=True,
                    progress=0.80,
                    message="Starting the guarded targeted 100-point hardware experiment.",
                    collection_plan_id=plan["plan_id"],
                    collection_total=TARGETED_COLLECTION_BUDGET,
                    targeted_surrogate_model_id=surrogate_id,
                    targeted_plan_allocation=plan.get("allocation"),
                )

            collection_config = _with_budget(config, TARGETED_COLLECTION_BUDGET)
            collection_experiment = replace(
                experiment,
                optimization_algorithm="drl-collection",
                drl_workflow_mode="collect",
                drl_model_id=surrogate_id,
                drl_collection_plan_id=str(plan["plan_id"]),
                drl_confirmation_count=3,
                ignore_pass_until_max_iterations=True,
            )
            self._start_and_monitor(
                collection_config,
                collection_experiment,
                "collection",
                TARGETED_COLLECTION_BUDGET,
                None,
            )
        except WorkflowStopped:
            return
        except Exception as exc:
            self._fail(exc)

    def _train(
        self,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
        signature: dict[str, Any],
    ) -> None:
        try:
            dataset, dataset_manifest = load_autotune_dataset(self.run_roots, config, experiment)
            self._raise_if_stopped()
            dataset_manifest.update(_artifact_context(signature))
            dataset_id = str(dataset_manifest["dataset_id"])
            dataset.save(self.dataset_root / dataset_id, dataset_manifest)
            self._progress(
                0.03,
                f"Loaded {dataset.size} usable measurements from {dataset_manifest.get('source_record_count', dataset.size)} records.",
                dataset_count=dataset.size,
                dataset_source_count=int(dataset_manifest.get("source_record_count", dataset.size)),
                dataset_id=dataset_id,
            )

            model_id = artifact_id("safe_sac")
            model_dir = self.model_root / model_id
            ensemble = train_surrogate_ensemble(
                dataset=dataset,
                config=config,
                artifact_dir=model_dir,
                operating_signature=signature,
                members=5,
                epochs=_env_int("DRL_SURROGATE_EPOCHS", 300),
                progress=lambda value, message: self._progress(0.03 + value * 0.37, message),
            )
            self._raise_if_stopped()
            hardware_protection_mode = bool(experiment.drl_hardware_protection_mode)
            if not ensemble.accepted and not hardware_protection_mode:
                with self._lock:
                    self._set_locked(
                        state="model_rejected",
                        busy=False,
                        progress=1.0,
                        model_id=model_id,
                        model_status="surrogate_rejected",
                        message="Surrogate acceptance gates failed; SAC and hardware validation remain disabled.",
                        acceptance=ensemble.manifest.get("acceptance"),
                    )
                return

            starts = validation_start_candidates(dataset, count=VALIDATION_EPISODES)
            if len(starts) < VALIDATION_EPISODES:
                raise RuntimeError("Four distinct safe validation starts could not be constructed.")
            atomic_write_json(
                model_dir / "validation_starts.json",
                {"candidates": [candidate_to_mapping(candidate) for candidate in starts]},
            )
            self._progress(0.42, "Training Safe SAC in the surrogate environment.", model_id=model_id)
            manifest = train_safe_sac_policy(
                ensemble=ensemble,
                dataset=dataset,
                config=config,
                # Bootstrap training is intentionally short: the first policy only
                # needs to be useful enough to begin guarded hardware learning.
                # Long 1M-step studies remain available through the environment
                # overrides used by the offline sweep.
                total_steps=_env_int("DRL_SAC_STEPS", 75_000),
                evaluation_episodes=_env_int("DRL_EVAL_EPISODES", 1_000),
                max_episode_steps=VALIDATION_EPISODE_BUDGET,
                progress=lambda value, message: self._progress(0.42 + value * 0.58, message),
                allow_unaccepted_surrogate=hardware_protection_mode,
            )
            self._raise_if_stopped()
            manifest.update(_artifact_context(signature))
            manifest["dataset_id"] = dataset_id
            manifest["hardware_ready"] = False
            manifest["hardware_protection_policy"] = hardware_protection_mode
            manifest["files_sha256"] = _directory_hashes(model_dir)
            atomic_write_json(model_dir / "manifest.json", manifest)
            ready = bool(manifest.get("ready"))
            protection_policy_ready = hardware_protection_mode and bool(manifest.get("policy_file"))
            with self._lock:
                self._set_locked(
                    state="ready_for_validation" if ready else ("hardware_protection_ready" if protection_policy_ready else "model_rejected"),
                    busy=False,
                    progress=1.0,
                    model_id=model_id,
                    model_status=(
                        "ready_for_validation"
                        if ready
                        else ("hardware_protection_ready" if protection_policy_ready else "synthetic_rejected")
                    ),
                    model_compatible=True,
                    message=(
                        "Safe SAC passed synthetic gates and is ready for guarded hardware validation."
                        if ready
                        else (
                            "Safe SAC policy is available in hardware-protection mode; trip recovery remains authoritative."
                            if protection_policy_ready
                            else "Safe SAC failed synthetic acceptance gates; hardware validation remains disabled."
                        )
                    ),
                    acceptance=manifest.get("acceptance"),
                    policy_evaluation=manifest.get("policy_evaluation"),
                )
        except WorkflowStopped:
            return
        except Exception as exc:
            self._fail(exc)

    def _start_and_monitor(
        self,
        config: TuningConfig,
        experiment: AutotuneExperimentConfig,
        workflow: str,
        budget: int,
        model_id: str | None,
    ) -> None:
        try:
            self._raise_if_stopped()
            if self._start_hardware is None:
                raise RuntimeError("DRL workflow is not bound to a tuning session.")
            status = self._start_hardware(config, experiment)
            self._remember_run(status)
            self._monitor_hardware(workflow, budget, model_id)
        except WorkflowStopped:
            return
        except Exception as exc:
            self._fail(exc)

    def _resume_and_monitor(self, workflow: str, budget: int, model_id: str | None) -> None:
        try:
            if self._resume_hardware is None:
                raise RuntimeError("DRL workflow is not bound to a run store.")
            run_id = str(self._state.get("run_id") or "")
            run_kind = str(self._state.get("run_kind") or "recent")
            if not run_id:
                raise RuntimeError("No persisted DRL hardware run is available to resume.")
            status = self._resume_hardware(run_id, run_kind)
            self._remember_run(status)
            self._monitor_hardware(workflow, budget, model_id)
        except WorkflowStopped:
            return
        except Exception as exc:
            self._fail(exc)

    def _monitor_hardware(self, workflow: str, budget: int, model_id: str | None) -> None:
        last_persisted = -1
        while True:
            self._raise_if_stopped()
            if self._session_status is None:
                raise RuntimeError("DRL workflow is not bound to a tuning session.")
            status = self._session_status()
            history = status.get("history") if isinstance(status.get("history"), list) else []
            completed = len(history)
            if completed != last_persisted and self._persist_hardware is not None:
                status = self._persist_hardware(status)
                last_persisted = completed
            field = "collection_completed" if workflow == "collection" else "validation_completed"
            with self._lock:
                self._set_locked(
                    state="collecting" if workflow == "collection" else "validating",
                    workflow=workflow,
                    busy=status.get("state") == "running",
                    message=str(status.get("message") or "Running hardware workflow."),
                    progress=min(1.0, completed / max(1, budget)),
                    **{field: completed},
                )
            state = str(status.get("state") or "")
            if state == "running":
                time.sleep(0.5)
                continue
            if self._persist_hardware is not None:
                status = self._persist_hardware(status)
            if state == "complete":
                if workflow == "validation":
                    self._finalize_validation(status, model_id or "")
                else:
                    with self._lock:
                        self._set_locked(
                            state="collection_complete",
                            busy=False,
                            progress=1.0,
                            collection_finished=True,
                            resume_available=False,
                            message=f"DRL collection complete: {completed}/{budget} measurements persisted.",
                        )
                return
            if state in {"paused", "stopped"}:
                with self._lock:
                    self._set_locked(
                        state="paused",
                        busy=False,
                        resume_available=True,
                        message=str(status.get("message") or "Hardware workflow paused."),
                    )
                return
            if state == "error":
                raise RuntimeError(str(status.get("message") or "Hardware workflow failed."))
            time.sleep(0.5)

    def _finalize_validation(self, status: dict[str, Any], model_id: str) -> None:
        history = status.get("history") if isinstance(status.get("history"), list) else []
        episodes: dict[int, list[dict[str, Any]]] = {}
        for record in history:
            if not isinstance(record, dict):
                continue
            metadata = record.get("optimizer_metadata") if isinstance(record.get("optimizer_metadata"), dict) else {}
            if str(metadata.get("algorithm") or "").lower() != "deep-reinforcement":
                continue
            episode = int(metadata.get("episode") or 0)
            episodes.setdefault(episode, []).append(record)
        successes = sum(_episode_has_confirmation(records, VALIDATION_CONFIRMATIONS) for records in episodes.values())
        accepted = len(episodes) >= VALIDATION_EPISODES and successes >= 3
        result = {
            "episodes_completed": len(episodes),
            "episodes_succeeded": successes,
            "required_successes": 3,
            "confirmation_count": VALIDATION_CONFIRMATIONS,
            "hardware_points": len(history),
            "accepted": accepted,
        }
        model_dir = self.model_root / model_id
        manifest = read_json(model_dir / "manifest.json")
        if manifest is not None:
            manifest["hardware_validation"] = result
            manifest["hardware_ready"] = accepted
            if accepted:
                manifest["offline_only"] = False
                manifest["hardware_validated_at"] = time.time()
            manifest["files_sha256"] = _directory_hashes(model_dir)
            atomic_write_json(model_dir / "manifest.json", manifest)
        with self._lock:
            self._set_locked(
                state="hardware_ready" if accepted else "validation_failed",
                busy=False,
                progress=1.0,
                validation_result=result,
                model_status="hardware_ready" if accepted else "validation_failed",
                resume_available=False,
                message=(
                    f"Hardware validation passed: {successes}/{len(episodes)} episodes succeeded."
                    if accepted
                    else f"Hardware validation failed: {successes}/{len(episodes)} episodes succeeded; DRL remains unavailable."
                ),
            )

    def _load_state(self) -> dict[str, Any]:
        state = read_json(self.status_path) or {}
        if state.get("state") in {"collecting", "validating", "preparing_collection", "training"}:
            state.update(
                {
                    "state": "paused",
                    "busy": False,
                    "resume_available": state.get("workflow") in {"collection", "validation"},
                    "message": "Server restarted. The persisted DRL hardware workflow is paused and can be resumed.",
                }
            )
        default = {
            "schema_version": 1,
            "state": "idle",
            "workflow": None,
            "busy": False,
            "message": "DRL workflow is idle.",
            "error": None,
            "progress": 0.0,
            "dataset_count": 0,
            "dataset_source_count": 0,
            "dataset_id": None,
            "collection_completed": 0,
            "collection_total": COLLECTION_BUDGET,
            "collection_finished": False,
            "validation_completed": 0,
            "validation_total": VALIDATION_BUDGET,
            "model_id": None,
            "model_status": "missing",
            "model_compatible": False,
            "validation_result": None,
            "resume_available": False,
            "run_id": None,
            "run_kind": None,
            "dependency": dependency_status(),
        }
        default.update(state)
        default["dependency"] = dependency_status()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.status_path, default)
        return default

    def _ensure_can_start_locked(self, requested: str) -> None:
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError(f"Another DRL {self._state.get('workflow') or 'workflow'} task is already active.")
        session_state = self._session_status().get("state") if self._session_status is not None else "idle"
        resuming_this_workflow = bool(
            self._state.get("state") == "paused"
            and self._state.get("resume_available")
            and self._state.get("workflow") == requested
        )
        if session_state == "running" or (session_state == "paused" and not resuming_this_workflow):
            raise RuntimeError("A hardware tuning run is already active or paused. Stop it before starting DRL.")
        if self._state.get("state") == "paused" and self._state.get("resume_available"):
            if self._state.get("workflow") != requested:
                raise RuntimeError(f"Resume or stop the paused DRL {self._state.get('workflow')} workflow first.")
        self._stop_event.clear()

    def _can_resume_locked(self, workflow: str, signature: dict[str, Any]) -> bool:
        return bool(
            self._state.get("state") == "paused"
            and self._state.get("resume_available")
            and self._state.get("workflow") == workflow
            and self._state.get("run_id")
            and signatures_compatible(self._state.get("operating_signature") or {}, signature)
        )

    def _spawn_locked(self, target: Callable[..., None], *args: Any) -> None:
        self._worker = threading.Thread(target=target, args=args, name="drl-workflow", daemon=True)
        self._worker.start()

    def _remember_run(self, status: dict[str, Any]) -> None:
        run = status.get("run") if isinstance(status.get("run"), dict) else {}
        with self._lock:
            self._set_locked(
                run_id=run.get("run_id") or self._state.get("run_id"),
                run_kind=run.get("kind") or self._state.get("run_kind") or "recent",
            )

    def _reconcile_hardware_locked(self) -> None:
        if self._state.get("workflow") not in {"collection", "validation"} or self._session_status is None:
            return
        if self._state.get("state") not in {"collecting", "validating", "paused", "stopped"}:
            return
        if self._state.get("state") == "stopped" and not self._state.get("resume_available"):
            return
        status = self._session_status()
        session_state = str(status.get("state") or "")
        if self._state.get("state") in {"paused", "stopped"} and session_state == "idle":
            return
        history = status.get("history") if isinstance(status.get("history"), list) else []
        field = "collection_completed" if self._state.get("workflow") == "collection" else "validation_completed"
        total = (
            int(self._state.get("collection_total") or COLLECTION_BUDGET)
            if self._state.get("workflow") == "collection"
            else VALIDATION_BUDGET
        )
        previous_completed = int(self._state.get(field) or 0)
        if len(history) != previous_completed and self._persist_hardware is not None:
            status = self._persist_hardware(status)
        self._state[field] = len(history)
        self._state["progress"] = min(1.0, len(history) / max(1, total))
        if session_state == "running":
            self._state["state"] = "collecting" if self._state.get("workflow") == "collection" else "validating"
            self._state["busy"] = True
        elif session_state in {"paused", "stopped"}:
            self._state["state"] = "paused"
            self._state["busy"] = False
            self._state["resume_available"] = True
        elif session_state == "complete":
            if self._persist_hardware is not None:
                status = self._persist_hardware(status)
            if self._state.get("workflow") == "validation":
                self._finalize_validation(status, str(self._state.get("model_id") or ""))
                return
            self._state.update(
                {
                    "state": "collection_complete",
                    "busy": False,
                    "progress": 1.0,
                    "collection_finished": True,
                    "resume_available": False,
                    "message": f"DRL collection complete: {len(history)}/{total} measurements persisted.",
                }
            )
        elif session_state == "error":
            self._state.update(
                {
                    "state": "error",
                    "busy": False,
                    "error": str(status.get("message") or "Hardware workflow failed."),
                    "message": str(status.get("message") or "Hardware workflow failed."),
                }
            )
        self._write_state_locked()

    def _reconcile_model_compatibility_locked(self) -> None:
        """Report compatibility against the session's current live settings.

        Compatibility used to be copied from the last workflow transition and
        could remain ``true`` after the action schema or GUI search space had
        changed. Recompute it when the session exposes complete config and
        experiment payloads, but leave synthetic/test bindings without those
        payloads untouched.
        """

        if self._session_status is None:
            return
        session = self._session_status()
        config_payload = session.get("config")
        experiment_payload = session.get("experiment")
        if not isinstance(config_payload, dict) or not isinstance(experiment_payload, dict):
            return

        raw_model_id = str(self._state.get("model_id") or "").strip()
        model_id = _safe_artifact_name(raw_model_id, "model") if raw_model_id else ""
        manifest = read_json(self.model_root / model_id / "manifest.json") if model_id else None
        compatible = False
        reason = "No persisted DRL model is selected."
        expected_signature: dict[str, Any] | None = None
        actual_signature: dict[str, Any] | None = None
        if manifest is not None:
            expected_signature = manifest.get("operating_signature") or {}
            try:
                # Runtime import avoids a module cycle during runner startup.
                from ..runner import _config_from_payload, _experiment_from_payload

                actual_signature = operating_signature(
                    _config_from_payload(config_payload),
                    _experiment_from_payload(experiment_payload),
                )
                compatible = signatures_compatible(expected_signature, actual_signature)
                if compatible:
                    reason = "The selected model matches the current 9-dimensional operating schema."
                else:
                    reason = "The selected model does not match the current operating/action schema. Retraining is required."
            except Exception as exc:
                reason = f"Could not verify model compatibility: {exc}"

        updates = {
            "model_compatible": compatible,
            "model_compatibility_reason": reason,
            "model_expected_signature": expected_signature,
            "model_actual_signature": actual_signature,
        }
        if any(self._state.get(key) != value for key, value in updates.items()):
            self._state.update(updates)
            self._state["updated_at"] = time.time()
            self._write_state_locked()

    def _set_locked(self, **updates: Any) -> None:
        self._state.update(updates)
        self._state["updated_at"] = time.time()
        self._write_state_locked()

    def _write_state_locked(self) -> None:
        atomic_write_json(self.status_path, self._state)

    def _progress(self, value: float, message: str, **updates: Any) -> None:
        self._raise_if_stopped()
        with self._lock:
            self._set_locked(progress=max(0.0, min(1.0, float(value))), message=message, **updates)

    def _raise_if_stopped(self) -> None:
        if self._stop_event.is_set():
            raise WorkflowStopped("DRL workflow stopped by user.")

    def _fail(self, exc: Exception) -> None:
        with self._lock:
            self._set_locked(
                state="error",
                busy=False,
                error=str(exc),
                message=str(exc),
                resume_available=False,
            )


def _with_budget(config: TuningConfig, budget: int) -> TuningConfig:
    search: SearchSpace = replace(
        config.search,
        max_iterations=budget,
        max_coarse_iterations=budget,
        max_refined_iterations=0,
    )
    return replace(config, search=search)


def _collection_signature_covers(
    collected: dict[str, Any],
    requested: dict[str, Any],
) -> bool:
    """Allow training on a strict search-space subset of a completed run."""

    if signatures_compatible(collected, requested):
        return True
    collected_core = {
        key: value
        for key, value in collected.items()
        if key not in {"signature", "search"}
    }
    requested_core = {
        key: value
        for key, value in requested.items()
        if key not in {"signature", "search"}
    }
    if not collected_core or collected_core != requested_core:
        return False
    collected_search = collected.get("search")
    requested_search = requested.get("search")
    if not isinstance(collected_search, dict) or not isinstance(requested_search, dict):
        return False
    for name, requested_range in requested_search.items():
        collected_range = collected_search.get(name)
        if not isinstance(requested_range, dict) or not isinstance(collected_range, dict):
            return False
        try:
            if float(requested_range["min"]) < float(collected_range["min"]):
                return False
            if float(requested_range["max"]) > float(collected_range["max"]):
                return False
        except (KeyError, TypeError, ValueError):
            return False
    return True


def _validate_fixed_condition(config: TuningConfig, experiment: AutotuneExperimentConfig) -> None:
    if not experiment.enable_transient_analysis or not experiment.enable_bode_analysis:
        raise RuntimeError("Fixed-condition Safe SAC requires both transient and Bode analysis.")
    if (
        str(experiment.board_address).strip().upper() != "0X5E"
        or int(experiment.board_page) != 0
        or str(experiment.board_adapter).strip().lower() != "xdp"
        or str(experiment.response_channel).strip().upper() != "CH3"
    ):
        raise RuntimeError("Fixed-condition Safe SAC only supports the current 0x5E/page-0 XDP board and CH3 response.")
    if abs(float(config.targets.vout_target_v) - 0.9296875) > 0.01:
        raise RuntimeError("Fixed-condition Safe SAC only supports the current approximately 0.93 V Vout target.")
    if abs(float(config.targets.settling_time_s) - 2e-6) > 1e-12:
        raise RuntimeError("Fixed-condition Safe SAC requires the current 2 us settling target.")
    if float(config.targets.overshoot_pct) != 3.0 or float(config.targets.undershoot_pct) != 3.0:
        raise RuntimeError("Fixed-condition Safe SAC requires the current 3% OS/US limits.")
    if float(config.targets.phase_margin_deg) != 45.0 or float(config.targets.crossover_frequency_hz) != 200_000.0:
        raise RuntimeError("Fixed-condition Safe SAC requires PM >= 45 deg and fc <= 200 kHz targets.")
    fg = experiment.function_generator_config or {}
    frequency = float(fg.get("frequency_hz", 0.0) or 0.0)
    low_v = float(fg.get("low_v", fg.get("low_level", float("nan"))))
    high_v = float(fg.get("high_v", fg.get("high_level", float("nan"))))
    if (
        not all(math.isfinite(value) for value in (frequency, low_v, high_v))
        or abs(frequency - 10_000.0) > 1.0
        or abs(low_v - 0.1) > 1e-6
        or abs(high_v - 1.1) > 1e-6
    ):
        raise RuntimeError("Fixed-condition Safe SAC requires the 10 kHz, 0.1-1.1 V load step.")
    if str(fg.get("mode", "square")).strip().lower() != "square":
        raise RuntimeError("Fixed-condition Safe SAC requires square-wave load excitation.")
    bode = experiment.bode_config or {}
    bode_values = (
        float(bode.get("start_hz", 0.0) or 0.0),
        float(bode.get("stop_hz", 0.0) or 0.0),
        float(bode.get("bandwidth_hz", 0.0) or 0.0),
        float(bode.get("source_vpp", 0.0) or 0.0),
    )
    if (
        not all(math.isfinite(value) for value in bode_values)
        or abs(bode_values[0] - 1_000.0) > 1.0
        or abs(bode_values[1] - 1_000_000.0) > 1.0
        or int(bode.get("points", 0) or 0) != 201
        or abs(bode_values[2] - 300.0) > 1.0
        or abs(bode_values[3] - 0.1) > 1e-6
    ):
        raise RuntimeError("Fixed-condition Safe SAC requires the current 1 kHz-1 MHz, 201-point Bode setup.")
    cm_gain = config.search.mod0_cm_gain
    if (
        abs(float(cm_gain.min)) > 1e-9
        or abs(float(cm_gain.max) - 9.0) > 1e-9
        or not 0.0 <= float(cm_gain.center) <= 9.0
    ):
        raise RuntimeError("Fixed-condition Safe SAC requires mod0_cm_gain integer search range 0-9.")
    ll_bw = config.search.mod0_ll_bw
    if (
        abs(float(ll_bw.min) - 47.0) > 1e-9
        or abs(float(ll_bw.max) - 79.0) > 1e-9
        or not 47.0 <= float(ll_bw.center) <= 79.0
    ):
        raise RuntimeError("Fixed-condition Safe SAC requires the capped Loop-A LS/LR bandwidth range 47-79.")
    for name in ("mod0_kpole1", "mod0_kpole2"):
        parameter = getattr(config.search, name)
        if abs(float(parameter.min) - 2.0) > 1e-9 or abs(float(parameter.max) - 6.0) > 1e-9:
            raise RuntimeError("Fixed-condition Safe SAC quantizes each kpole field independently to {2, 3, 4, 5, 6}.")


def _artifact_context(signature: dict[str, Any]) -> dict[str, Any]:
    analyzer_path = Path(__file__).resolve().parents[1] / "analyzer.py"
    return {
        "operating_signature": signature,
        "analyzer_schema_version": 1,
        "analyzer_sha256": _sha256(analyzer_path),
    }


def _episode_has_confirmation(records: list[dict[str, Any]], required: int) -> bool:
    last_key: tuple[Any, ...] | None = None
    streak = 0
    for record in records:
        metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
        candidate = record.get("candidate") if isinstance(record.get("candidate"), dict) else {}
        key = tuple(candidate.get(name) for name in (
            "mod0_kp",
            "mod0_ki",
            "mod0_kd",
            "mod0_kpole1",
            "mod0_kpole2",
            "mod0_cm_gain",
            "mod0_ll_bw",
            "output_inductance_nh",
            "effective_lc_inductance_nh",
        ))
        if bool(metrics.get("passed")) and key == last_key:
            streak += 1
        elif bool(metrics.get("passed")):
            last_key = key
            streak = 1
        else:
            last_key = None
            streak = 0
        if streak >= required:
            return True
    return False


def _candidate_from_payload(payload: dict[str, Any]) -> Any:
    from .common import candidate_from_mapping

    return candidate_from_mapping(payload, phase=str(payload.get("phase") or "drl_start"))


def _directory_hashes(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(directory.iterdir() if directory.exists() else []):
        if path.is_file() and path.name != "manifest.json":
            result[path.name] = _sha256(path)
    return result


def _safe_artifact_name(value: Any, label: str) -> str:
    name = str(value or "").strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise RuntimeError(f"Safe SAC is fail-closed because its {label} ID is missing or invalid.")
    return name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default
