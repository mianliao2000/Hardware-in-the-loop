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
from .dataset import build_collection_plan, load_autotune_dataset, save_collection_plan
from .model import SurrogateEnsemble, dependency_status, require_ml_dependencies, train_surrogate_ensemble
from .policy import SafeSacTuner, train_safe_sac_policy, validation_start_candidates
from .tuner import PlannedCandidateTuner


COLLECTION_BUDGET = 240
VALIDATION_BUDGET = 60
VALIDATION_EPISODES = 4
VALIDATION_EPISODE_BUDGET = 15
VALIDATION_CONFIRMATIONS = 3


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
            return PlannedCandidateTuner(plan_path, history)
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
        required_starts = VALIDATION_EPISODES if is_validation else 1
        if len(starts) < required_starts:
            raise RuntimeError(
                f"Safe SAC model '{model_id}' is missing its {required_starts} persisted validation start(s)."
            )
        return SafeSacTuner(
            ensemble=ensemble,
            policy_path=model_dir / "safe_sac_policy.zip",
            config=config,
            history=history,
            validation_starts=starts,
            episode_budget=experiment.drl_episode_budget,
            confirmation_count=experiment.drl_confirmation_count,
            validation_episodes=VALIDATION_EPISODES if is_validation else 1,
            hardware_protection_mode=hardware_protection_mode,
            run_full_budget=bool(experiment.ignore_pass_until_max_iterations),
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            self._reconcile_hardware_locked()
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
                self._set_locked(state="collecting", busy=True, message="Resuming DRL hardware collection.")
                self._spawn_locked(self._resume_and_monitor, "collection", COLLECTION_BUDGET, None)
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
            if not signatures_compatible(self._state.get("operating_signature") or {}, signature):
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
            if not bool(model_manifest.get("ready")):
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
                message="Starting four guarded Safe SAC hardware episodes.",
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
            plan.update(_artifact_context(signature))
            plan["provisional_model_id"] = provisional_id
            for item in plan.get("candidates", []):
                if isinstance(item, dict):
                    metadata = item.setdefault("optimizer_metadata", {})
                    if isinstance(metadata, dict):
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
                    message="Collection plan passed model safety screening; starting hardware collection.",
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
                total_steps=_env_int("DRL_SAC_STEPS", 1_000_000),
                evaluation_episodes=_env_int("DRL_EVAL_EPISODES", 10_000),
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
        total = COLLECTION_BUDGET if self._state.get("workflow") == "collection" else VALIDATION_BUDGET
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
        or abs(low_v) > 1e-6
        or abs(high_v - 1.0) > 1e-6
    ):
        raise RuntimeError("Fixed-condition Safe SAC requires the 10 kHz, 0-1 V load step.")
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
    if any(abs(float(value) - 2.0) > 1e-9 for value in (cm_gain.center, cm_gain.min, cm_gain.max)):
        raise RuntimeError("Fixed-condition Safe SAC keeps mod0_cm_gain fixed at 2.")
    for name in ("mod0_kpole1", "mod0_kpole2"):
        parameter = getattr(config.search, name)
        if abs(float(parameter.min) - 3.0) > 1e-9 or abs(float(parameter.max) - 6.0) > 1e-9:
            raise RuntimeError("Fixed-condition Safe SAC quantizes both kpole fields to the configured {3, 6} set.")
    kpole_centers = [
        3 if abs(float(getattr(config.search, name).center) - 3) <= abs(float(getattr(config.search, name).center) - 6) else 6
        for name in ("mod0_kpole1", "mod0_kpole2")
    ]
    if kpole_centers[0] != kpole_centers[1]:
        raise RuntimeError("Fixed-condition Safe SAC requires a shared kpole baseline for both hardware fields.")


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
