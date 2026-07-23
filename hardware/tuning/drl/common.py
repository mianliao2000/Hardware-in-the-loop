"""Shared DRL feature, scoring, compatibility, and persistence helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from ..models import (
    LL_BANDWIDTH_MAX,
    LL_BANDWIDTH_MIN,
    AutotuneExperimentConfig,
    HardwarePidCandidate,
    SearchSpace,
    TuningConfig,
    TuningTargets,
    hardware_candidate_key,
)


SCHEMA_VERSION = 18
ACTION_FIELDS = (
    "mod0_kp",
    "mod0_ki",
    "mod0_kd",
    "mod0_kpole1",
    "mod0_kpole2",
    "mod0_cm_gain",
    "mod0_ll_bw",
    "output_inductance_nh",
    "effective_lc_inductance_nh",
)
METRIC_FIELDS = (
    "overshoot_pct",
    "undershoot_pct",
    "overshoot_settling_time_us",
    "undershoot_settling_time_us",
    "phase_margin_deg",
    "crossover_frequency_khz",
    "gain_margin_db",
    "bode_gain_shape_penalty",
)
KPOLE_VALUES = (2, 3, 4, 5, 6)
# Retained for explicit diversity scheduling only. The SAC action now carries
# kpole1 and kpole2 as two independent dimensions; it no longer encodes an
# arbitrary ordinal index into this table.
KPOLE_PAIRS = tuple((kpole1, kpole2) for kpole1 in KPOLE_VALUES for kpole2 in KPOLE_VALUES)

SETTLING_WEIGHT_PER_US = 10.0
MAX_PENALTY = 300.0
BODE_SHAPE_PENALTY_PASS_PROXY = 13.0
INVALID_REASON_TOKENS = (
    "protection skipped",
    "transient protection",
    "scope safety check failed",
    "invalid transient waveform",
    "invalid bode",
    "duplicate 0 db crossover",
    "second 0 db crossover",
)


def action_bounds(search: SearchSpace) -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(
        [
            search.mod0_kp.min,
            search.mod0_ki.min,
            search.mod0_kd.min,
            search.mod0_kpole1.min,
            search.mod0_kpole2.min,
            search.mod0_cm_gain.min,
            search.mod0_ll_bw.min,
            search.output_inductance_nh.min,
            search.effective_lc_inductance_nh.min,
        ],
        dtype=np.float64,
    )
    high = np.asarray(
        [
            search.mod0_kp.max,
            search.mod0_ki.max,
            search.mod0_kd.max,
            search.mod0_kpole1.max,
            search.mod0_kpole2.max,
            search.mod0_cm_gain.max,
            search.mod0_ll_bw.max,
            search.output_inductance_nh.max,
            search.effective_lc_inductance_nh.max,
        ],
        dtype=np.float64,
    )
    return low, high


def candidate_to_action(candidate: HardwarePidCandidate) -> np.ndarray:
    return np.asarray(
        [
            candidate.mod0_kp,
            candidate.mod0_ki,
            candidate.mod0_kd,
            candidate.mod0_kpole1,
            candidate.mod0_kpole2,
            candidate.mod0_cm_gain,
            candidate.mod0_ll_bw,
            candidate.output_inductance_nh,
            candidate.effective_lc_inductance_nh,
        ],
        dtype=np.float64,
    )


def candidate_to_normalized(
    candidate: HardwarePidCandidate,
    search: SearchSpace,
    *,
    clip: bool = True,
) -> np.ndarray:
    low, high = action_bounds(search)
    span = np.maximum(high - low, 1e-12)
    normalized = (candidate_to_action(candidate) - low) / span * 2.0 - 1.0
    return np.clip(normalized, -1.0, 1.0) if clip else normalized


def candidate_from_normalized(
    normalized: np.ndarray | list[float],
    search: SearchSpace,
    phase: str,
) -> HardwarePidCandidate:
    values = np.asarray(normalized, dtype=np.float64).reshape(len(ACTION_FIELDS))
    low, high = action_bounds(search)
    raw = low + (np.clip(values, -1.0, 1.0) + 1.0) * 0.5 * (high - low)
    return HardwarePidCandidate(
        mod0_kp=int(round(float(raw[0]))),
        mod0_ki=int(round(float(raw[1]))),
        mod0_kd=int(round(float(raw[2]))),
        mod0_kpole1=min(6, max(2, int(round(float(raw[3]))))),
        mod0_kpole2=min(6, max(2, int(round(float(raw[4]))))),
        mod0_cm_gain=min(9, max(0, int(round(float(raw[5]))))),
        mod0_ll_bw=min(LL_BANDWIDTH_MAX, max(LL_BANDWIDTH_MIN, int(round(float(raw[6]))))),
        output_inductance_nh=float(raw[7]),
        effective_lc_inductance_nh=float(raw[8]),
        phase=phase,
    )


def candidate_with_delta(
    base: HardwarePidCandidate,
    delta: np.ndarray | list[float],
    search: SearchSpace,
    phase: str,
    trust_fraction: float = 0.10,
) -> HardwarePidCandidate:
    base_normalized = candidate_to_normalized(base, search)
    bounded_delta = np.clip(np.asarray(delta, dtype=np.float64), -1.0, 1.0)
    # A normalized range has width two, so 0.2 is ten percent of the hardware span.
    next_normalized = np.clip(base_normalized + bounded_delta * (2.0 * trust_fraction), -1.0, 1.0)
    return candidate_from_normalized(next_normalized, search, phase)


def candidate_key(candidate: HardwarePidCandidate) -> tuple[Any, ...]:
    return hardware_candidate_key(candidate)


def candidate_from_mapping(value: Mapping[str, Any], phase: str | None = None) -> HardwarePidCandidate:
    kpole1 = int(value.get("mod0_kpole1", value.get("kpole", 3)))
    kpole2 = int(value.get("mod0_kpole2", kpole1))
    return HardwarePidCandidate(
        mod0_kp=int(value.get("mod0_kp", 165)),
        mod0_ki=int(value.get("mod0_ki", 220)),
        mod0_kd=int(value.get("mod0_kd", 175)),
        mod0_kpole1=kpole1,
        mod0_kpole2=kpole2,
        mod0_cm_gain=int(value.get("mod0_cm_gain", 2)),
        mod0_ll_bw=int(value.get("mod0_ll_bw", 66)),
        output_inductance_nh=float(value.get("output_inductance_nh", 100.024)),
        effective_lc_inductance_nh=float(value.get("effective_lc_inductance_nh", 369.276)),
        phase=str(phase or value.get("phase") or "drl"),
    )


def candidate_to_mapping(candidate: HardwarePidCandidate) -> dict[str, Any]:
    return {
        "mod0_kp": int(candidate.mod0_kp),
        "mod0_ki": int(candidate.mod0_ki),
        "mod0_kd": int(candidate.mod0_kd),
        "mod0_kpole1": int(candidate.mod0_kpole1),
        "mod0_kpole2": int(candidate.mod0_kpole2),
        "mod0_cm_gain": int(candidate.mod0_cm_gain),
        "mod0_ll_bw": int(candidate.mod0_ll_bw),
        "output_inductance_nh": float(candidate.output_inductance_nh),
        "effective_lc_inductance_nh": float(candidate.effective_lc_inductance_nh),
        "phase": candidate.phase,
    }


def metric_vector(metrics: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    raw_values = (
        metrics.get("overshoot_pct"),
        metrics.get("undershoot_pct"),
        _seconds_to_microseconds(metrics.get("overshoot_settling_time_s")),
        _seconds_to_microseconds(metrics.get("undershoot_settling_time_s")),
        metrics.get("phase_margin_deg"),
        _hertz_to_kilohertz(metrics.get("crossover_frequency_hz")),
        metrics.get("gain_margin_db"),
        metrics.get("bode_gain_shape_penalty"),
    )
    values = np.zeros(len(METRIC_FIELDS), dtype=np.float64)
    mask = np.zeros(len(METRIC_FIELDS), dtype=np.float64)
    for index, value in enumerate(raw_values):
        parsed = finite_float(value)
        if parsed is not None:
            values[index] = parsed
            mask[index] = 1.0
    return values, mask


def vector_to_metric_mapping(values: np.ndarray | list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(len(METRIC_FIELDS))
    return {name: float(array[index]) for index, name in enumerate(METRIC_FIELDS)}


def relabeled_score(metrics: Mapping[str, Any], targets: TuningTargets) -> tuple[float, bool]:
    metric_payload = metrics.get("metrics") if isinstance(metrics.get("metrics"), Mapping) else metrics
    # Surrogate internals use the canonical vector units (us and kHz), while
    # hardware records use seconds and Hz.  Do not run canonical mappings
    # through the hardware-unit converter a second time.
    canonical_core_fields = METRIC_FIELDS[:-1]
    if all(field in metric_payload for field in canonical_core_fields):
        values = np.asarray(
            [finite_float(metric_payload.get(field)) for field in METRIC_FIELDS],
            dtype=np.float64,
        )
        mask = np.isfinite(values).astype(np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        values, mask = metric_vector(metric_payload)
    if not bool(np.all(mask[:4])):
        return MAX_PENALTY, False
    mapping = vector_to_metric_mapping(values)
    score = 0.0
    score += max(0.0, mapping["overshoot_pct"] - targets.overshoot_pct)
    score += max(0.0, mapping["undershoot_pct"] - targets.undershoot_pct)
    target_us = targets.settling_time_s * 1e6
    score += SETTLING_WEIGHT_PER_US * max(0.0, mapping["overshoot_settling_time_us"] - target_us)
    score += SETTLING_WEIGHT_PER_US * max(0.0, mapping["undershoot_settling_time_us"] - target_us)

    phase_ok = False
    crossover_ok = False
    if mask[4] > 0:
        phase_error = max(0.0, targets.phase_margin_deg - mapping["phase_margin_deg"])
        score += phase_error * 1.5
        phase_ok = phase_error <= 0.0
    else:
        score += 100.0

    if mask[7] > 0:
        score += max(0.0, mapping["bode_gain_shape_penalty"])
    target_fc_khz = targets.crossover_frequency_hz / 1e3
    if mask[5] > 0 and mapping["crossover_frequency_khz"] > 0 and target_fc_khz > 0:
        crossover_error_pct = max(
            0.0,
            (mapping["crossover_frequency_khz"] - target_fc_khz) / target_fc_khz * 100.0,
        )
        score += crossover_error_pct * 0.5
        crossover_ok = crossover_error_pct <= 0.0
    else:
        score += 100.0

    transient_ok = (
        mapping["overshoot_pct"] <= targets.overshoot_pct
        and mapping["undershoot_pct"] <= targets.undershoot_pct
        and mapping["overshoot_settling_time_us"] <= target_us
        and mapping["undershoot_settling_time_us"] <= target_us
    )
    reasons = " ".join(
        str(item).lower() for item in (metric_payload.get("pass_reasons", []) or [])
    )
    shape_ok = "bode gain shape failed" not in reasons
    if mask[7] > 0:
        # The measured path has an exact validity reason based on rebound and
        # flat-span thresholds. Synthetic SAC rollouts only see the scalar
        # shape target, whose largest still-valid value is about 12.6 points.
        shape_ok = shape_ok and (
            mapping["bode_gain_shape_penalty"] <= BODE_SHAPE_PENALTY_PASS_PROXY
        )
    passed = transient_ok and phase_ok and crossover_ok and shape_ok
    if max(invalid_labels(metrics)) > 0:
        return MAX_PENALTY, False
    if passed:
        reward = 0.15 * _headroom(targets.overshoot_pct, mapping["overshoot_pct"])
        reward += 0.15 * _headroom(targets.undershoot_pct, mapping["undershoot_pct"])
        reward += SETTLING_WEIGHT_PER_US * max(0.0, target_us - mapping["overshoot_settling_time_us"])
        reward += SETTLING_WEIGHT_PER_US * max(0.0, target_us - mapping["undershoot_settling_time_us"])
        # Match ResponseAnalyzer._passed_reward: every passing PM has zero
        # phase error and therefore receives the configured tolerance reward.
        reward += max(0.0, targets.phase_margin_tolerance_deg) * 0.05
        crossover_headroom = max(0.0, (target_fc_khz - mapping["crossover_frequency_khz"]) / target_fc_khz * 100.0)
        reward += min(crossover_headroom, 100.0) / 100.0 * 0.25
        score -= reward
    return float(min(MAX_PENALTY, score)), bool(passed)


def invalid_labels(metrics_or_record: Mapping[str, Any]) -> tuple[int, int, int]:
    metrics = metrics_or_record.get("metrics") if isinstance(metrics_or_record.get("metrics"), Mapping) else metrics_or_record
    reasons = " ".join(str(item).lower() for item in metrics.get("pass_reasons", []) or [])
    score = finite_float(metrics.get("score"))
    record_scope = metrics_or_record.get("scope_result") if isinstance(metrics_or_record.get("scope_result"), Mapping) else {}
    record_bode = metrics_or_record.get("bode_result") if isinstance(metrics_or_record.get("bode_result"), Mapping) else {}
    protection = int(
        "protection" in reasons
        or "scope safety check failed" in reasons
        or bool(record_scope.get("skipped") and "protection" in str(record_scope.get("reason", "")).lower())
    )
    invalid_transient = int("invalid transient waveform" in reasons or protection > 0)
    invalid_bode = int(
        "invalid bode" in reasons
        or "duplicate 0 db crossover" in reasons
        or "second 0 db crossover" in reasons
        or bool(record_bode.get("skipped") and "protection" not in str(record_bode.get("reason", "")).lower())
    )
    if score is not None and score >= 1e6:
        invalid_transient = 1
    return protection, invalid_transient, invalid_bode


def operating_signature(config: TuningConfig, experiment: AutotuneExperimentConfig) -> dict[str, Any]:
    fg = experiment.function_generator_config or {}
    bode = experiment.bode_config or {}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "board_address": experiment.board_address,
        "board_page": int(experiment.board_page),
        "board_adapter": experiment.board_adapter,
        "response_channel": experiment.response_channel,
        # One PMBus VOUT LSB can produce 0.9296875 vs 0.9297 in saved runs.
        "vout_target_v": round(float(config.targets.vout_target_v), 4),
        "overshoot_pct": float(config.targets.overshoot_pct),
        "undershoot_pct": float(config.targets.undershoot_pct),
        "settling_time_s": float(config.targets.settling_time_s),
        "phase_margin_deg": float(config.targets.phase_margin_deg),
        "crossover_frequency_hz": float(config.targets.crossover_frequency_hz),
        "function_generator": {
            "frequency_hz": finite_float(fg.get("frequency_hz")),
            "low_v": finite_float(fg.get("low_v", fg.get("low_level"))),
            "high_v": finite_float(fg.get("high_v", fg.get("high_level"))),
            "mode": str(fg.get("mode", "square")).lower(),
        },
        "bode": {
            "start_hz": finite_float(bode.get("start_hz")),
            "stop_hz": finite_float(bode.get("stop_hz")),
            "points": int(bode.get("points", 0) or 0),
            "bandwidth_hz": finite_float(bode.get("bandwidth_hz")),
            "source_vpp": finite_float(bode.get("source_vpp")),
        },
        "search": {
            field: {
                "min": float(getattr(config.search, field).min),
                "max": float(getattr(config.search, field).max),
            }
            for field in (
                "mod0_kp",
                "mod0_ki",
                "mod0_kd",
                "mod0_kpole1",
                "mod0_kpole2",
                "mod0_cm_gain",
                "mod0_ll_bw",
                "output_inductance_nh",
                "effective_lc_inductance_nh",
            )
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload["signature"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return payload


def signatures_compatible(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    return bool(expected.get("signature")) and expected.get("signature") == actual.get("signature")


def artifact_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}_{os.getpid()}"


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except Exception:
        return None


def _seconds_to_microseconds(value: Any) -> float | None:
    parsed = finite_float(value)
    return None if parsed is None else parsed * 1e6


def _hertz_to_kilohertz(value: Any) -> float | None:
    parsed = finite_float(value)
    return None if parsed is None else parsed / 1e3


def _headroom(limit: float, value: float) -> float:
    if limit <= 0:
        return 0.0
    return max(0.0, min(1.0, (limit - value) / limit))
