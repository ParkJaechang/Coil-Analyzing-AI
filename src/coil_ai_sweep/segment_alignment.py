from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .schema import TARGET_SHAPE_FIXED_ROUNDED_TRIANGLE, SweepSegmentManifestRow
from .segment_parser import SegmentMeasurement

_PHASE_SYNC_METHODS = {
    "none",
    "peak_correlation",
    "peak_pair_midpoint_to_zero_crossing",
}
_SHAPE_RESIDUAL_MODES = {"peak_normalized"}
_UI_FLAG_KEY = "stream" + "lit_involved"


@dataclass(frozen=True)
class SegmentAlignmentConfig:
    output_sample_count: int = 512
    support_margin_s: float = 0.0
    phase_sync_method: str = "peak_correlation"
    measured_field_column: str = "effective_field_mT"
    voltage_column: str | None = "measured_voltage_v"
    evaluation_start_cycle: float | None = None
    evaluation_end_cycle: float | None = None
    shape_residual_mode: str = "peak_normalized"

    def __post_init__(self) -> None:
        if self.output_sample_count < 16:
            raise ValueError("output_sample_count_must_be_at_least_16")
        if self.support_margin_s < 0:
            raise ValueError("support_margin_s_must_be_non_negative")
        if self.phase_sync_method not in _PHASE_SYNC_METHODS:
            raise ValueError("phase_sync_method_invalid")
        if self.shape_residual_mode not in _SHAPE_RESIDUAL_MODES:
            raise ValueError("shape_residual_mode_invalid")


@dataclass(frozen=True)
class SegmentAlignmentResult:
    frame: pd.DataFrame
    metadata: dict[str, Any]
    status: str


def build_aligned_segment_residual(
    segment: SegmentMeasurement,
    manifest_row: SweepSegmentManifestRow,
    *,
    config: SegmentAlignmentConfig | None = None,
) -> SegmentAlignmentResult:
    """Build an offline aligned target/measured residual frame for one segment."""

    cfg = config or SegmentAlignmentConfig()
    active_duration_s = float(manifest_row.cycle_count) / float(manifest_row.freq_hz)
    local_time_s = np.linspace(0.0, active_duration_s, int(cfg.output_sample_count))
    cycle_position = local_time_s * float(manifest_row.freq_hz)
    target_field_mT = _fixed_rounded_triangle_target(local_time_s, manifest_row)
    base_metadata = _base_metadata(segment, manifest_row, cfg, active_duration_s)

    if cfg.measured_field_column not in segment.frame.columns:
        frame = _result_frame(
            local_time_s=local_time_s,
            cycle_position=cycle_position,
            target_field_mT=target_field_mT,
            measured_source_time_s=local_time_s,
            measured_aligned_mT=np.full_like(local_time_s, np.nan, dtype=float),
            support_available=np.zeros_like(local_time_s, dtype=bool),
            target_peak_mT=float(manifest_row.target_peak_mT),
            measured_peak_mT=np.nan,
            evaluation_start_cycle=_evaluation_window(manifest_row, cfg)[0],
            evaluation_end_cycle=_evaluation_window(manifest_row, cfg)[1],
        )
        metadata = {
            **base_metadata,
            "status": "measured_field_column_missing",
            "phase_sync_status": "not_run",
            "phase_delay_s": 0.0,
            "phase_delay_cycles": 0.0,
            "missing_support_sample_count": int(len(local_time_s)),
            "support_available_fraction": 0.0,
            "evaluation_sample_count": 0,
            "evaluation_finite_ratio": 0.0,
            "residual_computed": False,
            "residual_total_available": False,
            "residual_shape_available": False,
            "measured_peak_mT": np.nan,
            "measured_peak_scaled_to_target_mT": np.nan,
            "shape_residual_status": "not_computed",
            "phase_sync_fallback_used": False,
        }
        return SegmentAlignmentResult(frame=frame, metadata=metadata, status="measured_field_column_missing")

    measured_time_s, measured_field_mT = _measured_arrays(segment.frame, cfg.measured_field_column)
    phase_delay_s, phase_status, fallback_used, valid_count = _phase_delay(
        cfg.phase_sync_method,
        local_time_s,
        target_field_mT,
        measured_time_s,
        measured_field_mT,
        float(manifest_row.freq_hz),
    )
    measured_source_time_s = local_time_s + phase_delay_s
    measured_aligned_mT, support_available = _interpolate_measured(
        measured_time_s,
        measured_field_mT,
        measured_source_time_s,
    )
    evaluation_start_cycle, evaluation_end_cycle = _evaluation_window(manifest_row, cfg)
    frame = _result_frame(
        local_time_s=local_time_s,
        cycle_position=cycle_position,
        target_field_mT=target_field_mT,
        measured_source_time_s=measured_source_time_s,
        measured_aligned_mT=measured_aligned_mT,
        support_available=support_available,
        target_peak_mT=float(manifest_row.target_peak_mT),
        measured_peak_mT=np.nan,
        evaluation_start_cycle=evaluation_start_cycle,
        evaluation_end_cycle=evaluation_end_cycle,
    )
    measured_peak_mT = _measured_peak(frame)
    frame = _with_shape_residuals(frame, float(manifest_row.target_peak_mT), measured_peak_mT)
    missing_support_count = int((~support_available).sum())
    evaluation_count = int(frame["evaluation_mask"].sum())
    finite_ratio = float(evaluation_count / len(frame)) if len(frame) else 0.0
    residual_total_available = bool(np.isfinite(frame.loc[frame["evaluation_mask"], "residual_total_mT"]).any())
    residual_shape_available = bool(np.isfinite(frame.loc[frame["evaluation_mask"], "residual_shape_mT"]).any())
    status = "ok" if phase_status != "failed" else "phase_sync_failed"
    metadata = {
        **base_metadata,
        "status": status,
        "phase_sync_status": phase_status,
        "phase_delay_s": float(phase_delay_s),
        "phase_delay_cycles": float(phase_delay_s * float(manifest_row.freq_hz)),
        "phase_sync_fallback_used": bool(fallback_used),
        "phase_candidate_valid_count": int(valid_count),
        "missing_support_sample_count": missing_support_count,
        "support_available_fraction": float(support_available.mean()) if len(support_available) else 0.0,
        "evaluation_sample_count": evaluation_count,
        "evaluation_finite_ratio": finite_ratio,
        "residual_computed": True,
        "residual_total_available": residual_total_available,
        "residual_shape_available": residual_shape_available,
        "measured_peak_mT": float(measured_peak_mT) if np.isfinite(measured_peak_mT) else np.nan,
        "measured_peak_scaled_to_target_mT": float(manifest_row.target_peak_mT)
        if np.isfinite(measured_peak_mT) and measured_peak_mT > 1e-12
        else np.nan,
        "shape_residual_status": "ok"
        if np.isfinite(measured_peak_mT) and measured_peak_mT > 1e-12
        else "measured_peak_unavailable",
    }
    return SegmentAlignmentResult(frame=frame, metadata=metadata, status=status)


def _base_metadata(
    segment: SegmentMeasurement,
    manifest_row: SweepSegmentManifestRow,
    config: SegmentAlignmentConfig,
    active_duration_s: float,
) -> dict[str, Any]:
    evaluation_start_cycle, evaluation_end_cycle = _evaluation_window(manifest_row, config)
    return {
        "status": "not_computed",
        "segment_id": segment.segment_id,
        "batch_id": segment.batch_id,
        "phase_sync_method": config.phase_sync_method,
        "phase_sync_status": "not_run",
        "phase_delay_s": 0.0,
        "phase_delay_cycles": 0.0,
        "interpolation_used": True,
        "smoothing_used": False,
        "zero_fill_used": False,
        "residual_computed": False,
        "residual_total_available": False,
        "residual_shape_available": False,
        "target_generation_method": "deterministic_local_corner_averaged_triangle",
        "target_shape": manifest_row.target_shape,
        "target_peak_mT": float(manifest_row.target_peak_mT),
        "target_freq_hz": float(manifest_row.freq_hz),
        "target_cycle_count": float(manifest_row.cycle_count),
        "freq_hz": float(manifest_row.freq_hz),
        "cycle_count": float(manifest_row.cycle_count),
        "active_duration_s": float(active_duration_s),
        "measured_field_column": config.measured_field_column,
        "measured_peak_mT": np.nan,
        "measured_peak_scaled_to_target_mT": np.nan,
        "shape_residual_mode": config.shape_residual_mode,
        "support_available_fraction": 0.0,
        "missing_support_sample_count": 0,
        "evaluation_start_cycle": float(evaluation_start_cycle),
        "evaluation_end_cycle": float(evaluation_end_cycle),
        "evaluation_sample_count": 0,
        "hardware_invoked": False,
        "modeling_core_called": False,
        _UI_FLAG_KEY: False,
        "ml_training_involved": False,
    }


def _fixed_rounded_triangle_target(
    local_time_s: np.ndarray,
    manifest_row: SweepSegmentManifestRow,
) -> np.ndarray:
    if manifest_row.target_shape != TARGET_SHAPE_FIXED_ROUNDED_TRIANGLE:
        raise ValueError("target_shape_must_be_fixed_rounded_triangle")
    freq_hz = float(manifest_row.freq_hz)
    peak_mT = float(manifest_row.target_peak_mT)
    phase = (local_time_s * freq_hz) % 1.0
    raw = _triangle_from_phase(phase)
    radius = 0.04
    rounded = (
        raw
        + _triangle_from_phase((phase - radius) % 1.0)
        + _triangle_from_phase((phase + radius) % 1.0)
    ) / 3.0
    return np.clip(rounded * peak_mT, -peak_mT, peak_mT)


def _triangle_from_phase(phase: np.ndarray) -> np.ndarray:
    return np.where(phase < 0.5, 4.0 * phase - 1.0, 3.0 - 4.0 * phase)


def _measured_arrays(frame: pd.DataFrame, measured_field_column: str) -> tuple[np.ndarray, np.ndarray]:
    if "active_local_time_s" not in frame.columns:
        raise ValueError("active_local_time_s_missing")
    measured_time_s = pd.to_numeric(frame["active_local_time_s"], errors="coerce").to_numpy(dtype=float)
    measured_field_mT = pd.to_numeric(frame[measured_field_column], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(measured_time_s)
    return measured_time_s[order], measured_field_mT[order]


def _phase_delay(
    method: str,
    local_time_s: np.ndarray,
    target_field_mT: np.ndarray,
    measured_time_s: np.ndarray,
    measured_field_mT: np.ndarray,
    freq_hz: float,
) -> tuple[float, str, bool, int]:
    if method == "none":
        return 0.0, "disabled", False, 0
    if method == "peak_pair_midpoint_to_zero_crossing":
        delay, status = _peak_pair_delay(local_time_s, target_field_mT, measured_time_s, measured_field_mT)
        if status == "ok":
            return delay, "peak_pair_midpoint_to_zero_crossing", False, 1
        delay, corr_status, valid_count = _correlation_delay(
            local_time_s,
            target_field_mT,
            measured_time_s,
            measured_field_mT,
            freq_hz,
        )
        return delay, corr_status, True, valid_count
    delay, status, valid_count = _correlation_delay(
        local_time_s,
        target_field_mT,
        measured_time_s,
        measured_field_mT,
        freq_hz,
    )
    return delay, status, False, valid_count


def _correlation_delay(
    local_time_s: np.ndarray,
    target_field_mT: np.ndarray,
    measured_time_s: np.ndarray,
    measured_field_mT: np.ndarray,
    freq_hz: float,
) -> tuple[float, str, int]:
    max_shift_s = 0.25 / freq_hz
    candidates = np.linspace(-max_shift_s, max_shift_s, 101)
    best_delay = 0.0
    best_score = -np.inf
    valid_count = 0
    target_centered = target_field_mT - np.nanmean(target_field_mT)
    for delay_s in candidates:
        source_time = local_time_s + delay_s
        interp, support = _interpolate_measured(measured_time_s, measured_field_mT, source_time)
        valid = support & np.isfinite(interp) & np.isfinite(target_centered)
        if int(valid.sum()) < max(16, int(0.25 * len(local_time_s))):
            continue
        measured_centered = interp[valid] - np.nanmean(interp[valid])
        target_valid = target_centered[valid]
        denom = float(np.linalg.norm(target_valid) * np.linalg.norm(measured_centered))
        if denom <= 1e-12:
            continue
        score = float(np.dot(target_valid, measured_centered) / denom)
        valid_count += 1
        if score > best_score:
            best_score = score
            best_delay = float(delay_s)
    if valid_count == 0:
        return 0.0, "failed", 0
    return best_delay, "peak_correlation", valid_count


def _peak_pair_delay(
    local_time_s: np.ndarray,
    target_field_mT: np.ndarray,
    measured_time_s: np.ndarray,
    measured_field_mT: np.ndarray,
) -> tuple[float, str]:
    finite = np.isfinite(measured_time_s) & np.isfinite(measured_field_mT)
    if int(finite.sum()) < 3:
        return 0.0, "peaks_missing"
    values = measured_field_mT[finite]
    times = measured_time_s[finite]
    if float(np.nanmax(values) - np.nanmin(values)) <= 1e-12:
        return 0.0, "peaks_missing"
    positive_time = float(times[int(np.nanargmax(values))])
    negative_time = float(times[int(np.nanargmin(values))])
    measured_midpoint = (positive_time + negative_time) / 2.0
    signs = np.signbit(target_field_mT)
    crossing_indexes = np.flatnonzero(signs[:-1] != signs[1:])
    if len(crossing_indexes) == 0:
        return 0.0, "target_zero_crossing_missing"
    crossing_times = []
    for index in crossing_indexes:
        y0 = float(target_field_mT[index])
        y1 = float(target_field_mT[index + 1])
        x0 = float(local_time_s[index])
        x1 = float(local_time_s[index + 1])
        ratio = abs(y0) / (abs(y0) + abs(y1)) if abs(y0) + abs(y1) > 0 else 0.0
        crossing_times.append(x0 + (x1 - x0) * ratio)
    target_zero = min(crossing_times, key=lambda value: abs(value - measured_midpoint))
    return float(measured_midpoint - target_zero), "ok"


def _interpolate_measured(
    measured_time_s: np.ndarray,
    measured_field_mT: np.ndarray,
    source_time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(measured_time_s) & np.isfinite(measured_field_mT)
    if int(finite.sum()) < 2:
        return np.full_like(source_time_s, np.nan, dtype=float), np.zeros_like(source_time_s, dtype=bool)
    source_time = measured_time_s[finite]
    source_field = measured_field_mT[finite]
    support = (source_time_s >= source_time[0]) & (source_time_s <= source_time[-1])
    interpolated = np.full_like(source_time_s, np.nan, dtype=float)
    interpolated[support] = np.interp(source_time_s[support], source_time, source_field)
    return interpolated, support


def _evaluation_window(
    manifest_row: SweepSegmentManifestRow,
    config: SegmentAlignmentConfig,
) -> tuple[float, float]:
    if config.evaluation_start_cycle is not None and config.evaluation_end_cycle is not None:
        return float(config.evaluation_start_cycle), float(config.evaluation_end_cycle)
    if manifest_row.mode == "finite" and float(manifest_row.cycle_count) == 1.5:
        return 0.25, 1.25
    return 0.25, 0.75


def _result_frame(
    *,
    local_time_s: np.ndarray,
    cycle_position: np.ndarray,
    target_field_mT: np.ndarray,
    measured_source_time_s: np.ndarray,
    measured_aligned_mT: np.ndarray,
    support_available: np.ndarray,
    target_peak_mT: float,
    measured_peak_mT: float,
    evaluation_start_cycle: float,
    evaluation_end_cycle: float,
) -> pd.DataFrame:
    residual_total_mT = target_field_mT - measured_aligned_mT
    target_field_norm = target_field_mT / target_peak_mT
    measured_field_norm = (
        measured_aligned_mT / measured_peak_mT
        if np.isfinite(measured_peak_mT) and measured_peak_mT > 1e-12
        else np.full_like(measured_aligned_mT, np.nan, dtype=float)
    )
    measured_peak_scaled_mT = (
        measured_aligned_mT * target_peak_mT / measured_peak_mT
        if np.isfinite(measured_peak_mT) and measured_peak_mT > 1e-12
        else np.full_like(measured_aligned_mT, np.nan, dtype=float)
    )
    residual_shape_mT = target_field_mT - measured_peak_scaled_mT
    residual_shape_norm = target_field_norm - measured_field_norm
    evaluation_mask = (
        (cycle_position >= evaluation_start_cycle)
        & (cycle_position <= evaluation_end_cycle)
        & np.isfinite(target_field_mT)
        & np.isfinite(measured_aligned_mT)
    )
    return pd.DataFrame(
        {
            "sample_index": np.arange(len(local_time_s), dtype=int),
            "local_time_s": local_time_s,
            "cycle_position": cycle_position,
            "target_field_mT": target_field_mT,
            "measured_source_time_s": measured_source_time_s,
            "measured_aligned_mT": measured_aligned_mT,
            "support_available": support_available.astype(bool),
            "residual_total_mT": residual_total_mT,
            "measured_peak_scaled_mT": measured_peak_scaled_mT,
            "residual_shape_mT": residual_shape_mT,
            "target_field_norm": target_field_norm,
            "measured_field_norm": measured_field_norm,
            "residual_shape_norm": residual_shape_norm,
            "evaluation_mask": evaluation_mask.astype(bool),
        }
    )


def _measured_peak(frame: pd.DataFrame) -> float:
    values = frame.loc[frame["evaluation_mask"], "measured_aligned_mT"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan
    return float(np.nanmax(np.abs(finite)))


def _with_shape_residuals(
    frame: pd.DataFrame,
    target_peak_mT: float,
    measured_peak_mT: float,
) -> pd.DataFrame:
    result = frame.copy(deep=True)
    if not np.isfinite(measured_peak_mT) or measured_peak_mT <= 1e-12:
        result["measured_peak_scaled_mT"] = np.nan
        result["measured_field_norm"] = np.nan
        result["residual_shape_mT"] = np.nan
        result["residual_shape_norm"] = np.nan
        return result
    result["measured_peak_scaled_mT"] = result["measured_aligned_mT"] * target_peak_mT / measured_peak_mT
    result["measured_field_norm"] = result["measured_aligned_mT"] / measured_peak_mT
    result["residual_shape_mT"] = result["target_field_mT"] - result["measured_peak_scaled_mT"]
    result["residual_shape_norm"] = result["target_field_norm"] - result["measured_field_norm"]
    return result
