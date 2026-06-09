from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from .schema import SweepSegmentManifestRow
from .segment_alignment import SegmentAlignmentResult

_SCHEMA_VERSION = "shape_training_packet.v1"
_UI_FLAG_KEY = "stream" + "lit_involved"
_FRAME_COLUMNS = [
    "sample_index",
    "local_time_s",
    "cycle_position",
    "target_field_mT",
    "measured_aligned_mT",
    "support_available",
    "residual_total_mT",
    "measured_peak_scaled_mT",
    "residual_shape_mT",
    "target_field_norm",
    "measured_field_norm",
    "residual_shape_norm",
    "evaluation_mask",
]


@dataclass(frozen=True)
class SegmentShapeMetrics:
    """Scalar residual summary for one aligned AI sweep segment."""

    batch_id: str | None
    segment_id: str | None
    status: str
    phase_sync_status: str | None
    evaluation_sample_count: int
    evaluation_finite_ratio: float
    support_available_fraction: float
    missing_support_sample_count: int
    target_peak_mT: float | None
    measured_peak_mT: float | None
    phase_delay_s: float
    total_mae_mT: float | None
    total_rmse_mT: float | None
    total_max_abs_mT: float | None
    shape_mae_mT: float | None
    shape_rmse_mT: float | None
    shape_max_abs_mT: float | None
    shape_norm_mae: float | None
    shape_norm_rmse: float | None
    shape_norm_max_abs: float | None
    zero_fill_used: bool
    hardware_invoked: bool
    modeling_core_called: bool
    ui_involved: bool
    ml_training_involved: bool

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def build_shape_metrics(alignment: SegmentAlignmentResult) -> SegmentShapeMetrics:
    """Summarize total and peak-normalized shape residuals over the evaluation mask."""

    metadata = dict(alignment.metadata)
    frame = alignment.frame
    evaluation = _evaluation_mask(frame)
    total = _residual_stats(frame, evaluation, "residual_total_mT")
    shape = _residual_stats(frame, evaluation, "residual_shape_mT")
    shape_norm = _residual_stats(frame, evaluation, "residual_shape_norm")
    return SegmentShapeMetrics(
        batch_id=_optional_str(metadata.get("batch_id")),
        segment_id=_optional_str(metadata.get("segment_id")),
        status=str(metadata.get("status", alignment.status)),
        phase_sync_status=_optional_str(metadata.get("phase_sync_status")),
        evaluation_sample_count=int(metadata.get("evaluation_sample_count", int(evaluation.sum()))),
        evaluation_finite_ratio=float(metadata.get("evaluation_finite_ratio", _finite_ratio(frame, evaluation))),
        support_available_fraction=float(metadata.get("support_available_fraction", _support_fraction(frame))),
        missing_support_sample_count=int(metadata.get("missing_support_sample_count", _missing_support_count(frame))),
        target_peak_mT=_finite_or_none(metadata.get("target_peak_mT")),
        measured_peak_mT=_finite_or_none(metadata.get("measured_peak_mT")),
        phase_delay_s=float(metadata.get("phase_delay_s", 0.0)),
        total_mae_mT=total["mae"],
        total_rmse_mT=total["rmse"],
        total_max_abs_mT=total["max_abs"],
        shape_mae_mT=shape["mae"],
        shape_rmse_mT=shape["rmse"],
        shape_max_abs_mT=shape["max_abs"],
        shape_norm_mae=shape_norm["mae"],
        shape_norm_rmse=shape_norm["rmse"],
        shape_norm_max_abs=shape_norm["max_abs"],
        zero_fill_used=bool(metadata.get("zero_fill_used", False)),
        hardware_invoked=bool(metadata.get("hardware_invoked", False)),
        modeling_core_called=bool(metadata.get("modeling_core_called", False)),
        ui_involved=bool(metadata.get(_UI_FLAG_KEY, False)),
        ml_training_involved=bool(metadata.get("ml_training_involved", False)),
    )


def build_segment_training_packet(
    alignment: SegmentAlignmentResult,
    manifest_row: SweepSegmentManifestRow,
) -> dict[str, Any]:
    """Build an in-memory JSON-safe training packet from one aligned segment."""

    metrics = build_shape_metrics(alignment)
    frame = alignment.frame.copy(deep=True)
    columns = [column for column in _FRAME_COLUMNS if column in frame.columns]
    packet = {
        "schema_version": _SCHEMA_VERSION,
        "segment": {
            "batch_id": manifest_row.batch_id,
            "segment_id": manifest_row.segment_id,
            "status": alignment.status,
        },
        "target": {
            "freq_hz": float(manifest_row.freq_hz),
            "cycle_count": float(manifest_row.cycle_count),
            "target_peak_mT": float(manifest_row.target_peak_mT),
            "target_shape": manifest_row.target_shape,
            "source_waveform_family": manifest_row.source_waveform_family,
            "mode": manifest_row.mode,
        },
        "variant": {
            "type": manifest_row.variant_type,
            "params": json.loads(manifest_row.variant_params_json),
        },
        "metrics": metrics.to_dict(),
        "metadata": _packet_metadata(alignment.metadata),
        "frame_columns": columns,
        "samples": _records(frame, columns),
    }
    return _json_safe(packet)


def _evaluation_mask(frame: pd.DataFrame) -> np.ndarray:
    if "evaluation_mask" not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    return frame["evaluation_mask"].astype(bool).to_numpy(dtype=bool)


def _residual_stats(frame: pd.DataFrame, evaluation: np.ndarray, column: str) -> dict[str, float | None]:
    if column not in frame.columns:
        return {"mae": None, "rmse": None, "max_abs": None}
    values = pd.to_numeric(frame.loc[evaluation, column], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"mae": None, "rmse": None, "max_abs": None}
    abs_values = np.abs(finite)
    return {
        "mae": float(np.mean(abs_values)),
        "rmse": float(np.sqrt(np.mean(np.square(finite)))),
        "max_abs": float(np.max(abs_values)),
    }


def _finite_ratio(frame: pd.DataFrame, evaluation: np.ndarray) -> float:
    if "residual_total_mT" not in frame.columns or len(frame) == 0:
        return 0.0
    values = pd.to_numeric(frame.loc[evaluation, "residual_total_mT"], errors="coerce").to_numpy(dtype=float)
    return float(np.isfinite(values).sum() / len(frame))


def _support_fraction(frame: pd.DataFrame) -> float:
    if "support_available" not in frame.columns or len(frame) == 0:
        return 0.0
    return float(frame["support_available"].astype(bool).mean())


def _missing_support_count(frame: pd.DataFrame) -> int:
    if "support_available" not in frame.columns:
        return len(frame)
    return int((~frame["support_available"].astype(bool)).sum())


def _packet_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return _json_safe(
        {
            "phase_sync_method": metadata.get("phase_sync_method"),
            "phase_sync_status": metadata.get("phase_sync_status"),
            "phase_delay_s": metadata.get("phase_delay_s", 0.0),
            "phase_delay_cycles": metadata.get("phase_delay_cycles", 0.0),
            "interpolation_used": metadata.get("interpolation_used", False),
            "smoothing_used": metadata.get("smoothing_used", False),
            "zero_fill_used": metadata.get("zero_fill_used", False),
            "hardware_invoked": metadata.get("hardware_invoked", False),
            "modeling_core_called": metadata.get("modeling_core_called", False),
            _UI_FLAG_KEY: metadata.get(_UI_FLAG_KEY, False),
            "ml_training_involved": metadata.get("ml_training_involved", False),
        }
    )


def _records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if not columns:
        return []
    records = frame.loc[:, columns].to_dict(orient="records")
    return [_json_safe(record) for record in records]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
