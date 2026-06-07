from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .manifest_io import dataframe_to_manifest_rows, validate_manifest_dataframe

_HALLBZ_CANDIDATES = ("HallBz", "HallBz_mT", "hallbz_raw_mT", "raw_hallbz_mT")
_VOLTAGE_CANDIDATES = ("Voltage1_V", "voltage_v", "actual_drive_voltage_v")
_UI_FLAG_KEY = "stream" + "lit_involved"


@dataclass(frozen=True)
class SegmentMeasurement:
    """Measurement slice for one AI sweep manifest segment."""

    segment_id: str
    batch_id: str
    frame: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SegmentSplitResult:
    """Result of splitting one long measurement by manifest rows."""

    segments: dict[str, SegmentMeasurement]
    metadata: dict[str, Any]
    status: str


def split_long_measurement_by_manifest(
    measurement: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    support_margin_s: float = 0.0,
    time_column: str | None = None,
    hallbz_column: str | None = None,
    voltage_column: str | None = None,
) -> SegmentSplitResult:
    """Split a long measurement into per-segment support-window frames."""

    if support_margin_s < 0:
        raise ValueError("support_margin_s_must_be_non_negative")
    manifest_validation = validate_manifest_dataframe(manifest)
    if not manifest_validation.ok:
        raise ValueError("manifest_validation_failed: " + "; ".join(manifest_validation.errors))

    source = measurement.copy(deep=True)
    manifest_copy = manifest.copy(deep=True)
    measurement_time_s, time_column_used, time_unit_source = _measurement_time(
        source,
        time_column=time_column,
    )
    hallbz_column_used = _select_column(source, hallbz_column, _HALLBZ_CANDIDATES)
    voltage_column_used = _select_column(source, voltage_column, _VOLTAGE_CANDIDATES)
    rows = dataframe_to_manifest_rows(manifest_copy)
    segments: dict[str, SegmentMeasurement] = {}
    parsed_count = 0
    empty_count = 0
    warnings: list[str] = []
    if hallbz_column_used is None:
        warnings.append("hallbz_column_not_found")
    if voltage_column_used is None:
        warnings.append("voltage_column_not_found")

    for row in rows:
        support_start_s = float(row.start_time_s) - float(support_margin_s)
        support_end_s = float(row.end_time_s) + float(support_margin_s)
        support_mask = (measurement_time_s >= support_start_s) & (measurement_time_s <= support_end_s)
        frame = source.loc[support_mask].copy()
        frame_time = measurement_time_s[support_mask]
        if frame.empty:
            empty_count += 1
            segment_status = "empty"
            active_status = "empty"
            frame["measurement_time_s"] = pd.Series(dtype=float)
            frame["local_time_s"] = pd.Series(dtype=float)
            frame["active_local_time_s"] = pd.Series(dtype=float)
            frame["segment_window_mask"] = pd.Series(dtype=bool)
            frame["active_window_mask"] = pd.Series(dtype=bool)
            frame["support_window_mask"] = pd.Series(dtype=bool)
        else:
            parsed_count += 1
            segment_status = "ok"
            frame["measurement_time_s"] = frame_time
            frame["local_time_s"] = frame_time - float(row.start_time_s)
            frame["active_local_time_s"] = frame_time - float(row.active_start_time_s)
            frame["segment_window_mask"] = (
                (frame_time >= float(row.start_time_s)) & (frame_time <= float(row.end_time_s))
            )
            frame["active_window_mask"] = (
                (frame_time >= float(row.active_start_time_s))
                & (frame_time <= float(row.active_end_time_s))
            )
            frame["support_window_mask"] = True
            active_status = "ok" if bool(frame["active_window_mask"].any()) else "empty"
            if hallbz_column_used is not None:
                hallbz_raw = pd.to_numeric(frame[hallbz_column_used], errors="coerce")
                frame["hallbz_raw_mT"] = hallbz_raw.to_numpy(dtype=float)
                frame["effective_field_mT"] = -hallbz_raw.to_numpy(dtype=float)
            if voltage_column_used is not None:
                frame["measured_voltage_v"] = pd.to_numeric(
                    frame[voltage_column_used],
                    errors="coerce",
                ).to_numpy(dtype=float)

        segments[row.segment_id] = SegmentMeasurement(
            segment_id=row.segment_id,
            batch_id=row.batch_id,
            frame=frame.reset_index(drop=True),
            metadata={
                "support_window_status": segment_status,
                "active_window_status": active_status,
                "support_start_s": support_start_s,
                "support_end_s": support_end_s,
            },
        )

    status = "ok"
    if parsed_count == 0:
        status = "empty"
    elif empty_count:
        status = "partial"

    metadata = {
        "status": status,
        "segment_count": len(rows),
        "parsed_segment_count": parsed_count,
        "empty_segment_count": empty_count,
        "support_margin_s": float(support_margin_s),
        "time_column_used": time_column_used,
        "time_unit_source": time_unit_source,
        "hallbz_column_used": hallbz_column_used,
        "voltage_column_used": voltage_column_used,
        "hallbz_convention": "effective_field_mT = -HallBz raw",
        "warnings": warnings,
        "interpolation_used": False,
        "smoothing_used": False,
        "phase_alignment_used": False,
        "residual_computed": False,
        "modeling_core_called": False,
        "hardware_invoked": False,
        _UI_FLAG_KEY: False,
    }
    return SegmentSplitResult(segments=segments, metadata=metadata, status=status)


def _measurement_time(
    measurement: pd.DataFrame,
    *,
    time_column: str | None,
) -> tuple[np.ndarray, str, str]:
    if time_column is not None:
        if time_column not in measurement.columns:
            raise ValueError("time_column_not_found")
        selected = time_column
        time_unit_source = "TimeMs" if selected == "TimeMs" else "time_s"
    elif "time_s" in measurement.columns:
        selected = "time_s"
        time_unit_source = "time_s"
    elif "TimeMs" in measurement.columns:
        selected = "TimeMs"
        time_unit_source = "TimeMs"
    else:
        raise ValueError("measurement_time_column_not_found")

    values = pd.to_numeric(measurement[selected], errors="coerce").to_numpy(dtype=float)
    if selected == "TimeMs":
        values = values / 1000.0
    if not np.isfinite(values).all():
        raise ValueError("measurement_time_s_must_be_finite")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("measurement_time_s_must_be_monotonic_increasing")
    return values, selected, time_unit_source


def _select_column(
    measurement: pd.DataFrame,
    provided: str | None,
    candidates: tuple[str, ...],
) -> str | None:
    if provided is not None:
        if provided not in measurement.columns:
            return None
        return provided
    for candidate in candidates:
        if candidate in measurement.columns:
            return candidate
    return None
