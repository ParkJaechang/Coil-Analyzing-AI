from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from coil_ai_sweep.schema import SweepSegmentManifestRow
from coil_ai_sweep.segment_alignment import SegmentAlignmentConfig, build_aligned_segment_residual
from coil_ai_sweep.segment_parser import SegmentMeasurement
from coil_ai_sweep.training_packet import (
    build_shape_metrics,
    build_segment_training_packet,
)


def _row(*, variant_params_json: str = '{"corner_radius": 0.04}') -> SweepSegmentManifestRow:
    return SweepSegmentManifestRow(
        batch_id="batch-a",
        segment_id="S0001",
        start_sample=0,
        end_sample=1000,
        active_start_sample=0,
        active_end_sample=1000,
        start_time_s=0.0,
        end_time_s=0.5,
        active_start_time_s=0.0,
        active_end_time_s=0.5,
        freq_hz=2.0,
        cycle_count=1.0,
        target_peak_mT=20.0,
        target_shape="fixed_rounded_triangle",
        source_waveform_family="triangle",
        mode="finite",
        variant_type="baseline",
        variant_params_json=variant_params_json,
    )


def _rounded_triangle(local_time_s: np.ndarray, row: SweepSegmentManifestRow) -> np.ndarray:
    phase = (local_time_s * float(row.freq_hz)) % 1.0
    radius = 0.04
    shifted = []
    for offset in (0.0, -radius, radius):
        shifted_phase = (phase + offset) % 1.0
        shifted.append(np.where(shifted_phase < 0.5, 4.0 * shifted_phase - 1.0, 3.0 - 4.0 * shifted_phase))
    return np.mean(shifted, axis=0) * float(row.target_peak_mT)


def _segment(row: SweepSegmentManifestRow, *, amplitude_scale: float = 1.0) -> SegmentMeasurement:
    time_s = np.linspace(0.0, 0.5, 1000)
    return SegmentMeasurement(
        segment_id=row.segment_id,
        batch_id=row.batch_id,
        frame=pd.DataFrame(
            {
                "active_local_time_s": time_s,
                "effective_field_mT": amplitude_scale * _rounded_triangle(time_s, row),
            }
        ),
        metadata={"source": "unit-test"},
    )


def test_build_shape_metrics_summarizes_evaluation_residuals() -> None:
    row = _row()
    alignment = build_aligned_segment_residual(
        _segment(row, amplitude_scale=0.8),
        row,
        config=SegmentAlignmentConfig(phase_sync_method="none"),
    )

    metrics = build_shape_metrics(alignment)

    evaluation = alignment.frame["evaluation_mask"]
    assert metrics.segment_id == "S0001"
    assert metrics.status == "ok"
    assert metrics.evaluation_sample_count == int(evaluation.sum())
    assert metrics.total_mae_mT > 1.0
    assert metrics.shape_mae_mT < metrics.total_mae_mT
    assert metrics.shape_norm_rmse < 0.1
    assert metrics.support_available_fraction == 1.0
    assert metrics.zero_fill_used is False
    assert metrics.hardware_invoked is False
    assert metrics.ui_involved is False
    assert metrics.ml_training_involved is False


def test_training_packet_is_json_serializable_and_keeps_alignment_columns() -> None:
    row = _row()
    alignment = build_aligned_segment_residual(
        _segment(row),
        row,
        config=SegmentAlignmentConfig(output_sample_count=32, phase_sync_method="none"),
    )

    packet = build_segment_training_packet(alignment, row)

    assert packet["schema_version"] == "shape_training_packet.v1"
    assert packet["segment"]["segment_id"] == "S0001"
    assert packet["target"]["freq_hz"] == 2.0
    assert packet["variant"]["params"] == {"corner_radius": 0.04}
    assert packet["metrics"]["total_rmse_mT"] < 0.25
    assert packet["metadata"]["hardware_invoked"] is False
    assert packet["metadata"]["streamlit_involved"] is False
    assert packet["metadata"]["ml_training_involved"] is False
    assert set(packet["frame_columns"]) == {
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
    }
    json.dumps(packet, allow_nan=False)


def test_shape_metrics_report_unavailable_values_as_none_for_missing_measurement() -> None:
    row = _row()
    segment = _segment(row)
    missing = SegmentMeasurement(
        segment_id=segment.segment_id,
        batch_id=segment.batch_id,
        frame=segment.frame.drop(columns=["effective_field_mT"]),
        metadata=segment.metadata,
    )
    alignment = build_aligned_segment_residual(missing, row)

    packet = build_segment_training_packet(alignment, row)

    assert packet["metrics"]["status"] == "measured_field_column_missing"
    assert packet["metrics"]["evaluation_sample_count"] == 0
    assert packet["metrics"]["total_mae_mT"] is None
    assert packet["metrics"]["shape_mae_mT"] is None
    json.dumps(packet, allow_nan=False)
