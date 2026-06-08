from __future__ import annotations

from pathlib import Path
import inspect
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from coil_ai_sweep.schema import SweepSegmentManifestRow
from coil_ai_sweep.segment_parser import SegmentMeasurement
from coil_ai_sweep.segment_alignment import (
    SegmentAlignmentConfig,
    build_aligned_segment_residual,
)


REQUIRED_COLUMNS = {
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


def _row(*, cycle_count: float = 1.0, freq_hz: float = 2.0) -> SweepSegmentManifestRow:
    active_duration_s = cycle_count / freq_hz
    return SweepSegmentManifestRow(
        batch_id="batch-a",
        segment_id="S0001",
        start_sample=0,
        end_sample=1000,
        active_start_sample=0,
        active_end_sample=1000,
        start_time_s=0.0,
        end_time_s=active_duration_s,
        active_start_time_s=0.0,
        active_end_time_s=active_duration_s,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
        target_peak_mT=20.0,
        target_shape="fixed_rounded_triangle",
        source_waveform_family="triangle",
        mode="finite",
        variant_type="baseline",
        variant_params_json="{}",
    )


def _rounded_triangle(local_time_s: np.ndarray, row: SweepSegmentManifestRow) -> np.ndarray:
    phase = (local_time_s * float(row.freq_hz)) % 1.0
    raw = np.where(phase < 0.5, 4.0 * phase - 1.0, 3.0 - 4.0 * phase)
    radius = 0.04
    samples = [raw]
    for offset in (-radius, radius):
        shifted_phase = ((local_time_s * float(row.freq_hz)) + offset) % 1.0
        samples.append(np.where(shifted_phase < 0.5, 4.0 * shifted_phase - 1.0, 3.0 - 4.0 * shifted_phase))
    return np.mean(samples, axis=0) * float(row.target_peak_mT)


def _segment(
    row: SweepSegmentManifestRow,
    *,
    delay_s: float = 0.0,
    amplitude_scale: float = 1.0,
    sample_count: int = 2000,
    support_start_s: float = 0.0,
    support_end_s: float | None = None,
    flat: bool = False,
) -> SegmentMeasurement:
    active_duration_s = float(row.cycle_count) / float(row.freq_hz)
    if support_end_s is None:
        support_end_s = active_duration_s
    time_s = np.linspace(support_start_s, support_end_s, sample_count)
    if flat:
        measured = np.zeros_like(time_s)
    else:
        measured = amplitude_scale * _rounded_triangle(time_s - delay_s, row)
    frame = pd.DataFrame(
        {
            "active_local_time_s": time_s,
            "effective_field_mT": measured,
            "measured_voltage_v": np.zeros_like(time_s),
        }
    )
    return SegmentMeasurement(
        segment_id=row.segment_id,
        batch_id=row.batch_id,
        frame=frame,
        metadata={"test_segment": True},
    )


def test_build_aligned_segment_residual_without_phase_shift() -> None:
    row = _row()
    segment = _segment(row)

    result = build_aligned_segment_residual(
        segment,
        row,
        config=SegmentAlignmentConfig(phase_sync_method="none"),
    )

    assert result.status == "ok"
    assert np.nanmax(np.abs(result.frame["residual_total_mT"])) < 0.25


def test_build_aligned_segment_residual_estimates_phase_delay() -> None:
    row = _row()
    delay_s = 0.035
    segment = _segment(row, delay_s=delay_s, support_start_s=0.0, support_end_s=0.6)

    aligned = build_aligned_segment_residual(
        segment,
        row,
        config=SegmentAlignmentConfig(phase_sync_method="peak_correlation"),
    )
    unaligned = build_aligned_segment_residual(
        segment,
        row,
        config=SegmentAlignmentConfig(phase_sync_method="none"),
    )

    assert abs(aligned.metadata["phase_delay_s"] - delay_s) < 0.01
    assert np.nanmean(np.abs(aligned.frame["residual_total_mT"])) < np.nanmean(
        np.abs(unaligned.frame["residual_total_mT"])
    )


def test_shape_residual_is_peak_normalized() -> None:
    row = _row()
    segment = _segment(row, amplitude_scale=0.5)

    result = build_aligned_segment_residual(
        segment,
        row,
        config=SegmentAlignmentConfig(phase_sync_method="none"),
    )

    total = np.nanmean(np.abs(result.frame.loc[result.frame["evaluation_mask"], "residual_total_mT"]))
    shape = np.nanmean(np.abs(result.frame.loc[result.frame["evaluation_mask"], "residual_shape_mT"]))
    assert total > 4.0
    assert shape < total * 0.25


def test_evaluation_window_for_1p0_and_1p5() -> None:
    one_cycle_row = _row(cycle_count=1.0)
    one_half_row = _row(cycle_count=1.5)
    one_cycle = build_aligned_segment_residual(_segment(one_cycle_row), one_cycle_row)
    one_half = build_aligned_segment_residual(_segment(one_half_row), one_half_row)

    assert one_cycle.metadata["evaluation_start_cycle"] == 0.25
    assert one_cycle.metadata["evaluation_end_cycle"] == 0.75
    assert one_half.metadata["evaluation_start_cycle"] == 0.25
    assert one_half.metadata["evaluation_end_cycle"] == 1.25


def test_missing_support_is_nan_not_zero_filled() -> None:
    row = _row()
    segment = _segment(row, sample_count=400, support_start_s=0.1, support_end_s=0.4)

    result = build_aligned_segment_residual(
        segment,
        row,
        config=SegmentAlignmentConfig(phase_sync_method="none"),
    )

    missing = ~result.frame["support_available"]
    assert bool(missing.any())
    assert result.frame.loc[missing, "measured_aligned_mT"].isna().all()
    assert result.metadata["zero_fill_used"] is False


def test_missing_measured_field_column_fails() -> None:
    row = _row()
    segment = _segment(row)
    segment = SegmentMeasurement(
        segment_id=segment.segment_id,
        batch_id=segment.batch_id,
        frame=segment.frame.drop(columns=["effective_field_mT"]),
        metadata=segment.metadata,
    )

    result = build_aligned_segment_residual(segment, row)

    assert result.status == "measured_field_column_missing"
    assert result.metadata["status"] == "measured_field_column_missing"


def test_peak_pair_midpoint_falls_back_when_peaks_missing() -> None:
    row = _row()
    segment = _segment(row, flat=True)

    result = build_aligned_segment_residual(
        segment,
        row,
        config=SegmentAlignmentConfig(phase_sync_method="peak_pair_midpoint_to_zero_crossing"),
    )

    assert result.status in {"ok", "phase_sync_failed"}
    assert result.metadata["phase_sync_fallback_used"] is True
    assert result.metadata["hardware_invoked"] is False


def test_output_columns_present() -> None:
    row = _row()
    result = build_aligned_segment_residual(_segment(row), row)

    assert REQUIRED_COLUMNS.issubset(result.frame.columns)


def test_metadata_flags() -> None:
    row = _row()
    result = build_aligned_segment_residual(_segment(row), row)

    assert result.metadata["hardware_invoked"] is False
    assert result.metadata["modeling_core_called"] is False
    assert result.metadata["streamlit_involved"] is False
    assert result.metadata["ml_training_involved"] is False


def test_no_streamlit_or_production_modeling_imports() -> None:
    import coil_ai_sweep.segment_alignment as segment_alignment

    source = inspect.getsource(segment_alignment)

    assert "streamlit" not in source
    assert "PySide6" not in source
    assert "field_analysis" not in source
    assert "finite_second_modeling" not in source
    assert "finite_first" not in source
    assert "continuous" not in source
    assert "app_ui_snapshot" not in source


def test_input_frame_not_mutated() -> None:
    row = _row()
    segment = _segment(row)
    original = segment.frame.copy(deep=True)

    build_aligned_segment_residual(segment, row)

    pd.testing.assert_frame_equal(segment.frame, original)
