from __future__ import annotations

from pathlib import Path
import inspect
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from coil_ai_sweep.peak_response_modeling import (
    PeakResponseConfig,
    build_peak_response_from_source_segment,
)
from coil_ai_sweep.schema import SweepSegmentManifestRow
from coil_ai_sweep.segment_parser import SegmentMeasurement


def _row(*, cycle_count: float = 1.5, target_peak_mT: float = 50.0) -> SweepSegmentManifestRow:
    duration_s = cycle_count / 2.0
    return SweepSegmentManifestRow(
        batch_id="batch-a",
        segment_id="S0001",
        start_sample=0,
        end_sample=1000,
        active_start_sample=0,
        active_end_sample=1000,
        start_time_s=0.0,
        end_time_s=duration_s,
        active_start_time_s=0.0,
        active_end_time_s=duration_s,
        freq_hz=2.0,
        cycle_count=cycle_count,
        target_peak_mT=target_peak_mT,
        target_shape="fixed_rounded_triangle",
        source_waveform_family="triangle",
        mode="finite",
        variant_type="baseline",
        variant_params_json="{}",
    )


def _triangle_voltage(cycle_position: np.ndarray, peak_v: float = 1.0) -> np.ndarray:
    phase = cycle_position % 1.0
    unit = np.interp(phase, [0.0, 0.25, 0.75, 1.0], [0.0, 1.0, -1.0, 0.0])
    return peak_v * unit


def _field_from_peak_map(cycle_position: np.ndarray) -> np.ndarray:
    anchors = np.array([0.25, 0.75, 1.25])
    values = np.array([40.0, -20.0, 30.0])
    baseline = np.interp(cycle_position, [0.0, 0.25, 0.75, 1.25, 1.5], [0.0, 40.0, -20.0, 30.0, 0.0])
    for anchor, value in zip(anchors, values):
        baseline[np.argmin(np.abs(cycle_position - anchor))] = value
    return baseline


def _segment(row: SweepSegmentManifestRow, *, voltage_peak_v: float = 1.0, include_voltage: bool = True) -> SegmentMeasurement:
    active_duration_s = float(row.cycle_count) / float(row.freq_hz)
    time_s = np.linspace(0.0, active_duration_s, 1501, endpoint=False)
    cycle_position = time_s * float(row.freq_hz)
    frame = pd.DataFrame(
        {
            "active_local_time_s": time_s,
            "effective_field_mT": _field_from_peak_map(cycle_position),
        }
    )
    if include_voltage:
        frame["measured_voltage_v"] = _triangle_voltage(cycle_position, peak_v=voltage_peak_v)
    return SegmentMeasurement(segment_id=row.segment_id, batch_id=row.batch_id, frame=frame, metadata={})


def test_peak_response_extracts_1p5_three_peak_roles() -> None:
    row = _row(cycle_count=1.5)

    result = build_peak_response_from_source_segment(
        _segment(row),
        row,
        config=PeakResponseConfig(target_peak_mT=50.0),
    )

    required = {record.peak_role: record.required_voltage_peak_v for record in result.peak_records}
    assert required["positive_peak_1"] == pytest.approx(1.25, abs=0.01)
    assert required["negative_peak_1"] == pytest.approx(-2.5, abs=0.01)
    assert required["positive_peak_2"] == pytest.approx(1.6667, abs=0.01)
    assert result.metadata["detected_peak_count"] == 3


def test_peak_response_extracts_1p0_two_peak_roles() -> None:
    row = _row(cycle_count=1.0)

    result = build_peak_response_from_source_segment(
        _segment(row),
        row,
        config=PeakResponseConfig(target_peak_mT=50.0),
    )

    assert [record.peak_role for record in result.peak_records] == ["positive_peak_1", "negative_peak_1"]


def test_uses_effective_field_sign_convention_from_hallbz() -> None:
    row = _row(cycle_count=1.0)
    segment = _segment(row)
    frame = segment.frame.drop(columns=["effective_field_mT"]).copy()
    frame["hallbz_raw_mT"] = -segment.frame["effective_field_mT"]

    result = build_peak_response_from_source_segment(
        SegmentMeasurement(segment.segment_id, segment.batch_id, frame, segment.metadata),
        row,
        config=PeakResponseConfig(target_peak_mT=50.0),
    )

    assert result.metadata["hallbz_convention"] == "effective_field_mT = -HallBz raw"
    assert result.peak_records[0].peak_role == "positive_peak_1"
    assert result.peak_records[0].measured_field_peak_mT > 0


def test_uses_measured_voltage_peak_when_available() -> None:
    row = _row(cycle_count=1.0)

    result = build_peak_response_from_source_segment(
        _segment(row, voltage_peak_v=0.9),
        row,
        config=PeakResponseConfig(target_peak_mT=50.0),
    )

    assert result.peak_records[0].input_voltage_peak_v == pytest.approx(0.9, abs=0.01)
    assert result.metadata["voltage_peak_source"] == "measured_voltage_column"


def test_falls_back_to_nominal_voltage_peak_when_voltage_missing() -> None:
    row = _row(cycle_count=1.0)

    result = build_peak_response_from_source_segment(
        _segment(row, include_voltage=False),
        row,
        config=PeakResponseConfig(target_peak_mT=50.0, source_voltage_vpp=2.0),
    )

    assert abs(result.peak_records[0].input_voltage_peak_v) == pytest.approx(1.0)
    assert result.metadata["voltage_peak_source"] == "nominal_source_voltage_peak"


def test_required_voltage_limit_blocks_command_profile() -> None:
    row = _row(cycle_count=1.0)
    segment = _segment(row)
    frame = segment.frame.copy()
    frame["effective_field_mT"] = frame["effective_field_mT"] * 0.01

    result = build_peak_response_from_source_segment(
        SegmentMeasurement(segment.segment_id, segment.batch_id, frame, segment.metadata),
        row,
        config=PeakResponseConfig(target_peak_mT=50.0),
    )

    assert len(result.peak_table) == 2
    assert result.command_profile is None
    assert result.status == "blocked_required_voltage_exceeds_limit"


def test_keypoint_command_profile_columns_and_duration() -> None:
    row = _row(cycle_count=1.0)

    result = build_peak_response_from_source_segment(
        _segment(row),
        row,
        config=PeakResponseConfig(target_peak_mT=50.0, keypoint_command_sample_rate_hz=1000.0),
    )

    assert result.command_profile is not None
    assert list(result.command_profile.columns) == ["time_s", "voltage_v"]
    assert result.command_profile["time_s"].iloc[0] == 0.0
    assert result.command_profile["time_s"].iloc[-1] < float(row.cycle_count) / float(row.freq_hz)
    assert result.command_profile["voltage_v"].abs().max() == pytest.approx(2.5, abs=0.02)


def test_no_phase_lead_applied_initially() -> None:
    row = _row(cycle_count=1.0)

    result = build_peak_response_from_source_segment(
        _segment(row),
        row,
        config=PeakResponseConfig(target_peak_mT=50.0),
    )

    assert result.metadata["keypoint_phase_lead_applied"] is False


def test_unsupported_cycle_rejected() -> None:
    row = _row(cycle_count=1.0)
    unsupported = object.__new__(SweepSegmentManifestRow)
    for key, value in row.to_dict().items():
        object.__setattr__(unsupported, key, value)
    object.__setattr__(unsupported, "cycle_count", 1.25)

    result = build_peak_response_from_source_segment(
        _segment(row),
        unsupported,
        config=PeakResponseConfig(target_peak_mT=50.0),
    )

    assert result.status == "unsupported_cycle_count_for_peak_response"


def test_no_streamlit_or_production_modeling_imports() -> None:
    import coil_ai_sweep.peak_response_modeling as peak_response_modeling

    source = inspect.getsource(peak_response_modeling)
    forbidden = [
        "streamlit",
        "PySide6",
        "field_analysis",
        "finite_second_modeling",
        "finite_first",
        "continuous",
        "app_ui_snapshot",
    ]
    for needle in forbidden:
        assert needle not in source
