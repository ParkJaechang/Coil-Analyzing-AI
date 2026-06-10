from __future__ import annotations

from pathlib import Path
import inspect
import json
import math
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from coil_ai_sweep.batch_dataset_export import (
    BatchDatasetExportConfig,
    build_batch_dataset_export_packet,
)
from coil_ai_sweep.batch_experiment_builder import (
    BatchExperimentConfig,
    BatchSourceSegment,
    build_batch_experiment_from_peak_responses,
)
from coil_ai_sweep.peak_response_modeling import PeakResponseConfig
from coil_ai_sweep.schema import SweepSegmentManifestRow
from coil_ai_sweep.segment_parser import SegmentMeasurement


def _row(
    *,
    batch_id: str = "batch-a",
    segment_id: str = "S0001",
    cycle_count: float = 1.0,
) -> SweepSegmentManifestRow:
    duration_s = cycle_count / 2.0
    return SweepSegmentManifestRow(
        batch_id=batch_id,
        segment_id=segment_id,
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
        target_peak_mT=20.0,
        target_shape="fixed_rounded_triangle",
        source_waveform_family="triangle",
        mode="finite",
        variant_type="source_response",
        variant_params_json="{}",
    )


def _triangle_voltage(cycle_position: np.ndarray) -> np.ndarray:
    phase = cycle_position % 1.0
    return np.interp(phase, [0.0, 0.25, 0.75, 1.0], [0.0, 1.0, -1.0, 0.0])


def _field(cycle_position: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    return scale * np.interp(cycle_position, [0.0, 0.25, 0.75, 1.0], [0.0, 40.0, -20.0, 0.0])


def _source(*, segment_id: str = "S0001", field_scale: float = 1.0) -> BatchSourceSegment:
    row = _row(segment_id=segment_id)
    active_duration_s = float(row.cycle_count) / float(row.freq_hz)
    time_s = np.linspace(0.0, active_duration_s, 501, endpoint=False)
    cycle_position = time_s * float(row.freq_hz)
    frame = pd.DataFrame(
        {
            "active_local_time_s": time_s,
            "effective_field_mT": _field(cycle_position, scale=field_scale),
            "measured_voltage_v": _triangle_voltage(cycle_position),
        }
    )
    segment = SegmentMeasurement(segment_id=row.segment_id, batch_id=row.batch_id, frame=frame, metadata={})
    return BatchSourceSegment(segment=segment, manifest_row=row)


def _batch_result(*, blocked: bool = False):
    sources = [_source(segment_id="S0001")]
    if blocked:
        sources.append(_source(segment_id="S0002", field_scale=0.01))
    return build_batch_experiment_from_peak_responses(
        sources,
        config=BatchExperimentConfig(
            peak_response_config=PeakResponseConfig(
                target_peak_mT=50.0,
                keypoint_command_sample_rate_hz=10.0,
            ),
            output_pre_idle_s=0.0,
            output_post_idle_s=0.0,
            output_sample_rate_hz=10.0,
        ),
    )


def _walk(value: Any):
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def test_build_batch_dataset_export_packet_json_safe() -> None:
    result = build_batch_dataset_export_packet(
        _batch_result(),
        config=BatchDatasetExportConfig(dataset_id="dataset-a"),
    )

    encoded = json.dumps(result.packet, allow_nan=False)

    assert "NaN" not in encoded
    for value in _walk(result.packet):
        assert not isinstance(value, pd.DataFrame)
        if isinstance(value, float):
            assert math.isfinite(value)


def test_export_packet_contains_summary_and_peak_records() -> None:
    batch_result = _batch_result()

    result = build_batch_dataset_export_packet(
        batch_result,
        config=BatchDatasetExportConfig(dataset_id="dataset-a"),
    )

    assert result.packet["summary"]["source_segment_count"] == 1
    assert result.packet["summary"]["batch_result_status"] == batch_result.status
    assert len(result.packet["peak_records"]) == len(batch_result.peak_table)
    assert {
        "source_segment_id",
        "peak_role",
        "measured_field_peak_mT",
        "input_voltage_peak_v",
        "required_voltage_peak_v",
        "required_voltage_gain",
        "phase_delay_s",
        "phase_delay_cycles",
        "voltage_limit_status",
        "peak_response_status",
    }.issubset(result.packet["peak_records"][0])


def test_export_packet_generated_command_segments_summary() -> None:
    result = build_batch_dataset_export_packet(
        _batch_result(),
        config=BatchDatasetExportConfig(dataset_id="dataset-a"),
    )

    command = result.packet["generated_command_segments"][0]
    assert command["sample_count"] > 0
    assert command["voltage_min_v"] <= command["voltage_max_v"]
    assert command["voltage_peak_abs_v"] > 0
    assert "samples" not in command
    assert "voltage_v" not in command


def test_export_packet_sweep_lut_summary_schema_ok() -> None:
    result = build_batch_dataset_export_packet(
        _batch_result(),
        config=BatchDatasetExportConfig(dataset_id="dataset-a"),
    )

    summary = result.packet["sweep_lut_summary"]
    assert summary["hardware_lut_schema_ok"] is True
    assert summary["lut_columns"] == ["sample_index", "time_s", "voltage_v"]


def test_export_packet_blocked_sources() -> None:
    result = build_batch_dataset_export_packet(
        _batch_result(blocked=True),
        config=BatchDatasetExportConfig(dataset_id="dataset-a"),
    )

    assert [item["source_segment_id"] for item in result.packet["blocked_sources"]] == ["S0002"]
    assert [item["segment_id"] for item in result.packet["generated_command_segments"]] == ["K0001"]


def test_full_lut_samples_default_omitted() -> None:
    result = build_batch_dataset_export_packet(
        _batch_result(),
        config=BatchDatasetExportConfig(dataset_id="dataset-a"),
    )

    assert "full_lut_samples" not in result.packet
    assert result.metadata["full_lut_samples_included"] is False


def test_full_lut_samples_small_inline() -> None:
    result = build_batch_dataset_export_packet(
        _batch_result(),
        config=BatchDatasetExportConfig(
            dataset_id="dataset-a",
            include_full_lut_samples=True,
            max_lut_samples_inline=100,
        ),
    )

    assert result.metadata["full_lut_samples_included"] is True
    assert result.packet["full_lut_samples"]
    assert set(result.packet["full_lut_samples"][0]) == {"sample_index", "time_s", "voltage_v"}


def test_full_lut_samples_limit_blocks_large_inline() -> None:
    with pytest.raises(ValueError, match="full_lut_samples_exceed_inline_limit"):
        build_batch_dataset_export_packet(
            _batch_result(),
            config=BatchDatasetExportConfig(
                dataset_id="dataset-a",
                include_full_lut_samples=True,
                max_lut_samples_inline=1,
            ),
        )


def test_export_metadata_flags() -> None:
    result = build_batch_dataset_export_packet(
        _batch_result(),
        config=BatchDatasetExportConfig(dataset_id="dataset-a"),
    )

    assert result.packet["safety"]["hardware_invoked"] is False
    assert result.packet["safety"]["modeling_core_called"] is False
    assert result.packet["safety"]["streamlit_involved"] is False
    assert result.packet["safety"]["winapp_involved"] is False
    assert result.packet["safety"]["ml_training_involved"] is False
    assert result.packet["safety"]["file_written"] is False


def test_no_streamlit_or_production_modeling_imports() -> None:
    import coil_ai_sweep.batch_dataset_export as batch_dataset_export

    source = inspect.getsource(batch_dataset_export)
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
