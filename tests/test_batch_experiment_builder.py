from __future__ import annotations

from copy import deepcopy
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
    cycle_count: float = 1.5,
    target_peak_mT: float = 20.0,
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
        target_peak_mT=target_peak_mT,
        target_shape="fixed_rounded_triangle",
        source_waveform_family="triangle",
        mode="finite",
        variant_type="source_response",
        variant_params_json="{}",
    )


def _triangle_voltage(cycle_position: np.ndarray, peak_v: float = 1.0) -> np.ndarray:
    phase = cycle_position % 1.0
    unit = np.interp(phase, [0.0, 0.25, 0.75, 1.0], [0.0, 1.0, -1.0, 0.0])
    return peak_v * unit


def _field_from_peak_map(cycle_position: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    baseline = np.interp(cycle_position, [0.0, 0.25, 0.75, 1.25, 1.5], [0.0, 40.0, -20.0, 30.0, 0.0])
    return baseline * scale


def _source(
    *,
    batch_id: str = "batch-a",
    segment_id: str = "S0001",
    cycle_count: float = 1.5,
    field_scale: float = 1.0,
) -> BatchSourceSegment:
    row = _row(batch_id=batch_id, segment_id=segment_id, cycle_count=cycle_count)
    active_duration_s = float(row.cycle_count) / float(row.freq_hz)
    time_s = np.linspace(0.0, active_duration_s, 1501, endpoint=False)
    cycle_position = time_s * float(row.freq_hz)
    frame = pd.DataFrame(
        {
            "active_local_time_s": time_s,
            "effective_field_mT": _field_from_peak_map(cycle_position, scale=field_scale),
            "measured_voltage_v": _triangle_voltage(cycle_position),
        }
    )
    segment = SegmentMeasurement(segment_id=row.segment_id, batch_id=row.batch_id, frame=frame, metadata={})
    return BatchSourceSegment(segment=segment, manifest_row=row)


def _config(*, include_blocked_commands: bool = False, preserve_original_segment_ids: bool = False) -> BatchExperimentConfig:
    return BatchExperimentConfig(
        peak_response_config=PeakResponseConfig(
            target_peak_mT=50.0,
            keypoint_command_sample_rate_hz=1000.0,
        ),
        include_blocked_commands=include_blocked_commands,
        preserve_original_segment_ids=preserve_original_segment_ids,
        output_pre_idle_s=0.002,
        output_post_idle_s=0.002,
        output_sample_rate_hz=1000.0,
    )


def test_batch_experiment_builds_peak_table_and_sweep_lut() -> None:
    result = build_batch_experiment_from_peak_responses(
        [_source(segment_id="S0001"), _source(segment_id="S0002")],
        config=_config(),
    )

    assert result.status == "ok"
    assert len(result.peak_table) == 6
    assert result.sweep_lut_result is not None
    assert list(result.sweep_lut_result.lut.columns) == ["sample_index", "time_s", "voltage_v"]
    assert result.metadata["command_segment_count"] == 2


def test_batch_experiment_excludes_blocked_commands_but_keeps_peak_table() -> None:
    result = build_batch_experiment_from_peak_responses(
        [_source(segment_id="S0001"), _source(segment_id="S0002", field_scale=0.01)],
        config=_config(),
    )

    assert set(result.peak_table["source_segment_id"]) == {"S0001", "S0002"}
    assert len(result.segment_commands) == 1
    assert result.segment_commands[0].spec.variant_params["source_segment_id"] == "S0001"
    assert result.metadata["blocked_source_count"] == 1
    assert result.status == "partial"


def test_batch_experiment_no_commands_when_all_blocked() -> None:
    result = build_batch_experiment_from_peak_responses(
        [_source(segment_id="S0001", field_scale=0.01), _source(segment_id="S0002", field_scale=0.01)],
        config=_config(),
    )

    assert result.sweep_lut_result is None
    assert result.status == "blocked_all_sources"


def test_batch_experiment_variant_params_include_peak_response_summary() -> None:
    result = build_batch_experiment_from_peak_responses([_source(segment_id="S0001")], config=_config())

    params = result.segment_commands[0].spec.variant_params
    assert params["source_segment_id"] == "S0001"
    assert params["generated_from"] == "peak_response_modeling"
    assert set(params["required_voltage_peaks_by_role"]) == {
        "positive_peak_1",
        "negative_peak_1",
        "positive_peak_2",
    }
    assert set(params["phase_delays_by_role"]) == {
        "positive_peak_1",
        "negative_peak_1",
        "positive_peak_2",
    }


def test_batch_experiment_generated_segment_ids_are_stable() -> None:
    result = build_batch_experiment_from_peak_responses(
        [_source(segment_id="S0101"), _source(segment_id="S0102")],
        config=_config(),
    )

    assert [command.spec.segment_id for command in result.segment_commands] == ["K0001", "K0002"]


def test_batch_experiment_rejects_include_blocked_commands() -> None:
    with pytest.raises(ValueError, match="include_blocked_commands_not_supported"):
        BatchExperimentConfig(
            peak_response_config=PeakResponseConfig(target_peak_mT=50.0),
            include_blocked_commands=True,
        )


def test_batch_experiment_does_not_mutate_inputs() -> None:
    source = _source(segment_id="S0001")
    original_frame = source.segment.frame.copy(deep=True)
    original_row = deepcopy(source.manifest_row)

    build_batch_experiment_from_peak_responses([source], config=_config())

    pd.testing.assert_frame_equal(source.segment.frame, original_frame)
    assert source.manifest_row == original_row


def test_batch_experiment_metadata_flags() -> None:
    result = build_batch_experiment_from_peak_responses([_source(segment_id="S0001")], config=_config())

    assert result.metadata["hardware_invoked"] is False
    assert result.metadata["modeling_core_called"] is False
    assert result.metadata["streamlit_involved"] is False
    assert result.metadata["winapp_involved"] is False
    assert result.metadata["ml_training_involved"] is False
    assert result.metadata["residual_computed"] is False


def test_no_streamlit_or_production_modeling_imports() -> None:
    import coil_ai_sweep.batch_experiment_builder as batch_experiment_builder

    source = inspect.getsource(batch_experiment_builder)
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
