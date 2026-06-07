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

from coil_ai_sweep.manifest_io import manifest_rows_to_dataframe
from coil_ai_sweep.schema import SweepSegmentManifestRow
from coil_ai_sweep.segment_parser import split_long_measurement_by_manifest


def _row(
    *,
    segment_id: str = "S0001",
    start_sample: int = 0,
    end_sample: int = 4,
    active_start_sample: int = 1,
    active_end_sample: int = 3,
    start_time_s: float = 0.0,
    end_time_s: float = 0.004,
    active_start_time_s: float = 0.001,
    active_end_time_s: float = 0.003,
) -> SweepSegmentManifestRow:
    return SweepSegmentManifestRow(
        batch_id="batch-a",
        segment_id=segment_id,
        start_sample=start_sample,
        end_sample=end_sample,
        active_start_sample=active_start_sample,
        active_end_sample=active_end_sample,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        active_start_time_s=active_start_time_s,
        active_end_time_s=active_end_time_s,
        freq_hz=1.0,
        cycle_count=1.0,
        target_peak_mT=20.0,
        target_shape="fixed_rounded_triangle",
        source_waveform_family="triangle",
        mode="finite",
        variant_type="baseline",
        variant_params_json="{}",
    )


def _manifest(rows: list[SweepSegmentManifestRow] | None = None) -> pd.DataFrame:
    return manifest_rows_to_dataframe(
        rows
        if rows is not None
        else [
            _row(segment_id="S0001"),
            _row(
                segment_id="S0002",
                start_sample=5,
                end_sample=9,
                active_start_sample=6,
                active_end_sample=8,
                start_time_s=0.005,
                end_time_s=0.009,
                active_start_time_s=0.006,
                active_end_time_s=0.008,
            ),
        ]
    )


def _measurement() -> pd.DataFrame:
    time_s = np.arange(0.0, 0.010, 0.001)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "HallBz": np.arange(len(time_s), dtype=float),
            "Voltage1_V": np.linspace(0.0, 1.0, len(time_s)),
        }
    )


def test_split_long_measurement_by_manifest_splits_segments() -> None:
    result = split_long_measurement_by_manifest(_measurement(), _manifest())

    assert result.status == "ok"
    assert set(result.segments) == {"S0001", "S0002"}
    assert "local_time_s" in result.segments["S0001"].frame.columns
    assert "active_local_time_s" in result.segments["S0001"].frame.columns


def test_split_adds_effective_field_from_hallbz() -> None:
    result = split_long_measurement_by_manifest(_measurement(), _manifest())
    frame = result.segments["S0001"].frame

    assert "hallbz_raw_mT" in frame.columns
    assert "effective_field_mT" in frame.columns
    assert np.allclose(frame["effective_field_mT"], -frame["HallBz"])


def test_split_accepts_timems() -> None:
    measurement = _measurement().drop(columns=["time_s"])
    measurement["TimeMs"] = np.arange(0.0, 10.0, 1.0)

    result = split_long_measurement_by_manifest(measurement, _manifest())
    frame = result.segments["S0001"].frame

    assert result.metadata["time_unit_source"] == "TimeMs"
    assert np.allclose(frame["measurement_time_s"], frame["TimeMs"] / 1000.0)


def test_split_adds_measured_voltage_when_available() -> None:
    result = split_long_measurement_by_manifest(_measurement(), _manifest())

    assert "measured_voltage_v" in result.segments["S0001"].frame.columns
    assert np.allclose(result.segments["S0001"].frame["measured_voltage_v"], result.segments["S0001"].frame["Voltage1_V"])


def test_split_support_margin_includes_rows_outside_segment() -> None:
    result = split_long_measurement_by_manifest(_measurement(), _manifest([_row()]), support_margin_s=0.001)
    frame = result.segments["S0001"].frame

    assert frame["measurement_time_s"].min() < 0.0 + 1e-12 or np.isclose(frame["measurement_time_s"].min(), 0.0)
    assert frame["measurement_time_s"].max() > 0.004
    assert not bool(frame.loc[frame["measurement_time_s"] > 0.004, "segment_window_mask"].any())
    assert bool(frame["support_window_mask"].all())


def test_split_active_window_mask() -> None:
    result = split_long_measurement_by_manifest(_measurement(), _manifest([_row()]))
    frame = result.segments["S0001"].frame

    expected = (frame["measurement_time_s"] >= 0.001) & (frame["measurement_time_s"] <= 0.003)
    assert frame["active_window_mask"].tolist() == expected.tolist()


def test_split_does_not_zero_fill_empty_support() -> None:
    result = split_long_measurement_by_manifest(
        _measurement(),
        _manifest([_row(start_sample=100, end_sample=104, active_start_sample=101, active_end_sample=103, start_time_s=1.0, end_time_s=1.004, active_start_time_s=1.001, active_end_time_s=1.003)]),
    )

    assert result.status == "empty"
    assert result.segments["S0001"].frame.empty
    assert result.segments["S0001"].metadata["support_window_status"] == "empty"


def test_split_rejects_non_monotonic_time() -> None:
    measurement = _measurement()
    measurement.loc[2, "time_s"] = 0.0

    with pytest.raises(ValueError, match="measurement_time_s_must_be_monotonic_increasing"):
        split_long_measurement_by_manifest(measurement, _manifest())


def test_split_rejects_negative_support_margin() -> None:
    with pytest.raises(ValueError, match="support_margin_s_must_be_non_negative"):
        split_long_measurement_by_manifest(_measurement(), _manifest(), support_margin_s=-0.001)


def test_segment_parser_metadata_flags() -> None:
    result = split_long_measurement_by_manifest(_measurement(), _manifest())

    assert result.metadata["interpolation_used"] is False
    assert result.metadata["smoothing_used"] is False
    assert result.metadata["phase_alignment_used"] is False
    assert result.metadata["residual_computed"] is False
    assert result.metadata["modeling_core_called"] is False
    assert result.metadata["hardware_invoked"] is False
    assert result.metadata["streamlit_involved"] is False


def test_ai_sweep_segment_parser_does_not_import_streamlit_or_production_modeling() -> None:
    import coil_ai_sweep.segment_parser as segment_parser

    source = inspect.getsource(segment_parser)

    assert "streamlit" not in source
    assert "finite_second_modeling" not in source
    assert "finite_first" not in source
    assert "continuous" not in source
    assert "app_ui_snapshot" not in source
