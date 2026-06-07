from __future__ import annotations

from pathlib import Path
import inspect
import json
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from coil_ai_sweep.manifest_io import validate_manifest_dataframe
from coil_ai_sweep.schema import SweepSegmentSpec, SweepTargetConfig
from coil_ai_sweep.sweep_lut_generator import (
    SegmentCommandInput,
    build_sweep_lut_from_segment_commands,
)
from coil_ai_sweep.core_adapter import get_voltage_limit_v


def _spec(
    *,
    segment_id: str = "S0001",
    sample_rate_hz: float = 1000.0,
    pre_idle_s: float = 0.002,
    post_idle_s: float = 0.002,
    freq_hz: float = 1.0,
    cycle_count: float = 1.0,
    target_peak_mT: float = 20.0,
    variant_params: dict[str, object] | None = None,
) -> SweepSegmentSpec:
    return SweepSegmentSpec(
        batch_id="batch-a",
        segment_id=segment_id,
        target=SweepTargetConfig(
            freq_hz=freq_hz,
            cycle_count=cycle_count,
            target_peak_mT=target_peak_mT,
        ),
        variant_params=variant_params if variant_params is not None else {"gain": 1.0},
        pre_idle_s=pre_idle_s,
        post_idle_s=post_idle_s,
        sample_rate_hz=sample_rate_hz,
    )


def _profile(values: list[float] | None = None, *, column: str = "voltage_v") -> pd.DataFrame:
    voltage = values if values is not None else [0.0, 1.0, 0.0]
    return pd.DataFrame(
        {
            "time_s": [index * 0.001 for index in range(len(voltage))],
            column: voltage,
        }
    )


def _segment(
    *,
    segment_id: str = "S0001",
    sample_rate_hz: float = 1000.0,
    pre_idle_s: float = 0.002,
    post_idle_s: float = 0.002,
    values: list[float] | None = None,
) -> SegmentCommandInput:
    return SegmentCommandInput(
        spec=_spec(
            segment_id=segment_id,
            sample_rate_hz=sample_rate_hz,
            pre_idle_s=pre_idle_s,
            post_idle_s=post_idle_s,
        ),
        command_profile=_profile(values),
    )


def test_build_sweep_lut_outputs_only_hardware_columns() -> None:
    result = build_sweep_lut_from_segment_commands(
        [
            _segment(segment_id="S0001"),
            _segment(segment_id="S0002", values=[0.0, -1.0, 0.0]),
        ]
    )

    assert result.status == "ok"
    assert list(result.lut.columns) == ["sample_index", "time_s", "voltage_v"]
    assert "segment_id" not in result.lut.columns
    assert "freq_hz" not in result.lut.columns
    assert "target_peak_mT" not in result.lut.columns


def test_build_sweep_lut_creates_valid_manifest_rows() -> None:
    result = build_sweep_lut_from_segment_commands(
        [
            _segment(segment_id="S0001"),
            _segment(segment_id="S0002", values=[0.0, -1.0, 0.0]),
        ]
    )

    validation = validate_manifest_dataframe(result.manifest)

    assert validation.ok is True
    assert len(result.manifest_rows) == 2
    assert result.manifest_rows[0].end_sample < result.manifest_rows[1].start_sample
    for row in result.manifest_rows:
        assert row.start_sample <= row.active_start_sample < row.active_end_sample <= row.end_sample


def test_build_sweep_lut_inserts_pre_and_post_idle_zero_voltage() -> None:
    result = build_sweep_lut_from_segment_commands(
        [_segment(pre_idle_s=0.002, post_idle_s=0.002, values=[0.0, 1.0, -1.0])]
    )

    assert result.lut["voltage_v"].tolist() == [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0]
    row = result.manifest_rows[0]
    assert row.start_sample == 0
    assert row.active_start_sample == 2
    assert row.active_end_sample == 4
    assert row.end_sample == 6


def test_build_sweep_lut_rejects_voltage_limit_exceedance() -> None:
    with pytest.raises(ValueError, match="voltage_limit_exceeded"):
        build_sweep_lut_from_segment_commands(
            [_segment(values=[0.0, get_voltage_limit_v() + 0.001, 0.0])]
        )


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"time_s": [0.0, np.nan, 0.002], "voltage_v": [0.0, 1.0, 0.0]}),
        pd.DataFrame({"time_s": [0.0, 0.001, 0.002], "voltage_v": [0.0, np.nan, 0.0]}),
    ],
)
def test_build_sweep_lut_rejects_nan_time_or_voltage(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        build_sweep_lut_from_segment_commands([SegmentCommandInput(spec=_spec(), command_profile=frame)])


def test_build_sweep_lut_rejects_mixed_sample_rates() -> None:
    with pytest.raises(ValueError, match="mixed_sample_rate_hz_not_supported"):
        build_sweep_lut_from_segment_commands(
            [
                _segment(segment_id="S0001", sample_rate_hz=1000.0),
                _segment(segment_id="S0002", sample_rate_hz=2000.0),
            ]
        )


def test_build_sweep_lut_uses_existing_voltage_policy_constant() -> None:
    import coil_ai_sweep.sweep_lut_generator as generator

    source = inspect.getsource(generator)

    assert "get_voltage_limit_v()" in source
    assert "10.0" not in source


def test_build_sweep_lut_metadata() -> None:
    result = build_sweep_lut_from_segment_commands([_segment()])

    assert result.metadata["modeling_core_called"] is False
    assert result.metadata["streamlit_involved"] is False
    assert result.metadata["hardware_invoked"] is False
    assert result.metadata["ai_sweep_lut_generation_mode"] == "concatenate_prebuilt_segment_commands"
    assert result.metadata["voltage_policy_source"] in {"standalone_fallback", "core_dependency"}


def test_ai_sweep_lut_generator_does_not_import_streamlit() -> None:
    import coil_ai_sweep.sweep_lut_generator as generator

    assert "streamlit" not in inspect.getsource(generator)


def test_ai_sweep_lut_generator_does_not_import_production_modeling() -> None:
    import coil_ai_sweep.sweep_lut_generator as generator

    source = inspect.getsource(generator)

    assert "finite_second_modeling" not in source
    assert "finite_first" not in source
    assert "continuous" not in source
    assert "app_ui_snapshot" not in source
    assert "streamlit" not in source
