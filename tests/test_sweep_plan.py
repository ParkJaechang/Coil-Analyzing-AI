from __future__ import annotations

from pathlib import Path
import inspect
import json
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from coil_ai_sweep.sweep_plan import SweepPlanConfig, build_sweep_plan, plan_to_dataframe


def _config(**overrides: object) -> SweepPlanConfig:
    values = {
        "batch_id": "batch-a",
        "frequency_grid_hz": [1.0],
        "cycle_counts": [1.0],
        "target_peak_mT_values": [10.0],
        "sample_rate_hz": 10_000.0,
        "pre_idle_s": 0.01,
        "post_idle_s": 0.02,
    }
    values.update(overrides)
    return SweepPlanConfig(**values)


def test_build_sweep_plan_finite_grid_count() -> None:
    plan = build_sweep_plan(
        _config(
            frequency_grid_hz=[0.5, 0.25],
            cycle_counts=[1.5, 1.0],
            target_peak_mT_values=[30.0, 10.0],
        )
    )

    assert len(plan) == 8
    assert [segment.segment_id for segment in plan] == [f"S{index:04d}" for index in range(1, 9)]
    assert [
        (
            segment.target.freq_hz,
            segment.target.cycle_count,
            segment.target.target_peak_mT,
        )
        for segment in plan
    ] == [
        (0.25, 1.0, 10.0),
        (0.25, 1.0, 30.0),
        (0.25, 1.5, 10.0),
        (0.25, 1.5, 30.0),
        (0.5, 1.0, 10.0),
        (0.5, 1.0, 30.0),
        (0.5, 1.5, 10.0),
        (0.5, 1.5, 30.0),
    ]


def test_build_sweep_plan_rejects_continuous_1p5() -> None:
    with pytest.raises(ValueError, match="continuous_cycle_count_must_be_1p0"):
        build_sweep_plan(_config(mode="continuous", cycle_counts=[1.0, 1.5]))


def test_build_sweep_plan_preserves_user_target_values() -> None:
    plan = build_sweep_plan(
        _config(
            frequency_grid_hz=[1.25],
            cycle_counts=[1.5],
            target_peak_mT_values=[27.0],
            source_waveform_family="triangle",
        )
    )

    target = plan[0].target
    assert target.freq_hz == 1.25
    assert target.cycle_count == 1.5
    assert target.target_peak_mT == 27.0
    assert target.target_shape == "fixed_rounded_triangle"
    assert target.source_waveform_family == "triangle"


def test_build_sweep_plan_rejects_empty_grids() -> None:
    with pytest.raises(ValueError, match="frequency_grid_hz_must_be_non_empty"):
        build_sweep_plan(_config(frequency_grid_hz=[]))
    with pytest.raises(ValueError, match="cycle_counts_must_be_non_empty"):
        build_sweep_plan(_config(cycle_counts=[]))
    with pytest.raises(ValueError, match="target_peak_mT_values_must_be_non_empty"):
        build_sweep_plan(_config(target_peak_mT_values=[]))


@pytest.mark.parametrize(
    ("field_name", "value", "error"),
    [
        ("frequency_grid_hz", [0.0], "freq_hz_must_be_positive"),
        ("target_peak_mT_values", [0.0], "target_peak_mT_must_be_positive"),
        ("sample_rate_hz", 0.0, "sample_rate_hz_must_be_positive"),
        ("pre_idle_s", -0.01, "pre_idle_s_must_be_non_negative"),
        ("post_idle_s", -0.01, "post_idle_s_must_be_non_negative"),
    ],
)
def test_build_sweep_plan_rejects_non_positive_values(
    field_name: str,
    value: object,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        build_sweep_plan(_config(**{field_name: value}))


def test_build_sweep_plan_anchor_repeats() -> None:
    plan = build_sweep_plan(
        _config(
            frequency_grid_hz=[1.0],
            cycle_counts=[1.0, 1.5],
            target_peak_mT_values=[10.0, 30.0],
            include_anchor_repeats=True,
            anchor_peak_mT=20.0,
        )
    )

    assert len(plan) == 6
    assert [segment.variant_type for segment in plan] == [
        "baseline",
        "baseline",
        "anchor_repeat",
        "baseline",
        "baseline",
        "anchor_repeat",
    ]
    assert [segment.target.target_peak_mT for segment in plan if segment.variant_type == "anchor_repeat"] == [
        20.0,
        20.0,
    ]


def test_plan_to_dataframe_if_implemented() -> None:
    plan = build_sweep_plan(_config(variant_params={"gain": 1.0}))

    frame = plan_to_dataframe(plan)

    assert set(
        [
            "segment_id",
            "batch_id",
            "mode",
            "freq_hz",
            "cycle_count",
            "target_peak_mT",
            "target_shape",
            "source_waveform_family",
            "variant_type",
            "variant_params_json",
            "pre_idle_s",
            "post_idle_s",
            "sample_rate_hz",
        ]
    ).issubset(frame.columns)
    assert json.loads(frame.loc[0, "variant_params_json"]) == {"gain": 1.0}


def test_ai_sweep_plan_does_not_import_streamlit() -> None:
    import coil_ai_sweep.sweep_plan as sweep_plan

    assert "streamlit" not in inspect.getsource(sweep_plan)
