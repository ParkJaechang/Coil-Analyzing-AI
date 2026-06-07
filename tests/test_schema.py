from __future__ import annotations

from pathlib import Path
import inspect
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from coil_ai_sweep.schema import SweepSegmentSpec, SweepTargetConfig


def test_target_config_accepts_finite_1p0_and_1p5() -> None:
    one_cycle = SweepTargetConfig(freq_hz=1.0, cycle_count=1.0, target_peak_mT=50.0)
    one_and_half = SweepTargetConfig(freq_hz=1.0, cycle_count=1.5, target_peak_mT=50.0)

    assert one_cycle.cycle_count == 1.0
    assert one_and_half.cycle_count == 1.5


def test_target_config_rejects_invalid_finite_cycle() -> None:
    with pytest.raises(ValueError, match="finite_cycle_count_must_be_1p0_or_1p5"):
        SweepTargetConfig(freq_hz=1.0, cycle_count=1.25, target_peak_mT=50.0)


def test_target_config_rejects_continuous_1p5() -> None:
    with pytest.raises(ValueError, match="continuous_cycle_count_must_be_1p0"):
        SweepTargetConfig(freq_hz=1.0, cycle_count=1.5, target_peak_mT=50.0, mode="continuous")


def test_target_shape_must_be_fixed_rounded_triangle() -> None:
    with pytest.raises(ValueError, match="target_shape_must_be_fixed_rounded_triangle"):
        SweepTargetConfig(freq_hz=1.0, cycle_count=1.0, target_peak_mT=50.0, target_shape="sine")


def test_segment_spec_requires_positive_sample_rate() -> None:
    target = SweepTargetConfig(freq_hz=1.0, cycle_count=1.0, target_peak_mT=50.0)

    with pytest.raises(ValueError, match="sample_rate_hz_must_be_positive"):
        SweepSegmentSpec(
            batch_id="batch-a",
            segment_id="seg-001",
            target=target,
            variant_params={},
            pre_idle_s=0.0,
            post_idle_s=0.0,
            sample_rate_hz=0.0,
        )


def test_ai_sweep_package_does_not_import_streamlit() -> None:
    import coil_ai_sweep as ai_sweep
    import coil_ai_sweep.manifest_io as manifest_io
    import coil_ai_sweep.schema as schema

    imported_modules = {ai_sweep.__name__, manifest_io.__name__, schema.__name__}
    module_sources = "\n".join(
        [
            inspect.getsource(ai_sweep),
            inspect.getsource(manifest_io),
            inspect.getsource(schema),
        ]
    )

    assert "streamlit" not in module_sources
    assert imported_modules == {
        "coil_ai_sweep",
        "coil_ai_sweep.manifest_io",
        "coil_ai_sweep.schema",
    }
