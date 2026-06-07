from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from coil_ai_sweep.manifest_io import (
    dataframe_to_manifest_rows,
    manifest_rows_to_dataframe,
    read_manifest_csv,
    validate_manifest_dataframe,
    write_manifest_csv,
)
from coil_ai_sweep.schema import SweepSegmentManifestRow


def _row(
    *,
    batch_id: str = "batch-a",
    segment_id: str = "seg-001",
    start_sample: int = 0,
    end_sample: int = 199,
    active_start_sample: int = 20,
    active_end_sample: int = 180,
    start_time_s: float = 0.0,
    end_time_s: float = 0.199,
    active_start_time_s: float = 0.020,
    active_end_time_s: float = 0.180,
    freq_hz: float = 5.0,
    cycle_count: float = 1.5,
    target_peak_mT: float = 42.0,
    mode: str = "finite",
    variant_params_json: str = '{"gain": 1.0}',
) -> SweepSegmentManifestRow:
    return SweepSegmentManifestRow(
        batch_id=batch_id,
        segment_id=segment_id,
        start_sample=start_sample,
        end_sample=end_sample,
        active_start_sample=active_start_sample,
        active_end_sample=active_end_sample,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        active_start_time_s=active_start_time_s,
        active_end_time_s=active_end_time_s,
        freq_hz=freq_hz,
        cycle_count=cycle_count,
        target_peak_mT=target_peak_mT,
        target_shape="fixed_rounded_triangle",
        source_waveform_family="triangle",
        mode=mode,
        variant_type="baseline",
        variant_params_json=variant_params_json,
    )


def test_manifest_round_trip_csv(tmp_path: Path) -> None:
    rows = [
        _row(segment_id="seg-001"),
        _row(
            segment_id="seg-002",
            start_sample=200,
            end_sample=399,
            active_start_sample=220,
            active_end_sample=380,
            start_time_s=0.200,
            end_time_s=0.399,
            active_start_time_s=0.220,
            active_end_time_s=0.380,
            freq_hz=7.5,
            cycle_count=1.0,
            target_peak_mT=35.0,
        ),
    ]
    path = tmp_path / "manifest.csv"

    write_manifest_csv(rows, path)
    loaded = read_manifest_csv(path)

    assert loaded == rows


def test_manifest_required_columns() -> None:
    frame = manifest_rows_to_dataframe([_row()]).drop(columns=["target_peak_mT"])

    result = validate_manifest_dataframe(frame)

    assert result.ok is False
    assert "missing_required_columns: target_peak_mT" in result.errors


def test_manifest_rejects_overlapping_segments() -> None:
    frame = manifest_rows_to_dataframe(
        [
            _row(
                segment_id="seg-001",
                start_sample=0,
                end_sample=100,
                active_start_sample=20,
                active_end_sample=80,
            ),
            _row(
                segment_id="seg-002",
                start_sample=100,
                end_sample=200,
                active_start_sample=120,
                active_end_sample=180,
            ),
        ]
    )

    result = validate_manifest_dataframe(frame)

    assert result.ok is False
    assert "overlapping_sample_range: batch-a seg-001 seg-002" in result.errors


def test_manifest_rejects_invalid_active_range() -> None:
    frame = manifest_rows_to_dataframe([_row()])
    frame.loc[0, "active_start_sample"] = 180
    frame.loc[0, "active_end_sample"] = 180

    result = validate_manifest_dataframe(frame)

    assert result.ok is False
    assert "invalid_sample_range: seg-001" in result.errors


def test_manifest_rejects_invalid_json_variant_params() -> None:
    frame = manifest_rows_to_dataframe([_row()])
    frame.loc[0, "variant_params_json"] = '["not", "object"]'

    result = validate_manifest_dataframe(frame)

    assert result.ok is False
    assert "variant_params_json_must_be_object: seg-001" in result.errors


def test_manifest_preserves_user_target_config_fields() -> None:
    original = _row(freq_hz=12.5, cycle_count=1.5, target_peak_mT=27.0)
    frame = pd.DataFrame([original.to_dict()])

    loaded = dataframe_to_manifest_rows(frame)

    assert loaded[0].freq_hz == 12.5
    assert loaded[0].cycle_count == 1.5
    assert loaded[0].target_peak_mT == 27.0
