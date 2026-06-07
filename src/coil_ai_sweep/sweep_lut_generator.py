from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pandas as pd

from .core_adapter import get_voltage_limit_v, get_voltage_policy_metadata
from .manifest_io import manifest_rows_to_dataframe, validate_manifest_dataframe
from .schema import SweepSegmentManifestRow, SweepSegmentSpec

HARDWARE_LUT_COLUMNS = ["sample_index", "time_s", "voltage_v"]
_UI_FLAG_KEY = "stream" + "lit_involved"


@dataclass(frozen=True)
class SegmentCommandInput:
    """Already-built command samples for one sweep segment."""

    spec: SweepSegmentSpec
    command_profile: pd.DataFrame
    voltage_column: str = "voltage_v"

    def __post_init__(self) -> None:
        _validated_voltage_samples(self.command_profile, self.voltage_column)


@dataclass(frozen=True)
class SweepLutBuildResult:
    """Long sweep LUT plus its segment manifest."""

    lut: pd.DataFrame
    manifest: pd.DataFrame
    manifest_rows: list[SweepSegmentManifestRow]
    metadata: dict[str, Any]
    status: str


def build_sweep_lut_from_segment_commands(
    segments: list[SegmentCommandInput],
) -> SweepLutBuildResult:
    """Join prebuilt segment commands into one hardware LUT and manifest."""

    if not segments:
        raise ValueError("segments_must_be_non_empty")
    sample_rate_hz = float(segments[0].spec.sample_rate_hz)
    for segment in segments:
        if not np.isclose(float(segment.spec.sample_rate_hz), sample_rate_hz):
            raise ValueError("mixed_sample_rate_hz_not_supported")

    voltage_blocks: list[np.ndarray] = []
    manifest_rows: list[SweepSegmentManifestRow] = []
    next_sample = 0
    max_abs_voltage_v = 0.0

    for segment in segments:
        active_voltage = _validated_voltage_samples(segment.command_profile, segment.voltage_column)
        max_abs_voltage_v = max(max_abs_voltage_v, float(np.max(np.abs(active_voltage))))
        pre_count = int(round(float(segment.spec.pre_idle_s) * sample_rate_hz))
        post_count = int(round(float(segment.spec.post_idle_s) * sample_rate_hz))
        segment_voltage = np.concatenate(
            [
                np.zeros(pre_count, dtype=float),
                active_voltage,
                np.zeros(post_count, dtype=float),
            ]
        )
        start_sample = next_sample
        end_sample = start_sample + int(segment_voltage.size) - 1
        active_start_sample = start_sample + pre_count
        active_end_sample = active_start_sample + int(active_voltage.size) - 1
        voltage_blocks.append(segment_voltage)
        manifest_rows.append(
            _manifest_row(
                segment=segment,
                start_sample=start_sample,
                end_sample=end_sample,
                active_start_sample=active_start_sample,
                active_end_sample=active_end_sample,
                sample_rate_hz=sample_rate_hz,
            )
        )
        next_sample = end_sample + 1

    voltage = np.concatenate(voltage_blocks) if voltage_blocks else np.asarray([], dtype=float)
    sample_index = np.arange(voltage.size, dtype=int)
    lut = pd.DataFrame(
        {
            "sample_index": sample_index,
            "time_s": sample_index.astype(float) / sample_rate_hz,
            "voltage_v": voltage,
        },
        columns=HARDWARE_LUT_COLUMNS,
    )
    manifest = manifest_rows_to_dataframe(manifest_rows)
    validation = validate_manifest_dataframe(manifest)
    if not validation.ok:
        raise ValueError("manifest_validation_failed: " + "; ".join(validation.errors))
    voltage_policy = get_voltage_policy_metadata()

    metadata = {
        "status": "ok",
        "segment_count": len(segments),
        "sample_rate_hz": sample_rate_hz,
        "total_sample_count": int(len(lut)),
        "total_duration_s": float(len(lut) / sample_rate_hz),
        "voltage_limit_v": float(voltage_policy["voltage_limit_v"]),
        "voltage_policy_source": voltage_policy["voltage_policy_source"],
        "voltage_policy_metadata": voltage_policy,
        "max_abs_voltage_v": float(max_abs_voltage_v),
        "lut_columns": list(lut.columns),
        "manifest_columns": list(manifest.columns),
        "ai_sweep_lut_generation_mode": "concatenate_prebuilt_segment_commands",
        "modeling_core_called": False,
        _UI_FLAG_KEY: False,
        "hardware_invoked": False,
    }
    return SweepLutBuildResult(
        lut=lut,
        manifest=manifest,
        manifest_rows=manifest_rows,
        metadata=metadata,
        status="ok",
    )


def _validated_voltage_samples(command_profile: pd.DataFrame, voltage_column: str) -> np.ndarray:
    if command_profile.empty:
        raise ValueError("command_profile_must_be_non_empty")
    if "time_s" not in command_profile.columns:
        raise ValueError("time_s_column_required")
    if voltage_column not in command_profile.columns:
        raise ValueError("voltage_column_required")
    time_s = pd.to_numeric(command_profile["time_s"], errors="coerce").to_numpy(dtype=float)
    voltage = pd.to_numeric(command_profile[voltage_column], errors="coerce").to_numpy(dtype=float)
    if len(voltage) < 2:
        raise ValueError("active_command_must_have_at_least_two_samples")
    if not np.isfinite(time_s).all():
        raise ValueError("time_s_must_be_finite")
    if not np.isfinite(voltage).all():
        raise ValueError("voltage_v_must_be_finite")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("time_s_must_be_monotonic_increasing")
    if float(np.max(np.abs(voltage))) > get_voltage_limit_v():
        raise ValueError("voltage_limit_exceeded")
    return voltage


def _manifest_row(
    *,
    segment: SegmentCommandInput,
    start_sample: int,
    end_sample: int,
    active_start_sample: int,
    active_end_sample: int,
    sample_rate_hz: float,
) -> SweepSegmentManifestRow:
    spec = segment.spec
    target = spec.target
    return SweepSegmentManifestRow(
        batch_id=spec.batch_id,
        segment_id=spec.segment_id,
        start_sample=start_sample,
        end_sample=end_sample,
        active_start_sample=active_start_sample,
        active_end_sample=active_end_sample,
        start_time_s=start_sample / sample_rate_hz,
        end_time_s=end_sample / sample_rate_hz,
        active_start_time_s=active_start_sample / sample_rate_hz,
        active_end_time_s=active_end_sample / sample_rate_hz,
        freq_hz=target.freq_hz,
        cycle_count=target.cycle_count,
        target_peak_mT=target.target_peak_mT,
        target_shape=target.target_shape,
        source_waveform_family=target.source_waveform_family,
        mode=target.mode,
        variant_type=spec.variant_type,
        variant_params_json=json.dumps(spec.variant_params, sort_keys=True),
    )
