from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any

import pandas as pd

from .schema import SweepSegmentSpec, SweepTargetConfig


@dataclass(frozen=True)
class SweepPlanConfig:
    """User-provided grid for deterministic experimental AI sweep planning."""

    batch_id: str
    frequency_grid_hz: list[float]
    cycle_counts: list[float]
    target_peak_mT_values: list[float]
    sample_rate_hz: float
    pre_idle_s: float
    post_idle_s: float
    mode: str = "finite"
    source_waveform_family: str = "triangle"
    variant_type: str = "baseline"
    variant_params: dict[str, Any] = field(default_factory=dict)
    include_anchor_repeats: bool = False
    anchor_peak_mT: float | None = None


PLAN_COLUMNS = [
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


def build_sweep_plan(config: SweepPlanConfig) -> list[SweepSegmentSpec]:
    """Build validated sweep segment specs from sorted user target grids."""

    _validate_plan_config(config)
    sorted_frequencies = sorted(float(value) for value in config.frequency_grid_hz)
    sorted_cycles = sorted(float(value) for value in config.cycle_counts)
    sorted_peaks = sorted(float(value) for value in config.target_peak_mT_values)
    plan: list[SweepSegmentSpec] = []

    for freq_hz in sorted_frequencies:
        for cycle_count in sorted_cycles:
            for target_peak_mT in sorted_peaks:
                plan.append(
                    _make_segment(
                        config=config,
                        segment_index=len(plan) + 1,
                        freq_hz=freq_hz,
                        cycle_count=cycle_count,
                        target_peak_mT=target_peak_mT,
                        variant_type=config.variant_type,
                    )
                )
            if config.include_anchor_repeats:
                plan.append(
                    _make_segment(
                        config=config,
                        segment_index=len(plan) + 1,
                        freq_hz=freq_hz,
                        cycle_count=cycle_count,
                        target_peak_mT=float(config.anchor_peak_mT),
                        variant_type="anchor_repeat",
                    )
                )
    return plan


def plan_to_dataframe(plan: list[SweepSegmentSpec]) -> pd.DataFrame:
    """Convert sweep plan specs to a review/debug DataFrame."""

    records = []
    for segment in plan:
        records.append(
            {
                "segment_id": segment.segment_id,
                "batch_id": segment.batch_id,
                "mode": segment.target.mode,
                "freq_hz": segment.target.freq_hz,
                "cycle_count": segment.target.cycle_count,
                "target_peak_mT": segment.target.target_peak_mT,
                "target_shape": segment.target.target_shape,
                "source_waveform_family": segment.target.source_waveform_family,
                "variant_type": segment.variant_type,
                "variant_params_json": json.dumps(segment.variant_params, sort_keys=True),
                "pre_idle_s": segment.pre_idle_s,
                "post_idle_s": segment.post_idle_s,
                "sample_rate_hz": segment.sample_rate_hz,
            }
        )
    return pd.DataFrame(records, columns=PLAN_COLUMNS)


def _validate_plan_config(config: SweepPlanConfig) -> None:
    if not str(config.batch_id).strip():
        raise ValueError("batch_id_must_be_non_empty")
    if not config.frequency_grid_hz:
        raise ValueError("frequency_grid_hz_must_be_non_empty")
    if not config.cycle_counts:
        raise ValueError("cycle_counts_must_be_non_empty")
    if not config.target_peak_mT_values:
        raise ValueError("target_peak_mT_values_must_be_non_empty")
    if config.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz_must_be_positive")
    if config.pre_idle_s < 0:
        raise ValueError("pre_idle_s_must_be_non_negative")
    if config.post_idle_s < 0:
        raise ValueError("post_idle_s_must_be_non_negative")
    if not isinstance(config.variant_params, dict):
        raise ValueError("variant_params_must_be_dict")
    if config.include_anchor_repeats:
        if config.anchor_peak_mT is None:
            raise ValueError("anchor_peak_mT_must_be_provided")
        if config.anchor_peak_mT <= 0:
            raise ValueError("anchor_peak_mT_must_be_positive")


def _make_segment(
    *,
    config: SweepPlanConfig,
    segment_index: int,
    freq_hz: float,
    cycle_count: float,
    target_peak_mT: float,
    variant_type: str,
) -> SweepSegmentSpec:
    target = SweepTargetConfig(
        freq_hz=freq_hz,
        cycle_count=cycle_count,
        target_peak_mT=target_peak_mT,
        source_waveform_family=config.source_waveform_family,
        mode=config.mode,
    )
    return SweepSegmentSpec(
        batch_id=config.batch_id,
        segment_id=f"S{segment_index:04d}",
        target=target,
        variant_type=variant_type,
        variant_params=deepcopy(config.variant_params),
        pre_idle_s=config.pre_idle_s,
        post_idle_s=config.post_idle_s,
        sample_rate_hz=config.sample_rate_hz,
    )
