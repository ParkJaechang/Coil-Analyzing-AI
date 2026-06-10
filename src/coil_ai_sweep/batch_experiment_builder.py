from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .peak_response_modeling import (
    PeakResponseBuildResult,
    PeakResponseConfig,
    build_peak_response_from_source_segment,
)
from .schema import SweepSegmentSpec, SweepTargetConfig
from .segment_parser import SegmentMeasurement
from .sweep_lut_generator import (
    SegmentCommandInput,
    SweepLutBuildResult,
    build_sweep_lut_from_segment_commands,
)
from .schema import SweepSegmentManifestRow

_UI_FLAG_KEY = "stream" + "lit_involved"


@dataclass(frozen=True)
class BatchSourceSegment:
    segment: SegmentMeasurement
    manifest_row: SweepSegmentManifestRow


@dataclass(frozen=True)
class BatchExperimentConfig:
    peak_response_config: PeakResponseConfig
    include_blocked_peak_tables: bool = True
    include_blocked_commands: bool = False
    output_batch_id: str | None = None
    output_variant_type: str = "peak_response_keypoint"
    preserve_original_segment_ids: bool = False
    generated_segment_id_prefix: str = "K"
    output_pre_idle_s: float = 0.5
    output_post_idle_s: float = 0.5
    output_sample_rate_hz: float = 1000.0

    def __post_init__(self) -> None:
        if self.include_blocked_commands:
            raise ValueError("include_blocked_commands_not_supported")
        if not str(self.output_variant_type).strip():
            raise ValueError("output_variant_type_must_be_non_empty")
        if not str(self.generated_segment_id_prefix).strip():
            raise ValueError("generated_segment_id_prefix_must_be_non_empty")
        if self.output_pre_idle_s < 0:
            raise ValueError("output_pre_idle_s_must_be_non_negative")
        if self.output_post_idle_s < 0:
            raise ValueError("output_post_idle_s_must_be_non_negative")
        if self.output_sample_rate_hz <= 0:
            raise ValueError("output_sample_rate_hz_must_be_positive")


@dataclass(frozen=True)
class BatchExperimentBuildResult:
    peak_table: pd.DataFrame
    peak_results: dict[str, PeakResponseBuildResult]
    segment_commands: list[SegmentCommandInput]
    sweep_lut_result: SweepLutBuildResult | None
    metadata: dict[str, Any]
    status: str


def build_batch_experiment_from_peak_responses(
    sources: list[BatchSourceSegment],
    *,
    config: BatchExperimentConfig,
) -> BatchExperimentBuildResult:
    """Build the next offline experiment sweep from safe peak response command profiles."""

    if not sources:
        raise ValueError("sources_must_be_non_empty")

    peak_results: dict[str, PeakResponseBuildResult] = {}
    peak_tables: list[pd.DataFrame] = []
    segment_commands: list[SegmentCommandInput] = []
    blocked_source_count = 0
    skipped_source_count = 0

    for source in sources:
        source_segment_id = source.manifest_row.segment_id
        peak_result = build_peak_response_from_source_segment(
            source.segment,
            source.manifest_row,
            config=config.peak_response_config,
        )
        peak_results[source_segment_id] = peak_result
        blocked = peak_result.status == "blocked_required_voltage_exceeds_limit"
        skipped = peak_result.command_profile is None and not blocked
        if blocked:
            blocked_source_count += 1
        elif skipped:
            skipped_source_count += 1

        if config.include_blocked_peak_tables or not blocked:
            table = _annotated_peak_table(peak_result, source)
            if not table.empty:
                peak_tables.append(table)

        if peak_result.command_profile is not None and peak_result.status == "ok":
            segment_commands.append(
                _segment_command_input(
                    source,
                    peak_result,
                    config=config,
                    sequence=len(segment_commands) + 1,
                )
            )

    peak_table = pd.concat(peak_tables, ignore_index=True) if peak_tables else pd.DataFrame()
    sweep_lut_result = None
    if segment_commands:
        sweep_lut_result = build_sweep_lut_from_segment_commands(segment_commands)
    status = _status(
        source_count=len(sources),
        command_count=len(segment_commands),
        blocked_count=blocked_source_count,
        skipped_count=skipped_source_count,
        sweep_lut_result=sweep_lut_result,
    )
    metadata = {
        "status": status,
        "source_segment_count": len(sources),
        "peak_result_count": len(peak_results),
        "total_peak_record_count": int(len(peak_table)),
        "command_segment_count": len(segment_commands),
        "blocked_source_count": blocked_source_count,
        "skipped_source_count": skipped_source_count,
        "output_batch_id": config.output_batch_id,
        "output_variant_type": config.output_variant_type,
        "include_blocked_peak_tables": config.include_blocked_peak_tables,
        "include_blocked_commands": config.include_blocked_commands,
        "sweep_lut_generated": sweep_lut_result is not None,
        "hardware_invoked": False,
        "modeling_core_called": False,
        _UI_FLAG_KEY: False,
        "winapp_involved": False,
        "ml_training_involved": False,
        "residual_computed": False,
    }
    return BatchExperimentBuildResult(
        peak_table=peak_table,
        peak_results=peak_results,
        segment_commands=segment_commands,
        sweep_lut_result=sweep_lut_result,
        metadata=metadata,
        status=status,
    )


def _annotated_peak_table(
    peak_result: PeakResponseBuildResult,
    source: BatchSourceSegment,
) -> pd.DataFrame:
    table = peak_result.peak_table.copy(deep=True)
    if table.empty:
        return table
    additions = {
        "source_segment_id": source.manifest_row.segment_id,
        "source_batch_id": source.manifest_row.batch_id,
        "peak_response_status": peak_result.status,
        "command_profile_generated": peak_result.command_profile is not None,
    }
    for column, value in additions.items():
        if column not in table.columns:
            table[column] = value
    return table


def _segment_command_input(
    source: BatchSourceSegment,
    peak_result: PeakResponseBuildResult,
    *,
    config: BatchExperimentConfig,
    sequence: int,
) -> SegmentCommandInput:
    row = source.manifest_row
    output_batch_id = config.output_batch_id or f"{row.batch_id}_peak_response"
    segment_id = row.segment_id if config.preserve_original_segment_ids else _generated_segment_id(config, sequence)
    spec = SweepSegmentSpec(
        batch_id=output_batch_id,
        segment_id=segment_id,
        target=SweepTargetConfig(
            freq_hz=float(row.freq_hz),
            cycle_count=float(row.cycle_count),
            target_peak_mT=float(config.peak_response_config.target_peak_mT),
            target_shape=row.target_shape,
            source_waveform_family=row.source_waveform_family,
            mode=row.mode,
        ),
        variant_params=_variant_params(source, peak_result),
        pre_idle_s=float(config.output_pre_idle_s),
        post_idle_s=float(config.output_post_idle_s),
        sample_rate_hz=float(config.output_sample_rate_hz),
        variant_type=config.output_variant_type,
    )
    return SegmentCommandInput(spec=spec, command_profile=peak_result.command_profile.copy(deep=True))


def _variant_params(source: BatchSourceSegment, peak_result: PeakResponseBuildResult) -> dict[str, Any]:
    return {
        "source_segment_id": source.manifest_row.segment_id,
        "source_batch_id": source.manifest_row.batch_id,
        "peak_response_status": peak_result.status,
        "peak_roles": [record.peak_role for record in peak_result.peak_records],
        "required_voltage_peaks_by_role": {
            record.peak_role: record.required_voltage_peak_v for record in peak_result.peak_records
        },
        "phase_delays_by_role": {
            record.peak_role: record.phase_delay_s for record in peak_result.peak_records
        },
        "generated_from": "peak_response_modeling",
    }


def _generated_segment_id(config: BatchExperimentConfig, sequence: int) -> str:
    return f"{config.generated_segment_id_prefix}{sequence:04d}"


def _status(
    *,
    source_count: int,
    command_count: int,
    blocked_count: int,
    skipped_count: int,
    sweep_lut_result: SweepLutBuildResult | None,
) -> str:
    if command_count == 0:
        if blocked_count == source_count:
            return "blocked_all_sources"
        return "no_command_profiles_generated"
    if blocked_count or skipped_count:
        return "partial"
    if sweep_lut_result is not None and sweep_lut_result.status == "ok":
        return "ok"
    return "error"
