from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .batch_experiment_builder import BatchExperimentBuildResult
from .sweep_lut_generator import HARDWARE_LUT_COLUMNS, SegmentCommandInput

_SCHEMA_VERSION = "1.0"
_PACKET_TYPE = "source_response_batch_experiment"
_CREATED_BY = "coil_ai_sweep.batch_dataset_export"
_UI_FLAG_KEY = "stream" + "lit_involved"


@dataclass(frozen=True)
class BatchDatasetExportConfig:
    dataset_id: str
    source_batch_label: str | None = None
    include_peak_table_records: bool = True
    include_generated_command_summaries: bool = True
    include_sweep_lut_summary: bool = True
    include_blocked_sources: bool = True
    include_full_lut_samples: bool = False
    max_lut_samples_inline: int = 0

    def __post_init__(self) -> None:
        if not str(self.dataset_id).strip():
            raise ValueError("dataset_id_must_be_non_empty")
        if self.max_lut_samples_inline < 0:
            raise ValueError("max_lut_samples_inline_must_be_non_negative")


@dataclass(frozen=True)
class BatchDatasetExportResult:
    packet: dict[str, Any]
    metadata: dict[str, Any]
    status: str


def build_batch_dataset_export_packet(
    batch_result: BatchExperimentBuildResult,
    *,
    config: BatchDatasetExportConfig,
) -> BatchDatasetExportResult:
    """Build an in-memory JSON-safe packet for source-response batch experiment exports."""

    omitted_sections: list[str] = []
    peak_records = _peak_records(batch_result, config, omitted_sections)
    command_segments = _generated_command_segments(batch_result, config, omitted_sections)
    sweep_lut_summary = _sweep_lut_summary(batch_result, config, omitted_sections)
    blocked_sources = _blocked_sources(batch_result, config, omitted_sections)
    full_lut_samples = _full_lut_samples(batch_result, config)
    full_lut_samples_included = full_lut_samples is not None

    packet: dict[str, Any] = {
        "dataset_id": config.dataset_id,
        "source_batch_label": config.source_batch_label,
        "packet_type": _PACKET_TYPE,
        "schema_version": _SCHEMA_VERSION,
        "batch_status": batch_result.status,
        "created_by": _CREATED_BY,
        "summary": _summary(batch_result),
        "peak_records": peak_records,
        "generated_command_segments": command_segments,
        "sweep_lut_summary": sweep_lut_summary,
        "blocked_sources": blocked_sources,
        "safety": {
            "hardware_invoked": False,
            "modeling_core_called": False,
            _UI_FLAG_KEY: False,
            "winapp_involved": False,
            "ml_training_involved": False,
            "file_written": False,
            "generated_artifact_committed": False,
            "blocked_commands_included": False,
            "full_lut_samples_included": full_lut_samples_included,
        },
    }
    if full_lut_samples_included:
        packet["full_lut_samples"] = full_lut_samples

    metadata = {
        "status": "ok",
        "dataset_id": config.dataset_id,
        "schema_version": _SCHEMA_VERSION,
        "packet_type": _PACKET_TYPE,
        "json_safe": True,
        "peak_records_count": len(peak_records),
        "generated_command_segment_count": len(command_segments),
        "blocked_sources_count": len(blocked_sources),
        "full_lut_samples_included": full_lut_samples_included,
        "full_lut_sample_count": len(full_lut_samples) if full_lut_samples is not None else 0,
        "omitted_sections": omitted_sections,
    }
    return BatchDatasetExportResult(
        packet=_json_safe(packet),
        metadata=_json_safe(metadata),
        status="ok",
    )


def _summary(batch_result: BatchExperimentBuildResult) -> dict[str, Any]:
    metadata = batch_result.metadata
    return {
        "source_segment_count": metadata.get("source_segment_count", 0),
        "peak_result_count": metadata.get("peak_result_count", len(batch_result.peak_results)),
        "total_peak_record_count": metadata.get("total_peak_record_count", len(batch_result.peak_table)),
        "command_segment_count": metadata.get("command_segment_count", len(batch_result.segment_commands)),
        "blocked_source_count": metadata.get("blocked_source_count", 0),
        "skipped_source_count": metadata.get("skipped_source_count", 0),
        "sweep_lut_generated": metadata.get("sweep_lut_generated", batch_result.sweep_lut_result is not None),
        "output_batch_id": metadata.get("output_batch_id"),
        "output_variant_type": metadata.get("output_variant_type"),
        "batch_result_status": batch_result.status,
    }


def _peak_records(
    batch_result: BatchExperimentBuildResult,
    config: BatchDatasetExportConfig,
    omitted_sections: list[str],
) -> list[dict[str, Any]]:
    if not config.include_peak_table_records:
        omitted_sections.append("peak_records")
        return []
    if batch_result.peak_table.empty:
        return []
    return _records(batch_result.peak_table)


def _generated_command_segments(
    batch_result: BatchExperimentBuildResult,
    config: BatchDatasetExportConfig,
    omitted_sections: list[str],
) -> list[dict[str, Any]]:
    if not config.include_generated_command_summaries:
        omitted_sections.append("generated_command_segments")
        return []
    return [_command_summary(command) for command in batch_result.segment_commands]


def _command_summary(command: SegmentCommandInput) -> dict[str, Any]:
    spec = command.spec
    target = spec.target
    frame = command.command_profile
    time_s = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    voltage_v = pd.to_numeric(frame[command.voltage_column], errors="coerce").to_numpy(dtype=float)
    return {
        "batch_id": spec.batch_id,
        "segment_id": spec.segment_id,
        "freq_hz": float(target.freq_hz),
        "cycle_count": float(target.cycle_count),
        "target_peak_mT": float(target.target_peak_mT),
        "target_shape": target.target_shape,
        "source_waveform_family": target.source_waveform_family,
        "mode": target.mode,
        "variant_type": spec.variant_type,
        "variant_params": spec.variant_params,
        "sample_count": int(len(frame)),
        "time_start_s": _nan_safe_min(time_s),
        "time_end_s": _nan_safe_max(time_s),
        "voltage_min_v": _nan_safe_min(voltage_v),
        "voltage_max_v": _nan_safe_max(voltage_v),
        "voltage_peak_abs_v": _nan_safe_max(np.abs(voltage_v)),
    }


def _sweep_lut_summary(
    batch_result: BatchExperimentBuildResult,
    config: BatchDatasetExportConfig,
    omitted_sections: list[str],
) -> dict[str, Any]:
    if not config.include_sweep_lut_summary:
        omitted_sections.append("sweep_lut_summary")
        return {"status": "omitted"}
    if batch_result.sweep_lut_result is None:
        return {"status": "not_generated"}
    result = batch_result.sweep_lut_result
    metadata = result.metadata
    lut_columns = list(result.lut.columns)
    return {
        "status": result.status,
        "total_sample_count": metadata.get("total_sample_count", len(result.lut)),
        "total_duration_s": metadata.get("total_duration_s"),
        "sample_rate_hz": metadata.get("sample_rate_hz"),
        "max_abs_voltage_v": metadata.get("max_abs_voltage_v"),
        "voltage_limit_v": metadata.get("voltage_limit_v"),
        "voltage_policy_source": metadata.get("voltage_policy_source"),
        "lut_columns": lut_columns,
        "manifest_columns": list(result.manifest.columns),
        "hardware_lut_schema_ok": lut_columns == HARDWARE_LUT_COLUMNS,
        "segment_count": metadata.get("segment_count", len(result.manifest_rows)),
    }


def _blocked_sources(
    batch_result: BatchExperimentBuildResult,
    config: BatchDatasetExportConfig,
    omitted_sections: list[str],
) -> list[dict[str, Any]]:
    if not config.include_blocked_sources:
        omitted_sections.append("blocked_sources")
        return []
    blocked: list[dict[str, Any]] = []
    for source_segment_id, peak_result in batch_result.peak_results.items():
        command_generated = peak_result.command_profile is not None
        if peak_result.status == "ok" and command_generated:
            continue
        blocked.append(
            {
                "source_segment_id": source_segment_id,
                "peak_response_status": peak_result.status,
                "command_profile_generated": command_generated,
                "detected_peak_count": peak_result.metadata.get("detected_peak_count", len(peak_result.peak_records)),
                "skipped_peak_roles": peak_result.metadata.get("skipped_peak_roles", []),
                "warnings": peak_result.metadata.get("warnings", []),
            }
        )
    return blocked


def _full_lut_samples(
    batch_result: BatchExperimentBuildResult,
    config: BatchDatasetExportConfig,
) -> list[dict[str, Any]] | None:
    if not config.include_full_lut_samples:
        return None
    if batch_result.sweep_lut_result is None:
        return []
    lut = batch_result.sweep_lut_result.lut
    if len(lut) > config.max_lut_samples_inline:
        raise ValueError("full_lut_samples_exceed_inline_limit")
    return _records(lut.loc[:, HARDWARE_LUT_COLUMNS])


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(record) for record in frame.to_dict(orient="records")]


def _nan_safe_min(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if len(finite) else None


def _nan_safe_max(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if len(finite) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
