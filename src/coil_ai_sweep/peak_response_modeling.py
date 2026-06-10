from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .core_adapter import get_voltage_policy_metadata
from .schema import SweepSegmentManifestRow
from .segment_parser import SegmentMeasurement

_UI_FLAG_KEY = "stream" + "lit_involved"
_EPS = 1.0e-12
_ROLE_ANCHORS = {
    "positive_peak_1": (1, 1, 0.25),
    "negative_peak_1": (-1, 1, 0.75),
    "positive_peak_2": (1, 2, 1.25),
}
_PEAK_TABLE_COLUMNS = [
    "segment_id",
    "batch_id",
    "freq_hz",
    "cycle_count",
    "peak_role",
    "peak_polarity",
    "peak_order",
    "target_peak_mT",
    "desired_field_peak_mT",
    "measured_field_peak_mT",
    "input_voltage_peak_v",
    "signed_field_per_volt_mT_per_v",
    "abs_field_per_volt_mT_per_v",
    "required_voltage_peak_v",
    "required_voltage_gain",
    "measured_peak_time_s",
    "voltage_peak_time_s",
    "phase_delay_s",
    "phase_delay_cycles",
    "voltage_limit_status",
]


@dataclass(frozen=True)
class PeakResponseConfig:
    target_peak_mT: float
    source_voltage_vpp: float = 2.0
    source_voltage_peak_v: float | None = None
    measured_field_column: str = "effective_field_mT"
    voltage_column: str = "measured_voltage_v"
    phase_sync_method: str = "peak_role"
    required_voltage_headroom_ratio: float = 1.0
    generate_keypoint_command: bool = True
    keypoint_command_endpoint_mode: str = "endpoint_exclusive"
    keypoint_command_sample_rate_hz: float | None = None
    keypoint_rounding_fraction: float = 0.04

    def __post_init__(self) -> None:
        if self.target_peak_mT <= 0:
            raise ValueError("target_peak_mT_must_be_positive")
        if self.source_voltage_vpp <= 0:
            raise ValueError("source_voltage_vpp_must_be_positive")
        if self.source_voltage_peak_v is not None and self.source_voltage_peak_v <= 0:
            raise ValueError("source_voltage_peak_v_must_be_positive")
        if self.required_voltage_headroom_ratio <= 0 or self.required_voltage_headroom_ratio > 1:
            raise ValueError("required_voltage_headroom_ratio_must_be_gt_0_and_lte_1")
        if self.phase_sync_method not in {"peak_role", "none"}:
            raise ValueError("phase_sync_method_must_be_peak_role_or_none")
        if self.keypoint_command_endpoint_mode != "endpoint_exclusive":
            raise ValueError("keypoint_command_endpoint_mode_must_be_endpoint_exclusive")
        if self.keypoint_rounding_fraction < 0 or self.keypoint_rounding_fraction >= 0.25:
            raise ValueError("keypoint_rounding_fraction_must_be_gte_0_and_lt_0p25")


@dataclass(frozen=True)
class PeakResponseRecord:
    peak_role: str
    peak_polarity: int
    peak_order: int
    target_peak_mT: float
    measured_field_peak_mT: float
    input_voltage_peak_v: float
    signed_field_per_volt_mT_per_v: float
    abs_field_per_volt_mT_per_v: float
    required_voltage_peak_v: float
    required_voltage_gain: float
    measured_peak_time_s: float
    voltage_peak_time_s: float
    phase_delay_s: float
    phase_delay_cycles: float
    voltage_limit_status: str


@dataclass(frozen=True)
class PeakResponseBuildResult:
    peak_records: list[PeakResponseRecord]
    peak_table: pd.DataFrame
    command_profile: pd.DataFrame | None
    metadata: dict[str, Any]
    status: str


def build_peak_response_from_source_segment(
    segment: SegmentMeasurement,
    manifest_row: SweepSegmentManifestRow,
    *,
    config: PeakResponseConfig,
) -> PeakResponseBuildResult:
    """Build offline peak-role field-per-volt response records from one source segment."""

    voltage_policy = get_voltage_policy_metadata()
    voltage_limit_v = float(voltage_policy["voltage_limit_v"])
    roles = _expected_roles(float(manifest_row.cycle_count))
    if roles is None:
        metadata = _base_metadata(segment, manifest_row, config, voltage_policy, voltage_peak_source=None)
        metadata.update(
            {
                "status": "unsupported_cycle_count_for_peak_response",
                "expected_peak_roles": [],
                "detected_peak_count": 0,
                "skipped_peak_roles": [],
                "command_profile_generated": False,
                "keypoint_phase_lead_applied": False,
            }
        )
        return PeakResponseBuildResult([], _empty_peak_table(), None, metadata, metadata["status"])

    frame = segment.frame.copy(deep=True)
    field, hallbz_convention = _effective_field(frame, config)
    local_time = _local_time(frame)
    voltage = _voltage(frame, config)
    source_voltage_peak_v = float(config.source_voltage_peak_v or config.source_voltage_vpp / 2.0)
    voltage_peak_source = "measured_voltage_column" if voltage is not None else "nominal_source_voltage_peak"

    records: list[PeakResponseRecord] = []
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    warnings: list[str] = []
    freq_hz = float(manifest_row.freq_hz)
    allowed_voltage_v = voltage_limit_v * float(config.required_voltage_headroom_ratio)

    for role in roles:
        polarity, peak_order, anchor = _ROLE_ANCHORS[role]
        field_peak = _detect_peak(
            local_time,
            field,
            freq_hz=freq_hz,
            anchor_cycle=anchor,
            polarity=polarity,
        )
        if field_peak is None:
            skipped.append(role)
            warnings.append(f"field_peak_not_detected: {role}")
            continue
        if voltage is None:
            voltage_peak_value = polarity * source_voltage_peak_v
            voltage_peak_time_s = anchor / freq_hz
        else:
            voltage_peak = _detect_peak(
                local_time,
                voltage,
                freq_hz=freq_hz,
                anchor_cycle=anchor,
                polarity=polarity,
            )
            if voltage_peak is None:
                skipped.append(role)
                warnings.append(f"voltage_peak_not_detected: {role}")
                continue
            voltage_peak_time_s, voltage_peak_value = voltage_peak

        measured_peak_time_s, measured_field_peak_mT = field_peak
        signed_per_volt = measured_field_peak_mT / voltage_peak_value if abs(voltage_peak_value) > _EPS else math.nan
        abs_per_volt = abs(measured_field_peak_mT) / max(abs(voltage_peak_value), _EPS)
        desired_field_peak_mT = float(config.target_peak_mT) * polarity
        required_voltage_peak_v = desired_field_peak_mT / signed_per_volt
        required_voltage_gain = required_voltage_peak_v / voltage_peak_value
        phase_delay_s = measured_peak_time_s - voltage_peak_time_s
        voltage_limit_status = (
            "exceeds_headroom_limit" if abs(required_voltage_peak_v) > allowed_voltage_v else "ok"
        )
        record = PeakResponseRecord(
            peak_role=role,
            peak_polarity=polarity,
            peak_order=peak_order,
            target_peak_mT=float(config.target_peak_mT),
            measured_field_peak_mT=float(measured_field_peak_mT),
            input_voltage_peak_v=float(voltage_peak_value),
            signed_field_per_volt_mT_per_v=float(signed_per_volt),
            abs_field_per_volt_mT_per_v=float(abs_per_volt),
            required_voltage_peak_v=float(required_voltage_peak_v),
            required_voltage_gain=float(required_voltage_gain),
            measured_peak_time_s=float(measured_peak_time_s),
            voltage_peak_time_s=float(voltage_peak_time_s),
            phase_delay_s=float(phase_delay_s),
            phase_delay_cycles=float(phase_delay_s * freq_hz),
            voltage_limit_status=voltage_limit_status,
        )
        records.append(record)
        row = {
            "segment_id": manifest_row.segment_id,
            "batch_id": manifest_row.batch_id,
            "freq_hz": freq_hz,
            "cycle_count": float(manifest_row.cycle_count),
            "desired_field_peak_mT": desired_field_peak_mT,
        }
        row.update(asdict(record))
        rows.append(row)

    peak_table = pd.DataFrame(rows, columns=_PEAK_TABLE_COLUMNS)
    blocked = any(record.voltage_limit_status != "ok" for record in records)
    status = "blocked_required_voltage_exceeds_limit" if blocked else "ok"
    command_profile = None
    command_generated = False
    keypoint_phase_lead_applied = False
    if config.generate_keypoint_command and records and not blocked:
        command_profile = _build_keypoint_command_profile(
            records,
            manifest_row,
            local_time,
            sample_rate_hz=config.keypoint_command_sample_rate_hz,
        )
        command_generated = command_profile is not None

    metadata = _base_metadata(segment, manifest_row, config, voltage_policy, voltage_peak_source=voltage_peak_source)
    metadata.update(
        {
            "status": status,
            "source_voltage_peak_v": source_voltage_peak_v,
            "expected_peak_roles": roles,
            "detected_peak_count": len(records),
            "skipped_peak_roles": skipped,
            "phase_sync_method": config.phase_sync_method,
            "hallbz_convention": hallbz_convention,
            "command_profile_generated": command_generated,
            "keypoint_phase_lead_applied": keypoint_phase_lead_applied,
            "warnings": warnings,
        }
    )
    if command_profile is None and config.generate_keypoint_command and records and not blocked:
        metadata["warnings"].append("keypoint_command_sample_rate_fallback_used")
    return PeakResponseBuildResult(records, peak_table, command_profile, metadata, status)


def _expected_roles(cycle_count: float) -> list[str] | None:
    if math.isclose(cycle_count, 1.0, abs_tol=1.0e-9):
        return ["positive_peak_1", "negative_peak_1"]
    if math.isclose(cycle_count, 1.5, abs_tol=1.0e-9):
        return ["positive_peak_1", "negative_peak_1", "positive_peak_2"]
    return None


def _effective_field(frame: pd.DataFrame, config: PeakResponseConfig) -> tuple[np.ndarray, str]:
    if config.measured_field_column in frame.columns:
        values = pd.to_numeric(frame[config.measured_field_column], errors="coerce").to_numpy(dtype=float)
        return values, "effective_field_mT"
    if "hallbz_raw_mT" in frame.columns:
        values = -pd.to_numeric(frame["hallbz_raw_mT"], errors="coerce").to_numpy(dtype=float)
        return values, "effective_field_mT = -HallBz raw"
    raise ValueError("measured_field_column_not_found")


def _voltage(frame: pd.DataFrame, config: PeakResponseConfig) -> np.ndarray | None:
    if config.voltage_column not in frame.columns:
        return None
    return pd.to_numeric(frame[config.voltage_column], errors="coerce").to_numpy(dtype=float)


def _local_time(frame: pd.DataFrame) -> np.ndarray:
    if "active_local_time_s" not in frame.columns:
        raise ValueError("active_local_time_s_column_not_found")
    return pd.to_numeric(frame["active_local_time_s"], errors="coerce").to_numpy(dtype=float)


def _detect_peak(
    local_time_s: np.ndarray,
    values: np.ndarray,
    *,
    freq_hz: float,
    anchor_cycle: float,
    polarity: int,
) -> tuple[float, float] | None:
    cycle_position = local_time_s * freq_hz
    mask = (cycle_position >= anchor_cycle - 0.20) & (cycle_position <= anchor_cycle + 0.20)
    finite = mask & np.isfinite(local_time_s) & np.isfinite(values)
    if not bool(finite.any()):
        return None
    candidate_indices = np.flatnonzero(finite)
    candidate_values = values[candidate_indices]
    selected_offset = int(np.argmax(candidate_values) if polarity > 0 else np.argmin(candidate_values))
    selected_index = int(candidate_indices[selected_offset])
    return float(local_time_s[selected_index]), float(values[selected_index])


def _build_keypoint_command_profile(
    records: list[PeakResponseRecord],
    manifest_row: SweepSegmentManifestRow,
    local_time_s: np.ndarray,
    *,
    sample_rate_hz: float | None,
) -> pd.DataFrame | None:
    freq_hz = float(manifest_row.freq_hz)
    active_duration_s = float(manifest_row.cycle_count) / freq_hz
    sample_rate = sample_rate_hz or _infer_sample_rate_hz(local_time_s) or 1000.0
    n_samples = max(1, int(math.ceil(active_duration_s * sample_rate)))
    time_s = np.arange(n_samples, dtype=float) / sample_rate
    time_s = time_s[time_s < active_duration_s]
    if len(time_s) == 0:
        return None
    keypoint_times = [0.0]
    keypoint_voltages = [0.0]
    for record in records:
        keypoint_times.append(record.voltage_peak_time_s)
        keypoint_voltages.append(record.required_voltage_peak_v)
    keypoint_times.append(active_duration_s)
    keypoint_voltages.append(0.0)
    order = np.argsort(np.asarray(keypoint_times, dtype=float))
    sorted_times = np.asarray(keypoint_times, dtype=float)[order]
    sorted_voltages = np.asarray(keypoint_voltages, dtype=float)[order]
    unique_times, unique_indices = np.unique(sorted_times, return_index=True)
    unique_voltages = sorted_voltages[unique_indices]
    voltage_v = np.interp(time_s, unique_times, unique_voltages)
    return pd.DataFrame({"time_s": time_s, "voltage_v": voltage_v})


def _infer_sample_rate_hz(local_time_s: np.ndarray) -> float | None:
    finite = local_time_s[np.isfinite(local_time_s)]
    if len(finite) < 2:
        return None
    diffs = np.diff(np.sort(finite))
    positive = diffs[diffs > 0]
    if len(positive) == 0:
        return None
    dt = float(np.median(positive))
    return 1.0 / dt if dt > 0 else None


def _base_metadata(
    segment: SegmentMeasurement,
    manifest_row: SweepSegmentManifestRow,
    config: PeakResponseConfig,
    voltage_policy: dict[str, Any],
    *,
    voltage_peak_source: str | None,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "segment_id": segment.segment_id,
        "batch_id": segment.batch_id,
        "freq_hz": float(manifest_row.freq_hz),
        "cycle_count": float(manifest_row.cycle_count),
        "target_peak_mT": float(config.target_peak_mT),
        "source_voltage_vpp": float(config.source_voltage_vpp),
        "source_voltage_peak_v": float(config.source_voltage_peak_v or config.source_voltage_vpp / 2.0),
        "voltage_peak_source": voltage_peak_source,
        "expected_peak_roles": [],
        "detected_peak_count": 0,
        "skipped_peak_roles": [],
        "phase_sync_method": config.phase_sync_method,
        "hallbz_convention": None,
        "field_sign_convention_preserved": True,
        "command_profile_generated": False,
        "keypoint_phase_lead_applied": False,
        "voltage_limit_v": float(voltage_policy["voltage_limit_v"]),
        "voltage_policy_source": voltage_policy["voltage_policy_source"],
        "hardware_invoked": False,
        "modeling_core_called": False,
        _UI_FLAG_KEY: False,
        "winapp_involved": False,
        "ml_training_involved": False,
        "residual_computed": False,
    }


def _empty_peak_table() -> pd.DataFrame:
    return pd.DataFrame(columns=_PEAK_TABLE_COLUMNS)
