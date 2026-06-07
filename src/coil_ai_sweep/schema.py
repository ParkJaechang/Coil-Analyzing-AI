from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any

TARGET_SHAPE_FIXED_ROUNDED_TRIANGLE = "fixed_rounded_triangle"
FINITE_CYCLE_COUNTS = {1.0, 1.5}
CONTINUOUS_CYCLE_COUNTS = {1.0}


@dataclass(frozen=True)
class ManifestValidationResult:
    """Validation status for an AI sweep manifest table."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SweepTargetConfig:
    """User target configuration for one AI sweep segment."""

    freq_hz: float
    cycle_count: float
    target_peak_mT: float
    target_shape: str = TARGET_SHAPE_FIXED_ROUNDED_TRIANGLE
    source_waveform_family: str = "triangle"
    mode: str = "finite"

    def __post_init__(self) -> None:
        validate_target_fields(
            freq_hz=self.freq_hz,
            cycle_count=self.cycle_count,
            target_peak_mT=self.target_peak_mT,
            target_shape=self.target_shape,
            mode=self.mode,
        )


@dataclass(frozen=True)
class SweepSegmentSpec:
    """Input contract for planning one long-sweep LUT segment."""

    batch_id: str
    segment_id: str
    target: SweepTargetConfig
    variant_params: dict[str, Any]
    pre_idle_s: float
    post_idle_s: float
    sample_rate_hz: float
    variant_type: str = "baseline"

    def __post_init__(self) -> None:
        if not str(self.batch_id).strip():
            raise ValueError("batch_id_must_be_non_empty")
        if not str(self.segment_id).strip():
            raise ValueError("segment_id_must_be_non_empty")
        if self.pre_idle_s < 0:
            raise ValueError("pre_idle_s_must_be_non_negative")
        if self.post_idle_s < 0:
            raise ValueError("post_idle_s_must_be_non_negative")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz_must_be_positive")
        if not isinstance(self.variant_params, dict):
            raise ValueError("variant_params_must_be_dict")
        _require_json_object_serializable(self.variant_params, "variant_params_must_be_json_serializable")


@dataclass(frozen=True)
class SweepSegmentManifestRow:
    """CSV row describing one segment within a long AI sweep hardware LUT."""

    batch_id: str
    segment_id: str
    start_sample: int
    end_sample: int
    active_start_sample: int
    active_end_sample: int
    start_time_s: float
    end_time_s: float
    active_start_time_s: float
    active_end_time_s: float
    freq_hz: float
    cycle_count: float
    target_peak_mT: float
    target_shape: str
    source_waveform_family: str
    mode: str
    variant_type: str
    variant_params_json: str

    def __post_init__(self) -> None:
        errors = validate_manifest_row_fields(self)
        if errors:
            raise ValueError(errors[0])

    def to_dict(self) -> dict[str, Any]:
        """Return a manifest-row dictionary using stable CSV column names."""

        return asdict(self)


MANIFEST_COLUMNS = [
    "batch_id",
    "segment_id",
    "start_sample",
    "end_sample",
    "active_start_sample",
    "active_end_sample",
    "start_time_s",
    "end_time_s",
    "active_start_time_s",
    "active_end_time_s",
    "freq_hz",
    "cycle_count",
    "target_peak_mT",
    "target_shape",
    "source_waveform_family",
    "mode",
    "variant_type",
    "variant_params_json",
]


def validate_target_fields(
    *,
    freq_hz: float,
    cycle_count: float,
    target_peak_mT: float,
    target_shape: str,
    mode: str,
) -> None:
    """Validate user target fields without inferring values from metadata."""

    if freq_hz <= 0:
        raise ValueError("freq_hz_must_be_positive")
    if target_peak_mT <= 0:
        raise ValueError("target_peak_mT_must_be_positive")
    if target_shape != TARGET_SHAPE_FIXED_ROUNDED_TRIANGLE:
        raise ValueError("target_shape_must_be_fixed_rounded_triangle")
    if mode == "finite":
        if float(cycle_count) not in FINITE_CYCLE_COUNTS:
            raise ValueError("finite_cycle_count_must_be_1p0_or_1p5")
    elif mode == "continuous":
        if float(cycle_count) not in CONTINUOUS_CYCLE_COUNTS:
            raise ValueError("continuous_cycle_count_must_be_1p0")
    else:
        raise ValueError("mode_must_be_finite_or_continuous")


def validate_manifest_row_fields(row: SweepSegmentManifestRow) -> list[str]:
    """Return stable validation errors for a manifest row."""

    errors: list[str] = []
    segment_id = str(row.segment_id)
    if not str(row.batch_id).strip():
        errors.append("batch_id_must_be_non_empty")
    if not segment_id.strip():
        errors.append("segment_id_must_be_non_empty")
    if not (
        row.start_sample
        <= row.active_start_sample
        < row.active_end_sample
        <= row.end_sample
    ):
        errors.append(f"invalid_sample_range: {segment_id}")
    if not (
        row.start_time_s
        <= row.active_start_time_s
        < row.active_end_time_s
        <= row.end_time_s
    ):
        errors.append(f"invalid_time_range: {segment_id}")
    try:
        validate_target_fields(
            freq_hz=float(row.freq_hz),
            cycle_count=float(row.cycle_count),
            target_peak_mT=float(row.target_peak_mT),
            target_shape=str(row.target_shape),
            mode=str(row.mode),
        )
    except ValueError as exc:
        errors.append(f"{exc}: {segment_id}")
    try:
        parsed = json.loads(row.variant_params_json)
    except (TypeError, json.JSONDecodeError):
        errors.append(f"variant_params_json_invalid: {segment_id}")
    else:
        if not isinstance(parsed, dict):
            errors.append(f"variant_params_json_must_be_object: {segment_id}")
    return errors


def _require_json_object_serializable(value: dict[str, Any], error_code: str) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error_code) from exc
