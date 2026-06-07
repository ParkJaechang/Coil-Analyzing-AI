from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .schema import (
    MANIFEST_COLUMNS,
    ManifestValidationResult,
    SweepSegmentManifestRow,
)


def manifest_rows_to_dataframe(rows: list[SweepSegmentManifestRow]) -> pd.DataFrame:
    """Convert validated manifest rows to a stable-column DataFrame."""

    return pd.DataFrame([row.to_dict() for row in rows], columns=MANIFEST_COLUMNS)


def dataframe_to_manifest_rows(df: pd.DataFrame) -> list[SweepSegmentManifestRow]:
    """Validate and convert a manifest DataFrame to row dataclasses."""

    result = validate_manifest_dataframe(df)
    if not result.ok:
        raise ValueError("; ".join(result.errors))
    rows = []
    for record in df.sort_values(["batch_id", "start_sample"]).to_dict(orient="records"):
        rows.append(_manifest_row_from_record(record))
    return rows


def write_manifest_csv(rows: list[SweepSegmentManifestRow], path: str | Path) -> None:
    """Write AI sweep manifest rows as CSV."""

    manifest_rows_to_dataframe(rows).to_csv(Path(path), index=False)


def read_manifest_csv(path: str | Path) -> list[SweepSegmentManifestRow]:
    """Read, validate, and return AI sweep manifest rows from CSV."""

    return dataframe_to_manifest_rows(pd.read_csv(Path(path)))


def validate_manifest_dataframe(df: pd.DataFrame) -> ManifestValidationResult:
    """Validate required columns, row contracts, sorting, and batch overlaps."""

    errors: list[str] = []
    warnings: list[str] = []
    missing = [column for column in MANIFEST_COLUMNS if column not in df.columns]
    if missing:
        return ManifestValidationResult(
            ok=False,
            errors=["missing_required_columns: " + ", ".join(missing)],
            warnings=[],
        )

    normalized_rows: list[SweepSegmentManifestRow] = []
    for index, record in enumerate(df[MANIFEST_COLUMNS].to_dict(orient="records")):
        row_errors, row = _coerce_manifest_record(record, index)
        errors.extend(row_errors)
        if row is not None:
            normalized_rows.append(row)

    if normalized_rows:
        starts = [row.start_sample for row in normalized_rows]
        if starts != sorted(starts):
            warnings.append("manifest_rows_not_sorted_by_start_sample")
        errors.extend(_validate_batch_sample_overlaps(normalized_rows))

    return ManifestValidationResult(ok=not errors, errors=errors, warnings=warnings)


def _coerce_manifest_record(
    record: dict[str, Any],
    index: int,
) -> tuple[list[str], SweepSegmentManifestRow | None]:
    errors: list[str] = []
    segment_id = str(record.get("segment_id", f"row_{index}"))
    integer_fields = ("start_sample", "end_sample", "active_start_sample", "active_end_sample")
    float_fields = (
        "start_time_s",
        "end_time_s",
        "active_start_time_s",
        "active_end_time_s",
        "freq_hz",
        "cycle_count",
        "target_peak_mT",
    )
    coerced = dict(record)
    for field_name in integer_fields:
        value = coerced[field_name]
        if not _is_integer_like(value):
            errors.append(f"{field_name}_must_be_integer_like: {segment_id}")
        else:
            coerced[field_name] = int(float(value))
    for field_name in float_fields:
        try:
            coerced[field_name] = float(coerced[field_name])
        except (TypeError, ValueError):
            errors.append(f"{field_name}_must_be_numeric: {segment_id}")
    if errors:
        return errors, None
    try:
        row = _manifest_row_from_record(coerced)
    except ValueError as exc:
        return [str(exc)], None
    return [], row


def _manifest_row_from_record(record: dict[str, Any]) -> SweepSegmentManifestRow:
    return SweepSegmentManifestRow(
        batch_id=str(record["batch_id"]),
        segment_id=str(record["segment_id"]),
        start_sample=int(record["start_sample"]),
        end_sample=int(record["end_sample"]),
        active_start_sample=int(record["active_start_sample"]),
        active_end_sample=int(record["active_end_sample"]),
        start_time_s=float(record["start_time_s"]),
        end_time_s=float(record["end_time_s"]),
        active_start_time_s=float(record["active_start_time_s"]),
        active_end_time_s=float(record["active_end_time_s"]),
        freq_hz=float(record["freq_hz"]),
        cycle_count=float(record["cycle_count"]),
        target_peak_mT=float(record["target_peak_mT"]),
        target_shape=str(record["target_shape"]),
        source_waveform_family=str(record["source_waveform_family"]),
        mode=str(record["mode"]),
        variant_type=str(record["variant_type"]),
        variant_params_json=str(record["variant_params_json"]),
    )


def _is_integer_like(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return numeric.is_integer()


def _validate_batch_sample_overlaps(rows: list[SweepSegmentManifestRow]) -> list[str]:
    errors: list[str] = []
    by_batch: dict[str, list[SweepSegmentManifestRow]] = {}
    for row in rows:
        by_batch.setdefault(row.batch_id, []).append(row)
    for batch_id, batch_rows in by_batch.items():
        sorted_rows = sorted(batch_rows, key=lambda row: row.start_sample)
        previous: SweepSegmentManifestRow | None = None
        for row in sorted_rows:
            if previous is not None and row.start_sample <= previous.end_sample:
                errors.append(
                    f"overlapping_sample_range: {batch_id} {previous.segment_id} {row.segment_id}"
                )
            previous = row
    return errors
