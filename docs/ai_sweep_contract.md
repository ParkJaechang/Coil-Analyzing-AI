# AI Sweep Contract

## Target Config

- `target_shape` is fixed to `fixed_rounded_triangle`.
- `target_peak_mT` is user configured.
- Finite mode supports `cycle_count` 1.0 and 1.5.
- Continuous mode supports `cycle_count` 1.0 only.
- Source metadata must not overwrite target frequency, cycle count, or peak.

## Sweep Plan

Sweep plans are deterministic grids of target frequency, cycle count, and target peak. Segment IDs are stable human-readable IDs such as `S0001`.

## Hardware Sweep LUT

The hardware LUT schema is exactly:

```text
sample_index,time_s,voltage_v
```

Segment metadata must not be embedded in the hardware LUT.

## Manifest

The manifest stores segment sample/time ranges, target config fields, mode, variant type, and JSON object variant parameters. Rows must not overlap within a batch.

## Segment Parser

The segment parser splits a long measurement DataFrame by manifest support windows. It does not interpolate, smooth, phase-align, normalize measured field, compute residuals, or fill missing support with zeros.

## Segment Alignment / Residual Builder

The segment alignment builder is offline analysis only. It takes one parsed segment and its manifest row, generates a deterministic `fixed_rounded_triangle` target grid, interpolates measured effective field onto that grid, and computes `residual_total_mT` plus peak-normalized `residual_shape_mT` for dataset building.

Phase sync is limited to deterministic offline alignment methods. Missing measured support is represented as `NaN`, not zero-filled. This module does not invoke hardware, does not call production modeling code, and does not perform ML/RL training.

## Shape Metrics / Training Packet

The training packet builder is offline and in-memory only. It summarizes one `SegmentAlignmentResult` over the evaluation mask with total residual, peak-normalized shape residual, and normalized shape residual metrics, then packages those metrics with manifest target fields, variant parameters, selected aligned-frame samples, and safety metadata.

Packets are JSON-safe dictionaries for downstream dataset generation. Non-finite values are represented as `null`, not `NaN`, and the builder does not write files, invoke hardware, call production modeling code, or perform ML/RL training.

## Peak-Centered Source Response Builder

The peak response builder uses real 2Vpp triangle actual-drive source data to extract peak-role field-per-volt responses for offline dataset generation and review. It supports finite `cycle_count` 1.0 roles `positive_peak_1`, `negative_peak_1`, and finite `cycle_count` 1.5 roles `positive_peak_1`, `negative_peak_1`, `positive_peak_2`.

The builder preserves the HallBz convention: `effective_field_mT = -HallBz raw`. If `effective_field_mT` is already present, it is used as-is; if only `hallbz_raw_mT` is present, the sign is inverted before peak detection.

For each detected peak role, it computes measured field peak per voltage peak, phase delay metadata, and the required signed voltage peak for the requested `target_peak_mT`. The initial keypoint command candidate uses the original voltage peak times and does not apply phase lead.

The builder does not invoke hardware, does not connect to WebApp or WinApp, does not call production modeling code, does not compute full residuals, and does not train ML/RL models.

## Voltage Policy

Voltage policy comes from `coil_ai_sweep.core_adapter`. Production integration should use a pinned core dependency through `COIL_ANALYZING_CORE_SRC`. Standalone fallback is marked in metadata.

## HallBz Convention

`effective_field_mT = -HallBz raw`

## Exclusions

- No hardware invocation
- No WebApp or WinApp wiring
- No ML/RL in bootstrap
- No finite/continuous modeling core imports
- No real generated data or model artifacts
