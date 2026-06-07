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
