# Core Dependency

Intended core repository: `ParkJaechang/Coil-Analyzing`

Latest reviewed core SHA: `d68ed6a7b06f20a59b3522ede08746944261ec58`

Set `COIL_ANALYZING_CORE_SRC` to the core repository `src` directory when running against the production core package.

When `field_analysis.voltage_policy` is importable from that path, `coil_ai_sweep.core_adapter` uses `COMMAND_VOLTAGE_LIMIT_V` from core and reports `voltage_policy_source = "core_dependency"`.

When core is unavailable, tests may use the standalone fallback. The fallback is explicitly reported as `voltage_policy_source = "standalone_fallback"`.

Core/source metadata must not overwrite user target config. Frequency, cycle count, target peak, target shape, waveform family, and mode remain user target fields.
