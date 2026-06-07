from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_FALLBACK_VOLTAGE_LIMIT_V = 10.0


def get_voltage_limit_v() -> float:
    """Return the command voltage limit from core when available."""

    metadata = get_voltage_policy_metadata()
    return float(metadata["voltage_limit_v"])


def get_voltage_policy_metadata() -> dict[str, Any]:
    """Return voltage policy metadata with an explicit source marker."""

    core_src = os.environ.get("COIL_ANALYZING_CORE_SRC")
    if core_src:
        core_path = str(Path(core_src).expanduser().resolve())
        inserted = False
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
            inserted = True
        try:
            from field_analysis.voltage_policy import COMMAND_VOLTAGE_LIMIT_V

            return {
                "voltage_limit_v": float(COMMAND_VOLTAGE_LIMIT_V),
                "voltage_policy_source": "core_dependency",
                "core_src": core_path,
            }
        except Exception:
            if inserted:
                try:
                    sys.path.remove(core_path)
                except ValueError:
                    pass

    return {
        "voltage_limit_v": _FALLBACK_VOLTAGE_LIMIT_V,
        "voltage_policy_source": "standalone_fallback",
        "core_src": None,
    }
