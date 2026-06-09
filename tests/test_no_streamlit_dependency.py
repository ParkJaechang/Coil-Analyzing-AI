from __future__ import annotations

import importlib
import inspect
import os
import sys

from coil_ai_sweep.core_adapter import get_voltage_limit_v, get_voltage_policy_metadata


MODULE_NAMES = [
    "coil_ai_sweep",
    "coil_ai_sweep.core_adapter",
    "coil_ai_sweep.schema",
    "coil_ai_sweep.manifest_io",
    "coil_ai_sweep.sweep_plan",
    "coil_ai_sweep.sweep_lut_generator",
    "coil_ai_sweep.segment_parser",
    "coil_ai_sweep.training_packet",
]


def test_no_streamlit_or_ui_dependency_imported() -> None:
    for module_name in MODULE_NAMES:
        importlib.import_module(module_name)

    forbidden_modules = [
        "streamlit",
        "PySide6",
        "field_analysis.app_ui_snapshot",
        "field_analysis.finite_second_modeling",
    ]
    assert not any(module_name in sys.modules for module_name in forbidden_modules)
    assert not any(
        module_name in sys.modules
        for module_name in sys.modules
        if module_name.startswith("field_analysis.finite_first")
        or module_name.startswith("field_analysis.continuous")
    )


def test_source_does_not_reference_webapp_or_modeling_modules() -> None:
    forbidden_text = [
        "import streamlit",
        "from streamlit",
        "import PySide6",
        "from PySide6",
        "app_ui_snapshot",
        "finite_second_modeling",
        "finite_first",
        "field_analysis.continuous",
    ]
    for module_name in MODULE_NAMES:
        source = inspect.getsource(importlib.import_module(module_name))
        for text in forbidden_text:
            assert text not in source


def test_core_adapter_fallback_voltage_policy_metadata(monkeypatch) -> None:
    monkeypatch.delenv("COIL_ANALYZING_CORE_SRC", raising=False)

    metadata = get_voltage_policy_metadata()

    assert metadata["voltage_policy_source"] == "standalone_fallback"
    assert metadata["voltage_limit_v"] == 10.0
    assert get_voltage_limit_v() == 10.0
