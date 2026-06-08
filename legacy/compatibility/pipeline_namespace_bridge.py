"""Shared symbol binding for split active pipeline modules."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, MutableMapping


_IMPLEMENTATION_MODULE_NAMES = (
    "checks.preflight",
    "state.online_state_runtime",
    "features.conditional_context",
    "io.conditional_cache",
    "conditional.train",
    "conditional.infer",
    "io.event_artifacts",
    "features.semantic_features",
    "outputs.conditional_reports",
    "io.cache_payloads",
    "entrypoints.arguments",
    "outputs.metrics_summary",
    "outputs.alert_output",
)

_BINDING = False


def _runtime_modules() -> list[ModuleType]:
    """Import split implementation modules in dependency order."""
    modules = [importlib.import_module("scripts.pipeline.config.runtime_config")]
    modules.extend(
        importlib.import_module(f"scripts.pipeline.{module_name}")
        for module_name in _IMPLEMENTATION_MODULE_NAMES
    )
    return modules


def _runtime_namespace(modules: list[ModuleType]) -> dict[str, Any]:
    """Collect all implementation symbols, including private cross-module helpers."""
    namespace: dict[str, Any] = {}
    for module in modules:
        for name, value in vars(module).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            namespace[name] = value
    return namespace


def link_runtime_modules(target_globals: MutableMapping[str, Any] | None = None) -> None:
    """Populate split-module globals so direct imports can call active entrypoints."""
    global _BINDING

    if _BINDING:
        return
    _BINDING = True
    try:
        modules = _runtime_modules()
        namespace = _runtime_namespace(modules)
        for module in modules:
            module.__dict__.update(namespace)
        if target_globals is not None:
            target_globals.update(namespace)
    finally:
        _BINDING = False
