"""Compatibility aggregator for the split active CS4M conditional pipeline.

The executable implementation is split across responsibility modules in
:mod:`scripts.pipeline`. This module keeps the historical import path available for tests and
legacy callers while avoiding a second copy of the runtime logic.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

_RUNTIME_MODULE_NAMES = (
    "config.runtime_config",
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


def _load_runtime_modules() -> list[ModuleType]:
    """Import all split pipeline modules in dependency order."""
    return [
        importlib.import_module(f"scripts.pipeline.{module_name}")
        for module_name in _RUNTIME_MODULE_NAMES
    ]


def _public_runtime_namespace(modules: list[ModuleType]) -> dict[str, Any]:
    """Collect runtime symbols, including private helpers used by contract tests."""
    namespace: dict[str, Any] = {}
    for module in modules:
        for name, value in vars(module).items():
            if name.startswith("__") and name.endswith("__"):
                continue
            namespace[name] = value
    return namespace


def _bind_runtime_modules(modules: list[ModuleType], namespace: dict[str, Any]) -> None:
    """Share split-module helpers so functions keep the old monolith call graph."""
    for module in modules:
        module.__dict__.update(namespace)


_MODULES = _load_runtime_modules()
_NAMESPACE = _public_runtime_namespace(_MODULES)
_bind_runtime_modules(_MODULES, _NAMESPACE)
globals().update(_NAMESPACE)

__all__ = sorted(name for name in _NAMESPACE if not name.startswith("__"))


def __getattr__(name: str) -> Any:
    """Resolve compatibility attributes from the shared runtime namespace."""
    try:
        return _NAMESPACE[name]
    except KeyError as exc:  # pragma: no cover - mirrors normal module AttributeError behavior
        raise AttributeError(name) from exc


if __name__ == "__main__":
    raise SystemExit(main())
