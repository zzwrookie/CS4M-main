#!/usr/bin/env python3
"""Executable entrypoint for the active CS4M conditional pipeline.

The implementation lives in :mod:`legacy.compatibility.pipeline_runtime_exports` so this file
stays small and only handles CLI delegation plus compatibility re-exports for tests/importers.
"""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legacy.compatibility import pipeline_runtime_exports as _runtime
from scripts.pipeline.entrypoints.conditional_e4 import main


def __getattr__(name: str) -> object:
    return getattr(_runtime, name)


if __name__ == "__main__":
    raise SystemExit(main())
