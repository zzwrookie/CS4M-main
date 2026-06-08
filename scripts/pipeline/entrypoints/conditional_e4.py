"""Executable CLI entrypoint for the active CS4M conditional pipeline."""

from __future__ import annotations

from scripts.pipeline.config.runtime_config import SlimConfig
from scripts.pipeline.entrypoints.arguments import (
    config_from_args,
    main,
    parse_args,
)


__all__ = ["SlimConfig", "config_from_args", "main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())
