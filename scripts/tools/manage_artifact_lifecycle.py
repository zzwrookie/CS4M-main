#!/usr/bin/env python3
"""Manage generated CS4M artifacts across train/validation/test phases."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class LifecycleProfile(str, Enum):
    """Artifact cleanup phase profile."""

    TRAIN_TO_VALIDATION = "train_to_validation"
    VALIDATION_TO_TEST = "validation_to_test"
    TEST_LIGHT = "test_light"
    TEST_REPRODUCTION = "test_reproduction"


@dataclass(frozen=True)
class LifecycleCleanupPlan:
    """Concrete lifecycle cleanup manifest."""

    keep_paths: list[Path]
    delete_paths: list[Path]


def build_lifecycle_cleanup_plan(
    output_dir: Path,
    profile: LifecycleProfile,
) -> LifecycleCleanupPlan:
    """Build a lifecycle cleanup plan without deleting anything."""
    output_path = Path(output_dir)
    keep_names = _keep_names(profile)
    keep_paths: list[Path] = []
    delete_paths: list[Path] = []
    for path in sorted(output_path.iterdir()) if output_path.exists() else []:
        if path.name in keep_names or _matches_keep_suffix(path.name, keep_names):
            keep_paths.append(path)
        elif _should_delete(path, profile):
            delete_paths.append(path)
        else:
            keep_paths.append(path)
    return LifecycleCleanupPlan(keep_paths=keep_paths, delete_paths=delete_paths)


def write_lifecycle_plan(plan: LifecycleCleanupPlan, output_path: Path) -> None:
    """Write lifecycle cleanup manifest as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "keep_paths": [str(path) for path in plan.keep_paths],
        "delete_paths": [str(path) for path in plan.delete_paths],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _keep_names(profile: LifecycleProfile) -> set[str]:
    common = {
        "config_effective.yaml",
        "eval_causal_semantics_slim.json",
        "eval_metrics.json",
        "manifest.json",
        "metrics.json",
        "train_group_score_summary.csv",
        "validation_global_threshold.json",
        "validation_group_threshold_summary.csv",
    }
    if profile == LifecycleProfile.TEST_LIGHT:
        return common | {"online_event_alerts.csv", "online_node_alerts.csv"}
    if profile == LifecycleProfile.TEST_REPRODUCTION:
        return common | {
            "online_event_alerts.csv",
            "online_event_score_trace.csv",
            "online_node_alerts.csv",
        }
    return common | {"model.pkl"}


def _matches_keep_suffix(name: str, keep_names: set[str]) -> bool:
    return any(name.endswith(suffix) for suffix in keep_names)


def _should_delete(path: Path, profile: LifecycleProfile) -> bool:
    if profile == LifecycleProfile.TEST_REPRODUCTION:
        return False
    name = path.name
    if profile == LifecycleProfile.TRAIN_TO_VALIDATION:
        return name.endswith((".memmap", ".npy", ".tmp")) or "temporary" in name
    if profile == LifecycleProfile.VALIDATION_TO_TEST:
        return "validation" in name and not name.endswith(("_summary.csv", ".json"))
    if profile == LifecycleProfile.TEST_LIGHT:
        return name == "online_event_score_trace.csv" or name.endswith(".tmp")
    return False


def main() -> int:
    """Write a lifecycle cleanup manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", required=True, choices=[item.value for item in LifecycleProfile])
    parser.add_argument("--manifest-path", required=True)
    args = parser.parse_args()
    plan = build_lifecycle_cleanup_plan(
        Path(args.output_dir),
        LifecycleProfile(args.profile),
    )
    write_lifecycle_plan(plan, Path(args.manifest_path))
    print(json.dumps({"delete_count": len(plan.delete_paths)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
