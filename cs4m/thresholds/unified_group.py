"""Unified group threshold resolver for CS4M raw alerts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class GroupKey:
    """Target/action/type group key."""

    target_case: str
    action: str
    src_type: str
    dst_type: str

    def as_string(self) -> str:
        """Return stable text representation."""
        return f"{self.target_case}|{self.action}|{self.src_type}->{self.dst_type}"


@dataclass(frozen=True)
class ResolvedThreshold:
    """Resolved threshold and explainability metadata."""

    group_key: str
    threshold: float
    threshold_source: str
    threshold_level: str
    validation_count: int
    train_count: int


class UnifiedGroupThresholdResolver:
    """Resolve validation group, train max, then validation global threshold."""

    def __init__(
        self,
        validation_groups: Mapping[GroupKey, Mapping[str, Any]],
        train_groups: Mapping[GroupKey, Mapping[str, Any]],
        validation_global_threshold: float,
        min_validation_count: int = 100,
        min_train_count: int = 20,
        train_margin_abs: float = 0.005,
        train_margin_rel: float = 0.05,
    ) -> None:
        self.validation_groups = dict(validation_groups)
        self.train_groups = dict(train_groups)
        self.validation_global_threshold = float(validation_global_threshold)
        self.min_validation_count = int(min_validation_count)
        self.min_train_count = int(min_train_count)
        self.train_margin_abs = float(train_margin_abs)
        self.train_margin_rel = float(train_margin_rel)

    def resolve(self, key: GroupKey) -> ResolvedThreshold:
        """Resolve one group key without using test labels."""
        validation = self.validation_groups.get(key, {})
        validation_count = _to_int(validation.get("validation_count"))
        if validation_count >= self.min_validation_count:
            return ResolvedThreshold(
                group_key=key.as_string(),
                threshold=_to_float(validation.get("validation_quantile_threshold")),
                threshold_source="validation_group_quantile",
                threshold_level="exact_group",
                validation_count=validation_count,
                train_count=_to_int(self.train_groups.get(key, {}).get("train_count")),
            )

        train = self.train_groups.get(key, {})
        train_count = _to_int(train.get("train_count"))
        if train_count >= self.min_train_count:
            train_max = _to_float(train.get("train_max_score"))
            margin = max(self.train_margin_abs, self.train_margin_rel * train_max)
            return ResolvedThreshold(
                group_key=key.as_string(),
                threshold=train_max + margin,
                threshold_source="train_group_max",
                threshold_level="exact_group_train_fallback",
                validation_count=validation_count,
                train_count=train_count,
            )

        return ResolvedThreshold(
            group_key=key.as_string(),
            threshold=self.validation_global_threshold,
            threshold_source="validation_global_quantile",
            threshold_level="global",
            validation_count=validation_count,
            train_count=train_count,
        )


def _to_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
