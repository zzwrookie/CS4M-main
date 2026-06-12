#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


E5_SPLITS = {
    "train": {8, 9},
    "val": {11},
    "test": {14, 15, 17},
}
FALLBACK_TOKENS = {
    "file_other",
    "unknown_file",
    "dev_other",
    "socket_other",
    "netflow",
}


@dataclass(frozen=True)
class AuditNode:
    """One node row after ClearScope semantic tokenization."""

    index_id: int
    node_uuid: str
    node_type: str
    raw_detail: str
    semantic_tokens: tuple[str, ...]


def assign_split(day: int) -> str:
    """Return the E5 split name for a calendar day."""
    day_int = int(day)
    for split, days in E5_SPLITS.items():
        if day_int in days:
            return split
    return "unused"


def extract_detail_token(node: AuditNode) -> str:
    """Return the most specific semantic token for a node."""
    if not node.semantic_tokens:
        return "unknown"
    return str(node.semantic_tokens[-1])


def collect_fallback_counts(nodes: Iterable[AuditNode]) -> dict[str, int]:
    """Count fallback semantic buckets in tokenized nodes."""
    counts: Counter[str] = Counter()
    for node in nodes:
        for token in set(node.semantic_tokens):
            if token in FALLBACK_TOKENS:
                counts[token] += 1
    return dict(sorted(counts.items()))


def build_label_free_summary(nodes: Iterable[AuditNode]) -> dict[str, object]:
    """Build the label-free audit summary for tokenized nodes."""
    node_list = list(nodes)
    node_type_counts: Counter[str] = Counter(node.node_type for node in node_list)
    detail_token_counts: Counter[str] = Counter(
        extract_detail_token(node) for node in node_list
    )
    return {
        "node_count": len(node_list),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "fallback_counts": collect_fallback_counts(node_list),
        "top_detail_tokens": detail_token_counts.most_common(50),
    }


def build_label_aware_diagnostics(
    nodes: Iterable[AuditNode],
    malicious_index_ids: set[int],
) -> dict[str, object]:
    """Build the label-aware diagnostic appendix without changing primary audit."""
    malicious = [node for node in nodes if int(node.index_id) in malicious_index_ids]
    fallback_counts = collect_fallback_counts(malicious)
    collision_groups: dict[str, set[str]] = defaultdict(set)
    for node in malicious:
        collision_groups[extract_detail_token(node)].add(node.raw_detail)
    return {
        "warning": "label_aware_diagnostic_only_not_runtime_policy",
        "malicious_node_count": len(malicious),
        "malicious_fallback_counts": fallback_counts,
        "malicious_detail_collision_groups": {
            token: sorted(raw_values)
            for token, raw_values in sorted(collision_groups.items())
            if len(raw_values) > 1
        },
    }
