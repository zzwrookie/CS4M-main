#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from cs4m.semantics.clearscope_android import (
    CLEARSCOPE_V31_SEMANTIC_MODE,
    android_process_natural_tokens_refined,
    clearscope_file_natural_tokens_v31,
    clearscope_netflow_natural_tokens_refined,
    normalize_clearscope_semantic_mode,
)


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


def _row_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def tokenize_node_row(row: dict[str, Any], semantic_mode: str) -> AuditNode:
    """Tokenize one ClearScope E5 node row with the requested semantic mode."""
    node_type = str(row.get("node_type", "")).strip().lower()
    normalized_mode = normalize_clearscope_semantic_mode(semantic_mode)
    if normalized_mode != CLEARSCOPE_V31_SEMANTIC_MODE:
        raise ValueError(f"unsupported E5 audit semantic mode: {semantic_mode}")
    if node_type == "file":
        raw_detail = _row_text(row, "path")
        tokens = clearscope_file_natural_tokens_v31(raw_detail)
    elif node_type == "subject":
        raw_detail = _row_text(row, "cmd", "path")
        tokens = android_process_natural_tokens_refined(raw_detail)
    elif node_type == "netflow":
        raw_detail = (
            f"{_row_text(row, 'src_addr')}:{_row_text(row, 'src_port')}"
            f"->{_row_text(row, 'dst_addr')}:{_row_text(row, 'dst_port')}"
        )
        tokens = clearscope_netflow_natural_tokens_refined()
    else:
        raw_detail = ""
        tokens = ("unknown", "unknown", "unknown")
    return AuditNode(
        index_id=int(row["index_id"]),
        node_uuid=str(row.get("node_uuid", "")),
        node_type=node_type,
        raw_detail=raw_detail,
        semantic_tokens=tuple(str(token) for token in tokens),
    )


def connect_db(database: str):
    """Connect to local PostgreSQL using CLAD_DB_* environment variables."""
    try:
        import psycopg2
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "psycopg2 is required for connect_db; install psycopg2 or psycopg2-binary"
        ) from error

    return psycopg2.connect(
        host=os.getenv("CLAD_DB_HOST", "localhost"),
        port=int(os.getenv("CLAD_DB_PORT", "5433")),
        user=os.getenv("CLAD_DB_USER", "postgres"),
        password=os.getenv("CLAD_DB_PASSWORD", ""),
        dbname=database,
    )


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
