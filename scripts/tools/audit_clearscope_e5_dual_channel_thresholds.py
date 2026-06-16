#!/usr/bin/env python3
"""Audit ClearScope E5 dual-channel threshold evidence.

This tool is read-only and post-stream. It reports diagnostic TP-node union
coverage, event-semantic score collapse evidence, and validation-only threshold
candidates without mutating original pipeline outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.backfill_clearscope_e5_full_eval import load_gt_db_nodes
from scripts.tools.diagnose_clearscope_e5_event_semantic_node_separation import (
    DEFAULT_GT_DIR,
    DEFAULT_NODE_MAP,
    DEFAULT_RESULT_DIR,
    DEFAULT_SEMANTIC_MODE,
    _alert_key,
    _float_field,
    _int_field,
    _percentile,
    _row_nodes,
    load_node_id_to_idx,
)


TOPK_VALUES = [50, 100, 200, 500, 1000]
EVENT_SCORE_FIELDS = [
    "event_semantic_max_score",
    "event_semantic_p95_score",
    "event_semantic_event_count",
]


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _format_float(value: float) -> float:
    return float(value)


def _sorted_ints(values: Iterable[int]) -> list[int]:
    return sorted(int(value) for value in values)


def build_both_cold_channel_sets(
    alert_rows: Sequence[Mapping[str, Any]],
    gt_nodes: set[int],
) -> dict[str, dict[str, Any]]:
    """Build label-free both-cold full and support-gated channel sets."""
    both_cold_rows = [
        row
        for row in alert_rows
        if str(row.get("target_case", "")) == "both_cold_action_target"
    ]
    node_counts: Counter[int] = Counter()
    endpoint_counts: Counter[str] = Counter()
    for row in both_cold_rows:
        endpoint_counts[_alert_key(row)] += 1
        for node_idx in _row_nodes(row):
            node_counts[node_idx] += 1

    selectors = {
        "both_cold_full": lambda row: True,
        "both_cold_node_repeat_ge_2": lambda row: max(
            (node_counts[node_idx] for node_idx in _row_nodes(row)),
            default=0,
        )
        >= 2,
        "both_cold_endpoint_action_repeat_ge_2": lambda row: endpoint_counts[
            _alert_key(row)
        ]
        >= 2,
    }

    channels: dict[str, dict[str, Any]] = {}
    for name, selector in selectors.items():
        selected_rows = [row for row in both_cold_rows if selector(row)]
        selected_nodes: set[int] = set()
        for row in selected_rows:
            selected_nodes.update(_row_nodes(row))
        channels[name] = {
            "channel_name": name,
            "selected_event_count": int(len(selected_rows)),
            "selected_nodes": selected_nodes,
            "tp_nodes": selected_nodes & gt_nodes,
            "fp_nodes": selected_nodes - gt_nodes,
            "labels_used_for_selection": False,
            "labels_used_after_selection": True,
        }
    return channels


def build_event_semantic_topk_channels(
    node_rows: Sequence[Mapping[str, Any]],
    gt_nodes: set[int],
    *,
    score_fields: Sequence[str] = EVENT_SCORE_FIELDS,
    k_values: Sequence[int] = TOPK_VALUES,
) -> list[dict[str, Any]]:
    """Build event-semantic diagnostic top-k channel sets."""
    channels: list[dict[str, Any]] = []
    for score_field in score_fields:
        ranked = sorted(
            node_rows,
            key=lambda row: (_float_field(row, score_field), -_int_field(row, "node_idx")),
            reverse=True,
        )
        for k in k_values:
            selected = ranked[: min(int(k), len(ranked))]
            selected_nodes = {_int_field(row, "node_idx") for row in selected}
            channels.append(
                {
                    "channel_name": f"{score_field}_top_{int(k)}",
                    "score_field": score_field,
                    "k": int(k),
                    "selected_nodes": selected_nodes,
                    "tp_nodes": selected_nodes & gt_nodes,
                    "diagnostic_topk_only": True,
                },
            )
    return channels


def compute_union_coverage(
    *,
    both_cold_channels: Mapping[str, Mapping[str, Any]],
    event_semantic_channels: Sequence[Mapping[str, Any]],
    gt_nodes: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute TP-node union, overlap, and per-node membership rows."""
    summary_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    total_gt = len(gt_nodes)
    for both_name, both_channel in sorted(both_cold_channels.items()):
        both_tp = set(both_channel.get("tp_nodes", set()))
        for event_channel in event_semantic_channels:
            event_tp = set(event_channel.get("tp_nodes", set()))
            union_tp = both_tp | event_tp
            overlap_tp = both_tp & event_tp
            only_both = both_tp - event_tp
            only_event = event_tp - both_tp
            combination_name = (
                f"{both_name}__{event_channel['channel_name']}"
            )
            summary_rows.append(
                {
                    "combination_name": combination_name,
                    "both_cold_channel": both_name,
                    "event_semantic_channel": event_channel["channel_name"],
                    "score_field": event_channel.get("score_field", ""),
                    "k": int(event_channel.get("k", 0)),
                    "both_cold_tp_node_count": int(len(both_tp)),
                    "event_semantic_tp_node_count": int(len(event_tp)),
                    "union_tp_node_count": int(len(union_tp)),
                    "overlap_tp_node_count": int(len(overlap_tp)),
                    "only_both_cold_tp_node_count": int(len(only_both)),
                    "only_event_semantic_tp_node_count": int(len(only_event)),
                    "missed_tp_node_count": int(total_gt - len(union_tp)),
                    "total_gt_node_count": int(total_gt),
                    "unique_tp_node_recall": float(len(union_tp) / total_gt)
                    if total_gt
                    else 0.0,
                    "only_both_cold_tp_nodes": _sorted_ints(only_both),
                    "only_event_semantic_tp_nodes": _sorted_ints(only_event),
                    "overlap_tp_nodes": _sorted_ints(overlap_tp),
                    "union_tp_nodes": _sorted_ints(union_tp),
                    "labels_used_after_selection": True,
                    "topk_is_diagnostic_not_runtime_policy": True,
                },
            )
            for node_idx in sorted(gt_nodes):
                in_both = node_idx in both_tp
                in_event = node_idx in event_tp
                if in_both and in_event:
                    membership = "both_channels"
                elif in_both:
                    membership = "only_both_cold"
                elif in_event:
                    membership = "only_event_semantic"
                else:
                    membership = "missed_by_both"
                node_rows.append(
                    {
                        "node_idx": int(node_idx),
                        "combination_name": combination_name,
                        "node_membership": membership,
                        "in_both_cold_channel": bool(in_both),
                        "in_event_semantic_channel": bool(in_event),
                        "score_field": event_channel.get("score_field", ""),
                        "k": int(event_channel.get("k", 0)),
                    },
                )
    return summary_rows, node_rows


def _node_group_for_row(row: Mapping[str, Any], node_groups: Mapping[int, str]) -> str:
    groups = {
        node_groups[node_idx]
        for node_idx in _row_nodes(row)
        if node_idx in node_groups
    }
    if "missed_gt_node" in groups:
        return "missed_gt_node"
    if "current_both_cold_tp_node" in groups:
        return "current_both_cold_tp_node"
    if "current_fp_node" in groups:
        return "current_fp_node"
    if groups:
        return sorted(groups)[0]
    return "other_benign_node"


def _summarize_scores(
    *,
    key: str,
    scores: Sequence[float],
    margins: Sequence[float],
    thresholds: Sequence[float],
    row_count: int,
) -> dict[str, Any]:
    return {
        "node_group": key,
        "row_count": int(row_count),
        "max_score": max(scores) if scores else 0.0,
        "p95_score": _percentile(scores, 0.95),
        "mean_score": _safe_mean(scores),
        "max_margin": max(margins) if margins else 0.0,
        "p95_margin": _percentile(margins, 0.95),
        "mean_margin": _safe_mean(margins),
        "max_threshold": max(thresholds) if thresholds else 0.0,
        "mean_threshold": _safe_mean(thresholds),
    }


def _cluster_sort_key(row: Mapping[str, Any]) -> tuple[float, int, str]:
    return (
        _float_field(row, "max_score"),
        _int_field(row, "row_count"),
        str(row.get("cluster_key", "")),
    )


def compute_score_collapse_summary(
    *,
    score_rows: Iterable[Mapping[str, Any]],
    node_groups: Mapping[int, str],
    high_score_limit: int = 25,
) -> dict[str, Any]:
    """Aggregate event-semantic score collapse evidence from streamed rows."""
    group_scores: dict[str, list[float]] = defaultdict(list)
    group_margins: dict[str, list[float]] = defaultdict(list)
    group_thresholds: dict[str, list[float]] = defaultdict(list)
    threshold_source_counts: Counter[str] = Counter()
    threshold_values: Counter[str] = Counter()
    endpoint_clusters: dict[str, dict[str, Any]] = {}
    action_type_clusters: dict[str, dict[str, Any]] = {}
    row_count = 0

    for row in score_rows:
        if str(row.get("target_case", "")) != "event_semantic_target":
            continue
        row_count += 1
        score = _float_field(row, "score")
        threshold = _float_field(row, "threshold")
        margin = score - threshold
        group = _node_group_for_row(row, node_groups)
        threshold_source = str(row.get("threshold_basis", "")).strip() or "unknown"
        threshold_source_counts[threshold_source] += 1
        threshold_values[f"{threshold:.12g}"] += 1
        group_scores[group].append(score)
        group_margins[group].append(margin)
        group_thresholds[group].append(threshold)

        if group == "current_fp_node":
            endpoint_key = _alert_key(row)
            action_type_key = "|".join(
                [
                    str(row.get("action", "")),
                    str(row.get("src_type", "")),
                    str(row.get("dst_type", "")),
                ],
            )
            _update_cluster(
                endpoint_clusters,
                cluster_key=endpoint_key,
                cluster_kind="endpoint_action",
                row=row,
                score=score,
                margin=margin,
                threshold_source=threshold_source,
            )
            _update_cluster(
                action_type_clusters,
                cluster_key=action_type_key,
                cluster_kind="action_type",
                row=row,
                score=score,
                margin=margin,
                threshold_source=threshold_source,
            )

    group_summary = [
        _summarize_scores(
            key=group,
            scores=group_scores[group],
            margins=group_margins[group],
            thresholds=group_thresholds[group],
            row_count=len(group_scores[group]),
        )
        for group in sorted(group_scores)
    ]
    clusters = [
        _finalize_cluster(row)
        for row in list(endpoint_clusters.values()) + list(action_type_clusters.values())
    ]
    clusters = sorted(clusters, key=_cluster_sort_key, reverse=True)[:high_score_limit]
    conclusion = _collapse_conclusion(group_summary, clusters)
    return {
        "event_semantic_row_count": int(row_count),
        "threshold_source_counts": dict(threshold_source_counts),
        "threshold_values": dict(threshold_values),
        "group_summary": group_summary,
        "high_score_fp_clusters": clusters,
        "conclusion_bucket": conclusion,
    }


def _update_cluster(
    clusters: dict[str, dict[str, Any]],
    *,
    cluster_key: str,
    cluster_kind: str,
    row: Mapping[str, Any],
    score: float,
    margin: float,
    threshold_source: str,
) -> None:
    cluster = clusters.setdefault(
        f"{cluster_kind}:{cluster_key}",
        {
            "cluster_key": cluster_key,
            "cluster_kind": cluster_kind,
            "scores": [],
            "margins": [],
            "nodes": set(),
            "threshold_sources": Counter(),
            "example_action": str(row.get("action", "")),
            "example_src_type": str(row.get("src_type", "")),
            "example_dst_type": str(row.get("dst_type", "")),
        },
    )
    cluster["scores"].append(score)
    cluster["margins"].append(margin)
    cluster["nodes"].update(_row_nodes(row))
    cluster["threshold_sources"][threshold_source] += 1


def _finalize_cluster(cluster: Mapping[str, Any]) -> dict[str, Any]:
    scores = list(cluster.get("scores", []))
    margins = list(cluster.get("margins", []))
    threshold_sources: Counter[str] = cluster.get("threshold_sources", Counter())
    return {
        "cluster_key": str(cluster.get("cluster_key", "")),
        "cluster_kind": str(cluster.get("cluster_kind", "")),
        "row_count": int(len(scores)),
        "node_count": int(len(cluster.get("nodes", set()))),
        "max_score": max(scores) if scores else 0.0,
        "p95_score": _percentile(scores, 0.95),
        "mean_score": _safe_mean(scores),
        "max_margin": max(margins) if margins else 0.0,
        "threshold_source": threshold_sources.most_common(1)[0][0]
        if threshold_sources
        else "",
        "example_action": str(cluster.get("example_action", "")),
        "example_src_type": str(cluster.get("example_src_type", "")),
        "example_dst_type": str(cluster.get("example_dst_type", "")),
    }


def _collapse_conclusion(
    group_summary: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
) -> str:
    by_group = {str(row.get("node_group", "")): row for row in group_summary}
    missed = by_group.get("missed_gt_node", {})
    fp = by_group.get("current_fp_node", {})
    if _float_field(missed, "max_margin") < 0 and _float_field(fp, "max_margin") < 0:
        if clusters and _float_field(clusters[0], "max_score") > _float_field(
            missed,
            "max_score",
        ):
            return "mixed_causes"
        return "threshold_above_scores"
    if _float_field(fp, "p95_score") >= _float_field(missed, "p95_score"):
        return "fp_high_score_cluster_dominates"
    if _float_field(missed, "mean_score") > 0 and _float_field(missed, "max_margin") < 0:
        return "node_level_signal_event_level_weak"
    return "score_not_separable"


def build_validation_threshold_candidates(
    summary_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build validation-only threshold candidate records."""
    event_row = next(
        (
            row
            for row in summary_rows
            if str(row.get("target_case", "")) == "event_semantic_target"
        ),
        {},
    )
    p999 = _float_field(event_row, "test_p999")
    parent = _float_field(event_row, "parent_threshold")
    global_threshold = _float_field(event_row, "global_threshold")
    threshold = max(p999, parent, global_threshold)
    candidates = [
        {
            "candidate_name": "event_semantic_validation_quantile_p999",
            "channel": "event_semantic_target",
            "threshold_kind": "score_quantile",
            "score_threshold": p999,
            "support_threshold": 0,
            "source_artifact": "conditional_score_summary_by_target_action_type.csv",
            "source_field": "test_p999",
            "runtime_visible": True,
            "validation_only": True,
            "uses_test_labels_for_selection": False,
            "description": "Use validation p999 as an event-semantic score floor.",
        },
        {
            "candidate_name": "event_semantic_validation_quantile_parent_or_group",
            "channel": "event_semantic_target",
            "threshold_kind": "score_quantile",
            "score_threshold": threshold,
            "support_threshold": 0,
            "source_artifact": "conditional_score_summary_by_target_action_type.csv",
            "source_field": "max(test_p999,parent_threshold,global_threshold)",
            "runtime_visible": True,
            "validation_only": True,
            "uses_test_labels_for_selection": False,
            "description": "Use the conservative validation-derived score floor.",
        },
        {
            "candidate_name": "event_semantic_node_max_near_validation_p999",
            "channel": "event_semantic_target",
            "threshold_kind": "node_score_support",
            "score_threshold": p999 * 0.5 if p999 else 0.0,
            "support_threshold": 1,
            "source_artifact": "conditional_score_summary_by_target_action_type.csv",
            "source_field": "0.5 * test_p999",
            "runtime_visible": True,
            "validation_only": True,
            "uses_test_labels_for_selection": False,
            "description": "Trigger node support when max score is near validation p999.",
        },
        {
            "candidate_name": "event_semantic_node_near_threshold_count_ge_1",
            "channel": "event_semantic_target",
            "threshold_kind": "node_support_count",
            "score_threshold": p999 * 0.5 if p999 else 0.0,
            "support_threshold": 1,
            "source_artifact": "conditional_score_summary_by_target_action_type.csv",
            "source_field": "0.5 * test_p999 plus runtime support count",
            "runtime_visible": True,
            "validation_only": True,
            "uses_test_labels_for_selection": False,
            "description": "Require at least one near-threshold event for a node.",
        },
        {
            "candidate_name": "event_semantic_node_max_and_support_count",
            "channel": "event_semantic_target",
            "threshold_kind": "node_score_support",
            "score_threshold": p999 * 0.5 if p999 else 0.0,
            "support_threshold": 2,
            "source_artifact": "conditional_score_summary_by_target_action_type.csv",
            "source_field": "0.5 * test_p999 plus runtime support count",
            "runtime_visible": True,
            "validation_only": True,
            "uses_test_labels_for_selection": False,
            "description": "Combine near-validation score with repeated node support.",
        },
        {
            "candidate_name": "both_cold_node_repeat_ge_2",
            "channel": "both_cold_action_target",
            "threshold_kind": "support_gate",
            "score_threshold": 0.0,
            "support_threshold": 2,
            "source_artifact": "online runtime support counts",
            "source_field": "node repeat count",
            "runtime_visible": True,
            "validation_only": True,
            "uses_test_labels_for_selection": False,
            "description": "Retain both-cold alerts with repeated node support.",
        },
        {
            "candidate_name": "both_cold_endpoint_action_repeat_ge_2",
            "channel": "both_cold_action_target",
            "threshold_kind": "support_gate",
            "score_threshold": 0.0,
            "support_threshold": 2,
            "source_artifact": "online runtime support counts",
            "source_field": "endpoint action repeat count",
            "runtime_visible": True,
            "validation_only": True,
            "uses_test_labels_for_selection": False,
            "description": "Retain both-cold alerts with repeated endpoint/action support.",
        },
    ]
    return candidates


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a CSV file into dictionaries."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stream_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    """Stream a CSV file as dictionaries."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_ready(value: Any) -> Any:
    if isinstance(value, set):
        return _sorted_ints(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _node_groups_from_scores(node_rows: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    return {
        _int_field(row, "node_idx"): str(row.get("node_group", "other_benign_node"))
        for row in node_rows
    }


def _load_gt_nodes(gt_dir: Path, node_map_path: Path) -> set[int]:
    gt_db_nodes = load_gt_db_nodes(sorted(Path(gt_dir).glob("*.csv")))
    node_map = load_node_id_to_idx(node_map_path)
    return {
        int(node_map[int(db_node)])
        for db_node in gt_db_nodes
        if int(db_node) in node_map
    }


def run_audit(
    *,
    result_dir: Path,
    gt_dir: Path,
    node_id_to_idx_path: Path,
) -> dict[str, Any]:
    """Run the read-only dual-channel threshold audit."""
    result_dir = Path(result_dir)
    gt_nodes = _load_gt_nodes(gt_dir, node_id_to_idx_path)
    alert_rows = load_csv_rows(result_dir / "online_event_alerts.csv")
    node_rows = load_csv_rows(result_dir / "e5_event_semantic_node_scores.csv")
    summary_rows = load_csv_rows(
        result_dir / "conditional_score_summary_by_target_action_type.csv",
    )
    both_cold_channels = build_both_cold_channel_sets(alert_rows, gt_nodes)
    event_channels = build_event_semantic_topk_channels(node_rows, gt_nodes)
    union_rows, union_node_rows = compute_union_coverage(
        both_cold_channels=both_cold_channels,
        event_semantic_channels=event_channels,
        gt_nodes=gt_nodes,
    )
    collapse_summary = compute_score_collapse_summary(
        score_rows=stream_csv_rows(result_dir / "online_event_score_trace.csv"),
        node_groups=_node_groups_from_scores(node_rows),
    )
    candidates = build_validation_threshold_candidates(summary_rows)
    paths = {
        "union_coverage": result_dir / "e5_dual_channel_union_coverage.json",
        "union_nodes": result_dir / "e5_dual_channel_union_nodes.csv",
        "score_collapse": result_dir / "e5_event_semantic_score_collapse.json",
        "high_score_fp_clusters": (
            result_dir / "e5_event_semantic_high_score_fp_clusters.csv"
        ),
        "threshold_candidates": result_dir / "e5_validation_threshold_candidates.csv",
    }
    write_json(
        paths["union_coverage"],
        {
            "dataset": "CLEARSCOPE_E5",
            "semantic_mode": DEFAULT_SEMANTIC_MODE,
            "result_dir": str(result_dir),
            "total_gt_node_count": int(len(gt_nodes)),
            "union_coverage": union_rows,
            "leakage_contract": {
                "post_stream_only": True,
                "labels_used_for_threshold_selection": False,
                "labels_used_for_runtime_policy": False,
                "labels_used_after_channel_selection_for_diagnostics": True,
                "topk_is_diagnostic_not_runtime_policy": True,
                "original_outputs_mutated": False,
            },
        },
    )
    write_json(
        paths["score_collapse"],
        {
            "dataset": "CLEARSCOPE_E5",
            "semantic_mode": DEFAULT_SEMANTIC_MODE,
            "result_dir": str(result_dir),
            **collapse_summary,
            "leakage_contract": {
                "post_stream_only": True,
                "labels_used_for_threshold_selection": False,
                "original_outputs_mutated": False,
            },
        },
    )
    write_csv(
        paths["union_nodes"],
        [
            "node_idx",
            "combination_name",
            "node_membership",
            "in_both_cold_channel",
            "in_event_semantic_channel",
            "score_field",
            "k",
        ],
        union_node_rows,
    )
    write_csv(
        paths["high_score_fp_clusters"],
        [
            "cluster_key",
            "cluster_kind",
            "row_count",
            "node_count",
            "max_score",
            "p95_score",
            "mean_score",
            "max_margin",
            "threshold_source",
            "example_action",
            "example_src_type",
            "example_dst_type",
        ],
        collapse_summary["high_score_fp_clusters"],
    )
    write_csv(
        paths["threshold_candidates"],
        [
            "candidate_name",
            "channel",
            "threshold_kind",
            "score_threshold",
            "support_threshold",
            "source_artifact",
            "source_field",
            "runtime_visible",
            "validation_only",
            "uses_test_labels_for_selection",
            "description",
        ],
        candidates,
    )
    summary = {
        "dataset": "CLEARSCOPE_E5",
        "semantic_mode": DEFAULT_SEMANTIC_MODE,
        "result_dir": str(result_dir),
        "total_gt_node_count": int(len(gt_nodes)),
        "best_union_rows": _best_union_rows(union_rows),
        "score_collapse_conclusion": collapse_summary["conclusion_bucket"],
        "threshold_candidate_count": int(len(candidates)),
        "paths": {key: str(path) for key, path in paths.items()},
    }
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    return summary


def _best_union_rows(union_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best_by_field: dict[str, Mapping[str, Any]] = {}
    for row in union_rows:
        key = f"{row.get('both_cold_channel')}::{row.get('score_field')}"
        current = best_by_field.get(key)
        if current is None or _int_field(row, "union_tp_node_count") > _int_field(
            current,
            "union_tp_node_count",
        ):
            best_by_field[key] = row
    return [
        {
            "combination_name": str(row.get("combination_name", "")),
            "union_tp_node_count": int(row.get("union_tp_node_count", 0)),
            "unique_tp_node_recall": _format_float(
                _float_field(row, "unique_tp_node_recall"),
            ),
        }
        for row in sorted(
            best_by_field.values(),
            key=lambda item: (
                _int_field(item, "union_tp_node_count"),
                str(item.get("combination_name", "")),
            ),
            reverse=True,
        )
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--gt_dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--node_id_to_idx", type=Path, default=DEFAULT_NODE_MAP)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit CLI."""
    args = parse_args(argv)
    run_audit(
        result_dir=args.result_dir,
        gt_dir=args.gt_dir,
        node_id_to_idx_path=args.node_id_to_idx,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
