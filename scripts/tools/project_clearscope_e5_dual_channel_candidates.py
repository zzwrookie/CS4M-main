#!/usr/bin/env python3
"""Project ClearScope E5 dual-channel validation candidates.

This tool is read-only and post-stream. It projects validation-only candidate
sets from existing sidecars and current alerts; it does not implement runtime
policy or mutate original pipeline outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.audit_clearscope_e5_dual_channel_thresholds import (
    build_both_cold_channel_sets,
)
from scripts.tools.backfill_clearscope_e5_full_eval import load_gt_db_nodes
from scripts.tools.diagnose_clearscope_e5_event_semantic_node_separation import (
    DEFAULT_GT_DIR,
    DEFAULT_NODE_MAP,
    DEFAULT_RESULT_DIR,
    DEFAULT_SEMANTIC_MODE,
    _float_field,
    _int_field,
    _row_nodes,
    load_node_id_to_idx,
)


def _sorted_ints(values: Iterable[int]) -> list[int]:
    return sorted(int(value) for value in values)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _node_stats(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event_semantic_max_score": _float_field(row, "event_semantic_max_score"),
        "event_semantic_p95_score": _float_field(row, "event_semantic_p95_score"),
        "event_semantic_event_count": _int_field(row, "event_semantic_event_count", 0),
        "event_semantic_near_threshold_count": _int_field(
            row,
            "event_semantic_near_threshold_count",
            0,
        ),
        "event_semantic_above_threshold_count": _int_field(
            row,
            "event_semantic_above_threshold_count",
            0,
        ),
    }


def build_channel_b_variants(
    node_rows: Sequence[Mapping[str, Any]],
    gt_nodes: set[int],
    *,
    score_floor: float,
) -> dict[str, dict[str, Any]]:
    """Build B_none, B_node_support_ge_1, and B_node_support_ge_2 variants."""
    node_by_id = {_int_field(row, "node_idx"): dict(row) for row in node_rows}
    variants: dict[str, dict[str, Any]] = {
        "B_none": {
            "variant_name": "B_none",
            "selected_nodes": set(),
            "tp_nodes": set(),
            "fp_nodes": set(),
            "node_rows": {},
            "score_floor": float(score_floor),
            "support_threshold": 0,
            "validation_only": True,
            "uses_topk": False,
            "runtime_visible": True,
        },
    }
    for support_threshold in (1, 2):
        selected_nodes: set[int] = set()
        selected_rows: dict[int, dict[str, Any]] = {}
        for node_idx, row in node_by_id.items():
            max_score = _float_field(row, "event_semantic_max_score")
            near_count = _int_field(row, "event_semantic_near_threshold_count", 0)
            if max_score >= score_floor and near_count >= support_threshold:
                selected_nodes.add(node_idx)
                selected_rows[node_idx] = dict(row)
        name = f"B_node_support_ge_{support_threshold}"
        variants[name] = {
            "variant_name": name,
            "selected_nodes": selected_nodes,
            "tp_nodes": selected_nodes & gt_nodes,
            "fp_nodes": selected_nodes - gt_nodes,
            "node_rows": selected_rows,
            "score_floor": float(score_floor),
            "support_threshold": int(support_threshold),
            "validation_only": True,
            "uses_topk": False,
            "runtime_visible": True,
        }
    return variants


def normalize_channel_a_variants(
    channels: Mapping[str, Mapping[str, Any]],
    alert_rows: Sequence[Mapping[str, Any]],
    gt_nodes: set[int],
) -> dict[str, dict[str, Any]]:
    """Rename both-cold channels into candidate Channel A variants."""
    name_map = {
        "both_cold_full": "A_full",
        "both_cold_node_repeat_ge_2": "A_node_support",
        "both_cold_endpoint_action_repeat_ge_2": "A_endpoint_support",
    }
    row_sets = _selected_alert_rows_by_channel(alert_rows)
    normalized: dict[str, dict[str, Any]] = {}
    for source_name, target_name in name_map.items():
        channel = dict(channels[source_name])
        selected_rows = row_sets[source_name]
        tp_event_count = sum(
            1
            for row in selected_rows
            if bool(_row_nodes(row) & gt_nodes)
        )
        channel["variant_name"] = target_name
        channel["selected_event_rows"] = selected_rows
        channel["selected_event_count"] = int(len(selected_rows))
        channel["tp_event_count"] = int(tp_event_count)
        normalized[target_name] = channel
    return normalized


def _selected_alert_rows_by_channel(
    alert_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    both_cold_rows = [
        row
        for row in alert_rows
        if str(row.get("target_case", "")) == "both_cold_action_target"
    ]
    node_counts: dict[int, int] = {}
    endpoint_counts: dict[str, int] = {}
    for row in both_cold_rows:
        key = _endpoint_key(row)
        endpoint_counts[key] = endpoint_counts.get(key, 0) + 1
        for node_idx in _row_nodes(row):
            node_counts[node_idx] = node_counts.get(node_idx, 0) + 1
    return {
        "both_cold_full": list(both_cold_rows),
        "both_cold_node_repeat_ge_2": [
            row
            for row in both_cold_rows
            if max((node_counts.get(node_idx, 0) for node_idx in _row_nodes(row)), default=0)
            >= 2
        ],
        "both_cold_endpoint_action_repeat_ge_2": [
            row
            for row in both_cold_rows
            if endpoint_counts.get(_endpoint_key(row), 0) >= 2
        ],
    }


def _endpoint_key(row: Mapping[str, Any]) -> str:
    key = str(row.get("endpoint_action_key", "")).strip()
    if key:
        return key
    return "|".join(
        [
            str(row.get("src_idx", "")),
            str(row.get("dst_idx", "")),
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
        ],
    )


def project_candidate_matrix(
    *,
    channel_a_variants: Mapping[str, Mapping[str, Any]],
    channel_b_variants: Mapping[str, Mapping[str, Any]],
    gt_nodes: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Project node/event metrics for every A/B candidate combination."""
    candidate_rows: list[dict[str, Any]] = []
    candidate_node_rows: list[dict[str, Any]] = []
    candidate_event_rows: list[dict[str, Any]] = []
    total_gt = len(gt_nodes)
    for a_name, a_variant in sorted(channel_a_variants.items()):
        a_nodes = set(a_variant.get("selected_nodes", set()))
        a_tp = set(a_variant.get("tp_nodes", set()))
        a_fp = set(a_variant.get("fp_nodes", set()))
        for b_name, b_variant in sorted(channel_b_variants.items()):
            b_nodes = set(b_variant.get("selected_nodes", set()))
            b_tp = set(b_variant.get("tp_nodes", set()))
            b_fp = set(b_variant.get("fp_nodes", set()))
            union_tp = a_tp | b_tp
            union_fp = (a_nodes | b_nodes) - gt_nodes
            candidate_name = f"{a_name}__{b_name}"
            b_node_rows = b_variant.get("node_rows", {})
            support_stats = _support_sums(b_nodes, b_node_rows)
            candidate_rows.append(
                {
                    "candidate_name": candidate_name,
                    "channel_a_variant": a_name,
                    "channel_b_variant": b_name,
                    "channel_a_event_count": int(a_variant.get("selected_event_count", 0)),
                    "channel_a_tp_event_count": int(a_variant.get("tp_event_count", 0)),
                    "channel_a_tp_node_count": int(len(a_tp)),
                    "channel_a_fp_node_count": int(len(a_fp)),
                    "channel_b_tp_node_count": int(len(b_tp)),
                    "channel_b_fp_node_count": int(len(b_fp)),
                    "union_tp_node_count": int(len(union_tp)),
                    "union_fp_node_count": int(len(union_fp)),
                    "unique_tp_node_recall": float(len(union_tp) / total_gt)
                    if total_gt
                    else 0.0,
                    "overlap_tp_node_count": int(len(a_tp & b_tp)),
                    "only_channel_a_tp_node_count": int(len(a_tp - b_tp)),
                    "only_channel_b_tp_node_count": int(len(b_tp - a_tp)),
                    "event_semantic_support_event_count": support_stats["event_count"],
                    "event_semantic_near_threshold_count": support_stats["near_count"],
                    "event_semantic_above_threshold_count": support_stats["above_count"],
                    "channel_b_score_floor": float(b_variant.get("score_floor", 0.0)),
                    "channel_b_support_threshold": int(
                        b_variant.get("support_threshold", 0),
                    ),
                    "channel_b_validation_only": bool(
                        b_variant.get("validation_only", False),
                    ),
                    "channel_b_uses_topk": bool(b_variant.get("uses_topk", True)),
                    "union_tp_nodes": _sorted_ints(union_tp),
                    "only_channel_b_tp_nodes": _sorted_ints(b_tp - a_tp),
                },
            )
            candidate_node_rows.extend(
                _candidate_node_rows(
                    candidate_name=candidate_name,
                    a_nodes=a_nodes,
                    b_nodes=b_nodes,
                    gt_nodes=gt_nodes,
                    b_node_rows=b_node_rows,
                ),
            )
            candidate_event_rows.extend(
                _candidate_event_rows(
                    candidate_name=candidate_name,
                    channel_a_variant=a_name,
                    rows=a_variant.get("selected_event_rows", []),
                    gt_nodes=gt_nodes,
                ),
            )
    return candidate_rows, candidate_node_rows, candidate_event_rows


def _support_sums(
    selected_nodes: set[int],
    node_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, int]:
    event_count = 0
    near_count = 0
    above_count = 0
    for node_idx in selected_nodes:
        row = node_rows.get(node_idx, {})
        event_count += _int_field(row, "event_semantic_event_count", 0)
        near_count += _int_field(row, "event_semantic_near_threshold_count", 0)
        above_count += _int_field(row, "event_semantic_above_threshold_count", 0)
    return {
        "event_count": int(event_count),
        "near_count": int(near_count),
        "above_count": int(above_count),
    }


def _candidate_node_rows(
    *,
    candidate_name: str,
    a_nodes: set[int],
    b_nodes: set[int],
    gt_nodes: set[int],
    b_node_rows: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node_idx in sorted(a_nodes | b_nodes):
        in_a = node_idx in a_nodes
        in_b = node_idx in b_nodes
        is_tp = node_idx in gt_nodes
        if in_a and in_b and is_tp:
            membership = "both_channels_tp"
        elif in_a and is_tp:
            membership = "only_channel_a_tp"
        elif in_b and is_tp:
            membership = "only_channel_b_tp"
        elif in_a and in_b:
            membership = "both_channels_fp"
        elif in_a:
            membership = "channel_a_fp"
        else:
            membership = "channel_b_fp"
        stats = _node_stats(b_node_rows.get(node_idx, {}))
        rows.append(
            {
                "candidate_name": candidate_name,
                "node_idx": int(node_idx),
                "node_label": "malicious" if is_tp else "benign",
                "node_membership": membership,
                "in_channel_a": bool(in_a),
                "in_channel_b": bool(in_b),
                **stats,
            },
        )
    return rows


def _candidate_event_rows(
    *,
    candidate_name: str,
    channel_a_variant: str,
    rows: Sequence[Mapping[str, Any]],
    gt_nodes: set[int],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append(
            {
                "candidate_name": candidate_name,
                "stream_pos": row.get("stream_pos", ""),
                "event_index": row.get("event_index", ""),
                "src_idx": row.get("src_idx", ""),
                "dst_idx": row.get("dst_idx", ""),
                "action": row.get("action", ""),
                "event_score": row.get("event_score", ""),
                "threshold": row.get("threshold", ""),
                "target_case": row.get("target_case", ""),
                "event_has_gt_node": bool(_row_nodes(row) & gt_nodes),
                "channel_a_variant": channel_a_variant,
            },
        )
    return output


def recommend_candidates(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_name: str,
) -> list[dict[str, Any]]:
    """Apply bounded-smoke recommendation criteria."""
    baseline = next(
        row
        for row in candidate_rows
        if str(row.get("candidate_name", "")) == baseline_name
    )
    baseline_tp = _int_field(baseline, "union_tp_node_count", 0)
    baseline_fp = _int_field(baseline, "union_fp_node_count", 0)
    evaluated = []
    for row in candidate_rows:
        item = dict(row)
        reasons: list[str] = []
        tp_gain = _int_field(row, "union_tp_node_count", 0) - baseline_tp
        fp_count = _int_field(row, "union_fp_node_count", 0)
        if tp_gain < 4:
            reasons.append("tp_gain_lt_4")
        if fp_count > 2 * baseline_fp:
            reasons.append("fp_node_count_gt_2x_baseline")
        if not _bool(row.get("channel_b_validation_only", False)):
            reasons.append("channel_b_not_validation_only")
        if _bool(row.get("channel_b_uses_topk", True)):
            reasons.append("channel_b_uses_topk")
        item["tp_node_gain_over_baseline"] = int(tp_gain)
        item["recommended"] = not reasons
        item["rejection_reasons"] = reasons
        evaluated.append(item)
    return evaluated


def _score_floor_from_candidates(rows: Sequence[Mapping[str, Any]]) -> float:
    for row in rows:
        if row.get("candidate_name") == "event_semantic_node_max_near_validation_p999":
            return _float_field(row, "score_threshold")
    raise ValueError("missing event_semantic_node_max_near_validation_p999 candidate")


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load CSV rows."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    """Write CSV rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_ready(value: Any) -> Any:
    if isinstance(value, set):
        return _sorted_ints(value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_gt_nodes(gt_dir: Path, node_map_path: Path) -> set[int]:
    gt_db_nodes = load_gt_db_nodes(sorted(Path(gt_dir).glob("*.csv")))
    node_map = load_node_id_to_idx(node_map_path)
    return {
        int(node_map[int(db_node)])
        for db_node in gt_db_nodes
        if int(db_node) in node_map
    }


def run_projection(
    *,
    result_dir: Path,
    gt_dir: Path,
    node_id_to_idx_path: Path,
) -> dict[str, Any]:
    """Run candidate projection and write sidecars."""
    result_dir = Path(result_dir)
    gt_nodes = _load_gt_nodes(gt_dir, node_id_to_idx_path)
    alert_rows = load_csv_rows(result_dir / "online_event_alerts.csv")
    node_rows = load_csv_rows(result_dir / "e5_event_semantic_node_scores.csv")
    candidate_source_rows = load_csv_rows(
        result_dir / "e5_validation_threshold_candidates.csv",
    )
    score_floor = _score_floor_from_candidates(candidate_source_rows)
    channel_a = normalize_channel_a_variants(
        build_both_cold_channel_sets(alert_rows, gt_nodes),
        alert_rows,
        gt_nodes,
    )
    channel_b = build_channel_b_variants(
        node_rows,
        gt_nodes,
        score_floor=score_floor,
    )
    candidate_rows, node_projection_rows, event_projection_rows = project_candidate_matrix(
        channel_a_variants=channel_a,
        channel_b_variants=channel_b,
        gt_nodes=gt_nodes,
    )
    evaluated_rows = recommend_candidates(
        candidate_rows,
        baseline_name="A_full__B_none",
    )
    paths = {
        "projection": result_dir / "e5_dual_channel_candidate_projection.json",
        "nodes": result_dir / "e5_dual_channel_candidate_nodes.csv",
        "events": result_dir / "e5_dual_channel_candidate_events.csv",
    }
    payload = {
        "dataset": "CLEARSCOPE_E5",
        "semantic_mode": DEFAULT_SEMANTIC_MODE,
        "result_dir": str(result_dir),
        "score_floor": float(score_floor),
        "baseline_name": "A_full__B_none",
        "candidate_projection": evaluated_rows,
        "recommended_candidates": [
            row["candidate_name"]
            for row in evaluated_rows
            if row.get("recommended")
        ],
        "leakage_contract": {
            "post_stream_projection_only": True,
            "runtime_policy_implemented": False,
            "bounded_smoke_run": False,
            "labels_used_for_threshold_selection": False,
            "channel_b_uses_topk": False,
            "original_outputs_mutated": False,
        },
    }
    write_json(paths["projection"], payload)
    write_csv(
        paths["nodes"],
        [
            "candidate_name",
            "node_idx",
            "node_label",
            "node_membership",
            "in_channel_a",
            "in_channel_b",
            "event_semantic_max_score",
            "event_semantic_p95_score",
            "event_semantic_event_count",
            "event_semantic_near_threshold_count",
            "event_semantic_above_threshold_count",
        ],
        node_projection_rows,
    )
    write_csv(
        paths["events"],
        [
            "candidate_name",
            "stream_pos",
            "event_index",
            "src_idx",
            "dst_idx",
            "action",
            "event_score",
            "threshold",
            "target_case",
            "event_has_gt_node",
            "channel_a_variant",
        ],
        event_projection_rows,
    )
    summary = {
        "dataset": "CLEARSCOPE_E5",
        "semantic_mode": DEFAULT_SEMANTIC_MODE,
        "score_floor": float(score_floor),
        "recommended_candidates": payload["recommended_candidates"],
        "candidate_count": int(len(evaluated_rows)),
        "paths": {key: str(path) for key, path in paths.items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--gt_dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--node_id_to_idx", type=Path, default=DEFAULT_NODE_MAP)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run projection CLI."""
    args = parse_args(argv)
    run_projection(
        result_dir=args.result_dir,
        gt_dir=args.gt_dir,
        node_id_to_idx_path=args.node_id_to_idx,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
