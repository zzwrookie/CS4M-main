#!/usr/bin/env python3
"""Diagnose ClearScope E5 event-semantic and node-level separability.

This tool is read-only and post-stream. It reads existing E5 full-run outputs,
uses labels only for diagnostics after alert selection, and writes sidecar
reports without mutating original pipeline outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.backfill_clearscope_e5_full_eval import load_gt_db_nodes


DEFAULT_RESULT_DIR = Path("outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE")
DEFAULT_NODE_MAP = Path(
    "outputs/cache/phase3e/node_embeddings/"
    "CLEARSCOPE_E5_v33b_full_latent64/node_id_to_idx.pkl",
)
DEFAULT_GT_DIR = Path("ground_truth/E5-CLEARSCOPE")
DEFAULT_SEMANTIC_MODE = "raw_detail_v33b_e5_android_safe"


def _int_field(row: Mapping[str, Any], key: str, default: int = -1) -> int:
    try:
        return int(str(row.get(key, "")).strip())
    except ValueError:
        return int(default)


def _float_field(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(str(row.get(key, "")).strip())
    except ValueError:
        return float(default)


def _row_nodes(row: Mapping[str, Any]) -> set[int]:
    nodes: set[int] = set()
    for key in ("src_idx", "dst_idx", "info_src", "info_dst"):
        value = _int_field(row, key)
        if value >= 0:
            nodes.add(value)
    return nodes


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _new_node_state(node_idx: int) -> dict[str, Any]:
    return {
        "node_idx": int(node_idx),
        "event_semantic_event_count": 0,
        "event_semantic_max_score": 0.0,
        "event_semantic_max_threshold": 0.0,
        "event_semantic_max_margin": 0.0,
        "event_semantic_above_threshold_count": 0,
        "event_semantic_near_threshold_count": 0,
        "scores": [],
    }


def _classify_node_group(
    *,
    node_idx: int,
    gt_nodes: set[int],
    current_alert_nodes: set[int],
    current_gt_alert_nodes: set[int],
    current_fp_nodes: set[int],
) -> tuple[str, str, str]:
    label = "malicious" if node_idx in gt_nodes else "benign"
    covered = node_idx in current_alert_nodes
    if node_idx in current_gt_alert_nodes:
        return label, "current_both_cold_tp_node", "current_gt_alert"
    if node_idx in gt_nodes:
        return label, "missed_gt_node", "missed_gt"
    if node_idx in current_fp_nodes:
        return label, "current_fp_node", "current_fp_alert"
    if covered:
        return label, "current_alert_other_node", "current_alert_other"
    return label, "other_benign_node", "none"


def compute_event_semantic_node_scores(
    *,
    score_rows: Iterable[Mapping[str, Any]],
    gt_nodes: set[int],
    current_alert_nodes: set[int],
    current_gt_alert_nodes: set[int],
    current_fp_nodes: set[int],
    near_threshold_ratio: float = 0.5,
) -> list[dict[str, Any]]:
    """Aggregate event-semantic score trace rows into node-level diagnostics."""
    states: dict[int, dict[str, Any]] = {}
    for node_idx in gt_nodes:
        states[int(node_idx)] = _new_node_state(int(node_idx))

    for row in score_rows:
        if str(row.get("target_case", "")) != "event_semantic_target":
            continue
        score = _float_field(row, "score")
        threshold = _float_field(row, "threshold")
        for node_idx in _row_nodes(row):
            state = states.setdefault(node_idx, _new_node_state(node_idx))
            state["event_semantic_event_count"] += 1
            state["scores"].append(score)
            if score >= threshold:
                state["event_semantic_above_threshold_count"] += 1
            if threshold and score >= threshold * near_threshold_ratio:
                state["event_semantic_near_threshold_count"] += 1
            if score >= state["event_semantic_max_score"]:
                state["event_semantic_max_score"] = score
                state["event_semantic_max_threshold"] = threshold
                state["event_semantic_max_margin"] = score - threshold

    rows: list[dict[str, Any]] = []
    for node_idx in sorted(states):
        state = states[node_idx]
        scores = list(state.pop("scores"))
        label, group, alert_role = _classify_node_group(
            node_idx=node_idx,
            gt_nodes=gt_nodes,
            current_alert_nodes=current_alert_nodes,
            current_gt_alert_nodes=current_gt_alert_nodes,
            current_fp_nodes=current_fp_nodes,
        )
        rows.append(
            {
                "node_idx": int(node_idx),
                "node_label": label,
                "node_group": group,
                "covered_by_current_alert": bool(node_idx in current_alert_nodes),
                "current_alert_role": alert_role,
                "event_semantic_event_count": int(state["event_semantic_event_count"]),
                "event_semantic_max_score": float(state["event_semantic_max_score"]),
                "event_semantic_p95_score": _percentile(scores, 0.95),
                "event_semantic_p99_score": _percentile(scores, 0.99),
                "event_semantic_mean_score": float(mean(scores)) if scores else 0.0,
                "event_semantic_max_threshold": float(state["event_semantic_max_threshold"]),
                "event_semantic_max_margin": float(state["event_semantic_max_margin"]),
                "event_semantic_above_threshold_count": int(
                    state["event_semantic_above_threshold_count"],
                ),
                "event_semantic_near_threshold_count": int(
                    state["event_semantic_near_threshold_count"],
                ),
            },
        )
    return rows


def compute_topk_metrics(
    node_rows: Sequence[Mapping[str, Any]],
    *,
    gt_nodes: set[int],
    score_fields: Sequence[str],
    k_values: Sequence[int],
) -> list[dict[str, Any]]:
    """Compute node-level GT coverage after ranking by score fields."""
    metrics: list[dict[str, Any]] = []
    total_gt = len(gt_nodes)
    for score_field in score_fields:
        ranked = sorted(
            node_rows,
            key=lambda row: (_float_field(row, score_field), -_int_field(row, "node_idx")),
            reverse=True,
        )
        for k in k_values:
            selected = ranked[: min(int(k), len(ranked))]
            selected_nodes = {_int_field(row, "node_idx") for row in selected}
            tp = len(selected_nodes & gt_nodes)
            count = len(selected_nodes)
            fp = count - tp
            metrics.append(
                {
                    "score_field": score_field,
                    "k": int(k),
                    "count": int(count),
                    "tp": int(tp),
                    "fp": int(fp),
                    "precision": float(tp / count) if count else 0.0,
                    "recall": float(tp / total_gt) if total_gt else 0.0,
                },
            )
    return metrics


def _alert_key(row: Mapping[str, Any]) -> str:
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


def _event_has_gt(row: Mapping[str, Any], gt_nodes: set[int]) -> bool:
    return bool(_row_nodes(row) & gt_nodes)


def _node_repeat_counts(alert_rows: Sequence[Mapping[str, Any]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in alert_rows:
        for node in _row_nodes(row):
            counts[node] += 1
    return counts


def _endpoint_action_counts(alert_rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(_alert_key(row) for row in alert_rows)


def _event_semantic_p95_for_alert(
    row: Mapping[str, Any],
    event_semantic_scores: Mapping[int, Mapping[str, Any]],
) -> float:
    values = []
    for node in _row_nodes(row):
        values.append(
            _float_field(
                event_semantic_scores.get(node, {}),
                "event_semantic_p95_score",
            ),
        )
    return max(values) if values else 0.0


def _summarize_gate(
    *,
    gate_name: str,
    retained_rows: Sequence[Mapping[str, Any]],
    total_rows: int,
    gt_nodes: set[int],
) -> dict[str, Any]:
    retained_nodes: set[int] = set()
    retained_gt_nodes: set[int] = set()
    tp_events = 0
    for row in retained_rows:
        nodes = _row_nodes(row)
        retained_nodes.update(nodes)
        if nodes & gt_nodes:
            tp_events += 1
            retained_gt_nodes.update(nodes & gt_nodes)
    fp_events = len(retained_rows) - tp_events
    return {
        "gate_name": gate_name,
        "total_candidate_event_alerts": int(total_rows),
        "retained_event_alerts": int(len(retained_rows)),
        "suppressed_event_alerts": int(total_rows - len(retained_rows)),
        "retained_tp_any_endpoint": int(tp_events),
        "retained_fp": int(fp_events),
        "retained_gt_node_count": int(len(retained_gt_nodes)),
        "retained_alert_node_count": int(len(retained_nodes)),
        "fp_reduction": float((total_rows - len(retained_rows)) / total_rows)
        if total_rows
        else 0.0,
        "tp_event_retention": float(tp_events / max(tp_events, 1)),
        "labels_used_after_selection": True,
    }


def simulate_both_cold_gates(
    *,
    alert_rows: Sequence[Mapping[str, Any]],
    gt_nodes: set[int],
    event_semantic_scores: Mapping[int, Mapping[str, Any]],
    event_semantic_global_p999: float = 0.0,
) -> list[dict[str, Any]]:
    """Simulate label-free gates over current both-cold alert rows."""
    both_cold_rows = [
        row
        for row in alert_rows
        if str(row.get("target_case", "")) == "both_cold_action_target"
    ]
    node_counts = _node_repeat_counts(both_cold_rows)
    endpoint_counts = _endpoint_action_counts(both_cold_rows)

    original_tp = sum(1 for row in both_cold_rows if _event_has_gt(row, gt_nodes))

    candidates: list[tuple[str, list[Mapping[str, Any]]]] = [
        (
            "node_repeat_ge_2",
            [
                row
                for row in both_cold_rows
                if max((node_counts[node] for node in _row_nodes(row)), default=0) >= 2
            ],
        ),
        (
            "node_repeat_ge_3",
            [
                row
                for row in both_cold_rows
                if max((node_counts[node] for node in _row_nodes(row)), default=0) >= 3
            ],
        ),
        (
            "node_repeat_ge_5",
            [
                row
                for row in both_cold_rows
                if max((node_counts[node] for node in _row_nodes(row)), default=0) >= 5
            ],
        ),
        (
            "endpoint_action_repeat_ge_2",
            [row for row in both_cold_rows if endpoint_counts[_alert_key(row)] >= 2],
        ),
        (
            "endpoint_action_repeat_ge_3",
            [row for row in both_cold_rows if endpoint_counts[_alert_key(row)] >= 3],
        ),
        (
            "node_repeat_ge_2_or_event_semantic_p95_ge_global_p999",
            [
                row
                for row in both_cold_rows
                if max((node_counts[node] for node in _row_nodes(row)), default=0) >= 2
                or _event_semantic_p95_for_alert(row, event_semantic_scores)
                >= event_semantic_global_p999
            ],
        ),
    ]

    diagnostics = []
    for name, rows in candidates:
        summary = _summarize_gate(
            gate_name=name,
            retained_rows=rows,
            total_rows=len(both_cold_rows),
            gt_nodes=gt_nodes,
        )
        retained_tp = int(summary["retained_tp_any_endpoint"])
        summary["original_tp_any_endpoint"] = int(original_tp)
        summary["lost_tp_any_endpoint"] = int(max(original_tp - retained_tp, 0))
        summary["tp_event_retention"] = (
            float(retained_tp / original_tp) if original_tp else 0.0
        )
        diagnostics.append(summary)
    return diagnostics


def load_node_id_to_idx(path: Path) -> dict[int, int]:
    """Load DB-node id to compact node-index mapping."""
    payload = pickle.loads(Path(path).read_bytes())
    if not isinstance(payload, Mapping):
        raise TypeError(f"node_id_to_idx must be a mapping: {path}")
    return {int(key): int(value) for key, value in payload.items()}


def load_alert_rows(path: Path) -> list[dict[str, str]]:
    """Load online event alert rows."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _current_alert_sets(
    alert_rows: Sequence[Mapping[str, Any]],
    gt_nodes: set[int],
) -> tuple[set[int], set[int], set[int]]:
    alert_nodes: set[int] = set()
    gt_alert_nodes: set[int] = set()
    fp_nodes: set[int] = set()
    for row in alert_rows:
        nodes = _row_nodes(row)
        alert_nodes.update(nodes)
        if nodes & gt_nodes:
            gt_alert_nodes.update(nodes & gt_nodes)
        fp_nodes.update(nodes - gt_nodes)
    return alert_nodes, gt_alert_nodes, fp_nodes


def _stream_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def _event_semantic_global_p999(summary_path: Path) -> float:
    if not summary_path.exists():
        return 0.0
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("target_case", "")) == "event_semantic_target":
                return _float_field(row, "test_p999")
    return 0.0


def _summary_by_group(node_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in node_rows:
        grouped[str(row.get("node_group", ""))].append(row)

    summaries = []
    for group, rows in sorted(grouped.items()):
        max_scores = [_float_field(row, "event_semantic_max_score") for row in rows]
        p95_scores = [_float_field(row, "event_semantic_p95_score") for row in rows]
        event_counts = [_int_field(row, "event_semantic_event_count") for row in rows]
        summaries.append(
            {
                "node_group": group,
                "node_count": int(len(rows)),
                "max_score_mean": float(mean(max_scores)) if max_scores else 0.0,
                "max_score_p95": _percentile(max_scores, 0.95),
                "max_score_max": max(max_scores) if max_scores else 0.0,
                "p95_score_mean": float(mean(p95_scores)) if p95_scores else 0.0,
                "event_count_mean": float(mean(event_counts)) if event_counts else 0.0,
                "event_count_max": max(event_counts) if event_counts else 0,
            },
        )
    return summaries


def _counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"key": key, "count": int(count)}
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _recommended_next_step(
    *,
    topk_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
) -> str:
    max_score_top100 = [
        row
        for row in topk_rows
        if row.get("score_field") == "event_semantic_max_score" and int(row.get("k", 0)) == 100
    ]
    top100_recall = _float_field(max_score_top100[0], "recall") if max_score_top100 else 0.0
    best_gate = max(gate_rows, key=lambda row: _float_field(row, "fp_reduction"), default={})
    if top100_recall >= 0.25:
        return "event_semantic_validation_threshold_audit"
    if _float_field(best_gate, "fp_reduction") >= 0.25 and _float_field(
        best_gate,
        "tp_event_retention",
    ) >= 0.75:
        return "validation_only_cold_start_gate_smoke"
    return "phase3g_event_semantic_score_collapse_audit"


def run_diagnosis(
    *,
    result_dir: Path,
    gt_paths: Sequence[Path],
    node_id_to_idx_path: Path,
) -> dict[str, Any]:
    """Run the E5 event-semantic/node separation diagnostic."""
    result_dir = Path(result_dir)
    gt_db_nodes = load_gt_db_nodes(gt_paths)
    node_map = load_node_id_to_idx(node_id_to_idx_path)
    gt_nodes = {
        int(node_map[db_node])
        for db_node in gt_db_nodes
        if int(db_node) in node_map
    }

    alert_rows = load_alert_rows(result_dir / "online_event_alerts.csv")
    current_alert_nodes, current_gt_alert_nodes, current_fp_nodes = _current_alert_sets(
        alert_rows,
        gt_nodes,
    )
    node_rows = compute_event_semantic_node_scores(
        score_rows=_stream_csv_rows(result_dir / "online_event_score_trace.csv"),
        gt_nodes=gt_nodes,
        current_alert_nodes=current_alert_nodes,
        current_gt_alert_nodes=current_gt_alert_nodes,
        current_fp_nodes=current_fp_nodes,
    )
    event_semantic_by_node = {
        int(row["node_idx"]): row
        for row in node_rows
    }
    topk_rows = compute_topk_metrics(
        node_rows,
        gt_nodes=gt_nodes,
        score_fields=[
            "event_semantic_max_score",
            "event_semantic_p95_score",
            "event_semantic_event_count",
        ],
        k_values=[50, 100, 200, 500, 1000],
    )
    p999 = _event_semantic_global_p999(
        result_dir / "conditional_score_summary_by_target_action_type.csv",
    )
    gate_rows = simulate_both_cold_gates(
        alert_rows=alert_rows,
        gt_nodes=gt_nodes,
        event_semantic_scores=event_semantic_by_node,
        event_semantic_global_p999=p999,
    )
    group_summary = _summary_by_group(node_rows)
    target_case_counts = Counter(str(row.get("target_case", "")) for row in alert_rows)

    paths = {
        "summary": result_dir / "e5_event_semantic_node_separation.json",
        "node_scores": result_dir / "e5_event_semantic_node_scores.csv",
        "topk_metrics": result_dir / "e5_event_semantic_topk_metrics.csv",
        "gate_diagnostic": result_dir / "e5_both_cold_tp_fp_gate_diagnostic.csv",
    }
    _write_csv(
        paths["node_scores"],
        [
            "node_idx",
            "node_label",
            "node_group",
            "covered_by_current_alert",
            "current_alert_role",
            "event_semantic_event_count",
            "event_semantic_max_score",
            "event_semantic_p95_score",
            "event_semantic_p99_score",
            "event_semantic_mean_score",
            "event_semantic_max_threshold",
            "event_semantic_max_margin",
            "event_semantic_above_threshold_count",
            "event_semantic_near_threshold_count",
        ],
        node_rows,
    )
    _write_csv(
        paths["topk_metrics"],
        ["score_field", "k", "count", "tp", "fp", "precision", "recall"],
        topk_rows,
    )
    _write_csv(
        paths["gate_diagnostic"],
        [
            "gate_name",
            "total_candidate_event_alerts",
            "retained_event_alerts",
            "suppressed_event_alerts",
            "original_tp_any_endpoint",
            "retained_tp_any_endpoint",
            "lost_tp_any_endpoint",
            "retained_fp",
            "retained_gt_node_count",
            "retained_alert_node_count",
            "fp_reduction",
            "tp_event_retention",
            "labels_used_after_selection",
        ],
        gate_rows,
    )

    summary = {
        "dataset": "CLEARSCOPE_E5",
        "semantic_mode": DEFAULT_SEMANTIC_MODE,
        "result_dir": str(result_dir),
        "gt_db_node_count": int(len(gt_db_nodes)),
        "gt_mapped_node_count": int(len(gt_nodes)),
        "current_event_alert_count": int(len(alert_rows)),
        "current_alert_target_case_counts": dict(target_case_counts),
        "current_alert_gt_node_count": int(len(current_gt_alert_nodes)),
        "current_alert_fp_node_count": int(len(current_fp_nodes)),
        "event_semantic_global_p999": float(p999),
        "event_semantic_node_count": int(len(node_rows)),
        "event_semantic_group_summary": group_summary,
        "topk_metrics": topk_rows,
        "both_cold_gate_diagnostic": gate_rows,
        "recommended_next_step": _recommended_next_step(
            topk_rows=topk_rows,
            gate_rows=gate_rows,
        ),
        "paths": {key: str(value) for key, value in paths.items()},
        "leakage_contract": {
            "post_stream_only": True,
            "labels_used_for_training": False,
            "labels_used_for_threshold_or_policy": False,
            "labels_used_for_runtime_gate_selection": False,
            "labels_used_after_candidate_selection_for_diagnostics": True,
            "original_outputs_mutated": False,
        },
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _default_gt_paths() -> list[Path]:
    return sorted(DEFAULT_GT_DIR.glob("*.csv"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--node_id_to_idx", type=Path, default=DEFAULT_NODE_MAP)
    parser.add_argument("--gt_csv", type=Path, action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the diagnostic CLI."""
    args = parse_args(argv)
    gt_paths = list(args.gt_csv) if args.gt_csv else _default_gt_paths()
    if not gt_paths:
        raise FileNotFoundError(f"no E5 GT CSV files found under {DEFAULT_GT_DIR}")
    summary = run_diagnosis(
        result_dir=args.result_dir,
        gt_paths=gt_paths,
        node_id_to_idx_path=args.node_id_to_idx,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
