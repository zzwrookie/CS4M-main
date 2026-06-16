#!/usr/bin/env python3
"""Diagnose ClearScope E5 missed GT nodes from full-run score traces.

This is a read-only, post-stream label-aware diagnostic. It does not mutate
pipeline outputs or feed labels back into runtime logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.backfill_clearscope_e5_full_eval import load_gt_db_nodes


DEFAULT_RESULT_DIR = Path("outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE")
DEFAULT_NODE_MAP = Path(
    "outputs/cache/phase3e/node_embeddings/"
    "CLEARSCOPE_E5_v33b_full_latent64/node_id_to_idx.pkl",
)
DEFAULT_EVENT_INDEX = Path(
    "outputs/cache/phase3e/event_indices/"
    "CLEARSCOPE_E5_v33b_full/event_index_test.memmap",
)
DEFAULT_GT_DIR = Path("ground_truth/E5-CLEARSCOPE")

EVENT_INDEX_DTYPE = np.dtype(
    [
        ("event_id", "<i8"),
        ("src_node_idx", "<i4"),
        ("dst_node_idx", "<i4"),
        ("action_id", "<i2"),
        ("src_type_id", "i1"),
        ("dst_type_id", "i1"),
    ],
)


def load_node_id_to_idx(path: Path) -> dict[int, int]:
    """Load DB-node id to compact node-index mapping."""
    payload = pickle.loads(Path(path).read_bytes())
    if not isinstance(payload, Mapping):
        raise TypeError(f"node_id_to_idx must be a mapping: {path}")
    return {int(key): int(value) for key, value in payload.items()}


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


def load_alert_covered_nodes(path: Path) -> set[int]:
    """Return compact node indices touched by emitted alerts."""
    covered: set[int] = set()
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for key in ("src_idx", "dst_idx"):
                value = _int_field(row, key)
                if value >= 0:
                    covered.add(value)
    return covered


def load_seen_test_nodes(path: Path | None) -> set[int]:
    """Return compact node indices that appear in the test event-index memmap."""
    if path is None or not Path(path).exists():
        return set()
    records = np.memmap(path, dtype=EVENT_INDEX_DTYPE, mode="r")
    seen = set(int(value) for value in np.unique(records["src_node_idx"]))
    seen.update(int(value) for value in np.unique(records["dst_node_idx"]))
    return seen


def _new_node_state(db_node: int, node_idx: int) -> dict[str, Any]:
    return {
        "db_node_id": int(db_node),
        "node_idx": int(node_idx),
        "score_row_count": 0,
        "max_score": 0.0,
        "max_threshold": 0.0,
        "max_margin": 0.0,
        "max_event_id": "",
        "max_stream_pos": "",
        "max_action": "",
        "max_src_type": "",
        "max_dst_type": "",
        "max_target_case": "",
        "target_case_counts": Counter(),
        "action_type_counts": Counter(),
    }


def _score_row_nodes(row: Mapping[str, Any]) -> set[int]:
    nodes = set()
    for key in ("src_idx", "dst_idx", "info_src", "info_dst"):
        value = _int_field(row, key)
        if value >= 0:
            nodes.add(value)
    return nodes


def _update_node_state(state: dict[str, Any], row: Mapping[str, Any]) -> None:
    score = _float_field(row, "score")
    threshold = _float_field(row, "threshold")
    target_case = str(row.get("target_case", ""))
    action_key = "|".join(
        [
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
            target_case,
        ],
    )
    state["score_row_count"] = int(state["score_row_count"]) + 1
    state["target_case_counts"][target_case] += 1
    state["action_type_counts"][action_key] += 1
    if score >= float(state["max_score"]):
        state["max_score"] = float(score)
        state["max_threshold"] = float(threshold)
        state["max_margin"] = float(score - threshold)
        state["max_event_id"] = str(row.get("event_id", ""))
        state["max_stream_pos"] = str(row.get("stream_pos", ""))
        state["max_action"] = str(row.get("action", ""))
        state["max_src_type"] = str(row.get("src_type", ""))
        state["max_dst_type"] = str(row.get("dst_type", ""))
        state["max_target_case"] = target_case


def _counter_to_string(counter: Counter[str]) -> str:
    return ";".join(f"{key}:{value}" for key, value in counter.most_common())


def _classify_recommendation(summary: Mapping[str, Any]) -> str:
    missed = int(summary.get("missed_gt_count", 0))
    absent = int(summary.get("absent_from_test_gt_count", 0))
    missed_event_semantic = int(summary.get("missed_event_semantic_node_count", 0))
    missed_below = int(summary.get("missed_below_threshold_count", 0))
    both_cold = int(summary.get("missed_both_cold_node_count", 0))
    if missed and absent / max(missed, 1) >= 0.5:
        return "split_or_filter_audit"
    if missed_event_semantic >= max(both_cold, 1):
        return "event_semantic_score_collapse_audit"
    if both_cold >= max(missed_event_semantic, 1):
        return "both_cold_support_gate_audit"
    if missed_below:
        return "score_below_threshold_audit"
    return "score_below_threshold_audit"


def diagnose_scores(
    *,
    score_trace_path: Path,
    gt_db_nodes: set[int],
    node_id_to_idx: Mapping[int, int],
    alert_covered_nodes: set[int],
    seen_test_nodes: set[int],
) -> dict[str, Any]:
    """Aggregate score-trace evidence for E5 GT nodes."""
    gt_db_to_idx = {
        int(db_node): int(node_id_to_idx[db_node])
        for db_node in gt_db_nodes
        if int(db_node) in node_id_to_idx
    }
    idx_to_db_nodes: dict[int, list[int]] = defaultdict(list)
    states: dict[int, dict[str, Any]] = {}
    for db_node, node_idx in gt_db_to_idx.items():
        idx_to_db_nodes[node_idx].append(db_node)
        states[node_idx] = _new_node_state(db_node, node_idx)
    gt_idx = set(states)

    with Path(score_trace_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for node_idx in _score_row_nodes(row) & gt_idx:
                _update_node_state(states[node_idx], row)

    node_rows = []
    score_rows = []
    target_case_node_counts: Counter[str] = Counter()
    target_case_max_scores: dict[str, list[float]] = defaultdict(list)
    missed_count = 0
    missed_below = 0
    missed_event_semantic = 0
    missed_both_cold = 0
    absent_from_test = 0
    with_score_rows = 0
    covered_count = 0

    for node_idx in sorted(states):
        state = states[node_idx]
        covered = node_idx in alert_covered_nodes
        seen = node_idx in seen_test_nodes
        has_scores = int(state["score_row_count"]) > 0
        if covered:
            covered_count += 1
        else:
            missed_count += 1
        if not seen:
            absent_from_test += 1
        if has_scores:
            with_score_rows += 1
        if not covered and has_scores and float(state["max_score"]) < float(state["max_threshold"]):
            missed_below += 1
        if not covered and state["max_target_case"] == "event_semantic_target":
            missed_event_semantic += 1
        if not covered and state["max_target_case"] == "both_cold_action_target":
            missed_both_cold += 1
        target_case = str(state["max_target_case"] or "no_score_trace")
        target_case_node_counts[target_case] += 1
        target_case_max_scores[target_case].append(float(state["max_score"]))
        row = {
            "db_node_id": int(state["db_node_id"]),
            "node_idx": int(node_idx),
            "covered_by_alert": bool(covered),
            "seen_in_test": bool(seen),
            "has_score_trace": bool(has_scores),
            "score_row_count": int(state["score_row_count"]),
            "max_score": float(state["max_score"]),
            "max_threshold": float(state["max_threshold"]),
            "max_margin": float(state["max_margin"]),
            "max_event_id": str(state["max_event_id"]),
            "max_stream_pos": str(state["max_stream_pos"]),
            "max_action": str(state["max_action"]),
            "max_src_type": str(state["max_src_type"]),
            "max_dst_type": str(state["max_dst_type"]),
            "max_target_case": str(state["max_target_case"]),
            "target_case_counts": _counter_to_string(state["target_case_counts"]),
            "action_type_counts": _counter_to_string(state["action_type_counts"]),
        }
        node_rows.append(row)
        if not covered:
            score_rows.append(row)

    target_case_summary = []
    for target_case, count in sorted(target_case_node_counts.items()):
        scores = target_case_max_scores[target_case]
        target_case_summary.append(
            {
                "target_case": target_case,
                "gt_node_count": int(count),
                "max_score_mean": float(sum(scores) / len(scores)) if scores else 0.0,
                "max_score_max": max(scores) if scores else 0.0,
            },
        )

    summary = {
        "dataset": "CLEARSCOPE_E5",
        "gt_db_node_count": int(len(gt_db_nodes)),
        "gt_mapped_node_count": int(len(gt_db_to_idx)),
        "gt_nodes_seen_in_test_count": int(len(gt_idx & seen_test_nodes)),
        "gt_nodes_with_score_rows": int(with_score_rows),
        "alert_covered_gt_count": int(covered_count),
        "missed_gt_count": int(missed_count),
        "absent_from_test_gt_count": int(absent_from_test),
        "missed_below_threshold_count": int(missed_below),
        "missed_event_semantic_node_count": int(missed_event_semantic),
        "missed_both_cold_node_count": int(missed_both_cold),
        "target_case_node_counts": dict(target_case_node_counts),
        "target_case_score_summary": target_case_summary,
        "leakage_contract": {
            "post_stream_only": True,
            "labels_used_for_training": False,
            "labels_used_for_threshold_or_policy": False,
            "original_outputs_mutated": False,
        },
    }
    summary["recommendation_bucket"] = _classify_recommendation(summary)
    return {"summary": summary, "node_rows": node_rows, "missed_rows": score_rows}


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(result_dir: Path, diagnosis: Mapping[str, Any]) -> dict[str, Path]:
    """Write missed-GT score diagnosis sidecar outputs."""
    result_dir = Path(result_dir)
    summary = dict(diagnosis["summary"])
    node_rows = list(diagnosis["node_rows"])
    missed_rows = list(diagnosis["missed_rows"])
    paths = {
        "diagnosis": result_dir / "e5_missed_gt_score_diagnosis.json",
        "missed_rows": result_dir / "e5_missed_gt_score_rows.csv",
        "node_summary": result_dir / "e5_gt_score_summary_by_node.csv",
        "target_case_summary": result_dir / "e5_gt_score_summary_by_target_case.csv",
    }
    paths["diagnosis"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fieldnames = [
        "db_node_id",
        "node_idx",
        "covered_by_alert",
        "seen_in_test",
        "has_score_trace",
        "score_row_count",
        "max_score",
        "max_threshold",
        "max_margin",
        "max_event_id",
        "max_stream_pos",
        "max_action",
        "max_src_type",
        "max_dst_type",
        "max_target_case",
        "target_case_counts",
        "action_type_counts",
    ]
    _write_csv(paths["missed_rows"], fieldnames, missed_rows)
    _write_csv(paths["node_summary"], fieldnames, node_rows)
    _write_csv(
        paths["target_case_summary"],
        ["target_case", "gt_node_count", "max_score_mean", "max_score_max"],
        list(summary.get("target_case_score_summary", [])),
    )
    return paths


def run_diagnosis(
    *,
    result_dir: Path,
    gt_paths: Sequence[Path],
    node_id_to_idx_path: Path,
    event_index_path: Path,
) -> dict[str, Any]:
    """Run ClearScope E5 missed-GT score diagnosis and write sidecars."""
    result_dir = Path(result_dir)
    gt_nodes = load_gt_db_nodes(gt_paths)
    node_map = load_node_id_to_idx(node_id_to_idx_path)
    alert_nodes = load_alert_covered_nodes(result_dir / "online_event_alerts.csv")
    seen_test_nodes = load_seen_test_nodes(event_index_path)
    diagnosis = diagnose_scores(
        score_trace_path=result_dir / "online_event_score_trace.csv",
        gt_db_nodes=gt_nodes,
        node_id_to_idx=node_map,
        alert_covered_nodes=alert_nodes,
        seen_test_nodes=seen_test_nodes,
    )
    paths = write_outputs(result_dir, diagnosis)
    return {
        "summary": diagnosis["summary"],
        "paths": {key: str(value) for key, value in paths.items()},
    }


def _default_gt_paths() -> list[Path]:
    return sorted(DEFAULT_GT_DIR.glob("*.csv"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--node_id_to_idx", type=Path, default=DEFAULT_NODE_MAP)
    parser.add_argument("--event_index_test", type=Path, default=DEFAULT_EVENT_INDEX)
    parser.add_argument("--gt_csv", type=Path, action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the missed-GT score diagnosis CLI."""
    args = parse_args(argv)
    gt_paths = list(args.gt_csv) if args.gt_csv else _default_gt_paths()
    if not gt_paths:
        raise FileNotFoundError(f"no E5 GT CSV files found under {DEFAULT_GT_DIR}")
    result = run_diagnosis(
        result_dir=args.result_dir,
        gt_paths=gt_paths,
        node_id_to_idx_path=args.node_id_to_idx,
        event_index_path=args.event_index_test,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
