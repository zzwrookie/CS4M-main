#!/usr/bin/env python3
"""Backfill ClearScope E5 full-run metrics with E5 ground truth.

This tool is post-stream only. It reads existing full-run alert outputs and E5
ground-truth CSV files, then writes `e5_gt_*` sidecar reports without mutating the
original pipeline outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_RESULT_DIR = Path("outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE")
DEFAULT_NODE_MAP = Path(
    "outputs/cache/phase3e/node_embeddings/"
    "CLEARSCOPE_E5_v33b_full_latent64/node_id_to_idx.pkl",
)
DEFAULT_EVENT_INDEX = Path(
    "outputs/cache/phase3e/event_indices/"
    "CLEARSCOPE_E5_v33b_full/event_index_test.memmap",
)
DEFAULT_EVAL = DEFAULT_RESULT_DIR / "eval_causal_semantics_slim.json"
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


def load_gt_db_nodes(paths: Sequence[Path]) -> set[int]:
    """Load E5 GT DB-node ids from read-only CSV files."""
    nodes: set[int] = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    nodes.add(int(str(row[2]).strip()))
                except ValueError:
                    continue
    return nodes


def load_node_id_to_idx(path: Path) -> dict[int, int]:
    """Load DB-node id to compact node-index mapping."""
    payload = pickle.loads(Path(path).read_bytes())
    if not isinstance(payload, Mapping):
        raise TypeError(f"node_id_to_idx must be a mapping: {path}")
    return {int(key): int(value) for key, value in payload.items()}


def load_alert_rows(path: Path) -> list[dict[str, str]]:
    """Load online event alert rows as dictionaries."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _alert_nodes(row: Mapping[str, Any]) -> set[int]:
    src = _int_field(row, "src_idx")
    dst = _int_field(row, "dst_idx")
    return {value for value in (src, dst) if value >= 0}


def _group_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
            str(row.get("target_case", "")),
        ],
    )


def _counter_rows(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"key": key, "count": int(count)}
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def load_seen_test_nodes(event_index_path: Path | None) -> set[int]:
    """Return compact node indices that appear in the test event index."""
    if event_index_path is None:
        return set()
    path = Path(event_index_path)
    if not path.exists():
        return set()
    records = np.memmap(path, dtype=EVENT_INDEX_DTYPE, mode="r")
    seen = set(int(value) for value in np.unique(records["src_node_idx"]))
    seen.update(int(value) for value in np.unique(records["dst_node_idx"]))
    return seen


def _original_eval_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "dataset": payload.get("dataset", ""),
        "out_tag": payload.get("out_tag", ""),
        "semantic_mode": dict(payload.get("config", {})).get("semantic_mode", ""),
        "original_primary_online_metrics": payload.get("primary_online_metrics", {}),
        "ground_truth_used_only_for_evaluation": payload.get(
            "ground_truth_used_only_for_evaluation",
        ),
        "labels_attached_after_streaming": payload.get("labels_attached_after_streaming"),
    }


def compute_backfill_metrics(
    *,
    alert_rows: Sequence[Mapping[str, str]],
    gt_db_nodes: set[int],
    node_id_to_idx: Mapping[int, int],
    seen_test_nodes: set[int],
    original_eval: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute post-stream E5 GT backfill metrics from alert rows."""
    gt_db_to_idx = {
        int(db_node): int(node_id_to_idx[db_node])
        for db_node in gt_db_nodes
        if int(db_node) in node_id_to_idx
    }
    idx_to_gt_db: dict[int, set[int]] = defaultdict(set)
    for db_node, idx in gt_db_to_idx.items():
        idx_to_gt_db[int(idx)].add(int(db_node))
    gt_idx = set(idx_to_gt_db)

    alert_nodes: set[int] = set()
    hit_rows: list[Mapping[str, str]] = []
    event_group_counts: Counter[str] = Counter()
    hit_group_counts: Counter[str] = Counter()
    fp_group_counts: Counter[str] = Counter()
    score_by_target_case: dict[str, list[float]] = defaultdict(list)
    threshold_by_target_case: dict[str, list[float]] = defaultdict(list)

    for row in alert_rows:
        nodes = _alert_nodes(row)
        alert_nodes.update(nodes)
        key = _group_key(row)
        event_group_counts[key] += 1
        target_case = str(row.get("target_case", ""))
        score_by_target_case[target_case].append(_float_field(row, "event_score"))
        threshold_by_target_case[target_case].append(_float_field(row, "threshold"))
        if nodes & gt_idx:
            hit_rows.append(row)
            hit_group_counts[key] += 1
        else:
            fp_group_counts[key] += 1

    covered_gt_idx = alert_nodes & gt_idx
    covered_gt_db = sorted({db for idx in covered_gt_idx for db in idx_to_gt_db[idx]})
    missed_gt_db = sorted(set(gt_db_to_idx) - set(covered_gt_db))
    seen_gt_idx = gt_idx & set(seen_test_nodes)
    seen_gt_db = sorted({db for idx in seen_gt_idx for db in idx_to_gt_db[idx]})

    target_case_summary = {}
    for target_case in sorted(score_by_target_case):
        scores = score_by_target_case[target_case]
        thresholds = threshold_by_target_case[target_case]
        target_case_summary[target_case] = {
            "alert_count": int(len(scores)),
            "score_min": min(scores) if scores else 0.0,
            "score_mean": float(sum(scores) / len(scores)) if scores else 0.0,
            "score_max": max(scores) if scores else 0.0,
            "threshold_mean": float(sum(thresholds) / len(thresholds)) if thresholds else 0.0,
        }

    metrics = {
        "dataset": "CLEARSCOPE_E5",
        "gt_source": "ground_truth/E5-CLEARSCOPE/*.csv",
        "gt_db_node_count": int(len(gt_db_nodes)),
        "gt_mapped_node_count": int(len(gt_db_to_idx)),
        "gt_nodes_seen_in_test_count": int(len(seen_gt_db)),
        "alert_event_count": int(len(alert_rows)),
        "alert_node_count": int(len(alert_nodes)),
        "event_tp_any_endpoint": int(len(hit_rows)),
        "event_fp": int(len(alert_rows) - len(hit_rows)),
        "strict_node_tp": int(len(covered_gt_idx)),
        "strict_node_fp": int(len(alert_nodes - gt_idx)),
        "relaxed_node_tp": int(len(covered_gt_idx)),
        "relaxed_node_fp": int(len(alert_nodes - gt_idx)),
        "covered_gt_db_node_ids": covered_gt_db,
        "missed_gt_db_node_ids": missed_gt_db,
        "missed_gt_node_count": int(len(missed_gt_db)),
        "per_action_type_target_case": _counter_rows(event_group_counts),
        "hit_per_action_type_target_case": _counter_rows(hit_group_counts),
        "fp_per_action_type_target_case": _counter_rows(fp_group_counts),
        "target_case_score_summary": target_case_summary,
        "original_eval": dict(original_eval or {}),
        "leakage_contract": {
            "post_stream_only": True,
            "labels_used_for_training": False,
            "labels_used_for_threshold_or_policy": False,
            "original_outputs_mutated": False,
        },
    }
    return metrics


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _node_alert_rows(
    alert_rows: Sequence[Mapping[str, str]],
    gt_idx: set[int],
) -> list[dict[str, Any]]:
    node_scores: dict[int, float] = {}
    node_counts: Counter[int] = Counter()
    for row in alert_rows:
        score = _float_field(row, "event_score")
        for node in _alert_nodes(row):
            node_scores[node] = max(node_scores.get(node, 0.0), score)
            node_counts[node] += 1
    return [
        {
            "node_idx": node,
            "node_score": node_scores[node],
            "alert_count": int(node_counts[node]),
            "node_label": "malicious" if node in gt_idx else "benign",
            "eval_result": "TP" if node in gt_idx else "FP",
        }
        for node in sorted(node_scores, key=lambda item: (-node_scores[item], item))
    ]


def _missed_node_rows(
    gt_db_nodes: set[int],
    node_id_to_idx: Mapping[int, int],
    covered_db_nodes: set[int],
    seen_test_nodes: set[int],
) -> list[dict[str, Any]]:
    rows = []
    for db_node in sorted(gt_db_nodes):
        mapped = int(node_id_to_idx[db_node]) if db_node in node_id_to_idx else ""
        covered = db_node in covered_db_nodes
        rows.append(
            {
                "db_node_id": db_node,
                "node_idx": mapped,
                "mapped": bool(mapped != ""),
                "seen_in_test": bool(mapped != "" and int(mapped) in seen_test_nodes),
                "covered_by_alert": bool(covered),
            },
        )
    return [row for row in rows if not bool(row["covered_by_alert"])]


def write_backfill_outputs(
    *,
    result_dir: Path,
    alert_rows: Sequence[Mapping[str, str]],
    metrics: Mapping[str, Any],
    gt_db_nodes: set[int],
    node_id_to_idx: Mapping[int, int],
    seen_test_nodes: set[int],
) -> dict[str, Path]:
    """Write E5 GT sidecar metrics and diagnostics."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    gt_idx = {
        int(node_id_to_idx[db_node])
        for db_node in gt_db_nodes
        if int(db_node) in node_id_to_idx
    }
    covered_db = set(int(value) for value in metrics.get("covered_gt_db_node_ids", []))
    hit_alerts = [row for row in alert_rows if _alert_nodes(row) & gt_idx]
    event_rows = []
    for row in alert_rows:
        nodes = _alert_nodes(row)
        event_rows.append(
            {
                **dict(row),
                "e5_gt_event_label": "TP_any_endpoint" if nodes & gt_idx else "FP",
                "e5_gt_hit_node_indices": ";".join(str(node) for node in sorted(nodes & gt_idx)),
            },
        )

    paths = {
        "metrics": result_dir / "e5_gt_backfill_metrics.json",
        "event_alerts": result_dir / "e5_gt_backfill_event_alerts.csv",
        "node_alerts": result_dir / "e5_gt_backfill_node_alerts.csv",
        "diagnostic": result_dir / "e5_gt_diagnostic_report.json",
        "missed_nodes": result_dir / "e5_gt_missed_nodes.csv",
        "hit_alerts": result_dir / "e5_gt_hit_alerts.csv",
    }
    paths["metrics"].write_text(
        json.dumps(dict(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    event_fields = list(alert_rows[0].keys()) if alert_rows else []
    event_fields.extend(["e5_gt_event_label", "e5_gt_hit_node_indices"])
    _write_csv(paths["event_alerts"], event_fields, event_rows)
    _write_csv(
        paths["node_alerts"],
        ["node_idx", "node_score", "alert_count", "node_label", "eval_result"],
        _node_alert_rows(alert_rows, gt_idx),
    )
    missed_rows = _missed_node_rows(gt_db_nodes, node_id_to_idx, covered_db, seen_test_nodes)
    _write_csv(
        paths["missed_nodes"],
        ["db_node_id", "node_idx", "mapped", "seen_in_test", "covered_by_alert"],
        missed_rows,
    )
    hit_event_rows = [
        row for row in event_rows if row["e5_gt_event_label"] == "TP_any_endpoint"
    ]
    _write_csv(paths["hit_alerts"], event_fields, hit_event_rows)
    diagnostic = {
        "dataset": "CLEARSCOPE_E5",
        "summary": dict(metrics),
        "missed_node_count": int(len(missed_rows)),
        "hit_alert_count": int(len(hit_alerts)),
        "diagnostic_scope": "label-aware post-stream only",
    }
    paths["diagnostic"].write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def run_backfill(
    *,
    result_dir: Path,
    gt_paths: Sequence[Path],
    node_id_to_idx_path: Path,
    event_index_path: Path | None,
    original_eval_path: Path | None,
) -> dict[str, Any]:
    """Run E5 GT backfill and write sidecar outputs."""
    result_dir = Path(result_dir)
    alert_rows = load_alert_rows(result_dir / "online_event_alerts.csv")
    gt_db_nodes = load_gt_db_nodes(gt_paths)
    node_id_to_idx = load_node_id_to_idx(node_id_to_idx_path)
    seen_test_nodes = load_seen_test_nodes(event_index_path)
    original_eval = _original_eval_summary(original_eval_path)
    metrics = compute_backfill_metrics(
        alert_rows=alert_rows,
        gt_db_nodes=gt_db_nodes,
        node_id_to_idx=node_id_to_idx,
        seen_test_nodes=seen_test_nodes,
        original_eval=original_eval,
    )
    paths = write_backfill_outputs(
        result_dir=result_dir,
        alert_rows=alert_rows,
        metrics=metrics,
        gt_db_nodes=gt_db_nodes,
        node_id_to_idx=node_id_to_idx,
        seen_test_nodes=seen_test_nodes,
    )
    return {"metrics": metrics, "paths": {key: str(value) for key, value in paths.items()}}


def _default_gt_paths() -> list[Path]:
    return sorted(DEFAULT_GT_DIR.glob("*.csv"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--node_id_to_idx", type=Path, default=DEFAULT_NODE_MAP)
    parser.add_argument("--event_index_test", type=Path, default=DEFAULT_EVENT_INDEX)
    parser.add_argument("--original_eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--gt_csv", type=Path, action="append", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ClearScope E5 full-run GT backfill CLI."""
    args = parse_args(argv)
    gt_paths = list(args.gt_csv) if args.gt_csv else _default_gt_paths()
    if not gt_paths:
        raise FileNotFoundError(f"no E5 GT CSV files found under {DEFAULT_GT_DIR}")
    result = run_backfill(
        result_dir=args.result_dir,
        gt_paths=gt_paths,
        node_id_to_idx_path=args.node_id_to_idx,
        event_index_path=args.event_index_test,
        original_eval_path=args.original_eval,
    )
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
