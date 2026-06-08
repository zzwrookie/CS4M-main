#!/usr/bin/env python3
"""Diagnose SSPM node-state mixing risk from train/test streams.

The script is read-only with respect to model training: it streams CADETS/THEIA
events, derives the same info-flow node ids and residual tokens used by SSPM,
and summarizes node frequency, neighbor fanout, action entropy, residual detail
entropy, and test-label collisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.config.provnet_utils import init_database_connection
from scripts.data.get_dataset import get_dataset_splits, parse_split_days, use_event_type_filter
from legacy.compatibility.pipeline_runtime_exports import (
    SlimConfig,
    _load_process_config,
    _residual_tokens_for_row,
    build_node_maps,
    load_ground_truth_indices_readonly,
    row_fields,
    stream_dataset_rows,
)
from scripts.pipeline.io.db_stream import _cfg_for_dataset


def entropy_bits(counts: Mapping[str, int]) -> float:
    """Return Shannon entropy in bits for a count mapping."""
    total = float(sum(int(value) for value in counts.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        count = int(value)
        if count <= 0:
            continue
        probability = float(count / total)
        entropy -= probability * math.log2(probability)
    return float(entropy)


def label_group(label: object) -> str:
    """Return benign, suspicious, or malicious for event labels."""
    text = str(label).strip().lower()
    if text in {"2", "malicious"}:
        return "malicious"
    if text in {"1", "suspicious", "true"}:
        return "suspicious"
    return "benign"


@dataclass
class NodeStats:
    """Per-node streaming diagnostic counters."""

    node_id: int
    node_role: str = ""
    node_type: int = -1
    event_count: int = 0
    neighbor_ids: set[int] = field(default_factory=set)
    action_counts: Counter[str] = field(default_factory=Counter)
    detail_counts: Counter[str] = field(default_factory=Counter)
    label_counts: Counter[str] = field(default_factory=Counter)

    def observe(
        self,
        neighbor_id: int,
        action: str,
        details: Iterable[str],
        label: str,
        role: str,
        node_type: int,
    ) -> None:
        """Observe one event touching this node."""
        self.event_count += 1
        self.neighbor_ids.add(int(neighbor_id))
        self.action_counts[str(action)] += 1
        for detail in details:
            self.detail_counts[str(detail)] += 1
        self.label_counts[str(label)] += 1
        if not self.node_role:
            self.node_role = str(role)
        if self.node_type < 0:
            self.node_type = int(node_type)

    def row(self, split: str) -> dict[str, Any]:
        """Return a CSV-safe row."""
        benign = int(self.label_counts.get("benign", 0))
        suspicious = int(self.label_counts.get("suspicious", 0))
        malicious = int(self.label_counts.get("malicious", 0))
        attack = suspicious + malicious
        return {
            "split": split,
            "node_id": int(self.node_id),
            "node_role": self.node_role,
            "node_type": int(self.node_type),
            "event_count": int(self.event_count),
            "neighbor_count": int(len(self.neighbor_ids)),
            "action_entropy": float(entropy_bits(self.action_counts)),
            "detail_entropy": float(entropy_bits(self.detail_counts)),
            "distinct_actions": int(len(self.action_counts)),
            "distinct_details": int(len(self.detail_counts)),
            "benign_events": benign,
            "suspicious_events": suspicious,
            "malicious_events": malicious,
            "attack_events": attack,
            "label_collision": int(benign > 0 and attack > 0),
            "top_actions": _format_top(self.action_counts, 5),
            "top_details": _format_top(self.detail_counts, 8),
        }


def _format_top(counter: Counter[str], limit: int) -> str:
    return ";".join(f"{key}:{value}" for key, value in counter.most_common(int(limit)))


def _detail_tokens(tokens: list[str], action: str) -> list[str]:
    return [token for token in tokens if token not in {"process", "file", "netflow", action}]


def _role_for_info_node(row: Mapping[str, Any], fields: Mapping[str, Any], node_id: int) -> str:
    """Return the original endpoint role for an info-flow node id."""
    src_idx = int(row.get("src_idx", -1))
    dst_idx = int(row.get("dst_idx", -1))
    if int(node_id) == src_idx:
        return str(fields["src_role"])
    if int(node_id) == dst_idx:
        return str(fields["dst_role"])
    if int(node_id) == int(fields.get("info_src", -2)):
        return str(fields["src_role"])
    if int(node_id) == int(fields.get("info_dst", -3)):
        return str(fields["dst_role"])
    return ""


def _observe_row(
    stats: dict[int, NodeStats],
    row: Mapping[str, Any],
    config: SlimConfig,
    process_cfg: Any,
) -> None:
    fields = row_fields(
        row,
        config.max_tokens_per_node,
        dataset=config.dataset,
        process_semantic_config=process_cfg,
        theia_netflow_policy=config.theia_netflow_policy,
    )
    tokens = _residual_tokens_for_row(row, config, process_cfg)
    action = str(fields["action"])
    details = _detail_tokens(tokens, action)
    label = label_group(row.get("label", 0))
    src = int(fields["info_src"])
    dst = int(fields["info_dst"])
    if src not in stats:
        stats[src] = NodeStats(node_id=src)
    if dst not in stats:
        stats[dst] = NodeStats(node_id=dst)
    stats[src].observe(
        neighbor_id=dst,
        action=action,
        details=details,
        label=label,
        role=_role_for_info_node(row, fields, src),
        node_type=int(fields["info_src_type"]),
    )
    stats[dst].observe(
        neighbor_id=src,
        action=action,
        details=details,
        label=label,
        role=_role_for_info_node(row, fields, dst),
        node_type=int(fields["info_dst_type"]),
    )


def _quantiles(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    def pick(q: float) -> float:
        index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
        return float(ordered[index])
    return {
        "min": float(ordered[0]),
        "p50": pick(0.50),
        "p90": pick(0.90),
        "p99": pick(0.99),
        "max": float(ordered[-1]),
    }


def _split_summary(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    event_counts = [int(row["event_count"]) for row in rows]
    neighbor_counts = [int(row["neighbor_count"]) for row in rows]
    action_entropy = [float(row["action_entropy"]) for row in rows]
    detail_entropy = [float(row["detail_entropy"]) for row in rows]
    collisions = [row for row in rows if int(row["label_collision"]) == 1]
    return {
        "split": split,
        "node_count": int(len(rows)),
        "event_count_quantiles": _quantiles(event_counts),
        "neighbor_count_quantiles": _quantiles(neighbor_counts),
        "action_entropy_quantiles": _quantiles(action_entropy),
        "detail_entropy_quantiles": _quantiles(detail_entropy),
        "nodes_with_lt3_events": int(sum(1 for value in event_counts if value < 3)),
        "nodes_with_lt3_events_ratio": float(
            sum(1 for value in event_counts if value < 3) / max(len(rows), 1),
        ),
        "label_collision_nodes": int(len(collisions)),
        "top_by_event_count": sorted(rows, key=lambda row: int(row["event_count"]), reverse=True)[:20],
        "top_by_neighbor_count": sorted(
            rows,
            key=lambda row: int(row["neighbor_count"]),
            reverse=True,
        )[:20],
        "top_collision_nodes": sorted(
            collisions,
            key=lambda row: (
                int(row["attack_events"]),
                int(row["event_count"]),
                int(row["neighbor_count"]),
            ),
            reverse=True,
        )[:50],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "node_id",
        "node_role",
        "node_type",
        "event_count",
        "neighbor_count",
        "action_entropy",
        "detail_entropy",
        "distinct_actions",
        "distinct_details",
        "benign_events",
        "suspicious_events",
        "malicious_events",
        "attack_events",
        "label_collision",
        "top_actions",
        "top_details",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _days_for_split(db_cfg: Any, split: str) -> list[int]:
    return parse_split_days(get_dataset_splits(db_cfg, split))


def diagnose_split(
    conn: Any,
    dataset: str,
    db_cfg: Any,
    node_maps: Mapping[str, Any],
    split: str,
    days: list[int],
    abnormal_nodes: set[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Diagnose one split and return node rows plus summary."""
    config = SlimConfig(dataset=dataset)
    process_cfg = _load_process_config(config)
    include_labels = split == "test"
    stats: dict[int, NodeStats] = {}
    started = time.perf_counter()
    rows = stream_dataset_rows(
        conn,
        str(db_cfg.dataset.year_month),
        days,
        node_maps,
        use_event_type_filter(db_cfg),
        int(args.fetch_size),
        int(args.max_events),
        abnormal_nodes=abnormal_nodes if include_labels else set(),
    )
    count = 0
    for row in rows:
        _observe_row(stats, row, config, process_cfg)
        count += 1
        if count % int(args.progress_interval_events) == 0:
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(
                f"[node-mixing] split={split} events={count} nodes={len(stats)} "
                f"rate={count / elapsed:.1f}/s",
                file=sys.stderr,
                flush=True,
            )
    node_rows = [node_stats.row(split) for node_stats in stats.values()]
    node_rows.sort(key=lambda row: int(row["event_count"]), reverse=True)
    summary = _split_summary(split, node_rows)
    summary["event_rows"] = int(count)
    summary["elapsed_sec"] = float(time.perf_counter() - started)
    summary["days"] = [int(day) for day in days]
    summary["labels_included"] = bool(include_labels)
    return node_rows, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run train/test node mixing diagnostics."""
    dataset = str(args.dataset).upper()
    db_cfg = _cfg_for_dataset(dataset)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    cur, conn = init_database_connection(db_cfg)
    try:
        print(f"[node-mixing] build_node_maps_start dataset={dataset}", file=sys.stderr, flush=True)
        node_maps = build_node_maps(cur)
        abnormal_nodes = load_ground_truth_indices_readonly(db_cfg, node_maps["uuid2index"])
        print(
            f"[node-mixing] build_node_maps_end dataset={dataset} abnormal_nodes={len(abnormal_nodes)}",
            file=sys.stderr,
            flush=True,
        )
        out_dir = Path(args.out_dir)
        payload: dict[str, Any] = {
            "dataset": dataset,
            "node_key": "SSPM info_src/info_dst after information_flow",
            "validation_used": False,
            "train_labels_included": False,
            "test_labels_included_for_diagnostics_only": True,
            "splits": {},
        }
        for split in ("train", "test"):
            split_rows, split_summary = diagnose_split(
                conn,
                dataset,
                db_cfg,
                node_maps,
                split,
                _days_for_split(db_cfg, split),
                abnormal_nodes,
                args,
            )
            csv_path = out_dir / f"{dataset.lower()}_{split}_node_mixing.csv"
            _write_csv(csv_path, split_rows)
            split_summary["csv_path"] = str(csv_path)
            payload["splits"][split] = split_summary
        json_path = out_dir / f"{dataset.lower()}_node_mixing_summary.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return payload
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CADETS_E3")
    parser.add_argument("--out_dir", default="outputs/diagnostics/node_mixing")
    parser.add_argument("--max_events", type=int, default=0)
    parser.add_argument("--fetch_size", type=int, default=10000)
    parser.add_argument("--progress_interval_events", type=int, default=500000)
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    run(parse_args())


if __name__ == "__main__":
    main()
