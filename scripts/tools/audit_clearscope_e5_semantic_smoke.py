#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.semantics.clearscope_android import (
    CLEARSCOPE_V31_SEMANTIC_MODE,
    CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE,
    CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE,
    android_process_natural_tokens_refined,
    clearscope_file_natural_tokens_v31,
    clearscope_file_natural_tokens_v33b_e5_android_safe,
    clearscope_file_natural_tokens_v33_e5_android_safe,
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
DEFAULT_GROUND_TRUTH_PATHS = [
    "ground_truth/E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv",
    "ground_truth/E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv",
    "ground_truth/E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv",
]


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
    if normalized_mode not in {
        CLEARSCOPE_V31_SEMANTIC_MODE,
        CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE,
        CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE,
    }:
        raise ValueError(f"unsupported E5 audit semantic mode: {semantic_mode}")
    if node_type == "file":
        raw_detail = _row_text(row, "path")
        if normalized_mode == CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE:
            tokens = clearscope_file_natural_tokens_v33b_e5_android_safe(raw_detail)
        elif normalized_mode == CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE:
            tokens = clearscope_file_natural_tokens_v33_e5_android_safe(raw_detail)
        else:
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
            token: sorted({_display_raw_detail(raw_value) for raw_value in raw_values})
            for token, raw_values in sorted(collision_groups.items())
            if len(raw_values) > 1
        },
    }


def _event_tuple(
    event: dict[str, Any],
    nodes_by_index: dict[int, AuditNode],
) -> tuple[str, ...] | None:
    src = nodes_by_index.get(int(event["src_index_id"]))
    dst = nodes_by_index.get(int(event["dst_index_id"]))
    if src is None or dst is None:
        return None
    return (
        str(event["operation"]),
        src.node_type,
        dst.node_type,
        extract_detail_token(src),
        extract_detail_token(dst),
    )


def build_event_tuple_summary(
    events: Iterable[dict[str, Any]],
    nodes_by_index: dict[int, AuditNode],
) -> dict[str, object]:
    """Summarize label-free event tuple support and held-out OOV."""
    input_event_count = 0
    event_count = 0
    skipped_event_count = 0
    skipped_missing_src_count = 0
    skipped_missing_dst_count = 0
    train_tuples: Counter[tuple[str, ...]] = Counter()
    val_tuples: Counter[tuple[str, ...]] = Counter()
    test_tuples: Counter[tuple[str, ...]] = Counter()
    split_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    for event in events:
        input_event_count += 1
        src = nodes_by_index.get(int(event["src_index_id"]))
        dst = nodes_by_index.get(int(event["dst_index_id"]))
        if src is None or dst is None:
            skipped_event_count += 1
            if src is None:
                skipped_missing_src_count += 1
            if dst is None:
                skipped_missing_dst_count += 1
            continue
        tuple_key = (
            str(event["operation"]),
            src.node_type,
            dst.node_type,
            extract_detail_token(src),
            extract_detail_token(dst),
        )
        event_count += 1
        split = str(event["split"])
        operation = str(event["operation"])
        split_counts[split] += 1
        operation_counts[operation] += 1
        if split == "train":
            train_tuples[tuple_key] += 1
        elif split == "val":
            val_tuples[tuple_key] += 1
        elif split == "test":
            test_tuples[tuple_key] += 1
    val_oov = {
        tuple_key: count
        for tuple_key, count in val_tuples.items()
        if tuple_key not in train_tuples
    }
    val_seen = {
        tuple_key: count
        for tuple_key, count in val_tuples.items()
        if tuple_key in train_tuples
    }
    test_oov = {
        tuple_key: count
        for tuple_key, count in test_tuples.items()
        if tuple_key not in train_tuples
    }
    test_seen = {
        tuple_key: count
        for tuple_key, count in test_tuples.items()
        if tuple_key in train_tuples
    }
    skipped_rate = skipped_event_count / input_event_count if input_event_count else 0.0
    usable_event_rate = event_count / input_event_count if input_event_count else 0.0
    report_warnings = []
    if skipped_rate > 0.20:
        report_warnings.append("high_skipped_event_rate")
    return {
        "input_event_count": input_event_count,
        "event_count": event_count,
        "skipped_event_count": skipped_event_count,
        "skipped_missing_src_count": skipped_missing_src_count,
        "skipped_missing_dst_count": skipped_missing_dst_count,
        "skipped_rate": skipped_rate,
        "usable_event_rate": usable_event_rate,
        "report_warnings": report_warnings,
        "split_counts": dict(sorted(split_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "train_tuple_count": len(train_tuples),
        "val_tuple_count": len(val_tuples),
        "val_oov_tuple_count": len(val_oov),
        "val_seen_tuple_count": len(val_seen),
        "test_tuple_count": len(test_tuples),
        "test_oov_tuple_count": len(test_oov),
        "test_seen_tuple_count": len(test_seen),
        "top_val_oov_tuples": [
            {"tuple": list(tuple_key), "count": count}
            for tuple_key, count in Counter(val_oov).most_common(50)
        ],
        "top_test_oov_tuples": [
            {"tuple": list(tuple_key), "count": count}
            for tuple_key, count in Counter(test_oov).most_common(50)
        ],
    }


def build_collision_rows(
    nodes: Iterable[AuditNode],
    min_raw_details: int = 2,
) -> list[dict[str, object]]:
    """Return detail tokens that collapse multiple raw details."""
    raw_by_token: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        raw_by_token[extract_detail_token(node)].add(node.raw_detail)
    rows = [
        {
            "token": token,
            "raw_detail_count": len(raw_values),
            "examples": sorted({_display_raw_detail(raw_value) for raw_value in raw_values})[:10],
        }
        for token, raw_values in raw_by_token.items()
        if len(raw_values) >= int(min_raw_details)
    ]
    return sorted(
        rows,
        key=lambda row: (-int(row["raw_detail_count"]), str(row["token"])),
    )


def _display_raw_detail(raw_detail: object) -> str:
    text = str(raw_detail or "")
    if text == "/acct":
        return "/acct"
    if re.fullmatch(r"/acct/uid_[0-9]+", text):
        return "/acct/uid_<uid>"
    if re.fullmatch(r"/acct/uid_[0-9]+/pid_[0-9]+", text):
        return "/acct/uid_<uid>/pid_<pid>"
    return text


def fallback_rows_from_summary(summary: dict[str, object]) -> list[dict[str, object]]:
    """Return fallback count rows for CSV output."""
    counts = dict(summary.get("fallback_counts", {}))
    return [{"token": token, "count": count} for token, count in sorted(counts.items())]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if not fieldnames:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def write_reports(
    output_dir: str | os.PathLike[str],
    label_free_summary: dict[str, object],
    label_aware_diagnostics: dict[str, object],
    fallback_rows: list[dict[str, object]],
    collision_rows: list[dict[str, object]],
) -> dict[str, str]:
    """Write E5 audit reports and return generated paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "label_free_json": str(out / "label_free_summary.json"),
        "label_aware_json": str(out / "label_aware_diagnostics.json"),
        "fallback_csv": str(out / "fallback_counts.csv"),
        "collision_csv": str(out / "collision_groups.csv"),
    }
    _write_json(Path(paths["label_free_json"]), label_free_summary)
    _write_json(Path(paths["label_aware_json"]), label_aware_diagnostics)
    _write_csv(Path(paths["fallback_csv"]), fallback_rows)
    _write_csv(Path(paths["collision_csv"]), collision_rows)
    return paths


def parse_ground_truth_indices(paths: list[str]) -> set[int]:
    """Read E5 ground-truth node index IDs from CSV files."""
    indices: set[int] = set()
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"ground-truth file is missing: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                try:
                    indices.add(int(row[-1]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid ground-truth index in {path}: {row}"
                    ) from exc
    return indices


def _subject_node_table() -> str:
    table = os.getenv("CLAD_SUBJECT_NODE_TABLE", "subject_node_table").strip()
    if not table.replace("_", "").isalnum():
        raise ValueError(f"invalid CLAD_SUBJECT_NODE_TABLE: {table}")
    return table


def load_nodes(conn, semantic_mode: str, max_nodes_per_type: int) -> list[AuditNode]:
    """Load and tokenize bounded ClearScope E5 nodes from PostgreSQL."""
    subject_node_table = _subject_node_table()
    queries = [
        (
            "file",
            """
            select index_id, node_uuid, path
            from file_node_table
            order by index_id
            limit %s
            """,
        ),
        (
            "subject",
            f"""
            select index_id, node_uuid, path, cmd
            from {subject_node_table}
            order by index_id
            limit %s
            """,
        ),
        (
            "netflow",
            """
            select index_id, node_uuid, src_addr, src_port, dst_addr, dst_port
            from netflow_node_table
            order by index_id
            limit %s
            """,
        ),
    ]
    nodes: list[AuditNode] = []
    with conn.cursor() as cur:
        for node_type, sql in queries:
            cur.execute(sql, (int(max_nodes_per_type),))
            columns = [desc[0] for desc in cur.description]
            for values in cur.fetchall():
                row = dict(zip(columns, values))
                row["node_type"] = node_type
                nodes.append(tokenize_node_row(row, semantic_mode=semantic_mode))
    return nodes


def _load_typed_nodes_by_index_ids(
    conn,
    semantic_mode: str,
    index_ids: tuple[int, ...],
    node_type: str,
    sql: str,
) -> list[AuditNode]:
    nodes: list[AuditNode] = []
    with conn.cursor() as cur:
        cur.execute(sql, (index_ids,))
        columns = [desc[0] for desc in cur.description]
        for values in cur.fetchall():
            row = dict(zip(columns, values))
            row["node_type"] = node_type
            nodes.append(tokenize_node_row(row, semantic_mode=semantic_mode))
    return nodes


def load_nodes_by_index_ids(
    conn,
    semantic_mode: str,
    index_ids: Iterable[int],
) -> list[AuditNode]:
    """Load and tokenize ClearScope E5 nodes referenced by sampled events."""
    ids = tuple(sorted({int(index_id) for index_id in index_ids}))
    if not ids:
        return []
    subject_node_table = _subject_node_table()
    queries = [
        (
            "file",
            """
            select index_id, node_uuid, path
            from file_node_table
            where index_id in %s
            order by index_id
            """,
        ),
        (
            "subject",
            f"""
            select index_id, node_uuid, path, cmd
            from {subject_node_table}
            where index_id in %s
            order by index_id
            """,
        ),
        (
            "netflow",
            """
            select index_id, node_uuid, src_addr, src_port, dst_addr, dst_port
            from netflow_node_table
            where index_id in %s
            order by index_id
            """,
        ),
    ]
    nodes: list[AuditNode] = []
    for node_type, sql in queries:
        nodes.extend(
            _load_typed_nodes_by_index_ids(conn, semantic_mode, ids, node_type, sql)
        )
    return nodes


def load_events(conn, max_events_per_split: int) -> list[dict[str, object]]:
    """Load bounded E5 event rows by configured split days."""
    from scripts.pipeline.io.db_stream import _day_bounds

    year_month = "2019-05"
    split_days = {
        "train": [8, 9],
        "val": [11],
        "test": [14, 15, 17],
    }
    events: list[dict[str, object]] = []
    with conn.cursor() as cur:
        for split, days in split_days.items():
            remaining = int(max_events_per_split)
            for day in days:
                if remaining <= 0:
                    break
                start_ns, end_ns = _day_bounds(year_month, day)
                cur.execute(
                    """
                    select operation, src_index_id, dst_index_id, timestamp_rec, _id
                    from event_table
                    where timestamp_rec >= %s and timestamp_rec < %s
                    order by timestamp_rec, _id
                    limit %s
                    """,
                    (start_ns, end_ns, remaining),
                )
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                for values in rows:
                    row = dict(zip(columns, values))
                    row["split"] = split
                    events.append(row)
                remaining -= len(rows)
    return events


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the E5 semantic audit smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="clearscope_e5")
    parser.add_argument("--semantic_mode", default="raw_detail_v31_discriminative")
    parser.add_argument("--max_nodes_per_type", type=int, default=50000)
    parser.add_argument("--max_events_per_split", type=int, default=100000)
    parser.add_argument(
        "--output_dir",
        default="tmp/clearscope_e5_semantic_audit_smoke",
    )
    parser.add_argument(
        "--ground_truth",
        action="append",
        default=None,
    )
    args = parser.parse_args(argv)
    if args.ground_truth is None:
        args.ground_truth = list(DEFAULT_GROUND_TRUTH_PATHS)
    return args


def run_audit(args: argparse.Namespace) -> dict[str, str]:
    """Run the bounded ClearScope E5 semantic audit smoke."""
    with connect_db(args.database) as conn:
        nodes = load_nodes(conn, args.semantic_mode, args.max_nodes_per_type)
        events = load_events(conn, args.max_events_per_split)
        endpoint_index_ids = {
            int(event[index_key])
            for event in events
            for index_key in ("src_index_id", "dst_index_id")
        }
        endpoint_nodes = load_nodes_by_index_ids(
            conn,
            args.semantic_mode,
            endpoint_index_ids,
        )
    nodes_by_index = {node.index_id: node for node in endpoint_nodes}
    nodes_by_index.update({node.index_id: node for node in nodes})
    label_free_summary = build_label_free_summary(nodes)
    sampled_node_ids = {node.index_id for node in nodes}
    label_free_summary["endpoint_node_count"] = len(
        {node.index_id for node in endpoint_nodes if node.index_id not in sampled_node_ids}
    )
    label_free_summary["tuple_summary_node_count"] = len(nodes_by_index)
    label_free_summary["event_tuple_summary"] = build_event_tuple_summary(
        events,
        nodes_by_index,
    )
    malicious_index_ids = parse_ground_truth_indices(list(args.ground_truth))
    label_aware_diagnostics = build_label_aware_diagnostics(nodes, malicious_index_ids)
    return write_reports(
        output_dir=args.output_dir,
        label_free_summary=label_free_summary,
        label_aware_diagnostics=label_aware_diagnostics,
        fallback_rows=fallback_rows_from_summary(label_free_summary),
        collision_rows=build_collision_rows(nodes),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    paths = run_audit(args)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
