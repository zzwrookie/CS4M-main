#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psycopg2
except ModuleNotFoundError:  # pragma: no cover - dependency checked at runtime.
    psycopg2 = None

from cs4m.config.config import DATASET_DEFAULT_CONFIG
from legacy.compatibility.pipeline_runtime_exports import (
    SLIM_SPLIT_OVERRIDES,
    SlimConfig,
    build_node_maps,
    load_ground_truth_indices_readonly,
    _malicious_pair_audit_row,
    _cfg_for_dataset,
    residual_text,
    row_fields,
    stream_dataset_rows,
    stream_dataset_rows_slim,
)
from cs4m.utils.common import stable_hash
from cs4m.semantics.semantic_router import (
    ProcessSemanticConfig,
    load_process_semantic_config,
)
from legacy.experiments.residual_hash_doc2vec import SemanticSketchSlim


TRACE_FIELDS = (
    "split",
    "dataset",
    "event_index",
    "timestamp_ns",
    "src_idx",
    "dst_idx",
    "src_kind",
    "dst_kind",
    "action",
    "object_type",
    "src_summary",
    "dst_summary",
    "src_process_cmd",
    "dst_process_cmd",
    "src_file_path",
    "dst_file_path",
    "dst_addr",
    "dst_port",
    "src_role",
    "dst_role",
    "relation",
    "info_src",
    "info_dst",
    "relation_id",
    "info_src_type",
    "info_dst_type",
    "residual_text",
    "hash_tokens",
    "token_count",
    "token_projection",
    "nonzero_vector",
    "latent_dim",
    "max_tokens",
    "trace_group",
)

BALANCED_TRACE_GROUPS = ("process_file", "process_process", "process_netflow")


def build_trace_row(
    row: Mapping[str, Any],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
    sketch: SemanticSketchSlim,
    split: str = "",
    theia_netflow_policy: str = "fixed",
) -> dict[str, Any]:
    """Return a label-free trace row for the exact residual hash sketch input."""
    fields = row_fields(
        row,
        config.max_tokens_per_node,
        dataset=config.dataset,
        process_semantic_config=process_cfg,
        theia_netflow_policy=theia_netflow_policy,
    )
    text = residual_text(
        row,
        dataset=config.dataset,
        process_semantic_config=process_cfg,
        max_tokens_per_node=config.max_tokens_per_node,
        theia_netflow_policy=theia_netflow_policy,
    )
    tokens = list(sketch.text_only_tokens(text))
    projection, vector = project_tokens(tokens, sketch.latent_dim)
    output = {
        "split": split,
        "dataset": config.dataset,
        "event_index": int(row.get("event_index", -1)),
        "timestamp_ns": int(row.get("timestamp_ns", 0)),
        "src_idx": int(row.get("src_idx", -1)),
        "dst_idx": int(row.get("dst_idx", -1)),
        "src_kind": str(row.get("src_kind", "")),
        "dst_kind": str(row.get("dst_kind", "")),
        "action": str(row.get("action", "")),
        "object_type": str(row.get("object_type", "")),
        "src_summary": str(row.get("src_summary", "")),
        "dst_summary": str(row.get("dst_summary", row.get("text", ""))),
        "src_process_cmd": str(row.get("src_process_cmd", "")),
        "dst_process_cmd": str(row.get("dst_process_cmd", "")),
        "src_file_path": str(row.get("src_file_path", "")),
        "dst_file_path": str(row.get("dst_file_path", "")),
        "dst_addr": str(row.get("dst_addr", "")),
        "dst_port": str(row.get("dst_port", "")),
        "src_role": fields["src_role"],
        "dst_role": fields["dst_role"],
        "relation": fields["relation"],
        "info_src": int(fields["info_src"]),
        "info_dst": int(fields["info_dst"]),
        "relation_id": int(fields["relation_id"]),
        "info_src_type": int(fields["info_src_type"]),
        "info_dst_type": int(fields["info_dst_type"]),
        "residual_text": text,
        "hash_tokens": " ".join(tokens),
        "token_count": len(tokens),
        "token_projection": " ".join(projection),
        "nonzero_vector": format_nonzero_vector(vector),
        "latent_dim": int(sketch.latent_dim),
        "max_tokens": int(sketch.max_tokens),
    }
    return {field: output.get(field, "") for field in TRACE_FIELDS}


def project_tokens(tokens: Sequence[str], latent_dim: int) -> tuple[list[str], np.ndarray]:
    """Project tokens into the same signed hash buckets used by SemanticSketchSlim."""
    dim = int(latent_dim)
    if dim <= 0:
        raise ValueError("latent_dim must be positive")
    vector = np.zeros((dim,), dtype=np.float32)
    projection: list[str] = []
    for token in tokens:
        bucket = stable_hash(token, seed=0) % dim
        sign = 1.0 if (stable_hash(token, seed=17) & 1) == 0 else -1.0
        vector[bucket] += np.float32(sign)
        sign_text = "+1" if sign > 0 else "-1"
        projection.append(f"{token}=>{bucket}:{sign_text}")
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= np.float32(norm)
    return projection, vector


def format_nonzero_vector(vector: np.ndarray) -> str:
    """Return sparse normalized vector entries as bucket:value text."""
    entries = []
    for index, value in enumerate(np.asarray(vector, dtype=np.float32)):
        if float(value) != 0.0:
            entries.append(f"{index}:{float(value):.6g}")
    return " ".join(entries)


def trace_group(row: Mapping[str, Any]) -> str:
    """Return the balanced trace interaction group for surface event kinds."""
    src_kind = str(row.get("src_kind", "")).lower()
    dst_kind = str(row.get("dst_kind", "")).lower()
    kinds = {src_kind, dst_kind}
    if "process" not in kinds:
        return ""
    if "file" in kinds:
        return "process_file"
    if src_kind == "process" and dst_kind == "process":
        return "process_process"
    if "netflow" in kinds:
        return "process_netflow"
    return ""


def balanced_trace_rows(
    rows: Iterable[Mapping[str, Any]],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
    sketch: SemanticSketchSlim,
    split: str,
    per_group: int,
    theia_netflow_policy: str = "fixed",
) -> Iterable[Mapping[str, Any]]:
    """Yield label-free trace rows with at most per_group rows for each interaction group."""
    limit = max(int(per_group), 0)
    counts = {group: 0 for group in BALANCED_TRACE_GROUPS}
    if limit == 0:
        return
    for row in rows:
        group = trace_group(row)
        if not group or counts[group] >= limit:
            continue
        output = build_trace_row(
            row,
            config,
            process_cfg,
            sketch,
            split=split,
            theia_netflow_policy=theia_netflow_policy,
        )
        output["trace_group"] = group
        counts[group] += 1
        yield output
        if all(count >= limit for count in counts.values()):
            return


def collect_balanced_trace_rows(
    rows: Iterable[Mapping[str, Any]],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
    sketch: SemanticSketchSlim,
    split: str,
    per_group: int,
    max_scan_events: int,
    theia_netflow_policy: str = "fixed",
) -> list[Mapping[str, Any]]:
    """Materialize balanced trace rows with a bounded scan and deterministic order."""
    limit = max(int(per_group), 0)
    scan_limit = int(max_scan_events)
    output: list[Mapping[str, Any]] = []
    counts = {group: 0 for group in BALANCED_TRACE_GROUPS}
    if limit == 0:
        return output
    for scan_index, row in enumerate(rows):
        if scan_limit > 0 and scan_index >= scan_limit:
            break
        group = trace_group(row)
        if not group or counts[group] >= limit:
            continue
        trace_row = build_trace_row(
            row,
            config,
            process_cfg,
            sketch,
            split=split,
            theia_netflow_policy=theia_netflow_policy,
        )
        trace_row["trace_group"] = group
        output.append(trace_row)
        counts[group] += 1
        if all(count >= limit for count in counts.values()):
            break
    return output


def write_trace_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write trace rows with a stable label-free header."""
    write_trace_csv_with_fields(path, rows, TRACE_FIELDS)


def write_trace_csv_with_fields(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Write trace rows with a stable caller-selected header."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def export_trace(args: argparse.Namespace) -> int:
    """Export residual semantic trace rows from a DB stream."""
    if psycopg2 is None:
        raise ModuleNotFoundError("psycopg2 is required for DB trace export")
    dataset = str(args.dataset)
    dataset_cfg = DATASET_DEFAULT_CONFIG.get(dataset)
    if dataset_cfg is None:
        raise ValueError(f"unknown dataset: {dataset}")
    password = os.getenv("CLAD_DB_PASSWORD", "")
    if not password:
        raise ValueError("CLAD_DB_PASSWORD must be set in the environment")

    days = resolve_days(dataset, str(args.split), args.days, bool(args.slim_split_override))
    year_month = resolve_year_month(dataset, bool(args.slim_split_override), dataset_cfg)
    config = SlimConfig(
        dataset=dataset,
        max_tokens_per_node=int(args.max_tokens_per_node),
        latent_dim=int(args.latent_dim),
        fetch_size=int(args.fetch_size),
        db_stream_mode=str(args.db_stream_mode),
        process_semantics_config=str(args.process_semantics_config),
        theia_netflow_policy=str(args.theia_netflow_policy),
    )
    process_cfg = load_process_semantic_config(args.process_semantics_config)
    sketch = SemanticSketchSlim(latent_dim=args.latent_dim, max_tokens=args.max_tokens)

    conn = psycopg2.connect(
        database=str(dataset_cfg["db_name"]),
        host=os.getenv("CLAD_DB_HOST", "localhost"),
        user=os.getenv("CLAD_DB_USER", "postgres"),
        password=password,
        port=os.getenv("CLAD_DB_PORT", "5433"),
    )
    try:
        cur = conn.cursor()
        node_maps = build_node_maps(cur)
        stream_status: dict[str, str] = {}
        if bool(args.malicious_pair_only):
            db_cfg = _cfg_for_dataset(dataset)
            abnormal_nodes = load_ground_truth_indices_readonly(db_cfg, node_maps["uuid2index"])
            rows = stream_dataset_rows(
                conn,
                year_month,
                days,
                node_maps,
                bool(args.event_filter),
                int(args.fetch_size),
                int(args.limit),
                abnormal_nodes=abnormal_nodes,
            )
            output_rows = (
                row
                for row in (
                    _malicious_pair_audit_row(
                        source_row,
                        abnormal_nodes,
                        config,
                        process_cfg,
                    )
                    for source_row in rows
                )
                if row is not None
            )
        else:
            stream_limit = int(args.limit)
            if bool(args.balanced_groups):
                stream_limit = 0
            rows = stream_dataset_rows_slim(
                conn,
                year_month,
                days,
                node_maps,
                bool(args.event_filter),
                int(args.fetch_size),
                stream_limit,
                str(args.db_stream_mode),
                abnormal_nodes=None,
                stream_status=stream_status,
            )
            if bool(args.balanced_groups):
                per_group = int(args.balanced_groups_per_group)
                output_rows = collect_balanced_trace_rows(
                    rows,
                    config,
                    process_cfg,
                    sketch,
                    str(args.split),
                    per_group=per_group,
                    max_scan_events=int(args.balanced_groups_max_scan_events),
                    theia_netflow_policy=str(args.theia_netflow_policy),
                )
            else:
                output_rows = (
                    build_trace_row(
                        row,
                        config,
                        process_cfg,
                        sketch,
                        split=str(args.split),
                        theia_netflow_policy=str(args.theia_netflow_policy),
                    )
                    for row in rows
                )
        if bool(args.malicious_pair_only):
            from legacy.compatibility.pipeline_runtime_exports import MALICIOUS_PAIR_AUDIT_FIELDS

            write_trace_csv_with_fields(args.output, output_rows, MALICIOUS_PAIR_AUDIT_FIELDS)
        else:
            write_trace_csv(args.output, output_rows)
    finally:
        conn.close()
    print(f"wrote_trace_csv={args.output}")
    print(f"dataset={dataset} split={args.split} days={','.join(str(day) for day in days)}")
    return 0


def resolve_days(
    dataset: str,
    split: str,
    days_arg: str,
    slim_split_override: bool,
) -> list[int]:
    """Resolve trace days from explicit CLI days, slim override, or dataset config."""
    if str(days_arg).strip():
        return [int(part) for part in str(days_arg).replace(" ", "").split(",") if part]
    split_key = str(split).strip().lower()
    if slim_split_override and dataset in SLIM_SPLIT_OVERRIDES:
        override = SLIM_SPLIT_OVERRIDES[dataset]
        if split_key == "validation":
            split_key = "val"
        return [int(day) for day in override[split_key]]
    dataset_cfg = DATASET_DEFAULT_CONFIG[dataset]
    config_key = {
        "train": "train_splits",
        "validation": "val_splits",
        "val": "val_splits",
        "test": "test_splits",
    }[split_key]
    return [int(str(value).replace("day_", "")) for value in dataset_cfg[config_key]]


def resolve_year_month(
    dataset: str,
    slim_split_override: bool,
    dataset_cfg: Mapping[str, Any],
) -> str:
    """Return the year-month used for DB day filters."""
    if slim_split_override and dataset in SLIM_SPLIT_OVERRIDES:
        return str(SLIM_SPLIT_OVERRIDES[dataset]["year_month"])
    return str(dataset_cfg["year_month"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export label-free residual semantic hash tokens for manual inspection.",
    )
    parser.add_argument("--dataset", default="CLEARSCOPE_E3")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--days", default="", help="Comma-separated day numbers; overrides split.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--fetch_size", type=int, default=10000)
    parser.add_argument(
        "--db_stream_mode",
        choices=("auto", "temp_table", "python_lookup"),
        default="auto",
    )
    parser.add_argument("--event_filter", action="store_true")
    parser.add_argument(
        "--balanced_groups",
        action="store_true",
        help="Export a label-free balanced sample of process-file/process-process/process-netflow.",
    )
    parser.add_argument(
        "--balanced_groups_per_group",
        type=int,
        default=5,
        help="Rows per interaction group when --balanced_groups is set.",
    )
    parser.add_argument(
        "--balanced_groups_max_scan_events",
        type=int,
        default=2000000,
        help="Maximum stream events to scan in --balanced_groups mode; 0 means no cap.",
    )
    parser.add_argument(
        "--malicious_pair_only",
        action="store_true",
        help="Post-stream audit: export only events whose source and destination are GT nodes.",
    )
    parser.add_argument("--slim_split_override", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=48)
    parser.add_argument("--max_tokens_per_node", type=int, default=8)
    parser.add_argument("--process_semantics_config", default="configs/common/process_semantics.yaml")
    parser.add_argument(
        "--theia_netflow_policy",
        choices=("scope_port", "fixed"),
        default=SlimConfig.theia_netflow_policy,
    )
    parser.add_argument(
        "--output",
        default="outputs/diagnostics/residual_semantic_trace.csv",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return export_trace(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
