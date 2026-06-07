#!/usr/bin/env python3
"""Smoke-check raw detail residual tokens on train rows only."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.config.provnet_utils import init_database_connection
from scripts.data.get_dataset import parse_split_days, use_event_type_filter
from scripts.tools.causal_semantics_slim import (
    SlimConfig,
    _residual_tokens_for_row,
    build_node_maps,
    residual_text,
    stream_dataset_rows,
)
from scripts.tools.db_stream_utils import _cfg_for_dataset
from cs4m.semantics.semantic_router import load_process_semantic_config


CHECK_PATHS = {
    "THEIA_E3": {
        "/tmp/tmux-1002": "tmux_1002",
        "/proc/20432/oom_adj": "oom_adj",
        "/bin/sh -c ./gtcache &>/dev/null &": "gtcache",
    },
    "CADETS_E3": {
        "/var/log/salt/minion": "salt_minion",
        "/usr/home/user/eraseme/index.html": "eraseme_index_html",
        "/tmp/tmux-1002": "tmux_1002",
    },
}


def _split_days(db_cfg: Any) -> list[int]:
    return parse_split_days(db_cfg.dataset.train_splits)


def _row_contains(row: dict[str, Any], needle: str) -> bool:
    text = " ".join(str(value) for value in row.values())
    return needle in text


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "event_index",
        "action",
        "src_kind",
        "dst_kind",
        "src_process_cmd",
        "src_process_path",
        "dst_file_path",
        "dst_addr",
        "src_addr",
        "residual_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def smoke_dataset(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    """Run one label-free train-split token smoke check."""
    dataset_name = str(dataset).upper()
    config = SlimConfig(
        dataset=dataset_name,
        semantic_embedding_method="word2vec",
        latent_dim=64,
        max_train_events=int(args.max_train_events),
        fetch_size=int(args.fetch_size),
        theia_netflow_policy=str(args.theia_netflow_policy),
    )
    process_cfg = load_process_semantic_config(str(args.process_semantics_config))
    db_cfg = _cfg_for_dataset(dataset_name)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    train_days = _split_days(db_cfg)
    cur, conn = init_database_connection(db_cfg)
    vocab: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {
        needle: {"expected_token": expected, "seen": False, "matched": False, "example": ""}
        for needle, expected in CHECK_PATHS.get(dataset_name, {}).items()
    }
    count = 0
    try:
        node_maps = build_node_maps(cur)
        rows = stream_dataset_rows(
            conn,
            str(db_cfg.dataset.year_month),
            train_days,
            node_maps,
            use_event_type_filter(db_cfg),
            int(args.fetch_size),
            int(args.max_train_events),
            abnormal_nodes=set(),
        )
        for row in rows:
            tokens = _residual_tokens_for_row(row, config, process_cfg)
            vocab.update(tokens)
            text = residual_text(
                row,
                dataset=dataset_name,
                process_semantic_config=process_cfg,
                max_tokens_per_node=config.max_tokens_per_node,
                theia_netflow_policy=config.theia_netflow_policy,
            )
            if len(samples) < int(args.sample_rows):
                samples.append(
                    {
                        "dataset": dataset_name,
                        "event_index": row.get("event_index", ""),
                        "action": row.get("action", ""),
                        "src_kind": row.get("src_kind", ""),
                        "dst_kind": row.get("dst_kind", ""),
                        "src_process_cmd": row.get("src_process_cmd", ""),
                        "src_process_path": row.get("src_process_path", ""),
                        "dst_file_path": row.get("dst_file_path", ""),
                        "dst_addr": row.get("dst_addr", ""),
                        "src_addr": row.get("src_addr", ""),
                        "residual_text": text,
                    },
                )
            for needle, check in checks.items():
                if check["seen"]:
                    continue
                if _row_contains(dict(row), needle):
                    check["seen"] = True
                    check["matched"] = str(check["expected_token"]) in tokens
                    check["example"] = text
            count += 1
    finally:
        cur.close()
        conn.close()

    payload = {
        "dataset": dataset_name,
        "train_days": train_days,
        "sample_rows": count,
        "sample_train_vocab_size": len(vocab),
        "top_tokens": [{"token": token, "count": freq} for token, freq in vocab.most_common(50)],
        "checks": checks,
        "samples": samples,
        "leakage_contract": "train_split_only_no_labels_no_gt_no_test_rows",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{dataset_name.lower()}_raw_detail_smoke.json"
    csv_path = out_dir / f"{dataset_name.lower()}_raw_detail_samples.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(csv_path, samples)
    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "sample_rows": count,
                "sample_train_vocab_size": len(vocab),
                "json_path": str(json_path),
                "csv_path": str(csv_path),
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return payload


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--max_train_events", type=int, default=200000)
    parser.add_argument("--fetch_size", type=int, default=100000)
    parser.add_argument("--sample_rows", type=int, default=200)
    parser.add_argument("--out_dir", default="outputs/diagnostics/raw_detail_token_smoke")
    parser.add_argument("--process_semantics_config", default="configs/common/process_semantics.yaml")
    parser.add_argument("--theia_netflow_policy", default="fixed")
    return parser.parse_args()


def main() -> None:
    """Run smoke checks for all requested datasets."""
    args = parse_args()
    summaries = [smoke_dataset(args, dataset) for dataset in args.datasets]
    print(
        json.dumps(
            [
                {
                    "dataset": item["dataset"],
                    "sample_rows": item["sample_rows"],
                    "sample_train_vocab_size": item["sample_train_vocab_size"],
                }
                for item in summaries
            ],
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
