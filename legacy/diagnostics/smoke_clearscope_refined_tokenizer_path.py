#!/usr/bin/env python3
"""Smoke-check that ClearScope residual tokens use raw_detail_v2_refined."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.config.provnet_utils import init_database_connection
from scripts.data.get_dataset import parse_split_days, use_event_type_filter
from legacy.compatibility.pipeline_runtime_exports import (
    SlimConfig,
    _residual_tokens_for_row,
    build_node_maps,
    residual_text,
    stream_dataset_rows,
)
from scripts.pipeline.io.db_stream import _cfg_for_dataset
from cs4m.semantics.clearscope_android import CLEARSCOPE_REFINED_SEMANTIC_MODE
from cs4m.semantics.semantic_router import load_process_semantic_config


REQUIRED_REFINED_TOKENS = {
    "dev_ion",
    "dev_ashmem",
    "socket_logdw",
    "socket_lmkd",
    "system_app_camera2_apk",
    "data_app_vmdl_base_apk",
    "fennec_firefox_dev_doomed_num",
    "fennec_firefox_dev_entries_hexblob",
    "net_xt_qtaguid_ctrl",
    "email_body_txt",
    "email_body_html",
    "settings_cryptkeeper",
}


def _split_days(db_cfg: Any) -> list[int]:
    return parse_split_days(db_cfg.dataset.train_splits)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _node_tokens_for_side(tokens: list[str], side: str) -> str:
    try:
        event_index = tokens.index("event")
    except ValueError:
        event_index = len(tokens)
    if side == "src":
        return " ".join(tokens[:event_index])
    action_end = min(event_index + 2, len(tokens))
    return " ".join(tokens[action_end:])


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run the train-split smoke audit without labels or test rows."""
    dataset = "CLEARSCOPE_E3"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = SlimConfig(
        dataset=dataset,
        semantic_mode=CLEARSCOPE_REFINED_SEMANTIC_MODE,
        semantic_embedding_method="word2vec",
        latent_dim=64,
        max_train_events=int(args.max_train_events),
        fetch_size=int(args.fetch_size),
    )
    process_cfg = load_process_semantic_config(str(args.process_semantics_config))
    db_cfg = _cfg_for_dataset(dataset)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    train_days = _split_days(db_cfg)

    token_frequency: Counter[str] = Counter()
    tuple_frequency: Counter[tuple[str, ...]] = Counter()
    examples: list[dict[str, Any]] = []
    required_seen: dict[str, bool] = {token: False for token in REQUIRED_REFINED_TOKENS}
    count = 0
    cur, conn = init_database_connection(db_cfg)
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
            token_frequency.update(tokens)
            tuple_frequency.update([tuple(tokens)])
            for token in required_seen:
                if token in tokens:
                    required_seen[token] = True
            has_required_token = any(token in tokens for token in required_seen)
            if len(examples) < int(args.example_rows) or has_required_token:
                text = residual_text(
                    row,
                    dataset=dataset,
                    process_semantic_config=process_cfg,
                    max_tokens_per_node=config.max_tokens_per_node,
                    theia_netflow_policy=config.theia_netflow_policy,
                    semantic_mode=config.semantic_mode,
                )
                examples.append(
                    {
                        "event_id": row.get("event_id", row.get("event_index", "")),
                        "src_node_tokens": _node_tokens_for_side(tokens, "src"),
                        "action_tokens": (
                            " ".join(tokens[tokens.index("event") : tokens.index("event") + 2])
                            if "event" in tokens
                            else ""
                        ),
                        "dst_node_tokens": _node_tokens_for_side(tokens, "dst"),
                        "residual_text": text,
                        "raw_src": row.get("src_process_cmd", row.get("src_summary", "")),
                        "raw_dst": row.get(
                            "dst_file_path",
                            row.get(
                                "dst_process_cmd",
                                row.get("dst_addr", row.get("dst_summary", "")),
                            ),
                        ),
                    },
                )
            count += 1
    finally:
        cur.close()
        conn.close()

    _write_csv(
        out_dir / "residual_sentence_examples.csv",
        examples,
        [
            "event_id",
            "src_node_tokens",
            "action_tokens",
            "dst_node_tokens",
            "residual_text",
            "raw_src",
            "raw_dst",
        ],
    )
    _write_csv(
        out_dir / "refined_token_frequency_smoke.csv",
        [
            {"token": token, "count": count}
            for token, count in token_frequency.most_common()
        ],
        ["token", "count"],
    )
    _write_csv(
        out_dir / "refined_token_tuple_frequency_smoke.csv",
        [
            {"token_tuple": " ".join(tokens), "count": count}
            for tokens, count in tuple_frequency.most_common()
        ],
        ["token_tuple", "count"],
    )
    contains_required = any(required_seen.values())
    summary = {
        "semantic_mode": CLEARSCOPE_REFINED_SEMANTIC_MODE,
        "legacy_path_used": False,
        "contains_required_refined_tokens": bool(contains_required),
        "required_refined_tokens_seen": required_seen,
        "unique_tokens_smoke": int(len(token_frequency)),
        "unique_token_tuples_smoke": int(len(tuple_frequency)),
        "examples_checked": int(count),
        "train_splits": [f"day_{int(day)}" for day in train_days],
        "event_filter": "Orthrus10" if use_event_type_filter(db_cfg) else "none",
        "leakage_contract": "train_split_only_no_labels_no_ground_truth_no_test_rows",
    }
    summary_path = out_dir / "refined_path_smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_tmp_root = Path(os.environ.get("TMP_ROOT", "tmp"))
    parser.add_argument(
        "--out_dir",
        default=str(default_tmp_root / "clearscope_refined_tokenizer_path_smoke"),
    )
    parser.add_argument("--max_train_events", type=int, default=10000)
    parser.add_argument("--fetch_size", type=int, default=10000)
    parser.add_argument("--example_rows", type=int, default=200)
    parser.add_argument("--process_semantics_config", default="configs/common/process_semantics.yaml")
    return parser.parse_args()


def main() -> None:
    """Run the smoke audit."""
    run_smoke(parse_args())


if __name__ == "__main__":
    main()
