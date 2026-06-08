#!/usr/bin/env python3
"""Smoke-check ClearScope Phase3G node tokens against refined Word2Vec metadata."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.config.provnet_utils import init_database_connection
from scripts.data.get_dataset import parse_split_days, use_event_type_filter
from legacy.compatibility.pipeline_runtime_exports import (
    SlimConfig,
    _effective_latent_dim,
    _phase3e_detect_event_id_column,
    _phase3e_fetch_used_node_lookup_batched,
    _phase3e_node_tokens_from_db_node,
    _phase3e_scan_used_nodes_pass1,
    load_pretrained_residual_embedder,
    load_word2vec_semantic_metadata,
    validate_word2vec_semantic_mode_matches_config,
)
from scripts.pipeline.io.db_stream import _cfg_for_dataset
from cs4m.semantics.clearscope_android import (
    CLEARSCOPE_REFINED_SEMANTIC_MODE,
    normalize_clearscope_semantic_mode,
)
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


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _split_days(db_cfg: Any) -> list[int]:
    return parse_split_days(db_cfg.dataset.train_splits)


def _train_vocab_from_word2vec(path: Path) -> set[str]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, Mapping):
        state = payload.get("model_state", payload)
        if isinstance(state, Mapping):
            return {str(token) for token in state.get("train_vocab", [])}
    return set()


def _metadata_semantic_mode(metadata: Mapping[str, Any]) -> str:
    semantic_mode = str(metadata.get("semantic_mode", "")).strip()
    semantic_rules = metadata.get("semantic_rules", {})
    if not semantic_mode and isinstance(semantic_rules, Mapping):
        semantic_mode = str(semantic_rules.get("clearscope", "")).strip()
    return normalize_clearscope_semantic_mode(semantic_mode)


def _node_maps_for_phase3g_smoke(
    indexid2summary: Mapping[int, tuple[str, str]],
    meta_by_kind: Mapping[str, Mapping[int, Mapping[str, str]]],
) -> dict[str, Any]:
    return {
        "indexid2summary": indexid2summary,
        "process_meta": dict(meta_by_kind.get("process_meta", {})),
        "file_meta": dict(meta_by_kind.get("file_meta", {})),
        "netflow_meta": dict(meta_by_kind.get("netflow_meta", {})),
    }


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run train-split Phase3G node-token smoke without labels or test rows."""
    dataset = "CLEARSCOPE_E3"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    word2vec_path = Path(args.word2vec_path)
    config = SlimConfig(
        dataset=dataset,
        semantic_mode=CLEARSCOPE_REFINED_SEMANTIC_MODE,
        pretrained_residual_embedder_path=str(word2vec_path),
        latent_dim=int(args.latent_dim),
        node_embedding_dim=int(args.latent_dim),
        sspm_target_dim=int(args.latent_dim),
        max_train_events=int(args.max_train_events),
        max_ref_events=0,
        max_test_events=0,
        fetch_size=int(args.fetch_size),
        phase3e_node_lookup_batch_size=int(args.node_lookup_batch_size),
        process_semantics_config=str(args.process_semantics_config),
        progress_interval_events=int(args.progress_interval_events),
    )
    validate_word2vec_semantic_mode_matches_config(config)
    process_cfg = load_process_semantic_config(config.process_semantics_config)
    load_pretrained_residual_embedder(
        str(word2vec_path),
        expected_dim=_effective_latent_dim(config, process_cfg),
    )
    metadata = load_word2vec_semantic_metadata(word2vec_path)
    word2vec_semantic_mode = _metadata_semantic_mode(metadata)
    word2vec_vocab = _train_vocab_from_word2vec(word2vec_path)

    db_cfg = _cfg_for_dataset(dataset)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    train_days = _split_days(db_cfg)
    cur, conn = init_database_connection(db_cfg)
    try:
        cur.close()
        event_id_column = _phase3e_detect_event_id_column(conn)
        pass1 = _phase3e_scan_used_nodes_pass1(
            conn=conn,
            year_month=str(db_cfg.dataset.year_month),
            split_days={"train": train_days, "validation": [], "test": []},
            event_filter=use_event_type_filter(db_cfg),
            fetch_size=int(config.fetch_size),
            max_events_by_split={
                "train": int(config.max_train_events),
                "validation": 0,
                "test": 0,
            },
            event_id_column=event_id_column,
            config=config,
        )
        used_node_ids = set(int(node_id) for node_id in pass1["used_node_ids"])
        indexid2summary, meta_by_kind, node_lookup_seconds = (
            _phase3e_fetch_used_node_lookup_batched(
                conn=conn,
                used_node_ids=used_node_ids,
                batch_size=int(config.phase3e_node_lookup_batch_size),
            )
        )
    finally:
        conn.close()

    node_maps = _node_maps_for_phase3g_smoke(indexid2summary, meta_by_kind)
    token_frequency: Counter[str] = Counter()
    required_seen = {token: False for token in REQUIRED_REFINED_TOKENS}
    total_tokens = 0
    oov_tokens = 0
    examples: list[dict[str, Any]] = []

    for node_id in sorted(used_node_ids):
        if node_id not in indexid2summary:
            continue
        node_kind, node_summary = indexid2summary[node_id]
        tokens = _phase3e_node_tokens_from_db_node(
            node_id=int(node_id),
            node_kind=str(node_kind),
            node_summary=str(node_summary),
            node_maps=node_maps,
            config=config,
            process_cfg=process_cfg,
        )
        token_frequency.update(tokens)
        token_oov = [token for token in tokens if token not in word2vec_vocab]
        total_tokens += len(tokens)
        oov_tokens += len(token_oov)
        required_present = [token for token in tokens if token in required_seen]
        for token in required_present:
            required_seen[token] = True
        if len(examples) < int(args.example_rows) or required_present:
            examples.append(
                {
                    "node_id": int(node_id),
                    "node_kind": str(node_kind),
                    "node_tokens": " ".join(tokens),
                    "raw_summary": str(node_summary),
                    "required_tokens": " ".join(required_present),
                    "oov_tokens": " ".join(token_oov),
                },
            )

    _write_csv(
        out_dir / "phase3g_node_token_examples.csv",
        examples,
        [
            "node_id",
            "node_kind",
            "node_tokens",
            "raw_summary",
            "required_tokens",
            "oov_tokens",
        ],
    )
    oov_summary = {
        "total_tokens": int(total_tokens),
        "oov_tokens": int(oov_tokens),
        "oov_rate": float(oov_tokens / total_tokens) if total_tokens else 0.0,
        "unique_node_tokens": int(len(token_frequency)),
        "word2vec_vocab_size": int(len(word2vec_vocab)),
    }
    (out_dir / "phase3g_oov_summary.json").write_text(
        json.dumps(oov_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    phase3g_semantic_mode = normalize_clearscope_semantic_mode(config.semantic_mode)
    summary = {
        "phase3g_semantic_mode": phase3g_semantic_mode,
        "word2vec_semantic_mode": word2vec_semantic_mode,
        "semantic_mode_match": bool(phase3g_semantic_mode == word2vec_semantic_mode),
        "legacy_path_used": False,
        "required_refined_tokens_observed": bool(any(required_seen.values())),
        "required_refined_tokens_seen": required_seen,
        "oov_rate": oov_summary["oov_rate"],
        "oov_summary_path": str(out_dir / "phase3g_oov_summary.json"),
        "node_examples_path": str(out_dir / "phase3g_node_token_examples.csv"),
        "word2vec_path": str(word2vec_path),
        "used_node_count": int(len(used_node_ids)),
        "node_lookup_seconds": float(node_lookup_seconds),
        "examples_checked": int(pass1["split_counts"].get("train", 0)),
        "train_splits": [f"day_{int(day)}" for day in train_days],
        "event_filter": "Orthrus10" if use_event_type_filter(db_cfg) else "none",
        "leakage_contract": "train_split_only_no_labels_no_ground_truth_no_test_rows",
    }
    summary_path = out_dir / "phase3g_refined_token_path_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word2vec_path", required=True)
    default_tmp_root = Path(os.environ.get("TMP_ROOT", "tmp"))
    parser.add_argument(
        "--out_dir",
        default=str(default_tmp_root / "clearscope_phase3g_refined_token_path_smoke"),
    )
    parser.add_argument("--max_train_events", type=int, default=10000)
    parser.add_argument("--fetch_size", type=int, default=10000)
    parser.add_argument("--node_lookup_batch_size", type=int, default=50000)
    parser.add_argument("--progress_interval_events", type=int, default=100000)
    parser.add_argument("--example_rows", type=int, default=200)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--process_semantics_config", default="configs/common/process_semantics.yaml")
    return parser.parse_args()


def main() -> None:
    """Run the smoke audit."""
    run_smoke(parse_args())


if __name__ == "__main__":
    main()
