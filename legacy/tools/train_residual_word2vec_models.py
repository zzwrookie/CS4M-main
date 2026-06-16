#!/usr/bin/env python3
"""Train and save train-only residual Word2Vec models.

This utility intentionally stops after fitting the residual embedder on the
training split. It does not calibrate thresholds, score validation/test rows, or
read ground-truth labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.config.provnet_utils import init_database_connection
from scripts.data.get_dataset import parse_split_days, use_event_type_filter
from legacy.compatibility.pipeline_runtime_exports import (
    NETWORK_SEMANTIC_RULES_VERSION,
    CADETS_SEMANTIC_RULES_VERSION,
    THEIA_SEMANTIC_RULES_VERSION,
    SlimConfig,
    _make_residual_embedding_config,
    _residual_tokens_for_row,
    build_node_maps,
    stream_dataset_rows,
)
from scripts.pipeline.io.db_stream import _cfg_for_dataset
from cs4m.semantics.clearscope_android import (
    CLEARSCOPE_LEGACY_SEMANTIC_MODE,
    CLEARSCOPE_V3_SEMANTIC_MODE,
    is_clearscope_dataset,
    normalize_clearscope_semantic_mode,
)
from cs4m.semantics.cadets_freebsd import is_cadets_dataset
from cs4m.semantics.optc_windows import is_optc_dataset
from cs4m.semantics.semantic_router import load_process_semantic_config
from cs4m.embeddings.residual import build_residual_embedder


def _split_days(db_cfg: Any) -> list[int]:
    return parse_split_days(db_cfg.dataset.train_splits)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _word2vec_metadata_semantic_payload(
    args: argparse.Namespace,
    dataset: str,
) -> dict[str, Any]:
    """Return semantic-mode metadata for a residual Word2Vec training run."""
    raw_mode = str(getattr(args, "semantic_mode", CLEARSCOPE_V3_SEMANTIC_MODE))
    if is_clearscope_dataset(dataset):
        semantic_mode = normalize_clearscope_semantic_mode(raw_mode)
        legacy_path_used = semantic_mode == CLEARSCOPE_LEGACY_SEMANTIC_MODE
        clearscope_rule = semantic_mode
        cadets_rule = CADETS_SEMANTIC_RULES_VERSION
    elif is_cadets_dataset(dataset):
        semantic_mode = raw_mode
        legacy_path_used = False
        clearscope_rule = ""
        cadets_rule = semantic_mode
    else:
        semantic_mode = raw_mode
        legacy_path_used = False
        clearscope_rule = ""
        cadets_rule = CADETS_SEMANTIC_RULES_VERSION
    return {
        "semantic_mode": semantic_mode,
        "semantic_rules": {
            "network": NETWORK_SEMANTIC_RULES_VERSION,
            "theia": THEIA_SEMANTIC_RULES_VERSION,
            "cadets": cadets_rule,
            "clearscope": clearscope_rule,
        },
        "legacy_path_used": bool(legacy_path_used),
    }


def _stream_train_sentences(
    rows: Any,
    config: SlimConfig,
    process_cfg: Any,
    dataset_name: str,
    progress_interval: int,
    stats: dict[str, Any],
) -> Iterator[list[str]]:
    vocab: set[str] = set()
    frequencies: dict[str, int] = {}
    print(f"[WORD2VEC][{dataset_name}] train_read_start", file=sys.stderr, flush=True)
    for row in rows:
        tokens = _residual_tokens_for_row(row, config, process_cfg)
        vocab.update(tokens)
        for token in tokens:
            frequencies[token] = int(frequencies.get(token, 0)) + 1
        stats["count"] += 1
        stats["vocab"] = len(vocab)
        stats["frequencies"] = frequencies
        if progress_interval > 0 and stats["count"] % progress_interval == 0:
            print(
                f"[WORD2VEC][{dataset_name}] train_read_progress "
                f"count={stats['count']} vocab={stats['vocab']}",
                file=sys.stderr,
                flush=True,
            )
        yield tokens
    print(
        f"[WORD2VEC][{dataset_name}] train_read_end "
        f"count={stats['count']} vocab={stats['vocab']}",
        file=sys.stderr,
        flush=True,
    )


def train_dataset(args: argparse.Namespace, dataset: str) -> dict[str, Any]:
    """Train one dataset Word2Vec residual model and save its state."""
    dataset_name = str(dataset).upper()
    config = SlimConfig(
        dataset=dataset_name,
        semantic_embedding_method="word2vec",
        word2vec_window=int(args.word2vec_window),
        word2vec_min_count=int(args.word2vec_min_count),
        word2vec_sg=int(args.word2vec_sg),
        word2vec_negative=int(args.word2vec_negative),
        word2vec_epochs=int(args.word2vec_epochs),
        word2vec_workers=int(args.word2vec_workers),
        word2vec_seed=int(args.word2vec_seed),
        word2vec_oov_policy=str(args.word2vec_oov_policy),
        word2vec_corpus_file_dir=str(args.word2vec_corpus_file_dir),
        max_tokens_per_node=int(args.max_tokens_per_node),
        fetch_size=int(args.fetch_size),
        progress_interval_events=int(args.progress_interval_events),
        process_semantics_config=str(args.process_semantics_config),
        semantic_mode=str(args.semantic_mode),
    )
    process_cfg = load_process_semantic_config(config.process_semantics_config)
    embed_config = _make_residual_embedding_config(config, process_cfg)
    embedder = build_residual_embedder(embed_config)

    db_cfg = _cfg_for_dataset(dataset_name)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    train_days = _split_days(db_cfg)
    cur, conn = init_database_connection(db_cfg)
    started = time.time()
    read_stats = {"count": 0, "vocab": 0}
    try:
        node_maps = build_node_maps(cur)
        rows = stream_dataset_rows(
            conn,
            str(db_cfg.dataset.year_month),
            train_days,
            node_maps,
            use_event_type_filter(db_cfg),
            int(config.fetch_size),
            int(args.max_train_events),
            abnormal_nodes=set(),
            optc_action_mode=is_optc_dataset(dataset_name),
        )
        print(
            f"[WORD2VEC][{dataset_name}] fit_start "
            f"days={train_days} window={embed_config.word2vec_window} "
            f"epochs={embed_config.word2vec_epochs}",
            file=sys.stderr,
            flush=True,
        )
        embedder.fit(
            _stream_train_sentences(
                rows,
                config,
                process_cfg,
                dataset_name,
                int(config.progress_interval_events),
                read_stats,
            ),
        )
        print(f"[WORD2VEC][{dataset_name}] fit_end", file=sys.stderr, flush=True)
    finally:
        cur.close()
        conn.close()

    elapsed_sec = time.time() - started
    out_dir = Path(args.out_dir)
    tag = str(args.out_tag)
    base_name = f"{dataset_name}_{tag}_word2vec_window{int(args.word2vec_window)}"
    model_path = out_dir / f"{base_name}.pkl"
    meta_path = out_dir / f"{base_name}.json"
    semantic_payload = _word2vec_metadata_semantic_payload(args, dataset_name)
    frequencies = dict(read_stats.get("frequencies", {}))
    vocab = sorted(str(token) for token in frequencies)
    state = {
        "schema_version": "residual_word2vec_model_v1",
        "dataset": dataset_name.lower(),
        "dataset_id": dataset_name,
        "model_state": embedder.state_dict(),
        "embedding_config": asdict(embed_config),
        "semantic_mode": semantic_payload["semantic_mode"],
        "train_days": train_days,
        "train_splits": [f"day_{int(day)}" for day in train_days],
        "year_month": str(db_cfg.dataset.year_month),
        "event_type_filter": bool(use_event_type_filter(db_cfg)),
        "event_filter": "Orthrus10" if use_event_type_filter(db_cfg) else "none",
        "semantic_rules": semantic_payload["semantic_rules"],
        "legacy_path_used": bool(semantic_payload["legacy_path_used"]),
        "vector_dim": int(embed_config.latent_dim),
        "window": int(embed_config.word2vec_window),
        "min_count": int(embed_config.word2vec_min_count),
        "word2vec_min_count": int(embed_config.word2vec_min_count),
        "leakage_contract": "train_split_only_no_labels_no_validation_no_test_scoring",
        "stats": embedder.stats(),
        "elapsed_sec": elapsed_sec,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = dict(state)
    metadata.pop("model_state", None)
    metadata["model_path"] = str(model_path)
    _write_json(meta_path, metadata)
    run_dir = out_dir / f"{dataset_name}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "vocab.json", {"tokens": vocab, "vocab_size": len(vocab)})
    _write_json(
        run_dir / "residual_sentence_count.json",
        {"sentence_count": int(read_stats["count"])},
    )
    _write_json(
        run_dir / "word2vec_training_summary.json",
        {
            **metadata,
            "vocab_size": len(vocab),
            "sentence_count": int(read_stats["count"]),
        },
    )
    with (run_dir / "token_frequency.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token", "count"])
        writer.writeheader()
        for token, count in sorted(frequencies.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow({"token": token, "count": int(count)})
    with (run_dir / "word2vec_training.log").open("w", encoding="utf-8") as handle:
        handle.write(
            f"dataset={dataset_name}\n"
            f"semantic_mode={semantic_payload['semantic_mode']}\n"
            f"sentence_count={int(read_stats['count'])}\n"
            f"vocab_size={len(vocab)}\n"
            f"model_path={model_path}\n"
        )
    print(
        f"[WORD2VEC][{dataset_name}] saved model={model_path} meta={meta_path}",
        file=sys.stderr,
        flush=True,
    )
    return {
        "dataset": dataset_name,
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "stats": embedder.stats(),
        "elapsed_sec": elapsed_sec,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--out_dir", default="outputs/models/residual_word2vec")
    parser.add_argument("--out_tag", default="DETAIL_SEMANTICS")
    parser.add_argument("--max_train_events", type=int, default=0)
    parser.add_argument("--fetch_size", type=int, default=10000)
    parser.add_argument("--progress_interval_events", type=int, default=100000)
    parser.add_argument("--max_tokens_per_node", type=int, default=8)
    parser.add_argument("--process_semantics_config", default="configs/common/process_semantics.yaml")
    parser.add_argument("--semantic_mode", default=CLEARSCOPE_V3_SEMANTIC_MODE)
    parser.add_argument("--word2vec_window", type=int, default=3)
    parser.add_argument("--word2vec_min_count", type=int, default=1)
    parser.add_argument("--word2vec_sg", type=int, default=1)
    parser.add_argument("--word2vec_negative", type=int, default=5)
    parser.add_argument("--word2vec_epochs", type=int, default=10)
    parser.add_argument("--word2vec_workers", type=int, default=4)
    parser.add_argument("--word2vec_seed", type=int, default=0)
    parser.add_argument("--word2vec_oov_policy", choices=("unk", "zero"), default="unk")
    parser.add_argument(
        "--word2vec_corpus_file_dir",
        default=os.getenv("TMP_ROOT", str(Path(tempfile.gettempdir()) / "cs4m_word2vec")),
    )
    return parser.parse_args()


def main() -> None:
    """Train all requested dataset models."""
    args = parse_args()
    summaries = [train_dataset(args, dataset) for dataset in args.datasets]
    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
