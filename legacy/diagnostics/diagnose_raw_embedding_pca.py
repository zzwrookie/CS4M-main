#!/usr/bin/env python3
"""Visualize raw residual Word2Vec event embeddings with PCA.

This diagnostic loads a pre-trained residual embedder and encodes test events.
It does not train SSPM, read validation rows, or calibrate thresholds.
Ground truth is used only after streaming row construction to color diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.config.provnet_utils import init_database_connection
from scripts.data.get_dataset import parse_split_days, use_event_type_filter
from legacy.compatibility.pipeline_runtime_exports import (
    SlimConfig,
    _encode_row,
    _load_process_config,
    _residual_tokens_for_row,
    _test_scoring_node_maps,
    build_node_maps,
    load_ground_truth_indices_readonly,
    stream_dataset_rows_slim,
)
from scripts.pipeline.io.db_stream import _cfg_for_dataset
from cs4m.embeddings.residual import residual_embedder_from_state_dict


def label_group(label: object) -> str:
    """Return benign, suspicious, or malicious label group."""
    text = str(label).strip().lower()
    if text in {"2", "malicious"}:
        return "malicious"
    if text in {"1", "suspicious", "true"}:
        return "suspicious"
    return "benign"


def pca_2d(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project a matrix to two PCA coordinates using deterministic SVD."""
    data = np.asarray(matrix, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")
    if data.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((2,), dtype=np.float32)
    centered = data - np.mean(data, axis=0, keepdims=True)
    if data.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32), np.zeros((2,), dtype=np.float32)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    coords = centered @ components.T
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)
    variances = (singular_values ** 2) / max(data.shape[0] - 1, 1)
    total = float(np.sum(variances))
    ratios = np.zeros((2,), dtype=np.float32)
    if total > 0.0:
        available = min(2, variances.shape[0])
        ratios[:available] = (variances[:available] / total).astype(np.float32)
    return coords[:, :2].astype(np.float32), ratios


class EmbeddingSampleCollector:
    """Keep all attack rows and a bounded deterministic reservoir of benign rows."""

    def __init__(self, max_benign: int, seed: int = 0) -> None:
        self.max_benign = max(int(max_benign), 0)
        self._rng = random.Random(int(seed))
        self._benign_seen = 0
        self._benign_rows: list[dict[str, Any]] = []
        self._attack_rows: list[dict[str, Any]] = []
        self._pending_benign_slot: int | None = None

    def observe(self, row: Mapping[str, Any]) -> None:
        """Observe one encoded event row."""
        safe_row = dict(row)
        if str(safe_row.get("label_group", "benign")) != "benign":
            self._attack_rows.append(safe_row)
            return
        self._benign_seen += 1
        if len(self._benign_rows) < self.max_benign:
            self._benign_rows.append(safe_row)
            return
        if self.max_benign <= 0:
            return
        index = self._rng.randrange(self._benign_seen)
        if index < self.max_benign:
            self._benign_rows[index] = safe_row

    def rows(self) -> list[dict[str, Any]]:
        """Return sampled rows ordered by event index."""
        rows = [*self._benign_rows, *self._attack_rows]
        rows.sort(key=lambda row: int(row.get("event_index", 0)))
        return rows

    def should_encode_benign(self) -> bool:
        """Return true when the next benign row should be encoded for the reservoir."""
        self._benign_seen += 1
        if len(self._benign_rows) < self.max_benign:
            self._pending_benign_slot = len(self._benign_rows)
            return True
        if self.max_benign <= 0:
            self._pending_benign_slot = None
            return False
        index = self._rng.randrange(self._benign_seen)
        if index < self.max_benign:
            self._pending_benign_slot = index
            return True
        self._pending_benign_slot = None
        return False

    def observe_selected_benign(self, row: Mapping[str, Any]) -> None:
        """Store one benign row selected by should_encode_benign."""
        slot = getattr(self, "_pending_benign_slot", None)
        if slot is None:
            return
        safe_row = dict(row)
        if int(slot) >= len(self._benign_rows):
            self._benign_rows.append(safe_row)
        else:
            self._benign_rows[int(slot)] = safe_row
        self._pending_benign_slot = None


def _read_embedder(model_path: Path):
    with model_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("Word2Vec model payload must be a mapping")
    state = payload.get("model_state", payload.get("embedder"))
    if not isinstance(state, Mapping):
        raise KeyError("model payload must contain model_state or embedder")
    return residual_embedder_from_state_dict(dict(state)), dict(payload)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "event_index",
        "timestamp_ns",
        "label",
        "label_group",
        "src_idx",
        "dst_idx",
        "info_src",
        "info_dst",
        "action",
        "object_type",
        "pc1",
        "pc2",
        "token_count",
        "residual_text",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_scatter(path: Path, rows: Sequence[Mapping[str, Any]], title: str) -> None:
    colors = {"benign": "#8a8f98", "suspicious": "#d69e00", "malicious": "#d33f49"}
    sizes = {"benign": 6, "suspicious": 18, "malicious": 24}
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    for group in ("benign", "suspicious", "malicious"):
        group_rows = [row for row in rows if str(row.get("label_group")) == group]
        if not group_rows:
            continue
        ax.scatter(
            [float(row["pc1"]) for row in group_rows],
            [float(row["pc2"]) for row in group_rows],
            s=sizes[group],
            c=colors[group],
            alpha=0.45 if group == "benign" else 0.82,
            edgecolors="none",
            label=f"{group} ({len(group_rows)})",
        )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _write_html(path: Path, rows: Sequence[Mapping[str, Any]], title: str) -> None:
    colors = {"benign": "#8a8f98", "suspicious": "#d69e00", "malicious": "#d33f49"}
    if not rows:
        path.write_text("<!doctype html><meta charset='utf-8'><p>No rows</p>", encoding="utf-8")
        return
    width = 1000
    height = 800
    margin = 70
    pc1_values = [float(row["pc1"]) for row in rows]
    pc2_values = [float(row["pc2"]) for row in rows]
    min_x, max_x = min(pc1_values), max(pc1_values)
    min_y, max_y = min(pc2_values), max(pc2_values)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        "xmlns='http://www.w3.org/2000/svg'>",
        "<style>text{font-family:Arial,sans-serif;font-size:13px}.pt{opacity:.62}</style>",
        f"<text x='{margin}' y='32' style='font-size:18px;font-weight:700'>"
        f"{html.escape(title)}</text>",
    ]
    for row in rows:
        x = margin + ((float(row["pc1"]) - min_x) / span_x) * (width - 2 * margin)
        y = height - margin - ((float(row["pc2"]) - min_y) / span_y) * (height - 2 * margin)
        group = str(row.get("label_group", "benign"))
        radius = 2.2 if group == "benign" else 4.4
        tooltip = (
            f"event={row.get('event_index')} label={group} "
            f"action={row.get('action')} object={row.get('object_type')}"
        )
        parts.append(
            f"<circle class='pt' cx='{x:.2f}' cy='{y:.2f}' r='{radius}' "
            f"fill='{colors.get(group, '#777')}'><title>{html.escape(tooltip)}</title></circle>",
        )
    legend_x = margin
    for offset, group in enumerate(("benign", "suspicious", "malicious")):
        x = legend_x + offset * 145
        parts.append(f"<circle cx='{x}' cy='{height - 28}' r='5' fill='{colors[group]}'/>")
        parts.append(f"<text x='{x + 10}' y='{height - 24}'>{group}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _config_for_model(dataset: str, model_payload: Mapping[str, Any], max_test_events: int) -> SlimConfig:
    embedding_config = dict(model_payload.get("embedding_config", {}))
    return SlimConfig(
        dataset=str(dataset).upper(),
        latent_dim=int(embedding_config.get("latent_dim", 64)),
        semantic_embedding_method=str(embedding_config.get("method", "word2vec")),
        word2vec_window=int(embedding_config.get("word2vec_window", 3)),
        max_test_events=int(max_test_events),
        theia_netflow_policy="fixed",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the PCA diagnostic."""
    dataset = str(args.dataset).upper()
    model_path = Path(args.model_path)
    embedder, model_payload = _read_embedder(model_path)
    config = _config_for_model(dataset, model_payload, int(args.max_test_events))
    process_cfg = _load_process_config(config)
    db_cfg = _cfg_for_dataset(dataset)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    test_days = parse_split_days(db_cfg.dataset.test_splits)
    out_dir = Path(args.out_dir)
    prefix = f"{dataset.lower()}_raw_word2vec_pca"
    started = time.perf_counter()
    collector = EmbeddingSampleCollector(max_benign=int(args.max_benign_points), seed=0)
    label_counts: Counter[str] = Counter()
    scanned_count = 0
    encoded_count = 0
    cur, conn = init_database_connection(db_cfg)
    try:
        print(f"[raw-pca] build_node_maps_start dataset={dataset}", file=sys.stderr, flush=True)
        node_maps = build_node_maps(cur)
        print(f"[raw-pca] build_node_maps_end dataset={dataset}", file=sys.stderr, flush=True)
        abnormal_nodes = load_ground_truth_indices_readonly(db_cfg, node_maps["uuid2index"])
        test_node_maps = _test_scoring_node_maps(node_maps)
        stream_status: dict[str, str] = {}
        rows = stream_dataset_rows_slim(
            conn,
            str(db_cfg.dataset.year_month),
            test_days,
            test_node_maps,
            use_event_type_filter(db_cfg),
            int(args.fetch_size),
            int(args.max_test_events),
            str(args.db_stream_mode),
            abnormal_nodes=abnormal_nodes,
            stream_status=stream_status,
        )
        for row in rows:
            scanned_count += 1
            group = label_group(row.get("label", 0))
            label_counts[group] += 1
            encode_row = group != "benign" or collector.should_encode_benign()
            if not encode_row:
                if scanned_count % int(args.progress_interval_events) == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"[raw-pca] scanned={scanned_count} encoded={encoded_count} "
                        f"rate={scanned_count / elapsed:.1f}/s labels={dict(label_counts)}",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            vector = _encode_row(embedder, row, config, process_cfg)
            tokens = _residual_tokens_for_row(row, config, process_cfg)
            selected_row = {
                "event_index": int(row["event_index"]),
                "timestamp_ns": int(row["timestamp_ns"]),
                "label": int(row.get("label", 0)),
                "label_group": group,
                "src_idx": int(row["src_idx"]),
                "dst_idx": int(row["dst_idx"]),
                "info_src": int(row["info_src"]),
                "info_dst": int(row["info_dst"]),
                "action": str(row["action"]),
                "object_type": str(row["object_type"]),
                "embedding": vector.astype(np.float32, copy=False),
                "token_count": int(len(tokens)),
                "residual_text": " ".join(tokens),
            }
            if group == "benign":
                collector.observe_selected_benign(selected_row)
            else:
                collector.observe(selected_row)
            encoded_count += 1
            if scanned_count % int(args.progress_interval_events) == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(
                    f"[raw-pca] scanned={scanned_count} encoded={encoded_count} "
                    f"rate={scanned_count / elapsed:.1f}/s labels={dict(label_counts)}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        conn.close()

    sampled_rows = collector.rows()
    matrix = np.asarray([row.pop("embedding") for row in sampled_rows], dtype=np.float32)
    coords, explained = pca_2d(matrix)
    for row, coord in zip(sampled_rows, coords, strict=True):
        row["pc1"] = float(coord[0])
        row["pc2"] = float(coord[1])
    csv_path = out_dir / f"{prefix}.csv"
    png_path = out_dir / f"{prefix}.png"
    html_path = out_dir / f"{prefix}.html"
    meta_path = out_dir / f"{prefix}.json"
    title = (
        f"{dataset} raw Word2Vec event embedding PCA "
        f"(latent={getattr(embedder, 'latent_dim', 'unknown')})"
    )
    _write_csv(csv_path, sampled_rows)
    _write_scatter(png_path, sampled_rows, title)
    _write_html(html_path, sampled_rows, title)
    metadata = {
        "dataset": dataset,
        "model_path": str(model_path),
        "out_csv": str(csv_path),
        "out_png": str(png_path),
        "out_html": str(html_path),
        "scanned_count": int(scanned_count),
        "encoded_count": int(encoded_count),
        "sampled_count": int(len(sampled_rows)),
        "label_counts": dict(label_counts),
        "pca_explained_variance_ratio": [float(value) for value in explained],
        "test_days": [int(day) for day in test_days],
        "uses_validation": False,
        "trains_sspm": False,
        "db_stream_status": stream_status if "stream_status" in locals() else {},
        "elapsed_sec": float(time.perf_counter() - started),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    return metadata


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CADETS_E3")
    parser.add_argument(
        "--model_path",
        default=(
            "outputs/models/residual_word2vec/"
            "CADETS_E3_RAW_DETAIL_RULES_V1_LATENT64_word2vec_window3.pkl"
        ),
    )
    parser.add_argument("--out_dir", default="outputs/diagnostics/raw_embedding_pca")
    parser.add_argument("--max_test_events", type=int, default=0)
    parser.add_argument("--max_benign_points", type=int, default=60000)
    parser.add_argument("--fetch_size", type=int, default=10000)
    parser.add_argument("--progress_interval_events", type=int, default=100000)
    parser.add_argument("--db_stream_mode", choices=("auto", "temp_table", "python_lookup"), default="auto")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    run(parse_args())


if __name__ == "__main__":
    main()
