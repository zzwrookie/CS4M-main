from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from cs4m.phase3e.word2vec_adapter import (
    ResidualWord2VecTokenAdapter,
    word2vec_adapter_fingerprint,
)


ORTHRUS10_ACTION_NAMES: tuple[str, ...] = (
    "EVENT_CLONE",
    "EVENT_CONNECT",
    "EVENT_EXECUTE",
    "EVENT_OPEN",
    "EVENT_READ",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG",
    "EVENT_SENDMSG",
    "EVENT_SENDTO",
    "EVENT_WRITE",
)

DEFAULT_COVERAGE_AUDIT_FILENAME = (
    "phase3e_residual_word2vec_node_action_coverage_audit.json"
)
DEFAULT_COVERED_SPLITS = ("train", "validation", "test")


@dataclass
class NodeEmbeddingTable:
    """In-memory Phase3E node embedding table plus label-free audit metadata."""

    embeddings: np.ndarray
    node_id_to_idx: dict[int, int]
    meta: dict[str, object]
    coverage: dict[str, object]


@dataclass
class ActionEmbeddingTable:
    """In-memory Phase3E ORTHRUS10 action embedding table."""

    embeddings: np.ndarray
    action_id_to_name: dict[int, str]
    name_to_action_id: dict[str, int]
    meta: dict[str, object]


def _tokenize_action_name(name: str) -> list[str]:
    parts = [part.lower() for part in re.split(r"[^A-Za-z0-9]+", str(name)) if part]
    if len(parts) >= 2 and parts[0] == "event":
        return [parts[0], "_".join(parts)]
    return parts


def _as_sorted_ints(values: Sequence[int] | set[int] | frozenset[int]) -> list[int]:
    return sorted(int(value) for value in values)


def _coverage_for_tokens(
    tokens: Sequence[object],
    adapter: ResidualWord2VecTokenAdapter,
) -> dict[str, object]:
    coverage = adapter.coverage(tokens)
    total = int(coverage["total"])
    known = int(coverage["known"])
    oov = int(coverage["oov"])
    return {
        "total": total,
        "known": known,
        "oov": oov,
        "all_oov": known == 0,
        "partial_oov": known > 0 and oov > 0,
        "zero": known == 0,
    }


def _token_coverage(
    *,
    node_ids: Sequence[int],
    node_tokens_by_id: Mapping[int, Sequence[object]],
    adapter: ResidualWord2VecTokenAdapter,
) -> float:
    total = 0
    known = 0
    for node_id in node_ids:
        coverage = adapter.coverage(node_tokens_by_id[int(node_id)])
        total += int(coverage["total"])
        known += int(coverage["known"])
    return float(known / total) if total else 0.0


def _split_counts(
    split_node_ids: Mapping[str, Sequence[int] | set[int] | frozenset[int]],
) -> dict[str, int]:
    return {
        "train": len(set(split_node_ids.get("train", set()))),
        "validation": len(set(split_node_ids.get("validation", set()))),
        "test": len(set(split_node_ids.get("test", set()))),
    }


def build_node_embedding_table(
    *,
    node_tokens_by_id: Mapping[int, Sequence[object]],
    split_node_ids: Mapping[str, Sequence[int] | set[int] | frozenset[int]],
    adapter: ResidualWord2VecTokenAdapter,
    src_node_ids: Sequence[int] | set[int] | frozenset[int] | None = None,
    dst_node_ids: Sequence[int] | set[int] | frozenset[int] | None = None,
) -> NodeEmbeddingTable:
    """Build node embeddings for nodes referenced by train/validation/test streams."""
    node_ids: list[int] = []
    seen: set[int] = set()
    for split in DEFAULT_COVERED_SPLITS:
        for node_id in _as_sorted_ints(split_node_ids.get(split, set())):
            if node_id not in seen:
                node_ids.append(node_id)
                seen.add(node_id)

    missing = [node_id for node_id in node_ids if int(node_id) not in node_tokens_by_id]
    if missing:
        raise ValueError(f"missing node semantic tokens for node ids: {missing[:5]}")

    embeddings = np.zeros((len(node_ids), int(adapter.dim)), dtype=np.float32)
    node_id_to_idx = {int(node_id): index for index, node_id in enumerate(node_ids)}
    split_sets = {
        split: set(_as_sorted_ints(split_node_ids.get(split, set())))
        for split in DEFAULT_COVERED_SPLITS
    }

    all_oov = 0
    partial_oov = 0
    zero_nodes = 0
    token_total = 0
    token_known = 0
    for node_id in node_ids:
        tokens = list(node_tokens_by_id[int(node_id)])
        idx = node_id_to_idx[int(node_id)]
        embeddings[idx] = adapter.encode_tokens(tokens)
        coverage = _coverage_for_tokens(tokens, adapter)
        token_total += int(coverage["total"])
        token_known += int(coverage["known"])
        all_oov += int(bool(coverage["all_oov"]))
        partial_oov += int(bool(coverage["partial_oov"]))
        zero_nodes += int(bool(coverage["zero"]))

    counts = _split_counts(split_node_ids)
    node_count = len(node_ids)
    fingerprint = word2vec_adapter_fingerprint(adapter)
    src_nodes = set(_as_sorted_ints(src_node_ids or set()))
    dst_nodes = set(_as_sorted_ints(dst_node_ids or set()))
    coverage_summary: dict[str, object] = {
        "train_node_token_coverage": _token_coverage(
            node_ids=_as_sorted_ints(split_sets["train"]),
            node_tokens_by_id=node_tokens_by_id,
            adapter=adapter,
        ),
        "validation_node_token_coverage": _token_coverage(
            node_ids=_as_sorted_ints(split_sets["validation"]),
            node_tokens_by_id=node_tokens_by_id,
            adapter=adapter,
        ),
        "test_node_token_coverage": _token_coverage(
            node_ids=_as_sorted_ints(split_sets["test"]),
            node_tokens_by_id=node_tokens_by_id,
            adapter=adapter,
        ),
        "all_node_token_coverage": float(token_known / token_total) if token_total else 0.0,
        "num_nodes_total": int(node_count),
        "num_nodes_all_oov": int(all_oov),
        "num_nodes_partial_oov": int(partial_oov),
        "zero_node_count": int(zero_nodes),
        "fallback_count": int(all_oov),
        "all_oov_node_ratio": float(all_oov / node_count) if node_count else 0.0,
        "partial_oov_node_ratio": float(partial_oov / node_count) if node_count else 0.0,
        "word2vec_model_path": str(adapter.model_path),
        "word2vec_fingerprint": fingerprint,
        "node_word2vec_source": "residual_pretrained",
    }
    meta: dict[str, object] = {
        "node_universe_source": "event_stream_splits",
        "covered_splits": ",".join(DEFAULT_COVERED_SPLITS),
        "covered_split_names": list(DEFAULT_COVERED_SPLITS),
        "covers_only_event_referenced_nodes": True,
        "covers_full_db_node_table": False,
        "num_nodes_total_in_node_table": int(node_count),
        "num_train_nodes": int(counts["train"]),
        "num_validation_nodes": int(counts["validation"]),
        "num_test_nodes": int(counts["test"]),
        "num_src_nodes": int(len(src_nodes)),
        "num_dst_nodes": int(len(dst_nodes)),
        "num_nodes_seen_in_multiple_splits": int(
            sum(
                1
                for node_id in node_ids
                if sum(1 for split_nodes in split_sets.values() if node_id in split_nodes) > 1
            ),
        ),
        "word2vec_model_path": str(adapter.model_path),
        "word2vec_fingerprint": fingerprint,
        "node_word2vec_source": "residual_pretrained",
    }
    return NodeEmbeddingTable(embeddings, node_id_to_idx, meta, coverage_summary)


def build_action_embedding_table(adapter: ResidualWord2VecTokenAdapter) -> ActionEmbeddingTable:
    """Build the stable ORTHRUS10 action embedding table."""
    action_id_to_name = {index: name for index, name in enumerate(ORTHRUS10_ACTION_NAMES)}
    name_to_action_id = {name: index for index, name in action_id_to_name.items()}
    embeddings = np.zeros((len(action_id_to_name), int(adapter.dim)), dtype=np.float32)
    known = 0
    total = 0
    for index, name in action_id_to_name.items():
        tokens = _tokenize_action_name(name)
        coverage = adapter.coverage(tokens)
        known += int(coverage["known"])
        total += int(coverage["total"])
        embeddings[index] = adapter.encode_tokens(tokens)
    meta: dict[str, object] = {
        "action_count": int(len(action_id_to_name)),
        "action_token_coverage": float(known / total) if total else 0.0,
        "word2vec_model_path": str(adapter.model_path),
        "word2vec_fingerprint": word2vec_adapter_fingerprint(adapter),
        "node_word2vec_source": "residual_pretrained",
    }
    return ActionEmbeddingTable(embeddings, action_id_to_name, name_to_action_id, meta)


def compute_node_action_target(
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    src_node_idx: int,
    dst_node_idx: int,
    action_id: int,
) -> np.ndarray:
    """Compute z_t = (e_src + e_dst + e_action) / 3 for one indexed event."""
    _validate_embedding_index("src_node_idx", src_node_idx, len(node_embeddings))
    _validate_embedding_index("dst_node_idx", dst_node_idx, len(node_embeddings))
    _validate_embedding_index("action_id", action_id, len(action_embeddings))
    src = np.asarray(node_embeddings[int(src_node_idx)], dtype=np.float32)
    dst = np.asarray(node_embeddings[int(dst_node_idx)], dtype=np.float32)
    action = np.asarray(action_embeddings[int(action_id)], dtype=np.float32)
    return ((src + dst + action) / np.float32(3.0)).astype(np.float32, copy=False)


def _validate_embedding_index(name: str, value: int, size: int) -> None:
    index = int(value)
    if index < 0 or index >= int(size):
        raise IndexError(f"{name} out of bounds: {index} not in [0, {int(size)})")


def build_node_action_coverage_audit(
    node_table: NodeEmbeddingTable,
    action_table: ActionEmbeddingTable,
) -> dict[str, object]:
    """Combine node and action coverage metadata into a JSON-safe audit record."""
    node_fingerprint = str(node_table.meta.get("word2vec_fingerprint", ""))
    action_fingerprint = str(action_table.meta.get("word2vec_fingerprint", ""))
    if node_fingerprint and action_fingerprint and node_fingerprint != action_fingerprint:
        raise ValueError("node/action Word2Vec fingerprints do not match")

    audit: dict[str, object] = {
        **node_table.coverage,
        **node_table.meta,
        "action_token_coverage": float(action_table.meta.get("action_token_coverage", 0.0)),
        "action_count": int(
            action_table.meta.get("action_count", len(action_table.action_id_to_name)),
        ),
    }
    if "covered_split_names" in audit:
        audit["covered_split_names"] = list(audit["covered_split_names"])
    if action_fingerprint and not audit.get("word2vec_fingerprint"):
        audit["word2vec_fingerprint"] = action_fingerprint
    if "node_word2vec_source" not in audit:
        audit["node_word2vec_source"] = "residual_pretrained"
    return audit


def save_node_embedding_table(table: NodeEmbeddingTable, root: str | Path) -> dict[str, str]:
    """Save node embeddings, id map, and JSON-safe metadata sidecar."""
    output_dir = Path(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "node_embeddings.npy"
    id_map_path = output_dir / "node_id_to_idx.pkl"
    meta_path = output_dir / "node_embedding_meta.json"

    np.save(embeddings_path, table.embeddings.astype(np.float32, copy=False))
    with id_map_path.open("wb") as handle:
        pickle.dump(table.node_id_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path.write_text(
        json.dumps({**table.coverage, **table.meta}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "node_embeddings": str(embeddings_path),
        "node_id_to_idx": str(id_map_path),
        "node_embedding_meta": str(meta_path),
    }


def save_action_embedding_table(table: ActionEmbeddingTable, root: str | Path) -> dict[str, str]:
    """Save action embeddings, stable id map, and JSON-safe metadata sidecar."""
    output_dir = Path(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "action_embeddings.npy"
    action_map_path = output_dir / "action_id_to_name.json"
    meta_path = output_dir / "action_embedding_meta.json"

    np.save(embeddings_path, table.embeddings.astype(np.float32, copy=False))
    action_map_path.write_text(
        json.dumps(table.action_id_to_name, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    meta_path.write_text(
        json.dumps(table.meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "action_embeddings": str(embeddings_path),
        "action_id_to_name": str(action_map_path),
        "action_embedding_meta": str(meta_path),
    }


def save_node_action_coverage_audit(
    audit: Mapping[str, object],
    root_or_path: str | Path,
    filename: str = DEFAULT_COVERAGE_AUDIT_FILENAME,
) -> str:
    """Save a combined Phase3E node/action coverage audit JSON file."""
    target = Path(root_or_path)
    if target.suffix.lower() == ".json":
        audit_path = target
    else:
        audit_path = target / str(filename)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(dict(audit), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(audit_path)
