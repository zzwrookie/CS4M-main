from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cs4m.phase3e.event_index import (
    EVENT_INDEX_DTYPE,
    event_index_fingerprint,
    open_event_index_memmap,
)


def used_node_count_summary(
    *,
    node_embeddings: np.ndarray,
    split_event_indexes: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Return label-free counts for nodes referenced by train/validation/test event indexes."""
    used_by_split = {
        str(split): _used_nodes_for_event_index(index)
        for split, index in split_event_indexes.items()
    }
    used_all: set[int] = set()
    for nodes in used_by_split.values():
        used_all.update(nodes)
    global_count = int(np.asarray(node_embeddings).shape[0])
    used_count = int(len(used_all))
    state_dim = int(np.asarray(node_embeddings).shape[1])
    bytes_per_embedding = int(state_dim * 4)
    return {
        "node_embedding_lookup_mode": "compact_used_nodes",
        "num_nodes_global_embedding_table": global_count,
        "num_used_nodes_train": int(len(used_by_split.get("train", set()))),
        "num_used_nodes_validation": int(len(used_by_split.get("validation", set()))),
        "num_used_nodes_test": int(len(used_by_split.get("test", set()))),
        "num_used_nodes_all": used_count,
        "embedding_dtype": "float32",
        "embedding_dim": state_dim,
        "estimated_global_float32_mb": _bytes_to_mb(global_count * bytes_per_embedding),
        "estimated_compact_float32_mb": _bytes_to_mb(used_count * bytes_per_embedding),
    }


def embedding_memory_breakdown(
    *,
    node_embeddings: np.ndarray,
    split_event_indexes: Mapping[str, np.ndarray],
    active_state_table_mb: float = 0.0,
    endpoint_cache_mb: float = 0.0,
    online_lazy_node_count_estimate: int | None = None,
    global_embedding_node_count: int | None = None,
) -> dict[str, Any]:
    """Return label-free embedding memory counts for pipeline, test replay, and lazy deploy."""
    used_by_split = {
        str(split): _used_nodes_for_event_index(index)
        for split, index in split_event_indexes.items()
    }
    train_nodes = used_by_split.get("train", set())
    validation_nodes = used_by_split.get("validation", set())
    test_nodes = used_by_split.get("test", set())
    union_nodes = set(train_nodes)
    union_nodes.update(validation_nodes)
    union_nodes.update(test_nodes)
    loaded_embedding_count = int(np.asarray(node_embeddings).shape[0])
    global_count = int(
        loaded_embedding_count
        if global_embedding_node_count is None
        else global_embedding_node_count,
    )
    embedding_dim = int(np.asarray(node_embeddings).shape[1])
    bytes_per_embedding = int(embedding_dim * 4)
    if online_lazy_node_count_estimate is None:
        lazy_count = int(len(test_nodes))
    else:
        lazy_count = int(online_lazy_node_count_estimate)
    full_pipeline_mb = _bytes_to_mb(len(union_nodes) * bytes_per_embedding)
    test_stream_mb = _bytes_to_mb(len(test_nodes) * bytes_per_embedding)
    lazy_mb = _bytes_to_mb(lazy_count * bytes_per_embedding)
    active_state_mb = float(active_state_table_mb)
    endpoint_mb = float(endpoint_cache_mb)
    return {
        "definition": (
            "full_pipeline covers train/validation/test union nodes; test_stream covers only "
            "test replay nodes and is not strict future-blind online deployment; "
            "online_lazy is an estimate for active/cached embedding lookup."
        ),
        "global_embedding_nodes": global_count,
        "loaded_embedding_nodes": loaded_embedding_count,
        "train_nodes": int(len(train_nodes)),
        "validation_nodes": int(len(validation_nodes)),
        "test_nodes": int(len(test_nodes)),
        "union_train_val_test_nodes": int(len(union_nodes)),
        "test_only_nodes": int(len(test_nodes - train_nodes - validation_nodes)),
        "train_val_intersection": int(len(train_nodes & validation_nodes)),
        "train_test_intersection": int(len(train_nodes & test_nodes)),
        "validation_test_intersection": int(len(validation_nodes & test_nodes)),
        "train_val_test_intersection": int(len(train_nodes & validation_nodes & test_nodes)),
        "embedding_dim": embedding_dim,
        "embedding_dtype": "float32",
        "global_embedding_float32_mb": float(_bytes_to_mb(global_count * bytes_per_embedding)),
        "loaded_embedding_float32_mb": float(
            _bytes_to_mb(loaded_embedding_count * bytes_per_embedding),
        ),
        "full_pipeline_embedding_float32_mb": float(full_pipeline_mb),
        "test_stream_embedding_float32_mb": float(test_stream_mb),
        "online_lazy_embedding_cache_mb_estimate": float(lazy_mb),
        "online_lazy_embedding_node_count_estimate": lazy_count,
        "endpoint_cache_mb": endpoint_mb,
        "active_state_table_mb": active_state_mb,
        "online_deploy_primary_memory_mb_full_pipeline": float(
            full_pipeline_mb + active_state_mb + endpoint_mb,
        ),
        "online_deploy_primary_memory_mb_test_stream": float(
            test_stream_mb + active_state_mb + endpoint_mb,
        ),
        "online_deploy_primary_memory_mb_lazy_estimate": float(
            lazy_mb + active_state_mb + endpoint_mb,
        ),
    }


def build_compact_used_node_artifacts(
    *,
    node_embeddings: np.ndarray,
    split_event_indexes: Mapping[str, np.ndarray],
    output_dir: str | Path,
    source_node_embedding_path: str | Path,
    source_event_index_paths: Mapping[str, str | Path],
    source_node_id_to_idx: Mapping[int, int] | None = None,
) -> dict[str, str]:
    """Build float32 compact used-node embeddings and remapped event-index memmaps."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    source_embeddings = np.asarray(node_embeddings)
    if source_embeddings.ndim != 2:
        raise ValueError("node_embeddings must be a 2-D array")
    if source_embeddings.dtype != np.float32:
        source_embeddings = source_embeddings.astype(np.float32, copy=False)

    used_nodes = sorted(_all_used_nodes(split_event_indexes))
    if used_nodes and (used_nodes[0] < 0 or used_nodes[-1] >= int(source_embeddings.shape[0])):
        raise IndexError("event_index references node index outside node_embeddings")
    old_to_compact = {int(old): int(new) for new, old in enumerate(used_nodes)}

    compact_path = output_root / "compact_node_embeddings.npy"
    compact = np.asarray(source_embeddings[used_nodes], dtype=np.float32)
    np.save(compact_path, compact)

    original_indices_path = output_root / "compact_original_node_indices.npy"
    np.save(original_indices_path, np.asarray(used_nodes, dtype=np.int32))
    original_to_compact_path = output_root / "original_node_idx_to_compact_idx.npy"
    original_to_compact = np.asarray(
        [[int(old), int(new)] for old, new in old_to_compact.items()],
        dtype=np.int32,
    )
    np.save(original_to_compact_path, original_to_compact)

    artifacts: dict[str, str] = {
        "compact_node_embeddings": str(compact_path),
        "compact_original_node_indices": str(original_indices_path),
        "original_node_idx_to_compact_idx": str(original_to_compact_path),
    }
    compact_node_id_to_idx: dict[int, int] = {}
    if source_node_id_to_idx is not None:
        for raw_node_id, raw_old_idx in source_node_id_to_idx.items():
            old_idx = int(raw_old_idx)
            if old_idx in old_to_compact:
                compact_node_id_to_idx[int(raw_node_id)] = int(old_to_compact[old_idx])
        node_id_to_idx_path = output_root / "node_id_to_idx.pkl"
        with node_id_to_idx_path.open("wb") as handle:
            pickle.dump(compact_node_id_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        compact_node_id_to_idx_alias = output_root / "compact_node_id_to_idx.pkl"
        with compact_node_id_to_idx_alias.open("wb") as handle:
            pickle.dump(compact_node_id_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
        artifacts["node_id_to_idx"] = str(node_id_to_idx_path)
        artifacts["compact_node_id_to_idx"] = str(compact_node_id_to_idx_alias)
    split_fingerprints: dict[str, dict[str, Any]] = {}
    compact_split_fingerprints: dict[str, dict[str, Any]] = {}
    for split, index in split_event_indexes.items():
        split_name = str(split)
        array = _as_event_index(index)
        split_fingerprints[split_name] = event_index_fingerprint(array)
        remapped = array.copy()
        remapped["src_node_idx"] = _remap_indices(array["src_node_idx"], old_to_compact)
        remapped["dst_node_idx"] = _remap_indices(array["dst_node_idx"], old_to_compact)
        compact_split_fingerprints[split_name] = event_index_fingerprint(remapped)
        path = output_root / f"event_index_{split_name}_compact.memmap"
        mapped = np.memmap(path, dtype=EVENT_INDEX_DTYPE, mode="w+", shape=remapped.shape)
        mapped[:] = remapped[:]
        mapped.flush()
        del mapped
        artifacts[f"event_index_{split_name}_compact"] = str(path)

    summary = used_node_count_summary(
        node_embeddings=source_embeddings,
        split_event_indexes=split_event_indexes,
    )
    source_node_embedding_fingerprint = _file_or_array_fingerprint(
        source_node_embedding_path,
        source_embeddings,
    )
    meta = {
        **summary,
        "source_node_embedding_path": str(source_node_embedding_path),
        "source_node_embedding_fingerprint": source_node_embedding_fingerprint,
        "source_event_index_paths": {
            str(key): str(value) for key, value in source_event_index_paths.items()
        },
        "source_event_index_fingerprints": split_fingerprints,
        "compact_event_index_fingerprints": compact_split_fingerprints,
        "compact_node_embedding_fingerprint": _file_fingerprint(compact_path),
        "compact_original_node_indices_fingerprint": _file_fingerprint(original_indices_path),
        "original_node_idx_to_compact_idx_fingerprint": _file_fingerprint(
            original_to_compact_path,
        ),
        "original_node_indices": [int(value) for value in used_nodes],
        "compact_node_id_to_idx_count": int(len(compact_node_id_to_idx)),
        "remap_fingerprint": _mapping_fingerprint(used_nodes),
    }
    meta_path = output_root / "compact_embedding_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    summary_path = output_root / "phase3g_used_node_count_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    artifacts["compact_embedding_meta"] = str(meta_path)
    artifacts["phase3g_used_node_count_summary"] = str(summary_path)
    return artifacts


def load_compact_used_node_artifacts(
    artifact_dir: str | Path,
    *,
    expected_source_node_embedding_fingerprint: Mapping[str, Any],
    expected_source_event_index_fingerprints: Mapping[str, Mapping[str, Any]],
) -> dict[str, Path]:
    """Load compact artifact paths and fail fast on source fingerprint mismatch."""
    root = Path(artifact_dir)
    meta_path = root / "compact_embedding_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing compact embedding meta: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    actual_node_fp = meta.get("source_node_embedding_fingerprint")
    if not _fingerprint_matches_relocated_path(
        actual_node_fp,
        expected_source_node_embedding_fingerprint,
    ):
        raise ValueError("node embedding fingerprint mismatch for compact_used_nodes")
    actual_event_fps = meta.get("source_event_index_fingerprints", {})
    for split, expected in expected_source_event_index_fingerprints.items():
        actual = actual_event_fps.get(str(split))
        if actual != dict(expected):
            raise ValueError(f"event index fingerprint mismatch for split {split}")

    paths: dict[str, Path] = {
        "node_embeddings": root / "compact_node_embeddings.npy",
        "compact_original_node_indices": root / "compact_original_node_indices.npy",
        "original_node_idx_to_compact_idx": root / "original_node_idx_to_compact_idx.npy",
        "compact_embedding_meta": meta_path,
        "phase3g_used_node_count_summary": root / "phase3g_used_node_count_summary.json",
    }
    node_id_to_idx = root / "node_id_to_idx.pkl"
    if node_id_to_idx.exists():
        paths["node_id_to_idx"] = node_id_to_idx
        paths["compact_node_id_to_idx"] = root / "compact_node_id_to_idx.pkl"
    for split in actual_event_fps:
        paths[f"event_index_{split}"] = root / f"event_index_{split}_compact.memmap"
    return paths


def _fingerprint_matches_relocated_path(
    actual: Any,
    expected: Mapping[str, Any],
) -> bool:
    """Return true when two file fingerprints match aside from repository path."""
    if not isinstance(actual, Mapping):
        return False
    actual_without_path = {key: value for key, value in dict(actual).items() if key != "path"}
    expected_without_path = {key: value for key, value in dict(expected).items() if key != "path"}
    return actual_without_path == expected_without_path


def source_node_embedding_fingerprint(
    path: str | Path,
    node_embeddings: np.ndarray,
) -> dict[str, Any]:
    """Return the source fingerprint used by compact artifact metadata."""
    return _file_or_array_fingerprint(path, np.asarray(node_embeddings))


def source_event_index_fingerprints(
    split_event_indexes: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Return source event-index fingerprints used by compact artifact metadata."""
    return {
        str(split): event_index_fingerprint(_as_event_index(index))
        for split, index in split_event_indexes.items()
    }


def open_compact_split_event_index(
    artifact_dir: str | Path,
    *,
    split: str,
    num_events: int,
    mode: str = "r",
) -> np.memmap:
    """Open one compact split event-index memmap."""
    return open_event_index_memmap(
        Path(artifact_dir) / f"event_index_{split}_compact.memmap",
        num_events=int(num_events),
        mode=mode,
    )


def _used_nodes_for_event_index(index: np.ndarray) -> set[int]:
    array = _as_event_index(index)
    nodes = set(int(value) for value in np.asarray(array["src_node_idx"], dtype=np.int64))
    nodes.update(int(value) for value in np.asarray(array["dst_node_idx"], dtype=np.int64))
    return nodes


def _all_used_nodes(split_event_indexes: Mapping[str, np.ndarray]) -> set[int]:
    used: set[int] = set()
    for index in split_event_indexes.values():
        used.update(_used_nodes_for_event_index(index))
    return used


def _as_event_index(index: np.ndarray) -> np.ndarray:
    array = np.asarray(index)
    if array.dtype != EVENT_INDEX_DTYPE:
        raise TypeError(f"event index must use EVENT_INDEX_DTYPE, got {array.dtype}")
    if array.ndim != 1:
        raise ValueError(f"event index must be one-dimensional, got {array.ndim}")
    return array


def _remap_indices(values: np.ndarray, old_to_compact: Mapping[int, int]) -> np.ndarray:
    output = np.empty(values.shape, dtype=np.int32)
    for idx, value in enumerate(values):
        key = int(value)
        if key not in old_to_compact:
            raise KeyError(f"missing compact node index for source node index {key}")
        output[idx] = np.int32(old_to_compact[key])
    return output


def _file_or_array_fingerprint(path: str | Path, array: np.ndarray) -> dict[str, Any]:
    candidate = Path(path)
    if candidate.exists():
        return _file_fingerprint(candidate)
    contiguous = np.ascontiguousarray(array)
    return {
        "path": str(path),
        "shape": [int(value) for value in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "num_bytes": int(contiguous.nbytes),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _file_fingerprint(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(candidate),
        "num_bytes": int(candidate.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def _mapping_fingerprint(used_nodes: list[int]) -> dict[str, Any]:
    payload = json.dumps([int(value) for value in used_nodes], separators=(",", ":"))
    return {
        "num_used_nodes": int(len(used_nodes)),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _bytes_to_mb(num_bytes: int) -> float:
    return float(int(num_bytes) / (1024.0 * 1024.0))
