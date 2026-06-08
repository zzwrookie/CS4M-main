from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from os import PathLike
from pathlib import Path
from typing import Mapping

import numpy as np

from cs4m.phase3e.event_index import EVENT_INDEX_DTYPE
from cs4m.phase3e.node_action_tables import (
    ORTHRUS10_ACTION_NAMES,
    compute_node_action_target,
)


def x_context_fingerprint(inputs: Mapping[str, object]) -> str:
    """Return a stable hash for JSON-safe Phase3E X_context fingerprint inputs."""
    encoded = json.dumps(
        _json_safe(dict(inputs)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_x_context_memmap(
    path: str | Path,
    event_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    model,
    fingerprint_inputs: Mapping[str, object],
    reset_state: bool = True,
    reset_on_exit: bool = True,
) -> dict[str, object]:
    """Build train-only Phase3E X_context memmap with context-before-update state order."""
    records = _as_event_index_array(event_index)
    node_table = _as_finite_float32_table(node_embeddings, "node_embeddings")
    action_table = _as_finite_float32_table(action_embeddings, "action_embeddings")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = x_context_fingerprint(
        {
            "caller": dict(fingerprint_inputs),
            "event_index": _array_fingerprint(records),
            "node_embeddings": _array_fingerprint(node_table),
            "action_embeddings": _array_fingerprint(action_table),
            "model_class": type(model).__name__,
            "context_dim": int(model.context_dim),
            "rank": _optional_int(model, "rank"),
            "state_dim": _optional_int(model, "state_dim"),
            "target_dim": _optional_int(model, "target_dim"),
            "latent_dim": _optional_int(model, "latent_dim"),
            "model_config": _model_config_snapshot(model),
            "state_model": _state_model_snapshot(getattr(model, "state_model", None)),
        },
    )
    x_context = np.memmap(
        output_path,
        dtype=np.float32,
        mode="w+",
        shape=(int(records.shape[0]), int(model.context_dim)),
    )

    if reset_state:
        model.reset_state()

    start = time.perf_counter()
    rss_peak = _rss_mb()
    try:
        for index, row in enumerate(records):
            fields = _fields_from_event_index(row)
            z_t = compute_node_action_target(
                node_table,
                action_table,
                int(row["src_node_idx"]),
                int(row["dst_node_idx"]),
                int(row["action_id"]),
            )
            x_context[index] = model.make_context(fields)
            model.update_states(fields, z_t)
            rss_peak = max(rss_peak, _rss_mb())
        x_context.flush()
    finally:
        del x_context
        if reset_on_exit:
            model.reset_state()

    elapsed = max(time.perf_counter() - start, 1e-9)
    size_mb = float(output_path.stat().st_size) / 1024.0 / 1024.0
    return {
        "x_context_memmap_enabled": True,
        "enabled": True,
        "x_context_memmap_path": str(output_path),
        "path": str(output_path),
        "x_context_memmap_size_mb": float(size_mb),
        "size_mb": float(size_mb),
        "x_context_memmap_reused_by": None,
        "x_context_fingerprint": fingerprint,
        "fingerprint": fingerprint,
        "x_context_build_rss_peak_mb": float(rss_peak),
        "build_rss_peak_mb": float(rss_peak),
        "x_context_build_events_per_sec": float(records.shape[0] / elapsed),
        "events_per_sec": float(records.shape[0] / elapsed),
        "state_model": _state_model_name(model),
        "context_dim": int(model.context_dim),
        "target_dim": _optional_int(model, "target_dim"),
        "state_dim": _optional_int(model, "state_dim"),
        "row_count": int(records.shape[0]),
    }


def _fields_from_event_index(row: np.void) -> dict[str, object]:
    action_id = int(row["action_id"])
    if action_id < 0 or action_id >= len(ORTHRUS10_ACTION_NAMES):
        action = action_id
    else:
        action = ORTHRUS10_ACTION_NAMES[action_id]
    return {
        "info_src": int(row["src_node_idx"]),
        "info_dst": int(row["dst_node_idx"]),
        "info_src_type": int(row["src_type_id"]),
        "info_dst_type": int(row["dst_type_id"]),
        "src_role": int(row["src_type_id"]),
        "dst_role": int(row["dst_type_id"]),
        "action_id": action_id,
        "raw_action": action,
        "action": action,
    }


def _as_event_index_array(event_index: np.ndarray) -> np.ndarray:
    records = np.asarray(event_index)
    if records.dtype != EVENT_INDEX_DTYPE:
        raise TypeError(f"event_index must use EVENT_INDEX_DTYPE, got {records.dtype}")
    if records.ndim != 1:
        raise ValueError(f"event_index must be one-dimensional, got {records.ndim}")
    return records


def _as_finite_float32_table(values: np.ndarray, name: str) -> np.ndarray:
    table = np.asarray(values, dtype=np.float32)
    if table.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {table.ndim}")
    if not bool(np.all(np.isfinite(table))):
        raise ValueError(f"{name} must contain only finite values")
    return table


def _array_fingerprint(values: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(values)
    _ensure_array_finite_if_needed(array)
    payload = {
        "dtype": str(array.dtype),
        "shape": [int(value) for value in array.shape],
        "num_bytes": int(array.nbytes),
        "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        **payload,
        "fingerprint_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def _ensure_array_finite_if_needed(array: np.ndarray) -> None:
    if array.dtype.fields is not None:
        for name in array.dtype.fields:
            _ensure_array_finite_if_needed(np.asarray(array[name]))
        return
    if array.dtype.kind == "O":
        raise TypeError("fingerprint arrays must not use object dtype")
    if array.dtype.kind in ("f", "c") and not bool(np.all(np.isfinite(array))):
        raise ValueError("fingerprint arrays must contain only finite values")


def _state_model_name(model) -> str | None:
    state_model = getattr(model, "state_model", None)
    name = getattr(state_model, "name", None)
    if name is not None:
        return str(name)
    config = getattr(model, "config", None)
    configured = getattr(config, "state_model", None)
    if configured is not None:
        return str(configured)
    return None


def _model_config_snapshot(model) -> object:
    config = getattr(model, "config", None)
    if config is None:
        return None
    return _object_public_snapshot(config)


def _state_model_snapshot(state_model: object) -> object:
    if state_model is None:
        return None
    state_dict = getattr(state_model, "state_dict", None)
    if callable(state_dict):
        return {
            "class": type(state_model).__name__,
            "state_dict": _json_safe(state_dict()),
        }
    return {
        "class": type(state_model).__name__,
        "public_attrs": _object_public_snapshot(state_model),
    }


def _object_public_snapshot(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    attrs = getattr(value, "__dict__", None)
    if attrs is None:
        return _json_safe(value)
    public_attrs = {
        key: item
        for key, item in attrs.items()
        if not str(key).startswith("_") and not callable(item)
    }
    return _json_safe(public_attrs)


def _optional_int(model, name: str) -> int | None:
    value = getattr(model, name, None)
    if value is None:
        return None
    return int(value)


def _rss_mb() -> float:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
    except Exception:
        return 0.0
    value = float(usage.ru_maxrss)
    if value <= 0.0:
        return 0.0
    if value > 1024.0 * 1024.0:
        return value / 1024.0 / 1024.0
    return value / 1024.0


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {_json_key(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=_json_sort_key)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            "__ndarray_fingerprint__": _array_fingerprint(value),
        }
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, complex):
        if not math.isfinite(float(value.real)) or not math.isfinite(float(value.imag)):
            raise ValueError("fingerprint inputs must contain only finite complex values")
        return {
            "__complex__": {
                "real": float(value.real),
                "imag": float(value.imag),
            },
        }
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fingerprint inputs must contain only finite floats")
        return value
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"fingerprint input {type(value).__name__} is not JSON-safe")


def _json_sort_key(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _json_key(value: object) -> str:
    if isinstance(value, str):
        return value
    raise TypeError(f"fingerprint key {type(value).__name__} is not JSON-safe")
