from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np


EVENT_INDEX_DTYPE_DESCRIPTOR = [
    ["event_id", "<i8"],
    ["src_node_idx", "<i4"],
    ["dst_node_idx", "<i4"],
    ["action_id", "<i2"],
    ["src_type_id", "i1"],
    ["dst_type_id", "i1"],
]
EVENT_INDEX_DTYPE = np.dtype(
    [(name, descriptor) for name, descriptor in EVENT_INDEX_DTYPE_DESCRIPTOR],
)
_FORBIDDEN_ROW_INDEX_FIELDS = frozenset(("src_node_idx", "dst_node_idx", "action_id"))
_PLAIN_SIGNED_INTEGER_RE = re.compile(r"^[+-]?[0-9]+$")


def build_event_index_records(
    rows: Sequence[Mapping[str, object]],
    node_id_to_idx: Mapping[object, int],
    action_name_to_id: Mapping[object, int],
) -> np.ndarray:
    """Map compact stream rows into Phase3E event-index dtype records."""
    records = np.zeros((len(rows),), dtype=EVENT_INDEX_DTYPE)
    for row_index, row in enumerate(rows):
        _reject_direct_index_fields(row)
        event_id = _required_strict_int(row, "event_id", np.int64, allow_negative=False)
        src_node_id = _required_strict_int(row, "src_node_id", np.int64, allow_negative=False)
        dst_node_id = _required_strict_int(row, "dst_node_id", np.int64, allow_negative=False)
        action_name = _required_value(row, "action_name")
        src_node_idx = _mapped_id(
            "src_node_idx",
            "src_node_id",
            src_node_id,
            node_id_to_idx,
            np.int32,
        )
        dst_node_idx = _mapped_id(
            "dst_node_idx",
            "dst_node_id",
            dst_node_id,
            node_id_to_idx,
            np.int32,
        )
        action_id = _mapped_id(
            "action_id",
            "action_name",
            action_name,
            action_name_to_id,
            np.int16,
        )

        records[row_index] = (
            event_id,
            src_node_idx,
            dst_node_idx,
            action_id,
            _required_strict_int(row, "src_type_id", np.int8, allow_negative=False),
            _required_strict_int(row, "dst_type_id", np.int8, allow_negative=False),
        )
    return records


def event_index_fingerprint(records: np.ndarray) -> dict[str, object]:
    """Return a stable JSON-safe fingerprint carrying dtype, shape, and bytes."""
    array = _as_event_index_array(records)
    contiguous = np.ascontiguousarray(array)
    payload = {
        "dtype": _dtype_descriptor(EVENT_INDEX_DTYPE),
        "shape": [int(value) for value in contiguous.shape],
        "num_bytes": int(contiguous.nbytes),
        "bytes_sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        **payload,
        "fingerprint_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def write_event_index_memmap(
    records: np.ndarray,
    path: str | Path,
    *,
    order_by: Sequence[object] | object,
    event_id_source: str,
    split: str | None = None,
) -> dict[str, object]:
    """Write a compact event-index memmap and return JSON-safe metadata."""
    array = _as_event_index_array(records)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapped = np.memmap(output_path, dtype=EVENT_INDEX_DTYPE, mode="w+", shape=array.shape)
    mapped[:] = array[:]
    mapped.flush()
    del mapped

    fingerprint = event_index_fingerprint(array)
    meta: dict[str, object] = {
        "row_order_is_stream_order": True,
        "order_by": _json_safe_order_by(order_by),
        "event_id_source": str(event_id_source),
        "num_events": int(array.shape[0]),
        "path": str(output_path),
        "event_index_dtype": _dtype_descriptor(EVENT_INDEX_DTYPE),
        "dtype_descriptor": _dtype_descriptor(EVENT_INDEX_DTYPE),
        "event_index_fingerprint": fingerprint,
        "fingerprint": fingerprint,
    }
    if split is not None:
        meta["split"] = str(split)
    return meta


def open_event_index_memmap(
    path: str | Path,
    *,
    num_events: int,
    mode: str = "r",
) -> np.memmap:
    """Open a compact Phase3E event-index memmap with the canonical dtype."""
    count = int(num_events)
    if count < 0:
        raise ValueError(f"num_events must be non-negative, got {count}")
    return np.memmap(Path(path), dtype=EVENT_INDEX_DTYPE, mode=mode, shape=(count,))


def _as_event_index_array(records: np.ndarray) -> np.ndarray:
    array = np.asarray(records)
    if array.dtype != EVENT_INDEX_DTYPE:
        raise TypeError(f"event index records must use EVENT_INDEX_DTYPE, got {array.dtype}")
    if array.ndim != 1:
        raise ValueError(f"event index records must be one-dimensional, got {array.ndim}")
    return array


def _dtype_descriptor(dtype: np.dtype) -> list[list[str]]:
    if dtype != EVENT_INDEX_DTYPE:
        raise ValueError(f"unsupported event index dtype descriptor: {dtype}")
    return [list(item) for item in EVENT_INDEX_DTYPE_DESCRIPTOR]


def _json_safe_order_by(order_by: Sequence[object] | object) -> list[str]:
    if isinstance(order_by, (str, bytes)):
        return [order_by.decode("utf-8") if isinstance(order_by, bytes) else order_by]
    try:
        return [str(value) for value in order_by]  # type: ignore[union-attr]
    except TypeError:
        return [str(order_by)]


def _required_value(row: Mapping[str, object], key: str) -> object:
    if key not in row:
        raise KeyError(f"missing required event index field {key}")
    return row[key]


def _reject_direct_index_fields(row: Mapping[str, object]) -> None:
    for field in _FORBIDDEN_ROW_INDEX_FIELDS:
        if field in row:
            raise ValueError(f"{field} must be generated by mapping, not supplied in rows")


def _required_strict_int(
    row: Mapping[str, object],
    key: str,
    dtype: type[np.integer],
    *,
    allow_negative: bool,
) -> int:
    value = _required_value(row, key)
    return _strict_int(key, value, dtype, allow_negative=allow_negative)


def _mapped_id(
    output_name: str,
    input_name: str,
    input_value: object,
    mapping: Mapping[object, int],
    dtype: type[np.integer],
) -> int:
    if input_value in mapping:
        mapped = _strict_int(output_name, mapping[input_value], dtype, allow_negative=False)
    else:
        raise KeyError(f"missing {input_name} in mapping: {input_value!r}")
    return mapped


def _strict_int(
    name: str,
    value: object,
    dtype: type[np.integer],
    *,
    allow_negative: bool,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a plain signed integer, got bool {value!r}")
    if isinstance(value, np.integer):
        integer = int(value)
    elif isinstance(value, int):
        integer = value
    elif isinstance(value, str):
        if not _PLAIN_SIGNED_INTEGER_RE.fullmatch(value):
            raise ValueError(f"{name} must be a plain signed integer, got {value!r}")
        integer = int(value)
    else:
        raise ValueError(f"{name} must be a plain signed integer, got {value!r}")
    if not allow_negative and integer < 0:
        raise ValueError(f"{name} must not be negative, got {integer}")
    info = np.iinfo(dtype)
    if integer < int(info.min) or integer > int(info.max):
        raise ValueError(
            f"{name}={integer} outside {np.dtype(dtype).name} range "
            f"[{int(info.min)}, {int(info.max)}]",
        )
    return integer
