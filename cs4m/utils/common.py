from __future__ import annotations

import math
import pickle
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ENTITY_TYPES = {"process": 0, "file": 1, "netflow": 2, "unknown": 3}
RELATIONS = {"read": 0, "recv": 1, "write": 2, "send": 3, "spawn": 4, "open": 5, "other": 6}


def stable_hash(text: str, seed: int = 0) -> int:
    # crc32 is deterministic and much faster than cryptographic hashes for
    # per-event streaming feature hashing. The seed is folded into the initial
    # CRC value to keep independent hash families for bucket/sign projections.
    payload = str(text).encode("utf-8", errors="ignore")
    return int(zlib.crc32(payload, int(seed) & 0xFFFFFFFF) & 0xFFFFFFFF)


def bucket(text: str, num_buckets: int, seed: int = 0) -> int:
    return stable_hash(text, seed=seed) % int(num_buckets)


def sign_hash(text: str, seed: int = 17) -> float:
    return 1.0 if (stable_hash(text, seed=seed) & 1) == 0 else -1.0


def load_pickle(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict from {path}, got {type(payload)!r}")
    return payload


def load_structured(path: str | Path) -> dict[str, Any]:
    payload = load_pickle(path)
    required = {"event_index", "timestamp_ns", "src_idx", "dst_idx", "action", "object_type", "text", "label"}
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    return payload


def as_array(payload: dict[str, Any], key: str, dtype=None) -> np.ndarray:
    arr = np.asarray(payload[key], dtype=dtype)
    return arr


def sorted_positions(payload: dict[str, Any], max_events: int = 0) -> np.ndarray:
    ts = as_array(payload, "timestamp_ns", np.int64)
    event_index = as_array(payload, "event_index", np.int64)
    order = np.lexsort((event_index, ts))
    if int(max_events) > 0:
        order = order[: int(max_events)]
    return order


def relation_from_action(action: object) -> int:
    text = str(action).upper()
    if text in {"EVENT_READ"}:
        return RELATIONS["read"]
    if text in {"EVENT_RECVFROM", "EVENT_RECVMSG", "EVENT_ACCEPT"}:
        return RELATIONS["recv"]
    if text in {"EVENT_WRITE"}:
        return RELATIONS["write"]
    if text in {"EVENT_SENDTO", "EVENT_SENDMSG", "EVENT_CONNECT"}:
        return RELATIONS["send"]
    if text in {"EVENT_EXECUTE", "EVENT_CLONE"}:
        return RELATIONS["spawn"]
    if text in {"EVENT_OPEN"}:
        return RELATIONS["open"]
    return RELATIONS["other"]


def type_id(value: object) -> int:
    return ENTITY_TYPES.get(str(value).strip().lower(), ENTITY_TYPES["unknown"])


def information_flow(
    src_idx: int,
    dst_idx: int,
    action: object,
    object_type: object,
) -> tuple[int, int, int, int, int]:
    """Return info_src, info_dst, relation_id, info_src_type, info_dst_type."""
    rel = relation_from_action(action)
    obj_type = type_id(object_type)
    proc_type = ENTITY_TYPES["process"]
    if rel in {RELATIONS["read"], RELATIONS["recv"]}:
        return int(dst_idx), int(src_idx), rel, obj_type, proc_type
    if rel == RELATIONS["open"]:
        return int(src_idx), int(src_idx), rel, proc_type, proc_type
    return int(src_idx), int(dst_idx), rel, proc_type, obj_type


def log1p_seconds(now_ns: int, prev_ns: int) -> float:
    if int(prev_ns) < 0:
        return 0.0
    return float(math.log1p(max(int(now_ns) - int(prev_ns), 0) / 1_000_000_000.0))


def one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros((int(size),), dtype=np.float32)
    if 0 <= int(index) < int(size):
        out[int(index)] = 1.0
    return out


def robust_stats(values: np.ndarray, labels: np.ndarray | None = None) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(x)
    if labels is not None:
        valid &= np.asarray(labels, dtype=np.int64) == 0
    data = x[valid]
    if data.size <= 0:
        data = x[np.isfinite(x)]
    if data.size <= 0:
        return {"median": 0.0, "mad": 1.0, "q95": 0.0, "q99": 0.0, "q995": 0.0, "q999": 0.0}
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)) + 1e-6)
    return {
        "median": median,
        "mad": mad,
        "q95": float(np.quantile(data, 0.95)),
        "q99": float(np.quantile(data, 0.99)),
        "q995": float(np.quantile(data, 0.995)),
        "q999": float(np.quantile(data, 0.999)) if data.size >= 1000 else float(np.max(data)),
    }


def upper_z(values: np.ndarray, stats: dict[str, float], clip: float = 12.0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    scale = max(float(stats.get("mad", 1.0)) * 1.4826, 1e-6)
    out = np.maximum((x - float(stats.get("median", 0.0))) / scale, 0.0)
    out[~np.isfinite(x)] = np.nan
    if float(clip) > 0:
        out = np.clip(out, 0.0, float(clip))
    return out.astype(np.float32, copy=False)


def iter_event_rows(payload: dict[str, Any], positions: Iterable[int]):
    event_index = as_array(payload, "event_index", np.int64)
    timestamp_ns = as_array(payload, "timestamp_ns", np.int64)
    process_idx = np.asarray(payload.get("process_idx", np.full(event_index.shape, -1)), dtype=np.int64)
    src_idx = as_array(payload, "src_idx", np.int64)
    dst_idx = as_array(payload, "dst_idx", np.int64)
    action = as_array(payload, "action", object)
    object_type = as_array(payload, "object_type", object)
    text = as_array(payload, "text", object)
    label = as_array(payload, "label", np.int64)
    for pos in positions:
        i = int(pos)
        info_src, info_dst, rel, info_src_type, info_dst_type = information_flow(
            int(src_idx[i]), int(dst_idx[i]), action[i], object_type[i]
        )
        yield {
            "pos": i,
            "event_index": int(event_index[i]),
            "timestamp_ns": int(timestamp_ns[i]),
            "process_idx": int(process_idx[i]),
            "src_idx": int(src_idx[i]),
            "dst_idx": int(dst_idx[i]),
            "info_src": int(info_src),
            "info_dst": int(info_dst),
            "relation_id": int(rel),
            "info_src_type": int(info_src_type),
            "info_dst_type": int(info_dst_type),
            "action": str(action[i]),
            "object_type": str(object_type[i]),
            "text": str(text[i]),
            "label": int(label[i]),
        }
