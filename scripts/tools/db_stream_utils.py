#!/usr/bin/env python3
from __future__ import annotations

import os
import itertools
from typing import Any, Iterable

from cs4m.config.config import get_runtime_required_args, get_yml_cfg
from cs4m.config.provnet_utils import PROCESS_NODE_TYPE, datetime_to_ns_time_US
from scripts.data.get_dataset import (
    ORTHRUS10_EVENT_TYPES,
    NODE_TYPE_TOKENS,
    summarize_node_payload,
    tokenize_msg,
)
from cs4m.utils.common import information_flow


_STREAM_CURSOR_COUNTER = itertools.count()


def _cfg_for_dataset(dataset: str):
    cli = get_runtime_required_args(args=[dataset])
    return get_yml_cfg(cli)


def _day_bounds(year_month: str, day: int) -> tuple[int, int]:
    start = datetime_to_ns_time_US(f"{year_month}-{int(day):02d} 00:00:00")
    end = datetime_to_ns_time_US(f"{year_month}-{int(day) + 1:02d} 00:00:00")
    return int(start), int(end)


def _query_count(cur, year_month: str, days: list[int], event_filter: bool) -> int:
    total = 0
    for day in days:
        start_ns, end_ns = _day_bounds(year_month, day)
        if event_filter:
            cur.execute(
                """
                select count(*)
                from event_table
                where timestamp_rec >= %s and timestamp_rec < %s
                    and operation = any(%s)
                """,
                (start_ns, end_ns, list(ORTHRUS10_EVENT_TYPES)),
            )
        else:
            cur.execute(
                """
                select count(*)
                from event_table
                where timestamp_rec >= %s and timestamp_rec < %s
                """,
                (start_ns, end_ns),
            )
        total += int(cur.fetchone()[0])
    return int(total)


def _stream_events(
    conn,
    year_month: str,
    days: list[int],
    indexid2summary: dict[int, tuple[str, str]],
    hash2type: dict[str, str],
    hash2uuid_index: dict[str, tuple[str, int]],
    abnormal_nodes: set[int],
    event_filter: bool,
    fetch_size: int,
    max_events: int,
) -> Iterable[dict[str, Any]]:
    produced = 0
    next_event_index = 0
    for day in days:
        start_ns, end_ns = _day_bounds(year_month, day)
        name = f"tflr_light_day_{day}_{os.getpid()}_{next(_STREAM_CURSOR_COUNTER)}"
        cur = conn.cursor(name=name)
        try:
            cur.itersize = int(fetch_size)
            if event_filter:
                cur.execute(
                    """
                    select src_node, src_index_id, operation, dst_node, dst_index_id,
                        event_uuid, timestamp_rec, _id
                    from event_table
                    where timestamp_rec >= %s and timestamp_rec < %s
                        and operation = any(%s)
                    order by timestamp_rec, _id
                    """,
                    (start_ns, end_ns, list(ORTHRUS10_EVENT_TYPES)),
                )
            else:
                cur.execute(
                    """
                    select src_node, src_index_id, operation, dst_node, dst_index_id,
                        event_uuid, timestamp_rec, _id
                    from event_table
                    where timestamp_rec >= %s and timestamp_rec < %s
                    order by timestamp_rec, _id
                    """,
                    (start_ns, end_ns),
                )
            while True:
                rows = cur.fetchmany(int(fetch_size))
                if not rows:
                    break
                for src_hash, src_idx, op, dst_hash, dst_idx, event_uuid, ts, db_id in rows:
                    if int(max_events) > 0 and produced >= int(max_events):
                        return
                    try:
                        src_idx_int = int(src_idx)
                        dst_idx_int = int(dst_idx)
                    except (TypeError, ValueError):
                        continue
                    actor_hash = str(src_hash)
                    obj_hash = str(dst_hash)
                    actor_type = hash2type.get(actor_hash)
                    obj_type = hash2type.get(obj_hash)
                    if actor_type != PROCESS_NODE_TYPE and obj_type != PROCESS_NODE_TYPE:
                        continue
                    if actor_type != PROCESS_NODE_TYPE:
                        actor_hash, obj_hash = obj_hash, actor_hash
                        src_idx_int, dst_idx_int = dst_idx_int, src_idx_int
                        actor_type, obj_type = obj_type, actor_type
                    actor_idx = hash2uuid_index.get(actor_hash, (None, None))[1]
                    obj_idx = hash2uuid_index.get(obj_hash, (None, None))[1]
                    if actor_idx is None or obj_idx is None:
                        continue
                    actor_idx = int(actor_idx)
                    obj_idx = int(obj_idx)
                    src_bad = actor_idx in abnormal_nodes
                    dst_bad = obj_idx in abnormal_nodes
                    label = 2 if src_bad and dst_bad else (1 if src_bad or dst_bad else 0)
                    object_type = str(hash2type.get(obj_hash, "unknown"))
                    actor_summary = indexid2summary.get(actor_idx)
                    obj_summary = indexid2summary.get(obj_idx)
                    if actor_summary is None or obj_summary is None:
                        continue
                    actor_kind, actor_text = actor_summary
                    obj_kind, obj_text = obj_summary
                    action_text = str(op)
                    text = " ".join(
                        [
                            f"A_{NODE_TYPE_TOKENS.get(actor_kind, NODE_TYPE_TOKENS['unknown'])}",
                            actor_text,
                            action_text,
                            f"O_{NODE_TYPE_TOKENS.get(obj_kind, NODE_TYPE_TOKENS['unknown'])}",
                            obj_text,
                        ]
                    )
                    info_src, info_dst, rel, info_src_type, info_dst_type = information_flow(
                        actor_idx,
                        obj_idx,
                        action_text,
                        object_type,
                    )
                    yield {
                        "pos": next_event_index,
                        "event_index": next_event_index,
                        "event_id": int(db_id),
                        "timestamp_ns": int(ts),
                        "process_idx": actor_idx,
                        "src_idx": actor_idx,
                        "dst_idx": obj_idx,
                        "info_src": int(info_src),
                        "info_dst": int(info_dst),
                        "relation_id": int(rel),
                        "info_src_type": int(info_src_type),
                        "info_dst_type": int(info_dst_type),
                        "action": action_text,
                        "object_type": object_type,
                        "src_kind": actor_kind,
                        "dst_kind": obj_kind,
                        "src_summary": actor_text,
                        "dst_summary": obj_text,
                        "text": text,
                        "label": int(label),
                    }
                    next_event_index += 1
                    produced += 1
        finally:
            cur.close()


def _build_index_summaries(indexid2msg: dict[int, list[str]]) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for index_id, value in indexid2msg.items():
        if not value:
            continue
        node_type = str(value[0]).strip().lower() or "unknown"
        payload = " ".join(map(str, value[1:])).strip()
        summary = summarize_node_payload(node_type, tokenize_msg(payload))
        out[int(index_id)] = (node_type, " ".join(summary))
    return out
