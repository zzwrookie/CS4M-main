"""Dataset split, DB stream preparation, and residual embedding preflight helpers."""

from __future__ import annotations

from cs4m.semantics.optc_windows import is_optc_dataset
from scripts.pipeline.config.runtime_config import *


def _load_process_config(config: SlimConfig) -> ProcessSemanticConfig:
    process_cfg = load_process_semantic_config(config.process_semantics_config)
    if str(config.dataset).upper().startswith("SYNTHETIC"):
        return ProcessSemanticConfig(rules_version="legacy")
    return process_cfg


def _slim_split_override_days(dataset: str) -> dict[str, list[int]] | None:
    value = SLIM_SPLIT_OVERRIDES.get(str(dataset))
    if value is None:
        return None
    return {
        "train": [int(day) for day in value["train"]],
        "val": [int(day) for day in value["val"]],
        "test": [int(day) for day in value["test"]],
    }


def _resolve_slim_split_metadata(
    dataset: str,
    year_month: str,
    train_days: Sequence[int],
    validation_days: Sequence[int],
    test_days: Sequence[int],
    slim_split_override: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "split_source": "config",
        "slim_split_override": bool(slim_split_override),
        "slim_split_override_applied": False,
        "year_month": str(year_month),
        "train_days": _int_list(train_days),
        "validation_days": _int_list(validation_days),
        "test_days": _int_list(test_days),
    }
    if not slim_split_override:
        return metadata

    override = SLIM_SPLIT_OVERRIDES.get(str(dataset))
    if override is None:
        metadata["split_override_reason"] = "no_slim_split_override_for_dataset"
        return metadata

    metadata.update(
        {
            "split_source": "slim_split_override",
            "slim_split_override_applied": True,
            "year_month": str(override.get("year_month", year_month)),
            "train_days": _int_list(override["train"]),
            "validation_days": _int_list(override["val"]),
            "test_days": _int_list(override["test"]),
        },
    )
    return metadata


def _resolve_db_split_metadata(config: SlimConfig, db_cfg: Any) -> dict[str, Any]:
    year_month = str(db_cfg.dataset.year_month)
    train_days = parse_split_days(get_dataset_splits(db_cfg, "train"))
    validation_days = parse_split_days(get_dataset_splits(db_cfg, "val"))
    test_days = parse_split_days(get_dataset_splits(db_cfg, "test"))
    return _resolve_slim_split_metadata(
        config.dataset,
        year_month,
        train_days,
        validation_days,
        test_days,
        config.slim_split_override,
    )


def build_node_maps(cur) -> dict[str, Any]:
    """Build DB node summaries and lookup maps without reading event labels."""
    netflow_nodes, process_nodes, file_nodes = fetch_node_tables(cur)
    indexid2msg: dict[int, list[str]] = {}
    netflow_meta: dict[int, dict[str, str]] = {}
    process_meta: dict[int, dict[str, str]] = {}
    file_meta: dict[int, dict[str, str]] = {}
    for _uuid, _hash_id, src_addr, src_port, dst_addr, dst_port, index_id in netflow_nodes:
        index_id_int = int(index_id)
        indexid2msg[index_id_int] = [
            "netflow",
            str(src_addr),
            str(src_port),
            str(dst_addr),
            str(dst_port),
        ]
        netflow_meta[index_id_int] = {
            "src_addr": str(src_addr),
            "src_port": str(src_port),
            "dst_addr": str(dst_addr),
            "dst_port": str(dst_port),
        }
    for _uuid, _hash_id, *payload, index_id in process_nodes:
        index_id_int = int(index_id)
        payload_text = [str(value or "") for value in payload]
        path_text = payload_text[0] if len(payload_text) >= 1 else ""
        cmd_text = payload_text[1] if len(payload_text) >= 2 else ""
        indexid2msg[index_id_int] = ["process", *payload_text]
        process_meta[index_id_int] = {"path": path_text, "cmd": cmd_text}
    for _uuid, _hash_id, path, index_id in file_nodes:
        index_id_int = int(index_id)
        indexid2msg[index_id_int] = ["file", str(path)]
        file_meta[index_id_int] = {"path": str(path)}
    return {
        "indexid2summary": _build_index_summaries(indexid2msg),
        "hash2type": build_hash_to_type(netflow_nodes, process_nodes, file_nodes),
        "hash2uuid_index": build_hash_uuid_index_map(netflow_nodes, process_nodes, file_nodes),
        "uuid2index": build_uuid_index_map(netflow_nodes, process_nodes, file_nodes),
        "netflow_meta": netflow_meta,
        "process_meta": process_meta,
        "file_meta": file_meta,
    }


OPTC_ECAR_TO_ORTHRUS10_ACTION = {
    "OPEN": "EVENT_OPEN",
    "READ": "EVENT_READ",
    "WRITE": "EVENT_WRITE",
    "MODIFY": "EVENT_WRITE",
    "CREATE": "EVENT_WRITE",
    "RENAME": "EVENT_WRITE",
    "DELETE": "EVENT_WRITE",
    "START": "EVENT_EXECUTE",
    "MESSAGE": "EVENT_SENDMSG",
    "TERMINATE": "EVENT_CLONE",
}


def optc_ecar_to_orthrus10_action(action: object) -> str:
    """Map OpTC eCAR action names into the existing ORTHRUS10 action space."""
    text = str(action or "").strip().upper()
    if text in ORTHRUS10_EVENT_TYPES:
        return text
    return OPTC_ECAR_TO_ORTHRUS10_ACTION.get(text, "EVENT_OPEN")


def _coerce_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ground_truth_indices_from_rows(
    rows: Sequence[Sequence[Any]],
    uuid2index: Mapping[str, int],
) -> set[int]:
    indices: set[int] = set()
    for row in rows:
        if not row:
            continue
        uuid = str(row[0]).strip()
        if not uuid or uuid.lower() == "uuid":
            continue
        mapped = uuid2index.get(uuid)
        if mapped is None and len(row) >= 3:
            mapped = _coerce_optional_int(row[2])
        if mapped is not None:
            indices.add(int(mapped))
    return indices


def load_ground_truth_indices_readonly(cfg, uuid2index: Mapping[str, int]) -> set[int]:
    """Read GT node ids for post-stream evaluation without mutating GT files."""
    indices: set[int] = set()
    for rel_path in get_ground_truth_paths(cfg):
        path = Path(resolve_gt_path(rel_path))
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            first_line = handle.readline()
            delimiter = "\t" if "\t" in first_line else ","
            handle.seek(0)
            reader = csv.reader(handle, delimiter=delimiter)
            indices.update(_ground_truth_indices_from_rows(list(reader), uuid2index))
    return indices


def enrich_row_with_netflow(
    row: Mapping[str, Any],
    netflow_meta: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach exact netflow destination fields for residual text, else blank them."""
    enriched = dict(row)
    dst_kind = normalize_piece(enriched.get("dst_kind", enriched.get("object_type", "")))
    object_type = normalize_piece(enriched.get("object_type", ""))
    if dst_kind == "netflow" or object_type == "netflow":
        meta = netflow_meta.get(int(enriched.get("dst_idx", -1)), {})
        enriched["dst_addr"] = str(meta.get("dst_addr", ""))
        enriched["dst_port"] = str(meta.get("dst_port", ""))
        return enriched
    enriched["dst_addr"] = ""
    enriched["dst_port"] = ""
    return enriched


def enrich_row_with_process_meta(
    row: Mapping[str, Any],
    process_meta: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach process path/cmd metadata for process endpoints without labels."""
    enriched = dict(row)
    for side in ("src", "dst"):
        kind = normalize_piece(enriched.get(f"{side}_kind", ""))
        if kind != "process":
            enriched[f"{side}_process_path"] = ""
            enriched[f"{side}_process_cmd"] = ""
            continue
        try:
            node_id = int(enriched.get(f"{side}_idx", -1))
        except (TypeError, ValueError):
            node_id = -1
        meta = process_meta.get(node_id, {})
        enriched[f"{side}_process_path"] = str(meta.get("path", ""))
        enriched[f"{side}_process_cmd"] = str(meta.get("cmd", ""))
    return enriched


def enrich_row_with_file_meta(
    row: Mapping[str, Any],
    file_meta: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Attach raw file path metadata for file endpoints without labels."""
    enriched = dict(row)
    for side in ("src", "dst"):
        kind = normalize_piece(enriched.get(f"{side}_kind", ""))
        if kind != "file":
            enriched[f"{side}_file_path"] = ""
            continue
        try:
            node_id = int(enriched.get(f"{side}_idx", -1))
        except (TypeError, ValueError):
            node_id = -1
        meta = file_meta.get(node_id, {})
        enriched[f"{side}_file_path"] = str(meta.get("path", ""))
    return enriched


def _db_stream_status(
    requested: str,
    actual: str | None = None,
    fallback_reason: str = "",
) -> dict[str, str]:
    """Return DB stream mode metadata for eval JSON output."""
    actual_text = str(requested if actual is None else actual)
    reason_text = str(fallback_reason)
    if actual is not None and actual_text not in DB_STREAM_ACTUAL_MODES and not reason_text:
        reason_text = actual_text
        actual_text = str(requested)
    return {
        "db_stream_mode_requested": str(requested),
        "db_stream_mode_actual": actual_text,
        "db_stream_mode_fallback_reason": reason_text,
        "db_stream_mode": actual_text,
        "db_stream_fallback_reason": reason_text,
    }


def _publish_db_stream_status(
    target: dict[str, str] | None,
    requested: str,
    actual: str,
    fallback_reason: str = "",
) -> None:
    if target is None:
        return
    target.clear()
    target.update(_db_stream_status(requested, actual, fallback_reason))


def _short_db_stream_failure(exc: Exception) -> str:
    return str(exc)[:200]


def _rollback_after_db_stream_failure(conn) -> None:
    rollback = getattr(conn, "rollback", None)
    if rollback is None:
        return
    try:
        rollback()
    except Exception:
        return


def _stream_events_optc_index(
    conn,
    year_month: str,
    days: Sequence[int],
    node_maps: Mapping[str, Any],
    fetch_size: int,
    max_events: int,
    abnormal_nodes: set[int],
):
    """Stream OpTC rows by src/dst index ids and map eCAR actions into ORTHRUS10."""
    produced = 0
    next_event_index = 0
    indexid2summary = node_maps["indexid2summary"]
    for day in [int(value) for value in days]:
        start_ns, end_ns = _day_bounds(year_month, day)
        name = f"optc_index_stream_{os.getpid()}_{day}_{time.monotonic_ns()}"
        cur = conn.cursor(name=name)
        try:
            cur.itersize = int(fetch_size)
            cur.execute(
                """
                select src_index_id, operation, dst_index_id, event_uuid,
                    timestamp_rec, _id
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
                for src_idx, op, dst_idx, event_uuid, ts, db_id in rows:
                    if int(max_events) > 0 and produced >= int(max_events):
                        return
                    try:
                        src_idx_int = int(src_idx)
                        dst_idx_int = int(dst_idx)
                    except (TypeError, ValueError):
                        continue
                    src_summary = indexid2summary.get(src_idx_int)
                    dst_summary = indexid2summary.get(dst_idx_int)
                    if src_summary is None or dst_summary is None:
                        continue
                    actor_idx = src_idx_int
                    obj_idx = dst_idx_int
                    actor_kind, actor_text = src_summary
                    obj_kind, obj_text = dst_summary
                    if actor_kind != PROCESS_NODE_TYPE and obj_kind != PROCESS_NODE_TYPE:
                        continue
                    if actor_kind != PROCESS_NODE_TYPE:
                        actor_idx, obj_idx = obj_idx, actor_idx
                        actor_kind, obj_kind = obj_kind, actor_kind
                        actor_text, obj_text = obj_text, actor_text
                    action_text = optc_ecar_to_orthrus10_action(op)
                    object_type = str(obj_kind or "unknown")
                    info_src, info_dst, rel, info_src_type, info_dst_type = information_flow(
                        int(actor_idx),
                        int(obj_idx),
                        action_text,
                        object_type,
                    )
                    src_bad = int(actor_idx) in abnormal_nodes
                    dst_bad = int(obj_idx) in abnormal_nodes
                    row: dict[str, Any] = {
                        "pos": int(next_event_index),
                        "event_index": int(next_event_index),
                        "event_id": int(db_id),
                        "timestamp_ns": int(ts),
                        "process_idx": int(actor_idx),
                        "src_idx": int(actor_idx),
                        "dst_idx": int(obj_idx),
                        "info_src": int(info_src),
                        "info_dst": int(info_dst),
                        "relation_id": int(rel),
                        "info_src_type": int(info_src_type),
                        "info_dst_type": int(info_dst_type),
                        "action": action_text,
                        "object_type": object_type,
                        "src_kind": str(actor_kind),
                        "dst_kind": str(obj_kind),
                        "src_summary": str(actor_text),
                        "dst_summary": str(obj_text),
                        "text": " ".join(
                            (
                                f"A_{NODE_TYPE_TOKENS.get(str(actor_kind), 'UNK')}",
                                str(actor_text),
                                action_text,
                                f"O_{NODE_TYPE_TOKENS.get(str(obj_kind), 'UNK')}",
                                str(obj_text),
                            ),
                        ),
                        "label": 2 if src_bad and dst_bad else (1 if src_bad or dst_bad else 0),
                    }
                    yield row
                    produced += 1
                    next_event_index += 1
        finally:
            cur.close()


def stream_dataset_rows(
    conn,
    year_month: str,
    days: Sequence[int],
    node_maps: Mapping[str, Any],
    event_filter: bool,
    fetch_size: int,
    max_events: int,
    abnormal_nodes: set[int] | None = None,
    optc_action_mode: bool = False,
):
    """Yield DB stream rows with labels disabled by default and netflow metadata enriched."""
    safe_abnormal_nodes = set() if abnormal_nodes is None else set(abnormal_nodes)
    include_labels = bool(safe_abnormal_nodes)
    if bool(optc_action_mode):
        rows = _stream_events_optc_index(
            conn,
            year_month,
            [int(day) for day in days],
            node_maps,
            int(fetch_size),
            int(max_events),
            safe_abnormal_nodes,
        )
    else:
        rows = _stream_events(
            conn,
            year_month,
            [int(day) for day in days],
            node_maps["indexid2summary"],
            node_maps["hash2type"],
            node_maps["hash2uuid_index"],
            safe_abnormal_nodes,
            bool(event_filter),
            int(fetch_size),
            int(max_events),
        )
    for row in rows:
        enriched = enrich_row_with_netflow(row, node_maps["netflow_meta"])
        enriched = enrich_row_with_process_meta(
            enriched,
            node_maps.get("process_meta", {}),
        )
        enriched = enrich_row_with_file_meta(
            enriched,
            node_maps.get("file_meta", {}),
        )
        if not include_labels:
            enriched.pop("label", None)
        yield enriched


def stream_dataset_rows_slim(
    conn,
    year_month: str,
    days: Sequence[int],
    node_maps: Mapping[str, Any],
    event_filter: bool,
    fetch_size: int,
    max_events: int,
    db_stream_mode: str,
    abnormal_nodes: set[int] | None = None,
    stream_status: dict[str, str] | None = None,
    optc_action_mode: bool = False,
):
    """Select the slim DB test stream implementation and record actual mode."""
    requested_mode = str(db_stream_mode)
    if requested_mode not in DB_STREAM_MODES:
        raise ValueError(f"unknown db_stream_mode: {requested_mode}")

    if requested_mode == "python_lookup":
        _publish_db_stream_status(stream_status, requested_mode, "python_lookup")
        return stream_dataset_rows(
            conn,
            year_month,
            days,
            node_maps,
            event_filter,
            fetch_size,
            max_events,
            abnormal_nodes=abnormal_nodes,
            optc_action_mode=optc_action_mode,
        )
    if bool(optc_action_mode):
        _publish_db_stream_status(stream_status, requested_mode, "python_lookup")
        return stream_dataset_rows(
            conn,
            year_month,
            days,
            node_maps,
            event_filter,
            fetch_size,
            max_events,
            abnormal_nodes=abnormal_nodes,
            optc_action_mode=True,
        )

    try:
        temp_stream = _stream_events_slim_temp_table(
            conn,
            year_month,
            days,
            node_maps,
            event_filter,
            fetch_size,
            max_events,
            abnormal_nodes=abnormal_nodes,
            optc_action_mode=optc_action_mode,
        )
    except SlimTempStreamError as exc:
        if requested_mode == "temp_table":
            _publish_db_stream_status(stream_status, requested_mode, "temp_table")
            raise
        reason = _short_db_stream_failure(exc)
        _rollback_after_db_stream_failure(conn)
        _publish_db_stream_status(
            stream_status,
            requested_mode,
            "python_lookup_fallback",
            reason,
        )
        return stream_dataset_rows(
            conn,
            year_month,
            days,
            node_maps,
            event_filter,
            fetch_size,
            max_events,
            abnormal_nodes=abnormal_nodes,
        )

    _publish_db_stream_status(stream_status, requested_mode, "temp_table")
    return temp_stream


@dataclass
class Phase3EStreamEvent:
    """Compact label-free DB event row used only during streaming precompute."""

    event_id: int
    src_node_id: int
    dst_node_id: int
    operation: str
    event_uuid: str
    timestamp_rec: int


def _phase3e_detect_event_id_column(conn: Any) -> str:
    """Return the physical event id column used by this database."""
    with conn.cursor() as cur:
        cur.execute(
            """
            select column_name
            from information_schema.columns
            where table_name = 'event_table'
                and column_name in ('id', '_id')
            """,
        )
        columns = {str(row[0]) for row in cur.fetchall()}
    if "id" in columns:
        return "id"
    if "_id" in columns:
        return "_id"
    raise RuntimeError("event_table must contain either id or _id for Phase3E event_id")


def _phase3e_stream_split_events_direct(
    conn: Any,
    *,
    year_month: str,
    days: Sequence[int],
    event_filter: bool,
    fetch_size: int,
    max_events: int,
    event_id_column: str,
    config: SlimConfig | None = None,
    split: str = "unknown",
):
    """Stream raw event table rows in stable Phase3E order without node table joins."""
    optc_action_mode = config is not None and is_optc_dataset(config.dataset)

    def log(stage: str, **values: object) -> None:
        if config is None:
            return
        _stage_log(config, f"phase3e_pass1_{split}_{stage}", **values)

    day_sql, day_params = _slim_temp_day_filter(year_month, days)
    if not day_params:
        return iter(())
    operation_sql = ""
    params: list[Any] = [*day_params]
    if event_filter and not optc_action_mode:
        operation_sql = "and e.operation = any(%s)"
        params.append(list(ORTHRUS10_EVENT_TYPES))
    limit_sql = ""
    if int(max_events) > 0:
        limit_sql = "limit %s"
        params.append(int(max_events))
    name = f"phase3e_events_{os.getpid()}_{time.monotonic_ns()}"
    query = f"""
        select
            e.src_index_id,
            e.operation,
            e.dst_index_id,
            e.event_uuid,
            e.timestamp_rec,
            e.{event_id_column}
        from event_table e
        where {day_sql}
            {operation_sql}
        order by e.timestamp_rec, e.event_uuid
        {limit_sql}
        """
    order_by = "e.timestamp_rec, e.event_uuid"
    log(
        "query_prepare_start",
        split=split,
        operation_filter=bool(event_filter),
        order_by=order_by,
        max_events=int(max_events),
        fetch_size=int(fetch_size),
    )
    log(
        "query_sql",
        sql=" ".join(query.split()),
        params_count=len(params),
        days=",".join(str(int(day)) for day in days),
    )
    log("cursor_open_start", cursor_name=name)
    cursor_start = time.monotonic()
    cur = conn.cursor(name=name)
    cur.itersize = int(fetch_size)
    cur.execute(query, tuple(params))
    cursor_seconds = time.monotonic() - cursor_start
    log("cursor_open_end", cursor_open_seconds=f"{cursor_seconds:.3f}")

    def generator():
        try:
            produced = 0
            first_fetch_done = False
            first_fetch_wait_start = time.monotonic()
            while True:
                if not first_fetch_done:
                    log("first_fetch_start")
                fetch_start = time.monotonic()
                wait_stop = threading.Event()

                def heartbeat() -> None:
                    while not wait_stop.wait(300.0):
                        elapsed = time.monotonic() - first_fetch_wait_start
                        log(
                            "waiting_for_first_fetch",
                            elapsed_seconds=f"{elapsed:.3f}",
                        )

                heartbeat_thread: threading.Thread | None = None
                if not first_fetch_done:
                    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
                    heartbeat_thread.start()
                rows = cur.fetchmany(int(fetch_size))
                wait_stop.set()
                if heartbeat_thread is not None:
                    heartbeat_thread.join(timeout=0.1)
                fetch_seconds = time.monotonic() - fetch_start
                if not first_fetch_done:
                    log(
                        "first_fetch_end",
                        first_fetch_seconds=f"{fetch_seconds:.3f}",
                        row_count=len(rows),
                    )
                    first_fetch_done = True
                if not rows:
                    break
                for row in rows:
                    src_idx, op, dst_idx, event_uuid, timestamp_rec, event_id = row
                    if produced == 0:
                        log(
                            "first_row_seen",
                            event_id=int(event_id),
                            operation=str(op),
                            timestamp_rec=int(timestamp_rec),
                        )
                    yield Phase3EStreamEvent(
                        event_id=int(event_id),
                        src_node_id=int(src_idx),
                        dst_node_id=int(dst_idx),
                        operation=(
                            optc_ecar_to_orthrus10_action(op)
                            if optc_action_mode
                            else str(op)
                        ),
                        event_uuid=str(event_uuid),
                        timestamp_rec=int(timestamp_rec),
                    )
                    produced += 1
                    if int(max_events) > 0 and produced >= int(max_events):
                        return
        finally:
            cur.close()

    return generator()


def _phase3e_scan_used_nodes_pass1(
    *,
    conn: Any,
    year_month: str,
    split_days: Mapping[str, Sequence[int]],
    event_filter: bool,
    fetch_size: int,
    max_events_by_split: Mapping[str, int],
    event_id_column: str,
    config: SlimConfig,
) -> dict[str, Any]:
    """Pass 1: stream events once to collect used node ids and split counts."""
    used_node_ids: set[int] = set()
    split_used_nodes: dict[str, set[int]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    split_counts: dict[str, int] = {}
    progress_interval = int(config.progress_interval_events)
    for split in ("train", "validation", "test"):
        count = 0
        _stage_log(config, f"phase3e_pass1_{split}_scan_start")
        for event in _phase3e_stream_split_events_direct(
            conn,
            year_month=year_month,
            days=split_days.get(split, ()),
            event_filter=event_filter,
            fetch_size=int(fetch_size),
            max_events=int(max_events_by_split.get(split, 0)),
            event_id_column=event_id_column,
            config=config,
            split=split,
        ):
            used_node_ids.add(int(event.src_node_id))
            used_node_ids.add(int(event.dst_node_id))
            split_used_nodes[split].add(int(event.src_node_id))
            split_used_nodes[split].add(int(event.dst_node_id))
            count += 1
            if progress_interval > 0 and count % progress_interval == 0:
                _stage_log(config, f"phase3e_pass1_{split}_scan_progress", count=count)
        split_counts[split] = int(count)
        _stage_log(
            config,
            f"phase3e_pass1_{split}_scan_end",
            count=count,
            used_nodes=len(split_used_nodes[split]),
        )
    return {
        "used_node_ids": used_node_ids,
        "split_used_nodes": split_used_nodes,
        "split_counts": split_counts,
    }


def run_phase3e_precompute_db(*_args: object, **_kwargs: object) -> Path:
    """Legacy Phase3E precompute entrypoint disabled in the active pipeline."""
    _phase3e_require_legacy_precompute_path_disabled()



def _bounded_stream_node_hashes(
    conn,
    year_month: str,
    days: Sequence[int],
    event_filter: bool,
    max_events: int,
) -> set[str] | None:
    """Return node hashes touched by the bounded stream, or None for unbounded full scans."""
    if int(max_events) <= 0:
        return None
    day_sql, day_params = _slim_temp_day_filter(year_month, days)
    if not day_params:
        return set()
    branch_params: list[Any] = [*day_params]
    event_filter_sql = ""
    if event_filter:
        event_filter_sql = "and e.operation = any(%s)"
        branch_params.append(list(ORTHRUS10_EVENT_TYPES))
    params: list[Any] = [
        *branch_params,
        int(max_events),
        *branch_params,
        int(max_events),
    ]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select distinct node_hash
            from (
                select bounded_events.src_node::text as node_hash
                from (
                    select e.src_node, e.dst_node
                    from event_table e
                    where {day_sql}
                        {event_filter_sql}
                    order by e.timestamp_rec, e._id
                    limit %s
                ) bounded_events
                union all
                select bounded_events.dst_node::text as node_hash
                from (
                    select e.src_node, e.dst_node
                    from event_table e
                    where {day_sql}
                        {event_filter_sql}
                    order by e.timestamp_rec, e._id
                    limit %s
                ) bounded_events
            ) bounded_node_hashes
            """,
            tuple(params),
        )
        return {str(row[0]) for row in cur.fetchall()}


def _prepare_slim_node_lookup_temp_table(
    conn,
    node_maps: Mapping[str, Any],
    allowed_hashes: set[str] | None = None,
) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("drop table if exists pg_temp.temp_slim_node_lookup")
            cur.execute(
                """
                create temp table temp_slim_node_lookup (
                    hash_id text primary key,
                    index_id bigint not null,
                    node_type text not null,
                    node_kind text not null,
                    node_summary text not null,
                    dst_addr text not null,
                    dst_port text not null,
                    process_path text not null,
                    process_cmd text not null,
                    file_path text not null
                ) on commit preserve rows
                """,
            )
            cur.executemany(
                """
                insert into temp_slim_node_lookup(
                    hash_id,
                    index_id,
                    node_type,
                    node_kind,
                    node_summary,
                    dst_addr,
                    dst_port,
                    process_path,
                    process_cmd,
                    file_path
                ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                _slim_temp_lookup_rows(node_maps, allowed_hashes),
            )
            cur.execute("analyze temp_slim_node_lookup")
    except DB_EXCEPTION_TYPES as exc:
        raise SlimTempStreamError(f"temp table setup failed: {exc}") from exc


def _slim_temp_lookup_rows(
    node_maps: Mapping[str, Any],
    allowed_hashes: set[str] | None = None,
):
    indexid2summary = node_maps["indexid2summary"]
    hash2type = node_maps["hash2type"]
    hash2uuid_index = node_maps["hash2uuid_index"]
    netflow_meta = node_maps.get("netflow_meta", {})
    process_meta = node_maps.get("process_meta", {})
    file_meta = node_maps.get("file_meta", {})
    for hash_id, value in hash2uuid_index.items():
        hash_text = str(hash_id)
        if allowed_hashes is not None and hash_text not in allowed_hashes:
            continue
        index_id = int(value[1])
        node_type = str(hash2type.get(hash_text, "unknown"))
        node_kind, node_summary = indexid2summary.get(index_id, (node_type, ""))
        netflow = netflow_meta.get(index_id, {})
        process = process_meta.get(index_id, {})
        file_node = file_meta.get(index_id, {})
        yield (
            hash_text,
            index_id,
            node_type,
            str(node_kind),
            str(node_summary),
            str(netflow.get("dst_addr", "")),
            str(netflow.get("dst_port", "")),
            str(process.get("path", "")),
            str(process.get("cmd", "")),
            str(file_node.get("path", "")),
        )


def _stream_events_slim_temp_table(
    conn,
    year_month: str,
    days: Sequence[int],
    node_maps: Mapping[str, Any],
    event_filter: bool,
    fetch_size: int,
    max_events: int,
    abnormal_nodes: set[int] | None = None,
):
    """Prepare session-local TEMP lookup tables and return compact event rows."""
    safe_abnormal_nodes = set() if abnormal_nodes is None else set(abnormal_nodes)
    allowed_hashes = _bounded_stream_node_hashes(
        conn,
        year_month,
        days,
        event_filter,
        max_events,
    )
    _prepare_slim_node_lookup_temp_table(conn, node_maps, allowed_hashes)
    cursor = _open_slim_temp_stream_cursor(
        conn,
        year_month,
        [int(day) for day in days],
        bool(event_filter),
        int(fetch_size),
    )
    return SlimTempTableStream(
        cursor=cursor,
        fetch_size=int(fetch_size),
        max_events=int(max_events),
        abnormal_nodes=safe_abnormal_nodes,
    )


def _open_slim_temp_stream_cursor(
    conn,
    year_month: str,
    days: Sequence[int],
    event_filter: bool,
    fetch_size: int,
):
    day_sql, day_params = _slim_temp_day_filter(year_month, days)
    params: list[Any] = [
        *day_params,
        PROCESS_NODE_TYPE,
        PROCESS_NODE_TYPE,
    ]
    event_filter_sql = ""
    if event_filter:
        event_filter_sql = "and e.operation = any(%s)"
        params.append(list(ORTHRUS10_EVENT_TYPES))
    name = f"causal_slim_temp_{os.getpid()}_{time.monotonic_ns()}"
    cur = None
    try:
        cur = conn.cursor(name=name)
        cur.itersize = int(fetch_size)
        cur.execute(
            f"""
            select src_lookup.index_id, src_lookup.node_type,
                src_lookup.node_kind, src_lookup.node_summary,
                src_lookup.dst_addr, src_lookup.dst_port,
                src_lookup.process_path, src_lookup.process_cmd,
                src_lookup.file_path,
                e.operation,
                dst_lookup.index_id, dst_lookup.node_type,
                dst_lookup.node_kind, dst_lookup.node_summary,
                dst_lookup.dst_addr, dst_lookup.dst_port,
                dst_lookup.process_path, dst_lookup.process_cmd,
                dst_lookup.file_path,
                e.event_uuid, e.timestamp_rec, e._id
            from event_table e
            join temp_slim_node_lookup src_lookup
                on src_lookup.hash_id = e.src_node::text
            join temp_slim_node_lookup dst_lookup
                on dst_lookup.hash_id = e.dst_node::text
            where {day_sql}
                and (
                    src_lookup.node_type = %s
                    or dst_lookup.node_type = %s
                )
                {event_filter_sql}
            order by e.timestamp_rec, e._id
            """,
            tuple(params),
        )
        return cur
    except DB_EXCEPTION_TYPES as exc:
        if cur is not None:
            cur.close()
        raise SlimTempStreamError(f"temp table stream query failed: {exc}") from exc


def _slim_temp_day_filter(year_month: str, days: Sequence[int]) -> tuple[str, list[int]]:
    if not days:
        return "false", []
    clauses: list[str] = []
    params: list[int] = []
    for day in days:
        start_ns, end_ns = _day_bounds(year_month, int(day))
        clauses.append("(e.timestamp_rec >= %s and e.timestamp_rec < %s)")
        params.extend([start_ns, end_ns])
    return f"({' or '.join(clauses)})", params


def _compact_temp_event_row(
    row: Sequence[Any],
    event_index: int,
    abnormal_nodes: set[int],
    optc_action_mode: bool = False,
) -> dict[str, Any] | None:
    event_id: int | None = None
    if len(row) == 15:
        (
            src_idx,
            src_type,
            src_kind,
            src_summary,
            src_addr,
            src_port,
            op,
            dst_idx,
            dst_type,
            dst_kind,
            dst_summary,
            dst_addr,
            dst_port,
            _event_uuid,
            ts,
        ) = row
        src_process_path = ""
        src_process_cmd = ""
        dst_process_path = ""
        dst_process_cmd = ""
        src_file_path = ""
        dst_file_path = ""
    elif len(row) == 16:
        (
            src_idx,
            src_type,
            src_kind,
            src_summary,
            src_addr,
            src_port,
            op,
            dst_idx,
            dst_type,
            dst_kind,
            dst_summary,
            dst_addr,
            dst_port,
            _event_uuid,
            ts,
            event_id,
        ) = row
        src_process_path = ""
        src_process_cmd = ""
        dst_process_path = ""
        dst_process_cmd = ""
        src_file_path = ""
        dst_file_path = ""
    elif len(row) == 19:
        (
            src_idx,
            src_type,
            src_kind,
            src_summary,
            src_addr,
            src_port,
            src_process_path,
            src_process_cmd,
            op,
            dst_idx,
            dst_type,
            dst_kind,
            dst_summary,
            dst_addr,
            dst_port,
            dst_process_path,
            dst_process_cmd,
            _event_uuid,
            ts,
        ) = row
        src_file_path = ""
        dst_file_path = ""
    elif len(row) == 20:
        (
            src_idx,
            src_type,
            src_kind,
            src_summary,
            src_addr,
            src_port,
            src_process_path,
            src_process_cmd,
            op,
            dst_idx,
            dst_type,
            dst_kind,
            dst_summary,
            dst_addr,
            dst_port,
            dst_process_path,
            dst_process_cmd,
            _event_uuid,
            ts,
            event_id,
        ) = row
        src_file_path = ""
        dst_file_path = ""
    else:
        (
            src_idx,
            src_type,
            src_kind,
            src_summary,
            src_addr,
            src_port,
            src_process_path,
            src_process_cmd,
            src_file_path,
            op,
            dst_idx,
            dst_type,
            dst_kind,
            dst_summary,
            dst_addr,
            dst_port,
            dst_process_path,
            dst_process_cmd,
            dst_file_path,
            _event_uuid,
            ts,
            *maybe_event_id,
        ) = row
        if maybe_event_id:
            event_id = maybe_event_id[0]
    try:
        src_idx_int = int(src_idx)
        dst_idx_int = int(dst_idx)
    except (TypeError, ValueError):
        return None

    actor_idx = src_idx_int
    obj_idx = dst_idx_int
    actor_type = str(src_type)
    obj_type = str(dst_type)
    actor_kind = str(src_kind)
    obj_kind = str(dst_kind)
    actor_text = str(src_summary)
    obj_text = str(dst_summary)
    obj_addr = str(dst_addr)
    obj_port = str(dst_port)
    actor_process_path = str(src_process_path)
    actor_process_cmd = str(src_process_cmd)
    obj_process_path = str(dst_process_path)
    obj_process_cmd = str(dst_process_cmd)
    actor_file_path = str(src_file_path)
    obj_file_path = str(dst_file_path)
    if actor_type != PROCESS_NODE_TYPE and obj_type != PROCESS_NODE_TYPE:
        return None
    if actor_type != PROCESS_NODE_TYPE:
        actor_idx, obj_idx = obj_idx, actor_idx
        actor_type, obj_type = obj_type, actor_type
        actor_kind, obj_kind = obj_kind, actor_kind
        actor_text, obj_text = obj_text, actor_text
        obj_addr = str(src_addr)
        obj_port = str(src_port)
        actor_process_path, obj_process_path = obj_process_path, actor_process_path
        actor_process_cmd, obj_process_cmd = obj_process_cmd, actor_process_cmd
        actor_file_path, obj_file_path = obj_file_path, actor_file_path
    action_text = optc_ecar_to_orthrus10_action(op) if optc_action_mode else str(op)
    object_type = str(obj_type or "unknown")
    text = " ".join(
        [
            f"A_{NODE_TYPE_TOKENS.get(actor_kind, NODE_TYPE_TOKENS['unknown'])}",
            actor_text,
            action_text,
            f"O_{NODE_TYPE_TOKENS.get(obj_kind, NODE_TYPE_TOKENS['unknown'])}",
            obj_text,
        ],
    )
    info_src, info_dst, rel, info_src_type, info_dst_type = information_flow(
        int(actor_idx),
        int(obj_idx),
        action_text,
        object_type,
    )
    stream_row: dict[str, Any] = {
        "pos": int(event_index),
        "event_index": int(event_index),
        "event_id": int(event_index if event_id is None else event_id),
        "timestamp_ns": int(ts),
        "process_idx": int(actor_idx),
        "src_idx": int(actor_idx),
        "dst_idx": int(obj_idx),
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
        "dst_addr": obj_addr if obj_kind == "netflow" or object_type == "netflow" else "",
        "dst_port": obj_port if obj_kind == "netflow" or object_type == "netflow" else "",
        "src_process_path": actor_process_path if actor_kind == "process" else "",
        "src_process_cmd": actor_process_cmd if actor_kind == "process" else "",
        "dst_process_path": obj_process_path if obj_kind == "process" else "",
        "dst_process_cmd": obj_process_cmd if obj_kind == "process" else "",
        "src_file_path": actor_file_path if actor_kind == "file" else "",
        "dst_file_path": obj_file_path if obj_kind == "file" else "",
    }
    include_labels = bool(abnormal_nodes)
    if include_labels:
        src_bad = int(actor_idx) in abnormal_nodes
        dst_bad = int(obj_idx) in abnormal_nodes
        stream_row["label"] = 2 if src_bad and dst_bad else (1 if src_bad or dst_bad else 0)
    return stream_row


def _test_scoring_node_maps(node_maps: Mapping[str, Any]) -> dict[str, Any]:
    """Return the label-free node-map view needed by DB test scoring."""
    scoring_maps = {
        key: node_maps[key]
        for key in TEST_SCORING_NODE_MAP_KEYS
        if key in node_maps
    }
    scoring_maps.setdefault("process_meta", {})
    return scoring_maps


def _effective_latent_dim(
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
) -> int:
    if _process_semantics_is_v1(process_cfg):
        assert process_cfg is not None
        return int(process_cfg.semantic_sketch_latent_dim)
    return int(config.latent_dim)


def _config_with_latent_dim(config: SlimConfig, latent_dim: int) -> SlimConfig:
    values = asdict(config)
    values["latent_dim"] = int(latent_dim)
    return SlimConfig(**values)


def _latent_dim_from_vocab_size(vocab_size: int) -> int:
    """Return bounded latent dimension from train-only residual vocabulary size."""
    if int(vocab_size) <= 128:
        return 32
    if int(vocab_size) <= 512:
        return 64
    if int(vocab_size) <= 2048:
        return 128
    return 256


def _make_residual_embedding_config(
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
) -> ResidualEmbeddingConfig:
    """Return the configured residual embedding setup."""
    max_tokens = max(int(config.max_tokens_per_node), 1)
    if _process_semantics_is_v1(process_cfg):
        assert process_cfg is not None
        max_tokens = max(int(process_cfg.semantic_sketch_max_tokens), 1)
    return ResidualEmbeddingConfig(
        method=str(config.semantic_embedding_method),
        latent_dim=_effective_latent_dim(config, process_cfg),
        max_tokens=max_tokens,
        word2vec_window=int(config.word2vec_window),
        word2vec_min_count=int(config.word2vec_min_count),
        word2vec_sg=int(config.word2vec_sg),
        word2vec_negative=int(config.word2vec_negative),
        word2vec_epochs=int(config.word2vec_epochs),
        word2vec_workers=int(config.word2vec_workers),
        word2vec_seed=int(config.word2vec_seed),
        word2vec_oov_policy=str(config.word2vec_oov_policy),
        word2vec_corpus_file_dir=str(config.word2vec_corpus_file_dir),
    )


# Explicit cross-module imports; replaces the migration namespace bridge.
from scripts.pipeline.features.semantic_features import (
    _process_semantics_is_v1,
    normalize_piece,
)
from scripts.pipeline.io.cache_payloads import _int_list
from scripts.pipeline.io.db_stream import _build_index_summaries, _day_bounds, _stream_events
from scripts.pipeline.io.event_artifacts import _phase3e_require_legacy_precompute_path_disabled
from scripts.pipeline.state.online_state_runtime import _stage_log
