"""Phase3E event-index, node/action artifact, and active artifact guards."""

from __future__ import annotations

import csv

from cs4m.semantics.optc_windows import (
    OPTC_NETFLOW_CANONICALIZATION_V1_3,
    OPTC_WINDOWS_V1_COARSE_SEMANTIC_MODE,
    OPTC_WINDOWS_V1_DETAIL_SEMANTIC_MODE,
    OPTC_WINDOWS_V1_3_COARSE_SEMANTIC_MODE,
    OPTC_WINDOWS_V1_3_DETAIL_SEMANTIC_MODE,
    OPTC_WINDOWS_V1_3B_COARSE_SEMANTIC_MODE,
    OPTC_WINDOWS_V1_3B_DETAIL_SEMANTIC_MODE,
    OPTC_WINDOWS_V1_3C_COARSE_SEMANTIC_MODE,
    OPTC_WINDOWS_V1_3C_DETAIL_SEMANTIC_MODE,
    is_optc_dataset,
    optc_file_natural_tokens_v1,
    optc_netflow_canonical_id_v1_3,
    optc_netflow_canonical_key_v1_3,
    optc_netflow_natural_tokens_v1,
    optc_netflow_natural_tokens_v1_3,
    optc_process_natural_tokens_v1,
    optc_process_natural_tokens_v1_3,
    optc_process_natural_tokens_v1_3b,
    optc_process_natural_tokens_v1_3c,
)
from scripts.pipeline.config.runtime_config import *


def _phase3e_validate_index_row(row: np.void) -> None:
    dtype = getattr(row, "dtype", None)
    if dtype != EVENT_INDEX_DTYPE:
        raise TypeError(f"event index row must use EVENT_INDEX_DTYPE, got {dtype}")


def _phase3e_entity_type_name(field_name: str, type_id: int) -> str:
    if int(type_id) not in ENTITY_TYPE_NAMES:
        raise ValueError(f"invalid {field_name}: {int(type_id)}")
    return ENTITY_TYPE_NAMES[int(type_id)]


def _phase3e_optc_canonical_node_id(
    node_id: int,
    node_kind: object,
    node_maps: Mapping[str, Any],
    canonicalization: object,
) -> int:
    """Return the optional OpTC netflow canonical id for one node."""
    mode = str(canonicalization)
    if mode != OPTC_NETFLOW_CANONICALIZATION_V1_3:
        return int(node_id)
    if normalize_piece(node_kind) != "netflow":
        return int(node_id)
    meta = dict(node_maps.get("netflow_meta", {}).get(int(node_id), {}))
    key = optc_netflow_canonical_key_v1_3(
        src_addr=meta.get("src_addr", ""),
        src_port=meta.get("src_port", ""),
        dst_addr=meta.get("dst_addr", ""),
        dst_port=meta.get("dst_port", ""),
    )
    return int(optc_netflow_canonical_id_v1_3(key))


def _phase3e_event_index_array(event_index: np.ndarray) -> np.ndarray:
    records = np.asarray(event_index)
    if records.dtype != EVENT_INDEX_DTYPE:
        raise TypeError(f"event_index must use EVENT_INDEX_DTYPE, got {records.dtype}")
    if records.ndim != 1:
        raise ValueError(f"event_index must be one-dimensional, got {records.ndim}")
    return records


def _phase3e_node_tokens_for_row_node(
    row: Mapping[str, Any],
    node_id: int,
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
) -> tuple[str, ...]:
    """Return label-free node semantic tokens for an information-flow node id."""
    raw_src = int(row.get("src_idx", row.get("info_src", -1)))
    raw_dst = int(row.get("dst_idx", row.get("info_dst", -1)))
    fields = row_fields(
        row,
        config.max_tokens_per_node,
        dataset=config.dataset,
        process_semantic_config=process_cfg,
        theia_netflow_policy=config.theia_netflow_policy,
    )
    if int(node_id) == raw_src:
        role_text = fields.get("src_role", row.get("src_summary", "unknown"))
    elif int(node_id) == raw_dst:
        role_text = fields.get("dst_role", row.get("dst_summary", "unknown"))
    else:
        role_text = fields.get("src_role", fields.get("dst_role", "unknown"))
    tokens = tuple(str(token) for token in residual_text_tokens(role_text))
    return tokens or ("unknown",)


def _phase3e_compact_event_index_row(
    row: Mapping[str, Any],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
) -> dict[str, object]:
    """Map one stream row into the compact Phase3E event-index source fields."""
    fields = row_fields(
        row,
        config.max_tokens_per_node,
        dataset=config.dataset,
        process_semantic_config=process_cfg,
        theia_netflow_policy=config.theia_netflow_policy,
    )
    raw_action = str(fields["raw_action"])
    if raw_action not in ORTHRUS10_ACTION_NAMES:
        raise ValueError(f"Phase3E event action is outside ORTHRUS10: {raw_action}")
    event_id_value = row.get("event_id", row.get("db_event_id", row.get("event_index")))
    if event_id_value is None:
        raise KeyError("Phase3E precompute rows require event_id or db_event_id")
    return {
        "event_id": int(event_id_value),
        "src_node_id": int(fields["info_src"]),
        "dst_node_id": int(fields["info_dst"]),
        "action_name": raw_action,
        "src_type_id": int(fields["info_src_type"]),
        "dst_type_id": int(fields["info_dst_type"]),
    }


def _phase3e_collect_node_universe(
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
) -> tuple[
    dict[int, tuple[str, ...]],
    dict[str, set[int]],
    set[int],
    set[int],
    dict[str, int],
]:
    """Collect only nodes referenced by the configured train/validation/test streams."""
    node_tokens_by_id: dict[int, tuple[str, ...]] = {}
    split_node_ids = {split: set() for split in ("train", "validation", "test")}
    src_node_ids: set[int] = set()
    dst_node_ids: set[int] = set()
    split_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        count = 0
        for row in split_rows.get(split, ()):
            compact = _phase3e_compact_event_index_row(row, config, process_cfg)
            src_id = int(compact["src_node_id"])
            dst_id = int(compact["dst_node_id"])
            split_node_ids[split].update((src_id, dst_id))
            src_node_ids.add(src_id)
            dst_node_ids.add(dst_id)
            node_tokens_by_id.setdefault(
                src_id,
                _phase3e_node_tokens_for_row_node(row, src_id, config, process_cfg),
            )
            node_tokens_by_id.setdefault(
                dst_id,
                _phase3e_node_tokens_for_row_node(row, dst_id, config, process_cfg),
            )
            count += 1
        split_counts[split] = int(count)
    return node_tokens_by_id, split_node_ids, src_node_ids, dst_node_ids, split_counts


def _phase3e_node_tokens_from_db_node(
    *,
    node_id: int,
    node_kind: str,
    node_summary: str,
    node_maps: Mapping[str, Any],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
) -> tuple[str, ...]:
    """Return label-free node tokens from process/file/netflow node tables."""
    del process_cfg
    kind = normalize_piece(node_kind)
    if kind == "process":
        meta = node_maps.get("process_meta", {}).get(int(node_id), {})
        if is_optc_dataset(config.dataset):
            if str(config.semantic_mode) in {
                OPTC_WINDOWS_V1_DETAIL_SEMANTIC_MODE,
            }:
                return optc_process_natural_tokens_v1(
                    meta.get("path", ""),
                    meta.get("cmd", ""),
                    detail=True,
                )
            if str(config.semantic_mode) in {
                OPTC_WINDOWS_V1_COARSE_SEMANTIC_MODE,
            }:
                return optc_process_natural_tokens_v1(
                    meta.get("path", ""),
                    meta.get("cmd", ""),
                    detail=False,
                )
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_3_DETAIL_SEMANTIC_MODE:
                return optc_process_natural_tokens_v1_3(
                    meta.get("path", ""),
                    meta.get("cmd", ""),
                    detail=True,
                )
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_3_COARSE_SEMANTIC_MODE:
                return optc_process_natural_tokens_v1_3(
                    meta.get("path", ""),
                    meta.get("cmd", ""),
                    detail=False,
                )
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_3B_DETAIL_SEMANTIC_MODE:
                return optc_process_natural_tokens_v1_3b(
                    meta.get("path", ""),
                    meta.get("cmd", ""),
                    detail=True,
                )
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_3B_COARSE_SEMANTIC_MODE:
                return optc_process_natural_tokens_v1_3b(
                    meta.get("path", ""),
                    meta.get("cmd", ""),
                    detail=False,
                )
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_3C_DETAIL_SEMANTIC_MODE:
                return optc_process_natural_tokens_v1_3c(
                    meta.get("path", ""),
                    meta.get("cmd", ""),
                    detail=True,
                )
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_3C_COARSE_SEMANTIC_MODE:
                return optc_process_natural_tokens_v1_3c(
                    meta.get("path", ""),
                    meta.get("cmd", ""),
                    detail=False,
                )
        if is_clearscope_dataset(config.dataset):
            if (
                normalize_clearscope_semantic_mode(config.semantic_mode)
                != CLEARSCOPE_LEGACY_SEMANTIC_MODE
            ):
                return android_process_natural_tokens_refined(meta.get("cmd", ""))
            return android_process_natural_tokens(meta.get("cmd", ""))
        if is_theia_dataset(config.dataset):
            return linux_process_natural_tokens(meta.get("path", ""), meta.get("cmd", ""))
        if is_cadets_dataset(config.dataset):
            if cadets_semantic_mode_is_v3_safe_lexical(config.semantic_mode):
                return freebsd_process_natural_tokens_v3_safe_lexical(meta.get("cmd", ""))
            return freebsd_process_natural_tokens(meta.get("cmd", ""))
    if kind == "file":
        meta = node_maps.get("file_meta", {}).get(int(node_id), {})
        if is_optc_dataset(config.dataset):
            if str(config.semantic_mode) in {
                OPTC_WINDOWS_V1_DETAIL_SEMANTIC_MODE,
                OPTC_WINDOWS_V1_3_DETAIL_SEMANTIC_MODE,
                OPTC_WINDOWS_V1_3B_DETAIL_SEMANTIC_MODE,
                OPTC_WINDOWS_V1_3C_DETAIL_SEMANTIC_MODE,
            }:
                return optc_file_natural_tokens_v1(meta.get("path", ""), detail=True)
            if str(config.semantic_mode) in {
                OPTC_WINDOWS_V1_COARSE_SEMANTIC_MODE,
                OPTC_WINDOWS_V1_3_COARSE_SEMANTIC_MODE,
                OPTC_WINDOWS_V1_3B_COARSE_SEMANTIC_MODE,
                OPTC_WINDOWS_V1_3C_COARSE_SEMANTIC_MODE,
            }:
                return optc_file_natural_tokens_v1(meta.get("path", ""), detail=False)
        if is_clearscope_dataset(config.dataset):
            clearscope_mode = normalize_clearscope_semantic_mode(config.semantic_mode)
            if clearscope_mode == CLEARSCOPE_LEGACY_SEMANTIC_MODE:
                return clearscope_file_natural_tokens(meta.get("path", ""))
            if clearscope_mode == CLEARSCOPE_REFINED_SEMANTIC_MODE:
                return clearscope_file_natural_tokens_refined(meta.get("path", ""))
            if clearscope_mode == CLEARSCOPE_V32_CACHE_ONLY_SEMANTIC_MODE:
                return clearscope_file_natural_tokens_v32_cache_only(meta.get("path", ""))
            if clearscope_mode == CLEARSCOPE_V32_SEMANTIC_MODE:
                return clearscope_file_natural_tokens_v32(meta.get("path", ""))
            if clearscope_mode == CLEARSCOPE_V31_SEMANTIC_MODE:
                return clearscope_file_natural_tokens_v31(meta.get("path", ""))
            return clearscope_file_natural_tokens_v3(meta.get("path", ""))
        if is_theia_dataset(config.dataset):
            return linux_file_natural_tokens(meta.get("path", ""))
        if is_cadets_dataset(config.dataset):
            if cadets_semantic_mode_is_v3_safe_lexical(config.semantic_mode):
                return freebsd_file_natural_tokens_v3_safe_lexical(meta.get("path", ""))
            return freebsd_file_natural_tokens(meta.get("path", ""))
    if kind == "netflow":
        meta = node_maps.get("netflow_meta", {}).get(int(node_id), {})
        src_addr = meta.get("src_addr", "")
        src_port = meta.get("src_port", "")
        dst_addr = meta.get("dst_addr", "")
        dst_port = meta.get("dst_port", "")
        if is_optc_dataset(config.dataset):
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_DETAIL_SEMANTIC_MODE:
                return optc_netflow_natural_tokens_v1(
                    src_addr,
                    dst_addr,
                    dst_port,
                    detail=True,
                )
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_COARSE_SEMANTIC_MODE:
                return optc_netflow_natural_tokens_v1(
                    src_addr,
                    dst_addr,
                    dst_port,
                    detail=False,
                )
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_3_DETAIL_SEMANTIC_MODE:
                return optc_netflow_natural_tokens_v1_3(
                    src_addr,
                    dst_addr,
                    dst_port,
                    src_port=src_port,
                    detail=True,
                )
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_3_COARSE_SEMANTIC_MODE:
                return optc_netflow_natural_tokens_v1_3(
                    src_addr,
                    dst_addr,
                    dst_port,
                    src_port=src_port,
                    detail=False,
                )
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_3B_DETAIL_SEMANTIC_MODE:
                return optc_netflow_natural_tokens_v1_3(
                    src_addr,
                    dst_addr,
                    dst_port,
                    src_port=src_port,
                    detail=True,
                )
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_3B_COARSE_SEMANTIC_MODE:
                return optc_netflow_natural_tokens_v1_3(
                    src_addr,
                    dst_addr,
                    dst_port,
                    src_port=src_port,
                    detail=False,
                )
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_3C_DETAIL_SEMANTIC_MODE:
                return optc_netflow_natural_tokens_v1_3(
                    src_addr,
                    dst_addr,
                    dst_port,
                    src_port=src_port,
                    detail=True,
                )
            if str(config.semantic_mode) == OPTC_WINDOWS_V1_3C_COARSE_SEMANTIC_MODE:
                return optc_netflow_natural_tokens_v1_3(
                    src_addr,
                    dst_addr,
                    dst_port,
                    src_port=src_port,
                    detail=False,
                )
        if is_clearscope_dataset(config.dataset):
            if (
                normalize_clearscope_semantic_mode(config.semantic_mode)
                != CLEARSCOPE_LEGACY_SEMANTIC_MODE
            ):
                return clearscope_netflow_natural_tokens_refined()
            return clearscope_netflow_natural_tokens()
        if is_theia_dataset(config.dataset):
            if str(config.theia_netflow_policy) == "fixed":
                return linux_netflow_detail_or_fixed_natural_tokens(dst_addr, src_addr)
            return linux_netflow_natural_tokens(dst_addr, src_addr)
        if is_cadets_dataset(config.dataset):
            if cadets_semantic_mode_is_v3_safe_lexical(config.semantic_mode):
                return freebsd_netflow_natural_tokens_v3_safe_lexical(dst_addr, src_addr)
            return freebsd_netflow_natural_tokens(dst_addr, src_addr)
    tokens = split_summary_tokens(node_summary, int(config.max_tokens_per_node))
    return tokens or (kind or "unknown",)


def _phase3e_node_table_counts_db(conn: Any) -> dict[str, int]:
    """Return total DB node counts without fetching node table rows."""
    with conn.cursor() as cur:
        cur.execute("select count(*) from netflow_node_table")
        netflow_count = int(cur.fetchone()[0])
        cur.execute(f"select count(*) from {SUBJECT_NODE_TABLE}")
        process_count = int(cur.fetchone()[0])
        cur.execute("select count(*) from file_node_table")
        file_count = int(cur.fetchone()[0])
    return {
        "process": process_count,
        "file": file_count,
        "netflow": netflow_count,
        "total": process_count + file_count + netflow_count,
    }


def _phase3e_fetch_used_node_lookup_batched(
    *,
    conn: Any,
    used_node_ids: set[int],
    batch_size: int,
) -> tuple[dict[int, tuple[str, str]], dict[str, dict[int, dict[str, str]]], float]:
    """Pass 2: query process/file/netflow rows only for used node ids."""
    sorted_ids = sorted(int(node_id) for node_id in used_node_ids)
    indexid2summary: dict[int, tuple[str, str]] = {}
    meta_by_kind: dict[str, dict[int, dict[str, str]]] = {
        "process_meta": {},
        "file_meta": {},
        "netflow_meta": {},
    }
    started = time.perf_counter()
    safe_batch_size = max(int(batch_size), 1)
    with conn.cursor() as cur:
        for offset in range(0, len(sorted_ids), safe_batch_size):
            batch = sorted_ids[offset : offset + safe_batch_size]
            cur.execute(
                f"""
                select index_id, path, cmd
                from {SUBJECT_NODE_TABLE}
                where index_id = any(%s)
                """,
                (batch,),
            )
            for index_id, path, cmd in cur.fetchall():
                node_id = int(index_id)
                path_text = str(path or "")
                cmd_text = str(cmd or "")
                indexid2summary[node_id] = ("process", " ".join((path_text, cmd_text)))
                meta_by_kind["process_meta"][node_id] = {
                    "path": path_text,
                    "cmd": cmd_text,
                }

            cur.execute(
                """
                select index_id, path
                from file_node_table
                where index_id = any(%s)
                """,
                (batch,),
            )
            for index_id, path in cur.fetchall():
                node_id = int(index_id)
                path_text = str(path or "")
                indexid2summary[node_id] = ("file", path_text)
                meta_by_kind["file_meta"][node_id] = {"path": path_text}

            cur.execute(
                """
                select index_id, src_addr, src_port, dst_addr, dst_port
                from netflow_node_table
                where index_id = any(%s)
                """,
                (batch,),
            )
            for index_id, src_addr, src_port, dst_addr, dst_port in cur.fetchall():
                node_id = int(index_id)
                values = {
                    "src_addr": str(src_addr or ""),
                    "src_port": str(src_port or ""),
                    "dst_addr": str(dst_addr or ""),
                    "dst_port": str(dst_port or ""),
                }
                indexid2summary[node_id] = (
                    "netflow",
                    " ".join(
                        (
                            values["src_addr"],
                            values["src_port"],
                            values["dst_addr"],
                            values["dst_port"],
                        ),
                    ),
                )
                meta_by_kind["netflow_meta"][node_id] = values
    return indexid2summary, meta_by_kind, float(time.perf_counter() - started)


def _phase3e_empty_node_lookup_shape() -> tuple[
    dict[int, tuple[str, str]],
    dict[str, dict[int, dict[str, str]]],
]:
    """Return the shared empty shape for used-node lookup helpers."""
    return (
        {},
        {
            "process_meta": {},
            "file_meta": {},
            "netflow_meta": {},
        },
    )


def _phase3e_prepare_temp_used_node_ids(
    conn: Any,
    used_node_ids: set[int],
    insert_page_size: int = 10000,
) -> None:
    """Create and populate a session-local used-node id table for join lookup."""
    sorted_ids = sorted(int(node_id) for node_id in used_node_ids)
    with conn.cursor() as cur:
        cur.execute("drop table if exists temp_phase3e_used_node_ids")
        cur.execute(
            """
            create temp table temp_phase3e_used_node_ids (
                index_id bigint primary key
            ) on commit drop
            """,
        )
        if not sorted_ids:
            return
        rows = [(node_id,) for node_id in sorted_ids]
        try:
            from psycopg2.extras import execute_values

            execute_values(
                cur,
                "insert into temp_phase3e_used_node_ids (index_id) values %s",
                rows,
                page_size=max(int(insert_page_size), 1),
            )
        except Exception:
            cur.executemany(
                "insert into temp_phase3e_used_node_ids (index_id) values (%s)",
                rows,
            )


def _phase3e_fetch_used_node_lookup_joined(
    *,
    conn: Any,
    used_node_ids: set[int],
) -> tuple[dict[int, tuple[str, str]], dict[str, dict[int, dict[str, str]]], float]:
    """Fetch used process/file/netflow metadata via a temp used-node table join."""
    if not used_node_ids:
        indexid2summary, meta_by_kind = _phase3e_empty_node_lookup_shape()
        return indexid2summary, meta_by_kind, 0.0
    started = time.perf_counter()
    indexid2summary, meta_by_kind = _phase3e_empty_node_lookup_shape()
    _phase3e_prepare_temp_used_node_ids(conn, set(used_node_ids))
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select n.index_id, n.path, n.cmd
            from {SUBJECT_NODE_TABLE} n
            join temp_phase3e_used_node_ids u on u.index_id = n.index_id
            """,
        )
        for index_id, path, cmd in cur.fetchall():
            node_id = int(index_id)
            path_text = str(path or "")
            cmd_text = str(cmd or "")
            indexid2summary[node_id] = ("process", " ".join((path_text, cmd_text)))
            meta_by_kind["process_meta"][node_id] = {
                "path": path_text,
                "cmd": cmd_text,
            }

        cur.execute(
            """
            select n.index_id, n.path
            from file_node_table n
            join temp_phase3e_used_node_ids u on u.index_id = n.index_id
            """,
        )
        for index_id, path in cur.fetchall():
            node_id = int(index_id)
            path_text = str(path or "")
            indexid2summary[node_id] = ("file", path_text)
            meta_by_kind["file_meta"][node_id] = {"path": path_text}

        cur.execute(
            """
            select n.index_id, n.src_addr, n.src_port, n.dst_addr, n.dst_port
            from netflow_node_table n
            join temp_phase3e_used_node_ids u on u.index_id = n.index_id
            """,
        )
        for index_id, src_addr, src_port, dst_addr, dst_port in cur.fetchall():
            node_id = int(index_id)
            values = {
                "src_addr": str(src_addr or ""),
                "src_port": str(src_port or ""),
                "dst_addr": str(dst_addr or ""),
                "dst_port": str(dst_port or ""),
            }
            indexid2summary[node_id] = (
                "netflow",
                " ".join(
                    (
                        values["src_addr"],
                        values["src_port"],
                        values["dst_addr"],
                        values["dst_port"],
                    ),
                ),
            )
            meta_by_kind["netflow_meta"][node_id] = values
    return indexid2summary, meta_by_kind, float(time.perf_counter() - started)


def _phase3e_build_node_table_from_used_lookup(
    *,
    used_node_ids: set[int],
    split_used_nodes: Mapping[str, set[int]],
    indexid2summary: Mapping[int, tuple[str, str]],
    meta_by_kind: Mapping[str, Mapping[int, Mapping[str, str]]],
    adapter: ResidualWord2VecTokenAdapter,
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
) -> NodeEmbeddingTable:
    """Build Phase3E embeddings only for nodes used by the filtered event stream."""
    node_maps = {
        "indexid2summary": indexid2summary,
        "process_meta": meta_by_kind.get("process_meta", {}),
        "file_meta": meta_by_kind.get("file_meta", {}),
        "netflow_meta": meta_by_kind.get("netflow_meta", {}),
    }
    node_tokens_by_id: dict[int, tuple[str, ...]] = {}
    kind_counts = {"process": 0, "file": 0, "netflow": 0}
    for node_id in sorted(int(value) for value in used_node_ids):
        if node_id not in indexid2summary:
            continue
        node_kind, node_summary = indexid2summary[node_id]
        kind = normalize_piece(node_kind)
        if kind in kind_counts:
            kind_counts[kind] += 1
        node_tokens_by_id[node_id] = _phase3e_node_tokens_from_db_node(
            node_id=node_id,
            node_kind=kind,
            node_summary=str(node_summary),
            node_maps=node_maps,
            config=config,
            process_cfg=process_cfg,
        )
    node_table = build_node_embedding_table(
        node_tokens_by_id=node_tokens_by_id,
        split_node_ids={
            "train": set(split_used_nodes.get("train", set())),
            "validation": set(split_used_nodes.get("validation", set())),
            "test": set(split_used_nodes.get("test", set())),
        },
        adapter=adapter,
        src_node_ids=set(),
        dst_node_ids=set(),
    )
    node_count = int(len(node_table.node_id_to_idx))
    multiple_splits = 0
    for node_id in node_table.node_id_to_idx:
        appearances = sum(
            1 for split_nodes in split_used_nodes.values() if int(node_id) in split_nodes
        )
        if appearances > 1:
            multiple_splits += 1
    node_table.meta.update(
        {
            "node_universe_source": "event_stream_used_nodes_batched_db_lookup",
            "covered_splits": "train,validation,test",
            "covered_split_names": ["train", "validation", "test"],
            "covers_only_event_referenced_nodes": True,
            "covers_full_db_node_table": False,
            "num_nodes_total_in_node_table": node_count,
            "num_train_nodes": int(len(split_used_nodes.get("train", set()))),
            "num_validation_nodes": int(len(split_used_nodes.get("validation", set()))),
            "num_test_nodes": int(len(split_used_nodes.get("test", set()))),
            "num_src_nodes": None,
            "num_dst_nodes": None,
            "num_nodes_seen_in_multiple_splits": int(multiple_splits),
            "num_process_nodes": int(kind_counts["process"]),
            "num_file_nodes": int(kind_counts["file"]),
            "num_netflow_nodes": int(kind_counts["netflow"]),
            "node_lookup_source_tables": ["process", "file", "netflow"],
        },
    )
    node_table.coverage.update(
        {
            "node_universe_source": "event_stream_used_nodes_batched_db_lookup",
            "covers_only_event_referenced_nodes": True,
            "covers_full_db_node_table": False,
            "num_nodes_total_in_node_table": node_count,
            "num_process_nodes": int(kind_counts["process"]),
            "num_file_nodes": int(kind_counts["file"]),
            "num_netflow_nodes": int(kind_counts["netflow"]),
        },
    )
    return node_table


def _phase3e_build_optc_canonical_artifact_inputs(
    *,
    used_node_ids: set[int],
    split_used_nodes: Mapping[str, set[int]],
    indexid2summary: Mapping[int, tuple[str, str]],
    meta_by_kind: Mapping[str, Mapping[int, Mapping[str, str]]],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
) -> dict[str, Any]:
    """Build canonical node maps for OpTC netflow identity compression."""
    node_maps = {
        "indexid2summary": indexid2summary,
        "process_meta": meta_by_kind.get("process_meta", {}),
        "file_meta": meta_by_kind.get("file_meta", {}),
        "netflow_meta": meta_by_kind.get("netflow_meta", {}),
    }
    original_to_canonical: dict[int, int] = {}
    canonical_kind_by_id: dict[int, str] = {}
    canonical_tokens_by_id: dict[int, tuple[str, ...]] = {}
    original_netflow_ids: set[int] = set()
    canonical_netflow_ids: set[int] = set()
    canonical_source_count: dict[int, int] = {}
    progress_interval = max(int(getattr(config, "progress_interval_events", 100000)), 1)
    started = time.perf_counter()
    for node_id in sorted(int(value) for value in used_node_ids):
        if node_id not in indexid2summary:
            continue
        node_kind, node_summary = indexid2summary[node_id]
        kind = normalize_piece(node_kind)
        canonical_id = _phase3e_optc_canonical_node_id(
            node_id,
            kind,
            node_maps,
            getattr(config, "optc_netflow_node_canonicalization", "none"),
        )
        original_to_canonical[node_id] = int(canonical_id)
        canonical_kind_by_id.setdefault(int(canonical_id), kind)
        if int(canonical_id) not in canonical_tokens_by_id:
            canonical_tokens_by_id[int(canonical_id)] = _phase3e_node_tokens_from_db_node(
                node_id=node_id,
                node_kind=kind,
                node_summary=str(node_summary),
                node_maps=node_maps,
                config=config,
                process_cfg=process_cfg,
            )
        canonical_source_count[int(canonical_id)] = canonical_source_count.get(int(canonical_id), 0) + 1
        if kind == "netflow":
            original_netflow_ids.add(node_id)
            canonical_netflow_ids.add(int(canonical_id))
        if len(original_to_canonical) % progress_interval == 0:
            _stage_log(
                config,
                "phase3e_optc_canonical_inputs_progress",
                count=len(original_to_canonical),
                canonical_nodes=len(canonical_tokens_by_id),
                canonical_netflow_nodes=len(canonical_netflow_ids),
            )
    canonical_split_used_nodes: dict[str, set[int]] = {}
    for split, node_ids in split_used_nodes.items():
        canonical_split_used_nodes[str(split)] = {
            int(original_to_canonical.get(int(node_id), int(node_id)))
            for node_id in set(node_ids)
            if int(node_id) in original_to_canonical
        }
    summary = {
        "optc_netflow_node_canonicalization": str(
            getattr(config, "optc_netflow_node_canonicalization", "none"),
        ),
        "hybrid_identity_mode": "canonical_state_original_eval",
        "original_netflow_node_count": int(len(original_netflow_ids)),
        "canonical_netflow_node_count": int(len(canonical_netflow_ids)),
        "netflow_compression_ratio": (
            float(len(original_netflow_ids) / len(canonical_netflow_ids))
            if canonical_netflow_ids
            else 0.0
        ),
        "canonical_node_count": int(len(canonical_tokens_by_id)),
        "original_node_count": int(len(original_to_canonical)),
        "max_originals_per_canonical_node": int(max(canonical_source_count.values(), default=0)),
        "canonical_input_build_seconds": float(time.perf_counter() - started),
    }
    return {
        "original_to_canonical": original_to_canonical,
        "canonical_kind_by_id": canonical_kind_by_id,
        "canonical_tokens_by_id": canonical_tokens_by_id,
        "canonical_split_used_nodes": canonical_split_used_nodes,
        "summary": summary,
    }


def _phase3e_write_optc_original_to_canonical_sidecar(
    *,
    path: Path,
    original_to_canonical: Mapping[int, int],
    indexid2summary: Mapping[int, tuple[str, str]],
) -> dict[str, object]:
    """Write original-to-canonical netflow mapping sidecar for hybrid identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["original_node_id", "canonical_node_id", "node_kind"],
        )
        writer.writeheader()
        for original_id, canonical_id in sorted(
            original_to_canonical.items(),
            key=lambda item: int(item[0]),
        ):
            node_kind = normalize_piece(indexid2summary.get(int(original_id), ("unknown", ""))[0])
            if node_kind != "netflow":
                continue
            writer.writerow(
                {
                    "original_node_id": int(original_id),
                    "canonical_node_id": int(canonical_id),
                    "node_kind": node_kind,
                },
            )
            count += 1
    return {"path": str(path), "row_count": int(count)}


def _phase3e_build_node_table_from_db_node_maps(
    *,
    node_maps: Mapping[str, Any],
    adapter: ResidualWord2VecTokenAdapter,
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
) -> NodeEmbeddingTable:
    """Build Phase3E node embeddings from process/file/netflow node tables."""
    indexid2summary = node_maps.get("indexid2summary", {})
    allowed_kinds = {"process", "file", "netflow"}
    node_tokens_by_id: dict[int, tuple[str, ...]] = {}
    kind_counts = {"process": 0, "file": 0, "netflow": 0}
    unknown_kinds: dict[int, str] = {}
    for raw_node_id, summary in sorted(indexid2summary.items(), key=lambda item: int(item[0])):
        node_id = int(raw_node_id)
        node_kind, node_summary = summary
        kind = normalize_piece(node_kind)
        if kind not in allowed_kinds:
            unknown_kinds[node_id] = str(node_kind)
            continue
        kind_counts[kind] += 1
        node_tokens_by_id[node_id] = _phase3e_node_tokens_from_db_node(
            node_id=node_id,
            node_kind=kind,
            node_summary=str(node_summary),
            node_maps=node_maps,
            config=config,
            process_cfg=process_cfg,
        )
    if unknown_kinds:
        sample = dict(list(unknown_kinds.items())[:5])
        raise ValueError(f"Phase3E DB node table contains unsupported node kinds: {sample}")

    all_node_ids = set(node_tokens_by_id)
    node_table = build_node_embedding_table(
        node_tokens_by_id=node_tokens_by_id,
        split_node_ids={"train": all_node_ids, "validation": set(), "test": set()},
        adapter=adapter,
        src_node_ids=set(),
        dst_node_ids=set(),
    )
    node_count = int(len(all_node_ids))
    node_table.meta.update(
        {
            "node_universe_source": "db_node_tables",
            "covered_splits": "train,validation,test",
            "covered_split_names": ["train", "validation", "test"],
            "covers_only_event_referenced_nodes": False,
            "covers_full_db_node_table": True,
            "num_nodes_total_in_node_table": node_count,
            "num_train_nodes": None,
            "num_validation_nodes": None,
            "num_test_nodes": None,
            "num_src_nodes": None,
            "num_dst_nodes": None,
            "num_nodes_seen_in_multiple_splits": None,
            "num_process_nodes": int(kind_counts["process"]),
            "num_file_nodes": int(kind_counts["file"]),
            "num_netflow_nodes": int(kind_counts["netflow"]),
            "node_table_source_tables": ["process", "file", "netflow"],
        },
    )
    node_table.coverage.update(
        {
            "node_universe_source": "db_node_tables",
            "covers_full_db_node_table": True,
            "num_nodes_total_in_node_table": node_count,
            "num_process_nodes": int(kind_counts["process"]),
            "num_file_nodes": int(kind_counts["file"]),
            "num_netflow_nodes": int(kind_counts["netflow"]),
        },
    )
    return node_table


def _phase3e_count_split_events_db(
    *,
    conn: Any,
    year_month: str,
    days: Sequence[int],
    node_maps: Mapping[str, Any],
    event_filter: bool,
    max_events: int,
) -> tuple[int, int]:
    """Count one Phase3E split in PostgreSQL with the same slim stream predicate."""
    if is_optc_dataset(config.dataset):
        count = _phase3e_count_optc_index_events(
            conn=conn,
            year_month=year_month,
            days=days,
            node_maps=node_maps,
            max_events=int(max_events),
        )
        return int(count), int(count)
    if int(max_events) > 0:
        bounded_count = _phase3e_count_bounded_split_events_db(
            conn=conn,
            year_month=year_month,
            days=days,
            node_maps=node_maps,
            event_filter=event_filter,
            max_events=int(max_events),
        )
        return int(bounded_count), int(bounded_count)
    full_count = _phase3e_count_split_events_with_lookup(
        conn=conn,
        year_month=year_month,
        days=days,
        node_maps=node_maps,
        event_filter=event_filter,
        allowed_hashes=None,
    )
    return int(full_count), int(full_count)


def _phase3e_count_optc_index_events(
    *,
    conn: Any,
    year_month: str,
    days: Sequence[int],
    node_maps: Mapping[str, Any],
    max_events: int,
) -> int:
    """Count OpTC process-involved events using src/dst index ids."""
    day_sql, day_params = _slim_temp_day_filter(year_month, days)
    if not day_params:
        return 0
    limit_sql = ""
    params: list[Any] = [*day_params]
    if int(max_events) > 0:
        limit_sql = "limit %s"
        params.append(int(max_events))
    process_ids = [
        int(node_id)
        for node_id, (kind, _summary) in node_maps.get("indexid2summary", {}).items()
        if normalize_piece(kind) == PROCESS_NODE_TYPE
    ]
    if not process_ids:
        return 0
    params.extend([process_ids, process_ids])
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select count(*)
            from (
                select e.src_index_id, e.dst_index_id
                from event_table e
                where {day_sql}
                order by e.timestamp_rec, e._id
                {limit_sql}
            ) bounded_events
            where bounded_events.src_index_id::bigint = any(%s)
                or bounded_events.dst_index_id::bigint = any(%s)
            """,
            tuple(params),
        )
        return int(cur.fetchone()[0])


def _phase3e_count_bounded_split_events_db(
    *,
    conn: Any,
    year_month: str,
    days: Sequence[int],
    node_maps: Mapping[str, Any],
    event_filter: bool,
    max_events: int,
) -> int:
    """Count the actual bounded stream without scanning the whole split first."""
    day_sql, day_params = _slim_temp_day_filter(year_month, days)
    if not day_params:
        return 0
    allowed_hashes = _bounded_stream_node_hashes(
        conn,
        year_month,
        days,
        event_filter,
        int(max_events),
    )
    _prepare_slim_node_lookup_temp_table(conn, node_maps, allowed_hashes=allowed_hashes)
    params: list[Any] = [*day_params]
    event_filter_sql = ""
    if event_filter:
        event_filter_sql = "and e.operation = any(%s)"
        params.append(list(ORTHRUS10_EVENT_TYPES))
    params.append(int(max_events))
    params.extend([PROCESS_NODE_TYPE, PROCESS_NODE_TYPE])
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select count(*)
            from (
                select e.src_node, e.dst_node, e.operation
                from event_table e
                where {day_sql}
                    {event_filter_sql}
                order by e.timestamp_rec, e._id
                limit %s
            ) bounded_events
            join temp_slim_node_lookup src_lookup
                on src_lookup.hash_id = bounded_events.src_node::text
            join temp_slim_node_lookup dst_lookup
                on dst_lookup.hash_id = bounded_events.dst_node::text
            where (
                src_lookup.node_type = %s
                or dst_lookup.node_type = %s
            )
            """,
            tuple(params),
        )
        return int(cur.fetchone()[0])


def _phase3e_count_split_events_with_lookup(
    *,
    conn: Any,
    year_month: str,
    days: Sequence[int],
    node_maps: Mapping[str, Any],
    event_filter: bool,
    allowed_hashes: set[str] | None,
) -> int:
    """Count one split through the temporary process-involved node lookup."""
    day_sql, day_params = _slim_temp_day_filter(year_month, days)
    if not day_params:
        return 0
    _prepare_slim_node_lookup_temp_table(conn, node_maps, allowed_hashes=allowed_hashes)
    params: list[Any] = [*day_params, PROCESS_NODE_TYPE, PROCESS_NODE_TYPE]
    event_filter_sql = ""
    if event_filter:
        event_filter_sql = "and e.operation = any(%s)"
        params.append(list(ORTHRUS10_EVENT_TYPES))
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select count(*)
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
            """,
            tuple(params),
        )
        full_count = int(cur.fetchone()[0])
    return full_count


def _phase3e_write_event_index_split_from_db(
    *,
    conn: Any,
    year_month: str,
    days: Sequence[int],
    node_maps: Mapping[str, Any],
    event_filter: bool,
    fetch_size: int,
    max_events: int,
    expected_count: int,
    path: Path,
    split: str,
    node_id_to_idx: Mapping[int, int],
    action_name_to_id: Mapping[str, int],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
    order_by: Sequence[object] | object,
    event_id_source: str,
) -> dict[str, object]:
    """Stream one DB split directly into a compact event-index memmap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if int(expected_count) <= 0:
        path.write_bytes(b"")
        fingerprint = event_index_fingerprint(np.zeros((0,), dtype=EVENT_INDEX_DTYPE))
        return {
            "split": str(split),
            "row_order_is_stream_order": True,
            "order_by": _phase3e_json_order_by(order_by),
            "event_id_source": str(event_id_source),
            "num_events": 0,
            "path": str(path),
            "event_index_dtype": _phase3e_event_index_dtype_descriptor(),
            "dtype_descriptor": _phase3e_event_index_dtype_descriptor(),
            "event_index_fingerprint": fingerprint,
            "fingerprint": fingerprint,
            "event_count_source": "postgres_count",
            "write_mode": "postgres_stream_to_memmap",
        }
    mapped = np.memmap(path, dtype=EVENT_INDEX_DTYPE, mode="w+", shape=(expected_count,))
    progress_interval = int(config.progress_interval_events)
    count = 0
    _stage_log(config, f"phase3e_{split}_event_index_stream_start")
    try:
        rows = stream_dataset_rows_slim(
            conn,
            year_month,
            days,
            node_maps,
            event_filter,
            int(fetch_size),
            int(max_events),
            "temp_table",
            abnormal_nodes=None,
            optc_action_mode=is_optc_dataset(config.dataset),
        )
        for row in rows:
            if count >= int(expected_count):
                raise ValueError(
                    f"Phase3E {split} stream exceeded PostgreSQL COUNT: {expected_count}",
                )
            compact = _phase3e_compact_event_index_row(row, config, process_cfg)
            src_node_id = int(compact["src_node_id"])
            dst_node_id = int(compact["dst_node_id"])
            action_name = str(compact["action_name"])
            if src_node_id not in node_id_to_idx:
                raise KeyError(f"event src node is absent from DB node tables: {src_node_id}")
            if dst_node_id not in node_id_to_idx:
                raise KeyError(f"event dst node is absent from DB node tables: {dst_node_id}")
            if action_name not in action_name_to_id:
                raise KeyError(f"missing action_id for action: {action_name}")
            mapped[count] = (
                int(compact["event_id"]),
                int(node_id_to_idx[src_node_id]),
                int(node_id_to_idx[dst_node_id]),
                int(action_name_to_id[action_name]),
                int(compact["src_type_id"]),
                int(compact["dst_type_id"]),
            )
            count += 1
            if progress_interval > 0 and count % progress_interval == 0:
                _stage_log(config, f"phase3e_{split}_event_index_progress", count=count)
        if count != int(expected_count):
            raise ValueError(
                f"Phase3E {split} stream count mismatch: wrote {count}, "
                f"expected {expected_count}",
            )
        mapped.flush()
        fingerprint = event_index_fingerprint(np.asarray(mapped))
    finally:
        del mapped
    _stage_log(config, f"phase3e_{split}_event_index_stream_end", count=count)
    return {
        "split": str(split),
        "row_order_is_stream_order": True,
        "order_by": _phase3e_json_order_by(order_by),
        "event_id_source": str(event_id_source),
        "num_events": int(count),
        "path": str(path),
        "event_index_dtype": _phase3e_event_index_dtype_descriptor(),
        "dtype_descriptor": _phase3e_event_index_dtype_descriptor(),
        "event_index_fingerprint": fingerprint,
        "fingerprint": fingerprint,
        "event_count_source": "postgres_count",
        "write_mode": "postgres_stream_to_memmap",
    }


def _phase3e_compact_event_from_direct_stream(
    event: Phase3EStreamEvent,
    action_name_to_id: Mapping[str, int],
    node_id_to_idx: Mapping[int, int],
    node_kind_by_id: Mapping[int, str],
    original_to_canonical: Mapping[int, int] | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Return the six-field event_index tuple from one direct DB event."""
    action = str(event.operation)
    if action not in action_name_to_id:
        raise KeyError(f"Phase3E event action is outside ORTHRUS10: {action}")
    mapping = original_to_canonical or {}
    src_node_id = int(mapping.get(int(event.src_node_id), int(event.src_node_id)))
    dst_node_id = int(mapping.get(int(event.dst_node_id), int(event.dst_node_id)))
    if src_node_id not in node_id_to_idx:
        raise KeyError(f"missing src node in used_node_lookup: {src_node_id}")
    if dst_node_id not in node_id_to_idx:
        raise KeyError(f"missing dst node in used_node_lookup: {dst_node_id}")
    src_kind = normalize_piece(node_kind_by_id.get(src_node_id, "unknown"))
    dst_kind = normalize_piece(node_kind_by_id.get(dst_node_id, "unknown"))
    if src_kind != PROCESS_NODE_TYPE and dst_kind != PROCESS_NODE_TYPE:
        raise ValueError(
            f"Phase3E event has no process endpoint: event_id={event.event_id} "
            f"src={src_node_id}:{src_kind} dst={dst_node_id}:{dst_kind}",
        )
    actor_id = src_node_id
    object_id = dst_node_id
    object_type = dst_kind
    if src_kind != PROCESS_NODE_TYPE:
        actor_id = dst_node_id
        object_id = src_node_id
        object_type = src_kind
    info_src, info_dst, _rel, info_src_type, info_dst_type = information_flow(
        int(node_id_to_idx[actor_id]),
        int(node_id_to_idx[object_id]),
        action,
        object_type,
    )
    return (
        int(event.event_id),
        int(info_src),
        int(info_dst),
        int(action_name_to_id[action]),
        int(info_src_type),
        int(info_dst_type),
    )


def _phase3e_write_event_index_split_from_direct_db(
    *,
    conn: Any,
    year_month: str,
    days: Sequence[int],
    event_filter: bool,
    fetch_size: int,
    max_events: int,
    expected_count: int,
    path: Path,
    split: str,
    node_id_to_idx: Mapping[int, int],
    action_name_to_id: Mapping[str, int],
    node_kind_by_id: Mapping[int, str],
    config: SlimConfig,
    event_id_column: str,
    original_to_canonical: Mapping[int, int] | None = None,
) -> dict[str, object]:
    """Stream one split directly from event_table into event_index memmap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if int(expected_count) <= 0:
        path.write_bytes(b"")
        fingerprint = event_index_fingerprint(np.zeros((0,), dtype=EVENT_INDEX_DTYPE))
        return {
            "split": str(split),
            "row_order_is_stream_order": True,
            "order_by": _phase3e_json_order_by(PHASE3E_EVENT_ORDER_BY),
            "event_id_source": str(event_id_column),
            "operation_filter": "ORTHRUS10" if event_filter else "none",
            "num_events": 0,
            "path": str(path),
            "event_index_dtype": _phase3e_event_index_dtype_descriptor(),
            "dtype_descriptor": _phase3e_event_index_dtype_descriptor(),
            "event_index_fingerprint": fingerprint,
            "fingerprint": fingerprint,
            "event_count_source": "postgres_count_with_split_and_action_filter",
            "write_mode": "postgres_stream_to_memmap",
        }
    mapped = np.memmap(path, dtype=EVENT_INDEX_DTYPE, mode="w+", shape=(expected_count,))
    progress_interval = int(config.progress_interval_events)
    count = 0
    _stage_log(config, f"phase3e_{split}_event_index_stream_start")
    try:
        for event in _phase3e_stream_split_events_direct(
            conn,
            year_month=year_month,
            days=days,
            event_filter=event_filter,
            fetch_size=fetch_size,
            max_events=max_events,
            event_id_column=event_id_column,
            config=config,
            split=split,
        ):
            if count >= int(expected_count):
                raise ValueError(
                    f"Phase3E {split} stream exceeded Pass1 count: {expected_count}",
                )
            mapped[count] = _phase3e_compact_event_from_direct_stream(
                event,
                action_name_to_id,
                node_id_to_idx,
                node_kind_by_id,
                original_to_canonical=original_to_canonical,
            )
            count += 1
            if progress_interval > 0 and count % progress_interval == 0:
                _stage_log(config, f"phase3e_{split}_event_index_progress", count=count)
        if count != int(expected_count):
            raise ValueError(
                f"Phase3E {split} stream count mismatch: wrote {count}, "
                f"expected {expected_count}",
            )
        mapped.flush()
        fingerprint = event_index_fingerprint(np.asarray(mapped))
    finally:
        del mapped
    _stage_log(config, f"phase3e_{split}_event_index_stream_end", count=count)
    return {
        "split": str(split),
        "row_order_is_stream_order": True,
        "order_by": _phase3e_json_order_by(PHASE3E_EVENT_ORDER_BY),
        "event_id_source": str(event_id_column),
        "operation_filter": "ORTHRUS10" if event_filter else "none",
        "num_events": int(count),
        "path": str(path),
        "event_index_dtype": _phase3e_event_index_dtype_descriptor(),
        "dtype_descriptor": _phase3e_event_index_dtype_descriptor(),
        "event_index_fingerprint": fingerprint,
        "fingerprint": fingerprint,
        "event_count_source": "postgres_count_with_split_and_action_filter",
        "write_mode": "postgres_stream_to_memmap",
    }


def _phase3e_write_event_index_split(
    *,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    split: str,
    node_id_to_idx: Mapping[int, int],
    action_name_to_id: Mapping[str, int],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
    order_by: Sequence[object] | object,
    event_id_source: str,
) -> dict[str, object]:
    """Write one compact Phase3E event-index memmap without caching raw rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mapped = np.memmap(path, dtype=EVENT_INDEX_DTYPE, mode="w+", shape=(len(rows),))
    try:
        for index, row in enumerate(rows):
            compact = _phase3e_compact_event_index_row(row, config, process_cfg)
            src_node_id = int(compact["src_node_id"])
            dst_node_id = int(compact["dst_node_id"])
            action_name = str(compact["action_name"])
            if src_node_id not in node_id_to_idx:
                raise KeyError(f"missing src_node_idx for node id: {src_node_id}")
            if dst_node_id not in node_id_to_idx:
                raise KeyError(f"missing dst_node_idx for node id: {dst_node_id}")
            if action_name not in action_name_to_id:
                raise KeyError(f"missing action_id for action: {action_name}")
            mapped[index] = (
                int(compact["event_id"]),
                int(node_id_to_idx[src_node_id]),
                int(node_id_to_idx[dst_node_id]),
                int(action_name_to_id[action_name]),
                int(compact["src_type_id"]),
                int(compact["dst_type_id"]),
            )
        mapped.flush()
        fingerprint = event_index_fingerprint(np.asarray(mapped))
    finally:
        del mapped
    return {
        "split": str(split),
        "row_order_is_stream_order": True,
        "order_by": _phase3e_json_order_by(order_by),
        "event_id_source": str(event_id_source),
        "num_events": int(len(rows)),
        "path": str(path),
        "event_index_dtype": _phase3e_event_index_dtype_descriptor(),
        "dtype_descriptor": _phase3e_event_index_dtype_descriptor(),
        "event_index_fingerprint": fingerprint,
        "fingerprint": fingerprint,
    }


def build_phase3e_artifacts_from_db(config: SlimConfig) -> Path:
    """Build Phase3E node/action/event-index artifacts from label-free DB streams."""
    started = time.perf_counter()
    output_dir = Path(config.result_root) / config.out_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    process_cfg = _load_process_config(config)
    db_cfg = _cfg_for_dataset(config.dataset)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    split_metadata = _resolve_db_split_metadata(config, db_cfg)
    split_days = {
        "train": [int(day) for day in split_metadata["train_days"]],
        "validation": [int(day) for day in split_metadata["validation_days"]],
        "test": [int(day) for day in split_metadata["test_days"]],
    }
    max_events_by_split = {
        "train": int(config.max_train_events),
        "validation": int(config.max_ref_events),
        "test": int(config.max_test_events),
    }
    paths = _phase3e_artifact_paths(config)
    node_root = paths["node_embeddings"].parent
    action_root = paths["action_embeddings"].parent
    event_root = paths["event_meta"].parent
    adapter = ResidualWord2VecTokenAdapter(
        load_pretrained_residual_embedder(
            config.pretrained_residual_embedder_path,
            expected_dim=int(config.sspm_target_dim),
        ),
        str(config.pretrained_residual_embedder_path),
    )
    cur, conn = init_database_connection(db_cfg)
    try:
        node_maps = build_node_maps(cur)
        event_filter = bool(use_event_type_filter(db_cfg))
        event_id_column = _phase3e_detect_event_id_column(conn)
        pass1 = _phase3e_scan_used_nodes_pass1(
            conn=conn,
            year_month=str(db_cfg.dataset.year_month),
            split_days=split_days,
            event_filter=event_filter,
            fetch_size=int(config.fetch_size),
            max_events_by_split=max_events_by_split,
            event_id_column=event_id_column,
            config=config,
        )
        lookup_strategy = "temp_used_node_join"
        indexid2summary, meta_by_kind, lookup_seconds = _phase3e_fetch_used_node_lookup_joined(
            conn=conn,
            used_node_ids=set(pass1["used_node_ids"]),
        )
        canonical_summary: dict[str, object] = {}
        original_to_canonical: dict[int, int] = {}
        original_to_canonical_sidecar: dict[str, object] = {}
        if (
            is_optc_dataset(config.dataset)
            and str(getattr(config, "optc_netflow_node_canonicalization", "none"))
            == OPTC_NETFLOW_CANONICALIZATION_V1_3
        ):
            canonical_inputs = _phase3e_build_optc_canonical_artifact_inputs(
                used_node_ids=set(pass1["used_node_ids"]),
                split_used_nodes=pass1["split_used_nodes"],
                indexid2summary=indexid2summary,
                meta_by_kind=meta_by_kind,
                config=config,
                process_cfg=process_cfg,
            )
            node_table = build_node_embedding_table(
                node_tokens_by_id=canonical_inputs["canonical_tokens_by_id"],
                split_node_ids=canonical_inputs["canonical_split_used_nodes"],
                adapter=adapter,
                src_node_ids=set(),
                dst_node_ids=set(),
            )
            canonical_summary = dict(canonical_inputs["summary"])
            node_table.meta.update(canonical_summary)
            node_table.coverage.update(canonical_summary)
            original_to_canonical = dict(canonical_inputs["original_to_canonical"])
            node_kind_by_id = dict(canonical_inputs["canonical_kind_by_id"])
            original_to_canonical_sidecar = _phase3e_write_optc_original_to_canonical_sidecar(
                path=paths["event_meta"].with_name("original_to_canonical_netflow.csv"),
                original_to_canonical=original_to_canonical,
                indexid2summary=indexid2summary,
            )
            canonical_summary["original_to_canonical_sidecar"] = str(
                original_to_canonical_sidecar.get("path", ""),
            )
            canonical_summary["original_to_canonical_sidecar_rows"] = int(
                original_to_canonical_sidecar.get("row_count", 0),
            )
        else:
            node_table = _phase3e_build_node_table_from_used_lookup(
                used_node_ids=set(pass1["used_node_ids"]),
                split_used_nodes=pass1["split_used_nodes"],
                indexid2summary=indexid2summary,
                meta_by_kind=meta_by_kind,
                adapter=adapter,
                config=config,
                process_cfg=process_cfg,
            )
            node_kind_by_id = {
                int(node_id): normalize_piece(kind)
                for node_id, (kind, _summary) in indexid2summary.items()
            }
        action_table = build_action_embedding_table(adapter)
        node_paths = save_node_embedding_table(node_table, node_root)
        action_paths = save_action_embedding_table(action_table, action_root)
        node_id_to_idx = node_table.node_id_to_idx
        action_name_to_id = action_table.name_to_action_id
        split_meta: dict[str, dict[str, object]] = {}
        for split in ("train", "validation", "test"):
            split_meta[split] = _phase3e_write_event_index_split_from_direct_db(
                conn=conn,
                year_month=str(db_cfg.dataset.year_month),
                days=split_days[split],
                event_filter=event_filter,
                fetch_size=int(config.fetch_size),
                max_events=int(max_events_by_split[split]),
                expected_count=int(pass1["split_counts"].get(split, 0)),
                path=paths[f"event_index_{split}"],
                split=split,
                node_id_to_idx=node_id_to_idx,
                action_name_to_id=action_name_to_id,
                node_kind_by_id=node_kind_by_id,
                config=config,
                event_id_column=event_id_column,
                original_to_canonical=original_to_canonical,
            )
    finally:
        cur.close()
        conn.close()

    coverage_audit = build_node_action_coverage_audit(node_table, action_table)
    coverage_path = save_node_action_coverage_audit(coverage_audit, node_root)
    event_meta = {
        "schema_version": "phase3e_event_index_v1",
        "dataset": str(config.dataset),
        "semantic_mode": str(config.semantic_mode),
        "splits": split_meta,
        "split_metadata": split_metadata,
        "event_type_filter": bool(use_event_type_filter(db_cfg)),
        "event_filter": "Orthrus10" if use_event_type_filter(db_cfg) else "none",
        "node_embedding_meta_path": str(paths["node_meta"]),
        "action_embedding_meta_path": str(paths["action_meta"]),
        "node_embedding_path": str(paths["node_embeddings"]),
        "action_embedding_path": str(paths["action_embeddings"]),
        "word2vec_model_path": str(adapter.model_path),
        "word2vec_fingerprint": str(coverage_audit.get("word2vec_fingerprint", "")),
        "node_word2vec_source": "residual_pretrained",
        "node_lookup_seconds": float(lookup_seconds),
        "node_lookup_strategy": str(lookup_strategy),
        "elapsed_seconds": float(time.perf_counter() - started),
        "optc_netflow_canonicalization": canonical_summary,
        "original_to_canonical_sidecar": original_to_canonical_sidecar,
        "leakage_contract": {
            "labels_used_for_artifacts": False,
            "ground_truth_used": False,
            "test_labels_used_before_emission": False,
        },
    }
    paths["event_meta"].parent.mkdir(parents=True, exist_ok=True)
    paths["event_meta"].write_text(
        json.dumps(event_meta, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    eval_payload = {
        "dataset": str(config.dataset),
        "out_tag": str(config.out_tag),
        "phase3e_artifact_build_only": True,
        "semantic_mode": str(config.semantic_mode),
        "train_events_actual": int(split_meta["train"]["num_events"]),
        "validation_events_actual": int(split_meta["validation"]["num_events"]),
        "test_events_actual": int(split_meta["test"]["num_events"]),
        "outputs": {
            **node_paths,
            **action_paths,
            "event_index_meta": str(paths["event_meta"]),
            "event_index_train": str(paths["event_index_train"]),
            "event_index_validation": str(paths["event_index_validation"]),
            "event_index_test": str(paths["event_index_test"]),
            "coverage_audit": str(coverage_path),
        },
        "node_action_coverage": coverage_audit,
        "leakage_check": {
            "labels_used_for_artifacts": False,
            "ground_truth_used": False,
        },
    }
    eval_path = output_dir / "eval_causal_semantics_slim.json"
    eval_path.write_text(
        json.dumps(eval_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return eval_path


def _phase3e_json_order_by(order_by: Sequence[object] | object) -> list[str]:
    if isinstance(order_by, (str, bytes)):
        return [order_by.decode("utf-8") if isinstance(order_by, bytes) else order_by]
    try:
        return [str(value) for value in order_by]  # type: ignore[union-attr]
    except TypeError:
        return [str(order_by)]


def _phase3e_event_index_dtype_descriptor() -> list[list[str]]:
    return [[str(name), str(dtype)] for name, dtype in EVENT_INDEX_DTYPE.descr]


def _phase3e_array_fingerprint(values: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(values))
    payload = {
        "dtype": str(array.dtype),
        "shape": [int(value) for value in array.shape],
        "num_bytes": int(array.nbytes),
        "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }
    payload["fingerprint_sha256"] = stable_json_hash(payload)
    return payload


def _phase3e_file_fingerprint(path: str | Path) -> dict[str, object]:
    """Return a stable SHA-256 fingerprint for a precomputed Phase3E artifact."""
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(target),
        "num_bytes": int(target.stat().st_size),
        "bytes_sha256": digest.hexdigest(),
    }


def _phase3e_load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Phase3E JSON artifact must be an object: {path}")
    return value


def _phase3e_artifact_paths(config: SlimConfig) -> dict[str, Path]:
    node_dir = _phase3e_resolve_cache_dir(config.node_embedding_cache_dir, config)
    action_dir = _phase3e_resolve_cache_dir(config.action_embedding_cache_dir, config)
    event_dir = _phase3e_resolve_cache_dir(config.event_index_cache_dir, config)
    return {
        "node_embeddings": node_dir / "node_embeddings.npy",
        "node_meta": node_dir / "node_embedding_meta.json",
        "action_embeddings": action_dir / "action_embeddings.npy",
        "action_meta": action_dir / "action_embedding_meta.json",
        "event_index_train": event_dir / "event_index_train.memmap",
        "event_index_validation": event_dir / "event_index_validation.memmap",
        "event_index_test": event_dir / "event_index_test.memmap",
        "event_meta": event_dir / "event_index_meta.json",
    }


def _phase3e_active_artifact_paths(config: SlimConfig) -> dict[str, Path]:
    return _phase3e_artifact_paths(config)


def _phase3e_require_named_artifacts(paths: Mapping[str, Path], label: str) -> dict[str, Path]:
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        detail = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(f"{label} artifact missing: {detail}")
    return dict(paths)


def _phase3e_require_artifacts(config: SlimConfig) -> dict[str, Path]:
    """Require active Phase3E runtime artifacts."""
    return _phase3e_require_named_artifacts(
        _phase3e_active_artifact_paths(config),
        "Phase3E precompute",
    )


def _phase3e_require_legacy_head_path_disabled() -> None:
    raise RuntimeError(
        "The Phase3E X_context/low-rank-head path is legacy-only in this cleanup branch. "
        "Use legacy.compatibility.phase3e_context_memmap and "
        "legacy.compatibility.phase3e_head_training outside the active E4 best-path runner.",
    )


def _phase3e_require_legacy_precompute_path_disabled() -> None:
    raise RuntimeError(
        "Phase3E precompute/X_context generation is legacy-only in this cleanup branch. "
        "The active E4 best path consumes existing Phase3E event-index/node/action artifacts "
        "and scores with the Phase3G conditional head.",
    )


def _phase3g_effective_artifacts(config: SlimConfig) -> tuple[dict[str, Path], dict[str, Any]]:
    """Return Phase3E artifacts, optionally remapped to compact used-node embeddings."""
    paths = _phase3e_require_artifacts(config)
    event_meta = _phase3e_load_json(paths["event_meta"])
    lookup_mode = str(config.node_embedding_lookup_mode)
    if lookup_mode == "global_memmap":
        return paths, event_meta
    build_compact = lookup_mode == "compact_used_nodes" or (
        lookup_mode == "lazy_mmap_lru"
        and str(config.node_embedding_lazy_backing) == "compact_used_nodes"
    )
    if not build_compact:
        if lookup_mode == "lazy_mmap_lru" and str(config.node_embedding_lazy_backing) == "global_memmap":
            lazy_event_meta = json.loads(json.dumps(event_meta))
            lazy_event_meta["node_embedding_lookup_mode"] = "lazy_mmap_lru"
            lazy_event_meta["node_embedding_lazy_backing"] = "global_memmap"
            return paths, lazy_event_meta
        raise ValueError(
            f"unsupported NODE_EMBEDDING_LOOKUP_MODE: {config.node_embedding_lookup_mode}",
        )
    compact_dir = _phase3e_resolve_cache_dir(config.compact_used_node_cache_dir, config)
    split_indexes: dict[str, np.memmap] = {}
    split_counts: dict[str, int] = {}
    try:
        for split in ("train", "validation", "test"):
            index, count = _phase3e_open_split_event_index(paths, event_meta, split)
            split_indexes[split] = index
            split_counts[split] = int(count)
        node_embeddings = np.load(paths["node_embeddings"], mmap_mode="r")
        expected_node_fp = source_node_embedding_fingerprint(
            paths["node_embeddings"],
            node_embeddings,
        )
        expected_event_fps = source_event_index_fingerprints(split_indexes)
        needs_compact_build = not (compact_dir / "compact_embedding_meta.json").exists()
        if not (compact_dir / "node_id_to_idx.pkl").exists():
            needs_compact_build = True
        if needs_compact_build:
            node_id_to_idx_path = Path(paths["node_embeddings"]).with_name("node_id_to_idx.pkl")
            with node_id_to_idx_path.open("rb") as handle:
                node_id_to_idx = pickle.load(handle)
            build_compact_used_node_artifacts(
                node_embeddings=node_embeddings,
                split_event_indexes=split_indexes,
                output_dir=compact_dir,
                source_node_embedding_path=paths["node_embeddings"],
                source_event_index_paths={
                    split: paths[f"event_index_{split}"] for split in split_indexes
                },
                source_node_id_to_idx=node_id_to_idx,
            )
        compact_paths = load_compact_used_node_artifacts(
            compact_dir,
            expected_source_node_embedding_fingerprint=expected_node_fp,
            expected_source_event_index_fingerprints=expected_event_fps,
        )
    finally:
        for index in split_indexes.values():
            del index
    effective = dict(paths)
    effective["node_embeddings"] = Path(compact_paths["node_embeddings"])
    if "node_id_to_idx" in compact_paths:
        effective["node_id_to_idx"] = Path(compact_paths["node_id_to_idx"])
    for split in ("train", "validation", "test"):
        key = f"event_index_{split}"
        if key in compact_paths:
            effective[key] = Path(compact_paths[key])
    effective["compact_embedding_meta"] = Path(compact_paths["compact_embedding_meta"])
    effective["phase3g_used_node_count_summary"] = Path(
        compact_paths["phase3g_used_node_count_summary"],
    )
    compact_event_meta = json.loads(json.dumps(event_meta))
    compact_event_meta["node_embedding_lookup_mode"] = lookup_mode
    if lookup_mode == "lazy_mmap_lru":
        compact_event_meta["node_embedding_lazy_backing"] = "compact_used_nodes"
    compact_event_meta["compact_used_node_cache_dir"] = str(compact_dir)
    for split, count in split_counts.items():
        compact_event_meta.setdefault("splits", {}).setdefault(split, {})["num_events"] = count
    return effective, compact_event_meta


def _phase3e_train_lr(config: SlimConfig) -> float:
    lr = float(config.sspm_torch_lr)
    if lr > 0.0:
        return lr
    return float(config.sspm_learning_rate)


def _phase3e_open_precompute_arrays(
    config: SlimConfig,
) -> tuple[dict[str, Path], dict[str, Any], np.memmap, np.ndarray, np.ndarray]:
    """Open Phase3E train event-index, node embeddings, and action embeddings lazily."""
    paths = _phase3e_require_artifacts(config)
    event_meta = _phase3e_load_json(paths["event_meta"])
    train_meta = dict(dict(event_meta.get("splits", {})).get("train", {}))
    train_count = int(train_meta.get("num_events", 0))
    if train_count <= 0:
        raise ValueError("Phase3E train_base requires non-empty train event_index")
    train_index = open_event_index_memmap(paths["event_index_train"], num_events=train_count)
    node_embeddings = np.load(paths["node_embeddings"], mmap_mode="r")
    action_embeddings = np.load(paths["action_embeddings"], mmap_mode="r")
    if int(node_embeddings.shape[1]) != int(config.sspm_target_dim):
        raise ValueError("Phase3E node embedding dim does not match SSPM target dim")
    if int(action_embeddings.shape[1]) != int(config.sspm_target_dim):
        raise ValueError("Phase3E action embedding dim does not match SSPM target dim")
    return paths, event_meta, train_index, node_embeddings, action_embeddings


def _phase3e_open_split_event_index(
    paths: Mapping[str, Path],
    event_meta: Mapping[str, Any],
    split: str,
    max_events: int = 0,
    event_offset: int = 0,
) -> tuple[np.memmap, int]:
    """Open one Phase3E event-index split without materializing rows."""
    split_meta = dict(dict(event_meta.get("splits", {})).get(str(split), {}))
    total_count = int(split_meta.get("num_events", 0))
    if total_count <= 0:
        raise ValueError(f"Phase3E {split} event_index is empty")
    offset = int(event_offset)
    if offset < 0:
        raise ValueError(f"Phase3E {split} event_offset must be non-negative, got {offset}")
    count = max(total_count - offset, 0)
    if int(max_events) > 0:
        count = min(count, int(max_events))
    key = f"event_index_{split}"
    if key not in paths:
        raise KeyError(f"Phase3E artifact path missing: {key}")
    full_index = open_event_index_memmap(paths[key], num_events=total_count)
    return full_index[offset : offset + count], count


def _phase3g_open_node_embeddings(config: SlimConfig, paths: Mapping[str, Path]) -> Any:
    if str(config.node_embedding_lookup_mode) != "lazy_mmap_lru":
        return np.load(paths["node_embeddings"], mmap_mode="r")
    return LazyMmapLruEmbeddingLookup(
        paths["node_embeddings"],
        max_nodes=int(config.node_embedding_cache_max_nodes),
        backing=str(config.node_embedding_lazy_backing),
    )



def _phase3e_fit_checkpoint_calibration_from_validation(
    *,
    config: SlimConfig,
    model: SSPMLowRankModel,
    validation_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
) -> dict[str, Any]:
    """Fit validation-only calibration/gate state needed by Phase3E checkpoints."""
    records = _phase3e_event_index_array(validation_index)
    if int(records.shape[0]) <= 0:
        raise ValueError("Phase3E checkpoint calibration requires validation events")
    progress_interval = int(config.progress_interval_events)
    stats_payload: dict[str, Any] = {
        "validation_events_actual": int(records.shape[0]),
        "validation_two_pass": False,
    }

    if str(config.sspm_residual_score_mode) == "var_calibrated":
        if str(config.sspm_residual_calibration) != "action_diag":
            raise ValueError(
                "Phase3E var_calibrated checkpoint requires "
                "SSPM_RESIDUAL_CALIBRATION=action_diag",
            )
        calibration_stats = ResidualCalibrationStats(
            latent_dim=int(model.latent_dim),
            epsilon=float(config.sspm_residual_var_eps),
            min_count=int(config.sspm_residual_calibration_min_count),
        )
        model.reset_state()
        _stage_log(
            config,
            "phase3e_validation_calibration_fit_start",
            count=int(records.shape[0]),
            update_gate_enabled=False,
        )
        for index, row in enumerate(records, start=1):
            fields = _phase3e_fields_from_index_row(row)
            z = _phase3e_target_from_index_row(row, node_embeddings, action_embeddings)
            pred = model.predict(fields)
            calibration_stats.update(
                fields.get("raw_action", fields.get("action")),
                z_hat=pred,
                z_true=z,
            )
            model.update_states(fields, z, enable_update_gate=False)
            if progress_interval > 0 and index % progress_interval == 0:
                _stage_log(
                    config,
                    "phase3e_validation_calibration_fit_progress",
                    count=index,
                )
        calibration_model = calibration_stats.finalize()
        model.set_residual_calibration(calibration_model)
        model.reset_state()
        stats_payload["validation_two_pass"] = True
        stats_payload["calibration"] = calibration_model.summary()
        _stage_log(
            config,
            "phase3e_validation_calibration_fit_end",
            count=int(records.shape[0]),
        )
    elif str(config.sspm_residual_calibration) != "none":
        raise ValueError(
            "Phase3E checkpoint calibration only supports none or "
            "var_calibrated/action_diag",
        )

    if (
        str(config.sspm_update_gate_mode) == "quantile"
        or str(config.sspm_residual_score_mode) == "var_calibrated"
    ):
        _stage_log(
            config,
            "phase3e_validation_score_fit_start",
            count=int(records.shape[0]),
            update_gate_enabled=False,
        )
        validation_scores, validation_count = _phase3e_score_event_index_stream(
            config,
            model,
            records,
            node_embeddings,
            action_embeddings,
            enable_update_gate=False,
        )
        if int(validation_count) <= 0:
            raise ValueError("Phase3E validation scoring produced no scores")
        _fit_update_gate_if_needed(config, model, validation_scores)
        stats_payload.update(
            {
                "validation_score_count": int(validation_count),
                "validation_score_min": float(np.min(validation_scores)),
                "validation_score_max": float(np.max(validation_scores)),
                "validation_score_mean": float(np.mean(validation_scores)),
                "update_gate": _model_update_gate_summary(config, model),
            },
        )
        _stage_log(
            config,
            "phase3e_validation_score_fit_end",
            count=int(validation_count),
            score_max=f"{float(np.max(validation_scores)):.6f}",
        )
    return stats_payload


def _phase3e_train_base_output_payload(
    *,
    config: SlimConfig,
    checkpoint_path: Path,
    train_stats: Mapping[str, Any],
    train_count: int,
    validation_count: int,
    output_dir: Path,
    started: float,
) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "method_name": "causal_semantics_slim",
        "method_version": METHOD_VERSION,
        "phase3e_train_base_only": True,
        "dataset": str(config.dataset),
        "out_tag": str(config.out_tag),
        "checkpoint_path": str(checkpoint_path),
        "train_events_actual": int(train_count),
        "validation_events_actual": int(validation_count),
        "test_events_actual": 0,
        "train_seconds": float(train_stats.get("train_seconds", elapsed)),
        "runtime": {
            "elapsed_seconds": float(elapsed),
            "events_scored": 0,
            "throughput_events_per_second": 0.0,
        },
        "memory": {
            "train_rss_peak_mb": float(
                train_stats.get("train_rss_peak_mb", _current_rss_mb()),
            ),
            "rss_final_mb": float(_current_rss_mb()),
            "deploy_rss_valid": False,
        },
        "phase3e": {
            "target_mode": str(config.sspm_target_mode),
            "node_word2vec_source": str(config.node_word2vec_source),
            "train_backend": str(config.sspm_train_backend),
            "infer_backend": str(config.sspm_infer_backend),
        },
        "sspm_training": dict(train_stats),
        "leakage_check": {
            "labels_used_for_training": False,
            "checkpoint_contains_train_state_memory": False,
            "test_labels_used_before_emission": False,
            "test_rankings_used_before_emission": False,
        },
        "outputs": {
            "eval_json": str(output_dir / "eval_causal_semantics_slim.json"),
            "metrics_json": str(output_dir / "metrics.json"),
            "checkpoint": str(checkpoint_path),
        },
    }


def _phase3e_write_base_profile_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    """Write Phase3E base profile interval rows and aggregate summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase3e_base_profile.csv"
    summary_path = output_dir / "phase3e_base_profile_summary.json"
    fields = [
        "dataset",
        "out_tag",
        "event_count",
        "events_per_second",
        "target_lookup_seconds",
        "make_context_seconds",
        "update_states_seconds",
        "batch_stack_seconds",
        "train_batch_step_seconds",
        "rss_mb",
        "active_state_nodes",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    summary = {
        "interval_count": int(len(rows)),
        "total_events": int(max((int(row.get("event_count", 0)) for row in rows), default=0)),
        "total_target_lookup_seconds": float(
            sum(float(row.get("target_lookup_seconds", 0.0)) for row in rows),
        ),
        "total_make_context_seconds": float(
            sum(float(row.get("make_context_seconds", 0.0)) for row in rows),
        ),
        "total_update_states_seconds": float(
            sum(float(row.get("update_states_seconds", 0.0)) for row in rows),
        ),
        "total_batch_stack_seconds": float(
            sum(float(row.get("batch_stack_seconds", 0.0)) for row in rows),
        ),
        "total_train_batch_step_seconds": float(
            sum(float(row.get("train_batch_step_seconds", 0.0)) for row in rows),
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "summary": summary_path}


def train_phase3e_real_diag_from_event_index(
    *,
    config: SlimConfig,
    event_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
) -> tuple[SSPMLowRankModel, dict[str, Any]]:
    """Train Phase3E E3 from compact event_index with online gamma sensitivity."""
    if str(config.sspm_state_model) != "real_diag_learnable":
        raise ValueError("Phase3E real_diag train requires sspm_state_model=real_diag_learnable")
    records = _phase3e_event_index_array(event_index)
    model = SSPMLowRankModel(_make_sspm_config(config, None))
    progress_interval = int(config.progress_interval_events)
    train_rss_peak_mb = _current_rss_mb()
    started = time.perf_counter()
    epochs = max(int(config.sspm_epochs), 1)
    min_delta = float(config.sspm_early_stop_min_delta)
    patience = int(config.sspm_early_stop_patience)
    best_loss = float("inf")
    plateau_epochs = 0
    loss_by_epoch: list[float] = []
    train_events_seen_total = 0
    stopped_early = False
    stop_reason = ""

    _stage_log(
        config,
        "phase3e_e3_train_event_index_start",
        count=int(records.shape[0]),
        real_diag_train_gamma=True,
        epochs=epochs,
    )
    stats: dict[str, Any] = {}
    for epoch in range(epochs):
        epoch_count = 0
        _stage_log(
            config,
            "phase3e_e3_train_event_index_epoch_start",
            epoch=epoch + 1,
            epochs=epochs,
            count=int(records.shape[0]),
        )

        def encoded_items():
            nonlocal epoch_count, train_events_seen_total, train_rss_peak_mb
            for index, row in enumerate(records, start=1):
                fields = _phase3e_fields_from_index_row(row)
                z = _phase3e_target_from_index_row(row, node_embeddings, action_embeddings)
                epoch_count += 1
                train_events_seen_total += 1
                if progress_interval > 0 and index % progress_interval == 0:
                    _stage_log(
                        config,
                        "phase3e_e3_train_event_index_progress",
                        epoch=epoch + 1,
                        count=index,
                    )
                train_rss_peak_mb = max(float(train_rss_peak_mb), _current_rss_mb())
                yield fields, z

        stats = dict(
            model.train_real_diag_online(
                encoded_items(),
                log_prefix=f"[SSPM][{config.dataset}][PHASE3E_E3_GAMMA]",
            ),
        )
        epoch_loss = float(stats.get("final_loss", stats.get("loss", 0.0)))
        loss_by_epoch.append(epoch_loss)
        _stage_log(
            config,
            "phase3e_e3_train_event_index_epoch_end",
            epoch=epoch + 1,
            count=epoch_count,
            loss=f"{epoch_loss:.6f}",
        )
        if best_loss - epoch_loss > min_delta:
            best_loss = epoch_loss
            plateau_epochs = 0
        else:
            plateau_epochs += 1
            if patience > 0 and plateau_epochs >= patience:
                stopped_early = True
                stop_reason = "loss_plateau"
                break
    stats = dict(stats)
    stats.update(model.real_diag_gamma_stats())
    stats["loss_by_epoch"] = list(loss_by_epoch)
    stats["loss"] = float(np.mean(loss_by_epoch)) if loss_by_epoch else 0.0
    stats["final_loss"] = float(loss_by_epoch[-1]) if loss_by_epoch else 0.0
    stats["epochs"] = int(epochs)
    stats["configured_epochs"] = int(epochs)
    stats["epochs_completed"] = int(len(loss_by_epoch))
    stats["train_epochs_completed"] = int(len(loss_by_epoch))
    stats["stopped_early"] = bool(stopped_early)
    stats["stop_reason"] = str(stop_reason)
    stats["early_stop_reason"] = str(stop_reason)
    stats["early_stop_min_delta"] = float(min_delta)
    stats["early_stop_patience"] = int(patience)
    stats["train_seconds"] = float(time.perf_counter() - started)
    stats["train_rss_peak_mb"] = float(train_rss_peak_mb)
    stats["train_events_actual"] = int(records.shape[0])
    stats["train_events_seen_total"] = int(train_events_seen_total)
    stats["batch_events"] = 1
    model.training_stats = dict(stats)
    model.real_diag_gamma_training_stats = dict(stats)
    _stage_log(
        config,
        "phase3e_e3_train_event_index_end",
        count=int(stats.get("train_events_actual", records.shape[0])),
    )
    return model, stats


def _phase3e_train_base_from_event_index(
    *,
    config: SlimConfig,
    train_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
) -> tuple[SSPMLowRankModel, dict[str, Any]]:
    """Train a Phase3E base SSPM checkpoint from compact event-index records."""
    if bool(config.real_diag_train_gamma):
        return train_phase3e_real_diag_from_event_index(
            config=config,
            event_index=train_index,
            node_embeddings=node_embeddings,
            action_embeddings=action_embeddings,
        )
    records = _phase3e_event_index_array(train_index)
    model = SSPMLowRankModel(_make_sspm_config(config, None))
    progress_interval = int(config.progress_interval_events)
    profile_enabled = bool(getattr(config, "phase3e_base_profile", False))
    profile_interval = max(int(getattr(config, "phase3e_profile_interval_events", 100000)), 1)
    profile_rows: list[dict[str, Any]] = []
    profile_target_seconds = 0.0
    profile_context_seconds = 0.0
    profile_update_seconds = 0.0
    profile_batch_stack_seconds = 0.0
    profile_train_step_seconds = 0.0
    profile_events_since_flush = 0
    profile_events_seen_total = 0
    profile_interval_started = time.perf_counter()
    train_rss_peak_mb = _current_rss_mb()
    started = time.perf_counter()

    def profile_batch_callback(row: Mapping[str, Any]) -> None:
        nonlocal profile_batch_stack_seconds
        nonlocal profile_train_step_seconds
        profile_batch_stack_seconds += float(row.get("batch_stack_seconds", 0.0))
        profile_train_step_seconds += float(row.get("train_batch_step_seconds", 0.0))

    def rows_for_epoch(epoch: int):
        nonlocal profile_batch_stack_seconds
        nonlocal profile_context_seconds
        nonlocal profile_events_seen_total
        nonlocal profile_events_since_flush
        nonlocal profile_interval_started
        nonlocal profile_train_step_seconds
        nonlocal profile_target_seconds
        nonlocal profile_update_seconds
        nonlocal train_rss_peak_mb
        epoch_count = 0
        model.reset_state_memory()
        _stage_log(
            config,
            "phase3e_train_base_epoch_start",
            epoch=int(epoch) + 1,
            epochs=int(config.sspm_epochs),
            count=int(records.shape[0]),
        )
        for index, row in enumerate(records, start=1):
            fields = _phase3e_fields_from_index_row(row)
            target_started = time.perf_counter()
            z = _phase3e_target_from_index_row(row, node_embeddings, action_embeddings)
            profile_target_seconds += time.perf_counter() - target_started
            context_started = time.perf_counter()
            context = model.make_context(fields)
            profile_context_seconds += time.perf_counter() - context_started
            update_started = time.perf_counter()
            model.update_states(fields, z)
            profile_update_seconds += time.perf_counter() - update_started
            epoch_count += 1
            profile_events_seen_total += 1
            profile_events_since_flush += 1
            train_rss_peak_mb = max(float(train_rss_peak_mb), _current_rss_mb())
            if progress_interval > 0 and index % progress_interval == 0:
                _stage_log(
                    config,
                    "phase3e_train_base_epoch_progress",
                    epoch=int(epoch) + 1,
                    count=index,
                )
            if profile_enabled and index % profile_interval == 0:
                elapsed = max(time.perf_counter() - profile_interval_started, 1e-9)
                profile_rows.append(
                    {
                        "dataset": str(config.dataset),
                        "out_tag": str(config.out_tag),
                        "event_count": int(profile_events_seen_total),
                        "events_per_second": float(profile_events_since_flush / elapsed),
                        "target_lookup_seconds": float(profile_target_seconds),
                        "make_context_seconds": float(profile_context_seconds),
                        "update_states_seconds": float(profile_update_seconds),
                        "batch_stack_seconds": float(profile_batch_stack_seconds),
                        "train_batch_step_seconds": float(profile_train_step_seconds),
                        "rss_mb": float(_current_rss_mb()),
                        "active_state_nodes": int(len(getattr(model, "node_to_slot", {}))),
                    },
                )
                profile_target_seconds = 0.0
                profile_context_seconds = 0.0
                profile_update_seconds = 0.0
                profile_batch_stack_seconds = 0.0
                profile_train_step_seconds = 0.0
                profile_events_since_flush = 0
                profile_interval_started = time.perf_counter()
            yield context, z
        _stage_log(
            config,
            "phase3e_train_base_epoch_end",
            epoch=int(epoch) + 1,
            count=epoch_count,
        )

    _stage_log(
        config,
        "phase3e_train_base_start",
        count=int(records.shape[0]),
        epochs=int(config.sspm_epochs),
        batch_events=int(config.sspm_train_batch_events),
    )
    stats = dict(
        model.train_stream_batches(
            rows_for_epoch,
            epochs=int(config.sspm_epochs),
            batch_events=int(config.sspm_train_batch_events),
            log_prefix=f"[SSPM][{config.dataset}][PHASE3E_BASE]",
            profile_callback=profile_batch_callback if profile_enabled else None,
        ),
    )
    stats["train_seconds"] = float(time.perf_counter() - started)
    stats["train_rss_peak_mb"] = float(train_rss_peak_mb)
    stats["train_events_actual"] = int(records.shape[0])
    if profile_enabled:
        if (
            profile_events_since_flush > 0
            or profile_target_seconds > 0.0
            or profile_context_seconds > 0.0
            or profile_update_seconds > 0.0
            or profile_batch_stack_seconds > 0.0
            or profile_train_step_seconds > 0.0
        ):
            elapsed = max(time.perf_counter() - profile_interval_started, 1e-9)
            profile_rows.append(
                {
                    "dataset": str(config.dataset),
                    "out_tag": str(config.out_tag),
                    "event_count": int(profile_events_seen_total),
                    "events_per_second": float(profile_events_since_flush / elapsed),
                    "target_lookup_seconds": float(profile_target_seconds),
                    "make_context_seconds": float(profile_context_seconds),
                    "update_states_seconds": float(profile_update_seconds),
                    "batch_stack_seconds": float(profile_batch_stack_seconds),
                    "train_batch_step_seconds": float(profile_train_step_seconds),
                    "rss_mb": float(_current_rss_mb()),
                    "active_state_nodes": int(len(getattr(model, "node_to_slot", {}))),
                },
            )
        stats["phase3e_base_profile_rows"] = profile_rows
    stats.setdefault("early_stop_reason", stats.get("stop_reason", ""))
    model.reset_state()
    model.training_stats = dict(stats)
    _stage_log(config, "phase3e_train_base_end", count=int(records.shape[0]))
    return model, stats


def _phase3e_run_train_base_checkpoint(config: SlimConfig) -> Path:
    """Train and save a Phase3E base SSPM checkpoint from existing artifacts."""
    started = time.perf_counter()
    output_dir = Path(config.result_root) / config.out_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, event_meta, train_index, node_embeddings, action_embeddings = (
        _phase3e_open_precompute_arrays(config)
    )
    validation_index, validation_count = _phase3e_open_split_event_index(
        paths,
        event_meta,
        "validation",
        max_events=int(config.max_ref_events),
    )
    model, train_stats = _phase3e_train_base_from_event_index(
        config=config,
        train_index=train_index,
        node_embeddings=node_embeddings,
        action_embeddings=action_embeddings,
    )
    validation_stats = _phase3e_fit_checkpoint_calibration_from_validation(
        config=config,
        model=model,
        validation_index=validation_index,
        node_embeddings=node_embeddings,
        action_embeddings=action_embeddings,
    )
    train_stats = {**dict(train_stats), "validation": dict(validation_stats)}
    checkpoint_path = Path(config.sspm_checkpoint_path)
    if not str(checkpoint_path):
        checkpoint_path = (
            Path(config.sspm_base_checkpoint_root)
            / f"{config.dataset}_E4_PHASE3E_BASE_FULL.pkl"
        )
    checkpoint_path = save_sspm_checkpoint(
        checkpoint_path,
        config=config,
        model=model,
        embedder=load_pretrained_residual_embedder(
            config.pretrained_residual_embedder_path,
            expected_dim=int(config.sspm_target_dim),
        ),
        process_cfg=None,
        train_count=int(train_index.shape[0]),
        validation_count=int(validation_count),
        phase3e_meta={
            "event_index_meta_path": str(paths["event_meta"]),
            "node_embedding_meta_path": str(paths["node_meta"]),
            "action_embedding_meta_path": str(paths["action_meta"]),
            "word2vec_model_path": str(config.pretrained_residual_embedder_path),
        },
    )
    eval_payload = _phase3e_train_base_output_payload(
        config=config,
        checkpoint_path=checkpoint_path,
        train_stats=train_stats,
        train_count=int(train_index.shape[0]),
        validation_count=int(validation_count),
        output_dir=output_dir,
        started=started,
    )
    if bool(getattr(config, "phase3e_base_profile", False)):
        profile_paths = _phase3e_write_base_profile_outputs(
            output_dir,
            list(train_stats.get("phase3e_base_profile_rows", [])),
        )
        eval_payload.setdefault("outputs", {}).update(
            {
                "phase3e_base_profile_csv": str(profile_paths["csv"]),
                "phase3e_base_profile_summary": str(profile_paths["summary"]),
            },
        )
    eval_path = output_dir / "eval_causal_semantics_slim.json"
    eval_path.write_text(
        json.dumps(eval_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    write_effective_config(
        output_dir,
        config,
        model,
        embedder_loaded=True,
        train_count=int(train_index.shape[0]),
        validation_count=int(validation_count),
        test_count=0,
    )
    write_metrics_json(
        output_dir,
        config,
        model,
        eval_payload,
        train_count=int(train_index.shape[0]),
        validation_count=int(validation_count),
        test_count=0,
        embedder_loaded=True,
    )
    return eval_path


def run_phase3e_train_base_from_precompute(config: SlimConfig) -> Path:
    """Train one Phase3E base checkpoint from existing precompute artifacts."""
    if str(config.sspm_score_head) != "conditional_action_semantic":
        _phase3e_require_legacy_head_path_disabled()
    return _phase3e_run_train_base_checkpoint(config)


def _phase3e_config_for_state_model(config: SlimConfig, state_model: str) -> SlimConfig:
    values = asdict(config)
    values["sspm_state_model"] = str(state_model)
    values["real_diag_train_gamma"] = False
    return SlimConfig(**values)


def _phase3e_fields_from_compact_event(
    compact: Mapping[str, object],
) -> dict[str, Any]:
    """Map one compact DB event into SSPM context fields with indexed node ids."""
    action_id = int(compact["action_id"])
    action_name = str(ORTHRUS10_ACTION_NAMES[action_id])
    src_idx = int(compact["src_node_idx"])
    dst_idx = int(compact["dst_node_idx"])
    src_type = int(compact["src_type_id"])
    dst_type = int(compact["dst_type_id"])
    return {
        "info_src": src_idx,
        "info_dst": dst_idx,
        "src_type": src_type,
        "dst_type": dst_type,
        "info_src_type": src_type,
        "info_dst_type": dst_type,
        "src_type_name": _phase3e_entity_type_name("src_type_id", src_type),
        "dst_type_name": _phase3e_entity_type_name("dst_type_id", dst_type),
        "src_role": _phase3e_entity_type_name("src_type_id", src_type),
        "dst_role": _phase3e_entity_type_name("dst_type_id", dst_type),
        "action_id": action_id,
        "relation_id": action_id,
        "raw_action": action_name,
        "action": action_name,
        "action_token": raw_action_token(action_name),
    }


def _phase3e_target_from_fields(
    fields: Mapping[str, Any],
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
) -> np.ndarray:
    """Build one Phase3E target from already-indexed streaming fields."""
    return compute_node_action_target(
        node_embeddings,
        action_embeddings,
        int(fields["info_src"]),
        int(fields["info_dst"]),
        int(fields["action_id"]),
    )


def _uses_action_diag_calibration(config: SlimConfig) -> bool:
    return (
        str(config.sspm_residual_score_mode) == "var_calibrated"
        and str(config.sspm_residual_calibration) == "action_diag"
    )


def _calibration_disabled_summary(config: SlimConfig) -> dict[str, Any]:
    return {
        "calibration_fitted": False,
        "calibration_actions": [],
        "calibration_actions_count_by_action": {},
        "calibration_actions_global_fallback": {},
        "calibration_num_validation_events": 0,
        "global_count": 0,
        "min_count": int(config.sspm_residual_calibration_min_count),
        "fallback_actions": [],
        "fallback_actions_count": 0,
        "warnings": [],
    }


def _model_calibration_summary(config: SlimConfig, model: SSPMLowRankModel) -> dict[str, Any]:
    calibration = getattr(model, "residual_calibration_model", None)
    if calibration is None:
        return _calibration_disabled_summary(config)
    raw_summary = calibration.summary()
    actions = dict(raw_summary.get("actions", {}))
    return {
        "calibration_fitted": bool(raw_summary.get("calibration_fitted", False)),
        "calibration_actions": list(raw_summary.get("calibration_actions", [])),
        "calibration_actions_count_by_action": {
            str(action): int(value.get("count", 0))
            for action, value in actions.items()
            if isinstance(value, Mapping)
        },
        "calibration_actions_global_fallback": {
            str(action): bool(value.get("global_fallback", False))
            for action, value in actions.items()
            if isinstance(value, Mapping)
        },
        "calibration_num_validation_events": int(raw_summary.get("global_count", 0)),
        "global_count": int(raw_summary.get("global_count", 0)),
        "min_count": int(raw_summary.get("min_count", config.sspm_residual_calibration_min_count)),
        "fallback_actions": list(raw_summary.get("fallback_actions", [])),
        "fallback_actions_count": int(raw_summary.get("fallback_actions_count", 0)),
        "warnings": list(raw_summary.get("warnings", [])),
    }


def _update_gate_disabled_summary(config: SlimConfig) -> dict[str, Any]:
    return {
        "mode": str(config.sspm_update_gate_mode),
        "quantile": float(config.sspm_update_gate_quantile),
        "threshold_mode": str(config.sspm_update_gate_threshold_mode),
        "q_min": float(config.sspm_update_gate_q_min),
        "eta": float(config.sspm_update_gate_eta),
        "threshold": None,
        "num_validation_events": 0,
        "fitted": False,
    }


def _model_update_gate_summary(config: SlimConfig, model: SSPMLowRankModel) -> dict[str, Any]:
    calibrator = getattr(model, "update_gate_calibrator", None)
    if calibrator is None:
        return _update_gate_disabled_summary(config)
    return calibrator.summary()


def _fit_update_gate_if_needed(
    config: SlimConfig,
    model: SSPMLowRankModel,
    validation_residual_scores: Sequence[float],
) -> None:
    if str(config.sspm_update_gate_mode) == "none":
        model.set_update_gate_calibrator(None)
        return
    if str(config.sspm_update_gate_mode) != "quantile":
        raise ValueError(f"unknown SSPM update gate mode: {config.sspm_update_gate_mode}")
    gate = UpdateGateCalibrator(
        quantile=float(config.sspm_update_gate_quantile),
        q_min=float(config.sspm_update_gate_q_min),
        eta=float(config.sspm_update_gate_eta),
        threshold_mode=str(config.sspm_update_gate_threshold_mode),
    )
    gate.fit(validation_residual_scores)
    model.set_update_gate_calibrator(gate)


def fit_residual_calibration_from_stream(
    rows,
    config: SlimConfig,
    model: SSPMLowRankModel,
    embedder: ResidualEmbedder,
    process_cfg: ProcessSemanticConfig | None = None,
    process_audit: ProcessSemanticAudit | None = None,
) -> int:
    """Validation pass 1: fit action_diag residual variance with q_t forced to one."""
    stats = ResidualCalibrationStats(
        latent_dim=int(model.latent_dim),
        epsilon=float(config.sspm_residual_var_eps),
        min_count=int(config.sspm_residual_calibration_min_count),
    )
    count = 0
    progress_interval = int(config.progress_interval_events)
    model.reset_state()
    _stage_log(config, "validation_calibration_fit_start")
    for row in rows:
        if process_cfg is not None:
            _observe_process_semantic_audit(
                process_audit,
                "validation",
                row,
                config.dataset,
                process_cfg,
                config.max_tokens_per_node,
            )
        fields = row_fields(
            row,
            config.max_tokens_per_node,
            dataset=config.dataset,
            process_semantic_config=process_cfg,
        )
        z = _encode_row(embedder, row, config, process_cfg)
        pred = model.predict(fields)
        action = fields.get("raw_action", fields.get("action"))
        stats.update(action, z_hat=pred, z_true=z)
        model.update_states(fields, z, enable_update_gate=False)
        count += 1
        if progress_interval > 0 and count % progress_interval == 0:
            _stage_log(config, "validation_calibration_fit_progress", count=count)
    calibration_model = stats.finalize()
    model.set_residual_calibration(calibration_model)
    model.reset_state()
    _stage_log(config, "validation_calibration_fit_end", count=count)
    return int(count)


def _embedder_loaded(config: SlimConfig) -> bool:
    return bool(str(config.pretrained_residual_embedder_path).strip())


def write_effective_config(
    output_dir: Path,
    config: SlimConfig,
    sspm: SSPMLowRankModel,
    embedder_loaded: bool,
    train_count: int | None = None,
    validation_count: int | None = None,
    test_count: int | None = None,
) -> Path:
    """Write effective runtime configuration without secrets."""
    payload = {
        "config": asdict(config),
        "sspm_config": asdict(sspm.config),
        "sspm_context_dim": int(sspm.context_dim),
        "sspm_calibration": _model_calibration_summary(config, sspm),
        "sspm_update_gate": _model_update_gate_summary(config, sspm),
        "real_diag_gamma_training": getattr(sspm, "real_diag_gamma_training_stats", {}),
        "pretrained_residual_embedder_path": str(config.pretrained_residual_embedder_path),
        "embedder_loaded": bool(embedder_loaded),
        "phase3d": {
            "sspm_train_mode": str(config.sspm_train_mode),
            "sspm_train_data_mode": str(config.sspm_train_data_mode),
            "sspm_checkpoint_path": str(config.sspm_checkpoint_path),
            "sspm_base_checkpoint_root": str(config.sspm_base_checkpoint_root),
            "sspm_train_batch_events": int(config.sspm_train_batch_events),
            "residual_embed_cache_mode": str(config.residual_embed_cache_mode),
            "residual_embed_cache_path": str(_resolved_residual_embed_cache_path(config)),
        },
    }
    if train_count is not None or validation_count is not None or test_count is not None:
        payload["event_counts_actual"] = {
            "train_events_actual": None if train_count is None else int(train_count),
            "validation_events_actual": None
            if validation_count is None
            else int(validation_count),
            "test_events_actual": None if test_count is None else int(test_count),
        }
    path = output_dir / "config_effective.yaml"
    try:
        import yaml

        text = yaml.safe_dump(payload, sort_keys=True, allow_unicode=False)
    except Exception:
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    path.write_text(text, encoding="utf-8")
    return path


def write_metrics_json(
    output_dir: Path,
    config: SlimConfig,
    sspm: SSPMLowRankModel,
    eval_payload: Mapping[str, Any],
    train_count: int,
    validation_count: int,
    test_count: int,
    embedder_loaded: bool,
) -> Path:
    """Write a compact metrics.json sidecar for smoke/full run inspection."""
    outputs = dict(eval_payload.get("outputs", {})) if isinstance(eval_payload, Mapping) else {}
    threshold_summary = dict(eval_payload.get("validation_event_score_summary", {}))
    metrics_payload = {
        "dataset": str(config.dataset),
        "out_tag": str(config.out_tag),
        "method_version": METHOD_VERSION,
        "events_scored": int(test_count),
        "train_count": int(train_count),
        "validation_count": int(validation_count),
        "embedder_loaded": bool(embedder_loaded),
        "pretrained_residual_embedder_path": str(config.pretrained_residual_embedder_path),
        "phase3d": {
            "sspm_train_mode": str(config.sspm_train_mode),
            "sspm_train_data_mode": str(config.sspm_train_data_mode),
            "sspm_checkpoint_path": str(config.sspm_checkpoint_path),
            "sspm_base_checkpoint_root": str(config.sspm_base_checkpoint_root),
            "sspm_train_batch_events": int(config.sspm_train_batch_events),
            "residual_embed_cache_mode": str(config.residual_embed_cache_mode),
            "residual_embed_cache_path": str(_resolved_residual_embed_cache_path(config)),
            "skip_train": str(config.sspm_train_mode) == "load_and_infer",
        },
        "phase3f": {
            "sspm_infer_fast_path": bool(config.sspm_infer_fast_path),
            "sspm_infer_chunk_events": int(config.sspm_infer_chunk_events),
            "rss_profile_mode": str(config.rss_profile_mode),
            "fast_path_enabled": bool(
                eval_payload.get("phase3f", {}).get("fast_path_enabled", False)
                if isinstance(eval_payload.get("phase3f", {}), Mapping)
                else False
            ),
        },
        "sspm": {
            "state_model": str(config.sspm_state_model),
            "target_dim": int(config.sspm_target_dim),
            "state_dim": int(config.sspm_state_dim),
            "rank": int(config.rank),
            "context_dim": int(sspm.context_dim),
            "context_mode": str(config.sspm_context_mode),
            "context_action_mode": str(config.sspm_context_action_mode),
            "global_context_mode": str(config.sspm_global_context_mode),
            "state_memory_mode": str(config.state_memory_mode),
            "state_memory_policy": str(config.sspm_state_memory_policy),
            "update_gate_mode": str(config.sspm_update_gate_mode),
            "update_gate_quantile": float(config.sspm_update_gate_quantile),
            "update_gate_threshold_mode": str(config.sspm_update_gate_threshold_mode),
            "update_gate_q_min": float(config.sspm_update_gate_q_min),
            "update_gate_eta": float(config.sspm_update_gate_eta),
            "residual_score_mode": str(config.sspm_residual_score_mode),
            "residual_calibration": str(config.sspm_residual_calibration),
            "residual_alpha_cos": float(config.sspm_residual_alpha_cos),
            "residual_beta_mse": float(config.sspm_residual_beta_mse),
            "residual_beta_var": float(config.sspm_residual_beta_var),
            "residual_var_eps": float(config.sspm_residual_var_eps),
            "residual_calibration_min_count": int(
                config.sspm_residual_calibration_min_count,
            ),
            "real_diag_train_gamma": bool(config.real_diag_train_gamma),
            "real_diag_gamma_lr": float(config.real_diag_gamma_lr),
            "real_diag_gamma_weight_decay": float(config.real_diag_gamma_weight_decay),
            "real_diag_gamma_grad_clip": float(config.real_diag_gamma_grad_clip),
            "real_diag_sensitivity_mode": str(config.real_diag_sensitivity_mode),
            "real_diag_gamma_min": float(config.real_diag_gamma_min),
            "real_diag_gamma_max": float(config.real_diag_gamma_max),
            "real_diag_max_sensitivity_nodes": int(config.real_diag_max_sensitivity_nodes),
            "real_diag_gamma_training": getattr(sspm, "real_diag_gamma_training_stats", {}),
            "calibration": _model_calibration_summary(config, sspm),
            "update_gate": _model_update_gate_summary(config, sspm),
            "checkpoint_state_merge_mode": eval_payload.get(
                "ofsm_runtime_config",
                {},
            ).get("checkpoint_state_merge_mode"),
            "requested_state_merge_mode": eval_payload.get(
                "ofsm_runtime_config",
                {},
            ).get("requested_state_merge_mode"),
            "actual_state_merge_mode": eval_payload.get(
                "ofsm_runtime_config",
                {},
            ).get("actual_state_merge_mode", str(sspm.config.state_merge_mode)),
            "state_merge_mode": str(config.sspm_state_merge_mode),
            "state_merge_scope": str(config.sspm_state_merge_scope),
            "state_merge_threshold": float(config.sspm_state_merge_threshold),
        },
        "train_events_actual": int(train_count),
        "validation_events_actual": int(validation_count),
        "test_events_actual": int(test_count),
        "event_threshold": {
            "threshold_mode": threshold_summary.get("threshold_mode"),
            "final_threshold": threshold_summary.get("final_threshold"),
            "validation_max_score": threshold_summary.get("validation_max_score"),
            "threshold_quantile": threshold_summary.get("threshold_quantile"),
        },
        "score_summary": dict(eval_payload.get("score_summary", {})),
        "online_minimal_rss": {
            key: eval_payload.get("memory", {}).get(key)
            for key in (
                "online_minimal_rss_start_mb",
                "online_minimal_rss_peak_mb",
                "online_minimal_rss_final_mb",
                "online_minimal_rss_delta_mb",
                "online_minimal_events_per_sec",
                "online_minimal_state_array_mb",
                "online_minimal_alert_buffer_mb",
                "online_minimal_endpoint_cache_mb",
                "online_minimal_suppression_summary_mb",
                "online_minimal_anonymous_rss_mb",
                "online_minimal_file_backed_rss_mb",
                "post_stream_eval_rss_mb",
            )
            if isinstance(eval_payload.get("memory", {}), Mapping)
        },
        "rss_breakdown_online_event_core": dict(
            eval_payload.get("rss_breakdown_online_event_core", {}),
        ),
        "online_event_node_coverage": dict(
            eval_payload.get("online_event_node_coverage", {}),
        ),
        "ofsm_runtime_config": dict(eval_payload.get("ofsm_runtime_config", {})),
        "ofsm_compression": _ofsm_compression_summary(
            sspm,
            memory=eval_payload.get("memory", {}),
            events_per_sec=eval_payload.get("runtime", {}).get(
                "throughput_events_per_second",
            )
            if isinstance(eval_payload.get("runtime", {}), Mapping)
            else None,
        ),
        "paths": outputs,
        "eval": dict(eval_payload),
    }
    path = output_dir / "metrics.json"
    path.write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    if str(config.rss_profile_mode) in {"deploy_light", "online_minimal"}:
        _write_ofsm_deploy_rss_summary(output_dir, metrics_payload["ofsm_compression"])
    _print_ofsm_summary(metrics_payload["ofsm_compression"])
    return path


def _write_ofsm_deploy_rss_summary(output_dir: Path, summary: Mapping[str, Any]) -> Path:
    path = output_dir / "ofsm_deploy_rss_summary.csv"
    fields = [
        "state_merge_mode",
        "deploy_rss_valid",
        "deploy_infer_rss_peak_mb",
        "deploy_infer_rss_final_mb",
        "deploy_infer_events_per_sec",
        "logical_node_count",
        "num_physical_states",
        "compression_ratio",
        "estimated_state_mb_saved",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: summary.get(field, "") for field in fields})
    return path


def _print_ofsm_summary(summary: Mapping[str, Any]) -> None:
    print("[OFSM Summary]", file=sys.stderr, flush=True)
    fields = [
        ("state_merge_mode", "state_merge_mode"),
        ("logical_nodes", "logical_node_count"),
        ("physical_states", "num_physical_states"),
        ("clusters", "num_clusters"),
        ("clustered_nodes", "num_clustered_nodes"),
        ("compression_ratio", "compression_ratio"),
        ("estimated_state_mb_no_merge", "estimated_state_mb_no_merge"),
        ("estimated_state_mb_after_merge", "estimated_state_mb_after_merge"),
        ("estimated_state_mb_saved", "estimated_state_mb_saved"),
        ("rss_peak_mb", "rss_peak_mb"),
        ("rss_final_mb", "rss_final_mb"),
    ]
    for label, key in fields:
        print(f"{label}={summary.get(key, '')}", file=sys.stderr, flush=True)


def stable_json_hash(payload: object) -> str:
    """Return a deterministic SHA-256 hash for a JSON-serializable payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Explicit cross-module imports; replaces the migration namespace bridge.
from scripts.pipeline.checks.preflight import (
    _bounded_stream_node_hashes,
    build_node_maps,
    _load_process_config,
    _phase3e_detect_event_id_column,
    _phase3e_scan_used_nodes_pass1,
    _phase3e_stream_split_events_direct,
    _prepare_slim_node_lookup_temp_table,
    _resolve_db_split_metadata,
    _slim_temp_day_filter,
    stream_dataset_rows_slim,
)
from scripts.pipeline.conditional.train import run_phase3g_conditional_train_from_precompute
from scripts.pipeline.features.conditional_context import LazyMmapLruEmbeddingLookup
from scripts.pipeline.features.semantic_features import (
    _observe_process_semantic_audit,
    normalize_piece,
    raw_action_token,
    row_fields,
    split_summary_tokens,
)
from scripts.pipeline.io.db_stream import _cfg_for_dataset
from scripts.data.get_dataset import use_event_type_filter
from scripts.pipeline.outputs.alert_output import _current_rss_mb
from scripts.pipeline.outputs.metrics_summary import _encode_row, _ofsm_compression_summary
from scripts.pipeline.state.online_state_runtime import (
    _make_sspm_config,
    _phase3e_fields_from_index_row,
    _phase3e_resolve_cache_dir,
    _phase3e_score_event_index_stream,
    _phase3e_target_from_index_row,
    _resolved_residual_embed_cache_path,
    _stage_log,
    load_pretrained_residual_embedder,
    save_sspm_checkpoint,
)
