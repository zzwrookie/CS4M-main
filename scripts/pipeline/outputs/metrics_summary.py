"""Compact summaries, profiling rows, and memory/RSS helpers."""

from __future__ import annotations

from scripts.pipeline.config.runtime_config import *


def _compact_print_summary(
    eval_path: str | Path,
    payload: Mapping[str, Any] | None = None,
) -> None:
    if payload is None:
        with Path(eval_path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
    runtime = dict(payload.get("runtime", {}))
    timing = dict(payload.get("timing", {}))
    memory = dict(payload.get("memory", {}))
    ofsm_summary = dict(payload.get("ofsm_compression", {}))
    if not ofsm_summary:
        ofsm_summary = dict(payload.get("state_merge", {}))
        if ofsm_summary:
            state_dim = int(ofsm_summary.get("state_dim", 64) or 64)
            logical_count = int(ofsm_summary.get("logical_node_count", 0) or 0)
            physical_count = int(ofsm_summary.get("num_physical_states", 0) or 0)
            no_merge = float(logical_count * state_dim * 4) / (1024.0 * 1024.0)
            after_merge = float(physical_count * state_dim * 4) / (1024.0 * 1024.0)
            ofsm_summary.update(
                {
                    "state_dim": state_dim,
                    "state_dtype": "float32",
                    "estimated_state_mb_no_merge": no_merge,
                    "estimated_state_mb_after_merge": after_merge,
                    "estimated_state_mb_saved": float(no_merge - after_merge),
                    "rss_peak_mb": float(memory.get("rss_test_peak_mb", 0.0)),
                    "rss_final_mb": float(memory.get("rss_after_label_attach_mb", 0.0)),
                },
            )
    metrics = payload.get("primary_online_metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    event_alerts = metrics.get("event_alerts", {})
    node_topk = metrics.get("node_pool_topk", {})
    event_alerts = event_alerts if isinstance(event_alerts, Mapping) else {}
    node_topk = node_topk if isinstance(node_topk, Mapping) else {}
    top1000 = node_topk.get("top1000", {})
    top3000 = node_topk.get("top3000", {})
    top1000 = top1000 if isinstance(top1000, Mapping) else {}
    top3000 = top3000 if isinstance(top3000, Mapping) else {}
    config = payload.get("config", {})
    config = config if isinstance(config, Mapping) else {}
    summary = {
        "out_json": str(eval_path),
        "method_name": payload.get("method_name", ""),
        "method_version": payload.get("method_version", ""),
        "dataset": str(config.get("dataset", payload.get("dataset", ""))),
        "out_tag": str(config.get("out_tag", payload.get("out_tag", ""))),
        "event_score_mode": str(config.get("event_score_mode", "")),
        "node_pool_score_mode": str(config.get("node_pool_score_mode", "")),
        "event_alerts": {
            "tp": int(event_alerts.get("tp", 0)),
            "fp": int(event_alerts.get("fp", 0)),
            "count": int(event_alerts.get("count", 0)),
            "precision": float(event_alerts.get("precision", event_alerts.get("ratio", 0.0))),
        },
        "node_pool_topk": {
            "top1000_tp": int(top1000.get("tp", 0)),
            "top3000_tp": int(top3000.get("tp", 0)),
        },
        "runtime": {
            "test_scoring_seconds": float(timing.get("test_scoring_seconds", 0.0)),
            "throughput_events_per_second": float(
                runtime.get("throughput_events_per_second", 0.0),
            ),
        },
            "memory": {
                "rss_after_test_cleanup_mb": float(memory.get("rss_after_test_cleanup_mb", 0.0)),
                "rss_test_peak_mb": float(memory.get("rss_test_peak_mb", 0.0)),
                "state_array_mb": float(memory.get("state_array_mb", 0.0)),
                "deploy_rss_valid": bool(memory.get("deploy_rss_valid", False)),
                "deploy_infer_rss_peak_mb": float(
                    memory.get("deploy_infer_rss_peak_mb", 0.0),
                ),
                "deploy_infer_rss_final_mb": float(
                    memory.get("deploy_infer_rss_final_mb", 0.0),
                ),
            },
        }
    if ofsm_summary:
        summary["ofsm_summary"] = {
            "state_merge_mode": str(ofsm_summary.get("state_merge_mode", "none")),
            "logical_nodes": int(ofsm_summary.get("logical_node_count", 0) or 0),
            "physical_states": int(ofsm_summary.get("num_physical_states", 0) or 0),
            "clusters": int(ofsm_summary.get("num_clusters", 0) or 0),
            "clustered_nodes": int(ofsm_summary.get("num_clustered_nodes", 0) or 0),
            "compression_ratio": float(ofsm_summary.get("compression_ratio", 0.0) or 0.0),
            "state_dim": int(ofsm_summary.get("state_dim", 0) or 0),
            "state_dtype": str(ofsm_summary.get("state_dtype", "float32")),
            "estimated_state_mb_no_merge": float(
                ofsm_summary.get("estimated_state_mb_no_merge", 0.0) or 0.0,
            ),
            "estimated_state_mb_after_merge": float(
                ofsm_summary.get("estimated_state_mb_after_merge", 0.0) or 0.0,
            ),
            "estimated_state_mb_saved": float(
                ofsm_summary.get("estimated_state_mb_saved", 0.0) or 0.0,
            ),
            "rss_peak_mb": float(ofsm_summary.get("rss_peak_mb", 0.0) or 0.0),
            "rss_final_mb": float(ofsm_summary.get("rss_final_mb", 0.0) or 0.0),
            "deploy_rss_valid": bool(ofsm_summary.get("deploy_rss_valid", False)),
            "deploy_infer_rss_peak_mb": float(
                ofsm_summary.get("deploy_infer_rss_peak_mb", 0.0) or 0.0,
            ),
            "deploy_infer_rss_final_mb": float(
                ofsm_summary.get("deploy_infer_rss_final_mb", 0.0) or 0.0,
            ),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _synthetic_row(
    event_index: int,
    seconds: int,
    src_idx: int,
    dst_idx: int,
    dst_addr: str,
    dst_port: str,
    label: str,
) -> dict[str, Any]:
    return {
        "event_index": int(event_index),
        "timestamp_ns": int(seconds) * 1_000_000_000,
        "src_idx": int(src_idx),
        "dst_idx": int(dst_idx),
        "action": "EVENT_SENDTO",
        "object_type": "netflow",
        "text": f"netflow to {dst_addr}:{dst_port}",
        "src_summary": "process curl client",
        "dst_summary": f"netflow {dst_addr}",
        "dst_addr": dst_addr,
        "dst_port": dst_port,
        "label": label,
    }


def _limit_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if int(limit) <= 0:
        return rows
    return rows[: int(limit)]


def _label_free_rows(rows: Sequence[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    for row in rows:
        safe_row = dict(row)
        safe_row.pop("label", None)
        yield safe_row


def _encode_row(
    embedder: ResidualEmbedder,
    row: Mapping[str, Any],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None = None,
) -> np.ndarray:
    tokens = _residual_tokens_for_row(row, config, process_cfg)
    return embedder.encode(tokens)


def _residual_projection(tokens: Sequence[str], latent_dim: int) -> tuple[str, str]:
    """Return signed hash projection text and sparse normalized vector for audit output."""
    dim = int(latent_dim)
    if dim <= 0:
        raise ValueError("latent_dim must be positive")
    vector = np.zeros((dim,), dtype=np.float32)
    projection: list[str] = []
    for token in tokens:
        bucket = stable_hash(token, seed=0) % dim
        sign = 1.0 if (stable_hash(token, seed=17) & 1) == 0 else -1.0
        vector[bucket] += np.float32(sign)
        projection.append(f"{token}=>{bucket}:{'+1' if sign > 0 else '-1'}")
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= np.float32(norm)
    nonzero = " ".join(
        f"{index}:{float(value):.6g}"
        for index, value in enumerate(vector)
        if float(value) != 0.0
    )
    return " ".join(projection), nonzero


def _residual_example_group(row_or_fields: Mapping[str, Any]) -> str:
    """Return the audit group for process interactions, or an empty string."""
    src_kind = str(row_or_fields.get("src_kind", "")).lower()
    dst_kind = str(row_or_fields.get("dst_kind", "")).lower()
    kinds = {src_kind, dst_kind}
    if "process" not in kinds:
        return ""
    if "file" in kinds:
        return "process_file"
    if "netflow" in kinds:
        return "process_netflow"
    if src_kind == "process" and dst_kind == "process":
        return "process_process"
    return ""


def _residual_audit_base_row(
    row: Mapping[str, Any],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
    group: str = "",
) -> dict[str, Any]:
    """Build a bounded residual-token audit row from runtime-visible fields only."""
    fields = row_fields(
        row,
        config.max_tokens_per_node,
        dataset=config.dataset,
        process_semantic_config=process_cfg,
        theia_netflow_policy=config.theia_netflow_policy,
    )
    text = residual_text(
        row,
        dataset=config.dataset,
        process_semantic_config=process_cfg,
        max_tokens_per_node=config.max_tokens_per_node,
        theia_netflow_policy=config.theia_netflow_policy,
        semantic_mode=config.semantic_mode,
    )
    tokens = list(residual_text_tokens(text))
    latent_dim = int(config.latent_dim)
    projection, nonzero = _residual_projection(tokens, latent_dim)
    return {
        "dataset": str(config.dataset),
        "group": group,
        "event_index": int(row.get("event_index", -1)),
        "timestamp_ns": int(row.get("timestamp_ns", 0)),
        "src_idx": int(row.get("src_idx", -1)),
        "dst_idx": int(row.get("dst_idx", -1)),
        "src_kind": str(row.get("src_kind", "")),
        "dst_kind": str(row.get("dst_kind", "")),
        "action": str(fields.get("action", "")),
        "object_type": str(fields.get("object_type", "")),
        "src_role": fields["src_role"],
        "dst_role": fields["dst_role"],
        "residual_text": text,
        "hash_tokens": " ".join(tokens),
        "token_count": int(len(tokens)),
        "token_projection": projection,
        "nonzero_vector": nonzero,
        "latent_dim": latent_dim,
    }


def _collect_residual_semantic_examples(
    rows,
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
    per_group: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Collect label-free process/file/netflow residual examples for eval JSON."""
    limit = max(int(per_group), 0)
    examples: dict[str, list[dict[str, Any]]] = {
        group: []
        for group in RESIDUAL_EXAMPLE_GROUPS
    }
    if limit == 0:
        return examples
    for row in rows:
        group = _residual_example_group(row)
        if not group or len(examples[group]) >= limit:
            continue
        audit_row = _residual_audit_base_row(row, config, process_cfg, group=group)
        audit_row.pop("label", None)
        examples[group].append(
            {field: audit_row.get(field, "") for field in RESIDUAL_EXAMPLE_FIELDS}
        )
        if all(len(items) >= limit for items in examples.values()):
            break
    return examples


def _flatten_residual_examples(
    examples: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Flatten grouped residual examples for CSV writing."""
    rows: list[dict[str, Any]] = []
    for group in RESIDUAL_EXAMPLE_GROUPS:
        for row in examples.get(group, []):
            materialized = {field: row.get(field, "") for field in RESIDUAL_EXAMPLE_FIELDS}
            materialized["group"] = group
            rows.append(materialized)
    return rows


def _malicious_pair_audit_row(
    row: Mapping[str, Any],
    abnormal_nodes: set[int],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
) -> dict[str, Any] | None:
    """Return a GT-only audit row when both surface event nodes are abnormal."""
    src_idx = int(row.get("src_idx", -1))
    dst_idx = int(row.get("dst_idx", -1))
    if src_idx not in abnormal_nodes or dst_idx not in abnormal_nodes:
        return None
    base = _residual_audit_base_row(row, config, process_cfg)
    base.pop("group", None)
    base["event_label"] = _label_name(row.get("label", "malicious"))
    base["both_nodes_abnormal"] = True
    return {field: base.get(field, "") for field in MALICIOUS_PAIR_AUDIT_FIELDS}


def _embedding_eval_metadata(config: SlimConfig, embedder: ResidualEmbedder) -> dict[str, Any]:
    """Return label-free residual embedding and latent-dimension metadata."""
    stats = dict(embedder.stats())
    train_vocab_size = int(stats.get("train_vocab_size", 0))
    selected_latent_dim = int(stats.get("latent_dim", _effective_latent_dim(config, None)))
    return {
        "semantic_embedding": stats,
        "embedder_loaded": _embedder_loaded(config),
        "pretrained_residual_embedder_path": str(config.pretrained_residual_embedder_path),
        "vocab_stats": {
            "train_vocab_size": train_vocab_size,
            "train_sentence_count": int(stats.get("train_sentence_count", 0)),
        },
        "latent_dim_selection": {
            "policy": str(config.latent_dim_policy),
            "configured_latent_dim": int(config.latent_dim),
            "selected_latent_dim": selected_latent_dim,
            "train_vocab_size": train_vocab_size,
        },
    }


def _combined_tail_scores(
    residual_scores: Sequence[float],
) -> list[float]:
    return _validation_event_scores_from_raw(residual_scores, "res_only")


def _validation_event_scores_from_raw(
    residual_scores: Sequence[float],
    mode: str,
) -> list[float]:
    _validate_active_event_score_mode(mode)
    length = len(residual_scores)
    if length <= 0:
        return [0.0]
    residual_sorted = _compact_sorted_scores(residual_scores)
    out = []
    for index in range(length):
        residual_tail = _empirical_tail_score_compact(residual_scores[index], residual_sorted)
        out.append(_event_score_from_residual_tail(residual_tail, mode))
    return sorted(out or [0.0])


def _validate_active_event_score_mode(mode: str) -> None:
    if str(mode) in ARCHIVED_EVENT_SCORE_MODES:
        raise ValueError(ARCHIVED_SEMANTIC_NLL_MESSAGE)
    if str(mode) not in EVENT_SCORE_MODES:
        raise ValueError(f"unknown event_score_mode: {mode}")


def _event_score_from_residual_tail(residual_tail: float, mode: str) -> float:
    _validate_active_event_score_mode(mode)
    if mode == "res_only":
        return float(residual_tail)
    raise ValueError(f"unknown event_score_mode: {mode}")


def _empty_timing() -> dict[str, float]:
    return {
        "node_map_seconds": 0.0,
        "cache_read_seconds": 0.0,
        "cache_write_seconds": 0.0,
        "train_seconds": 0.0,
        "validation_seconds": 0.0,
        "test_scoring_seconds": 0.0,
        "label_attach_seconds": 0.0,
        "csv_write_seconds": 0.0,
        "stream_csv_write_seconds": 0.0,
        "post_stream_output_seconds": 0.0,
    }


def _empty_memory_profile() -> dict[str, Any]:
    return {
        "rss_after_node_map_mb": 0.0,
        "rss_after_train_mb": 0.0,
        "rss_after_validation_mb": 0.0,
        "rss_before_test_cleanup_mb": 0.0,
        "rss_after_test_cleanup_mb": 0.0,
        "rss_test_peak_mb": 0.0,
        "rss_after_label_attach_mb": 0.0,
        "state_slots": 0,
        "state_array_mb": 0.0,
        "process_peak_rss_mb": 0.0,
        "test_cleanup_applied": False,
        "test_cleanup_released_payload": False,
        "test_cleanup_released_train_validation": False,
        "test_cleanup_gc_collected": 0,
        "test_node_map_mode": "",
    }


def _finish_test_phase_cleanup(
    memory_profile: dict[str, Any],
    *,
    released_payload: bool,
    node_map_mode: str,
    released_train_validation: bool,
) -> None:
    memory_profile["test_cleanup_gc_collected"] = int(gc.collect())
    memory_profile["rss_after_test_cleanup_mb"] = _current_rss_mb()
    memory_profile["test_cleanup_applied"] = True
    memory_profile["test_cleanup_released_payload"] = bool(released_payload)
    memory_profile["test_cleanup_released_train_validation"] = bool(
        released_train_validation,
    )
    memory_profile["test_node_map_mode"] = str(node_map_mode)


def _profiling_row(
    phase: str,
    processed_events: int,
    started_at: float,
    event_alert_count: int,
    node_pool_size: int,
    sspm: SSPMLowRankModel,
    current_rss_mb: float | None = None,
    test_phase_peak_rss_mb: float | None = None,
) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    current_rss = _current_rss_mb() if current_rss_mb is None else float(current_rss_mb)
    test_peak = current_rss if test_phase_peak_rss_mb is None else float(
        test_phase_peak_rss_mb,
    )
    state_stats = (
        sspm.state_memory_stats()
        if hasattr(sspm, "state_memory_stats")
        else {}
    )
    return {
        "phase": str(phase),
        "processed_events": int(processed_events),
        "elapsed_seconds": float(elapsed),
        "throughput_events_per_second": float(processed_events / elapsed),
        "current_rss_mb": current_rss,
        "peak_rss_mb": _safe_peak_rss_mb(),
        "test_phase_peak_rss_mb": test_peak,
        "event_alert_count": int(event_alert_count),
        "node_pool_size": int(node_pool_size),
        "state_slots": _safe_state_slots(sspm),
        "state_array_mb": _safe_state_array_mb(sspm),
        "state_metadata_mb": float(state_stats.get("state_metadata_mb", 0.0)),
        "state_eviction_count": int(state_stats.get("eviction_count", 0)),
        "state_overflow_count": int(state_stats.get("overflow_count", 0)),
        "state_prototype_hit_count": int(state_stats.get("prototype_hit_count", 0)),
    }


def _state_merge_enabled(config: SlimConfig) -> bool:
    return str(config.sspm_state_merge_mode) != "none"


def _ofsm_compression_summary(
    sspm: SSPMLowRankModel,
    memory: Mapping[str, Any] | None = None,
    events_per_sec: float | None = None,
) -> dict[str, Any]:
    stats = dict(sspm.state_merge_profile_stats())
    state_dim = int(stats.get("state_dim", getattr(sspm, "state_dim", 0)) or 0)
    logical_count = int(stats.get("logical_node_count", 0) or 0)
    singleton_count = int(stats.get("num_singleton_states", logical_count) or 0)
    cluster_count = int(stats.get("num_clusters", 0) or 0)
    physical_count = int(stats.get("num_physical_states", singleton_count + cluster_count) or 0)
    clustered_count = int(stats.get("num_clustered_nodes", 0) or 0)
    compression_ratio = (
        float(logical_count) / float(max(physical_count, 1))
        if logical_count
        else 0.0
    )
    state_mb_no_merge = float(logical_count * state_dim * 4) / (1024.0 * 1024.0)
    state_mb_after_merge = float(physical_count * state_dim * 4) / (1024.0 * 1024.0)
    memory_data = dict(memory or {})
    rss_peak = float(
        memory_data.get(
            "rss_test_peak_mb",
            memory_data.get("process_peak_rss_mb", _safe_peak_rss_mb()),
        )
        or 0.0,
    )
    rss_final = float(
        memory_data.get(
            "rss_after_label_attach_mb",
            memory_data.get("rss_after_test_cleanup_mb", _current_rss_mb()),
        )
        or 0.0,
    )
    deploy_valid = bool(memory_data.get("deploy_rss_valid", False))
    deploy_peak = float(memory_data.get("deploy_infer_rss_peak_mb", rss_peak) or 0.0)
    deploy_final = float(memory_data.get("deploy_infer_rss_final_mb", rss_final) or 0.0)
    deploy_eps = memory_data.get("deploy_infer_events_per_sec", events_per_sec)
    payload = {
        "state_merge_mode": str(stats.get("state_merge_mode", "none")),
        "state_merge_scope": str(stats.get("state_merge_scope", "same_type")),
        "state_merge_threshold": stats.get("merge_threshold", stats.get("state_merge_threshold", "")),
        "logical_node_count": logical_count,
        "num_singleton_states": singleton_count,
        "num_clusters": cluster_count,
        "num_clustered_nodes": clustered_count,
        "num_physical_states": physical_count,
        "compression_ratio": float(compression_ratio),
        "state_dim": state_dim,
        "state_dtype": "float32",
        "estimated_state_mb_no_merge": state_mb_no_merge,
        "estimated_state_mb_after_merge": state_mb_after_merge,
        "estimated_state_mb_saved": float(state_mb_no_merge - state_mb_after_merge),
        "rss_peak_mb": rss_peak,
        "rss_final_mb": rss_final,
        "events_per_sec": "" if events_per_sec is None else float(events_per_sec),
        "deploy_rss_valid": deploy_valid,
        "deploy_infer_rss_peak_mb": deploy_peak,
        "deploy_infer_rss_final_mb": deploy_final,
        "deploy_infer_events_per_sec": (
            "" if deploy_eps is None else float(deploy_eps)
        ),
    }
    runtime_config = getattr(sspm, "load_and_infer_runtime_config_summary", {})
    if isinstance(runtime_config, Mapping):
        payload.update(
            {
                "checkpoint_state_merge_mode": runtime_config.get(
                    "checkpoint_state_merge_mode",
                ),
                "requested_state_merge_mode": runtime_config.get(
                    "requested_state_merge_mode",
                ),
                "actual_state_merge_mode": runtime_config.get("actual_state_merge_mode"),
            },
        )
    for key in (
        "num_copy_on_write_total",
        "num_merge_attempts_total",
        "num_merge_success_total",
        "num_remerged_total",
        "num_became_singleton_total",
        "num_random_merge_attempts_total",
        "num_random_merge_success_total",
        "candidate_scan_count_total",
        "num_candidates_checked_total",
        "max_candidates_per_attempt",
        "candidate_cap",
        "merge_interval",
        "merge_min_count",
        "skipped_merge_due_to_interval",
        "skipped_merge_due_to_min_count",
    ):
        payload[key] = int(stats.get(key, 0) or 0)
    payload["match_backend"] = str(stats.get("match_backend", "exact"))
    payload["candidate_scan_time_sec"] = float(stats.get("candidate_scan_time_sec", 0.0) or 0.0)
    payload["avg_candidates_per_attempt"] = float(
        stats.get("avg_candidates_per_attempt", 0.0) or 0.0,
    )
    return payload


def _empty_node_count_state() -> dict[str, int]:
    return {
        "event_count": 0,
        "as_src_count": 0,
        "as_dst_count": 0,
    }


def _node_count_snapshot(
    node_counts: Mapping[int, Mapping[str, int]],
    src_id: int,
    dst_id: int,
) -> dict[str, int]:
    src_counts = node_counts.get(int(src_id), _empty_node_count_state())
    dst_counts = node_counts.get(int(dst_id), _empty_node_count_state())
    return {
        "src_event_count": int(src_counts.get("event_count", 0)),
        "dst_event_count": int(dst_counts.get("event_count", 0)),
        "src_as_src_count": int(src_counts.get("as_src_count", 0)),
        "dst_as_dst_count": int(dst_counts.get("as_dst_count", 0)),
    }


def _update_node_count_state(
    node_counts: dict[int, dict[str, int]],
    src_id: int,
    dst_id: int,
) -> None:
    src_key = int(src_id)
    dst_key = int(dst_id)
    src_state = node_counts.setdefault(src_key, _empty_node_count_state())
    src_state["event_count"] += 1
    src_state["as_src_count"] += 1
    if src_key == dst_key:
        src_state["as_dst_count"] += 1
        return
    dst_state = node_counts.setdefault(dst_key, _empty_node_count_state())
    dst_state["event_count"] += 1
    dst_state["as_dst_count"] += 1


def _top_residual_dim_fields(delta: np.ndarray) -> dict[str, int | str]:
    abs_delta = np.abs(np.asarray(delta, dtype=np.float32)).reshape(-1)
    order = np.argsort(-abs_delta)
    fields: dict[str, int | str] = {}
    for idx in range(3):
        key = f"top_residual_dim_{idx + 1}"
        fields[key] = int(order[idx]) if idx < int(order.shape[0]) else ""
    return fields


def _residual_feature_fields(
    z_hat: np.ndarray,
    z_true: np.ndarray,
    context_snapshot: Mapping[str, Any],
    count_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    pred_vec = np.asarray(z_hat, dtype=np.float32).reshape(-1)
    true_vec = np.asarray(z_true, dtype=np.float32).reshape(-1)
    delta = true_vec - pred_vec
    pred_norm = float(np.linalg.norm(pred_vec))
    true_norm = float(np.linalg.norm(true_vec))
    if pred_norm <= 1e-8 or true_norm <= 1e-8:
        residual_cos = 1.0
    else:
        residual_cos = float(1.0 - np.dot(pred_vec / pred_norm, true_vec / true_norm))
    return {
        "residual_l2": float(np.linalg.norm(delta)),
        "residual_cos": residual_cos,
        **_top_residual_dim_fields(delta),
        "src_state_norm": float(context_snapshot.get("src_state_norm", 0.0)),
        "dst_state_norm": float(context_snapshot.get("dst_state_norm", 0.0)),
        "src_event_count": int(count_snapshot.get("src_event_count", 0)),
        "dst_event_count": int(count_snapshot.get("dst_event_count", 0)),
        "src_as_src_count": int(count_snapshot.get("src_as_src_count", 0)),
        "dst_as_dst_count": int(count_snapshot.get("dst_as_dst_count", 0)),
    }


def _state_merge_profile_row(
    sspm: SSPMLowRankModel,
    event_idx: int,
    started_at: float,
) -> dict[str, Any]:
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    stats = dict(sspm.state_merge_profile_stats())
    events_per_sec = float(event_idx) / elapsed
    compression = _ofsm_compression_summary(
        sspm,
        memory={
            "rss_test_peak_mb": _safe_peak_rss_mb(),
            "rss_after_label_attach_mb": _current_rss_mb(),
        },
        events_per_sec=events_per_sec,
    )
    return {
        **stats,
        **compression,
        "event_idx": int(event_idx),
        "rss_mb": f"{_current_rss_mb():.6f}",
        "events_per_sec": f"{events_per_sec:.6f}",
    }


def _ofsm_runtime_profile_payload(
    *,
    config: SlimConfig,
    model: SSPMLowRankModel,
    events: int,
    stage: str,
    started_at: float,
    test_seconds: float,
    embedding_metrics: Mapping[str, Any] | None = None,
    smaps: Mapping[str, Any] | None = None,
    timing: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return a compact OFSM runtime profile row for profiling and sweep runs."""
    elapsed = max(time.perf_counter() - started_at, 1e-9)
    stats = dict(model.state_merge_profile_stats())
    compression = _ofsm_compression_summary(
        model,
        memory={
            "rss_test_peak_mb": _safe_peak_rss_mb(),
            "rss_after_label_attach_mb": _current_rss_mb(),
            "deploy_rss_valid": True,
            "deploy_infer_events_per_sec": float(events) / max(float(test_seconds), 1e-9),
        },
        events_per_sec=float(events) / max(float(test_seconds), 1e-9),
    )
    state_stats = (
        model.state_memory_stats()
        if hasattr(model, "state_memory_stats")
        else {}
    )
    embed_metrics = dict(embedding_metrics or {})
    smaps_data = dict(smaps or {})
    extra_timing = dict(timing or {})
    return {
        "run": str(config.out_tag),
        "events": int(events),
        "stage": str(stage),
        "match_backend": str(stats.get("match_backend", config.sspm_ofsm_match_backend)),
        "candidate_cap": int(stats.get("candidate_cap", config.sspm_ofsm_candidate_cap) or 0),
        "merge_interval": int(stats.get("merge_interval", config.sspm_ofsm_merge_interval) or 0),
        "min_count": int(stats.get("merge_min_count", config.sspm_ofsm_min_count) or 0),
        "total_seconds": float(elapsed),
        "events_per_sec": float(events) / max(float(test_seconds), 1e-9),
        "validation_seconds": float(extra_timing.get("validation_seconds", 0.0) or 0.0),
        "test_seconds": float(test_seconds),
        "candidate_scan_seconds": float(stats.get("candidate_scan_time_sec", 0.0) or 0.0),
        "bucket_lookup_seconds": float(stats.get("bucket_lookup_time_sec", 0.0) or 0.0),
        "signature_update_seconds": float(stats.get("signature_update_time_sec", 0.0) or 0.0),
        "copy_on_write_seconds": float(stats.get("copy_on_write_time_sec", 0.0) or 0.0),
        "state_read_seconds": float(extra_timing.get("state_read_seconds", 0.0) or 0.0),
        "state_update_seconds": float(extra_timing.get("state_update_seconds", 0.0) or 0.0),
        "lazy_embedding_lookup_seconds": float(
            embed_metrics.get("embedding_lookup_seconds", 0.0) or 0.0,
        ),
        "endpoint_suppression_seconds": float(
            extra_timing.get("endpoint_suppression_seconds", 0.0) or 0.0,
        ),
        "csv_write_seconds": float(extra_timing.get("csv_write_seconds", 0.0) or 0.0),
        "score_compute_seconds": float(extra_timing.get("score_compute_seconds", 0.0) or 0.0),
        "threshold_lookup_seconds": float(
            extra_timing.get("threshold_lookup_seconds", 0.0) or 0.0,
        ),
        "num_merge_attempts_total": int(stats.get("num_merge_attempts_total", 0) or 0),
        "num_candidates_checked": int(stats.get("num_candidates_checked_total", 0) or 0),
        "avg_candidates_per_attempt": float(
            stats.get("avg_candidates_per_attempt", 0.0) or 0.0,
        ),
        "max_candidates_per_attempt": int(stats.get("max_candidates_per_attempt", 0) or 0),
        "merge_success_count": int(stats.get("num_merge_success_total", 0) or 0),
        "skipped_merge_due_to_interval": int(
            stats.get("skipped_merge_due_to_interval", 0) or 0,
        ),
        "skipped_merge_due_to_min_count": int(
            stats.get("skipped_merge_due_to_min_count", 0) or 0,
        ),
        "copy_on_write_count": int(stats.get("num_copy_on_write_total", 0) or 0),
        "num_clusters": int(stats.get("num_clusters", 0) or 0),
        "clustered_nodes": int(stats.get("num_clustered_nodes", 0) or 0),
        "compression_ratio": float(compression.get("compression_ratio", 0.0) or 0.0),
        "online_deploy_primary_memory_mb": float(
            extra_timing.get("online_deploy_primary_memory_mb", 0.0) or 0.0,
        ),
        "state_table_mb": float(state_stats.get("state_array_mb", _safe_state_array_mb(model))),
        "process_rss_peak_mb": float(_safe_peak_rss_mb()),
        "anonymous_rss_mb": smaps_data.get("anonymous_rss_mb", ""),
        "file_backed_rss_mb": smaps_data.get("file_backed_rss_mb", ""),
    }


def _write_ofsm_runtime_profile(
    output_dir: Path,
    payload: Mapping[str, Any],
) -> dict[str, str]:
    """Write OFSM runtime profile sidecars for short profiling and full runs."""
    csv_path = Path(output_dir) / "ofsm_runtime_profile.csv"
    json_path = Path(output_dir) / "ofsm_runtime_profile.json"
    _write_csv(csv_path, [payload], OFSM_RUNTIME_PROFILE_FIELDS)
    json_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return {
        "ofsm_runtime_profile_csv": str(csv_path),
        "ofsm_runtime_profile_json": str(json_path),
    }


def _ofsm_empty_node_snapshot(role: str, mode: str = "none") -> dict[str, Any]:
    return {
        f"{role}_cluster_id": "",
        f"{role}_cluster_size": 1,
        f"{role}_merge_similarity": 0.0,
        f"{role}_is_clustered": 0,
        f"{role}_copy_on_write": 0,
        f"{role}_merge_mode": str(mode),
        f"{role}_physical_state_id": "",
    }


def _ofsm_alert_fields(
    score_snapshot: Mapping[str, Any],
    update_metadata: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    src = dict(score_snapshot.get("src", {}))
    dst = dict(score_snapshot.get("dst", {}))
    actions = list(update_metadata.get("actions", []))
    row = {}
    for role, snapshot in (("src", src), ("dst", dst)):
        role_actions = [
            action for action in actions if str(action.get("src_or_dst", "")) == role
        ]
        copy_on_write = any(
            str(action.get("action_type", "")) == "copy_on_write"
            for action in role_actions
        )
        similarities = [
            float(action.get("merge_similarity", 0.0) or 0.0)
            for action in role_actions
            if str(action.get("action_type", "")) in {
                "merge_into_cluster",
                "merge_with_singleton_create_cluster",
                "random_merge_success",
            }
        ]
        row.update(
            {
                f"{role}_cluster_id": snapshot.get("cluster_id", ""),
                f"{role}_cluster_size": int(snapshot.get("cluster_size", 1) or 1),
                f"{role}_merge_similarity": float(max(similarities, default=0.0)),
                f"{role}_is_clustered": int(snapshot.get("is_clustered", 0) or 0),
                f"{role}_copy_on_write": int(bool(copy_on_write)),
                f"{role}_merge_mode": str(snapshot.get("merge_mode", mode)),
                f"{role}_physical_state_id": snapshot.get("physical_state_id", ""),
            },
        )
    return row


def _emit_profile_row(
    profile_writer: StreamingCsvWriter | None,
    row: Mapping[str, Any],
    verbose: bool = False,
) -> None:
    if profile_writer is not None:
        profile_writer.write_row(row)
        profile_writer.flush()
    _verbose(verbose, json.dumps(dict(row), sort_keys=True))


def _verbose(enabled: bool, message: str) -> None:
    if enabled:
        print(message, file=sys.stderr, flush=True)


def _stream_csv_write_seconds(*writers: StreamingCsvWriter | None) -> float:
    return float(sum(float(writer.write_seconds) for writer in writers if writer is not None))


def _current_smaps_rollup_mb() -> dict[str, float | None]:
    """Return lightweight RSS split from /proc/self/smaps_rollup when available."""
    path = Path("/proc/self/smaps_rollup")
    if not path.exists():
        return {
            "rss_mb": None,
            "anonymous_rss_mb": None,
            "file_backed_rss_mb": None,
            "shared_clean_mb": None,
            "private_clean_mb": None,
            "private_dirty_mb": None,
        }
    values_kb = {
        "Rss": 0.0,
        "Anonymous": 0.0,
        "Shared_Clean": 0.0,
        "Private_Clean": 0.0,
        "Private_Dirty": 0.0,
    }
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                key = line.split(":", 1)[0]
                if key in values_kb:
                    values_kb[key] = float(line.split()[1])
    except OSError:
        return {
            "rss_mb": None,
            "anonymous_rss_mb": None,
            "file_backed_rss_mb": None,
            "shared_clean_mb": None,
            "private_clean_mb": None,
            "private_dirty_mb": None,
        }
    file_backed_kb = max(values_kb["Rss"] - values_kb["Anonymous"], 0.0)
    return {
        "rss_mb": float(values_kb["Rss"] / 1024.0),
        "anonymous_rss_mb": float(values_kb["Anonymous"] / 1024.0),
        "file_backed_rss_mb": float(file_backed_kb / 1024.0),
        "shared_clean_mb": float(values_kb["Shared_Clean"] / 1024.0),
        "private_clean_mb": float(values_kb["Private_Clean"] / 1024.0),
        "private_dirty_mb": float(values_kb["Private_Dirty"] / 1024.0),
    }


# Explicit cross-module imports; replaces the migration namespace bridge.
from scripts.pipeline.checks.preflight import _effective_latent_dim
from scripts.pipeline.features.semantic_features import (
    _empirical_tail_score_compact,
    residual_text,
    row_fields,
)
from scripts.pipeline.io.cache_payloads import _compact_sorted_scores
from scripts.pipeline.io.event_artifacts import _embedder_loaded
from scripts.pipeline.outputs.alert_output import (
    _current_rss_mb,
    _label_name,
    _safe_peak_rss_mb,
    _safe_state_array_mb,
    _safe_state_slots,
    _write_csv,
)
from scripts.pipeline.state.online_state_runtime import _residual_tokens_for_row
