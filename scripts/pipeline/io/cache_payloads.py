"""Leakage-safe cache payload, fingerprint, and read/write validation helpers."""

from __future__ import annotations

from scripts.pipeline.config.runtime_config import *


def _phase3g_load_abnormal_db_nodes_for_eval(config: SlimConfig) -> set[int]:
    """Load abnormal DB-node ids after streaming, preferring local GT pickle files."""
    dataset = str(config.dataset)
    local_candidates: list[Path] = []
    if is_cadets_dataset(dataset):
        local_candidates = [
            Path("ground_truth/E3-CADETS/abnormal_nodes.pkl"),
            Path("ground_truth/E3-CADETS/sp25_gt/abnormal_nodes.pkl"),
        ]
    elif is_theia_dataset(dataset):
        local_candidates = [Path("ground_truth/E3-THEIA/abnormal_nodes.pkl")]
    elif is_clearscope_dataset(dataset):
        local_candidates = [Path("ground_truth/E3-CLEARSCOPE/abnormal_nodes.pkl")]
    for candidate in local_candidates:
        if candidate.exists():
            payload = pickle.loads(candidate.read_bytes())
            if isinstance(payload, Mapping):
                return {int(key) for key in payload.keys()}
            return {int(value) for value in payload}
    db_cfg = _cfg_for_dataset(dataset)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    cur = None
    conn = None
    try:
        cur, conn = init_database_connection(db_cfg)
        eval_node_maps = build_node_maps(cur)
        return load_ground_truth_indices_readonly(db_cfg, eval_node_maps["uuid2index"])
    except DB_EXCEPTION_TYPES:
        return set()
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def _phase3e_score_summary_payload(
    validation: EventCalibration,
    test_summary: Mapping[str, Any],
    event_alert_count: int,
) -> dict[str, Any]:
    """Return the score_summary.json payload for Phase3E inference runs."""
    validation_summary = _score_summary_from_event_calibration(validation)
    test_payload = dict(test_summary)
    test_payload["event_alert_count"] = int(event_alert_count)
    test_payload.setdefault(
        "test_above_threshold_count",
        int(test_payload.get("above_threshold_count", 0) or 0),
    )
    return {
        "validation": validation_summary,
        "test": test_payload,
        "score_min": test_payload.get("score_min"),
        "score_mean": test_payload.get("score_mean"),
        "score_std": test_payload.get("score_std"),
        "score_p99": test_payload.get("score_p99"),
        "score_p999": test_payload.get("score_p999"),
        "score_p9995": test_payload.get("score_p9995"),
        "score_p9999": test_payload.get("score_p9999"),
        "score_max": test_payload.get("score_max"),
        "final_threshold": float(validation.threshold),
        "threshold_mode": str(validation.summary.get("threshold_mode", "")),
        "threshold_quantile": float(validation.summary.get("threshold_quantile", 0.0)),
        "event_alert_count": int(event_alert_count),
        "test_above_threshold_count": int(test_payload.get("test_above_threshold_count", 0)),
    }


def _fingerprint_split_metadata(
    values: Mapping[str, Any],
    split_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if split_metadata is not None:
        metadata = dict(split_metadata)
    elif _has_inline_split_metadata(values):
        metadata = dict(values)
    else:
        dataset = str(values.get("dataset", SlimConfig.dataset))
        slim_split_override = bool(
            values.get("slim_split_override", SlimConfig.slim_split_override),
        )
        try:
            config = SlimConfig(
                dataset=dataset,
                slim_split_override=slim_split_override,
            )
            metadata = _resolve_db_split_metadata(config, _cfg_for_dataset(dataset))
        except Exception:
            metadata = _resolve_slim_split_metadata(
                dataset,
                "",
                [],
                [],
                [],
                slim_split_override,
            )
    return {
        "split_source": str(metadata.get("split_source", "config")),
        "slim_split_override": bool(
            metadata.get("slim_split_override", SlimConfig.slim_split_override),
        ),
        "year_month": str(metadata.get("year_month", "")),
        "train_days": _int_list(metadata.get("train_days", [])),
        "validation_days": _int_list(metadata.get("validation_days", [])),
        "test_days": _int_list(metadata.get("test_days", [])),
    }


def _has_inline_split_metadata(values: Mapping[str, Any]) -> bool:
    return any(
        key in values
        for key in (
            "split_source",
            "year_month",
            "train_days",
            "validation_days",
            "test_days",
        )
    )


def _eval_split_metadata(
    config: SlimConfig,
    split_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if split_metadata is not None:
        metadata = dict(split_metadata)
    else:
        metadata = _fingerprint_split_metadata(_config_values(config), None)
    payload = {
        "split_source": str(metadata.get("split_source", "config")),
        "slim_split_override": bool(
            metadata.get("slim_split_override", config.slim_split_override),
        ),
        "slim_split_override_applied": bool(
            metadata.get(
                "slim_split_override_applied",
                metadata.get("split_source") == "slim_split_override",
            ),
        ),
        "year_month": str(metadata.get("year_month", "")),
        "train_days": _int_list(metadata.get("train_days", [])),
        "validation_days": _int_list(metadata.get("validation_days", [])),
        "test_days": _int_list(metadata.get("test_days", [])),
    }
    if "split_override_reason" in metadata:
        payload["split_override_reason"] = str(metadata["split_override_reason"])
    payload["split_override"] = (
        _slim_split_override_days(config.dataset)
        if payload["slim_split_override_applied"]
        else {}
    )
    return payload


def model_fingerprint_payload(
    args_or_dict: object,
    split_metadata: Mapping[str, Any] | None = None,
    process_semantic_config: ProcessSemanticConfig | None = None,
) -> dict[str, Any]:
    """Return only train/validation artifact-affecting configuration fields."""
    values = _config_values(args_or_dict)
    dataset = str(values.get("dataset", SlimConfig.dataset))
    payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_scope": "train_validation_base_scores",
        "method_version": METHOD_VERSION,
        "dataset": dataset,
        "max_train_events": int(values.get("max_train_events", SlimConfig.max_train_events)),
        "max_ref_events": int(values.get("max_ref_events", SlimConfig.max_ref_events)),
        "max_tokens_per_node": int(
            values.get("max_tokens_per_node", SlimConfig.max_tokens_per_node),
        ),
        "network_semantic_rules_version": NETWORK_SEMANTIC_RULES_VERSION,
        "latent_dim": int(values.get("latent_dim", SlimConfig.latent_dim)),
        "rank": int(values.get("rank", SlimConfig.rank)),
        "tau": [int(value) for value in DEFAULT_TAU],
        "loss": SSPM_LOSS_VERSION,
        "clearscope_semantic_rules_version": (
            normalize_clearscope_semantic_mode(
                values.get("semantic_mode", SlimConfig.semantic_mode),
            )
            if is_clearscope_dataset(dataset)
            else ""
        ),
        "semantic_mode": str(values.get("semantic_mode", SlimConfig.semantic_mode)),
        "theia_semantic_rules_version": (
            THEIA_SEMANTIC_RULES_VERSION
            if is_theia_dataset(dataset)
            else ""
        ),
        "cadets_semantic_rules_version": (
            CADETS_SEMANTIC_RULES_VERSION
            if is_cadets_dataset(dataset)
            else ""
        ),
        "sspm_batch_size": int(values.get("sspm_batch_size", SlimConfig.sspm_batch_size)),
        "sspm_learning_rate": float(
            values.get("sspm_learning_rate", SlimConfig.sspm_learning_rate),
        ),
        "sspm_l2": float(values.get("sspm_l2", SlimConfig.sspm_l2)),
        "sspm_epochs": int(values.get("sspm_epochs", SlimConfig.sspm_epochs)),
        "sspm_early_stop_min_delta": float(
            values.get(
                "sspm_early_stop_min_delta",
                SlimConfig.sspm_early_stop_min_delta,
            ),
        ),
        "sspm_early_stop_patience": int(
            values.get(
                "sspm_early_stop_patience",
                SlimConfig.sspm_early_stop_patience,
            ),
        ),
        "sspm_state_model": str(values.get("sspm_state_model", SlimConfig.sspm_state_model)),
        "sspm_target_dim": int(values.get("sspm_target_dim", SlimConfig.sspm_target_dim)),
        "sspm_state_dim": int(values.get("sspm_state_dim", SlimConfig.sspm_state_dim)),
        "sspm_context_mode": str(
            values.get("sspm_context_mode", SlimConfig.sspm_context_mode),
        ),
        "sspm_context_action_mode": str(
            values.get("sspm_context_action_mode", SlimConfig.sspm_context_action_mode),
        ),
        "sspm_global_context_mode": str(
            values.get("sspm_global_context_mode", SlimConfig.sspm_global_context_mode),
        ),
        "state_memory_mode": str(values.get("state_memory_mode", SlimConfig.state_memory_mode)),
        "sspm_state_memory_policy": str(
            values.get("sspm_state_memory_policy", SlimConfig.sspm_state_memory_policy),
        ),
        "sspm_active_max_nodes": int(
            values.get("sspm_active_max_nodes", SlimConfig.sspm_active_max_nodes),
        ),
        "sspm_probationary_max_nodes": int(
            values.get(
                "sspm_probationary_max_nodes",
                SlimConfig.sspm_probationary_max_nodes,
            ),
        ),
        "sspm_probationary_min_count": int(
            values.get(
                "sspm_probationary_min_count",
                SlimConfig.sspm_probationary_min_count,
            ),
        ),
        "sspm_lru_evict_batch": int(
            values.get("sspm_lru_evict_batch", SlimConfig.sspm_lru_evict_batch),
        ),
        "sspm_alert_protect_events": int(
            values.get("sspm_alert_protect_events", SlimConfig.sspm_alert_protect_events),
        ),
        "sspm_update_gate_mode": str(
            values.get("sspm_update_gate_mode", SlimConfig.sspm_update_gate_mode),
        ),
        "sspm_update_gate_quantile": float(
            values.get("sspm_update_gate_quantile", SlimConfig.sspm_update_gate_quantile),
        ),
        "sspm_update_gate_threshold_mode": str(
            values.get(
                "sspm_update_gate_threshold_mode",
                SlimConfig.sspm_update_gate_threshold_mode,
            ),
        ),
        "sspm_update_gate_score_space": str(
            values.get(
                "sspm_update_gate_score_space",
                SlimConfig.sspm_update_gate_score_space,
            ),
        ),
        "sspm_update_gate_q_min": float(
            values.get("sspm_update_gate_q_min", SlimConfig.sspm_update_gate_q_min),
        ),
        "sspm_update_gate_eta": float(
            values.get("sspm_update_gate_eta", SlimConfig.sspm_update_gate_eta),
        ),
        "sspm_residual_score_mode": str(
            values.get("sspm_residual_score_mode", SlimConfig.sspm_residual_score_mode),
        ),
        "sspm_residual_calibration": str(
            values.get("sspm_residual_calibration", SlimConfig.sspm_residual_calibration),
        ),
        "sspm_residual_alpha_cos": float(
            values.get("sspm_residual_alpha_cos", SlimConfig.sspm_residual_alpha_cos),
        ),
        "sspm_residual_beta_mse": float(
            values.get("sspm_residual_beta_mse", SlimConfig.sspm_residual_beta_mse),
        ),
        "sspm_residual_beta_var": float(
            values.get("sspm_residual_beta_var", SlimConfig.sspm_residual_beta_var),
        ),
        "sspm_residual_var_eps": float(
            values.get("sspm_residual_var_eps", SlimConfig.sspm_residual_var_eps),
        ),
        "sspm_residual_calibration_min_count": int(
            values.get(
                "sspm_residual_calibration_min_count",
                SlimConfig.sspm_residual_calibration_min_count,
            ),
        ),
        "pretrained_residual_embedder_path": str(
            values.get(
                "pretrained_residual_embedder_path",
                SlimConfig.pretrained_residual_embedder_path,
            ),
        ),
        "max_exact_states": int(values.get("max_exact_states", SlimConfig.max_exact_states)),
        "min_inactive_events": int(
            values.get("min_inactive_events", SlimConfig.min_inactive_events),
        ),
        "prototype_count": int(values.get("prototype_count", SlimConfig.prototype_count)),
        "prototype_update_alpha": float(
            values.get("prototype_update_alpha", SlimConfig.prototype_update_alpha),
        ),
        "action_count": int(values.get("action_count", SlimConfig.action_count)),
        "latent_dim_policy": str(
            values.get("latent_dim_policy", SlimConfig.latent_dim_policy),
        ),
        "semantic_embedding_method": str(
            values.get("semantic_embedding_method", SlimConfig.semantic_embedding_method),
        ),
        "theia_netflow_policy": str(
            values.get("theia_netflow_policy", SlimConfig.theia_netflow_policy),
        ),
        "action_type_alert_policy": str(
            values.get("action_type_alert_policy", SlimConfig.action_type_alert_policy),
        ),
        "word2vec_window": int(values.get("word2vec_window", SlimConfig.word2vec_window)),
        "word2vec_min_count": int(
            values.get("word2vec_min_count", SlimConfig.word2vec_min_count),
        ),
        "word2vec_sg": int(values.get("word2vec_sg", SlimConfig.word2vec_sg)),
        "word2vec_negative": int(
            values.get("word2vec_negative", SlimConfig.word2vec_negative),
        ),
        "word2vec_epochs": int(values.get("word2vec_epochs", SlimConfig.word2vec_epochs)),
        "word2vec_workers": int(values.get("word2vec_workers", SlimConfig.word2vec_workers)),
        "word2vec_seed": int(values.get("word2vec_seed", SlimConfig.word2vec_seed)),
        "word2vec_oov_policy": str(
            values.get("word2vec_oov_policy", SlimConfig.word2vec_oov_policy),
        ),
        "process_semantics": _effective_process_semantic_fingerprint(
            dataset,
            process_semantic_config,
        ),
        "tail_normalization": TAIL_NORMALIZATION_VERSION,
    }
    payload.update(_fingerprint_split_metadata(values, split_metadata))
    return payload


def _legacy_event_mode_fingerprint_payload(
    args_or_dict: object,
    split_metadata: Mapping[str, Any] | None = None,
    process_semantic_config: ProcessSemanticConfig | None = None,
) -> dict[str, Any]:
    values = _config_values(args_or_dict)
    payload = model_fingerprint_payload(
        args_or_dict,
        split_metadata=split_metadata,
        process_semantic_config=process_semantic_config,
    )
    payload.pop("cache_scope", None)
    payload["event_score_mode"] = str(
        values.get("event_score_mode", SlimConfig.event_score_mode),
    )
    return payload


def _effective_process_semantic_fingerprint(
    dataset: str,
    process_cfg: ProcessSemanticConfig | None,
) -> dict[str, Any]:
    if process_cfg is None or str(process_cfg.rules_version).lower() == "legacy":
        return {"process_semantic_rules_version": "legacy"}
    return {
        "process_semantic_rules_version": str(process_cfg.rules_version),
        "dataset_os_profile": resolve_dataset_profile(dataset, process_cfg),
        "process_nll_max_tokens": int(process_cfg.process_nll_max_tokens),
        "process_residual_max_tokens": int(process_cfg.process_residual_max_tokens),
        "path_parent_depth": int(process_cfg.path_parent_depth),
        "package_namespace_segments": int(process_cfg.package_namespace_segments),
        "package_component_index": int(process_cfg.package_component_index),
        "file_nll_max_tokens": int(process_cfg.file_nll_max_tokens),
        "semantic_sketch_latent_dim": int(process_cfg.semantic_sketch_latent_dim),
        "semantic_sketch_max_tokens": int(process_cfg.semantic_sketch_max_tokens),
    }


def cache_path(cache_dir: str | os.PathLike[str], dataset: object, fingerprint: object) -> Path:
    """Return the deterministic precompute cache path for a dataset and fingerprint."""
    dataset_text = str(dataset).strip() or "unknown"
    safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_text).strip("._-")
    if not safe_dataset:
        safe_dataset = "unknown"
    fingerprint_text = str(fingerprint).strip()
    if not fingerprint_text:
        raise ValueError("fingerprint must not be empty")
    return Path(cache_dir) / f"{safe_dataset}_{fingerprint_text}.pkl"


def leakage_contract() -> dict[str, bool]:
    """Return the cache leakage contract for train/validation-only artifacts."""
    return {
        "contains_train_artifacts": True,
        "contains_validation_unlabeled_calibration": True,
        "contains_test_scores": False,
        "contains_test_labels": False,
        "contains_test_rankings": False,
        "ground_truth_used": False,
    }


def _cache_mode_writes_base_cache(cache_mode: str) -> bool:
    return str(cache_mode) in {"write", "auto", "force"}


def _cache_info_payload(
    *,
    cache_scope: str = "train_validation_base_scores",
    cache_path: str = "",
    cache_fingerprint: str = "",
    cache_hit: bool = False,
    cache_mode: str = "off",
    event_score_mode_in_fingerprint: bool = False,
    validation_event_scores_derived_at_runtime: bool = True,
    validation_paired_raw_scores_cached: bool = False,
    legacy_event_mode_cache: bool = False,
    train_validation_reused_from_cache: bool = False,
    train_rows_processed: int = 0,
    validation_rows_processed: int = 0,
    validation_event_score_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return normalized cache metadata for eval JSON."""
    return {
        "cache_scope": str(cache_scope),
        "cache_path": str(cache_path),
        "cache_fingerprint": str(cache_fingerprint),
        "cache_hit": bool(cache_hit),
        "cache_mode": str(cache_mode),
        "event_score_mode_in_fingerprint": bool(event_score_mode_in_fingerprint),
        "node_pool_score_mode_in_fingerprint": False,
        "validation_event_scores_derived_at_runtime": bool(
            validation_event_scores_derived_at_runtime,
        ),
        "validation_paired_raw_scores_cached": bool(validation_paired_raw_scores_cached),
        "legacy_event_mode_cache": bool(legacy_event_mode_cache),
        "train_validation_reused_from_cache": bool(train_validation_reused_from_cache),
        "train_rows_processed": int(train_rows_processed),
        "validation_rows_processed": int(validation_rows_processed),
        "validation_event_score_summary": dict(validation_event_score_summary or {}),
        "mode": str(cache_mode),
        "hit": bool(cache_hit),
        "path": str(cache_path),
        "fingerprint": str(cache_fingerprint),
    }


def build_cache_payload(
    args_or_dict: object,
    model: SSPMLowRankModel,
    validation_residual_scores: Sequence[float],
    train_count: int,
    validation_count: int,
    split_metadata: Mapping[str, Any] | None = None,
    process_semantic_config: ProcessSemanticConfig | None = None,
    embedder: ResidualEmbedder | None = None,
) -> dict[str, Any]:
    """Build a label-free train/validation precompute cache payload."""
    residual_raw = np.asarray(validation_residual_scores, dtype=np.float32)
    if int(validation_count) != int(residual_raw.size):
        raise ValueError("validation_count must equal residual score length")
    fingerprint_payload = model_fingerprint_payload(
        args_or_dict,
        split_metadata=split_metadata,
        process_semantic_config=process_semantic_config,
    )
    if embedder is None:
        cfg = SlimConfig(**_config_values(args_or_dict))
        embedder = build_residual_embedder(
            _make_residual_embedding_config(cfg, process_semantic_config),
        )
        embedder.fit([])
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "method": METHOD_VERSION,
        "cache_scope": "train_validation_base_scores",
        "dataset": fingerprint_payload["dataset"],
        "fingerprint_payload": fingerprint_payload,
        "leakage_contract": leakage_contract(),
        "model": model.state_dict(),
        "embedder": embedder.state_dict(),
        "validation_residual_scores": residual_raw,
        "validation_residual_sorted": _compact_sorted_scores(residual_raw),
        "train_count": int(train_count),
        "validation_count": int(validation_count),
    }
    validate_cache_payload(payload)
    return payload


def write_cache(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Atomically write a validated precompute cache payload with pickle."""
    validate_cache_payload(payload)
    cache_file = Path(path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_name(
        f".{cache_file.name}.{os.getpid()}.{time.time_ns()}.tmp",
    )
    try:
        with tmp_file.open("wb") as handle:
            pickle.dump(dict(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        with tmp_file.open("rb") as handle:
            written_payload = pickle.load(handle)
        if not isinstance(written_payload, dict):
            raise TypeError("cache payload must be a dict")
        validate_cache_payload(written_payload)
        os.replace(tmp_file, cache_file)
    finally:
        if tmp_file.exists():
            tmp_file.unlink()


def read_cache(
    path: str | os.PathLike[str],
    expected_fingerprint_payload: Mapping[str, Any] | None = None,
    expected_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Read and validate a precompute cache payload from pickle."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("cache payload must be a dict")
    validate_cache_payload(payload)
    if payload.get("schema") != CACHE_SCHEMA_VERSION:
        raise ValueError("cache schema mismatch")
    if payload.get("method") != METHOD_VERSION:
        raise ValueError("cache method mismatch")
    if expected_fingerprint_payload is not None:
        expected_payload = dict(expected_fingerprint_payload)
        actual_payload = payload.get("fingerprint_payload")
        if actual_payload != expected_payload:
            raise ValueError(
                f"cache fingerprint payload mismatch for {path}: "
                f"expected={expected_payload!r} actual={actual_payload!r}",
            )
    if expected_fingerprint is not None:
        actual_fingerprint = stable_json_hash(payload.get("fingerprint_payload", {}))
        if actual_fingerprint != str(expected_fingerprint):
            raise ValueError(
                f"cache fingerprint mismatch for {path}: "
                f"expected={expected_fingerprint} actual={actual_fingerprint}",
            )
    return payload


def _is_legacy_event_mode_cache(payload: Mapping[str, Any]) -> bool:
    """Return true for pre-base-cache payloads that cached event-mode scores."""
    return (
        payload.get("cache_scope") in {None, "", "legacy_event_mode_cache"}
        and "validation_event_sorted" in payload
    )


def _legacy_cache_event_score_mode(payload: Mapping[str, Any]) -> str:
    fingerprint_payload = payload.get("fingerprint_payload", {})
    if isinstance(fingerprint_payload, Mapping):
        return str(fingerprint_payload.get("event_score_mode", ""))
    return str(payload.get("event_score_mode", ""))


def _restore_cache_artifacts(
    payload: Mapping[str, Any],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None = None,
) -> dict[str, Any]:
    """Restore train/validation cache artifacts and drop the source payload reference."""
    model = SSPMLowRankModel.from_state_dict(payload["model"])
    embedder = build_residual_embedder(_make_residual_embedding_config(config, process_cfg))
    if payload.get("cache_scope") == "train_validation_base_scores":
        validation_residual_scores = np.asarray(
            payload["validation_residual_scores"],
            dtype=np.float32,
        )
        validation_residual_sorted = _compact_sorted_scores(
            payload.get("validation_residual_sorted", validation_residual_scores),
        )
        restored = {
            "model": model,
            "embedder": embedder,
            "validation_residual_scores": validation_residual_scores,
            "validation_residual_sorted": validation_residual_sorted,
            "residual_scores": validation_residual_sorted,
            "train_count": int(payload.get("train_count", 0)),
            "validation_count": int(payload.get("validation_count", 0)),
            "cache_scope": "train_validation_base_scores",
            "payload": None,
            "released_payload": True,
        }
        payload = None
        return restored
    raise ValueError(
        "legacy semantic NLL event-mode caches are archived and no longer supported "
        "in active SSPM pipeline. Regenerate a residual-only cache.",
    )


def validate_cache_payload(payload: Mapping[str, Any]) -> None:
    """Reject cache payloads containing test scores, labels, rankings, or ground truth."""
    if not isinstance(payload, Mapping):
        raise TypeError("cache payload must be a mapping")
    _validate_cache_payload_recursive(payload, ())
    _validate_base_cache_payload(payload)
    _validate_cache_state_shapes(payload)


def _validate_base_cache_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("cache_scope") != "train_validation_base_scores":
        return
    for key in _BASE_CACHE_FORBIDDEN_KEYS:
        if key in payload:
            raise ValueError(f"base cache must not contain {key}")
    residual_scores = np.asarray(payload.get("validation_residual_scores"), dtype=np.float32)
    if residual_scores.size == 0:
        raise ValueError("base cache residual validation scores must not be empty")
    if not bool(np.all(np.isfinite(residual_scores))):
        raise ValueError("base cache residual validation scores must be finite")
    if int(payload.get("validation_count", -1)) != int(residual_scores.size):
        raise ValueError("validation_count mismatch")
    residual_sorted = np.asarray(
        payload.get("validation_residual_sorted", _compact_sorted_scores(residual_scores)),
        dtype=np.float32,
    )
    if residual_sorted.shape != residual_scores.shape:
        raise ValueError("validation_residual_sorted shape mismatch")
    if not np.array_equal(residual_sorted, _compact_sorted_scores(residual_scores)):
        raise ValueError("validation_residual_sorted does not match raw scores")


def _validate_cache_payload_recursive(value: Any, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            key_norm = normalize_piece(key_text, max_len=120)
            current_path = (*path, key_text)
            if key_norm in _CACHE_FORBIDDEN_KEYS:
                raise ValueError(f"cache payload must not contain {'.'.join(current_path)}")
            if key_norm in _CACHE_SAFETY_FLAG_KEYS and bool(item):
                raise ValueError(f"cache payload has unsafe true flag {'.'.join(current_path)}")
            _validate_cache_payload_recursive(item, current_path)
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_cache_payload_recursive(item, (*path, str(index)))


def _validate_cache_state_shapes(payload: Mapping[str, Any]) -> None:
    if "model" in payload:
        _validate_sspm_state(payload["model"])


def _validate_sspm_state(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("cache model must be an explicit state dict")
    required = {"config", "c1", "c2", "bias"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"cache model state missing keys: {missing}")
    blocked = {"node_to_slot", "state_array", "_state_array", "global_state"}
    present = sorted(blocked.intersection(value))
    if present:
        raise ValueError(f"cache model state contains online state keys: {present}")
    _reject_nested_online_model_state(value, ())


def _reject_nested_online_model_state(value: Any, path: tuple[str, ...]) -> None:
    blocked = {"node_to_slot", "state_array", "_state_array", "global_state"}
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            current_path = (*path, key_text)
            if key_text in blocked:
                joined = ".".join(current_path)
                raise ValueError(f"cache model state contains online state key: {joined}")
            _reject_nested_online_model_state(item, current_path)
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_nested_online_model_state(item, (*path, str(index)))


def _checked_sorted_scores(sorted_scores: Sequence[float]) -> list[float]:
    scores = [float(score) for score in sorted_scores]
    if not scores:
        raise ValueError("validation scores must not be empty")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("validation scores must be finite")
    if any(scores[index] > scores[index + 1] for index in range(len(scores) - 1)):
        scores = sorted(scores)
    return scores


def _compact_sorted_scores(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float32)
    if array.size == 0:
        return np.asarray([0.0], dtype=np.float32)
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("validation scores must be finite")
    array.sort()
    return array


def _int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, str):
        return _parse_int_list(values)
    out: list[int] = []
    for value in values:
        out.append(int(value))
    return out


def _config_values(args_or_dict: object) -> dict[str, Any]:
    if isinstance(args_or_dict, Mapping):
        values = dict(args_or_dict)
    elif is_dataclass(args_or_dict) and not isinstance(args_or_dict, type):
        values = asdict(args_or_dict)
    elif isinstance(args_or_dict, argparse.Namespace):
        values = vars(args_or_dict).copy()
    else:
        values = {}
        for field in SlimConfig.__dataclass_fields__:
            if hasattr(args_or_dict, field):
                values[field] = getattr(args_or_dict, field)
    if "sspm_epochs" not in values and "sspm_train_epochs" in values:
        values["sspm_epochs"] = values["sspm_train_epochs"]
    return values


def _cache_sorted_scores(scores: Sequence[float]) -> list[float]:
    return [float(score) for score in _compact_sorted_scores(scores)]


def _relation_name(relation_id: int) -> str:
    for name, value in RELATIONS.items():
        if int(value) == int(relation_id):
            return name
    return "other"


# Explicit cross-module imports; replaces the migration namespace bridge.
from scripts.pipeline.checks.preflight import (
    _make_residual_embedding_config,
    _resolve_db_split_metadata,
    _resolve_slim_split_metadata,
    _slim_split_override_days,
    build_node_maps,
    load_ground_truth_indices_readonly,
)
from scripts.pipeline.features.semantic_features import (
    _score_summary_from_event_calibration,
    normalize_piece,
)
from scripts.pipeline.io.db_stream import _cfg_for_dataset
from scripts.pipeline.io.event_artifacts import stable_json_hash
from scripts.pipeline.outputs.alert_output import _parse_int_list
