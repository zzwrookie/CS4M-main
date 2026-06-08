"""Legacy synthetic and DB fallback paths kept outside the E4 conditional best path."""

from __future__ import annotations

from scripts.pipeline.config.runtime_config import *


def synthetic_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return small train/validation/test netflow rows for smoke validation."""
    train = [
        _synthetic_row(1, 1, 101, 201, "128.55.12.10", "80", "benign"),
        _synthetic_row(2, 2, 101, 202, "128.55.12.11", "80", "benign"),
        _synthetic_row(3, 3, 102, 203, "128.55.12.12", "443", "benign"),
        _synthetic_row(4, 4, 102, 204, "128.55.12.13", "443", "benign"),
    ]
    validation = [
        _synthetic_row(101, 101, 101, 205, "128.55.12.14", "80", "benign"),
        _synthetic_row(102, 102, 102, 206, "128.55.12.15", "443", "benign"),
        _synthetic_row(103, 103, 103, 207, "128.55.12.16", "80", "benign"),
    ]
    test = [
        _synthetic_row(201, 201, 101, 208, "128.55.12.17", "80", "benign"),
        _synthetic_row(202, 202, 102, 209, "203.0.113.77", "4444", "suspicious"),
        _synthetic_row(203, 203, 102, 210, "198.51.100.44", "5555", "malicious"),
        _synthetic_row(204, 204, 103, 211, "128.55.12.18", "443", "benign"),
    ]
    return train, validation, test


def run_synthetic_smoke(config: SlimConfig) -> Path:
    """Run the label-safe synthetic slim causal semantics smoke pipeline."""
    _validate_active_event_score_mode(config.event_score_mode)
    started = time.perf_counter()
    timing = _empty_timing()
    memory_profile = _empty_memory_profile()
    process_cfg = _load_process_config(config)
    process_audit = (
        ProcessSemanticAudit(config.dataset, process_cfg, labels_used=False)
        if config.process_semantic_audit
        else None
    )
    train_rows, validation_rows, test_rows = synthetic_rows()
    train_rows = _limit_rows(train_rows, config.max_train_events)
    validation_rows = _limit_rows(validation_rows, config.max_ref_events)
    test_rows = _limit_rows(test_rows, config.max_test_events)
    test_label_rows = test_rows
    test_rows = None
    output_dir = Path(config.result_root) / config.out_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_profile["rss_after_node_map_mb"] = _current_rss_mb()

    phase_started = time.perf_counter()
    if _uses_real_diag_online_gamma_training(config):
        sspm, embedder, train_count = train_real_diag_gamma_from_stream(
            iter(train_rows),
            config,
            process_cfg,
            process_audit,
        )
        train_sentences = [None] * int(train_count)
        train_contexts: list[np.ndarray] = []
        train_targets: list[np.ndarray] = []
        sspm_training = dict(sspm.training_stats)
    else:
        embedder = build_residual_embedder(_make_residual_embedding_config(config, process_cfg))
        sspm = SSPMLowRankModel(_make_sspm_config(config, process_cfg))
        train_contexts = []
        train_targets = []
        train_sentences = [
            _residual_tokens_for_row(row, config, process_cfg)
            for row in train_rows
        ]
        embedder.fit(train_sentences)
        for row in train_rows:
            _observe_process_semantic_audit(
                process_audit,
                "train",
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
            train_contexts.append(sspm.make_context(fields))
            train_targets.append(z)
            sspm.update_states(fields, z)
        if train_contexts:
            sspm_training = sspm.train_batch(
                np.stack(train_contexts),
                np.stack(train_targets),
                log_prefix=f"[SSPM][{config.dataset}]",
            )
        else:
            sspm_training = {}
        sspm.training_stats = sspm_training
    timing["train_seconds"] = float(time.perf_counter() - phase_started)
    memory_profile["rss_after_train_mb"] = _current_rss_mb()

    phase_started = time.perf_counter()
    if _uses_action_diag_calibration(config):
        fit_residual_calibration_from_stream(
            validation_rows,
            config,
            sspm,
            embedder,
            process_cfg,
            process_audit,
        )
    validation_residual_scores, validation_count = calibrate_from_stream(
        validation_rows,
        config,
        sspm,
        embedder,
        process_cfg,
        process_audit,
    )
    _fit_update_gate_if_needed(config, sspm, validation_residual_scores)
    horizon = config.expected_alert_horizon_events or max(len(test_label_rows), 1)
    event_calibration = derive_event_calibration(
        validation_residual_scores,
        config.event_score_mode,
        config.expected_event_alert_budget,
        horizon,
        threshold_mode=config.event_threshold_mode,
        fixed_quantile=config.event_threshold_quantile,
    )
    residual_scores = event_calibration.residual_sorted
    event_scores = event_calibration.event_sorted
    threshold = event_calibration.threshold
    validation_event_score_summary = event_calibration.summary
    timing["validation_seconds"] = float(time.perf_counter() - phase_started)
    memory_profile["rss_after_validation_mb"] = _current_rss_mb()

    memory_profile["rss_before_test_cleanup_mb"] = _current_rss_mb()
    train_contexts.clear()
    train_targets.clear()
    validation_residual_scores = None
    event_calibration = None
    event_scores = None
    train_rows = None
    validation_rows = None
    _finish_test_phase_cleanup(
        memory_profile,
        released_payload=False,
        node_map_mode="synthetic_no_db_node_maps",
        released_train_validation=True,
    )
    stream_outputs = _score_test_stream(
        config,
        embedder,
        sspm,
        _label_free_rows(test_label_rows),
        residual_scores,
        threshold,
        process_cfg=process_cfg,
        process_audit=process_audit,
        output_dir=output_dir,
        collect_in_memory=_task6_collect_in_memory_outputs(config),
        initial_test_peak_rss_mb=memory_profile["rss_after_test_cleanup_mb"],
    )
    timing["test_scoring_seconds"] = float(stream_outputs.get("test_scoring_seconds", 0.0))
    memory_profile["rss_test_peak_mb"] = float(stream_outputs.get("rss_test_peak_mb", 0.0))
    memory_profile["deploy_rss_valid"] = bool(_deploy_light_enabled(config))
    memory_profile["deploy_infer_rss_peak_mb"] = float(memory_profile["rss_test_peak_mb"])
    memory_profile["deploy_infer_rss_final_mb"] = float(_current_rss_mb())
    memory_profile["deploy_infer_events_per_sec"] = float(
        stream_outputs.get("test_count", 0)
    ) / max(float(stream_outputs.get("test_scoring_seconds", 0.0)), 1e-9)
    phase_started = time.perf_counter()
    stream_outputs["event_labels_by_event"] = _labels_from_rows(test_label_rows)
    stream_outputs["node_labels_by_node"] = _node_labels_from_rows(
        test_label_rows,
        config.max_tokens_per_node,
        dataset=config.dataset,
        process_semantic_config=process_cfg,
    )
    stream_outputs.update(_embedding_eval_metadata(config, embedder))
    timing["label_attach_seconds"] = float(time.perf_counter() - phase_started)
    memory_profile["rss_after_label_attach_mb"] = _current_rss_mb()
    memory_profile["state_slots"] = _safe_state_slots(sspm)
    memory_profile["state_array_mb"] = _safe_state_array_mb(sspm)
    elapsed = max(time.perf_counter() - started, 1e-9)
    eval_payload = _eval_payload(
        config,
        stream_outputs,
        sspm,
        elapsed,
        len(test_label_rows),
        timing,
        memory_profile,
    )
    eval_payload["validation_event_score_summary"] = validation_event_score_summary
    eval_payload["train_events_actual"] = int(len(train_sentences))
    eval_payload["validation_events_actual"] = int(validation_count)
    eval_payload["test_events_actual"] = int(len(test_label_rows))
    write_effective_config(
        output_dir,
        config,
        sspm,
        _embedder_loaded(config),
        train_count=len(train_sentences),
        validation_count=int(validation_count),
        test_count=len(test_label_rows),
    )
    _write_process_semantic_audit_outputs(output_dir, process_audit)
    final_eval_payload = _write_outputs(output_dir, stream_outputs, eval_payload)
    if final_eval_payload is not None:
        eval_payload = final_eval_payload
    write_metrics_json(
        output_dir,
        config,
        sspm,
        eval_payload,
        train_count=len(train_sentences),
        validation_count=int(validation_count),
        test_count=len(test_label_rows),
        embedder_loaded=_embedder_loaded(config),
    )
    return output_dir / "eval_causal_semantics_slim.json"


def run_db(args: argparse.Namespace, config: SlimConfig) -> Path:
    """Run the DB streaming slim pipeline with label-free scoring."""
    _validate_active_event_score_mode(config.event_score_mode)
    started = time.perf_counter()
    timing = _empty_timing()
    memory_profile = _empty_memory_profile()
    output_dir = Path(config.result_root) / config.out_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    process_cfg = _load_process_config(config)
    process_audit = (
        ProcessSemanticAudit(config.dataset, process_cfg, labels_used=False)
        if config.process_semantic_audit
        else None
    )
    db_cfg = _cfg_for_dataset(config.dataset)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    cur, conn = init_database_connection(db_cfg)
    cache_info: dict[str, Any] = _cache_info_payload(
        cache_mode=str(config.precompute_cache_mode),
    )
    train_seconds = 0.0
    validation_seconds = 0.0
    evaluation_label_seconds = 0.0
    train_count = 0
    validation_count = 0
    cache_payload_released = False
    stream_status = _db_stream_status(config.db_stream_mode, "python_lookup", "")
    train_artifact_count = 0
    validation_artifact_count = 0
    try:
        phase_started = time.perf_counter()
        node_maps = build_node_maps(cur)
        timing["node_map_seconds"] = float(time.perf_counter() - phase_started)
        memory_profile["rss_after_node_map_mb"] = _current_rss_mb()
        split_metadata = _resolve_db_split_metadata(config, db_cfg)
        year_month = str(split_metadata["year_month"])
        train_days = list(split_metadata["train_days"])
        val_days = list(split_metadata["validation_days"])
        test_days = list(split_metadata["test_days"])
        event_filter = use_event_type_filter(db_cfg)
        fingerprint_payload = model_fingerprint_payload(
            config,
            split_metadata=split_metadata,
            process_semantic_config=process_cfg,
        )
        fingerprint = stable_json_hash(fingerprint_payload)
        legacy_fingerprint_payload = _legacy_event_mode_fingerprint_payload(
            config,
            split_metadata=split_metadata,
            process_semantic_config=process_cfg,
        )
        legacy_fingerprint = stable_json_hash(legacy_fingerprint_payload)
        cache_mode = str(config.precompute_cache_mode)
        cache_info.update(
            _cache_info_payload(
                cache_fingerprint=fingerprint,
                cache_mode=cache_mode,
            ),
        )
        cache_file: Path | None = None
        legacy_cache_file: Path | None = None
        if config.precompute_cache_dir:
            cache_file = cache_path(config.precompute_cache_dir, config.dataset, fingerprint)
            legacy_cache_file = cache_path(
                config.precompute_cache_dir,
                config.dataset,
                legacy_fingerprint,
            )
            cache_info["cache_path"] = str(cache_file)
            cache_info["path"] = str(cache_file)
        payload = None
        attempted_base_cache = False
        if cache_mode in {"read", "auto"} and cache_file is not None and cache_file.exists():
            attempted_base_cache = True
            phase_started = time.perf_counter()
            try:
                payload = read_cache(
                    cache_file,
                    expected_fingerprint_payload=fingerprint_payload,
                    expected_fingerprint=fingerprint,
                )
            except Exception as exc:
                if cache_mode != "auto":
                    raise
                timing["cache_read_seconds"] += float(time.perf_counter() - phase_started)
                cache_info["invalid_cache_reason"] = str(exc)
                payload = None
            else:
                timing["cache_read_seconds"] += float(time.perf_counter() - phase_started)
                cache_info["hit"] = True
                cache_info["cache_hit"] = True
        if (
            payload is None
            and cache_mode in {"read", "auto"}
            and legacy_cache_file is not None
            and legacy_cache_file.exists()
        ):
            phase_started = time.perf_counter()
            try:
                payload = read_cache(
                    legacy_cache_file,
                    expected_fingerprint_payload=legacy_fingerprint_payload,
                    expected_fingerprint=legacy_fingerprint,
                )
            except Exception as exc:
                if cache_mode != "auto":
                    raise
                timing["cache_read_seconds"] += float(time.perf_counter() - phase_started)
                cache_info["invalid_legacy_cache_reason"] = str(exc)
                payload = None
            else:
                timing["cache_read_seconds"] += float(time.perf_counter() - phase_started)
                cache_info["cache_path"] = str(legacy_cache_file)
                cache_info["path"] = str(legacy_cache_file)
                cache_info["cache_fingerprint"] = legacy_fingerprint
                cache_info["fingerprint"] = legacy_fingerprint
                cache_info["hit"] = True
                cache_info["cache_hit"] = True
        elif cache_mode == "read" and not attempted_base_cache:
            raise FileNotFoundError(f"precompute cache not found: {cache_file}")
        if payload is not None:
            is_legacy_cache = _is_legacy_event_mode_cache(payload)
            if is_legacy_cache:
                reason = (
                    "legacy semantic NLL event-mode cache is archived and unsupported "
                    "by active residual-only pipeline"
                )
                if cache_mode != "auto":
                    raise ValueError(reason)
                cache_info["invalid_cache_reason"] = reason
                payload = None
        if str(config.sspm_train_mode) == "load_and_infer":
            if payload is not None:
                cache_info["skipped_precompute_cache_reason"] = "load_and_infer_uses_checkpoint"
                payload = None
            phase_started = time.perf_counter()
            model, checkpoint_payload = load_sspm_checkpoint(config.sspm_checkpoint_path, config)
            embedder = load_pretrained_residual_embedder(
                str(config.pretrained_residual_embedder_path),
                expected_dim=_effective_latent_dim(config, process_cfg),
            )
            validate_checkpoint_embedder_fingerprints(
                checkpoint_payload,
                embedder,
                config,
                process_cfg,
            )
            embedder = maybe_wrap_residual_embed_cache(embedder, config, process_cfg)
            timing["cache_read_seconds"] += float(time.perf_counter() - phase_started)
            counts = dict(checkpoint_payload.get("event_counts_actual", {}))
            train_artifact_count = int(counts.get("train_events_actual", 0))
            train_count = 0
            validation_count = 0
            memory_profile["rss_after_train_mb"] = _current_rss_mb()
            phase_started = time.perf_counter()
            validation_residual_scores, validation_count = calibrate_from_stream(
                stream_dataset_rows(
                    conn,
                    year_month,
                    val_days,
                    node_maps,
                    event_filter,
                    config.fetch_size,
                    config.max_ref_events,
                    abnormal_nodes=set(),
                ),
                config,
                model,
                embedder,
                process_cfg,
                process_audit,
            )
            horizon = config.expected_alert_horizon_events or max(config.max_test_events, 1)
            event_calibration = derive_event_calibration(
                validation_residual_scores,
                config.event_score_mode,
                config.expected_event_alert_budget,
                horizon,
                threshold_mode=config.event_threshold_mode,
                fixed_quantile=config.event_threshold_quantile,
            )
            residual_scores = event_calibration.residual_sorted
            event_scores = event_calibration.event_sorted
            threshold = event_calibration.threshold
            validation_event_score_summary = event_calibration.summary
            validation_artifact_count = int(validation_count)
            validation_seconds = float(time.perf_counter() - phase_started)
            timing["validation_seconds"] = validation_seconds
            memory_profile["rss_after_validation_mb"] = _current_rss_mb()
            cache_info.update(
                {
                    "cache_scope": "sspm_checkpoint_fanout",
                    "cache_mode": str(config.precompute_cache_mode),
                    "checkpoint_path": str(config.sspm_checkpoint_path),
                    "skip_train": True,
                    "train_mode": str(config.sspm_train_mode),
                    "train_data_mode": str(config.sspm_train_data_mode),
                    "train_rows_processed": 0,
                    "validation_rows_processed": int(validation_count),
                    "validation_event_score_summary": validation_event_score_summary,
                    "residual_embedding_cache": _residual_embed_cache_summary(
                        embedder,
                        config,
                        process_cfg,
                    ),
                },
            )
        if payload is not None:
            is_legacy_cache = _is_legacy_event_mode_cache(payload)
            restored = _restore_cache_artifacts(payload, config, process_cfg)
            model = restored["model"]
            embedder = restored["embedder"]
            if is_legacy_cache:
                raise ValueError("legacy semantic NLL cache should have been rejected")
            else:
                train_artifact_count = int(restored["train_count"])
                validation_artifact_count = int(restored["validation_count"])
                _fit_update_gate_if_needed(
                    config,
                    model,
                    restored["validation_residual_scores"],
                )
                horizon = config.expected_alert_horizon_events or max(config.max_test_events, 1)
                event_calibration = derive_event_calibration(
                    restored["validation_residual_scores"],
                    config.event_score_mode,
                    config.expected_event_alert_budget,
                    horizon,
                    residual_sorted=restored["validation_residual_sorted"],
                    threshold_mode=config.event_threshold_mode,
                    fixed_quantile=config.event_threshold_quantile,
                )
                residual_scores = event_calibration.residual_sorted
                event_scores = event_calibration.event_sorted
                threshold = event_calibration.threshold
                validation_event_score_summary = event_calibration.summary
                cache_info.update(
                    _cache_info_payload(
                        cache_scope="train_validation_base_scores",
                        cache_path=str(cache_file or ""),
                        cache_fingerprint=fingerprint,
                        cache_hit=True,
                        cache_mode=cache_mode,
                        validation_event_scores_derived_at_runtime=True,
                        validation_paired_raw_scores_cached=True,
                        legacy_event_mode_cache=False,
                        train_validation_reused_from_cache=True,
                        train_rows_processed=0,
                        validation_rows_processed=0,
                        validation_event_score_summary=validation_event_score_summary,
                    ),
                )
            train_count = 0
            validation_count = 0
            payload = restored["payload"]
            cache_payload_released = bool(restored["released_payload"])
            restored = None
            memory_profile["rss_after_train_mb"] = _current_rss_mb()
            memory_profile["rss_after_validation_mb"] = memory_profile["rss_after_train_mb"]
        elif str(config.sspm_train_mode) == "train_and_save":
            phase_started = time.perf_counter()
            if str(config.sspm_train_data_mode) == "stream_event":
                def train_row_factory():
                    return stream_dataset_rows(
                        conn,
                        year_month,
                        train_days,
                        node_maps,
                        event_filter,
                        config.fetch_size,
                        config.max_train_events,
                        abnormal_nodes=set(),
                    )

                if _uses_real_diag_online_gamma_training(config):
                    model, embedder, train_count = train_real_diag_gamma_from_factory(
                        train_row_factory,
                        config,
                        process_cfg,
                        process_audit,
                    )
                else:
                    model, embedder, train_count = train_stream_event_from_factory(
                        train_row_factory,
                        config,
                        process_cfg,
                        process_audit,
                    )
            else:
                train_rows_stream = stream_dataset_rows(
                    conn,
                    year_month,
                    train_days,
                    node_maps,
                    event_filter,
                    config.fetch_size,
                    config.max_train_events,
                    abnormal_nodes=set(),
                )
                if _uses_real_diag_online_gamma_training(config):
                    model, embedder, train_count = train_real_diag_gamma_from_stream(
                        train_rows_stream,
                        config,
                        process_cfg,
                        process_audit,
                    )
                else:
                    model, embedder, train_count = train_from_stream(
                        train_rows_stream,
                        config,
                        process_cfg,
                        process_audit,
                    )
            train_seconds = float(time.perf_counter() - phase_started)
            timing["train_seconds"] = train_seconds
            memory_profile["rss_after_train_mb"] = _current_rss_mb()
            phase_started = time.perf_counter()
            if _uses_action_diag_calibration(config):
                fit_residual_calibration_from_stream(
                    stream_dataset_rows(
                        conn,
                        year_month,
                        val_days,
                        node_maps,
                        event_filter,
                        config.fetch_size,
                        config.max_ref_events,
                        abnormal_nodes=set(),
                    ),
                    config,
                    model,
                    embedder,
                    process_cfg,
                    process_audit,
                )
            calibration = calibrate_from_stream(
                stream_dataset_rows(
                    conn,
                    year_month,
                    val_days,
                    node_maps,
                    event_filter,
                    config.fetch_size,
                    config.max_ref_events,
                    abnormal_nodes=set(),
                ),
                config,
                model,
                embedder,
                process_cfg,
                process_audit,
            )
            validation_residual_scores, validation_count = calibration
            _fit_update_gate_if_needed(config, model, validation_residual_scores)
            horizon = config.expected_alert_horizon_events or max(config.max_test_events, 1)
            event_calibration = derive_event_calibration(
                validation_residual_scores,
                config.event_score_mode,
                config.expected_event_alert_budget,
                horizon,
                threshold_mode=config.event_threshold_mode,
                fixed_quantile=config.event_threshold_quantile,
            )
            residual_scores = event_calibration.residual_sorted
            event_scores = event_calibration.event_sorted
            threshold = event_calibration.threshold
            validation_event_score_summary = event_calibration.summary
            train_artifact_count = int(train_count)
            validation_artifact_count = int(validation_count)
            if str(config.sspm_checkpoint_path).strip():
                save_sspm_checkpoint(
                    config.sspm_checkpoint_path,
                    config=config,
                    model=model,
                    embedder=embedder,
                    process_cfg=process_cfg,
                    train_count=int(train_count),
                    validation_count=int(validation_count),
                )
            validation_seconds = float(time.perf_counter() - phase_started)
            timing["validation_seconds"] = validation_seconds
            memory_profile["rss_after_validation_mb"] = _current_rss_mb()
            cache_info.update(
                _cache_info_payload(
                    cache_scope="train_validation_base_scores",
                    cache_path=str(cache_file or ""),
                    cache_fingerprint=fingerprint,
                    cache_hit=False,
                    cache_mode=cache_mode,
                    validation_event_scores_derived_at_runtime=True,
                    validation_paired_raw_scores_cached=bool(
                        _cache_mode_writes_base_cache(cache_mode) and cache_file is not None,
                    ),
                    legacy_event_mode_cache=False,
                    train_validation_reused_from_cache=False,
                    train_rows_processed=train_count,
                    validation_rows_processed=validation_count,
                    validation_event_score_summary=validation_event_score_summary,
                ),
            )
            cache_info["checkpoint_path"] = str(config.sspm_checkpoint_path)
            cache_info["skip_train"] = False
            cache_info["train_mode"] = str(config.sspm_train_mode)
            cache_info["train_data_mode"] = str(config.sspm_train_data_mode)
            cache_info["residual_embedding_cache"] = _residual_embed_cache_summary(
                embedder,
                config,
                process_cfg,
            )
            if _cache_mode_writes_base_cache(cache_mode) and cache_file is not None:
                phase_started = time.perf_counter()
                write_cache(
                    cache_file,
                    build_cache_payload(
                        config,
                        model,
                        validation_residual_scores,
                        train_count,
                        validation_count,
                        split_metadata=split_metadata,
                        process_semantic_config=process_cfg,
                        embedder=embedder,
                    ),
                )
                timing["cache_write_seconds"] += float(time.perf_counter() - phase_started)
        else:
            raise ValueError(f"unsupported SSPM_TRAIN_MODE={config.sspm_train_mode}")
        validation_residual_scores = None
        event_calibration = None
        event_scores = None
        memory_profile["rss_before_test_cleanup_mb"] = _current_rss_mb()
        test_node_maps = _test_scoring_node_maps(node_maps)
        node_maps = None
        payload = None
        calibration = None
        test_rows = stream_dataset_rows_slim(
            conn,
            year_month,
            test_days,
            test_node_maps,
            event_filter,
            config.fetch_size,
            config.max_test_events,
            config.db_stream_mode,
            abnormal_nodes=set(),
            stream_status=stream_status,
        )
        if stream_status.get("db_stream_mode_actual") == "temp_table":
            node_map_mode = "db_temp_table_compact_stream"
        else:
            node_map_mode = "db_scoring_view_without_uuid2index_rebuild_for_eval"
        test_node_maps = None
        _finish_test_phase_cleanup(
            memory_profile,
            released_payload=cache_payload_released,
            node_map_mode=node_map_mode,
            released_train_validation=True,
        )
        stream_outputs = _score_test_stream(
            config,
            embedder,
            model,
            test_rows,
            residual_scores,
            threshold,
            process_cfg=process_cfg,
            process_audit=process_audit,
            output_dir=output_dir,
            collect_in_memory=_task6_collect_in_memory_outputs(config),
            initial_test_peak_rss_mb=memory_profile["rss_after_test_cleanup_mb"],
        )
        stream_outputs = dict(stream_outputs)
        stream_outputs["db_stream_status"] = dict(stream_status)
        timing["test_scoring_seconds"] = float(stream_outputs.get("test_scoring_seconds", 0.0))
        memory_profile["rss_test_peak_mb"] = float(stream_outputs.get("rss_test_peak_mb", 0.0))
        memory_profile["deploy_rss_valid"] = bool(_deploy_light_enabled(config))
        memory_profile["deploy_infer_rss_peak_mb"] = float(memory_profile["rss_test_peak_mb"])
        memory_profile["deploy_infer_rss_final_mb"] = float(_current_rss_mb())
        memory_profile["deploy_infer_events_per_sec"] = float(
            stream_outputs.get("test_count", 0),
        ) / max(float(stream_outputs.get("test_scoring_seconds", 0.0)), 1e-9)
        label_started = time.perf_counter()
        alert_event_indices = _alert_event_indices(stream_outputs)
        test_node_maps = None
        eval_node_maps = build_node_maps(cur)
        abnormal_nodes = load_ground_truth_indices_readonly(db_cfg, eval_node_maps["uuid2index"])
        event_labels_by_event = collect_event_labels(
            stream_dataset_rows(
                conn,
                year_month,
                test_days,
                eval_node_maps,
                event_filter,
                config.fetch_size,
                config.max_test_events,
                abnormal_nodes=abnormal_nodes,
            ),
            alert_event_indices,
        )
        malicious_pair_audit_path = output_dir / "malicious_abnormal_pair_residual_tokens.csv"
        malicious_pair_count = _write_malicious_pair_audit_csv(
            malicious_pair_audit_path,
            stream_dataset_rows(
                conn,
                year_month,
                test_days,
                eval_node_maps,
                event_filter,
                config.fetch_size,
                config.max_test_events,
                abnormal_nodes=abnormal_nodes,
            ),
            abnormal_nodes,
            config,
            process_cfg,
        )
        eval_node_maps = None
        gc.collect()
        stream_outputs["event_labels_by_event"] = event_labels_by_event
        stream_outputs["node_labels_by_node"] = {
            int(node_id): "malicious"
            for node_id in abnormal_nodes
        }
        stream_outputs.update(_embedding_eval_metadata(config, embedder))
        residual_semantic_examples = dict(stream_outputs.get("residual_semantic_examples", {}))
        _write_csv(
            output_dir / "residual_semantic_examples.csv",
            _flatten_residual_examples(residual_semantic_examples),
            RESIDUAL_EXAMPLE_FIELDS,
        )
        stream_outputs["residual_semantic_audit_outputs"] = {
            "residual_semantic_examples_csv": str(output_dir / "residual_semantic_examples.csv"),
            "malicious_abnormal_pair_residual_tokens_csv": str(malicious_pair_audit_path),
            "malicious_abnormal_pair_rows": int(malicious_pair_count),
        }
        evaluation_label_seconds = float(time.perf_counter() - label_started)
        timing["label_attach_seconds"] = evaluation_label_seconds
        memory_profile["rss_after_label_attach_mb"] = _current_rss_mb()
        memory_profile["state_slots"] = _safe_state_slots(model)
        memory_profile["state_array_mb"] = _safe_state_array_mb(model)
        elapsed = max(time.perf_counter() - started, 1e-9)
        test_count = int(stream_outputs.get("test_count", 0))
        eval_payload = _eval_payload(
            config,
            stream_outputs,
            model,
            elapsed,
            test_count,
            timing,
            memory_profile,
            split_metadata=split_metadata,
        )
        eval_payload.update(
            {
                "dataset": str(config.dataset),
                "out_tag": str(config.out_tag),
                "cache": cache_info,
                "train_seconds": float(timing["train_seconds"]),
                "validation_seconds": float(timing["validation_seconds"]),
                "evaluation_label_seconds": float(timing["label_attach_seconds"]),
                "train_events": int(train_artifact_count),
                "validation_events": int(validation_artifact_count),
                "test_events": int(test_count),
                "train_events_actual": int(train_artifact_count),
                "validation_events_actual": int(validation_artifact_count),
                "test_events_actual": int(test_count),
                "validation_event_score_summary": validation_event_score_summary,
            },
        )
        write_effective_config(
            output_dir,
            config,
            model,
            _embedder_loaded(config),
            train_count=int(train_artifact_count),
            validation_count=int(validation_artifact_count),
            test_count=int(test_count),
        )
        audit_skipped_reason = ""
        train_audit_available = True
        validation_audit_available = True
        if bool(cache_info.get("hit", False)):
            audit_skipped_reason = "base_cache_hit_skips_train_validation_stream"
            train_audit_available = False
            validation_audit_available = False
        _write_process_semantic_audit_outputs(
            output_dir,
            process_audit,
            train_available=train_audit_available,
            validation_available=validation_audit_available,
            test_available=True,
            skipped_reason=audit_skipped_reason,
        )
        final_eval_payload = _write_outputs(output_dir, stream_outputs, eval_payload)
        if final_eval_payload is not None:
            eval_payload = final_eval_payload
        write_metrics_json(
            output_dir,
            config,
            model,
            eval_payload,
            train_count=int(train_artifact_count),
            validation_count=int(validation_artifact_count),
            test_count=int(test_count),
            embedder_loaded=_embedder_loaded(config),
        )
        return output_dir / "eval_causal_semantics_slim.json"
    finally:
        cur.close()
        conn.close()


from legacy.compatibility.pipeline_namespace_bridge import link_runtime_modules as _link_runtime_modules

_link_runtime_modules(globals())
