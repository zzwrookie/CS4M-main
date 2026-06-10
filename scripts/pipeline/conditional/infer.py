"""Phase3G conditional inference and Phase3E load-and-infer entrypoints."""

from __future__ import annotations

from scripts.pipeline.config.runtime_config import *


def _phase3g_should_suppress_both_cold_unseen_alert(
    *,
    config: SlimConfig,
    target_case: str,
    threshold_level: str,
    validation_group_count: int,
) -> bool:
    """Return true when both_cold unseen groups should remain observation-only."""
    if str(config.conditional_both_cold_unseen_policy) != "observation_only_no_alert":
        return False
    if str(target_case) != BOTH_COLD_ACTION_TARGET:
        return False
    if str(threshold_level) != "unseen_group_extreme":
        return False
    return int(validation_group_count) <= 0


def _phase3g_apply_both_cold_unseen_alert_policy(
    *,
    config: SlimConfig,
    raw_alert: bool,
    target_case: str,
    threshold_level: str,
    validation_group_count: int,
) -> tuple[bool, int]:
    """Apply both_cold unseen policy and return final alert plus suppressed delta."""
    if not bool(raw_alert):
        return False, 0
    if _phase3g_should_suppress_both_cold_unseen_alert(
        config=config,
        target_case=target_case,
        threshold_level=threshold_level,
        validation_group_count=validation_group_count,
    ):
        return False, 1
    return True, 0


def _phase3g_conditional_score_summary_payload(
    *,
    config: SlimConfig,
    cache_meta: Mapping[str, Any],
    stream_outputs: Mapping[str, Any],
    group_summary_path: str,
    demoted_group_summary_path: Path,
    group_threshold_sweep_path: Path,
    endpoint_suppression_eval: Mapping[str, Any],
    update_gate_score_space_summary: Mapping[str, Any],
    group_alert_policy_rows: Sequence[Mapping[str, Any]],
    required_dual_head_summary_paths: Mapping[str, str],
) -> dict[str, Any]:
    """Build the Phase3G conditional score summary JSON payload."""
    test_summary = dict(stream_outputs.get("test_score_summary", {}))
    test_by_case = dict(stream_outputs.get("test_score_summary_by_target_case", {}))
    validation_summary = dict(cache_meta.get("summary", {}))
    validation_by_case = dict(cache_meta.get("summary_by_target_case", {}))
    return {
        "score_head": "conditional_action_semantic",
        "conditional_head_arch": str(config.sspm_conditional_head_arch),
        "head_arch": str(config.sspm_conditional_head_arch),
        "conditional_semantic_loss": str(config.conditional_semantic_loss),
        "event_threshold_mode": str(config.event_threshold_mode),
        "event_threshold_quantile": float(config.event_threshold_quantile),
        "conditional_group_min_count": int(config.conditional_group_min_count),
        "final_threshold": float(cache_meta.get("threshold", 0.0)),
        "thresholds_by_target_case": dict(cache_meta.get("thresholds_by_target_case", {})),
        "group_thresholds_path": str(
            Path(cache_meta["validation_conditional_scores"]).parent
            / "validation_conditional_group_thresholds.json"
        )
        if cache_meta.get("group_thresholds")
        else "",
        "conditional_group_summary_csv": group_summary_path,
        "validation": validation_summary,
        "validation_by_target_case": validation_by_case,
        "test": test_summary,
        "test_by_target_case": test_by_case,
        "test_by_target_action_type": stream_outputs.get(
            "test_score_summary_by_target_action_type",
            [],
        ),
        "test_target_case_counts": dict(stream_outputs.get("target_case_counts", {})),
        "conditional_endpoint_suppression": dict(
            stream_outputs.get("conditional_endpoint_suppression", {}),
        ),
        "action_type_alert_policy": dict(stream_outputs.get("action_type_alert_policy", {})),
        "both_cold_unseen_policy": dict(stream_outputs.get("both_cold_unseen_policy", {})),
        "update_gate_score_space": dict(update_gate_score_space_summary),
        "conditional_endpoint_suppression_eval": dict(endpoint_suppression_eval),
        "phase3f_fast_path_enabled": bool(
            stream_outputs.get("phase3f_fast_path_enabled", False),
        ),
        "phase3g_reference_update_path": bool(
            stream_outputs.get("phase3g_reference_update_path", False),
        ),
        "update_gate_enabled": bool(stream_outputs.get("update_gate_enabled", False)),
        "update_gate_q_t_min": float(stream_outputs.get("update_gate_q_t_min", 1.0)),
        "update_gate_q_t_mean": float(stream_outputs.get("update_gate_q_t_mean", 1.0)),
        "update_gate_q_t_max": float(stream_outputs.get("update_gate_q_t_max", 1.0)),
        "update_gate_q_t_count": int(stream_outputs.get("update_gate_q_t_count", 0)),
        "update_gate_q_t_std": float(stream_outputs.get("update_gate_q_t_std", 0.0)),
        "update_gate_applied_count": int(stream_outputs.get("update_gate_applied_count", 0)),
        "update_gate_skipped_count": int(stream_outputs.get("update_gate_skipped_count", 0)),
        "q_t_by_action_type_csv": str(stream_outputs.get("q_t_by_action_type_csv", "")),
        "group_alert_policy_summary_csv": str(
            stream_outputs.get("group_alert_policy_summary_csv", ""),
        ),
        "demoted_group_summary_csv": str(demoted_group_summary_path),
        "group_threshold_sweep_summary_csv": str(group_threshold_sweep_path),
        "node_coverage_by_group_csv": str(stream_outputs.get("node_coverage_by_group_csv", "")),
        "target_case_summary_csv": str(
            required_dual_head_summary_paths.get("target_case_summary_csv", ""),
        ),
        "group_alert_summary_csv": str(
            required_dual_head_summary_paths.get("group_alert_summary_csv", ""),
        ),
        "group_alert_policy_eval_rows": list(group_alert_policy_rows),
    }



def _score_phase3g_conditional_fast_stream(
    *,
    config: SlimConfig,
    head: ConditionalSemanticHead,
    profile_model: SSPMLowRankModel,
    test_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    cache_meta: Mapping[str, Any],
    endpoint_pair_counts: Mapping[tuple[int, str, int], int] | None = None,
    endpoint_src_endpoint_counts: Mapping[tuple[int, str], int] | None = None,
    endpoint_counts: Mapping[tuple[str, int], int] | None = None,
    netflow_endpoint_by_idx: Mapping[int, str] | None = None,
    endpoint_lookup: CompactEndpointSuppressionLookup | None = None,
    endpoint_cache_mb: float = 0.0,
    output_dir: Path | None,
    reference_update_path: bool = False,
) -> dict[str, Any]:
    scoring_started = time.perf_counter()
    records = _phase3e_event_index_array(test_index)
    model = profile_model
    type_eye = _phase3g_type_eye()
    global_threshold = float(cache_meta.get("threshold", 0.0))
    score_summary_obj = _StreamingScoreSummary(global_threshold)
    case_summary_obj = _TargetCaseStreamingSummary()
    group_summary_obj = _ConditionalGroupStreamingSummary()
    group_meta = dict(cache_meta.get("group_thresholds") or {})
    use_group_thresholds = str(config.event_threshold_mode) == CONDITIONAL_GROUP_THRESHOLD_MODE
    endpoint_pair_counts = dict(endpoint_pair_counts or {}) if endpoint_lookup is None else {}
    endpoint_src_endpoint_counts = (
        dict(endpoint_src_endpoint_counts or {}) if endpoint_lookup is None else {}
    )
    endpoint_counts = dict(endpoint_counts or {}) if endpoint_lookup is None else {}
    endpoint_summary: dict[tuple[int, int, int], dict[str, Any]] = {}
    endpoint_summary_mode = str(config.conditional_endpoint_suppression_summary_mode)
    endpoint_summary_enabled = bool(
        config.conditional_endpoint_aware_suppression
        and endpoint_summary_mode == "analysis"
        and not _online_minimal_enabled(config)
    )
    score_compute_seconds = 0.0
    threshold_lookup_seconds = 0.0
    state_update_seconds = 0.0
    endpoint_suppression_seconds = 0.0
    q_t_min = float("inf")
    q_t_max = 0.0
    q_t_sum = 0.0
    q_t_sq_sum = 0.0
    q_t_count = 0
    q_t_applied_count = 0
    q_t_skipped_count = 0
    q_t_by_group: dict[tuple[int, int, int], dict[str, Any]] = {}
    raw_alert_count_before_suppression = 0
    suppressed_alert_count = 0
    both_cold_unseen_suppressed_alert_count = 0
    action_policy_counts: dict[tuple[int, int, int], dict[str, Any]] = {}
    node_evidence_count = 0
    demoted_event_count = 0
    budget_capped_event_count = 0
    high_priority_event_alert_count = 0
    suppressed_event_alerts_raw: list[dict[str, Any]] = []
    both_cold_unseen_suppressed_events_raw: list[dict[str, Any]] = []
    suppressed_raw_path = ""
    both_cold_unseen_suppressed_raw_path = ""
    node_evidence_raw_path = ""
    case_thresholds = {
        EVENT_SEMANTIC_TARGET: _phase3g_conditional_threshold_for_case(
            cache_meta,
            EVENT_SEMANTIC_TARGET,
        ),
        BOTH_COLD_ACTION_TARGET: _phase3g_conditional_threshold_for_case(
            cache_meta,
            BOTH_COLD_ACTION_TARGET,
        ),
    }
    raw_paths: dict[str, Path] = {}
    event_alert_count = 0
    test_count = 0
    start_rss_mb = _current_rss_mb()
    test_phase_peak_rss_mb = start_rss_mb
    node_coverage = OnlineNodeCoverageTracker()
    target_case_counts = {
        EVENT_SEMANTIC_TARGET: 0,
        BOTH_COLD_ACTION_TARGET: 0,
    }
    _stage_log(
        config,
        "phase3g_conditional_fast_test_start"
        if not bool(reference_update_path)
        else "phase3g_conditional_reference_test_start",
        count=int(records.shape[0]),
    )
    model.reset_state()
    with ExitStack() as stack:
        event_writer = None
        profile_writer = None
        state_merge_profile_writer = None
        score_trace_writer = None
        node_evidence_writer = None
        both_cold_unseen_suppressed_writer = None
        if output_dir is not None:
            raw_paths = _raw_output_paths(Path(output_dir))
            raw_paths["events"] = Path(output_dir) / "online_event_alerts.csv"
            if int(config.max_test_events) > 0 or not _online_minimal_enabled(config):
                raw_paths["event_score_trace"] = Path(output_dir) / "online_event_score_trace.csv"
            raw_paths["suppressed_endpoint_events"] = (
                Path(output_dir) / "conditional_endpoint_suppressed_events.raw.csv"
            )
            raw_paths["both_cold_unseen_suppressed_events"] = (
                Path(output_dir) / "conditional_both_cold_unseen_suppressed_events.csv"
            )
            raw_paths["action_type_node_evidence_events"] = (
                Path(output_dir) / "action_type_node_evidence_events.csv"
            )
            event_writer = stack.enter_context(
                StreamingCsvWriter(raw_paths["events"], EVENT_RAW_FIELDS),
            )
            node_evidence_writer = stack.enter_context(
                StreamingCsvWriter(
                    raw_paths["action_type_node_evidence_events"],
                    ACTION_TYPE_POLICY_EVENT_FIELDS,
                ),
            )
            node_evidence_raw_path = str(raw_paths["action_type_node_evidence_events"])
            both_cold_unseen_suppressed_writer = stack.enter_context(
                StreamingCsvWriter(
                    raw_paths["both_cold_unseen_suppressed_events"],
                    EVENT_RAW_FIELDS
                    + ["both_cold_unseen_policy", "both_cold_unseen_suppressed"],
                ),
            )
            both_cold_unseen_suppressed_raw_path = str(
                raw_paths["both_cold_unseen_suppressed_events"],
            )
            if "event_score_trace" in raw_paths:
                score_trace_writer = stack.enter_context(
                    StreamingCsvWriter(
                        raw_paths["event_score_trace"],
                        EVENT_SCORE_TRACE_FIELDS,
                    ),
                )
            suppressed_writer = None
            if bool(config.conditional_endpoint_aware_suppression):
                suppressed_writer = stack.enter_context(
                    StreamingCsvWriter(
                        raw_paths["suppressed_endpoint_events"],
                        EVENT_RAW_FIELDS,
                    ),
                )
                suppressed_raw_path = str(raw_paths["suppressed_endpoint_events"])
            profile_writer = stack.enter_context(
                StreamingCsvWriter(raw_paths["profiling"], PROFILING_FIELDS),
            )
            if _state_merge_enabled(config):
                state_merge_profile_writer = stack.enter_context(
                    StreamingCsvWriter(
                        raw_paths["state_merge_profile"],
                        STATE_MERGE_PROFILE_FIELDS,
                    ),
                )
        else:
            suppressed_writer = None
        chunk_size = max(int(config.sspm_infer_chunk_events), 1)
        use_gate_sequential = _phase3g_update_gate_enabled(config, model)
        use_reference_sequential = bool(reference_update_path) or bool(use_gate_sequential)
        for start in range(0, int(records.shape[0]), chunk_size):
            end = min(start + chunk_size, int(records.shape[0]))
            chunk = records[start:end]
            if use_reference_sequential:
                score_iter: list[tuple[int, np.void, float, str, int, float]] = []
                for offset, row in enumerate(chunk):
                    z_state = _phase3e_target_from_index_row(
                        row,
                        node_embeddings,
                        action_embeddings,
                    )
                    context, src_type_name, dst_type_name, src_has_state, dst_has_state = (
                        _phase3g_action_context_from_model(
                            model,
                            row,
                            node_embeddings,
                            type_eye,
                        )
                    )
                    target, target_case = select_conditional_target(
                        row,
                        node_embeddings,
                        action_embeddings,
                        src_has_state=src_has_state,
                        dst_has_state=dst_has_state,
                    )
                    score_started = time.perf_counter()
                    event_score = _phase3g_conditional_event_score(
                        head,
                        context,
                        target,
                        str(config.conditional_semantic_loss),
                        target_case,
                        score_target_mode=str(config.sspm_score_target_mode),
                        row=row,
                        node_embeddings=node_embeddings,
                        action_embeddings=action_embeddings,
                    )
                    score_compute_seconds += time.perf_counter() - score_started
                    case_id = target_case_id(target_case)
                    target_case_counts[target_case] = int(
                        target_case_counts.get(target_case, 0),
                    ) + 1
                    action = str(ORTHRUS10_ACTION_NAMES[int(row["action_id"])])
                    update_started = time.perf_counter()
                    if bool(reference_update_path):
                        fields = _phase3e_fields_from_index_row(row)
                        model.update_states(
                            fields,
                            z_state,
                            residual_score=event_score if use_gate_sequential else None,
                            enable_update_gate=bool(use_gate_sequential),
                        )
                    else:
                        _phase3f_e2_none_update_from_numeric_row(
                            model,
                            row,
                            z_state,
                            action,
                            src_type_name,
                            dst_type_name,
                            residual_score=event_score,
                        )
                    state_update_seconds += time.perf_counter() - update_started
                    q_value = 1.0
                    if bool(use_gate_sequential):
                        q_value = float(getattr(model, "last_q_t", 1.0))
                        q_t_min = min(q_t_min, q_value)
                        q_t_max = max(q_t_max, q_value)
                        q_t_sum += q_value
                        q_t_sq_sum += q_value * q_value
                        q_t_count += 1
                        if q_value < 1.0:
                            q_t_applied_count += 1
                        else:
                            q_t_skipped_count += 1
                    else:
                        q_t_skipped_count += 1
                    score_iter.append((offset, row, event_score, target_case, case_id, q_value))
            else:
                contexts = np.zeros(
                    (int(chunk.shape[0]), int(head.config.input_dim)),
                    dtype=np.float32,
                )
                targets = np.zeros(
                    (int(chunk.shape[0]), int(head.config.output_dim)),
                    dtype=np.float32,
                )
                case_names: list[str] = []
                case_ids: list[int] = []
                for offset, row in enumerate(chunk):
                    z_state = _phase3e_target_from_index_row(
                        row,
                        node_embeddings,
                        action_embeddings,
                    )
                    context, src_type_name, dst_type_name, src_has_state, dst_has_state = (
                        _phase3g_action_context_from_model(
                            model,
                            row,
                            node_embeddings,
                            type_eye,
                        )
                    )
                    target, target_case = select_conditional_target(
                        row,
                        node_embeddings,
                        action_embeddings,
                        src_has_state=src_has_state,
                        dst_has_state=dst_has_state,
                    )
                    contexts[offset] = context
                    if str(config.sspm_score_target_mode) == SCORE_TARGET_NODE_PAIR:
                        src_embedding = np.asarray(
                            node_embeddings[int(row["src_node_idx"])],
                            dtype=np.float32,
                        )
                        dst_embedding = np.asarray(
                            node_embeddings[int(row["dst_node_idx"])],
                            dtype=np.float32,
                        )
                        targets[offset] = node_pair_no_action_target(
                            src_embedding,
                            dst_embedding,
                        )
                    elif str(config.sspm_score_target_mode) == SCORE_TARGET_EVENT_ACTION:
                        targets[offset] = target
                    else:
                        raise ValueError(
                            f"unsupported SSPM_SCORE_TARGET_MODE: {config.sspm_score_target_mode}",
                        )
                    case_names.append(target_case)
                    case_ids.append(target_case_id(target_case))
                    target_case_counts[target_case] = int(
                        target_case_counts.get(target_case, 0),
                    ) + 1
                    action = str(ORTHRUS10_ACTION_NAMES[int(row["action_id"])])
                    update_started = time.perf_counter()
                    _phase3f_e2_none_update_from_numeric_row(
                        model,
                        row,
                        z_state,
                        action,
                        src_type_name,
                        dst_type_name,
                        residual_score=0.0,
                    )
                    state_update_seconds += time.perf_counter() - update_started
                score_started = time.perf_counter()
                predictions = head.predict(
                    contexts,
                    target_case_ids=np.asarray(case_ids, dtype=np.int8)
                    if head.is_dual_head
                    else None,
                )
                scores = conditional_distance(
                    predictions,
                    targets,
                    str(config.conditional_semantic_loss),
                )
                score_compute_seconds += time.perf_counter() - score_started
                score_iter = [
                    (
                        offset,
                        row,
                        float(scores[offset]),
                        str(case_names[offset]),
                        int(case_ids[offset]),
                        1.0,
                    )
                    for offset, row in enumerate(chunk)
                ]
            for offset, row, event_score, target_case, case_id, q_value_for_group in score_iter:
                stream_pos = start + offset
                test_count = int(stream_pos) + 1
                group_key_for_q = (
                    int(row["action_id"]),
                    int(row["src_type_id"]),
                    int(row["dst_type_id"]),
                )
                group_q = q_t_by_group.setdefault(
                    group_key_for_q,
                    {
                        "count": 0,
                        "q_t_sum": 0.0,
                        "q_t_sq_sum": 0.0,
                        "q_t_min": float("inf"),
                        "q_t_max": 0.0,
                        "q_t_applied_count": 0,
                        "q_t_skipped_count": 0,
                    },
                )
                group_q["count"] = int(group_q["count"]) + 1
                group_q["q_t_sum"] = float(group_q["q_t_sum"]) + q_value_for_group
                group_q["q_t_sq_sum"] = float(group_q["q_t_sq_sum"]) + (
                    q_value_for_group * q_value_for_group
                )
                group_q["q_t_min"] = min(float(group_q["q_t_min"]), q_value_for_group)
                group_q["q_t_max"] = max(float(group_q["q_t_max"]), q_value_for_group)
                if q_value_for_group < 1.0:
                    group_q["q_t_applied_count"] = int(group_q["q_t_applied_count"]) + 1
                else:
                    group_q["q_t_skipped_count"] = int(group_q["q_t_skipped_count"]) + 1
                threshold_level = ""
                threshold_group_key = ""
                validation_group_count = 0
                threshold_trace = {
                    "low_support_policy": "",
                    "group_validation_max": "",
                    "parent_threshold": "",
                    "global_threshold": "",
                    "final_threshold_source": "",
                    "adaptive_margin_used": "",
                    "validation_count_bucket": "",
                }
                if use_group_thresholds:
                    threshold_started = time.perf_counter()
                    threshold_record = resolve_conditional_group_threshold(
                        group_meta,
                        int(case_id),
                        int(row["action_id"]),
                        int(row["src_type_id"]),
                        int(row["dst_type_id"]),
                    )
                    threshold_lookup_seconds += time.perf_counter() - threshold_started
                    threshold_value = float(threshold_record.get("threshold", global_threshold))
                    threshold_level = str(threshold_record.get("threshold_level", ""))
                    threshold_group_key = str(threshold_record.get("threshold_group_key", ""))
                    validation_group_count = int(
                        threshold_record.get("validation_group_count", 0),
                    )
                    threshold_trace = {
                        "low_support_policy": threshold_record.get("low_support_policy", ""),
                        "group_validation_max": threshold_record.get("group_validation_max", ""),
                        "parent_threshold": threshold_record.get("parent_threshold", ""),
                        "global_threshold": threshold_record.get("global_threshold", ""),
                        "final_threshold_source": threshold_record.get(
                            "final_threshold_source",
                            "",
                        ),
                        "adaptive_margin_used": threshold_record.get(
                            "adaptive_margin_used",
                            "",
                        ),
                        "validation_count_bucket": threshold_record.get(
                            "validation_count_bucket",
                            "",
                        ),
                    }
                else:
                    threshold_value = float(case_thresholds.get(target_case, global_threshold))
                    threshold_level = "target_case" if target_case in case_thresholds else "global"
                    threshold_group_key = target_case
                    validation_group_count = int(
                        dict(cache_meta.get("thresholds_by_target_case", {}))
                        .get(target_case, {})
                        .get("count", 0),
                    )
                score_summary_obj.observe(event_score)
                case_summary_obj.observe(target_case, event_score)
                if score_trace_writer is not None:
                    score_trace_writer.write_row(
                        {
                            "stream_pos": int(stream_pos),
                            "event_id": int(row["event_id"]),
                            "src_idx": int(row["src_node_idx"]),
                            "dst_idx": int(row["dst_node_idx"]),
                            "info_src": int(row["src_node_idx"]),
                            "info_dst": int(row["dst_node_idx"]),
                            "score": float(event_score),
                            "threshold": float(threshold_value),
                            "action": str(
                                ORTHRUS10_ACTION_NAMES[int(row["action_id"])],
                            ),
                            "src_type": _phase3e_entity_type_name(
                                "src_type_id",
                                int(row["src_type_id"]),
                            ),
                            "dst_type": _phase3e_entity_type_name(
                                "dst_type_id",
                                int(row["dst_type_id"]),
                            ),
                            "target_case": target_case,
                            "threshold_level": threshold_level,
                            "threshold_group_key": threshold_group_key,
                        },
                    )
                alert = event_score >= threshold_value
                raw_alert = bool(alert)
                suppressed_by_endpoint = False
                endpoint_validation_pair_count = 0
                endpoint_src_endpoint_count = 0
                endpoint_validation_count = 0
                endpoint_match_level = "unknown"
                alert_row: dict[str, Any] | None = None
                if raw_alert:
                    raw_alert_count_before_suppression += 1
                    alert_row = _phase3f_raw_alert_row(
                        row,
                        stream_pos,
                        event_score,
                        threshold_value,
                        event_score,
                        event_score,
                        f"validation_{config.event_threshold_mode}_{target_case}",
                    )
                    alert_row["target_case"] = target_case
                    alert_row["target_case_threshold"] = threshold_value
                    alert_row["threshold_level"] = threshold_level
                    alert_row["threshold_group_key"] = threshold_group_key
                    alert_row["validation_group_count"] = validation_group_count
                    alert_row.update(threshold_trace)
                    alert, both_cold_delta = _phase3g_apply_both_cold_unseen_alert_policy(
                        config=config,
                        raw_alert=raw_alert,
                        target_case=target_case,
                        threshold_level=threshold_level,
                        validation_group_count=validation_group_count,
                    )
                    alert_row["both_cold_unseen_policy"] = str(
                        config.conditional_both_cold_unseen_policy,
                    )
                    alert_row["both_cold_unseen_suppressed"] = bool(
                        both_cold_delta,
                    )
                    if both_cold_delta:
                        both_cold_unseen_suppressed_alert_count += int(both_cold_delta)
                        suppressed_both_cold_row = {
                            **alert_row,
                            "both_cold_unseen_policy": str(
                                config.conditional_both_cold_unseen_policy,
                            ),
                            "both_cold_unseen_suppressed": True,
                        }
                        if both_cold_unseen_suppressed_writer is not None:
                            both_cold_unseen_suppressed_writer.write_row(
                                suppressed_both_cold_row,
                            )
                        elif not _online_minimal_enabled(config):
                            both_cold_unseen_suppressed_events_raw.append(
                                suppressed_both_cold_row,
                            )
                    if alert:
                        suppression_started = time.perf_counter()
                        suppression = _conditional_endpoint_suppression_decision(
                            config=config,
                            row=row,
                            event_score=event_score,
                            threshold=threshold_value,
                            pair_counts=endpoint_pair_counts,
                            src_endpoint_counts=endpoint_src_endpoint_counts,
                            endpoint_counts=endpoint_counts,
                            netflow_endpoint_by_idx=netflow_endpoint_by_idx,
                            endpoint_lookup=endpoint_lookup,
                        )
                        endpoint_suppression_seconds += time.perf_counter() - (
                            suppression_started
                        )
                        endpoint_validation_pair_count = int(
                            suppression.get("validation_pair_count", 0),
                        )
                        endpoint_validation_count = int(
                            suppression.get("validation_endpoint_count", 0),
                        )
                        endpoint_src_endpoint_count = int(
                            suppression.get("validation_src_endpoint_count", 0),
                        )
                        endpoint_match_level = str(suppression.get("match_level", "unknown"))
                        alert_row["endpoint_signature"] = str(
                            suppression.get("endpoint_signature", ""),
                        )
                        alert_row["endpoint_pair_key"] = str(suppression.get("pair_key", ""))
                        alert_row["src_endpoint_key"] = str(
                            suppression.get("src_endpoint_key", ""),
                        )
                        alert_row["endpoint_action_key"] = str(
                            suppression.get("endpoint_action_key", ""),
                        )
                        alert_row["endpoint_validation_pair_count"] = (
                            endpoint_validation_pair_count
                        )
                        alert_row["src_endpoint_count"] = endpoint_src_endpoint_count
                        alert_row["endpoint_validation_count"] = endpoint_validation_count
                        alert_row["endpoint_suppression_mode"] = str(
                            suppression.get("suppression_mode", ""),
                        )
                        alert_row["endpoint_suppression_match_level"] = endpoint_match_level
                        alert_row["endpoint_suppression_reason"] = str(
                            suppression.get("suppression_reason", ""),
                        )
                        if bool(suppression.get("suppressed", False)):
                            alert = False
                            suppressed_by_endpoint = True
                            suppressed_alert_count += 1
                            suppressed_row = dict(
                                alert_row,
                                endpoint_suppression_reason=str(
                                    suppression.get("suppression_reason", ""),
                                ),
                                endpoint_validation_pair_count=endpoint_validation_pair_count,
                                src_endpoint_count=endpoint_src_endpoint_count,
                                endpoint_validation_count=endpoint_validation_count,
                                endpoint_suppression_mode=str(
                                    suppression.get("suppression_mode", ""),
                                ),
                                endpoint_suppression_match_level=endpoint_match_level,
                            )
                            if suppressed_writer is not None:
                                suppressed_writer.write_row(suppressed_row)
                            elif not _online_minimal_enabled(config):
                                suppressed_event_alerts_raw.append(suppressed_row)
                policy_decision = _action_type_alert_policy_decision(
                    config=config,
                    row=row,
                    raw_alert=bool(alert),
                )
                group_key_for_policy = (
                    int(row["action_id"]),
                    int(row["src_type_id"]),
                    int(row["dst_type_id"]),
                )
                action_policy = action_policy_counts.setdefault(
                    group_key_for_policy,
                    {
                        "event_count": 0,
                        "alert_count": 0,
                        "node_evidence_count": 0,
                        "demoted_event_count": 0,
                        "budget_capped_event_count": 0,
                        "high_priority_event_alert_count": 0,
                        "threshold_sum": 0.0,
                        "threshold_count": 0,
                    },
                )
                action_policy["event_count"] = int(action_policy["event_count"]) + 1
                action_policy["threshold_sum"] = float(action_policy["threshold_sum"]) + float(
                    threshold_value,
                )
                action_policy["threshold_count"] = int(action_policy["threshold_count"]) + 1
                if bool(policy_decision["node_evidence"]):
                    node_evidence_count += 1
                    action_policy["node_evidence_count"] = (
                        int(action_policy["node_evidence_count"]) + 1
                    )
                    if alert_row is not None and node_evidence_writer is not None:
                        node_evidence_writer.write_row(
                            {
                                **alert_row,
                                "alert_policy": policy_decision["policy_name"],
                                "alert_decision": policy_decision["alert_decision"],
                                "alert_priority": policy_decision["alert_priority"],
                                "node_evidence": policy_decision["node_evidence"],
                                "budget_capped": policy_decision["budget_capped"],
                            },
                        )
                if str(policy_decision["alert_decision"]) == "demoted_event":
                    demoted_event_count += 1
                    action_policy["demoted_event_count"] = (
                        int(action_policy["demoted_event_count"]) + 1
                    )
                if bool(policy_decision["budget_capped"]):
                    budget_capped_event_count += 1
                    action_policy["budget_capped_event_count"] = (
                        int(action_policy["budget_capped_event_count"]) + 1
                    )
                alert = bool(policy_decision["final_alert"])
                if bool(config.conditional_endpoint_aware_suppression):
                    if not raw_alert and endpoint_lookup is not None:
                        endpoint_validation_pair_count = endpoint_lookup.pair_count(
                            int(row["src_node_idx"]),
                            int(row["dst_node_idx"]),
                            int(row["action_id"]),
                        )
                        endpoint_src_endpoint_count = endpoint_lookup.src_endpoint_count(
                            int(row["src_node_idx"]),
                            int(row["dst_node_idx"]),
                        )
                        endpoint_validation_count = endpoint_lookup.endpoint_count(
                            int(row["dst_node_idx"]),
                            int(row["action_id"]),
                        )
                        endpoint_match_level = (
                            "pair"
                            if endpoint_validation_pair_count
                            >= int(config.conditional_known_pair_min_count)
                            else (
                                "same_process_endpoint_history"
                                if endpoint_src_endpoint_count
                                >= int(config.conditional_same_process_endpoint_min_count)
                                else (
                                    "endpoint"
                                    if endpoint_validation_count
                                    >= int(config.conditional_known_pair_min_count)
                                    else "unknown"
                                )
                            )
                        )
                    elif not raw_alert:
                        pair_key = _conditional_endpoint_pair_signature(
                            row,
                            netflow_endpoint_by_idx,
                        )
                        endpoint_key = _conditional_endpoint_action_signature(
                            row,
                            netflow_endpoint_by_idx,
                        )
                        src_endpoint_key = _conditional_src_endpoint_signature(
                            row,
                            netflow_endpoint_by_idx,
                        )
                        endpoint_validation_pair_count = int(
                            endpoint_pair_counts.get(pair_key, 0),
                        )
                        endpoint_src_endpoint_count = int(
                            endpoint_src_endpoint_counts.get(src_endpoint_key, 0),
                        )
                        endpoint_validation_count = int(endpoint_counts.get(endpoint_key, 0))
                        endpoint_match_level = (
                            "pair"
                            if endpoint_validation_pair_count
                            >= int(config.conditional_known_pair_min_count)
                            else (
                                "same_process_endpoint_history"
                                if endpoint_src_endpoint_count
                                >= int(config.conditional_same_process_endpoint_min_count)
                                else (
                                    "endpoint"
                                    if endpoint_validation_count
                                    >= int(config.conditional_known_pair_min_count)
                                    else "unknown"
                                )
                            )
                        )
                    if endpoint_summary_enabled:
                        _conditional_endpoint_summary_update(
                            endpoint_summary,
                            row,
                            include_read=bool(config.conditional_endpoint_suppression_read),
                            netflow_endpoint_by_idx=netflow_endpoint_by_idx,
                            validation_pair_count=endpoint_validation_pair_count,
                            validation_endpoint_count=endpoint_validation_count,
                            match_level=endpoint_match_level,
                            raw_alert=raw_alert,
                            suppressed=suppressed_by_endpoint,
                            event_score=event_score,
                            threshold=threshold_value,
                        )
                group_summary_obj.observe(
                    int(case_id),
                    int(row["action_id"]),
                    int(row["src_type_id"]),
                    int(row["dst_type_id"]),
                    event_score,
                    alert=alert,
                )
                if alert:
                    action_policy["alert_count"] = int(action_policy["alert_count"]) + 1
                    if str(policy_decision["alert_priority"]) == "high":
                        high_priority_event_alert_count += 1
                        action_policy["high_priority_event_alert_count"] = (
                            int(action_policy["high_priority_event_alert_count"]) + 1
                        )
                    event_alert_count += 1
                    if alert_row is None:
                        alert_row = _phase3f_raw_alert_row(
                            row,
                            stream_pos,
                            event_score,
                            threshold_value,
                            event_score,
                            event_score,
                            f"validation_{config.event_threshold_mode}_{target_case}",
                        )
                        alert_row["target_case"] = target_case
                        alert_row["target_case_threshold"] = threshold_value
                        alert_row["threshold_level"] = threshold_level
                        alert_row["threshold_group_key"] = threshold_group_key
                        alert_row["validation_group_count"] = validation_group_count
                        alert_row.update(threshold_trace)
                    if event_writer is not None:
                        event_writer.write_row(alert_row)
                    src_idx = int(row["src_node_idx"])
                    dst_idx = int(row["dst_node_idx"])
                    event_id = int(row["event_id"])
                    node_coverage.observe(
                        node_idx=src_idx,
                        node_type=str(alert_row.get("src_type", "")),
                        event_id=event_id,
                        event_score=event_score,
                        side="src",
                    )
                    node_coverage.observe(
                        node_idx=dst_idx,
                        node_type=str(alert_row.get("dst_type", "")),
                        event_id=event_id,
                        event_score=event_score,
                        side="dst",
                    )
            if (
                int(config.progress_interval_events) > 0
                and test_count > 0
                and test_count % int(config.progress_interval_events) == 0
            ):
                current_rss_mb = _current_rss_mb()
                test_phase_peak_rss_mb = max(test_phase_peak_rss_mb, current_rss_mb)
                _stage_log(
                    config,
                    "phase3g_conditional_fast_test_progress",
                    count=test_count,
                    event_alerts=event_alert_count,
                    node_coverage=len(node_coverage),
                )
                _emit_profile_row(
                    profile_writer,
                    _profiling_row(
                        "test_scoring",
                        test_count,
                        scoring_started,
                        event_alert_count,
                        len(node_coverage),
                        profile_model,
                        current_rss_mb=current_rss_mb,
                        test_phase_peak_rss_mb=test_phase_peak_rss_mb,
                    ),
                    verbose=config.verbose,
                )
                if state_merge_profile_writer is not None:
                    state_merge_profile_writer.write_row(
                        _state_merge_profile_row(model, test_count, scoring_started),
                    )
        final_rss_mb = _current_rss_mb()
        test_phase_peak_rss_mb = max(test_phase_peak_rss_mb, final_rss_mb)
        if state_merge_profile_writer is not None:
            state_merge_profile_writer.write_row(
                _state_merge_profile_row(model, test_count, scoring_started),
            )
    test_scoring_seconds = float(time.perf_counter() - scoring_started)
    smaps = _current_smaps_rollup_mb()
    test_summary = score_summary_obj.summary(event_alert_count=event_alert_count)
    test_summary["test_above_threshold_count"] = int(raw_alert_count_before_suppression)
    group_summary_rows = group_summary_obj.rows(cache_meta)
    q_t_group_rows = _phase3g_q_t_group_rows(config, q_t_by_group)
    action_policy_rows = _phase3g_action_policy_group_rows(
        config,
        action_policy_counts,
        q_t_group_rows,
    )
    group_summary_path = ""
    q_t_by_action_type_path = ""
    group_alert_policy_summary_path = ""
    node_coverage_by_group_path = ""
    if output_dir is not None:
        group_summary_path = str(
            _write_conditional_group_test_summary(Path(output_dir), group_summary_rows),
        )
        q_t_by_action_type_path = str(Path(output_dir) / "q_t_by_action_type.csv")
        group_alert_policy_summary_path = str(
            Path(output_dir) / "group_alert_policy_summary.csv",
        )
        node_coverage_by_group_path = str(Path(output_dir) / "node_coverage_by_group.csv")
        _write_csv(Path(q_t_by_action_type_path), q_t_group_rows, Q_T_BY_ACTION_TYPE_FIELDS)
        _write_csv(
            Path(group_alert_policy_summary_path),
            action_policy_rows,
            ACTION_TYPE_POLICY_SUMMARY_FIELDS,
        )
        _write_csv(
            Path(node_coverage_by_group_path),
            action_policy_rows,
            NODE_COVERAGE_BY_GROUP_FIELDS,
        )
        if bool(config.conditional_endpoint_aware_suppression) and endpoint_summary_enabled:
            endpoint_summary_path = str(
                _write_conditional_endpoint_suppression_summary(
                    Path(output_dir),
                    endpoint_summary,
                ),
            )
        else:
            endpoint_summary_path = ""
    else:
        endpoint_summary_path = ""
    embedding_metrics = (
        node_embeddings.metrics() if hasattr(node_embeddings, "metrics") else {}
    )
    runtime_timing = {
        "score_compute_seconds": float(score_compute_seconds),
        "threshold_lookup_seconds": float(threshold_lookup_seconds),
        "state_update_seconds": float(state_update_seconds),
        "endpoint_suppression_seconds": float(endpoint_suppression_seconds),
    }
    ofsm_runtime_profile: dict[str, Any] = {}
    ofsm_runtime_paths: dict[str, str] = {}
    if _state_merge_enabled(config):
        ofsm_runtime_profile = _ofsm_runtime_profile_payload(
            config=config,
            model=model,
            events=int(test_count),
            stage="test",
            started_at=scoring_started,
            test_seconds=test_scoring_seconds,
            embedding_metrics=embedding_metrics,
            smaps=smaps,
            timing=runtime_timing,
        )
        if output_dir is not None:
            ofsm_runtime_paths = _write_ofsm_runtime_profile(
                Path(output_dir),
                ofsm_runtime_profile,
            )
    q_t_mean = float(q_t_sum / max(q_t_count, 1)) if q_t_count else 1.0
    q_t_min_value = 1.0 if q_t_count == 0 else float(q_t_min)
    q_t_max_value = 1.0 if q_t_count == 0 else float(q_t_max)
    q_t_variance = (
        max(float(q_t_sq_sum / max(q_t_count, 1)) - q_t_mean * q_t_mean, 0.0)
        if q_t_count
        else 0.0
    )
    q_t_std = float(math.sqrt(q_t_variance))
    fast_path_enabled = bool(config.sspm_infer_fast_path) and not bool(reference_update_path)
    return {
        "test_count": int(test_count),
        "final_nodes_raw": [],
        "event_node_coverage_rows": node_coverage.rows(),
        "threshold": global_threshold,
        "thresholds_by_target_case": case_thresholds,
        "target_case_counts": target_case_counts,
        "test_score_summary": test_summary,
        "test_score_summary_by_target_case": case_summary_obj.summary(case_thresholds),
        "test_score_summary_by_target_action_type": group_summary_rows,
        "conditional_group_summary_csv": group_summary_path,
        "raw_outputs_streamed": output_dir is not None,
        "output_dir": str(output_dir) if output_dir is not None else "",
        "raw_output_paths": {key: str(path) for key, path in raw_paths.items()},
        "event_alert_count": int(event_alert_count),
        "raw_alert_count_before_suppression": int(raw_alert_count_before_suppression),
        "suppressed_alert_count": int(suppressed_alert_count),
        "both_cold_unseen_suppressed_alert_count": int(
            both_cold_unseen_suppressed_alert_count,
        ),
        "node_evidence_count": int(node_evidence_count),
        "demoted_event_count": int(demoted_event_count),
        "budget_capped_event_count": int(budget_capped_event_count),
        "high_priority_event_alert_count": int(high_priority_event_alert_count),
        "action_type_alert_policy": {
            "policy_name": str(config.action_type_alert_policy),
            "node_evidence_count": int(node_evidence_count),
            "demoted_event_count": int(demoted_event_count),
            "budget_capped_event_count": int(budget_capped_event_count),
            "high_priority_event_alert_count": int(high_priority_event_alert_count),
            "node_evidence_events_csv": node_evidence_raw_path,
            "group_alert_policy_summary_csv": group_alert_policy_summary_path,
        },
        "both_cold_unseen_policy": {
            "policy_name": str(config.conditional_both_cold_unseen_policy),
            "suppressed_alert_count": int(both_cold_unseen_suppressed_alert_count),
        },
        "both_cold_unseen_suppressed_events_raw": both_cold_unseen_suppressed_events_raw,
        "both_cold_unseen_suppressed_events_csv": both_cold_unseen_suppressed_raw_path,
        "suppressed_event_alerts_raw": suppressed_event_alerts_raw,
        "suppressed_event_alerts_raw_csv": suppressed_raw_path,
        "q_t_by_action_type_csv": q_t_by_action_type_path,
        "group_alert_policy_summary_csv": group_alert_policy_summary_path,
        "node_coverage_by_group_csv": node_coverage_by_group_path,
        "conditional_endpoint_suppression_summary_csv": endpoint_summary_path,
        "conditional_endpoint_suppression": {
            "enabled": bool(config.conditional_endpoint_aware_suppression),
            "suppression_mode": str(config.conditional_endpoint_suppression_mode),
            "known_pair_min_count": int(config.conditional_known_pair_min_count),
            "margin": float(config.conditional_pair_suppression_margin),
            "same_process_endpoint_min_count": int(
                config.conditional_same_process_endpoint_min_count,
            ),
            "same_process_endpoint_margin": float(
                config.conditional_same_process_endpoint_margin,
            ),
            "raw_alert_count_before_suppression": int(raw_alert_count_before_suppression),
            "suppressed_alert_count": int(suppressed_alert_count),
            "final_alert_count": int(event_alert_count),
            "summary_mode": endpoint_summary_mode,
        },
        "checkpoint_count": 0,
        "residual_semantic_examples": {group: [] for group in RESIDUAL_EXAMPLE_GROUPS},
        "test_scoring_seconds": test_scoring_seconds,
        "stream_csv_write_seconds": 0.0,
        "threshold_controller": {
            "mode": str(config.event_threshold_mode),
            "initial_threshold": global_threshold,
            "final_threshold": global_threshold,
            "score_target_mode": str(config.sspm_score_target_mode),
        },
        "compat_in_memory_outputs_active": False,
        "raw_alerts_written": bool(output_dir is not None),
        "analysis_outputs_written": False,
        "state_merge_diagnostics_written": False,
        "state_merge_profile_written": bool(_state_merge_enabled(config)),
        "rss_test_peak_mb": float(test_phase_peak_rss_mb),
        "process_peak_rss_mb": _safe_peak_rss_mb(),
        "ofsm_compression": _ofsm_compression_summary(
            model,
            memory={
                "rss_test_peak_mb": float(test_phase_peak_rss_mb),
                "rss_after_label_attach_mb": float(final_rss_mb),
                "deploy_rss_valid": True,
                "deploy_infer_rss_peak_mb": float(test_phase_peak_rss_mb),
                "deploy_infer_rss_final_mb": float(final_rss_mb),
                "deploy_infer_events_per_sec": float(
                    test_count / max(test_scoring_seconds, 1e-9),
                ),
            },
            events_per_sec=float(test_count / max(test_scoring_seconds, 1e-9)),
        ),
        "state_merge": dict(model.state_merge_profile_stats()),
        "ofsm_runtime_profile": ofsm_runtime_profile,
        "ofsm_runtime_profile_csv": ofsm_runtime_paths.get("ofsm_runtime_profile_csv", ""),
        "ofsm_runtime_profile_json": ofsm_runtime_paths.get("ofsm_runtime_profile_json", ""),
        "state_merge_diagnostics_truncated": False,
        "update_gate_enabled": bool(use_gate_sequential),
        "update_gate_q_t_min": q_t_min_value,
        "update_gate_q_t_mean": q_t_mean,
        "update_gate_q_t_max": q_t_max_value,
        "update_gate_q_t_count": int(q_t_count),
        "update_gate_q_t_std": q_t_std,
        "update_gate_applied_count": int(q_t_applied_count),
        "update_gate_skipped_count": int(q_t_skipped_count),
        "score_compute_seconds": float(score_compute_seconds),
        "threshold_lookup_seconds": float(threshold_lookup_seconds),
        "state_update_seconds": float(state_update_seconds),
        "endpoint_suppression_seconds": float(endpoint_suppression_seconds),
        "phase3g_conditional_action_semantic_enabled": True,
        "phase3f_fast_path_enabled": fast_path_enabled,
        "phase3g_reference_update_path": bool(reference_update_path),
        "online_minimal": True,
        "online_minimal_rss_start_mb": float(start_rss_mb),
        "online_minimal_rss_peak_mb": float(test_phase_peak_rss_mb),
        "online_minimal_rss_final_mb": float(final_rss_mb),
        "online_minimal_rss_delta_mb": float(test_phase_peak_rss_mb - start_rss_mb),
        "online_minimal_events_per_sec": float(
            test_count / max(test_scoring_seconds, 1e-9),
        ),
        "online_minimal_state_array_mb": float(_safe_state_array_mb(model)),
        "online_minimal_alert_buffer_mb": 0.0,
        "online_minimal_endpoint_cache_mb": float(endpoint_cache_mb),
        "online_minimal_event_node_coverage_writer_mb": float(node_coverage.estimated_mb()),
        "online_minimal_suppression_summary_mb": 0.0
        if not endpoint_summary_enabled
        else float(sys.getsizeof(endpoint_summary) / 1024.0 / 1024.0),
        "online_minimal_anonymous_rss_mb": smaps.get("anonymous_rss_mb"),
        "online_minimal_file_backed_rss_mb": smaps.get("file_backed_rss_mb"),
        **embedding_metrics,
    }



def _phase3e_event_labels_for_alerts(
    *,
    records: np.ndarray,
    alert_event_indices: set[int],
    abnormal_db_node_ids: set[int],
    idx_to_db_node_id: Mapping[int, int],
) -> dict[int, Any]:
    """Attach event labels after streaming using DB-node ground truth only."""
    labels: dict[int, Any] = {}
    needed = set(int(value) for value in alert_event_indices)
    if not needed:
        return labels
    for row in _phase3e_event_index_array(records):
        event_id = int(row["event_id"])
        if event_id not in needed:
            continue
        src_db = int(idx_to_db_node_id.get(int(row["src_node_idx"]), -1))
        dst_db = int(idx_to_db_node_id.get(int(row["dst_node_idx"]), -1))
        src_bad = src_db in abnormal_db_node_ids
        dst_bad = dst_db in abnormal_db_node_ids
        labels[event_id] = 2 if src_bad and dst_bad else (1 if src_bad or dst_bad else 0)
        if len(labels) >= len(needed):
            break
    return labels


def _phase3e_node_labels_for_alert_pool(
    final_nodes_raw: Sequence[Mapping[str, Any]],
    abnormal_db_node_ids: set[int],
    idx_to_db_node_id: Mapping[int, int],
) -> dict[int, str]:
    """Attach compact node labels after streaming from DB-node ground truth."""
    labels: dict[int, str] = {}
    for row in final_nodes_raw:
        node_idx = int(row.get("node_id", -1))
        db_node_id = int(idx_to_db_node_id.get(node_idx, -1))
        if db_node_id in abnormal_db_node_ids:
            labels[node_idx] = "malicious"
    return labels


def run_phase3g_conditional_load_and_infer_from_precompute(config: SlimConfig) -> Path:
    """Run Phase3G conditional action-semantic inference from Phase3E artifacts."""
    _validate_active_event_score_mode(config.event_score_mode)
    if str(config.sspm_train_mode) != "load_and_infer":
        raise ValueError("conditional_action_semantic infer requires SSPM_TRAIN_MODE=load_and_infer")
    checkpoint_path = _phase3g_resolved_head_checkpoint_path(config)
    started = time.perf_counter()
    output_dir = Path(config.result_root) / config.out_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, event_meta = _phase3g_effective_artifacts(config)
    validation_index, validation_count = _phase3e_open_split_event_index(
        paths,
        event_meta,
        "validation",
        max_events=int(config.max_ref_events),
    )
    test_index, test_count = _phase3e_open_split_event_index(
        paths,
        event_meta,
        "test",
        max_events=int(config.max_test_events),
    )
    node_embeddings = _phase3g_open_node_embeddings(config, paths)
    action_embeddings = np.load(paths["action_embeddings"], mmap_mode="r")
    head, head_metadata = ConditionalSemanticHead.load(
        checkpoint_path,
        expected_head_arch=str(config.sspm_conditional_head_arch),
    )
    if int(head.config.input_dim) != _phase3g_action_input_dim(node_embeddings):
        raise ValueError("conditional head input_dim does not match Phase3E embeddings")
    state_model, checkpoint_payload = load_sspm_checkpoint_state_for_conditional(
        config.sspm_checkpoint_path,
        config,
    )
    runtime_config_summary = _phase3e_apply_load_and_infer_runtime_config(
        state_model,
        config,
        checkpoint_payload,
    )
    endpoint_pair_counts: Counter[tuple[int, str, int]] = Counter()
    endpoint_src_endpoint_counts: Counter[tuple[int, str]] = Counter()
    endpoint_counts: Counter[tuple[str, int]] = Counter()
    netflow_endpoint_by_idx: dict[int, str] = {}
    endpoint_lookup: CompactEndpointSuppressionLookup | None = None
    endpoint_cache_mb = 0.0
    endpoint_cache_meta_for_rss: dict[str, Any] = {}
    rss_timeline: list[dict[str, Any]] = [_phase3g_smaps_timeline_row("after_model_load")]
    endpoint_pair_count_summary: dict[str, Any] = {}
    endpoint_fingerprint: dict[str, Any] | None = None
    if bool(config.conditional_endpoint_aware_suppression):
        train_index_for_pairs, _ = _phase3e_open_split_event_index(
            paths,
            event_meta,
            "train",
            max_events=int(config.max_train_events),
        )
        endpoint_fingerprint = _conditional_endpoint_cache_fingerprint(
            config=config,
            paths=paths,
            event_meta=event_meta,
            train_index=train_index_for_pairs,
            validation_index=validation_index,
        )
        endpoint_cache_dir = _conditional_endpoint_cache_dir(config, endpoint_fingerprint)
        endpoint_cache_meta: dict[str, Any] = {}
        endpoint_cache_status = "loaded"
        try:
            endpoint_cache = _load_conditional_endpoint_suppression_cache(
                endpoint_cache_dir,
                expected_fingerprint=endpoint_fingerprint,
            )
            endpoint_lookup = endpoint_cache.get("endpoint_lookup")
            endpoint_pair_counts = endpoint_cache.get("pair_counts", Counter())
            endpoint_src_endpoint_counts = endpoint_cache.get("src_endpoint_counts", Counter())
            endpoint_counts = endpoint_cache.get("endpoint_counts", Counter())
            netflow_endpoint_by_idx = endpoint_cache.get("netflow_endpoint_by_idx", {})
            endpoint_cache_meta = dict(endpoint_cache.get("meta", {}))
            endpoint_cache_meta_for_rss = dict(endpoint_cache_meta)
            endpoint_cache_mb = float(endpoint_cache_meta.get("endpoint_cache_mb", 0.0) or 0.0)
            _stage_log(
                config,
                "phase3g_endpoint_suppression_cache_load",
                path=str(endpoint_cache_dir),
                pairs=int(endpoint_cache_meta.get("known_pair_count", 0)),
                endpoints=int(endpoint_cache_meta.get("known_endpoint_count", 0)),
                compact=bool(endpoint_cache_meta.get("compact_cache", False)),
            )
        except FileNotFoundError:
            endpoint_cache_status = "built"
            _stage_log(
                config,
                "phase3g_endpoint_suppression_cache_build_start",
                path=str(endpoint_cache_dir),
            )
            netflow_endpoint_by_idx = _phase3g_load_netflow_endpoint_by_idx(config, paths)
            endpoint_pair_counts = _conditional_endpoint_pair_counts(
                [train_index_for_pairs, validation_index],
                netflow_endpoint_by_idx=netflow_endpoint_by_idx,
                include_read=bool(config.conditional_endpoint_suppression_read),
            )
            endpoint_counts = _conditional_endpoint_action_counts(
                [train_index_for_pairs, validation_index],
                netflow_endpoint_by_idx=netflow_endpoint_by_idx,
                include_read=bool(config.conditional_endpoint_suppression_read),
            )
            endpoint_src_endpoint_counts = _conditional_src_endpoint_counts(
                [train_index_for_pairs, validation_index],
                netflow_endpoint_by_idx=netflow_endpoint_by_idx,
                include_read=bool(config.conditional_endpoint_suppression_read),
            )
            endpoint_cache_meta = _write_conditional_endpoint_suppression_cache(
                endpoint_cache_dir,
                fingerprint=endpoint_fingerprint,
                pair_counts=endpoint_pair_counts,
                endpoint_counts=endpoint_counts,
                src_endpoint_counts=endpoint_src_endpoint_counts,
                netflow_endpoint_by_idx=netflow_endpoint_by_idx,
                summary={
                    "pair_count_source": "train_plus_validation_event_index",
                    "endpoint_signature_source": "netflow_dst_addr_port_or_node_idx",
                },
            )
            endpoint_lookup = _load_conditional_endpoint_suppression_cache(
                endpoint_cache_dir,
                expected_fingerprint=endpoint_fingerprint,
            )["endpoint_lookup"]
            endpoint_cache_meta_for_rss = dict(endpoint_cache_meta)
            endpoint_cache_mb = float(endpoint_cache_meta.get("endpoint_cache_mb", 0.0) or 0.0)
            endpoint_pair_counts = Counter()
            endpoint_src_endpoint_counts = Counter()
            endpoint_counts = Counter()
            netflow_endpoint_by_idx = {}
            _stage_log(
                config,
                "phase3g_endpoint_suppression_cache_build_end",
                path=str(endpoint_cache_dir),
                pairs=int(endpoint_cache_meta.get("known_pair_count", 0)),
                endpoints=int(endpoint_cache_meta.get("known_endpoint_count", 0)),
            )
        endpoint_pair_count_summary = {
            "enabled": True,
            "cache_status": endpoint_cache_status,
            "cache_dir": str(endpoint_cache_dir),
            "cache_meta_path": str(endpoint_cache_dir / "endpoint_suppression_meta.json"),
            "cache_meta_alias_path": str(endpoint_cache_dir / "endpoint_cache_meta.json"),
            "cache_counts_path": str(endpoint_cache_dir / "endpoint_pair_keys.npy"),
            "cache_summary_path": str(endpoint_cache_dir / "endpoint_suppression_summary.csv"),
            "cache_fingerprint_sha256": stable_json_hash(dict(endpoint_fingerprint)),
            "cache_schema": str(endpoint_cache_meta.get("schema", "")),
            "compact_cache": bool(endpoint_cache_meta.get("compact_cache", False)),
            "endpoint_cache_mb": float(endpoint_cache_mb),
            "pair_count_source": "train_plus_validation_event_index",
            "suppression_mode": str(config.conditional_endpoint_suppression_mode),
            "known_pair_min_count": int(config.conditional_known_pair_min_count),
            "margin": float(config.conditional_pair_suppression_margin),
            "same_process_endpoint_min_count": int(
                config.conditional_same_process_endpoint_min_count,
            ),
            "same_process_endpoint_margin": float(
                config.conditional_same_process_endpoint_margin,
            ),
            "num_known_candidate_pairs": int(endpoint_cache_meta.get("known_pair_count", 0)),
            "num_known_src_endpoints": int(
                endpoint_cache_meta.get("known_src_endpoint_count", 0),
            ),
            "num_known_endpoints": int(endpoint_cache_meta.get("known_endpoint_count", 0)),
            "endpoint_signature_source": "netflow_dst_addr_port_or_node_idx",
            "netflow_endpoint_map_size": int(len(netflow_endpoint_by_idx)),
            "known_pair_count": int(endpoint_cache_meta.get("known_pair_count", 0)),
            "known_endpoint_count": int(endpoint_cache_meta.get("known_endpoint_count", 0)),
        }
    rss_timeline.append(_phase3g_smaps_timeline_row("after_endpoint_cache_load"))
    validation_started = time.perf_counter()
    rss_timeline.append(_phase3g_smaps_timeline_row("before_validation"))
    cache_meta, validation_scores, validation_cases = (
        _phase3g_load_or_build_conditional_validation_cache(
            config=config,
            head=head,
            head_metadata=head_metadata,
            paths=paths,
            event_meta=event_meta,
            validation_index=validation_index,
            node_embeddings=node_embeddings,
            action_embeddings=action_embeddings,
            state_model=state_model,
            endpoint_suppression_cache_fingerprint=endpoint_fingerprint
            if bool(config.conditional_endpoint_aware_suppression)
            else None,
        )
    )
    timing = {
        "validation_seconds": float(time.perf_counter() - validation_started),
    }
    update_gate_score_space_summary = _phase3g_apply_conditional_update_gate_score_space(
        config,
        state_model,
        validation_scores,
    )
    rss_timeline.append(_phase3g_smaps_timeline_row("after_validation"))
    rss_timeline.append(_phase3g_smaps_timeline_row("after_threshold_cache_load"))
    del validation_scores
    del validation_cases
    gc.collect()
    _phase3g_malloc_trim()
    rss_timeline.append(_phase3g_smaps_timeline_row("after_validation_cleanup"))
    rss_timeline.append(_phase3g_smaps_timeline_row("before_test_streaming"))
    stream_outputs = _score_phase3g_conditional_fast_stream(
        config=config,
        head=head,
        profile_model=state_model,
        test_index=test_index,
        node_embeddings=node_embeddings,
        action_embeddings=action_embeddings,
        cache_meta=cache_meta,
        endpoint_pair_counts=endpoint_pair_counts,
        endpoint_src_endpoint_counts=endpoint_src_endpoint_counts,
        endpoint_counts=endpoint_counts,
        netflow_endpoint_by_idx=netflow_endpoint_by_idx,
        endpoint_lookup=endpoint_lookup,
        endpoint_cache_mb=endpoint_cache_mb,
        output_dir=output_dir,
        reference_update_path=not bool(config.sspm_infer_fast_path),
    )
    rss_timeline.append(_phase3g_smaps_timeline_row("test_streaming_peak"))
    rss_timeline.append(_phase3g_smaps_timeline_row("after_test_streaming_before_eval"))
    if endpoint_pair_count_summary:
        stream_outputs = dict(stream_outputs)
        suppression_payload = dict(stream_outputs.get("conditional_endpoint_suppression", {}))
        suppression_payload.update(endpoint_pair_count_summary)
        stream_outputs["conditional_endpoint_suppression"] = suppression_payload
    group_eval_started = time.perf_counter()
    group_eval = _phase3g_conditional_post_stream_group_eval(
        config=config,
        paths=paths,
        test_index=test_index,
        stream_outputs=stream_outputs,
        output_dir=output_dir,
    )
    endpoint_suppression_eval = {}
    post_stream_eval_rss_mb = 0.0
    if bool(config.conditional_endpoint_aware_suppression):
        endpoint_suppression_eval = _phase3g_endpoint_suppression_post_eval(
            config=config,
            paths=paths,
            test_index=test_index,
            stream_outputs=stream_outputs,
            output_dir=output_dir,
        )
        if endpoint_suppression_eval:
            post_stream_eval_rss_mb = float(
                endpoint_suppression_eval.get("post_stream_eval_rss_delta_mb", 0.0)
                or 0.0,
            )
            stream_outputs = dict(stream_outputs)
            stream_outputs["conditional_endpoint_suppression_eval"] = dict(
                endpoint_suppression_eval,
            )
    if group_eval:
        stream_outputs = dict(stream_outputs)
        stream_outputs["event_labels_by_event"] = dict(group_eval.get("event_labels_by_event", {}))
        stream_outputs["conditional_group_eval"] = dict(group_eval)
    idx_to_db_node_id = _phase3e_load_idx_to_node_id(paths)
    abnormal_nodes = _phase3g_load_abnormal_db_nodes_for_eval(config)
    coverage_report = _phase3g_write_event_coverage_report(
        output_dir=output_dir,
        alert_path=output_dir / "online_event_alerts.csv",
        coverage_rows=stream_outputs.get("event_node_coverage_rows", []),
        idx_to_db_node_id=idx_to_db_node_id,
        abnormal_db_node_ids=abnormal_nodes,
        topk_values=_parse_int_list(config.node_pool_topk_values),
    )
    group_alert_policy_rows = _phase3g_update_group_report_rows_with_eval(
        path=output_dir / "group_alert_policy_summary.csv",
        alert_path=output_dir / "online_event_alerts.csv",
        config=config,
        idx_to_db_node_id=idx_to_db_node_id,
        abnormal_db_node_ids=abnormal_nodes,
    )
    _phase3g_update_group_report_rows_with_eval(
        path=output_dir / "node_coverage_by_group.csv",
        alert_path=output_dir / "online_event_alerts.csv",
        config=config,
        idx_to_db_node_id=idx_to_db_node_id,
        abnormal_db_node_ids=abnormal_nodes,
    )
    _phase3g_update_group_report_rows_with_eval(
        path=output_dir / "q_t_by_action_type.csv",
        alert_path=output_dir / "online_event_alerts.csv",
        config=config,
        idx_to_db_node_id=idx_to_db_node_id,
        abnormal_db_node_ids=abnormal_nodes,
    )
    demoted_group_summary_path = _phase3g_write_demoted_group_summary(config, output_dir)
    if stream_outputs.get("action_type_alert_policy"):
        stream_outputs = dict(stream_outputs)
        policy_payload = dict(stream_outputs.get("action_type_alert_policy", {}))
        policy_payload["demoted_group_summary_csv"] = str(demoted_group_summary_path)
        stream_outputs["action_type_alert_policy"] = policy_payload
    group_threshold_sweep_path = _phase3g_write_group_threshold_sweep_summary(
        config=config,
        cache_meta=cache_meta,
        output_dir=output_dir,
        idx_to_db_node_id=idx_to_db_node_id,
        abnormal_db_node_ids=abnormal_nodes,
    )
    rss_timeline.append(_phase3g_smaps_timeline_row("after_post_stream_eval"))
    rss_timeline.append(_phase3g_smaps_timeline_row("after_event_node_coverage_flush"))
    timing["test_scoring_seconds"] = float(stream_outputs.get("test_scoring_seconds", 0.0))
    timing["label_attach_seconds"] = float(time.perf_counter() - group_eval_started)
    validation_summary = dict(cache_meta.get("summary", {}))
    validation_by_case = dict(cache_meta.get("summary_by_target_case", {}))
    group_summary_path = str(stream_outputs.get("conditional_group_summary_csv", ""))
    required_dual_head_summary_paths = _phase3g_write_required_dual_head_summary_csvs(
        output_dir=output_dir,
        stream_outputs=stream_outputs,
        group_summary_csv=group_summary_path,
    )
    score_summary_payload = _phase3g_conditional_score_summary_payload(
        config=config,
        cache_meta=cache_meta,
        stream_outputs=stream_outputs,
        group_summary_path=group_summary_path,
        demoted_group_summary_path=demoted_group_summary_path,
        group_threshold_sweep_path=group_threshold_sweep_path,
        endpoint_suppression_eval=endpoint_suppression_eval,
        update_gate_score_space_summary=update_gate_score_space_summary,
        group_alert_policy_rows=group_alert_policy_rows,
        required_dual_head_summary_paths=required_dual_head_summary_paths,
    )
    score_summary_path = output_dir / "score_summary.json"
    score_summary_path.write_text(
        json.dumps(score_summary_payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    memory_profile = _empty_memory_profile()
    memory_profile["rss_test_peak_mb"] = float(stream_outputs.get("rss_test_peak_mb", 0.0))
    memory_profile["deploy_rss_valid"] = True
    memory_profile["deploy_infer_rss_peak_mb"] = float(stream_outputs.get("rss_test_peak_mb", 0.0))
    memory_profile["deploy_infer_rss_final_mb"] = float(_current_rss_mb())
    memory_profile["deploy_infer_events_per_sec"] = float(stream_outputs.get("test_count", 0)) / max(
        float(stream_outputs.get("test_scoring_seconds", 0.0)),
        1e-9,
    )
    memory_profile["post_stream_eval_rss_mb"] = float(post_stream_eval_rss_mb)
    memory_profile["process_peak_rss_mb"] = float(stream_outputs.get("process_peak_rss_mb", 0.0))
    for key in (
        "embedding_lookup_mode",
        "embedding_lazy_backing",
        "embedding_cache_max_nodes",
        "embedding_cache_active_nodes",
        "embedding_lru_cache_mb",
        "embedding_lru_cache_mb_actual",
        "embedding_lru_cache_mb_capacity",
        "embedding_cache_hit_count",
        "embedding_cache_miss_count",
        "embedding_cache_hit_rate",
        "embedding_cache_miss_rate",
        "embedding_lookup_seconds",
        "embedding_backing_path",
    ):
        if key in stream_outputs:
            memory_profile[key] = stream_outputs[key]
    memory_profile["online_minimal_event_node_coverage_writer_mb"] = float(
        stream_outputs.get("online_minimal_event_node_coverage_writer_mb", 0.0) or 0.0,
    )
    memory_profile["online_event_node_coverage_count"] = int(
        coverage_report.get("coverage_node_count", 0),
    )
    param_mb = _phase3g_conditional_model_param_breakdown_mb(head, state_model)
    memory_profile["online_deploy_model_param_mb"] = float(param_mb["model_param_mb"])
    memory_profile["conditional_head_param_mb"] = float(param_mb["conditional_head_param_mb"])
    memory_profile["state_model_param_mb"] = float(param_mb["state_model_param_mb"])
    memory_profile["calibration_param_mb"] = float(param_mb["calibration_param_mb"])
    state_storage_audit = state_model.memory.storage_audit(
        state_memory_mode=str(config.state_memory_mode),
        state_memory_policy=str(config.sspm_state_memory_policy),
    )
    state_storage_audit_path = output_dir / "state_memory_storage_audit.json"
    state_storage_audit_path.write_text(
        json.dumps(state_storage_audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    eval_payload = _eval_payload(
        config,
        stream_outputs,
        state_model,
        max(time.perf_counter() - started, 1e-9),
        int(test_count),
        timing=timing,
        memory=memory_profile,
        split_metadata={
            "split_source": "phase3e_event_index_meta",
            "train_days": [],
            "validation_days": [],
            "test_days": [],
            "year_month": "",
            "slim_split_override": bool(config.slim_split_override),
            "slim_split_override_applied": False,
        },
    )
    eval_payload.update(
        {
            "dataset": str(config.dataset),
            "out_tag": str(config.out_tag),
            "train_events": 0,
            "validation_events": int(validation_count),
            "test_events": int(test_count),
            "train_events_actual": 0,
            "validation_events_actual": int(validation_count),
            "test_events_actual": int(test_count),
            "validation_event_score_summary": validation_summary,
            "validation_score_summary_by_target_case": validation_by_case,
            "score_summary": score_summary_payload,
            "score_summary_json": str(score_summary_path),
            "primary_online_metrics": {
                "event_alerts": dict(group_eval.get("event_alerts", {})),
            }
            if group_eval
            else eval_payload.get("primary_online_metrics", {}),
            "conditional_group_summary_csv": group_summary_path,
            "conditional_endpoint_suppression_summary_csv": stream_outputs.get(
                "conditional_endpoint_suppression_summary_csv",
                "",
            ),
            "online_event_node_coverage": coverage_report,
            "state_memory_storage_audit": state_storage_audit,
            "state_merge": dict(stream_outputs.get("state_merge", {})),
            "ofsm_compression": dict(stream_outputs.get("ofsm_compression", {})),
            "ofsm_runtime_profile": dict(stream_outputs.get("ofsm_runtime_profile", {})),
            "ofsm_runtime_profile_csv": str(stream_outputs.get("ofsm_runtime_profile_csv", "")),
            "ofsm_runtime_profile_json": str(stream_outputs.get("ofsm_runtime_profile_json", "")),
            "ofsm_runtime_config": runtime_config_summary,
            "phase3g": {
                "score_head": "conditional_action_semantic",
                **_phase3g_conditional_arch_metrics(config, head_metadata),
                "score_target_mode": str(config.sspm_score_target_mode),
                "node_repr_fusion": str(config.node_repr_fusion),
                "conditional_semantic_loss": str(config.conditional_semantic_loss),
                "action_head_checkpoint_path": str(checkpoint_path),
                "conditional_head_metadata": head_metadata,
                "validation_cache_dir": str(
                    Path(cache_meta["validation_conditional_scores"]).parent,
                ),
                "validation_cache_meta": cache_meta,
                "ofsm_runtime_config": runtime_config_summary,
                "conditional_group_summary_csv": group_summary_path,
                "target_case_summary_csv": str(
                    required_dual_head_summary_paths.get("target_case_summary_csv", ""),
                ),
                "group_alert_summary_csv": str(
                    required_dual_head_summary_paths.get("group_alert_summary_csv", ""),
                ),
                "phase3f_fast_path_enabled": bool(
                    stream_outputs.get("phase3f_fast_path_enabled", False),
                ),
                "phase3g_reference_update_path": bool(
                    stream_outputs.get("phase3g_reference_update_path", False),
                ),
                "conditional_endpoint_suppression": dict(
                    stream_outputs.get("conditional_endpoint_suppression", {}),
                ),
                "action_type_alert_policy": dict(
                    stream_outputs.get("action_type_alert_policy", {}),
                ),
                "update_gate_score_space": dict(update_gate_score_space_summary),
            },
        },
    )
    rss_breakdown = _phase3g_write_online_event_core_rss_breakdown(
        output_dir=output_dir,
        eval_payload=eval_payload,
        paths=paths,
        node_embeddings=node_embeddings,
        endpoint_cache_mb=endpoint_cache_mb,
        endpoint_cache_meta=endpoint_cache_meta_for_rss,
        rss_timeline=rss_timeline,
    )
    eval_payload["rss_breakdown_online_event_core"] = rss_breakdown
    eval_payload["outputs"] = dict(eval_payload.get("outputs", {}))
    eval_payload["outputs"].update(
        {
            "rss_breakdown_online_event_core_json": str(
                output_dir / "rss_breakdown_online_event_core.json",
            ),
            "rss_timeline_csv": str(output_dir / "rss_timeline.csv"),
            "online_event_node_coverage_csv": str(
                output_dir / "online_event_node_coverage.csv",
            ),
            "online_event_node_coverage_strict_csv": str(
                output_dir / "online_event_node_coverage_strict.csv",
            ),
            "online_event_node_coverage_relaxed_csv": str(
                output_dir / "online_event_node_coverage_relaxed.csv",
            ),
            "online_event_node_coverage_summary_json": str(
                output_dir / "online_event_node_coverage_summary.json",
            ),
            "node_pool_rebuilt_summary_json": str(
                output_dir / "node_pool_rebuilt_summary.json",
            ),
            "node_topk_metrics_csv": str(output_dir / "node_topk_metrics.csv"),
            "node_topk_metrics_json": str(output_dir / "node_topk_metrics.json"),
            "event_fp_group_summary_csv": str(output_dir / "event_fp_group_summary.csv"),
            "node_fp_group_summary_csv": str(output_dir / "node_fp_group_summary.csv"),
            "state_memory_storage_audit_json": str(state_storage_audit_path),
            "ofsm_runtime_profile_csv": str(stream_outputs.get("ofsm_runtime_profile_csv", "")),
            "ofsm_runtime_profile_json": str(stream_outputs.get("ofsm_runtime_profile_json", "")),
            "event_score_trace_csv": str(stream_outputs.get("event_score_trace_csv", "")),
            "q_t_by_action_type_csv": str(stream_outputs.get("q_t_by_action_type_csv", "")),
            "group_alert_policy_summary_csv": str(
                stream_outputs.get("group_alert_policy_summary_csv", ""),
            ),
            "target_case_summary_csv": str(
                required_dual_head_summary_paths.get("target_case_summary_csv", ""),
            ),
            "group_alert_summary_csv": str(
                required_dual_head_summary_paths.get("group_alert_summary_csv", ""),
            ),
            "node_coverage_by_group_csv": str(
                stream_outputs.get("node_coverage_by_group_csv", ""),
            ),
            "group_threshold_sweep_summary_csv": str(group_threshold_sweep_path),
        },
    )
    eval_payload["primary_online_metrics"] = dict(eval_payload.get("primary_online_metrics", {}))
    eval_payload["primary_online_metrics"]["online_event_node_coverage"] = {
        key: value
        for key, value in coverage_report.items()
        if not str(key).endswith("_csv") and not str(key).endswith("_json")
    }
    eval_payload["primary_online_metrics"]["node_pool_topk"] = coverage_report.get(
        "node_pool_topk",
        {},
    )
    eval_payload["runtime"]["elapsed_seconds"] = float(time.perf_counter() - started)
    eval_path = output_dir / "eval_causal_semantics_slim.json"
    eval_path.write_text(
        json.dumps(eval_payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_effective_config(
        output_dir,
        config,
        state_model,
        True,
        train_count=0,
        validation_count=int(validation_count),
        test_count=int(test_count),
    )
    write_metrics_json(
        output_dir,
        config,
        state_model,
        eval_payload,
        train_count=0,
        validation_count=int(validation_count),
        test_count=int(test_count),
        embedder_loaded=True,
    )
    return eval_path


def run_phase3e_load_and_infer_from_precompute(config: SlimConfig) -> Path:
    """Run one Phase3E checkpoint fanout inference from compact precompute artifacts."""
    if str(config.sspm_score_head) == "conditional_action_semantic":
        return run_phase3g_conditional_load_and_infer_from_precompute(config)
    _validate_active_event_score_mode(config.event_score_mode)
    if str(config.sspm_train_mode) != "load_and_infer":
        raise ValueError("Phase3E infer_ablation requires SSPM_TRAIN_MODE=load_and_infer")
    if str(config.sspm_target_mode) != "node_action_semantic_mean":
        raise ValueError("Phase3E infer_ablation requires node_action_semantic_mean")
    if not str(config.sspm_checkpoint_path).strip():
        raise ValueError("Phase3E infer_ablation requires --sspm_checkpoint_path")
    started = time.perf_counter()
    output_dir = Path(config.result_root) / config.out_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    process_cfg = _load_process_config(config)
    paths = _phase3e_require_artifacts(config)
    event_meta = _phase3e_load_json(paths["event_meta"])
    validation_index, validation_count = _phase3e_open_split_event_index(
        paths,
        event_meta,
        "validation",
        max_events=int(config.max_ref_events),
    )
    test_index, test_count = _phase3e_open_split_event_index(
        paths,
        event_meta,
        "test",
        max_events=int(config.max_test_events),
    )
    node_embeddings = np.load(paths["node_embeddings"], mmap_mode="r")
    action_embeddings = np.load(paths["action_embeddings"], mmap_mode="r")
    embedder = load_pretrained_residual_embedder(
        str(config.pretrained_residual_embedder_path),
        expected_dim=int(config.sspm_target_dim),
    )
    model, checkpoint_payload = load_sspm_checkpoint(config.sspm_checkpoint_path, config)
    runtime_config_summary = _phase3e_apply_load_and_infer_runtime_config(
        model,
        config,
        checkpoint_payload,
    )
    validate_checkpoint_embedder_fingerprints(checkpoint_payload, embedder, config, process_cfg)

    memory_profile = _empty_memory_profile()
    timing: dict[str, float] = {}
    train_artifact_count = int(
        dict(checkpoint_payload.get("event_counts_actual", {})).get("train_events_actual", 0),
    )
    _finish_test_phase_cleanup(
        memory_profile,
        released_payload=True,
        node_map_mode="phase3e_compact_event_index",
        released_train_validation=True,
    )
    validation_started = time.perf_counter()
    validation_residual_scores, validation_count_actual = _phase3e_score_event_index_stream(
        config,
        model,
        validation_index,
        node_embeddings,
        action_embeddings,
        enable_update_gate=False,
    )
    if int(validation_count_actual) != int(validation_count):
        raise ValueError("Phase3E validation event_index count mismatch")
    event_calibration = derive_event_calibration(
        validation_residual_scores,
        config.event_score_mode,
        config.expected_event_alert_budget,
        config.expected_alert_horizon_events or max(int(test_count), 1),
        threshold_mode=config.event_threshold_mode,
        fixed_quantile=config.event_threshold_quantile,
    )
    validation_seconds = float(time.perf_counter() - validation_started)
    timing["validation_seconds"] = validation_seconds
    memory_profile["rss_after_validation_mb"] = _current_rss_mb()

    stream_outputs = _score_phase3e_event_index_stream(
        config,
        model,
        test_index,
        node_embeddings,
        action_embeddings,
        event_calibration.residual_sorted,
        event_calibration.threshold,
        output_dir=output_dir,
        collect_in_memory=_task6_collect_in_memory_outputs(config),
        initial_test_peak_rss_mb=memory_profile["rss_after_test_cleanup_mb"],
    )
    score_summary = _phase3e_score_summary_payload(
        event_calibration,
        stream_outputs.get("test_score_summary", {}),
        event_alert_count=int(stream_outputs.get("event_alert_count", 0)),
    )
    score_summary_path = output_dir / "score_summary.json"
    score_summary_path.write_text(
        json.dumps(score_summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    timing["test_scoring_seconds"] = float(stream_outputs.get("test_scoring_seconds", 0.0))
    memory_profile["rss_test_peak_mb"] = float(stream_outputs.get("rss_test_peak_mb", 0.0))
    memory_profile["deploy_rss_valid"] = bool(
        _deploy_light_enabled(config) or _online_minimal_enabled(config),
    )
    memory_profile["deploy_infer_rss_peak_mb"] = float(memory_profile["rss_test_peak_mb"])
    memory_profile["deploy_infer_rss_final_mb"] = float(_current_rss_mb())
    memory_profile["deploy_infer_events_per_sec"] = float(stream_outputs.get("test_count", 0)) / max(
        float(stream_outputs.get("test_scoring_seconds", 0.0)),
        1e-9,
    )
    if _online_minimal_enabled(config):
        timing["label_attach_seconds"] = 0.0
        memory_profile["rss_after_label_attach_mb"] = _current_rss_mb()
        memory_profile["process_peak_rss_mb"] = float(
            stream_outputs.get("process_peak_rss_mb", 0.0),
        )
        elapsed_before_eval = max(time.perf_counter() - started, 1e-9)
        eval_payload = _eval_payload(
            config,
            stream_outputs,
            model,
            elapsed_before_eval,
            int(test_count),
            timing=timing,
            memory=memory_profile,
            split_metadata={
                "split_source": "phase3e_event_index_meta",
                "train_days": [],
                "validation_days": [],
                "test_days": [],
                "year_month": "",
                "slim_split_override": bool(config.slim_split_override),
                "slim_split_override_applied": False,
            },
        )
        eval_payload.update(
            {
                "dataset": str(config.dataset),
                "out_tag": str(config.out_tag),
                "train_events": int(train_artifact_count),
                "validation_events": int(validation_count),
                "test_events": int(test_count),
                "train_events_actual": int(train_artifact_count),
                "validation_events_actual": int(validation_count),
                "test_events_actual": int(test_count),
                "validation_event_score_summary": event_calibration.summary,
                "score_summary": score_summary,
                "score_summary_json": str(score_summary_path),
                "ofsm_runtime_config": runtime_config_summary,
                "phase3f": {
                    "fast_path_enabled": bool(
                        stream_outputs.get("phase3f_fast_path_enabled", False),
                    ),
                    "online_minimal": True,
                    "infer_chunk_events": int(config.sspm_infer_chunk_events),
                },
            },
        )
        eval_payload["phase3e"] = {
            "target_mode": str(config.sspm_target_mode),
            "node_word2vec_source": str(config.node_word2vec_source),
            "train_backend": str(config.sspm_train_backend),
            "infer_backend": str(config.sspm_infer_backend),
            "event_index_meta_path": str(paths["event_meta"]),
            "node_embedding_path": str(paths["node_embeddings"]),
            "action_embedding_path": str(paths["action_embeddings"]),
            "checkpoint_path": str(config.sspm_checkpoint_path),
            "ofsm_runtime_config": runtime_config_summary,
        }
        eval_payload["runtime"]["elapsed_seconds"] = float(time.perf_counter() - started)
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_path = output_dir / "eval_causal_semantics_slim.json"
        eval_path.write_text(
            json.dumps(eval_payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        write_effective_config(
            output_dir,
            config,
            model,
            True,
            train_count=int(train_artifact_count),
            validation_count=int(validation_count),
            test_count=int(test_count),
        )
        write_metrics_json(
            output_dir,
            config,
            model,
            eval_payload,
            train_count=int(train_artifact_count),
            validation_count=int(validation_count),
            test_count=int(test_count),
            embedder_loaded=True,
        )
        return eval_path

    label_started = time.perf_counter()
    db_cfg = _cfg_for_dataset(config.dataset)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    cur = None
    conn = None
    try:
        cur, conn = init_database_connection(db_cfg)
        eval_node_maps = build_node_maps(cur)
        abnormal_nodes = load_ground_truth_indices_readonly(
            db_cfg,
            eval_node_maps["uuid2index"],
        )
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
    idx_to_db_node_id = _phase3e_load_idx_to_node_id(paths)
    event_labels_by_event = _phase3e_event_labels_for_alerts(
        records=test_index,
        alert_event_indices=_alert_event_indices(stream_outputs),
        abnormal_db_node_ids=abnormal_nodes,
        idx_to_db_node_id=idx_to_db_node_id,
    )
    node_labels_by_node = _phase3e_node_labels_for_alert_pool(
        list(stream_outputs.get("final_nodes_raw", [])),
        abnormal_nodes,
        idx_to_db_node_id,
    )
    stream_outputs["event_labels_by_event"] = event_labels_by_event
    stream_outputs["node_labels_by_node"] = node_labels_by_node
    stream_outputs.update(_embedding_eval_metadata(config, embedder))
    timing["label_attach_seconds"] = float(time.perf_counter() - label_started)
    memory_profile["rss_after_label_attach_mb"] = _current_rss_mb()
    memory_profile["process_peak_rss_mb"] = float(stream_outputs.get("process_peak_rss_mb", 0.0))

    del validation_index
    del test_index
    del node_embeddings
    del action_embeddings
    del validation_residual_scores
    gc.collect()

    elapsed_before_eval = max(time.perf_counter() - started, 1e-9)
    eval_payload = _eval_payload(
        config,
        stream_outputs,
        model,
        elapsed_before_eval,
        int(test_count),
        timing=timing,
        memory=memory_profile,
        split_metadata={
            "split_source": "phase3e_event_index_meta",
            "train_days": [],
            "validation_days": [],
            "test_days": [],
            "year_month": "",
            "slim_split_override": bool(config.slim_split_override),
            "slim_split_override_applied": False,
        },
    )
    eval_payload.update(
        {
            "dataset": str(config.dataset),
            "out_tag": str(config.out_tag),
            "cache": {
            "cache_scope": "phase3e_checkpoint_fanout",
            "checkpoint_path": str(config.sspm_checkpoint_path),
            "skip_train": True,
            "train_mode": str(config.sspm_train_mode),
            "train_data_mode": str(config.sspm_train_data_mode),
            "validation_rows_processed": int(validation_count),
            "test_rows_processed": int(test_count),
            },
            "train_events": int(train_artifact_count),
            "validation_events": int(validation_count),
            "test_events": int(test_count),
            "train_events_actual": int(train_artifact_count),
            "validation_events_actual": int(validation_count),
            "test_events_actual": int(test_count),
            "validation_event_score_summary": event_calibration.summary,
            "score_summary": score_summary,
            "score_summary_json": str(score_summary_path),
            "ofsm_runtime_config": runtime_config_summary,
        },
    )
    eval_payload["phase3e"] = {
        "target_mode": str(config.sspm_target_mode),
        "node_word2vec_source": str(config.node_word2vec_source),
        "train_backend": str(config.sspm_train_backend),
        "infer_backend": str(config.sspm_infer_backend),
        "event_index_meta_path": str(paths["event_meta"]),
        "node_embedding_path": str(paths["node_embeddings"]),
        "action_embedding_path": str(paths["action_embeddings"]),
        "checkpoint_path": str(config.sspm_checkpoint_path),
        "ofsm_runtime_config": runtime_config_summary,
    }
    eval_payload["runtime"]["elapsed_seconds"] = float(time.perf_counter() - started)
    final_payload = _write_outputs(output_dir, stream_outputs, eval_payload)
    write_effective_config(
        output_dir,
        config,
        model,
        True,
        train_count=int(train_artifact_count),
        validation_count=int(validation_count),
        test_count=int(test_count),
    )
    write_metrics_json(
        output_dir,
        config,
        model,
        final_payload,
        train_count=int(train_artifact_count),
        validation_count=int(validation_count),
        test_count=int(test_count),
        embedder_loaded=True,
    )
    return output_dir / "eval_causal_semantics_slim.json"


# Explicit cross-module imports; replaces the migration namespace bridge.
from scripts.pipeline.checks.preflight import (
    _load_process_config,
    build_node_maps,
    load_ground_truth_indices_readonly,
)
from scripts.pipeline.features.conditional_context import (
    _action_type_alert_policy_decision,
    _conditional_endpoint_action_counts,
    _conditional_endpoint_action_signature,
    _conditional_endpoint_cache_dir,
    _conditional_endpoint_cache_fingerprint,
    _conditional_endpoint_pair_counts,
    _conditional_endpoint_pair_signature,
    _conditional_endpoint_summary_update,
    _conditional_endpoint_suppression_decision,
    _conditional_src_endpoint_counts,
    _conditional_src_endpoint_signature,
    _load_conditional_endpoint_suppression_cache,
    _phase3e_load_idx_to_node_id,
    _phase3g_action_context_from_model,
    _phase3g_action_input_dim,
    _phase3g_action_policy_group_rows,
    _phase3g_conditional_arch_metrics,
    _phase3g_conditional_event_score,
    _phase3g_load_netflow_endpoint_by_idx,
    _phase3g_q_t_group_rows,
    _phase3g_resolved_head_checkpoint_path,
    _phase3g_type_eye,
    _score_phase3e_event_index_stream,
    _write_conditional_endpoint_suppression_cache,
    _write_conditional_endpoint_suppression_summary,
)
from scripts.pipeline.features.semantic_features import _StreamingScoreSummary, derive_event_calibration
from scripts.pipeline.io.cache_payloads import (
    _phase3e_score_summary_payload,
    _phase3g_load_abnormal_db_nodes_for_eval,
)
from scripts.pipeline.io.conditional_cache import (
    _phase3g_conditional_threshold_for_case,
    _phase3g_load_or_build_conditional_validation_cache,
)
from scripts.pipeline.io.db_stream import _cfg_for_dataset
from scripts.pipeline.io.event_artifacts import (
    _phase3e_entity_type_name,
    _phase3e_event_index_array,
    _phase3e_load_json,
    _phase3e_open_split_event_index,
    _phase3e_require_artifacts,
    _phase3g_effective_artifacts,
    _phase3g_open_node_embeddings,
    stable_json_hash,
    write_effective_config,
    write_metrics_json,
)
from scripts.pipeline.outputs.alert_output import (
    _alert_event_indices,
    _current_rss_mb,
    _deploy_light_enabled,
    _eval_payload,
    _final_node_pool_minimal,
    _minimal_node_pool_update,
    _online_minimal_enabled,
    _parse_int_list,
    _raw_output_paths,
    _safe_peak_rss_mb,
    _safe_state_array_mb,
    _task6_collect_in_memory_outputs,
    _write_csv,
    _write_outputs,
)
from scripts.pipeline.outputs.conditional_reports import (
    _ConditionalGroupStreamingSummary,
    _TargetCaseStreamingSummary,
    _phase3g_conditional_model_param_breakdown_mb,
    _phase3g_conditional_post_stream_group_eval,
    _phase3g_endpoint_suppression_post_eval,
    _phase3g_malloc_trim,
    _phase3g_smaps_timeline_row,
    _phase3g_update_group_report_rows_with_eval,
    _phase3g_write_demoted_group_summary,
    _phase3g_write_event_coverage_report,
    _phase3g_write_group_threshold_sweep_summary,
    _phase3g_write_online_event_core_rss_breakdown,
    _phase3g_write_required_dual_head_summary_csvs,
    _write_conditional_group_test_summary,
)
from scripts.pipeline.outputs.metrics_summary import (
    _current_smaps_rollup_mb,
    _embedding_eval_metadata,
    _emit_profile_row,
    _empty_memory_profile,
    _finish_test_phase_cleanup,
    _ofsm_compression_summary,
    _ofsm_runtime_profile_payload,
    _profiling_row,
    _state_merge_enabled,
    _state_merge_profile_row,
    _validate_active_event_score_mode,
    _write_ofsm_runtime_profile,
)
from scripts.pipeline.state.online_state_runtime import (
    _make_sspm_config,
    _phase3e_apply_load_and_infer_runtime_config,
    _phase3e_fields_from_index_row,
    _phase3e_score_event_index_stream,
    _phase3e_target_from_index_row,
    _phase3f_e2_none_update_from_numeric_row,
    _phase3f_raw_alert_row,
    _phase3g_apply_conditional_update_gate_score_space,
    _phase3g_update_gate_enabled,
    _stage_log,
    load_pretrained_residual_embedder,
    load_sspm_checkpoint,
    load_sspm_checkpoint_state_for_conditional,
    validate_checkpoint_embedder_fingerprints,
)
