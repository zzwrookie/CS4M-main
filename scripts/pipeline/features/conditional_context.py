"""Phase3G conditional context construction, endpoint suppression, and score streams."""

from __future__ import annotations

from scripts.pipeline.config.runtime_config import *


def _event_group_names_from_row(row: np.void) -> tuple[str, str, str]:
    action = str(ORTHRUS10_ACTION_NAMES[int(row["action_id"])])
    src_type = _phase3e_entity_type_name("src_type_id", int(row["src_type_id"]))
    dst_type = _phase3e_entity_type_name("dst_type_id", int(row["dst_type_id"]))
    return action, src_type, dst_type


def _action_type_alert_policy_decision(
    *,
    config: SlimConfig,
    row: np.void,
    raw_alert: bool,
    score: float | None = None,
    threshold: float | None = None,
    alert_row: Mapping[str, Any] | None = None,
    policy_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return alert eligibility for action/type policy without changing event score."""
    policy = str(config.action_type_alert_policy)
    if policy not in ACTION_TYPE_ALERT_POLICIES:
        raise ValueError(
            "ACTION_TYPE_ALERT_POLICY must be default, theia_v1, "
            "cadets_e4_v2_group_v1, cadets_e4_v3_policy_smoke_v1, "
            "clearscope_node_pair_v1, "
            "clearscope_android_v2, clearscope_v31_fp_guard_v1, "
            "clearscope_v31_fp_guard_v2, clearscope_v31_fp_guard_v3, "
            "clearscope_v31_fp_guard_v3b, or clearscope_v31_fp_guard_v3c",
        )
    if not bool(raw_alert):
        return {
            "policy_name": policy,
            "alert_decision": "not_alert",
            "alert_priority": "",
            "final_alert": False,
            "node_evidence": False,
            "budget_capped": False,
        }
    action, src_type, dst_type = _event_group_names_from_row(row)
    if policy == "default":
        return {
            "policy_name": policy,
            "alert_decision": "event_alert",
            "alert_priority": "default",
            "final_alert": True,
            "node_evidence": False,
            "budget_capped": False,
        }

    group = (action, src_type, dst_type)
    if policy in {
        "clearscope_v31_fp_guard_v1",
        "clearscope_v31_fp_guard_v2",
        "clearscope_v31_fp_guard_v3",
        "clearscope_v31_fp_guard_v3b",
        "clearscope_v31_fp_guard_v3c",
    }:
        demoted_to_evidence = {
            ("EVENT_OPEN", "process", "process"),
            ("EVENT_READ", "process", "process"),
        }
        if group in demoted_to_evidence:
            return {
                "policy_name": policy,
                "alert_decision": "demoted_event",
                "alert_priority": "node_evidence",
                "final_alert": False,
                "node_evidence": True,
                "budget_capped": False,
            }
        if policy in {
            "clearscope_v31_fp_guard_v2",
            "clearscope_v31_fp_guard_v3",
            "clearscope_v31_fp_guard_v3b",
            "clearscope_v31_fp_guard_v3c",
        }:
            floor_by_group = {
                ("EVENT_WRITE", "process", "file"): float(
                    config.clearscope_v31_fp_guard_write_floor,
                ),
                ("EVENT_READ", "file", "process"): float(
                    config.clearscope_v31_fp_guard_read_floor,
                ),
            }
            if group in floor_by_group:
                score_value = float(score if score is not None else 0.0)
                threshold_value = float(threshold if threshold is not None else 0.0)
                floor = max(float(floor_by_group[group]), threshold_value)
                if score_value < floor:
                    return {
                        "policy_name": policy,
                        "alert_decision": "demoted_event",
                        "alert_priority": "node_evidence",
                        "final_alert": False,
                        "node_evidence": True,
                        "budget_capped": False,
                    }
                if policy == "clearscope_v31_fp_guard_v3" and alert_row is not None:
                    validation_group_count = int(
                        float(alert_row.get("validation_group_count", 0) or 0),
                    )
                    endpoint_pair_count = int(
                        float(alert_row.get("endpoint_validation_pair_count", 0) or 0),
                    )
                    src_endpoint_count = int(
                        float(alert_row.get("src_endpoint_count", 0) or 0),
                    )
                    target_case = str(alert_row.get("target_case", "")).lower()
                    validation_bucket = str(
                        alert_row.get("validation_count_bucket", ""),
                    ).lower()
                    low_support = target_case in {"both_cold", "cold", "low_support"}
                    low_support = low_support or validation_bucket in {
                        "low",
                        "low_support",
                    }
                    narrow_margin = score_value < floor + 0.005
                    common_endpoint = (
                        endpoint_pair_count >= 20 or src_endpoint_count >= 20
                    )
                    if (
                        narrow_margin
                        and validation_group_count >= 1000
                        and common_endpoint
                        and not low_support
                    ):
                        return {
                            "policy_name": policy,
                            "alert_decision": "demoted_event",
                            "alert_priority": "node_evidence",
                            "final_alert": False,
                            "node_evidence": True,
                            "budget_capped": False,
                        }
                if (
                    policy in {
                        "clearscope_v31_fp_guard_v3b",
                        "clearscope_v31_fp_guard_v3c",
                    }
                    and alert_row is not None
                ):
                    validation_group_count = int(
                        float(alert_row.get("validation_group_count", 0) or 0),
                    )
                    target_case = str(alert_row.get("target_case", "")).lower()
                    validation_bucket = str(
                        alert_row.get("validation_count_bucket", ""),
                    ).lower()
                    low_support = target_case in {
                        "both_cold",
                        "both_cold_action_target",
                        "cold",
                        "low_support",
                    }
                    low_support = low_support or validation_bucket in {
                        "low",
                        "low_support",
                    }
                    if policy == "clearscope_v31_fp_guard_v3c":
                        margin_by_group = {
                            ("EVENT_READ", "file", "process"): 0.0015,
                            ("EVENT_WRITE", "process", "file"): 0.002,
                        }
                        support_by_group = {
                            ("EVENT_READ", "file", "process"): (2, 4),
                            ("EVENT_WRITE", "process", "file"): (1, 2),
                        }
                    else:
                        margin_by_group = {
                            ("EVENT_READ", "file", "process"): 0.003,
                            ("EVENT_WRITE", "process", "file"): 0.002,
                        }
                        support_by_group = {
                            ("EVENT_READ", "file", "process"): (1, 2),
                            ("EVENT_WRITE", "process", "file"): (1, 2),
                        }
                    margin = float(margin_by_group.get(group, 0.0))
                    alert_support_threshold, evidence_support_threshold = (
                        support_by_group.get(group, (1, 2))
                    )
                    context = dict(policy_context or {})
                    prior_alert_support = max(
                        int(context.get("src_alert_count", 0) or 0),
                        int(context.get("dst_alert_count", 0) or 0),
                    )
                    prior_evidence_support = max(
                        int(context.get("src_node_evidence_count", 0) or 0),
                        int(context.get("dst_node_evidence_count", 0) or 0),
                    )
                    prior_support = (
                        prior_alert_support >= alert_support_threshold
                        or prior_evidence_support >= evidence_support_threshold
                    )
                    if (
                        margin > 0.0
                        and score_value < floor + margin
                        and validation_group_count >= 1000
                        and prior_support
                        and not low_support
                    ):
                        if policy == "clearscope_v31_fp_guard_v3c" and group == (
                            "EVENT_READ",
                            "file",
                            "process",
                        ):
                            if prior_alert_support >= alert_support_threshold:
                                support_reason = "v3c_read_strict_prior_alert"
                            else:
                                support_reason = "v3c_read_strict_prior_evidence"
                        elif prior_alert_support >= alert_support_threshold:
                            support_reason = "v3b_node_evidence_prior_alert"
                        else:
                            support_reason = "v3b_node_evidence_prior_evidence"
                        return {
                            "policy_name": policy,
                            "alert_decision": "demoted_event",
                            "alert_priority": "node_evidence",
                            "final_alert": False,
                            "node_evidence": True,
                            "budget_capped": False,
                            "policy_support_reason": support_reason,
                            "policy_margin_used": margin,
                            "src_prior_alert_count": int(
                                context.get("src_alert_count", 0) or 0,
                            ),
                            "dst_prior_alert_count": int(
                                context.get("dst_alert_count", 0) or 0,
                            ),
                            "src_prior_node_evidence_count": int(
                                context.get("src_node_evidence_count", 0) or 0,
                            ),
                            "dst_prior_node_evidence_count": int(
                                context.get("dst_node_evidence_count", 0) or 0,
                            ),
                        }
        return {
            "policy_name": policy,
            "alert_decision": "event_alert",
            "alert_priority": "default",
            "final_alert": True,
            "node_evidence": False,
            "budget_capped": False,
        }

    if policy == "cadets_e4_v2_group_v1":
        high_priority = {
            ("EVENT_RECVFROM", "netflow", "process"),
        }
        demoted_to_evidence = {
            ("EVENT_CONNECT", "process", "netflow"),
            ("EVENT_WRITE", "process", "netflow"),
        }
        if group in high_priority:
            return {
                "policy_name": policy,
                "alert_decision": "event_alert",
                "alert_priority": "high",
                "final_alert": True,
                "node_evidence": False,
                "budget_capped": False,
            }
        if group in demoted_to_evidence:
            return {
                "policy_name": policy,
                "alert_decision": "demoted_event",
                "alert_priority": "node_evidence",
                "final_alert": False,
                "node_evidence": True,
                "budget_capped": False,
            }
        return {
            "policy_name": policy,
            "alert_decision": "event_alert",
            "alert_priority": "default",
            "final_alert": True,
            "node_evidence": False,
            "budget_capped": False,
        }

    if policy == "cadets_e4_v3_policy_smoke_v1":
        high_priority = {
            ("EVENT_CONNECT", "process", "netflow"),
            ("EVENT_RECVFROM", "netflow", "process"),
        }
        demoted_to_evidence = {
            ("EVENT_CONNECT", "process", "file"),
            ("EVENT_SENDTO", "process", "file"),
            ("EVENT_SENDMSG", "process", "file"),
            ("EVENT_RECVMSG", "file", "process"),
        }
        if group in high_priority:
            return {
                "policy_name": policy,
                "alert_decision": "event_alert",
                "alert_priority": "high",
                "final_alert": True,
                "node_evidence": False,
                "budget_capped": False,
            }
        if group in demoted_to_evidence:
            return {
                "policy_name": policy,
                "alert_decision": "demoted_event",
                "alert_priority": "node_evidence",
                "final_alert": False,
                "node_evidence": True,
                "budget_capped": False,
            }
        return {
            "policy_name": policy,
            "alert_decision": "event_alert",
            "alert_priority": "default",
            "final_alert": True,
            "node_evidence": False,
            "budget_capped": False,
        }

    if policy == "clearscope_node_pair_v1":
        high_priority = {
            ("EVENT_READ", "file", "process"),
            ("EVENT_RECVFROM", "netflow", "process"),
            ("EVENT_CONNECT", "process", "netflow"),
        }
        demoted_to_evidence = {
            ("EVENT_WRITE", "process", "file"),
            ("EVENT_OPEN", "process", "process"),
        }
        if group in high_priority:
            return {
                "policy_name": policy,
                "alert_decision": "event_alert",
                "alert_priority": "high",
                "final_alert": True,
                "node_evidence": False,
                "budget_capped": False,
            }
        if group in demoted_to_evidence:
            return {
                "policy_name": policy,
                "alert_decision": "demoted_event",
                "alert_priority": "node_evidence",
                "final_alert": False,
                "node_evidence": True,
                "budget_capped": False,
            }
        return {
            "policy_name": policy,
            "alert_decision": "demoted_event",
            "alert_priority": "node_evidence",
            "final_alert": False,
            "node_evidence": True,
            "budget_capped": False,
        }

    if policy == "clearscope_android_v2":
        high_priority = {
            ("EVENT_RECVFROM", "netflow", "process"),
            ("EVENT_CONNECT", "process", "netflow"),
        }
        demoted_to_evidence = {
            ("EVENT_READ", "file", "process"),
            ("EVENT_READ", "process", "process"),
            ("EVENT_WRITE", "process", "file"),
            ("EVENT_OPEN", "process", "process"),
            ("EVENT_SENDTO", "process", "file"),
            ("EVENT_SENDTO", "process", "netflow"),
            ("EVENT_SENDMSG", "process", "file"),
            ("EVENT_SENDMSG", "process", "netflow"),
            ("EVENT_RECVMSG", "file", "process"),
            ("EVENT_RECVMSG", "netflow", "process"),
        }
        if group in high_priority:
            return {
                "policy_name": policy,
                "alert_decision": "event_alert",
                "alert_priority": "high",
                "final_alert": True,
                "node_evidence": False,
                "budget_capped": False,
            }
        if group in demoted_to_evidence:
            return {
                "policy_name": policy,
                "alert_decision": "demoted_event",
                "alert_priority": "node_evidence",
                "final_alert": False,
                "node_evidence": True,
                "budget_capped": False,
            }
        return {
            "policy_name": policy,
            "alert_decision": "demoted_event",
            "alert_priority": "node_evidence",
            "final_alert": False,
            "node_evidence": True,
            "budget_capped": False,
        }

    high_priority = {
        ("EVENT_RECVFROM", "netflow", "process"),
        ("EVENT_CONNECT", "process", "netflow"),
    }
    demoted_to_evidence = {
        ("EVENT_OPEN", "process", "process"),
        ("EVENT_WRITE", "process", "file"),
        ("EVENT_READ", "file", "process"),
    }
    if group in high_priority:
        return {
            "policy_name": policy,
            "alert_decision": "event_alert",
            "alert_priority": "high",
            "final_alert": True,
            "node_evidence": False,
            "budget_capped": False,
        }
    if group in demoted_to_evidence:
        return {
            "policy_name": policy,
            "alert_decision": "demoted_event",
            "alert_priority": "node_evidence",
            "final_alert": False,
            "node_evidence": True,
            "budget_capped": False,
        }
    return {
        "policy_name": policy,
        "alert_decision": "event_alert",
        "alert_priority": "default",
        "final_alert": True,
        "node_evidence": False,
        "budget_capped": False,
    }


def _phase3g_group_names_from_ids(
    action_id: int,
    src_type_id: int,
    dst_type_id: int,
) -> tuple[str, str, str]:
    action = str(ORTHRUS10_ACTION_NAMES[int(action_id)])
    src_type = _phase3e_entity_type_name("src_type_id", int(src_type_id))
    dst_type = _phase3e_entity_type_name("dst_type_id", int(dst_type_id))
    return action, src_type, dst_type


def _phase3g_empty_report_metrics(config: SlimConfig) -> dict[str, Any]:
    return {
        "dataset": str(config.dataset),
        "run": str(config.out_tag),
        "state_model": str(config.sspm_state_model),
        "policy_name": str(config.action_type_alert_policy),
        "threshold": "",
        "event_count": 0,
        "alert_count": 0,
        "TP": 0,
        "FP": 0,
        "precision": 0.0,
        "covered_malicious_nodes": 0,
        "strict_node_TP": 0,
        "strict_node_FP": 0,
        "relaxed_node_TP": 0,
        "relaxed_node_FP": 0,
        "q_t_mean": "",
        "q_t_applied_count": "",
        "speed": "",
        "online_deploy_primary_memory_mb": "",
    }


def _phase3g_q_t_group_rows(
    config: SlimConfig,
    q_t_by_group: Mapping[tuple[int, int, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(q_t_by_group):
        action, src_type, dst_type = _phase3g_group_names_from_ids(*key)
        payload = dict(q_t_by_group[key])
        count = int(payload.get("count", 0) or 0)
        q_t_sum = float(payload.get("q_t_sum", 0.0) or 0.0)
        q_t_sq_sum = float(payload.get("q_t_sq_sum", 0.0) or 0.0)
        q_t_mean = float(q_t_sum / max(count, 1)) if count else 1.0
        variance = max(float(q_t_sq_sum / max(count, 1)) - q_t_mean * q_t_mean, 0.0)
        rows.append(
            {
                **_phase3g_empty_report_metrics(config),
                "action": action,
                "src_type": src_type,
                "dst_type": dst_type,
                "event_count": count,
                "q_t_mean": q_t_mean,
                "q_t_min": 1.0 if count == 0 else float(payload.get("q_t_min", 1.0)),
                "q_t_max": 1.0 if count == 0 else float(payload.get("q_t_max", 1.0)),
                "q_t_std": float(math.sqrt(variance)),
                "q_t_applied_count": int(payload.get("q_t_applied_count", 0) or 0),
                "q_t_skipped_count": int(payload.get("q_t_skipped_count", 0) or 0),
            },
        )
    return rows


def _phase3g_action_policy_group_rows(
    config: SlimConfig,
    action_policy_counts: Mapping[tuple[int, int, int], Mapping[str, Any]],
    q_t_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    q_lookup = {
        (
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
        ): row
        for row in q_t_rows
    }
    rows: list[dict[str, Any]] = []
    for key in sorted(action_policy_counts):
        action, src_type, dst_type = _phase3g_group_names_from_ids(*key)
        payload = dict(action_policy_counts[key])
        threshold_count = int(payload.get("threshold_count", 0) or 0)
        threshold = (
            float(payload.get("threshold_sum", 0.0) or 0.0) / float(threshold_count)
            if threshold_count
            else ""
        )
        q_row = q_lookup.get((action, src_type, dst_type), {})
        rows.append(
            {
                **_phase3g_empty_report_metrics(config),
                "action": action,
                "src_type": src_type,
                "dst_type": dst_type,
                "threshold": threshold,
                "event_count": int(payload.get("event_count", 0) or 0),
                "alert_count": int(payload.get("alert_count", 0) or 0),
                "q_t_mean": q_row.get("q_t_mean", ""),
                "q_t_applied_count": q_row.get("q_t_applied_count", ""),
                "node_evidence_count": int(payload.get("node_evidence_count", 0) or 0),
                "demoted_event_count": int(payload.get("demoted_event_count", 0) or 0),
                "budget_capped_event_count": int(
                    payload.get("budget_capped_event_count", 0) or 0,
                ),
                "high_priority_event_alert_count": int(
                    payload.get("high_priority_event_alert_count", 0) or 0,
                ),
            },
        )
    return rows


def _phase3g_conditional_event_score(
    head: ConditionalSemanticHead,
    context: np.ndarray,
    target: np.ndarray,
    loss_type: str,
    target_case: str | int | None = None,
    *,
    score_target_mode: str = SCORE_TARGET_EVENT_ACTION,
    row: np.void | None = None,
    node_embeddings: np.ndarray | None = None,
    action_embeddings: np.ndarray | None = None,
) -> float:
    """Score one conditional semantic event without allocating a Python row."""
    del action_embeddings
    score_target = np.asarray(target, dtype=np.float32)
    if str(score_target_mode) == SCORE_TARGET_NODE_PAIR:
        if row is None or node_embeddings is None:
            raise ValueError("node_pair_no_action scoring requires row and node embeddings")
        src = np.asarray(node_embeddings[int(row["src_node_idx"])], dtype=np.float32)
        dst = np.asarray(node_embeddings[int(row["dst_node_idx"])], dtype=np.float32)
        score_target = node_pair_no_action_target(src, dst)
    elif str(score_target_mode) != SCORE_TARGET_EVENT_ACTION:
        raise ValueError(f"unsupported SSPM_SCORE_TARGET_MODE: {score_target_mode}")
    target_case_ids = None
    if head.is_dual_head:
        if target_case is None:
            raise ValueError("dual-head v2 event scoring requires target_case")
        if isinstance(target_case, str):
            target_case_ids = np.asarray([target_case_id(target_case)], dtype=np.int8)
        else:
            target_case_ids = np.asarray([int(target_case)], dtype=np.int8)
    prediction = head.predict(
        np.asarray(context, dtype=np.float32).reshape(1, -1),
        target_case_ids=target_case_ids,
    )
    score = conditional_distance(
        prediction,
        score_target.reshape(1, -1),
        str(loss_type),
    )
    return float(score[0])


def _score_phase3f_e2_none_fast_stream(*_args: object, **_kwargs: object) -> dict[str, Any]:
    """Legacy Phase3E low-rank fast scorer disabled in the active pipeline."""
    _phase3e_require_legacy_head_path_disabled()



def _score_phase3e_event_index_stream(
    config: SlimConfig,
    sspm: SSPMLowRankModel,
    test_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    residual_scores: Sequence[float],
    threshold: float,
    output_dir: Path | None = None,
    collect_in_memory: bool | None = None,
    initial_test_peak_rss_mb: float | None = None,
) -> dict[str, Any]:
    """Score Phase3E compact test events with node/action semantic targets."""
    if _phase3f_fast_path_enabled(config, sspm):
        return _score_phase3f_e2_none_fast_stream(
            config,
            sspm,
            test_index,
            node_embeddings,
            action_embeddings,
            residual_scores,
            threshold,
            output_dir=output_dir,
            initial_test_peak_rss_mb=initial_test_peak_rss_mb,
        )
    scoring_started = time.perf_counter()
    if collect_in_memory is None:
        collect_in_memory = bool(config.compat_in_memory_outputs)
    keep_alert_rows_for_eval = bool(collect_in_memory) or not _write_raw_alerts_enabled(config)
    sspm.reset_state()
    event_alerts_raw: list[dict[str, Any]] | None = [] if keep_alert_rows_for_eval else None
    checkpoints_raw: list[dict[str, Any]] | None = [] if collect_in_memory else None
    node_pool: dict[int, dict[str, Any]] = {}
    residual_sorted = _compact_sorted_scores(residual_scores)
    records = _phase3e_event_index_array(test_index)
    test_count = 0
    event_alert_count = 0
    checkpoint_count = 0
    test_score_summary = _StreamingScoreSummary(threshold)
    node_feature_counts: dict[int, dict[str, int]] = {}
    last_checkpoint_written_stream_pos: int | None = None
    checkpoint_interval = max(int(config.progress_interval_events), 1)
    progress_interval = int(config.progress_interval_events)
    raw_paths: dict[str, Path] = {}
    threshold_controller: AdaptiveRateThresholdController | None = None
    if str(config.event_threshold_mode) == "adaptive_rate":
        target_rate = float(config.expected_event_alert_budget) / float(
            max(int(config.expected_alert_horizon_events), 1),
        )
        threshold_controller = AdaptiveRateThresholdController(
            initial_threshold=float(threshold),
            target_rate=target_rate,
        )
    if initial_test_peak_rss_mb is None:
        test_phase_peak_rss_mb = _current_rss_mb()
    else:
        test_phase_peak_rss_mb = float(initial_test_peak_rss_mb)
    _stage_log(config, "phase3e_test_scoring_start", count=int(records.shape[0]))
    with ExitStack() as stack:
        event_writer = None
        checkpoint_writer = None
        profile_writer = None
        sspm_profiler = None
        state_merge_profile_writer = None
        state_merge_diagnostics_writer = None
        state_merge_diagnostics_rows = 0
        state_merge_diagnostics_truncated = False
        if output_dir is not None:
            raw_paths = _raw_output_paths(Path(output_dir))
            if _write_raw_alerts_enabled(config):
                event_writer = stack.enter_context(
                    StreamingCsvWriter(raw_paths["events"], EVENT_RAW_FIELDS),
                )
                checkpoint_writer = stack.enter_context(
                    StreamingCsvWriter(raw_paths["checkpoints"], CHECKPOINT_RAW_FIELDS),
                )
            profile_writer = stack.enter_context(
                StreamingCsvWriter(raw_paths["profiling"], PROFILING_FIELDS),
            )
            sspm_profiler = SSPMProfiler(raw_paths["profiling_sspm_raw"])
            if _state_merge_enabled(config):
                state_merge_profile_writer = stack.enter_context(
                    StreamingCsvWriter(
                        raw_paths["state_merge_profile"],
                        STATE_MERGE_PROFILE_FIELDS,
                    ),
                )
                if _write_state_merge_diagnostics_enabled(config):
                    state_merge_diagnostics_writer = stack.enter_context(
                        StreamingCsvWriter(
                            raw_paths["state_merge_diagnostics"],
                            STATE_MERGE_DIAGNOSTIC_FIELDS,
                        ),
                    )
        for stream_pos, row in enumerate(records):
            test_count = int(stream_pos) + 1
            fields = _phase3e_fields_from_index_row(row)
            synthetic_row = _phase3e_synthetic_row_from_index_row(row, stream_pos)
            z = _phase3e_target_from_index_row(row, node_embeddings, action_embeddings)
            z_hat = sspm.predict(fields)
            context_snapshot = dict(getattr(sspm, "last_context_snapshot", {}))
            count_snapshot = _node_count_snapshot(
                node_feature_counts,
                int(fields["info_src"]),
                int(fields["info_dst"]),
            )
            residual_features = _residual_feature_fields(
                z_hat,
                z,
                context_snapshot,
                count_snapshot,
            )
            residual_score = sspm.residual_score(
                z_hat,
                z,
                action=fields.get("raw_action", fields.get("action")),
            )
            residual_tail = _empirical_tail_score_compact(residual_score, residual_sorted)
            event_score = _event_score_from_residual_tail(
                residual_tail,
                config.event_score_mode,
            )
            test_score_summary.observe(float(event_score))
            current_threshold = (
                threshold_controller.threshold if threshold_controller is not None else threshold
            )
            alerted = bool(event_score >= current_threshold)
            score_time_merge_snapshot = sspm.state_merge_snapshot(fields)
            pending_alert: dict[str, Any] | None = None
            alert_node_ids: set[int] = set()
            if alerted:
                pending_alert = _raw_event_alert(
                    synthetic_row,
                    fields,
                    stream_pos,
                    event_score,
                    current_threshold,
                )
                pending_alert["threshold_basis"] = f"validation_{config.event_threshold_mode}"
                pending_alert.update(
                    {
                        "residual_tail": residual_tail,
                        "residual_score": residual_score,
                        **residual_features,
                    },
                )
                event_alert_count += 1
                alert_node_ids = {int(fields["info_src"]), int(fields["info_dst"])}
            if threshold_controller is not None:
                threshold_controller.observe(alerted=alerted)
            sspm.update_states(
                fields,
                z,
                residual_score=residual_score,
                enable_update_gate=True,
            )
            _update_node_count_state(
                node_feature_counts,
                int(fields["info_src"]),
                int(fields["info_dst"]),
            )
            merge_metadata = sspm.last_state_merge_metadata
            if pending_alert is not None:
                pending_alert.update(
                    _ofsm_alert_fields(
                        score_time_merge_snapshot,
                        merge_metadata,
                        str(config.sspm_state_merge_mode),
                    ),
                )
                if event_writer is not None:
                    event_writer.write_row(pending_alert)
                if event_alerts_raw is not None:
                    event_alerts_raw.append(pending_alert)
                for node_id in alert_node_ids:
                    _add_node_pool_alert(node_pool, node_id, pending_alert, config)
            if state_merge_diagnostics_writer is not None:
                for action in merge_metadata.get("actions", []):
                    if state_merge_diagnostics_rows >= int(
                        config.sspm_state_merge_diagnostics_max_rows,
                    ):
                        state_merge_diagnostics_truncated = True
                        break
                    raw_action = dict(action)
                    raw_action["event_idx"] = int(test_count)
                    raw_action["raw_action"] = str(fields.get("raw_action", fields.get("action", "")))
                    state_merge_diagnostics_writer.write_row(raw_action)
                    state_merge_diagnostics_rows += 1
            if sspm_profiler is not None:
                sspm_profiler.observe_gates(
                    lambda_t=float(getattr(sspm, "last_lambda_t", 0.0)),
                    rho_t=float(getattr(sspm, "last_rho_t", 0.0)),
                    q_t=float(getattr(sspm, "last_q_t", 1.0)),
                )
            for node_id in alert_node_ids:
                sspm.mark_risk_node(int(node_id))
            should_checkpoint = (
                stream_pos == 0
                or (stream_pos + 1) % checkpoint_interval == 0
            )
            if should_checkpoint:
                checkpoint = _node_checkpoint(stream_pos, node_pool)
                if checkpoint_writer is not None:
                    checkpoint_writer.write_row(checkpoint)
                if checkpoints_raw is not None:
                    checkpoints_raw.append(checkpoint)
                last_checkpoint_written_stream_pos = int(checkpoint["stream_pos"])
                checkpoint_count += 1
            if progress_interval > 0 and test_count % progress_interval == 0:
                current_rss_mb = _current_rss_mb()
                test_phase_peak_rss_mb = max(test_phase_peak_rss_mb, current_rss_mb)
                _stage_log(
                    config,
                    "phase3e_test_scoring_progress",
                    count=test_count,
                    event_alerts=event_alert_count,
                    node_pool=len(node_pool),
                )
                _emit_profile_row(
                    profile_writer,
                    _profiling_row(
                        "test_scoring",
                        test_count,
                        scoring_started,
                        event_alert_count,
                        len(node_pool),
                        sspm,
                        current_rss_mb=current_rss_mb,
                        test_phase_peak_rss_mb=test_phase_peak_rss_mb,
                    ),
                    verbose=config.verbose,
                )
                if sspm_profiler is not None:
                    sspm_profiler.write_row(
                        event_idx=test_count,
                        state_memory_stats=sspm.state_memory_stats(),
                        model_config={
                            **asdict(sspm.config),
                            "context_dim": int(sspm.context_dim),
                            "calibration": _model_calibration_summary(config, sspm),
                            "update_gate": _model_update_gate_summary(config, sspm),
                        },
                        events_per_sec=float(
                            test_count / max(time.perf_counter() - scoring_started, 1e-9),
                        ),
                    )
                if state_merge_profile_writer is not None:
                    state_merge_profile_writer.write_row(
                        _state_merge_profile_row(sspm, test_count, scoring_started),
                    )
        final_stream_pos = test_count - 1
        if (
            final_stream_pos >= 0
            and (
                last_checkpoint_written_stream_pos is None
                or last_checkpoint_written_stream_pos != final_stream_pos
            )
        ):
            final_checkpoint = _node_checkpoint(final_stream_pos, node_pool)
            if checkpoint_writer is not None:
                checkpoint_writer.write_row(final_checkpoint)
            if checkpoints_raw is not None:
                checkpoints_raw.append(final_checkpoint)
            checkpoint_count += 1
        end_current_rss_mb = _current_rss_mb()
        test_phase_peak_rss_mb = max(test_phase_peak_rss_mb, end_current_rss_mb)
        _emit_profile_row(
            profile_writer,
            _profiling_row(
                "test_scoring_end",
                test_count,
                scoring_started,
                event_alert_count,
                len(node_pool),
                sspm,
                current_rss_mb=end_current_rss_mb,
                test_phase_peak_rss_mb=test_phase_peak_rss_mb,
            ),
            verbose=config.verbose,
        )
        if sspm_profiler is not None:
            sspm_profiler.write_row(
                event_idx=test_count,
                state_memory_stats=sspm.state_memory_stats(),
                model_config={
                    **asdict(sspm.config),
                    "context_dim": int(sspm.context_dim),
                    "calibration": _model_calibration_summary(config, sspm),
                    "update_gate": _model_update_gate_summary(config, sspm),
                },
                events_per_sec=float(
                    test_count / max(time.perf_counter() - scoring_started, 1e-9),
                ),
            )
        if state_merge_profile_writer is not None:
            state_merge_profile_writer.write_row(
                _state_merge_profile_row(sspm, test_count, scoring_started),
            )
        _stage_log(
            config,
            "phase3e_test_scoring_end",
            count=test_count,
            event_alerts=event_alert_count,
            node_pool=len(node_pool),
        )
    stream_csv_write_seconds = _stream_csv_write_seconds(
        event_writer,
        checkpoint_writer,
        profile_writer,
        state_merge_profile_writer,
        state_merge_diagnostics_writer,
    )
    final_nodes_raw = _final_node_pool(node_pool, config)
    test_phase_peak_rss_mb = max(test_phase_peak_rss_mb, _current_rss_mb())
    test_scoring_seconds = float(time.perf_counter() - scoring_started)
    stream_outputs: dict[str, Any] = {
        "test_count": int(test_count),
        "final_nodes_raw": final_nodes_raw,
        "threshold": threshold,
        "test_score_summary": test_score_summary.summary(event_alert_count=event_alert_count),
        "raw_outputs_streamed": output_dir is not None,
        "output_dir": str(output_dir) if output_dir is not None else "",
        "raw_output_paths": {key: str(path) for key, path in raw_paths.items()},
        "event_score_trace_csv": str(raw_paths.get("event_score_trace", "")),
        "event_alert_count": int(event_alert_count),
        "checkpoint_count": int(checkpoint_count),
        "residual_semantic_examples": {group: [] for group in RESIDUAL_EXAMPLE_GROUPS},
        "test_scoring_seconds": test_scoring_seconds,
        "stream_csv_write_seconds": stream_csv_write_seconds,
        "threshold_controller": (
            threshold_controller.summary()
            if threshold_controller is not None
            else {
                "mode": str(config.event_threshold_mode),
                "initial_threshold": float(threshold),
                "final_threshold": float(threshold),
            }
        ),
        "compat_in_memory_outputs_active": bool(collect_in_memory),
        "raw_alerts_written": bool(_write_raw_alerts_enabled(config)),
        "analysis_outputs_written": bool(_write_analysis_outputs_enabled(config)),
        "state_merge_diagnostics_written": bool(
            _write_state_merge_diagnostics_enabled(config),
        ),
        "rss_test_peak_mb": float(test_phase_peak_rss_mb),
        "process_peak_rss_mb": _safe_peak_rss_mb(),
        "state_merge_diagnostics_truncated": bool(state_merge_diagnostics_truncated),
    }
    if event_alerts_raw is not None:
        stream_outputs.update(
            {
                "event_alerts_raw": event_alerts_raw or [],
            },
        )
    if checkpoints_raw is not None:
        stream_outputs.update(
            {
                "checkpoints_raw": checkpoints_raw or [],
            },
        )
    return stream_outputs


def _phase3e_load_idx_to_node_id(paths: Mapping[str, Path]) -> dict[int, int]:
    """Return Phase3E compact node index to DB node id mapping for eval labels."""
    raw_id_map_path = paths.get("node_id_to_idx")
    id_map_path = Path(raw_id_map_path) if raw_id_map_path is not None else Path()
    if raw_id_map_path is None or not id_map_path.is_file():
        id_map_path = Path(paths["node_embeddings"]).with_name("node_id_to_idx.pkl")
    with id_map_path.open("rb") as handle:
        node_id_to_idx = pickle.load(handle)
    if not isinstance(node_id_to_idx, Mapping):
        raise TypeError("Phase3E node_id_to_idx.pkl must contain a mapping")
    idx_to_node: dict[int, int] = {}
    for raw_node_id, raw_idx in node_id_to_idx.items():
        node_id = int(raw_node_id)
        node_idx = int(raw_idx)
        if node_idx in idx_to_node:
            raise ValueError(f"duplicate Phase3E node_idx: {node_idx}")
        idx_to_node[node_idx] = node_id
    return idx_to_node


def _phase3g_load_netflow_endpoint_by_idx(
    config: SlimConfig,
    paths: Mapping[str, Path],
) -> dict[int, str]:
    """Return compact netflow node index to endpoint signature for online suppression."""
    idx_to_node = _phase3e_load_idx_to_node_id(paths)
    db_cfg = _cfg_for_dataset(config.dataset)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    cur = None
    conn = None
    endpoint_by_idx: dict[int, str] = {}
    try:
        cur, conn = init_database_connection(db_cfg)
        node_ids = list(idx_to_node.values())
        idx_by_node = {int(node_id): int(idx) for idx, node_id in idx_to_node.items()}
        batch_size = max(int(config.phase3e_node_lookup_batch_size), 1)
        for offset in range(0, len(node_ids), batch_size):
            batch = node_ids[offset : offset + batch_size]
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
                node_idx = idx_by_node.get(node_id)
                if node_idx is None:
                    continue
                endpoint_by_idx[node_idx] = _conditional_endpoint_signature(
                    {
                        "src_addr": src_addr,
                        "src_port": src_port,
                        "dst_addr": dst_addr,
                        "dst_port": dst_port,
                    },
                    node_idx,
                )
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
    return endpoint_by_idx


def _phase3g_action_input_dim(node_embeddings: np.ndarray) -> int:
    type_dim = len(ENTITY_TYPES)
    return int(node_embeddings.shape[1]) * 2 + type_dim * 2


def _phase3g_resolved_action_checkpoint_path(config: SlimConfig) -> Path:
    configured = str(config.action_head_checkpoint_path).strip()
    if configured:
        return Path(configured)
    root = Path(config.sspm_base_checkpoint_root)
    return root / f"{config.dataset}_E2_PHASE3G_ACTION_HEAD.pkl"


def _phase3g_resolved_head_checkpoint_path(config: SlimConfig) -> Path:
    configured = str(config.action_head_checkpoint_path).strip()
    if configured:
        return Path(configured)
    root = Path(config.sspm_base_checkpoint_root)
    if str(config.sspm_score_head) == "conditional_action_semantic":
        if str(config.sspm_conditional_head_arch) == CONDITIONAL_HEAD_ARCH_DUAL_V2:
            return (
                Path("outputs/models/phase3g_action_heads_v2")
                / f"{config.dataset}_E2_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl"
            )
        return root / f"{config.dataset}_E2_PHASE3G_CONDITIONAL_HEAD.pkl"
    if str(config.sspm_score_head) == CONDITIONAL_ACTION_EMBEDDING_SCORE_HEAD:
        return root / f"{config.dataset}_E2_PHASE3G_CONDITIONAL_ACTION_EMBEDDING_HEAD.pkl"
    return root / f"{config.dataset}_E2_PHASE3G_ACTION_HEAD.pkl"


def _phase3g_action_cache_dir(config: SlimConfig, fingerprint: Mapping[str, Any]) -> Path:
    root = _phase3e_resolve_cache_dir(config.action_validation_cache_dir, config)
    cache_id = stable_json_hash(dict(fingerprint))[:16]
    return Path(root) / cache_id


def _phase3g_run_only_from_out_tag(out_tag: str) -> str:
    """Return the Phase3E ablation run key encoded in an output tag."""
    value = str(out_tag)
    for run_key in (
        "E2_OFSM_TIME_DOMAIN_T098",
        "E2_OFSM_RANDOM",
        "E2_OFSM_FOURIER_T085",
        "E2_OFSM_FOURIER_T090",
        "E2_OFSM_FOURIER_T095",
        "E2_OFSM_FOURIER_T098",
        "E3_NONE",
        "E3_OFSM_FOURIER_T098",
        "E4_OFSM_FOURIER_T098",
        "E5_OFSM_FOURIER_T098",
        "E2_NONE",
        "E4_NONE",
        "E5_NONE",
    ):
        if value.endswith(run_key):
            return run_key
    return value


def _phase3g_file_fingerprint(path: str | Path) -> dict[str, Any]:
    return _phase3e_file_fingerprint(path)



def _phase3g_conditional_fingerprint(
    *,
    config: SlimConfig,
    head: ConditionalSemanticHead,
    head_metadata: Mapping[str, Any] | None = None,
    paths: Mapping[str, Path],
    event_meta: Mapping[str, Any],
    event_index: np.ndarray,
    split: str,
    endpoint_suppression_cache_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    split_meta = dict(dict(event_meta.get("splits", {})).get(split, {}))
    metadata = dict(head_metadata or {})
    head_checkpoint_fingerprint = dict(metadata.get("fingerprint", head.fingerprint()))
    checkpoint_schema = str(
        metadata.get(
            "checkpoint_schema",
            head_checkpoint_fingerprint.get("schema", head.fingerprint().get("schema", "")),
        ),
    )
    fingerprint = {
        "schema": "phase3g_conditional_validation_cache_v1",
        "score_head": str(config.sspm_score_head),
        "conditional_head_fingerprint": head_checkpoint_fingerprint.get("fingerprint_sha256"),
        "conditional_head_arch": str(config.sspm_conditional_head_arch),
        "head_arch": str(config.sspm_conditional_head_arch),
        "checkpoint_schema": checkpoint_schema,
        "node_repr_fusion": str(config.node_repr_fusion),
        "loss_type": str(config.conditional_semantic_loss),
        "score_target_mode": str(config.sspm_score_target_mode),
        "target_case_mode": "conditional_cold_action_else_event",
        "target_case_head_mode": str(head.config.target_case_head_mode),
        "target_case_fingerprint": dict(
            head_checkpoint_fingerprint.get(
                "target_case_fingerprint",
                metadata.get("target_case_fingerprint", {}),
            ),
        ),
        "target_case_logic": "state_exists_event_semantic_else_both_cold_action",
        "dataset": str(config.dataset),
        "split": str(split),
        "event_index_fingerprint": event_index_fingerprint(_phase3e_event_index_array(event_index)),
        "source_split_event_index_fingerprint": split_meta.get("event_index_fingerprint")
        or split_meta.get("fingerprint"),
        "node_embedding_fingerprint": _phase3g_file_fingerprint(paths["node_embeddings"]),
        "action_embedding_fingerprint": _phase3g_file_fingerprint(paths["action_embeddings"]),
        "state_model": str(config.sspm_state_model),
        "state_merge_mode": str(config.sspm_state_merge_mode),
        "state_merge_threshold": float(config.sspm_state_merge_threshold),
        "threshold_mode": str(config.event_threshold_mode),
        "threshold_quantile": float(config.event_threshold_quantile),
        "endpoint_suppression_mode": str(config.conditional_endpoint_suppression_mode),
        "conditional_group_min_count": int(config.conditional_group_min_count),
        "conditional_low_support_policy": str(config.conditional_low_support_policy),
        "conditional_low_support_margin": float(config.conditional_low_support_margin),
        "conditional_unseen_group_policy": str(config.conditional_unseen_group_policy),
        "conditional_global_extreme_quantile": float(
            config.conditional_global_extreme_quantile,
        ),
        "conditional_adaptive_margin_n1": int(config.conditional_adaptive_margin_n1),
        "conditional_adaptive_margin_n2": int(config.conditional_adaptive_margin_n2),
        "conditional_adaptive_margin_low": float(config.conditional_adaptive_margin_low),
        "conditional_adaptive_margin_mid": float(config.conditional_adaptive_margin_mid),
        "conditional_adaptive_margin_high": float(config.conditional_adaptive_margin_high),
        "input_dim": int(head.config.input_dim),
        "context_dim": int(head.config.input_dim),
        "rank": int(head.config.rank),
        "output_dim": int(head.config.output_dim),
    }
    if bool(config.conditional_endpoint_aware_suppression):
        fingerprint["endpoint_suppression_cache_fingerprint"] = dict(
            endpoint_suppression_cache_fingerprint or {},
        )
    return fingerprint


def _phase3g_conditional_arch_metrics(
    config: SlimConfig,
    head_metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Return compact conditional-head architecture metadata for reports."""
    metadata = dict(head_metadata or {})
    fingerprint = dict(metadata.get("fingerprint", {}))
    arch = str(
        metadata.get(
            "conditional_head_arch",
            metadata.get("head_arch", config.sspm_conditional_head_arch),
        ),
    )
    return {
        "conditional_head_arch": arch,
        "head_arch": arch,
        "checkpoint_schema": str(
            metadata.get("checkpoint_schema", fingerprint.get("schema", "")),
        ),
        "target_case_head_mode": str(metadata.get("target_case_head_mode", "")),
    }



def _phase3g_type_eye() -> np.ndarray:
    return np.eye(len(ENTITY_TYPES), dtype=np.float32)


def _phase3g_state_arrays(num_nodes: int, state_dim: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((int(num_nodes), int(state_dim)), dtype=np.float32),
        np.zeros((int(num_nodes),), dtype=np.bool_),
    )


def _phase3g_node_has_active_state(model: SSPMLowRankModel, node_id: int) -> bool:
    if model._uses_probationary_memory():
        return int(node_id) in model.memory.node_to_slot
    return int(node_id) in model.node_to_slot


def _phase3g_action_context_from_model(
    model: SSPMLowRankModel,
    row: np.void,
    node_embeddings: np.ndarray,
    type_one_hot: np.ndarray,
) -> tuple[np.ndarray, str, str, bool, bool]:
    src_idx = int(row["src_node_idx"])
    dst_idx = int(row["dst_node_idx"])
    src_type_id = int(row["src_type_id"])
    dst_type_id = int(row["dst_type_id"])
    src_type_name = _phase3e_entity_type_name("src_type_id", src_type_id)
    dst_type_name = _phase3e_entity_type_name("dst_type_id", dst_type_id)
    src_has_state = _phase3g_node_has_active_state(model, src_idx)
    dst_has_state = _phase3g_node_has_active_state(model, dst_idx)
    src_state = model.state_model.read(model.get_context_state(src_idx, src_type_name))
    dst_state = model.state_model.read(model.get_context_state(dst_idx, dst_type_name))
    src_static = np.asarray(node_embeddings[src_idx], dtype=np.float32)
    dst_static = np.asarray(node_embeddings[dst_idx], dtype=np.float32)
    src_repr = (src_state + src_static) / np.float32(2.0) if src_has_state else src_static
    dst_repr = (dst_state + dst_static) / np.float32(2.0) if dst_has_state else dst_static
    context = np.concatenate(
        [
            src_repr.astype(np.float32, copy=False),
            dst_repr.astype(np.float32, copy=False),
            type_one_hot[src_type_id],
            type_one_hot[dst_type_id],
        ],
        axis=0,
    )
    return context.astype(np.float32, copy=False), src_type_name, dst_type_name, src_has_state, dst_has_state


def _conditional_endpoint_pair_key(row: np.void) -> tuple[int, int, int]:
    return (int(row["src_node_idx"]), int(row["dst_node_idx"]), int(row["action_id"]))


def _conditional_endpoint_signature(meta: Mapping[str, Any] | None, fallback_idx: int) -> str:
    if meta:
        dst_addr = str(meta.get("dst_addr", "") or "").strip()
        dst_port = str(meta.get("dst_port", "") or "").strip()
        src_addr = str(meta.get("src_addr", "") or "").strip()
        src_port = str(meta.get("src_port", "") or "").strip()
        if dst_addr and dst_port:
            return f"{dst_addr}:{dst_port}"
        if dst_addr:
            return f"{dst_addr}:"
        if src_addr and src_port:
            return f"{src_addr}:{src_port}"
        if src_addr:
            return f"{src_addr}:"
    return f"node:{int(fallback_idx)}"


def _conditional_endpoint_pair_signature(
    row: np.void,
    netflow_endpoint_by_idx: Mapping[int, str] | None,
) -> tuple[int, str, int]:
    endpoint_idx = int(row["dst_node_idx"])
    return (
        int(row["src_node_idx"]),
        str((netflow_endpoint_by_idx or {}).get(endpoint_idx, f"node:{endpoint_idx}")),
        int(row["action_id"]),
    )


def _conditional_endpoint_action_signature(
    row: np.void,
    netflow_endpoint_by_idx: Mapping[int, str] | None,
) -> tuple[str, int]:
    endpoint_idx = int(row["dst_node_idx"])
    return (
        str((netflow_endpoint_by_idx or {}).get(endpoint_idx, f"node:{endpoint_idx}")),
        int(row["action_id"]),
    )


def _conditional_src_endpoint_signature(
    row: np.void,
    netflow_endpoint_by_idx: Mapping[int, str] | None,
) -> tuple[int, str]:
    endpoint_idx = int(row["dst_node_idx"])
    return (
        int(row["src_node_idx"]),
        str((netflow_endpoint_by_idx or {}).get(endpoint_idx, f"node:{endpoint_idx}")),
    )


def _conditional_endpoint_pair_signature_text(pair_key: tuple[int, str, int]) -> str:
    return f"{int(pair_key[0])}:{pair_key[1]}:{int(pair_key[2])}"


def _conditional_src_endpoint_signature_text(src_endpoint_key: tuple[int, str]) -> str:
    return f"{int(src_endpoint_key[0])}:{src_endpoint_key[1]}"


def _endpoint_hash(endpoint: str) -> int:
    encoded = str(endpoint).encode("utf-8", errors="surrogatepass")
    return int(zlib.crc32(encoded) & 0xFFFFFFFF)


def _endpoint_pair_compact_key(src_node_idx: int, endpoint_hash: int, action_id: int) -> int:
    src = int(src_node_idx) & 0xFFFFFFFF
    endpoint = int(endpoint_hash) & 0xFFFFFFFF
    action = int(action_id) & 0xFF
    return int((src << 40) | (endpoint << 8) | action)


def _endpoint_action_compact_key(endpoint_hash: int, action_id: int) -> int:
    endpoint = int(endpoint_hash) & 0xFFFFFFFF
    action = int(action_id) & 0xFF
    return int((endpoint << 8) | action)


def _endpoint_src_compact_key(src_node_idx: int, endpoint_hash: int) -> int:
    src = int(src_node_idx) & 0xFFFFFFFF
    endpoint = int(endpoint_hash) & 0xFFFFFFFF
    return int((src << 32) | endpoint)


class LazyMmapLruEmbeddingLookup:
    """Float32 mmap-backed node embedding lookup with a bounded LRU row cache."""

    def __init__(self, path: str | Path, *, max_nodes: int, backing: str) -> None:
        if int(max_nodes) <= 0:
            raise ValueError("NODE_EMBEDDING_CACHE_MAX_NODES must be positive")
        self.path = str(path)
        self.backing = str(backing)
        self.max_nodes = int(max_nodes)
        self._mmap = np.load(path, mmap_mode="r")
        if self._mmap.ndim != 2:
            raise ValueError(f"node embedding mmap must be 2-D: {path}")
        self.shape = self._mmap.shape
        self.dtype = self._mmap.dtype
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.lookup_seconds = 0.0

    def __getitem__(self, index: Any) -> np.ndarray:
        if isinstance(index, (int, np.integer)):
            return self.get(int(index))
        started = time.perf_counter()
        try:
            return np.asarray(self._mmap[index], dtype=np.float32)
        finally:
            self.lookup_seconds += float(time.perf_counter() - started)

    def __len__(self) -> int:
        return int(self.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self._mmap.nbytes)

    def get(self, index: int) -> np.ndarray:
        started = time.perf_counter()
        try:
            cached = self._cache.get(int(index))
            if cached is not None:
                self.hit_count += 1
                self._cache.move_to_end(int(index))
                return cached
            self.miss_count += 1
            value = np.asarray(self._mmap[int(index)], dtype=np.float32).copy()
            self._cache[int(index)] = value
            if len(self._cache) > self.max_nodes:
                self._cache.popitem(last=False)
            return value
        finally:
            self.lookup_seconds += float(time.perf_counter() - started)

    def metrics(self) -> dict[str, Any]:
        total = int(self.hit_count + self.miss_count)
        active_nodes = int(len(self._cache))
        row_bytes = int(self.shape[1]) * int(np.dtype(self.dtype).itemsize)
        cache_actual_mb = float(active_nodes * row_bytes / 1024.0 / 1024.0)
        cache_capacity_mb = float(int(self.max_nodes) * row_bytes / 1024.0 / 1024.0)
        return {
            "embedding_lookup_mode": "lazy_mmap_lru",
            "embedding_lazy_backing": self.backing,
            "embedding_cache_max_nodes": int(self.max_nodes),
            "embedding_cache_active_nodes": active_nodes,
            "embedding_lru_cache_mb": cache_actual_mb,
            "embedding_lru_cache_mb_actual": cache_actual_mb,
            "embedding_lru_cache_mb_capacity": cache_capacity_mb,
            "embedding_cache_hit_count": int(self.hit_count),
            "embedding_cache_miss_count": int(self.miss_count),
            "embedding_cache_hit_rate": float(self.hit_count / max(total, 1)),
            "embedding_cache_miss_rate": float(self.miss_count / max(total, 1)),
            "embedding_lookup_seconds": float(self.lookup_seconds),
            "embedding_backing_path": self.path,
        }


@dataclass
class CompactEndpointSuppressionLookup:
    pair_keys: np.ndarray
    pair_counts: np.ndarray
    src_endpoint_keys: np.ndarray
    src_endpoint_counts: np.ndarray
    endpoint_action_keys: np.ndarray
    endpoint_action_counts: np.ndarray
    netflow_idx_keys: np.ndarray
    netflow_endpoint_hashes: np.ndarray

    def endpoint_hash_for_node(self, node_idx: int) -> int:
        if self.netflow_idx_keys.size == 0:
            return _endpoint_hash(f"node:{int(node_idx)}")
        pos = int(np.searchsorted(self.netflow_idx_keys, np.uint64(int(node_idx))))
        if pos < int(self.netflow_idx_keys.size) and int(self.netflow_idx_keys[pos]) == int(node_idx):
            return int(self.netflow_endpoint_hashes[pos])
        return _endpoint_hash(f"node:{int(node_idx)}")

    def pair_count(self, src_node_idx: int, dst_node_idx: int, action_id: int) -> int:
        endpoint_hash = self.endpoint_hash_for_node(dst_node_idx)
        key = np.uint64(_endpoint_pair_compact_key(src_node_idx, endpoint_hash, action_id))
        pos = int(np.searchsorted(self.pair_keys, key))
        if pos < int(self.pair_keys.size) and int(self.pair_keys[pos]) == int(key):
            return int(self.pair_counts[pos])
        return 0

    def src_endpoint_count(self, src_node_idx: int, dst_node_idx: int) -> int:
        endpoint_hash = self.endpoint_hash_for_node(dst_node_idx)
        key = np.uint64(_endpoint_src_compact_key(src_node_idx, endpoint_hash))
        pos = int(np.searchsorted(self.src_endpoint_keys, key))
        if (
            pos < int(self.src_endpoint_keys.size)
            and int(self.src_endpoint_keys[pos]) == int(key)
        ):
            return int(self.src_endpoint_counts[pos])
        return 0

    def endpoint_count(self, dst_node_idx: int, action_id: int) -> int:
        endpoint_hash = self.endpoint_hash_for_node(dst_node_idx)
        key = np.uint64(_endpoint_action_compact_key(endpoint_hash, action_id))
        pos = int(np.searchsorted(self.endpoint_action_keys, key))
        if (
            pos < int(self.endpoint_action_keys.size)
            and int(self.endpoint_action_keys[pos]) == int(key)
        ):
            return int(self.endpoint_action_counts[pos])
        return 0


def _conditional_endpoint_is_suppression_candidate(
    row: np.void,
    *,
    include_read: bool = False,
) -> bool:
    action = str(ORTHRUS10_ACTION_NAMES[int(row["action_id"])])
    src_type = int(row["src_type_id"])
    dst_type = int(row["dst_type_id"])
    if action == "EVENT_WRITE" and src_type == ENTITY_TYPES["process"] and dst_type == ENTITY_TYPES["netflow"]:
        return True
    if (
        include_read
        and action == "EVENT_READ"
        and src_type == ENTITY_TYPES["netflow"]
        and dst_type == ENTITY_TYPES["process"]
    ):
        return True
    return False


def _conditional_endpoint_pair_counts(
    event_indices: Sequence[np.ndarray],
    *,
    netflow_endpoint_by_idx: Mapping[int, str] | None = None,
    include_read: bool = False,
) -> Counter[tuple[int, str, int]]:
    counts: Counter[tuple[int, str, int]] = Counter()
    for event_index in event_indices:
        for row in _phase3e_event_index_array(event_index):
            if _conditional_endpoint_is_suppression_candidate(row, include_read=include_read):
                counts[_conditional_endpoint_pair_signature(row, netflow_endpoint_by_idx)] += 1
    return counts


def _conditional_endpoint_action_counts(
    event_indices: Sequence[np.ndarray],
    *,
    netflow_endpoint_by_idx: Mapping[int, str] | None = None,
    include_read: bool = False,
) -> Counter[tuple[str, int]]:
    counts: Counter[tuple[str, int]] = Counter()
    for event_index in event_indices:
        for row in _phase3e_event_index_array(event_index):
            if _conditional_endpoint_is_suppression_candidate(row, include_read=include_read):
                counts[_conditional_endpoint_action_signature(row, netflow_endpoint_by_idx)] += 1
    return counts


def _conditional_src_endpoint_counts(
    event_indices: Sequence[np.ndarray],
    *,
    netflow_endpoint_by_idx: Mapping[int, str] | None = None,
    include_read: bool = False,
) -> Counter[tuple[int, str]]:
    counts: Counter[tuple[int, str]] = Counter()
    for event_index in event_indices:
        for row in _phase3e_event_index_array(event_index):
            if _conditional_endpoint_is_suppression_candidate(row, include_read=include_read):
                counts[_conditional_src_endpoint_signature(row, netflow_endpoint_by_idx)] += 1
    return counts


def _conditional_endpoint_cache_dir(config: SlimConfig, fingerprint: Mapping[str, Any]) -> Path:
    root_template = str(config.conditional_endpoint_suppression_cache_dir)
    default_template = str(SlimConfig.conditional_endpoint_suppression_cache_dir)
    uses_compact_idx = str(config.node_embedding_lookup_mode) == "compact_used_nodes" or (
        str(config.node_embedding_lookup_mode) == "lazy_mmap_lru"
        and str(config.node_embedding_lazy_backing) == "compact_used_nodes"
    )
    if uses_compact_idx and root_template == default_template:
        root_template = "outputs/cache/phase3g_endpoint_suppression_compact/{DATASET}"
    root = root_template.replace(
        "{DATASET}",
        str(config.dataset),
    )
    return Path(root) / stable_json_hash(dict(fingerprint))[:16]


def _conditional_endpoint_cache_fingerprint(
    *,
    config: SlimConfig,
    paths: Mapping[str, Path],
    event_meta: Mapping[str, Any],
    train_index: np.ndarray,
    validation_index: np.ndarray,
) -> dict[str, Any]:
    train_meta = dict(dict(event_meta.get("splits", {})).get("train", {}))
    validation_meta = dict(dict(event_meta.get("splits", {})).get("validation", {}))
    node_map_path = Path(paths["node_embeddings"]).with_name("node_id_to_idx.pkl")
    compact_meta_path = paths.get("compact_embedding_meta")
    compact_meta = {}
    if compact_meta_path is not None and Path(compact_meta_path).exists():
        compact_meta = json.loads(Path(compact_meta_path).read_text(encoding="utf-8"))
    return {
        "schema": "phase3g_endpoint_suppression_cache_v2",
        "dataset": str(config.dataset),
        "node_embedding_lookup_mode": str(config.node_embedding_lookup_mode),
        "train_event_index_fingerprint": event_index_fingerprint(
            _phase3e_event_index_array(train_index),
        ),
        "validation_event_index_fingerprint": event_index_fingerprint(
            _phase3e_event_index_array(validation_index),
        ),
        "source_train_event_index_fingerprint": train_meta.get("event_index_fingerprint")
        or train_meta.get("fingerprint"),
        "source_validation_event_index_fingerprint": validation_meta.get(
            "event_index_fingerprint",
        )
        or validation_meta.get("fingerprint"),
        "node_id_to_idx_fingerprint": _phase3g_file_fingerprint(node_map_path),
        "node_embedding_fingerprint": _phase3g_file_fingerprint(paths["node_embeddings"]),
        "compact_remap_fingerprint": compact_meta.get("remap_fingerprint", ""),
        "compact_node_embedding_fingerprint": compact_meta.get(
            "compact_node_embedding_fingerprint",
            "",
        ),
        "compact_original_node_indices_fingerprint": compact_meta.get(
            "compact_original_node_indices_fingerprint",
            "",
        ),
        "action_embedding_fingerprint": _phase3g_file_fingerprint(paths["action_embeddings"]),
        "action_vocab": list(ORTHRUS10_ACTION_NAMES),
        "action_vocab_fingerprint": stable_json_hash(list(ORTHRUS10_ACTION_NAMES)),
        "type_vocab": dict(sorted(ENTITY_TYPES.items())),
        "type_vocab_fingerprint": stable_json_hash(dict(sorted(ENTITY_TYPES.items()))),
        "operation_filter": "ORTHRUS10",
        "endpoint_key_schema": "netflow_dst_addr_port_or_node_idx",
        "pair_key_schema": "uint64(src_node_idx,crc32(endpoint_signature),action_id)",
        "src_endpoint_key_schema": "uint64(src_node_idx,crc32(endpoint_signature))",
        "endpoint_action_key_schema": "uint64(crc32(endpoint_signature),action_id)",
        "known_pair_min_count": int(config.conditional_known_pair_min_count),
        "pair_suppression_margin": float(config.conditional_pair_suppression_margin),
        "same_process_endpoint_min_count": int(
            config.conditional_same_process_endpoint_min_count,
        ),
        "same_process_endpoint_margin": float(
            config.conditional_same_process_endpoint_margin,
        ),
        "suppression_mode": str(config.conditional_endpoint_suppression_mode),
        "endpoint_suppression_enabled": bool(config.conditional_endpoint_aware_suppression),
        "include_read_direction": bool(config.conditional_endpoint_suppression_read),
    }


def _serialize_endpoint_pair_counts(
    counts: Mapping[tuple[int, str, int], int],
) -> dict[str, int]:
    return {
        json.dumps([int(key[0]), str(key[1]), int(key[2])], separators=(",", ":")): int(value)
        for key, value in counts.items()
    }


def _deserialize_endpoint_pair_counts(payload: Mapping[str, Any]) -> Counter[tuple[int, str, int]]:
    counts: Counter[tuple[int, str, int]] = Counter()
    for raw_key, raw_value in payload.items():
        parts = json.loads(str(raw_key))
        if not isinstance(parts, list) or len(parts) != 3:
            raise ValueError(f"invalid endpoint pair cache key: {raw_key}")
        counts[(int(parts[0]), str(parts[1]), int(parts[2]))] = int(raw_value)
    return counts


def _serialize_endpoint_action_counts(
    counts: Mapping[tuple[str, int], int],
) -> dict[str, int]:
    return {
        json.dumps([str(key[0]), int(key[1])], separators=(",", ":")): int(value)
        for key, value in counts.items()
    }


def _deserialize_endpoint_action_counts(payload: Mapping[str, Any]) -> Counter[tuple[str, int]]:
    counts: Counter[tuple[str, int]] = Counter()
    for raw_key, raw_value in payload.items():
        parts = json.loads(str(raw_key))
        if not isinstance(parts, list) or len(parts) != 2:
            raise ValueError(f"invalid endpoint action cache key: {raw_key}")
        counts[(str(parts[0]), int(parts[1]))] = int(raw_value)
    return counts


def _serialize_src_endpoint_counts(counts: Mapping[tuple[int, str], int]) -> dict[str, int]:
    return {
        json.dumps([int(key[0]), str(key[1])], separators=(",", ":")): int(value)
        for key, value in counts.items()
    }


def _deserialize_src_endpoint_counts(payload: Mapping[str, Any]) -> Counter[tuple[int, str]]:
    counts: Counter[tuple[int, str]] = Counter()
    for raw_key, raw_value in payload.items():
        parts = json.loads(str(raw_key))
        if not isinstance(parts, list) or len(parts) != 2:
            raise ValueError(f"invalid src endpoint cache key: {raw_key}")
        counts[(int(parts[0]), str(parts[1]))] = int(raw_value)
    return counts


def _write_conditional_endpoint_suppression_cache(
    cache_dir: Path,
    *,
    fingerprint: Mapping[str, Any],
    pair_counts: Mapping[tuple[int, str, int], int],
    endpoint_counts: Mapping[tuple[str, int], int],
    src_endpoint_counts: Mapping[tuple[int, str], int] | None = None,
    netflow_endpoint_by_idx: Mapping[int, str],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pair_keys_path = cache_dir / "endpoint_pair_keys.npy"
    pair_counts_path = cache_dir / "endpoint_pair_counts.npy"
    src_endpoint_keys_path = cache_dir / "endpoint_src_endpoint_keys.npy"
    src_endpoint_counts_path = cache_dir / "endpoint_src_endpoint_counts.npy"
    endpoint_keys_path = cache_dir / "endpoint_action_keys.npy"
    endpoint_counts_path = cache_dir / "endpoint_action_counts.npy"
    netflow_idx_path = cache_dir / "endpoint_netflow_idx_keys.npy"
    netflow_hash_path = cache_dir / "endpoint_netflow_hashes.npy"
    meta_path = cache_dir / "endpoint_suppression_meta.json"
    endpoint_cache_meta_path = cache_dir / "endpoint_cache_meta.json"
    summary_path = cache_dir / "endpoint_suppression_summary.csv"

    compact_pairs = [
        (
            _endpoint_pair_compact_key(int(key[0]), _endpoint_hash(str(key[1])), int(key[2])),
            int(value),
        )
        for key, value in pair_counts.items()
    ]
    compact_pairs.sort(key=lambda item: item[0])
    src_endpoint_counts = src_endpoint_counts or {}
    compact_src_endpoints = [
        (_endpoint_src_compact_key(int(key[0]), _endpoint_hash(str(key[1]))), int(value))
        for key, value in src_endpoint_counts.items()
    ]
    compact_src_endpoints.sort(key=lambda item: item[0])
    compact_endpoints = [
        (_endpoint_action_compact_key(_endpoint_hash(str(key[0])), int(key[1])), int(value))
        for key, value in endpoint_counts.items()
    ]
    compact_endpoints.sort(key=lambda item: item[0])
    compact_netflows = [
        (int(node_idx), _endpoint_hash(str(endpoint)))
        for node_idx, endpoint in netflow_endpoint_by_idx.items()
    ]
    compact_netflows.sort(key=lambda item: item[0])

    pair_keys = np.array([key for key, _ in compact_pairs], dtype=np.uint64)
    pair_count_values = np.array([value for _, value in compact_pairs], dtype=np.int32)
    src_endpoint_keys = np.array([key for key, _ in compact_src_endpoints], dtype=np.uint64)
    src_endpoint_count_values = np.array(
        [value for _, value in compact_src_endpoints],
        dtype=np.int32,
    )
    endpoint_keys = np.array([key for key, _ in compact_endpoints], dtype=np.uint64)
    endpoint_count_values = np.array([value for _, value in compact_endpoints], dtype=np.int32)
    netflow_idx_keys = np.array([key for key, _ in compact_netflows], dtype=np.uint64)
    netflow_endpoint_hashes = np.array([value for _, value in compact_netflows], dtype=np.uint32)
    np.save(pair_keys_path, pair_keys)
    np.save(pair_counts_path, pair_count_values)
    np.save(src_endpoint_keys_path, src_endpoint_keys)
    np.save(src_endpoint_counts_path, src_endpoint_count_values)
    np.save(endpoint_keys_path, endpoint_keys)
    np.save(endpoint_counts_path, endpoint_count_values)
    np.save(netflow_idx_path, netflow_idx_keys)
    np.save(netflow_hash_path, netflow_endpoint_hashes)
    compact_cache_mb = float(
        sum(
            array.nbytes
            for array in (
                pair_keys,
                pair_count_values,
                src_endpoint_keys,
                src_endpoint_count_values,
                endpoint_keys,
                endpoint_count_values,
                netflow_idx_keys,
                netflow_endpoint_hashes,
            )
        )
        / 1024.0
        / 1024.0
    )
    meta = {
        "schema": "phase3g_endpoint_suppression_cache_v2",
        "compact_cache": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": dict(fingerprint),
        "fingerprint_sha256": stable_json_hash(dict(fingerprint)),
        "endpoint_pair_keys_path": str(pair_keys_path),
        "endpoint_pair_counts_path": str(pair_counts_path),
        "endpoint_src_endpoint_keys_path": str(src_endpoint_keys_path),
        "endpoint_src_endpoint_counts_path": str(src_endpoint_counts_path),
        "endpoint_action_keys_path": str(endpoint_keys_path),
        "endpoint_action_counts_path": str(endpoint_counts_path),
        "endpoint_netflow_idx_keys_path": str(netflow_idx_path),
        "endpoint_netflow_hashes_path": str(netflow_hash_path),
        "summary_path": str(summary_path),
        "process_endpoint_action_count": int(len(pair_counts)),
        "process_endpoint_count": int(len(src_endpoint_counts)),
        "endpoint_action_count": int(len(endpoint_counts)),
        "pair_seen_count": int(sum(int(value) for value in pair_counts.values())),
        "src_endpoint_seen_count": int(
            sum(int(value) for value in src_endpoint_counts.values()),
        ),
        "endpoint_seen_count": int(sum(int(value) for value in endpoint_counts.values())),
        "known_pair_count": int(len(pair_counts)),
        "known_src_endpoint_count": int(len(src_endpoint_counts)),
        "known_endpoint_count": int(len(endpoint_counts)),
        "netflow_endpoint_map_size": int(len(netflow_endpoint_by_idx)),
        "endpoint_cache_mb": compact_cache_mb,
        "summary": dict(summary),
    }
    meta_text = json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n"
    meta_path.write_text(meta_text)
    endpoint_cache_meta_path.write_text(meta_text)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "metric",
            "value",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(meta):
            if key in {"fingerprint", "summary"}:
                continue
            writer.writerow({"metric": key, "value": meta[key]})
        for key, value in sorted(dict(summary).items()):
            writer.writerow({"metric": f"summary.{key}", "value": value})
    return meta


def _load_conditional_endpoint_suppression_cache(
    cache_dir: Path,
    *,
    expected_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "endpoint_suppression_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(str(meta_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_hash = stable_json_hash(dict(expected_fingerprint))
    actual_hash = str(meta.get("fingerprint_sha256", ""))
    if actual_hash != expected_hash:
        raise ValueError(
            "Phase3G endpoint suppression cache fingerprint mismatch: "
            f"expected={expected_hash} actual={actual_hash}",
        )
    if bool(meta.get("compact_cache", False)):
        pair_keys = np.load(meta["endpoint_pair_keys_path"], mmap_mode="r")
        pair_counts = np.load(meta["endpoint_pair_counts_path"], mmap_mode="r")
        src_endpoint_keys = np.load(meta["endpoint_src_endpoint_keys_path"], mmap_mode="r")
        src_endpoint_counts = np.load(
            meta["endpoint_src_endpoint_counts_path"],
            mmap_mode="r",
        )
        endpoint_keys = np.load(meta["endpoint_action_keys_path"], mmap_mode="r")
        endpoint_counts = np.load(meta["endpoint_action_counts_path"], mmap_mode="r")
        netflow_idx_keys = np.load(meta["endpoint_netflow_idx_keys_path"], mmap_mode="r")
        netflow_hashes = np.load(meta["endpoint_netflow_hashes_path"], mmap_mode="r")
        return {
            "meta": meta,
            "endpoint_pair_keys": pair_keys,
            "endpoint_pair_counts": pair_counts,
            "endpoint_src_endpoint_keys": src_endpoint_keys,
            "endpoint_src_endpoint_counts": src_endpoint_counts,
            "endpoint_action_keys": endpoint_keys,
            "endpoint_action_counts": endpoint_counts,
            "endpoint_netflow_idx_keys": netflow_idx_keys,
            "endpoint_netflow_hashes": netflow_hashes,
            "endpoint_lookup": CompactEndpointSuppressionLookup(
                pair_keys=pair_keys,
                pair_counts=pair_counts,
                src_endpoint_keys=src_endpoint_keys,
                src_endpoint_counts=src_endpoint_counts,
                endpoint_action_keys=endpoint_keys,
                endpoint_action_counts=endpoint_counts,
                netflow_idx_keys=netflow_idx_keys,
                netflow_endpoint_hashes=netflow_hashes,
            ),
        }
    counts_path = Path(str(meta.get("counts_path", cache_dir / "endpoint_suppression_counts.pkl")))
    with counts_path.open("rb") as handle:
        payload = pickle.load(handle)
    pair_counts = _deserialize_endpoint_pair_counts(dict(payload.get("pair_counts", {})))
    src_endpoint_counts = _deserialize_src_endpoint_counts(
        dict(payload.get("src_endpoint_counts", {})),
    )
    endpoint_counts = _deserialize_endpoint_action_counts(dict(payload.get("endpoint_counts", {})))
    netflow_endpoint_by_idx = {
        int(key): str(value)
        for key, value in dict(payload.get("netflow_endpoint_by_idx", {})).items()
    }
    lookup = _compact_lookup_from_legacy_counts(
        pair_counts,
        endpoint_counts,
        netflow_endpoint_by_idx,
    )
    return {
        "meta": meta,
        "pair_counts": pair_counts,
        "src_endpoint_counts": src_endpoint_counts,
        "endpoint_counts": endpoint_counts,
        "netflow_endpoint_by_idx": netflow_endpoint_by_idx,
        "endpoint_lookup": lookup,
    }


def _compact_lookup_from_legacy_counts(
    pair_counts: Mapping[tuple[int, str, int], int],
    endpoint_counts: Mapping[tuple[str, int], int],
    netflow_endpoint_by_idx: Mapping[int, str],
    src_endpoint_counts: Mapping[tuple[int, str], int] | None = None,
) -> CompactEndpointSuppressionLookup:
    compact_pairs = [
        (
            _endpoint_pair_compact_key(int(key[0]), _endpoint_hash(str(key[1])), int(key[2])),
            int(value),
        )
        for key, value in pair_counts.items()
    ]
    compact_pairs.sort(key=lambda item: item[0])
    src_endpoint_counts = src_endpoint_counts or {}
    compact_src_endpoints = [
        (_endpoint_src_compact_key(int(key[0]), _endpoint_hash(str(key[1]))), int(value))
        for key, value in src_endpoint_counts.items()
    ]
    compact_src_endpoints.sort(key=lambda item: item[0])
    compact_endpoints = [
        (_endpoint_action_compact_key(_endpoint_hash(str(key[0])), int(key[1])), int(value))
        for key, value in endpoint_counts.items()
    ]
    compact_endpoints.sort(key=lambda item: item[0])
    compact_netflows = [
        (int(node_idx), _endpoint_hash(str(endpoint)))
        for node_idx, endpoint in netflow_endpoint_by_idx.items()
    ]
    compact_netflows.sort(key=lambda item: item[0])
    return CompactEndpointSuppressionLookup(
        pair_keys=np.array([key for key, _ in compact_pairs], dtype=np.uint64),
        pair_counts=np.array([value for _, value in compact_pairs], dtype=np.int32),
        src_endpoint_keys=np.array([key for key, _ in compact_src_endpoints], dtype=np.uint64),
        src_endpoint_counts=np.array(
            [value for _, value in compact_src_endpoints],
            dtype=np.int32,
        ),
        endpoint_action_keys=np.array([key for key, _ in compact_endpoints], dtype=np.uint64),
        endpoint_action_counts=np.array([value for _, value in compact_endpoints], dtype=np.int32),
        netflow_idx_keys=np.array([key for key, _ in compact_netflows], dtype=np.uint64),
        netflow_endpoint_hashes=np.array([value for _, value in compact_netflows], dtype=np.uint32),
    )


def _conditional_endpoint_suppression_decision(
    *,
    config: SlimConfig,
    row: np.void,
    event_score: float,
    threshold: float,
    pair_counts: Mapping[tuple[int, str, int], int] | None = None,
    src_endpoint_counts: Mapping[tuple[int, str], int] | None = None,
    endpoint_counts: Mapping[tuple[str, int], int] | None = None,
    netflow_endpoint_by_idx: Mapping[int, str] | None = None,
    endpoint_lookup: CompactEndpointSuppressionLookup | None = None,
) -> dict[str, Any]:
    raw_pair_key = _conditional_endpoint_pair_key(row)
    pair_key = _conditional_endpoint_pair_signature(row, netflow_endpoint_by_idx)
    src_endpoint_key = _conditional_src_endpoint_signature(row, netflow_endpoint_by_idx)
    endpoint_key = _conditional_endpoint_action_signature(row, netflow_endpoint_by_idx)
    suppression_mode = str(config.conditional_endpoint_suppression_mode)
    if endpoint_lookup is not None:
        validation_pair_count = int(
            endpoint_lookup.pair_count(
                int(row["src_node_idx"]),
                int(row["dst_node_idx"]),
                int(row["action_id"]),
            ),
        )
        validation_src_endpoint_count = int(
            endpoint_lookup.src_endpoint_count(
                int(row["src_node_idx"]),
                int(row["dst_node_idx"]),
            ),
        )
        validation_endpoint_count = int(
            endpoint_lookup.endpoint_count(
                int(row["dst_node_idx"]),
                int(row["action_id"]),
            ),
        )
        endpoint_hash = endpoint_lookup.endpoint_hash_for_node(int(row["dst_node_idx"]))
        endpoint_signature = str(int(endpoint_hash))
        pair_key_text = (
            f"{int(row['src_node_idx'])}:{int(endpoint_hash)}:{int(row['action_id'])}"
        )
        src_endpoint_key_text = f"{int(row['src_node_idx'])}:{int(endpoint_hash)}"
        endpoint_key_text = f"{int(endpoint_hash)}:{int(row['action_id'])}"
    else:
        validation_pair_count = int((pair_counts or {}).get(pair_key, 0))
        validation_src_endpoint_count = int((src_endpoint_counts or {}).get(src_endpoint_key, 0))
        validation_endpoint_count = int((endpoint_counts or {}).get(endpoint_key, 0))
        endpoint_signature = str(pair_key[1])
        pair_key_text = _conditional_endpoint_pair_signature_text(pair_key)
        src_endpoint_key_text = _conditional_src_endpoint_signature_text(src_endpoint_key)
        endpoint_key_text = f"{endpoint_key[0]}:{int(endpoint_key[1])}"
    candidate = bool(
        config.conditional_endpoint_aware_suppression
        and _conditional_endpoint_is_suppression_candidate(
            row,
            include_read=bool(config.conditional_endpoint_suppression_read),
        )
    )
    known_pair = validation_pair_count >= int(config.conditional_known_pair_min_count)
    known_src_endpoint = validation_src_endpoint_count >= int(
        config.conditional_same_process_endpoint_min_count,
    )
    known_endpoint = validation_endpoint_count >= int(config.conditional_known_pair_min_count)
    if suppression_mode not in {
        "off",
        "pair_only",
        "pair_or_same_process_endpoint_history",
        "pair_then_endpoint",
    }:
        raise ValueError(
            "CONDITIONAL_ENDPOINT_SUPPRESSION_MODE must be pair_only, "
            "pair_or_same_process_endpoint_history, pair_then_endpoint, or off",
        )
    pair_margin = float(config.conditional_pair_suppression_margin)
    same_process_margin = float(config.conditional_same_process_endpoint_margin)
    suppressed = False
    if suppression_mode == "off":
        match_level = "off"
    elif known_pair:
        match_level = "pair"
        suppressed = bool(candidate and float(event_score) < float(threshold) + pair_margin)
    elif (
        suppression_mode == "pair_or_same_process_endpoint_history"
        and known_src_endpoint
    ):
        match_level = "same_process_endpoint_history"
        suppressed = bool(candidate and float(event_score) < float(threshold) + same_process_margin)
    elif suppression_mode == "pair_then_endpoint" and known_endpoint:
        match_level = "endpoint"
        suppressed = bool(candidate and float(event_score) < float(threshold) + pair_margin)
    elif known_src_endpoint:
        match_level = "same_process_endpoint_history_not_enabled"
    elif known_endpoint:
        match_level = "endpoint_not_enabled"
    else:
        match_level = "unknown"
    reason = ""
    if suppressed:
        reason = str(match_level)
    return {
        "candidate": candidate,
        "suppressed": suppressed,
        "validation_pair_count": validation_pair_count,
        "validation_src_endpoint_count": validation_src_endpoint_count,
        "validation_endpoint_count": validation_endpoint_count,
        "process_key": int(row["src_node_idx"]),
        "endpoint_key": int(row["dst_node_idx"]),
        "endpoint_signature": endpoint_signature,
        "pair_key": pair_key_text,
        "src_endpoint_key": src_endpoint_key_text,
        "endpoint_action_key": endpoint_key_text,
        "raw_pair_key": f"{int(raw_pair_key[0])}:{int(raw_pair_key[1])}:{int(raw_pair_key[2])}",
        "match_level": match_level,
        "suppression_mode": suppression_mode,
        "suppression_reason": reason,
        "threshold": float(threshold),
    }


def _conditional_endpoint_pair_key_text_from_row(row: Mapping[str, Any]) -> str:
    action_name = str(row.get("action", ""))
    try:
        action_id = int(ORTHRUS10_ACTION_NAMES.index(action_name))
    except ValueError:
        action_id = int(row.get("action_id", -1) or -1)
    return f"{int(row['src_idx'])}:{int(row['dst_idx'])}:{action_id}"


def _conditional_endpoint_signature_text_from_row(row: Mapping[str, Any]) -> str:
    action_name = str(row.get("action", ""))
    try:
        action_id = int(ORTHRUS10_ACTION_NAMES.index(action_name))
    except ValueError:
        action_id = int(row.get("action_id", -1) or -1)
    endpoint = str(row.get("endpoint_signature", "") or f"node:{int(row['dst_idx'])}")
    return f"{int(row['src_idx'])}:{endpoint}:{action_id}"


def _conditional_endpoint_summary_update(
    summary: dict[tuple[int, int, int], dict[str, Any]],
    row: np.void,
    *,
    include_read: bool,
    netflow_endpoint_by_idx: Mapping[int, str] | None,
    validation_pair_count: int,
    validation_endpoint_count: int,
    match_level: str,
    raw_alert: bool,
    suppressed: bool,
    event_score: float,
    threshold: float,
) -> None:
    if not _conditional_endpoint_is_suppression_candidate(row, include_read=include_read):
        return
    key = _conditional_endpoint_pair_key(row)
    signature_key = _conditional_endpoint_pair_signature(row, netflow_endpoint_by_idx)
    payload = summary.setdefault(
        key,
        {
            "action": str(ORTHRUS10_ACTION_NAMES[int(row["action_id"])]),
            "src_type": _phase3e_entity_type_name("src_type_id", int(row["src_type_id"])),
            "dst_type": _phase3e_entity_type_name("dst_type_id", int(row["dst_type_id"])),
            "process_key": int(row["src_node_idx"]),
            "endpoint_key": int(row["dst_node_idx"]),
            "pair_key": f"{int(key[0])}:{int(key[1])}:{int(key[2])}",
            "endpoint_signature": str(signature_key[1]),
            "pair_signature": _conditional_endpoint_pair_signature_text(signature_key),
            "validation_pair_count": int(validation_pair_count),
            "validation_endpoint_count": int(validation_endpoint_count),
            "suppression_match_level": str(match_level),
            "test_pair_count": 0,
            "raw_alert_count": 0,
            "suppressed_alert_count": 0,
            "final_alert_count": 0,
            "tp_before": 0,
            "fp_before": 0,
            "tp_after": 0,
            "fp_after": 0,
            "max_score": float(event_score),
            "score_sum": 0.0,
            "mean_score": 0.0,
            "threshold": float(threshold),
        },
    )
    payload["test_pair_count"] = int(payload.get("test_pair_count", 0)) + 1
    payload["validation_pair_count"] = int(validation_pair_count)
    payload["validation_endpoint_count"] = int(validation_endpoint_count)
    payload["suppression_match_level"] = str(match_level)
    payload["max_score"] = max(float(payload.get("max_score", 0.0)), float(event_score))
    payload["score_sum"] = float(payload.get("score_sum", 0.0)) + float(event_score)
    payload["mean_score"] = float(payload["score_sum"]) / max(int(payload["test_pair_count"]), 1)
    payload["threshold"] = float(threshold)
    if raw_alert:
        payload["raw_alert_count"] = int(payload.get("raw_alert_count", 0)) + 1
    if suppressed:
        payload["suppressed_alert_count"] = int(payload.get("suppressed_alert_count", 0)) + 1
    elif raw_alert:
        payload["final_alert_count"] = int(payload.get("final_alert_count", 0)) + 1


def _write_conditional_endpoint_suppression_summary(
    output_dir: Path,
    summary: Mapping[tuple[int, int, int], Mapping[str, Any]],
) -> Path:
    fields = [
        "action",
        "src_type",
        "dst_type",
        "process_key",
        "endpoint_key",
        "pair_key",
        "endpoint_signature",
        "pair_signature",
        "validation_pair_count",
        "validation_endpoint_count",
        "suppression_match_level",
        "test_pair_count",
        "raw_alert_count",
        "suppressed_alert_count",
        "final_alert_count",
        "tp_before",
        "fp_before",
        "tp_after",
        "fp_after",
        "max_score",
        "mean_score",
        "threshold",
    ]
    path = Path(output_dir) / "conditional_endpoint_suppression_summary.csv"
    rows = sorted(
        (dict(row) for row in summary.values()),
        key=lambda item: (
            -int(item.get("raw_alert_count", 0)),
            -int(item.get("suppressed_alert_count", 0)),
            str(item.get("pair_key", "")),
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return path


def _write_conditional_endpoint_suppressed_tp_events(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    fields = [
        *EVENT_EVAL_FIELDS,
        "endpoint_suppression_reason",
        "endpoint_validation_pair_count",
        "endpoint_validation_count",
        "endpoint_suppression_match_level",
    ]
    path = Path(output_dir) / "conditional_endpoint_suppressed_tp_events.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _write_conditional_endpoint_suppressed_tp_audit(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    fields = [
        "event_id",
        "action_name",
        "src_node_idx",
        "dst_node_idx",
        "src_type",
        "dst_type",
        "score",
        "threshold",
        "score_minus_threshold",
        "suppression_mode",
        "pair_count",
        "endpoint_action_count",
        "pair_key",
        "endpoint_action_key",
        "src_label",
        "dst_label",
        "is_tp",
        "is_fp",
        "endpoint_suppression_match_level",
        "endpoint_suppression_reason",
    ]
    path = Path(output_dir) / "conditional_endpoint_suppressed_tp_audit.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            score = float(row.get("event_score", 0.0) or 0.0)
            threshold = float(row.get("threshold", 0.0) or 0.0)
            src_label = str(row.get("src_node_label", row.get("src_label", "")))
            dst_label = str(row.get("dst_node_label", row.get("dst_label", "")))
            is_tp = bool(_parse_bool(row.get("is_correct_event_alert", False)))
            writer.writerow(
                {
                    "event_id": row.get("event_index", row.get("event_id", "")),
                    "action_name": row.get("action", ""),
                    "src_node_idx": row.get("info_src", row.get("src_idx", "")),
                    "dst_node_idx": row.get("info_dst", row.get("dst_idx", "")),
                    "src_type": row.get("src_type", ""),
                    "dst_type": row.get("dst_type", ""),
                    "score": score,
                    "threshold": threshold,
                    "score_minus_threshold": score - threshold,
                    "suppression_mode": row.get("endpoint_suppression_mode", ""),
                    "pair_count": row.get("endpoint_validation_pair_count", ""),
                    "endpoint_action_count": row.get("endpoint_validation_count", ""),
                    "pair_key": row.get("endpoint_pair_key", ""),
                    "endpoint_action_key": row.get("endpoint_action_key", ""),
                    "src_label": src_label,
                    "dst_label": dst_label,
                    "is_tp": is_tp,
                    "is_fp": not is_tp,
                    "endpoint_suppression_match_level": row.get(
                        "endpoint_suppression_match_level",
                        "",
                    ),
                    "endpoint_suppression_reason": row.get(
                        "endpoint_suppression_reason",
                        "",
                    ),
                },
            )
    return path


def _phase3g_update_dense_e2_state(
    row: np.void,
    z: np.ndarray,
    state_array: np.ndarray,
    has_state: np.ndarray,
    a: np.ndarray,
    g: np.ndarray,
) -> None:
    src_idx = int(row["src_node_idx"])
    dst_idx = int(row["dst_node_idx"])
    z_vec = np.asarray(z, dtype=np.float32)
    if src_idx == dst_idx:
        h_old = state_array[src_idx].copy() if bool(has_state[src_idx]) else np.zeros_like(z_vec)
        message = normalize_vector(z_vec + h_old)
        state_array[src_idx] = (a * h_old + g * message).astype(np.float32, copy=False)
        has_state[src_idx] = True
        return
    h_src = state_array[src_idx].copy() if bool(has_state[src_idx]) else np.zeros_like(z_vec)
    h_dst = state_array[dst_idx].copy() if bool(has_state[dst_idx]) else np.zeros_like(z_vec)
    src_message = z_vec
    dst_message = normalize_vector(z_vec + h_src)
    state_array[src_idx] = (a * h_src + g * src_message).astype(np.float32, copy=False)
    state_array[dst_idx] = (a * h_dst + g * dst_message).astype(np.float32, copy=False)
    has_state[src_idx] = True
    has_state[dst_idx] = True



def _phase3g_conditional_scores_stream(
    *,
    config: SlimConfig,
    head: ConditionalSemanticHead,
    event_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    state_model: SSPMLowRankModel | None = None,
    return_profile: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]]
):
    records = _phase3e_event_index_array(event_index)
    model = state_model
    if model is None:
        model = SSPMLowRankModel(_make_sspm_config(config, _load_process_config(config)))
    type_eye = _phase3g_type_eye()
    scores = np.zeros((int(records.shape[0]),), dtype=np.float32)
    target_cases = np.zeros((int(records.shape[0]),), dtype=np.int8)
    action_ids = np.zeros((int(records.shape[0]),), dtype=np.int16)
    src_type_ids = np.zeros((int(records.shape[0]),), dtype=np.int16)
    dst_type_ids = np.zeros((int(records.shape[0]),), dtype=np.int16)
    chunk_size = max(int(config.sspm_infer_chunk_events), 1)
    model.reset_state()
    if _phase3g_update_gate_enabled(config, model):
        for idx, row in enumerate(records):
            z_state = _phase3e_target_from_index_row(row, node_embeddings, action_embeddings)
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
            score = _phase3g_conditional_event_score(
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
            scores[idx] = np.float32(score)
            target_cases[idx] = target_case_id(target_case)
            action_ids[idx] = int(row["action_id"])
            src_type_ids[idx] = int(row["src_type_id"])
            dst_type_ids[idx] = int(row["dst_type_id"])
            action = str(ORTHRUS10_ACTION_NAMES[int(row["action_id"])])
            _phase3f_e2_none_update_from_numeric_row(
                model,
                row,
                z_state,
                action,
                src_type_name,
                dst_type_name,
                residual_score=score,
            )
        if bool(return_profile):
            return (
                scores,
                target_cases,
                action_ids,
                src_type_ids,
                dst_type_ids,
                int(records.shape[0]),
                dict(model.state_merge_profile_stats()),
            )
        return scores, target_cases, action_ids, src_type_ids, dst_type_ids, int(records.shape[0])
    for start in range(0, int(records.shape[0]), chunk_size):
        end = min(start + chunk_size, int(records.shape[0]))
        chunk = records[start:end]
        contexts = np.zeros((int(chunk.shape[0]), int(head.config.input_dim)), dtype=np.float32)
        targets = np.zeros((int(chunk.shape[0]), int(head.config.output_dim)), dtype=np.float32)
        case_ids = np.zeros((int(chunk.shape[0]),), dtype=np.int8)
        chunk_actions = np.zeros((int(chunk.shape[0]),), dtype=np.int16)
        chunk_src_types = np.zeros((int(chunk.shape[0]),), dtype=np.int16)
        chunk_dst_types = np.zeros((int(chunk.shape[0]),), dtype=np.int16)
        for offset, row in enumerate(chunk):
            z_state = _phase3e_target_from_index_row(row, node_embeddings, action_embeddings)
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
            targets[offset] = target
            case_ids[offset] = target_case_id(target_case)
            chunk_actions[offset] = int(row["action_id"])
            chunk_src_types[offset] = int(row["src_type_id"])
            chunk_dst_types[offset] = int(row["dst_type_id"])
            action = str(ORTHRUS10_ACTION_NAMES[int(row["action_id"])])
            _phase3f_e2_none_update_from_numeric_row(
                model,
                row,
                z_state,
                action,
                src_type_name,
                dst_type_name,
                residual_score=0.0,
            )
        predictions = head.predict(contexts, target_case_ids=case_ids if head.is_dual_head else None)
        scores[start:end] = conditional_distance(
            predictions,
            targets,
            str(config.conditional_semantic_loss),
        )
        target_cases[start:end] = case_ids
        action_ids[start:end] = chunk_actions
        src_type_ids[start:end] = chunk_src_types
        dst_type_ids[start:end] = chunk_dst_types
    if bool(return_profile):
        return (
            scores,
            target_cases,
            action_ids,
            src_type_ids,
            dst_type_ids,
            int(records.shape[0]),
            dict(model.state_merge_profile_stats()),
        )
    return scores, target_cases, action_ids, src_type_ids, dst_type_ids, int(records.shape[0])



# Explicit cross-module imports; replaces the migration namespace bridge.
from scripts.pipeline.checks.preflight import _load_process_config
from scripts.pipeline.features.semantic_features import (
    _StreamingScoreSummary,
    _empirical_tail_score_compact,
)
from scripts.pipeline.io.cache_payloads import _compact_sorted_scores
from scripts.pipeline.io.db_stream import _cfg_for_dataset
from scripts.pipeline.io.event_artifacts import (
    _model_calibration_summary,
    _model_update_gate_summary,
    _phase3e_entity_type_name,
    _phase3e_event_index_array,
    _phase3e_file_fingerprint,
    _phase3e_require_legacy_head_path_disabled,
    stable_json_hash,
)
from scripts.pipeline.outputs.alert_output import (
    _add_node_pool_alert,
    _current_rss_mb,
    _final_node_pool,
    _node_checkpoint,
    _raw_event_alert,
    _raw_output_paths,
    _safe_peak_rss_mb,
    _write_analysis_outputs_enabled,
    _write_raw_alerts_enabled,
    _write_state_merge_diagnostics_enabled,
)
from scripts.pipeline.outputs.metrics_summary import (
    _emit_profile_row,
    _event_score_from_residual_tail,
    _node_count_snapshot,
    _ofsm_alert_fields,
    _profiling_row,
    _residual_feature_fields,
    _state_merge_enabled,
    _state_merge_profile_row,
    _stream_csv_write_seconds,
    _update_node_count_state,
)
from scripts.pipeline.state.online_state_runtime import (
    _make_sspm_config,
    _parse_bool,
    _phase3e_fields_from_index_row,
    _phase3e_resolve_cache_dir,
    _phase3e_synthetic_row_from_index_row,
    _phase3e_target_from_index_row,
    _phase3f_e2_none_update_from_numeric_row,
    _phase3f_fast_path_enabled,
    _phase3g_update_gate_enabled,
    _stage_log,
)
