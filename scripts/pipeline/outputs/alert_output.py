"""Online alert CSV output, post-stream evaluation, and metrics payload helpers."""

from __future__ import annotations

from scripts.pipeline.config.runtime_config import *


def _task6_collect_in_memory_outputs(config: SlimConfig) -> bool:
    return bool(config.compat_in_memory_outputs)


def _deploy_light_enabled(config: SlimConfig) -> bool:
    return str(config.rss_profile_mode) == "deploy_light"


def _write_raw_alerts_enabled(config: SlimConfig) -> bool:
    return bool(config.write_raw_alerts) and not _deploy_light_enabled(config)


def _write_analysis_outputs_enabled(config: SlimConfig) -> bool:
    return bool(config.write_analysis_outputs) and not _deploy_light_enabled(config)


def _write_state_merge_diagnostics_enabled(config: SlimConfig) -> bool:
    return (
        bool(config.sspm_state_merge_write_diagnostics)
        and not _deploy_light_enabled(config)
    )


def _online_minimal_enabled(config: SlimConfig) -> bool:
    return str(config.rss_profile_mode) == "online_minimal"


def _minimal_node_pool_update(
    node_pool: dict[int, dict[str, Any]],
    node_id: int,
    event_id: int,
    stream_pos: int,
    event_score: float,
    residual_score: float,
) -> None:
    state = node_pool.setdefault(
        int(node_id),
        {
            "node_id": int(node_id),
            "node_score": 0.0,
            "candidate_mass": 0.0,
            "residual_mass": 0.0,
            "residual_max": 0.0,
            "residual_mean": 0.0,
            "candidate_event_count": 0,
            "first_alert_event_idx": int(event_id),
            "last_alert_event_idx": int(event_id),
            "first_alert_stream_pos": int(stream_pos),
            "last_alert_stream_pos": int(stream_pos),
        },
    )
    count = int(state.get("candidate_event_count", 0)) + 1
    state["candidate_event_count"] = count
    state["last_alert_event_idx"] = int(event_id)
    state["last_alert_stream_pos"] = int(stream_pos)
    state["candidate_mass"] = max(float(state.get("candidate_mass", 0.0)), float(event_score))
    state["residual_mass"] = max(float(state.get("residual_mass", 0.0)), float(residual_score))
    state["residual_max"] = max(float(state.get("residual_max", 0.0)), float(residual_score))
    state["residual_mean"] = float(state["residual_mass"])
    state["node_score"] = float(state["candidate_mass"])


def _final_node_pool_minimal(
    node_pool: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for state in node_pool.values():
        rows.append(
            {
                "node_id": int(state.get("node_id", -1)),
                "node_score": float(state.get("node_score", 0.0)),
                "candidate_mass": float(state.get("candidate_mass", 0.0)),
                "residual_mass": float(state.get("residual_mass", 0.0)),
                "residual_max": float(state.get("residual_max", 0.0)),
                "residual_mean": float(state.get("residual_mean", 0.0)),
                "association_support": 0,
                "adaptive_memory_deviation": 0.0,
                "compact_chain_support": 0.0,
                "chain_diversity_pool": 1,
                "repeated_consistency": float(state.get("candidate_event_count", 0)),
                "candidate_event_count": int(state.get("candidate_event_count", 0)),
            },
        )
    return sorted(rows, key=lambda item: (-float(item["node_score"]), int(item["node_id"])))


def _score_test_stream(
    config: SlimConfig,
    embedder: ResidualEmbedder,
    sspm: SSPMLowRankModel,
    test_rows,
    residual_scores: Sequence[float],
    threshold: float,
    process_cfg: ProcessSemanticConfig | None = None,
    process_audit: ProcessSemanticAudit | None = None,
    output_dir: Path | None = None,
    collect_in_memory: bool | None = None,
    initial_test_peak_rss_mb: float | None = None,
) -> dict[str, Any]:
    scoring_started = time.perf_counter()
    if collect_in_memory is None:
        collect_in_memory = bool(config.compat_in_memory_outputs)
    keep_alert_rows_for_eval = bool(collect_in_memory) or not _write_raw_alerts_enabled(config)
    sspm.reset_state()
    event_alerts_raw: list[dict[str, Any]] | None = [] if keep_alert_rows_for_eval else None
    checkpoints_raw: list[dict[str, Any]] | None = [] if collect_in_memory else None
    node_pool: dict[int, dict[str, Any]] = {}
    residual_sorted = _compact_sorted_scores(residual_scores)
    test_count = 0
    event_alert_count = 0
    checkpoint_count = 0
    node_feature_counts: dict[int, dict[str, int]] = {}
    last_checkpoint_written_stream_pos: int | None = None
    checkpoint_interval = max(int(config.progress_interval_events), 1)
    progress_interval = int(config.progress_interval_events)
    residual_semantic_examples: dict[str, list[dict[str, Any]]] = {
        group: []
        for group in RESIDUAL_EXAMPLE_GROUPS
    }
    threshold_controller: AdaptiveRateThresholdController | None = None
    if str(config.event_threshold_mode) == "adaptive_rate":
        target_rate = float(config.expected_event_alert_budget) / float(
            max(int(config.expected_alert_horizon_events), 1),
        )
        threshold_controller = AdaptiveRateThresholdController(
            initial_threshold=float(threshold),
            target_rate=target_rate,
        )
    raw_paths: dict[str, Path] = {}
    if initial_test_peak_rss_mb is None:
        test_phase_peak_rss_mb = _current_rss_mb()
    else:
        test_phase_peak_rss_mb = float(initial_test_peak_rss_mb)
    _stage_log(config, "test_scoring_start")
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
        for stream_pos, row in enumerate(test_rows):
            test_count = int(stream_pos) + 1
            if process_cfg is not None:
                _observe_process_semantic_audit(
                    process_audit,
                    "test",
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
            example_group = _residual_example_group(row)
            if example_group and len(residual_semantic_examples[example_group]) < 5:
                residual_semantic_examples[example_group].append(
                    _residual_audit_base_row(
                        row,
                        config,
                        process_cfg,
                        group=example_group,
                    ),
            )
            z = _encode_row(embedder, row, config, process_cfg)
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
            current_threshold = (
                threshold_controller.threshold if threshold_controller is not None else threshold
            )
            alerted = bool(event_score >= current_threshold)
            score_time_merge_snapshot = sspm.state_merge_snapshot(fields)
            pending_alert: dict[str, Any] | None = None
            alert_node_ids: set[int] = set()
            if alerted:
                pending_alert = _raw_event_alert(
                    row,
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
                    "test_scoring_progress",
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
            "test_scoring_end",
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
        "raw_outputs_streamed": output_dir is not None,
        "output_dir": str(output_dir) if output_dir is not None else "",
        "raw_output_paths": {key: str(path) for key, path in raw_paths.items()},
        "event_alert_count": int(event_alert_count),
        "checkpoint_count": int(checkpoint_count),
        "residual_semantic_examples": residual_semantic_examples,
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


def _raw_event_alert(
    row: Mapping[str, Any],
    fields: Mapping[str, Any],
    stream_pos: int,
    event_score: float,
    threshold: float,
) -> dict[str, Any]:
    return {
        "stream_pos": int(stream_pos),
        "event_index": int(row["event_index"]),
        "timestamp_ns": int(row["timestamp_ns"]),
        "src_idx": int(row["src_idx"]),
        "dst_idx": int(row["dst_idx"]),
        "info_src": int(fields["info_src"]),
        "info_dst": int(fields["info_dst"]),
        "action": fields["action"],
        "src_type": fields.get("src_type_name", ""),
        "dst_type": fields.get("dst_type_name", ""),
        "object_type": fields["object_type"],
        "dst_role": fields["dst_role"],
        "event_score": float(event_score),
        "threshold": float(threshold),
        "threshold_basis": "validation_expected_budget",
        "residual_tail": "",
        "residual_score": "",
        "residual_l2": "",
        "residual_cos": "",
        "top_residual_dim_1": "",
        "top_residual_dim_2": "",
        "top_residual_dim_3": "",
        "src_state_norm": "",
        "dst_state_norm": "",
        "src_event_count": "",
        "dst_event_count": "",
        "src_as_src_count": "",
        "dst_as_dst_count": "",
        **_ofsm_empty_node_snapshot("src"),
        **_ofsm_empty_node_snapshot("dst"),
    }


def _add_node_pool_alert(
    node_pool: dict[int, dict[str, Any]],
    node_id: int,
    alert: Mapping[str, Any],
    config: SlimConfig,
) -> None:
    state = node_pool.setdefault(
        int(node_id),
        {
            "node_id": int(node_id),
            "candidate_scores": [],
            "residual_scores": [],
            "chains": set(),
            "action_peer_pairs": set(),
            "candidate_event_count": 0,
        },
    )
    state["candidate_scores"].append(float(alert["event_score"]))
    state["candidate_event_count"] += 1
    state["residual_scores"].append(float(alert["residual_score"]))
    state["chains"].add((int(alert["info_src"]), int(alert["info_dst"])))
    if int(node_id) == int(alert["info_src"]):
        peer_role = str(alert.get("dst_role", "unknown"))
    else:
        peer_role = str(alert.get("object_type", "unknown"))
    state["action_peer_pairs"].add((str(alert.get("action", "")), peer_role))
    _refresh_node_pool_score(state, config)


def _node_pool_score(
    state: Mapping[str, Any],
    mode: str,
) -> float:
    candidate_mass = float(state.get("candidate_mass", 0.0))
    base_conf = candidate_mass
    if mode == "base":
        return candidate_mass
    if mode == "base_conf":
        return base_conf
    if mode == "base_conf_repeat":
        chains = max(len(state.get("chains", ())), 1)
        repeated = float(state.get("candidate_event_count", 0)) / float(chains)
        return base_conf + math.log1p(max(repeated, 0.0))
    if mode == "base_conf_chain":
        chains = len(state.get("chains", ()))
        return base_conf + math.log1p(max(chains, 0))
    residual_bonus = math.log1p(max(float(state.get("residual_mass", 0.0)), 0.0))
    chains = len(state.get("chains", ()))
    chain_bonus = math.log1p(max(chains, 0))
    assoc_bonus = math.log1p(max(len(state.get("action_peer_pairs", ())), 0))
    repeated = float(state.get("candidate_event_count", 0)) / float(max(chains, 1))
    repeat_bonus = math.log1p(max(repeated, 0.0))
    if mode == "base_conf_residual":
        return base_conf + residual_bonus
    if mode == "base_conf_repeat_chain":
        return base_conf + repeat_bonus + chain_bonus
    if mode == "base_conf_residual_repeat_chain":
        return base_conf + residual_bonus + repeat_bonus + chain_bonus
    if mode == "pool_base":
        return candidate_mass
    if mode == "pool_base_adaptive":
        return candidate_mass + residual_bonus
    if mode == "pool_base_assoc":
        return candidate_mass + assoc_bonus
    if mode == "pool_base_chain":
        return candidate_mass + chain_bonus
    if mode == "pool_base_adaptive_assoc_chain":
        return candidate_mass + residual_bonus + assoc_bonus + chain_bonus
    raise ValueError(f"unknown node_pool_score_mode: {mode}")


def _refresh_node_pool_score(state: dict[str, Any], config: SlimConfig) -> None:
    top_m = max(int(config.max_recent_events_per_node), 1)
    candidate_top = sorted(state["candidate_scores"], reverse=True)[:top_m]
    residual_top = sorted(state["residual_scores"], reverse=True)[:top_m]
    state["candidate_scores"] = candidate_top
    state["residual_scores"] = residual_top
    state["candidate_mass"] = float(sum(candidate_top))
    state["residual_mass"] = float(sum(residual_top))
    state["association_support"] = int(len(state.get("action_peer_pairs", ())))
    state["adaptive_memory_deviation"] = float(math.log1p(max(state["residual_mass"], 0.0)))
    state["compact_chain_support"] = float(
        math.log1p(max(len(state.get("chains", ())), 0)),
    )
    state["node_score"] = _node_pool_score(
        state,
        config.node_pool_score_mode,
    )


def _node_checkpoint(
    stream_pos: int,
    node_pool: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    best = max(
        node_pool.values(),
        key=lambda item: float(item.get("node_score", 0.0)),
        default=None,
    )
    if best is None:
        return {"stream_pos": int(stream_pos), "node_id": "", "node_score": 0.0, "pool_size": 0}
    return {
        "stream_pos": int(stream_pos),
        "node_id": int(best["node_id"]),
        "node_score": float(best["node_score"]),
        "pool_size": int(len(node_pool)),
    }


def _raw_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "events": output_dir / "online_event_alerts.raw.csv",
        "checkpoints": output_dir / "node_pool_checkpoints.raw.csv",
        "final_nodes": output_dir / "final_node_pool_alerts.raw.csv",
        "profiling": output_dir / "profiling.csv",
        "profiling_sspm_raw": output_dir / "profiling_sspm.raw.csv",
        "state_merge_profile": output_dir / "state_merge_profile.csv",
        "state_merge_diagnostics": output_dir / "state_merge_diagnostics.csv",
    }


def _output_policy() -> dict[str, bool]:
    return {
        "default_outputs_are_minimal": True,
        "raw_outputs_are_label_free": True,
        "evaluated_outputs_are_post_stream": True,
        "forensic_outputs_disabled_by_default": True,
        "debug_outputs_disabled_by_default": True,
    }


def _outputs_schema(output_dir: Path, config: SlimConfig) -> dict[str, str]:
    debug_audit_json = ""
    debug_audit_csv = ""
    if config.process_semantic_audit:
        debug_audit_json = str(output_dir / "debug_process_semantic_audit.json")
        debug_audit_csv = str(output_dir / "debug_process_semantic_audit.csv")
    legacy_eval_alias = ""
    if config.write_legacy_eval_alias:
        legacy_eval_alias = str(output_dir / "eval_causal_semantics.json")
    return {
        "eval_json": str(output_dir / "eval_causal_semantics_slim.json"),
        "online_event_alerts_raw_csv": str(output_dir / "online_event_alerts.raw.csv"),
        "final_node_pool_alerts_raw_csv": str(output_dir / "final_node_pool_alerts.raw.csv"),
        "node_pool_checkpoints_raw_csv": str(output_dir / "node_pool_checkpoints.raw.csv"),
        "online_event_alerts_csv": str(output_dir / "online_event_alerts.csv"),
        "online_node_alerts_csv": str(output_dir / "online_node_alerts.csv"),
        "final_node_pool_alerts_csv": str(output_dir / "final_node_pool_alerts.csv"),
        "node_pool_checkpoints_csv": str(output_dir / "node_pool_checkpoints.csv"),
        "profiling_csv": str(output_dir / "profiling.csv"),
        "profiling_sspm_raw_csv": str(output_dir / "profiling_sspm.raw.csv"),
        "state_merge_profile_csv": str(output_dir / "state_merge_profile.csv"),
        "state_merge_diagnostics_csv": str(output_dir / "state_merge_diagnostics.csv"),
        "online_node_alerts_strict_csv": str(output_dir / "online_node_alerts_strict.csv"),
        "online_node_alerts_relaxed_csv": str(output_dir / "online_node_alerts_relaxed.csv"),
        "alert_feature_dump_csv": str(output_dir / "alert_feature_dump.csv"),
        "fp_tp_categorical_lift_csv": str(output_dir / "fp_tp_categorical_lift.csv"),
        "fp_tp_numeric_summary_csv": str(output_dir / "fp_tp_numeric_summary.csv"),
        "ofsm_alert_merge_analysis_csv": str(output_dir / "ofsm_alert_merge_analysis.csv"),
        "metrics_json": str(output_dir / "metrics.json"),
        "config_effective_yaml": str(output_dir / "config_effective.yaml"),
        "legacy_eval_alias_json": legacy_eval_alias,
        "debug_process_semantic_audit_json": debug_audit_json,
        "debug_process_semantic_audit_csv": debug_audit_csv,
        "forensic_event_score_trace_csv": "",
        "forensic_ranked_events_csv": "",
        "debug_online_event_score_trace_csv": "",
    }


def _iter_csv_rows(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"required raw CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def _raw_path_from_stream_outputs(
    stream_outputs: Mapping[str, Any],
    path_key: str,
) -> Path | None:
    raw_paths = stream_outputs.get("raw_output_paths", {})
    if isinstance(raw_paths, Mapping) and raw_paths.get(path_key):
        return Path(str(raw_paths[path_key]))
    output_dir = str(stream_outputs.get("output_dir", ""))
    if output_dir:
        return _raw_output_paths(Path(output_dir))[path_key]
    return None


def _alert_event_indices(stream_outputs: Mapping[str, Any]) -> set[int]:
    event_indices: set[int] = set()
    for row in stream_outputs.get("event_alerts_raw", []):
        event_indices.add(int(row["event_index"]))
    raw_alerts_written = bool(
        stream_outputs.get("raw_alerts_written", stream_outputs.get("raw_outputs_streamed", False))
    )
    if (
        event_indices
        or not bool(stream_outputs.get("raw_outputs_streamed", False))
        or not raw_alerts_written
    ):
        return event_indices
    raw_path = _raw_path_from_stream_outputs(stream_outputs, "events")
    if raw_path is not None:
        for row in _iter_csv_rows(raw_path):
            event_indices.add(int(row["event_index"]))
    return event_indices


def collect_event_labels(
    rows,
    event_indices: set[int],
) -> dict[int, Any]:
    labels: dict[int, Any] = {}
    needed = set(int(value) for value in event_indices)
    if not needed:
        return labels
    for row in rows:
        event_index = int(row["event_index"])
        if event_index in needed:
            labels[event_index] = row.get("label", "benign")
            if len(labels) >= len(needed):
                break
    return labels


def _labels_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[int, Any]:
    return {
        int(row["event_index"]): row.get("label", "benign")
        for row in rows
    }


def _node_labels_from_rows(
    rows: Sequence[Mapping[str, Any]],
    max_tokens_per_node: int,
    dataset: str = "SYNTHETIC",
    process_semantic_config: ProcessSemanticConfig | None = None,
) -> dict[int, str]:
    labels: dict[int, str] = {}
    for row in rows:
        label = row.get("label", "benign")
        if not _is_attack_label(label):
            continue
        fields = row_fields(
            row,
            max_tokens_per_node,
            dataset=dataset,
            process_semantic_config=process_semantic_config,
        )
        for node_id in {int(fields["info_src"]), int(fields["info_dst"])}:
            labels[node_id] = _label_name(label)
    return labels


def _final_node_pool(
    node_pool: Mapping[int, Mapping[str, Any]],
    config: SlimConfig,
) -> list[dict[str, Any]]:
    rows = []
    for state in node_pool.values():
        residual_scores = list(state["residual_scores"])
        candidate_count = int(state["candidate_event_count"])
        rows.append(
            {
                "node_id": int(state["node_id"]),
                "node_score": float(state.get("node_score", 0.0)),
                "candidate_mass": float(state["candidate_mass"]),
                "residual_mass": float(state.get("residual_mass", 0.0)),
                "residual_max": float(max(residual_scores, default=0.0)),
                "residual_mean": float(np.mean(residual_scores)) if residual_scores else 0.0,
                "association_support": int(state.get("association_support", 0)),
                "adaptive_memory_deviation": float(
                    state.get("adaptive_memory_deviation", 0.0),
                ),
                "compact_chain_support": float(state.get("compact_chain_support", 0.0)),
                "chain_diversity_pool": int(len(state["chains"])),
                "repeated_consistency": float(
                    candidate_count / max(len(state["chains"]), 1),
                ),
                "candidate_event_count": candidate_count,
            },
        )
    return sorted(
        rows,
        key=lambda item: (-float(item["node_score"]), int(item["node_id"])),
    )


def _write_outputs(
    output_dir: Path,
    stream_outputs: Mapping[str, Any],
    eval_payload: Mapping[str, Any],
) -> dict[str, Any]:
    csv_started = time.perf_counter()
    event_labels_by_event = dict(stream_outputs.get("event_labels_by_event", {}))
    node_labels_by_node = dict(stream_outputs.get("node_labels_by_node", {}))
    final_nodes_raw = list(stream_outputs.get("final_nodes_raw", []))
    raw_outputs_streamed = bool(stream_outputs.get("raw_outputs_streamed", False))
    raw_alerts_written = bool(stream_outputs.get("raw_alerts_written", raw_outputs_streamed))
    analysis_outputs_enabled = bool(stream_outputs.get("analysis_outputs_written", True))
    compat_in_memory_active = bool(stream_outputs.get("compat_in_memory_outputs_active", False))
    evaluated_outputs_are_disk_backed = (
        raw_outputs_streamed and raw_alerts_written and not compat_in_memory_active
    )
    if not raw_alerts_written:
        raw_events = list(stream_outputs.get("event_alerts_raw", []))
        checkpoints_raw = list(stream_outputs.get("checkpoints_raw", []))
    elif not raw_outputs_streamed:
        raw_events = list(stream_outputs.get("event_alerts_raw", []))
        checkpoints_raw = list(stream_outputs.get("checkpoints_raw", []))
        _write_csv(output_dir / "online_event_alerts.raw.csv", raw_events, EVENT_RAW_FIELDS)
        _write_csv(
            output_dir / "node_pool_checkpoints.raw.csv",
            checkpoints_raw,
            CHECKPOINT_RAW_FIELDS,
        )
    if raw_alerts_written:
        _write_csv(output_dir / "final_node_pool_alerts.raw.csv", final_nodes_raw, NODE_RAW_FIELDS)
    if evaluated_outputs_are_disk_backed:
        evaluation_summary = _write_evaluated_outputs_from_raw(
            output_dir,
            event_labels_by_event,
            final_nodes_raw,
            node_labels_by_node,
            write_analysis_outputs=analysis_outputs_enabled,
        )
    else:
        raw_events = list(stream_outputs.get("event_alerts_raw", []))
        checkpoints_raw = list(stream_outputs.get("checkpoints_raw", []))
        evaluated_event_rows = _attach_event_labels(raw_events, event_labels_by_event)
        _write_csv(
            output_dir / "online_event_alerts.csv",
            evaluated_event_rows,
            EVENT_EVAL_FIELDS,
        )
        _write_csv(
            output_dir / "node_pool_checkpoints.csv",
            _attach_checkpoint_labels(checkpoints_raw, node_labels_by_node),
            CHECKPOINT_EVAL_FIELDS,
        )
        _write_csv(
            output_dir / "final_node_pool_alerts.csv",
            _attach_node_labels(final_nodes_raw, node_labels_by_node),
            NODE_EVAL_FIELDS,
        )
        _write_csv(
            output_dir / "online_node_alerts.csv",
            _attach_node_labels(final_nodes_raw, node_labels_by_node),
            NODE_EVAL_FIELDS,
        )
        node_pool_metrics = _write_strict_relaxed_node_alerts(
            output_dir,
            evaluated_event_rows,
            final_nodes_raw,
            node_labels_by_node,
        )
        alert_analysis_metrics = {}
        if analysis_outputs_enabled:
            alert_analysis_metrics = _write_alert_feature_analysis(
                output_dir,
                evaluated_event_rows,
                node_labels_by_node,
            )
        evaluation_summary = {
            "event_alerts": _event_metrics(raw_events, event_labels_by_event),
            **node_pool_metrics,
            **alert_analysis_metrics,
        }
    evaluation_summary = dict(evaluation_summary)
    evaluation_summary["node_pool_topk"] = _node_pool_topk_metrics(
        final_nodes_raw,
        node_labels_by_node,
        _node_pool_topk_values_from_payload(eval_payload),
    )
    payload_to_write = _payload_with_post_stream_output_seconds(
        eval_payload,
        time.perf_counter() - csv_started,
    )
    payload_to_write["primary_online_metrics"] = dict(evaluation_summary)
    output_mode = dict(payload_to_write.get("output_mode", {}))
    output_mode["evaluated_outputs_are_disk_backed"] = bool(evaluated_outputs_are_disk_backed)
    output_mode["compat_in_memory_outputs_active"] = bool(compat_in_memory_active)
    output_mode["raw_alerts_written"] = bool(raw_alerts_written)
    output_mode["analysis_outputs_written"] = bool(analysis_outputs_enabled)
    output_mode["state_merge_diagnostics_written"] = bool(
        stream_outputs.get("state_merge_diagnostics_written", False),
    )
    if evaluated_outputs_are_disk_backed:
        disk_backed_status = "active"
    elif compat_in_memory_active:
        disk_backed_status = "compat_in_memory"
    else:
        disk_backed_status = "in_memory_no_stream"
    output_mode["disk_backed_evaluated_csv_status"] = disk_backed_status
    output_mode["csv_write_seconds_scope"] = (
        "stream_csv_write_seconds + post_stream_output_seconds"
    )
    payload_to_write["output_mode"] = output_mode
    leakage_check = dict(payload_to_write.get("leakage_check", {}))
    leakage_check["task6_keeps_eval_rows_in_memory"] = bool(compat_in_memory_active)
    leakage_check["post_stream_evaluation_from_raw_csv"] = bool(evaluated_outputs_are_disk_backed)
    payload_to_write["leakage_check"] = leakage_check
    cache_data = payload_to_write.get("cache", {})
    checkpoint_path = ""
    if isinstance(cache_data, Mapping):
        checkpoint_path = str(cache_data.get("checkpoint_path", ""))
    if checkpoint_path:
        outputs_with_checkpoint = dict(payload_to_write.get("outputs", {}))
        outputs_with_checkpoint["sspm_checkpoint_path"] = checkpoint_path
        payload_to_write["outputs"] = outputs_with_checkpoint
    eval_path = output_dir / "eval_causal_semantics_slim.json"
    with eval_path.open("w", encoding="utf-8") as handle:
        json.dump(payload_to_write, handle, indent=2, sort_keys=True)
        handle.write("\n")
    payload_config = payload_to_write.get("config", {})
    if isinstance(payload_config, Mapping) and payload_config.get("write_legacy_eval_alias"):
        with (output_dir / "eval_causal_semantics.json").open("w", encoding="utf-8") as handle:
            json.dump(payload_to_write, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return payload_to_write


def _write_process_semantic_audit_outputs(
    output_dir: Path,
    audit: ProcessSemanticAudit | None,
    train_available: bool = True,
    validation_available: bool = True,
    test_available: bool = True,
    skipped_reason: str = "",
) -> None:
    if audit is None:
        return
    with (output_dir / "debug_process_semantic_audit.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            audit.to_json_summary(
                train_available=train_available,
                validation_available=validation_available,
                test_available=test_available,
                skipped_reason=skipped_reason,
            ),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    _write_csv(
        output_dir / "debug_process_semantic_audit.csv",
        audit.csv_rows(),
        PROCESS_AUDIT_FIELDS,
    )


def _write_evaluated_outputs_from_raw(
    output_dir: Path,
    event_labels_by_event: Mapping[int, Any],
    final_nodes_raw: Sequence[Mapping[str, Any]],
    node_labels_by_node: Mapping[int, Any],
    write_analysis_outputs: bool = True,
) -> dict[str, Any]:
    paths = _raw_output_paths(output_dir)
    event_metrics = _write_evaluated_event_csv_from_raw(
        paths["events"],
        output_dir / "online_event_alerts.csv",
        event_labels_by_event,
        EVENT_EVAL_FIELDS,
    )
    evaluated_event_rows = list(_iter_csv_rows(output_dir / "online_event_alerts.csv"))
    group_summary_path = output_dir / "conditional_score_summary_by_target_action_type.csv"
    if group_summary_path.exists():
        group_rows = list(_iter_csv_rows(group_summary_path))
        if group_rows:
            _write_conditional_group_test_summary(
                output_dir,
                _conditional_group_summary_with_eval(group_rows, evaluated_event_rows),
            )
    _write_evaluated_checkpoints_from_raw(
        paths["checkpoints"],
        output_dir / "node_pool_checkpoints.csv",
        node_labels_by_node,
    )
    _write_csv(
        output_dir / "final_node_pool_alerts.csv",
        _attach_node_labels(final_nodes_raw, node_labels_by_node),
        NODE_EVAL_FIELDS,
    )
    _write_csv(
        output_dir / "online_node_alerts.csv",
        _attach_node_labels(final_nodes_raw, node_labels_by_node),
        NODE_EVAL_FIELDS,
    )
    node_pool_metrics = _write_strict_relaxed_node_alerts(
        output_dir,
        evaluated_event_rows,
        final_nodes_raw,
        node_labels_by_node,
    )
    alert_analysis_metrics = {}
    if bool(write_analysis_outputs):
        alert_analysis_metrics = _write_alert_feature_analysis(
            output_dir,
            evaluated_event_rows,
            node_labels_by_node,
        )
    return {
        "event_alerts": event_metrics,
        **node_pool_metrics,
        **alert_analysis_metrics,
    }


def _write_evaluated_event_csv_from_raw(
    raw_path: Path,
    output_path: Path,
    labels_by_event: Mapping[int, Any],
    fieldnames: Sequence[str],
) -> dict[str, Any]:
    tp = 0
    fp = 0
    with StreamingCsvWriter(output_path, fieldnames) as writer:
        for row in _iter_csv_rows(raw_path):
            evaluated = _evaluated_event_row(row, labels_by_event)
            if bool(evaluated["is_correct_event_alert"]):
                tp += 1
            else:
                fp += 1
            writer.write_row(evaluated)
    return _event_metrics_from_counts(tp, fp)


def _write_evaluated_checkpoints_from_raw(
    raw_path: Path,
    output_path: Path,
    labels_by_node: Mapping[int, Any],
) -> None:
    with StreamingCsvWriter(output_path, CHECKPOINT_EVAL_FIELDS) as writer:
        for row in _iter_csv_rows(raw_path):
            writer.write_row(_evaluated_checkpoint_row(row, labels_by_node))


def _write_strict_relaxed_node_alerts(
    output_dir: Path,
    event_rows: Sequence[Mapping[str, Any]],
    final_nodes_raw: Sequence[Mapping[str, Any]],
    node_labels_by_node: Mapping[int, Any],
) -> dict[str, Any]:
    """Write strict and relaxed node-pool evaluation views after streaming."""
    support = _node_alert_support(event_rows)
    strict_rows = []
    relaxed_rows = []
    for raw_node in final_nodes_raw:
        node_id = _node_id_from_row(raw_node)
        if node_id is None:
            continue
        node_support = support.get(int(node_id), {})
        node_label = _label_name(node_labels_by_node.get(int(node_id), "benign"))
        base = {
            "node_id": int(node_id),
            "node_type": str(raw_node.get("node_type", "")),
            "node_label": node_label,
            "first_alert_event_idx": node_support.get("first_alert_event_idx", ""),
            "alert_count": int(node_support.get("alert_count", 0)),
            "max_event_score": float(node_support.get("max_event_score", 0.0)),
            "supporting_event_count": int(node_support.get("supporting_event_count", 0)),
        }
        strict_rows.append(
            {
                **base,
                "pool_type": "strict",
                "eval_result": "TP" if node_label == "malicious" else "FP",
            },
        )
        relaxed_result = _relaxed_node_eval_result(node_label, node_support)
        relaxed_rows.append(
            {
                **base,
                "pool_type": "relaxed",
                "eval_result": relaxed_result,
            },
        )
    _write_csv(output_dir / "online_node_alerts_strict.csv", strict_rows, NODE_POOL_EVAL_FIELDS)
    _write_csv(output_dir / "online_node_alerts_relaxed.csv", relaxed_rows, NODE_POOL_EVAL_FIELDS)
    return {
        "node_pool_strict": _eval_result_counts(strict_rows),
        "node_pool_relaxed": _eval_result_counts(relaxed_rows),
    }


def _node_alert_support(event_rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    support: dict[int, dict[str, Any]] = {}
    for row in event_rows:
        event_index = int(row.get("event_index", -1))
        event_score = _safe_float(row.get("event_score", 0.0))
        event_label = _label_name(row.get("event_label", "benign"))
        for key in ("info_src", "info_dst"):
            node_id = int(row.get(key, -1))
            if node_id < 0:
                continue
            state = support.setdefault(
                node_id,
                {
                    "first_alert_event_idx": event_index,
                    "alert_count": 0,
                    "max_event_score": 0.0,
                    "supporting_event_count": 0,
                    "supporting_event_labels": [],
                },
            )
            state["first_alert_event_idx"] = min(
                int(state["first_alert_event_idx"]),
                event_index,
            )
            state["alert_count"] = int(state["alert_count"]) + 1
            state["supporting_event_count"] = int(state["supporting_event_count"]) + 1
            state["max_event_score"] = max(float(state["max_event_score"]), event_score)
            state["supporting_event_labels"].append(event_label)
    return support


def _relaxed_node_eval_result(node_label: str, node_support: Mapping[str, Any]) -> str:
    if node_label in {"malicious", "suspicious"}:
        return "TP"
    labels = set(str(value) for value in node_support.get("supporting_event_labels", []))
    if labels and labels.issubset({"malicious", "suspicious"}):
        return "IGNORE"
    return "FP"


def _eval_result_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    tp = sum(1 for row in rows if str(row.get("eval_result", "")) == "TP")
    fp = sum(1 for row in rows if str(row.get("eval_result", "")) == "FP")
    ignored = sum(1 for row in rows if str(row.get("eval_result", "")) == "IGNORE")
    return {
        "tp": int(tp),
        "fp": int(fp),
        "ignore": int(ignored),
        "count": int(len(rows)),
    }


def _write_alert_feature_analysis(
    output_dir: Path,
    event_rows: Sequence[Mapping[str, Any]],
    node_labels_by_node: Mapping[int, Any],
) -> dict[str, Any]:
    feature_rows = [
        _alert_feature_row(row, node_labels_by_node)
        for row in event_rows
    ]
    _write_csv(output_dir / "alert_feature_dump.csv", feature_rows, ALERT_FEATURE_DUMP_FIELDS)
    _write_csv(
        output_dir / "fp_tp_categorical_lift.csv",
        _categorical_lift_rows(feature_rows),
        FP_TP_CATEGORICAL_LIFT_FIELDS,
    )
    _write_csv(
        output_dir / "fp_tp_numeric_summary.csv",
        _numeric_summary_rows(feature_rows),
        FP_TP_NUMERIC_SUMMARY_FIELDS,
    )
    _write_csv(
        output_dir / "ofsm_alert_merge_analysis.csv",
        _ofsm_alert_merge_rows(feature_rows),
        OFSM_ALERT_MERGE_ANALYSIS_FIELDS,
    )
    return {
        "alert_feature_analysis": {
            "alert_feature_rows": int(len(feature_rows)),
            "covers_alert_events_only": True,
        },
    }


def _alert_feature_row(
    row: Mapping[str, Any],
    node_labels_by_node: Mapping[int, Any],
) -> dict[str, Any]:
    src_node = int(row.get("info_src", -1))
    dst_node = int(row.get("info_dst", -1))
    src_label = _label_name(node_labels_by_node.get(src_node, "benign"))
    dst_label = _label_name(node_labels_by_node.get(dst_node, "benign"))
    action = str(row.get("action", row.get("raw_action", "")))
    event_score = _safe_float(row.get("event_score", 0.0))
    threshold = _safe_float(row.get("threshold", 0.0))
    base = {
        "event_index": int(row.get("event_index", -1)),
        "stream_pos": int(row.get("stream_pos", -1)),
        "event_label": _label_name(row.get("event_label", "benign")),
        "src_node_label": src_label,
        "dst_node_label": dst_label,
        "raw_action": action,
        "action_family": _action_family(action),
        "src_type": str(row.get("src_type", "")),
        "dst_type": str(row.get("dst_type", "")),
        "has_tmp": int(_row_contains_fragment(row, "/tmp")),
        "has_so": int(_row_contains_fragment(row, ".so")),
        "has_private_ip": "",
        "has_public_ip": "",
        "has_loopback_ip": "",
        "port_bucket": "",
        "event_score": event_score,
        "threshold": threshold,
        "score_margin": float(event_score - threshold),
        "residual_score": _safe_float(row.get("residual_score", 0.0)),
        "residual_l2": _safe_float(row.get("residual_l2", 0.0)),
        "residual_cos": _safe_float(row.get("residual_cos", 0.0)),
        "top_residual_dim_1": row.get("top_residual_dim_1", ""),
        "top_residual_dim_2": row.get("top_residual_dim_2", ""),
        "top_residual_dim_3": row.get("top_residual_dim_3", ""),
        "src_state_norm": _safe_float(row.get("src_state_norm", 0.0)),
        "dst_state_norm": _safe_float(row.get("dst_state_norm", 0.0)),
        "src_event_count": int(_safe_float(row.get("src_event_count", 0.0))),
        "dst_event_count": int(_safe_float(row.get("dst_event_count", 0.0))),
        "src_as_src_count": int(_safe_float(row.get("src_as_src_count", 0.0))),
        "dst_as_dst_count": int(_safe_float(row.get("dst_as_dst_count", 0.0))),
    }
    base["src_dst_type"] = f"{base['src_type']}->{base['dst_type']}"
    for field in OFSM_EVENT_RAW_FIELDS:
        base[field] = row.get(field, "")
    base["alert_group"] = _alert_group(base)
    return base


def _alert_group(row: Mapping[str, Any]) -> str:
    src_label = _label_name(row.get("src_node_label", "benign"))
    dst_label = _label_name(row.get("dst_node_label", "benign"))
    event_label = _label_name(row.get("event_label", "benign"))
    if src_label == "malicious" and dst_label == "malicious":
        return "TP_both_malicious"
    if src_label == "malicious" or dst_label == "malicious" or event_label == "malicious":
        return "TP_any_malicious"
    if src_label == "suspicious" or dst_label == "suspicious" or event_label == "suspicious":
        return "TP_suspicious"
    return "FP_pure_benign"


def _categorical_lift_rows(feature_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "raw_action",
        "action_family",
        "src_type",
        "dst_type",
        "src_dst_type",
        "has_tmp",
        "has_so",
        "has_private_ip",
        "has_public_ip",
        "has_loopback_ip",
        "port_bucket",
    ]
    fp_rows = [row for row in feature_rows if row.get("alert_group") == "FP_pure_benign"]
    tp_rows = [row for row in feature_rows if row.get("alert_group") == "TP_both_malicious"]
    out = []
    for feature in features:
        values = sorted({str(row.get(feature, "")) for row in feature_rows})
        for value in values:
            fp_count = sum(1 for row in fp_rows if str(row.get(feature, "")) == value)
            tp_count = sum(1 for row in tp_rows if str(row.get(feature, "")) == value)
            fp_rate = float(fp_count / max(len(fp_rows), 1))
            tp_rate = float(tp_count / max(len(tp_rows), 1))
            out.append(
                {
                    "feature": feature,
                    "value": value,
                    "fp_count": int(fp_count),
                    "tp_both_malicious_count": int(tp_count),
                    "fp_rate": fp_rate,
                    "tp_both_malicious_rate": tp_rate,
                    "lift": float(fp_rate / (tp_rate + 1e-8)),
                },
            )
    return out


def _numeric_summary_rows(feature_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    features = [
        "event_score",
        "score_margin",
        "residual_score",
        "residual_l2",
        "residual_cos",
        "src_state_norm",
        "dst_state_norm",
        "src_cluster_size",
        "dst_cluster_size",
        "src_merge_similarity",
        "dst_merge_similarity",
    ]
    fp_rows = [row for row in feature_rows if row.get("alert_group") == "FP_pure_benign"]
    tp_rows = [row for row in feature_rows if row.get("alert_group") == "TP_both_malicious"]
    out = []
    for feature in features:
        fp_stats = _numeric_stats([_safe_float(row.get(feature), None) for row in fp_rows])
        tp_stats = _numeric_stats([_safe_float(row.get(feature), None) for row in tp_rows])
        difference = (
            ""
            if fp_stats is None or tp_stats is None
            else float(fp_stats["mean"] - tp_stats["mean"])
        )
        ratio = (
            ""
            if fp_stats is None or tp_stats is None or abs(float(tp_stats["mean"])) < 1e-12
            else float(fp_stats["mean"] / tp_stats["mean"])
        )
        out.append(
            {
                "feature": feature,
                "fp_mean": "" if fp_stats is None else fp_stats["mean"],
                "fp_p50": "" if fp_stats is None else fp_stats["p50"],
                "fp_p90": "" if fp_stats is None else fp_stats["p90"],
                "fp_p99": "" if fp_stats is None else fp_stats["p99"],
                "tp_both_malicious_mean": "" if tp_stats is None else tp_stats["mean"],
                "tp_both_malicious_p50": "" if tp_stats is None else tp_stats["p50"],
                "tp_both_malicious_p90": "" if tp_stats is None else tp_stats["p90"],
                "tp_both_malicious_p99": "" if tp_stats is None else tp_stats["p99"],
                "difference": difference,
                "ratio": ratio,
            },
        )
    return out


def _numeric_stats(values: Sequence[float | None]) -> dict[str, float] | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not clean:
        return None
    array = np.asarray(clean, dtype=np.float64)
    p50, p90, p99 = np.percentile(array, [50, 90, 99])
    return {
        "mean": float(np.mean(array)),
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
    }


def _ofsm_alert_merge_rows(feature_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in feature_rows:
        src_cluster = str(row.get("src_cluster_id", "")).strip()
        dst_cluster = str(row.get("dst_cluster_id", "")).strip()
        same_cluster = bool(src_cluster and dst_cluster and src_cluster == dst_cluster)
        rows.append(
            {
                "event_index": row.get("event_index", ""),
                "alert_group": row.get("alert_group", ""),
                "src_is_clustered": row.get("src_is_clustered", 0),
                "dst_is_clustered": row.get("dst_is_clustered", 0),
                "src_cluster_size": row.get("src_cluster_size", 1),
                "dst_cluster_size": row.get("dst_cluster_size", 1),
                "src_merge_similarity": row.get("src_merge_similarity", 0.0),
                "dst_merge_similarity": row.get("dst_merge_similarity", 0.0),
                "same_cluster": int(same_cluster),
                "src_copy_on_write": row.get("src_copy_on_write", 0),
                "dst_copy_on_write": row.get("dst_copy_on_write", 0),
                "event_score": row.get("event_score", 0.0),
                "score_margin": row.get("score_margin", 0.0),
            },
        )
    return rows


def _action_family(action: object) -> str:
    text = str(action).lower()
    for family in ("read", "recv", "write", "send", "connect", "open", "execute", "clone"):
        if family in text:
            return family
    return "other"


def _row_contains_fragment(row: Mapping[str, Any], fragment: str) -> bool:
    needle = str(fragment).lower()
    return any(needle in str(value).lower() for value in row.values())


def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed):
        return default
    return parsed


def _write_malicious_pair_audit_csv(
    output_path: Path,
    rows,
    abnormal_nodes: set[int],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
) -> int:
    """Write post-stream GT abnormal-pair residual token audit rows."""
    count = 0
    try:
        with StreamingCsvWriter(output_path, MALICIOUS_PAIR_AUDIT_FIELDS) as writer:
            for row in rows:
                audit_row = _malicious_pair_audit_row(row, abnormal_nodes, config, process_cfg)
                if audit_row is None:
                    continue
                writer.write_row(audit_row)
                count += 1
    except AttributeError as exc:
        if "cursor" not in str(exc):
            raise
    return count


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _payload_with_post_stream_output_seconds(
    eval_payload: Mapping[str, Any],
    post_stream_output_seconds: float,
) -> dict[str, Any]:
    payload = dict(eval_payload)
    timing = _empty_timing()
    timing.update(dict(payload.get("timing", {})))
    timing.pop("post_stream_csv_write_seconds", None)
    timing["post_stream_output_seconds"] = float(
        timing["post_stream_output_seconds"],
    ) + float(post_stream_output_seconds)
    timing["csv_write_seconds"] = (
        float(timing["stream_csv_write_seconds"])
        + float(timing["post_stream_output_seconds"])
    )
    payload["timing"] = timing
    return payload


def _attach_event_labels(
    rows: Sequence[Mapping[str, Any]],
    labels_by_event: Mapping[int, Any],
) -> list[dict[str, Any]]:
    return [_evaluated_event_row(row, labels_by_event) for row in rows]


def _attach_checkpoint_labels(
    rows: Sequence[Mapping[str, Any]],
    labels_by_node: Mapping[int, Any] | None = None,
) -> list[dict[str, Any]]:
    return [_evaluated_checkpoint_row(row, labels_by_node or {}) for row in rows]


def _attach_node_labels(
    rows: Sequence[Mapping[str, Any]],
    labels_by_node: Mapping[int, Any] | None = None,
) -> list[dict[str, Any]]:
    return [_evaluated_node_row(row, labels_by_node or {}) for row in rows]


def _evaluated_event_row(
    row: Mapping[str, Any],
    labels_by_event: Mapping[int, Any],
) -> dict[str, Any]:
    label = labels_by_event.get(int(row["event_index"]), "benign")
    return dict(
        row,
        event_label=label,
        is_correct_event_alert=_is_attack_label(label),
    )


def _evaluated_checkpoint_row(
    row: Mapping[str, Any],
    labels_by_node: Mapping[int, Any] | None = None,
) -> dict[str, Any]:
    return _evaluated_node_row(row, labels_by_node or {})


def _evaluated_node_row(
    row: Mapping[str, Any],
    labels_by_node: Mapping[int, Any],
) -> dict[str, Any]:
    node_id = _node_id_from_row(row)
    label = labels_by_node.get(node_id, "benign") if node_id is not None else "benign"
    return dict(
        row,
        node_label=_label_name(label),
        is_correct_node_alert=_is_attack_label(label),
    )


def _node_id_from_row(row: Mapping[str, Any]) -> int | None:
    text = str(row.get("node_id", "")).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _eval_payload(
    config: SlimConfig,
    stream_outputs: Mapping[str, Any],
    sspm: SSPMLowRankModel,
    elapsed_seconds: float,
    test_count: int,
    timing: Mapping[str, float] | None = None,
    memory: Mapping[str, Any] | None = None,
    split_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    event_labels_by_event = stream_outputs.get("event_labels_by_event", {})
    event_metrics = _event_metrics(
        stream_outputs.get("event_alerts_raw", []),
        event_labels_by_event,
    )
    payload_timing = _empty_timing()
    if timing is not None:
        payload_timing.update({key: float(value) for key, value in dict(timing).items()})
    payload_timing["stream_csv_write_seconds"] = float(
        stream_outputs.get("stream_csv_write_seconds", payload_timing["stream_csv_write_seconds"]),
    )
    payload_timing["csv_write_seconds"] = (
        float(payload_timing["stream_csv_write_seconds"])
        + float(payload_timing["post_stream_output_seconds"])
    )
    payload_memory = _final_memory_profile(memory, sspm)
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
    ):
        if key in stream_outputs:
            payload_memory[key] = stream_outputs.get(key)
    compat_in_memory_active = bool(
        stream_outputs.get("compat_in_memory_outputs_active", config.compat_in_memory_outputs),
    )
    disk_backed_eval = bool(stream_outputs.get("raw_outputs_streamed", False)) and (
        not compat_in_memory_active
    )
    if disk_backed_eval:
        disk_backed_status = "active"
    elif compat_in_memory_active:
        disk_backed_status = "compat_in_memory"
    else:
        disk_backed_status = "in_memory_no_stream"
    stream_status = stream_outputs.get("db_stream_status")
    if isinstance(stream_status, Mapping):
        db_stream_status = _db_stream_status(
            str(stream_status.get("db_stream_mode_requested", config.db_stream_mode)),
            str(
                stream_status.get(
                    "db_stream_mode_actual",
                    stream_status.get("db_stream_mode", "python_lookup"),
                ),
            ),
            str(
                stream_status.get(
                    "db_stream_mode_fallback_reason",
                    stream_status.get("db_stream_fallback_reason", ""),
                ),
            ),
        )
    else:
        default_actual = "not_applicable" if config.synthetic_smoke else "python_lookup"
        db_stream_status = _db_stream_status(config.db_stream_mode, default_actual, "")
    output_dir_text = str(stream_outputs.get("output_dir", ""))
    outputs = _outputs_schema(Path(output_dir_text), config) if output_dir_text else {}
    payload = {
        "method_name": "causal_semantics_slim",
        "method_version": METHOD_VERSION,
        "config": asdict(config),
        "residual_method": "sspm",
        "labels_attached_after_streaming": True,
        "raw_outputs_are_label_free": True,
        "ground_truth_used_only_for_evaluation": True,
        "output_mode": {
            "raw_outputs_streamed_to_disk": bool(stream_outputs.get("raw_outputs_streamed", False)),
            "compat_in_memory_outputs_config": bool(config.compat_in_memory_outputs),
            "compat_in_memory_outputs_active": compat_in_memory_active,
            "evaluated_outputs_are_disk_backed": disk_backed_eval,
            "disk_backed_evaluated_csv_status": disk_backed_status,
            "csv_write_seconds_scope": (
                "stream_csv_write_seconds + post_stream_output_seconds"
            ),
            "rss_profile_mode": str(config.rss_profile_mode),
            "raw_alerts_written": bool(stream_outputs.get("raw_alerts_written", False)),
            "analysis_outputs_written": bool(
                stream_outputs.get("analysis_outputs_written", True),
            ),
            "state_merge_diagnostics_written": bool(
                stream_outputs.get("state_merge_diagnostics_written", False),
            ),
        },
        "primary_online_metrics": {
            "event_alerts": event_metrics,
        },
        "runtime": {
            "elapsed_seconds": float(elapsed_seconds),
            "events_scored": int(test_count),
            "throughput_events_per_second": float(test_count / elapsed_seconds),
        },
        "semantic_embedding": dict(stream_outputs.get("semantic_embedding", {})),
        "vocab_stats": dict(stream_outputs.get("vocab_stats", {})),
        "latent_dim_selection": dict(stream_outputs.get("latent_dim_selection", {})),
        "residual_semantic_examples": dict(
            stream_outputs.get("residual_semantic_examples", {}),
        ),
        "residual_semantic_audit_outputs": dict(
            stream_outputs.get("residual_semantic_audit_outputs", {}),
        ),
        "sspm_context": {
            "context_mode": str(config.sspm_context_mode),
            "global_context_mode": str(config.sspm_global_context_mode),
            "action_count": int(config.action_count),
            "context_dim": int(sspm.context_dim),
        },
        "sspm_training": dict(getattr(sspm, "training_stats", {}) or {}),
        "threshold_controller": dict(stream_outputs.get("threshold_controller", {})),
        "state_merge": dict(sspm.state_merge_profile_stats()),
        "timing": payload_timing,
        "memory": payload_memory,
        "leakage_check": {
            "test_labels_used_before_scoring": False,
            "labels_attached_after_streaming": True,
            "raw_outputs_are_label_free": True,
            "ground_truth_used_only_for_evaluation": True,
            "training_uses_train_rows_only": True,
            "validation_calibration_uses_no_labels": True,
            "test_labels_loaded_after_streaming_only": True,
            "raw_outputs_have_no_label_columns": True,
            "task6_keeps_eval_rows_in_memory": compat_in_memory_active,
            "state_merge_online_outputs_are_label_free": True,
        },
        "output_policy": _output_policy(),
        "outputs": outputs,
    }
    payload.update(db_stream_status)
    payload.update(_eval_split_metadata(config, split_metadata))
    return payload


def _event_metrics(
    alerts: Sequence[Mapping[str, Any]],
    labels_by_event: Mapping[int, Any],
) -> dict[str, Any]:
    tp = 0
    fp = 0
    for alert in alerts:
        label = labels_by_event.get(int(alert["event_index"]), "benign")
        if _is_attack_label(label):
            tp += 1
        else:
            fp += 1
    return _event_metrics_from_counts(tp, fp)


def _event_metrics_from_counts(tp: int, fp: int) -> dict[str, Any]:
    total = int(tp) + int(fp)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "ratio": float(tp / total) if total else 0.0,
        "count": int(total),
    }


def _node_pool_topk_metrics(
    rows: Sequence[Mapping[str, Any]],
    labels_by_node: Mapping[int, Any],
    topk_values: Sequence[int],
) -> dict[str, dict[str, Any]]:
    sorted_rows = sorted(rows, key=_node_pool_sort_key)
    metrics: dict[str, dict[str, Any]] = {}
    for raw_topk in topk_values:
        topk = int(raw_topk)
        if topk <= 0:
            continue
        selected = sorted_rows[:topk]
        tp = 0
        fp = 0
        for row in selected:
            node_id = _node_id_from_row(row)
            label = labels_by_node.get(node_id, "benign") if node_id is not None else "benign"
            if _is_attack_label(label):
                tp += 1
            else:
                fp += 1
        count = int(tp + fp)
        ratio = float(tp / count) if count else 0.0
        metrics[f"top{topk}"] = {
            "k": int(topk),
            "alert_count_at_k": count,
            "count": count,
            "tp": int(tp),
            "fp": int(fp),
            "precision": ratio,
            "ratio": ratio,
        }
    return metrics


def _node_pool_sort_key(row: Mapping[str, Any]) -> tuple[float, int]:
    node_id = _node_id_from_row(row)
    sort_node_id = int(node_id) if node_id is not None else sys.maxsize
    return (-_node_score_from_row(row), sort_node_id)


def _node_score_from_row(row: Mapping[str, Any]) -> float:
    try:
        return float(row.get("node_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _node_pool_topk_values_from_payload(eval_payload: Mapping[str, Any]) -> list[int]:
    config_payload = eval_payload.get("config", {})
    if isinstance(config_payload, Mapping):
        raw_values = config_payload.get(
            "node_pool_topk_values",
            SlimConfig.node_pool_topk_values,
        )
    else:
        raw_values = SlimConfig.node_pool_topk_values
    return _parse_int_list(str(raw_values))


def _label_name(label: Any) -> str:
    text = str(label).strip()
    if _is_attack_label(label):
        lowered = text.lower()
        if lowered in {"suspicious", "malicious"}:
            return lowered
        return "malicious"
    return "benign"


def _is_attack_label(label: Any) -> bool:
    text = str(label).strip().lower()
    if text in {"1", "2", "suspicious", "malicious", "true"}:
        return True
    if isinstance(label, int | np.integer):
        return int(label) in {1, 2}
    return False


def _peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    scale = 1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0
    return float(usage.ru_maxrss / scale)


def _safe_peak_rss_mb() -> float:
    peak = _peak_rss_mb()
    if peak is None:
        return 0.0
    return float(peak)


def _current_rss_mb() -> float:
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            pages = int(handle.read().split()[1])
        return float(pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024))
    except (OSError, IndexError, ValueError):
        return 0.0


def _safe_state_slots(sspm: Any) -> int:
    try:
        return int(getattr(sspm, "state_slots"))
    except (AttributeError, TypeError, ValueError):
        node_to_slot = getattr(sspm, "node_to_slot", {})
        try:
            return int(len(node_to_slot))
        except TypeError:
            return 0
    return 0


def _safe_state_array_mb(sspm: Any) -> float:
    try:
        return float(getattr(sspm, "state_array_mb"))
    except (AttributeError, TypeError, ValueError):
        state_array = getattr(sspm, "state_array", getattr(sspm, "_state_array", None))
        nbytes = getattr(state_array, "nbytes", 0)
        try:
            return float(nbytes) / (1024.0 * 1024.0)
        except (TypeError, ValueError):
            return 0.0


def _final_memory_profile(
    memory: Mapping[str, Any] | None,
    sspm: SSPMLowRankModel,
) -> dict[str, Any]:
    profile = _empty_memory_profile()
    if memory is not None:
        profile.update(dict(memory))
    profile["state_slots"] = _safe_state_slots(sspm)
    profile["state_array_mb"] = _safe_state_array_mb(sspm)
    if hasattr(sspm, "state_memory_stats"):
        profile["state_memory"] = dict(sspm.state_memory_stats())
    profile.pop("peak_rss_mb", None)
    profile["process_peak_rss_mb"] = _safe_peak_rss_mb()
    if float(profile.get("rss_test_peak_mb", 0.0)) <= 0.0:
        profile["rss_test_peak_mb"] = max(
            float(profile.get("rss_after_test_cleanup_mb", 0.0)),
            _current_rss_mb(),
        )
    profile.setdefault("deploy_rss_valid", False)
    profile.setdefault("deploy_infer_rss_peak_mb", 0.0)
    profile.setdefault("deploy_infer_rss_final_mb", 0.0)
    profile.setdefault("deploy_infer_events_per_sec", 0.0)
    return profile


def _parse_int_list(value: str) -> list[int]:
    values: list[int] = []
    for raw in str(value).split(","):
        text = raw.strip()
        if not text:
            continue
        parsed = int(text)
        if parsed > 0:
            values.append(parsed)
    return values


# Explicit cross-module imports; replaces the migration namespace bridge.
from scripts.pipeline.checks.preflight import _db_stream_status
from scripts.pipeline.features.semantic_features import (
    _empirical_tail_score_compact,
    _observe_process_semantic_audit,
    row_fields,
)
from scripts.pipeline.io.cache_payloads import _compact_sorted_scores, _eval_split_metadata
from scripts.pipeline.io.event_artifacts import (
    _model_calibration_summary,
    _model_update_gate_summary,
)
from scripts.pipeline.outputs.conditional_reports import (
    _conditional_group_summary_with_eval,
    _write_conditional_group_test_summary,
)
from scripts.pipeline.outputs.metrics_summary import (
    _emit_profile_row,
    _empty_memory_profile,
    _empty_timing,
    _encode_row,
    _event_score_from_residual_tail,
    _malicious_pair_audit_row,
    _node_count_snapshot,
    _ofsm_alert_fields,
    _ofsm_empty_node_snapshot,
    _profiling_row,
    _residual_audit_base_row,
    _residual_example_group,
    _residual_feature_fields,
    _state_merge_enabled,
    _state_merge_profile_row,
    _stream_csv_write_seconds,
    _update_node_count_state,
)
from scripts.pipeline.state.online_state_runtime import _stage_log
