"""Active CLI argument parsing, config validation, and top-level dispatch."""

from __future__ import annotations

from scripts.pipeline.config.runtime_config import *


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the slim causal semantics runner."""
    parser = argparse.ArgumentParser(description="Slim causal semantics synthetic smoke runner.")
    parser.add_argument("--dataset", default=SlimConfig.dataset)
    parser.add_argument("--out_tag", default=SlimConfig.out_tag)
    parser.add_argument("--result_root", default=SlimConfig.result_root)
    parser.add_argument("--max_train_events", type=int, default=SlimConfig.max_train_events)
    parser.add_argument("--max_ref_events", type=int, default=SlimConfig.max_ref_events)
    parser.add_argument("--max_test_events", type=int, default=SlimConfig.max_test_events)
    parser.add_argument("--max_tokens_per_node", type=int, default=SlimConfig.max_tokens_per_node)
    parser.add_argument(
        "--expected_event_alert_budget",
        type=float,
        default=SlimConfig.expected_event_alert_budget,
    )
    parser.add_argument(
        "--expected_alert_horizon_events",
        type=int,
        default=SlimConfig.expected_alert_horizon_events,
    )
    parser.add_argument(
        "--event_threshold_mode",
        choices=sorted(EVENT_THRESHOLD_MODES),
        default=SlimConfig.event_threshold_mode,
    )
    parser.add_argument(
        "--event_threshold_quantile",
        type=float,
        default=SlimConfig.event_threshold_quantile,
    )
    parser.add_argument("--latent_dim", type=int, default=SlimConfig.latent_dim)
    parser.add_argument(
        "--latent_dim_policy",
        choices=("fixed", "auto_vocab"),
        default=SlimConfig.latent_dim_policy,
    )
    parser.add_argument("--rank", type=int, default=SlimConfig.rank)
    parser.add_argument("--sspm_batch_size", type=int, default=SlimConfig.sspm_batch_size)
    parser.add_argument("--sspm_learning_rate", type=float, default=SlimConfig.sspm_learning_rate)
    parser.add_argument("--sspm_l2", type=float, default=SlimConfig.sspm_l2)
    parser.add_argument(
        "--sspm_epochs",
        "--sspm_train_epochs",
        dest="sspm_epochs",
        type=int,
        default=SlimConfig.sspm_epochs,
    )
    parser.add_argument(
        "--sspm_early_stop_min_delta",
        type=float,
        default=SlimConfig.sspm_early_stop_min_delta,
    )
    parser.add_argument(
        "--sspm_early_stop_patience",
        type=int,
        default=SlimConfig.sspm_early_stop_patience,
    )
    parser.add_argument(
        "--sspm_train_mode",
        choices=(
            "build_phase3e_artifacts",
            "train_phase3e_base",
            "train_conditional_and_save",
            "load_and_infer",
        ),
        default=SlimConfig.sspm_train_mode,
    )
    parser.add_argument(
        "--sspm_train_data_mode",
        choices=("full_cache", "stream_event", "phase3e_memmap"),
        default=SlimConfig.sspm_train_data_mode,
    )
    parser.add_argument(
        "--sspm_checkpoint_path",
        default=SlimConfig.sspm_checkpoint_path,
    )
    parser.add_argument(
        "--sspm_base_checkpoint_root",
        default=SlimConfig.sspm_base_checkpoint_root,
    )
    parser.add_argument(
        "--sspm_train_batch_events",
        type=int,
        default=SlimConfig.sspm_train_batch_events,
    )
    parser.add_argument(
        "--sspm_target_mode",
        choices=sorted(SSPM_TARGET_MODES),
        default=SlimConfig.sspm_target_mode,
    )
    parser.add_argument(
        "--node_word2vec_source",
        choices=sorted(NODE_WORD2VEC_SOURCES),
        default=SlimConfig.node_word2vec_source,
    )
    parser.add_argument(
        "--node_embedding_lookup_mode",
        choices=sorted(NODE_EMBEDDING_LOOKUP_MODES),
        default=SlimConfig.node_embedding_lookup_mode,
    )
    parser.add_argument(
        "--node_embedding_lazy_backing",
        choices=("global_memmap", "compact_used_nodes"),
        default=SlimConfig.node_embedding_lazy_backing,
    )
    parser.add_argument(
        "--node_embedding_cache_max_nodes",
        type=int,
        default=SlimConfig.node_embedding_cache_max_nodes,
    )
    parser.add_argument(
        "--node_embedding_cache_evict_policy",
        choices=("lru",),
        default=SlimConfig.node_embedding_cache_evict_policy,
    )
    parser.add_argument("--node_embedding_dim", type=int, default=SlimConfig.node_embedding_dim)
    parser.add_argument(
        "--node_embedding_cache_dir",
        default=SlimConfig.node_embedding_cache_dir,
    )
    parser.add_argument(
        "--compact_used_node_cache_dir",
        default=SlimConfig.compact_used_node_cache_dir,
    )
    parser.add_argument(
        "--phase3e_node_lookup_batch_size",
        type=int,
        default=SlimConfig.phase3e_node_lookup_batch_size,
    )
    parser.add_argument(
        "--action_embedding_dim",
        type=int,
        default=SlimConfig.action_embedding_dim,
    )
    parser.add_argument(
        "--action_embedding_cache_dir",
        default=SlimConfig.action_embedding_cache_dir,
    )
    parser.add_argument(
        "--event_index_cache_mode",
        choices=sorted(EVENT_INDEX_CACHE_MODES),
        default=SlimConfig.event_index_cache_mode,
    )
    parser.add_argument("--event_index_cache_dir", default=SlimConfig.event_index_cache_dir)
    parser.add_argument("--event_index_debug_fields", action="store_true")
    parser.add_argument(
        "--sspm_train_backend",
        choices=sorted(SSPM_TRAIN_BACKENDS),
        default=SlimConfig.sspm_train_backend,
    )
    parser.add_argument(
        "--sspm_infer_backend",
        choices=sorted(SSPM_INFER_BACKENDS),
        default=SlimConfig.sspm_infer_backend,
    )
    parser.add_argument(
        "--sspm_score_head",
        choices=sorted(SSPM_SCORE_HEADS),
        default=SlimConfig.sspm_score_head,
    )
    parser.add_argument(
        "--node_repr_fusion",
        choices=sorted(NODE_REPR_FUSIONS),
        default=SlimConfig.node_repr_fusion,
    )
    parser.add_argument(
        "--conditional_semantic_loss",
        choices=("cosine", "mse"),
        default=SlimConfig.conditional_semantic_loss,
    )
    parser.add_argument(
        "--sspm_conditional_head_arch",
        choices=sorted(SSPM_CONDITIONAL_HEAD_ARCHES),
        default=SlimConfig.sspm_conditional_head_arch,
    )
    parser.add_argument(
        "--action_head_checkpoint_path",
        default=SlimConfig.action_head_checkpoint_path,
    )
    parser.add_argument(
        "--action_validation_cache_dir",
        default=SlimConfig.action_validation_cache_dir,
    )
    parser.add_argument(
        "--conditional_group_min_count",
        type=int,
        default=SlimConfig.conditional_group_min_count,
    )
    parser.add_argument(
        "--conditional_low_support_policy",
        choices=("conservative_max", "adaptive_margin"),
        default=SlimConfig.conditional_low_support_policy,
    )
    parser.add_argument(
        "--conditional_low_support_margin",
        type=float,
        default=SlimConfig.conditional_low_support_margin,
    )
    parser.add_argument(
        "--conditional_unseen_group_policy",
        choices=("observation_only",),
        default=SlimConfig.conditional_unseen_group_policy,
    )
    parser.add_argument(
        "--conditional_both_cold_unseen_policy",
        choices=("alert", "observation_only_no_alert"),
        default=SlimConfig.conditional_both_cold_unseen_policy,
    )
    parser.add_argument(
        "--conditional_global_extreme_quantile",
        type=float,
        default=SlimConfig.conditional_global_extreme_quantile,
    )
    parser.add_argument(
        "--conditional_adaptive_margin_n1",
        type=int,
        default=SlimConfig.conditional_adaptive_margin_n1,
    )
    parser.add_argument(
        "--conditional_adaptive_margin_n2",
        type=int,
        default=SlimConfig.conditional_adaptive_margin_n2,
    )
    parser.add_argument(
        "--conditional_adaptive_margin_low",
        type=float,
        default=SlimConfig.conditional_adaptive_margin_low,
    )
    parser.add_argument(
        "--conditional_adaptive_margin_mid",
        type=float,
        default=SlimConfig.conditional_adaptive_margin_mid,
    )
    parser.add_argument(
        "--conditional_adaptive_margin_high",
        type=float,
        default=SlimConfig.conditional_adaptive_margin_high,
    )
    parser.add_argument(
        "--conditional_endpoint_aware_suppression",
        type=_parse_bool,
        default=SlimConfig.conditional_endpoint_aware_suppression,
    )
    parser.add_argument(
        "--conditional_endpoint_suppression_read",
        type=_parse_bool,
        default=SlimConfig.conditional_endpoint_suppression_read,
    )
    parser.add_argument(
        "--conditional_endpoint_suppression_mode",
        choices=("pair_only", "pair_or_same_process_endpoint_history", "pair_then_endpoint", "off"),
        default=SlimConfig.conditional_endpoint_suppression_mode,
    )
    parser.add_argument(
        "--conditional_known_pair_min_count",
        type=int,
        default=SlimConfig.conditional_known_pair_min_count,
    )
    parser.add_argument(
        "--conditional_pair_suppression_margin",
        type=float,
        default=SlimConfig.conditional_pair_suppression_margin,
    )
    parser.add_argument(
        "--conditional_same_process_endpoint_min_count",
        type=int,
        default=SlimConfig.conditional_same_process_endpoint_min_count,
    )
    parser.add_argument(
        "--conditional_same_process_endpoint_margin",
        type=float,
        default=SlimConfig.conditional_same_process_endpoint_margin,
    )
    parser.add_argument(
        "--conditional_endpoint_suppression_summary_mode",
        choices=("online_minimal", "analysis"),
        default=SlimConfig.conditional_endpoint_suppression_summary_mode,
    )
    parser.add_argument(
        "--conditional_endpoint_suppression_cache_dir",
        default=SlimConfig.conditional_endpoint_suppression_cache_dir,
    )
    parser.add_argument(
        "--sspm_conditional_train_data_mode",
        choices=("memmap", "stream_event"),
        default=SlimConfig.sspm_conditional_train_data_mode,
    )
    parser.add_argument(
        "--sspm_conditional_max_epochs",
        type=int,
        default=SlimConfig.sspm_conditional_max_epochs,
    )
    parser.add_argument(
        "--sspm_conditional_e3_max_epochs",
        type=int,
        default=SlimConfig.sspm_conditional_e3_max_epochs,
    )
    parser.add_argument(
        "--phase3g_build_conditional_memmap_only",
        action="store_true",
        help="Build train-only Phase3G conditional memmaps and stop before head training.",
    )
    parser.add_argument(
        "--sspm_torch_batch_events",
        type=int,
        default=SlimConfig.sspm_torch_batch_events,
    )
    parser.add_argument("--sspm_torch_lr", type=float, default=SlimConfig.sspm_torch_lr)
    parser.add_argument(
        "--sspm_torch_weight_decay",
        type=float,
        default=SlimConfig.sspm_torch_weight_decay,
    )
    parser.add_argument("--sspm_torch_device", default=SlimConfig.sspm_torch_device)
    parser.add_argument(
        "--phase3g_backfill_result_dir",
        default="",
        help="Backfill Phase3G RSS and node coverage reports for an existing result dir.",
    )
    parser.add_argument(
        "--sspm_state_model",
        choices=("ema_fixed", "real_diag_learnable", "s4d_complex_node"),
        default=SlimConfig.sspm_state_model,
    )
    parser.add_argument("--sspm_target_dim", type=int, default=SlimConfig.sspm_target_dim)
    parser.add_argument("--sspm_state_dim", type=int, default=SlimConfig.sspm_state_dim)
    parser.add_argument(
        "--sspm_context_mode",
        choices=("with_action", "no_action"),
        default=SlimConfig.sspm_context_mode,
    )
    parser.add_argument(
        "--sspm_global_context_mode",
        choices=("with_global", "no_global"),
        default=SlimConfig.sspm_global_context_mode,
    )
    parser.add_argument(
        "--state_memory_mode",
        choices=("unbounded", "bounded"),
        default=SlimConfig.state_memory_mode,
    )
    parser.add_argument(
        "--sspm_state_memory_policy",
        choices=("legacy", "probationary_lru"),
        default=SlimConfig.sspm_state_memory_policy,
    )
    parser.add_argument(
        "--sspm_active_max_nodes",
        type=int,
        default=SlimConfig.sspm_active_max_nodes,
    )
    parser.add_argument(
        "--sspm_probationary_max_nodes",
        type=int,
        default=SlimConfig.sspm_probationary_max_nodes,
    )
    parser.add_argument(
        "--sspm_probationary_min_count",
        type=int,
        default=SlimConfig.sspm_probationary_min_count,
    )
    parser.add_argument(
        "--sspm_lru_evict_batch",
        type=int,
        default=SlimConfig.sspm_lru_evict_batch,
    )
    parser.add_argument(
        "--sspm_alert_protect_events",
        type=int,
        default=SlimConfig.sspm_alert_protect_events,
    )
    parser.add_argument(
        "--sspm_update_gate_mode",
        choices=("none", "quantile"),
        default=SlimConfig.sspm_update_gate_mode,
    )
    parser.add_argument(
        "--sspm_update_gate_quantile",
        type=float,
        default=SlimConfig.sspm_update_gate_quantile,
    )
    parser.add_argument(
        "--sspm_update_gate_threshold_mode",
        choices=("quantile", "validation_max"),
        default=SlimConfig.sspm_update_gate_threshold_mode,
    )
    parser.add_argument(
        "--sspm_update_gate_score_space",
        choices=sorted(SSPM_UPDATE_GATE_SCORE_SPACES),
        default=SlimConfig.sspm_update_gate_score_space,
    )
    parser.add_argument(
        "--sspm_update_gate_q_min",
        type=float,
        default=SlimConfig.sspm_update_gate_q_min,
    )
    parser.add_argument(
        "--sspm_update_gate_eta",
        type=float,
        default=SlimConfig.sspm_update_gate_eta,
    )
    parser.add_argument(
        "--sspm_score_target_mode",
        choices=sorted(SSPM_SCORE_TARGET_MODES),
        default=SlimConfig.sspm_score_target_mode,
    )
    parser.add_argument(
        "--sspm_residual_score_mode",
        choices=("legacy", "raw_mse", "var_calibrated"),
        default=SlimConfig.sspm_residual_score_mode,
    )
    parser.add_argument(
        "--sspm_residual_calibration",
        choices=("none", "action_diag"),
        default=SlimConfig.sspm_residual_calibration,
    )
    parser.add_argument(
        "--sspm_residual_alpha_cos",
        type=float,
        default=SlimConfig.sspm_residual_alpha_cos,
    )
    parser.add_argument(
        "--sspm_residual_beta_mse",
        type=float,
        default=SlimConfig.sspm_residual_beta_mse,
    )
    parser.add_argument(
        "--sspm_residual_beta_var",
        type=float,
        default=SlimConfig.sspm_residual_beta_var,
    )
    parser.add_argument(
        "--sspm_residual_var_eps",
        type=float,
        default=SlimConfig.sspm_residual_var_eps,
    )
    parser.add_argument(
        "--sspm_residual_calibration_min_count",
        type=int,
        default=SlimConfig.sspm_residual_calibration_min_count,
    )
    parser.add_argument(
        "--real_diag_train_gamma",
        action="store_true",
        default=SlimConfig.real_diag_train_gamma,
    )
    parser.add_argument(
        "--real_diag_gamma_lr",
        type=float,
        default=SlimConfig.real_diag_gamma_lr,
    )
    parser.add_argument(
        "--real_diag_gamma_weight_decay",
        type=float,
        default=SlimConfig.real_diag_gamma_weight_decay,
    )
    parser.add_argument(
        "--real_diag_gamma_grad_clip",
        type=float,
        default=SlimConfig.real_diag_gamma_grad_clip,
    )
    parser.add_argument(
        "--real_diag_sensitivity_mode",
        choices=("online_stop_message",),
        default=SlimConfig.real_diag_sensitivity_mode,
    )
    parser.add_argument(
        "--real_diag_gamma_min",
        type=float,
        default=SlimConfig.real_diag_gamma_min,
    )
    parser.add_argument(
        "--real_diag_gamma_max",
        type=float,
        default=SlimConfig.real_diag_gamma_max,
    )
    parser.add_argument(
        "--real_diag_max_sensitivity_nodes",
        type=int,
        default=SlimConfig.real_diag_max_sensitivity_nodes,
    )
    parser.add_argument(
        "--sspm_state_merge_mode",
        choices=("none", "online_fourier", "online_time_domain", "random"),
        default=SlimConfig.sspm_state_merge_mode,
    )
    parser.add_argument(
        "--sspm_state_merge_scope",
        choices=("same_type",),
        default=SlimConfig.sspm_state_merge_scope,
    )
    parser.add_argument(
        "--sspm_state_merge_threshold",
        type=float,
        default=SlimConfig.sspm_state_merge_threshold,
    )
    parser.add_argument(
        "--sspm_state_merge_trunc_ratio",
        type=float,
        default=SlimConfig.sspm_state_merge_trunc_ratio,
    )
    parser.add_argument(
        "--sspm_state_merge_use_rfft",
        type=_parse_bool,
        default=SlimConfig.sspm_state_merge_use_rfft,
    )
    parser.add_argument(
        "--sspm_state_merge_use_real_only",
        type=_parse_bool,
        default=SlimConfig.sspm_state_merge_use_real_only,
    )
    parser.add_argument(
        "--sspm_state_merge_center_state",
        type=_parse_bool,
        default=SlimConfig.sspm_state_merge_center_state,
    )
    parser.add_argument(
        "--sspm_state_merge_normalize",
        choices=("l2",),
        default=SlimConfig.sspm_state_merge_normalize,
    )
    parser.add_argument(
        "--sspm_state_merge_eps",
        type=float,
        default=SlimConfig.sspm_state_merge_eps,
    )
    parser.add_argument(
        "--sspm_state_merge_min_cluster_size",
        type=int,
        default=SlimConfig.sspm_state_merge_min_cluster_size,
    )
    parser.add_argument(
        "--sspm_state_merge_copy_on_write",
        type=_parse_bool,
        default=SlimConfig.sspm_state_merge_copy_on_write,
    )
    parser.add_argument(
        "--sspm_state_merge_random_prob",
        type=float,
        default=SlimConfig.sspm_state_merge_random_prob,
    )
    parser.add_argument(
        "--sspm_state_merge_random_seed",
        type=int,
        default=SlimConfig.sspm_state_merge_random_seed,
    )
    parser.add_argument(
        "--sspm_state_merge_diagnostics_max_rows",
        type=int,
        default=SlimConfig.sspm_state_merge_diagnostics_max_rows,
    )
    parser.add_argument(
        "--sspm_ofsm_match_backend",
        choices=("exact", "bucketed"),
        default=SlimConfig.sspm_ofsm_match_backend,
    )
    parser.add_argument(
        "--sspm_ofsm_candidate_cap",
        type=int,
        default=SlimConfig.sspm_ofsm_candidate_cap,
    )
    parser.add_argument(
        "--sspm_ofsm_merge_interval",
        type=int,
        default=SlimConfig.sspm_ofsm_merge_interval,
    )
    parser.add_argument(
        "--sspm_ofsm_min_count",
        type=int,
        default=SlimConfig.sspm_ofsm_min_count,
    )
    parser.add_argument(
        "--sspm_ofsm_diagnostics",
        type=_parse_bool,
        default=SlimConfig.sspm_ofsm_diagnostics,
    )
    parser.add_argument("--max_exact_states", type=int, default=SlimConfig.max_exact_states)
    parser.add_argument(
        "--min_inactive_events",
        type=int,
        default=SlimConfig.min_inactive_events,
    )
    parser.add_argument("--prototype_count", type=int, default=SlimConfig.prototype_count)
    parser.add_argument(
        "--prototype_update_alpha",
        type=float,
        default=SlimConfig.prototype_update_alpha,
    )
    parser.add_argument("--action_count", type=int, default=SlimConfig.action_count)
    parser.add_argument(
        "--max_recent_events_per_node",
        type=int,
        default=SlimConfig.max_recent_events_per_node,
    )
    parser.add_argument("--fetch_size", type=int, default=SlimConfig.fetch_size)
    parser.add_argument("--print_summary", action="store_true")
    parser.add_argument(
        "--rss_profile_mode",
        choices=("analysis", "deploy_light", "online_minimal"),
        default=SlimConfig.rss_profile_mode,
    )
    parser.add_argument(
        "--sspm_infer_fast_path",
        type=_parse_bool,
        default=SlimConfig.sspm_infer_fast_path,
    )
    parser.add_argument(
        "--sspm_infer_chunk_events",
        type=int,
        default=SlimConfig.sspm_infer_chunk_events,
    )
    parser.add_argument(
        "--write_raw_alerts",
        type=_parse_bool,
        default=SlimConfig.write_raw_alerts,
    )
    parser.add_argument(
        "--write_analysis_outputs",
        type=_parse_bool,
        default=SlimConfig.write_analysis_outputs,
    )
    parser.add_argument(
        "--sspm_state_merge_write_diagnostics",
        type=_parse_bool,
        default=SlimConfig.sspm_state_merge_write_diagnostics,
    )
    parser.add_argument(
        "--pretrained_residual_embedder_path",
        default=SlimConfig.pretrained_residual_embedder_path,
    )
    parser.add_argument(
        "--residual_embed_cache_mode",
        choices=("off", "auto"),
        default=SlimConfig.residual_embed_cache_mode,
    )
    parser.add_argument(
        "--residual_embed_cache_path",
        default=SlimConfig.residual_embed_cache_path,
    )
    parser.add_argument("--precompute_cache_dir", default=SlimConfig.precompute_cache_dir)
    parser.add_argument(
        "--precompute_cache_mode",
        default="off",
        choices=("off", "read", "write", "auto", "force"),
        help="Use train/validation-only precompute cache artifacts.",
    )
    parser.add_argument(
        "--progress_interval_events",
        type=int,
        default=SlimConfig.progress_interval_events,
    )
    parser.add_argument(
        "--db_stream_mode",
        choices=sorted(DB_STREAM_MODES),
        default=SlimConfig.db_stream_mode,
    )
    parser.add_argument(
        "--event_score_mode",
        choices=sorted(EVENT_SCORE_MODES | ARCHIVED_EVENT_SCORE_MODES),
        default=SlimConfig.event_score_mode,
    )
    parser.add_argument(
        "--semantic_embedding_method",
        choices=("word2vec",),
        default=SlimConfig.semantic_embedding_method,
    )
    parser.add_argument("--semantic_mode", default=SlimConfig.semantic_mode)
    parser.add_argument("--word2vec_window", type=int, default=SlimConfig.word2vec_window)
    parser.add_argument("--word2vec_min_count", type=int, default=SlimConfig.word2vec_min_count)
    parser.add_argument("--word2vec_sg", type=int, default=SlimConfig.word2vec_sg)
    parser.add_argument("--word2vec_negative", type=int, default=SlimConfig.word2vec_negative)
    parser.add_argument("--word2vec_epochs", type=int, default=SlimConfig.word2vec_epochs)
    parser.add_argument("--word2vec_workers", type=int, default=SlimConfig.word2vec_workers)
    parser.add_argument("--word2vec_seed", type=int, default=SlimConfig.word2vec_seed)
    parser.add_argument(
        "--word2vec_oov_policy",
        choices=("unk", "zero"),
        default=SlimConfig.word2vec_oov_policy,
    )
    parser.add_argument(
        "--theia_netflow_policy",
        choices=("scope_port", "fixed"),
        default=SlimConfig.theia_netflow_policy,
    )
    parser.add_argument(
        "--action_type_alert_policy",
        choices=sorted(ACTION_TYPE_ALERT_POLICIES),
        default=SlimConfig.action_type_alert_policy,
    )
    parser.add_argument(
        "--node_pool_score_mode",
        choices=sorted(NODE_POOL_SCORE_MODES),
        default=SlimConfig.node_pool_score_mode,
    )
    parser.add_argument(
        "--node_pool_topk_values",
        default=SlimConfig.node_pool_topk_values,
    )
    parser.add_argument(
        "--clearscope_v31_fp_guard_write_floor",
        type=float,
        default=SlimConfig.clearscope_v31_fp_guard_write_floor,
    )
    parser.add_argument(
        "--clearscope_v31_fp_guard_read_floor",
        type=float,
        default=SlimConfig.clearscope_v31_fp_guard_read_floor,
    )
    parser.add_argument("--slim_split_override", action="store_true")
    parser.add_argument("--compat_in_memory_outputs", action="store_true")
    parser.add_argument(
        "--process_semantics_config",
        default=SlimConfig.process_semantics_config,
    )
    parser.add_argument("--process_semantic_audit", action="store_true")
    parser.add_argument("--write_legacy_eval_alias", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SlimConfig:
    """Build a SlimConfig from parsed CLI arguments."""
    values = {
        field: getattr(args, field)
        for field in SlimConfig.__dataclass_fields__
        if hasattr(args, field)
    }
    config = SlimConfig(**values)
    _validate_active_event_score_mode(config.event_score_mode)
    _validate_phase3e_config(config)
    if str(config.sspm_train_mode) == "load_and_infer" and not str(
        config.sspm_checkpoint_path,
    ).strip():
        raise ValueError("SSPM_TRAIN_MODE=load_and_infer requires --sspm_checkpoint_path")
    if str(config.sspm_train_data_mode) == "stream_event" and not _embedder_loaded(config):
        raise ValueError("SSPM_TRAIN_DATA_MODE=stream_event requires pretrained embedder")
    validate_word2vec_semantic_mode_matches_config(config)
    return config


def validate_config(config: SlimConfig) -> None:
    """Validate config values without requiring runtime artifacts."""
    _validate_active_event_score_mode(config.event_score_mode)
    _validate_phase3e_config(config)
    validate_word2vec_semantic_mode_matches_config(config)


def build_phase3e_artifacts_from_db(config: SlimConfig) -> Path:
    """Dispatch to the Phase3E artifact builder when that implementation is available."""
    from scripts.pipeline.io.event_artifacts import (
        build_phase3e_artifacts_from_db as _build_phase3e_artifacts_from_db,
    )

    return _build_phase3e_artifacts_from_db(config)


def _validate_phase3e_config(config: SlimConfig) -> None:
    """Validate the Phase3E config surface without launching DB precompute."""
    if str(config.sspm_score_head) not in SSPM_SCORE_HEADS:
        raise ValueError(f"unsupported SSPM score head: {config.sspm_score_head}")
    if str(config.node_repr_fusion) not in NODE_REPR_FUSIONS:
        raise ValueError(f"unsupported NODE_REPR_FUSION: {config.node_repr_fusion}")
    if str(config.sspm_update_gate_score_space) not in SSPM_UPDATE_GATE_SCORE_SPACES:
        raise ValueError(
            "SSPM_UPDATE_GATE_SCORE_SPACE must be checkpoint or conditional_event_score",
        )
    if str(config.sspm_score_target_mode) not in SSPM_SCORE_TARGET_MODES:
        raise ValueError(
            "SSPM_SCORE_TARGET_MODE must be event_action_semantic or node_pair_no_action",
        )
    if str(config.sspm_conditional_head_arch) not in SSPM_CONDITIONAL_HEAD_ARCHES:
        raise ValueError(
            "SSPM_CONDITIONAL_HEAD_ARCH must be shared_lowrank_v1 or "
            "dual_lowrank_by_target_case_v2",
        )
    if (
        str(config.sspm_conditional_head_arch) == CONDITIONAL_HEAD_ARCH_DUAL_V2
        and str(config.sspm_score_head) != "conditional_action_semantic"
    ):
        raise ValueError(
            "SSPM_CONDITIONAL_HEAD_ARCH=dual_lowrank_by_target_case_v2 requires "
            "conditional_action_semantic",
        )
    if str(config.action_type_alert_policy) not in ACTION_TYPE_ALERT_POLICIES:
        raise ValueError(
            "ACTION_TYPE_ALERT_POLICY must be default, theia_v1, "
            "cadets_e4_v2_group_v1, clearscope_node_pair_v1, "
            "clearscope_android_v2, clearscope_v31_fp_guard_v1, "
            "clearscope_v31_fp_guard_v2, clearscope_v31_fp_guard_v3, "
            "clearscope_v31_fp_guard_v3b, or clearscope_v31_fp_guard_v3c",
        )
    if (
        str(config.sspm_update_gate_score_space) == "conditional_event_score"
        and str(config.sspm_score_head) != "conditional_action_semantic"
    ):
        raise ValueError(
            "SSPM_UPDATE_GATE_SCORE_SPACE=conditional_event_score requires "
            "conditional_action_semantic",
        )
    if (
        str(config.sspm_score_target_mode) == SCORE_TARGET_NODE_PAIR
        and str(config.sspm_score_head) != "conditional_action_semantic"
    ):
        raise ValueError(
            "SSPM_SCORE_TARGET_MODE=node_pair_no_action requires conditional_action_semantic",
        )
    if str(config.sspm_score_head) == "conditional_action_semantic":
        if str(config.sspm_target_mode) != "node_action_semantic_mean":
            raise ValueError(
                f"{config.sspm_score_head} requires SSPM_TARGET_MODE=node_action_semantic_mean",
            )
        if str(config.node_repr_fusion) != "simple_mean":
            raise ValueError(f"{config.sspm_score_head} v1 requires NODE_REPR_FUSION=simple_mean")
        if int(config.rank) <= 0:
            raise ValueError(f"{config.sspm_score_head} requires positive rank")
    if str(config.sspm_score_head) == "conditional_action_semantic":
        if str(config.conditional_semantic_loss) not in {"cosine", "mse"}:
            raise ValueError("CONDITIONAL_SEMANTIC_LOSS must be cosine or mse")
        allowed_conditional_modes = {
            "target_case_quantile",
            "quantile",
            CONDITIONAL_GROUP_THRESHOLD_MODE,
        }
        if str(config.event_threshold_mode) not in allowed_conditional_modes:
            raise ValueError(
                f"{config.sspm_score_head} does not support event_threshold_mode="
                f"{config.event_threshold_mode}",
            )
        if int(config.conditional_group_min_count) <= 0:
            raise ValueError("conditional_group_min_count must be positive")
        if str(config.conditional_low_support_policy) != "conservative_max":
            if str(config.conditional_low_support_policy) != "adaptive_margin":
                raise ValueError(
                    "conditional_low_support_policy must be conservative_max or adaptive_margin",
                )
        if float(config.conditional_low_support_margin) < 0.0:
            raise ValueError("conditional_low_support_margin must be non-negative")
        if str(config.conditional_unseen_group_policy) != "observation_only":
            raise ValueError("conditional_unseen_group_policy must be observation_only")
        if str(config.conditional_both_cold_unseen_policy) not in {
            "alert",
            "observation_only_no_alert",
        }:
            raise ValueError(
                "conditional_both_cold_unseen_policy must be alert or "
                "observation_only_no_alert",
            )
        if not (0.0 < float(config.conditional_global_extreme_quantile) <= 1.0):
            raise ValueError("conditional_global_extreme_quantile must be in (0, 1]")
        if (
            int(config.conditional_adaptive_margin_n1) <= 0
            or int(config.conditional_adaptive_margin_n2)
            <= int(config.conditional_adaptive_margin_n1)
        ):
            raise ValueError("conditional adaptive margin requires 0 < n1 < n2")
        if int(config.conditional_known_pair_min_count) <= 0:
            raise ValueError("conditional_known_pair_min_count must be positive")
        if float(config.conditional_pair_suppression_margin) < 0.0:
            raise ValueError("conditional_pair_suppression_margin must be non-negative")
        if str(config.conditional_endpoint_suppression_mode) not in {
            "pair_only",
            "pair_or_same_process_endpoint_history",
            "pair_then_endpoint",
            "off",
        }:
            raise ValueError(
                "CONDITIONAL_ENDPOINT_SUPPRESSION_MODE must be pair_only, "
                "pair_or_same_process_endpoint_history, pair_then_endpoint, or off",
            )
        if int(config.conditional_same_process_endpoint_min_count) <= 0:
            raise ValueError("conditional_same_process_endpoint_min_count must be positive")
        if float(config.conditional_same_process_endpoint_margin) < 0.0:
            raise ValueError("conditional_same_process_endpoint_margin must be non-negative")
        if str(config.sspm_conditional_train_data_mode) not in {"memmap", "stream_event"}:
            raise ValueError("SSPM_CONDITIONAL_TRAIN_DATA_MODE must be memmap or stream_event")
        if int(config.sspm_conditional_max_epochs) <= 0:
            raise ValueError("SSPM_CONDITIONAL_MAX_EPOCHS must be positive")
        if int(config.sspm_conditional_e3_max_epochs) <= 0:
            raise ValueError("SSPM_CONDITIONAL_E3_MAX_EPOCHS must be positive")
    if bool(config.sspm_infer_fast_path):
        if str(config.sspm_train_mode) != "load_and_infer":
            raise ValueError("SSPM_INFER_FAST_PATH requires SSPM_TRAIN_MODE=load_and_infer")
        if str(config.rss_profile_mode) != "online_minimal":
            raise ValueError("SSPM_INFER_FAST_PATH requires RSS_PROFILE_MODE=online_minimal")
        if str(config.sspm_state_model) not in {"ema_fixed", "s4d_complex_node"}:
            raise ValueError(
                "Phase3F/Phase3G fast path supports only ema_fixed or s4d_complex_node",
            )
        if (
            str(config.sspm_state_model) == "s4d_complex_node"
            and str(config.sspm_score_head) != "conditional_action_semantic"
        ):
            raise ValueError(
                "s4d_complex_node fast path is supported only for conditional_action_semantic",
            )
        if (
            str(config.sspm_state_merge_mode) != "none"
            and str(config.sspm_score_head) != "conditional_action_semantic"
        ):
            raise ValueError(
                "Phase3F fast path supports OFSM only for conditional_action_semantic",
            )
        if str(config.sspm_state_merge_mode) != "none":
            if str(config.state_memory_mode) != "bounded" or (
                str(config.sspm_state_memory_policy) != "probationary_lru"
            ):
                raise ValueError(
                    "OFSM fast path requires state_memory_mode=bounded and "
                    "sspm_state_memory_policy=probationary_lru",
                )
            if str(config.sspm_ofsm_match_backend) not in {"exact", "bucketed"}:
                raise ValueError("SSPM_OFSM_MATCH_BACKEND must be exact or bucketed")
            if int(config.sspm_ofsm_candidate_cap) <= 0:
                raise ValueError("SSPM_OFSM_CANDIDATE_CAP must be positive")
            if int(config.sspm_ofsm_merge_interval) <= 0:
                raise ValueError("SSPM_OFSM_MERGE_INTERVAL must be positive")
            if int(config.sspm_ofsm_min_count) <= 0:
                raise ValueError("SSPM_OFSM_MIN_COUNT must be positive")
        if str(config.sspm_update_gate_mode) != "none":
            if str(config.sspm_score_head) != "conditional_action_semantic":
                raise ValueError(
                    "update-gate fast path is supported only for conditional_action_semantic",
                )
            if str(config.sspm_state_model) != "ema_fixed":
                raise ValueError("update-gate fast path currently requires ema_fixed")
        if (
            str(config.sspm_score_head)
            not in {
                "conditional_action_semantic",
            }
            and str(config.sspm_residual_score_mode) != "legacy"
        ):
            raise ValueError("Phase3F fast path v1 supports only legacy residual scoring")
        if int(config.sspm_infer_chunk_events) <= 0:
            raise ValueError("SSPM_INFER_CHUNK_EVENTS must be positive")
    if str(config.sspm_target_mode) not in SSPM_TARGET_MODES:
        raise ValueError(f"unsupported SSPM target mode: {config.sspm_target_mode}")
    if str(config.node_word2vec_source) not in NODE_WORD2VEC_SOURCES:
        raise ValueError(f"unsupported node_word2vec_source: {config.node_word2vec_source}")
    if str(config.node_embedding_lookup_mode) not in NODE_EMBEDDING_LOOKUP_MODES:
        raise ValueError(
            f"unsupported NODE_EMBEDDING_LOOKUP_MODE: {config.node_embedding_lookup_mode}",
        )
    if str(config.node_embedding_lazy_backing) not in {"global_memmap", "compact_used_nodes"}:
        raise ValueError("NODE_EMBEDDING_LAZY_BACKING must be global_memmap or compact_used_nodes")
    if int(config.node_embedding_cache_max_nodes) <= 0:
        raise ValueError("NODE_EMBEDDING_CACHE_MAX_NODES must be positive")
    if str(config.node_embedding_cache_evict_policy) != "lru":
        raise ValueError("NODE_EMBEDDING_CACHE_EVICT_POLICY must be lru")
    if str(config.sspm_target_mode) != "node_action_semantic_mean":
        if str(config.sspm_train_backend) not in SSPM_TRAIN_BACKENDS:
            raise ValueError(f"unsupported SSPM train backend: {config.sspm_train_backend}")
        if str(config.sspm_infer_backend) not in SSPM_INFER_BACKENDS:
            raise ValueError(f"unsupported SSPM infer backend: {config.sspm_infer_backend}")
        if str(config.event_index_cache_mode) not in EVENT_INDEX_CACHE_MODES:
            raise ValueError(
                f"unsupported event_index_cache_mode: {config.event_index_cache_mode}",
            )
        return
    if str(config.node_word2vec_source) != "residual_pretrained":
        raise ValueError(
            "Phase3E v1 supports NODE_WORD2VEC_SOURCE=residual_pretrained for "
            "node_action_semantic_mean",
        )
    if not str(config.pretrained_residual_embedder_path).strip():
        raise ValueError(
            "node_action_semantic_mean requires --pretrained_residual_embedder_path",
        )
    if int(config.node_embedding_dim) != int(config.sspm_target_dim):
        raise ValueError("node_embedding_dim must match SSPM target dim")
    if int(config.action_embedding_dim) != int(config.sspm_target_dim):
        raise ValueError("action_embedding_dim must match SSPM target dim")
    if str(config.sspm_infer_backend) != "numpy":
        raise ValueError("node_action_semantic_mean requires sspm_infer_backend=numpy")
    if str(config.sspm_train_backend) not in SSPM_TRAIN_BACKENDS:
        raise ValueError(f"unsupported SSPM train backend: {config.sspm_train_backend}")
    if str(config.event_index_cache_mode) not in EVENT_INDEX_CACHE_MODES:
        raise ValueError(f"unsupported event_index_cache_mode: {config.event_index_cache_mode}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested slim causal semantics pipeline."""
    args = parse_args(argv)
    config = config_from_args(args)
    if str(config.sspm_score_head) != "conditional_action_semantic":
        raise ValueError("active CLI supports only conditional_action_semantic scoring")
    if str(getattr(args, "phase3g_backfill_result_dir", "")).strip():
        eval_path = _phase3g_backfill_result_reports_from_config(
            result_dir=Path(str(args.phase3g_backfill_result_dir)),
            config=config,
        )
    elif str(config.sspm_train_mode) == "build_phase3e_artifacts":
        eval_path = build_phase3e_artifacts_from_db(config)
    elif (
        str(config.sspm_target_mode) == "node_action_semantic_mean"
        and str(config.sspm_train_mode) == "train_phase3e_base"
    ):
        eval_path = run_phase3e_train_base_from_precompute(config)
    elif (
        str(config.sspm_target_mode) == "node_action_semantic_mean"
        and str(config.sspm_train_mode) == "train_conditional_and_save"
    ):
        eval_path = run_phase3g_conditional_train_from_precompute(config)
    elif (
        str(config.sspm_target_mode) == "node_action_semantic_mean"
        and str(config.sspm_train_mode) == "load_and_infer"
    ):
        eval_path = run_phase3e_load_and_infer_from_precompute(config)
    else:
        raise ValueError("active CLI supports only E4 conditional train/memmap or load-and-infer")
    if config.print_summary:
        _compact_print_summary(eval_path)
    return 0


# Explicit active imports; old synthetic/DB compatibility dispatch moved to legacy.
from scripts.pipeline.conditional.infer import run_phase3e_load_and_infer_from_precompute
from scripts.pipeline.io.event_artifacts import (
    _embedder_loaded,
    run_phase3e_train_base_from_precompute,
)
from scripts.pipeline.conditional.train import (
    run_phase3g_conditional_train_from_precompute,
)
from scripts.pipeline.outputs.conditional_reports import (
    _phase3g_backfill_result_reports_from_config,
)
from scripts.pipeline.outputs.metrics_summary import (
    _compact_print_summary,
    _validate_active_event_score_mode,
)
from scripts.pipeline.state.online_state_runtime import (
    _parse_bool,
    validate_word2vec_semantic_mode_matches_config,
)
