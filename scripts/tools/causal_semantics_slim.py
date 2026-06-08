from __future__ import annotations

import argparse
import csv
import bisect
import ctypes
import gc
import hashlib
import json
import math
import os
import pickle
import re
import sys
import threading
import time
import zlib
from collections import Counter, OrderedDict
from contextlib import ExitStack
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

try:
    import psycopg2
except ModuleNotFoundError:  # pragma: no cover - optional until DB mode runs
    psycopg2 = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.utils.common import ENTITY_TYPES, RELATIONS, information_flow, stable_hash
from cs4m.semantics.clearscope_android import (
    CLEARSCOPE_LEGACY_SEMANTIC_MODE,
    CLEARSCOPE_REFINED_SEMANTIC_MODE,
    CLEARSCOPE_SEMANTIC_RULES_VERSION,
    android_process_natural_tokens,
    android_process_natural_tokens_refined,
    clearscope_residual_text_refined,
    clearscope_file_nll_role,
    clearscope_file_natural_tokens,
    clearscope_file_natural_tokens_refined,
    clearscope_file_residual_text,
    clearscope_netflow_nll_role,
    clearscope_netflow_natural_tokens,
    clearscope_netflow_natural_tokens_refined,
    clearscope_netflow_residual_text,
    is_clearscope_dataset,
    normalize_clearscope_semantic_mode,
)
from cs4m.semantics.cadets_freebsd import (
    CADETS_SEMANTIC_RULES_VERSION,
    freebsd_file_natural_tokens,
    freebsd_file_nll_role,
    freebsd_netflow_natural_tokens,
    freebsd_netflow_nll_role,
    freebsd_process_natural_tokens,
    is_cadets_dataset,
)
from cs4m.models.cs4m_lowrank import (
    DEFAULT_TAU,
    SSPMLowRankConfig,
    SSPMLowRankModel,
    normalize_vector,
)
from cs4m.phase3e.event_index import EVENT_INDEX_DTYPE
from cs4m.phase3e.event_index import (
    event_index_fingerprint,
    open_event_index_memmap,
)
from cs4m.phase3g.compact_node_embeddings import (
    build_compact_used_node_artifacts,
    embedding_memory_breakdown,
    load_compact_used_node_artifacts,
    source_event_index_fingerprints,
    source_node_embedding_fingerprint,
)
from cs4m.phase3e.node_action_tables import (
    ActionEmbeddingTable,
    ORTHRUS10_ACTION_NAMES,
    NodeEmbeddingTable,
    build_action_embedding_table,
    build_node_action_coverage_audit,
    build_node_embedding_table,
    compute_node_action_target,
    save_action_embedding_table,
    save_node_action_coverage_audit,
    save_node_embedding_table,
)
from cs4m.phase3e.head_training import train_lowrank_head_torch
from cs4m.phase3e.word2vec_adapter import ResidualWord2VecTokenAdapter
from cs4m.phase3e.context_memmap import build_x_context_memmap, x_context_fingerprint
from cs4m.phase3g.conditional_head import (
    BOTH_COLD_ACTION_TARGET,
    BOTH_COLD_ACTION_TARGET_ID,
    CONDITIONAL_HEAD_ARCH_DUAL_V2,
    CONDITIONAL_HEAD_ARCH_SHARED_V1,
    CONDITIONAL_ACTION_EMBEDDING_SCORE_HEAD,
    CONDITIONAL_ADAPTIVE_MARGIN_HIGH_DEFAULT,
    CONDITIONAL_ADAPTIVE_MARGIN_LOW_DEFAULT,
    CONDITIONAL_ADAPTIVE_MARGIN_MID_DEFAULT,
    CONDITIONAL_ADAPTIVE_MARGIN_N1_DEFAULT,
    CONDITIONAL_ADAPTIVE_MARGIN_N2_DEFAULT,
    CONDITIONAL_GLOBAL_EXTREME_QUANTILE_DEFAULT,
    CONDITIONAL_GROUP_SUMMARY_FIELDS,
    CONDITIONAL_GROUP_THRESHOLD_MODE,
    CONDITIONAL_LOW_SUPPORT_MARGIN_DEFAULT,
    CONDITIONAL_LOW_SUPPORT_POLICY_DEFAULT,
    CONDITIONAL_UNSEEN_GROUP_POLICY_DEFAULT,
    GROUP_LEVEL_TARGET_ACTION_SRC_DST,
    ConditionalSemanticHead,
    ConditionalSemanticHeadConfig,
    EVENT_SEMANTIC_TARGET,
    EVENT_SEMANTIC_TARGET_ID,
    TARGET_CASE_ID_TO_NAME,
    TARGET_CASE_NAME_TO_ID,
    conditional_case_loss_summary,
    conditional_distance,
    decode_conditional_group_key,
    encode_conditional_group_key,
    fit_conditional_group_thresholds,
    make_conditional_group_key,
    load_action_embedding_validation_cache,
    load_conditional_validation_cache,
    quantile_threshold as conditional_quantile_threshold,
    resolve_conditional_group_threshold,
    score_summary as conditional_score_summary,
    select_action_embedding_target,
    select_conditional_target,
    target_case_id,
    write_action_embedding_validation_cache,
    write_conditional_validation_cache,
)
from cs4m.scoring.target_builder import (
    SCORE_TARGET_EVENT_ACTION,
    SCORE_TARGET_NODE_PAIR,
    node_pair_no_action_target,
)
from cs4m.scoring.simple_gates import lambda_rho_for_event, raw_action_one_hot
from cs4m.semantics.semantic_router import (
    ProcessSemanticAudit,
    ProcessSemanticConfig,
    load_process_semantic_config,
    normalize_process_semantics,
    resolve_dataset_profile,
)
from cs4m.embeddings.residual import (
    ResidualEmbedder,
    ResidualEmbeddingConfig,
    build_residual_embedder,
    residual_embedder_from_state_dict,
)
from cs4m.semantics.residual_tokens import netflow_nll_role, residual_text_tokens
from cs4m.scoring.calibration import ResidualCalibrationStats, UpdateGateCalibrator
from cs4m.utils.profiling import SSPMProfiler
from cs4m.utils.residual_embed_cache import (
    ResidualEmbeddingSqliteCache,
    residual_embedding_cache_fingerprint,
)
from cs4m.semantics.theia_linux import (
    THEIA_SEMANTIC_RULES_VERSION,
    is_theia_dataset,
    linux_file_natural_tokens,
    linux_file_nll_role,
    linux_netflow_detail_or_fixed_natural_tokens,
    linux_netflow_fixed_natural_tokens,
    linux_netflow_fixed_nll_role,
    linux_netflow_natural_tokens,
    linux_netflow_nll_role,
    linux_process_natural_tokens,
)
from cs4m.semantics.residual_tokens import netflow_residual_text as _netflow_residual_text
from cs4m.config.provnet_utils import (
    PROCESS_NODE_TYPE,
    SUBJECT_NODE_TABLE,
    init_database_connection,
)
from scripts.data.get_dataset import (
    NODE_TYPE_TOKENS,
    ORTHRUS10_EVENT_TYPES,
    build_hash_to_type,
    build_hash_uuid_index_map,
    build_uuid_index_map,
    fetch_node_tables,
    get_dataset_splits,
    get_ground_truth_paths,
    parse_split_days,
    resolve_gt_path,
    use_event_type_filter,
)
from scripts.tools.db_stream_utils import (
    _build_index_summaries,
    _cfg_for_dataset,
    _day_bounds,
    _stream_events,
)


CACHE_SCHEMA_VERSION = "causal_semantics_slim_cache_v1"
SSPM_CHECKPOINT_SCHEMA_VERSION = "sspm_phase3d_checkpoint_v1"
METHOD_VERSION = "causal_semantics_slim_sspm_v5_cadets_natural_surface_residual"
NETWORK_SEMANTIC_RULES_VERSION = "slim_netflow_scope_port_v2_clearscope_fixed"
SSPM_LOSS_VERSION = "normalized_cosine_plus_lambda_mse"
TAIL_NORMALIZATION_VERSION = "empirical_ge_count_plus_one"
ENTITY_TYPE_NAMES = {int(value): str(key) for key, value in ENTITY_TYPES.items()}
ACTION_FAMILIES = {
    "read": 0,
    "recv": 1,
    "write": 2,
    "send": 3,
    "spawn": 4,
    "open": 5,
    "fork": 6,
    "close": 7,
    "boot": 8,
    "seek": 9,
    "principal": 10,
    "modify": 11,
    "other": 12,
}
DB_STREAM_MODES = {"auto", "temp_table", "python_lookup"}
DB_EXCEPTION_TYPES = (psycopg2.Error,) if psycopg2 is not None else ()
DB_STREAM_ACTUAL_MODES = {
    "not_applicable",
    "python_lookup",
    "python_lookup_fallback",
    "temp_table",
}
PHASE3E_EVENT_ORDER_BY = ("timestamp_rec", "event_uuid")
EVENT_SCORE_MODES = {"res_only"}
ARCHIVED_EVENT_SCORE_MODES = {"sem_only", "sem_res"}
ARCHIVED_SEMANTIC_NLL_MESSAGE = (
    "semantic NLL scoring modes are archived and no longer supported in active SSPM "
    "pipeline. Use event_score_mode=res_only. Legacy code is under "
    "archive/semantic_nll_legacy/."
)
EVENT_THRESHOLD_MODES = {
    "adaptive_rate",
    "budget",
    "quantile",
    "max",
    "validation_max",
    "target_case_quantile",
    CONDITIONAL_GROUP_THRESHOLD_MODE,
}
SSPM_UPDATE_GATE_SCORE_SPACES = {"checkpoint", "conditional_event_score"}
SSPM_SCORE_TARGET_MODES = {SCORE_TARGET_EVENT_ACTION, SCORE_TARGET_NODE_PAIR}
ACTION_TYPE_ALERT_POLICIES = {
    "default",
    "theia_v1",
    "cadets_e4_v2_group_v1",
    "clearscope_node_pair_v1",
    "clearscope_android_v2",
}
SSPM_CONDITIONAL_HEAD_ARCHES = {
    CONDITIONAL_HEAD_ARCH_SHARED_V1,
    CONDITIONAL_HEAD_ARCH_DUAL_V2,
}
NODE_POOL_SCORE_MODES = {
    "base",
    "base_conf",
    "base_conf_repeat",
    "base_conf_chain",
    "base_conf_residual",
    "base_conf_repeat_chain",
    "base_conf_residual_repeat_chain",
    "pool_base",
    "pool_base_adaptive",
    "pool_base_assoc",
    "pool_base_chain",
    "pool_base_adaptive_assoc_chain",
}
SSPM_TARGET_MODES = {"event_residual", "node_action_semantic_mean"}
NODE_WORD2VEC_SOURCES = {"residual_pretrained", "train_node_new"}
NODE_EMBEDDING_LOOKUP_MODES = {"global_memmap", "compact_used_nodes", "lazy_mmap_lru"}
SSPM_TRAIN_BACKENDS = {"numpy", "torch", "torch_cpu"}
SSPM_INFER_BACKENDS = {"numpy"}
SSPM_SCORE_HEADS = {"semantic_residual", "conditional_action_semantic"}
NODE_REPR_FUSIONS = {"simple_mean"}
EVENT_INDEX_CACHE_MODES = {"off", "auto", "force"}
_PHASE3E_CHECKPOINT_FORBIDDEN_KEYS = {
    "node_embeddings",
    "action_embeddings",
    "event_index",
    "x_context",
    "X_context",
    "train_state_memory",
    "validation_state_memory",
    "test_state_memory",
    "probationary_memory",
    "ofsm_clusters",
    "OFSM_clusters",
    "node_pool",
    "rows",
    "tokens",
    "raw_rows",
    "raw_text",
    "residual_tokens",
    "train_rows",
    "train_sentences",
    "train_items",
}
_PHASE3E_METADATA_ALLOWED_KEYS = {
    "target_mode",
    "node_word2vec_source",
    "word2vec_model_path",
    "word2vec_fingerprint",
    "node_embedding_fingerprint",
    "action_embedding_fingerprint",
    "event_index_fingerprint",
    "x_context_fingerprint",
    "x_context_reason",
    "coverage_audit_fingerprint",
    "train_backend",
    "infer_backend",
    "node_embedding_path",
    "action_embedding_path",
    "event_index_path",
    "x_context_path",
    "node_embedding_dim",
    "action_embedding_dim",
    "event_index_cache_mode",
    "event_index_debug_fields",
    "x_context_memmap_enabled",
    "phase3e_precompute_only",
    "sspm_torch_batch_events",
    "sspm_torch_lr",
    "sspm_torch_weight_decay",
    "sspm_torch_device",
    "fingerprint",
    "cache_version",
    "target_dim",
    "state_dim",
    "context_dim",
    "rank",
    "train_events_actual",
    "validation_events_actual",
    "test_events_actual",
    "coverage_audit_path",
    "node_token_coverage",
    "action_token_coverage",
    "all_node_token_coverage",
    "all_oov_node_ratio",
    "partial_oov_node_ratio",
    "num_nodes_total",
    "num_nodes_all_oov",
    "num_nodes_partial_oov",
    "zero_node_count",
    "fallback_count",
    "node_embedding_meta_path",
    "action_embedding_meta_path",
    "event_index_meta_path",
    "x_context_meta_path",
    "num_nodes",
    "num_actions",
    "coverage",
    "warnings",
    "train_stats",
    "loss_by_epoch",
    "torch_device",
    "torch_cuda_available",
    "torch_device_name",
    "torch_optimizer",
    "torch_lr",
    "torch_weight_decay",
    "torch_batch_events",
    "torch_epochs_completed",
    "torch_train_loss_best",
    "torch_early_stop_reason",
    "torch_train_time_sec",
    "torch_peak_gpu_memory_mb",
    "torch_early_stop_patience",
    "torch_early_stop_min_delta",
    "real_diag_train_gamma",
    "real_diag_gamma_active",
    "real_diag_sensitivity_mode",
    "real_diag_gamma_min",
    "real_diag_gamma_max",
    "real_diag_gamma_init",
    "gamma_initial_mean",
    "gamma_initial_min",
    "gamma_initial_max",
    "gamma_final_mean",
    "gamma_final_min",
    "gamma_final_max",
    "gamma_delta_l2",
    "gamma_grad_norm_mean",
    "gamma_grad_norm_max",
    "sensitivity_mode",
    "sensitivity_nodes",
    "train_seconds",
    "train_rss_peak_mb",
    "train_epochs_completed",
    "train_events_seen_total",
    "early_stop_reason",
    "batch_events",
}
_PHASE3E_METADATA_LEAKAGE_KEY_TERMS = (
    "label",
    "labels",
    "ground_truth",
    "attack",
    "malicious",
    "suspicious",
    "benign",
    "ranking",
    "test_score",
    "test_label",
    "window",
)
_PHASE3E_METADATA_MAX_LIST_LENGTH = 128
SLIM_SPLIT_OVERRIDES: dict[str, dict[str, Any]] = {
    "CLEARSCOPE_E3": {
        "year_month": "2018-04",
        "train": [2, 3, 4, 5, 7, 8, 9],
        "val": [10],
        "test": [11, 12],
    },
    "CADETS_E3": {
        "year_month": "2018-04",
        "train": [2, 3, 4, 5, 7, 8, 9],
        "val": [10],
        "test": [6, 11, 12, 13],
    },
}
TEST_SCORING_NODE_MAP_KEYS = (
    "indexid2summary",
    "hash2type",
    "hash2uuid_index",
    "netflow_meta",
    "process_meta",
    "file_meta",
)
RESIDUAL_EXAMPLE_GROUPS = ("process_file", "process_netflow", "process_process")
RESIDUAL_EXAMPLE_FIELDS = (
    "dataset",
    "group",
    "event_index",
    "timestamp_ns",
    "src_idx",
    "dst_idx",
    "src_kind",
    "dst_kind",
    "action",
    "object_type",
    "src_role",
    "dst_role",
    "residual_text",
    "hash_tokens",
    "token_count",
    "token_projection",
    "nonzero_vector",
    "latent_dim",
)
MALICIOUS_PAIR_AUDIT_FIELDS = (
    "dataset",
    "event_index",
    "timestamp_ns",
    "src_idx",
    "dst_idx",
    "src_kind",
    "dst_kind",
    "action",
    "object_type",
    "event_label",
    "both_nodes_abnormal",
    "src_role",
    "dst_role",
    "residual_text",
    "hash_tokens",
    "token_count",
    "token_projection",
    "nonzero_vector",
    "latent_dim",
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:/-]+")
_CACHE_FORBIDDEN_KEYS = {
    "test_scores",
    "test_labels",
    "test_rankings",
    "ground_truth",
    "labels",
    "rankings",
}
_CACHE_SAFETY_FLAG_KEYS = {
    "contains_test_scores",
    "contains_test_labels",
    "contains_test_rankings",
    "ground_truth_used",
}
_BASE_CACHE_FORBIDDEN_KEYS = {
    "validation_event_scores",
    "validation_event_sorted",
    "validation_semantic_scores",
    "validation_semantic_sorted",
    "semantic",
    "threshold",
    "event_score_mode",
}

BASE_EVENT_RAW_FIELDS = [
    "stream_pos",
    "event_index",
    "timestamp_ns",
    "src_idx",
    "dst_idx",
    "info_src",
    "info_dst",
    "action",
    "src_type",
    "dst_type",
    "object_type",
    "dst_role",
    "event_score",
    "threshold",
    "threshold_basis",
    "residual_tail",
    "residual_score",
    "residual_l2",
    "residual_cos",
    "top_residual_dim_1",
    "top_residual_dim_2",
    "top_residual_dim_3",
    "src_state_norm",
    "dst_state_norm",
    "src_event_count",
    "dst_event_count",
    "src_as_src_count",
    "dst_as_dst_count",
    "target_case",
    "target_case_threshold",
    "threshold_level",
    "threshold_group_key",
    "validation_group_count",
    "low_support_policy",
    "group_validation_max",
    "parent_threshold",
    "global_threshold",
    "final_threshold_source",
    "adaptive_margin_used",
    "validation_count_bucket",
    "endpoint_signature",
    "endpoint_pair_key",
    "src_endpoint_key",
    "endpoint_action_key",
    "endpoint_validation_pair_count",
    "src_endpoint_count",
    "endpoint_validation_count",
    "endpoint_suppression_mode",
    "endpoint_suppression_match_level",
    "endpoint_suppression_reason",
]
OFSM_EVENT_RAW_FIELDS = [
    "src_cluster_id",
    "dst_cluster_id",
    "src_cluster_size",
    "dst_cluster_size",
    "src_merge_similarity",
    "dst_merge_similarity",
    "src_is_clustered",
    "dst_is_clustered",
    "src_copy_on_write",
    "dst_copy_on_write",
    "src_merge_mode",
    "dst_merge_mode",
    "src_physical_state_id",
    "dst_physical_state_id",
]
EVENT_RAW_FIELDS = BASE_EVENT_RAW_FIELDS + OFSM_EVENT_RAW_FIELDS
EVENT_EVAL_FIELDS = EVENT_RAW_FIELDS + ["event_label", "is_correct_event_alert"]
NODE_POOL_EVAL_FIELDS = [
    "node_id",
    "node_type",
    "node_label",
    "first_alert_event_idx",
    "alert_count",
    "max_event_score",
    "pool_type",
    "eval_result",
    "supporting_event_count",
]
NODE_RAW_FIELDS = [
    "node_id",
    "node_score",
    "candidate_mass",
    "residual_mass",
    "residual_max",
    "residual_mean",
    "association_support",
    "adaptive_memory_deviation",
    "compact_chain_support",
    "chain_diversity_pool",
    "repeated_consistency",
    "candidate_event_count",
]
NODE_EVAL_FIELDS = NODE_RAW_FIELDS + ["node_label", "is_correct_node_alert"]
CHECKPOINT_RAW_FIELDS = ["stream_pos", "node_id", "node_score", "pool_size"]
CHECKPOINT_EVAL_FIELDS = CHECKPOINT_RAW_FIELDS + ["node_label", "is_correct_node_alert"]
EVENT_NODE_COVERAGE_FIELDS = [
    "node_idx",
    "node_type",
    "first_alert_event_id",
    "last_alert_event_id",
    "alert_count",
    "src_alert_count",
    "dst_alert_count",
    "max_event_score",
]
EVENT_NODE_COVERAGE_EVAL_FIELDS = EVENT_NODE_COVERAGE_FIELDS + [
    "node_label",
    "eval_result",
]
EVENT_FP_GROUP_FIELDS = [
    "action",
    "src_type",
    "dst_type",
    "event_label",
    "event_count",
    "tp",
    "fp",
    "precision",
    "score_min",
    "score_mean",
    "score_max",
]
EVENT_SCORE_TRACE_FIELDS = [
    "stream_pos",
    "event_id",
    "src_idx",
    "dst_idx",
    "info_src",
    "info_dst",
    "score",
    "threshold",
    "action",
    "src_type",
    "dst_type",
    "target_case",
    "threshold_level",
    "threshold_group_key",
]
ACTION_TYPE_POLICY_EVENT_FIELDS = EVENT_RAW_FIELDS + [
    "alert_policy",
    "alert_decision",
    "alert_priority",
    "node_evidence",
    "budget_capped",
]
ACTION_TYPE_POLICY_SUMMARY_FIELDS = [
    "dataset",
    "run",
    "state_model",
    "policy_name",
    "action",
    "src_type",
    "dst_type",
    "threshold",
    "event_count",
    "alert_count",
    "TP",
    "FP",
    "precision",
    "covered_malicious_nodes",
    "strict_node_TP",
    "strict_node_FP",
    "relaxed_node_TP",
    "relaxed_node_FP",
    "q_t_mean",
    "q_t_applied_count",
    "speed",
    "online_deploy_primary_memory_mb",
    "node_evidence_count",
    "demoted_event_count",
    "budget_capped_event_count",
    "high_priority_event_alert_count",
]
DEMOTED_GROUP_SUMMARY_FIELDS = [
    "dataset",
    "run",
    "state_model",
    "policy_name",
    "action",
    "src_type",
    "dst_type",
    "target_case",
    "event_count",
    "node_evidence_count",
    "demoted_event_count",
    "budget_capped_event_count",
]
TARGET_CASE_SUMMARY_FIELDS = [
    "target_case",
    "event_count",
    "alert_count",
    "TP",
    "FP",
    "precision",
    "threshold_mean",
    "score_mean",
    "score_p99",
    "score_p999",
]
GROUP_ALERT_SUMMARY_FIELDS = [
    "action",
    "src_type",
    "dst_type",
    "target_case",
    "event_count",
    "alert_count",
    "TP",
    "FP",
    "precision",
    "covered_malicious_nodes",
]
GROUP_THRESHOLD_SWEEP_SUMMARY_FIELDS = [
    "dataset",
    "run",
    "state_model",
    "policy_name",
    "action",
    "src_type",
    "dst_type",
    "threshold",
    "event_count",
    "alert_count",
    "TP",
    "FP",
    "precision",
    "covered_malicious_nodes",
    "strict_node_TP",
    "strict_node_FP",
    "relaxed_node_TP",
    "relaxed_node_FP",
    "q_t_mean",
    "q_t_applied_count",
    "speed",
    "online_deploy_primary_memory_mb",
    "quantile",
    "margin",
    "threshold_policy",
]
Q_T_BY_ACTION_TYPE_FIELDS = [
    "dataset",
    "run",
    "state_model",
    "policy_name",
    "action",
    "src_type",
    "dst_type",
    "threshold",
    "event_count",
    "alert_count",
    "TP",
    "FP",
    "precision",
    "covered_malicious_nodes",
    "strict_node_TP",
    "strict_node_FP",
    "relaxed_node_TP",
    "relaxed_node_FP",
    "q_t_mean",
    "q_t_applied_count",
    "speed",
    "online_deploy_primary_memory_mb",
    "q_t_min",
    "q_t_max",
    "q_t_std",
    "q_t_skipped_count",
]
NODE_COVERAGE_BY_GROUP_FIELDS = [
    "dataset",
    "run",
    "state_model",
    "policy_name",
    "action",
    "src_type",
    "dst_type",
    "threshold",
    "event_count",
    "alert_count",
    "TP",
    "FP",
    "precision",
    "covered_malicious_nodes",
    "strict_node_TP",
    "strict_node_FP",
    "relaxed_node_TP",
    "relaxed_node_FP",
    "q_t_mean",
    "q_t_applied_count",
    "speed",
    "online_deploy_primary_memory_mb",
]
NODE_FP_GROUP_FIELDS = [
    "node_type",
    "dominant_action",
    "dominant_peer_type",
    "node_count",
    "strict_tp",
    "strict_fp",
    "total_event_support",
    "max_event_score",
]
PROFILING_FIELDS = [
    "phase",
    "processed_events",
    "elapsed_seconds",
    "throughput_events_per_second",
    "current_rss_mb",
    "peak_rss_mb",
    "test_phase_peak_rss_mb",
    "event_alert_count",
    "node_pool_size",
    "state_slots",
    "state_array_mb",
    "state_metadata_mb",
    "state_eviction_count",
    "state_overflow_count",
    "state_prototype_hit_count",
]
STATE_MERGE_PROFILE_FIELDS = [
    "event_idx",
    "state_merge_mode",
    "merge_threshold",
    "trunc_ratio",
    "random_prob",
    "random_seed",
    "logical_node_count",
    "num_physical_states",
    "num_singleton_states",
    "num_clusters",
    "num_clustered_nodes",
    "compression_ratio",
    "state_dim",
    "state_dtype",
    "estimated_state_mb_no_merge",
    "estimated_state_mb_after_merge",
    "estimated_state_mb_saved",
    "rss_peak_mb",
    "rss_final_mb",
    "avg_cluster_size",
    "max_cluster_size",
    "num_copy_on_write_total",
    "num_merge_attempts_total",
    "num_merge_success_total",
    "num_remerged_total",
    "num_became_singleton_total",
    "num_random_merge_attempts_total",
    "num_random_merge_success_total",
    "candidate_scan_count_total",
    "candidate_scan_time_sec",
    "num_candidates_checked_total",
    "avg_candidates_per_attempt",
    "max_candidates_per_attempt",
    "match_backend",
    "candidate_cap",
    "merge_interval",
    "merge_min_count",
    "skipped_merge_due_to_interval",
    "skipped_merge_due_to_min_count",
    "rss_mb",
    "events_per_sec",
]
OFSM_RUNTIME_PROFILE_FIELDS = [
    "run",
    "events",
    "stage",
    "match_backend",
    "candidate_cap",
    "merge_interval",
    "min_count",
    "total_seconds",
    "events_per_sec",
    "validation_seconds",
    "test_seconds",
    "candidate_scan_seconds",
    "bucket_lookup_seconds",
    "signature_update_seconds",
    "copy_on_write_seconds",
    "state_read_seconds",
    "state_update_seconds",
    "lazy_embedding_lookup_seconds",
    "endpoint_suppression_seconds",
    "csv_write_seconds",
    "score_compute_seconds",
    "threshold_lookup_seconds",
    "num_merge_attempts_total",
    "num_candidates_checked",
    "avg_candidates_per_attempt",
    "max_candidates_per_attempt",
    "merge_success_count",
    "skipped_merge_due_to_interval",
    "skipped_merge_due_to_min_count",
    "copy_on_write_count",
    "num_clusters",
    "clustered_nodes",
    "compression_ratio",
    "online_deploy_primary_memory_mb",
    "state_table_mb",
    "process_rss_peak_mb",
    "anonymous_rss_mb",
    "file_backed_rss_mb",
]
STATE_MERGE_DIAGNOSTIC_FIELDS = [
    "event_idx",
    "action_type",
    "node_id",
    "node_type",
    "old_cluster_id",
    "new_cluster_id",
    "candidate_type",
    "candidate_id",
    "merge_mode",
    "merge_similarity",
    "merge_threshold",
    "cluster_size_before",
    "cluster_size_after",
    "event_score",
    "raw_action",
    "src_or_dst",
    "physical_state_id_before",
    "physical_state_id_after",
    "representative_node_id",
    "representative_known",
]
ALERT_GROUPS = {
    "TP_both_malicious",
    "TP_any_malicious",
    "TP_suspicious",
    "FP_pure_benign",
}
ALERT_FEATURE_DUMP_FIELDS = [
    "event_index",
    "stream_pos",
    "alert_group",
    "event_label",
    "src_node_label",
    "dst_node_label",
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
    "event_score",
    "threshold",
    "score_margin",
    "residual_score",
    "residual_l2",
    "residual_cos",
    "top_residual_dim_1",
    "top_residual_dim_2",
    "top_residual_dim_3",
    "src_state_norm",
    "dst_state_norm",
    "src_event_count",
    "dst_event_count",
    "src_as_src_count",
    "dst_as_dst_count",
    *OFSM_EVENT_RAW_FIELDS,
]
FP_TP_CATEGORICAL_LIFT_FIELDS = [
    "feature",
    "value",
    "fp_count",
    "tp_both_malicious_count",
    "fp_rate",
    "tp_both_malicious_rate",
    "lift",
]
FP_TP_NUMERIC_SUMMARY_FIELDS = [
    "feature",
    "fp_mean",
    "fp_p50",
    "fp_p90",
    "fp_p99",
    "tp_both_malicious_mean",
    "tp_both_malicious_p50",
    "tp_both_malicious_p90",
    "tp_both_malicious_p99",
    "difference",
    "ratio",
]
OFSM_ALERT_MERGE_ANALYSIS_FIELDS = [
    "event_index",
    "alert_group",
    "src_is_clustered",
    "dst_is_clustered",
    "src_cluster_size",
    "dst_cluster_size",
    "src_merge_similarity",
    "dst_merge_similarity",
    "same_cluster",
    "src_copy_on_write",
    "dst_copy_on_write",
    "event_score",
    "score_margin",
]
PROCESS_AUDIT_FIELDS = [
    "split",
    "dataset",
    "rules_version",
    "profile",
    "event_index",
    "node_id",
    "path",
    "cmd",
    "nll_role",
    "residual_text",
    "residual_token_count",
    "used_unknown",
]


class StreamingCsvWriter:
    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = None
        self.writer = None
        self.write_seconds = 0.0
        handle = self.path.open("w", newline="", encoding="utf-8")
        try:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
            write_started = time.perf_counter()
            writer.writeheader()
            self.write_seconds += float(time.perf_counter() - write_started)
        except Exception:
            handle.close()
            raise
        self.handle = handle
        self.writer = writer

    def write_row(self, row: Mapping[str, Any]) -> None:
        if self.writer is None:
            raise ValueError("streaming CSV writer is closed")
        write_started = time.perf_counter()
        try:
            self.writer.writerow({name: row.get(name, "") for name in self.writer.fieldnames})
        finally:
            self.write_seconds += float(time.perf_counter() - write_started)

    def flush(self) -> None:
        if self.handle is not None:
            write_started = time.perf_counter()
            try:
                self.handle.flush()
            finally:
                self.write_seconds += float(time.perf_counter() - write_started)

    def close(self) -> None:
        if self.handle is not None:
            write_started = time.perf_counter()
            try:
                self.handle.close()
            finally:
                self.write_seconds += float(time.perf_counter() - write_started)
            self.handle = None
            self.writer = None

    def __enter__(self) -> "StreamingCsvWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class OnlineNodeCoverageTracker:
    """Track alert-covered nodes with label-free counters during event streaming."""

    def __init__(self) -> None:
        self._nodes: dict[int, dict[str, Any]] = {}

    def observe(
        self,
        *,
        node_idx: int,
        node_type: str,
        event_id: int,
        event_score: float,
        side: str,
    ) -> None:
        node_id = int(node_idx)
        state = self._nodes.setdefault(
            node_id,
            {
                "node_idx": node_id,
                "node_type": str(node_type),
                "first_alert_event_id": int(event_id),
                "last_alert_event_id": int(event_id),
                "alert_count": 0,
                "src_alert_count": 0,
                "dst_alert_count": 0,
                "max_event_score": 0.0,
            },
        )
        state["last_alert_event_id"] = int(event_id)
        state["alert_count"] = int(state.get("alert_count", 0)) + 1
        if str(side) == "src":
            state["src_alert_count"] = int(state.get("src_alert_count", 0)) + 1
        elif str(side) == "dst":
            state["dst_alert_count"] = int(state.get("dst_alert_count", 0)) + 1
        state["max_event_score"] = max(float(state.get("max_event_score", 0.0)), event_score)

    def rows(self) -> list[dict[str, Any]]:
        return sorted(
            (dict(row) for row in self._nodes.values()),
            key=lambda row: (-float(row["max_event_score"]), int(row["node_idx"])),
        )

    def estimated_mb(self) -> float:
        total = sys.getsizeof(self._nodes)
        for key, value in self._nodes.items():
            total += sys.getsizeof(key)
            total += sys.getsizeof(value)
            for sub_key, sub_value in value.items():
                total += sys.getsizeof(sub_key)
                total += sys.getsizeof(sub_value)
        return float(total) / 1024.0 / 1024.0

    def __len__(self) -> int:
        return len(self._nodes)


class SlimTempStreamError(RuntimeError):
    """Expected TEMP TABLE stream setup/query failure eligible for auto fallback."""


@dataclass
class SlimTempTableStream:
    cursor: Any
    fetch_size: int
    max_events: int
    abnormal_nodes: set[int]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        produced = 0
        next_event_index = 0
        try:
            while True:
                try:
                    rows = self.cursor.fetchmany(int(self.fetch_size))
                except DB_EXCEPTION_TYPES as exc:
                    raise SlimTempStreamError(f"temp table stream fetch failed: {exc}") from exc
                if not rows:
                    break
                for row in rows:
                    if int(self.max_events) > 0 and produced >= int(self.max_events):
                        return
                    stream_row = _compact_temp_event_row(
                        row,
                        next_event_index,
                        self.abnormal_nodes,
                    )
                    if stream_row is None:
                        continue
                    yield stream_row
                    next_event_index += 1
                    produced += 1
        finally:
            self.cursor.close()


@dataclass
class SlimConfig:
    """Configuration for the slim causal semantics smoke/deployment pipeline."""

    dataset: str = "SYNTHETIC"
    out_tag: str = "CAUSAL_SEMANTICS_SLIM_SYNTHETIC"
    result_root: str = "outputs/results/tflr_light"
    max_train_events: int = 200000
    max_ref_events: int = 62748
    max_test_events: int = 0
    max_tokens_per_node: int = 8
    expected_event_alert_budget: float = 100.0
    expected_alert_horizon_events: int = 1200000
    event_threshold_mode: str = "budget"
    event_threshold_quantile: float = 0.999
    latent_dim: int = 64
    latent_dim_policy: str = "fixed"
    rank: int = 32
    sspm_batch_size: int = 1024
    sspm_learning_rate: float = 0.005
    sspm_l2: float = 1e-4
    sspm_epochs: int = 1
    sspm_early_stop_min_delta: float = 1e-4
    sspm_early_stop_patience: int = 5
    sspm_train_mode: str = "train_and_save"
    sspm_train_data_mode: str = "full_cache"
    sspm_checkpoint_path: str = ""
    sspm_base_checkpoint_root: str = "outputs/models/sspm_phase3d"
    sspm_train_batch_events: int = 8192
    # Phase3E runner opts into node_action_semantic_mean; event_residual stays
    # the default for compatibility with Phase3D checkpoints and runners.
    sspm_target_mode: str = "event_residual"
    node_word2vec_source: str = "residual_pretrained"
    node_embedding_lookup_mode: str = "global_memmap"
    node_embedding_lazy_backing: str = "compact_used_nodes"
    node_embedding_cache_max_nodes: int = 50000
    node_embedding_cache_evict_policy: str = "lru"
    node_embedding_dim: int = 64
    node_embedding_cache_dir: str = "outputs/cache/node_embeddings/{DATASET}_latent64"
    compact_used_node_cache_dir: str = "outputs/cache/compact_used_node_embeddings/{DATASET}"
    phase3e_node_lookup_batch_size: int = 50000
    action_embedding_dim: int = 64
    action_embedding_cache_dir: str = "outputs/cache/action_embeddings/{DATASET}_latent64"
    event_index_cache_mode: str = "auto"
    event_index_cache_dir: str = "outputs/cache/event_indices/{DATASET}"
    event_index_debug_fields: bool = False
    x_context_cache_dir: str = "outputs/cache/x_context/{DATASET}"
    x_context_memmap_enabled: bool = False
    sspm_train_backend: str = "numpy"
    sspm_infer_backend: str = "numpy"
    sspm_score_head: str = "semantic_residual"
    node_repr_fusion: str = "simple_mean"
    conditional_semantic_loss: str = "cosine"
    sspm_conditional_head_arch: str = CONDITIONAL_HEAD_ARCH_SHARED_V1
    action_head_checkpoint_path: str = ""
    action_validation_cache_dir: str = "outputs/cache/phase3g_action_validation/{DATASET}"
    conditional_group_min_count: int = 1000
    conditional_low_support_policy: str = CONDITIONAL_LOW_SUPPORT_POLICY_DEFAULT
    conditional_low_support_margin: float = CONDITIONAL_LOW_SUPPORT_MARGIN_DEFAULT
    conditional_unseen_group_policy: str = CONDITIONAL_UNSEEN_GROUP_POLICY_DEFAULT
    conditional_global_extreme_quantile: float = CONDITIONAL_GLOBAL_EXTREME_QUANTILE_DEFAULT
    conditional_adaptive_margin_n1: int = CONDITIONAL_ADAPTIVE_MARGIN_N1_DEFAULT
    conditional_adaptive_margin_n2: int = CONDITIONAL_ADAPTIVE_MARGIN_N2_DEFAULT
    conditional_adaptive_margin_low: float = CONDITIONAL_ADAPTIVE_MARGIN_LOW_DEFAULT
    conditional_adaptive_margin_mid: float = CONDITIONAL_ADAPTIVE_MARGIN_MID_DEFAULT
    conditional_adaptive_margin_high: float = CONDITIONAL_ADAPTIVE_MARGIN_HIGH_DEFAULT
    conditional_endpoint_aware_suppression: bool = False
    conditional_endpoint_suppression_read: bool = False
    conditional_endpoint_suppression_mode: str = "pair_only"
    conditional_known_pair_min_count: int = 5
    conditional_pair_suppression_margin: float = 0.02
    conditional_same_process_endpoint_min_count: int = 5
    conditional_same_process_endpoint_margin: float = 0.01
    conditional_endpoint_suppression_cache_dir: str = (
        "outputs/cache/phase3g_endpoint_suppression/{DATASET}"
    )
    conditional_endpoint_suppression_summary_mode: str = "online_minimal"
    sspm_conditional_train_data_mode: str = "memmap"
    sspm_conditional_max_epochs: int = 80
    sspm_conditional_e3_max_epochs: int = 2
    phase3g_build_conditional_memmap_only: bool = False
    sspm_infer_fast_path: bool = False
    sspm_infer_chunk_events: int = 8192
    sspm_torch_batch_events: int = 8192
    sspm_torch_lr: float = 0.0
    sspm_torch_weight_decay: float = 0.0
    sspm_torch_device: str = "auto"
    phase3e_precompute_only: bool = False
    sspm_state_model: str = "ema_fixed"
    sspm_target_dim: int = 64
    sspm_state_dim: int = 64
    sspm_context_mode: str = "with_action"
    sspm_context_action_mode: str = "raw_orthrus10"
    sspm_global_context_mode: str = "no_global"
    state_memory_mode: str = "bounded"
    sspm_state_memory_policy: str = "probationary_lru"
    sspm_active_max_nodes: int = 500000
    sspm_probationary_max_nodes: int = 500000
    sspm_probationary_min_count: int = 2
    sspm_lru_evict_batch: int = 10000
    sspm_alert_protect_events: int = 100000
    sspm_update_gate_mode: str = "none"
    sspm_update_gate_quantile: float = 0.999
    sspm_update_gate_threshold_mode: str = "quantile"
    sspm_update_gate_score_space: str = "checkpoint"
    sspm_update_gate_q_min: float = 0.05
    sspm_update_gate_eta: float = 1.0
    sspm_score_target_mode: str = SCORE_TARGET_EVENT_ACTION
    sspm_residual_score_mode: str = "legacy"
    sspm_residual_calibration: str = "none"
    sspm_residual_alpha_cos: float = 1.0
    sspm_residual_beta_mse: float = 0.25
    sspm_residual_beta_var: float = 0.25
    sspm_residual_var_eps: float = 1e-6
    sspm_residual_calibration_min_count: int = 100
    real_diag_train_gamma: bool = False
    real_diag_gamma_lr: float = 0.001
    real_diag_gamma_weight_decay: float = 0.0
    real_diag_gamma_grad_clip: float = 1.0
    real_diag_sensitivity_mode: str = "online_stop_message"
    real_diag_gamma_min: float = 1e-4
    real_diag_gamma_max: float = 1.0
    real_diag_max_sensitivity_nodes: int = 500000
    sspm_state_merge_mode: str = "none"
    sspm_state_merge_scope: str = "same_type"
    sspm_state_merge_threshold: float = 0.98
    sspm_state_merge_trunc_ratio: float = 0.50
    sspm_state_merge_use_rfft: bool = True
    sspm_state_merge_use_real_only: bool = True
    sspm_state_merge_center_state: bool = False
    sspm_state_merge_normalize: str = "l2"
    sspm_state_merge_eps: float = 1e-8
    sspm_state_merge_min_cluster_size: int = 2
    sspm_state_merge_copy_on_write: bool = True
    sspm_state_merge_random_prob: float = 0.02
    sspm_state_merge_random_seed: int = 0
    sspm_state_merge_diagnostics_max_rows: int = 1_000_000
    sspm_ofsm_match_backend: str = "exact"
    sspm_ofsm_candidate_cap: int = 64
    sspm_ofsm_merge_interval: int = 1
    sspm_ofsm_min_count: int = 1
    sspm_ofsm_diagnostics: bool = True
    max_exact_states: int = 0
    min_inactive_events: int = 1_000_000
    prototype_count: int = 512
    prototype_update_alpha: float = 0.05
    action_count: int = 10
    pretrained_residual_embedder_path: str = ""
    residual_embed_cache_mode: str = "off"
    residual_embed_cache_path: str = (
        "outputs/cache/residual_embeddings/{DATASET}_latent64_word2vec_cache.sqlite"
    )
    max_recent_events_per_node: int = 32
    fetch_size: int = 10000
    precompute_cache_dir: str = ""
    precompute_cache_mode: str = "off"
    progress_interval_events: int = 100000
    db_stream_mode: str = "auto"
    event_score_mode: str = "res_only"
    semantic_embedding_method: str = "word2vec"
    semantic_mode: str = CLEARSCOPE_REFINED_SEMANTIC_MODE
    word2vec_window: int = 3
    word2vec_min_count: int = 1
    word2vec_sg: int = 1
    word2vec_negative: int = 5
    word2vec_epochs: int = 10
    word2vec_workers: int = 4
    word2vec_seed: int = 0
    word2vec_oov_policy: str = "unk"
    theia_netflow_policy: str = "fixed"
    action_type_alert_policy: str = "default"
    node_pool_score_mode: str = "base_conf"
    node_pool_topk_values: str = "100,200,500,1000,1500,3000,5000,10000,20000,30000"
    slim_split_override: bool = False
    compat_in_memory_outputs: bool = False
    synthetic_smoke: bool = False
    print_summary: bool = False
    rss_profile_mode: str = "analysis"
    write_raw_alerts: bool = True
    write_analysis_outputs: bool = True
    sspm_state_merge_write_diagnostics: bool = True
    process_semantics_config: str = "configs/common/process_semantics.yaml"
    process_semantic_audit: bool = False
    write_legacy_eval_alias: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class EventCalibration:
    """Runtime-derived validation calibration for the selected event score mode."""

    residual_sorted: np.ndarray
    event_scores: np.ndarray
    event_sorted: np.ndarray
    threshold: float
    summary: dict[str, float | int]


@dataclass
class AdaptiveRateThresholdController:
    """Label-free online alert-rate controller for event thresholds."""

    initial_threshold: float
    target_rate: float
    upper_tolerance: float = 1.5
    lower_tolerance: float = 0.5
    increase_factor: float = 1.05
    decrease_factor: float = 0.95
    min_events_before_adjust: int = 100

    def __post_init__(self) -> None:
        self.threshold = float(self.initial_threshold)
        self.processed = 0
        self.alerts = 0
        self.increases = 0
        self.decreases = 0

    def observe(self, alerted: bool) -> None:
        """Update threshold using only no-label online alert-rate counters."""
        self.processed += 1
        if bool(alerted):
            self.alerts += 1
        if self.processed < int(self.min_events_before_adjust):
            return
        observed_rate = float(self.alerts / max(self.processed, 1))
        target_rate = max(float(self.target_rate), 1e-12)
        if observed_rate > target_rate * float(self.upper_tolerance):
            self.threshold *= float(self.increase_factor)
            self.increases += 1
        elif observed_rate < target_rate * float(self.lower_tolerance):
            self.threshold *= float(self.decrease_factor)
            self.decreases += 1

    def summary(self) -> dict[str, float | int | str]:
        """Return label-free controller telemetry."""
        return {
            "mode": "adaptive_rate",
            "initial_threshold": float(self.initial_threshold),
            "final_threshold": float(self.threshold),
            "target_rate": float(self.target_rate),
            "processed": int(self.processed),
            "alerts": int(self.alerts),
            "observed_rate": float(self.alerts / max(self.processed, 1)),
            "increases": int(self.increases),
            "decreases": int(self.decreases),
        }


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


def stream_dataset_rows(
    conn,
    year_month: str,
    days: Sequence[int],
    node_maps: Mapping[str, Any],
    event_filter: bool,
    fetch_size: int,
    max_events: int,
    abnormal_nodes: set[int] | None = None,
):
    """Yield DB stream rows with labels disabled by default and netflow metadata enriched."""
    safe_abnormal_nodes = set() if abnormal_nodes is None else set(abnormal_nodes)
    include_labels = bool(safe_abnormal_nodes)
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
    def log(stage: str, **values: object) -> None:
        if config is None:
            return
        _stage_log(config, f"phase3e_pass1_{split}_{stage}", **values)

    day_sql, day_params = _slim_temp_day_filter(year_month, days)
    if not day_params:
        return iter(())
    operation_sql = ""
    params: list[Any] = [*day_params]
    if event_filter:
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
                        operation=str(op),
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


def run_phase3e_precompute_db(
    *,
    config: SlimConfig,
    conn: Any,
    year_month: str,
    train_days: Sequence[int],
    validation_days: Sequence[int],
    test_days: Sequence[int],
    node_maps: Mapping[str, Any] | None,
    event_filter: bool,
    process_cfg: ProcessSemanticConfig | None,
) -> Path:
    """Run real DB-backed Phase3E precompute without materializing event rows."""
    started = time.perf_counter()
    output_dir = Path(config.result_root) / config.out_tag
    node_dir = _phase3e_resolve_cache_dir(config.node_embedding_cache_dir, config)
    action_dir = _phase3e_resolve_cache_dir(config.action_embedding_cache_dir, config)
    event_dir = _phase3e_resolve_cache_dir(config.event_index_cache_dir, config)
    x_dir = _phase3e_resolve_cache_dir(config.x_context_cache_dir, config)
    for path in (output_dir, node_dir, action_dir, event_dir, x_dir):
        path.mkdir(parents=True, exist_ok=True)

    embedder = load_pretrained_residual_embedder(
        str(config.pretrained_residual_embedder_path),
        expected_dim=_effective_latent_dim(config, process_cfg),
    )
    adapter = ResidualWord2VecTokenAdapter(
        embedder,
        model_path=str(config.pretrained_residual_embedder_path),
    )
    event_id_column = _phase3e_detect_event_id_column(conn)
    split_days = {
        "train": list(train_days),
        "validation": list(validation_days),
        "test": list(test_days),
    }
    max_events_by_split = {
        "train": int(config.max_train_events),
        "validation": int(config.max_ref_events),
        "test": int(config.max_test_events),
    }
    pass1 = _phase3e_scan_used_nodes_pass1(
        conn=conn,
        year_month=year_month,
        split_days=split_days,
        event_filter=event_filter,
        fetch_size=int(config.fetch_size),
        max_events_by_split=max_events_by_split,
        event_id_column=event_id_column,
        config=config,
    )
    used_node_ids = set(pass1["used_node_ids"])
    split_used_nodes = pass1["split_used_nodes"]
    split_counts = {key: int(value) for key, value in pass1["split_counts"].items()}
    db_node_counts = _phase3e_node_table_counts_db(conn)
    indexid2summary, meta_by_kind, node_lookup_time = _phase3e_fetch_used_node_lookup_batched(
        conn=conn,
        used_node_ids=used_node_ids,
        batch_size=int(config.phase3e_node_lookup_batch_size),
    )
    missing_node_ids = sorted(used_node_ids - set(indexid2summary))
    if missing_node_ids:
        raise RuntimeError(
            "Phase3E used event nodes are absent from process/file/netflow tables; "
            f"missing_node_count={len(missing_node_ids)} sample={missing_node_ids[:10]}",
        )
    node_kind_by_id = {
        int(node_id): normalize_piece(summary[0])
        for node_id, summary in indexid2summary.items()
    }
    node_table = _phase3e_build_node_table_from_used_lookup(
        used_node_ids=used_node_ids,
        split_used_nodes=split_used_nodes,
        indexid2summary=indexid2summary,
        meta_by_kind=meta_by_kind,
        adapter=adapter,
        config=config,
        process_cfg=process_cfg,
    )
    action_table = build_action_embedding_table(adapter)
    node_paths = save_node_embedding_table(node_table, node_dir)
    action_paths = save_action_embedding_table(action_table, action_dir)
    audit = build_node_action_coverage_audit(node_table, action_table)
    audit_path = save_node_action_coverage_audit(audit, output_dir)

    split_meta: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        split_meta[split] = _phase3e_write_event_index_split_from_direct_db(
            conn=conn,
            year_month=year_month,
            days=split_days[split],
            event_filter=event_filter,
            fetch_size=int(config.fetch_size),
            max_events=int(max_events_by_split[split]),
            expected_count=int(split_counts[split]),
            path=event_dir / f"event_index_{split}.memmap",
            split=split,
            node_id_to_idx=node_table.node_id_to_idx,
            action_name_to_id=action_table.name_to_action_id,
            node_kind_by_id=node_kind_by_id,
            config=config,
            event_id_column=event_id_column,
        )
        split_meta[split]["postgres_count_effective"] = int(split_counts[split])
        split_meta[split]["used_node_count"] = int(len(split_used_nodes.get(split, set())))

    event_index_meta = {
        "row_order_is_stream_order": True,
        "order_by": list(PHASE3E_EVENT_ORDER_BY),
        "event_id_source": str(event_id_column),
        "operation_filter": "ORTHRUS10" if event_filter else "none",
        "target_mode": str(config.sspm_target_mode),
        "node_word2vec_source": str(config.node_word2vec_source),
        "event_index_dtype": _phase3e_event_index_dtype_descriptor(),
        "dtype_descriptor": _phase3e_event_index_dtype_descriptor(),
        "splits": split_meta,
        "event_count_source": "postgres_count_with_split_and_action_filter",
        "node_universe_source": "event_stream_used_nodes_batched_db_lookup",
        "used_node_count": int(len(used_node_ids)),
        "missing_node_count": 0,
        "fingerprint": stable_json_hash(split_meta),
    }
    event_index_meta_path = event_dir / "event_index_meta.json"
    event_index_meta_path.write_text(
        json.dumps(event_index_meta, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    x_context = _phase3e_build_x_contexts_from_db_stream(
        config=config,
        conn=conn,
        year_month=year_month,
        train_days=train_days,
        node_maps={},
        event_filter=event_filter,
        fetch_size=int(config.fetch_size),
        max_events=int(config.max_train_events),
        expected_count=int(split_counts["train"]),
        node_table=node_table,
        action_table=action_table,
        node_kind_by_id=node_kind_by_id,
        x_context_dir=x_dir,
        event_index_meta=split_meta["train"],
        process_cfg=process_cfg,
    )
    x_context_meta_path = x_dir / "x_context_meta.json"
    x_context_meta_path.write_text(
        json.dumps(x_context, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    elapsed = max(time.perf_counter() - started, 1e-9)
    total_events = int(sum(split_counts.values()))
    metrics = _phase3e_precompute_metrics(
        config=config,
        output_dir=output_dir,
        node_table=node_table,
        action_table=action_table,
        node_paths=node_paths,
        action_paths=action_paths,
        audit=audit,
        audit_path=str(audit_path),
        event_index_meta=event_index_meta,
        event_index_meta_path=str(event_index_meta_path),
        split_counts=split_counts,
        split_full_counts=split_counts,
        x_context=x_context,
        x_context_meta_path=str(x_context_meta_path),
        elapsed=elapsed,
        order_by=PHASE3E_EVENT_ORDER_BY,
        event_id_source=str(event_id_column),
    )
    metrics["precompute_data_flow"] = {
        "event_count_source": "postgres_count_with_split_and_action_filter",
        "node_universe_source": "event_stream_used_nodes_batched_db_lookup",
        "event_index_write_mode": "postgres_stream_to_memmap",
        "x_context_build_mode": "postgres_stream_or_event_index_to_memmap",
        "materialized_event_rows": False,
        "train_rows_materialized": False,
        "validation_rows_materialized": False,
        "test_rows_materialized": False,
    }
    metrics.update(
        {
            "db_node_table_total_count": int(db_node_counts["total"]),
            "used_node_count": int(len(used_node_ids)),
            "filtered_out_db_node_count": int(db_node_counts["total"] - len(used_node_ids)),
            "train_used_node_count": int(len(split_used_nodes.get("train", set()))),
            "validation_used_node_count": int(
                len(split_used_nodes.get("validation", set())),
            ),
            "test_used_node_count": int(len(split_used_nodes.get("test", set()))),
            "missing_node_count": 0,
            "node_lookup_query_batch_size": int(config.phase3e_node_lookup_batch_size),
            "node_lookup_query_time_sec": float(node_lookup_time),
            "node_embedding_size_mb": float(Path(node_paths["node_embeddings"]).stat().st_size)
            / 1024.0
            / 1024.0,
            "db_node_table_counts": db_node_counts,
        },
    )
    metrics_path = output_dir / "precompute_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    _phase3e_write_precompute_eval_and_config(
        output_dir=output_dir,
        metrics_path=metrics_path,
        audit_path=str(audit_path),
        event_index_meta_path=str(event_index_meta_path),
        x_context_meta_path=str(x_context_meta_path),
        node_paths=node_paths,
        action_paths=action_paths,
        metrics=metrics,
        config=config,
        process_cfg=process_cfg,
        elapsed=elapsed,
        train_count=int(split_counts["train"]),
        validation_count=int(split_counts["validation"]),
        test_count=int(split_counts["test"]),
    )
    _stage_log(
        config,
        "phase3e_precompute_db_end",
        events=total_events,
        rss_mb=f"{_current_rss_mb():.3f}",
    )
    return metrics_path


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
    action_text = str(op)
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
    )


def _make_sspm_config(config: SlimConfig, process_cfg: ProcessSemanticConfig | None) -> SSPMLowRankConfig:
    """Return the configured SSPM residual predictor setup."""
    return SSPMLowRankConfig(
        latent_dim=_effective_latent_dim(config, process_cfg),
        target_dim=int(config.sspm_target_dim),
        state_dim=int(config.sspm_state_dim),
        rank=config.rank,
        batch_size=config.sspm_batch_size,
        learning_rate=config.sspm_learning_rate,
        l2=config.sspm_l2,
        epochs=config.sspm_epochs,
        early_stop_min_delta=config.sspm_early_stop_min_delta,
        early_stop_patience=config.sspm_early_stop_patience,
        state_model=str(config.sspm_state_model),
        context_mode=config.sspm_context_mode,
        context_action_mode=str(config.sspm_context_action_mode),
        action_count=config.action_count,
        global_context_mode=config.sspm_global_context_mode,
        state_memory_mode=config.state_memory_mode,
        state_memory_policy=str(config.sspm_state_memory_policy),
        active_max_nodes=int(config.sspm_active_max_nodes),
        probationary_max_nodes=int(config.sspm_probationary_max_nodes),
        probationary_min_count=int(config.sspm_probationary_min_count),
        lru_evict_batch=int(config.sspm_lru_evict_batch),
        alert_protect_events=int(config.sspm_alert_protect_events),
        update_gate_mode=str(config.sspm_update_gate_mode),
        update_gate_quantile=float(config.sspm_update_gate_quantile),
        update_gate_q_min=float(config.sspm_update_gate_q_min),
        update_gate_eta=float(config.sspm_update_gate_eta),
        residual_score_mode=str(config.sspm_residual_score_mode),
        residual_calibration=str(config.sspm_residual_calibration),
        residual_alpha_cos=float(config.sspm_residual_alpha_cos),
        residual_beta_mse=float(config.sspm_residual_beta_mse),
        residual_beta_var=float(config.sspm_residual_beta_var),
        residual_var_eps=float(config.sspm_residual_var_eps),
        residual_calibration_min_count=int(config.sspm_residual_calibration_min_count),
        real_diag_train_gamma=bool(config.real_diag_train_gamma),
        real_diag_gamma_lr=float(config.real_diag_gamma_lr),
        real_diag_gamma_weight_decay=float(config.real_diag_gamma_weight_decay),
        real_diag_gamma_grad_clip=float(config.real_diag_gamma_grad_clip),
        real_diag_sensitivity_mode=str(config.real_diag_sensitivity_mode),
        real_diag_gamma_min=float(config.real_diag_gamma_min),
        real_diag_gamma_max=float(config.real_diag_gamma_max),
        real_diag_max_sensitivity_nodes=int(config.real_diag_max_sensitivity_nodes),
        state_merge_mode=str(config.sspm_state_merge_mode),
        state_merge_scope=str(config.sspm_state_merge_scope),
        state_merge_threshold=float(config.sspm_state_merge_threshold),
        state_merge_trunc_ratio=float(config.sspm_state_merge_trunc_ratio),
        state_merge_use_rfft=bool(config.sspm_state_merge_use_rfft),
        state_merge_use_real_only=bool(config.sspm_state_merge_use_real_only),
        state_merge_center_state=bool(config.sspm_state_merge_center_state),
        state_merge_normalize=str(config.sspm_state_merge_normalize),
        state_merge_eps=float(config.sspm_state_merge_eps),
        state_merge_min_cluster_size=int(config.sspm_state_merge_min_cluster_size),
        state_merge_copy_on_write=bool(config.sspm_state_merge_copy_on_write),
        state_merge_random_prob=float(config.sspm_state_merge_random_prob),
        state_merge_random_seed=int(config.sspm_state_merge_random_seed),
        state_merge_diagnostics_max_rows=int(
            config.sspm_state_merge_diagnostics_max_rows,
        ),
        state_merge_match_backend=str(config.sspm_ofsm_match_backend),
        state_merge_candidate_cap=int(config.sspm_ofsm_candidate_cap),
        state_merge_interval=int(config.sspm_ofsm_merge_interval),
        state_merge_min_count=int(config.sspm_ofsm_min_count),
        state_merge_diagnostics_enabled=bool(config.sspm_ofsm_diagnostics),
        max_exact_states=config.max_exact_states,
        min_inactive_events=config.min_inactive_events,
        prototype_count=config.prototype_count,
        prototype_update_alpha=config.prototype_update_alpha,
    )


def _phase3e_resolve_cache_dir(raw: str | Path, config: SlimConfig) -> Path:
    """Resolve a Phase3E cache directory template for the configured dataset."""
    return Path(str(raw).replace("{DATASET}", str(config.dataset)))


def _residual_tokens_for_row(
    row: Mapping[str, Any],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None = None,
) -> list[str]:
    text = residual_text(
        row,
        dataset=config.dataset,
        process_semantic_config=process_cfg,
        max_tokens_per_node=config.max_tokens_per_node,
        theia_netflow_policy=config.theia_netflow_policy,
        semantic_mode=config.semantic_mode,
    )
    return list(residual_text_tokens(text))


def _stage_log(config: SlimConfig, stage: str, **values: object) -> None:
    parts = [f"[PIPELINE][{config.dataset}] {stage}"]
    for key, value in values.items():
        parts.append(f"{key}={value}")
    parts.append(f"rss_mb={_current_rss_mb():.3f}")
    print(" ".join(parts), file=sys.stderr, flush=True)


def load_pretrained_residual_embedder(path: str, expected_dim: int) -> ResidualEmbedder:
    """Load a residual embedder pickle and validate its output dimension."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"pretrained residual embedder not found: {path_obj}")
    with path_obj.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, Mapping) and "model_state" in payload:
        embedder = residual_embedder_from_state_dict(dict(payload["model_state"]))
    elif isinstance(payload, Mapping) and "embedder" in payload:
        embedder_value = payload["embedder"]
        embedder = (
            residual_embedder_from_state_dict(dict(embedder_value))
            if isinstance(embedder_value, Mapping)
            else embedder_value
        )
    elif isinstance(payload, Mapping) and "embedder_state" in payload:
        embedder = residual_embedder_from_state_dict(dict(payload["embedder_state"]))
    elif isinstance(payload, Mapping) and "type" in payload:
        embedder = residual_embedder_from_state_dict(dict(payload))
    else:
        raise ValueError(f"unsupported pretrained residual embedder payload: {path_obj}")
    probe = embedder.encode(["__probe__"])
    if probe.shape != (int(expected_dim),):
        raise ValueError(
            f"pretrained residual embedder dim {probe.shape} != ({int(expected_dim)},)",
        )
    return embedder


def _word2vec_metadata_sidecar_path(path: str | Path) -> Path:
    """Return the JSON sidecar path for a residual Word2Vec pickle."""
    path_obj = Path(path)
    return path_obj.with_suffix(".json")


def load_word2vec_semantic_metadata(path: str | Path) -> dict[str, Any]:
    """Load JSON-safe semantic metadata for a residual Word2Vec artifact."""
    path_obj = Path(path)
    sidecar = _word2vec_metadata_sidecar_path(path_obj)
    if sidecar.exists():
        with sidecar.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return dict(payload) if isinstance(payload, Mapping) else {}
    if not path_obj.exists():
        return {}
    with path_obj.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        return {}
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state", "embedder", "embedder_state", "model"}
    }
    return dict(metadata)


def _expected_semantic_mode_for_config(config: SlimConfig) -> str:
    """Return the canonical semantic mode expected by a pipeline config."""
    if is_clearscope_dataset(config.dataset):
        return normalize_clearscope_semantic_mode(config.semantic_mode)
    return str(config.semantic_mode)


def _word2vec_semantic_rule_for_dataset(
    semantic_rules: Mapping[str, Any],
    dataset: str,
) -> str:
    """Return the dataset-specific semantic rule recorded by Word2Vec metadata."""
    dataset_name = str(dataset).upper()
    if is_clearscope_dataset(dataset_name):
        return str(semantic_rules.get("clearscope", "")).strip()
    if is_theia_dataset(dataset_name):
        return str(semantic_rules.get("theia", "")).strip()
    if is_cadets_dataset(dataset_name):
        return str(semantic_rules.get("cadets", "")).strip()
    return ""


def validate_word2vec_semantic_mode_matches_config(config: SlimConfig) -> None:
    """Fail if a pretrained Word2Vec artifact uses a different semantic mode."""
    path = str(config.pretrained_residual_embedder_path).strip()
    if not path:
        return
    metadata = load_word2vec_semantic_metadata(path)
    if not metadata:
        return
    expected = _expected_semantic_mode_for_config(config)
    actual = str(metadata.get("semantic_mode", "")).strip()
    semantic_rules = metadata.get("semantic_rules", {})
    if not actual and isinstance(semantic_rules, Mapping):
        actual = _word2vec_semantic_rule_for_dataset(semantic_rules, config.dataset)
    if is_clearscope_dataset(config.dataset):
        actual = normalize_clearscope_semantic_mode(actual)
    if actual and actual != expected:
        raise ValueError(
            "Word2Vec semantic_mode mismatch: "
            f"config={expected} artifact={actual} path={path}",
        )


class CachedResidualEmbedder:
    """Residual embedder wrapper backed by the Phase3D sqlite z_t cache."""

    def __init__(self, embedder: ResidualEmbedder, cache: ResidualEmbeddingSqliteCache) -> None:
        self.embedder = embedder
        self.cache = cache
        self.latent_dim = int(getattr(embedder, "latent_dim", 0))

    def fit(self, token_sequences) -> None:
        """Delegate train-only fitting to the wrapped embedder."""
        self.embedder.fit(token_sequences)

    def encode(self, tokens: Sequence[str]) -> np.ndarray:
        """Encode one token sequence through the on-demand sqlite cache."""
        return self.cache.get_or_compute(tokens, self.embedder)

    def stats(self) -> dict[str, object]:
        """Return wrapped embedder stats plus cache counters."""
        stats = dict(self.embedder.stats())
        stats["residual_embed_cache"] = self.cache.stats()
        return stats

    def state_dict(self) -> dict[str, object]:
        """Return the wrapped embedder state without cache entries."""
        if hasattr(self.embedder, "state_dict"):
            return self.embedder.state_dict()
        raise TypeError("wrapped residual embedder does not expose state_dict()")


def _embedder_state_hash(embedder: ResidualEmbedder) -> str:
    """Return a stable hash for the residual embedder state metadata."""
    if isinstance(embedder, CachedResidualEmbedder):
        embedder = embedder.embedder
    if not hasattr(embedder, "state_dict"):
        return stable_json_hash(dict(embedder.stats()))
    state = dict(embedder.state_dict())
    model = state.pop("model", None)
    if model is not None:
        state["model_fingerprint"] = _gensim_model_fingerprint(model)
    return stable_json_hash(state)


def _gensim_model_fingerprint(model: Any) -> dict[str, Any] | str:
    """Return a stable summary for a gensim model without serializing Python objects."""
    try:
        wv = getattr(model, "wv")
        keys = [str(key) for key in getattr(wv, "index_to_key", [])]
        vectors = np.asarray(getattr(wv, "vectors"), dtype=np.float32)
        vector_hash = hashlib.sha256(vectors.tobytes(order="C")).hexdigest()
        return {
            "model_type": type(model).__name__,
            "vector_size": int(getattr(model, "vector_size", vectors.shape[-1])),
            "vocab_size": int(len(keys)),
            "keys_hash": stable_json_hash(keys),
            "vectors_shape": [int(value) for value in vectors.shape],
            "vectors_sha256": vector_hash,
        }
    except Exception:
        return str(model)


def _resolved_residual_embed_cache_path(config: SlimConfig) -> Path:
    """Return the configured Phase3D residual embedding sqlite cache path."""
    raw = str(config.residual_embed_cache_path).strip()
    if not raw:
        raw = str(SlimConfig.residual_embed_cache_path)
    return Path(raw.replace("{DATASET}", str(config.dataset)))


def _residual_embed_cache_enabled(config: SlimConfig) -> bool:
    return str(config.residual_embed_cache_mode).strip().lower() in {"auto", "readwrite"}


def _make_residual_embed_cache_fingerprint(
    config: SlimConfig,
    embedder: ResidualEmbedder,
    process_cfg: ProcessSemanticConfig | None,
) -> str:
    """Return the sqlite residual embedding cache fingerprint string."""
    latent_dim = _effective_latent_dim(config, process_cfg)
    return residual_embedding_cache_fingerprint(
        embedder_path=str(config.pretrained_residual_embedder_path),
        embedder_dim=int(getattr(embedder, "latent_dim", latent_dim)),
        word2vec_window=int(config.word2vec_window),
        z_dim=int(latent_dim),
        embedder_state_hash=_embedder_state_hash(embedder),
    )


def validate_checkpoint_embedder_fingerprints(
    payload: Mapping[str, Any],
    embedder: ResidualEmbedder,
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
) -> None:
    """Fail fast if a checkpoint was trained with a different residual embedder."""
    checkpoint_embedder_hash = str(payload.get("embedder_fingerprint", ""))
    current_embedder_hash = _embedder_state_hash(embedder)
    if checkpoint_embedder_hash and checkpoint_embedder_hash != current_embedder_hash:
        raise ValueError("SSPM checkpoint embedder fingerprint mismatch")
    checkpoint_cache = payload.get("residual_embedding_cache", {})
    checkpoint_cache_hash = ""
    if isinstance(checkpoint_cache, Mapping):
        checkpoint_cache_hash = str(checkpoint_cache.get("residual_embed_cache_fingerprint", ""))
    if checkpoint_cache_hash:
        current_cache_hash = _make_residual_embed_cache_fingerprint(config, embedder, process_cfg)
        if checkpoint_cache_hash != current_cache_hash:
            raise ValueError("SSPM checkpoint residual embed cache fingerprint mismatch")


def maybe_wrap_residual_embed_cache(
    embedder: ResidualEmbedder,
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
    cache_refs: list[ResidualEmbeddingSqliteCache] | None = None,
) -> ResidualEmbedder:
    """Wrap the embedder with the on-demand sqlite cache when configured."""
    if not _residual_embed_cache_enabled(config):
        return embedder
    fingerprint = _make_residual_embed_cache_fingerprint(config, embedder, process_cfg)
    cache = ResidualEmbeddingSqliteCache(
        _resolved_residual_embed_cache_path(config),
        fingerprint,
        z_dim=_effective_latent_dim(config, process_cfg),
    )
    if cache_refs is not None:
        cache_refs.append(cache)
    return CachedResidualEmbedder(embedder, cache)


def _residual_embed_cache_summary(
    embedder: ResidualEmbedder,
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
) -> dict[str, Any]:
    """Return JSON-safe residual embedding cache metadata."""
    cache = getattr(embedder, "cache", None)
    if cache is None:
        return {
            "residual_embed_cache_mode": str(config.residual_embed_cache_mode),
            "residual_embed_cache_enabled": False,
            "residual_embed_cache_path": str(_resolved_residual_embed_cache_path(config)),
            "residual_embed_cache_fingerprint": "",
        }
    return {
        "residual_embed_cache_mode": str(config.residual_embed_cache_mode),
        "residual_embed_cache_enabled": True,
        "residual_embed_cache_path": str(cache.path),
        "residual_embed_cache_fingerprint": str(cache.fingerprint),
        "residual_embed_cache_stats": cache.stats(),
    }


def _phase3e_checkpoint_metadata(
    config: SlimConfig,
    phase3e_meta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return small Phase3E checkpoint metadata, excluding cache arrays and rows."""
    metadata = {
        "target_mode": str(config.sspm_target_mode),
        "node_word2vec_source": str(config.node_word2vec_source),
        "train_backend": str(config.sspm_train_backend),
        "infer_backend": str(config.sspm_infer_backend),
        "node_embedding_dim": int(config.node_embedding_dim),
        "action_embedding_dim": int(config.action_embedding_dim),
        "event_index_cache_mode": str(config.event_index_cache_mode),
        "event_index_debug_fields": bool(config.event_index_debug_fields),
        "x_context_memmap_enabled": bool(config.x_context_memmap_enabled),
        "phase3e_precompute_only": bool(config.phase3e_precompute_only),
        "sspm_torch_batch_events": int(config.sspm_torch_batch_events),
        "sspm_torch_lr": float(config.sspm_torch_lr),
        "sspm_torch_weight_decay": float(config.sspm_torch_weight_decay),
        "sspm_torch_device": str(config.sspm_torch_device),
    }
    extra = dict(phase3e_meta or {})
    _validate_phase3e_checkpoint_metadata(extra)
    for key, value in extra.items():
        metadata[str(key)] = value
    if "fingerprint" not in metadata:
        metadata["fingerprint"] = stable_json_hash(metadata)
    _validate_phase3e_checkpoint_metadata(metadata)
    return metadata


def _validate_phase3e_checkpoint_metadata(value: Any) -> None:
    """Fail fast when Phase3E checkpoint metadata contains large runtime state."""
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise TypeError("phase3e checkpoint metadata must be a mapping")
    if not value:
        return
    _reject_forbidden_phase3e_checkpoint_keys(value, ("phase3e",))
    _validate_phase3e_metadata_allowed_keys(value, ("phase3e",))
    _validate_phase3e_metadata_value(value, ("phase3e",))


def _reject_forbidden_phase3e_checkpoint_keys(value: Any, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            current_path = (*path, key_text)
            if key_text in _PHASE3E_CHECKPOINT_FORBIDDEN_KEYS:
                joined = ".".join(current_path)
                raise ValueError(f"forbidden phase3e metadata key: {joined}")
            _reject_forbidden_phase3e_checkpoint_keys(item, current_path)
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _reject_forbidden_phase3e_checkpoint_keys(item, (*path, str(index)))


def _validate_phase3e_metadata_allowed_keys(
    value: Any,
    path: tuple[str, ...],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            current_path = (*path, key_text)
            key_lower = key_text.lower()
            if any(term in key_lower for term in _PHASE3E_METADATA_LEAKAGE_KEY_TERMS):
                joined = ".".join(current_path)
                raise ValueError(f"phase3e leakage metadata key is not allowed: {joined}")
            if len(path) == 1 and key_text not in _PHASE3E_METADATA_ALLOWED_KEYS:
                joined = ".".join(current_path)
                raise ValueError(f"phase3e metadata key is not allowed: {joined}")
            _validate_phase3e_metadata_allowed_keys(item, current_path)
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_phase3e_metadata_allowed_keys(item, (*path, str(index)))


def _validate_phase3e_metadata_value(value: Any, path: tuple[str, ...]) -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(float(value)):
            joined = ".".join(path)
            raise ValueError(f"phase3e metadata value must be finite: {joined}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_phase3e_metadata_value(item, (*path, str(key)))
        return
    if isinstance(value, list | tuple):
        if len(value) > _PHASE3E_METADATA_MAX_LIST_LENGTH:
            joined = ".".join(path)
            raise ValueError(f"phase3e metadata value list is too large: {joined}")
        for index, item in enumerate(value):
            _validate_phase3e_metadata_value(item, (*path, str(index)))
        return
    joined = ".".join(path)
    raise ValueError(f"phase3e metadata value must be primitive-only: {joined}")


def close_residual_embed_caches(cache_refs: Sequence[ResidualEmbeddingSqliteCache]) -> None:
    """Close all opened residual embedding sqlite cache handles."""
    for cache in cache_refs:
        cache.close()


def save_sspm_checkpoint(
    path: str | Path,
    *,
    config: SlimConfig,
    model: SSPMLowRankModel,
    embedder: ResidualEmbedder,
    process_cfg: ProcessSemanticConfig | None,
    train_count: int,
    validation_count: int,
    phase3e_meta: Mapping[str, Any] | None = None,
) -> Path:
    """Save a Phase3D SSPM checkpoint without online state memory."""
    checkpoint_path = Path(path)
    if not str(checkpoint_path):
        raise ValueError("sspm_checkpoint_path must not be empty")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SSPM_CHECKPOINT_SCHEMA_VERSION,
        "method": METHOD_VERSION,
        "config_effective": asdict(config),
        "model": model.state_dict(),
        "context_dim": int(model.context_dim),
        "target_dim": int(model.target_dim),
        "state_dim": int(model.state_dim),
        "rank": int(model.rank),
        "context_mode": str(model.config.context_mode),
        "context_action_mode": str(model.config.context_action_mode),
        "global_context_mode": str(model.config.global_context_mode),
        "residual_score_mode": str(model.config.residual_score_mode),
        "residual_calibration": str(model.config.residual_calibration),
        "embedder_fingerprint": _embedder_state_hash(
            embedder.embedder if isinstance(embedder, CachedResidualEmbedder) else embedder,
        ),
        "residual_embedding_cache": _residual_embed_cache_summary(
            embedder,
            config,
            process_cfg,
        ),
        "event_counts_actual": {
            "train_events_actual": int(train_count),
            "validation_events_actual": int(validation_count),
        },
        "leakage_contract": {
            "contains_train_state_memory": False,
            "contains_validation_state_memory": False,
            "contains_test_scores": False,
            "contains_test_labels": False,
            "contains_test_rankings": False,
            "ground_truth_used": False,
        },
    }
    if str(config.sspm_target_mode) == "node_action_semantic_mean":
        payload["phase3e"] = _phase3e_checkpoint_metadata(config, phase3e_meta)
    validate_sspm_checkpoint_payload(payload)
    tmp_path = checkpoint_path.with_name(
        f".{checkpoint_path.name}.{os.getpid()}.{time.time_ns()}.tmp",
    )
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, checkpoint_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return checkpoint_path


def load_sspm_checkpoint(path: str | Path, config: SlimConfig) -> tuple[SSPMLowRankModel, dict[str, Any]]:
    """Load and validate a Phase3D SSPM checkpoint for load-and-infer."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SSPM checkpoint not found: {checkpoint_path}")
    with checkpoint_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("SSPM checkpoint payload must be a dict")
    validate_sspm_checkpoint_payload(payload)
    model = SSPMLowRankModel.from_state_dict(payload["model"])
    _validate_loaded_checkpoint_against_config(payload, model, config)
    return model, payload


def load_sspm_checkpoint_state_for_conditional(
    path: str | Path,
    config: SlimConfig,
) -> tuple[SSPMLowRankModel, dict[str, Any]]:
    """Load state-model and calibration parameters for Phase3G conditional scoring."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SSPM checkpoint not found: {checkpoint_path}")
    with checkpoint_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("SSPM checkpoint payload must be a dict")
    validate_sspm_checkpoint_payload(payload)
    model = SSPMLowRankModel.from_state_dict(payload["model"])
    checks = {
        "state_model": (str(model.config.state_model), str(config.sspm_state_model)),
        "target_dim": (int(model.target_dim), int(config.sspm_target_dim)),
        "state_dim": (int(model.state_dim), int(config.sspm_state_dim)),
        "rank": (int(model.rank), int(config.rank)),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"SSPM checkpoint {key} mismatch: actual={actual} expected={expected}")
    if str(config.sspm_residual_score_mode) == "var_calibrated":
        if model.residual_calibration_model is None:
            raise ValueError("E5 conditional load_and_infer requires checkpoint calibration stats")
    if str(config.sspm_update_gate_mode) == "quantile":
        gate = getattr(model, "update_gate_calibrator", None)
        if gate is None or getattr(gate, "threshold", None) is None:
            raise ValueError("E5 conditional load_and_infer requires checkpoint update gate threshold")
    return model, payload


def _phase3e_load_and_infer_runtime_config_summary(
    checkpoint_payload: Mapping[str, Any],
    model: SSPMLowRankModel,
    config: SlimConfig,
) -> dict[str, Any]:
    """Return checkpoint/request/actual runtime OFSM config metadata."""
    checkpoint_model = checkpoint_payload.get("model", {})
    checkpoint_model = checkpoint_model if isinstance(checkpoint_model, Mapping) else {}
    checkpoint_config = checkpoint_model.get("config", {})
    checkpoint_config = checkpoint_config if isinstance(checkpoint_config, Mapping) else {}
    return {
        "checkpoint_state_merge_mode": str(checkpoint_config.get("state_merge_mode", "")),
        "requested_state_merge_mode": str(config.sspm_state_merge_mode),
        "actual_state_merge_mode": str(model.config.state_merge_mode),
        "checkpoint_state_memory_mode": str(checkpoint_config.get("state_memory_mode", "")),
        "requested_state_memory_mode": str(config.state_memory_mode),
        "actual_state_memory_mode": str(model.config.state_memory_mode),
        "checkpoint_state_memory_policy": str(checkpoint_config.get("state_memory_policy", "")),
        "requested_state_memory_policy": str(config.sspm_state_memory_policy),
        "actual_state_memory_policy": str(model.config.state_memory_policy),
        "state_merge_threshold": float(model.config.state_merge_threshold),
        "state_merge_random_prob": float(model.config.state_merge_random_prob),
        "state_merge_random_seed": int(model.config.state_merge_random_seed),
    }


def _phase3e_apply_load_and_infer_runtime_config(
    model: SSPMLowRankModel,
    config: SlimConfig,
    checkpoint_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply runtime-only memory/OFSM config after loading checkpoint parameters."""
    runtime_config = _make_sspm_config(config, None)
    model.config.state_memory_mode = str(runtime_config.state_memory_mode)
    model.config.state_memory_policy = str(runtime_config.state_memory_policy)
    model.config.active_max_nodes = int(runtime_config.active_max_nodes)
    model.config.probationary_max_nodes = int(runtime_config.probationary_max_nodes)
    model.config.probationary_min_count = int(runtime_config.probationary_min_count)
    model.config.lru_evict_batch = int(runtime_config.lru_evict_batch)
    model.config.alert_protect_events = int(runtime_config.alert_protect_events)
    model.config.state_merge_mode = str(runtime_config.state_merge_mode)
    model.config.state_merge_scope = str(runtime_config.state_merge_scope)
    model.config.state_merge_threshold = float(runtime_config.state_merge_threshold)
    model.config.state_merge_trunc_ratio = float(runtime_config.state_merge_trunc_ratio)
    model.config.state_merge_use_rfft = bool(runtime_config.state_merge_use_rfft)
    model.config.state_merge_use_real_only = bool(runtime_config.state_merge_use_real_only)
    model.config.state_merge_center_state = bool(runtime_config.state_merge_center_state)
    model.config.state_merge_normalize = str(runtime_config.state_merge_normalize)
    model.config.state_merge_eps = float(runtime_config.state_merge_eps)
    model.config.state_merge_min_cluster_size = int(runtime_config.state_merge_min_cluster_size)
    model.config.state_merge_copy_on_write = bool(runtime_config.state_merge_copy_on_write)
    model.config.state_merge_random_prob = float(runtime_config.state_merge_random_prob)
    model.config.state_merge_random_seed = int(runtime_config.state_merge_random_seed)
    model.config.state_merge_diagnostics_max_rows = int(
        runtime_config.state_merge_diagnostics_max_rows,
    )
    model.config.state_merge_match_backend = str(runtime_config.state_merge_match_backend)
    model.config.state_merge_candidate_cap = int(runtime_config.state_merge_candidate_cap)
    model.config.state_merge_interval = int(runtime_config.state_merge_interval)
    model.config.state_merge_min_count = int(runtime_config.state_merge_min_count)
    model.config.state_merge_diagnostics_enabled = bool(
        runtime_config.state_merge_diagnostics_enabled,
    )
    model.reset_state_memory()
    summary = _phase3e_load_and_infer_runtime_config_summary(checkpoint_payload, model, config)
    setattr(model, "load_and_infer_runtime_config_summary", dict(summary))
    return summary


def validate_sspm_checkpoint_payload(payload: Mapping[str, Any]) -> None:
    """Validate the Phase3D checkpoint leakage and shape contract."""
    if payload.get("schema") != SSPM_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("SSPM checkpoint schema mismatch")
    if payload.get("method") != METHOD_VERSION:
        raise ValueError("SSPM checkpoint method mismatch")
    _reject_forbidden_checkpoint_top_level_keys(payload)
    model_state = payload.get("model")
    _validate_sspm_state(model_state)
    leakage = payload.get("leakage_contract", {})
    if isinstance(leakage, Mapping) and bool(leakage.get("contains_train_state_memory", False)):
        raise ValueError("SSPM checkpoint must not contain train state memory")
    if "phase3e" in payload:
        _validate_phase3e_checkpoint_metadata(payload["phase3e"])


def _reject_forbidden_checkpoint_top_level_keys(payload: Mapping[str, Any]) -> None:
    """Reject large or leakage-prone payload entries at checkpoint top level."""
    present = sorted(_PHASE3E_CHECKPOINT_FORBIDDEN_KEYS.intersection(payload))
    if present:
        raise ValueError(f"forbidden checkpoint key: {present[0]}")


def _validate_loaded_checkpoint_against_config(
    payload: Mapping[str, Any],
    model: SSPMLowRankModel,
    config: SlimConfig,
) -> None:
    """Fail fast if a checkpoint does not match the requested inference config."""
    action_dim = 0
    if str(config.sspm_context_mode) == "with_action":
        action_dim = 10 if str(config.sspm_context_action_mode) == "raw_orthrus10" else int(
            config.action_count,
        )
    global_dim = (
        int(config.sspm_target_dim)
        if str(config.sspm_global_context_mode) == "with_global"
        else 0
    )
    requested_context_dim = global_dim + 2 * int(config.sspm_target_dim)
    requested_context_dim += action_dim + 2 * len(ENTITY_TYPES)
    checks = {
        "state_model": (str(model.config.state_model), str(config.sspm_state_model)),
        "context_dim": (int(model.context_dim), int(requested_context_dim)),
        "target_dim": (int(model.target_dim), int(config.sspm_target_dim)),
        "state_dim": (int(model.state_dim), int(config.sspm_state_dim)),
        "rank": (int(model.rank), int(config.rank)),
        "context_mode": (str(model.config.context_mode), str(config.sspm_context_mode)),
        "context_action_mode": (
            str(model.config.context_action_mode),
            str(config.sspm_context_action_mode),
        ),
        "global_context_mode": (
            str(model.config.global_context_mode),
            str(config.sspm_global_context_mode),
        ),
    }
    for key, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(f"SSPM checkpoint {key} mismatch: actual={actual} expected={expected}")
    if str(config.sspm_residual_score_mode) == "var_calibrated":
        if model.residual_calibration_model is None:
            raise ValueError("E5 load_and_infer requires checkpoint calibration stats")
    if str(config.sspm_update_gate_mode) == "quantile":
        gate = getattr(model, "update_gate_calibrator", None)
        if gate is None or getattr(gate, "threshold", None) is None:
            raise ValueError("E5 load_and_infer requires checkpoint update gate threshold")


def train_from_stream(
    rows,
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None = None,
    process_audit: ProcessSemanticAudit | None = None,
) -> tuple[SSPMLowRankModel, ResidualEmbedder, int]:
    """Train SSPM parameters using train rows only."""
    embedder_loaded = bool(str(config.pretrained_residual_embedder_path).strip())
    if embedder_loaded:
        _stage_log(
            config,
            "residual_embedder_load_start",
            path=str(config.pretrained_residual_embedder_path),
        )
        embedder = load_pretrained_residual_embedder(
            str(config.pretrained_residual_embedder_path),
            expected_dim=_effective_latent_dim(config, process_cfg),
        )
        _stage_log(
            config,
            "residual_embedder_load_end",
            path=str(config.pretrained_residual_embedder_path),
            embedder_loaded=True,
        )
    else:
        embedder = build_residual_embedder(_make_residual_embedding_config(config, process_cfg))
    model = SSPMLowRankModel(_make_sspm_config(config, None))
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    train_items: list[tuple[dict[str, Any], list[str]]] = []
    train_sentences: list[list[str]] = []
    train_vocab: set[str] = set()
    count = 0
    progress_interval = int(config.progress_interval_events)
    _stage_log(config, "train_read_start", embedding_method=config.semantic_embedding_method)
    for row in rows:
        if process_cfg is not None:
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
        tokens = _residual_tokens_for_row(row, config, process_cfg)
        train_items.append((fields, tokens))
        train_sentences.append(tokens)
        train_vocab.update(tokens)
        count += 1
        if progress_interval > 0 and count % progress_interval == 0:
            _stage_log(config, "train_read_progress", count=count, vocab=len(train_vocab))
    _stage_log(config, "train_read_end", count=count, vocab=len(train_vocab))
    if str(config.latent_dim_policy) == "auto_vocab":
        selected_dim = _latent_dim_from_vocab_size(len(train_vocab))
        config = _config_with_latent_dim(config, selected_dim)
        if not embedder_loaded:
            embedder = build_residual_embedder(_make_residual_embedding_config(config, process_cfg))
        model = SSPMLowRankModel(_make_sspm_config(config, process_cfg))
    if embedder_loaded:
        _stage_log(config, "embedding_fit_skip", count=count, embedder_loaded=True)
    else:
        _stage_log(
            config,
            "embedding_fit_start",
            count=count,
            method=config.semantic_embedding_method,
        )
        embedder.fit(train_sentences)
        _stage_log(config, "embedding_fit_end", count=count, method=config.semantic_embedding_method)
    _stage_log(config, "train_encode_start", count=count)
    for index, (fields, tokens) in enumerate(train_items, start=1):
        z = embedder.encode(tokens)
        contexts.append(model.make_context(fields))
        targets.append(z)
        model.update_states(fields, z)
        if progress_interval > 0 and index % progress_interval == 0:
            _stage_log(config, "train_encode_progress", count=index)
    _stage_log(config, "train_encode_end", count=len(train_items))
    if contexts:
        _stage_log(config, "sspm_train_start", count=len(contexts), epochs=config.sspm_epochs)
        model_training = model.train_batch(
            np.stack(contexts),
            np.stack(targets),
            log_prefix=f"[SSPM][{config.dataset}]",
        )
        _stage_log(config, "sspm_train_end", count=len(contexts))
    else:
        _stage_log(config, "sspm_train_skip", count=0)
        model_training = {}
    model.reset_state()
    model.training_stats = model_training
    return model, embedder, count


def train_stream_event_from_factory(
    row_factory,
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None = None,
    process_audit: ProcessSemanticAudit | None = None,
    embedder: ResidualEmbedder | None = None,
) -> tuple[SSPMLowRankModel, ResidualEmbedder, int]:
    """Train W1/W2/b from repeated train streams without full train arrays."""
    if not _embedder_loaded(config) and embedder is None:
        raise ValueError("stream_event train requires pretrained residual embedder")
    cache_refs: list[ResidualEmbeddingSqliteCache] = []
    if embedder is None:
        _stage_log(
            config,
            "residual_embedder_load_start",
            path=str(config.pretrained_residual_embedder_path),
        )
        embedder = load_pretrained_residual_embedder(
            str(config.pretrained_residual_embedder_path),
            expected_dim=_effective_latent_dim(config, process_cfg),
        )
        embedder = maybe_wrap_residual_embed_cache(embedder, config, process_cfg, cache_refs)
        _stage_log(
            config,
            "residual_embedder_load_end",
            path=str(config.pretrained_residual_embedder_path),
            embedder_loaded=True,
        )
    model = SSPMLowRankModel(_make_sspm_config(config, process_cfg))
    progress_interval = int(config.progress_interval_events)
    train_count_first_epoch = 0
    train_rss_peak_mb = _current_rss_mb()

    def rows_for_epoch(epoch: int):
        nonlocal train_count_first_epoch, train_rss_peak_mb
        epoch_count = 0
        model.reset_state_memory()
        _stage_log(
            config,
            "train_stream_epoch_start",
            epoch=int(epoch) + 1,
            epochs=int(config.sspm_epochs),
            train_data_mode="stream_event",
        )
        for row in row_factory():
            if process_cfg is not None:
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
            context = model.make_context(fields)
            model.update_states(fields, z)
            epoch_count += 1
            if int(epoch) == 0:
                train_count_first_epoch += 1
            train_rss_peak_mb = max(float(train_rss_peak_mb), _current_rss_mb())
            if progress_interval > 0 and epoch_count % progress_interval == 0:
                _stage_log(
                    config,
                    "train_stream_epoch_progress",
                    epoch=int(epoch) + 1,
                    count=epoch_count,
                )
            yield context, z
        _stage_log(
            config,
            "train_stream_epoch_end",
            epoch=int(epoch) + 1,
            count=epoch_count,
        )

    _stage_log(
        config,
        "sspm_train_start",
        count="streaming",
        epochs=config.sspm_epochs,
        train_data_mode="stream_event",
        batch_events=config.sspm_train_batch_events,
    )
    model_training = model.train_stream_batches(
        rows_for_epoch,
        epochs=int(config.sspm_epochs),
        batch_events=int(config.sspm_train_batch_events),
        log_prefix=f"[SSPM][{config.dataset}][STREAM]",
    )
    model_training["train_rss_peak_mb"] = float(train_rss_peak_mb)
    model_training.setdefault("early_stop_reason", model_training.get("stop_reason", ""))
    _stage_log(
        config,
        "sspm_train_end",
        count=int(train_count_first_epoch),
        train_events_seen_total=int(model_training.get("train_events_seen_total", 0)),
    )
    model.reset_state()
    model.training_stats = model_training
    return model, embedder, int(train_count_first_epoch)


def _uses_real_diag_online_gamma_training(config: SlimConfig) -> bool:
    return (
        str(config.sspm_state_model) == "real_diag_learnable"
        and bool(config.real_diag_train_gamma)
    )


def _parse_bool(value: object) -> bool:
    """Parse CLI boolean-like values without accepting ambiguous text."""
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def train_real_diag_gamma_from_stream(
    rows,
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None = None,
    process_audit: ProcessSemanticAudit | None = None,
) -> tuple[SSPMLowRankModel, ResidualEmbedder, int]:
    """Train E3 online with real-diag gamma sensitivity using train rows only."""
    embedder_loaded = bool(str(config.pretrained_residual_embedder_path).strip())
    if embedder_loaded:
        _stage_log(
            config,
            "residual_embedder_load_start",
            path=str(config.pretrained_residual_embedder_path),
        )
        embedder = load_pretrained_residual_embedder(
            str(config.pretrained_residual_embedder_path),
            expected_dim=_effective_latent_dim(config, process_cfg),
        )
        _stage_log(
            config,
            "residual_embedder_load_end",
            path=str(config.pretrained_residual_embedder_path),
            embedder_loaded=True,
        )
    else:
        embedder = build_residual_embedder(_make_residual_embedding_config(config, process_cfg))
    model = SSPMLowRankModel(_make_sspm_config(config, process_cfg))
    train_items: list[tuple[dict[str, Any], list[str]]] = []
    train_sentences: list[list[str]] = []
    train_vocab: set[str] = set()
    count = 0
    progress_interval = int(config.progress_interval_events)
    if embedder_loaded:
        _stage_log(config, "train_stream_start", embedding_method=config.semantic_embedding_method)

        def encoded_items():
            nonlocal count
            for row in rows:
                if process_cfg is not None:
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
                tokens = _residual_tokens_for_row(row, config, process_cfg)
                count += 1
                if progress_interval > 0 and count % progress_interval == 0:
                    _stage_log(config, "train_stream_progress", count=count)
                yield fields, embedder.encode(tokens)
            _stage_log(config, "train_stream_end", count=count)

        _stage_log(
            config,
            "sspm_train_start",
            count="streaming",
            epochs=1,
            configured_epochs=config.sspm_epochs,
            real_diag_train_gamma=True,
        )
        model_training = model.train_real_diag_online(
            encoded_items(),
            log_prefix=f"[SSPM][{config.dataset}][E3_GAMMA]",
        )
        count = int(model_training.get("train_events_actual", count))
        _stage_log(config, "sspm_train_end", count=count, real_diag_train_gamma=True)
    else:
        _stage_log(config, "train_read_start", embedding_method=config.semantic_embedding_method)
        for row in rows:
            if process_cfg is not None:
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
            tokens = _residual_tokens_for_row(row, config, process_cfg)
            train_items.append((fields, tokens))
            train_sentences.append(tokens)
            train_vocab.update(tokens)
            count += 1
            if progress_interval > 0 and count % progress_interval == 0:
                _stage_log(config, "train_read_progress", count=count, vocab=len(train_vocab))
        _stage_log(config, "train_read_end", count=count, vocab=len(train_vocab))
        _stage_log(
            config,
            "embedding_fit_start",
            count=count,
            method=config.semantic_embedding_method,
        )
        embedder.fit(train_sentences)
        _stage_log(config, "embedding_fit_end", count=count, method=config.semantic_embedding_method)

        def encoded_items_from_cache():
            _stage_log(config, "train_encode_start", count=count)
            for index, (fields, tokens) in enumerate(train_items, start=1):
                z = embedder.encode(tokens)
                if progress_interval > 0 and index % progress_interval == 0:
                    _stage_log(config, "train_encode_progress", count=index)
                yield fields, z
            _stage_log(config, "train_encode_end", count=len(train_items))

        if count:
            _stage_log(
                config,
                "sspm_train_start",
                count=count,
                epochs=1,
                configured_epochs=config.sspm_epochs,
                real_diag_train_gamma=True,
            )
            model_training = model.train_real_diag_online(
                encoded_items_from_cache(),
                log_prefix=f"[SSPM][{config.dataset}][E3_GAMMA]",
            )
            _stage_log(config, "sspm_train_end", count=count, real_diag_train_gamma=True)
        else:
            _stage_log(config, "sspm_train_skip", count=0)
            model_training = {}
    model.training_stats = model_training
    model.real_diag_gamma_training_stats = dict(model_training)
    model.reset_state()
    return model, embedder, count


def train_real_diag_gamma_from_factory(
    row_factory,
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None = None,
    process_audit: ProcessSemanticAudit | None = None,
    embedder: ResidualEmbedder | None = None,
) -> tuple[SSPMLowRankModel, ResidualEmbedder, int]:
    """Train E3 real-diag gamma from repeated train streams without full train cache."""
    if not _embedder_loaded(config) and embedder is None:
        raise ValueError("E3 stream_event train requires pretrained residual embedder")
    cache_refs: list[ResidualEmbeddingSqliteCache] = []
    if embedder is None:
        _stage_log(
            config,
            "residual_embedder_load_start",
            path=str(config.pretrained_residual_embedder_path),
        )
        embedder = load_pretrained_residual_embedder(
            str(config.pretrained_residual_embedder_path),
            expected_dim=_effective_latent_dim(config, process_cfg),
        )
        embedder = maybe_wrap_residual_embed_cache(embedder, config, process_cfg, cache_refs)
        _stage_log(
            config,
            "residual_embedder_load_end",
            path=str(config.pretrained_residual_embedder_path),
            embedder_loaded=True,
        )
    model = SSPMLowRankModel(_make_sspm_config(config, process_cfg))
    progress_interval = int(config.progress_interval_events)
    train_count_first_epoch = 0
    train_events_seen_total = 0
    loss_by_epoch: list[float] = []
    train_rss_peak_mb = _current_rss_mb()
    stopped_early = False
    stop_reason = ""
    best_loss = float("inf")
    plateau_epochs = 0
    min_delta = float(config.sspm_early_stop_min_delta)
    patience = int(config.sspm_early_stop_patience)
    for epoch in range(max(int(config.sspm_epochs), 1)):
        epoch_count = 0
        _stage_log(
            config,
            "train_stream_epoch_start",
            epoch=epoch + 1,
            epochs=config.sspm_epochs,
            train_data_mode="stream_event",
            real_diag_train_gamma=True,
        )

        def encoded_items():
            nonlocal epoch_count, train_count_first_epoch, train_events_seen_total
            nonlocal train_rss_peak_mb
            for row in row_factory():
                if process_cfg is not None:
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
                epoch_count += 1
                train_events_seen_total += 1
                if epoch == 0:
                    train_count_first_epoch += 1
                train_rss_peak_mb = max(float(train_rss_peak_mb), _current_rss_mb())
                if progress_interval > 0 and epoch_count % progress_interval == 0:
                    _stage_log(
                        config,
                        "train_stream_epoch_progress",
                        epoch=epoch + 1,
                        count=epoch_count,
                    )
                yield fields, z

        model_training = model.train_real_diag_online(
            encoded_items(),
            log_prefix=f"[SSPM][{config.dataset}][E3_GAMMA]",
        )
        epoch_loss = float(model_training.get("final_loss", model_training.get("loss", 0.0)))
        loss_by_epoch.append(epoch_loss)
        _stage_log(
            config,
            "train_stream_epoch_end",
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
    gamma_stats = dict(getattr(model, "real_diag_gamma_training_stats", {}) or {})
    gamma_stats.update(
        {
            "train_data_mode": "stream_event",
            "epochs": int(config.sspm_epochs),
            "train_epochs_completed": int(len(loss_by_epoch)),
            "epochs_completed": int(len(loss_by_epoch)),
            "loss_by_epoch": list(loss_by_epoch),
            "loss": float(np.mean(loss_by_epoch)) if loss_by_epoch else 0.0,
            "final_loss": float(loss_by_epoch[-1]) if loss_by_epoch else 0.0,
            "train_events_actual": int(train_count_first_epoch),
            "train_events_seen_total": int(train_events_seen_total),
            "batch_events": 1,
            "train_rss_peak_mb": float(train_rss_peak_mb),
            "stopped_early": bool(stopped_early),
            "stop_reason": str(stop_reason),
            "early_stop_reason": str(stop_reason),
            "early_stop_min_delta": float(min_delta),
            "early_stop_patience": int(patience),
        },
    )
    model.training_stats = gamma_stats
    model.real_diag_gamma_training_stats = dict(gamma_stats)
    model.reset_state()
    return model, embedder, int(train_count_first_epoch)


def calibrate_from_stream(
    rows,
    config: SlimConfig,
    model: SSPMLowRankModel,
    embedder: ResidualEmbedder,
    process_cfg: ProcessSemanticConfig | None = None,
    process_audit: ProcessSemanticAudit | None = None,
) -> tuple[np.ndarray, int]:
    """Calibrate validation residual tail distributions."""
    residual_validation: list[float] = []
    count = 0
    progress_interval = int(config.progress_interval_events)
    model.reset_state()
    _stage_log(config, "validation_encode_start")
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
        residual_score = model.residual_score(
            pred,
            z,
            action=fields.get("raw_action", fields.get("action")),
        )
        residual_validation.append(residual_score)
        model.update_states(fields, z)
        count += 1
        if progress_interval > 0 and count % progress_interval == 0:
            _stage_log(config, "validation_encode_progress", count=count)
    _stage_log(config, "validation_encode_end", count=count)
    residual_scores = np.asarray(residual_validation, dtype=np.float32)
    model.reset_state()
    return residual_scores, count


def _phase3e_target_from_index_row(
    row: np.void,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
) -> np.ndarray:
    """Build one Phase3E node/action semantic target from a compact event-index row."""
    _phase3e_validate_index_row(row)
    return compute_node_action_target(
        node_embeddings,
        action_embeddings,
        int(row["src_node_idx"]),
        int(row["dst_node_idx"]),
        int(row["action_id"]),
    )


def _phase3e_fields_from_index_row(row: np.void) -> dict[str, Any]:
    """Map one compact Phase3E event-index row into SSPM causal context fields."""
    _phase3e_validate_index_row(row)
    action_id = int(row["action_id"])
    if action_id < 0 or action_id >= len(ORTHRUS10_ACTION_NAMES):
        raise ValueError(f"invalid action_id for ORTHRUS10 action table: {action_id}")
    raw_action = str(ORTHRUS10_ACTION_NAMES[action_id])
    if raw_action not in ORTHRUS10_EVENT_TYPES:
        raise ValueError(f"action_id {action_id} maps outside ORTHRUS10_EVENT_TYPES")

    src_type = int(row["src_type_id"])
    dst_type = int(row["dst_type_id"])
    src_type_name = _phase3e_entity_type_name("src_type_id", src_type)
    dst_type_name = _phase3e_entity_type_name("dst_type_id", dst_type)
    return {
        "info_src": int(row["src_node_idx"]),
        "info_dst": int(row["dst_node_idx"]),
        "src_type": src_type,
        "dst_type": dst_type,
        "info_src_type": src_type,
        "info_dst_type": dst_type,
        "src_type_name": src_type_name,
        "dst_type_name": dst_type_name,
        "object_type": dst_type_name,
        "src_role": src_type_name,
        "dst_role": dst_type_name,
        "action_id": action_id,
        "relation_id": action_id,
        "raw_action": raw_action,
        "action": raw_action,
        "action_token": raw_action_token(raw_action),
    }


def _phase3e_score_event_index_stream(
    config: SlimConfig,
    model: SSPMLowRankModel,
    event_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    enable_update_gate: bool,
) -> tuple[np.ndarray, int]:
    """Score a compact Phase3E event-index stream with context-before-update semantics."""
    del config
    records = _phase3e_event_index_array(event_index)
    residual_scores: list[float] = []
    model.reset_state()
    try:
        for row in records:
            fields = _phase3e_fields_from_index_row(row)
            z = _phase3e_target_from_index_row(row, node_embeddings, action_embeddings)
            pred = model.predict(fields)
            residual_score = model.residual_score(
                pred,
                z,
                action=fields.get("raw_action", fields.get("action")),
            )
            residual_scores.append(float(residual_score))
            model.update_states(
                fields,
                z,
                residual_score=float(residual_score) if bool(enable_update_gate) else None,
                enable_update_gate=bool(enable_update_gate),
            )
        return np.asarray(residual_scores, dtype=np.float32), int(len(residual_scores))
    finally:
        model.reset_state()


def _phase3f_fast_path_enabled(config: SlimConfig, model: SSPMLowRankModel) -> bool:
    """Return whether Phase3F/Phase3G online-minimal fast scoring can run."""
    state_merge_mode = str(config.sspm_state_merge_mode)
    score_head = str(config.sspm_score_head)
    state_model = str(config.sspm_state_model)
    model_state_model = str(model.config.state_model)
    merge_supported = state_merge_mode == "none" or (
        score_head == "conditional_action_semantic"
        and state_merge_mode in {"online_time_domain", "online_fourier", "random"}
    )
    state_supported = state_model == "ema_fixed" or (
        score_head == "conditional_action_semantic" and state_model == "s4d_complex_node"
    )
    model_state_supported = model_state_model == state_model
    update_gate_supported = str(config.sspm_update_gate_mode) == "none" or (
        score_head == "conditional_action_semantic" and state_model == "ema_fixed"
    )
    return (
        bool(config.sspm_infer_fast_path)
        and str(config.rss_profile_mode) == "online_minimal"
        and str(config.sspm_train_mode) == "load_and_infer"
        and str(config.sspm_target_mode) == "node_action_semantic_mean"
        and state_supported
        and merge_supported
        and model_state_supported
        and update_gate_supported
    )


def _phase3f_residual_scores_e2_none_fast(
    config: SlimConfig,
    model: SSPMLowRankModel,
    event_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return E2_NONE residual scores using chunked numeric Phase3E rows."""
    records = _phase3e_event_index_array(event_index)
    chunk_size = max(int(config.sspm_infer_chunk_events), 1)
    residual_scores = np.zeros((int(records.shape[0]),), dtype=np.float32)
    model.reset_state()
    try:
        for start in range(0, int(records.shape[0]), chunk_size):
            end = min(start + chunk_size, int(records.shape[0]))
            chunk = records[start:end]
            contexts = np.zeros((int(chunk.shape[0]), int(model.context_dim)), dtype=np.float32)
            targets = np.zeros((int(chunk.shape[0]), int(model.latent_dim)), dtype=np.float32)
            actions: list[str] = []
            for offset, row in enumerate(chunk):
                fields = _phase3e_fields_from_index_row(row)
                z = _phase3e_target_from_index_row(row, node_embeddings, action_embeddings)
                contexts[offset] = model.make_context(fields)
                targets[offset] = z
                actions.append(str(fields.get("raw_action", fields.get("action", ""))))
                model.update_states(fields, z, residual_score=None, enable_update_gate=False)
            preds = model.predict_from_context(contexts)
            for offset in range(int(chunk.shape[0])):
                residual_scores[start + offset] = np.float32(
                    model.residual_score(
                        preds[offset],
                        targets[offset],
                        action=actions[offset],
                    ),
                )
        return residual_scores, int(records.shape[0])
    finally:
        model.reset_state()


def _phase3f_legacy_residual_scores_vectorized(
    preds: np.ndarray,
    targets: np.ndarray,
    *,
    lambda_mse: float,
) -> np.ndarray:
    """Return legacy residual scores for a prediction/target batch."""
    pred = np.asarray(preds, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    pred_norm = np.linalg.norm(pred, axis=1, keepdims=True).astype(np.float32)
    target_norm = np.linalg.norm(target, axis=1, keepdims=True).astype(np.float32)
    pred_unit = np.divide(
        pred,
        np.maximum(pred_norm, np.float32(1e-8)),
        out=np.zeros_like(pred, dtype=np.float32),
        where=pred_norm > np.float32(1e-8),
    )
    target_unit = np.divide(
        target,
        np.maximum(target_norm, np.float32(1e-8)),
        out=np.zeros_like(target, dtype=np.float32),
        where=target_norm > np.float32(1e-8),
    )
    cosine = (np.float32(1.0) - np.sum(pred_unit * target_unit, axis=1)).astype(np.float32)
    mse = np.mean((pred_unit - target_unit) ** 2, axis=1).astype(np.float32)
    scores = cosine + np.float32(float(lambda_mse)) * mse
    return np.maximum(scores, np.float32(0.0)).astype(np.float32, copy=False)


def _phase3f_empirical_tail_scores_vectorized(
    values: np.ndarray,
    sorted_scores: np.ndarray,
) -> np.ndarray:
    """Return empirical tail scores for a residual-score batch."""
    residuals = np.asarray(values, dtype=np.float32)
    scores = np.asarray(sorted_scores, dtype=np.float32)
    count = int(scores.size)
    if count <= 0:
        raise ValueError("validation scores must not be empty")
    first_ge = np.searchsorted(scores, residuals, side="left").astype(np.int64)
    ge_count = np.int64(count) - first_ge
    probabilities = (ge_count.astype(np.float64) + 1.0) / (float(count) + 1.0)
    return (-np.log(probabilities)).astype(np.float32)


def _phase3e_synthetic_row_from_index_row(row: np.void, stream_pos: int) -> dict[str, Any]:
    """Build a label-free row shell for Phase3E online output from compact event_index."""
    fields = _phase3e_fields_from_index_row(row)
    src_type = str(fields.get("src_type_name", "unknown"))
    dst_type = str(fields.get("dst_type_name", "unknown"))
    return {
        "event_index": int(row["event_id"]),
        "timestamp_ns": int(stream_pos),
        "src_idx": int(row["src_node_idx"]),
        "dst_idx": int(row["dst_node_idx"]),
        "action": str(fields.get("raw_action", "")),
        "object_type": dst_type,
        "src_kind": src_type,
        "dst_kind": dst_type,
        "text": "",
    }


def _phase3f_raw_alert_row(
    row: np.void,
    stream_pos: int,
    event_score: float,
    threshold: float,
    residual_tail: float,
    residual_score: float,
    threshold_basis: str,
) -> dict[str, Any]:
    action_id = int(row["action_id"])
    src_type_id = int(row["src_type_id"])
    dst_type_id = int(row["dst_type_id"])
    action = str(ORTHRUS10_ACTION_NAMES[action_id])
    src_type = _phase3e_entity_type_name("src_type_id", src_type_id)
    dst_type = _phase3e_entity_type_name("dst_type_id", dst_type_id)
    return {
        "stream_pos": int(stream_pos),
        "event_index": int(row["event_id"]),
        "timestamp_ns": int(stream_pos),
        "src_idx": int(row["src_node_idx"]),
        "dst_idx": int(row["dst_node_idx"]),
        "info_src": int(row["src_node_idx"]),
        "info_dst": int(row["dst_node_idx"]),
        "action": action,
        "src_type": src_type,
        "dst_type": dst_type,
        "object_type": dst_type,
        "dst_role": dst_type,
        "event_score": float(event_score),
        "threshold": float(threshold),
        "threshold_basis": str(threshold_basis),
        "residual_tail": float(residual_tail),
        "residual_score": float(residual_score),
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
        "target_case": "",
        "target_case_threshold": "",
        "threshold_level": "",
        "threshold_group_key": "",
        "validation_group_count": "",
        "low_support_policy": "",
        "group_validation_max": "",
        "parent_threshold": "",
        "global_threshold": "",
        "final_threshold_source": "",
        "adaptive_margin_used": "",
        "validation_count_bucket": "",
        "endpoint_signature": "",
        "endpoint_pair_key": "",
        "src_endpoint_key": "",
        "endpoint_validation_pair_count": "",
        "src_endpoint_count": "",
        "endpoint_validation_count": "",
        "endpoint_action_key": "",
        "endpoint_suppression_mode": "",
        "endpoint_suppression_match_level": "",
        "endpoint_suppression_reason": "",
        **_ofsm_empty_node_snapshot("src"),
        **_ofsm_empty_node_snapshot("dst"),
    }


def _phase3f_e2_none_context_from_numeric_row(
    model: SSPMLowRankModel,
    row: np.void,
    type_one_hot: np.ndarray,
    action_one_hot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    """Build an E2_NONE context directly from compact numeric event-index fields."""
    src_idx = int(row["src_node_idx"])
    dst_idx = int(row["dst_node_idx"])
    action_id = int(row["action_id"])
    src_type_id = int(row["src_type_id"])
    dst_type_id = int(row["dst_type_id"])
    src_type_name = _phase3e_entity_type_name("src_type_id", src_type_id)
    dst_type_name = _phase3e_entity_type_name("dst_type_id", dst_type_id)
    action = str(ORTHRUS10_ACTION_NAMES[action_id])
    src_state = model.state_model.read(model.memory.read_state(src_idx, src_type_name))
    dst_state = model.state_model.read(model.memory.read_state(dst_idx, dst_type_name))
    model.last_context_snapshot = {
        "src_state_norm": float(np.linalg.norm(src_state)),
        "dst_state_norm": float(np.linalg.norm(dst_state)),
    }
    context = np.concatenate(
        [
            src_state,
            dst_state,
            action_one_hot[action_id],
            type_one_hot[src_type_id],
            type_one_hot[dst_type_id],
        ],
        axis=0,
    )
    return (
        context.astype(np.float32, copy=False),
        src_state.astype(np.float32, copy=False),
        dst_state.astype(np.float32, copy=False),
        action,
        src_type_name,
        dst_type_name,
    )


def _phase3f_e2_none_update_from_numeric_row(
    model: SSPMLowRankModel,
    row: np.void,
    z: np.ndarray,
    action: str,
    src_type_name: str,
    dst_type_name: str,
    residual_score: float,
) -> None:
    """Apply the exact E2/probationary post-score update without per-event dicts."""
    src_idx = int(row["src_node_idx"])
    dst_idx = int(row["dst_node_idx"])
    z_vec = np.asarray(z, dtype=np.float32)
    q_t = model._compute_update_q(residual_score, enable_update_gate=True)
    lambda_t, rho_t = lambda_rho_for_event(action, src_type_name)
    model.last_lambda_t = float(lambda_t)
    model.last_rho_t = float(rho_t)
    model.last_q_t = float(q_t)
    current = {src_idx, dst_idx}
    if src_idx == dst_idx:
        h_old = model.state_model.read(model.memory.read_state(src_idx, src_type_name))
        message = normalize_vector(z_vec + np.float32(lambda_t) * h_old)
        model.memory.update_state(
            src_idx,
            src_type_name,
            message,
            model.state_model,
            q_t=q_t,
            protected=False,
            current_node_ids=current,
            role="src",
        )
    else:
        h_src = model.state_model.read(model.memory.read_state(src_idx, src_type_name))
        dst_message = normalize_vector(z_vec + np.float32(lambda_t) * h_src)
        src_message = (np.float32(rho_t) * z_vec).astype(np.float32, copy=False)
        model.memory.update_state(
            src_idx,
            src_type_name,
            src_message,
            model.state_model,
            q_t=q_t,
            protected=False,
            current_node_ids=current,
            role="src",
        )
        model.memory.update_state(
            dst_idx,
            dst_type_name,
            dst_message,
            model.state_model,
            q_t=q_t,
            protected=False,
            current_node_ids=current,
            role="dst",
        )
    model.global_state = (model.a * model.global_state + model.g * z_vec).astype(
        np.float32,
        copy=False,
    )


def _phase3g_update_gate_enabled(config: SlimConfig, model: SSPMLowRankModel) -> bool:
    """Return whether the conditional fast stream must score before every update."""
    return (
        str(config.sspm_update_gate_mode) != "none"
        or str(model.config.update_gate_mode) != "none"
    )


def _phase3g_apply_conditional_update_gate_score_space(
    config: SlimConfig,
    model: SSPMLowRankModel,
    validation_scores: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Fit the E5 update gate from conditional validation event scores when requested."""
    score_space = str(config.sspm_update_gate_score_space)
    if score_space not in SSPM_UPDATE_GATE_SCORE_SPACES:
        raise ValueError(
            "SSPM_UPDATE_GATE_SCORE_SPACE must be checkpoint or conditional_event_score",
        )
    existing = _model_update_gate_summary(config, model)
    if str(config.sspm_update_gate_mode) == "none":
        return {
            **existing,
            "score_space": score_space,
            "applied_runtime_refit": False,
        }
    if str(config.sspm_update_gate_mode) != "quantile":
        raise ValueError(f"unknown SSPM update gate mode: {config.sspm_update_gate_mode}")
    if score_space != "conditional_event_score":
        return {
            **existing,
            "score_space": score_space,
            "applied_runtime_refit": False,
        }

    gate = UpdateGateCalibrator(
        quantile=float(config.sspm_update_gate_quantile),
        q_min=float(config.sspm_update_gate_q_min),
        eta=float(config.sspm_update_gate_eta),
        threshold_mode=str(config.sspm_update_gate_threshold_mode),
    )
    gate.fit(validation_scores)
    model.set_update_gate_calibrator(gate)
    return {
        **gate.summary(),
        "score_space": score_space,
        "applied_runtime_refit": True,
    }


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
) -> dict[str, Any]:
    """Return alert eligibility for action/type policy without changing event score."""
    policy = str(config.action_type_alert_policy)
    if policy not in ACTION_TYPE_ALERT_POLICIES:
        raise ValueError(
            "ACTION_TYPE_ALERT_POLICY must be default, theia_v1, "
            "cadets_e4_v2_group_v1, clearscope_node_pair_v1, "
            "or clearscope_android_v2",
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


def _score_phase3f_e2_none_fast_stream(
    config: SlimConfig,
    sspm: SSPMLowRankModel,
    test_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    residual_scores: Sequence[float],
    threshold: float,
    output_dir: Path | None = None,
    initial_test_peak_rss_mb: float | None = None,
) -> dict[str, Any]:
    """Score E2_NONE Phase3E events through the Phase3F online-minimal fast path."""
    scoring_started = time.perf_counter()
    records = _phase3e_event_index_array(test_index)
    residual_sorted = _compact_sorted_scores(residual_scores)
    threshold_value = float(threshold)
    chunk_size = max(int(config.sspm_infer_chunk_events), 1)
    sspm.reset_state()
    test_count = 0
    event_alert_count = 0
    checkpoint_count = 0
    node_pool: dict[int, dict[str, Any]] = {}
    score_summary = _StreamingScoreSummary(threshold_value)
    raw_paths: dict[str, Path] = {}
    start_rss_mb = _current_rss_mb()
    if initial_test_peak_rss_mb is None:
        test_phase_peak_rss_mb = start_rss_mb
    else:
        test_phase_peak_rss_mb = max(float(initial_test_peak_rss_mb), start_rss_mb)
    _stage_log(config, "phase3f_fast_test_scoring_start", count=int(records.shape[0]))
    type_count = len(ENTITY_TYPES)
    type_one_hot = np.eye(type_count, dtype=np.float32)
    action_one_hot = np.stack(
        [raw_action_one_hot(action) for action in ORTHRUS10_ACTION_NAMES],
        axis=0,
    ).astype(np.float32, copy=False)
    with ExitStack() as stack:
        event_writer = None
        profile_writer = None
        sspm_profiler = None
        if output_dir is not None:
            raw_paths = _raw_output_paths(Path(output_dir))
            raw_paths["events"] = Path(output_dir) / "online_event_alerts.csv"
            event_writer = stack.enter_context(
                StreamingCsvWriter(raw_paths["events"], EVENT_RAW_FIELDS),
            )
            profile_writer = stack.enter_context(
                StreamingCsvWriter(raw_paths["profiling"], PROFILING_FIELDS),
            )
            sspm_profiler = SSPMProfiler(raw_paths["profiling_sspm_raw"])
        for start in range(0, int(records.shape[0]), chunk_size):
            end = min(start + chunk_size, int(records.shape[0]))
            chunk = records[start:end]
            for offset, row in enumerate(chunk):
                stream_pos = start + offset
                test_count = int(stream_pos) + 1
                z = _phase3e_target_from_index_row(row, node_embeddings, action_embeddings)
                context, _src_state, _dst_state, action, src_type_name, dst_type_name = (
                    _phase3f_e2_none_context_from_numeric_row(
                        sspm,
                        row,
                        type_one_hot,
                        action_one_hot,
                    )
                )
                pred = sspm.predict_from_context(context)
                residual_score = float(
                    sspm.residual_score(pred, z, action=action),
                )
                residual_tail = _empirical_tail_score_compact(residual_score, residual_sorted)
                event_score = _event_score_from_residual_tail(
                    residual_tail,
                    config.event_score_mode,
                )
                score_summary.observe(float(event_score))
                if event_score >= threshold_value:
                    event_alert_count += 1
                    alert_row = _phase3f_raw_alert_row(
                        row,
                        stream_pos,
                        event_score,
                        threshold_value,
                        residual_tail,
                        residual_score,
                        f"validation_{config.event_threshold_mode}",
                    )
                    if event_writer is not None:
                        event_writer.write_row(alert_row)
                    src_idx = int(row["src_node_idx"])
                    dst_idx = int(row["dst_node_idx"])
                    _minimal_node_pool_update(
                        node_pool,
                        src_idx,
                        int(row["event_id"]),
                        stream_pos,
                        event_score,
                        residual_score,
                    )
                    if dst_idx != src_idx:
                        _minimal_node_pool_update(
                            node_pool,
                            dst_idx,
                            int(row["event_id"]),
                            stream_pos,
                            event_score,
                            residual_score,
                        )
                    alert_node_ids = (src_idx, dst_idx)
                else:
                    alert_node_ids = ()
                _phase3f_e2_none_update_from_numeric_row(
                    sspm,
                    row,
                    z,
                    action,
                    src_type_name,
                    dst_type_name,
                    residual_score,
                )
                for node_id in alert_node_ids:
                    sspm.mark_risk_node(int(node_id))
            if (
                int(config.progress_interval_events) > 0
                and test_count > 0
                and test_count % int(config.progress_interval_events) == 0
            ):
                current_rss_mb = _current_rss_mb()
                test_phase_peak_rss_mb = max(test_phase_peak_rss_mb, current_rss_mb)
                _stage_log(
                    config,
                    "phase3f_fast_test_scoring_progress",
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
        _stage_log(
            config,
            "phase3f_fast_test_scoring_end",
            count=test_count,
            event_alerts=event_alert_count,
            node_pool=len(node_pool),
        )
    test_scoring_seconds = float(time.perf_counter() - scoring_started)
    smaps = _current_smaps_rollup_mb()
    final_rss_mb = _current_rss_mb()
    final_nodes_raw = _final_node_pool_minimal(node_pool)
    online_minimal_summary = {
        "online_minimal_rss_start_mb": float(start_rss_mb),
        "online_minimal_rss_peak_mb": float(test_phase_peak_rss_mb),
        "online_minimal_rss_final_mb": float(final_rss_mb),
        "online_minimal_rss_delta_mb": float(test_phase_peak_rss_mb - start_rss_mb),
        "online_minimal_events_per_sec": float(
            test_count / max(test_scoring_seconds, 1e-9),
        ),
        "online_minimal_state_array_mb": float(_safe_state_array_mb(sspm)),
        "online_minimal_alert_buffer_mb": 0.0,
        "online_minimal_anonymous_rss_mb": smaps.get("anonymous_rss_mb"),
        "online_minimal_file_backed_rss_mb": smaps.get("file_backed_rss_mb"),
    }
    return {
        "test_count": int(test_count),
        "final_nodes_raw": final_nodes_raw,
        "threshold": threshold_value,
        "test_score_summary": score_summary.summary(event_alert_count=event_alert_count),
        "raw_outputs_streamed": output_dir is not None,
        "output_dir": str(output_dir) if output_dir is not None else "",
        "raw_output_paths": {key: str(path) for key, path in raw_paths.items()},
        "event_score_trace_csv": str(raw_paths.get("event_score_trace", "")),
        "event_alert_count": int(event_alert_count),
        "checkpoint_count": int(checkpoint_count),
        "residual_semantic_examples": {group: [] for group in RESIDUAL_EXAMPLE_GROUPS},
        "test_scoring_seconds": test_scoring_seconds,
        "stream_csv_write_seconds": 0.0,
        "threshold_controller": {
            "mode": str(config.event_threshold_mode),
            "initial_threshold": threshold_value,
            "final_threshold": threshold_value,
        },
        "compat_in_memory_outputs_active": False,
        "raw_alerts_written": bool(output_dir is not None),
        "analysis_outputs_written": False,
        "state_merge_diagnostics_written": False,
        "rss_test_peak_mb": float(test_phase_peak_rss_mb),
        "process_peak_rss_mb": _safe_peak_rss_mb(),
        "state_merge_diagnostics_truncated": False,
        "phase3f_fast_path_enabled": True,
        "online_minimal": True,
        **online_minimal_summary,
    }


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


def _phase3g_action_fingerprint(*_args: object, **_kwargs: object) -> dict[str, Any]:
    """Fail fast for the historical action-predict validation fingerprint path."""
    raise ValueError(
        "historical Phase3G action-predict head moved to "
        "legacy.experiments.phase3g_action_head",
    )


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
    run_only = _phase3g_run_only_from_out_tag(config.out_tag)
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
        "RUN_ONLY": run_only,
        "run_only": run_only,
        "run": str(config.out_tag),
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


def _phase3g_action_embedding_fingerprint(
    *,
    config: SlimConfig,
    head: ConditionalSemanticHead,
    paths: Mapping[str, Path],
    event_meta: Mapping[str, Any],
    event_index: np.ndarray,
    split: str,
) -> dict[str, Any]:
    split_meta = dict(dict(event_meta.get("splits", {})).get(split, {}))
    return {
        "schema": "phase3g_conditional_action_embedding_validation_cache_v1",
        "score_head": CONDITIONAL_ACTION_EMBEDDING_SCORE_HEAD,
        "conditional_action_embedding_head_fingerprint": head.fingerprint().get(
            "fingerprint_sha256",
        ),
        "context_dim": int(head.config.input_dim),
        "target": "e_action",
        "node_repr_fusion": str(config.node_repr_fusion),
        "loss_type": str(config.conditional_semantic_loss),
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
        "rank": int(head.config.rank),
        "output_dim": int(head.config.output_dim),
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


def _phase3g_action_scores_stream(*_args: object, **_kwargs: object) -> tuple[np.ndarray, int]:
    """Fail fast for the historical action-predict validation stream."""
    raise ValueError(
        "historical Phase3G action-predict head moved to "
        "legacy.experiments.phase3g_action_head",
    )


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


def _phase3g_action_embedding_scores_stream(
    *,
    config: SlimConfig,
    head: ConditionalSemanticHead,
    event_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
) -> tuple[np.ndarray, int]:
    records = _phase3e_event_index_array(event_index)
    model = SSPMLowRankModel(_make_sspm_config(config, _load_process_config(config)))
    type_eye = _phase3g_type_eye()
    scores = np.zeros((int(records.shape[0]),), dtype=np.float32)
    chunk_size = max(int(config.sspm_infer_chunk_events), 1)
    model.reset_state()
    for start in range(0, int(records.shape[0]), chunk_size):
        end = min(start + chunk_size, int(records.shape[0]))
        chunk = records[start:end]
        contexts = np.zeros((int(chunk.shape[0]), int(head.config.input_dim)), dtype=np.float32)
        targets = np.zeros((int(chunk.shape[0]), int(head.config.output_dim)), dtype=np.float32)
        for offset, row in enumerate(chunk):
            z_state = _phase3e_target_from_index_row(row, node_embeddings, action_embeddings)
            context, src_type_name, dst_type_name, _, _ = _phase3g_action_context_from_model(
                model,
                row,
                node_embeddings,
                type_eye,
            )
            contexts[offset] = context
            targets[offset] = select_action_embedding_target(row, action_embeddings)
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
        predictions = head.predict(contexts)
        scores[start:end] = conditional_distance(
            predictions,
            targets,
            str(config.conditional_semantic_loss),
        )
    return scores, int(records.shape[0])


def _phase3g_conditional_memmap_dir(config: SlimConfig) -> Path:
    """Return the Phase3G conditional train memmap cache directory."""
    root = Path(str(config.action_validation_cache_dir).format(DATASET=config.dataset))
    return root / "conditional_train_memmaps"


def _phase3g_conditional_memmap_key(config: SlimConfig) -> str:
    """Return the cache key for the conditional train state trajectory."""
    if str(config.sspm_state_model) == "s4d_complex_node":
        return "s4d_complex_node"
    if str(config.sspm_update_gate_mode) == "quantile":
        return "ema_fixed_update_gate_quantile"
    return "ema_fixed"


def _phase3g_conditional_memmap_paths(config: SlimConfig) -> dict[str, Path]:
    """Return file paths for the selected conditional train memmap."""
    key = _phase3g_conditional_memmap_key(config)
    root = _phase3g_conditional_memmap_dir(config)
    return {
        "dir": root,
        "x": root / f"X_conditional_{key}.memmap",
        "y": root / f"Y_conditional_{key}.memmap",
        "target_case": root / f"target_case_{key}.memmap",
        "meta": root / f"conditional_memmap_{key}_meta.json",
    }


def _phase3g_conditional_memmap_fingerprint(
    *,
    config: SlimConfig,
    paths: Mapping[str, Path],
    event_meta: Mapping[str, Any],
    input_dim: int,
) -> dict[str, Any]:
    """Return the expected fingerprint for a conditional train memmap."""
    train_split = dict(dict(event_meta.get("splits", {})).get("train", {}))
    payload = {
        "schema": "phase3g_conditional_train_memmap_v1",
        "dataset": str(config.dataset),
        "score_head": str(config.sspm_score_head),
        "target_logic": "state_exists_event_semantic_else_both_cold_action",
        "node_repr_fusion": str(config.node_repr_fusion),
        "conditional_semantic_loss": str(config.conditional_semantic_loss),
        "state_model": str(config.sspm_state_model),
        "state_dim": int(config.sspm_state_dim),
        "target_dim": int(config.sspm_target_dim),
        "input_dim": int(input_dim),
        "rank": int(config.rank),
        "event_index_fingerprint": train_split.get("fingerprint", {}),
        "node_embedding_fingerprint": _phase3g_file_fingerprint(paths["node_embeddings"]),
        "action_embedding_fingerprint": _phase3g_file_fingerprint(paths["action_embeddings"]),
        "event_index_path": str(paths["event_index_train"]),
        "node_embedding_path": str(paths["node_embeddings"]),
        "action_embedding_path": str(paths["action_embeddings"]),
    }
    if str(config.sspm_update_gate_mode) == "quantile":
        payload.update(
            {
                "update_gate_mode": str(config.sspm_update_gate_mode),
                "update_gate_quantile": float(config.sspm_update_gate_quantile),
                "update_gate_threshold_mode": str(config.sspm_update_gate_threshold_mode),
                "update_gate_score_space": str(config.sspm_update_gate_score_space),
                "update_gate_q_min": float(config.sspm_update_gate_q_min),
                "update_gate_eta": float(config.sspm_update_gate_eta),
                "residual_score_mode": str(config.sspm_residual_score_mode),
                "residual_calibration": str(config.sspm_residual_calibration),
            },
        )
    payload["fingerprint_sha256"] = stable_json_hash(payload)
    return payload


def _phase3g_conditional_memmap_sentinel(config: SlimConfig) -> dict[str, Any]:
    """Return a fingerprint sentinel for valid non-memmap conditional training."""
    return {
        "schema": "phase3g_conditional_train_memmap_v1",
        "status": "not_applicable",
        "reason": "real_diag_stream_event_gamma",
        "state_model": str(config.sspm_state_model),
        "train_data_mode": "stream_event",
        "fingerprint_sha256": "not_applicable:real_diag_stream_event_gamma",
    }


def _phase3g_load_conditional_memmap_meta(path: Path) -> dict[str, Any] | None:
    """Load conditional memmap metadata if present."""
    if not Path(path).exists():
        return None
    return _phase3e_load_json(path)


def _phase3g_validate_conditional_memmap(
    *,
    meta: Mapping[str, Any],
    fingerprint: Mapping[str, Any],
    count: int,
    input_dim: int,
    output_dim: int,
) -> None:
    """Fail fast when a conditional memmap sidecar does not match the request."""
    if str(meta.get("schema", "")) != "phase3g_conditional_train_memmap_v1":
        raise ValueError("conditional memmap schema mismatch")
    actual_fingerprint = dict(meta.get("fingerprint", {}))
    if actual_fingerprint != dict(fingerprint):
        raise ValueError("conditional memmap fingerprint mismatch")
    if int(meta.get("num_events", -1)) != int(count):
        raise ValueError("conditional memmap event count mismatch")
    if int(meta.get("input_dim", -1)) != int(input_dim):
        raise ValueError("conditional memmap input_dim mismatch")
    if int(meta.get("output_dim", -1)) != int(output_dim):
        raise ValueError("conditional memmap output_dim mismatch")


def _phase3g_build_conditional_train_memmap(
    *,
    config: SlimConfig,
    train_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    paths: Mapping[str, Path],
    event_meta: Mapping[str, Any],
    input_dim: int,
    output_dim: int,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """Build or validate train-only conditional X/Y/target_case memmaps."""
    records = _phase3e_event_index_array(train_index)
    count = int(records.shape[0])
    memmap_paths = _phase3g_conditional_memmap_paths(config)
    memmap_paths["dir"].mkdir(parents=True, exist_ok=True)
    fingerprint = _phase3g_conditional_memmap_fingerprint(
        config=config,
        paths=paths,
        event_meta=event_meta,
        input_dim=input_dim,
    )
    existing = _phase3g_load_conditional_memmap_meta(memmap_paths["meta"])
    if existing and not bool(force_rebuild):
        _phase3g_validate_conditional_memmap(
            meta=existing,
            fingerprint=fingerprint,
            count=count,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        for key in ("x", "y", "target_case"):
            if not memmap_paths[key].exists():
                raise FileNotFoundError(f"conditional memmap sidecar missing: {memmap_paths[key]}")
        return dict(existing)

    started = time.perf_counter()
    rss_peak = _current_rss_mb()
    if str(config.sspm_checkpoint_path).strip():
        model, _ = load_sspm_checkpoint_state_for_conditional(
            config.sspm_checkpoint_path,
            config,
        )
    else:
        model = SSPMLowRankModel(_make_sspm_config(config, _load_process_config(config)))
    model.reset_state()
    type_eye = _phase3g_type_eye()
    x_map = np.memmap(
        memmap_paths["x"],
        dtype=np.float32,
        mode="w+",
        shape=(count, int(input_dim)),
    )
    y_map = np.memmap(
        memmap_paths["y"],
        dtype=np.float32,
        mode="w+",
        shape=(count, int(output_dim)),
    )
    case_map = np.memmap(
        memmap_paths["target_case"],
        dtype=np.int8,
        mode="w+",
        shape=(count,),
    )
    target_case_counts = {EVENT_SEMANTIC_TARGET: 0, BOTH_COLD_ACTION_TARGET: 0}
    chunk_size = max(int(config.sspm_torch_batch_events), 1)
    _stage_log(
        config,
        "phase3g_conditional_memmap_build_start",
        mode="conditional_memmap",
        state_model=str(config.sspm_state_model),
        count=count,
        x_path=str(memmap_paths["x"]),
        y_path=str(memmap_paths["y"]),
    )
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        for offset, row in enumerate(records[start:end], start=start):
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
            x_map[offset] = context
            y_map[offset] = target
            case_map[offset] = np.int8(target_case_id(target_case))
            target_case_counts[target_case] = target_case_counts.get(target_case, 0) + 1
            action = str(ORTHRUS10_ACTION_NAMES[int(row["action_id"])])
            update_residual_score = 0.0
            if _phase3g_update_gate_enabled(config, model):
                fields = _phase3e_fields_from_index_row(row)
                pred_state = model.predict(fields)
                update_residual_score = float(
                    model.residual_score(
                        pred_state,
                        z_state,
                        action=fields.get("raw_action", fields.get("action")),
                    ),
                )
            _phase3f_e2_none_update_from_numeric_row(
                model,
                row,
                z_state,
                action,
                src_type_name,
                dst_type_name,
                residual_score=update_residual_score,
            )
        rss_peak = max(rss_peak, _current_rss_mb())
        if int(config.progress_interval_events) > 0 and end % int(config.progress_interval_events) == 0:
            _stage_log(
                config,
                "phase3g_conditional_memmap_build_progress",
                count=end,
                rss_mb=f"{rss_peak:.3f}",
            )
    x_map.flush()
    y_map.flush()
    case_map.flush()
    elapsed = float(time.perf_counter() - started)
    meta = {
        "schema": "phase3g_conditional_train_memmap_v1",
        "train_data_mode": "conditional_memmap",
        "state_model": str(config.sspm_state_model),
        "num_events": int(count),
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "target_case_counts": {key: int(value) for key, value in target_case_counts.items()},
        "x_path": str(memmap_paths["x"]),
        "y_path": str(memmap_paths["y"]),
        "target_case_path": str(memmap_paths["target_case"]),
        "x_size_mb": float(memmap_paths["x"].stat().st_size / 1024.0 / 1024.0),
        "y_size_mb": float(memmap_paths["y"].stat().st_size / 1024.0 / 1024.0),
        "target_case_size_mb": float(memmap_paths["target_case"].stat().st_size / 1024.0 / 1024.0),
        "build_time_sec": elapsed,
        "build_events_per_sec": float(count / max(elapsed, 1e-9)),
        "build_rss_peak_mb": float(rss_peak),
        "fingerprint": fingerprint,
    }
    memmap_paths["meta"].write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _stage_log(
        config,
        "phase3g_conditional_memmap_build_end",
        count=count,
        events_per_sec=f"{float(meta['build_events_per_sec']):.3f}",
        rss_peak_mb=f"{rss_peak:.3f}",
    )
    return meta


def _phase3g_train_conditional_head_torch_from_memmap(
    *,
    config: SlimConfig,
    head: ConditionalSemanticHead,
    memmap_meta: Mapping[str, Any],
    max_epochs: int,
) -> dict[str, Any]:
    """Train the conditional head by batch-reading prebuilt X/Y memmaps."""
    import torch

    device, cuda_available, device_name = _resolve_phase3g_torch_device(
        torch,
        str(config.sspm_torch_device),
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if head.is_dual_head:
        event_w1 = torch.nn.Parameter(
            torch.tensor(head.event_w1, dtype=torch.float32, device=device),
        )
        event_w2 = torch.nn.Parameter(
            torch.tensor(head.event_w2, dtype=torch.float32, device=device),
        )
        event_bias = torch.nn.Parameter(
            torch.tensor(head.event_bias, dtype=torch.float32, device=device),
        )
        action_w1 = torch.nn.Parameter(
            torch.tensor(head.action_w1, dtype=torch.float32, device=device),
        )
        action_w2 = torch.nn.Parameter(
            torch.tensor(head.action_w2, dtype=torch.float32, device=device),
        )
        action_bias = torch.nn.Parameter(
            torch.tensor(head.action_bias, dtype=torch.float32, device=device),
        )
        train_parameters = [
            event_w1,
            event_w2,
            event_bias,
            action_w1,
            action_w2,
            action_bias,
        ]
    else:
        w1 = torch.nn.Parameter(torch.tensor(head.w1, dtype=torch.float32, device=device))
        w2 = torch.nn.Parameter(torch.tensor(head.w2, dtype=torch.float32, device=device))
        bias = torch.nn.Parameter(torch.tensor(head.bias, dtype=torch.float32, device=device))
        train_parameters = [w1, w2, bias]
    lr = _phase3e_train_lr(config)
    optimizer = torch.optim.Adam(
        train_parameters,
        lr=float(lr),
        weight_decay=float(config.sspm_torch_weight_decay),
    )
    count = int(memmap_meta["num_events"])
    input_dim = int(memmap_meta["input_dim"])
    output_dim = int(memmap_meta["output_dim"])
    x_map = np.memmap(
        str(memmap_meta["x_path"]),
        dtype=np.float32,
        mode="r",
        shape=(count, input_dim),
    )
    y_map = np.memmap(
        str(memmap_meta["y_path"]),
        dtype=np.float32,
        mode="r",
        shape=(count, output_dim),
    )
    case_map = np.memmap(
        str(memmap_meta["target_case_path"]),
        dtype=np.int8,
        mode="r",
        shape=(count,),
    )
    chunk_size = max(int(config.sspm_torch_batch_events), 1)
    best_loss = float("inf")
    loss_by_epoch: list[float] = []
    loss_event_semantic_by_epoch: list[float] = []
    loss_both_cold_action_by_epoch: list[float] = []
    plateau_epochs = 0
    early_stop_reason = ""
    started = time.perf_counter()
    _stage_log(
        config,
        "phase3g_conditional_memmap_train_start",
        train_data_mode="conditional_memmap",
        count=count,
        epochs=int(max_epochs),
        batch_events=chunk_size,
        conditional_head_arch=str(config.sspm_conditional_head_arch),
        conditional_memmap_path=str(memmap_meta["x_path"]),
        conditional_memmap_fingerprint=str(
            dict(memmap_meta.get("fingerprint", {})).get("fingerprint_sha256", ""),
        ),
    )

    def _row_loss(pred: Any, target: Any) -> Any:
        if str(config.conditional_semantic_loss) == "mse":
            return torch.mean((pred - target) ** 2, dim=1)
        pred_norm = torch.nn.functional.normalize(pred, dim=1, eps=1e-8)
        y_norm = torch.nn.functional.normalize(target, dim=1, eps=1e-8)
        return 1.0 - torch.sum(pred_norm * y_norm, dim=1)

    for _epoch in range(max(int(max_epochs), 1)):
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_event_loss_sum = 0.0
        epoch_action_loss_sum = 0.0
        epoch_event_count = 0
        epoch_action_count = 0
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            x = torch.as_tensor(
                np.asarray(x_map[start:end]).copy(),
                dtype=torch.float32,
                device=device,
            )
            y = torch.as_tensor(
                np.asarray(y_map[start:end]).copy(),
                dtype=torch.float32,
                device=device,
            )
            if head.is_dual_head:
                case_ids = torch.as_tensor(
                    np.asarray(case_map[start:end]).copy(),
                    dtype=torch.int64,
                    device=device,
                )
                batch_losses: list[Any] = []
                event_mask = case_ids == int(EVENT_SEMANTIC_TARGET_ID)
                action_mask = case_ids == int(BOTH_COLD_ACTION_TARGET_ID)
                if bool(torch.any(event_mask).detach().cpu().item()):
                    event_pred = x[event_mask] @ event_w1 @ event_w2 + event_bias
                    event_losses = _row_loss(event_pred, y[event_mask])
                    batch_losses.append(event_losses)
                    event_count = int(event_losses.shape[0])
                    epoch_event_loss_sum += float(
                        torch.sum(event_losses).detach().cpu().item(),
                    )
                    epoch_event_count += event_count
                if bool(torch.any(action_mask).detach().cpu().item()):
                    action_pred = x[action_mask] @ action_w1 @ action_w2 + action_bias
                    action_losses = _row_loss(action_pred, y[action_mask])
                    batch_losses.append(action_losses)
                    action_count = int(action_losses.shape[0])
                    epoch_action_loss_sum += float(
                        torch.sum(action_losses).detach().cpu().item(),
                    )
                    epoch_action_count += action_count
                if not batch_losses:
                    raise ValueError("conditional memmap batch has no supported target cases")
                loss = torch.mean(torch.cat(batch_losses, dim=0))
            else:
                pred = x @ w1 @ w2 + bias
                row_losses = _row_loss(pred, y)
                loss = torch.mean(row_losses)
                cases_np = np.asarray(case_map[start:end])
                event_mask_np = cases_np == int(EVENT_SEMANTIC_TARGET_ID)
                action_mask_np = cases_np == int(BOTH_COLD_ACTION_TARGET_ID)
                row_losses_np = row_losses.detach().cpu().numpy()
                if np.any(event_mask_np):
                    epoch_event_loss_sum += float(np.sum(row_losses_np[event_mask_np]))
                    epoch_event_count += int(np.count_nonzero(event_mask_np))
                if np.any(action_mask_np):
                    epoch_action_loss_sum += float(np.sum(row_losses_np[action_mask_np]))
                    epoch_action_count += int(np.count_nonzero(action_mask_np))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_parameters, 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            epoch_batches += 1
        mean_loss = float(epoch_loss / max(epoch_batches, 1))
        loss_by_epoch.append(mean_loss)
        event_loss = float(epoch_event_loss_sum / max(epoch_event_count, 1))
        action_loss = float(epoch_action_loss_sum / max(epoch_action_count, 1))
        loss_event_semantic_by_epoch.append(event_loss)
        loss_both_cold_action_by_epoch.append(action_loss)
        if best_loss - mean_loss > float(config.sspm_early_stop_min_delta):
            best_loss = mean_loss
            plateau_epochs = 0
        else:
            plateau_epochs += 1
            if int(config.sspm_early_stop_patience) > 0 and plateau_epochs >= int(
                config.sspm_early_stop_patience,
            ):
                early_stop_reason = "loss_plateau"
                break
        _stage_log(
            config,
            "phase3g_conditional_memmap_train_epoch",
            epoch=len(loss_by_epoch),
            loss=f"{mean_loss:.6f}",
            loss_event_semantic=f"{event_loss:.6f}",
            loss_both_cold_action=f"{action_loss:.6f}",
            event_semantic=int(
                dict(memmap_meta.get("target_case_counts", {})).get(EVENT_SEMANTIC_TARGET, 0),
            ),
            both_cold=int(
                dict(memmap_meta.get("target_case_counts", {})).get(BOTH_COLD_ACTION_TARGET, 0),
            ),
        )
    if device == "cuda":
        torch.cuda.synchronize()
        peak_gpu_mb = float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
    else:
        peak_gpu_mb = 0.0
    if head.is_dual_head:
        head.event_w1 = event_w1.detach().cpu().numpy().astype(np.float32, copy=True)
        head.event_w2 = event_w2.detach().cpu().numpy().astype(np.float32, copy=True)
        head.event_bias = event_bias.detach().cpu().numpy().astype(np.float32, copy=True)
        head.action_w1 = action_w1.detach().cpu().numpy().astype(np.float32, copy=True)
        head.action_w2 = action_w2.detach().cpu().numpy().astype(np.float32, copy=True)
        head.action_bias = action_bias.detach().cpu().numpy().astype(np.float32, copy=True)
        head.w1 = head.event_w1
        head.w2 = head.event_w2
        head.bias = head.event_bias
    else:
        head.w1 = w1.detach().cpu().numpy().astype(np.float32, copy=True)
        head.w2 = w2.detach().cpu().numpy().astype(np.float32, copy=True)
        head.bias = bias.detach().cpu().numpy().astype(np.float32, copy=True)
    del x_map
    del y_map
    del case_map
    final_case_summary = conditional_case_loss_summary(
        np.asarray(
            [
                loss_event_semantic_by_epoch[-1] if loss_event_semantic_by_epoch else 0.0,
                loss_both_cold_action_by_epoch[-1] if loss_both_cold_action_by_epoch else 0.0,
            ],
            dtype=np.float32,
        ),
        np.asarray(
            [EVENT_SEMANTIC_TARGET_ID, BOTH_COLD_ACTION_TARGET_ID],
            dtype=np.int8,
        ),
    )
    return {
        "score_head": str(config.sspm_score_head),
        "conditional_head_arch": str(config.sspm_conditional_head_arch),
        "head_arch": str(config.sspm_conditional_head_arch),
        "node_repr_fusion": str(config.node_repr_fusion),
        "loss_type": str(config.conditional_semantic_loss),
        "target_case_mode": "conditional_cold_action_else_event",
        "target_case_head_mode": str(head.config.target_case_head_mode),
        "target": "conditional_target",
        "target_case_counts": dict(memmap_meta.get("target_case_counts", {})),
        "count_event_semantic": int(
            dict(memmap_meta.get("target_case_counts", {})).get(EVENT_SEMANTIC_TARGET, 0),
        ),
        "count_both_cold_action": int(
            dict(memmap_meta.get("target_case_counts", {})).get(BOTH_COLD_ACTION_TARGET, 0),
        ),
        "input_dim": int(head.config.input_dim),
        "context_dim": int(head.config.input_dim),
        "rank": int(head.config.rank),
        "output_dim": int(head.config.output_dim),
        "train_data_mode": "conditional_memmap",
        "conditional_train_data_mode": "memmap",
        "conditional_memmap_path": str(memmap_meta["x_path"]),
        "conditional_memmap_y_path": str(memmap_meta["y_path"]),
        "conditional_memmap_target_case_path": str(memmap_meta["target_case_path"]),
        "conditional_memmap_fingerprint": dict(memmap_meta.get("fingerprint", {})),
        "conditional_memmap_size_mb": float(
            float(memmap_meta.get("x_size_mb", 0.0))
            + float(memmap_meta.get("y_size_mb", 0.0))
            + float(memmap_meta.get("target_case_size_mb", 0.0)),
        ),
        "torch_device": device,
        "torch_cuda_available": bool(cuda_available),
        "torch_device_name": str(device_name),
        "torch_optimizer": "Adam",
        "torch_lr": float(lr),
        "torch_weight_decay": float(config.sspm_torch_weight_decay),
        "torch_batch_events": int(config.sspm_torch_batch_events),
        "torch_epochs_completed": int(len(loss_by_epoch)),
        "torch_train_loss_best": float(best_loss if loss_by_epoch else 0.0),
        "torch_train_loss_final": float(loss_by_epoch[-1] if loss_by_epoch else 0.0),
        "epochs_completed": int(len(loss_by_epoch)),
        "best_loss": float(best_loss if loss_by_epoch else 0.0),
        "final_loss": float(loss_by_epoch[-1] if loss_by_epoch else 0.0),
        "loss_event_semantic": float(
            loss_event_semantic_by_epoch[-1] if loss_event_semantic_by_epoch else 0.0,
        ),
        "loss_both_cold_action": float(
            loss_both_cold_action_by_epoch[-1] if loss_both_cold_action_by_epoch else 0.0,
        ),
        "loss_event_semantic_by_epoch": loss_event_semantic_by_epoch,
        "loss_both_cold_action_by_epoch": loss_both_cold_action_by_epoch,
        "case_loss_summary": final_case_summary,
        "torch_early_stop_reason": str(early_stop_reason),
        "torch_train_time_sec": float(time.perf_counter() - started),
        "torch_peak_gpu_memory_mb": float(peak_gpu_mb),
        "loss_by_epoch": loss_by_epoch,
        "train_events_actual": int(count),
        "train_events_seen_total": int(count) * int(len(loss_by_epoch)),
    }


def _phase3g_train_action_head_torch(*_args: object, **_kwargs: object) -> dict[str, Any]:
    """Fail fast for historical action-predict head training."""
    raise ValueError(
        "historical Phase3G action-predict head training moved to legacy; "
        "current best chains use conditional_action_semantic",
    )


def _phase3g_train_conditional_head_torch(
    *,
    config: SlimConfig,
    head: ConditionalSemanticHead,
    train_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
) -> dict[str, Any]:
    import torch

    device, cuda_available, device_name = _resolve_phase3g_torch_device(
        torch,
        str(config.sspm_torch_device),
    )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    w1 = torch.nn.Parameter(torch.tensor(head.w1, dtype=torch.float32, device=device))
    w2 = torch.nn.Parameter(torch.tensor(head.w2, dtype=torch.float32, device=device))
    bias = torch.nn.Parameter(torch.tensor(head.bias, dtype=torch.float32, device=device))
    lr = _phase3e_train_lr(config)
    optimizer = torch.optim.Adam(
        [w1, w2, bias],
        lr=float(lr),
        weight_decay=float(config.sspm_torch_weight_decay),
    )
    records = _phase3e_event_index_array(train_index)
    row_count = int(records.shape[0])
    chunk_size = max(int(config.sspm_torch_batch_events), 1)
    best_loss = float("inf")
    loss_by_epoch: list[float] = []
    is_action_embedding_head = str(config.sspm_score_head) == CONDITIONAL_ACTION_EMBEDDING_SCORE_HEAD
    target_case_totals = (
        {"action_embedding_target": 0}
        if is_action_embedding_head
        else {
            EVENT_SEMANTIC_TARGET: 0,
            BOTH_COLD_ACTION_TARGET: 0,
        }
    )
    plateau_epochs = 0
    early_stop_reason = ""
    started = time.perf_counter()
    for _epoch in range(max(int(config.sspm_epochs), 1)):
        model = SSPMLowRankModel(_make_sspm_config(config, _load_process_config(config)))
        model.reset_state()
        type_eye = _phase3g_type_eye()
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_cases = (
            {"action_embedding_target": 0}
            if is_action_embedding_head
            else {
                EVENT_SEMANTIC_TARGET: 0,
                BOTH_COLD_ACTION_TARGET: 0,
            }
        )
        for start in range(0, row_count, chunk_size):
            end = min(start + chunk_size, row_count)
            chunk = records[start:end]
            contexts = np.zeros((int(chunk.shape[0]), int(head.config.input_dim)), dtype=np.float32)
            targets = np.zeros((int(chunk.shape[0]), int(head.config.output_dim)), dtype=np.float32)
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
                if is_action_embedding_head:
                    target = select_action_embedding_target(row, action_embeddings)
                    target_case = "action_embedding_target"
                else:
                    target, target_case = select_conditional_target(
                        row,
                        node_embeddings,
                        action_embeddings,
                        src_has_state=src_has_state,
                        dst_has_state=dst_has_state,
                    )
                contexts[offset] = context
                targets[offset] = target
                epoch_cases[target_case] += 1
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
            x = torch.as_tensor(contexts, dtype=torch.float32, device=device)
            y = torch.as_tensor(targets, dtype=torch.float32, device=device)
            pred = x @ w1 @ w2 + bias
            if str(config.conditional_semantic_loss) == "mse":
                loss = torch.mean((pred - y) ** 2)
            else:
                pred_norm = torch.nn.functional.normalize(pred, dim=1, eps=1e-8)
                y_norm = torch.nn.functional.normalize(y, dim=1, eps=1e-8)
                loss = torch.mean(1.0 - torch.sum(pred_norm * y_norm, dim=1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([w1, w2, bias], 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu().item())
            epoch_batches += 1
        for key, value in epoch_cases.items():
            target_case_totals[key] = target_case_totals.get(key, 0) + int(value)
        mean_loss = float(epoch_loss / max(epoch_batches, 1))
        loss_by_epoch.append(mean_loss)
        if best_loss - mean_loss > float(config.sspm_early_stop_min_delta):
            best_loss = mean_loss
            plateau_epochs = 0
        else:
            plateau_epochs += 1
            if int(config.sspm_early_stop_patience) > 0 and plateau_epochs >= int(
                config.sspm_early_stop_patience,
            ):
                early_stop_reason = "loss_plateau"
                break
        _stage_log(
            config,
            "phase3g_conditional_train_epoch",
            epoch=len(loss_by_epoch),
            loss=f"{mean_loss:.6f}",
            event_semantic=epoch_cases.get(EVENT_SEMANTIC_TARGET, 0),
            both_cold=epoch_cases.get(BOTH_COLD_ACTION_TARGET, 0),
            action_embedding=epoch_cases.get("action_embedding_target", 0),
        )
    if device == "cuda":
        torch.cuda.synchronize()
        peak_gpu_mb = float(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
    else:
        peak_gpu_mb = 0.0
    head.w1 = w1.detach().cpu().numpy().astype(np.float32, copy=True)
    head.w2 = w2.detach().cpu().numpy().astype(np.float32, copy=True)
    head.bias = bias.detach().cpu().numpy().astype(np.float32, copy=True)
    return {
        "score_head": str(config.sspm_score_head),
        "node_repr_fusion": str(config.node_repr_fusion),
        "loss_type": str(config.conditional_semantic_loss),
        "target_case_mode": (
            "action_embedding_only"
            if is_action_embedding_head
            else "conditional_cold_action_else_event"
        ),
        "target": "e_action" if is_action_embedding_head else "conditional_target",
        "target_case_counts": target_case_totals,
        "input_dim": int(head.config.input_dim),
        "rank": int(head.config.rank),
        "output_dim": int(head.config.output_dim),
        "torch_device": device,
        "torch_cuda_available": bool(cuda_available),
        "torch_device_name": str(device_name),
        "torch_optimizer": "Adam",
        "torch_lr": float(lr),
        "torch_weight_decay": float(config.sspm_torch_weight_decay),
        "torch_batch_events": int(config.sspm_torch_batch_events),
        "torch_epochs_completed": int(len(loss_by_epoch)),
        "torch_train_loss_best": float(best_loss if loss_by_epoch else 0.0),
        "torch_train_loss_final": float(loss_by_epoch[-1] if loss_by_epoch else 0.0),
        "torch_early_stop_reason": str(early_stop_reason),
        "torch_train_time_sec": float(time.perf_counter() - started),
        "torch_peak_gpu_memory_mb": float(peak_gpu_mb),
        "loss_by_epoch": loss_by_epoch,
        "train_events_actual": int(row_count),
    }


def _resolve_phase3g_torch_device(torch_module: Any, requested: str) -> tuple[str, bool, str]:
    cuda_available = bool(torch_module.cuda.is_available())
    requested_device = str(requested).strip().lower()
    if requested_device == "auto":
        if cuda_available:
            return "cuda", True, str(torch_module.cuda.get_device_name(0))
        return "cpu", False, ""
    if requested_device == "cuda":
        if not cuda_available:
            raise ValueError("SSPM_TORCH_DEVICE=cuda requested but CUDA is unavailable")
        return "cuda", True, str(torch_module.cuda.get_device_name(0))
    if requested_device in {"cpu", "torch_cpu"}:
        return "cpu", cuda_available, str(torch_module.cuda.get_device_name(0)) if cuda_available else ""
    raise ValueError(f"unsupported torch device: {requested}")


def run_phase3g_action_train_from_precompute(config: SlimConfig) -> Path:
    """Reject historical action-predict head training from active code."""
    del config
    raise ValueError(
        "historical Phase3G action-predict head training moved to legacy; "
        "current best chains use conditional_action_semantic",
    )


def run_phase3g_conditional_train_from_precompute(config: SlimConfig) -> Path:
    """Train one Phase3G conditional vector head from Phase3E artifacts."""
    _validate_active_event_score_mode(config.event_score_mode)
    if str(config.sspm_train_mode) not in {"train_and_save", "train_conditional_and_save"}:
        raise ValueError(
            f"{config.sspm_score_head} train requires "
            "SSPM_TRAIN_MODE=train_and_save or train_conditional_and_save",
        )
    started = time.perf_counter()
    output_dir = Path(config.result_root) / config.out_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, event_meta = _phase3g_effective_artifacts(config)
    train_index, train_count = _phase3e_open_split_event_index(
        paths,
        event_meta,
        "train",
        max_events=int(config.max_train_events),
    )
    node_embeddings = _phase3g_open_node_embeddings(config, paths)
    action_embeddings = np.load(paths["action_embeddings"], mmap_mode="r")
    validation_count = int(
        dict(dict(event_meta.get("splits", {})).get("validation", {})).get("num_events", 0),
    )
    input_dim = _phase3g_action_input_dim(node_embeddings)
    if is_cadets_dataset(config.dataset) and input_dim != 136:
        raise ValueError(f"CADETS_E3 conditional input_dim must be 136, got {input_dim}")
    head = ConditionalSemanticHead(
        ConditionalSemanticHeadConfig(
            input_dim=input_dim,
            rank=int(config.rank),
            output_dim=int(config.sspm_target_dim),
            loss_type=str(config.conditional_semantic_loss),
            seed=29,
            node_repr_fusion=str(config.node_repr_fusion),
            conditional_head_arch=str(config.sspm_conditional_head_arch),
        ),
    )
    _stage_log(
        config,
        "phase3g_conditional_train_start",
        score_head=str(config.sspm_score_head),
        conditional_train_data_mode=str(config.sspm_conditional_train_data_mode),
        conditional_head_arch=str(config.sspm_conditional_head_arch),
        count=train_count,
        input_dim=input_dim,
        rank=int(config.rank),
        loss=str(config.conditional_semantic_loss),
    )
    conditional_memmap_meta: dict[str, Any] | None = None
    stream_e3_gamma = (
        str(config.sspm_state_model) == "real_diag_learnable"
        and bool(config.real_diag_train_gamma)
    )
    if (
        str(config.sspm_conditional_train_data_mode) == "memmap"
        and not stream_e3_gamma
        and str(config.sspm_score_head) == "conditional_action_semantic"
    ):
        conditional_memmap_meta = _phase3g_build_conditional_train_memmap(
            config=config,
            train_index=train_index,
            node_embeddings=node_embeddings,
            action_embeddings=action_embeddings,
            paths=paths,
            event_meta=event_meta,
            input_dim=input_dim,
            output_dim=int(config.sspm_target_dim),
        )
        if bool(config.phase3g_build_conditional_memmap_only):
            eval_payload = _phase3g_conditional_memmap_only_payload(
                config=config,
                output_dir=output_dir,
                memmap_meta=conditional_memmap_meta,
                train_count=int(train_count),
                validation_count=int(validation_count),
                elapsed_seconds=float(time.perf_counter() - started),
            )
            return _write_phase3g_conditional_memmap_only_outputs(
                config=config,
                output_dir=output_dir,
                eval_payload=eval_payload,
                train_count=int(train_count),
                validation_count=int(validation_count),
            )
        train_stats = _phase3g_train_conditional_head_torch_from_memmap(
            config=config,
            head=head,
            memmap_meta=conditional_memmap_meta,
            max_epochs=int(config.sspm_conditional_max_epochs),
        )
    elif str(config.sspm_conditional_train_data_mode) == "stream_event" or stream_e3_gamma:
        stream_sentinel = _phase3g_conditional_memmap_sentinel(config)
        if stream_e3_gamma:
            original_epochs = int(config.sspm_epochs)
            config.sspm_epochs = min(
                int(config.sspm_epochs),
                int(config.sspm_conditional_e3_max_epochs),
            )
            _stage_log(
                config,
                "phase3g_conditional_train_stream_event_e3_gamma",
                requested_epochs=original_epochs,
                effective_epochs=int(config.sspm_epochs),
            )
        try:
            train_stats = _phase3g_train_conditional_head_torch(
                config=config,
                head=head,
                train_index=train_index,
                node_embeddings=node_embeddings,
                action_embeddings=action_embeddings,
            )
        finally:
            if stream_e3_gamma:
                config.sspm_epochs = original_epochs
        train_stats["conditional_train_data_mode"] = "stream_event"
        train_stats["train_data_mode"] = "stream_event"
        train_stats["conditional_memmap_fingerprint"] = dict(stream_sentinel)
        train_stats["conditional_memmap_path"] = ""
    else:
        raise ValueError(
            "conditional_action_semantic full training requires "
            "SSPM_CONDITIONAL_TRAIN_DATA_MODE=memmap; stream_event must be explicit",
        )
    checkpoint_path = _phase3g_resolved_head_checkpoint_path(config)
    target_case_mode = (
        "action_embedding_only"
        if str(config.sspm_score_head) == CONDITIONAL_ACTION_EMBEDDING_SCORE_HEAD
        else "conditional_cold_action_else_event"
    )
    target_name = (
        "e_action"
        if str(config.sspm_score_head) == CONDITIONAL_ACTION_EMBEDDING_SCORE_HEAD
        else "conditional_target"
    )
    metadata = {
        "score_head": str(config.sspm_score_head),
        "conditional_head_arch": str(config.sspm_conditional_head_arch),
        "head_arch": str(config.sspm_conditional_head_arch),
        "node_repr_fusion": str(config.node_repr_fusion),
        "loss_type": str(config.conditional_semantic_loss),
        "target_case_mode": target_case_mode,
        "target_case_head_mode": str(head.config.target_case_head_mode),
        "target": target_name,
        "dataset": str(config.dataset),
        "target_mode": str(config.sspm_target_mode),
        "node_word2vec_source": str(config.node_word2vec_source),
        "node_embedding_fingerprint": _phase3g_file_fingerprint(paths["node_embeddings"]),
        "action_embedding_fingerprint": _phase3g_file_fingerprint(paths["action_embeddings"]),
        "event_index_fingerprint": dict(dict(event_meta.get("splits", {})).get("train", {})).get(
            "event_index_fingerprint",
        ),
        "state_model": str(config.sspm_state_model),
        "train_backend": str(config.sspm_train_backend),
        "infer_backend": str(config.sspm_infer_backend),
        "context_dim": int(input_dim),
        "target_logic": "state_exists_event_semantic_else_both_cold_action",
        "conditional_train_data_mode": str(
            train_stats.get("conditional_train_data_mode", config.sspm_conditional_train_data_mode),
        ),
        "conditional_memmap_fingerprint": (
            dict(conditional_memmap_meta.get("fingerprint", {}))
            if conditional_memmap_meta is not None
            else dict(
                train_stats.get(
                    "conditional_memmap_fingerprint",
                    _phase3g_conditional_memmap_sentinel(config),
                ),
            )
        ),
        "train_stats": train_stats,
    }
    target_case_path_for_fp = str(train_stats.get("conditional_memmap_target_case_path", ""))
    metadata["target_case_fingerprint"] = (
        _phase3g_file_fingerprint(target_case_path_for_fp)
        if target_case_path_for_fp
        else {}
    )
    head_fingerprint = head.fingerprint()
    metadata["checkpoint_schema"] = str(head_fingerprint.get("schema", ""))
    saved_path = head.save(checkpoint_path, metadata=metadata)
    _, saved_metadata = ConditionalSemanticHead.load(
        saved_path,
        expected_head_arch=str(config.sspm_conditional_head_arch),
    )
    checkpoint_fingerprint = dict(saved_metadata.get("fingerprint", {}))
    checkpoint_schema = str(
        saved_metadata.get("checkpoint_schema", checkpoint_fingerprint.get("schema", "")),
    )
    _stage_log(
        config,
        "phase3g_conditional_train_end",
        score_head=str(config.sspm_score_head),
        checkpoint=str(saved_path),
    )
    eval_payload = {
        "dataset": str(config.dataset),
        "out_tag": str(config.out_tag),
        "phase3g": {
            "score_head": str(config.sspm_score_head),
            "conditional_head_arch": str(config.sspm_conditional_head_arch),
            "head_arch": str(config.sspm_conditional_head_arch),
            "checkpoint_schema": checkpoint_schema,
            "node_repr_fusion": str(config.node_repr_fusion),
            "conditional_semantic_loss": str(config.conditional_semantic_loss),
            "target": target_name,
            "target_case_mode": target_case_mode,
            "target_case_head_mode": str(head.config.target_case_head_mode),
            "action_head_checkpoint_path": str(saved_path),
            "checkpoint_path": str(saved_path),
            "input_dim": int(input_dim),
            "context_dim": int(input_dim),
            "rank": int(config.rank),
            "output_dim": int(config.sspm_target_dim),
            "fingerprint": checkpoint_fingerprint,
            "train_stats": train_stats,
        },
        "train_events_actual": int(train_count),
        "validation_events_actual": int(validation_count),
        "test_events_actual": 0,
        "runtime": {"elapsed_seconds": float(time.perf_counter() - started)},
        "leakage_contract": {
            "contains_test_scores": False,
            "contains_test_labels": False,
            "ground_truth_used": False,
        },
    }
    eval_path = output_dir / "eval_causal_semantics_slim.json"
    eval_path.write_text(
        json.dumps(eval_payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    dummy_model = SSPMLowRankModel(_make_sspm_config(config, _load_process_config(config)))
    write_effective_config(
        output_dir,
        config,
        dummy_model,
        True,
        train_count=int(train_count),
        validation_count=int(validation_count),
        test_count=0,
    )
    write_metrics_json(
        output_dir,
        config,
        dummy_model,
        eval_payload,
        train_count=int(train_count),
        validation_count=int(validation_count),
        test_count=0,
        embedder_loaded=True,
    )
    return eval_path


def _phase3g_conditional_memmap_only_payload(
    *,
    config: SlimConfig,
    output_dir: Path,
    memmap_meta: Mapping[str, Any],
    train_count: int,
    validation_count: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Build the metadata payload for a train-only conditional memmap run."""
    return {
        "dataset": str(config.dataset),
        "out_tag": str(config.out_tag),
        "phase3g": {
            "score_head": str(config.sspm_score_head),
            "conditional_head_arch": str(config.sspm_conditional_head_arch),
            "head_arch": str(config.sspm_conditional_head_arch),
            "train_data_mode": "conditional_memmap",
            "conditional_train_data_mode": "memmap",
            "conditional_memmap": dict(memmap_meta),
            "checkpoint_path": "",
            "action_head_checkpoint_path": "",
        },
        "outputs": {
            "result_dir": str(output_dir),
            "x_path": str(memmap_meta.get("x_path", "")),
            "y_path": str(memmap_meta.get("y_path", "")),
            "target_case_path": str(memmap_meta.get("target_case_path", "")),
        },
        "train_events_actual": int(train_count),
        "validation_events_actual": int(validation_count),
        "test_events_actual": 0,
        "runtime": {"elapsed_seconds": float(elapsed_seconds)},
        "leakage_contract": {
            "contains_test_scores": False,
            "contains_test_labels": False,
            "ground_truth_used": False,
            "purpose": "train-only conditional memmap cache build",
        },
    }


def _write_phase3g_conditional_memmap_only_outputs(
    *,
    config: SlimConfig,
    output_dir: Path,
    eval_payload: Mapping[str, Any],
    train_count: int,
    validation_count: int,
) -> Path:
    """Write sidecars for a conditional-memmap-only run without training a head."""
    eval_path = output_dir / "eval_causal_semantics_slim.json"
    eval_path.write_text(
        json.dumps(dict(eval_payload), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    dummy_model = SSPMLowRankModel(_make_sspm_config(config, _load_process_config(config)))
    write_effective_config(
        output_dir,
        config,
        dummy_model,
        True,
        train_count=int(train_count),
        validation_count=int(validation_count),
        test_count=0,
    )
    write_metrics_json(
        output_dir,
        config,
        dummy_model,
        dict(eval_payload),
        train_count=int(train_count),
        validation_count=int(validation_count),
        test_count=0,
        embedder_loaded=True,
    )
    return eval_path


def _phase3g_load_or_build_validation_cache(
    *_args: object,
    **_kwargs: object,
) -> tuple[dict[str, Any], float, np.ndarray]:
    """Reject historical action-predict validation caches from active code."""
    raise ValueError(
        "historical Phase3G action-predict validation cache moved to legacy; "
        "current best chains use conditional validation caches",
    )


def _phase3g_load_or_build_conditional_validation_cache(
    *,
    config: SlimConfig,
    head: ConditionalSemanticHead,
    head_metadata: Mapping[str, Any] | None = None,
    paths: Mapping[str, Path],
    event_meta: Mapping[str, Any],
    validation_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    state_model: SSPMLowRankModel | None = None,
    endpoint_suppression_cache_fingerprint: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    fingerprint = _phase3g_conditional_fingerprint(
        config=config,
        head=head,
        head_metadata=head_metadata,
        paths=paths,
        event_meta=event_meta,
        event_index=validation_index,
        split="validation",
        endpoint_suppression_cache_fingerprint=endpoint_suppression_cache_fingerprint,
    )
    cache_dir = _phase3g_action_cache_dir(config, fingerprint)
    try:
        meta = load_conditional_validation_cache(cache_dir, expected_fingerprint=fingerprint)
        scores = np.memmap(
            meta["validation_conditional_scores"],
            dtype=np.float32,
            mode="r",
            shape=(int(meta["num_scores"]),),
        )
        cases = np.memmap(
            meta["validation_target_case"],
            dtype=np.int8,
            mode="r",
            shape=(int(meta["num_scores"]),),
        )
        return meta, scores, cases
    except FileNotFoundError:
        pass
    _stage_log(config, "phase3g_conditional_validation_cache_build_start", path=str(cache_dir))
    (
        scores,
        cases,
        action_ids,
        src_type_ids,
        dst_type_ids,
        count,
        validation_state_merge_profile,
    ) = _phase3g_conditional_scores_stream(
        config=config,
        head=head,
        event_index=validation_index,
        node_embeddings=node_embeddings,
        action_embeddings=action_embeddings,
        state_model=state_model,
        return_profile=True,
    )
    if int(count) != int(validation_index.shape[0]):
        raise ValueError("Phase3G conditional validation count mismatch")
    meta = write_conditional_validation_cache(
        cache_dir,
        scores=scores,
        target_case_ids=cases,
        fingerprint=fingerprint,
        threshold_mode=str(config.event_threshold_mode),
        threshold_quantile=float(config.event_threshold_quantile),
        min_case_count=2,
        action_ids=action_ids,
        src_type_ids=src_type_ids,
        dst_type_ids=dst_type_ids,
        group_min_count=int(config.conditional_group_min_count),
        low_support_policy=str(config.conditional_low_support_policy),
        low_support_margin=float(config.conditional_low_support_margin),
        unseen_group_policy=str(config.conditional_unseen_group_policy),
        global_extreme_quantile=float(config.conditional_global_extreme_quantile),
        adaptive_margin_n1=int(config.conditional_adaptive_margin_n1),
        adaptive_margin_n2=int(config.conditional_adaptive_margin_n2),
        adaptive_margin_low=float(config.conditional_adaptive_margin_low),
        adaptive_margin_mid=float(config.conditional_adaptive_margin_mid),
        adaptive_margin_high=float(config.conditional_adaptive_margin_high),
    )
    meta["validation_state_merge_profile"] = dict(validation_state_merge_profile)
    meta_path = Path(cache_dir) / "validation_calibration_meta.json"
    if meta_path.exists():
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    _stage_log(config, "phase3g_conditional_validation_cache_build_end", count=count)
    mapped_scores = np.memmap(
        meta["validation_conditional_scores"],
        dtype=np.float32,
        mode="r",
        shape=(int(meta["num_scores"]),),
    )
    mapped_cases = np.memmap(
        meta["validation_target_case"],
        dtype=np.int8,
        mode="r",
        shape=(int(meta["num_scores"]),),
    )
    return meta, mapped_scores, mapped_cases


def _phase3g_load_or_build_action_embedding_validation_cache(
    *,
    config: SlimConfig,
    head: ConditionalSemanticHead,
    paths: Mapping[str, Path],
    event_meta: Mapping[str, Any],
    validation_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
) -> tuple[dict[str, Any], float, np.ndarray]:
    fingerprint = _phase3g_action_embedding_fingerprint(
        config=config,
        head=head,
        paths=paths,
        event_meta=event_meta,
        event_index=validation_index,
        split="validation",
    )
    cache_dir = _phase3g_action_cache_dir(config, fingerprint)
    try:
        meta = load_action_embedding_validation_cache(
            cache_dir,
            expected_fingerprint=fingerprint,
        )
        scores = np.memmap(
            meta["validation_action_embedding_scores"],
            dtype=np.float32,
            mode="r",
            shape=(int(meta["num_scores"]),),
        )
        return meta, float(meta["threshold"]), scores
    except FileNotFoundError:
        pass
    _stage_log(
        config,
        "phase3g_action_embedding_validation_cache_build_start",
        path=str(cache_dir),
    )
    scores, count = _phase3g_action_embedding_scores_stream(
        config=config,
        head=head,
        event_index=validation_index,
        node_embeddings=node_embeddings,
        action_embeddings=action_embeddings,
    )
    if int(count) != int(validation_index.shape[0]):
        raise ValueError("Phase3G conditional action embedding validation count mismatch")
    meta = write_action_embedding_validation_cache(
        cache_dir,
        scores=scores,
        fingerprint=fingerprint,
        threshold_mode=str(config.event_threshold_mode),
        threshold_quantile=float(config.event_threshold_quantile),
    )
    _stage_log(config, "phase3g_action_embedding_validation_cache_build_end", count=count)
    mapped_scores = np.memmap(
        meta["validation_action_embedding_scores"],
        dtype=np.float32,
        mode="r",
        shape=(int(meta["num_scores"]),),
    )
    return meta, float(meta["threshold"]), mapped_scores


def _phase3g_conditional_threshold_for_case(
    meta: Mapping[str, Any],
    case_name: str,
) -> float:
    thresholds = dict(meta.get("thresholds_by_target_case", {}))
    if str(case_name) in thresholds:
        return float(dict(thresholds[str(case_name)]).get("threshold", meta.get("threshold", 0.0)))
    return float(meta.get("threshold", 0.0))


def _score_phase3g_action_fast_stream(*_args: object, **_kwargs: object) -> dict[str, Any]:
    """Reject historical action-predict inference from active code."""
    raise ValueError(
        "historical Phase3G action-predict inference moved to legacy; "
        "current best chains use conditional_action_semantic",
    )


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
    action_policy_counts: dict[tuple[int, int, int], dict[str, Any]] = {}
    node_evidence_count = 0
    demoted_event_count = 0
    budget_capped_event_count = 0
    high_priority_event_alert_count = 0
    suppressed_event_alerts_raw: list[dict[str, Any]] = []
    suppressed_raw_path = ""
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
        if output_dir is not None:
            raw_paths = _raw_output_paths(Path(output_dir))
            raw_paths["events"] = Path(output_dir) / "online_event_alerts.csv"
            if int(config.max_test_events) > 0 or not _online_minimal_enabled(config):
                raw_paths["event_score_trace"] = Path(output_dir) / "online_event_score_trace.csv"
            raw_paths["suppressed_endpoint_events"] = (
                Path(output_dir) / "conditional_endpoint_suppressed_events.raw.csv"
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
                    endpoint_suppression_seconds += time.perf_counter() - suppression_started
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
                    alert_row["endpoint_validation_pair_count"] = endpoint_validation_pair_count
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
    test_summary["test_above_threshold_count"] = int(event_alert_count)
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


def _score_phase3g_action_embedding_fast_stream(
    *,
    config: SlimConfig,
    head: ConditionalSemanticHead,
    profile_model: SSPMLowRankModel,
    test_index: np.ndarray,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    threshold: float,
    output_dir: Path | None,
) -> dict[str, Any]:
    scoring_started = time.perf_counter()
    records = _phase3e_event_index_array(test_index)
    model = SSPMLowRankModel(_make_sspm_config(config, _load_process_config(config)))
    type_eye = _phase3g_type_eye()
    threshold_value = float(threshold)
    score_summary_obj = _StreamingScoreSummary(threshold_value)
    raw_paths: dict[str, Path] = {}
    event_alert_count = 0
    test_count = 0
    start_rss_mb = _current_rss_mb()
    test_phase_peak_rss_mb = start_rss_mb
    node_pool: dict[int, dict[str, Any]] = {}
    _stage_log(
        config,
        "phase3g_action_embedding_fast_test_start",
        count=int(records.shape[0]),
    )
    model.reset_state()
    with ExitStack() as stack:
        event_writer = None
        profile_writer = None
        if output_dir is not None:
            raw_paths = _raw_output_paths(Path(output_dir))
            raw_paths["events"] = Path(output_dir) / "online_event_alerts.csv"
            event_writer = stack.enter_context(
                StreamingCsvWriter(raw_paths["events"], EVENT_RAW_FIELDS),
            )
            profile_writer = stack.enter_context(
                StreamingCsvWriter(raw_paths["profiling"], PROFILING_FIELDS),
            )
        chunk_size = max(int(config.sspm_infer_chunk_events), 1)
        for start in range(0, int(records.shape[0]), chunk_size):
            end = min(start + chunk_size, int(records.shape[0]))
            chunk = records[start:end]
            contexts = np.zeros((int(chunk.shape[0]), int(head.config.input_dim)), dtype=np.float32)
            targets = np.zeros((int(chunk.shape[0]), int(head.config.output_dim)), dtype=np.float32)
            for offset, row in enumerate(chunk):
                z_state = _phase3e_target_from_index_row(row, node_embeddings, action_embeddings)
                context, src_type_name, dst_type_name, _, _ = _phase3g_action_context_from_model(
                    model,
                    row,
                    node_embeddings,
                    type_eye,
                )
                contexts[offset] = context
                targets[offset] = select_action_embedding_target(row, action_embeddings)
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
            predictions = head.predict(contexts)
            scores = conditional_distance(
                predictions,
                targets,
                str(config.conditional_semantic_loss),
            )
            for offset, row in enumerate(chunk):
                stream_pos = start + offset
                test_count = int(stream_pos) + 1
                event_score = float(scores[offset])
                score_summary_obj.observe(event_score)
                if event_score >= threshold_value:
                    event_alert_count += 1
                    alert_row = _phase3f_raw_alert_row(
                        row,
                        stream_pos,
                        event_score,
                        threshold_value,
                        event_score,
                        event_score,
                        f"validation_{config.event_threshold_mode}_action_embedding",
                    )
                    alert_row["target_case"] = "action_embedding_target"
                    alert_row["target_case_threshold"] = threshold_value
                    alert_row["threshold_level"] = "global"
                    alert_row["threshold_group_key"] = "action_embedding"
                    alert_row["validation_group_count"] = ""
                    if event_writer is not None:
                        event_writer.write_row(alert_row)
                    src_idx = int(row["src_node_idx"])
                    dst_idx = int(row["dst_node_idx"])
                    _minimal_node_pool_update(
                        node_pool,
                        src_idx,
                        int(row["event_id"]),
                        stream_pos,
                        event_score,
                        event_score,
                    )
                    if dst_idx != src_idx:
                        _minimal_node_pool_update(
                            node_pool,
                            dst_idx,
                            int(row["event_id"]),
                            stream_pos,
                            event_score,
                            event_score,
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
                    "phase3g_action_embedding_fast_test_progress",
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
                        profile_model,
                        current_rss_mb=current_rss_mb,
                        test_phase_peak_rss_mb=test_phase_peak_rss_mb,
                    ),
                    verbose=config.verbose,
                )
        final_rss_mb = _current_rss_mb()
        test_phase_peak_rss_mb = max(test_phase_peak_rss_mb, final_rss_mb)
    test_scoring_seconds = float(time.perf_counter() - scoring_started)
    smaps = _current_smaps_rollup_mb()
    test_summary = score_summary_obj.summary(event_alert_count=event_alert_count)
    test_summary["test_above_threshold_count"] = int(event_alert_count)
    return {
        "test_count": int(test_count),
        "final_nodes_raw": _final_node_pool_minimal(node_pool),
        "threshold": threshold_value,
        "test_score_summary": test_summary,
        "raw_outputs_streamed": output_dir is not None,
        "output_dir": str(output_dir) if output_dir is not None else "",
        "raw_output_paths": {key: str(path) for key, path in raw_paths.items()},
        "event_alert_count": int(event_alert_count),
        "checkpoint_count": 0,
        "residual_semantic_examples": {group: [] for group in RESIDUAL_EXAMPLE_GROUPS},
        "test_scoring_seconds": test_scoring_seconds,
        "stream_csv_write_seconds": 0.0,
        "threshold_controller": {
            "mode": str(config.event_threshold_mode),
            "initial_threshold": threshold_value,
            "final_threshold": threshold_value,
        },
        "compat_in_memory_outputs_active": False,
        "raw_alerts_written": bool(output_dir is not None),
        "analysis_outputs_written": False,
        "state_merge_diagnostics_written": False,
        "rss_test_peak_mb": float(test_phase_peak_rss_mb),
        "process_peak_rss_mb": _safe_peak_rss_mb(),
        "state_merge_diagnostics_truncated": False,
        "phase3g_conditional_action_embedding_enabled": True,
        "phase3f_fast_path_enabled": True,
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
        "online_minimal_anonymous_rss_mb": smaps.get("anonymous_rss_mb"),
        "online_minimal_file_backed_rss_mb": smaps.get("file_backed_rss_mb"),
    }


def run_phase3g_action_load_and_infer_from_precompute(config: SlimConfig) -> Path:
    """Reject historical action-predict checkpoint inference from active code."""
    del config
    raise ValueError(
        "historical Phase3G action-predict inference moved to legacy; "
        "current best chains use conditional_action_semantic",
    )


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
    test_summary = dict(stream_outputs.get("test_score_summary", {}))
    test_by_case = dict(stream_outputs.get("test_score_summary_by_target_case", {}))
    validation_summary = dict(cache_meta.get("summary", {}))
    validation_by_case = dict(cache_meta.get("summary_by_target_case", {}))
    group_summary_path = str(stream_outputs.get("conditional_group_summary_csv", ""))
    required_dual_head_summary_paths = _phase3g_write_required_dual_head_summary_csvs(
        output_dir=output_dir,
        stream_outputs=stream_outputs,
        group_summary_csv=group_summary_path,
    )
    score_summary_payload = {
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
        "action_type_alert_policy": dict(
            stream_outputs.get("action_type_alert_policy", {}),
        ),
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
        "group_alert_policy_eval_rows": group_alert_policy_rows,
    }
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


def run_phase3g_action_embedding_load_and_infer_from_precompute(config: SlimConfig) -> Path:
    """Run Phase3G conditional-action-embedding inference from Phase3E artifacts."""
    _validate_active_event_score_mode(config.event_score_mode)
    if str(config.sspm_train_mode) != "load_and_infer":
        raise ValueError(
            "conditional_action_embedding infer requires SSPM_TRAIN_MODE=load_and_infer",
        )
    checkpoint_path = _phase3g_resolved_head_checkpoint_path(config)
    started = time.perf_counter()
    output_dir = Path(config.result_root) / config.out_tag
    output_dir.mkdir(parents=True, exist_ok=True)
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
    head, head_metadata = ConditionalSemanticHead.load(
        checkpoint_path,
        expected_head_arch=CONDITIONAL_HEAD_ARCH_SHARED_V1,
    )
    if int(head.config.input_dim) != _phase3g_action_input_dim(node_embeddings):
        raise ValueError("conditional action embedding input_dim does not match artifacts")
    validation_started = time.perf_counter()
    cache_meta, threshold, validation_scores = (
        _phase3g_load_or_build_action_embedding_validation_cache(
            config=config,
            head=head,
            paths=paths,
            event_meta=event_meta,
            validation_index=validation_index,
            node_embeddings=node_embeddings,
            action_embeddings=action_embeddings,
        )
    )
    timing = {"validation_seconds": float(time.perf_counter() - validation_started)}
    model = SSPMLowRankModel(_make_sspm_config(config, _load_process_config(config)))
    stream_outputs = _score_phase3g_action_embedding_fast_stream(
        config=config,
        head=head,
        profile_model=model,
        test_index=test_index,
        node_embeddings=node_embeddings,
        action_embeddings=action_embeddings,
        threshold=float(threshold),
        output_dir=output_dir,
    )
    timing["test_scoring_seconds"] = float(stream_outputs.get("test_scoring_seconds", 0.0))
    test_summary = dict(stream_outputs.get("test_score_summary", {}))
    validation_summary = dict(cache_meta.get("summary", {}))
    score_summary_payload = {
        "score_head": CONDITIONAL_ACTION_EMBEDDING_SCORE_HEAD,
        "conditional_semantic_loss": str(config.conditional_semantic_loss),
        "target": "e_action",
        "event_threshold_mode": str(config.event_threshold_mode),
        "event_threshold_quantile": float(config.event_threshold_quantile),
        "final_threshold": float(threshold),
        "validation": validation_summary,
        "test": test_summary,
    }
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
    memory_profile["process_peak_rss_mb"] = float(stream_outputs.get("process_peak_rss_mb", 0.0))
    eval_payload = _eval_payload(
        config,
        stream_outputs,
        model,
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
            "score_summary": score_summary_payload,
            "score_summary_json": str(score_summary_path),
            "phase3g": {
                "score_head": CONDITIONAL_ACTION_EMBEDDING_SCORE_HEAD,
                "node_repr_fusion": str(config.node_repr_fusion),
                "conditional_semantic_loss": str(config.conditional_semantic_loss),
                "target": "e_action",
                "action_head_checkpoint_path": str(checkpoint_path),
                "conditional_action_embedding_head_metadata": head_metadata,
                "validation_cache_dir": str(
                    Path(cache_meta["validation_action_embedding_scores"]).parent,
                ),
                "validation_cache_meta": cache_meta,
            },
        },
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
        model,
        True,
        train_count=0,
        validation_count=int(validation_count),
        test_count=int(test_count),
    )
    write_metrics_json(
        output_dir,
        config,
        model,
        eval_payload,
        train_count=0,
        validation_count=int(validation_count),
        test_count=int(test_count),
        embedder_loaded=True,
    )
    del validation_scores
    return eval_path


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


def _phase3e_validate_index_row(row: np.void) -> None:
    dtype = getattr(row, "dtype", None)
    if dtype != EVENT_INDEX_DTYPE:
        raise TypeError(f"event index row must use EVENT_INDEX_DTYPE, got {dtype}")


def _phase3e_entity_type_name(field_name: str, type_id: int) -> str:
    if int(type_id) not in ENTITY_TYPE_NAMES:
        raise ValueError(f"invalid {field_name}: {int(type_id)}")
    return ENTITY_TYPE_NAMES[int(type_id)]


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
            return freebsd_process_natural_tokens(meta.get("cmd", ""))
    if kind == "file":
        meta = node_maps.get("file_meta", {}).get(int(node_id), {})
        if is_clearscope_dataset(config.dataset):
            if (
                normalize_clearscope_semantic_mode(config.semantic_mode)
                != CLEARSCOPE_LEGACY_SEMANTIC_MODE
            ):
                return clearscope_file_natural_tokens_refined(meta.get("path", ""))
            return clearscope_file_natural_tokens(meta.get("path", ""))
        if is_theia_dataset(config.dataset):
            return linux_file_natural_tokens(meta.get("path", ""))
        if is_cadets_dataset(config.dataset):
            return freebsd_file_natural_tokens(meta.get("path", ""))
    if kind == "netflow":
        meta = node_maps.get("netflow_meta", {}).get(int(node_id), {})
        src_addr = meta.get("src_addr", "")
        dst_addr = meta.get("dst_addr", "")
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
) -> tuple[int, int, int, int, int, int]:
    """Return the six-field event_index tuple from one direct DB event."""
    action = str(event.operation)
    if action not in action_name_to_id:
        raise KeyError(f"Phase3E event action is outside ORTHRUS10: {action}")
    src_node_id = int(event.src_node_id)
    dst_node_id = int(event.dst_node_id)
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
    x_dir = _phase3e_resolve_cache_dir(config.x_context_cache_dir, config)
    return {
        "node_embeddings": node_dir / "node_embeddings.npy",
        "node_meta": node_dir / "node_embedding_meta.json",
        "action_embeddings": action_dir / "action_embeddings.npy",
        "action_meta": action_dir / "action_embedding_meta.json",
        "event_index_train": event_dir / "event_index_train.memmap",
        "event_index_validation": event_dir / "event_index_validation.memmap",
        "event_index_test": event_dir / "event_index_test.memmap",
        "event_meta": event_dir / "event_index_meta.json",
        "x_context_meta": x_dir / "x_context_meta.json",
        "x_context_ema_fixed": x_dir / "X_context_ema_fixed.memmap",
        "x_context_s4d_complex_node": x_dir / "X_context_s4d_complex_node.memmap",
    }


def _phase3e_require_artifacts(config: SlimConfig) -> dict[str, Path]:
    paths = _phase3e_artifact_paths(config)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        detail = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(f"Phase3E precompute artifact missing: {detail}")
    return paths


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


def _phase3e_checkpoint_meta(
    *,
    config: SlimConfig,
    paths: Mapping[str, Path],
    event_meta: Mapping[str, Any],
    x_context_key: str,
    train_stats: Mapping[str, Any],
    model: SSPMLowRankModel,
) -> dict[str, Any]:
    """Build small Phase3E checkpoint metadata from precompute sidecars."""
    node_meta = _phase3e_load_json(paths["node_meta"])
    action_meta = _phase3e_load_json(paths["action_meta"])
    x_meta = _phase3e_load_json(paths["x_context_meta"])
    train_split = dict(dict(event_meta.get("splits", {})).get("train", {}))
    test_split = dict(dict(event_meta.get("splits", {})).get("test", {}))
    validation_split = dict(dict(event_meta.get("splits", {})).get("validation", {}))
    x_info = dict(x_meta.get(x_context_key, {}))
    if not x_info and str(x_context_key) == "event_index_real_diag":
        x_info = {
            "fingerprint": "not_applicable:real_diag_streaming_gamma",
            "path": "",
            "reason": "E3 real_diag_learnable uses streaming gamma train from event_index",
        }
    meta: dict[str, Any] = {
        "word2vec_model_path": str(config.pretrained_residual_embedder_path),
        "word2vec_fingerprint": str(node_meta.get("word2vec_fingerprint", "")),
        "node_embedding_fingerprint": _phase3e_file_fingerprint(paths["node_embeddings"]),
        "action_embedding_fingerprint": _phase3e_file_fingerprint(paths["action_embeddings"]),
        "event_index_fingerprint": train_split.get("fingerprint", {}),
        "x_context_fingerprint": str(x_info.get("fingerprint", "")),
        "node_embedding_path": str(paths["node_embeddings"]),
        "action_embedding_path": str(paths["action_embeddings"]),
        "event_index_path": str(paths["event_index_train"]),
        "x_context_path": str(x_info.get("path", "")),
        "x_context_reason": str(x_info.get("reason", "")),
        "node_embedding_meta_path": str(paths["node_meta"]),
        "action_embedding_meta_path": str(paths["action_meta"]),
        "event_index_meta_path": str(paths["event_meta"]),
        "x_context_meta_path": str(paths["x_context_meta"]),
        "target_dim": int(model.target_dim),
        "state_dim": int(model.state_dim),
        "context_dim": int(model.context_dim),
        "rank": int(model.rank),
        "train_events_actual": int(train_split.get("num_events", 0)),
        "validation_events_actual": int(validation_split.get("num_events", 0)),
        "test_events_actual": int(test_split.get("num_events", 0)),
        "all_node_token_coverage": float(node_meta.get("all_node_token_coverage", 0.0)),
        "action_token_coverage": float(action_meta.get("action_token_coverage", 0.0)),
        "all_oov_node_ratio": float(node_meta.get("all_oov_node_ratio", 0.0)),
        "partial_oov_node_ratio": float(node_meta.get("partial_oov_node_ratio", 0.0)),
        "num_nodes_total": int(node_meta.get("num_nodes_total", 0)),
        "num_nodes_all_oov": int(node_meta.get("num_nodes_all_oov", 0)),
        "num_nodes_partial_oov": int(node_meta.get("num_nodes_partial_oov", 0)),
        "zero_node_count": int(node_meta.get("zero_node_count", 0)),
        "fallback_count": int(node_meta.get("fallback_count", 0)),
        "num_nodes": int(node_meta.get("num_nodes_total", 0)),
        "num_actions": 10,
        "train_stats": dict(train_stats),
        "fingerprint": stable_json_hash(
            {
                "target_mode": str(config.sspm_target_mode),
                "state_model": str(config.sspm_state_model),
                "node_embedding": _phase3e_file_fingerprint(paths["node_embeddings"]),
                "action_embedding": _phase3e_file_fingerprint(paths["action_embeddings"]),
                "event_index": train_split.get("fingerprint", {}),
                "x_context": str(x_info.get("fingerprint", "")),
            },
        ),
    }
    for key in (
        "loss_by_epoch",
        "torch_device",
        "torch_cuda_available",
        "torch_device_name",
        "torch_optimizer",
        "torch_lr",
        "torch_weight_decay",
        "torch_batch_events",
        "torch_epochs_completed",
        "torch_train_loss_best",
        "torch_early_stop_reason",
        "torch_train_time_sec",
        "torch_peak_gpu_memory_mb",
        "torch_early_stop_patience",
        "torch_early_stop_min_delta",
        "real_diag_train_gamma",
        "real_diag_gamma_active",
        "real_diag_sensitivity_mode",
        "gamma_initial_mean",
        "gamma_initial_min",
        "gamma_initial_max",
        "gamma_final_mean",
        "gamma_final_min",
        "gamma_final_max",
        "gamma_delta_l2",
        "gamma_grad_norm_mean",
        "gamma_grad_norm_max",
        "sensitivity_mode",
        "sensitivity_nodes",
        "train_seconds",
        "train_rss_peak_mb",
        "train_epochs_completed",
        "train_events_seen_total",
        "early_stop_reason",
        "batch_events",
    ):
        if key in train_stats:
            meta[key] = train_stats[key]
    return meta


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


def run_phase3e_train_base_from_precompute(config: SlimConfig) -> Path:
    """Train one Phase3E base checkpoint from existing precompute artifacts."""
    if str(config.sspm_score_head) in {
        "conditional_action_semantic",
    }:
        return run_phase3g_conditional_train_from_precompute(config)
    _validate_active_event_score_mode(config.event_score_mode)
    if str(config.sspm_train_mode) != "train_and_save":
        raise ValueError("Phase3E train_base requires SSPM_TRAIN_MODE=train_and_save")
    if str(config.sspm_target_mode) != "node_action_semantic_mean":
        raise ValueError("Phase3E train_base requires node_action_semantic_mean")
    if not str(config.sspm_checkpoint_path).strip():
        raise ValueError("Phase3E train_base requires --sspm_checkpoint_path")
    started = time.perf_counter()
    output_dir = Path(config.result_root) / config.out_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    process_cfg = _load_process_config(config)
    paths, event_meta, train_index, node_embeddings, action_embeddings = (
        _phase3e_open_precompute_arrays(config)
    )
    train_count = int(dict(dict(event_meta.get("splits", {})).get("train", {})).get("num_events", 0))
    validation_count = int(
        dict(dict(event_meta.get("splits", {})).get("validation", {})).get("num_events", 0),
    )
    if validation_count <= 0:
        raise ValueError("Phase3E train_base requires non-empty validation event_index")
    embedder = load_pretrained_residual_embedder(
        str(config.pretrained_residual_embedder_path),
        expected_dim=int(config.sspm_target_dim),
    )
    model = SSPMLowRankModel(_make_sspm_config(config, process_cfg))
    x_context_key = ""
    train_stats: dict[str, Any]
    if str(config.sspm_state_model) in {"ema_fixed", "s4d_complex_node"}:
        x_context_key = (
            "ema_fixed"
            if str(config.sspm_state_model) == "ema_fixed"
            else "s4d_complex_node"
        )
        x_context_path = paths[f"x_context_{x_context_key}"]
        x_context = np.memmap(
            x_context_path,
            dtype=np.float32,
            mode="r",
            shape=(train_count, int(model.context_dim)),
        )
        _stage_log(
            config,
            "phase3e_torch_train_start",
            state_model=str(config.sspm_state_model),
            count=train_count,
            batch_events=int(config.sspm_torch_batch_events),
        )
        torch_stats = train_lowrank_head_torch(
            model=model,
            x_context=x_context,
            event_index=train_index,
            node_embeddings=node_embeddings,
            action_embeddings=action_embeddings,
            epochs=int(config.sspm_epochs),
            batch_events=int(config.sspm_torch_batch_events),
            lr=_phase3e_train_lr(config),
            weight_decay=float(config.sspm_torch_weight_decay),
            device=str(config.sspm_torch_device),
            patience=int(config.sspm_early_stop_patience),
            min_delta=float(config.sspm_early_stop_min_delta),
        )
        del x_context
        train_stats = dict(torch_stats)
        train_stats["train_events_actual"] = int(train_count)
        train_stats["train_events_seen_total"] = int(
            int(train_count) * int(train_stats.get("torch_epochs_completed", 0)),
        )
        train_stats["train_seconds"] = float(train_stats.get("torch_train_time_sec", 0.0))
        train_stats["train_rss_peak_mb"] = float(_current_rss_mb())
        _stage_log(
            config,
            "phase3e_torch_train_end",
            state_model=str(config.sspm_state_model),
            epochs=int(train_stats.get("torch_epochs_completed", 0)),
            best_loss=f"{float(train_stats.get('torch_train_loss_best', 0.0)):.6f}",
        )
    elif str(config.sspm_state_model) == "real_diag_learnable":
        x_context_key = "event_index_real_diag"
        if not bool(config.real_diag_train_gamma):
            raise ValueError("Phase3E E3 train_base requires --real_diag_train_gamma")
        model, train_stats = train_phase3e_real_diag_from_event_index(
            config=config,
            event_index=train_index,
            node_embeddings=node_embeddings,
            action_embeddings=action_embeddings,
        )
    else:
        raise ValueError(f"unsupported Phase3E state model: {config.sspm_state_model}")
    if (
        str(config.sspm_update_gate_mode) != "none"
        or str(config.sspm_residual_score_mode) == "var_calibrated"
        or str(config.sspm_residual_calibration) != "none"
    ):
        validation_index, validation_index_count = _phase3e_open_split_event_index(
            paths,
            event_meta,
            "validation",
        )
        try:
            calibration_stats = _phase3e_fit_checkpoint_calibration_from_validation(
                config=config,
                model=model,
                validation_index=validation_index,
                node_embeddings=node_embeddings,
                action_embeddings=action_embeddings,
            )
        finally:
            del validation_index
        if int(validation_index_count) != int(validation_count):
            raise ValueError("Phase3E validation count mismatch during checkpoint calibration")
        train_stats["checkpoint_validation_calibration"] = calibration_stats
    model.training_stats = dict(train_stats)
    phase3e_meta = _phase3e_checkpoint_meta(
        config=config,
        paths=paths,
        event_meta=event_meta,
        x_context_key=x_context_key,
        train_stats=train_stats,
        model=model,
    )
    checkpoint_path = save_sspm_checkpoint(
        config.sspm_checkpoint_path,
        config=config,
        model=model,
        embedder=embedder,
        process_cfg=process_cfg,
        train_count=int(train_count),
        validation_count=int(validation_count),
        phase3e_meta=phase3e_meta,
    )
    eval_payload = _phase3e_train_base_output_payload(
        config=config,
        checkpoint_path=checkpoint_path,
        train_stats=train_stats,
        train_count=int(train_count),
        validation_count=int(validation_count),
        output_dir=output_dir,
        started=started,
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
        True,
        train_count=int(train_count),
        validation_count=int(validation_count),
        test_count=0,
    )
    write_metrics_json(
        output_dir,
        config,
        model,
        eval_payload,
        train_count=int(train_count),
        validation_count=int(validation_count),
        test_count=0,
        embedder_loaded=True,
    )
    del train_index
    del node_embeddings
    del action_embeddings
    return eval_path


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


def _phase3e_x_context_memmap_from_db_stream(
    *,
    path: Path,
    conn: Any,
    year_month: str,
    train_days: Sequence[int],
    node_maps: Mapping[str, Any],
    event_filter: bool,
    fetch_size: int,
    max_events: int,
    expected_count: int,
    node_table: NodeEmbeddingTable,
    action_table: ActionEmbeddingTable,
    node_kind_by_id: Mapping[int, str],
    model: SSPMLowRankModel,
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
    fingerprint_inputs: Mapping[str, object],
) -> dict[str, object]:
    """Build train X_context by streaming PostgreSQL events directly into memmap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if int(expected_count) <= 0:
        path.write_bytes(b"")
        fingerprint = x_context_fingerprint(dict(fingerprint_inputs))
        return {
            "x_context_memmap_enabled": True,
            "enabled": True,
            "x_context_memmap_path": str(path),
            "path": str(path),
            "x_context_memmap_size_mb": 0.0,
            "size_mb": 0.0,
            "x_context_memmap_reused_by": None,
            "x_context_fingerprint": fingerprint,
            "fingerprint": fingerprint,
            "x_context_build_rss_peak_mb": float(_current_rss_mb()),
            "build_rss_peak_mb": float(_current_rss_mb()),
            "x_context_build_events_per_sec": 0.0,
            "events_per_sec": 0.0,
            "state_model": str(model.config.state_model),
            "context_dim": int(model.context_dim),
            "target_dim": int(model.target_dim),
            "state_dim": int(model.state_dim),
            "row_count": 0,
            "build_mode": "postgres_stream_to_memmap",
            "context_order": "context_before_update",
        }
    fingerprint = x_context_fingerprint(
        {
            "caller": dict(fingerprint_inputs),
            "node_embeddings": _phase3e_array_fingerprint(node_table.embeddings),
            "action_embeddings": _phase3e_array_fingerprint(action_table.embeddings),
            "context_dim": int(model.context_dim),
            "rank": int(model.rank),
            "state_dim": int(model.state_dim),
            "target_dim": int(model.target_dim),
            "state_model": str(model.config.state_model),
        },
    )
    x_context = np.memmap(
        path,
        dtype=np.float32,
        mode="w+",
        shape=(int(expected_count), int(model.context_dim)),
    )
    progress_interval = int(config.progress_interval_events)
    model.reset_state()
    start = time.perf_counter()
    rss_peak = _current_rss_mb()
    count = 0
    _stage_log(
        config,
        f"phase3e_x_context_{model.config.state_model}_db_stream_start",
        count=int(expected_count),
    )
    try:
        events = _phase3e_stream_split_events_direct(
            conn,
            year_month=year_month,
            days=train_days,
            event_filter=event_filter,
            fetch_size=int(fetch_size),
            max_events=int(max_events),
            event_id_column=str(fingerprint_inputs.get("event_id_source", "id")),
        )
        for event in events:
            if count >= int(expected_count):
                raise ValueError(
                    f"Phase3E X_context stream exceeded PostgreSQL COUNT: {expected_count}",
                )
            compact_tuple = _phase3e_compact_event_from_direct_stream(
                event,
                action_table.name_to_action_id,
                node_table.node_id_to_idx,
                node_kind_by_id,
            )
            compact = {
                "event_id": compact_tuple[0],
                "src_node_idx": compact_tuple[1],
                "dst_node_idx": compact_tuple[2],
                "action_id": compact_tuple[3],
                "src_type_id": compact_tuple[4],
                "dst_type_id": compact_tuple[5],
            }
            fields = _phase3e_fields_from_compact_event(compact)
            z_t = _phase3e_target_from_fields(
                fields,
                node_table.embeddings,
                action_table.embeddings,
            )
            x_context[count] = model.make_context(fields)
            model.update_states(fields, z_t)
            count += 1
            rss_peak = max(rss_peak, _current_rss_mb())
            if progress_interval > 0 and count % progress_interval == 0:
                _stage_log(
                    config,
                    f"phase3e_x_context_{model.config.state_model}_db_stream_progress",
                    count=count,
                )
        if count != int(expected_count):
            raise ValueError(
                f"Phase3E X_context stream count mismatch: wrote {count}, "
                f"expected {expected_count}",
            )
        x_context.flush()
    finally:
        del x_context
        model.reset_state()
    elapsed = max(time.perf_counter() - start, 1e-9)
    size_mb = float(path.stat().st_size) / 1024.0 / 1024.0
    return {
        "x_context_memmap_enabled": True,
        "enabled": True,
        "x_context_memmap_path": str(path),
        "path": str(path),
        "x_context_memmap_size_mb": float(size_mb),
        "size_mb": float(size_mb),
        "x_context_memmap_reused_by": None,
        "x_context_fingerprint": fingerprint,
        "fingerprint": fingerprint,
        "x_context_build_rss_peak_mb": float(rss_peak),
        "build_rss_peak_mb": float(rss_peak),
        "x_context_build_events_per_sec": float(count / elapsed),
        "events_per_sec": float(count / elapsed),
        "state_model": str(model.config.state_model),
        "context_dim": int(model.context_dim),
        "target_dim": int(model.target_dim),
        "state_dim": int(model.state_dim),
        "row_count": int(count),
        "build_mode": "postgres_stream_to_memmap",
        "context_order": "context_before_update",
    }


def _phase3e_build_x_contexts_from_db_stream(
    *,
    config: SlimConfig,
    conn: Any,
    year_month: str,
    train_days: Sequence[int],
    node_maps: Mapping[str, Any],
    event_filter: bool,
    fetch_size: int,
    max_events: int,
    expected_count: int,
    node_table: NodeEmbeddingTable,
    action_table: ActionEmbeddingTable,
    node_kind_by_id: Mapping[int, str],
    x_context_dir: Path,
    event_index_meta: Mapping[str, Any],
    process_cfg: ProcessSemanticConfig | None,
) -> dict[str, dict[str, object]]:
    """Build train X_context memmaps from PostgreSQL stream, not cached row lists."""
    if not bool(config.x_context_memmap_enabled):
        return {}
    results: dict[str, dict[str, object]] = {}
    for state_model, filename in (
        ("ema_fixed", "X_context_ema_fixed.memmap"),
        ("s4d_complex_node", "X_context_s4d_complex_node.memmap"),
    ):
        model_config = _phase3e_config_for_state_model(config, state_model)
        model = SSPMLowRankModel(_make_sspm_config(model_config, process_cfg))
        meta = _phase3e_x_context_memmap_from_db_stream(
            path=x_context_dir / filename,
            conn=conn,
            year_month=year_month,
            train_days=train_days,
            node_maps=node_maps,
            event_filter=event_filter,
            fetch_size=fetch_size,
            max_events=max_events,
            expected_count=expected_count,
            node_table=node_table,
            action_table=action_table,
            node_kind_by_id=node_kind_by_id,
            model=model,
            config=config,
            process_cfg=process_cfg,
            fingerprint_inputs={
                "dataset": str(config.dataset),
                "split": "train",
                "target_mode": str(config.sspm_target_mode),
                "state_model": str(state_model),
                "event_id_source": str(event_index_meta.get("event_id_source", "id")),
                "event_index_fingerprint": dict(event_index_meta.get("fingerprint", {})),
                "node_embedding_fingerprint": _phase3e_array_fingerprint(
                    node_table.embeddings,
                ),
                "action_embedding_fingerprint": _phase3e_array_fingerprint(
                    action_table.embeddings,
                ),
            },
        )
        if state_model == "ema_fixed":
            meta["x_context_memmap_reused_by"] = ["E2", "E5"]
        elif state_model == "s4d_complex_node":
            meta["x_context_memmap_reused_by"] = ["E4"]
        results[state_model] = dict(meta)
    return results


def _phase3e_build_x_contexts(
    *,
    config: SlimConfig,
    train_event_index_path: str | Path,
    train_event_count: int,
    node_table: NodeEmbeddingTable,
    action_table: ActionEmbeddingTable,
    x_context_dir: Path,
    event_index_meta: Mapping[str, Any],
    process_cfg: ProcessSemanticConfig | None,
) -> dict[str, dict[str, object]]:
    """Build train-only X_context memmaps for E2/E5 and E4."""
    if not bool(config.x_context_memmap_enabled):
        return {}
    records = open_event_index_memmap(train_event_index_path, num_events=train_event_count)
    results: dict[str, dict[str, object]] = {}
    try:
        for state_model, filename in (
            ("ema_fixed", "X_context_ema_fixed.memmap"),
            ("s4d_complex_node", "X_context_s4d_complex_node.memmap"),
        ):
            model_config = _phase3e_config_for_state_model(config, state_model)
            model = SSPMLowRankModel(_make_sspm_config(model_config, process_cfg))
            meta = build_x_context_memmap(
                path=x_context_dir / filename,
                event_index=records,
                node_embeddings=node_table.embeddings,
                action_embeddings=action_table.embeddings,
                model=model,
                fingerprint_inputs={
                    "dataset": str(config.dataset),
                    "split": "train",
                    "target_mode": str(config.sspm_target_mode),
                    "state_model": str(state_model),
                    "event_index_fingerprint": dict(event_index_meta.get("fingerprint", {})),
                    "node_embedding_fingerprint": _phase3e_array_fingerprint(
                        node_table.embeddings,
                    ),
                    "action_embedding_fingerprint": _phase3e_array_fingerprint(
                        action_table.embeddings,
                    ),
                },
            )
            if state_model == "ema_fixed":
                meta["x_context_memmap_reused_by"] = ["E2", "E5"]
            elif state_model == "s4d_complex_node":
                meta["x_context_memmap_reused_by"] = ["E4"]
            results[state_model] = dict(meta)
    finally:
        del records
    return results


def _phase3e_precompute_metrics(
    *,
    config: SlimConfig,
    output_dir: Path,
    node_table: NodeEmbeddingTable,
    action_table: ActionEmbeddingTable,
    node_paths: Mapping[str, str],
    action_paths: Mapping[str, str],
    audit: Mapping[str, object],
    audit_path: str,
    event_index_meta: Mapping[str, Any],
    event_index_meta_path: str,
    split_counts: Mapping[str, int],
    split_full_counts: Mapping[str, int] | None,
    x_context: Mapping[str, Mapping[str, object]],
    x_context_meta_path: str,
    elapsed: float,
    order_by: Sequence[object] | object,
    event_id_source: str,
) -> dict[str, Any]:
    """Return JSON-safe Phase3E precompute metrics without embedding arrays."""
    del output_dir
    total_events = int(sum(int(value) for value in split_counts.values()))
    rss_peak = float(
        max(
            _current_rss_mb(),
            *[
                float(meta.get("x_context_build_rss_peak_mb", 0.0))
                for meta in x_context.values()
            ],
        ),
    )
    metrics: dict[str, Any] = {
        "dataset": str(config.dataset),
        "out_tag": str(config.out_tag),
        "target_mode": str(config.sspm_target_mode),
        "node_word2vec_source": str(config.node_word2vec_source),
        "word2vec_model_path": str(config.pretrained_residual_embedder_path),
        "word2vec_fingerprint": str(node_table.meta.get("word2vec_fingerprint", "")),
        "node_embeddings_path": node_paths["node_embeddings"],
        "node_id_to_idx_path": node_paths["node_id_to_idx"],
        "node_embedding_meta_path": node_paths["node_embedding_meta"],
        "node_embedding_fingerprint": _phase3e_array_fingerprint(node_table.embeddings),
        "action_embeddings_path": action_paths["action_embeddings"],
        "action_id_to_name_path": action_paths["action_id_to_name"],
        "action_embedding_meta_path": action_paths["action_embedding_meta"],
        "action_embedding_fingerprint": _phase3e_array_fingerprint(action_table.embeddings),
        "coverage_audit_path": str(audit_path),
        "coverage": dict(audit),
        "event_index_meta_path": str(event_index_meta_path),
        "event_index": dict(event_index_meta),
        "x_context": {str(key): dict(value) for key, value in x_context.items()},
        "x_context_meta_path": str(x_context_meta_path),
        "train_events_actual": int(split_counts.get("train", 0)),
        "validation_events_actual": int(split_counts.get("validation", 0)),
        "test_events_actual": int(split_counts.get("test", 0)),
        "postgres_train_events_full": int((split_full_counts or {}).get("train", 0)),
        "postgres_validation_events_full": int(
            (split_full_counts or {}).get("validation", 0),
        ),
        "postgres_test_events_full": int((split_full_counts or {}).get("test", 0)),
        "row_order_is_stream_order": True,
        "order_by": _phase3e_json_order_by(order_by),
        "event_id_source": str(event_id_source),
        "event_id_preserved": True,
        "timestamp_order_saved_per_row": False,
        "raw_rows_saved": False,
        "tokens_saved": False,
        "event_embedding_saved": False,
        "x_context_memmap_enabled": bool(config.x_context_memmap_enabled),
        "precompute_seconds": float(elapsed),
        "precompute_events_per_sec": float(total_events / max(float(elapsed), 1e-9)),
        "precompute_rss_peak_mb": rss_peak,
        "phase3e": {
            "target_mode": str(config.sspm_target_mode),
            "train_backend": str(config.sspm_train_backend),
            "infer_backend": str(config.sspm_infer_backend),
            "node_embedding_path": node_paths["node_embeddings"],
            "action_embedding_path": action_paths["action_embeddings"],
            "event_index_path": str(
                dict(event_index_meta.get("splits", {}))
                .get("train", {})
                .get("path", ""),
            ),
            "x_context_path": str(x_context.get("ema_fixed", {}).get("path", "")),
        },
    }
    return metrics


def _phase3e_write_precompute_eval_and_config(
    *,
    output_dir: Path,
    metrics_path: Path,
    audit_path: str,
    event_index_meta_path: str,
    x_context_meta_path: str,
    node_paths: Mapping[str, str],
    action_paths: Mapping[str, str],
    metrics: Mapping[str, Any],
    config: SlimConfig,
    process_cfg: ProcessSemanticConfig | None,
    elapsed: float,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> None:
    """Write Phase3E precompute-only eval JSON and config sidecars."""
    eval_path = output_dir / "eval_causal_semantics_slim.json"
    eval_path.write_text(
        json.dumps(
            {
                "method_name": "causal_semantics_slim",
                "method_version": METHOD_VERSION,
                "phase3e_precompute_only": True,
                "config": asdict(config),
                "runtime": {
                    "elapsed_seconds": float(elapsed),
                    "events_scored": 0,
                    "throughput_events_per_second": 0.0,
                },
                "timing": {
                    "precompute_seconds": float(elapsed),
                },
                "memory": {
                    "precompute_rss_peak_mb": metrics["precompute_rss_peak_mb"],
                },
                "outputs": {
                    "precompute_metrics_json": str(metrics_path),
                    "coverage_audit_json": str(audit_path),
                    "event_index_meta_json": str(event_index_meta_path),
                    "x_context_meta_json": str(x_context_meta_path),
                    "node_embeddings_npy": node_paths["node_embeddings"],
                    "action_embeddings_npy": action_paths["action_embeddings"],
                },
                "phase3e": dict(metrics["phase3e"]),
                "leakage_check": {
                    "labels_used_for_precompute": False,
                    "event_index_contains_labels": False,
                    "event_index_contains_raw_rows": False,
                    "event_index_contains_timestamp_order_columns": False,
                },
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    model = SSPMLowRankModel(_make_sspm_config(config, process_cfg))
    write_effective_config(
        output_dir,
        config,
        model,
        embedder_loaded=True,
        train_count=int(train_count),
        validation_count=int(validation_count),
        test_count=int(test_count),
    )


def run_phase3e_precompute_from_split_rows(
    *,
    config: SlimConfig,
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    embedder: ResidualEmbedder,
    output_dir: str | Path,
    node_embedding_dir: str | Path,
    action_embedding_dir: str | Path,
    event_index_dir: str | Path,
    x_context_dir: str | Path,
    process_cfg: ProcessSemanticConfig | None,
    order_by: Sequence[object] | object,
    event_id_source: str,
) -> Path:
    """Run label-free Phase3E precompute from already-streamed split rows."""
    started = time.perf_counter()
    output_path = Path(output_dir)
    node_dir = Path(node_embedding_dir)
    action_dir = Path(action_embedding_dir)
    event_dir = Path(event_index_dir)
    x_dir = Path(x_context_dir)
    for path in (output_path, node_dir, action_dir, event_dir, x_dir):
        path.mkdir(parents=True, exist_ok=True)

    adapter = ResidualWord2VecTokenAdapter(
        embedder,
        model_path=str(config.pretrained_residual_embedder_path),
    )
    (
        node_tokens_by_id,
        split_node_ids,
        src_node_ids,
        dst_node_ids,
        split_counts,
    ) = _phase3e_collect_node_universe(split_rows, config, process_cfg)
    node_table = build_node_embedding_table(
        node_tokens_by_id=node_tokens_by_id,
        split_node_ids=split_node_ids,
        adapter=adapter,
        src_node_ids=src_node_ids,
        dst_node_ids=dst_node_ids,
    )
    action_table = build_action_embedding_table(adapter)
    node_paths = save_node_embedding_table(node_table, node_dir)
    action_paths = save_action_embedding_table(action_table, action_dir)
    audit = build_node_action_coverage_audit(node_table, action_table)
    audit_path = save_node_action_coverage_audit(audit, output_path)

    split_meta: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        split_meta[split] = _phase3e_write_event_index_split(
            rows=list(split_rows.get(split, ())),
            path=event_dir / f"event_index_{split}.memmap",
            split=split,
            node_id_to_idx=node_table.node_id_to_idx,
            action_name_to_id=action_table.name_to_action_id,
            config=config,
            process_cfg=process_cfg,
            order_by=order_by,
            event_id_source=event_id_source,
        )

    event_index_meta = {
        "row_order_is_stream_order": True,
        "order_by": _phase3e_json_order_by(order_by),
        "event_id_source": str(event_id_source),
        "target_mode": str(config.sspm_target_mode),
        "node_word2vec_source": str(config.node_word2vec_source),
        "event_index_dtype": _phase3e_event_index_dtype_descriptor(),
        "dtype_descriptor": _phase3e_event_index_dtype_descriptor(),
        "splits": split_meta,
        "fingerprint": stable_json_hash(split_meta),
    }
    event_index_meta_path = event_dir / "event_index_meta.json"
    event_index_meta_path.write_text(
        json.dumps(event_index_meta, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    x_context = _phase3e_build_x_contexts(
        config=config,
        train_event_index_path=str(split_meta["train"]["path"]),
        train_event_count=int(split_meta["train"]["num_events"]),
        node_table=node_table,
        action_table=action_table,
        x_context_dir=x_dir,
        event_index_meta=split_meta["train"],
        process_cfg=process_cfg,
    )
    x_context_meta_path = x_dir / "x_context_meta.json"
    x_context_meta_path.write_text(
        json.dumps(x_context, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    elapsed = max(time.perf_counter() - started, 1e-9)
    metrics = {
        "dataset": str(config.dataset),
        "out_tag": str(config.out_tag),
        "target_mode": str(config.sspm_target_mode),
        "node_word2vec_source": str(config.node_word2vec_source),
        "word2vec_model_path": str(config.pretrained_residual_embedder_path),
        "word2vec_fingerprint": str(node_table.meta.get("word2vec_fingerprint", "")),
        "node_embeddings_path": node_paths["node_embeddings"],
        "node_id_to_idx_path": node_paths["node_id_to_idx"],
        "node_embedding_meta_path": node_paths["node_embedding_meta"],
        "node_embedding_fingerprint": _phase3e_array_fingerprint(node_table.embeddings),
        "action_embeddings_path": action_paths["action_embeddings"],
        "action_id_to_name_path": action_paths["action_id_to_name"],
        "action_embedding_meta_path": action_paths["action_embedding_meta"],
        "action_embedding_fingerprint": _phase3e_array_fingerprint(action_table.embeddings),
        "coverage_audit_path": str(audit_path),
        "coverage": audit,
        "event_index_meta_path": str(event_index_meta_path),
        "event_index": event_index_meta,
        "x_context": x_context,
        "x_context_meta_path": str(x_context_meta_path),
        "train_events_actual": int(split_counts.get("train", 0)),
        "validation_events_actual": int(split_counts.get("validation", 0)),
        "test_events_actual": int(split_counts.get("test", 0)),
        "row_order_is_stream_order": True,
        "order_by": _phase3e_json_order_by(order_by),
        "event_id_source": str(event_id_source),
        "event_id_preserved": True,
        "timestamp_order_saved_per_row": False,
        "raw_rows_saved": False,
        "tokens_saved": False,
        "event_embedding_saved": False,
        "x_context_memmap_enabled": bool(config.x_context_memmap_enabled),
        "precompute_seconds": float(elapsed),
        "precompute_events_per_sec": float(sum(split_counts.values()) / elapsed),
        "precompute_rss_peak_mb": float(
            max(
                _current_rss_mb(),
                *[
                    float(meta.get("x_context_build_rss_peak_mb", 0.0))
                    for meta in x_context.values()
                ],
            ),
        ),
        "phase3e": {
            "target_mode": str(config.sspm_target_mode),
            "train_backend": str(config.sspm_train_backend),
            "infer_backend": str(config.sspm_infer_backend),
            "node_embedding_path": node_paths["node_embeddings"],
            "action_embedding_path": action_paths["action_embeddings"],
            "event_index_path": str(split_meta["train"]["path"]),
            "x_context_path": str(
                x_context.get("ema_fixed", {}).get("path", ""),
            ),
        },
    }
    metrics_path = output_path / "precompute_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    eval_path = output_path / "eval_causal_semantics_slim.json"
    eval_path.write_text(
        json.dumps(
            {
                "method_name": "causal_semantics_slim",
                "method_version": METHOD_VERSION,
                "phase3e_precompute_only": True,
                "config": asdict(config),
                "runtime": {
                    "elapsed_seconds": float(elapsed),
                    "events_scored": 0,
                    "throughput_events_per_second": 0.0,
                },
                "timing": {
                    "precompute_seconds": float(elapsed),
                },
                "memory": {
                    "precompute_rss_peak_mb": metrics["precompute_rss_peak_mb"],
                },
                "outputs": {
                    "precompute_metrics_json": str(metrics_path),
                    "coverage_audit_json": str(audit_path),
                    "event_index_meta_json": str(event_index_meta_path),
                    "x_context_meta_json": str(x_context_meta_path),
                    "node_embeddings_npy": node_paths["node_embeddings"],
                    "action_embeddings_npy": action_paths["action_embeddings"],
                },
                "phase3e": metrics["phase3e"],
                "leakage_check": {
                    "labels_used_for_precompute": False,
                    "event_index_contains_labels": False,
                    "event_index_contains_raw_rows": False,
                    "event_index_contains_timestamp_order_columns": False,
                },
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    model = SSPMLowRankModel(_make_sspm_config(config, process_cfg))
    write_effective_config(
        output_path,
        config,
        model,
        embedder_loaded=True,
        train_count=int(split_counts.get("train", 0)),
        validation_count=int(split_counts.get("validation", 0)),
        test_count=int(split_counts.get("test", 0)),
    )
    return metrics_path


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


def normalize_piece(value: object, max_len: int = 40) -> str:
    """Normalize a role/key fragment to a bounded lowercase token."""
    text = "" if value is None else str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_.:/|-]+", "_", text).strip("_")
    if not text:
        text = "unknown"
    limit = max(int(max_len), 1)
    return text[:limit]


def action_family(action: object) -> str:
    """Map raw event actions to a stable coarse action family."""
    text = normalize_piece(action, max_len=80).upper()
    if text in {"EVENT_READ"}:
        return "read"
    if text in {"EVENT_RECVFROM", "EVENT_RECVMSG", "EVENT_ACCEPT"}:
        return "recv"
    if text in {"EVENT_WRITE"}:
        return "write"
    if text in {"EVENT_SENDTO", "EVENT_SENDMSG", "EVENT_CONNECT"}:
        return "send"
    if text in {"EVENT_EXECUTE", "EVENT_CLONE"}:
        return "spawn"
    if text in {"EVENT_OPEN"}:
        return "open"
    if text in {"EVENT_FORK"}:
        return "fork"
    if text in {"EVENT_CLOSE"}:
        return "close"
    if text in {"EVENT_BOOT"}:
        return "boot"
    if text in {"EVENT_LSEEK"}:
        return "seek"
    if text in {"EVENT_CHANGE_PRINCIPAL"}:
        return "principal"
    if text in {"EVENT_MODIFY_PROCESS"}:
        return "modify"
    return "other"


RAW_ACTION_TOKENS = tuple(
    normalize_piece(action, max_len=80).lower()
    for action in sorted(ORTHRUS10_EVENT_TYPES)
)
RAW_ACTION_IDS = {token: index for index, token in enumerate(RAW_ACTION_TOKENS)}
RAW_ACTION_OTHER_ID = len(RAW_ACTION_IDS)


def raw_action_token(action: object) -> str:
    """Return the exact normalized event-action token used by slim residual semantics."""
    token = normalize_piece(action, max_len=80).lower()
    return token if token else "unknown"


def normalize_raw_action_token_for_sspm(action: object) -> str:
    """Return an uppercase raw action token for SSPM context/gates."""
    return str(action).strip().upper()


def raw_action_id(action: object) -> int:
    """Return a stable ID for a raw event action, with an explicit unknown bucket."""
    return int(RAW_ACTION_IDS.get(raw_action_token(action), RAW_ACTION_OTHER_ID))


def split_summary_tokens(summary: object, max_tokens: int) -> tuple[str, ...]:
    """Return normalized bounded summary tokens."""
    limit = max(int(max_tokens), 0)
    if limit <= 0 or summary is None:
        return ()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(str(summary).lower()):
        token = normalize_piece(match.group(0), max_len=40)
        if token != "unknown":
            tokens.append(token)
        if len(tokens) >= limit:
            break
    return tuple(tokens)


def role_head(
    kind: object,
    summary: object,
    max_tokens: int,
    dst_addr: object = "",
    dst_port: object = "",
) -> str:
    """Return a bounded role head, using coarse netflow scope and port for netflow NLL."""
    kind_token = normalize_piece(kind)
    if kind_token == "netflow":
        return netflow_nll_role(dst_addr, dst_port)
    tokens = split_summary_tokens(summary, max_tokens)
    if tokens:
        return "|".join((kind_token, *tokens))
    return kind_token


def _process_semantics_is_v1(config: ProcessSemanticConfig | None) -> bool:
    if config is None:
        return False
    return str(config.rules_version).strip().lower() in {
        "v1",
        "v2_coarse_nll",
        "v3_process_kind_nll",
    }


def _process_role_for_row(
    row: Mapping[str, Any],
    side: str,
    dataset: str,
    config: ProcessSemanticConfig | None,
    max_tokens_per_node: int,
) -> str:
    if not _process_semantics_is_v1(config):
        return role_head(
            "process",
            row.get(f"{side}_summary", row.get("process_name", "")),
            max_tokens_per_node,
        )
    assert config is not None
    result = normalize_process_semantics(
        dataset=dataset,
        path=row.get(f"{side}_process_path", ""),
        cmd=row.get(f"{side}_process_cmd", ""),
        config=config,
        max_tokens_per_node=max_tokens_per_node,
    )
    return result.nll_role


def _role_for_row_side(
    row: Mapping[str, Any],
    side: str,
    kind: str,
    dataset: str,
    process_semantic_config: ProcessSemanticConfig | None,
    max_tokens_per_node: int,
    summary_default: object = "",
    theia_netflow_policy: str = "scope_port",
) -> str:
    if kind == "process":
        return _process_role_for_row(
            row,
            side,
            dataset,
            process_semantic_config,
            max_tokens_per_node,
        )
    if is_clearscope_dataset(dataset) and kind == "netflow":
        return clearscope_netflow_nll_role()
    if is_clearscope_dataset(dataset) and kind == "file":
        return clearscope_file_nll_role(row.get(f"{side}_file_path", ""))
    if is_theia_dataset(dataset) and kind == "netflow":
        if str(theia_netflow_policy) == "fixed":
            return linux_netflow_fixed_nll_role()
        return linux_netflow_nll_role(
            row.get(f"{side}_addr", row.get("dst_addr", row.get("remote_ip", ""))),
            row.get(f"{side}_port", row.get("dst_port", row.get("remote_port", ""))),
        )
    if is_theia_dataset(dataset) and kind == "file":
        return linux_file_nll_role(row.get(f"{side}_file_path", ""))
    if is_cadets_dataset(dataset) and kind == "netflow":
        return freebsd_netflow_nll_role(
            row.get(f"{side}_addr", row.get("dst_addr", row.get("remote_ip", ""))),
            row.get(f"{side}_port", row.get("dst_port", row.get("remote_port", ""))),
        )
    if is_cadets_dataset(dataset) and kind == "file":
        return freebsd_file_nll_role()
    token_limit = max_tokens_per_node
    if kind == "file" and _process_semantics_is_v1(process_semantic_config):
        assert process_semantic_config is not None
        token_limit = process_semantic_config.file_nll_max_tokens
    return role_head(
        kind,
        row.get(f"{side}_summary", summary_default),
        token_limit,
        dst_addr=row.get("dst_addr", row.get("remote_ip", "")),
        dst_port=row.get("dst_port", row.get("remote_port", "")),
    )


def row_fields(
    row: Mapping[str, Any],
    max_tokens_per_node: int,
    dataset: str = "SYNTHETIC",
    process_semantic_config: ProcessSemanticConfig | None = None,
    theia_netflow_policy: str = "scope_port",
) -> dict[str, Any]:
    """Build the slim causal semantic fields for one event row."""
    action = row.get("action", "")
    object_type = row.get("object_type", "unknown")
    src_idx = int(row.get("src_idx", row.get("info_src", -1)))
    dst_idx = int(row.get("dst_idx", row.get("info_dst", -1)))
    info_src, info_dst, relation_id, info_src_type, info_dst_type = information_flow(
        src_idx,
        dst_idx,
        action,
        object_type,
    )
    relation_name = _relation_name(relation_id)
    object_kind = normalize_piece(object_type)
    src_kind = normalize_piece(row.get("src_kind", "process"))
    dst_kind = normalize_piece(row.get("dst_kind", object_kind))
    surface_src_type_name = "process" if src_kind == "unknown" else src_kind
    surface_dst_type_name = (
        object_kind
        if object_kind != "unknown"
        else ("unknown" if dst_kind == "unknown" else dst_kind)
    )
    src_type_name = ENTITY_TYPE_NAMES.get(int(info_src_type), "unknown")
    dst_type_name = ENTITY_TYPE_NAMES.get(int(info_dst_type), "unknown")
    text = row.get("text", "")
    src_role = _role_for_row_side(
        row,
        "src",
        src_kind,
        dataset,
        process_semantic_config,
        max_tokens_per_node,
        summary_default=row.get("process_name", row.get("src_name", "")),
        theia_netflow_policy=theia_netflow_policy,
    )
    dst_role = _role_for_row_side(
        row,
        "dst",
        (
            "process"
            if dst_kind == "process"
            else (object_kind if object_kind != "unknown" else dst_kind)
        ),
        dataset,
        process_semantic_config,
        max_tokens_per_node,
        summary_default=text,
        theia_netflow_policy=theia_netflow_policy,
    )
    action_key = raw_action_token(action)
    raw_action = normalize_raw_action_token_for_sspm(action)
    return {
        "action": raw_action,
        "raw_action": raw_action,
        "action_token": action_key,
        "action_id": raw_action_id(action),
        "object_type": object_kind,
        "src_role": src_role,
        "dst_role": dst_role,
        "src_type_name": src_type_name,
        "dst_type_name": dst_type_name,
        "surface_src_type_name": surface_src_type_name,
        "surface_dst_type_name": surface_dst_type_name,
        "relation": relation_name,
        "info_src": int(info_src),
        "info_dst": int(info_dst),
        "relation_id": int(relation_id),
        "info_src_type": int(info_src_type),
        "info_dst_type": int(info_dst_type),
        "action_given_src": (src_role, action_key),
        "dst_given_src_action": (src_role, action_key, dst_role),
        "relation_given_roles": (src_role, dst_role, relation_name),
    }


def _process_residual_text_for_row(
    row: Mapping[str, Any],
    side: str,
    dataset: str,
    process_semantic_config: ProcessSemanticConfig,
    max_tokens_per_node: int,
) -> str:
    return normalize_process_semantics(
        dataset=dataset,
        path=row.get(f"{side}_process_path", ""),
        cmd=row.get(f"{side}_process_cmd", ""),
        config=process_semantic_config,
        max_tokens_per_node=max_tokens_per_node,
    ).residual_text


def _clearscope_natural_node_tokens(row: Mapping[str, Any], side: str) -> tuple[str, ...]:
    kind = normalize_piece(row.get(f"{side}_kind", ""))
    if kind == "process":
        return android_process_natural_tokens(row.get(f"{side}_process_cmd", ""))
    if kind == "file":
        return clearscope_file_natural_tokens(row.get(f"{side}_file_path", ""))
    if kind == "netflow":
        return clearscope_netflow_natural_tokens()
    return (kind or "unknown",)


def _clearscope_natural_residual_text(row: Mapping[str, Any]) -> str:
    action = raw_action_token(row.get("action", "unknown"))
    tokens = [
        *_clearscope_natural_node_tokens(row, "src"),
        action,
        *_clearscope_natural_node_tokens(row, "dst"),
    ]
    return " ".join(str(token) for token in tokens if str(token).strip())


def _theia_natural_node_tokens(
    row: Mapping[str, Any],
    side: str,
    theia_netflow_policy: str = "scope_port",
) -> tuple[str, ...]:
    kind = normalize_piece(row.get(f"{side}_kind", ""))
    if kind == "process":
        return linux_process_natural_tokens(
            row.get(f"{side}_process_path", ""),
            row.get(f"{side}_process_cmd", ""),
        )
    if kind == "file":
        return linux_file_natural_tokens(row.get(f"{side}_file_path", ""))
    if kind == "netflow":
        if str(theia_netflow_policy) == "fixed":
            return linux_netflow_detail_or_fixed_natural_tokens(
                row.get(f"{side}_addr", row.get("dst_addr", row.get("remote_ip", ""))),
                row.get("src_addr", ""),
            )
        return linux_netflow_natural_tokens(
            row.get(f"{side}_addr", row.get("dst_addr", row.get("remote_ip", ""))),
            row.get("src_addr", ""),
        )
    return (kind or "unknown",)


def _theia_natural_residual_text(
    row: Mapping[str, Any],
    theia_netflow_policy: str = "scope_port",
) -> str:
    action = raw_action_token(row.get("action", "unknown"))
    tokens = [
        *_theia_natural_node_tokens(row, "src", theia_netflow_policy),
        action,
        *_theia_natural_node_tokens(row, "dst", theia_netflow_policy),
    ]
    return " ".join(str(token) for token in tokens if str(token).strip())


def _cadets_natural_node_tokens(row: Mapping[str, Any], side: str) -> tuple[str, ...]:
    kind = normalize_piece(row.get(f"{side}_kind", ""))
    if kind == "process":
        return freebsd_process_natural_tokens(row.get(f"{side}_process_cmd", ""))
    if kind == "file":
        return freebsd_file_natural_tokens(row.get(f"{side}_file_path", ""))
    if kind == "netflow":
        return freebsd_netflow_natural_tokens(
            row.get(f"{side}_addr", row.get("dst_addr", row.get("remote_ip", ""))),
            row.get("src_addr", ""),
        )
    return (kind or "unknown",)


def _cadets_natural_residual_text(row: Mapping[str, Any]) -> str:
    action = raw_action_token(row.get("action", "unknown"))
    tokens = [
        *_cadets_natural_node_tokens(row, "src"),
        action,
        *_cadets_natural_node_tokens(row, "dst"),
    ]
    return " ".join(str(token) for token in tokens if str(token).strip())


def _observe_process_semantic_audit(
    audit: ProcessSemanticAudit | None,
    split: str,
    row: Mapping[str, Any],
    dataset: str,
    process_cfg: ProcessSemanticConfig,
    max_tokens_per_node: int,
) -> None:
    if audit is None:
        return
    object_kind = normalize_piece(row.get("object_type", "unknown"))
    side_defaults = {"src": "process", "dst": object_kind}
    for side in ("src", "dst"):
        kind = normalize_piece(row.get(f"{side}_kind", side_defaults[side]))
        if kind != "process":
            continue
        result = normalize_process_semantics(
            dataset=dataset,
            path=row.get(f"{side}_process_path", ""),
            cmd=row.get(f"{side}_process_cmd", ""),
            config=process_cfg,
            max_tokens_per_node=max_tokens_per_node,
        )
        audit.observe(
            split=split,
            event_index=int(row.get("event_index", -1)),
            node_id=int(row.get(f"{side}_idx", -1)),
            path=row.get(f"{side}_process_path", ""),
            cmd=row.get(f"{side}_process_cmd", ""),
            semantic=result,
        )


def residual_text(
    row: Mapping[str, Any],
    dataset: str = "SYNTHETIC",
    process_semantic_config: ProcessSemanticConfig | None = None,
    max_tokens_per_node: int = 8,
    theia_netflow_policy: str = "scope_port",
    semantic_mode: str = CLEARSCOPE_REFINED_SEMANTIC_MODE,
) -> str:
    """Return residual text, using exact IP only for netflow rows."""
    if is_clearscope_dataset(dataset):
        if (
            normalize_clearscope_semantic_mode(semantic_mode)
            == CLEARSCOPE_LEGACY_SEMANTIC_MODE
        ):
            return _clearscope_natural_residual_text(row)
        return clearscope_residual_text_refined(dict(row))
    if is_theia_dataset(dataset):
        return _theia_natural_residual_text(row, theia_netflow_policy)
    if is_cadets_dataset(dataset):
        return _cadets_natural_residual_text(row)
    object_kind = normalize_piece(row.get("object_type", "unknown"))
    src_kind = normalize_piece(row.get("src_kind", "process"))
    dst_kind = normalize_piece(row.get("dst_kind", object_kind))
    parts = [
        f"action:{normalize_piece(row.get('action', 'unknown'))}",
        f"object:{object_kind}",
    ]
    if src_kind == "process" and _process_semantics_is_v1(process_semantic_config):
        assert process_semantic_config is not None
        parts.append(
            _process_residual_text_for_row(
                row,
                "src",
                dataset,
                process_semantic_config,
                max_tokens_per_node,
            ),
        )
    else:
        parts.extend(
            [
                f"a_{src_kind}",
                str(row.get("src_summary", row.get("process_name", ""))),
            ],
        )
    if object_kind == "netflow":
        netflow_text = (
            clearscope_netflow_residual_text()
            if is_clearscope_dataset(dataset)
            else _netflow_residual_text(row.get("dst_addr", row.get("remote_ip", "")))
        )
        parts.extend(
            [
                "o_netflow",
                netflow_text,
            ],
        )
        return " ".join(part for part in parts if str(part).strip())
    if is_clearscope_dataset(dataset) and (object_kind == "file" or dst_kind == "file"):
        parts.append(clearscope_file_residual_text(row.get("dst_file_path", "")))
        return " ".join(str(part) for part in parts if str(part).strip())
    if (
        (object_kind == "process" or dst_kind == "process")
        and _process_semantics_is_v1(process_semantic_config)
    ):
        assert process_semantic_config is not None
        parts.append(
            _process_residual_text_for_row(
                row,
                "dst",
                dataset,
                process_semantic_config,
                max_tokens_per_node,
            ),
        )
        return " ".join(str(part) for part in parts if str(part).strip())
    parts.append(str(row.get("dst_summary", row.get("dst_name", row.get("text", "")))))
    return " ".join(str(part) for part in parts if str(part).strip())


def empirical_tail_score(value: float, sorted_scores: Sequence[float]) -> float:
    """Return empirical validation-tail surprisal using (ge_count + 1) smoothing."""
    scores = _checked_sorted_scores(sorted_scores)
    val = float(value)
    if not math.isfinite(val):
        raise ValueError("value must be finite")
    first_ge = bisect.bisect_left(scores, val)
    ge_count = len(scores) - first_ge
    probability = (ge_count + 1.0) / (len(scores) + 1.0)
    return float(-math.log(probability))


def _empirical_tail_score_compact(value: float, sorted_scores: np.ndarray) -> float:
    val = float(value)
    if not math.isfinite(val):
        raise ValueError("value must be finite")
    count = int(sorted_scores.size)
    if count <= 0:
        raise ValueError("validation scores must not be empty")
    first_ge = int(np.searchsorted(sorted_scores, val, side="left"))
    ge_count = count - first_ge
    probability = (ge_count + 1.0) / (count + 1.0)
    return float(-math.log(probability))


def event_threshold_from_budget(
    sorted_scores: Sequence[float],
    expected_budget: float,
    horizon_events: int,
) -> float:
    """Return a validation empirical-tail threshold for an expected alert budget."""
    scores = _checked_sorted_scores(sorted_scores)
    horizon = int(horizon_events)
    if horizon <= 0:
        raise ValueError("horizon_events must be positive")
    budget = float(expected_budget)
    if not math.isfinite(budget) or budget <= 0.0:
        return float("inf")
    target_rate = min(max(budget / float(horizon), 0.0), 1.0)
    if target_rate >= 1.0:
        return float(scores[0])
    quantile = 1.0 - target_rate
    return float(np.quantile(np.asarray(scores, dtype=np.float64), quantile, method="higher"))


def event_threshold_from_quantile(
    sorted_scores: Sequence[float],
    quantile: float,
) -> float:
    """Return a validation threshold at a fixed empirical quantile."""
    scores = _checked_sorted_scores(sorted_scores)
    q = float(quantile)
    if not math.isfinite(q):
        raise ValueError("event threshold quantile must be finite")
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(np.asarray(scores, dtype=np.float64), q, method="higher"))


def event_threshold_from_mode(
    sorted_scores: Sequence[float],
    expected_budget: float,
    horizon_events: int,
    threshold_mode: str,
    fixed_quantile: float,
) -> float:
    """Return the validation-only event threshold for the requested ablation mode."""
    mode = str(threshold_mode or "budget").strip().lower()
    if mode in {"budget", "adaptive_rate"}:
        return event_threshold_from_budget(sorted_scores, expected_budget, horizon_events)
    if mode == "quantile":
        return event_threshold_from_quantile(sorted_scores, fixed_quantile)
    if mode in {"max", "validation_max"}:
        scores = _checked_sorted_scores(sorted_scores)
        return float(scores[-1])
    raise ValueError(f"unknown event_threshold_mode: {threshold_mode}")


def derive_event_calibration(
    validation_residual_scores: Sequence[float],
    event_score_mode: str,
    expected_budget: float,
    horizon_events: int,
    residual_sorted: Sequence[float] | None = None,
    threshold_mode: str = "budget",
    fixed_quantile: float = 0.999,
) -> EventCalibration:
    """Derive residual-only event calibration from raw validation residual scores."""
    _validate_active_event_score_mode(event_score_mode)
    residual_raw = np.asarray(validation_residual_scores, dtype=np.float32)
    if residual_raw.size == 0:
        raise ValueError("validation residual score array must not be empty")
    res_sorted = (
        _compact_sorted_scores(residual_sorted)
        if residual_sorted is not None
        else _compact_sorted_scores(residual_raw)
    )
    event_values = np.empty((int(residual_raw.size),), dtype=np.float32)
    for index in range(int(residual_raw.size)):
        residual_tail = _empirical_tail_score_compact(float(residual_raw[index]), res_sorted)
        event_values[index] = np.float32(
            _event_score_from_residual_tail(residual_tail, event_score_mode),
        )
    event_sorted = _compact_sorted_scores(event_values)
    threshold = event_threshold_from_mode(
        event_sorted,
        expected_budget,
        horizon_events,
        threshold_mode,
        fixed_quantile,
    )
    summary: dict[str, float | int] = {
        "count": int(event_values.size),
        "min": float(np.min(event_values)),
        "max": float(np.max(event_values)),
        "mean": float(np.mean(event_values)),
        "std": float(np.std(event_values)),
        "threshold": float(threshold),
        "final_threshold": float(threshold),
        "validation_max_score": float(event_sorted[-1]),
        "threshold_mode": str(threshold_mode),
        "threshold_quantile": float(fixed_quantile),
    }
    return EventCalibration(
        res_sorted,
        event_values,
        event_sorted,
        float(threshold),
        summary,
    )


def _score_distribution_summary(
    values: Sequence[float] | np.ndarray,
    threshold: float,
    prefix: str = "score",
) -> dict[str, float | int]:
    """Return finite score summary fields for validation/test diagnostics."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p99": 0.0,
            f"{prefix}_p999": 0.0,
            f"{prefix}_p9995": 0.0,
            f"{prefix}_p9999": 0.0,
            f"{prefix}_max": 0.0,
            "above_threshold_count": 0,
        }
    return {
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_p99": float(np.quantile(finite, 0.99, method="higher")),
        f"{prefix}_p999": float(np.quantile(finite, 0.999, method="higher")),
        f"{prefix}_p9995": float(np.quantile(finite, 0.9995, method="higher")),
        f"{prefix}_p9999": float(np.quantile(finite, 0.9999, method="higher")),
        f"{prefix}_max": float(np.max(finite)),
        "above_threshold_count": int(np.count_nonzero(finite >= float(threshold))),
    }


def _score_summary_from_event_calibration(
    calibration: EventCalibration,
) -> dict[str, float | int | str]:
    """Return validation score summary using the selected event-score distribution."""
    summary = _score_distribution_summary(
        calibration.event_scores,
        float(calibration.threshold),
    )
    summary.update(
        {
            "count": int(calibration.event_scores.size),
            "final_threshold": float(calibration.threshold),
            "threshold_mode": str(calibration.summary.get("threshold_mode", "")),
            "threshold_quantile": float(calibration.summary.get("threshold_quantile", 0.0)),
            "validation_max_score": float(calibration.summary.get("validation_max_score", 0.0)),
        },
    )
    return summary


def _phase3e_test_score_summary(
    event_scores: Sequence[float],
    threshold: float,
    event_alert_count: int | None = None,
) -> dict[str, float | int]:
    """Return Phase3E test event-score diagnostics."""
    summary = _score_distribution_summary(event_scores, threshold)
    summary["count"] = int(len(event_scores))
    summary["final_threshold"] = float(threshold)
    summary["test_above_threshold_count"] = int(summary.pop("above_threshold_count"))
    if event_alert_count is not None:
        summary["event_alert_count"] = int(event_alert_count)
    return summary


class _StreamingScoreSummary:
    """Bounded-memory score summary for long online test streams."""

    def __init__(self, threshold: float, bins: int = 20000) -> None:
        self.threshold = float(threshold)
        self.bins = max(int(bins), 100)
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_score = float("inf")
        self.max_score = float("-inf")
        self.above_threshold_count = 0
        self._hist_counts = np.zeros((self.bins,), dtype=np.int64)
        self._hist_min = 0.0
        self._hist_max = 1.0
        self._hist_overflow_rebuilds = 0

    def observe(self, value: float) -> None:
        score = float(value)
        if not math.isfinite(score):
            return
        self.count += 1
        self.sum += score
        self.sum_sq += score * score
        self.min_score = min(self.min_score, score)
        self.max_score = max(self.max_score, score)
        if score >= self.threshold:
            self.above_threshold_count += 1
        self._observe_histogram(score)

    def _observe_histogram(self, score: float) -> None:
        if score < self._hist_min or score > self._hist_max:
            self._expand_histogram(score)
        if self._hist_max <= self._hist_min:
            index = 0
        else:
            relative = (score - self._hist_min) / max(self._hist_max - self._hist_min, 1e-12)
            index = int(relative * (self.bins - 1))
        index = min(max(index, 0), self.bins - 1)
        self._hist_counts[index] += 1

    def _expand_histogram(self, score: float) -> None:
        old_counts = self._hist_counts
        old_min = float(self._hist_min)
        old_max = float(self._hist_max)
        new_min = min(old_min, float(score), 0.0)
        new_max = max(old_max, float(score), 1.0)
        if new_max <= new_min:
            new_max = new_min + 1.0
        rebuilt = np.zeros_like(old_counts)
        old_total = int(old_counts.sum())
        if old_total > 0:
            old_width = (old_max - old_min) / float(self.bins)
            centers = old_min + (np.arange(self.bins, dtype=np.float64) + 0.5) * old_width
            relative = (centers - new_min) / max(new_max - new_min, 1e-12)
            indices = np.floor(relative * self.bins).astype(np.int64)
            indices = np.clip(indices, 0, self.bins - 1)
            np.add.at(rebuilt, indices, old_counts)
        self._hist_counts = rebuilt
        self._hist_min = float(new_min)
        self._hist_max = float(new_max)
        self._hist_overflow_rebuilds += 1

    def _hist_quantile(self, quantile: float) -> float:
        if self.count <= 0:
            return 0.0
        q = min(max(float(quantile), 0.0), 1.0)
        rank = max(int(math.ceil(q * self.count)), 1)
        cumulative = np.cumsum(self._hist_counts)
        index = int(np.searchsorted(cumulative, rank, side="left"))
        index = min(max(index, 0), self.bins - 1)
        width = (self._hist_max - self._hist_min) / float(self.bins)
        estimate = self._hist_min + (float(index) + 1.0) * width
        return float(min(max(estimate, self.min_score), self.max_score))

    def summary(self, event_alert_count: int | None = None) -> dict[str, float | int]:
        if self.count <= 0:
            payload = _phase3e_test_score_summary([], self.threshold, event_alert_count)
            payload["test_score_quantile_method"] = "streaming_histogram"
            payload["score_quantile_method"] = "streaming_histogram"
            return payload
        mean = float(self.sum / self.count)
        variance = max(float(self.sum_sq / self.count) - mean * mean, 0.0)
        payload: dict[str, float | int] = {
            "score_min": float(self.min_score),
            "score_mean": mean,
            "score_std": float(math.sqrt(variance)),
            "score_p99": self._hist_quantile(0.99),
            "score_p999": self._hist_quantile(0.999),
            "score_p9995": self._hist_quantile(0.9995),
            "score_p9999": self._hist_quantile(0.9999),
            "score_max": float(self.max_score),
            "count": int(self.count),
            "final_threshold": float(self.threshold),
            "test_above_threshold_count": int(self.above_threshold_count),
            "test_score_quantile_method": "streaming_histogram",
            "score_quantile_method": "streaming_histogram",
            "score_histogram_bins": int(self.bins),
            "score_histogram_rebuilds": int(self._hist_overflow_rebuilds),
        }
        if event_alert_count is not None:
            payload["event_alert_count"] = int(event_alert_count)
        return payload


class _TargetCaseStreamingSummary:
    """Streaming score summary split by conditional target case."""

    def __init__(self, bins: int = 20000) -> None:
        self._summaries: dict[str, _StreamingScoreSummary] = {
            EVENT_SEMANTIC_TARGET: _StreamingScoreSummary(0.0, bins=bins),
            BOTH_COLD_ACTION_TARGET: _StreamingScoreSummary(0.0, bins=bins),
        }
        self._bins = int(bins)

    def observe(self, case_name: str, score: float) -> None:
        key = str(case_name)
        if key not in self._summaries:
            self._summaries[key] = _StreamingScoreSummary(0.0, bins=self._bins)
        self._summaries[key].observe(float(score))

    def summary(self, thresholds: Mapping[str, float]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for case_name, summary in self._summaries.items():
            threshold = float(thresholds.get(case_name, 0.0))
            summary.threshold = threshold
            payload[case_name] = summary.summary()
            payload[case_name]["target_case"] = case_name
        return payload


class _ConditionalGroupStreamingSummary:
    """Collect compact test score and alert counts by conditional level-1 group."""

    def __init__(self, bins: int = 2000) -> None:
        self._summaries: dict[int, _StreamingScoreSummary] = {}
        self.alert_counts: dict[int, int] = {}
        self._bins = int(bins)

    def observe(
        self,
        target_case_id_value: int,
        action_id: int,
        src_type_id: int,
        dst_type_id: int,
        score: float,
        *,
        alert: bool,
    ) -> None:
        key = int(
            encode_conditional_group_key(
                int(target_case_id_value),
                int(action_id),
                int(src_type_id),
                int(dst_type_id),
            ),
        )
        self._summaries.setdefault(key, _StreamingScoreSummary(0.0, bins=self._bins)).observe(
            float(score),
        )
        if bool(alert):
            self.alert_counts[key] = int(self.alert_counts.get(key, 0)) + 1

    def rows(self, cache_meta: Mapping[str, Any]) -> list[dict[str, Any]]:
        group_meta = dict(cache_meta.get("group_thresholds") or {})
        level1 = dict(
            dict(group_meta.get("thresholds", {})).get(GROUP_LEVEL_TARGET_ACTION_SRC_DST, {}),
        )
        all_keys = set(self._summaries)
        for record in level1.values():
            key = _conditional_group_key_from_threshold_key(
                str(dict(record).get("threshold_group_key", "")),
            )
            if key is not None:
                all_keys.add(int(key))
        rows: list[dict[str, Any]] = []
        for encoded_key in sorted(all_keys):
            decoded = decode_conditional_group_key(int(encoded_key))
            target_case = TARGET_CASE_ID_TO_NAME.get(
                int(decoded["target_case_id"]),
                f"case_{int(decoded['target_case_id'])}",
            )
            action_id = int(decoded["action_id"])
            src_type_id = int(decoded["src_type_id"])
            dst_type_id = int(decoded["dst_type_id"])
            resolved = resolve_conditional_group_threshold(
                group_meta,
                int(decoded["target_case_id"]),
                action_id,
                src_type_id,
                dst_type_id,
            )
            threshold = float(resolved.get("threshold", cache_meta.get("threshold", 0.0)))
            validation_record = _conditional_validation_record_for_level1(
                group_meta,
                int(decoded["target_case_id"]),
                action_id,
                src_type_id,
                dst_type_id,
            )
            val_summary = dict(validation_record.get("summary", {}))
            summary = self._summaries.get(int(encoded_key))
            if summary is None:
                test_summary = conditional_score_summary(np.asarray([], dtype=np.float32), threshold)
                test_summary["score_quantile_method"] = "empty"
            else:
                summary.threshold = threshold
                test_summary = summary.summary()
            rows.append(
                {
                    "target_case": target_case,
                    "action_id": action_id,
                    "action_name": _conditional_action_name(action_id),
                    "src_type_id": src_type_id,
                    "src_type_name": _phase3e_entity_type_name("src_type_id", src_type_id),
                    "dst_type_id": dst_type_id,
                    "dst_type_name": _phase3e_entity_type_name("dst_type_id", dst_type_id),
                    "validation_count": int(validation_record.get("count", 0)),
                    "test_count": int(test_summary.get("count", 0)),
                    "threshold": threshold,
                    "threshold_level": str(resolved.get("threshold_level", "")),
                    "low_support_policy": str(resolved.get("low_support_policy", "")),
                    "group_validation_max": float(resolved.get("group_validation_max", 0.0)),
                    "parent_threshold": float(resolved.get("parent_threshold", 0.0)),
                    "global_threshold": float(resolved.get("global_threshold", 0.0)),
                    "final_threshold_source": str(
                        resolved.get("final_threshold_source", ""),
                    ),
                    "adaptive_margin_used": float(resolved.get("adaptive_margin_used", 0.0)),
                    "validation_count_bucket": str(resolved.get("validation_count_bucket", "")),
                    "val_p99": float(val_summary.get("score_p99", 0.0)),
                    "val_p999": float(val_summary.get("score_p999", 0.0)),
                    "val_p9995": float(val_summary.get("score_p9995", 0.0)),
                    "val_p9999": float(val_summary.get("score_p9999", 0.0)),
                    "val_max": float(val_summary.get("score_max", 0.0)),
                    "test_p99": float(test_summary.get("score_p99", 0.0)),
                    "test_p999": float(test_summary.get("score_p999", 0.0)),
                    "test_p9995": float(test_summary.get("score_p9995", 0.0)),
                    "test_p9999": float(test_summary.get("score_p9999", 0.0)),
                    "test_max": float(test_summary.get("score_max", 0.0)),
                    "score_quantile_method": str(
                        test_summary.get("score_quantile_method", "streaming_histogram"),
                    ),
                    "alert_count": int(self.alert_counts.get(int(encoded_key), 0)),
                    "tp": 0,
                    "fp": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                },
            )
        return rows


def _conditional_action_name(action_id: int) -> str:
    if 0 <= int(action_id) < len(ORTHRUS10_ACTION_NAMES):
        return str(ORTHRUS10_ACTION_NAMES[int(action_id)])
    return ""


def _conditional_group_key_from_threshold_key(value: str) -> int | None:
    parts: dict[str, str] = {}
    for item in str(value).split("|"):
        if "=" not in item:
            continue
        left, right = item.split("=", 1)
        parts[left] = right
    try:
        case_name = str(parts["target_case"])
        case_id = TARGET_CASE_NAME_TO_ID[case_name]
        return int(
            encode_conditional_group_key(
                int(case_id),
                int(parts["action_id"]),
                int(parts["src_type_id"]),
                int(parts["dst_type_id"]),
            ),
        )
    except (KeyError, ValueError):
        return None


def _conditional_validation_record_for_level1(
    group_meta: Mapping[str, Any],
    target_case_id_value: int,
    action_id: int,
    src_type_id: int,
    dst_type_id: int,
) -> dict[str, Any]:
    key = make_conditional_group_key(
        GROUP_LEVEL_TARGET_ACTION_SRC_DST,
        int(target_case_id_value),
        int(action_id),
        int(src_type_id),
        int(dst_type_id),
    )
    return dict(
        dict(dict(group_meta.get("thresholds", {})).get(GROUP_LEVEL_TARGET_ACTION_SRC_DST, {}))
        .get(key, {})
    )


def _write_conditional_group_test_summary(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = Path(output_dir) / "conditional_score_summary_by_target_action_type.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONDITIONAL_GROUP_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CONDITIONAL_GROUP_SUMMARY_FIELDS})
    return path


def _conditional_group_summary_with_eval(
    rows: Sequence[Mapping[str, Any]],
    evaluated_alert_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return conditional group summary rows with post-stream TP/FP fields populated."""
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {
        (
            str(row.get("target_case", "")),
            str(row.get("action_name", "")),
            str(row.get("src_type_name", "")),
            str(row.get("dst_type_name", "")),
        ): dict(row)
        for row in rows
    }
    tp_by_key: dict[tuple[str, str, str, str], int] = {}
    fp_by_key: dict[tuple[str, str, str, str], int] = {}
    total_tp = 0
    for alert in evaluated_alert_rows:
        key = (
            str(alert.get("target_case", "")),
            str(alert.get("action", "")),
            str(alert.get("src_type", "")),
            str(alert.get("dst_type", "")),
        )
        correct = bool(_parse_bool(alert.get("is_correct_event_alert", False)))
        if correct:
            tp_by_key[key] = int(tp_by_key.get(key, 0)) + 1
            total_tp += 1
        else:
            fp_by_key[key] = int(fp_by_key.get(key, 0)) + 1
    for key, row in by_key.items():
        tp = int(tp_by_key.get(key, 0))
        fp = int(fp_by_key.get(key, 0))
        denom = tp + fp
        row["tp"] = tp
        row["fp"] = fp
        row["precision"] = float(tp / denom) if denom > 0 else 0.0
        row["recall"] = float(tp / max(total_tp, 1)) if total_tp > 0 else 0.0
    return [by_key[key] for key in sorted(by_key)]


def _phase3g_conditional_post_stream_group_eval(
    *,
    config: SlimConfig,
    paths: Mapping[str, Path],
    test_index: np.ndarray,
    stream_outputs: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Populate conditional group TP/FP after streaming without changing online alerts."""
    group_summary_csv = str(stream_outputs.get("conditional_group_summary_csv", "")).strip()
    if not group_summary_csv:
        return {}
    group_path = Path(group_summary_csv)
    alert_path = Path(output_dir) / "online_event_alerts.csv"
    if not group_path.exists() or not alert_path.exists():
        return {}
    abnormal_nodes = _phase3g_load_abnormal_db_nodes_for_eval(config)
    if not abnormal_nodes:
        return {
            "conditional_group_summary_csv": str(group_path),
            "event_alerts": _event_metrics_from_counts(0, 0),
            "event_labels_by_event": {},
        }
    idx_to_db_node_id = _phase3e_load_idx_to_node_id(paths)
    labels_by_event = _phase3e_event_labels_for_alerts(
        records=test_index,
        alert_event_indices=_alert_event_indices(stream_outputs),
        abnormal_db_node_ids=abnormal_nodes,
        idx_to_db_node_id=idx_to_db_node_id,
    )
    evaluated_alert_rows = [
        _evaluated_event_row(row, labels_by_event)
        for row in _iter_csv_rows(alert_path)
    ]
    group_rows = list(_iter_csv_rows(group_path))
    if group_rows:
        _write_conditional_group_test_summary(
            output_dir,
            _conditional_group_summary_with_eval(group_rows, evaluated_alert_rows),
        )
    tp = sum(1 for row in evaluated_alert_rows if bool(_parse_bool(row["is_correct_event_alert"])))
    fp = max(len(evaluated_alert_rows) - tp, 0)
    return {
        "event_alerts": _event_metrics_from_counts(tp, fp),
        "event_labels_by_event": labels_by_event,
        "conditional_group_summary_csv": str(group_path),
    }


def _phase3g_csv_int(row: Mapping[str, Any], key: str) -> int:
    try:
        return int(float(str(row.get(key, 0) or 0)))
    except (TypeError, ValueError):
        return 0


def _phase3g_write_required_dual_head_summary_csvs(
    *,
    output_dir: Path,
    stream_outputs: Mapping[str, Any],
    group_summary_csv: str,
) -> dict[str, str]:
    """Write required target-case and group alert summary CSVs for Phase3G reports."""
    output_path = Path(output_dir)
    target_case_path = output_path / "target_case_summary.csv"
    group_alert_path = output_path / "group_alert_summary.csv"
    score_by_case = dict(stream_outputs.get("test_score_summary_by_target_case", {}))
    grouped_by_case: dict[str, dict[str, int]] = {}
    group_rows_out: list[dict[str, Any]] = []
    group_path = Path(str(group_summary_csv)) if str(group_summary_csv).strip() else None
    if group_path is not None and group_path.exists():
        for row in _iter_csv_rows(group_path):
            case = str(row.get("target_case", ""))
            tp = _phase3g_csv_int(row, "tp")
            fp = _phase3g_csv_int(row, "fp")
            alert_count = _phase3g_csv_int(row, "alert_count")
            event_count = _phase3g_csv_int(row, "test_count")
            case_counts = grouped_by_case.setdefault(
                case,
                {"event_count": 0, "alert_count": 0, "TP": 0, "FP": 0},
            )
            case_counts["event_count"] += int(event_count)
            case_counts["alert_count"] += int(alert_count)
            case_counts["TP"] += int(tp)
            case_counts["FP"] += int(fp)
            precision = float(tp / max(tp + fp, 1)) if tp + fp > 0 else 0.0
            group_rows_out.append(
                {
                    "action": row.get("action_name", ""),
                    "src_type": row.get("src_type_name", ""),
                    "dst_type": row.get("dst_type_name", ""),
                    "target_case": case,
                    "event_count": event_count,
                    "alert_count": alert_count,
                    "TP": tp,
                    "FP": fp,
                    "precision": row.get("precision", precision),
                    "covered_malicious_nodes": row.get("covered_malicious_nodes", ""),
                },
            )
    case_names = sorted(set(score_by_case) | set(grouped_by_case))
    target_rows: list[dict[str, Any]] = []
    for case in case_names:
        score_summary = dict(score_by_case.get(case, {}))
        counts = grouped_by_case.get(case, {})
        event_count = counts.get("event_count", score_summary.get("count", 0))
        alert_count = int(counts.get("alert_count", 0))
        tp = int(counts.get("TP", 0))
        fp = int(counts.get("FP", 0))
        precision = float(tp / max(tp + fp, 1)) if tp + fp > 0 else 0.0
        target_rows.append(
            {
                "target_case": case,
                "event_count": event_count,
                "alert_count": alert_count,
                "TP": tp,
                "FP": fp,
                "precision": precision,
                "threshold_mean": score_summary.get(
                    "final_threshold",
                    score_summary.get("threshold", ""),
                ),
                "score_mean": score_summary.get("score_mean", ""),
                "score_p99": score_summary.get("score_p99", ""),
                "score_p999": score_summary.get("score_p999", ""),
            },
        )
    _write_csv(target_case_path, target_rows, TARGET_CASE_SUMMARY_FIELDS)
    def group_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
            str(row.get("target_case", "")),
        )

    _write_csv(
        group_alert_path,
        sorted(group_rows_out, key=group_sort_key),
        GROUP_ALERT_SUMMARY_FIELDS,
    )
    return {
        "target_case_summary_csv": str(target_case_path),
        "group_alert_summary_csv": str(group_alert_path),
    }


def _phase3g_group_eval_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
) -> dict[str, Any]:
    tp = 0
    fp = 0
    covered: set[int] = set()
    coverage_tracker = OnlineNodeCoverageTracker()
    evaluated_rows: list[dict[str, Any]] = []
    for row in rows:
        label, src_bad, dst_bad = _phase3g_label_for_alert_row(
            row,
            idx_to_db_node_id,
            abnormal_db_node_ids,
        )
        event_id = int(row.get("event_index", -1))
        event_score = float(_safe_float(row.get("event_score"), 0.0) or 0.0)
        if _is_attack_label(label):
            tp += 1
        else:
            fp += 1
        if src_bad:
            covered.add(int(idx_to_db_node_id.get(int(row.get("info_src", -1)), -1)))
        if dst_bad:
            covered.add(int(idx_to_db_node_id.get(int(row.get("info_dst", -1)), -1)))
        coverage_tracker.observe(
            node_idx=int(row.get("info_src") or row.get("src_idx") or -1),
            node_type=str(row.get("src_type", "")),
            event_id=event_id,
            event_score=event_score,
            side="src",
        )
        coverage_tracker.observe(
            node_idx=int(row.get("info_dst") or row.get("dst_idx") or -1),
            node_type=str(row.get("dst_type", "")),
            event_id=event_id,
            event_score=event_score,
            side="dst",
        )
        evaluated_rows.append(_evaluated_event_row(row, {event_id: label}))
    support = _node_alert_support(evaluated_rows)
    strict_tp = 0
    strict_fp = 0
    relaxed_tp = 0
    relaxed_fp = 0
    for coverage_row in coverage_tracker.rows():
        node_idx = int(coverage_row.get("node_idx", -1))
        db_node_id = int(idx_to_db_node_id.get(node_idx, -1))
        node_label = "malicious" if db_node_id in abnormal_db_node_ids else "benign"
        if node_label == "malicious":
            strict_tp += 1
        else:
            strict_fp += 1
        relaxed = _relaxed_node_eval_result(node_label, support.get(node_idx, {}))
        if relaxed == "TP":
            relaxed_tp += 1
        elif relaxed == "FP":
            relaxed_fp += 1
    return {
        "TP": int(tp),
        "FP": int(fp),
        "precision": float(tp / max(tp + fp, 1)),
        "covered_malicious_nodes": int(len(covered)),
        "strict_node_TP": int(strict_tp),
        "strict_node_FP": int(strict_fp),
        "relaxed_node_TP": int(relaxed_tp),
        "relaxed_node_FP": int(relaxed_fp),
    }


def _phase3g_update_group_report_rows_with_eval(
    *,
    path: Path,
    alert_path: Path,
    config: SlimConfig,
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = list(_iter_csv_rows(path))
    if not rows:
        return []
    final_alerts = list(_iter_csv_rows(alert_path))
    alerts_by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for alert in final_alerts:
        key = (
            str(alert.get("action", "")),
            str(alert.get("src_type", "")),
            str(alert.get("dst_type", "")),
        )
        alerts_by_group.setdefault(key, []).append(alert)
    updated: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
        )
        metrics = _phase3g_group_eval_counts(
            alerts_by_group.get(key, []),
            idx_to_db_node_id=idx_to_db_node_id,
            abnormal_db_node_ids=abnormal_db_node_ids,
        )
        payload = dict(row)
        payload.update(metrics)
        updated.append(payload)
    fieldnames = ACTION_TYPE_POLICY_SUMMARY_FIELDS
    if path.name == "node_coverage_by_group.csv":
        fieldnames = NODE_COVERAGE_BY_GROUP_FIELDS
    elif path.name == "q_t_by_action_type.csv":
        fieldnames = Q_T_BY_ACTION_TYPE_FIELDS
    _write_csv(path, updated, fieldnames)
    return updated


def _phase3g_write_demoted_group_summary(config: SlimConfig, output_dir: Path) -> str:
    """Aggregate action/type policy demotions without changing streaming outputs."""
    node_evidence_path = output_dir / "action_type_node_evidence_events.csv"
    summary_path = output_dir / "demoted_group_summary.csv"
    if not node_evidence_path.exists():
        _write_csv(summary_path, [], DEMOTED_GROUP_SUMMARY_FIELDS)
        return str(summary_path)
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in _iter_csv_rows(node_evidence_path):
        key = (
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
            str(row.get("target_case", "")),
        )
        payload = grouped.setdefault(
            key,
            {
                "dataset": str(config.dataset),
                "run": _phase3g_run_only_from_out_tag(config.out_tag),
                "state_model": str(config.sspm_state_model),
                "policy_name": str(config.action_type_alert_policy),
                "action": key[0],
                "src_type": key[1],
                "dst_type": key[2],
                "target_case": key[3],
                "event_count": 0,
                "node_evidence_count": 0,
                "demoted_event_count": 0,
                "budget_capped_event_count": 0,
            },
        )
        payload["event_count"] = int(payload["event_count"]) + 1
        if _parse_bool(row.get("node_evidence", False)):
            payload["node_evidence_count"] = int(payload["node_evidence_count"]) + 1
        if str(row.get("alert_decision", "")) == "demoted_event":
            payload["demoted_event_count"] = int(payload["demoted_event_count"]) + 1
        if _parse_bool(row.get("budget_capped", False)):
            payload["budget_capped_event_count"] = (
                int(payload["budget_capped_event_count"]) + 1
            )
    rows = [grouped[key] for key in sorted(grouped)]
    _write_csv(summary_path, rows, DEMOTED_GROUP_SUMMARY_FIELDS)
    return str(summary_path)


def _phase3g_load_validation_group_scores(
    cache_meta: Mapping[str, Any],
) -> dict[tuple[int, int, int], list[float]]:
    score_path = Path(str(cache_meta.get("validation_conditional_scores", "")))
    group_path = Path(str(cache_meta.get("validation_conditional_group_keys", "")))
    if not score_path.exists() or not group_path.exists():
        return {}
    count = int(cache_meta.get("count", 0) or 0)
    scores = np.memmap(score_path, dtype=np.float32, mode="r", shape=(count,))
    group_keys = np.memmap(group_path, dtype=np.int64, mode="r", shape=(count,))
    grouped: dict[tuple[int, int, int], list[float]] = {}
    try:
        for score, encoded in zip(scores, group_keys):
            decoded = decode_conditional_group_key(int(encoded))
            key = (
                int(decoded["action_id"]),
                int(decoded["src_type_id"]),
                int(decoded["dst_type_id"]),
            )
            grouped.setdefault(key, []).append(float(score))
    finally:
        del scores
        del group_keys
    return grouped


def _phase3g_candidate_threshold_for_group(
    validation_group_scores: Mapping[tuple[int, int, int], Sequence[float]],
    key: tuple[int, int, int],
    quantile: float,
    margin: float,
    fallback_threshold: float,
) -> float:
    values = validation_group_scores.get(key)
    if values:
        return float(conditional_quantile_threshold(np.asarray(values, dtype=np.float32), quantile))
    return float(fallback_threshold)


def _phase3g_write_group_threshold_sweep_summary(
    *,
    config: SlimConfig,
    cache_meta: Mapping[str, Any],
    output_dir: Path,
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
) -> str:
    score_trace_path = output_dir / "online_event_score_trace.csv"
    if not score_trace_path.exists():
        _write_csv(
            output_dir / "group_threshold_sweep_summary.csv",
            [],
            GROUP_THRESHOLD_SWEEP_SUMMARY_FIELDS,
        )
        return str(output_dir / "group_threshold_sweep_summary.csv")
    targets = {
        ("EVENT_RECVFROM", "netflow", "process"),
        ("EVENT_CONNECT", "process", "netflow"),
        ("EVENT_READ", "file", "process"),
    }
    quantiles = (0.998, 0.999, 0.9995, 0.9999)
    margins = (-0.05, -0.02, 0.0, 0.02, 0.05)
    validation_group_scores = _phase3g_load_validation_group_scores(cache_meta)
    trace_by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in _iter_csv_rows(score_trace_path):
        group = (
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
        )
        if group in targets:
            trace_by_group.setdefault(group, []).append(row)
    rows: list[dict[str, Any]] = []
    for group in sorted(targets):
        action, src_type, dst_type = group
        try:
            key = (
                int(ORTHRUS10_ACTION_NAMES.index(action)),
                int(ENTITY_TYPES[src_type]),
                int(ENTITY_TYPES[dst_type]),
            )
        except (KeyError, ValueError):
            continue
        trace_rows = trace_by_group.get(group, [])
        fallback_threshold = float(cache_meta.get("threshold", 0.0))
        if trace_rows:
            fallback_threshold = float(_safe_float(trace_rows[0].get("threshold", 0.0), 0.0))
        for quantile in quantiles:
            for margin in margins:
                threshold = _phase3g_candidate_threshold_for_group(
                    validation_group_scores,
                    key,
                    quantile,
                    margin,
                    fallback_threshold,
                ) + float(margin)
                alert_rows = [
                    {
                        "event_index": int(trace.get("event_id", -1)),
                        "info_src": str(trace.get("info_src") or trace.get("src_idx") or -1),
                        "info_dst": str(trace.get("info_dst") or trace.get("dst_idx") or -1),
                        "src_idx": str(trace.get("src_idx") or trace.get("info_src") or -1),
                        "dst_idx": str(trace.get("dst_idx") or trace.get("info_dst") or -1),
                        "action": action,
                        "src_type": src_type,
                        "dst_type": dst_type,
                        "event_score": float(_safe_float(trace.get("score", 0.0), 0.0)),
                    }
                    for trace in trace_rows
                    if float(_safe_float(trace.get("score", 0.0), 0.0)) >= threshold
                ]
                metrics = _phase3g_group_eval_counts(
                    alert_rows,
                    idx_to_db_node_id=idx_to_db_node_id,
                    abnormal_db_node_ids=abnormal_db_node_ids,
                )
                rows.append(
                    {
                        **_phase3g_empty_report_metrics(config),
                        "policy_name": "per_action_type_quantile",
                        "action": action,
                        "src_type": src_type,
                        "dst_type": dst_type,
                        "threshold": threshold,
                        "event_count": int(len(trace_rows)),
                        "alert_count": int(len(alert_rows)),
                        **metrics,
                        "quantile": float(quantile),
                        "margin": float(margin),
                        "threshold_policy": "per_action_type_quantile",
                    },
                )
    path = output_dir / "group_threshold_sweep_summary.csv"
    _write_csv(path, rows, GROUP_THRESHOLD_SWEEP_SUMMARY_FIELDS)
    return str(path)


def _phase3g_endpoint_suppression_post_eval(
    *,
    config: SlimConfig,
    paths: Mapping[str, Path],
    test_index: np.ndarray,
    stream_outputs: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Backfill endpoint suppression TP/FP diagnostics after streaming."""
    post_eval_start_rss = _current_rss_mb()
    summary_path_text = str(stream_outputs.get("conditional_endpoint_suppression_summary_csv", ""))
    suppressed_csv_text = str(stream_outputs.get("suppressed_event_alerts_raw_csv", ""))
    if not summary_path_text and not suppressed_csv_text:
        return {
            "post_stream_eval_rss_start_mb": float(post_eval_start_rss),
            "post_stream_eval_rss_final_mb": float(_current_rss_mb()),
            "post_stream_eval_rss_delta_mb": 0.0,
        }
    summary_path = Path(summary_path_text)
    has_summary = bool(summary_path_text) and summary_path.exists()
    abnormal_nodes = _phase3g_load_abnormal_db_nodes_for_eval(config)
    if not abnormal_nodes:
        return {
            "summary_csv": str(summary_path) if has_summary else "",
            "post_stream_eval_rss_start_mb": float(post_eval_start_rss),
            "post_stream_eval_rss_final_mb": float(_current_rss_mb()),
            "post_stream_eval_rss_delta_mb": float(_current_rss_mb() - post_eval_start_rss),
        }
    idx_to_db_node_id = _phase3e_load_idx_to_node_id(paths)
    rows = list(_iter_csv_rows(summary_path)) if has_summary else []
    rows_by_pair: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        key = str(payload.get("pair_signature", payload.get("pair_key", "")))
        if key in rows_by_pair:
            rows_by_pair[key]["test_pair_count"] = int(
                float(rows_by_pair[key].get("test_pair_count", 0) or 0),
            ) + int(float(payload.get("test_pair_count", 0) or 0))
            rows_by_pair[key]["raw_alert_count"] = int(
                float(rows_by_pair[key].get("raw_alert_count", 0) or 0),
            ) + int(float(payload.get("raw_alert_count", 0) or 0))
            rows_by_pair[key]["suppressed_alert_count"] = int(
                float(rows_by_pair[key].get("suppressed_alert_count", 0) or 0),
            ) + int(float(payload.get("suppressed_alert_count", 0) or 0))
            rows_by_pair[key]["final_alert_count"] = int(
                float(rows_by_pair[key].get("final_alert_count", 0) or 0),
            ) + int(float(payload.get("final_alert_count", 0) or 0))
            rows_by_pair[key]["max_score"] = max(
                float(rows_by_pair[key].get("max_score", 0.0) or 0.0),
                float(payload.get("max_score", 0.0) or 0.0),
            )
            continue
        rows_by_pair[key] = payload

    def update_pair_counts(alert_rows: Sequence[Mapping[str, Any]], suffix: str) -> tuple[int, int]:
        tp = 0
        fp = 0
        for alert in alert_rows:
            pair_key = str(
                alert.get("endpoint_pair_key", "")
                or _conditional_endpoint_signature_text_from_row(alert)
            )
            row = rows_by_pair.get(pair_key)
            if row is None:
                continue
            correct = bool(_parse_bool(alert.get("is_correct_event_alert", False)))
            if correct:
                row[f"tp_{suffix}"] = int(float(row.get(f"tp_{suffix}", 0) or 0)) + 1
                tp += 1
            else:
                row[f"fp_{suffix}"] = int(float(row.get(f"fp_{suffix}", 0) or 0)) + 1
                fp += 1
        return tp, fp

    final_alert_rows = list(_iter_csv_rows(Path(output_dir) / "online_event_alerts.csv"))
    if suppressed_csv_text and Path(suppressed_csv_text).exists():
        suppressed_raw_rows = list(_iter_csv_rows(Path(suppressed_csv_text)))
    else:
        suppressed_raw_rows = [
            dict(row)
            for row in stream_outputs.get("suppressed_event_alerts_raw", [])
            if isinstance(row, Mapping)
        ]
    alert_event_indices = {int(row["event_index"]) for row in final_alert_rows}
    alert_event_indices.update(int(row["event_index"]) for row in suppressed_raw_rows)
    labels_by_event = _phase3e_event_labels_for_alerts(
        records=test_index,
        alert_event_indices=alert_event_indices,
        abnormal_db_node_ids=abnormal_nodes,
        idx_to_db_node_id=idx_to_db_node_id,
    )
    evaluated_final_rows = [
        _evaluated_event_row(row, labels_by_event)
        for row in final_alert_rows
    ]
    evaluated_suppressed_rows = [
        _evaluated_event_row(row, labels_by_event)
        for row in suppressed_raw_rows
    ]
    suppressed_tp = 0
    suppressed_fp = 0
    final_tp, final_fp = update_pair_counts(evaluated_final_rows, "after")
    before_rows = [*evaluated_final_rows, *evaluated_suppressed_rows]
    before_tp, before_fp = update_pair_counts(before_rows, "before")
    for row in evaluated_suppressed_rows:
        if bool(_parse_bool(row.get("is_correct_event_alert", False))):
            suppressed_tp += 1
        else:
            suppressed_fp += 1
    if has_summary and rows_by_pair:
        rows = [rows_by_pair[key] for key in sorted(rows_by_pair)]
        fieldnames = list(rows[0].keys())
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    suppressed_tp_path = ""
    suppressed_tp_rows = [
        row
        for row in evaluated_suppressed_rows
        if bool(_parse_bool(row.get("is_correct_event_alert", False)))
    ]
    if suppressed_tp_rows:
        suppressed_tp_path = str(
            _write_conditional_endpoint_suppressed_tp_events(output_dir, suppressed_tp_rows),
        )
        _write_conditional_endpoint_suppressed_tp_audit(output_dir, suppressed_tp_rows)
    post_eval_final_rss = _current_rss_mb()
    return {
        "summary_csv": str(summary_path) if has_summary else "",
        "suppressed_raw_csv": suppressed_csv_text,
        "tp_before": int(before_tp),
        "fp_before": int(before_fp),
        "tp_after": int(final_tp),
        "fp_after": int(final_fp),
        "suppressed_tp": int(suppressed_tp),
        "suppressed_fp": int(suppressed_fp),
        "suppressed_alert_count": int(suppressed_tp + suppressed_fp),
        "suppressed_tp_events_csv": suppressed_tp_path,
        "pair_label_scope": "post_stream_event_level",
        "post_stream_eval_rss_start_mb": float(post_eval_start_rss),
        "post_stream_eval_rss_final_mb": float(post_eval_final_rss),
        "post_stream_eval_rss_delta_mb": float(post_eval_final_rss - post_eval_start_rss),
    }


def _phase3g_label_for_alert_row(
    row: Mapping[str, Any],
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
) -> tuple[int, bool, bool]:
    src_idx = int(row.get("info_src") or row.get("src_idx") or -1)
    dst_idx = int(row.get("info_dst") or row.get("dst_idx") or -1)
    src_db = int(idx_to_db_node_id.get(src_idx, -1))
    dst_db = int(idx_to_db_node_id.get(dst_idx, -1))
    src_bad = src_db in abnormal_db_node_ids
    dst_bad = dst_db in abnormal_db_node_ids
    label = 2 if src_bad and dst_bad else (1 if src_bad or dst_bad else 0)
    return label, src_bad, dst_bad


def _phase3g_event_node_coverage_from_alerts(
    alert_path: Path,
) -> list[dict[str, Any]]:
    tracker = OnlineNodeCoverageTracker()
    for row in _iter_csv_rows(alert_path):
        event_id = int(row.get("event_index", -1))
        event_score = float(_safe_float(row.get("event_score"), 0.0) or 0.0)
        tracker.observe(
            node_idx=int(row.get("info_src") or row.get("src_idx") or -1),
            node_type=str(row.get("src_type", "")),
            event_id=event_id,
            event_score=event_score,
            side="src",
        )
        tracker.observe(
            node_idx=int(row.get("info_dst") or row.get("dst_idx") or -1),
            node_type=str(row.get("dst_type", "")),
            event_id=event_id,
            event_score=event_score,
            side="dst",
        )
    return tracker.rows()


def _phase3g_write_event_node_coverage_outputs(
    *,
    output_dir: Path,
    coverage_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    raw_path = output_dir / "online_event_node_coverage.csv"
    strict_path = output_dir / "online_event_node_coverage_strict.csv"
    relaxed_path = output_dir / "online_event_node_coverage_relaxed.csv"
    summary_path = output_dir / "online_event_node_coverage_summary.json"

    support = _node_alert_support(event_rows)
    strict_rows: list[dict[str, Any]] = []
    relaxed_rows: list[dict[str, Any]] = []
    covered_malicious: set[int] = set()
    for raw_row in coverage_rows:
        node_idx = int(raw_row.get("node_idx", -1))
        db_node_id = int(idx_to_db_node_id.get(node_idx, -1))
        node_label = "malicious" if db_node_id in abnormal_db_node_ids else "benign"
        if node_label == "malicious":
            covered_malicious.add(db_node_id)
        base = dict(raw_row)
        base["node_label"] = node_label
        strict_rows.append(
            {
                **base,
                "eval_result": "TP" if node_label == "malicious" else "FP",
            },
        )
        relaxed_result = _relaxed_node_eval_result(node_label, support.get(node_idx, {}))
        relaxed_rows.append({**base, "eval_result": relaxed_result})

    _write_csv(raw_path, list(coverage_rows), EVENT_NODE_COVERAGE_FIELDS)
    _write_csv(strict_path, strict_rows, EVENT_NODE_COVERAGE_EVAL_FIELDS)
    _write_csv(relaxed_path, relaxed_rows, EVENT_NODE_COVERAGE_EVAL_FIELDS)
    legacy_strict_rows = [
        {
            "node_id": int(row.get("node_idx", -1)),
            "node_type": row.get("node_type", ""),
            "node_label": row.get("node_label", ""),
            "first_alert_event_idx": row.get("first_alert_event_id", ""),
            "alert_count": row.get("alert_count", 0),
            "max_event_score": row.get("max_event_score", 0.0),
            "pool_type": "strict",
            "eval_result": row.get("eval_result", ""),
            "supporting_event_count": row.get("alert_count", 0),
        }
        for row in strict_rows
    ]
    legacy_relaxed_rows = [
        {
            "node_id": int(row.get("node_idx", -1)),
            "node_type": row.get("node_type", ""),
            "node_label": row.get("node_label", ""),
            "first_alert_event_idx": row.get("first_alert_event_id", ""),
            "alert_count": row.get("alert_count", 0),
            "max_event_score": row.get("max_event_score", 0.0),
            "pool_type": "relaxed",
            "eval_result": row.get("eval_result", ""),
            "supporting_event_count": row.get("alert_count", 0),
        }
        for row in relaxed_rows
    ]
    _write_csv(output_dir / "online_node_alerts_strict.csv", legacy_strict_rows, NODE_POOL_EVAL_FIELDS)
    _write_csv(
        output_dir / "online_node_alerts_relaxed.csv",
        legacy_relaxed_rows,
        NODE_POOL_EVAL_FIELDS,
    )

    strict_counts = _eval_result_counts(strict_rows)
    relaxed_counts = _eval_result_counts(relaxed_rows)
    total_malicious = int(len(abnormal_db_node_ids))
    summary = {
        "online_event_node_coverage_csv": str(raw_path),
        "online_event_node_coverage_strict_csv": str(strict_path),
        "online_event_node_coverage_relaxed_csv": str(relaxed_path),
        "coverage_node_count": int(len(coverage_rows)),
        "covered_malicious_nodes": int(len(covered_malicious)),
        "total_malicious_nodes": total_malicious,
        "malicious_node_recall": float(len(covered_malicious) / total_malicious)
        if total_malicious
        else 0.0,
        "strict_node_tp": int(strict_counts["tp"]),
        "strict_node_fp": int(strict_counts["fp"]),
        "strict_node_precision": float(
            strict_counts["tp"] / max(strict_counts["tp"] + strict_counts["fp"], 1),
        ),
        "strict_node_recall": float(strict_counts["tp"] / max(total_malicious, 1)),
        "relaxed_node_tp": int(relaxed_counts["tp"]),
        "relaxed_node_fp": int(relaxed_counts["fp"]),
        "relaxed_node_ignore": int(relaxed_counts["ignore"]),
        "relaxed_node_precision": float(
            relaxed_counts["tp"] / max(relaxed_counts["tp"] + relaxed_counts["fp"], 1),
        ),
        "relaxed_node_recall": float(relaxed_counts["tp"] / max(total_malicious, 1)),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def _phase3g_write_node_pool_rebuilt_outputs(
    *,
    output_dir: Path,
    coverage_rows: Sequence[Mapping[str, Any]],
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
    topk_values: Sequence[int],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    final_nodes_raw = [
        {
            "node_id": int(row.get("node_idx", -1)),
            "node_score": float(row.get("max_event_score", 0.0)),
            "candidate_mass": float(row.get("max_event_score", 0.0)),
            "residual_mass": float(row.get("max_event_score", 0.0)),
            "residual_max": float(row.get("max_event_score", 0.0)),
            "residual_mean": float(row.get("max_event_score", 0.0)),
            "association_support": 0,
            "adaptive_memory_deviation": 0.0,
            "compact_chain_support": 0.0,
            "chain_diversity_pool": 1,
            "repeated_consistency": int(row.get("alert_count", 0)),
            "candidate_event_count": int(row.get("alert_count", 0)),
        }
        for row in coverage_rows
    ]
    node_labels = {
        int(row.get("node_idx", -1)): "malicious"
        for row in coverage_rows
        if int(idx_to_db_node_id.get(int(row.get("node_idx", -1)), -1)) in abnormal_db_node_ids
    }
    _write_csv(output_dir / "final_node_pool_alerts.csv", _attach_node_labels(final_nodes_raw, node_labels), NODE_EVAL_FIELDS)
    _write_csv(output_dir / "online_node_alerts.csv", _attach_node_labels(final_nodes_raw, node_labels), NODE_EVAL_FIELDS)
    topk_metrics = _node_pool_topk_metrics(final_nodes_raw, node_labels, topk_values)
    topk_csv = output_dir / "node_topk_metrics.csv"
    topk_fields = ["k", "alert_count_at_k", "count", "tp", "fp", "precision", "ratio"]
    with topk_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=topk_fields)
        writer.writeheader()
        for key in sorted(topk_metrics, key=lambda item: int(item.replace("top", ""))):
            writer.writerow({field: topk_metrics[key].get(field, "") for field in topk_fields})
    topk_json = output_dir / "node_topk_metrics.json"
    topk_json.write_text(
        json.dumps(topk_metrics, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    summary = {
        "node_pool_rebuilt_from": "online_event_node_coverage",
        "node_pool_count": int(len(final_nodes_raw)),
        "node_topk_metrics_csv": str(topk_csv),
        "node_topk_metrics_json": str(topk_json),
        "node_pool_topk": topk_metrics,
    }
    summary_path = output_dir / "node_pool_rebuilt_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def _phase3g_write_fp_group_outputs(
    *,
    output_dir: Path,
    event_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    output_dir = Path(output_dir)
    event_groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in event_rows:
        key = (
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
            _label_name(row.get("event_label", "benign")),
        )
        group = event_groups.setdefault(
            key,
            {"scores": [], "tp": 0, "fp": 0, "event_count": 0},
        )
        score = float(_safe_float(row.get("event_score"), 0.0) or 0.0)
        group["scores"].append(score)
        group["event_count"] = int(group["event_count"]) + 1
        if bool(_parse_bool(row.get("is_correct_event_alert", False))):
            group["tp"] = int(group["tp"]) + 1
        else:
            group["fp"] = int(group["fp"]) + 1
    event_rows_out = []
    for key, group in event_groups.items():
        scores = list(group["scores"])
        tp = int(group["tp"])
        fp = int(group["fp"])
        event_count = int(group["event_count"])
        event_rows_out.append(
            {
                "action": key[0],
                "src_type": key[1],
                "dst_type": key[2],
                "event_label": key[3],
                "event_count": event_count,
                "tp": tp,
                "fp": fp,
                "precision": float(tp / max(tp + fp, 1)),
                "score_min": min(scores) if scores else 0.0,
                "score_mean": float(sum(scores) / len(scores)) if scores else 0.0,
                "score_max": max(scores) if scores else 0.0,
            },
        )
    event_rows_out.sort(key=lambda row: (-int(row["fp"]), -int(row["tp"]), str(row["action"])))
    event_fp_path = output_dir / "event_fp_group_summary.csv"
    _write_csv(event_fp_path, event_rows_out, EVENT_FP_GROUP_FIELDS)

    node_rows_out = []
    for row in coverage_rows:
        node_rows_out.append(
            {
                "node_type": str(row.get("node_type", "")),
                "dominant_action": "",
                "dominant_peer_type": "",
                "node_count": 1,
                "strict_tp": 1 if str(row.get("node_label", "")) == "malicious" else 0,
                "strict_fp": 0 if str(row.get("node_label", "")) == "malicious" else 1,
                "total_event_support": int(row.get("alert_count", 0)),
                "max_event_score": float(row.get("max_event_score", 0.0)),
            },
        )
    by_node_type: dict[str, dict[str, Any]] = {}
    for row in node_rows_out:
        group = by_node_type.setdefault(
            str(row["node_type"]),
            {
                "node_type": str(row["node_type"]),
                "dominant_action": "",
                "dominant_peer_type": "",
                "node_count": 0,
                "strict_tp": 0,
                "strict_fp": 0,
                "total_event_support": 0,
                "max_event_score": 0.0,
            },
        )
        group["node_count"] = int(group["node_count"]) + 1
        group["strict_tp"] = int(group["strict_tp"]) + int(row["strict_tp"])
        group["strict_fp"] = int(group["strict_fp"]) + int(row["strict_fp"])
        group["total_event_support"] = int(group["total_event_support"]) + int(
            row["total_event_support"],
        )
        group["max_event_score"] = max(
            float(group["max_event_score"]),
            float(row["max_event_score"]),
        )
    node_fp_path = output_dir / "node_fp_group_summary.csv"
    _write_csv(node_fp_path, list(by_node_type.values()), NODE_FP_GROUP_FIELDS)
    return {
        "event_fp_group_summary_csv": str(event_fp_path),
        "node_fp_group_summary_csv": str(node_fp_path),
    }


def _phase3g_array_nbytes_mb(array: Any) -> float:
    if isinstance(array, LazyMmapLruEmbeddingLookup):
        return float(Path(array.path).stat().st_size) / 1024.0 / 1024.0
    nbytes = getattr(array, "nbytes", 0)
    try:
        return float(nbytes) / 1024.0 / 1024.0
    except (TypeError, ValueError):
        return 0.0


def _phase3g_file_size_mb(path_value: Any) -> float:
    text = str(path_value or "").strip()
    if not text:
        return 0.0
    path = Path(text)
    if not path.exists():
        return 0.0
    return float(path.stat().st_size) / 1024.0 / 1024.0


def _phase3g_mapped_file_rss_mb(path_value: Any) -> float | None:
    text = str(path_value or "").strip()
    if not text:
        return None
    try:
        target = str(Path(text).resolve())
    except OSError:
        return None
    smaps_path = Path("/proc/self/smaps")
    if not smaps_path.exists():
        return None
    total_kb = 0.0
    current_path = ""
    try:
        with smaps_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n").split(maxsplit=5)
                if parts and "-" in parts[0] and len(parts[0].split("-", 1)[0]) >= 4:
                    current_path = parts[5] if len(parts) >= 6 else ""
                    if current_path.endswith(" (deleted)"):
                        current_path = current_path[: -len(" (deleted)")]
                    continue
                if current_path != target or not line.startswith("Rss:"):
                    continue
                value_parts = line.split()
                if len(value_parts) >= 2:
                    total_kb += float(value_parts[1])
    except (OSError, ValueError):
        return None
    return float(total_kb) / 1024.0


def _phase3g_malloc_trim() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (AttributeError, OSError):
        return


def _phase3g_smaps_timeline_row(stage: str) -> dict[str, Any]:
    smaps = _current_smaps_rollup_mb()
    return {
        "stage": str(stage),
        "rss_mb": _current_rss_mb(),
        "anonymous_rss_mb": smaps.get("anonymous_rss_mb"),
        "file_backed_rss_mb": smaps.get("file_backed_rss_mb"),
        "shared_clean_mb": smaps.get("shared_clean_mb"),
        "private_clean_mb": smaps.get("private_clean_mb"),
        "private_dirty_mb": smaps.get("private_dirty_mb"),
    }


def _phase3g_write_rss_timeline(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path = Path(output_dir) / "rss_timeline.csv"
    fields = [
        "stage",
        "rss_mb",
        "anonymous_rss_mb",
        "file_backed_rss_mb",
        "shared_clean_mb",
        "private_clean_mb",
        "private_dirty_mb",
    ]
    _write_csv(path, list(rows), fields)
    return str(path)


def _phase3g_write_online_event_core_rss_breakdown(
    *,
    output_dir: Path,
    eval_payload: Mapping[str, Any],
    paths: Mapping[str, Path],
    node_embeddings: Any,
    endpoint_cache_mb: float,
    endpoint_cache_meta: Mapping[str, Any] | None = None,
    rss_timeline: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    memory = dict(eval_payload.get("memory", {})) if isinstance(eval_payload, Mapping) else {}
    smaps = _current_smaps_rollup_mb()
    event_index_test_mb = _phase3g_file_size_mb(paths["event_index_test"])
    event_index_validation_mb = _phase3g_file_size_mb(paths["event_index_validation"])
    embedding_total_mb = _phase3g_array_nbytes_mb(node_embeddings)
    embedding_breakdown = _phase3g_embedding_memory_breakdown_for_paths(
        paths=paths,
        node_embeddings=node_embeddings,
        state_table_mb=float(
            memory.get("online_minimal_state_array_mb")
            or memory.get("state_array_mb")
            or 0.0,
        ),
        endpoint_cache_mb=float(endpoint_cache_mb),
    )
    file_backed_peak = float(
        memory.get("online_minimal_file_backed_rss_mb")
        or smaps.get("file_backed_rss_mb")
        or 0.0,
    )
    mapped_test_mb = _phase3g_mapped_file_rss_mb(paths.get("event_index_test"))
    mapped_validation_mb = _phase3g_mapped_file_rss_mb(paths.get("event_index_validation"))
    mapped_embedding_mb = _phase3g_mapped_file_rss_mb(paths.get("node_embeddings"))
    excluded_test_file_backed_mb = (
        float(mapped_test_mb)
        if mapped_test_mb is not None and mapped_test_mb > 0.0
        else min(event_index_test_mb, file_backed_peak)
    )
    endpoint_meta = dict(endpoint_cache_meta or {})
    endpoint_pair_keys_mb = _phase3g_file_size_mb(endpoint_meta.get("endpoint_pair_keys_path"))
    endpoint_pair_counts_mb = _phase3g_file_size_mb(endpoint_meta.get("endpoint_pair_counts_path"))
    endpoint_action_keys_mb = _phase3g_file_size_mb(endpoint_meta.get("endpoint_action_keys_path"))
    endpoint_action_counts_mb = _phase3g_file_size_mb(
        endpoint_meta.get("endpoint_action_counts_path"),
    )
    state_table_mb = float(
        memory.get("online_minimal_state_array_mb")
        or memory.get("state_array_mb")
        or 0.0,
    )
    embedding_lookup_mode = str(
        memory.get("embedding_lookup_mode")
        or getattr(node_embeddings, "lookup_mode", "")
        or "",
    )
    lazy_embedding_mb = float(memory.get("embedding_lru_cache_mb", 0.0) or 0.0)
    deploy_embedding_mb = lazy_embedding_mb if embedding_lookup_mode == "lazy_mmap_lru" else embedding_total_mb
    model_param_mb = float(memory.get("online_deploy_model_param_mb", 0.0) or 0.0)
    runtime_buffer_mb = float(memory.get("online_deploy_runtime_buffer_mb", 0.0) or 0.0)
    threshold_cache_mb = float(memory.get("online_deploy_threshold_cache_mb", 0.0) or 0.0)
    event_writer_buffer_mb = float(
        memory.get("online_minimal_alert_buffer_mb", 0.0) or 0.0,
    )
    excluded_embedding_file_backed_mb = float(mapped_embedding_mb or 0.0)
    excluded_validation_file_backed_mb = (
        float(mapped_validation_mb)
        if mapped_validation_mb is not None and mapped_validation_mb > 0.0
        else min(
            event_index_validation_mb,
            max(file_backed_peak - excluded_test_file_backed_mb - excluded_embedding_file_backed_mb, 0.0),
        )
    )
    core_file_backed_mb = max(
        file_backed_peak
        - excluded_test_file_backed_mb
        - excluded_embedding_file_backed_mb
        - excluded_validation_file_backed_mb,
        0.0,
    )
    core_anonymous_mb = float(
        memory.get("online_minimal_anonymous_rss_mb")
        or smaps.get("anonymous_rss_mb")
        or 0.0,
    )
    core_peak = core_anonymous_mb + core_file_backed_mb
    payload = {
        "definition": (
            "online_event_core_rss excludes replay event_index file-backed pages, "
            "validation cache, label attach, and offline node aggregation from the "
            "process RSS. It keeps model/state/threshold/endpoint-cache/event-writer "
            "objects required for online event scoring."
        ),
        "process_rss_peak_mb": float(
            memory.get("rss_test_peak_mb", memory.get("online_minimal_rss_peak_mb", 0.0))
            or memory.get("online_minimal_rss_peak_mb", 0.0)
            or 0.0,
        ),
        "process_rss_final_mb": float(memory.get("online_minimal_rss_final_mb", 0.0) or 0.0),
        "online_event_core_rss_mb": float(core_peak),
        "online_event_core_rss_peak_mb": float(core_peak),
        "online_event_core_anonymous_rss_mb": float(core_anonymous_mb),
        "online_event_core_file_backed_rss_mb": float(core_file_backed_mb),
        "online_deploy_required_memory_mb": float(
            deploy_embedding_mb
            + state_table_mb
            + float(endpoint_cache_mb)
            + threshold_cache_mb
            + model_param_mb
            + runtime_buffer_mb
            + event_writer_buffer_mb
        ),
        "online_deploy_primary_memory_mb": float(
            deploy_embedding_mb + state_table_mb + float(endpoint_cache_mb),
        ),
        "online_deploy_primary_memory_mb_full_pipeline": float(
            embedding_breakdown.get("online_deploy_primary_memory_mb_full_pipeline", 0.0),
        ),
        "online_deploy_primary_memory_mb_test_stream": float(
            embedding_breakdown.get("online_deploy_primary_memory_mb_test_stream", 0.0),
        ),
        "online_deploy_primary_memory_mb_lazy_estimate": float(
            embedding_breakdown.get("online_deploy_primary_memory_mb_lazy_estimate", 0.0),
        ),
        "online_deploy_embedding_mb": float(deploy_embedding_mb),
        "online_deploy_state_table_mb": float(state_table_mb),
        "online_deploy_endpoint_cache_mb": float(endpoint_cache_mb),
        "online_deploy_threshold_cache_mb": float(threshold_cache_mb),
        "online_deploy_model_param_mb": float(model_param_mb),
        "conditional_head_param_mb": float(memory.get("conditional_head_param_mb", 0.0) or 0.0),
        "state_model_param_mb": float(memory.get("state_model_param_mb", 0.0) or 0.0),
        "calibration_param_mb": float(memory.get("calibration_param_mb", 0.0) or 0.0),
        "online_deploy_runtime_buffer_mb": float(runtime_buffer_mb),
        "python_runtime_overhead_estimate_mb": float(
            max(core_anonymous_mb - state_table_mb - model_param_mb - runtime_buffer_mb, 0.0),
        ),
        "state_table_mb": state_table_mb,
        "embedding_table_total_mb": embedding_total_mb,
        "embedding_lookup_mode": embedding_lookup_mode or str(
            getattr(node_embeddings, "lookup_mode", "global_or_compact_mmap"),
        ),
        "embedding_lazy_backing": str(memory.get("embedding_lazy_backing", "")),
        "embedding_cache_max_nodes": int(memory.get("embedding_cache_max_nodes", 0) or 0),
        "embedding_cache_active_nodes": int(memory.get("embedding_cache_active_nodes", 0) or 0),
        "embedding_lru_cache_mb": float(memory.get("embedding_lru_cache_mb", 0.0) or 0.0),
        "embedding_lru_cache_mb_actual": float(
            memory.get("embedding_lru_cache_mb_actual")
            or memory.get("embedding_lru_cache_mb")
            or 0.0,
        ),
        "embedding_lru_cache_mb_capacity": float(
            memory.get("embedding_lru_cache_mb_capacity", 0.0) or 0.0,
        ),
        "embedding_cache_hit_count": int(memory.get("embedding_cache_hit_count", 0) or 0),
        "embedding_cache_miss_count": int(memory.get("embedding_cache_miss_count", 0) or 0),
        "embedding_cache_hit_rate": float(memory.get("embedding_cache_hit_rate", 0.0) or 0.0),
        "embedding_cache_miss_rate": float(memory.get("embedding_cache_miss_rate", 0.0) or 0.0),
        "embedding_lookup_seconds": float(memory.get("embedding_lookup_seconds", 0.0) or 0.0),
        "embedding_file_backed_rss_mb": float(excluded_embedding_file_backed_mb),
        "embedding_table_file_backed_mb": float(excluded_embedding_file_backed_mb),
        "embedding_lookup_active_pages_mb": float(excluded_embedding_file_backed_mb),
        "embedding_table_file_size_mb": _phase3g_file_size_mb(paths.get("node_embeddings")),
        "embedding_dtype": str(getattr(node_embeddings, "dtype", "")),
        "embedding_shape": list(getattr(node_embeddings, "shape", [])),
        "endpoint_cache_mb": float(endpoint_cache_mb),
        "endpoint_pair_keys_mb": float(endpoint_pair_keys_mb),
        "endpoint_pair_counts_mb": float(endpoint_pair_counts_mb),
        "endpoint_action_keys_mb": float(endpoint_action_keys_mb),
        "endpoint_action_counts_mb": float(endpoint_action_counts_mb),
        "endpoint_cache_loaded_mode": "compact_v2"
        if bool(endpoint_meta.get("compact_cache", False))
        else "",
        "threshold_cache_mb": threshold_cache_mb,
        "event_writer_buffer_mb": event_writer_buffer_mb,
        "event_node_coverage_writer_mb": float(
            memory.get("online_minimal_event_node_coverage_writer_mb", 0.0) or 0.0,
        ),
        "test_event_index_file_size_mb": float(event_index_test_mb),
        "validation_event_index_file_size_mb": float(event_index_validation_mb),
        "excluded_test_event_index_file_backed_mb": float(excluded_test_file_backed_mb),
        "excluded_validation_event_index_file_backed_mb": float(
            excluded_validation_file_backed_mb,
        ),
        "excluded_test_cache_file_backed_mb": float(excluded_test_file_backed_mb),
        "excluded_embedding_table_file_backed_mb": float(excluded_embedding_file_backed_mb),
        "excluded_validation_cache_mb": 0.0,
        "validation_cache_rss_mb": 0.0,
        "post_stream_eval_rss_mb": float(memory.get("post_stream_eval_rss_mb", 0.0) or 0.0),
        "node_aggregation_rss_mb": 0.0,
        "label_attach_rss_mb": 0.0,
        "pandas_or_dataframe_rss_mb": 0.0,
        "rss_timeline_csv": _phase3g_write_rss_timeline(output_dir, rss_timeline or []),
        "phase3g_embedding_memory_breakdown_json": str(
            Path(output_dir) / "phase3g_embedding_memory_breakdown.json",
        ),
    }
    payload["embedding_memory_breakdown"] = dict(embedding_breakdown)
    path = Path(output_dir) / "rss_breakdown_online_event_core.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    breakdown_path = Path(output_dir) / "phase3g_embedding_memory_breakdown.json"
    breakdown_path.write_text(
        json.dumps(embedding_breakdown, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def _phase3g_conditional_head_param_mb(head: ConditionalSemanticHead) -> float:
    if head.is_dual_head:
        total = int(
            head.event_w1.nbytes
            + head.event_w2.nbytes
            + head.event_bias.nbytes
            + head.action_w1.nbytes
            + head.action_w2.nbytes
            + head.action_bias.nbytes,
        )
    else:
        total = int(head.w1.nbytes + head.w2.nbytes + head.bias.nbytes)
    return float(total / (1024.0 * 1024.0))


def _phase3g_conditional_model_param_breakdown_mb(
    head: ConditionalSemanticHead,
    model: SSPMLowRankModel,
) -> dict[str, float]:
    head_mb = _phase3g_conditional_head_param_mb(head)
    state_model_mb = _phase3g_state_model_param_mb(model)
    calibration_mb = _phase3g_calibration_param_mb(model)
    return {
        "conditional_head_param_mb": float(head_mb),
        "state_model_param_mb": float(state_model_mb),
        "calibration_param_mb": float(calibration_mb),
        "model_param_mb": float(head_mb + state_model_mb + calibration_mb),
    }


def _phase3g_state_model_param_mb(model: SSPMLowRankModel) -> float:
    state = model.state_model.state_dict()
    total = 0
    for value in state.values():
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
    return float(total / (1024.0 * 1024.0))


def _phase3g_calibration_param_mb(model: SSPMLowRankModel) -> float:
    total = 0
    calibration = getattr(model, "residual_calibration_model", None)
    if calibration is not None:
        for value in getattr(calibration, "__dict__", {}).values():
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
            elif isinstance(value, Mapping):
                for nested in value.values():
                    if isinstance(nested, np.ndarray):
                        total += int(nested.nbytes)
    gate = getattr(model, "update_gate_calibrator", None)
    if gate is not None:
        for value in getattr(gate, "__dict__", {}).values():
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
    return float(total / (1024.0 * 1024.0))


def _phase3g_embedding_memory_breakdown_for_paths(
    *,
    paths: Mapping[str, Path],
    node_embeddings: Any,
    state_table_mb: float,
    endpoint_cache_mb: float,
) -> dict[str, Any]:
    split_indexes: dict[str, np.ndarray] = {}
    global_embedding_node_count = _phase3g_global_embedding_node_count(paths)
    try:
        for split in ("train", "validation", "test"):
            key = f"event_index_{split}"
            if key not in paths:
                continue
            path = Path(paths[key])
            item_size = np.dtype(EVENT_INDEX_DTYPE).itemsize
            if not path.exists() or item_size <= 0:
                continue
            num_events = int(path.stat().st_size // item_size)
            split_indexes[split] = open_event_index_memmap(path, num_events=num_events, mode="r")
        payload = embedding_memory_breakdown(
            node_embeddings=node_embeddings,
            split_event_indexes=split_indexes,
            active_state_table_mb=float(state_table_mb),
            endpoint_cache_mb=float(endpoint_cache_mb),
            global_embedding_node_count=global_embedding_node_count,
        )
        payload["event_index_source"] = "phase3e_or_compact_event_index_memmap"
        return payload
    finally:
        for index in split_indexes.values():
            del index


def _phase3g_global_embedding_node_count(paths: Mapping[str, Path]) -> int | None:
    compact_meta_path = paths.get("compact_embedding_meta")
    if compact_meta_path is None:
        return None
    path = Path(compact_meta_path)
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("num_nodes_global_embedding_table", "global_embedding_nodes"):
        value = meta.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _phase3g_write_event_coverage_report(
    *,
    output_dir: Path,
    alert_path: Path,
    coverage_rows: Sequence[Mapping[str, Any]] | None,
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
    topk_values: Sequence[int],
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    evaluated_event_rows = []
    for raw_row in _iter_csv_rows(alert_path):
        label, _, _ = _phase3g_label_for_alert_row(
            raw_row,
            idx_to_db_node_id,
            abnormal_db_node_ids,
        )
        evaluated_event_rows.append(_evaluated_event_row(raw_row, {int(raw_row["event_index"]): label}))
    coverage = list(coverage_rows or _phase3g_event_node_coverage_from_alerts(alert_path))
    coverage_summary = _phase3g_write_event_node_coverage_outputs(
        output_dir=output_dir,
        coverage_rows=coverage,
        event_rows=evaluated_event_rows,
        idx_to_db_node_id=idx_to_db_node_id,
        abnormal_db_node_ids=abnormal_db_node_ids,
    )
    pool_summary = _phase3g_write_node_pool_rebuilt_outputs(
        output_dir=output_dir,
        coverage_rows=coverage,
        idx_to_db_node_id=idx_to_db_node_id,
        abnormal_db_node_ids=abnormal_db_node_ids,
        topk_values=topk_values,
    )
    fp_paths = _phase3g_write_fp_group_outputs(
        output_dir=output_dir,
        event_rows=evaluated_event_rows,
        coverage_rows=[
            {
                **dict(row),
                "node_label": "malicious"
                if int(idx_to_db_node_id.get(int(row.get("node_idx", -1)), -1))
                in abnormal_db_node_ids
                else "benign",
            }
            for row in coverage
        ],
    )
    return {**coverage_summary, **pool_summary, **fp_paths}


def _phase3g_backfill_result_reports_from_config(
    *,
    result_dir: Path,
    config: SlimConfig,
) -> Path:
    output_dir = Path(result_dir)
    paths = _phase3e_require_artifacts(config)
    idx_to_db_node_id = _phase3e_load_idx_to_node_id(paths)
    abnormal_nodes = _phase3g_load_abnormal_db_nodes_for_eval(config)
    node_embeddings = np.load(paths["node_embeddings"], mmap_mode="r")
    metrics_path = output_dir / "metrics.json"
    eval_path = output_dir / "eval_causal_semantics_slim.json"
    eval_payload = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    metrics_payload = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    )
    endpoint_meta: dict[str, Any] = {}
    endpoint_cache_mb = 0.0
    phase3g = metrics_payload.get("eval", {}).get("phase3g", {})
    if isinstance(phase3g, Mapping):
        endpoint_payload = phase3g.get("conditional_endpoint_suppression", {})
        if isinstance(endpoint_payload, Mapping):
            endpoint_cache_mb = float(endpoint_payload.get("endpoint_cache_mb", 0.0) or 0.0)
            meta_path = endpoint_payload.get("cache_meta_path")
            if meta_path and Path(str(meta_path)).exists():
                endpoint_meta = json.loads(Path(str(meta_path)).read_text(encoding="utf-8"))
    report = _phase3g_write_event_coverage_report(
        output_dir=output_dir,
        alert_path=output_dir / "online_event_alerts.csv",
        coverage_rows=None,
        idx_to_db_node_id=idx_to_db_node_id,
        abnormal_db_node_ids=abnormal_nodes,
        topk_values=_parse_int_list(config.node_pool_topk_values),
    )
    memory_source = eval_payload if eval_payload else dict(metrics_payload.get("eval", {}))
    rss_payload = _phase3g_write_online_event_core_rss_breakdown(
        output_dir=output_dir,
        eval_payload=memory_source,
        paths=paths,
        node_embeddings=node_embeddings,
        endpoint_cache_mb=endpoint_cache_mb,
        endpoint_cache_meta=endpoint_meta,
        rss_timeline=[],
    )
    if eval_payload:
        outputs = dict(eval_payload.get("outputs", {}))
        outputs.update(
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
            },
        )
        eval_payload["outputs"] = outputs
        eval_payload["online_event_node_coverage"] = report
        eval_payload["rss_breakdown_online_event_core"] = rss_payload
        primary = dict(eval_payload.get("primary_online_metrics", {}))
        primary["online_event_node_coverage"] = {
            key: value
            for key, value in report.items()
            if not str(key).endswith("_csv") and not str(key).endswith("_json")
        }
        primary["node_pool_topk"] = report.get("node_pool_topk", {})
        eval_payload["primary_online_metrics"] = primary
        eval_path.write_text(
            json.dumps(eval_payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if metrics_payload:
        metrics_payload["online_event_node_coverage"] = report
        metrics_payload["rss_breakdown_online_event_core"] = rss_payload
        metrics_payload.setdefault("paths", {}).update(
            {
                "rss_breakdown_online_event_core_json": str(
                    output_dir / "rss_breakdown_online_event_core.json",
                ),
                "online_event_node_coverage_csv": str(
                    output_dir / "online_event_node_coverage.csv",
                ),
                "online_event_node_coverage_summary_json": str(
                    output_dir / "online_event_node_coverage_summary.json",
                ),
            },
        )
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return output_dir / "online_event_node_coverage_summary.json"


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
        choices=("train_and_save", "train_conditional_and_save", "load_and_infer"),
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
    parser.add_argument("--x_context_cache_dir", default=SlimConfig.x_context_cache_dir)
    parser.add_argument("--x_context_memmap_enabled", action="store_true")
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
    parser.add_argument("--phase3e_precompute_only", action="store_true")
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
        "--sspm_context_action_mode",
        choices=("raw_orthrus10", "legacy"),
        default=SlimConfig.sspm_context_action_mode,
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
    parser.add_argument("--synthetic_smoke", action="store_true")
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
            "or clearscope_android_v2",
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


def _handle_phase3e_precompute_only(
    args: argparse.Namespace,
    config: SlimConfig,
) -> Path:
    """Run Phase3E precompute-only path or fail before normal train/infer flow."""
    del args
    _validate_active_event_score_mode(config.event_score_mode)
    process_cfg = _load_process_config(config)
    db_cfg = _cfg_for_dataset(config.dataset)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    cur = None
    conn = None
    try:
        cur, conn = init_database_connection(db_cfg)
        split_metadata = _resolve_db_split_metadata(config, db_cfg)
        event_filter = use_event_type_filter(db_cfg)
        return run_phase3e_precompute_db(
            config=config,
            conn=conn,
            year_month=str(split_metadata["year_month"]),
            train_days=list(split_metadata["train_days"]),
            validation_days=list(split_metadata["validation_days"]),
            test_days=list(split_metadata["test_days"]),
            node_maps=None,
            event_filter=event_filter,
            process_cfg=process_cfg,
        )
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested slim causal semantics pipeline."""
    args = parse_args(argv)
    config = config_from_args(args)
    if str(getattr(args, "phase3g_backfill_result_dir", "")).strip():
        eval_path = _phase3g_backfill_result_reports_from_config(
            result_dir=Path(str(args.phase3g_backfill_result_dir)),
            config=config,
        )
    elif config.phase3e_precompute_only:
        eval_path = _handle_phase3e_precompute_only(args, config)
    elif (
        str(config.sspm_target_mode) == "node_action_semantic_mean"
        and str(config.sspm_train_mode) in {"train_and_save", "train_conditional_and_save"}
        and not config.synthetic_smoke
    ):
        eval_path = run_phase3e_train_base_from_precompute(config)
    elif (
        str(config.sspm_target_mode) == "node_action_semantic_mean"
        and str(config.sspm_train_mode) == "load_and_infer"
        and not config.synthetic_smoke
    ):
        eval_path = run_phase3e_load_and_infer_from_precompute(config)
    elif config.synthetic_smoke:
        eval_path = run_synthetic_smoke(config)
    else:
        eval_path = run_db(args, config)
    if config.print_summary:
        _compact_print_summary(eval_path)
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
