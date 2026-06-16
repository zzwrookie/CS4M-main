"""Shared imports, constants, lightweight state classes, and SlimConfig."""

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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.utils.common import ENTITY_TYPES, RELATIONS, information_flow, stable_hash
from cs4m.semantics.clearscope_android import (
    CLEARSCOPE_LEGACY_SEMANTIC_MODE,
    CLEARSCOPE_REFINED_SEMANTIC_MODE,
    CLEARSCOPE_SEMANTIC_RULES_VERSION,
    CLEARSCOPE_V3_SEMANTIC_MODE,
    CLEARSCOPE_V31_SEMANTIC_MODE,
    CLEARSCOPE_V32_SEMANTIC_MODE,
    CLEARSCOPE_V32_CACHE_ONLY_SEMANTIC_MODE,
    android_process_natural_tokens,
    android_process_natural_tokens_refined,
    clearscope_residual_text_refined,
    clearscope_residual_text_v3,
    clearscope_residual_text_v31,
    clearscope_residual_text_v32,
    clearscope_residual_text_v32_cache_only,
    clearscope_file_nll_role,
    clearscope_file_natural_tokens,
    clearscope_file_natural_tokens_refined,
    clearscope_file_natural_tokens_v3,
    clearscope_file_natural_tokens_v31,
    clearscope_file_natural_tokens_v32,
    clearscope_file_natural_tokens_v32_cache_only,
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
    cadets_semantic_mode_is_v3_safe_lexical,
    freebsd_file_natural_tokens,
    freebsd_file_natural_tokens_v3_safe_lexical,
    freebsd_file_nll_role,
    freebsd_netflow_natural_tokens,
    freebsd_netflow_natural_tokens_v3_safe_lexical,
    freebsd_netflow_nll_role,
    freebsd_process_natural_tokens,
    freebsd_process_natural_tokens_v3_safe_lexical,
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
from cs4m.phase3e.word2vec_adapter import ResidualWord2VecTokenAdapter
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
from cs4m.scoring.simple_gates import lambda_rho_for_event
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
from scripts.pipeline.io.db_stream import (
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
    "cadets_e4_v3_policy_smoke_v1",
    "clearscope_node_pair_v1",
    "clearscope_android_v2",
    "clearscope_v31_fp_guard_v1",
    "clearscope_v31_fp_guard_v2",
    "clearscope_v31_fp_guard_v3",
    "clearscope_v31_fp_guard_v3b",
    "clearscope_v31_fp_guard_v3c",
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
    "base_conf_v31_support",
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
SSPM_SCORE_HEADS = {"conditional_action_semantic"}
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
    "coverage_audit_fingerprint",
    "train_backend",
    "infer_backend",
    "node_embedding_path",
    "action_embedding_path",
    "event_index_path",
    "node_embedding_dim",
    "action_embedding_dim",
    "event_index_cache_mode",
    "event_index_debug_fields",
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
    "OPTC_051": {
        "year_month": "2019-09",
        "train": [19, 20, 21],
        "val": [22],
        "test": [23, 24, 25],
    },
    "OPTC_201": {
        "year_month": "2019-09",
        "train": [19, 20, 21],
        "val": [22],
        "test": [23, 24, 25],
    },
    "OPTC_501": {
        "year_month": "2019-09",
        "train": [19, 20, 21],
        "val": [22],
        "test": [23, 24, 25],
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
    "policy_support_reason",
    "policy_margin_used",
    "src_prior_alert_count",
    "dst_prior_alert_count",
    "src_prior_node_evidence_count",
    "dst_prior_node_evidence_count",
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

    def alert_count(self, node_idx: int) -> int:
        """Return the online alert count observed so far for one node."""
        state = self._nodes.get(int(node_idx))
        if state is None:
            return 0
        return int(state.get("alert_count", 0))

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
    sspm_train_mode: str = "load_and_infer"
    sspm_train_data_mode: str = "phase3e_memmap"
    sspm_checkpoint_path: str = ""
    sspm_base_checkpoint_root: str = "outputs/models/sspm_phase3d"
    sspm_train_batch_events: int = 8192
    sspm_target_mode: str = "node_action_semantic_mean"
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
    optc_netflow_node_canonicalization: str = "none"
    phase3e_base_profile: bool = False
    phase3e_profile_interval_events: int = 100000
    sspm_train_backend: str = "numpy"
    sspm_infer_backend: str = "numpy"
    sspm_score_head: str = "conditional_action_semantic"
    node_repr_fusion: str = "simple_mean"
    conditional_semantic_loss: str = "cosine"
    sspm_conditional_head_arch: str = CONDITIONAL_HEAD_ARCH_SHARED_V1
    action_head_checkpoint_path: str = ""
    action_validation_cache_dir: str = "outputs/cache/phase3g_action_validation/{DATASET}"
    conditional_group_min_count: int = 1000
    conditional_low_support_policy: str = CONDITIONAL_LOW_SUPPORT_POLICY_DEFAULT
    conditional_low_support_margin: float = CONDITIONAL_LOW_SUPPORT_MARGIN_DEFAULT
    conditional_unseen_group_policy: str = CONDITIONAL_UNSEEN_GROUP_POLICY_DEFAULT
    conditional_both_cold_unseen_policy: str = "alert"
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
    dual_channel_node_support_enabled: bool = False
    dual_channel_node_support_score_floor: float = 0.0
    dual_channel_node_support_threshold: int = 2
    dual_channel_node_support_channel: str = "event_semantic_node_support_ge_2"
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
    sspm_state_model: str = "s4d_complex_node"
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
    semantic_mode: str = CLEARSCOPE_V3_SEMANTIC_MODE
    word2vec_window: int = 3
    word2vec_min_count: int = 1
    word2vec_sg: int = 1
    word2vec_negative: int = 5
    word2vec_epochs: int = 10
    word2vec_workers: int = 4
    word2vec_seed: int = 0
    word2vec_oov_policy: str = "unk"
    word2vec_corpus_file_dir: str = ""
    theia_netflow_policy: str = "fixed"
    action_type_alert_policy: str = "default"
    node_pool_score_mode: str = "base_conf"
    node_pool_topk_values: str = "100,200,500,1000,1500,3000,5000,10000,20000,30000"
    clearscope_v31_fp_guard_write_floor: float = 0.085
    clearscope_v31_fp_guard_read_floor: float = 0.060
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
