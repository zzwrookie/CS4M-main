#!/usr/bin/env python3
"""Train-only causal semantic diagnostics for TFLR Phase 1/2.

Purpose:
  Fit a frozen tokenizer, role vocabulary, and field rarity/NLL proxy on the
  train split, calibrate on validation, then score the test stream causally.
  Phase 2 adds an online predictable-activity credit for repeated node-centric
  role/action/field patterns that are also supported by many prior nodes in
  the same coarse event pattern.
  Phase 3 adds positive attack evidence from temporal burst shape, causal edge
  novelty, and multi-hop episode support.
  This is a diagnostic substrate for the proposed TFLR-CMTE redesign; it is
  not part of the main detector unless later full validation supports it.

Inputs/outputs:
  Reads the same PostgreSQL event/node tables as the TFLR low-rank runner.
  Writes:
    - eval_causal_semantics.json
    - semantic_audit.json when --semantic_audit_enabled is set
    - online_node_alerts.csv
    - online_episode_alerts.csv
    - online_event_alerts.csv
    - online_confirmed_node_alerts.csv
    - online_confirmed_event_alerts.csv
    - causal_semantics_ranked.csv
    - causal_semantics_top_events.csv

How to enable/test:
  python scripts/tools/causal_semantics.py --dataset CLEARSCOPE_E3 \
    --out_tag CLEARSCOPE_E3_CAUSAL_SEMANTICS_SMOKE --max_train_events 20000 \
    --max_ref_events 10000 --max_test_events 10000

Runtime/memory:
  Streaming counters only. Memory is bounded by explicit vocabulary sizes,
  role-key limits, pair-key limits, active node arrays, and the predictable
  activity pattern-key limit, edge-key limit, per-node recent-peer cap, and
  per-node recent-chain cap. The optional two-stage confirmer keeps only
  pending candidate states inside a bounded time/event window.

Leakage risk:
  Tokenizer, roles, pair statistics, and calibration are fit from train and
  validation only. Test labels/GT nodes are loaded only for evaluation and
  reporting after the score stream is complete.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.config.provnet_utils import PROCESS_NODE_TYPE, init_database_connection, log
from scripts.data.get_dataset import (
    fetch_node_tables,
    get_dataset_splits,
    load_ground_truth_indices,
    parse_split_days,
    summarize_node_payload,
    tokenize_msg,
    use_event_type_filter,
)
from scripts.pipeline.io.db_stream import _cfg_for_dataset, _query_count, _stream_events
from scripts.pipeline.outputs.evaluation import best_sweep_row, best_under_fp_target, memory_snapshot, node_confusion_from_masks, rank_scores, target_for_dataset
from cs4m.utils.common import robust_stats
from legacy.baselines.lowrank import LowRankConfig, LowRankStreamModel


def _build_node_maps_and_summaries(
    netflow_nodes: list[tuple[Any, ...]],
    process_nodes: list[tuple[Any, ...]],
    file_nodes: list[tuple[Any, ...]],
) -> tuple[
    dict[int, tuple[str, str]],
    dict[str, str],
    dict[str, tuple[str, int]],
    dict[str, int],
]:
    """Build all node lookup tables from one DB node-table fetch.

    This avoids the previous duplicate full-table materialization from
    `fetch_node_tables()` plus `get_indexid2msg()`, which was a major RSS
    contributor on CADETS/THEIA.
    """
    indexid2summary: dict[int, tuple[str, str]] = {}
    hash2type: dict[str, str] = {}
    hash2uuid_index: dict[str, tuple[str, int]] = {}
    uuid2index: dict[str, int] = {}

    def add_node(uuid: Any, h: Any, index_id: Any, kind: str, payload: str) -> None:
        idx = int(index_id)
        hash_key = str(h)
        uuid_key = str(uuid)
        hash2type[hash_key] = str(kind)
        hash2uuid_index[hash_key] = (uuid_key, idx)
        uuid2index[uuid_key] = idx
        indexid2summary[idx] = (
            str(kind),
            " ".join(summarize_node_payload(str(kind), tokenize_msg(str(payload)))),
        )

    for uuid, h, src_addr, src_port, dst_addr, dst_port, index_id in netflow_nodes:
        remote = str(dst_addr or "")
        if os.getenv("CLAD_EVENT_NETFLOW_USE_PORT", "").strip().lower() in {"1", "true", "yes", "on"} and dst_port not in {None, ""}:
            remote = f"{remote}:{dst_port}"
        add_node(uuid, h, index_id, "netflow", remote)

    for uuid, h, path, cmd, index_id in process_nodes:
        payload = " ".join(part for part in [str(path or "").strip(), str(cmd or "").strip()] if part and part.lower() != "none")
        add_node(uuid, h, index_id, PROCESS_NODE_TYPE, payload)

    for uuid, h, path, index_id in file_nodes:
        add_node(uuid, h, index_id, "file", str(path or ""))

    return indexid2summary, hash2type, hash2uuid_index, uuid2index


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HEX_RE = re.compile(r"\b[0-9a-f]{12,}\b", flags=re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d+\b")
RESIDUAL_COMPONENTS = ("pred_mse", "semantic_ae_residual", "factor_residual", "state_cosine")
SEMANTIC_STRING_POLICIES = ("raw", "compressed", "compressed_v2", "no_identity", "grouped")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="CLEARSCOPE_E3")
    p.add_argument("--out_tag", default="")
    p.add_argument("--result_root", default="outputs/results/tflr_light")
    p.add_argument("--fetch_size", type=int, default=200000)
    p.add_argument("--max_train_events", type=int, default=0)
    p.add_argument("--max_ref_events", type=int, default=0)
    p.add_argument("--max_test_events", type=int, default=0)
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--min_token_count", type=int, default=2)
    p.add_argument("--max_tokens_per_node", type=int, default=12)
    p.add_argument(
        "--semantic_string_policy",
        choices=list(SEMANTIC_STRING_POLICIES),
        default="raw",
        help=(
            "Semantic identity policy for causal role/shape tokens. raw keeps the old "
            "role heads; compressed drops fine-grained process/file identity; "
            "compressed_v2 also separates coarse shape from role head to avoid "
            "duplicate shape/head token weighting; no_identity keeps only "
            "type/action/edge structure; grouped uses one shared stream model "
            "with struct/coarse/raw feature-group tokens instead of three "
            "separate multi-view residual heads."
        ),
    )
    p.add_argument("--semantic_audit_enabled", action="store_true")
    p.add_argument("--semantic_audit_top_k", type=int, default=100)
    p.add_argument("--max_role_keys", type=int, default=500000)
    p.add_argument("--max_pair_keys", type=int, default=750000)
    p.add_argument("--smoothing", type=float, default=0.5)
    p.add_argument("--role_min_count", type=int, default=20)
    p.add_argument("--role_activity_log_base", type=float, default=2.0)
    p.add_argument("--score_clip", type=float, default=16.0)
    p.add_argument("--node_score_mode", choices=["max", "tail_excess", "activity_adjusted", "predictable_activity", "attack_evidence"], default="activity_adjusted")
    p.add_argument("--predictable_activity_min_seen", type=int, default=16)
    p.add_argument("--predictable_activity_min_global_nodes", type=int, default=64)
    p.add_argument("--predictable_activity_weight", type=float, default=1.10)
    p.add_argument("--predictable_activity_power", type=float, default=1.25)
    p.add_argument("--predictable_activity_cap", type=float, default=6.0)
    p.add_argument("--predictable_activity_max_pattern_keys", type=int, default=1000000)
    p.add_argument("--predictable_activity_state_mode", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--attack_evidence_edge_weight", type=float, default=0.70)
    p.add_argument("--attack_evidence_burst_weight", type=float, default=0.55)
    p.add_argument("--attack_evidence_episode_weight", type=float, default=1.35)
    p.add_argument("--attack_evidence_chain_weight", type=float, default=0.80)
    p.add_argument("--attack_evidence_component_cap", type=float, default=6.0)
    p.add_argument("--attack_evidence_burst_halflife_seconds", type=float, default=60.0)
    p.add_argument("--attack_evidence_episode_halflife_seconds", type=float, default=180.0)
    p.add_argument("--attack_evidence_episode_window_seconds", type=float, default=300.0)
    p.add_argument("--attack_evidence_min_distinct_peers", type=int, default=4)
    p.add_argument("--attack_evidence_recent_peer_cap", type=int, default=128)
    p.add_argument("--attack_evidence_propagation_weight", type=float, default=0.25)
    p.add_argument("--attack_evidence_state_cap", type=float, default=32.0)
    p.add_argument("--attack_evidence_max_edge_keys", type=int, default=1000000)
    p.add_argument("--attack_evidence_edge_overflow_policy", choices=["not_novel", "old_drop"], default="not_novel")
    p.add_argument("--learned_evidence_composer_enabled", action="store_true")
    p.add_argument("--learned_evidence_composer_mode", choices=["robust_mahalanobis", "additive_autoencoder"], default="robust_mahalanobis")
    p.add_argument("--learned_evidence_composer_max_train_events", type=int, default=200000)
    p.add_argument("--learned_evidence_composer_ridge", type=float, default=1e-3)
    p.add_argument("--learned_evidence_composer_hidden_dim", type=int, default=4)
    p.add_argument("--learned_evidence_composer_epochs", type=int, default=35)
    p.add_argument("--learned_evidence_composer_lr", type=float, default=0.01)
    p.add_argument("--learned_evidence_composer_l2", type=float, default=1e-4)
    p.add_argument("--learned_evidence_composer_clip", type=float, default=32.0)
    p.add_argument("--disable_residual_fusion", action="store_true")
    p.add_argument("--disable_edge_novelty", action="store_true")
    p.add_argument("--disable_temporal_burst", action="store_true")
    p.add_argument("--disable_episode_support", action="store_true")
    p.add_argument("--disable_chain_diversity", action="store_true")
    p.add_argument("--disable_confirmation", action="store_true")
    p.add_argument("--max_online_calibration_events", type=int, default=0)
    p.add_argument("--online_threshold_min_tail_samples", type=int, default=25)
    p.add_argument("--online_node_alert_quantile", type=float, default=0.9999)
    p.add_argument("--online_episode_alert_quantile", type=float, default=0.9999)
    p.add_argument("--online_event_alert_quantile", type=float, default=0.9999)
    p.add_argument("--online_node_alert_threshold", type=float, default=0.0)
    p.add_argument("--online_episode_alert_threshold", type=float, default=0.0)
    p.add_argument("--online_event_alert_threshold", type=float, default=0.0)
    p.add_argument("--online_expected_alert_budget", type=int, default=0)
    p.add_argument("--online_expected_alert_horizon_events", type=int, default=0)
    p.add_argument("--online_max_node_alerts", type=int, default=200000)
    p.add_argument("--online_max_episode_alerts", type=int, default=200000)
    p.add_argument("--online_max_event_alerts", type=int, default=200000)
    p.add_argument("--online_confirm_window_seconds", type=float, default=300.0)
    p.add_argument("--online_confirm_window_events", type=int, default=200000)
    p.add_argument("--online_confirm_min_evidence_kinds", type=int, default=3)
    p.add_argument("--online_confirmed_node_quantile", type=float, default=0.99)
    p.add_argument("--online_confirmed_event_quantile", type=float, default=0.99)
    p.add_argument("--online_confirmed_node_threshold", type=float, default=0.0)
    p.add_argument("--online_confirmed_event_threshold", type=float, default=0.0)
    p.add_argument("--online_max_confirmed_node_alerts", type=int, default=200000)
    p.add_argument("--online_max_confirmed_event_alerts", type=int, default=200000)
    p.add_argument(
        "--online_confirm_node_candidate_policy",
        choices=["episode_all", "candidate_only", "candidate_or_strong"],
        default="episode_all",
        help=(
            "Which nodes can receive bounded-delay confirmed-node alerts. "
            "episode_all preserves the old behavior; candidate_only confirms "
            "only nodes that already crossed the immediate node threshold; "
            "candidate_or_strong also allows non-candidate endpoints with strong "
            "multi-evidence endpoint risk."
        ),
    )
    p.add_argument("--online_confirm_node_strong_risk_ratio", type=float, default=1.05)
    p.add_argument("--online_node_alert_policy", choices=["one_shot", "stateful_update"], default="stateful_update")
    p.add_argument("--online_node_update_gain_ratio", type=float, default=0.15)
    p.add_argument("--stream_alert_csv", dest="stream_alert_csv", action="store_true", default=True)
    p.add_argument("--no_stream_alert_csv", dest="stream_alert_csv", action="store_false")
    p.add_argument("--online_score_trace_enabled", action="store_true")
    p.add_argument("--online_score_trace_top_n", type=int, default=3000)
    p.add_argument("--residual_fusion_enabled", action="store_true")
    p.add_argument("--residual_semantic_weight", type=float, default=1.0)
    p.add_argument("--residual_fusion_weight", type=float, default=1.0)
    p.add_argument("--residual_fusion_clip", type=float, default=8.0)
    p.add_argument("--residual_latent_dim", type=int, default=32)
    p.add_argument("--residual_rank", type=int, default=8)
    p.add_argument("--residual_semantic_buckets", type=int, default=65536)
    p.add_argument("--residual_identity_buckets", type=int, default=65536)
    p.add_argument("--residual_max_tokens", type=int, default=96)
    p.add_argument("--residual_max_train_samples", type=int, default=200000)
    p.add_argument("--residual_vector_cache_size", type=int, default=200000)
    p.add_argument("--residual_ridge_lambda", type=float, default=1e-2)
    p.add_argument("--residual_direct_lowrank_enabled", action="store_true")
    p.add_argument("--residual_direct_lowrank_iters", type=int, default=5)
    p.add_argument("--residual_direct_lowrank_l2", type=float, default=1e-2)
    p.add_argument(
        "--multi_view_behavior_enabled",
        action="store_true",
        help=(
            "Experimental train-only multi-view residual behavior model. It fits "
            "separate raw/compressed/no_identity low-rank predictors and uses "
            "validation-only one-class calibration to produce the main streaming "
            "event risk. Existing rule evidence can still be emitted as explanation."
        ),
    )
    p.add_argument("--multi_view_behavior_policies", default="raw,compressed,no_identity")
    p.add_argument("--multi_view_behavior_clip", type=float, default=16.0)
    p.add_argument("--multi_view_behavior_ridge", type=float, default=1e-3)
    p.add_argument("--multi_view_behavior_store_all_components", action="store_true")
    p.add_argument("--memory_trim_enabled", dest="memory_trim_enabled", action="store_true", default=True)
    p.add_argument("--no_memory_trim", dest="memory_trim_enabled", action="store_false")
    p.add_argument(
        "--online_risk_composition",
        choices=["weighted_evidence", "score_only"],
        default="weighted_evidence",
        help=(
            "weighted_evidence preserves the old hand-weighted rule/evidence risk. "
            "score_only uses the learned/base event score as online_risk while still "
            "calculating rule evidence for explanations and bounded confirmation."
        ),
    )
    p.add_argument(
        "--ablation_mode",
        choices=[
            "CUSTOM",
            "SEM_ONLY",
            "RES_ONLY",
            "SEM_RES",
            "EDGE_ONLY",
            "EDGE_BURST",
            "EDGE_EPISODE",
            "EDGE_CHAIN",
            "FULL_EVIDENCE",
            "FULL_CONFIRM",
        ],
        default="CUSTOM",
    )
    p.add_argument(
        "--synthetic_smoke",
        action="store_true",
        help="Run a no-database synthetic smoke path that writes core output files.",
    )
    p.add_argument("--topk_values", default="100,200,300,400,433,500,700,900,1000,1200,1500,1800,2000,2200,2500,3000,3500,4000,5000,7000,10000,15000,18000,20000")
    p.add_argument("--top_events", type=int, default=5000)
    p.add_argument("--print_summary", action="store_true")
    return p.parse_args()


def _configure_ablation(args: argparse.Namespace) -> None:
    mode = str(getattr(args, "ablation_mode", "CUSTOM"))
    args.online_confirmation_enabled = not bool(getattr(args, "disable_confirmation", False))
    args.attack_evidence_edge_enabled = not bool(getattr(args, "disable_edge_novelty", False))
    args.attack_evidence_burst_enabled = not bool(getattr(args, "disable_temporal_burst", False))
    args.attack_evidence_episode_enabled = not bool(getattr(args, "disable_episode_support", False))
    args.attack_evidence_chain_enabled = not bool(getattr(args, "disable_chain_diversity", False))
    if bool(getattr(args, "disable_residual_fusion", False)):
        args.residual_fusion_enabled = False
    if mode == "CUSTOM":
        args.ablation_effective_mode = "CUSTOM"
        return

    args.ablation_effective_mode = mode
    args.residual_semantic_weight = 1.0
    args.residual_fusion_enabled = mode != "SEM_ONLY"
    args.online_confirmation_enabled = mode == "FULL_CONFIRM"
    args.attack_evidence_edge_enabled = mode in {
        "EDGE_ONLY",
        "EDGE_BURST",
        "EDGE_EPISODE",
        "EDGE_CHAIN",
        "FULL_EVIDENCE",
        "FULL_CONFIRM",
    }
    args.attack_evidence_burst_enabled = mode in {"EDGE_BURST", "FULL_EVIDENCE", "FULL_CONFIRM"}
    args.attack_evidence_episode_enabled = mode in {"EDGE_EPISODE", "FULL_EVIDENCE", "FULL_CONFIRM"}
    args.attack_evidence_chain_enabled = mode in {"EDGE_CHAIN", "FULL_EVIDENCE", "FULL_CONFIRM"}
    if mode == "RES_ONLY":
        args.residual_semantic_weight = 0.0
    if mode in {"SEM_ONLY", "RES_ONLY", "SEM_RES"}:
        args.node_score_mode = "max"
    else:
        args.node_score_mode = "attack_evidence"


def _predictable_activity_state_enabled(args: argparse.Namespace) -> bool:
    mode = str(getattr(args, "predictable_activity_state_mode", "auto"))
    if mode == "on":
        return True
    if mode == "off":
        return False
    return str(getattr(args, "node_score_mode", "")) == "predictable_activity"


def _disabled_threshold(name: str) -> dict[str, float | int | str]:
    return {
        "source": f"{name}_disabled",
        "threshold": float("inf"),
        "quantile": 1.0,
        "validation_samples": 0,
    }


def _stable_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8", errors="ignore")).hexdigest()[:16]


def _score_quantiles(scores: list[float]) -> dict[str, float]:
    finite = [float(x) for x in scores if math.isfinite(float(x))]
    arr = np.asarray(finite, dtype=np.float32)
    if arr.size == 0:
        return {}
    return {
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "q999": float(np.quantile(arr, 0.999)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _online_calibration_limit(args: argparse.Namespace) -> int:
    value = int(getattr(args, "max_online_calibration_events", 0))
    if value >= 0:
        return value
    return int(args.max_ref_events)


def _upper_z_scalar(value: float, stats: dict[str, float], clip: float) -> float:
    scale = max(float(stats.get("mad", 1.0)) * 1.4826, 1e-6)
    out = max((float(value) - float(stats.get("median", 0.0))) / scale, 0.0)
    if float(clip) > 0.0:
        out = min(out, float(clip))
    return float(out)


def _residual_component_values(aux: dict[str, float]) -> np.ndarray:
    return np.asarray([float(aux.get(name, 0.0)) for name in RESIDUAL_COMPONENTS], dtype=np.float32)


def _trim_process_memory() -> bool:
    try:
        gc.collect()
        libc = ctypes.CDLL("libc.so.6")
        return int(libc.malloc_trim(0)) == 1
    except Exception:
        return False


def _new_residual_model(args: argparse.Namespace) -> LowRankStreamModel:
    return LowRankStreamModel(
        LowRankConfig(
            latent_dim=int(args.residual_latent_dim),
            rank=int(args.residual_rank),
            semantic_buckets=int(args.residual_semantic_buckets),
            identity_buckets=int(args.residual_identity_buckets),
            max_tokens=int(args.residual_max_tokens),
            include_bigrams=False,
            vector_cache_size=int(args.residual_vector_cache_size),
            max_train_samples=int(args.residual_max_train_samples),
            ridge_lambda=float(args.residual_ridge_lambda),
            direct_lowrank_enabled=bool(args.residual_direct_lowrank_enabled),
            direct_lowrank_iters=int(args.residual_direct_lowrank_iters),
            direct_lowrank_l2=float(args.residual_direct_lowrank_l2),
        )
    )


def _new_residual_state(model: LowRankStreamModel) -> dict[str, Any]:
    return {
        "states": {},
        "last_seen": {},
        "last_edge_seen": {},
        "global_state": model._zero_state(),
        "last_global_ts": -1,
    }


def _residual_score_pre_update(
    model: LowRankStreamModel,
    state: dict[str, Any],
    row: dict[str, Any],
    stats: list[dict[str, float]] | None,
    clip: float,
) -> tuple[np.ndarray, dict[str, float], float]:
    z, aux, _pred = model.score_pre_update(
        row,
        state["states"],
        state["last_seen"],
        state["last_edge_seen"],
        state["global_state"],
        int(state["last_global_ts"]),
    )
    values = _residual_component_values(aux)
    component_scores: dict[str, float] = {}
    residual_score = 0.0
    if stats:
        total = 0.0
        used = 0
        for idx, name in enumerate(RESIDUAL_COMPONENTS):
            score = _upper_z_scalar(float(values[idx]), stats[idx], float(clip))
            component_scores[f"residual_{name}_z"] = float(score)
            total += score
            used += 1
        residual_score = float(total / max(used, 1))
    state["global_state"] = model.update_states(
        row,
        z,
        state["states"],
        state["last_seen"],
        state["last_edge_seen"],
        state["global_state"],
    )
    state["last_global_ts"] = int(row["timestamp_ns"])
    return values, component_scores, float(residual_score)


def _apply_residual_fusion(
    args: argparse.Namespace,
    causal_score: float,
    detail: dict[str, Any],
    residual_model: LowRankStreamModel | None,
    residual_state: dict[str, Any] | None,
    residual_stats: list[dict[str, float]],
    row: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    out = dict(detail)
    out["causal_semantic_score"] = float(causal_score)
    if not bool(args.residual_fusion_enabled) or residual_model is None or residual_state is None or not residual_stats:
        out["residual_fusion_score"] = 0.0
        effective_score = float(getattr(args, "residual_semantic_weight", 1.0)) * float(causal_score)
        out["effective_event_score"] = float(effective_score)
        return float(effective_score), out
    residual_row = _residual_row_view(row, args)
    _values, component_scores, residual_score = _residual_score_pre_update(
        residual_model,
        residual_state,
        residual_row,
        residual_stats,
        float(args.residual_fusion_clip),
    )
    effective_score = (
        float(getattr(args, "residual_semantic_weight", 1.0)) * float(causal_score)
        + float(args.residual_fusion_weight) * float(residual_score)
    )
    out.update(component_scores)
    out["residual_fusion_score"] = float(residual_score)
    out["effective_event_score"] = float(effective_score)
    return float(effective_score), out


def _apply_event_behavior_score(
    args: argparse.Namespace,
    causal_score: float,
    detail: dict[str, Any],
    row: dict[str, Any],
    residual_model: LowRankStreamModel | None,
    residual_state: dict[str, Any] | None,
    residual_stats: list[dict[str, float]],
    multi_view_model: "MultiViewBehaviorModel | None",
    multi_view_state: dict[str, dict[str, Any]] | None,
) -> tuple[float, dict[str, Any]]:
    score, out = _apply_residual_fusion(
        args,
        float(causal_score),
        dict(detail),
        residual_model,
        residual_state,
        residual_stats,
        row,
    )
    if bool(args.multi_view_behavior_enabled) and multi_view_model is not None and multi_view_state is not None:
        multi_view_score, multi_view_detail = multi_view_model.score_pre_update(row, multi_view_state)
        out.update(multi_view_detail)
        out["base_effective_event_score"] = float(score)
        out["multi_view_behavior_score"] = float(multi_view_score)
        out["effective_event_score"] = float(multi_view_score)
        score = float(multi_view_score)
    return float(score), out


def _parse_multi_view_policies(value: str) -> list[str]:
    policies: list[str] = []
    seen: set[str] = set()
    for raw in str(value or "").split(","):
        item = raw.strip()
        if not item:
            continue
        policy = _semantic_policy(item)
        if policy in seen:
            continue
        policies.append(policy)
        seen.add(policy)
    return policies or ["raw", "compressed", "no_identity"]


class MultiViewBehaviorModel:
    """Train-only multi-view residual normality model.

    Purpose:
      Replace manual evidence-weight composition with a compact learned
      behavior score. Each view maps the same event to a different semantic
      text policy, fits a low-rank next-event predictor on train, and produces
      validation-normalized residual components at inference time.

    Leakage/runtime:
      Models are fit on train only. Residual component normalization and
      Mahalanobis fusion are calibrated on validation only. Test scoring is
      causal: each event is scored before updating per-view node/global state.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.policies = _parse_multi_view_policies(str(args.multi_view_behavior_policies))
        self.models: dict[str, LowRankStreamModel] = {policy: _new_residual_model(args) for policy in self.policies}
        self.component_names: list[str] = [f"{policy}_{name}" for policy in self.policies for name in RESIDUAL_COMPONENTS]
        self.component_stats: list[dict[str, float]] = []
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.inv_cov: np.ndarray | None = None
        self.train_summaries: dict[str, dict[str, Any]] = {}
        self.ref_processed = 0
        self.fit_seconds = 0.0
        self.calibration_seconds = 0.0

    def fit(
        self,
        conn: Any,
        year_month: str,
        train_days: list[int],
        indexid2summary: dict[int, tuple[str, str]],
        hash2type: dict[str, str],
        hash2uuid_index: dict[str, tuple[str, int]],
        event_filter: bool,
        fetch_size: int,
        max_train_events: int,
    ) -> dict[str, Any]:
        phase = time.perf_counter()
        for policy, view_model in self.models.items():
            log(f"[CausalSemantics] fitting multi-view behavior residual head policy={policy}")
            self.train_summaries[policy] = view_model.fit_rows(
                (
                    _row_with_semantic_view(row, int(self.args.max_tokens_per_node), policy)
                    for row in _stream_events(
                        conn,
                        year_month,
                        train_days,
                        indexid2summary,
                        hash2type,
                        hash2uuid_index,
                        set(),
                        event_filter,
                        int(fetch_size),
                        int(max_train_events),
                    )
                )
            )
            if bool(getattr(self.args, "memory_trim_enabled", True)):
                _trim_process_memory()
        self.fit_seconds = float(time.perf_counter() - phase)
        return self.summary()

    def new_state(self) -> dict[str, dict[str, Any]]:
        return {policy: _new_residual_state(model) for policy, model in self.models.items()}

    def _score_vector_pre_update(
        self,
        row: dict[str, Any],
        state: dict[str, dict[str, Any]],
        stats: list[dict[str, float]] | None,
        clip: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        raw_values: list[float] = []
        detail: dict[str, Any] = {}
        idx = 0
        view_data = _semantic_view_fields_and_texts(row, int(self.args.max_tokens_per_node), self.policies)
        for policy in self.policies:
            model = self.models[policy]
            policy_state = state[policy]
            _fields, view_text = view_data[policy]
            view_row = _row_with_text_view(row, view_text)
            values, _component_scores, _residual_score = _residual_score_pre_update(
                model,
                policy_state,
                view_row,
                None,
                float(clip),
            )
            for component_idx, name in enumerate(RESIDUAL_COMPONENTS):
                raw_value = float(values[component_idx])
                raw_values.append(raw_value)
                if bool(getattr(self.args, "multi_view_behavior_store_all_components", False)):
                    detail[f"multi_view_{policy}_{name}"] = raw_value
                if stats and bool(getattr(self.args, "multi_view_behavior_store_all_components", False)):
                    z_value = _upper_z_scalar(raw_value, stats[idx], float(clip))
                    detail[f"multi_view_{policy}_{name}_z"] = float(z_value)
                idx += 1
        return np.asarray(raw_values, dtype=np.float32), detail

    def calibrate(
        self,
        conn: Any,
        year_month: str,
        val_days: list[int],
        indexid2summary: dict[int, tuple[str, str]],
        hash2type: dict[str, str],
        hash2uuid_index: dict[str, tuple[str, int]],
        event_filter: bool,
        fetch_size: int,
        max_ref_events: int,
    ) -> dict[str, Any]:
        phase = time.perf_counter()
        ref_state = self.new_state()
        vectors: list[np.ndarray] = []
        for row in _stream_events(
            conn,
            year_month,
            val_days,
            indexid2summary,
            hash2type,
            hash2uuid_index,
            set(),
            event_filter,
            int(fetch_size),
            int(max_ref_events),
        ):
            vector, _detail = self._score_vector_pre_update(
                row,
                ref_state,
                None,
                float(self.args.multi_view_behavior_clip),
            )
            vectors.append(vector)
            self.ref_processed += 1
            if self.ref_processed % 1_000_000 == 0:
                log(f"[CausalSemantics] multi-view behavior validation processed={self.ref_processed}")
        if vectors:
            matrix = np.stack(vectors, axis=0).astype(np.float32, copy=False)
            self.component_stats = [robust_stats(matrix[:, idx]) for idx in range(matrix.shape[1])]
            z_matrix = np.zeros_like(matrix, dtype=np.float32)
            for idx in range(matrix.shape[1]):
                for row_idx in range(matrix.shape[0]):
                    z_matrix[row_idx, idx] = np.float32(
                        _upper_z_scalar(float(matrix[row_idx, idx]), self.component_stats[idx], float(self.args.multi_view_behavior_clip))
                    )
            self.center = np.mean(z_matrix, axis=0).astype(np.float32)
            cov = np.cov(z_matrix.astype(np.float64), rowvar=False)
            cov = np.atleast_2d(cov).astype(np.float32, copy=False)
            cov = cov + np.eye(cov.shape[0], dtype=np.float32) * max(float(self.args.multi_view_behavior_ridge), 1e-6)
            self.scale = (np.std(z_matrix, axis=0) + 1e-6).astype(np.float32)
            self.inv_cov = np.linalg.pinv(cov).astype(np.float32, copy=False)
            del matrix
            del z_matrix
            del cov
        else:
            dim = len(self.component_names)
            self.component_stats = [robust_stats(np.zeros((1,), dtype=np.float32)) for _ in range(dim)]
            self.center = np.zeros((dim,), dtype=np.float32)
            self.scale = np.ones((dim,), dtype=np.float32)
            self.inv_cov = np.eye(dim, dtype=np.float32)
        self.calibration_seconds = float(time.perf_counter() - phase)
        del vectors
        if bool(getattr(self.args, "memory_trim_enabled", True)):
            _trim_process_memory()
        return self.summary()

    def score_pre_update(
        self,
        row: dict[str, Any],
        state: dict[str, dict[str, Any]],
    ) -> tuple[float, dict[str, float | str]]:
        vector, detail = self._score_vector_pre_update(
            row,
            state,
            self.component_stats,
            float(self.args.multi_view_behavior_clip),
        )
        if not self.component_stats:
            score = 0.0
            detail["multi_view_behavior_score"] = 0.0
            detail["multi_view_behavior_mode"] = "uncalibrated"
            return score, detail
        z_values = np.asarray(
            [
                _upper_z_scalar(float(vector[idx]), self.component_stats[idx], float(self.args.multi_view_behavior_clip))
                for idx in range(vector.shape[0])
            ],
            dtype=np.float32,
        )
        center = self.center if self.center is not None else np.zeros_like(z_values)
        inv_cov = self.inv_cov if self.inv_cov is not None else np.eye(z_values.shape[0], dtype=np.float32)
        delta = (z_values - center).astype(np.float32, copy=False)
        weighted = inv_cov @ delta
        contrib = np.maximum(delta * weighted, 0.0).astype(np.float32, copy=False)
        score = float(np.mean(contrib)) if contrib.size else 0.0
        if float(self.args.multi_view_behavior_clip) > 0.0:
            score = float(min(max(score, 0.0), float(self.args.multi_view_behavior_clip)))
        best_idx = int(np.argmax(contrib)) if contrib.size else -1
        if bool(getattr(self.args, "multi_view_behavior_store_all_components", False)):
            for idx, name in enumerate(self.component_names):
                detail[f"multi_view_{name}_contrib"] = float(contrib[idx])
        top_items = sorted(
            (
                {"component": name, "value": float(contrib[idx])}
                for idx, name in enumerate(self.component_names)
            ),
            key=lambda item: (-float(item["value"]), str(item["component"])),
        )[:5]
        detail.update(
            {
                "multi_view_behavior_score": float(score),
                "multi_view_behavior_mode": "validation_mahalanobis",
                "multi_view_behavior_policies": ",".join(self.policies),
                "multi_view_top_component": self.component_names[best_idx] if best_idx >= 0 else "",
                "multi_view_top_contrib": float(contrib[best_idx]) if best_idx >= 0 else 0.0,
                "multi_view_top_contribs": top_items,
            }
        )
        return float(score), detail

    @property
    def model_size_bytes(self) -> int:
        total = sum(int(model.model_size_bytes) for model in self.models.values())
        for arr in [self.center, self.scale, self.inv_cov]:
            if arr is not None:
                total += int(np.asarray(arr).nbytes)
        return int(total)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "policies": list(self.policies),
            "component_names": list(self.component_names),
            "train_summaries": self.train_summaries,
            "ref_processed": int(self.ref_processed),
            "fit_seconds": float(self.fit_seconds),
            "calibration_seconds": float(self.calibration_seconds),
            "model_size_bytes": int(self.model_size_bytes),
            "model_size_mb": float(self.model_size_bytes / (1024.0 * 1024.0)),
            "vector_cache_sizes": {policy: int(len(model._vector_cache)) for policy, model in self.models.items()},
            "fusion": "validation_mahalanobis_over_per_view_residual_z_scores",
            "leakage_check": "all view heads fit on train only; fusion statistics fit on validation only; labels/test rankings/full-test statistics are not used",
        }


def _normalize_piece(text: str, max_len: int = 32) -> str:
    value = str(text).strip().lower()
    if not value:
        return "na"
    value = _IP_RE.sub(" ip ", value)
    value = _HEX_RE.sub(" hex ", value)
    value = _NUM_RE.sub(" num ", value)
    value = _NON_ALNUM_RE.sub("_", value).strip("_")
    if not value:
        return "na"
    return value[:max_len].strip("_") or "na"


def _tokenize_summary(text: str, max_tokens: int) -> list[str]:
    out: list[str] = []
    for raw in str(text).split():
        token = _normalize_piece(raw, max_len=40)
        if token and token != "na":
            out.append(token)
        if len(out) >= int(max_tokens):
            break
    return out or ["na"]


def _semantic_policy(policy: str) -> str:
    value = str(policy or "raw").strip().lower()
    if value not in SEMANTIC_STRING_POLICIES:
        raise ValueError(f"Unsupported semantic_string_policy: {policy}")
    return value


def _action_family(action: str) -> str:
    text = str(action).upper()
    if "READ" in text or "RECV" in text:
        return "read_recv"
    if "WRITE" in text or "SEND" in text:
        return "write_send"
    if "EXEC" in text or "CLONE" in text:
        return "execute_clone"
    if "CONNECT" in text:
        return "connect"
    if "OPEN" in text:
        return "open"
    return _normalize_piece(text, max_len=24)


def _bucket_count(value: int, base: float = 2.0) -> str:
    n = max(0, int(value))
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    base = max(float(base), 1.1)
    power = int(math.floor(math.log(max(n, 1), base)))
    lo = int(base**power)
    hi = int(base ** (power + 1) - 1)
    return f"{lo}_{hi}"


def _base_row_fields(
    row: dict[str, Any],
    max_tokens_per_node: int,
    semantic_string_policy: str = "raw",
) -> dict[str, str]:
    semantic_policy = _semantic_policy(semantic_string_policy)
    field_policy = "compressed_v2" if semantic_policy == "grouped" else semantic_policy
    src_kind = _normalize_piece(row.get("src_kind", "unknown"), max_len=24)
    dst_kind = _normalize_piece(row.get("dst_kind", "unknown"), max_len=24)
    action = _normalize_piece(row.get("action", "EVENT_UNKNOWN"), max_len=40)
    action_family = _action_family(action)
    object_type = _normalize_piece(row.get("object_type", "unknown"), max_len=24)
    relation = f"rel_{int(row.get('relation_id', -1))}"
    src_tokens = _tokenize_summary(str(row.get("src_summary", "")), int(max_tokens_per_node))
    dst_tokens = _tokenize_summary(str(row.get("dst_summary", "")), int(max_tokens_per_node))
    src_shape = _node_shape(src_kind, src_tokens, field_policy)
    dst_shape = _node_shape(dst_kind, dst_tokens, field_policy)
    src_head = _node_role_head(src_kind, src_tokens, field_policy)
    dst_head = _node_role_head(dst_kind, dst_tokens, field_policy)
    return {
        "action": action,
        "action_family": action_family,
        "src_kind": src_kind,
        "dst_kind": dst_kind,
        "object_type": object_type,
        "relation": relation,
        "src_shape": src_shape,
        "dst_shape": dst_shape,
        "src_head": src_head,
        "dst_head": dst_head,
    }


def _row_fields(
    row: dict[str, Any],
    max_tokens_per_node: int,
    semantic_string_policy: str = "raw",
) -> dict[str, Any]:
    semantic_policy = _semantic_policy(semantic_string_policy)
    fields = _base_row_fields(row, int(max_tokens_per_node), semantic_policy)
    if semantic_policy == "grouped":
        fields["__grouped_tokens__"] = _grouped_semantic_tokens(row, int(max_tokens_per_node))
    return fields


def _semantic_view_text(
    row: dict[str, Any],
    max_tokens_per_node: int,
    semantic_string_policy: str,
    fields: dict[str, str] | None = None,
) -> str:
    if _semantic_policy(semantic_string_policy) == "grouped":
        return _grouped_semantic_view_text(row, int(max_tokens_per_node))
    if fields is None:
        fields = _row_fields(row, int(max_tokens_per_node), semantic_string_policy)
    tokens = _event_tokens(fields, semantic_string_policy)
    return " ".join(tokens)


def _semantic_view_fields_and_texts(
    row: dict[str, Any],
    max_tokens_per_node: int,
    policies: list[str],
) -> dict[str, tuple[dict[str, str], str]]:
    out: dict[str, tuple[dict[str, str], str]] = {}
    for policy in policies:
        fields = _row_fields(row, int(max_tokens_per_node), policy)
        out[policy] = (fields, _semantic_view_text(row, int(max_tokens_per_node), policy, fields))
    return out


def _row_with_text_view(row: dict[str, Any], view_text: str) -> dict[str, Any]:
    view_row = dict(row)
    view_row["text"] = str(view_text)
    return view_row


def _row_with_semantic_view(row: dict[str, Any], max_tokens_per_node: int, semantic_string_policy: str) -> dict[str, Any]:
    fields = _row_fields(row, int(max_tokens_per_node), semantic_string_policy)
    return _row_with_text_view(
        row,
        _semantic_view_text(row, int(max_tokens_per_node), semantic_string_policy, fields),
    )


def _residual_row_view(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if _semantic_policy(str(args.semantic_string_policy)) == "grouped":
        return _row_with_semantic_view(row, int(args.max_tokens_per_node), "grouped")
    return row


def _prefixed_tokens(prefix: str, tokens: list[str]) -> list[str]:
    return [f"{prefix}{token}" for token in tokens]


def _grouped_semantic_tokens(row: dict[str, Any], max_tokens_per_node: int) -> list[str]:
    """One-view, multi-granularity event tokens.

    The grouped policy keeps a single streaming model but exposes three
    auditable feature groups: structure, coarse identity, and raw identity. It
    avoids the runtime/RSS cost of three separate residual heads while keeping
    the raw/coarse/no-id signals available to the same encoder.
    """
    raw_fields = _base_row_fields(row, int(max_tokens_per_node), "raw")
    coarse_fields = _base_row_fields(row, int(max_tokens_per_node), "compressed_v2")
    struct_fields = _base_row_fields(row, int(max_tokens_per_node), "no_identity")

    struct_tokens = [
        f"act:{struct_fields['action']}",
        f"af:{struct_fields['action_family']}",
        f"sk:{struct_fields['src_kind']}",
        f"dk:{struct_fields['dst_kind']}",
        f"ot:{struct_fields['object_type']}",
        f"rel:{struct_fields['relation']}",
        f"ss:{struct_fields['src_shape']}",
        f"ds:{struct_fields['dst_shape']}",
    ]
    coarse_tokens = [
        f"ss:{coarse_fields['src_shape']}",
        f"ds:{coarse_fields['dst_shape']}",
        f"sh:{coarse_fields['src_head']}",
        f"dh:{coarse_fields['dst_head']}",
    ]
    raw_tokens = [
        f"sh:{raw_fields['src_head']}",
        f"dh:{raw_fields['dst_head']}",
        f"ss:{raw_fields['src_shape']}",
        f"ds:{raw_fields['dst_shape']}",
    ]
    tokens = (
        _prefixed_tokens("struct:", struct_tokens)
        + _prefixed_tokens("coarse:", coarse_tokens)
        + _prefixed_tokens("raw:", raw_tokens)
    )
    return list(dict.fromkeys(tokens))


def _grouped_semantic_view_text(row: dict[str, Any], max_tokens_per_node: int) -> str:
    return " ".join(_grouped_semantic_tokens(row, int(max_tokens_per_node)))


def _port_bucket(port_token: str) -> str:
    text = str(port_token).lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return "port_na"
    port = int(digits)
    if port <= 0:
        return "port_0"
    if port < 1024:
        return "port_system"
    if port < 49152:
        return "port_registered"
    return "port_ephemeral"


def _ip_scope(token: str) -> str:
    text = str(token).lower()
    if "ip_127_" in text or "localhost" in text:
        return "ip_loopback"
    if "ip_10_" in text or "ip_192_168_" in text or "ip_172_" in text:
        return "ip_private"
    if text.startswith("ip_0") or text in {"ip", "ip_0"}:
        return "ip_zero"
    if text.startswith("ip_"):
        return "ip_public"
    return "ip_unknown"


def _node_role_head(kind: str, tokens: list[str], semantic_string_policy: str = "raw") -> str:
    kind = str(kind)
    semantic_policy = _semantic_policy(semantic_string_policy)
    if semantic_policy == "no_identity":
        return "identity_removed"
    if kind == "netflow":
        ip = next((tok for tok in tokens if tok.startswith("ip")), "ip_unknown")
        port = next((tok for tok in tokens if tok.startswith("port")), "port_na")
        return f"{_ip_scope(ip)}|{_port_bucket(port)}"
    if kind == "file":
        root = next((tok for tok in tokens if tok.startswith("root_")), "")
        directory = next((tok for tok in tokens if tok.startswith("dir_")), "")
        if semantic_policy in {"compressed", "compressed_v2"}:
            return "|".join(tok for tok in [root, directory] if tok) or "file|coarse_path"
        path = next((tok for tok in tokens if tok.startswith("path_")), tokens[0] if tokens else "path_na")
        return "|".join(tok for tok in [root, directory, path] if tok) or "path_na"
    if kind == "process":
        directory = next((tok for tok in tokens if tok.startswith("dir_")), "")
        if semantic_policy in {"compressed", "compressed_v2"}:
            return directory or "coarse_process"
        exe = next((tok for tok in tokens if tok.startswith("exe_")), tokens[0] if tokens else "proc_na")
        return "|".join(tok for tok in [exe, directory] if tok)
    if semantic_policy in {"compressed", "compressed_v2"}:
        return "coarse"
    return tokens[0] if tokens else "na"


def _node_shape(kind: str, tokens: list[str], semantic_string_policy: str = "raw") -> str:
    semantic_policy = _semantic_policy(semantic_string_policy)
    head = _node_role_head(kind, tokens, semantic_policy)
    if kind == "process" and semantic_policy == "raw":
        flags = [tok for tok in tokens if tok.startswith("flag_")][:2]
        return "|".join([head, *flags])
    if semantic_policy == "compressed_v2":
        if kind == "process":
            return "process_shape"
        if kind == "file":
            root = next((tok for tok in tokens if tok.startswith("root_")), "")
            return f"file_shape|{root}" if root else "file_shape"
        if kind == "netflow":
            return "netflow_shape"
        return f"{kind}_shape" if kind else "unknown_shape"
    return head


def _event_tokens(fields: dict[str, str], semantic_string_policy: str = "raw") -> list[str]:
    if _semantic_policy(semantic_string_policy) == "grouped" and "__grouped_tokens__" in fields:
        return list(fields["__grouped_tokens__"])  # type: ignore[arg-type]
    tokens = [
        f"act:{fields['action']}",
        f"af:{fields['action_family']}",
        f"sk:{fields['src_kind']}",
        f"dk:{fields['dst_kind']}",
        f"ot:{fields['object_type']}",
        f"rel:{fields['relation']}",
        f"ss:{fields['src_shape']}",
        f"ds:{fields['dst_shape']}",
        f"sh:{fields['src_head']}",
        f"dh:{fields['dst_head']}",
    ]
    if _semantic_policy(semantic_string_policy) == "compressed_v2":
        return list(dict.fromkeys(tokens))
    return tokens


def _semantic_token_category(token: str) -> str:
    text = str(token).lower()
    if text.startswith("struct:"):
        return "group_struct"
    if text.startswith("coarse:"):
        return "group_coarse_identity"
    if text.startswith("raw:"):
        return "group_raw_identity"
    payload = text.split(":", 1)[1] if ":" in text else text
    if "identity_removed" in payload:
        return "identity_removed"
    if text.startswith(("act:", "af:")):
        return "action"
    if text.startswith("rel:"):
        return "relation"
    if text.startswith(("sk:", "dk:", "ot:")):
        return "type_role"
    if "ip_" in payload or "port_" in payload or "netflow" in payload:
        return "network_identity"
    if "coarse_process" in payload:
        return "process_identity"
    if "coarse_path" in payload:
        return "file_identity"
    if "root_" in payload or "dir_" in payload or "path_" in payload or "file|" in payload:
        return "file_identity"
    if "exe_" in payload or "flag_" in payload or "process" in payload or "proc_" in payload:
        return "process_identity"
    if text.startswith(("ss:", "ds:", "sh:", "dh:")):
        return "role_shape"
    return "other"


def _counter_top(counter: Counter[str], top_k: int) -> list[dict[str, int | str]]:
    return [
        {"token": str(token), "count": int(count), "category": _semantic_token_category(str(token))}
        for token, count in counter.most_common(max(int(top_k), 0))
    ]


def _category_counts_from_counter(counter: Counter[str]) -> dict[str, int]:
    out: Counter[str] = Counter()
    for token, count in counter.items():
        out[_semantic_token_category(str(token))] += int(count)
    return {key: int(out[key]) for key in sorted(out)}


def _build_semantic_audit(
    model: "CausalSemanticModel",
    policy: str,
    top_k: int,
    alert_tokens_by_pos: dict[int, list[str]] | None,
    event_labels_by_pos: dict[int, int],
) -> dict[str, Any]:
    vocab_counter = Counter({token: int(model.token_counts.get(token, 0)) for token in model.vocab})
    alert_counter: Counter[str] = Counter()
    correct_counter: Counter[str] = Counter()
    false_counter: Counter[str] = Counter()
    unknown_counter: Counter[str] = Counter()
    alert_events = 0
    correct_events = 0
    false_events = 0
    unknown_events = 0
    for event_pos, tokens in (alert_tokens_by_pos or {}).items():
        label = int(event_labels_by_pos.get(int(event_pos), -1))
        alert_events += 1
        alert_counter.update(tokens)
        if label in {1, 2}:
            correct_events += 1
            correct_counter.update(tokens)
        elif label == 0:
            false_events += 1
            false_counter.update(tokens)
        else:
            unknown_events += 1
            unknown_counter.update(tokens)
    rare_alert_tokens = sorted(
        (
            {
                "token": str(token),
                "alert_count": int(alert_count),
                "train_count": int(model.token_counts.get(token, 0)),
                "category": _semantic_token_category(str(token)),
            }
            for token, alert_count in alert_counter.items()
        ),
        key=lambda row: (int(row["train_count"]), -int(row["alert_count"]), str(row["token"])),
    )[: max(int(top_k), 0)]
    return {
        "purpose": (
            "Post-inference audit only. It checks whether online alerts are driven by "
            "action/type/causal-role structure or fine-grained process/file/network "
            "identity tokens. Labels are used only after alerts were emitted."
        ),
        "semantic_string_policy": _semantic_policy(policy),
        "top_k": int(top_k),
        "train_token_total": int(sum(model.token_counts.values())),
        "train_token_unique": int(len(model.token_counts)),
        "frozen_vocab_size": int(len(model.vocab)),
        "train_token_category_counts": _category_counts_from_counter(model.token_counts),
        "frozen_vocab_category_counts": _category_counts_from_counter(vocab_counter),
        "top_train_tokens": _counter_top(model.token_counts, top_k),
        "top_vocab_tokens": _counter_top(vocab_counter, top_k),
        "event_alert_drivers": {
            "alert_events": int(alert_events),
            "correct_events": int(correct_events),
            "false_events": int(false_events),
            "unknown_events": int(unknown_events),
            "token_category_counts": _category_counts_from_counter(alert_counter),
            "correct_token_category_counts": _category_counts_from_counter(correct_counter),
            "false_token_category_counts": _category_counts_from_counter(false_counter),
            "unknown_token_category_counts": _category_counts_from_counter(unknown_counter),
            "top_alert_tokens": _counter_top(alert_counter, top_k),
            "top_correct_alert_tokens": _counter_top(correct_counter, top_k),
            "top_false_alert_tokens": _counter_top(false_counter, top_k),
            "rare_alert_tokens": rare_alert_tokens,
        },
        "leakage_check": {
            "used_for_scoring": False,
            "labels_used": "after_stream_evaluation_only",
            "full_test_statistics_used_before_scoring": False,
        },
    }


def _predictable_activity_pattern(detail: dict[str, Any], is_src: bool) -> str:
    """Node-centric coarse pattern used only with past test-stream counts."""
    src_role = str(detail.get("src_role", ""))
    dst_role = str(detail.get("dst_role", ""))
    action_family = str(detail.get("action_family", ""))
    max_field = str(detail.get("max_field", ""))
    if is_src:
        return f"S|self={src_role}|af={action_family}|peer={dst_role}|field={max_field}"
    return f"D|self={dst_role}|af={action_family}|peer={src_role}|field={max_field}"


def _predictable_activity_global_pattern(detail: dict[str, Any]) -> str:
    """Coarse event pattern for online cross-node support."""
    src_role = str(detail.get("src_role", ""))
    dst_role = str(detail.get("dst_role", ""))
    action_family = str(detail.get("action_family", ""))
    max_field = str(detail.get("max_field", ""))
    return f"src={src_role}|af={action_family}|dst={dst_role}|field={max_field}"


def _role_kind(role: str) -> str:
    kind = str(role).split("|", 1)[0].strip()
    return kind or "unknown"


def _node_chain_signature(detail: dict[str, Any], is_src: bool) -> tuple[str, str, str]:
    """Node-centric causal chain signature used by the unified online risk."""
    src_role = str(detail.get("src_role", ""))
    dst_role = str(detail.get("dst_role", ""))
    action_family = str(detail.get("action_family", ""))
    if is_src:
        self_kind = _role_kind(src_role)
        peer_kind = _role_kind(dst_role)
        direction = "src"
    else:
        self_kind = _role_kind(dst_role)
        peer_kind = _role_kind(src_role)
        direction = "dst"
    transition = f"{direction}:{self_kind}->{peer_kind}:{action_family}"
    return peer_kind, action_family, transition


def _decay_value(value: float, last_ts_ns: int, now_ts_ns: int, halflife_seconds: float) -> float:
    if last_ts_ns < 0:
        return 0.0
    delta_seconds = max(0.0, float(now_ts_ns - int(last_ts_ns)) / 1_000_000_000.0)
    halflife = max(float(halflife_seconds), 1e-6)
    return float(value) * math.exp(-delta_seconds / halflife)


def _prune_recent_peers(
    recent: deque[tuple[int, int]],
    now_ts_ns: int,
    window_ns: int,
) -> None:
    cutoff = int(now_ts_ns) - int(window_ns)
    while recent and int(recent[0][0]) < cutoff:
        recent.popleft()


def _recent_distinct_peer_count(
    recent_peers: dict[int, deque[tuple[int, int]]],
    node: int,
    now_ts_ns: int,
    window_ns: int,
) -> int:
    recent = recent_peers.get(int(node))
    if recent is None:
        return 0
    _prune_recent_peers(recent, now_ts_ns, window_ns)
    return len({int(peer) for _ts, peer in recent})


def _append_recent_peer(
    recent_peers: dict[int, deque[tuple[int, int]]],
    node: int,
    peer: int,
    now_ts_ns: int,
    window_ns: int,
    cap: int,
) -> None:
    recent = recent_peers.setdefault(int(node), deque())
    _prune_recent_peers(recent, now_ts_ns, window_ns)
    recent.append((int(now_ts_ns), int(peer)))
    while len(recent) > int(cap):
        recent.popleft()


def _prune_recent_chains(
    recent: deque[tuple[int, str, str, str]],
    now_ts_ns: int,
    window_ns: int,
    count_bundle: tuple[Counter[str], Counter[str], Counter[str]] | None = None,
) -> None:
    cutoff = int(now_ts_ns) - int(window_ns)
    while recent and int(recent[0][0]) < cutoff:
        _ts, peer_kind, action_family, transition = recent.popleft()
        if count_bundle is not None:
            for counter, key in zip(count_bundle, (peer_kind, action_family, transition)):
                counter[str(key)] -= 1
                if counter[str(key)] <= 0:
                    del counter[str(key)]


def _chain_count_bundle(
    chain_counts: dict[int, tuple[Counter[str], Counter[str], Counter[str]]],
    node: int,
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    return chain_counts.setdefault(int(node), (Counter(), Counter(), Counter()))


def _recent_chain_diversity(
    recent_chains: dict[int, deque[tuple[int, str, str, str]]],
    chain_counts: dict[int, tuple[Counter[str], Counter[str], Counter[str]]],
    node: int,
    now_ts_ns: int,
    window_ns: int,
    current_signature: tuple[str, str, str],
) -> tuple[int, int, int]:
    recent = recent_chains.get(int(node))
    peer_kind, action_family, transition = current_signature
    if recent is not None:
        count_bundle = _chain_count_bundle(chain_counts, int(node))
        _prune_recent_chains(recent, now_ts_ns, window_ns, count_bundle)
        peer_counts, action_counts, transition_counts = count_bundle
    else:
        peer_counts, action_counts, transition_counts = Counter(), Counter(), Counter()
    peer_extra = 0 if str(peer_kind) in peer_counts else 1
    action_extra = 0 if str(action_family) in action_counts else 1
    transition_extra = 0 if str(transition) in transition_counts else 1
    return len(peer_counts) + peer_extra, len(action_counts) + action_extra, len(transition_counts) + transition_extra


def _append_recent_chain(
    recent_chains: dict[int, deque[tuple[int, str, str, str]]],
    chain_counts: dict[int, tuple[Counter[str], Counter[str], Counter[str]]],
    node: int,
    signature: tuple[str, str, str],
    now_ts_ns: int,
    window_ns: int,
    cap: int,
) -> None:
    recent = recent_chains.setdefault(int(node), deque())
    count_bundle = _chain_count_bundle(chain_counts, int(node))
    _prune_recent_chains(recent, now_ts_ns, window_ns, count_bundle)
    peer_kind, action_family, transition = signature
    recent.append((int(now_ts_ns), str(peer_kind), str(action_family), str(transition)))
    for counter, key in zip(count_bundle, (peer_kind, action_family, transition)):
        counter[str(key)] += 1
    while len(recent) > int(cap):
        _ts, old_peer_kind, old_action_family, old_transition = recent.popleft()
        for counter, key in zip(count_bundle, (old_peer_kind, old_action_family, old_transition)):
            counter[str(key)] -= 1
            if counter[str(key)] <= 0:
                del counter[str(key)]


def _risk_threshold(
    scores: list[float],
    quantile: float,
    explicit_threshold: float,
    min_tail_samples: int = 0,
    expected_alert_budget: int = 0,
    expected_alert_horizon_events: int = 0,
) -> dict[str, float | int | str]:
    finite = [float(x) for x in scores if math.isfinite(float(x))]
    if float(explicit_threshold) > 0.0:
        return {
            "source": "explicit",
            "threshold": float(explicit_threshold),
            "quantile": float(quantile),
            "validation_samples": int(len(finite)),
            "min_tail_samples": int(min_tail_samples),
            "expected_alert_budget": int(expected_alert_budget),
            "expected_alert_horizon_events": int(expected_alert_horizon_events),
        }
    if not finite:
        return {
            "source": "fallback_inf",
            "threshold": float("inf"),
            "quantile": float(quantile),
            "validation_samples": 0,
            "min_tail_samples": int(min_tail_samples),
            "expected_alert_budget": int(expected_alert_budget),
            "expected_alert_horizon_events": int(expected_alert_horizon_events),
        }
    arr = np.asarray(finite, dtype=np.float32)
    if int(expected_alert_budget) > 0 and int(expected_alert_horizon_events) > 0:
        expected_tail_rate = min(
            max(float(expected_alert_budget) / max(float(expected_alert_horizon_events), 1.0), 0.0),
            1.0,
        )
        q_budget = 1.0 - expected_tail_rate
        q_budget = min(max(float(q_budget), 0.0), 1.0)
        tail_samples = float(len(finite)) * max(1.0 - q_budget, 0.0)
        return {
            "source": "validation_expected_alert_budget",
            "threshold": float(np.quantile(arr, q_budget)),
            "quantile": float(q_budget),
            "validation_samples": int(len(finite)),
            "tail_samples": float(tail_samples),
            "min_tail_samples": int(min_tail_samples),
            "expected_alert_budget": int(expected_alert_budget),
            "expected_alert_horizon_events": int(expected_alert_horizon_events),
            "expected_tail_rate": float(expected_tail_rate),
        }
    q = min(max(float(quantile), 0.0), 1.0)
    tail_samples = float(len(finite)) * max(1.0 - q, 0.0)
    if int(min_tail_samples) > 0 and q < 1.0 and tail_samples < float(min_tail_samples):
        return {
            "source": "validation_max_underpowered_tail",
            "threshold": float(np.max(arr)),
            "quantile": float(q),
            "validation_samples": int(len(finite)),
            "tail_samples": float(tail_samples),
            "min_tail_samples": int(min_tail_samples),
            "expected_alert_budget": int(expected_alert_budget),
            "expected_alert_horizon_events": int(expected_alert_horizon_events),
        }
    return {
        "source": "validation_quantile",
        "threshold": float(np.quantile(arr, q)),
        "quantile": float(q),
        "validation_samples": int(len(finite)),
        "tail_samples": float(tail_samples),
        "min_tail_samples": int(min_tail_samples),
        "expected_alert_budget": int(expected_alert_budget),
        "expected_alert_horizon_events": int(expected_alert_horizon_events),
    }


LEARNED_EVIDENCE_FEATURES = (
    "event_score",
    "causal_semantic",
    "residual_fusion",
    "edge_novelty",
    "temporal_burst",
    "episode_support",
    "chain_diversity",
)


def _evidence_vector_from_values(
    event_score: float,
    causal_semantic_score: float,
    residual_fusion_score: float,
    edge_novelty: float,
    temporal_burst: float,
    episode_support: float,
    chain_diversity: float,
) -> np.ndarray:
    return np.asarray(
        [
            float(event_score),
            float(causal_semantic_score),
            float(residual_fusion_score),
            float(edge_novelty),
            float(temporal_burst),
            float(episode_support),
            float(chain_diversity),
        ],
        dtype=np.float32,
    )


def _evidence_vector_from_record(record: dict[str, Any]) -> np.ndarray:
    return _evidence_vector_from_values(
        float(record.get("event_score", 0.0)),
        float(record.get("event_causal_semantic_score", 0.0)),
        float(record.get("event_residual_fusion_score", 0.0)),
        float(record.get("event_edge_novelty", 0.0)),
        float(record.get("event_temporal_burst", 0.0)),
        float(record.get("event_episode_support", 0.0)),
        float(record.get("event_chain_diversity", 0.0)),
    )


class LearnedEvidenceComposer:
    """Train-only one-class evidence composer.

    Purpose:
      Replace hand-weighted online evidence composition with a small learned
      nonconformity model over the existing evidence vector.

    Inputs/outputs:
      Fit receives train-only streaming evidence vectors. Score returns a
      scalar nonconformity score plus per-feature contributions for alerts.

    Leakage/runtime:
      Fit never reads labels. Validation uses the frozen composer only for
      threshold calibration. Test scoring is event-order streaming.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.mode = str(args.learned_evidence_composer_mode)
        self.feature_names = list(LEARNED_EVIDENCE_FEATURES)
        self.clip = float(args.learned_evidence_composer_clip)
        self.ridge = float(args.learned_evidence_composer_ridge)
        self.hidden_dim = max(int(args.learned_evidence_composer_hidden_dim), 1)
        self.epochs = max(int(args.learned_evidence_composer_epochs), 0)
        self.lr = float(args.learned_evidence_composer_lr)
        self.l2 = float(args.learned_evidence_composer_l2)
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.inv_cov: np.ndarray | None = None
        self.w1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.w2: np.ndarray | None = None
        self.b2: np.ndarray | None = None
        self.count = 0
        self.sum_vec = np.zeros((len(self.feature_names),), dtype=np.float64)
        self.sum_outer = np.zeros((len(self.feature_names), len(self.feature_names)), dtype=np.float64)
        self.fit_summary: dict[str, Any] = {}

    def partial_fit(self, vector: np.ndarray) -> None:
        x = np.nan_to_num(np.asarray(vector, dtype=np.float32), nan=0.0, posinf=self.clip, neginf=0.0)
        if self.clip > 0.0:
            x = np.clip(x, 0.0, self.clip)
        if x.shape[0] != len(self.feature_names):
            return
        self.count += 1
        self.sum_vec += np.asarray(x, dtype=np.float64)
        self.sum_outer += np.outer(np.asarray(x, dtype=np.float64), np.asarray(x, dtype=np.float64))
        if self.mode == "additive_autoencoder":
            self._autoencoder_step(np.asarray(x, dtype=np.float32))

    def finalize(self) -> dict[str, Any]:
        dim = len(self.feature_names)
        if self.count <= 0:
            self.center = np.zeros((dim,), dtype=np.float32)
            self.scale = np.ones((dim,), dtype=np.float32)
            self.inv_cov = np.eye(dim, dtype=np.float32)
            self.fit_summary = {"enabled": True, "fit_samples": 0, "mode": self.mode}
            return dict(self.fit_summary)
        mean = (self.sum_vec / float(self.count)).astype(np.float32, copy=False)
        raw_second = (self.sum_outer / float(self.count)).astype(np.float32, copy=False)
        raw_cov = raw_second - np.outer(mean, mean).astype(np.float32, copy=False)
        scale = np.sqrt(np.maximum(np.diag(raw_cov), 1e-6)).astype(np.float32, copy=False)
        scale = np.maximum(scale, 1e-3).astype(np.float32, copy=False)
        self.center = mean
        self.scale = scale
        if self.mode != "additive_autoencoder":
            z_cov = raw_cov / np.outer(scale, scale)
            z_cov = z_cov.astype(np.float32, copy=False)
            z_cov = z_cov + np.eye(dim, dtype=np.float32) * max(float(self.ridge), 1e-6)
            self.inv_cov = np.linalg.pinv(z_cov).astype(np.float32, copy=False)
        self.fit_summary = {
            "enabled": True,
            "mode": self.mode,
            "feature_names": list(self.feature_names),
            "fit_samples": int(self.count),
            "center": [float(v) for v in mean.tolist()],
            "scale": [float(v) for v in scale.tolist()],
            "model_size_bytes": int(self.model_size_bytes()),
            "leakage_check": "fit on train split evidence vectors only; no labels, attack windows, test rankings, or full-test statistics",
        }
        return dict(self.fit_summary)

    def _init_autoencoder(self, in_dim: int) -> None:
        rng = np.random.default_rng(20260512)
        h = min(max(int(self.hidden_dim), 1), in_dim)
        if self.w1 is not None and self.w2 is not None:
            return
        self.w1 = (0.05 * rng.standard_normal((in_dim, h))).astype(np.float32)
        self.b1 = np.zeros((h,), dtype=np.float32)
        self.w2 = (0.05 * rng.standard_normal((h, in_dim))).astype(np.float32)
        self.b2 = np.zeros((in_dim,), dtype=np.float32)
    def _autoencoder_step(self, x: np.ndarray) -> None:
        self._init_autoencoder(int(x.shape[0]))
        if self.w1 is None or self.w2 is None or self.b1 is None or self.b2 is None:
            return
        hidden_pre = x @ self.w1 + self.b1
        hidden = np.tanh(hidden_pre)
        recon = hidden @ self.w2 + self.b2
        err = (recon - x).astype(np.float32, copy=False)
        scale = 2.0 / max(float(x.shape[0]), 1.0)
        grad_recon = err * scale
        grad_w2 = np.outer(hidden, grad_recon) + float(self.l2) * self.w2
        grad_b2 = grad_recon
        grad_hidden = grad_recon @ self.w2.T
        grad_hidden_pre = grad_hidden * (1.0 - hidden * hidden)
        grad_w1 = np.outer(x, grad_hidden_pre) + float(self.l2) * self.w1
        grad_b1 = grad_hidden_pre
        lr = float(self.lr)
        self.w2 -= lr * grad_w2.astype(np.float32, copy=False)
        self.b2 -= lr * grad_b2.astype(np.float32, copy=False)
        self.w1 -= lr * grad_w1.astype(np.float32, copy=False)
        self.b1 -= lr * grad_b1.astype(np.float32, copy=False)

    def fit(self, vectors: list[np.ndarray]) -> dict[str, Any]:
        for vec in vectors:
            self.partial_fit(vec)
        return self.finalize()

    def model_size_bytes(self) -> int:
        total = 0
        for arr in [self.center, self.scale, self.inv_cov, self.w1, self.b1, self.w2, self.b2]:
            if arr is not None:
                total += int(np.asarray(arr).nbytes)
        return int(total)

    def score_vector(self, vector: np.ndarray) -> tuple[float, dict[str, float]]:
        if self.center is None or self.scale is None:
            score = float(np.sum(np.maximum(vector, 0.0)))
            return score, {}
        x = np.nan_to_num(np.asarray(vector, dtype=np.float32), nan=0.0, posinf=self.clip, neginf=0.0)
        if self.clip > 0.0:
            x = np.clip(x, 0.0, self.clip)
        z = ((x - self.center) / self.scale).astype(np.float32, copy=False)
        if self.mode == "additive_autoencoder" and self.w1 is not None and self.w2 is not None and self.b1 is not None and self.b2 is not None:
            hidden = np.tanh(z @ self.w1 + self.b1)
            recon = hidden @ self.w2 + self.b2
            contrib_arr = np.square(z - recon).astype(np.float32, copy=False)
            score = float(np.mean(contrib_arr))
        else:
            inv_cov = self.inv_cov if self.inv_cov is not None else np.eye(z.shape[0], dtype=np.float32)
            weighted = inv_cov @ z
            contrib_arr = np.maximum(z * weighted, 0.0).astype(np.float32, copy=False)
            score = float(np.mean(contrib_arr))
        score = float(min(max(score, 0.0), max(float(self.clip), 0.0))) if self.clip > 0.0 else float(max(score, 0.0))
        contrib = {
            f"learned_evidence_{name}_contrib": float(contrib_arr[idx])
            for idx, name in enumerate(self.feature_names)
        }
        return score, contrib


def _apply_learned_evidence_composer(
    composer: LearnedEvidenceComposer | None,
    node_records: list[dict[str, Any]],
    episode_record: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if composer is None:
        return node_records, episode_record

    learned_nodes: list[dict[str, Any]] = []
    for record in node_records:
        node_vector = _evidence_vector_from_record(record)
        learned_score, learned_contrib = composer.score_vector(node_vector)
        learned_node = dict(record)
        learned_node["base_online_risk"] = float(record.get("online_risk", 0.0))
        learned_node["learned_evidence_score"] = float(learned_score)
        learned_node["learned_evidence_mode"] = str(composer.mode)
        learned_node.update(learned_contrib)
        learned_node["online_risk"] = float(learned_score)
        learned_node.setdefault("explanation_json", "")
        explanation = {
            "learned_evidence_mode": str(composer.mode),
            "learned_evidence_score": float(learned_score),
            **learned_contrib,
        }
        if learned_node["explanation_json"]:
            try:
                prev = json.loads(str(learned_node["explanation_json"]))
                if isinstance(prev, dict):
                    explanation = {**prev, **explanation}
            except Exception:
                pass
        learned_node["explanation_json"] = json.dumps(explanation, sort_keys=True)
        learned_nodes.append(learned_node)

    episode_vector = _evidence_vector_from_record(episode_record)
    learned_score, learned_contrib = composer.score_vector(episode_vector)
    learned_episode = dict(episode_record)
    learned_episode["base_online_risk"] = float(episode_record.get("online_risk", 0.0))
    learned_episode["learned_evidence_score"] = float(learned_score)
    learned_episode["learned_evidence_mode"] = str(composer.mode)
    learned_episode.update(learned_contrib)
    learned_episode["online_risk"] = float(learned_score)
    learned_episode.setdefault("explanation_json", "")
    explanation = {
        "learned_evidence_mode": str(composer.mode),
        "learned_evidence_score": float(learned_score),
        **learned_contrib,
    }
    if learned_episode["explanation_json"]:
        try:
            prev = json.loads(str(learned_episode["explanation_json"]))
            if isinstance(prev, dict):
                explanation = {**prev, **explanation}
        except Exception:
            pass
    learned_episode["explanation_json"] = json.dumps(explanation, sort_keys=True)
    return learned_nodes, learned_episode


def _multi_view_explanation_fields(detail_or_record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "base_effective_event_score",
        "multi_view_behavior_score",
        "multi_view_behavior_mode",
        "multi_view_behavior_policies",
        "multi_view_top_component",
        "multi_view_top_contrib",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        if key in detail_or_record:
            value = detail_or_record.get(key)
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            out[key] = value
    contrib_items = [
        (str(key), float(value))
        for key, value in detail_or_record.items()
        if str(key).startswith("multi_view_") and str(key).endswith("_contrib")
    ]
    contrib_items.sort(key=lambda item: (-item[1], item[0]))
    if contrib_items:
        out["multi_view_top_contribs"] = [
            {"component": key, "value": value}
            for key, value in contrib_items[:5]
        ]
    elif "multi_view_top_contribs" in detail_or_record:
        out["multi_view_top_contribs"] = detail_or_record.get("multi_view_top_contribs", [])
    return out


def _attach_multi_view_fields(record: dict[str, Any], detail: dict[str, Any]) -> None:
    for key, value in _multi_view_explanation_fields(detail).items():
        record[key] = value


def _merge_explanation_json(record: dict[str, Any], extra: dict[str, Any]) -> str:
    explanation = dict(extra)
    existing = record.get("explanation_json", "")
    if existing:
        try:
            prev = json.loads(str(existing))
            if isinstance(prev, dict):
                explanation = {**prev, **explanation}
        except Exception:
            pass
    return json.dumps(explanation, sort_keys=True)


class StatefulNodeAlertPolicy:
    """Track per-node alert history for one-shot or risk-update emission."""

    def __init__(self, max_node_id: int, update_gain_ratio: float, policy_name: str, mode: str = "stateful_update") -> None:
        self.max_node_id = int(max_node_id)
        self.update_gain_ratio = float(update_gain_ratio)
        if str(mode) not in {"one_shot", "stateful_update"}:
            raise ValueError(f"Unsupported node alert policy: {mode}")
        self.mode = str(mode)
        self.policy_name = str(policy_name)
        self.last_alert_risk = np.full((self.max_node_id,), np.nan, dtype=np.float32)
        self.last_alert_event_pos = np.full((self.max_node_id,), -1, dtype=np.int64)
        self.last_alert_timestamp_ns = np.full((self.max_node_id,), -1, dtype=np.int64)
        self.alert_version = np.zeros((self.max_node_id,), dtype=np.int32)

    def maybe_emit(
        self,
        node: int,
        current_risk: float,
        threshold: float,
        event_pos: int,
        timestamp_ns: int,
    ) -> tuple[bool, dict[str, Any]]:
        node = int(node)
        if node < 0 or node >= self.max_node_id:
            return False, {}
        current_risk = float(current_risk)
        threshold = float(threshold)
        if not math.isfinite(current_risk) or current_risk < threshold:
            return False, {}

        previous_risk = float(self.last_alert_risk[node])
        if not math.isfinite(previous_risk):
            reason = "threshold_crossing"
            risk_gain = current_risk
            risk_gain_ratio = float("inf") if current_risk > 0.0 else 0.0
        else:
            if self.mode == "one_shot":
                return False, {}
            risk_gain = current_risk - previous_risk
            absolute_gain_floor = max(0.5, 0.10 * max(threshold, 1.0))
            relative_gain_floor = max(previous_risk, threshold, 1.0) * max(self.update_gain_ratio, 0.0)
            if risk_gain < absolute_gain_floor and risk_gain < relative_gain_floor:
                return False, {}
            reason = "risk_update"
            risk_gain_ratio = current_risk / max(previous_risk, 1e-6)

        self.alert_version[node] += 1
        info = {
            "alert_version": int(self.alert_version[node]),
            "alert_is_update": int(self.alert_version[node] > 1),
            "alert_policy": str(self.mode),
            "alert_policy_stage": str(self.policy_name),
            "alert_policy_reason": str(reason),
            "previous_alert_risk": float(previous_risk if math.isfinite(previous_risk) else 0.0),
            "risk_gain": float(risk_gain),
            "risk_gain_ratio": float(risk_gain_ratio),
            "state_last_alert_event_pos": int(self.last_alert_event_pos[node]),
            "state_last_alert_timestamp_ns": int(self.last_alert_timestamp_ns[node]),
        }
        self.last_alert_risk[node] = np.float32(current_risk)
        self.last_alert_event_pos[node] = np.int64(event_pos)
        self.last_alert_timestamp_ns[node] = np.int64(timestamp_ns)
        return True, info


class OnlineAttackEvidenceState:
    """Causal state for novelty-gated online attack evidence."""

    def __init__(self, max_node_id: int, args: argparse.Namespace, tail_q95: float) -> None:
        self.max_node_id = int(max_node_id)
        self.args = args
        self.tail_q95 = float(tail_q95)
        self.node_attack_edge_sum = np.zeros((self.max_node_id,), dtype=np.float32)
        self.node_attack_burst_sum = np.zeros((self.max_node_id,), dtype=np.float32)
        self.node_attack_episode_sum = np.zeros((self.max_node_id,), dtype=np.float32)
        self.node_attack_chain_sum = np.zeros((self.max_node_id,), dtype=np.float32)
        self.node_burst_state = np.zeros((self.max_node_id,), dtype=np.float32)
        self.node_episode_state = np.zeros((self.max_node_id,), dtype=np.float32)
        self.node_burst_last_ts = np.full((self.max_node_id,), -1, dtype=np.int64)
        self.node_episode_last_ts = np.full((self.max_node_id,), -1, dtype=np.int64)
        self.node_episode_recent_peer_max = np.zeros((self.max_node_id,), dtype=np.int32)
        self.node_chain_diversity_max = np.zeros((self.max_node_id,), dtype=np.int32)
        self.node_attack_episode_event_count = np.zeros((self.max_node_id,), dtype=np.int32)
        self.edge_counts: dict[tuple[int, int, str], int] = {}
        self.recent_peers: dict[int, deque[tuple[int, int]]] = {}
        self.recent_chains: dict[int, deque[tuple[int, str, str, str]]] = {}
        self.recent_chain_counts: dict[int, tuple[Counter[str], Counter[str], Counter[str]]] = {}
        self.edge_key_dropped = 0
        self.edge_novelty_suppressed_overflow = 0
        self.attack_window_ns = int(max(float(args.attack_evidence_episode_window_seconds), 0.001) * 1_000_000_000.0)

    def score_event(
        self,
        row: dict[str, Any],
        score: float,
        detail: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        edge_enabled = bool(getattr(self.args, "attack_evidence_edge_enabled", True))
        burst_enabled = bool(getattr(self.args, "attack_evidence_burst_enabled", True))
        episode_enabled = bool(getattr(self.args, "attack_evidence_episode_enabled", True))
        chain_enabled = bool(getattr(self.args, "attack_evidence_chain_enabled", True))
        needs_edge_signal = bool(edge_enabled or burst_enabled or episode_enabled or chain_enabled)
        timestamp_ns = int(row["timestamp_ns"])
        src_idx = int(row["src_idx"])
        dst_idx = int(row["dst_idx"])
        action_family = str(detail.get("action_family", ""))
        edge_key = (src_idx, dst_idx, action_family)
        tail_signal = float(min(max(float(score) - self.tail_q95, 0.0), float(self.args.score_clip)))
        edge_seen_before = 1
        edge_novelty_signal = 0.0
        if needs_edge_signal:
            edge_seen_raw = self.edge_counts.get(edge_key)
            edge_tracked = edge_seen_raw is not None
            edge_table_full = len(self.edge_counts) >= int(self.args.attack_evidence_max_edge_keys)
            edge_overflow_untracked = bool((not edge_tracked) and edge_table_full)
            edge_seen_before = int(edge_seen_raw or 0)
            if edge_overflow_untracked and str(self.args.attack_evidence_edge_overflow_policy) == "not_novel":
                edge_seen_before = 1
                self.edge_novelty_suppressed_overflow += 1
            edge_novelty_signal = float(tail_signal if edge_seen_before == 0 else 0.0)
        src_recent_peers = (
            _recent_distinct_peer_count(self.recent_peers, src_idx, timestamp_ns, self.attack_window_ns)
            if episode_enabled and 0 <= src_idx < self.max_node_id
            else 0
        )
        dst_recent_peers = (
            _recent_distinct_peer_count(self.recent_peers, dst_idx, timestamp_ns, self.attack_window_ns)
            if episode_enabled and 0 <= dst_idx < self.max_node_id
            else 0
        )
        episode_peer_support = max(src_recent_peers, dst_recent_peers)
        episode_core = 0.0
        if episode_enabled and episode_peer_support >= int(self.args.attack_evidence_min_distinct_peers):
            episode_core = float(edge_novelty_signal * math.log1p(float(episode_peer_support)))

        records: list[dict[str, Any]] = []
        cap = float(self.args.attack_evidence_component_cap)
        for node_pos, node in enumerate([src_idx, dst_idx]):
            if node < 0 or node >= self.max_node_id:
                continue
            peer = dst_idx if node_pos == 0 else src_idx
            chain_signature = _node_chain_signature(detail, is_src=(node_pos == 0)) if chain_enabled else ""
            if chain_enabled:
                chain_peer_kind_count, chain_action_family_count, chain_transition_count = _recent_chain_diversity(
                    self.recent_chains,
                    self.recent_chain_counts,
                    node,
                    timestamp_ns,
                    self.attack_window_ns,
                    chain_signature,
                )
            else:
                chain_peer_kind_count, chain_action_family_count, chain_transition_count = (0, 0, 0)
            chain_diversity_units = (
                max(int(chain_peer_kind_count) - 1, 0)
                + max(int(chain_action_family_count) - 1, 0)
                + max(int(chain_transition_count) - 1, 0)
            )
            chain_diversity_signal = 0.0
            if chain_enabled and edge_novelty_signal > 0.0 and chain_diversity_units > 0:
                chain_diversity_signal = float(edge_novelty_signal * min(math.log1p(float(chain_diversity_units)), cap))
            if chain_enabled:
                self.node_attack_chain_sum[node] += np.float32(min(chain_diversity_signal, cap))
                if chain_diversity_units > int(self.node_chain_diversity_max[node]):
                    self.node_chain_diversity_max[node] = np.int32(chain_diversity_units)

            burst_signal = 0.0
            if burst_enabled:
                prior_burst_state = _decay_value(
                    float(self.node_burst_state[node]),
                    int(self.node_burst_last_ts[node]),
                    timestamp_ns,
                    float(self.args.attack_evidence_burst_halflife_seconds),
                )
                burst_signal = float(edge_novelty_signal * math.log1p(prior_burst_state))
                self.node_burst_state[node] = np.float32(min(prior_burst_state + edge_novelty_signal, float(self.args.attack_evidence_state_cap)))
                self.node_burst_last_ts[node] = np.int64(timestamp_ns)
                self.node_attack_burst_sum[node] += np.float32(min(burst_signal, cap))

            episode_signal = 0.0
            if episode_enabled:
                prior_episode_state = _decay_value(
                    float(self.node_episode_state[node]),
                    int(self.node_episode_last_ts[node]),
                    timestamp_ns,
                    float(self.args.attack_evidence_episode_halflife_seconds),
                )
                peer_episode_state = 0.0
                if 0 <= peer < self.max_node_id:
                    peer_episode_state = _decay_value(
                        float(self.node_episode_state[peer]),
                        int(self.node_episode_last_ts[peer]),
                        timestamp_ns,
                        float(self.args.attack_evidence_episode_halflife_seconds),
                    )
                if episode_core > 0.0:
                    episode_signal = float(episode_core + float(self.args.attack_evidence_propagation_weight) * peer_episode_state)
                if episode_signal > 0.0:
                    self.node_attack_episode_event_count[node] += 1
                    self.node_attack_episode_sum[node] += np.float32(min(episode_signal, cap))
                self.node_episode_state[node] = np.float32(min(prior_episode_state + episode_signal, float(self.args.attack_evidence_state_cap)))
                self.node_episode_last_ts[node] = np.int64(timestamp_ns)
                if episode_peer_support > int(self.node_episode_recent_peer_max[node]):
                    self.node_episode_recent_peer_max[node] = np.int32(episode_peer_support)
            if edge_enabled:
                self.node_attack_edge_sum[node] += np.float32(min(edge_novelty_signal, cap))

            weighted_evidence_risk = (
                float(score)
                + (float(self.args.attack_evidence_edge_weight) * min(edge_novelty_signal, cap) if edge_enabled else 0.0)
                + (float(self.args.attack_evidence_burst_weight) * min(burst_signal, cap) if burst_enabled else 0.0)
                + (float(self.args.attack_evidence_episode_weight) * min(episode_signal, cap) if episode_enabled else 0.0)
                + (float(self.args.attack_evidence_chain_weight) * min(chain_diversity_signal, cap) if chain_enabled else 0.0)
            )
            if str(getattr(self.args, "online_risk_composition", "weighted_evidence")) == "score_only":
                online_risk = float(score)
            else:
                online_risk = float(weighted_evidence_risk)
            records.append(
                {
                    "node_id": int(node),
                    "peer_idx": int(peer),
                    "online_risk": float(online_risk),
                    "weighted_evidence_risk": float(weighted_evidence_risk),
                    "event_score": float(score),
                    "event_causal_semantic_score": float(detail.get("causal_semantic_score", score)),
                    "event_residual_fusion_score": float(detail.get("residual_fusion_score", 0.0)),
                    "event_edge_novelty": float(min(edge_novelty_signal, cap) if edge_enabled else 0.0),
                    "event_temporal_burst": float(min(burst_signal, cap) if burst_enabled else 0.0),
                    "event_episode_support": float(min(episode_signal, cap) if episode_enabled else 0.0),
                    "event_chain_diversity": float(min(chain_diversity_signal, cap) if chain_enabled else 0.0),
                    "episode_recent_peer_count": int(episode_peer_support),
                    "chain_diversity_units": int(chain_diversity_units),
                    "chain_peer_kind_count": int(chain_peer_kind_count),
                    "chain_action_family_count": int(chain_action_family_count),
                    "chain_transition_count": int(chain_transition_count),
                }
            )

        if 0 <= src_idx < self.max_node_id and 0 <= dst_idx < self.max_node_id:
            if episode_enabled:
                _append_recent_peer(self.recent_peers, src_idx, dst_idx, timestamp_ns, self.attack_window_ns, int(self.args.attack_evidence_recent_peer_cap))
                _append_recent_peer(self.recent_peers, dst_idx, src_idx, timestamp_ns, self.attack_window_ns, int(self.args.attack_evidence_recent_peer_cap))
            if chain_enabled:
                _append_recent_chain(
                    self.recent_chains,
                    self.recent_chain_counts,
                    src_idx,
                    _node_chain_signature(detail, is_src=True),
                    timestamp_ns,
                    self.attack_window_ns,
                    int(self.args.attack_evidence_recent_peer_cap),
                )
                _append_recent_chain(
                    self.recent_chains,
                    self.recent_chain_counts,
                    dst_idx,
                    _node_chain_signature(detail, is_src=False),
                    timestamp_ns,
                    self.attack_window_ns,
                    int(self.args.attack_evidence_recent_peer_cap),
                )
        if needs_edge_signal:
            if edge_key in self.edge_counts or len(self.edge_counts) < int(self.args.attack_evidence_max_edge_keys):
                self.edge_counts[edge_key] = edge_seen_before + 1
            else:
                self.edge_key_dropped += 1

        episode_risk = max((float(record["online_risk"]) for record in records), default=float(score))
        episode_record = {
            "episode_id": f"{src_idx}:{dst_idx}:{action_family}",
            "online_risk": float(episode_risk),
            "weighted_evidence_risk": float(max((record.get("weighted_evidence_risk", record["online_risk"]) for record in records), default=float(score))),
            "event_score": float(score),
            "event_causal_semantic_score": float(detail.get("causal_semantic_score", score)),
            "event_residual_fusion_score": float(detail.get("residual_fusion_score", 0.0)),
            "event_edge_novelty": float(min(edge_novelty_signal, cap)),
            "event_temporal_burst": float(max((record["event_temporal_burst"] for record in records), default=0.0)),
            "event_episode_support": float(max((record["event_episode_support"] for record in records), default=0.0)),
            "event_chain_diversity": float(max((record["event_chain_diversity"] for record in records), default=0.0)),
            "episode_recent_peer_count": int(episode_peer_support),
            "chain_diversity_units": int(max((record["chain_diversity_units"] for record in records), default=0)),
            "chain_peer_kind_count": int(max((record["chain_peer_kind_count"] for record in records), default=0)),
            "chain_action_family_count": int(max((record["chain_action_family_count"] for record in records), default=0)),
            "chain_transition_count": int(max((record["chain_transition_count"] for record in records), default=0)),
            "src_idx": int(src_idx),
            "dst_idx": int(dst_idx),
            "action_family": action_family,
        }
        return records, episode_record


class OnlineCascadeConfirmer:
    """Bounded-delay causal-episode confirmer for candidate alerts.

    Purpose:
      Keep immediate candidate alerts high-recall, then confirm a short
      endpoint-anchored causal episode and propagate that confirmation to
      candidate events and nodes inside the same bounded window.

    Inputs/outputs:
      Receives streaming candidate event/node records and returns confirmed
      event rows plus confirmed node rows. It does not read labels or future
      events.

    Leakage/runtime:
      State is created only from online candidate alerts and updated with
      current/past events inside the configured time/event window. It is
      bounded by the window and periodically pruned.
    """

    def __init__(self, args: argparse.Namespace, max_node_id: int, node_threshold: float, event_threshold: float) -> None:
        self.args = args
        self.max_node_id = int(max_node_id)
        self.node_threshold = float(node_threshold)
        self.event_threshold = float(event_threshold)
        self.window_ns = int(max(float(args.online_confirm_window_seconds), 0.001) * 1_000_000_000.0)
        self.window_events = max(int(args.online_confirm_window_events), 1)
        self.min_evidence_kinds = max(int(args.online_confirm_min_evidence_kinds), 1)
        self.pending_episodes: dict[str, dict[str, Any]] = {}
        self.emitted_event_positions: set[int] = set()
        self.node_alert_policy = StatefulNodeAlertPolicy(
            self.max_node_id,
            float(getattr(args, "online_node_update_gain_ratio", 0.15)),
            "confirmed_node_stateful_update",
            str(getattr(args, "online_node_alert_policy", "stateful_update")),
        )
        self.last_prune_event_pos = -1

    def _in_window(self, state: dict[str, Any], event_pos: int, timestamp_ns: int) -> bool:
        if int(event_pos) - int(state["candidate_event_pos"]) > self.window_events:
            return False
        if int(timestamp_ns) - int(state["candidate_timestamp_ns"]) > self.window_ns:
            return False
        return True

    def _prune_expired(self, event_pos: int, timestamp_ns: int) -> None:
        interval = min(max(self.window_events // 20, 1000), 10000)
        if int(event_pos) - int(self.last_prune_event_pos) < int(interval):
            return
        self.last_prune_event_pos = int(event_pos)
        expired = [
            key
            for key, state in self.pending_episodes.items()
            if not self._in_window(state, int(event_pos), int(timestamp_ns))
        ]
        for key in expired:
            self.pending_episodes.pop(key, None)

    @staticmethod
    def _evidence_flags(record: dict[str, Any]) -> dict[str, int]:
        peer_count = int(record.get("episode_recent_peer_count", 0))
        chain_units = int(record.get("chain_diversity_units", 0))
        return {
            "edge_novelty": int(float(record.get("event_edge_novelty", 0.0)) > 0.0),
            "temporal_burst": int(float(record.get("event_temporal_burst", 0.0)) > 0.0),
            "episode_support": int(float(record.get("event_episode_support", 0.0)) > 0.0 or peer_count > 0),
            "chain_diversity": int(float(record.get("event_chain_diversity", 0.0)) > 0.0 or chain_units > 0),
        }

    @staticmethod
    def _roles_from_detail(detail: dict[str, Any]) -> tuple[str, str]:
        return str(detail.get("src_role", "")), str(detail.get("dst_role", ""))

    @staticmethod
    def _anchor_keys(src_idx: int, dst_idx: int) -> list[tuple[str, int]]:
        anchors: list[tuple[str, int]] = []
        seen: set[str] = set()
        for node in [int(src_idx), int(dst_idx)]:
            if node < 0:
                continue
            key = f"anchor_node:{node}"
            if key not in seen:
                anchors.append((key, node))
                seen.add(key)
        return anchors

    @staticmethod
    def _record_from_episode(episode_record: dict[str, Any]) -> dict[str, Any]:
        return {
            "node_id": -1,
            "peer_idx": int(episode_record.get("dst_idx", -1)),
            "online_risk": float(episode_record.get("online_risk", 0.0)),
            "event_score": float(episode_record.get("event_score", 0.0)),
            "event_causal_semantic_score": float(episode_record.get("event_causal_semantic_score", 0.0)),
            "event_residual_fusion_score": float(episode_record.get("event_residual_fusion_score", 0.0)),
            "event_edge_novelty": float(episode_record.get("event_edge_novelty", 0.0)),
            "event_temporal_burst": float(episode_record.get("event_temporal_burst", 0.0)),
            "event_episode_support": float(episode_record.get("event_episode_support", 0.0)),
            "event_chain_diversity": float(episode_record.get("event_chain_diversity", 0.0)),
            "episode_recent_peer_count": int(episode_record.get("episode_recent_peer_count", 0)),
            "chain_diversity_units": int(episode_record.get("chain_diversity_units", 0)),
        }

    def _endpoint_node_records_for_confirmation(
        self,
        node_records: list[dict[str, Any]],
        node_candidate_records: list[dict[str, Any]],
        event_candidate_crossed: bool,
    ) -> list[dict[str, Any]]:
        if not event_candidate_crossed:
            return []
        mode = str(getattr(self.args, "online_confirm_node_candidate_policy", "episode_all"))
        if mode == "episode_all":
            return list(node_records)
        candidate_nodes = {int(record.get("node_id", -1)) for record in node_candidate_records}
        out: list[dict[str, Any]] = []
        for record in node_records:
            node = int(record.get("node_id", -1))
            if node < 0:
                continue
            if node in candidate_nodes:
                out.append(record)
                continue
            if mode != "candidate_or_strong":
                continue
            evidence_kinds = int(sum(int(v) for v in self._evidence_flags(record).values()))
            strong_threshold = float(self.node_threshold) * max(float(getattr(self.args, "online_confirm_node_strong_risk_ratio", 1.05)), 1.0)
            if (
                evidence_kinds >= self.min_evidence_kinds
                and float(record.get("online_risk", 0.0)) >= strong_threshold
            ):
                out.append(record)
        return out

    def _new_state(
        self,
        anchor_key: str,
        anchor_node: int,
        record: dict[str, Any],
        event_pos: int,
        timestamp_ns: int,
        src_idx: int,
        dst_idx: int,
        detail: dict[str, Any],
    ) -> dict[str, Any]:
        src_role, dst_role = self._roles_from_detail(detail)
        return {
            "anchor_key": str(anchor_key),
            "anchor_node": int(anchor_node),
            "candidate_event_pos": int(event_pos),
            "candidate_timestamp_ns": int(timestamp_ns),
            "candidate_online_risk": float(record.get("online_risk", 0.0)),
            "candidate_event_score": float(record.get("event_score", 0.0)),
            "src_idx": int(src_idx),
            "dst_idx": int(dst_idx),
            "action_family": str(detail.get("action_family", "")),
            "max_field": str(detail.get("max_field", "")),
            "src_role": src_role,
            "dst_role": dst_role,
            "roles_seen": {src_role, dst_role},
            "peers_seen": {int(src_idx), int(dst_idx)},
            "actions_seen": {str(detail.get("action_family", ""))},
            "episode_ids_seen": {f"{int(src_idx)}:{int(dst_idx)}:{str(detail.get('action_family', ''))}"},
            "candidate_events": {},
            "candidate_nodes": {},
            "event_count": 0,
            "max_online_risk": float(record.get("online_risk", 0.0)),
            "max_candidate_online_risk": 0.0,
            "sum_edge_novelty": 0.0,
            "sum_temporal_burst": 0.0,
            "sum_episode_support": 0.0,
            "sum_chain_diversity": 0.0,
            "max_recent_peer_count": 0,
            "max_chain_diversity_units": 0,
            "evidence_flags": self._evidence_flags(record),
        }

    def _update_state(
        self,
        state: dict[str, Any],
        record: dict[str, Any],
        event_pos: int,
        timestamp_ns: int,
        src_idx: int,
        dst_idx: int,
        detail: dict[str, Any],
    ) -> None:
        src_role, dst_role = self._roles_from_detail(detail)
        flags = self._evidence_flags(record)
        state["event_count"] = int(state.get("event_count", 0)) + 1
        state["last_event_pos"] = int(event_pos)
        state["last_timestamp_ns"] = int(timestamp_ns)
        state["max_online_risk"] = max(float(state.get("max_online_risk", 0.0)), float(record.get("online_risk", 0.0)))
        state["sum_edge_novelty"] = float(state.get("sum_edge_novelty", 0.0)) + float(record.get("event_edge_novelty", 0.0))
        state["sum_temporal_burst"] = float(state.get("sum_temporal_burst", 0.0)) + float(record.get("event_temporal_burst", 0.0))
        state["sum_episode_support"] = float(state.get("sum_episode_support", 0.0)) + float(record.get("event_episode_support", 0.0))
        state["sum_chain_diversity"] = float(state.get("sum_chain_diversity", 0.0)) + float(record.get("event_chain_diversity", 0.0))
        state["max_recent_peer_count"] = max(int(state.get("max_recent_peer_count", 0)), int(record.get("episode_recent_peer_count", 0)))
        state["max_chain_diversity_units"] = max(int(state.get("max_chain_diversity_units", 0)), int(record.get("chain_diversity_units", 0)))
        state.setdefault("roles_seen", set()).update([src_role, dst_role])
        state.setdefault("peers_seen", set()).update([int(src_idx), int(dst_idx), int(record.get("peer_idx", -1))])
        state.setdefault("actions_seen", set()).add(str(detail.get("action_family", "")))
        state.setdefault("episode_ids_seen", set()).add(f"{int(src_idx)}:{int(dst_idx)}:{str(detail.get('action_family', ''))}")
        for key, value in flags.items():
            state.setdefault("evidence_flags", {})[key] = int(max(int(state["evidence_flags"].get(key, 0)), int(value)))

    def _add_event_candidate(
        self,
        state: dict[str, Any],
        episode_record: dict[str, Any],
        event_pos: int,
        timestamp_ns: int,
        detail: dict[str, Any],
    ) -> None:
        pos = int(event_pos)
        if pos in self.emitted_event_positions:
            return
        candidate_events = state.setdefault("candidate_events", {})
        if pos in candidate_events:
            return
        state["max_candidate_online_risk"] = max(
            float(state.get("max_candidate_online_risk", 0.0)),
            float(episode_record.get("online_risk", 0.0)),
        )
        row = dict(episode_record)
        row.update(
            {
                "event_pos": int(event_pos),
                "timestamp_ns": int(timestamp_ns),
                "candidate_event_pos": int(event_pos),
                "candidate_timestamp_ns": int(timestamp_ns),
                "candidate_online_risk": float(episode_record.get("online_risk", 0.0)),
                "max_field": str(detail.get("max_field", "")),
                "src_role": str(detail.get("src_role", "")),
                "dst_role": str(detail.get("dst_role", "")),
                "action_family": str(detail.get("action_family", "")),
            }
        )
        candidate_events[pos] = row

    def _add_node_candidate(
        self,
        state: dict[str, Any],
        record: dict[str, Any],
        event_pos: int,
        timestamp_ns: int,
        src_idx: int,
        dst_idx: int,
        detail: dict[str, Any],
    ) -> None:
        node = int(record.get("node_id", -1))
        if node < 0:
            return
        candidate_nodes = state.setdefault("candidate_nodes", {})
        current_risk = float(record.get("online_risk", 0.0))
        existing = candidate_nodes.get(node)
        if existing is not None:
            previous_risk = float(existing.get("candidate_online_risk", existing.get("online_risk", 0.0)))
            if current_risk <= previous_risk:
                return
            updated = dict(existing)
            updated.update(
                {
                    "node_id": int(node),
                    "peer_idx": int(record.get("peer_idx", -1)),
                    "candidate_update_count": int(existing.get("candidate_update_count", 1)) + 1,
                    "candidate_first_event_pos": int(existing.get("candidate_first_event_pos", existing.get("candidate_event_pos", event_pos))),
                    "candidate_first_timestamp_ns": int(existing.get("candidate_first_timestamp_ns", existing.get("candidate_timestamp_ns", timestamp_ns))),
                    "candidate_event_pos": int(event_pos),
                    "candidate_timestamp_ns": int(timestamp_ns),
                    "candidate_online_risk": float(current_risk),
                    "event_score": float(record.get("event_score", 0.0)),
                    "event_causal_semantic_score": float(record.get("event_causal_semantic_score", 0.0)),
                    "event_residual_fusion_score": float(record.get("event_residual_fusion_score", 0.0)),
                    "event_edge_novelty": float(record.get("event_edge_novelty", 0.0)),
                    "event_temporal_burst": float(record.get("event_temporal_burst", 0.0)),
                    "event_episode_support": float(record.get("event_episode_support", 0.0)),
                    "event_chain_diversity": float(record.get("event_chain_diversity", 0.0)),
                    "episode_recent_peer_count": int(record.get("episode_recent_peer_count", 0)),
                    "chain_diversity_units": int(record.get("chain_diversity_units", 0)),
                    "chain_peer_kind_count": int(record.get("chain_peer_kind_count", 0)),
                    "chain_action_family_count": int(record.get("chain_action_family_count", 0)),
                    "chain_transition_count": int(record.get("chain_transition_count", 0)),
                    "src_idx": int(src_idx),
                    "dst_idx": int(dst_idx),
                    "action_family": str(detail.get("action_family", "")),
                    "max_field": str(detail.get("max_field", "")),
                    "src_role": str(detail.get("src_role", "")),
                    "dst_role": str(detail.get("dst_role", "")),
                }
            )
            candidate_nodes[node] = updated
            state["max_candidate_online_risk"] = max(float(state.get("max_candidate_online_risk", 0.0)), current_risk)
            return
        state["max_candidate_online_risk"] = max(
            float(state.get("max_candidate_online_risk", 0.0)),
            current_risk,
        )
        candidate_nodes[node] = {
            **dict(record),
            "node_id": int(node),
            "peer_idx": int(record.get("peer_idx", -1)),
            "candidate_update_count": 1,
            "candidate_first_event_pos": int(event_pos),
            "candidate_first_timestamp_ns": int(timestamp_ns),
            "candidate_event_pos": int(event_pos),
            "candidate_timestamp_ns": int(timestamp_ns),
            "candidate_online_risk": float(current_risk),
            "event_score": float(record.get("event_score", 0.0)),
            "event_causal_semantic_score": float(record.get("event_causal_semantic_score", 0.0)),
            "event_residual_fusion_score": float(record.get("event_residual_fusion_score", 0.0)),
            "event_edge_novelty": float(record.get("event_edge_novelty", 0.0)),
            "event_temporal_burst": float(record.get("event_temporal_burst", 0.0)),
            "event_episode_support": float(record.get("event_episode_support", 0.0)),
            "event_chain_diversity": float(record.get("event_chain_diversity", 0.0)),
            "episode_recent_peer_count": int(record.get("episode_recent_peer_count", 0)),
            "chain_diversity_units": int(record.get("chain_diversity_units", 0)),
            "chain_peer_kind_count": int(record.get("chain_peer_kind_count", 0)),
            "chain_action_family_count": int(record.get("chain_action_family_count", 0)),
            "chain_transition_count": int(record.get("chain_transition_count", 0)),
            "src_idx": int(src_idx),
            "dst_idx": int(dst_idx),
            "action_family": str(detail.get("action_family", "")),
            "max_field": str(detail.get("max_field", "")),
            "src_role": str(detail.get("src_role", "")),
            "dst_role": str(detail.get("dst_role", "")),
        }

    def _score_state(self, state: dict[str, Any]) -> tuple[float, int]:
        role_diversity = len(state.get("roles_seen", set()))
        peer_diversity = len(state.get("peers_seen", set()))
        action_diversity = len(state.get("actions_seen", set()))
        episode_diversity = len(state.get("episode_ids_seen", set()))
        evidence_kinds = int(sum(int(v) for v in state.get("evidence_flags", {}).values()))
        event_count = max(int(state.get("event_count", 0)), 1)
        candidate_risk = max(
            float(state.get("max_candidate_online_risk", 0.0)),
            float(state.get("candidate_online_risk", 0.0)),
        )
        cap = max(float(self.args.attack_evidence_component_cap), 0.0)
        avg_edge = min(float(state.get("sum_edge_novelty", 0.0)) / float(event_count), cap)
        avg_burst = min(float(state.get("sum_temporal_burst", 0.0)) / float(event_count), cap)
        avg_episode = min(float(state.get("sum_episode_support", 0.0)) / float(event_count), cap)
        avg_chain = min(float(state.get("sum_chain_diversity", 0.0)) / float(event_count), cap)
        multi_event_consistency = math.log1p(float(min(max(event_count - 1, 0), 8)))
        role_consistency = math.log1p(float(min(max(role_diversity - 1, 0), 3)))
        peer_consistency = math.log1p(float(min(max(peer_diversity - 1, 0), 5)))
        action_consistency = math.log1p(float(min(max(action_diversity - 1, 0), 3)))
        episode_consistency = math.log1p(float(min(max(episode_diversity - 1, 0), 5)))
        compact_consistency = (
            0.45 * multi_event_consistency
            + 0.22 * role_consistency
            + 0.18 * peer_consistency
            + 0.22 * action_consistency
            + 0.22 * episode_consistency
            + 0.16 * avg_edge
            + 0.12 * avg_burst
            + 0.16 * avg_episode
            + 0.16 * avg_chain
            + 0.38 * float(max(evidence_kinds - 1, 0))
        )
        confirm_score = candidate_risk + compact_consistency
        state["last_candidate_risk_component"] = float(candidate_risk)
        state["last_compact_consistency_component"] = float(compact_consistency)
        state["last_avg_edge_novelty"] = float(avg_edge)
        state["last_avg_temporal_burst"] = float(avg_burst)
        state["last_avg_episode_support"] = float(avg_episode)
        state["last_avg_chain_diversity"] = float(avg_chain)
        return float(confirm_score), int(evidence_kinds)

    @staticmethod
    def _score_sample(state: dict[str, Any], confirm_score: float, evidence_kinds: int) -> dict[str, Any]:
        return {
            "confirm_score": float(confirm_score),
            "evidence_kinds": int(evidence_kinds),
            "event_count": int(state.get("event_count", 0)),
            "event_candidate_count": int(len(state.get("candidate_events", {}))),
            "node_candidate_count": int(len(state.get("candidate_nodes", {}))),
            "candidate_risk_component": float(state.get("last_candidate_risk_component", 0.0)),
            "compact_consistency_component": float(state.get("last_compact_consistency_component", 0.0)),
        }

    def observe_causal_episode(
        self,
        episode_record: dict[str, Any],
        node_records: list[dict[str, Any]],
        event_pos: int,
        timestamp_ns: int,
        detail: dict[str, Any],
        event_candidate_crossed: bool = False,
        node_candidate_records: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        self._prune_expired(int(event_pos), int(timestamp_ns))
        src_idx = int(episode_record.get("src_idx", -1))
        dst_idx = int(episode_record.get("dst_idx", -1))
        anchors = self._anchor_keys(src_idx, dst_idx)
        if not anchors:
            return [], [], []
        node_candidate_records = list(node_candidate_records or [])
        if event_candidate_crossed:
            # P0 strict mode: event candidates can confirm node endpoints only if
            # the endpoint is also a node candidate or has strong multi-evidence risk.
            node_candidate_records.extend(
                self._endpoint_node_records_for_confirmation(
                    node_records,
                    node_candidate_records,
                    event_candidate_crossed,
                )
            )
        record = self._record_from_episode(episode_record)
        has_candidate = bool(event_candidate_crossed or node_candidate_records)
        confirmed_event_rows: list[dict[str, Any]] = []
        confirmed_node_rows: list[dict[str, Any]] = []
        samples: list[dict[str, Any]] = []
        for anchor_key, anchor_node in anchors:
            state = self.pending_episodes.get(anchor_key)
            if state is not None and not self._in_window(state, int(event_pos), int(timestamp_ns)):
                self.pending_episodes.pop(anchor_key, None)
                state = None
            if state is None:
                if not has_candidate:
                    continue
                state = self._new_state(anchor_key, anchor_node, record, event_pos, timestamp_ns, src_idx, dst_idx, detail)
                self.pending_episodes[anchor_key] = state
            if event_candidate_crossed:
                self._add_event_candidate(state, episode_record, event_pos, timestamp_ns, detail)
            for candidate_record in node_candidate_records:
                self._add_node_candidate(state, candidate_record, event_pos, timestamp_ns, src_idx, dst_idx, detail)
            self._update_state(state, record, event_pos, timestamp_ns, src_idx, dst_idx, detail)
            confirm_score, evidence_kinds = self._score_state(state)
            state["last_confirm_score"] = float(confirm_score)
            state["last_evidence_kinds"] = int(evidence_kinds)
            sample = self._score_sample(state, confirm_score, evidence_kinds)
            samples.append(sample)
            eligible = (
                int(state.get("event_count", 0)) >= 2
                and evidence_kinds >= self.min_evidence_kinds
            )
            if not eligible:
                continue
            emitted_this_anchor = False
            if confirm_score >= self.event_threshold:
                for candidate in sorted(state.get("candidate_events", {}).values(), key=lambda item: int(item["event_pos"])):
                    candidate_pos = int(candidate.get("event_pos", -1))
                    if candidate_pos in self.emitted_event_positions:
                        continue
                    confirmed_event_rows.append(
                        self._confirmed_event_row(state, candidate, event_pos, timestamp_ns, confirm_score, evidence_kinds)
                    )
                    self.emitted_event_positions.add(candidate_pos)
                    emitted_this_anchor = True
            if confirm_score >= self.node_threshold:
                for candidate in sorted(state.get("candidate_nodes", {}).values(), key=lambda item: int(item["candidate_event_pos"])):
                    node = int(candidate.get("node_id", -1))
                    if node < 0:
                        continue
                    current_risk = max(
                        float(candidate.get("candidate_online_risk", candidate.get("online_risk", 0.0))),
                        float(confirm_score),
                    )
                    should_emit, alert_meta = self.node_alert_policy.maybe_emit(
                        node,
                        current_risk,
                        self.node_threshold,
                        int(event_pos),
                        int(timestamp_ns),
                    )
                    if not should_emit:
                        continue
                    confirmed_node_rows.append(
                        self._confirmed_node_row(state, candidate, event_pos, timestamp_ns, confirm_score, evidence_kinds, alert_meta)
                    )
                    emitted_this_anchor = True
        return confirmed_event_rows, confirmed_node_rows, samples

    def _confirmed_node_row(
        self,
        state: dict[str, Any],
        candidate: dict[str, Any],
        event_pos: int,
        timestamp_ns: int,
        confirm_score: float,
        evidence_kinds: int,
        alert_meta: dict[str, Any],
    ) -> dict[str, Any]:
        delay_events = int(event_pos) - int(candidate["candidate_event_pos"])
        delay_seconds = (int(timestamp_ns) - int(candidate["candidate_timestamp_ns"])) / 1_000_000_000.0
        flags = {key: int(value) for key, value in state.get("evidence_flags", {}).items()}
        return {
            "event_pos": int(event_pos),
            "timestamp_ns": int(timestamp_ns),
            "episode_anchor": str(state.get("anchor_key", "")),
            "node_id": int(candidate["node_id"]),
            "peer_idx": int(candidate.get("peer_idx", -1)),
            "confirm_score": float(confirm_score),
            "confirm_threshold": float(self.node_threshold),
            "candidate_risk_component": float(state.get("last_candidate_risk_component", 0.0)),
            "compact_consistency_component": float(state.get("last_compact_consistency_component", 0.0)),
            "candidate_online_risk": float(candidate.get("candidate_online_risk", 0.0)),
            "max_online_risk": float(state.get("max_online_risk", 0.0)),
            "candidate_first_event_pos": int(candidate.get("candidate_first_event_pos", candidate["candidate_event_pos"])),
            "candidate_first_timestamp_ns": int(candidate.get("candidate_first_timestamp_ns", candidate["candidate_timestamp_ns"])),
            "candidate_update_count": int(candidate.get("candidate_update_count", 1)),
            "candidate_event_pos": int(candidate["candidate_event_pos"]),
            "candidate_timestamp_ns": int(candidate["candidate_timestamp_ns"]),
            "confirmation_delay_events": int(delay_events),
            "confirmation_delay_seconds": float(delay_seconds),
            "event_count": int(state.get("event_count", 0)),
            "evidence_kinds": int(evidence_kinds),
            "role_diversity": int(len(state.get("roles_seen", set()))),
            "peer_diversity": int(len(state.get("peers_seen", set()))),
            "action_diversity": int(len(state.get("actions_seen", set()))),
            "episode_diversity": int(len(state.get("episode_ids_seen", set()))),
            "sum_edge_novelty": float(state.get("sum_edge_novelty", 0.0)),
            "sum_temporal_burst": float(state.get("sum_temporal_burst", 0.0)),
            "sum_episode_support": float(state.get("sum_episode_support", 0.0)),
            "sum_chain_diversity": float(state.get("sum_chain_diversity", 0.0)),
            "max_recent_peer_count": int(state.get("max_recent_peer_count", 0)),
            "max_chain_diversity_units": int(state.get("max_chain_diversity_units", 0)),
            "avg_edge_novelty": float(state.get("last_avg_edge_novelty", 0.0)),
            "avg_temporal_burst": float(state.get("last_avg_temporal_burst", 0.0)),
            "avg_episode_support": float(state.get("last_avg_episode_support", 0.0)),
            "avg_chain_diversity": float(state.get("last_avg_chain_diversity", 0.0)),
            "src_idx": int(candidate.get("src_idx", -1)),
            "dst_idx": int(candidate.get("dst_idx", -1)),
            "action_family": str(candidate.get("action_family", "")),
            "max_field": str(candidate.get("max_field", "")),
            "src_role": str(candidate.get("src_role", "")),
            "dst_role": str(candidate.get("dst_role", "")),
            "alert_version": int(alert_meta.get("alert_version", 1)),
            "alert_is_update": int(alert_meta.get("alert_is_update", 0)),
            "alert_policy": str(alert_meta.get("alert_policy", "")),
            "alert_policy_stage": str(alert_meta.get("alert_policy_stage", "")),
            "alert_policy_reason": str(alert_meta.get("alert_policy_reason", "")),
            "previous_alert_risk": float(alert_meta.get("previous_alert_risk", 0.0)),
            "risk_gain": float(alert_meta.get("risk_gain", 0.0)),
            "risk_gain_ratio": float(alert_meta.get("risk_gain_ratio", 0.0)),
            "state_last_alert_event_pos": int(alert_meta.get("state_last_alert_event_pos", -1)),
            "state_last_alert_timestamp_ns": int(alert_meta.get("state_last_alert_timestamp_ns", -1)),
            "explanation_json": _merge_explanation_json(
                candidate,
                {
                    "stage": "bounded_delay_causal_episode_node_confirmation",
                    "episode_anchor": str(state.get("anchor_key", "")),
                    "confirm_score": float(confirm_score),
                    "confirm_threshold": float(self.node_threshold),
                    "score_form": "candidate_risk_plus_compact_multi_evidence_consistency",
                    "candidate_risk_component": float(state.get("last_candidate_risk_component", 0.0)),
                    "compact_consistency_component": float(state.get("last_compact_consistency_component", 0.0)),
                    "evidence_kinds": int(evidence_kinds),
                    "evidence_flags": flags,
                    "window_seconds": float(self.args.online_confirm_window_seconds),
                    "window_events": int(self.args.online_confirm_window_events),
                    "candidate_event_pos": int(candidate["candidate_event_pos"]),
                    "confirmation_event_pos": int(event_pos),
                    "confirmation_delay_events": int(delay_events),
                    "confirmation_delay_seconds": float(delay_seconds),
                    "alert_version": int(alert_meta.get("alert_version", 1)),
                    "alert_is_update": int(alert_meta.get("alert_is_update", 0)),
                    "alert_policy": str(alert_meta.get("alert_policy", "")),
                    "alert_policy_stage": str(alert_meta.get("alert_policy_stage", "")),
                    "alert_policy_reason": str(alert_meta.get("alert_policy_reason", "")),
                    "previous_alert_risk": float(alert_meta.get("previous_alert_risk", 0.0)),
                    "risk_gain": float(alert_meta.get("risk_gain", 0.0)),
                    "risk_gain_ratio": float(alert_meta.get("risk_gain_ratio", 0.0)),
                    "state_last_alert_event_pos": int(alert_meta.get("state_last_alert_event_pos", -1)),
                    "state_last_alert_timestamp_ns": int(alert_meta.get("state_last_alert_timestamp_ns", -1)),
                },
            ),
        }

    def _confirmed_event_row(
        self,
        state: dict[str, Any],
        candidate: dict[str, Any],
        event_pos: int,
        timestamp_ns: int,
        confirm_score: float,
        evidence_kinds: int,
    ) -> dict[str, Any]:
        delay_events = int(event_pos) - int(candidate["candidate_event_pos"])
        delay_seconds = (int(timestamp_ns) - int(candidate["candidate_timestamp_ns"])) / 1_000_000_000.0
        flags = {key: int(value) for key, value in state.get("evidence_flags", {}).items()}
        row = dict(candidate)
        row.update(
            {
                "event_pos": int(candidate["candidate_event_pos"]),
                "timestamp_ns": int(candidate["candidate_timestamp_ns"]),
                "confirmation_event_pos": int(event_pos),
                "confirmation_timestamp_ns": int(timestamp_ns),
                "episode_anchor": str(state.get("anchor_key", "")),
                "candidate_event_pos": int(candidate["candidate_event_pos"]),
                "candidate_timestamp_ns": int(candidate["candidate_timestamp_ns"]),
                "confirmation_delay_events": int(delay_events),
                "confirmation_delay_seconds": float(delay_seconds),
                "confirm_score": float(confirm_score),
                "confirm_threshold": float(self.event_threshold),
                "candidate_risk_component": float(state.get("last_candidate_risk_component", 0.0)),
                "compact_consistency_component": float(state.get("last_compact_consistency_component", 0.0)),
                "candidate_online_risk": float(candidate.get("candidate_online_risk", 0.0)),
                "max_online_risk": float(state.get("max_online_risk", 0.0)),
                "event_count": int(state.get("event_count", 0)),
                "evidence_kinds": int(evidence_kinds),
                "role_diversity": int(len(state.get("roles_seen", set()))),
                "peer_diversity": int(len(state.get("peers_seen", set()))),
                "action_diversity": int(len(state.get("actions_seen", set()))),
                "episode_diversity": int(len(state.get("episode_ids_seen", set()))),
                "sum_edge_novelty": float(state.get("sum_edge_novelty", 0.0)),
                "sum_temporal_burst": float(state.get("sum_temporal_burst", 0.0)),
                "sum_episode_support": float(state.get("sum_episode_support", 0.0)),
                "sum_chain_diversity": float(state.get("sum_chain_diversity", 0.0)),
                "max_recent_peer_count": int(state.get("max_recent_peer_count", 0)),
                "max_chain_diversity_units": int(state.get("max_chain_diversity_units", 0)),
                "avg_edge_novelty": float(state.get("last_avg_edge_novelty", 0.0)),
                "avg_temporal_burst": float(state.get("last_avg_temporal_burst", 0.0)),
                "avg_episode_support": float(state.get("last_avg_episode_support", 0.0)),
                "avg_chain_diversity": float(state.get("last_avg_chain_diversity", 0.0)),
                "max_field": str(candidate.get("max_field", "")),
                "src_role": str(candidate.get("src_role", "")),
                "dst_role": str(candidate.get("dst_role", "")),
                "event_label": -1,
                "explanation_json": _merge_explanation_json(
                    candidate,
                    {
                        "stage": "bounded_delay_causal_episode_event_confirmation",
                        "episode_anchor": str(state.get("anchor_key", "")),
                        "confirm_score": float(confirm_score),
                        "confirm_threshold": float(self.event_threshold),
                        "score_form": "candidate_risk_plus_compact_multi_evidence_consistency",
                        "candidate_risk_component": float(state.get("last_candidate_risk_component", 0.0)),
                        "compact_consistency_component": float(state.get("last_compact_consistency_component", 0.0)),
                        "evidence_kinds": int(evidence_kinds),
                        "evidence_flags": flags,
                        "candidate_event_pos": int(candidate["candidate_event_pos"]),
                        "confirmation_event_pos": int(event_pos),
                        "confirmation_delay_events": int(delay_events),
                        "confirmation_delay_seconds": float(delay_seconds),
                        "candidate_first_event_pos": int(candidate.get("candidate_first_event_pos", candidate["candidate_event_pos"])),
                        "candidate_first_timestamp_ns": int(candidate.get("candidate_first_timestamp_ns", candidate["candidate_timestamp_ns"])),
                        "candidate_update_count": int(candidate.get("candidate_update_count", 1)),
                        "window_seconds": float(self.args.online_confirm_window_seconds),
                        "window_events": int(self.args.online_confirm_window_events),
                },
            ),
        }
        )
        return row


def _field_keys(fields: dict[str, str]) -> list[tuple[str, str, str]]:
    src_role_base = f"{fields['src_kind']}|{fields['src_head']}"
    dst_role_base = f"{fields['dst_kind']}|{fields['dst_head']}"
    return [
        ("action_given_src", src_role_base, fields["action_family"]),
        ("dst_given_src_action", f"{src_role_base}|{fields['action_family']}", f"{fields['dst_kind']}|{fields['dst_head']}"),
        ("src_given_dst_action", f"{dst_role_base}|{fields['action_family']}", f"{fields['src_kind']}|{fields['src_head']}"),
        ("object_given_action", fields["action_family"], fields["object_type"]),
        ("relation_given_roles", f"{fields['src_kind']}|{fields['dst_kind']}", fields["relation"]),
    ]


class CausalSemanticModel:
    def __init__(
        self,
        vocab_size: int,
        min_token_count: int,
        max_role_keys: int,
        max_pair_keys: int,
        smoothing: float,
        role_min_count: int,
        activity_log_base: float,
        score_clip: float,
        semantic_string_policy: str = "raw",
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.min_token_count = int(min_token_count)
        self.max_role_keys = int(max_role_keys)
        self.max_pair_keys = int(max_pair_keys)
        self.smoothing = float(smoothing)
        self.role_min_count = int(role_min_count)
        self.activity_log_base = float(activity_log_base)
        self.score_clip = float(score_clip)
        self.semantic_string_policy = _semantic_policy(semantic_string_policy)
        self.token_counts: Counter[str] = Counter()
        self.vocab: dict[str, int] = {}
        self.global_field_counts: dict[str, Counter[str]] = {}
        self.context_counts: dict[str, Counter[str]] = {}
        self.pair_counts: dict[tuple[str, str], Counter[str]] = {}
        self.role_counts: Counter[str] = Counter()
        self.role_vocab: dict[str, int] = {}
        self.role_totals: Counter[str] = Counter()
        self.field_value_sizes: dict[str, int] = {}
        self.validation_scores: list[float] = []
        self.validation_quantiles: dict[str, float] = {}

    def observe_vocab(self, row: dict[str, Any], max_tokens_per_node: int) -> None:
        fields = _row_fields(row, max_tokens_per_node, self.semantic_string_policy)
        self.token_counts.update(_event_tokens(fields, self.semantic_string_policy))

    def freeze_vocab(self) -> None:
        items = [
            (tok, cnt)
            for tok, cnt in self.token_counts.items()
            if int(cnt) >= int(self.min_token_count)
        ]
        items.sort(key=lambda kv: (-int(kv[1]), kv[0]))
        self.vocab = {tok: idx for idx, (tok, _cnt) in enumerate(items[: self.vocab_size])}

    def _bounded_context_key(self, field: str, context: str) -> str:
        key = f"{field}|{context}"
        if len(self.context_counts) < self.max_role_keys or key in self.context_counts:
            return key
        return f"{field}|__OTHER_CTX__|{_stable_hash(context)}"

    def _bounded_pair_key(self, field: str, context: str) -> tuple[str, str]:
        key = (field, context)
        if len(self.pair_counts) < self.max_pair_keys or key in self.pair_counts:
            return key
        return (field, f"__OTHER_PAIR__|{_stable_hash(context)}")

    def observe_train(self, row: dict[str, Any], max_tokens_per_node: int) -> None:
        fields = _row_fields(row, max_tokens_per_node, self.semantic_string_policy)
        src_role = f"{fields['src_kind']}|{fields['src_head']}"
        dst_role = f"{fields['dst_kind']}|{fields['dst_head']}"
        self.role_counts[src_role] += 1
        self.role_counts[dst_role] += 1
        for field, context, value in _field_keys(fields):
            self.global_field_counts.setdefault(field, Counter())[value] += 1
            self.context_counts.setdefault(self._bounded_context_key(field, context), Counter())[value] += 1
            self.pair_counts.setdefault(self._bounded_pair_key(field, context), Counter())[value] += 1

    def freeze_roles(self) -> None:
        roles = [(role, count) for role, count in self.role_counts.items() if count >= self.role_min_count]
        roles.sort(key=lambda kv: (-int(kv[1]), kv[0]))
        self.role_vocab = {role: idx for idx, (role, _count) in enumerate(roles)}
        self.field_value_sizes = {
            field: max(1, len(counter))
            for field, counter in self.global_field_counts.items()
        }

    def score_row(self, row: dict[str, Any], max_tokens_per_node: int) -> tuple[float, dict[str, float | str]]:
        fields = _row_fields(row, max_tokens_per_node, self.semantic_string_policy)
        field_scores: list[float] = []
        detail: dict[str, float | str] = {}
        max_field = ("", 0.0)
        for field, context, value in _field_keys(fields):
            global_counter = self.global_field_counts.get(field, Counter())
            global_total = sum(global_counter.values())
            value_space = max(int(self.field_value_sizes.get(field, len(global_counter) or 1)), 1)
            global_prob = (float(global_counter.get(value, 0)) + self.smoothing) / (
                float(global_total) + self.smoothing * (value_space + 1)
            )
            ctx_counter = self.context_counts.get(self._bounded_context_key(field, context), Counter())
            ctx_total = sum(ctx_counter.values())
            if ctx_total > 0:
                ctx_prob = (float(ctx_counter.get(value, 0)) + self.smoothing) / (
                    float(ctx_total) + self.smoothing * (value_space + 1)
                )
            else:
                ctx_prob = global_prob
            pair_counter = self.pair_counts.get(self._bounded_pair_key(field, context), Counter())
            pair_total = sum(pair_counter.values())
            if pair_total > 0:
                pair_prob = (float(pair_counter.get(value, 0)) + self.smoothing) / (
                    float(pair_total) + self.smoothing * (value_space + 1)
                )
            else:
                pair_prob = ctx_prob
            nll = -math.log(max(min(pair_prob, ctx_prob), 1e-12))
            nll = min(nll, self.score_clip)
            field_scores.append(nll)
            detail[f"{field}_nll"] = float(nll)
            if nll > max_field[1]:
                max_field = (field, nll)
        token_unseen = 0
        tokens = _event_tokens(fields, self.semantic_string_policy)
        for token in tokens:
            if token not in self.vocab:
                token_unseen += 1
        token_unseen_ratio = float(token_unseen / max(len(tokens), 1))
        score = float(sum(field_scores) / max(len(field_scores), 1))
        score += 1.5 * token_unseen_ratio
        detail.update(
            {
                "score": float(score),
                "token_unseen_ratio": float(token_unseen_ratio),
                "max_field": str(max_field[0]),
                "max_field_nll": float(max_field[1]),
                "src_role": f"{fields['src_kind']}|{fields['src_head']}",
                "dst_role": f"{fields['dst_kind']}|{fields['dst_head']}",
                "action_family": fields["action_family"],
                "semantic_string_policy": self.semantic_string_policy,
            }
        )
        return score, detail

    def calibrate_validation(self, scores: list[float]) -> None:
        self.validation_scores = [float(x) for x in scores if math.isfinite(float(x))]
        arr = np.asarray(self.validation_scores, dtype=np.float32)
        if arr.size == 0:
            self.validation_quantiles = {}
            return
        self.validation_quantiles = {
            "q50": float(np.quantile(arr, 0.50)),
            "q90": float(np.quantile(arr, 0.90)),
            "q95": float(np.quantile(arr, 0.95)),
            "q99": float(np.quantile(arr, 0.99)),
            "q999": float(np.quantile(arr, 0.999)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }


def _node_sweep(
    node_score: np.ndarray,
    all_nodes: np.ndarray,
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    topk_values: list[int],
) -> list[dict[str, object]]:
    ranked = rank_scores(node_score)
    sweep: list[dict[str, object]] = []
    for topk in topk_values:
        alerted = np.zeros((node_score.shape[0],), dtype=bool)
        keep = ranked[: min(int(topk), ranked.shape[0])]
        alerted[keep] = True
        sweep.append(
            {
                "topk": int(topk),
                "num_alerted_nodes": int(keep.shape[0]),
                "strict_node": node_confusion_from_masks(all_nodes, positive_nodes, suspect_nodes, alerted, relaxed=False),
                "relaxed_node": node_confusion_from_masks(all_nodes, positive_nodes, suspect_nodes, alerted, relaxed=True),
            }
        )
    return sweep


def _write_ranked_csv(
    path: str,
    ranked: np.ndarray,
    node_score: np.ndarray,
    node_score_max: np.ndarray,
    node_score_tail_excess: np.ndarray,
    node_score_activity_adjusted: np.ndarray,
    node_score_predictable_activity: np.ndarray,
    node_score_attack_evidence: np.ndarray,
    node_residual_fusion_max: np.ndarray,
    node_predictable_activity_credit: np.ndarray,
    node_repeated_pattern_max: np.ndarray,
    node_global_pattern_nodes_max: np.ndarray,
    node_predictable_event_count: np.ndarray,
    node_attack_edge_component: np.ndarray,
    node_attack_burst_component: np.ndarray,
    node_attack_episode_component: np.ndarray,
    node_attack_chain_component: np.ndarray,
    node_episode_recent_peer_max: np.ndarray,
    node_chain_diversity_max: np.ndarray,
    node_attack_episode_event_count: np.ndarray,
    all_nodes: np.ndarray,
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    node_count: np.ndarray,
    node_top_event: np.ndarray,
    node_top_label: np.ndarray,
    node_top_field: list[str],
    node_top_detail: list[dict[str, Any] | None],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "rank",
        "node_id",
        "score",
        "score_max",
        "score_tail_excess",
        "score_activity_adjusted",
        "score_predictable_activity",
        "score_attack_evidence",
        "residual_fusion_max",
        "predictable_activity_credit",
        "repeated_pattern_max",
        "global_pattern_nodes_max",
        "predictable_event_count",
        "attack_edge_novelty",
        "attack_temporal_burst",
        "attack_episode_support",
        "attack_chain_diversity",
        "episode_recent_peer_max",
        "chain_diversity_max",
        "attack_episode_event_count",
        "observed",
        "is_gt_positive",
        "is_suspect",
        "event_count",
        "top_event_pos",
        "top_event_label",
        "top_field",
        "token_unseen_ratio",
        "src_role",
        "dst_role",
        "action_family",
        "explanation_json",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, node in enumerate(ranked, start=1):
            detail = node_top_detail[int(node)] or {}
            writer.writerow(
                {
                    "rank": int(rank),
                    "node_id": int(node),
                    "score": float(node_score[int(node)]),
                    "score_max": float(node_score_max[int(node)]),
                    "score_tail_excess": float(node_score_tail_excess[int(node)]),
                    "score_activity_adjusted": float(node_score_activity_adjusted[int(node)]),
                    "score_predictable_activity": float(node_score_predictable_activity[int(node)]),
                    "score_attack_evidence": float(node_score_attack_evidence[int(node)]),
                    "residual_fusion_max": float(node_residual_fusion_max[int(node)]),
                    "predictable_activity_credit": float(node_predictable_activity_credit[int(node)]),
                    "repeated_pattern_max": int(node_repeated_pattern_max[int(node)]),
                    "global_pattern_nodes_max": int(node_global_pattern_nodes_max[int(node)]),
                    "predictable_event_count": int(node_predictable_event_count[int(node)]),
                    "attack_edge_novelty": float(node_attack_edge_component[int(node)]),
                    "attack_temporal_burst": float(node_attack_burst_component[int(node)]),
                    "attack_episode_support": float(node_attack_episode_component[int(node)]),
                    "attack_chain_diversity": float(node_attack_chain_component[int(node)]),
                    "episode_recent_peer_max": int(node_episode_recent_peer_max[int(node)]),
                    "chain_diversity_max": int(node_chain_diversity_max[int(node)]),
                    "attack_episode_event_count": int(node_attack_episode_event_count[int(node)]),
                    "observed": int(bool(all_nodes[int(node)])),
                    "is_gt_positive": int(bool(positive_nodes[int(node)])),
                    "is_suspect": int(bool(suspect_nodes[int(node)])),
                    "event_count": int(node_count[int(node)]),
                    "top_event_pos": int(node_top_event[int(node)]),
                    "top_event_label": int(node_top_label[int(node)]),
                    "top_field": node_top_field[int(node)],
                    "token_unseen_ratio": float(detail.get("token_unseen_ratio", 0.0)),
                    "src_role": str(detail.get("src_role", "")),
                    "dst_role": str(detail.get("dst_role", "")),
                    "action_family": str(detail.get("action_family", "")),
                    "explanation_json": json.dumps(detail, sort_keys=True),
                }
            )


def _write_top_events_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "event_rank",
        "event_pos",
        "timestamp_ns",
        "score",
        "label",
        "src_idx",
        "dst_idx",
        "max_field",
        "max_field_nll",
        "token_unseen_ratio",
        "causal_semantic_score",
        "residual_fusion_score",
        "src_role",
        "dst_role",
        "action_family",
    ]
    rows = sorted(rows, key=lambda row: (-float(row["score"]), int(row["event_pos"])))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            out = {key: row.get(key, "") for key in fieldnames}
            out["event_rank"] = int(idx)
            writer.writerow(out)


def _online_node_alert_metrics(
    rows: list[dict[str, Any]],
    all_nodes: np.ndarray,
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    first_positive_event: np.ndarray,
) -> dict[str, Any]:
    positive_observed = positive_nodes & all_nodes
    num_positive = int(np.sum(positive_observed))
    rows_by_node: dict[int, list[dict[str, Any]]] = {}
    tp_any = 0
    tp_after = 0
    early_positive = 0
    fp_strict = 0
    fp_relaxed = 0
    ignored_relaxed = 0
    delays_events: list[int] = []
    for row in rows:
        node = int(row["node_id"])
        if 0 <= node < positive_nodes.shape[0]:
            rows_by_node.setdefault(node, []).append(row)
        is_positive = bool(positive_nodes[node])
        is_suspect = bool(suspect_nodes[node])
        first_pos = int(first_positive_event[node])
        alert_pos = int(row["event_pos"])
        if is_positive:
            tp_any += 1
            if first_pos >= 0 and alert_pos >= first_pos:
                tp_after += 1
                delays_events.append(alert_pos - first_pos)
            else:
                early_positive += 1
        else:
            fp_strict += 1
            if is_suspect:
                ignored_relaxed += 1
            else:
                fp_relaxed += 1
    total = int(len(rows))
    unique_metrics = _unique_online_node_alert_metrics(
        rows_by_node,
        num_positive,
        positive_nodes,
        suspect_nodes,
        first_positive_event,
    )
    return {
        "metric_level": "alert_row",
        "num_alerts": total,
        "num_unique_alerted_nodes": int(len(rows_by_node)),
        "num_positive_nodes": num_positive,
        "tp_anytime": int(tp_any),
        "tp_after_attack_start": int(tp_after),
        "early_positive_alerts": int(early_positive),
        "fp_strict": int(fp_strict),
        "fp_relaxed": int(fp_relaxed),
        "ignored_suspect_relaxed": int(ignored_relaxed),
        "precision_anytime_strict": float(tp_any / max(tp_any + fp_strict, 1)),
        "recall_anytime": float(tp_any / max(num_positive, 1)),
        "precision_after_attack_strict": float(tp_after / max(tp_after + fp_strict + early_positive, 1)),
        "recall_after_attack": float(tp_after / max(num_positive, 1)),
        "precision_after_attack_relaxed": float(tp_after / max(tp_after + fp_relaxed + early_positive, 1)),
        "mean_delay_events": float(np.mean(delays_events)) if delays_events else None,
        "median_delay_events": float(np.median(delays_events)) if delays_events else None,
        "p90_delay_events": float(np.quantile(np.asarray(delays_events, dtype=np.float32), 0.90)) if delays_events else None,
        "unique_node_metrics": unique_metrics,
    }


def _unique_online_node_alert_metrics(
    rows_by_node: dict[int, list[dict[str, Any]]],
    num_positive: int,
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    first_positive_event: np.ndarray,
) -> dict[str, Any]:
    tp_any = 0
    tp_after = 0
    early_positive = 0
    fp_strict = 0
    fp_relaxed = 0
    ignored_relaxed = 0
    delays_events: list[int] = []
    update_rows = 0
    updated_nodes = 0
    max_alert_version = 0
    for node, node_rows in rows_by_node.items():
        if not node_rows:
            continue
        node_rows = sorted(node_rows, key=lambda row: (int(row.get("event_pos", -1)), int(row.get("alert_rank", 0) or 0)))
        first_row = node_rows[0]
        is_positive = bool(positive_nodes[node])
        is_suspect = bool(suspect_nodes[node])
        first_pos = int(first_positive_event[node])
        alert_positions = [int(row.get("event_pos", -1)) for row in node_rows]
        after_positions = [pos for pos in alert_positions if first_pos >= 0 and pos >= first_pos]
        node_update_rows = sum(int(row.get("alert_is_update", 0) or 0) for row in node_rows)
        update_rows += int(node_update_rows)
        if node_update_rows > 0:
            updated_nodes += 1
        max_alert_version = max(max_alert_version, max((int(row.get("alert_version", 0) or 0) for row in node_rows), default=0))
        if is_positive:
            tp_any += 1
            if after_positions:
                tp_after += 1
                delays_events.append(min(after_positions) - first_pos)
            else:
                early_positive += 1
        else:
            fp_strict += 1
            if is_suspect:
                ignored_relaxed += 1
            else:
                fp_relaxed += 1
    unique_total = int(len(rows_by_node))
    return {
        "metric_level": "unique_node",
        "num_unique_alerted_nodes": unique_total,
        "num_positive_nodes": int(num_positive),
        "tp_anytime_nodes": int(tp_any),
        "tp_after_attack_start_nodes": int(tp_after),
        "early_positive_nodes": int(early_positive),
        "fp_strict_nodes": int(fp_strict),
        "fp_relaxed_nodes": int(fp_relaxed),
        "ignored_suspect_relaxed_nodes": int(ignored_relaxed),
        "precision_anytime_strict": float(tp_any / max(tp_any + fp_strict, 1)),
        "recall_anytime": float(tp_any / max(num_positive, 1)),
        "precision_after_attack_strict": float(tp_after / max(tp_after + fp_strict + early_positive, 1)),
        "recall_after_attack": float(tp_after / max(num_positive, 1)),
        "precision_after_attack_relaxed": float(tp_after / max(tp_after + fp_relaxed + early_positive, 1)),
        "mean_first_delay_events": float(np.mean(delays_events)) if delays_events else None,
        "median_first_delay_events": float(np.median(delays_events)) if delays_events else None,
        "p90_first_delay_events": float(np.quantile(np.asarray(delays_events, dtype=np.float32), 0.90)) if delays_events else None,
        "update_rows": int(update_rows),
        "updated_nodes": int(updated_nodes),
        "max_alert_version": int(max_alert_version),
    }


def _online_episode_alert_metrics(
    rows: list[dict[str, Any]],
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    first_positive_event: np.ndarray,
) -> dict[str, Any]:
    tp_any = 0
    tp_after = 0
    fp_strict = 0
    fp_relaxed = 0
    ignored_relaxed = 0
    early_positive = 0
    for row in rows:
        src = int(row["src_idx"])
        dst = int(row["dst_idx"])
        pos_events = [
            int(first_positive_event[node])
            for node in [src, dst]
            if 0 <= node < positive_nodes.shape[0] and bool(positive_nodes[node]) and int(first_positive_event[node]) >= 0
        ]
        has_positive = bool(pos_events)
        has_suspect = any(0 <= node < suspect_nodes.shape[0] and bool(suspect_nodes[node]) for node in [src, dst])
        alert_pos = int(row["event_pos"])
        if has_positive:
            tp_any += 1
            first_pos = min(pos_events)
            if alert_pos >= first_pos:
                tp_after += 1
            else:
                early_positive += 1
        else:
            fp_strict += 1
            if has_suspect:
                ignored_relaxed += 1
            else:
                fp_relaxed += 1
    return {
        "num_alerts": int(len(rows)),
        "tp_any_endpoint": int(tp_any),
        "tp_after_attack_start": int(tp_after),
        "early_positive_endpoint_alerts": int(early_positive),
        "fp_strict": int(fp_strict),
        "fp_relaxed": int(fp_relaxed),
        "ignored_suspect_relaxed": int(ignored_relaxed),
        "precision_any_endpoint_strict": float(tp_any / max(tp_any + fp_strict, 1)),
        "precision_after_attack_strict": float(tp_after / max(tp_after + fp_strict + early_positive, 1)),
        "precision_after_attack_relaxed": float(tp_after / max(tp_after + fp_relaxed + early_positive, 1)),
    }


def _online_event_alert_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    suspicious = 0
    malicious = 0
    missing = 0
    delay_events: list[int] = []
    delay_seconds: list[float] = []
    for row in rows:
        label = int(row.get("event_label", row.get("label", 0)))
        if label < 0:
            missing += 1
        if label == 1:
            suspicious += 1
        elif label == 2:
            malicious += 1
        if "confirmation_delay_events" in row and row.get("confirmation_delay_events", "") != "":
            delay_events.append(int(row["confirmation_delay_events"]))
        if "confirmation_delay_seconds" in row and row.get("confirmation_delay_seconds", "") != "":
            delay_seconds.append(float(row["confirmation_delay_seconds"]))
    correct = int(suspicious + malicious)
    total = int(len(rows))
    return {
        "num_alerts": total,
        "correct_event_alerts": correct,
        "correct_alert_ratio": float(correct / max(total, 1)),
        "suspicious_event_alerts": int(suspicious),
        "malicious_event_alerts": int(malicious),
        "false_event_alerts": int(total - correct),
        "missing_event_labels": int(missing),
        "mean_confirmation_delay_events": float(np.mean(delay_events)) if delay_events else None,
        "median_confirmation_delay_events": float(np.median(delay_events)) if delay_events else None,
        "p90_confirmation_delay_events": float(np.quantile(np.asarray(delay_events, dtype=np.float32), 0.90)) if delay_events else None,
        "mean_confirmation_delay_seconds": float(np.mean(delay_seconds)) if delay_seconds else None,
        "median_confirmation_delay_seconds": float(np.median(delay_seconds)) if delay_seconds else None,
        "p90_confirmation_delay_seconds": float(np.quantile(np.asarray(delay_seconds, dtype=np.float32), 0.90)) if delay_seconds else None,
        "correct_rule": "event_label in {1, 2}; labels are used only after online threshold emission for evaluation",
    }


def _write_online_event_alerts_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "alert_rank",
        "event_pos",
        "timestamp_ns",
        "episode_id",
        "src_idx",
        "dst_idx",
        "online_risk",
        "threshold",
        "base_online_risk",
        "event_label",
        "is_correct_event_alert",
        "is_suspicious_event",
        "is_malicious_event",
        "event_score",
        "event_causal_semantic_score",
        "event_residual_fusion_score",
        "multi_view_behavior_score",
        "multi_view_behavior_mode",
        "multi_view_behavior_policies",
        "multi_view_top_component",
        "multi_view_top_contrib",
        "learned_evidence_score",
        "learned_evidence_mode",
        "learned_evidence_event_score_contrib",
        "learned_evidence_causal_semantic_contrib",
        "learned_evidence_residual_fusion_contrib",
        "learned_evidence_edge_novelty_contrib",
        "learned_evidence_temporal_burst_contrib",
        "learned_evidence_episode_support_contrib",
        "learned_evidence_chain_diversity_contrib",
        "event_edge_novelty",
        "event_temporal_burst",
        "event_episode_support",
        "event_chain_diversity",
        "episode_recent_peer_count",
        "chain_diversity_units",
        "chain_peer_kind_count",
        "chain_action_family_count",
        "chain_transition_count",
        "action_family",
        "max_field",
        "src_role",
        "dst_role",
        "explanation_json",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            label = int(row.get("event_label", row.get("label", 0)))
            out = {key: row.get(key, "") for key in fieldnames}
            out.update(
                {
                    "alert_rank": int(rank),
                    "event_label": int(label),
                    "is_correct_event_alert": int(label in {1, 2}),
                    "is_suspicious_event": int(label == 1),
                    "is_malicious_event": int(label == 2),
                }
            )
            writer.writerow(out)


def _write_confirmed_event_alerts_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "alert_rank",
        "event_pos",
        "timestamp_ns",
        "confirmation_event_pos",
        "confirmation_timestamp_ns",
        "confirmation_delay_events",
        "confirmation_delay_seconds",
        "episode_anchor",
        "episode_id",
        "src_idx",
        "dst_idx",
        "confirm_score",
        "confirm_threshold",
        "candidate_risk_component",
        "compact_consistency_component",
        "candidate_online_risk",
        "max_online_risk",
        "candidate_event_pos",
        "candidate_timestamp_ns",
        "event_label",
        "is_correct_event_alert",
        "is_suspicious_event",
        "is_malicious_event",
        "event_count",
        "evidence_kinds",
        "role_diversity",
        "peer_diversity",
        "action_diversity",
        "episode_diversity",
        "sum_edge_novelty",
        "sum_temporal_burst",
        "sum_episode_support",
        "sum_chain_diversity",
        "max_recent_peer_count",
        "max_chain_diversity_units",
        "avg_edge_novelty",
        "avg_temporal_burst",
        "avg_episode_support",
        "avg_chain_diversity",
        "event_score",
        "event_causal_semantic_score",
        "event_residual_fusion_score",
        "multi_view_behavior_score",
        "multi_view_behavior_mode",
        "multi_view_behavior_policies",
        "multi_view_top_component",
        "multi_view_top_contrib",
        "learned_evidence_score",
        "learned_evidence_mode",
        "learned_evidence_event_score_contrib",
        "learned_evidence_causal_semantic_contrib",
        "learned_evidence_residual_fusion_contrib",
        "learned_evidence_edge_novelty_contrib",
        "learned_evidence_temporal_burst_contrib",
        "learned_evidence_episode_support_contrib",
        "learned_evidence_chain_diversity_contrib",
        "event_edge_novelty",
        "event_temporal_burst",
        "event_episode_support",
        "event_chain_diversity",
        "episode_recent_peer_count",
        "chain_diversity_units",
        "chain_peer_kind_count",
        "chain_action_family_count",
        "chain_transition_count",
        "action_family",
        "max_field",
        "src_role",
        "dst_role",
        "explanation_json",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            label = int(row.get("event_label", row.get("label", 0)))
            out = {key: row.get(key, "") for key in fieldnames}
            out.update(
                {
                    "alert_rank": int(rank),
                    "event_label": int(label),
                    "is_correct_event_alert": int(label in {1, 2}),
                    "is_suspicious_event": int(label == 1),
                    "is_malicious_event": int(label == 2),
                }
            )
            writer.writerow(out)


def _write_online_node_alerts_csv(
    path: str,
    rows: list[dict[str, Any]],
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    first_positive_event: np.ndarray,
    first_positive_ts: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "alert_rank",
        "event_pos",
        "timestamp_ns",
        "node_id",
        "peer_idx",
        "online_risk",
        "threshold",
        "base_online_risk",
        "candidate_update_count",
        "candidate_first_event_pos",
        "candidate_first_timestamp_ns",
        "alert_version",
        "alert_is_update",
        "alert_policy",
        "alert_policy_stage",
        "alert_policy_reason",
        "previous_alert_risk",
        "risk_gain",
        "risk_gain_ratio",
        "state_last_alert_event_pos",
        "state_last_alert_timestamp_ns",
        "event_score",
        "event_causal_semantic_score",
        "event_residual_fusion_score",
        "multi_view_behavior_score",
        "multi_view_behavior_mode",
        "multi_view_behavior_policies",
        "multi_view_top_component",
        "multi_view_top_contrib",
        "learned_evidence_score",
        "learned_evidence_mode",
        "learned_evidence_event_score_contrib",
        "learned_evidence_causal_semantic_contrib",
        "learned_evidence_residual_fusion_contrib",
        "learned_evidence_edge_novelty_contrib",
        "learned_evidence_temporal_burst_contrib",
        "learned_evidence_episode_support_contrib",
        "learned_evidence_chain_diversity_contrib",
        "event_edge_novelty",
        "event_temporal_burst",
        "event_episode_support",
        "event_chain_diversity",
        "episode_recent_peer_count",
        "chain_diversity_units",
        "chain_peer_kind_count",
        "chain_action_family_count",
        "chain_transition_count",
        "src_idx",
        "dst_idx",
        "action_family",
        "max_field",
        "src_role",
        "dst_role",
        "is_gt_positive",
        "is_suspect",
        "first_positive_event_pos",
        "detection_delay_events",
        "detection_delay_seconds",
        "is_tp_anytime",
        "is_tp_after_attack_start",
        "is_early_positive_alert",
        "is_fp_strict",
        "is_fp_relaxed",
        "explanation_json",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            node = int(row["node_id"])
            is_positive = bool(positive_nodes[node])
            is_suspect = bool(suspect_nodes[node])
            first_pos = int(first_positive_event[node])
            first_ts = int(first_positive_ts[node])
            alert_pos = int(row["event_pos"])
            alert_ts = int(row["timestamp_ns"])
            delay_events = alert_pos - first_pos if is_positive and first_pos >= 0 and alert_pos >= first_pos else ""
            delay_seconds = (alert_ts - first_ts) / 1_000_000_000.0 if is_positive and first_ts >= 0 and alert_ts >= first_ts else ""
            is_tp_after = bool(is_positive and first_pos >= 0 and alert_pos >= first_pos)
            is_early = bool(is_positive and first_pos >= 0 and alert_pos < first_pos)
            out = {key: row.get(key, "") for key in fieldnames}
            out.update(
                {
                    "alert_rank": int(rank),
                    "is_gt_positive": int(is_positive),
                    "is_suspect": int(is_suspect),
                    "first_positive_event_pos": int(first_pos),
                    "detection_delay_events": delay_events,
                    "detection_delay_seconds": delay_seconds,
                    "is_tp_anytime": int(is_positive),
                    "is_tp_after_attack_start": int(is_tp_after),
                    "is_early_positive_alert": int(is_early),
                    "is_fp_strict": int(not is_positive),
                    "is_fp_relaxed": int((not is_positive) and (not is_suspect)),
                }
            )
            writer.writerow(out)


def _write_confirmed_node_alerts_csv(
    path: str,
    rows: list[dict[str, Any]],
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    first_positive_event: np.ndarray,
    first_positive_ts: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "alert_rank",
        "event_pos",
        "timestamp_ns",
        "episode_anchor",
        "node_id",
        "peer_idx",
        "confirm_score",
        "confirm_threshold",
        "base_online_risk",
        "candidate_risk_component",
        "compact_consistency_component",
        "candidate_online_risk",
        "max_online_risk",
        "candidate_update_count",
        "candidate_first_event_pos",
        "candidate_first_timestamp_ns",
        "candidate_event_pos",
        "candidate_timestamp_ns",
        "alert_version",
        "alert_is_update",
        "alert_policy",
        "alert_policy_stage",
        "alert_policy_reason",
        "previous_alert_risk",
        "risk_gain",
        "risk_gain_ratio",
        "state_last_alert_event_pos",
        "state_last_alert_timestamp_ns",
        "confirmation_delay_events",
        "confirmation_delay_seconds",
        "event_count",
        "evidence_kinds",
        "role_diversity",
        "peer_diversity",
        "action_diversity",
        "episode_diversity",
        "sum_edge_novelty",
        "sum_temporal_burst",
        "sum_episode_support",
        "sum_chain_diversity",
        "max_recent_peer_count",
        "max_chain_diversity_units",
        "avg_edge_novelty",
        "avg_temporal_burst",
        "avg_episode_support",
        "avg_chain_diversity",
        "event_score",
        "event_causal_semantic_score",
        "event_residual_fusion_score",
        "multi_view_behavior_score",
        "multi_view_behavior_mode",
        "multi_view_behavior_policies",
        "multi_view_top_component",
        "multi_view_top_contrib",
        "learned_evidence_score",
        "learned_evidence_mode",
        "learned_evidence_event_score_contrib",
        "learned_evidence_causal_semantic_contrib",
        "learned_evidence_residual_fusion_contrib",
        "learned_evidence_edge_novelty_contrib",
        "learned_evidence_temporal_burst_contrib",
        "learned_evidence_episode_support_contrib",
        "learned_evidence_chain_diversity_contrib",
        "event_edge_novelty",
        "event_temporal_burst",
        "event_episode_support",
        "event_chain_diversity",
        "src_idx",
        "dst_idx",
        "action_family",
        "max_field",
        "src_role",
        "dst_role",
        "is_gt_positive",
        "is_suspect",
        "first_positive_event_pos",
        "detection_delay_events",
        "detection_delay_seconds",
        "is_tp_anytime",
        "is_tp_after_attack_start",
        "is_early_positive_alert",
        "is_fp_strict",
        "is_fp_relaxed",
        "explanation_json",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            node = int(row["node_id"])
            is_positive = bool(positive_nodes[node])
            is_suspect = bool(suspect_nodes[node])
            first_pos = int(first_positive_event[node])
            first_ts = int(first_positive_ts[node])
            alert_pos = int(row["event_pos"])
            alert_ts = int(row["timestamp_ns"])
            delay_events = alert_pos - first_pos if is_positive and first_pos >= 0 and alert_pos >= first_pos else ""
            delay_seconds = (alert_ts - first_ts) / 1_000_000_000.0 if is_positive and first_ts >= 0 and alert_ts >= first_ts else ""
            is_tp_after = bool(is_positive and first_pos >= 0 and alert_pos >= first_pos)
            is_early = bool(is_positive and first_pos >= 0 and alert_pos < first_pos)
            out = {key: row.get(key, "") for key in fieldnames}
            out.update(
                {
                    "alert_rank": int(rank),
                    "is_gt_positive": int(is_positive),
                    "is_suspect": int(is_suspect),
                    "first_positive_event_pos": int(first_pos),
                    "detection_delay_events": delay_events,
                    "detection_delay_seconds": delay_seconds,
                    "is_tp_anytime": int(is_positive),
                    "is_tp_after_attack_start": int(is_tp_after),
                    "is_early_positive_alert": int(is_early),
                    "is_fp_strict": int(not is_positive),
                    "is_fp_relaxed": int((not is_positive) and (not is_suspect)),
                }
            )
            writer.writerow(out)


def _write_online_episode_alerts_csv(
    path: str,
    rows: list[dict[str, Any]],
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    first_positive_event: np.ndarray,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "alert_rank",
        "event_pos",
        "timestamp_ns",
        "episode_id",
        "src_idx",
        "dst_idx",
        "online_risk",
        "threshold",
        "base_online_risk",
        "event_score",
        "event_causal_semantic_score",
        "event_residual_fusion_score",
        "multi_view_behavior_score",
        "multi_view_behavior_mode",
        "multi_view_behavior_policies",
        "multi_view_top_component",
        "multi_view_top_contrib",
        "learned_evidence_score",
        "learned_evidence_mode",
        "learned_evidence_event_score_contrib",
        "learned_evidence_causal_semantic_contrib",
        "learned_evidence_residual_fusion_contrib",
        "learned_evidence_edge_novelty_contrib",
        "learned_evidence_temporal_burst_contrib",
        "learned_evidence_episode_support_contrib",
        "learned_evidence_chain_diversity_contrib",
        "event_edge_novelty",
        "event_temporal_burst",
        "event_episode_support",
        "event_chain_diversity",
        "episode_recent_peer_count",
        "chain_diversity_units",
        "chain_peer_kind_count",
        "chain_action_family_count",
        "chain_transition_count",
        "action_family",
        "max_field",
        "src_role",
        "dst_role",
        "src_is_gt_positive",
        "dst_is_gt_positive",
        "src_is_suspect",
        "dst_is_suspect",
        "is_tp_any_endpoint",
        "is_tp_after_attack_start",
        "is_fp_strict",
        "is_fp_relaxed",
        "explanation_json",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            src = int(row["src_idx"])
            dst = int(row["dst_idx"])
            src_pos = bool(0 <= src < positive_nodes.shape[0] and positive_nodes[src])
            dst_pos = bool(0 <= dst < positive_nodes.shape[0] and positive_nodes[dst])
            src_sus = bool(0 <= src < suspect_nodes.shape[0] and suspect_nodes[src])
            dst_sus = bool(0 <= dst < suspect_nodes.shape[0] and suspect_nodes[dst])
            pos_events = [
                int(first_positive_event[node])
                for node in [src, dst]
                if 0 <= node < positive_nodes.shape[0] and bool(positive_nodes[node]) and int(first_positive_event[node]) >= 0
            ]
            is_tp_any = bool(pos_events)
            is_tp_after = bool(pos_events and int(row["event_pos"]) >= min(pos_events))
            out = {key: row.get(key, "") for key in fieldnames}
            out.update(
                {
                    "alert_rank": int(rank),
                    "src_is_gt_positive": int(src_pos),
                    "dst_is_gt_positive": int(dst_pos),
                    "src_is_suspect": int(src_sus),
                    "dst_is_suspect": int(dst_sus),
                    "is_tp_any_endpoint": int(is_tp_any),
                    "is_tp_after_attack_start": int(is_tp_after),
                    "is_fp_strict": int(not is_tp_any),
                    "is_fp_relaxed": int((not is_tp_any) and (not src_sus) and (not dst_sus)),
                }
            )
            writer.writerow(out)


class StreamingRawCsvWriter:
    """Append alert rows during inference, then replay for post-hoc evaluation."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.count = 0
        self._handle: Any | None = None
        self._writer: csv.DictWriter | None = None
        self._fieldnames: list[str] | None = None

    def write(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self._fieldnames = list(row.keys())
            self._handle = open(self.path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._handle, fieldnames=self._fieldnames, extrasaction="ignore")
            self._writer.writeheader()
        assert self._writer is not None
        assert self._fieldnames is not None
        self._writer.writerow({key: row.get(key, "") for key in self._fieldnames})
        self.count += 1

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None


def _load_streamed_rows(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path) or os.path.getsize(path) <= 0:
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _alert_sink_count(rows: list[dict[str, Any]], writer: StreamingRawCsvWriter | None) -> int:
    return int(writer.count if writer is not None else len(rows))


def _emit_alert_row(rows: list[dict[str, Any]], writer: StreamingRawCsvWriter | None, row: dict[str, Any]) -> None:
    if writer is not None:
        writer.write(row)
    else:
        rows.append(row)


def _attach_event_labels(rows: list[dict[str, Any]], event_labels_by_pos: dict[int, int]) -> list[dict[str, Any]]:
    for row in rows:
        event_pos = int(row.get("event_pos", -1))
        label = int(event_labels_by_pos.get(event_pos, -1))
        row["event_label"] = int(label)
        row["is_correct_event_alert"] = int(label in {1, 2})
    return rows


def _residual_state_profile(prefix: str, state: dict[str, Any] | None) -> dict[str, int]:
    if state is None:
        return {}
    return {
        f"{prefix}_states": int(len(state.get("states", {}))),
        f"{prefix}_last_seen": int(len(state.get("last_seen", {}))),
        f"{prefix}_last_edge_seen": int(len(state.get("last_edge_seen", {}))),
    }


def _online_state_profile(
    attack_state: OnlineAttackEvidenceState | None = None,
    confirm_state: OnlineCascadeConfirmer | None = None,
    residual_state: dict[str, Any] | None = None,
    multi_view_state: dict[str, dict[str, Any]] | None = None,
    node_pattern_counts: dict[tuple[int, str], int] | None = None,
    global_pattern_node_counts: Counter[str] | None = None,
    alert_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if attack_state is not None:
        out.update(
            {
                "attack_edge_keys": int(len(attack_state.edge_counts)),
                "attack_recent_peer_nodes": int(len(attack_state.recent_peers)),
                "attack_recent_chain_nodes": int(len(attack_state.recent_chains)),
                "attack_recent_chain_count_nodes": int(len(attack_state.recent_chain_counts)),
                "attack_edge_key_dropped": int(attack_state.edge_key_dropped),
                "attack_edge_novelty_suppressed_overflow": int(attack_state.edge_novelty_suppressed_overflow),
            }
        )
    if confirm_state is not None:
        out["pending_confirmation_episodes"] = int(len(confirm_state.pending_episodes))
    out.update(_residual_state_profile("residual", residual_state))
    if isinstance(multi_view_state, dict):
        for policy, policy_state in multi_view_state.items():
            out.update(_residual_state_profile(f"multi_view_{policy}", policy_state))
    if node_pattern_counts is not None:
        out["predictable_activity_pattern_keys"] = int(len(node_pattern_counts))
    if global_pattern_node_counts is not None:
        out["predictable_activity_global_patterns"] = int(len(global_pattern_node_counts))
    if alert_counts:
        out.update({f"alert_count_{key}": int(value) for key, value in alert_counts.items()})
    return out


def _memory_profile_sample(
    phase: str,
    processed: int = 0,
    phase_start: float | None = None,
    **state_profile: Any,
) -> dict[str, Any]:
    sample: dict[str, Any] = {"phase": phase, **memory_snapshot()}
    if int(processed) > 0:
        sample["processed"] = int(processed)
    if phase_start is not None:
        elapsed = float(time.perf_counter() - phase_start)
        sample["elapsed_seconds"] = elapsed
        if int(processed) > 0:
            sample["logs_per_second"] = float(int(processed) / max(elapsed, 1e-12))
    sample.update(state_profile)
    return sample


def _insert_top_event(rows: list[dict[str, Any]], row: dict[str, Any], limit: int) -> None:
    if limit <= 0:
        return
    rows.append(row)
    if len(rows) > limit * 2:
        rows.sort(key=lambda item: (-float(item["score"]), int(item["event_pos"])))
        del rows[limit:]


def _insert_top_risk(rows: list[dict[str, Any]], row: dict[str, Any], limit: int) -> None:
    if limit <= 0:
        return
    rows.append(row)
    if len(rows) > limit * 2:
        rows.sort(key=lambda item: (-float(item["online_risk"]), int(item["event_pos"])))
        del rows[limit:]


def _trim_top_risk(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    rows.sort(key=lambda item: (-float(item["online_risk"]), int(item["event_pos"])))
    del rows[limit:]
    return rows


def _label_test_outputs(
    conn: Any,
    year_month: str,
    test_days: list[int],
    indexid2summary: dict[int, tuple[str, str]],
    hash2type: dict[str, str],
    hash2uuid_index: dict[str, tuple[str, int]],
    abnormal_nodes: set[int],
    event_filter: bool,
    fetch_size: int,
    max_test_events: int,
    max_node_id: int,
    event_alerts_by_pos: dict[int, list[dict[str, Any]]],
    node_top_event: np.ndarray,
    node_top_label: np.ndarray,
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    first_positive_event: np.ndarray,
    first_positive_ts: np.ndarray,
    event_label_positions: set[int] | None = None,
    event_labels_by_pos: dict[int, int] | None = None,
) -> int:
    labeled = 0
    for event_pos, row in enumerate(
        _stream_events(
            conn,
            year_month,
            test_days,
            indexid2summary,
            hash2type,
            hash2uuid_index,
            abnormal_nodes,
            event_filter,
            int(fetch_size),
            int(max_test_events),
        )
    ):
        label = int(row["label"])
        if event_label_positions is not None and event_labels_by_pos is not None and int(event_pos) in event_label_positions:
            event_labels_by_pos[int(event_pos)] = int(label)
        for event_alert in event_alerts_by_pos.get(int(event_pos), []):
            event_alert["event_label"] = int(label)
            event_alert["is_correct_event_alert"] = int(label in {1, 2})
        for node in [int(row["src_idx"]), int(row["dst_idx"])]:
            if node < 0 or node >= int(max_node_id):
                continue
            if label == 2:
                positive_nodes[node] = True
                if int(first_positive_event[node]) < 0:
                    first_positive_event[node] = np.int64(event_pos)
                    first_positive_ts[node] = np.int64(row["timestamp_ns"])
            elif label == 1:
                if node in abnormal_nodes:
                    positive_nodes[node] = True
                    if int(first_positive_event[node]) < 0:
                        first_positive_event[node] = np.int64(event_pos)
                        first_positive_ts[node] = np.int64(row["timestamp_ns"])
                else:
                    suspect_nodes[node] = True
            if int(node_top_event[node]) == int(event_pos):
                node_top_label[node] = np.int8(label)
        labeled += 1
    return int(labeled)


def _run_synthetic_smoke(args: argparse.Namespace) -> None:
    """Write minimal causal-semantics outputs without touching PostgreSQL."""
    out_tag = str(args.out_tag or "SYNTHETIC_SMOKE")
    result_dir = os.path.join(str(args.result_root), out_tag)
    os.makedirs(result_dir, exist_ok=True)
    max_events = int(args.max_test_events or 12)
    max_events = max(max_events, 1)
    events: list[dict[str, Any]] = []
    node_scores: dict[int, float] = {}
    for event_id in range(max_events):
        src = event_id % 4
        dst = 100 + (event_id % 5)
        event_score = float(1.0 + (event_id % 6))
        label = 2 if event_id in {3, 7} else (1 if event_id in {5, 9} else 0)
        if event_score >= 5.0:
            events.append(
                {
                    "event_pos": event_id,
                    "event_id": event_id,
                    "timestamp": float(event_id),
                    "src_node": src,
                    "dst_node": dst,
                    "event_score": event_score,
                    "online_risk": event_score,
                    "threshold": 5.0,
                    "label": label,
                    "top_components": "synthetic_event_score",
                }
            )
        node_scores[src] = max(node_scores.get(src, 0.0), event_score)
        node_scores[dst] = max(node_scores.get(dst, 0.0), event_score * 0.8)

    event_alerts_csv = os.path.join(result_dir, "online_event_alerts.csv")
    with open(event_alerts_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "event_pos",
                "event_id",
                "timestamp",
                "src_node",
                "dst_node",
                "event_score",
                "online_risk",
                "threshold",
                "label",
                "top_components",
            ],
        )
        writer.writeheader()
        writer.writerows(events)

    node_pool_csv = os.path.join(result_dir, "final_node_pool_alerts.csv")
    with open(node_pool_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "node", "score", "basis"])
        writer.writeheader()
        for rank, (node, score) in enumerate(
            sorted(node_scores.items(), key=lambda item: item[1], reverse=True),
            start=1,
        ):
            writer.writerow(
                {
                    "rank": rank,
                    "node": node,
                    "score": float(score),
                    "basis": "synthetic_event_pool",
                }
            )

    ranked_csv = os.path.join(result_dir, "causal_semantics_ranked.csv")
    with open(ranked_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "node", "score"])
        writer.writeheader()
        for rank, (node, score) in enumerate(
            sorted(node_scores.items(), key=lambda item: item[1], reverse=True),
            start=1,
        ):
            writer.writerow({"rank": rank, "node": node, "score": float(score)})

    correct = sum(1 for row in events if int(row["label"]) in {1, 2})
    false = sum(1 for row in events if int(row["label"]) == 0)
    summary = {
        "dataset": str(args.dataset),
        "method": "causal_semantics_synthetic_smoke",
        "out_tag": out_tag,
        "synthetic_smoke": True,
        "counts": {
            "test_processed": max_events,
            "event_alerts": len(events),
            "node_pool_rows": len(node_scores),
        },
        "primary_online_metrics": {
            "immediate_candidate_alerts": {
                "event_alerts": {
                    "num_alerts": len(events),
                    "correct_event_alerts": correct,
                    "false_event_alerts": false,
                    "correct_alert_ratio": float(correct / max(len(events), 1)),
                    "correct_rule": "synthetic label in {1, 2}; smoke only",
                }
            },
            "note": "Synthetic smoke validates output schema only; it is not a model result.",
        },
        "leakage_check": {
            "synthetic_smoke_only": True,
            "test_labels_used_before_scoring": False,
            "test_rankings_used_before_scoring": False,
            "full_test_statistics_used_before_scoring": False,
            "gt_usage": "none; synthetic labels are attached only inside generated smoke rows",
        },
        "outputs": {
            "online_event_alerts_csv": event_alerts_csv,
            "final_node_pool_alerts_csv": node_pool_csv,
            "ranked_csv": ranked_csv,
        },
    }
    out_json = os.path.join(result_dir, "eval_causal_semantics.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    if bool(getattr(args, "print_summary", False)):
        print(json.dumps({"out_json": out_json, "summary": summary}, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    _configure_ablation(args)
    if bool(getattr(args, "synthetic_smoke", False)):
        _run_synthetic_smoke(args)
        return
    t0 = time.perf_counter()
    out_tag = str(args.out_tag).strip() or f"{args.dataset}_CAUSAL_SEMANTICS_PHASE2"
    result_dir = os.path.abspath(os.path.join(str(args.result_root), out_tag))
    os.makedirs(result_dir, exist_ok=True)

    cfg = _cfg_for_dataset(args.dataset)
    conn_cur, conn = init_database_connection(cfg)
    memory_samples: list[dict[str, Any]] = []
    try:
        phase_start = time.perf_counter()
        log("[CausalSemantics] loading node tables")
        netflow_nodes, process_nodes, file_nodes = fetch_node_tables(conn_cur)
        indexid2summary, hash2type, hash2uuid_index, uuid2index = _build_node_maps_and_summaries(
            netflow_nodes,
            process_nodes,
            file_nodes,
        )
        del netflow_nodes
        del process_nodes
        del file_nodes
        if bool(args.memory_trim_enabled):
            _trim_process_memory()
        max_node_id = max(int(v[1]) for v in hash2uuid_index.values() if v[1] is not None) + 1
        year_month = cfg.dataset.year_month
        train_days = parse_split_days(get_dataset_splits(cfg, "train"))
        val_days = parse_split_days(get_dataset_splits(cfg, "val"))
        test_days = parse_split_days(get_dataset_splits(cfg, "test"))
        event_filter = use_event_type_filter(cfg)
        counts_meta = {
            "train_query_events": _query_count(conn_cur, year_month, train_days, event_filter),
            "val_query_events": _query_count(conn_cur, year_month, val_days, event_filter),
            "test_query_events": _query_count(conn_cur, year_month, test_days, event_filter),
        }
        metadata_seconds = float(time.perf_counter() - phase_start)
        memory_samples.append({"phase": "after_metadata", **memory_snapshot()})
        log(f"[CausalSemantics] split query counts: {counts_meta}")

        model = CausalSemanticModel(
            vocab_size=int(args.vocab_size),
            min_token_count=int(args.min_token_count),
            max_role_keys=int(args.max_role_keys),
            max_pair_keys=int(args.max_pair_keys),
            smoothing=float(args.smoothing),
            role_min_count=int(args.role_min_count),
            activity_log_base=float(args.role_activity_log_base),
            score_clip=float(args.score_clip),
            semantic_string_policy=str(args.semantic_string_policy),
        )

        phase_start = time.perf_counter()
        log("[CausalSemantics] fitting train-only tokenizer vocabulary")
        vocab_events = 0
        for row in _stream_events(conn, year_month, train_days, indexid2summary, hash2type, hash2uuid_index, set(), event_filter, int(args.fetch_size), int(args.max_train_events)):
            model.observe_vocab(row, int(args.max_tokens_per_node))
            vocab_events += 1
            if vocab_events % 1_000_000 == 0:
                log(f"[CausalSemantics] vocab train processed={vocab_events}")
        model.freeze_vocab()
        log(f"[CausalSemantics] frozen vocab size={len(model.vocab)} from events={vocab_events}")

        train_events = 0
        for row in _stream_events(conn, year_month, train_days, indexid2summary, hash2type, hash2uuid_index, set(), event_filter, int(args.fetch_size), int(args.max_train_events)):
            model.observe_train(row, int(args.max_tokens_per_node))
            train_events += 1
            if train_events % 1_000_000 == 0:
                log(f"[CausalSemantics] train stats processed={train_events}")
        model.freeze_roles()
        train_seconds = float(time.perf_counter() - phase_start)
        memory_samples.append({"phase": "after_train", **memory_snapshot()})
        log(f"[CausalSemantics] train stats processed={train_events} roles={len(model.role_vocab)}")

        residual_model: LowRankStreamModel | None = None
        residual_train_summary: dict[str, Any] = {}
        residual_component_stats: list[dict[str, float]] = []
        residual_ref_processed = 0
        residual_fit_seconds = 0.0
        if bool(args.residual_fusion_enabled):
            phase_residual = time.perf_counter()
            log("[CausalSemantics] fitting train-only low-rank residual fusion head")
            residual_model = _new_residual_model(args)
            residual_train_summary = residual_model.fit_rows(
                (
                    _residual_row_view(row, args)
                    for row in _stream_events(
                        conn,
                        year_month,
                        train_days,
                        indexid2summary,
                        hash2type,
                        hash2uuid_index,
                        set(),
                        event_filter,
                        int(args.fetch_size),
                        int(args.max_train_events),
                    )
                )
            )
            residual_fit_seconds = float(time.perf_counter() - phase_residual)
            memory_samples.append({"phase": "after_residual_fit", **memory_snapshot()})
            if bool(args.memory_trim_enabled):
                _trim_process_memory()
                memory_samples.append({"phase": "after_residual_fit_trim", **memory_snapshot()})
            log(
                "[CausalSemantics] residual fusion head fitted: "
                f"events={residual_train_summary.get('processed_events')} "
                f"model_mb={residual_model.model_size_bytes / (1024.0 * 1024.0):.3f}"
            )

        multi_view_model: MultiViewBehaviorModel | None = None
        multi_view_summary: dict[str, Any] = {}
        if bool(args.multi_view_behavior_enabled):
            multi_view_model = MultiViewBehaviorModel(args)
            multi_view_summary = multi_view_model.fit(
                conn,
                year_month,
                train_days,
                indexid2summary,
                hash2type,
                hash2uuid_index,
                event_filter,
                int(args.fetch_size),
                int(args.max_train_events),
            )
            memory_samples.append({"phase": "after_multi_view_behavior_fit", **memory_snapshot()})
            if bool(args.memory_trim_enabled):
                _trim_process_memory()
                memory_samples.append({"phase": "after_multi_view_behavior_fit_trim", **memory_snapshot()})
            log(
                "[CausalSemantics] multi-view behavior fitted: "
                f"policies={','.join(multi_view_model.policies)} "
                f"model_mb={multi_view_model.model_size_bytes / (1024.0 * 1024.0):.3f}"
            )

        phase_start = time.perf_counter()
        log("[CausalSemantics] calibrating validation field rarity")
        val_scores: list[float] = []
        residual_ref_components: list[np.ndarray] = []
        residual_ref_state = _new_residual_state(residual_model) if residual_model is not None else None
        val_processed = 0
        for row in _stream_events(conn, year_month, val_days, indexid2summary, hash2type, hash2uuid_index, set(), event_filter, int(args.fetch_size), int(args.max_ref_events)):
            score, _detail = model.score_row(row, int(args.max_tokens_per_node))
            val_scores.append(float(score))
            if residual_model is not None and residual_ref_state is not None:
                residual_row = _residual_row_view(row, args)
                values, _component_scores, _residual_score = _residual_score_pre_update(
                    residual_model,
                    residual_ref_state,
                    residual_row,
                    None,
                    float(args.residual_fusion_clip),
                )
                residual_ref_components.append(values)
                residual_ref_processed += 1
            val_processed += 1
            if val_processed % 1_000_000 == 0:
                log(f"[CausalSemantics] validation processed={val_processed}")
        model.calibrate_validation(val_scores)
        if residual_ref_components:
            residual_ref_matrix = np.stack(residual_ref_components, axis=0).astype(np.float32, copy=False)
            residual_component_stats = [
                robust_stats(residual_ref_matrix[:, idx])
                for idx in range(residual_ref_matrix.shape[1])
            ]
        if multi_view_model is not None:
            log("[CausalSemantics] calibrating train-only multi-view behavior model on validation split")
            multi_view_summary = multi_view_model.calibrate(
                conn,
                year_month,
                val_days,
                indexid2summary,
                hash2type,
                hash2uuid_index,
                event_filter,
                int(args.fetch_size),
                int(args.max_ref_events),
            )
        validation_seconds = float(time.perf_counter() - phase_start)
        memory_samples.append({"phase": "after_validation", **memory_snapshot()})
        if bool(args.memory_trim_enabled):
            _trim_process_memory()
            memory_samples.append({"phase": "after_validation_trim", **memory_snapshot()})

        phase_start = time.perf_counter()
        log("[CausalSemantics] calibrating validation online alert thresholds")
        online_calibration_limit = _online_calibration_limit(args)
        effective_validation_quantiles = dict(model.validation_quantiles)
        if residual_model is not None and residual_component_stats:
            log("[CausalSemantics] calibrating fused residual+causal validation score distribution")
            residual_score_state = _new_residual_state(residual_model)
            residual_score_multi_view_state = multi_view_model.new_state() if multi_view_model is not None else None
            effective_scores: list[float] = []
            for row in _stream_events(conn, year_month, val_days, indexid2summary, hash2type, hash2uuid_index, set(), event_filter, int(args.fetch_size), int(online_calibration_limit)):
                causal_score, causal_detail = model.score_row(row, int(args.max_tokens_per_node))
                effective_score, _effective_detail = _apply_event_behavior_score(
                    args,
                    float(causal_score),
                    dict(causal_detail),
                    row,
                    residual_model,
                    residual_score_state,
                    residual_component_stats,
                    multi_view_model,
                    residual_score_multi_view_state,
                )
                effective_scores.append(float(effective_score))
            effective_validation_quantiles = _score_quantiles(effective_scores)
        tail_q95 = float(effective_validation_quantiles.get("q95", model.validation_quantiles.get("q95", 0.0)))
        learned_evidence_composer: LearnedEvidenceComposer | None = None
        learned_evidence_fit_summary: dict[str, Any] = {}
        learned_evidence_fit_seconds = 0.0
        if bool(args.learned_evidence_composer_enabled):
            phase_learned = time.perf_counter()
            log("[CausalSemantics] fitting learned evidence composer on train stream")
            learned_evidence_composer = LearnedEvidenceComposer(args)
            train_limit = int(args.learned_evidence_composer_max_train_events)
            if int(args.max_train_events) > 0:
                train_limit = min(train_limit, int(args.max_train_events))
            composer_train_state = OnlineAttackEvidenceState(max_node_id, args, tail_q95)
            composer_train_residual_state = _new_residual_state(residual_model) if residual_model is not None else None
            composer_train_multi_view_state = multi_view_model.new_state() if multi_view_model is not None else None
            composer_train_processed = 0
            for row in _stream_events(conn, year_month, train_days, indexid2summary, hash2type, hash2uuid_index, set(), event_filter, int(args.fetch_size), int(train_limit)):
                causal_score, detail = model.score_row(row, int(args.max_tokens_per_node))
                score, detail = _apply_event_behavior_score(
                    args,
                    float(causal_score),
                    dict(detail),
                    row,
                    residual_model,
                    composer_train_residual_state,
                    residual_component_stats,
                    multi_view_model,
                    composer_train_multi_view_state,
                )
                node_records, episode_record = composer_train_state.score_event(row, float(score), detail)
                for record in node_records:
                    _attach_multi_view_fields(record, detail)
                _attach_multi_view_fields(episode_record, detail)
                for record in node_records:
                    learned_evidence_composer.partial_fit(_evidence_vector_from_record(record))
                learned_evidence_composer.partial_fit(_evidence_vector_from_record(episode_record))
                composer_train_processed += 1
                if composer_train_processed % 1_000_000 == 0:
                    log(f"[CausalSemantics] learned evidence composer train processed={composer_train_processed}")
            learned_evidence_fit_summary = learned_evidence_composer.finalize()
            learned_evidence_fit_seconds = float(time.perf_counter() - phase_learned)
            memory_samples.append(
                {
                    "phase": "after_learned_evidence_fit",
                    **memory_snapshot(),
                }
            )
            log(
                "[CausalSemantics] learned evidence composer fitted: "
                f"mode={learned_evidence_fit_summary.get('mode', '')} "
                f"fit_samples={learned_evidence_fit_summary.get('fit_samples', 0)} "
                f"model_bytes={learned_evidence_fit_summary.get('model_size_bytes', 0)}"
            )
        val_online_state = OnlineAttackEvidenceState(max_node_id, args, tail_q95)
        val_residual_online_state = _new_residual_state(residual_model) if residual_model is not None else None
        val_multi_view_online_state = multi_view_model.new_state() if multi_view_model is not None else None
        val_online_node_scores: list[float] = []
        val_online_episode_scores: list[float] = []
        val_online_event_scores: list[float] = []
        val_confirmed_node_scores: list[float] = []
        val_confirmed_event_scores: list[float] = []
        val_online_processed = 0
        for row in _stream_events(conn, year_month, val_days, indexid2summary, hash2type, hash2uuid_index, set(), event_filter, int(args.fetch_size), int(online_calibration_limit)):
            causal_score, detail = model.score_row(row, int(args.max_tokens_per_node))
            score, detail = _apply_event_behavior_score(
                args,
                float(causal_score),
                dict(detail),
                row,
                residual_model,
                val_residual_online_state,
                residual_component_stats,
                multi_view_model,
                val_multi_view_online_state,
            )
            node_records, episode_record = val_online_state.score_event(row, float(score), detail)
            for record in node_records:
                _attach_multi_view_fields(record, detail)
            _attach_multi_view_fields(episode_record, detail)
            node_records, episode_record = _apply_learned_evidence_composer(learned_evidence_composer, node_records, episode_record)
            val_online_node_scores.extend(float(record["online_risk"]) for record in node_records)
            val_online_episode_scores.append(float(episode_record["online_risk"]))
            val_online_event_scores.append(float(episode_record["online_risk"]))
            val_online_processed += 1
            if val_online_processed % 1_000_000 == 0:
                log(f"[CausalSemantics] validation online processed={val_online_processed}")
        online_min_tail = int(args.online_threshold_min_tail_samples)
        expected_budget = int(args.online_expected_alert_budget)
        expected_horizon = int(args.online_expected_alert_horizon_events)
        online_node_threshold = _risk_threshold(
            val_online_node_scores,
            float(args.online_node_alert_quantile),
            float(args.online_node_alert_threshold),
            online_min_tail,
            expected_budget,
            expected_horizon,
        )
        online_episode_threshold = _risk_threshold(
            val_online_episode_scores,
            float(args.online_episode_alert_quantile),
            float(args.online_episode_alert_threshold),
            online_min_tail,
            expected_budget,
            expected_horizon,
        )
        online_event_threshold = _risk_threshold(
            val_online_event_scores,
            float(args.online_event_alert_quantile),
            float(args.online_event_alert_threshold),
            online_min_tail,
            expected_budget,
            expected_horizon,
        )
        confirm_calibration_node_seed_threshold = _risk_threshold(
            val_online_node_scores,
            min(float(args.online_node_alert_quantile), 0.99),
            0.0,
        )
        confirm_calibration_event_seed_threshold = _risk_threshold(
            val_online_event_scores,
            min(float(args.online_event_alert_quantile), 0.99),
            0.0,
        )
        val_confirm_processed = 0
        val_confirm_state: OnlineCascadeConfirmer | None = None
        val_confirm_online_state: OnlineAttackEvidenceState | None = None
        val_confirm_residual_state: dict[str, Any] | None = None
        val_confirm_multi_view_state: dict[str, dict[str, Any]] | None = None
        if bool(args.online_confirmation_enabled):
            val_confirm_state = OnlineCascadeConfirmer(args, max_node_id, float("inf"), float("inf"))
            val_confirm_online_state = OnlineAttackEvidenceState(max_node_id, args, tail_q95)
            val_confirm_residual_state = _new_residual_state(residual_model) if residual_model is not None else None
            val_confirm_multi_view_state = multi_view_model.new_state() if multi_view_model is not None else None
            for row in _stream_events(conn, year_month, val_days, indexid2summary, hash2type, hash2uuid_index, set(), event_filter, int(args.fetch_size), int(online_calibration_limit)):
                causal_score, detail = model.score_row(row, int(args.max_tokens_per_node))
                score, detail = _apply_event_behavior_score(
                    args,
                    float(causal_score),
                    dict(detail),
                    row,
                    residual_model,
                    val_confirm_residual_state,
                    residual_component_stats,
                    multi_view_model,
                    val_confirm_multi_view_state,
                )
                node_records, episode_record = val_confirm_online_state.score_event(row, float(score), detail)
                for record in node_records:
                    _attach_multi_view_fields(record, detail)
                _attach_multi_view_fields(episode_record, detail)
                node_records, episode_record = _apply_learned_evidence_composer(learned_evidence_composer, node_records, episode_record)
                timestamp_ns = int(row["timestamp_ns"])
                node_candidate_records = [
                    record
                    for record in node_records
                    if float(record["online_risk"]) >= float(confirm_calibration_node_seed_threshold["threshold"])
                ]
                event_candidate_crossed = float(episode_record["online_risk"]) >= float(confirm_calibration_event_seed_threshold["threshold"])
                _confirmed_events, _confirmed_nodes, samples = val_confirm_state.observe_causal_episode(
                    episode_record,
                    node_records,
                    val_confirm_processed,
                    timestamp_ns,
                    detail,
                    event_candidate_crossed=event_candidate_crossed,
                    node_candidate_records=node_candidate_records,
                )
                for sample in samples:
                    if (
                        int(sample["event_count"]) >= 2
                        and int(sample["evidence_kinds"]) >= int(args.online_confirm_min_evidence_kinds)
                    ):
                        if int(sample.get("node_candidate_count", 0)) > 0:
                            val_confirmed_node_scores.append(float(sample["confirm_score"]))
                        if int(sample.get("event_candidate_count", 0)) > 0:
                            val_confirmed_event_scores.append(float(sample["confirm_score"]))
                val_confirm_processed += 1
            confirmed_node_threshold = _risk_threshold(val_confirmed_node_scores, float(args.online_confirmed_node_quantile), float(args.online_confirmed_node_threshold))
            confirmed_event_threshold = _risk_threshold(val_confirmed_event_scores, float(args.online_confirmed_event_quantile), float(args.online_confirmed_event_threshold))
        else:
            confirmed_node_threshold = _disabled_threshold("confirmed_node")
            confirmed_event_threshold = _disabled_threshold("confirmed_event")
        online_calibration_seconds = float(time.perf_counter() - phase_start)
        memory_samples.append(
            _memory_profile_sample(
                "after_online_calibration",
                processed=val_online_processed,
                phase_start=phase_start,
                **_online_state_profile(
                    attack_state=val_online_state,
                    confirm_state=val_confirm_state,
                    residual_state=val_residual_online_state,
                    multi_view_state=val_multi_view_online_state,
                ),
            )
        )
        log(
            "[CausalSemantics] online thresholds: "
            f"node={online_node_threshold['threshold']} episode={online_episode_threshold['threshold']} "
            f"event={online_event_threshold['threshold']} "
            f"confirmed_node={confirmed_node_threshold['threshold']} confirmed_event={confirmed_event_threshold['threshold']} "
            f"validation_events={val_online_processed}"
        )
        del val_online_state
        del val_residual_online_state
        del val_multi_view_online_state
        del val_online_node_scores
        del val_online_episode_scores
        del val_online_event_scores
        del val_confirmed_node_scores
        del val_confirmed_event_scores
        del val_confirm_state
        del val_confirm_online_state
        del val_confirm_residual_state
        del val_confirm_multi_view_state
        if "residual_score_state" in locals():
            del residual_score_state
        if "residual_score_multi_view_state" in locals():
            del residual_score_multi_view_state
        if "composer_train_multi_view_state" in locals():
            del composer_train_multi_view_state
        if "effective_scores" in locals():
            del effective_scores
        if "residual_ref_state" in locals():
            del residual_ref_state
        if "residual_ref_components" in locals():
            del residual_ref_components
        if "residual_ref_matrix" in locals():
            del residual_ref_matrix
        model.validation_scores = []
        gc.collect()
        if bool(args.memory_trim_enabled):
            _trim_process_memory()
        memory_samples.append(_memory_profile_sample("after_validation_cleanup"))

        phase_start = time.perf_counter()
        log("[CausalSemantics] scoring test stream")
        node_score = np.zeros((max_node_id,), dtype=np.float32)
        node_score_max = np.zeros((max_node_id,), dtype=np.float32)
        node_tail_excess = np.zeros((max_node_id,), dtype=np.float32)
        node_tail_count = np.zeros((max_node_id,), dtype=np.int32)
        node_benign_like_count = np.zeros((max_node_id,), dtype=np.int32)
        node_repeated_pattern_max = np.zeros((max_node_id,), dtype=np.int32)
        node_global_pattern_nodes_max = np.zeros((max_node_id,), dtype=np.int32)
        node_predictable_event_count = np.zeros((max_node_id,), dtype=np.int32)
        node_residual_fusion_max = np.zeros((max_node_id,), dtype=np.float32)
        node_count = np.zeros((max_node_id,), dtype=np.int32)
        node_top_event = np.full((max_node_id,), -1, dtype=np.int64)
        node_top_label = np.full((max_node_id,), -1, dtype=np.int8)
        node_top_field = [""] * max_node_id
        node_top_detail: list[dict[str, Any] | None] = [None] * max_node_id
        all_nodes = np.zeros((max_node_id,), dtype=bool)
        positive_nodes = np.zeros((max_node_id,), dtype=bool)
        suspect_nodes = np.zeros((max_node_id,), dtype=bool)
        top_events: list[dict[str, Any]] = []
        node_pattern_counts: dict[tuple[int, str], int] = {}
        global_pattern_node_counts: Counter[str] = Counter()
        pattern_key_dropped = 0
        test_processed = 0
        tail_q95 = float(effective_validation_quantiles.get("q95", model.validation_quantiles.get("q95", 0.0)))
        tail_q99 = float(effective_validation_quantiles.get("q99", tail_q95))
        attack_state = OnlineAttackEvidenceState(max_node_id, args, tail_q95)
        test_residual_state = _new_residual_state(residual_model) if residual_model is not None else None
        test_multi_view_state = multi_view_model.new_state() if multi_view_model is not None else None
        stream_alert_csv = bool(args.stream_alert_csv)
        raw_alert_dir = os.path.join(result_dir, "_stream_alert_raw")
        raw_online_node_alerts_csv = os.path.join(raw_alert_dir, "online_node_alerts.raw.csv")
        raw_online_episode_alerts_csv = os.path.join(raw_alert_dir, "online_episode_alerts.raw.csv")
        raw_online_event_alerts_csv = os.path.join(raw_alert_dir, "online_event_alerts.raw.csv")
        raw_online_confirmed_node_alerts_csv = os.path.join(raw_alert_dir, "online_confirmed_node_alerts.raw.csv")
        raw_online_confirmed_event_alerts_csv = os.path.join(raw_alert_dir, "online_confirmed_event_alerts.raw.csv")
        if stream_alert_csv:
            os.makedirs(raw_alert_dir, exist_ok=True)
            for raw_path in [
                raw_online_node_alerts_csv,
                raw_online_episode_alerts_csv,
                raw_online_event_alerts_csv,
                raw_online_confirmed_node_alerts_csv,
                raw_online_confirmed_event_alerts_csv,
            ]:
                if os.path.exists(raw_path):
                    os.remove(raw_path)
        raw_online_node_writer = StreamingRawCsvWriter(raw_online_node_alerts_csv) if stream_alert_csv else None
        raw_online_episode_writer = StreamingRawCsvWriter(raw_online_episode_alerts_csv) if stream_alert_csv else None
        raw_online_event_writer = StreamingRawCsvWriter(raw_online_event_alerts_csv) if stream_alert_csv else None
        raw_online_confirmed_node_writer = StreamingRawCsvWriter(raw_online_confirmed_node_alerts_csv) if stream_alert_csv else None
        raw_online_confirmed_event_writer = StreamingRawCsvWriter(raw_online_confirmed_event_alerts_csv) if stream_alert_csv else None
        online_node_alerts: list[dict[str, Any]] = []
        online_episode_alerts: list[dict[str, Any]] = []
        online_event_alerts: list[dict[str, Any]] = []
        online_confirmed_node_alerts: list[dict[str, Any]] = []
        online_confirmed_event_alerts: list[dict[str, Any]] = []
        online_event_score_trace: list[dict[str, Any]] = []
        online_node_score_trace: list[dict[str, Any]] = []
        online_episode_score_trace: list[dict[str, Any]] = []
        event_alerts_by_pos: dict[int, list[dict[str, Any]]] = {}
        event_label_positions: set[int] = set()
        event_labels_by_pos: dict[int, int] = {}
        semantic_audit_alert_tokens_by_pos: dict[int, list[str]] | None = (
            {} if bool(args.semantic_audit_enabled) else None
        )
        online_node_policy = StatefulNodeAlertPolicy(
            max_node_id,
            float(args.online_node_update_gain_ratio),
            "online_node_stateful_update",
            str(args.online_node_alert_policy),
        )
        online_episode_alerted: set[str] = set()
        score_trace_enabled = bool(args.online_score_trace_enabled)
        score_trace_limit = int(args.online_score_trace_top_n)
        first_positive_event = np.full((max_node_id,), -1, dtype=np.int64)
        first_positive_ts = np.full((max_node_id,), -1, dtype=np.int64)
        node_alert_threshold = float(online_node_threshold["threshold"])
        episode_alert_threshold = float(online_episode_threshold["threshold"])
        event_alert_threshold = float(online_event_threshold["threshold"])
        confirmed_node_alert_threshold = float(confirmed_node_threshold["threshold"])
        confirmed_event_alert_threshold = float(confirmed_event_threshold["threshold"])
        confirm_state = (
            OnlineCascadeConfirmer(
                args,
                max_node_id,
                confirmed_node_alert_threshold,
                confirmed_event_alert_threshold,
            )
            if bool(args.online_confirmation_enabled)
            else None
        )
        predictable_activity_state_enabled = _predictable_activity_state_enabled(args)
        for row in _stream_events(conn, year_month, test_days, indexid2summary, hash2type, hash2uuid_index, set(), event_filter, int(args.fetch_size), int(args.max_test_events)):
            causal_score, detail = model.score_row(row, int(args.max_tokens_per_node))
            score, detail = _apply_event_behavior_score(
                args,
                float(causal_score),
                dict(detail),
                row,
                residual_model,
                test_residual_state,
                residual_component_stats,
                multi_view_model,
                test_multi_view_state,
            )
            timestamp_ns = int(row["timestamp_ns"])
            src_idx = int(row["src_idx"])
            dst_idx = int(row["dst_idx"])
            action_family = str(detail.get("action_family", ""))
            online_node_records, online_episode_record = attack_state.score_event(row, float(score), detail)
            for record in online_node_records:
                _attach_multi_view_fields(record, detail)
            _attach_multi_view_fields(online_episode_record, detail)
            online_node_records, online_episode_record = _apply_learned_evidence_composer(learned_evidence_composer, online_node_records, online_episode_record)
            online_node_by_id = {int(record["node_id"]): record for record in online_node_records}
            if score_trace_enabled and score_trace_limit > 0:
                trace_event_row = dict(online_episode_record)
                trace_event_row.update(
                    {
                        "event_pos": int(test_processed),
                        "timestamp_ns": int(timestamp_ns),
                        "threshold": float(event_alert_threshold),
                        "event_label": -1,
                        "max_field": str(detail.get("max_field", "")),
                        "src_role": str(detail.get("src_role", "")),
                        "dst_role": str(detail.get("dst_role", "")),
                        "trace_source": "score_trace_top_risk",
                    }
                )
                _insert_top_risk(online_event_score_trace, trace_event_row, score_trace_limit)
                trace_episode_row = dict(online_episode_record)
                trace_episode_row.update(
                    {
                        "event_pos": int(test_processed),
                        "timestamp_ns": int(timestamp_ns),
                        "threshold": float(episode_alert_threshold),
                        "max_field": str(detail.get("max_field", "")),
                        "src_role": str(detail.get("src_role", "")),
                        "dst_role": str(detail.get("dst_role", "")),
                        "trace_source": "score_trace_top_risk",
                    }
                )
                _insert_top_risk(online_episode_score_trace, trace_episode_row, score_trace_limit)
                for trace_node_record in online_node_records:
                    trace_node_row = dict(trace_node_record)
                    trace_node_row.update(
                        {
                            "event_pos": int(test_processed),
                            "timestamp_ns": int(timestamp_ns),
                            "threshold": float(node_alert_threshold),
                            "src_idx": int(src_idx),
                            "dst_idx": int(dst_idx),
                            "action_family": action_family,
                            "max_field": str(detail.get("max_field", "")),
                            "src_role": str(detail.get("src_role", "")),
                            "dst_role": str(detail.get("dst_role", "")),
                            "trace_source": "score_trace_top_risk",
                        }
                    )
                    _insert_top_risk(online_node_score_trace, trace_node_row, score_trace_limit)
            event_candidate_crossed = float(online_episode_record["online_risk"]) >= event_alert_threshold
            node_candidate_records = [
                record
                for record in online_node_records
                if float(record["online_risk"]) >= node_alert_threshold
            ]
            if (
                event_candidate_crossed
                and _alert_sink_count(online_event_alerts, raw_online_event_writer) < int(args.online_max_event_alerts)
            ):
                event_alert_row = dict(online_episode_record)
                event_alert_row.update(
                    {
                        "event_pos": int(test_processed),
                        "timestamp_ns": int(timestamp_ns),
                        "threshold": float(event_alert_threshold),
                        "event_label": -1,
                        "max_field": str(detail.get("max_field", "")),
                        "src_role": str(detail.get("src_role", "")),
                        "dst_role": str(detail.get("dst_role", "")),
                        "explanation_json": _merge_explanation_json(
                            online_episode_record,
                            {
                                "threshold_source": online_event_threshold["source"],
                                "max_field": str(detail.get("max_field", "")),
                                "src_role": str(detail.get("src_role", "")),
                                "dst_role": str(detail.get("dst_role", "")),
                                "action_family": action_family,
                                "semantic_string_policy": str(args.semantic_string_policy),
                                "causal_semantic_score": float(online_episode_record.get("event_causal_semantic_score", 0.0)),
                                "residual_fusion_score": float(online_episode_record.get("event_residual_fusion_score", 0.0)),
                                "event_edge_novelty": float(online_episode_record.get("event_edge_novelty", 0.0)),
                                "event_temporal_burst": float(online_episode_record.get("event_temporal_burst", 0.0)),
                                "event_episode_support": float(online_episode_record.get("event_episode_support", 0.0)),
                                "event_chain_diversity": float(online_episode_record.get("event_chain_diversity", 0.0)),
                                **_multi_view_explanation_fields(online_episode_record),
                                "correctness_rule": "event_label in {1, 2}; evaluated after alert emission",
                            },
                        ),
                    }
                )
                _emit_alert_row(online_event_alerts, raw_online_event_writer, event_alert_row)
                if semantic_audit_alert_tokens_by_pos is not None:
                    if _semantic_policy(str(args.semantic_string_policy)) == "grouped":
                        semantic_audit_alert_tokens_by_pos[int(test_processed)] = _grouped_semantic_tokens(
                            row,
                            int(args.max_tokens_per_node),
                        )
                    else:
                        semantic_audit_alert_tokens_by_pos[int(test_processed)] = _event_tokens(
                            _row_fields(row, int(args.max_tokens_per_node), str(args.semantic_string_policy)),
                            str(args.semantic_string_policy),
                        )
                if stream_alert_csv:
                    event_label_positions.add(int(test_processed))
                else:
                    event_alerts_by_pos.setdefault(int(test_processed), []).append(event_alert_row)
            if confirm_state is not None:
                confirmed_event_rows, confirmed_node_rows, _confirmed_samples = confirm_state.observe_causal_episode(
                    online_episode_record,
                    online_node_records,
                    test_processed,
                    timestamp_ns,
                    detail,
                    event_candidate_crossed=event_candidate_crossed,
                    node_candidate_records=node_candidate_records,
                )
            else:
                confirmed_event_rows, confirmed_node_rows = [], []
            for confirmed_event_row in confirmed_event_rows:
                if _alert_sink_count(online_confirmed_event_alerts, raw_online_confirmed_event_writer) >= int(args.online_max_confirmed_event_alerts):
                    break
                _emit_alert_row(online_confirmed_event_alerts, raw_online_confirmed_event_writer, confirmed_event_row)
                if stream_alert_csv:
                    event_label_positions.add(int(confirmed_event_row["event_pos"]))
                else:
                    event_alerts_by_pos.setdefault(int(confirmed_event_row["event_pos"]), []).append(confirmed_event_row)
            for confirmed_node_row in confirmed_node_rows:
                node = int(confirmed_node_row.get("node_id", -1))
                if node < 0 or node >= max_node_id:
                    continue
                if _alert_sink_count(online_confirmed_node_alerts, raw_online_confirmed_node_writer) < int(args.online_max_confirmed_node_alerts):
                    _emit_alert_row(online_confirmed_node_alerts, raw_online_confirmed_node_writer, confirmed_node_row)
            if (
                float(online_episode_record["online_risk"]) >= episode_alert_threshold
                and str(online_episode_record["episode_id"]) not in online_episode_alerted
                and _alert_sink_count(online_episode_alerts, raw_online_episode_writer) < int(args.online_max_episode_alerts)
            ):
                online_episode_alerted.add(str(online_episode_record["episode_id"]))
                episode_alert_row = dict(online_episode_record)
                episode_alert_row.update(
                    {
                        "event_pos": int(test_processed),
                        "timestamp_ns": int(timestamp_ns),
                        "threshold": float(episode_alert_threshold),
                        "max_field": str(detail.get("max_field", "")),
                        "src_role": str(detail.get("src_role", "")),
                        "dst_role": str(detail.get("dst_role", "")),
                        "explanation_json": _merge_explanation_json(
                            online_episode_record,
                            {
                                "threshold_source": online_episode_threshold["source"],
                                "max_field": str(detail.get("max_field", "")),
                                "src_role": str(detail.get("src_role", "")),
                                "dst_role": str(detail.get("dst_role", "")),
                                "action_family": action_family,
                                "semantic_string_policy": str(args.semantic_string_policy),
                                "causal_semantic_score": float(online_episode_record.get("event_causal_semantic_score", 0.0)),
                                "residual_fusion_score": float(online_episode_record.get("event_residual_fusion_score", 0.0)),
                                "event_edge_novelty": float(online_episode_record.get("event_edge_novelty", 0.0)),
                                "event_temporal_burst": float(online_episode_record.get("event_temporal_burst", 0.0)),
                                "event_episode_support": float(online_episode_record.get("event_episode_support", 0.0)),
                                "event_chain_diversity": float(online_episode_record.get("event_chain_diversity", 0.0)),
                                **_multi_view_explanation_fields(online_episode_record),
                            },
                        ),
                    }
                )
                _emit_alert_row(online_episode_alerts, raw_online_episode_writer, episode_alert_row)
            nodes = [src_idx, dst_idx]
            for node_pos, node in enumerate(nodes):
                if node < 0 or node >= max_node_id:
                    continue
                online_record = online_node_by_id.get(int(node))
                if online_record is None:
                    continue
                current_risk = float(online_record["online_risk"])
                should_emit, alert_meta = online_node_policy.maybe_emit(
                    node,
                    current_risk,
                    node_alert_threshold,
                    int(test_processed),
                    int(timestamp_ns),
                )
                if not should_emit:
                    continue
                if _alert_sink_count(online_node_alerts, raw_online_node_writer) >= int(args.online_max_node_alerts):
                    continue
                node_alert_row = dict(online_record)
                node_alert_row.update(
                    {
                        "event_pos": int(test_processed),
                        "timestamp_ns": int(timestamp_ns),
                        "threshold": float(node_alert_threshold),
                        "src_idx": int(src_idx),
                        "dst_idx": int(dst_idx),
                        "action_family": action_family,
                        "max_field": str(detail.get("max_field", "")),
                        "src_role": str(detail.get("src_role", "")),
                        "dst_role": str(detail.get("dst_role", "")),
                        "explanation_json": _merge_explanation_json(
                            online_record,
                            {
                                "threshold_source": online_node_threshold["source"],
                                "max_field": str(detail.get("max_field", "")),
                                "src_role": str(detail.get("src_role", "")),
                                "dst_role": str(detail.get("dst_role", "")),
                                "action_family": action_family,
                                "semantic_string_policy": str(args.semantic_string_policy),
                                "causal_semantic_score": float(online_record.get("event_causal_semantic_score", 0.0)),
                                "residual_fusion_score": float(online_record.get("event_residual_fusion_score", 0.0)),
                                "event_edge_novelty": float(online_record.get("event_edge_novelty", 0.0)),
                                "event_temporal_burst": float(online_record.get("event_temporal_burst", 0.0)),
                                "event_episode_support": float(online_record.get("event_episode_support", 0.0)),
                                "event_chain_diversity": float(online_record.get("event_chain_diversity", 0.0)),
                                **_multi_view_explanation_fields(online_record),
                                **alert_meta,
                            },
                        ),
                        **alert_meta,
                    }
                )
                _emit_alert_row(online_node_alerts, raw_online_node_writer, node_alert_row)
                if predictable_activity_state_enabled:
                    pattern = _predictable_activity_pattern(detail, is_src=(node_pos == 0))
                    global_pattern = _predictable_activity_global_pattern(detail)
                    pattern_key = (int(node), pattern)
                    seen_before = int(node_pattern_counts.get(pattern_key, 0))
                    global_nodes_before = int(global_pattern_node_counts.get(global_pattern, 0))
                    if global_nodes_before > int(node_global_pattern_nodes_max[node]):
                        node_global_pattern_nodes_max[node] = np.int32(global_nodes_before)
                    if (
                        seen_before >= int(args.predictable_activity_min_seen)
                        and global_nodes_before >= int(args.predictable_activity_min_global_nodes)
                    ):
                        node_predictable_event_count[node] += 1
                    seen_after = seen_before + 1
                    if seen_after > int(node_repeated_pattern_max[node]):
                        node_repeated_pattern_max[node] = np.int32(seen_after)
                    if pattern_key in node_pattern_counts or len(node_pattern_counts) < int(args.predictable_activity_max_pattern_keys):
                        node_pattern_counts[pattern_key] = seen_after
                        if seen_before == 0:
                            global_pattern_node_counts[global_pattern] += 1
                    else:
                        pattern_key_dropped += 1
                all_nodes[node] = True
                node_count[node] += 1
                excess = max(float(score) - tail_q95, 0.0)
                node_tail_excess[node] += np.float32(min(excess, float(args.score_clip)))
                if float(score) >= tail_q99:
                    node_tail_count[node] += 1
                else:
                    node_benign_like_count[node] += 1
                if float(score) > float(node_score[node]):
                    node_score[node] = np.float32(score)
                    node_score_max[node] = np.float32(score)
                    node_residual_fusion_max[node] = np.float32(float(detail.get("residual_fusion_score", 0.0)))
                    node_top_event[node] = int(test_processed)
                    node_top_field[node] = str(detail.get("max_field", ""))
                    node_top_detail[node] = dict(detail)
            _insert_top_event(
                top_events,
                {
                    "event_pos": int(test_processed),
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "score": float(score),
                    "label": -1,
                    "src_idx": int(row["src_idx"]),
                    "dst_idx": int(row["dst_idx"]),
                    "max_field": str(detail.get("max_field", "")),
                    "max_field_nll": float(detail.get("max_field_nll", 0.0)),
                    "token_unseen_ratio": float(detail.get("token_unseen_ratio", 0.0)),
                    "causal_semantic_score": float(detail.get("causal_semantic_score", score)),
                    "residual_fusion_score": float(detail.get("residual_fusion_score", 0.0)),
                    "src_role": str(detail.get("src_role", "")),
                    "dst_role": str(detail.get("dst_role", "")),
                    "action_family": str(detail.get("action_family", "")),
                },
                int(args.top_events),
            )
            test_processed += 1
            if test_processed % 1_000_000 == 0:
                log(f"[CausalSemantics] test processed={test_processed}")
                memory_samples.append(
                    _memory_profile_sample(
                        "test_scoring",
                        processed=test_processed,
                        phase_start=phase_start,
                        **_online_state_profile(
                            attack_state=attack_state,
                            confirm_state=confirm_state,
                            residual_state=test_residual_state,
                            multi_view_state=test_multi_view_state,
                            node_pattern_counts=node_pattern_counts,
                            global_pattern_node_counts=global_pattern_node_counts,
                            alert_counts={
                                "event": _alert_sink_count(online_event_alerts, raw_online_event_writer),
                                "episode": _alert_sink_count(online_episode_alerts, raw_online_episode_writer),
                                "node": _alert_sink_count(online_node_alerts, raw_online_node_writer),
                                "confirmed_event": _alert_sink_count(online_confirmed_event_alerts, raw_online_confirmed_event_writer),
                                "confirmed_node": _alert_sink_count(online_confirmed_node_alerts, raw_online_confirmed_node_writer),
                            },
                        ),
                    )
                )
        for writer in [
            raw_online_node_writer,
            raw_online_episode_writer,
            raw_online_event_writer,
            raw_online_confirmed_node_writer,
            raw_online_confirmed_event_writer,
        ]:
            if writer is not None:
                writer.close()
        test_seconds = float(time.perf_counter() - phase_start)
        memory_samples.append(
            _memory_profile_sample(
                "after_test",
                processed=test_processed,
                phase_start=phase_start,
                **_online_state_profile(
                    attack_state=attack_state,
                    confirm_state=confirm_state,
                    residual_state=test_residual_state,
                    multi_view_state=test_multi_view_state,
                    node_pattern_counts=node_pattern_counts,
                    global_pattern_node_counts=global_pattern_node_counts,
                    alert_counts={
                        "event": _alert_sink_count(online_event_alerts, raw_online_event_writer),
                        "episode": _alert_sink_count(online_episode_alerts, raw_online_episode_writer),
                        "node": _alert_sink_count(online_node_alerts, raw_online_node_writer),
                        "confirmed_event": _alert_sink_count(online_confirmed_event_alerts, raw_online_confirmed_event_writer),
                        "confirmed_node": _alert_sink_count(online_confirmed_node_alerts, raw_online_confirmed_node_writer),
                    },
                ),
            )
        )

        phase_start = time.perf_counter()
        log("[CausalSemantics] labeling test outputs for evaluation")
        if score_trace_enabled and score_trace_limit > 0:
            _trim_top_risk(online_event_score_trace, score_trace_limit)
            _trim_top_risk(online_node_score_trace, score_trace_limit)
            _trim_top_risk(online_episode_score_trace, score_trace_limit)
            for trace_event_row in online_event_score_trace:
                event_alerts_by_pos.setdefault(int(trace_event_row["event_pos"]), []).append(trace_event_row)
        abnormal_nodes = set(int(x) for x in load_ground_truth_indices(cfg, uuid2index, result_dir))
        labeled_test_events = _label_test_outputs(
            conn,
            year_month,
            test_days,
            indexid2summary,
            hash2type,
            hash2uuid_index,
            abnormal_nodes,
            event_filter,
            int(args.fetch_size),
            int(args.max_test_events),
            max_node_id,
            event_alerts_by_pos,
            node_top_event,
            node_top_label,
            positive_nodes,
            suspect_nodes,
            first_positive_event,
            first_positive_ts,
            event_label_positions,
            event_labels_by_pos,
        )
        evaluation_label_seconds = float(time.perf_counter() - phase_start)
        memory_samples.append({"phase": "after_evaluation_labeling", "test_processed": int(labeled_test_events), **memory_snapshot()})

        node_attack_edge_sum = attack_state.node_attack_edge_sum
        node_attack_burst_sum = attack_state.node_attack_burst_sum
        node_attack_episode_sum = attack_state.node_attack_episode_sum
        node_attack_chain_sum = attack_state.node_attack_chain_sum
        node_episode_recent_peer_max = attack_state.node_episode_recent_peer_max
        node_chain_diversity_max = attack_state.node_chain_diversity_max
        node_attack_episode_event_count = attack_state.node_attack_episode_event_count

        topk_values = [int(x.strip()) for x in str(args.topk_values).split(",") if x.strip()]
        node_score_tail_excess = (
            node_score_max
            + 0.75 * np.log1p(np.maximum(node_tail_count.astype(np.float32), 0.0))
            + 0.10 * np.minimum(node_tail_excess, float(args.score_clip))
        ).astype(np.float32, copy=False)
        node_score_activity_adjusted = (
            node_score_tail_excess
            - 0.25 * np.log1p(np.maximum(node_benign_like_count.astype(np.float32), 0.0))
        ).astype(np.float32, copy=False)
        predictable_repeat_strength = np.maximum(
            np.log1p(np.maximum(node_repeated_pattern_max.astype(np.float32), 0.0))
            - math.log1p(max(int(args.predictable_activity_min_seen), 1)),
            0.0,
        )
        predictable_global_strength = np.maximum(
            np.log1p(np.maximum(node_global_pattern_nodes_max.astype(np.float32), 0.0))
            - math.log1p(max(int(args.predictable_activity_min_global_nodes), 1)),
            0.0,
        )
        node_predictable_activity_credit = (
            float(args.predictable_activity_weight)
            * np.power(predictable_repeat_strength, float(args.predictable_activity_power))
            * (1.0 + predictable_global_strength)
        ).astype(np.float32, copy=False)
        node_predictable_activity_credit = np.where(
            predictable_global_strength > 0.0,
            node_predictable_activity_credit,
            0.0,
        ).astype(np.float32, copy=False)
        node_predictable_activity_credit = np.minimum(
            node_predictable_activity_credit,
            np.float32(max(float(args.predictable_activity_cap), 0.0)),
        ).astype(np.float32, copy=False)
        node_score_predictable_activity = (
            node_score_activity_adjusted
            - node_predictable_activity_credit
        ).astype(np.float32, copy=False)
        node_attack_edge_component = np.minimum(
            np.log1p(np.maximum(node_attack_edge_sum, 0.0)),
            np.float32(max(float(args.attack_evidence_component_cap), 0.0)),
        ).astype(np.float32, copy=False)
        node_attack_burst_component = np.minimum(
            np.log1p(np.maximum(node_attack_burst_sum, 0.0)),
            np.float32(max(float(args.attack_evidence_component_cap), 0.0)),
        ).astype(np.float32, copy=False)
        node_attack_episode_component = np.minimum(
            np.log1p(np.maximum(node_attack_episode_sum, 0.0)),
            np.float32(max(float(args.attack_evidence_component_cap), 0.0)),
        ).astype(np.float32, copy=False)
        node_attack_chain_component = np.minimum(
            np.log1p(np.maximum(node_attack_chain_sum, 0.0)),
            np.float32(max(float(args.attack_evidence_component_cap), 0.0)),
        ).astype(np.float32, copy=False)
        node_score_attack_evidence = (
            node_score_activity_adjusted
            + float(args.attack_evidence_edge_weight) * node_attack_edge_component
            + float(args.attack_evidence_burst_weight) * node_attack_burst_component
            + float(args.attack_evidence_episode_weight) * node_attack_episode_component
            + float(args.attack_evidence_chain_weight) * node_attack_chain_component
        ).astype(np.float32, copy=False)
        if str(args.node_score_mode) == "max":
            final_node_score = node_score_max
        elif str(args.node_score_mode) == "tail_excess":
            final_node_score = node_score_tail_excess
        elif str(args.node_score_mode) == "activity_adjusted":
            final_node_score = node_score_activity_adjusted
        elif str(args.node_score_mode) == "predictable_activity":
            final_node_score = node_score_predictable_activity
        else:
            final_node_score = node_score_attack_evidence
        node_score = final_node_score.astype(np.float32, copy=False)
        sweep = _node_sweep(node_score, all_nodes, positive_nodes, suspect_nodes, topk_values)
        sweep_by_mode = {
            "max": _node_sweep(node_score_max, all_nodes, positive_nodes, suspect_nodes, topk_values),
            "tail_excess": _node_sweep(node_score_tail_excess, all_nodes, positive_nodes, suspect_nodes, topk_values),
            "activity_adjusted": _node_sweep(node_score_activity_adjusted, all_nodes, positive_nodes, suspect_nodes, topk_values),
            "predictable_activity": _node_sweep(node_score_predictable_activity, all_nodes, positive_nodes, suspect_nodes, topk_values),
            "attack_evidence": _node_sweep(node_score_attack_evidence, all_nodes, positive_nodes, suspect_nodes, topk_values),
        }
        target = target_for_dataset(args.dataset)
        best = best_sweep_row(sweep, target)
        best_fp = best_under_fp_target(sweep, target)
        best_by_mode = {
            mode: best_under_fp_target(rows, target)
            for mode, rows in sweep_by_mode.items()
        }
        ranked = rank_scores(node_score)
        ranked_csv = os.path.join(result_dir, "causal_semantics_ranked.csv")
        top_events_csv = os.path.join(result_dir, "causal_semantics_top_events.csv")
        online_node_alerts_csv = os.path.join(result_dir, "online_node_alerts.csv")
        online_episode_alerts_csv = os.path.join(result_dir, "online_episode_alerts.csv")
        online_event_alerts_csv = os.path.join(result_dir, "online_event_alerts.csv")
        online_confirmed_node_alerts_csv = os.path.join(result_dir, "online_confirmed_node_alerts.csv")
        online_confirmed_event_alerts_csv = os.path.join(result_dir, "online_confirmed_event_alerts.csv")
        online_event_score_trace_csv = os.path.join(result_dir, "online_event_score_trace.csv")
        online_node_score_trace_csv = os.path.join(result_dir, "online_node_score_trace.csv")
        online_episode_score_trace_csv = os.path.join(result_dir, "online_episode_score_trace.csv")
        semantic_audit_json = os.path.join(result_dir, "semantic_audit.json")
        _write_ranked_csv(
            ranked_csv,
            ranked,
            node_score,
            node_score_max,
            node_score_tail_excess,
            node_score_activity_adjusted,
            node_score_predictable_activity,
            node_score_attack_evidence,
            node_residual_fusion_max,
            node_predictable_activity_credit,
            node_repeated_pattern_max,
            node_global_pattern_nodes_max,
            node_predictable_event_count,
            node_attack_edge_component,
            node_attack_burst_component,
            node_attack_episode_component,
            node_attack_chain_component,
            node_episode_recent_peer_max,
            node_chain_diversity_max,
            node_attack_episode_event_count,
            all_nodes,
            positive_nodes,
            suspect_nodes,
            node_count,
            node_top_event,
            node_top_label,
            node_top_field,
            node_top_detail,
        )
        _write_top_events_csv(top_events_csv, top_events[: int(args.top_events)])
        if stream_alert_csv:
            online_node_alerts = _load_streamed_rows(raw_online_node_alerts_csv)
            online_episode_alerts = _load_streamed_rows(raw_online_episode_alerts_csv)
            online_event_alerts = _attach_event_labels(
                _load_streamed_rows(raw_online_event_alerts_csv),
                event_labels_by_pos,
            )
            online_confirmed_node_alerts = _load_streamed_rows(raw_online_confirmed_node_alerts_csv)
            online_confirmed_event_alerts = _attach_event_labels(
                _load_streamed_rows(raw_online_confirmed_event_alerts_csv),
                event_labels_by_pos,
            )
        _write_online_node_alerts_csv(
            online_node_alerts_csv,
            online_node_alerts,
            positive_nodes,
            suspect_nodes,
            first_positive_event,
            first_positive_ts,
        )
        _write_online_episode_alerts_csv(
            online_episode_alerts_csv,
            online_episode_alerts,
            positive_nodes,
            suspect_nodes,
            first_positive_event,
        )
        _write_online_event_alerts_csv(
            online_event_alerts_csv,
            online_event_alerts,
        )
        _write_confirmed_node_alerts_csv(
            online_confirmed_node_alerts_csv,
            online_confirmed_node_alerts,
            positive_nodes,
            suspect_nodes,
            first_positive_event,
            first_positive_ts,
        )
        _write_confirmed_event_alerts_csv(
            online_confirmed_event_alerts_csv,
            online_confirmed_event_alerts,
        )
        if score_trace_enabled and score_trace_limit > 0:
            _write_online_event_alerts_csv(
                online_event_score_trace_csv,
                online_event_score_trace,
            )
            _write_online_node_alerts_csv(
                online_node_score_trace_csv,
                online_node_score_trace,
                positive_nodes,
                suspect_nodes,
                first_positive_event,
                first_positive_ts,
            )
            _write_online_episode_alerts_csv(
                online_episode_score_trace_csv,
                online_episode_score_trace,
                positive_nodes,
                suspect_nodes,
                first_positive_event,
            )
        online_node_metrics = _online_node_alert_metrics(
            online_node_alerts,
            all_nodes,
            positive_nodes,
            suspect_nodes,
            first_positive_event,
        )
        online_episode_metrics = _online_episode_alert_metrics(
            online_episode_alerts,
            positive_nodes,
            suspect_nodes,
            first_positive_event,
        )
        online_event_metrics = _online_event_alert_metrics(online_event_alerts)
        online_confirmed_node_metrics = _online_node_alert_metrics(
            online_confirmed_node_alerts,
            all_nodes,
            positive_nodes,
            suspect_nodes,
            first_positive_event,
        )
        online_confirmed_event_metrics = _online_event_alert_metrics(online_confirmed_event_alerts)

        total_seconds = float(time.perf_counter() - t0)
        summary = {
            "dataset": str(args.dataset),
            "method": "causal_semantics_phase3_chain_diversity_online_alerts",
            "purpose": "Diagnostic only. Tests whether train-only role/shape vocabulary plus a unified causal risk with temporal burst shape, edge novelty, multi-hop episode support, and episode chain diversity improves online alerts before implementing CMTE.",
            "out_tag": out_tag,
            "counts": counts_meta,
            "config": {
                "vocab_size": int(args.vocab_size),
                "actual_vocab_size": int(len(model.vocab)),
                "min_token_count": int(args.min_token_count),
                "max_tokens_per_node": int(args.max_tokens_per_node),
                "semantic_string_policy": str(args.semantic_string_policy),
                "semantic_audit_enabled": bool(args.semantic_audit_enabled),
                "semantic_audit_top_k": int(args.semantic_audit_top_k),
                "role_min_count": int(args.role_min_count),
                "roles": int(len(model.role_vocab)),
                "context_keys": int(len(model.context_counts)),
                "pair_keys": int(len(model.pair_counts)),
                "smoothing": float(args.smoothing),
                "score_clip": float(args.score_clip),
                "ablation_mode": str(args.ablation_mode),
                "ablation_effective_mode": str(getattr(args, "ablation_effective_mode", "CUSTOM")),
                "node_score_mode": str(args.node_score_mode),
                "stream_alert_csv": bool(args.stream_alert_csv),
                "predictable_activity_min_seen": int(args.predictable_activity_min_seen),
                "predictable_activity_min_global_nodes": int(args.predictable_activity_min_global_nodes),
                "predictable_activity_weight": float(args.predictable_activity_weight),
                "predictable_activity_power": float(args.predictable_activity_power),
                "predictable_activity_cap": float(args.predictable_activity_cap),
                "predictable_activity_max_pattern_keys": int(args.predictable_activity_max_pattern_keys),
                "predictable_activity_state_mode": str(args.predictable_activity_state_mode),
                "predictable_activity_state_enabled": bool(predictable_activity_state_enabled),
                "predictable_activity_pattern_keys": int(len(node_pattern_counts)),
                "predictable_activity_global_patterns": int(len(global_pattern_node_counts)),
                "predictable_activity_pattern_key_dropped": int(pattern_key_dropped),
                "attack_evidence_edge_enabled": bool(args.attack_evidence_edge_enabled),
                "attack_evidence_burst_enabled": bool(args.attack_evidence_burst_enabled),
                "attack_evidence_episode_enabled": bool(args.attack_evidence_episode_enabled),
                "attack_evidence_chain_enabled": bool(args.attack_evidence_chain_enabled),
                "attack_evidence_edge_weight": float(args.attack_evidence_edge_weight),
                "attack_evidence_burst_weight": float(args.attack_evidence_burst_weight),
                "attack_evidence_episode_weight": float(args.attack_evidence_episode_weight),
                "attack_evidence_chain_weight": float(args.attack_evidence_chain_weight),
                "attack_evidence_component_cap": float(args.attack_evidence_component_cap),
                "attack_evidence_burst_halflife_seconds": float(args.attack_evidence_burst_halflife_seconds),
                "attack_evidence_episode_halflife_seconds": float(args.attack_evidence_episode_halflife_seconds),
                "attack_evidence_episode_window_seconds": float(args.attack_evidence_episode_window_seconds),
                "attack_evidence_min_distinct_peers": int(args.attack_evidence_min_distinct_peers),
                "attack_evidence_recent_peer_cap": int(args.attack_evidence_recent_peer_cap),
                "attack_evidence_propagation_weight": float(args.attack_evidence_propagation_weight),
                "attack_evidence_state_cap": float(args.attack_evidence_state_cap),
                "attack_evidence_max_edge_keys": int(args.attack_evidence_max_edge_keys),
                "attack_evidence_edge_overflow_policy": str(args.attack_evidence_edge_overflow_policy),
                "attack_evidence_edge_keys": int(len(attack_state.edge_counts)),
                "attack_evidence_edge_key_dropped": int(attack_state.edge_key_dropped),
                "attack_evidence_edge_novelty_suppressed_overflow": int(attack_state.edge_novelty_suppressed_overflow),
                "attack_evidence_recent_peer_nodes": int(len(attack_state.recent_peers)),
                "attack_evidence_recent_chain_nodes": int(len(attack_state.recent_chains)),
                "max_online_calibration_events": int(args.max_online_calibration_events),
                "online_calibration_events_used": int(val_online_processed),
                "online_threshold_min_tail_samples": int(args.online_threshold_min_tail_samples),
                "online_node_alert_quantile": float(args.online_node_alert_quantile),
                "online_episode_alert_quantile": float(args.online_episode_alert_quantile),
                "online_event_alert_quantile": float(args.online_event_alert_quantile),
                "online_node_alert_threshold_arg": float(args.online_node_alert_threshold),
                "online_episode_alert_threshold_arg": float(args.online_episode_alert_threshold),
                "online_event_alert_threshold_arg": float(args.online_event_alert_threshold),
                "online_expected_alert_budget": int(args.online_expected_alert_budget),
                "online_expected_alert_horizon_events": int(args.online_expected_alert_horizon_events),
                "online_max_node_alerts": int(args.online_max_node_alerts),
                "online_max_episode_alerts": int(args.online_max_episode_alerts),
                "online_max_event_alerts": int(args.online_max_event_alerts),
                "online_node_alert_policy": str(args.online_node_alert_policy),
                "online_node_update_gain_ratio": float(args.online_node_update_gain_ratio),
                "online_risk_composition": str(args.online_risk_composition),
                "online_confirm_window_seconds": float(args.online_confirm_window_seconds),
                "online_confirm_window_events": int(args.online_confirm_window_events),
                "online_confirm_min_evidence_kinds": int(args.online_confirm_min_evidence_kinds),
                "online_confirmation_enabled": bool(args.online_confirmation_enabled),
                "online_confirmed_node_quantile": float(args.online_confirmed_node_quantile),
                "online_confirmed_event_quantile": float(args.online_confirmed_event_quantile),
                "online_confirmed_node_threshold_arg": float(args.online_confirmed_node_threshold),
                "online_confirmed_event_threshold_arg": float(args.online_confirmed_event_threshold),
                "online_max_confirmed_node_alerts": int(args.online_max_confirmed_node_alerts),
                "online_max_confirmed_event_alerts": int(args.online_max_confirmed_event_alerts),
                "online_confirm_node_candidate_policy": str(args.online_confirm_node_candidate_policy),
                "online_confirm_node_strong_risk_ratio": float(args.online_confirm_node_strong_risk_ratio),
                "online_score_trace_enabled": bool(args.online_score_trace_enabled),
                "online_score_trace_top_n": int(args.online_score_trace_top_n),
                "residual_fusion_enabled": bool(args.residual_fusion_enabled),
                "residual_semantic_weight": float(args.residual_semantic_weight),
                "residual_fusion_weight": float(args.residual_fusion_weight),
                "residual_fusion_clip": float(args.residual_fusion_clip),
                "residual_vector_cache_size": int(args.residual_vector_cache_size),
                "residual_components": list(RESIDUAL_COMPONENTS),
                "residual_component_stats": residual_component_stats,
                "residual_ref_processed": int(residual_ref_processed),
                "residual_train_summary": residual_train_summary,
                "residual_model_size_bytes": int(residual_model.model_size_bytes) if residual_model is not None else 0,
                "residual_model_size_mb": float(residual_model.model_size_bytes / (1024.0 * 1024.0)) if residual_model is not None else 0.0,
                "multi_view_behavior_enabled": bool(args.multi_view_behavior_enabled),
                "multi_view_behavior_policies": str(args.multi_view_behavior_policies),
                "multi_view_behavior_clip": float(args.multi_view_behavior_clip),
                "multi_view_behavior_ridge": float(args.multi_view_behavior_ridge),
                "multi_view_behavior_store_all_components": bool(args.multi_view_behavior_store_all_components),
                "memory_trim_enabled": bool(args.memory_trim_enabled),
                "multi_view_behavior_summary": multi_view_summary,
                "multi_view_behavior_model_size_bytes": int(multi_view_model.model_size_bytes) if multi_view_model is not None else 0,
                "multi_view_behavior_model_size_mb": float(multi_view_model.model_size_bytes / (1024.0 * 1024.0)) if multi_view_model is not None else 0.0,
                "learned_evidence_composer_enabled": bool(args.learned_evidence_composer_enabled),
                "learned_evidence_composer_mode": str(args.learned_evidence_composer_mode),
                "learned_evidence_fit_seconds": float(learned_evidence_fit_seconds),
                "learned_evidence_fit_summary": learned_evidence_fit_summary,
                "learned_evidence_model_size_bytes": int(learned_evidence_composer.model_size_bytes()) if learned_evidence_composer is not None else 0,
                "learned_evidence_model_size_mb": float(learned_evidence_composer.model_size_bytes() / (1024.0 * 1024.0)) if learned_evidence_composer is not None else 0.0,
                "max_train_events": int(args.max_train_events),
                "max_ref_events": int(args.max_ref_events),
                "max_test_events": int(args.max_test_events),
            },
            "leakage_check": {
                "tokenizer_fit_split": "train_only",
                "role_vocab_fit_split": "train_only",
                "field_statistics_fit_split": "train_only",
                "calibration_split": "validation_only",
                "online_thresholds": "chosen from validation online risk distributions unless explicit threshold args are set",
                "residual_fusion": "optional low-rank residual head is fit on train only, calibrated on validation only, and scored causally before each event update",
                "multi_view_behavior": "optional experimental multi-view residual model; raw/compressed/no_identity view heads are fit on train only, fusion is calibrated on validation only, and test events are scored before per-view state updates",
                "semantic_string_policy": "predeclared CLI option; raw/compressed/no_identity/grouped changes train-only role tokens and online scoring tokens without labels or test statistics. grouped keeps one stream model with struct/coarse/raw token groups rather than fitting separate per-view heads",
                "semantic_audit": "optional post-inference report; labels are attached only after alerts are emitted and are not used to choose thresholds or scores",
                "predictable_activity_credit": "test_stream_past_only; each event uses node-pattern counts and cross-node pattern support observed before that event, with no labels or future events",
                "attack_evidence": "test_stream_past_only; temporal burst, edge novelty, episode support, and chain diversity use frozen train/validation scores plus causal online state before or at the current event, with no labels or future events",
                "two_stage_cascade": "candidate alerts are emitted immediately; confirmed alerts use only bounded-delay candidate-window evidence and validation-calibrated confirmation thresholds",
                "confirmed_node_candidate_policy": "predeclared CLI option; it restricts confirmed-node propagation using only current/past candidate status and endpoint evidence, never labels or test rankings",
                "event_alert_evaluation": "event labels are attached only after an online threshold alert is emitted; label in {1, 2} is counted correct for event-level evaluation",
                "test_labels_used_before_scoring": False,
                "test_rankings_used_before_scoring": False,
                "full_test_statistics_used_before_scoring": False,
                "online_score_trace": "optional diagnostic-only top-risk score trace; labels are attached after inference using the same evaluation pass as alert CSVs",
                "gt_usage": "loaded only after the online test stream has completed, for post-stream evaluation labels",
                "learned_evidence_composer": "train-only fit on frozen evidence vectors; validation/test use the frozen composer causally with no labels or future events",
            },
            "validation_score_quantiles": effective_validation_quantiles,
            "causal_semantic_validation_score_quantiles": model.validation_quantiles,
            "online_alert_thresholds": {
                "node": online_node_threshold,
                "episode": online_episode_threshold,
                "event": online_event_threshold,
                "confirmed_node": confirmed_node_threshold,
                "confirmed_event": confirmed_event_threshold,
                "confirmation_calibration_node_seed": confirm_calibration_node_seed_threshold,
                "confirmation_calibration_event_seed": confirm_calibration_event_seed_threshold,
                "validation_online_events": int(val_online_processed),
                "validation_confirmation_events": int(val_confirm_processed),
            },
            "test_processed": int(test_processed),
            "node_population": {
                "observed_nodes": int(np.sum(all_nodes)),
                "positive_nodes": int(np.sum(positive_nodes & all_nodes)),
                "suspect_nodes": int(np.sum(suspect_nodes & all_nodes)),
                "ranked_nodes": int(ranked.shape[0]),
            },
            "sweep": sweep,
            "sweep_by_mode": sweep_by_mode,
            "best": best,
            "best_under_fp_target": best_fp,
            "best_under_fp_target_by_mode": best_by_mode,
            "primary_online_metrics": {
                "immediate_candidate_alerts": {
                    "node_alerts": online_node_metrics,
                    "episode_alerts": online_episode_metrics,
                    "event_alerts": online_event_metrics,
                },
                "bounded_delay_confirmed_alerts": {
                    "node_alerts": online_confirmed_node_metrics,
                    "event_alerts": online_confirmed_event_metrics,
                },
                "node_alerts": online_node_metrics,
                "episode_alerts": online_episode_metrics,
                "event_alerts": online_event_metrics,
                "confirmed_node_alerts": online_confirmed_node_metrics,
                "confirmed_event_alerts": online_confirmed_event_metrics,
                "note": "Primary paper-facing evaluation: immediate candidate alerts and bounded-delay confirmed alerts are emitted online using validation-calibrated thresholds. Forensic top-k sweeps below are auxiliary.",
            },
            "runtime": {
                "metadata_seconds": float(metadata_seconds),
                "train_seconds": float(train_seconds),
                "residual_fit_seconds": float(residual_fit_seconds),
                "validation_seconds": float(validation_seconds),
                "online_calibration_seconds": float(online_calibration_seconds),
                "test_seconds": float(test_seconds),
                "test_logs_per_second": float(test_processed / max(test_seconds, 1e-12)),
                "evaluation_label_seconds": float(evaluation_label_seconds),
                "evaluation_label_logs_per_second": float(labeled_test_events / max(evaluation_label_seconds, 1e-12)),
                "total_seconds": float(total_seconds),
                "fetch_size": int(args.fetch_size),
            },
            "memory": {
                "definition": "RSS process memory sampled at phase boundaries; model state is counters and frozen vocab/role tables.",
                "samples": memory_samples,
                "max_current_rss_mb": float(max((sample.get("current_rss_mb", 0.0) for sample in memory_samples), default=0.0)),
                "peak_rss_mb": float(max((sample.get("peak_rss_mb", 0.0) for sample in memory_samples), default=0.0)),
            },
            "outputs": {
                "online_node_alerts_csv": online_node_alerts_csv,
                "online_episode_alerts_csv": online_episode_alerts_csv,
                "online_event_alerts_csv": online_event_alerts_csv,
                "online_confirmed_node_alerts_csv": online_confirmed_node_alerts_csv,
                "online_confirmed_event_alerts_csv": online_confirmed_event_alerts_csv,
                "raw_online_node_alerts_csv": raw_online_node_alerts_csv if stream_alert_csv else "",
                "raw_online_episode_alerts_csv": raw_online_episode_alerts_csv if stream_alert_csv else "",
                "raw_online_event_alerts_csv": raw_online_event_alerts_csv if stream_alert_csv else "",
                "raw_online_confirmed_node_alerts_csv": raw_online_confirmed_node_alerts_csv if stream_alert_csv else "",
                "raw_online_confirmed_event_alerts_csv": raw_online_confirmed_event_alerts_csv if stream_alert_csv else "",
                "online_event_score_trace_csv": online_event_score_trace_csv if score_trace_enabled and score_trace_limit > 0 else "",
                "online_node_score_trace_csv": online_node_score_trace_csv if score_trace_enabled and score_trace_limit > 0 else "",
                "online_episode_score_trace_csv": online_episode_score_trace_csv if score_trace_enabled and score_trace_limit > 0 else "",
                "semantic_audit_json": semantic_audit_json if bool(args.semantic_audit_enabled) else "",
                "ranked_csv": ranked_csv,
                "top_events_csv": top_events_csv,
            },
        }
        if bool(args.semantic_audit_enabled):
            semantic_audit = _build_semantic_audit(
                model,
                str(args.semantic_string_policy),
                int(args.semantic_audit_top_k),
                semantic_audit_alert_tokens_by_pos,
                event_labels_by_pos,
            )
            with open(semantic_audit_json, "w", encoding="utf-8") as handle:
                json.dump(semantic_audit, handle, indent=2, sort_keys=True)
            summary["semantic_audit"] = {
                "json": semantic_audit_json,
                "event_alert_driver_summary": semantic_audit["event_alert_drivers"],
                "frozen_vocab_category_counts": semantic_audit["frozen_vocab_category_counts"],
            }
        out_json = os.path.join(result_dir, "eval_causal_semantics.json")
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        log(f"[CausalSemantics] wrote {out_json}")
        if args.print_summary:
            print(
                json.dumps(
                    {
                        "out_json": out_json,
                        "primary_online_metrics": summary["primary_online_metrics"],
                        "online_alert_thresholds": summary["online_alert_thresholds"],
                        "auxiliary_forensic_best": best,
                        "auxiliary_forensic_best_under_fp_target": best_fp,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    finally:
        try:
            conn_cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
