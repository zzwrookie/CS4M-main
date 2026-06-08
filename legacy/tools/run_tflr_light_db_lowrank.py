#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.config.provnet_utils import get_indexid2msg, init_database_connection, log
from legacy.tools.evaluate_event_scores import _attack_windows_for_dataset
from scripts.pipeline.outputs.evaluation import (
    best_under_fp_target,
    best_sweep_row,
    empty_event_counts,
    event_metrics,
    label_distribution,
    memory_snapshot,
    node_confusion_from_masks,
    rank_scores,
    target_for_dataset,
    update_event_counts,
)
from scripts.data.get_dataset import (
    build_hash_to_type,
    build_hash_uuid_index_map,
    build_uuid_index_map,
    fetch_node_tables,
    get_dataset_splits,
    load_ground_truth_indices,
    parse_split_days,
    use_event_type_filter,
)
from scripts.pipeline.io.db_stream import _build_index_summaries, _cfg_for_dataset, _query_count, _stream_events
from cs4m.utils.common import robust_stats, stable_hash
from legacy.baselines.chain_profile import BenignChainProfile, ChainProfileConfig
from legacy.baselines.lowrank import LowRankConfig, LowRankStreamModel


COMPONENTS = [
    "pred_mse",
    "pred_cosine",
    "state_cosine",
    "factor_residual",
    "semantic_ae_residual",
    "semantic_rarity",
    "semantic_max_rarity",
    "semantic_top3_rarity",
    "semantic_unseen_ratio",
    "edge_rarity",
    "src_rarity",
    "dst_rarity",
    "rel_dst_rarity",
    "endpoint_self_rarity",
]
EXTRA_COMPONENTS = ["propagation", "backtrack", "compact_chain", "chain_profile"]
PRECOMPUTE_CACHE_SCHEMA_VERSION = 4


def _bool_int_arg(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return 1
    if text in {"0", "false", "no", "n", "off"}:
        return 0
    raise argparse.ArgumentTypeError(f"expected boolean/int value, got {value!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-free low-rank TFLR-Light DB detector.")
    p.add_argument("--dataset", default="THEIA_E3")
    p.add_argument("--out_tag", default="THEIA_E3_TFLR_LOW_RANK")
    p.add_argument("--result_dir", default="")
    p.add_argument("--fetch_size", type=int, default=200000)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=24)
    p.add_argument("--include_bigrams", action="store_true")
    p.add_argument("--vector_cache_size", type=int, default=200000)
    p.add_argument("--semantic_buckets", type=int, default=131072)
    p.add_argument("--identity_buckets", type=int, default=131072)
    p.add_argument("--max_train_samples", type=int, default=300000)
    p.add_argument("--max_train_events", type=int, default=0)
    p.add_argument("--max_ref_events", type=int, default=0)
    p.add_argument("--max_test_events", type=int, default=0)
    p.add_argument("--ridge_lambda", type=float, default=1e-2)
    p.add_argument("--direct_lowrank_enabled", action="store_true")
    p.add_argument("--direct_lowrank_iters", type=int, default=5)
    p.add_argument("--direct_lowrank_l2", type=float, default=1e-2)
    p.add_argument("--weights", default="2.0,1.0,0.0,0.0,0.0,0.5,1.5,1.0,0.5,1.0,0.15,0.15,0.5,1.0")
    p.add_argument("--score_clip", type=float, default=12.0)
    p.add_argument("--event_threshold_quantile", type=float, default=0.999)
    p.add_argument("--prop_threshold_quantile", type=float, default=0.999)
    p.add_argument(
        "--topk_values",
        default="100,200,300,400,433,500,700,900,1000,1200,1500,1800,2000,2200,2500,3000,3500,4000,5000,7000,10000,15000,18000,20000,22000,24000,26000,28000,29000,29500,30000,50000",
    )
    p.add_argument("--prop_decay_sec", type=float, default=3600.0)
    p.add_argument("--prop_boost", type=float, default=0.95)
    p.add_argument("--prop_update_threshold_ratio", type=float, default=1.0)
    p.add_argument("--undirected_propagation", action="store_true")
    p.add_argument("--node_count_weight", type=float, default=0.75)
    p.add_argument("--node_sum_weight", type=float, default=0.15)
    p.add_argument("--node_soft_cap", type=float, default=8.0)
    p.add_argument("--backtrack_hops", type=int, default=0)
    p.add_argument("--backtrack_boost", type=float, default=0.85)
    p.add_argument("--backtrack_decay_sec", type=float, default=3600.0)
    p.add_argument("--backtrack_min_score_ratio", type=float, default=1.0)
    p.add_argument("--global_decay_sec", type=float, default=1800.0)
    p.add_argument("--global_boost", type=float, default=0.0)
    p.add_argument("--compact_chain_enabled", action="store_true")
    p.add_argument("--compact_chain_slots", type=int, default=3)
    p.add_argument("--compact_chain_decay_sec", type=float, default=3600.0)
    p.add_argument("--compact_chain_update_ratio", type=float, default=0.9)
    p.add_argument("--compact_chain_min_diversity", type=int, default=2)
    p.add_argument("--compact_chain_support_cap", type=float, default=8.0)
    p.add_argument("--compact_chain_support_weight", type=float, default=0.35)
    p.add_argument("--compact_chain_diversity_weight", type=float, default=0.45)
    p.add_argument("--compact_chain_neighbor_weight", type=float, default=0.65)
    p.add_argument("--compact_chain_score_weight", type=float, default=1.0)
    p.add_argument("--compact_chain_bonus_cap", type=float, default=3.0)
    p.add_argument("--chain_profile_enabled", action="store_true")
    p.add_argument("--chain_profile_buckets", type=int, default=65536)
    p.add_argument("--chain_profile_score_weight", type=float, default=0.5)
    p.add_argument("--chain_profile_clip", type=float, default=8.0)
    p.add_argument("--chain_profile_require_transition", action="store_true")
    p.add_argument("--chain_profile_score_mode", choices=["z", "conformal", "agreement"], default="z")
    p.add_argument("--chain_profile_calibration_knots", type=int, default=256)
    p.add_argument("--chain_profile_memory_slots", type=int, default=1)
    p.add_argument("--chain_profile_association_weight", type=float, default=0.0)
    p.add_argument("--activity_normalized_chain_enabled", action="store_true")
    p.add_argument("--activity_normalized_chain_start_topk", type=int, default=1000)
    p.add_argument("--activity_normalized_chain_end_topk", type=int, default=3000)
    p.add_argument("--activity_normalized_chain_min_z", type=float, default=1.0)
    p.add_argument("--activity_normalized_chain_weight", type=float, default=0.45)
    p.add_argument("--activity_normalized_chain_boost_cap", type=float, default=1.0)
    p.add_argument("--activity_normalized_chain_calibration_mode", choices=["tail", "z"], default="tail")
    p.add_argument("--compact_malicious_chain_enabled", action="store_true")
    p.add_argument("--compact_malicious_chain_weight", type=float, default=1.20)
    p.add_argument("--causal_episode_support_enabled", action="store_true")
    p.add_argument("--causal_episode_selective_enabled", action="store_true")
    p.add_argument("--causal_episode_min_event_tail", type=float, default=1.0)
    p.add_argument("--causal_episode_min_chain", type=float, default=1.0)
    p.add_argument("--causal_episode_min_assoc", type=float, default=1.0)
    p.add_argument("--causal_episode_min_families", type=int, default=2)
    p.add_argument("--adaptive_memory_enabled", action="store_true")
    p.add_argument("--adaptive_memory_buckets", type=int, default=1024)
    p.add_argument("--adaptive_memory_min_count", type=int, default=3)
    p.add_argument("--adaptive_memory_fit_alpha", type=float, default=0.05)
    p.add_argument("--adaptive_memory_update_alpha", type=float, default=0.02)
    p.add_argument("--adaptive_memory_weight", type=float, default=0.60)
    p.add_argument("--adaptive_memory_clip", type=float, default=8.0)
    p.add_argument("--adaptive_memory_calibration_mode", choices=["tail", "z"], default="tail")
    p.add_argument("--adaptive_memory_calibration_knots", type=int, default=256)
    p.add_argument("--adaptive_memory_update_tail_max", type=float, default=0.35)
    p.add_argument("--adaptive_memory_update_assoc_z_max", type=float, default=0.50)
    p.add_argument("--adaptive_memory_quarantine_tail_min", type=float, default=1.0)
    p.add_argument("--adaptive_memory_quarantine_assoc_z_min", type=float, default=3.0)
    p.add_argument("--node_rerank_enabled", action="store_true")
    p.add_argument("--node_rerank_weight", type=float, default=0.75)
    p.add_argument("--node_rerank_penalty_weight", type=float, default=0.85)
    p.add_argument("--node_rerank_margin", type=float, default=1.0)
    p.add_argument("--node_rerank_clip", type=float, default=8.0)
    p.add_argument("--node_rerank_calibration_mode", choices=["z", "tail"], default="tail")
    p.add_argument("--node_rerank_calibration_knots", type=int, default=256)
    p.add_argument("--node_rerank_high_activity_weight", type=float, default=0.20)
    p.add_argument("--node_rerank_high_activity_margin", type=float, default=0.5)
    p.add_argument("--node_rerank_high_activity_cap", type=float, default=2.0)
    p.add_argument("--node_rerank_joint_activity_pressure", action="store_true")
    p.add_argument("--node_rerank_mode", choices=["additive", "replace"], default="additive")
    p.add_argument("--node_rerank_feature_weights", default="0.60,0.10,0.10,0.80,0.90,0.15,0.30,0.90,0.50,1.00,1.20")
    p.add_argument("--node_postprocess_cache_enabled", action="store_true")
    p.add_argument("--node_postprocess_cache_path", default="")
    p.add_argument("--precompute_cache_enabled", action="store_true")
    p.add_argument("--precompute_cache_path", default="")
    p.add_argument("--precompute_cache_mode", choices=["auto", "read", "write", "off"], default="auto")
    p.add_argument("--precompute_cache_strict", action="store_true")
    p.add_argument("--online_node_alerts_enabled", action="store_true")
    p.add_argument("--online_node_alerts_path", default="")
    p.add_argument("--online_node_alert_threshold", type=float, default=-1.0)
    p.add_argument("--online_node_alert_threshold_quantile", type=float, default=0.999)
    p.add_argument("--online_node_risk_decay_sec", type=float, default=3600.0)
    p.add_argument("--risk_decay", type=float, default=-1.0)
    p.add_argument("--active_node_budget", type=int, default=0)
    p.add_argument("--edge_sketch_size", type=int, default=0)
    p.add_argument("--chain_sketch_size", type=int, default=0)
    p.add_argument("--memory_prototype_count", type=int, default=0)
    p.add_argument("--node_state_ttl_sec", type=float, default=0.0)
    p.add_argument("--node_state_maintenance_interval", type=int, default=65536)
    p.add_argument("--node_state_eviction_batch", type=int, default=4096)
    p.add_argument("--online_node_activity_weight", type=float, default=0.20)
    p.add_argument("--online_node_activity_margin", type=float, default=0.50)
    p.add_argument("--online_node_activity_cap", type=float, default=2.0)
    p.add_argument("--node_rank_mode", choices=["score", "intrinsic_first"], default="score")
    p.add_argument("--node_intrinsic_support_weight", type=float, default=0.80)
    p.add_argument("--node_intrinsic_tie_weight", type=float, default=0.05)
    p.add_argument("--node_intrinsic_tie_cap", type=float, default=1.0)
    p.add_argument("--node_hard_gate_enabled", action="store_true")
    p.add_argument("--node_hard_gate_min_support_z", type=float, default=1.0)
    p.add_argument("--node_hard_gate_min_families", type=int, default=2)
    p.add_argument("--node_hard_gate_prop_z", type=float, default=0.5)
    p.add_argument("--node_hard_gate_penalty_weight", type=float, default=1.0)
    p.add_argument("--node_hard_gate_cap_margin", type=float, default=0.25)
    p.add_argument("--evidence_band_gate_enabled", action="store_true")
    p.add_argument("--evidence_band_calibration_mode", choices=["tail", "z"], default="tail")
    p.add_argument("--evidence_band_candidate_local", type=float, default=1.0)
    p.add_argument("--evidence_band_candidate_activity", type=float, default=2.0)
    p.add_argument("--evidence_band_candidate_relation", type=float, default=1.0)
    p.add_argument("--evidence_band_min_local", type=float, default=0.75)
    p.add_argument("--evidence_band_min_relation", type=float, default=0.75)
    p.add_argument("--evidence_band_min_assoc", type=float, default=0.75)
    p.add_argument("--evidence_band_min_semantic", type=float, default=0.75)
    p.add_argument("--evidence_band_min_role", type=float, default=0.50)
    p.add_argument("--evidence_band_min_adaptive", type=float, default=0.50)
    p.add_argument("--evidence_band_min_families", type=int, default=3)
    p.add_argument("--evidence_band_weight", type=float, default=0.45)
    p.add_argument("--evidence_band_boost_cap", type=float, default=0.85)
    p.add_argument("--evidence_band_penalty_weight", type=float, default=0.55)
    p.add_argument("--evidence_band_penalty_cap", type=float, default=2.5)
    p.add_argument("--evidence_band_activity_weight", type=float, default=0.35)
    p.add_argument("--evidence_band_activity_margin", type=float, default=0.75)
    p.add_argument("--evidence_band_update_min_count", type=float, default=8.0)
    p.add_argument("--evidence_band_update_weight", type=float, default=0.20)
    p.add_argument("--evidence_band_update_cap", type=float, default=2.0)
    p.add_argument("--evidence_band_prop_weight", type=float, default=0.15)
    p.add_argument("--evidence_band_prop_cap", type=float, default=1.0)
    p.add_argument("--mid_rank_gate_enabled", action="store_true")
    p.add_argument("--mid_rank_gate_protect_topk", type=int, default=2000)
    p.add_argument("--mid_rank_gate_end_topk", type=int, default=4000)
    p.add_argument("--mid_rank_gate_min_assoc_z", type=float, default=1.0)
    p.add_argument("--mid_rank_gate_min_relation_z", type=float, default=1.0)
    p.add_argument("--mid_rank_gate_min_adaptive_z", type=float, default=1.0)
    p.add_argument("--mid_rank_gate_require_quarantine", type=_bool_int_arg, default=1)
    p.add_argument("--mid_rank_gate_penalty_weight", type=float, default=0.80)
    p.add_argument("--mid_rank_gate_penalty_cap", type=float, default=4.0)
    p.add_argument("--mid_rank_gate_benign_update_min_count", type=float, default=8.0)
    p.add_argument("--mid_rank_gate_benign_update_weight", type=float, default=0.20)
    p.add_argument("--mid_rank_gate_benign_update_cap", type=float, default=2.0)
    p.add_argument("--local_closure_enabled", action="store_true")
    p.add_argument("--local_closure_protect_topk", type=int, default=1800)
    p.add_argument("--local_closure_end_topk", type=int, default=20000)
    p.add_argument("--local_closure_min_score_z", type=float, default=1.0)
    p.add_argument("--local_closure_min_families", type=int, default=6)
    p.add_argument("--local_closure_min_assoc_z", type=float, default=1.0)
    p.add_argument("--local_closure_min_role_z", type=float, default=1.0)
    p.add_argument("--local_closure_min_adaptive_z", type=float, default=1.0)
    p.add_argument("--local_closure_weight", type=float, default=0.55)
    p.add_argument("--local_closure_boost_cap", type=float, default=1.0)
    p.add_argument("--local_closure_penalty_weight", type=float, default=0.25)
    p.add_argument("--local_closure_penalty_cap", type=float, default=2.0)
    p.add_argument("--local_closure_prop_penalty_weight", type=float, default=0.15)
    p.add_argument("--local_closure_prop_penalty_cap", type=float, default=1.0)
    p.add_argument("--local_closure_update_min_count", type=float, default=8.0)
    p.add_argument("--local_closure_update_weight", type=float, default=0.25)
    p.add_argument("--local_closure_update_cap", type=float, default=2.0)
    p.add_argument("--local_closure_require_quarantine", type=_bool_int_arg, default=1)
    p.add_argument("--local_closure_calibration_mode", choices=["tail", "z"], default="tail")
    p.add_argument("--local_closure_calibration_knots", type=int, default=256)
    p.add_argument("--snc_enabled", action="store_true")
    p.add_argument("--snc_decay_sec", type=float, default=3600.0)
    p.add_argument("--snc_update_min_support", type=float, default=0.75)
    p.add_argument("--snc_update_min_assoc_z", type=float, default=0.75)
    p.add_argument("--snc_calibration_mode", choices=["tail", "z"], default="tail")
    p.add_argument("--snc_calibration_knots", type=int, default=256)
    p.add_argument("--snc_clip", type=float, default=8.0)
    p.add_argument("--snc_weight", type=float, default=0.25)
    p.add_argument("--snc_boost_cap", type=float, default=0.75)
    p.add_argument("--snc_penalty_weight", type=float, default=0.15)
    p.add_argument("--snc_penalty_cap", type=float, default=0.75)
    p.add_argument("--snc_candidate_start_topk", type=int, default=1000)
    p.add_argument("--snc_candidate_end_topk", type=int, default=5000)
    p.add_argument("--snc_min_cluster_z", type=float, default=1.0)
    p.add_argument("--snc_min_object_z", type=float, default=0.5)
    p.add_argument("--snc_min_diversity_z", type=float, default=0.5)
    p.add_argument("--snc_high_count_penalty_min", type=float, default=512.0)
    p.add_argument("--snc_benign_update_penalty_min", type=float, default=8.0)
    p.add_argument("--snc_benign_update_weight", type=float, default=0.20)
    p.add_argument("--profiling_interval_events", type=int, default=0)
    p.add_argument("--profiling_enabled", action="store_true")
    return p.parse_args()


def _upper_z_scalar(value: float, stats: dict[str, float], clip: float) -> float:
    scale = max(float(stats["mad"]) * 1.4826, 1e-6)
    out = max((float(value) - float(stats["median"])) / scale, 0.0)
    if float(clip) > 0:
        out = min(out, float(clip))
    return float(out)


def _quantile_calibration(values: np.ndarray, num_knots: int) -> dict[str, object]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size <= 0:
        return {"num_values": 0, "quantiles": [], "values": []}
    knots = max(int(num_knots), 8)
    q_main = np.linspace(0.50, 0.999, knots, dtype=np.float64)
    q_tail = np.asarray([0.9995, 0.9999], dtype=np.float64) if data.size >= 10000 else np.asarray([], dtype=np.float64)
    quantiles = np.unique(np.concatenate([q_main, q_tail]))
    q_values_raw = np.quantile(data, quantiles).astype(np.float64, copy=False)
    order = np.argsort(q_values_raw, kind="mergesort")
    sorted_values = q_values_raw[order]
    sorted_quantiles = quantiles[order]
    unique_values: list[float] = []
    unique_quantiles: list[float] = []
    # Repeated validation values are common in sparse audit features. Treat an
    # exact plateau value as the first quantile where it appears, otherwise a
    # constant benign feature would look like an extreme tail event.
    for value, quantile in zip(sorted_values.tolist(), sorted_quantiles.tolist()):
        if unique_values and math.isclose(float(value), float(unique_values[-1]), rel_tol=0.0, abs_tol=1e-12):
            continue
        unique_values.append(float(value))
        unique_quantiles.append(float(quantile))
    return {
        "num_values": int(data.size),
        "quantiles": unique_quantiles,
        "values": unique_values,
    }


def _calibration_size_bytes(calibration: dict[str, object]) -> int:
    return int(4 * len(calibration.get("quantiles", [])) + 4 * len(calibration.get("values", [])))


def _stable_config_fingerprint(payload: dict[str, object]) -> str:
    return str(stable_hash(json.dumps(payload, sort_keys=True, default=str), seed=7349))


def _default_precompute_cache_path(result_dir: str, dataset: str, fingerprint: str) -> str:
    safe_dataset = str(dataset).replace("/", "_")
    base_dir = os.path.dirname(os.path.abspath(result_dir))
    return os.path.join(base_dir, "precompute_cache", f"{safe_dataset}_{fingerprint}.pkl")


def _write_pickle_cache(path: str, payload: dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def _read_pickle_cache(path: str) -> dict[str, object]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("cache payload is not a dict")
    return payload


def _tail_surprise_scalar(value: float, calibration: dict[str, object], clip: float) -> float:
    q_raw = calibration.get("quantiles", [])
    v_raw = calibration.get("values", [])
    if not q_raw or not v_raw or not math.isfinite(float(value)):
        return 0.0
    quantiles = np.asarray(q_raw, dtype=np.float64)
    q_values = np.asarray(v_raw, dtype=np.float64)
    q = float(np.interp(float(value), q_values, quantiles, left=0.0, right=float(quantiles[-1])))
    surprise = -math.log10(max(1.0 - q, 1e-9))
    if float(clip) > 0.0:
        surprise = min(surprise, float(clip))
    return float(max(surprise, 0.0))


def _calibration_score_scalar(
    value: float,
    stats: dict[str, float],
    calibration: dict[str, object],
    clip: float,
    mode: str,
) -> float:
    if str(mode) == "tail":
        return _tail_surprise_scalar(value, calibration, clip)
    return _upper_z_scalar(value, stats, clip)


def _calibration_score_array(
    values: np.ndarray,
    stats: dict[str, float],
    calibration: dict[str, object],
    clip: float,
    mode: str,
) -> np.ndarray:
    if str(mode) == "tail":
        return _tail_surprise_array(values, calibration, clip)
    return _upper_z_array(values, stats, clip)


def _fused_score(values: list[float], stats: list[dict[str, float]], weights: list[float], clip: float) -> float:
    total = 0.0
    denom = 0.0
    for value, stat, weight in zip(values, stats, weights):
        if float(weight) <= 0:
            continue
        total += float(weight) * _upper_z_scalar(float(value), stat, clip)
        denom += float(weight)
    return float(total / max(denom, 1e-12))


def _decayed_scalar_state(state: np.ndarray, last_ts: np.ndarray, node: int, ts: int, decay_sec: float) -> float:
    if node < 0 or node >= state.shape[0]:
        return 0.0
    prev = float(state[node])
    if prev <= 0.0:
        return 0.0
    last = int(last_ts[node])
    if last < 0:
        return prev
    dt = max(float(ts - last) / 1e9, 0.0)
    decayed = prev * math.exp(-dt / max(float(decay_sec), 1e-3))
    state[node] = np.float32(decayed)
    last_ts[node] = int(ts)
    return float(decayed)


def _node_score(
    max_score: np.ndarray,
    high_count: np.ndarray,
    soft_sum: np.ndarray,
    count_weight: float,
    sum_weight: float,
    compact_chain_bonus: np.ndarray | None = None,
    compact_chain_weight: float = 0.0,
) -> np.ndarray:
    score = max_score + float(count_weight) * np.log1p(high_count) + float(sum_weight) * np.log1p(soft_sum)
    if compact_chain_bonus is not None and float(compact_chain_weight) > 0.0:
        score = score + float(compact_chain_weight) * compact_chain_bonus
    return score


def _empty_feature_stats() -> dict[str, float]:
    return {"median": 0.0, "mad": 1.0, "q95": 0.0, "q99": 0.0, "q995": 0.0, "q999": 0.0}


def _empty_calibration() -> dict[str, object]:
    return {"num_values": 0, "quantiles": [], "values": []}


def _activity_normalized_chain_density(chain_profile_raw: np.ndarray, node_high_count: np.ndarray) -> np.ndarray:
    if chain_profile_raw.size <= 0:
        return np.zeros((node_high_count.shape[0],), dtype=np.float32)
    chain = np.maximum(np.asarray(chain_profile_raw, dtype=np.float32), 0.0)
    activity = np.maximum(np.asarray(node_high_count, dtype=np.float32), 0.0)
    denom = np.maximum(np.log1p(activity), 1.0)
    return (chain / denom).astype(np.float32, copy=False)


def _new_activity_normalized_chain_arrays(max_node_id: int, enabled: bool) -> dict[str, np.ndarray]:
    size = int(max_node_id) if bool(enabled) else 0
    return {
        "candidate": np.zeros((size,), dtype=bool),
        "applied": np.zeros((size,), dtype=bool),
        "density": np.zeros((size,), dtype=np.float32),
        "score_z": np.zeros((size,), dtype=np.float32),
        "boost": np.zeros((size,), dtype=np.float32),
    }


def _activity_normalized_chain_meta() -> dict[str, object]:
    return {
        "candidate_nodes": 0,
        "applied_nodes": 0,
        "max_boost": 0.0,
        "mean_boost": 0.0,
        "mean_density_candidate": 0.0,
        "mean_z_candidate": 0.0,
    }


def _apply_activity_normalized_chain_score(
    node_score: np.ndarray,
    ranked_before_activity: np.ndarray,
    chain_profile_raw: np.ndarray,
    node_high_count: np.ndarray,
    density_stats: dict[str, float],
    density_calibration: dict[str, object],
    calibration_mode: str,
    clip: float,
    start_topk: int,
    end_topk: int,
    min_z: float,
    weight: float,
    boost_cap: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    arrays = _new_activity_normalized_chain_arrays(int(node_score.shape[0]), True)
    meta = _activity_normalized_chain_meta()
    start = max(int(start_topk), 0)
    end = max(int(end_topk), start)
    end = min(end, int(ranked_before_activity.shape[0]))
    if end <= start or chain_profile_raw.size <= 0:
        return node_score, arrays, meta

    density = _activity_normalized_chain_density(chain_profile_raw, node_high_count)
    arrays["density"][:] = density.astype(np.float32, copy=False)
    if int(density_calibration.get("num_values", 0)) <= 0:
        return node_score, arrays, meta
    z = _calibration_score_array(density, density_stats, density_calibration, float(clip), str(calibration_mode))
    arrays["score_z"][:] = z.astype(np.float32, copy=False)

    candidate_nodes = ranked_before_activity[start:end].astype(np.int64, copy=False)
    candidate_nodes = candidate_nodes[(candidate_nodes >= 0) & (candidate_nodes < node_score.shape[0])]
    if candidate_nodes.size <= 0:
        return node_score, arrays, meta
    arrays["candidate"][candidate_nodes] = True

    candidate_density = density[candidate_nodes]
    candidate_z = z[candidate_nodes]
    support_mask = (candidate_density > 0.0) & (candidate_z >= float(min_z))
    if not np.any(support_mask):
        meta.update(
            {
                "candidate_nodes": int(candidate_nodes.size),
                "mean_density_candidate": float(np.mean(candidate_density)) if candidate_density.size else 0.0,
                "mean_z_candidate": float(np.mean(candidate_z)) if candidate_z.size else 0.0,
            }
        )
        return node_score, arrays, meta

    apply_nodes = candidate_nodes[support_mask]
    raw_boost = float(weight) * np.maximum(candidate_z[support_mask] - float(min_z), 0.0)
    if float(boost_cap) > 0.0:
        raw_boost = np.minimum(raw_boost, float(boost_cap))
    apply_mask = raw_boost > 0.0
    if np.any(apply_mask):
        boosted_nodes = apply_nodes[apply_mask]
        boost = raw_boost[apply_mask].astype(np.float32, copy=False)
        node_score[boosted_nodes] = node_score[boosted_nodes] + boost
        arrays["applied"][boosted_nodes] = True
        arrays["boost"][boosted_nodes] = boost

    active_boost = arrays["boost"][apply_nodes]
    meta.update(
        {
            "candidate_nodes": int(candidate_nodes.size),
            "applied_nodes": int(np.sum(apply_mask)),
            "max_boost": float(np.max(active_boost)) if active_boost.size else 0.0,
            "mean_boost": float(np.mean(active_boost[active_boost > 0.0])) if np.any(active_boost > 0.0) else 0.0,
            "mean_density_candidate": float(np.mean(candidate_density)) if candidate_density.size else 0.0,
            "mean_z_candidate": float(np.mean(candidate_z)) if candidate_z.size else 0.0,
        }
    )
    return node_score, arrays, meta


class OnlineNodeRiskState:
    """Causal node risk state for optional stopping-time node alerts."""

    def __init__(
        self,
        max_node_id: int,
        count_weight: float,
        sum_weight: float,
        compact_chain_weight: float,
        chain_profile_weight: float,
        adaptive_memory_weight: float,
        activity_weight: float,
        activity_margin: float,
        activity_cap: float,
        decay_sec: float,
        active_node_budget: int,
        node_state_ttl_sec: float,
        maintenance_interval: int,
        eviction_batch: int,
    ) -> None:
        self.count_weight = float(count_weight)
        self.sum_weight = float(sum_weight)
        self.compact_chain_weight = float(compact_chain_weight)
        self.chain_profile_weight = float(chain_profile_weight)
        self.adaptive_memory_weight = float(adaptive_memory_weight)
        self.activity_weight = float(activity_weight)
        self.activity_margin = float(activity_margin)
        self.activity_cap = float(activity_cap)
        self.decay_sec = max(float(decay_sec), 1e-3)
        self.active_node_budget = max(int(active_node_budget), 0)
        self.node_state_ttl_sec = max(float(node_state_ttl_sec), 0.0)
        self.maintenance_interval = max(int(maintenance_interval), 1)
        self.eviction_batch = max(int(eviction_batch), 1)
        self.max_node_id = int(max_node_id)
        self.state: dict[int, dict[str, float | int]] = {}
        self.active_count = 0
        self.peak_active_count = 0
        self.evicted_count = 0
        self.update_count = 0
        self.maintenance_runs = 0
        self.last_maintenance_ts = -1
        self.ttl_evicted_count = 0
        self.budget_evicted_count = 0
        self.max_budget_overflow = 0

    def _new_node_state(self) -> dict[str, float | int]:
        return {
            "risk": 0.0,
            "last_ts": -1,
            "intrinsic_residual": 0.0,
            "compact_causal_evidence": 0.0,
            "episode_subgraph_support": 0.0,
            "adaptive_memory_shift": 0.0,
            "activity_only_pressure": 0.0,
            "peak_event_score": 0.0,
            "peak_top_component": -1,
        }

    def _decay(self, node_state: dict[str, float | int], ts: int) -> None:
        last = int(node_state["last_ts"])
        if last >= 0 and float(node_state["risk"]) > 0.0:
            dt = max(float(int(ts) - last) / 1e9, 0.0)
            node_state["risk"] = float(node_state["risk"]) * math.exp(-dt / self.decay_sec)
        node_state["last_ts"] = int(ts)

    def _drop_node(self, node: int) -> None:
        if node in self.state:
            del self.state[node]

    def _maintenance_due(self, ts: int) -> bool:
        if not self.state:
            return False
        if self.update_count % self.maintenance_interval == 0:
            return True
        if self.active_node_budget > 0 and len(self.state) > self.active_node_budget + self.eviction_batch:
            return True
        if self.node_state_ttl_sec > 0.0 and self.last_maintenance_ts >= 0:
            dt = max(float(int(ts) - int(self.last_maintenance_ts)) / 1e9, 0.0)
            return dt >= max(self.node_state_ttl_sec / 4.0, 60.0)
        return self.last_maintenance_ts < 0

    def _enforce_budget(self, ts: int, force: bool = False) -> None:
        self.active_count = int(len(self.state))
        if (not force) and (not self._maintenance_due(ts)):
            self.peak_active_count = max(int(self.peak_active_count), int(self.active_count))
            return
        self.maintenance_runs += 1
        self.last_maintenance_ts = int(ts)
        if self.node_state_ttl_sec > 0.0 and self.state:
            cutoff_ts = int(ts) - int(self.node_state_ttl_sec * 1e9)
            expired = [node for node, node_state in self.state.items() if int(node_state["last_ts"]) < cutoff_ts]
            for node in expired:
                self._drop_node(int(node))
            expired_count = int(len(expired))
            self.evicted_count += expired_count
            self.ttl_evicted_count += expired_count
        self.active_count = int(len(self.state))
        if self.active_node_budget <= 0 or self.active_count <= self.active_node_budget:
            self.peak_active_count = max(int(self.peak_active_count), int(self.active_count))
            return
        overflow = int(self.active_count - self.active_node_budget)
        self.max_budget_overflow = max(int(self.max_budget_overflow), overflow)
        evict_count = int(min(max(overflow, self.eviction_batch), self.active_count))
        evict_nodes = heapq.nsmallest(
            evict_count,
            self.state,
            key=lambda node: (float(self.state[node]["risk"]), int(self.state[node]["last_ts"])),
        )
        for node in evict_nodes:
            self._drop_node(int(node))
        self.active_count = int(len(self.state))
        evicted = int(len(evict_nodes))
        self.evicted_count += evicted
        self.budget_evicted_count += evicted
        self.peak_active_count = max(int(self.peak_active_count), int(self.active_count))

    def update(
        self,
        node: int,
        ts: int,
        event_score: float,
        intrinsic_score: float,
        top_component: int,
        intrinsic_high_count: float,
        intrinsic_soft_sum: float,
        activity_high_count: float,
        activity_soft_sum: float,
        compact_chain_bonus: float,
        chain_profile_bonus: float,
        adaptive_memory_bonus: float,
        association_support: float,
        semantic_support: float,
        local_chain_support: float,
        episode_support: float,
        malicious_chain_support: float,
    ) -> dict[str, float | int]:
        if node < 0 or node >= self.max_node_id:
            return {}
        self.update_count += 1
        node_state = self.state.setdefault(int(node), self._new_node_state())
        self.active_count = int(len(self.state))
        self._decay(node_state, ts)
        intrinsic = (
            float(intrinsic_score)
            + self.count_weight * math.log1p(max(float(intrinsic_high_count), 0.0))
            + self.sum_weight * math.log1p(max(float(intrinsic_soft_sum), 0.0))
        )
        # Semantic rarity stays part of the intrinsic residual, but it should
        # not by itself neutralize activity pressure.
        compact = max(
            0.0,
            self.compact_chain_weight * max(float(compact_chain_bonus), 0.0),
            self.chain_profile_weight * max(float(chain_profile_bonus), 0.0),
            max(float(association_support), 0.0),
            max(float(local_chain_support), 0.0),
            max(float(episode_support), 0.0),
            max(float(malicious_chain_support), 0.0),
        )
        memory_shift = self.adaptive_memory_weight * max(float(adaptive_memory_bonus), 0.0)
        activity = math.log1p(max(float(activity_high_count), 0.0)) + math.log1p(max(float(activity_soft_sum), 0.0))
        support = max(compact, memory_shift)
        pressure = self.activity_weight * max(activity - support - self.activity_margin, 0.0)
        if self.activity_cap > 0.0:
            pressure = min(pressure, self.activity_cap)
        risk = intrinsic + compact + memory_shift - pressure
        node_state["risk"] = max(float(node_state["risk"]), float(risk))
        node_state["intrinsic_residual"] = float(intrinsic)
        node_state["compact_causal_evidence"] = float(compact)
        node_state["episode_subgraph_support"] = float(max(float(episode_support), 0.0))
        node_state["adaptive_memory_shift"] = float(memory_shift)
        node_state["activity_only_pressure"] = float(pressure)
        if float(event_score) >= float(node_state["peak_event_score"]):
            node_state["peak_event_score"] = float(event_score)
            node_state["peak_top_component"] = int(top_component)
        out = {
            "risk_score": float(node_state["risk"]),
            "intrinsic_residual": float(intrinsic),
            "compact_causal_evidence": float(compact),
            "episode_subgraph_support": float(node_state["episode_subgraph_support"]),
            "adaptive_memory_shift": float(memory_shift),
            "activity_only_pressure": float(pressure),
            "activity": float(activity),
            "support": float(support),
            "peak_event_score": float(node_state["peak_event_score"]),
            "top_component_index": int(node_state["peak_top_component"]),
        }
        self._enforce_budget(ts)
        if int(node) not in self.state:
            out["evicted_after_update"] = 1
            out["risk_score"] = 0.0
        return out

    def close(self, ts: int) -> None:
        self._enforce_budget(ts, force=True)

    def memory_bytes(self) -> int:
        # Approximate active-state payload: node id + nine scalar fields.
        return int(len(self.state) * (8 + 9 * 8))


class NodeAlertEmitter:
    """Writes one online alert per node when the current risk crosses threshold."""

    FIELDNAMES = [
        "timestamp",
        "node_id",
        "node_type",
        "risk_score",
        "threshold",
        "first_trigger_event_id",
        "intrinsic_residual",
        "compact_causal_evidence",
        "episode_subgraph_support",
        "adaptive_memory_shift",
        "activity_only_pressure",
        "activity_normalized_chain_density",
        "explanation_json",
    ]

    def __init__(
        self,
        path: str,
        threshold: float,
        threshold_source: str,
        index_to_type: dict[int, str],
        component_names: list[str],
        max_node_id: int,
    ) -> None:
        self.path = str(path)
        self.threshold = float(threshold)
        self.threshold_source = str(threshold_source)
        self.index_to_type = index_to_type
        self.component_names = component_names
        self.alerted = np.zeros((max_node_id,), dtype=bool)
        self.alerts: list[dict[str, int | float | str]] = []
        self.count = 0
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.handle = open(self.path, "w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDNAMES)
        self.writer.writeheader()

    def maybe_emit(
        self,
        node: int,
        timestamp_ns: int,
        event_id: int,
        state: dict[str, float | int],
        evidence: dict[str, object],
    ) -> bool:
        if node < 0 or node >= self.alerted.shape[0] or self.alerted[node]:
            return False
        risk = float(state.get("risk_score", 0.0))
        if risk < self.threshold:
            return False
        self.alerted[node] = True
        self.count += 1
        node_type = self.index_to_type.get(int(node), "unknown")
        comp_idx = int(state.get("top_component_index", -1))
        explanation = {
            "threshold_source": self.threshold_source,
            "top_component": self.component_names[comp_idx] if 0 <= comp_idx < len(self.component_names) else "unknown",
            **evidence,
        }
        self.alerts.append(
            {
                "alert_order": int(self.count),
                "timestamp": int(timestamp_ns),
                "node_id": int(node),
                "node_type": str(node_type),
                "risk_score": float(risk),
                "threshold": float(self.threshold),
                "first_trigger_event_id": int(event_id),
            }
        )
        self.writer.writerow(
            {
                "timestamp": int(timestamp_ns),
                "node_id": int(node),
                "node_type": node_type,
                "risk_score": risk,
                "threshold": float(self.threshold),
                "first_trigger_event_id": int(event_id),
                "intrinsic_residual": float(state.get("intrinsic_residual", 0.0)),
                "compact_causal_evidence": float(state.get("compact_causal_evidence", 0.0)),
                "episode_subgraph_support": float(state.get("episode_subgraph_support", 0.0)),
                "adaptive_memory_shift": float(state.get("adaptive_memory_shift", 0.0)),
                "activity_only_pressure": float(state.get("activity_only_pressure", 0.0)),
                "activity_normalized_chain_density": float(evidence.get("activity_normalized_chain_density", 0.0)),
                "explanation_json": json.dumps(explanation, sort_keys=True, ensure_ascii=True),
            }
        )
        return True

    def close(self) -> None:
        self.handle.close()


def _set_decayed_state_max(state: np.ndarray, last_ts: np.ndarray, node: int, ts: int, score: float, decay_sec: float) -> None:
    if node < 0 or node >= state.shape[0]:
        return
    current = _decayed_scalar_state(state, last_ts, node, ts, decay_sec)
    if float(score) > current:
        state[node] = np.float32(score)
        last_ts[node] = int(ts)


def _popcount(value: int) -> int:
    return int(int(value).bit_count())


NODE_RERANK_FEATURES = [
    "node_max",
    "log_high_count",
    "log_soft_sum",
    "chain_profile_bonus",
    "association_discrepancy_max",
    "log_prop_source_diversity",
    "log_role_type_diversity",
    "semantic_rarity_stability",
    "benign_profile_chain_rarity",
    "local_chain_agreement",
    "compact_malicious_chain_signal",
]


SNC_FEATURES = [
    "actor_cluster_score",
    "log_rare_neighbor_count",
    "log_relation_diversity",
    "log_type_diversity",
    "quarantine_ratio",
    "object_support",
    "object_low_recurrence",
]


def _bit_for_node(node: int) -> np.uint32:
    hashed = (int(node) * 2654435761) & 0xFFFFFFFF
    return np.uint32(1 << (hashed & 31))


def _popcount_array(values: np.ndarray) -> np.ndarray:
    return np.asarray([int(x).bit_count() for x in values.tolist()], dtype=np.float32)


def _new_node_rerank_extra(max_node_id: int) -> dict[str, np.ndarray]:
    return {
        "seen": np.zeros((max_node_id,), dtype=bool),
        "association_max": np.zeros((max_node_id,), dtype=np.float32),
        "association_score_max": np.zeros((max_node_id,), dtype=np.float32),
        "prop_source_mask": np.zeros((max_node_id,), dtype=np.uint32),
        "role_type_mask": np.zeros((max_node_id,), dtype=np.uint8),
        "semantic_peak": np.zeros((max_node_id,), dtype=np.float32),
        "semantic_sum": np.zeros((max_node_id,), dtype=np.float32),
        "semantic_count": np.zeros((max_node_id,), dtype=np.float32),
        "local_chain_agreement": np.zeros((max_node_id,), dtype=np.float32),
        "malicious_chain_peak": np.zeros((max_node_id,), dtype=np.float32),
        "malicious_chain_sum": np.zeros((max_node_id,), dtype=np.float32),
        "malicious_chain_count": np.zeros((max_node_id,), dtype=np.float32),
        "malicious_chain_rel_mask": np.zeros((max_node_id,), dtype=np.uint32),
    }


def _new_snc_state(max_node_id: int) -> dict[str, np.ndarray]:
    return {
        "actor_score": np.zeros((max_node_id,), dtype=np.float32),
        "actor_rare_neighbors": np.zeros((max_node_id,), dtype=np.float32),
        "actor_rel_mask": np.zeros((max_node_id,), dtype=np.uint32),
        "actor_type_mask": np.zeros((max_node_id,), dtype=np.uint8),
        "actor_quarantine": np.zeros((max_node_id,), dtype=np.float32),
        "actor_total": np.zeros((max_node_id,), dtype=np.float32),
        "actor_last_ts": np.full((max_node_id,), -1, dtype=np.int64),
        "object_support": np.zeros((max_node_id,), dtype=np.float32),
        "object_low_recurrence": np.zeros((max_node_id,), dtype=np.float32),
        "object_actor": np.full((max_node_id,), -1, dtype=np.int64),
        "object_event_pos": np.full((max_node_id,), -1, dtype=np.int64),
    }


def _decay_snc_actor(state: dict[str, np.ndarray], actor: int, ts: int, decay_sec: float) -> None:
    if not state or actor < 0 or actor >= state["actor_score"].shape[0]:
        return
    last = int(state["actor_last_ts"][actor])
    if last >= 0:
        dt = max(float(ts - last) / 1e9, 0.0)
        decay = math.exp(-dt / max(float(decay_sec), 1e-3))
        for key in ("actor_score", "actor_rare_neighbors", "actor_quarantine", "actor_total"):
            state[key][actor] = np.float32(float(state[key][actor]) * decay)
        if float(state["actor_total"][actor]) < 1e-4:
            state["actor_rel_mask"][actor] = np.uint32(0)
            state["actor_type_mask"][actor] = np.uint8(0)
    state["actor_last_ts"][actor] = int(ts)


def _snc_event_support(
    event_tail: float,
    semantic_z: float,
    chain_bonus: float,
    chain_profile_score: float,
    association_z: float,
    adaptive_bonus: float,
    quarantined: bool,
) -> float:
    local = max(float(event_tail), float(semantic_z), 0.0)
    relation = max(float(chain_bonus), float(chain_profile_score), 0.0)
    assoc = max(float(association_z), 0.0)
    memory = max(float(adaptive_bonus), 1.0 if bool(quarantined) else 0.0, 0.0)
    return float(0.35 * local + 0.30 * relation + 0.25 * assoc + 0.10 * memory)


def _update_snc_state(
    state: dict[str, np.ndarray],
    actor: int,
    obj: int,
    ts: int,
    support: float,
    association_z: float,
    relation_id: int,
    obj_type: int,
    object_count: float,
    quarantined: bool,
    event_pos: int,
    decay_sec: float,
    update_min_support: float,
    update_min_assoc_z: float,
) -> None:
    if not state or actor < 0 or obj < 0 or actor >= state["actor_score"].shape[0] or obj >= state["actor_score"].shape[0]:
        return
    _decay_snc_actor(state, actor, ts, decay_sec)
    state["actor_total"][actor] += np.float32(1.0)
    if bool(quarantined):
        state["actor_quarantine"][actor] += np.float32(1.0)
    rare_event = float(support) >= float(update_min_support) and float(association_z) >= float(update_min_assoc_z)
    if rare_event:
        state["actor_rare_neighbors"][actor] += np.float32(1.0)
        if float(support) > float(state["actor_score"][actor]):
            state["actor_score"][actor] = np.float32(min(float(support), 64.0))
        rel_bit = np.uint32(1 << (int(relation_id) & 31))
        type_bit = np.uint8(1 << (int(obj_type) & 7))
        state["actor_rel_mask"][actor] = np.uint32(int(state["actor_rel_mask"][actor]) | int(rel_bit))
        state["actor_type_mask"][actor] = np.uint8(int(state["actor_type_mask"][actor]) | int(type_bit))
        low_recurrence = 1.0 / math.sqrt(max(float(object_count), 1.0))
        object_support = float(support) * (1.0 + 0.50 * low_recurrence)
        if object_support > float(state["object_support"][obj]):
            state["object_support"][obj] = np.float32(min(object_support, 64.0))
            state["object_low_recurrence"][obj] = np.float32(low_recurrence)
            state["object_actor"][obj] = int(actor)
            state["object_event_pos"][obj] = int(event_pos)


def _snc_feature_matrix(
    mask: np.ndarray,
    state: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.flatnonzero(np.asarray(mask, dtype=bool))
    if keep.size <= 0 or not state:
        return keep, np.zeros((0, len(SNC_FEATURES)), dtype=np.float32)
    actor = state["object_actor"][keep].astype(np.int64, copy=False)
    valid = (actor >= 0) & (actor < state["actor_score"].shape[0])
    actor_safe = np.where(valid, actor, 0)
    total = state["actor_total"][actor_safe].astype(np.float32, copy=False)
    quarantine_ratio = np.divide(
        state["actor_quarantine"][actor_safe].astype(np.float32, copy=False),
        np.maximum(total, 1.0),
        out=np.zeros_like(total, dtype=np.float32),
        where=valid,
    )
    matrix = np.column_stack(
        [
            np.where(valid, state["actor_score"][actor_safe], 0.0),
            np.where(valid, np.log1p(state["actor_rare_neighbors"][actor_safe]), 0.0),
            np.where(valid, np.log1p(_popcount_array(state["actor_rel_mask"][actor_safe])), 0.0),
            np.where(valid, np.log1p(_popcount_array(state["actor_type_mask"][actor_safe])), 0.0),
            quarantine_ratio,
            state["object_support"][keep].astype(np.float32, copy=False),
            state["object_low_recurrence"][keep].astype(np.float32, copy=False),
        ]
    ).astype(np.float32, copy=False)
    return keep, matrix


def _snc_feature_stats(matrix: np.ndarray) -> list[dict[str, float]]:
    if matrix.size <= 0:
        return [{"median": 0.0, "mad": 1.0, "q95": 0.0, "q99": 0.0, "q995": 0.0, "q999": 0.0} for _ in SNC_FEATURES]
    return [robust_stats(matrix[:, i]) for i in range(matrix.shape[1])]


def _snc_feature_calibrations(matrix: np.ndarray, num_knots: int) -> list[dict[str, object]]:
    if matrix.size <= 0:
        return [_quantile_calibration(np.zeros((0,), dtype=np.float32), num_knots) for _ in SNC_FEATURES]
    return [_quantile_calibration(matrix[:, i], int(num_knots)) for i in range(matrix.shape[1])]


def _node_feature_calibrations(matrix: np.ndarray, num_knots: int) -> list[dict[str, object]]:
    if matrix.size <= 0:
        return [_quantile_calibration(np.zeros((0,), dtype=np.float32), num_knots) for _ in NODE_RERANK_FEATURES]
    return [_quantile_calibration(matrix[:, i], int(num_knots)) for i in range(matrix.shape[1])]


def _calibrated_feature_matrix(
    matrix: np.ndarray,
    stats: list[dict[str, float]],
    calibrations: list[dict[str, object]],
    calibration_mode: str,
    clip: float,
) -> np.ndarray:
    if matrix.size <= 0:
        return np.zeros((0, len(stats)), dtype=np.float32)
    if str(calibration_mode) == "tail":
        cols = [
            _tail_surprise_array(col, calibration, clip)
            for col, calibration in zip(matrix.T, calibrations)
        ]
    else:
        cols = [
            _upper_z_array(col, stat, clip)
            for col, stat in zip(matrix.T, stats)
        ]
    return np.column_stack(cols).astype(np.float32, copy=False)


def _tail_surprise_array(values: np.ndarray, calibration: dict[str, object], clip: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    q_raw = calibration.get("quantiles", [])
    v_raw = calibration.get("values", [])
    if not q_raw or not v_raw or arr.size <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    quantiles = np.asarray(q_raw, dtype=np.float64)
    q_values = np.asarray(v_raw, dtype=np.float64)
    q = np.interp(arr.astype(np.float64, copy=False), q_values, quantiles, left=0.0, right=float(quantiles[-1]))
    surprise = -np.log10(np.maximum(1.0 - q, 1e-9))
    surprise = np.maximum(surprise, 0.0)
    if float(clip) > 0.0:
        surprise = np.minimum(surprise, float(clip))
    return surprise.astype(np.float32, copy=False)


def _apply_snc_score(
    node_score: np.ndarray,
    ranked_before_snc: np.ndarray,
    snc_keep: np.ndarray,
    snc_matrix: np.ndarray,
    snc_stats: list[dict[str, float]],
    snc_calibrations: list[dict[str, object]],
    node_high_count: np.ndarray,
    adaptive_memory_update_node: np.ndarray,
    calibration_mode: str,
    clip: float,
    candidate_start_topk: int,
    candidate_end_topk: int,
    weight: float,
    boost_cap: float,
    penalty_weight: float,
    penalty_cap: float,
    min_cluster_z: float,
    min_object_z: float,
    min_diversity_z: float,
    high_count_penalty_min: float,
    benign_update_penalty_min: float,
    benign_update_weight: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    arrays = {
        "candidate": np.zeros((node_score.shape[0],), dtype=bool),
        "applied": np.zeros((node_score.shape[0],), dtype=bool),
        "pass": np.zeros((node_score.shape[0],), dtype=bool),
        "score": np.zeros((node_score.shape[0],), dtype=np.float32),
        "boost": np.zeros((node_score.shape[0],), dtype=np.float32),
        "penalty": np.zeros((node_score.shape[0],), dtype=np.float32),
        "cluster_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "object_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "diversity_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "quarantine_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "low_recurrence": np.zeros((node_score.shape[0],), dtype=np.float32),
        "benign_update_penalty": np.zeros((node_score.shape[0],), dtype=np.float32),
        "high_count_penalty": np.zeros((node_score.shape[0],), dtype=np.float32),
    }
    start = min(max(int(candidate_start_topk), 0), int(ranked_before_snc.shape[0]))
    end = min(max(int(candidate_end_topk), start), int(ranked_before_snc.shape[0]))
    if end <= start or snc_matrix.size <= 0 or snc_keep.size <= 0:
        return node_score, arrays, {
            "candidate_nodes": 0,
            "applied_nodes": 0,
            "passed_nodes": 0,
            "max_boost": 0.0,
            "mean_boost": 0.0,
            "max_penalty": 0.0,
            "mean_score": 0.0,
        }
    if str(calibration_mode) == "tail":
        z = np.column_stack(
            [_tail_surprise_array(col, calibration, clip) for col, calibration in zip(snc_matrix.T, snc_calibrations)]
        ).astype(np.float32, copy=False)
    else:
        z = np.column_stack([_upper_z_array(col, stat, clip) for col, stat in zip(snc_matrix.T, snc_stats)]).astype(np.float32, copy=False)
    cluster_z = np.maximum(
        z[:, SNC_FEATURES.index("actor_cluster_score")],
        z[:, SNC_FEATURES.index("log_rare_neighbor_count")],
    )
    diversity_z = np.maximum(
        z[:, SNC_FEATURES.index("log_relation_diversity")],
        z[:, SNC_FEATURES.index("log_type_diversity")],
    )
    quarantine_z = z[:, SNC_FEATURES.index("quarantine_ratio")]
    object_z = z[:, SNC_FEATURES.index("object_support")]
    low_recurrence = snc_matrix[:, SNC_FEATURES.index("object_low_recurrence")]

    arrays["cluster_z"][snc_keep] = cluster_z
    arrays["diversity_z"][snc_keep] = diversity_z
    arrays["quarantine_z"][snc_keep] = quarantine_z
    arrays["object_z"][snc_keep] = object_z
    arrays["low_recurrence"][snc_keep] = low_recurrence

    candidate_nodes = ranked_before_snc[start:end].astype(np.int64, copy=False)
    candidate_nodes = candidate_nodes[(candidate_nodes >= 0) & (candidate_nodes < node_score.shape[0])]
    arrays["candidate"][candidate_nodes] = True
    snc_pos = {int(node): idx for idx, node in enumerate(snc_keep.tolist())}
    positions = [snc_pos[int(node)] for node in candidate_nodes.tolist() if int(node) in snc_pos]
    if not positions:
        return node_score, arrays, {
            "candidate_nodes": int(candidate_nodes.size),
            "applied_nodes": 0,
            "passed_nodes": 0,
            "max_boost": 0.0,
            "mean_boost": 0.0,
            "max_penalty": 0.0,
            "mean_score": 0.0,
        }
    pos = np.asarray(positions, dtype=np.int64)
    nodes = snc_keep[pos].astype(np.int64, copy=False)
    node_cluster = cluster_z[pos]
    node_diversity = diversity_z[pos]
    node_object = object_z[pos]
    node_quarantine = quarantine_z[pos]
    node_low_recur = low_recurrence[pos]
    closure_score = (
        0.35 * node_cluster
        + 0.25 * node_object
        + 0.20 * node_diversity
        + 0.10 * node_quarantine
        + 0.10 * np.minimum(node_low_recur, 1.0)
    )
    passed = (
        (node_cluster >= float(min_cluster_z))
        & (node_object >= float(min_object_z))
        & (node_diversity >= float(min_diversity_z))
    )
    boost = np.where(passed, float(weight) * closure_score, 0.0)
    if float(boost_cap) > 0.0:
        boost = np.minimum(boost, float(boost_cap))

    high_count_excess = np.maximum(node_high_count[nodes].astype(np.float32, copy=False) - float(high_count_penalty_min), 0.0)
    high_count_penalty = float(penalty_weight) * np.log1p(high_count_excess)
    benign_update_excess = np.maximum(adaptive_memory_update_node[nodes].astype(np.float32, copy=False) - float(benign_update_penalty_min), 0.0)
    update_penalty = float(benign_update_weight) * np.log1p(benign_update_excess)
    missing_gap = (
        np.maximum(float(min_cluster_z) - node_cluster, 0.0)
        + np.maximum(float(min_object_z) - node_object, 0.0)
        + np.maximum(float(min_diversity_z) - node_diversity, 0.0)
    )
    penalty = np.where(~passed, float(penalty_weight) * missing_gap, 0.0) + high_count_penalty + update_penalty
    if float(penalty_cap) > 0.0:
        penalty = np.minimum(penalty, float(penalty_cap))
    delta = boost - penalty
    apply_mask = np.abs(delta) > 0.0
    if np.any(apply_mask):
        apply_nodes = nodes[apply_mask]
        node_score[apply_nodes] = node_score[apply_nodes] + delta[apply_mask].astype(np.float32, copy=False)
        arrays["applied"][apply_nodes] = True
    arrays["pass"][nodes[passed]] = True
    arrays["score"][nodes] = closure_score.astype(np.float32, copy=False)
    arrays["boost"][nodes] = boost.astype(np.float32, copy=False)
    arrays["penalty"][nodes] = penalty.astype(np.float32, copy=False)
    arrays["benign_update_penalty"][nodes] = update_penalty.astype(np.float32, copy=False)
    arrays["high_count_penalty"][nodes] = high_count_penalty.astype(np.float32, copy=False)
    return node_score, arrays, {
        "candidate_start_topk": int(start),
        "candidate_end_topk": int(end),
        "candidate_nodes": int(nodes.size),
        "applied_nodes": int(np.sum(apply_mask)),
        "passed_nodes": int(np.sum(passed)),
        "max_boost": float(np.max(boost)) if boost.size else 0.0,
        "mean_boost": float(np.mean(boost[boost > 0.0])) if np.any(boost > 0.0) else 0.0,
        "max_penalty": float(np.max(penalty)) if penalty.size else 0.0,
        "mean_score": float(np.mean(closure_score)) if closure_score.size else 0.0,
    }


def _compact_malicious_chain_signal(
    event_tail: float,
    semantic_z: float,
    chain_bonus: float,
    chain_profile_score: float,
    transition_rarity: float,
    association_discrepancy: float,
    association_score: float,
    has_transition: float,
) -> float:
    """Streaming short-chain discriminator from benign-profile disagreement.

    The signal is intentionally label-free: it rewards agreement between local
    event residual/semantic rarity and benign-profile chain/association rarity.
    It is later calibrated against validation node distributions.
    """
    local = max(float(event_tail), float(semantic_z), 0.0)
    chain = max(float(chain_bonus), float(chain_profile_score), float(transition_rarity), 0.0)
    assoc = max(float(association_discrepancy), float(association_score), 0.0)
    if local <= 0.0 or chain <= 0.0:
        return 0.0
    local_chain_agreement = min(local, chain)
    association_agreement = min(max(local, chain), assoc) if assoc > 0.0 else 0.0
    transition_agreement = min(chain, max(float(transition_rarity), 0.0)) if float(has_transition) > 0.0 else 0.0
    return float(local_chain_agreement + 0.50 * association_agreement + 0.25 * transition_agreement)


def _causal_episode_bucket_key(row: dict[str, object]) -> int:
    return int(
        stable_hash(
            "|".join(
                [
                    f"src={int(row['info_src'])}",
                    f"rel={int(row['relation_id'])}",
                    f"src_t={int(row['info_src_type'])}",
                    f"dst_t={int(row['info_dst_type'])}",
                ]
            ),
            seed=4127,
        )
    )


class CausalEpisodeSketch:
    """Bounded causal episode/subgraph support for unified C_t(v)."""

    def __init__(self, max_node_id: int, num_buckets: int, decay_sec: float, score_cap: float) -> None:
        self.max_node_id = int(max_node_id)
        self.num_buckets = max(int(num_buckets), 8)
        self.decay_sec = max(float(decay_sec), 1e-3)
        self.score_cap = max(float(score_cap), 0.0)
        self.bucket_score = np.zeros((self.num_buckets,), dtype=np.float32)
        self.bucket_last_ts = np.full((self.num_buckets,), -1, dtype=np.int64)
        self.bucket_count = np.zeros((self.num_buckets,), dtype=np.float32)
        self.bucket_rel_mask = np.zeros((self.num_buckets,), dtype=np.uint32)
        self.bucket_type_mask = np.zeros((self.num_buckets,), dtype=np.uint8)
        self.node_support = np.zeros((self.max_node_id,), dtype=np.float32)
        self.node_count = np.zeros((self.max_node_id,), dtype=np.float32)
        self.node_rel_mask = np.zeros((self.max_node_id,), dtype=np.uint32)
        self.node_type_mask = np.zeros((self.max_node_id,), dtype=np.uint8)
        self.node_event_pos = np.full((self.max_node_id,), -1, dtype=np.int64)
        self.update_count = 0
        self.score_count = 0
        self.skipped_count = 0
        self.selective_reject_count = 0
        self.active_buckets = 0

    def _decay_bucket(self, bucket: int, ts: int) -> float:
        last = int(self.bucket_last_ts[bucket])
        score = float(self.bucket_score[bucket])
        if last >= 0 and score > 0.0:
            dt = max(float(int(ts) - last) / 1e9, 0.0)
            score *= math.exp(-dt / self.decay_sec)
            self.bucket_score[bucket] = np.float32(score)
        self.bucket_last_ts[bucket] = int(ts)
        return score

    def score_pre_update(
        self,
        row: dict[str, object],
        ts: int,
        event_tail: float,
        chain_profile_bonus: float,
        association_support: float,
        transition_rarity: float,
        adaptive_support: float = 0.0,
        quarantined: bool = False,
        min_event_tail: float = 0.0,
        min_chain: float = 0.0,
        min_assoc: float = 0.0,
        min_families: int = 1,
        selective: bool = False,
    ) -> dict[str, float | int]:
        bucket = int(_causal_episode_bucket_key(row) % self.num_buckets)
        chain_value = max(float(chain_profile_bonus), float(transition_rarity), 0.0)
        assoc_value = max(float(association_support), 0.0)
        adaptive_value = max(float(adaptive_support), 0.0)
        event_tail_hit = float(event_tail) >= float(min_event_tail)
        nonsemantic_family_count = (
            float(chain_value >= float(min_chain))
            + float(assoc_value >= float(min_assoc))
            + float(adaptive_value >= float(min_assoc))
            + float(bool(quarantined))
        )
        min_family_count = float(max(int(min_families), 1))
        family_count = nonsemantic_family_count + float(event_tail_hit)
        selector = bool(event_tail_hit or nonsemantic_family_count >= min_family_count or bool(quarantined))
        candidate = bool(nonsemantic_family_count >= min_family_count)
        if bool(selective) and not candidate:
            self.skipped_count += 1
            self.selective_reject_count += 1
            return {
                "bucket": int(bucket),
                "support": 0.0,
                "raw_support": 0.0,
                "context_support": 0.0,
                "family_count": 0.0,
                "nonsemantic_family_count": float(nonsemantic_family_count),
                "selector": int(selector),
                "candidate": 0,
            }
        self.score_count += 1
        context = self._decay_bucket(bucket, ts)
        nonsemantic_seed = max(chain_value, assoc_value, adaptive_value, 0.0)
        if (nonsemantic_seed <= 0.0 and context <= 0.0) or (float(event_tail) <= 0.0 and nonsemantic_seed <= 0.0):
            self.skipped_count += 1
            return {
                "bucket": int(bucket),
                "support": 0.0,
                "raw_support": 0.0,
                "context_support": float(context),
                "family_count": float(family_count),
                "nonsemantic_family_count": float(nonsemantic_family_count),
                "selector": int(selector),
                "candidate": int(candidate),
            }
        current = max(float(event_tail), nonsemantic_seed)
        support = min(current, nonsemantic_seed + 0.5 * float(context))
        support_family_count = family_count + float(context > 0.0)
        support = min(
            float(support)
            + 0.15 * math.log1p(float(self.bucket_count[bucket]))
            + 0.10 * support_family_count,
            float(self.score_cap) if float(self.score_cap) > 0.0 else float("inf"),
        )
        support = max(float(support), 0.0)
        return {
            "bucket": int(bucket),
            "support": float(support),
            "raw_support": float(current),
            "context_support": float(context),
            "family_count": float(support_family_count),
            "nonsemantic_family_count": float(nonsemantic_family_count),
            "selector": int(selector),
            "candidate": int(candidate),
        }

    def update_state(
        self,
        row: dict[str, object],
        nodes: set[int],
        ts: int,
        support: float,
        raw_support: float,
        relation_id: int,
        src_type: int,
        dst_type: int,
        event_pos: int,
    ) -> None:
        bucket = int(_causal_episode_bucket_key(row) % self.num_buckets)
        current = self._decay_bucket(bucket, ts)
        was_active = current > 0.0
        self.update_count += 1
        current = max(current, float(raw_support), float(support))
        if float(self.score_cap) > 0.0:
            current = min(current, float(self.score_cap))
        self.bucket_score[bucket] = np.float32(current)
        self.bucket_count[bucket] += np.float32(1.0)
        self.bucket_rel_mask[bucket] = np.uint32(int(self.bucket_rel_mask[bucket]) | int(1 << (int(relation_id) & 31)))
        self.bucket_type_mask[bucket] = np.uint8(int(self.bucket_type_mask[bucket]) | int((1 << (int(src_type) & 7)) | (1 << (int(dst_type) & 7))))
        if (not was_active) and current > 0.0:
            self.active_buckets += 1
        for node in nodes:
            node_int = int(node)
            if node_int < 0 or node_int >= self.max_node_id:
                continue
            if float(support) > float(self.node_support[node_int]):
                self.node_support[node_int] = np.float32(min(float(support), float(self.score_cap) if float(self.score_cap) > 0.0 else float(support)))
                self.node_event_pos[node_int] = int(event_pos)
            self.node_count[node_int] += np.float32(1.0)
            self.node_rel_mask[node_int] = np.uint32(int(self.node_rel_mask[node_int]) | int(1 << (int(relation_id) & 31)))
            self.node_type_mask[node_int] = np.uint8(int(self.node_type_mask[node_int]) | int((1 << (int(src_type) & 7)) | (1 << (int(dst_type) & 7))))

    def metadata(self) -> dict[str, object]:
        return {
            "num_buckets": int(self.num_buckets),
            "decay_sec": float(self.decay_sec),
            "score_cap": float(self.score_cap),
            "active_buckets": int(np.count_nonzero(self.bucket_score > 0.0)),
            "active_nodes": int(np.count_nonzero(self.node_support > 0.0)),
            "state_size_bytes": int(self.memory_bytes()),
            "score_count": int(self.score_count),
            "update_count": int(self.update_count),
            "skipped_count": int(self.skipped_count),
            "selective_reject_count": int(self.selective_reject_count),
            "definition": "Bounded causal episode/subgraph sketch from recent non-semantic support in a hashed short-chain neighborhood.",
        }

    def memory_bytes(self) -> int:
        return int(
            self.bucket_score.nbytes
            + self.bucket_last_ts.nbytes
            + self.bucket_count.nbytes
            + self.bucket_rel_mask.nbytes
            + self.bucket_type_mask.nbytes
            + self.node_support.nbytes
            + self.node_count.nbytes
            + self.node_rel_mask.nbytes
            + self.node_type_mask.nbytes
            + self.node_event_pos.nbytes
        )


def _new_adaptive_memory(num_buckets: int, latent_dim: int) -> dict[str, np.ndarray | int]:
    buckets = max(int(num_buckets), 8)
    dim = max(int(latent_dim), 1)
    return {
        "mean": np.zeros((buckets, dim), dtype=np.float32),
        "count": np.zeros((buckets,), dtype=np.int32),
        "updates": 0,
        "quarantined": 0,
    }


def _copy_adaptive_memory(memory: dict[str, np.ndarray | int]) -> dict[str, np.ndarray | int]:
    return {
        "mean": np.asarray(memory["mean"], dtype=np.float32).copy(),
        "count": np.asarray(memory["count"], dtype=np.int32).copy(),
        "updates": int(memory.get("updates", 0)),
        "quarantined": int(memory.get("quarantined", 0)),
    }


def _adaptive_memory_key(row: dict[str, object], num_buckets: int) -> int:
    key = (
        f"rel={int(row['relation_id'])}|"
        f"types={int(row['info_src_type'])}>{int(row['info_dst_type'])}|"
        f"action={row.get('action', '')}|obj={row.get('object_type', '')}"
    )
    return int(stable_hash(key, seed=1907) % max(int(num_buckets), 8))


def _adaptive_memory_distance(
    memory: dict[str, np.ndarray | int],
    row: dict[str, object],
    z: np.ndarray,
    min_count: int,
) -> tuple[float, bool, int]:
    mean = np.asarray(memory["mean"], dtype=np.float32)
    count = np.asarray(memory["count"], dtype=np.int32)
    key = _adaptive_memory_key(row, mean.shape[0])
    if int(count[key]) < max(int(min_count), 1):
        return 0.0, False, key
    vec = np.asarray(z, dtype=np.float32)
    diff = vec - mean[key]
    distance = float(math.sqrt(float(np.mean(diff * diff))))
    return distance, True, key


def _adaptive_memory_update(
    memory: dict[str, np.ndarray | int],
    key: int,
    z: np.ndarray,
    alpha: float,
) -> None:
    mean = np.asarray(memory["mean"], dtype=np.float32)
    count = np.asarray(memory["count"], dtype=np.int32)
    if key < 0 or key >= mean.shape[0]:
        return
    vec = np.asarray(z, dtype=np.float32)
    current = int(count[key])
    if current <= 0:
        mean[key] = vec
    else:
        rate = float(alpha) if float(alpha) > 0.0 else 1.0 / float(current + 1)
        rate = min(max(rate, 1e-4), 1.0)
        mean[key] = (1.0 - rate) * mean[key] + rate * vec
    count[key] = np.int32(min(current + 1, 2_000_000_000))
    memory["updates"] = int(memory.get("updates", 0)) + 1


def _adaptive_memory_assoc_value(chain_details: dict[str, float]) -> float:
    return float(max(float(chain_details.get("association_discrepancy", 0.0)), float(chain_details.get("association_score", 0.0)), 0.0))


def _adaptive_memory_quarantine(
    event_tail: float,
    chain_tail: float,
    assoc_z: float,
    quarantine_tail_min: float,
    quarantine_assoc_z_min: float,
) -> bool:
    return bool(
        float(event_tail) >= float(quarantine_tail_min)
        or float(chain_tail) >= float(quarantine_tail_min)
        or float(assoc_z) >= float(quarantine_assoc_z_min)
    )


def _adaptive_memory_can_update(
    event_tail: float,
    chain_tail: float,
    assoc_z: float,
    update_tail_max: float,
    update_assoc_z_max: float,
    quarantined: bool,
) -> bool:
    return bool(
        (not quarantined)
        and float(event_tail) <= float(update_tail_max)
        and float(chain_tail) <= float(update_tail_max)
        and float(assoc_z) <= float(update_assoc_z_max)
    )


def _update_node_rerank_extra(
    extra: dict[str, np.ndarray],
    nodes: set[int],
    prop_source: int,
    prop_active: bool,
    semantic_z: float,
    chain_bonus: float,
    chain_raw: float,
    association_discrepancy: float,
    association_score: float,
    malicious_chain_signal: float,
    relation_id: int,
    src_type: int,
    dst_type: int,
    local_score: float,
    max_node_id: int,
) -> None:
    if not extra:
        return
    source_bit = _bit_for_node(prop_source) if prop_active and 0 <= int(prop_source) < max_node_id else np.uint32(0)
    semantic_value = max(float(semantic_z), 0.0)
    assoc_value = max(float(association_discrepancy), float(association_score), 0.0)
    malicious_value = max(float(malicious_chain_signal), 0.0)
    rel_bit = np.uint32(1 << (int(relation_id) & 31)) if malicious_value > 0.0 else np.uint32(0)
    type_bits = np.uint8((1 << (int(src_type) & 7)) | (1 << (int(dst_type) & 7)))
    agreement = min(max(float(local_score), 0.0), max(float(chain_bonus), 0.0))
    if semantic_value > 0.0 and assoc_value > 0.0:
        agreement = max(agreement, min(semantic_value, assoc_value))
    for raw_node in nodes:
        node = int(raw_node)
        if node < 0 or node >= max_node_id:
            continue
        extra["seen"][node] = True
        if semantic_value > 0.0:
            if semantic_value > float(extra["semantic_peak"][node]):
                extra["semantic_peak"][node] = np.float32(semantic_value)
            extra["semantic_sum"][node] += np.float32(min(semantic_value, 32.0))
            extra["semantic_count"][node] += np.float32(1.0)
        if assoc_value > float(extra["association_max"][node]):
            extra["association_max"][node] = np.float32(assoc_value)
        if float(association_score) > float(extra["association_score_max"][node]):
            extra["association_score_max"][node] = np.float32(max(float(association_score), 0.0))
        if source_bit:
            extra["prop_source_mask"][node] = np.uint32(int(extra["prop_source_mask"][node]) | int(source_bit))
        if type_bits:
            extra["role_type_mask"][node] = np.uint8(int(extra["role_type_mask"][node]) | int(type_bits))
        if agreement > float(extra["local_chain_agreement"][node]):
            extra["local_chain_agreement"][node] = np.float32(agreement)
        if malicious_value > 0.0:
            if malicious_value > float(extra["malicious_chain_peak"][node]):
                extra["malicious_chain_peak"][node] = np.float32(malicious_value)
            extra["malicious_chain_sum"][node] += np.float32(min(malicious_value, 64.0))
            extra["malicious_chain_count"][node] += np.float32(1.0)
            extra["malicious_chain_rel_mask"][node] = np.uint32(int(extra["malicious_chain_rel_mask"][node]) | int(rel_bit))


def _node_rerank_feature_matrix(
    mask: np.ndarray,
    node_max: np.ndarray,
    node_high_count: np.ndarray,
    node_soft_sum: np.ndarray,
    chain_profile_bonus: np.ndarray,
    chain_profile_raw: np.ndarray,
    extra: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.flatnonzero(np.asarray(mask, dtype=bool))
    if keep.size <= 0:
        return keep, np.zeros((0, len(NODE_RERANK_FEATURES)), dtype=np.float32)
    if chain_profile_bonus.size <= 0:
        chain_bonus = np.zeros((keep.size,), dtype=np.float32)
        chain_raw = np.zeros((keep.size,), dtype=np.float32)
    else:
        chain_bonus = chain_profile_bonus[keep].astype(np.float32, copy=False)
        chain_raw = chain_profile_raw[keep].astype(np.float32, copy=False)
    if extra:
        assoc = extra["association_max"][keep].astype(np.float32, copy=False)
        prop_div = _popcount_array(extra["prop_source_mask"][keep])
        role_type_div = _popcount_array(extra["role_type_mask"][keep])
        semantic_peak = extra["semantic_peak"][keep].astype(np.float32, copy=False)
        semantic_sum = extra["semantic_sum"][keep].astype(np.float32, copy=False)
        semantic_count = extra["semantic_count"][keep].astype(np.float32, copy=False)
        agreement = extra["local_chain_agreement"][keep].astype(np.float32, copy=False)
        malicious_peak = extra["malicious_chain_peak"][keep].astype(np.float32, copy=False)
        malicious_sum = extra["malicious_chain_sum"][keep].astype(np.float32, copy=False)
        malicious_count = extra["malicious_chain_count"][keep].astype(np.float32, copy=False)
        malicious_rel_div = _popcount_array(extra["malicious_chain_rel_mask"][keep])
    else:
        assoc = np.zeros((keep.size,), dtype=np.float32)
        prop_div = np.zeros((keep.size,), dtype=np.float32)
        role_type_div = np.zeros((keep.size,), dtype=np.float32)
        semantic_peak = np.zeros((keep.size,), dtype=np.float32)
        semantic_sum = np.zeros((keep.size,), dtype=np.float32)
        semantic_count = np.zeros((keep.size,), dtype=np.float32)
        agreement = np.zeros((keep.size,), dtype=np.float32)
        malicious_peak = np.zeros((keep.size,), dtype=np.float32)
        malicious_sum = np.zeros((keep.size,), dtype=np.float32)
        malicious_count = np.zeros((keep.size,), dtype=np.float32)
        malicious_rel_div = np.zeros((keep.size,), dtype=np.float32)
    semantic_stability = semantic_peak * np.log1p(semantic_count) + 0.10 * semantic_sum
    malicious_chain_signal = (
        malicious_peak * np.log1p(malicious_count)
        + 0.10 * malicious_sum
        + 0.25 * np.log1p(malicious_rel_div)
    )
    matrix = np.column_stack(
        [
            node_max[keep],
            np.log1p(node_high_count[keep]),
            np.log1p(node_soft_sum[keep]),
            chain_bonus,
            assoc,
            np.log1p(prop_div),
            np.log1p(role_type_div),
            semantic_stability.astype(np.float32, copy=False),
            chain_raw,
            agreement,
            malicious_chain_signal.astype(np.float32, copy=False),
        ]
    ).astype(np.float32, copy=False)
    return keep, matrix


def _node_feature_stats(matrix: np.ndarray) -> list[dict[str, float]]:
    if matrix.size <= 0:
        return [{"median": 0.0, "mad": 1.0, "q95": 0.0, "q99": 0.0, "q995": 0.0, "q999": 0.0} for _ in NODE_RERANK_FEATURES]
    return [robust_stats(matrix[:, i]) for i in range(matrix.shape[1])]


def _upper_z_array(values: np.ndarray, stats: dict[str, float], clip: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    scale = max(float(stats["mad"]) * 1.4826, 1e-6)
    out = np.maximum((arr - float(stats["median"])) / scale, 0.0)
    if float(clip) > 0.0:
        out = np.minimum(out, float(clip))
    return out.astype(np.float32, copy=False)


def _node_rerank_delta(
    matrix: np.ndarray,
    stats: list[dict[str, float]],
    calibrations: list[dict[str, object]],
    calibration_mode: str,
    feature_weights: np.ndarray,
    clip: float,
    penalty_weight: float,
    margin: float,
    high_activity_weight: float,
    high_activity_margin: float,
    high_activity_cap: float,
    joint_activity_pressure: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if matrix.size <= 0:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty, empty, empty, empty, empty, empty
    z = _calibrated_feature_matrix(matrix, stats, calibrations, calibration_mode, clip)
    weights = feature_weights.astype(np.float32, copy=False)
    weights = weights / max(float(np.sum(np.maximum(weights, 0.0))), 1e-6)
    support = np.sum(z * weights.reshape(1, -1), axis=1)
    intrinsic_feature_indices = [0, 3, 4, 6, 7, 8, 9, 10]
    intrinsic = np.maximum.reduce([z[:, idx] for idx in intrinsic_feature_indices])
    hard_support = np.maximum.reduce([z[:, idx] for idx in [3, 4, 6, 7, 8, 9, 10]])
    evidence_family_count = (
        (np.maximum(z[:, 3], z[:, 7]) >= 1.0).astype(np.float32)
        + (z[:, 4] >= 1.0).astype(np.float32)
        + (z[:, 6] >= 1.0).astype(np.float32)
        + (z[:, 7] >= 1.0).astype(np.float32)
        + (z[:, 8] >= 1.0).astype(np.float32)
        + (z[:, 9] >= 1.0).astype(np.float32)
        + (z[:, 10] >= 1.0).astype(np.float32)
    )
    propagation_pressure = 0.65 * z[:, 5] + 0.35 * z[:, 1]
    prop_only_penalty = np.maximum(propagation_pressure - intrinsic - float(margin), 0.0)
    if bool(joint_activity_pressure):
        high_activity = np.maximum(z[:, 1], z[:, 2]) + 0.5 * np.minimum(z[:, 1], z[:, 2])
    else:
        high_activity = np.maximum(z[:, 1], z[:, 2])
    # Treat semantic rarity as a residual cue, not as a blanket shield against
    # high-activity suppression.
    benign_support = np.maximum.reduce(
        [
            z[:, 3],
            z[:, 4],
            z[:, 6],
            z[:, 8],
            z[:, 9],
            z[:, 10],
        ]
    )
    high_activity_gap = np.maximum(high_activity - benign_support - float(high_activity_margin), 0.0)
    high_activity_penalty = float(high_activity_weight) * high_activity_gap
    if float(high_activity_cap) > 0.0:
        high_activity_penalty = np.minimum(high_activity_penalty, float(high_activity_cap))
    intrinsic_weights = weights.copy()
    intrinsic_weights[5] = np.float32(0.0)
    intrinsic_weights = intrinsic_weights / max(float(np.sum(np.maximum(intrinsic_weights, 0.0))), 1e-6)
    intrinsic_rank_support = np.sum(z * intrinsic_weights.reshape(1, -1), axis=1)
    delta = support - float(penalty_weight) * prop_only_penalty - high_activity_penalty
    return (
        delta.astype(np.float32, copy=False),
        support.astype(np.float32, copy=False),
        prop_only_penalty.astype(np.float32, copy=False),
        high_activity_penalty.astype(np.float32, copy=False),
        hard_support.astype(np.float32, copy=False),
        propagation_pressure.astype(np.float32, copy=False),
        evidence_family_count.astype(np.float32, copy=False),
        intrinsic_rank_support.astype(np.float32, copy=False),
    )


def _apply_evidence_band_gate(
    node_score: np.ndarray,
    rerank_keep: np.ndarray,
    rerank_matrix: np.ndarray,
    feature_stats: list[dict[str, float]],
    feature_calibrations: list[dict[str, object]],
    adaptive_memory_bonus: np.ndarray,
    adaptive_memory_quarantine_node: np.ndarray,
    adaptive_memory_update_node: np.ndarray,
    calibration_mode: str,
    clip: float,
    candidate_local: float,
    candidate_activity: float,
    candidate_relation: float,
    min_local: float,
    min_relation: float,
    min_assoc: float,
    min_semantic: float,
    min_role: float,
    min_adaptive: float,
    min_families: int,
    weight: float,
    boost_cap: float,
    penalty_weight: float,
    penalty_cap: float,
    activity_weight: float,
    activity_margin: float,
    update_min_count: float,
    update_weight: float,
    update_cap: float,
    prop_weight: float,
    prop_cap: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    arrays = {
        "candidate": np.zeros((node_score.shape[0],), dtype=bool),
        "applied": np.zeros((node_score.shape[0],), dtype=bool),
        "pass": np.zeros((node_score.shape[0],), dtype=bool),
        "score": np.zeros((node_score.shape[0],), dtype=np.float32),
        "boost": np.zeros((node_score.shape[0],), dtype=np.float32),
        "penalty": np.zeros((node_score.shape[0],), dtype=np.float32),
        "local": np.zeros((node_score.shape[0],), dtype=np.float32),
        "relation": np.zeros((node_score.shape[0],), dtype=np.float32),
        "association": np.zeros((node_score.shape[0],), dtype=np.float32),
        "semantic": np.zeros((node_score.shape[0],), dtype=np.float32),
        "role": np.zeros((node_score.shape[0],), dtype=np.float32),
        "adaptive": np.zeros((node_score.shape[0],), dtype=np.float32),
        "activity": np.zeros((node_score.shape[0],), dtype=np.float32),
        "propagation": np.zeros((node_score.shape[0],), dtype=np.float32),
        "family_count": np.zeros((node_score.shape[0],), dtype=np.float32),
        "activity_penalty": np.zeros((node_score.shape[0],), dtype=np.float32),
        "benign_update_penalty": np.zeros((node_score.shape[0],), dtype=np.float32),
        "propagation_penalty": np.zeros((node_score.shape[0],), dtype=np.float32),
        "quarantine": np.zeros((node_score.shape[0],), dtype=np.float32),
    }
    if rerank_matrix.size <= 0 or rerank_keep.size <= 0:
        return node_score, arrays, {
            "candidate_nodes": 0,
            "applied_nodes": 0,
            "passed_nodes": 0,
            "max_boost": 0.0,
            "mean_boost": 0.0,
            "max_penalty": 0.0,
            "mean_penalty": 0.0,
            "mean_score": 0.0,
        }

    z = _calibrated_feature_matrix(
        rerank_matrix,
        feature_stats,
        feature_calibrations,
        calibration_mode,
        clip,
    )
    local_z = z[:, NODE_RERANK_FEATURES.index("node_max")]
    activity_z = np.maximum(
        z[:, NODE_RERANK_FEATURES.index("log_high_count")],
        z[:, NODE_RERANK_FEATURES.index("log_soft_sum")],
    )
    relation_z = np.maximum.reduce(
        [
            z[:, NODE_RERANK_FEATURES.index("chain_profile_bonus")],
            z[:, NODE_RERANK_FEATURES.index("benign_profile_chain_rarity")],
            z[:, NODE_RERANK_FEATURES.index("local_chain_agreement")],
            z[:, NODE_RERANK_FEATURES.index("compact_malicious_chain_signal")],
        ]
    )
    assoc_z = z[:, NODE_RERANK_FEATURES.index("association_discrepancy_max")]
    semantic_z = z[:, NODE_RERANK_FEATURES.index("semantic_rarity_stability")]
    role_z = z[:, NODE_RERANK_FEATURES.index("log_role_type_diversity")]
    prop_z = z[:, NODE_RERANK_FEATURES.index("log_prop_source_diversity")]
    adaptive_z_all = np.asarray(adaptive_memory_bonus, dtype=np.float32)
    quarantine_all = np.asarray(adaptive_memory_quarantine_node, dtype=np.float32)
    update_all = np.asarray(adaptive_memory_update_node, dtype=np.float32)
    adaptive_z = adaptive_z_all[rerank_keep]
    quarantine = quarantine_all[rerank_keep] > 0.0

    arrays["local"][rerank_keep] = local_z
    arrays["relation"][rerank_keep] = relation_z
    arrays["association"][rerank_keep] = assoc_z
    arrays["semantic"][rerank_keep] = semantic_z
    arrays["role"][rerank_keep] = role_z
    arrays["adaptive"][rerank_keep] = adaptive_z
    arrays["activity"][rerank_keep] = activity_z
    arrays["propagation"][rerank_keep] = prop_z
    arrays["quarantine"][rerank_keep] = quarantine.astype(np.float32, copy=False)

    local_pass = local_z >= float(min_local)
    relation_pass = relation_z >= float(min_relation)
    assoc_pass = assoc_z >= float(min_assoc)
    semantic_pass = semantic_z >= float(min_semantic)
    role_pass = role_z >= float(min_role)
    adaptive_pass = adaptive_z >= float(min_adaptive)
    quarantine_pass = quarantine
    family_count = (
        local_pass.astype(np.float32)
        + relation_pass.astype(np.float32)
        + assoc_pass.astype(np.float32)
        + semantic_pass.astype(np.float32)
        + role_pass.astype(np.float32)
        + adaptive_pass.astype(np.float32)
        + quarantine_pass.astype(np.float32)
    )
    candidate = (
        (local_z >= float(candidate_local))
        | (activity_z >= float(candidate_activity))
        | (relation_z >= float(candidate_relation))
        | quarantine
    )
    candidate = candidate & (
        (local_z > 0.0)
        | (relation_z > 0.0)
        | (assoc_z > 0.0)
        | (semantic_z > 0.0)
        | (adaptive_z > 0.0)
    )
    evidence_core = np.minimum.reduce(
        [
            np.maximum(local_z, 0.25 * semantic_z),
            np.maximum(relation_z, 0.25 * role_z),
            np.maximum(assoc_z, 0.25 * adaptive_z),
        ]
    )
    memory_support = np.maximum(adaptive_z, quarantine.astype(np.float32) * float(min_adaptive))
    diversity_support = np.maximum(role_z, 0.5 * semantic_z)
    evidence_score = (
        evidence_core
        + 0.25 * memory_support
        + 0.20 * diversity_support
        + 0.10 * np.minimum(activity_z, relation_z)
    )
    pass_mask = (
        candidate
        & local_pass
        & relation_pass
        & assoc_pass
        & (family_count >= float(min_families))
    )
    boost = np.where(pass_mask, float(weight) * evidence_score, 0.0)
    if float(boost_cap) > 0.0:
        boost = np.minimum(boost, float(boost_cap))

    support_gap = (
        np.maximum(float(min_local) - local_z, 0.0)
        + np.maximum(float(min_relation) - relation_z, 0.0)
        + np.maximum(float(min_assoc) - assoc_z, 0.0)
        + np.maximum(float(min_semantic) - semantic_z, 0.0)
        + np.maximum(float(min_role) - role_z, 0.0)
        + np.maximum(float(min_adaptive) - adaptive_z, 0.0)
        + np.maximum(float(min_families) - family_count, 0.0)
    )
    missing_penalty = np.where(candidate & ~pass_mask, float(penalty_weight) * support_gap, 0.0)
    compact_support = np.maximum.reduce([relation_z, assoc_z, semantic_z, role_z, adaptive_z])
    activity_gap = np.maximum(activity_z - compact_support - float(activity_margin), 0.0)
    activity_penalty = np.where(candidate, float(activity_weight) * activity_gap, 0.0)
    update_excess = np.maximum(update_all[rerank_keep] - float(update_min_count), 0.0)
    update_penalty = np.where(candidate, float(update_weight) * np.log1p(update_excess), 0.0)
    if float(update_cap) > 0.0:
        update_penalty = np.minimum(update_penalty, float(update_cap))
    prop_excess = np.maximum(prop_z - compact_support, 0.0)
    prop_penalty = np.where(candidate, float(prop_weight) * prop_excess, 0.0)
    if float(prop_cap) > 0.0:
        prop_penalty = np.minimum(prop_penalty, float(prop_cap))
    penalty = missing_penalty + activity_penalty + update_penalty + prop_penalty
    if float(penalty_cap) > 0.0:
        penalty = np.minimum(penalty, float(penalty_cap))
    delta = boost - penalty
    apply_mask = candidate & (np.abs(delta) > 0.0)
    if np.any(apply_mask):
        apply_nodes = rerank_keep[apply_mask].astype(np.int64, copy=False)
        node_score[apply_nodes] = node_score[apply_nodes] + delta[apply_mask].astype(np.float32, copy=False)
        arrays["applied"][apply_nodes] = True
    candidate_nodes = rerank_keep[candidate].astype(np.int64, copy=False)
    pass_nodes = rerank_keep[pass_mask].astype(np.int64, copy=False)
    arrays["candidate"][candidate_nodes] = True
    arrays["pass"][pass_nodes] = True
    arrays["score"][rerank_keep] = evidence_score.astype(np.float32, copy=False)
    arrays["boost"][rerank_keep] = boost.astype(np.float32, copy=False)
    arrays["penalty"][rerank_keep] = penalty.astype(np.float32, copy=False)
    arrays["family_count"][rerank_keep] = family_count.astype(np.float32, copy=False)
    arrays["activity_penalty"][rerank_keep] = activity_penalty.astype(np.float32, copy=False)
    arrays["benign_update_penalty"][rerank_keep] = update_penalty.astype(np.float32, copy=False)
    arrays["propagation_penalty"][rerank_keep] = prop_penalty.astype(np.float32, copy=False)
    applied_penalty = penalty[apply_mask]
    return node_score, arrays, {
        "candidate_nodes": int(np.sum(candidate)),
        "applied_nodes": int(np.sum(apply_mask)),
        "passed_nodes": int(np.sum(pass_mask)),
        "max_boost": float(np.max(boost)) if boost.size else 0.0,
        "mean_boost": float(np.mean(boost[boost > 0.0])) if np.any(boost > 0.0) else 0.0,
        "max_penalty": float(np.max(applied_penalty)) if applied_penalty.size else 0.0,
        "mean_penalty": float(np.mean(applied_penalty[applied_penalty > 0.0])) if np.any(applied_penalty > 0.0) else 0.0,
        "mean_score": float(np.mean(evidence_score[candidate])) if np.any(candidate) else 0.0,
    }


def _apply_mid_rank_gate(
    node_score: np.ndarray,
    ranked_before_gate: np.ndarray,
    rerank_keep: np.ndarray,
    rerank_matrix: np.ndarray,
    feature_stats: list[dict[str, float]],
    adaptive_memory_bonus: np.ndarray,
    adaptive_memory_quarantine_node: np.ndarray,
    adaptive_memory_update_node: np.ndarray,
    clip: float,
    protect_topk: int,
    end_topk: int,
    min_assoc_z: float,
    min_relation_z: float,
    min_adaptive_z: float,
    require_quarantine: bool,
    penalty_weight: float,
    penalty_cap: float,
    benign_update_min_count: float,
    benign_update_weight: float,
    benign_update_cap: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    gate_arrays = {
        "applied": np.zeros((node_score.shape[0],), dtype=bool),
        "candidate": np.zeros((node_score.shape[0],), dtype=bool),
        "penalty": np.zeros((node_score.shape[0],), dtype=np.float32),
        "assoc_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "relation_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "adaptive_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "quarantine": np.zeros((node_score.shape[0],), dtype=np.float32),
        "benign_update_penalty": np.zeros((node_score.shape[0],), dtype=np.float32),
        "pass": np.zeros((node_score.shape[0],), dtype=bool),
    }
    protect = max(int(protect_topk), 0)
    end = max(int(end_topk), protect)
    end = min(end, int(ranked_before_gate.shape[0]))
    if end <= protect or rerank_matrix.size <= 0 or rerank_keep.size <= 0:
        meta = {
            "candidate_nodes": 0,
            "applied_nodes": 0,
            "passed_nodes": 0,
            "max_penalty": 0.0,
            "mean_penalty": 0.0,
        }
        return node_score, gate_arrays, meta

    z_cols = []
    for col, stat in zip(rerank_matrix.T, feature_stats):
        z_cols.append(_upper_z_array(col, stat, clip))
    z = np.column_stack(z_cols).astype(np.float32, copy=False)
    assoc_z = z[:, NODE_RERANK_FEATURES.index("association_discrepancy_max")]
    relation_z = np.maximum.reduce(
        [
            z[:, NODE_RERANK_FEATURES.index("chain_profile_bonus")],
            z[:, NODE_RERANK_FEATURES.index("benign_profile_chain_rarity")],
            z[:, NODE_RERANK_FEATURES.index("local_chain_agreement")],
            z[:, NODE_RERANK_FEATURES.index("compact_malicious_chain_signal")],
        ]
    )
    adaptive_z_all = np.asarray(adaptive_memory_bonus, dtype=np.float32)
    quarantine_all = np.asarray(adaptive_memory_quarantine_node, dtype=np.float32)
    update_all = np.asarray(adaptive_memory_update_node, dtype=np.float32)

    gate_arrays["assoc_z"][rerank_keep] = assoc_z
    gate_arrays["relation_z"][rerank_keep] = relation_z
    gate_arrays["adaptive_z"][:] = adaptive_z_all
    gate_arrays["quarantine"][:] = quarantine_all

    candidate_nodes = ranked_before_gate[protect:end].astype(np.int64, copy=False)
    candidate_nodes = candidate_nodes[(candidate_nodes >= 0) & (candidate_nodes < node_score.shape[0])]
    gate_arrays["candidate"][candidate_nodes] = True
    rerank_pos = {int(node): idx for idx, node in enumerate(rerank_keep.tolist())}
    candidate_positions = [rerank_pos[int(node)] for node in candidate_nodes.tolist() if int(node) in rerank_pos]
    if not candidate_positions:
        meta = {
            "candidate_nodes": int(candidate_nodes.size),
            "applied_nodes": 0,
            "passed_nodes": 0,
            "max_penalty": 0.0,
            "mean_penalty": 0.0,
        }
        return node_score, gate_arrays, meta

    pos = np.asarray(candidate_positions, dtype=np.int64)
    nodes = rerank_keep[pos].astype(np.int64, copy=False)
    node_assoc = assoc_z[pos]
    node_relation = relation_z[pos]
    node_adaptive = adaptive_z_all[nodes]
    node_quarantine = quarantine_all[nodes] > 0.0
    assoc_gap = np.maximum(float(min_assoc_z) - node_assoc, 0.0)
    relation_gap = np.maximum(float(min_relation_z) - node_relation, 0.0)
    adaptive_gap = np.maximum(float(min_adaptive_z) - node_adaptive, 0.0)
    quarantine_gap = (
        (~node_quarantine).astype(np.float32)
        if bool(require_quarantine)
        else np.zeros((nodes.shape[0],), dtype=np.float32)
    )
    support_pass = (
        (node_assoc >= float(min_assoc_z))
        & (node_relation >= float(min_relation_z))
        & (node_adaptive >= float(min_adaptive_z))
    )
    if bool(require_quarantine):
        support_pass = support_pass & node_quarantine
    update_excess = np.maximum(update_all[nodes] - float(benign_update_min_count), 0.0)
    update_penalty = float(benign_update_weight) * np.log1p(update_excess)
    if float(benign_update_cap) > 0.0:
        update_penalty = np.minimum(update_penalty, float(benign_update_cap))
    missing_penalty = float(penalty_weight) * (assoc_gap + relation_gap + adaptive_gap + quarantine_gap)
    penalty = missing_penalty + update_penalty
    if float(penalty_cap) > 0.0:
        penalty = np.minimum(penalty, float(penalty_cap))
    apply_mask = penalty > 0.0
    if np.any(apply_mask):
        apply_nodes = nodes[apply_mask]
        node_score[apply_nodes] = node_score[apply_nodes] - penalty[apply_mask].astype(np.float32, copy=False)
        gate_arrays["applied"][apply_nodes] = True
        gate_arrays["penalty"][apply_nodes] = penalty[apply_mask].astype(np.float32, copy=False)
        gate_arrays["benign_update_penalty"][apply_nodes] = update_penalty[apply_mask].astype(np.float32, copy=False)
    gate_arrays["pass"][nodes[support_pass]] = True
    meta = {
        "candidate_nodes": int(nodes.size),
        "applied_nodes": int(np.sum(apply_mask)),
        "passed_nodes": int(np.sum(support_pass)),
        "max_penalty": float(np.max(penalty[apply_mask])) if np.any(apply_mask) else 0.0,
        "mean_penalty": float(np.mean(penalty[apply_mask])) if np.any(apply_mask) else 0.0,
    }
    return node_score, gate_arrays, meta


def _apply_local_closure_score(
    node_score: np.ndarray,
    ranked_before_closure: np.ndarray,
    rerank_keep: np.ndarray,
    rerank_matrix: np.ndarray,
    feature_stats: list[dict[str, float]],
    adaptive_memory_bonus: np.ndarray,
    adaptive_memory_quarantine_node: np.ndarray,
    adaptive_memory_update_node: np.ndarray,
    clip: float,
    protect_topk: int,
    end_topk: int,
    min_score_z: float,
    min_families: int,
    min_assoc_z: float,
    min_role_z: float,
    min_adaptive_z: float,
    weight: float,
    boost_cap: float,
    penalty_weight: float,
    penalty_cap: float,
    prop_penalty_weight: float,
    prop_penalty_cap: float,
    update_min_count: float,
    update_weight: float,
    update_cap: float,
    require_quarantine: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    arrays = {
        "candidate": np.zeros((node_score.shape[0],), dtype=bool),
        "applied": np.zeros((node_score.shape[0],), dtype=bool),
        "pass": np.zeros((node_score.shape[0],), dtype=bool),
        "score": np.zeros((node_score.shape[0],), dtype=np.float32),
        "boost": np.zeros((node_score.shape[0],), dtype=np.float32),
        "penalty": np.zeros((node_score.shape[0],), dtype=np.float32),
        "local_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "relation_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "association_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "adaptive_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "role_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "role_type_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "propagation_z": np.zeros((node_score.shape[0],), dtype=np.float32),
        "family_count": np.zeros((node_score.shape[0],), dtype=np.float32),
        "benign_update_penalty": np.zeros((node_score.shape[0],), dtype=np.float32),
        "propagation_penalty": np.zeros((node_score.shape[0],), dtype=np.float32),
        "hard_assoc_pass": np.zeros((node_score.shape[0],), dtype=bool),
        "hard_role_pass": np.zeros((node_score.shape[0],), dtype=bool),
    }
    protect = max(int(protect_topk), 0)
    end = max(int(end_topk), protect)
    end = min(end, int(ranked_before_closure.shape[0]))
    if end <= protect or rerank_matrix.size <= 0 or rerank_keep.size <= 0:
        meta = {
            "candidate_nodes": 0,
            "applied_nodes": 0,
            "passed_nodes": 0,
            "max_boost": 0.0,
            "mean_boost": 0.0,
            "max_penalty": 0.0,
            "mean_score": 0.0,
        }
        return node_score, arrays, meta

    z_cols = [_upper_z_array(col, stat, clip) for col, stat in zip(rerank_matrix.T, feature_stats)]
    z = np.column_stack(z_cols).astype(np.float32, copy=False)
    local_z = z[:, NODE_RERANK_FEATURES.index("node_max")]
    relation_z = np.maximum.reduce(
        [
            z[:, NODE_RERANK_FEATURES.index("chain_profile_bonus")],
            z[:, NODE_RERANK_FEATURES.index("benign_profile_chain_rarity")],
            z[:, NODE_RERANK_FEATURES.index("local_chain_agreement")],
            z[:, NODE_RERANK_FEATURES.index("compact_malicious_chain_signal")],
        ]
    )
    assoc_z = z[:, NODE_RERANK_FEATURES.index("association_discrepancy_max")]
    semantic_z = z[:, NODE_RERANK_FEATURES.index("semantic_rarity_stability")]
    prop_z = z[:, NODE_RERANK_FEATURES.index("log_prop_source_diversity")]
    role_type_z = z[:, NODE_RERANK_FEATURES.index("log_role_type_diversity")]
    compact_chain_z = z[:, NODE_RERANK_FEATURES.index("compact_malicious_chain_signal")]
    role_z = np.maximum(
        role_type_z,
        0.5 * compact_chain_z,
    )
    adaptive_z_all = np.asarray(adaptive_memory_bonus, dtype=np.float32)
    quarantine_all = np.asarray(adaptive_memory_quarantine_node, dtype=np.float32)
    update_all = np.asarray(adaptive_memory_update_node, dtype=np.float32)

    arrays["local_z"][rerank_keep] = local_z
    arrays["relation_z"][rerank_keep] = relation_z
    arrays["association_z"][rerank_keep] = assoc_z
    arrays["adaptive_z"][:] = adaptive_z_all
    arrays["role_z"][rerank_keep] = role_z
    arrays["role_type_z"][rerank_keep] = role_type_z
    arrays["propagation_z"][rerank_keep] = prop_z

    candidate_nodes = ranked_before_closure[protect:end].astype(np.int64, copy=False)
    candidate_nodes = candidate_nodes[(candidate_nodes >= 0) & (candidate_nodes < node_score.shape[0])]
    arrays["candidate"][candidate_nodes] = True
    rerank_pos = {int(node): idx for idx, node in enumerate(rerank_keep.tolist())}
    candidate_positions = [rerank_pos[int(node)] for node in candidate_nodes.tolist() if int(node) in rerank_pos]
    if not candidate_positions:
        meta = {
            "candidate_nodes": int(candidate_nodes.size),
            "applied_nodes": 0,
            "passed_nodes": 0,
            "max_boost": 0.0,
            "mean_boost": 0.0,
            "max_penalty": 0.0,
            "mean_score": 0.0,
        }
        return node_score, arrays, meta

    pos = np.asarray(candidate_positions, dtype=np.int64)
    nodes = rerank_keep[pos].astype(np.int64, copy=False)
    node_local = local_z[pos]
    node_relation = relation_z[pos]
    node_assoc = assoc_z[pos]
    node_semantic = semantic_z[pos]
    node_role = role_z[pos]
    node_role_type = role_type_z[pos]
    node_prop = prop_z[pos]
    node_adaptive = adaptive_z_all[nodes]
    node_quarantine = quarantine_all[nodes] > 0.0
    hard_assoc = node_assoc >= float(min_assoc_z)
    hard_role = node_role_type >= float(min_role_z)
    hard_adaptive = node_adaptive >= float(min_adaptive_z)
    family_count = (
        (node_local >= float(min_score_z)).astype(np.float32)
        + (node_relation >= float(min_score_z)).astype(np.float32)
        + hard_assoc.astype(np.float32)
        + hard_adaptive.astype(np.float32)
        + hard_role.astype(np.float32)
        + (node_semantic >= float(min_score_z)).astype(np.float32)
        + (node_quarantine.astype(np.float32) if bool(require_quarantine) else np.ones_like(node_local, dtype=np.float32))
    )
    closure_core = np.minimum.reduce([node_local, node_relation, node_assoc])
    memory_support = np.maximum(node_adaptive, node_quarantine.astype(np.float32) * float(min_score_z))
    role_support = np.maximum(node_role_type, 0.5 * node_semantic)
    raw_score = closure_core + 0.35 * memory_support + 0.25 * role_support
    update_excess = np.maximum(update_all[nodes] - float(update_min_count), 0.0)
    update_penalty = float(update_weight) * np.log1p(update_excess)
    if float(update_cap) > 0.0:
        update_penalty = np.minimum(update_penalty, float(update_cap))
    intrinsic_support = np.maximum.reduce([node_local, node_relation, node_assoc, node_role_type, node_adaptive])
    prop_excess = np.maximum(node_prop - intrinsic_support, 0.0)
    prop_penalty = float(prop_penalty_weight) * prop_excess
    if float(prop_penalty_cap) > 0.0:
        prop_penalty = np.minimum(prop_penalty, float(prop_penalty_cap))
    closure_score = np.maximum(raw_score - update_penalty - prop_penalty, 0.0)
    passed = (
        (closure_score >= float(min_score_z))
        & (family_count >= float(min_families))
        & hard_assoc
        & hard_role
        & hard_adaptive
    )
    if bool(require_quarantine):
        passed = passed & node_quarantine
    boost = np.where(passed, float(weight) * closure_score, 0.0)
    if float(boost_cap) > 0.0:
        boost = np.minimum(boost, float(boost_cap))
    support_gap = (
        np.maximum(float(min_score_z) - closure_score, 0.0)
        + np.maximum(float(min_families) - family_count, 0.0)
        + np.maximum(float(min_assoc_z) - node_assoc, 0.0)
        + np.maximum(float(min_role_z) - node_role_type, 0.0)
        + np.maximum(float(min_adaptive_z) - node_adaptive, 0.0)
    )
    penalty = np.where(~passed, float(penalty_weight) * support_gap, 0.0)
    if float(penalty_cap) > 0.0:
        penalty = np.minimum(penalty, float(penalty_cap))
    delta = boost - penalty
    apply_mask = np.abs(delta) > 0.0
    if np.any(apply_mask):
        apply_nodes = nodes[apply_mask]
        node_score[apply_nodes] = node_score[apply_nodes] + delta[apply_mask].astype(np.float32, copy=False)
        arrays["applied"][apply_nodes] = True
    arrays["pass"][nodes[passed]] = True
    arrays["score"][nodes] = closure_score.astype(np.float32, copy=False)
    arrays["boost"][nodes] = boost.astype(np.float32, copy=False)
    arrays["penalty"][nodes] = penalty.astype(np.float32, copy=False)
    arrays["family_count"][nodes] = family_count.astype(np.float32, copy=False)
    arrays["benign_update_penalty"][nodes] = update_penalty.astype(np.float32, copy=False)
    arrays["propagation_penalty"][nodes] = prop_penalty.astype(np.float32, copy=False)
    arrays["hard_assoc_pass"][nodes] = hard_assoc
    arrays["hard_role_pass"][nodes] = hard_role
    meta = {
        "candidate_nodes": int(nodes.size),
        "applied_nodes": int(np.sum(apply_mask)),
        "passed_nodes": int(np.sum(passed)),
        "hard_assoc_passed_nodes": int(np.sum(hard_assoc)),
        "hard_role_passed_nodes": int(np.sum(hard_role)),
        "max_boost": float(np.max(boost)) if boost.size else 0.0,
        "mean_boost": float(np.mean(boost[boost > 0.0])) if np.any(boost > 0.0) else 0.0,
        "max_penalty": float(np.max(penalty)) if penalty.size else 0.0,
        "max_propagation_penalty": float(np.max(prop_penalty)) if prop_penalty.size else 0.0,
        "mean_propagation_penalty": float(np.mean(prop_penalty[prop_penalty > 0.0])) if np.any(prop_penalty > 0.0) else 0.0,
        "mean_score": float(np.mean(closure_score)) if closure_score.size else 0.0,
    }
    return node_score, arrays, meta


def _node_sweep(
    node_score: np.ndarray,
    all_nodes: np.ndarray,
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    topk_values: list[int],
) -> list[dict[str, object]]:
    ranked = rank_scores(node_score)
    sweep = []
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


def _with_f1(metrics: dict[str, object]) -> dict[str, object]:
    out = dict(metrics)
    precision = float(out.get("precision", 0.0))
    recall = float(out.get("recall", 0.0))
    out["f1"] = float((2.0 * precision * recall) / max(precision + recall, 1e-12))
    return out


def _numeric_summary(values: list[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def _online_alert_prefix_sweep(
    alerts: list[dict[str, int | float | str]],
    max_node_id: int,
    all_nodes: np.ndarray,
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    topk_values: list[int],
) -> list[dict[str, object]]:
    if not alerts:
        return []
    prefix_values = sorted({int(k) for k in topk_values if int(k) > 0} | {len(alerts)})
    sweep: list[dict[str, object]] = []
    for topk in prefix_values:
        alerted = np.zeros((max_node_id,), dtype=bool)
        for alert in alerts[: min(int(topk), len(alerts))]:
            node = int(alert["node_id"])
            if 0 <= node < max_node_id:
                alerted[node] = True
        sweep.append(
            {
                "topk": int(topk),
                "prefix_alerts": int(min(int(topk), len(alerts))),
                "num_alerted_nodes": int(np.sum(alerted)),
                "strict_node": _with_f1(node_confusion_from_masks(all_nodes, positive_nodes, suspect_nodes, alerted, relaxed=False)),
                "relaxed_node": _with_f1(node_confusion_from_masks(all_nodes, positive_nodes, suspect_nodes, alerted, relaxed=True)),
            }
        )
    return sweep


def _online_alert_performance(
    online_alert_emitter: NodeAlertEmitter | None,
    max_node_id: int,
    all_nodes: np.ndarray,
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    positive_node_first_event: dict[int, int],
    positive_node_first_ts: dict[int, int],
    topk_values: list[int],
    target: dict[str, int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if online_alert_emitter is None:
        return (
            {
                "enabled": False,
                "role": "real_time_stopping_time_node_alerting",
                "definition": "Disabled for this run. Enable --online_node_alerts_enabled to emit and evaluate real-time node alerts.",
            },
            [],
        )
    alerted = np.asarray(online_alert_emitter.alerted, dtype=bool)
    alerts = list(online_alert_emitter.alerts)
    strict = _with_f1(node_confusion_from_masks(all_nodes, positive_nodes, suspect_nodes, alerted, relaxed=False))
    relaxed = _with_f1(node_confusion_from_masks(all_nodes, positive_nodes, suspect_nodes, alerted, relaxed=True))
    prefix_sweep = _online_alert_prefix_sweep(alerts, max_node_id, all_nodes, positive_nodes, suspect_nodes, topk_values)
    tp_delay_events: list[float] = []
    tp_delay_seconds: list[float] = []
    first_tp_alert: dict[str, object] = {}
    relaxed_fp_alerts = 0
    ignored_suspect_alerts = 0
    for alert in alerts:
        node = int(alert["node_id"])
        if node < 0 or node >= max_node_id:
            continue
        is_positive = bool(positive_nodes[node])
        is_ignored_suspect = bool(suspect_nodes[node] and not positive_nodes[node])
        if is_positive:
            first_event = int(positive_node_first_event.get(node, -1))
            first_ts = int(positive_node_first_ts.get(node, -1))
            delay_events = int(alert["first_trigger_event_id"]) - first_event if first_event >= 0 else None
            delay_seconds = (int(alert["timestamp"]) - first_ts) / 1e9 if first_ts >= 0 else None
            if delay_events is not None:
                tp_delay_events.append(float(delay_events))
            if delay_seconds is not None:
                tp_delay_seconds.append(float(delay_seconds))
            if not first_tp_alert:
                first_tp_alert = {
                    "alert_order": int(alert["alert_order"]),
                    "node_id": node,
                    "timestamp": int(alert["timestamp"]),
                    "first_trigger_event_id": int(alert["first_trigger_event_id"]),
                    "first_observed_positive_event_id": first_event,
                    "delay_events": delay_events,
                    "delay_seconds": delay_seconds,
                    "risk_score": float(alert["risk_score"]),
                }
        elif is_ignored_suspect:
            ignored_suspect_alerts += 1
        elif bool(all_nodes[node]):
            relaxed_fp_alerts += 1
    best_prefix = best_under_fp_target(prefix_sweep, target)
    return (
        {
            "enabled": True,
            "role": "real_time_stopping_time_node_alerting",
            "num_alerts": int(len(alerts)),
            "num_alerted_nodes": int(np.sum(alerted)),
            "strict_node": strict,
            "relaxed_node": relaxed,
            "relaxed_fp_alerts": int(relaxed_fp_alerts),
            "ignored_suspect_alerts": int(ignored_suspect_alerts),
            "first_tp_alert": first_tp_alert,
            "detection_delay_events": _numeric_summary(tp_delay_events),
            "detection_delay_seconds": _numeric_summary(tp_delay_seconds),
            "prefix_sweep": prefix_sweep,
            "best_prefix_under_fp_target": best_prefix,
            "prefix_sweep_role": "Diagnostic only. The online system emits by the fixed threshold; these rows evaluate the first N emitted alerts without changing the threshold.",
            "delay_definition": "Delay is measured from the first scored test event where an observed GT abnormal node appears as an endpoint to that node's first online alert.",
            "leakage_check": "GT labels are used only here, after online alerts have already been emitted and the test stream has finished.",
        },
        prefix_sweep,
    )


def _write_online_alert_prefix_sweep_csv(path: str, sweep: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "topk",
        "prefix_alerts",
        "num_alerted_nodes",
        "strict_tp",
        "strict_fp",
        "strict_fn",
        "strict_precision",
        "strict_recall",
        "strict_f1",
        "relaxed_tp",
        "relaxed_fp",
        "relaxed_fn",
        "relaxed_precision",
        "relaxed_recall",
        "relaxed_f1",
        "relaxed_ignored_suspect_nodes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sweep:
            strict = row.get("strict_node", {})
            relaxed = row.get("relaxed_node", {})
            assert isinstance(strict, dict)
            assert isinstance(relaxed, dict)
            writer.writerow(
                {
                    "topk": int(row.get("topk", 0)),
                    "prefix_alerts": int(row.get("prefix_alerts", 0)),
                    "num_alerted_nodes": int(row.get("num_alerted_nodes", 0)),
                    "strict_tp": int(strict.get("tp", 0)),
                    "strict_fp": int(strict.get("fp", 0)),
                    "strict_fn": int(strict.get("fn", 0)),
                    "strict_precision": float(strict.get("precision", 0.0)),
                    "strict_recall": float(strict.get("recall", 0.0)),
                    "strict_f1": float(strict.get("f1", 0.0)),
                    "relaxed_tp": int(relaxed.get("tp", 0)),
                    "relaxed_fp": int(relaxed.get("fp", 0)),
                    "relaxed_fn": int(relaxed.get("fn", 0)),
                    "relaxed_precision": float(relaxed.get("precision", 0.0)),
                    "relaxed_recall": float(relaxed.get("recall", 0.0)),
                    "relaxed_f1": float(relaxed.get("f1", 0.0)),
                    "relaxed_ignored_suspect_nodes": int(relaxed.get("num_ignored_suspect_nodes", 0)),
                }
            )


def _bytes_mb(value: int) -> float:
    return float(int(value) / (1024.0 * 1024.0))


def _arrays_nbytes(*arrays: np.ndarray) -> int:
    return int(sum(int(np.asarray(arr).nbytes) for arr in arrays))


def _unique_arrays_nbytes(*arrays: np.ndarray) -> int:
    seen: set[int] = set()
    total = 0
    for arr in arrays:
        view = np.asarray(arr)
        key = id(view)
        if key in seen:
            continue
        seen.add(key)
        total += int(view.nbytes)
    return int(total)


def _array_map_nbytes(arrays: dict[str, np.ndarray]) -> int:
    return int(sum(int(np.asarray(arr).nbytes) for arr in arrays.values()))


def _write_memory_samples_csv(path: str, samples: list[dict[str, float | int | str]]) -> None:
    if not samples:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = sorted({str(key) for sample in samples for key in sample.keys()})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)


def _best_node_evidence_summary(
    ranked: np.ndarray,
    best: dict[str, object],
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    node_top_label: np.ndarray,
    node_top_component: np.ndarray,
) -> dict[str, object]:
    if not best:
        return {}
    keep = ranked[: min(int(best["topk"]), ranked.shape[0])]
    positive_mask = positive_nodes[keep]
    suspect_mask = suspect_nodes[keep] & ~positive_mask
    negative_mask = ~positive_mask & ~suspect_mask
    component_names = np.asarray(COMPONENTS + EXTRA_COMPONENTS, dtype=object)

    def comp_dist(indices: np.ndarray) -> dict[str, int]:
        out = {str(name): 0 for name in component_names.tolist()}
        for raw in node_top_component[indices].tolist():
            idx = int(raw)
            if 0 <= idx < component_names.shape[0]:
                out[str(component_names[idx])] += 1
        return out

    return {
        "topk": int(best["topk"]),
        "num_alerted_nodes": int(keep.shape[0]),
        "note": "top_event_label is the label of the event producing each node's max leakage-free low-rank local score.",
        "all_top_event_label_distribution": label_distribution(node_top_label[keep]),
        "tp_top_event_label_distribution": label_distribution(node_top_label[keep][positive_mask]),
        "suspect_only_top_event_label_distribution": label_distribution(node_top_label[keep][suspect_mask]),
        "negative_top_event_label_distribution": label_distribution(node_top_label[keep][negative_mask]),
        "all_top_component_distribution": comp_dist(keep),
        "tp_top_component_distribution": comp_dist(keep[positive_mask]),
        "suspect_only_top_component_distribution": comp_dist(keep[suspect_mask]),
        "negative_top_component_distribution": comp_dist(keep[negative_mask]),
    }


def _component_values(aux: dict[str, float]) -> list[float]:
    return [float(aux[name]) for name in COMPONENTS]


def _component_z_values(aux: dict[str, float], stats: list[dict[str, float]], weights: list[float], clip: float) -> list[float]:
    values = _component_values(aux)
    denom = max(sum(w for w in weights if w > 0.0), 1e-12)
    out = []
    for value, stat, weight in zip(values, stats, weights):
        out.append(float(weight) * _upper_z_scalar(float(value), stat, clip) / denom)
    return out


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    if bool(args.node_hard_gate_enabled) and not bool(args.node_rerank_enabled):
        raise ValueError("--node_hard_gate_enabled requires --node_rerank_enabled")
    if bool(args.evidence_band_gate_enabled) and not bool(args.node_rerank_enabled):
        raise ValueError("--evidence_band_gate_enabled requires --node_rerank_enabled")
    if bool(args.evidence_band_gate_enabled) and not bool(args.adaptive_memory_enabled):
        raise ValueError("--evidence_band_gate_enabled requires --adaptive_memory_enabled")
    if bool(args.mid_rank_gate_enabled) and not bool(args.node_rerank_enabled):
        raise ValueError("--mid_rank_gate_enabled requires --node_rerank_enabled")
    if bool(args.mid_rank_gate_enabled) and not bool(args.adaptive_memory_enabled):
        raise ValueError("--mid_rank_gate_enabled requires --adaptive_memory_enabled")
    if bool(args.mid_rank_gate_enabled) and int(args.mid_rank_gate_end_topk) <= int(args.mid_rank_gate_protect_topk):
        raise ValueError("--mid_rank_gate_end_topk must be larger than --mid_rank_gate_protect_topk")
    if bool(args.local_closure_enabled) and not bool(args.node_rerank_enabled):
        raise ValueError("--local_closure_enabled requires --node_rerank_enabled")
    if bool(args.local_closure_enabled) and not bool(args.adaptive_memory_enabled):
        raise ValueError("--local_closure_enabled requires --adaptive_memory_enabled")
    if bool(args.local_closure_enabled) and int(args.local_closure_end_topk) <= int(args.local_closure_protect_topk):
        raise ValueError("--local_closure_end_topk must be larger than --local_closure_protect_topk")
    if bool(args.snc_enabled) and not bool(args.node_rerank_enabled):
        raise ValueError("--snc_enabled requires --node_rerank_enabled")
    if bool(args.snc_enabled) and not bool(args.chain_profile_enabled):
        raise ValueError("--snc_enabled requires --chain_profile_enabled")
    if bool(args.compact_malicious_chain_enabled) and not bool(args.node_rerank_enabled):
        raise ValueError("--compact_malicious_chain_enabled requires --node_rerank_enabled")
    if bool(args.compact_malicious_chain_enabled) and not bool(args.chain_profile_enabled):
        raise ValueError("--compact_malicious_chain_enabled requires --chain_profile_enabled")
    if bool(args.causal_episode_support_enabled) and not bool(args.chain_profile_enabled):
        raise ValueError("--causal_episode_support_enabled requires --chain_profile_enabled")
    if bool(args.adaptive_memory_enabled) and not bool(args.chain_profile_enabled):
        raise ValueError("--adaptive_memory_enabled requires --chain_profile_enabled")
    if bool(args.activity_normalized_chain_enabled) and not bool(args.chain_profile_enabled):
        raise ValueError("--activity_normalized_chain_enabled requires --chain_profile_enabled")
    if bool(args.activity_normalized_chain_enabled) and int(args.activity_normalized_chain_end_topk) <= int(args.activity_normalized_chain_start_topk):
        raise ValueError("--activity_normalized_chain_end_topk must be larger than --activity_normalized_chain_start_topk")
    if float(args.risk_decay) >= 0.0:
        args.online_node_risk_decay_sec = float(args.risk_decay)
    if int(args.edge_sketch_size) > 0:
        args.identity_buckets = int(args.edge_sketch_size)
    if int(args.chain_sketch_size) > 0:
        args.chain_profile_buckets = int(args.chain_sketch_size)
    if int(args.memory_prototype_count) > 0:
        args.adaptive_memory_buckets = int(args.memory_prototype_count)
    run_start = time.perf_counter()
    memory_samples: list[dict[str, float | int | str]] = []
    result_dir = args.result_dir or str(REPO_ROOT / "outputs" / "results" / "tflr_light" / args.out_tag)
    os.makedirs(result_dir, exist_ok=True)
    cfg = _cfg_for_dataset(args.dataset)
    conn_cur, conn = init_database_connection(cfg)
    try:
        phase_start = time.perf_counter()
        log("[TFLR-LowRank] loading node tables")
        netflow_nodes, process_nodes, file_nodes = fetch_node_tables(conn_cur)
        indexid2summary = _build_index_summaries(get_indexid2msg(conn_cur))
        hash2type = build_hash_to_type(netflow_nodes, process_nodes, file_nodes)
        hash2uuid_index = build_hash_uuid_index_map(netflow_nodes, process_nodes, file_nodes)
        uuid2index = build_uuid_index_map(netflow_nodes, process_nodes, file_nodes)
        max_node_id = max(int(v[1]) for v in hash2uuid_index.values() if v[1] is not None) + 1
        index_to_type = {
            int(index_id): str(hash2type.get(str(node_hash), "unknown"))
            for node_hash, (_uuid, index_id) in hash2uuid_index.items()
            if index_id is not None
        }
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
        log(f"[TFLR-LowRank] split query counts: {counts_meta}")

        weights = [float(x.strip()) for x in str(args.weights).split(",") if x.strip()]
        if len(weights) != len(COMPONENTS):
            raise ValueError(f"--weights must contain {len(COMPONENTS)} values")
        node_rerank_feature_weights = np.asarray(
            [float(x.strip()) for x in str(args.node_rerank_feature_weights).split(",") if x.strip()],
            dtype=np.float32,
        )
        if node_rerank_feature_weights.shape[0] == 10 and len(NODE_RERANK_FEATURES) == 11:
            # Legacy weights before role-diversity were introduced.
            node_rerank_feature_weights = np.concatenate(
                [
                    node_rerank_feature_weights[:6],
                    np.asarray([0.30], dtype=np.float32),
                    node_rerank_feature_weights[6:],
                ]
            )
        if node_rerank_feature_weights.shape[0] != len(NODE_RERANK_FEATURES):
            raise ValueError(f"--node_rerank_feature_weights must contain {len(NODE_RERANK_FEATURES)} values")

        precompute_fingerprint_payload = {
            "version": PRECOMPUTE_CACHE_SCHEMA_VERSION,
            "dataset": str(args.dataset),
            "year_month": str(year_month),
            "train_days": train_days,
            "val_days": val_days,
            "event_filter": bool(event_filter),
            "max_train_events": int(args.max_train_events),
            "max_ref_events": int(args.max_ref_events),
            "latent_dim": int(args.latent_dim),
            "rank": int(args.rank),
            "semantic_buckets": int(args.semantic_buckets),
            "identity_buckets": int(args.identity_buckets),
            "max_tokens": int(args.max_tokens),
            "include_bigrams": bool(args.include_bigrams),
            "max_train_samples": int(args.max_train_samples),
            "ridge_lambda": float(args.ridge_lambda),
            "direct_lowrank_enabled": bool(args.direct_lowrank_enabled),
            "direct_lowrank_iters": int(args.direct_lowrank_iters),
            "direct_lowrank_l2": float(args.direct_lowrank_l2),
            "weights": [float(x) for x in weights],
            "score_clip": float(args.score_clip),
            "event_threshold_quantile": float(args.event_threshold_quantile),
            "prop_threshold_quantile": float(args.prop_threshold_quantile),
            "chain_profile_enabled": bool(args.chain_profile_enabled),
            "chain_profile_buckets": int(args.chain_profile_buckets),
            "chain_profile_require_transition": bool(args.chain_profile_require_transition),
            "chain_profile_score_mode": str(args.chain_profile_score_mode),
            "chain_profile_calibration_knots": int(args.chain_profile_calibration_knots),
            "chain_profile_memory_slots": int(args.chain_profile_memory_slots),
            "chain_profile_association_weight": float(args.chain_profile_association_weight),
            "adaptive_memory_enabled": bool(args.adaptive_memory_enabled),
            "adaptive_memory_buckets": int(args.adaptive_memory_buckets),
            "adaptive_memory_min_count": int(args.adaptive_memory_min_count),
            "adaptive_memory_fit_alpha": float(args.adaptive_memory_fit_alpha),
            "adaptive_memory_update_alpha": float(args.adaptive_memory_update_alpha),
            "adaptive_memory_update_tail_max": float(args.adaptive_memory_update_tail_max),
            "adaptive_memory_update_assoc_z_max": float(args.adaptive_memory_update_assoc_z_max),
            "adaptive_memory_quarantine_tail_min": float(args.adaptive_memory_quarantine_tail_min),
            "adaptive_memory_quarantine_assoc_z_min": float(args.adaptive_memory_quarantine_assoc_z_min),
            "node_rerank_enabled": bool(args.node_rerank_enabled),
            "node_rerank_calibration_mode": str(args.node_rerank_calibration_mode),
            "node_rerank_calibration_knots": int(args.node_rerank_calibration_knots),
            "online_node_alerts_enabled": bool(args.online_node_alerts_enabled),
            "online_node_alert_threshold": float(args.online_node_alert_threshold),
            "online_node_alert_threshold_quantile": float(args.online_node_alert_threshold_quantile),
            "online_node_risk_decay_sec": float(args.online_node_risk_decay_sec),
            "online_node_activity_weight": float(args.online_node_activity_weight),
            "online_node_activity_margin": float(args.online_node_activity_margin),
            "online_node_activity_cap": float(args.online_node_activity_cap),
            "causal_episode_support_enabled": bool(args.causal_episode_support_enabled),
            "causal_episode_selective_enabled": bool(args.causal_episode_selective_enabled),
            "causal_episode_min_event_tail": float(args.causal_episode_min_event_tail),
            "causal_episode_min_chain": float(args.causal_episode_min_chain),
            "causal_episode_min_assoc": float(args.causal_episode_min_assoc),
            "causal_episode_min_families": int(args.causal_episode_min_families),
            "compact_malicious_chain_enabled": bool(args.compact_malicious_chain_enabled),
            "snc_enabled": bool(args.snc_enabled),
        }
        if bool(args.activity_normalized_chain_enabled):
            precompute_fingerprint_payload.update(
                {
                    "activity_normalized_chain_enabled": True,
                    "activity_normalized_chain_calibration_mode": str(args.activity_normalized_chain_calibration_mode),
                }
            )
        precompute_cache_fingerprint = _stable_config_fingerprint(precompute_fingerprint_payload)
        precompute_cache_enabled = bool(args.precompute_cache_enabled) and str(args.precompute_cache_mode) != "off"
        precompute_cache_path = str(args.precompute_cache_path).strip() or _default_precompute_cache_path(
            result_dir,
            str(args.dataset),
            precompute_cache_fingerprint,
        )
        precompute_cache_loaded = False
        precompute_cache_written = False
        precompute_cache_error = ""
        precompute_payload: dict[str, object] | None = None
        if precompute_cache_enabled and str(args.precompute_cache_mode) in {"auto", "read"} and os.path.exists(precompute_cache_path):
            try:
                candidate_payload = _read_pickle_cache(precompute_cache_path)
                if str(candidate_payload.get("fingerprint", "")) != precompute_cache_fingerprint:
                    raise ValueError("cache fingerprint mismatch")
                precompute_payload = candidate_payload
                precompute_cache_loaded = True
                log(f"[TFLR-LowRank] loaded train/validation precompute cache: {precompute_cache_path}")
            except Exception as exc:
                precompute_cache_error = f"{type(exc).__name__}: {exc}"
                if str(args.precompute_cache_mode) == "read" or bool(args.precompute_cache_strict):
                    raise
                log(f"[TFLR-LowRank] ignoring unusable precompute cache: {precompute_cache_error}")

        model = LowRankStreamModel(
            LowRankConfig(
                latent_dim=int(args.latent_dim),
                rank=int(args.rank),
                semantic_buckets=int(args.semantic_buckets),
                identity_buckets=int(args.identity_buckets),
                max_tokens=int(args.max_tokens),
                include_bigrams=bool(args.include_bigrams),
                vector_cache_size=int(args.vector_cache_size),
                max_train_samples=int(args.max_train_samples),
                ridge_lambda=float(args.ridge_lambda),
                direct_lowrank_enabled=bool(args.direct_lowrank_enabled),
                direct_lowrank_iters=int(args.direct_lowrank_iters),
                direct_lowrank_l2=float(args.direct_lowrank_l2),
            )
        )
        chain_profile_enabled = bool(args.chain_profile_enabled)
        chain_profile = (
            BenignChainProfile(
                ChainProfileConfig(
                    num_buckets=int(args.chain_profile_buckets),
                    require_transition=bool(args.chain_profile_require_transition),
                    memory_slots=int(args.chain_profile_memory_slots),
                    association_weight=float(args.chain_profile_association_weight),
                )
            )
            if chain_profile_enabled
            else None
        )

        def train_rows_with_optional_chain_profile(rows):
            for train_row in rows:
                if chain_profile is not None:
                    chain_profile.profile_row(train_row)
                yield train_row

        if precompute_payload is not None:
            phase_start = time.perf_counter()
            model = precompute_payload["model"]  # type: ignore[assignment]
            chain_profile = precompute_payload.get("chain_profile")  # type: ignore[assignment]
            train_summary = dict(precompute_payload.get("train_summary", {}))
            train_seconds = 0.0
            comp_stats = list(precompute_payload["comp_stats"])  # type: ignore[arg-type]
            chain_profile_stats = dict(precompute_payload["chain_profile_stats"])  # type: ignore[arg-type]
            chain_assoc_stats = dict(precompute_payload["chain_assoc_stats"])  # type: ignore[arg-type]
            fused_calibration = dict(precompute_payload["fused_calibration"])  # type: ignore[arg-type]
            chain_profile_calibration = dict(precompute_payload["chain_profile_calibration"])  # type: ignore[arg-type]
            threshold = float(precompute_payload["threshold"])
            prop_threshold = float(precompute_payload["prop_threshold"])
            online_node_alert_threshold = float(precompute_payload["online_node_alert_threshold"])
            online_node_alert_threshold_source = str(precompute_payload["online_node_alert_threshold_source"])
            ref_processed = int(precompute_payload.get("ref_processed", 0))
            adaptive_memory_enabled = bool(args.adaptive_memory_enabled)
            adaptive_memory = _copy_adaptive_memory(precompute_payload["adaptive_memory"])  # type: ignore[arg-type]
            adaptive_memory_ref_distances = [float(x) for x in precompute_payload.get("adaptive_memory_ref_distances", [])]  # type: ignore[union-attr]
            adaptive_train_processed = int(precompute_payload.get("adaptive_train_processed", 0))
            adaptive_train_updates = int(precompute_payload.get("adaptive_train_updates", 0))
            adaptive_train_quarantined = int(precompute_payload.get("adaptive_train_quarantined", 0))
            node_rerank_feature_stats = list(precompute_payload.get("node_rerank_feature_stats", []))  # type: ignore[arg-type]
            node_rerank_feature_calibrations = list(precompute_payload.get("node_rerank_feature_calibrations", []))  # type: ignore[arg-type]
            snc_feature_stats = list(precompute_payload.get("snc_feature_stats", []))  # type: ignore[arg-type]
            snc_feature_calibrations = list(precompute_payload.get("snc_feature_calibrations", []))  # type: ignore[arg-type]
            snc_ref_nodes = int(precompute_payload.get("snc_ref_nodes", 0))
            activity_normalized_chain_density_stats = dict(
                precompute_payload.get("activity_normalized_chain_density_stats", _empty_feature_stats())
            )
            activity_normalized_chain_density_calibration = dict(
                precompute_payload.get("activity_normalized_chain_density_calibration", _empty_calibration())
            )
            activity_normalized_chain_ref_nodes = int(precompute_payload.get("activity_normalized_chain_ref_nodes", 0))
            node_rerank_ref_nodes = int(precompute_payload.get("node_rerank_ref_nodes", 0))
            node_rerank_ref_processed = int(precompute_payload.get("node_rerank_ref_processed", 0))
            adaptive_val_updates = int(precompute_payload.get("adaptive_val_updates", 0))
            adaptive_val_quarantined = int(precompute_payload.get("adaptive_val_quarantined", 0))
            adaptive_memory_stats = dict(precompute_payload["adaptive_memory_stats"])  # type: ignore[arg-type]
            adaptive_memory_calibration = dict(precompute_payload["adaptive_memory_calibration"])  # type: ignore[arg-type]
            validation_seconds = float(time.perf_counter() - phase_start)
            memory_samples.append({"phase": "after_precompute_cache_load", **memory_snapshot()})
        else:
            log("[TFLR-LowRank] fitting leakage-free low-rank predictor on train stream")
            phase_start = time.perf_counter()
            train_summary = model.fit_rows(
                train_rows_with_optional_chain_profile(
                    _stream_events(
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
            train_seconds = float(time.perf_counter() - phase_start)
            memory_samples.append({"phase": "after_train_fit", **memory_snapshot()})

            log("[TFLR-LowRank] scoring validation for robust stats")
            phase_start = time.perf_counter()
            ref_components = []
            states: dict[int, np.ndarray] = {}
            last_seen: dict[int, int] = {}
            last_edge_seen: dict[str, int] = {}
            global_latent_state = model._zero_state()
            last_global_ts = -1
            chain_ref_scores: list[float] = []
            chain_assoc_ref_scores: list[float] = []
            chain_ref_state = chain_profile.new_state() if chain_profile is not None else None
            ref_processed = 0
            for row in _stream_events(conn, year_month, val_days, indexid2summary, hash2type, hash2uuid_index, set(), event_filter, int(args.fetch_size), int(args.max_ref_events)):
                if chain_profile is not None and chain_ref_state is not None:
                    ref_chain_details = chain_profile.score_pre_update(row, chain_ref_state)
                    chain_ref_scores.append(float(ref_chain_details["score"]))
                    chain_assoc_ref_scores.append(_adaptive_memory_assoc_value(ref_chain_details))
                z, aux, _pred = model.score_pre_update(row, states, last_seen, last_edge_seen, global_latent_state, last_global_ts)
                ref_components.append(np.asarray(_component_values(aux), dtype=np.float32))
                global_latent_state = model.update_states(row, z, states, last_seen, last_edge_seen, global_latent_state)
                last_global_ts = int(row["timestamp_ns"])
                if chain_profile is not None and chain_ref_state is not None:
                    chain_profile.update_state(row, chain_ref_state)
                ref_processed += 1
                if ref_processed % 1_000_000 == 0:
                    log(f"[TFLR-LowRank] val processed={ref_processed}")
            ref_matrix = np.stack(ref_components, axis=0) if ref_components else np.zeros((0, len(COMPONENTS)), dtype=np.float32)
            comp_stats = [robust_stats(ref_matrix[:, i]) for i in range(ref_matrix.shape[1])]
            chain_profile_stats = (
                robust_stats(np.asarray(chain_ref_scores, dtype=np.float32))
                if chain_profile is not None
                else {"median": 0.0, "mad": 1.0, "q95": 0.0, "q99": 0.0, "q995": 0.0, "q999": 0.0}
            )
            chain_assoc_stats = (
                robust_stats(np.asarray(chain_assoc_ref_scores, dtype=np.float32))
                if chain_profile is not None
                else {"median": 0.0, "mad": 1.0, "q95": 0.0, "q99": 0.0, "q995": 0.0, "q999": 0.0}
            )
            ref_fused = np.asarray([_fused_score(row.tolist(), comp_stats, weights, float(args.score_clip)) for row in ref_matrix], dtype=np.float32)
            fused_calibration = _quantile_calibration(ref_fused, int(args.chain_profile_calibration_knots))
            chain_profile_calibration = _quantile_calibration(
                np.asarray(chain_ref_scores, dtype=np.float32),
                int(args.chain_profile_calibration_knots),
            )
            finite_ref_fused = ref_fused[np.isfinite(ref_fused)]
            threshold = float(np.quantile(finite_ref_fused, float(args.event_threshold_quantile))) if finite_ref_fused.size else 0.0
            prop_threshold = float(np.quantile(finite_ref_fused, float(args.prop_threshold_quantile))) if finite_ref_fused.size else threshold
            online_node_alert_threshold_source = "explicit"
            if float(args.online_node_alert_threshold) >= 0.0:
                online_node_alert_threshold = float(args.online_node_alert_threshold)
            else:
                online_node_alert_threshold_source = "validation_fused_event_quantile"
                online_node_alert_threshold = (
                    float(np.quantile(finite_ref_fused, float(args.online_node_alert_threshold_quantile)))
                    if finite_ref_fused.size
                    else float(threshold)
                )
            adaptive_memory_enabled = bool(args.adaptive_memory_enabled)
            adaptive_memory = _new_adaptive_memory(int(args.adaptive_memory_buckets), int(args.latent_dim)) if adaptive_memory_enabled else _new_adaptive_memory(8, int(args.latent_dim))
            adaptive_memory_ref_distances: list[float] = []
            adaptive_train_processed = 0
            adaptive_train_updates = 0
            adaptive_train_quarantined = 0
            if adaptive_memory_enabled:
                log("[TFLR-LowRank] fitting adaptive benign memory on high-confidence train stream")
                adaptive_train_states: dict[int, np.ndarray] = {}
                adaptive_train_last_seen: dict[int, int] = {}
                adaptive_train_last_edge_seen: dict[str, int] = {}
                adaptive_train_global_state = model._zero_state()
                adaptive_train_last_global_ts = -1
                adaptive_train_chain_state = chain_profile.new_state() if chain_profile is not None else None
                for row in _stream_events(conn, year_month, train_days, indexid2summary, hash2type, hash2uuid_index, set(), event_filter, int(args.fetch_size), int(args.max_train_events)):
                    chain_details = {
                        "score": 0.0,
                        "association_discrepancy": 0.0,
                        "association_score": 0.0,
                    }
                    chain_profile_score = 0.0
                    chain_profile_tail = 0.0
                    if chain_profile is not None and adaptive_train_chain_state is not None:
                        chain_details = chain_profile.score_pre_update(row, adaptive_train_chain_state)
                        chain_profile_score = float(chain_details["score"])
                        chain_profile_tail = _tail_surprise_scalar(chain_profile_score, chain_profile_calibration, float(args.chain_profile_clip))
                    z, aux, _pred = model.score_pre_update(row, adaptive_train_states, adaptive_train_last_seen, adaptive_train_last_edge_seen, adaptive_train_global_state, adaptive_train_last_global_ts)
                    score = _fused_score(_component_values(aux), comp_stats, weights, float(args.score_clip))
                    event_tail = _tail_surprise_scalar(score, fused_calibration, float(args.chain_profile_clip))
                    assoc_value = _adaptive_memory_assoc_value(chain_details)
                    assoc_z = _upper_z_scalar(assoc_value, chain_assoc_stats, float(args.adaptive_memory_clip))
                    _distance, _has_proto, adaptive_key = _adaptive_memory_distance(adaptive_memory, row, z, int(args.adaptive_memory_min_count))
                    quarantined = _adaptive_memory_quarantine(
                        event_tail,
                        chain_profile_tail,
                        assoc_z,
                        float(args.adaptive_memory_quarantine_tail_min),
                        float(args.adaptive_memory_quarantine_assoc_z_min),
                    )
                    if _adaptive_memory_can_update(
                        event_tail,
                        chain_profile_tail,
                        assoc_z,
                        float(args.adaptive_memory_update_tail_max),
                        float(args.adaptive_memory_update_assoc_z_max),
                        quarantined,
                    ):
                        _adaptive_memory_update(adaptive_memory, adaptive_key, z, float(args.adaptive_memory_fit_alpha))
                        adaptive_train_updates += 1
                    elif quarantined:
                        adaptive_memory["quarantined"] = int(adaptive_memory.get("quarantined", 0)) + 1
                        adaptive_train_quarantined += 1
                    adaptive_train_global_state = model.update_states(row, z, adaptive_train_states, adaptive_train_last_seen, adaptive_train_last_edge_seen, adaptive_train_global_state)
                    adaptive_train_last_global_ts = int(row["timestamp_ns"])
                    if chain_profile is not None and adaptive_train_chain_state is not None:
                        chain_profile.update_state(row, adaptive_train_chain_state)
                    adaptive_train_processed += 1
                    if adaptive_train_processed % 1_000_000 == 0:
                        log(f"[TFLR-LowRank] adaptive memory train processed={adaptive_train_processed} updates={adaptive_train_updates} quarantined={adaptive_train_quarantined}")
            node_rerank_feature_stats: list[dict[str, float]] = []
            node_rerank_feature_calibrations: list[dict[str, object]] = []
            snc_feature_stats: list[dict[str, float]] = []
            snc_feature_calibrations: list[dict[str, object]] = []
            snc_ref_nodes = 0
            activity_normalized_chain_density_stats = _empty_feature_stats()
            activity_normalized_chain_density_calibration = _empty_calibration()
            activity_normalized_chain_ref_nodes = 0
            node_rerank_ref_nodes = 0
            node_rerank_ref_processed = 0
            adaptive_val_updates = 0
            adaptive_val_quarantined = 0
            if (
                bool(args.node_rerank_enabled)
                or adaptive_memory_enabled
                or bool(args.online_node_alerts_enabled)
                or bool(args.activity_normalized_chain_enabled)
            ):
                log("[TFLR-LowRank] calibrating node reranker/adaptive memory on validation stream")
                val_seen = np.zeros((max_node_id,), dtype=bool)
                val_node_max = np.zeros((max_node_id,), dtype=np.float32)
                val_node_high_count = np.zeros((max_node_id,), dtype=np.float32)
                val_node_soft_sum = np.zeros((max_node_id,), dtype=np.float32)
                val_chain_profile_bonus = np.zeros((max_node_id,), dtype=np.float32) if chain_profile_enabled else np.zeros((0,), dtype=np.float32)
                val_chain_profile_raw = np.zeros((max_node_id,), dtype=np.float32) if chain_profile_enabled else np.zeros((0,), dtype=np.float32)
                val_extra = _new_node_rerank_extra(max_node_id)
                val_prop_state = np.zeros((max_node_id,), dtype=np.float32)
                val_prop_last_ts = np.full((max_node_id,), -1, dtype=np.int64)
                val_states: dict[int, np.ndarray] = {}
                val_last_seen: dict[int, int] = {}
                val_last_edge_seen: dict[str, int] = {}
                val_global_latent_state = model._zero_state()
                val_last_global_ts = -1
                val_chain_state = chain_profile.new_state() if chain_profile is not None else None
                val_episode_sketch = (
                    CausalEpisodeSketch(
                        max_node_id=max_node_id,
                        num_buckets=int(args.chain_profile_buckets),
                        decay_sec=float(args.online_node_risk_decay_sec),
                        score_cap=float(args.chain_profile_clip),
                    )
                    if bool(args.causal_episode_support_enabled)
                    else None
                )
                adaptive_val_memory = _copy_adaptive_memory(adaptive_memory) if adaptive_memory_enabled else adaptive_memory
                val_snc_state = _new_snc_state(max_node_id) if bool(args.snc_enabled) else {}
    
                def apply_val_node(node: int, score_value: float) -> None:
                    if node < 0 or node >= max_node_id:
                        return
                    val_seen[node] = True
                    if score_value > float(val_node_max[node]):
                        val_node_max[node] = np.float32(score_value)
                    if score_value >= prop_threshold:
                        val_node_high_count[node] += 1.0
                        val_node_soft_sum[node] += np.float32(min(max(float(score_value) - float(prop_threshold), 0.0), float(args.node_soft_cap)))
    
                for row in _stream_events(conn, year_month, val_days, indexid2summary, hash2type, hash2uuid_index, set(), event_filter, int(args.fetch_size), int(args.max_ref_events)):
                    chain_details = {
                        "score": 0.0,
                        "association_discrepancy": 0.0,
                        "association_score": 0.0,
                        "transition_rarity": 0.0,
                        "has_transition": 0.0,
                    }
                    chain_profile_score = 0.0
                    chain_profile_tail = 0.0
                    if chain_profile is not None and val_chain_state is not None:
                        chain_details = chain_profile.score_pre_update(row, val_chain_state)
                        chain_profile_score = float(chain_details["score"])
                        chain_profile_tail = _tail_surprise_scalar(
                            chain_profile_score,
                            chain_profile_calibration,
                            float(args.chain_profile_clip),
                        )
                    z, aux, _pred = model.score_pre_update(row, val_states, val_last_seen, val_last_edge_seen, val_global_latent_state, val_last_global_ts)
                    values = _component_values(aux)
                    score = _fused_score(values, comp_stats, weights, float(args.score_clip))
                    event_tail = _tail_surprise_scalar(score, fused_calibration, float(args.chain_profile_clip)) if chain_profile_enabled else 0.0
                    if str(args.chain_profile_score_mode) == "conformal":
                        chain_bonus_value = float(chain_profile_tail)
                    elif str(args.chain_profile_score_mode) == "agreement":
                        chain_bonus_value = float(min(chain_profile_tail, event_tail))
                    else:
                        chain_bonus_value = _upper_z_scalar(chain_profile_score, chain_profile_stats, float(args.chain_profile_clip))
                    component_scores = _component_z_values(aux, comp_stats, weights, float(args.score_clip))
                    endpoint_bonus = float(component_scores[COMPONENTS.index("endpoint_self_rarity")])
                    semantic_z = _upper_z_scalar(float(aux["semantic_max_rarity"]), comp_stats[COMPONENTS.index("semantic_max_rarity")], float(args.score_clip))
                    malicious_chain_signal = (
                        _compact_malicious_chain_signal(
                            event_tail,
                            semantic_z,
                            chain_bonus_value,
                            chain_profile_score,
                            float(chain_details.get("transition_rarity", 0.0)),
                            float(chain_details.get("association_discrepancy", 0.0)),
                            float(chain_details.get("association_score", 0.0)),
                            float(chain_details.get("has_transition", 0.0)),
                        )
                        if bool(args.compact_malicious_chain_enabled)
                        else 0.0
                    )
                    adaptive_distance = 0.0
                    adaptive_has_proto = False
                    adaptive_key = 0
                    adaptive_assoc_z = 0.0
                    adaptive_quarantined = False
                    if adaptive_memory_enabled:
                        adaptive_distance, adaptive_has_proto, adaptive_key = _adaptive_memory_distance(
                            adaptive_val_memory,
                            row,
                            z,
                            int(args.adaptive_memory_min_count),
                        )
                        if adaptive_has_proto:
                            adaptive_memory_ref_distances.append(float(adaptive_distance))
                        adaptive_assoc_z = _upper_z_scalar(
                            _adaptive_memory_assoc_value(chain_details),
                            chain_assoc_stats,
                            float(args.adaptive_memory_clip),
                        )
                        adaptive_quarantined = _adaptive_memory_quarantine(
                            event_tail,
                            chain_profile_tail,
                            adaptive_assoc_z,
                            float(args.adaptive_memory_quarantine_tail_min),
                            float(args.adaptive_memory_quarantine_assoc_z_min),
                        )
                    else:
                        adaptive_assoc_z = _upper_z_scalar(
                            _adaptive_memory_assoc_value(chain_details),
                            chain_assoc_stats,
                            float(args.snc_clip),
                        )
                    src = int(row["src_idx"])
                    dst = int(row["dst_idx"])
                    info_src = int(row["info_src"])
                    info_dst = int(row["info_dst"])
                    ts = int(row["timestamp_ns"])
                    episode_support = 0.0
                    episode_raw_support = 0.0
                    if val_episode_sketch is not None:
                        episode_details = val_episode_sketch.score_pre_update(
                            row,
                            ts,
                            float(event_tail),
                            float(chain_bonus_value),
                            float(max(float(chain_details.get("association_score", 0.0)), float(chain_details.get("association_discrepancy", 0.0)), 0.0)),
                            float(chain_details.get("transition_rarity", 0.0)),
                            adaptive_support=0.0,
                            quarantined=bool(adaptive_quarantined),
                            min_event_tail=float(args.causal_episode_min_event_tail),
                            min_chain=float(args.causal_episode_min_chain),
                            min_assoc=float(args.causal_episode_min_assoc),
                            min_families=int(args.causal_episode_min_families),
                            selective=bool(args.causal_episode_selective_enabled),
                        )
                        episode_support = float(episode_details["support"])
                        episode_raw_support = float(episode_details["raw_support"])
                    src_prop = _decayed_scalar_state(val_prop_state, val_prop_last_ts, info_src, ts, float(args.prop_decay_sec))
                    propagation_candidate = float(args.prop_boost) * float(src_prop)
                    local_score = max(float(score), propagation_candidate, float(endpoint_bonus))
                    nodes = {src, dst, info_src, info_dst}
                    for node in nodes:
                        apply_val_node(int(node), float(local_score))
                        if chain_profile_enabled and chain_bonus_value > 0.0 and 0 <= int(node) < max_node_id:
                            if chain_bonus_value > float(val_chain_profile_bonus[int(node)]):
                                val_chain_profile_bonus[int(node)] = np.float32(chain_bonus_value)
                                val_chain_profile_raw[int(node)] = np.float32(chain_profile_score)
                    _update_node_rerank_extra(
                        val_extra,
                        nodes,
                        info_src,
                        bool(propagation_candidate > float(score)),
                        semantic_z,
                        chain_bonus_value,
                        chain_profile_score,
                        float(chain_details.get("association_discrepancy", 0.0)),
                        float(chain_details.get("association_score", 0.0)),
                        float(malicious_chain_signal),
                        int(row["relation_id"]),
                        int(row["info_src_type"]),
                        int(row["info_dst_type"]),
                        local_score,
                        max_node_id,
                    )
                    if val_episode_sketch is not None and episode_support > 0.0:
                        val_episode_sketch.update_state(
                            row,
                            nodes,
                            ts,
                            episode_support,
                            episode_raw_support,
                            int(row["relation_id"]),
                            int(row["info_src_type"]),
                            int(row["info_dst_type"]),
                            int(node_rerank_ref_processed),
                        )
                    if bool(args.snc_enabled):
                        snc_support = _snc_event_support(
                            event_tail,
                            semantic_z,
                            chain_bonus_value,
                            chain_profile_score,
                            adaptive_assoc_z,
                            0.0,
                            adaptive_quarantined,
                        )
                        _update_snc_state(
                            val_snc_state,
                            info_src,
                            info_dst,
                            ts,
                            snc_support,
                            adaptive_assoc_z,
                            int(row["relation_id"]),
                            int(row["info_dst_type"]),
                            float(val_node_high_count[info_dst]) if 0 <= info_dst < max_node_id else 0.0,
                            adaptive_quarantined,
                            node_rerank_ref_processed,
                            float(args.snc_decay_sec),
                            float(args.snc_update_min_support),
                            float(args.snc_update_min_assoc_z),
                        )
                        _update_snc_state(
                            val_snc_state,
                            src,
                            dst,
                            ts,
                            snc_support,
                            adaptive_assoc_z,
                            int(row["relation_id"]),
                            int(row["info_dst_type"]),
                            float(val_node_high_count[dst]) if 0 <= dst < max_node_id else 0.0,
                            adaptive_quarantined,
                            node_rerank_ref_processed,
                            float(args.snc_decay_sec),
                            float(args.snc_update_min_support),
                            float(args.snc_update_min_assoc_z),
                        )
                    if score >= threshold:
                        _set_decayed_state_max(val_prop_state, val_prop_last_ts, info_src, ts, float(score), float(args.prop_decay_sec))
                        _set_decayed_state_max(val_prop_state, val_prop_last_ts, info_dst, ts, float(score), float(args.prop_decay_sec))
                    if local_score >= prop_threshold * float(args.prop_update_threshold_ratio):
                        _set_decayed_state_max(val_prop_state, val_prop_last_ts, info_dst, ts, float(local_score), float(args.prop_decay_sec))
                    if adaptive_memory_enabled:
                        if _adaptive_memory_can_update(
                            event_tail,
                            chain_profile_tail,
                            adaptive_assoc_z,
                            float(args.adaptive_memory_update_tail_max),
                            float(args.adaptive_memory_update_assoc_z_max),
                            adaptive_quarantined,
                        ):
                            _adaptive_memory_update(adaptive_val_memory, adaptive_key, z, float(args.adaptive_memory_update_alpha))
                            adaptive_val_updates += 1
                        elif adaptive_quarantined:
                            adaptive_val_memory["quarantined"] = int(adaptive_val_memory.get("quarantined", 0)) + 1
                            adaptive_val_quarantined += 1
                    val_global_latent_state = model.update_states(row, z, val_states, val_last_seen, val_last_edge_seen, val_global_latent_state)
                    val_last_global_ts = int(row["timestamp_ns"])
                    if chain_profile is not None and val_chain_state is not None:
                        chain_profile.update_state(row, val_chain_state)
                    node_rerank_ref_processed += 1
                    if node_rerank_ref_processed % 1_000_000 == 0:
                        log(f"[TFLR-LowRank] node rerank val processed={node_rerank_ref_processed}")
                _, val_feature_matrix = _node_rerank_feature_matrix(
                    val_seen,
                    val_node_max,
                    val_node_high_count,
                    val_node_soft_sum,
                    val_chain_profile_bonus,
                    val_chain_profile_raw,
                    val_extra,
                )
                node_rerank_ref_nodes = int(val_feature_matrix.shape[0])
                node_rerank_feature_stats = _node_feature_stats(val_feature_matrix)
                node_rerank_feature_calibrations = _node_feature_calibrations(
                    val_feature_matrix,
                    int(args.node_rerank_calibration_knots),
                )
                if bool(args.activity_normalized_chain_enabled):
                    val_activity_chain_density = _activity_normalized_chain_density(
                        val_chain_profile_raw,
                        val_node_high_count,
                    )
                    val_density_mask = (
                        val_seen
                        & np.isfinite(val_activity_chain_density)
                        & (val_chain_profile_raw > 0.0)
                    )
                    activity_normalized_chain_ref_nodes = int(np.sum(val_density_mask))
                    if activity_normalized_chain_ref_nodes > 0:
                        val_density_ref = val_activity_chain_density[val_density_mask]
                        activity_normalized_chain_density_stats = robust_stats(val_density_ref)
                        activity_normalized_chain_density_calibration = _quantile_calibration(
                            val_density_ref,
                            int(args.node_rerank_calibration_knots),
                        )
                if bool(args.online_node_alerts_enabled) and float(args.online_node_alert_threshold) < 0.0:
                    val_node_intrinsic_ref = _node_score(
                        val_node_max,
                        val_node_high_count,
                        val_node_soft_sum,
                        float(args.node_count_weight),
                        float(args.node_sum_weight),
                    )
                    val_compact_ref = np.zeros((max_node_id,), dtype=np.float32)
                    if chain_profile_enabled:
                        val_compact_ref = np.maximum(
                            val_compact_ref,
                            float(args.chain_profile_score_weight) * val_chain_profile_bonus,
                        )
                    if val_episode_sketch is not None:
                        val_compact_ref = np.maximum(
                            val_compact_ref,
                            float(args.chain_profile_score_weight) * val_episode_sketch.node_support,
                        )
                    if val_extra:
                        val_compact_ref = np.maximum.reduce(
                            [
                                val_compact_ref,
                                val_extra["association_max"],
                                val_extra["local_chain_agreement"],
                                val_extra["malicious_chain_peak"],
                            ]
                        ).astype(np.float32, copy=False)
                    val_activity_ref = np.log1p(np.maximum(val_node_high_count, 0.0)) + np.log1p(np.maximum(val_node_soft_sum, 0.0))
                    val_activity_pressure = float(args.online_node_activity_weight) * np.maximum(
                        val_activity_ref - val_compact_ref - float(args.online_node_activity_margin),
                        0.0,
                    )
                    if float(args.online_node_activity_cap) > 0.0:
                        val_activity_pressure = np.minimum(val_activity_pressure, float(args.online_node_activity_cap))
                    val_node_risk_ref = val_node_intrinsic_ref + val_compact_ref - val_activity_pressure.astype(np.float32, copy=False)
                    finite_val_node_risk = val_node_risk_ref[val_seen & np.isfinite(val_node_risk_ref)]
                    if finite_val_node_risk.size:
                        online_node_alert_threshold = float(
                            np.quantile(finite_val_node_risk, float(args.online_node_alert_threshold_quantile))
                        )
                        online_node_alert_threshold_source = "validation_node_online_risk_quantile"
                if bool(args.snc_enabled):
                    _, val_snc_matrix = _snc_feature_matrix(val_seen, val_snc_state)
                    snc_ref_nodes = int(val_snc_matrix.shape[0])
                    snc_feature_stats = _snc_feature_stats(val_snc_matrix)
                    snc_feature_calibrations = _snc_feature_calibrations(val_snc_matrix, int(args.snc_calibration_knots))
                if adaptive_memory_enabled:
                    adaptive_memory = _copy_adaptive_memory(adaptive_val_memory)
            adaptive_memory_stats = (
            robust_stats(np.asarray(adaptive_memory_ref_distances, dtype=np.float32))
            if adaptive_memory_enabled and adaptive_memory_ref_distances
            else {"median": 0.0, "mad": 1.0, "q95": 0.0, "q99": 0.0, "q995": 0.0, "q999": 0.0}
        )
        adaptive_memory_calibration = (
            _quantile_calibration(np.asarray(adaptive_memory_ref_distances, dtype=np.float32), int(args.adaptive_memory_calibration_knots))
            if adaptive_memory_enabled and adaptive_memory_ref_distances
            else {"num_values": 0, "quantiles": [], "values": []}
        )
        validation_seconds = float(time.perf_counter() - phase_start)
        memory_samples.append({"phase": "after_validation", **memory_snapshot()})
        if (
            precompute_payload is None
            and precompute_cache_enabled
            and str(args.precompute_cache_mode) in {"auto", "write"}
        ):
            try:
                _write_pickle_cache(
                    precompute_cache_path,
                    {
                        "fingerprint": precompute_cache_fingerprint,
                        "fingerprint_payload": precompute_fingerprint_payload,
                        "model": model,
                        "chain_profile": chain_profile,
                        "train_summary": train_summary,
                        "comp_stats": comp_stats,
                        "chain_profile_stats": chain_profile_stats,
                        "chain_assoc_stats": chain_assoc_stats,
                        "fused_calibration": fused_calibration,
                        "chain_profile_calibration": chain_profile_calibration,
                        "threshold": float(threshold),
                        "prop_threshold": float(prop_threshold),
                        "online_node_alert_threshold": float(online_node_alert_threshold),
                        "online_node_alert_threshold_source": str(online_node_alert_threshold_source),
                        "ref_processed": int(ref_processed),
                        "adaptive_memory": _copy_adaptive_memory(adaptive_memory),
                        "adaptive_memory_ref_distances": [float(x) for x in adaptive_memory_ref_distances],
                        "adaptive_train_processed": int(adaptive_train_processed),
                        "adaptive_train_updates": int(adaptive_train_updates),
                        "adaptive_train_quarantined": int(adaptive_train_quarantined),
                        "node_rerank_feature_stats": node_rerank_feature_stats,
                        "node_rerank_feature_calibrations": node_rerank_feature_calibrations,
                        "snc_feature_stats": snc_feature_stats,
                        "snc_feature_calibrations": snc_feature_calibrations,
                        "snc_ref_nodes": int(snc_ref_nodes),
                        "activity_normalized_chain_density_stats": activity_normalized_chain_density_stats,
                        "activity_normalized_chain_density_calibration": activity_normalized_chain_density_calibration,
                        "activity_normalized_chain_ref_nodes": int(activity_normalized_chain_ref_nodes),
                        "node_rerank_ref_nodes": int(node_rerank_ref_nodes),
                        "node_rerank_ref_processed": int(node_rerank_ref_processed),
                        "adaptive_val_updates": int(adaptive_val_updates),
                        "adaptive_val_quarantined": int(adaptive_val_quarantined),
                        "adaptive_memory_stats": adaptive_memory_stats,
                        "adaptive_memory_calibration": adaptive_memory_calibration,
                    },
                )
                precompute_cache_written = True
                log(f"[TFLR-LowRank] wrote train/validation precompute cache: {precompute_cache_path}")
            except Exception as exc:
                precompute_cache_error = f"{type(exc).__name__}: {exc}"
                if bool(args.precompute_cache_strict):
                    raise
                log(f"[TFLR-LowRank] failed to write precompute cache: {precompute_cache_error}")

        log("[TFLR-LowRank] loading ground truth for evaluation only")
        abnormal_nodes = set(int(x) for x in load_ground_truth_indices(cfg, uuid2index, result_dir))

        log("[TFLR-LowRank] scoring test and aggregating nodes")
        phase_start = time.perf_counter()
        all_nodes = np.zeros((max_node_id,), dtype=bool)
        positive_nodes = np.zeros((max_node_id,), dtype=bool)
        suspect_nodes = np.zeros((max_node_id,), dtype=bool)
        positive_node_first_event: dict[int, int] = {}
        positive_node_first_ts: dict[int, int] = {}
        node_max = np.zeros((max_node_id,), dtype=np.float32)
        node_high_count = np.zeros((max_node_id,), dtype=np.float32)
        node_soft_sum = np.zeros((max_node_id,), dtype=np.float32)
        node_intrinsic_max = np.zeros((max_node_id,), dtype=np.float32)
        node_intrinsic_high_count = np.zeros((max_node_id,), dtype=np.float32)
        node_intrinsic_soft_sum = np.zeros((max_node_id,), dtype=np.float32)
        node_top_event = np.full((max_node_id,), -1, dtype=np.int64)
        node_top_label = np.full((max_node_id,), -1, dtype=np.int8)
        node_top_component = np.full((max_node_id,), -1, dtype=np.int8)
        prop_state = np.zeros((max_node_id,), dtype=np.float32)
        prop_last_ts = np.full((max_node_id,), -1, dtype=np.int64)
        recent_neighbor = np.full((max_node_id,), -1, dtype=np.int64)
        recent_neighbor_ts = np.full((max_node_id,), -1, dtype=np.int64)
        compact_chain_enabled = bool(args.compact_chain_enabled)
        compact_chain_slots = max(int(args.compact_chain_slots), 0) if compact_chain_enabled else 0
        compact_chain_bonus = np.zeros((max_node_id,), dtype=np.float32) if compact_chain_enabled else np.zeros((0,), dtype=np.float32)
        compact_chain_support = np.zeros((max_node_id,), dtype=np.float32) if compact_chain_enabled else np.zeros((0,), dtype=np.float32)
        compact_chain_last_ts = np.full((max_node_id,), -1, dtype=np.int64) if compact_chain_enabled else np.full((0,), -1, dtype=np.int64)
        compact_chain_rel_mask = np.zeros((max_node_id,), dtype=np.uint16) if compact_chain_enabled else np.zeros((0,), dtype=np.uint16)
        compact_chain_type_mask = np.zeros((max_node_id,), dtype=np.uint8) if compact_chain_enabled else np.zeros((0,), dtype=np.uint8)
        compact_chain_neighbor_mask = np.zeros((max_node_id,), dtype=np.uint32) if compact_chain_enabled else np.zeros((0,), dtype=np.uint32)
        compact_chain_slot_nodes = (
            np.full((max_node_id, compact_chain_slots), -1, dtype=np.int32)
            if compact_chain_slots > 0
            else np.full((0, 0), -1, dtype=np.int32)
        )
        compact_chain_slot_scores = (
            np.zeros((max_node_id, compact_chain_slots), dtype=np.float32)
            if compact_chain_slots > 0
            else np.zeros((0, 0), dtype=np.float32)
        )
        compact_chain_slot_ts = (
            np.full((max_node_id, compact_chain_slots), -1, dtype=np.int64)
            if compact_chain_slots > 0
            else np.full((0, 0), -1, dtype=np.int64)
        )
        chain_profile_bonus = np.zeros((max_node_id,), dtype=np.float32) if chain_profile_enabled else np.zeros((0,), dtype=np.float32)
        chain_profile_raw = np.zeros((max_node_id,), dtype=np.float32) if chain_profile_enabled else np.zeros((0,), dtype=np.float32)
        chain_profile_conformal = np.zeros((max_node_id,), dtype=np.float32) if chain_profile_enabled else np.zeros((0,), dtype=np.float32)
        chain_profile_event_surprise = np.zeros((max_node_id,), dtype=np.float32) if chain_profile_enabled else np.zeros((0,), dtype=np.float32)
        chain_profile_event_pos = np.full((max_node_id,), -1, dtype=np.int64) if chain_profile_enabled else np.full((0,), -1, dtype=np.int64)
        causal_episode_enabled = bool(args.causal_episode_support_enabled)
        causal_episode_sketch = (
            CausalEpisodeSketch(
                max_node_id=max_node_id,
                num_buckets=int(args.chain_profile_buckets),
                decay_sec=float(args.online_node_risk_decay_sec),
                score_cap=float(args.chain_profile_clip),
            )
            if causal_episode_enabled
            else None
        )
        adaptive_memory_bonus = np.zeros((max_node_id,), dtype=np.float32) if adaptive_memory_enabled else np.zeros((0,), dtype=np.float32)
        adaptive_memory_raw = np.zeros((max_node_id,), dtype=np.float32) if adaptive_memory_enabled else np.zeros((0,), dtype=np.float32)
        adaptive_memory_event_pos = np.full((max_node_id,), -1, dtype=np.int64) if adaptive_memory_enabled else np.full((0,), -1, dtype=np.int64)
        adaptive_memory_quarantine_node = np.zeros((max_node_id,), dtype=np.float32) if adaptive_memory_enabled else np.zeros((0,), dtype=np.float32)
        adaptive_memory_update_node = np.zeros((max_node_id,), dtype=np.float32) if adaptive_memory_enabled else np.zeros((0,), dtype=np.float32)
        node_rerank_extra = _new_node_rerank_extra(max_node_id) if bool(args.node_rerank_enabled) else {}
        snc_state = _new_snc_state(max_node_id) if bool(args.snc_enabled) else {}
        snc_arrays = {
            "candidate": np.zeros((max_node_id,), dtype=bool) if bool(args.snc_enabled) else np.zeros((0,), dtype=bool),
            "applied": np.zeros((max_node_id,), dtype=bool) if bool(args.snc_enabled) else np.zeros((0,), dtype=bool),
            "pass": np.zeros((max_node_id,), dtype=bool) if bool(args.snc_enabled) else np.zeros((0,), dtype=bool),
            "score": np.zeros((max_node_id,), dtype=np.float32) if bool(args.snc_enabled) else np.zeros((0,), dtype=np.float32),
            "boost": np.zeros((max_node_id,), dtype=np.float32) if bool(args.snc_enabled) else np.zeros((0,), dtype=np.float32),
            "penalty": np.zeros((max_node_id,), dtype=np.float32) if bool(args.snc_enabled) else np.zeros((0,), dtype=np.float32),
            "cluster_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.snc_enabled) else np.zeros((0,), dtype=np.float32),
            "object_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.snc_enabled) else np.zeros((0,), dtype=np.float32),
            "diversity_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.snc_enabled) else np.zeros((0,), dtype=np.float32),
            "quarantine_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.snc_enabled) else np.zeros((0,), dtype=np.float32),
            "low_recurrence": np.zeros((max_node_id,), dtype=np.float32) if bool(args.snc_enabled) else np.zeros((0,), dtype=np.float32),
            "benign_update_penalty": np.zeros((max_node_id,), dtype=np.float32) if bool(args.snc_enabled) else np.zeros((0,), dtype=np.float32),
            "high_count_penalty": np.zeros((max_node_id,), dtype=np.float32) if bool(args.snc_enabled) else np.zeros((0,), dtype=np.float32),
        }
        snc_meta: dict[str, object] = {
            "candidate_nodes": 0,
            "applied_nodes": 0,
            "passed_nodes": 0,
            "max_boost": 0.0,
            "mean_boost": 0.0,
            "max_penalty": 0.0,
            "mean_score": 0.0,
        }
        node_rerank_delta_values = np.zeros((max_node_id,), dtype=np.float32) if bool(args.node_rerank_enabled) else np.zeros((0,), dtype=np.float32)
        node_rerank_support_values = np.zeros((max_node_id,), dtype=np.float32) if bool(args.node_rerank_enabled) else np.zeros((0,), dtype=np.float32)
        node_intrinsic_rank_support_values = np.zeros((max_node_id,), dtype=np.float32) if bool(args.node_rerank_enabled) else np.zeros((0,), dtype=np.float32)
        node_rerank_penalty_values = np.zeros((max_node_id,), dtype=np.float32) if bool(args.node_rerank_enabled) else np.zeros((0,), dtype=np.float32)
        node_rerank_high_activity_penalty_values = np.zeros((max_node_id,), dtype=np.float32) if bool(args.node_rerank_enabled) else np.zeros((0,), dtype=np.float32)
        node_hard_gate_support = np.zeros((max_node_id,), dtype=np.float32) if bool(args.node_hard_gate_enabled) else np.zeros((0,), dtype=np.float32)
        node_hard_gate_prop_pressure = np.zeros((max_node_id,), dtype=np.float32) if bool(args.node_hard_gate_enabled) else np.zeros((0,), dtype=np.float32)
        node_hard_gate_family_count = np.zeros((max_node_id,), dtype=np.float32) if bool(args.node_hard_gate_enabled) else np.zeros((0,), dtype=np.float32)
        node_hard_gate_penalty = np.zeros((max_node_id,), dtype=np.float32) if bool(args.node_hard_gate_enabled) else np.zeros((0,), dtype=np.float32)
        node_hard_gate_applied = np.zeros((max_node_id,), dtype=bool) if bool(args.node_hard_gate_enabled) else np.zeros((0,), dtype=bool)
        evidence_band_arrays = {
            "candidate": np.zeros((max_node_id,), dtype=bool) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=bool),
            "applied": np.zeros((max_node_id,), dtype=bool) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=bool),
            "pass": np.zeros((max_node_id,), dtype=bool) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=bool),
            "score": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "boost": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "penalty": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "local": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "relation": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "association": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "semantic": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "role": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "adaptive": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "activity": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "propagation": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "family_count": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "activity_penalty": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "benign_update_penalty": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "propagation_penalty": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "quarantine": np.zeros((max_node_id,), dtype=np.float32) if bool(args.evidence_band_gate_enabled) else np.zeros((0,), dtype=np.float32),
        }
        evidence_band_meta: dict[str, object] = {
            "candidate_nodes": 0,
            "applied_nodes": 0,
            "passed_nodes": 0,
            "max_boost": 0.0,
            "mean_boost": 0.0,
            "max_penalty": 0.0,
            "mean_penalty": 0.0,
            "mean_score": 0.0,
        }
        mid_rank_gate_arrays = {
            "applied": np.zeros((max_node_id,), dtype=bool) if bool(args.mid_rank_gate_enabled) else np.zeros((0,), dtype=bool),
            "candidate": np.zeros((max_node_id,), dtype=bool) if bool(args.mid_rank_gate_enabled) else np.zeros((0,), dtype=bool),
            "penalty": np.zeros((max_node_id,), dtype=np.float32) if bool(args.mid_rank_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "assoc_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.mid_rank_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "relation_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.mid_rank_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "adaptive_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.mid_rank_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "quarantine": np.zeros((max_node_id,), dtype=np.float32) if bool(args.mid_rank_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "benign_update_penalty": np.zeros((max_node_id,), dtype=np.float32) if bool(args.mid_rank_gate_enabled) else np.zeros((0,), dtype=np.float32),
            "pass": np.zeros((max_node_id,), dtype=bool) if bool(args.mid_rank_gate_enabled) else np.zeros((0,), dtype=bool),
        }
        mid_rank_gate_meta: dict[str, object] = {
            "candidate_nodes": 0,
            "applied_nodes": 0,
            "passed_nodes": 0,
            "max_penalty": 0.0,
            "mean_penalty": 0.0,
        }
        activity_normalized_chain_arrays = _new_activity_normalized_chain_arrays(
            max_node_id,
            bool(args.activity_normalized_chain_enabled),
        )
        activity_normalized_chain_meta = _activity_normalized_chain_meta()
        local_closure_arrays = {
            "candidate": np.zeros((max_node_id,), dtype=bool) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=bool),
            "applied": np.zeros((max_node_id,), dtype=bool) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=bool),
            "pass": np.zeros((max_node_id,), dtype=bool) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=bool),
            "score": np.zeros((max_node_id,), dtype=np.float32) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=np.float32),
            "boost": np.zeros((max_node_id,), dtype=np.float32) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=np.float32),
            "penalty": np.zeros((max_node_id,), dtype=np.float32) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=np.float32),
            "local_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=np.float32),
            "relation_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=np.float32),
            "association_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=np.float32),
            "adaptive_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=np.float32),
            "role_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=np.float32),
            "role_type_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=np.float32),
            "propagation_z": np.zeros((max_node_id,), dtype=np.float32) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=np.float32),
            "family_count": np.zeros((max_node_id,), dtype=np.float32) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=np.float32),
            "benign_update_penalty": np.zeros((max_node_id,), dtype=np.float32) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=np.float32),
            "propagation_penalty": np.zeros((max_node_id,), dtype=np.float32) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=np.float32),
            "hard_assoc_pass": np.zeros((max_node_id,), dtype=bool) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=bool),
            "hard_role_pass": np.zeros((max_node_id,), dtype=bool) if bool(args.local_closure_enabled) else np.zeros((0,), dtype=bool),
        }
        local_closure_meta: dict[str, object] = {
            "candidate_nodes": 0,
            "applied_nodes": 0,
            "passed_nodes": 0,
            "max_boost": 0.0,
            "mean_boost": 0.0,
            "max_penalty": 0.0,
            "mean_score": 0.0,
        }
        chain_test_state = chain_profile.new_state() if chain_profile is not None else None
        test_episode_sketch = causal_episode_sketch
        adaptive_test_memory = _copy_adaptive_memory(adaptive_memory) if adaptive_memory_enabled else adaptive_memory
        adaptive_test_updates = 0
        adaptive_test_quarantined = 0
        adaptive_test_scored = 0
        states = {}
        last_seen = {}
        last_edge_seen = {}
        global_latent_state = model._zero_state()
        last_global_ts = -1
        global_risk_state = 0.0
        global_risk_last_ts = -1
        event_counts_strict = empty_event_counts()
        event_counts_relaxed = empty_event_counts()
        windows = _attack_windows_for_dataset(args.dataset, [0, 1])
        window_rows = [{"name": name, "start": start, "end": end, "detected": False, "ttd": None} for name, start, end in windows]
        test_processed = 0
        profiling_interval = max(int(args.profiling_interval_events), 0)
        profiling_enabled = bool(args.profiling_enabled) or profiling_interval > 0
        if bool(args.profiling_enabled) and profiling_interval <= 0:
            profiling_interval = 1000000
        profile_acc = {
            "chain_profile_seconds": 0.0,
            "event_scoring_seconds": 0.0,
            "adaptive_memory_seconds": 0.0,
            "episode_sketch_seconds": 0.0,
            "online_alert_seconds": 0.0,
            "csv_write_seconds": 0.0,
            "state_update_seconds": 0.0,
        }
        profile_last = {key: 0.0 for key in profile_acc}
        profile_rows: list[dict[str, float | int]] = []

        propagation_component = len(COMPONENTS)
        backtrack_component = len(COMPONENTS) + 1
        chain_profile_component = len(COMPONENTS) + 3
        component_names = COMPONENTS + EXTRA_COMPONENTS
        online_alert_path = str(args.online_node_alerts_path).strip() or os.path.join(result_dir, "online_node_alerts.csv")
        online_risk_state = (
            OnlineNodeRiskState(
                max_node_id=max_node_id,
                count_weight=float(args.node_count_weight),
                sum_weight=float(args.node_sum_weight),
                compact_chain_weight=float(args.compact_chain_score_weight) if compact_chain_enabled else 0.0,
                chain_profile_weight=float(args.chain_profile_score_weight) if chain_profile_enabled else 0.0,
                adaptive_memory_weight=float(args.adaptive_memory_weight) if adaptive_memory_enabled else 0.0,
                activity_weight=float(args.online_node_activity_weight),
                activity_margin=float(args.online_node_activity_margin),
                activity_cap=float(args.online_node_activity_cap),
                decay_sec=float(args.online_node_risk_decay_sec),
                active_node_budget=int(args.active_node_budget),
                node_state_ttl_sec=float(args.node_state_ttl_sec),
                maintenance_interval=int(args.node_state_maintenance_interval),
                eviction_batch=int(args.node_state_eviction_batch),
            )
            if bool(args.online_node_alerts_enabled)
            else None
        )
        online_alert_emitter = (
            NodeAlertEmitter(
                online_alert_path,
                online_node_alert_threshold,
                online_node_alert_threshold_source,
                index_to_type,
                component_names,
                max_node_id,
            )
            if bool(args.online_node_alerts_enabled)
            else None
        )

        def apply_node_evidence(node: int, score_value: float, event_label: int, component: int, update_membership: bool, intrinsic_score: float | None = None) -> None:
            if node < 0 or node >= max_node_id:
                return
            if update_membership:
                all_nodes[node] = True
                if int(node) in abnormal_nodes:
                    positive_nodes[node] = True
                elif event_label == 1:
                    suspect_nodes[node] = True
            if score_value > float(node_max[node]):
                node_max[node] = np.float32(score_value)
                node_top_event[node] = int(test_processed)
                node_top_label[node] = int(event_label)
                node_top_component[node] = int(component)
            if score_value >= prop_threshold:
                node_high_count[node] += 1.0
                node_soft_sum[node] += np.float32(min(max(float(score_value) - float(prop_threshold), 0.0), float(args.node_soft_cap)))
            intrinsic_value = float(score_value if intrinsic_score is None else intrinsic_score)
            if intrinsic_value > float(node_intrinsic_max[node]):
                node_intrinsic_max[node] = np.float32(intrinsic_value)
            if intrinsic_value >= prop_threshold:
                node_intrinsic_high_count[node] += 1.0
                node_intrinsic_soft_sum[node] += np.float32(min(max(float(intrinsic_value) - float(prop_threshold), 0.0), float(args.node_soft_cap)))

        def remember_pair(left: int, right: int, ts_value: int) -> None:
            if 0 <= left < max_node_id and 0 <= right < max_node_id and left != right:
                recent_neighbor[left] = int(right)
                recent_neighbor_ts[left] = int(ts_value)

        def backtrack_recent(seed: int, seed_score: float, ts_value: int, event_label: int) -> None:
            if int(args.backtrack_hops) <= 0 or seed < 0 or seed >= max_node_id:
                return
            frontier = [int(seed)]
            seen_nodes = {int(seed)}
            for hop in range(int(args.backtrack_hops)):
                next_frontier: list[int] = []
                for node in frontier:
                    neighbor = int(recent_neighbor[node]) if 0 <= node < max_node_id else -1
                    if neighbor < 0 or neighbor >= max_node_id or neighbor in seen_nodes:
                        continue
                    prev_ts = int(recent_neighbor_ts[node])
                    if prev_ts < 0:
                        continue
                    dt = max(float(ts_value - prev_ts) / 1e9, 0.0)
                    boost = (float(args.backtrack_boost) ** float(hop + 1)) * math.exp(-dt / max(float(args.backtrack_decay_sec), 1e-3))
                    score_value = float(seed_score) * boost
                    if score_value < float(prop_threshold) * float(args.backtrack_min_score_ratio):
                        continue
                    apply_node_evidence(neighbor, score_value, event_label, backtrack_component, update_membership=False, intrinsic_score=0.0)
                    _set_decayed_state_max(prop_state, prop_last_ts, neighbor, ts_value, score_value, float(args.prop_decay_sec))
                    seen_nodes.add(neighbor)
                    next_frontier.append(neighbor)
                frontier = next_frontier
                if not frontier:
                    break

        def compact_neighbor_bit(neighbor: int) -> int:
            hashed = (int(neighbor) * 2654435761) & 0xFFFFFFFF
            return int(1 << (hashed & 31))

        def compact_chain_diversity(node: int) -> int:
            if not compact_chain_enabled or node < 0 or node >= max_node_id:
                return 0
            return int(
                _popcount(int(compact_chain_rel_mask[node]))
                + _popcount(int(compact_chain_type_mask[node]))
                + min(_popcount(int(compact_chain_neighbor_mask[node])), 8)
            )

        def decay_compact_chain_node(node: int, ts_value: int) -> None:
            if not compact_chain_enabled or node < 0 or node >= max_node_id:
                return
            last = int(compact_chain_last_ts[node])
            if last >= 0:
                dt = max(float(ts_value - last) / 1e9, 0.0)
                decay = math.exp(-dt / max(float(args.compact_chain_decay_sec), 1e-3))
                compact_chain_support[node] = np.float32(float(compact_chain_support[node]) * decay)
                compact_chain_bonus[node] = np.float32(float(compact_chain_bonus[node]) * decay)
            compact_chain_last_ts[node] = int(ts_value)
            if float(compact_chain_support[node]) <= 1e-4 and float(compact_chain_bonus[node]) <= 1e-4:
                compact_chain_rel_mask[node] = np.uint16(0)
                compact_chain_type_mask[node] = np.uint8(0)
                compact_chain_neighbor_mask[node] = np.uint32(0)
                if compact_chain_slots > 0:
                    compact_chain_slot_nodes[node, :] = -1
                    compact_chain_slot_scores[node, :] = 0.0
                    compact_chain_slot_ts[node, :] = -1

        def max_compact_neighbor_score(node: int, ts_value: int) -> float:
            if not compact_chain_enabled or compact_chain_slots <= 0 or node < 0 or node >= max_node_id:
                return 0.0
            best_score = 0.0
            for slot in range(compact_chain_slots):
                neighbor = int(compact_chain_slot_nodes[node, slot])
                raw_score = float(compact_chain_slot_scores[node, slot])
                raw_ts = int(compact_chain_slot_ts[node, slot])
                if neighbor < 0 or raw_score <= 0.0 or raw_ts < 0:
                    continue
                dt = max(float(ts_value - raw_ts) / 1e9, 0.0)
                decayed = raw_score * math.exp(-dt / max(float(args.compact_chain_decay_sec), 1e-3))
                if decayed > best_score:
                    best_score = float(decayed)
            return float(best_score)

        def refresh_compact_chain_bonus(node: int, ts_value: int) -> float:
            if not compact_chain_enabled or node < 0 or node >= max_node_id:
                return 0.0
            decay_compact_chain_node(node, ts_value)
            diversity = compact_chain_diversity(node)
            if diversity < int(args.compact_chain_min_diversity):
                return 0.0
            support_bonus = float(args.compact_chain_support_weight) * math.log1p(float(compact_chain_support[node]))
            diversity_bonus = float(args.compact_chain_diversity_weight) * math.log1p(float(diversity))
            neighbor_excess = max(
                max_compact_neighbor_score(node, ts_value) - float(prop_threshold) * float(args.compact_chain_update_ratio),
                0.0,
            )
            bonus = float(support_bonus + diversity_bonus + float(args.compact_chain_neighbor_weight) * neighbor_excess)
            if float(args.compact_chain_bonus_cap) > 0.0:
                bonus = min(bonus, float(args.compact_chain_bonus_cap))
            if bonus > float(compact_chain_bonus[node]):
                compact_chain_bonus[node] = np.float32(bonus)
            return float(compact_chain_bonus[node])

        def update_compact_slot(node: int, neighbor: int, score_value: float, ts_value: int) -> None:
            if not compact_chain_enabled or compact_chain_slots <= 0 or node < 0 or node >= max_node_id or neighbor < 0:
                return
            replace_slot = -1
            weakest_slot = 0
            weakest_score = float("inf")
            for slot in range(compact_chain_slots):
                current = int(compact_chain_slot_nodes[node, slot])
                raw_score = float(compact_chain_slot_scores[node, slot])
                raw_ts = int(compact_chain_slot_ts[node, slot])
                decayed = 0.0
                if raw_score > 0.0 and raw_ts >= 0:
                    dt = max(float(ts_value - raw_ts) / 1e9, 0.0)
                    decayed = raw_score * math.exp(-dt / max(float(args.compact_chain_decay_sec), 1e-3))
                if current == int(neighbor):
                    replace_slot = slot
                    break
                if current < 0:
                    replace_slot = slot
                    break
                if decayed < weakest_score:
                    weakest_score = float(decayed)
                    weakest_slot = slot
            if replace_slot < 0:
                replace_slot = weakest_slot
            compact_chain_slot_nodes[node, replace_slot] = int(neighbor)
            compact_chain_slot_scores[node, replace_slot] = np.float32(max(float(score_value), float(compact_chain_slot_scores[node, replace_slot])))
            compact_chain_slot_ts[node, replace_slot] = int(ts_value)

        def update_compact_chain_edge(left: int, right: int, rel: int, right_type: int, score_value: float, ts_value: int) -> None:
            if not compact_chain_enabled or left < 0 or left >= max_node_id or right < 0 or right >= max_node_id or left == right:
                return
            if float(score_value) < float(prop_threshold) * float(args.compact_chain_update_ratio):
                return
            decay_compact_chain_node(left, ts_value)
            increment = min(max(float(score_value) / max(float(prop_threshold), 1e-6), 0.0), float(args.compact_chain_support_cap))
            compact_chain_support[left] = np.float32(min(float(compact_chain_support[left]) + increment, float(args.compact_chain_support_cap)))
            rel_bit = 1 << max(min(int(rel), 15), 0)
            type_bit = 1 << max(min(int(right_type), 7), 0)
            compact_chain_rel_mask[left] = np.uint16(int(compact_chain_rel_mask[left]) | rel_bit)
            compact_chain_type_mask[left] = np.uint8(int(compact_chain_type_mask[left]) | type_bit)
            compact_chain_neighbor_mask[left] = np.uint32(int(compact_chain_neighbor_mask[left]) | compact_neighbor_bit(right))
            update_compact_slot(left, right, score_value, ts_value)
            refresh_compact_chain_bonus(left, ts_value)

        def spread_compact_chain_from(node: int, ts_value: int) -> None:
            if not compact_chain_enabled or compact_chain_slots <= 0 or node < 0 or node >= max_node_id:
                return
            seed_bonus = refresh_compact_chain_bonus(node, ts_value)
            if seed_bonus <= 0.0:
                return
            for slot in range(compact_chain_slots):
                neighbor = int(compact_chain_slot_nodes[node, slot])
                raw_score = float(compact_chain_slot_scores[node, slot])
                raw_ts = int(compact_chain_slot_ts[node, slot])
                if neighbor < 0 or neighbor >= max_node_id or raw_score <= 0.0 or raw_ts < 0:
                    continue
                dt = max(float(ts_value - raw_ts) / 1e9, 0.0)
                decayed_score = raw_score * math.exp(-dt / max(float(args.compact_chain_decay_sec), 1e-3))
                neighbor_bonus = max(decayed_score - float(prop_threshold) * float(args.compact_chain_update_ratio), 0.0)
                bonus = float(seed_bonus + float(args.compact_chain_neighbor_weight) * neighbor_bonus)
                if float(args.compact_chain_bonus_cap) > 0.0:
                    bonus = min(bonus, float(args.compact_chain_bonus_cap))
                if bonus > float(compact_chain_bonus[neighbor]):
                    decay_compact_chain_node(neighbor, ts_value)
                    compact_chain_bonus[neighbor] = np.float32(bonus)

        for row in _stream_events(conn, year_month, test_days, indexid2summary, hash2type, hash2uuid_index, abnormal_nodes, event_filter, int(args.fetch_size), int(args.max_test_events)):
            chain_details = {
                "score": 0.0,
                "association_discrepancy": 0.0,
                "association_score": 0.0,
                "transition_rarity": 0.0,
                "has_transition": 0.0,
            }
            chain_profile_score = 0.0
            chain_profile_z = 0.0
            chain_profile_tail = 0.0
            if profiling_enabled:
                _profile_t0 = time.perf_counter()
            if chain_profile is not None and chain_test_state is not None:
                chain_details = chain_profile.score_pre_update(row, chain_test_state)
                chain_profile_score = float(chain_details["score"])
                chain_profile_z = _upper_z_scalar(
                    chain_profile_score,
                    chain_profile_stats,
                    float(args.chain_profile_clip),
                )
                chain_profile_tail = _tail_surprise_scalar(
                    chain_profile_score,
                    chain_profile_calibration,
                    float(args.chain_profile_clip),
                )
            if profiling_enabled:
                profile_acc["chain_profile_seconds"] += float(time.perf_counter() - _profile_t0)
                _profile_t0 = time.perf_counter()
            z, aux, _pred = model.score_pre_update(row, states, last_seen, last_edge_seen, global_latent_state, last_global_ts)
            values = _component_values(aux)
            score = _fused_score(values, comp_stats, weights, float(args.score_clip))
            event_tail = _tail_surprise_scalar(score, fused_calibration, float(args.chain_profile_clip)) if chain_profile_enabled else 0.0
            if str(args.chain_profile_score_mode) == "conformal":
                chain_profile_bonus_value = float(chain_profile_tail)
            elif str(args.chain_profile_score_mode) == "agreement":
                chain_profile_bonus_value = float(min(chain_profile_tail, event_tail))
            else:
                chain_profile_bonus_value = float(chain_profile_z)
            component_scores = _component_z_values(aux, comp_stats, weights, float(args.score_clip))
            endpoint_bonus = float(component_scores[COMPONENTS.index("endpoint_self_rarity")])
            semantic_z = _upper_z_scalar(float(aux["semantic_max_rarity"]), comp_stats[COMPONENTS.index("semantic_max_rarity")], float(args.score_clip))
            malicious_chain_signal = (
                _compact_malicious_chain_signal(
                    event_tail,
                    semantic_z,
                    chain_profile_bonus_value,
                    chain_profile_score,
                    float(chain_details.get("transition_rarity", 0.0)),
                    float(chain_details.get("association_discrepancy", 0.0)),
                    float(chain_details.get("association_score", 0.0)),
                    float(chain_details.get("has_transition", 0.0)),
                )
                if bool(args.compact_malicious_chain_enabled)
                else 0.0
            )
            if profiling_enabled:
                profile_acc["event_scoring_seconds"] += float(time.perf_counter() - _profile_t0)
            adaptive_distance = 0.0
            adaptive_bonus_value = 0.0
            adaptive_has_proto = False
            adaptive_key = 0
            adaptive_assoc_z = 0.0
            adaptive_quarantined = False
            if profiling_enabled:
                _profile_t0 = time.perf_counter()
            if adaptive_memory_enabled:
                adaptive_distance, adaptive_has_proto, adaptive_key = _adaptive_memory_distance(
                    adaptive_test_memory,
                    row,
                    z,
                    int(args.adaptive_memory_min_count),
                )
                adaptive_bonus_value = (
                    _calibration_score_scalar(
                        adaptive_distance,
                        adaptive_memory_stats,
                        adaptive_memory_calibration,
                        float(args.adaptive_memory_clip),
                        str(args.adaptive_memory_calibration_mode),
                    )
                    if adaptive_has_proto
                    else 0.0
                )
                adaptive_assoc_z = _upper_z_scalar(
                    _adaptive_memory_assoc_value(chain_details),
                    chain_assoc_stats,
                    float(args.adaptive_memory_clip),
                )
                adaptive_quarantined = _adaptive_memory_quarantine(
                    event_tail,
                    chain_profile_tail,
                    adaptive_assoc_z,
                    float(args.adaptive_memory_quarantine_tail_min),
                    float(args.adaptive_memory_quarantine_assoc_z_min),
                )
                adaptive_test_scored += int(adaptive_has_proto)
            else:
                adaptive_assoc_z = _upper_z_scalar(
                    _adaptive_memory_assoc_value(chain_details),
                    chain_assoc_stats,
                    float(args.snc_clip),
                )
            if profiling_enabled:
                profile_acc["adaptive_memory_seconds"] += float(time.perf_counter() - _profile_t0)
            label = int(row["label"])
            alert = score >= threshold
            update_event_counts(event_counts_strict, event_counts_relaxed, label, alert)
            src = int(row["src_idx"])
            dst = int(row["dst_idx"])
            info_src = int(row["info_src"])
            info_dst = int(row["info_dst"])
            ts = int(row["timestamp_ns"])
            episode_support = 0.0
            episode_raw_support = 0.0
            if test_episode_sketch is not None:
                if profiling_enabled:
                    _profile_t0 = time.perf_counter()
                episode_details = test_episode_sketch.score_pre_update(
                    row,
                    ts,
                    float(event_tail),
                    float(chain_profile_bonus_value),
                    float(max(float(chain_details.get("association_score", 0.0)), float(chain_details.get("association_discrepancy", 0.0)), 0.0)),
                    float(chain_details.get("transition_rarity", 0.0)),
                    adaptive_support=float(max(adaptive_bonus_value, 0.0)),
                    quarantined=bool(adaptive_quarantined),
                    min_event_tail=float(args.causal_episode_min_event_tail),
                    min_chain=float(args.causal_episode_min_chain),
                    min_assoc=float(args.causal_episode_min_assoc),
                    min_families=int(args.causal_episode_min_families),
                    selective=bool(args.causal_episode_selective_enabled),
                )
                episode_support = float(episode_details["support"])
                episode_raw_support = float(episode_details["raw_support"])
                if profiling_enabled:
                    profile_acc["episode_sketch_seconds"] += float(time.perf_counter() - _profile_t0)
            if chain_profile_enabled and chain_profile_bonus_value > 0.0:
                for node in {src, dst, info_src, info_dst}:
                    if 0 <= int(node) < max_node_id and chain_profile_bonus_value > float(chain_profile_bonus[int(node)]):
                        chain_profile_bonus[int(node)] = np.float32(chain_profile_bonus_value)
                        chain_profile_raw[int(node)] = np.float32(chain_profile_score)
                        chain_profile_conformal[int(node)] = np.float32(chain_profile_tail)
                        chain_profile_event_surprise[int(node)] = np.float32(event_tail)
                        chain_profile_event_pos[int(node)] = int(test_processed)
            if adaptive_memory_enabled and adaptive_bonus_value > 0.0:
                for node in {src, dst, info_src, info_dst}:
                    if 0 <= int(node) < max_node_id and adaptive_bonus_value > float(adaptive_memory_bonus[int(node)]):
                        adaptive_memory_bonus[int(node)] = np.float32(adaptive_bonus_value)
                        adaptive_memory_raw[int(node)] = np.float32(adaptive_distance)
                        adaptive_memory_event_pos[int(node)] = int(test_processed)
                        adaptive_memory_quarantine_node[int(node)] = np.float32(1.0 if adaptive_quarantined else 0.0)
            if float(args.global_boost) > 0.0 and global_risk_last_ts >= 0 and global_risk_state > 0.0:
                global_risk_state *= math.exp(-max(float(ts - global_risk_last_ts) / 1e9, 0.0) / max(float(args.global_decay_sec), 1e-3))
            global_risk_last_ts = ts
            src_prop = _decayed_scalar_state(prop_state, prop_last_ts, info_src, ts, float(args.prop_decay_sec))
            if bool(args.undirected_propagation):
                dst_prop = _decayed_scalar_state(prop_state, prop_last_ts, info_dst, ts, float(args.prop_decay_sec))
                raw_src_prop = _decayed_scalar_state(prop_state, prop_last_ts, src, ts, float(args.prop_decay_sec))
                raw_dst_prop = _decayed_scalar_state(prop_state, prop_last_ts, dst, ts, float(args.prop_decay_sec))
                incoming_prop = max(src_prop, dst_prop, raw_src_prop, raw_dst_prop)
            else:
                incoming_prop = src_prop
            propagation_candidate = float(args.prop_boost) * float(incoming_prop)
            global_candidate = float(args.global_boost) * float(global_risk_state)
            propagated_score = max(float(score), propagation_candidate, global_candidate)
            intrinsic_local_score = max(float(score), float(endpoint_bonus))
            local_score = max(float(propagated_score), float(endpoint_bonus))
            top_component = int(np.argmax(component_scores))
            if max(propagation_candidate, global_candidate) > float(score):
                top_component = propagation_component
            for node in {src, dst, info_src, info_dst}:
                node_int = int(node)
                if 0 <= node_int < max_node_id and node_int in abnormal_nodes and node_int not in positive_node_first_event:
                    positive_node_first_event[node_int] = int(test_processed)
                    positive_node_first_ts[node_int] = int(ts)
                apply_node_evidence(node_int, float(local_score), int(label), int(top_component), update_membership=True, intrinsic_score=float(intrinsic_local_score))
            if online_risk_state is not None and online_alert_emitter is not None:
                if profiling_enabled:
                    _profile_t0 = time.perf_counter()
                chain_event_bonus = float(chain_profile_bonus_value) if chain_profile_enabled else 0.0
                adaptive_event_bonus = float(adaptive_bonus_value) if adaptive_memory_enabled else 0.0
                assoc_support = max(float(adaptive_assoc_z), 0.0)
                semantic_support = max(float(semantic_z), 0.0)
                local_chain_support = max(float(chain_profile_bonus_value), 0.0)
                episode_chain_support = float(episode_support)
                malicious_support = max(float(malicious_chain_signal), 0.0)
                for node in {src, dst, info_src, info_dst}:
                    node_int = int(node)
                    if node_int < 0 or node_int >= max_node_id:
                        continue
                    compact_event_bonus = float(compact_chain_bonus[node_int]) if compact_chain_enabled else 0.0
                    activity_chain_density_event = (
                        max(float(chain_profile_score), 0.0)
                        / max(1.0, math.log1p(max(float(node_high_count[node_int]), 0.0)))
                        if chain_profile_enabled
                        else 0.0
                    )
                    node_state = online_risk_state.update(
                        node=node_int,
                        ts=ts,
                        event_score=float(local_score),
                        intrinsic_score=float(intrinsic_local_score),
                        top_component=int(top_component),
                        intrinsic_high_count=float(node_intrinsic_high_count[node_int]),
                        intrinsic_soft_sum=float(node_intrinsic_soft_sum[node_int]),
                        activity_high_count=float(node_high_count[node_int]),
                        activity_soft_sum=float(node_soft_sum[node_int]),
                        compact_chain_bonus=compact_event_bonus,
                        chain_profile_bonus=chain_event_bonus,
                        adaptive_memory_bonus=adaptive_event_bonus,
                        association_support=assoc_support,
                        semantic_support=semantic_support,
                        local_chain_support=local_chain_support,
                        episode_support=episode_chain_support,
                        malicious_chain_support=malicious_support,
                    )
                    if profiling_enabled:
                        _profile_write_t0 = time.perf_counter()
                    online_alert_emitter.maybe_emit(
                        node_int,
                        ts,
                        test_processed,
                        node_state,
                        {
                            "event_score": float(local_score),
                            "intrinsic_event_score": float(intrinsic_local_score),
                            "event_tail": float(event_tail),
                            "semantic_z": float(semantic_z),
                            "association_discrepancy": float(chain_details.get("association_discrepancy", 0.0)),
                            "association_score": float(chain_details.get("association_score", 0.0)),
                            "compact_chain_bonus": float(compact_event_bonus),
                            "chain_profile_bonus": float(chain_event_bonus),
                            "activity_normalized_chain_density": float(activity_chain_density_event),
                            "episode_subgraph_support": float(episode_chain_support),
                            "adaptive_memory_bonus": float(adaptive_event_bonus),
                            "adaptive_quarantined": bool(adaptive_quarantined),
                            "activity_high_count": float(node_high_count[node_int]),
                            "activity_soft_sum": float(node_soft_sum[node_int]),
                            "src": int(src),
                            "dst": int(dst),
                            "info_src": int(info_src),
                            "info_dst": int(info_dst),
                            "relation_id": int(row["relation_id"]),
                        },
                    )
                    if profiling_enabled:
                        profile_acc["csv_write_seconds"] += float(time.perf_counter() - _profile_write_t0)
                if profiling_enabled:
                    profile_acc["online_alert_seconds"] += float(time.perf_counter() - _profile_t0)
            if bool(args.node_rerank_enabled):
                _update_node_rerank_extra(
                    node_rerank_extra,
                    {src, dst, info_src, info_dst},
                    info_src,
                    bool(propagation_candidate > float(score)),
                    semantic_z,
                    chain_profile_bonus_value,
                    chain_profile_score,
                    float(chain_details.get("association_discrepancy", 0.0)),
                    float(chain_details.get("association_score", 0.0)),
                    float(malicious_chain_signal),
                    int(row["relation_id"]),
                    int(row["info_src_type"]),
                    int(row["info_dst_type"]),
                    local_score,
                    max_node_id,
                )
            if test_episode_sketch is not None and episode_support > 0.0:
                if profiling_enabled:
                    _profile_t0 = time.perf_counter()
                test_episode_sketch.update_state(
                    row,
                    {src, dst, info_src, info_dst},
                    ts,
                    episode_support,
                    episode_raw_support,
                    int(row["relation_id"]),
                    int(row["info_src_type"]),
                    int(row["info_dst_type"]),
                    int(test_processed),
                )
                if profiling_enabled:
                    profile_acc["episode_sketch_seconds"] += float(time.perf_counter() - _profile_t0)
            if bool(args.snc_enabled):
                snc_support = _snc_event_support(
                    event_tail,
                    semantic_z,
                    chain_profile_bonus_value,
                    chain_profile_score,
                    adaptive_assoc_z,
                    adaptive_bonus_value,
                    adaptive_quarantined,
                )
                _update_snc_state(
                    snc_state,
                    info_src,
                    info_dst,
                    ts,
                    snc_support,
                    adaptive_assoc_z,
                    int(row["relation_id"]),
                    int(row["info_dst_type"]),
                    float(node_high_count[info_dst]) if 0 <= info_dst < max_node_id else 0.0,
                    adaptive_quarantined,
                    test_processed,
                    float(args.snc_decay_sec),
                    float(args.snc_update_min_support),
                    float(args.snc_update_min_assoc_z),
                )
                _update_snc_state(
                    snc_state,
                    src,
                    dst,
                    ts,
                    snc_support,
                    adaptive_assoc_z,
                    int(row["relation_id"]),
                    int(row["info_dst_type"]),
                    float(node_high_count[dst]) if 0 <= dst < max_node_id else 0.0,
                    adaptive_quarantined,
                    test_processed,
                    float(args.snc_decay_sec),
                    float(args.snc_update_min_support),
                    float(args.snc_update_min_assoc_z),
                )
            if alert:
                _set_decayed_state_max(prop_state, prop_last_ts, info_src, ts, float(score), float(args.prop_decay_sec))
                _set_decayed_state_max(prop_state, prop_last_ts, info_dst, ts, float(score), float(args.prop_decay_sec))
            if local_score >= prop_threshold * float(args.prop_update_threshold_ratio):
                _set_decayed_state_max(prop_state, prop_last_ts, info_dst, ts, float(local_score), float(args.prop_decay_sec))
                for seed_node in {src, dst, info_src, info_dst}:
                    backtrack_recent(int(seed_node), float(local_score), int(ts), int(label))
            if compact_chain_enabled:
                rel = int(row["relation_id"])
                update_compact_chain_edge(info_src, info_dst, rel, int(row["info_dst_type"]), float(local_score), ts)
                update_compact_chain_edge(info_dst, info_src, rel, int(row["info_src_type"]), float(local_score), ts)
                update_compact_chain_edge(src, dst, rel, int(row["info_dst_type"]), float(local_score), ts)
                update_compact_chain_edge(dst, src, rel, int(row["info_src_type"]), float(local_score), ts)
                for seed_node in {src, dst, info_src, info_dst}:
                    spread_compact_chain_from(int(seed_node), ts)
            if adaptive_memory_enabled:
                adaptive_proto_tail = _calibration_score_scalar(
                    adaptive_distance,
                    adaptive_memory_stats,
                    adaptive_memory_calibration,
                    float(args.adaptive_memory_clip),
                    str(args.adaptive_memory_calibration_mode),
                )
                if _adaptive_memory_can_update(
                    adaptive_proto_tail,
                    chain_profile_tail,
                    adaptive_assoc_z,
                    float(args.adaptive_memory_update_tail_max),
                    float(args.adaptive_memory_update_assoc_z_max),
                    adaptive_quarantined,
                ):
                    _adaptive_memory_update(adaptive_test_memory, adaptive_key, z, float(args.adaptive_memory_update_alpha))
                    adaptive_test_updates += 1
                    for node in {src, dst, info_src, info_dst}:
                        if 0 <= int(node) < max_node_id:
                            adaptive_memory_update_node[int(node)] += np.float32(1.0)
                elif adaptive_quarantined:
                    adaptive_test_memory["quarantined"] = int(adaptive_test_memory.get("quarantined", 0)) + 1
                    adaptive_test_quarantined += 1
            if alert:
                if float(args.global_boost) > 0.0:
                    global_risk_state = max(float(global_risk_state), float(score))
                for win in window_rows:
                    if (not win["detected"]) and int(win["start"]) <= ts <= int(win["end"]):
                        win["detected"] = True
                        win["ttd"] = max(float(ts - int(win["start"])) / 1e9, 0.0)
            remember_pair(info_src, info_dst, ts)
            remember_pair(info_dst, info_src, ts)
            remember_pair(src, dst, ts)
            remember_pair(dst, src, ts)
            if profiling_enabled:
                _profile_t0 = time.perf_counter()
            global_latent_state = model.update_states(row, z, states, last_seen, last_edge_seen, global_latent_state)
            last_global_ts = int(row["timestamp_ns"])
            if chain_profile is not None and chain_test_state is not None:
                chain_profile.update_state(row, chain_test_state)
            if profiling_enabled:
                profile_acc["state_update_seconds"] += float(time.perf_counter() - _profile_t0)
            test_processed += 1
            if profiling_enabled and profiling_interval > 0 and test_processed % profiling_interval == 0:
                elapsed = float(time.perf_counter() - phase_start)
                row_profile: dict[str, float | int] = {
                    "test_processed": int(test_processed),
                    "elapsed_seconds": elapsed,
                    "logs_per_second": float(test_processed / max(elapsed, 1e-12)),
                    "event_alerts": int(event_counts_strict["tp"] + event_counts_strict["fp"]),
                }
                for key, value in profile_acc.items():
                    delta = float(value - profile_last[key])
                    row_profile[key] = float(value)
                    row_profile[f"{key}_delta"] = float(delta)
                    profile_last[key] = float(value)
                profile_rows.append(row_profile)
                log(
                    "[TFLR-LowRank] test processed="
                    f"{test_processed} event_alerts={event_counts_strict['tp'] + event_counts_strict['fp']} "
                    f"lps={row_profile['logs_per_second']:.2f} "
                    f"event_scoring_delta={row_profile['event_scoring_seconds_delta']:.2f}s "
                    f"adaptive_delta={row_profile['adaptive_memory_seconds_delta']:.2f}s "
                    f"episode_delta={row_profile['episode_sketch_seconds_delta']:.2f}s "
                    f"online_delta={row_profile['online_alert_seconds_delta']:.2f}s "
                    f"csv_write_delta={row_profile['csv_write_seconds_delta']:.2f}s"
                )
                memory_samples.append({"phase": "test_inference", "test_processed": int(test_processed), **memory_snapshot()})
        if online_alert_emitter is not None:
            online_alert_emitter.close()
        if online_risk_state is not None:
            online_risk_state.close(last_global_ts if last_global_ts >= 0 else 0)
        test_seconds = float(time.perf_counter() - phase_start)
        memory_samples.append({"phase": "after_test_inference", "test_processed": int(test_processed), **memory_snapshot()})

        topk_values = [int(x.strip()) for x in str(args.topk_values).split(",") if x.strip()]
        target = target_for_dataset(args.dataset)
        node_score_without_chain = _node_score(
            node_max,
            node_high_count,
            node_soft_sum,
            float(args.node_count_weight),
            float(args.node_sum_weight),
        )
        node_score_intrinsic = _node_score(
            node_intrinsic_max,
            node_intrinsic_high_count,
            node_intrinsic_soft_sum,
            float(args.node_count_weight),
            float(args.node_sum_weight),
        )
        node_score = _node_score(
            node_max,
            node_high_count,
            node_soft_sum,
            float(args.node_count_weight),
            float(args.node_sum_weight),
            compact_chain_bonus if compact_chain_enabled else None,
            float(args.compact_chain_score_weight) if compact_chain_enabled else 0.0,
        )
        if chain_profile_enabled:
            node_score = node_score + float(args.chain_profile_score_weight) * chain_profile_bonus
            node_score_intrinsic = node_score_intrinsic + float(args.chain_profile_score_weight) * chain_profile_bonus
        if causal_episode_sketch is not None:
            node_score = node_score + float(args.chain_profile_score_weight) * causal_episode_sketch.node_support
            node_score_intrinsic = node_score_intrinsic + float(args.chain_profile_score_weight) * causal_episode_sketch.node_support
        if adaptive_memory_enabled:
            node_score = node_score + float(args.adaptive_memory_weight) * adaptive_memory_bonus
            node_score_intrinsic = node_score_intrinsic + float(args.adaptive_memory_weight) * adaptive_memory_bonus
        activity_only_pressure = np.zeros((max_node_id,), dtype=np.float32)
        if bool(args.node_rerank_joint_activity_pressure):
            compact_support_for_pressure = np.zeros((max_node_id,), dtype=np.float32)
            if chain_profile_enabled:
                compact_support_for_pressure = np.maximum(compact_support_for_pressure, chain_profile_bonus)
            if causal_episode_sketch is not None:
                compact_support_for_pressure = np.maximum(compact_support_for_pressure, causal_episode_sketch.node_support)
            if adaptive_memory_enabled:
                compact_support_for_pressure = np.maximum(
                    compact_support_for_pressure,
                    float(args.adaptive_memory_weight) * adaptive_memory_bonus,
                )
            if node_rerank_extra:
                compact_support_for_pressure = np.maximum.reduce(
                    [
                        compact_support_for_pressure,
                        node_rerank_extra["association_max"],
                        node_rerank_extra["local_chain_agreement"],
                        node_rerank_extra["malicious_chain_peak"],
                    ]
                ).astype(np.float32, copy=False)
            activity_raw = np.log1p(np.maximum(node_high_count, 0.0)) + np.log1p(np.maximum(node_soft_sum, 0.0))
            activity_only_pressure = float(args.online_node_activity_weight) * np.maximum(
                activity_raw - compact_support_for_pressure - float(args.online_node_activity_margin),
                0.0,
            )
            if float(args.online_node_activity_cap) > 0.0:
                activity_only_pressure = np.minimum(activity_only_pressure, float(args.online_node_activity_cap))
            node_score = (node_score - activity_only_pressure).astype(np.float32, copy=False)
            node_score_intrinsic = (node_score_intrinsic - activity_only_pressure).astype(np.float32, copy=False)
        node_score_before_rerank = node_score.copy()
        node_score_before_intrinsic_first = node_score_before_rerank
        node_rerank_active = 0
        node_rerank_max_delta = 0.0
        node_rerank_min_delta = 0.0
        node_intrinsic_first_applied = False
        node_hard_gate_count = 0
        node_hard_gate_max_penalty = 0.0
        rerank_keep = np.zeros((0,), dtype=np.int64)
        rerank_matrix = np.zeros((0, len(NODE_RERANK_FEATURES)), dtype=np.float32)
        if bool(args.node_rerank_enabled) and node_rerank_feature_stats:
            rerank_keep, rerank_matrix = _node_rerank_feature_matrix(
                all_nodes,
                node_max,
                node_high_count,
                node_soft_sum,
                chain_profile_bonus,
                chain_profile_raw,
                node_rerank_extra,
            )
            (
                rerank_delta,
                rerank_support,
                rerank_penalty,
                rerank_high_activity_penalty,
                hard_support,
                propagation_pressure,
                evidence_family_count,
                intrinsic_rank_support,
            ) = _node_rerank_delta(
                rerank_matrix,
                node_rerank_feature_stats,
                node_rerank_feature_calibrations,
                str(args.node_rerank_calibration_mode),
                node_rerank_feature_weights,
                float(args.node_rerank_clip),
                float(args.node_rerank_penalty_weight),
                float(args.node_rerank_margin),
                float(args.node_rerank_high_activity_weight),
                float(args.node_rerank_high_activity_margin),
                float(args.node_rerank_high_activity_cap),
                bool(args.node_rerank_joint_activity_pressure),
            )
            node_rerank_delta_values[rerank_keep] = rerank_delta
            node_rerank_support_values[rerank_keep] = rerank_support
            node_intrinsic_rank_support_values[rerank_keep] = intrinsic_rank_support
            node_rerank_penalty_values[rerank_keep] = rerank_penalty
            node_rerank_high_activity_penalty_values[rerank_keep] = rerank_high_activity_penalty
            if bool(args.node_hard_gate_enabled):
                node_hard_gate_support[rerank_keep] = hard_support
                node_hard_gate_prop_pressure[rerank_keep] = propagation_pressure
                node_hard_gate_family_count[rerank_keep] = evidence_family_count
            node_rerank_active = int(np.sum(np.abs(rerank_delta) > 0.0))
            node_rerank_max_delta = float(np.max(rerank_delta)) if rerank_delta.size else 0.0
            node_rerank_min_delta = float(np.min(rerank_delta)) if rerank_delta.size else 0.0
            if str(args.node_rerank_mode) == "replace":
                node_score = np.zeros_like(node_score)
                node_score[rerank_keep] = rerank_delta
            else:
                node_score = node_score + float(args.node_rerank_weight) * node_rerank_delta_values
            if bool(args.node_hard_gate_enabled):
                weak_support = (hard_support < float(args.node_hard_gate_min_support_z)) | (evidence_family_count < float(args.node_hard_gate_min_families))
                propagation_active = propagation_pressure >= float(args.node_hard_gate_prop_z)
                propagation_lift = node_score[rerank_keep] > (node_score_intrinsic[rerank_keep] + float(args.node_hard_gate_cap_margin))
                gate_mask = weak_support & propagation_active & propagation_lift
                gate_nodes = rerank_keep[gate_mask]
                if gate_nodes.size > 0:
                    support_gap = np.maximum(float(args.node_hard_gate_min_support_z) - hard_support[gate_mask], 0.0)
                    family_gap = np.maximum(float(args.node_hard_gate_min_families) - evidence_family_count[gate_mask], 0.0)
                    prop_gap = np.maximum(propagation_pressure[gate_mask] - float(args.node_hard_gate_prop_z), 0.0)
                    hard_penalty = float(args.node_hard_gate_penalty_weight) * (support_gap + family_gap + prop_gap)
                    capped = np.minimum(
                        node_score[gate_nodes],
                        node_score_intrinsic[gate_nodes] + float(args.node_hard_gate_cap_margin),
                    )
                    node_score[gate_nodes] = capped - hard_penalty.astype(np.float32, copy=False)
                    node_hard_gate_penalty[gate_nodes] = hard_penalty.astype(np.float32, copy=False)
                    node_hard_gate_applied[gate_nodes] = True
                    node_hard_gate_count = int(gate_nodes.size)
                    node_hard_gate_max_penalty = float(np.max(hard_penalty)) if hard_penalty.size else 0.0
        node_score_before_intrinsic_first = node_score.copy() if str(args.node_rank_mode) == "intrinsic_first" else node_score
        if str(args.node_rank_mode) == "intrinsic_first":
            propagation_secondary = np.maximum(node_score - node_score_intrinsic, 0.0)
            if float(args.node_intrinsic_tie_cap) > 0.0:
                propagation_secondary = np.minimum(propagation_secondary, float(args.node_intrinsic_tie_cap))
            intrinsic_support_term = (
                float(args.node_intrinsic_support_weight) * node_intrinsic_rank_support_values
                if bool(args.node_rerank_enabled)
                else 0.0
            )
            node_score = (
                node_score_intrinsic
                + intrinsic_support_term
                + float(args.node_intrinsic_tie_weight) * propagation_secondary
            ).astype(np.float32, copy=False)
            node_intrinsic_first_applied = True
        node_score_before_evidence_band = node_score.copy() if bool(args.evidence_band_gate_enabled) and node_rerank_feature_stats else node_score
        if bool(args.evidence_band_gate_enabled) and node_rerank_feature_stats:
            (
                node_score,
                evidence_band_arrays,
                evidence_band_meta,
            ) = _apply_evidence_band_gate(
                node_score=node_score,
                rerank_keep=rerank_keep,
                rerank_matrix=rerank_matrix,
                feature_stats=node_rerank_feature_stats,
                feature_calibrations=node_rerank_feature_calibrations,
                adaptive_memory_bonus=adaptive_memory_bonus,
                adaptive_memory_quarantine_node=adaptive_memory_quarantine_node,
                adaptive_memory_update_node=adaptive_memory_update_node,
                calibration_mode=str(args.evidence_band_calibration_mode),
                clip=float(args.node_rerank_clip),
                candidate_local=float(args.evidence_band_candidate_local),
                candidate_activity=float(args.evidence_band_candidate_activity),
                candidate_relation=float(args.evidence_band_candidate_relation),
                min_local=float(args.evidence_band_min_local),
                min_relation=float(args.evidence_band_min_relation),
                min_assoc=float(args.evidence_band_min_assoc),
                min_semantic=float(args.evidence_band_min_semantic),
                min_role=float(args.evidence_band_min_role),
                min_adaptive=float(args.evidence_band_min_adaptive),
                min_families=int(args.evidence_band_min_families),
                weight=float(args.evidence_band_weight),
                boost_cap=float(args.evidence_band_boost_cap),
                penalty_weight=float(args.evidence_band_penalty_weight),
                penalty_cap=float(args.evidence_band_penalty_cap),
                activity_weight=float(args.evidence_band_activity_weight),
                activity_margin=float(args.evidence_band_activity_margin),
                update_min_count=float(args.evidence_band_update_min_count),
                update_weight=float(args.evidence_band_update_weight),
                update_cap=float(args.evidence_band_update_cap),
                prop_weight=float(args.evidence_band_prop_weight),
                prop_cap=float(args.evidence_band_prop_cap),
            )
        node_score_before_snc = node_score.copy() if bool(args.snc_enabled) and snc_feature_stats else node_score
        snc_keep = np.zeros((0,), dtype=np.int64)
        snc_matrix = np.zeros((0, len(SNC_FEATURES)), dtype=np.float32)
        if bool(args.snc_enabled) and snc_feature_stats:
            snc_keep, snc_matrix = _snc_feature_matrix(all_nodes, snc_state)
            ranked_before_snc = rank_scores(node_score)
            (
                node_score,
                snc_arrays,
                snc_meta,
            ) = _apply_snc_score(
                node_score=node_score,
                ranked_before_snc=ranked_before_snc,
                snc_keep=snc_keep,
                snc_matrix=snc_matrix,
                snc_stats=snc_feature_stats,
                snc_calibrations=snc_feature_calibrations,
                node_high_count=node_high_count,
                adaptive_memory_update_node=adaptive_memory_update_node if adaptive_memory_enabled else np.zeros_like(node_high_count),
                calibration_mode=str(args.snc_calibration_mode),
                clip=float(args.snc_clip),
                candidate_start_topk=int(args.snc_candidate_start_topk),
                candidate_end_topk=int(args.snc_candidate_end_topk),
                weight=float(args.snc_weight),
                boost_cap=float(args.snc_boost_cap),
                penalty_weight=float(args.snc_penalty_weight),
                penalty_cap=float(args.snc_penalty_cap),
                min_cluster_z=float(args.snc_min_cluster_z),
                min_object_z=float(args.snc_min_object_z),
                min_diversity_z=float(args.snc_min_diversity_z),
                high_count_penalty_min=float(args.snc_high_count_penalty_min),
                benign_update_penalty_min=float(args.snc_benign_update_penalty_min),
                benign_update_weight=float(args.snc_benign_update_weight),
            )
        node_score_before_local_closure = node_score.copy() if bool(args.local_closure_enabled) and node_rerank_feature_stats else node_score
        if bool(args.local_closure_enabled) and node_rerank_feature_stats:
            ranked_before_closure = rank_scores(node_score)
            (
                node_score,
                local_closure_arrays,
                local_closure_meta,
            ) = _apply_local_closure_score(
                node_score=node_score,
                ranked_before_closure=ranked_before_closure,
                rerank_keep=rerank_keep,
                rerank_matrix=rerank_matrix,
                feature_stats=node_rerank_feature_stats,
                adaptive_memory_bonus=adaptive_memory_bonus,
                adaptive_memory_quarantine_node=adaptive_memory_quarantine_node,
                adaptive_memory_update_node=adaptive_memory_update_node,
                clip=float(args.node_rerank_clip),
                protect_topk=int(args.local_closure_protect_topk),
                end_topk=int(args.local_closure_end_topk),
                min_score_z=float(args.local_closure_min_score_z),
                min_families=int(args.local_closure_min_families),
                min_assoc_z=float(args.local_closure_min_assoc_z),
                min_role_z=float(args.local_closure_min_role_z),
                min_adaptive_z=float(args.local_closure_min_adaptive_z),
                weight=float(args.local_closure_weight),
                boost_cap=float(args.local_closure_boost_cap),
                penalty_weight=float(args.local_closure_penalty_weight),
                penalty_cap=float(args.local_closure_penalty_cap),
                prop_penalty_weight=float(args.local_closure_prop_penalty_weight),
                prop_penalty_cap=float(args.local_closure_prop_penalty_cap),
                update_min_count=float(args.local_closure_update_min_count),
                update_weight=float(args.local_closure_update_weight),
                update_cap=float(args.local_closure_update_cap),
                require_quarantine=bool(int(args.local_closure_require_quarantine)),
            )
        node_score_before_mid_rank_gate = node_score.copy() if bool(args.mid_rank_gate_enabled) and node_rerank_feature_stats else node_score
        if bool(args.mid_rank_gate_enabled) and node_rerank_feature_stats:
            ranked_before_mid_gate = rank_scores(node_score)
            (
                node_score,
                mid_rank_gate_arrays,
                mid_rank_gate_meta,
            ) = _apply_mid_rank_gate(
                node_score=node_score,
                ranked_before_gate=ranked_before_mid_gate,
                rerank_keep=rerank_keep,
                rerank_matrix=rerank_matrix,
                feature_stats=node_rerank_feature_stats,
                adaptive_memory_bonus=adaptive_memory_bonus,
                adaptive_memory_quarantine_node=adaptive_memory_quarantine_node,
                adaptive_memory_update_node=adaptive_memory_update_node,
                clip=float(args.node_rerank_clip),
                protect_topk=int(args.mid_rank_gate_protect_topk),
                end_topk=int(args.mid_rank_gate_end_topk),
                min_assoc_z=float(args.mid_rank_gate_min_assoc_z),
                min_relation_z=float(args.mid_rank_gate_min_relation_z),
                min_adaptive_z=float(args.mid_rank_gate_min_adaptive_z),
                require_quarantine=bool(int(args.mid_rank_gate_require_quarantine)),
                penalty_weight=float(args.mid_rank_gate_penalty_weight),
                penalty_cap=float(args.mid_rank_gate_penalty_cap),
                benign_update_min_count=float(args.mid_rank_gate_benign_update_min_count),
                benign_update_weight=float(args.mid_rank_gate_benign_update_weight),
                benign_update_cap=float(args.mid_rank_gate_benign_update_cap),
            )
        activity_normalized_chain_density_values = np.zeros((0,), dtype=np.float32)
        node_score_before_activity_normalized_chain = node_score.copy() if bool(args.activity_normalized_chain_enabled) else node_score
        if bool(args.activity_normalized_chain_enabled):
            ranked_before_activity_chain = rank_scores(node_score)
            (
                node_score,
                activity_normalized_chain_arrays,
                activity_normalized_chain_meta,
            ) = _apply_activity_normalized_chain_score(
                node_score=node_score,
                ranked_before_activity=ranked_before_activity_chain,
                chain_profile_raw=chain_profile_raw,
                node_high_count=node_high_count,
                density_stats=activity_normalized_chain_density_stats,
                density_calibration=activity_normalized_chain_density_calibration,
                calibration_mode=str(args.activity_normalized_chain_calibration_mode),
                clip=float(args.node_rerank_clip),
                start_topk=int(args.activity_normalized_chain_start_topk),
                end_topk=int(args.activity_normalized_chain_end_topk),
                min_z=float(args.activity_normalized_chain_min_z),
                weight=float(args.activity_normalized_chain_weight),
                boost_cap=float(args.activity_normalized_chain_boost_cap),
            )
            if activity_normalized_chain_arrays["density"].size > 0:
                activity_normalized_chain_density_values = activity_normalized_chain_arrays["density"]
        ranked = rank_scores(node_score)
        cache_path = ""
        cache_written = False
        cache_error = ""
        if bool(args.node_postprocess_cache_enabled):
            cache_path = str(args.node_postprocess_cache_path).strip() or os.path.join(result_dir, "node_postprocess_cache.npz")
            try:
                os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
                np.savez_compressed(
                    cache_path,
                    all_nodes=all_nodes.astype(bool, copy=False),
                    positive_nodes=positive_nodes.astype(bool, copy=False),
                    suspect_nodes=suspect_nodes.astype(bool, copy=False),
                    node_score_final=node_score.astype(np.float32, copy=False),
                    node_score_before_rerank=node_score_before_rerank.astype(np.float32, copy=False),
                    node_score_before_intrinsic_first=node_score_before_intrinsic_first.astype(np.float32, copy=False),
                    node_score_before_evidence_band=node_score_before_evidence_band.astype(np.float32, copy=False),
                    node_score_before_snc=node_score_before_snc.astype(np.float32, copy=False),
                    node_score_before_local_closure=node_score_before_local_closure.astype(np.float32, copy=False),
                    node_score_before_mid_rank_gate=node_score_before_mid_rank_gate.astype(np.float32, copy=False),
                    node_score_before_activity_normalized_chain=node_score_before_activity_normalized_chain.astype(np.float32, copy=False),
                    node_score_intrinsic=node_score_intrinsic.astype(np.float32, copy=False),
                    node_score_without_chain=node_score_without_chain.astype(np.float32, copy=False),
                    node_max=node_max.astype(np.float32, copy=False),
                    node_high_count=node_high_count.astype(np.float32, copy=False),
                    node_soft_sum=node_soft_sum.astype(np.float32, copy=False),
                    chain_profile_bonus=chain_profile_bonus.astype(np.float32, copy=False),
                    chain_profile_raw=chain_profile_raw.astype(np.float32, copy=False),
                    adaptive_memory_bonus=adaptive_memory_bonus.astype(np.float32, copy=False),
                    adaptive_memory_quarantine_node=adaptive_memory_quarantine_node.astype(np.float32, copy=False),
                    adaptive_memory_update_node=adaptive_memory_update_node.astype(np.float32, copy=False),
                    rerank_keep=rerank_keep.astype(np.int64, copy=False),
                    rerank_matrix=rerank_matrix.astype(np.float32, copy=False),
                    node_rerank_delta=node_rerank_delta_values.astype(np.float32, copy=False),
                    node_rerank_support=node_rerank_support_values.astype(np.float32, copy=False),
                    node_intrinsic_rank_support=node_intrinsic_rank_support_values.astype(np.float32, copy=False),
                    node_rerank_prop_only_penalty=node_rerank_penalty_values.astype(np.float32, copy=False),
                    node_rerank_high_activity_penalty=node_rerank_high_activity_penalty_values.astype(np.float32, copy=False),
                    activity_only_pressure=activity_only_pressure.astype(np.float32, copy=False),
                    activity_normalized_chain_density=activity_normalized_chain_density_values.astype(np.float32, copy=False),
                    activity_normalized_chain_candidate=activity_normalized_chain_arrays["candidate"].astype(bool, copy=False),
                    activity_normalized_chain_applied=activity_normalized_chain_arrays["applied"].astype(bool, copy=False),
                    activity_normalized_chain_z=activity_normalized_chain_arrays["score_z"].astype(np.float32, copy=False),
                    activity_normalized_chain_boost=activity_normalized_chain_arrays["boost"].astype(np.float32, copy=False),
                    node_top_event=node_top_event.astype(np.int64, copy=False),
                    node_top_label=node_top_label.astype(np.int8, copy=False),
                    node_top_component=node_top_component.astype(np.int16, copy=False),
                )
                cache_written = True
            except Exception as exc:  # pragma: no cover - diagnostic output only
                cache_error = f"{type(exc).__name__}: {exc}"
        sweep = _node_sweep(node_score, all_nodes, positive_nodes, suspect_nodes, topk_values)
        best = best_sweep_row(sweep, target)
        best_fp_constrained = best_under_fp_target(sweep, target)
        compact_active = int(np.sum(compact_chain_bonus > 0.0)) if compact_chain_enabled else 0
        compact_max_bonus = float(np.max(compact_chain_bonus)) if compact_active > 0 else 0.0
        chain_profile_active = int(np.sum(chain_profile_bonus > 0.0)) if chain_profile_enabled else 0
        chain_profile_max_bonus = float(np.max(chain_profile_bonus)) if chain_profile_active > 0 else 0.0
        detected = sum(1 for win in window_rows if win["detected"])
        ttds = [float(win["ttd"]) for win in window_rows if win["ttd"] is not None]
        attack_metrics = {
            "num_windows": len(window_rows),
            "detected_windows": int(detected),
            "attack_window_recall": float(detected / max(len(window_rows), 1)),
            "mean_time_to_detect_seconds": float(np.mean(ttds)) if ttds else float("nan"),
            "windows": [
                {"name": win["name"], "detected": bool(win["detected"]), "time_to_detect_seconds": win["ttd"]}
                for win in window_rows
            ],
        }
        observed_gt_abnormal_nodes = {
            int(node)
            for node in abnormal_nodes
            if 0 <= int(node) < max_node_id and bool(all_nodes[int(node)])
        }
        unobserved_gt_abnormal_nodes = {int(node) for node in abnormal_nodes} - observed_gt_abnormal_nodes
        strict_event_metrics = event_metrics(event_counts_strict)
        relaxed_event_metrics = event_metrics(event_counts_relaxed)
        model_meta = model.metadata()
        chain_profile_meta = chain_profile.metadata() if chain_profile is not None else {}
        adaptive_memory_model_size_bytes = int(
            np.asarray(adaptive_memory["mean"], dtype=np.float32).nbytes
            + np.asarray(adaptive_memory["count"], dtype=np.int32).nbytes
        ) if adaptive_memory_enabled else 0
        node_rerank_calibration_size_bytes = int(
            len(node_rerank_feature_stats) * 6 * 4
            + sum(_calibration_size_bytes(cal) for cal in node_rerank_feature_calibrations)
            + node_rerank_feature_weights.size * 4
            + 7 * 4
        ) if bool(args.node_rerank_enabled) else 0
        snc_calibration_size_bytes = int(
            len(snc_feature_stats) * 6 * 4
            + sum(_calibration_size_bytes(cal) for cal in snc_feature_calibrations)
            + len(SNC_FEATURES) * 4
        ) if bool(args.snc_enabled) else 0
        activity_normalized_chain_calibration_size_bytes = int(
            6 * 4 + _calibration_size_bytes(activity_normalized_chain_density_calibration)
        ) if bool(args.activity_normalized_chain_enabled) else 0
        calibration_size_bytes = (
            _calibration_size_bytes(fused_calibration)
            + _calibration_size_bytes(chain_profile_calibration)
            + node_rerank_calibration_size_bytes
            + snc_calibration_size_bytes
            + activity_normalized_chain_calibration_size_bytes
            + (6 * 4 if adaptive_memory_enabled else 0)
        )
        stored_model_size_bytes = (
            int(model_meta["model_size_bytes"])
            + int(chain_profile_meta.get("model_size_bytes", 0))
            + int(adaptive_memory_model_size_bytes)
            + int(calibration_size_bytes)
        )
        stored_model_size_mb = float(stored_model_size_bytes / (1024.0 * 1024.0))
        runtime_state_breakdown_bytes = {
            "propagation_state": _arrays_nbytes(prop_state, prop_last_ts, recent_neighbor, recent_neighbor_ts),
            "node_activity_scores": _arrays_nbytes(
                node_max,
                node_high_count,
                node_soft_sum,
                node_intrinsic_max,
                node_intrinsic_high_count,
                node_intrinsic_soft_sum,
                node_top_event,
                node_top_label,
                node_top_component,
            ),
            "compact_chain_state": _arrays_nbytes(
                compact_chain_bonus,
                compact_chain_support,
                compact_chain_last_ts,
                compact_chain_rel_mask,
                compact_chain_type_mask,
                compact_chain_neighbor_mask,
                compact_chain_slot_nodes,
                compact_chain_slot_scores,
                compact_chain_slot_ts,
            ),
            "chain_profile_node_state": _arrays_nbytes(
                chain_profile_bonus,
                chain_profile_raw,
                chain_profile_conformal,
                chain_profile_event_surprise,
                chain_profile_event_pos,
            ),
            "causal_episode_state": int(causal_episode_sketch.memory_bytes() if causal_episode_sketch is not None else 0),
            "adaptive_memory_node_state": _arrays_nbytes(
                adaptive_memory_bonus,
                adaptive_memory_raw,
                adaptive_memory_event_pos,
                adaptive_memory_quarantine_node,
                adaptive_memory_update_node,
            ),
            "adaptive_memory_test_prototypes": int(
                (np.asarray(adaptive_test_memory["mean"], dtype=np.float32).nbytes if adaptive_memory_enabled else 0)
                + (np.asarray(adaptive_test_memory["count"], dtype=np.int32).nbytes if adaptive_memory_enabled else 0)
            ),
            "node_rerank_feature_state": _array_map_nbytes(node_rerank_extra),
            "node_rerank_outputs": _arrays_nbytes(
                node_rerank_delta_values,
                node_rerank_support_values,
                node_intrinsic_rank_support_values,
                node_rerank_penalty_values,
                node_rerank_high_activity_penalty_values,
            ),
            "postprocess_snapshots": _unique_arrays_nbytes(
                node_score_before_rerank,
                node_score_before_intrinsic_first,
                node_score_before_evidence_band,
                node_score_before_snc,
                node_score_before_local_closure,
                node_score_before_mid_rank_gate,
                node_score_before_activity_normalized_chain,
            ),
            "diagnostic_gate_state": int(
                _array_map_nbytes(evidence_band_arrays)
                + _array_map_nbytes(snc_state)
                + _array_map_nbytes(snc_arrays)
                + _array_map_nbytes(local_closure_arrays)
                + _array_map_nbytes(mid_rank_gate_arrays)
                + _array_map_nbytes(activity_normalized_chain_arrays)
                + (
                    0
                    if activity_normalized_chain_arrays["density"].size > 0
                    else _arrays_nbytes(activity_normalized_chain_density_values)
                )
                + _arrays_nbytes(
                    node_hard_gate_support,
                    node_hard_gate_prop_pressure,
                    node_hard_gate_family_count,
                    node_hard_gate_penalty,
                    node_hard_gate_applied,
                )
            ),
            "online_alert_state": int(
                (online_risk_state.memory_bytes() if online_risk_state is not None else 0)
                + (online_alert_emitter.alerted.nbytes if online_alert_emitter is not None else 0)
            ),
        }
        runtime_state_bytes = int(sum(runtime_state_breakdown_bytes.values()))
        total_seconds = float(time.perf_counter() - run_start)
        out_json = os.path.join(result_dir, "eval_tflr_lowrank.json")
        ranked_csv = os.path.join(result_dir, "eval_tflr_lowrank_ranked.csv")
        online_alert_prefix_sweep_csv = os.path.join(result_dir, "eval_online_node_alert_prefix_sweep.csv")
        profiling_csv = os.path.join(result_dir, "profiling_checkpoints.csv")
        memory_samples_csv = os.path.join(result_dir, "memory_samples.csv")
        _write_memory_samples_csv(memory_samples_csv, memory_samples)
        memory = {
            "unit": "MB",
            "definition": "RSS is process resident memory. Samples are captured at phase boundaries and at profiling checkpoints when profiling is enabled.",
            "samples": memory_samples,
            "samples_csv": memory_samples_csv if memory_samples else "",
            "peak_rss_mb": float(max(float(sample["peak_rss_mb"]) for sample in memory_samples)) if memory_samples else 0.0,
            "max_current_rss_mb": float(max(float(sample["current_rss_mb"]) for sample in memory_samples)) if memory_samples else 0.0,
            "stored_model_mb": float(stored_model_size_mb),
            "stored_model_bytes": int(stored_model_size_bytes),
            "runtime_state_mb": float(runtime_state_bytes / (1024.0 * 1024.0)),
            "runtime_state_bytes": int(runtime_state_bytes),
            "runtime_state_breakdown_bytes": {key: int(value) for key, value in runtime_state_breakdown_bytes.items()},
            "runtime_state_breakdown_mb": {key: _bytes_mb(int(value)) for key, value in runtime_state_breakdown_bytes.items()},
        }
        online_node_alert_performance, online_alert_prefix_sweep = _online_alert_performance(
            online_alert_emitter,
            max_node_id,
            all_nodes,
            positive_nodes,
            suspect_nodes,
            positive_node_first_event,
            positive_node_first_ts,
            topk_values,
            target,
        )
        if bool(args.online_node_alerts_enabled):
            _write_online_alert_prefix_sweep_csv(online_alert_prefix_sweep_csv, online_alert_prefix_sweep)
        if profile_rows:
            with open(profiling_csv, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(profile_rows[0].keys()))
                writer.writeheader()
                writer.writerows(profile_rows)
        summary = {
            "dataset": args.dataset,
            "method": "tflr_lowrank_leakage_free",
            "config_profile": {
                "entry": "tflr_cssm_unified" if bool(getattr(args, "cssm_unified_entry", False)) else "tflr_lowrank",
                "profile": str(getattr(args, "cssm_config_profile", "custom")),
                "parameter_policy": str(getattr(args, "cssm_parameter_policy", "")),
                "paper_parameters": getattr(args, "cssm_paper_parameters", {}),
                "implementation_defaults": getattr(args, "cssm_implementation_defaults", {}),
                "fixed_main_components": getattr(args, "cssm_fixed_main_components", {}),
                "override_note": "Command-line overrides are allowed for diagnostics/deployment. They should not be tuned per test dataset or reported as dataset-specific method choices.",
            },
            "leakage_free_protocol": {
                "train_uses_ground_truth": False,
                "validation_uses_ground_truth": False,
                "handcrafted_rescue_enabled": False,
                "test_ground_truth_usage": "evaluation_only",
            },
            "counts": counts_meta,
            "train_summary": train_summary,
            "train_processed": int(train_summary["processed_events"]),
            "num_profile_events": int(model.num_profile_events),
            "ref_processed": int(ref_processed),
            "test_processed": int(test_processed),
            "model_size_bytes": int(stored_model_size_bytes),
            "model_size_mb": float(stored_model_size_mb),
            "under_5mb": bool(int(stored_model_size_bytes) < 5 * 1024 * 1024),
            "under_10mb": bool(int(stored_model_size_bytes) < 10 * 1024 * 1024),
            "model_statistics": {
                **model_meta,
                "low_rank_model_size_bytes": int(model_meta["model_size_bytes"]),
                "low_rank_model_size_mb": float(model_meta["model_size_mb"]),
                "chain_profile_model": chain_profile_meta,
                "adaptive_memory_model_size_bytes": int(adaptive_memory_model_size_bytes),
                "adaptive_memory_model_size_mb": float(adaptive_memory_model_size_bytes / (1024.0 * 1024.0)),
                "calibration_size_bytes": int(calibration_size_bytes),
                "total_stored_model_size_bytes": int(stored_model_size_bytes),
                "total_stored_model_size_mb": float(stored_model_size_mb),
                "runtime_state_bytes": int(runtime_state_bytes),
                "runtime_state_mb": float(runtime_state_bytes / (1024.0 * 1024.0)),
                "low_rank_formula": "z_hat = ((context - mean) / std @ A) @ B + bias",
                "predictor_training_mode": str(model_meta.get("predictor_training_mode", "")),
                "direct_lowrank_enabled": bool(args.direct_lowrank_enabled),
                "direct_lowrank_iters": int(args.direct_lowrank_iters),
                "direct_lowrank_l2": float(args.direct_lowrank_l2),
                "components": COMPONENTS,
                "singular_values": model.singular_values.tolist() if model.singular_values is not None else [],
                "semantic_singular_values": model.semantic_singular_values.tolist() if model.semantic_singular_values is not None else [],
            },
            "runtime": {
                "total_seconds": total_seconds,
                "metadata_seconds": float(metadata_seconds),
                "train_fit_seconds": float(train_seconds),
                "validation_seconds": float(validation_seconds),
                "test_inference_seconds": float(test_seconds),
                "test_logs_per_second": float(test_processed / max(test_seconds, 1e-12)),
                "fetch_size": int(args.fetch_size),
                "profiling_enabled": bool(profiling_enabled),
                "profiling_interval_events": int(profiling_interval),
                "profiling": {
                    **{key: float(value) for key, value in profile_acc.items()},
                    "enabled": bool(profiling_enabled),
                    "checkpoints": profile_rows,
                    "checkpoints_csv": profiling_csv if profile_rows else "",
                    "definition": "Optional phase counters sampled during streaming test inference. When disabled, only aggregate runtime is recorded to avoid per-event timer overhead. When enabled, times are additive instrumentation around event scoring, AdaptiveMemory, episode sketch, online alert risk/update, alert CSV write calls, and causal state updates.",
                },
            },
            "memory": memory,
            "inference_mode": "streaming_event_by_event_low_rank_residual_node_aggregation",
            "metric_definitions": {
                "event_at_threshold_strict": "Leakage-free fused score >= event_threshold. Label 2 is positive; labels 0 and 1 are negative.",
                "event_at_threshold_relaxed": "Same event threshold, but label 1 suspect events are ignored and not counted as FP.",
                "online_node_alerts": "Stopping-time alerts written during streaming inference when current node risk crosses a train/validation-calibrated threshold. These alerts are the real-time node-alerting output and contain no test labels at emission time. Their performance is evaluated only after inference under online_node_alert_performance.",
                "node_metrics": "Forensic/evaluation-only top-k node ranking computed after the full test stream from node_score. It must not be described as real-time node alerting. Node TP/FN denominator is the subset of GT abnormal nodes observed in the scored test stream under the configured time/event filters. Unobserved GT nodes are not counted as FN. Relaxed node metrics count observed GT abnormal endpoints as positives and ignore benign endpoints from suspect events.",
            },
            "performance_outputs": {
                "online_realtime_alerting": {
                    "role": "real_time_stopping_time_node_alerting",
                    "json_key": "online_node_alert_performance",
                    "alerts_csv": online_alert_path if bool(args.online_node_alerts_enabled) else "",
                    "prefix_sweep_csv": online_alert_prefix_sweep_csv if bool(args.online_node_alerts_enabled) else "",
                    "threshold_source": str(online_node_alert_threshold_source),
                    "definition": "Primary real-time alerting performance. Alerts are emitted online; GT is applied only afterward for metrics.",
                },
                "final_forensic_ranking": {
                    "role": "post_stream_forensic_ranking",
                    "json_keys": ["sweep", "best", "best_under_fp_target"],
                    "ranked_csv": ranked_csv,
                    "definition": "Retrospective final ranking after the test stream. Useful for forensic review and diagnosis, not a real-time alert claim.",
                },
            },
            "node_population": {
                "all_observed_nodes": int(np.sum(all_nodes)),
                "gt_abnormal_nodes_loaded": int(len(abnormal_nodes)),
                "observed_gt_abnormal_nodes": int(len(observed_gt_abnormal_nodes)),
                "unobserved_gt_abnormal_nodes_not_counted_as_fn": int(len(unobserved_gt_abnormal_nodes)),
                "suspect_only_observed_nodes": int(np.sum(suspect_nodes & ~positive_nodes & all_nodes)),
                "definition": "Observed means the node appeared in at least one scored test event endpoint after the configured split, time, and event filters. Only observed GT abnormal nodes can contribute TP/FN.",
            },
            "node_postprocess_cache": {
                "enabled": bool(args.node_postprocess_cache_enabled),
                "written": bool(cache_written),
                "path": cache_path,
                "error": cache_error,
                "definition": "Optional post-inference cache for fast diagnostic reranking. It stores node-level streaming outputs, validation-calibrated rerank features, and evaluation masks after inference has completed. The cache is disabled by default and does not change scoring.",
                "leakage_check": "The cache is written only after streaming test inference and stores already-computed node evidence plus masks for evaluation. It is intended for offline diagnostics; any scoring script must not use positive/suspect masks before ranking.",
            },
            "precompute_cache": {
                "enabled": bool(precompute_cache_enabled),
                "mode": str(args.precompute_cache_mode),
                "loaded": bool(precompute_cache_loaded),
                "written": bool(precompute_cache_written),
                "path": precompute_cache_path if precompute_cache_enabled else "",
                "fingerprint": precompute_cache_fingerprint if precompute_cache_enabled else "",
                "error": precompute_cache_error,
                "definition": "Optional train/validation precompute cache. It stores the fitted low-rank model, benign chain profile, validation calibrations, AdaptiveMemory initial state, and node-rerank validation statistics so later diagnostic runs can rescore only the test stream under an identical configuration fingerprint.",
                "leakage_check": "The precompute cache is produced only from train and validation splits. It contains no GT nodes, attack windows, test labels, test rankings, full-test statistics, or attack-specific tokens.",
            },
            "online_node_alerts": {
                "enabled": bool(args.online_node_alerts_enabled),
                "path": online_alert_path if bool(args.online_node_alerts_enabled) else "",
                "prefix_sweep_path": online_alert_prefix_sweep_csv if bool(args.online_node_alerts_enabled) else "",
                "alerts_emitted": int(online_alert_emitter.count) if online_alert_emitter is not None else 0,
                "threshold": float(online_node_alert_threshold),
                "threshold_source": str(online_node_alert_threshold_source),
                "threshold_quantile": float(args.online_node_alert_threshold_quantile),
                "risk_decay_sec": float(args.online_node_risk_decay_sec),
                "active_node_budget": int(args.active_node_budget),
                "edge_sketch_size": int(args.identity_buckets),
                "chain_sketch_size": int(args.chain_profile_buckets),
                "memory_prototype_count": int(args.adaptive_memory_buckets),
                "node_state_ttl_sec": float(args.node_state_ttl_sec),
                "node_state_maintenance_interval": int(args.node_state_maintenance_interval),
                "node_state_eviction_batch": int(args.node_state_eviction_batch),
                "active_nodes_end": int(online_risk_state.active_count) if online_risk_state is not None else 0,
                "peak_active_nodes": int(online_risk_state.peak_active_count) if online_risk_state is not None else 0,
                "evicted_nodes": int(online_risk_state.evicted_count) if online_risk_state is not None else 0,
                "ttl_evicted_nodes": int(online_risk_state.ttl_evicted_count) if online_risk_state is not None else 0,
                "budget_evicted_nodes": int(online_risk_state.budget_evicted_count) if online_risk_state is not None else 0,
                "maintenance_runs": int(online_risk_state.maintenance_runs) if online_risk_state is not None else 0,
                "risk_state_updates": int(online_risk_state.update_count) if online_risk_state is not None else 0,
                "max_budget_overflow": int(online_risk_state.max_budget_overflow) if online_risk_state is not None else 0,
                "activity_weight": float(args.online_node_activity_weight),
                "activity_margin": float(args.online_node_activity_margin),
                "activity_cap": float(args.online_node_activity_cap),
                "state_memory_bytes": int(online_risk_state.memory_bytes()) if online_risk_state is not None else 0,
                "definition": "Optional online stopping-time node alert output. For each streamed test event, node risk R_t(v)=intrinsic residual + compact non-semantic causal evidence + optional bounded episode/subgraph support + adaptive-memory shift - activity-only pressure is updated immediately for event endpoints; one alert is written as soon as the risk crosses the validation-calibrated threshold. Semantic rarity contributes to intrinsic residual but is not counted as compact causal support. The emitted CSV records real-time evidence components; final ranked CSV remains a separate forensic/evaluation output.",
                "leakage_check": "Online node alerts use only train/validation-calibrated thresholds and causal current/past test state. They do not use GT nodes, attack windows, test labels, final rankings, full-test statistics, future events, attack-specific tokens, or dataset-specific rescue rules before emission.",
            },
            "online_node_alert_performance": online_node_alert_performance,
            "components": COMPONENTS,
            "weights": weights,
            "component_stats": comp_stats,
            "event_threshold_quantile": float(args.event_threshold_quantile),
            "event_threshold": float(threshold),
            "prop_threshold_quantile": float(args.prop_threshold_quantile),
            "prop_threshold": float(prop_threshold),
            "prop_decay_sec": float(args.prop_decay_sec),
            "prop_boost": float(args.prop_boost),
            "prop_update_threshold_ratio": float(args.prop_update_threshold_ratio),
            "undirected_propagation": bool(args.undirected_propagation),
            "node_count_weight": float(args.node_count_weight),
            "node_sum_weight": float(args.node_sum_weight),
            "node_soft_cap": float(args.node_soft_cap),
            "backtrack_hops": int(args.backtrack_hops),
            "backtrack_boost": float(args.backtrack_boost),
            "backtrack_decay_sec": float(args.backtrack_decay_sec),
            "backtrack_min_score_ratio": float(args.backtrack_min_score_ratio),
            "global_decay_sec": float(args.global_decay_sec),
            "global_boost": float(args.global_boost),
            "compact_chain": {
                "enabled": bool(compact_chain_enabled),
                "slots": int(compact_chain_slots),
                "decay_sec": float(args.compact_chain_decay_sec),
                "update_ratio": float(args.compact_chain_update_ratio),
                "min_diversity": int(args.compact_chain_min_diversity),
                "support_cap": float(args.compact_chain_support_cap),
                "support_weight": float(args.compact_chain_support_weight),
                "diversity_weight": float(args.compact_chain_diversity_weight),
                "neighbor_weight": float(args.compact_chain_neighbor_weight),
                "score_weight": float(args.compact_chain_score_weight),
                "bonus_cap": float(args.compact_chain_bonus_cap),
                "active_nodes": int(compact_active),
                "max_bonus": float(compact_max_bonus),
                "definition": "Streaming compact-chain node bonus from past high-score neighbor slots, relation/type diversity, and decayed short-chain support.",
            },
            "chain_profile": {
                "enabled": bool(chain_profile_enabled),
                "buckets": int(args.chain_profile_buckets),
                "score_weight": float(args.chain_profile_score_weight),
                "clip": float(args.chain_profile_clip),
                "require_transition": bool(args.chain_profile_require_transition),
                "score_mode": str(args.chain_profile_score_mode),
                "calibration_knots": int(args.chain_profile_calibration_knots),
                "memory_slots": int(args.chain_profile_memory_slots),
                "association_weight": float(args.chain_profile_association_weight),
                "validation_stats": chain_profile_stats,
                "association_validation_stats": chain_assoc_stats,
                "conformal_calibration": chain_profile_calibration,
                "event_fused_calibration": fused_calibration,
                "active_nodes": int(chain_profile_active),
                "max_bonus": float(chain_profile_max_bonus),
                "metadata": chain_profile_meta,
                "definition": "Benign train-profiled hashed short-chain rarity. Validation calibrates rarity z-scores; test updates state only after each event is scored.",
                "leakage_check": "Counts are learned only from train rows; validation scores only calibrate robust stats; no GT, attack windows, test labels, full-test statistics, or attack-specific tokens are used before evaluation.",
            },
            "causal_episode_support": {
                "enabled": bool(causal_episode_enabled),
                "selective_enabled": bool(args.causal_episode_selective_enabled),
                "min_event_tail": float(args.causal_episode_min_event_tail),
                "min_chain": float(args.causal_episode_min_chain),
                "min_assoc": float(args.causal_episode_min_assoc),
                "min_families": int(args.causal_episode_min_families),
                "buckets": int(args.chain_profile_buckets),
                "score_weight": float(args.chain_profile_score_weight),
                "decay_sec": float(args.online_node_risk_decay_sec),
                "clip": float(args.chain_profile_clip),
                "active_nodes": int(np.sum(causal_episode_sketch.node_support > 0.0)) if causal_episode_sketch is not None else 0,
                "max_support": float(np.max(causal_episode_sketch.node_support)) if causal_episode_sketch is not None and np.any(causal_episode_sketch.node_support > 0.0) else 0.0,
                "metadata": causal_episode_sketch.metadata() if causal_episode_sketch is not None else {},
                "definition": "Bounded episode/subgraph support C_t(v) from hashed short-chain neighborhoods and recent causal context. In selective mode validation-tail residual can select an event for consideration, but support contributes only when at least min_families compact non-semantic evidence families co-occur, such as chain rarity, association discrepancy, AdaptiveMemory shift, or quarantine.",
                "leakage_check": "The sketch is updated causally from current/past streamed test events only. It does not use GT nodes, attack windows, test labels, test rankings, full-test statistics, future events, attack-specific tokens, or dataset-specific rescue rules.",
            },
            "compact_malicious_chain": {
                "enabled": bool(args.compact_malicious_chain_enabled),
                "feature_weight": float(args.compact_malicious_chain_weight),
                "definition": "Validation-calibrated compact short-chain discriminator. Per event, it scores agreement between local residual/semantic rarity, benign-profile chain rarity, transition rarity, and association discrepancy. Per node, it aggregates peak, repeated support, and relation diversity.",
                "leakage_check": "The signal is computed from benign train-profile counts plus validation-calibrated robust statistics. It uses no GT, attack windows, test labels, test-ranking feedback, or attack-specific tokens.",
            },
            "adaptive_memory": {
                "enabled": bool(adaptive_memory_enabled),
                "buckets": int(args.adaptive_memory_buckets),
                "min_count": int(args.adaptive_memory_min_count),
                "fit_alpha": float(args.adaptive_memory_fit_alpha),
                "update_alpha": float(args.adaptive_memory_update_alpha),
                "score_weight": float(args.adaptive_memory_weight),
                "clip": float(args.adaptive_memory_clip),
                "update_tail_max": float(args.adaptive_memory_update_tail_max),
                "update_assoc_z_max": float(args.adaptive_memory_update_assoc_z_max),
                "quarantine_tail_min": float(args.adaptive_memory_quarantine_tail_min),
                "quarantine_assoc_z_min": float(args.adaptive_memory_quarantine_assoc_z_min),
                "distance_stats": adaptive_memory_stats,
                "train_processed": int(adaptive_train_processed),
                "train_updates": int(adaptive_train_updates),
                "train_quarantined": int(adaptive_train_quarantined),
                "validation_distance_samples": int(len(adaptive_memory_ref_distances)),
                "validation_updates": int(adaptive_val_updates),
                "validation_quarantined": int(adaptive_val_quarantined),
                "test_scored_with_prototype": int(adaptive_test_scored),
                "test_updates": int(adaptive_test_updates),
                "test_quarantined": int(adaptive_test_quarantined),
                "stored_model_size_bytes": int(adaptive_memory_model_size_bytes),
                "active_nodes": int(np.sum(adaptive_memory_bonus > 0.0)) if adaptive_memory_enabled else 0,
                "max_bonus": float(np.max(adaptive_memory_bonus)) if adaptive_memory_enabled and np.any(adaptive_memory_bonus > 0.0) else 0.0,
                "definition": "Frozen base detector plus small benign prototype memory. Train and validation high-confidence benign events initialize the memory. During test, each event is scored against the current prototype before any update; only high-confidence benign events update memory, while high-score/chain-suspicious/association-discrepant events are quarantined.",
                "leakage_check": "No GT, attack windows, test labels, future events, full-test statistics, or attack-specific tokens are used for memory construction or updates. Test memory updates are causal and based only on predeclared confidence/quarantine thresholds calibrated from train/validation.",
            },
            "node_rank": {
                "mode": str(args.node_rank_mode),
                "intrinsic_first_applied": bool(node_intrinsic_first_applied),
                "intrinsic_support_weight": float(args.node_intrinsic_support_weight),
                "intrinsic_tie_weight": float(args.node_intrinsic_tie_weight),
                "intrinsic_tie_cap": float(args.node_intrinsic_tie_cap),
                "definition": "score mode preserves the legacy final node score. intrinsic_first mode ranks primarily by no-propagation local residual, benign-profile chain rarity, association discrepancy, and compact causal-chain evidence; propagation contributes only through a clipped secondary tie term. Semantic rarity is retained as intrinsic evidence but is not sufficient causal support by itself.",
            },
            "node_rerank": {
                "enabled": bool(args.node_rerank_enabled),
                "mode": str(args.node_rerank_mode),
                "joint_activity_pressure": bool(args.node_rerank_joint_activity_pressure),
                "activity_only_pressure_active_nodes": int(np.sum(activity_only_pressure > 0.0)),
                "activity_only_pressure_max": float(np.max(activity_only_pressure)) if np.any(activity_only_pressure > 0.0) else 0.0,
                "activity_only_pressure_mean_active": float(np.mean(activity_only_pressure[activity_only_pressure > 0.0])) if np.any(activity_only_pressure > 0.0) else 0.0,
                "weight": float(args.node_rerank_weight),
                "penalty_weight": float(args.node_rerank_penalty_weight),
                "margin": float(args.node_rerank_margin),
                "clip": float(args.node_rerank_clip),
                "calibration_mode": str(args.node_rerank_calibration_mode),
                "calibration_knots": int(args.node_rerank_calibration_knots),
                "high_activity_weight": float(args.node_rerank_high_activity_weight),
                "high_activity_margin": float(args.node_rerank_high_activity_margin),
                "high_activity_cap": float(args.node_rerank_high_activity_cap),
                "feature_names": NODE_RERANK_FEATURES,
                "feature_weights": [float(x) for x in node_rerank_feature_weights.tolist()],
                "feature_stats": node_rerank_feature_stats,
                "feature_calibrations": node_rerank_feature_calibrations,
                "validation_nodes": int(node_rerank_ref_nodes),
                "validation_events": int(node_rerank_ref_processed),
                "active_nodes": int(node_rerank_active),
                "max_delta": float(node_rerank_max_delta),
                "min_delta": float(node_rerank_min_delta),
                "calibration_size_bytes": int(node_rerank_calibration_size_bytes),
                "definition": "Small second-stage node reranker calibrated only on validation node-feature distributions. In tail mode it uses validation empirical-tail/conformal rarity to avoid robust-z saturation. It rewards local residual, chain/profile, association, semantic rarity, and compact causal support, while penalizing propagation-only or benign high-activity pressure when activity lacks non-semantic causal evidence.",
                "leakage_check": "Validation calibration uses no GT or test labels. Test features are accumulated online from current/past event evidence before final evaluation.",
            },
            "activity_normalized_chain": {
                "enabled": bool(args.activity_normalized_chain_enabled),
                "start_topk": int(args.activity_normalized_chain_start_topk),
                "end_topk": int(args.activity_normalized_chain_end_topk),
                "min_z": float(args.activity_normalized_chain_min_z),
                "weight": float(args.activity_normalized_chain_weight),
                "boost_cap": float(args.activity_normalized_chain_boost_cap),
                "calibration_mode": str(args.activity_normalized_chain_calibration_mode),
                "validation_nodes": int(activity_normalized_chain_ref_nodes),
                "density_stats": activity_normalized_chain_density_stats,
                "density_calibration": activity_normalized_chain_density_calibration,
                "calibration_size_bytes": int(activity_normalized_chain_calibration_size_bytes),
                "active_density_nodes": int(np.sum(activity_normalized_chain_density_values > 0.0)),
                "max_density": float(np.max(activity_normalized_chain_density_values)) if np.any(activity_normalized_chain_density_values > 0.0) else 0.0,
                **activity_normalized_chain_meta,
                "definition": "Experimental ActivityNormalizedChainDensity_t(v)=max_past_chain_profile_raw_t(v)/max(1, log1p(high_count_t(v))). The numerator is accumulated causally from benign-profile chain rarity before each test event updates state; the denominator normalizes by streaming high-score activity. The bounded boost is applied only to a configured post-stream forensic candidate rank band, not as a global sort key.",
                "online_explanation": "online_node_alerts.csv includes the current event's activity_normalized_chain_density inside explanation_json for audit. The online alert threshold and risk score are unchanged by this experimental forensic reranker.",
                "leakage_check": "Validation calibrates the density distribution without GT, attack windows, test labels, test rankings, full-test statistics, future events, attack-specific tokens, or dataset-specific rescue rules. Test density uses only current/past streamed evidence; final candidate-band reranking is forensic/evaluation output.",
            },
            "evidence_band_gate": {
                "enabled": bool(args.evidence_band_gate_enabled),
                "calibration_mode": str(args.evidence_band_calibration_mode),
                "candidate_local": float(args.evidence_band_candidate_local),
                "candidate_activity": float(args.evidence_band_candidate_activity),
                "candidate_relation": float(args.evidence_band_candidate_relation),
                "min_local": float(args.evidence_band_min_local),
                "min_relation": float(args.evidence_band_min_relation),
                "min_assoc": float(args.evidence_band_min_assoc),
                "min_semantic": float(args.evidence_band_min_semantic),
                "min_role": float(args.evidence_band_min_role),
                "min_adaptive": float(args.evidence_band_min_adaptive),
                "min_families": int(args.evidence_band_min_families),
                "weight": float(args.evidence_band_weight),
                "boost_cap": float(args.evidence_band_boost_cap),
                "penalty_weight": float(args.evidence_band_penalty_weight),
                "penalty_cap": float(args.evidence_band_penalty_cap),
                "activity_weight": float(args.evidence_band_activity_weight),
                "activity_margin": float(args.evidence_band_activity_margin),
                "update_min_count": float(args.evidence_band_update_min_count),
                "update_weight": float(args.evidence_band_update_weight),
                "update_cap": float(args.evidence_band_update_cap),
                "prop_weight": float(args.evidence_band_prop_weight),
                "prop_cap": float(args.evidence_band_prop_cap),
                **evidence_band_meta,
                "definition": "Unified validation-calibrated evidence-band gate. Candidate nodes are selected by benign-validation tail evidence, not by fixed rank or dataset name. It boosts nodes only when local residual, benign-profile chain rarity, association discrepancy, semantic stability, role diversity, adaptive-memory shift/quarantine, and family-count evidence jointly support an alert. It penalizes high residual/activity, frequent benign-memory updates, or propagation pressure when compact causal evidence is insufficient.",
                "leakage_check": "Uses train/validation benign profiles and validation feature calibrations only. Test features are accumulated causally before final evaluation. It does not use GT, attack windows, test labels, test rankings, full-test statistics, attack-specific tokens, dataset names, or fixed dataset-specific rank bands.",
            },
            "snc": {
                "enabled": bool(args.snc_enabled),
                "decay_sec": float(args.snc_decay_sec),
                "update_min_support": float(args.snc_update_min_support),
                "update_min_assoc_z": float(args.snc_update_min_assoc_z),
                "calibration_mode": str(args.snc_calibration_mode),
                "calibration_knots": int(args.snc_calibration_knots),
                "clip": float(args.snc_clip),
                "weight": float(args.snc_weight),
                "boost_cap": float(args.snc_boost_cap),
                "penalty_weight": float(args.snc_penalty_weight),
                "penalty_cap": float(args.snc_penalty_cap),
                "candidate_start_topk": int(args.snc_candidate_start_topk),
                "candidate_end_topk": int(args.snc_candidate_end_topk),
                "min_cluster_z": float(args.snc_min_cluster_z),
                "min_object_z": float(args.snc_min_object_z),
                "min_diversity_z": float(args.snc_min_diversity_z),
                "high_count_penalty_min": float(args.snc_high_count_penalty_min),
                "benign_update_penalty_min": float(args.snc_benign_update_penalty_min),
                "benign_update_weight": float(args.snc_benign_update_weight),
                "feature_names": SNC_FEATURES,
                "feature_stats": snc_feature_stats,
                "feature_calibrations": snc_feature_calibrations,
                "validation_nodes": int(snc_ref_nodes),
                "calibration_size_bytes": int(snc_calibration_size_bytes),
                **snc_meta,
                "definition": "TFLR-SNC, a streaming suspicious-neighborhood cohesion reranker. It maintains causal actor/process states with rare object-neighbor support, relation/type diversity, quarantine ratio, and object low-recurrence. In tail mode, validation empirical-tail/conformal rarity replaces median/MAD z-scoring to avoid saturated z features. SNC acts only on a narrow candidate rank band as a bounded tie-break, not as broad propagation or top-20000 boosting.",
                "leakage_check": "SNC uses only train/validation-calibrated robust feature distributions and causal test-time state from current/past events. It uses no GT, attack windows, test labels, test rankings, full-test statistics, raw attack strings, or dataset-specific rescue rules before evaluation.",
            },
            "node_hard_gate": {
                "enabled": bool(args.node_hard_gate_enabled),
                "min_support_z": float(args.node_hard_gate_min_support_z),
                "min_families": int(args.node_hard_gate_min_families),
                "prop_z": float(args.node_hard_gate_prop_z),
                "penalty_weight": float(args.node_hard_gate_penalty_weight),
                "cap_margin": float(args.node_hard_gate_cap_margin),
                "applied_nodes": int(node_hard_gate_count),
                "max_penalty": float(node_hard_gate_max_penalty),
                "definition": "Validation-calibrated hard gate: if semantic/association/chain support is weak, propagation evidence cannot lift a node above the no-propagation intrinsic score plus a small margin.",
                "leakage_check": "Gate uses validation-calibrated z features and fixed thresholds only; no GT, attack windows, test labels, or test-ranking feedback are used before evaluation.",
            },
            "local_closure": {
                "enabled": bool(args.local_closure_enabled),
                "protect_topk": int(args.local_closure_protect_topk),
                "end_topk": int(args.local_closure_end_topk),
                "min_score_z": float(args.local_closure_min_score_z),
                "min_families": int(args.local_closure_min_families),
                "min_assoc_z": float(args.local_closure_min_assoc_z),
                "min_role_z": float(args.local_closure_min_role_z),
                "min_adaptive_z": float(args.local_closure_min_adaptive_z),
                "weight": float(args.local_closure_weight),
                "boost_cap": float(args.local_closure_boost_cap),
                "penalty_weight": float(args.local_closure_penalty_weight),
                "penalty_cap": float(args.local_closure_penalty_cap),
                "prop_penalty_weight": float(args.local_closure_prop_penalty_weight),
                "prop_penalty_cap": float(args.local_closure_prop_penalty_cap),
                "update_min_count": float(args.local_closure_update_min_count),
                "update_weight": float(args.local_closure_update_weight),
                "update_cap": float(args.local_closure_update_cap),
                "require_quarantine": bool(int(args.local_closure_require_quarantine)),
                **local_closure_meta,
                "definition": "Validation-calibrated local malicious-chain closure score. It acts only on a configured candidate rank band after intrinsic-first scoring, and scores closure among local residual, relation-transition/benign-chain rarity, association discrepancy, adaptive-memory quarantine/prototype distance, entity/relation role diversity, and low benign-memory update frequency. Association discrepancy and role-type diversity are hard gates; propagation-source diversity can only add a small penalty when it exceeds intrinsic support. It does not add new propagation.",
                "leakage_check": "Uses validation-calibrated robust z-scores and causal AdaptiveMemory counters only. No GT, attack windows, test labels, test rankings, full-test statistics, attack-specific tokens, or dataset-specific rescue rules are used before evaluation.",
            },
            "mid_rank_gate": {
                "enabled": bool(args.mid_rank_gate_enabled),
                "protect_topk": int(args.mid_rank_gate_protect_topk),
                "end_topk": int(args.mid_rank_gate_end_topk),
                "min_assoc_z": float(args.mid_rank_gate_min_assoc_z),
                "min_relation_z": float(args.mid_rank_gate_min_relation_z),
                "min_adaptive_z": float(args.mid_rank_gate_min_adaptive_z),
                "require_quarantine": bool(int(args.mid_rank_gate_require_quarantine)),
                "penalty_weight": float(args.mid_rank_gate_penalty_weight),
                "penalty_cap": float(args.mid_rank_gate_penalty_cap),
                "benign_update_min_count": float(args.mid_rank_gate_benign_update_min_count),
                "benign_update_weight": float(args.mid_rank_gate_benign_update_weight),
                "benign_update_cap": float(args.mid_rank_gate_benign_update_cap),
                **mid_rank_gate_meta,
                "definition": "Experimental local-chain/memory gate applied only to the configured middle rank band after intrinsic-first scoring. Top high-confidence nodes are protected. Candidate nodes are downranked unless they jointly show association discrepancy, relation-transition/chain support, adaptive-memory prototype distance, and quarantine evidence; nodes that frequently update benign memory receive an additional causal update-count penalty.",
                "leakage_check": "The gate uses validation-calibrated node-feature robust z-scores, fixed predeclared thresholds, and causal AdaptiveMemory quarantine/update counters. It does not use GT, attack windows, test labels, attack-specific tokens, future events, full-test statistics, or expanded global propagation.",
            },
            "runtime_environment": {
                "CLAD_EVENT_NETFLOW_USE_PORT": os.getenv("CLAD_EVENT_NETFLOW_USE_PORT", ""),
                "CLAD_EVENT_PAYLOAD_STYLE": os.getenv("CLAD_EVENT_PAYLOAD_STYLE", ""),
                "CLAD_SUBJECT_NODE_TABLE": os.getenv("CLAD_SUBJECT_NODE_TABLE", ""),
            },
            "event_at_threshold": strict_event_metrics,
            "event_at_threshold_strict": strict_event_metrics,
            "event_at_threshold_relaxed": relaxed_event_metrics,
            "attack_window_metrics": attack_metrics,
            "sweep": sweep,
            "best": best,
            "best_under_fp_target": best_fp_constrained,
            "best_node_evidence_summary": _best_node_evidence_summary(ranked, best, positive_nodes, suspect_nodes, node_top_label, node_top_component),
            "best_under_fp_target_node_evidence_summary": _best_node_evidence_summary(ranked, best_fp_constrained, positive_nodes, suspect_nodes, node_top_label, node_top_component),
            "target": {"dataset": args.dataset, **target},
            "meets_theia_relaxed_target": bool(best and int(best["relaxed_node"]["tp"]) > 70 and int(best["relaxed_node"]["fp"]) < 1000),
            "meets_dataset_relaxed_target": bool(best and int(best["relaxed_node"]["tp"]) > int(target["relaxed_tp_gt"]) and int(best["relaxed_node"]["fp"]) < int(target["relaxed_fp_lt"])),
        }
        out_json = os.path.join(result_dir, "eval_tflr_lowrank.json")
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        ranked_csv = os.path.join(result_dir, "eval_tflr_lowrank_ranked.csv")
        with open(ranked_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "rank",
                    "node_id",
                    "node_score",
                    "node_score_before_rerank",
                    "node_score_before_intrinsic_first",
                    "node_score_before_evidence_band",
                    "node_score_before_snc",
                    "node_score_before_local_closure",
                    "node_score_before_mid_rank_gate",
                    "node_score_before_activity_normalized_chain",
                    "node_rerank_delta",
                    "node_rerank_support",
                    "node_intrinsic_rank_support",
                    "node_rerank_prop_only_penalty",
                    "node_rerank_high_activity_penalty",
                    "activity_only_pressure",
                    "activity_normalized_chain_candidate",
                    "activity_normalized_chain_applied",
                    "activity_normalized_chain_density",
                    "activity_normalized_chain_z",
                    "activity_normalized_chain_boost",
                    "node_hard_gate_applied",
                    "node_hard_gate_support",
                    "node_hard_gate_prop_pressure",
                    "node_hard_gate_family_count",
                    "node_hard_gate_penalty",
                    "evidence_band_candidate",
                    "evidence_band_applied",
                    "evidence_band_pass",
                    "evidence_band_score",
                    "evidence_band_boost",
                    "evidence_band_penalty",
                    "evidence_band_local",
                    "evidence_band_relation",
                    "evidence_band_association",
                    "evidence_band_semantic",
                    "evidence_band_role",
                    "evidence_band_adaptive",
                    "evidence_band_activity",
                    "evidence_band_propagation",
                    "evidence_band_family_count",
                    "evidence_band_activity_penalty",
                    "evidence_band_benign_update_penalty",
                    "evidence_band_propagation_penalty",
                    "evidence_band_quarantine",
                    "local_closure_candidate",
                    "local_closure_applied",
                    "local_closure_pass",
                    "local_closure_score",
                    "local_closure_boost",
                    "local_closure_penalty",
                    "local_closure_local_z",
                    "local_closure_relation_z",
                    "local_closure_association_z",
                    "local_closure_adaptive_z",
                    "local_closure_role_z",
                    "local_closure_role_type_z",
                    "local_closure_propagation_z",
                    "local_closure_family_count",
                    "local_closure_benign_update_penalty",
                    "local_closure_propagation_penalty",
                    "local_closure_hard_assoc_pass",
                    "local_closure_hard_role_pass",
                    "mid_rank_gate_candidate",
                    "mid_rank_gate_applied",
                    "mid_rank_gate_pass",
                    "mid_rank_gate_penalty",
                    "mid_rank_gate_assoc_z",
                    "mid_rank_gate_relation_z",
                    "mid_rank_gate_adaptive_z",
                    "mid_rank_gate_quarantine",
                    "mid_rank_gate_benign_update_penalty",
                    "snc_candidate",
                    "snc_applied",
                    "snc_pass",
                    "snc_score",
                    "snc_boost",
                    "snc_penalty",
                    "snc_cluster_z",
                    "snc_object_z",
                    "snc_diversity_z",
                    "snc_quarantine_z",
                    "snc_low_recurrence",
                    "snc_benign_update_penalty",
                    "snc_high_count_penalty",
                    "snc_actor",
                    "snc_event_pos",
                    "node_score_intrinsic",
                    "node_score_without_chain",
                    "node_max",
                    "node_intrinsic_max",
                    "node_high_count",
                    "node_intrinsic_high_count",
                    "node_soft_sum",
                    "node_intrinsic_soft_sum",
                    "compact_chain_bonus",
                    "compact_chain_support",
                    "compact_chain_diversity",
                    "chain_profile_bonus",
                    "chain_profile_raw",
                    "chain_profile_conformal",
                    "chain_profile_event_surprise",
                    "chain_profile_event_pos",
                    "causal_episode_support",
                    "causal_episode_count",
                    "causal_episode_rel_diversity",
                    "causal_episode_type_diversity",
                    "causal_episode_event_pos",
                    "adaptive_memory_bonus",
                    "adaptive_memory_raw_distance",
                    "adaptive_memory_event_pos",
                    "adaptive_memory_quarantined",
                    "adaptive_memory_update_count",
                    "association_discrepancy_max",
                    "association_score_max",
                    "prop_source_diversity",
                    "semantic_rarity_stability",
                    "local_chain_agreement",
                    "compact_malicious_chain_signal",
                    "top_event_pos",
                    "top_event_label",
                    "top_component",
                ],
            )
            writer.writeheader()
            component_names = COMPONENTS + EXTRA_COMPONENTS
            for rank, node in enumerate(ranked.tolist(), start=1):
                idx = int(node_top_component[node])
                chain_bonus = float(compact_chain_bonus[node]) if compact_chain_enabled else 0.0
                chain_support = float(compact_chain_support[node]) if compact_chain_enabled else 0.0
                chain_diversity = compact_chain_diversity(int(node)) if compact_chain_enabled else 0
                profile_bonus = float(chain_profile_bonus[node]) if chain_profile_enabled else 0.0
                profile_raw = float(chain_profile_raw[node]) if chain_profile_enabled else 0.0
                profile_conformal = float(chain_profile_conformal[node]) if chain_profile_enabled else 0.0
                profile_event_surprise = float(chain_profile_event_surprise[node]) if chain_profile_enabled else 0.0
                profile_event_pos = int(chain_profile_event_pos[node]) if chain_profile_enabled else -1
                if activity_normalized_chain_density_values.size > int(node):
                    activity_chain_density = float(activity_normalized_chain_density_values[node])
                elif chain_profile_enabled:
                    activity_chain_density = float(profile_raw / max(1.0, math.log1p(max(float(node_high_count[node]), 0.0))))
                else:
                    activity_chain_density = 0.0
                activity_chain_candidate = (
                    bool(activity_normalized_chain_arrays["candidate"][node])
                    if bool(args.activity_normalized_chain_enabled)
                    else False
                )
                activity_chain_applied = (
                    bool(activity_normalized_chain_arrays["applied"][node])
                    if bool(args.activity_normalized_chain_enabled)
                    else False
                )
                activity_chain_z = (
                    float(activity_normalized_chain_arrays["score_z"][node])
                    if bool(args.activity_normalized_chain_enabled)
                    else 0.0
                )
                activity_chain_boost = (
                    float(activity_normalized_chain_arrays["boost"][node])
                    if bool(args.activity_normalized_chain_enabled)
                    else 0.0
                )
                causal_episode_support = float(causal_episode_sketch.node_support[node]) if causal_episode_sketch is not None else 0.0
                causal_episode_count = float(causal_episode_sketch.node_count[node]) if causal_episode_sketch is not None else 0.0
                causal_episode_rel_diversity = int(int(causal_episode_sketch.node_rel_mask[node]).bit_count()) if causal_episode_sketch is not None else 0
                causal_episode_type_diversity = int(int(causal_episode_sketch.node_type_mask[node]).bit_count()) if causal_episode_sketch is not None else 0
                causal_episode_event_pos = int(causal_episode_sketch.node_event_pos[node]) if causal_episode_sketch is not None else -1
                adaptive_bonus = float(adaptive_memory_bonus[node]) if adaptive_memory_enabled else 0.0
                adaptive_raw = float(adaptive_memory_raw[node]) if adaptive_memory_enabled else 0.0
                adaptive_pos = int(adaptive_memory_event_pos[node]) if adaptive_memory_enabled else -1
                adaptive_quarantined_node = float(adaptive_memory_quarantine_node[node]) if adaptive_memory_enabled else 0.0
                adaptive_update_count = float(adaptive_memory_update_node[node]) if adaptive_memory_enabled else 0.0
                if bool(args.node_rerank_enabled):
                    prop_source_diversity = int(int(node_rerank_extra["prop_source_mask"][node]).bit_count())
                    semantic_stability = float(node_rerank_extra["semantic_peak"][node]) * math.log1p(float(node_rerank_extra["semantic_count"][node])) + 0.10 * float(node_rerank_extra["semantic_sum"][node])
                    association_max = float(node_rerank_extra["association_max"][node])
                    association_score_max = float(node_rerank_extra["association_score_max"][node])
                    local_chain_agreement = float(node_rerank_extra["local_chain_agreement"][node])
                    malicious_chain_signal = (
                        float(node_rerank_extra["malicious_chain_peak"][node]) * math.log1p(float(node_rerank_extra["malicious_chain_count"][node]))
                        + 0.10 * float(node_rerank_extra["malicious_chain_sum"][node])
                        + 0.25 * math.log1p(float(int(node_rerank_extra["malicious_chain_rel_mask"][node]).bit_count()))
                    )
                    rerank_delta_value = float(node_rerank_delta_values[node])
                    rerank_support_value = float(node_rerank_support_values[node])
                    intrinsic_rank_support_value = float(node_intrinsic_rank_support_values[node])
                    rerank_penalty_value = float(node_rerank_penalty_values[node])
                    rerank_high_activity_penalty_value = float(node_rerank_high_activity_penalty_values[node])
                    activity_only_pressure_value = float(activity_only_pressure[node])
                    hard_gate_applied = bool(node_hard_gate_applied[node]) if bool(args.node_hard_gate_enabled) else False
                    hard_gate_support = float(node_hard_gate_support[node]) if bool(args.node_hard_gate_enabled) else 0.0
                    hard_gate_prop_pressure = float(node_hard_gate_prop_pressure[node]) if bool(args.node_hard_gate_enabled) else 0.0
                    hard_gate_family_count = float(node_hard_gate_family_count[node]) if bool(args.node_hard_gate_enabled) else 0.0
                    hard_gate_penalty = float(node_hard_gate_penalty[node]) if bool(args.node_hard_gate_enabled) else 0.0
                    score_before_rerank = float(node_score_before_rerank[node])
                    evidence_band_candidate = bool(evidence_band_arrays["candidate"][node]) if bool(args.evidence_band_gate_enabled) else False
                    evidence_band_applied = bool(evidence_band_arrays["applied"][node]) if bool(args.evidence_band_gate_enabled) else False
                    evidence_band_pass = bool(evidence_band_arrays["pass"][node]) if bool(args.evidence_band_gate_enabled) else False
                    evidence_band_score = float(evidence_band_arrays["score"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_boost = float(evidence_band_arrays["boost"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_penalty = float(evidence_band_arrays["penalty"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_local = float(evidence_band_arrays["local"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_relation = float(evidence_band_arrays["relation"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_association = float(evidence_band_arrays["association"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_semantic = float(evidence_band_arrays["semantic"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_role = float(evidence_band_arrays["role"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_adaptive = float(evidence_band_arrays["adaptive"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_activity = float(evidence_band_arrays["activity"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_propagation = float(evidence_band_arrays["propagation"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_family_count = float(evidence_band_arrays["family_count"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_activity_penalty = float(evidence_band_arrays["activity_penalty"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_update_penalty = float(evidence_band_arrays["benign_update_penalty"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_prop_penalty = float(evidence_band_arrays["propagation_penalty"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    evidence_band_quarantine = float(evidence_band_arrays["quarantine"][node]) if bool(args.evidence_band_gate_enabled) else 0.0
                    local_closure_candidate = bool(local_closure_arrays["candidate"][node]) if bool(args.local_closure_enabled) else False
                    local_closure_applied = bool(local_closure_arrays["applied"][node]) if bool(args.local_closure_enabled) else False
                    local_closure_pass = bool(local_closure_arrays["pass"][node]) if bool(args.local_closure_enabled) else False
                    local_closure_score = float(local_closure_arrays["score"][node]) if bool(args.local_closure_enabled) else 0.0
                    local_closure_boost = float(local_closure_arrays["boost"][node]) if bool(args.local_closure_enabled) else 0.0
                    local_closure_penalty = float(local_closure_arrays["penalty"][node]) if bool(args.local_closure_enabled) else 0.0
                    local_closure_local_z = float(local_closure_arrays["local_z"][node]) if bool(args.local_closure_enabled) else 0.0
                    local_closure_relation_z = float(local_closure_arrays["relation_z"][node]) if bool(args.local_closure_enabled) else 0.0
                    local_closure_association_z = float(local_closure_arrays["association_z"][node]) if bool(args.local_closure_enabled) else 0.0
                    local_closure_adaptive_z = float(local_closure_arrays["adaptive_z"][node]) if bool(args.local_closure_enabled) else 0.0
                    local_closure_role_z = float(local_closure_arrays["role_z"][node]) if bool(args.local_closure_enabled) else 0.0
                    local_closure_role_type_z = float(local_closure_arrays["role_type_z"][node]) if bool(args.local_closure_enabled) else 0.0
                    local_closure_propagation_z = float(local_closure_arrays["propagation_z"][node]) if bool(args.local_closure_enabled) else 0.0
                    local_closure_family_count = float(local_closure_arrays["family_count"][node]) if bool(args.local_closure_enabled) else 0.0
                    local_closure_update_penalty = float(local_closure_arrays["benign_update_penalty"][node]) if bool(args.local_closure_enabled) else 0.0
                    local_closure_prop_penalty = float(local_closure_arrays["propagation_penalty"][node]) if bool(args.local_closure_enabled) else 0.0
                    local_closure_hard_assoc_pass = bool(local_closure_arrays["hard_assoc_pass"][node]) if bool(args.local_closure_enabled) else False
                    local_closure_hard_role_pass = bool(local_closure_arrays["hard_role_pass"][node]) if bool(args.local_closure_enabled) else False
                    mid_gate_candidate = bool(mid_rank_gate_arrays["candidate"][node]) if bool(args.mid_rank_gate_enabled) else False
                    mid_gate_applied = bool(mid_rank_gate_arrays["applied"][node]) if bool(args.mid_rank_gate_enabled) else False
                    mid_gate_pass = bool(mid_rank_gate_arrays["pass"][node]) if bool(args.mid_rank_gate_enabled) else False
                    mid_gate_penalty = float(mid_rank_gate_arrays["penalty"][node]) if bool(args.mid_rank_gate_enabled) else 0.0
                    mid_gate_assoc_z = float(mid_rank_gate_arrays["assoc_z"][node]) if bool(args.mid_rank_gate_enabled) else 0.0
                    mid_gate_relation_z = float(mid_rank_gate_arrays["relation_z"][node]) if bool(args.mid_rank_gate_enabled) else 0.0
                    mid_gate_adaptive_z = float(mid_rank_gate_arrays["adaptive_z"][node]) if bool(args.mid_rank_gate_enabled) else 0.0
                    mid_gate_quarantine = float(mid_rank_gate_arrays["quarantine"][node]) if bool(args.mid_rank_gate_enabled) else 0.0
                    mid_gate_update_penalty = float(mid_rank_gate_arrays["benign_update_penalty"][node]) if bool(args.mid_rank_gate_enabled) else 0.0
                    snc_candidate = bool(snc_arrays["candidate"][node]) if bool(args.snc_enabled) else False
                    snc_applied = bool(snc_arrays["applied"][node]) if bool(args.snc_enabled) else False
                    snc_pass = bool(snc_arrays["pass"][node]) if bool(args.snc_enabled) else False
                    snc_score = float(snc_arrays["score"][node]) if bool(args.snc_enabled) else 0.0
                    snc_boost = float(snc_arrays["boost"][node]) if bool(args.snc_enabled) else 0.0
                    snc_penalty = float(snc_arrays["penalty"][node]) if bool(args.snc_enabled) else 0.0
                    snc_cluster_z = float(snc_arrays["cluster_z"][node]) if bool(args.snc_enabled) else 0.0
                    snc_object_z = float(snc_arrays["object_z"][node]) if bool(args.snc_enabled) else 0.0
                    snc_diversity_z = float(snc_arrays["diversity_z"][node]) if bool(args.snc_enabled) else 0.0
                    snc_quarantine_z = float(snc_arrays["quarantine_z"][node]) if bool(args.snc_enabled) else 0.0
                    snc_low_recurrence = float(snc_arrays["low_recurrence"][node]) if bool(args.snc_enabled) else 0.0
                    snc_update_penalty = float(snc_arrays["benign_update_penalty"][node]) if bool(args.snc_enabled) else 0.0
                    snc_high_count_penalty = float(snc_arrays["high_count_penalty"][node]) if bool(args.snc_enabled) else 0.0
                    snc_actor = int(snc_state["object_actor"][node]) if bool(args.snc_enabled) else -1
                    snc_event_pos = int(snc_state["object_event_pos"][node]) if bool(args.snc_enabled) else -1
                else:
                    prop_source_diversity = 0
                    semantic_stability = 0.0
                    association_max = 0.0
                    association_score_max = 0.0
                    local_chain_agreement = 0.0
                    malicious_chain_signal = 0.0
                    rerank_delta_value = 0.0
                    rerank_support_value = 0.0
                    intrinsic_rank_support_value = 0.0
                    rerank_penalty_value = 0.0
                    rerank_high_activity_penalty_value = 0.0
                    activity_only_pressure_value = float(activity_only_pressure[node])
                    hard_gate_applied = False
                    hard_gate_support = 0.0
                    hard_gate_prop_pressure = 0.0
                    hard_gate_family_count = 0.0
                    hard_gate_penalty = 0.0
                    score_before_rerank = float(node_score[node])
                    evidence_band_candidate = False
                    evidence_band_applied = False
                    evidence_band_pass = False
                    evidence_band_score = 0.0
                    evidence_band_boost = 0.0
                    evidence_band_penalty = 0.0
                    evidence_band_local = 0.0
                    evidence_band_relation = 0.0
                    evidence_band_association = 0.0
                    evidence_band_semantic = 0.0
                    evidence_band_role = 0.0
                    evidence_band_adaptive = 0.0
                    evidence_band_activity = 0.0
                    evidence_band_propagation = 0.0
                    evidence_band_family_count = 0.0
                    evidence_band_activity_penalty = 0.0
                    evidence_band_update_penalty = 0.0
                    evidence_band_prop_penalty = 0.0
                    evidence_band_quarantine = 0.0
                    local_closure_candidate = False
                    local_closure_applied = False
                    local_closure_pass = False
                    local_closure_score = 0.0
                    local_closure_boost = 0.0
                    local_closure_penalty = 0.0
                    local_closure_local_z = 0.0
                    local_closure_relation_z = 0.0
                    local_closure_association_z = 0.0
                    local_closure_adaptive_z = 0.0
                    local_closure_role_z = 0.0
                    local_closure_role_type_z = 0.0
                    local_closure_propagation_z = 0.0
                    local_closure_family_count = 0.0
                    local_closure_update_penalty = 0.0
                    local_closure_prop_penalty = 0.0
                    local_closure_hard_assoc_pass = False
                    local_closure_hard_role_pass = False
                    mid_gate_candidate = False
                    mid_gate_applied = False
                    mid_gate_pass = False
                    mid_gate_penalty = 0.0
                    mid_gate_assoc_z = 0.0
                    mid_gate_relation_z = 0.0
                    mid_gate_adaptive_z = 0.0
                    mid_gate_quarantine = 0.0
                    mid_gate_update_penalty = 0.0
                    snc_candidate = False
                    snc_applied = False
                    snc_pass = False
                    snc_score = 0.0
                    snc_boost = 0.0
                    snc_penalty = 0.0
                    snc_cluster_z = 0.0
                    snc_object_z = 0.0
                    snc_diversity_z = 0.0
                    snc_quarantine_z = 0.0
                    snc_low_recurrence = 0.0
                    snc_update_penalty = 0.0
                    snc_high_count_penalty = 0.0
                    snc_actor = -1
                    snc_event_pos = -1
                writer.writerow(
                    {
                        "rank": rank,
                        "node_id": int(node),
                        "node_score": float(node_score[node]),
                        "node_score_before_rerank": float(score_before_rerank),
                        "node_score_before_intrinsic_first": float(node_score_before_intrinsic_first[node]),
                        "node_score_before_evidence_band": float(node_score_before_evidence_band[node]),
                        "node_score_before_snc": float(node_score_before_snc[node]),
                        "node_score_before_local_closure": float(node_score_before_local_closure[node]),
                        "node_score_before_mid_rank_gate": float(node_score_before_mid_rank_gate[node]),
                        "node_score_before_activity_normalized_chain": float(node_score_before_activity_normalized_chain[node]),
                        "node_rerank_delta": float(rerank_delta_value),
                        "node_rerank_support": float(rerank_support_value),
                        "node_intrinsic_rank_support": float(intrinsic_rank_support_value),
                        "node_rerank_prop_only_penalty": float(rerank_penalty_value),
                        "node_rerank_high_activity_penalty": float(rerank_high_activity_penalty_value),
                        "activity_only_pressure": float(activity_only_pressure_value),
                        "activity_normalized_chain_candidate": int(activity_chain_candidate),
                        "activity_normalized_chain_applied": int(activity_chain_applied),
                        "activity_normalized_chain_density": float(activity_chain_density),
                        "activity_normalized_chain_z": float(activity_chain_z),
                        "activity_normalized_chain_boost": float(activity_chain_boost),
                        "node_hard_gate_applied": int(hard_gate_applied),
                        "node_hard_gate_support": float(hard_gate_support),
                        "node_hard_gate_prop_pressure": float(hard_gate_prop_pressure),
                        "node_hard_gate_family_count": float(hard_gate_family_count),
                        "node_hard_gate_penalty": float(hard_gate_penalty),
                        "evidence_band_candidate": int(evidence_band_candidate),
                        "evidence_band_applied": int(evidence_band_applied),
                        "evidence_band_pass": int(evidence_band_pass),
                        "evidence_band_score": float(evidence_band_score),
                        "evidence_band_boost": float(evidence_band_boost),
                        "evidence_band_penalty": float(evidence_band_penalty),
                        "evidence_band_local": float(evidence_band_local),
                        "evidence_band_relation": float(evidence_band_relation),
                        "evidence_band_association": float(evidence_band_association),
                        "evidence_band_semantic": float(evidence_band_semantic),
                        "evidence_band_role": float(evidence_band_role),
                        "evidence_band_adaptive": float(evidence_band_adaptive),
                        "evidence_band_activity": float(evidence_band_activity),
                        "evidence_band_propagation": float(evidence_band_propagation),
                        "evidence_band_family_count": float(evidence_band_family_count),
                        "evidence_band_activity_penalty": float(evidence_band_activity_penalty),
                        "evidence_band_benign_update_penalty": float(evidence_band_update_penalty),
                        "evidence_band_propagation_penalty": float(evidence_band_prop_penalty),
                        "evidence_band_quarantine": float(evidence_band_quarantine),
                        "local_closure_candidate": int(local_closure_candidate),
                        "local_closure_applied": int(local_closure_applied),
                        "local_closure_pass": int(local_closure_pass),
                        "local_closure_score": float(local_closure_score),
                        "local_closure_boost": float(local_closure_boost),
                        "local_closure_penalty": float(local_closure_penalty),
                        "local_closure_local_z": float(local_closure_local_z),
                        "local_closure_relation_z": float(local_closure_relation_z),
                        "local_closure_association_z": float(local_closure_association_z),
                        "local_closure_adaptive_z": float(local_closure_adaptive_z),
                        "local_closure_role_z": float(local_closure_role_z),
                        "local_closure_role_type_z": float(local_closure_role_type_z),
                        "local_closure_propagation_z": float(local_closure_propagation_z),
                        "local_closure_family_count": float(local_closure_family_count),
                        "local_closure_benign_update_penalty": float(local_closure_update_penalty),
                        "local_closure_propagation_penalty": float(local_closure_prop_penalty),
                        "local_closure_hard_assoc_pass": int(local_closure_hard_assoc_pass),
                        "local_closure_hard_role_pass": int(local_closure_hard_role_pass),
                        "mid_rank_gate_candidate": int(mid_gate_candidate),
                        "mid_rank_gate_applied": int(mid_gate_applied),
                        "mid_rank_gate_pass": int(mid_gate_pass),
                        "mid_rank_gate_penalty": float(mid_gate_penalty),
                        "mid_rank_gate_assoc_z": float(mid_gate_assoc_z),
                        "mid_rank_gate_relation_z": float(mid_gate_relation_z),
                        "mid_rank_gate_adaptive_z": float(mid_gate_adaptive_z),
                        "mid_rank_gate_quarantine": float(mid_gate_quarantine),
                        "mid_rank_gate_benign_update_penalty": float(mid_gate_update_penalty),
                        "snc_candidate": int(snc_candidate),
                        "snc_applied": int(snc_applied),
                        "snc_pass": int(snc_pass),
                        "snc_score": float(snc_score),
                        "snc_boost": float(snc_boost),
                        "snc_penalty": float(snc_penalty),
                        "snc_cluster_z": float(snc_cluster_z),
                        "snc_object_z": float(snc_object_z),
                        "snc_diversity_z": float(snc_diversity_z),
                        "snc_quarantine_z": float(snc_quarantine_z),
                        "snc_low_recurrence": float(snc_low_recurrence),
                        "snc_benign_update_penalty": float(snc_update_penalty),
                        "snc_high_count_penalty": float(snc_high_count_penalty),
                        "snc_actor": int(snc_actor),
                        "snc_event_pos": int(snc_event_pos),
                        "node_score_intrinsic": float(node_score_intrinsic[node]),
                        "node_score_without_chain": float(node_score_without_chain[node]),
                        "node_max": float(node_max[node]),
                        "node_intrinsic_max": float(node_intrinsic_max[node]),
                        "node_high_count": float(node_high_count[node]),
                        "node_intrinsic_high_count": float(node_intrinsic_high_count[node]),
                        "node_soft_sum": float(node_soft_sum[node]),
                        "node_intrinsic_soft_sum": float(node_intrinsic_soft_sum[node]),
                        "compact_chain_bonus": float(chain_bonus),
                        "compact_chain_support": float(chain_support),
                        "compact_chain_diversity": int(chain_diversity),
                        "chain_profile_bonus": float(profile_bonus),
                        "chain_profile_raw": float(profile_raw),
                        "chain_profile_conformal": float(profile_conformal),
                        "chain_profile_event_surprise": float(profile_event_surprise),
                        "chain_profile_event_pos": int(profile_event_pos),
                        "causal_episode_support": float(causal_episode_support),
                        "causal_episode_count": float(causal_episode_count),
                        "causal_episode_rel_diversity": int(causal_episode_rel_diversity),
                        "causal_episode_type_diversity": int(causal_episode_type_diversity),
                        "causal_episode_event_pos": int(causal_episode_event_pos),
                        "adaptive_memory_bonus": float(adaptive_bonus),
                        "adaptive_memory_raw_distance": float(adaptive_raw),
                        "adaptive_memory_event_pos": int(adaptive_pos),
                        "adaptive_memory_quarantined": float(adaptive_quarantined_node),
                        "adaptive_memory_update_count": float(adaptive_update_count),
                        "association_discrepancy_max": float(association_max),
                        "association_score_max": float(association_score_max),
                        "prop_source_diversity": int(prop_source_diversity),
                        "semantic_rarity_stability": float(semantic_stability),
                        "local_chain_agreement": float(local_chain_agreement),
                        "compact_malicious_chain_signal": float(malicious_chain_signal),
                        "top_event_pos": int(node_top_event[node]),
                        "top_event_label": int(node_top_label[node]),
                        "top_component": component_names[idx] if 0 <= idx < len(component_names) else "unknown",
                    }
                )
        print(json.dumps({"out_json": out_json, "best": best, "meets_dataset_relaxed_target": summary["meets_dataset_relaxed_target"]}, indent=2, sort_keys=True))
    finally:
        conn_cur.close()
        conn.close()


if __name__ == "__main__":
    main()
