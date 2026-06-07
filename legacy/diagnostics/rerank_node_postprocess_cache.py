#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.eval_utils import best_sweep_row, best_under_fp_target, node_confusion_from_masks, rank_scores, target_for_dataset
from legacy.tools.run_tflr_light_db_lowrank import (
    NODE_RERANK_FEATURES,
    _apply_evidence_band_gate,
    _apply_local_closure_score,
    _apply_mid_rank_gate,
    _node_rerank_delta,
)


def _bool_arg(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast offline node post-processing reranker. Prefer --cache_npz produced by "
            "run_tflr_light_db_lowrank.py --node_postprocess_cache_enabled. Legacy "
            "--ranked_csv mode is diagnostic because it reconstructs a subset of node features."
        )
    )
    parser.add_argument("--source_json", required=True)
    parser.add_argument("--cache_npz", default="")
    parser.add_argument("--ranked_csv", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--out_prefix", default="rerank_node_postprocess")
    parser.add_argument("--topk_values", default="")
    parser.add_argument("--max_ranked_rows", type=int, default=0)

    parser.add_argument(
        "--base_score",
        choices=[
            "before_rerank",
            "before_intrinsic_first",
            "before_evidence_band",
            "before_snc",
            "before_local_closure",
            "before_mid_rank_gate",
            "final",
        ],
        default="before_rerank",
    )
    parser.add_argument("--node_rerank_enabled", type=_bool_arg, default=None)
    parser.add_argument("--node_rerank_weight", type=float, default=None)
    parser.add_argument("--node_rerank_penalty_weight", type=float, default=None)
    parser.add_argument("--node_rerank_margin", type=float, default=None)
    parser.add_argument("--node_rerank_clip", type=float, default=None)
    parser.add_argument("--node_rerank_calibration_mode", choices=["z", "tail"], default="")
    parser.add_argument("--node_rerank_high_activity_weight", type=float, default=None)
    parser.add_argument("--node_rerank_high_activity_margin", type=float, default=None)
    parser.add_argument("--node_rerank_high_activity_cap", type=float, default=None)
    parser.add_argument("--node_rerank_mode", choices=["additive", "replace"], default="")
    parser.add_argument("--node_rerank_feature_weights", default="")

    parser.add_argument("--node_rank_mode", choices=["score", "intrinsic_first"], default="")
    parser.add_argument("--node_intrinsic_support_weight", type=float, default=None)
    parser.add_argument("--node_intrinsic_tie_weight", type=float, default=None)
    parser.add_argument("--node_intrinsic_tie_cap", type=float, default=None)

    parser.add_argument("--evidence_band_gate_enabled", type=_bool_arg, default=None)
    parser.add_argument("--evidence_band_calibration_mode", choices=["tail", "z"], default="")
    parser.add_argument("--evidence_band_candidate_local", type=float, default=None)
    parser.add_argument("--evidence_band_candidate_activity", type=float, default=None)
    parser.add_argument("--evidence_band_candidate_relation", type=float, default=None)
    parser.add_argument("--evidence_band_min_local", type=float, default=None)
    parser.add_argument("--evidence_band_min_relation", type=float, default=None)
    parser.add_argument("--evidence_band_min_assoc", type=float, default=None)
    parser.add_argument("--evidence_band_min_semantic", type=float, default=None)
    parser.add_argument("--evidence_band_min_role", type=float, default=None)
    parser.add_argument("--evidence_band_min_adaptive", type=float, default=None)
    parser.add_argument("--evidence_band_min_families", type=int, default=None)
    parser.add_argument("--evidence_band_weight", type=float, default=None)
    parser.add_argument("--evidence_band_boost_cap", type=float, default=None)
    parser.add_argument("--evidence_band_penalty_weight", type=float, default=None)
    parser.add_argument("--evidence_band_penalty_cap", type=float, default=None)
    parser.add_argument("--evidence_band_activity_weight", type=float, default=None)
    parser.add_argument("--evidence_band_activity_margin", type=float, default=None)
    parser.add_argument("--evidence_band_update_min_count", type=float, default=None)
    parser.add_argument("--evidence_band_update_weight", type=float, default=None)
    parser.add_argument("--evidence_band_update_cap", type=float, default=None)
    parser.add_argument("--evidence_band_prop_weight", type=float, default=None)
    parser.add_argument("--evidence_band_prop_cap", type=float, default=None)

    parser.add_argument("--local_closure_enabled", type=_bool_arg, default=None)
    parser.add_argument("--local_closure_protect_topk", type=int, default=None)
    parser.add_argument("--local_closure_end_topk", type=int, default=None)
    parser.add_argument("--local_closure_min_score_z", type=float, default=None)
    parser.add_argument("--local_closure_min_families", type=int, default=None)
    parser.add_argument("--local_closure_min_assoc_z", type=float, default=None)
    parser.add_argument("--local_closure_min_role_z", type=float, default=None)
    parser.add_argument("--local_closure_min_adaptive_z", type=float, default=None)
    parser.add_argument("--local_closure_weight", type=float, default=None)
    parser.add_argument("--local_closure_boost_cap", type=float, default=None)
    parser.add_argument("--local_closure_penalty_weight", type=float, default=None)
    parser.add_argument("--local_closure_penalty_cap", type=float, default=None)
    parser.add_argument("--local_closure_prop_penalty_weight", type=float, default=None)
    parser.add_argument("--local_closure_prop_penalty_cap", type=float, default=None)
    parser.add_argument("--local_closure_update_min_count", type=float, default=None)
    parser.add_argument("--local_closure_update_weight", type=float, default=None)
    parser.add_argument("--local_closure_update_cap", type=float, default=None)
    parser.add_argument("--local_closure_require_quarantine", type=_bool_arg, default=None)

    parser.add_argument("--mid_rank_gate_enabled", type=_bool_arg, default=None)
    parser.add_argument("--mid_rank_gate_protect_topk", type=int, default=None)
    parser.add_argument("--mid_rank_gate_end_topk", type=int, default=None)
    parser.add_argument("--mid_rank_gate_min_assoc_z", type=float, default=None)
    parser.add_argument("--mid_rank_gate_min_relation_z", type=float, default=None)
    parser.add_argument("--mid_rank_gate_min_adaptive_z", type=float, default=None)
    parser.add_argument("--mid_rank_gate_require_quarantine", type=_bool_arg, default=None)
    parser.add_argument("--mid_rank_gate_penalty_weight", type=float, default=None)
    parser.add_argument("--mid_rank_gate_penalty_cap", type=float, default=None)
    parser.add_argument("--mid_rank_gate_benign_update_min_count", type=float, default=None)
    parser.add_argument("--mid_rank_gate_benign_update_weight", type=float, default=None)
    parser.add_argument("--mid_rank_gate_benign_update_cap", type=float, default=None)

    parser.add_argument("--legacy_role_type_log_default", type=float, default=0.0)
    parser.add_argument("--legacy_metrics_from_top_event_label", type=_bool_arg, default=True)
    return parser.parse_args()


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _section(summary: dict[str, Any], name: str) -> dict[str, Any]:
    value = summary.get(name, {})
    return value if isinstance(value, dict) else {}


def _cfg_value(cli_value: Any, summary: dict[str, Any], section: str, key: str, default: Any) -> Any:
    if cli_value not in (None, ""):
        return cli_value
    data = _section(summary, section)
    return data.get(key, default)


def _feature_weights(args: argparse.Namespace, summary: dict[str, Any]) -> np.ndarray:
    raw = str(args.node_rerank_feature_weights).strip()
    if not raw:
        values = _section(summary, "node_rerank").get("feature_weights", [])
        if isinstance(values, list) and values:
            weights = np.asarray([float(x) for x in values], dtype=np.float32)
        else:
            raw = "0.60,0.10,0.10,0.80,0.90,0.15,0.30,0.90,0.50,1.00,1.20"
            weights = np.asarray([float(x.strip()) for x in raw.split(",") if x.strip()], dtype=np.float32)
    else:
        weights = np.asarray([float(x.strip()) for x in raw.split(",") if x.strip()], dtype=np.float32)
    if weights.shape[0] == 10 and len(NODE_RERANK_FEATURES) == 11:
        weights = np.concatenate(
            [
                weights[:6],
                np.asarray([0.30], dtype=np.float32),
                weights[6:],
            ]
        )
    if weights.shape[0] != len(NODE_RERANK_FEATURES):
        raise ValueError(f"node rerank weights must contain {len(NODE_RERANK_FEATURES)} values")
    return weights.astype(np.float32, copy=False)


def _topk_values(args: argparse.Namespace, summary: dict[str, Any], ranked_len: int) -> list[int]:
    if str(args.topk_values).strip():
        values = [int(x.strip()) for x in str(args.topk_values).split(",") if x.strip()]
    else:
        sweep = summary.get("sweep", [])
        if isinstance(sweep, list) and sweep:
            values = [int(row["topk"]) for row in sweep if isinstance(row, dict) and "topk" in row]
        else:
            values = [100, 200, 300, 400, 433, 500, 700, 900, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500, 4000, 5000, 7000, 10000, 12000, 15000, 20000, 30000, 50000]
    values = sorted({int(v) for v in values if int(v) > 0 and int(v) <= max(int(ranked_len), 1)})
    return values or [min(int(ranked_len), 100)]


def _empty_bool(size: int) -> np.ndarray:
    return np.zeros((size,), dtype=bool)


def _empty_float(size: int) -> np.ndarray:
    return np.zeros((size,), dtype=np.float32)


def _load_cache_npz(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as loaded:
        data = {name: loaded[name] for name in loaded.files}
    size = int(data["node_score_final"].shape[0])
    for key in [
        "all_nodes",
        "positive_nodes",
        "suspect_nodes",
        "node_score_before_rerank",
        "node_score_before_intrinsic_first",
        "node_score_before_evidence_band",
        "node_score_before_snc",
        "node_score_before_local_closure",
        "node_score_before_mid_rank_gate",
        "node_score_intrinsic",
        "node_score_without_chain",
        "node_max",
        "node_high_count",
        "node_soft_sum",
        "chain_profile_bonus",
        "chain_profile_raw",
        "adaptive_memory_bonus",
        "adaptive_memory_quarantine_node",
        "adaptive_memory_update_node",
        "node_top_event",
        "node_top_label",
        "node_top_component",
    ]:
        if key not in data:
            if key in {"all_nodes", "positive_nodes", "suspect_nodes"}:
                data[key] = _empty_bool(size)
            else:
                data[key] = _empty_float(size)
    data["source_rank"] = _rank_position(data["node_score_final"])
    data["input_mode"] = "node_postprocess_cache_npz"
    data["exact_masks_available"] = True
    return data


def _rank_position(scores: np.ndarray) -> np.ndarray:
    ranked = rank_scores(scores)
    out = np.zeros((scores.shape[0],), dtype=np.int64)
    out[ranked] = np.arange(1, ranked.shape[0] + 1, dtype=np.int64)
    return out


def _float_row(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return float(default)
    return float(value)


def _int_row(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value in ("", None):
        return int(default)
    return int(float(value))


def _load_ranked_csv(path: str, role_type_default: float) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "node_id" not in (reader.fieldnames or []):
            raise KeyError(f"{path} missing node_id column")
        rows = [row for row in reader]
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    nodes = np.asarray([_int_row(row, "node_id") for row in rows], dtype=np.int64)
    size = int(np.max(nodes)) + 1
    data: dict[str, Any] = {
        "input_mode": "legacy_ranked_csv",
        "exact_masks_available": False,
        "all_nodes": _empty_bool(size),
        "positive_nodes": _empty_bool(size),
        "suspect_nodes": _empty_bool(size),
        "node_score_final": _empty_float(size),
        "node_score_before_rerank": _empty_float(size),
        "node_score_before_intrinsic_first": _empty_float(size),
        "node_score_before_evidence_band": _empty_float(size),
        "node_score_before_snc": _empty_float(size),
        "node_score_before_local_closure": _empty_float(size),
        "node_score_before_mid_rank_gate": _empty_float(size),
        "node_score_intrinsic": _empty_float(size),
        "node_score_without_chain": _empty_float(size),
        "node_max": _empty_float(size),
        "node_high_count": _empty_float(size),
        "node_soft_sum": _empty_float(size),
        "chain_profile_bonus": _empty_float(size),
        "chain_profile_raw": _empty_float(size),
        "adaptive_memory_bonus": _empty_float(size),
        "adaptive_memory_quarantine_node": _empty_float(size),
        "adaptive_memory_update_node": _empty_float(size),
        "node_top_event": np.full((size,), -1, dtype=np.int64),
        "node_top_label": np.zeros((size,), dtype=np.int8),
        "node_top_component": np.full((size,), -1, dtype=np.int16),
        "top_component_text": {},
        "source_rank": np.zeros((size,), dtype=np.int64),
    }
    feature_rows = np.zeros((nodes.shape[0], len(NODE_RERANK_FEATURES)), dtype=np.float32)
    component_to_index: dict[str, int] = {}
    for idx, (row, node_raw) in enumerate(zip(rows, nodes.tolist())):
        node = int(node_raw)
        data["all_nodes"][node] = True
        data["node_score_final"][node] = np.float32(_float_row(row, "node_score", 0.0))
        data["node_score_before_rerank"][node] = np.float32(_float_row(row, "node_score_before_rerank", data["node_score_final"][node]))
        data["node_score_before_intrinsic_first"][node] = np.float32(_float_row(row, "node_score_before_intrinsic_first", data["node_score_final"][node]))
        data["node_score_before_evidence_band"][node] = np.float32(_float_row(row, "node_score_before_evidence_band", data["node_score_final"][node]))
        data["node_score_before_snc"][node] = np.float32(_float_row(row, "node_score_before_snc", data["node_score_final"][node]))
        data["node_score_before_local_closure"][node] = np.float32(_float_row(row, "node_score_before_local_closure", data["node_score_final"][node]))
        data["node_score_before_mid_rank_gate"][node] = np.float32(_float_row(row, "node_score_before_mid_rank_gate", data["node_score_final"][node]))
        data["node_score_intrinsic"][node] = np.float32(_float_row(row, "node_score_intrinsic", data["node_score_before_rerank"][node]))
        data["node_score_without_chain"][node] = np.float32(_float_row(row, "node_score_without_chain", 0.0))
        data["node_max"][node] = np.float32(_float_row(row, "node_max", 0.0))
        data["node_high_count"][node] = np.float32(_float_row(row, "node_high_count", 0.0))
        data["node_soft_sum"][node] = np.float32(_float_row(row, "node_soft_sum", 0.0))
        data["chain_profile_bonus"][node] = np.float32(_float_row(row, "chain_profile_bonus", 0.0))
        data["chain_profile_raw"][node] = np.float32(_float_row(row, "chain_profile_raw", 0.0))
        data["adaptive_memory_bonus"][node] = np.float32(_float_row(row, "adaptive_memory_bonus", 0.0))
        data["adaptive_memory_quarantine_node"][node] = np.float32(_float_row(row, "adaptive_memory_quarantined", 0.0))
        data["adaptive_memory_update_node"][node] = np.float32(_float_row(row, "adaptive_memory_update_count", 0.0))
        data["node_top_event"][node] = np.int64(_int_row(row, "top_event_pos", -1))
        label = _int_row(row, "top_event_label", 0)
        data["node_top_label"][node] = np.int8(label)
        if label == 2:
            data["positive_nodes"][node] = True
        elif label == 1:
            data["suspect_nodes"][node] = True
        component = str(row.get("top_component", "unknown"))
        if component not in component_to_index:
            component_to_index[component] = len(component_to_index)
        data["node_top_component"][node] = np.int16(component_to_index[component])
        data["top_component_text"][node] = component
        data["source_rank"][node] = np.int64(_int_row(row, "rank", idx + 1))

        prop_div = _float_row(row, "prop_source_diversity", 0.0)
        role_type_log = role_type_default
        if row.get("log_role_type_diversity", ""):
            role_type_log = _float_row(row, "log_role_type_diversity", role_type_default)
        elif row.get("role_type_diversity", ""):
            role_type_log = float(np.log1p(_float_row(row, "role_type_diversity", 0.0)))
        feature_rows[idx] = np.asarray(
            [
                data["node_max"][node],
                np.log1p(data["node_high_count"][node]),
                np.log1p(data["node_soft_sum"][node]),
                data["chain_profile_bonus"][node],
                _float_row(row, "association_discrepancy_max", 0.0),
                np.log1p(prop_div),
                role_type_log,
                _float_row(row, "semantic_rarity_stability", 0.0),
                data["chain_profile_raw"][node],
                _float_row(row, "local_chain_agreement", 0.0),
                _float_row(row, "compact_malicious_chain_signal", 0.0),
            ],
            dtype=np.float32,
        )
    data["rerank_keep"] = nodes.astype(np.int64, copy=False)
    data["rerank_matrix"] = feature_rows
    data["legacy_rows_loaded"] = int(nodes.shape[0])
    return data


def _base_score(data: dict[str, Any], mode: str) -> np.ndarray:
    key = {
        "before_rerank": "node_score_before_rerank",
        "before_intrinsic_first": "node_score_before_intrinsic_first",
        "before_evidence_band": "node_score_before_evidence_band",
        "before_snc": "node_score_before_snc",
        "before_local_closure": "node_score_before_local_closure",
        "before_mid_rank_gate": "node_score_before_mid_rank_gate",
        "final": "node_score_final",
    }[mode]
    return np.asarray(data[key], dtype=np.float32).copy()


def _valid_stats(summary: dict[str, Any]) -> list[dict[str, float]]:
    stats = _section(summary, "node_rerank").get("feature_stats", [])
    if not isinstance(stats, list) or len(stats) != len(NODE_RERANK_FEATURES):
        raise ValueError("source_json does not contain complete node_rerank.feature_stats")
    return stats


def _valid_calibrations(summary: dict[str, Any], mode: str, notes: list[str]) -> tuple[list[dict[str, Any]], str]:
    calibrations = _section(summary, "node_rerank").get("feature_calibrations", [])
    if str(mode) == "tail" and (not isinstance(calibrations, list) or len(calibrations) != len(NODE_RERANK_FEATURES)):
        notes.append("node_rerank_calibration_mode was tail but source_json lacks complete feature_calibrations; fell back to z mode.")
        return [], "z"
    return calibrations if isinstance(calibrations, list) else [], str(mode)


def _node_sweep(node_score: np.ndarray, all_nodes: np.ndarray, positive_nodes: np.ndarray, suspect_nodes: np.ndarray, topk_values: list[int]) -> list[dict[str, Any]]:
    ranked = rank_scores(node_score)
    sweep: list[dict[str, Any]] = []
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


def _apply_postprocess(args: argparse.Namespace, summary: dict[str, Any], data: dict[str, Any], notes: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    node_score = _base_score(data, str(args.base_score))
    rerank_keep = np.asarray(data["rerank_keep"], dtype=np.int64)
    rerank_matrix = np.asarray(data["rerank_matrix"], dtype=np.float32)
    size = int(node_score.shape[0])
    feature_stats = _valid_stats(summary)
    feature_weights = _feature_weights(args, summary)

    node_rerank_enabled = bool(_cfg_value(args.node_rerank_enabled, summary, "node_rerank", "enabled", True))
    node_rerank_mode = str(_cfg_value(args.node_rerank_mode, summary, "node_rerank", "mode", "additive"))
    node_rerank_weight = float(_cfg_value(args.node_rerank_weight, summary, "node_rerank", "weight", 0.75))
    node_rerank_clip = float(_cfg_value(args.node_rerank_clip, summary, "node_rerank", "clip", 8.0))
    node_rerank_calibration_mode = str(_cfg_value(args.node_rerank_calibration_mode, summary, "node_rerank", "calibration_mode", "tail"))
    feature_calibrations, node_rerank_calibration_mode = _valid_calibrations(summary, node_rerank_calibration_mode, notes)

    delta_values = _empty_float(size)
    support_values = _empty_float(size)
    intrinsic_support_values = _empty_float(size)
    prop_penalty_values = _empty_float(size)
    high_activity_penalty_values = _empty_float(size)
    meta: dict[str, Any] = {
        "node_rerank": {"enabled": node_rerank_enabled},
        "evidence_band_gate": {"enabled": False},
        "local_closure": {"enabled": False},
        "mid_rank_gate": {"enabled": False},
    }
    if node_rerank_enabled and rerank_keep.size > 0 and rerank_matrix.size > 0:
        (
            rerank_delta,
            rerank_support,
            rerank_prop_penalty,
            rerank_high_activity_penalty,
            _hard_support,
            _propagation_pressure,
            _family_count,
            intrinsic_rank_support,
        ) = _node_rerank_delta(
            rerank_matrix,
            feature_stats,
            feature_calibrations,
            node_rerank_calibration_mode,
            feature_weights,
            node_rerank_clip,
            float(_cfg_value(args.node_rerank_penalty_weight, summary, "node_rerank", "penalty_weight", 0.85)),
            float(_cfg_value(args.node_rerank_margin, summary, "node_rerank", "margin", 1.0)),
            float(_cfg_value(args.node_rerank_high_activity_weight, summary, "node_rerank", "high_activity_weight", 0.20)),
            float(_cfg_value(args.node_rerank_high_activity_margin, summary, "node_rerank", "high_activity_margin", 0.5)),
            float(_cfg_value(args.node_rerank_high_activity_cap, summary, "node_rerank", "high_activity_cap", 2.0)),
        )
        delta_values[rerank_keep] = rerank_delta
        support_values[rerank_keep] = rerank_support
        intrinsic_support_values[rerank_keep] = intrinsic_rank_support
        prop_penalty_values[rerank_keep] = rerank_prop_penalty
        high_activity_penalty_values[rerank_keep] = rerank_high_activity_penalty
        if node_rerank_mode == "replace":
            node_score = _empty_float(size)
            node_score[rerank_keep] = rerank_delta
        else:
            node_score = node_score + node_rerank_weight * delta_values
        meta["node_rerank"] = {
            "enabled": True,
            "mode": node_rerank_mode,
            "weight": node_rerank_weight,
            "calibration_mode": node_rerank_calibration_mode,
            "active_nodes": int(np.sum(np.abs(rerank_delta) > 0.0)),
            "max_delta": float(np.max(rerank_delta)) if rerank_delta.size else 0.0,
            "min_delta": float(np.min(rerank_delta)) if rerank_delta.size else 0.0,
            "feature_names": NODE_RERANK_FEATURES,
            "feature_weights": [float(x) for x in feature_weights.tolist()],
        }

    node_rank_mode = str(_cfg_value(args.node_rank_mode, summary, "node_rank", "mode", "score"))
    if node_rank_mode == "intrinsic_first":
        intrinsic = np.asarray(data["node_score_intrinsic"], dtype=np.float32)
        propagation_secondary = np.maximum(node_score - intrinsic, 0.0)
        tie_cap = float(_cfg_value(args.node_intrinsic_tie_cap, summary, "node_rank", "intrinsic_tie_cap", 1.0))
        if tie_cap > 0.0:
            propagation_secondary = np.minimum(propagation_secondary, tie_cap)
        node_score = (
            intrinsic
            + float(_cfg_value(args.node_intrinsic_support_weight, summary, "node_rank", "intrinsic_support_weight", 0.80)) * intrinsic_support_values
            + float(_cfg_value(args.node_intrinsic_tie_weight, summary, "node_rank", "intrinsic_tie_weight", 0.05)) * propagation_secondary
        ).astype(np.float32, copy=False)
    meta["node_rank"] = {
        "mode": node_rank_mode,
        "intrinsic_first_applied": bool(node_rank_mode == "intrinsic_first"),
    }

    adaptive_bonus = np.asarray(data["adaptive_memory_bonus"], dtype=np.float32)
    adaptive_quarantine = np.asarray(data["adaptive_memory_quarantine_node"], dtype=np.float32)
    adaptive_updates = np.asarray(data["adaptive_memory_update_node"], dtype=np.float32)

    evidence_enabled = bool(_cfg_value(args.evidence_band_gate_enabled, summary, "evidence_band_gate", "enabled", False))
    if evidence_enabled:
        evidence_mode = str(_cfg_value(args.evidence_band_calibration_mode, summary, "evidence_band_gate", "calibration_mode", "tail"))
        if evidence_mode == "tail" and not feature_calibrations:
            notes.append("evidence_band_calibration_mode was tail but no calibrations were available; fell back to z mode.")
            evidence_mode = "z"
        node_score, _arrays, evidence_meta = _apply_evidence_band_gate(
            node_score=node_score,
            rerank_keep=rerank_keep,
            rerank_matrix=rerank_matrix,
            feature_stats=feature_stats,
            feature_calibrations=feature_calibrations,
            adaptive_memory_bonus=adaptive_bonus,
            adaptive_memory_quarantine_node=adaptive_quarantine,
            adaptive_memory_update_node=adaptive_updates,
            calibration_mode=evidence_mode,
            clip=node_rerank_clip,
            candidate_local=float(_cfg_value(args.evidence_band_candidate_local, summary, "evidence_band_gate", "candidate_local", 1.0)),
            candidate_activity=float(_cfg_value(args.evidence_band_candidate_activity, summary, "evidence_band_gate", "candidate_activity", 2.0)),
            candidate_relation=float(_cfg_value(args.evidence_band_candidate_relation, summary, "evidence_band_gate", "candidate_relation", 1.0)),
            min_local=float(_cfg_value(args.evidence_band_min_local, summary, "evidence_band_gate", "min_local", 0.75)),
            min_relation=float(_cfg_value(args.evidence_band_min_relation, summary, "evidence_band_gate", "min_relation", 0.75)),
            min_assoc=float(_cfg_value(args.evidence_band_min_assoc, summary, "evidence_band_gate", "min_assoc", 0.75)),
            min_semantic=float(_cfg_value(args.evidence_band_min_semantic, summary, "evidence_band_gate", "min_semantic", 0.75)),
            min_role=float(_cfg_value(args.evidence_band_min_role, summary, "evidence_band_gate", "min_role", 0.50)),
            min_adaptive=float(_cfg_value(args.evidence_band_min_adaptive, summary, "evidence_band_gate", "min_adaptive", 0.50)),
            min_families=int(_cfg_value(args.evidence_band_min_families, summary, "evidence_band_gate", "min_families", 3)),
            weight=float(_cfg_value(args.evidence_band_weight, summary, "evidence_band_gate", "weight", 0.45)),
            boost_cap=float(_cfg_value(args.evidence_band_boost_cap, summary, "evidence_band_gate", "boost_cap", 0.85)),
            penalty_weight=float(_cfg_value(args.evidence_band_penalty_weight, summary, "evidence_band_gate", "penalty_weight", 0.55)),
            penalty_cap=float(_cfg_value(args.evidence_band_penalty_cap, summary, "evidence_band_gate", "penalty_cap", 2.5)),
            activity_weight=float(_cfg_value(args.evidence_band_activity_weight, summary, "evidence_band_gate", "activity_weight", 0.35)),
            activity_margin=float(_cfg_value(args.evidence_band_activity_margin, summary, "evidence_band_gate", "activity_margin", 0.75)),
            update_min_count=float(_cfg_value(args.evidence_band_update_min_count, summary, "evidence_band_gate", "update_min_count", 8.0)),
            update_weight=float(_cfg_value(args.evidence_band_update_weight, summary, "evidence_band_gate", "update_weight", 0.20)),
            update_cap=float(_cfg_value(args.evidence_band_update_cap, summary, "evidence_band_gate", "update_cap", 2.0)),
            prop_weight=float(_cfg_value(args.evidence_band_prop_weight, summary, "evidence_band_gate", "prop_weight", 0.15)),
            prop_cap=float(_cfg_value(args.evidence_band_prop_cap, summary, "evidence_band_gate", "prop_cap", 1.0)),
        )
        meta["evidence_band_gate"] = {"enabled": True, "calibration_mode": evidence_mode, **evidence_meta}

    local_enabled = bool(_cfg_value(args.local_closure_enabled, summary, "local_closure", "enabled", False))
    if local_enabled:
        ranked_before = rank_scores(node_score)
        node_score, _arrays, local_meta = _apply_local_closure_score(
            node_score=node_score,
            ranked_before_closure=ranked_before,
            rerank_keep=rerank_keep,
            rerank_matrix=rerank_matrix,
            feature_stats=feature_stats,
            adaptive_memory_bonus=adaptive_bonus,
            adaptive_memory_quarantine_node=adaptive_quarantine,
            adaptive_memory_update_node=adaptive_updates,
            clip=node_rerank_clip,
            protect_topk=int(_cfg_value(args.local_closure_protect_topk, summary, "local_closure", "protect_topk", 1800)),
            end_topk=int(_cfg_value(args.local_closure_end_topk, summary, "local_closure", "end_topk", 20000)),
            min_score_z=float(_cfg_value(args.local_closure_min_score_z, summary, "local_closure", "min_score_z", 1.0)),
            min_families=int(_cfg_value(args.local_closure_min_families, summary, "local_closure", "min_families", 6)),
            min_assoc_z=float(_cfg_value(args.local_closure_min_assoc_z, summary, "local_closure", "min_assoc_z", 1.0)),
            min_role_z=float(_cfg_value(args.local_closure_min_role_z, summary, "local_closure", "min_role_z", 1.0)),
            min_adaptive_z=float(_cfg_value(args.local_closure_min_adaptive_z, summary, "local_closure", "min_adaptive_z", 1.0)),
            weight=float(_cfg_value(args.local_closure_weight, summary, "local_closure", "weight", 0.55)),
            boost_cap=float(_cfg_value(args.local_closure_boost_cap, summary, "local_closure", "boost_cap", 1.0)),
            penalty_weight=float(_cfg_value(args.local_closure_penalty_weight, summary, "local_closure", "penalty_weight", 0.25)),
            penalty_cap=float(_cfg_value(args.local_closure_penalty_cap, summary, "local_closure", "penalty_cap", 2.0)),
            prop_penalty_weight=float(_cfg_value(args.local_closure_prop_penalty_weight, summary, "local_closure", "prop_penalty_weight", 0.15)),
            prop_penalty_cap=float(_cfg_value(args.local_closure_prop_penalty_cap, summary, "local_closure", "prop_penalty_cap", 1.0)),
            update_min_count=float(_cfg_value(args.local_closure_update_min_count, summary, "local_closure", "update_min_count", 8.0)),
            update_weight=float(_cfg_value(args.local_closure_update_weight, summary, "local_closure", "update_weight", 0.25)),
            update_cap=float(_cfg_value(args.local_closure_update_cap, summary, "local_closure", "update_cap", 2.0)),
            require_quarantine=bool(_cfg_value(args.local_closure_require_quarantine, summary, "local_closure", "require_quarantine", True)),
        )
        meta["local_closure"] = {"enabled": True, **local_meta}

    mid_enabled = bool(_cfg_value(args.mid_rank_gate_enabled, summary, "mid_rank_gate", "enabled", False))
    if mid_enabled:
        ranked_before = rank_scores(node_score)
        node_score, _arrays, mid_meta = _apply_mid_rank_gate(
            node_score=node_score,
            ranked_before_gate=ranked_before,
            rerank_keep=rerank_keep,
            rerank_matrix=rerank_matrix,
            feature_stats=feature_stats,
            adaptive_memory_bonus=adaptive_bonus,
            adaptive_memory_quarantine_node=adaptive_quarantine,
            adaptive_memory_update_node=adaptive_updates,
            clip=node_rerank_clip,
            protect_topk=int(_cfg_value(args.mid_rank_gate_protect_topk, summary, "mid_rank_gate", "protect_topk", 2000)),
            end_topk=int(_cfg_value(args.mid_rank_gate_end_topk, summary, "mid_rank_gate", "end_topk", 4000)),
            min_assoc_z=float(_cfg_value(args.mid_rank_gate_min_assoc_z, summary, "mid_rank_gate", "min_assoc_z", 1.0)),
            min_relation_z=float(_cfg_value(args.mid_rank_gate_min_relation_z, summary, "mid_rank_gate", "min_relation_z", 1.0)),
            min_adaptive_z=float(_cfg_value(args.mid_rank_gate_min_adaptive_z, summary, "mid_rank_gate", "min_adaptive_z", 1.0)),
            require_quarantine=bool(_cfg_value(args.mid_rank_gate_require_quarantine, summary, "mid_rank_gate", "require_quarantine", True)),
            penalty_weight=float(_cfg_value(args.mid_rank_gate_penalty_weight, summary, "mid_rank_gate", "penalty_weight", 0.80)),
            penalty_cap=float(_cfg_value(args.mid_rank_gate_penalty_cap, summary, "mid_rank_gate", "penalty_cap", 4.0)),
            benign_update_min_count=float(_cfg_value(args.mid_rank_gate_benign_update_min_count, summary, "mid_rank_gate", "benign_update_min_count", 8.0)),
            benign_update_weight=float(_cfg_value(args.mid_rank_gate_benign_update_weight, summary, "mid_rank_gate", "benign_update_weight", 0.20)),
            benign_update_cap=float(_cfg_value(args.mid_rank_gate_benign_update_cap, summary, "mid_rank_gate", "benign_update_cap", 2.0)),
        )
        meta["mid_rank_gate"] = {"enabled": True, **mid_meta}

    meta["node_arrays"] = {
        "node_rerank_delta": delta_values,
        "node_rerank_support": support_values,
        "node_intrinsic_rank_support": intrinsic_support_values,
        "node_rerank_prop_only_penalty": prop_penalty_values,
        "node_rerank_high_activity_penalty": high_activity_penalty_values,
    }
    return node_score.astype(np.float32, copy=False), meta


def _write_ranked_csv(path: str, ranked: np.ndarray, node_score: np.ndarray, data: dict[str, Any], meta: dict[str, Any], max_rows: int) -> None:
    arrays = meta["node_arrays"]
    limit = ranked.shape[0] if int(max_rows) <= 0 else min(int(max_rows), ranked.shape[0])
    fieldnames = [
        "rank",
        "node_id",
        "node_score",
        "source_rank",
        "source_node_score",
        "base_node_score_before_rerank",
        "node_score_intrinsic",
        "node_rerank_delta",
        "node_rerank_support",
        "node_intrinsic_rank_support",
        "node_rerank_prop_only_penalty",
        "node_rerank_high_activity_penalty",
        "node_max",
        "node_high_count",
        "node_soft_sum",
        "chain_profile_bonus",
        "chain_profile_raw",
        "adaptive_memory_bonus",
        "adaptive_memory_quarantined",
        "adaptive_memory_update_count",
        "top_event_pos",
        "top_event_label",
        "top_component",
    ]
    component_text = data.get("top_component_text", {})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, node_raw in enumerate(ranked[:limit].tolist(), start=1):
            node = int(node_raw)
            top_component = component_text.get(node, str(int(np.asarray(data["node_top_component"])[node])))
            writer.writerow(
                {
                    "rank": rank,
                    "node_id": node,
                    "node_score": float(node_score[node]),
                    "source_rank": int(np.asarray(data["source_rank"])[node]),
                    "source_node_score": float(np.asarray(data["node_score_final"], dtype=np.float32)[node]),
                    "base_node_score_before_rerank": float(np.asarray(data["node_score_before_rerank"], dtype=np.float32)[node]),
                    "node_score_intrinsic": float(np.asarray(data["node_score_intrinsic"], dtype=np.float32)[node]),
                    "node_rerank_delta": float(arrays["node_rerank_delta"][node]),
                    "node_rerank_support": float(arrays["node_rerank_support"][node]),
                    "node_intrinsic_rank_support": float(arrays["node_intrinsic_rank_support"][node]),
                    "node_rerank_prop_only_penalty": float(arrays["node_rerank_prop_only_penalty"][node]),
                    "node_rerank_high_activity_penalty": float(arrays["node_rerank_high_activity_penalty"][node]),
                    "node_max": float(np.asarray(data["node_max"], dtype=np.float32)[node]),
                    "node_high_count": float(np.asarray(data["node_high_count"], dtype=np.float32)[node]),
                    "node_soft_sum": float(np.asarray(data["node_soft_sum"], dtype=np.float32)[node]),
                    "chain_profile_bonus": float(np.asarray(data["chain_profile_bonus"], dtype=np.float32)[node]),
                    "chain_profile_raw": float(np.asarray(data["chain_profile_raw"], dtype=np.float32)[node]),
                    "adaptive_memory_bonus": float(np.asarray(data["adaptive_memory_bonus"], dtype=np.float32)[node]),
                    "adaptive_memory_quarantined": float(np.asarray(data["adaptive_memory_quarantine_node"], dtype=np.float32)[node]),
                    "adaptive_memory_update_count": float(np.asarray(data["adaptive_memory_update_node"], dtype=np.float32)[node]),
                    "top_event_pos": int(np.asarray(data["node_top_event"])[node]),
                    "top_event_label": int(np.asarray(data["node_top_label"])[node]),
                    "top_component": top_component,
                }
            )


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    if bool(args.cache_npz) == bool(args.ranked_csv):
        raise ValueError("Provide exactly one of --cache_npz or --ranked_csv")
    summary = _load_json(args.source_json)
    dataset = str(args.dataset or summary.get("dataset") or _section(summary, "target").get("dataset") or "")
    if not dataset:
        raise ValueError("--dataset is required when source_json does not contain dataset")
    notes: list[str] = []
    if args.cache_npz:
        data = _load_cache_npz(args.cache_npz)
    else:
        data = _load_ranked_csv(args.ranked_csv, float(args.legacy_role_type_log_default))
        notes.append("Legacy ranked_csv mode reconstructs rerank_matrix from exported columns. log_role_type_diversity is not available in older ranked CSV files and defaults to --legacy_role_type_log_default.")

    node_score, meta = _apply_postprocess(args, summary, data, notes)
    ranked = rank_scores(node_score)
    topk_values = _topk_values(args, summary, int(ranked.shape[0]))
    target = target_for_dataset(dataset)
    exact = bool(data.get("exact_masks_available", False))
    diagnostic_only = not exact
    if exact:
        sweep = _node_sweep(
            node_score,
            np.asarray(data["all_nodes"], dtype=bool),
            np.asarray(data["positive_nodes"], dtype=bool),
            np.asarray(data["suspect_nodes"], dtype=bool),
            topk_values,
        )
    elif bool(args.legacy_metrics_from_top_event_label):
        sweep = _node_sweep(
            node_score,
            np.asarray(data["all_nodes"], dtype=bool),
            np.asarray(data["positive_nodes"], dtype=bool),
            np.asarray(data["suspect_nodes"], dtype=bool),
            topk_values,
        )
        notes.append("Legacy metrics are diagnostic only: positive/suspect masks come from top_event_label in the ranked CSV, not from the official DB relaxed endpoint masks.")
    else:
        sweep = []

    best = best_sweep_row(sweep, target) if sweep else {}
    best_fp = best_under_fp_target(sweep, target) if sweep else {}
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{args.out_prefix}_ranked.csv")
    out_json = os.path.join(out_dir, f"{args.out_prefix}.json")
    _write_ranked_csv(out_csv, ranked, node_score, data, meta, int(args.max_ranked_rows))

    result = {
        "method": "tflr_node_postprocess_cache_rerank",
        "dataset": dataset,
        "source_json": os.path.abspath(args.source_json),
        "cache_npz": os.path.abspath(args.cache_npz) if args.cache_npz else "",
        "ranked_csv": os.path.abspath(args.ranked_csv) if args.ranked_csv else "",
        "input_mode": data["input_mode"],
        "diagnostic_only": bool(diagnostic_only),
        "ranked_csv_out": out_csv,
        "runtime_seconds": float(time.perf_counter() - started),
        "config": {
            "base_score": str(args.base_score),
            "topk_values": topk_values,
        },
        "postprocess": {k: v for k, v in meta.items() if k != "node_arrays"},
        "target": {"dataset": dataset, **target},
        "sweep": sweep,
        "best": best,
        "best_under_fp_target": best_fp,
        "notes": notes,
        "leakage_free_protocol": {
            "scoring_uses_ground_truth": False,
            "scoring_uses_test_labels": False,
            "scoring_uses_future_events": False,
            "scoring_uses_full_test_statistics": False,
            "evaluation_masks_used_after_ranking": bool(exact),
            "legacy_top_event_label_metrics_are_diagnostic_only": bool(diagnostic_only),
        },
        "recount_command_for_official_relaxed_metrics": (
            f"python scripts/tools/recompute_relaxed_metrics_db.py --dataset {dataset} "
            f"--result_dir {out_dir} --ranked_csv {out_csv} --out_json {os.path.join(out_dir, args.out_prefix + '_official_recount.json')}"
        ),
    }
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps({"out_json": out_json, "ranked_csv": out_csv, "diagnostic_only": diagnostic_only, "best_under_fp_target": best_fp}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
