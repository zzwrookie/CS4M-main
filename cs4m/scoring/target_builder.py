from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


SCORE_TARGET_EVENT_ACTION = "event_action_semantic"
SCORE_TARGET_NODE_PAIR = "node_pair_no_action"
SCORE_TARGET_WEIGHTED_EVENT = "weighted_event_semantic"
SCORE_TARGET_RESIDUAL_MIXTURE = "residual_mixture"
SUPPORTED_SCORE_TARGET_MODES = {
    SCORE_TARGET_EVENT_ACTION,
    SCORE_TARGET_NODE_PAIR,
    SCORE_TARGET_WEIGHTED_EVENT,
    SCORE_TARGET_RESIDUAL_MIXTURE,
}


def weighted_event_target(
    src_embedding: np.ndarray,
    action_embedding: np.ndarray,
    dst_embedding: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Return (src + alpha * action + dst) / (2 + alpha)."""
    alpha_value = float(alpha)
    if alpha_value < 0.0:
        raise ValueError("score target action alpha must be non-negative")
    src = np.asarray(src_embedding, dtype=np.float32)
    action = np.asarray(action_embedding, dtype=np.float32)
    dst = np.asarray(dst_embedding, dtype=np.float32)
    if src.shape != action.shape or src.shape != dst.shape:
        raise ValueError("score target embeddings must have matching shape")
    denom = np.float32(2.0 + alpha_value)
    return ((src + np.float32(alpha_value) * action + dst) / denom).astype(np.float32)


def node_pair_no_action_target(src_embedding: np.ndarray, dst_embedding: np.ndarray) -> np.ndarray:
    """Return mean(src_embedding, dst_embedding)."""
    src = np.asarray(src_embedding, dtype=np.float32)
    dst = np.asarray(dst_embedding, dtype=np.float32)
    if src.shape != dst.shape:
        raise ValueError("node-pair target embeddings must have matching shape")
    return ((src + dst) / np.float32(2.0)).astype(np.float32)


def residual_mixture_score(
    *,
    residual_no_action: np.ndarray,
    residual_event_action: np.ndarray,
    mixture_lambda: float,
) -> np.ndarray:
    """Return lambda * no-action residual + (1 - lambda) * event-action residual."""
    lam = float(mixture_lambda)
    if not (0.0 <= lam <= 1.0):
        raise ValueError("residual mixture lambda must be in [0, 1]")
    no_action = np.asarray(residual_no_action, dtype=np.float32)
    event_action = np.asarray(residual_event_action, dtype=np.float32)
    if no_action.shape != event_action.shape:
        raise ValueError("residual mixture arrays must have matching shape")
    return (np.float32(lam) * no_action + np.float32(1.0 - lam) * event_action).astype(
        np.float32,
    )


def validation_alert_budget_threshold(
    scores: np.ndarray,
    *,
    budget_per_1m: int,
    events_per_budget: int = 1_000_000,
) -> dict[str, Any]:
    """Derive a threshold from validation scores for an alert budget per 1M events."""
    values = np.asarray(scores, dtype=np.float32)
    if values.size <= 0:
        raise ValueError("validation alert budget threshold requires validation scores")
    budget = int(budget_per_1m)
    if budget <= 0:
        raise ValueError("validation alert budget must be positive")
    denominator = int(events_per_budget)
    if denominator <= 0:
        raise ValueError("events_per_budget must be positive")
    target_alert_count = int(round(float(budget) * float(values.size) / float(denominator)))
    target_alert_count = min(max(target_alert_count, 1), int(values.size))
    quantile = 1.0 - float(target_alert_count) / float(values.size)
    sorted_values = np.sort(values)
    threshold = float(sorted_values[int(values.size) - target_alert_count])
    validation_alert_count = int(np.count_nonzero(values >= np.float32(threshold)))
    return {
        "threshold_mode": "validation_alert_budget",
        "budget_per_1m": budget,
        "events_per_budget": denominator,
        "validation_event_count": int(values.size),
        "target_alert_count": int(target_alert_count),
        "validation_alert_count": int(validation_alert_count),
        "budget_quantile": float(quantile),
        "derived_threshold": float(threshold),
        "score_min": float(np.min(values)),
        "score_mean": float(np.mean(values)),
        "score_p99": float(np.quantile(values, 0.99)),
        "score_p999": float(np.quantile(values, 0.999)),
        "score_max": float(np.max(values)),
    }


def score_target_cache_fingerprint(
    *,
    dataset: str,
    run_only: str,
    state_model: str,
    score_head: str,
    conditional_head_arch: str,
    checkpoint_fingerprint: str,
    score_target_mode: str,
    score_target_action_alpha: float,
    residual_mixture_lambda: float,
    event_index_fingerprint: Mapping[str, Any],
    node_embedding_fingerprint: Mapping[str, Any],
    action_embedding_fingerprint: Mapping[str, Any],
    threshold_mode: str,
    threshold_quantile: float,
    endpoint_suppression_mode: str,
) -> dict[str, Any]:
    """Return validation-cache fingerprint for a score-target ablation."""
    mode = str(score_target_mode)
    if mode not in SUPPORTED_SCORE_TARGET_MODES:
        raise ValueError(f"unsupported score target mode: {mode}")
    payload = {
        "schema": "phase3g_score_target_validation_cache_v1",
        "dataset": str(dataset),
        "RUN_ONLY": str(run_only),
        "state_model": str(state_model),
        "score_head": str(score_head),
        "conditional_head_arch": str(conditional_head_arch),
        "checkpoint_fingerprint": str(checkpoint_fingerprint),
        "score_target_mode": mode,
        "score_target_action_alpha": float(score_target_action_alpha),
        "residual_mixture_lambda": float(residual_mixture_lambda),
        "event_index_fingerprint": dict(event_index_fingerprint),
        "node_embedding_fingerprint": dict(node_embedding_fingerprint),
        "action_embedding_fingerprint": dict(action_embedding_fingerprint),
        "threshold_mode": str(threshold_mode),
        "threshold_quantile": float(threshold_quantile),
        "endpoint_suppression_mode": str(endpoint_suppression_mode),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["fingerprint_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload
