"""Conditional cache, memmap path, fingerprint, and validation-cache helpers."""

from __future__ import annotations

from scripts.pipeline.config.runtime_config import *


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
        "action_id": root / f"action_id_{key}.memmap",
        "src_type_id": root / f"src_type_id_{key}.memmap",
        "dst_type_id": root / f"dst_type_id_{key}.memmap",
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



def _phase3g_conditional_threshold_for_case(
    meta: Mapping[str, Any],
    case_name: str,
) -> float:
    thresholds = dict(meta.get("thresholds_by_target_case", {}))
    if str(case_name) in thresholds:
        return float(dict(thresholds[str(case_name)]).get("threshold", meta.get("threshold", 0.0)))
    return float(meta.get("threshold", 0.0))


# Explicit cross-module imports; replaces the migration namespace bridge.
from scripts.pipeline.features.conditional_context import (
    _phase3g_action_cache_dir,
    _phase3g_conditional_fingerprint,
    _phase3g_conditional_scores_stream,
    _phase3g_file_fingerprint,
)
from scripts.pipeline.io.event_artifacts import _phase3e_load_json, stable_json_hash
from scripts.pipeline.state.online_state_runtime import _stage_log
