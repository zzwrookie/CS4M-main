"""Phase3E state runtime, checkpoint, calibration, and read-update helpers."""

from __future__ import annotations

from scripts.pipeline.config.runtime_config import *
from scripts.pipeline.config.runtime_config import (
    _PHASE3E_CHECKPOINT_FORBIDDEN_KEYS,
    _PHASE3E_METADATA_ALLOWED_KEYS,
    _PHASE3E_METADATA_LEAKAGE_KEY_TERMS,
    _PHASE3E_METADATA_MAX_LIST_LENGTH,
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
    setattr(model, "phase3e_runtime_role", "state_runtime_only")
    setattr(model, "phase3e_lowrank_final_score_enabled", False)
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


def _phase3f_e2_none_context_from_numeric_row(*_args: object, **_kwargs: object) -> tuple:
    """Legacy 146-dim Phase3E context builder disabled in the active pipeline."""
    _phase3e_require_legacy_head_path_disabled()



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


# Explicit cross-module imports; replaces the migration namespace bridge.
from scripts.pipeline.checks.preflight import (
    _config_with_latent_dim,
    _effective_latent_dim,
    _latent_dim_from_vocab_size,
    _make_residual_embedding_config,
)
from scripts.pipeline.features.semantic_features import (
    _observe_process_semantic_audit,
    raw_action_token,
    residual_text,
    row_fields,
)
from scripts.pipeline.io.cache_payloads import _validate_sspm_state
from scripts.pipeline.io.event_artifacts import (
    _embedder_loaded,
    _model_update_gate_summary,
    _phase3e_entity_type_name,
    _phase3e_event_index_array,
    _phase3e_require_legacy_head_path_disabled,
    _phase3e_validate_index_row,
    stable_json_hash,
)
from scripts.pipeline.outputs.alert_output import _current_rss_mb
from scripts.pipeline.outputs.metrics_summary import _encode_row, _ofsm_empty_node_snapshot
