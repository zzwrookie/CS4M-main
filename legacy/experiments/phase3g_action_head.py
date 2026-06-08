from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass
class ActionHeadConfig:
    """Configuration for the Phase3G low-rank action-prediction head."""

    input_dim: int
    rank: int = 32
    output_dim: int = 10
    seed: int = 17
    node_repr_fusion: str = "simple_mean"


class ActionHead:
    """Low-rank linear classifier for ORTHRUS action prediction."""

    def __init__(
        self,
        config: ActionHeadConfig,
        *,
        w1: np.ndarray | None = None,
        w2: np.ndarray | None = None,
        bias: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self._validate_config(config)
        rng = np.random.default_rng(int(config.seed))
        if w1 is None:
            scale1 = 1.0 / np.sqrt(max(int(config.input_dim), 1))
            w1 = rng.normal(0.0, scale1, size=(config.input_dim, config.rank))
        if w2 is None:
            scale2 = 1.0 / np.sqrt(max(int(config.rank), 1))
            w2 = rng.normal(0.0, scale2, size=(config.rank, config.output_dim))
        if bias is None:
            bias = np.zeros((config.output_dim,), dtype=np.float32)
        self.w1 = self._checked_array(w1, (config.input_dim, config.rank), "w1")
        self.w2 = self._checked_array(w2, (config.rank, config.output_dim), "w2")
        self.bias = self._checked_array(bias, (config.output_dim,), "bias")

    @staticmethod
    def _validate_config(config: ActionHeadConfig) -> None:
        if int(config.input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(config.rank) <= 0:
            raise ValueError("rank must be positive")
        if int(config.output_dim) <= 1:
            raise ValueError("output_dim must be greater than one")
        if str(config.node_repr_fusion) != "simple_mean":
            raise ValueError("Phase3G v1 supports NODE_REPR_FUSION=simple_mean")

    @staticmethod
    def _checked_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"{name} shape mismatch: expected {shape}, got {array.shape}")
        if not bool(np.all(np.isfinite(array))):
            raise ValueError(f"{name} must contain only finite values")
        return array.astype(np.float32, copy=True)

    def logits(self, x: np.ndarray) -> np.ndarray:
        """Return action logits for one context row or a context batch."""
        context = np.asarray(x, dtype=np.float32)
        if context.ndim == 1:
            if context.shape != (int(self.config.input_dim),):
                raise ValueError("context shape mismatch")
        elif context.ndim == 2:
            if int(context.shape[1]) != int(self.config.input_dim):
                raise ValueError("context batch shape mismatch")
        else:
            raise ValueError("context must be one- or two-dimensional")
        return (context @ self.w1 @ self.w2 + self.bias).astype(np.float32, copy=False)

    def fingerprint(self) -> dict[str, Any]:
        """Return a stable fingerprint over head config and weights."""
        payload = {
            "schema": "phase3g_action_head_v1",
            "config": asdict(self.config),
            "w1_sha256": hashlib.sha256(self.w1.tobytes(order="C")).hexdigest(),
            "w2_sha256": hashlib.sha256(self.w2.tobytes(order="C")).hexdigest(),
            "bias_sha256": hashlib.sha256(self.bias.tobytes(order="C")).hexdigest(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["fingerprint_sha256"] = hashlib.sha256(encoded).hexdigest()
        return payload

    def save(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Persist action-head weights and primitive metadata."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "phase3g_action_head_v1",
            "config": asdict(self.config),
            "w1": self.w1,
            "w2": self.w2,
            "bias": self.bias,
            "metadata": dict(metadata or {}),
            "fingerprint": self.fingerprint(),
        }
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> tuple["ActionHead", dict[str, Any]]:
        """Load an action-head checkpoint."""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"action head checkpoint not found: {checkpoint_path}")
        with checkpoint_path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, Mapping):
            raise TypeError("action head checkpoint payload must be a mapping")
        if str(payload.get("schema", "")) != "phase3g_action_head_v1":
            raise ValueError("unsupported action head checkpoint schema")
        config = ActionHeadConfig(**dict(payload["config"]))
        head = cls(
            config,
            w1=np.asarray(payload["w1"], dtype=np.float32),
            w2=np.asarray(payload["w2"], dtype=np.float32),
            bias=np.asarray(payload["bias"], dtype=np.float32),
        )
        metadata = dict(payload.get("metadata", {}))
        metadata["fingerprint"] = dict(payload.get("fingerprint", head.fingerprint()))
        return head, metadata


def build_action_context(
    row: np.void,
    node_embeddings: np.ndarray,
    state_array: np.ndarray,
    has_state: np.ndarray,
    type_one_hot: np.ndarray,
) -> np.ndarray:
    """Build the Phase3G simple-mean action-prediction context."""
    src_idx = int(row["src_node_idx"])
    dst_idx = int(row["dst_node_idx"])
    src_type = int(row["src_type_id"])
    dst_type = int(row["dst_type_id"])
    src_static = np.asarray(node_embeddings[src_idx], dtype=np.float32)
    dst_static = np.asarray(node_embeddings[dst_idx], dtype=np.float32)
    src_repr = _fused_node_repr(src_static, np.asarray(state_array[src_idx]), bool(has_state[src_idx]))
    dst_repr = _fused_node_repr(dst_static, np.asarray(state_array[dst_idx]), bool(has_state[dst_idx]))
    return np.concatenate(
        [
            src_repr,
            dst_repr,
            np.asarray(type_one_hot[src_type], dtype=np.float32),
            np.asarray(type_one_hot[dst_type], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=False)


def _fused_node_repr(static: np.ndarray, state: np.ndarray, has_state: bool) -> np.ndarray:
    if not bool(has_state):
        return np.asarray(static, dtype=np.float32).astype(np.float32, copy=False)
    return ((np.asarray(static, dtype=np.float32) + np.asarray(state, dtype=np.float32)) / 2.0).astype(
        np.float32,
        copy=False,
    )


def action_nll_scores_from_logits(logits: np.ndarray, action_ids: np.ndarray) -> np.ndarray:
    """Return `-log softmax(logits)[true_action]` for each row."""
    raw_logits = np.asarray(logits, dtype=np.float32)
    actions = np.asarray(action_ids, dtype=np.int64)
    if raw_logits.ndim != 2:
        raise ValueError("logits must be two-dimensional")
    if actions.shape != (int(raw_logits.shape[0]),):
        raise ValueError("action_ids must have one value per logits row")
    if np.any(actions < 0) or np.any(actions >= int(raw_logits.shape[1])):
        raise ValueError("action_ids contain out-of-range values")
    max_logits = np.max(raw_logits, axis=1, keepdims=True)
    shifted = raw_logits - max_logits
    logsumexp = np.log(np.sum(np.exp(shifted), axis=1)) + max_logits[:, 0]
    true_logits = raw_logits[np.arange(int(raw_logits.shape[0])), actions]
    return (logsumexp - true_logits).astype(np.float32)


def score_summary(scores: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    """Return compact quantile summary for action scores."""
    values = np.asarray(scores, dtype=np.float32)
    if values.size == 0:
        return {
            "count": 0,
            "score_min": 0.0,
            "score_mean": 0.0,
            "score_std": 0.0,
            "score_p99": 0.0,
            "score_p999": 0.0,
            "score_p9995": 0.0,
            "score_p9999": 0.0,
            "score_max": 0.0,
            "final_threshold": None if threshold is None else float(threshold),
            "above_threshold_count": 0,
        }
    threshold_value = None if threshold is None else float(threshold)
    return {
        "count": int(values.size),
        "score_min": float(np.min(values)),
        "score_mean": float(np.mean(values)),
        "score_std": float(np.std(values)),
        "score_p99": float(np.quantile(values, 0.99, method="higher")),
        "score_p999": float(np.quantile(values, 0.999, method="higher")),
        "score_p9995": float(np.quantile(values, 0.9995, method="higher")),
        "score_p9999": float(np.quantile(values, 0.9999, method="higher")),
        "score_max": float(np.max(values)),
        "final_threshold": threshold_value,
        "above_threshold_count": (
            0 if threshold_value is None else int(np.count_nonzero(values >= threshold_value))
        ),
    }


def quantile_threshold(scores: np.ndarray, quantile: float) -> float:
    """Return a deterministic validation quantile threshold."""
    values = np.asarray(scores, dtype=np.float32)
    if values.size <= 0:
        raise ValueError("scores must not be empty")
    q = float(quantile)
    if not (0.0 < q <= 1.0):
        raise ValueError("quantile must be in (0, 1]")
    return float(np.quantile(values, q, method="higher"))


def write_validation_cache(
    cache_dir: str | Path,
    *,
    scores: np.ndarray,
    fingerprint: Mapping[str, Any],
    threshold_mode: str,
    threshold_quantile: float,
) -> dict[str, Any]:
    """Write action validation scores and threshold metadata."""
    output_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(scores, dtype=np.float32)
    score_path = output_dir / "validation_action_scores.memmap"
    loss_path = output_dir / "validation_action_loss.memmap"
    mapped_scores = np.memmap(score_path, dtype=np.float32, mode="w+", shape=values.shape)
    mapped_scores[:] = values[:]
    mapped_scores.flush()
    del mapped_scores
    mapped_loss = np.memmap(loss_path, dtype=np.float32, mode="w+", shape=values.shape)
    mapped_loss[:] = values[:]
    mapped_loss.flush()
    del mapped_loss

    if str(threshold_mode) == "validation_max":
        threshold = float(np.max(values)) if values.size else 0.0
    elif str(threshold_mode) == "quantile":
        threshold = quantile_threshold(values, float(threshold_quantile))
    else:
        raise ValueError(f"unsupported action validation threshold mode: {threshold_mode}")
    summary = score_summary(values, threshold)
    summary.update(
        {
            "threshold_mode": str(threshold_mode),
            "threshold_quantile": float(threshold_quantile),
        },
    )
    fingerprint_payload = dict(fingerprint)
    meta = {
        "schema": "phase3g_action_validation_cache_v1",
        "fingerprint": fingerprint_payload,
        "num_scores": int(values.size),
        "score_dtype": "float32",
        "validation_action_scores": str(score_path),
        "validation_action_loss": str(loss_path),
        "threshold": float(threshold),
        "summary": summary,
        "created_unix_time": float(time.time()),
    }
    (output_dir / "validation_score_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validation_threshold_cache.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validation_calibration_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def load_validation_cache(
    cache_dir: str | Path,
    *,
    expected_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Load validation cache metadata and fail fast on fingerprint mismatch."""
    cache_path = Path(cache_dir) / "validation_threshold_cache.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"action validation cache not found: {cache_path}")
    meta = json.loads(cache_path.read_text(encoding="utf-8"))
    if dict(meta.get("fingerprint", {})) != dict(expected_fingerprint):
        raise ValueError("action validation cache fingerprint mismatch")
    score_path = Path(str(meta["validation_action_scores"]))
    if not score_path.exists():
        raise FileNotFoundError(f"validation action scores missing: {score_path}")
    return meta

