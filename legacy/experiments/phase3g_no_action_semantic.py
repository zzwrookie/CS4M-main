from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


NO_ACTION_SCORE_HEAD = "conditional_no_action_semantic"
NO_ACTION_TARGET_MODE = "mean_src_dst_no_action"
NO_ACTION_HEAD_SCHEMA_V3 = "phase3g_conditional_no_action_semantic_head_v3"
NODE_PAIR_SEMANTIC_TARGET = "node_pair_semantic_target"
NODE_PAIR_SEMANTIC_TARGET_ID = np.int8(2)


@dataclass
class NoActionSemanticHeadConfig:
    """Configuration for the Phase3G v3 no-action semantic head."""

    input_dim: int
    rank: int = 32
    output_dim: int = 64
    loss_type: str = "cosine"
    seed: int = 31
    node_repr_fusion: str = "simple_mean"
    target_mode: str = NO_ACTION_TARGET_MODE
    score_head: str = NO_ACTION_SCORE_HEAD


class NoActionSemanticHead:
    """Single low-rank head predicting mean(src_embedding, dst_embedding)."""

    def __init__(
        self,
        config: NoActionSemanticHeadConfig,
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
    def _validate_config(config: NoActionSemanticHeadConfig) -> None:
        if int(config.input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(config.rank) <= 0:
            raise ValueError("rank must be positive")
        if int(config.output_dim) <= 0:
            raise ValueError("output_dim must be positive")
        if str(config.loss_type) not in {"cosine", "mse"}:
            raise ValueError("loss_type must be cosine or mse")
        if str(config.node_repr_fusion) != "simple_mean":
            raise ValueError("conditional_no_action_semantic requires simple_mean fusion")
        if str(config.target_mode) != NO_ACTION_TARGET_MODE:
            raise ValueError("unsupported no-action target mode")
        if str(config.score_head) != NO_ACTION_SCORE_HEAD:
            raise ValueError("unsupported no-action score head")

    @staticmethod
    def _checked_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"{name} shape mismatch: expected {shape}, got {array.shape}")
        if not bool(np.all(np.isfinite(array))):
            raise ValueError(f"{name} must contain only finite values")
        return array.astype(np.float32, copy=True)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return predicted no-action semantic vectors."""
        context = np.asarray(x, dtype=np.float32)
        one_dimensional = False
        if context.ndim == 1:
            if context.shape != (int(self.config.input_dim),):
                raise ValueError("context shape mismatch")
            one_dimensional = True
            context = context.reshape(1, -1)
        elif context.ndim == 2:
            if int(context.shape[1]) != int(self.config.input_dim):
                raise ValueError("context batch shape mismatch")
        else:
            raise ValueError("context must be one- or two-dimensional")
        pred = context @ self.w1 @ self.w2 + self.bias
        if one_dimensional:
            return pred.reshape(-1).astype(np.float32, copy=False)
        return pred.astype(np.float32, copy=False)

    def fingerprint(self) -> dict[str, Any]:
        """Return a stable fingerprint over config and weights."""
        payload = {
            "schema": NO_ACTION_HEAD_SCHEMA_V3,
            "config": asdict(self.config),
            "w1_sha256": hashlib.sha256(self.w1.tobytes(order="C")).hexdigest(),
            "w2_sha256": hashlib.sha256(self.w2.tobytes(order="C")).hexdigest(),
            "bias_sha256": hashlib.sha256(self.bias.tobytes(order="C")).hexdigest(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["fingerprint_sha256"] = hashlib.sha256(encoded).hexdigest()
        return payload

    def save(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Persist no-action head weights and metadata."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_payload = dict(metadata or {})
        fingerprint = self.fingerprint()
        for key in (
            "dataset",
            "state_model",
            "event_index_fingerprint",
            "node_embedding_fingerprint",
            "action_embedding_fingerprint",
            "conditional_memmap_fingerprint",
            "target_case_fingerprint",
        ):
            if key in metadata_payload:
                fingerprint[key] = metadata_payload[key]
        fingerprint["schema"] = NO_ACTION_HEAD_SCHEMA_V3
        fingerprint["score_head"] = NO_ACTION_SCORE_HEAD
        fingerprint["target_mode"] = NO_ACTION_TARGET_MODE
        encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        fingerprint["fingerprint_sha256"] = hashlib.sha256(encoded).hexdigest()
        payload = {
            "schema": NO_ACTION_HEAD_SCHEMA_V3,
            "config": asdict(self.config),
            "metadata": metadata_payload,
            "fingerprint": fingerprint,
            "w1": self.w1,
            "w2": self.w2,
            "bias": self.bias,
        }
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> tuple["NoActionSemanticHead", dict[str, Any]]:
        """Load a v3 no-action semantic head checkpoint."""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"no-action head checkpoint not found: {checkpoint_path}")
        with checkpoint_path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, Mapping):
            raise TypeError("no-action checkpoint payload must be a mapping")
        schema = str(payload.get("schema", ""))
        if schema != NO_ACTION_HEAD_SCHEMA_V3:
            raise ValueError("unsupported no-action checkpoint schema")
        config = NoActionSemanticHeadConfig(**dict(payload["config"]))
        head = cls(
            config,
            w1=np.asarray(payload["w1"], dtype=np.float32),
            w2=np.asarray(payload["w2"], dtype=np.float32),
            bias=np.asarray(payload["bias"], dtype=np.float32),
        )
        metadata = dict(payload.get("metadata", {}))
        metadata["checkpoint_schema"] = schema
        metadata["score_head"] = NO_ACTION_SCORE_HEAD
        metadata["target_mode"] = NO_ACTION_TARGET_MODE
        metadata["fingerprint"] = dict(payload.get("fingerprint", head.fingerprint()))
        return head, metadata


def select_no_action_target(
    row: np.void,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Return mean(src_embedding, dst_embedding) and the no-action target case."""
    del action_embeddings
    src_idx = int(row["src_node_idx"])
    dst_idx = int(row["dst_node_idx"])
    src = np.asarray(node_embeddings[src_idx], dtype=np.float32)
    dst = np.asarray(node_embeddings[dst_idx], dtype=np.float32)
    return ((src + dst) / np.float32(2.0)).astype(np.float32), NODE_PAIR_SEMANTIC_TARGET


def no_action_memmap_fingerprint(
    *,
    dataset: str,
    state_model: str,
    x_fingerprint: Mapping[str, Any],
    event_index_fingerprint: Mapping[str, Any],
    node_embedding_fingerprint: Mapping[str, Any],
    action_embedding_fingerprint: Mapping[str, Any],
    count: int,
    input_dim: int,
    output_dim: int,
) -> dict[str, Any]:
    """Return a stable fingerprint for a v3 no-action train memmap."""
    payload = {
        "schema": "phase3g_no_action_train_memmap_v1",
        "dataset": str(dataset),
        "score_head": NO_ACTION_SCORE_HEAD,
        "target_mode": NO_ACTION_TARGET_MODE,
        "target_case": NODE_PAIR_SEMANTIC_TARGET,
        "target_logic": "mean_src_dst_no_action",
        "node_repr_fusion": "simple_mean",
        "state_model": str(state_model),
        "num_events": int(count),
        "input_dim": int(input_dim),
        "output_dim": int(output_dim),
        "x_fingerprint": dict(x_fingerprint),
        "event_index_fingerprint": dict(event_index_fingerprint),
        "node_embedding_fingerprint": dict(node_embedding_fingerprint),
        "action_embedding_fingerprint": dict(action_embedding_fingerprint),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["fingerprint_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload
