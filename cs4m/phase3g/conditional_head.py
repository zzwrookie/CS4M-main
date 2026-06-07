from __future__ import annotations

import hashlib
import json
import pickle
import time
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EVENT_SEMANTIC_TARGET = "event_semantic_target"
BOTH_COLD_ACTION_TARGET = "both_cold_action_target"
EVENT_SEMANTIC_TARGET_ID = np.int8(0)
BOTH_COLD_ACTION_TARGET_ID = np.int8(1)
TARGET_CASE_ID_TO_NAME = {
    int(EVENT_SEMANTIC_TARGET_ID): EVENT_SEMANTIC_TARGET,
    int(BOTH_COLD_ACTION_TARGET_ID): BOTH_COLD_ACTION_TARGET,
}
TARGET_CASE_NAME_TO_ID = {
    EVENT_SEMANTIC_TARGET: int(EVENT_SEMANTIC_TARGET_ID),
    BOTH_COLD_ACTION_TARGET: int(BOTH_COLD_ACTION_TARGET_ID),
}
CONDITIONAL_GROUP_THRESHOLD_MODE = "conditional_target_action_type_group_quantile"
CONDITIONAL_GROUP_MIN_COUNT_DEFAULT = 1000
CONDITIONAL_LOW_SUPPORT_POLICY_DEFAULT = "conservative_max"
CONDITIONAL_LOW_SUPPORT_MARGIN_DEFAULT = 0.25
CONDITIONAL_UNSEEN_GROUP_POLICY_DEFAULT = "observation_only"
CONDITIONAL_GLOBAL_EXTREME_QUANTILE_DEFAULT = 0.9999
CONDITIONAL_ADAPTIVE_MARGIN_N1_DEFAULT = 50
CONDITIONAL_ADAPTIVE_MARGIN_N2_DEFAULT = 200
CONDITIONAL_ADAPTIVE_MARGIN_LOW_DEFAULT = 0.15
CONDITIONAL_ADAPTIVE_MARGIN_MID_DEFAULT = 0.05
CONDITIONAL_ADAPTIVE_MARGIN_HIGH_DEFAULT = 0.02
CONDITIONAL_ACTION_EMBEDDING_SCORE_HEAD = "conditional_action_embedding"
CONDITIONAL_HEAD_ARCH_SHARED_V1 = "shared_lowrank_v1"
CONDITIONAL_HEAD_ARCH_DUAL_V2 = "dual_lowrank_by_target_case_v2"
CONDITIONAL_HEAD_SCHEMA_SHARED_V1 = "phase3g_conditional_semantic_head_v1"
CONDITIONAL_HEAD_SCHEMA_DUAL_V2 = "phase3g_conditional_semantic_dual_head_v2"
GROUP_LEVEL_TARGET_ACTION_SRC_DST = "level1_target_action_src_dst"
GROUP_LEVEL_TARGET_ACTION = "level2_target_action"
GROUP_LEVEL_TARGET_SRC_DST = "level3_target_src_dst"
GROUP_LEVEL_TARGET_CASE = "level4_target_case"
GROUP_LEVEL_GLOBAL = "level5_global_conditional"
CONDITIONAL_GROUP_SUMMARY_FIELDS = [
    "target_case",
    "action_id",
    "action_name",
    "src_type_id",
    "src_type_name",
    "dst_type_id",
    "dst_type_name",
    "validation_count",
    "test_count",
    "threshold",
    "threshold_level",
    "low_support_policy",
    "group_validation_max",
    "parent_threshold",
    "global_threshold",
    "final_threshold_source",
    "adaptive_margin_used",
    "validation_count_bucket",
    "val_p99",
    "val_p999",
    "val_p9995",
    "val_p9999",
    "val_max",
    "test_p99",
    "test_p999",
    "test_p9995",
    "test_p9999",
    "test_max",
    "score_quantile_method",
    "alert_count",
    "tp",
    "fp",
    "precision",
    "recall",
]


@dataclass
class ConditionalSemanticHeadConfig:
    """Configuration for the Phase3G conditional semantic head."""

    input_dim: int
    rank: int = 32
    output_dim: int = 64
    loss_type: str = "cosine"
    seed: int = 23
    node_repr_fusion: str = "simple_mean"
    target_case_mode: str = "conditional_cold_action_else_event"
    conditional_head_arch: str = ""
    head_arch: str = ""
    target_case_head_mode: str = ""

    def __post_init__(self) -> None:
        if not str(self.head_arch).strip() and not str(self.conditional_head_arch).strip():
            self.conditional_head_arch = CONDITIONAL_HEAD_ARCH_SHARED_V1
            self.head_arch = CONDITIONAL_HEAD_ARCH_SHARED_V1
        elif not str(self.head_arch).strip():
            self.head_arch = str(self.conditional_head_arch)
        elif not str(self.conditional_head_arch).strip():
            self.conditional_head_arch = str(self.head_arch)
        if str(self.conditional_head_arch) != str(self.head_arch):
            raise ValueError("conditional_head_arch and head_arch must match")
        if str(self.conditional_head_arch) == CONDITIONAL_HEAD_ARCH_DUAL_V2:
            self.target_case_head_mode = "target_case_specific"
        elif not str(self.target_case_head_mode).strip():
            self.target_case_head_mode = "shared"


class ConditionalSemanticHead:
    """Low-rank 64-dimensional regressor for conditional semantic scoring."""

    def __init__(
        self,
        config: ConditionalSemanticHeadConfig,
        *,
        w1: np.ndarray | None = None,
        w2: np.ndarray | None = None,
        bias: np.ndarray | None = None,
        event_w1: np.ndarray | None = None,
        event_w2: np.ndarray | None = None,
        event_bias: np.ndarray | None = None,
        action_w1: np.ndarray | None = None,
        action_w2: np.ndarray | None = None,
        action_bias: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self._validate_config(config)
        rng = np.random.default_rng(int(config.seed))
        if self.is_dual_head:
            if event_w1 is None:
                scale1 = 1.0 / np.sqrt(max(int(config.input_dim), 1))
                event_w1 = rng.normal(0.0, scale1, size=(config.input_dim, config.rank))
            if event_w2 is None:
                scale2 = 1.0 / np.sqrt(max(int(config.rank), 1))
                event_w2 = rng.normal(0.0, scale2, size=(config.rank, config.output_dim))
            if event_bias is None:
                event_bias = np.zeros((config.output_dim,), dtype=np.float32)
            if action_w1 is None:
                scale1 = 1.0 / np.sqrt(max(int(config.input_dim), 1))
                action_w1 = rng.normal(0.0, scale1, size=(config.input_dim, config.rank))
            if action_w2 is None:
                scale2 = 1.0 / np.sqrt(max(int(config.rank), 1))
                action_w2 = rng.normal(0.0, scale2, size=(config.rank, config.output_dim))
            if action_bias is None:
                action_bias = np.zeros((config.output_dim,), dtype=np.float32)
            self.event_w1 = self._checked_array(
                event_w1,
                (config.input_dim, config.rank),
                "event_w1",
            )
            self.event_w2 = self._checked_array(
                event_w2,
                (config.rank, config.output_dim),
                "event_w2",
            )
            self.event_bias = self._checked_array(event_bias, (config.output_dim,), "event_bias")
            self.action_w1 = self._checked_array(
                action_w1,
                (config.input_dim, config.rank),
                "action_w1",
            )
            self.action_w2 = self._checked_array(
                action_w2,
                (config.rank, config.output_dim),
                "action_w2",
            )
            self.action_bias = self._checked_array(
                action_bias,
                (config.output_dim,),
                "action_bias",
            )
            self.w1 = self.event_w1
            self.w2 = self.event_w2
            self.bias = self.event_bias
            return
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
        self.event_w1 = self.w1
        self.event_w2 = self.w2
        self.event_bias = self.bias
        self.action_w1 = self.w1
        self.action_w2 = self.w2
        self.action_bias = self.bias

    @property
    def is_dual_head(self) -> bool:
        """Return true when target cases use separate low-rank heads."""
        return str(self.config.conditional_head_arch) == CONDITIONAL_HEAD_ARCH_DUAL_V2

    @staticmethod
    def _validate_config(config: ConditionalSemanticHeadConfig) -> None:
        if int(config.input_dim) <= 0:
            raise ValueError("input_dim must be positive")
        if int(config.rank) <= 0:
            raise ValueError("rank must be positive")
        if int(config.output_dim) <= 0:
            raise ValueError("output_dim must be positive")
        if str(config.loss_type) not in {"cosine", "mse"}:
            raise ValueError("CONDITIONAL_SEMANTIC_LOSS must be cosine or mse")
        if str(config.node_repr_fusion) != "simple_mean":
            raise ValueError("conditional_action_semantic requires simple_mean fusion")
        if str(config.target_case_mode) != "conditional_cold_action_else_event":
            raise ValueError("unsupported conditional target case mode")
        if str(config.conditional_head_arch) not in {
            CONDITIONAL_HEAD_ARCH_SHARED_V1,
            CONDITIONAL_HEAD_ARCH_DUAL_V2,
        }:
            raise ValueError("unsupported conditional head arch")
        if (
            str(config.conditional_head_arch) == CONDITIONAL_HEAD_ARCH_DUAL_V2
            and str(config.target_case_head_mode) != "target_case_specific"
        ):
            raise ValueError("dual-head v2 requires target_case_specific head mode")
        if (
            str(config.conditional_head_arch) == CONDITIONAL_HEAD_ARCH_SHARED_V1
            and str(config.target_case_head_mode) not in {"", "shared"}
        ):
            raise ValueError("shared lowrank v1 requires shared target-case head mode")

    @staticmethod
    def _checked_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"{name} shape mismatch: expected {shape}, got {array.shape}")
        if not bool(np.all(np.isfinite(array))):
            raise ValueError(f"{name} must contain only finite values")
        return array.astype(np.float32, copy=True)

    def predict(self, x: np.ndarray, target_case_ids: np.ndarray | None = None) -> np.ndarray:
        """Return predicted conditional semantic vectors."""
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
        if not self.is_dual_head:
            pred = context @ self.w1 @ self.w2 + self.bias
            if one_dimensional:
                return pred.reshape(-1).astype(np.float32, copy=False)
            return pred.astype(np.float32, copy=False)
        if target_case_ids is None:
            raise ValueError("dual-head v2 prediction requires target_case_ids")
        cases = np.asarray(target_case_ids, dtype=np.int8)
        if cases.ndim == 0:
            cases = cases.reshape(1)
        if cases.shape != (int(context.shape[0]),):
            raise ValueError("target_case_ids must have one id per context row")
        pred = np.empty((int(context.shape[0]), int(self.config.output_dim)), dtype=np.float32)
        event_mask = cases == EVENT_SEMANTIC_TARGET_ID
        action_mask = cases == BOTH_COLD_ACTION_TARGET_ID
        if np.any(~(event_mask | action_mask)):
            raise ValueError("target_case_ids contain unsupported target case")
        if np.any(event_mask):
            event_context = context[event_mask]
            pred[event_mask] = event_context @ self.event_w1 @ self.event_w2 + self.event_bias
        if np.any(action_mask):
            action_context = context[action_mask]
            pred[action_mask] = (
                action_context @ self.action_w1 @ self.action_w2 + self.action_bias
            )
        if one_dimensional:
            return pred.reshape(-1).astype(np.float32, copy=False)
        return pred.astype(np.float32, copy=False)

    def fingerprint(self) -> dict[str, Any]:
        """Return a stable fingerprint over config and weights."""
        payload = {
            "schema": (
                CONDITIONAL_HEAD_SCHEMA_DUAL_V2
                if self.is_dual_head
                else CONDITIONAL_HEAD_SCHEMA_SHARED_V1
            ),
            "config": asdict(self.config),
        }
        if self.is_dual_head:
            payload.update(
                {
                    "event_w1_sha256": hashlib.sha256(
                        self.event_w1.tobytes(order="C"),
                    ).hexdigest(),
                    "event_w2_sha256": hashlib.sha256(
                        self.event_w2.tobytes(order="C"),
                    ).hexdigest(),
                    "event_bias_sha256": hashlib.sha256(
                        self.event_bias.tobytes(order="C"),
                    ).hexdigest(),
                    "action_w1_sha256": hashlib.sha256(
                        self.action_w1.tobytes(order="C"),
                    ).hexdigest(),
                    "action_w2_sha256": hashlib.sha256(
                        self.action_w2.tobytes(order="C"),
                    ).hexdigest(),
                    "action_bias_sha256": hashlib.sha256(
                        self.action_bias.tobytes(order="C"),
                    ).hexdigest(),
                },
            )
        else:
            payload.update(
                {
                    "w1_sha256": hashlib.sha256(self.w1.tobytes(order="C")).hexdigest(),
                    "w2_sha256": hashlib.sha256(self.w2.tobytes(order="C")).hexdigest(),
                    "bias_sha256": hashlib.sha256(self.bias.tobytes(order="C")).hexdigest(),
                },
            )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["fingerprint_sha256"] = hashlib.sha256(encoded).hexdigest()
        return payload

    def save(self, path: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Persist conditional head weights and metadata."""
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
        fingerprint["head_arch"] = str(self.config.conditional_head_arch)
        fingerprint["schema"] = (
            CONDITIONAL_HEAD_SCHEMA_DUAL_V2
            if self.is_dual_head
            else CONDITIONAL_HEAD_SCHEMA_SHARED_V1
        )
        encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        fingerprint["fingerprint_sha256"] = hashlib.sha256(encoded).hexdigest()
        payload = {
            "schema": (
                CONDITIONAL_HEAD_SCHEMA_DUAL_V2
                if self.is_dual_head
                else CONDITIONAL_HEAD_SCHEMA_SHARED_V1
            ),
            "config": asdict(self.config),
            "metadata": metadata_payload,
            "fingerprint": fingerprint,
        }
        if self.is_dual_head:
            payload.update(
                {
                    "event_w1": self.event_w1,
                    "event_w2": self.event_w2,
                    "event_bias": self.event_bias,
                    "action_w1": self.action_w1,
                    "action_w2": self.action_w2,
                    "action_bias": self.action_bias,
                },
            )
        else:
            payload.update({"w1": self.w1, "w2": self.w2, "bias": self.bias})
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_head_arch: str | None = None,
    ) -> tuple["ConditionalSemanticHead", dict[str, Any]]:
        """Load a conditional semantic head checkpoint."""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"conditional head checkpoint not found: {checkpoint_path}")
        with checkpoint_path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, Mapping):
            raise TypeError("conditional head checkpoint payload must be a mapping")
        schema = str(payload.get("schema", ""))
        if schema not in {CONDITIONAL_HEAD_SCHEMA_SHARED_V1, CONDITIONAL_HEAD_SCHEMA_DUAL_V2}:
            raise ValueError("unsupported conditional head checkpoint schema")
        actual_arch = (
            CONDITIONAL_HEAD_ARCH_DUAL_V2
            if schema == CONDITIONAL_HEAD_SCHEMA_DUAL_V2
            else CONDITIONAL_HEAD_ARCH_SHARED_V1
        )
        if expected_head_arch is not None and str(expected_head_arch) != actual_arch:
            raise ValueError(
                f"conditional head arch mismatch: expected {expected_head_arch}, "
                f"got {actual_arch}",
            )
        config = ConditionalSemanticHeadConfig(**dict(payload["config"]))
        if str(config.conditional_head_arch) != actual_arch:
            raise ValueError("conditional head config arch does not match checkpoint schema")
        if actual_arch == CONDITIONAL_HEAD_ARCH_DUAL_V2:
            head = cls(
                config,
                event_w1=np.asarray(payload["event_w1"], dtype=np.float32),
                event_w2=np.asarray(payload["event_w2"], dtype=np.float32),
                event_bias=np.asarray(payload["event_bias"], dtype=np.float32),
                action_w1=np.asarray(payload["action_w1"], dtype=np.float32),
                action_w2=np.asarray(payload["action_w2"], dtype=np.float32),
                action_bias=np.asarray(payload["action_bias"], dtype=np.float32),
            )
        else:
            head = cls(
                config,
                w1=np.asarray(payload["w1"], dtype=np.float32),
                w2=np.asarray(payload["w2"], dtype=np.float32),
                bias=np.asarray(payload["bias"], dtype=np.float32),
            )
        metadata = dict(payload.get("metadata", {}))
        metadata["conditional_head_arch"] = actual_arch
        metadata["checkpoint_schema"] = schema
        metadata["fingerprint"] = dict(payload.get("fingerprint", head.fingerprint()))
        return head, metadata


def conditional_case_loss_summary(
    losses: np.ndarray,
    target_case_ids: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Summarize row-wise losses separately for each conditional target case."""
    values = np.asarray(losses, dtype=np.float32)
    cases = np.asarray(target_case_ids, dtype=np.int8)
    if values.shape != cases.shape:
        raise ValueError("losses and target_case_ids must have matching shape")
    summary: dict[str, dict[str, float | int]] = {}
    for name, case_id in (
        (EVENT_SEMANTIC_TARGET, EVENT_SEMANTIC_TARGET_ID),
        (BOTH_COLD_ACTION_TARGET, BOTH_COLD_ACTION_TARGET_ID),
    ):
        selected = values[cases == case_id]
        summary[name] = {
            "count": int(selected.size),
            "loss_mean": float(np.mean(selected)) if selected.size else 0.0,
            "loss_final": float(selected[-1]) if selected.size else 0.0,
        }
    return summary


def select_conditional_target(
    row: np.void,
    node_embeddings: np.ndarray,
    action_embeddings: np.ndarray,
    *,
    src_has_state: bool,
    dst_has_state: bool,
) -> tuple[np.ndarray, str]:
    """Return score target and target-case name for one event."""
    src_idx = int(row["src_node_idx"])
    dst_idx = int(row["dst_node_idx"])
    action_id = int(row["action_id"])
    src = np.asarray(node_embeddings[src_idx], dtype=np.float32)
    dst = np.asarray(node_embeddings[dst_idx], dtype=np.float32)
    action = np.asarray(action_embeddings[action_id], dtype=np.float32)
    if bool(src_has_state) or bool(dst_has_state):
        return ((src + action + dst) / np.float32(3.0)).astype(np.float32), EVENT_SEMANTIC_TARGET
    return action.astype(np.float32, copy=True), BOTH_COLD_ACTION_TARGET


def select_action_embedding_target(row: np.void, action_embeddings: np.ndarray) -> np.ndarray:
    """Return the pure action-embedding target for one event."""
    action_id = int(row["action_id"])
    return np.asarray(action_embeddings[action_id], dtype=np.float32).astype(
        np.float32,
        copy=True,
    )


def target_case_id(name: str) -> int:
    """Return compact target-case id."""
    if str(name) not in TARGET_CASE_NAME_TO_ID:
        raise ValueError(f"unsupported target case: {name}")
    return int(TARGET_CASE_NAME_TO_ID[str(name)])


def conditional_distance(prediction: np.ndarray, target: np.ndarray, loss_type: str) -> np.ndarray:
    """Return row-wise conditional semantic distance."""
    pred = np.asarray(prediction, dtype=np.float32)
    true = np.asarray(target, dtype=np.float32)
    if pred.shape != true.shape:
        raise ValueError("prediction and target shape mismatch")
    if str(loss_type) == "mse":
        return np.mean((pred - true) ** 2, axis=-1).astype(np.float32)
    if str(loss_type) == "cosine":
        pred_norm = np.linalg.norm(pred, axis=-1)
        true_norm = np.linalg.norm(true, axis=-1)
        denom = np.maximum(pred_norm * true_norm, np.float32(1e-8))
        cosine = np.sum(pred * true, axis=-1) / denom
        return (1.0 - np.clip(cosine, -1.0, 1.0)).astype(np.float32)
    raise ValueError(f"unsupported conditional loss: {loss_type}")


def quantile_threshold(scores: np.ndarray, quantile: float) -> float:
    """Return deterministic quantile threshold."""
    values = np.asarray(scores, dtype=np.float32)
    if values.size <= 0:
        raise ValueError("scores must not be empty")
    q = float(quantile)
    if not (0.0 < q <= 1.0):
        raise ValueError("quantile must be in (0, 1]")
    return float(np.quantile(values, q, method="higher"))


def score_summary(scores: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    """Return compact score summary."""
    values = np.asarray(scores, dtype=np.float32)
    threshold_value = None if threshold is None else float(threshold)
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
            "final_threshold": threshold_value,
            "above_threshold_count": 0,
        }
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


def encode_conditional_group_key(
    target_case_id: int,
    action_id: int,
    src_type_id: int,
    dst_type_id: int,
) -> np.int64:
    """Return a compact int64 key for target-case/action/src-type/dst-type groups."""
    case = int(target_case_id)
    action = int(action_id)
    src = int(src_type_id)
    dst = int(dst_type_id)
    if case < 0 or action < 0 or src < 0 or dst < 0:
        raise ValueError("conditional group ids must be non-negative")
    return np.int64((((case * 16) + action) * 8 + src) * 8 + dst)


def decode_conditional_group_key(encoded: int) -> dict[str, int]:
    """Decode an int64 conditional group key."""
    value = int(encoded)
    dst = value % 8
    value //= 8
    src = value % 8
    value //= 8
    action = value % 16
    case = value // 16
    return {
        "target_case_id": int(case),
        "action_id": int(action),
        "src_type_id": int(src),
        "dst_type_id": int(dst),
    }


def make_conditional_group_key(
    level: str,
    target_case_id: int,
    action_id: int | None = None,
    src_type_id: int | None = None,
    dst_type_id: int | None = None,
) -> str:
    """Return the JSON key for one conditional threshold fallback level."""
    case_name = TARGET_CASE_ID_TO_NAME.get(int(target_case_id), f"case_{int(target_case_id)}")
    if level == GROUP_LEVEL_TARGET_ACTION_SRC_DST:
        return (
            f"target_case={case_name}|action_id={int(action_id)}|"
            f"src_type_id={int(src_type_id)}|dst_type_id={int(dst_type_id)}"
        )
    if level == GROUP_LEVEL_TARGET_ACTION:
        return f"target_case={case_name}|action_id={int(action_id)}"
    if level == GROUP_LEVEL_TARGET_SRC_DST:
        return (
            f"target_case={case_name}|src_type_id={int(src_type_id)}|"
            f"dst_type_id={int(dst_type_id)}"
        )
    if level == GROUP_LEVEL_TARGET_CASE:
        return f"target_case={case_name}"
    if level == GROUP_LEVEL_GLOBAL:
        return "global_conditional"
    raise ValueError(f"unsupported conditional threshold level: {level}")


def _threshold_record(
    values: np.ndarray,
    *,
    threshold_quantile: float,
    level: str,
    key: str,
    min_group_count: int,
    eligible: bool,
) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    threshold = quantile_threshold(arr, float(threshold_quantile))
    return {
        "threshold": float(threshold),
        "count": int(arr.size),
        "threshold_level": str(level),
        "threshold_group_key": str(key),
        "min_group_count": int(min_group_count),
        "eligible": bool(eligible),
        "validation_max": float(np.max(arr)) if arr.size else 0.0,
        "summary": score_summary(arr, threshold),
    }


def _adaptive_margin_for_count(
    count: int,
    *,
    n1: int,
    n2: int,
    low: float,
    mid: float,
    high: float,
) -> tuple[float, str]:
    if int(count) < int(n1):
        return float(low), "low"
    if int(count) < int(n2):
        return float(mid), "mid"
    return float(high), "high"


def _add_threshold_record(
    store: dict[str, dict[str, Any]],
    key: str,
    values: list[float],
    *,
    threshold_quantile: float,
    level: str,
    min_group_count: int,
) -> None:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return
    store[str(key)] = _threshold_record(
        arr,
        threshold_quantile=float(threshold_quantile),
        level=str(level),
        key=str(key),
        min_group_count=int(min_group_count),
        eligible=arr.size >= int(min_group_count),
    )


def fit_conditional_group_thresholds(
    scores: np.ndarray,
    target_case_ids: np.ndarray,
    action_ids: np.ndarray,
    src_type_ids: np.ndarray,
    dst_type_ids: np.ndarray,
    *,
    threshold_quantile: float,
    min_group_count: int = CONDITIONAL_GROUP_MIN_COUNT_DEFAULT,
    low_support_policy: str = CONDITIONAL_LOW_SUPPORT_POLICY_DEFAULT,
    low_support_margin: float = CONDITIONAL_LOW_SUPPORT_MARGIN_DEFAULT,
    unseen_group_policy: str = CONDITIONAL_UNSEEN_GROUP_POLICY_DEFAULT,
    global_extreme_quantile: float = CONDITIONAL_GLOBAL_EXTREME_QUANTILE_DEFAULT,
    adaptive_margin_n1: int = CONDITIONAL_ADAPTIVE_MARGIN_N1_DEFAULT,
    adaptive_margin_n2: int = CONDITIONAL_ADAPTIVE_MARGIN_N2_DEFAULT,
    adaptive_margin_low: float = CONDITIONAL_ADAPTIVE_MARGIN_LOW_DEFAULT,
    adaptive_margin_mid: float = CONDITIONAL_ADAPTIVE_MARGIN_MID_DEFAULT,
    adaptive_margin_high: float = CONDITIONAL_ADAPTIVE_MARGIN_HIGH_DEFAULT,
) -> dict[str, Any]:
    """Fit hierarchical conditional thresholds from validation scores only."""
    if str(low_support_policy) not in {"conservative_max", "adaptive_margin"}:
        raise ValueError(
            "conditional low-support policy v1 supports conservative_max or adaptive_margin",
        )
    if str(unseen_group_policy) != "observation_only":
        raise ValueError("conditional unseen-group policy v1 supports only observation_only")
    if int(adaptive_margin_n1) <= 0 or int(adaptive_margin_n2) <= int(adaptive_margin_n1):
        raise ValueError("adaptive margin buckets require 0 < n1 < n2")
    values = np.asarray(scores, dtype=np.float32)
    cases = np.asarray(target_case_ids, dtype=np.int16)
    actions = np.asarray(action_ids, dtype=np.int16)
    srcs = np.asarray(src_type_ids, dtype=np.int16)
    dsts = np.asarray(dst_type_ids, dtype=np.int16)
    if not (values.shape == cases.shape == actions.shape == srcs.shape == dsts.shape):
        raise ValueError("conditional group threshold arrays must have matching shape")
    if values.size <= 0:
        raise ValueError("conditional group thresholds require validation scores")

    level_values: dict[str, dict[str, list[float]]] = {
        GROUP_LEVEL_TARGET_ACTION_SRC_DST: {},
        GROUP_LEVEL_TARGET_ACTION: {},
        GROUP_LEVEL_TARGET_SRC_DST: {},
        GROUP_LEVEL_TARGET_CASE: {},
    }
    for score, case_id, action_id, src_type_id, dst_type_id in zip(
        values,
        cases,
        actions,
        srcs,
        dsts,
    ):
        score_value = float(score)
        keys = {
            GROUP_LEVEL_TARGET_ACTION_SRC_DST: make_conditional_group_key(
                GROUP_LEVEL_TARGET_ACTION_SRC_DST,
                int(case_id),
                int(action_id),
                int(src_type_id),
                int(dst_type_id),
            ),
            GROUP_LEVEL_TARGET_ACTION: make_conditional_group_key(
                GROUP_LEVEL_TARGET_ACTION,
                int(case_id),
                int(action_id),
            ),
            GROUP_LEVEL_TARGET_SRC_DST: make_conditional_group_key(
                GROUP_LEVEL_TARGET_SRC_DST,
                int(case_id),
                src_type_id=int(src_type_id),
                dst_type_id=int(dst_type_id),
            ),
            GROUP_LEVEL_TARGET_CASE: make_conditional_group_key(
                GROUP_LEVEL_TARGET_CASE,
                int(case_id),
            ),
        }
        for level, key in keys.items():
            level_values[level].setdefault(key, []).append(score_value)

    thresholds: dict[str, dict[str, Any]] = {}
    for level, values_by_key in level_values.items():
        level_records: dict[str, Any] = {}
        for key, grouped_values in values_by_key.items():
            _add_threshold_record(
                level_records,
                key,
                grouped_values,
                threshold_quantile=float(threshold_quantile),
                level=level,
                min_group_count=int(min_group_count),
            )
        thresholds[level] = level_records
    global_key = make_conditional_group_key(GROUP_LEVEL_GLOBAL, int(cases[0]))
    global_record = _threshold_record(
        values,
        threshold_quantile=float(threshold_quantile),
        level=GROUP_LEVEL_GLOBAL,
        key=global_key,
        min_group_count=int(min_group_count),
        eligible=True,
    )
    global_extreme_key = "global_conditional_extreme"
    global_extreme_record = _threshold_record(
        values,
        threshold_quantile=float(global_extreme_quantile),
        level="unseen_group_extreme",
        key=global_extreme_key,
        min_group_count=int(min_group_count),
        eligible=True,
    )
    return {
        "schema": "phase3g_conditional_group_thresholds_v1",
        "threshold_mode": CONDITIONAL_GROUP_THRESHOLD_MODE,
        "threshold_quantile": float(threshold_quantile),
        "group_min_count": int(min_group_count),
        "low_support_policy": str(low_support_policy),
        "low_support_margin": float(low_support_margin),
        "unseen_group_policy": str(unseen_group_policy),
        "global_extreme_quantile": float(global_extreme_quantile),
        "adaptive_margin_n1": int(adaptive_margin_n1),
        "adaptive_margin_n2": int(adaptive_margin_n2),
        "adaptive_margin_low": float(adaptive_margin_low),
        "adaptive_margin_mid": float(adaptive_margin_mid),
        "adaptive_margin_high": float(adaptive_margin_high),
        "levels": [
            GROUP_LEVEL_TARGET_ACTION_SRC_DST,
            GROUP_LEVEL_TARGET_ACTION,
            GROUP_LEVEL_TARGET_SRC_DST,
            GROUP_LEVEL_TARGET_CASE,
            GROUP_LEVEL_GLOBAL,
        ],
        "thresholds": thresholds,
        "global": global_record,
        "global_extreme": global_extreme_record,
    }


def resolve_conditional_group_threshold(
    group_meta: Mapping[str, Any],
    target_case_id: int,
    action_id: int,
    src_type_id: int,
    dst_type_id: int,
) -> dict[str, Any]:
    """Resolve a threshold through the conditional target/action/type fallback hierarchy."""
    thresholds = dict(group_meta.get("thresholds", {}))
    min_count = int(group_meta.get("group_min_count", CONDITIONAL_GROUP_MIN_COUNT_DEFAULT))
    low_support_policy = str(
        group_meta.get("low_support_policy", CONDITIONAL_LOW_SUPPORT_POLICY_DEFAULT),
    )
    low_support_margin = float(
        group_meta.get("low_support_margin", CONDITIONAL_LOW_SUPPORT_MARGIN_DEFAULT),
    )
    global_record = dict(group_meta.get("global", {}))
    global_threshold = float(global_record.get("threshold", 0.0))
    global_extreme_record = dict(group_meta.get("global_extreme", global_record))
    adaptive_n1 = int(group_meta.get("adaptive_margin_n1", CONDITIONAL_ADAPTIVE_MARGIN_N1_DEFAULT))
    adaptive_n2 = int(group_meta.get("adaptive_margin_n2", CONDITIONAL_ADAPTIVE_MARGIN_N2_DEFAULT))
    adaptive_low = float(
        group_meta.get("adaptive_margin_low", CONDITIONAL_ADAPTIVE_MARGIN_LOW_DEFAULT),
    )
    adaptive_mid = float(
        group_meta.get("adaptive_margin_mid", CONDITIONAL_ADAPTIVE_MARGIN_MID_DEFAULT),
    )
    adaptive_high = float(
        group_meta.get("adaptive_margin_high", CONDITIONAL_ADAPTIVE_MARGIN_HIGH_DEFAULT),
    )
    candidates = [
        (
            GROUP_LEVEL_TARGET_ACTION_SRC_DST,
            make_conditional_group_key(
                GROUP_LEVEL_TARGET_ACTION_SRC_DST,
                int(target_case_id),
                int(action_id),
                int(src_type_id),
                int(dst_type_id),
            ),
        ),
        (
            GROUP_LEVEL_TARGET_ACTION,
            make_conditional_group_key(
                GROUP_LEVEL_TARGET_ACTION,
                int(target_case_id),
                int(action_id),
            ),
        ),
        (
            GROUP_LEVEL_TARGET_SRC_DST,
            make_conditional_group_key(
                GROUP_LEVEL_TARGET_SRC_DST,
                int(target_case_id),
                src_type_id=int(src_type_id),
                dst_type_id=int(dst_type_id),
            ),
        ),
        (
            GROUP_LEVEL_TARGET_CASE,
            make_conditional_group_key(GROUP_LEVEL_TARGET_CASE, int(target_case_id)),
        ),
    ]
    level1_key = candidates[0][1]
    level1_records = dict(thresholds.get(GROUP_LEVEL_TARGET_ACTION_SRC_DST, {}))
    level1_record = dict(level1_records.get(level1_key, {}))
    if int(level1_record.get("count", 0)) >= min_count and bool(
        level1_record.get("eligible", False),
    ):
        return {
            "threshold": float(level1_record["threshold"]),
            "threshold_level": "level1_group_quantile",
            "threshold_group_key": str(level1_key),
            "validation_group_count": int(level1_record.get("count", 0)),
            "low_support_policy": str(low_support_policy),
            "group_validation_max": float(level1_record.get("validation_max", 0.0)),
            "parent_threshold": float(level1_record["threshold"]),
            "global_threshold": float(global_threshold),
            "final_threshold_source": "level1_group_quantile",
            "adaptive_margin_used": 0.0,
            "validation_count_bucket": "sufficient",
        }
    if int(level1_record.get("count", 0)) <= 0:
        return {
            "threshold": float(global_extreme_record.get("threshold", global_threshold)),
            "threshold_level": "unseen_group_extreme",
            "threshold_group_key": str(level1_key),
            "validation_group_count": 0,
            "low_support_policy": str(low_support_policy),
            "group_validation_max": 0.0,
            "parent_threshold": float(global_threshold),
            "global_threshold": float(global_threshold),
            "final_threshold_source": "global_extreme",
            "adaptive_margin_used": 0.0,
            "validation_count_bucket": "unseen",
        }
    parent_record: dict[str, Any] | None = None
    parent_level_name = ""
    parent_source = ""
    parent_candidates = [
        (GROUP_LEVEL_TARGET_ACTION, candidates[1][1], "fallback_target_action"),
        (GROUP_LEVEL_TARGET_SRC_DST, candidates[2][1], "fallback_target_type"),
        (GROUP_LEVEL_TARGET_CASE, candidates[3][1], "fallback_target_case"),
    ]
    for level, key, source in parent_candidates:
        record = dict(dict(thresholds.get(level, {})).get(key, {}))
        if int(record.get("count", 0)) >= min_count and bool(record.get("eligible", False)):
            parent_record = record
            parent_level_name = str(level)
            parent_source = str(source)
            break
    parent_threshold = (
        float(parent_record["threshold"]) if parent_record is not None else float(global_threshold)
    )
    if int(level1_record.get("count", 0)) > 0:
        group_validation_max = float(level1_record.get("validation_max", 0.0))
        count = int(level1_record.get("count", 0))
        if low_support_policy == "adaptive_margin":
            margin, bucket = _adaptive_margin_for_count(
                count,
                n1=adaptive_n1,
                n2=adaptive_n2,
                low=adaptive_low,
                mid=adaptive_mid,
                high=adaptive_high,
            )
            threshold = max(group_validation_max + margin, parent_threshold, global_threshold)
            return {
                "threshold": float(threshold),
                "threshold_level": "low_support_adaptive_margin",
                "threshold_group_key": str(level1_key),
                "validation_group_count": count,
                "low_support_policy": str(low_support_policy),
                "group_validation_max": group_validation_max,
                "parent_threshold": float(parent_threshold),
                "parent_threshold_level": parent_level_name or GROUP_LEVEL_GLOBAL,
                "global_threshold": float(global_threshold),
                "final_threshold_source": "adaptive_margin",
                "adaptive_margin_used": float(margin),
                "validation_count_bucket": str(bucket),
            }
        if low_support_policy == "conservative_max":
            threshold = max(
                group_validation_max + float(low_support_margin),
                parent_threshold,
                global_threshold,
            )
            return {
                "threshold": float(threshold),
                "threshold_level": "low_support_conservative_max",
                "threshold_group_key": str(level1_key),
                "validation_group_count": int(level1_record.get("count", 0)),
                "low_support_policy": str(low_support_policy),
                "group_validation_max": group_validation_max,
                "parent_threshold": float(parent_threshold),
                "parent_threshold_level": parent_level_name or GROUP_LEVEL_GLOBAL,
                "global_threshold": float(global_threshold),
                "final_threshold_source": "low_support_conservative_max",
                "adaptive_margin_used": 0.0,
                "validation_count_bucket": "fixed_margin",
            }
    return {
        "threshold": float(global_record.get("threshold", 0.0)),
        "threshold_level": "global_conditional",
        "threshold_group_key": str(global_record.get("threshold_group_key", "global_conditional")),
        "validation_group_count": int(global_record.get("count", 0)),
        "low_support_policy": str(low_support_policy),
        "group_validation_max": 0.0,
        "parent_threshold": float(global_threshold),
        "global_threshold": float(global_threshold),
        "final_threshold_source": "global_conditional",
        "adaptive_margin_used": 0.0,
        "validation_count_bucket": "global",
    }


def _case_summary(
    scores: np.ndarray,
    target_case_ids: np.ndarray,
    *,
    case_id: int,
    global_threshold: float,
    threshold_mode: str,
    threshold_quantile: float,
    min_case_count: int,
) -> dict[str, Any]:
    mask = np.asarray(target_case_ids, dtype=np.int8) == int(case_id)
    values = np.asarray(scores, dtype=np.float32)[mask]
    fallback = bool(values.size < int(min_case_count))
    if str(threshold_mode) == "target_case_quantile" and not fallback:
        threshold = quantile_threshold(values, float(threshold_quantile))
    else:
        threshold = float(global_threshold)
    payload = score_summary(values, threshold)
    payload.update(
        {
            "target_case": TARGET_CASE_ID_TO_NAME[int(case_id)],
            "target_case_id": int(case_id),
            "threshold": float(threshold),
            "global_fallback": fallback or str(threshold_mode) != "target_case_quantile",
        },
    )
    return payload


def write_conditional_validation_cache(
    cache_dir: str | Path,
    *,
    scores: np.ndarray,
    target_case_ids: np.ndarray,
    fingerprint: Mapping[str, Any],
    threshold_mode: str,
    threshold_quantile: float,
    min_case_count: int = 2,
    action_ids: np.ndarray | None = None,
    src_type_ids: np.ndarray | None = None,
    dst_type_ids: np.ndarray | None = None,
    group_min_count: int = CONDITIONAL_GROUP_MIN_COUNT_DEFAULT,
    low_support_policy: str = CONDITIONAL_LOW_SUPPORT_POLICY_DEFAULT,
    low_support_margin: float = CONDITIONAL_LOW_SUPPORT_MARGIN_DEFAULT,
    unseen_group_policy: str = CONDITIONAL_UNSEEN_GROUP_POLICY_DEFAULT,
    global_extreme_quantile: float = CONDITIONAL_GLOBAL_EXTREME_QUANTILE_DEFAULT,
    adaptive_margin_n1: int = CONDITIONAL_ADAPTIVE_MARGIN_N1_DEFAULT,
    adaptive_margin_n2: int = CONDITIONAL_ADAPTIVE_MARGIN_N2_DEFAULT,
    adaptive_margin_low: float = CONDITIONAL_ADAPTIVE_MARGIN_LOW_DEFAULT,
    adaptive_margin_mid: float = CONDITIONAL_ADAPTIVE_MARGIN_MID_DEFAULT,
    adaptive_margin_high: float = CONDITIONAL_ADAPTIVE_MARGIN_HIGH_DEFAULT,
) -> dict[str, Any]:
    """Write conditional validation scores, target cases, and thresholds."""
    output_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(scores, dtype=np.float32)
    cases = np.asarray(target_case_ids, dtype=np.int8)
    if values.shape != cases.shape:
        raise ValueError("scores and target_case_ids shape mismatch")
    global_threshold = (
        float(np.max(values))
        if str(threshold_mode) == "validation_max" and values.size
        else quantile_threshold(values, float(threshold_quantile))
    )
    if str(threshold_mode) not in {
        "quantile",
        "target_case_quantile",
        "validation_max",
        CONDITIONAL_GROUP_THRESHOLD_MODE,
    }:
        raise ValueError(f"unsupported conditional threshold mode: {threshold_mode}")
    score_path = output_dir / "validation_conditional_scores.memmap"
    case_path = output_dir / "validation_target_case.memmap"
    mapped_scores = np.memmap(score_path, dtype=np.float32, mode="w+", shape=values.shape)
    mapped_scores[:] = values[:]
    mapped_scores.flush()
    del mapped_scores
    mapped_cases = np.memmap(case_path, dtype=np.int8, mode="w+", shape=cases.shape)
    mapped_cases[:] = cases[:]
    mapped_cases.flush()
    del mapped_cases
    group_keys_path = output_dir / "validation_conditional_group_keys.memmap"
    group_meta: dict[str, Any] | None = None
    if str(threshold_mode) == CONDITIONAL_GROUP_THRESHOLD_MODE:
        if action_ids is None or src_type_ids is None or dst_type_ids is None:
            raise ValueError("conditional group threshold mode requires action/type arrays")
        actions = np.asarray(action_ids, dtype=np.int16)
        srcs = np.asarray(src_type_ids, dtype=np.int16)
        dsts = np.asarray(dst_type_ids, dtype=np.int16)
        if not (actions.shape == values.shape == srcs.shape == dsts.shape):
            raise ValueError("conditional group action/type arrays must match scores shape")
        encoded_keys = np.zeros(values.shape, dtype=np.int64)
        for index, (case_id, action_id, src_type_id, dst_type_id) in enumerate(
            zip(cases, actions, srcs, dsts),
        ):
            encoded_keys[index] = encode_conditional_group_key(
                int(case_id),
                int(action_id),
                int(src_type_id),
                int(dst_type_id),
            )
        mapped_group_keys = np.memmap(
            group_keys_path,
            dtype=np.int64,
            mode="w+",
            shape=encoded_keys.shape,
        )
        mapped_group_keys[:] = encoded_keys[:]
        mapped_group_keys.flush()
        del mapped_group_keys
        group_meta = fit_conditional_group_thresholds(
            values,
            cases,
            actions,
            srcs,
            dsts,
            threshold_quantile=float(threshold_quantile),
            min_group_count=int(group_min_count),
            low_support_policy=str(low_support_policy),
            low_support_margin=float(low_support_margin),
            unseen_group_policy=str(unseen_group_policy),
            global_extreme_quantile=float(global_extreme_quantile),
            adaptive_margin_n1=int(adaptive_margin_n1),
            adaptive_margin_n2=int(adaptive_margin_n2),
            adaptive_margin_low=float(adaptive_margin_low),
            adaptive_margin_mid=float(adaptive_margin_mid),
            adaptive_margin_high=float(adaptive_margin_high),
        )
    summary = score_summary(values, global_threshold)
    summary.update(
        {
            "threshold_mode": str(threshold_mode),
            "threshold_quantile": float(threshold_quantile),
        },
    )
    by_case = {
        name: _case_summary(
            values,
            cases,
            case_id=case_id,
            global_threshold=float(global_threshold),
            threshold_mode=str(threshold_mode),
            threshold_quantile=float(threshold_quantile),
            min_case_count=int(min_case_count),
        )
        for case_id, name in TARGET_CASE_ID_TO_NAME.items()
    }
    thresholds_by_case = {
        name: {
            "threshold": float(case_payload["threshold"]),
            "count": int(case_payload["count"]),
            "global_fallback": bool(case_payload["global_fallback"]),
        }
        for name, case_payload in by_case.items()
    }
    meta = {
        "schema": "phase3g_conditional_validation_cache_v1",
        "fingerprint": dict(fingerprint),
        "num_scores": int(values.size),
        "score_dtype": "float32",
        "target_case_dtype": "int8",
        "validation_conditional_scores": str(score_path),
        "validation_target_case": str(case_path),
        "threshold": float(global_threshold),
        "thresholds_by_target_case": thresholds_by_case,
        "group_thresholds": group_meta,
        "summary": summary,
        "summary_by_target_case": by_case,
        "min_case_count": int(min_case_count),
        "group_min_count": int(group_min_count),
        "low_support_policy": str(low_support_policy),
        "low_support_margin": float(low_support_margin),
        "unseen_group_policy": str(unseen_group_policy),
        "global_extreme_quantile": float(global_extreme_quantile),
        "adaptive_margin_n1": int(adaptive_margin_n1),
        "adaptive_margin_n2": int(adaptive_margin_n2),
        "adaptive_margin_low": float(adaptive_margin_low),
        "adaptive_margin_mid": float(adaptive_margin_mid),
        "adaptive_margin_high": float(adaptive_margin_high),
        "created_unix_time": float(time.time()),
    }
    if group_meta is not None:
        meta["validation_conditional_group_keys"] = str(group_keys_path)
        (output_dir / "validation_conditional_group_thresholds.json").write_text(
            json.dumps(group_meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_conditional_group_summary_csv(
            output_dir / "validation_conditional_group_summary.csv",
            group_meta,
        )
    (output_dir / "validation_score_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validation_score_summary_by_target_case.json").write_text(
        json.dumps(by_case, indent=2, sort_keys=True) + "\n",
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


def load_conditional_validation_cache(
    cache_dir: str | Path,
    *,
    expected_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Load conditional validation cache metadata and fail fast on mismatch."""
    cache_path = Path(cache_dir) / "validation_threshold_cache.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"conditional validation cache not found: {cache_path}")
    meta = json.loads(cache_path.read_text(encoding="utf-8"))
    if dict(meta.get("fingerprint", {})) != dict(expected_fingerprint):
        raise ValueError("conditional validation cache fingerprint mismatch")
    score_path = Path(str(meta["validation_conditional_scores"]))
    case_path = Path(str(meta["validation_target_case"]))
    if not score_path.exists():
        raise FileNotFoundError(f"validation conditional scores missing: {score_path}")
    if not case_path.exists():
        raise FileNotFoundError(f"validation target cases missing: {case_path}")
    if "validation_conditional_group_keys" in meta:
        group_path = Path(str(meta["validation_conditional_group_keys"]))
        if not group_path.exists():
            raise FileNotFoundError(f"validation conditional group keys missing: {group_path}")
    return meta


def write_action_embedding_validation_cache(
    cache_dir: str | Path,
    *,
    scores: np.ndarray,
    fingerprint: Mapping[str, Any],
    threshold_mode: str,
    threshold_quantile: float,
) -> dict[str, Any]:
    """Write conditional-action-embedding validation scores and threshold metadata."""
    output_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    values = np.asarray(scores, dtype=np.float32)
    if str(threshold_mode) == "validation_max":
        threshold = float(np.max(values)) if values.size else 0.0
    elif str(threshold_mode) == "quantile":
        threshold = quantile_threshold(values, float(threshold_quantile))
    else:
        raise ValueError(
            f"unsupported conditional action embedding threshold mode: {threshold_mode}",
        )
    score_path = output_dir / "validation_action_embedding_scores.memmap"
    mapped_scores = np.memmap(score_path, dtype=np.float32, mode="w+", shape=values.shape)
    mapped_scores[:] = values[:]
    mapped_scores.flush()
    del mapped_scores
    summary = score_summary(values, threshold)
    summary.update(
        {
            "threshold_mode": str(threshold_mode),
            "threshold_quantile": float(threshold_quantile),
        },
    )
    meta = {
        "schema": "phase3g_conditional_action_embedding_validation_cache_v1",
        "score_head": CONDITIONAL_ACTION_EMBEDDING_SCORE_HEAD,
        "fingerprint": dict(fingerprint),
        "num_scores": int(values.size),
        "score_dtype": "float32",
        "validation_action_embedding_scores": str(score_path),
        "threshold": float(threshold),
        "summary": summary,
        "created_unix_time": float(time.time()),
    }
    threshold_path = output_dir / "validation_action_embedding_thresholds.json"
    threshold_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_csv = output_dir / "validation_action_embedding_score_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    calibration_path = output_dir / "validation_action_embedding_calibration_meta.json"
    calibration_path.write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validation_threshold_cache.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def load_action_embedding_validation_cache(
    cache_dir: str | Path,
    *,
    expected_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Load conditional-action-embedding validation cache metadata."""
    cache_path = Path(cache_dir) / "validation_action_embedding_thresholds.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"conditional action embedding validation cache not found: {cache_path}",
        )
    meta = json.loads(cache_path.read_text(encoding="utf-8"))
    if dict(meta.get("fingerprint", {})) != dict(expected_fingerprint):
        raise ValueError("conditional action embedding validation cache fingerprint mismatch")
    score_path = Path(str(meta["validation_action_embedding_scores"]))
    if not score_path.exists():
        raise FileNotFoundError(f"validation action embedding scores missing: {score_path}")
    return meta


def write_conditional_group_summary_csv(
    output_path: str | Path,
    group_meta: Mapping[str, Any],
) -> Path:
    """Write validation-only conditional group threshold summary rows."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    level1 = dict(dict(group_meta.get("thresholds", {})).get(GROUP_LEVEL_TARGET_ACTION_SRC_DST, {}))
    for key, record_obj in sorted(level1.items()):
        record = dict(record_obj)
        resolved = resolve_conditional_group_threshold(
            group_meta,
            int(TARGET_CASE_NAME_TO_ID[str(key).split("|", 1)[0].split("=", 1)[-1]]),
            int(str(key).split("action_id=", 1)[1].split("|", 1)[0]),
            int(str(key).split("src_type_id=", 1)[1].split("|", 1)[0]),
            int(str(key).rsplit("dst_type_id=", 1)[-1]),
        )
        encoded_parts: dict[str, int] = {}
        for part in str(key).split("|"):
            if "=" not in part:
                continue
            left, right = part.split("=", 1)
            if left in {"action_id", "src_type_id", "dst_type_id"}:
                encoded_parts[left] = int(right)
        summary = dict(record.get("summary", {}))
        rows.append(
            {
                "target_case": str(key).split("|", 1)[0].split("=", 1)[-1],
                "action_id": encoded_parts.get("action_id", -1),
                "action_name": "",
                "src_type_id": encoded_parts.get("src_type_id", -1),
                "src_type_name": "",
                "dst_type_id": encoded_parts.get("dst_type_id", -1),
                "dst_type_name": "",
                "validation_count": int(record.get("count", 0)),
                "test_count": 0,
                "threshold": float(resolved.get("threshold", record.get("threshold", 0.0))),
                "threshold_level": str(resolved.get("threshold_level", "")),
                "low_support_policy": str(resolved.get("low_support_policy", "")),
                "group_validation_max": float(
                    resolved.get("group_validation_max", record.get("validation_max", 0.0)),
                ),
                "parent_threshold": float(resolved.get("parent_threshold", 0.0)),
                "global_threshold": float(resolved.get("global_threshold", 0.0)),
                "final_threshold_source": str(resolved.get("final_threshold_source", "")),
                "adaptive_margin_used": float(resolved.get("adaptive_margin_used", 0.0)),
                "validation_count_bucket": str(resolved.get("validation_count_bucket", "")),
                "val_p99": float(summary.get("score_p99", 0.0)),
                "val_p999": float(summary.get("score_p999", 0.0)),
                "val_p9995": float(summary.get("score_p9995", 0.0)),
                "val_p9999": float(summary.get("score_p9999", 0.0)),
                "val_max": float(summary.get("score_max", 0.0)),
                "test_p99": 0.0,
                "test_p999": 0.0,
                "test_p9995": 0.0,
                "test_p9999": 0.0,
                "test_max": 0.0,
                "alert_count": 0,
                "tp": 0,
                "fp": 0,
                "precision": 0.0,
                "recall": 0.0,
            },
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONDITIONAL_GROUP_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path
