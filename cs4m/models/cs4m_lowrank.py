from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cs4m.scoring.calibration import ResidualCalibrationModel, UpdateGateCalibrator
from cs4m.scoring.simple_gates import fixed_update_gate, lambda_rho_for_event, raw_action_one_hot
from cs4m.state.online_state_merging import OnlineStateMergeConfig
from cs4m.state.state_memory import ProbationaryLRUStateMemory
from cs4m.state.state_models import build_state_model
from cs4m.utils.common import ENTITY_TYPES, one_hot, stable_hash


DEFAULT_TAU = (2, 4, 8, 16, 32, 64, 128, 256)
GRAD_CLIP_NORM = np.float32(5.0)
CONTEXT_RMS_EPS = np.float32(1e-6)


def normalize_vector(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return an L2-normalized float32 vector, preserving zero vectors."""
    vector = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= float(eps):
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / np.float32(norm)).astype(np.float32, copy=False)


@dataclass
class SSPMLowRankConfig:
    """Configuration for the SSPM low-rank residual predictor."""

    latent_dim: int = 64
    target_dim: int = 64
    state_dim: int = 64
    rank: int = 32
    batch_size: int = 1024
    learning_rate: float = 0.005
    l2: float = 1e-4
    lambda_mse: float = 0.25
    epochs: int = 1
    early_stop_min_delta: float = 1e-4
    early_stop_patience: int = 5
    seed: int = 17
    state_model: str = "ema_fixed"
    context_mode: str = "with_action"
    context_action_mode: str = "raw_orthrus10"
    action_count: int = 10
    global_context_mode: str = "no_global"
    state_memory_mode: str = "bounded"
    state_memory_policy: str = "probationary_lru"
    active_max_nodes: int = 500000
    probationary_max_nodes: int = 500000
    probationary_min_count: int = 2
    lru_evict_batch: int = 10000
    alert_protect_events: int = 100000
    update_gate_mode: str = "none"
    update_gate_quantile: float = 0.99
    update_gate_q_min: float = 0.05
    update_gate_eta: float = 1.0
    residual_score_mode: str = "legacy"
    residual_calibration: str = "none"
    residual_alpha_cos: float = 1.0
    residual_beta_mse: float = 0.25
    residual_beta_var: float = 0.25
    residual_var_eps: float = 1e-6
    residual_calibration_min_count: int = 2
    real_diag_train_gamma: bool = False
    real_diag_gamma_lr: float = 0.001
    real_diag_gamma_weight_decay: float = 0.0
    real_diag_gamma_grad_clip: float = 1.0
    real_diag_sensitivity_mode: str = "online_stop_message"
    real_diag_gamma_min: float = 1e-4
    real_diag_gamma_max: float = 1.0
    real_diag_max_sensitivity_nodes: int = 500000
    state_merge_mode: str = "none"
    state_merge_scope: str = "same_type"
    state_merge_threshold: float = 0.98
    state_merge_trunc_ratio: float = 0.50
    state_merge_use_rfft: bool = True
    state_merge_use_real_only: bool = True
    state_merge_center_state: bool = False
    state_merge_normalize: str = "l2"
    state_merge_eps: float = 1e-8
    state_merge_min_cluster_size: int = 2
    state_merge_copy_on_write: bool = True
    state_merge_random_prob: float = 0.02
    state_merge_random_seed: int = 0
    state_merge_diagnostics_max_rows: int = 1_000_000
    state_merge_match_backend: str = "exact"
    state_merge_candidate_cap: int = 64
    state_merge_interval: int = 1
    state_merge_min_count: int = 1
    state_merge_diagnostics_enabled: bool = True
    max_exact_states: int = 0
    min_inactive_events: int = 1_000_000
    prototype_count: int = 512
    prototype_update_alpha: float = 0.05


class SSPMLowRankModel:
    """Streaming SSPM low-rank residual model with compact array node state."""

    def __init__(self, config: SSPMLowRankConfig | None = None) -> None:
        self.config = config if config is not None else SSPMLowRankConfig()
        if (
            int(self.config.latent_dim) != int(self.config.target_dim)
            and int(self.config.target_dim) == SSPMLowRankConfig.target_dim
        ):
            self.config.target_dim = int(self.config.latent_dim)
        if (
            int(self.config.latent_dim) != int(self.config.state_dim)
            and int(self.config.state_dim) == SSPMLowRankConfig.state_dim
        ):
            self.config.state_dim = int(self.config.latent_dim)
        self._validate_config()
        self.latent_dim = int(self.config.latent_dim)
        self.target_dim = int(self.config.target_dim)
        self.state_dim = int(self.config.state_dim)
        self.rank = int(self.config.rank)
        if str(self.config.context_mode) == "with_action":
            if str(self.config.context_action_mode) == "raw_orthrus10":
                action_dim = 10
            else:
                action_dim = int(self.config.action_count)
        else:
            action_dim = 0
        global_dim = self.target_dim if str(self.config.global_context_mode) == "with_global" else 0
        self.context_dim = global_dim + 2 * self.target_dim + action_dim + 2 * len(ENTITY_TYPES)
        self.state_model = build_state_model(
            str(self.config.state_model),
            state_dim=self.state_dim,
            target_dim=self.target_dim,
            seed=int(self.config.seed),
            gamma_min=float(self.config.real_diag_gamma_min),
            gamma_max=float(self.config.real_diag_gamma_max),
        )
        self._real_diag_initial_gamma = (
            self.state_model.gamma.astype(np.float32, copy=True)
            if str(self.config.state_model) == "real_diag_learnable"
            and hasattr(self.state_model, "gamma")
            else None
        )
        tau = np.resize(np.asarray(DEFAULT_TAU, dtype=np.float32), self.latent_dim)
        self.a = np.exp(np.float32(-1.0) / tau).astype(np.float32)
        self.g = (np.float32(1.0) - self.a).astype(np.float32)
        rng = np.random.default_rng(int(self.config.seed))
        scale1 = 1.0 / np.sqrt(max(self.context_dim, 1))
        scale2 = 1.0 / np.sqrt(max(self.rank, 1))
        self.c1 = rng.normal(0.0, scale1, size=(self.context_dim, self.rank)).astype(np.float32)
        self.c2 = rng.normal(0.0, scale2, size=(self.rank, self.latent_dim)).astype(np.float32)
        self.bias = np.zeros((self.latent_dim,), dtype=np.float32)
        self.memory = ProbationaryLRUStateMemory(
            state_dim=self.state_dim,
            active_max_nodes=int(self.config.active_max_nodes),
            probationary_max_nodes=int(self.config.probationary_max_nodes),
            probationary_min_count=int(self.config.probationary_min_count),
            lru_evict_batch=int(self.config.lru_evict_batch),
            alert_protect_events=int(self.config.alert_protect_events),
            state_merge_config=self._make_state_merge_config(),
        )
        self.node_to_slot: dict[int, int] = {}
        self._state_size = 0
        self._state_array = np.zeros((0, self.latent_dim), dtype=np.float32)
        self._slot_node_ids = np.zeros((0,), dtype=np.int64)
        self._slot_role_ids = np.zeros((0,), dtype=np.int32)
        self._slot_last_touch = np.zeros((0,), dtype=np.int64)
        self._slot_touch_count = np.zeros((0,), dtype=np.int32)
        self._slot_risk = np.zeros((0,), dtype=np.uint8)
        self._clock = 0
        self._prototype_states = np.zeros(
            (max(int(self.config.prototype_count), 1), self.latent_dim),
            dtype=np.float32,
        )
        self._prototype_counts = np.zeros(
            (max(int(self.config.prototype_count), 1),),
            dtype=np.int32,
        )
        self._eviction_count = 0
        self._prototype_hit_count = 0
        self._overflow_count = 0
        self.global_state = np.zeros((self.latent_dim,), dtype=np.float32)
        self.last_lambda_t = 0.0
        self.last_rho_t = 0.0
        self.last_q_t = 1.0
        self.residual_calibration_model: ResidualCalibrationModel | None = None
        self.update_gate_calibrator: UpdateGateCalibrator | None = None
        self.real_diag_gamma_training_stats: dict[str, Any] = {}
        self.last_context_snapshot: dict[str, float] = {
            "src_state_norm": 0.0,
            "dst_state_norm": 0.0,
        }

    @property
    def state_array(self) -> np.ndarray:
        """Return the active compact node-state slice."""
        if self._uses_probationary_memory():
            return self.memory.state_array
        return self._state_array[: self._state_size]

    @property
    def state_slots(self) -> int:
        """Return the number of allocated compact node-state slots."""
        if self._uses_probationary_memory():
            return int(self.memory.active_count)
        return int(self._state_size)

    @property
    def state_array_mb(self) -> float:
        """Return allocated compact online state array size in MiB."""
        if self._uses_probationary_memory():
            return float(self.memory.state_array_mb)
        return float(self._state_array.nbytes / (1024.0 * 1024.0))

    def reset_state(self) -> None:
        """Clear online state without changing learned low-rank parameters."""
        self.last_context_snapshot = {
            "src_state_norm": 0.0,
            "dst_state_norm": 0.0,
        }
        if self._uses_probationary_memory():
            self.memory.reset()
            self.global_state = np.zeros((self.latent_dim,), dtype=np.float32)
            self.last_lambda_t = 0.0
            self.last_rho_t = 0.0
            self.last_q_t = 1.0
            return
        self.node_to_slot = {}
        self._state_size = 0
        self._state_array = np.zeros((0, self.latent_dim), dtype=np.float32)
        self._slot_node_ids = np.zeros((0,), dtype=np.int64)
        self._slot_role_ids = np.zeros((0,), dtype=np.int32)
        self._slot_last_touch = np.zeros((0,), dtype=np.int64)
        self._slot_touch_count = np.zeros((0,), dtype=np.int32)
        self._slot_risk = np.zeros((0,), dtype=np.uint8)
        self._clock = 0
        self._prototype_states = np.zeros(
            (max(int(self.config.prototype_count), 1), self.latent_dim),
            dtype=np.float32,
        )
        self._prototype_counts = np.zeros(
            (max(int(self.config.prototype_count), 1),),
            dtype=np.int32,
        )
        self._eviction_count = 0
        self._prototype_hit_count = 0
        self._overflow_count = 0
        self.global_state = np.zeros((self.latent_dim,), dtype=np.float32)

    def reset_state_memory(self) -> None:
        """Reinitialize online state memory without changing learned parameters."""
        if self._uses_probationary_memory():
            self.memory = ProbationaryLRUStateMemory(
                state_dim=self.state_dim,
                active_max_nodes=int(self.config.active_max_nodes),
                probationary_max_nodes=int(self.config.probationary_max_nodes),
                probationary_min_count=int(self.config.probationary_min_count),
                lru_evict_batch=int(self.config.lru_evict_batch),
                alert_protect_events=int(self.config.alert_protect_events),
                state_merge_config=self._make_state_merge_config(),
            )
            self.global_state = np.zeros((self.latent_dim,), dtype=np.float32)
            self.last_lambda_t = 0.0
            self.last_rho_t = 0.0
            self.last_q_t = 1.0
            self.last_context_snapshot = {
                "src_state_norm": 0.0,
                "dst_state_norm": 0.0,
            }
            return
        self.reset_state()

    def get_state(self, node_id: int) -> np.ndarray:
        """Return a copy of a node state, or a zero vector for unseen nodes."""
        if self._uses_probationary_memory():
            return self.memory.read_state(int(node_id), "unknown")
        slot = self.node_to_slot.get(int(node_id))
        if slot is None:
            return np.zeros((self.latent_dim,), dtype=np.float32)
        return self._state_array[int(slot)].astype(np.float32, copy=True)

    def get_context_state(self, node_id: int, role: str | int | None = None) -> np.ndarray:
        """Return exact or prototype node state for pre-update context construction."""
        if self._uses_probationary_memory():
            return self.memory.read_state(int(node_id), str(role or "unknown"))
        slot = self.node_to_slot.get(int(node_id))
        if slot is not None:
            return self._state_array[int(slot)].astype(np.float32, copy=True)
        if str(self.config.state_memory_mode) != "bounded":
            return np.zeros((self.latent_dim,), dtype=np.float32)
        role_id = self._role_id(role)
        if int(self._prototype_counts[role_id]) <= 0:
            return np.zeros((self.latent_dim,), dtype=np.float32)
        self._prototype_hit_count += 1
        return self._prototype_states[role_id].astype(np.float32, copy=True)

    def make_context(self, row: dict[str, Any]) -> np.ndarray:
        """Build the causal pre-update context from global, src/dst, action, and types."""
        src_state = self.state_model.read(
            self.get_context_state(
                int(row["info_src"]),
                row.get("src_type_name", row.get("src_role")),
            ),
        )
        dst_state = self.state_model.read(
            self.get_context_state(
                int(row["info_dst"]),
                row.get("dst_type_name", row.get("dst_role")),
            ),
        )
        self.last_context_snapshot = {
            "src_state_norm": float(np.linalg.norm(src_state)),
            "dst_state_norm": float(np.linalg.norm(dst_state)),
        }
        parts = []
        if str(self.config.global_context_mode) == "with_global":
            parts.append(self.global_state.astype(np.float32, copy=False))
        parts.extend([src_state, dst_state])
        if str(self.config.context_mode) == "with_action":
            if str(self.config.context_action_mode) == "raw_orthrus10":
                action = row.get("raw_action", row.get("action", row.get("action_token", "")))
                parts.append(raw_action_one_hot(action))
            else:
                action_id = int(row.get("action_id", row.get("relation_id", 0)))
                parts.append(one_hot(action_id, int(self.config.action_count)))
        parts.extend(
            [
                one_hot(int(row["info_src_type"]), len(ENTITY_TYPES)),
                one_hot(int(row["info_dst_type"]), len(ENTITY_TYPES)),
            ],
        )
        context = np.concatenate(parts, axis=0)
        return context.astype(np.float32, copy=False)

    def predict_from_context(self, context: np.ndarray) -> np.ndarray:
        """Predict z_hat from one or more already-built context rows."""
        ctx = np.asarray(context, dtype=np.float32)
        if ctx.ndim == 1:
            if ctx.shape != (self.context_dim,):
                raise ValueError(f"context must have shape ({self.context_dim},)")
        elif ctx.ndim == 2:
            if ctx.shape[1] != self.context_dim:
                raise ValueError(f"context must have shape (n, {self.context_dim})")
        else:
            raise ValueError("context must be a 1-D or 2-D array")
        if not bool(np.all(np.isfinite(ctx))):
            raise ValueError("context must contain only finite values")
        pred = ctx @ self.c1 @ self.c2 + self.bias
        return pred.astype(np.float32, copy=False)

    def set_lowrank_weights(
        self,
        c1: np.ndarray,
        c2: np.ndarray,
        bias: np.ndarray,
    ) -> None:
        """Replace W1/W2/b with trained NumPy weights."""
        c1_arr = np.asarray(c1, dtype=np.float32)
        c2_arr = np.asarray(c2, dtype=np.float32)
        bias_arr = np.asarray(bias, dtype=np.float32)
        if c1_arr.shape != (self.context_dim, self.rank):
            raise ValueError("c1 shape mismatch")
        if c2_arr.shape != (self.rank, self.latent_dim):
            raise ValueError("c2 shape mismatch")
        if bias_arr.shape != (self.latent_dim,):
            raise ValueError("bias shape mismatch")
        if not bool(np.all(np.isfinite(c1_arr))):
            raise ValueError("c1 must contain only finite values")
        if not bool(np.all(np.isfinite(c2_arr))):
            raise ValueError("c2 must contain only finite values")
        if not bool(np.all(np.isfinite(bias_arr))):
            raise ValueError("bias must contain only finite values")
        self.c1 = c1_arr.astype(np.float32, copy=True)
        self.c2 = c2_arr.astype(np.float32, copy=True)
        self.bias = bias_arr.astype(np.float32, copy=True)

    def predict(self, row: dict[str, Any]) -> np.ndarray:
        """Predict the current event latent vector from pre-update online state."""
        return self.predict_from_context(self.make_context(row))

    def set_residual_calibration(self, calibration: ResidualCalibrationModel | None) -> None:
        """Attach finalized validation residual calibration statistics."""
        self.residual_calibration_model = calibration

    def set_update_gate_calibrator(self, calibrator: UpdateGateCalibrator | None) -> None:
        """Attach a validation-fitted update-gate calibrator."""
        self.update_gate_calibrator = calibrator

    def residual_score(
        self,
        pred: np.ndarray,
        z: np.ndarray,
        action: object | None = None,
    ) -> float:
        """Return the configured residual anomaly score for one event."""
        pred_vec = self._checked_vector(pred, "pred")
        z_vec = self._checked_vector(z, "z")
        pred_norm = normalize_vector(pred_vec)
        z_norm = normalize_vector(z_vec)
        cosine = float(1.0 - np.dot(pred_norm, z_norm))
        mode = str(self.config.residual_score_mode)
        if mode == "legacy":
            mse = float(np.mean((pred_norm - z_norm) ** 2))
            score = cosine + float(self.config.lambda_mse) * mse
        elif mode == "raw_mse":
            mse = float(np.mean((pred_vec - z_vec) ** 2))
            score = float(self.config.residual_alpha_cos) * cosine
            score += float(self.config.residual_beta_mse) * mse
        elif mode == "var_calibrated":
            if self.residual_calibration_model is None:
                raise ValueError("residual calibration is not fitted")
            if action is None:
                raise ValueError("action is required for var_calibrated residual scoring")
            score = self.residual_calibration_model.score_var_calibrated(
                action,
                pred_vec,
                z_vec,
                alpha_cos=float(self.config.residual_alpha_cos),
                beta_var=float(self.config.residual_beta_var),
            )
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unsupported residual_score_mode: {mode}")
        return float(max(score, 0.0))

    def update_states(
        self,
        row: dict[str, Any],
        z: np.ndarray,
        residual_score: float | None = None,
        enable_update_gate: bool = False,
    ) -> None:
        """Update compact node and global states after caller-side event scoring."""
        if self._uses_probationary_memory():
            self._update_states_probationary(
                row,
                z,
                residual_score=residual_score,
                enable_update_gate=enable_update_gate,
            )
            return
        self._clock += 1
        src = int(row["info_src"])
        dst = int(row["info_dst"])
        z_vec = self._checked_vector(z, "z")
        q_t = self._compute_update_q(residual_score, enable_update_gate)
        if src == dst:
            slot = self._ensure_slot(src, row.get("src_role", row.get("dst_role")))
            h_old = self._state_array[slot].copy()
            message = normalize_vector(z_vec + h_old)
            self._state_array[slot] = (
                self.a * h_old + self.g * np.float32(q_t) * message
            ).astype(
                np.float32,
                copy=False,
            )
            self._touch_slot(slot)
        else:
            src_slot = self._ensure_slot(src, row.get("src_role"))
            dst_slot = self._ensure_slot(dst, row.get("dst_role"))
            h_src_old = self._state_array[src_slot].copy()
            h_dst_old = self._state_array[dst_slot].copy()
            u_src = z_vec
            u_dst = normalize_vector(z_vec + h_src_old)
            self._state_array[src_slot] = (
                self.a * h_src_old + self.g * np.float32(q_t) * u_src
            ).astype(
                np.float32,
                copy=False,
            )
            self._state_array[dst_slot] = (
                self.a * h_dst_old + self.g * np.float32(q_t) * u_dst
            ).astype(
                np.float32,
                copy=False,
            )
            self._touch_slot(src_slot)
            self._touch_slot(dst_slot)
        self.global_state = (self.a * self.global_state + self.g * z_vec).astype(
            np.float32,
            copy=False,
        )

    def mark_risk_node(self, node_id: int) -> None:
        """Mark a model-alerted node as protected from normal bounded eviction."""
        if self._uses_probationary_memory():
            self.memory.mark_protected(int(node_id), state_model=self.state_model)
            return
        slot = self.node_to_slot.get(int(node_id))
        if slot is not None:
            self._slot_risk[int(slot)] = np.uint8(1)

    def state_memory_stats(self) -> dict[str, int | float | str]:
        """Return label-free online state memory diagnostics."""
        if self._uses_probationary_memory():
            stats = dict(self.memory.stats())
            stats.update(
                {
                    "state_memory_mode": str(self.config.state_memory_mode),
                    "global_context_mode": str(self.config.global_context_mode),
                    "state_model": str(self.config.state_model),
                    "state_dim": int(self.state_dim),
                    "target_dim": int(self.target_dim),
                },
            )
            return stats
        metadata_bytes = (
            self._slot_node_ids.nbytes
            + self._slot_role_ids.nbytes
            + self._slot_last_touch.nbytes
            + self._slot_touch_count.nbytes
            + self._slot_risk.nbytes
        )
        return {
            "state_memory_mode": str(self.config.state_memory_mode),
            "global_context_mode": str(self.config.global_context_mode),
            "state_slots": int(self.state_slots),
            "state_array_mb": float(self.state_array_mb),
            "state_metadata_mb": float(metadata_bytes / (1024.0 * 1024.0)),
            "prototype_count": int(self._prototype_states.shape[0]),
            "prototype_active_count": int(np.count_nonzero(self._prototype_counts)),
            "eviction_count": int(self._eviction_count),
            "prototype_hit_count": int(self._prototype_hit_count),
            "overflow_count": int(self._overflow_count),
            "risk_slot_count": int(np.count_nonzero(self._slot_risk[: self._state_size])),
            **self.state_merge_profile_stats(),
        }

    @property
    def last_state_merge_metadata(self) -> dict[str, Any]:
        """Return label-free OFSM metadata from the most recent state update."""
        if self._uses_probationary_memory():
            return self.memory.last_merge_metadata()
        return {
            "mode": "none",
            "actions": [],
            "node": {},
            "src": {},
            "dst": {},
            "diagnostics_truncated": False,
        }

    def state_merge_profile_stats(self) -> dict[str, Any]:
        """Return label-free OFSM profile counters."""
        if self._uses_probationary_memory():
            return dict(self.memory.state_merge_profile_stats())
        logical_node_count = int(self.state_slots)
        compression_ratio = 1.0 if logical_node_count else 0.0
        return {
            "state_merge_mode": "none",
            "state_merge_scope": str(self.config.state_merge_scope),
            "merge_threshold": float(self.config.state_merge_threshold),
            "trunc_ratio": float(self.config.state_merge_trunc_ratio),
            "random_prob": float(self.config.state_merge_random_prob),
            "random_seed": int(self.config.state_merge_random_seed),
            "logical_node_count": logical_node_count,
            "num_physical_states": logical_node_count,
            "num_singleton_states": logical_node_count,
            "num_clusters": 0,
            "num_clustered_nodes": 0,
            "compression_ratio": float(compression_ratio),
            "avg_cluster_size": 0.0,
            "max_cluster_size": 0,
            "num_copy_on_write_total": 0,
            "num_merge_attempts_total": 0,
            "num_merge_success_total": 0,
            "num_remerged_total": 0,
            "num_became_singleton_total": 0,
            "num_random_merge_attempts_total": 0,
            "num_random_merge_success_total": 0,
        }

    def state_merge_snapshot(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return score-time OFSM snapshots for source and destination nodes."""
        src = int(row.get("info_src", 0))
        dst = int(row.get("info_dst", 0))
        src_type = row.get("src_type_name", row.get("src_role", "unknown"))
        dst_type = row.get("dst_type_name", row.get("dst_role", "unknown"))
        if self._uses_probationary_memory():
            src_snapshot = self.memory.state_merge_node_snapshot(src, str(src_type))
            dst_snapshot = self.memory.state_merge_node_snapshot(dst, str(dst_type))
        else:
            src_snapshot = self._empty_state_merge_snapshot()
            dst_snapshot = self._empty_state_merge_snapshot()
        return {
            "src": src_snapshot,
            "dst": dst_snapshot,
        }

    def train_batch(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        log_prefix: str = "",
    ) -> dict[str, Any]:
        """Run deterministic mini-batch SGD on normalized cosine and MSE surrogate loss."""
        x = np.asarray(contexts, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.context_dim:
            raise ValueError(f"contexts must have shape (n, {self.context_dim})")
        if y.ndim != 2 or y.shape[1] != self.latent_dim or y.shape[0] != x.shape[0]:
            raise ValueError(f"targets must have shape ({x.shape[0]}, {self.latent_dim})")
        if not bool(np.all(np.isfinite(x))):
            raise ValueError("contexts must contain only finite values")
        if not bool(np.all(np.isfinite(y))):
            raise ValueError("targets must contain only finite values")
        if x.shape[0] == 0:
            return {
                "loss": 0.0,
                "final_loss": 0.0,
                "loss_by_epoch": [],
                "epochs": 0,
                "batches": 0.0,
                "batches_per_epoch": 0,
                "stopped_early": False,
                "stop_reason": "",
            }
        context_rms = np.sqrt(np.mean(x * x, axis=1, keepdims=True)).astype(np.float32)
        x = (x / np.maximum(context_rms, CONTEXT_RMS_EPS)).astype(np.float32, copy=False)
        batch_size = max(int(self.config.batch_size), 1)
        learning_rate = float(self.config.learning_rate)
        l2 = float(self.config.l2)
        epochs = max(int(self.config.epochs), 1)
        total_loss = 0.0
        loss_by_epoch: list[float] = []
        batches = 0
        stopped_early = False
        stop_reason = ""
        best_loss = float("inf")
        plateau_epochs = 0
        min_delta = float(self.config.early_stop_min_delta)
        patience = int(self.config.early_stop_patience)
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            for start in range(0, x.shape[0], batch_size):
                xb = x[start : start + batch_size]
                yb_raw = y[start : start + batch_size]
                step = self._train_batch_step_normalized(
                    xb,
                    yb_raw,
                    learning_rate=learning_rate,
                    l2=l2,
                )
                loss = float(step["loss"])
                if str(step["status"]) != "ok":
                    stopped_early = True
                    stop_reason = str(step["status"])
                    break
                batches += 1
                epoch_batches += 1
                total_loss += loss
                epoch_loss += loss
            mean_epoch_loss = float(epoch_loss / max(epoch_batches, 1))
            loss_by_epoch.append(mean_epoch_loss)
            if log_prefix:
                print(
                    f"{log_prefix} epoch={epoch + 1}/{epochs} loss={mean_epoch_loss:.6f}",
                    file=sys.stderr,
                    flush=True,
                )
            if stopped_early:
                break
            if best_loss - mean_epoch_loss > min_delta:
                best_loss = mean_epoch_loss
                plateau_epochs = 0
            else:
                plateau_epochs += 1
                if patience > 0 and plateau_epochs >= patience:
                    stopped_early = True
                    stop_reason = "loss_plateau"
                    break
        final_loss = float(loss_by_epoch[-1]) if loss_by_epoch else 0.0
        return {
            "loss": float(total_loss / max(batches, 1)),
            "final_loss": final_loss,
            "loss_by_epoch": loss_by_epoch,
            "epochs": int(epochs),
            "epochs_completed": len(loss_by_epoch),
            "batches": float(batches),
            "batches_per_epoch": int(np.ceil(x.shape[0] / batch_size)),
            "stopped_early": bool(stopped_early),
            "stop_reason": str(stop_reason),
            "gradient_clip_norm": float(GRAD_CLIP_NORM),
            "context_rms_normalized": True,
            "early_stop_min_delta": float(min_delta),
            "early_stop_patience": int(patience),
        }

    def train_batch_step(self, contexts: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
        """Run one low-rank SGD update on a finite context and target batch."""
        x = np.asarray(contexts, dtype=np.float32)
        y = np.asarray(targets, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.context_dim:
            raise ValueError(f"contexts must have shape (n, {self.context_dim})")
        if y.ndim != 2 or y.shape[1] != self.latent_dim or y.shape[0] != x.shape[0]:
            raise ValueError(f"targets must have shape ({x.shape[0]}, {self.latent_dim})")
        if not bool(np.all(np.isfinite(x))):
            raise ValueError("contexts must contain only finite values")
        if not bool(np.all(np.isfinite(y))):
            raise ValueError("targets must contain only finite values")
        if x.shape[0] == 0:
            return {
                "loss": 0.0,
                "batch_size": 0,
                "status": "empty_batch",
                "updated": False,
                "gradient_clip_norm": float(GRAD_CLIP_NORM),
                "context_rms_normalized": True,
            }
        context_rms = np.sqrt(np.mean(x * x, axis=1, keepdims=True)).astype(np.float32)
        x_norm = (x / np.maximum(context_rms, CONTEXT_RMS_EPS)).astype(np.float32, copy=False)
        return self._train_batch_step_normalized(
            x_norm,
            y,
            learning_rate=float(self.config.learning_rate),
            l2=float(self.config.l2),
        )

    def train_stream_batches(
        self,
        rows: Any,
        *,
        epochs: int,
        batch_events: int,
        log_prefix: str = "",
        profile_callback: Any | None = None,
    ) -> dict[str, Any]:
        """Train from per-epoch row iterators while holding only temporary batch buffers."""
        total_epochs = int(epochs)
        batch_size = int(batch_events)
        if total_epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_events must be positive")
        loss_by_epoch: list[float] = []
        train_events_seen_total = 0
        train_batches_total = 0
        stopped_early = False
        stop_reason = ""
        best_loss = float("inf")
        plateau_epochs = 0
        min_delta = float(self.config.early_stop_min_delta)
        patience = int(self.config.early_stop_patience)
        for epoch in range(total_epochs):
            batch_contexts: list[np.ndarray] = []
            batch_targets: list[np.ndarray] = []
            epoch_loss = 0.0
            epoch_batches = 0
            epoch_events = 0
            for context, target in rows(epoch):
                batch_contexts.append(self._checked_context(context))
                batch_targets.append(self._checked_vector(target, "target"))
                epoch_events += 1
                train_events_seen_total += 1
                if len(batch_contexts) >= batch_size:
                    stack_started = time.perf_counter()
                    x_batch = np.stack(batch_contexts, axis=0)
                    y_batch = np.stack(batch_targets, axis=0)
                    stack_seconds = time.perf_counter() - stack_started
                    step_started = time.perf_counter()
                    step = self.train_batch_step(
                        x_batch,
                        y_batch,
                    )
                    step_seconds = time.perf_counter() - step_started
                    if profile_callback is not None:
                        profile_callback(
                            {
                                "batch_stack_seconds": float(stack_seconds),
                                "train_batch_step_seconds": float(step_seconds),
                                "batch_events": int(x_batch.shape[0]),
                            },
                        )
                    batch_contexts = []
                    batch_targets = []
                    if str(step["status"]) != "ok":
                        stopped_early = True
                        stop_reason = str(step["status"])
                        break
                    epoch_loss += float(step["loss"])
                    epoch_batches += 1
                    train_batches_total += 1
            if not stopped_early and batch_contexts:
                stack_started = time.perf_counter()
                x_batch = np.stack(batch_contexts, axis=0)
                y_batch = np.stack(batch_targets, axis=0)
                stack_seconds = time.perf_counter() - stack_started
                step_started = time.perf_counter()
                step = self.train_batch_step(
                    x_batch,
                    y_batch,
                )
                step_seconds = time.perf_counter() - step_started
                if profile_callback is not None:
                    profile_callback(
                        {
                            "batch_stack_seconds": float(stack_seconds),
                            "train_batch_step_seconds": float(step_seconds),
                            "batch_events": int(x_batch.shape[0]),
                        },
                    )
                if str(step["status"]) != "ok":
                    stopped_early = True
                    stop_reason = str(step["status"])
                else:
                    epoch_loss += float(step["loss"])
                    epoch_batches += 1
                    train_batches_total += 1
            mean_epoch_loss = float(epoch_loss / max(epoch_batches, 1))
            loss_by_epoch.append(mean_epoch_loss)
            if log_prefix:
                print(
                    (
                        f"{log_prefix} epoch={epoch + 1}/{total_epochs} "
                        f"events={epoch_events} loss={mean_epoch_loss:.6f}"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            if stopped_early:
                break
            if best_loss - mean_epoch_loss > min_delta:
                best_loss = mean_epoch_loss
                plateau_epochs = 0
            else:
                plateau_epochs += 1
                if patience > 0 and plateau_epochs >= patience:
                    stopped_early = True
                    stop_reason = "loss_plateau"
                    break
        return {
            "train_data_mode": "stream_event",
            "loss": float(np.mean(loss_by_epoch)) if loss_by_epoch else 0.0,
            "final_loss": float(loss_by_epoch[-1]) if loss_by_epoch else 0.0,
            "loss_by_epoch": loss_by_epoch,
            "epochs": int(total_epochs),
            "train_epochs_completed": len(loss_by_epoch),
            "epochs_completed": len(loss_by_epoch),
            "train_events_seen_total": int(train_events_seen_total),
            "batch_events": int(batch_size),
            "batches": float(train_batches_total),
            "train_batches_total": int(train_batches_total),
            "stopped_early": bool(stopped_early),
            "stop_reason": str(stop_reason),
            "gradient_clip_norm": float(GRAD_CLIP_NORM),
            "context_rms_normalized": True,
            "early_stop_min_delta": float(min_delta),
            "early_stop_patience": int(patience),
        }

    def _train_batch_step_normalized(
        self,
        contexts: np.ndarray,
        targets: np.ndarray,
        *,
        learning_rate: float,
        l2: float,
    ) -> dict[str, Any]:
        xb = np.asarray(contexts, dtype=np.float32)
        yb_raw = np.asarray(targets, dtype=np.float32)
        hidden = xb @ self.c1
        pred = hidden @ self.c2 + self.bias
        loss, grad_pred = self._loss_and_grad_pred(pred, yb_raw)
        if not np.isfinite(loss):
            return self._batch_step_stats(loss, xb.shape[0], "non_finite_loss", False)
        grad_c2 = hidden.T @ grad_pred + np.float32(l2) * self.c2
        grad_hidden = grad_pred @ self.c2.T
        grad_c1 = xb.T @ grad_hidden + np.float32(l2) * self.c1
        grad_bias = np.sum(grad_pred, axis=0)
        if not self._clip_gradients(grad_c1, grad_c2, grad_bias):
            return self._batch_step_stats(loss, xb.shape[0], "non_finite_gradient", False)
        old_c1 = self.c1.copy()
        old_c2 = self.c2.copy()
        old_bias = self.bias.copy()
        self.c1 = (self.c1 - np.float32(learning_rate) * grad_c1).astype(np.float32)
        self.c2 = (self.c2 - np.float32(learning_rate) * grad_c2).astype(np.float32)
        self.bias = (self.bias - np.float32(learning_rate) * grad_bias).astype(np.float32)
        if not self._parameters_are_finite():
            self.c1 = old_c1
            self.c2 = old_c2
            self.bias = old_bias
            return self._batch_step_stats(
                loss,
                xb.shape[0],
                "non_finite_parameter_update",
                False,
            )
        return self._batch_step_stats(loss, xb.shape[0], "ok", True)

    def _batch_step_stats(
        self,
        loss: float,
        batch_size: int,
        status: str,
        updated: bool,
    ) -> dict[str, Any]:
        return {
            "loss": float(loss) if np.isfinite(loss) else float("inf"),
            "batch_size": int(batch_size),
            "status": str(status),
            "updated": bool(updated),
            "gradient_clip_norm": float(GRAD_CLIP_NORM),
            "context_rms_normalized": True,
        }

    def _checked_context(self, value: np.ndarray) -> np.ndarray:
        context = np.asarray(value, dtype=np.float32)
        if context.shape != (self.context_dim,):
            raise ValueError(f"context must have shape ({self.context_dim},)")
        if not bool(np.all(np.isfinite(context))):
            raise ValueError("context must contain only finite values")
        return context.astype(np.float32, copy=False)

    def _loss_and_grad_pred(
        self,
        pred: np.ndarray,
        target: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        pred_arr = np.asarray(pred, dtype=np.float32)
        target_arr = np.asarray(target, dtype=np.float32)
        if pred_arr.ndim == 1:
            pred_arr = pred_arr.reshape(1, -1)
        if target_arr.ndim == 1:
            target_arr = target_arr.reshape(1, -1)
        yb_norm = np.stack([normalize_vector(row) for row in target_arr])
        pred_norm = np.stack([normalize_vector(row) for row in pred_arr])
        raw_diff = pred_arr - target_arr
        loss = float(np.mean(1.0 - np.sum(pred_norm * yb_norm, axis=1)))
        loss += float(self.config.lambda_mse) * float(np.mean(raw_diff ** 2))
        pred_norms = np.linalg.norm(pred_arr, axis=1, keepdims=True).astype(np.float32)
        safe_pred_norms = np.maximum(pred_norms, np.float32(1e-8))
        cosine_projection = np.sum(pred_norm * yb_norm, axis=1, keepdims=True)
        grad_cosine = ((cosine_projection * pred_norm - yb_norm) / safe_pred_norms)
        grad_mse = 2.0 * float(self.config.lambda_mse) * raw_diff / self.latent_dim
        grad_pred = ((grad_cosine + grad_mse) / float(max(pred_arr.shape[0], 1))).astype(
            np.float32,
            copy=False,
        )
        return loss, grad_pred

    def state_dict(self) -> dict[str, Any]:
        """Return persistent config and learned parameters without online state."""
        return {
            "config": asdict(self.config),
            "c1": self.c1.astype(np.float32, copy=True),
            "c2": self.c2.astype(np.float32, copy=True),
            "bias": self.bias.astype(np.float32, copy=True),
            "state_model": self.state_model.state_dict(),
            "residual_calibration": (
                None
                if self.residual_calibration_model is None
                else self.residual_calibration_model.state_dict()
            ),
            "update_gate_calibrator": (
                None
                if self.update_gate_calibrator is None
                else self.update_gate_calibrator.state_dict()
            ),
        }

    def real_diag_gamma_stats(self) -> dict[str, Any]:
        """Return real-diag gamma diagnostics for config/eval sidecars."""
        if str(self.config.state_model) != "real_diag_learnable" or not hasattr(
            self.state_model,
            "gamma",
        ):
            return {
                "real_diag_train_gamma": bool(self.config.real_diag_train_gamma),
                "real_diag_gamma_active": False,
            }
        gamma = np.asarray(self.state_model.gamma, dtype=np.float32)
        initial = (
            np.asarray(self._real_diag_initial_gamma, dtype=np.float32)
            if self._real_diag_initial_gamma is not None
            else gamma
        )
        return {
            "real_diag_train_gamma": bool(self.config.real_diag_train_gamma),
            "real_diag_gamma_active": True,
            "real_diag_sensitivity_mode": str(self.config.real_diag_sensitivity_mode),
            "real_diag_gamma_min": float(self.config.real_diag_gamma_min),
            "real_diag_gamma_max": float(self.config.real_diag_gamma_max),
            "real_diag_gamma_init": "ema_tau",
            "gamma_initial_mean": float(np.mean(initial)),
            "gamma_initial_min": float(np.min(initial)),
            "gamma_initial_max": float(np.max(initial)),
            "gamma_final_mean": float(np.mean(gamma)),
            "gamma_final_min": float(np.min(gamma)),
            "gamma_final_max": float(np.max(gamma)),
            "gamma_delta_l2": float(np.linalg.norm(gamma - initial)),
        }

    def train_real_diag_online(
        self,
        rows_and_targets,
        log_prefix: str = "",
    ) -> dict[str, Any]:
        """Train E3 low-rank parameters and real-diag gamma with online sensitivity."""
        if str(self.config.state_model) != "real_diag_learnable":
            raise ValueError("real_diag online training requires state_model=real_diag_learnable")
        if str(self.config.real_diag_sensitivity_mode) != "online_stop_message":
            raise ValueError(
                f"unsupported real_diag_sensitivity_mode: {self.config.real_diag_sensitivity_mode}",
            )
        if not bool(self.config.real_diag_train_gamma):
            raise ValueError("real_diag online training requires real_diag_train_gamma=True")
        if not hasattr(self.state_model, "update_theta"):
            raise ValueError("real_diag state model does not support theta updates")

        self.reset_state()
        max_nodes = int(self.config.real_diag_max_sensitivity_nodes)
        sensitivity: dict[int, np.ndarray] = {}
        pending_sensitivity: dict[int, np.ndarray] = {}
        initial_gamma = np.asarray(self.state_model.gamma, dtype=np.float32).copy()
        grad_norms: list[float] = []
        loss_by_epoch: list[float] = []
        batches = 0
        train_events_actual = 0
        learning_rate = float(self.config.learning_rate)
        l2 = float(self.config.l2)

        def sensitivity_for(node_id: int, node_type: object) -> np.ndarray:
            node_key = int(node_id)
            if self._uses_probationary_memory() and node_key not in self.memory.node_to_slot:
                return pending_sensitivity.get(
                    node_key,
                    np.zeros((self.state_dim,), dtype=np.float32),
                )
            return sensitivity.get(node_key, np.zeros((self.state_dim,), dtype=np.float32))

        def set_sensitivity(node_id: int, node_type: object, value: np.ndarray) -> None:
            node_key = int(node_id)
            total_nodes = len(sensitivity) + len(pending_sensitivity)
            if (
                max_nodes > 0
                and node_key not in sensitivity
                and node_key not in pending_sensitivity
                and total_nodes >= max_nodes
            ):
                raise RuntimeError("E3 gamma training sensitivity memory exceeded limit.")
            if self._uses_probationary_memory() and node_key not in self.memory.node_to_slot:
                type_text = str(node_type)
                if "process" in type_text.lower():
                    sensitivity[node_key] = value.astype(np.float32, copy=True)
                    pending_sensitivity.pop(node_key, None)
                else:
                    pending_sensitivity[node_key] = value.astype(np.float32, copy=True)
            else:
                sensitivity[node_key] = value.astype(np.float32, copy=True)
                pending_sensitivity.pop(node_key, None)

        def sync_promoted(node_id: int) -> None:
            node_key = int(node_id)
            if self._uses_probationary_memory() and node_key in self.memory.node_to_slot:
                pending = pending_sensitivity.pop(node_key, None)
                if pending is not None:
                    sensitivity[node_key] = pending.astype(np.float32, copy=True)

        configured_epochs = max(int(self.config.epochs), 1)
        min_delta = float(self.config.early_stop_min_delta)
        patience = int(self.config.early_stop_patience)
        best_loss = float("inf")
        plateau_epochs = 0
        stopped_early = False
        stop_reason = ""
        epoch_loss = 0.0
        epoch_batches = 0
        for fields, target in rows_and_targets:
            z = self._checked_vector(target, "z")
            context = self.make_context(fields)
            context_rms = np.sqrt(np.mean(context * context)).astype(np.float32)
            x = (context / np.maximum(context_rms, CONTEXT_RMS_EPS)).astype(np.float32)
            hidden = x @ self.c1
            pred = hidden @ self.c2 + self.bias
            loss, grad_pred_batch = self._loss_and_grad_pred(pred, z)
            grad_pred = grad_pred_batch.reshape(1, -1)
            if not np.isfinite(loss):
                stopped_early = True
                stop_reason = "non_finite_loss"
                break
            grad_c2 = np.outer(hidden, grad_pred[0]) + np.float32(l2) * self.c2
            grad_hidden = grad_pred @ self.c2.T
            grad_c1 = np.outer(x, grad_hidden[0]) + np.float32(l2) * self.c1
            grad_bias = grad_pred[0]
            grad_context = (grad_hidden @ self.c1.T).reshape(-1).astype(np.float32)
            src = int(fields["info_src"])
            dst = int(fields["info_dst"])
            src_type = fields.get("src_type_name", fields.get("src_role", "unknown"))
            dst_type = fields.get("dst_type_name", fields.get("dst_role", "unknown"))
            src_sens = sensitivity_for(src, src_type)
            dst_sens = sensitivity_for(dst, dst_type)
            offset = self.target_dim if str(self.config.global_context_mode) == "with_global" else 0
            grad_h_src = grad_context[offset : offset + self.target_dim]
            grad_h_dst = grad_context[offset + self.target_dim : offset + 2 * self.target_dim]
            grad_theta = (grad_h_src * src_sens + grad_h_dst * dst_sens).astype(np.float32)

            if not self._clip_gradients(grad_c1, grad_c2, grad_bias):
                stopped_early = True
                stop_reason = "non_finite_gradient"
                break
            self.c1 = (self.c1 - np.float32(learning_rate) * grad_c1).astype(np.float32)
            self.c2 = (self.c2 - np.float32(learning_rate) * grad_c2).astype(np.float32)
            self.bias = (self.bias - np.float32(learning_rate) * grad_bias).astype(np.float32)
            grad_norm = self.state_model.update_theta(
                grad_theta,
                lr=float(self.config.real_diag_gamma_lr),
                weight_decay=float(self.config.real_diag_gamma_weight_decay),
                clip=float(self.config.real_diag_gamma_grad_clip),
            )
            grad_norms.append(float(grad_norm))

            gamma = np.asarray(self.state_model.gamma, dtype=np.float32)
            gamma_prime = self.state_model.gamma_derivative()
            a = np.exp(-gamma).astype(np.float32)
            action = fields.get("raw_action", fields.get("action", ""))
            lambda_t, rho_t = lambda_rho_for_event(action, src_type)
            if src == dst:
                h_old = self.state_model.read(self.get_context_state(src, src_type))
                message = normalize_vector(z + np.float32(lambda_t) * h_old)
                old_s = sensitivity_for(src, src_type)
                new_s = (a * old_s - a * gamma_prime * (h_old - message)).astype(np.float32)
                self.update_states(fields, z, enable_update_gate=False)
                sync_promoted(src)
                set_sensitivity(src, src_type, new_s)
            else:
                h_src = self.state_model.read(self.get_context_state(src, src_type))
                h_dst = self.state_model.read(self.get_context_state(dst, dst_type))
                src_message = (np.float32(rho_t) * z).astype(np.float32, copy=False)
                dst_message = normalize_vector(z + np.float32(lambda_t) * h_src)
                src_new_s = (
                    a * src_sens - a * gamma_prime * (h_src - src_message)
                ).astype(np.float32)
                dst_new_s = (
                    a * dst_sens - a * gamma_prime * (h_dst - dst_message)
                ).astype(np.float32)
                self.update_states(fields, z, enable_update_gate=False)
                sync_promoted(src)
                sync_promoted(dst)
                set_sensitivity(src, src_type, src_new_s)
                set_sensitivity(dst, dst_type, dst_new_s)

            epoch_loss += loss
            epoch_batches += 1
            batches += 1
            train_events_actual += 1
        mean_epoch_loss = float(epoch_loss / max(epoch_batches, 1))
        loss_by_epoch.append(mean_epoch_loss)
        if log_prefix:
            print(
                f"{log_prefix} epoch=1/1 loss={mean_epoch_loss:.6f}",
                file=sys.stderr,
                flush=True,
            )

        gamma = np.asarray(self.state_model.gamma, dtype=np.float32)
        stats = {
            "loss": float(np.mean(loss_by_epoch)) if loss_by_epoch else 0.0,
            "final_loss": float(loss_by_epoch[-1]) if loss_by_epoch else 0.0,
            "loss_by_epoch": loss_by_epoch,
            "epochs": 1,
            "configured_epochs": int(configured_epochs),
            "epochs_completed": len(loss_by_epoch),
            "batches": float(batches),
            "batches_per_epoch": int(epoch_batches),
            "stopped_early": bool(stopped_early),
            "stop_reason": str(stop_reason),
            "gradient_clip_norm": float(GRAD_CLIP_NORM),
            "context_rms_normalized": True,
            "early_stop_min_delta": float(min_delta),
            "early_stop_patience": int(patience),
            "real_diag_train_gamma": True,
            "REAL_DIAG_TRAIN_GAMMA": True,
            "REAL_DIAG_SENSITIVITY_MODE": str(self.config.real_diag_sensitivity_mode),
            "REAL_DIAG_GAMMA_MIN": float(self.config.real_diag_gamma_min),
            "REAL_DIAG_GAMMA_MAX": float(self.config.real_diag_gamma_max),
            "REAL_DIAG_GAMMA_INIT": "ema_tau",
            "gamma_initial_mean": float(np.mean(initial_gamma)),
            "gamma_initial_min": float(np.min(initial_gamma)),
            "gamma_initial_max": float(np.max(initial_gamma)),
            "gamma_final_mean": float(np.mean(gamma)),
            "gamma_final_min": float(np.min(gamma)),
            "gamma_final_max": float(np.max(gamma)),
            "gamma_delta_l2": float(np.linalg.norm(gamma - initial_gamma)),
            "gamma_grad_norm_mean": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "gamma_grad_norm_max": float(np.max(grad_norms)) if grad_norms else 0.0,
            "sensitivity_mode": str(self.config.real_diag_sensitivity_mode),
            "sensitivity_nodes": int(len(sensitivity) + len(pending_sensitivity)),
            "train_events_actual": int(train_events_actual),
        }
        self.real_diag_gamma_training_stats = dict(stats)
        self.reset_state()
        return stats

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SSPMLowRankModel":
        """Restore a model from persistent config and low-rank parameters only."""
        config = SSPMLowRankConfig(**dict(state["config"]))
        model = cls(config)
        if "state_model" in state:
            model.state_model = build_state_model(
                str(config.state_model),
                state_dim=int(config.state_dim),
                target_dim=int(config.target_dim),
                seed=int(config.seed),
                state=state["state_model"],
                gamma_min=float(config.real_diag_gamma_min),
                gamma_max=float(config.real_diag_gamma_max),
            )
        try:
            model.set_lowrank_weights(state["c1"], state["c2"], state["bias"])
        except ValueError as exc:
            raise ValueError(str(exc).replace("mismatch", "does not match config")) from exc
        residual_calibration = state.get("residual_calibration")
        if residual_calibration is not None:
            model.set_residual_calibration(
                ResidualCalibrationModel.from_state_dict(dict(residual_calibration)),
            )
        update_gate_calibrator = state.get("update_gate_calibrator")
        if update_gate_calibrator is not None:
            model.set_update_gate_calibrator(
                UpdateGateCalibrator.from_state_dict(dict(update_gate_calibrator)),
            )
        model.reset_state()
        return model

    def _uses_probationary_memory(self) -> bool:
        return (
            str(self.config.state_memory_mode) == "bounded"
            and str(self.config.state_memory_policy) == "probationary_lru"
        )

    def _make_state_merge_config(self) -> OnlineStateMergeConfig:
        return OnlineStateMergeConfig(
            mode=str(self.config.state_merge_mode),
            scope=str(self.config.state_merge_scope),
            state_dim=int(self.state_dim),
            threshold=float(self.config.state_merge_threshold),
            trunc_ratio=float(self.config.state_merge_trunc_ratio),
            use_rfft=bool(self.config.state_merge_use_rfft),
            use_real_only=bool(self.config.state_merge_use_real_only),
            center_state=bool(self.config.state_merge_center_state),
            normalize=str(self.config.state_merge_normalize),
            eps=float(self.config.state_merge_eps),
            min_cluster_size=int(self.config.state_merge_min_cluster_size),
            copy_on_write=bool(self.config.state_merge_copy_on_write),
            random_prob=float(self.config.state_merge_random_prob),
            random_seed=int(self.config.state_merge_random_seed),
            diagnostics_max_rows=int(self.config.state_merge_diagnostics_max_rows),
            match_backend=str(self.config.state_merge_match_backend),
            candidate_cap=int(self.config.state_merge_candidate_cap),
            merge_interval=int(self.config.state_merge_interval),
            merge_min_count=int(self.config.state_merge_min_count),
            diagnostics_enabled=bool(self.config.state_merge_diagnostics_enabled),
        )

    def _empty_state_merge_snapshot(self) -> dict[str, Any]:
        return {
            "cluster_id": "",
            "cluster_size": 1,
            "merge_similarity": 0.0,
            "is_clustered": 0,
            "merge_mode": "none",
            "physical_state_id": "",
        }

    def _update_states_probationary(
        self,
        row: dict[str, Any],
        z: np.ndarray,
        residual_score: float | None = None,
        enable_update_gate: bool = False,
    ) -> None:
        src = int(row["info_src"])
        dst = int(row["info_dst"])
        z_vec = self._checked_vector(z, "z")
        action = row.get("raw_action", row.get("action", ""))
        src_type = row.get("src_type_name", row.get("src_role", "unknown"))
        dst_type = row.get("dst_type_name", row.get("dst_role", "unknown"))
        q_t = self._compute_update_q(residual_score, enable_update_gate)
        lambda_t, rho_t = lambda_rho_for_event(action, src_type)
        self.last_lambda_t = float(lambda_t)
        self.last_rho_t = float(rho_t)
        self.last_q_t = float(q_t)
        current = {src, dst}
        if src == dst:
            h_old = self.state_model.read(self.get_context_state(src, src_type))
            message = normalize_vector(z_vec + np.float32(lambda_t) * h_old)
            self.memory.update_state(
                src,
                str(src_type),
                message,
                self.state_model,
                q_t=q_t,
                protected=False,
                current_node_ids=current,
                role="src",
            )
        else:
            h_src = self.state_model.read(self.get_context_state(src, src_type))
            dst_message = normalize_vector(z_vec + np.float32(lambda_t) * h_src)
            src_message = (np.float32(rho_t) * z_vec).astype(np.float32, copy=False)
            self.memory.update_state(
                src,
                str(src_type),
                src_message,
                self.state_model,
                q_t=q_t,
                protected=False,
                current_node_ids=current,
                role="src",
            )
            self.memory.update_state(
                dst,
                str(dst_type),
                dst_message,
                self.state_model,
                q_t=q_t,
                protected=False,
                current_node_ids=current,
                role="dst",
            )
        self.global_state = (self.a * self.global_state + self.g * z_vec).astype(
            np.float32,
            copy=False,
        )

    def _ensure_slot(self, node_id: int, role: str | int | None = None) -> int:
        slot = self.node_to_slot.get(int(node_id))
        if slot is not None:
            return int(slot)
        slot = self._slot_for_new_node()
        self._ensure_capacity(slot + 1)
        self.node_to_slot[int(node_id)] = slot
        role_id = self._role_id(role)
        self._slot_node_ids[slot] = int(node_id)
        self._slot_role_ids[slot] = int(role_id)
        self._slot_last_touch[slot] = int(self._clock)
        self._slot_touch_count[slot] = 0
        self._slot_risk[slot] = np.uint8(0)
        if int(self._prototype_counts[role_id]) > 0:
            self._state_array[slot] = self._prototype_states[role_id]
            self._prototype_hit_count += 1
        else:
            self._state_array[slot].fill(0.0)
        if slot == int(self._state_size):
            self._state_size += 1
        return slot

    def _ensure_capacity(self, required: int) -> None:
        if required <= int(self._state_array.shape[0]):
            return
        current = int(self._state_array.shape[0])
        new_capacity = max(required, max(16, current * 2))
        new_array = np.zeros((new_capacity, self.latent_dim), dtype=np.float32)
        if self._state_size:
            new_array[: self._state_size] = self._state_array[: self._state_size]
        self._state_array = new_array
        self._slot_node_ids = self._grow_1d_array(self._slot_node_ids, new_capacity, np.int64)
        self._slot_role_ids = self._grow_1d_array(self._slot_role_ids, new_capacity, np.int32)
        self._slot_last_touch = self._grow_1d_array(self._slot_last_touch, new_capacity, np.int64)
        self._slot_touch_count = self._grow_1d_array(
            self._slot_touch_count,
            new_capacity,
            np.int32,
        )
        self._slot_risk = self._grow_1d_array(self._slot_risk, new_capacity, np.uint8)

    def _slot_for_new_node(self) -> int:
        if str(self.config.state_memory_mode) != "bounded":
            return int(self._state_size)
        max_exact = int(self.config.max_exact_states)
        if max_exact <= 0 or int(self._state_size) < max_exact:
            return int(self._state_size)
        evict_slot = self._find_evictable_slot()
        if evict_slot is None:
            self._overflow_count += 1
            return int(self._state_size)
        self._evict_slot(int(evict_slot))
        return int(self._state_size)

    def _find_evictable_slot(self) -> int | None:
        if self._state_size <= 0:
            return None
        ages = int(self._clock) - self._slot_last_touch[: self._state_size]
        eligible = np.where(
            (self._slot_risk[: self._state_size] == 0)
            & (ages >= int(self.config.min_inactive_events)),
        )[0]
        if eligible.size == 0:
            return None
        touches = self._slot_touch_count[eligible]
        oldest = self._slot_last_touch[eligible]
        order = np.lexsort((oldest, touches))
        return int(eligible[int(order[0])])

    def _evict_slot(self, slot: int) -> None:
        self._update_prototype_from_slot(slot)
        old_node = int(self._slot_node_ids[slot])
        self.node_to_slot.pop(old_node, None)
        last_slot = int(self._state_size) - 1
        if slot != last_slot:
            moved_node = int(self._slot_node_ids[last_slot])
            self._state_array[slot] = self._state_array[last_slot]
            self._slot_node_ids[slot] = self._slot_node_ids[last_slot]
            self._slot_role_ids[slot] = self._slot_role_ids[last_slot]
            self._slot_last_touch[slot] = self._slot_last_touch[last_slot]
            self._slot_touch_count[slot] = self._slot_touch_count[last_slot]
            self._slot_risk[slot] = self._slot_risk[last_slot]
            self.node_to_slot[moved_node] = slot
        self._state_array[last_slot].fill(0.0)
        self._slot_node_ids[last_slot] = 0
        self._slot_role_ids[last_slot] = 0
        self._slot_last_touch[last_slot] = 0
        self._slot_touch_count[last_slot] = 0
        self._slot_risk[last_slot] = np.uint8(0)
        self._state_size -= 1
        self._eviction_count += 1

    def _update_prototype_from_slot(self, slot: int) -> None:
        role_id = int(self._slot_role_ids[slot])
        alpha = np.float32(float(self.config.prototype_update_alpha))
        count = int(self._prototype_counts[role_id])
        if count <= 0:
            self._prototype_states[role_id] = self._state_array[slot]
        else:
            self._prototype_states[role_id] = (
                (np.float32(1.0) - alpha) * self._prototype_states[role_id]
                + alpha * self._state_array[slot]
            ).astype(np.float32, copy=False)
        self._prototype_counts[role_id] = np.int32(min(count + 1, np.iinfo(np.int32).max))

    def _touch_slot(self, slot: int) -> None:
        self._slot_last_touch[int(slot)] = int(self._clock)
        current = int(self._slot_touch_count[int(slot)])
        self._slot_touch_count[int(slot)] = np.int32(min(current + 1, np.iinfo(np.int32).max))

    def _role_id(self, role: str | int | None) -> int:
        count = max(int(self.config.prototype_count), 1)
        if role is None:
            return 0
        return int(stable_hash(str(role), seed=29) % count)

    @staticmethod
    def _grow_1d_array(array: np.ndarray, new_capacity: int, dtype: Any) -> np.ndarray:
        grown = np.zeros((new_capacity,), dtype=dtype)
        if array.size:
            grown[: array.size] = array
        return grown

    def _checked_vector(self, value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (self.latent_dim,):
            raise ValueError(f"{name} must have shape ({self.latent_dim},)")
        if not bool(np.all(np.isfinite(vector))):
            raise ValueError(f"{name} must contain only finite values")
        return vector.astype(np.float32, copy=False)

    def _compute_update_q(
        self,
        residual_score: float | None,
        enable_update_gate: bool,
    ) -> float:
        if not bool(enable_update_gate):
            q_t = 1.0
        elif str(self.config.update_gate_mode) == "none":
            q_t = fixed_update_gate("none", residual_score)
        elif str(self.config.update_gate_mode) == "quantile":
            if self.update_gate_calibrator is None:
                raise ValueError("update gate quantile threshold is not fitted")
            if residual_score is None:
                raise ValueError("residual_score is required for quantile update gate")
            q_t = self.update_gate_calibrator.compute_q(float(residual_score))
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unsupported update_gate_mode: {self.config.update_gate_mode}")
        self.last_q_t = float(q_t)
        return float(q_t)

    def _clip_gradients(
        self,
        grad_c1: np.ndarray,
        grad_c2: np.ndarray,
        grad_bias: np.ndarray,
    ) -> bool:
        if not (
            bool(np.all(np.isfinite(grad_c1)))
            and bool(np.all(np.isfinite(grad_c2)))
            and bool(np.all(np.isfinite(grad_bias)))
        ):
            return False
        norm_sq = (
            float(np.sum(grad_c1.astype(np.float64) ** 2))
            + float(np.sum(grad_c2.astype(np.float64) ** 2))
            + float(np.sum(grad_bias.astype(np.float64) ** 2))
        )
        grad_norm = float(np.sqrt(max(norm_sq, 0.0)))
        clip_norm = float(GRAD_CLIP_NORM)
        if grad_norm > clip_norm:
            scale = np.float32(clip_norm / max(grad_norm, 1e-12))
            grad_c1 *= scale
            grad_c2 *= scale
            grad_bias *= scale
        return True

    def _parameters_are_finite(self) -> bool:
        return (
            bool(np.all(np.isfinite(self.c1)))
            and bool(np.all(np.isfinite(self.c2)))
            and bool(np.all(np.isfinite(self.bias)))
        )

    def _validate_config(self) -> None:
        if int(self.config.latent_dim) <= 0:
            raise ValueError("latent_dim must be positive")
        if int(self.config.target_dim) <= 0:
            raise ValueError("target_dim must be positive")
        if int(self.config.state_dim) <= 0:
            raise ValueError("state_dim must be positive")
        if int(self.config.target_dim) != int(self.config.latent_dim):
            raise ValueError("target_dim must match latent_dim")
        if int(self.config.state_dim) != int(self.config.target_dim):
            raise ValueError("state_dim must match target_dim")
        if int(self.config.rank) <= 0:
            raise ValueError("rank must be positive")
        if int(self.config.batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if float(self.config.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")
        if float(self.config.l2) < 0.0:
            raise ValueError("l2 must be non-negative")
        if float(self.config.lambda_mse) < 0.0:
            raise ValueError("lambda_mse must be non-negative")
        if int(self.config.epochs) <= 0:
            raise ValueError("epochs must be positive")
        if float(self.config.early_stop_min_delta) < 0.0:
            raise ValueError("early_stop_min_delta must be non-negative")
        if int(self.config.early_stop_patience) < 0:
            raise ValueError("early_stop_patience must be non-negative")
        if str(self.config.context_mode) not in {"with_action", "no_action"}:
            raise ValueError("context_mode must be with_action or no_action")
        if str(self.config.context_action_mode) not in {"raw_orthrus10", "legacy"}:
            raise ValueError("context_action_mode must be raw_orthrus10 or legacy")
        if int(self.config.action_count) <= 0:
            raise ValueError("action_count must be positive")
        if str(self.config.global_context_mode) not in {"with_global", "no_global"}:
            raise ValueError("global_context_mode must be with_global or no_global")
        if str(self.config.state_memory_mode) not in {"unbounded", "bounded"}:
            raise ValueError("state_memory_mode must be unbounded or bounded")
        if str(self.config.state_memory_policy) not in {"legacy", "probationary_lru"}:
            raise ValueError("state_memory_policy must be legacy or probationary_lru")
        if int(self.config.active_max_nodes) < 0:
            raise ValueError("active_max_nodes must be non-negative")
        if int(self.config.probationary_max_nodes) < 0:
            raise ValueError("probationary_max_nodes must be non-negative")
        if int(self.config.probationary_min_count) <= 0:
            raise ValueError("probationary_min_count must be positive")
        if int(self.config.lru_evict_batch) <= 0:
            raise ValueError("lru_evict_batch must be positive")
        if int(self.config.alert_protect_events) < 0:
            raise ValueError("alert_protect_events must be non-negative")
        if str(self.config.update_gate_mode) not in {"none", "quantile"}:
            raise ValueError("update_gate_mode must be none or quantile")
        if not 0.0 <= float(self.config.update_gate_quantile) <= 1.0:
            raise ValueError("update_gate_quantile must be in [0, 1]")
        if not 0.0 < float(self.config.update_gate_q_min) <= 1.0:
            raise ValueError("update_gate_q_min must be in (0, 1]")
        if float(self.config.update_gate_eta) <= 0.0:
            raise ValueError("update_gate_eta must be positive")
        if str(self.config.residual_score_mode) not in {"legacy", "raw_mse", "var_calibrated"}:
            raise ValueError("residual_score_mode must be legacy, raw_mse, or var_calibrated")
        if str(self.config.residual_calibration) not in {"none", "action_diag"}:
            raise ValueError("residual_calibration must be none or action_diag")
        if str(self.config.residual_score_mode) == "var_calibrated" and (
            str(self.config.residual_calibration) != "action_diag"
        ):
            raise ValueError("var_calibrated residual scoring requires action_diag calibration")
        if str(self.config.residual_score_mode) != "var_calibrated" and (
            str(self.config.residual_calibration) == "action_diag"
        ):
            raise ValueError("action_diag calibration requires var_calibrated residual scoring")
        if float(self.config.residual_alpha_cos) < 0.0:
            raise ValueError("residual_alpha_cos must be non-negative")
        if float(self.config.residual_beta_mse) < 0.0:
            raise ValueError("residual_beta_mse must be non-negative")
        if float(self.config.residual_beta_var) < 0.0:
            raise ValueError("residual_beta_var must be non-negative")
        if float(self.config.residual_var_eps) <= 0.0:
            raise ValueError("residual_var_eps must be positive")
        if int(self.config.residual_calibration_min_count) < 2:
            raise ValueError("residual_calibration_min_count must be at least 2")
        if str(self.config.state_merge_mode) not in {
            "none",
            "online_fourier",
            "online_time_domain",
            "random",
        }:
            raise ValueError(
                "state_merge_mode must be none, online_fourier, online_time_domain, or random",
            )
        if str(self.config.state_merge_scope) != "same_type":
            raise ValueError("state_merge_scope must be same_type")
        if str(self.config.state_merge_mode) != "none":
            if (
                str(self.config.state_memory_mode) != "bounded"
                or str(self.config.state_memory_policy) != "probationary_lru"
            ):
                raise ValueError(
                    "OFSM requires state_memory_mode=bounded and "
                    "sspm_state_memory_policy=probationary_lru.",
                )
        if not 0.0 <= float(self.config.state_merge_threshold) <= 1.0:
            raise ValueError("state_merge_threshold must be in [0, 1]")
        if not 0.0 < float(self.config.state_merge_trunc_ratio) <= 1.0:
            raise ValueError("state_merge_trunc_ratio must be in (0, 1]")
        if str(self.config.state_merge_normalize) != "l2":
            raise ValueError("state_merge_normalize must be l2")
        if float(self.config.state_merge_eps) <= 0.0:
            raise ValueError("state_merge_eps must be positive")
        if int(self.config.state_merge_min_cluster_size) < 2:
            raise ValueError("state_merge_min_cluster_size must be at least 2")
        if not 0.0 <= float(self.config.state_merge_random_prob) <= 1.0:
            raise ValueError("state_merge_random_prob must be in [0, 1]")
        if int(self.config.state_merge_diagnostics_max_rows) < 0:
            raise ValueError("state_merge_diagnostics_max_rows must be non-negative")
        if int(self.config.max_exact_states) < 0:
            raise ValueError("max_exact_states must be non-negative")
        if int(self.config.min_inactive_events) < 0:
            raise ValueError("min_inactive_events must be non-negative")
        if int(self.config.prototype_count) <= 0:
            raise ValueError("prototype_count must be positive")
        if not 0.0 < float(self.config.prototype_update_alpha) <= 1.0:
            raise ValueError("prototype_update_alpha must be in (0, 1]")
        int(self.config.seed)
