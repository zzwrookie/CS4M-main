from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from cs4m.scoring.simple_gates import ORTHRUS10_RAW_ACTIONS, checked_raw_action


def _normalize_vector(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vector = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= float(eps):
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / np.float32(norm)).astype(np.float32, copy=False)


@dataclass
class _RunningDiagStats:
    count: int
    mean: np.ndarray
    m2: np.ndarray

    @classmethod
    def create(cls, latent_dim: int) -> "_RunningDiagStats":
        return cls(
            count=0,
            mean=np.zeros((int(latent_dim),), dtype=np.float64),
            m2=np.zeros((int(latent_dim),), dtype=np.float64),
        )

    def update(self, value: np.ndarray) -> None:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != self.mean.shape:
            raise ValueError(f"residual must have shape {self.mean.shape}")
        if not bool(np.all(np.isfinite(vector))):
            raise ValueError("residual must contain only finite values")
        self.count += 1
        delta = vector - self.mean
        self.mean += delta / float(self.count)
        delta2 = vector - self.mean
        self.m2 += delta * delta2

    def variance(self, epsilon: float) -> np.ndarray:
        if int(self.count) < 2:
            raise ValueError("at least two samples are required to estimate variance")
        variance = self.m2 / float(max(int(self.count) - 1, 1))
        return np.maximum(variance, float(epsilon)).astype(np.float32)


class ResidualCalibrationModel:
    """Finalized action-conditioned diagonal residual variance calibration."""

    def __init__(
        self,
        *,
        latent_dim: int,
        epsilon: float,
        min_count: int,
        action_variance: Mapping[str, np.ndarray],
        action_counts: Mapping[str, int],
        action_fallbacks: Mapping[str, bool],
        global_variance: np.ndarray,
        global_count: int,
        warnings_list: Sequence[str],
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.epsilon = float(epsilon)
        self.min_count = int(min_count)
        self.action_variance = {
            checked_raw_action(action): np.asarray(variance, dtype=np.float32)
            for action, variance in action_variance.items()
        }
        self.action_counts = {
            checked_raw_action(action): int(count)
            for action, count in action_counts.items()
        }
        self.action_fallbacks = {
            checked_raw_action(action): bool(value)
            for action, value in action_fallbacks.items()
        }
        self.global_variance = np.asarray(global_variance, dtype=np.float32)
        self.global_count = int(global_count)
        self.warnings = [str(item) for item in warnings_list]
        self._validate()

    def score_var_calibrated(
        self,
        action: object,
        z_hat: np.ndarray,
        z_true: np.ndarray,
        alpha_cos: float = 1.0,
        beta_var: float = 0.25,
    ) -> float:
        """Return cosine plus action-conditioned diagonal variance score."""
        token = checked_raw_action(action)
        pred = self._checked_vector(z_hat, "z_hat")
        true = self._checked_vector(z_true, "z_true")
        pred_norm = _normalize_vector(pred)
        true_norm = _normalize_vector(true)
        cosine = float(1.0 - np.dot(pred_norm, true_norm))
        diff = (true - pred).astype(np.float32, copy=False)
        variance = self.action_variance[token]
        var_score = float(np.sum((diff * diff) / (variance + np.float32(self.epsilon))))
        score = float(alpha_cos) * cosine + float(beta_var) * var_score
        return float(max(score, 0.0))

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe metadata for config, metrics, and profiling output."""
        actions: dict[str, dict[str, Any]] = {}
        fallback_actions = []
        for action in ORTHRUS10_RAW_ACTIONS:
            fallback = bool(self.action_fallbacks[action])
            if fallback:
                fallback_actions.append(action)
            actions[action] = {
                "count": int(self.action_counts[action]),
                "global_fallback": fallback,
                "variance_min": float(np.min(self.action_variance[action])),
                "variance_max": float(np.max(self.action_variance[action])),
            }
        return {
            "calibration_fitted": True,
            "calibration_actions": list(ORTHRUS10_RAW_ACTIONS),
            "actions": actions,
            "fallback_actions": fallback_actions,
            "fallback_actions_count": int(len(fallback_actions)),
            "global_count": int(self.global_count),
            "min_count": int(self.min_count),
            "epsilon": float(self.epsilon),
            "warnings": list(self.warnings),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a pickle-safe calibration state without validation rows."""
        return {
            "latent_dim": int(self.latent_dim),
            "epsilon": float(self.epsilon),
            "min_count": int(self.min_count),
            "action_variance": {
                action: self.action_variance[action].astype(np.float32, copy=True)
                for action in ORTHRUS10_RAW_ACTIONS
            },
            "action_counts": dict(self.action_counts),
            "action_fallbacks": dict(self.action_fallbacks),
            "global_variance": self.global_variance.astype(np.float32, copy=True),
            "global_count": int(self.global_count),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "ResidualCalibrationModel":
        """Restore finalized calibration state."""
        return cls(
            latent_dim=int(state["latent_dim"]),
            epsilon=float(state["epsilon"]),
            min_count=int(state["min_count"]),
            action_variance=dict(state["action_variance"]),
            action_counts=dict(state["action_counts"]),
            action_fallbacks=dict(state["action_fallbacks"]),
            global_variance=np.asarray(state["global_variance"], dtype=np.float32),
            global_count=int(state["global_count"]),
            warnings_list=list(state.get("warnings", [])),
        )

    def _validate(self) -> None:
        expected = set(ORTHRUS10_RAW_ACTIONS)
        if set(self.action_variance) != expected:
            raise ValueError("action_diag calibration must cover all ORTHRUS10 actions")
        if set(self.action_counts) != expected:
            raise ValueError("action counts must cover all ORTHRUS10 actions")
        if set(self.action_fallbacks) != expected:
            raise ValueError("action fallback flags must cover all ORTHRUS10 actions")
        if self.global_variance.shape != (self.latent_dim,):
            raise ValueError("global variance shape does not match latent_dim")
        for action, variance in self.action_variance.items():
            if variance.shape != (self.latent_dim,):
                raise ValueError(f"variance shape mismatch for {action}")
            if not bool(np.all(np.isfinite(variance))):
                raise ValueError(f"variance must be finite for {action}")
            if bool(np.any(variance < float(self.epsilon))):
                raise ValueError(f"variance below epsilon for {action}")

    def _checked_vector(self, value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (self.latent_dim,):
            raise ValueError(f"{name} must have shape ({self.latent_dim},)")
        if not bool(np.all(np.isfinite(vector))):
            raise ValueError(f"{name} must contain only finite values")
        return vector.astype(np.float32, copy=False)


class ResidualCalibrationStats:
    """Collect validation residuals for action-conditioned diagonal calibration."""

    def __init__(self, latent_dim: int, epsilon: float = 1e-6, min_count: int = 2) -> None:
        if int(latent_dim) <= 0:
            raise ValueError("latent_dim must be positive")
        if float(epsilon) <= 0.0:
            raise ValueError("epsilon must be positive")
        if int(min_count) < 2:
            raise ValueError("min_count must be at least 2")
        self.latent_dim = int(latent_dim)
        self.epsilon = float(epsilon)
        self.min_count = int(min_count)
        self._global = _RunningDiagStats.create(self.latent_dim)
        self._by_action = {
            action: _RunningDiagStats.create(self.latent_dim)
            for action in ORTHRUS10_RAW_ACTIONS
        }
        self._warnings: list[str] = []

    def update(
        self,
        action: object,
        residual: np.ndarray | None = None,
        z_hat: np.ndarray | None = None,
        z_true: np.ndarray | None = None,
    ) -> None:
        """Update validation residual statistics for one raw ORTHRUS action."""
        token = checked_raw_action(action)
        if residual is None:
            if z_hat is None or z_true is None:
                raise ValueError("residual or z_hat/z_true must be provided")
            residual_value = np.asarray(z_true, dtype=np.float32) - np.asarray(
                z_hat,
                dtype=np.float32,
            )
        else:
            residual_value = np.asarray(residual, dtype=np.float32)
        if residual_value.shape != (self.latent_dim,):
            raise ValueError(f"residual must have shape ({self.latent_dim},)")
        if not bool(np.all(np.isfinite(residual_value))):
            raise ValueError("residual must contain only finite values")
        self._global.update(residual_value)
        self._by_action[token].update(residual_value)

    def finalize(self) -> ResidualCalibrationModel:
        """Finalize action variances, using global fallback for low-count actions."""
        if int(self._global.count) < int(self.min_count):
            raise ValueError(
                f"global residual count {self._global.count} < min_count {self.min_count}",
            )
        global_variance = self._global.variance(self.epsilon)
        action_variance: dict[str, np.ndarray] = {}
        action_counts: dict[str, int] = {}
        action_fallbacks: dict[str, bool] = {}
        warnings_list = list(self._warnings)
        for action in ORTHRUS10_RAW_ACTIONS:
            stats = self._by_action[action]
            action_counts[action] = int(stats.count)
            if int(stats.count) >= int(self.min_count):
                action_variance[action] = stats.variance(self.epsilon)
                action_fallbacks[action] = False
            else:
                action_variance[action] = global_variance.astype(np.float32, copy=True)
                action_fallbacks[action] = True
                warning = (
                    f"action {action} validation count {stats.count} < "
                    f"min_count {self.min_count}; using global variance fallback"
                )
                warnings_list.append(warning)
                warnings.warn(warning, RuntimeWarning, stacklevel=2)
        return ResidualCalibrationModel(
            latent_dim=self.latent_dim,
            epsilon=self.epsilon,
            min_count=self.min_count,
            action_variance=action_variance,
            action_counts=action_counts,
            action_fallbacks=action_fallbacks,
            global_variance=global_variance,
            global_count=int(self._global.count),
            warnings_list=warnings_list,
        )


class UpdateGateCalibrator:
    """Validation-only calibrator for anomaly-aware state update gates."""

    def __init__(
        self,
        quantile: float = 0.99,
        q_min: float = 0.05,
        eta: float = 1.0,
        threshold_mode: str = "quantile",
    ) -> None:
        if not math.isfinite(float(quantile)) or not 0.0 <= float(quantile) <= 1.0:
            raise ValueError("quantile must be finite and in [0, 1]")
        if not math.isfinite(float(q_min)) or not 0.0 < float(q_min) <= 1.0:
            raise ValueError("q_min must be finite and in (0, 1]")
        if not math.isfinite(float(eta)) or float(eta) <= 0.0:
            raise ValueError("eta must be positive")
        if str(threshold_mode) not in {"quantile", "validation_max"}:
            raise ValueError("threshold_mode must be quantile or validation_max")
        self.quantile = float(quantile)
        self.q_min = float(q_min)
        self.eta = float(eta)
        self.threshold_mode = str(threshold_mode)
        self.threshold: float | None = None
        self.num_validation_events = 0

    def fit(self, validation_scores: Sequence[float] | np.ndarray) -> "UpdateGateCalibrator":
        """Fit Q from validation residual scores only."""
        scores = np.asarray(validation_scores, dtype=np.float64)
        if scores.ndim != 1 or scores.size == 0:
            raise ValueError("validation residual scores must be a non-empty 1D sequence")
        if not bool(np.all(np.isfinite(scores))):
            raise ValueError("validation residual scores must be finite")
        if self.threshold_mode == "validation_max":
            self.threshold = float(np.max(scores))
        else:
            self.threshold = float(np.quantile(scores, self.quantile, method="higher"))
        self.num_validation_events = int(scores.size)
        return self

    def compute_q(self, score: float) -> float:
        """Return q_t for a current event score using the fitted threshold."""
        if self.threshold is None:
            raise ValueError("update gate quantile threshold is not fitted")
        score_value = float(score)
        if not math.isfinite(score_value):
            raise ValueError("score must be finite")
        if score_value <= float(self.threshold):
            return 1.0
        decayed = math.exp(-float(self.eta) * (score_value - float(self.threshold)))
        return float(max(float(self.q_min), decayed))

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe update gate calibration metadata."""
        return {
            "mode": "quantile",
            "threshold_mode": str(self.threshold_mode),
            "quantile": float(self.quantile),
            "q_min": float(self.q_min),
            "eta": float(self.eta),
            "threshold": self.threshold if self.threshold is None else float(self.threshold),
            "num_validation_events": int(self.num_validation_events),
            "fitted": self.threshold is not None,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return a pickle-safe fitted gate state."""
        return self.summary()

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "UpdateGateCalibrator":
        """Restore a fitted or unfitted gate calibrator."""
        calibrator = cls(
            quantile=float(state.get("quantile", 0.99)),
            q_min=float(state.get("q_min", 0.05)),
            eta=float(state.get("eta", 1.0)),
            threshold_mode=str(state.get("threshold_mode", "quantile")),
        )
        threshold = state.get("threshold")
        calibrator.threshold = None if threshold is None else float(threshold)
        calibrator.num_validation_events = int(state.get("num_validation_events", 0))
        return calibrator
