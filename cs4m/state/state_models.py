from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


DEFAULT_TAU = (2, 4, 8, 16, 32, 64, 128, 256)
GAMMA_MIN = np.float32(1e-4)
GAMMA_MAX = np.float32(1.0)


class SSPMStateModel(Protocol):
    name: str
    state_dim: int
    target_dim: int

    def read(self, state: np.ndarray) -> np.ndarray:
        """Return the real target-space readout for a node state."""

    def step(self, state: np.ndarray, message: np.ndarray, q_t: float = 1.0) -> np.ndarray:
        """Return the next state after one recurrent update."""

    def state_dict(self) -> dict[str, Any]:
        """Return serializable fixed/initialized parameters."""


def _tau(dim: int) -> np.ndarray:
    return np.resize(np.asarray(DEFAULT_TAU, dtype=np.float32), int(dim))


def _checked_vector(value: np.ndarray, dim: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (int(dim),):
        raise ValueError(f"{name} must have shape ({int(dim)},)")
    if not bool(np.all(np.isfinite(vector))):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float32, copy=False)


def _softplus(theta: np.ndarray) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float32)
    return (np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)).astype(np.float32)


def _sigmoid(theta: np.ndarray) -> np.ndarray:
    values = np.asarray(theta, dtype=np.float32)
    positive = values >= 0
    out = np.empty_like(values, dtype=np.float32)
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out.astype(np.float32, copy=False)


def _inverse_softplus(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=np.float32), np.float32(1e-8))
    return np.log(np.expm1(clipped)).astype(np.float32)


@dataclass
class EmaFixedStateModel:
    name: str
    state_dim: int
    target_dim: int
    a: np.ndarray
    g: np.ndarray

    @classmethod
    def create(cls, state_dim: int, target_dim: int) -> "EmaFixedStateModel":
        if int(state_dim) != int(target_dim):
            raise ValueError("ema_fixed requires state_dim == target_dim")
        tau = _tau(int(state_dim))
        a = np.exp(np.float32(-1.0) / tau).astype(np.float32)
        g = (np.float32(1.0) - a).astype(np.float32)
        return cls("ema_fixed", int(state_dim), int(target_dim), a, g)

    def read(self, state: np.ndarray) -> np.ndarray:
        return _checked_vector(state, self.state_dim, "state").astype(np.float32, copy=True)

    def step(self, state: np.ndarray, message: np.ndarray, q_t: float = 1.0) -> np.ndarray:
        current = _checked_vector(state, self.state_dim, "state")
        msg = _checked_vector(message, self.target_dim, "message")
        return (self.a * current + self.g * np.float32(float(q_t)) * msg).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state_dim": int(self.state_dim),
            "target_dim": int(self.target_dim),
            "a": self.a.astype(np.float32, copy=True),
            "g": self.g.astype(np.float32, copy=True),
            "trainable": False,
        }


@dataclass
class RealDiagLearnableStateModel:
    name: str
    state_dim: int
    target_dim: int
    theta: np.ndarray
    gamma: np.ndarray
    gamma_min: float = float(GAMMA_MIN)
    gamma_max: float = float(GAMMA_MAX)
    gamma_init: str = "ema_tau"

    @classmethod
    def create(
        cls,
        state_dim: int,
        target_dim: int,
        state: dict[str, Any] | None = None,
        gamma_min: float = float(GAMMA_MIN),
        gamma_max: float = float(GAMMA_MAX),
    ) -> "RealDiagLearnableStateModel":
        if int(state_dim) != int(target_dim):
            raise ValueError("real_diag_learnable requires state_dim == target_dim")
        min_value = np.float32(float(gamma_min))
        max_value = np.float32(float(gamma_max))
        if not np.isfinite(min_value) or not np.isfinite(max_value) or min_value <= 0:
            raise ValueError("real_diag gamma bounds must be finite and positive")
        if max_value <= min_value:
            raise ValueError("real_diag gamma_max must be greater than gamma_min")
        if state is not None:
            theta = np.asarray(state["theta"], dtype=np.float32)
            gamma = np.asarray(state["gamma"], dtype=np.float32)
            min_value = np.float32(float(state.get("gamma_min", min_value)))
            max_value = np.float32(float(state.get("gamma_max", max_value)))
            gamma_init = str(state.get("gamma_init", "ema_tau"))
        else:
            gamma0 = np.float32(1.0) / _tau(int(state_dim))
            theta = _inverse_softplus(np.maximum(gamma0 - min_value, np.float32(1e-8)))
            gamma = np.clip(_softplus(theta) + min_value, min_value, max_value)
            gamma_init = "ema_tau"
        if theta.shape != (int(state_dim),) or gamma.shape != (int(state_dim),):
            raise ValueError("real_diag_learnable theta/gamma shape does not match state_dim")
        return cls(
            "real_diag_learnable",
            int(state_dim),
            int(target_dim),
            theta.astype(np.float32, copy=True),
            gamma.astype(np.float32, copy=True),
            float(min_value),
            float(max_value),
            gamma_init,
        )

    def read(self, state: np.ndarray) -> np.ndarray:
        return _checked_vector(state, self.state_dim, "state").astype(np.float32, copy=True)

    def step(self, state: np.ndarray, message: np.ndarray, q_t: float = 1.0) -> np.ndarray:
        current = _checked_vector(state, self.state_dim, "state")
        msg = _checked_vector(message, self.target_dim, "message")
        a = np.exp(-self.gamma).astype(np.float32)
        g = (np.float32(1.0) - a).astype(np.float32)
        return (a * current + g * np.float32(float(q_t)) * msg).astype(np.float32)

    def gamma_derivative(self) -> np.ndarray:
        return _sigmoid(self.theta).astype(np.float32, copy=False)

    def recompute_gamma(self) -> np.ndarray:
        self.gamma = np.clip(
            _softplus(self.theta) + np.float32(float(self.gamma_min)),
            np.float32(float(self.gamma_min)),
            np.float32(float(self.gamma_max)),
        ).astype(np.float32, copy=False)
        return self.gamma

    def update_theta(
        self,
        grad_theta: np.ndarray,
        lr: float,
        weight_decay: float = 0.0,
        clip: float = 1.0,
    ) -> float:
        grad = _checked_vector(grad_theta, self.state_dim, "grad_theta").astype(
            np.float32,
            copy=True,
        )
        if float(weight_decay) != 0.0:
            grad += np.float32(float(weight_decay)) * self.theta
        grad_norm = float(np.linalg.norm(grad))
        clip_value = float(clip)
        if clip_value > 0.0 and grad_norm > clip_value:
            grad *= np.float32(clip_value / max(grad_norm, 1e-12))
        self.theta = (self.theta - np.float32(float(lr)) * grad).astype(np.float32)
        self.recompute_gamma()
        return grad_norm

    def gamma_stats(self, initial_gamma: np.ndarray | None = None) -> dict[str, float | str]:
        stats: dict[str, float | str] = {
            "gamma_mean": float(np.mean(self.gamma)),
            "gamma_min": float(np.min(self.gamma)),
            "gamma_max": float(np.max(self.gamma)),
            "gamma_config_min": float(self.gamma_min),
            "gamma_config_max": float(self.gamma_max),
            "gamma_init": str(self.gamma_init),
        }
        if initial_gamma is not None:
            init = _checked_vector(initial_gamma, self.state_dim, "initial_gamma")
            stats["gamma_delta_l2"] = float(np.linalg.norm(self.gamma - init))
        return stats

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state_dim": int(self.state_dim),
            "target_dim": int(self.target_dim),
            "theta": self.theta.astype(np.float32, copy=True),
            "gamma": self.gamma.astype(np.float32, copy=True),
            "gamma_min": float(self.gamma_min),
            "gamma_max": float(self.gamma_max),
            "gamma_init": str(self.gamma_init),
            "trainable": True,
        }


@dataclass
class S4DComplexNodeStateModel:
    name: str
    state_dim: int
    target_dim: int
    log_A_real: np.ndarray
    A_imag: np.ndarray
    log_dt: np.ndarray
    B_in: np.ndarray
    C_read: np.ndarray

    @classmethod
    def create(
        cls,
        state_dim: int,
        target_dim: int,
        seed: int,
        state: dict[str, Any] | None = None,
    ) -> "S4DComplexNodeStateModel":
        if int(state_dim) % 2 != 0:
            raise ValueError("s4d_complex_node requires even state_dim")
        modes = int(state_dim) // 2
        if state is not None:
            return cls(
                "s4d_complex_node",
                int(state_dim),
                int(target_dim),
                np.asarray(state["log_A_real"], dtype=np.float32),
                np.asarray(state["A_imag"], dtype=np.float32),
                np.asarray(state["log_dt"], dtype=np.float32),
                np.asarray(state["B_in"], dtype=np.complex64),
                np.asarray(state["C_read"], dtype=np.complex64),
            )
        rng = np.random.default_rng(int(seed))
        log_A_real = np.full((modes,), np.log(0.5), dtype=np.float32)
        A_imag = (np.arange(modes, dtype=np.float32) * np.float32(np.pi)).astype(np.float32)
        log_dt = rng.uniform(
            np.log(0.001),
            np.log(0.1),
            size=(modes,),
        ).astype(np.float32)
        scale_b = 1.0 / np.sqrt(max(int(target_dim), 1))
        scale_c = 1.0 / np.sqrt(max(modes, 1))
        B_in = (
            rng.normal(0.0, scale_b, size=(modes, int(target_dim)))
            + 1j * rng.normal(0.0, scale_b, size=(modes, int(target_dim)))
        ).astype(np.complex64)
        C_read = (
            rng.normal(0.0, scale_c, size=(int(target_dim), modes))
            + 1j * rng.normal(0.0, scale_c, size=(int(target_dim), modes))
        ).astype(np.complex64)
        return cls(
            "s4d_complex_node",
            int(state_dim),
            int(target_dim),
            log_A_real,
            A_imag,
            log_dt,
            B_in,
            C_read,
        )

    def _complex_state(self, state: np.ndarray) -> np.ndarray:
        vector = _checked_vector(state, self.state_dim, "state")
        return (vector[0::2] + 1j * vector[1::2]).astype(np.complex64)

    def _real_state(self, state: np.ndarray) -> np.ndarray:
        out = np.empty((self.state_dim,), dtype=np.float32)
        out[0::2] = np.real(state).astype(np.float32)
        out[1::2] = np.imag(state).astype(np.float32)
        return out

    def read(self, state: np.ndarray) -> np.ndarray:
        x = self._complex_state(state)
        readout = np.float32(2.0) * np.real(self.C_read @ x)
        return readout.astype(np.float32)

    def step(self, state: np.ndarray, message: np.ndarray, q_t: float = 1.0) -> np.ndarray:
        x = self._complex_state(state)
        msg = _checked_vector(message, self.target_dim, "message")
        a_diag = (-np.exp(self.log_A_real) + 1j * self.A_imag).astype(np.complex64)
        dt = np.exp(self.log_dt).astype(np.float32)
        dt_a = (dt.astype(np.complex64) * a_diag).astype(np.complex64)
        abar = np.exp(dt_a).astype(np.complex64)
        phi = ((abar - np.complex64(1.0)) / a_diag).astype(np.complex64)
        b_t = (self.B_in @ msg.astype(np.float32)).astype(np.complex64)
        updated = abar * x + np.float32(float(q_t)) * phi * b_t
        return self._real_state(updated)

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state_dim": int(self.state_dim),
            "target_dim": int(self.target_dim),
            "log_A_real": self.log_A_real.astype(np.float32, copy=True),
            "A_imag": self.A_imag.astype(np.float32, copy=True),
            "log_dt": self.log_dt.astype(np.float32, copy=True),
            "B_in": self.B_in.astype(np.complex64, copy=True),
            "C_read": self.C_read.astype(np.complex64, copy=True),
            "trainable": False,
        }


def build_state_model(
    name: str,
    state_dim: int,
    target_dim: int,
    seed: int,
    state: dict[str, Any] | None = None,
    gamma_min: float = float(GAMMA_MIN),
    gamma_max: float = float(GAMMA_MAX),
) -> SSPMStateModel:
    """Build or restore an SSPM node state model by name."""
    key = str(name)
    if key == "ema_fixed":
        return EmaFixedStateModel.create(state_dim, target_dim)
    if key == "real_diag_learnable":
        return RealDiagLearnableStateModel.create(
            state_dim,
            target_dim,
            state,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
    if key == "s4d_complex_node":
        return S4DComplexNodeStateModel.create(state_dim, target_dim, seed, state)
    raise ValueError(f"unknown SSPM state_model: {name}")
