from __future__ import annotations

import csv
import os
import resource
import time
from pathlib import Path
from typing import Any, Mapping


SSPM_PROFILING_FIELDS: tuple[str, ...] = (
    "event_idx",
    "active_nodes",
    "probationary_nodes",
    "promoted_nodes_total",
    "evicted_active_nodes_total",
    "evicted_probationary_nodes_total",
    "rss_mb",
    "state_memory_mb_est",
    "state_model",
    "state_dim",
    "target_dim",
    "rank",
    "context_dim",
    "context_mode",
    "context_action_mode",
    "global_context_mode",
    "update_gate_mode",
    "residual_score_mode",
    "residual_calibration",
    "calibration_fitted",
    "update_gate_threshold",
    "calibration_validation_events",
    "calibration_fallback_actions_count",
    "state_merge_mode",
    "num_physical_states",
    "num_clusters",
    "num_clustered_nodes",
    "compression_ratio",
    "num_copy_on_write",
    "avg_lambda_t",
    "avg_rho_t",
    "avg_q_t",
    "num_q_suppressed",
    "events_per_sec",
)


def current_rss_mb() -> float:
    """Return current process RSS in MiB on Linux-like systems."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1024.0 if os.name != "darwin" else 1024.0 * 1024.0
    return float(usage / scale)


class SSPMProfiler:
    """Append periodic SSPM memory/gate profiling rows to CSV."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._gate_count = 0
        self._lambda_sum = 0.0
        self._rho_sum = 0.0
        self._q_sum = 0.0
        self._q_suppressed = 0
        self._start = time.monotonic()
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SSPM_PROFILING_FIELDS)
            writer.writeheader()

    def observe_gates(self, lambda_t: float, rho_t: float, q_t: float) -> None:
        """Accumulate label-free gate statistics since process start."""
        self._gate_count += 1
        self._lambda_sum += float(lambda_t)
        self._rho_sum += float(rho_t)
        self._q_sum += float(q_t)
        if float(q_t) <= 0.0:
            self._q_suppressed += 1

    def write_row(
        self,
        event_idx: int,
        state_memory_stats: Mapping[str, Any],
        model_config: Mapping[str, Any],
        events_per_sec: float | None = None,
    ) -> None:
        """Append one profiling row with the fixed SSPM schema."""
        count = max(int(self._gate_count), 1)
        elapsed = max(time.monotonic() - self._start, 1e-9)
        target_dim = int(model_config.get("target_dim", model_config.get("latent_dim", 0)))
        state_dim = int(model_config.get("state_dim", target_dim))
        context_dim = int(model_config.get("context_dim", 2 * state_dim + 2 * 4 + 10))
        calibration = model_config.get("calibration", {})
        if not isinstance(calibration, Mapping):
            calibration = {}
        update_gate = model_config.get("update_gate", {})
        if not isinstance(update_gate, Mapping):
            update_gate = {}
        threshold = update_gate.get("threshold")
        effective_events_per_sec = (
            float(events_per_sec)
            if events_per_sec is not None
            else float(event_idx) / elapsed
        )
        row = {
            "event_idx": int(event_idx),
            "active_nodes": int(state_memory_stats.get("active_nodes", 0)),
            "probationary_nodes": int(state_memory_stats.get("probationary_nodes", 0)),
            "promoted_nodes_total": int(state_memory_stats.get("promoted_nodes_total", 0)),
            "evicted_active_nodes_total": int(
                state_memory_stats.get("evicted_active_nodes_total", 0),
            ),
            "evicted_probationary_nodes_total": int(
                state_memory_stats.get("evicted_probationary_nodes_total", 0),
            ),
            "rss_mb": f"{current_rss_mb():.6f}",
            "state_memory_mb_est": (
                f"{float(state_memory_stats.get('state_memory_mb_est', 0.0)):.6f}"
            ),
            "state_model": str(model_config.get("state_model", "")),
            "state_dim": state_dim,
            "target_dim": target_dim,
            "rank": int(model_config.get("rank", 0)),
            "context_dim": context_dim,
            "context_mode": str(model_config.get("context_mode", "")),
            "context_action_mode": str(model_config.get("context_action_mode", "")),
            "global_context_mode": str(model_config.get("global_context_mode", "")),
            "update_gate_mode": str(model_config.get("update_gate_mode", "")),
            "residual_score_mode": str(model_config.get("residual_score_mode", "")),
            "residual_calibration": str(model_config.get("residual_calibration", "")),
            "calibration_fitted": bool(calibration.get("calibration_fitted", False)),
            "update_gate_threshold": "" if threshold is None else f"{float(threshold):.6f}",
            "calibration_validation_events": int(
                calibration.get("calibration_num_validation_events", 0),
            ),
            "calibration_fallback_actions_count": int(
                calibration.get("fallback_actions_count", 0),
            ),
            "state_merge_mode": str(state_memory_stats.get("state_merge_mode", "none")),
            "num_physical_states": int(state_memory_stats.get("num_physical_states", 0)),
            "num_clusters": int(state_memory_stats.get("num_clusters", 0)),
            "num_clustered_nodes": int(state_memory_stats.get("num_clustered_nodes", 0)),
            "compression_ratio": (
                f"{float(state_memory_stats.get('compression_ratio', 0.0)):.6f}"
            ),
            "num_copy_on_write": int(
                state_memory_stats.get("num_copy_on_write_total", 0),
            ),
            "avg_lambda_t": f"{self._lambda_sum / count:.6f}",
            "avg_rho_t": f"{self._rho_sum / count:.6f}",
            "avg_q_t": f"{self._q_sum / count:.6f}",
            "num_q_suppressed": int(self._q_suppressed),
            "events_per_sec": f"{effective_events_per_sec:.6f}",
        }
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SSPM_PROFILING_FIELDS)
            writer.writerow(row)
