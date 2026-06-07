from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import math
import numpy as np

from legacy.baselines.semantic_sketch import SemanticSketch, bucket_rarity, update_bucket_count
from cs4m.utils.common import ENTITY_TYPES, RELATIONS, log1p_seconds, one_hot


@dataclass
class LowRankConfig:
    latent_dim: int = 32
    rank: int = 8
    semantic_buckets: int = 131072
    identity_buckets: int = 131072
    max_tokens: int = 96
    include_bigrams: bool = False
    vector_cache_size: int = 200000
    max_train_samples: int = 300000
    ridge_lambda: float = 1e-2
    direct_lowrank_enabled: bool = False
    direct_lowrank_iters: int = 5
    direct_lowrank_l2: float = 1e-2
    state_update_alpha: float = 0.35
    state_source_alpha: float = 0.10
    source_mix: float = 0.25
    reservoir_seed: int = 2026


class LowRankStreamModel:
    """Leakage-free low-rank latent predictor over a global provenance stream.

    The model predicts the current event semantic sketch z_t from causal state
    available before seeing z_t. The predictor is factorized as:

        z_hat = ((context - mean) / std @ A) @ B + bias

    where A is context_dim x rank and B is rank x latent_dim.
    """

    def __init__(self, config: LowRankConfig):
        self.config = config
        self.sketch = SemanticSketch(
            latent_dim=int(config.latent_dim),
            num_buckets=int(config.semantic_buckets),
            max_tokens=int(config.max_tokens),
            include_bigrams=bool(config.include_bigrams),
        )
        self.semantic_counts = self.sketch.empty_counts()
        self.counts = {
            "edge": np.zeros((int(config.identity_buckets),), dtype=np.int32),
            "src": np.zeros((int(config.identity_buckets),), dtype=np.int32),
            "dst": np.zeros((int(config.identity_buckets),), dtype=np.int32),
            "rel_dst": np.zeros((int(config.identity_buckets),), dtype=np.int32),
            "self": np.zeros((int(config.identity_buckets),), dtype=np.int32),
        }
        self.num_profile_events = 0
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.factor_a: np.ndarray | None = None
        self.factor_b: np.ndarray | None = None
        self.latent_basis: np.ndarray | None = None
        self.z_mean: np.ndarray | None = None
        self.semantic_basis: np.ndarray | None = None
        self.semantic_singular_values: np.ndarray | None = None
        self.bias: np.ndarray | None = None
        self.singular_values: np.ndarray | None = None
        self.predictor_training_mode = "ridge_svd"
        self._vector_cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}

    @property
    def context_dim(self) -> int:
        return int(self.config.latent_dim) * 3 + len(RELATIONS) + 2 * len(ENTITY_TYPES) + 8

    @property
    def parameter_count(self) -> int:
        total = int(self.semantic_counts.size)
        total += sum(int(arr.size) for arr in self.counts.values())
        for arr in (
            self.x_mean,
            self.x_std,
            self.factor_a,
            self.factor_b,
            self.latent_basis,
            self.z_mean,
            self.semantic_basis,
            self.semantic_singular_values,
            self.bias,
            self.singular_values,
        ):
            if arr is not None:
                total += int(arr.size)
        return int(total)

    @property
    def model_size_bytes(self) -> int:
        total = int(self.semantic_counts.nbytes)
        total += sum(int(arr.nbytes) for arr in self.counts.values())
        for arr in (
            self.x_mean,
            self.x_std,
            self.factor_a,
            self.factor_b,
            self.latent_basis,
            self.z_mean,
            self.semantic_basis,
            self.semantic_singular_values,
            self.bias,
            self.singular_values,
        ):
            if arr is not None:
                total += int(arr.nbytes)
        return int(total)

    def metadata(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "context_dim": int(self.context_dim),
            "latent_dim": int(self.config.latent_dim),
            "rank": int(self.config.rank),
            "parameter_count": int(self.parameter_count),
            "trainable_parameter_count": int(
                (0 if self.factor_a is None else self.factor_a.size)
                + (0 if self.factor_b is None else self.factor_b.size)
                + (0 if self.latent_basis is None else self.latent_basis.size)
                + (0 if self.semantic_basis is None else self.semantic_basis.size)
                + (0 if self.z_mean is None else self.z_mean.size)
                + (0 if self.bias is None else self.bias.size)
            ),
            "model_size_bytes": int(self.model_size_bytes),
            "model_size_mb": float(self.model_size_bytes / (1024.0 * 1024.0)),
            "num_profile_events": int(self.num_profile_events),
            "predictor_training_mode": str(self.predictor_training_mode),
            "direct_lowrank_enabled": bool(self.config.direct_lowrank_enabled),
            "direct_lowrank_iters": int(self.config.direct_lowrank_iters),
        }

    def _zero_state(self) -> np.ndarray:
        return np.zeros((int(self.config.latent_dim),), dtype=np.float32)

    def encode_event(self, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        key = (str(row["text"]), str(row["action"]), str(row["object_type"]))
        cached = self._vector_cache.get(key)
        if cached is not None:
            return cached
        z, semantic_buckets = self.sketch.encode(key[0], key[1], key[2])
        if int(self.config.vector_cache_size) > 0 and len(self._vector_cache) < int(self.config.vector_cache_size):
            self._vector_cache[key] = (z, semantic_buckets)
        return z, semantic_buckets

    def vectorize_event(self, row: dict[str, Any]) -> np.ndarray:
        return self.encode_event(row)[0]

    def _count_key_values(self, row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
        rel = int(row["relation_id"])
        return (
            f"E={int(row['info_src'])}|{rel}|{int(row['info_dst'])}",
            f"S={int(row['info_src'])}",
            f"D={int(row['info_dst'])}",
            f"RD={rel}|{int(row['info_dst_type'])}",
            f"N={row.get('src_kind', '')}|{row.get('src_summary', '')}",
            f"N={row.get('dst_kind', '')}|{row.get('dst_summary', '')}",
        )

    def update_profile_counts(self, row: dict[str, Any], semantic_buckets: np.ndarray | None = None) -> None:
        if semantic_buckets is None:
            semantic_buckets = self.encode_event(row)[1]
        self.sketch.update_count_buckets(
            self.semantic_counts,
            semantic_buckets,
        )
        edge_key, src_key, dst_key, rel_dst_key, src_self_key, dst_self_key = self._count_key_values(row)
        update_bucket_count(self.counts["edge"], edge_key)
        update_bucket_count(self.counts["src"], src_key)
        update_bucket_count(self.counts["dst"], dst_key)
        update_bucket_count(self.counts["rel_dst"], rel_dst_key)
        update_bucket_count(self.counts["self"], src_self_key)
        update_bucket_count(self.counts["self"], dst_self_key)
        self.num_profile_events += 1

    def rarity_components(self, row: dict[str, Any], semantic_buckets: np.ndarray | None = None) -> dict[str, float]:
        if semantic_buckets is None:
            semantic_buckets = self.encode_event(row)[1]
        edge_key, src_key, dst_key, rel_dst_key, src_self_key, dst_self_key = self._count_key_values(row)
        src_self = bucket_rarity(self.counts["self"], self.num_profile_events * 2, src_self_key)
        dst_self = bucket_rarity(self.counts["self"], self.num_profile_events * 2, dst_self_key)
        semantic_stats = self.sketch.rarity_stats_from_buckets(
            self.semantic_counts,
            self.num_profile_events,
            semantic_buckets,
        )
        return {
            "semantic_rarity": float(semantic_stats["mean"]),
            "semantic_max_rarity": float(semantic_stats["max"]),
            "semantic_top3_rarity": float(semantic_stats["top3_mean"]),
            "semantic_unseen_ratio": float(semantic_stats["unseen_ratio"]),
            "edge_rarity": float(bucket_rarity(self.counts["edge"], self.num_profile_events, edge_key)),
            "src_rarity": float(bucket_rarity(self.counts["src"], self.num_profile_events, src_key)),
            "dst_rarity": float(bucket_rarity(self.counts["dst"], self.num_profile_events, dst_key)),
            "rel_dst_rarity": float(bucket_rarity(self.counts["rel_dst"], self.num_profile_events, rel_dst_key)),
            "src_self_rarity": float(src_self),
            "dst_self_rarity": float(dst_self),
            "endpoint_self_rarity": float(max(src_self, dst_self)),
        }

    def make_context(
        self,
        row: dict[str, Any],
        states: dict[int, np.ndarray],
        last_seen: dict[int, int],
        last_edge_seen: dict[str, int],
        global_state: np.ndarray,
        last_global_ts: int,
        semantic_buckets: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        src = int(row["info_src"])
        dst = int(row["info_dst"])
        rel = int(row["relation_id"])
        ts = int(row["timestamp_ns"])
        src_state = states.get(src, self._zero_state())
        dst_state = states.get(dst, self._zero_state())
        edge_key = f"{src}|{rel}|{dst}"
        rarity = self.rarity_components(row, semantic_buckets=semantic_buckets)
        scalars = np.asarray(
            [
                log1p_seconds(ts, last_global_ts),
                log1p_seconds(ts, last_seen.get(src, -1)),
                log1p_seconds(ts, last_seen.get(dst, -1)),
                log1p_seconds(ts, last_edge_seen.get(edge_key, -1)),
                rarity["edge_rarity"],
                rarity["src_rarity"],
                rarity["dst_rarity"],
                rarity["rel_dst_rarity"],
            ],
            dtype=np.float32,
        )
        ctx = np.concatenate(
            [
                global_state.astype(np.float32, copy=False),
                src_state.astype(np.float32, copy=False),
                dst_state.astype(np.float32, copy=False),
                one_hot(rel, len(RELATIONS)),
                one_hot(int(row["info_src_type"]), len(ENTITY_TYPES)),
                one_hot(int(row["info_dst_type"]), len(ENTITY_TYPES)),
                scalars,
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        aux = {
            **rarity,
            "src_state_norm": float(np.linalg.norm(src_state)),
            "dst_state_norm": float(np.linalg.norm(dst_state)),
            "global_state_norm": float(np.linalg.norm(global_state)),
            "dt_global_log": float(scalars[0]),
            "dt_src_log": float(scalars[1]),
            "dt_dst_log": float(scalars[2]),
            "dt_edge_log": float(scalars[3]),
        }
        return ctx, aux

    def update_states(
        self,
        row: dict[str, Any],
        z: np.ndarray,
        states: dict[int, np.ndarray],
        last_seen: dict[int, int],
        last_edge_seen: dict[str, int],
        global_state: np.ndarray,
    ) -> np.ndarray:
        src = int(row["info_src"])
        dst = int(row["info_dst"])
        rel = int(row["relation_id"])
        ts = int(row["timestamp_ns"])
        src_state = states.get(src, self._zero_state())
        dst_state = states.get(dst, self._zero_state())
        message = (1.0 - float(self.config.source_mix)) * z + float(self.config.source_mix) * src_state
        alpha = float(self.config.state_update_alpha)
        beta = float(self.config.state_source_alpha)
        states[dst] = ((1.0 - alpha) * dst_state + alpha * message).astype(np.float32, copy=False)
        states[src] = ((1.0 - beta) * src_state + beta * z).astype(np.float32, copy=False)
        last_seen[src] = ts
        last_seen[dst] = ts
        last_edge_seen[f"{src}|{rel}|{dst}"] = ts
        return ((1.0 - beta) * global_state + beta * z).astype(np.float32, copy=False)

    def fit_rows(self, rows) -> dict[str, Any]:
        states: dict[int, np.ndarray] = {}
        last_seen: dict[int, int] = {}
        last_edge_seen: dict[str, int] = {}
        global_state = self._zero_state()
        last_global_ts = -1
        max_samples = int(self.config.max_train_samples)
        rng = np.random.default_rng(int(self.config.reservoir_seed))
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        seen = 0
        processed = 0
        for row in rows:
            z, semantic_buckets = self.encode_event(row)
            ctx, _aux = self.make_context(row, states, last_seen, last_edge_seen, global_state, last_global_ts, semantic_buckets)
            if max_samples <= 0 or len(xs) < max_samples:
                xs.append(ctx.copy())
                ys.append(z.copy())
            else:
                j = int(rng.integers(0, seen + 1))
                if j < max_samples:
                    xs[j] = ctx.copy()
                    ys[j] = z.copy()
            seen += 1
            self.update_profile_counts(row, semantic_buckets=semantic_buckets)
            global_state = self.update_states(row, z, states, last_seen, last_edge_seen, global_state)
            last_global_ts = int(row["timestamp_ns"])
            processed += 1
        if not xs:
            raise RuntimeError("No training samples were collected.")
        x = np.stack(xs, axis=0).astype(np.float32, copy=False)
        y = np.stack(ys, axis=0).astype(np.float32, copy=False)
        self.z_mean = np.mean(y, axis=0).astype(np.float32)
        yc = (y - self.z_mean).astype(np.float32, copy=False)
        _zu, zs, zvt = np.linalg.svd(yc.astype(np.float64), full_matrices=False)
        semantic_rank = min(int(self.config.rank), int(zs.shape[0]))
        self.semantic_basis = zvt[:semantic_rank, :].astype(np.float32)
        self.semantic_singular_values = zs[:semantic_rank].astype(np.float32)
        self.x_mean = np.mean(x, axis=0).astype(np.float32)
        self.x_std = (np.std(x, axis=0) + 1e-6).astype(np.float32)
        xn = (x - self.x_mean) / self.x_std
        if bool(self.config.direct_lowrank_enabled):
            rank = min(int(self.config.rank), int(xn.shape[1]), int(y.shape[1]), int(y.shape[0]))
            self.bias = self.z_mean.astype(np.float32, copy=True)
            self.factor_a, self.factor_b = self._fit_direct_factorized_predictor(
                xn.astype(np.float32, copy=False),
                yc.astype(np.float32, copy=False),
                rank,
            )
            self.latent_basis = (
                self.semantic_basis[:rank, :].astype(np.float32, copy=True)
                if self.semantic_basis is not None and self.semantic_basis.shape[0] >= rank
                else self.factor_b.astype(np.float32, copy=True)
            )
            self.singular_values = None
            self.predictor_training_mode = "direct_factorized_als"
            return {
                "processed_events": int(processed),
                "train_samples": int(len(xs)),
                "context_dim": int(self.context_dim),
                "latent_dim": int(self.config.latent_dim),
                "rank": int(rank),
                "num_profile_events": int(self.num_profile_events),
                "predictor_training_mode": str(self.predictor_training_mode),
            }
        xb = np.concatenate([xn, np.ones((xn.shape[0], 1), dtype=np.float32)], axis=1)
        reg = float(self.config.ridge_lambda) * np.eye(xb.shape[1], dtype=np.float64)
        reg[-1, -1] = 0.0
        xtx = xb.T.astype(np.float64) @ xb.astype(np.float64) + reg
        xty = xb.T.astype(np.float64) @ y.astype(np.float64)
        full_w = np.linalg.solve(xtx, xty).astype(np.float32)
        w = full_w[:-1, :]
        self.bias = full_w[-1, :].astype(np.float32)
        u, s, vt = np.linalg.svd(w.astype(np.float64), full_matrices=False)
        rank = min(int(self.config.rank), int(s.shape[0]))
        root_s = np.sqrt(s[:rank]).astype(np.float32)
        self.factor_a = (u[:, :rank].astype(np.float32) * root_s.reshape(1, -1)).astype(np.float32)
        self.factor_b = (root_s.reshape(-1, 1) * vt[:rank, :].astype(np.float32)).astype(np.float32)
        self.latent_basis = vt[:rank, :].astype(np.float32)
        self.singular_values = s[:rank].astype(np.float32)
        self.predictor_training_mode = "ridge_then_svd"
        return {
            "processed_events": int(processed),
            "train_samples": int(len(xs)),
            "context_dim": int(self.context_dim),
            "latent_dim": int(self.config.latent_dim),
            "rank": int(rank),
            "num_profile_events": int(self.num_profile_events),
            "predictor_training_mode": str(self.predictor_training_mode),
        }

    def _fit_direct_factorized_predictor(self, xn: np.ndarray, yc: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
        if rank <= 0:
            raise RuntimeError("Direct low-rank training requires rank > 0.")
        l2 = max(float(self.config.direct_lowrank_l2), 1e-8)
        rng = np.random.default_rng(int(self.config.reservoir_seed))
        if self.semantic_basis is not None and self.semantic_basis.shape[0] >= rank:
            b = self.semantic_basis[:rank, :].astype(np.float64, copy=True)
        else:
            b = rng.normal(0.0, 1.0 / math.sqrt(max(int(yc.shape[1]), 1)), size=(rank, yc.shape[1])).astype(np.float64)
        x64 = xn.astype(np.float64, copy=False)
        y64 = yc.astype(np.float64, copy=False)
        xtx = x64.T @ x64
        xty_cache = x64.T
        a = np.zeros((x64.shape[1], rank), dtype=np.float64)
        for _ in range(max(int(self.config.direct_lowrank_iters), 1)):
            bbt = b @ b.T + l2 * np.eye(rank, dtype=np.float64)
            target_rank = y64 @ b.T @ np.linalg.inv(bbt)
            a = np.linalg.solve(xtx + l2 * np.eye(xtx.shape[0], dtype=np.float64), xty_cache @ target_rank)
            h = x64 @ a
            hth = h.T @ h
            b = np.linalg.solve(hth + l2 * np.eye(rank, dtype=np.float64), h.T @ y64)
            scale = np.maximum(np.linalg.norm(a, axis=0), 1e-8)
            a = a / scale.reshape(1, -1)
            b = b * scale.reshape(-1, 1)
        return a.astype(np.float32), b.astype(np.float32)

    def predict(self, ctx: np.ndarray) -> np.ndarray:
        if self.x_mean is None or self.x_std is None or self.factor_a is None or self.factor_b is None or self.bias is None:
            raise RuntimeError("Model is not fitted.")
        xn = (ctx.astype(np.float32, copy=False) - self.x_mean) / self.x_std
        return ((xn @ self.factor_a) @ self.factor_b + self.bias).astype(np.float32, copy=False)

    def score_pre_update(
        self,
        row: dict[str, Any],
        states: dict[int, np.ndarray],
        last_seen: dict[int, int],
        last_edge_seen: dict[str, int],
        global_state: np.ndarray,
        last_global_ts: int,
    ) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
        z, semantic_buckets = self.encode_event(row)
        ctx, aux = self.make_context(row, states, last_seen, last_edge_seen, global_state, last_global_ts, semantic_buckets)
        pred = self.predict(ctx)
        pred_norm = float(np.linalg.norm(pred))
        z_norm = float(np.linalg.norm(z))
        aux["pred_mse"] = float(np.mean((pred - z) ** 2))
        aux["pred_cosine"] = float(1.0 - np.dot(pred, z) / max(pred_norm * z_norm, 1e-8))
        if self.z_mean is not None and self.semantic_basis is not None and self.semantic_basis.size > 0:
            centered = z - self.z_mean
            reconstructed = (centered @ self.semantic_basis.T) @ self.semantic_basis + self.z_mean
            aux["semantic_ae_residual"] = float(np.mean((z - reconstructed.astype(np.float32, copy=False)) ** 2))
        else:
            aux["semantic_ae_residual"] = 0.0
        src_state = states.get(int(row["info_src"]), self._zero_state())
        dst_state = states.get(int(row["info_dst"]), self._zero_state())

        def state_cosine_residual(state: np.ndarray) -> float:
            state_norm = float(np.linalg.norm(state))
            if state_norm <= 1e-8 or z_norm <= 1e-8:
                return 1.0
            return float(1.0 - np.dot(state, z) / max(state_norm * z_norm, 1e-8))

        aux["state_cosine"] = float(
            min(
                state_cosine_residual(src_state),
                state_cosine_residual(dst_state),
                state_cosine_residual(global_state),
            )
        )
        if self.latent_basis is not None and self.latent_basis.size > 0:
            projection = (z @ self.latent_basis.T) @ self.latent_basis
            aux["factor_residual"] = float(np.mean((z - projection.astype(np.float32, copy=False)) ** 2))
        else:
            aux["factor_residual"] = 0.0
        aux["pred_norm"] = pred_norm
        aux["latent_norm"] = z_norm
        return z, aux, pred
