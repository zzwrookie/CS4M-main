from __future__ import annotations

import math

import numpy as np

from cs4m.utils.common import stable_hash


def update_bucket_count(counts: np.ndarray, key: str, amount: int = 1) -> int:
    idx = int(stable_hash(str(key), seed=31) % int(counts.shape[0]))
    next_value = int(counts[idx]) + int(amount)
    counts[idx] = np.iinfo(counts.dtype).max if next_value > np.iinfo(counts.dtype).max else next_value
    return idx


def bucket_rarity(counts: np.ndarray, total: int, key: str) -> float:
    if int(total) <= 0 or counts.shape[0] <= 0:
        return 0.0
    idx = int(stable_hash(str(key), seed=31) % int(counts.shape[0]))
    count = int(counts[idx])
    prob = float(count + 1) / float(int(total) + int(counts.shape[0]))
    return float(-math.log(max(prob, 1e-12)))


class SemanticSketch:
    """Tiny feature-hashing semantic encoder.

    This is not the removed AE branch. It is the active lightweight semantic
    sketch used by the low-rank mainline and benign chain profile.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        num_buckets: int = 131072,
        max_tokens: int = 96,
        include_bigrams: bool = False,
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.num_buckets = int(num_buckets)
        self.max_tokens = int(max_tokens)
        self.include_bigrams = bool(include_bigrams)

    def empty_counts(self) -> np.ndarray:
        return np.zeros((self.num_buckets,), dtype=np.int32)

    def _tokens(self, text: str, action: str, object_type: str) -> list[str]:
        raw = [f"action:{action}", f"object:{object_type}"]
        raw.extend(str(text).replace("|", " ").replace("/", " ").split())
        tokens = [tok.strip().lower() for tok in raw if tok and tok.strip()]
        tokens = tokens[: max(int(self.max_tokens), 1)]
        if self.include_bigrams and len(tokens) > 1:
            tokens.extend([f"{tokens[idx]}__{tokens[idx + 1]}" for idx in range(len(tokens) - 1)])
        return tokens

    def encode(self, text: str, action: str, object_type: str) -> tuple[np.ndarray, np.ndarray]:
        z = np.zeros((self.latent_dim,), dtype=np.float32)
        bucket_ids: list[int] = []
        for token in self._tokens(text, action, object_type):
            h = int(stable_hash(token, seed=17))
            bucket_ids.append(int(h % self.num_buckets))
            dim = int(h % self.latent_dim)
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            z[dim] += sign
        norm = float(np.linalg.norm(z))
        if norm > 0.0:
            z /= norm
        return z, np.asarray(bucket_ids, dtype=np.int64)

    def update_count_buckets(self, counts: np.ndarray, bucket_ids: np.ndarray) -> None:
        for idx in np.asarray(bucket_ids, dtype=np.int64).tolist():
            if 0 <= int(idx) < int(counts.shape[0]):
                next_value = int(counts[int(idx)]) + 1
                counts[int(idx)] = np.iinfo(counts.dtype).max if next_value > np.iinfo(counts.dtype).max else next_value

    def rarity_stats_from_buckets(
        self,
        counts: np.ndarray,
        total_events: int,
        bucket_ids: np.ndarray,
    ) -> dict[str, float]:
        ids = np.asarray(bucket_ids, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < int(counts.shape[0]))]
        if ids.size <= 0 or int(total_events) <= 0:
            return {"mean": 0.0, "max": 0.0, "top3_mean": 0.0, "unseen_ratio": 0.0}
        denom = float(int(total_events) + int(counts.shape[0]))
        vals = np.asarray([-math.log(max(float(int(counts[int(idx)]) + 1) / denom, 1e-12)) for idx in ids.tolist()], dtype=np.float32)
        vals_sorted = np.sort(vals)
        top = vals_sorted[-min(3, vals_sorted.size) :]
        unseen = np.asarray([int(counts[int(idx)]) <= 0 for idx in ids.tolist()], dtype=bool)
        return {
            "mean": float(np.mean(vals)),
            "max": float(np.max(vals)),
            "top3_mean": float(np.mean(top)),
            "unseen_ratio": float(np.mean(unseen)),
        }
