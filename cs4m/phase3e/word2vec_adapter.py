from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np


def _normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    data = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(data))
    if norm <= float(eps):
        return np.zeros_like(data, dtype=np.float32)
    return (data / np.float32(norm)).astype(np.float32, copy=False)


class ResidualWord2VecTokenAdapter:
    """Token-level access to a pretrained residual Word2Vec embedder."""

    def __init__(self, embedder: Any, model_path: str) -> None:
        if not hasattr(embedder, "model") or getattr(embedder, "model") is None:
            raise ValueError("residual_pretrained requires a fitted Word2VecResidualEmbedder")
        self.embedder = embedder
        self.model_path = str(model_path)
        self.model = embedder.model
        self.wv = self.model.wv
        self.dim = int(getattr(embedder, "latent_dim", getattr(self.wv, "vector_size")))
        if self.dim <= 0:
            raise ValueError("residual_pretrained Word2Vec vector dimension must be positive")

    def token_vector(self, token: object) -> np.ndarray:
        """Return the pretrained vector for `token`, or a zero vector for OOV."""
        text = str(token)
        if text in self.wv:
            return np.asarray(self.wv[text], dtype=np.float32)
        return np.zeros((self.dim,), dtype=np.float32)

    def encode_tokens(self, tokens: Sequence[object]) -> np.ndarray:
        """Encode tokens as the normalized mean of token vectors with OOV-zero policy."""
        vectors = [self.token_vector(token) for token in tokens]
        if not vectors:
            return np.zeros((self.dim,), dtype=np.float32)
        pooled = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
        return _normalize(pooled)

    def coverage(self, tokens: Sequence[object]) -> dict[str, int | float]:
        """Return JSON-safe known/OOV token coverage statistics."""
        total = len(tokens)
        known = sum(1 for token in tokens if str(token) in self.wv)
        oov = int(total - known)
        return {
            "total": int(total),
            "known": int(known),
            "oov": int(oov),
            "coverage": float(known / total) if total else 0.0,
        }


def word2vec_adapter_fingerprint(adapter: ResidualWord2VecTokenAdapter) -> str:
    """Return a stable JSON-safe SHA-256 fingerprint for an adapter's frozen vectors."""
    keys = [str(key) for key in adapter.wv.index_to_key]
    vectors = np.ascontiguousarray(np.asarray(adapter.wv.vectors, dtype=np.float32))
    keys_payload = json.dumps(keys, ensure_ascii=True, separators=(",", ":"))
    payload = {
        "source": "residual_pretrained",
        "dim": int(adapter.dim),
        "vocab_size": int(len(keys)),
        "keys_sha256": hashlib.sha256(keys_payload.encode("utf-8")).hexdigest(),
        "vectors_shape": [int(value) for value in vectors.shape],
        "vectors_sha256": hashlib.sha256(vectors.tobytes(order="C")).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
