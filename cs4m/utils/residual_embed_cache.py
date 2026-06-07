from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CACHE_VERSION = "residual_embedding_sqlite_v1"


def normalize_residual_tokens(tokens: Sequence[object]) -> tuple[str, ...]:
    """Normalize residual tokens before cache key construction."""
    return tuple(str(token) for token in tokens)


def residual_token_sequence_key(tokens: Sequence[object]) -> str:
    """Return a stable SHA256 key for a normalized residual token sequence."""
    normalized = normalize_residual_tokens(tokens)
    payload = json.dumps(normalized, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def residual_embedding_cache_fingerprint(
    *,
    embedder_path: str,
    embedder_dim: int,
    word2vec_window: int,
    z_dim: int,
    embedder_state_hash: str,
) -> str:
    """Return a stable fingerprint for residual embedding cache compatibility."""
    payload = {
        "cache_version": CACHE_VERSION,
        "embedder_path": str(embedder_path),
        "embedder_dim": int(embedder_dim),
        "word2vec_window": int(word2vec_window),
        "z_dim": int(z_dim),
        "embedder_state_hash": str(embedder_state_hash),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ResidualEmbeddingSqliteCache:
    """On-demand sqlite cache for residual embedding vectors."""

    def __init__(self, path: str | Path, fingerprint: str, z_dim: int) -> None:
        self.path = Path(path)
        self.fingerprint = str(fingerprint)
        self.z_dim = int(z_dim)
        self.eager_loaded_entries = 0
        self.hits = 0
        self.misses = 0
        self._closed = False
        self._conn = sqlite3.connect(str(self.path))
        self._initialize()

    def get_or_compute(self, tokens: Sequence[object], embedder: Any) -> np.ndarray:
        """Return cached z_t or compute, persist, and return a float32 z_t vector."""
        self._ensure_open()
        key = residual_token_sequence_key(tokens)
        row = self._conn.execute(
            "select z_blob from residual_embeddings where key = ?",
            (key,),
        ).fetchone()
        if row is not None:
            self.hits += 1
            return self._blob_to_vector(row[0])

        self.misses += 1
        normalized = normalize_residual_tokens(tokens)
        vector = np.asarray(embedder.encode(normalized), dtype=np.float32)
        if vector.shape != (self.z_dim,):
            raise ValueError(f"z_t dimension mismatch: expected {self.z_dim}, got {vector.shape}")
        self._conn.execute(
            """
            insert into residual_embeddings(key, token_count, z_blob)
            values (?, ?, ?)
            """,
            (key, len(normalized), sqlite3.Binary(vector.tobytes(order="C"))),
        )
        self._conn.commit()
        return vector.astype(np.float32, copy=False)

    def stats(self) -> dict[str, object]:
        """Return cache statistics without loading cached vectors into memory."""
        self._ensure_open()
        row = self._conn.execute("select count(*) from residual_embeddings").fetchone()
        entries = int(row[0]) if row is not None else 0
        return {
            "path": str(self.path),
            "entries": entries,
            "eager_loaded_entries": int(self.eager_loaded_entries),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "z_dim": int(self.z_dim),
            "fingerprint": self.fingerprint,
        }

    def close(self) -> None:
        """Close the sqlite connection."""
        if not self._closed:
            self._conn.close()
            self._closed = True

    def _initialize(self) -> None:
        self._conn.execute(
            """
            create table if not exists cache_metadata(
                name text primary key,
                value text not null
            )
            """,
        )
        self._conn.execute(
            """
            create table if not exists residual_embeddings(
                key text primary key,
                token_count integer not null,
                z_blob blob not null,
                created_at text not null default current_timestamp
            )
            """,
        )
        self._conn.commit()
        row = self._conn.execute(
            "select value from cache_metadata where name = 'fingerprint'",
        ).fetchone()
        if row is None:
            self._conn.execute(
                "insert into cache_metadata(name, value) values ('fingerprint', ?)",
                (self.fingerprint,),
            )
            self._conn.execute(
                "insert into cache_metadata(name, value) values ('z_dim', ?)",
                (str(self.z_dim),),
            )
            self._conn.commit()
            return
        if str(row[0]) != self.fingerprint:
            self.close()
            raise ValueError("fingerprint mismatch for residual embedding sqlite cache")
        z_dim_row = self._conn.execute(
            "select value from cache_metadata where name = 'z_dim'",
        ).fetchone()
        if z_dim_row is not None and int(z_dim_row[0]) != self.z_dim:
            self.close()
            raise ValueError("fingerprint mismatch for residual embedding sqlite cache")

    def _blob_to_vector(self, blob: bytes) -> np.ndarray:
        vector = np.frombuffer(blob, dtype=np.float32).copy()
        if vector.shape != (self.z_dim,):
            raise ValueError(
                f"cached z_t dimension mismatch: expected {self.z_dim}, got {vector.shape}",
            )
        return vector

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("residual embedding sqlite cache is closed")
