from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Iterable, Protocol, Sequence

import numpy as np


UNK_TOKEN = "<UNK>"


@dataclass
class ResidualEmbeddingConfig:
    """Configuration for residual semantic event embeddings."""

    method: str = "word2vec"
    latent_dim: int = 32
    max_tokens: int = 96
    word2vec_window: int = 3
    word2vec_min_count: int = 1
    word2vec_sg: int = 1
    word2vec_negative: int = 5
    word2vec_epochs: int = 10
    word2vec_workers: int = 4
    word2vec_seed: int = 0
    word2vec_oov_policy: str = "unk"


class ResidualEmbedder(Protocol):
    """Protocol for train-only residual semantic embedders."""

    latent_dim: int

    def fit(self, token_sequences: Iterable[Sequence[str]]) -> None:
        """Fit using train-only token sequences."""

    def encode(self, tokens: Sequence[str]) -> np.ndarray:
        """Encode one event token sequence as a fixed-size float32 vector."""

    def stats(self) -> dict[str, object]:
        """Return label-free train/transform statistics."""


def _normalize_vector(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    data = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(data))
    if norm <= float(eps):
        return np.zeros_like(data, dtype=np.float32)
    return (data / np.float32(norm)).astype(np.float32, copy=False)


class Word2VecResidualEmbedder:
    """Train-only Word2Vec residual semantic embedder."""

    def __init__(self, config: ResidualEmbeddingConfig) -> None:
        self.config = config
        self.latent_dim = int(config.latent_dim)
        self.max_tokens = int(config.max_tokens)
        self.model = None
        self._train_vocab: set[str] = set()
        self._train_sentences = 0
        self._oov_count = 0
        self._token_count = 0

    def fit(self, token_sequences: Iterable[Sequence[str]]) -> None:
        """Train Word2Vec using train-only residual token sequences."""
        try:
            from gensim.models import Word2Vec
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "gensim is required for semantic_embedding_method=word2vec",
            ) from exc
        corpus: list[list[str]] = []
        vocab: set[str] = set()
        for tokens in token_sequences:
            sentence = [str(token) for token in tokens][: self.max_tokens]
            if not sentence:
                sentence = [UNK_TOKEN]
            corpus.append(sentence)
            vocab.update(sentence)
        corpus.append([UNK_TOKEN])
        vocab.add(UNK_TOKEN)
        self._train_vocab = vocab
        self._train_sentences = len(corpus) - 1
        self.model = Word2Vec(
            sentences=corpus,
            vector_size=int(self.latent_dim),
            window=int(self.config.word2vec_window),
            min_count=int(self.config.word2vec_min_count),
            sg=int(self.config.word2vec_sg),
            negative=int(self.config.word2vec_negative),
            epochs=int(self.config.word2vec_epochs),
            workers=int(self.config.word2vec_workers),
            seed=int(self.config.word2vec_seed),
        )

    def encode(self, tokens: Sequence[str]) -> np.ndarray:
        """Encode one event as mean-pooled train-only Word2Vec vectors."""
        if self.model is None:
            raise RuntimeError("Word2VecResidualEmbedder.fit() must be called before encode()")
        vectors = []
        for index, token in enumerate(tokens):
            if index >= self.max_tokens:
                break
            text = str(token)
            self._token_count += 1
            if text in self.model.wv:
                vectors.append(self.model.wv[text])
            else:
                self._oov_count += 1
                if str(self.config.word2vec_oov_policy) == "zero":
                    vectors.append(np.zeros((self.latent_dim,), dtype=np.float32))
                else:
                    vectors.append(self.model.wv[UNK_TOKEN])
        if not vectors:
            return np.zeros((self.latent_dim,), dtype=np.float32)
        pooled = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
        return _normalize_vector(pooled)

    def stats(self) -> dict[str, object]:
        """Return Word2Vec train and transform statistics."""
        token_count = int(self._token_count)
        oov_count = int(self._oov_count)
        return {
            "method": "word2vec",
            "config": asdict(self.config),
            "latent_dim": int(self.latent_dim),
            "train_vocab_size": int(len(self._train_vocab)),
            "train_sentence_count": int(self._train_sentences),
            "oov_count": oov_count,
            "token_count": token_count,
            "oov_ratio": float(oov_count / token_count) if token_count else 0.0,
        }

    def state_dict(self) -> dict[str, object]:
        """Return persistent Word2Vec embedder state."""
        return {
            "type": "word2vec",
            "config": asdict(self.config),
            "model": self.model,
            "train_vocab": sorted(self._train_vocab),
            "train_sentence_count": int(self._train_sentences),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "Word2VecResidualEmbedder":
        """Restore a Word2Vec embedder from persistent state."""
        raw_config = dict(state["config"])
        allowed_config_keys = {field.name for field in fields(ResidualEmbeddingConfig)}
        config = ResidualEmbeddingConfig(
            **{key: value for key, value in raw_config.items() if key in allowed_config_keys},
        )
        embedder = cls(config)
        embedder.model = state["model"]
        embedder._train_vocab = set(str(token) for token in state.get("train_vocab", []))
        embedder._train_sentences = int(state.get("train_sentence_count", 0))
        return embedder


def build_residual_embedder(config: ResidualEmbeddingConfig) -> ResidualEmbedder:
    """Build the configured residual semantic embedder."""
    method = str(config.method).strip().lower()
    if method == "word2vec":
        return Word2VecResidualEmbedder(config)
    raise ValueError(f"unsupported residual embedding method: {config.method}")


def residual_embedder_from_state_dict(state: dict[str, object]) -> ResidualEmbedder:
    """Restore a residual embedder from persistent state."""
    kind = str(state.get("type", "")).strip().lower()
    if kind == "word2vec":
        return Word2VecResidualEmbedder.from_state_dict(state)
    raise ValueError(f"unsupported residual embedder state type: {kind}")
