from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Protocol, Sequence

import numpy as np

from cs4m.utils.common import stable_hash


UNK_TOKEN = "<UNK>"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:/-]+")


@dataclass
class ResidualEmbeddingConfig:
    """Historical residual embedding configuration for archived experiments."""

    method: str = "hash_sketch"
    latent_dim: int = 32
    max_tokens: int = 96
    doc2vec_window: int = 5
    doc2vec_min_count: int = 1
    doc2vec_dm: int = 1
    doc2vec_negative: int = 5
    doc2vec_epochs: int = 20
    doc2vec_workers: int = 4
    doc2vec_seed: int = 0
    doc2vec_infer_epochs: int = 20
    doc2vec_infer_alpha: float = 0.025


class ResidualEmbedder(Protocol):
    """Protocol for historical residual semantic embedders."""

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


class SemanticSketchSlim:
    """Historical signed feature-hashing text sketch for residual semantics."""

    def __init__(self, latent_dim: int = 32, max_tokens: int = 96) -> None:
        if int(latent_dim) <= 0:
            raise ValueError("latent_dim must be positive")
        if int(max_tokens) <= 0:
            raise ValueError("max_tokens must be positive")
        self.latent_dim = int(latent_dim)
        self.max_tokens = int(max_tokens)

    def encode(self, action: object, object_type: object, text: object = "") -> np.ndarray:
        """Encode action, object type, and residual text as an L2-normalized vector."""
        del action, object_type
        return self.encode_text(text)

    def encode_text(self, text: object = "") -> np.ndarray:
        """Encode residual text tokens only as an L2-normalized vector."""
        vector = np.zeros((self.latent_dim,), dtype=np.float32)
        for token in self.text_only_tokens(text):
            bucket = stable_hash(token, seed=0) % self.latent_dim
            sign = 1.0 if (stable_hash(token, seed=17) & 1) == 0 else -1.0
            vector[bucket] += np.float32(sign)
        return _normalize_vector(vector)

    def _tokens(self, action: object, object_type: object, text: object) -> Iterable[str]:
        del action, object_type
        return self.text_only_tokens(text)

    def text_only_tokens(self, text: object) -> Iterable[str]:
        """Yield natural residual text tokens without field-name prefixes."""
        yielded = 0
        for token in self._text_tokens(text):
            if yielded >= self.max_tokens:
                return
            yielded += 1
            yield token

    @staticmethod
    def _field_token(name: str, value: object) -> str:
        text = str(value).strip().lower() if value is not None else ""
        return f"{name}:{text or 'unknown'}"

    @staticmethod
    def _text_tokens(value: object) -> Iterable[str]:
        if value is None:
            return ()
        text = str(value).lower()
        return (match.group(0) for match in _TOKEN_RE.finditer(text))


class HashSketchResidualEmbedder:
    """Historical deterministic signed hash sketch residual embedder."""

    def __init__(self, config: ResidualEmbeddingConfig) -> None:
        self.config = config
        self.latent_dim = int(config.latent_dim)
        self.max_tokens = int(config.max_tokens)
        self._train_vocab: set[str] = set()
        self._train_sentences = 0

    def fit(self, token_sequences: Iterable[Sequence[str]]) -> None:
        """Record train-only vocabulary statistics."""
        self._train_vocab = set()
        self._train_sentences = 0
        for tokens in token_sequences:
            self._train_sentences += 1
            self._train_vocab.update(str(token) for token in tokens)

    def encode(self, tokens: Sequence[str]) -> np.ndarray:
        """Encode residual tokens using signed feature hashing."""
        vector = np.zeros((self.latent_dim,), dtype=np.float32)
        for index, token in enumerate(tokens):
            if index >= self.max_tokens:
                break
            text = str(token)
            bucket = stable_hash(text, seed=0) % self.latent_dim
            sign = 1.0 if (stable_hash(text, seed=17) & 1) == 0 else -1.0
            vector[bucket] += np.float32(sign)
        return _normalize_vector(vector)

    def stats(self) -> dict[str, object]:
        """Return train-only hash sketch statistics."""
        return {
            "method": "hash_sketch",
            "latent_dim": int(self.latent_dim),
            "max_tokens": int(self.max_tokens),
            "train_vocab_size": int(len(self._train_vocab)),
            "train_sentence_count": int(self._train_sentences),
        }

    def state_dict(self) -> dict[str, object]:
        """Return persistent hash sketch state."""
        return {
            "type": "hash_sketch",
            "config": asdict(self.config),
            "train_vocab": sorted(self._train_vocab),
            "train_sentence_count": int(self._train_sentences),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "HashSketchResidualEmbedder":
        """Restore a hash sketch embedder from persistent state."""
        config = ResidualEmbeddingConfig(**dict(state["config"]))
        embedder = cls(config)
        embedder._train_vocab = set(str(token) for token in state.get("train_vocab", []))
        embedder._train_sentences = int(state.get("train_sentence_count", 0))
        return embedder


class Doc2VecResidualEmbedder:
    """Historical train-only Doc2Vec residual semantic event embedder."""

    def __init__(self, config: ResidualEmbeddingConfig) -> None:
        self.config = config
        self.latent_dim = int(config.latent_dim)
        self.max_tokens = int(config.max_tokens)
        self.model = None
        self._train_vocab: set[str] = set()
        self._train_sentences = 0
        self._infer_count = 0
        self._empty_infer_count = 0

    def fit(self, token_sequences: Iterable[Sequence[str]]) -> None:
        """Train Doc2Vec using train-only residual token sequences."""
        try:
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("gensim is required for legacy Doc2Vec") from exc
        documents = []
        vocab: set[str] = set()
        for index, tokens in enumerate(token_sequences):
            sentence = [str(token) for token in tokens][: self.max_tokens]
            if not sentence:
                sentence = [UNK_TOKEN]
            documents.append(TaggedDocument(words=sentence, tags=[f"event_{index}"]))
            vocab.update(sentence)
        documents.append(TaggedDocument(words=[UNK_TOKEN], tags=["__unk__"]))
        vocab.add(UNK_TOKEN)
        self._train_vocab = vocab
        self._train_sentences = max(len(documents) - 1, 0)
        self.model = Doc2Vec(
            documents=documents,
            vector_size=int(self.latent_dim),
            window=int(self.config.doc2vec_window),
            min_count=int(self.config.doc2vec_min_count),
            dm=int(self.config.doc2vec_dm),
            negative=int(self.config.doc2vec_negative),
            epochs=int(self.config.doc2vec_epochs),
            workers=int(self.config.doc2vec_workers),
            seed=int(self.config.doc2vec_seed),
        )

    def encode(self, tokens: Sequence[str]) -> np.ndarray:
        """Infer one fixed event vector from residual tokens without updating the model."""
        if self.model is None:
            raise RuntimeError("Doc2VecResidualEmbedder.fit() must be called before encode()")
        sentence = [str(token) for token in tokens[: self.max_tokens] if str(token)]
        if not sentence:
            sentence = [UNK_TOKEN]
            self._empty_infer_count += 1
        self._infer_count += 1
        vector = self.model.infer_vector(
            sentence,
            alpha=float(self.config.doc2vec_infer_alpha),
            min_alpha=float(self.config.doc2vec_infer_alpha) * 0.01,
            epochs=int(self.config.doc2vec_infer_epochs),
        )
        return _normalize_vector(np.asarray(vector, dtype=np.float32))

    def stats(self) -> dict[str, object]:
        """Return Doc2Vec train and inference statistics."""
        return {
            "method": "doc2vec",
            "config": asdict(self.config),
            "latent_dim": int(self.latent_dim),
            "train_vocab_size": int(len(self._train_vocab)),
            "train_sentence_count": int(self._train_sentences),
            "infer_count": int(self._infer_count),
            "empty_infer_count": int(self._empty_infer_count),
        }

    def state_dict(self) -> dict[str, object]:
        """Return persistent Doc2Vec embedder state."""
        return {
            "type": "doc2vec",
            "config": asdict(self.config),
            "model": self.model,
            "train_vocab": sorted(self._train_vocab),
            "train_sentence_count": int(self._train_sentences),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "Doc2VecResidualEmbedder":
        """Restore a Doc2Vec embedder from persistent state."""
        config = ResidualEmbeddingConfig(**dict(state["config"]))
        embedder = cls(config)
        embedder.model = state["model"]
        embedder._train_vocab = set(str(token) for token in state.get("train_vocab", []))
        embedder._train_sentences = int(state.get("train_sentence_count", 0))
        return embedder


def build_residual_embedder(config: ResidualEmbeddingConfig) -> ResidualEmbedder:
    """Build the configured historical residual semantic embedder."""
    method = str(config.method).strip().lower()
    if method == "hash_sketch":
        return HashSketchResidualEmbedder(config)
    if method == "doc2vec":
        return Doc2VecResidualEmbedder(config)
    raise ValueError(f"unsupported legacy residual embedding method: {config.method}")


def residual_embedder_from_state_dict(state: dict[str, object]) -> ResidualEmbedder:
    """Restore a historical residual embedder from persistent state."""
    kind = str(state.get("type", "")).strip().lower()
    if kind == "hash_sketch":
        return HashSketchResidualEmbedder.from_state_dict(state)
    if kind == "doc2vec":
        return Doc2VecResidualEmbedder.from_state_dict(state)
    raise ValueError(f"unsupported legacy residual embedder state type: {kind}")
