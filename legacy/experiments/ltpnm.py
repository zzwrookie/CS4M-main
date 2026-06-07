"""Lightweight Temporal Provenance Normality Model primitives.

Purpose:
    Provide the P1/P2 implementation for the LTPNM branch:
    train-only event encoders, streaming order checks, validation-only
    threshold calibration helpers, and a bounded-memory temporal normality
    scorer.

Inputs:
    Chronological provenance events with timestamp, src/dst node ids,
    relation/action, object type, and node-summary text already read from the
    database by `legacy/experiments/run_ltpnm.py`.

Outputs:
    Encoded events, train-only vocab summaries, per-event nonconformity scores,
    online state profiles, and alert explanation components.

Enable/test:
    Run `legacy/experiments/run_ltpnm.py --phase p1` for encoder checks or
    `--phase p2` for the temporal normality scorer.

Runtime/memory:
    P1 state is train vocabulary only. P2 adds count tables plus bounded online
    node memories and edge keys; the runner reports RSS and state sizes.

Leakage risk:
    Vocabularies and count tables must be fit from train rows only. Validation
    scores may calibrate thresholds. Test rows must be scored before updating
    online state, and labels/ground truth must only be attached after scoring.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

import math
import re
import sys

import numpy as np

from cs4m.utils.common import log1p_seconds, stable_hash


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_HEX_RE = re.compile(r"\b[0-9a-f]{12,}\b", flags=re.IGNORECASE)
_NUM_RE = re.compile(r"\b\d+\b")


UNK = "<UNK>"


LTPNM_COMPONENT_NAMES = [
    "action_nll",
    "dst_type_nll",
    "relation_nll",
    "edge_transition_score",
    "coarse_semantic_nll",
    "raw_identity_nll",
    "memory_transition_residual",
]


def normalize_piece(text: object, max_len: int = 48) -> str:
    value = str(text).strip().lower()
    if not value:
        return "na"
    value = _IP_RE.sub(" ip ", value)
    value = _HEX_RE.sub(" hex ", value)
    value = _NUM_RE.sub(" num ", value)
    value = _NON_ALNUM_RE.sub("_", value).strip("_")
    if not value:
        return "na"
    return value[: int(max_len)].strip("_") or "na"


def tokenize_summary(text: object, max_tokens: int) -> list[str]:
    out: list[str] = []
    for raw in str(text).split():
        token = normalize_piece(raw, max_len=48)
        if token and token != "na":
            out.append(token)
        if len(out) >= int(max_tokens):
            break
    return out or ["na"]


def action_family(action: object) -> str:
    text = str(action).upper()
    if "READ" in text or "RECV" in text:
        return "read_recv"
    if "WRITE" in text or "SEND" in text:
        return "write_send"
    if "EXEC" in text or "CLONE" in text:
        return "execute_clone"
    if "CONNECT" in text:
        return "connect"
    if "OPEN" in text:
        return "open"
    return normalize_piece(text, max_len=32)


def ip_scope(token: str) -> str:
    text = str(token).lower()
    if "ip_127_" in text or "localhost" in text:
        return "ip_loopback"
    if "ip_10_" in text or "ip_192_168_" in text or "ip_172_" in text:
        return "ip_private"
    if text.startswith("ip_0") or text in {"ip", "ip_0"}:
        return "ip_zero"
    if text.startswith("ip_"):
        return "ip_public"
    return "ip_unknown"


def port_bucket(token: str) -> str:
    digits = "".join(ch for ch in str(token).lower() if ch.isdigit())
    if not digits:
        return "port_na"
    port = int(digits)
    if port <= 0:
        return "port_0"
    if port < 1024:
        return "port_system"
    if port < 49152:
        return "port_registered"
    return "port_ephemeral"


def coarse_role(kind: str, tokens: list[str]) -> str:
    node_kind = normalize_piece(kind, max_len=24)
    if node_kind == "netflow":
        ip = next((tok for tok in tokens if tok.startswith("ip")), "ip_unknown")
        port = next((tok for tok in tokens if tok.startswith("port")), "port_na")
        return f"netflow|{ip_scope(ip)}|{port_bucket(port)}"
    if node_kind == "file":
        root = next((tok for tok in tokens if tok.startswith("root_")), "")
        directory = next((tok for tok in tokens if tok.startswith("dir_")), "")
        return "|".join(tok for tok in ["file", root, directory] if tok) or "file|coarse_path"
    if node_kind == "process":
        directory = next((tok for tok in tokens if tok.startswith("dir_")), "")
        return f"process|{directory or 'coarse_process'}"
    return f"{node_kind}|coarse"


def raw_role(kind: str, tokens: list[str]) -> str:
    node_kind = normalize_piece(kind, max_len=24)
    if node_kind == "netflow":
        return coarse_role(node_kind, tokens)
    if node_kind == "file":
        root = next((tok for tok in tokens if tok.startswith("root_")), "")
        directory = next((tok for tok in tokens if tok.startswith("dir_")), "")
        path = next((tok for tok in tokens if tok.startswith("path_")), tokens[0] if tokens else "path_na")
        return "|".join(tok for tok in ["file", root, directory, path] if tok)
    if node_kind == "process":
        exe = next((tok for tok in tokens if tok.startswith("exe_")), tokens[0] if tokens else "proc_na")
        directory = next((tok for tok in tokens if tok.startswith("dir_")), "")
        return "|".join(tok for tok in ["process", exe, directory] if tok)
    return f"{node_kind}|{tokens[0] if tokens else 'na'}"


@dataclass
class EncodedEvent:
    event_pos: int
    timestamp_ns: int
    src_idx: int
    dst_idx: int
    info_src: int
    info_dst: int
    action_id: int
    action_unk: bool
    action_family_id: int
    action_family_unk: bool
    relation_id: int
    src_type_id: int
    dst_type_id: int
    object_type_id: int
    object_type_unk: bool
    coarse_src_id: int
    coarse_src_unk: bool
    coarse_dst_id: int
    coarse_dst_unk: bool
    raw_src_id: int
    raw_src_unk: bool
    raw_dst_id: int
    raw_dst_unk: bool
    coarse_src: str
    coarse_dst: str
    raw_src: str
    raw_dst: str
    action: str
    action_family: str
    object_type: str
    src_kind: str
    dst_kind: str


class FrozenVocab:
    def __init__(self, name: str, max_size: int, min_count: int = 1) -> None:
        self.name = str(name)
        self.max_size = max(int(max_size), 1)
        self.min_count = max(int(min_count), 1)
        self.counter: Counter[str] = Counter()
        self.token_to_id: dict[str, int] = {UNK: 0}
        self.frozen = False

    def observe(self, token: str) -> None:
        if self.frozen:
            raise RuntimeError(f"Cannot observe token after vocab is frozen: {self.name}")
        self.counter[str(token)] += 1

    def freeze(self) -> None:
        items = [
            (token, count)
            for token, count in self.counter.items()
            if int(count) >= int(self.min_count) and token != UNK
        ]
        items.sort(key=lambda kv: (-int(kv[1]), str(kv[0])))
        for token, _count in items[: max(int(self.max_size) - 1, 0)]:
            self.token_to_id[str(token)] = len(self.token_to_id)
        self.frozen = True

    def encode(self, token: str) -> tuple[int, bool]:
        if not self.frozen:
            raise RuntimeError(f"Vocab must be frozen before encode: {self.name}")
        idx = self.token_to_id.get(str(token), 0)
        return int(idx), bool(idx == 0 and str(token) != UNK)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": int(len(self.token_to_id)),
            "max_size": int(self.max_size),
            "min_count": int(self.min_count),
            "observed_unique_train": int(len(self.counter)),
            "top_train_tokens": [
                {"token": str(token), "count": int(count)}
                for token, count in self.counter.most_common(20)
            ],
        }

    def approx_bytes(self) -> int:
        total = sys.getsizeof(self.token_to_id) + sys.getsizeof(self.counter)
        for token in self.token_to_id:
            total += sys.getsizeof(token) + sys.getsizeof(int(self.token_to_id[token]))
        for token, count in self.counter.items():
            total += sys.getsizeof(token) + sys.getsizeof(int(count))
        return int(total)


@dataclass
class LTPNMEncoderConfig:
    max_action_vocab: int = 256
    max_action_family_vocab: int = 64
    max_object_type_vocab: int = 64
    max_coarse_vocab: int = 4096
    max_raw_vocab: int = 8192
    min_count: int = 1
    max_tokens_per_node: int = 12


@dataclass
class StreamCheck:
    processed: int = 0
    timestamp_order_violations: int = 0
    previous_ts: int = -1
    unknown_counts: Counter[str] = field(default_factory=Counter)
    first_events: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, event: EncodedEvent) -> None:
        if self.previous_ts > int(event.timestamp_ns):
            self.timestamp_order_violations += 1
        self.previous_ts = int(event.timestamp_ns)
        self.processed += 1
        for name in [
            "action",
            "action_family",
            "object_type",
            "coarse_src",
            "coarse_dst",
            "raw_src",
            "raw_dst",
        ]:
            if bool(getattr(event, f"{name}_unk", False)):
                self.unknown_counts[name] += 1
        if len(self.first_events) < 5:
            self.first_events.append(
                {
                    "event_pos": int(event.event_pos),
                    "timestamp_ns": int(event.timestamp_ns),
                    "src_idx": int(event.src_idx),
                    "dst_idx": int(event.dst_idx),
                    "action": event.action,
                    "action_id": int(event.action_id),
                    "relation_id": int(event.relation_id),
                    "coarse_src": event.coarse_src,
                    "coarse_src_id": int(event.coarse_src_id),
                    "coarse_dst": event.coarse_dst,
                    "coarse_dst_id": int(event.coarse_dst_id),
                    "raw_src": event.raw_src,
                    "raw_src_id": int(event.raw_src_id),
                    "raw_dst": event.raw_dst,
                    "raw_dst_id": int(event.raw_dst_id),
                }
            )

    def summary(self) -> dict[str, Any]:
        denom = max(int(self.processed), 1)
        return {
            "processed": int(self.processed),
            "timestamp_order_violations": int(self.timestamp_order_violations),
            "unknown_counts": {key: int(value) for key, value in sorted(self.unknown_counts.items())},
            "unknown_rates": {
                key: float(value / denom)
                for key, value in sorted(self.unknown_counts.items())
            },
            "first_events": list(self.first_events),
        }


class LTPNMEventEncoder:
    """Train-only categorical encoder for streaming provenance events.

    Purpose:
      P1 data interface for LTPNM. It builds categorical vocabularies on train
      rows only, then encodes validation/test rows with deterministic UNK.

    Leakage:
      `observe_train` must only be called on train rows. After `freeze`,
      validation/test calls to `encode` never add tokens or update statistics.
    """

    def __init__(self, config: LTPNMEncoderConfig | None = None) -> None:
        self.config = config or LTPNMEncoderConfig()
        self.action = FrozenVocab("action", self.config.max_action_vocab, self.config.min_count)
        self.action_family = FrozenVocab("action_family", self.config.max_action_family_vocab, self.config.min_count)
        self.object_type = FrozenVocab("object_type", self.config.max_object_type_vocab, self.config.min_count)
        self.coarse = FrozenVocab("coarse_semantic", self.config.max_coarse_vocab, self.config.min_count)
        self.raw = FrozenVocab("raw_semantic", self.config.max_raw_vocab, self.config.min_count)
        self.train_events = 0
        self.frozen = False

    def _fields(self, row: dict[str, Any]) -> dict[str, str]:
        src_tokens = tokenize_summary(row.get("src_summary", ""), int(self.config.max_tokens_per_node))
        dst_tokens = tokenize_summary(row.get("dst_summary", ""), int(self.config.max_tokens_per_node))
        src_kind = normalize_piece(row.get("src_kind", "unknown"), max_len=24)
        dst_kind = normalize_piece(row.get("dst_kind", "unknown"), max_len=24)
        action = normalize_piece(row.get("action", "EVENT_UNKNOWN"), max_len=40)
        object_type = normalize_piece(row.get("object_type", "unknown"), max_len=24)
        return {
            "action": action,
            "action_family": action_family(action),
            "object_type": object_type,
            "src_kind": src_kind,
            "dst_kind": dst_kind,
            "coarse_src": coarse_role(src_kind, src_tokens),
            "coarse_dst": coarse_role(dst_kind, dst_tokens),
            "raw_src": raw_role(src_kind, src_tokens),
            "raw_dst": raw_role(dst_kind, dst_tokens),
        }

    def observe_train(self, row: dict[str, Any]) -> None:
        if self.frozen:
            raise RuntimeError("Cannot observe train rows after freeze")
        fields = self._fields(row)
        self.action.observe(fields["action"])
        self.action_family.observe(fields["action_family"])
        self.object_type.observe(fields["object_type"])
        self.coarse.observe(fields["coarse_src"])
        self.coarse.observe(fields["coarse_dst"])
        self.raw.observe(fields["raw_src"])
        self.raw.observe(fields["raw_dst"])
        self.train_events += 1

    def freeze(self) -> None:
        for vocab in self.vocabs():
            vocab.freeze()
        self.frozen = True

    def vocabs(self) -> list[FrozenVocab]:
        return [self.action, self.action_family, self.object_type, self.coarse, self.raw]

    def encode(self, row: dict[str, Any]) -> EncodedEvent:
        if not self.frozen:
            raise RuntimeError("Encoder must be frozen before encoding")
        fields = self._fields(row)
        action_id, action_unk = self.action.encode(fields["action"])
        family_id, family_unk = self.action_family.encode(fields["action_family"])
        object_id, object_unk = self.object_type.encode(fields["object_type"])
        coarse_src_id, coarse_src_unk = self.coarse.encode(fields["coarse_src"])
        coarse_dst_id, coarse_dst_unk = self.coarse.encode(fields["coarse_dst"])
        raw_src_id, raw_src_unk = self.raw.encode(fields["raw_src"])
        raw_dst_id, raw_dst_unk = self.raw.encode(fields["raw_dst"])
        return EncodedEvent(
            event_pos=int(row.get("event_index", row.get("pos", 0))),
            timestamp_ns=int(row["timestamp_ns"]),
            src_idx=int(row["src_idx"]),
            dst_idx=int(row["dst_idx"]),
            info_src=int(row["info_src"]),
            info_dst=int(row["info_dst"]),
            action_id=action_id,
            action_unk=action_unk,
            action_family_id=family_id,
            action_family_unk=family_unk,
            relation_id=int(row["relation_id"]),
            src_type_id=int(row["info_src_type"]),
            dst_type_id=int(row["info_dst_type"]),
            object_type_id=object_id,
            object_type_unk=object_unk,
            coarse_src_id=coarse_src_id,
            coarse_src_unk=coarse_src_unk,
            coarse_dst_id=coarse_dst_id,
            coarse_dst_unk=coarse_dst_unk,
            raw_src_id=raw_src_id,
            raw_src_unk=raw_src_unk,
            raw_dst_id=raw_dst_id,
            raw_dst_unk=raw_dst_unk,
            coarse_src=fields["coarse_src"],
            coarse_dst=fields["coarse_dst"],
            raw_src=fields["raw_src"],
            raw_dst=fields["raw_dst"],
            action=fields["action"],
            action_family=fields["action_family"],
            object_type=fields["object_type"],
            src_kind=fields["src_kind"],
            dst_kind=fields["dst_kind"],
        )

    def stream_check(self, rows: Iterable[dict[str, Any]]) -> StreamCheck:
        check = StreamCheck()
        for row in rows:
            check.observe(self.encode(row))
        return check

    def summary(self) -> dict[str, Any]:
        vocab_summaries = {vocab.name: vocab.summary() for vocab in self.vocabs()}
        approx_bytes = int(sum(vocab.approx_bytes() for vocab in self.vocabs()))
        return {
            "purpose": "P1 train-only categorical encoder for LTPNM streaming events",
            "train_events": int(self.train_events),
            "frozen": bool(self.frozen),
            "config": {
                "max_action_vocab": int(self.config.max_action_vocab),
                "max_action_family_vocab": int(self.config.max_action_family_vocab),
                "max_object_type_vocab": int(self.config.max_object_type_vocab),
                "max_coarse_vocab": int(self.config.max_coarse_vocab),
                "max_raw_vocab": int(self.config.max_raw_vocab),
                "min_count": int(self.config.min_count),
                "max_tokens_per_node": int(self.config.max_tokens_per_node),
            },
            "vocabs": vocab_summaries,
            "approx_vocab_state_bytes": approx_bytes,
            "approx_vocab_state_mb": float(approx_bytes / (1024.0 * 1024.0)),
            "leakage_check": (
                "Vocabularies are fit only by observe_train before freeze. "
                "Validation/test encode uses deterministic UNK and does not mutate vocabs."
            ),
        }


def expected_alert_threshold(scores: list[float], budget: int, horizon_events: int, default_quantile: float = 0.999) -> dict[str, Any]:
    data = np.asarray([float(x) for x in scores if math.isfinite(float(x))], dtype=np.float32)
    if data.size <= 0:
        return {"threshold": float("inf"), "source": "empty_validation_scores", "validation_samples": 0}
    if int(budget) > 0 and int(horizon_events) > 0:
        tail_rate = min(max(float(budget) / max(float(horizon_events), 1.0), 0.0), 1.0)
        quantile = max(0.0, min(1.0, 1.0 - tail_rate))
        source = "validation_expected_alert_budget"
    else:
        quantile = float(default_quantile)
        source = "validation_quantile"
    return {
        "threshold": float(np.quantile(data, quantile)),
        "source": source,
        "quantile": float(quantile),
        "expected_alert_budget": int(budget),
        "expected_alert_horizon_events": int(horizon_events),
        "validation_samples": int(data.size),
    }


def component_top(scores: dict[str, float]) -> str:
    if not scores:
        return ""
    return max(scores.items(), key=lambda kv: (float(kv[1]), str(kv[0])))[0]


def component_vector(detail: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(detail.get(name, 0.0)) for name in LTPNM_COMPONENT_NAMES], dtype=np.float32)


def _bucket_log1p(value: float, cap: float = 16.0) -> float:
    return float(min(math.log1p(max(float(value), 0.0)), float(cap)) / max(float(cap), 1e-12))


def _counter_nll(counter: Counter[int], value: int, value_space: int, smoothing: float) -> float:
    total = int(sum(counter.values()))
    denom = float(total) + float(smoothing) * float(max(int(value_space), 1) + 1)
    prob = (float(counter.get(int(value), 0)) + float(smoothing)) / max(denom, 1e-12)
    return float(-math.log(max(prob, 1e-12)))


@dataclass
class LTPNMReliabilityConfig:
    """Configuration for the experimental component reliability gate.

    Purpose:
      Learn a small train-only component reliability function, while leaving
      validation for threshold calibration. It decides when each LTPNM
      component should be trusted and is disabled unless the runner uses
      `--score_mode rcg`.

    Leakage:
      The gate may be fit only from train component traces and train-only
      synthetic native anomalies. Validation is reserved for threshold
      calibration by the runner. It never reads test labels,
      attack windows, attack-specific tokens, or full-test statistics.
    """

    enabled: bool = False
    train_gate: bool = False
    epochs: int = 4
    learning_rate: float = 0.08
    l2: float = 0.001
    synthetic_per_event: int = 2
    max_trace_events: int = 200000
    edge_only_penalty: float = 0.35
    q_high: float = 0.95
    q_std_floor: float = 1e-3
    uniform_prior_weight: float = 0.35
    seed: int = 13


class LTPNMReliabilityCalibrator:
    """Tiny component reliability calibrator for LTPNM_RCG.

    The model is a bounded logistic reliability gate:

    `score_t = sum_i reliability_i(x_t) * standardized_component_i(t)`.

    It keeps alerts explainable because each event still exposes raw component
    values, learned component weights, and weighted contributions.
    """

    def __init__(self, config: LTPNMReliabilityConfig | None = None) -> None:
        self.config = config or LTPNMReliabilityConfig()
        self.component_names = list(LTPNM_COMPONENT_NAMES)
        self.feature_names = self._feature_names()
        self.mean = np.zeros((len(self.component_names),), dtype=np.float32)
        self.scale = np.ones((len(self.component_names),), dtype=np.float32)
        self.q_high = np.ones((len(self.component_names),), dtype=np.float32)
        self.weights = np.zeros((len(self.component_names), len(self.feature_names)), dtype=np.float32)
        self.bias = np.zeros((len(self.component_names),), dtype=np.float32)
        self.trained = False
        self.training_summary: dict[str, Any] = {}

    @staticmethod
    def _feature_names() -> list[str]:
        names = [f"z_{name}" for name in LTPNM_COMPONENT_NAMES]
        names.extend(
            [
                "edge_train_seen_log",
                "edge_online_seen_log",
                "dt_src_log_norm",
                "dt_dst_log_norm",
                "component_agreement",
                "edge_only_flag",
                "bias_context",
            ]
        )
        return names

    def fit_component_stats(self, component_rows: list[np.ndarray]) -> None:
        if not component_rows:
            self.mean = np.zeros((len(self.component_names),), dtype=np.float32)
            self.scale = np.ones((len(self.component_names),), dtype=np.float32)
            self.q_high = np.ones((len(self.component_names),), dtype=np.float32)
            return
        matrix = np.stack(component_rows, axis=0).astype(np.float32, copy=False)
        self.mean = np.median(matrix, axis=0).astype(np.float32, copy=False)
        q25 = np.quantile(matrix, 0.25, axis=0).astype(np.float32, copy=False)
        q75 = np.quantile(matrix, 0.75, axis=0).astype(np.float32, copy=False)
        robust = (q75 - q25) / np.float32(1.349)
        std = np.std(matrix, axis=0).astype(np.float32, copy=False)
        floor = np.float32(max(float(self.config.q_std_floor), 1e-6))
        self.scale = np.maximum(np.maximum(robust, std * np.float32(0.25)), floor).astype(np.float32, copy=False)
        self.q_high = np.quantile(matrix, float(self.config.q_high), axis=0).astype(np.float32, copy=False)

    def standardize(self, detail: dict[str, Any] | np.ndarray) -> np.ndarray:
        values = component_vector(detail) if isinstance(detail, dict) else np.asarray(detail, dtype=np.float32)
        z = (values.astype(np.float32, copy=False) - self.mean) / self.scale
        return np.maximum(z, np.float32(0.0)).astype(np.float32, copy=False)

    def features_from_detail(self, detail: dict[str, Any]) -> np.ndarray:
        z = self.standardize(detail)
        high = component_vector(detail) >= self.q_high
        agreement = float(np.sum(high)) / max(float(len(self.component_names)), 1.0)
        edge_idx = self.component_names.index("edge_transition_score")
        edge_only = bool(high[edge_idx] and int(np.sum(high)) == 1)
        extra = np.asarray(
            [
                _bucket_log1p(float(detail.get("train_edge_seen", 0.0))),
                _bucket_log1p(float(detail.get("online_edge_seen", 0.0))),
                _bucket_log1p(float(detail.get("dt_src_log", 0.0)), cap=4.0),
                _bucket_log1p(float(detail.get("dt_dst_log", 0.0)), cap=4.0),
                float(agreement),
                1.0 if edge_only else 0.0,
                1.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([z, extra], axis=0).astype(np.float32, copy=False)

    def _sigmoid(self, logits: np.ndarray) -> np.ndarray:
        clipped = np.clip(logits.astype(np.float32, copy=False), -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)

    def weights_for_features(self, features: np.ndarray) -> np.ndarray:
        if not self.trained:
            return np.full((len(self.component_names),), 1.0, dtype=np.float32)
        logits = self.weights @ features.astype(np.float32, copy=False) + self.bias
        learned = self._sigmoid(logits)
        floor = min(max(float(self.config.uniform_prior_weight), 0.0), 0.95)
        return (np.float32(floor) + np.float32(1.0 - floor) * learned).astype(np.float32, copy=False)

    def score_detail(self, detail: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        z = self.standardize(detail)
        features = self.features_from_detail(detail)
        weights = self.weights_for_features(features)
        if not self.trained:
            contributions = component_vector(detail)
            score = float(np.sum(contributions))
            reliability = np.ones_like(contributions, dtype=np.float32)
        else:
            contributions = weights * z
            score = float(np.sum(contributions))
            reliability = weights
        contrib = {name: float(contributions[idx]) for idx, name in enumerate(self.component_names)}
        weight_map = {f"{name}_weight": float(reliability[idx]) for idx, name in enumerate(self.component_names)}
        z_map = {f"{name}_z": float(z[idx]) for idx, name in enumerate(self.component_names)}
        edge_idx = self.component_names.index("edge_transition_score")
        top_idx = int(np.argmax(contributions)) if contributions.size else -1
        high = component_vector(detail) >= self.q_high
        edge_only = bool(high[edge_idx] and int(np.sum(high)) == 1)
        reason = "uniform_untrained"
        if self.trained:
            if edge_only and float(reliability[edge_idx]) <= float(np.mean(reliability)):
                reason = "edge_only_downweighted"
            elif int(np.sum(high)) >= 2:
                reason = "multi_component_supported"
            elif float(detail.get("online_edge_seen", 0.0)) > 0.0:
                reason = "repeated_online_context"
            else:
                reason = "learned_component_reliability"
        return score, {
            "rcg_score": float(score),
            "rcg_trained": int(self.trained),
            "rcg_top_component": self.component_names[top_idx] if top_idx >= 0 else "",
            "rcg_edge_reliability": float(reliability[edge_idx]) if reliability.size else 0.0,
            "rcg_component_agreement": float(features[self.feature_names.index("component_agreement")]),
            "rcg_edge_only_flag": int(edge_only),
            "rcg_reliability_reason": reason,
            "rcg_contributions": contrib,
            **weight_map,
            **z_map,
        }

    def _synthetic_detail(self, detail: dict[str, Any], rng: np.random.Generator, variant: int) -> dict[str, Any]:
        out = dict(detail)
        edge_name = "edge_transition_score"
        sem_name = "coarse_semantic_nll"
        rel_name = "relation_nll"
        raw_name = "raw_identity_nll"
        mem_name = "memory_transition_residual"
        q = {name: float(self.q_high[idx]) for idx, name in enumerate(self.component_names)}
        if variant % 4 == 0:
            out[edge_name] = max(float(out.get(edge_name, 0.0)), q[edge_name] * 1.15)
            out[sem_name] = max(float(out.get(sem_name, 0.0)), q[sem_name] * 1.05)
            out["train_edge_seen"] = 0
            out["online_edge_seen"] = 0
        elif variant % 4 == 1:
            out[rel_name] = max(float(out.get(rel_name, 0.0)), q[rel_name] * 1.20)
            out[edge_name] = max(float(out.get(edge_name, 0.0)), q[edge_name])
            out[mem_name] = max(float(out.get(mem_name, 0.0)), q[mem_name] * 1.10)
        elif variant % 4 == 2:
            out[sem_name] = max(float(out.get(sem_name, 0.0)), q[sem_name] * 1.25)
            out[raw_name] = max(float(out.get(raw_name, 0.0)), q[raw_name] * 1.10)
            out["dt_src_log"] = max(float(out.get("dt_src_log", 0.0)), float(rng.uniform(0.5, 2.5)))
        else:
            out["action_nll"] = max(float(out.get("action_nll", 0.0)), q["action_nll"] * 1.15)
            out["dst_type_nll"] = max(float(out.get("dst_type_nll", 0.0)), q["dst_type_nll"] * 1.15)
            out[edge_name] = max(float(out.get(edge_name, 0.0)), q[edge_name])
        return out

    def fit_gate(self, normal_details: list[dict[str, Any]]) -> dict[str, Any]:
        self.trained = False
        if not normal_details:
            self.training_summary = {"trained": False, "reason": "empty_normal_trace"}
            return dict(self.training_summary)
        self.fit_component_stats([component_vector(detail) for detail in normal_details])
        rng = np.random.default_rng(int(self.config.seed))
        max_trace = max(int(self.config.max_trace_events), 1)
        if len(normal_details) > max_trace:
            indices = rng.choice(len(normal_details), size=max_trace, replace=False)
            sampled = [normal_details[int(idx)] for idx in indices]
        else:
            sampled = list(normal_details)
        pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        synth_per_event = max(int(self.config.synthetic_per_event), 0)
        for idx, detail in enumerate(sampled):
            for variant in range(synth_per_event):
                pairs.append((detail, self._synthetic_detail(detail, rng, idx + variant)))
        if not pairs:
            self.training_summary = {"trained": False, "reason": "empty_training_pairs"}
            return dict(self.training_summary)

        dim = len(self.feature_names)
        comp = len(self.component_names)
        self.weights = rng.normal(0.0, 0.01, size=(comp, dim)).astype(np.float32)
        self.bias = np.zeros((comp,), dtype=np.float32)
        lr = float(self.config.learning_rate)
        l2 = float(self.config.l2)
        edge_idx = self.component_names.index("edge_transition_score")
        losses: list[float] = []

        def score_parts(detail: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
            x = self.features_from_detail(detail)
            z = self.standardize(detail)
            logits = self.weights @ x + self.bias
            sigmoid = self._sigmoid(logits)
            floor = min(max(float(self.config.uniform_prior_weight), 0.0), 0.95)
            gate = (np.float32(floor) + np.float32(1.0 - floor) * sigmoid).astype(np.float32, copy=False)
            score = float(np.dot(gate, z))
            return x, z, sigmoid, gate, score

        def reliability_derivative(sigmoid: np.ndarray) -> np.ndarray:
            learned_mix = np.float32(1.0 - min(max(float(self.config.uniform_prior_weight), 0.0), 0.95))
            return (learned_mix * sigmoid * (np.float32(1.0) - sigmoid)).astype(np.float32, copy=False)

        def grad_logits(sigmoid: np.ndarray, z: np.ndarray, grad_score: float) -> np.ndarray:
            return (np.float32(grad_score) * reliability_derivative(sigmoid) * z).astype(np.float32, copy=False)

        for epoch in range(max(int(self.config.epochs), 1)):
            rng.shuffle(pairs)
            total_loss = 0.0
            for normal_detail, synthetic_detail in pairs:
                x_n, z_n, sigmoid_n, gate_n, score_n = score_parts(normal_detail)
                x_s, z_s, sigmoid_s, _gate_s, score_s = score_parts(synthetic_detail)
                margin_gap = 1.0 - (score_s - score_n)
                grad_n = np.zeros((comp,), dtype=np.float32)
                grad_s = np.zeros((comp,), dtype=np.float32)
                if margin_gap > 0.0:
                    grad_n += grad_logits(sigmoid_n, z_n, 2.0 * margin_gap)
                    grad_s += grad_logits(sigmoid_s, z_s, -2.0 * margin_gap)
                    total_loss += float(margin_gap * margin_gap)
                if bool(x_n[self.feature_names.index("edge_only_flag")] > 0.5):
                    penalty = float(self.config.edge_only_penalty)
                    grad_n[edge_idx] += np.float32(penalty) * reliability_derivative(sigmoid_n)[edge_idx]
                    total_loss += penalty * float(gate_n[edge_idx])
                self.weights -= np.float32(lr) * (
                    np.outer(grad_n, x_n).astype(np.float32)
                    + np.outer(grad_s, x_s).astype(np.float32)
                    + np.float32(l2) * self.weights
                )
                self.bias -= np.float32(lr) * (grad_n + grad_s).astype(np.float32)
            losses.append(float(total_loss / max(len(pairs), 1)))
        self.trained = True
        edge_weights = []
        edge_only_edge_weights = []
        normal_scores = []
        synthetic_scores = []
        for normal_detail, synthetic_detail in pairs[: min(len(pairs), 5000)]:
            score, extra = self.score_detail(normal_detail)
            normal_scores.append(float(score))
            edge_weight = float(extra.get("edge_transition_score_weight", 0.0))
            edge_weights.append(edge_weight)
            if int(extra.get("rcg_edge_only_flag", 0)):
                edge_only_edge_weights.append(edge_weight)
            synth_score, _synth_extra = self.score_detail(synthetic_detail)
            synthetic_scores.append(float(synth_score))
        self.training_summary = {
            "trained": True,
            "normal_trace_events": int(len(normal_details)),
            "sampled_normal_events": int(len(sampled)),
            "synthetic_per_event": int(synth_per_event),
            "training_pairs": int(len(pairs)),
            "epochs": int(max(int(self.config.epochs), 1)),
            "loss_by_epoch": [float(x) for x in losses],
            "component_names": list(self.component_names),
            "feature_names": list(self.feature_names),
            "component_mean": {name: float(self.mean[idx]) for idx, name in enumerate(self.component_names)},
            "component_scale": {name: float(self.scale[idx]) for idx, name in enumerate(self.component_names)},
            "component_q_high": {name: float(self.q_high[idx]) for idx, name in enumerate(self.component_names)},
            "mean_edge_weight_sample": float(np.mean(edge_weights)) if edge_weights else None,
            "mean_edge_only_edge_weight_sample": float(np.mean(edge_only_edge_weights)) if edge_only_edge_weights else None,
            "normal_score_mean_sample": float(np.mean(normal_scores)) if normal_scores else None,
            "synthetic_score_mean_sample": float(np.mean(synthetic_scores)) if synthetic_scores else None,
            "leakage_check": "Gate fit uses train normal traces and train-derived synthetic perturbations only; validation is reserved for threshold calibration; no test labels or rankings.",
        }
        return dict(self.training_summary)

    def summary(self) -> dict[str, Any]:
        return {
            "purpose": "Experimental LTPNM_RCG component reliability calibration gate",
            "enabled": bool(self.config.enabled),
            "trained": bool(self.trained),
            "config": {
                "train_gate": bool(self.config.train_gate),
                "epochs": int(self.config.epochs),
                "learning_rate": float(self.config.learning_rate),
                "l2": float(self.config.l2),
                "synthetic_per_event": int(self.config.synthetic_per_event),
                "max_trace_events": int(self.config.max_trace_events),
                "edge_only_penalty": float(self.config.edge_only_penalty),
                "q_high": float(self.config.q_high),
                "uniform_prior_weight": float(self.config.uniform_prior_weight),
                "seed": int(self.config.seed),
            },
            "component_names": list(self.component_names),
            "feature_names": list(self.feature_names),
            "weight_shape": [int(x) for x in self.weights.shape],
            "approx_parameter_bytes": int(self.weights.nbytes + self.bias.nbytes + self.mean.nbytes + self.scale.nbytes + self.q_high.nbytes),
            "training_summary": dict(self.training_summary),
            "leakage_check": "Disabled by default. When enabled, score_pre_update runs before online state updates and threshold calibration still uses validation only.",
        }


class LTPNMNormalityModel:
    """Bounded, streaming temporal normality scorer for P2.

    This is the first lightweight LTPNM core. It learns conditional event
    distributions from train only, then scores validation/test events before
    updating online state. It is intentionally simple and explainable: every
    score component is a conditional NLL or pre-update memory residual.
    """

    def __init__(
        self,
        encoder: LTPNMEventEncoder,
        smoothing: float = 0.5,
        raw_weight: float = 0.25,
        raw_cap: float = 4.0,
        memory_dim: int = 16,
        memory_alpha: float = 0.25,
        edge_max_keys: int = 1_000_000,
    ) -> None:
        self.encoder = encoder
        self.smoothing = float(smoothing)
        self.raw_weight = float(raw_weight)
        self.raw_cap = float(raw_cap)
        self.memory_dim = max(int(memory_dim), 4)
        self.memory_alpha = float(memory_alpha)
        self.edge_max_keys = max(int(edge_max_keys), 1)

        self.global_action: Counter[int] = Counter()
        self.global_dst_type: Counter[int] = Counter()
        self.global_relation: Counter[int] = Counter()
        self.global_coarse: Counter[int] = Counter()
        self.global_raw: Counter[int] = Counter()

        self.action_by_src_type: dict[int, Counter[int]] = {}
        self.dst_type_by_src_action: dict[tuple[int, int], Counter[int]] = {}
        self.relation_by_types: dict[tuple[int, int], Counter[int]] = {}
        self.coarse_dst_by_src_action: dict[tuple[int, int], Counter[int]] = {}
        self.raw_dst_by_coarse_action: dict[tuple[int, int], Counter[int]] = {}
        self.edge_counts: dict[tuple[int, int, int], int] = {}

        self.train_events = 0
        self.edge_key_dropped = 0

    def _event_vector(self, event: EncodedEvent) -> np.ndarray:
        vec = np.zeros((self.memory_dim,), dtype=np.float32)
        items = [
            ("a", event.action_id),
            ("f", event.action_family_id),
            ("r", event.relation_id),
            ("st", event.src_type_id),
            ("dt", event.dst_type_id),
            ("ot", event.object_type_id),
            ("cs", event.coarse_src_id),
            ("cd", event.coarse_dst_id),
            ("rs", event.raw_src_id),
            ("rd", event.raw_dst_id),
        ]
        for prefix, value in items:
            token = f"{prefix}:{int(value)}"
            h = int(stable_hash(token, seed=97))
            dim = int(h % self.memory_dim)
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[dim] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def fit_event(self, event: EncodedEvent) -> None:
        self.global_action[int(event.action_id)] += 1
        self.global_dst_type[int(event.dst_type_id)] += 1
        self.global_relation[int(event.relation_id)] += 1
        self.global_coarse[int(event.coarse_dst_id)] += 1
        self.global_raw[int(event.raw_dst_id)] += 1

        self.action_by_src_type.setdefault(int(event.src_type_id), Counter())[int(event.action_id)] += 1
        self.dst_type_by_src_action.setdefault((int(event.src_type_id), int(event.action_id)), Counter())[int(event.dst_type_id)] += 1
        self.relation_by_types.setdefault((int(event.src_type_id), int(event.dst_type_id)), Counter())[int(event.relation_id)] += 1
        self.coarse_dst_by_src_action.setdefault((int(event.coarse_src_id), int(event.action_id)), Counter())[int(event.coarse_dst_id)] += 1
        self.raw_dst_by_coarse_action.setdefault((int(event.coarse_dst_id), int(event.action_id)), Counter())[int(event.raw_dst_id)] += 1
        key = (int(event.info_src), int(event.relation_id), int(event.info_dst))
        if key in self.edge_counts or len(self.edge_counts) < self.edge_max_keys:
            self.edge_counts[key] = int(self.edge_counts.get(key, 0)) + 1
        else:
            self.edge_key_dropped += 1
        self.train_events += 1

    def fit(self, events: Iterable[EncodedEvent]) -> dict[str, Any]:
        for event in events:
            self.fit_event(event)
        return self.summary()

    def new_state(self) -> dict[str, Any]:
        return {
            "node_memory": {},
            "node_last_ts": {},
            "edge_seen": {},
            "global_memory": np.zeros((self.memory_dim,), dtype=np.float32),
            "last_global_ts": -1,
        }

    def score_pre_update(self, event: EncodedEvent, state: dict[str, Any]) -> tuple[float, dict[str, float | str | int]]:
        action_space = len(self.encoder.action.token_to_id)
        dst_type_space = 4
        relation_space = 7
        coarse_space = len(self.encoder.coarse.token_to_id)
        raw_space = len(self.encoder.raw.token_to_id)

        action_counter = self.action_by_src_type.get(int(event.src_type_id), self.global_action)
        dst_counter = self.dst_type_by_src_action.get((int(event.src_type_id), int(event.action_id)), self.global_dst_type)
        relation_counter = self.relation_by_types.get((int(event.src_type_id), int(event.dst_type_id)), self.global_relation)
        coarse_counter = self.coarse_dst_by_src_action.get((int(event.coarse_src_id), int(event.action_id)), self.global_coarse)
        raw_counter = self.raw_dst_by_coarse_action.get((int(event.coarse_dst_id), int(event.action_id)), self.global_raw)

        action_nll = _counter_nll(action_counter, int(event.action_id), action_space, self.smoothing)
        dst_type_nll = _counter_nll(dst_counter, int(event.dst_type_id), dst_type_space, self.smoothing)
        relation_nll = _counter_nll(relation_counter, int(event.relation_id), relation_space, self.smoothing)
        coarse_nll = _counter_nll(coarse_counter, int(event.coarse_dst_id), coarse_space, self.smoothing)
        raw_nll_raw = _counter_nll(raw_counter, int(event.raw_dst_id), raw_space, self.smoothing)
        raw_identity_nll = min(raw_nll_raw, self.raw_cap)

        edge_key = (int(event.info_src), int(event.relation_id), int(event.info_dst))
        train_edge_seen = int(self.edge_counts.get(edge_key, 0))
        online_edge_seen = int(state["edge_seen"].get(edge_key, 0))
        edge_transition_nll = float(math.log1p(self.train_events) - math.log1p(train_edge_seen + online_edge_seen))
        edge_transition_nll = max(edge_transition_nll, 0.0)

        z = self._event_vector(event)
        src_mem = state["node_memory"].get(int(event.info_src))
        dst_mem = state["node_memory"].get(int(event.info_dst))
        global_mem = state["global_memory"]
        candidates = [global_mem]
        if src_mem is not None:
            candidates.append(src_mem)
        if dst_mem is not None:
            candidates.append(dst_mem)
        pred = np.mean(np.stack(candidates, axis=0), axis=0).astype(np.float32)
        memory_transition_residual = float(np.mean((z - pred) ** 2))

        dt_src = log1p_seconds(int(event.timestamp_ns), int(state["node_last_ts"].get(int(event.info_src), -1)))
        dt_dst = log1p_seconds(int(event.timestamp_ns), int(state["node_last_ts"].get(int(event.info_dst), -1)))

        components = {
            "action_nll": float(action_nll),
            "dst_type_nll": float(dst_type_nll),
            "relation_nll": float(relation_nll),
            "edge_transition_score": float(edge_transition_nll),
            "coarse_semantic_nll": float(coarse_nll),
            "raw_identity_nll": float(self.raw_weight * raw_identity_nll),
            "memory_transition_residual": float(memory_transition_residual),
        }
        components = {name: float(components.get(name, 0.0)) for name in LTPNM_COMPONENT_NAMES}
        score = float(sum(components.values()))
        detail: dict[str, float | str | int] = {
            **components,
            "raw_identity_nll_uncapped": float(raw_nll_raw),
            "raw_identity_cap": float(self.raw_cap),
            "raw_identity_weight": float(self.raw_weight),
            "train_edge_seen": int(train_edge_seen),
            "online_edge_seen": int(online_edge_seen),
            "dt_src_log": float(dt_src),
            "dt_dst_log": float(dt_dst),
            "top_component": component_top(components),
            "coarse_src": event.coarse_src,
            "coarse_dst": event.coarse_dst,
            "raw_src": event.raw_src,
            "raw_dst": event.raw_dst,
        }
        return score, detail

    def update_state(self, event: EncodedEvent, state: dict[str, Any]) -> None:
        z = self._event_vector(event)
        alpha = min(max(float(self.memory_alpha), 0.0), 1.0)
        node_memory: dict[int, np.ndarray] = state["node_memory"]
        for node in [int(event.info_src), int(event.info_dst)]:
            prev = node_memory.get(node)
            if prev is None:
                node_memory[node] = z.copy()
            else:
                node_memory[node] = ((1.0 - alpha) * prev + alpha * z).astype(np.float32, copy=False)
            state["node_last_ts"][node] = int(event.timestamp_ns)
        state["global_memory"] = ((1.0 - alpha) * state["global_memory"] + alpha * z).astype(np.float32, copy=False)
        state["last_global_ts"] = int(event.timestamp_ns)
        edge_key = (int(event.info_src), int(event.relation_id), int(event.info_dst))
        state["edge_seen"][edge_key] = int(state["edge_seen"].get(edge_key, 0)) + 1

    def state_profile(self, state: dict[str, Any]) -> dict[str, int]:
        return {
            "node_memory_entries": int(len(state.get("node_memory", {}))),
            "node_last_ts_entries": int(len(state.get("node_last_ts", {}))),
            "edge_seen_entries": int(len(state.get("edge_seen", {}))),
            "memory_dim": int(self.memory_dim),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "purpose": "P2 lightweight temporal conditional normality model",
            "train_events": int(self.train_events),
            "smoothing": float(self.smoothing),
            "raw_weight": float(self.raw_weight),
            "raw_cap": float(self.raw_cap),
            "memory_dim": int(self.memory_dim),
            "memory_alpha": float(self.memory_alpha),
            "edge_max_keys": int(self.edge_max_keys),
            "edge_keys": int(len(self.edge_counts)),
            "edge_key_dropped": int(self.edge_key_dropped),
            "context_tables": {
                "action_by_src_type": int(len(self.action_by_src_type)),
                "dst_type_by_src_action": int(len(self.dst_type_by_src_action)),
                "relation_by_types": int(len(self.relation_by_types)),
                "coarse_dst_by_src_action": int(len(self.coarse_dst_by_src_action)),
                "raw_dst_by_coarse_action": int(len(self.raw_dst_by_coarse_action)),
            },
            "score_components": [
                "action_nll",
                "dst_type_nll",
                "relation_nll",
                "edge_transition_score",
                "coarse_semantic_nll",
                "raw_identity_nll",
                "memory_transition_residual",
            ],
            "leakage_check": "All count tables are fit from train events only. Online state is updated only after score_pre_update.",
        }
