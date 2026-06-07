from __future__ import annotations

from typing import Mapping

import numpy as np


ORTHRUS10_RAW_ACTIONS: tuple[str, ...] = (
    "EVENT_CLONE",
    "EVENT_CONNECT",
    "EVENT_EXECUTE",
    "EVENT_OPEN",
    "EVENT_READ",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG",
    "EVENT_SENDMSG",
    "EVENT_SENDTO",
    "EVENT_WRITE",
)

RAW_ACTION_IDS: dict[str, int] = {
    action: index for index, action in enumerate(ORTHRUS10_RAW_ACTIONS)
}

LAMBDA_ACTION: dict[str, float] = {
    "EVENT_CLONE": 0.4,
    "EVENT_CONNECT": 0.5,
    "EVENT_EXECUTE": 0.5,
    "EVENT_OPEN": 0.1,
    "EVENT_READ": 0.1,
    "EVENT_RECVFROM": 0.1,
    "EVENT_RECVMSG": 0.1,
    "EVENT_SENDMSG": 0.5,
    "EVENT_SENDTO": 0.5,
    "EVENT_WRITE": 0.5,
}

RHO_ACTION: dict[str, float] = {
    "EVENT_CLONE": 0.2,
    "EVENT_CONNECT": 0.2,
    "EVENT_EXECUTE": 0.2,
    "EVENT_OPEN": 0.05,
    "EVENT_READ": 0.05,
    "EVENT_RECVFROM": 0.05,
    "EVENT_RECVMSG": 0.05,
    "EVENT_SENDMSG": 0.2,
    "EVENT_SENDTO": 0.2,
    "EVENT_WRITE": 0.2,
}

SRC_TYPE_SCALE: dict[str, float] = {
    "process": 1.0,
    "file": 0.3,
    "netflow": 0.3,
    "unknown": 0.1,
}


def _table_delta(table: Mapping[str, object]) -> tuple[set[str], set[str]]:
    expected = set(ORTHRUS10_RAW_ACTIONS)
    actual = set(table)
    return expected - actual, actual - expected


def validate_raw_action_tables() -> None:
    """Fail fast unless all raw action tables cover the same ORTHRUS10 set."""
    failures = []
    for name, table in (
        ("RAW_ACTION_IDS", RAW_ACTION_IDS),
        ("LAMBDA_ACTION", LAMBDA_ACTION),
        ("RHO_ACTION", RHO_ACTION),
    ):
        missing, extra = _table_delta(table)
        if missing or extra:
            failures.append(f"{name} missing={sorted(missing)} extra={sorted(extra)}")
    if failures:
        raise ValueError("; ".join(failures))


def normalize_raw_action(action: object) -> str:
    """Return an uppercase ORTHRUS-style raw event action token."""
    text = "" if action is None else str(action).strip()
    token = text.upper()
    if token.startswith("EVENT_"):
        return token
    if token.startswith("EVENT-"):
        return token.replace("-", "_")
    if token.startswith("EVENT "):
        return token.replace(" ", "_")
    return token


def checked_raw_action(action: object) -> str:
    """Return a known raw action or raise with the unknown token."""
    validate_raw_action_tables()
    token = normalize_raw_action(action)
    if token not in RAW_ACTION_IDS:
        raise ValueError(f"unknown raw ORTHRUS10 action: {token}")
    return token


def raw_action_one_hot(action: object) -> np.ndarray:
    """Return the 10-dimensional raw ORTHRUS action one-hot vector."""
    token = checked_raw_action(action)
    vector = np.zeros((len(ORTHRUS10_RAW_ACTIONS),), dtype=np.float32)
    vector[int(RAW_ACTION_IDS[token])] = np.float32(1.0)
    return vector


def lambda_rho_for_event(action: object, src_type: object) -> tuple[float, float]:
    """Return label-free propagation gates for a raw action and source type."""
    token = checked_raw_action(action)
    type_key = "unknown" if src_type is None else str(src_type).strip().lower()
    scale = float(SRC_TYPE_SCALE.get(type_key, SRC_TYPE_SCALE["unknown"]))
    return float(LAMBDA_ACTION[token] * scale), float(RHO_ACTION[token] * scale)


def fixed_update_gate(mode: str, residual_score: float | None = None) -> float:
    """Return q_t for disabled update gating."""
    if str(mode) != "none":
        raise ValueError(f"unsupported SSPM update gate mode without calibrator: {mode}")
    _ = residual_score
    return 1.0
