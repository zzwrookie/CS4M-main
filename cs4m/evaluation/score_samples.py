"""Build full scored event and node samples for post-inference evaluation."""

from __future__ import annotations

from typing import Any, Iterable


NODE_ID_FIELDS = ("src_idx", "dst_idx", "info_src", "info_dst")


def derive_node_scores_from_event_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[str, float]:
    """Assign each node the maximum score of any associated scored event."""
    scores: dict[str, float] = {}
    for row in rows:
        score = _float(row.get("score"))
        for field in NODE_ID_FIELDS:
            node_id = str(row.get(field, "")).strip()
            if not node_id:
                continue
            scores[node_id] = max(scores.get(node_id, 0.0), score)
    return scores


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
