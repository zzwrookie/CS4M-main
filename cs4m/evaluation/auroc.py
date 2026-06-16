"""AUROC and ROC helpers for post-inference evaluation."""

from __future__ import annotations

from typing import Any, Iterable


def compute_auroc(samples: Iterable[dict[str, Any]]) -> float | None:
    """Compute AUROC from full scored samples or return None for one class."""
    pairs = [(_float(row["score"]), int(row["y_true"])) for row in samples]
    positives = [score for score, label in pairs if label == 1]
    negatives = [score for score, label in pairs if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for positive in positives:
        for negative in negatives:
            total += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return float(wins / max(total, 1))


def roc_curve_points(samples: Iterable[dict[str, Any]]) -> list[dict[str, float | str]]:
    """Return threshold,FPR,TPR rows for full scored samples."""
    rows = [(_float(row["score"]), int(row["y_true"])) for row in samples]
    positives = sum(1 for _, label in rows if label == 1)
    negatives = sum(1 for _, label in rows if label == 0)
    out: list[dict[str, float | str]] = [{"threshold": "inf", "fpr": 0.0, "tpr": 0.0}]
    if positives == 0 or negatives == 0:
        return out
    for threshold in sorted({score for score, _ in rows}, reverse=True):
        tp = sum(1 for score, label in rows if score >= threshold and label == 1)
        fp = sum(1 for score, label in rows if score >= threshold and label == 0)
        out.append(
            {
                "threshold": float(threshold),
                "fpr": float(fp / negatives),
                "tpr": float(tp / positives),
            }
        )
    return out


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
