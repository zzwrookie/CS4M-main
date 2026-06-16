#!/usr/bin/env python3
"""Evaluate full scored CS4M samples after inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.evaluation.auroc import compute_auroc
from cs4m.evaluation.score_samples import derive_node_scores_from_event_rows


def evaluate_result_dir(
    result_dir: Path,
    event_labels: Mapping[str, int],
    node_labels: Mapping[str, int],
) -> dict[str, Any]:
    """Evaluate full scored event and node samples from one result directory."""
    result_path = Path(result_dir)
    score_path = result_path / "online_event_score_trace.csv"
    if not score_path.exists():
        return {"status": "missing_input", "missing_input": "online_event_score_trace.csv"}

    rows = _read_csv(score_path)
    event_samples = _event_samples(rows, event_labels)
    node_scores = derive_node_scores_from_event_rows(rows)
    node_samples = [
        {"id": node_id, "score": score, "y_true": int(node_labels.get(node_id, 0))}
        for node_id, score in sorted(node_scores.items())
    ]
    payload = {
        "event_auroc": compute_auroc(event_samples),
        "event_confusion": _confusion(event_samples),
        "node_auroc_strict": compute_auroc(node_samples),
        "node_confusion_strict": _confusion(node_samples),
        "status": "ok",
    }
    (result_path / "eval_metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def _event_samples(
    rows: list[dict[str, str]],
    event_labels: Mapping[str, int],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in rows:
        event_id = str(row.get("event_id", "")).strip()
        if not event_id:
            continue
        samples.append(
            {
                "id": event_id,
                "score": _float(row.get("score")),
                "y_true": int(event_labels.get(event_id, 0)),
            }
        )
    return samples


def _confusion(samples: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, int]:
    out = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in samples:
        y_true = int(row.get("y_true", 0)) == 1
        predicted = _float(row.get("score")) >= threshold
        if predicted and y_true:
            out["tp"] += 1
        elif predicted and not y_true:
            out["fp"] += 1
        elif not predicted and y_true:
            out["fn"] += 1
        else:
            out["tn"] += 1
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> int:
    """Run full-score evaluation with empty external labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    result = evaluate_result_dir(Path(args.result_dir), event_labels={}, node_labels={})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
