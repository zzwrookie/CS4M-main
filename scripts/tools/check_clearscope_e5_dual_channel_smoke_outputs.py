#!/usr/bin/env python3
"""Check ClearScope E5 dual-channel bounded-smoke output contracts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LEAKAGE_HEADER_TOKENS = ("label", "malicious", "ground_truth", "attack")

REQUIRED_EXPLANATION_FIELDS = {
    "stream_pos",
    "event_index",
    "trigger_node_idx",
    "trigger_node_role",
    "event_score",
    "score_floor",
    "event_semantic_max_score_seen",
    "event_semantic_near_threshold_count",
    "event_semantic_support_event_count",
    "support_threshold",
    "source_target_case",
    "alert_channel",
    "threshold_source",
}


class SmokeCheckError(RuntimeError):
    """Raised when bounded-smoke outputs violate the expected contract."""


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise SmokeCheckError(f"missing required output: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def _reject_leakage_headers(fieldnames: list[str]) -> None:
    for field in fieldnames:
        lowered = field.lower()
        for token in LEAKAGE_HEADER_TOKENS:
            if token in lowered:
                raise SmokeCheckError(f"alert header contains leakage token: {field}")


def _check_nondecreasing(rows: list[dict[str, str]], field: str, label: str) -> None:
    previous: int | None = None
    for row in rows:
        if row.get(field, "") == "":
            continue
        current = int(row[field])
        if previous is not None and current < previous:
            raise SmokeCheckError(f"{label} is not nondecreasing by {field}")
        previous = current


def _check_explanations(
    rows: list[dict[str, str]],
    *,
    score_floor: float,
    support_threshold: int,
) -> None:
    for row in rows:
        max_score_seen = float(row["event_semantic_max_score_seen"])
        near_count = int(row["event_semantic_near_threshold_count"])
        row_score_floor = float(row["score_floor"])
        row_support_threshold = int(row["support_threshold"])
        if max_score_seen < float(score_floor):
            raise SmokeCheckError("explanation max score is below configured floor")
        if row_score_floor != float(score_floor):
            raise SmokeCheckError("explanation score floor does not match checker input")
        if near_count < int(support_threshold):
            raise SmokeCheckError("explanation support count is below threshold")
        if row_support_threshold != int(support_threshold):
            raise SmokeCheckError("explanation support threshold mismatch")
        if row["source_target_case"] != "event_semantic_target":
            raise SmokeCheckError("explanation source target case is not event-semantic")
        if row["threshold_source"] != "validation_event_semantic_p999_half":
            raise SmokeCheckError("explanation threshold source is not validation-only")


def check_outputs(
    result_dir: Path,
    *,
    score_floor: float,
    support_threshold: int,
) -> dict[str, Any]:
    """Validate dual-channel bounded-smoke outputs and return a summary."""
    alert_path = result_dir / "online_event_alerts.csv"
    explanation_path = result_dir / "online_event_alert_explanations.csv"
    alert_fields, alert_rows = _read_csv(alert_path)
    explanation_fields, explanation_rows = _read_csv(explanation_path)

    _reject_leakage_headers(alert_fields)
    missing = REQUIRED_EXPLANATION_FIELDS.difference(explanation_fields)
    if missing:
        raise SmokeCheckError(f"explanation sidecar missing fields: {sorted(missing)}")

    _check_nondecreasing(alert_rows, "event_index", "alerts")
    _check_nondecreasing(explanation_rows, "stream_pos", "explanations")
    _check_explanations(
        explanation_rows,
        score_floor=score_floor,
        support_threshold=support_threshold,
    )
    return {
        "result_dir": str(result_dir),
        "alert_rows": len(alert_rows),
        "explanation_rows": len(explanation_rows),
        "score_floor": float(score_floor),
        "support_threshold": int(support_threshold),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check ClearScope E5 dual-channel bounded-smoke outputs.",
    )
    parser.add_argument("--result_dir", required=True, type=Path)
    parser.add_argument("--score_floor", required=True, type=float)
    parser.add_argument("--support_threshold", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    """Run the smoke output checker from the command line."""
    args = _parse_args()
    try:
        summary = check_outputs(
            args.result_dir,
            score_floor=args.score_floor,
            support_threshold=args.support_threshold,
        )
    except SmokeCheckError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        return 1
    print(json.dumps({"ok": True, **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
