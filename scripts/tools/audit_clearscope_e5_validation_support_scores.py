#!/usr/bin/env python3
"""Audit ClearScope E5 validation support and target-case score behavior."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_RESULT_DIR = Path("outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE")


def _int_field(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(str(row.get(key, "")).strip()))
    except ValueError:
        return int(default)


def _float_field(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(str(row.get(key, "")).strip())
    except ValueError:
        return float(default)


def load_group_rows(path: Path) -> list[dict[str, str]]:
    """Load per-target/action/type group score summary rows."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_target_case_rows(path: Path) -> list[dict[str, str]]:
    """Load target-case score summary rows."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _group_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("target_case", "")),
            str(row.get("action_name", "")),
            str(row.get("src_type_name", "")),
            str(row.get("dst_type_name", "")),
        ],
    )


def _is_unseen(row: Mapping[str, Any]) -> bool:
    return (
        _int_field(row, "validation_count") == 0
        or str(row.get("validation_count_bucket", "")).lower() == "unseen"
        or str(row.get("threshold_level", "")).lower().startswith("unseen")
    )


def _recommend(summary: Mapping[str, Any]) -> str:
    if int(summary.get("both_cold_unseen_alert_group_count", 0)) > 0:
        return "validation_only_cold_start_gate_smoke"
    if float(summary.get("event_semantic_threshold_gap", 0.0)) > 0.0:
        return "validation_only_target_case_threshold_audit"
    return "score_model_diagnosis"


def compute_audit(
    *,
    group_rows: Sequence[Mapping[str, str]],
    target_case_rows: Sequence[Mapping[str, str]],
    score_summary: Mapping[str, Any],
    missed_gt_diagnosis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute validation support and target-case score audit metrics."""
    final_threshold = float(score_summary.get("final_threshold", 0.0) or 0.0)
    both_cold_policy = dict(score_summary.get("both_cold_unseen_policy", {}) or {})
    endpoint_suppression = dict(score_summary.get("conditional_endpoint_suppression", {}) or {})

    unseen_rows = [row for row in group_rows if _is_unseen(row)]
    both_cold_unseen_alert_rows = [
        row
        for row in unseen_rows
        if str(row.get("target_case", "")) == "both_cold_action_target"
        and _int_field(row, "alert_count") > 0
    ]
    event_semantic_rows = [
        row for row in target_case_rows if str(row.get("target_case", "")) == "event_semantic_target"
    ]
    both_cold_rows = [
        row for row in target_case_rows if str(row.get("target_case", "")) == "both_cold_action_target"
    ]
    event_semantic_alert_count = sum(_int_field(row, "alert_count") for row in event_semantic_rows)
    event_semantic_score_p999 = max(
        [_float_field(row, "score_p999") for row in event_semantic_rows] or [0.0],
    )
    event_semantic_threshold = max(
        [_float_field(row, "threshold_mean") for row in event_semantic_rows] or [final_threshold],
    )
    event_semantic_threshold_gap = max(
        float(event_semantic_threshold - event_semantic_score_p999),
        0.0,
    )
    both_cold_alert_count = sum(_int_field(row, "alert_count") for row in both_cold_rows)
    missed = dict(missed_gt_diagnosis or {})

    audit_rows = []
    for row in group_rows:
        audit_rows.append(
            {
                "key": _group_key(row),
                "target_case": row.get("target_case", ""),
                "action": row.get("action_name", ""),
                "src_type": row.get("src_type_name", ""),
                "dst_type": row.get("dst_type_name", ""),
                "validation_count": _int_field(row, "validation_count"),
                "test_count": _int_field(row, "test_count"),
                "alert_count": _int_field(row, "alert_count"),
                "is_unseen_validation": _is_unseen(row),
                "test_p999": _float_field(row, "test_p999"),
                "test_max": _float_field(row, "test_max"),
            },
        )

    summary = {
        "dataset": "CLEARSCOPE_E5",
        "group_count": int(len(group_rows)),
        "unseen_group_count": int(len(unseen_rows)),
        "unseen_test_event_count": int(sum(_int_field(row, "test_count") for row in unseen_rows)),
        "both_cold_unseen_alert_group_count": int(len(both_cold_unseen_alert_rows)),
        "both_cold_unseen_alert_count": int(
            sum(_int_field(row, "alert_count") for row in both_cold_unseen_alert_rows),
        ),
        "both_cold_alert_count": int(both_cold_alert_count),
        "event_semantic_alert_count": int(event_semantic_alert_count),
        "event_semantic_score_p999": float(event_semantic_score_p999),
        "event_semantic_threshold": float(event_semantic_threshold),
        "event_semantic_threshold_gap": float(event_semantic_threshold_gap),
        "both_cold_unseen_policy": str(both_cold_policy.get("policy_name", "")),
        "endpoint_suppression_enabled": bool(endpoint_suppression.get("enabled", False)),
        "missed_gt_count": int(missed.get("missed_gt_count", 0) or 0),
        "missed_both_cold_node_count": int(missed.get("missed_both_cold_node_count", 0) or 0),
        "missed_event_semantic_node_count": int(
            missed.get("missed_event_semantic_node_count", 0) or 0,
        ),
        "leakage_contract": {
            "post_stream_only": True,
            "labels_used_for_threshold_or_policy": False,
            "labels_used_for_training": False,
            "original_outputs_mutated": False,
        },
    }
    summary["recommended_next_experiment"] = _recommend(summary)
    return {"summary": summary, "rows": audit_rows}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "key",
        "target_case",
        "action",
        "src_type",
        "dst_type",
        "validation_count",
        "test_count",
        "alert_count",
        "is_unseen_validation",
        "test_p999",
        "test_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_outputs(result_dir: Path, audit: Mapping[str, Any]) -> dict[str, Path]:
    """Write validation support audit sidecars."""
    result_dir = Path(result_dir)
    paths = {
        "json": result_dir / "e5_validation_support_score_audit.json",
        "csv": result_dir / "e5_validation_support_score_audit.csv",
    }
    paths["json"].write_text(
        json.dumps(dict(audit["summary"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(paths["csv"], list(audit["rows"]))
    return paths


def run_audit(
    *,
    result_dir: Path,
    group_summary_path: Path,
    target_case_summary_path: Path,
    score_summary_path: Path,
    missed_gt_diagnosis_path: Path | None,
) -> dict[str, Any]:
    """Run validation support and score audit."""
    audit = compute_audit(
        group_rows=load_group_rows(group_summary_path),
        target_case_rows=load_target_case_rows(target_case_summary_path),
        score_summary=_load_json(score_summary_path),
        missed_gt_diagnosis=_load_json(missed_gt_diagnosis_path),
    )
    paths = write_outputs(result_dir, audit)
    return {
        "summary": audit["summary"],
        "paths": {key: str(value) for key, value in paths.items()},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--group_summary",
        type=Path,
        default=DEFAULT_RESULT_DIR / "conditional_score_summary_by_target_action_type.csv",
    )
    parser.add_argument(
        "--target_case_summary",
        type=Path,
        default=DEFAULT_RESULT_DIR / "target_case_summary.csv",
    )
    parser.add_argument(
        "--score_summary",
        type=Path,
        default=DEFAULT_RESULT_DIR / "score_summary.json",
    )
    parser.add_argument(
        "--missed_gt_diagnosis",
        type=Path,
        default=DEFAULT_RESULT_DIR / "e5_missed_gt_score_diagnosis.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI."""
    args = parse_args(argv)
    result = run_audit(
        result_dir=args.result_dir,
        group_summary_path=args.group_summary,
        target_case_summary_path=args.target_case_summary,
        score_summary_path=args.score_summary,
        missed_gt_diagnosis_path=args.missed_gt_diagnosis,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
