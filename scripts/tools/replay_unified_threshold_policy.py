#!/usr/bin/env python3
"""Replay unified group thresholds over existing online score traces."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.thresholds.unified_group import GroupKey, UnifiedGroupThresholdResolver


DEFAULT_RESULT_DIRS = [
    "outputs/results/tflr_light/CADETS_E3_FREEBSD_V3_SAFE_LEXICAL_E4_FULL",
    "outputs/results/phase3g_theia_e3_e4_theia_v1_lazy100k_pair_only_full_rerun_20260602_142411",
    "outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE",
    "outputs/results/tflr_light/optc_windows_v1_3c_full/OPTC_051_OPTC_WINDOWS_V1_3C_DETAIL_FULL_INFER",
    "outputs/results/tflr_light/optc_windows_v1_3c_full/OPTC_201_OPTC_WINDOWS_V1_3C_DETAIL_FULL_INFER",
    "outputs/results/tflr_light/optc_windows_v1_3c_full/OPTC_501_OPTC_WINDOWS_V1_3C_DETAIL_FULL_INFER",
]


class ThresholdResolver:
    """Compatibility wrapper around the shared unified group resolver."""

    def __init__(
        self,
        validation_group_thresholds: dict[GroupKey, dict[str, Any]],
        train_group_thresholds: dict[GroupKey, dict[str, Any]],
        validation_global_threshold: float,
    ) -> None:
        self._resolver = UnifiedGroupThresholdResolver(
            validation_groups={
                key: {
                    "validation_count": value.get("validation_count"),
                    "validation_quantile_threshold": value.get("threshold"),
                }
                for key, value in validation_group_thresholds.items()
            },
            train_groups=train_group_thresholds,
            validation_global_threshold=validation_global_threshold,
            min_validation_count=1,
            min_train_count=1,
            train_margin_abs=0.0,
            train_margin_rel=0.0,
        )

    def resolve(self, key: GroupKey) -> dict[str, Any]:
        """Resolve threshold metadata for one group key."""
        resolved = self._resolver.resolve(key)
        threshold_level = resolved.threshold_level
        if threshold_level == "exact_group":
            threshold_level = "group"
        elif threshold_level == "exact_group_train_fallback":
            threshold_level = "group_train_fallback"
        return {
            "group_key": resolved.group_key,
            "threshold": resolved.threshold,
            "threshold_source": resolved.threshold_source,
            "threshold_level": threshold_level,
            "threshold_quantile": 0.999
            if resolved.threshold_source != "train_group_max"
            else "",
            "validation_count": resolved.validation_count,
            "train_count": resolved.train_count,
        }


def replay_result_dir(result_dir: Path, max_rows: int = 0) -> dict[str, Any]:
    """Replay unified thresholds for one existing result directory."""
    result_dir = Path(result_dir)
    score_path = result_dir / "online_event_score_trace.csv"
    if not score_path.exists():
        return {
            "result_dir": str(result_dir),
            "missing_score_trace": True,
            "score_rows": 0,
            "raw_alert_rows": 0,
            "final_alert_rows": _count_csv_rows(result_dir / "online_event_alerts.csv"),
            "threshold_sources": {},
            "train_fallback_rows": 0,
            "elapsed_seconds": 0.0,
        }

    start = time.monotonic()
    train_groups = _load_train_group_summary(result_dir / "train_group_score_summary.csv")
    validation_groups = _load_validation_group_summary(
        result_dir / "validation_group_threshold_summary.csv"
    )
    global_values: list[float] = []
    source_counts: Counter[str] = Counter()
    score_rows = 0
    raw_alert_rows = 0
    train_fallback_rows = 0

    with score_path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), 1):
            if max_rows > 0 and index > max_rows:
                break
            key = _key_from_row(row)
            validation_count = _to_int(row.get("validation_count"))
            threshold = _to_float(row.get("threshold"))
            score = _to_float(row.get("score"))
            if validation_count > 0 and key not in validation_groups:
                validation_groups[key] = {
                    "threshold": threshold,
                    "threshold_quantile": _to_float(row.get("threshold_quantile"), 0.999),
                    "validation_count": validation_count,
                }
            global_values.append(threshold if threshold > 0 else score)

    validation_global = max(global_values) if global_values else 0.0
    resolver = ThresholdResolver(
        validation_group_thresholds=validation_groups,
        train_group_thresholds=train_groups,
        validation_global_threshold=validation_global,
    )

    with score_path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), 1):
            if max_rows > 0 and index > max_rows:
                break
            score_rows += 1
            resolved = resolver.resolve(_key_from_row(row))
            source = str(resolved["threshold_source"])
            source_counts[source] += 1
            if source == "train_group_max":
                train_fallback_rows += 1
            if _to_float(row.get("score")) >= _to_float(resolved.get("threshold")):
                raw_alert_rows += 1

    return {
        "result_dir": str(result_dir),
        "missing_score_trace": False,
        "score_rows": score_rows,
        "raw_alert_rows": raw_alert_rows,
        "final_alert_rows": _count_csv_rows(result_dir / "online_event_alerts.csv"),
        "threshold_sources": dict(sorted(source_counts.items())),
        "train_fallback_rows": train_fallback_rows,
        "max_rows_applied": max_rows,
        "elapsed_seconds": time.monotonic() - start,
    }


def write_replay_outputs(
    result_dirs: list[Path],
    output_dir: Path,
    max_rows_per_result: int = 0,
) -> dict[str, Any]:
    """Replay multiple result directories and write JSON/CSV summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [replay_result_dir(path, max_rows=max_rows_per_result) for path in result_dirs]
    total = {
        "result_count": len(summaries),
        "score_rows": sum(int(item.get("score_rows", 0)) for item in summaries),
        "raw_alert_rows": sum(int(item.get("raw_alert_rows", 0)) for item in summaries),
        "final_alert_rows": sum(int(item.get("final_alert_rows", 0)) for item in summaries),
        "train_fallback_rows": sum(
            int(item.get("train_fallback_rows", 0)) for item in summaries
        ),
        "max_rows_per_result": max_rows_per_result,
    }
    source_rows = _threshold_source_rows(summaries)
    payload = {"summary": total, "results": summaries}
    (output_dir / "unified_threshold_replay_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_dict_csv(output_dir / "unified_threshold_replay_by_dataset.csv", summaries)
    _write_dict_csv(output_dir / "unified_threshold_source_summary.csv", source_rows)
    _write_missing_inputs(output_dir / "unified_threshold_missing_inputs.csv", summaries)
    _write_policy_summary(output_dir / "unified_threshold_policy_summary.csv", summaries)
    return payload


def _load_train_group_summary(path: Path) -> dict[GroupKey, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[GroupKey, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records[_key_from_row(row)] = {
                "train_count": _to_int(row.get("train_count")),
                "train_max_score": _to_float(row.get("train_max_score")),
            }
    return records


def _load_validation_group_summary(path: Path) -> dict[GroupKey, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[GroupKey, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records[_key_from_row(row)] = {
                "threshold": _to_float(row.get("threshold")),
                "threshold_quantile": _to_float(row.get("threshold_quantile"), 0.9995),
                "validation_count": _to_int(row.get("validation_count")),
            }
    return records


def _key_from_row(row: dict[str, Any]) -> GroupKey:
    return GroupKey(
        str(row.get("target_case", "")),
        str(row.get("action", "")),
        str(row.get("src_type", "")),
        str(row.get("dst_type", "")),
    )


def _threshold_source_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summaries:
        for source, count in dict(item.get("threshold_sources", {})).items():
            rows.append(
                {
                    "result_dir": item.get("result_dir", ""),
                    "threshold_source": source,
                    "count": count,
                }
            )
    return rows


def _write_missing_inputs(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = [
        {
            "result_dir": item.get("result_dir", ""),
            "missing_input": "online_event_score_trace.csv",
        }
        for item in summaries
        if item.get("missing_score_trace")
    ]
    _write_dict_csv(path, rows)


def _write_policy_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    rows = [
        {
            "result_dir": item.get("result_dir", ""),
            "raw_alert_rows": item.get("raw_alert_rows", 0),
            "final_alert_rows": item.get("final_alert_rows", 0),
        }
        for item in summaries
    ]
    _write_dict_csv(path, rows)


def _write_dict_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if not fieldnames:
        fieldnames = ["result_dir", "missing_input"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for _ in reader)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def main() -> int:
    """Run unified threshold replay from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows-per-result", type=int, default=0)
    args = parser.parse_args()
    result_dirs = [Path(path) for path in (args.result_dir or DEFAULT_RESULT_DIRS)]
    payload = write_replay_outputs(
        result_dirs,
        Path(args.output_dir),
        max_rows_per_result=max(0, int(args.max_rows_per_result)),
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
