from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

EVAL_JSON_NAME = "eval_causal_semantics_slim.json"
DEFAULT_GLOB = f"*CAUSAL_SEMANTICS_SLIM_*_BUDGET*/{EVAL_JSON_NAME}"
COMMON_NODE_TOPK_VALUES = (
    100,
    200,
    500,
    1000,
    1500,
    3000,
    5000,
    10000,
    20000,
    30000,
)

BASE_SUMMARY_FIELDS = [
    "dataset",
    "out_tag",
    "event_score_mode",
    "node_pool_score_mode",
    "expected_event_alert_budget",
    "expected_alert_horizon_events",
    "test_events",
    "events_scored",
    "event_tp",
    "event_fp",
    "event_count",
    "event_ratio",
    "event_precision",
    "node_pool_topk_json",
    "runtime_elapsed_seconds",
    "throughput_events_per_second",
    "timing_train_seconds",
    "timing_validation_seconds",
    "timing_test_scoring_seconds",
    "timing_label_attach_seconds",
    "timing_csv_write_seconds",
    "timing_stream_csv_write_seconds",
    "timing_post_stream_output_seconds",
    "rss_after_test_cleanup_mb",
    "rss_test_peak_mb",
    "process_peak_rss_mb",
    "state_slots",
    "state_array_mb",
    "cache_hit",
    "db_stream_mode",
    "db_stream_mode_requested",
    "db_stream_mode_actual",
    "db_stream_mode_fallback_reason",
    "split_source",
    "slim_split_override",
    "slim_split_override_applied",
    "year_month",
    "train_days",
    "validation_days",
    "test_days",
]

NODE_TOPK_SUMMARY_FIELDS = [
    field
    for topk in COMMON_NODE_TOPK_VALUES
    for field in (
        f"node_top{topk}_tp",
        f"node_top{topk}_fp",
        f"node_top{topk}_count",
        f"node_top{topk}_alert_count_at_k",
        f"node_top{topk}_ratio",
        f"node_top{topk}_precision",
    )
]

SUMMARY_FIELDS = BASE_SUMMARY_FIELDS + NODE_TOPK_SUMMARY_FIELDS


@dataclass(frozen=True)
class SummaryResult:
    """Counters describing one ablation summary write."""

    rows_written: int
    files_seen: int
    skipped_invalid_json_count: int
    output_csv: Path


def summarize(
    result_root: Path | str,
    output_csv: Path | str,
    glob_pattern: str = DEFAULT_GLOB,
    strict_json: bool = False,
) -> SummaryResult:
    """Write one CSV row per slim ablation eval JSON under result_root."""
    root = Path(result_root)
    output_path = Path(output_csv)
    rows: list[dict[str, Any]] = []
    skipped_invalid = 0
    output_resolved = _resolve_for_compare(output_path)
    paths = [
        path
        for path in _eval_json_paths(root, glob_pattern)
        if _resolve_for_compare(path) != output_resolved
    ]

    for path in paths:
        try:
            payload = _load_payload(path)
        except ValueError as exc:
            if strict_json:
                raise
            skipped_invalid += 1
            LOGGER.warning("Skipping invalid eval JSON %s: %s", path, exc)
            continue
        rows.append(_row_from_payload(payload, path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info(
        "Wrote %d summary rows to %s; skipped_invalid_json_count=%d",
        len(rows),
        output_path,
        skipped_invalid,
    )
    return SummaryResult(
        rows_written=len(rows),
        files_seen=len(paths),
        skipped_invalid_json_count=skipped_invalid,
        output_csv=output_path,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the ablation summary script."""
    parser = argparse.ArgumentParser(
        description="Summarize causal_semantics_slim ablation eval JSON files.",
    )
    parser.add_argument("--result_root", required=True)
    parser.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help=(
            "Glob pattern relative to result_root. Defaults to slim ablation "
            f"result directories containing {EVAL_JSON_NAME}."
        ),
    )
    parser.add_argument(
        "--output_csv",
        default="",
        help="Output CSV path. Defaults to result_root/ablation_summary.csv.",
    )
    parser.add_argument(
        "--strict_json",
        action="store_true",
        help="Fail on invalid eval JSON instead of warning and skipping it.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ablation summary CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    args = parse_args(argv)
    result_root = Path(args.result_root)
    output_csv = Path(args.output_csv) if args.output_csv else result_root / "ablation_summary.csv"
    summarize(
        result_root,
        output_csv,
        glob_pattern=args.glob,
        strict_json=bool(args.strict_json),
    )
    return 0


def _eval_json_paths(result_root: Path, glob_pattern: str) -> list[Path]:
    pattern_path = Path(glob_pattern)
    if pattern_path.is_absolute():
        search_pattern = pattern_path
    else:
        search_pattern = result_root / glob_pattern
    paths = [Path(path) for path in glob(str(search_pattern), recursive=True)]
    return sorted(path for path in paths if path.is_file())


def _resolve_for_compare(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _load_payload(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(payload, Mapping):
        raise ValueError("eval JSON root must be an object")
    return payload


def _row_from_payload(payload: Mapping[str, Any], path: Path) -> dict[str, Any]:
    config = _mapping(payload.get("config"))
    metrics = _mapping(payload.get("primary_online_metrics"))
    event = _mapping(metrics.get("event_alerts"))
    node_topk = _mapping(metrics.get("node_pool_topk"))
    runtime = _mapping(payload.get("runtime"))
    timing = _mapping(payload.get("timing"))
    memory = _mapping(payload.get("memory"))
    cache = _mapping(payload.get("cache"))
    events_scored = _number(runtime, "events_scored")
    event_ratio = _metric_ratio(event)

    row: dict[str, Any] = {
        "dataset": _first_present(payload.get("dataset"), config.get("dataset")),
        "out_tag": _first_present(
            payload.get("out_tag"),
            config.get("out_tag"),
            path.parent.name,
        ),
        "event_score_mode": _first_present(
            payload.get("event_score_mode"),
            config.get("event_score_mode"),
        ),
        "node_pool_score_mode": _first_present(
            payload.get("node_pool_score_mode"),
            config.get("node_pool_score_mode"),
        ),
        "expected_event_alert_budget": _first_present(
            payload.get("expected_event_alert_budget"),
            config.get("expected_event_alert_budget"),
            0,
        ),
        "expected_alert_horizon_events": _first_present(
            payload.get("expected_alert_horizon_events"),
            config.get("expected_alert_horizon_events"),
            0,
        ),
        "test_events": _first_present(payload.get("test_events"), events_scored, 0),
        "events_scored": events_scored,
        "event_tp": _number(event, "tp"),
        "event_fp": _number(event, "fp"),
        "event_count": _number(event, "count"),
        "event_ratio": event_ratio,
        "event_precision": _first_present(event.get("precision"), event_ratio, 0),
        "node_pool_topk_json": _json_string(node_topk),
        "runtime_elapsed_seconds": _number(runtime, "elapsed_seconds"),
        "throughput_events_per_second": _number(
            runtime,
            "throughput_events_per_second",
        ),
        "timing_train_seconds": _number(timing, "train_seconds"),
        "timing_validation_seconds": _number(timing, "validation_seconds"),
        "timing_test_scoring_seconds": _number(timing, "test_scoring_seconds"),
        "timing_label_attach_seconds": _number(timing, "label_attach_seconds"),
        "timing_csv_write_seconds": _number(timing, "csv_write_seconds"),
        "timing_stream_csv_write_seconds": _number(
            timing,
            "stream_csv_write_seconds",
        ),
        "timing_post_stream_output_seconds": _number(
            timing,
            "post_stream_output_seconds",
        ),
        "rss_after_test_cleanup_mb": _number(memory, "rss_after_test_cleanup_mb"),
        "rss_test_peak_mb": _number(memory, "rss_test_peak_mb"),
        "process_peak_rss_mb": _number(memory, "process_peak_rss_mb"),
        "state_slots": _number(memory, "state_slots"),
        "state_array_mb": _number(memory, "state_array_mb"),
        "cache_hit": _first_present(cache.get("hit"), ""),
        "db_stream_mode": _first_present(
            payload.get("db_stream_mode"),
            payload.get("db_stream_mode_actual"),
        ),
        "db_stream_mode_requested": _first_present(
            payload.get("db_stream_mode_requested"),
            config.get("db_stream_mode"),
        ),
        "db_stream_mode_actual": _first_present(
            payload.get("db_stream_mode_actual"),
            payload.get("db_stream_mode"),
        ),
        "db_stream_mode_fallback_reason": _first_present(
            payload.get("db_stream_mode_fallback_reason"),
            payload.get("db_stream_fallback_reason"),
        ),
        "split_source": _first_present(payload.get("split_source")),
        "slim_split_override": _first_present(
            payload.get("slim_split_override"),
            config.get("slim_split_override"),
        ),
        "slim_split_override_applied": _first_present(
            payload.get("slim_split_override_applied"),
        ),
        "year_month": _first_present(payload.get("year_month")),
        "train_days": _json_string(payload.get("train_days", [])),
        "validation_days": _json_string(payload.get("validation_days", [])),
        "test_days": _json_string(payload.get("test_days", [])),
    }
    row.update(_node_topk_columns(node_topk))
    return row


def _node_topk_columns(node_topk: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for topk in COMMON_NODE_TOPK_VALUES:
        metrics = _mapping(node_topk.get(f"top{topk}"))
        ratio = _metric_ratio(metrics)
        row.update(
            {
                f"node_top{topk}_tp": _number(metrics, "tp"),
                f"node_top{topk}_fp": _number(metrics, "fp"),
                f"node_top{topk}_count": _number(metrics, "count"),
                f"node_top{topk}_alert_count_at_k": _number(
                    metrics,
                    "alert_count_at_k",
                ),
                f"node_top{topk}_ratio": ratio,
                f"node_top{topk}_precision": _first_present(
                    metrics.get("precision"),
                    ratio,
                    0,
                ),
            }
        )
    return row


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _number(values: Mapping[str, Any], key: str, default: Any = 0) -> Any:
    value = values.get(key, default)
    if value is None or value == "":
        return default
    return value


def _metric_ratio(values: Mapping[str, Any]) -> Any:
    return _first_present(values.get("ratio"), values.get("precision"), 0)


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return ""


def _json_string(value: Any) -> str:
    if value is None:
        value = {}
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
