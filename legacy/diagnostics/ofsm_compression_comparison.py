#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


COMPARISON_FIELDS = [
    "run_name",
    "target_mode",
    "train_backend",
    "infer_backend",
    "state_model",
    "state_merge_mode",
    "state_merge_threshold",
    "logical_node_count",
    "num_physical_states",
    "num_singleton_states",
    "num_clusters",
    "num_clustered_nodes",
    "compression_ratio",
    "estimated_state_mb_no_merge",
    "estimated_state_mb_after_merge",
    "estimated_state_mb_saved",
    "rss_peak_mb",
    "rss_final_mb",
    "events_per_sec",
    "deploy_rss_valid",
    "deploy_infer_rss_peak_mb",
    "deploy_infer_rss_final_mb",
    "deploy_infer_events_per_sec",
    "train_rss_peak_mb",
    "train_events_seen",
    "train_epochs_completed",
    "checkpoint_path",
    "node_embedding_path",
    "action_embedding_path",
    "event_index_path",
    "x_context_path",
    "skip_train",
    "baseline_run_name",
    "baseline_num_physical_states",
    "physical_states_saved",
    "physical_states_saved_pct",
    "baseline_deploy_infer_rss_peak_mb",
    "deploy_rss_peak_saved_mb",
    "deploy_rss_peak_saved_pct",
    "baseline_rss_peak_mb",
    "rss_peak_saved_mb",
    "rss_peak_saved_pct",
    "event_alert_count",
    "event_tp",
    "event_fp",
    "event_precision",
    "event_recall",
]


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _warn(message: str) -> None:
    print(f"warning: {message}", file=sys.stderr)


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _event_precision(event_metrics: Mapping[str, Any]) -> Any:
    precision = event_metrics.get("precision", event_metrics.get("ratio", ""))
    if precision != "":
        return precision
    tp = _safe_float(event_metrics.get("tp"))
    fp = _safe_float(event_metrics.get("fp"))
    if tp is None or fp is None:
        return ""
    total = tp + fp
    return "" if total <= 0.0 else tp / total


def _compression_from_metrics(metrics: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    compression = metrics.get("ofsm_compression")
    if isinstance(compression, Mapping):
        data = dict(compression)
    else:
        data = {}
        fallback = _nested(metrics, "eval", "state_merge")
        if isinstance(fallback, Mapping):
            data.update(dict(fallback))
        _warn(f"{run_dir}: metrics.json missing ofsm_compression; using available fallbacks")

    memory = _nested(metrics, "eval", "memory")
    runtime = _nested(metrics, "eval", "runtime")
    event_metrics = _nested(metrics, "eval", "primary_online_metrics", "event_alerts")
    if not isinstance(memory, Mapping):
        memory = {}
    if not isinstance(runtime, Mapping):
        runtime = {}
    if not isinstance(event_metrics, Mapping):
        event_metrics = {}
    sspm = metrics.get("sspm")
    if not isinstance(sspm, Mapping):
        sspm = {}
    phase3d = metrics.get("phase3d")
    if not isinstance(phase3d, Mapping):
        phase3d = {}
    phase3e = metrics.get("phase3e")
    if not isinstance(phase3e, Mapping):
        phase3e = {}
    training = _nested(metrics, "eval", "sspm_training")
    if not isinstance(training, Mapping):
        training = _nested(metrics, "eval", "runtime", "sspm_training")
    if not isinstance(training, Mapping):
        training = {}
    cache = _nested(metrics, "eval", "cache")
    if not isinstance(cache, Mapping):
        cache = {}

    run_name = str(metrics.get("out_tag") or run_dir.name)
    state_dim = _safe_int(data.get("state_dim")) or _safe_int(
        _nested(metrics, "sspm", "state_dim"),
    ) or 64
    logical_count = _safe_int(data.get("logical_node_count"))
    physical_count = _safe_int(data.get("num_physical_states"))
    if logical_count is None:
        _warn(f"{run_dir}: missing logical_node_count")
    if physical_count is None:
        _warn(f"{run_dir}: missing num_physical_states")
    logical_value = int(logical_count or 0)
    physical_value = int(physical_count or 0)
    no_merge = data.get("estimated_state_mb_no_merge")
    after_merge = data.get("estimated_state_mb_after_merge")
    saved = data.get("estimated_state_mb_saved")
    if no_merge is None:
        no_merge = float(logical_value * state_dim * 4) / (1024.0 * 1024.0)
    if after_merge is None:
        after_merge = float(physical_value * state_dim * 4) / (1024.0 * 1024.0)
    if saved is None:
        saved = float(no_merge) - float(after_merge)

    rss_peak = data.get("rss_peak_mb")
    if rss_peak is None:
        rss_peak = memory.get("rss_test_peak_mb", memory.get("process_peak_rss_mb", ""))
        if rss_peak == "":
            _warn(f"{run_dir}: missing rss_peak_mb")
    rss_final = data.get("rss_final_mb")
    if rss_final is None:
        rss_final = memory.get("rss_after_label_attach_mb", memory.get("rss_after_test_cleanup_mb", ""))
        if rss_final == "":
            _warn(f"{run_dir}: missing rss_final_mb")

    return {
        "run_name": run_name,
        "target_mode": phase3e.get("target_mode", ""),
        "train_backend": phase3e.get("train_backend", ""),
        "infer_backend": phase3e.get("infer_backend", ""),
        "state_model": str(sspm.get("state_model", "")),
        "state_merge_mode": str(data.get("state_merge_mode", "")),
        "state_merge_threshold": data.get(
            "state_merge_threshold",
            sspm.get("state_merge_threshold", ""),
        ),
        "logical_node_count": "" if logical_count is None else logical_value,
        "num_physical_states": "" if physical_count is None else physical_value,
        "num_singleton_states": data.get("num_singleton_states", ""),
        "num_clusters": data.get("num_clusters", ""),
        "num_clustered_nodes": data.get("num_clustered_nodes", ""),
        "compression_ratio": data.get("compression_ratio", ""),
        "estimated_state_mb_no_merge": no_merge,
        "estimated_state_mb_after_merge": after_merge,
        "estimated_state_mb_saved": saved,
        "rss_peak_mb": rss_peak,
        "rss_final_mb": rss_final,
        "events_per_sec": data.get(
            "events_per_sec",
            runtime.get("throughput_events_per_second", ""),
        ),
        "deploy_rss_valid": data.get("deploy_rss_valid", ""),
        "deploy_infer_rss_peak_mb": data.get("deploy_infer_rss_peak_mb", rss_peak),
        "deploy_infer_rss_final_mb": data.get("deploy_infer_rss_final_mb", rss_final),
        "deploy_infer_events_per_sec": data.get(
            "deploy_infer_events_per_sec",
            runtime.get("throughput_events_per_second", ""),
        ),
        "train_rss_peak_mb": training.get("train_rss_peak_mb", ""),
        "train_events_seen": training.get(
            "train_events_seen_total",
            training.get("train_events_actual", ""),
        ),
        "train_epochs_completed": training.get(
            "train_epochs_completed",
            training.get("epochs_completed", ""),
        ),
        "checkpoint_path": cache.get(
            "checkpoint_path",
            phase3d.get("sspm_checkpoint_path", ""),
        ),
        "node_embedding_path": phase3e.get("node_embedding_path", ""),
        "action_embedding_path": phase3e.get("action_embedding_path", ""),
        "event_index_path": phase3e.get("event_index_path", ""),
        "x_context_path": phase3e.get("x_context_path", ""),
        "skip_train": phase3d.get("skip_train", cache.get("skip_train", "")),
        "event_alert_count": event_metrics.get("count", ""),
        "event_tp": event_metrics.get("tp", ""),
        "event_fp": event_metrics.get("fp", ""),
        "event_precision": _event_precision(event_metrics),
        "event_recall": event_metrics.get("recall", ""),
    }


def _run_dirs_from_args(args: argparse.Namespace) -> list[Path]:
    if args.run_dirs:
        return [Path(value) for value in args.run_dirs]
    if not args.result_root or not args.run_tags:
        raise ValueError("provide either --run_dirs or both --result_root and --run_tags")
    root = Path(args.result_root)
    return [root / str(tag) for tag in args.run_tags]


def _load_run_row(run_dir: Path) -> dict[str, Any]:
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.exists():
        _warn(f"{run_dir}: metrics.json not found; writing empty row fields")
        return {
            "run_name": Path(run_dir).name,
            "state_merge_mode": "",
        }
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{metrics_path} must contain a JSON object")
    return _compression_from_metrics(metrics, Path(run_dir))


def _matches_baseline(
    row: Mapping[str, Any],
    baseline_mode: str,
    baseline_tag: str,
) -> bool:
    if baseline_tag and str(row.get("run_name", "")) == str(baseline_tag):
        return True
    return str(row.get("state_merge_mode", "")) == str(baseline_mode)


def _baseline_key(row: Mapping[str, Any], comparison_kind: str) -> str:
    if comparison_kind == "model_ablation":
        model = str(row.get("state_model", ""))
        return f"{model}:none"
    return "global:none"


def _is_baseline_row(row: Mapping[str, Any], comparison_kind: str) -> bool:
    if str(row.get("state_merge_mode", "")) != "none":
        return False
    if comparison_kind == "model_ablation":
        return bool(str(row.get("state_model", "")))
    return True


def _format_number(value: Any) -> Any:
    parsed = _safe_float(value)
    if parsed is None:
        return value
    if abs(parsed - round(parsed)) < 1e-12:
        return int(round(parsed))
    return parsed


def _rows_with_baseline_deltas(
    rows: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, Mapping[str, Any]],
    comparison_kind: str,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        key = _baseline_key(row, comparison_kind)
        baseline = baselines.get(key)
        if baseline is None:
            raise ValueError(f"baseline run not found for {key}")
        baseline_name = str(baseline.get("run_name", ""))
        baseline_physical = _safe_float(baseline.get("num_physical_states"))
        physical = _safe_float(row.get("num_physical_states"))
        baseline_rss = _safe_float(baseline.get("rss_peak_mb"))
        rss_peak = _safe_float(row.get("rss_peak_mb"))
        baseline_deploy_rss = _safe_float(baseline.get("deploy_infer_rss_peak_mb"))
        deploy_rss = _safe_float(row.get("deploy_infer_rss_peak_mb"))
        physical_saved = (
            "" if baseline_physical is None or physical is None else baseline_physical - physical
        )
        physical_saved_pct = (
            ""
            if physical_saved == "" or baseline_physical is None
            else float(physical_saved) / max(float(baseline_physical), 1.0) * 100.0
        )
        rss_saved = "" if baseline_rss is None or rss_peak is None else baseline_rss - rss_peak
        rss_saved_pct = (
            ""
            if rss_saved == "" or baseline_rss is None
            else float(rss_saved) / max(float(baseline_rss), 1e-6) * 100.0
        )
        deploy_saved = (
            ""
            if baseline_deploy_rss is None or deploy_rss is None
            else baseline_deploy_rss - deploy_rss
        )
        deploy_saved_pct = (
            ""
            if deploy_saved == "" or baseline_deploy_rss is None
            else float(deploy_saved) / max(float(baseline_deploy_rss), 1e-6) * 100.0
        )
        out_rows.append(
            {
                **{field: row.get(field, "") for field in COMPARISON_FIELDS},
                "baseline_run_name": baseline_name,
                "baseline_num_physical_states": (
                    "" if baseline_physical is None else baseline_physical
                ),
                "physical_states_saved": physical_saved,
                "physical_states_saved_pct": physical_saved_pct,
                "baseline_deploy_infer_rss_peak_mb": (
                    "" if baseline_deploy_rss is None else baseline_deploy_rss
                ),
                "deploy_rss_peak_saved_mb": deploy_saved,
                "deploy_rss_peak_saved_pct": deploy_saved_pct,
                "baseline_rss_peak_mb": "" if baseline_rss is None else baseline_rss,
                "rss_peak_saved_mb": rss_saved,
                "rss_peak_saved_pct": rss_saved_pct,
            },
        )
    return out_rows


def _write_rows(output_csv: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_number(row.get(field, "")) for field in COMPARISON_FIELDS})


def write_phase3c_comparison_csv(
    run_dirs: Sequence[str | Path],
    comparison_kind: str,
    output_csv: str | Path,
) -> list[dict[str, Any]]:
    if comparison_kind not in {"e2_ofsm", "model_ablation", "e2_threshold"}:
        raise ValueError("comparison_kind must be e2_ofsm, model_ablation, or e2_threshold")
    rows = [_load_run_row(Path(run_dir)) for run_dir in run_dirs]
    baselines = {
        _baseline_key(row, comparison_kind): row
        for row in rows
        if _is_baseline_row(row, comparison_kind)
    }
    out_rows = _rows_with_baseline_deltas(rows, baselines, comparison_kind)
    _write_rows(output_csv, out_rows)
    return out_rows


def write_comparison_csv(
    run_dirs: Sequence[str | Path],
    baseline_mode: str,
    baseline_tag: str,
    output_csv: str | Path,
) -> list[dict[str, Any]]:
    rows = [_load_run_row(Path(run_dir)) for run_dir in run_dirs]
    baseline = next(
        (row for row in rows if _matches_baseline(row, baseline_mode, baseline_tag)),
        None,
    )
    if baseline is None:
        raise ValueError("baseline run not found; provide --baseline_tag or --baseline_mode none")

    out_rows = _rows_with_baseline_deltas(
        rows,
        {"global:none": baseline},
        comparison_kind="generic",
    )
    _write_rows(output_csv, out_rows)
    return out_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OFSM compression comparison CSV.")
    parser.add_argument("--result_root", default="")
    parser.add_argument("--run_tags", nargs="*", default=[])
    parser.add_argument("--run_dirs", nargs="*", default=[])
    parser.add_argument("--baseline_tag", default="")
    parser.add_argument("--baseline_mode", default="none")
    parser.add_argument(
        "--comparison_kind",
        choices=("generic", "e2_ofsm", "model_ablation", "e2_threshold"),
        default="generic",
    )
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dirs = _run_dirs_from_args(args)
    if str(args.comparison_kind) == "generic":
        write_comparison_csv(
            run_dirs=run_dirs,
            baseline_mode=str(args.baseline_mode),
            baseline_tag=str(args.baseline_tag),
            output_csv=args.output_csv,
        )
    else:
        write_phase3c_comparison_csv(
            run_dirs=run_dirs,
            comparison_kind=str(args.comparison_kind),
            output_csv=args.output_csv,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
