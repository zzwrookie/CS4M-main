#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _get(mapping: Mapping[str, Any], path: str, default: Any = "") -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def summarize(result_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(result_root.glob("*/eval_causal_semantics_slim.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        config = payload.get("config", {})
        event_metrics = _get(payload, "primary_online_metrics.event_alerts", {})
        memory = payload.get("memory", {})
        runtime = payload.get("runtime", {})
        controller = payload.get("threshold_controller", {})
        rows.append(
            {
                "dataset": config.get("dataset", payload.get("dataset", "")),
                "out_tag": config.get("out_tag", path.parent.name),
                "event_score_mode": config.get("event_score_mode", ""),
                "context": _get(payload, "sspm_context.context_mode", ""),
                "embedding": _get(payload, "semantic_embedding.method", ""),
                "threshold": config.get("event_threshold_mode", ""),
                "event_tp": event_metrics.get("tp", ""),
                "event_fp": event_metrics.get("fp", ""),
                "event_precision": event_metrics.get("precision", ""),
                "rss_peak_mb": memory.get("rss_test_peak_mb", ""),
                "throughput": runtime.get("throughput_events_per_second", ""),
                "final_threshold": controller.get("final_threshold", ""),
            },
        )
    return rows


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "dataset",
        "out_tag",
        "event_score_mode",
        "context",
        "embedding",
        "threshold",
        "event_tp",
        "event_fp",
        "event_precision",
        "rss_peak_mb",
        "throughput",
        "final_threshold",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(header, "")) for header in headers) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="outputs/results/tflr_light")
    parser.add_argument("--out_md", default="outputs/results/cs4m/residual_ablation_summary.md")
    args = parser.parse_args()
    rows = summarize(Path(args.result_root))
    write_markdown(rows, Path(args.out_md))
    print(f"wrote {args.out_md} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
