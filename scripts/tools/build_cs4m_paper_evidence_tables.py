#!/usr/bin/env python3
"""Build read-only paper evidence tables for the CS4M manuscript."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


K_VALUES = (100, 200, 500, 1000, 3000, 5000)


def format_pair(tp: int | None, fp: int | None) -> str:
    """Return a paper-table TP/FP pair or a recompute marker."""
    if tp is None or fp is None:
        return "needs_recompute"
    return f"{tp} / {fp}"


def metric_status_for_clearscope_e5(default_topk_tp: int | None, audit_node_tp: int | None) -> str:
    """Return ClearScope E5 status when default top-k and audit rows disagree."""
    if default_topk_tp is None or audit_node_tp is None:
        return "needs_recompute_51_gt"
    if default_topk_tp != audit_node_tp:
        return "needs_recompute_51_gt_source_conflict"
    return "locked"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ratio_pair(data: dict[str, Any], hit_key: str, total_key: str) -> str:
    hit = data.get(hit_key)
    total = data.get(total_key)
    if hit is None or total is None:
        return "missing"
    return f"{hit}/{total}"


def extract_optc_canonical_summary(summary_path: Path) -> dict[str, str]:
    """Extract OPTC_051 canonical-aware evaluation metrics from a summary JSON."""
    payload = _load_json(summary_path)
    canonical = payload.get("canonical_aware", {})
    events = payload.get("event_alerts_by_canonical_gt", {})
    by_kind = canonical.get("by_kind", {})

    def kind_pair(kind: str) -> str:
        row = by_kind.get(kind, {})
        hit = row.get("hit")
        total = row.get("total")
        if hit is None or total is None:
            return "missing"
        return f"{hit}/{total}"

    return {
        "dataset": str(payload.get("dataset", "OPTC_051")),
        "original_gt_hit": _ratio_pair(canonical, "original_gt_hit", "original_gt_total"),
        "unique_compact_gt_hit": _ratio_pair(
            canonical,
            "unique_compact_gt_hit",
            "unique_compact_gt_total",
        ),
        "netflow_gt_hit": kind_pair("netflow"),
        "process_gt_hit": kind_pair("process"),
        "file_gt_hit": kind_pair("file"),
        "event_tp_fp": format_pair(events.get("tp"), events.get("fp")),
        "total_alerts": str(events.get("alerts", "missing")),
        "source": str(summary_path),
        "status": "locked_post_stream_canonical_eval",
    }


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _manual_topk_rows() -> list[dict[str, str]]:
    values: dict[str, dict[str, tuple[int | None, int | None]]] = {
        "THEIA_E3": {
            "top100": (34, 66),
            "top200": (42, 158),
            "top500": (42, 458),
            "top1000": (42, 958),
            "top3000": (42, 2958),
            "top5000": (42, 4958),
            "full_pool": (54, 28692),
        },
        "CADETS_E3": {
            "top100": (32, 68),
            "top200": (32, 168),
            "top500": (34, 466),
            "top1000": (34, 966),
            "top3000": (43, 2957),
            "top5000": (47, 4953),
            "full_pool": (53, 6050),
        },
        "CLEARSCOPE_E3": {
            "top100": (1, 99),
            "top200": (1, 199),
            "top500": (3, 497),
            "top1000": (13, 987),
            "top3000": (40, 2960),
            "top5000": (40, 3297),
            "full_pool": (40, 3297),
        },
        "CLEARSCOPE_E5": {
            "top100": (None, None),
            "top200": (None, None),
            "top500": (None, None),
            "top1000": (None, None),
            "top3000": (None, None),
            "top5000": (None, None),
            "full_pool": (14, 377),
        },
        "THEIA_E5": {
            "top100": (2, 98),
            "top200": (2, 198),
            "top500": (3, 497),
            "top1000": (10, 751),
            "top3000": (10, 751),
            "top5000": (10, 751),
            "full_pool": (10, 751),
        },
        "OPTC_051": {
            "top100": (3, 97),
            "top200": (3, 197),
            "top500": (17, 377),
            "top1000": (17, 377),
            "top3000": (17, 377),
            "top5000": (17, 377),
            "full_pool": (17, 377),
        },
    }
    rows: list[dict[str, str]] = []
    for dataset, row in values.items():
        status = "locked_user_provided"
        if dataset == "CLEARSCOPE_E5":
            status = metric_status_for_clearscope_e5(default_topk_tp=0, audit_node_tp=14)
        if dataset == "OPTC_051":
            status = "locked_from_node_topk_metrics_and_canonical_audit"
        rows.append(
            {
                "dataset": dataset,
                "top100_tp_fp": format_pair(*row["top100"]),
                "top200_tp_fp": format_pair(*row["top200"]),
                "top500_tp_fp": format_pair(*row["top500"]),
                "top1000_tp_fp": format_pair(*row["top1000"]),
                "top3000_tp_fp": format_pair(*row["top3000"]),
                "top5000_tp_fp": format_pair(*row["top5000"]),
                "full_pool_tp_fp": format_pair(*row["full_pool"]),
                "status": status,
            }
        )
    return rows


def _runtime_rows() -> list[dict[str, str]]:
    return [
        {
            "dataset": "THEIA_E3",
            "selected_source": "THEIA_E3_PHASE3E_E4_NONE",
            "selection_reason": "current best",
            "event_tp_fp": "152075 / 15311",
            "node_tp_fp": "54 / 28692",
            "online_primary_rss_mb": "154.21",
            "speed_events_s": "3450.39",
            "speed_source": "deploy/test scoring",
            "status": "locked_user_provided",
        },
        {
            "dataset": "CADETS_E3",
            "selected_source": "CADETS_E3_PHASE3E_E4_NONE",
            "selection_reason": "best node top-k TP",
            "event_tp_fp": "2397 / 8992",
            "node_tp_fp": "53 / 6050",
            "online_primary_rss_mb": "89.92",
            "speed_events_s": "3897.52",
            "speed_source": "recorded E4 deploy/test scoring",
            "status": "locked_user_provided",
        },
        {
            "dataset": "CLEARSCOPE_E3",
            "selected_source": "V31...FP_GUARD_V2_AGGRESSIVE",
            "selection_reason": "best full-pool FP among 40 TP",
            "event_tp_fp": "413 / 9028",
            "node_tp_fp": "40 / 3297",
            "online_primary_rss_mb": "73.94",
            "speed_events_s": "3543.37",
            "speed_source": "online minimal",
            "status": "locked_user_provided",
        },
        {
            "dataset": "CLEARSCOPE_E5",
            "selected_source": "V33B_DUAL_CHANNEL_NODE_SUPPORT_FULL audit",
            "selection_reason": "dual-channel union",
            "event_tp_fp": "15 / 394",
            "node_tp_fp": "14 / 377",
            "online_primary_rss_mb": "103.95",
            "speed_events_s": "3363.85",
            "speed_source": "online minimal",
            "status": "needs_recompute_51_gt_source_conflict_for_topk",
        },
        {
            "dataset": "THEIA_E5",
            "selected_source": "theia_v5c0_plus_policy provided",
            "selection_reason": "provided result",
            "event_tp_fp": "11 / 1300",
            "node_tp_fp": "10 / 751",
            "online_primary_rss_mb": "152.93",
            "speed_events_s": "6291.01 / 6481.20",
            "speed_source": "log / profiling",
            "status": "locked_user_provided",
        },
        {
            "dataset": "OPTC_051",
            "selected_source": "OPTC_051_OPTC_WINDOWS_V1_3C_DETAIL_FULL_INFER",
            "selection_reason": "current generated Windows result",
            "event_tp_fp": "208 / 1158",
            "node_tp_fp": "17 / 377",
            "online_primary_rss_mb": "see_metric_provenance",
            "speed_events_s": "see_metric_provenance",
            "speed_source": "result artifacts",
            "status": "locked_canonical_event_eval_node_topk",
        },
    ]


def build_evidence_lock(output_dir: Path) -> None:
    """Write CS4M paper evidence-lock CSV and Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    optc_summary_path = Path(
        "outputs/diagnostics/optc_windows_v13c_canonical_eval_audit_20260615_224932/summary.json"
    )
    optc_row = extract_optc_canonical_summary(optc_summary_path)

    topk_rows = _manual_topk_rows()
    runtime_rows = _runtime_rows()
    _write_csv(
        output_dir / "evidence_lock_topk_table.csv",
        topk_rows,
        [
            "dataset",
            "top100_tp_fp",
            "top200_tp_fp",
            "top500_tp_fp",
            "top1000_tp_fp",
            "top3000_tp_fp",
            "top5000_tp_fp",
            "full_pool_tp_fp",
            "status",
        ],
    )
    _write_csv(
        output_dir / "evidence_lock_runtime_table.csv",
        runtime_rows,
        [
            "dataset",
            "selected_source",
            "selection_reason",
            "event_tp_fp",
            "node_tp_fp",
            "online_primary_rss_mb",
            "speed_events_s",
            "speed_source",
            "status",
        ],
    )
    _write_csv(
        output_dir / "evidence_lock_optc051_summary.csv",
        [optc_row],
        [
            "dataset",
            "original_gt_hit",
            "unique_compact_gt_hit",
            "netflow_gt_hit",
            "process_gt_hit",
            "file_gt_hit",
            "event_tp_fp",
            "total_alerts",
            "source",
            "status",
        ],
    )

    provenance = output_dir / "evidence_lock_metric_provenance.md"
    provenance.write_text(
        "\n".join(
            [
                "# CS4M Paper Evidence Lock Metric Provenance",
                "",
                "GT is post-stream evaluation only. These tables do not define runtime rules.",
                "",
                "## Source Status",
                "",
                "- THEIA_E3, CADETS_E3, CLEARSCOPE_E3, THEIA_E5: values are user-provided",
                "  paper-table locks and must be traced to final result files before final LaTeX.",
                "- OPTC_051: canonical-aware event and GT-kind metrics are extracted from",
                f"  `{optc_summary_path}`; node top-k values come from the generated",
                "  `node_topk_metrics.csv` in the OPTC v1.3c full infer result.",
                "- CLEARSCOPE_E5: status is",
                "  `needs_recompute_51_gt_source_conflict` because the user-provided",
                "  dual-channel audit row reports 14 / 377 node TP/FP, while the default",
                "  result-directory `node_topk_metrics.csv` currently reports zero TP under",
                "  its current mapping.",
                "",
                "## Required Before Final Manuscript",
                "",
                "1. Recompute CLEARSCOPE_E5 top-k under the 51-GT canonical-aware mapping.",
                "2. Replace user-provided locked rows with exact source-file extraction where",
                "   possible, or keep them labeled as user-provided table locks.",
                "3. Use OPTC_051 original GT and unique compact GT denominators separately.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = output_dir / "evidence_lock_metrics.md"
    metrics.write_text(
        "\n".join(
            [
                "# CS4M Paper Evidence Lock Metrics",
                "",
                "## Locked Outputs",
                "",
                "- `evidence_lock_topk_table.csv`",
                "- `evidence_lock_runtime_table.csv`",
                "- `evidence_lock_optc051_summary.csv`",
                "- `evidence_lock_metric_provenance.md`",
                "",
                "## OPTC_051 Summary",
                "",
                f"- Original GT hit: `{optc_row['original_gt_hit']}`",
                f"- Unique compact GT hit: `{optc_row['unique_compact_gt_hit']}`",
                f"- Netflow GT hit: `{optc_row['netflow_gt_hit']}`",
                f"- Process GT hit: `{optc_row['process_gt_hit']}`",
                f"- File GT hit: `{optc_row['file_gt_hit']}`",
                f"- Event TP/FP: `{optc_row['event_tp_fp']}`",
                f"- Total alerts: `{optc_row['total_alerts']}`",
                "",
                "## ClearScope E5 Risk",
                "",
                "ClearScope E5 top-k remains unresolved for the paper table. Keep the",
                "`needs_recompute_51_gt_source_conflict` status until a 51-GT top-k",
                "canonical-aware table is generated.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    """CLI entrypoint for evidence-lock generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="paper_rewriting_output")
    args = parser.parse_args()
    build_evidence_lock(Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
