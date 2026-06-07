#!/usr/bin/env python3
"""Audit whether existing online alert tails contain positives.

Purpose:
    Diagnose whether event, node, or episode online alert outputs already contain
    true positives in their high-score tail under fixed alert budgets.

Inputs:
    A causal-semantics result directory containing:
      - online_event_alerts.csv
      - online_node_alerts.csv
      - online_episode_alerts.csv
      - causal_semantics_ranked.csv
    If present, these diagnostic top-risk score traces are preferred:
      - online_event_score_trace.csv
      - online_node_score_trace.csv
      - online_episode_score_trace.csv

Outputs:
    JSON and Markdown summaries with per-layer budget metrics and a short
    recommendation about whether to tune event thresholds or focus on
    bounded-delay node/episode confirmation.

Pipeline status:
    Diagnostic only. It is not imported by the main detector and does not change
    thresholds, scores, alerts, or rankings.

How to test:
    python -m py_compile scripts/tools/diagnose_online_tail_reachability.py
    python scripts/tools/diagnose_online_tail_reachability.py \
      --result_dir outputs/results/cs4m/CLEARSCOPE_E3_CAUSAL_SEMANTICS_RESIDUAL_FUSION_CALIB_FULL

Runtime/memory:
    Reads online alert CSVs into memory. These are capped by alert limits and are
    small relative to full event streams. Reads only the first max(budgets) rows
    from the ranked CSV.

Leakage risk:
    Uses labels only after inference, for post-hoc diagnostic reporting. Do not
    use this script to choose final test thresholds. Use it only to decide
    whether another validation-calibrated experiment is worth running.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRUTHY = {"1", "true", "t", "yes", "y"}


@dataclass(frozen=True)
class LayerSpec:
    name: str
    filename: str
    trace_filename: str
    score_fields: tuple[str, ...]
    rank_field: str
    tp_fields: tuple[str, ...]
    suspect_fields: tuple[str, ...] = ()


LAYER_SPECS = (
    LayerSpec(
        name="event",
        filename="online_event_alerts.csv",
        trace_filename="online_event_score_trace.csv",
        score_fields=("online_risk", "event_score", "score"),
        rank_field="alert_rank",
        tp_fields=("is_correct_event_alert",),
        suspect_fields=("is_suspicious_event", "is_malicious_event"),
    ),
    LayerSpec(
        name="node",
        filename="online_node_alerts.csv",
        trace_filename="online_node_score_trace.csv",
        score_fields=("online_risk", "event_score", "score"),
        rank_field="alert_rank",
        tp_fields=("is_tp_after_attack_start", "is_tp_anytime", "is_gt_positive"),
        suspect_fields=("is_suspect",),
    ),
    LayerSpec(
        name="episode",
        filename="online_episode_alerts.csv",
        trace_filename="online_episode_score_trace.csv",
        score_fields=("online_risk", "event_score", "score"),
        rank_field="alert_rank",
        tp_fields=("is_tp_after_attack_start", "is_tp_any_endpoint"),
        suspect_fields=("src_is_suspect", "dst_is_suspect"),
    ),
)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY


def _safe_float(value: Any, default: float = float("-inf")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _first_present(row: dict[str, str], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        if field in row:
            return field
    return None


def _is_event_correct(row: dict[str, str]) -> bool:
    if "is_correct_event_alert" in row:
        return _truthy(row.get("is_correct_event_alert"))
    if "event_label" in row:
        return str(row.get("event_label", "")).strip() in {"1", "2"}
    if "label" in row:
        return str(row.get("label", "")).strip() in {"1", "2"}
    return False


def _is_tp(row: dict[str, str], spec: LayerSpec) -> bool:
    if spec.name == "event":
        return _is_event_correct(row)
    for field in spec.tp_fields:
        if field in row and _truthy(row.get(field)):
            return True
    return False


def _is_suspect(row: dict[str, str], spec: LayerSpec) -> bool:
    for field in spec.suspect_fields:
        if field in row and _truthy(row.get(field)):
            return True
    return False


def _read_csv(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows: list[dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _score_sorted_rows(rows: list[dict[str, str]], spec: LayerSpec) -> tuple[list[dict[str, str]], str | None]:
    score_field = _first_present(rows[0], spec.score_fields) if rows else None
    if score_field is None:
        return [], None
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda item: (_safe_float(item[1].get(score_field)), -item[0]), reverse=True)
    return [row for _idx, row in indexed], score_field


def _budget_metrics(rows: list[dict[str, str]], spec: LayerSpec, budgets: list[int], source: str) -> dict[str, Any]:
    sorted_rows, score_field = _score_sorted_rows(rows, spec)
    total_rows = len(sorted_rows)
    all_tp = sum(1 for row in sorted_rows if _is_tp(row, spec))
    all_suspect = sum(1 for row in sorted_rows if _is_suspect(row, spec))
    first_tp_rank = next((idx for idx, row in enumerate(sorted_rows, 1) if _is_tp(row, spec)), None)

    out: dict[str, Any] = {
        "csv_rows": total_rows,
        "source": source,
        "score_field": score_field,
        "tp_in_captured_alerts": all_tp,
        "suspect_or_relaxed_positive_in_captured_alerts": all_suspect,
        "first_tp_score_tail_rank": first_tp_rank,
        "budgets_are_exact_only_when_budget_leq_csv_rows": True,
        "source_note": "top-risk score trace" if source == "score_trace" else "already-emitted alerts only",
        "budget_metrics": [],
    }
    for budget in budgets:
        selected = sorted_rows[: min(int(budget), total_rows)]
        tp = sum(1 for row in selected if _is_tp(row, spec))
        suspect = sum(1 for row in selected if _is_suspect(row, spec))
        selected_count = len(selected)
        fp = selected_count - tp
        threshold = None
        if selected and score_field is not None:
            threshold = _safe_float(selected[-1].get(score_field), default=float("nan"))
        first_tp = next((idx for idx, row in enumerate(selected, 1) if _is_tp(row, spec)), None)
        out["budget_metrics"].append(
            {
                "budget": int(budget),
                "selected_alerts": selected_count,
                "exact_for_budget": total_rows >= int(budget),
                "threshold_at_budget_if_exact_else_captured_floor": threshold,
                "tp": tp,
                "fp": fp,
                "precision": (tp / selected_count) if selected_count else 0.0,
                "suspect_or_relaxed_positive": suspect,
                "first_tp_rank_within_budget": first_tp,
                "has_tp": tp > 0,
                "limitation": ""
                if total_rows >= int(budget)
                else (
                    "Only the configured top-risk score trace is available; increase --online_score_trace_top_n for larger exact budgets."
                    if source == "score_trace"
                    else "Only already-emitted alerts are available; lower-threshold alerts below the original threshold are not in this CSV."
                ),
            }
        )
    return out


def _ranked_metrics(result_dir: Path, budgets: list[int]) -> dict[str, Any]:
    path = result_dir / "causal_semantics_ranked.csv"
    max_rows = max(budgets) if budgets else 0
    rows = _read_csv(path, max_rows=max_rows)
    out: dict[str, Any] = {
        "filename": str(path),
        "rows_read": len(rows),
        "note": "Auxiliary forensic/post-stream node ranking. This is not an online threshold replay.",
        "budget_metrics": [],
    }
    for budget in budgets:
        selected = rows[: min(int(budget), len(rows))]
        gt = sum(1 for row in selected if _truthy(row.get("is_gt_positive")))
        suspect = sum(1 for row in selected if _truthy(row.get("is_suspect")))
        fp_strict = len(selected) - gt
        out["budget_metrics"].append(
            {
                "budget": int(budget),
                "selected_nodes": len(selected),
                "gt_positive_nodes": gt,
                "strict_fp_nodes": fp_strict,
                "precision": (gt / len(selected)) if selected else 0.0,
                "suspect_nodes": suspect,
                "has_gt": gt > 0,
            }
        )
    return out


def _dataset_name(result_dir: Path) -> str:
    name = result_dir.name
    for dataset in ("CLEARSCOPE_E3", "CADETS_E3", "THEIA_E3"):
        if name.startswith(dataset):
            return dataset
    return name.split("_")[0] if name else "unknown"


def _recommend(layers: dict[str, Any], ranked: dict[str, Any]) -> dict[str, Any]:
    event_tp = int(layers.get("event", {}).get("tp_in_captured_alerts", 0))
    node_tp = int(layers.get("node", {}).get("tp_in_captured_alerts", 0))
    episode_tp = int(layers.get("episode", {}).get("tp_in_captured_alerts", 0))
    ranked_has_gt = any(item.get("has_gt") for item in ranked.get("budget_metrics", []))

    if event_tp <= 0:
        event_action = (
            "Do not tune only the single-event threshold from these alert CSVs. "
            "The captured event tail has no true positives; lowering the threshold may mostly add false positives unless a full score trace shows positives below the original threshold."
        )
    else:
        event_action = (
            "The captured event tail already contains true positives. Event-threshold budget calibration may be worth testing, using validation-only thresholds."
        )

    if node_tp > 0 or episode_tp > 0:
        online_action = (
            "Node/episode tails contain true positives. Prefer bounded-delay episode/node confirmation as the online primary metric over forcing single-event alerts."
        )
    elif ranked_has_gt:
        online_action = (
            "Immediate node/episode alert tails do not contain positives, but the ranked CSV does. Investigate converting post-stream node aggregation into streaming bounded-delay node/episode risk before another full run."
        )
    else:
        online_action = (
            "No layer in the captured tail contains positives. Do not spend a full run on threshold tuning before collecting fuller score traces or changing the risk function."
        )

    return {
        "event_threshold_action": event_action,
        "online_primary_action": online_action,
        "event_captured_tp": event_tp,
        "node_captured_tp": node_tp,
        "episode_captured_tp": episode_tp,
        "ranked_has_gt_in_budgets": bool(ranked_has_gt),
    }


def analyze_result_dir(result_dir: Path, budgets: list[int]) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for spec in LAYER_SPECS:
        trace_path = result_dir / spec.trace_filename
        alert_path = result_dir / spec.filename
        if trace_path.exists():
            path = trace_path
            source = "score_trace"
        else:
            path = alert_path
            source = "alert_csv"
        layer = _budget_metrics(_read_csv(path), spec, budgets, source)
        layer["filename"] = str(path)
        layer["missing"] = not path.exists()
        layer["fallback_alert_filename"] = str(alert_path)
        layer["trace_filename"] = str(trace_path)
        layers[spec.name] = layer
    ranked = _ranked_metrics(result_dir, budgets)
    return {
        "dataset": _dataset_name(result_dir),
        "result_dir": str(result_dir),
        "budgets": budgets,
        "limitation": (
            "This audit uses already-emitted online alert CSVs, not full online score traces. "
            "For budgets larger than a CSV row count, metrics are lower bounds over captured alerts only."
        ),
        "layers": layers,
        "ranked_auxiliary": ranked,
        "recommendation": _recommend(layers, ranked),
    }


def _format_num(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, analyses: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Online Tail Reachability Audit")
    lines.append("")
    lines.append(
        "Diagnostic only: labels are used after inference to decide whether another validation-calibrated experiment is worth running."
    )
    lines.append(
        "The online CSVs contain already-emitted alerts, not full score traces; budgets above the CSV row count are lower-bound diagnostics."
    )
    lines.append("")
    for analysis in analyses:
        lines.append(f"## {analysis['dataset']}")
        lines.append("")
        lines.append(f"Result dir: `{analysis['result_dir']}`")
        lines.append("")
        rec = analysis["recommendation"]
        lines.append("### Recommendation")
        lines.append("")
        lines.append(f"- Event threshold: {rec['event_threshold_action']}")
        lines.append(f"- Online primary: {rec['online_primary_action']}")
        lines.append("")
        for layer_name in ("event", "node", "episode"):
            layer = analysis["layers"][layer_name]
            lines.append(f"### {layer_name.title()} Tail")
            lines.append("")
            lines.append(
                f"Rows: `{layer['csv_rows']}`, source: `{layer['source']}`, TP in tail: `{layer['tp_in_captured_alerts']}`, first TP rank: `{layer['first_tp_score_tail_rank']}`"
            )
            lines.append("")
            lines.append("| Budget | Selected | Exact | TP | FP | Precision | Threshold/Floor | Note |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
            for item in layer["budget_metrics"]:
                note = "exact" if item["exact_for_budget"] else "captured lower-bound"
                lines.append(
                    "| {budget} | {selected_alerts} | {exact_for_budget} | {tp} | {fp} | {precision} | {threshold} | {note} |".format(
                        budget=item["budget"],
                        selected_alerts=item["selected_alerts"],
                        exact_for_budget=str(item["exact_for_budget"]).lower(),
                        tp=item["tp"],
                        fp=item["fp"],
                        precision=_format_num(item["precision"]),
                        threshold=_format_num(item["threshold_at_budget_if_exact_else_captured_floor"]),
                        note=note,
                    )
                )
            lines.append("")
        ranked = analysis["ranked_auxiliary"]
        lines.append("### Ranked CSV Auxiliary")
        lines.append("")
        lines.append(ranked["note"])
        lines.append("")
        lines.append("| Budget | Selected Nodes | GT Nodes | Strict FP | Precision | Suspect Nodes |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for item in ranked["budget_metrics"]:
            lines.append(
                "| {budget} | {selected_nodes} | {gt_positive_nodes} | {strict_fp_nodes} | {precision} | {suspect_nodes} |".format(
                    budget=item["budget"],
                    selected_nodes=item["selected_nodes"],
                    gt_positive_nodes=item["gt_positive_nodes"],
                    strict_fp_nodes=item["strict_fp_nodes"],
                    precision=_format_num(item["precision"]),
                    suspect_nodes=item["suspect_nodes"],
                )
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result_dir",
        action="append",
        required=True,
        help="Result directory containing online alert CSVs and causal_semantics_ranked.csv. Repeat for multiple datasets.",
    )
    parser.add_argument("--budgets", default="500,1000,3000", help="Comma-separated alert/node budgets.")
    parser.add_argument("--out_json", default="", help="Optional JSON output path.")
    parser.add_argument("--out_md", default="", help="Optional Markdown output path.")
    parser.add_argument("--print_summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    budgets = sorted({int(x.strip()) for x in str(args.budgets).split(",") if x.strip()})
    if not budgets or any(x <= 0 for x in budgets):
        raise SystemExit("--budgets must contain positive integers")

    analyses = [analyze_result_dir(Path(path), budgets) for path in args.result_dir]
    payload = {
        "purpose": "post-inference online tail reachability diagnostic",
        "budgets": budgets,
        "analyses": analyses,
    }

    out_json = Path(args.out_json) if args.out_json else None
    out_md = Path(args.out_md) if args.out_md else None
    if out_json is None and out_md is None:
        first = Path(args.result_dir[0])
        out_json = first / "online_tail_reachability_audit.json"
        out_md = first / "online_tail_reachability_audit.md"

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(out_md, analyses)

    if args.print_summary:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        if out_json is not None:
            print(f"wrote {out_json}")
        if out_md is not None:
            print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
