"""Read-only CADETS_E3 v3 netflow action score attribution audit."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


CONNECT_GROUP = ("EVENT_CONNECT", "process", "netflow")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _score_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("target_case", "")),
        str(row.get("action", "")),
        str(row.get("src_type", "")),
        str(row.get("dst_type", "")),
    )


def _policy_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("action", "")),
        str(row.get("src_type", "")),
        str(row.get("dst_type", "")),
    )


def _validation_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("target_case", "")),
        str(row.get("action_name", row.get("action", ""))),
        str(row.get("src_type_name", row.get("src_type", ""))),
        str(row.get("dst_type_name", row.get("dst_type", ""))),
    )


def _is_netflow_group(row: Mapping[str, object]) -> bool:
    return str(row.get("src_type", "")) == "netflow" or str(row.get("dst_type", "")) == "netflow"


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = float(q) * float(len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - float(lower)
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def _interpret(row: Mapping[str, object]) -> str:
    event_count = _to_int(row.get("event_count"))
    if event_count == 0:
        return "absent_from_score_trace"
    if _to_int(row.get("demoted_event_count")) > 0:
        return "policy_demoted"
    above = _to_int(row.get("above_threshold_count"))
    validation_count = _to_int(row.get("validation_count"))
    if above == 0 and validation_count > 0:
        return "below_threshold_with_validation_support"
    if above == 0:
        return "below_threshold_without_validation_support"
    if _to_int(row.get("final_alert_count")) > 0:
        return "above_threshold_final_alerts"
    return "above_threshold_no_final_alert"


def _score_group_rows(score_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in score_rows:
        if not _is_netflow_group(row):
            continue
        key = _score_key(row)
        score = _to_float(row.get("score"))
        threshold = _to_float(row.get("threshold"))
        bucket = grouped.setdefault(key, {"scores": [], "thresholds": [], "above": 0})
        bucket["scores"].append(score)
        bucket["thresholds"].append(threshold)
        if score >= threshold:
            bucket["above"] += 1

    rows: list[dict[str, object]] = []
    for (target_case, action, src_type, dst_type), payload in sorted(grouped.items()):
        scores = sorted(float(value) for value in payload["scores"])
        thresholds = [float(value) for value in payload["thresholds"]]
        margins = [score - threshold for score, threshold in zip(payload["scores"], thresholds)]
        count = len(scores)
        rows.append(
            {
                "target_case": target_case,
                "action": action,
                "src_type": src_type,
                "dst_type": dst_type,
                "event_count": count,
                "above_threshold_count": int(payload["above"]),
                "score_mean": float(sum(scores) / count) if count else 0.0,
                "score_max": float(max(scores)) if scores else 0.0,
                "score_p95": _percentile(scores, 0.95),
                "score_p99": _percentile(scores, 0.99),
                "score_p999": _percentile(scores, 0.999),
                "threshold_mean": float(sum(thresholds) / count) if count else 0.0,
                "threshold_min": float(min(thresholds)) if thresholds else 0.0,
                "threshold_max": float(max(thresholds)) if thresholds else 0.0,
                "margin_mean": float(sum(margins) / count) if count else 0.0,
                "margin_max": float(max(margins)) if margins else 0.0,
            },
        )
    return rows


def _best_sweep_by_group(
    sweep_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], dict[str, object]]:
    best: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in sweep_rows:
        if str(row.get("src_type", "")) != "netflow" and str(row.get("dst_type", "")) != "netflow":
            continue
        key = _policy_key(row)
        current = best.get(key)
        if current is None:
            best[key] = dict(row)
            continue
        score = (
            _to_float(row.get("precision")),
            _to_int(row.get("covered_malicious_nodes")),
            _to_int(row.get("TP")),
            -_to_int(row.get("FP")),
        )
        current_score = (
            _to_float(current.get("precision")),
            _to_int(current.get("covered_malicious_nodes")),
            _to_int(current.get("TP")),
            -_to_int(current.get("FP")),
        )
        if score > current_score:
            best[key] = dict(row)
    return best


def build_attribution_rows(
    *,
    score_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    policy_rows: Sequence[Mapping[str, object]],
    sweep_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build netflow score attribution rows from test, validation, and policy sidecars."""
    validation_by_key = {_validation_key(row): row for row in validation_rows}
    policy_by_key = {_policy_key(row): row for row in policy_rows}
    sweep_by_key = _best_sweep_by_group(sweep_rows)
    rows: list[dict[str, object]] = []
    for score in _score_group_rows(score_rows):
        key = (
            str(score["target_case"]),
            str(score["action"]),
            str(score["src_type"]),
            str(score["dst_type"]),
        )
        group_key = (str(score["action"]), str(score["src_type"]), str(score["dst_type"]))
        validation = validation_by_key.get(key, {})
        policy = policy_by_key.get(group_key, {})
        sweep = sweep_by_key.get(group_key, {})
        row = {
            **score,
            "validation_count": _to_int(validation.get("validation_count")),
            "validation_count_bucket": str(validation.get("validation_count_bucket", "")),
            "threshold_level": str(validation.get("threshold_level", "")),
            "threshold_source": str(validation.get("final_threshold_source", "")),
            "validation_p99": _to_float(validation.get("val_p99")),
            "validation_p999": _to_float(validation.get("val_p999")),
            "validation_p9995": _to_float(validation.get("val_p9995")),
            "validation_p9999": _to_float(validation.get("val_p9999")),
            "validation_max": _to_float(validation.get("val_max")),
            "validation_test_max_sidecar": _to_float(validation.get("test_max")),
            "final_alert_count": _to_int(policy.get("alert_count")),
            "demoted_event_count": _to_int(policy.get("demoted_event_count")),
            "node_evidence_count": _to_int(policy.get("node_evidence_count")),
            "eval_only_tp": _to_int(policy.get("TP")),
            "eval_only_fp": _to_int(policy.get("FP")),
            "eval_only_covered_malicious_nodes": _to_int(
                policy.get("covered_malicious_nodes"),
            ),
            "eval_only_best_sweep_threshold": _to_float(sweep.get("threshold")),
            "eval_only_best_sweep_margin": _to_float(sweep.get("margin")),
            "eval_only_best_sweep_alert_count": _to_int(sweep.get("alert_count")),
            "eval_only_best_sweep_tp": _to_int(sweep.get("TP")),
            "eval_only_best_sweep_fp": _to_int(sweep.get("FP")),
            "eval_only_best_sweep_covered_nodes": _to_int(
                sweep.get("covered_malicious_nodes"),
            ),
        }
        row["interpretation"] = _interpret(row)
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            str(row["action"]) != CONNECT_GROUP[0]
            or str(row["src_type"]) != CONNECT_GROUP[1]
            or str(row["dst_type"]) != CONNECT_GROUP[2],
            str(row["action"]),
            str(row["src_type"]),
            str(row["dst_type"]),
            str(row["target_case"]),
        ),
    )


def _connect_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    for row in rows:
        if (
            str(row.get("action")) == CONNECT_GROUP[0]
            and str(row.get("src_type")) == CONNECT_GROUP[1]
            and str(row.get("dst_type")) == CONNECT_GROUP[2]
        ):
            return dict(row)
    return {
        "action": CONNECT_GROUP[0],
        "src_type": CONNECT_GROUP[1],
        "dst_type": CONNECT_GROUP[2],
        "event_count": 0,
        "interpretation": "absent_from_score_trace",
    }


def _shift_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "action": row.get("action", ""),
            "src_type": row.get("src_type", ""),
            "dst_type": row.get("dst_type", ""),
            "target_case": row.get("target_case", ""),
            "event_count": row.get("event_count", 0),
            "above_threshold_count": row.get("above_threshold_count", 0),
            "score_max": row.get("score_max", 0.0),
            "final_alert_count": row.get("final_alert_count", 0),
            "interpretation": row.get("interpretation", ""),
        }
        for row in rows
        if _to_int(row.get("above_threshold_count")) > 0
        or _to_int(row.get("final_alert_count")) > 0
    ]


def _write_report(
    path: Path,
    *,
    result_dir: Path,
    connect: Mapping[str, object],
    shifts: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# CADETS_E3 v3 Netflow Action Score Attribution Audit",
        "",
        f"Result directory: `{result_dir}`",
        "",
        "This is a read-only post-stream audit. Sweep TP/FP columns are evaluation-only",
        "and are not used to select runtime thresholds.",
        "",
        "## CONNECT process->netflow",
        "",
        f"- interpretation: {connect.get('interpretation', '')}",
        f"- event_count: {connect.get('event_count', 0)}",
        f"- above_threshold_count: {connect.get('above_threshold_count', 0)}",
        f"- score_max: {connect.get('score_max', 0.0)}",
        f"- threshold_mean: {connect.get('threshold_mean', 0.0)}",
        f"- margin_max: {connect.get('margin_max', 0.0)}",
        f"- validation_count: {connect.get('validation_count', 0)}",
        f"- validation_bucket: {connect.get('validation_count_bucket', '')}",
        f"- threshold_source: {connect.get('threshold_source', '')}",
        f"- final_alert_count: {connect.get('final_alert_count', 0)}",
        f"- demoted_event_count: {connect.get('demoted_event_count', 0)}",
        "",
        "## Netflow Alert Shift",
        "",
        "| action | src | dst | target case | events | above threshold | max score | alerts |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in shifts:
        lines.append(
            "| "
            f"{row.get('action', '')} | {row.get('src_type', '')} | "
            f"{row.get('dst_type', '')} | {row.get('target_case', '')} | "
            f"{row.get('event_count', 0)} | {row.get('above_threshold_count', 0)} | "
            f"{row.get('score_max', 0.0)} | {row.get('final_alert_count', 0)} |",
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "If CONNECT remains below threshold with validation support, the next phase should",
            "audit semantic/action mapping or Phase3G score-target behavior before changing",
            "runtime policy.",
        ],
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fields() -> list[str]:
    return [
        "target_case",
        "action",
        "src_type",
        "dst_type",
        "event_count",
        "above_threshold_count",
        "score_mean",
        "score_max",
        "score_p95",
        "score_p99",
        "score_p999",
        "threshold_mean",
        "threshold_min",
        "threshold_max",
        "margin_mean",
        "margin_max",
        "validation_count",
        "validation_count_bucket",
        "threshold_level",
        "threshold_source",
        "validation_p99",
        "validation_p999",
        "validation_p9995",
        "validation_p9999",
        "validation_max",
        "validation_test_max_sidecar",
        "final_alert_count",
        "demoted_event_count",
        "node_evidence_count",
        "eval_only_tp",
        "eval_only_fp",
        "eval_only_covered_malicious_nodes",
        "eval_only_best_sweep_threshold",
        "eval_only_best_sweep_margin",
        "eval_only_best_sweep_alert_count",
        "eval_only_best_sweep_tp",
        "eval_only_best_sweep_fp",
        "eval_only_best_sweep_covered_nodes",
        "interpretation",
    ]


def run_audit(result_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the read-only netflow action score attribution audit."""
    result_dir = Path(result_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_attribution_rows(
        score_rows=_read_csv(result_dir / "online_event_score_trace.csv"),
        validation_rows=_read_csv(result_dir / "conditional_score_summary_by_target_action_type.csv"),
        policy_rows=_read_csv(result_dir / "group_alert_policy_summary.csv"),
        sweep_rows=_read_csv(result_dir / "group_threshold_sweep_summary.csv"),
    )
    connect = _connect_summary(rows)
    shifts = _shift_summary(rows)
    fields = _fields()
    _write_csv(output_dir / "netflow_action_score_attribution.csv", rows, fields)
    _write_csv(
        output_dir / "netflow_action_threshold_source.csv",
        rows,
        [
            "target_case",
            "action",
            "src_type",
            "dst_type",
            "validation_count",
            "validation_count_bucket",
            "threshold_level",
            "threshold_source",
            "validation_p9995",
            "threshold_mean",
            "score_max",
            "margin_max",
            "interpretation",
        ],
    )
    _write_csv(
        output_dir / "netflow_action_sweep_sensitivity.csv",
        rows,
        [
            "action",
            "src_type",
            "dst_type",
            "eval_only_best_sweep_threshold",
            "eval_only_best_sweep_margin",
            "eval_only_best_sweep_alert_count",
            "eval_only_best_sweep_tp",
            "eval_only_best_sweep_fp",
            "eval_only_best_sweep_covered_nodes",
        ],
    )
    summary = {
        "result_dir": str(result_dir),
        "output_dir": str(output_dir),
        "connect_process_netflow": connect,
        "netflow_alert_shift_groups": shifts,
        "leakage_check": _read_json(result_dir / "eval_causal_semantics_slim.json").get(
            "leakage_check",
            {},
        ),
        "evaluation_only_notice": "Sweep TP/FP fields are post-stream audit evidence only.",
    }
    (output_dir / "cadets_e3_v3_netflow_action_score_attribution.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / "cadets_e3_v3_netflow_action_score_attribution_report.md",
        result_dir=result_dir,
        connect=connect,
        shifts=shifts,
    )
    return summary


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/diagnostics") / (
        f"cadets_e3_v3_netflow_action_score_attribution_{timestamp}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result_dir",
        type=Path,
        default=Path(
            "outputs/results/tflr_light/"
            "CADETS_E3_FREEBSD_V3_SAFE_LEXICAL_E4_POLICY_SMOKE_500K",
        ),
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = run_audit(result_dir=args.result_dir, output_dir=args.output_dir or _default_output_dir())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
