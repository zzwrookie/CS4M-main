"""Audit CADETS_E3 v3 CONNECT scores and bounded-smoke speed/cache costs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


CONNECT_GROUP = ("EVENT_CONNECT", "process", "netflow")
NETFLOW_VARIANT_GROUPS = {
    ("EVENT_SENDMSG", "process", "netflow"),
    ("EVENT_SENDTO", "process", "netflow"),
    ("EVENT_WRITE", "process", "netflow"),
    ("EVENT_RECVFROM", "netflow", "process"),
    ("EVENT_RECVMSG", "netflow", "process"),
}


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


def _group_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("action", "")),
        str(row.get("src_type", "")),
        str(row.get("dst_type", "")),
    )


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


def build_group_score_rows(score_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate score-trace rows by action/source/destination/target case."""
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in score_rows:
        action, src_type, dst_type = _group_key(row)
        target_case = str(row.get("target_case", ""))
        key = (action, src_type, dst_type, target_case)
        score = _to_float(row.get("score"))
        threshold = _to_float(row.get("threshold"))
        bucket = grouped.setdefault(
            key,
            {
                "scores": [],
                "thresholds": [],
                "above_threshold_count": 0,
            },
        )
        bucket["scores"].append(score)
        bucket["thresholds"].append(threshold)
        if score >= threshold:
            bucket["above_threshold_count"] += 1

    rows: list[dict[str, object]] = []
    for (action, src_type, dst_type, target_case), payload in sorted(grouped.items()):
        scores = sorted(float(value) for value in payload["scores"])
        thresholds = [float(value) for value in payload["thresholds"]]
        margins = [score - threshold for score, threshold in zip(payload["scores"], thresholds)]
        event_count = len(scores)
        rows.append(
            {
                "action": action,
                "src_type": src_type,
                "dst_type": dst_type,
                "target_case": target_case,
                "event_count": event_count,
                "above_threshold_count": int(payload["above_threshold_count"]),
                "score_mean": float(sum(scores) / event_count) if event_count else 0.0,
                "score_max": float(max(scores)) if scores else 0.0,
                "score_p95": _percentile(scores, 0.95),
                "score_p99": _percentile(scores, 0.99),
                "score_p999": _percentile(scores, 0.999),
                "threshold_mean": float(sum(thresholds) / event_count) if event_count else 0.0,
                "threshold_min": float(min(thresholds)) if thresholds else 0.0,
                "threshold_max": float(max(thresholds)) if thresholds else 0.0,
                "margin_mean": float(sum(margins) / event_count) if event_count else 0.0,
                "margin_max": float(max(margins)) if margins else 0.0,
            },
        )
    return rows


def _find_group(
    rows: Sequence[Mapping[str, object]],
    group: tuple[str, str, str],
) -> dict[str, object]:
    matches = [row for row in rows if _group_key(row) == group]
    if not matches:
        return {}
    return max(matches, key=lambda row: _to_int(row.get("event_count")))


def _connect_conclusion(
    *,
    score_row: Mapping[str, object],
    policy_row: Mapping[str, object],
) -> str:
    event_count = _to_int(score_row.get("event_count") or policy_row.get("event_count"))
    if event_count == 0:
        return "connect_group_absent_from_score_trace"
    if _to_int(policy_row.get("demoted_event_count")) > 0:
        return "policy_demoted_connect_group"
    if _to_int(score_row.get("above_threshold_count")) == 0:
        return "scores_below_validation_threshold_not_policy_demoted"
    if _to_int(policy_row.get("alert_count")) == 0:
        return "above_threshold_scores_exist_but_no_final_alert"
    return "connect_group_alerted"


def build_speed_summary(eval_payload: Mapping[str, Any]) -> dict[str, object]:
    """Extract bounded-smoke speed and cache-cost evidence from eval payload."""
    timing = dict(eval_payload.get("timing", {}))
    runtime = dict(eval_payload.get("runtime", {}))
    memory = dict(eval_payload.get("memory", {}))
    validation_seconds = _to_float(timing.get("validation_seconds"))
    test_seconds = _to_float(timing.get("test_scoring_seconds"))
    label_seconds = _to_float(timing.get("label_attach_seconds"))
    elapsed_seconds = _to_float(runtime.get("elapsed_seconds"))
    if elapsed_seconds <= 0:
        elapsed_seconds = validation_seconds + test_seconds + label_seconds
    events_scored = _to_int(runtime.get("events_scored") or eval_payload.get("test_events"))
    online_eps = _to_float(memory.get("online_minimal_events_per_sec"))
    if online_eps <= 0 and test_seconds > 0 and events_scored > 0:
        online_eps = float(events_scored) / test_seconds
    dominant_cost = "online_scoring"
    if validation_seconds > test_seconds and validation_seconds > label_seconds:
        dominant_cost = "validation_cache_build"
    elif label_seconds > test_seconds:
        dominant_cost = "label_attach"
    return {
        "events_scored": events_scored,
        "elapsed_seconds": elapsed_seconds,
        "validation_seconds": validation_seconds,
        "test_scoring_seconds": test_seconds,
        "label_attach_seconds": label_seconds,
        "end_to_end_events_per_sec": _to_float(runtime.get("throughput_events_per_second")),
        "online_scoring_events_per_sec": online_eps,
        "validation_share": validation_seconds / elapsed_seconds if elapsed_seconds else 0.0,
        "test_scoring_share": test_seconds / elapsed_seconds if elapsed_seconds else 0.0,
        "dominant_cost": dominant_cost,
        "db_stream_mode": str(
            eval_payload.get("db_stream_mode_actual") or eval_payload.get("db_stream_mode") or "",
        ),
        "node_embedding_lookup_mode": str(
            dict(eval_payload.get("config", {})).get("node_embedding_lookup_mode", ""),
        ),
        "rss_profile_mode": str(dict(eval_payload.get("config", {})).get("rss_profile_mode", "")),
    }


def _write_report(
    path: Path,
    *,
    result_dir: Path,
    connect_summary: Mapping[str, object],
    speed_summary: Mapping[str, object],
    variant_rows: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# CADETS_E3 v3 CONNECT Speed Cache Audit",
        "",
        f"Result directory: `{result_dir}`",
        "",
        "## CONNECT process->netflow",
        "",
        f"- event_count: {connect_summary.get('event_count', 0)}",
        f"- above_threshold_count: {connect_summary.get('above_threshold_count', 0)}",
        f"- score_max: {connect_summary.get('score_max', 0.0)}",
        f"- threshold_mean: {connect_summary.get('threshold_mean', 0.0)}",
        f"- margin_max: {connect_summary.get('margin_max', 0.0)}",
        f"- policy_alert_count: {connect_summary.get('policy_alert_count', 0)}",
        f"- policy_demoted_event_count: {connect_summary.get('policy_demoted_event_count', 0)}",
        f"- conclusion: {connect_summary.get('conclusion', '')}",
        "",
        "## Netflow Variants",
        "",
        "| action | src | dst | events | above threshold | max score | alerts |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in variant_rows:
        lines.append(
            "| "
            f"{row.get('action', '')} | {row.get('src_type', '')} | {row.get('dst_type', '')} | "
            f"{row.get('event_count', 0)} | {row.get('above_threshold_count', 0)} | "
            f"{row.get('score_max', 0.0)} | {row.get('policy_alert_count', 0)} |",
        )
    lines.extend(
        [
            "",
            "## Speed",
            "",
            f"- events_scored: {speed_summary.get('events_scored', 0)}",
            f"- validation_seconds: {speed_summary.get('validation_seconds', 0.0)}",
            f"- test_scoring_seconds: {speed_summary.get('test_scoring_seconds', 0.0)}",
            f"- label_attach_seconds: {speed_summary.get('label_attach_seconds', 0.0)}",
            f"- end_to_end_events_per_sec: {speed_summary.get('end_to_end_events_per_sec', 0.0)}",
            f"- online_scoring_events_per_sec: {speed_summary.get('online_scoring_events_per_sec', 0.0)}",
            f"- dominant_cost: {speed_summary.get('dominant_cost', '')}",
            "",
            "## Interpretation",
            "",
            "This audit is post-stream and read-only. It does not design thresholds from test labels.",
        ],
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(result_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the read-only CONNECT/speed audit and write sidecar outputs."""
    result_dir = Path(result_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_rows = _read_csv(result_dir / "online_event_score_trace.csv")
    group_rows = build_group_score_rows(score_rows)
    policy_rows = _read_csv(result_dir / "group_alert_policy_summary.csv")
    eval_payload = _read_json(result_dir / "eval_causal_semantics_slim.json")

    policy_by_group = {_group_key(row): row for row in policy_rows}
    enriched_rows: list[dict[str, object]] = []
    for row in group_rows:
        policy = policy_by_group.get(_group_key(row), {})
        enriched_rows.append(
            {
                **row,
                "policy_alert_count": _to_int(policy.get("alert_count")),
                "policy_node_evidence_count": _to_int(policy.get("node_evidence_count")),
                "policy_demoted_event_count": _to_int(policy.get("demoted_event_count")),
            },
        )

    connect_score = _find_group(enriched_rows, CONNECT_GROUP)
    connect_policy = policy_by_group.get(CONNECT_GROUP, {})
    connect_summary = {
        **connect_score,
        "policy_alert_count": _to_int(connect_policy.get("alert_count")),
        "policy_node_evidence_count": _to_int(connect_policy.get("node_evidence_count")),
        "policy_demoted_event_count": _to_int(connect_policy.get("demoted_event_count")),
    }
    connect_summary["event_count"] = _to_int(
        connect_summary.get("event_count") or connect_policy.get("event_count"),
    )
    connect_summary["conclusion"] = _connect_conclusion(
        score_row=connect_summary,
        policy_row=connect_policy,
    )

    variant_rows = [
        row for row in enriched_rows if _group_key(row) in NETFLOW_VARIANT_GROUPS | {CONNECT_GROUP}
    ]
    speed_summary = build_speed_summary(eval_payload)

    score_fields = [
        "action",
        "src_type",
        "dst_type",
        "target_case",
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
        "policy_alert_count",
        "policy_node_evidence_count",
        "policy_demoted_event_count",
    ]
    _write_csv(output_dir / "connect_group_score_audit.csv", enriched_rows, score_fields)
    _write_csv(
        output_dir / "speed_cache_audit_summary.csv",
        [speed_summary],
        list(speed_summary.keys()),
    )
    summary = {
        "result_dir": str(result_dir),
        "output_dir": str(output_dir),
        "connect_process_netflow": connect_summary,
        "netflow_variant_groups": variant_rows,
        "speed": speed_summary,
    }
    (output_dir / "cadets_e3_v3_connect_speed_cache_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / "cadets_e3_v3_connect_speed_cache_audit_report.md",
        result_dir=result_dir,
        connect_summary=connect_summary,
        speed_summary=speed_summary,
        variant_rows=variant_rows,
    )
    return summary


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/diagnostics") / f"cadets_e3_v3_connect_speed_cache_{timestamp}"


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

    output_dir = args.output_dir or _default_output_dir()
    summary = run_audit(result_dir=args.result_dir, output_dir=output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
