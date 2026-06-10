"""Summarize ClearScope E3 v3 Phase3G FP-reduction runs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_JSON_FILES = ("eval_causal_semantics_slim.json", "score_summary.json")


def _load_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required JSON file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"required JSON file must contain an object: {path}")
    return payload


def _as_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _as_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _first_value(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        if name in row:
            return row.get(name, "")
    return ""


def _csv_group_fp(path: Path, action: str, src_type: str, dst_type: str) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("action") == action
                and row.get("src_type") == src_type
                and row.get("dst_type") == dst_type
            ):
                return _as_int(_first_value(row, ("fp", "FP")))
    return 0


def _target_case_alerts(path: Path, target_case: str) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("target_case") == target_case:
                alerts = _as_int(_first_value(row, ("alert_count", "alerts")))
                fp = _as_int(_first_value(row, ("FP", "fp")))
                return alerts, fp
    return 0, 0


def _primary_online_metrics(eval_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    primary = eval_payload.get("primary_online_metrics", {})
    if not isinstance(primary, dict):
        return {}, {}
    event_alerts = primary.get("event_alerts", {})
    node_coverage = primary.get("online_event_node_coverage", {})
    return (
        event_alerts if isinstance(event_alerts, dict) else {},
        node_coverage if isinstance(node_coverage, dict) else {},
    )


def _state_model(eval_payload: dict[str, Any], score_summary: dict[str, Any]) -> str:
    value = score_summary.get("state_model")
    if value:
        return str(value)
    config = eval_payload.get("config", {})
    if isinstance(config, dict):
        return str(config.get("sspm_state_model", ""))
    return ""


def _both_cold_unseen_policy(score_summary: dict[str, Any]) -> dict[str, Any]:
    payload = score_summary.get("both_cold_unseen_policy", {})
    return payload if isinstance(payload, dict) else {}


def _endpoint_suppression(score_summary: dict[str, Any]) -> dict[str, Any]:
    payload = score_summary.get("conditional_endpoint_suppression", {})
    return payload if isinstance(payload, dict) else {}


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """Return a compact FP-policy summary for one ClearScope v3 run directory."""
    eval_payload = _load_required_json(run_dir / REQUIRED_JSON_FILES[0])
    score_summary = _load_required_json(run_dir / REQUIRED_JSON_FILES[1])
    event_alerts, node_coverage = _primary_online_metrics(eval_payload)
    both_cold_alerts, both_cold_fp = _target_case_alerts(
        run_dir / "target_case_summary.csv",
        "both_cold_action_target",
    )
    return {
        "run": run_dir.name,
        "event_alert_count": _as_int(event_alerts.get("count")),
        "tp": _as_int(event_alerts.get("tp")),
        "fp": _as_int(event_alerts.get("fp")),
        "precision": _as_float(event_alerts.get("ratio")),
        "malicious_node_recall": _as_float(node_coverage.get("malicious_node_recall")),
        "event_read_netflow_process_fp": _csv_group_fp(
            run_dir / "event_fp_group_summary.csv",
            "EVENT_READ",
            "netflow",
            "process",
        ),
        "both_cold_alert_count": both_cold_alerts,
        "both_cold_fp": both_cold_fp,
        "threshold_mode": str(score_summary.get("event_threshold_mode", "")),
        "state_model": _state_model(eval_payload, score_summary),
        "endpoint_suppression": _endpoint_suppression(score_summary),
        "both_cold_unseen_policy": _both_cold_unseen_policy(score_summary),
        "leakage_check": eval_payload.get("leakage_check", {}),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: clearscope_v3_fp_policy_report.py RUN_DIR [RUN_DIR ...]",
            file=sys.stderr,
        )
        return 2
    try:
        summaries = [summarize_run(Path(raw)) for raw in argv[1:]]
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
