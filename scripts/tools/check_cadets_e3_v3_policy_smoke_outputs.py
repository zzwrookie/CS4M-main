"""Validate CADETS_E3 v3 policy smoke outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


LEAKAGE_HEADER_TOKENS = ("label", "malicious", "ground_truth", "attack")
EXPECTED_POLICY = "cadets_e4_v3_policy_smoke_v1"
REQUIRED_FILES = (
    "online_event_alerts.csv",
    "group_alert_policy_summary.csv",
    "demoted_group_summary.csv",
    "target_case_summary.csv",
    "eval_causal_semantics_slim.json",
)
DEMOTED_PROCESS_FILE_GROUPS = {
    ("EVENT_CONNECT", "process", "file"),
    ("EVENT_SENDTO", "process", "file"),
    ("EVENT_SENDMSG", "process", "file"),
    ("EVENT_RECVMSG", "file", "process"),
}


class SmokeCheckError(RuntimeError):
    """Raised when CADETS v3 policy smoke outputs violate the contract."""


def _to_int(value: object, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(str(value))
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _group_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("action", "")),
        str(row.get("src_type", "")),
        str(row.get("dst_type", "")),
    )


def _validate_required_files(result_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (result_dir / name).is_file()]
    if missing:
        raise SmokeCheckError(f"missing required smoke outputs: {', '.join(missing)}")


def _validate_no_leakage_headers(headers: Sequence[str]) -> None:
    bad = [
        header
        for header in headers
        if any(token in str(header).lower() for token in LEAKAGE_HEADER_TOKENS)
    ]
    if bad:
        raise SmokeCheckError(f"online_event_alerts.csv has leakage-like headers: {bad}")


def _validate_policy_groups(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    policy_names = {str(row.get("policy_name", "")) for row in rows if row.get("policy_name")}
    if EXPECTED_POLICY not in policy_names:
        raise SmokeCheckError(f"expected policy {EXPECTED_POLICY}, got {sorted(policy_names)}")

    by_group = {_group_key(row): row for row in rows}
    connect = by_group.get(("EVENT_CONNECT", "process", "netflow"), {})
    connect_alerts = _to_int(connect.get("alert_count"))
    connect_demoted = _to_int(connect.get("demoted_event_count"))
    connect_evidence = _to_int(connect.get("node_evidence_count"))
    if connect_demoted > 0 and connect_alerts == 0:
        raise SmokeCheckError("EVENT_CONNECT process->netflow is still fully demoted")

    process_file_alerts = 0
    process_file_demoted = 0
    for key in DEMOTED_PROCESS_FILE_GROUPS:
        row = by_group.get(key)
        if not row:
            continue
        alerts = _to_int(row.get("alert_count"))
        demoted = _to_int(row.get("demoted_event_count"))
        evidence = _to_int(row.get("node_evidence_count"))
        process_file_alerts += alerts
        process_file_demoted += demoted
        if alerts > 0:
            raise SmokeCheckError(f"{key} should be demoted, but has {alerts} alerts")
        if evidence > 0 and demoted <= 0:
            raise SmokeCheckError(f"{key} has node evidence without demotion count")

    return {
        "connect_process_netflow_alert_count": connect_alerts,
        "connect_process_netflow_demoted_count": connect_demoted,
        "connect_process_netflow_node_evidence_count": connect_evidence,
        "demoted_process_file_alert_count": process_file_alerts,
        "demoted_process_file_demoted_count": process_file_demoted,
    }


def check_outputs(result_dir: Path) -> dict[str, Any]:
    """Validate smoke outputs and persist a compact summary JSON."""
    result_dir = Path(result_dir)
    _validate_required_files(result_dir)

    alert_headers, alert_rows = _read_csv(result_dir / "online_event_alerts.csv")
    _validate_no_leakage_headers(alert_headers)
    _, group_rows = _read_csv(result_dir / "group_alert_policy_summary.csv")
    policy_summary = _validate_policy_groups(group_rows)
    eval_payload = _read_json(result_dir / "eval_causal_semantics_slim.json")
    config = eval_payload.get("config", {})
    runtime = eval_payload.get("runtime", {})
    timing = eval_payload.get("timing", {})
    ofsm = eval_payload.get("ofsm_compression", {})

    if eval_payload.get("ground_truth_used_only_for_evaluation") is not True:
        raise SmokeCheckError("ground_truth_used_only_for_evaluation is not true")
    if config.get("action_type_alert_policy") != EXPECTED_POLICY:
        raise SmokeCheckError("eval config did not record expected policy")
    throughput = _to_float(runtime.get("throughput_events_per_second"))
    if throughput <= 0.0:
        raise SmokeCheckError("throughput_events_per_second must be positive")

    summary: dict[str, Any] = {
        "result_dir": str(result_dir),
        "policy_name": EXPECTED_POLICY,
        "alert_rows": len(alert_rows),
        "events_scored": _to_int(runtime.get("events_scored") or eval_payload.get("test_events")),
        "elapsed_seconds": _to_float(runtime.get("elapsed_seconds")),
        "throughput_events_per_second": throughput,
        "validation_seconds": _to_float(timing.get("validation_seconds")),
        "test_scoring_seconds": _to_float(timing.get("test_scoring_seconds")),
        "label_attach_seconds": _to_float(timing.get("label_attach_seconds")),
        "db_stream_mode": eval_payload.get("db_stream_mode_actual")
        or eval_payload.get("db_stream_mode")
        or "",
        "node_embedding_lookup_mode": config.get("node_embedding_lookup_mode", ""),
        "sspm_infer_fast_path": bool(config.get("sspm_infer_fast_path", False)),
        "rss_profile_mode": config.get("rss_profile_mode", ""),
        "logical_node_count": _to_int(ofsm.get("logical_node_count")),
    }
    summary.update(policy_summary)
    out_path = result_dir / "cadets_e3_v3_policy_smoke_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    summary = check_outputs(args.result_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
