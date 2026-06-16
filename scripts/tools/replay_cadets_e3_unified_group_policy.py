#!/usr/bin/env python3
"""Replay CADETS E3 unified group thresholds and policy offline."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


POLICY_NAME = "cadets_e3_unified_group_train_fallback_v1"
DEFAULT_RESULT_DIR = (
    "outputs/results/tflr_light/CADETS_E3_FREEBSD_V3_SAFE_LEXICAL_E4_FULL"
)

HIGH_RETAIN_GROUPS = {
    ("EVENT_READ", "file", "process"),
    ("EVENT_OPEN", "process", "process"),
    ("EVENT_SENDTO", "process", "netflow"),
    ("EVENT_EXECUTE", "process", "file"),
    ("EVENT_WRITE", "process", "file"),
}
FP_GUARD_GROUPS = {
    ("EVENT_SENDMSG", "process", "file"),
    ("EVENT_RECVMSG", "file", "process"),
    ("EVENT_SENDTO", "process", "file"),
    ("EVENT_RECVFROM", "file", "process"),
    ("EVENT_CONNECT", "process", "file"),
}


@dataclass(frozen=True)
class ReplayParams:
    """Fixed replay parameters for CADETS E3 phase-1 validation."""

    validation_group_quantile: float = 0.9995
    validation_global_quantile: float = 0.999
    min_validation_count: int = 100
    min_train_count: int = 20
    train_margin_abs: float = 0.005
    train_margin_rel: float = 0.05
    fp_guard_margin: float = 0.02
    cold_unseen_margin: float = 0.05
    node_evidence_min: int = 2


@dataclass(frozen=True)
class ThresholdInputs:
    """Calibration inputs visible before test replay."""

    validation_groups: dict[str, dict[str, Any]]
    train_groups: dict[str, dict[str, Any]]
    validation_global_threshold: float
    train_fallback_available: bool


@dataclass(frozen=True)
class ThresholdDecision:
    """Resolved threshold and provenance for one group."""

    threshold: float
    threshold_source: str
    threshold_level: str
    validation_count: int
    train_count: int


@dataclass(frozen=True)
class PolicyDecision:
    """CADETS policy decision for one raw alert."""

    final_alert: bool
    high_priority: bool
    policy_reason: str


def make_group_key(target_case: str, action: str, src_type: str, dst_type: str) -> str:
    """Build the unified threshold group key."""
    return f"{target_case}|{action}|{src_type}->{dst_type}"


def resolve_threshold(
    group_key: str,
    inputs: ThresholdInputs,
    params: ReplayParams,
) -> ThresholdDecision:
    """Resolve threshold from validation group, train group, then validation global."""
    validation = inputs.validation_groups.get(group_key, {})
    validation_count = _to_int(validation.get("validation_count"))
    if validation_count >= params.min_validation_count:
        return ThresholdDecision(
            threshold=_to_float(validation.get("threshold")),
            threshold_source="validation_group_quantile",
            threshold_level="group",
            validation_count=validation_count,
            train_count=_to_int(inputs.train_groups.get(group_key, {}).get("train_count")),
        )

    train = inputs.train_groups.get(group_key, {})
    train_count = _to_int(train.get("train_count"))
    if train_count >= params.min_train_count:
        train_max = _to_float(train.get("train_max_score"))
        margin = max(params.train_margin_abs, params.train_margin_rel * train_max)
        return ThresholdDecision(
            threshold=train_max + margin,
            threshold_source="train_group_max",
            threshold_level="group_train_fallback",
            validation_count=validation_count,
            train_count=train_count,
        )

    return ThresholdDecision(
        threshold=float(inputs.validation_global_threshold),
        threshold_source="validation_global_quantile",
        threshold_level="global",
        validation_count=validation_count,
        train_count=train_count,
    )


def decide_cadets_policy(
    *,
    action: str,
    src_type: str,
    dst_type: str,
    score_margin: float,
    threshold_source: str,
    validation_count: int,
    train_count: int,
    node_evidence_count: int,
    params: ReplayParams | None = None,
) -> PolicyDecision:
    """Apply fixed CADETS E3 policy to one raw alert without labels."""
    params = params or ReplayParams()
    group = (str(action), str(src_type), str(dst_type))
    is_cold_unseen = (
        int(validation_count) == 0
        and int(train_count) == 0
        and str(threshold_source) == "validation_global_quantile"
    )

    if group in HIGH_RETAIN_GROUPS:
        return PolicyDecision(True, float(score_margin) >= 0.02, "high_retain_group")

    if (
        group == ("EVENT_CONNECT", "process", "netflow")
        and str(threshold_source) in {"train_group_max", "validation_group_quantile"}
    ):
        return PolicyDecision(True, False, "netflow_connect_train_or_validation")

    if is_cold_unseen and not (
        float(score_margin) >= params.cold_unseen_margin
        and int(node_evidence_count) >= params.node_evidence_min
    ):
        return PolicyDecision(False, False, "cold_unseen_demoted")

    if group in FP_GUARD_GROUPS:
        final = (
            float(score_margin) >= params.fp_guard_margin
            or int(node_evidence_count) >= params.node_evidence_min
        )
        return PolicyDecision(final, False, "fp_guard_pass" if final else "fp_guard_demoted")

    if group == ("EVENT_RECVFROM", "netflow", "process"):
        final = (
            float(score_margin) >= params.fp_guard_margin
            or int(node_evidence_count) >= params.node_evidence_min
        )
        return PolicyDecision(
            final,
            False,
            "netflow_recvfrom_guard" if final else "netflow_recvfrom_demoted",
        )

    if src_type == "netflow" or dst_type == "netflow":
        final = float(score_margin) >= 0.03 or int(node_evidence_count) >= params.node_evidence_min
        return PolicyDecision(
            final,
            False,
            "netflow_guarded_pass" if final else "netflow_guarded_demoted",
        )

    return PolicyDecision(True, False, "default_retain")


def replay_cadets_result(
    result_dir: Path,
    output_dir: Path,
    params: ReplayParams | None = None,
) -> dict[str, Any]:
    """Replay CADETS E3 score trace and write raw/final/demoted diagnostics."""
    params = params or ReplayParams()
    result_dir = Path(result_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = result_dir / "online_event_score_trace.csv"
    if not score_path.exists():
        raise FileNotFoundError(f"missing required score trace: {score_path}")

    start = time.monotonic()
    inputs = load_threshold_inputs(result_dir, params)
    baseline_final_rows = _count_csv_rows(result_dir / "online_event_alerts.csv")
    raw_path = output_dir / "raw_alerts.csv"
    final_path = output_dir / "final_alerts.csv"
    demoted_path = output_dir / "demoted_events.csv"
    raw_count = 0
    final_count = 0
    demoted_count = 0
    source_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    node_evidence: Counter[str] = Counter()
    connect_summary: Counter[str] = Counter()

    fields = _output_fields()
    with raw_path.open("w", encoding="utf-8", newline="") as raw_handle, final_path.open(
        "w", encoding="utf-8", newline=""
    ) as final_handle, demoted_path.open("w", encoding="utf-8", newline="") as demoted_handle:
        raw_writer = csv.DictWriter(raw_handle, fieldnames=fields)
        final_writer = csv.DictWriter(final_handle, fieldnames=fields)
        demoted_writer = csv.DictWriter(demoted_handle, fieldnames=fields)
        raw_writer.writeheader()
        final_writer.writeheader()
        demoted_writer.writeheader()

        with score_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                target_case = str(row.get("target_case", ""))
                action = str(row.get("action", ""))
                src_type = str(row.get("src_type", ""))
                dst_type = str(row.get("dst_type", ""))
                group_key = make_group_key(target_case, action, src_type, dst_type)
                threshold = resolve_threshold(group_key, inputs, params)
                score = _to_float(row.get("score"))
                score_margin = score - float(threshold.threshold)
                if score_margin < 0:
                    continue

                src_idx = str(row.get("src_idx", ""))
                dst_idx = str(row.get("dst_idx", ""))
                evidence_count = max(node_evidence[src_idx], node_evidence[dst_idx])
                policy = decide_cadets_policy(
                    action=action,
                    src_type=src_type,
                    dst_type=dst_type,
                    score_margin=score_margin,
                    threshold_source=threshold.threshold_source,
                    validation_count=threshold.validation_count,
                    train_count=threshold.train_count,
                    node_evidence_count=evidence_count,
                    params=params,
                )
                out = _output_row(
                    row=row,
                    score=score,
                    score_margin=score_margin,
                    group_key=group_key,
                    threshold=threshold,
                    node_evidence_count=evidence_count,
                    policy=policy,
                )
                raw_writer.writerow(out)
                raw_count += 1
                source_counts[threshold.threshold_source] += 1
                policy_counts[policy.policy_reason] += 1
                group_counts[group_key]["raw"] += 1

                if action == "EVENT_CONNECT" and src_type == "process" and dst_type == "netflow":
                    connect_summary["raw"] += 1
                    connect_summary[f"source:{threshold.threshold_source}"] += 1
                    connect_summary[f"policy:{policy.policy_reason}"] += 1

                if policy.final_alert:
                    final_writer.writerow(out)
                    final_count += 1
                    group_counts[group_key]["final"] += 1
                    if action == "EVENT_CONNECT" and src_type == "process" and dst_type == "netflow":
                        connect_summary["final"] += 1
                else:
                    demoted_writer.writerow(out)
                    demoted_count += 1
                    group_counts[group_key]["demoted"] += 1
                    if action == "EVENT_CONNECT" and src_type == "process" and dst_type == "netflow":
                        connect_summary["demoted"] += 1

                node_evidence[src_idx] += 1
                node_evidence[dst_idx] += 1

    _write_counter_csv(output_dir / "threshold_source_summary.csv", "threshold_source", source_counts)
    _write_counter_csv(output_dir / "policy_summary.csv", "policy_reason", policy_counts)
    _write_group_summary(output_dir / "group_summary.csv", group_counts)
    connect_payload = dict(connect_summary)
    connect_payload["group"] = "event_semantic_target|EVENT_CONNECT|process->netflow"
    (output_dir / "connect_process_netflow_summary.json").write_text(
        json.dumps(connect_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "policy_name": POLICY_NAME,
        "result_dir": str(result_dir),
        "baseline_final_alert_rows": baseline_final_rows,
        "raw_alert_rows": raw_count,
        "final_alert_rows": final_count,
        "demoted_event_rows": demoted_count,
        "train_fallback_available": inputs.train_fallback_available,
        "elapsed_seconds": time.monotonic() - start,
        "output_dir": str(output_dir),
    }
    (output_dir / "cadets_e3_unified_group_policy_replay_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def load_threshold_inputs(result_dir: Path, params: ReplayParams) -> ThresholdInputs:
    """Load validation and optional train summaries for replay."""
    validation_groups = _load_validation_groups(
        Path(result_dir) / "conditional_score_summary_by_target_action_type.csv"
    )
    train_path = Path(result_dir) / "train_group_score_summary.csv"
    train_groups = _load_train_groups(train_path)
    global_threshold = _validation_global_threshold(result_dir, validation_groups, params)
    return ThresholdInputs(
        validation_groups=validation_groups,
        train_groups=train_groups,
        validation_global_threshold=global_threshold,
        train_fallback_available=train_path.exists(),
    )


def _load_validation_groups(path: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return groups
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            target_case = str(row.get("target_case", ""))
            action = str(row.get("action_name") or row.get("action") or "")
            src_type = str(row.get("src_type_name") or row.get("src_type") or "")
            dst_type = str(row.get("dst_type_name") or row.get("dst_type") or "")
            key = make_group_key(target_case, action, src_type, dst_type)
            groups[key] = {
                "threshold": _to_float(row.get("threshold")),
                "validation_count": _to_int(row.get("validation_count")),
            }
    return groups


def _load_train_groups(path: Path) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return groups
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = make_group_key(
                str(row.get("target_case", "")),
                str(row.get("action", "")),
                str(row.get("src_type", "")),
                str(row.get("dst_type", "")),
            )
            groups[key] = {
                "train_count": _to_int(row.get("train_count")),
                "train_max_score": _to_float(row.get("train_max_score")),
            }
    return groups


def _validation_global_threshold(
    result_dir: Path,
    validation_groups: dict[str, dict[str, Any]],
    params: ReplayParams,
) -> float:
    score_summary_threshold = _score_summary_validation_threshold(
        Path(result_dir) / "score_summary.json"
    )
    if score_summary_threshold > 0:
        return score_summary_threshold
    thresholds = [
        _to_float(record.get("threshold"))
        for record in validation_groups.values()
        if _to_int(record.get("validation_count")) >= params.min_validation_count
    ]
    return max(thresholds) if thresholds else 0.0


def _score_summary_validation_threshold(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.0
    candidates = [
        data.get("validation", {}).get("final_threshold"),
        data.get("validation", {}).get("threshold"),
        data.get("final_threshold"),
    ]
    for value in candidates:
        threshold = _to_float(value)
        if threshold > 0:
            return threshold
    return 0.0


def _output_row(
    *,
    row: dict[str, Any],
    score: float,
    score_margin: float,
    group_key: str,
    threshold: ThresholdDecision,
    node_evidence_count: int,
    policy: PolicyDecision,
) -> dict[str, Any]:
    return {
        "stream_pos": row.get("stream_pos", ""),
        "event_id": row.get("event_id", ""),
        "src_idx": row.get("src_idx", ""),
        "dst_idx": row.get("dst_idx", ""),
        "score": score,
        "threshold": threshold.threshold,
        "score_margin": score_margin,
        "threshold_source": threshold.threshold_source,
        "threshold_level": threshold.threshold_level,
        "validation_count": threshold.validation_count,
        "train_count": threshold.train_count,
        "target_case": row.get("target_case", ""),
        "action": row.get("action", ""),
        "src_type": row.get("src_type", ""),
        "dst_type": row.get("dst_type", ""),
        "group_key": group_key,
        "node_evidence_count": node_evidence_count,
        "policy_name": POLICY_NAME,
        "policy_reason": policy.policy_reason,
        "high_priority_event_alert": int(policy.high_priority),
    }


def _output_fields() -> list[str]:
    return [
        "stream_pos",
        "event_id",
        "src_idx",
        "dst_idx",
        "score",
        "threshold",
        "score_margin",
        "threshold_source",
        "threshold_level",
        "validation_count",
        "train_count",
        "target_case",
        "action",
        "src_type",
        "dst_type",
        "group_key",
        "node_evidence_count",
        "policy_name",
        "policy_reason",
        "high_priority_event_alert",
    ]


def _write_counter_csv(path: Path, key_name: str, counter: Counter[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key_name, "count"])
        writer.writeheader()
        for key, count in counter.most_common():
            writer.writerow({key_name: key, "count": count})


def _write_group_summary(path: Path, groups: dict[str, Counter[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group_key", "raw", "final", "demoted"])
        writer.writeheader()
        for key in sorted(groups):
            counter = groups[key]
            writer.writerow(
                {
                    "group_key": key,
                    "raw": counter.get("raw", 0),
                    "final": counter.get("final", 0),
                    "demoted": counter.get("demoted", 0),
                }
            )


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


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
    """Run CADETS E3 unified group policy replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = replay_cadets_result(Path(args.result_dir), Path(args.output_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
