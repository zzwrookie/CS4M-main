"""Read-only root-cause audit for unseen CADETS_E3 v3 CONNECT netflow group."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


CONNECT_GROUP = ("event_semantic_target", "EVENT_CONNECT", "process", "netflow")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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


def _validation_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("target_case", "")),
        str(row.get("action_name", row.get("action", ""))),
        str(row.get("src_type_name", row.get("src_type", ""))),
        str(row.get("dst_type_name", row.get("dst_type", ""))),
    )


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


def build_group_matrix(
    *,
    validation_rows: Sequence[Mapping[str, object]],
    score_rows: Sequence[Mapping[str, object]],
    policy_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build validation/test/policy matrix by target-case action and type pair."""
    rows_by_key: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in validation_rows:
        key = _validation_key(row)
        rows_by_key[key] = {
            "target_case": key[0],
            "action": key[1],
            "src_type": key[2],
            "dst_type": key[3],
            "validation_count": _to_int(row.get("validation_count")),
            "validation_test_count_sidecar": _to_int(row.get("test_count")),
            "threshold_source": str(row.get("final_threshold_source", "")),
            "threshold_level": str(row.get("threshold_level", "")),
            "validation_bucket": str(row.get("validation_count_bucket", "")),
            "score_trace_count": 0,
            "score_trace_max": 0.0,
            "score_trace_above_threshold_count": 0,
            "final_alert_count": 0,
            "demoted_event_count": 0,
        }
    for row in score_rows:
        key = _score_key(row)
        item = rows_by_key.setdefault(
            key,
            {
                "target_case": key[0],
                "action": key[1],
                "src_type": key[2],
                "dst_type": key[3],
                "validation_count": 0,
                "validation_test_count_sidecar": 0,
                "threshold_source": "",
                "threshold_level": "",
                "validation_bucket": "",
                "score_trace_count": 0,
                "score_trace_max": 0.0,
                "score_trace_above_threshold_count": 0,
                "final_alert_count": 0,
                "demoted_event_count": 0,
            },
        )
        score = _to_float(row.get("score"))
        threshold = _to_float(row.get("threshold"))
        item["score_trace_count"] = _to_int(item.get("score_trace_count")) + 1
        item["score_trace_max"] = max(_to_float(item.get("score_trace_max")), score)
        if score >= threshold:
            item["score_trace_above_threshold_count"] = (
                _to_int(item.get("score_trace_above_threshold_count")) + 1
            )
    policy_by_key = {_policy_key(row): row for row in policy_rows}
    for item in rows_by_key.values():
        policy = policy_by_key.get(
            (str(item["action"]), str(item["src_type"]), str(item["dst_type"])),
            {},
        )
        item["final_alert_count"] = _to_int(policy.get("alert_count"))
        item["demoted_event_count"] = _to_int(policy.get("demoted_event_count"))
    return sorted(
        rows_by_key.values(),
        key=lambda row: (
            str(row["action"]) != CONNECT_GROUP[1]
            or str(row["src_type"]) != CONNECT_GROUP[2]
            or str(row["dst_type"]) != CONNECT_GROUP[3],
            str(row["target_case"]),
            str(row["action"]),
            str(row["src_type"]),
            str(row["dst_type"]),
        ),
    )


def _connect_row(matrix_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    for row in matrix_rows:
        if (
            str(row.get("target_case")) == CONNECT_GROUP[0]
            and str(row.get("action")) == CONNECT_GROUP[1]
            and str(row.get("src_type")) == CONNECT_GROUP[2]
            and str(row.get("dst_type")) == CONNECT_GROUP[3]
        ):
            return dict(row)
    return {
        "target_case": CONNECT_GROUP[0],
        "action": CONNECT_GROUP[1],
        "src_type": CONNECT_GROUP[2],
        "dst_type": CONNECT_GROUP[3],
        "validation_count": 0,
        "score_trace_count": 0,
        "final_alert_count": 0,
        "demoted_event_count": 0,
    }


def classify_connect_root_cause(matrix_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Classify the most likely sidecar-visible root cause for CONNECT being unseen."""
    row = _connect_row(matrix_rows)
    validation_count = _to_int(row.get("validation_count"))
    score_trace_count = _to_int(row.get("score_trace_count"))
    demoted_count = _to_int(row.get("demoted_event_count"))
    if demoted_count > 0:
        root_cause = "policy_demoted"
    elif score_trace_count == 0:
        root_cause = "test_group_absent"
    elif validation_count == 0 and score_trace_count > 0:
        root_cause = "validation_group_absent_test_group_present"
    else:
        root_cause = "unknown_sidecar_gap"
    return {
        **row,
        "root_cause": root_cause,
        "validation_count": validation_count,
        "score_trace_count": score_trace_count,
        "demoted_event_count": demoted_count,
    }


def build_neighbor_rows(matrix_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Return same-action and same-type-pair neighbors for CONNECT process->netflow."""
    rows: list[dict[str, object]] = []
    for row in matrix_rows:
        action = str(row.get("action", ""))
        src_type = str(row.get("src_type", ""))
        dst_type = str(row.get("dst_type", ""))
        if (action, src_type, dst_type) == CONNECT_GROUP[1:]:
            continue
        relation = ""
        if action == CONNECT_GROUP[1]:
            relation = "same_action"
        elif src_type == CONNECT_GROUP[2] and dst_type == CONNECT_GROUP[3]:
            relation = "same_type_pair"
        elif src_type == "netflow" or dst_type == "netflow":
            relation = "other_netflow"
        if relation:
            rows.append({**row, "neighbor_relation": relation})
    return sorted(
        rows,
        key=lambda row: (
            str(row["neighbor_relation"]) != "same_type_pair",
            str(row["neighbor_relation"]),
            str(row["action"]),
            str(row["src_type"]),
            str(row["dst_type"]),
            str(row["target_case"]),
        ),
    )


def _matrix_fields() -> list[str]:
    return [
        "target_case",
        "action",
        "src_type",
        "dst_type",
        "validation_count",
        "validation_test_count_sidecar",
        "threshold_source",
        "threshold_level",
        "validation_bucket",
        "score_trace_count",
        "score_trace_max",
        "score_trace_above_threshold_count",
        "final_alert_count",
        "demoted_event_count",
    ]


def _write_report(
    path: Path,
    *,
    result_dir: Path,
    root_cause: Mapping[str, object],
    neighbors: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# CADETS_E3 v3 CONNECT Unseen Root Cause Audit",
        "",
        f"Result directory: `{result_dir}`",
        "",
        "This is a read-only sidecar audit. It does not train, run inference,",
        "change thresholds, or design rules from test labels.",
        "",
        "## CONNECT Root Cause",
        "",
        f"- root_cause: {root_cause.get('root_cause', '')}",
        f"- validation_count: {root_cause.get('validation_count', 0)}",
        f"- score_trace_count: {root_cause.get('score_trace_count', 0)}",
        f"- score_trace_max: {root_cause.get('score_trace_max', 0.0)}",
        f"- score_trace_above_threshold_count: {root_cause.get('score_trace_above_threshold_count', 0)}",
        f"- final_alert_count: {root_cause.get('final_alert_count', 0)}",
        f"- demoted_event_count: {root_cause.get('demoted_event_count', 0)}",
        f"- threshold_source: {root_cause.get('threshold_source', '')}",
        "",
        "## Neighbor Evidence",
        "",
        "| relation | action | src | dst | target case | validation | test rows | alerts |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in neighbors[:30]:
        lines.append(
            "| "
            f"{row.get('neighbor_relation', '')} | {row.get('action', '')} | "
            f"{row.get('src_type', '')} | {row.get('dst_type', '')} | "
            f"{row.get('target_case', '')} | {row.get('validation_count', 0)} | "
            f"{row.get('score_trace_count', 0)} | {row.get('final_alert_count', 0)} |",
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(result_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the CONNECT unseen root-cause audit and write sidecars."""
    result_dir = Path(result_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_group_matrix(
        validation_rows=_read_csv(result_dir / "conditional_score_summary_by_target_action_type.csv"),
        score_rows=_read_csv(result_dir / "online_event_score_trace.csv"),
        policy_rows=_read_csv(result_dir / "group_alert_policy_summary.csv"),
    )
    root_cause = classify_connect_root_cause(matrix)
    neighbors = build_neighbor_rows(matrix)
    _write_csv(output_dir / "connect_unseen_group_matrix.csv", matrix, _matrix_fields())
    _write_csv(
        output_dir / "connect_unseen_action_direction_neighbors.csv",
        neighbors,
        [*_matrix_fields(), "neighbor_relation"],
    )
    summary = {
        "result_dir": str(result_dir),
        "output_dir": str(output_dir),
        "connect_root_cause": root_cause,
        "neighbor_count": len(neighbors),
        "same_action_neighbor_count": sum(
            1 for row in neighbors if row.get("neighbor_relation") == "same_action"
        ),
        "same_type_pair_neighbor_count": sum(
            1 for row in neighbors if row.get("neighbor_relation") == "same_type_pair"
        ),
        "top_neighbors": neighbors[:20],
    }
    (output_dir / "cadets_e3_v3_connect_unseen_root_cause.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        output_dir / "cadets_e3_v3_connect_unseen_root_cause_report.md",
        result_dir=result_dir,
        root_cause=root_cause,
        neighbors=neighbors,
    )
    return summary


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/diagnostics") / f"cadets_e3_v3_connect_unseen_root_cause_{timestamp}"


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
