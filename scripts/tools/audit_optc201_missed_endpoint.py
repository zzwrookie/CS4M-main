"""Read-only audit for the missed OPTC_201 canonical netflow endpoint."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


DEFAULT_INFER_DIR = Path(
    "outputs/results/tflr_light/optc_windows_v1_3c_full/"
    "OPTC_201_OPTC_WINDOWS_V1_3C_DETAIL_FULL_INFER",
)
DEFAULT_ARTIFACT_DIR = Path(
    "outputs/cache/phase3e_optc_windows_v1_3c/"
    "OPTC_201_OPTC_WINDOWS_V1_3C_DETAIL_full",
)
DEFAULT_CANONICAL_AUDIT_DIR = Path(
    "outputs/diagnostics/optc_windows_v13c_canonical_eval_audit_OPTC_201_20260616_095827",
)
DEFAULT_GT_PATH = Path("ground_truth/h201/node_h201_0923.csv")
EVENT_INDEX_DTYPE = np.dtype(
    [
        ("event_id", "<i8"),
        ("src_node_idx", "<i4"),
        ("dst_node_idx", "<i4"),
        ("action_id", "<i2"),
        ("src_type_id", "i1"),
        ("dst_type_id", "i1"),
    ],
)
ACTION_ID_TO_NAME = {
    0: "EVENT_CLONE",
    1: "EVENT_CLOSE",
    2: "EVENT_EXECUTE",
    3: "EVENT_LOADLIBRARY",
    4: "EVENT_OPEN",
    5: "EVENT_READ",
    6: "EVENT_RECVFROM",
    7: "EVENT_SENDMSG",
    8: "EVENT_WRITE",
    9: "EVENT_FORK",
}
TYPE_ID_TO_NAME = {
    0: "process",
    1: "file",
    2: "netflow",
    3: "registry",
}


def normalize_ip_token(value: object) -> str:
    """Normalize dotted or underscore IPv4 endpoint tokens."""
    text = str(value or "").strip()
    if re.fullmatch(r"\d{1,3}(?:_\d{1,3}){3}", text):
        return text.replace("_", ".")
    return text


def service_bucket_for_port(port: object) -> str:
    """Return the OpTC v1.3 service bucket for a port."""
    text = str(port or "").strip()
    if not text.isdigit():
        return "unknown_port"
    value = int(text)
    named = {
        53: "dns",
        80: "http",
        443: "https",
        3389: "rdp",
        445: "smb",
        135: "rpc",
        139: "netbios",
        5355: "llmnr",
        5353: "mdns",
    }
    if value in named:
        return named[value]
    if 1 <= value <= 1023:
        return "system"
    if 1024 <= value <= 49151:
        return "registered"
    if 49152 <= value <= 65535:
        return "ephemeral"
    return "unknown_port"


def parse_netflow_detail(raw_detail: object) -> tuple[str, str, str, str] | None:
    """Extract src_ip, src_port, dst_ip, dst_port from a raw GT netflow detail cell."""
    text = str(raw_detail or "")
    match = re.search(
        r"([0-9a-fA-F:.]+):(\d+)->([0-9a-fA-F:.]+):(\d+)",
        text,
    )
    if not match:
        return None
    return match.group(1), match.group(2), match.group(3), match.group(4)


def row_matches_endpoint(raw_detail: object, remote_ip: object, service: str) -> bool:
    """Return true when a raw netflow detail contains the target IP and service port."""
    target_ip = normalize_ip_token(remote_ip)
    parsed = parse_netflow_detail(raw_detail)
    if parsed is None:
        return False
    src_ip, src_port, dst_ip, dst_port = parsed
    return (
        (src_ip == target_ip and service_bucket_for_port(src_port) == service)
        or (dst_ip == target_ip and service_bucket_for_port(dst_port) == service)
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _latest_canonical_audit_dir() -> Path:
    candidates = sorted(
        path
        for path in Path("outputs/diagnostics").glob(
            "optc_windows_v13c_canonical_eval_audit_OPTC_201_*",
        )
        if path.is_dir()
    )
    return candidates[-1] if candidates else DEFAULT_CANONICAL_AUDIT_DIR


def _load_event_index_meta(artifact_dir: Path) -> dict[str, Any]:
    path = Path(artifact_dir) / "event_index" / "event_index_meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_ids_for_endpoint(
    canonical_rows: Iterable[Mapping[str, str]],
    *,
    remote_ip: str,
    service: str,
) -> tuple[list[dict[str, Any]], set[int]]:
    matches: list[dict[str, Any]] = []
    compact_ids: set[int] = set()
    for row in canonical_rows:
        if not row_matches_endpoint(row.get("raw_detail", ""), remote_ip, service):
            continue
        payload = dict(row)
        parsed = parse_netflow_detail(row.get("raw_detail", ""))
        if parsed is not None:
            src_ip, src_port, dst_ip, dst_port = parsed
            payload.update(
                {
                    "src_ip": src_ip,
                    "src_port": src_port,
                    "dst_ip": dst_ip,
                    "dst_port": dst_port,
                    "src_service": service_bucket_for_port(src_port),
                    "dst_service": service_bucket_for_port(dst_port),
                },
            )
        try:
            compact_ids.add(int(row["compact_node_idx"]))
        except (KeyError, TypeError, ValueError):
            pass
        matches.append(payload)
    return matches, compact_ids


def _scan_score_trace(
    path: Path,
    compact_ids: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    summary: dict[tuple[str, str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "event_count": 0,
            "above_threshold_count": 0,
            "score_sum": 0.0,
            "threshold_sum": 0.0,
            "max_score": 0.0,
            "min_threshold": None,
            "example_event_id": "",
        },
    )
    sample_rows: list[dict[str, str]] = []
    if not path.exists():
        return [], sample_rows
    compact_text = {str(value) for value in compact_ids}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not (
                row.get("info_src") in compact_text
                or row.get("info_dst") in compact_text
                or row.get("src_idx") in compact_text
                or row.get("dst_idx") in compact_text
            ):
                continue
            if len(sample_rows) < 200:
                sample_rows.append(dict(row))
            score = float(row.get("score", 0.0) or 0.0)
            threshold = float(row.get("threshold", 0.0) or 0.0)
            key = (
                row.get("action", ""),
                row.get("src_type", ""),
                row.get("dst_type", ""),
                row.get("target_case", ""),
                row.get("threshold_group_key", ""),
            )
            item = summary[key]
            item["event_count"] = int(item["event_count"]) + 1
            item["above_threshold_count"] = int(item["above_threshold_count"]) + int(
                score >= threshold,
            )
            item["score_sum"] = float(item["score_sum"]) + score
            item["threshold_sum"] = float(item["threshold_sum"]) + threshold
            item["max_score"] = max(float(item["max_score"]), score)
            item["min_threshold"] = (
                threshold
                if item["min_threshold"] is None
                else min(float(item["min_threshold"]), threshold)
            )
            if not item["example_event_id"]:
                item["example_event_id"] = row.get("event_id", "")
    rows: list[dict[str, Any]] = []
    for key, item in sorted(summary.items()):
        count = max(int(item["event_count"]), 1)
        rows.append(
            {
                "action": key[0],
                "src_type": key[1],
                "dst_type": key[2],
                "target_case": key[3],
                "threshold_group_key": key[4],
                "event_count": int(item["event_count"]),
                "above_threshold_count": int(item["above_threshold_count"]),
                "mean_score": float(item["score_sum"]) / count,
                "mean_threshold": float(item["threshold_sum"]) / count,
                "max_score": float(item["max_score"]),
                "min_threshold": "" if item["min_threshold"] is None else item["min_threshold"],
                "example_event_id": item["example_event_id"],
            },
        )
    return rows, sample_rows


def _scan_alerts(path: Path, compact_ids: set[int]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    compact_text = {str(value) for value in compact_ids}
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (
                row.get("info_src") in compact_text
                or row.get("info_dst") in compact_text
                or row.get("src_idx") in compact_text
                or row.get("dst_idx") in compact_text
            ):
                rows.append(dict(row))
    return rows


def _event_index_split_rows(meta: Mapping[str, Any], compact_ids: set[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, payload in sorted(dict(meta.get("splits", {})).items()):
        path = Path(str(payload.get("path", "")))
        count = int(payload.get("num_events", 0) or 0)
        if not path.exists() or count <= 0:
            rows.append({"split": split, "event_count": 0, "target_event_count": 0})
            continue
        arr = np.memmap(path, dtype=EVENT_INDEX_DTYPE, mode="r", shape=(count,))
        mask = np.isin(arr["src_node_idx"], list(compact_ids)) | np.isin(
            arr["dst_node_idx"],
            list(compact_ids),
        )
        selected = arr[mask]
        action_counts = Counter(int(value) for value in selected["action_id"])
        type_counts = Counter(
            (
                TYPE_ID_TO_NAME.get(int(src), str(int(src))),
                TYPE_ID_TO_NAME.get(int(dst), str(int(dst))),
            )
            for src, dst in zip(selected["src_type_id"], selected["dst_type_id"])
        )
        rows.append(
            {
                "split": split,
                "event_count": int(count),
                "target_event_count": int(mask.sum()),
                "action_counts": json.dumps(
                    {
                        ACTION_ID_TO_NAME.get(key, str(key)): value
                        for key, value in sorted(action_counts.items())
                    },
                    sort_keys=True,
                ),
                "src_dst_type_counts": json.dumps(
                    {f"{src}->{dst}": value for (src, dst), value in type_counts.items()},
                    sort_keys=True,
                ),
                "example_event_id": int(selected["event_id"][0]) if len(selected) else "",
            },
        )
        del arr
    return rows


def _threshold_rows(
    path: Path,
    score_summary_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    wanted = {
        (
            str(row.get("target_case", "")),
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
        )
        for row in score_summary_rows
    }
    rows: list[dict[str, Any]] = []
    for row in _read_csv_rows(path):
        key = (
            row.get("target_case", ""),
            row.get("action_name", ""),
            row.get("src_type_name", ""),
            row.get("dst_type_name", ""),
        )
        if key in wanted:
            rows.append(row)
    return rows


def run_audit(
    *,
    output_dir: Path,
    infer_dir: Path,
    artifact_dir: Path,
    canonical_audit_dir: Path,
    gt_path: Path,
    remote_ip: str,
    service: str,
) -> dict[str, Any]:
    """Run the missed endpoint audit and write CSV/JSON/Markdown artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_mapping_path = canonical_audit_dir / "canonical_gt_mapping.csv"
    canonical_rows = _read_csv_rows(canonical_mapping_path)
    endpoint_rows, compact_ids = _compact_ids_for_endpoint(
        canonical_rows,
        remote_ip=remote_ip,
        service=service,
    )
    score_summary_rows, score_sample_rows = _scan_score_trace(
        infer_dir / "online_event_score_trace.csv",
        compact_ids,
    )
    alert_rows = _scan_alerts(infer_dir / "online_event_alerts.csv", compact_ids)
    event_index_rows = _event_index_split_rows(_load_event_index_meta(artifact_dir), compact_ids)
    threshold_rows = _threshold_rows(
        infer_dir / "conditional_score_summary_by_target_action_type.csv",
        score_summary_rows,
    )

    _write_csv(
        output_dir / "missed_endpoint_gt_mapping.csv",
        endpoint_rows,
        [
            "uuid",
            "kind",
            "original_node_id",
            "lookup_node_id",
            "compact_node_idx",
            "mapping_status",
            "pool_hit",
            "src_ip",
            "src_port",
            "dst_ip",
            "dst_port",
            "src_service",
            "dst_service",
            "raw_detail",
        ],
    )
    _write_csv(
        output_dir / "missed_endpoint_event_index_split_counts.csv",
        event_index_rows,
        [
            "split",
            "event_count",
            "target_event_count",
            "action_counts",
            "src_dst_type_counts",
            "example_event_id",
        ],
    )
    _write_csv(
        output_dir / "missed_endpoint_score_trace_summary.csv",
        score_summary_rows,
        [
            "action",
            "src_type",
            "dst_type",
            "target_case",
            "threshold_group_key",
            "event_count",
            "above_threshold_count",
            "mean_score",
            "mean_threshold",
            "max_score",
            "min_threshold",
            "example_event_id",
        ],
    )
    _write_csv(
        output_dir / "missed_endpoint_score_trace_sample.csv",
        score_sample_rows,
        list(score_sample_rows[0].keys()) if score_sample_rows else ["stream_pos"],
    )
    _write_csv(
        output_dir / "missed_endpoint_alert_rows.csv",
        alert_rows,
        list(alert_rows[0].keys()) if alert_rows else ["stream_pos"],
    )
    _write_csv(
        output_dir / "missed_endpoint_threshold_groups.csv",
        threshold_rows,
        list(threshold_rows[0].keys()) if threshold_rows else ["target_case"],
    )

    split_counts = {
        str(row["split"]): int(row["target_event_count"])
        for row in event_index_rows
    }
    summary = {
        "dataset": "OPTC_201",
        "ground_truth_path": str(gt_path),
        "canonical_mapping_path": str(canonical_mapping_path),
        "infer_dir": str(infer_dir),
        "artifact_dir": str(artifact_dir),
        "target_remote_ip": normalize_ip_token(remote_ip),
        "target_service": service,
        "matched_original_gt_rows": int(len(endpoint_rows)),
        "matched_compact_node_ids": sorted(int(value) for value in compact_ids),
        "event_index_target_counts_by_split": split_counts,
        "score_trace_target_event_count": int(
            sum(int(row.get("event_count", 0) or 0) for row in score_summary_rows),
        ),
        "alert_target_event_count": int(len(alert_rows)),
        "threshold_group_count": int(len(threshold_rows)),
        "conclusion": _conclusion(split_counts, score_summary_rows, alert_rows),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir, summary, event_index_rows, score_summary_rows)
    return summary


def _conclusion(
    split_counts: Mapping[str, int],
    score_summary_rows: Sequence[Mapping[str, Any]],
    alert_rows: Sequence[Mapping[str, Any]],
) -> str:
    if int(split_counts.get("test", 0) or 0) == 0:
        return "target_endpoint_absent_from_test_event_index"
    if not score_summary_rows:
        return "target_endpoint_present_in_test_but_absent_from_score_trace"
    if not alert_rows:
        return "target_endpoint_scored_but_below_alert_threshold"
    return "target_endpoint_alerted"


def _write_report(
    output_dir: Path,
    summary: Mapping[str, Any],
    event_index_rows: Sequence[Mapping[str, Any]],
    score_summary_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# OPTC_201 Missed Endpoint Audit",
        "",
        f"- target: `{summary['target_remote_ip']}` / `{summary['target_service']}`",
        f"- GT file: `{summary['ground_truth_path']}`",
        f"- canonical mapping: `{summary['canonical_mapping_path']}`",
        f"- matched original GT rows: `{summary['matched_original_gt_rows']}`",
        f"- matched compact ids: `{summary['matched_compact_node_ids']}`",
        f"- conclusion: `{summary['conclusion']}`",
        "",
        "## Event Index Split Counts",
        "",
        "| split | target events | action counts |",
        "|---|---:|---|",
    ]
    for row in event_index_rows:
        lines.append(
            f"| {row.get('split', '')} | {row.get('target_event_count', 0)} | "
            f"`{row.get('action_counts', '{}')}` |",
        )
    lines.extend(["", "## Score Trace", ""])
    if score_summary_rows:
        lines.append("| action | src | dst | target_case | count | above threshold | max score |")
        lines.append("|---|---|---|---|---:|---:|---:|")
        for row in score_summary_rows:
            lines.append(
                f"| {row.get('action', '')} | {row.get('src_type', '')} | "
                f"{row.get('dst_type', '')} | {row.get('target_case', '')} | "
                f"{row.get('event_count', 0)} | {row.get('above_threshold_count', 0)} | "
                f"{row.get('max_score', 0)} |",
            )
    else:
        lines.append("No target compact node events appeared in `online_event_score_trace.csv`.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If target events are absent from the test split, lowering alert thresholds cannot "
            "recover them in the current test-only online evaluation. The next fix should audit "
            "split alignment and GT scope before changing OpTC alert policy.",
        ],
    )
    (output_dir / "OPTC_201_missed_endpoint_132_197_158_98_http_audit_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--infer-dir", type=Path, default=DEFAULT_INFER_DIR)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--canonical-audit-dir", type=Path, default=None)
    parser.add_argument("--gt-path", type=Path, default=DEFAULT_GT_PATH)
    parser.add_argument("--remote-ip", default="132_197_158_98")
    parser.add_argument("--service", default="http")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(
            f"outputs/diagnostics/optc201_missed_endpoint_132_197_158_98_http_{stamp}",
        )
    canonical_audit_dir = args.canonical_audit_dir or _latest_canonical_audit_dir()
    summary = run_audit(
        output_dir=output_dir,
        infer_dir=args.infer_dir,
        artifact_dir=args.artifact_dir,
        canonical_audit_dir=canonical_audit_dir,
        gt_path=args.gt_path,
        remote_ip=args.remote_ip,
        service=args.service,
    )
    print(json.dumps({"output_dir": str(output_dir), **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
