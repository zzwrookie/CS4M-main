"""Read-only CADETS_E3 E4 v2-v3 regression, policy, and speed audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FOCUS_GROUPS = {
    ("EVENT_OPEN", "process", "process"),
    ("EVENT_READ", "file", "process"),
    ("EVENT_CONNECT", "process", "netflow"),
    ("EVENT_RECVFROM", "netflow", "process"),
}


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


def _group_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(row.get("action", "")),
        str(row.get("src_type", "")),
        str(row.get("dst_type", "")),
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read CSV rows from path, returning an empty list when the file is absent."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """Write rows to CSV with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from path, returning an empty dict when absent."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_tp_nodes(path: Path) -> tuple[set[int], dict[int, dict[str, str]]]:
    """Load post-stream TP nodes from online_event_node_coverage_strict.csv."""
    tp_nodes: set[int] = set()
    by_node: dict[int, dict[str, str]] = {}
    for row in read_csv_rows(path):
        node_idx = _to_int(row.get("node_idx"), default=-1)
        if node_idx < 0:
            continue
        by_node[node_idx] = row
        if str(row.get("eval_result", "")).upper() == "TP":
            tp_nodes.add(node_idx)
    return tp_nodes, by_node


def build_tp_node_delta_rows(
    *,
    v2_tp_nodes: set[int],
    v3_tp_nodes: set[int],
    total_gt_nodes: int,
    v2_nodes: Mapping[int, Mapping[str, object]],
    v3_nodes: Mapping[int, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return TP node delta rows between v2 and v3 post-stream coverage."""
    rows: list[dict[str, object]] = []
    union_nodes = sorted(v2_tp_nodes | v3_tp_nodes)
    for node_idx in union_nodes:
        in_v2 = node_idx in v2_tp_nodes
        in_v3 = node_idx in v3_tp_nodes
        if in_v2 and in_v3:
            category = "both_covered"
        elif in_v2:
            category = "v2_only"
        else:
            category = "v3_only"
        v2_row = v2_nodes.get(node_idx, {})
        v3_row = v3_nodes.get(node_idx, {})
        rows.append(
            {
                "node_idx": node_idx,
                "coverage_category": category,
                "node_type": v3_row.get("node_type") or v2_row.get("node_type", ""),
                "v2_alert_count": _to_int(v2_row.get("alert_count")),
                "v3_alert_count": _to_int(v3_row.get("alert_count")),
                "v2_max_event_score": _to_float(v2_row.get("max_event_score")),
                "v3_max_event_score": _to_float(v3_row.get("max_event_score")),
            }
        )

    missed_count = max(0, int(total_gt_nodes) - len(union_nodes))
    for idx in range(1, missed_count + 1):
        rows.append(
            {
                "node_idx": f"unknown_gt_{idx}",
                "coverage_category": "both_missed",
                "node_type": "",
                "v2_alert_count": 0,
                "v3_alert_count": 0,
                "v2_max_event_score": 0.0,
                "v3_max_event_score": 0.0,
            }
        )
    return rows


def _summarize_group(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "alert_count": _to_int(row.get("alert_count")),
        "event_tp": _to_int(row.get("TP")),
        "event_fp": _to_int(row.get("FP")),
        "precision": _to_float(row.get("precision")),
        "covered_malicious_nodes": _to_int(row.get("covered_malicious_nodes")),
        "strict_node_tp": _to_int(row.get("strict_node_TP")),
        "strict_node_fp": _to_int(row.get("strict_node_FP")),
        "threshold": _to_float(row.get("threshold")),
        "event_count": _to_int(row.get("event_count")),
    }


def build_group_regression_rows(
    v2_group_rows: Sequence[Mapping[str, object]],
    v3_group_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Compare v2 and v3 group-level post-stream metrics."""
    v2_by_key = {_group_key(row): _summarize_group(row) for row in v2_group_rows}
    v3_by_key = {_group_key(row): _summarize_group(row) for row in v3_group_rows}
    keys = sorted(set(v2_by_key) | set(v3_by_key))
    rows: list[dict[str, object]] = []
    for action, src_type, dst_type in keys:
        v2 = v2_by_key.get((action, src_type, dst_type), {})
        v3 = v3_by_key.get((action, src_type, dst_type), {})
        process_file_fp = src_type == "process" and dst_type == "file" and (
            "SEND" in action or "RECV" in action or action == "EVENT_CONNECT"
        )
        rows.append(
            {
                "action": action,
                "src_type": src_type,
                "dst_type": dst_type,
                "focus_group": (action, src_type, dst_type) in FOCUS_GROUPS or process_file_fp,
                "v2_alert_count": int(v2.get("alert_count", 0)),
                "v3_alert_count": int(v3.get("alert_count", 0)),
                "alert_delta_v3_minus_v2": int(v3.get("alert_count", 0))
                - int(v2.get("alert_count", 0)),
                "v2_tp": int(v2.get("event_tp", 0)),
                "v3_tp": int(v3.get("event_tp", 0)),
                "tp_delta_v3_minus_v2": int(v3.get("event_tp", 0)) - int(v2.get("event_tp", 0)),
                "v2_fp": int(v2.get("event_fp", 0)),
                "v3_fp": int(v3.get("event_fp", 0)),
                "fp_delta_v3_minus_v2": int(v3.get("event_fp", 0)) - int(v2.get("event_fp", 0)),
                "v2_precision": float(v2.get("precision", 0.0)),
                "v3_precision": float(v3.get("precision", 0.0)),
                "v2_covered_nodes": int(v2.get("covered_malicious_nodes", 0)),
                "v3_covered_nodes": int(v3.get("covered_malicious_nodes", 0)),
                "covered_node_delta_v3_minus_v2": int(v3.get("covered_malicious_nodes", 0))
                - int(v2.get("covered_malicious_nodes", 0)),
                "v2_threshold": float(v2.get("threshold", 0.0)),
                "v3_threshold": float(v3.get("threshold", 0.0)),
            }
        )
    return sorted(rows, key=lambda row: (not row["focus_group"], row["tp_delta_v3_minus_v2"]))


def _best_sweep_row(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    candidates = list(rows)
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda row: (
            _to_float(row.get("precision")),
            _to_int(row.get("covered_malicious_nodes")),
            _to_int(row.get("TP")),
            -_to_int(row.get("FP")),
        ),
    )


def join_demotion_with_sweep(
    demoted_rows: Sequence[Mapping[str, object]],
    sweep_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Join v3 demoted groups with their best post-stream sweep candidate."""
    sweep_by_key: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in sweep_rows:
        sweep_by_key[_group_key(row)].append(row)

    rows: list[dict[str, object]] = []
    for demoted in demoted_rows:
        key = _group_key(demoted)
        best = _best_sweep_row(sweep_by_key.get(key, []))
        precision = _to_float(best.get("precision"))
        tp = _to_int(best.get("TP"))
        fp = _to_int(best.get("FP"))
        covered = _to_int(best.get("covered_malicious_nodes"))
        if tp > 0 and precision >= 0.9 and fp <= 2:
            actionability = "strong_posthoc_candidate"
        elif tp > 0 and precision >= 0.3:
            actionability = "mixed_posthoc_candidate"
        elif tp > 0:
            actionability = "weak_posthoc_candidate"
        else:
            actionability = "no_tp_evidence"
        rows.append(
            {
                "action": key[0],
                "src_type": key[1],
                "dst_type": key[2],
                "target_case": demoted.get("target_case", ""),
                "demoted_event_count": _to_int(demoted.get("demoted_event_count")),
                "node_evidence_count": _to_int(demoted.get("node_evidence_count")),
                "best_candidate_threshold": _to_float(best.get("threshold")),
                "best_candidate_alert_count": _to_int(best.get("alert_count")),
                "best_candidate_tp": tp,
                "best_candidate_fp": fp,
                "best_candidate_precision": precision,
                "best_candidate_covered_nodes": covered,
                "best_candidate_strict_node_tp": _to_int(best.get("strict_node_TP")),
                "best_candidate_strict_node_fp": _to_int(best.get("strict_node_FP")),
                "best_candidate_quantile": best.get("quantile", ""),
                "best_candidate_margin": best.get("margin", ""),
                "candidate_rule_basis": "validation/runtime-visible group threshold plus fixed margin",
                "posthoc_actionability": actionability,
            }
        )
    return sorted(rows, key=lambda row: (-row["best_candidate_tp"], row["action"]))


def _current_from_eval(eval_payload: Mapping[str, Any]) -> dict[str, int]:
    primary = eval_payload.get("primary_online_metrics", {})
    alerts = primary.get("event_alerts", {}) if isinstance(primary, dict) else {}
    coverage = primary.get("online_event_node_coverage", {}) if isinstance(primary, dict) else {}
    return {
        "event_alerts": _to_int(alerts.get("count")),
        "event_tp": _to_int(alerts.get("tp")),
        "event_fp": _to_int(alerts.get("fp")),
        "node_tp": _to_int(coverage.get("covered_malicious_nodes")),
    }


def _select_sweep(
    sweep_rows: Sequence[Mapping[str, object]],
    action: str,
    src_type: str,
    dst_type: str,
    margin: str,
) -> dict[str, object]:
    matches = [
        row
        for row in sweep_rows
        if _group_key(row) == (action, src_type, dst_type)
        and str(row.get("margin", "")) == margin
    ]
    return _best_sweep_row(matches)


def build_policy_candidate_projection_rows(
    *,
    current: Mapping[str, int],
    baseline: Mapping[str, int],
    sweep_rows: Sequence[Mapping[str, object]],
    total_gt_nodes: int,
) -> list[dict[str, object]]:
    """Build post-stream projection rows for label-free candidate families."""
    rows: list[dict[str, object]] = []

    def add_candidate(
        name: str,
        base: Mapping[str, int],
        additions: Sequence[Mapping[str, object]] = (),
        rule_source: str = "existing_runtime_policy",
    ) -> None:
        add_alerts = sum(_to_int(row.get("alert_count")) for row in additions)
        add_tp = sum(_to_int(row.get("TP")) for row in additions)
        add_fp = sum(_to_int(row.get("FP")) for row in additions)
        add_nodes = sum(_to_int(row.get("covered_malicious_nodes")) for row in additions)
        projected_alerts = int(base.get("event_alerts", 0)) + add_alerts
        projected_tp = int(base.get("event_tp", 0)) + add_tp
        projected_fp = int(base.get("event_fp", 0)) + add_fp
        projected_nodes = min(int(total_gt_nodes), int(base.get("node_tp", 0)) + add_nodes)
        rows.append(
            {
                "candidate_name": name,
                "rule_source": rule_source,
                "uses_test_labels_for_rule": False,
                "poststream_eval_uses_labels": True,
                "projected_event_alerts": projected_alerts,
                "projected_event_tp": projected_tp,
                "projected_event_fp": projected_fp,
                "projected_event_precision": projected_tp / projected_alerts if projected_alerts else 0.0,
                "projected_unique_tp_nodes_upper_bound": projected_nodes,
                "projected_node_recall_upper_bound": projected_nodes / total_gt_nodes
                if total_gt_nodes
                else 0.0,
                "added_alerts": add_alerts,
                "added_tp": add_tp,
                "added_fp": add_fp,
                "added_covered_nodes_upper_bound": add_nodes,
            }
        )

    connect_002 = _select_sweep(sweep_rows, "EVENT_CONNECT", "process", "netflow", "0.02")
    connect_005 = _select_sweep(sweep_rows, "EVENT_CONNECT", "process", "netflow", "0.05")
    recv_005 = _select_sweep(sweep_rows, "EVENT_RECVFROM", "netflow", "process", "0.05")
    read_002 = _select_sweep(sweep_rows, "EVENT_READ", "file", "process", "0.02")

    add_candidate("v3_current_policy", current)
    add_candidate(
        "v3_restore_demoted_netflow_connect_margin_0p02",
        current,
        [connect_002] if connect_002 else [],
        "validation_runtime_visible_projection",
    )
    add_candidate(
        "v3_restore_demoted_netflow_connect_margin_0p05",
        current,
        [connect_005] if connect_005 else [],
        "validation_runtime_visible_projection",
    )
    add_candidate(
        "v3_high_precision_netflow_recvfrom_margin_0p05",
        current,
        [recv_005] if recv_005 else [],
        "validation_runtime_visible_projection",
    )
    add_candidate(
        "v3_high_precision_read_file_process_margin_0p02",
        current,
        [read_002] if read_002 else [],
        "validation_runtime_visible_projection",
    )
    combined = [row for row in (connect_002, recv_005, read_002) if row]
    add_candidate(
        "v3_combined_high_precision_recoveries",
        current,
        combined,
        "validation_runtime_visible_projection",
    )
    add_candidate("v2_best_baseline", baseline, [], "existing_v2_best_runtime_policy")
    return rows


def build_speed_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, object]]:
    """Extract speed and likely bottleneck rows from eval payloads."""
    rows: list[dict[str, object]] = []
    for run_name, payload in payloads.items():
        runtime = payload.get("runtime", {})
        config = payload.get("config", {})
        if not isinstance(config, dict):
            config = {}
        ofsm = payload.get("ofsm_compression", {})
        timing = payload.get("timing", {})
        memory = payload.get("memory", {})
        if not isinstance(memory, dict):
            memory = {}
        state_memory = memory.get("state_memory", {})
        if not isinstance(state_memory, dict):
            state_memory = {}
        db_mode = (
            payload.get("db_stream_mode_actual")
            or payload.get("db_stream_mode")
            or config.get("db_stream_mode_actual")
            or config.get("db_stream_mode")
            or ""
        )
        lookup_mode = config.get("node_embedding_lookup_mode", "")
        notes: list[str] = []
        if db_mode:
            notes.append(f"db_stream={db_mode}")
        if "python_lookup" in str(db_mode):
            notes.append("python_lookup may dominate DB/row materialization cost")
        if lookup_mode:
            notes.append(f"node_lookup={lookup_mode}")
        logical_nodes = _to_int(
            ofsm.get("logical_nodes")
            or ofsm.get("logical_node_count")
            or state_memory.get("logical_node_count")
        )
        physical_states = _to_int(
            ofsm.get("physical_states")
            or ofsm.get("num_physical_states")
            or state_memory.get("num_physical_states")
        )
        if logical_nodes > 150000:
            notes.append("large logical-node/state count")
        if _to_float(timing.get("stream_csv_write_seconds")) > 0:
            notes.append("CSV streaming/write overhead present")
        embedding_seconds = _to_float(memory.get("embedding_lookup_seconds"))
        if embedding_seconds > 0:
            notes.append(f"embedding_lookup_seconds={embedding_seconds:.2f}")
        rows.append(
            {
                "run_name": run_name,
                "dataset": payload.get("dataset", ""),
                "out_tag": payload.get("out_tag", ""),
                "events_scored": _to_int(runtime.get("events_scored") or payload.get("test_events")),
                "elapsed_seconds": _to_float(runtime.get("elapsed_seconds")),
                "throughput_events_per_second": _to_float(
                    runtime.get("throughput_events_per_second")
                ),
                "db_stream_mode": db_mode,
                "node_embedding_lookup_mode": lookup_mode,
                "node_embedding_lazy_backing": config.get("node_embedding_lazy_backing", ""),
                "sspm_infer_backend": config.get("sspm_infer_backend", ""),
                "logical_nodes": logical_nodes,
                "physical_states": physical_states,
                "deploy_infer_rss_peak_mb": _to_float(
                    ofsm.get("deploy_infer_rss_peak_mb")
                    or memory.get("deploy_infer_rss_peak_mb")
                ),
                "validation_seconds": _to_float(timing.get("validation_seconds")),
                "test_scoring_seconds": _to_float(timing.get("test_scoring_seconds")),
                "stream_csv_write_seconds": _to_float(timing.get("stream_csv_write_seconds")),
                "likely_bottleneck_notes": "; ".join(notes),
            }
        )
    return sorted(rows, key=lambda row: str(row["run_name"]))


def _summary_total_nodes(path: Path) -> int:
    payload = read_json(path)
    return _to_int(payload.get("total_malicious_nodes"))


def _render_report(
    *,
    out_dir: Path,
    v2_eval: Mapping[str, Any],
    v3_eval: Mapping[str, Any],
    node_rows: Sequence[Mapping[str, object]],
    group_rows: Sequence[Mapping[str, object]],
    demotion_rows: Sequence[Mapping[str, object]],
    projection_rows: Sequence[Mapping[str, object]],
    speed_rows: Sequence[Mapping[str, object]],
) -> str:
    v2_current = _current_from_eval(v2_eval)
    v3_current = _current_from_eval(v3_eval)
    categories = defaultdict(int)
    for row in node_rows:
        categories[str(row["coverage_category"])] += 1
    worst_groups = sorted(group_rows, key=lambda row: _to_int(row["tp_delta_v3_minus_v2"]))[:8]
    best_projection = max(
        projection_rows,
        key=lambda row: (
            _to_int(row["projected_unique_tp_nodes_upper_bound"]),
            _to_float(row["projected_event_precision"]),
        ),
    )
    lines = [
        "# CADETS_E3 E4 v2-v3 Regression Policy and Speed Audit",
        "",
        f"Output directory: `{out_dir}`",
        "",
        "## Scope and Safety",
        "",
        "- Read-only audit over existing sidecars.",
        "- No tokenizer changes, no Word2Vec training, no Phase3E/Phase3G training, no inference.",
        "- GT/test labels are used only for post-stream evaluation columns.",
        "- Candidate rule descriptions are validation/runtime-visible projections, not runtime policy.",
        "",
        "## Headline Comparison",
        "",
        "| Run | Event alerts | Event TP | Event FP | Event precision | TP nodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| v2/Q09995 best | {v2_current['event_alerts']} | {v2_current['event_tp']} | "
        f"{v2_current['event_fp']} | "
        f"{v2_current['event_tp'] / v2_current['event_alerts'] if v2_current['event_alerts'] else 0:.4f} | "
        f"{v2_current['node_tp']} |",
        f"| v3 safe lexical | {v3_current['event_alerts']} | {v3_current['event_tp']} | "
        f"{v3_current['event_fp']} | "
        f"{v3_current['event_tp'] / v3_current['event_alerts'] if v3_current['event_alerts'] else 0:.4f} | "
        f"{v3_current['node_tp']} |",
        "",
        "## TP Node Delta",
        "",
        f"- both covered: {categories['both_covered']}",
        f"- v2 only: {categories['v2_only']}",
        f"- v3 only: {categories['v3_only']}",
        f"- both missed estimate: {categories['both_missed']}",
        "",
        "## Worst Group Regressions",
        "",
        "| Group | v2 TP | v3 TP | TP delta | v2 FP | v3 FP | Node delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in worst_groups:
        lines.append(
            f"| {row['action']} {row['src_type']}->{row['dst_type']} | "
            f"{row['v2_tp']} | {row['v3_tp']} | {row['tp_delta_v3_minus_v2']} | "
            f"{row['v2_fp']} | {row['v3_fp']} | {row['covered_node_delta_v3_minus_v2']} |"
        )
    display_speed_rows = [
        row
        for row in speed_rows
        if _to_int(row.get("events_scored")) > 0
        and (
            str(row.get("dataset")) in {"CADETS_E3", "CLEARSCOPE_E3", "CLEARSCOPE_E5", "THEIA_E3"}
            or str(row.get("run_name", "")).startswith("cadets_")
        )
    ]
    lines.extend(
        [
            "",
            "## Demotion Audit",
            "",
            "Demoted groups with strong post-hoc candidate evidence:",
        ]
    )
    for row in demotion_rows:
        if row["posthoc_actionability"] == "strong_posthoc_candidate":
            lines.append(
                f"- {row['action']} {row['src_type']}->{row['dst_type']}: "
                f"demoted={row['demoted_event_count']}, candidate TP={row['best_candidate_tp']}, "
                f"FP={row['best_candidate_fp']}, precision={row['best_candidate_precision']:.4f}, "
                f"covered_nodes={row['best_candidate_covered_nodes']}"
            )
    lines.extend(
        [
            "",
            "## Candidate Projection",
            "",
            f"Best projected upper-bound candidate by node coverage: "
            f"`{best_projection['candidate_name']}` with "
            f"{best_projection['projected_unique_tp_nodes_upper_bound']} TP nodes upper bound "
            f"and event precision {best_projection['projected_event_precision']:.4f}.",
            "",
            "Projection caveat: candidate rows use label-free, runtime-visible group identities and "
            "validation-style margins as the rule basis; TP/FP counts are post-stream labels and "
            "must not be used directly to choose production thresholds.",
            "",
            "## Speed Audit",
            "",
            "| Run | Dataset | Events | Throughput | DB mode | Lookup | Logical nodes | Notes |",
            "| --- | --- | ---: | ---: | --- | --- | ---: | --- |",
        ]
    )
    for row in sorted(
        display_speed_rows,
        key=lambda item: (
            str(item.get("dataset", "")) != "CADETS_E3",
            -_to_float(item.get("throughput_events_per_second")),
        ),
    )[:30]:
        lines.append(
            f"| {row['run_name']} | {row['dataset']} | {row['events_scored']} | "
            f"{row['throughput_events_per_second']:.2f} | {row['db_stream_mode']} | "
            f"{row['node_embedding_lookup_mode']} | {row['logical_nodes']} | "
            f"{row['likely_bottleneck_notes']} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Do not promote v3 as the new best result yet. First run a bounded, label-free policy "
            "candidate smoke focused on restoring validation-visible netflow recovery while "
            "suppressing high-FP process->file groups. In parallel, inspect CADETS runtime "
            "configuration because its python lookup mode, large logical-node count, and CSV "
            "streaming/evaluation overhead likely explain why it remains slower than ClearScope "
            "and THEIA despite v3 being faster than v2.",
        ]
    )
    return "\n".join(lines) + "\n"


def _collect_speed_payloads(v2_root: Path, v3_root: Path, extra_roots: Sequence[Path]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {
        "cadets_v2_q09995_e4": read_json(v2_root / "eval_causal_semantics_slim.json"),
        "cadets_v3_safe_lexical_e4": read_json(v3_root / "eval_causal_semantics_slim.json"),
    }
    for root in extra_roots:
        if root.is_file():
            candidates = [root]
        else:
            candidates = sorted(root.glob("**/eval_causal_semantics_slim.json"))
        for path in candidates:
            payload = read_json(path)
            dataset = str(payload.get("dataset", ""))
            if dataset and dataset != "CADETS_E3":
                key = f"{dataset}_{payload.get('out_tag', path.parent.name)}"
                payloads.setdefault(key, payload)
    return {key: value for key, value in payloads.items() if value}


def run_audit(args: argparse.Namespace) -> Path:
    """Run the read-only audit and return the output directory."""
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path(
        f"outputs/diagnostics/cadets_e3_e4_v2_v3_regression_policy_speed_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    v2_root = Path(args.v2_root)
    v3_root = Path(args.v3_root)
    v2_eval = read_json(v2_root / "eval_causal_semantics_slim.json")
    v3_eval = read_json(v3_root / "eval_causal_semantics_slim.json")

    v2_tp, v2_nodes = load_tp_nodes(v2_root / "online_event_node_coverage_strict.csv")
    v3_tp, v3_nodes = load_tp_nodes(v3_root / "online_event_node_coverage_strict.csv")
    total_gt_nodes = max(
        _summary_total_nodes(v2_root / "online_event_node_coverage_summary.json"),
        _summary_total_nodes(v3_root / "online_event_node_coverage_summary.json"),
        len(v2_tp | v3_tp),
    )

    node_rows = build_tp_node_delta_rows(
        v2_tp_nodes=v2_tp,
        v3_tp_nodes=v3_tp,
        total_gt_nodes=total_gt_nodes,
        v2_nodes=v2_nodes,
        v3_nodes=v3_nodes,
    )
    group_rows = build_group_regression_rows(
        read_csv_rows(v2_root / "group_alert_policy_summary.csv"),
        read_csv_rows(v3_root / "group_alert_policy_summary.csv"),
    )
    demotion_rows = join_demotion_with_sweep(
        read_csv_rows(v3_root / "demoted_group_summary.csv"),
        read_csv_rows(v3_root / "group_threshold_sweep_summary.csv"),
    )
    projection_rows = build_policy_candidate_projection_rows(
        current=_current_from_eval(v3_eval),
        baseline=_current_from_eval(v2_eval),
        sweep_rows=read_csv_rows(v3_root / "group_threshold_sweep_summary.csv"),
        total_gt_nodes=total_gt_nodes,
    )
    speed_payloads = _collect_speed_payloads(v2_root, v3_root, [Path(path) for path in args.speed_roots])
    speed_rows = build_speed_rows(speed_payloads)

    write_csv(
        out_dir / "v2_v3_tp_node_delta.csv",
        node_rows,
        [
            "node_idx",
            "coverage_category",
            "node_type",
            "v2_alert_count",
            "v3_alert_count",
            "v2_max_event_score",
            "v3_max_event_score",
        ],
    )
    write_csv(
        out_dir / "v2_v3_group_regression.csv",
        group_rows,
        [
            "action",
            "src_type",
            "dst_type",
            "focus_group",
            "v2_alert_count",
            "v3_alert_count",
            "alert_delta_v3_minus_v2",
            "v2_tp",
            "v3_tp",
            "tp_delta_v3_minus_v2",
            "v2_fp",
            "v3_fp",
            "fp_delta_v3_minus_v2",
            "v2_precision",
            "v3_precision",
            "v2_covered_nodes",
            "v3_covered_nodes",
            "covered_node_delta_v3_minus_v2",
            "v2_threshold",
            "v3_threshold",
        ],
    )
    write_csv(
        out_dir / "v3_demotion_audit.csv",
        demotion_rows,
        [
            "action",
            "src_type",
            "dst_type",
            "target_case",
            "demoted_event_count",
            "node_evidence_count",
            "best_candidate_threshold",
            "best_candidate_alert_count",
            "best_candidate_tp",
            "best_candidate_fp",
            "best_candidate_precision",
            "best_candidate_covered_nodes",
            "best_candidate_strict_node_tp",
            "best_candidate_strict_node_fp",
            "best_candidate_quantile",
            "best_candidate_margin",
            "candidate_rule_basis",
            "posthoc_actionability",
        ],
    )
    write_csv(
        out_dir / "v3_policy_candidate_projection.csv",
        projection_rows,
        [
            "candidate_name",
            "rule_source",
            "uses_test_labels_for_rule",
            "poststream_eval_uses_labels",
            "projected_event_alerts",
            "projected_event_tp",
            "projected_event_fp",
            "projected_event_precision",
            "projected_unique_tp_nodes_upper_bound",
            "projected_node_recall_upper_bound",
            "added_alerts",
            "added_tp",
            "added_fp",
            "added_covered_nodes_upper_bound",
        ],
    )
    write_csv(
        out_dir / "speed_audit_summary.csv",
        speed_rows,
        [
            "run_name",
            "dataset",
            "out_tag",
            "events_scored",
            "elapsed_seconds",
            "throughput_events_per_second",
            "db_stream_mode",
            "node_embedding_lookup_mode",
            "node_embedding_lazy_backing",
            "sspm_infer_backend",
            "logical_nodes",
            "physical_states",
            "deploy_infer_rss_peak_mb",
            "validation_seconds",
            "test_scoring_seconds",
            "stream_csv_write_seconds",
            "likely_bottleneck_notes",
        ],
    )
    report = _render_report(
        out_dir=out_dir,
        v2_eval=v2_eval,
        v3_eval=v3_eval,
        node_rows=node_rows,
        group_rows=group_rows,
        demotion_rows=demotion_rows,
        projection_rows=projection_rows,
        speed_rows=speed_rows,
    )
    (out_dir / "cadets_e3_e4_v2_v3_regression_policy_speed_report.md").write_text(
        report,
        encoding="utf-8",
    )
    return out_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v2-root",
        type=Path,
        default=Path("tmp/zhanzhongwei/tflr_light_phase3g_v2_Q09995_full/CADETS_E3_PHASE3E_E4_NONE"),
    )
    parser.add_argument(
        "--v3-root",
        type=Path,
        default=Path("outputs/results/tflr_light/CADETS_E3_FREEBSD_V3_SAFE_LEXICAL_E4_FULL"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--timestamp", default="")
    parser.add_argument(
        "--speed-roots",
        type=Path,
        nargs="*",
        default=[Path("outputs/results"), Path("tmp")],
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    out_dir = run_audit(args)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
