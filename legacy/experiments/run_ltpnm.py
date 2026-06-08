#!/usr/bin/env python3
"""Run Lightweight Temporal Provenance Normality Model diagnostics.

Purpose:
  P1/P2/P3/P5 runner for the LTPNM branch. P1 builds train-only
  categorical encoders and validates deterministic streaming encode behavior.
  P2 trains a lightweight temporal normality model on top of the same data
  interface, calibrates an alert threshold from validation scores, then scores
  test events in arrival order before updating online state. P3/P5 attach
  labels only after online alerts have been emitted and report event/node
  TP/FP, delay, throughput, RSS, and baseline comparison.

Inputs/outputs:
  Reads PostgreSQL event/node tables through the same streaming utilities as
  the current causal-semantics baseline. Writes `ltpnm_model_summary.json`,
  `eval_ltpnm.json`, P2 `online_event_alerts.csv`, and
  `online_node_alerts.csv` under the result directory.

Leakage:
  Vocabularies and model state are fit only from train rows. Validation rows
  are used only for calibration/checks. Test rows are streamed after all train
  fitting is frozen; score is computed before online state is updated.
  Labels/ground truth are not used before scoring in P1/P2/P3/P5; they are
  loaded only after raw online alerts have been produced.

Runtime/memory:
  The summary JSON records phase timings, RSS samples, vocabulary size, and P2
  online node/edge state sizes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.config.provnet_utils import PROCESS_NODE_TYPE, init_database_connection, log
from scripts.data.get_dataset import (
    fetch_node_tables,
    get_dataset_splits,
    get_ground_truth_paths,
    parse_split_days,
    resolve_gt_path,
    summarize_node_payload,
    tokenize_msg,
    use_event_type_filter,
)
from scripts.pipeline.io.db_stream import _cfg_for_dataset, _query_count, _stream_events
from scripts.pipeline.outputs.evaluation import memory_snapshot, target_for_dataset
from legacy.experiments.ltpnm import (
    LTPNM_COMPONENT_NAMES,
    LTPNMEncoderConfig,
    LTPNMEventEncoder,
    LTPNMNormalityModel,
    LTPNMReliabilityCalibrator,
    LTPNMReliabilityConfig,
    expected_alert_threshold,
)


def _build_node_maps_and_summaries(
    netflow_nodes: list[tuple[Any, ...]],
    process_nodes: list[tuple[Any, ...]],
    file_nodes: list[tuple[Any, ...]],
) -> tuple[dict[int, tuple[str, str]], dict[str, str], dict[str, tuple[str, int]], dict[str, int]]:
    indexid2summary: dict[int, tuple[str, str]] = {}
    hash2type: dict[str, str] = {}
    hash2uuid_index: dict[str, tuple[str, int]] = {}
    uuid2index: dict[str, int] = {}

    def add_node(uuid: Any, h: Any, index_id: Any, kind: str, payload: str) -> None:
        idx = int(index_id)
        hash_key = str(h)
        uuid_key = str(uuid)
        hash2type[hash_key] = str(kind)
        hash2uuid_index[hash_key] = (uuid_key, idx)
        uuid2index[uuid_key] = idx
        indexid2summary[idx] = (
            str(kind),
            " ".join(summarize_node_payload(str(kind), tokenize_msg(str(payload)))),
        )

    for uuid, h, src_addr, src_port, dst_addr, dst_port, index_id in netflow_nodes:
        remote = str(dst_addr or "")
        if os.getenv("CLAD_EVENT_NETFLOW_USE_PORT", "").strip().lower() in {"1", "true", "yes", "on"} and dst_port not in {None, ""}:
            remote = f"{remote}:{dst_port}"
        add_node(uuid, h, index_id, "netflow", remote)

    for uuid, h, path, cmd, index_id in process_nodes:
        payload = " ".join(part for part in [str(path or "").strip(), str(cmd or "").strip()] if part and part.lower() != "none")
        add_node(uuid, h, index_id, PROCESS_NODE_TYPE, payload)

    for uuid, h, path, index_id in file_nodes:
        add_node(uuid, h, index_id, "file", str(path or ""))

    return indexid2summary, hash2type, hash2uuid_index, uuid2index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="CLEARSCOPE_E3")
    p.add_argument("--out_tag", default="")
    p.add_argument("--result_root", default="outputs/results/tflr_light")
    p.add_argument("--phase", choices=["p1", "p2"], default="p1")
    p.add_argument("--fetch_size", type=int, default=100000)
    p.add_argument("--max_train_events", type=int, default=200000)
    p.add_argument("--max_ref_events", type=int, default=62748)
    p.add_argument("--max_test_events", type=int, default=1200000)
    p.add_argument("--stream_check_max_events", type=int, default=0)
    p.add_argument("--progress_interval_events", type=int, default=250000)
    p.add_argument("--max_action_vocab", type=int, default=256)
    p.add_argument("--max_action_family_vocab", type=int, default=64)
    p.add_argument("--max_object_type_vocab", type=int, default=64)
    p.add_argument("--max_coarse_vocab", type=int, default=4096)
    p.add_argument("--max_raw_vocab", type=int, default=8192)
    p.add_argument("--min_count", type=int, default=1)
    p.add_argument("--max_tokens_per_node", type=int, default=12)
    p.add_argument("--online_expected_alert_budget", type=int, default=1000)
    p.add_argument("--online_expected_alert_horizon_events", type=int, default=1200000)
    p.add_argument("--smoothing", type=float, default=0.5)
    p.add_argument("--raw_weight", type=float, default=0.25)
    p.add_argument("--raw_cap", type=float, default=4.0)
    p.add_argument("--memory_dim", type=int, default=16)
    p.add_argument("--memory_alpha", type=float, default=0.25)
    p.add_argument("--edge_max_keys", type=int, default=1000000)
    p.add_argument("--score_mode", choices=["fixed", "rcg"], default="fixed")
    p.add_argument("--rcg_epochs", type=int, default=4)
    p.add_argument("--rcg_learning_rate", type=float, default=0.08)
    p.add_argument("--rcg_l2", type=float, default=0.001)
    p.add_argument("--rcg_synthetic_per_event", type=int, default=2)
    p.add_argument("--rcg_max_trace_events", type=int, default=200000)
    p.add_argument("--rcg_edge_only_penalty", type=float, default=0.35)
    p.add_argument("--rcg_q_high", type=float, default=0.95)
    p.add_argument("--rcg_uniform_prior_weight", type=float, default=0.35)
    p.add_argument("--rcg_seed", type=int, default=13)
    p.add_argument("--online_max_event_alerts", type=int, default=200000)
    p.add_argument("--online_max_node_alerts", type=int, default=200000)
    p.add_argument("--print_summary", action="store_true")
    return p.parse_args()


def _check_limit(max_events: int, check_max_events: int) -> int:
    if int(check_max_events) <= 0:
        return int(max_events)
    if int(max_events) <= 0:
        return int(check_max_events)
    return int(min(int(max_events), int(check_max_events)))


def _stream(
    conn: Any,
    year_month: str,
    days: list[int],
    indexid2summary: dict[int, tuple[str, str]],
    hash2type: dict[str, str],
    hash2uuid_index: dict[str, tuple[str, int]],
    event_filter: bool,
    fetch_size: int,
    max_events: int,
    abnormal_nodes: set[int] | None = None,
):
    return _stream_events(
        conn,
        year_month,
        days,
        indexid2summary,
        hash2type,
        hash2uuid_index,
        set() if abnormal_nodes is None else set(abnormal_nodes),
        event_filter,
        int(fetch_size),
        int(max_events),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _score_event_with_mode(
    normality: LTPNMNormalityModel,
    event: Any,
    state: dict[str, Any],
    score_mode: str,
    calibrator: LTPNMReliabilityCalibrator | None,
) -> tuple[float, dict[str, Any]]:
    fixed_score, detail_raw = normality.score_pre_update(event, state)
    detail: dict[str, Any] = dict(detail_raw)
    detail["fixed_score"] = float(fixed_score)
    detail["fixed_top_component"] = str(detail.get("top_component", ""))
    if str(score_mode) != "rcg":
        detail["score_mode"] = "fixed"
        detail["score_form"] = "streaming_conditional_nll_plus_memory_residual"
        return float(fixed_score), detail
    if calibrator is None:
        raise RuntimeError("score_mode=rcg requires an LTPNMReliabilityCalibrator")
    rcg_score, rcg_detail = calibrator.score_detail(detail)
    detail.update(rcg_detail)
    detail["score_mode"] = "rcg"
    detail["score_form"] = "reliability_calibrated_component_gate"
    detail["top_component"] = str(rcg_detail.get("rcg_top_component", detail.get("top_component", "")))
    return float(rcg_score), detail


def _alert_component_fields(detail: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "fixed_score": float(detail.get("fixed_score", 0.0)),
        "score_mode": str(detail.get("score_mode", "")),
        "score_form": str(detail.get("score_form", "")),
        "top_component": str(detail.get("top_component", "")),
        "fixed_top_component": str(detail.get("fixed_top_component", "")),
        "rcg_top_component": str(detail.get("rcg_top_component", "")),
        "rcg_edge_reliability": float(detail.get("rcg_edge_reliability", 0.0)),
        "rcg_component_agreement": float(detail.get("rcg_component_agreement", 0.0)),
        "rcg_edge_only_flag": int(detail.get("rcg_edge_only_flag", 0) or 0),
        "rcg_reliability_reason": str(detail.get("rcg_reliability_reason", "")),
        "rcg_contributions_json": json.dumps(detail.get("rcg_contributions", {}), sort_keys=True),
    }
    for name in LTPNM_COMPONENT_NAMES:
        out[name] = float(detail.get(name, 0.0))
        out[f"{name}_weight"] = float(detail.get(f"{name}_weight", 1.0))
        out[f"{name}_z"] = float(detail.get(f"{name}_z", 0.0))
    return out


def _append_numeric(values: list[float], value: Any) -> None:
    if value in {"", None}:
        return
    try:
        values.append(float(value))
    except (TypeError, ValueError):
        return


def _event_alert_metrics(rows: list[dict[str, Any]], test_label_counts: dict[str, int]) -> dict[str, Any]:
    suspicious = 0
    malicious = 0
    missing = 0
    delay_events: list[float] = []
    delay_seconds: list[float] = []
    for row in rows:
        label = int(row.get("event_label", -1))
        if label < 0:
            missing += 1
        elif label == 1:
            suspicious += 1
        elif label == 2:
            malicious += 1
        _append_numeric(delay_events, row.get("detection_delay_events"))
        _append_numeric(delay_seconds, row.get("detection_delay_seconds"))

    correct = int(suspicious + malicious)
    total = int(len(rows))
    positives = int(test_label_counts.get("suspect_1", 0) + test_label_counts.get("positive_2", 0))
    false_alerts = int(total - correct)
    return {
        "metric_level": "event_alert",
        "num_alerts": total,
        "tp": correct,
        "fp": false_alerts,
        "correct_event_alerts": correct,
        "false_event_alerts": false_alerts,
        "suspicious_event_alerts": int(suspicious),
        "malicious_event_alerts": int(malicious),
        "missing_event_labels": int(missing),
        "test_positive_or_suspicious_events": positives,
        "precision": float(correct / max(total, 1)),
        "recall": float(correct / max(positives, 1)),
        "mean_detection_delay_events": float(np.mean(delay_events)) if delay_events else None,
        "median_detection_delay_events": float(np.median(delay_events)) if delay_events else None,
        "p90_detection_delay_events": float(np.quantile(np.asarray(delay_events, dtype=np.float32), 0.90)) if delay_events else None,
        "mean_detection_delay_seconds": float(np.mean(delay_seconds)) if delay_seconds else None,
        "median_detection_delay_seconds": float(np.median(delay_seconds)) if delay_seconds else None,
        "p90_detection_delay_seconds": float(np.quantile(np.asarray(delay_seconds, dtype=np.float32), 0.90)) if delay_seconds else None,
        "correct_rule": "event_label in {1, 2}; labels are used only after online alert emission for evaluation",
    }


def _node_alert_metrics(
    rows: list[dict[str, Any]],
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    first_positive_event: np.ndarray,
) -> dict[str, Any]:
    num_positive = int(np.sum(positive_nodes))
    tp_any = 0
    tp_after = 0
    early_positive = 0
    fp_strict = 0
    fp_relaxed = 0
    ignored_relaxed = 0
    delay_events: list[float] = []
    delay_seconds: list[float] = []
    for row in rows:
        node = int(row.get("node_id", -1))
        is_positive = bool(0 <= node < positive_nodes.shape[0] and positive_nodes[node])
        is_suspect = bool(0 <= node < suspect_nodes.shape[0] and suspect_nodes[node])
        if is_positive:
            tp_any += 1
            if int(row.get("is_tp_after_attack_start", 0) or 0):
                tp_after += 1
                _append_numeric(delay_events, row.get("detection_delay_events"))
                _append_numeric(delay_seconds, row.get("detection_delay_seconds"))
            else:
                early_positive += 1
        else:
            fp_strict += 1
            if is_suspect:
                ignored_relaxed += 1
            else:
                fp_relaxed += 1
    total = int(len(rows))
    return {
        "metric_level": "unique_endpoint_node_alert",
        "node_alert_policy": "event_endpoint_first_alert",
        "num_alerts": total,
        "num_unique_alerted_nodes": total,
        "num_positive_nodes": num_positive,
        "tp_anytime": int(tp_any),
        "tp_after_attack_start": int(tp_after),
        "early_positive_alerts": int(early_positive),
        "fp_strict": int(fp_strict),
        "fp_relaxed": int(fp_relaxed),
        "ignored_suspect_relaxed": int(ignored_relaxed),
        "precision_anytime_strict": float(tp_any / max(tp_any + fp_strict, 1)),
        "recall_anytime": float(tp_any / max(num_positive, 1)),
        "precision_after_attack_strict": float(tp_after / max(tp_after + fp_strict + early_positive, 1)),
        "recall_after_attack": float(tp_after / max(num_positive, 1)),
        "precision_after_attack_relaxed": float(tp_after / max(tp_after + fp_relaxed + early_positive, 1)),
        "mean_delay_events": float(np.mean(delay_events)) if delay_events else None,
        "median_delay_events": float(np.median(delay_events)) if delay_events else None,
        "p90_delay_events": float(np.quantile(np.asarray(delay_events, dtype=np.float32), 0.90)) if delay_events else None,
        "mean_delay_seconds": float(np.mean(delay_seconds)) if delay_seconds else None,
        "median_delay_seconds": float(np.median(delay_seconds)) if delay_seconds else None,
        "p90_delay_seconds": float(np.quantile(np.asarray(delay_seconds, dtype=np.float32), 0.90)) if delay_seconds else None,
        "relaxed_semantics": "GT abnormal nodes observed in positive/suspect events count as TP; suspect-only benign endpoints are ignored in relaxed FP.",
    }


def _rough_size_bytes(obj: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, np.ndarray):
        return int(sys.getsizeof(obj) + obj.nbytes)
    size = int(sys.getsizeof(obj))
    if isinstance(obj, dict):
        for key, value in obj.items():
            size += _rough_size_bytes(key, seen)
            size += _rough_size_bytes(value, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            size += _rough_size_bytes(value, seen)
    elif hasattr(obj, "__dict__"):
        size += _rough_size_bytes(vars(obj), seen)
    return int(size)


def _load_baseline_summary(dataset: str, result_root: Path) -> dict[str, Any]:
    path = result_root / f"{dataset}_CAUSAL_SEMANTICS_BUDGET1000_1P2M" / "eval_causal_semantics.json"
    if not path.exists():
        return {"available": False, "path": str(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"available": False, "path": str(path), "error": str(exc)}
    primary = payload.get("primary_online_metrics", {})
    immediate = primary.get("immediate_candidate_alerts", {}) if isinstance(primary, dict) else {}
    confirmed = primary.get("bounded_delay_confirmed_alerts", {}) if isinstance(primary, dict) else {}
    return {
        "available": True,
        "path": str(path),
        "method": "CAUSAL_SEMANTICS_BUDGET1000_1P2M",
        "immediate_event_alerts": immediate.get("event_alerts", {}),
        "immediate_node_alerts": immediate.get("node_alerts", {}),
        "confirmed_event_alerts": confirmed.get("event_alerts", {}),
        "confirmed_node_alerts": confirmed.get("node_alerts", {}),
        "runtime": payload.get("runtime", {}),
        "memory": payload.get("memory", {}),
    }


def _load_ground_truth_indices_readonly(cfg: Any, uuid2index: dict[str, int]) -> set[int]:
    gt_indices: set[int] = set()
    for rel_path in get_ground_truth_paths(cfg):
        abs_path = resolve_gt_path(str(rel_path))
        if not os.path.exists(abs_path):
            log(f"[WARN] ground truth file not found: {abs_path}")
            continue
        with open(abs_path, "r", encoding="utf-8", newline="") as handle:
            first_line = handle.readline()
            delimiter = "\t" if "\t" in first_line else ","
            handle.seek(0)
            reader = csv.reader(handle, delimiter=delimiter)
            for row in reader:
                if not row:
                    continue
                uuid = str(row[0]).strip()
                fallback = None
                if len(row) >= 3 and str(row[2]).strip():
                    try:
                        fallback = int(float(str(row[2]).strip()))
                    except ValueError:
                        fallback = None
                idx = uuid2index.get(uuid, fallback)
                if idx is not None:
                    gt_indices.add(int(idx))
    return gt_indices


def _metric_delta(new_value: Any, old_value: Any) -> float | None:
    try:
        return float(new_value) - float(old_value)
    except (TypeError, ValueError):
        return None


def _baseline_comparison(
    dataset: str,
    result_root: Path,
    event_metrics: dict[str, Any],
    node_metrics: dict[str, Any],
    runtime: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any]:
    baseline = _load_baseline_summary(dataset, result_root)
    if not baseline.get("available"):
        return {"baseline": baseline, "note": "Baseline result not found; comparison unavailable."}
    be = baseline.get("immediate_event_alerts", {})
    bn = baseline.get("immediate_node_alerts", {})
    br = baseline.get("runtime", {})
    bm = baseline.get("memory", {})
    return {
        "baseline": baseline,
        "comparison_level": "LTPNM immediate event alerts vs CAUSAL_SEMANTICS_BUDGET1000_1P2M immediate candidate alerts",
        "event": {
            "ltpnm_alerts": int(event_metrics.get("num_alerts", 0)),
            "baseline_alerts": int(be.get("num_alerts", 0) or 0),
            "ltpnm_tp_correct": int(event_metrics.get("tp", 0)),
            "baseline_tp_correct": int(be.get("correct_event_alerts", 0) or 0),
            "ltpnm_fp": int(event_metrics.get("fp", 0)),
            "baseline_fp": int(be.get("false_event_alerts", 0) or 0),
            "precision_delta": _metric_delta(event_metrics.get("precision"), be.get("correct_alert_ratio")),
            "tp_delta": _metric_delta(event_metrics.get("tp"), be.get("correct_event_alerts")),
            "fp_delta": _metric_delta(event_metrics.get("fp"), be.get("false_event_alerts")),
        },
        "node": {
            "ltpnm_policy": node_metrics.get("node_alert_policy"),
            "baseline_policy": "causal_semantics_online_node_alerts",
            "ltpnm_alerts": int(node_metrics.get("num_alerts", 0)),
            "baseline_alerts": int(bn.get("num_alerts", 0) or 0),
            "ltpnm_tp_after_attack_start": int(node_metrics.get("tp_after_attack_start", 0)),
            "baseline_tp_after_attack_start": int(bn.get("tp_after_attack_start", 0) or 0),
            "ltpnm_fp_strict": int(node_metrics.get("fp_strict", 0)),
            "baseline_fp_strict": int(bn.get("fp_strict", 0) or 0),
            "tp_delta": _metric_delta(node_metrics.get("tp_after_attack_start"), bn.get("tp_after_attack_start")),
            "fp_strict_delta": _metric_delta(node_metrics.get("fp_strict"), bn.get("fp_strict")),
        },
        "runtime_memory": {
            "ltpnm_test_logs_per_second": runtime.get("p2_test_logs_per_second"),
            "baseline_test_logs_per_second": br.get("test_logs_per_second"),
            "ltpnm_peak_rss_mb": memory.get("peak_rss_mb"),
            "baseline_peak_rss_mb": bm.get("peak_rss_mb"),
            "test_lps_delta": _metric_delta(runtime.get("p2_test_logs_per_second"), br.get("test_logs_per_second")),
            "peak_rss_mb_delta": _metric_delta(memory.get("peak_rss_mb"), bm.get("peak_rss_mb")),
        },
    }


def _label_alert_outputs(
    conn: Any,
    cfg: Any,
    year_month: str,
    test_days: list[int],
    indexid2summary: dict[int, tuple[str, str]],
    hash2type: dict[str, str],
    hash2uuid_index: dict[str, tuple[str, int]],
    uuid2index: dict[str, int],
    result_dir: Path,
    event_filter: bool,
    fetch_size: int,
    max_test_events: int,
    event_alerts: list[dict[str, Any]],
    node_alerts: list[dict[str, Any]],
) -> dict[str, Any]:
    phase_start = time.perf_counter()
    abnormal_nodes = _load_ground_truth_indices_readonly(cfg, uuid2index)
    event_alerts_by_pos: dict[int, list[dict[str, Any]]] = {}
    for row in event_alerts:
        event_alerts_by_pos.setdefault(int(row.get("event_pos", -1)), []).append(row)

    alert_nodes = [int(row.get("node_id", -1)) for row in node_alerts if int(row.get("node_id", -1)) >= 0]
    max_node_id = max([0, *[int(x) for x in uuid2index.values()], *alert_nodes]) + 1
    positive_nodes = np.zeros((max_node_id,), dtype=bool)
    suspect_nodes = np.zeros((max_node_id,), dtype=bool)
    first_positive_event = np.full((max_node_id,), -1, dtype=np.int64)
    first_positive_ts = np.full((max_node_id,), -1, dtype=np.int64)

    label_counts = {"benign_0": 0, "suspect_1": 0, "positive_2": 0, "unknown": 0}
    labeled_events = 0
    for event_pos, row in enumerate(
        _stream(
            conn,
            year_month,
            test_days,
            indexid2summary,
            hash2type,
            hash2uuid_index,
            event_filter,
            int(fetch_size),
            int(max_test_events),
            abnormal_nodes=abnormal_nodes,
        )
    ):
        label = int(row.get("label", -1))
        if label == 0:
            label_counts["benign_0"] += 1
        elif label == 1:
            label_counts["suspect_1"] += 1
        elif label == 2:
            label_counts["positive_2"] += 1
        else:
            label_counts["unknown"] += 1
        for alert_row in event_alerts_by_pos.get(int(event_pos), []):
            alert_row["event_label"] = int(label)
            alert_row["is_correct_event_alert"] = int(label in {1, 2})
            alert_row["is_suspicious_event"] = int(label == 1)
            alert_row["is_malicious_event"] = int(label == 2)
            alert_row["detection_delay_events"] = 0 if label in {1, 2} else ""
            alert_row["detection_delay_seconds"] = 0.0 if label in {1, 2} else ""
        for node in [int(row["src_idx"]), int(row["dst_idx"])]:
            if node < 0 or node >= max_node_id:
                continue
            if label == 2:
                positive_nodes[node] = True
                if int(first_positive_event[node]) < 0:
                    first_positive_event[node] = np.int64(event_pos)
                    first_positive_ts[node] = np.int64(row["timestamp_ns"])
            elif label == 1:
                if node in abnormal_nodes:
                    positive_nodes[node] = True
                    if int(first_positive_event[node]) < 0:
                        first_positive_event[node] = np.int64(event_pos)
                        first_positive_ts[node] = np.int64(row["timestamp_ns"])
                else:
                    suspect_nodes[node] = True
        labeled_events += 1

    for row in node_alerts:
        node = int(row.get("node_id", -1))
        is_positive = bool(0 <= node < max_node_id and positive_nodes[node])
        is_suspect = bool(0 <= node < max_node_id and suspect_nodes[node])
        first_pos = int(first_positive_event[node]) if 0 <= node < max_node_id else -1
        first_ts = int(first_positive_ts[node]) if 0 <= node < max_node_id else -1
        alert_pos = int(row.get("event_pos", -1))
        alert_ts = int(row.get("timestamp_ns", -1))
        is_after = bool(is_positive and first_pos >= 0 and alert_pos >= first_pos)
        is_early = bool(is_positive and first_pos >= 0 and alert_pos < first_pos)
        row["is_gt_positive"] = int(is_positive)
        row["is_suspect"] = int(is_suspect)
        row["first_positive_event_pos"] = int(first_pos)
        row["first_positive_timestamp_ns"] = int(first_ts)
        row["detection_delay_events"] = int(alert_pos - first_pos) if is_after else ""
        row["detection_delay_seconds"] = float((alert_ts - first_ts) / 1_000_000_000.0) if is_after else ""
        row["is_tp_anytime"] = int(is_positive)
        row["is_tp_after_attack_start"] = int(is_after)
        row["is_early_positive_alert"] = int(is_early)
        row["is_fp_strict"] = int(not is_positive)
        row["is_fp_relaxed"] = int((not is_positive) and (not is_suspect))

    elapsed = float(time.perf_counter() - phase_start)
    return {
        "abnormal_nodes_loaded": int(len(abnormal_nodes)),
        "labeled_test_events": int(labeled_events),
        "test_label_counts": label_counts,
        "positive_nodes_observed": int(np.sum(positive_nodes)),
        "suspect_nodes_observed": int(np.sum(suspect_nodes)),
        "event_metrics": _event_alert_metrics(event_alerts, label_counts),
        "node_metrics": _node_alert_metrics(node_alerts, positive_nodes, suspect_nodes, first_positive_event),
        "evaluation_label_seconds": elapsed,
        "evaluation_label_logs_per_second": float(labeled_events / max(elapsed, 1e-12)),
        "leakage_note": "Ground truth is loaded only in this post-stream evaluation pass after raw online alerts have been emitted.",
    }


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    out_tag = str(args.out_tag).strip() or f"{args.dataset}_LTPNM_P1"
    result_dir = Path(args.result_root).joinpath(out_tag).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    memory_samples: list[dict[str, Any]] = []
    cfg = _cfg_for_dataset(args.dataset)
    conn_cur, conn = init_database_connection(cfg)
    try:
        phase_start = time.perf_counter()
        log("[LTPNM] loading node tables")
        netflow_nodes, process_nodes, file_nodes = fetch_node_tables(conn_cur)
        indexid2summary, hash2type, hash2uuid_index, uuid2index = _build_node_maps_and_summaries(
            netflow_nodes,
            process_nodes,
            file_nodes,
        )
        del netflow_nodes
        del process_nodes
        del file_nodes

        year_month = cfg.dataset.year_month
        train_days = parse_split_days(get_dataset_splits(cfg, "train"))
        val_days = parse_split_days(get_dataset_splits(cfg, "val"))
        test_days = parse_split_days(get_dataset_splits(cfg, "test"))
        event_filter = use_event_type_filter(cfg)
        counts_meta = {
            "train_query_events": _query_count(conn_cur, year_month, train_days, event_filter),
            "val_query_events": _query_count(conn_cur, year_month, val_days, event_filter),
            "test_query_events": _query_count(conn_cur, year_month, test_days, event_filter),
        }
        metadata_seconds = float(time.perf_counter() - phase_start)
        memory_samples.append({"phase": "after_metadata", **memory_snapshot()})
        log(f"[LTPNM] split query counts: {counts_meta}")

        encoder = LTPNMEventEncoder(
            LTPNMEncoderConfig(
                max_action_vocab=int(args.max_action_vocab),
                max_action_family_vocab=int(args.max_action_family_vocab),
                max_object_type_vocab=int(args.max_object_type_vocab),
                max_coarse_vocab=int(args.max_coarse_vocab),
                max_raw_vocab=int(args.max_raw_vocab),
                min_count=int(args.min_count),
                max_tokens_per_node=int(args.max_tokens_per_node),
            )
        )

        phase_start = time.perf_counter()
        train_seen = 0
        log("[LTPNM] P1 fitting train-only categorical vocabularies")
        for row in _stream(conn, year_month, train_days, indexid2summary, hash2type, hash2uuid_index, event_filter, int(args.fetch_size), int(args.max_train_events)):
            encoder.observe_train(row)
            train_seen += 1
            if train_seen % 1_000_000 == 0:
                log(f"[LTPNM] train vocab processed={train_seen}")
        encoder.freeze()
        train_seconds = float(time.perf_counter() - phase_start)
        memory_samples.append({"phase": "after_p1_train_vocab", **memory_snapshot()})
        log(f"[LTPNM] P1 vocab frozen train_events={train_seen}")

        phase_start = time.perf_counter()
        train_check = encoder.stream_check(
            _stream(
                conn,
                year_month,
                train_days,
                indexid2summary,
                hash2type,
                hash2uuid_index,
                event_filter,
                int(args.fetch_size),
                _check_limit(int(args.max_train_events), int(args.stream_check_max_events)),
            )
        )
        val_check = encoder.stream_check(
            _stream(
                conn,
                year_month,
                val_days,
                indexid2summary,
                hash2type,
                hash2uuid_index,
                event_filter,
                int(args.fetch_size),
                _check_limit(int(args.max_ref_events), int(args.stream_check_max_events)),
            )
        )
        test_check = encoder.stream_check(
            _stream(
                conn,
                year_month,
                test_days,
                indexid2summary,
                hash2type,
                hash2uuid_index,
                event_filter,
                int(args.fetch_size),
                _check_limit(int(args.max_test_events), int(args.stream_check_max_events)),
            )
        )
        check_seconds = float(time.perf_counter() - phase_start)
        memory_samples.append({"phase": "after_p1_stream_checks", **memory_snapshot()})

        model_summary: dict[str, Any] = {}
        threshold: dict[str, Any] = {}
        val_scores: list[float] = []
        event_alerts: list[dict[str, Any]] = []
        node_alerts: list[dict[str, Any]] = []
        p2_train_seconds = 0.0
        p2_validation_seconds = 0.0
        p2_test_seconds = 0.0
        p3_evaluation_seconds = 0.0
        p2_test_seen = 0
        p2_state_profile: dict[str, int] = {}
        evaluation_summary: dict[str, Any] = {}
        p2_model_approx_state_mb = 0.0
        rcg_training_summary: dict[str, Any] = {}
        rcg_validation_trace_count = 0
        rcg_training_seconds = 0.0
        if str(args.phase) == "p2":
            phase_start = time.perf_counter()
            log("[LTPNM] P2 fitting train-only temporal normality tables")
            normality = LTPNMNormalityModel(
                encoder,
                smoothing=float(args.smoothing),
                raw_weight=float(args.raw_weight),
                raw_cap=float(args.raw_cap),
                memory_dim=int(args.memory_dim),
                memory_alpha=float(args.memory_alpha),
                edge_max_keys=int(args.edge_max_keys),
            )
            train_model_state = normality.new_state()
            p2_train_seen = 0
            for row in _stream(conn, year_month, train_days, indexid2summary, hash2type, hash2uuid_index, event_filter, int(args.fetch_size), int(args.max_train_events)):
                event = encoder.encode(row)
                normality.fit_event(event)
                normality.update_state(event, train_model_state)
                p2_train_seen += 1
                if p2_train_seen % 1_000_000 == 0:
                    log(f"[LTPNM] P2 train model processed={p2_train_seen}")
            p2_train_seconds = float(time.perf_counter() - phase_start)
            memory_samples.append({"phase": "after_p2_train_model", **memory_snapshot()})

            score_mode = str(args.score_mode)
            rcg: LTPNMReliabilityCalibrator | None = None
            if score_mode == "rcg":
                rcg = LTPNMReliabilityCalibrator(
                    LTPNMReliabilityConfig(
                        enabled=True,
                        train_gate=True,
                        epochs=int(args.rcg_epochs),
                        learning_rate=float(args.rcg_learning_rate),
                        l2=float(args.rcg_l2),
                        synthetic_per_event=int(args.rcg_synthetic_per_event),
                        max_trace_events=int(args.rcg_max_trace_events),
                        edge_only_penalty=float(args.rcg_edge_only_penalty),
                        q_high=float(args.rcg_q_high),
                        uniform_prior_weight=float(args.rcg_uniform_prior_weight),
                        seed=int(args.rcg_seed),
                    )
                )
                phase_start = time.perf_counter()
                log("[LTPNM] RCG collecting train-only normality trace")
                rcg_trace: list[dict[str, Any]] = []
                trace_train_limit = max(int(args.rcg_max_trace_events), 1)
                trace_state = normality.new_state()
                for row in _stream(conn, year_month, train_days, indexid2summary, hash2type, hash2uuid_index, event_filter, int(args.fetch_size), min(int(args.max_train_events), trace_train_limit)):
                    event = encoder.encode(row)
                    _score, detail = normality.score_pre_update(event, trace_state)
                    rcg_trace.append(dict(detail))
                    normality.update_state(event, trace_state)
                    if len(rcg_trace) >= trace_train_limit:
                        break
                rcg_validation_trace_count = 0
                rcg_training_summary = rcg.fit_gate(rcg_trace)
                rcg_training_seconds = float(time.perf_counter() - phase_start)
                memory_samples.append({"phase": "after_rcg_train_gate", **memory_snapshot(), "rcg_trace_events": int(len(rcg_trace))})
                del rcg_trace

            phase_start = time.perf_counter()
            log("[LTPNM] P2 scoring validation for threshold calibration")
            val_state = normality.new_state()
            for row in _stream(conn, year_month, val_days, indexid2summary, hash2type, hash2uuid_index, event_filter, int(args.fetch_size), int(args.max_ref_events)):
                event = encoder.encode(row)
                score, _detail = _score_event_with_mode(normality, event, val_state, score_mode, rcg)
                val_scores.append(float(score))
                normality.update_state(event, val_state)
            threshold = expected_alert_threshold(
                val_scores,
                int(args.online_expected_alert_budget),
                int(args.online_expected_alert_horizon_events),
            )
            p2_validation_seconds = float(time.perf_counter() - phase_start)
            memory_samples.append({"phase": "after_p2_validation", **memory_snapshot()})

            phase_start = time.perf_counter()
            log("[LTPNM] P2 scoring test stream online")
            test_state = normality.new_state()
            alert_rank = 0
            node_alert_rank = 0
            test_seen = 0
            alerted_nodes: set[int] = set()
            score_threshold = float(threshold.get("threshold", float("inf")))
            for row in _stream(conn, year_month, test_days, indexid2summary, hash2type, hash2uuid_index, event_filter, int(args.fetch_size), int(args.max_test_events)):
                event = encoder.encode(row)
                score, detail = _score_event_with_mode(normality, event, test_state, score_mode, rcg)
                if score >= score_threshold and len(event_alerts) < int(args.online_max_event_alerts):
                    alert_rank += 1
                    row_out = {
                        "alert_rank": int(alert_rank),
                        "event_pos": int(test_seen),
                        "timestamp_ns": int(event.timestamp_ns),
                        "src_idx": int(event.src_idx),
                        "dst_idx": int(event.dst_idx),
                        "score": float(score),
                        "threshold": float(score_threshold),
                        "threshold_source": str(threshold.get("source", "")),
                        **_alert_component_fields(detail),
                        "action": event.action,
                        "action_family": event.action_family,
                        "object_type": event.object_type,
                        "coarse_src": event.coarse_src,
                        "coarse_dst": event.coarse_dst,
                        "raw_src": event.raw_src,
                        "raw_dst": event.raw_dst,
                        "event_label": -1,
                        "explanation_json": json.dumps(
                            {
                                "score_form": str(detail.get("score_form", "")),
                                "threshold_source": str(threshold.get("source", "")),
                                **detail,
                            },
                            sort_keys=True,
                        ),
                    }
                    event_alerts.append(row_out)
                    if len(node_alerts) < int(args.online_max_node_alerts):
                        for node_id, peer_id in [(int(event.src_idx), int(event.dst_idx)), (int(event.dst_idx), int(event.src_idx))]:
                            if node_id in alerted_nodes or len(node_alerts) >= int(args.online_max_node_alerts):
                                continue
                            alerted_nodes.add(node_id)
                            node_alert_rank += 1
                            node_alert = {
                                "alert_rank": int(node_alert_rank),
                                "event_pos": int(test_seen),
                                "timestamp_ns": int(event.timestamp_ns),
                                "node_id": int(node_id),
                                "peer_idx": int(peer_id),
                                "score": float(score),
                                "threshold": float(score_threshold),
                                "threshold_source": str(threshold.get("source", "")),
                                "node_alert_policy": "event_endpoint_first_alert",
                                "trigger_event_alert_rank": int(alert_rank),
                                **_alert_component_fields(detail),
                                "src_idx": int(event.src_idx),
                                "dst_idx": int(event.dst_idx),
                                "action": event.action,
                                "action_family": event.action_family,
                                "object_type": event.object_type,
                                "coarse_src": event.coarse_src,
                                "coarse_dst": event.coarse_dst,
                                "raw_src": event.raw_src,
                                "raw_dst": event.raw_dst,
                                "explanation_json": json.dumps(
                                    {
                                        "score_form": str(detail.get("score_form", "")),
                                        "threshold_source": str(threshold.get("source", "")),
                                        "node_alert_policy": "event_endpoint_first_alert",
                                        "trigger_event_alert_rank": int(alert_rank),
                                        **detail,
                                    },
                                    sort_keys=True,
                                ),
                            }
                            node_alerts.append(node_alert)
                normality.update_state(event, test_state)
                test_seen += 1
                if int(args.progress_interval_events) > 0 and test_seen % int(args.progress_interval_events) == 0:
                    elapsed = float(time.perf_counter() - phase_start)
                    log(
                        "[LTPNM] P2 test processed="
                        f"{test_seen} alerts={len(event_alerts)} nodes={len(node_alerts)} "
                        f"lps={test_seen / max(elapsed, 1e-12):.2f}"
                    )
            p2_test_seen = int(test_seen)
            p2_state_profile = normality.state_profile(test_state)
            p2_test_seconds = float(time.perf_counter() - phase_start)
            memory_samples.append({"phase": "after_p2_test", **memory_snapshot(), **p2_state_profile})
            model_summary = normality.summary()
            if rcg is not None:
                model_summary["rcg"] = rcg.summary()
            p2_model_approx_state_mb = float(_rough_size_bytes(normality) / (1024.0 * 1024.0))
            if rcg is not None:
                p2_model_approx_state_mb += float(_rough_size_bytes(rcg) / (1024.0 * 1024.0))
            _write_csv(result_dir / "online_event_alerts.raw.csv", event_alerts)
            _write_csv(result_dir / "online_node_alerts.raw.csv", node_alerts)
            log("[LTPNM] P3 labeling online alerts for post-stream evaluation")
            evaluation_summary = _label_alert_outputs(
                conn,
                cfg,
                year_month,
                test_days,
                indexid2summary,
                hash2type,
                hash2uuid_index,
                uuid2index,
                result_dir,
                event_filter,
                int(args.fetch_size),
                int(args.max_test_events),
                event_alerts,
                node_alerts,
            )
            p3_evaluation_seconds = float(evaluation_summary.get("evaluation_label_seconds", 0.0))
            memory_samples.append(
                {
                    "phase": "after_p3_evaluation_labeling",
                    "test_processed": int(evaluation_summary.get("labeled_test_events", 0)),
                    **memory_snapshot(),
                }
            )
            _write_csv(result_dir / "online_event_alerts.csv", event_alerts)
            _write_csv(result_dir / "online_node_alerts.csv", node_alerts)

        total_seconds = float(time.perf_counter() - t0)
        phase_name = str(args.phase)
        method_name = "ltpnm_p2_temporal_normality_p3_p5_eval" if phase_name == "p2" else "ltpnm_p1_data_interface"
        purpose = (
            "P2/P3/P5: train-only temporal normality model, validation-calibrated streaming event alerts, post-stream event/node evaluation, and baseline comparison."
            if phase_name == "p2"
            else "P1 only: train-only categorical encoders and deterministic streaming event adapter for LTPNM."
        )
        runtime_summary = {
            "metadata_seconds": float(metadata_seconds),
            "train_seconds": float(train_seconds),
            "check_seconds": float(check_seconds),
            "p2_train_seconds": float(p2_train_seconds),
            "rcg_training_seconds": float(rcg_training_seconds),
            "p2_validation_seconds": float(p2_validation_seconds),
            "p2_test_seconds": float(p2_test_seconds),
            "p3_evaluation_seconds": float(p3_evaluation_seconds),
            "total_seconds": float(total_seconds),
            "fetch_size": int(args.fetch_size),
            "p2_test_logs_per_second": float(p2_test_seen / max(p2_test_seconds, 1e-12)) if p2_test_seen else 0.0,
            "p3_evaluation_logs_per_second": float(evaluation_summary.get("evaluation_label_logs_per_second", 0.0)),
        }
        memory_summary = {
            "definition": "RSS process memory plus approximate Python vocabulary/model/online state samples.",
            "samples": memory_samples,
            "max_current_rss_mb": float(max((sample.get("current_rss_mb", 0.0) for sample in memory_samples), default=0.0)),
            "peak_rss_mb": float(max((sample.get("peak_rss_mb", 0.0) for sample in memory_samples), default=0.0)),
            "approx_vocab_state_mb": float(encoder.summary()["approx_vocab_state_mb"]),
            "approx_p2_model_state_mb": float(p2_model_approx_state_mb),
        }
        baseline_comparison = (
            _baseline_comparison(
                str(args.dataset),
                Path(args.result_root),
                evaluation_summary.get("event_metrics", {}),
                evaluation_summary.get("node_metrics", {}),
                runtime_summary,
                memory_summary,
            )
            if phase_name == "p2"
            else {}
        )
        summary = {
            "dataset": str(args.dataset),
            "method": method_name,
            "phase": phase_name,
            "purpose": purpose,
            "out_tag": out_tag,
            "counts": counts_meta,
            "limits": {
                "max_train_events": int(args.max_train_events),
                "max_ref_events": int(args.max_ref_events),
                "max_test_events": int(args.max_test_events),
                "fetch_size": int(args.fetch_size),
                "stream_check_max_events": int(args.stream_check_max_events),
                "progress_interval_events": int(args.progress_interval_events),
            },
            "score_config": {
                "score_mode": str(args.score_mode),
                "rcg_epochs": int(args.rcg_epochs),
                "rcg_learning_rate": float(args.rcg_learning_rate),
                "rcg_l2": float(args.rcg_l2),
                "rcg_synthetic_per_event": int(args.rcg_synthetic_per_event),
                "rcg_max_trace_events": int(args.rcg_max_trace_events),
                "rcg_edge_only_penalty": float(args.rcg_edge_only_penalty),
                "rcg_q_high": float(args.rcg_q_high),
                "rcg_uniform_prior_weight": float(args.rcg_uniform_prior_weight),
                "rcg_seed": int(args.rcg_seed),
            },
            "encoder": encoder.summary(),
            "stream_checks": {
                "train": train_check.summary(),
                "validation": val_check.summary(),
                "test": test_check.summary(),
            },
            "runtime": runtime_summary,
            "memory": memory_summary,
            "leakage_check": {
                "train_only_vocab": True,
                "train_only_model": bool(str(args.phase) == "p2"),
                "validation_updates_vocab": False,
                "test_updates_vocab": False,
                "ground_truth_used_before_scoring": False,
                "ground_truth_loaded_for_post_stream_evaluation": bool(str(args.phase) == "p2"),
                "test_labels_used_before_scoring": False,
                "full_test_statistics_used": False,
                "rcg_uses_test_labels": False,
                "rcg_uses_test_statistics": False,
                "notes": "Validation/test stream only after encoder.freeze(); P2 model tables are train-only; RCG, when enabled, is fit from train-only normality traces and train-only synthetic perturbations; validation only calibrates thresholds; online state updates after score_pre_update. Ground truth is loaded only after raw online alert CSVs are emitted.",
            },
            "p2_model": model_summary,
            "rcg_training": rcg_training_summary,
            "rcg_validation_trace_count": int(rcg_validation_trace_count),
            "p2_threshold": threshold,
            "p2_validation_score_count": int(len(val_scores)),
            "p2_online_event_alert_count": int(len(event_alerts)),
            "p2_online_node_alert_count": int(len(node_alerts)),
            "p2_test_events_scored": int(p2_test_seen),
            "p2_state_profile": p2_state_profile,
            "primary_online_metrics": {
                "immediate_candidate_alerts": {
                    "event_alerts": evaluation_summary.get("event_metrics", {}),
                    "node_alerts": evaluation_summary.get("node_metrics", {}),
                },
                "note": "LTPNM emits immediate event alerts online; node alerts are first endpoint alerts derived from the same online event threshold. Labels are attached only after online emission.",
            },
            "evaluation": evaluation_summary,
            "baseline_comparison": baseline_comparison,
            "deployment_budget": {
                "model_size_limit_mb": 10.0,
                "estimated_model_state_mb": float(p2_model_approx_state_mb),
                "peak_rss_target_mb": 1024.0,
                "peak_rss_mb": float(memory_summary["peak_rss_mb"]),
            },
            "outputs": {
                "model_summary_json": str(result_dir / "ltpnm_model_summary.json"),
                "eval_ltpnm_json": str(result_dir / "eval_ltpnm.json") if str(args.phase) == "p2" else "",
                "online_event_alerts_csv": str(result_dir / "online_event_alerts.csv") if str(args.phase) == "p2" else "",
                "online_event_alerts_raw_csv": str(result_dir / "online_event_alerts.raw.csv") if str(args.phase) == "p2" else "",
                "online_node_alerts_csv": str(result_dir / "online_node_alerts.csv") if str(args.phase) == "p2" else "",
                "online_node_alerts_raw_csv": str(result_dir / "online_node_alerts.raw.csv") if str(args.phase) == "p2" else "",
            },
        }
        out_json = result_dir / "ltpnm_model_summary.json"
        out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        if str(args.phase) == "p2":
            result_dir.joinpath("eval_ltpnm.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        log(f"[LTPNM] wrote {out_json}")
        if args.print_summary:
            print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        try:
            conn_cur.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
