"""Conditional group reports, coverage reports, RSS summaries, and backfill helpers."""

from __future__ import annotations

import pickle

from cs4m.semantics.optc_windows import is_optc_dataset
from scripts.pipeline.config.runtime_config import *


def _new_streaming_score_summary(threshold: float, bins: int):
    from scripts.pipeline.features.semantic_features import _StreamingScoreSummary

    return _StreamingScoreSummary(threshold, bins=bins)


class _TargetCaseStreamingSummary:
    """Streaming score summary split by conditional target case."""

    def __init__(self, bins: int = 20000) -> None:
        self._summaries: dict[str, Any] = {
            EVENT_SEMANTIC_TARGET: _new_streaming_score_summary(0.0, bins),
            BOTH_COLD_ACTION_TARGET: _new_streaming_score_summary(0.0, bins),
        }
        self._bins = int(bins)

    def observe(self, case_name: str, score: float) -> None:
        key = str(case_name)
        if key not in self._summaries:
            self._summaries[key] = _new_streaming_score_summary(0.0, self._bins)
        self._summaries[key].observe(float(score))

    def summary(self, thresholds: Mapping[str, float]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for case_name, summary in self._summaries.items():
            threshold = float(thresholds.get(case_name, 0.0))
            summary.threshold = threshold
            payload[case_name] = summary.summary()
            payload[case_name]["target_case"] = case_name
        return payload


class _ConditionalGroupStreamingSummary:
    """Collect compact test score and alert counts by conditional level-1 group."""

    def __init__(self, bins: int = 2000) -> None:
        self._summaries: dict[int, Any] = {}
        self.alert_counts: dict[int, int] = {}
        self._bins = int(bins)

    def observe(
        self,
        target_case_id_value: int,
        action_id: int,
        src_type_id: int,
        dst_type_id: int,
        score: float,
        *,
        alert: bool,
    ) -> None:
        key = int(
            encode_conditional_group_key(
                int(target_case_id_value),
                int(action_id),
                int(src_type_id),
                int(dst_type_id),
            ),
        )
        self._summaries.setdefault(key, _new_streaming_score_summary(0.0, self._bins)).observe(
            float(score),
        )
        if bool(alert):
            self.alert_counts[key] = int(self.alert_counts.get(key, 0)) + 1

    def rows(self, cache_meta: Mapping[str, Any]) -> list[dict[str, Any]]:
        group_meta = dict(cache_meta.get("group_thresholds") or {})
        level1 = dict(
            dict(group_meta.get("thresholds", {})).get(GROUP_LEVEL_TARGET_ACTION_SRC_DST, {}),
        )
        all_keys = set(self._summaries)
        for record in level1.values():
            key = _conditional_group_key_from_threshold_key(
                str(dict(record).get("threshold_group_key", "")),
            )
            if key is not None:
                all_keys.add(int(key))
        rows: list[dict[str, Any]] = []
        for encoded_key in sorted(all_keys):
            decoded = decode_conditional_group_key(int(encoded_key))
            target_case = TARGET_CASE_ID_TO_NAME.get(
                int(decoded["target_case_id"]),
                f"case_{int(decoded['target_case_id'])}",
            )
            action_id = int(decoded["action_id"])
            src_type_id = int(decoded["src_type_id"])
            dst_type_id = int(decoded["dst_type_id"])
            resolved = resolve_conditional_group_threshold(
                group_meta,
                int(decoded["target_case_id"]),
                action_id,
                src_type_id,
                dst_type_id,
            )
            threshold = float(resolved.get("threshold", cache_meta.get("threshold", 0.0)))
            validation_record = _conditional_validation_record_for_level1(
                group_meta,
                int(decoded["target_case_id"]),
                action_id,
                src_type_id,
                dst_type_id,
            )
            val_summary = dict(validation_record.get("summary", {}))
            summary = self._summaries.get(int(encoded_key))
            if summary is None:
                test_summary = conditional_score_summary(np.asarray([], dtype=np.float32), threshold)
                test_summary["score_quantile_method"] = "empty"
            else:
                summary.threshold = threshold
                test_summary = summary.summary()
            rows.append(
                {
                    "target_case": target_case,
                    "action_id": action_id,
                    "action_name": _conditional_action_name(action_id),
                    "src_type_id": src_type_id,
                    "src_type_name": _phase3e_entity_type_name("src_type_id", src_type_id),
                    "dst_type_id": dst_type_id,
                    "dst_type_name": _phase3e_entity_type_name("dst_type_id", dst_type_id),
                    "validation_count": int(validation_record.get("count", 0)),
                    "test_count": int(test_summary.get("count", 0)),
                    "threshold": threshold,
                    "threshold_level": str(resolved.get("threshold_level", "")),
                    "low_support_policy": str(resolved.get("low_support_policy", "")),
                    "group_validation_max": float(resolved.get("group_validation_max", 0.0)),
                    "parent_threshold": float(resolved.get("parent_threshold", 0.0)),
                    "global_threshold": float(resolved.get("global_threshold", 0.0)),
                    "final_threshold_source": str(
                        resolved.get("final_threshold_source", ""),
                    ),
                    "adaptive_margin_used": float(resolved.get("adaptive_margin_used", 0.0)),
                    "validation_count_bucket": str(resolved.get("validation_count_bucket", "")),
                    "val_p99": float(val_summary.get("score_p99", 0.0)),
                    "val_p999": float(val_summary.get("score_p999", 0.0)),
                    "val_p9995": float(val_summary.get("score_p9995", 0.0)),
                    "val_p9999": float(val_summary.get("score_p9999", 0.0)),
                    "val_max": float(val_summary.get("score_max", 0.0)),
                    "test_p99": float(test_summary.get("score_p99", 0.0)),
                    "test_p999": float(test_summary.get("score_p999", 0.0)),
                    "test_p9995": float(test_summary.get("score_p9995", 0.0)),
                    "test_p9999": float(test_summary.get("score_p9999", 0.0)),
                    "test_max": float(test_summary.get("score_max", 0.0)),
                    "score_quantile_method": str(
                        test_summary.get("score_quantile_method", "streaming_histogram"),
                    ),
                    "alert_count": int(self.alert_counts.get(int(encoded_key), 0)),
                    "tp": 0,
                    "fp": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                },
            )
        return rows


def _conditional_action_name(action_id: int) -> str:
    if 0 <= int(action_id) < len(ORTHRUS10_ACTION_NAMES):
        return str(ORTHRUS10_ACTION_NAMES[int(action_id)])
    return ""


def _conditional_group_key_from_threshold_key(value: str) -> int | None:
    parts: dict[str, str] = {}
    for item in str(value).split("|"):
        if "=" not in item:
            continue
        left, right = item.split("=", 1)
        parts[left] = right
    try:
        case_name = str(parts["target_case"])
        case_id = TARGET_CASE_NAME_TO_ID[case_name]
        return int(
            encode_conditional_group_key(
                int(case_id),
                int(parts["action_id"]),
                int(parts["src_type_id"]),
                int(parts["dst_type_id"]),
            ),
        )
    except (KeyError, ValueError):
        return None


def _conditional_validation_record_for_level1(
    group_meta: Mapping[str, Any],
    target_case_id_value: int,
    action_id: int,
    src_type_id: int,
    dst_type_id: int,
) -> dict[str, Any]:
    key = make_conditional_group_key(
        GROUP_LEVEL_TARGET_ACTION_SRC_DST,
        int(target_case_id_value),
        int(action_id),
        int(src_type_id),
        int(dst_type_id),
    )
    return dict(
        dict(dict(group_meta.get("thresholds", {})).get(GROUP_LEVEL_TARGET_ACTION_SRC_DST, {}))
        .get(key, {})
    )


def _write_conditional_group_test_summary(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path = Path(output_dir) / "conditional_score_summary_by_target_action_type.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONDITIONAL_GROUP_SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CONDITIONAL_GROUP_SUMMARY_FIELDS})
    return path


def _conditional_group_summary_with_eval(
    rows: Sequence[Mapping[str, Any]],
    evaluated_alert_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return conditional group summary rows with post-stream TP/FP fields populated."""
    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {
        (
            str(row.get("target_case", "")),
            str(row.get("action_name", "")),
            str(row.get("src_type_name", "")),
            str(row.get("dst_type_name", "")),
        ): dict(row)
        for row in rows
    }
    tp_by_key: dict[tuple[str, str, str, str], int] = {}
    fp_by_key: dict[tuple[str, str, str, str], int] = {}
    total_tp = 0
    for alert in evaluated_alert_rows:
        key = (
            str(alert.get("target_case", "")),
            str(alert.get("action", "")),
            str(alert.get("src_type", "")),
            str(alert.get("dst_type", "")),
        )
        correct = bool(_parse_bool(alert.get("is_correct_event_alert", False)))
        if correct:
            tp_by_key[key] = int(tp_by_key.get(key, 0)) + 1
            total_tp += 1
        else:
            fp_by_key[key] = int(fp_by_key.get(key, 0)) + 1
    for key, row in by_key.items():
        tp = int(tp_by_key.get(key, 0))
        fp = int(fp_by_key.get(key, 0))
        denom = tp + fp
        row["tp"] = tp
        row["fp"] = fp
        row["precision"] = float(tp / denom) if denom > 0 else 0.0
        row["recall"] = float(tp / max(total_tp, 1)) if total_tp > 0 else 0.0
    return [by_key[key] for key in sorted(by_key)]


def _phase3g_conditional_post_stream_group_eval(
    *,
    config: SlimConfig,
    paths: Mapping[str, Path],
    test_index: np.ndarray,
    stream_outputs: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Populate conditional group TP/FP after streaming without changing online alerts."""
    group_summary_csv = str(stream_outputs.get("conditional_group_summary_csv", "")).strip()
    if not group_summary_csv:
        return {}
    group_path = Path(group_summary_csv)
    alert_path = Path(output_dir) / "online_event_alerts.csv"
    if not group_path.exists() or not alert_path.exists():
        return {}
    abnormal_nodes = _phase3g_load_abnormal_db_nodes_for_eval(config)
    if not abnormal_nodes:
        return {
            "conditional_group_summary_csv": str(group_path),
            "event_alerts": _event_metrics_from_counts(0, 0),
            "event_labels_by_event": {},
        }
    idx_to_db_node_id = _phase3e_load_idx_to_node_id(paths)
    labels_by_event = _phase3e_event_labels_for_alerts(
        records=test_index,
        alert_event_indices=_alert_event_indices(stream_outputs),
        abnormal_db_node_ids=abnormal_nodes,
        idx_to_db_node_id=idx_to_db_node_id,
    )
    evaluated_alert_rows = [
        _evaluated_event_row(row, labels_by_event)
        for row in _iter_csv_rows(alert_path)
    ]
    group_rows = list(_iter_csv_rows(group_path))
    if group_rows:
        _write_conditional_group_test_summary(
            output_dir,
            _conditional_group_summary_with_eval(group_rows, evaluated_alert_rows),
        )
    tp = sum(1 for row in evaluated_alert_rows if bool(_parse_bool(row["is_correct_event_alert"])))
    fp = max(len(evaluated_alert_rows) - tp, 0)
    return {
        "event_alerts": _event_metrics_from_counts(tp, fp),
        "event_labels_by_event": labels_by_event,
        "conditional_group_summary_csv": str(group_path),
    }


def _phase3g_csv_int(row: Mapping[str, Any], key: str) -> int:
    try:
        return int(float(str(row.get(key, 0) or 0)))
    except (TypeError, ValueError):
        return 0


def _phase3g_write_required_dual_head_summary_csvs(
    *,
    output_dir: Path,
    stream_outputs: Mapping[str, Any],
    group_summary_csv: str,
) -> dict[str, str]:
    """Write required target-case and group alert summary CSVs for Phase3G reports."""
    output_path = Path(output_dir)
    target_case_path = output_path / "target_case_summary.csv"
    group_alert_path = output_path / "group_alert_summary.csv"
    score_by_case = dict(stream_outputs.get("test_score_summary_by_target_case", {}))
    grouped_by_case: dict[str, dict[str, int]] = {}
    group_rows_out: list[dict[str, Any]] = []
    group_path = Path(str(group_summary_csv)) if str(group_summary_csv).strip() else None
    if group_path is not None and group_path.exists():
        for row in _iter_csv_rows(group_path):
            case = str(row.get("target_case", ""))
            tp = _phase3g_csv_int(row, "tp")
            fp = _phase3g_csv_int(row, "fp")
            alert_count = _phase3g_csv_int(row, "alert_count")
            event_count = _phase3g_csv_int(row, "test_count")
            case_counts = grouped_by_case.setdefault(
                case,
                {"event_count": 0, "alert_count": 0, "TP": 0, "FP": 0},
            )
            case_counts["event_count"] += int(event_count)
            case_counts["alert_count"] += int(alert_count)
            case_counts["TP"] += int(tp)
            case_counts["FP"] += int(fp)
            precision = float(tp / max(tp + fp, 1)) if tp + fp > 0 else 0.0
            group_rows_out.append(
                {
                    "action": row.get("action_name", ""),
                    "src_type": row.get("src_type_name", ""),
                    "dst_type": row.get("dst_type_name", ""),
                    "target_case": case,
                    "event_count": event_count,
                    "alert_count": alert_count,
                    "TP": tp,
                    "FP": fp,
                    "precision": row.get("precision", precision),
                    "covered_malicious_nodes": row.get("covered_malicious_nodes", ""),
                },
            )
    case_names = sorted(set(score_by_case) | set(grouped_by_case))
    target_rows: list[dict[str, Any]] = []
    for case in case_names:
        score_summary = dict(score_by_case.get(case, {}))
        counts = grouped_by_case.get(case, {})
        event_count = counts.get("event_count", score_summary.get("count", 0))
        alert_count = int(counts.get("alert_count", 0))
        tp = int(counts.get("TP", 0))
        fp = int(counts.get("FP", 0))
        precision = float(tp / max(tp + fp, 1)) if tp + fp > 0 else 0.0
        target_rows.append(
            {
                "target_case": case,
                "event_count": event_count,
                "alert_count": alert_count,
                "TP": tp,
                "FP": fp,
                "precision": precision,
                "threshold_mean": score_summary.get(
                    "final_threshold",
                    score_summary.get("threshold", ""),
                ),
                "score_mean": score_summary.get("score_mean", ""),
                "score_p99": score_summary.get("score_p99", ""),
                "score_p999": score_summary.get("score_p999", ""),
            },
        )
    _write_csv(target_case_path, target_rows, TARGET_CASE_SUMMARY_FIELDS)
    def group_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
            str(row.get("target_case", "")),
        )

    _write_csv(
        group_alert_path,
        sorted(group_rows_out, key=group_sort_key),
        GROUP_ALERT_SUMMARY_FIELDS,
    )
    return {
        "target_case_summary_csv": str(target_case_path),
        "group_alert_summary_csv": str(group_alert_path),
    }


def _phase3g_group_eval_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
    compact_node_labels: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    tp = 0
    fp = 0
    covered: set[int] = set()
    coverage_tracker = OnlineNodeCoverageTracker()
    evaluated_rows: list[dict[str, Any]] = []
    for row in rows:
        label, src_bad, dst_bad = _phase3g_label_for_alert_row(
            row,
            idx_to_db_node_id,
            abnormal_db_node_ids,
            compact_node_labels=compact_node_labels,
        )
        event_id = int(row.get("event_index", -1))
        event_score = float(_safe_float(row.get("event_score"), 0.0) or 0.0)
        if _is_attack_label(label):
            tp += 1
        else:
            fp += 1
        if src_bad:
            src_idx = int(row.get("info_src") or row.get("src_idx") or -1)
            covered.add(src_idx if compact_node_labels else int(idx_to_db_node_id.get(src_idx, -1)))
        if dst_bad:
            dst_idx = int(row.get("info_dst") or row.get("dst_idx") or -1)
            covered.add(dst_idx if compact_node_labels else int(idx_to_db_node_id.get(dst_idx, -1)))
        coverage_tracker.observe(
            node_idx=int(row.get("info_src") or row.get("src_idx") or -1),
            node_type=str(row.get("src_type", "")),
            event_id=event_id,
            event_score=event_score,
            side="src",
        )
        coverage_tracker.observe(
            node_idx=int(row.get("info_dst") or row.get("dst_idx") or -1),
            node_type=str(row.get("dst_type", "")),
            event_id=event_id,
            event_score=event_score,
            side="dst",
        )
        evaluated_rows.append(_evaluated_event_row(row, {event_id: label}))
    support = _node_alert_support(evaluated_rows)
    strict_tp = 0
    strict_fp = 0
    relaxed_tp = 0
    relaxed_fp = 0
    for coverage_row in coverage_tracker.rows():
        node_idx = int(coverage_row.get("node_idx", -1))
        db_node_id = int(idx_to_db_node_id.get(node_idx, -1))
        node_label = (compact_node_labels or {}).get(
            node_idx,
            "malicious" if db_node_id in abnormal_db_node_ids else "benign",
        )
        if node_label == "malicious":
            strict_tp += 1
        else:
            strict_fp += 1
        relaxed = _relaxed_node_eval_result(node_label, support.get(node_idx, {}))
        if relaxed == "TP":
            relaxed_tp += 1
        elif relaxed == "FP":
            relaxed_fp += 1
    return {
        "TP": int(tp),
        "FP": int(fp),
        "precision": float(tp / max(tp + fp, 1)),
        "covered_malicious_nodes": int(len(covered)),
        "strict_node_TP": int(strict_tp),
        "strict_node_FP": int(strict_fp),
        "relaxed_node_TP": int(relaxed_tp),
        "relaxed_node_FP": int(relaxed_fp),
    }


def _phase3g_update_group_report_rows_with_eval(
    *,
    path: Path,
    alert_path: Path,
    config: SlimConfig,
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
    compact_node_labels: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    if compact_node_labels is None and is_optc_dataset(getattr(config, "dataset", "")):
        original_to_canonical = _phase3g_load_optc_original_to_canonical(config)
        if original_to_canonical:
            compact_node_labels, _ = _phase3g_compact_gt_labels_for_eval(
                abnormal_db_node_ids=abnormal_db_node_ids,
                idx_to_db_node_id=idx_to_db_node_id,
                original_to_canonical_netflow=original_to_canonical,
                node_id_to_idx=_phase3g_load_node_id_to_idx_for_eval(config),
            )
    if not path.exists():
        return []
    rows = list(_iter_csv_rows(path))
    if not rows:
        return []
    final_alerts = list(_iter_csv_rows(alert_path))
    alerts_by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for alert in final_alerts:
        key = (
            str(alert.get("action", "")),
            str(alert.get("src_type", "")),
            str(alert.get("dst_type", "")),
        )
        alerts_by_group.setdefault(key, []).append(alert)
    updated: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
        )
        metrics = _phase3g_group_eval_counts(
            alerts_by_group.get(key, []),
            idx_to_db_node_id=idx_to_db_node_id,
            abnormal_db_node_ids=abnormal_db_node_ids,
            compact_node_labels=compact_node_labels,
        )
        payload = dict(row)
        payload.update(metrics)
        updated.append(payload)
    fieldnames = ACTION_TYPE_POLICY_SUMMARY_FIELDS
    if path.name == "node_coverage_by_group.csv":
        fieldnames = NODE_COVERAGE_BY_GROUP_FIELDS
    elif path.name == "q_t_by_action_type.csv":
        fieldnames = Q_T_BY_ACTION_TYPE_FIELDS
    _write_csv(path, updated, fieldnames)
    return updated


def _phase3g_write_demoted_group_summary(config: SlimConfig, output_dir: Path) -> str:
    """Aggregate action/type policy demotions without changing streaming outputs."""
    node_evidence_path = output_dir / "action_type_node_evidence_events.csv"
    summary_path = output_dir / "demoted_group_summary.csv"
    if not node_evidence_path.exists():
        _write_csv(summary_path, [], DEMOTED_GROUP_SUMMARY_FIELDS)
        return str(summary_path)
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in _iter_csv_rows(node_evidence_path):
        key = (
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
            str(row.get("target_case", "")),
        )
        payload = grouped.setdefault(
            key,
            {
                "dataset": str(config.dataset),
                "run": _phase3g_run_only_from_out_tag(config.out_tag),
                "state_model": str(config.sspm_state_model),
                "policy_name": str(config.action_type_alert_policy),
                "action": key[0],
                "src_type": key[1],
                "dst_type": key[2],
                "target_case": key[3],
                "event_count": 0,
                "node_evidence_count": 0,
                "demoted_event_count": 0,
                "budget_capped_event_count": 0,
            },
        )
        payload["event_count"] = int(payload["event_count"]) + 1
        if _parse_bool(row.get("node_evidence", False)):
            payload["node_evidence_count"] = int(payload["node_evidence_count"]) + 1
        if str(row.get("alert_decision", "")) == "demoted_event":
            payload["demoted_event_count"] = int(payload["demoted_event_count"]) + 1
        if _parse_bool(row.get("budget_capped", False)):
            payload["budget_capped_event_count"] = (
                int(payload["budget_capped_event_count"]) + 1
            )
    rows = [grouped[key] for key in sorted(grouped)]
    _write_csv(summary_path, rows, DEMOTED_GROUP_SUMMARY_FIELDS)
    return str(summary_path)


def _phase3g_load_validation_group_scores(
    cache_meta: Mapping[str, Any],
) -> dict[tuple[int, int, int], list[float]]:
    score_value = str(cache_meta.get("validation_conditional_scores", "")).strip()
    group_value = str(cache_meta.get("validation_conditional_group_keys", "")).strip()
    if not score_value or not group_value:
        return {}
    count = int(cache_meta.get("count", 0) or 0)
    if count <= 0:
        return {}
    score_path = Path(score_value)
    group_path = Path(group_value)
    if not score_path.is_file() or not group_path.is_file():
        return {}
    scores = np.memmap(score_path, dtype=np.float32, mode="r", shape=(count,))
    group_keys = np.memmap(group_path, dtype=np.int64, mode="r", shape=(count,))
    grouped: dict[tuple[int, int, int], list[float]] = {}
    try:
        for score, encoded in zip(scores, group_keys):
            decoded = decode_conditional_group_key(int(encoded))
            key = (
                int(decoded["action_id"]),
                int(decoded["src_type_id"]),
                int(decoded["dst_type_id"]),
            )
            grouped.setdefault(key, []).append(float(score))
    finally:
        del scores
        del group_keys
    return grouped


def _phase3g_candidate_threshold_for_group(
    validation_group_scores: Mapping[tuple[int, int, int], Sequence[float]],
    key: tuple[int, int, int],
    quantile: float,
    margin: float,
    fallback_threshold: float,
) -> float:
    values = validation_group_scores.get(key)
    if values:
        return float(conditional_quantile_threshold(np.asarray(values, dtype=np.float32), quantile))
    return float(fallback_threshold)


def _phase3g_write_group_threshold_sweep_summary(
    *,
    config: SlimConfig,
    cache_meta: Mapping[str, Any],
    output_dir: Path,
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
) -> str:
    score_trace_path = output_dir / "online_event_score_trace.csv"
    if not score_trace_path.exists():
        _write_csv(
            output_dir / "group_threshold_sweep_summary.csv",
            [],
            GROUP_THRESHOLD_SWEEP_SUMMARY_FIELDS,
        )
        return str(output_dir / "group_threshold_sweep_summary.csv")
    targets = {
        ("EVENT_RECVFROM", "netflow", "process"),
        ("EVENT_CONNECT", "process", "netflow"),
        ("EVENT_READ", "file", "process"),
    }
    quantiles = (0.998, 0.999, 0.9995, 0.9999)
    margins = (-0.05, -0.02, 0.0, 0.02, 0.05)
    validation_group_scores = _phase3g_load_validation_group_scores(cache_meta)
    trace_by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in _iter_csv_rows(score_trace_path):
        group = (
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
        )
        if group in targets:
            trace_by_group.setdefault(group, []).append(row)
    rows: list[dict[str, Any]] = []
    for group in sorted(targets):
        action, src_type, dst_type = group
        try:
            key = (
                int(ORTHRUS10_ACTION_NAMES.index(action)),
                int(ENTITY_TYPES[src_type]),
                int(ENTITY_TYPES[dst_type]),
            )
        except (KeyError, ValueError):
            continue
        trace_rows = trace_by_group.get(group, [])
        fallback_threshold = float(cache_meta.get("threshold", 0.0))
        if trace_rows:
            fallback_threshold = float(_safe_float(trace_rows[0].get("threshold", 0.0), 0.0))
        for quantile in quantiles:
            for margin in margins:
                threshold = _phase3g_candidate_threshold_for_group(
                    validation_group_scores,
                    key,
                    quantile,
                    margin,
                    fallback_threshold,
                ) + float(margin)
                alert_rows = [
                    {
                        "event_index": int(trace.get("event_id", -1)),
                        "info_src": str(trace.get("info_src") or trace.get("src_idx") or -1),
                        "info_dst": str(trace.get("info_dst") or trace.get("dst_idx") or -1),
                        "src_idx": str(trace.get("src_idx") or trace.get("info_src") or -1),
                        "dst_idx": str(trace.get("dst_idx") or trace.get("info_dst") or -1),
                        "action": action,
                        "src_type": src_type,
                        "dst_type": dst_type,
                        "event_score": float(_safe_float(trace.get("score", 0.0), 0.0)),
                    }
                    for trace in trace_rows
                    if float(_safe_float(trace.get("score", 0.0), 0.0)) >= threshold
                ]
                metrics = _phase3g_group_eval_counts(
                    alert_rows,
                    idx_to_db_node_id=idx_to_db_node_id,
                    abnormal_db_node_ids=abnormal_db_node_ids,
                )
                rows.append(
                    {
                        **_phase3g_empty_report_metrics(config),
                        "policy_name": "per_action_type_quantile",
                        "action": action,
                        "src_type": src_type,
                        "dst_type": dst_type,
                        "threshold": threshold,
                        "event_count": int(len(trace_rows)),
                        "alert_count": int(len(alert_rows)),
                        **metrics,
                        "quantile": float(quantile),
                        "margin": float(margin),
                        "threshold_policy": "per_action_type_quantile",
                    },
                )
    path = output_dir / "group_threshold_sweep_summary.csv"
    _write_csv(path, rows, GROUP_THRESHOLD_SWEEP_SUMMARY_FIELDS)
    return str(path)


def _phase3g_endpoint_suppression_post_eval(
    *,
    config: SlimConfig,
    paths: Mapping[str, Path],
    test_index: np.ndarray,
    stream_outputs: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Backfill endpoint suppression TP/FP diagnostics after streaming."""
    post_eval_start_rss = _current_rss_mb()
    summary_path_text = str(stream_outputs.get("conditional_endpoint_suppression_summary_csv", ""))
    suppressed_csv_text = str(stream_outputs.get("suppressed_event_alerts_raw_csv", ""))
    if not summary_path_text and not suppressed_csv_text:
        return {
            "post_stream_eval_rss_start_mb": float(post_eval_start_rss),
            "post_stream_eval_rss_final_mb": float(_current_rss_mb()),
            "post_stream_eval_rss_delta_mb": 0.0,
        }
    summary_path = Path(summary_path_text)
    has_summary = bool(summary_path_text) and summary_path.exists()
    abnormal_nodes = _phase3g_load_abnormal_db_nodes_for_eval(config)
    if not abnormal_nodes:
        return {
            "summary_csv": str(summary_path) if has_summary else "",
            "post_stream_eval_rss_start_mb": float(post_eval_start_rss),
            "post_stream_eval_rss_final_mb": float(_current_rss_mb()),
            "post_stream_eval_rss_delta_mb": float(_current_rss_mb() - post_eval_start_rss),
        }
    idx_to_db_node_id = _phase3e_load_idx_to_node_id(paths)
    rows = list(_iter_csv_rows(summary_path)) if has_summary else []
    rows_by_pair: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = dict(row)
        key = str(payload.get("pair_signature", payload.get("pair_key", "")))
        if key in rows_by_pair:
            rows_by_pair[key]["test_pair_count"] = int(
                float(rows_by_pair[key].get("test_pair_count", 0) or 0),
            ) + int(float(payload.get("test_pair_count", 0) or 0))
            rows_by_pair[key]["raw_alert_count"] = int(
                float(rows_by_pair[key].get("raw_alert_count", 0) or 0),
            ) + int(float(payload.get("raw_alert_count", 0) or 0))
            rows_by_pair[key]["suppressed_alert_count"] = int(
                float(rows_by_pair[key].get("suppressed_alert_count", 0) or 0),
            ) + int(float(payload.get("suppressed_alert_count", 0) or 0))
            rows_by_pair[key]["final_alert_count"] = int(
                float(rows_by_pair[key].get("final_alert_count", 0) or 0),
            ) + int(float(payload.get("final_alert_count", 0) or 0))
            rows_by_pair[key]["max_score"] = max(
                float(rows_by_pair[key].get("max_score", 0.0) or 0.0),
                float(payload.get("max_score", 0.0) or 0.0),
            )
            continue
        rows_by_pair[key] = payload

    def update_pair_counts(alert_rows: Sequence[Mapping[str, Any]], suffix: str) -> tuple[int, int]:
        tp = 0
        fp = 0
        for alert in alert_rows:
            pair_key = str(
                alert.get("endpoint_pair_key", "")
                or _conditional_endpoint_signature_text_from_row(alert)
            )
            row = rows_by_pair.get(pair_key)
            if row is None:
                continue
            correct = bool(_parse_bool(alert.get("is_correct_event_alert", False)))
            if correct:
                row[f"tp_{suffix}"] = int(float(row.get(f"tp_{suffix}", 0) or 0)) + 1
                tp += 1
            else:
                row[f"fp_{suffix}"] = int(float(row.get(f"fp_{suffix}", 0) or 0)) + 1
                fp += 1
        return tp, fp

    final_alert_rows = list(_iter_csv_rows(Path(output_dir) / "online_event_alerts.csv"))
    if suppressed_csv_text and Path(suppressed_csv_text).exists():
        suppressed_raw_rows = list(_iter_csv_rows(Path(suppressed_csv_text)))
    else:
        suppressed_raw_rows = [
            dict(row)
            for row in stream_outputs.get("suppressed_event_alerts_raw", [])
            if isinstance(row, Mapping)
        ]
    alert_event_indices = {int(row["event_index"]) for row in final_alert_rows}
    alert_event_indices.update(int(row["event_index"]) for row in suppressed_raw_rows)
    labels_by_event = _phase3e_event_labels_for_alerts(
        records=test_index,
        alert_event_indices=alert_event_indices,
        abnormal_db_node_ids=abnormal_nodes,
        idx_to_db_node_id=idx_to_db_node_id,
    )
    evaluated_final_rows = [
        _evaluated_event_row(row, labels_by_event)
        for row in final_alert_rows
    ]
    evaluated_suppressed_rows = [
        _evaluated_event_row(row, labels_by_event)
        for row in suppressed_raw_rows
    ]
    suppressed_tp = 0
    suppressed_fp = 0
    final_tp, final_fp = update_pair_counts(evaluated_final_rows, "after")
    before_rows = [*evaluated_final_rows, *evaluated_suppressed_rows]
    before_tp, before_fp = update_pair_counts(before_rows, "before")
    for row in evaluated_suppressed_rows:
        if bool(_parse_bool(row.get("is_correct_event_alert", False))):
            suppressed_tp += 1
        else:
            suppressed_fp += 1
    if has_summary and rows_by_pair:
        rows = [rows_by_pair[key] for key in sorted(rows_by_pair)]
        fieldnames = list(rows[0].keys())
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    suppressed_tp_path = ""
    suppressed_tp_rows = [
        row
        for row in evaluated_suppressed_rows
        if bool(_parse_bool(row.get("is_correct_event_alert", False)))
    ]
    if suppressed_tp_rows:
        suppressed_tp_path = str(
            _write_conditional_endpoint_suppressed_tp_events(output_dir, suppressed_tp_rows),
        )
        _write_conditional_endpoint_suppressed_tp_audit(output_dir, suppressed_tp_rows)
    post_eval_final_rss = _current_rss_mb()
    return {
        "summary_csv": str(summary_path) if has_summary else "",
        "suppressed_raw_csv": suppressed_csv_text,
        "tp_before": int(before_tp),
        "fp_before": int(before_fp),
        "tp_after": int(final_tp),
        "fp_after": int(final_fp),
        "suppressed_tp": int(suppressed_tp),
        "suppressed_fp": int(suppressed_fp),
        "suppressed_alert_count": int(suppressed_tp + suppressed_fp),
        "suppressed_tp_events_csv": suppressed_tp_path,
        "pair_label_scope": "post_stream_event_level",
        "post_stream_eval_rss_start_mb": float(post_eval_start_rss),
        "post_stream_eval_rss_final_mb": float(post_eval_final_rss),
        "post_stream_eval_rss_delta_mb": float(post_eval_final_rss - post_eval_start_rss),
    }


def _phase3g_label_for_alert_row(
    row: Mapping[str, Any],
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
    *,
    compact_node_labels: Mapping[int, str] | None = None,
) -> tuple[int, bool, bool]:
    src_idx = int(row.get("info_src") or row.get("src_idx") or -1)
    dst_idx = int(row.get("info_dst") or row.get("dst_idx") or -1)
    src_db = int(idx_to_db_node_id.get(src_idx, -1))
    dst_db = int(idx_to_db_node_id.get(dst_idx, -1))
    compact_labels = compact_node_labels or {}
    src_bad = compact_labels.get(src_idx) == "malicious" or src_db in abnormal_db_node_ids
    dst_bad = compact_labels.get(dst_idx) == "malicious" or dst_db in abnormal_db_node_ids
    label = 2 if src_bad and dst_bad else (1 if src_bad or dst_bad else 0)
    return label, src_bad, dst_bad


def _phase3g_event_node_coverage_from_alerts(
    alert_path: Path,
) -> list[dict[str, Any]]:
    tracker = OnlineNodeCoverageTracker()
    for row in _iter_csv_rows(alert_path):
        event_id = int(row.get("event_index", -1))
        event_score = float(_safe_float(row.get("event_score"), 0.0) or 0.0)
        tracker.observe(
            node_idx=int(row.get("info_src") or row.get("src_idx") or -1),
            node_type=str(row.get("src_type", "")),
            event_id=event_id,
            event_score=event_score,
            side="src",
        )
        tracker.observe(
            node_idx=int(row.get("info_dst") or row.get("dst_idx") or -1),
            node_type=str(row.get("dst_type", "")),
            event_id=event_id,
            event_score=event_score,
            side="dst",
        )
    return tracker.rows()


def _phase3g_compact_gt_labels_for_eval(
    *,
    abnormal_db_node_ids: set[int],
    idx_to_db_node_id: Mapping[int, int],
    original_to_canonical_netflow: Mapping[int, int] | None = None,
    node_id_to_idx: Mapping[int, int] | None = None,
) -> tuple[dict[int, str], dict[str, Any]]:
    """Return compact node labels for post-stream canonical-aware GT evaluation."""
    original_to_canonical = {
        int(node_id): int(canonical_id)
        for node_id, canonical_id in dict(original_to_canonical_netflow or {}).items()
    }
    if node_id_to_idx is None:
        node_id_to_idx = {int(node_id): int(idx) for idx, node_id in idx_to_db_node_id.items()}
    else:
        node_id_to_idx = {int(node_id): int(idx) for node_id, idx in dict(node_id_to_idx).items()}

    labels: dict[int, str] = {}
    mapped_originals = 0
    canonical_netflow_originals = 0
    canonical_netflow_compact: set[int] = set()
    for original_node_id in sorted(int(node_id) for node_id in abnormal_db_node_ids):
        lookup_node_id = int(original_to_canonical.get(original_node_id, original_node_id))
        compact_idx = node_id_to_idx.get(lookup_node_id)
        if compact_idx is None:
            continue
        labels[int(compact_idx)] = "malicious"
        mapped_originals += 1
        if original_node_id in original_to_canonical:
            canonical_netflow_originals += 1
            canonical_netflow_compact.add(int(compact_idx))

    summary = {
        "original_gt_total": int(len(abnormal_db_node_ids)),
        "original_gt_compact_mapped": int(mapped_originals),
        "original_gt_compact_missing": int(len(abnormal_db_node_ids) - mapped_originals),
        "unique_compact_gt_total": int(len(labels)),
        "canonical_netflow_original_gt_count": int(canonical_netflow_originals),
        "canonical_netflow_unique_compact_count": int(len(canonical_netflow_compact)),
    }
    return labels, summary


def _phase3g_load_optc_original_to_canonical(config: SlimConfig) -> dict[int, int]:
    """Load OpTC original netflow node id to canonical node id mapping when available."""
    if not is_optc_dataset(getattr(config, "dataset", "")):
        return {}
    try:
        paths = _phase3e_require_artifacts(config)
    except Exception:
        return {}
    sidecar = Path(paths.get("event_meta", Path())).with_name("original_to_canonical_netflow.csv")
    if not sidecar.exists():
        return {}
    mapping: dict[int, int] = {}
    with sidecar.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                mapping[int(row["original_node_id"])] = int(row["canonical_node_id"])
            except (KeyError, TypeError, ValueError):
                continue
    return mapping


def _phase3g_load_node_id_to_idx_for_eval(config: SlimConfig) -> dict[int, int]:
    """Load Phase3E node id to compact index mapping for post-stream evaluation."""
    paths = _phase3e_require_artifacts(config)
    raw_path = paths.get("node_id_to_idx")
    path = Path(raw_path) if raw_path is not None else Path()
    if raw_path is None or not path.exists():
        path = Path(paths["node_embeddings"]).with_name("node_id_to_idx.pkl")
    with path.open("rb") as handle:
        loaded = pickle.load(handle)
    return {int(node_id): int(idx) for node_id, idx in dict(loaded).items()}


def _phase3g_write_event_node_coverage_outputs(
    *,
    output_dir: Path,
    coverage_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
    compact_node_labels: Mapping[int, str] | None = None,
    compact_gt_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    raw_path = output_dir / "online_event_node_coverage.csv"
    strict_path = output_dir / "online_event_node_coverage_strict.csv"
    relaxed_path = output_dir / "online_event_node_coverage_relaxed.csv"
    summary_path = output_dir / "online_event_node_coverage_summary.json"

    support = _node_alert_support(event_rows)
    strict_rows: list[dict[str, Any]] = []
    relaxed_rows: list[dict[str, Any]] = []
    covered_malicious: set[int] = set()
    compact_labels = dict(compact_node_labels or {})
    for raw_row in coverage_rows:
        node_idx = int(raw_row.get("node_idx", -1))
        db_node_id = int(idx_to_db_node_id.get(node_idx, -1))
        node_label = compact_labels.get(
            node_idx,
            "malicious" if db_node_id in abnormal_db_node_ids else "benign",
        )
        if node_label == "malicious":
            covered_malicious.add(node_idx if compact_labels else db_node_id)
        base = dict(raw_row)
        base["node_label"] = node_label
        strict_rows.append(
            {
                **base,
                "eval_result": "TP" if node_label == "malicious" else "FP",
            },
        )
        relaxed_result = _relaxed_node_eval_result(node_label, support.get(node_idx, {}))
        relaxed_rows.append({**base, "eval_result": relaxed_result})

    _write_csv(raw_path, list(coverage_rows), EVENT_NODE_COVERAGE_FIELDS)
    _write_csv(strict_path, strict_rows, EVENT_NODE_COVERAGE_EVAL_FIELDS)
    _write_csv(relaxed_path, relaxed_rows, EVENT_NODE_COVERAGE_EVAL_FIELDS)
    legacy_strict_rows = [
        {
            "node_id": int(row.get("node_idx", -1)),
            "node_type": row.get("node_type", ""),
            "node_label": row.get("node_label", ""),
            "first_alert_event_idx": row.get("first_alert_event_id", ""),
            "alert_count": row.get("alert_count", 0),
            "max_event_score": row.get("max_event_score", 0.0),
            "pool_type": "strict",
            "eval_result": row.get("eval_result", ""),
            "supporting_event_count": row.get("alert_count", 0),
        }
        for row in strict_rows
    ]
    legacy_relaxed_rows = [
        {
            "node_id": int(row.get("node_idx", -1)),
            "node_type": row.get("node_type", ""),
            "node_label": row.get("node_label", ""),
            "first_alert_event_idx": row.get("first_alert_event_id", ""),
            "alert_count": row.get("alert_count", 0),
            "max_event_score": row.get("max_event_score", 0.0),
            "pool_type": "relaxed",
            "eval_result": row.get("eval_result", ""),
            "supporting_event_count": row.get("alert_count", 0),
        }
        for row in relaxed_rows
    ]
    _write_csv(output_dir / "online_node_alerts_strict.csv", legacy_strict_rows, NODE_POOL_EVAL_FIELDS)
    _write_csv(
        output_dir / "online_node_alerts_relaxed.csv",
        legacy_relaxed_rows,
        NODE_POOL_EVAL_FIELDS,
    )

    strict_counts = _eval_result_counts(strict_rows)
    relaxed_counts = _eval_result_counts(relaxed_rows)
    total_malicious = int(len(abnormal_db_node_ids))
    if compact_gt_summary is not None:
        total_malicious = int(compact_gt_summary.get("unique_compact_gt_total", total_malicious))
    summary = {
        "online_event_node_coverage_csv": str(raw_path),
        "online_event_node_coverage_strict_csv": str(strict_path),
        "online_event_node_coverage_relaxed_csv": str(relaxed_path),
        "coverage_node_count": int(len(coverage_rows)),
        "covered_malicious_nodes": int(len(covered_malicious)),
        "total_malicious_nodes": total_malicious,
        "malicious_node_recall": float(len(covered_malicious) / total_malicious)
        if total_malicious
        else 0.0,
        "strict_node_tp": int(strict_counts["tp"]),
        "strict_node_fp": int(strict_counts["fp"]),
        "strict_node_precision": float(
            strict_counts["tp"] / max(strict_counts["tp"] + strict_counts["fp"], 1),
        ),
        "strict_node_recall": float(strict_counts["tp"] / max(total_malicious, 1)),
        "relaxed_node_tp": int(relaxed_counts["tp"]),
        "relaxed_node_fp": int(relaxed_counts["fp"]),
        "relaxed_node_ignore": int(relaxed_counts["ignore"]),
        "relaxed_node_precision": float(
            relaxed_counts["tp"] / max(relaxed_counts["tp"] + relaxed_counts["fp"], 1),
        ),
        "relaxed_node_recall": float(relaxed_counts["tp"] / max(total_malicious, 1)),
    }
    if compact_gt_summary is not None:
        summary["canonical_gt_eval"] = dict(compact_gt_summary)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def _phase3g_write_node_pool_rebuilt_outputs(
    *,
    output_dir: Path,
    coverage_rows: Sequence[Mapping[str, Any]],
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
    topk_values: Sequence[int],
    node_pool_score_mode: str = "base_conf",
    event_rows: Sequence[Mapping[str, Any]] | None = None,
    compact_node_labels: Mapping[int, str] | None = None,
    compact_gt_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    support_scores = _phase3g_v31_support_scores(event_rows or [])
    final_nodes_raw = [
        {
            "node_id": int(row.get("node_idx", -1)),
            "node_score": float(
                support_scores.get(int(row.get("node_idx", -1)), row.get("max_event_score", 0.0))
                if str(node_pool_score_mode) == "base_conf_v31_support"
                else row.get("max_event_score", 0.0),
            ),
            "candidate_mass": float(row.get("max_event_score", 0.0)),
            "residual_mass": float(row.get("max_event_score", 0.0)),
            "residual_max": float(row.get("max_event_score", 0.0)),
            "residual_mean": float(row.get("max_event_score", 0.0)),
            "association_support": 0,
            "adaptive_memory_deviation": 0.0,
            "compact_chain_support": 0.0,
            "chain_diversity_pool": 1,
            "repeated_consistency": int(row.get("alert_count", 0)),
            "candidate_event_count": int(row.get("alert_count", 0)),
        }
        for row in coverage_rows
    ]
    final_nodes_raw.sort(key=_node_pool_sort_key)
    if compact_node_labels is not None:
        node_labels = {
            int(node_idx): str(label)
            for node_idx, label in compact_node_labels.items()
            if str(label) == "malicious"
        }
    else:
        node_labels = {
            int(row.get("node_idx", -1)): "malicious"
            for row in coverage_rows
            if int(idx_to_db_node_id.get(int(row.get("node_idx", -1)), -1))
            in abnormal_db_node_ids
        }
    _write_csv(output_dir / "final_node_pool_alerts.csv", _attach_node_labels(final_nodes_raw, node_labels), NODE_EVAL_FIELDS)
    _write_csv(output_dir / "online_node_alerts.csv", _attach_node_labels(final_nodes_raw, node_labels), NODE_EVAL_FIELDS)
    topk_metrics = _node_pool_topk_metrics(final_nodes_raw, node_labels, topk_values)
    topk_csv = output_dir / "node_topk_metrics.csv"
    topk_fields = ["k", "alert_count_at_k", "count", "tp", "fp", "precision", "ratio"]
    with topk_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=topk_fields)
        writer.writeheader()
        for key in sorted(topk_metrics, key=lambda item: int(item.replace("top", ""))):
            writer.writerow({field: topk_metrics[key].get(field, "") for field in topk_fields})
    topk_json = output_dir / "node_topk_metrics.json"
    topk_json.write_text(
        json.dumps(topk_metrics, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    summary = {
        "node_pool_rebuilt_from": "online_event_node_coverage",
        "node_pool_score_mode": str(node_pool_score_mode),
        "node_pool_count": int(len(final_nodes_raw)),
        "node_topk_metrics_csv": str(topk_csv),
        "node_topk_metrics_json": str(topk_json),
        "node_pool_topk": topk_metrics,
    }
    if compact_gt_summary is not None:
        summary["canonical_gt_eval"] = dict(compact_gt_summary)
    summary_path = output_dir / "node_pool_rebuilt_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def _phase3g_v31_support_scores(
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[int, float]:
    """Build runtime-only support scores for rebuilt ClearScope v31 node pools."""
    states: dict[int, dict[str, Any]] = {}
    for row in event_rows:
        event_score = float(_safe_float(row.get("event_score"), 0.0) or 0.0)
        residual_score = float(_safe_float(row.get("residual_score"), event_score) or event_score)
        action = str(row.get("action", ""))
        src_type = str(row.get("src_type", ""))
        dst_type = str(row.get("dst_type", ""))
        info_src = str(row.get("info_src", ""))
        info_dst = str(row.get("info_dst", ""))
        cache_like = any(
            token in f"{info_src} {info_dst}".lower()
            for token in (
                "cache",
                "cache2",
                "body",
                "app_webview",
                "shared_files",
                "databases",
                "shared_prefs",
            )
        )
        for key, role_type in (("info_src", src_type), ("info_dst", dst_type)):
            try:
                node_id = int(row.get(key, -1))
            except (TypeError, ValueError):
                node_id = -1
            if node_id < 0:
                continue
            state = states.setdefault(
                node_id,
                {
                    "event_score_max": 0.0,
                    "residual_max": 0.0,
                    "alert_event_count": 0,
                    "file_cache_read": False,
                    "file_cache_write": False,
                    "process_file_read": False,
                    "process_file_write": False,
                    "process_process_count": 0,
                    "netflow_count": 0,
                    "other_count": 0,
                },
            )
            state["event_score_max"] = max(float(state["event_score_max"]), event_score)
            state["residual_max"] = max(float(state["residual_max"]), residual_score)
            state["alert_event_count"] = int(state["alert_event_count"]) + 1
            if role_type == "netflow":
                state["netflow_count"] = int(state["netflow_count"]) + 1
            elif src_type == "process" and dst_type == "process":
                state["process_process_count"] = int(state["process_process_count"]) + 1
            else:
                state["other_count"] = int(state["other_count"]) + 1
            if src_type == "file" and dst_type == "process" and action in {
                "EVENT_READ",
                "EVENT_RECVFROM",
            }:
                state["process_file_read"] = True
                if cache_like:
                    state["file_cache_read"] = True
            if src_type == "process" and dst_type == "file" and action == "EVENT_WRITE":
                state["process_file_write"] = True
                if cache_like:
                    state["file_cache_write"] = True

    scores: dict[int, float] = {}
    for node_id, state in states.items():
        alert_count = int(state["alert_event_count"])
        score = 0.45 * max(float(state["residual_max"]), 0.0)
        score += 0.25 * math.log1p(max(alert_count, 0))
        cache_read = bool(state["file_cache_read"])
        cache_write = bool(state["file_cache_write"])
        if cache_read and cache_write:
            score += 0.20
        elif cache_read or cache_write:
            score += 0.08
        if bool(state["process_file_read"]) and bool(state["process_file_write"]):
            score += 0.10
        event_count = max(alert_count, 1)
        if int(state["process_process_count"]) == event_count:
            score -= 0.25
        if int(state["netflow_count"]) == event_count:
            score -= 0.10
        scores[int(node_id)] = float(score)
    return scores


def _phase3g_write_fp_group_outputs(
    *,
    output_dir: Path,
    event_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    output_dir = Path(output_dir)
    event_groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in event_rows:
        key = (
            str(row.get("action", "")),
            str(row.get("src_type", "")),
            str(row.get("dst_type", "")),
            _label_name(row.get("event_label", "benign")),
        )
        group = event_groups.setdefault(
            key,
            {"scores": [], "tp": 0, "fp": 0, "event_count": 0},
        )
        score = float(_safe_float(row.get("event_score"), 0.0) or 0.0)
        group["scores"].append(score)
        group["event_count"] = int(group["event_count"]) + 1
        if bool(_parse_bool(row.get("is_correct_event_alert", False))):
            group["tp"] = int(group["tp"]) + 1
        else:
            group["fp"] = int(group["fp"]) + 1
    event_rows_out = []
    for key, group in event_groups.items():
        scores = list(group["scores"])
        tp = int(group["tp"])
        fp = int(group["fp"])
        event_count = int(group["event_count"])
        event_rows_out.append(
            {
                "action": key[0],
                "src_type": key[1],
                "dst_type": key[2],
                "event_label": key[3],
                "event_count": event_count,
                "tp": tp,
                "fp": fp,
                "precision": float(tp / max(tp + fp, 1)),
                "score_min": min(scores) if scores else 0.0,
                "score_mean": float(sum(scores) / len(scores)) if scores else 0.0,
                "score_max": max(scores) if scores else 0.0,
            },
        )
    event_rows_out.sort(key=lambda row: (-int(row["fp"]), -int(row["tp"]), str(row["action"])))
    event_fp_path = output_dir / "event_fp_group_summary.csv"
    _write_csv(event_fp_path, event_rows_out, EVENT_FP_GROUP_FIELDS)

    node_rows_out = []
    for row in coverage_rows:
        node_rows_out.append(
            {
                "node_type": str(row.get("node_type", "")),
                "dominant_action": "",
                "dominant_peer_type": "",
                "node_count": 1,
                "strict_tp": 1 if str(row.get("node_label", "")) == "malicious" else 0,
                "strict_fp": 0 if str(row.get("node_label", "")) == "malicious" else 1,
                "total_event_support": int(row.get("alert_count", 0)),
                "max_event_score": float(row.get("max_event_score", 0.0)),
            },
        )
    by_node_type: dict[str, dict[str, Any]] = {}
    for row in node_rows_out:
        group = by_node_type.setdefault(
            str(row["node_type"]),
            {
                "node_type": str(row["node_type"]),
                "dominant_action": "",
                "dominant_peer_type": "",
                "node_count": 0,
                "strict_tp": 0,
                "strict_fp": 0,
                "total_event_support": 0,
                "max_event_score": 0.0,
            },
        )
        group["node_count"] = int(group["node_count"]) + 1
        group["strict_tp"] = int(group["strict_tp"]) + int(row["strict_tp"])
        group["strict_fp"] = int(group["strict_fp"]) + int(row["strict_fp"])
        group["total_event_support"] = int(group["total_event_support"]) + int(
            row["total_event_support"],
        )
        group["max_event_score"] = max(
            float(group["max_event_score"]),
            float(row["max_event_score"]),
        )
    node_fp_path = output_dir / "node_fp_group_summary.csv"
    _write_csv(node_fp_path, list(by_node_type.values()), NODE_FP_GROUP_FIELDS)
    return {
        "event_fp_group_summary_csv": str(event_fp_path),
        "node_fp_group_summary_csv": str(node_fp_path),
    }


def _phase3g_array_nbytes_mb(array: Any) -> float:
    if isinstance(array, LazyMmapLruEmbeddingLookup):
        return float(Path(array.path).stat().st_size) / 1024.0 / 1024.0
    nbytes = getattr(array, "nbytes", 0)
    try:
        return float(nbytes) / 1024.0 / 1024.0
    except (TypeError, ValueError):
        return 0.0


def _phase3g_file_size_mb(path_value: Any) -> float:
    text = str(path_value or "").strip()
    if not text:
        return 0.0
    path = Path(text)
    if not path.exists():
        return 0.0
    return float(path.stat().st_size) / 1024.0 / 1024.0


def _phase3g_mapped_file_rss_mb(path_value: Any) -> float | None:
    text = str(path_value or "").strip()
    if not text:
        return None
    try:
        target = str(Path(text).resolve())
    except OSError:
        return None
    smaps_path = Path("/proc/self/smaps")
    if not smaps_path.exists():
        return None
    total_kb = 0.0
    current_path = ""
    try:
        with smaps_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n").split(maxsplit=5)
                if parts and "-" in parts[0] and len(parts[0].split("-", 1)[0]) >= 4:
                    current_path = parts[5] if len(parts) >= 6 else ""
                    if current_path.endswith(" (deleted)"):
                        current_path = current_path[: -len(" (deleted)")]
                    continue
                if current_path != target or not line.startswith("Rss:"):
                    continue
                value_parts = line.split()
                if len(value_parts) >= 2:
                    total_kb += float(value_parts[1])
    except (OSError, ValueError):
        return None
    return float(total_kb) / 1024.0


def _phase3g_malloc_trim() -> None:
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (AttributeError, OSError):
        return


def _phase3g_smaps_timeline_row(stage: str) -> dict[str, Any]:
    smaps = _current_smaps_rollup_mb()
    return {
        "stage": str(stage),
        "rss_mb": _current_rss_mb(),
        "anonymous_rss_mb": smaps.get("anonymous_rss_mb"),
        "file_backed_rss_mb": smaps.get("file_backed_rss_mb"),
        "shared_clean_mb": smaps.get("shared_clean_mb"),
        "private_clean_mb": smaps.get("private_clean_mb"),
        "private_dirty_mb": smaps.get("private_dirty_mb"),
    }


def _phase3g_write_rss_timeline(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    path = Path(output_dir) / "rss_timeline.csv"
    fields = [
        "stage",
        "rss_mb",
        "anonymous_rss_mb",
        "file_backed_rss_mb",
        "shared_clean_mb",
        "private_clean_mb",
        "private_dirty_mb",
    ]
    _write_csv(path, list(rows), fields)
    return str(path)


def _phase3g_write_online_event_core_rss_breakdown(
    *,
    output_dir: Path,
    eval_payload: Mapping[str, Any],
    paths: Mapping[str, Path],
    node_embeddings: Any,
    endpoint_cache_mb: float,
    endpoint_cache_meta: Mapping[str, Any] | None = None,
    rss_timeline: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    memory = dict(eval_payload.get("memory", {})) if isinstance(eval_payload, Mapping) else {}
    smaps = _current_smaps_rollup_mb()
    event_index_test_mb = _phase3g_file_size_mb(paths["event_index_test"])
    event_index_validation_mb = _phase3g_file_size_mb(paths["event_index_validation"])
    embedding_total_mb = _phase3g_array_nbytes_mb(node_embeddings)
    embedding_breakdown = _phase3g_embedding_memory_breakdown_for_paths(
        paths=paths,
        node_embeddings=node_embeddings,
        state_table_mb=float(
            memory.get("online_minimal_state_array_mb")
            or memory.get("state_array_mb")
            or 0.0,
        ),
        endpoint_cache_mb=float(endpoint_cache_mb),
    )
    file_backed_peak = float(
        memory.get("online_minimal_file_backed_rss_mb")
        or smaps.get("file_backed_rss_mb")
        or 0.0,
    )
    mapped_test_mb = _phase3g_mapped_file_rss_mb(paths.get("event_index_test"))
    mapped_validation_mb = _phase3g_mapped_file_rss_mb(paths.get("event_index_validation"))
    mapped_embedding_mb = _phase3g_mapped_file_rss_mb(paths.get("node_embeddings"))
    excluded_test_file_backed_mb = (
        float(mapped_test_mb)
        if mapped_test_mb is not None and mapped_test_mb > 0.0
        else min(event_index_test_mb, file_backed_peak)
    )
    endpoint_meta = dict(endpoint_cache_meta or {})
    endpoint_pair_keys_mb = _phase3g_file_size_mb(endpoint_meta.get("endpoint_pair_keys_path"))
    endpoint_pair_counts_mb = _phase3g_file_size_mb(endpoint_meta.get("endpoint_pair_counts_path"))
    endpoint_action_keys_mb = _phase3g_file_size_mb(endpoint_meta.get("endpoint_action_keys_path"))
    endpoint_action_counts_mb = _phase3g_file_size_mb(
        endpoint_meta.get("endpoint_action_counts_path"),
    )
    state_table_mb = float(
        memory.get("online_minimal_state_array_mb")
        or memory.get("state_array_mb")
        or 0.0,
    )
    embedding_lookup_mode = str(
        memory.get("embedding_lookup_mode")
        or getattr(node_embeddings, "lookup_mode", "")
        or "",
    )
    lazy_embedding_mb = float(memory.get("embedding_lru_cache_mb", 0.0) or 0.0)
    deploy_embedding_mb = lazy_embedding_mb if embedding_lookup_mode == "lazy_mmap_lru" else embedding_total_mb
    model_param_mb = float(memory.get("online_deploy_model_param_mb", 0.0) or 0.0)
    runtime_buffer_mb = float(memory.get("online_deploy_runtime_buffer_mb", 0.0) or 0.0)
    threshold_cache_mb = float(memory.get("online_deploy_threshold_cache_mb", 0.0) or 0.0)
    event_writer_buffer_mb = float(
        memory.get("online_minimal_alert_buffer_mb", 0.0) or 0.0,
    )
    excluded_embedding_file_backed_mb = float(mapped_embedding_mb or 0.0)
    excluded_validation_file_backed_mb = (
        float(mapped_validation_mb)
        if mapped_validation_mb is not None and mapped_validation_mb > 0.0
        else min(
            event_index_validation_mb,
            max(file_backed_peak - excluded_test_file_backed_mb - excluded_embedding_file_backed_mb, 0.0),
        )
    )
    core_file_backed_mb = max(
        file_backed_peak
        - excluded_test_file_backed_mb
        - excluded_embedding_file_backed_mb
        - excluded_validation_file_backed_mb,
        0.0,
    )
    core_anonymous_mb = float(
        memory.get("online_minimal_anonymous_rss_mb")
        or smaps.get("anonymous_rss_mb")
        or 0.0,
    )
    core_peak = core_anonymous_mb + core_file_backed_mb
    payload = {
        "definition": (
            "online_event_core_rss excludes replay event_index file-backed pages, "
            "validation cache, label attach, and offline node aggregation from the "
            "process RSS. It keeps model/state/threshold/endpoint-cache/event-writer "
            "objects required for online event scoring."
        ),
        "process_rss_peak_mb": float(
            memory.get("rss_test_peak_mb", memory.get("online_minimal_rss_peak_mb", 0.0))
            or memory.get("online_minimal_rss_peak_mb", 0.0)
            or 0.0,
        ),
        "process_rss_final_mb": float(memory.get("online_minimal_rss_final_mb", 0.0) or 0.0),
        "online_event_core_rss_mb": float(core_peak),
        "online_event_core_rss_peak_mb": float(core_peak),
        "online_event_core_anonymous_rss_mb": float(core_anonymous_mb),
        "online_event_core_file_backed_rss_mb": float(core_file_backed_mb),
        "online_deploy_required_memory_mb": float(
            deploy_embedding_mb
            + state_table_mb
            + float(endpoint_cache_mb)
            + threshold_cache_mb
            + model_param_mb
            + runtime_buffer_mb
            + event_writer_buffer_mb
        ),
        "online_deploy_primary_memory_mb": float(
            deploy_embedding_mb + state_table_mb + float(endpoint_cache_mb),
        ),
        "online_deploy_primary_memory_mb_full_pipeline": float(
            embedding_breakdown.get("online_deploy_primary_memory_mb_full_pipeline", 0.0),
        ),
        "online_deploy_primary_memory_mb_test_stream": float(
            embedding_breakdown.get("online_deploy_primary_memory_mb_test_stream", 0.0),
        ),
        "online_deploy_primary_memory_mb_lazy_estimate": float(
            embedding_breakdown.get("online_deploy_primary_memory_mb_lazy_estimate", 0.0),
        ),
        "online_deploy_embedding_mb": float(deploy_embedding_mb),
        "online_deploy_state_table_mb": float(state_table_mb),
        "online_deploy_endpoint_cache_mb": float(endpoint_cache_mb),
        "online_deploy_threshold_cache_mb": float(threshold_cache_mb),
        "online_deploy_model_param_mb": float(model_param_mb),
        "conditional_head_param_mb": float(memory.get("conditional_head_param_mb", 0.0) or 0.0),
        "state_model_param_mb": float(memory.get("state_model_param_mb", 0.0) or 0.0),
        "calibration_param_mb": float(memory.get("calibration_param_mb", 0.0) or 0.0),
        "online_deploy_runtime_buffer_mb": float(runtime_buffer_mb),
        "python_runtime_overhead_estimate_mb": float(
            max(core_anonymous_mb - state_table_mb - model_param_mb - runtime_buffer_mb, 0.0),
        ),
        "state_table_mb": state_table_mb,
        "embedding_table_total_mb": embedding_total_mb,
        "embedding_lookup_mode": embedding_lookup_mode or str(
            getattr(node_embeddings, "lookup_mode", "global_or_compact_mmap"),
        ),
        "embedding_lazy_backing": str(memory.get("embedding_lazy_backing", "")),
        "embedding_cache_max_nodes": int(memory.get("embedding_cache_max_nodes", 0) or 0),
        "embedding_cache_active_nodes": int(memory.get("embedding_cache_active_nodes", 0) or 0),
        "embedding_lru_cache_mb": float(memory.get("embedding_lru_cache_mb", 0.0) or 0.0),
        "embedding_lru_cache_mb_actual": float(
            memory.get("embedding_lru_cache_mb_actual")
            or memory.get("embedding_lru_cache_mb")
            or 0.0,
        ),
        "embedding_lru_cache_mb_capacity": float(
            memory.get("embedding_lru_cache_mb_capacity", 0.0) or 0.0,
        ),
        "embedding_cache_hit_count": int(memory.get("embedding_cache_hit_count", 0) or 0),
        "embedding_cache_miss_count": int(memory.get("embedding_cache_miss_count", 0) or 0),
        "embedding_cache_hit_rate": float(memory.get("embedding_cache_hit_rate", 0.0) or 0.0),
        "embedding_cache_miss_rate": float(memory.get("embedding_cache_miss_rate", 0.0) or 0.0),
        "embedding_lookup_seconds": float(memory.get("embedding_lookup_seconds", 0.0) or 0.0),
        "embedding_file_backed_rss_mb": float(excluded_embedding_file_backed_mb),
        "embedding_table_file_backed_mb": float(excluded_embedding_file_backed_mb),
        "embedding_lookup_active_pages_mb": float(excluded_embedding_file_backed_mb),
        "embedding_table_file_size_mb": _phase3g_file_size_mb(paths.get("node_embeddings")),
        "embedding_dtype": str(getattr(node_embeddings, "dtype", "")),
        "embedding_shape": list(getattr(node_embeddings, "shape", [])),
        "endpoint_cache_mb": float(endpoint_cache_mb),
        "endpoint_pair_keys_mb": float(endpoint_pair_keys_mb),
        "endpoint_pair_counts_mb": float(endpoint_pair_counts_mb),
        "endpoint_action_keys_mb": float(endpoint_action_keys_mb),
        "endpoint_action_counts_mb": float(endpoint_action_counts_mb),
        "endpoint_cache_loaded_mode": "compact_v2"
        if bool(endpoint_meta.get("compact_cache", False))
        else "",
        "threshold_cache_mb": threshold_cache_mb,
        "event_writer_buffer_mb": event_writer_buffer_mb,
        "event_node_coverage_writer_mb": float(
            memory.get("online_minimal_event_node_coverage_writer_mb", 0.0) or 0.0,
        ),
        "test_event_index_file_size_mb": float(event_index_test_mb),
        "validation_event_index_file_size_mb": float(event_index_validation_mb),
        "excluded_test_event_index_file_backed_mb": float(excluded_test_file_backed_mb),
        "excluded_validation_event_index_file_backed_mb": float(
            excluded_validation_file_backed_mb,
        ),
        "excluded_test_cache_file_backed_mb": float(excluded_test_file_backed_mb),
        "excluded_embedding_table_file_backed_mb": float(excluded_embedding_file_backed_mb),
        "excluded_validation_cache_mb": 0.0,
        "validation_cache_rss_mb": 0.0,
        "post_stream_eval_rss_mb": float(memory.get("post_stream_eval_rss_mb", 0.0) or 0.0),
        "node_aggregation_rss_mb": 0.0,
        "label_attach_rss_mb": 0.0,
        "pandas_or_dataframe_rss_mb": 0.0,
        "rss_timeline_csv": _phase3g_write_rss_timeline(output_dir, rss_timeline or []),
        "phase3g_embedding_memory_breakdown_json": str(
            Path(output_dir) / "phase3g_embedding_memory_breakdown.json",
        ),
    }
    payload["embedding_memory_breakdown"] = dict(embedding_breakdown)
    path = Path(output_dir) / "rss_breakdown_online_event_core.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    breakdown_path = Path(output_dir) / "phase3g_embedding_memory_breakdown.json"
    breakdown_path.write_text(
        json.dumps(embedding_breakdown, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def _phase3g_conditional_head_param_mb(head: ConditionalSemanticHead) -> float:
    if head.is_dual_head:
        total = int(
            head.event_w1.nbytes
            + head.event_w2.nbytes
            + head.event_bias.nbytes
            + head.action_w1.nbytes
            + head.action_w2.nbytes
            + head.action_bias.nbytes,
        )
    else:
        total = int(head.w1.nbytes + head.w2.nbytes + head.bias.nbytes)
    return float(total / (1024.0 * 1024.0))


def _phase3g_conditional_model_param_breakdown_mb(
    head: ConditionalSemanticHead,
    model: SSPMLowRankModel,
) -> dict[str, float]:
    head_mb = _phase3g_conditional_head_param_mb(head)
    state_model_mb = _phase3g_state_model_param_mb(model)
    calibration_mb = _phase3g_calibration_param_mb(model)
    return {
        "conditional_head_param_mb": float(head_mb),
        "state_model_param_mb": float(state_model_mb),
        "calibration_param_mb": float(calibration_mb),
        "model_param_mb": float(head_mb + state_model_mb + calibration_mb),
    }


def _phase3g_state_model_param_mb(model: SSPMLowRankModel) -> float:
    state = model.state_model.state_dict()
    total = 0
    for value in state.values():
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
    return float(total / (1024.0 * 1024.0))


def _phase3g_calibration_param_mb(model: SSPMLowRankModel) -> float:
    total = 0
    calibration = getattr(model, "residual_calibration_model", None)
    if calibration is not None:
        for value in getattr(calibration, "__dict__", {}).values():
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
            elif isinstance(value, Mapping):
                for nested in value.values():
                    if isinstance(nested, np.ndarray):
                        total += int(nested.nbytes)
    gate = getattr(model, "update_gate_calibrator", None)
    if gate is not None:
        for value in getattr(gate, "__dict__", {}).values():
            if isinstance(value, np.ndarray):
                total += int(value.nbytes)
    return float(total / (1024.0 * 1024.0))


def _phase3g_embedding_memory_breakdown_for_paths(
    *,
    paths: Mapping[str, Path],
    node_embeddings: Any,
    state_table_mb: float,
    endpoint_cache_mb: float,
) -> dict[str, Any]:
    split_indexes: dict[str, np.ndarray] = {}
    global_embedding_node_count = _phase3g_global_embedding_node_count(paths)
    try:
        for split in ("train", "validation", "test"):
            key = f"event_index_{split}"
            if key not in paths:
                continue
            path = Path(paths[key])
            item_size = np.dtype(EVENT_INDEX_DTYPE).itemsize
            if not path.exists() or item_size <= 0:
                continue
            num_events = int(path.stat().st_size // item_size)
            split_indexes[split] = open_event_index_memmap(path, num_events=num_events, mode="r")
        payload = embedding_memory_breakdown(
            node_embeddings=node_embeddings,
            split_event_indexes=split_indexes,
            active_state_table_mb=float(state_table_mb),
            endpoint_cache_mb=float(endpoint_cache_mb),
            global_embedding_node_count=global_embedding_node_count,
        )
        payload["event_index_source"] = "phase3e_or_compact_event_index_memmap"
        return payload
    finally:
        for index in split_indexes.values():
            del index


def _phase3g_global_embedding_node_count(paths: Mapping[str, Path]) -> int | None:
    compact_meta_path = paths.get("compact_embedding_meta")
    if compact_meta_path is None:
        return None
    path = Path(compact_meta_path)
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("num_nodes_global_embedding_table", "global_embedding_nodes"):
        value = meta.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _phase3g_write_event_coverage_report(
    *,
    output_dir: Path,
    alert_path: Path,
    coverage_rows: Sequence[Mapping[str, Any]] | None,
    idx_to_db_node_id: Mapping[int, int],
    abnormal_db_node_ids: set[int],
    topk_values: Sequence[int],
    node_pool_score_mode: str = "base_conf",
    config: SlimConfig | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    compact_node_labels = None
    compact_gt_summary = None
    if config is not None and is_optc_dataset(getattr(config, "dataset", "")):
        original_to_canonical = _phase3g_load_optc_original_to_canonical(config)
        node_id_to_idx = _phase3g_load_node_id_to_idx_for_eval(config)
        if original_to_canonical:
            compact_node_labels, compact_gt_summary = _phase3g_compact_gt_labels_for_eval(
                abnormal_db_node_ids=abnormal_db_node_ids,
                idx_to_db_node_id=idx_to_db_node_id,
                original_to_canonical_netflow=original_to_canonical,
                node_id_to_idx=node_id_to_idx,
            )
    evaluated_event_rows = []
    for raw_row in _iter_csv_rows(alert_path):
        label, _, _ = _phase3g_label_for_alert_row(
            raw_row,
            idx_to_db_node_id,
            abnormal_db_node_ids,
            compact_node_labels=compact_node_labels,
        )
        evaluated_event_rows.append(
            _evaluated_event_row(raw_row, {int(raw_row["event_index"]): label}),
        )
    coverage = list(coverage_rows or _phase3g_event_node_coverage_from_alerts(alert_path))
    coverage_summary = _phase3g_write_event_node_coverage_outputs(
        output_dir=output_dir,
        coverage_rows=coverage,
        event_rows=evaluated_event_rows,
        idx_to_db_node_id=idx_to_db_node_id,
        abnormal_db_node_ids=abnormal_db_node_ids,
        compact_node_labels=compact_node_labels,
        compact_gt_summary=compact_gt_summary,
    )
    pool_summary = _phase3g_write_node_pool_rebuilt_outputs(
        output_dir=output_dir,
        coverage_rows=coverage,
        idx_to_db_node_id=idx_to_db_node_id,
        abnormal_db_node_ids=abnormal_db_node_ids,
        topk_values=topk_values,
        node_pool_score_mode=str(node_pool_score_mode),
        event_rows=evaluated_event_rows,
        compact_node_labels=compact_node_labels,
        compact_gt_summary=compact_gt_summary,
    )
    fp_paths = _phase3g_write_fp_group_outputs(
        output_dir=output_dir,
        event_rows=evaluated_event_rows,
        coverage_rows=[
            {
                **dict(row),
                "node_label": (compact_node_labels or {}).get(
                    int(row.get("node_idx", -1)),
                    "malicious"
                    if int(idx_to_db_node_id.get(int(row.get("node_idx", -1)), -1))
                    in abnormal_db_node_ids
                    else "benign",
                ),
            }
            for row in coverage
        ],
    )
    return {**coverage_summary, **pool_summary, **fp_paths}


def _phase3g_backfill_result_reports_from_config(
    *,
    result_dir: Path,
    config: SlimConfig,
) -> Path:
    output_dir = Path(result_dir)
    paths = _phase3e_require_artifacts(config)
    idx_to_db_node_id = _phase3e_load_idx_to_node_id(paths)
    abnormal_nodes = _phase3g_load_abnormal_db_nodes_for_eval(config)
    node_embeddings = np.load(paths["node_embeddings"], mmap_mode="r")
    metrics_path = output_dir / "metrics.json"
    eval_path = output_dir / "eval_causal_semantics_slim.json"
    eval_payload = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    metrics_payload = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    )
    endpoint_meta: dict[str, Any] = {}
    endpoint_cache_mb = 0.0
    phase3g = metrics_payload.get("eval", {}).get("phase3g", {})
    if isinstance(phase3g, Mapping):
        endpoint_payload = phase3g.get("conditional_endpoint_suppression", {})
        if isinstance(endpoint_payload, Mapping):
            endpoint_cache_mb = float(endpoint_payload.get("endpoint_cache_mb", 0.0) or 0.0)
            meta_path = endpoint_payload.get("cache_meta_path")
            if meta_path and Path(str(meta_path)).exists():
                endpoint_meta = json.loads(Path(str(meta_path)).read_text(encoding="utf-8"))
    report = _phase3g_write_event_coverage_report(
        output_dir=output_dir,
        alert_path=output_dir / "online_event_alerts.csv",
        coverage_rows=None,
        idx_to_db_node_id=idx_to_db_node_id,
        abnormal_db_node_ids=abnormal_nodes,
        topk_values=_parse_int_list(config.node_pool_topk_values),
        node_pool_score_mode=str(config.node_pool_score_mode),
        config=config,
    )
    memory_source = eval_payload if eval_payload else dict(metrics_payload.get("eval", {}))
    rss_payload = _phase3g_write_online_event_core_rss_breakdown(
        output_dir=output_dir,
        eval_payload=memory_source,
        paths=paths,
        node_embeddings=node_embeddings,
        endpoint_cache_mb=endpoint_cache_mb,
        endpoint_cache_meta=endpoint_meta,
        rss_timeline=[],
    )
    if eval_payload:
        outputs = dict(eval_payload.get("outputs", {}))
        outputs.update(
            {
                "rss_breakdown_online_event_core_json": str(
                    output_dir / "rss_breakdown_online_event_core.json",
                ),
                "rss_timeline_csv": str(output_dir / "rss_timeline.csv"),
                "online_event_node_coverage_csv": str(
                    output_dir / "online_event_node_coverage.csv",
                ),
                "online_event_node_coverage_strict_csv": str(
                    output_dir / "online_event_node_coverage_strict.csv",
                ),
                "online_event_node_coverage_relaxed_csv": str(
                    output_dir / "online_event_node_coverage_relaxed.csv",
                ),
                "online_event_node_coverage_summary_json": str(
                    output_dir / "online_event_node_coverage_summary.json",
                ),
                "node_pool_rebuilt_summary_json": str(
                    output_dir / "node_pool_rebuilt_summary.json",
                ),
                "node_topk_metrics_csv": str(output_dir / "node_topk_metrics.csv"),
                "node_topk_metrics_json": str(output_dir / "node_topk_metrics.json"),
                "event_fp_group_summary_csv": str(output_dir / "event_fp_group_summary.csv"),
                "node_fp_group_summary_csv": str(output_dir / "node_fp_group_summary.csv"),
            },
        )
        eval_payload["outputs"] = outputs
        eval_payload["online_event_node_coverage"] = report
        eval_payload["rss_breakdown_online_event_core"] = rss_payload
        primary = dict(eval_payload.get("primary_online_metrics", {}))
        primary["online_event_node_coverage"] = {
            key: value
            for key, value in report.items()
            if not str(key).endswith("_csv") and not str(key).endswith("_json")
        }
        primary["node_pool_topk"] = report.get("node_pool_topk", {})
        eval_payload["primary_online_metrics"] = primary
        eval_path.write_text(
            json.dumps(eval_payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if metrics_payload:
        metrics_payload["online_event_node_coverage"] = report
        metrics_payload["rss_breakdown_online_event_core"] = rss_payload
        metrics_payload.setdefault("paths", {}).update(
            {
                "rss_breakdown_online_event_core_json": str(
                    output_dir / "rss_breakdown_online_event_core.json",
                ),
                "online_event_node_coverage_csv": str(
                    output_dir / "online_event_node_coverage.csv",
                ),
                "online_event_node_coverage_summary_json": str(
                    output_dir / "online_event_node_coverage_summary.json",
                ),
            },
        )
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return output_dir / "online_event_node_coverage_summary.json"


# Explicit cross-module imports; replaces the migration namespace bridge.
from scripts.pipeline.conditional.infer import _phase3e_event_labels_for_alerts
from scripts.pipeline.features.conditional_context import (
    LazyMmapLruEmbeddingLookup,
    _phase3e_load_idx_to_node_id,
    _phase3g_empty_report_metrics,
    _phase3g_run_only_from_out_tag,
    _write_conditional_endpoint_suppressed_tp_audit,
    _write_conditional_endpoint_suppressed_tp_events,
)
from scripts.pipeline.io.cache_payloads import _phase3g_load_abnormal_db_nodes_for_eval
from scripts.pipeline.io.event_artifacts import _phase3e_entity_type_name, _phase3e_require_artifacts
from scripts.pipeline.outputs.alert_output import (
    _alert_event_indices,
    _attach_node_labels,
    _current_rss_mb,
    _eval_result_counts,
    _evaluated_event_row,
    _event_metrics_from_counts,
    _is_attack_label,
    _iter_csv_rows,
    _label_name,
    _node_alert_support,
    _node_pool_sort_key,
    _node_pool_topk_metrics,
    _parse_int_list,
    _relaxed_node_eval_result,
    _safe_float,
    _write_csv,
)
from scripts.pipeline.outputs.metrics_summary import _current_smaps_rollup_mb
from scripts.pipeline.state.online_state_runtime import _parse_bool
