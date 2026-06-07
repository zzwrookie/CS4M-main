#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import math
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

from cs4m.config.config import DATASET_DEFAULT_CONFIG
from cs4m.config.provnet_utils import datetime_to_ns_time_US
from utils.score_tuning_utils import (
    false_positives_per_day_from_mask,
    process_level_recall_from_mask,
    threshold_from_reference_scores,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate event-level anomaly scores with alerting metrics.")
    p.add_argument("--pred_npz", required=True)
    p.add_argument("--ref_npz", required=True)
    p.add_argument("--score_key", required=True)
    p.add_argument("--structured_pkl", default="", help="Optional structured event pickle for raw node-level evaluation.")
    p.add_argument("--dataset_name", default=None)
    p.add_argument("--fixed_threshold", type=float, default=None)
    p.add_argument("--threshold_quantile", type=float, default=0.999)
    p.add_argument("--threshold_ref_mode", choices=("all", "benign_only"), default="benign_only")
    p.add_argument("--positive_mode", choices=("suspect_or_malicious", "malicious_only"), default="malicious_only")
    p.add_argument("--fp_label_mode", choices=("benign_only", "all_nonpositive"), default="benign_only")
    p.add_argument("--window_indices", default="", help="Optional comma-separated attack window indices to evaluate.")
    p.add_argument("--out_json", required=True)
    return p.parse_args()


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def _load_structured_subset(path: str, event_index: np.ndarray) -> Dict[str, np.ndarray] | None:
    path = str(path).strip()
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict payload from {path}, got {type(payload)!r}")
    required = {"event_index", "src_idx", "dst_idx", "label"}
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    src = np.asarray(payload["src_idx"], dtype=np.int64)
    dst = np.asarray(payload["dst_idx"], dtype=np.int64)
    labels = np.asarray(payload["label"], dtype=np.int64)
    index = np.asarray(payload["event_index"], dtype=np.int64)
    target = np.asarray(event_index, dtype=np.int64)
    if index.shape[0] == target.shape[0] and np.array_equal(index, target):
        return {
            "event_index": index,
            "src_idx": src,
            "dst_idx": dst,
            "label": labels,
        }
    order = np.argsort(index, kind="stable")
    sorted_index = index[order]
    positions = np.searchsorted(sorted_index, target)
    valid = (
        (positions >= 0)
        & (positions < sorted_index.shape[0])
        & (sorted_index[positions] == target)
    )
    if not np.all(valid):
        raise ValueError(f"{path} cannot align structured events to prediction event_index.")
    selected = order[positions]
    return {
        "event_index": index[selected],
        "src_idx": src[selected],
        "dst_idx": dst[selected],
        "label": labels[selected],
    }


def _positive_mask(labels: np.ndarray, positive_mode: str) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if positive_mode == "malicious_only":
        return labels == 2
    return labels > 0


def _pr_auc(scores: np.ndarray, labels: np.ndarray, positive_mode: str) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.isfinite(scores)
    if not np.any(valid):
        return float("nan")
    positive = _positive_mask(labels=labels, positive_mode=positive_mode)[valid].astype(np.int64)
    scores = scores[valid]
    num_pos = int(np.sum(positive))
    if num_pos <= 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    positive = positive[order]
    tp = np.cumsum(positive, dtype=np.int64)
    fp = np.cumsum(1 - positive, dtype=np.int64)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(num_pos)
    prev_recall = np.r_[0.0, recall[:-1]]
    gain = recall - prev_recall
    ap = float(np.sum(gain * precision))
    return ap


def _recall_at_topk(scores: np.ndarray, labels: np.ndarray, positive_mode: str, topk: int) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.isfinite(scores)
    if not np.any(valid):
        return float("nan")
    positive = _positive_mask(labels=labels, positive_mode=positive_mode)
    num_pos = int(np.sum(positive))
    if num_pos <= 0:
        return float("nan")
    order = np.argsort(-scores[valid], kind="stable")
    valid_idx = np.flatnonzero(valid)[order]
    keep = valid_idx[: min(int(topk), valid_idx.shape[0])]
    return float(np.sum(positive[keep]) / num_pos)


def _ns_from_str(value: str) -> int:
    return int(datetime_to_ns_time_US(str(value)))


def _parse_window_indices(raw: str) -> List[int]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def _attack_windows_for_dataset(dataset_name: str | None, window_indices: List[int] | None = None) -> List[Tuple[str, int, int]]:
    if not dataset_name:
        return []
    cfg = DATASET_DEFAULT_CONFIG.get(str(dataset_name))
    if cfg is None:
        return []
    windows = []
    selected = set(window_indices or [])
    for idx, item in enumerate(cfg.get("attack_windows", [])):
        if selected and idx not in selected:
            continue
        if len(item) != 3:
            continue
        windows.append((str(item[0]), _ns_from_str(str(item[1])), _ns_from_str(str(item[2]))))
    return windows


def _attack_window_metrics(
    timestamps_ns: np.ndarray,
    alerts: np.ndarray,
    dataset_name: str | None,
    window_indices: List[int] | None = None,
) -> Dict[str, object]:
    windows = _attack_windows_for_dataset(dataset_name=dataset_name, window_indices=window_indices)
    if not windows:
        return {
            "num_windows": 0,
            "detected_windows": 0,
            "attack_window_recall": float("nan"),
            "mean_time_to_detect_seconds": float("nan"),
            "windows": [],
        }
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    alerts = np.asarray(alerts, dtype=bool)
    detected = 0
    ttd_values = []
    rows = []
    for name, start_ns, end_ns in windows:
        mask = (timestamps_ns >= start_ns) & (timestamps_ns <= end_ns)
        alert_mask = mask & alerts
        if np.any(alert_mask):
            detected += 1
            first_alert = int(np.min(timestamps_ns[alert_mask]))
            ttd_sec = max(float(first_alert - start_ns) / 1e9, 0.0)
            ttd_values.append(ttd_sec)
            rows.append({"name": name, "detected": True, "time_to_detect_seconds": ttd_sec})
        else:
            rows.append({"name": name, "detected": False, "time_to_detect_seconds": None})
    return {
        "num_windows": int(len(windows)),
        "detected_windows": int(detected),
        "attack_window_recall": float(detected / max(len(windows), 1)),
        "mean_time_to_detect_seconds": float(np.mean(ttd_values)) if ttd_values else float("nan"),
        "windows": rows,
    }


def _process_confusion(
    process_idx: np.ndarray,
    labels: np.ndarray,
    alerts: np.ndarray,
    positive_mode: str,
) -> Dict[str, float | int]:
    process_idx = np.asarray(process_idx, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    alerts = np.asarray(alerts, dtype=bool)
    positive = _positive_mask(labels=labels, positive_mode=positive_mode)

    valid_process = process_idx >= 0
    all_processes = np.unique(process_idx[valid_process])
    positive_processes = np.unique(process_idx[valid_process & positive])
    negative_processes = np.setdiff1d(all_processes, positive_processes, assume_unique=False)
    alerted_processes = np.unique(process_idx[valid_process & alerts])

    tp = int(np.intersect1d(alerted_processes, positive_processes, assume_unique=False).shape[0])
    fp = int(np.intersect1d(alerted_processes, negative_processes, assume_unique=False).shape[0])
    fn = int(np.setdiff1d(positive_processes, alerted_processes, assume_unique=False).shape[0])
    tn = int(np.setdiff1d(negative_processes, alerted_processes, assume_unique=False).shape[0])

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(positive_processes.shape[0], 1))
    tnr = float(tn / max(negative_processes.shape[0], 1))
    fpr = float(fp / max(negative_processes.shape[0], 1))
    return {
        "num_positive_processes": int(positive_processes.shape[0]),
        "num_negative_processes": int(negative_processes.shape[0]),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "tpr": recall,
        "tnr": tnr,
        "fpr": fpr,
    }


def _node_confusion(
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    labels: np.ndarray,
    alerts: np.ndarray,
    positive_mode: str,
) -> Dict[str, float | int]:
    src_idx = np.asarray(src_idx, dtype=np.int64)
    dst_idx = np.asarray(dst_idx, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    alerts = np.asarray(alerts, dtype=bool)
    positive = _positive_mask(labels=labels, positive_mode=positive_mode)

    all_nodes = np.unique(
        np.concatenate(
            [
                src_idx[src_idx >= 0],
                dst_idx[dst_idx >= 0],
            ],
            axis=0,
        )
    )
    positive_nodes = np.unique(
        np.concatenate(
            [
                src_idx[(src_idx >= 0) & positive],
                dst_idx[(dst_idx >= 0) & positive],
            ],
            axis=0,
        )
    )
    negative_nodes = np.setdiff1d(all_nodes, positive_nodes, assume_unique=False)
    alerted_nodes = np.unique(
        np.concatenate(
            [
                src_idx[(src_idx >= 0) & alerts],
                dst_idx[(dst_idx >= 0) & alerts],
            ],
            axis=0,
        )
    )

    tp = int(np.intersect1d(alerted_nodes, positive_nodes, assume_unique=False).shape[0])
    fp = int(np.intersect1d(alerted_nodes, negative_nodes, assume_unique=False).shape[0])
    fn = int(np.setdiff1d(positive_nodes, alerted_nodes, assume_unique=False).shape[0])
    tn = int(np.setdiff1d(negative_nodes, alerted_nodes, assume_unique=False).shape[0])
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(positive_nodes.shape[0], 1))
    tnr = float(tn / max(negative_nodes.shape[0], 1))
    fpr = float(fp / max(negative_nodes.shape[0], 1))
    return {
        "num_positive_nodes": int(positive_nodes.shape[0]),
        "num_negative_nodes": int(negative_nodes.shape[0]),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "tpr": recall,
        "tnr": tnr,
        "fpr": fpr,
    }


def _node_confusion_relaxed(
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    labels: np.ndarray,
    alerts: np.ndarray,
    positive_mode: str,
) -> Dict[str, float | int]:
    src_idx = np.asarray(src_idx, dtype=np.int64)
    dst_idx = np.asarray(dst_idx, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    alerts = np.asarray(alerts, dtype=bool)
    positive = _positive_mask(labels=labels, positive_mode=positive_mode)

    all_nodes = np.unique(
        np.concatenate(
            [
                src_idx[src_idx >= 0],
                dst_idx[dst_idx >= 0],
            ],
            axis=0,
        )
    )
    positive_nodes = np.unique(
        np.concatenate(
            [
                src_idx[(src_idx >= 0) & positive],
                dst_idx[(dst_idx >= 0) & positive],
            ],
            axis=0,
        )
    )
    if positive_mode == "malicious_only":
        suspect_mask = labels == 1
        ignored_nodes = np.unique(
            np.concatenate(
                [
                    src_idx[(src_idx >= 0) & suspect_mask],
                    dst_idx[(dst_idx >= 0) & suspect_mask],
                ],
                axis=0,
            )
        )
        ignored_nodes = np.setdiff1d(ignored_nodes, positive_nodes, assume_unique=False)
    else:
        ignored_nodes = np.zeros((0,), dtype=np.int64)

    negative_nodes = np.setdiff1d(all_nodes, np.union1d(positive_nodes, ignored_nodes), assume_unique=False)
    alerted_nodes = np.unique(
        np.concatenate(
            [
                src_idx[(src_idx >= 0) & alerts],
                dst_idx[(dst_idx >= 0) & alerts],
            ],
            axis=0,
        )
    )

    tp = int(np.intersect1d(alerted_nodes, positive_nodes, assume_unique=False).shape[0])
    fp = int(np.intersect1d(alerted_nodes, negative_nodes, assume_unique=False).shape[0])
    fn = int(np.setdiff1d(positive_nodes, alerted_nodes, assume_unique=False).shape[0])
    tn = int(np.setdiff1d(negative_nodes, alerted_nodes, assume_unique=False).shape[0])
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(positive_nodes.shape[0], 1))
    tnr = float(tn / max(negative_nodes.shape[0], 1))
    fpr = float(fp / max(negative_nodes.shape[0], 1))
    return {
        "num_positive_nodes": int(positive_nodes.shape[0]),
        "num_negative_nodes": int(negative_nodes.shape[0]),
        "num_ignored_suspect_nodes": int(ignored_nodes.shape[0]),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "tpr": recall,
        "tnr": tnr,
        "fpr": fpr,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)

    pred = _load_npz(args.pred_npz)
    ref = _load_npz(args.ref_npz)
    pred_scores = np.asarray(pred[str(args.score_key)], dtype=np.float64)
    ref_scores = np.asarray(ref[str(args.score_key)], dtype=np.float64)
    pred_labels = np.asarray(pred["label"], dtype=np.int64)
    ref_labels = np.asarray(ref["label"], dtype=np.int64)
    pred_timestamps = np.asarray(pred["timestamp_ns"], dtype=np.int64)
    pred_process = np.asarray(pred["process_idx"], dtype=np.int64)
    structured = _load_structured_subset(path=args.structured_pkl, event_index=np.asarray(pred["event_index"], dtype=np.int64))

    if args.fixed_threshold is not None:
        threshold = float(args.fixed_threshold)
    else:
        threshold = threshold_from_reference_scores(
            ref_scores=ref_scores,
            ref_labels=ref_labels,
            q=float(args.threshold_quantile),
            ref_mode=str(args.threshold_ref_mode),
        )

    positive_mask = _positive_mask(labels=pred_labels, positive_mode=str(args.positive_mode))
    negative_mask = ~positive_mask
    alerts = np.isfinite(pred_scores) & (pred_scores >= float(threshold))

    if str(args.fp_label_mode) == "benign_only":
        fp_mask = (pred_labels == 0) & alerts
    else:
        fp_mask = (~positive_mask) & alerts

    tp = int(np.sum(alerts & positive_mask))
    fp_all_negative = int(np.sum(alerts & negative_mask))
    tn = int(np.sum((~alerts) & negative_mask))
    fn = int(np.sum((~alerts) & positive_mask))
    precision = float(tp / max(tp + fp_all_negative, 1))
    recall = float(tp / max(int(np.sum(positive_mask)), 1))
    tnr = float(tn / max(int(np.sum(negative_mask)), 1))
    fpr = float(fp_all_negative / max(int(np.sum(negative_mask)), 1))

    summary = {
        "meta": {
            "pred_npz": os.path.abspath(args.pred_npz),
            "score_key": str(args.score_key),
            "ref_npz": os.path.abspath(args.ref_npz),
            "fixed_threshold": float(args.fixed_threshold) if args.fixed_threshold is not None else None,
            "threshold_quantile": float(args.threshold_quantile),
            "threshold_ref_mode": str(args.threshold_ref_mode),
            "dataset_name": str(args.dataset_name) if args.dataset_name is not None else None,
            "positive_mode": str(args.positive_mode),
            "fp_label_mode": str(args.fp_label_mode),
        },
        "counts": {
            "events_total": int(pred_labels.shape[0]),
            "events_benign": int(np.sum(pred_labels == 0)),
            "events_suspect": int(np.sum(pred_labels == 1)),
            "events_malicious": int(np.sum(pred_labels == 2)),
            "events_positive": int(np.sum(positive_mask)),
            "events_negative": int(np.sum(~positive_mask)),
        },
        "ranking_metrics": {
            "pr_auc": _pr_auc(scores=pred_scores, labels=pred_labels, positive_mode=str(args.positive_mode)),
            "recall_at_topk": {
                "100": _recall_at_topk(scores=pred_scores, labels=pred_labels, positive_mode=str(args.positive_mode), topk=100),
                "1000": _recall_at_topk(scores=pred_scores, labels=pred_labels, positive_mode=str(args.positive_mode), topk=1000),
            },
        },
        "threshold_metrics": {
            "threshold": float(threshold),
            "false_positives_total": int(np.sum(fp_mask)),
            "false_positives_per_day": float(false_positives_per_day_from_mask(timestamps_ns=pred_timestamps, fp_mask=fp_mask)),
            "alerts_total": int(np.sum(alerts)),
            "confusion_matrix": {
                "tp": tp,
                "fp": fp_all_negative,
                "tn": tn,
                "fn": fn,
            },
            "classification_metrics": {
                "precision": precision,
                "recall": recall,
                "tpr": recall,
                "tnr": tnr,
                "fpr": fpr,
            },
            "process_level_recall": float(
                process_level_recall_from_mask(
                    process_idx=pred_process,
                    labels=pred_labels,
                    alerts=alerts,
                    positive_mode=str(args.positive_mode),
                )
            ),
            "process_confusion": _process_confusion(
                process_idx=pred_process,
                labels=pred_labels,
                alerts=alerts,
                positive_mode=str(args.positive_mode),
            ),
            "node_confusion": _node_confusion(
                src_idx=np.asarray(structured["src_idx"], dtype=np.int64) if structured is not None else np.full(pred_labels.shape, -1, dtype=np.int64),
                dst_idx=np.asarray(structured["dst_idx"], dtype=np.int64) if structured is not None else np.full(pred_labels.shape, -1, dtype=np.int64),
                labels=np.asarray(structured["label"], dtype=np.int64) if structured is not None else pred_labels,
                alerts=alerts,
                positive_mode=str(args.positive_mode),
            ) if structured is not None else None,
            "node_confusion_relaxed": _node_confusion_relaxed(
                src_idx=np.asarray(structured["src_idx"], dtype=np.int64) if structured is not None else np.full(pred_labels.shape, -1, dtype=np.int64),
                dst_idx=np.asarray(structured["dst_idx"], dtype=np.int64) if structured is not None else np.full(pred_labels.shape, -1, dtype=np.int64),
                labels=np.asarray(structured["label"], dtype=np.int64) if structured is not None else pred_labels,
                alerts=alerts,
                positive_mode=str(args.positive_mode),
            ) if structured is not None else None,
        },
        "attack_window_metrics": _attack_window_metrics(
            timestamps_ns=pred_timestamps,
            alerts=alerts,
            dataset_name=args.dataset_name,
            window_indices=_parse_window_indices(args.window_indices),
        ),
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[done] evaluation -> {args.out_json}")


if __name__ == "__main__":
    main()
