from __future__ import annotations

import resource
import sys
from typing import Any

import numpy as np


def target_for_dataset(dataset: str) -> dict[str, int]:
    name = str(dataset).upper()
    if name.startswith("THEIA"):
        return {"relaxed_tp_gt": 70, "relaxed_fp_lt": 1000}
    if name.startswith("CADETS"):
        return {"relaxed_tp_gt": 55, "relaxed_fp_lt": 1000}
    if name.startswith("CLEARSCOPE"):
        return {"relaxed_tp_gt": 20, "relaxed_fp_lt": 1000}
    return {"relaxed_tp_gt": 0, "relaxed_fp_lt": 1000}


def empty_event_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "ignored_suspect": 0, "ignored_suspect_alerts": 0}


def update_event_counts(
    strict_counts: dict[str, int],
    relaxed_counts: dict[str, int],
    label: int,
    alert: bool,
) -> None:
    positive = int(label) == 2
    suspect = int(label) == 1
    if alert and positive:
        strict_counts["tp"] += 1
    elif alert and not positive:
        strict_counts["fp"] += 1
    elif (not alert) and positive:
        strict_counts["fn"] += 1
    else:
        strict_counts["tn"] += 1

    if suspect:
        relaxed_counts["ignored_suspect"] += 1
        if alert:
            relaxed_counts["ignored_suspect_alerts"] += 1
        return
    if alert and positive:
        relaxed_counts["tp"] += 1
    elif alert and not positive:
        relaxed_counts["fp"] += 1
    elif (not alert) and positive:
        relaxed_counts["fn"] += 1
    else:
        relaxed_counts["tn"] += 1


def event_metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    tn = int(counts["tn"])
    fn = int(counts["fn"])
    out: dict[str, float | int] = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "tpr": float(tp / max(tp + fn, 1)),
        "tnr": float(tn / max(tn + fp, 1)),
        "fpr": float(fp / max(tn + fp, 1)),
    }
    if int(counts.get("ignored_suspect", 0)) > 0:
        out["ignored_suspect"] = int(counts["ignored_suspect"])
        out["ignored_suspect_alerts"] = int(counts["ignored_suspect_alerts"])
    return out


def node_confusion_from_masks(
    all_nodes: np.ndarray,
    positive_nodes: np.ndarray,
    suspect_nodes: np.ndarray,
    alerted: np.ndarray,
    relaxed: bool,
) -> dict[str, float | int]:
    alerted = np.asarray(alerted, dtype=bool)
    observed = np.asarray(all_nodes, dtype=bool)
    raw_positive = np.asarray(positive_nodes, dtype=bool)
    raw_suspect = np.asarray(suspect_nodes, dtype=bool)
    positive = raw_positive & observed
    suspect = raw_suspect & observed
    ignored = suspect & ~positive if relaxed else np.zeros_like(positive)
    negative = observed & ~positive & ~ignored
    tp = int(np.sum(alerted & positive))
    fp = int(np.sum(alerted & negative))
    fn = int(np.sum((~alerted) & positive))
    tn = int(np.sum((~alerted) & negative))
    out: dict[str, float | int] = {
        "num_positive_nodes": int(np.sum(positive)),
        "num_negative_nodes": int(np.sum(negative)),
        "num_unobserved_positive_nodes_not_counted": int(np.sum(raw_positive & ~observed)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(int(np.sum(positive)), 1)),
        "tpr": float(tp / max(int(np.sum(positive)), 1)),
        "tnr": float(tn / max(int(np.sum(negative)), 1)),
        "fpr": float(fp / max(int(np.sum(negative)), 1)),
    }
    if relaxed:
        out["num_ignored_suspect_nodes"] = int(np.sum(ignored))
        out["relaxed_semantics"] = "In suspect events, GT abnormal endpoints count as TP if alerted; benign suspect-side endpoints are ignored and never counted as FP."
    return out


def memory_snapshot() -> dict[str, float]:
    status: dict[str, float] = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, value = line.split(":", 1)
                    parts = value.strip().split()
                    if parts:
                        status[key] = float(parts[0]) / 1024.0
    except OSError:
        pass
    ru_maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        ru_peak_mb = ru_maxrss / (1024.0 * 1024.0)
    else:
        ru_peak_mb = ru_maxrss / 1024.0
    return {
        "current_rss_mb": float(status.get("VmRSS", ru_peak_mb)),
        "peak_rss_mb": float(status.get("VmHWM", ru_peak_mb)),
        "resource_peak_rss_mb": float(ru_peak_mb),
    }


def rank_scores(node_score: np.ndarray) -> np.ndarray:
    ranked = np.argsort(-node_score, kind="stable")
    return ranked[node_score[ranked] > 0]


def best_sweep_row(sweep: list[dict[str, Any]], target: dict[str, int]) -> dict[str, Any]:
    def objective(row: dict[str, Any]) -> tuple[int, int, int]:
        relaxed = row["relaxed_node"]
        assert isinstance(relaxed, dict)
        bonus = 1 if int(relaxed["tp"]) > int(target["relaxed_tp_gt"]) and int(relaxed["fp"]) < int(target["relaxed_fp_lt"]) else 0
        return (bonus, int(relaxed["tp"]), -int(relaxed["fp"]))

    return max(sweep, key=objective) if sweep else {}


def best_under_fp_target(sweep: list[dict[str, Any]], target: dict[str, int]) -> dict[str, Any]:
    rows = [
        row
        for row in sweep
        if int(row["relaxed_node"]["fp"]) < int(target["relaxed_fp_lt"])
    ]
    return max(rows, key=lambda row: (int(row["relaxed_node"]["tp"]), -int(row["relaxed_node"]["fp"]))) if rows else {}


def label_distribution(labels: np.ndarray) -> dict[str, int]:
    out = {"benign_0": 0, "suspect_1": 0, "positive_2": 0, "unknown": 0}
    for value in labels.tolist():
        label = int(value)
        if label == 0:
            out["benign_0"] += 1
        elif label == 1:
            out["suspect_1"] += 1
        elif label == 2:
            out["positive_2"] += 1
        else:
            out["unknown"] += 1
    return out
