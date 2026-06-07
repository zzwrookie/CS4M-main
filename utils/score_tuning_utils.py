from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np


def fbeta_score(tp: int, fp: int, fn: int, beta: float = 0.25) -> float:
    if tp <= 0:
        return 0.0
    precision = float(tp) / max(int(tp + fp), 1)
    recall = float(tp) / max(int(tp + fn), 1)
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    beta2 = float(beta) * float(beta)
    return float((1.0 + beta2) * precision * recall / max(beta2 * precision + recall, 1e-12))


def unique_days(timestamps_ns: np.ndarray) -> int:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    if timestamps.shape[0] <= 0:
        return 1
    return max(int(np.unique(timestamps.astype("datetime64[ns]").astype("datetime64[D]")).shape[0]), 1)


def false_positives_per_day_from_mask(timestamps_ns: np.ndarray, fp_mask: np.ndarray) -> float:
    fp_mask = np.asarray(fp_mask, dtype=bool)
    if fp_mask.sum() <= 0:
        return 0.0
    return float(int(fp_mask.sum()) / unique_days(timestamps_ns))


def process_level_recall_from_mask(process_idx: np.ndarray, labels: np.ndarray, alerts: np.ndarray, positive_mode: str) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    alerts = np.asarray(alerts, dtype=bool)
    if positive_mode == "malicious_only":
        positive_mask = labels == 2
    else:
        positive_mask = labels > 0
    positive_processes = set(int(pid) for pid in np.asarray(process_idx, dtype=np.int64)[positive_mask].tolist())
    if not positive_processes:
        return float("nan")
    alerted_processes = set(int(pid) for pid in np.asarray(process_idx, dtype=np.int64)[alerts].tolist())
    return float(len(positive_processes & alerted_processes) / len(positive_processes))


def fit_isotonic_regression(scores: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if sample_weight is None:
        w = np.ones_like(y, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    x = x[valid]
    y = y[valid]
    w = w[valid]
    if x.shape[0] <= 0:
        raise RuntimeError("No valid samples for isotonic regression.")

    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    w = w[order]

    xs = []
    ws = []
    ys = []
    for xi, yi, wi in zip(x.tolist(), y.tolist(), w.tolist()):
        if xs and xi == xs[-1]:
            total_w = ws[-1] + wi
            ys[-1] = (ys[-1] * ws[-1] + yi * wi) / max(total_w, 1e-12)
            ws[-1] = total_w
        else:
            xs.append(float(xi))
            ys.append(float(yi))
            ws.append(float(wi))

    block_starts = list(range(len(xs)))
    block_ends = list(range(len(xs)))
    block_sum_w = ws[:]
    block_sum_y = [ys[idx] * ws[idx] for idx in range(len(xs))]

    idx = 0
    while idx < len(block_sum_w) - 1:
        mean_left = block_sum_y[idx] / max(block_sum_w[idx], 1e-12)
        mean_right = block_sum_y[idx + 1] / max(block_sum_w[idx + 1], 1e-12)
        if mean_left <= mean_right + 1e-12:
            idx += 1
            continue
        block_sum_w[idx] += block_sum_w[idx + 1]
        block_sum_y[idx] += block_sum_y[idx + 1]
        block_ends[idx] = block_ends[idx + 1]
        del block_sum_w[idx + 1]
        del block_sum_y[idx + 1]
        del block_starts[idx + 1]
        del block_ends[idx + 1]
        if idx > 0:
            idx -= 1

    thresholds = []
    values = []
    for start, end, sum_w, sum_y in zip(block_starts, block_ends, block_sum_w, block_sum_y):
        thresholds.append(float(xs[end]))
        values.append(float(sum_y / max(sum_w, 1e-12)))

    return {
        "thresholds": np.asarray(thresholds, dtype=np.float32),
        "values": np.asarray(values, dtype=np.float32),
    }


def predict_isotonic_regression(scores: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(model["thresholds"], dtype=np.float64)
    values = np.asarray(model["values"], dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(x)
    if not np.any(valid):
        return out.astype(np.float32, copy=False)
    indices = np.searchsorted(thresholds, x[valid], side="left")
    indices = np.clip(indices, 0, len(values) - 1)
    out[valid] = values[indices]
    return out.astype(np.float32, copy=False)


def conformal_tail_pvalues(reference_scores: np.ndarray, query_scores: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference_scores, dtype=np.float64)
    ref = ref[np.isfinite(ref)]
    if ref.shape[0] <= 0:
        raise RuntimeError("No finite reference scores available for conformal p-values.")
    ref_sorted = np.sort(ref)
    q = np.asarray(query_scores, dtype=np.float64)
    out = np.full(q.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(q)
    if not np.any(valid):
        return out.astype(np.float32, copy=False)
    idx = np.searchsorted(ref_sorted, q[valid], side="left")
    tail_count = ref_sorted.shape[0] - idx
    out[valid] = (tail_count + 1.0) / float(ref_sorted.shape[0] + 1.0)
    return out.astype(np.float32, copy=False)


def threshold_from_reference_scores(
    ref_scores: np.ndarray,
    ref_labels: np.ndarray,
    q: float,
    ref_mode: str = "all",
) -> float:
    ref_scores = np.asarray(ref_scores, dtype=np.float64)
    ref_labels = np.asarray(ref_labels, dtype=np.int64)
    valid = np.isfinite(ref_scores)
    if ref_mode == "benign_only":
        valid &= ref_labels == 0
    x = ref_scores[valid]
    if x.shape[0] <= 0:
        raise RuntimeError("No reference scores available for threshold estimation.")
    return float(np.quantile(x, float(q)))


def search_threshold(
    ref_scores: np.ndarray,
    ref_labels: np.ndarray,
    ref_timestamps_ns: np.ndarray,
    positive_mode: str,
    beta: float = 0.25,
    max_fp_per_day: float | None = None,
    num_quantiles: int = 256,
    min_quantile: float = 0.99,
    max_quantile: float = 0.99995,
) -> Dict[str, float]:
    scores = np.asarray(ref_scores, dtype=np.float64)
    labels = np.asarray(ref_labels, dtype=np.int64)
    timestamps_ns = np.asarray(ref_timestamps_ns, dtype=np.int64)
    valid = np.isfinite(scores)
    benign_mask = valid & (labels == 0)
    if positive_mode == "malicious_only":
        positive_mask = valid & (labels == 2)
    else:
        positive_mask = valid & (labels > 0)
    if benign_mask.sum() <= 0 or positive_mask.sum() <= 0:
        raise RuntimeError("Need both benign and positive reference events for threshold search.")

    benign_scores = scores[benign_mask]
    quantiles = np.linspace(float(min_quantile), float(max_quantile), int(num_quantiles))
    candidates = np.unique(np.quantile(benign_scores, quantiles))

    best: Dict[str, float] | None = None
    for threshold in candidates.tolist():
        alerts = valid & (scores >= float(threshold))
        fp_day = false_positives_per_day_from_mask(timestamps_ns=timestamps_ns, fp_mask=benign_mask & alerts)
        if max_fp_per_day is not None and fp_day > float(max_fp_per_day):
            continue
        tp = int(np.sum(positive_mask & alerts))
        fp = int(np.sum(benign_mask & alerts))
        fn = int(np.sum(positive_mask & (~alerts)))
        score = fbeta_score(tp=tp, fp=fp, fn=fn, beta=float(beta))
        candidate = {
            "threshold": float(threshold),
            "fbeta": float(score),
            "fp_per_day": float(fp_day),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "recall": float(tp / max(int(positive_mask.sum()), 1)),
            "precision": float(tp / max(int(tp + fp), 1)),
        }
        if best is None:
            best = candidate
            continue
        if candidate["fbeta"] > best["fbeta"] + 1e-12:
            best = candidate
            continue
        if abs(candidate["fbeta"] - best["fbeta"]) <= 1e-12 and candidate["fp_per_day"] < best["fp_per_day"]:
            best = candidate

    if best is None:
        fallback_threshold = threshold_from_reference_scores(ref_scores=scores, ref_labels=labels, q=0.999, ref_mode="benign_only")
        alerts = valid & (scores >= fallback_threshold)
        tp = int(np.sum(positive_mask & alerts))
        fp = int(np.sum(benign_mask & alerts))
        fn = int(np.sum(positive_mask & (~alerts)))
        best = {
            "threshold": float(fallback_threshold),
            "fbeta": float(fbeta_score(tp=tp, fp=fp, fn=fn, beta=float(beta))),
            "fp_per_day": float(false_positives_per_day_from_mask(timestamps_ns=timestamps_ns, fp_mask=benign_mask & alerts)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "recall": float(tp / max(int(positive_mask.sum()), 1)),
            "precision": float(tp / max(int(tp + fp), 1)),
        }
    return best


def small_grid_search(
    components: Dict[str, np.ndarray],
    labels: np.ndarray,
    timestamps_ns: np.ndarray,
    positive_mode: str,
    target_fp_per_day: float,
    beta: float,
    candidate_weight_map: Iterable[Dict[str, float]],
    threshold_num_quantiles: int = 256,
    threshold_min_quantile: float = 0.99,
    threshold_max_quantile: float = 0.99995,
) -> Dict[str, object]:
    best: Dict[str, object] | None = None
    labels = np.asarray(labels, dtype=np.int64)
    timestamps_ns = np.asarray(timestamps_ns, dtype=np.int64)
    for weight_map in candidate_weight_map:
        score = np.zeros_like(np.asarray(next(iter(components.values())), dtype=np.float64), dtype=np.float64)
        for key, weight in weight_map.items():
            score += float(weight) * np.asarray(components[key], dtype=np.float64)
        result = search_threshold(
            ref_scores=score,
            ref_labels=labels,
            ref_timestamps_ns=timestamps_ns,
            positive_mode=positive_mode,
            beta=float(beta),
            max_fp_per_day=float(target_fp_per_day),
            num_quantiles=int(threshold_num_quantiles),
            min_quantile=float(threshold_min_quantile),
            max_quantile=float(threshold_max_quantile),
        )
        candidate = {
            "weight_map": {key: float(value) for key, value in weight_map.items()},
            "threshold": float(result["threshold"]),
            "fbeta": float(result["fbeta"]),
            "fp_per_day": float(result["fp_per_day"]),
            "recall": float(result["recall"]),
            "precision": float(result["precision"]),
        }
        if best is None:
            best = candidate
            continue
        if candidate["fbeta"] > float(best["fbeta"]) + 1e-12:
            best = candidate
            continue
        if abs(candidate["fbeta"] - float(best["fbeta"])) <= 1e-12 and candidate["fp_per_day"] < float(best["fp_per_day"]):
            best = candidate
    if best is None:
        raise RuntimeError("Grid search received no candidates.")
    return best
