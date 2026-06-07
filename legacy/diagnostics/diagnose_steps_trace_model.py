#!/usr/bin/env python3
"""Small post-hoc STEPS separability diagnostic on existing score traces.

Purpose:
    Decide whether the current typed evidence vectors are learnable enough for
    a compact STEPS-style streaming state model, before implementing a new
    online model or running full E3.

Inputs:
    Existing causal-semantics result directories with score traces:
      - online_event_score_trace.csv
      - online_node_score_trace.csv
      - online_episode_score_trace.csv

Outputs:
    JSON and Markdown reports comparing baseline `online_risk` ordering against
    tiny logistic models trained on different typed-evidence subsets.

Pipeline status:
    Diagnostic only. This script is not imported by the detector. It does not
    change thresholds, models, alerts, or rankings.

Leakage risk:
    This script uses labels after inference to test separability. Results are
    post-hoc diagnostics, not deployable performance. A production STEPS method
    must learn from train/validation only and then rerun streaming inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TRUTHY = {"1", "true", "t", "yes", "y"}


@dataclass(frozen=True)
class LayerSpec:
    name: str
    filename: str
    tp_fields: tuple[str, ...]
    id_field: str | None = None


LAYER_SPECS = (
    LayerSpec("event", "online_event_score_trace.csv", ("is_correct_event_alert",), None),
    LayerSpec("node", "online_node_score_trace.csv", ("is_tp_after_attack_start", "is_tp_anytime", "is_gt_positive"), "node_id"),
    LayerSpec("episode", "online_episode_score_trace.csv", ("is_tp_after_attack_start", "is_tp_any_endpoint"), None),
)


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "SEM_ONLY": ("semantic_surprise",),
    "SEM_RES": ("semantic_surprise", "residual_fusion"),
    "CORE_NO_CHAIN": ("semantic_surprise", "residual_fusion", "edge_novelty"),
    "CORE_WITH_CHAIN": ("semantic_surprise", "residual_fusion", "edge_novelty", "chain_diversity"),
    "CORE_WITH_TRANSITION": (
        "semantic_surprise",
        "residual_fusion",
        "edge_novelty",
        "chain_diversity",
        "transition_count",
    ),
    "ALL_TYPED": (
        "semantic_surprise",
        "residual_fusion",
        "edge_novelty",
        "burst_score",
        "episode_support",
        "chain_diversity",
        "transition_count",
        "peer_count",
    ),
    "WEAK_ONLY": ("burst_score", "episode_support"),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", action="append", default=[])
    p.add_argument("--budgets", default="500,1000,3000,5000")
    p.add_argument("--layers", default="event,node,episode")
    p.add_argument("--max_iter", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--l2", type=float, default=1e-2)
    p.add_argument("--out_json", default="outputs/diagnostics/steps_trace_model_diagnostic_e3_1p2m.json")
    p.add_argument("--out_md", default="outputs/diagnostics/steps_trace_model_diagnostic_e3_1p2m.md")
    p.add_argument("--print_summary", action="store_true")
    return p.parse_args()


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in TRUTHY


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _dataset_from_dir(path: Path) -> str:
    for dataset in ("CLEARSCOPE_E3", "CADETS_E3", "THEIA_E3"):
        if path.name.startswith(dataset):
            return dataset
    return path.name.split("_")[0]


def _row_label(row: dict[str, str], spec: LayerSpec) -> int:
    if spec.name == "event":
        if "is_correct_event_alert" in row:
            return int(_truthy(row.get("is_correct_event_alert")))
        return int(str(row.get("event_label", "")).strip() in {"1", "2"})
    return int(any(_truthy(row.get(field)) for field in spec.tp_fields if field in row))


def _feature(row: dict[str, str], name: str) -> float:
    if name == "semantic_surprise":
        return _safe_float(row.get("event_causal_semantic_score"))
    if name == "residual_fusion":
        return _safe_float(row.get("event_residual_fusion_score"))
    if name == "edge_novelty":
        return _safe_float(row.get("event_edge_novelty"))
    if name == "burst_score":
        return _safe_float(row.get("event_temporal_burst"))
    if name == "episode_support":
        return _safe_float(row.get("event_episode_support"))
    if name == "chain_diversity":
        return _safe_float(row.get("event_chain_diversity"))
    if name == "transition_count":
        return math.log1p(max(_safe_int(row.get("chain_transition_count"), 0) - 1, 0))
    if name == "peer_count":
        return math.log1p(max(_safe_int(row.get("episode_recent_peer_count"), 0), 0))
    raise KeyError(name)


def _matrix(rows: list[dict[str, str]], feature_names: tuple[str, ...]) -> np.ndarray:
    x = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_names):
            x[i, j] = np.float32(_feature(row, name))
    return x


def _labels(rows: list[dict[str, str]], spec: LayerSpec) -> np.ndarray:
    return np.asarray([_row_label(row, spec) for row in rows], dtype=np.float32)


def _baseline_scores(rows: list[dict[str, str]]) -> np.ndarray:
    return np.asarray([_safe_float(row.get("online_risk")) for row in rows], dtype=np.float32)


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(x, axis=0).astype(np.float32)
    scale = np.std(x, axis=0).astype(np.float32)
    scale = np.where(scale < 1e-4, 1.0, scale).astype(np.float32)
    return ((x - center) / scale).astype(np.float32), center, scale


def _standardize_apply(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((x - center) / scale).astype(np.float32)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def _fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int,
    lr: float,
    l2: float,
) -> dict[str, Any] | None:
    pos = int(np.sum(y > 0.5))
    neg = int(len(y) - pos)
    if pos <= 0 or neg <= 0:
        return None
    xz, center, scale = _standardize_fit(x)
    w = np.zeros((xz.shape[1],), dtype=np.float32)
    b = np.float32(0.0)
    weights = np.where(y > 0.5, len(y) / max(2.0 * pos, 1.0), len(y) / max(2.0 * neg, 1.0)).astype(np.float32)
    for _ in range(max(int(max_iter), 1)):
        pred = _sigmoid(xz @ w + b)
        err = (pred - y) * weights
        grad_w = (xz.T @ err) / float(len(y)) + np.float32(l2) * w
        grad_b = np.float32(np.mean(err))
        w -= np.float32(lr) * grad_w.astype(np.float32)
        b -= np.float32(lr) * grad_b
    return {"w": w, "b": float(b), "center": center, "scale": scale}


def _predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    xz = _standardize_apply(x, model["center"], model["scale"])
    return _sigmoid(xz @ model["w"] + np.float32(model["b"]))


def _average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y[order]
    pos = float(np.sum(y_sorted > 0.5))
    if pos <= 0.0:
        return 0.0
    tp = 0.0
    precision_sum = 0.0
    for i, label in enumerate(y_sorted, 1):
        if label > 0.5:
            tp += 1.0
            precision_sum += tp / float(i)
    return float(precision_sum / pos)


def _roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    pos = int(np.sum(y > 0.5))
    neg = int(len(y) - pos)
    if pos <= 0 or neg <= 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = float(np.sum(ranks[y > 0.5]))
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / float(pos * neg))


def _budget_metrics(
    rows: list[dict[str, str]],
    spec: LayerSpec,
    y: np.ndarray,
    scores: np.ndarray,
    budgets: list[int],
) -> list[dict[str, Any]]:
    order = np.argsort(-scores, kind="mergesort")
    out: list[dict[str, Any]] = []
    for budget in budgets:
        idx = order[: min(int(budget), len(order))]
        tp = int(np.sum(y[idx] > 0.5))
        selected = int(len(idx))
        row: dict[str, Any] = {
            "budget": int(budget),
            "selected": selected,
            "tp_rows": tp,
            "fp_rows": int(selected - tp),
            "precision_rows": float(tp / selected) if selected else 0.0,
        }
        if spec.id_field:
            unique_tp: set[int] = set()
            unique_nodes: set[int] = set()
            for i in idx:
                node = _safe_int(rows[int(i)].get(spec.id_field), -1)
                if node < 0:
                    continue
                unique_nodes.add(node)
                if y[int(i)] > 0.5:
                    unique_tp.add(node)
            row["unique_nodes"] = int(len(unique_nodes))
            row["unique_tp_nodes"] = int(len(unique_tp))
        out.append(row)
    return out


def _evaluate(
    rows: list[dict[str, str]],
    spec: LayerSpec,
    y: np.ndarray,
    scores: np.ndarray,
    budgets: list[int],
) -> dict[str, Any]:
    return {
        "rows": int(len(rows)),
        "positives": int(np.sum(y > 0.5)),
        "unique_positive_nodes": int(
            len({_safe_int(row.get(spec.id_field), -1) for row in rows if spec.id_field and _row_label(row, spec) and _safe_int(row.get(spec.id_field), -1) >= 0})
        )
        if spec.id_field
        else None,
        "auroc": _roc_auc(y, scores),
        "auprc": _average_precision(y, scores),
        "budgets": _budget_metrics(rows, spec, y, scores, budgets),
    }


def _load_dataset_layers(result_dirs: list[Path], layers: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for result_dir in result_dirs:
        dataset = _dataset_from_dir(result_dir)
        out[dataset] = {}
        for spec in LAYER_SPECS:
            if spec.name not in layers:
                continue
            rows = _read_csv(result_dir / spec.filename)
            if not rows:
                continue
            out[dataset][spec.name] = {"rows": rows, "spec": spec, "labels": _labels(rows, spec)}
    return out


def _concat_train(
    loaded: dict[str, dict[str, Any]],
    datasets: list[str],
    layer: str,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for dataset in datasets:
        layer_data = loaded.get(dataset, {}).get(layer)
        if not layer_data:
            continue
        xs.append(_matrix(layer_data["rows"], feature_names))
        ys.append(layer_data["labels"])
    if not xs:
        return None
    return np.vstack(xs), np.concatenate(ys)


def diagnose(loaded: dict[str, dict[str, Any]], budgets: list[int], max_iter: int, lr: float, l2: float) -> dict[str, Any]:
    datasets = sorted(loaded)
    out: dict[str, Any] = {"datasets": {}, "transfers": [], "leave_one_dataset_out": []}
    for dataset in datasets:
        out["datasets"][dataset] = {}
        for layer, layer_data in loaded[dataset].items():
            rows = layer_data["rows"]
            spec = layer_data["spec"]
            y = layer_data["labels"]
            baseline = _evaluate(rows, spec, y, _baseline_scores(rows), budgets)
            out["datasets"][dataset][layer] = {"baseline_online_risk": baseline}

    for layer in sorted({layer for data in loaded.values() for layer in data}):
        for train_dataset in datasets:
            for test_dataset in datasets:
                if train_dataset == test_dataset:
                    continue
                if layer not in loaded.get(train_dataset, {}) or layer not in loaded.get(test_dataset, {}):
                    continue
                for set_name, feature_names in FEATURE_SETS.items():
                    train = _concat_train(loaded, [train_dataset], layer, feature_names)
                    if train is None:
                        continue
                    train_x, train_y = train
                    model = _fit_logistic(train_x, train_y, max_iter=max_iter, lr=lr, l2=l2)
                    if model is None:
                        continue
                    test_data = loaded[test_dataset][layer]
                    test_x = _matrix(test_data["rows"], feature_names)
                    scores = _predict(model, test_x)
                    out["transfers"].append(
                        {
                            "layer": layer,
                            "train_dataset": train_dataset,
                            "test_dataset": test_dataset,
                            "feature_set": set_name,
                            "features": list(feature_names),
                            "model_size_bytes": int(model["w"].nbytes + model["center"].nbytes + model["scale"].nbytes + 4),
                            "metrics": _evaluate(test_data["rows"], test_data["spec"], test_data["labels"], scores, budgets),
                            "weights": {name: float(model["w"][i]) for i, name in enumerate(feature_names)},
                        }
                    )

        for test_dataset in datasets:
            train_datasets = [dataset for dataset in datasets if dataset != test_dataset]
            if layer not in loaded.get(test_dataset, {}):
                continue
            for set_name, feature_names in FEATURE_SETS.items():
                train = _concat_train(loaded, train_datasets, layer, feature_names)
                if train is None:
                    continue
                train_x, train_y = train
                model = _fit_logistic(train_x, train_y, max_iter=max_iter, lr=lr, l2=l2)
                if model is None:
                    continue
                test_data = loaded[test_dataset][layer]
                scores = _predict(model, _matrix(test_data["rows"], feature_names))
                out["leave_one_dataset_out"].append(
                    {
                        "layer": layer,
                        "train_datasets": train_datasets,
                        "test_dataset": test_dataset,
                        "feature_set": set_name,
                        "features": list(feature_names),
                        "model_size_bytes": int(model["w"].nbytes + model["center"].nbytes + model["scale"].nbytes + 4),
                        "metrics": _evaluate(test_data["rows"], test_data["spec"], test_data["labels"], scores, budgets),
                        "weights": {name: float(model["w"][i]) for i, name in enumerate(feature_names)},
                    }
                )
    return out


def _budget_lookup(metrics: dict[str, Any], budget: int) -> dict[str, Any]:
    return next((item for item in metrics.get("budgets", []) if int(item.get("budget", -1)) == int(budget)), {})


def _best_lodo_by_target(report: dict[str, Any], layer: str, budget: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in report.get("leave_one_dataset_out", []):
        if row.get("layer") != layer:
            continue
        metrics = row.get("metrics", {})
        item = _budget_lookup(metrics, budget)
        out.append(
            {
                "test_dataset": row["test_dataset"],
                "feature_set": row["feature_set"],
                "tp_rows": int(item.get("tp_rows", 0)),
                "fp_rows": int(item.get("fp_rows", 0)),
                "precision_rows": float(item.get("precision_rows", 0.0)),
                "auprc": float(metrics.get("auprc", 0.0)),
                "auroc": float(metrics.get("auroc", 0.0)),
                "weights": row.get("weights", {}),
            }
        )
    out.sort(key=lambda item: (item["test_dataset"], -item["tp_rows"], -item["precision_rows"], -item["auprc"]))
    best: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in out:
        if item["test_dataset"] in seen:
            continue
        best.append(item)
        seen.add(item["test_dataset"])
    return best


def _write_md(path: Path, report: dict[str, Any], budgets: list[int]) -> None:
    lines: list[str] = []
    primary_budget = 1000 if 1000 in budgets else budgets[0]
    lines.append("# STEPS Trace-Tail Separability Diagnostic")
    lines.append("")
    lines.append("Diagnostic only. This uses post-inference labels to test whether typed evidence is learnable; it is not deployable performance.")
    lines.append("")
    lines.append("## Baseline Online Risk")
    lines.append("")
    lines.append("| Dataset | Layer | Pos Rows | Unique Pos Nodes | Top1000 TP Rows | Top1000 FP Rows | Top1000 Precision | AUPRC | AUROC |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, layer_data in sorted(report["datasets"].items()):
        for layer, payload in sorted(layer_data.items()):
            metrics = payload["baseline_online_risk"]
            item = _budget_lookup(metrics, primary_budget)
            lines.append(
                f"| {dataset} | {layer} | {metrics['positives']} | {metrics.get('unique_positive_nodes') or ''} | {item.get('tp_rows', 0)} | {item.get('fp_rows', 0)} | {item.get('precision_rows', 0.0):.6g} | {metrics['auprc']:.6g} | {metrics['auroc']:.6g} |"
            )
    lines.append("")
    lines.append("## Leave-One-Dataset-Out Best Feature Set")
    lines.append("")
    lines.append("Train on the other datasets' trace labels, test on the held-out dataset trace. This is still post-hoc, but checks transferability better than fitting and testing on the same trace.")
    lines.append("")
    for layer in ("event", "node", "episode"):
        best = _best_lodo_by_target(report, layer, primary_budget)
        if not best:
            continue
        lines.append(f"### {layer.title()}")
        lines.append("")
        lines.append("| Test Dataset | Best Feature Set | Top1000 TP Rows | Top1000 FP Rows | Top1000 Precision | AUPRC | AUROC | Top Weights |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for item in best:
            weights = sorted(item["weights"].items(), key=lambda kv: abs(kv[1]), reverse=True)
            top_weights = ", ".join(f"{name}={value:.3g}" for name, value in weights[:4])
            lines.append(
                f"| {item['test_dataset']} | {item['feature_set']} | {item['tp_rows']} | {item['fp_rows']} | {item['precision_rows']:.6g} | {item['auprc']:.6g} | {item['auroc']:.6g} | {top_weights} |"
            )
        lines.append("")
    lines.append("## Evidence Reduction Reading")
    lines.append("")
    lines.append("- Keep as candidate typed inputs only if they improve transfer or are needed by ablation: `semantic_surprise`, `residual_fusion`, `edge_novelty`, and `chain_diversity`.")
    lines.append("- Treat `burst_score` and `episode_support` as weak until they beat core subsets; they are not automatically trusted.")
    lines.append("- `transition_count` is only a proxy for transition surprise because no standalone transition NLL exists in the active trace.")
    lines.append("- `node_state_delta` and `repeated_window_risk` are not full trace features here; they need targeted online-policy and confirmation ablations.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    budgets = [int(x.strip()) for x in str(args.budgets).split(",") if x.strip()]
    layers = {x.strip() for x in str(args.layers).split(",") if x.strip()}
    result_dirs = [Path(x) for x in args.result_dir]
    if not result_dirs:
        result_dirs = [
            Path("outputs/results/cs4m/CLEARSCOPE_E3_CAUSAL_SEMANTICS_BUDGET1000_1P2M"),
            Path("outputs/results/cs4m/CADETS_E3_CAUSAL_SEMANTICS_BUDGET1000_1P2M"),
            Path("outputs/results/cs4m/THEIA_E3_CAUSAL_SEMANTICS_BUDGET1000_1P2M"),
        ]
    loaded = _load_dataset_layers(result_dirs, layers)
    report = {
        "purpose": "post-hoc STEPS trace-tail separability diagnostic",
        "leakage_check": "uses labels after inference only; not deployable performance and not used for threshold/model selection",
        "budgets": budgets,
        "result_dirs": [str(x) for x in result_dirs],
        "diagnostic": diagnose(loaded, budgets, int(args.max_iter), float(args.lr), float(args.l2)),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True))
    _write_md(Path(args.out_md), report["diagnostic"], budgets)
    if args.print_summary:
        print(f"Wrote {out_json}")
        print(f"Wrote {args.out_md}")
        for layer in ("event", "node", "episode"):
            for item in _best_lodo_by_target(report["diagnostic"], layer, 1000):
                print(
                    f"{layer} {item['test_dataset']}: best={item['feature_set']} "
                    f"top1000_tp={item['tp_rows']} precision={item['precision_rows']:.4f}"
                )


if __name__ == "__main__":
    main()
