#!/usr/bin/env python3
"""Audit causal-semantics score traces and evidence ablations.

Purpose:
    Post-inference diagnostic for deciding whether the current evidence space
    is worth replacing with a small learned STEPS model, or whether the project
    must change semantic/graph representation first.

Inputs:
    One or more result directories containing optional top-risk trace CSVs:
      - online_event_score_trace.csv
      - online_node_score_trace.csv
      - online_episode_score_trace.csv
    If traces are absent, the script falls back to already emitted alert CSVs.
    Optionally, an ablation root containing:
      - {DATASET}_{MODE}_CAUSAL_SEMANTICS_ABLATION_1P2M/eval_causal_semantics.json

Outputs:
    JSON and Markdown summaries with:
      - event/node/episode TP reachability in top-risk tails;
      - TP vs FP evidence distributions;
      - top-k precision when each evidence component is used alone;
      - historical ablation deltas for TP-driving and FP-reducing evidence;
      - a diagnostic recommendation.

Pipeline status:
    Diagnostic only. It is not imported by the detector and does not change
    thresholds, scores, alerts, rankings, or training.

How to test:
    python -m py_compile scripts/tools/audit_score_trace_evidence.py
    python scripts/tools/audit_score_trace_evidence.py \
      --result_dir outputs/results/cs4m/CLEARSCOPE_E3_CAUSAL_SEMANTICS_BUDGET1000_1P2M \
      --ablation_root outputs/archive_20260512_redeploy/non_deploy_diagnostics

Runtime/memory:
    Reads bounded trace/alert CSVs and small eval JSONs. Default active traces
    are top-5000 rows per layer, so runtime is seconds and memory is small.

Leakage risk:
    Uses labels only after inference for audit reporting. Do not use this
    script to tune test thresholds. Any resulting model or threshold must be
    selected on train/validation only and re-evaluated in a fresh run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


TRUTHY = {"1", "true", "t", "yes", "y"}
ABLATION_MODES = (
    "SEM_ONLY",
    "RES_ONLY",
    "SEM_RES",
    "EDGE_ONLY",
    "EDGE_BURST",
    "EDGE_EPISODE",
    "EDGE_CHAIN",
    "FULL_EVIDENCE",
    "FULL_CONFIRM",
)


@dataclass(frozen=True)
class LayerSpec:
    name: str
    trace_filename: str
    fallback_filename: str
    score_fields: tuple[str, ...]
    tp_fields: tuple[str, ...]
    fp_fields: tuple[str, ...]
    relaxed_fields: tuple[str, ...] = ()


LAYER_SPECS = (
    LayerSpec(
        name="event",
        trace_filename="online_event_score_trace.csv",
        fallback_filename="online_event_alerts.csv",
        score_fields=("online_risk", "event_score", "score"),
        tp_fields=("is_correct_event_alert",),
        fp_fields=(),
        relaxed_fields=("is_suspicious_event", "is_malicious_event"),
    ),
    LayerSpec(
        name="node",
        trace_filename="online_node_score_trace.csv",
        fallback_filename="online_node_alerts.csv",
        score_fields=("online_risk", "event_score", "score"),
        tp_fields=("is_tp_after_attack_start", "is_tp_anytime", "is_gt_positive"),
        fp_fields=("is_fp_strict",),
        relaxed_fields=("is_suspect",),
    ),
    LayerSpec(
        name="episode",
        trace_filename="online_episode_score_trace.csv",
        fallback_filename="online_episode_alerts.csv",
        score_fields=("online_risk", "event_score", "score"),
        tp_fields=("is_tp_after_attack_start", "is_tp_any_endpoint"),
        fp_fields=("is_fp_strict",),
        relaxed_fields=("src_is_suspect", "dst_is_suspect"),
    ),
)


EVIDENCE_FIELDS = {
    "semantic_surprise": ("event_causal_semantic_score", "causal_semantic_score"),
    "transition_surprise": ("max_field_rarity_proxy",),
    "residual_fusion": ("event_residual_fusion_score", "residual_fusion_score"),
    "edge_novelty": ("event_edge_novelty",),
    "burst_score": ("event_temporal_burst",),
    "episode_support": ("event_episode_support",),
    "chain_diversity": ("event_chain_diversity",),
    "node_state_delta": ("risk_gain", "online_risk_delta"),
    "repeated_window_risk": ("confirm_score", "compact_consistency_component", "event_count"),
    "online_risk": ("online_risk",),
}

COMPONENT_SOURCE_NOTES = {
    "semantic_surprise": "train-only role/field rarity/NLL proxy from causal semantic scoring",
    "transition_surprise": "not stored as a standalone numeric score; proxied by max_field and chain_transition_count when available",
    "residual_fusion": "train-only low-rank residual fusion score, validation calibrated",
    "edge_novelty": "current causal edge unseen in past online state and event score above validation tail",
    "burst_score": "decayed repeated edge-novelty accumulation for the node",
    "episode_support": "short-window peer/causal-neighborhood support gated by novelty",
    "chain_diversity": "short-window diversity of peer kind, action family, and transition signatures",
    "node_state_delta": "stateful node alert risk increase over previous node alert; absent in older traces",
    "repeated_window_risk": "bounded-delay confirmation consistency over repeated events/roles/peers/actions",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", action="append", default=[], help="Causal-semantics result directory.")
    p.add_argument(
        "--ablation_root",
        action="append",
        default=[],
        help="Root containing ablation eval JSONs. Can be passed multiple times.",
    )
    p.add_argument("--budgets", default="500,1000,3000,5000")
    p.add_argument("--out_json", default="outputs/diagnostics/causal_semantics_score_trace_audit_e3_1p2m.json")
    p.add_argument("--out_md", default="outputs/diagnostics/causal_semantics_score_trace_audit_e3_1p2m.md")
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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _dataset_from_result_dir(result_dir: Path) -> str:
    for dataset in ("CLEARSCOPE_E3", "CADETS_E3", "THEIA_E3"):
        if result_dir.name.startswith(dataset):
            return dataset
    return result_dir.name.split("_")[0]


def _first_existing(row: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        if field in row:
            return field
    return None


def _is_tp(row: dict[str, str], spec: LayerSpec) -> bool:
    if spec.name == "event":
        if "is_correct_event_alert" in row:
            return _truthy(row.get("is_correct_event_alert"))
        label = str(row.get("event_label", row.get("label", ""))).strip()
        return label in {"1", "2"}
    return any(_truthy(row.get(field)) for field in spec.tp_fields if field in row)


def _is_relaxed(row: dict[str, str], spec: LayerSpec) -> bool:
    return any(_truthy(row.get(field)) for field in spec.relaxed_fields if field in row)


def _sort_by_score(rows: list[dict[str, str]], spec: LayerSpec) -> tuple[list[dict[str, str]], str | None]:
    if not rows:
        return [], None
    score_field = _first_existing(rows[0], spec.score_fields)
    if score_field is None:
        return rows, None
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda item: (_safe_float(item[1].get(score_field)), -item[0]), reverse=True)
    return [row for _idx, row in indexed], score_field


def _field_value(row: dict[str, str], component: str) -> float | None:
    if component == "transition_surprise":
        if "chain_transition_count" in row:
            return float(max(_safe_int(row.get("chain_transition_count")) - 1, 0))
        return None
    for field in EVIDENCE_FIELDS.get(component, ()):
        if field in row:
            return _safe_float(row.get(field))
    return None


def _num_summary(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {
            "count": 0,
            "nonzero": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    clean.sort()
    p90_idx = min(max(int(math.ceil(0.90 * len(clean))) - 1, 0), len(clean) - 1)
    return {
        "count": int(len(clean)),
        "nonzero": int(sum(1 for v in clean if abs(v) > 1e-12)),
        "mean": float(mean(clean)),
        "median": float(median(clean)),
        "p90": float(clean[p90_idx]),
        "max": float(clean[-1]),
    }


def _budget_tail(rows: list[dict[str, str]], spec: LayerSpec, budgets: list[int]) -> dict[str, Any]:
    sorted_rows, score_field = _sort_by_score(rows, spec)
    unique_tp_nodes_total: set[int] = set()
    if spec.name == "node":
        unique_tp_nodes_total = {
            _safe_int(row.get("node_id"), -1)
            for row in sorted_rows
            if _is_tp(row, spec) and _safe_int(row.get("node_id"), -1) >= 0
        }
    out: dict[str, Any] = {
        "rows": int(len(sorted_rows)),
        "score_field": score_field,
        "tp_total": int(sum(1 for row in sorted_rows if _is_tp(row, spec))),
        "tp_total_semantics": "row_count",
        "unique_tp_nodes_total": int(len(unique_tp_nodes_total)) if spec.name == "node" else None,
        "first_tp_rank": next((idx for idx, row in enumerate(sorted_rows, 1) if _is_tp(row, spec)), None),
        "budgets": [],
    }
    for budget in budgets:
        selected = sorted_rows[: min(int(budget), len(sorted_rows))]
        tp = sum(1 for row in selected if _is_tp(row, spec))
        relaxed = sum(1 for row in selected if _is_relaxed(row, spec))
        fp = len(selected) - tp
        unique_tp_nodes: set[int] = set()
        if spec.name == "node":
            unique_tp_nodes = {
                _safe_int(row.get("node_id"), -1)
                for row in selected
                if _is_tp(row, spec) and _safe_int(row.get("node_id"), -1) >= 0
            }
        threshold = _safe_float(selected[-1].get(score_field), float("nan")) if selected and score_field else None
        out["budgets"].append(
            {
                "budget": int(budget),
                "selected": int(len(selected)),
                "exact": bool(len(sorted_rows) >= int(budget)),
                "tp": int(tp),
                "tp_semantics": "row_count",
                "unique_tp_nodes": int(len(unique_tp_nodes)) if spec.name == "node" else None,
                "fp": int(fp),
                "relaxed_positive": int(relaxed),
                "precision": float(tp / len(selected)) if selected else 0.0,
                "threshold_or_floor": threshold,
            }
        )
    return out


def _component_audit(rows: list[dict[str, str]], spec: LayerSpec, budgets: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for component in EVIDENCE_FIELDS:
        component_rows: list[tuple[float, dict[str, str]]] = []
        tp_values: list[float] = []
        fp_values: list[float] = []
        for row in rows:
            value = _field_value(row, component)
            if value is None:
                continue
            component_rows.append((float(value), row))
            if _is_tp(row, spec):
                tp_values.append(float(value))
            else:
                fp_values.append(float(value))
        component_rows.sort(key=lambda item: item[0], reverse=True)
        topk: list[dict[str, Any]] = []
        for budget in budgets:
            selected = [row for _value, row in component_rows[: min(int(budget), len(component_rows))]]
            tp = sum(1 for row in selected if _is_tp(row, spec))
            fp = len(selected) - tp
            topk.append(
                {
                    "budget": int(budget),
                    "selected": int(len(selected)),
                    "exact": bool(len(component_rows) >= int(budget)),
                    "tp": int(tp),
                    "fp": int(fp),
                    "precision": float(tp / len(selected)) if selected else 0.0,
                }
            )
        tp_summary = _num_summary(tp_values)
        fp_summary = _num_summary(fp_values)
        tp_mean = tp_summary["mean"]
        fp_mean = fp_summary["mean"]
        if tp_mean is None or fp_mean is None:
            mean_gap = None
            mean_ratio = None
        else:
            mean_gap = float(tp_mean - fp_mean)
            mean_ratio = float((tp_mean + 1e-9) / (fp_mean + 1e-9))
        out[component] = {
            "available_rows": int(len(component_rows)),
            "source_note": COMPONENT_SOURCE_NOTES.get(component, ""),
            "tp_distribution": tp_summary,
            "fp_distribution": fp_summary,
            "tp_minus_fp_mean": mean_gap,
            "tp_to_fp_mean_ratio": mean_ratio,
            "top_by_component": topk,
        }
    return out


def _metric_block(metrics: dict[str, Any]) -> dict[str, Any]:
    event = metrics.get("event_alerts", {})
    node = metrics.get("node_alerts", {})
    confirmed_event = metrics.get("confirmed_event_alerts", {})
    confirmed_node = metrics.get("confirmed_node_alerts", {})
    return {
        "event_tp": _safe_int(event.get("correct_event_alerts")),
        "event_alerts": _safe_int(event.get("num_alerts")),
        "event_fp": _safe_int(event.get("false_event_alerts"), _safe_int(event.get("num_alerts")) - _safe_int(event.get("correct_event_alerts"))),
        "event_precision": _safe_float(event.get("correct_alert_ratio")),
        "node_tp": _safe_int(node.get("tp_after_attack_start")),
        "node_alerts": _safe_int(node.get("num_alerts")),
        "node_fp_strict": _safe_int(node.get("fp_strict"), _safe_int(node.get("num_alerts")) - _safe_int(node.get("tp_after_attack_start"))),
        "node_precision_strict": _safe_float(node.get("precision_after_attack_strict")),
        "confirmed_event_tp": _safe_int(confirmed_event.get("correct_event_alerts")),
        "confirmed_event_alerts": _safe_int(confirmed_event.get("num_alerts")),
        "confirmed_event_fp": _safe_int(confirmed_event.get("false_event_alerts"), _safe_int(confirmed_event.get("num_alerts")) - _safe_int(confirmed_event.get("correct_event_alerts"))),
        "confirmed_event_precision": _safe_float(confirmed_event.get("correct_alert_ratio")),
        "confirmed_node_tp": _safe_int(confirmed_node.get("tp_after_attack_start")),
        "confirmed_node_alerts": _safe_int(confirmed_node.get("num_alerts")),
        "confirmed_node_fp_strict": _safe_int(confirmed_node.get("fp_strict"), _safe_int(confirmed_node.get("num_alerts")) - _safe_int(confirmed_node.get("tp_after_attack_start"))),
        "confirmed_node_precision_strict": _safe_float(confirmed_node.get("precision_after_attack_strict")),
    }


def _find_ablation_eval(dataset: str, mode: str, ablation_roots: list[Path]) -> Path:
    rel = Path(f"{dataset}_{mode}_CAUSAL_SEMANTICS_ABLATION_1P2M") / "eval_causal_semantics.json"
    for root in ablation_roots:
        path = root / rel
        if path.exists():
            return path
    return ablation_roots[0] / rel if ablation_roots else rel


def _load_ablation(dataset: str, ablation_roots: list[Path]) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in ABLATION_MODES:
        path = _find_ablation_eval(dataset, mode, ablation_roots)
        payload = _read_json(path)
        if not payload:
            modes[mode] = {"missing": True, "path": str(path)}
            continue
        modes[mode] = {
            "missing": False,
            "path": str(path),
            "config": {
                "residual_fusion_enabled": payload.get("config", {}).get("residual_fusion_enabled"),
                "attack_evidence_edge_enabled": payload.get("config", {}).get("attack_evidence_edge_enabled"),
                "attack_evidence_burst_enabled": payload.get("config", {}).get("attack_evidence_burst_enabled"),
                "attack_evidence_episode_enabled": payload.get("config", {}).get("attack_evidence_episode_enabled"),
                "attack_evidence_chain_enabled": payload.get("config", {}).get("attack_evidence_chain_enabled"),
                "online_confirmation_enabled": payload.get("config", {}).get("online_confirmation_enabled"),
                "node_score_mode": payload.get("config", {}).get("node_score_mode"),
            },
            "metrics": _metric_block(payload.get("primary_online_metrics", {})),
        }

    def delta(a: str, b: str) -> dict[str, Any] | None:
        left = modes.get(a, {})
        right = modes.get(b, {})
        if left.get("missing") or right.get("missing"):
            return None
        lm = left["metrics"]
        rm = right["metrics"]
        fields = (
            "event_tp",
            "event_alerts",
            "event_fp",
            "node_tp",
            "node_alerts",
            "node_fp_strict",
            "confirmed_event_tp",
            "confirmed_event_alerts",
            "confirmed_event_fp",
            "confirmed_node_tp",
            "confirmed_node_alerts",
            "confirmed_node_fp_strict",
        )
        return {field: int(lm.get(field, 0)) - int(rm.get(field, 0)) for field in fields}

    evidence_deltas = {
        "residual_fusion_vs_semantic_only": delta("SEM_RES", "SEM_ONLY"),
        "edge_novelty_vs_sem_res": delta("EDGE_ONLY", "SEM_RES"),
        "burst_added_to_edge": delta("EDGE_BURST", "EDGE_ONLY"),
        "episode_added_to_edge": delta("EDGE_EPISODE", "EDGE_ONLY"),
        "chain_added_to_edge": delta("EDGE_CHAIN", "EDGE_ONLY"),
        "full_evidence_vs_edge_only": delta("FULL_EVIDENCE", "EDGE_ONLY"),
        "confirmation_vs_full_evidence": delta("FULL_CONFIRM", "FULL_EVIDENCE"),
    }
    confirmation_effect: dict[str, Any] | None = None
    full_confirm = modes.get("FULL_CONFIRM", {})
    full_evidence = modes.get("FULL_EVIDENCE", {})
    if not full_confirm.get("missing") and not full_evidence.get("missing"):
        cm = full_confirm.get("metrics", {})
        bm = full_evidence.get("metrics", {})
        confirmation_effect = {
            "confirmed_event_tp_delta_vs_immediate": int(cm.get("confirmed_event_tp", 0)) - int(bm.get("event_tp", 0)),
            "confirmed_event_fp_delta_vs_immediate": int(cm.get("confirmed_event_fp", 0)) - int(bm.get("event_fp", 0)),
            "confirmed_event_alert_delta_vs_immediate": int(cm.get("confirmed_event_alerts", 0)) - int(bm.get("event_alerts", 0)),
            "confirmed_node_tp_delta_vs_immediate": int(cm.get("confirmed_node_tp", 0)) - int(bm.get("node_tp", 0)),
            "confirmed_node_fp_delta_vs_immediate": int(cm.get("confirmed_node_fp_strict", 0)) - int(bm.get("node_fp_strict", 0)),
            "confirmed_node_alert_delta_vs_immediate": int(cm.get("confirmed_node_alerts", 0)) - int(bm.get("node_alerts", 0)),
            "baseline_full_evidence_immediate": {
                "event_tp": int(bm.get("event_tp", 0)),
                "event_fp": int(bm.get("event_fp", 0)),
                "event_alerts": int(bm.get("event_alerts", 0)),
                "node_tp": int(bm.get("node_tp", 0)),
                "node_fp_strict": int(bm.get("node_fp_strict", 0)),
                "node_alerts": int(bm.get("node_alerts", 0)),
            },
            "full_confirm_confirmed": {
                "event_tp": int(cm.get("confirmed_event_tp", 0)),
                "event_fp": int(cm.get("confirmed_event_fp", 0)),
                "event_alerts": int(cm.get("confirmed_event_alerts", 0)),
                "node_tp": int(cm.get("confirmed_node_tp", 0)),
                "node_fp_strict": int(cm.get("confirmed_node_fp_strict", 0)),
                "node_alerts": int(cm.get("confirmed_node_alerts", 0)),
            },
            "comparison_note": "FULL_CONFIRM confirmed alerts compared with FULL_EVIDENCE immediate alerts; this measures bounded-delay confirmation as an FP reducer.",
        }
    return {"modes": modes, "evidence_deltas": evidence_deltas, "confirmation_effect": confirmation_effect}


def _classify_roles(ablation: dict[str, Any]) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    checks = {
        "semantic_surprise": "SEM_ONLY",
        "residual_fusion": "residual_fusion_vs_semantic_only",
        "edge_novelty": "edge_novelty_vs_sem_res",
        "burst_score": "burst_added_to_edge",
        "episode_support": "episode_added_to_edge",
        "chain_diversity": "chain_added_to_edge",
        "repeated_window_risk": "confirmation_vs_full_evidence",
    }
    modes = ablation.get("modes", {})
    deltas = ablation.get("evidence_deltas", {})
    for component, source in checks.items():
        if source in modes:
            mode = modes.get(source, {})
            if mode.get("missing"):
                roles[component] = {"status": "unknown", "reason": f"{source} ablation missing"}
                continue
            metrics = mode.get("metrics", {})
            roles[component] = {
                "status": "baseline_signal" if metrics.get("event_tp", 0) > 0 or metrics.get("node_tp", 0) > 0 else "weak",
                "reason": f"{source}: event TP={metrics.get('event_tp', 0)}, node TP={metrics.get('node_tp', 0)}",
            }
            continue
        d = deltas.get(source)
        if component == "repeated_window_risk":
            effect = ablation.get("confirmation_effect")
            if not effect:
                roles[component] = {"status": "unknown", "reason": "confirmation effect unavailable"}
                continue
            event_tp_delta = int(effect.get("confirmed_event_tp_delta_vs_immediate", 0))
            node_tp_delta = int(effect.get("confirmed_node_tp_delta_vs_immediate", 0))
            event_fp_delta = int(effect.get("confirmed_event_fp_delta_vs_immediate", 0))
            node_fp_delta = int(effect.get("confirmed_node_fp_delta_vs_immediate", 0))
            if event_fp_delta < 0 or node_fp_delta < 0:
                status = "fp_reducer"
            elif event_tp_delta > 0 or node_tp_delta > 0:
                status = "tp_driver"
            elif event_tp_delta < 0 or node_tp_delta < 0:
                status = "tp_hurting_or_unstable"
            else:
                status = "neutral_or_redundant"
            roles[component] = {
                "status": status,
                "delta": effect,
                "reason": (
                    f"confirmed vs immediate: event TP delta={event_tp_delta}, event FP delta={event_fp_delta}, "
                    f"node TP delta={node_tp_delta}, node FP delta={node_fp_delta}"
                ),
            }
            continue
        d = deltas.get(source)
        if d is None:
            roles[component] = {"status": "unknown", "reason": f"{source} delta unavailable"}
            continue
        event_tp_delta = int(d.get("event_tp", 0))
        node_tp_delta = int(d.get("node_tp", 0))
        event_fp_delta = int(d.get("event_fp", 0))
        node_fp_delta = int(d.get("node_fp_strict", 0))
        confirmed_event_fp_delta = int(d.get("confirmed_event_fp", 0))
        if event_tp_delta > 0 or node_tp_delta > 0:
            status = "tp_driver"
        elif event_fp_delta < 0 or node_fp_delta < 0 or confirmed_event_fp_delta < 0:
            status = "fp_reducer"
        elif event_tp_delta < 0 or node_tp_delta < 0:
            status = "tp_hurting_or_unstable"
        else:
            status = "neutral_or_redundant"
        roles[component] = {
            "status": status,
            "delta": d,
            "reason": (
                f"event TP delta={event_tp_delta}, node TP delta={node_tp_delta}, "
                f"event FP delta={event_fp_delta}, node FP delta={node_fp_delta}"
            ),
        }
    roles["transition_surprise"] = {
        "status": "embedded_in_semantic_and_chain",
        "reason": "No standalone ablation exists; current outputs expose max_field and chain_transition_count but not a separate transition NLL.",
    }
    roles["node_state_delta"] = {
        "status": "needs_targeted_ablation",
        "reason": "Stateful node update fields exist in newer code, but the active 1.2M score traces do not isolate one_shot vs stateful_update.",
    }
    return roles


def _analyze_layers(result_dir: Path, budgets: list[int]) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for spec in LAYER_SPECS:
        trace_path = result_dir / spec.trace_filename
        fallback_path = result_dir / spec.fallback_filename
        if trace_path.exists():
            path = trace_path
            source = "score_trace_top_risk"
        else:
            path = fallback_path
            source = "alert_csv_fallback"
        rows = _read_csv(path)
        tail = _budget_tail(rows, spec, budgets)
        components = _component_audit(rows, spec, budgets)
        layers[spec.name] = {
            "source": source,
            "path": str(path),
            "tail": tail,
            "component_audit": components,
        }
    return layers


def _decision(dataset: str, layers: dict[str, Any], ablation: dict[str, Any]) -> dict[str, Any]:
    event_tp = int(layers.get("event", {}).get("tail", {}).get("tp_total", 0))
    node_tp = int(layers.get("node", {}).get("tail", {}).get("tp_total", 0))
    episode_tp = int(layers.get("episode", {}).get("tail", {}).get("tp_total", 0))
    role_summary = _classify_roles(ablation)
    positive_evidence_visible = event_tp > 0 or node_tp > 0 or episode_tp > 0
    strong_tp_driver = any(
        item.get("status") == "tp_driver"
        for item in role_summary.values()
        if isinstance(item, dict)
    )
    if positive_evidence_visible and strong_tp_driver:
        next_step = (
            "Worth testing STEPS as a small learned state/composer over typed evidence. "
            "The trace tail already contains positives and ablations show at least one evidence family changes TP."
        )
    elif positive_evidence_visible:
        next_step = (
            "Do a smaller learned reliability/state diagnostic before changing representation. "
            "The trace tail contains positives, but ablation evidence for TP gain is weak or incomplete."
        )
    else:
        next_step = (
            "Do not build STEPS on these features yet. Collect fuller traces or replace semantic/graph representation."
        )
    return {
        "dataset": dataset,
        "event_trace_tp": event_tp,
        "node_trace_tp": node_tp,
        "episode_trace_tp": episode_tp,
        "role_summary": role_summary,
        "next_step": next_step,
    }


def analyze_result(result_dir: Path, ablation_roots: list[Path], budgets: list[int]) -> dict[str, Any]:
    dataset = _dataset_from_result_dir(result_dir)
    eval_payload = _read_json(result_dir / "eval_causal_semantics.json")
    layers = _analyze_layers(result_dir, budgets)
    ablation = _load_ablation(dataset, ablation_roots)
    return {
        "dataset": dataset,
        "result_dir": str(result_dir),
        "eval_summary": _metric_block(eval_payload.get("primary_online_metrics", {})),
        "thresholds": eval_payload.get("online_alert_thresholds", {}),
        "config_subset": {
            key: eval_payload.get("config", {}).get(key)
            for key in (
                "online_expected_alert_budget",
                "online_expected_alert_horizon_events",
                "online_score_trace_enabled",
                "online_score_trace_top_n",
                "online_node_alert_policy",
                "online_risk_composition",
            )
        },
        "layers": layers,
        "ablation": ablation,
        "decision": _decision(dataset, layers, ablation),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)


def _write_markdown(path: Path, analyses: list[dict[str, Any]], budgets: list[int]) -> None:
    lines: list[str] = []
    lines.append("# Causal Semantics Score Trace Evidence Audit")
    lines.append("")
    lines.append("Diagnostic-only audit. Labels are read only after online inference has already produced trace/alert CSVs.")
    lines.append("It must not be used to tune test thresholds directly; it decides whether the next validation-only experiment is worthwhile.")
    lines.append("")
    lines.append("## Decision Summary")
    lines.append("")
    lines.append("| Dataset | Event Trace TP | Node Trace TP | Episode Trace TP | Recommendation |")
    lines.append("|---|---:|---:|---:|---|")
    for analysis in analyses:
        d = analysis["decision"]
        lines.append(
            f"| {analysis['dataset']} | {d['event_trace_tp']} | {d['node_trace_tp']} | {d['episode_trace_tp']} | {d['next_step']} |"
        )
    lines.append("")

    for analysis in analyses:
        dataset = analysis["dataset"]
        lines.append(f"## {dataset}")
        lines.append("")
        lines.append(f"Result directory: `{analysis['result_dir']}`")
        lines.append("")
        metrics = analysis["eval_summary"]
        lines.append("### Existing Online Metrics")
        lines.append("")
        lines.append("| Metric | TP | FP | Alerts | Precision |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(
            f"| immediate_event | {metrics['event_tp']} | {metrics['event_fp']} | {metrics['event_alerts']} | {_fmt(metrics['event_precision'])} |"
        )
        lines.append(
            f"| immediate_node | {metrics['node_tp']} | {metrics['node_fp_strict']} | {metrics['node_alerts']} | {_fmt(metrics['node_precision_strict'])} |"
        )
        lines.append(
            f"| confirmed_event | {metrics['confirmed_event_tp']} | {metrics['confirmed_event_fp']} | {metrics['confirmed_event_alerts']} | {_fmt(metrics['confirmed_event_precision'])} |"
        )
        lines.append(
            f"| confirmed_node | {metrics['confirmed_node_tp']} | {metrics['confirmed_node_fp_strict']} | {metrics['confirmed_node_alerts']} | {_fmt(metrics['confirmed_node_precision_strict'])} |"
        )
        lines.append("")

        for layer_name in ("event", "node", "episode"):
            layer = analysis["layers"][layer_name]
            tail = layer["tail"]
            lines.append(f"### {layer_name.title()} Score Tail")
            lines.append("")
            lines.append(
                f"Source: `{layer['source']}`; rows: `{tail['rows']}`; TP row total in captured tail: `{tail['tp_total']}`; first TP rank: `{tail['first_tp_rank']}`."
            )
            if layer_name == "node":
                lines.append("")
                lines.append(
                    f"Unique GT-positive nodes represented in this node trace tail: `{tail.get('unique_tp_nodes_total')}`. "
                    "Row TP can exceed the GT node count because the same node can appear in multiple high-risk online node records."
                )
            lines.append("")
            if layer_name == "node":
                lines.append("| Budget | Selected | Exact | TP Rows | Unique TP Nodes | FP Rows | Precision | Threshold/Floor |")
                lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
            else:
                lines.append("| Budget | Selected | Exact | TP Rows | FP Rows | Precision | Threshold/Floor |")
                lines.append("|---:|---:|---:|---:|---:|---:|---:|")
            for item in tail["budgets"]:
                if layer_name == "node":
                    lines.append(
                        f"| {item['budget']} | {item['selected']} | {item['exact']} | {item['tp']} | {item.get('unique_tp_nodes', '')} | {item['fp']} | {_fmt(item['precision'])} | {_fmt(item['threshold_or_floor'])} |"
                    )
                else:
                    lines.append(
                        f"| {item['budget']} | {item['selected']} | {item['exact']} | {item['tp']} | {item['fp']} | {_fmt(item['precision'])} | {_fmt(item['threshold_or_floor'])} |"
                    )
            lines.append("")

        lines.append("### Event Evidence Distribution")
        lines.append("")
        lines.append("| Evidence | TP mean | FP mean | TP/FP mean ratio | Top1000 TP | Top1000 precision | Interpretation |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        event_components = analysis["layers"]["event"]["component_audit"]
        roles = analysis["decision"]["role_summary"]
        for component in (
            "semantic_surprise",
            "transition_surprise",
            "residual_fusion",
            "edge_novelty",
            "burst_score",
            "episode_support",
            "chain_diversity",
            "node_state_delta",
            "repeated_window_risk",
        ):
            item = event_components.get(component, {})
            top1000 = next((x for x in item.get("top_by_component", []) if int(x.get("budget", -1)) == 1000), {})
            role = roles.get(component, {})
            lines.append(
                f"| {component} | {_fmt(item.get('tp_distribution', {}).get('mean'))} | {_fmt(item.get('fp_distribution', {}).get('mean'))} | {_fmt(item.get('tp_to_fp_mean_ratio'))} | {top1000.get('tp', 0)} | {_fmt(top1000.get('precision', 0.0))} | {role.get('status', 'unknown')}: {role.get('reason', '')} |"
            )
        lines.append("")

        lines.append("### Ablation Deltas")
        lines.append("")
        lines.append("Positive TP delta means the left mode produced more TP than the right mode. Negative FP delta means FP was reduced.")
        lines.append("")
        lines.append("| Comparison | Event TP Δ | Event FP Δ | Node TP Δ | Node FP Δ | Confirmed Event TP Δ | Confirmed Event FP Δ |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, delta in analysis["ablation"]["evidence_deltas"].items():
            if delta is None:
                lines.append(f"| {name} |  |  |  |  |  | missing |")
                continue
            lines.append(
                f"| {name} | {delta['event_tp']} | {delta['event_fp']} | {delta['node_tp']} | {delta['node_fp_strict']} | {delta['confirmed_event_tp']} | {delta['confirmed_event_fp']} |"
            )
        lines.append("")
        effect = analysis["ablation"].get("confirmation_effect")
        if effect:
            lines.append("### Confirmation Effect")
            lines.append("")
            lines.append("This compares `FULL_CONFIRM` confirmed alerts with `FULL_EVIDENCE` immediate alerts.")
            lines.append("")
            lines.append("| Layer | TP Δ | FP Δ | Alert Δ | Baseline Immediate | Confirmed |")
            lines.append("|---|---:|---:|---:|---|---|")
            base = effect["baseline_full_evidence_immediate"]
            conf = effect["full_confirm_confirmed"]
            lines.append(
                f"| event | {effect['confirmed_event_tp_delta_vs_immediate']} | {effect['confirmed_event_fp_delta_vs_immediate']} | {effect['confirmed_event_alert_delta_vs_immediate']} | TP={base['event_tp']}, FP={base['event_fp']}, alerts={base['event_alerts']} | TP={conf['event_tp']}, FP={conf['event_fp']}, alerts={conf['event_alerts']} |"
            )
            lines.append(
                f"| node | {effect['confirmed_node_tp_delta_vs_immediate']} | {effect['confirmed_node_fp_delta_vs_immediate']} | {effect['confirmed_node_alert_delta_vs_immediate']} | TP={base['node_tp']}, FP={base['node_fp_strict']}, alerts={base['node_alerts']} | TP={conf['node_tp']}, FP={conf['node_fp_strict']}, alerts={conf['node_alerts']} |"
            )
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    budgets = [int(x.strip()) for x in str(args.budgets).split(",") if x.strip()]
    result_dirs = [Path(x) for x in args.result_dir]
    ablation_roots = [Path(x) for x in args.ablation_root]
    if not ablation_roots:
        ablation_roots = [
            Path("outputs/results/tflr_light"),
            Path("outputs/archive_20260512_redeploy/non_deploy_diagnostics"),
        ]
    if not result_dirs:
        result_dirs = [
            Path("outputs/results/cs4m/CLEARSCOPE_E3_CAUSAL_SEMANTICS_BUDGET1000_1P2M"),
            Path("outputs/results/cs4m/CADETS_E3_CAUSAL_SEMANTICS_BUDGET1000_1P2M"),
            Path("outputs/results/cs4m/THEIA_E3_CAUSAL_SEMANTICS_BUDGET1000_1P2M"),
        ]
    analyses = [analyze_result(path, ablation_roots, budgets) for path in result_dirs]
    out = {
        "purpose": "post-inference score trace evidence audit",
        "leakage_check": "labels are used only after inference for diagnostic reporting; no thresholds or models are updated",
        "budgets": budgets,
        "ablation_roots": [str(path) for path in ablation_roots],
        "analyses": analyses,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, sort_keys=True))
    _write_markdown(Path(args.out_md), analyses, budgets)
    if args.print_summary:
        for analysis in analyses:
            d = analysis["decision"]
            print(
                f"{analysis['dataset']}: event_tp={d['event_trace_tp']} "
                f"node_tp={d['node_trace_tp']} episode_tp={d['episode_trace_tp']} "
                f"next={d['next_step']}"
            )
        print(f"Wrote {out_json}")
        print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
