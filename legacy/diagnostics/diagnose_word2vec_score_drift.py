#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.config.provnet_utils import init_database_connection
from scripts.tools.causal_semantics_slim import (
    SlimConfig,
    _cfg_for_dataset,
    _empirical_tail_score_compact,
    _event_score_from_residual_tail,
    _is_attack_label,
    _label_name,
    _load_process_config,
    _test_scoring_node_maps,
    build_node_maps,
    enrich_row_with_file_meta,
    enrich_row_with_netflow,
    enrich_row_with_process_meta,
    load_ground_truth_indices_readonly,
    row_fields,
    stream_dataset_rows,
    stream_dataset_rows_slim,
)
from cs4m.embeddings.residual import residual_embedder_from_state_dict
from cs4m.models.lowrank import SSPMLowRankModel


DATASETS = ("CADETS_E3", "THEIA_E3")
RESULT_ROOT = Path("outputs/results/tflr_light")
DIAG_ROOT = Path("outputs/diagnostics/word2vec_score_drift")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _config_from_eval(eval_payload: Mapping[str, Any]) -> SlimConfig:
    values = dict(eval_payload["config"])
    return SlimConfig(**values)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _label_group(label: object) -> str:
    text = str(label).strip().lower()
    if text in {"2", "malicious"}:
        return "malicious"
    if text in {"1", "suspicious"}:
        return "suspicious"
    if _is_attack_label(label):
        return "suspicious"
    return "benign"


def _iter_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(max(int(round((len(ordered) - 1) * float(q))), 0), len(ordered) - 1)
    return float(ordered[index])


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, list[float]] = {"benign": [], "suspicious": [], "malicious": []}
    for row in rows:
        by_group.setdefault(str(row["label_group"]), []).append(_safe_float(row["event_score"]))
    out: list[dict[str, Any]] = []
    for group, values in by_group.items():
        if values:
            arr = np.asarray(values, dtype=np.float64)
            out.append(
                {
                    "label_group": group,
                    "count": int(arr.size),
                    "min": float(np.min(arr)),
                    "p50": _quantile(values, 0.50),
                    "p90": _quantile(values, 0.90),
                    "p99": _quantile(values, 0.99),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                },
            )
        else:
            out.append({"label_group": group, "count": 0})
    return out


def _svg_scatter(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    title: str,
    max_points: int,
) -> str:
    sampled = list(rows)
    if len(sampled) > int(max_points):
        rng = random.Random(0)
        sampled = rng.sample(sampled, int(max_points))
    width = 1400
    height = 620
    margin_left = 80
    margin_right = 30
    margin_top = 45
    margin_bottom = 70
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_x = max((_safe_float(row.get("event_index")) for row in sampled), default=1.0)
    max_y = max(
        [threshold, *[_safe_float(row.get("event_score")) for row in sampled]],
        default=1.0,
    )
    max_y = max(max_y, 1.0)
    colors = {"benign": "#8a8f98", "suspicious": "#e0a106", "malicious": "#d33f49"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:13px}"
        ".axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}"
        ".pt{opacity:.55}</style>",
        f'<text x="{margin_left}" y="25" style="font-size:18px;font-weight:700">'
        f"{html.escape(title)}</text>",
    ]
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = margin_top + plot_h - frac * plot_h
        value = frac * max_y
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}"/>')
        parts.append(f'<text x="8" y="{y+4:.1f}">{value:.2f}</text>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top+plot_h}"/>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top+plot_h}" x2="{width-margin_right}" y2="{margin_top+plot_h}"/>')
    threshold_y = margin_top + plot_h - (float(threshold) / max_y) * plot_h
    parts.append(f'<line x1="{margin_left}" y1="{threshold_y:.1f}" x2="{width-margin_right}" y2="{threshold_y:.1f}" stroke="#222" stroke-width="2" stroke-dasharray="8 5"/>')
    parts.append(f'<text x="{width-margin_right-170}" y="{threshold_y-8:.1f}">threshold={threshold:.3f}</text>')
    for row in sampled:
        x = margin_left + (_safe_float(row.get("event_index")) / max_x) * plot_w
        y = margin_top + plot_h - (_safe_float(row.get("event_score")) / max_y) * plot_h
        label = str(row.get("label_group", "benign"))
        color = colors.get(label, "#777")
        radius = 2.2 if label == "benign" else 4.2
        parts.append(f'<circle class="pt" cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"/>')
    legend_x = margin_left + 10
    for offset, label in enumerate(("benign", "suspicious", "malicious")):
        x = legend_x + offset * 135
        parts.append(f'<circle cx="{x}" cy="{height-28}" r="5" fill="{colors[label]}"/>')
        parts.append(f'<text x="{x+10}" y="{height-24}">{label}</text>')
    parts.append(f'<text x="{width/2-80:.1f}" y="{height-12}">event_index</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _write_visual(path: Path, rows: Sequence[Mapping[str, Any]], threshold: float, title: str) -> None:
    svg = _svg_scatter(rows, threshold, title, max_points=60000)
    path.with_suffix(".svg").write_text(svg, encoding="utf-8")
    html_text = (
        "<!doctype html><meta charset='utf-8'><title>"
        + html.escape(title)
        + "</title><body>"
        + svg
        + "</body>"
    )
    path.with_suffix(".html").write_text(html_text, encoding="utf-8")
    _write_matplotlib_scatter(path.with_suffix(".png"), rows, threshold, title)


def _write_matplotlib_scatter(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
    title: str,
) -> None:
    sampled = list(rows)
    if len(sampled) > 120000:
        rng = random.Random(0)
        sampled = rng.sample(sampled, 120000)
    colors = {"benign": "#8a8f98", "suspicious": "#e0a106", "malicious": "#d33f49"}
    sizes = {"benign": 4, "suspicious": 14, "malicious": 18}
    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    for label in ("benign", "suspicious", "malicious"):
        group = [row for row in sampled if str(row.get("label_group", "benign")) == label]
        if not group:
            continue
        ax.scatter(
            [_safe_float(row.get("event_index")) for row in group],
            [_safe_float(row.get("event_score")) for row in group],
            s=sizes[label],
            c=colors[label],
            alpha=0.45 if label == "benign" else 0.78,
            label=f"{label} ({len(group)})",
            edgecolors="none",
        )
    ax.axhline(float(threshold), color="#222222", linestyle="--", linewidth=1.2)
    ax.text(
        0.99,
        float(threshold),
        f"threshold={threshold:.3f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax.set_title(title)
    ax.set_xlabel("event_index")
    ax.set_ylabel("event_score")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _alert_rows(result_dir: Path) -> list[dict[str, Any]]:
    path = result_dir / "online_event_alerts.csv"
    rows: list[dict[str, Any]] = []
    for row in _iter_csv(path):
        label = row.get("event_label", "0")
        rows.append(
            {
                "event_index": int(row["event_index"]),
                "timestamp_ns": int(row["timestamp_ns"]),
                "event_score": _safe_float(row["event_score"]),
                "residual_tail": _safe_float(row.get("residual_tail")),
                "residual_score": _safe_float(row.get("residual_score")),
                "threshold": _safe_float(row.get("threshold")),
                "label": label,
                "label_group": _label_group(label),
                "src_idx": row.get("src_idx", ""),
                "dst_idx": row.get("dst_idx", ""),
                "info_src": row.get("info_src", ""),
                "info_dst": row.get("info_dst", ""),
                "action": row.get("action", ""),
                "object_type": row.get("object_type", ""),
            },
        )
    return rows


def _restore_artifacts(eval_payload: Mapping[str, Any], config: SlimConfig):
    print(f"[diagnose] restore_artifacts_start dataset={config.dataset}", file=sys.stderr, flush=True)
    cache_info = eval_payload.get("cache", {})
    cache_path = Path(str(cache_info.get("cache_path") or cache_info.get("path") or ""))
    if not cache_path.exists():
        raise FileNotFoundError(f"cache not found: {cache_path}")
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    print(f"[diagnose] cache_loaded dataset={config.dataset}", file=sys.stderr, flush=True)
    if not isinstance(payload, Mapping):
        raise TypeError("cache payload must be a mapping")
    if payload.get("fingerprint_payload", {}).get("dataset") != config.dataset:
        raise ValueError(f"cache dataset mismatch for {cache_path}")
    if payload.get("cache_scope") != "train_validation_base_scores":
        raise ValueError(f"unsupported cache scope for diagnostics: {payload.get('cache_scope')}")
    process_cfg = _load_process_config(config)
    model = SSPMLowRankModel.from_state_dict(payload["model"])
    embedder = residual_embedder_from_state_dict(payload["embedder"])
    print(f"[diagnose] models_restored dataset={config.dataset}", file=sys.stderr, flush=True)
    validation_residual_sorted = np.asarray(payload["validation_residual_sorted"], dtype=np.float32)
    class Calibration:
        pass

    calibration = Calibration()
    calibration.residual_sorted = validation_residual_sorted
    calibration.threshold = float(
        eval_payload.get("validation_event_score_summary", {}).get("threshold", 0.0),
    )
    print(f"[diagnose] calibration_restored dataset={config.dataset}", file=sys.stderr, flush=True)
    return process_cfg, model, embedder, calibration


def _score_full_stream(
    dataset: str,
    result_dir: Path,
    out_dir: Path,
    sample_rate: float,
    max_events: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print(f"[diagnose] full_score_start dataset={dataset}", file=sys.stderr, flush=True)
    eval_payload = _read_json(result_dir / "eval_causal_semantics_slim.json")
    config = _config_from_eval(eval_payload)
    if int(max_events) > 0:
        config.max_test_events = int(max_events)
    process_cfg, model, embedder, calibration = _restore_artifacts(eval_payload, config)
    db_cfg = _cfg_for_dataset(dataset)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    cur, conn = init_database_connection(db_cfg)
    rows_for_visual: list[dict[str, Any]] = []
    summary_counter: Counter[str] = Counter()
    out_csv = out_dir / f"{dataset.lower()}_full_event_scores.csv"
    rng = random.Random(0)
    started = time.perf_counter()
    try:
        print(f"[diagnose] build_node_maps_start dataset={dataset}", file=sys.stderr, flush=True)
        node_maps = build_node_maps(cur)
        print(f"[diagnose] build_node_maps_end dataset={dataset}", file=sys.stderr, flush=True)
        abnormal_nodes = load_ground_truth_indices_readonly(db_cfg, node_maps["uuid2index"])
        test_node_maps = _test_scoring_node_maps(node_maps)
        stream_status: dict[str, str] = {}
        test_rows = stream_dataset_rows_slim(
            conn,
            str(eval_payload["year_month"]),
            [int(day) for day in eval_payload["test_days"]],
            test_node_maps,
            True,
            config.fetch_size,
            config.max_test_events,
            config.db_stream_mode,
            abnormal_nodes=abnormal_nodes,
            stream_status=stream_status,
        )
        residual_sorted = calibration.residual_sorted
        model.reset_state()
        fieldnames = [
            "event_index",
            "timestamp_ns",
            "event_score",
            "residual_tail",
            "residual_score",
            "threshold",
            "label",
            "label_group",
            "src_idx",
            "dst_idx",
            "info_src",
            "info_dst",
            "action",
            "object_type",
        ]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for count, row in enumerate(test_rows, start=1):
                fields = row_fields(
                    row,
                    config.max_tokens_per_node,
                    dataset=config.dataset,
                    process_semantic_config=process_cfg,
                )
                from scripts.tools.causal_semantics_slim import _encode_row

                z = _encode_row(embedder, row, config, process_cfg)
                residual_score = model.residual_score(model.predict(fields), z)
                residual_tail = _empirical_tail_score_compact(residual_score, residual_sorted)
                event_score = _event_score_from_residual_tail(
                    residual_tail,
                    config.event_score_mode,
                )
                label = row.get("label", 0)
                label_group = _label_group(label)
                out_row = {
                    "event_index": int(row["event_index"]),
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "event_score": float(event_score),
                    "residual_tail": float(residual_tail),
                    "residual_score": float(residual_score),
                    "threshold": float(calibration.threshold),
                    "label": int(label),
                    "label_group": label_group,
                    "src_idx": int(row["src_idx"]),
                    "dst_idx": int(row["dst_idx"]),
                    "info_src": int(row["info_src"]),
                    "info_dst": int(row["info_dst"]),
                    "action": str(row["action"]),
                    "object_type": str(row["object_type"]),
                }
                writer.writerow(out_row)
                summary_counter[label_group] += 1
                if label_group != "benign" or rng.random() <= float(sample_rate):
                    rows_for_visual.append(out_row)
                model.update_states(fields, z)
                if count % 500000 == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    print(
                        f"[diagnose] {dataset} scored={count} "
                        f"rate={count / elapsed:.1f}/s labels={dict(summary_counter)}",
                        file=sys.stderr,
                        flush=True,
                    )
    finally:
        try:
            conn.close()
        except Exception:
            pass
    metadata = {
        "dataset": dataset,
        "source_result_dir": str(result_dir),
        "full_scores_csv": str(out_csv),
        "visual_rows": len(rows_for_visual),
        "label_counts": dict(summary_counter),
        "threshold": float(calibration.threshold),
    }
    return rows_for_visual, metadata


def _export_malicious_nodes(dataset: str, node_kind: str, out_path: Path) -> dict[str, Any]:
    db_cfg = _cfg_for_dataset(dataset)
    db_cfg.database.host = os.getenv("CLAD_DB_HOST", "localhost")
    db_cfg.database.user = os.getenv("CLAD_DB_USER", "postgres")
    db_cfg.database.password = os.getenv("CLAD_DB_PASSWORD", "")
    db_cfg.database.port = int(os.getenv("CLAD_DB_PORT", "5432"))
    cur, conn = init_database_connection(db_cfg)
    try:
        node_maps = build_node_maps(cur)
        abnormal = load_ground_truth_indices_readonly(db_cfg, node_maps["uuid2index"])
        rows = []
        if node_kind == "netflow":
            for index_id, meta in sorted(node_maps["netflow_meta"].items()):
                if int(index_id) not in abnormal:
                    continue
                uuid = ""
                hash_id = ""
                for candidate_hash, value in node_maps["hash2uuid_index"].items():
                    if int(value[1]) == int(index_id):
                        uuid = str(value[0])
                        hash_id = str(candidate_hash)
                        break
                rows.append(
                    {
                        "dataset": dataset,
                        "index_id": int(index_id),
                        "node_uuid": uuid,
                        "hash_id": hash_id,
                        "node_type": "netflow",
                        "src_addr": meta.get("src_addr", ""),
                        "src_port": meta.get("src_port", ""),
                        "dst_addr": meta.get("dst_addr", ""),
                        "dst_port": meta.get("dst_port", ""),
                    },
                )
            fields = [
                "dataset",
                "index_id",
                "node_uuid",
                "hash_id",
                "node_type",
                "src_addr",
                "src_port",
                "dst_addr",
                "dst_port",
            ]
        else:
            for index_id, meta in sorted(node_maps["file_meta"].items()):
                if int(index_id) not in abnormal:
                    continue
                uuid = ""
                hash_id = ""
                for candidate_hash, value in node_maps["hash2uuid_index"].items():
                    if int(value[1]) == int(index_id):
                        uuid = str(value[0])
                        hash_id = str(candidate_hash)
                        break
                rows.append(
                    {
                        "dataset": dataset,
                        "index_id": int(index_id),
                        "node_uuid": uuid,
                        "hash_id": hash_id,
                        "node_type": "file",
                        "path": meta.get("path", ""),
                        "path_missing": int(str(meta.get("path", "")).strip().lower() in {"", "none", "null", "na"}),
                    },
                )
            fields = ["dataset", "index_id", "node_uuid", "hash_id", "node_type", "path", "path_missing"]
        _write_csv(out_path, rows, fields)
        return {"path": str(out_path), "count": len(rows), "node_kind": node_kind}
    finally:
        conn.close()


def _write_report(out_dir: Path, payload: Mapping[str, Any]) -> None:
    lines = ["# Word2Vec Score Drift Diagnostics", ""]
    for dataset, item in payload["datasets"].items():
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Alert CSV rows: {item['alert']['count']}",
                f"- Full score label counts: {item['full']['label_counts']}",
                f"- Validation threshold: {item['full']['threshold']:.6f}",
                f"- Alert scatter: `{item['alert']['html']}`",
                f"- Full score scatter: `{item['full']['html']}`",
                "",
            ],
        )
    lines.extend(
        [
            "## Historical Interpretation",
            "",
            "- CADETS historical best references are CSSM canonical / mid-rank style outputs, "
            "which rely on node-level accumulated evidence rather than only residual event tails.",
            "- THEIA historical best references are AdaptiveMemory, IntrinsicFirst, and CSSM V2; "
            "these are stronger when validation event-tail thresholds do not transfer.",
            "- The diagnostic plots should be read as post-stream analysis only. Labels are attached "
            "after scoring and are not used for thresholding.",
            "",
        ],
    )
    (out_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets_payload: dict[str, Any] = {}
    for dataset in DATASETS:
        result_dir = RESULT_ROOT / f"{dataset}_WORD2VEC_RES_ONLY_NOACTION_BOUNDED_FULL"
        eval_payload = _read_json(result_dir / "eval_causal_semantics_slim.json")
        threshold = float(eval_payload["validation_event_score_summary"]["threshold"])
        alerts = _alert_rows(result_dir)
        alert_prefix = out_dir / f"{dataset.lower()}_alert_event_scores"
        _write_csv(alert_prefix.with_suffix(".csv"), alerts, list(alerts[0].keys()) if alerts else [])
        _write_csv(
            out_dir / f"{dataset.lower()}_alert_score_summary.csv",
            _group_summary(alerts),
            ["label_group", "count", "min", "p50", "p90", "p99", "max", "mean"],
        )
        _write_visual(alert_prefix, alerts, threshold, f"{dataset} alert-only event scores")
        if bool(args.skip_full_scores):
            full_meta = {
                "dataset": dataset,
                "label_counts": {},
                "threshold": threshold,
                "skipped": True,
            }
            full_prefix = out_dir / f"{dataset.lower()}_full_event_scores_sample"
        else:
            full_rows, full_meta = _score_full_stream(
                dataset,
                result_dir,
                out_dir,
                sample_rate=float(args.sample_rate),
                max_events=int(args.max_events),
            )
            full_prefix = out_dir / f"{dataset.lower()}_full_event_scores_sample"
            _write_csv(
                full_prefix.with_suffix(".csv"),
                full_rows,
                list(full_rows[0].keys()) if full_rows else [],
            )
            _write_csv(
                out_dir / f"{dataset.lower()}_full_score_summary.csv",
                _group_summary(full_rows),
                ["label_group", "count", "min", "p50", "p90", "p99", "max", "mean"],
            )
            _write_visual(full_prefix, full_rows, threshold, f"{dataset} full event scores")
        datasets_payload[dataset] = {
            "alert": {
                "count": len(alerts),
                "html": str(alert_prefix.with_suffix(".html")),
                "svg": str(alert_prefix.with_suffix(".svg")),
                "csv": str(alert_prefix.with_suffix(".csv")),
            },
            "full": {
                **full_meta,
                "html": str(full_prefix.with_suffix(".html")),
                "svg": str(full_prefix.with_suffix(".svg")),
                "sample_csv": str(full_prefix.with_suffix(".csv")),
            },
        }
    node_exports = {
        "theia_malicious_netflow": _export_malicious_nodes(
            "THEIA_E3",
            "netflow",
            out_dir / "theia_e3_malicious_netflow_nodes.csv",
        ),
        "cadets_malicious_file": _export_malicious_nodes(
            "CADETS_E3",
            "file",
            out_dir / "cadets_e3_malicious_file_nodes.csv",
        ),
    }
    payload = {"datasets": datasets_payload, "node_exports": node_exports}
    (out_dir / "diagnostic_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(out_dir, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(DIAG_ROOT))
    parser.add_argument("--sample_rate", type=float, default=0.01)
    parser.add_argument("--max_events", type=int, default=0)
    parser.add_argument("--skip_full_scores", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
