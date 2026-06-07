#!/usr/bin/env python3
"""Inventory or quarantine wrong ClearScope refined artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Mapping


WRONG_LEGACY_MODES = {"legacy", "clearscope_android_semantics_v1"}
REFINED_MARKERS = ("CLEARSCOPE", "clearscope")
RAW_DETAIL_MARKERS = ("RAW_DETAIL_V2_REFINED", "raw_detail_v2_refined")
DEFAULT_TMP_ROOT = Path(os.environ.get("TMP_ROOT", "tmp"))
DEFAULT_QUARANTINE_DIR = DEFAULT_TMP_ROOT / "quarantine_wrong_clearscope_refined_artifacts"
DEFAULT_CLEARSCOPE_PHASE3G_V2 = DEFAULT_TMP_ROOT / "clearscope_phase3g_v2"
DEFAULT_CLEARSCOPE_AUDIT_REFINED = (
    DEFAULT_TMP_ROOT / "clearscope_detail_semantic_audit_v2_refined"
)
DEFAULT_CLEARSCOPE_AUDIT_FINAL_GATE = (
    DEFAULT_TMP_ROOT / "clearscope_detail_semantic_audit_v2_final_gate"
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return dict(payload) if isinstance(payload, Mapping) else {}
    except Exception:
        return {}


def _read_pickle_metadata(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return {
        key: value
        for key, value in payload.items()
        if key not in {"model_state", "embedder", "embedder_state", "model"}
    }


def _metadata_for_path(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return _read_json(path)
    if path.suffix.lower() == ".pkl":
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            payload = _read_json(sidecar)
            if payload:
                return payload
        return _read_pickle_metadata(path)
    if path.is_dir():
        for name in (
            "eval_causal_semantics_slim.json",
            "node_embedding_meta.json",
            "phase3e_residual_word2vec_node_action_coverage_audit.json",
            "bounded_ablation_summary.json",
            "config_resolved.json",
        ):
            candidate = path / name
            if candidate.exists():
                payload = _read_json(candidate)
                if payload:
                    return payload
    return {}


def _contains_clearscope_refined(path: Path) -> bool:
    text = str(path)
    return any(marker in text for marker in REFINED_MARKERS) and any(
        marker in text for marker in RAW_DETAIL_MARKERS
    )


def _semantic_mode_from_metadata(metadata: Mapping[str, Any]) -> tuple[str, str]:
    semantic_mode = str(metadata.get("semantic_mode", "") or "").strip()
    semantic_rules = metadata.get("semantic_rules", {})
    rules_text = json.dumps(semantic_rules, sort_keys=True) if isinstance(
        semantic_rules,
        Mapping,
    ) else str(semantic_rules or "")
    if not semantic_mode and isinstance(semantic_rules, Mapping):
        semantic_mode = str(semantic_rules.get("clearscope", "") or "").strip()
    config = metadata.get("config", {})
    if not semantic_mode and isinstance(config, Mapping):
        semantic_mode = str(config.get("semantic_mode", "") or "").strip()
    if not rules_text and isinstance(config, Mapping):
        rules_text = str(config.get("clearscope_semantic_rules_version", "") or "")
    return semantic_mode, rules_text


def _references_wrong_word2vec(metadata: Mapping[str, Any], wrong_word2vec: set[str]) -> bool:
    encoded = json.dumps(metadata, sort_keys=True, default=str)
    for path in wrong_word2vec:
        if path and path in encoded:
            return True
    return False


def _classify(path: Path, wrong_word2vec: set[str]) -> dict[str, Any] | None:
    if "CADETS" in str(path) or "THEIA" in str(path):
        return None
    if not _contains_clearscope_refined(path):
        return None
    metadata = _metadata_for_path(path)
    semantic_mode, rules_text = _semantic_mode_from_metadata(metadata)
    mode_lower = semantic_mode.lower()
    artifact_type = "directory" if path.is_dir() else path.suffix.lower().lstrip(".")
    reason = ""
    if mode_lower in WRONG_LEGACY_MODES or "clearscope_android_semantics_v1" in rules_text:
        reason = "clearscope_refined_name_metadata_legacy"
        if path.suffix.lower() == ".pkl":
            wrong_word2vec.add(str(path))
    elif _references_wrong_word2vec(metadata, wrong_word2vec):
        reason = "depends_on_wrong_clearscope_refined_word2vec"
    elif path.is_dir() and any(str(wrong) in str(path) for wrong in wrong_word2vec):
        reason = "depends_on_wrong_clearscope_refined_word2vec"
    if not reason:
        return None
    return {
        "path": str(path),
        "artifact_type": artifact_type,
        "size_bytes": _path_size(path),
        "reason": reason,
        "metadata_semantic_mode": semantic_mode,
        "metadata_semantic_rules": rules_text,
        "action": "pending",
        "quarantine_path": "",
    }


def _path_size(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                total += int(child.stat().st_size)
    return int(total)


def _candidate_paths(roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            paths.append(root)
            continue
        for child in root.rglob("*"):
            if not _contains_clearscope_refined(child):
                continue
            if child.is_dir():
                paths.append(child)
            elif child.suffix.lower() in {".pkl", ".json"}:
                paths.append(child)
    return sorted(set(paths), key=lambda value: str(value))


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "artifact_type",
        "size_bytes",
        "reason",
        "metadata_semantic_mode",
        "metadata_semantic_rules",
        "action",
        "quarantine_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _move_to_quarantine(row: dict[str, Any], quarantine_dir: Path) -> dict[str, Any]:
    src = Path(str(row["path"]))
    rel = Path(str(row["path"]).lstrip("/"))
    dst = quarantine_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        suffix = 1
        while dst.with_name(f"{dst.name}.dup{suffix}").exists():
            suffix += 1
        dst = dst.with_name(f"{dst.name}.dup{suffix}")
    shutil.move(str(src), str(dst))
    updated = dict(row)
    updated["action"] = "quarantined"
    updated["quarantine_path"] = str(dst)
    return updated


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Inventory or quarantine wrong ClearScope refined artifacts."""
    roots = [Path(path) for path in args.roots]
    wrong_word2vec: set[str] = set()
    rows: list[dict[str, Any]] = []
    for path in _candidate_paths(roots):
        row = _classify(path, wrong_word2vec)
        if row is not None:
            rows.append(row)
    quarantine_dir = Path(args.quarantine_dir)
    moved_rows: list[dict[str, Any]] = []
    if args.apply:
        for row in rows:
            source = Path(str(row["path"]))
            if source.exists():
                moved_rows.append(_move_to_quarantine(row, quarantine_dir))
            else:
                stale = dict(row)
                stale["action"] = "missing_before_quarantine"
                moved_rows.append(stale)
        rows = moved_rows
    manifest_path = Path(args.manifest)
    summary_json = Path(args.summary_json)
    summary_md = Path(args.summary_md)
    if not args.apply and manifest_path.exists():
        manifest_path = manifest_path.with_name("wrong_artifacts_inventory.csv")
        summary_json = summary_json.with_name(
            "cleanup_wrong_clearscope_refined_inventory_summary.json",
        )
        summary_md = summary_md.with_name(
            "cleanup_wrong_clearscope_refined_inventory_summary.md",
        )
    _write_manifest(manifest_path, rows)
    moved_count = sum(1 for row in rows if row.get("action") == "quarantined")
    moved_total = sum(
        int(row.get("size_bytes", 0) or 0)
        for row in rows
        if row.get("action") == "quarantined"
    )
    summary = {
        "quarantine_dir": str(quarantine_dir),
        "moved_count": int(moved_count),
        "moved_total_size_bytes": int(moved_total),
        "deleted_count": 0,
        "deleted_total_size_bytes": 0,
        "manifest_path": str(manifest_path),
        "kept_audit_dirs": [
            str(DEFAULT_CLEARSCOPE_AUDIT_REFINED),
            str(DEFAULT_CLEARSCOPE_AUDIT_FINAL_GATE),
        ],
        "notes": [
            "inventory only" if not args.apply else "moved candidates to quarantine",
            "CADETS and THEIA paths are ignored",
            "no rm/delete is performed by this script",
        ],
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_md.write_text(
        "# ClearScope Wrong Refined Artifact Cleanup\n\n"
        f"- quarantine_dir: `{summary['quarantine_dir']}`\n"
        f"- moved_count: {summary['moved_count']}\n"
        f"- moved_total_size_bytes: {summary['moved_total_size_bytes']}\n"
        "- deleted_count: 0\n"
        f"- manifest: `{summary['manifest_path']}`\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--quarantine_dir",
        default=DEFAULT_QUARANTINE_DIR,
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_QUARANTINE_DIR / "wrong_artifacts_manifest.csv",
    )
    parser.add_argument(
        "--summary_json",
        default=DEFAULT_QUARANTINE_DIR / "cleanup_wrong_clearscope_refined_summary.json",
    )
    parser.add_argument(
        "--summary_md",
        default=DEFAULT_QUARANTINE_DIR / "cleanup_wrong_clearscope_refined_summary.md",
    )
    parser.add_argument(
        "roots",
        nargs="*",
        default=[
            "outputs/models/residual_word2vec",
            "outputs/models/sspm_phase3e_clearscope_raw_detail_v2_refined_q09995",
            "outputs/models/phase3g_action_heads_v2_clearscope_raw_detail_v2_refined_q09995",
            "outputs/cache/phase3e_clearscope_raw_detail_v2_refined_q09995",
            "outputs/cache/phase3g_action_validation_q09995/CLEARSCOPE_E3",
            "outputs/cache/phase3g_endpoint_suppression_compact_q09995/CLEARSCOPE_E3",
            "outputs/results",
            "outputs/diagnostics",
            DEFAULT_CLEARSCOPE_PHASE3G_V2,
            DEFAULT_CLEARSCOPE_AUDIT_REFINED,
            DEFAULT_CLEARSCOPE_AUDIT_FINAL_GATE,
        ],
    )
    return parser.parse_args()


def main() -> None:
    """Run inventory or quarantine."""
    run(parse_args())


if __name__ == "__main__":
    main()
