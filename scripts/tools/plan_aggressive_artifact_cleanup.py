#!/usr/bin/env python3
"""Create and optionally apply aggressive generated-artifact cleanup manifests."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


DEFAULT_KEEP_PATHS = [
    "outputs/results/tflr_light/CADETS_E3_FREEBSD_V3_SAFE_LEXICAL_E4_FULL",
    "outputs/results/phase3g_theia_e3_e4_theia_v1_lazy100k_pair_only_full_rerun_20260602_142411",
    "outputs/results/tflr_light/CLEARSCOPE_E5_V33B_FULL_PIPELINE",
    "outputs/results/tflr_light/optc_windows_v1_3c_full/OPTC_051_OPTC_WINDOWS_V1_3C_DETAIL_FULL_INFER",
    "outputs/results/tflr_light/optc_windows_v1_3c_full/OPTC_201_OPTC_WINDOWS_V1_3C_DETAIL_FULL_INFER",
    "outputs/results/tflr_light/optc_windows_v1_3c_full/OPTC_501_OPTC_WINDOWS_V1_3C_DETAIL_FULL_INFER",
    "outputs/models",
]


@dataclass
class CleanupPlan:
    """Manifest of paths to keep and delete."""

    keep_paths: list[str]
    delete_paths: list[str]
    total_delete_bytes: int


def build_cleanup_plan(
    repo_root: Path,
    keep_paths: list[Path],
    output_dir: Path,
) -> CleanupPlan:
    """Build a cleanup manifest without deleting files."""
    repo_root = Path(repo_root).resolve()
    keep_abs = {_resolve(repo_root, path) for path in keep_paths}
    keep_abs.add(_resolve(repo_root, output_dir))
    outputs_root = repo_root / "outputs"
    delete_paths: list[Path] = []
    if outputs_root.exists():
        for child in _iter_top_cleanup_candidates(outputs_root):
            resolved = child.resolve()
            if _is_kept(resolved, keep_abs):
                continue
            delete_paths.append(resolved)
    total_bytes = sum(_path_size(path) for path in delete_paths)
    return CleanupPlan(
        keep_paths=[str(path) for path in sorted(keep_abs)],
        delete_paths=[str(path) for path in sorted(delete_paths)],
        total_delete_bytes=total_bytes,
    )


def validate_delete_paths(repo_root: Path, plan: CleanupPlan) -> None:
    """Refuse unsafe delete manifest entries."""
    repo_root = Path(repo_root).resolve()
    outputs_root = repo_root / "outputs"
    for raw_path in plan.delete_paths:
        path = Path(raw_path).resolve()
        if outputs_root not in path.parents:
            raise ValueError(f"refusing to delete path outside outputs: {path}")
        if path == outputs_root:
            raise ValueError("refusing to delete outputs root")
        for protected in ("ground_truth", "cs4m", "scripts", "configs", "docs"):
            protected_path = repo_root / protected
            if protected_path == path or protected_path in path.parents:
                raise ValueError(f"refusing to delete protected path: {path}")


def write_plan(plan: CleanupPlan, output_dir: Path) -> None:
    """Write keep/delete manifests and summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "keep_manifest.json").write_text(
        json.dumps({"keep_paths": plan.keep_paths}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "delete_manifest.json").write_text(
        json.dumps(
            {
                "delete_paths": plan.delete_paths,
                "total_delete_bytes": plan.total_delete_bytes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "cleanup_summary.json").write_text(
        json.dumps(
            {
                "keep_count": len(plan.keep_paths),
                "delete_count": len(plan.delete_paths),
                "total_delete_bytes": plan.total_delete_bytes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def apply_delete_plan(repo_root: Path, plan: CleanupPlan) -> None:
    """Delete exactly the paths listed in a validated manifest."""
    validate_delete_paths(repo_root, plan)
    for raw_path in plan.delete_paths:
        path = Path(raw_path)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _iter_top_cleanup_candidates(outputs_root: Path) -> list[Path]:
    roots = [
        outputs_root / "cache",
        outputs_root / "logs",
        outputs_root / "diagnostics",
        outputs_root / "results" / "tflr_light",
        outputs_root / "results" / "phase3g_theia_e3_e4_theia_v1_lazy100k_pair_only_full",
        outputs_root
        / "results"
        / "phase3g_theia_e3_e4_theia_v1_lazy100k_pair_only_full_rerun_20260602_142411",
    ]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.name in {"cache", "logs", "diagnostics", "tflr_light"}:
            candidates.extend(root.iterdir())
        else:
            candidates.append(root)
    return candidates


def _resolve(repo_root: Path, path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _is_kept(path: Path, keep_abs: set[Path]) -> bool:
    return any(path == keep or keep in path.parents or path in keep.parents for keep in keep_abs)


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    if path.is_dir():
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
    return total


def main() -> int:
    """Create or apply an aggressive cleanup manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--keep-path", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    repo_root = Path.cwd()
    keep_paths = [Path(path) for path in (args.keep_path or DEFAULT_KEEP_PATHS)]
    output_dir = Path(args.output_dir)
    plan = build_cleanup_plan(repo_root, keep_paths, output_dir)
    validate_delete_paths(repo_root, plan)
    write_plan(plan, output_dir)
    if args.apply:
        apply_delete_plan(repo_root, plan)
    print(
        json.dumps(
            {
                "apply": bool(args.apply),
                "keep_count": len(plan.keep_paths),
                "delete_count": len(plan.delete_paths),
                "total_delete_bytes": plan.total_delete_bytes,
                "output_dir": str(output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
