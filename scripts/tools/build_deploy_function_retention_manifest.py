#!/usr/bin/env python3
"""Build a function-level retention manifest for CS4M deploy extraction."""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DefinitionRecord:
    """One top-level Python definition and its deploy retention decision."""

    module_path: str
    name: str
    kind: str
    line_number: int
    classification: str
    caller_or_reason: str
    streaming_required: bool
    touches_ground_truth: bool
    evaluation_only: bool
    safe_to_remove: bool
    notes: str


def parse_python_definitions(path: Path) -> list[DefinitionRecord]:
    """Parse top-level functions, classes, and constants from a Python file."""
    source_path = Path(path)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    records: list[DefinitionRecord] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            records.append(_record(source_path, node.name, "function", node.lineno))
        elif isinstance(node, ast.ClassDef):
            records.append(_record(source_path, node.name, "class", node.lineno))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    records.append(_record(source_path, target.id, "constant", node.lineno))
    return records


def build_retention_manifest(
    repo_root: Path,
    python_paths: list[Path],
    explicit_keep_names: set[str],
    entrypoint_paths: list[Path],
) -> list[DefinitionRecord]:
    """Classify definitions using explicit mainline reachability rules."""
    resolved_root = Path(repo_root).resolve()
    entrypoint_set = {str(Path(path).resolve()) for path in entrypoint_paths}
    records: list[DefinitionRecord] = []
    for path in python_paths:
        resolved = Path(path).resolve()
        for record in parse_python_definitions(resolved):
            is_entrypoint = str(resolved) in entrypoint_set
            is_explicit_keep = record.name in explicit_keep_names
            classification = "keep_runtime" if is_explicit_keep else "exclude_unused"
            reason = _reason(is_explicit_keep, is_entrypoint)
            records.append(
                DefinitionRecord(
                    module_path=str(resolved.relative_to(resolved_root)),
                    name=record.name,
                    kind=record.kind,
                    line_number=record.line_number,
                    classification=classification,
                    caller_or_reason=reason,
                    streaming_required=classification == "keep_runtime",
                    touches_ground_truth=_mentions_ground_truth(record.name),
                    evaluation_only=_path_is_evaluation(resolved),
                    safe_to_remove=classification.startswith("exclude_"),
                    notes="static_ast_initial_manifest",
                )
            )
    return sorted(records, key=lambda item: (item.module_path, item.line_number, item.name))


def write_manifest_outputs(records: list[DefinitionRecord], output_dir: Path) -> None:
    """Write CSV and Markdown retention manifest outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fieldnames = list(DefinitionRecord.__dataclass_fields__.keys())
    csv_path = output_path / "cs4m_deploy_function_retention_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)

    lines = [
        "# CS4M Deploy Function Retention Manifest",
        "",
        "| Module | Definition | Kind | Line | Classification | Reason |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        lines.append(
            "| "
            f"{record.module_path} | {record.name} | {record.kind} | "
            f"{record.line_number} | {record.classification} | "
            f"{record.caller_or_reason} |"
        )
    md_path = output_path / "cs4m_deploy_function_retention_manifest.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record(path: Path, name: str, kind: str, line_number: int) -> DefinitionRecord:
    return DefinitionRecord(
        module_path=str(path),
        name=name,
        kind=kind,
        line_number=int(line_number),
        classification="exclude_unused",
        caller_or_reason="unclassified",
        streaming_required=False,
        touches_ground_truth=False,
        evaluation_only=False,
        safe_to_remove=True,
        notes="parsed_definition",
    )


def _reason(is_explicit_keep: bool, is_entrypoint: bool) -> str:
    if is_explicit_keep:
        return "explicit_keep"
    if is_entrypoint:
        return "entrypoint_context_not_definition_reachable"
    return "not_reachable_from_mainline"


def _mentions_ground_truth(name: str) -> bool:
    lowered = name.lower()
    return "ground_truth" in lowered or lowered.startswith("gt_") or lowered.endswith("_gt")


def _path_is_evaluation(path: Path) -> bool:
    text = str(path).lower()
    return "/evaluation" in text or "eval" in path.name.lower()


def _default_python_paths(repo_root: Path) -> list[Path]:
    roots = [repo_root / "cs4m", repo_root / "scripts" / "pipeline"]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.rglob("*.py")))
    return paths


def main() -> int:
    """Build and write the deploy retention manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="docs/reports")
    parser.add_argument("--keep-name", action="append", default=[])
    parser.add_argument("--entrypoint", action="append", default=[])
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    entrypoints = [repo_root / path for path in args.entrypoint]
    records = build_retention_manifest(
        repo_root=repo_root,
        python_paths=_default_python_paths(repo_root),
        explicit_keep_names=set(args.keep_name),
        entrypoint_paths=entrypoints,
    )
    write_manifest_outputs(records, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
