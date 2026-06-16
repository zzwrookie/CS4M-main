"""Audit OPTC_201 ground-truth node alignment against active event-index splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_ARTIFACT_DIR = Path(
    "outputs/cache/phase3e_optc_windows_v1_3c/"
    "OPTC_201_OPTC_WINDOWS_V1_3C_DETAIL_full",
)
DEFAULT_CANONICAL_AUDIT_DIR = Path(
    "outputs/diagnostics/optc_windows_v13c_canonical_eval_audit_OPTC_201_20260616_095827",
)
DEFAULT_GT_PATH = Path("ground_truth/h201/node_h201_0923.csv")
EVENT_INDEX_DTYPE = np.dtype(
    [
        ("event_id", "<i8"),
        ("src_node_idx", "<i4"),
        ("dst_node_idx", "<i4"),
        ("action_id", "<i2"),
        ("src_type_id", "i1"),
        ("dst_type_id", "i1"),
    ],
)


def split_presence_status(counts: Mapping[str, int]) -> str:
    """Classify where a GT compact node appears across train/validation/test."""
    train = int(counts.get("train", 0) or 0)
    validation = int(counts.get("validation", 0) or 0)
    test = int(counts.get("test", 0) or 0)
    if test > 0 and (train > 0 or validation > 0):
        return "mixed_with_test"
    if test > 0:
        return "test_present"
    if train > 0 and validation > 0:
        return "train_validation_only"
    if validation > 0:
        return "validation_only"
    if train > 0:
        return "train_only"
    return "not_in_event_index"


def _latest_canonical_audit_dir() -> Path:
    candidates = sorted(
        path
        for path in Path("outputs/diagnostics").glob(
            "optc_windows_v13c_canonical_eval_audit_OPTC_201_*",
        )
        if path.is_dir()
    )
    return candidates[-1] if candidates else DEFAULT_CANONICAL_AUDIT_DIR


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _load_event_index_meta(artifact_dir: Path) -> dict[str, Any]:
    path = Path(artifact_dir) / "event_index" / "event_index_meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _split_counts_by_compact_node(meta: Mapping[str, Any]) -> dict[int, dict[str, int]]:
    counts: dict[int, dict[str, int]] = defaultdict(
        lambda: {"train": 0, "validation": 0, "test": 0},
    )
    for split, payload in sorted(dict(meta.get("splits", {})).items()):
        path = Path(str(payload.get("path", "")))
        total = int(payload.get("num_events", 0) or 0)
        if split not in {"train", "validation", "test"} or not path.exists() or total <= 0:
            continue
        arr = np.memmap(path, dtype=EVENT_INDEX_DTYPE, mode="r", shape=(total,))
        split_counter: Counter[int] = Counter()
        split_counter.update(int(value) for value in arr["src_node_idx"])
        split_counter.update(int(value) for value in arr["dst_node_idx"])
        for node_idx, count in split_counter.items():
            counts[int(node_idx)][split] = int(count)
        del arr
    return counts


def _summary_rows(rows: Iterable[Mapping[str, Any]], group_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "row_count": 0,
            "unique_compact_nodes": set(),
            "train_events": 0,
            "validation_events": 0,
            "test_events": 0,
        },
    )
    for row in rows:
        key = str(row.get(group_field, ""))
        item = grouped[key]
        item["row_count"] = int(item["row_count"]) + 1
        compact = str(row.get("compact_node_idx", ""))
        if compact:
            item["unique_compact_nodes"].add(compact)
        item["train_events"] = int(item["train_events"]) + int(row.get("train_events", 0) or 0)
        item["validation_events"] = int(item["validation_events"]) + int(
            row.get("validation_events", 0) or 0,
        )
        item["test_events"] = int(item["test_events"]) + int(row.get("test_events", 0) or 0)
    out: list[dict[str, Any]] = []
    for key, item in sorted(grouped.items()):
        out.append(
            {
                group_field: key,
                "row_count": int(item["row_count"]),
                "unique_compact_nodes": len(item["unique_compact_nodes"]),
                "train_events": int(item["train_events"]),
                "validation_events": int(item["validation_events"]),
                "test_events": int(item["test_events"]),
            },
        )
    return out


def build_alignment_rows(
    canonical_rows: Iterable[Mapping[str, str]],
    split_counts: Mapping[int, Mapping[str, int]],
) -> list[dict[str, Any]]:
    """Attach split presence counts to canonical GT mapping rows."""
    rows: list[dict[str, Any]] = []
    for row in canonical_rows:
        compact_text = str(row.get("compact_node_idx", "")).strip()
        counts = {"train": 0, "validation": 0, "test": 0}
        if compact_text:
            try:
                counts.update(split_counts.get(int(compact_text), {}))
            except ValueError:
                pass
        rows.append(
            {
                **dict(row),
                "train_events": int(counts.get("train", 0) or 0),
                "validation_events": int(counts.get("validation", 0) or 0),
                "test_events": int(counts.get("test", 0) or 0),
                "split_presence_status": split_presence_status(counts),
            },
        )
    return rows


def run_audit(
    *,
    output_dir: Path,
    artifact_dir: Path,
    canonical_audit_dir: Path,
    gt_path: Path,
) -> dict[str, Any]:
    """Run the read-only split alignment audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = canonical_audit_dir / "canonical_gt_mapping.csv"
    canonical_rows = _read_csv_rows(mapping_path)
    split_counts = _split_counts_by_compact_node(_load_event_index_meta(artifact_dir))
    alignment_rows = build_alignment_rows(canonical_rows, split_counts)
    by_kind = _summary_rows(alignment_rows, "kind")
    by_status = _summary_rows(alignment_rows, "split_presence_status")

    _write_csv(
        output_dir / "gt_split_alignment_rows.csv",
        alignment_rows,
        [
            "uuid",
            "kind",
            "original_node_id",
            "lookup_node_id",
            "compact_node_idx",
            "mapping_status",
            "pool_hit",
            "train_events",
            "validation_events",
            "test_events",
            "split_presence_status",
            "raw_detail",
        ],
    )
    _write_csv(
        output_dir / "gt_split_alignment_summary_by_kind.csv",
        by_kind,
        [
            "kind",
            "row_count",
            "unique_compact_nodes",
            "train_events",
            "validation_events",
            "test_events",
        ],
    )
    _write_csv(
        output_dir / "gt_split_alignment_summary_by_status.csv",
        by_status,
        [
            "split_presence_status",
            "row_count",
            "unique_compact_nodes",
            "train_events",
            "validation_events",
            "test_events",
        ],
    )
    status_counts = Counter(str(row["split_presence_status"]) for row in alignment_rows)
    kind_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in alignment_rows:
        kind_status_counts[str(row.get("kind", ""))][str(row["split_presence_status"])] += 1
    dominant_endpoint_rows = [
        row for row in alignment_rows if "132.197.158.98" in str(row.get("raw_detail", ""))
    ]
    summary = {
        "dataset": "OPTC_201",
        "ground_truth_path": str(gt_path),
        "canonical_mapping_path": str(mapping_path),
        "artifact_dir": str(artifact_dir),
        "gt_row_count": int(len(alignment_rows)),
        "status_counts": dict(sorted(status_counts.items())),
        "kind_status_counts": {
            kind: dict(sorted(counter.items()))
            for kind, counter in sorted(kind_status_counts.items())
        },
        "dominant_endpoint_132_197_158_98_rows": int(len(dominant_endpoint_rows)),
        "dominant_endpoint_status_counts": dict(
            sorted(
                Counter(
                    str(row["split_presence_status"]) for row in dominant_endpoint_rows
                ).items(),
            ),
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir, summary, by_status, by_kind)
    return summary


def _write_report(
    output_dir: Path,
    summary: Mapping[str, Any],
    by_status: Iterable[Mapping[str, Any]],
    by_kind: Iterable[Mapping[str, Any]],
) -> None:
    lines = [
        "# OPTC_201 GT Split Alignment Report",
        "",
        f"- GT file: `{summary['ground_truth_path']}`",
        f"- canonical mapping: `{summary['canonical_mapping_path']}`",
        f"- GT rows: `{summary['gt_row_count']}`",
        f"- dominant endpoint rows: `{summary['dominant_endpoint_132_197_158_98_rows']}`",
        f"- dominant endpoint status: `{summary['dominant_endpoint_status_counts']}`",
        "",
        "## By Status",
        "",
        "| status | rows | unique compact nodes | train events | validation events | test events |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in by_status:
        lines.append(
            f"| {row.get('split_presence_status', '')} | {row.get('row_count', 0)} | "
            f"{row.get('unique_compact_nodes', 0)} | {row.get('train_events', 0)} | "
            f"{row.get('validation_events', 0)} | {row.get('test_events', 0)} |",
        )
    lines.extend(
        [
            "",
            "## By Kind",
            "",
            "| kind | rows | unique compact nodes | train events | validation events | "
            "test events |",
            "|---|---:|---:|---:|---:|---:|",
        ],
    )
    for row in by_kind:
        lines.append(
            f"| {row.get('kind', '')} | {row.get('row_count', 0)} | "
            f"{row.get('unique_compact_nodes', 0)} | {row.get('train_events', 0)} | "
            f"{row.get('validation_events', 0)} | {row.get('test_events', 0)} |",
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Rows marked `validation_only` cannot be recovered by changing test-time thresholds, "
            "because the corresponding compact node never appears in the active test event index.",
        ],
    )
    (output_dir / "OPTC_201_GT_Split_Alignment_Report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--canonical-audit-dir", type=Path, default=None)
    parser.add_argument("--gt-path", type=Path, default=DEFAULT_GT_PATH)
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/diagnostics/optc201_gt_split_alignment_{stamp}")
    summary = run_audit(
        output_dir=output_dir,
        artifact_dir=args.artifact_dir,
        canonical_audit_dir=args.canonical_audit_dir or _latest_canonical_audit_dir(),
        gt_path=args.gt_path,
    )
    print(json.dumps({"output_dir": str(output_dir), **summary}, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
