"""Read-only OpTC Windows canonical-aware evaluation audit."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_INFER_DIR = Path(
    "outputs/results/tflr_light/optc_windows_v1_3c_full/"
    "OPTC_051_OPTC_WINDOWS_V1_3C_DETAIL_FULL_INFER",
)
DEFAULT_ARTIFACT_DIR = Path(
    "outputs/cache/phase3e_optc_windows_v1_3c/"
    "OPTC_051_OPTC_WINDOWS_V1_3C_DETAIL_full",
)


@dataclass(frozen=True)
class GtNode:
    """One original ground-truth node row."""

    uuid: str
    kind: str
    original_node_id: int
    raw_detail: str


@dataclass(frozen=True)
class CanonicalGtNode:
    """One GT node mapped to the compact runtime node identity."""

    uuid: str
    kind: str
    original_node_id: int
    lookup_node_id: int
    compact_node_idx: int | None
    mapping_status: str
    raw_detail: str


def infer_gt_kind(raw_detail: object) -> str:
    """Infer GT entity kind from the OpTC ground-truth detail cell."""
    text = str(raw_detail or "").lower()
    if "netflow" in text:
        return "netflow"
    if "subject" in text:
        return "process"
    if "file" in text:
        return "file"
    return "other"


def parse_gt_rows(path: Path) -> list[GtNode]:
    """Parse OpTC ground-truth CSV rows."""
    rows: list[GtNode] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                node_id = int(row[2])
            except ValueError:
                continue
            rows.append(
                GtNode(
                    uuid=str(row[0]),
                    kind=infer_gt_kind(row[1]),
                    original_node_id=node_id,
                    raw_detail=str(row[1]),
                ),
            )
    return rows


def load_original_to_canonical_netflow(path: Path) -> dict[int, int]:
    """Load original netflow node id to canonical node id mapping."""
    mapping: dict[int, int] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                mapping[int(row["original_node_id"])] = int(row["canonical_node_id"])
            except (KeyError, TypeError, ValueError):
                continue
    return mapping


def load_node_id_to_idx(path: Path) -> dict[int, int]:
    """Load Phase3E node id to compact node index mapping."""
    with Path(path).open("rb") as handle:
        loaded = pickle.load(handle)
    return {int(node_id): int(index) for node_id, index in dict(loaded).items()}


def build_canonical_gt_mapping(
    gt_nodes: Iterable[GtNode],
    original_to_canonical_netflow: Mapping[int, int],
    node_id_to_idx: Mapping[int, int],
) -> list[CanonicalGtNode]:
    """Map original GT nodes to canonical compact node identities."""
    mapped: list[CanonicalGtNode] = []
    for node in gt_nodes:
        lookup_node_id = int(node.original_node_id)
        if node.kind == "netflow":
            lookup_node_id = int(
                original_to_canonical_netflow.get(node.original_node_id, node.original_node_id),
            )
        compact_node_idx = node_id_to_idx.get(lookup_node_id)
        status = "mapped" if compact_node_idx is not None else "missing_node_id_to_idx"
        mapped.append(
            CanonicalGtNode(
                uuid=node.uuid,
                kind=node.kind,
                original_node_id=node.original_node_id,
                lookup_node_id=lookup_node_id,
                compact_node_idx=int(compact_node_idx) if compact_node_idx is not None else None,
                mapping_status=status,
                raw_detail=node.raw_detail,
            ),
        )
    return mapped


def load_node_pool(path: Path) -> set[int]:
    """Load compact node ids from a node pool CSV."""
    nodes: set[int] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        id_field = "node_idx" if "node_idx" in fieldnames else "node_id"
        for row in reader:
            try:
                nodes.add(int(row[id_field]))
            except (KeyError, TypeError, ValueError):
                continue
    return nodes


def summarize_gt_hits(
    mapped_gt: Iterable[CanonicalGtNode],
    node_pool: set[int],
) -> tuple[dict[str, object], list[CanonicalGtNode], list[CanonicalGtNode]]:
    """Summarize original GT hits and misses by kind."""
    mapped_rows = list(mapped_gt)
    hit_rows: list[CanonicalGtNode] = []
    missed_rows: list[CanonicalGtNode] = []
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "hit": 0, "missed": 0})
    compact_by_kind: dict[str, set[int]] = defaultdict(set)
    compact_hit_by_kind: dict[str, set[int]] = defaultdict(set)

    for row in mapped_rows:
        by_kind[row.kind]["total"] += 1
        if row.compact_node_idx is not None:
            compact_by_kind[row.kind].add(row.compact_node_idx)
        hit = row.compact_node_idx is not None and row.compact_node_idx in node_pool
        if hit:
            hit_rows.append(row)
            by_kind[row.kind]["hit"] += 1
            compact_hit_by_kind[row.kind].add(int(row.compact_node_idx))
        else:
            missed_rows.append(row)
            by_kind[row.kind]["missed"] += 1

    compact_all = {row.compact_node_idx for row in mapped_rows if row.compact_node_idx is not None}
    compact_hits = {row.compact_node_idx for row in hit_rows if row.compact_node_idx is not None}
    summary = {
        "original_gt_total": len(mapped_rows),
        "original_gt_hit": len(hit_rows),
        "original_gt_missed": len(missed_rows),
        "original_gt_hit_ratio": float(len(hit_rows) / len(mapped_rows)) if mapped_rows else 0.0,
        "unique_compact_gt_total": len(compact_all),
        "unique_compact_gt_hit": len(compact_hits),
        "unique_compact_gt_missed": len(compact_all - compact_hits),
        "unique_compact_gt_hit_ratio": (
            float(len(compact_hits) / len(compact_all)) if compact_all else 0.0
        ),
        "by_kind": {
            kind: {
                **counts,
                "hit_ratio": float(counts["hit"] / counts["total"]) if counts["total"] else 0.0,
                "unique_compact_total": len(compact_by_kind.get(kind, set())),
                "unique_compact_hit": len(compact_hit_by_kind.get(kind, set())),
            }
            for kind, counts in sorted(by_kind.items())
        },
    }
    return summary, hit_rows, missed_rows


def write_counter_csv(path: Path, fieldnames: list[str], counter: Mapping[object, int]) -> None:
    """Write a counter-like mapping to CSV with ratios."""
    total = sum(int(value) for value in counter.values())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*fieldnames, "count", "ratio"])
        for key, count in sorted(counter.items(), key=lambda item: (-int(item[1]), str(item[0]))):
            values = key if isinstance(key, tuple) else (key,)
            writer.writerow([*values, int(count), float(int(count) / total) if total else 0.0])


def _write_gt_rows(path: Path, rows: Iterable[CanonicalGtNode], *, include_hit: bool = False) -> None:
    fieldnames = [
        "uuid",
        "kind",
        "original_node_id",
        "lookup_node_id",
        "compact_node_idx",
        "mapping_status",
    ]
    if include_hit:
        fieldnames.append("pool_hit")
    fieldnames.append("raw_detail")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {
                "uuid": row.uuid,
                "kind": row.kind,
                "original_node_id": row.original_node_id,
                "lookup_node_id": row.lookup_node_id,
                "compact_node_idx": "" if row.compact_node_idx is None else row.compact_node_idx,
                "mapping_status": row.mapping_status,
                "raw_detail": row.raw_detail,
            }
            if include_hit:
                payload["pool_hit"] = ""
            writer.writerow(payload)


def _write_mapping_rows(
    path: Path,
    rows: Iterable[CanonicalGtNode],
    node_pool: set[int],
) -> None:
    fieldnames = [
        "uuid",
        "kind",
        "original_node_id",
        "lookup_node_id",
        "compact_node_idx",
        "mapping_status",
        "pool_hit",
        "raw_detail",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            hit = row.compact_node_idx is not None and row.compact_node_idx in node_pool
            writer.writerow(
                {
                    "uuid": row.uuid,
                    "kind": row.kind,
                    "original_node_id": row.original_node_id,
                    "lookup_node_id": row.lookup_node_id,
                    "compact_node_idx": (
                        "" if row.compact_node_idx is None else row.compact_node_idx
                    ),
                    "mapping_status": row.mapping_status,
                    "pool_hit": hit,
                    "raw_detail": row.raw_detail,
                },
            )


def _write_node_pool_eval(
    path: Path,
    node_pool: set[int],
    mapped_gt: Iterable[CanonicalGtNode],
) -> None:
    labels_by_compact: dict[int, list[CanonicalGtNode]] = defaultdict(list)
    for row in mapped_gt:
        if row.compact_node_idx is not None:
            labels_by_compact[int(row.compact_node_idx)].append(row)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compact_node_idx",
                "canonical_gt_label",
                "gt_original_count",
                "gt_kinds",
                "gt_original_node_ids",
            ],
        )
        writer.writeheader()
        for node_idx in sorted(node_pool):
            rows = labels_by_compact.get(node_idx, [])
            writer.writerow(
                {
                    "compact_node_idx": node_idx,
                    "canonical_gt_label": "malicious" if rows else "benign",
                    "gt_original_count": len(rows),
                    "gt_kinds": ";".join(sorted({row.kind for row in rows})),
                    "gt_original_node_ids": ";".join(str(row.original_node_id) for row in rows),
                },
            )


def _event_alert_summary(
    alerts_path: Path,
    mapped_gt: Iterable[CanonicalGtNode],
) -> tuple[Counter, dict[str, int]]:
    compact_gt = {
        int(row.compact_node_idx)
        for row in mapped_gt
        if row.compact_node_idx is not None
    }
    counter: Counter = Counter()
    totals = {"alerts": 0, "tp": 0, "fp": 0}
    with Path(alerts_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            totals["alerts"] += 1
            try:
                src = int(row.get("info_src", row.get("src_idx", -1)))
                dst = int(row.get("info_dst", row.get("dst_idx", -1)))
            except ValueError:
                src = -1
                dst = -1
            is_tp = src in compact_gt or dst in compact_gt
            totals["tp" if is_tp else "fp"] += 1
            key = (
                row.get("action", ""),
                row.get("src_type", ""),
                row.get("dst_type", ""),
                row.get("object_type", ""),
                "TP" if is_tp else "FP",
            )
            counter[key] += 1
    return counter, totals


def _write_event_summary(path: Path, counter: Counter) -> None:
    total = sum(counter.values())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action", "src_type", "dst_type", "object_type", "eval_result", "count", "ratio"])
        for key, count in counter.most_common():
            writer.writerow([*key, int(count), float(count / total) if total else 0.0])


def _load_reported_strict_tp(eval_json_path: Path) -> int | None:
    if not Path(eval_json_path).exists():
        return None
    data = json.loads(Path(eval_json_path).read_text(encoding="utf-8"))
    try:
        return int(data["primary_online_metrics"]["online_event_node_coverage"]["strict_node_tp"])
    except (KeyError, TypeError, ValueError):
        return None


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    """Run canonical-aware audit and write diagnostics."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_nodes = parse_gt_rows(Path(args.gt_csv))
    original_to_canonical = load_original_to_canonical_netflow(Path(args.original_to_canonical))
    node_id_to_idx = load_node_id_to_idx(Path(args.node_id_to_idx))
    mapped_gt = build_canonical_gt_mapping(gt_nodes, original_to_canonical, node_id_to_idx)

    node_pool = load_node_pool(Path(args.online_event_node_coverage))
    online_node_pool = load_node_pool(Path(args.online_node_alerts))
    combined_pool = set(node_pool) | set(online_node_pool)
    summary, hit_rows, missed_rows = summarize_gt_hits(mapped_gt, combined_pool)
    event_counter, event_totals = _event_alert_summary(Path(args.online_event_alerts), mapped_gt)

    _write_mapping_rows(out_dir / "canonical_gt_mapping.csv", mapped_gt, combined_pool)
    _write_node_pool_eval(out_dir / "canonical_node_pool_eval.csv", combined_pool, mapped_gt)
    _write_gt_rows(out_dir / "gt_hit_by_kind.csv", hit_rows)
    _write_gt_rows(out_dir / "gt_missed_by_kind.csv", missed_rows)
    _write_event_summary(out_dir / "event_alert_tp_fp_by_action_kind.csv", event_counter)

    reported_strict_tp = _load_reported_strict_tp(Path(args.eval_json))
    summary_payload = {
        "dataset": args.dataset,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "reported_current_strict_node_tp": reported_strict_tp,
        "canonical_aware": summary,
        "event_alerts_by_canonical_gt": event_totals,
        "inputs": {
            "gt_csv": str(args.gt_csv),
            "original_to_canonical": str(args.original_to_canonical),
            "node_id_to_idx": str(args.node_id_to_idx),
            "online_event_node_coverage": str(args.online_event_node_coverage),
            "online_node_alerts": str(args.online_node_alerts),
            "online_event_alerts": str(args.online_event_alerts),
            "eval_json": str(args.eval_json),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_report(out_dir / "OPTC_051_v13c_full_gate_canonical_eval_report.md", summary_payload)
    return summary_payload


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    canonical = dict(summary.get("canonical_aware", {}))
    by_kind = dict(canonical.get("by_kind", {}))
    event_totals = dict(summary.get("event_alerts_by_canonical_gt", {}))
    lines = [
        "# OPTC 051 v1.3c Full Gate Canonical Evaluation Audit",
        "",
        f"- reported current strict node TP: {summary.get('reported_current_strict_node_tp')}",
        f"- canonical-aware original GT hit: {canonical.get('original_gt_hit')}/{canonical.get('original_gt_total')}",
        f"- canonical-aware unique compact GT hit: {canonical.get('unique_compact_gt_hit')}/{canonical.get('unique_compact_gt_total')}",
        f"- event alerts by canonical GT: TP={event_totals.get('tp')} FP={event_totals.get('fp')} total={event_totals.get('alerts')}",
        "",
        "## By Kind",
    ]
    for kind, counts in sorted(by_kind.items()):
        lines.append(
            f"- {kind}: {counts.get('hit')}/{counts.get('total')} original GT hit, "
            f"{counts.get('unique_compact_hit')}/{counts.get('unique_compact_total')} unique compact hit",
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "If netflow original GT hit is complete, netflow semantic canonicalization is not the next bottleneck.",
            "Remaining process/file misses and event FP should be handled in separate audits or policy ablations.",
        ],
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="OPTC_051")
    parser.add_argument("--gt-csv", default="ground_truth/h051/node_h051_0925.csv")
    parser.add_argument(
        "--original-to-canonical",
        default=str(DEFAULT_ARTIFACT_DIR / "event_index/original_to_canonical_netflow.csv"),
    )
    parser.add_argument(
        "--node-id-to-idx",
        default=str(DEFAULT_ARTIFACT_DIR / "node/node_id_to_idx.pkl"),
    )
    parser.add_argument(
        "--online-event-node-coverage",
        default=str(DEFAULT_INFER_DIR / "online_event_node_coverage.csv"),
    )
    parser.add_argument(
        "--online-node-alerts",
        default=str(DEFAULT_INFER_DIR / "online_node_alerts.csv"),
    )
    parser.add_argument(
        "--online-event-alerts",
        default=str(DEFAULT_INFER_DIR / "online_event_alerts.csv"),
    )
    parser.add_argument(
        "--eval-json",
        default=str(DEFAULT_INFER_DIR / "eval_causal_semantics_slim.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/diagnostics/optc_windows_v13c_canonical_eval_audit_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run CLI entrypoint."""
    summary = run_audit(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
