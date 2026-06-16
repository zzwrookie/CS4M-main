"""Read-only CADETS_E3 E4 FreeBSD semantic audit.

The audit inspects current CADETS FreeBSD semantic tokens and existing E4
artifacts. It does not train models, run inference, modify tokenizer code, or
create persistent database objects.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.semantics.cadets_freebsd import (  # noqa: E402
    CADETS_SEMANTIC_RULES_VERSION,
    freebsd_file_natural_tokens,
    freebsd_netflow_natural_tokens,
    freebsd_process_natural_tokens,
)
from scripts.pipeline.config.runtime_config import SUBJECT_NODE_TABLE  # noqa: E402


PROCESS_GENERIC = {
    "unknown_process",
    "generic_process",
    "process_other",
    "other_process",
    "unknown",
    "other",
}
FILE_GENERIC = {"file_other", "other_ext", "other_file", "unknown_file", "unknown", "other"}
NETFLOW_GENERIC = {"public", "registered", "unknown_ip"}
PAYLOAD_LIKE_DETAILS = {
    "main",
    "test",
    "tmux_1002",
    "vugefal",
    "peja72ma",
    "minions",
    "xim",
    "gtcache",
    "pass_mgr",
    "clean",
    "profile",
    "wdev",
    "xdev",
    "memtrace_so",
}


def token_key(tokens: Sequence[object]) -> str:
    """Return a stable pipe-delimited token key."""
    return "|".join(str(token) for token in tokens)


def token_is_generic(kind: str, tokens: Sequence[str]) -> bool:
    """Return whether current audit considers a token tuple generic."""
    token_set = set(tokens)
    if kind == "process":
        return bool(token_set & PROCESS_GENERIC)
    if kind == "file":
        return tuple(tokens) == ("file",) or bool(token_set & FILE_GENERIC)
    if kind == "netflow":
        return tuple(tokens) == ("netflow",) or bool(token_set & NETFLOW_GENERIC)
    return False


def has_exact_ip_token(tokens: Sequence[str]) -> bool:
    """Return whether a token tuple contains a normalized exact IP token."""
    for token in tokens:
        parts = str(token).split("_")
        if len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
            return True
    return False


def classify_focus_flags(kind: str, tokens: Sequence[str], raw_detail: object) -> tuple[str, ...]:
    """Return audit focus flags for one semantic tuple."""
    token_set = set(tokens)
    detail = str(raw_detail or "").strip().lower()
    flags: list[str] = []
    if kind == "process":
        if "process_other" in token_set:
            flags.append("process_other")
        if "unknown_process" in token_set:
            flags.append("unknown_process")
        if any(token in PAYLOAD_LIKE_DETAILS for token in token_set):
            flags.append("payload_like_detail")
    elif kind == "file":
        if tuple(tokens) == ("file",):
            flags.append("empty_file_tuple")
            flags.append("payload_like_detail")
        if "file_other" in token_set:
            flags.append("file_other")
        if "unknown_file" in token_set:
            flags.append("unknown_file")
        if "tmp_file" in token_set or detail.startswith("/tmp/"):
            flags.append("tmp_file")
        if "user_home_file" in token_set or detail.startswith(("/usr/home/", "/home/")):
            flags.append("user_home_file")
        if "system_log_file" in token_set or detail.startswith("/var/log/"):
            flags.append("system_log_file")
        if "device_file" in token_set and ("random" in token_set or detail == "/dev/random"):
            flags.append("device_random")
        if any(token in PAYLOAD_LIKE_DETAILS for token in token_set):
            flags.append("payload_like_detail")
    elif kind == "netflow":
        if has_exact_ip_token(tokens):
            flags.append("exact_ip_retained")
        if "ip_public_or_external" in token_set:
            flags.append("public_or_external")
    return tuple(dict.fromkeys(flags))


@dataclass
class AuditAccumulator:
    """Aggregate semantic audit statistics with GT used only post-hoc."""

    dataset: str
    gt_nodes: set[int]
    kind_counts: Counter[str] = field(default_factory=Counter)
    generic_counts: Counter[str] = field(default_factory=Counter)
    exact_ip_counts: Counter[str] = field(default_factory=Counter)
    gt_kind_counts: Counter[str] = field(default_factory=Counter)
    gt_generic_counts: Counter[str] = field(default_factory=Counter)
    collision_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    gt_collision_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    token_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    gt_token_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    focus_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    gt_focus_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    sample_node: dict[tuple[str, str], int] = field(default_factory=dict)
    sample_raw_detail: dict[tuple[str, str], str] = field(default_factory=dict)

    def add_node(
        self,
        *,
        node_id: int,
        kind: str,
        tokens: Sequence[object],
        raw_detail: object,
    ) -> None:
        """Add one node's current semantic tuple to the audit."""
        norm_tokens = tuple(str(token) for token in tokens if str(token).strip())
        key = token_key(norm_tokens)
        is_gt = int(node_id) in self.gt_nodes
        is_generic = token_is_generic(kind, norm_tokens)
        exact_ip = has_exact_ip_token(norm_tokens)
        sample_key = (kind, key)

        self.kind_counts[kind] += 1
        self.generic_counts[kind] += int(is_generic)
        self.exact_ip_counts[kind] += int(exact_ip)
        self.collision_counts[sample_key] += 1
        self.sample_node.setdefault(sample_key, int(node_id))
        self.sample_raw_detail.setdefault(sample_key, str(raw_detail or ""))
        for token in norm_tokens:
            self.token_counts[(kind, token)] += 1

        for flag in classify_focus_flags(kind, norm_tokens, raw_detail):
            self.focus_counts[(kind, flag)] += 1
            if is_gt:
                self.gt_focus_counts[(kind, flag)] += 1

        if is_gt:
            self.gt_kind_counts[kind] += 1
            self.gt_generic_counts[kind] += int(is_generic)
            self.gt_collision_counts[sample_key] += 1
            for token in norm_tokens:
                self.gt_token_counts[(kind, token)] += 1

    def summary_rows(self) -> list[dict[str, object]]:
        """Return per-kind summary rows."""
        rows = []
        for kind in ("process", "file", "netflow"):
            total = self.kind_counts[kind]
            generic = self.generic_counts[kind]
            gt_total = self.gt_kind_counts[kind]
            gt_generic = self.gt_generic_counts[kind]
            rows.append(
                {
                    "dataset": self.dataset,
                    "semantic_mode": CADETS_SEMANTIC_RULES_VERSION,
                    "kind": kind,
                    "total_nodes": total,
                    "generic_nodes": generic,
                    "generic_ratio": generic / total if total else 0.0,
                    "exact_ip_nodes": self.exact_ip_counts[kind],
                    "exact_ip_ratio": self.exact_ip_counts[kind] / total if total else 0.0,
                    "gt_nodes": gt_total,
                    "gt_generic_nodes": gt_generic,
                    "gt_generic_ratio": gt_generic / gt_total if gt_total else 0.0,
                    "focus_flags": ";".join(
                        f"{flag}:{count}"
                        for (flag_kind, flag), count in sorted(self.focus_counts.items())
                        if flag_kind == kind
                    ),
                    "gt_focus_flags": ";".join(
                        f"{flag}:{count}"
                        for (flag_kind, flag), count in sorted(self.gt_focus_counts.items())
                        if flag_kind == kind
                    ),
                }
            )
        return rows

    def collision_rows(self, kind: str, limit: int) -> list[dict[str, object]]:
        """Return top collision rows for one kind."""
        rows = []
        total = self.kind_counts[kind]
        for (row_kind, key), count in self.collision_counts.most_common():
            if row_kind != kind:
                continue
            sample_key = (row_kind, key)
            gt_count = self.gt_collision_counts[sample_key]
            rows.append(
                {
                    "dataset": self.dataset,
                    "kind": kind,
                    "token_tuple": key,
                    "node_count": count,
                    "ratio_within_kind": count / total if total else 0.0,
                    "gt_node_count": gt_count,
                    "gt_node_ratio_within_tuple": gt_count / count if count else 0.0,
                    "sample_node_id": self.sample_node.get(sample_key, ""),
                    "sample_raw_detail": self.sample_raw_detail.get(sample_key, ""),
                    "focus_flags": "|".join(
                        classify_focus_flags(kind, tuple(key.split("|")), self.sample_raw_detail.get(sample_key, ""))
                    ),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def gt_focus_rows(self, limit: int) -> list[dict[str, object]]:
        """Return GT token and tuple focus rows."""
        rows = []
        for (kind, token), count in self.gt_token_counts.most_common(limit):
            gt_total = self.gt_kind_counts[kind]
            rows.append(
                {
                    "dataset": self.dataset,
                    "row_type": "gt_focus_token",
                    "kind": kind,
                    "token_or_tuple": token,
                    "count": count,
                    "ratio_within_gt_kind": count / gt_total if gt_total else 0.0,
                    "sample_gt_node_id": "",
                }
            )
        for (kind, key), count in self.gt_collision_counts.most_common(limit):
            gt_total = self.gt_kind_counts[kind]
            rows.append(
                {
                    "dataset": self.dataset,
                    "row_type": "gt_collision_tuple",
                    "kind": kind,
                    "token_or_tuple": key,
                    "count": count,
                    "ratio_within_gt_kind": count / gt_total if gt_total else 0.0,
                    "sample_gt_node_id": self.sample_node.get((kind, key), ""),
                }
            )
        return rows


def build_candidate_design_rows() -> list[dict[str, object]]:
    """Return label-free semantic candidate rows for the report."""
    return [
        {
            "candidate_name": "process_other_safe_lexical_detail",
            "priority": 1,
            "kind": "process",
            "problem": "process_other can absorb payload-like FreeBSD commands.",
            "candidate": "Keep process_other role but add safe lexical shape/detail buckets.",
            "runtime_visible": True,
            "uses_gt_for_rule": False,
            "requires_word2vec_retrain": True,
        },
        {
            "candidate_name": "interpreter_argument_detail_stabilization",
            "priority": 2,
            "kind": "process",
            "problem": "Interpreter and shell commands may need stable meaningful argument detail.",
            "candidate": "Audit command args using path/basename shape without raw label-derived strings.",
            "runtime_visible": True,
            "uses_gt_for_rule": False,
            "requires_word2vec_retrain": True,
        },
        {
            "candidate_name": "file_empty_bucket_refinement",
            "priority": 3,
            "kind": "file",
            "problem": "Empty file tuples collapse many nodes into file-only semantics.",
            "candidate": "Explain missing path source and split known missing/special file cases.",
            "runtime_visible": True,
            "uses_gt_for_rule": False,
            "requires_word2vec_retrain": True,
        },
        {
            "candidate_name": "tmp_home_varlog_detail_refinement",
            "priority": 4,
            "kind": "file",
            "problem": "/tmp, /usr/home, and /var/log details may hold useful behavior context.",
            "candidate": "Preserve bounded basename or parent+basename detail for these families.",
            "runtime_visible": True,
            "uses_gt_for_rule": False,
            "requires_word2vec_retrain": True,
        },
        {
            "candidate_name": "netflow_endpoint_action_support_audit",
            "priority": 5,
            "kind": "netflow",
            "problem": "Exact IP is retained; semantic rewrite may not be first priority.",
            "candidate": "Defer semantic change and later audit endpoint/action support for policy.",
            "runtime_visible": True,
            "uses_gt_for_rule": False,
            "requires_word2vec_retrain": False,
        },
    ]


def render_markdown_report(
    *,
    output_dir: object,
    summary_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    missing_mapping_count: int,
    e4_artifact_dir: object,
) -> str:
    """Render the Markdown report body."""
    lines = [
        "# CADETS_E3 E4 FreeBSD Semantic Audit",
        "",
        "Scope: read-only semantic audit for the `CADETS_E3 + E4 dual` mainline.",
        "",
        "- No Word2Vec training.",
        "- No inference.",
        "- No runtime tokenizer modification.",
        "- No persistent database tables.",
        "- GT is post-hoc only and is not used to construct candidate rules.",
        "",
        f"Output directory: `{output_dir}`",
        f"E4 artifact directory: `{e4_artifact_dir}`",
        f"Missing GT mapping count: `{missing_mapping_count}`",
        "",
        "## Summary",
        "",
        "| Kind | Nodes | Generic | Generic Ratio | Exact-IP Ratio | GT Nodes | GT Generic Ratio | GT Focus Flags |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['kind']} | {row['total_nodes']} | {row['generic_nodes']} | "
            f"{float(row['generic_ratio']):.4f} | {float(row['exact_ip_ratio']):.4f} | "
            f"{row['gt_nodes']} | {float(row['gt_generic_ratio']):.4f} | "
            f"{row.get('gt_focus_flags', '')} |"
        )
    lines += [
        "",
        "## Candidate Semantic Design",
        "",
        "| Priority | Candidate | Kind | Runtime Visible | Uses GT For Rule | Requires Word2Vec Retrain |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in candidate_rows:
        lines.append(
            f"| {row['priority']} | {row['candidate_name']} | {row['kind']} | "
            f"{row['runtime_visible']} | {row['uses_gt_for_rule']} | "
            f"{row['requires_word2vec_retrain']} |"
        )
    lines += [
        "",
        "## Interpretation Checklist",
        "",
        "1. Prioritize process if `process_other` or payload-like GT focus is material.",
        "2. Prioritize file if empty `file`, `file_other`, `/tmp`, `/usr/home`, `/var/log`, "
        "or `/dev/random` buckets dominate collision rows.",
        "3. Defer netflow semantic rewrite if exact-IP ratio remains high; use endpoint/action "
        "support later during policy design.",
        "4. Promote a future `cadets_freebsd_raw_detail_v3_safe_lexical` mode only after "
        "candidate projection shows lower collision without label-derived logic.",
        "",
        "## Output Files",
        "",
        "- `cadets_e3_e4_semantic_audit_summary.csv`",
        "- `cadets_e3_e4_process_collision_tokens.csv`",
        "- `cadets_e3_e4_file_collision_tokens.csv`",
        "- `cadets_e3_e4_netflow_collision_tokens.csv`",
        "- `cadets_e3_e4_gt_focus_tokens.csv`",
        "- `cadets_e3_e4_candidate_design.csv`",
    ]
    return "\n".join(lines) + "\n"


def load_node_map(path: Path) -> dict[int, int]:
    """Load node id to index mapping."""
    with path.open("rb") as handle:
        return {int(key): int(value) for key, value in pickle.load(handle).items()}


def load_gt_nodes(gt_dir: Path) -> tuple[set[int], list[dict[str, object]]]:
    """Load CADETS GT node ids for post-hoc audit only."""
    nodes: set[int] = set()
    issues: list[dict[str, object]] = []
    if not gt_dir.exists():
        issues.append({"missing_type": "gt_dir_missing", "node_id": "", "detail": str(gt_dir)})
        return nodes, issues
    for path in sorted(gt_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) < 3:
                    continue
                try:
                    nodes.add(int(str(row[2]).strip()))
                except ValueError:
                    pass
    for path in sorted(gt_dir.glob("*.pkl")):
        try:
            payload = pickle.loads(path.read_bytes())
        except Exception as exc:  # pragma: no cover - defensive local artifact handling
            issues.append(
                {"missing_type": "gt_load_issue", "node_id": "", "detail": f"{path}:{type(exc).__name__}"}
            )
            continue
        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, (list, tuple, set)):
                stack.extend(item)
            else:
                try:
                    nodes.add(int(item))
                except Exception:
                    pass
    return nodes, issues


def connect(args: argparse.Namespace):
    """Open a PostgreSQL connection using CLI/env settings."""
    return psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        user=args.db_user,
        password=args.db_password,
        dbname=args.db_name,
    )


def stream_rows(conn, query: str, name: str, itersize: int = 50000):
    """Yield rows from a server-side cursor."""
    with conn.cursor(name=name) as cursor:
        cursor.itersize = itersize
        cursor.execute(query)
        yield from cursor


def scan_database(conn, used_node_ids: set[int], accumulator: AuditAccumulator) -> set[int]:
    """Stream CADETS node metadata and aggregate current semantic tokens."""
    seen: set[int] = set()
    query = f"select index_id, cmd from {SUBJECT_NODE_TABLE}"
    for index_id, cmd in stream_rows(conn, query, "cadets_semantic_audit_process"):
        node_id = int(index_id)
        if node_id not in used_node_ids:
            continue
        seen.add(node_id)
        accumulator.add_node(
            node_id=node_id,
            kind="process",
            tokens=freebsd_process_natural_tokens(cmd),
            raw_detail=cmd,
        )
    for index_id, path in stream_rows(
        conn,
        "select index_id, path from file_node_table",
        "cadets_semantic_audit_file",
    ):
        node_id = int(index_id)
        if node_id not in used_node_ids:
            continue
        seen.add(node_id)
        accumulator.add_node(
            node_id=node_id,
            kind="file",
            tokens=freebsd_file_natural_tokens(path),
            raw_detail=path,
        )
    query = "select index_id, src_addr, dst_addr from netflow_node_table"
    for index_id, src_addr, dst_addr in stream_rows(conn, query, "cadets_semantic_audit_netflow"):
        node_id = int(index_id)
        if node_id not in used_node_ids:
            continue
        seen.add(node_id)
        accumulator.add_node(
            node_id=node_id,
            kind="netflow",
            tokens=freebsd_netflow_natural_tokens(dst_addr, src_addr),
            raw_detail=f"{src_addr or ''}->{dst_addr or ''}",
        )
    return seen


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: Sequence[str]) -> None:
    """Write rows to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    *,
    out_dir: Path,
    accumulator: AuditAccumulator,
    candidate_rows: Sequence[Mapping[str, object]],
    missing_rows: Sequence[Mapping[str, object]],
    e4_artifact_dir: Path,
    collision_limit: int,
) -> None:
    """Write all audit CSV and Markdown outputs."""
    summary_rows = accumulator.summary_rows()
    write_csv(
        out_dir / "cadets_e3_e4_semantic_audit_summary.csv",
        summary_rows,
        [
            "dataset",
            "semantic_mode",
            "kind",
            "total_nodes",
            "generic_nodes",
            "generic_ratio",
            "exact_ip_nodes",
            "exact_ip_ratio",
            "gt_nodes",
            "gt_generic_nodes",
            "gt_generic_ratio",
            "focus_flags",
            "gt_focus_flags",
        ],
    )
    collision_fields = [
        "dataset",
        "kind",
        "token_tuple",
        "node_count",
        "ratio_within_kind",
        "gt_node_count",
        "gt_node_ratio_within_tuple",
        "sample_node_id",
        "sample_raw_detail",
        "focus_flags",
    ]
    for kind in ("process", "file", "netflow"):
        write_csv(
            out_dir / f"cadets_e3_e4_{kind}_collision_tokens.csv",
            accumulator.collision_rows(kind, collision_limit),
            collision_fields,
        )
    write_csv(
        out_dir / "cadets_e3_e4_gt_focus_tokens.csv",
        accumulator.gt_focus_rows(collision_limit),
        [
            "dataset",
            "row_type",
            "kind",
            "token_or_tuple",
            "count",
            "ratio_within_gt_kind",
            "sample_gt_node_id",
        ],
    )
    write_csv(
        out_dir / "cadets_e3_e4_candidate_design.csv",
        candidate_rows,
        [
            "candidate_name",
            "priority",
            "kind",
            "problem",
            "candidate",
            "runtime_visible",
            "uses_gt_for_rule",
            "requires_word2vec_retrain",
        ],
    )
    write_csv(
        out_dir / "cadets_e3_e4_missing_mapping.csv",
        missing_rows,
        ["missing_type", "node_id", "detail"],
    )
    report = render_markdown_report(
        output_dir=out_dir,
        summary_rows=summary_rows,
        candidate_rows=candidate_rows,
        missing_mapping_count=sum(1 for row in missing_rows if row["missing_type"] == "gt_not_in_node_map"),
        e4_artifact_dir=e4_artifact_dir,
    )
    (out_dir / "cadets_e3_e4_semantic_audit_report.md").write_text(report, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-name", default="cadets_e3")
    parser.add_argument("--db-host", default=os.environ.get("CLAD_DB_HOST", "localhost"))
    parser.add_argument("--db-port", default=os.environ.get("CLAD_DB_PORT", "5433"))
    parser.add_argument("--db-user", default=os.environ.get("CLAD_DB_USER", "postgres"))
    parser.add_argument("--db-password", default=os.environ.get("CLAD_DB_PASSWORD", ""))
    parser.add_argument(
        "--node-map",
        type=Path,
        default=Path("outputs/cache/phase3e/node_embeddings/CADETS_E3_latent64/node_id_to_idx.pkl"),
    )
    parser.add_argument("--gt-dir", type=Path, default=Path("ground_truth/E3-CADETS"))
    parser.add_argument(
        "--e4-artifact-dir",
        type=Path,
        default=Path("tmp/zhanzhongwei/tflr_light_phase3g_v2_Q09995_full/CADETS_E3_PHASE3E_E4_NONE"),
    )
    parser.add_argument("--out-root", type=Path, default=Path("outputs/diagnostics"))
    parser.add_argument("--collision-limit", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only audit."""
    args = parse_args(argv)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"cadets_e3_e4_freebsd_semantic_audit_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    node_map = load_node_map(args.node_map)
    used_node_ids = set(node_map)
    gt_nodes, missing_rows = load_gt_nodes(args.gt_dir)
    for node_id in sorted(gt_nodes - used_node_ids):
        missing_rows.append({"missing_type": "gt_not_in_node_map", "node_id": node_id, "detail": ""})

    accumulator = AuditAccumulator(dataset="CADETS_E3", gt_nodes=gt_nodes)
    conn = connect(args)
    try:
        seen = scan_database(conn, used_node_ids, accumulator)
    finally:
        conn.close()
    for node_id in sorted(used_node_ids - seen)[:5000]:
        missing_rows.append(
            {
                "missing_type": "used_node_missing_db_meta",
                "node_id": node_id,
                "detail": "truncated_to_5000_rows",
            }
        )

    write_outputs(
        out_dir=out_dir,
        accumulator=accumulator,
        candidate_rows=build_candidate_design_rows(),
        missing_rows=missing_rows,
        e4_artifact_dir=args.e4_artifact_dir,
        collision_limit=max(int(args.collision_limit), 1),
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
