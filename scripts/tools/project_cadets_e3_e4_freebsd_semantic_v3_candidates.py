"""Read-only CADETS_E3 E4 FreeBSD semantic v3 candidate projection.

This diagnostic retokenizes CADETS node metadata with the current CADETS
FreeBSD semantic tokenizer and an independent candidate v3 projection tokenizer.
It does not modify runtime tokenizer code, train Word2Vec, run inference, or
create persistent database objects.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import re
import shlex
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import psycopg2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.semantics.cadets_freebsd import (  # noqa: E402
    CADETS_SEMANTIC_RULES_VERSION,
    classify_freebsd_file_nll,
    classify_freebsd_process_nll,
    freebsd_file_detail,
    freebsd_file_natural_tokens,
    freebsd_netflow_natural_tokens,
    freebsd_process_detail,
    freebsd_process_natural_tokens,
    normalize_token,
)
from scripts.pipeline.config.runtime_config import SUBJECT_NODE_TABLE  # noqa: E402
from scripts.tools.audit_cadets_e3_e4_freebsd_semantics import (  # noqa: E402
    has_exact_ip_token,
    load_gt_nodes,
    load_node_map,
    token_is_generic,
    token_key,
)


CANDIDATE_VERSION = "cadets_freebsd_raw_detail_v3_safe_lexical_projection"
MISSING_DETAIL_VALUES = {"", "none", "null", "na", "n/a", "unknown", "path_none"}
HEX_RE = re.compile(r"^[a-f0-9]{12,}$")
HOME_WEB_SUFFIXES = {
    ".css",
    ".gif",
    ".htm",
    ".html",
    ".jpeg",
    ".jpg",
    ".js",
    ".png",
    ".shtml",
}


def split_command(cmd: object) -> list[str]:
    """Split a process command with shell-like parsing."""
    raw = str(cmd or "").strip()
    if not raw or raw.lower() in MISSING_DETAIL_VALUES:
        return []
    try:
        return shlex.split(raw)
    except ValueError:
        return raw.split()


def basename_token(value: object) -> str:
    """Return a bounded normalized basename token."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    return normalize_token(PurePosixPath(raw).name, max_len=60)


def looks_payload_shape(token: str, raw_context: object = "") -> bool:
    """Return whether a token has a label-free payload-like lexical shape."""
    value = normalize_token(token, max_len=60)
    if not value or value in {"unknown", "process_other", "file"}:
        return False
    if re.search(r"[a-z]", value) and re.search(r"[0-9]", value):
        return 5 <= len(value) <= 32
    if "_" in value and 5 <= len(value) <= 40:
        return True
    return False


def looks_path_payload_shape(path: object, detail: str) -> bool:
    """Return whether a path basename has a conservative payload-like shape."""
    raw = str(path or "").strip().lower()
    value = normalize_token(detail, max_len=60)
    if not raw.startswith(("/tmp/", "/var/tmp/", "/usr/home/", "/home/")):
        return False
    if PurePosixPath(raw).suffix.lower() == ".lock":
        return False
    if looks_payload_shape(value):
        return True
    suffix = PurePosixPath(raw).suffix.lower()
    return not suffix and bool(re.fullmatch(r"[a-z]{3,12}", value))


def process_other_shape(detail: str, raw_context: object) -> str:
    """Return a refined process_other candidate role from runtime-visible text."""
    token = normalize_token(detail, max_len=60)
    if not token or token == "unknown":
        return "process_other_generic"
    if looks_payload_shape(token, raw_context):
        return "process_other_payload_shape"
    if re.search(r"[0-9]", token) and len(token) >= 5:
        return "process_other_numeric_shape"
    return "process_other_named"


def candidate_process_tokens(cmd: object) -> tuple[str, ...]:
    """Return candidate v3 projection tokens for a FreeBSD process node."""
    label = classify_freebsd_process_nll(cmd)
    detail = freebsd_process_detail(cmd, label)
    if label != "process_other":
        return ("process", label, detail)

    parts = split_command(cmd)
    executable = basename_token(parts[0]) if parts else detail
    refined_detail = executable or detail or "unknown"
    return ("process", process_other_shape(refined_detail, cmd), refined_detail)


def candidate_file_tokens(path: object) -> tuple[str, ...]:
    """Return candidate v3 projection tokens for a FreeBSD file node."""
    raw = str(path or "").strip()
    if raw.lower() in MISSING_DETAIL_VALUES:
        return ("file", "missing_file_path", "missing_path")

    label = classify_freebsd_file_nll(raw)
    detail = freebsd_file_detail(raw, label)
    lowered = raw.lower()
    if lowered in {"/dev/random", "/dev/urandom"}:
        return ("file", "device_entropy_file", basename_token(raw) or "random")
    if lowered.startswith(("/tmp/", "/var/tmp/")) and looks_path_payload_shape(raw, detail):
        return ("file", "tmp_payload_shape_file", detail)
    if lowered.startswith(("/usr/home/", "/home/")):
        if PurePosixPath(lowered).suffix in HOME_WEB_SUFFIXES:
            return ("file", "user_home_web_artifact_file", detail)
        if looks_path_payload_shape(raw, detail):
            return ("file", "user_home_payload_shape_file", detail)
    return ("file", label, detail)


def candidate_netflow_tokens(dst_addr: object, src_addr: object = "") -> tuple[str, ...]:
    """Return candidate v3 projection tokens for a FreeBSD netflow node."""
    return freebsd_netflow_natural_tokens(dst_addr, src_addr)


def token_entropy_risk(tokens: Sequence[object]) -> tuple[str, ...]:
    """Return bounded high-entropy/OOV risk flags for one token tuple."""
    flags: list[str] = []
    for token in tokens:
        text = str(token or "")
        if len(text) > 80:
            flags.append("long_token")
        norm = text.lower().replace("_", "")
        if HEX_RE.fullmatch(norm):
            flags.append("hex_like_token")
        if len(norm) >= 16 and sum(ch.isdigit() for ch in norm) / len(norm) > 0.65:
            flags.append("numeric_heavy_token")
    return tuple(dict.fromkeys(flags))


def candidate_is_generic(kind: str, tokens: Sequence[str]) -> bool:
    """Return whether a candidate token tuple remains intentionally generic."""
    if kind == "file" and "missing_file_path" in set(tokens):
        return True
    if kind == "process" and "process_other_generic" in set(tokens):
        return True
    return token_is_generic(kind, tokens)


def focus_flags(kind: str, tokens: Sequence[str], raw_detail: object) -> tuple[str, ...]:
    """Return focus flags for current/candidate projection comparisons."""
    token_set = set(tokens)
    raw = str(raw_detail or "").lower()
    flags: list[str] = []
    if kind == "process":
        for flag in (
            "process_other",
            "process_other_payload_shape",
            "process_other_named",
            "process_other_numeric_shape",
            "process_other_generic",
        ):
            if flag in token_set:
                flags.append(flag)
    elif kind == "file":
        if tuple(tokens) == ("file",):
            flags.append("empty_file_tuple")
        for flag in (
            "missing_file_path",
            "tmp_payload_shape_file",
            "user_home_payload_shape_file",
            "device_entropy_file",
            "tmp_file",
            "user_home_file",
            "system_log_file",
            "file_other",
        ):
            if flag in token_set:
                flags.append(flag)
        if raw.startswith("/tmp/"):
            flags.append("raw_tmp_path")
        if raw.startswith(("/usr/home/", "/home/")):
            flags.append("raw_user_home_path")
    elif kind == "netflow":
        if has_exact_ip_token(tokens):
            flags.append("exact_ip_retained")
    return tuple(dict.fromkeys(flags))


@dataclass
class ProjectionAccumulator:
    """Aggregate current-vs-candidate projection statistics."""

    dataset: str
    gt_nodes: set[int]
    kind_counts: Counter[str] = field(default_factory=Counter)
    current_generic_counts: Counter[str] = field(default_factory=Counter)
    candidate_generic_counts: Counter[str] = field(default_factory=Counter)
    current_empty_file_counts: Counter[str] = field(default_factory=Counter)
    candidate_empty_file_counts: Counter[str] = field(default_factory=Counter)
    gt_kind_counts: Counter[str] = field(default_factory=Counter)
    gt_current_generic_counts: Counter[str] = field(default_factory=Counter)
    gt_candidate_generic_counts: Counter[str] = field(default_factory=Counter)
    current_collision_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    candidate_collision_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    gt_current_collision_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    gt_candidate_collision_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    current_focus_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    candidate_focus_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    gt_current_focus_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    gt_candidate_focus_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    risk_counts: Counter[tuple[str, str]] = field(default_factory=Counter)
    sample_node: dict[tuple[str, str, str], int] = field(default_factory=dict)
    sample_raw_detail: dict[tuple[str, str, str], str] = field(default_factory=dict)

    def add_node(
        self,
        *,
        node_id: int,
        kind: str,
        current_tokens: Sequence[object],
        candidate_tokens: Sequence[object],
        raw_detail: object,
    ) -> None:
        """Add one node's current and candidate token tuples."""
        current = tuple(str(token) for token in current_tokens if str(token).strip())
        candidate = tuple(str(token) for token in candidate_tokens if str(token).strip())
        current_key = token_key(current)
        candidate_key = token_key(candidate)
        is_gt = int(node_id) in self.gt_nodes

        self.kind_counts[kind] += 1
        self.current_generic_counts[kind] += int(token_is_generic(kind, current))
        self.candidate_generic_counts[kind] += int(candidate_is_generic(kind, candidate))
        self.current_empty_file_counts[kind] += int(kind == "file" and current == ("file",))
        self.candidate_empty_file_counts[kind] += int(kind == "file" and candidate == ("file",))
        self.current_collision_counts[(kind, current_key)] += 1
        self.candidate_collision_counts[(kind, candidate_key)] += 1

        self.sample_node.setdefault(("current", kind, current_key), int(node_id))
        self.sample_node.setdefault(("candidate", kind, candidate_key), int(node_id))
        self.sample_raw_detail.setdefault(("current", kind, current_key), str(raw_detail or ""))
        self.sample_raw_detail.setdefault(("candidate", kind, candidate_key), str(raw_detail or ""))

        for flag in focus_flags(kind, current, raw_detail):
            self.current_focus_counts[(kind, flag)] += 1
            if is_gt:
                self.gt_current_focus_counts[(kind, flag)] += 1
        for flag in focus_flags(kind, candidate, raw_detail):
            self.candidate_focus_counts[(kind, flag)] += 1
            if is_gt:
                self.gt_candidate_focus_counts[(kind, flag)] += 1
        for flag in token_entropy_risk(candidate):
            self.risk_counts[(kind, flag)] += 1

        if is_gt:
            self.gt_kind_counts[kind] += 1
            self.gt_current_generic_counts[kind] += int(token_is_generic(kind, current))
            self.gt_candidate_generic_counts[kind] += int(candidate_is_generic(kind, candidate))
            self.gt_current_collision_counts[(kind, current_key)] += 1
            self.gt_candidate_collision_counts[(kind, candidate_key)] += 1

    def summary_rows(self) -> list[dict[str, object]]:
        """Return per-kind current-vs-candidate summary rows."""
        rows = []
        for kind in ("process", "file", "netflow"):
            total = self.kind_counts[kind]
            current_generic = self.current_generic_counts[kind]
            candidate_generic = self.candidate_generic_counts[kind]
            gt_total = self.gt_kind_counts[kind]
            rows.append(
                {
                    "dataset": self.dataset,
                    "kind": kind,
                    "current_semantic_mode": CADETS_SEMANTIC_RULES_VERSION,
                    "candidate_semantic_mode": CANDIDATE_VERSION,
                    "total_nodes": total,
                    "current_generic_nodes": current_generic,
                    "candidate_generic_nodes": candidate_generic,
                    "generic_node_delta": candidate_generic - current_generic,
                    "current_generic_ratio": current_generic / total if total else 0.0,
                    "candidate_generic_ratio": candidate_generic / total if total else 0.0,
                    "current_empty_file_nodes": self.current_empty_file_counts[kind],
                    "candidate_empty_file_nodes": self.candidate_empty_file_counts[kind],
                    "gt_nodes": gt_total,
                    "gt_current_generic_nodes": self.gt_current_generic_counts[kind],
                    "gt_candidate_generic_nodes": self.gt_candidate_generic_counts[kind],
                    "gt_generic_node_delta": (
                        self.gt_candidate_generic_counts[kind]
                        - self.gt_current_generic_counts[kind]
                    ),
                }
            )
        return rows

    def top_collision_rows(self, version: str, limit: int) -> list[dict[str, object]]:
        """Return top current or candidate collision rows."""
        if version == "current":
            counts = self.current_collision_counts
            gt_counts = self.gt_current_collision_counts
        else:
            counts = self.candidate_collision_counts
            gt_counts = self.gt_candidate_collision_counts
        rows = []
        for (kind, key), count in counts.most_common():
            total = self.kind_counts[kind]
            sample_key = (version, kind, key)
            rows.append(
                {
                    "dataset": self.dataset,
                    "version": version,
                    "kind": kind,
                    "token_tuple": key,
                    "node_count": count,
                    "ratio_within_kind": count / total if total else 0.0,
                    "gt_node_count": gt_counts[(kind, key)],
                    "sample_node_id": self.sample_node.get(sample_key, ""),
                    "sample_raw_detail": self.sample_raw_detail.get(sample_key, ""),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def collision_delta_rows(self, limit: int) -> list[dict[str, object]]:
        """Return current and candidate collision rows by token key delta."""
        keys = set(self.current_collision_counts) | set(self.candidate_collision_counts)
        rows = []
        for kind, key in sorted(
            keys,
            key=lambda item: abs(
                self.candidate_collision_counts[item] - self.current_collision_counts[item]
            ),
            reverse=True,
        ):
            current_count = self.current_collision_counts[(kind, key)]
            candidate_count = self.candidate_collision_counts[(kind, key)]
            if current_count == candidate_count:
                continue
            rows.append(
                {
                    "dataset": self.dataset,
                    "kind": kind,
                    "token_tuple": key,
                    "current_node_count": current_count,
                    "candidate_node_count": candidate_count,
                    "node_count_delta": candidate_count - current_count,
                    "current_gt_node_count": self.gt_current_collision_counts[(kind, key)],
                    "candidate_gt_node_count": self.gt_candidate_collision_counts[(kind, key)],
                    "interpretation": (
                        "candidate_added_or_split"
                        if candidate_count > current_count
                        else "candidate_reduced_or_renamed"
                    ),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def gt_focus_delta_rows(self, limit: int) -> list[dict[str, object]]:
        """Return GT post-hoc focus flag deltas."""
        keys = set(self.gt_current_focus_counts) | set(self.gt_candidate_focus_counts)
        rows = []
        for kind, flag in sorted(
            keys,
            key=lambda item: (
                self.gt_candidate_focus_counts[item] + self.gt_current_focus_counts[item],
                item,
            ),
            reverse=True,
        ):
            rows.append(
                {
                    "dataset": self.dataset,
                    "kind": kind,
                    "focus_flag": flag,
                    "current_gt_count": self.gt_current_focus_counts[(kind, flag)],
                    "candidate_gt_count": self.gt_candidate_focus_counts[(kind, flag)],
                    "gt_count_delta": (
                        self.gt_candidate_focus_counts[(kind, flag)]
                        - self.gt_current_focus_counts[(kind, flag)]
                    ),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def risk_rows(self, limit: int) -> list[dict[str, object]]:
        """Return candidate token risk summary rows."""
        rows = []
        for (kind, flag), count in self.risk_counts.most_common(limit):
            total = self.kind_counts[kind]
            rows.append(
                {
                    "dataset": self.dataset,
                    "kind": kind,
                    "risk_flag": flag,
                    "node_count": count,
                    "ratio_within_kind": count / total if total else 0.0,
                }
            )
        return rows


def candidate_rule_rows() -> list[dict[str, object]]:
    """Return candidate rule provenance rows."""
    return [
        {
            "candidate_rule": "process_other_safe_lexical_shape",
            "kind": "process",
            "description": "Split process_other by executable lexical shape and path context.",
            "runtime_visible": True,
            "uses_gt_for_rule": False,
            "requires_word2vec_retrain": True,
        },
        {
            "candidate_rule": "file_missing_path_explicit_bucket",
            "kind": "file",
            "description": "Replace empty file tuple with explicit missing_file_path token.",
            "runtime_visible": True,
            "uses_gt_for_rule": False,
            "requires_word2vec_retrain": True,
        },
        {
            "candidate_rule": "file_tmp_home_payload_shape",
            "kind": "file",
            "description": "Refine tmp/home file families by bounded basename shape.",
            "runtime_visible": True,
            "uses_gt_for_rule": False,
            "requires_word2vec_retrain": True,
        },
        {
            "candidate_rule": "file_home_web_artifact_split",
            "kind": "file",
            "description": "Keep common home web/static files out of payload-shape buckets.",
            "runtime_visible": True,
            "uses_gt_for_rule": False,
            "requires_word2vec_retrain": True,
        },
        {
            "candidate_rule": "file_device_entropy_detail",
            "kind": "file",
            "description": "Separate /dev/random and /dev/urandom as entropy device files.",
            "runtime_visible": True,
            "uses_gt_for_rule": False,
            "requires_word2vec_retrain": True,
        },
        {
            "candidate_rule": "netflow_no_semantic_change",
            "kind": "netflow",
            "description": "Preserve current exact-IP netflow tokenization in this projection.",
            "runtime_visible": True,
            "uses_gt_for_rule": False,
            "requires_word2vec_retrain": False,
        },
    ]


def render_projection_report(
    *,
    output_dir: object,
    summary_rows: Sequence[Mapping[str, object]],
    rule_rows: Sequence[Mapping[str, object]],
    missing_mapping_count: int,
    prior_audit_dir: object,
) -> str:
    """Render the Markdown projection report."""
    lines = [
        "# CADETS_E3 E4 FreeBSD Semantic v3 Candidate Projection",
        "",
        "Scope: read-only projection for a candidate FreeBSD semantic v3 tokenizer.",
        "",
        "- The official tokenizer is not modified.",
        "- No Word2Vec training.",
        "- No inference.",
        "- No runtime policy or threshold changes.",
        "- GT is post-hoc only and is not used to construct candidate rules.",
        "",
        f"Output directory: `{output_dir}`",
        f"Prior audit directory: `{prior_audit_dir}`",
        f"Missing GT mapping count: `{missing_mapping_count}`",
        "",
        "## Summary",
        "",
        "| Kind | Nodes | Current Generic | Candidate Generic | Delta | "
        "Current Empty File | Candidate Empty File | GT Current Generic | "
        "GT Candidate Generic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['kind']} | {row['total_nodes']} | {row['current_generic_nodes']} | "
            f"{row['candidate_generic_nodes']} | {row['generic_node_delta']} | "
            f"{row['current_empty_file_nodes']} | {row['candidate_empty_file_nodes']} | "
            f"{row['gt_current_generic_nodes']} | {row['gt_candidate_generic_nodes']} |"
        )
    lines += [
        "",
        "## Candidate Rules",
        "",
        "| Rule | Kind | Runtime Visible | Uses GT For Rule | Requires Word2Vec Retrain |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rule_rows:
        lines.append(
            f"| {row['candidate_rule']} | {row['kind']} | {row['runtime_visible']} | "
            f"{row['uses_gt_for_rule']} | {row['requires_word2vec_retrain']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "1. `process_other` improvement should be judged by renamed/split candidate "
        "roles such as `process_other_payload_shape` and `process_other_named`.",
        "2. The empty `file` bucket should drop to zero in candidate projection and "
        "move into explicit `missing_file_path` semantics.",
        "3. Netflow is intentionally unchanged because current CADETS FreeBSD "
        "semantics already retain exact endpoint identity.",
        "4. A future formal v3 tokenizer should only be promoted after this "
        "projection is reviewed and Word2Vec retrain is explicitly approved.",
        "",
        "## Recommended next stage",
        "",
        "Proceed to a formal `cadets_freebsd_raw_detail_v3_safe_lexical` design only "
        "if this projection confirms process and file focus improvements without "
        "broad benign relabeling. Otherwise tighten candidate file buckets first.",
        "",
        "## Output Files",
        "",
        "- `cadets_e3_e4_semantic_v3_projection_summary.csv`",
        "- `cadets_e3_e4_semantic_v3_collision_delta.csv`",
        "- `cadets_e3_e4_semantic_v3_current_top_collisions.csv`",
        "- `cadets_e3_e4_semantic_v3_candidate_top_collisions.csv`",
        "- `cadets_e3_e4_semantic_v3_gt_focus_delta.csv`",
        "- `cadets_e3_e4_semantic_v3_token_risk.csv`",
        "- `cadets_e3_e4_semantic_v3_candidate_rules.csv`",
    ]
    return "\n".join(lines) + "\n"


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


def scan_database(conn, used_node_ids: set[int], accumulator: ProjectionAccumulator) -> set[int]:
    """Stream node metadata and aggregate current-vs-candidate tokens."""
    seen: set[int] = set()
    query = f"select index_id, cmd from {SUBJECT_NODE_TABLE}"
    for index_id, cmd in stream_rows(conn, query, "cadets_v3_projection_process"):
        node_id = int(index_id)
        if node_id not in used_node_ids:
            continue
        seen.add(node_id)
        accumulator.add_node(
            node_id=node_id,
            kind="process",
            current_tokens=freebsd_process_natural_tokens(cmd),
            candidate_tokens=candidate_process_tokens(cmd),
            raw_detail=cmd,
        )

    for index_id, path in stream_rows(
        conn,
        "select index_id, path from file_node_table",
        "cadets_v3_projection_file",
    ):
        node_id = int(index_id)
        if node_id not in used_node_ids:
            continue
        seen.add(node_id)
        accumulator.add_node(
            node_id=node_id,
            kind="file",
            current_tokens=freebsd_file_natural_tokens(path),
            candidate_tokens=candidate_file_tokens(path),
            raw_detail=path,
        )

    query = "select index_id, src_addr, dst_addr from netflow_node_table"
    for index_id, src_addr, dst_addr in stream_rows(conn, query, "cadets_v3_projection_netflow"):
        node_id = int(index_id)
        if node_id not in used_node_ids:
            continue
        seen.add(node_id)
        accumulator.add_node(
            node_id=node_id,
            kind="netflow",
            current_tokens=freebsd_netflow_natural_tokens(dst_addr, src_addr),
            candidate_tokens=candidate_netflow_tokens(dst_addr, src_addr),
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
    accumulator: ProjectionAccumulator,
    rule_rows: Sequence[Mapping[str, object]],
    missing_rows: Sequence[Mapping[str, object]],
    prior_audit_dir: object,
    collision_limit: int,
) -> None:
    """Write projection CSV and Markdown outputs."""
    summary_rows = accumulator.summary_rows()
    write_csv(
        out_dir / "cadets_e3_e4_semantic_v3_projection_summary.csv",
        summary_rows,
        [
            "dataset",
            "kind",
            "current_semantic_mode",
            "candidate_semantic_mode",
            "total_nodes",
            "current_generic_nodes",
            "candidate_generic_nodes",
            "generic_node_delta",
            "current_generic_ratio",
            "candidate_generic_ratio",
            "current_empty_file_nodes",
            "candidate_empty_file_nodes",
            "gt_nodes",
            "gt_current_generic_nodes",
            "gt_candidate_generic_nodes",
            "gt_generic_node_delta",
        ],
    )
    collision_fields = [
        "dataset",
        "version",
        "kind",
        "token_tuple",
        "node_count",
        "ratio_within_kind",
        "gt_node_count",
        "sample_node_id",
        "sample_raw_detail",
    ]
    write_csv(
        out_dir / "cadets_e3_e4_semantic_v3_current_top_collisions.csv",
        accumulator.top_collision_rows("current", collision_limit),
        collision_fields,
    )
    write_csv(
        out_dir / "cadets_e3_e4_semantic_v3_candidate_top_collisions.csv",
        accumulator.top_collision_rows("candidate", collision_limit),
        collision_fields,
    )
    write_csv(
        out_dir / "cadets_e3_e4_semantic_v3_collision_delta.csv",
        accumulator.collision_delta_rows(collision_limit),
        [
            "dataset",
            "kind",
            "token_tuple",
            "current_node_count",
            "candidate_node_count",
            "node_count_delta",
            "current_gt_node_count",
            "candidate_gt_node_count",
            "interpretation",
        ],
    )
    write_csv(
        out_dir / "cadets_e3_e4_semantic_v3_gt_focus_delta.csv",
        accumulator.gt_focus_delta_rows(collision_limit),
        [
            "dataset",
            "kind",
            "focus_flag",
            "current_gt_count",
            "candidate_gt_count",
            "gt_count_delta",
        ],
    )
    write_csv(
        out_dir / "cadets_e3_e4_semantic_v3_token_risk.csv",
        accumulator.risk_rows(collision_limit),
        ["dataset", "kind", "risk_flag", "node_count", "ratio_within_kind"],
    )
    write_csv(
        out_dir / "cadets_e3_e4_semantic_v3_candidate_rules.csv",
        rule_rows,
        [
            "candidate_rule",
            "kind",
            "description",
            "runtime_visible",
            "uses_gt_for_rule",
            "requires_word2vec_retrain",
        ],
    )
    write_csv(
        out_dir / "cadets_e3_e4_semantic_v3_missing_mapping.csv",
        missing_rows,
        ["missing_type", "node_id", "detail"],
    )
    report = render_projection_report(
        output_dir=out_dir,
        summary_rows=summary_rows,
        rule_rows=rule_rows,
        missing_mapping_count=sum(1 for row in missing_rows if row["missing_type"] == "gt_not_in_node_map"),
        prior_audit_dir=prior_audit_dir,
    )
    (out_dir / "cadets_e3_e4_semantic_v3_projection_report.md").write_text(
        report,
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-name", default="cadets_e3")
    parser.add_argument("--db-host", default=os.environ.get("CLAD_DB_HOST", "127.0.0.1"))
    parser.add_argument("--db-port", default=os.environ.get("CLAD_DB_PORT", "5433"))
    parser.add_argument("--db-user", default=os.environ.get("CLAD_DB_USER", "postgres"))
    parser.add_argument("--db-password", default=os.environ.get("CLAD_DB_PASSWORD", ""))
    parser.add_argument(
        "--node-map",
        type=Path,
        default=Path("outputs/cache/phase3e/node_embeddings/CADETS_E3_latent64/node_id_to_idx.pkl"),
    )
    parser.add_argument("--gt-dir", type=Path, default=Path("ground_truth/E3-CADETS"))
    parser.add_argument("--prior-audit-dir", default="")
    parser.add_argument("--collision-limit", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only candidate projection."""
    args = parse_args(argv)
    if not args.node_map.exists():
        raise FileNotFoundError(f"node map not found: {args.node_map}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path(
        f"outputs/diagnostics/cadets_e3_e4_freebsd_semantic_v3_projection_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    node_map = load_node_map(args.node_map)
    gt_nodes, missing_rows = load_gt_nodes(args.gt_dir)
    used_node_ids = set(node_map)
    for gt_node in sorted(gt_nodes - used_node_ids):
        missing_rows.append(
            {
                "missing_type": "gt_not_in_node_map",
                "node_id": gt_node,
                "detail": str(args.node_map),
            }
        )

    accumulator = ProjectionAccumulator(
        dataset="CADETS_E3",
        gt_nodes=gt_nodes & used_node_ids,
    )
    with connect(args) as conn:
        seen = scan_database(conn, used_node_ids, accumulator)

    for node_id in sorted(used_node_ids - seen)[:1000]:
        missing_rows.append(
            {
                "missing_type": "node_map_id_not_seen_in_metadata",
                "node_id": node_id,
                "detail": "limited_to_first_1000",
            }
        )

    write_outputs(
        out_dir=out_dir,
        accumulator=accumulator,
        rule_rows=candidate_rule_rows(),
        missing_rows=missing_rows,
        prior_audit_dir=args.prior_audit_dir,
        collision_limit=args.collision_limit,
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
