# ClearScope E5 Semantic Audit Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible ClearScope E5 semantic audit smoke with a label-free
primary report and a label-aware diagnostic appendix, then gate any E5 bounded
pipeline smoke on the audit result.

**Architecture:** Add one focused audit tool that reads `clearscope_e5` through
PostgreSQL, tokenizes node/event details with existing ClearScope semantic helpers,
and writes compact JSON/CSV reports under `tmp/`. Keep E5 pipeline smoke as a
separate gated task that validates artifact isolation before any bounded run.

**Tech Stack:** Python 3, `unittest`, `psycopg2`, existing `cs4m` ClearScope
semantic helpers, local PostgreSQL, existing shell runner conventions.

---

## File Structure

- Create `scripts/tools/audit_clearscope_e5_semantic_smoke.py`
  - CLI entrypoint for E5 audit smoke.
  - Reads PostgreSQL using `CLAD_DB_*` environment variables.
  - Generates label-free primary JSON/CSV and label-aware diagnostic JSON/CSV.
  - Does not mutate DB, source files, model artifacts, or result directories.
- Create `tests/test_clearscope_e5_semantic_audit_smoke.py`
  - Unit tests for split parsing, tokenization wrappers, collision aggregation,
    fallback-bucket accounting, label-aware diagnostic separation, and artifact
    isolation checks.
- Create `tmp/clearscope_e5_semantic_audit_smoke/` at runtime only.
  - Generated audit reports live here and are not source code.
- Modify no model, tokenizer, runner, training, inference, or evaluation code in
  Tasks 1-4.
- Modify runner or add E5 runner only in Task 5 if the audit decision explicitly
  allows pipeline smoke.

## Task 1: Unit-Test Audit Data Model And Label Separation

**Files:**
- Create: `tests/test_clearscope_e5_semantic_audit_smoke.py`
- Create: `scripts/tools/audit_clearscope_e5_semantic_smoke.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_clearscope_e5_semantic_audit_smoke.py` with:

```python
from __future__ import annotations

import unittest

from scripts.tools.audit_clearscope_e5_semantic_smoke import (
    FALLBACK_TOKENS,
    AuditNode,
    assign_split,
    build_label_aware_diagnostics,
    build_label_free_summary,
    collect_fallback_counts,
    extract_detail_token,
)


class ClearScopeE5SemanticAuditSmokeTests(unittest.TestCase):
    """Validate ClearScope E5 semantic audit smoke helpers."""

    def test_assign_split_uses_e5_day_contract(self) -> None:
        self.assertEqual(assign_split(8), "train")
        self.assertEqual(assign_split(9), "train")
        self.assertEqual(assign_split(11), "val")
        self.assertEqual(assign_split(14), "test")
        self.assertEqual(assign_split(15), "test")
        self.assertEqual(assign_split(17), "test")
        self.assertEqual(assign_split(10), "unused")
        self.assertEqual(assign_split(99), "unused")

    def test_extract_detail_token_returns_last_semantic_token(self) -> None:
        node = AuditNode(
            index_id=1,
            node_uuid="node-file",
            node_type="file",
            raw_detail="/data/data/org.mozilla.fennec_firefox_dev/cache/x/cache2/entries/ABCD",
            semantic_tokens=("file", "android_app_cache_file", "fennec_firefox_dev_cache2_entries_mixedid"),
        )

        self.assertEqual(extract_detail_token(node), "fennec_firefox_dev_cache2_entries_mixedid")

    def test_collect_fallback_counts_tracks_known_fallback_tokens(self) -> None:
        nodes = [
            AuditNode(1, "a", "file", "/unknown", ("file", "file_other", "other")),
            AuditNode(2, "b", "file", "", ("file", "unknown_file", "unknown")),
            AuditNode(3, "c", "netflow", "10.0.0.1:1->10.0.0.2:2", ("netflow", "fixed", "netflow")),
            AuditNode(4, "d", "file", "/system/bin/sh", ("file", "android_system_file", "bin_sh")),
        ]

        counts = collect_fallback_counts(nodes)

        self.assertEqual(counts["file_other"], 1)
        self.assertEqual(counts["unknown_file"], 1)
        self.assertEqual(counts["netflow"], 1)
        self.assertNotIn("android_system_file", FALLBACK_TOKENS)

    def test_label_free_summary_does_not_include_label_fields(self) -> None:
        nodes = [
            AuditNode(1, "benign", "file", "/system/bin/toybox", ("file", "android_system_file", "bin_toybox")),
            AuditNode(2, "mal", "file", "/data/local/tmp/tester", ("file", "android_tmp_file", "tmp_file")),
        ]

        summary = build_label_free_summary(nodes)

        self.assertEqual(summary["node_type_counts"]["file"], 2)
        self.assertNotIn("malicious", summary)
        self.assertNotIn("attack", summary)
        self.assertIn("fallback_counts", summary)

    def test_label_aware_diagnostics_are_separate_appendix(self) -> None:
        nodes = [
            AuditNode(1, "benign", "file", "/system/bin/toybox", ("file", "android_system_file", "bin_toybox")),
            AuditNode(2, "mal", "file", "/data/local/tmp/tester", ("file", "android_tmp_file", "tmp_file")),
        ]

        diagnostics = build_label_aware_diagnostics(nodes, malicious_index_ids={2})

        self.assertEqual(diagnostics["malicious_node_count"], 1)
        self.assertEqual(diagnostics["malicious_fallback_counts"], {})
        self.assertIn("label_aware_diagnostic_only", diagnostics["warning"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_semantic_audit_smoke
```

Expected: FAIL or ERROR because
`scripts.tools.audit_clearscope_e5_semantic_smoke` does not exist.

- [ ] **Step 3: Add the minimal audit module skeleton**

Create `scripts/tools/audit_clearscope_e5_semantic_smoke.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


E5_SPLITS = {
    "train": {8, 9},
    "val": {11},
    "test": {14, 15, 17},
}
FALLBACK_TOKENS = {
    "file_other",
    "unknown_file",
    "dev_other",
    "socket_other",
    "netflow",
}


@dataclass(frozen=True)
class AuditNode:
    """One node row after ClearScope semantic tokenization."""

    index_id: int
    node_uuid: str
    node_type: str
    raw_detail: str
    semantic_tokens: tuple[str, ...]


def assign_split(day: int) -> str:
    """Return the E5 split name for a calendar day."""
    day_int = int(day)
    for split, days in E5_SPLITS.items():
        if day_int in days:
            return split
    return "unused"


def extract_detail_token(node: AuditNode) -> str:
    """Return the most specific semantic token for a node."""
    if not node.semantic_tokens:
        return "unknown"
    return str(node.semantic_tokens[-1])


def collect_fallback_counts(nodes: Iterable[AuditNode]) -> dict[str, int]:
    """Count fallback semantic buckets in tokenized nodes."""
    counts: Counter[str] = Counter()
    for node in nodes:
        for token in node.semantic_tokens:
            if token in FALLBACK_TOKENS:
                counts[token] += 1
    return dict(sorted(counts.items()))


def build_label_free_summary(nodes: Iterable[AuditNode]) -> dict[str, object]:
    """Build the label-free audit summary for tokenized nodes."""
    node_list = list(nodes)
    node_type_counts: Counter[str] = Counter(node.node_type for node in node_list)
    detail_token_counts: Counter[str] = Counter(extract_detail_token(node) for node in node_list)
    return {
        "node_count": len(node_list),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "fallback_counts": collect_fallback_counts(node_list),
        "top_detail_tokens": detail_token_counts.most_common(50),
    }


def build_label_aware_diagnostics(
    nodes: Iterable[AuditNode],
    malicious_index_ids: set[int],
) -> dict[str, object]:
    """Build the label-aware diagnostic appendix without changing primary audit."""
    malicious = [node for node in nodes if int(node.index_id) in malicious_index_ids]
    fallback_counts = collect_fallback_counts(malicious)
    collision_groups: dict[str, set[str]] = defaultdict(set)
    for node in malicious:
        collision_groups[extract_detail_token(node)].add(node.raw_detail)
    return {
        "warning": "label_aware_diagnostic_only_not_runtime_policy",
        "malicious_node_count": len(malicious),
        "malicious_fallback_counts": fallback_counts,
        "malicious_detail_collision_groups": {
            token: sorted(raw_values)
            for token, raw_values in sorted(collision_groups.items())
            if len(raw_values) > 1
        },
    }
```

- [ ] **Step 4: Run the focused test and confirm it passes**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_semantic_audit_smoke
```

Expected: `Ran 5 tests ... OK`.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add scripts/tools/audit_clearscope_e5_semantic_smoke.py \
    tests/test_clearscope_e5_semantic_audit_smoke.py
git commit -m "test: define clearscope e5 audit smoke helpers"
```

## Task 2: Add PostgreSQL Node Loading And Tokenization

**Files:**
- Modify: `scripts/tools/audit_clearscope_e5_semantic_smoke.py`
- Modify: `tests/test_clearscope_e5_semantic_audit_smoke.py`

- [ ] **Step 1: Add unit tests for node row tokenization**

Append these tests inside `ClearScopeE5SemanticAuditSmokeTests`:

```python
    def test_tokenize_file_row_uses_v31_semantics(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import tokenize_node_row

        node = tokenize_node_row(
            {
                "index_id": 10,
                "node_uuid": "file-node",
                "node_type": "file",
                "path": "/data/local/tmp/tester",
            },
            semantic_mode="raw_detail_v31_discriminative",
        )

        self.assertEqual(node.node_type, "file")
        self.assertEqual(node.semantic_tokens[0], "file")
        self.assertIn("android_tmp_file", node.semantic_tokens)

    def test_tokenize_subject_row_uses_cmd_when_present(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import tokenize_node_row

        node = tokenize_node_row(
            {
                "index_id": 11,
                "node_uuid": "subject-node",
                "node_type": "subject",
                "path": "/system/bin/toybox",
                "cmd": "com.android.providers.contacts",
            },
            semantic_mode="raw_detail_v31_discriminative",
        )

        self.assertEqual(node.node_type, "subject")
        self.assertEqual(node.semantic_tokens[0], "process")
        self.assertIn("providers", node.semantic_tokens)

    def test_tokenize_netflow_row_keeps_fixed_clear_scope_netflow(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import tokenize_node_row

        node = tokenize_node_row(
            {
                "index_id": 12,
                "node_uuid": "netflow-node",
                "node_type": "netflow",
                "src_addr": "10.0.0.1",
                "src_port": "123",
                "dst_addr": "10.0.0.2",
                "dst_port": "443",
            },
            semantic_mode="raw_detail_v31_discriminative",
        )

        self.assertEqual(node.node_type, "netflow")
        self.assertIn("netflow", node.semantic_tokens)
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_semantic_audit_smoke
```

Expected: FAIL because `tokenize_node_row` is missing.

- [ ] **Step 3: Implement tokenization and DB connection helpers**

In `scripts/tools/audit_clearscope_e5_semantic_smoke.py`, add imports:

```python
import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import psycopg2

from cs4m.semantics.clearscope_android import (
    android_process_natural_tokens_refined,
    clearscope_file_natural_tokens_v31,
    clearscope_netflow_natural_tokens_refined,
)
```

Add these functions below `extract_detail_token`:

```python
def _row_text(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def tokenize_node_row(row: dict[str, Any], semantic_mode: str) -> AuditNode:
    """Tokenize one ClearScope E5 node row with the requested semantic mode."""
    node_type = str(row.get("node_type", "")).strip().lower()
    if semantic_mode != "raw_detail_v31_discriminative":
        raise ValueError(f"unsupported E5 audit semantic mode: {semantic_mode}")
    if node_type == "file":
        raw_detail = _row_text(row, "path")
        tokens = clearscope_file_natural_tokens_v31(raw_detail)
    elif node_type == "subject":
        raw_detail = _row_text(row, "cmd", "path")
        tokens = android_process_natural_tokens_refined(raw_detail)
    elif node_type == "netflow":
        raw_detail = (
            f"{_row_text(row, 'src_addr')}:{_row_text(row, 'src_port')}"
            f"->{_row_text(row, 'dst_addr')}:{_row_text(row, 'dst_port')}"
        )
        tokens = clearscope_netflow_natural_tokens_refined()
    else:
        raw_detail = ""
        tokens = ("unknown", "unknown", "unknown")
    return AuditNode(
        index_id=int(row["index_id"]),
        node_uuid=str(row.get("node_uuid", "")),
        node_type=node_type,
        raw_detail=raw_detail,
        semantic_tokens=tuple(str(token) for token in tokens),
    )


def connect_db(database: str):
    """Connect to local PostgreSQL using CLAD_DB_* environment variables."""
    return psycopg2.connect(
        host=os.getenv("CLAD_DB_HOST", "localhost"),
        port=int(os.getenv("CLAD_DB_PORT", "5433")),
        user=os.getenv("CLAD_DB_USER", "postgres"),
        password=os.getenv("CLAD_DB_PASSWORD", ""),
        dbname=database,
    )
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_semantic_audit_smoke
```

Expected: `Ran 8 tests ... OK`.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add scripts/tools/audit_clearscope_e5_semantic_smoke.py \
    tests/test_clearscope_e5_semantic_audit_smoke.py
git commit -m "feat: tokenize clearscope e5 audit nodes"
```

## Task 3: Implement Bounded Label-Free PostgreSQL Audit

**Files:**
- Modify: `scripts/tools/audit_clearscope_e5_semantic_smoke.py`
- Modify: `tests/test_clearscope_e5_semantic_audit_smoke.py`

- [ ] **Step 1: Add tests for event tuple aggregation and output schema**

Append these tests inside `ClearScopeE5SemanticAuditSmokeTests`:

```python
    def test_build_event_tuple_summary_counts_support_and_oov(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import build_event_tuple_summary

        nodes = {
            1: AuditNode(1, "p", "subject", "proc", ("process", "native_process", "toybox")),
            2: AuditNode(2, "f", "file", "/system/bin/sh", ("file", "android_system_file", "bin_sh")),
            3: AuditNode(3, "x", "file", "/data/local/tmp/x", ("file", "android_tmp_file", "tmp_file")),
        }
        events = [
            {"split": "train", "operation": "EVENT_READ", "src_index_id": 1, "dst_index_id": 2},
            {"split": "test", "operation": "EVENT_READ", "src_index_id": 1, "dst_index_id": 2},
            {"split": "test", "operation": "EVENT_WRITE", "src_index_id": 1, "dst_index_id": 3},
        ]

        summary = build_event_tuple_summary(events, nodes)

        self.assertEqual(summary["event_count"], 3)
        self.assertEqual(summary["test_oov_tuple_count"], 1)
        self.assertEqual(summary["test_seen_tuple_count"], 1)

    def test_write_reports_creates_json_and_csv(self) -> None:
        from tempfile import TemporaryDirectory
        from scripts.tools.audit_clearscope_e5_semantic_smoke import write_reports

        with TemporaryDirectory() as tmpdir:
            paths = write_reports(
                output_dir=tmpdir,
                label_free_summary={"node_count": 1, "fallback_counts": {"netflow": 1}},
                label_aware_diagnostics={"warning": "label_aware_diagnostic_only_not_runtime_policy"},
                fallback_rows=[{"token": "netflow", "count": 1}],
                collision_rows=[{"token": "tmp_file", "raw_detail_count": 2}],
            )

            self.assertTrue(paths["label_free_json"].endswith("label_free_summary.json"))
            self.assertTrue(paths["label_aware_json"].endswith("label_aware_diagnostics.json"))
            self.assertTrue(paths["fallback_csv"].endswith("fallback_counts.csv"))
            self.assertTrue(paths["collision_csv"].endswith("collision_groups.csv"))
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_semantic_audit_smoke
```

Expected: FAIL because `build_event_tuple_summary` and `write_reports` are missing.

- [ ] **Step 3: Implement event aggregation and report writers**

Add these functions to `scripts/tools/audit_clearscope_e5_semantic_smoke.py`:

```python
def _event_tuple(event: dict[str, Any], nodes_by_index: dict[int, AuditNode]) -> tuple[str, ...] | None:
    src = nodes_by_index.get(int(event["src_index_id"]))
    dst = nodes_by_index.get(int(event["dst_index_id"]))
    if src is None or dst is None:
        return None
    return (
        str(event["operation"]),
        src.node_type,
        dst.node_type,
        extract_detail_token(src),
        extract_detail_token(dst),
    )


def build_event_tuple_summary(
    events: Iterable[dict[str, Any]],
    nodes_by_index: dict[int, AuditNode],
) -> dict[str, object]:
    """Summarize label-free event tuple support and test OOV."""
    event_count = 0
    train_tuples: Counter[tuple[str, ...]] = Counter()
    test_tuples: Counter[tuple[str, ...]] = Counter()
    split_counts: Counter[str] = Counter()
    operation_counts: Counter[str] = Counter()
    for event in events:
        tuple_key = _event_tuple(event, nodes_by_index)
        if tuple_key is None:
            continue
        event_count += 1
        split = str(event["split"])
        operation = str(event["operation"])
        split_counts[split] += 1
        operation_counts[operation] += 1
        if split == "train":
            train_tuples[tuple_key] += 1
        elif split == "test":
            test_tuples[tuple_key] += 1
    test_oov = {
        tuple_key: count for tuple_key, count in test_tuples.items() if tuple_key not in train_tuples
    }
    test_seen = {
        tuple_key: count for tuple_key, count in test_tuples.items() if tuple_key in train_tuples
    }
    return {
        "event_count": event_count,
        "split_counts": dict(sorted(split_counts.items())),
        "operation_counts": dict(sorted(operation_counts.items())),
        "train_tuple_count": len(train_tuples),
        "test_tuple_count": len(test_tuples),
        "test_oov_tuple_count": len(test_oov),
        "test_seen_tuple_count": len(test_seen),
        "top_test_oov_tuples": [
            {"tuple": list(tuple_key), "count": count}
            for tuple_key, count in Counter(test_oov).most_common(50)
        ],
    }


def build_collision_rows(nodes: Iterable[AuditNode], min_raw_details: int = 2) -> list[dict[str, object]]:
    """Return detail tokens that collapse multiple raw details."""
    raw_by_token: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        raw_by_token[extract_detail_token(node)].add(node.raw_detail)
    rows = [
        {
            "token": token,
            "raw_detail_count": len(raw_values),
            "examples": sorted(raw_values)[:10],
        }
        for token, raw_values in raw_by_token.items()
        if len(raw_values) >= int(min_raw_details)
    ]
    return sorted(rows, key=lambda row: (-int(row["raw_detail_count"]), str(row["token"])))


def fallback_rows_from_summary(summary: dict[str, object]) -> list[dict[str, object]]:
    """Return fallback count rows for CSV output."""
    counts = dict(summary.get("fallback_counts", {}))
    return [{"token": token, "count": count} for token, count in sorted(counts.items())]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if not fieldnames:
        fieldnames = ["empty"]
        rows = [{"empty": ""}]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    output_dir: str | os.PathLike[str],
    label_free_summary: dict[str, object],
    label_aware_diagnostics: dict[str, object],
    fallback_rows: list[dict[str, object]],
    collision_rows: list[dict[str, object]],
) -> dict[str, str]:
    """Write E5 audit reports and return generated paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "label_free_json": str(out / "label_free_summary.json"),
        "label_aware_json": str(out / "label_aware_diagnostics.json"),
        "fallback_csv": str(out / "fallback_counts.csv"),
        "collision_csv": str(out / "collision_groups.csv"),
    }
    _write_json(Path(paths["label_free_json"]), label_free_summary)
    _write_json(Path(paths["label_aware_json"]), label_aware_diagnostics)
    _write_csv(Path(paths["fallback_csv"]), fallback_rows)
    _write_csv(Path(paths["collision_csv"]), collision_rows)
    return paths
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_semantic_audit_smoke
```

Expected: `Ran 10 tests ... OK`.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add scripts/tools/audit_clearscope_e5_semantic_smoke.py \
    tests/test_clearscope_e5_semantic_audit_smoke.py
git commit -m "feat: summarize clearscope e5 audit tuples"
```

## Task 4: Add CLI, Ground-Truth Loading, And Real DB Smoke

**Files:**
- Modify: `scripts/tools/audit_clearscope_e5_semantic_smoke.py`
- Modify: `tests/test_clearscope_e5_semantic_audit_smoke.py`

- [ ] **Step 1: Add tests for malicious index parsing**

Append this test inside `ClearScopeE5SemanticAuditSmokeTests`:

```python
    def test_parse_ground_truth_indices_reads_last_csv_column(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from scripts.tools.audit_clearscope_e5_semantic_smoke import parse_ground_truth_indices

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "gt.csv"
            path.write_text(
                "NODE,{'file': '/data/local/tmp'},123\n"
                "NODE,{'subject': 'None /system/bin/toybox'},456\n",
                encoding="utf-8",
            )

            self.assertEqual(parse_ground_truth_indices([str(path)]), {123, 456})
```

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_semantic_audit_smoke
```

Expected: FAIL because `parse_ground_truth_indices` is missing.

- [ ] **Step 3: Implement CLI and DB loaders**

Add these functions to `scripts/tools/audit_clearscope_e5_semantic_smoke.py`:

```python
def parse_ground_truth_indices(paths: list[str]) -> set[int]:
    """Read E5 ground-truth node index IDs from CSV files."""
    indices: set[int] = set()
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(f"ground-truth file is missing: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row:
                    continue
                try:
                    indices.add(int(row[-1]))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid ground-truth index in {path}: {row}") from exc
    return indices


def load_nodes(conn, semantic_mode: str, max_nodes_per_type: int) -> list[AuditNode]:
    """Load and tokenize bounded ClearScope E5 nodes from PostgreSQL."""
    queries = [
        (
            "file",
            """
            select index_id, node_uuid, path
            from file_node_table
            order by index_id
            limit %s
            """,
        ),
        (
            "subject",
            """
            select index_id, node_uuid, path, cmd
            from subject_node_table
            order by index_id
            limit %s
            """,
        ),
        (
            "netflow",
            """
            select index_id, node_uuid, src_addr, src_port, dst_addr, dst_port
            from netflow_node_table
            order by index_id
            limit %s
            """,
        ),
    ]
    nodes: list[AuditNode] = []
    with conn.cursor() as cur:
        for node_type, sql in queries:
            cur.execute(sql, (int(max_nodes_per_type),))
            columns = [desc[0] for desc in cur.description]
            for values in cur.fetchall():
                row = dict(zip(columns, values))
                row["node_type"] = node_type
                nodes.append(tokenize_node_row(row, semantic_mode=semantic_mode))
    return nodes


def load_events(conn, max_events_per_split: int) -> list[dict[str, object]]:
    """Load bounded E5 event rows by configured split days."""
    from scripts.pipeline.io.db_stream import _day_bounds

    year_month = "2019-05"
    split_days = {
        "train": [8, 9],
        "val": [11],
        "test": [14, 15, 17],
    }
    events: list[dict[str, object]] = []
    with conn.cursor() as cur:
        for split, days in split_days.items():
            remaining = int(max_events_per_split)
            for day in days:
                if remaining <= 0:
                    break
                start_ns, end_ns = _day_bounds(year_month, day)
                cur.execute(
                    """
                    select operation, src_index_id, dst_index_id, timestamp_rec, _id
                    from event_table
                    where timestamp_rec >= %s and timestamp_rec < %s
                    order by timestamp_rec, _id
                    limit %s
                    """,
                    (start_ns, end_ns, remaining),
                )
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                for values in rows:
                    row = dict(zip(columns, values))
                    row["split"] = split
                    events.append(row)
                remaining -= len(rows)
    return events


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the E5 semantic audit smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default="clearscope_e5")
    parser.add_argument("--semantic_mode", default="raw_detail_v31_discriminative")
    parser.add_argument("--max_nodes_per_type", type=int, default=50000)
    parser.add_argument("--max_events_per_split", type=int, default=100000)
    parser.add_argument(
        "--output_dir",
        default="tmp/clearscope_e5_semantic_audit_smoke",
    )
    parser.add_argument(
        "--ground_truth",
        action="append",
        default=[
            "ground_truth/E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv",
            "ground_truth/E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv",
            "ground_truth/E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv",
        ],
    )
    return parser.parse_args(argv)


def run_audit(args: argparse.Namespace) -> dict[str, str]:
    """Run the bounded ClearScope E5 semantic audit smoke."""
    with connect_db(args.database) as conn:
        nodes = load_nodes(conn, args.semantic_mode, args.max_nodes_per_type)
        events = load_events(conn, args.max_events_per_split)
    nodes_by_index = {node.index_id: node for node in nodes}
    label_free_summary = build_label_free_summary(nodes)
    label_free_summary["event_tuple_summary"] = build_event_tuple_summary(events, nodes_by_index)
    malicious_index_ids = parse_ground_truth_indices(list(args.ground_truth))
    label_aware_diagnostics = build_label_aware_diagnostics(nodes, malicious_index_ids)
    return write_reports(
        output_dir=args.output_dir,
        label_free_summary=label_free_summary,
        label_aware_diagnostics=label_aware_diagnostics,
        fallback_rows=fallback_rows_from_summary(label_free_summary),
        collision_rows=build_collision_rows(nodes),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    paths = run_audit(args)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_semantic_audit_smoke
```

Expected: all tests pass.

- [ ] **Step 5: Run syntax checks**

Run:

```bash
python3 -m compileall scripts/tools/audit_clearscope_e5_semantic_smoke.py \
    tests/test_clearscope_e5_semantic_audit_smoke.py
```

Expected: both files compile successfully.

- [ ] **Step 6: Run real DB audit smoke**

Run:

```bash
CLAD_DB_HOST=127.0.0.1 CLAD_DB_PORT=5433 CLAD_DB_USER=postgres \
CLAD_DB_PASSWORD=123456 python3 scripts/tools/audit_clearscope_e5_semantic_smoke.py \
    --max_nodes_per_type 20000 \
    --max_events_per_split 50000 \
    --output_dir tmp/clearscope_e5_semantic_audit_smoke
```

Expected:

- command exits 0;
- writes `tmp/clearscope_e5_semantic_audit_smoke/label_free_summary.json`;
- writes `tmp/clearscope_e5_semantic_audit_smoke/label_aware_diagnostics.json`;
- writes `tmp/clearscope_e5_semantic_audit_smoke/fallback_counts.csv`;
- writes `tmp/clearscope_e5_semantic_audit_smoke/collision_groups.csv`.

- [ ] **Step 7: Inspect generated audit summaries**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path
root = Path("tmp/clearscope_e5_semantic_audit_smoke")
label_free = json.loads((root / "label_free_summary.json").read_text())
label_aware = json.loads((root / "label_aware_diagnostics.json").read_text())
print("node_count", label_free.get("node_count"))
print("fallback_counts", label_free.get("fallback_counts"))
print("event_tuple_summary", label_free.get("event_tuple_summary"))
print("label_warning", label_aware.get("warning"))
print("malicious_node_count", label_aware.get("malicious_node_count"))
print("malicious_fallback_counts", label_aware.get("malicious_fallback_counts"))
PY
```

Expected:

- `label_warning` is `label_aware_diagnostic_only_not_runtime_policy`;
- label-free summary has no `malicious` or `attack` top-level keys;
- fallback and OOV counts are visible for decision review.

- [ ] **Step 8: Commit Task 4**

Run:

```bash
git add scripts/tools/audit_clearscope_e5_semantic_smoke.py \
    tests/test_clearscope_e5_semantic_audit_smoke.py
git commit -m "feat: add clearscope e5 semantic audit smoke"
```

Do not add files under `tmp/`.

## Task 5: Decide Whether E5 Pipeline Smoke Is Safe

**Files:**
- Read: `tmp/clearscope_e5_semantic_audit_smoke/label_free_summary.json`
- Read: `tmp/clearscope_e5_semantic_audit_smoke/label_aware_diagnostics.json`
- Read: `tmp/clearscope_e5_semantic_audit_smoke/fallback_counts.csv`
- Read: `tmp/clearscope_e5_semantic_audit_smoke/collision_groups.csv`
- Modify only if proceeding: E5-specific runner or routing tests named in the decision.

- [ ] **Step 1: Summarize audit decision**

Prepare a short decision note in the final response with one of:

- `reuse_current_semantics_for_e5_smoke`
- `needs_general_tokenizer_adjustment_before_smoke`
- `semantic_mismatch_blocks_pipeline_smoke`

The decision must cite:

- fallback bucket counts;
- top collision risks;
- test OOV tuple count;
- malicious diagnostic fallback counts;
- whether any observed risk can be solved by a general Android shape rule.

- [ ] **Step 2: If decision is not reuse, stop**

If the decision is `needs_general_tokenizer_adjustment_before_smoke` or
`semantic_mismatch_blocks_pipeline_smoke`, do not touch runner or pipeline code.
Report the candidate general tokenizer adjustment and ask for a separate spec.

- [ ] **Step 3: If decision is reuse, inspect E5 artifact availability**

Run:

```bash
find outputs -maxdepth 5 \( -path '*CLEARSCOPE_E5*' -o -path '*clearscope_e5*' \) 2>/dev/null | sort
```

Expected:

- If no E5 artifacts exist, stop and report that bounded artifact build must be
  planned before pipeline smoke.
- If E5 artifacts exist, verify none of the intended smoke paths contain
  `CLEARSCOPE_E3`.

- [ ] **Step 4: If E5 artifacts exist, dry-run an E5 smoke command only**

Use an E5-specific runner or a runner branch that fails fast on mixed artifacts.
Do not use `run_clearscope_e3_v3_phase3e_phase3g_full.sh` unless a prior task has
made it dataset-safe for E5 and tests prove it does not hardcode E3 artifact names.

The dry run must include:

```bash
DATASET=CLEARSCOPE_E5
SEMANTIC_MODE=raw_detail_v31_discriminative
MAX_TRAIN_EVENTS=200000
MAX_REF_EVENTS=50000
MAX_TEST_EVENTS=100000
OUT_TAG_OVERRIDE=CLEARSCOPE_E5_RAW_DETAIL_V31_DISCRIMINATIVE_SMOKE_AUDIT_GATE
DRY_RUN=1
```

Expected:

- command exits 0;
- printed command contains `CLEARSCOPE_E5`;
- printed command does not contain `CLEARSCOPE_E3`;
- no inference is launched.

- [ ] **Step 5: Stop before real pipeline smoke**

After dry-run validation, ask the user before launching a real bounded E5 pipeline
smoke. The approved spec does not authorize automatic full inference.

## Final Verification

Run:

```bash
python3 -m unittest tests.test_clearscope_e5_semantic_audit_smoke
python3 -m compileall scripts/tools/audit_clearscope_e5_semantic_smoke.py \
    tests/test_clearscope_e5_semantic_audit_smoke.py
```

If Task 4 was executed, also verify:

```bash
test -f tmp/clearscope_e5_semantic_audit_smoke/label_free_summary.json
test -f tmp/clearscope_e5_semantic_audit_smoke/label_aware_diagnostics.json
test -f tmp/clearscope_e5_semantic_audit_smoke/fallback_counts.csv
test -f tmp/clearscope_e5_semantic_audit_smoke/collision_groups.csv
```

## Completion Report Requirements

The final response must include:

- changed files;
- commands run;
- audit outputs checked;
- audit decision;
- remaining risks;
- whether streaming inference, no leakage, explainability, and reproducibility are
  still preserved.
