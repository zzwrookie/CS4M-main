from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import patch

from scripts.tools.audit_clearscope_e5_semantic_smoke import (
    FALLBACK_TOKENS,
    AuditNode,
    assign_split,
    build_label_aware_diagnostics,
    build_label_free_summary,
    collect_fallback_counts,
    extract_detail_token,
)


class FakeCursor:
    """Small DB cursor fake for audit loader tests."""

    def __init__(self, responses: list[tuple[list[str], list[tuple[object, ...]]]]):
        self.responses = responses
        self.queries: list[tuple[str, tuple[object, ...]]] = []
        self.description: list[tuple[str]] = []
        self.rows: list[tuple[object, ...]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...]) -> None:
        self.queries.append((sql, params))
        columns, rows = self.responses.pop(0)
        self.description = [(column,) for column in columns]
        self.rows = rows

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.rows


class FakeConnection:
    """Small DB connection fake that returns one reusable cursor."""

    def __init__(self, responses: list[tuple[list[str], list[tuple[object, ...]]]]):
        self.cursor_obj = FakeCursor(responses)

    def cursor(self) -> FakeCursor:
        return self.cursor_obj


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
            semantic_tokens=(
                "file",
                "android_app_cache_file",
                "fennec_firefox_dev_cache2_entries_mixedid",
            ),
        )

        self.assertEqual(
            extract_detail_token(node),
            "fennec_firefox_dev_cache2_entries_mixedid",
        )

    def test_collect_fallback_counts_tracks_known_fallback_tokens(self) -> None:
        nodes = [
            AuditNode(1, "a", "file", "/unknown", ("file", "file_other", "other")),
            AuditNode(2, "b", "file", "", ("file", "unknown_file", "unknown")),
            AuditNode(
                3,
                "c",
                "netflow",
                "10.0.0.1:1->10.0.0.2:2",
                ("netflow", "fixed", "netflow"),
            ),
            AuditNode(
                4,
                "d",
                "file",
                "/system/bin/sh",
                ("file", "android_system_file", "bin_sh"),
            ),
        ]

        counts = collect_fallback_counts(nodes)

        self.assertEqual(counts["file_other"], 1)
        self.assertEqual(counts["unknown_file"], 1)
        self.assertEqual(counts["netflow"], 1)
        self.assertNotIn("android_system_file", FALLBACK_TOKENS)

    def test_label_free_summary_does_not_include_label_fields(self) -> None:
        nodes = [
            AuditNode(
                1,
                "benign",
                "file",
                "/system/bin/toybox",
                ("file", "android_system_file", "bin_toybox"),
            ),
            AuditNode(
                2,
                "mal",
                "file",
                "/data/local/tmp/tester",
                ("file", "android_tmp_file", "tmp_file"),
            ),
        ]

        summary = build_label_free_summary(nodes)

        self.assertEqual(summary["node_type_counts"]["file"], 2)
        self.assertNotIn("malicious", summary)
        self.assertNotIn("attack", summary)
        self.assertIn("fallback_counts", summary)

    def test_label_aware_diagnostics_are_separate_appendix(self) -> None:
        nodes = [
            AuditNode(
                1,
                "benign",
                "file",
                "/system/bin/toybox",
                ("file", "android_system_file", "bin_toybox"),
            ),
            AuditNode(
                2,
                "mal",
                "file",
                "/data/local/tmp/tester",
                ("file", "android_tmp_file", "tmp_file"),
            ),
        ]

        diagnostics = build_label_aware_diagnostics(nodes, malicious_index_ids={2})

        self.assertEqual(diagnostics["malicious_node_count"], 1)
        self.assertEqual(diagnostics["malicious_fallback_counts"], {})
        self.assertIn("label_aware_diagnostic_only", diagnostics["warning"])

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

    def test_tokenize_file_row_v31_keeps_sdcardfs_appid_coarse(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import tokenize_node_row

        node = tokenize_node_row(
            {
                "index_id": 14,
                "node_uuid": "file-node-v31",
                "node_type": "file",
                "path": "/config/sdcardfs/de.belu.appstarter/appid",
            },
            semantic_mode="raw_detail_v31_discriminative",
        )

        self.assertEqual(node.semantic_tokens, ("file", "file_other", "other"))

    def test_tokenize_file_row_v33_uses_sdcardfs_appid_detail(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import tokenize_node_row

        node = tokenize_node_row(
            {
                "index_id": 15,
                "node_uuid": "file-node-v33",
                "node_type": "file",
                "path": "/config/sdcardfs/de.belu.appstarter/appid",
            },
            semantic_mode="raw_detail_v33_e5_android_safe",
        )

        self.assertEqual(
            node.semantic_tokens,
            ("file", "android_sdcardfs_appid", "de_belu_appstarter_appid"),
        )

    def test_tokenize_file_row_v33b_uses_accounting_detail(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import tokenize_node_row

        node = tokenize_node_row(
            {
                "index_id": 17,
                "node_uuid": "file-node-v33b",
                "node_type": "file",
                "path": "/acct/uid_0/pid_549",
            },
            semantic_mode="raw_detail_v33b_e5_android_safe",
        )

        self.assertEqual(
            node.semantic_tokens,
            ("file", "android_accounting_pid", "acct_pid"),
        )

    def test_tokenize_file_row_accepts_v31_semantic_alias(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import tokenize_node_row

        node = tokenize_node_row(
            {
                "index_id": 13,
                "node_uuid": "file-node-alias",
                "node_type": "file",
                "path": "/data/local/tmp/tester",
            },
            semantic_mode="clearscope_raw_detail_v31_discriminative",
        )

        self.assertEqual(node.node_type, "file")
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

    def test_tokenize_netflow_row_keeps_fixed_clear_scope_netflow_under_v33(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import tokenize_node_row

        node = tokenize_node_row(
            {
                "index_id": 16,
                "node_uuid": "netflow-node-v33",
                "node_type": "netflow",
                "src_addr": "10.0.0.1",
                "src_port": "123",
                "dst_addr": "10.0.0.2",
                "dst_port": "443",
            },
            semantic_mode="raw_detail_v33_e5_android_safe",
        )

        self.assertEqual(node.node_type, "netflow")
        self.assertEqual(node.semantic_tokens, ("netflow",))

    def test_tokenize_netflow_row_keeps_fixed_clear_scope_netflow_under_v33b(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import tokenize_node_row

        node = tokenize_node_row(
            {
                "index_id": 18,
                "node_uuid": "netflow-node-v33b",
                "node_type": "netflow",
                "src_addr": "10.0.0.1",
                "src_port": "123",
                "dst_addr": "10.0.0.2",
                "dst_port": "443",
            },
            semantic_mode="raw_detail_v33b_e5_android_safe",
        )

        self.assertEqual(node.semantic_tokens, ("netflow",))

    def test_module_import_without_psycopg2_and_connect_db_error_is_clear(self) -> None:
        code = textwrap.dedent(
            """
            import builtins

            real_import = builtins.__import__

            def blocked_import(name, *args, **kwargs):
                if name == "psycopg2":
                    raise ModuleNotFoundError("No module named 'psycopg2'")
                return real_import(name, *args, **kwargs)

            builtins.__import__ = blocked_import

            from scripts.tools.audit_clearscope_e5_semantic_smoke import connect_db

            try:
                connect_db("clearscope_e5")
            except ModuleNotFoundError as error:
                if "psycopg2 is required for connect_db" not in str(error):
                    raise
            else:
                raise AssertionError("connect_db unexpectedly succeeded")
            """
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            cwd=".",
            text=True,
            capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_build_event_tuple_summary_counts_support_and_oov(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import (
            build_event_tuple_summary,
        )

        nodes = {
            1: AuditNode(
                1,
                "p",
                "subject",
                "proc",
                ("process", "native_process", "toybox"),
            ),
            2: AuditNode(
                2,
                "f",
                "file",
                "/system/bin/sh",
                ("file", "android_system_file", "bin_sh"),
            ),
            3: AuditNode(
                3,
                "x",
                "file",
                "/data/local/tmp/x",
                ("file", "android_tmp_file", "tmp_file"),
            ),
        }
        events = [
            {
                "split": "train",
                "operation": "EVENT_READ",
                "src_index_id": 1,
                "dst_index_id": 2,
            },
            {
                "split": "test",
                "operation": "EVENT_READ",
                "src_index_id": 1,
                "dst_index_id": 2,
            },
            {
                "split": "test",
                "operation": "EVENT_WRITE",
                "src_index_id": 1,
                "dst_index_id": 3,
            },
            {
                "split": "val",
                "operation": "EVENT_WRITE",
                "src_index_id": 1,
                "dst_index_id": 3,
            },
            {
                "split": "val",
                "operation": "EVENT_READ",
                "src_index_id": 1,
                "dst_index_id": 2,
            },
            {
                "split": "test",
                "operation": "EVENT_READ",
                "src_index_id": 999,
                "dst_index_id": 2,
            },
            {
                "split": "test",
                "operation": "EVENT_READ",
                "src_index_id": 1,
                "dst_index_id": 999,
            },
            {
                "split": "test",
                "operation": "EVENT_READ",
                "src_index_id": 999,
                "dst_index_id": 998,
            },
        ]

        summary = build_event_tuple_summary(events, nodes)

        self.assertEqual(summary["input_event_count"], 8)
        self.assertEqual(summary["event_count"], 5)
        self.assertEqual(summary["skipped_event_count"], 3)
        self.assertEqual(summary["skipped_missing_src_count"], 2)
        self.assertEqual(summary["skipped_missing_dst_count"], 2)
        self.assertEqual(summary["val_tuple_count"], 2)
        self.assertEqual(summary["val_oov_tuple_count"], 1)
        self.assertEqual(summary["val_seen_tuple_count"], 1)
        self.assertEqual(summary["test_oov_tuple_count"], 1)
        self.assertEqual(summary["test_seen_tuple_count"], 1)
        self.assertEqual(
            summary["top_test_oov_tuples"],
            [
                {
                    "tuple": [
                        "EVENT_WRITE",
                        "subject",
                        "file",
                        "toybox",
                        "tmp_file",
                    ],
                    "count": 1,
                }
            ],
        )
        self.assertEqual(
            summary["top_val_oov_tuples"],
            [
                {
                    "tuple": [
                        "EVENT_WRITE",
                        "subject",
                        "file",
                        "toybox",
                        "tmp_file",
                    ],
                    "count": 1,
                }
            ],
        )

    def test_collision_rows_sanitize_accounting_high_cardinality_examples(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import build_collision_rows

        nodes = [
            AuditNode(
                1,
                "acct-a",
                "file",
                "/acct/uid_0/pid_549",
                ("file", "android_accounting_pid", "acct_pid"),
            ),
            AuditNode(
                2,
                "acct-b",
                "file",
                "/acct/uid_1000/pid_12345",
                ("file", "android_accounting_pid", "acct_pid"),
            ),
        ]

        rows = build_collision_rows(nodes)

        self.assertEqual(rows[0]["token"], "acct_pid")
        self.assertEqual(rows[0]["raw_detail_count"], 2)
        self.assertEqual(rows[0]["examples"], ["/acct/uid_<uid>/pid_<pid>"])

    def test_write_reports_creates_json_and_csv(self) -> None:
        from tempfile import TemporaryDirectory

        from scripts.tools.audit_clearscope_e5_semantic_smoke import write_reports

        with TemporaryDirectory() as tmpdir:
            paths = write_reports(
                output_dir=tmpdir,
                label_free_summary={"node_count": 1, "fallback_counts": {"netflow": 1}},
                label_aware_diagnostics={
                    "warning": "label_aware_diagnostic_only_not_runtime_policy"
                },
                fallback_rows=[{"token": "netflow", "count": 1}],
                collision_rows=[
                    {
                        "token": "tmp_file",
                        "raw_detail_count": 2,
                        "examples": ["/data/local/tmp/a", "/data/local/tmp/b"],
                    }
                ],
            )

            self.assertTrue(paths["label_free_json"].endswith("label_free_summary.json"))
            self.assertTrue(
                paths["label_aware_json"].endswith("label_aware_diagnostics.json")
            )
            self.assertTrue(paths["fallback_csv"].endswith("fallback_counts.csv"))
            self.assertTrue(paths["collision_csv"].endswith("collision_groups.csv"))

            for path in paths.values():
                self.assertTrue(Path(path).is_file(), path)

            label_free = json.loads(Path(paths["label_free_json"]).read_text())
            self.assertEqual(label_free["node_count"], 1)
            self.assertEqual(label_free["fallback_counts"], {"netflow": 1})

            fallback_csv = Path(paths["fallback_csv"]).read_text()
            self.assertIn("netflow", fallback_csv)
            self.assertTrue(
                "token,count" in fallback_csv or "count,token" in fallback_csv
            )

            collision_csv = Path(paths["collision_csv"]).read_text()
            self.assertIn("tmp_file", collision_csv)
            collision_rows = list(
                csv.DictReader(Path(paths["collision_csv"]).read_text().splitlines())
            )
            self.assertEqual(
                json.loads(collision_rows[0]["examples"]),
                ["/data/local/tmp/a", "/data/local/tmp/b"],
            )

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

    def test_parse_args_custom_ground_truth_replaces_defaults(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import parse_args

        default_args = parse_args([])
        custom_args = parse_args(["--ground_truth", "custom.csv"])

        self.assertEqual(
            default_args.ground_truth,
            [
                "ground_truth/E5-CLEARSCOPE/node_clearscope_e5_appstarter_0515.csv",
                "ground_truth/E5-CLEARSCOPE/node_clearscope_e5_lockwatch_0517.csv",
                "ground_truth/E5-CLEARSCOPE/node_clearscope_e5_tester_0517.csv",
            ],
        )
        self.assertEqual(custom_args.ground_truth, ["custom.csv"])

    def test_load_nodes_uses_configured_subject_table(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import load_nodes

        conn = FakeConnection(
            [
                (
                    ["index_id", "node_uuid", "path"],
                    [(1, "file-1", "/system/bin/sh")],
                ),
                (
                    ["index_id", "node_uuid", "path", "cmd"],
                    [(2, "subject-1", "/system/bin/app_process", "com.example")],
                ),
                (
                    [
                        "index_id",
                        "node_uuid",
                        "src_addr",
                        "src_port",
                        "dst_addr",
                        "dst_port",
                    ],
                    [(3, "netflow-1", "10.0.0.1", "1", "10.0.0.2", "2")],
                ),
            ]
        )

        with patch.dict(os.environ, {"CLAD_SUBJECT_NODE_TABLE": "custom_subjects"}):
            nodes = load_nodes(conn, "raw_detail_v31_discriminative", 10)

        self.assertEqual([node.index_id for node in nodes], [1, 2, 3])
        subject_sql = conn.cursor_obj.queries[1][0]
        self.assertIn("from custom_subjects", subject_sql)

    def test_load_nodes_by_index_ids_queries_endpoint_nodes(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import (
            load_nodes_by_index_ids,
        )

        conn = FakeConnection(
            [
                (
                    ["index_id", "node_uuid", "path"],
                    [(10, "file-10", "/data/local/tmp/a")],
                ),
                (
                    ["index_id", "node_uuid", "path", "cmd"],
                    [(20, "subject-20", "/system/bin/app_process", "com.example")],
                ),
                (
                    [
                        "index_id",
                        "node_uuid",
                        "src_addr",
                        "src_port",
                        "dst_addr",
                        "dst_port",
                    ],
                    [(30, "netflow-30", "10.0.0.1", "1", "10.0.0.2", "2")],
                ),
            ]
        )

        with patch.dict(os.environ, {"CLAD_SUBJECT_NODE_TABLE": "custom_subjects"}):
            nodes = load_nodes_by_index_ids(
                conn,
                "raw_detail_v31_discriminative",
                {10, 20, 30, 999},
            )

        self.assertEqual(sorted(node.index_id for node in nodes), [10, 20, 30])
        for _sql, params in conn.cursor_obj.queries:
            self.assertEqual(params, ((10, 20, 30, 999),))
        self.assertIn("from custom_subjects", conn.cursor_obj.queries[1][0])

    def test_event_tuple_summary_reports_usable_rate_and_warning(self) -> None:
        from scripts.tools.audit_clearscope_e5_semantic_smoke import (
            build_event_tuple_summary,
        )

        nodes = {
            1: AuditNode(1, "p", "subject", "proc", ("process", "native", "proc")),
            2: AuditNode(2, "f", "file", "/tmp/a", ("file", "android_tmp", "a")),
        }
        events = [
            {
                "split": "train",
                "operation": "EVENT_READ",
                "src_index_id": 1,
                "dst_index_id": 2,
            },
            {
                "split": "test",
                "operation": "EVENT_READ",
                "src_index_id": 999,
                "dst_index_id": 2,
            },
        ]

        summary = build_event_tuple_summary(events, nodes)

        self.assertEqual(summary["skipped_rate"], 0.5)
        self.assertEqual(summary["usable_event_rate"], 0.5)
        self.assertIn("high_skipped_event_rate", summary["report_warnings"])

    def test_run_audit_supplements_event_endpoint_nodes(self) -> None:
        import argparse

        from scripts.tools import audit_clearscope_e5_semantic_smoke as audit

        sampled_nodes = [
            AuditNode(1, "sample", "subject", "sample", ("process", "native", "sample")),
        ]
        endpoint_nodes = [
            AuditNode(2, "dst", "file", "/tmp/a", ("file", "android_tmp", "a")),
        ]
        events = [
            {
                "split": "train",
                "operation": "EVENT_READ",
                "src_index_id": 1,
                "dst_index_id": 2,
            }
        ]
        captured: dict[str, object] = {}

        class DummyConnection:
            def __enter__(self) -> "DummyConnection":
                return self

            def __exit__(self, *args: object) -> None:
                return None

        def fake_write_reports(**kwargs: object) -> dict[str, str]:
            captured.update(kwargs)
            return {"label_free_json": "out.json"}

        args = argparse.Namespace(
            database="clearscope_e5",
            semantic_mode="raw_detail_v31_discriminative",
            max_nodes_per_type=10,
            max_events_per_split=10,
            ground_truth=[],
            output_dir="tmp/out",
        )

        with patch.object(audit, "connect_db", return_value=DummyConnection()):
            with patch.object(audit, "load_nodes", return_value=sampled_nodes):
                with patch.object(audit, "load_events", return_value=events):
                    with patch.object(
                        audit,
                        "load_nodes_by_index_ids",
                        return_value=endpoint_nodes,
                    ) as load_endpoints:
                        with patch.object(
                            audit,
                            "parse_ground_truth_indices",
                            return_value=set(),
                        ):
                            with patch.object(audit, "write_reports", fake_write_reports):
                                paths = audit.run_audit(args)

        load_endpoints.assert_called_once()
        self.assertEqual(paths, {"label_free_json": "out.json"})
        summary = captured["label_free_summary"]
        self.assertEqual(summary["node_count"], 1)
        self.assertEqual(summary["endpoint_node_count"], 1)
        self.assertEqual(summary["event_tuple_summary"]["skipped_event_count"], 0)
        self.assertEqual(summary["event_tuple_summary"]["usable_event_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
