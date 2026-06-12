from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import textwrap
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


if __name__ == "__main__":
    unittest.main()
