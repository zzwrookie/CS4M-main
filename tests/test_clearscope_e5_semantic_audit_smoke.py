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


if __name__ == "__main__":
    unittest.main()
