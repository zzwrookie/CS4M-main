import unittest

from scripts.tools.audit_clearscope_e5_dual_channel_thresholds import (
    build_both_cold_channel_sets,
    build_validation_threshold_candidates,
    compute_score_collapse_summary,
    compute_union_coverage,
)


class ClearScopeE5DualChannelThresholdAuditTests(unittest.TestCase):
    def test_both_cold_support_gates_select_before_labeling(self):
        rows = [
            {
                "target_case": "both_cold_action_target",
                "src_idx": "1",
                "dst_idx": "2",
                "endpoint_action_key": "node:2:4",
            },
            {
                "target_case": "both_cold_action_target",
                "src_idx": "3",
                "dst_idx": "2",
                "endpoint_action_key": "node:2:4",
            },
            {
                "target_case": "both_cold_action_target",
                "src_idx": "4",
                "dst_idx": "5",
                "endpoint_action_key": "node:5:3",
            },
        ]

        channels = build_both_cold_channel_sets(rows, gt_nodes={2, 5})

        self.assertEqual(channels["both_cold_full"]["selected_event_count"], 3)
        self.assertEqual(channels["both_cold_full"]["tp_nodes"], {2, 5})
        self.assertEqual(
            channels["both_cold_node_repeat_ge_2"]["selected_event_count"],
            2,
        )
        self.assertEqual(channels["both_cold_node_repeat_ge_2"]["tp_nodes"], {2})
        self.assertFalse(
            channels["both_cold_node_repeat_ge_2"]["labels_used_for_selection"],
        )
        self.assertEqual(
            channels["both_cold_endpoint_action_repeat_ge_2"]["tp_nodes"],
            {2},
        )

    def test_union_coverage_reports_overlap_and_unique_tp_nodes(self):
        both_cold_channels = {
            "both_cold_full": {
                "tp_nodes": {1, 2, 3},
                "selected_nodes": {1, 2, 3, 9},
                "selected_event_count": 4,
            },
        }
        event_channels = [
            {
                "channel_name": "event_semantic_p95_score_top_1000",
                "score_field": "event_semantic_p95_score",
                "k": 1000,
                "selected_nodes": {3, 4, 5, 10},
                "tp_nodes": {3, 4, 5},
            },
        ]

        summary_rows, node_rows = compute_union_coverage(
            both_cold_channels=both_cold_channels,
            event_semantic_channels=event_channels,
            gt_nodes={1, 2, 3, 4, 5, 6},
        )

        self.assertEqual(len(summary_rows), 1)
        row = summary_rows[0]
        self.assertEqual(row["union_tp_node_count"], 5)
        self.assertEqual(row["overlap_tp_node_count"], 1)
        self.assertEqual(row["only_both_cold_tp_nodes"], [1, 2])
        self.assertEqual(row["only_event_semantic_tp_nodes"], [4, 5])
        memberships = {item["node_idx"]: item["node_membership"] for item in node_rows}
        self.assertEqual(memberships[1], "only_both_cold")
        self.assertEqual(memberships[3], "both_channels")
        self.assertEqual(memberships[4], "only_event_semantic")
        self.assertEqual(memberships[6], "missed_by_both")

    def test_score_collapse_summary_aggregates_groups_and_fp_clusters(self):
        score_rows = [
            {
                "target_case": "event_semantic_target",
                "src_idx": "1",
                "dst_idx": "7",
                "score": "0.20",
                "threshold": "0.30",
                "threshold_basis": "validation_quantile_event_semantic_target",
                "endpoint_action_key": "node:7:4",
                "action": "EVENT_READ",
                "src_type": "file",
                "dst_type": "process",
            },
            {
                "target_case": "event_semantic_target",
                "src_idx": "2",
                "dst_idx": "8",
                "score": "0.28",
                "threshold": "0.30",
                "threshold_basis": "validation_quantile_event_semantic_target",
                "endpoint_action_key": "node:8:4",
                "action": "EVENT_READ",
                "src_type": "file",
                "dst_type": "process",
            },
            {
                "target_case": "both_cold_action_target",
                "src_idx": "2",
                "dst_idx": "8",
                "score": "0.99",
                "threshold": "0.30",
            },
        ]

        summary = compute_score_collapse_summary(
            score_rows=score_rows,
            node_groups={1: "missed_gt_node", 2: "current_fp_node"},
            high_score_limit=10,
        )

        self.assertEqual(summary["event_semantic_row_count"], 2)
        self.assertEqual(
            summary["threshold_source_counts"],
            {"validation_quantile_event_semantic_target": 2},
        )
        groups = {row["node_group"]: row for row in summary["group_summary"]}
        self.assertAlmostEqual(groups["missed_gt_node"]["max_score"], 0.20)
        self.assertAlmostEqual(groups["current_fp_node"]["max_margin"], -0.02)
        cluster_kinds = {
            row["cluster_kind"]
            for row in summary["high_score_fp_clusters"]
        }
        self.assertEqual(cluster_kinds, {"endpoint_action", "action_type"})

    def test_validation_threshold_candidates_are_not_test_selected(self):
        summary_rows = [
            {
                "target_case": "event_semantic_target",
                "test_p999": "0.249",
                "parent_threshold": "0.276",
                "global_threshold": "0.300",
            },
        ]

        candidates = build_validation_threshold_candidates(summary_rows)

        names = {row["candidate_name"] for row in candidates}
        self.assertIn("event_semantic_validation_quantile_p999", names)
        self.assertIn("event_semantic_node_max_and_support_count", names)
        self.assertIn("both_cold_node_repeat_ge_2", names)
        for row in candidates:
            self.assertTrue(row["runtime_visible"])
            self.assertTrue(row["validation_only"])
            self.assertFalse(row["uses_test_labels_for_selection"])


if __name__ == "__main__":
    unittest.main()
