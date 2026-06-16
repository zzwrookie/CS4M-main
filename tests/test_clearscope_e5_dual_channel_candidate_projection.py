import unittest

from scripts.tools.project_clearscope_e5_dual_channel_candidates import (
    build_channel_b_variants,
    project_candidate_matrix,
    recommend_candidates,
)


class ClearScopeE5DualChannelCandidateProjectionTests(unittest.TestCase):
    def test_channel_b_requires_score_floor_and_near_threshold_support(self):
        node_rows = [
            {
                "node_idx": "1",
                "event_semantic_max_score": "0.20",
                "event_semantic_p95_score": "0.18",
                "event_semantic_event_count": "7",
                "event_semantic_near_threshold_count": "2",
                "event_semantic_above_threshold_count": "0",
            },
            {
                "node_idx": "2",
                "event_semantic_max_score": "0.20",
                "event_semantic_p95_score": "0.18",
                "event_semantic_event_count": "5",
                "event_semantic_near_threshold_count": "1",
                "event_semantic_above_threshold_count": "0",
            },
            {
                "node_idx": "3",
                "event_semantic_max_score": "0.10",
                "event_semantic_p95_score": "0.09",
                "event_semantic_event_count": "9",
                "event_semantic_near_threshold_count": "9",
                "event_semantic_above_threshold_count": "0",
            },
        ]

        variants = build_channel_b_variants(node_rows, {1, 3}, score_floor=0.15)

        self.assertEqual(variants["B_none"]["selected_nodes"], set())
        self.assertEqual(variants["B_node_support_ge_1"]["selected_nodes"], {1, 2})
        self.assertEqual(variants["B_node_support_ge_2"]["selected_nodes"], {1})
        self.assertEqual(variants["B_node_support_ge_1"]["tp_nodes"], {1})
        self.assertEqual(variants["B_node_support_ge_1"]["fp_nodes"], {2})

    def test_projection_matrix_reports_node_union_and_event_counts(self):
        channel_a = {
            "A_full": {
                "selected_event_count": 3,
                "tp_event_count": 2,
                "selected_nodes": {1, 2, 9},
                "tp_nodes": {1, 2},
                "fp_nodes": {9},
                "selected_event_rows": [
                    {"stream_pos": "1", "event_has_gt_node": True},
                    {"stream_pos": "2", "event_has_gt_node": True},
                    {"stream_pos": "3", "event_has_gt_node": False},
                ],
            },
        }
        channel_b = {
            "B_node_support_ge_1": {
                "selected_nodes": {2, 3, 10},
                "tp_nodes": {2, 3},
                "fp_nodes": {10},
                "node_rows": {
                    2: {
                        "event_semantic_event_count": 4,
                        "event_semantic_near_threshold_count": 1,
                        "event_semantic_above_threshold_count": 0,
                    },
                    3: {
                        "event_semantic_event_count": 5,
                        "event_semantic_near_threshold_count": 2,
                        "event_semantic_above_threshold_count": 0,
                    },
                    10: {
                        "event_semantic_event_count": 6,
                        "event_semantic_near_threshold_count": 3,
                        "event_semantic_above_threshold_count": 0,
                    },
                },
                "validation_only": True,
                "uses_topk": False,
            },
        }

        rows, node_rows, event_rows = project_candidate_matrix(
            channel_a_variants=channel_a,
            channel_b_variants=channel_b,
            gt_nodes={1, 2, 3},
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["candidate_name"], "A_full__B_node_support_ge_1")
        self.assertEqual(row["channel_a_event_count"], 3)
        self.assertEqual(row["channel_a_tp_event_count"], 2)
        self.assertEqual(row["union_tp_node_count"], 3)
        self.assertEqual(row["union_fp_node_count"], 2)
        self.assertEqual(row["only_channel_b_tp_node_count"], 1)
        self.assertEqual(row["event_semantic_support_event_count"], 15)
        self.assertEqual(row["event_semantic_near_threshold_count"], 6)
        memberships = {item["node_idx"]: item["node_membership"] for item in node_rows}
        self.assertEqual(memberships[1], "only_channel_a_tp")
        self.assertEqual(memberships[2], "both_channels_tp")
        self.assertEqual(memberships[3], "only_channel_b_tp")
        self.assertEqual(memberships[9], "channel_a_fp")
        self.assertEqual(memberships[10], "channel_b_fp")
        self.assertEqual(len(event_rows), 3)

    def test_recommend_candidates_applies_conservative_criteria(self):
        rows = [
            {
                "candidate_name": "A_full__B_none",
                "union_tp_node_count": 9,
                "union_fp_node_count": 10,
                "channel_b_validation_only": True,
                "channel_b_uses_topk": False,
            },
            {
                "candidate_name": "A_full__B_node_support_ge_1",
                "union_tp_node_count": 14,
                "union_fp_node_count": 19,
                "channel_b_validation_only": True,
                "channel_b_uses_topk": False,
            },
            {
                "candidate_name": "A_full__B_node_support_ge_2",
                "union_tp_node_count": 14,
                "union_fp_node_count": 25,
                "channel_b_validation_only": True,
                "channel_b_uses_topk": False,
            },
        ]

        evaluated = recommend_candidates(rows, baseline_name="A_full__B_none")
        by_name = {row["candidate_name"]: row for row in evaluated}

        self.assertTrue(by_name["A_full__B_node_support_ge_1"]["recommended"])
        self.assertFalse(by_name["A_full__B_node_support_ge_2"]["recommended"])
        self.assertIn(
            "fp_node_count_gt_2x_baseline",
            by_name["A_full__B_node_support_ge_2"]["rejection_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
