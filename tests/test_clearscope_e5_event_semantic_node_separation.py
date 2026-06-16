import unittest

from scripts.tools.diagnose_clearscope_e5_event_semantic_node_separation import (
    compute_event_semantic_node_scores,
    compute_topk_metrics,
    simulate_both_cold_gates,
)


class ClearScopeE5EventSemanticNodeSeparationTests(unittest.TestCase):
    def test_event_semantic_scores_are_aggregated_by_node(self):
        rows = [
            {
                "src_idx": "1",
                "dst_idx": "2",
                "score": "0.10",
                "threshold": "0.20",
                "target_case": "event_semantic_target",
            },
            {
                "src_idx": "1",
                "dst_idx": "3",
                "score": "0.30",
                "threshold": "0.20",
                "target_case": "event_semantic_target",
            },
            {
                "src_idx": "2",
                "dst_idx": "4",
                "score": "0.90",
                "threshold": "0.20",
                "target_case": "both_cold_action_target",
            },
        ]

        node_rows = compute_event_semantic_node_scores(
            score_rows=rows,
            gt_nodes={1, 5},
            current_alert_nodes={2, 4},
            current_gt_alert_nodes={1},
            current_fp_nodes={2, 4},
            near_threshold_ratio=0.5,
        )

        by_node = {row["node_idx"]: row for row in node_rows}
        self.assertEqual(by_node[1]["node_label"], "malicious")
        self.assertEqual(by_node[1]["node_group"], "current_both_cold_tp_node")
        self.assertEqual(by_node[1]["event_semantic_event_count"], 2)
        self.assertAlmostEqual(by_node[1]["event_semantic_max_score"], 0.30)
        self.assertEqual(by_node[1]["event_semantic_above_threshold_count"], 1)
        self.assertEqual(by_node[2]["node_group"], "current_fp_node")
        self.assertEqual(by_node[5]["node_label"], "malicious")
        self.assertEqual(by_node[5]["event_semantic_event_count"], 0)

    def test_topk_metrics_report_gt_coverage_for_score_fields(self):
        node_rows = [
            {"node_idx": 10, "event_semantic_max_score": 0.9},
            {"node_idx": 11, "event_semantic_max_score": 0.8},
            {"node_idx": 12, "event_semantic_max_score": 0.1},
        ]

        metrics = compute_topk_metrics(
            node_rows,
            gt_nodes={11, 12},
            score_fields=["event_semantic_max_score"],
            k_values=[1, 2, 3],
        )

        by_k = {
            (row["score_field"], row["k"]): row
            for row in metrics
        }
        self.assertEqual(by_k[("event_semantic_max_score", 1)]["tp"], 0)
        self.assertEqual(by_k[("event_semantic_max_score", 2)]["tp"], 1)
        self.assertEqual(by_k[("event_semantic_max_score", 3)]["tp"], 2)
        self.assertAlmostEqual(by_k[("event_semantic_max_score", 3)]["recall"], 1.0)

    def test_both_cold_gate_simulation_selects_before_labeling(self):
        alert_rows = [
            {
                "src_idx": "1",
                "dst_idx": "20",
                "target_case": "both_cold_action_target",
                "endpoint_action_key": "a",
                "event_score": "0.50",
            },
            {
                "src_idx": "1",
                "dst_idx": "20",
                "target_case": "both_cold_action_target",
                "endpoint_action_key": "a",
                "event_score": "0.40",
            },
            {
                "src_idx": "30",
                "dst_idx": "40",
                "target_case": "both_cold_action_target",
                "endpoint_action_key": "b",
                "event_score": "0.60",
            },
            {
                "src_idx": "50",
                "dst_idx": "60",
                "target_case": "event_semantic_target",
                "endpoint_action_key": "c",
                "event_score": "0.80",
            },
        ]
        event_semantic_scores = {
            1: {"event_semantic_p95_score": 0.11},
            30: {"event_semantic_p95_score": 0.01},
        }

        diagnostics = simulate_both_cold_gates(
            alert_rows=alert_rows,
            gt_nodes={1},
            event_semantic_scores=event_semantic_scores,
            event_semantic_global_p999=0.10,
        )

        by_gate = {row["gate_name"]: row for row in diagnostics}
        self.assertEqual(by_gate["node_repeat_ge_2"]["retained_event_alerts"], 2)
        self.assertEqual(by_gate["node_repeat_ge_2"]["retained_tp_any_endpoint"], 2)
        self.assertEqual(by_gate["node_repeat_ge_2"]["retained_fp"], 0)
        self.assertEqual(by_gate["endpoint_action_repeat_ge_2"]["retained_event_alerts"], 2)
        self.assertTrue(by_gate["node_repeat_ge_2"]["labels_used_after_selection"])


if __name__ == "__main__":
    unittest.main()
