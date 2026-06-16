import unittest

from scripts.tools.audit_cadets_e3_e4_v2_v3_regression_policy_speed import (
    build_group_regression_rows,
    build_policy_candidate_projection_rows,
    build_speed_rows,
    build_tp_node_delta_rows,
    join_demotion_with_sweep,
)


class CadetsE3E4V2V3RegressionPolicySpeedAuditTests(unittest.TestCase):
    def test_tp_node_delta_categorizes_overlap_and_misses(self):
        rows = build_tp_node_delta_rows(
            v2_tp_nodes={1, 2, 3},
            v3_tp_nodes={3, 4},
            total_gt_nodes=5,
            v2_nodes={1: {"node_type": "process"}, 2: {"node_type": "file"}, 3: {}},
            v3_nodes={3: {"node_type": "process"}, 4: {"node_type": "netflow"}},
        )

        by_category = {}
        for row in rows:
            by_category.setdefault(row["coverage_category"], set()).add(row["node_idx"])

        self.assertEqual(by_category["both_covered"], {3})
        self.assertEqual(by_category["v2_only"], {1, 2})
        self.assertEqual(by_category["v3_only"], {4})
        self.assertEqual(by_category["both_missed"], {"unknown_gt_1"})

    def test_group_regression_computes_deltas_and_focus(self):
        rows = build_group_regression_rows(
            [
                {
                    "action": "EVENT_OPEN",
                    "src_type": "process",
                    "dst_type": "process",
                    "alert_count": "20",
                    "TP": "10",
                    "FP": "10",
                    "covered_malicious_nodes": "5",
                    "precision": "0.5",
                }
            ],
            [
                {
                    "action": "EVENT_OPEN",
                    "src_type": "process",
                    "dst_type": "process",
                    "alert_count": "5",
                    "TP": "2",
                    "FP": "3",
                    "covered_malicious_nodes": "2",
                    "precision": "0.4",
                }
            ],
        )

        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["focus_group"])
        self.assertEqual(rows[0]["tp_delta_v3_minus_v2"], -8)
        self.assertEqual(rows[0]["covered_node_delta_v3_minus_v2"], -3)

    def test_demotion_join_selects_best_sweep_candidate(self):
        rows = join_demotion_with_sweep(
            [
                {
                    "action": "EVENT_CONNECT",
                    "src_type": "process",
                    "dst_type": "netflow",
                    "target_case": "event_semantic_target",
                    "demoted_event_count": "39",
                }
            ],
            [
                {
                    "action": "EVENT_CONNECT",
                    "src_type": "process",
                    "dst_type": "netflow",
                    "threshold": "0.144",
                    "alert_count": "15",
                    "TP": "15",
                    "FP": "0",
                    "precision": "1.0",
                    "covered_malicious_nodes": "22",
                    "strict_node_TP": "22",
                    "strict_node_FP": "2",
                    "margin": "0.02",
                    "quantile": "0.9995",
                    "threshold_policy": "per_action_type_quantile",
                }
            ],
        )

        self.assertEqual(rows[0]["best_candidate_alert_count"], 15)
        self.assertEqual(rows[0]["best_candidate_tp"], 15)
        self.assertEqual(rows[0]["best_candidate_precision"], 1.0)
        self.assertEqual(rows[0]["posthoc_actionability"], "strong_posthoc_candidate")

    def test_policy_projection_adds_candidates_without_changing_runtime(self):
        rows = build_policy_candidate_projection_rows(
            current={"event_alerts": 100, "event_tp": 20, "event_fp": 80, "node_tp": 10},
            baseline={"event_alerts": 120, "event_tp": 30, "event_fp": 90, "node_tp": 12},
            sweep_rows=[
                {
                    "action": "EVENT_CONNECT",
                    "src_type": "process",
                    "dst_type": "netflow",
                    "margin": "0.02",
                    "alert_count": "15",
                    "TP": "15",
                    "FP": "0",
                    "covered_malicious_nodes": "22",
                    "strict_node_TP": "22",
                    "strict_node_FP": "2",
                },
                {
                    "action": "EVENT_RECVFROM",
                    "src_type": "netflow",
                    "dst_type": "process",
                    "margin": "0.05",
                    "alert_count": "377",
                    "TP": "375",
                    "FP": "2",
                    "covered_malicious_nodes": "6",
                    "strict_node_TP": "6",
                    "strict_node_FP": "4",
                },
            ],
            total_gt_nodes=76,
        )

        names = {row["candidate_name"] for row in rows}
        self.assertIn("v3_current_policy", names)
        self.assertIn("v2_best_baseline", names)
        self.assertIn("v3_combined_high_precision_recoveries", names)
        combined = next(row for row in rows if row["candidate_name"] == "v3_combined_high_precision_recoveries")
        self.assertEqual(combined["rule_source"], "validation_runtime_visible_projection")
        self.assertEqual(combined["uses_test_labels_for_rule"], False)
        self.assertGreater(combined["projected_event_tp"], 20)

    def test_speed_rows_extract_core_runtime_config(self):
        rows = build_speed_rows(
            {
                "cadets": {
                    "dataset": "CADETS_E3",
                    "runtime": {"events_scored": 1000, "throughput_events_per_second": 2500},
                    "config": {
                        "node_embedding_lookup_mode": "compact_used_nodes",
                        "db_stream_mode_actual": "python_lookup",
                    },
                    "ofsm_compression": {"logical_nodes": 200},
                }
            }
        )

        self.assertEqual(rows[0]["dataset"], "CADETS_E3")
        self.assertEqual(rows[0]["throughput_events_per_second"], 2500.0)
        self.assertIn("python_lookup", rows[0]["likely_bottleneck_notes"])


if __name__ == "__main__":
    unittest.main()
