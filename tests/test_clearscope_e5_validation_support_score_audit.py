import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools import audit_clearscope_e5_validation_support_scores as audit


class ClearScopeE5ValidationSupportScoreAuditTests(unittest.TestCase):
    def test_audit_summarizes_unseen_both_cold_and_event_semantic_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group_path = root / "conditional_score_summary_by_target_action_type.csv"
            with group_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "target_case",
                        "action_name",
                        "src_type_name",
                        "dst_type_name",
                        "validation_count",
                        "test_count",
                        "threshold",
                        "threshold_level",
                        "validation_count_bucket",
                        "test_p999",
                        "test_max",
                        "alert_count",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "target_case": "both_cold_action_target",
                        "action_name": "EVENT_OPEN",
                        "src_type_name": "process",
                        "dst_type_name": "process",
                        "validation_count": "0",
                        "test_count": "100",
                        "threshold": "0",
                        "threshold_level": "unseen_group_extreme",
                        "validation_count_bucket": "unseen",
                        "test_p999": "0.5",
                        "test_max": "0.6",
                        "alert_count": "5",
                    },
                )
                writer.writerow(
                    {
                        "target_case": "both_cold_action_target",
                        "action_name": "EVENT_READ",
                        "src_type_name": "process",
                        "dst_type_name": "process",
                        "validation_count": "0",
                        "test_count": "50",
                        "threshold": "0",
                        "threshold_level": "unseen_group_extreme",
                        "validation_count_bucket": "unseen",
                        "test_p999": "0.1",
                        "test_max": "0.2",
                        "alert_count": "0",
                    },
                )
                writer.writerow(
                    {
                        "target_case": "event_semantic_target",
                        "action_name": "EVENT_READ",
                        "src_type_name": "process",
                        "dst_type_name": "process",
                        "validation_count": "0",
                        "test_count": "500",
                        "threshold": "0",
                        "threshold_level": "unseen_group_extreme",
                        "validation_count_bucket": "unseen",
                        "test_p999": "0.05",
                        "test_max": "0.12",
                        "alert_count": "0",
                    },
                )
            target_case_path = root / "target_case_summary.csv"
            with target_case_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "target_case",
                        "event_count",
                        "alert_count",
                        "threshold_mean",
                        "score_mean",
                        "score_p99",
                        "score_p999",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "target_case": "both_cold_action_target",
                        "event_count": "150",
                        "alert_count": "5",
                        "threshold_mean": "0.3",
                        "score_mean": "0.2",
                        "score_p99": "0.4",
                        "score_p999": "0.5",
                    },
                )
                writer.writerow(
                    {
                        "target_case": "event_semantic_target",
                        "event_count": "500",
                        "alert_count": "0",
                        "threshold_mean": "0.3",
                        "score_mean": "0.01",
                        "score_p99": "0.04",
                        "score_p999": "0.08",
                    },
                )
            score_summary_path = root / "score_summary.json"
            score_summary_path.write_text(
                json.dumps(
                    {
                        "final_threshold": 0.3,
                        "both_cold_unseen_policy": {"policy_name": "alert"},
                        "conditional_endpoint_suppression": {"enabled": False},
                    },
                ),
                encoding="utf-8",
            )

            result = audit.run_audit(
                result_dir=root,
                group_summary_path=group_path,
                target_case_summary_path=target_case_path,
                score_summary_path=score_summary_path,
                missed_gt_diagnosis_path=None,
            )

            summary = result["summary"]
            self.assertEqual(summary["group_count"], 3)
            self.assertEqual(summary["unseen_group_count"], 3)
            self.assertEqual(summary["both_cold_unseen_alert_group_count"], 1)
            self.assertEqual(summary["both_cold_unseen_alert_count"], 5)
            self.assertEqual(summary["event_semantic_alert_count"], 0)
            self.assertAlmostEqual(summary["event_semantic_threshold_gap"], 0.22)
            self.assertEqual(summary["recommended_next_experiment"], "validation_only_cold_start_gate_smoke")

            self.assertTrue((root / "e5_validation_support_score_audit.json").is_file())
            self.assertTrue((root / "e5_validation_support_score_audit.csv").is_file())


if __name__ == "__main__":
    unittest.main()
