import csv
import tempfile
import unittest
from pathlib import Path

from scripts.tools.audit_cadets_e3_v3_connect_unseen_root_cause import (
    build_group_matrix,
    build_neighbor_rows,
    classify_connect_root_cause,
    run_audit,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class CadetsE3V3ConnectUnseenRootCauseTests(unittest.TestCase):
    def test_classifies_validation_absent_test_present(self):
        matrix = build_group_matrix(
            validation_rows=[
                {
                    "target_case": "event_semantic_target",
                    "action_name": "EVENT_CONNECT",
                    "src_type_name": "process",
                    "dst_type_name": "netflow",
                    "validation_count": "0",
                    "test_count": "2",
                    "final_threshold_source": "global_extreme",
                }
            ],
            score_rows=[
                {
                    "target_case": "event_semantic_target",
                    "action": "EVENT_CONNECT",
                    "src_type": "process",
                    "dst_type": "netflow",
                    "score": "0.05",
                    "threshold": "0.12",
                }
            ],
            policy_rows=[
                {
                    "action": "EVENT_CONNECT",
                    "src_type": "process",
                    "dst_type": "netflow",
                    "alert_count": "0",
                    "demoted_event_count": "0",
                }
            ],
        )

        root_cause = classify_connect_root_cause(matrix)

        self.assertEqual(root_cause["root_cause"], "validation_group_absent_test_group_present")
        self.assertEqual(root_cause["validation_count"], 0)
        self.assertEqual(root_cause["score_trace_count"], 1)
        self.assertEqual(root_cause["demoted_event_count"], 0)

    def test_neighbor_rows_include_same_action_and_same_type_pair(self):
        matrix = build_group_matrix(
            validation_rows=[
                {
                    "target_case": "event_semantic_target",
                    "action_name": "EVENT_CONNECT",
                    "src_type_name": "process",
                    "dst_type_name": "file",
                    "validation_count": "20",
                    "test_count": "3",
                    "final_threshold_source": "target_case_quantile",
                },
                {
                    "target_case": "event_semantic_target",
                    "action_name": "EVENT_WRITE",
                    "src_type_name": "process",
                    "dst_type_name": "netflow",
                    "validation_count": "0",
                    "test_count": "5",
                    "final_threshold_source": "global_extreme",
                },
            ],
            score_rows=[],
            policy_rows=[],
        )

        rows = build_neighbor_rows(matrix)
        relation_by_action = {row["action"]: row["neighbor_relation"] for row in rows}

        self.assertEqual(relation_by_action["EVENT_CONNECT"], "same_action")
        self.assertEqual(relation_by_action["EVENT_WRITE"], "same_type_pair")

    def test_run_audit_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "result"
            output_dir = root / "diag"
            result_dir.mkdir()
            _write_csv(
                result_dir / "conditional_score_summary_by_target_action_type.csv",
                [
                    "target_case",
                    "action_name",
                    "src_type_name",
                    "dst_type_name",
                    "validation_count",
                    "test_count",
                    "final_threshold_source",
                ],
                [
                    {
                        "target_case": "event_semantic_target",
                        "action_name": "EVENT_CONNECT",
                        "src_type_name": "process",
                        "dst_type_name": "netflow",
                        "validation_count": "0",
                        "test_count": "1",
                        "final_threshold_source": "global_extreme",
                    }
                ],
            )
            _write_csv(
                result_dir / "online_event_score_trace.csv",
                ["target_case", "action", "src_type", "dst_type", "score", "threshold"],
                [
                    {
                        "target_case": "event_semantic_target",
                        "action": "EVENT_CONNECT",
                        "src_type": "process",
                        "dst_type": "netflow",
                        "score": "0.05",
                        "threshold": "0.12",
                    }
                ],
            )
            _write_csv(
                result_dir / "group_alert_policy_summary.csv",
                ["action", "src_type", "dst_type", "alert_count", "demoted_event_count"],
                [
                    {
                        "action": "EVENT_CONNECT",
                        "src_type": "process",
                        "dst_type": "netflow",
                        "alert_count": "0",
                        "demoted_event_count": "0",
                    }
                ],
            )

            summary = run_audit(result_dir=result_dir, output_dir=output_dir)

            self.assertEqual(
                summary["connect_root_cause"]["root_cause"],
                "validation_group_absent_test_group_present",
            )
            self.assertTrue((output_dir / "connect_unseen_group_matrix.csv").is_file())
            self.assertTrue(
                (output_dir / "connect_unseen_action_direction_neighbors.csv").is_file(),
            )
            self.assertTrue(
                (output_dir / "cadets_e3_v3_connect_unseen_root_cause.json").is_file(),
            )
            self.assertTrue(
                (output_dir / "cadets_e3_v3_connect_unseen_root_cause_report.md").is_file(),
            )


if __name__ == "__main__":
    unittest.main()
