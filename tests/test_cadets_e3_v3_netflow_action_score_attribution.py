import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools.audit_cadets_e3_v3_netflow_action_score_attribution import (
    build_attribution_rows,
    run_audit,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class CadetsE3V3NetflowActionScoreAttributionTests(unittest.TestCase):
    def test_build_attribution_rows_join_scores_validation_and_policy(self):
        score_rows = [
            {
                "score": "0.05",
                "threshold": "0.12",
                "action": "EVENT_CONNECT",
                "src_type": "process",
                "dst_type": "netflow",
                "target_case": "event_semantic_target",
            },
            {
                "score": "0.08",
                "threshold": "0.12",
                "action": "EVENT_CONNECT",
                "src_type": "process",
                "dst_type": "netflow",
                "target_case": "event_semantic_target",
            },
            {
                "score": "0.20",
                "threshold": "0.12",
                "action": "EVENT_WRITE",
                "src_type": "process",
                "dst_type": "netflow",
                "target_case": "event_semantic_target",
            },
        ]
        validation_rows = [
            {
                "target_case": "event_semantic_target",
                "action_name": "EVENT_CONNECT",
                "src_type_name": "process",
                "dst_type_name": "netflow",
                "validation_count": "847",
                "validation_count_bucket": "supported",
                "threshold_level": "target_case",
                "final_threshold_source": "target_case_quantile",
                "val_p9995": "0.124",
                "test_max": "0.08",
            }
        ]
        policy_rows = [
            {
                "action": "EVENT_CONNECT",
                "src_type": "process",
                "dst_type": "netflow",
                "event_count": "2",
                "alert_count": "0",
                "demoted_event_count": "0",
                "TP": "0",
                "FP": "0",
            },
            {
                "action": "EVENT_WRITE",
                "src_type": "process",
                "dst_type": "netflow",
                "event_count": "1",
                "alert_count": "1",
                "demoted_event_count": "0",
                "TP": "0",
                "FP": "1",
            },
        ]

        rows = build_attribution_rows(
            score_rows=score_rows,
            validation_rows=validation_rows,
            policy_rows=policy_rows,
            sweep_rows=[],
        )

        connect = next(row for row in rows if row["action"] == "EVENT_CONNECT")
        write = next(row for row in rows if row["action"] == "EVENT_WRITE")
        self.assertEqual(connect["event_count"], 2)
        self.assertEqual(connect["above_threshold_count"], 0)
        self.assertEqual(connect["validation_count"], 847)
        self.assertEqual(connect["threshold_source"], "target_case_quantile")
        self.assertEqual(connect["interpretation"], "below_threshold_with_validation_support")
        self.assertEqual(write["above_threshold_count"], 1)
        self.assertEqual(write["final_alert_count"], 1)

    def test_run_audit_writes_outputs_and_connect_conclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "result"
            output_dir = root / "diag"
            result_dir.mkdir()
            _write_csv(
                result_dir / "online_event_score_trace.csv",
                ["score", "threshold", "action", "src_type", "dst_type", "target_case"],
                [
                    {
                        "score": "0.05",
                        "threshold": "0.12",
                        "action": "EVENT_CONNECT",
                        "src_type": "process",
                        "dst_type": "netflow",
                        "target_case": "event_semantic_target",
                    }
                ],
            )
            _write_csv(
                result_dir / "conditional_score_summary_by_target_action_type.csv",
                [
                    "target_case",
                    "action_name",
                    "src_type_name",
                    "dst_type_name",
                    "validation_count",
                    "validation_count_bucket",
                    "threshold_level",
                    "final_threshold_source",
                    "val_p9995",
                    "test_max",
                ],
                [
                    {
                        "target_case": "event_semantic_target",
                        "action_name": "EVENT_CONNECT",
                        "src_type_name": "process",
                        "dst_type_name": "netflow",
                        "validation_count": "1",
                        "validation_count_bucket": "supported",
                        "threshold_level": "target_case",
                        "final_threshold_source": "target_case_quantile",
                        "val_p9995": "0.12",
                        "test_max": "0.05",
                    }
                ],
            )
            _write_csv(
                result_dir / "group_alert_policy_summary.csv",
                [
                    "action",
                    "src_type",
                    "dst_type",
                    "event_count",
                    "alert_count",
                    "demoted_event_count",
                    "TP",
                    "FP",
                    "covered_malicious_nodes",
                ],
                [
                    {
                        "action": "EVENT_CONNECT",
                        "src_type": "process",
                        "dst_type": "netflow",
                        "event_count": "1",
                        "alert_count": "0",
                        "demoted_event_count": "0",
                        "TP": "0",
                        "FP": "0",
                        "covered_malicious_nodes": "0",
                    }
                ],
            )
            _write_csv(
                result_dir / "group_threshold_sweep_summary.csv",
                [
                    "action",
                    "src_type",
                    "dst_type",
                    "threshold",
                    "margin",
                    "alert_count",
                    "TP",
                    "FP",
                    "covered_malicious_nodes",
                ],
                [],
            )
            (result_dir / "eval_causal_semantics_slim.json").write_text(
                json.dumps({"leakage_check": {"ground_truth_used_only_for_evaluation": True}}),
                encoding="utf-8",
            )

            summary = run_audit(result_dir=result_dir, output_dir=output_dir)

            self.assertEqual(
                summary["connect_process_netflow"]["interpretation"],
                "below_threshold_with_validation_support",
            )
            self.assertTrue((output_dir / "netflow_action_score_attribution.csv").is_file())
            self.assertTrue((output_dir / "netflow_action_threshold_source.csv").is_file())
            self.assertTrue((output_dir / "netflow_action_sweep_sensitivity.csv").is_file())
            self.assertTrue(
                (
                    output_dir
                    / "cadets_e3_v3_netflow_action_score_attribution_report.md"
                ).is_file(),
            )


if __name__ == "__main__":
    unittest.main()
