import csv
import tempfile
import unittest
from pathlib import Path

from scripts.tools.replay_cadets_e3_unified_group_policy import (
    ReplayParams,
    ThresholdInputs,
    decide_cadets_policy,
    replay_cadets_result,
    resolve_threshold,
)


class CadetsUnifiedGroupPolicyReplayTest(unittest.TestCase):
    def test_resolver_prefers_validation_group_threshold(self):
        params = ReplayParams()
        inputs = ThresholdInputs(
            validation_groups={
                "event_semantic_target|EVENT_READ|file->process": {
                    "threshold": 0.12,
                    "validation_count": 150,
                }
            },
            train_groups={
                "event_semantic_target|EVENT_READ|file->process": {
                    "train_count": 30,
                    "train_max_score": 0.09,
                }
            },
            validation_global_threshold=0.2,
            train_fallback_available=True,
        )

        decision = resolve_threshold(
            "event_semantic_target|EVENT_READ|file->process",
            inputs,
            params,
        )

        self.assertEqual(decision.threshold, 0.12)
        self.assertEqual(decision.threshold_source, "validation_group_quantile")
        self.assertEqual(decision.validation_count, 150)

    def test_resolver_uses_train_max_with_margin_when_validation_missing(self):
        params = ReplayParams()
        inputs = ThresholdInputs(
            validation_groups={},
            train_groups={
                "event_semantic_target|EVENT_CONNECT|process->netflow": {
                    "train_count": 30,
                    "train_max_score": 0.10,
                }
            },
            validation_global_threshold=0.2,
            train_fallback_available=True,
        )

        decision = resolve_threshold(
            "event_semantic_target|EVENT_CONNECT|process->netflow",
            inputs,
            params,
        )

        self.assertEqual(decision.threshold_source, "train_group_max")
        self.assertAlmostEqual(decision.threshold, 0.105)
        self.assertEqual(decision.train_count, 30)

    def test_resolver_uses_global_when_train_summary_missing(self):
        params = ReplayParams()
        inputs = ThresholdInputs(
            validation_groups={},
            train_groups={},
            validation_global_threshold=0.2,
            train_fallback_available=False,
        )

        decision = resolve_threshold(
            "event_semantic_target|EVENT_CONNECT|process->netflow",
            inputs,
            params,
        )

        self.assertEqual(decision.threshold_source, "validation_global_quantile")
        self.assertEqual(decision.threshold, 0.2)
        self.assertEqual(decision.train_count, 0)

    def test_policy_demotes_cold_unseen_without_evidence(self):
        decision = decide_cadets_policy(
            action="EVENT_CONNECT",
            src_type="process",
            dst_type="netflow",
            score_margin=0.01,
            threshold_source="validation_global_quantile",
            validation_count=0,
            train_count=0,
            node_evidence_count=0,
        )

        self.assertFalse(decision.final_alert)
        self.assertEqual(decision.policy_reason, "cold_unseen_demoted")

    def test_policy_keeps_sendto_netflow(self):
        decision = decide_cadets_policy(
            action="EVENT_SENDTO",
            src_type="process",
            dst_type="netflow",
            score_margin=0.0,
            threshold_source="validation_global_quantile",
            validation_count=0,
            train_count=0,
            node_evidence_count=0,
        )

        self.assertTrue(decision.final_alert)
        self.assertEqual(decision.policy_reason, "high_retain_group")

    def test_replay_writes_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "cadets"
            result_dir.mkdir()
            with (result_dir / "online_event_score_trace.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "stream_pos",
                        "event_id",
                        "src_idx",
                        "dst_idx",
                        "score",
                        "threshold",
                        "action",
                        "src_type",
                        "dst_type",
                        "target_case",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "stream_pos": "1",
                        "event_id": "10",
                        "src_idx": "100",
                        "dst_idx": "200",
                        "score": "0.3",
                        "threshold": "0.2",
                        "action": "EVENT_SENDTO",
                        "src_type": "process",
                        "dst_type": "netflow",
                        "target_case": "event_semantic_target",
                    }
                )
            with (result_dir / "conditional_score_summary_by_target_action_type.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "target_case",
                        "action_name",
                        "src_type_name",
                        "dst_type_name",
                        "validation_count",
                        "threshold",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "target_case": "event_semantic_target",
                        "action_name": "EVENT_SENDTO",
                        "src_type_name": "process",
                        "dst_type_name": "netflow",
                        "validation_count": "150",
                        "threshold": "0.2",
                    }
                )

            output_dir = root / "out"
            summary = replay_cadets_result(result_dir, output_dir)

            self.assertEqual(summary["raw_alert_rows"], 1)
            self.assertEqual(summary["final_alert_rows"], 1)
            self.assertTrue((output_dir / "raw_alerts.csv").exists())
            self.assertTrue((output_dir / "final_alerts.csv").exists())
            self.assertTrue((output_dir / "demoted_events.csv").exists())

    def test_replay_uses_score_summary_global_threshold_when_groups_are_unseen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "cadets"
            result_dir.mkdir()
            (result_dir / "score_summary.json").write_text(
                '{"validation": {"final_threshold": 0.2}}',
                encoding="utf-8",
            )
            with (result_dir / "online_event_score_trace.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "stream_pos",
                        "event_id",
                        "src_idx",
                        "dst_idx",
                        "score",
                        "threshold",
                        "action",
                        "src_type",
                        "dst_type",
                        "target_case",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "stream_pos": "1",
                        "event_id": "10",
                        "src_idx": "100",
                        "dst_idx": "200",
                        "score": "0.1",
                        "threshold": "0.0",
                        "action": "EVENT_CONNECT",
                        "src_type": "process",
                        "dst_type": "netflow",
                        "target_case": "event_semantic_target",
                    }
                )
            with (result_dir / "conditional_score_summary_by_target_action_type.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "target_case",
                        "action_name",
                        "src_type_name",
                        "dst_type_name",
                        "validation_count",
                        "threshold",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "target_case": "event_semantic_target",
                        "action_name": "EVENT_CONNECT",
                        "src_type_name": "process",
                        "dst_type_name": "netflow",
                        "validation_count": "0",
                        "threshold": "0.0",
                    }
                )

            summary = replay_cadets_result(result_dir, root / "out")

            self.assertEqual(summary["raw_alert_rows"], 0)


if __name__ == "__main__":
    unittest.main()
