import csv
import tempfile
import unittest
from pathlib import Path

from scripts.tools.check_clearscope_e5_dual_channel_smoke_outputs import (
    SmokeCheckError,
    check_outputs,
)


ALERT_FIELDS = [
    "event_index",
    "target_case",
    "score",
    "threshold",
]

EXPLANATION_FIELDS = [
    "stream_pos",
    "event_index",
    "trigger_node_idx",
    "trigger_node_role",
    "event_score",
    "score_floor",
    "event_semantic_max_score_seen",
    "event_semantic_near_threshold_count",
    "event_semantic_support_event_count",
    "support_threshold",
    "source_target_case",
    "alert_channel",
    "threshold_source",
]


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class ClearScopeE5DualChannelSmokeCheckerTests(unittest.TestCase):
    def test_accepts_valid_dual_channel_explanation(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            _write_csv(
                result_dir / "online_event_alerts.csv",
                ALERT_FIELDS,
                [
                    {
                        "event_index": "10",
                        "target_case": "event_semantic_node_support",
                        "score": "0.13",
                        "threshold": "0.12",
                    },
                ],
            )
            _write_csv(
                result_dir / "online_event_alert_explanations.csv",
                EXPLANATION_FIELDS,
                [
                    {
                        "stream_pos": "5",
                        "event_index": "10",
                        "trigger_node_idx": "42",
                        "trigger_node_role": "src",
                        "event_score": "0.13",
                        "score_floor": "0.12",
                        "event_semantic_max_score_seen": "0.13",
                        "event_semantic_near_threshold_count": "2",
                        "event_semantic_support_event_count": "4",
                        "support_threshold": "2",
                        "source_target_case": "event_semantic_target",
                        "alert_channel": "event_semantic_node_support_ge_2",
                        "threshold_source": "validation_event_semantic_p999_half",
                    },
                ],
            )

            summary = check_outputs(
                result_dir,
                score_floor=0.12,
                support_threshold=2,
            )

            self.assertEqual(summary["alert_rows"], 1)
            self.assertEqual(summary["explanation_rows"], 1)

    def test_rejects_leakage_header_in_alerts(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            _write_csv(
                result_dir / "online_event_alerts.csv",
                ALERT_FIELDS + ["ground_truth_label"],
                [],
            )
            _write_csv(
                result_dir / "online_event_alert_explanations.csv",
                EXPLANATION_FIELDS,
                [],
            )

            with self.assertRaises(SmokeCheckError):
                check_outputs(result_dir, score_floor=0.12, support_threshold=2)


if __name__ == "__main__":
    unittest.main()
