import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools.audit_cadets_e3_v3_connect_speed_cache import run_audit


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class CadetsE3V3ConnectSpeedCacheAuditTests(unittest.TestCase):
    def test_audit_reports_connect_below_threshold_and_validation_bottleneck(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "result"
            output_dir = root / "diag"
            result_dir.mkdir()
            score_fields = [
                "stream_pos",
                "event_id",
                "score",
                "threshold",
                "action",
                "src_type",
                "dst_type",
                "target_case",
            ]
            _write_csv(
                result_dir / "online_event_score_trace.csv",
                score_fields,
                [
                    {
                        "stream_pos": 1,
                        "event_id": 10,
                        "score": 0.10,
                        "threshold": 0.20,
                        "action": "EVENT_CONNECT",
                        "src_type": "process",
                        "dst_type": "netflow",
                        "target_case": "event_semantic_target",
                    },
                    {
                        "stream_pos": 2,
                        "event_id": 11,
                        "score": 0.30,
                        "threshold": 0.20,
                        "action": "EVENT_WRITE",
                        "src_type": "process",
                        "dst_type": "netflow",
                        "target_case": "event_semantic_target",
                    },
                ],
            )
            _write_csv(
                result_dir / "group_alert_policy_summary.csv",
                [
                    "policy_name",
                    "action",
                    "src_type",
                    "dst_type",
                    "event_count",
                    "alert_count",
                    "node_evidence_count",
                    "demoted_event_count",
                ],
                [
                    {
                        "policy_name": "cadets_e4_v3_policy_smoke_v1",
                        "action": "EVENT_CONNECT",
                        "src_type": "process",
                        "dst_type": "netflow",
                        "event_count": 1,
                        "alert_count": 0,
                        "node_evidence_count": 0,
                        "demoted_event_count": 0,
                    }
                ],
            )
            (result_dir / "eval_causal_semantics_slim.json").write_text(
                json.dumps(
                    {
                        "timing": {
                            "validation_seconds": 90.0,
                            "test_scoring_seconds": 30.0,
                            "label_attach_seconds": 5.0,
                        },
                        "runtime": {
                            "events_scored": 1000,
                            "throughput_events_per_second": 8.0,
                        },
                        "memory": {"online_minimal_events_per_sec": 33.0},
                        "config": {
                            "action_type_alert_policy": "cadets_e4_v3_policy_smoke_v1",
                            "node_embedding_lookup_mode": "compact_used_nodes",
                            "rss_profile_mode": "online_minimal",
                        },
                        "db_stream_mode_actual": "python_lookup",
                    }
                ),
                encoding="utf-8",
            )

            summary = run_audit(result_dir=result_dir, output_dir=output_dir)

            self.assertEqual(summary["connect_process_netflow"]["event_count"], 1)
            self.assertEqual(summary["connect_process_netflow"]["above_threshold_count"], 0)
            self.assertEqual(
                summary["connect_process_netflow"]["conclusion"],
                "scores_below_validation_threshold_not_policy_demoted",
            )
            self.assertEqual(summary["speed"]["dominant_cost"], "validation_cache_build")
            self.assertTrue((output_dir / "connect_group_score_audit.csv").is_file())
            self.assertTrue((output_dir / "speed_cache_audit_summary.csv").is_file())
            self.assertTrue(
                (output_dir / "cadets_e3_v3_connect_speed_cache_audit_report.md").is_file(),
            )


if __name__ == "__main__":
    unittest.main()
