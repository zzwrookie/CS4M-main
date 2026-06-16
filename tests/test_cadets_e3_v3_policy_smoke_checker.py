import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools.check_cadets_e3_v3_policy_smoke_outputs import (
    SmokeCheckError,
    check_outputs,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class CadetsE3V3PolicySmokeCheckerTests(unittest.TestCase):
    def test_accepts_valid_policy_smoke_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            _write_csv(
                result_dir / "online_event_alerts.csv",
                ["event_index", "action", "target_case", "event_score", "threshold"],
                [{"event_index": 1, "action": "EVENT_CONNECT", "target_case": "event_semantic_target"}],
            )
            group_fields = [
                "policy_name",
                "action",
                "src_type",
                "dst_type",
                "event_count",
                "alert_count",
                "node_evidence_count",
                "demoted_event_count",
            ]
            _write_csv(
                result_dir / "group_alert_policy_summary.csv",
                group_fields,
                [
                    {
                        "policy_name": "cadets_e4_v3_policy_smoke_v1",
                        "action": "EVENT_CONNECT",
                        "src_type": "process",
                        "dst_type": "netflow",
                        "event_count": 10,
                        "alert_count": 3,
                        "node_evidence_count": 0,
                        "demoted_event_count": 0,
                    },
                    {
                        "policy_name": "cadets_e4_v3_policy_smoke_v1",
                        "action": "EVENT_SENDTO",
                        "src_type": "process",
                        "dst_type": "file",
                        "event_count": 20,
                        "alert_count": 0,
                        "node_evidence_count": 8,
                        "demoted_event_count": 8,
                    },
                ],
            )
            _write_csv(
                result_dir / "demoted_group_summary.csv",
                ["action", "src_type", "dst_type", "demoted_event_count"],
                [{"action": "EVENT_SENDTO", "src_type": "process", "dst_type": "file", "demoted_event_count": 8}],
            )
            _write_csv(result_dir / "target_case_summary.csv", ["target_case", "alert_count"], [])
            (result_dir / "eval_causal_semantics_slim.json").write_text(
                json.dumps(
                    {
                        "ground_truth_used_only_for_evaluation": True,
                        "labels_attached_after_streaming": True,
                        "raw_outputs_are_label_free": True,
                        "runtime": {
                            "events_scored": 100000,
                            "elapsed_seconds": 40.0,
                            "throughput_events_per_second": 2500.0,
                        },
                        "timing": {
                            "validation_seconds": 5.0,
                            "test_scoring_seconds": 30.0,
                            "label_attach_seconds": 1.0,
                        },
                        "config": {
                            "action_type_alert_policy": "cadets_e4_v3_policy_smoke_v1",
                            "node_embedding_lookup_mode": "compact_used_nodes",
                            "sspm_infer_fast_path": False,
                        },
                        "db_stream_mode_actual": "python_lookup",
                        "ofsm_compression": {"logical_node_count": 1000},
                    }
                ),
                encoding="utf-8",
            )

            summary = check_outputs(result_dir)

        self.assertEqual(summary["policy_name"], "cadets_e4_v3_policy_smoke_v1")
        self.assertEqual(summary["connect_process_netflow_alert_count"], 3)
        self.assertEqual(summary["events_scored"], 100000)
        self.assertEqual(summary["throughput_events_per_second"], 2500.0)

    def test_rejects_leakage_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            _write_csv(
                result_dir / "online_event_alerts.csv",
                ["event_index", "ground_truth_label"],
                [],
            )
            for name in [
                "group_alert_policy_summary.csv",
                "demoted_group_summary.csv",
                "target_case_summary.csv",
            ]:
                _write_csv(result_dir / name, ["policy_name"], [])
            (result_dir / "eval_causal_semantics_slim.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(SmokeCheckError):
                check_outputs(result_dir)

    def test_rejects_demoted_connect_when_raw_connect_alerts_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            _write_csv(result_dir / "online_event_alerts.csv", ["event_index"], [])
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
                        "event_count": 10,
                        "alert_count": 0,
                        "node_evidence_count": 5,
                        "demoted_event_count": 5,
                    }
                ],
            )
            _write_csv(result_dir / "demoted_group_summary.csv", ["policy_name"], [])
            _write_csv(result_dir / "target_case_summary.csv", ["target_case"], [])
            (result_dir / "eval_causal_semantics_slim.json").write_text(
                json.dumps(
                    {
                        "ground_truth_used_only_for_evaluation": True,
                        "runtime": {"throughput_events_per_second": 1.0},
                        "config": {"action_type_alert_policy": "cadets_e4_v3_policy_smoke_v1"},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(SmokeCheckError):
                check_outputs(result_dir)


if __name__ == "__main__":
    unittest.main()
