import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.tools import diagnose_clearscope_e5_missed_gt_scores as diagnosis


EVENT_DTYPE = np.dtype(
    [
        ("event_id", "<i8"),
        ("src_node_idx", "<i4"),
        ("dst_node_idx", "<i4"),
        ("action_id", "<i2"),
        ("src_type_id", "i1"),
        ("dst_type_id", "i1"),
    ],
)


class ClearScopeE5MissedGtScoreDiagnosisTests(unittest.TestCase):
    def test_diagnoses_hit_missed_and_absent_gt_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_path = root / "gt.csv"
            gt_path.write_text(
                "uuid-hit,{},100\n"
                "uuid-low,{},200\n"
                "uuid-sem,{},300\n"
                "uuid-absent,{},400\n",
                encoding="utf-8",
            )
            node_map = {100: 10, 200: 20, 300: 30, 400: 40, 500: 50}
            node_map_path = root / "node_id_to_idx.pkl"
            node_map_path.write_bytes(pickle.dumps(node_map))
            alerts_path = root / "online_event_alerts.csv"
            with alerts_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["src_idx", "dst_idx", "event_score"])
                writer.writeheader()
                writer.writerow({"src_idx": "10", "dst_idx": "50", "event_score": "0.8"})
            score_trace_path = root / "online_event_score_trace.csv"
            with score_trace_path.open("w", newline="", encoding="utf-8") as handle:
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
                        "event_id": "101",
                        "src_idx": "10",
                        "dst_idx": "50",
                        "score": "0.8",
                        "threshold": "0.3",
                        "action": "EVENT_EXECUTE",
                        "src_type": "process",
                        "dst_type": "file",
                        "target_case": "both_cold_action_target",
                    },
                )
                writer.writerow(
                    {
                        "stream_pos": "2",
                        "event_id": "102",
                        "src_idx": "20",
                        "dst_idx": "50",
                        "score": "0.2",
                        "threshold": "0.3",
                        "action": "EVENT_READ",
                        "src_type": "process",
                        "dst_type": "file",
                        "target_case": "both_cold_action_target",
                    },
                )
                writer.writerow(
                    {
                        "stream_pos": "3",
                        "event_id": "103",
                        "src_idx": "30",
                        "dst_idx": "50",
                        "score": "0.05",
                        "threshold": "0.3",
                        "action": "EVENT_READ",
                        "src_type": "process",
                        "dst_type": "file",
                        "target_case": "event_semantic_target",
                    },
                )
            event_index_path = root / "event_index_test.memmap"
            event_index = np.memmap(event_index_path, dtype=EVENT_DTYPE, mode="w+", shape=(3,))
            event_index[0] = (101, 10, 50, 2, 0, 1)
            event_index[1] = (102, 20, 50, 4, 0, 1)
            event_index[2] = (103, 30, 50, 4, 0, 1)
            event_index.flush()

            result = diagnosis.run_diagnosis(
                result_dir=root,
                gt_paths=[gt_path],
                node_id_to_idx_path=node_map_path,
                event_index_path=event_index_path,
            )

            summary = result["summary"]
            self.assertEqual(summary["gt_db_node_count"], 4)
            self.assertEqual(summary["gt_nodes_seen_in_test_count"], 3)
            self.assertEqual(summary["gt_nodes_with_score_rows"], 3)
            self.assertEqual(summary["alert_covered_gt_count"], 1)
            self.assertEqual(summary["missed_gt_count"], 3)
            self.assertEqual(summary["absent_from_test_gt_count"], 1)
            self.assertEqual(summary["missed_below_threshold_count"], 2)
            self.assertEqual(summary["missed_event_semantic_node_count"], 1)
            self.assertIn(
                summary["recommendation_bucket"],
                {
                    "score_below_threshold_audit",
                    "event_semantic_score_collapse_audit",
                    "both_cold_support_gate_audit",
                },
            )

            for name in (
                "e5_missed_gt_score_diagnosis.json",
                "e5_missed_gt_score_rows.csv",
                "e5_gt_score_summary_by_node.csv",
                "e5_gt_score_summary_by_target_case.csv",
            ):
                self.assertTrue((root / name).is_file(), name)
            persisted = json.loads((root / "e5_missed_gt_score_diagnosis.json").read_text())
            self.assertEqual(persisted["missed_gt_count"], 3)


if __name__ == "__main__":
    unittest.main()
