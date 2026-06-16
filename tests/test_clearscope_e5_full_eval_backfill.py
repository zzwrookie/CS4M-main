import csv
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.tools import backfill_clearscope_e5_full_eval as backfill


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


class ClearScopeE5FullEvalBackfillTests(unittest.TestCase):
    def test_computes_backfill_metrics_from_synthetic_alerts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_path = root / "gt.csv"
            gt_path.write_text(
                "uuid-a,{'subject':'a'},100\n"
                "uuid-b,{'file':'b'},200\n"
                "uuid-c,{'file':'c'},300\n",
                encoding="utf-8",
            )
            node_map_path = root / "node_id_to_idx.pkl"
            node_map_path.write_bytes(pickle.dumps({100: 10, 200: 20, 300: 30, 400: 40}))
            alerts_path = root / "online_event_alerts.csv"
            with alerts_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "stream_pos",
                        "event_index",
                        "src_idx",
                        "dst_idx",
                        "action",
                        "src_type",
                        "dst_type",
                        "target_case",
                        "event_score",
                        "threshold",
                        "threshold_basis",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "stream_pos": "1",
                        "event_index": "101",
                        "src_idx": "10",
                        "dst_idx": "40",
                        "action": "EVENT_EXECUTE",
                        "src_type": "process",
                        "dst_type": "file",
                        "target_case": "both_cold_action_target",
                        "event_score": "0.5",
                        "threshold": "0.2",
                        "threshold_basis": "validation",
                    },
                )
                writer.writerow(
                    {
                        "stream_pos": "2",
                        "event_index": "102",
                        "src_idx": "40",
                        "dst_idx": "40",
                        "action": "EVENT_OPEN",
                        "src_type": "process",
                        "dst_type": "process",
                        "target_case": "event_semantic_target",
                        "event_score": "0.3",
                        "threshold": "0.2",
                        "threshold_basis": "validation",
                    },
                )
            event_index_path = root / "event_index_test.memmap"
            event_index = np.memmap(event_index_path, dtype=EVENT_DTYPE, mode="w+", shape=(2,))
            event_index[0] = (101, 10, 40, 2, 1, 2)
            event_index[1] = (103, 20, 30, 3, 1, 2)
            event_index.flush()

            result = backfill.run_backfill(
                result_dir=root,
                gt_paths=[gt_path],
                node_id_to_idx_path=node_map_path,
                event_index_path=event_index_path,
                original_eval_path=None,
            )

            metrics = result["metrics"]
            self.assertEqual(metrics["gt_db_node_count"], 3)
            self.assertEqual(metrics["gt_mapped_node_count"], 3)
            self.assertEqual(metrics["alert_event_count"], 2)
            self.assertEqual(metrics["alert_node_count"], 2)
            self.assertEqual(metrics["event_tp_any_endpoint"], 1)
            self.assertEqual(metrics["event_fp"], 1)
            self.assertEqual(metrics["strict_node_tp"], 1)
            self.assertEqual(metrics["strict_node_fp"], 1)
            self.assertEqual(metrics["missed_gt_node_count"], 2)
            self.assertEqual(metrics["gt_nodes_seen_in_test_count"], 3)
            self.assertEqual(
                metrics["per_action_type_target_case"][0]["key"],
                "EVENT_EXECUTE|process|file|both_cold_action_target",
            )

            for name in (
                "e5_gt_backfill_metrics.json",
                "e5_gt_backfill_event_alerts.csv",
                "e5_gt_backfill_node_alerts.csv",
                "e5_gt_diagnostic_report.json",
                "e5_gt_missed_nodes.csv",
                "e5_gt_hit_alerts.csv",
            ):
                self.assertTrue((root / name).is_file(), name)

            persisted = json.loads((root / "e5_gt_backfill_metrics.json").read_text())
            self.assertEqual(persisted["event_tp_any_endpoint"], 1)


if __name__ == "__main__":
    unittest.main()
