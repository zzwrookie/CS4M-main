import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools.prepare_cadets_unified_sidecars import prepare_cadets_sidecars


class PrepareCadetsUnifiedSidecarsTest(unittest.TestCase):
    def test_prepare_sidecars_writes_validation_summary_and_missing_train_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_dir = Path(tmp)
            with (result_dir / "conditional_score_summary_by_target_action_type.csv").open(
                "w", newline="", encoding="utf-8"
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
                        "val_p9995",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "target_case": "event_semantic_target",
                        "action_name": "EVENT_READ",
                        "src_type_name": "file",
                        "dst_type_name": "process",
                        "validation_count": "150",
                        "threshold": "0.7",
                        "val_p9995": "0.7",
                    }
                )

            summary = prepare_cadets_sidecars(result_dir)

            validation_path = result_dir / "validation_group_threshold_summary.csv"
            missing_path = result_dir / "missing_unified_sidecars.json"
            with validation_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            missing = json.loads(missing_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["validation_group_rows"], 1)
        self.assertFalse(summary["train_group_score_summary_exists"])
        self.assertEqual(rows[0]["action"], "EVENT_READ")
        self.assertEqual(rows[0]["threshold_source"], "validation_group_quantile")
        self.assertIn("train_group_score_summary.csv", missing["missing_required_sidecars"])


if __name__ == "__main__":
    unittest.main()
