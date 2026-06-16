import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools.build_cs4m_paper_evidence_tables import (
    build_evidence_lock,
    extract_optc_canonical_summary,
    format_pair,
    metric_status_for_clearscope_e5,
)


class CS4MPaperEvidenceLockTests(unittest.TestCase):
    def test_format_pair_marks_missing_values(self) -> None:
        self.assertEqual(format_pair(None, None), "needs_recompute")
        self.assertEqual(format_pair(14, 377), "14 / 377")

    def test_extract_optc_canonical_summary_preserves_both_gt_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = Path(tmp_dir) / "summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "dataset": "OPTC_051",
                        "event_alerts_by_canonical_gt": {
                            "alerts": 1366,
                            "tp": 208,
                            "fp": 1158,
                        },
                        "canonical_aware": {
                            "original_gt_hit": 66,
                            "original_gt_total": 114,
                            "unique_compact_gt_hit": 19,
                            "unique_compact_gt_total": 67,
                            "by_kind": {
                                "netflow": {"hit": 49, "total": 49},
                                "process": {"hit": 15, "total": 59},
                                "file": {"hit": 2, "total": 6},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )

            row = extract_optc_canonical_summary(summary)

        self.assertEqual(row["original_gt_hit"], "66/114")
        self.assertEqual(row["unique_compact_gt_hit"], "19/67")
        self.assertEqual(row["netflow_gt_hit"], "49/49")
        self.assertEqual(row["process_gt_hit"], "15/59")
        self.assertEqual(row["file_gt_hit"], "2/6")
        self.assertEqual(row["event_tp_fp"], "208 / 1158")

    def test_metric_status_for_clearscope_e5_flags_conflict(self) -> None:
        status = metric_status_for_clearscope_e5(default_topk_tp=0, audit_node_tp=14)
        self.assertEqual(status, "needs_recompute_51_gt_source_conflict")

    def test_build_evidence_lock_writes_required_outputs_and_risks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            build_evidence_lock(out_dir)

            expected = {
                "evidence_lock_topk_table.csv",
                "evidence_lock_runtime_table.csv",
                "evidence_lock_optc051_summary.csv",
                "evidence_lock_metric_provenance.md",
                "evidence_lock_metrics.md",
            }
            self.assertEqual(expected, {path.name for path in out_dir.iterdir()})
            provenance = (out_dir / "evidence_lock_metric_provenance.md").read_text(
                encoding="utf-8"
            )

        self.assertIn("CLEARSCOPE_E5", provenance)
        self.assertIn("needs_recompute_51_gt_source_conflict", provenance)
        self.assertIn("GT is post-stream evaluation only", provenance)


if __name__ == "__main__":
    unittest.main()
