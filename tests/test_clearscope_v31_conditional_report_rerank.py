"""Tests for ClearScope v31 support rerank in conditional rebuilt reports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.pipeline.outputs.conditional_reports import (
    _phase3g_write_node_pool_rebuilt_outputs,
)


class ClearScopeV31ConditionalReportRerankTests(unittest.TestCase):
    """Validate rebuilt node pool support scoring used by Phase3G reports."""

    def test_rebuilt_node_pool_uses_v31_support_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            coverage_rows = [
                {"node_idx": 10, "max_event_score": 0.30, "alert_count": 1},
                {"node_idx": 20, "max_event_score": 0.20, "alert_count": 8},
            ]
            event_rows = [
                {
                    "event_index": 1,
                    "event_score": 0.30,
                    "residual_score": 0.30,
                    "info_src": 10,
                    "info_dst": 11,
                    "src_type": "process",
                    "dst_type": "process",
                    "action": "EVENT_READ",
                },
                {
                    "event_index": 2,
                    "event_score": 0.20,
                    "residual_score": 0.20,
                    "info_src": 30,
                    "info_dst": 20,
                    "src_type": "process",
                    "dst_type": "file",
                    "action": "EVENT_WRITE",
                },
                {
                    "event_index": 3,
                    "event_score": 0.19,
                    "residual_score": 0.19,
                    "info_src": 20,
                    "info_dst": 30,
                    "src_type": "file",
                    "dst_type": "process",
                    "action": "EVENT_READ",
                },
            ]

            summary = _phase3g_write_node_pool_rebuilt_outputs(
                output_dir=output_dir,
                coverage_rows=coverage_rows,
                idx_to_db_node_id={10: 110, 20: 120},
                abnormal_db_node_ids={120},
                topk_values=[1],
                node_pool_score_mode="base_conf_v31_support",
                event_rows=event_rows,
            )

            self.assertEqual(summary["node_pool_topk"]["top1"]["tp"], 1)
            first_row = (output_dir / "final_node_pool_alerts.csv").read_text().splitlines()[1]
            self.assertTrue(first_row.startswith("20,"))


if __name__ == "__main__":
    unittest.main()
