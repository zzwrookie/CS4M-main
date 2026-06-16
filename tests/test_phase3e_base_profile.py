"""Tests for Phase3E base profiling helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.pipeline.io.event_artifacts import _phase3e_write_base_profile_outputs


class Phase3EBaseProfileTests(unittest.TestCase):
    """Validate profile CSV and summary output formatting."""

    def test_profile_writer_outputs_csv_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            rows = [
                {
                    "dataset": "OPTC_051",
                    "out_tag": "TAG",
                    "event_count": 100,
                    "events_per_second": 50.0,
                    "target_lookup_seconds": 1.0,
                    "make_context_seconds": 2.0,
                    "update_states_seconds": 3.0,
                    "batch_stack_seconds": 4.0,
                    "train_batch_step_seconds": 5.0,
                    "rss_mb": 123.0,
                    "active_state_nodes": 10,
                },
            ]

            paths = _phase3e_write_base_profile_outputs(output_dir, rows)

            csv_text = paths["csv"].read_text(encoding="utf-8")
            self.assertIn("target_lookup_seconds", csv_text)
            self.assertIn("OPTC_051", csv_text)
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary["interval_count"], 1)
            self.assertEqual(summary["total_events"], 100)
            self.assertEqual(summary["total_train_batch_step_seconds"], 5.0)


if __name__ == "__main__":
    unittest.main()
