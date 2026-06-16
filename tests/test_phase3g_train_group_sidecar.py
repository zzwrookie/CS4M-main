"""Tests for Phase3G train group calibration sidecars."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cs4m.phase3g.conditional_head import (
    BOTH_COLD_ACTION_TARGET_ID,
    EVENT_SEMANTIC_TARGET_ID,
)
from scripts.pipeline.conditional.train import (
    _phase3g_train_group_summary_rows,
    _write_phase3g_train_group_score_summary,
)
from scripts.pipeline.io.conditional_cache import _phase3g_conditional_memmap_paths


class _Config:
    action_validation_cache_dir = ""
    dataset = "CADETS_E3"
    sspm_state_model = "s4d_complex_node"
    sspm_update_gate_mode = "none"


class Phase3GTrainGroupSidecarTests(unittest.TestCase):
    """Validate train-derived group max sidecar generation."""

    def test_train_group_summary_rows_aggregate_exact_group_scores(self) -> None:
        rows = _phase3g_train_group_summary_rows(
            scores=np.asarray([0.1, 0.4, 0.2, 0.9], dtype=np.float32),
            target_case_ids=np.asarray(
                [
                    EVENT_SEMANTIC_TARGET_ID,
                    EVENT_SEMANTIC_TARGET_ID,
                    EVENT_SEMANTIC_TARGET_ID,
                    BOTH_COLD_ACTION_TARGET_ID,
                ],
                dtype=np.int8,
            ),
            action_ids=np.asarray([1, 1, 2, 1], dtype=np.int16),
            src_type_ids=np.asarray([1, 1, 1, 1], dtype=np.int16),
            dst_type_ids=np.asarray([0, 0, 0, 0], dtype=np.int16),
        )

        by_key = {
            (
                row["target_case"],
                row["action_id"],
                row["src_type_id"],
                row["dst_type_id"],
            ): row
            for row in rows
        }
        exact = by_key[("event_semantic_target", 1, 1, 0)]
        self.assertEqual(exact["train_count"], 2)
        self.assertAlmostEqual(exact["train_max_score"], 0.4, places=6)
        self.assertGreaterEqual(exact["train_p999"], 0.1)
        self.assertLessEqual(exact["train_p999"], 0.4)
        self.assertEqual(exact["score_source"], "train_conditional_head_score")

    def test_write_train_group_score_summary_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "train_group_score_summary.csv"
            path = _write_phase3g_train_group_score_summary(
                output,
                scores=np.asarray([0.1, 0.4], dtype=np.float32),
                target_case_ids=np.asarray([EVENT_SEMANTIC_TARGET_ID] * 2, dtype=np.int8),
                action_ids=np.asarray([1, 1], dtype=np.int16),
                src_type_ids=np.asarray([1, 1], dtype=np.int16),
                dst_type_ids=np.asarray([0, 0], dtype=np.int16),
            )

            text = path.read_text(encoding="utf-8")
        self.assertIn("target_case,action,src_type,dst_type", text)
        self.assertIn("train_count,train_max_score", text)
        self.assertIn("event_semantic_target", text)

    def test_conditional_memmap_paths_include_train_group_id_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = _Config()
            cfg.action_validation_cache_dir = temp_dir

            paths = _phase3g_conditional_memmap_paths(cfg)

        self.assertEqual(
            paths["action_id"].name,
            "action_id_s4d_complex_node.memmap",
        )
        self.assertEqual(
            paths["src_type_id"].name,
            "src_type_id_s4d_complex_node.memmap",
        )
        self.assertEqual(
            paths["dst_type_id"].name,
            "dst_type_id_s4d_complex_node.memmap",
        )


if __name__ == "__main__":
    unittest.main()
