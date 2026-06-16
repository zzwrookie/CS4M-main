"""Tests for copying train-derived sidecars into Phase3G inference outputs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.pipeline.conditional.infer import (
    _phase3g_copy_train_group_summary_from_head_metadata,
)


class Phase3GInferTrainSidecarCopyTests(unittest.TestCase):
    """Validate train group sidecar propagation from head metadata."""

    def test_copies_train_group_summary_from_top_level_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "train" / "train_group_score_summary.csv"
            output_dir = root / "infer"
            source.parent.mkdir(parents=True)
            output_dir.mkdir()
            source.write_text("target_case,train_count\nx,1\n", encoding="utf-8")

            copied = _phase3g_copy_train_group_summary_from_head_metadata(
                output_dir=output_dir,
                head_metadata={"train_group_score_summary_csv": str(source)},
            )

            copied_path = Path(copied)
            self.assertEqual(copied_path, output_dir / "train_group_score_summary.csv")
            self.assertEqual(copied_path.read_text(encoding="utf-8"), source.read_text())

    def test_uses_nested_train_stats_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "train_group_score_summary.csv"
            output_dir = root / "infer"
            output_dir.mkdir()
            source.write_text("target_case,train_count\nx,1\n", encoding="utf-8")

            copied = _phase3g_copy_train_group_summary_from_head_metadata(
                output_dir=output_dir,
                head_metadata={
                    "train_stats": {
                        "train_group_score_summary_csv": str(source),
                    },
                },
            )

            self.assertEqual(Path(copied).read_text(encoding="utf-8"), source.read_text())

    def test_missing_metadata_returns_empty_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = _phase3g_copy_train_group_summary_from_head_metadata(
                output_dir=Path(temp_dir),
                head_metadata={},
            )

        self.assertEqual(copied, "")


if __name__ == "__main__":
    unittest.main()
