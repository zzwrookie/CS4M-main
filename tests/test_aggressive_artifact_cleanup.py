import tempfile
import unittest
from pathlib import Path

from scripts.tools.plan_aggressive_artifact_cleanup import (
    CleanupPlan,
    build_cleanup_plan,
    validate_delete_paths,
)


class AggressiveArtifactCleanupTest(unittest.TestCase):
    def test_keeps_confirmed_replay_dir_and_deletes_other_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            keep_dir = root / "outputs" / "results" / "tflr_light" / "keep"
            old_dir = root / "outputs" / "results" / "tflr_light" / "old"
            keep_dir.mkdir(parents=True)
            old_dir.mkdir(parents=True)
            (keep_dir / "online_event_score_trace.csv").write_text("x\n")
            (old_dir / "online_event_score_trace.csv").write_text("x\n")

            plan = build_cleanup_plan(
                repo_root=root,
                keep_paths=[keep_dir],
                output_dir=root / "outputs" / "diagnostics" / "cleanup",
            )

            self.assertIn(str(keep_dir.resolve()), plan.keep_paths)
            self.assertIn(str(old_dir.resolve()), plan.delete_paths)
            self.assertNotIn(str(keep_dir.resolve()), plan.delete_paths)

    def test_validate_refuses_paths_outside_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = CleanupPlan(
                keep_paths=[],
                delete_paths=[str(root / "ground_truth")],
                total_delete_bytes=0,
            )

            with self.assertRaises(ValueError):
                validate_delete_paths(root, plan)


if __name__ == "__main__":
    unittest.main()
