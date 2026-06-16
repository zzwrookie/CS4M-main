import tempfile
import unittest
from pathlib import Path

from scripts.tools.manage_artifact_lifecycle import (
    LifecycleProfile,
    build_lifecycle_cleanup_plan,
)


class ArtifactLifecycleManagerTest(unittest.TestCase):
    def test_train_to_validation_keeps_train_group_summary_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "outputs" / "run"
            output.mkdir(parents=True)
            (output / "model.pkl").write_text("model", encoding="utf-8")
            (output / "train_group_score_summary.csv").write_text("group", encoding="utf-8")
            (output / "temporary_train.memmap").write_text("tmp", encoding="utf-8")

            plan = build_lifecycle_cleanup_plan(
                output,
                LifecycleProfile.TRAIN_TO_VALIDATION,
            )

        self.assertIn("temporary_train.memmap", {path.name for path in plan.delete_paths})
        self.assertNotIn("model.pkl", {path.name for path in plan.delete_paths})
        self.assertNotIn(
            "train_group_score_summary.csv",
            {path.name for path in plan.delete_paths},
        )

    def test_test_light_profile_keeps_eval_and_alerts_but_deletes_score_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for name in [
                "online_event_alerts.csv",
                "eval_metrics.json",
                "online_event_score_trace.csv",
            ]:
                (output / name).write_text("x", encoding="utf-8")

            plan = build_lifecycle_cleanup_plan(output, LifecycleProfile.TEST_LIGHT)

        self.assertIn("online_event_score_trace.csv", {path.name for path in plan.delete_paths})
        self.assertNotIn("online_event_alerts.csv", {path.name for path in plan.delete_paths})
        self.assertNotIn("eval_metrics.json", {path.name for path in plan.delete_paths})


if __name__ == "__main__":
    unittest.main()
