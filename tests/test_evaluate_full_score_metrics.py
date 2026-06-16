import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.tools.evaluate_full_score_metrics import evaluate_result_dir


class EvaluateFullScoreMetricsTest(unittest.TestCase):
    def test_evaluate_result_dir_requires_full_score_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = evaluate_result_dir(Path(tmp), event_labels={}, node_labels={})

        self.assertEqual(result["status"], "missing_input")
        self.assertEqual(result["missing_input"], "online_event_score_trace.csv")

    def test_cli_runs_when_invoked_by_script_path(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "tools" / "evaluate_full_score_metrics.py"
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [sys.executable, str(script), "--result-dir", tmp],
                cwd=repo_root,
                check=False,
                text=True,
                capture_output=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("missing_input", completed.stdout)

    def test_evaluate_result_dir_writes_eval_metrics_from_full_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "online_event_score_trace.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "event_id",
                        "src_idx",
                        "dst_idx",
                        "info_src",
                        "info_dst",
                        "score",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "event_id": "e1",
                        "src_idx": "1",
                        "dst_idx": "2",
                        "info_src": "1",
                        "info_dst": "2",
                        "score": "0.9",
                    }
                )
                writer.writerow(
                    {
                        "event_id": "e2",
                        "src_idx": "3",
                        "dst_idx": "4",
                        "info_src": "3",
                        "info_dst": "4",
                        "score": "0.1",
                    }
                )

            result = evaluate_result_dir(
                root,
                event_labels={"e1": 1, "e2": 0},
                node_labels={"1": 1, "2": 1, "3": 0, "4": 0},
            )

            payload = json.loads((root / "eval_metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "ok")
        self.assertEqual(payload["event_confusion"]["tp"], 1)
        self.assertEqual(payload["event_confusion"]["tn"], 1)
        self.assertEqual(payload["node_confusion_strict"]["tp"], 2)
        self.assertEqual(payload["node_confusion_strict"]["tn"], 2)
        self.assertEqual(payload["event_auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
