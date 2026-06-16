import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.tools.replay_unified_threshold_policy import (
    GroupKey,
    ThresholdResolver,
    replay_result_dir,
)


class UnifiedThresholdResolverTest(unittest.TestCase):
    def test_uses_validation_group_threshold_first(self):
        key = GroupKey("event_semantic_target", "EVENT_READ", "file", "process")
        resolver = ThresholdResolver(
            validation_group_thresholds={
                key: {
                    "threshold": 0.9,
                    "threshold_quantile": 0.999,
                    "validation_count": 25,
                }
            },
            train_group_thresholds={
                key: {
                    "train_max_score": 0.7,
                    "train_count": 40,
                }
            },
            validation_global_threshold=0.95,
        )

        resolved = resolver.resolve(key)

        self.assertEqual(resolved["threshold_source"], "validation_group_quantile")
        self.assertEqual(resolved["threshold_level"], "group")
        self.assertEqual(resolved["threshold"], 0.9)
        self.assertEqual(resolved["validation_count"], 25)

    def test_uses_train_max_when_validation_group_is_unseen(self):
        key = GroupKey("event_semantic_target", "EVENT_CONNECT", "process", "netflow")
        resolver = ThresholdResolver(
            validation_group_thresholds={
                key: {
                    "threshold": 0.0,
                    "threshold_quantile": 0.999,
                    "validation_count": 0,
                }
            },
            train_group_thresholds={
                key: {
                    "train_max_score": 0.42,
                    "train_count": 12,
                }
            },
            validation_global_threshold=0.95,
        )

        resolved = resolver.resolve(key)

        self.assertEqual(resolved["threshold_source"], "train_group_max")
        self.assertEqual(resolved["threshold_level"], "group_train_fallback")
        self.assertEqual(resolved["threshold"], 0.42)
        self.assertEqual(resolved["train_count"], 12)

    def test_uses_validation_global_when_group_has_no_validation_or_train(self):
        key = GroupKey("event_semantic_target", "EVENT_EXECUTE", "process", "file")
        resolver = ThresholdResolver(
            validation_group_thresholds={},
            train_group_thresholds={},
            validation_global_threshold=0.95,
        )

        resolved = resolver.resolve(key)

        self.assertEqual(resolved["threshold_source"], "validation_global_quantile")
        self.assertEqual(resolved["threshold_level"], "global")
        self.assertEqual(resolved["threshold"], 0.95)
        self.assertEqual(resolved["validation_count"], 0)
        self.assertEqual(resolved["train_count"], 0)


class UnifiedThresholdReplayTest(unittest.TestCase):
    def test_cli_runs_when_invoked_by_script_path(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "tools" / "replay_unified_threshold_policy.py"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "RESULT"
            output_dir = root / "OUT"
            result_dir.mkdir()
            with (result_dir / "online_event_score_trace.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "target_case",
                        "action",
                        "src_type",
                        "dst_type",
                        "score",
                        "threshold",
                        "validation_count",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "target_case": "event_semantic_target",
                        "action": "EVENT_READ",
                        "src_type": "file",
                        "dst_type": "process",
                        "score": "0.9",
                        "threshold": "0.8",
                        "validation_count": "1",
                    }
                )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--result-dir",
                    str(result_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=repo_root,
                check=False,
                text=True,
                capture_output=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("raw_alert_rows", completed.stdout)

    def test_replay_counts_raw_alerts_and_threshold_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "RESULT"
            result_dir.mkdir()
            with (result_dir / "online_event_score_trace.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "target_case",
                        "action",
                        "src_type",
                        "dst_type",
                        "score",
                        "threshold",
                        "validation_count",
                        "threshold_quantile",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "target_case": "event_semantic_target",
                        "action": "EVENT_READ",
                        "src_type": "file",
                        "dst_type": "process",
                        "score": "0.91",
                        "threshold": "0.90",
                        "validation_count": "3",
                        "threshold_quantile": "0.999",
                    }
                )
                writer.writerow(
                    {
                        "target_case": "event_semantic_target",
                        "action": "EVENT_CONNECT",
                        "src_type": "process",
                        "dst_type": "netflow",
                        "score": "0.50",
                        "threshold": "0.95",
                        "validation_count": "0",
                        "threshold_quantile": "0.999",
                    }
                )
            with (result_dir / "train_group_score_summary.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "target_case",
                        "action",
                        "src_type",
                        "dst_type",
                        "train_count",
                        "train_max_score",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "target_case": "event_semantic_target",
                        "action": "EVENT_CONNECT",
                        "src_type": "process",
                        "dst_type": "netflow",
                        "train_count": "10",
                        "train_max_score": "0.42",
                    }
                )

            summary = replay_result_dir(result_dir)

            self.assertEqual(summary["score_rows"], 2)
            self.assertEqual(summary["raw_alert_rows"], 2)
            self.assertEqual(
                summary["threshold_sources"],
                {
                    "train_group_max": 1,
                    "validation_group_quantile": 1,
                },
            )
            self.assertEqual(summary["train_fallback_rows"], 1)

    def test_replay_uses_validation_group_threshold_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "RESULT"
            result_dir.mkdir()
            with (result_dir / "online_event_score_trace.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "target_case",
                        "action",
                        "src_type",
                        "dst_type",
                        "score",
                        "threshold",
                        "validation_count",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "target_case": "event_semantic_target",
                        "action": "EVENT_READ",
                        "src_type": "file",
                        "dst_type": "process",
                        "score": "0.80",
                        "threshold": "0.99",
                        "validation_count": "0",
                    }
                )
            with (result_dir / "validation_group_threshold_summary.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "target_case",
                        "action",
                        "src_type",
                        "dst_type",
                        "validation_count",
                        "threshold",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "target_case": "event_semantic_target",
                        "action": "EVENT_READ",
                        "src_type": "file",
                        "dst_type": "process",
                        "validation_count": "150",
                        "threshold": "0.70",
                    }
                )

            summary = replay_result_dir(result_dir)

            self.assertEqual(summary["raw_alert_rows"], 1)
            self.assertEqual(summary["threshold_sources"], {"validation_group_quantile": 1})


if __name__ == "__main__":
    unittest.main()
