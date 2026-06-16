import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class CadetsV3FullSerialRunnerTests(unittest.TestCase):
    def test_dry_run_all_prints_v3_e4_dual_serial_commands(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts/run/run_cadets_e3_v3_safe_lexical_full.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            embedder = temp_path / "models" / (
                "CADETS_E3_FREEBSD_V3_SAFE_LEXICAL_LATENT64_word2vec_window3.pkl"
            )
            embedder.parent.mkdir(parents=True)
            embedder.write_bytes(b"test")
            capture_path = temp_path / "runner_args.txt"
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "CS4M_ROOT": str(repo_root),
                    "PRETRAINED_RESIDUAL_EMBEDDER_PATH": str(embedder),
                    "RESULT_ROOT": str(temp_path / "results"),
                    "PHASE3E_CACHE_ROOT": str(temp_path / "phase3e"),
                    "LOG_DIR": str(temp_path / "logs"),
                    "SSPM_BASE_CHECKPOINT_ROOT": str(temp_path / "base"),
                    "ACTION_HEAD_CHECKPOINT_ROOT": str(temp_path / "heads"),
                    "RUNNER_ARG_CAPTURE": str(capture_path),
                }
            )
            result = subprocess.run(
                ["bash", str(runner)],
                cwd=repo_root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, combined)
        self.assertIn("--dataset CADETS_E3", combined)
        self.assertIn(
            "--semantic_mode cadets_freebsd_raw_detail_v3_safe_lexical",
            combined,
        )
        self.assertIn("--sspm_score_target_mode event_action_semantic", combined)
        self.assertIn("--sspm_score_head conditional_action_semantic", combined)
        self.assertIn("--sspm_conditional_head_arch dual_lowrank_by_target_case_v2", combined)
        self.assertIn("--event_threshold_mode target_case_quantile", combined)
        self.assertIn("--event_threshold_quantile 0.9995", combined)
        self.assertIn("--action_type_alert_policy cadets_e4_v2_group_v1", combined)
        self.assertIn("--max_train_events 0", combined)
        self.assertIn("--max_ref_events 0", combined)
        self.assertIn("--max_test_events 0", combined)
        self.assertIn("CADETS_E3_FREEBSD_V3_SAFE_LEXICAL_E4_FULL", combined)
        self.assertIn("CADETS_E3_freebsd_v3_safe_lexical_latent64", combined)
        self.assertIn("phase3g_action_validation_v3_Q09995", combined)
        self.assertIn("--sspm_train_mode build_phase3e_artifacts", combined)
        self.assertIn("--sspm_train_mode train_phase3e_base", combined)
        self.assertIn("--sspm_train_mode train_conditional_and_save", combined)
        self.assertIn("--sspm_train_mode load_and_infer", combined)
        self.assertNotIn("train_residual_word2vec_models.py", combined)
        self.assertNotIn("RAW_DETAIL_RULES_V1", combined)

    def test_rejects_non_v3_semantic_mode(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts/run/run_cadets_e3_v3_safe_lexical_full.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            embedder = Path(temp_dir) / (
                "CADETS_E3_FREEBSD_V3_SAFE_LEXICAL_LATENT64_word2vec_window3.pkl"
            )
            embedder.write_bytes(b"test")
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "CS4M_ROOT": str(repo_root),
                    "SEMANTIC_MODE": "raw_detail_v2_refined",
                    "PRETRAINED_RESIDUAL_EMBEDDER_PATH": str(embedder),
                }
            )
            result = subprocess.run(
                ["bash", str(runner)],
                cwd=repo_root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SEMANTIC_MODE must be", result.stderr)


if __name__ == "__main__":
    unittest.main()
