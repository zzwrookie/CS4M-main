import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class CadetsE3V3PolicySmokeRunnerTests(unittest.TestCase):
    def test_dry_run_uses_v3_policy_and_existing_artifacts(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts/run/run_cadets_e3_v3_policy_smoke.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            embedder = temp_path / "CADETS_E3_FREEBSD_V3_SAFE_LEXICAL_LATENT64_word2vec_window3.pkl"
            base = temp_path / "CADETS_E3_FREEBSD_V3_SAFE_LEXICAL_E4_PHASE3E_BASE.pkl"
            head = temp_path / "CADETS_E3_FREEBSD_V3_SAFE_LEXICAL_E4_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl"
            event_index = temp_path / "event_index"
            node_cache = temp_path / "node"
            action_cache = temp_path / "action"
            for path in [embedder, base, head]:
                path.write_bytes(b"test")
            event_index.mkdir()
            node_cache.mkdir()
            action_cache.mkdir()
            (event_index / "event_index_meta.json").write_text("{}", encoding="utf-8")
            (node_cache / "node_embeddings.npy").write_bytes(b"test")
            (action_cache / "action_embeddings.npy").write_bytes(b"test")
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "CS4M_ROOT": str(repo_root),
                    "PRETRAINED_RESIDUAL_EMBEDDER_PATH": str(embedder),
                    "SSPM_CHECKPOINT_PATH": str(base),
                    "ACTION_HEAD_CHECKPOINT_PATH": str(head),
                    "EVENT_INDEX_CACHE_DIR": str(event_index),
                    "NODE_EMBEDDING_CACHE_DIR": str(node_cache),
                    "ACTION_EMBEDDING_CACHE_DIR": str(action_cache),
                    "COMPACT_USED_NODE_CACHE_DIR": str(temp_path / "compact"),
                    "ACTION_VALIDATION_CACHE_DIR": str(temp_path / "validation"),
                    "CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR": str(temp_path / "endpoint"),
                    "RESULT_ROOT": str(temp_path / "results"),
                    "LOG_DIR": str(temp_path / "logs"),
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
        self.assertIn("--max_test_events 100000", combined)
        self.assertIn("--max_train_events 0", combined)
        self.assertIn("--max_ref_events 0", combined)
        self.assertIn("--semantic_mode cadets_freebsd_raw_detail_v3_safe_lexical", combined)
        self.assertIn("--action_type_alert_policy cadets_e4_v3_policy_smoke_v1", combined)
        self.assertIn("--sspm_train_mode load_and_infer", combined)
        self.assertIn("--rss_profile_mode online_minimal", combined)
        self.assertIn("--write_raw_alerts true", combined)
        self.assertIn("--write_analysis_outputs true", combined)
        self.assertIn("CADETS_E3_FREEBSD_V3_SAFE_LEXICAL_E4_POLICY_SMOKE_100K", combined)
        self.assertNotIn("train_phase3e_base", combined)
        self.assertNotIn("train_conditional_and_save", combined)
        self.assertNotIn("train_residual_word2vec_models.py", combined)

    def test_rejects_non_v3_semantic_mode(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts/run/run_cadets_e3_v3_policy_smoke.sh"
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "CS4M_ROOT": str(repo_root),
                "SEMANTIC_MODE": "raw_detail_v2_refined",
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
