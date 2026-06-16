import argparse
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from cs4m.semantics.cadets_freebsd import CADETS_V3_SAFE_LEXICAL_SEMANTIC_MODE
from legacy.tools.train_residual_word2vec_models import _word2vec_metadata_semantic_payload


class CadetsV3Word2VecRunnerTests(unittest.TestCase):
    def test_word2vec_metadata_records_selected_cadets_v3_semantic_mode(self):
        args = argparse.Namespace(semantic_mode=CADETS_V3_SAFE_LEXICAL_SEMANTIC_MODE)

        payload = _word2vec_metadata_semantic_payload(args, "CADETS_E3")

        self.assertEqual(payload["semantic_mode"], CADETS_V3_SAFE_LEXICAL_SEMANTIC_MODE)
        self.assertEqual(
            payload["semantic_rules"]["cadets"],
            CADETS_V3_SAFE_LEXICAL_SEMANTIC_MODE,
        )
        self.assertFalse(payload["legacy_path_used"])

    def test_runner_dry_run_prints_train_only_v3_word2vec_command(self):
        repo_root = Path(__file__).resolve().parents[1]
        runner = repo_root / "scripts/run/run_cadets_e3_v3_safe_lexical_word2vec.sh"
        with tempfile.TemporaryDirectory() as temp_dir:
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "CS4M_ROOT": str(repo_root),
                    "LOG_DIR": str(Path(temp_dir) / "logs"),
                    "WORD2VEC_ROOT": str(Path(temp_dir) / "word2vec"),
                    "WORD2VEC_CORPUS_ROOT": str(Path(temp_dir) / "corpus"),
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
        self.assertIn("legacy/tools/train_residual_word2vec_models.py", combined)
        self.assertIn("--datasets CADETS_E3", combined)
        self.assertIn(
            "--semantic_mode cadets_freebsd_raw_detail_v3_safe_lexical",
            combined,
        )
        self.assertIn("--out_tag FREEBSD_V3_SAFE_LEXICAL_LATENT64", combined)
        self.assertNotIn("conditional_e4", combined)


if __name__ == "__main__":
    unittest.main()
