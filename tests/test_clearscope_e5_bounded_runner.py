import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts/run/run_clearscope_e5_v33b_bounded_phase3e_phase3g_smoke.sh"


def _run_runner(extra_env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "CS4M_ROOT": str(REPO_ROOT),
            "DRY_RUN": "1",
            "PYTHON_BIN": "python3",
        },
    )
    env.update(extra_env)
    return subprocess.run(
        ["bash", str(RUNNER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class ClearScopeE5BoundedRunnerTests(unittest.TestCase):
    def test_dry_run_all_uses_e5_v33b_bounded_artifacts(self) -> None:
        result = _run_runner({"STAGE": "all"})

        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        combined = result.stdout + result.stderr
        self.assertIn("train_word2vec", combined)
        self.assertIn("--datasets CLEARSCOPE_E5", combined)
        self.assertIn("--dataset CLEARSCOPE_E5", combined)
        self.assertIn("--semantic_mode raw_detail_v33b_e5_android_safe", combined)
        self.assertIn("--max_train_events 50000", combined)
        self.assertIn("--max_ref_events 50000", combined)
        self.assertIn("--max_test_events 50000", combined)
        self.assertIn("CLEARSCOPE_E5_V33B_BOUNDED", combined)
        self.assertNotIn("CLEARSCOPE_E3", combined)

    def test_rejects_e3_dataset_even_in_dry_run(self) -> None:
        result = _run_runner({"DATASET": "CLEARSCOPE_E3", "STAGE": "all"})

        self.assertEqual(result.returncode, 2)
        self.assertIn("restricted to CLEARSCOPE_E5", result.stderr)

    def test_rejects_non_v33b_semantic_mode_even_in_dry_run(self) -> None:
        result = _run_runner(
            {
                "SEMANTIC_MODE": "raw_detail_v33_e5_android_safe",
                "STAGE": "all",
            },
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("raw_detail_v33b_e5_android_safe", result.stderr)

    def test_shell_syntax_is_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
