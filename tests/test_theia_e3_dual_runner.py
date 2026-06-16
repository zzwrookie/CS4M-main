"""Tests for the THEIA E3 E4 wrapper dual-head routing."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts/run/run_phase3g_theia_e3_e4_theia_v1_lazy100k_pair_only_full.sh"


class TheiaE3DualRunnerTests(unittest.TestCase):
    """Validate THEIA wrapper preflight and dry-run command construction."""

    def _populate_required_inputs(self, root: Path, *, head_name: str) -> dict[str, str]:
        phase3e = root / "phase3e"
        checkpoint_root = root / "base"
        action_head_root = root / "heads"
        action_validation = root / "action_validation"
        endpoint_cache = root / "endpoint_cache"
        embedder = root / "models" / "THEIA_E3_RAW_DETAIL_RULES_V1_LATENT64.pkl"

        files = [
            embedder,
            checkpoint_root / "THEIA_E3_E4_PHASE3E_BASE_FULL.pkl",
            action_head_root / head_name,
            phase3e / "compact_used_node_embeddings/THEIA_E3/compact_node_embeddings.npy",
            phase3e / "compact_used_node_embeddings/THEIA_E3/compact_embedding_meta.json",
            phase3e / "event_indices/THEIA_E3/event_index_test.memmap",
            phase3e / "event_indices/THEIA_E3/event_index_validation.memmap",
            action_validation / "conditional_train_memmaps/X_conditional_s4d_complex_node.memmap",
            action_validation / "conditional_train_memmaps/Y_conditional_s4d_complex_node.memmap",
            action_validation / "conditional_train_memmaps/target_case_s4d_complex_node.memmap",
        ]
        for path in files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test")
        endpoint_cache.mkdir(parents=True, exist_ok=True)

        return {
            "DRY_RUN": "1",
            "CS4M_ROOT": str(REPO_ROOT),
            "PRETRAINED_RESIDUAL_EMBEDDER_PATH": str(embedder),
            "RESULT_ROOT": str(root / "results"),
            "LOG_DIR": str(root / "logs"),
            "PHASE3E_CACHE_ROOT": str(phase3e),
            "SSPM_BASE_CHECKPOINT_ROOT": str(checkpoint_root),
            "ACTION_HEAD_CHECKPOINT_ROOT": str(action_head_root),
            "ACTION_VALIDATION_CACHE_DIR": str(action_validation),
            "CONDITIONAL_ENDPOINT_SUPPRESSION_CACHE_DIR": str(endpoint_cache),
        }

    def _run(self, env_updates: dict[str, str]) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(env_updates)
        return subprocess.run(
            ["bash", str(RUNNER)],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_dual_arch_routes_dual_head_to_base_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = self._populate_required_inputs(
                Path(temp_dir),
                head_name="THEIA_E3_E4_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl",
            )
            env["SSPM_CONDITIONAL_HEAD_ARCH"] = "dual_lowrank_by_target_case_v2"
            result = self._run(env)

        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, combined)
        self.assertIn("--dataset THEIA_E3", combined)
        self.assertIn("--sspm_conditional_head_arch dual_lowrank_by_target_case_v2", combined)
        self.assertIn("THEIA_E3_E4_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl", combined)
        self.assertIn("--action_type_alert_policy theia_v1", combined)

    def test_default_arch_preserves_shared_head(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = self._populate_required_inputs(
                Path(temp_dir),
                head_name="THEIA_E3_E4_PHASE3G_CONDITIONAL_HEAD.pkl",
            )
            result = self._run(env)

        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, combined)
        self.assertIn("--sspm_conditional_head_arch shared_lowrank_v1", combined)
        self.assertIn("THEIA_E3_E4_PHASE3G_CONDITIONAL_HEAD.pkl", combined)

    def test_dual_arch_requires_dual_head_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = self._populate_required_inputs(
                Path(temp_dir),
                head_name="THEIA_E3_E4_PHASE3G_CONDITIONAL_HEAD.pkl",
            )
            env["SSPM_CONDITIONAL_HEAD_ARCH"] = "dual_lowrank_by_target_case_v2"
            result = self._run(env)

        combined = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("THEIA_E3 E4 conditional dual head", combined)
        self.assertIn("THEIA_E3_E4_PHASE3G_CONDITIONAL_DUAL_HEAD.pkl", combined)


if __name__ == "__main__":
    unittest.main()
