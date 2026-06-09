"""Tests for ClearScope v3 Phase3E/Phase3G runner policy routing."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh"


class ClearScopeV3RunnerPolicyTests(unittest.TestCase):
    """Validate ClearScope v3 runner dry-run command construction."""

    def _run_dry(self, **overrides: str) -> str:
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "STAGE": "infer_full",
                "CLAD_DB_PASSWORD": "dummy",
                "PRETRAINED_RESIDUAL_EMBEDDER_PATH": "/missing/clearscope/v3/embedder.pkl",
                "EVENT_INDEX_CACHE_DIR": "/missing/clearscope/v3/event_index",
                "NODE_EMBEDDING_CACHE_DIR": "/missing/clearscope/v3/node_embeddings",
                "ACTION_EMBEDDING_CACHE_DIR": "/missing/clearscope/v3/action_embeddings",
                "COMPACT_USED_NODE_CACHE_DIR": "/missing/clearscope/v3/compact_nodes",
                "SSPM_CHECKPOINT_PATH": "/missing/clearscope/v3/base.pkl",
                "ACTION_HEAD_CHECKPOINT_PATH": "/missing/clearscope/v3/head.pkl",
            },
        )
        env.update(overrides)
        result = subprocess.run(
            ["bash", str(RUNNER)],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.stdout + result.stderr

    def _run_dry_failure(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "STAGE": "infer_full",
                "CLAD_DB_PASSWORD": "dummy",
                "PRETRAINED_RESIDUAL_EMBEDDER_PATH": "/missing/clearscope/v3/embedder.pkl",
                "EVENT_INDEX_CACHE_DIR": "/missing/clearscope/v3/event_index",
                "NODE_EMBEDDING_CACHE_DIR": "/missing/clearscope/v3/node_embeddings",
                "ACTION_EMBEDDING_CACHE_DIR": "/missing/clearscope/v3/action_embeddings",
                "COMPACT_USED_NODE_CACHE_DIR": "/missing/clearscope/v3/compact_nodes",
                "SSPM_CHECKPOINT_PATH": "/missing/clearscope/v3/base.pkl",
                "ACTION_HEAD_CHECKPOINT_PATH": "/missing/clearscope/v3/head.pkl",
            },
        )
        env.update(overrides)
        return subprocess.run(
            ["bash", str(RUNNER)],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_runner_preserves_baseline_defaults(self) -> None:
        output = self._run_dry()
        self.assertIn(
            "--out_tag CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_FULL",
            output,
        )
        self.assertIn("--event_threshold_mode quantile", output)
        self.assertIn("--action_type_alert_policy default", output)
        self.assertIn("--conditional_endpoint_aware_suppression false", output)
        self.assertIn("--conditional_endpoint_suppression_read false", output)
        self.assertIn("--conditional_endpoint_suppression_mode pair_only", output)
        self.assertIn("--conditional_low_support_policy conservative_max", output)
        self.assertIn("--conditional_low_support_margin 0.05", output)
        self.assertIn("--sspm_state_model s4d_complex_node", output)
        self.assertIn("--sspm_conditional_head_arch shared_lowrank_v1", output)

    def test_runner_routes_policy_overrides_to_cli(self) -> None:
        output = self._run_dry(
            OUT_TAG_OVERRIDE="CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999",
            EVENT_THRESHOLD_MODE="conditional_target_action_type_group_quantile",
            ACTION_TYPE_ALERT_POLICY="clearscope_android_v2",
            CONDITIONAL_ENDPOINT_AWARE_SUPPRESSION="true",
            CONDITIONAL_ENDPOINT_SUPPRESSION_READ="true",
            CONDITIONAL_ENDPOINT_SUPPRESSION_MODE="pair_then_endpoint",
            CONDITIONAL_LOW_SUPPORT_POLICY="adaptive_margin",
            CONDITIONAL_LOW_SUPPORT_MARGIN="0.07",
            SSPM_STATE_MODEL="ema_fixed",
            SSPM_CONDITIONAL_HEAD_ARCH="dual_lowrank_by_target_case_v2",
            CONDITIONAL_BOTH_COLD_UNSEEN_POLICY="observation_only_no_alert",
        )
        self.assertIn(
            "--out_tag CLEARSCOPE_E3_RAW_DETAIL_V3_DISCRIMINATIVE_INFER_GROUPQ0999",
            output,
        )
        self.assertIn(
            "--event_threshold_mode conditional_target_action_type_group_quantile",
            output,
        )
        self.assertIn("--action_type_alert_policy clearscope_android_v2", output)
        self.assertIn("--conditional_endpoint_aware_suppression true", output)
        self.assertIn("--conditional_endpoint_suppression_read true", output)
        self.assertIn(
            "--conditional_endpoint_suppression_mode pair_then_endpoint",
            output,
        )
        self.assertIn("--conditional_low_support_policy adaptive_margin", output)
        self.assertIn("--conditional_low_support_margin 0.07", output)
        self.assertIn("--sspm_state_model ema_fixed", output)
        self.assertIn(
            "--sspm_conditional_head_arch dual_lowrank_by_target_case_v2",
            output,
        )
        self.assertIn(
            "--conditional_both_cold_unseen_policy observation_only_no_alert",
            output,
        )

    def test_policy_override_without_out_tag_override_fails(self) -> None:
        result = self._run_dry_failure(
            ACTION_TYPE_ALERT_POLICY="clearscope_android_v2",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("baseline infer_full out_tag protection", result.stderr)

    def test_out_tag_override_with_all_stage_fails(self) -> None:
        result = self._run_dry_failure(
            STAGE="all",
            OUT_TAG_OVERRIDE="CLEARSCOPE_E3_POLICY_EXPERIMENT",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "OUT_TAG_OVERRIDE is only supported with STAGE=infer_full",
            result.stderr,
        )

    def test_out_tag_override_rejects_path_traversal(self) -> None:
        result = self._run_dry_failure(OUT_TAG_OVERRIDE="../bad")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OUT_TAG_OVERRIDE must not contain", result.stderr)


if __name__ == "__main__":
    unittest.main()
