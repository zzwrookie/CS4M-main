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
        self.assertIn("--conditional_endpoint_suppression_mode pair_then_endpoint", output)
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


if __name__ == "__main__":
    unittest.main()
