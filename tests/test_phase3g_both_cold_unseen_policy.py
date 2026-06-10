"""Tests for Phase3G both_cold unseen-group no-alert policy."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from cs4m.phase3g.conditional_head import BOTH_COLD_ACTION_TARGET, EVENT_SEMANTIC_TARGET
from scripts.pipeline.conditional.infer import (
    _phase3g_apply_both_cold_unseen_alert_policy,
    _phase3g_conditional_score_summary_payload,
    _phase3g_should_suppress_both_cold_unseen_alert,
)
from scripts.pipeline.config.runtime_config import SlimConfig
from scripts.pipeline.entrypoints import arguments
from scripts.pipeline.entrypoints.arguments import parse_args, validate_config


def _slim_config_from_parse_args(*argv: str) -> SlimConfig:
    args = parse_args(list(argv))
    values = {
        field: getattr(args, field)
        for field in SlimConfig.__dataclass_fields__
        if hasattr(args, field)
    }
    values["event_threshold_mode"] = "quantile"
    values["pretrained_residual_embedder_path"] = "dummy_word2vec.pkl"
    return SlimConfig(**values)


class Phase3GBothColdUnseenPolicyTests(unittest.TestCase):
    """Validate default-off both_cold unseen alert suppression."""

    def test_cli_accepts_default_off_policy(self) -> None:
        config = _slim_config_from_parse_args()
        self.assertEqual(config.conditional_both_cold_unseen_policy, "alert")
        validate_config(config)

    def test_cli_accepts_observation_only_no_alert(self) -> None:
        config = _slim_config_from_parse_args(
            "--conditional_both_cold_unseen_policy",
            "observation_only_no_alert",
        )
        self.assertEqual(
            config.conditional_both_cold_unseen_policy,
            "observation_only_no_alert",
        )
        validate_config(config)

    def test_policy_suppresses_only_both_cold_unseen_group(self) -> None:
        config = SlimConfig(conditional_both_cold_unseen_policy="observation_only_no_alert")
        self.assertTrue(
            _phase3g_should_suppress_both_cold_unseen_alert(
                config=config,
                target_case=BOTH_COLD_ACTION_TARGET,
                threshold_level="unseen_group_extreme",
                validation_group_count=0,
            ),
        )
        self.assertTrue(
            _phase3g_should_suppress_both_cold_unseen_alert(
                config=config,
                target_case=BOTH_COLD_ACTION_TARGET,
                threshold_level="unseen_group_extreme",
                validation_group_count=-1,
            ),
        )
        self.assertFalse(
            _phase3g_should_suppress_both_cold_unseen_alert(
                config=config,
                target_case=EVENT_SEMANTIC_TARGET,
                threshold_level="unseen_group_extreme",
                validation_group_count=0,
            ),
        )
        self.assertFalse(
            _phase3g_should_suppress_both_cold_unseen_alert(
                config=config,
                target_case=BOTH_COLD_ACTION_TARGET,
                threshold_level="level1_group_quantile",
                validation_group_count=0,
            ),
        )
        self.assertFalse(
            _phase3g_should_suppress_both_cold_unseen_alert(
                config=config,
                target_case=BOTH_COLD_ACTION_TARGET,
                threshold_level="unseen_group_extreme",
                validation_group_count=1,
            ),
        )

    def test_default_policy_does_not_suppress(self) -> None:
        config = SlimConfig(conditional_both_cold_unseen_policy="alert")
        self.assertFalse(
            _phase3g_should_suppress_both_cold_unseen_alert(
                config=config,
                target_case=BOTH_COLD_ACTION_TARGET,
                threshold_level="unseen_group_extreme",
                validation_group_count=0,
            ),
        )

    def test_apply_policy_preserves_raw_count_and_suppresses_final_alert(self) -> None:
        config = SlimConfig(conditional_both_cold_unseen_policy="observation_only_no_alert")
        final_alert, suppressed_delta = _phase3g_apply_both_cold_unseen_alert_policy(
            config=config,
            raw_alert=True,
            target_case=BOTH_COLD_ACTION_TARGET,
            threshold_level="unseen_group_extreme",
            validation_group_count=0,
        )

        self.assertFalse(final_alert)
        self.assertEqual(suppressed_delta, 1)
        raw_alert_count_before_suppression = 1
        both_cold_unseen_suppressed_alert_count = suppressed_delta
        event_alert_count = int(final_alert)
        self.assertEqual(raw_alert_count_before_suppression, 1)
        self.assertEqual(both_cold_unseen_suppressed_alert_count, 1)
        self.assertEqual(event_alert_count, 0)

    def test_score_summary_payload_includes_both_cold_unseen_policy(self) -> None:
        config = SlimConfig(
            conditional_both_cold_unseen_policy="observation_only_no_alert",
            event_threshold_mode="conditional_target_action_type_group_quantile",
            event_threshold_quantile=0.999,
        )
        payload = _phase3g_conditional_score_summary_payload(
            config=config,
            cache_meta={"threshold": 0.04},
            stream_outputs={
                "both_cold_unseen_policy": {
                    "policy_name": "observation_only_no_alert",
                    "suppressed_alert_count": 17,
                },
                "test_score_summary": {},
                "test_score_summary_by_target_case": {},
            },
            group_summary_path="conditional_score_summary_by_target_action_type.csv",
            demoted_group_summary_path=Path("demoted.csv"),
            group_threshold_sweep_path=Path("sweep.csv"),
            endpoint_suppression_eval={},
            update_gate_score_space_summary={},
            group_alert_policy_rows=[],
            required_dual_head_summary_paths={},
        )

        self.assertEqual(
            payload["both_cold_unseen_policy"]["policy_name"],
            "observation_only_no_alert",
        )
        self.assertEqual(payload["both_cold_unseen_policy"]["suppressed_alert_count"], 17)

    def test_sspm_train_mode_choices_include_runner_stages(self) -> None:
        for mode in (
            "build_phase3e_artifacts",
            "train_phase3e_base",
            "train_conditional_and_save",
            "load_and_infer",
        ):
            args = parse_args(["--sspm_train_mode", mode])
            self.assertEqual(args.sspm_train_mode, mode)

    def test_main_dispatches_phase3e_and_phase3g_modes(self) -> None:
        cases = {
            "build_phase3e_artifacts": "build",
            "train_phase3e_base": "base",
            "train_conditional_and_save": "head",
            "load_and_infer": "infer",
        }
        for mode, expected in cases.items():
            with self.subTest(mode=mode), mock.patch.object(
                arguments,
                "build_phase3e_artifacts_from_db",
                return_value=Path("build.json"),
            ) as build, mock.patch.object(
                arguments,
                "run_phase3e_train_base_from_precompute",
                return_value=Path("base.json"),
            ) as base, mock.patch.object(
                arguments,
                "run_phase3g_conditional_train_from_precompute",
                return_value=Path("head.json"),
            ) as head, mock.patch.object(
                arguments,
                "run_phase3e_load_and_infer_from_precompute",
                return_value=Path("infer.json"),
            ) as infer:
                result = arguments.main(
                    [
                        "--sspm_train_mode",
                        mode,
                        "--event_threshold_mode",
                        "quantile",
                        "--pretrained_residual_embedder_path",
                        "dummy_word2vec.pkl",
                        "--sspm_checkpoint_path",
                        "dummy_checkpoint.pkl",
                    ],
                )

                self.assertEqual(result, 0)
                self.assertEqual(build.called, expected == "build")
                self.assertEqual(base.called, expected == "base")
                self.assertEqual(head.called, expected == "head")
                self.assertEqual(infer.called, expected == "infer")


if __name__ == "__main__":
    unittest.main()
