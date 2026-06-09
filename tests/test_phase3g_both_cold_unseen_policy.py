"""Tests for Phase3G both_cold unseen-group no-alert policy."""

from __future__ import annotations

import unittest

from cs4m.phase3g.conditional_head import BOTH_COLD_ACTION_TARGET, EVENT_SEMANTIC_TARGET
from scripts.pipeline.conditional.infer import _phase3g_should_suppress_both_cold_unseen_alert
from scripts.pipeline.config.runtime_config import SlimConfig
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


if __name__ == "__main__":
    unittest.main()
