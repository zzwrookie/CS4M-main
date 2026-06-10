"""Tests for Phase3G conditional group threshold resolution."""

from __future__ import annotations

import unittest

from cs4m.phase3g.conditional_head import (
    EVENT_SEMANTIC_TARGET_ID,
    GROUP_LEVEL_TARGET_ACTION_SRC_DST,
    make_conditional_group_key,
    resolve_conditional_group_threshold,
)


class Phase3GConditionalThresholdTests(unittest.TestCase):
    """Validate conservative threshold resolution for conditional groups."""

    def test_sufficient_level1_group_threshold_does_not_drop_below_global(self) -> None:
        group_key = make_conditional_group_key(
            GROUP_LEVEL_TARGET_ACTION_SRC_DST,
            int(EVENT_SEMANTIC_TARGET_ID),
            action_id=1,
            src_type_id=2,
            dst_type_id=3,
        )
        group_meta = {
            "group_min_count": 1000,
            "low_support_policy": "conservative_max",
            "thresholds": {
                GROUP_LEVEL_TARGET_ACTION_SRC_DST: {
                    group_key: {
                        "threshold": 0.01,
                        "count": 2000,
                        "eligible": True,
                        "validation_max": 0.02,
                    },
                },
            },
            "global": {
                "threshold": 0.04,
                "count": 10000,
                "threshold_group_key": "global_conditional",
            },
        }

        resolved = resolve_conditional_group_threshold(
            group_meta,
            int(EVENT_SEMANTIC_TARGET_ID),
            action_id=1,
            src_type_id=2,
            dst_type_id=3,
        )

        self.assertEqual(resolved["threshold"], 0.04)
        self.assertEqual(resolved["threshold_level"], "level1_group_quantile_global_floor")
        self.assertEqual(resolved["final_threshold_source"], "global_floor")
        self.assertEqual(resolved["parent_threshold"], 0.01)
        self.assertEqual(resolved["global_threshold"], 0.04)

    def test_sufficient_level1_group_threshold_keeps_group_value_above_global(self) -> None:
        group_key = make_conditional_group_key(
            GROUP_LEVEL_TARGET_ACTION_SRC_DST,
            int(EVENT_SEMANTIC_TARGET_ID),
            action_id=1,
            src_type_id=2,
            dst_type_id=3,
        )
        group_meta = {
            "group_min_count": 1000,
            "low_support_policy": "conservative_max",
            "thresholds": {
                GROUP_LEVEL_TARGET_ACTION_SRC_DST: {
                    group_key: {
                        "threshold": 0.07,
                        "count": 2000,
                        "eligible": True,
                        "validation_max": 0.08,
                    },
                },
            },
            "global": {
                "threshold": 0.04,
                "count": 10000,
                "threshold_group_key": "global_conditional",
            },
        }

        resolved = resolve_conditional_group_threshold(
            group_meta,
            int(EVENT_SEMANTIC_TARGET_ID),
            action_id=1,
            src_type_id=2,
            dst_type_id=3,
        )

        self.assertEqual(resolved["threshold"], 0.07)
        self.assertEqual(resolved["threshold_level"], "level1_group_quantile")
        self.assertEqual(resolved["final_threshold_source"], "level1_group_quantile")
        self.assertEqual(resolved["parent_threshold"], 0.07)
        self.assertEqual(resolved["global_threshold"], 0.04)


if __name__ == "__main__":
    unittest.main()
