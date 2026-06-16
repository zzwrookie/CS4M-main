"""Tests for ClearScope v3 TP/node recall audit helpers."""

from __future__ import annotations

import unittest

from tmp.clearscope_v3_tp_node_recall_audit import (
    classify_node_failure,
    node_pool_rank_lookup,
    percentile,
)


class ClearScopeV3TPNodeRecallAuditTests(unittest.TestCase):
    """Validate pure helper behavior for the post-inference audit."""

    def test_percentile_handles_empty_and_sorted_values(self) -> None:
        self.assertEqual(percentile([], 0.99), 0.0)
        self.assertEqual(percentile([1.0], 0.99), 1.0)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    def test_node_pool_rank_lookup_returns_one_based_rank(self) -> None:
        rows = [
            {"node_id": "10", "node_score": "0.9"},
            {"node_id": "20", "node_score": "0.8"},
        ]
        lookup = node_pool_rank_lookup(rows)

        self.assertEqual(lookup[10]["rank"], 1)
        self.assertEqual(lookup[20]["rank"], 2)
        self.assertEqual(lookup[20]["node_score"], 0.8)

    def test_failure_reason_no_test_events(self) -> None:
        self.assertEqual(
            classify_node_failure(
                test_event_count=0,
                above_threshold_count=0,
                is_in_pool=False,
                rank=0,
                token_collision=False,
                has_oov=False,
            ),
            "no_test_events",
        )

    def test_failure_reason_below_threshold(self) -> None:
        self.assertEqual(
            classify_node_failure(
                test_event_count=4,
                above_threshold_count=0,
                is_in_pool=False,
                rank=0,
                token_collision=False,
                has_oov=False,
            ),
            "events_scored_below_threshold",
        )

    def test_failure_reason_above_threshold_not_in_pool(self) -> None:
        self.assertEqual(
            classify_node_failure(
                test_event_count=4,
                above_threshold_count=2,
                is_in_pool=False,
                rank=0,
                token_collision=False,
                has_oov=False,
            ),
            "events_above_threshold_but_node_not_in_pool",
        )

    def test_failure_reason_rank_too_low_before_token_causes(self) -> None:
        self.assertEqual(
            classify_node_failure(
                test_event_count=4,
                above_threshold_count=2,
                is_in_pool=True,
                rank=5000,
                token_collision=True,
                has_oov=True,
            ),
            "node_in_pool_but_rank_too_low",
        )

    def test_failure_reason_token_collision(self) -> None:
        self.assertEqual(
            classify_node_failure(
                test_event_count=4,
                above_threshold_count=2,
                is_in_pool=True,
                rank=20,
                token_collision=True,
                has_oov=False,
            ),
            "node_token_collision",
        )

    def test_failure_reason_oov(self) -> None:
        self.assertEqual(
            classify_node_failure(
                test_event_count=4,
                above_threshold_count=2,
                is_in_pool=True,
                rank=20,
                token_collision=False,
                has_oov=True,
            ),
            "node_embedding_oov_or_generic",
        )


if __name__ == "__main__":
    unittest.main()
