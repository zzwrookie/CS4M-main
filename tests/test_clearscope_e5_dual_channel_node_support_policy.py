import unittest

from scripts.pipeline.conditional.infer import (
    _DualChannelNodeSupportState,
    _dual_channel_event_nodes,
    _dual_channel_node_support_update,
)


class ClearScopeE5DualChannelNodeSupportPolicyTests(unittest.TestCase):
    def test_second_near_threshold_event_triggers_once(self):
        state = _DualChannelNodeSupportState()
        first = _dual_channel_node_support_update(
            state,
            node_idx=10,
            node_role="src",
            stream_pos=1,
            event_index=101,
            event_score=0.13,
            score_floor=0.12,
            support_threshold=2,
            source_target_case="event_semantic_target",
            alert_channel="event_semantic_node_support_ge_2",
        )
        second = _dual_channel_node_support_update(
            state,
            node_idx=10,
            node_role="src",
            stream_pos=2,
            event_index=102,
            event_score=0.14,
            score_floor=0.12,
            support_threshold=2,
            source_target_case="event_semantic_target",
            alert_channel="event_semantic_node_support_ge_2",
        )
        third = _dual_channel_node_support_update(
            state,
            node_idx=10,
            node_role="src",
            stream_pos=3,
            event_index=103,
            event_score=0.15,
            score_floor=0.12,
            support_threshold=2,
            source_target_case="event_semantic_target",
            alert_channel="event_semantic_node_support_ge_2",
        )

        self.assertIsNone(first)
        self.assertIsNotNone(second)
        self.assertIsNone(third)
        self.assertEqual(second["trigger_node_idx"], 10)
        self.assertEqual(second["trigger_node_role"], "src")
        self.assertEqual(second["event_semantic_near_threshold_count"], 2)
        self.assertEqual(second["event_semantic_support_event_count"], 2)
        self.assertAlmostEqual(second["event_semantic_max_score_seen"], 0.14)
        self.assertAlmostEqual(second["score_floor"], 0.12)
        self.assertEqual(second["support_threshold"], 2)
        self.assertEqual(second["threshold_source"], "validation_event_semantic_p999_half")

    def test_below_floor_event_does_not_increment_near_threshold_count(self):
        state = _DualChannelNodeSupportState()
        result = _dual_channel_node_support_update(
            state,
            node_idx=11,
            node_role="dst",
            stream_pos=1,
            event_index=201,
            event_score=0.10,
            score_floor=0.12,
            support_threshold=1,
            source_target_case="event_semantic_target",
            alert_channel="event_semantic_node_support_ge_2",
        )

        self.assertIsNone(result)
        node_state = state.nodes[11]
        self.assertEqual(node_state["event_semantic_support_event_count"], 1)
        self.assertEqual(node_state["event_semantic_near_threshold_count"], 0)
        self.assertAlmostEqual(node_state["event_semantic_max_score_seen"], 0.10)

    def test_event_nodes_deduplicates_same_src_dst_node(self):
        row = {
            "src_node_idx": 33,
            "dst_node_idx": 33,
        }

        self.assertEqual(_dual_channel_event_nodes(row), [(33, "src")])


if __name__ == "__main__":
    unittest.main()
