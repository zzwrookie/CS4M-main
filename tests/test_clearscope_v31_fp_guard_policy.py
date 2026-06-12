"""Tests for ClearScope v31 false-positive guard alert policy."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.pipeline.config.runtime_config import (
    ACTION_TYPE_ALERT_POLICIES,
    ACTION_TYPE_POLICY_EVENT_FIELDS,
    NODE_POOL_SCORE_MODES,
    SlimConfig,
)
from scripts.pipeline.features.conditional_context import _action_type_alert_policy_decision


def _row(action_id: int, src_type_id: int, dst_type_id: int) -> np.ndarray:
    dtype = np.dtype(
        [
            ("action_id", "i4"),
            ("src_type_id", "i4"),
            ("dst_type_id", "i4"),
        ],
    )
    return np.array((action_id, src_type_id, dst_type_id), dtype=dtype)


class ClearScopeV31FpGuardPolicyTests(unittest.TestCase):
    """Validate v31 process/process READ/OPEN demotion policy."""

    def test_policy_is_registered(self) -> None:
        self.assertIn("clearscope_v31_fp_guard_v1", ACTION_TYPE_ALERT_POLICIES)
        self.assertIn("clearscope_v31_fp_guard_v2", ACTION_TYPE_ALERT_POLICIES)
        self.assertIn("clearscope_v31_fp_guard_v3", ACTION_TYPE_ALERT_POLICIES)
        self.assertIn("clearscope_v31_fp_guard_v3b", ACTION_TYPE_ALERT_POLICIES)
        self.assertIn("clearscope_v31_fp_guard_v3c", ACTION_TYPE_ALERT_POLICIES)
        self.assertIn("base_conf_v31_support", NODE_POOL_SCORE_MODES)

    def test_node_evidence_event_fields_include_policy_diagnostics(self) -> None:
        expected = {
            "policy_support_reason",
            "policy_margin_used",
            "src_prior_alert_count",
            "dst_prior_alert_count",
            "src_prior_node_evidence_count",
            "dst_prior_node_evidence_count",
        }

        self.assertTrue(expected.issubset(set(ACTION_TYPE_POLICY_EVENT_FIELDS)))

    def test_demotes_process_process_read_raw_alert(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v1"),
            row=_row(action_id=4, src_type_id=0, dst_type_id=0),
            raw_alert=True,
        )

        self.assertFalse(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "demoted_event")
        self.assertTrue(decision["node_evidence"])

    def test_demotes_process_process_open_raw_alert(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v1"),
            row=_row(action_id=3, src_type_id=0, dst_type_id=0),
            raw_alert=True,
        )

        self.assertFalse(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "demoted_event")
        self.assertTrue(decision["node_evidence"])

    def test_preserves_file_process_read_raw_alert(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v1"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "event_alert")
        self.assertFalse(decision["node_evidence"])

    def test_preserves_process_file_write_raw_alert(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v1"),
            row=_row(action_id=9, src_type_id=0, dst_type_id=1),
            raw_alert=True,
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "event_alert")
        self.assertFalse(decision["node_evidence"])

    def test_non_raw_alert_remains_not_alert(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v1"),
            row=_row(action_id=4, src_type_id=0, dst_type_id=0),
            raw_alert=False,
        )

        self.assertFalse(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "not_alert")

    def test_v2_demotes_process_file_write_below_floor(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v2"),
            row=_row(action_id=9, src_type_id=0, dst_type_id=1),
            raw_alert=True,
            score=0.084,
            threshold=0.057,
        )

        self.assertFalse(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "demoted_event")
        self.assertTrue(decision["node_evidence"])

    def test_v2_preserves_process_file_write_at_floor(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v2"),
            row=_row(action_id=9, src_type_id=0, dst_type_id=1),
            raw_alert=True,
            score=0.085,
            threshold=0.057,
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "event_alert")
        self.assertFalse(decision["node_evidence"])

    def test_v2_demotes_file_process_read_below_floor(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v2"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.059,
            threshold=0.051,
        )

        self.assertFalse(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "demoted_event")
        self.assertTrue(decision["node_evidence"])

    def test_v2_preserves_file_process_read_at_floor(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v2"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.060,
            threshold=0.051,
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "event_alert")
        self.assertFalse(decision["node_evidence"])

    def test_v3_demotes_narrow_common_file_process_read_to_node_evidence(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.062,
            threshold=0.051,
            alert_row={
                "validation_group_count": 2000,
                "endpoint_validation_pair_count": 25,
                "src_endpoint_count": 25,
                "validation_count_bucket": "high",
                "target_case": "warm",
            },
        )

        self.assertFalse(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "demoted_event")
        self.assertTrue(decision["node_evidence"])

    def test_v3_preserves_high_score_file_process_read(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.080,
            threshold=0.051,
            alert_row={
                "validation_group_count": 2000,
                "endpoint_validation_pair_count": 25,
                "src_endpoint_count": 25,
                "validation_count_bucket": "high",
                "target_case": "warm",
            },
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "event_alert")
        self.assertFalse(decision["node_evidence"])

    def test_v3_preserves_low_support_file_process_read(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.062,
            threshold=0.051,
            alert_row={
                "validation_group_count": 200,
                "endpoint_validation_pair_count": 2,
                "src_endpoint_count": 2,
                "validation_count_bucket": "low",
                "target_case": "warm",
            },
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "event_alert")
        self.assertFalse(decision["node_evidence"])

    def test_v3b_preserves_first_near_floor_file_process_read(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3b"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.062,
            threshold=0.051,
            alert_row={
                "validation_group_count": 2000,
                "validation_count_bucket": "sufficient",
                "target_case": "event_semantic_target",
            },
            policy_context={
                "src_alert_count": 0,
                "dst_alert_count": 0,
                "src_node_evidence_count": 0,
                "dst_node_evidence_count": 0,
            },
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "event_alert")
        self.assertFalse(decision["node_evidence"])

    def test_v3b_demotes_near_floor_read_with_prior_alert_support(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3b"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.062,
            threshold=0.051,
            alert_row={
                "validation_group_count": 2000,
                "validation_count_bucket": "sufficient",
                "target_case": "event_semantic_target",
            },
            policy_context={
                "src_alert_count": 0,
                "dst_alert_count": 1,
                "src_node_evidence_count": 0,
                "dst_node_evidence_count": 0,
            },
        )

        self.assertFalse(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "demoted_event")
        self.assertTrue(decision["node_evidence"])

    def test_v3b_demotes_near_floor_write_with_prior_evidence_support(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3b"),
            row=_row(action_id=9, src_type_id=0, dst_type_id=1),
            raw_alert=True,
            score=0.086,
            threshold=0.057,
            alert_row={
                "validation_group_count": 2000,
                "validation_count_bucket": "sufficient",
                "target_case": "event_semantic_target",
            },
            policy_context={
                "src_alert_count": 0,
                "dst_alert_count": 0,
                "src_node_evidence_count": 2,
                "dst_node_evidence_count": 0,
            },
        )

        self.assertFalse(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "demoted_event")
        self.assertTrue(decision["node_evidence"])

    def test_v3b_preserves_low_support_near_floor_read(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3b"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.062,
            threshold=0.051,
            alert_row={
                "validation_group_count": 200,
                "validation_count_bucket": "low",
                "target_case": "event_semantic_target",
            },
            policy_context={
                "src_alert_count": 0,
                "dst_alert_count": 1,
                "src_node_evidence_count": 0,
                "dst_node_evidence_count": 0,
            },
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "event_alert")
        self.assertFalse(decision["node_evidence"])

    def test_v3b_preserves_high_score_read_with_prior_support(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3b"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.070,
            threshold=0.051,
            alert_row={
                "validation_group_count": 2000,
                "validation_count_bucket": "sufficient",
                "target_case": "event_semantic_target",
            },
            policy_context={
                "src_alert_count": 0,
                "dst_alert_count": 1,
                "src_node_evidence_count": 0,
                "dst_node_evidence_count": 0,
            },
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "event_alert")
        self.assertFalse(decision["node_evidence"])

    def test_v3c_preserves_read_with_only_one_prior_alert(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3c"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.061,
            threshold=0.051,
            alert_row={
                "validation_group_count": 2000,
                "validation_count_bucket": "sufficient",
                "target_case": "event_semantic_target",
            },
            policy_context={
                "src_alert_count": 0,
                "dst_alert_count": 1,
                "src_node_evidence_count": 0,
                "dst_node_evidence_count": 0,
            },
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "event_alert")
        self.assertFalse(decision["node_evidence"])

    def test_v3c_demotes_read_with_two_prior_alerts_and_narrow_margin(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3c"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.061,
            threshold=0.051,
            alert_row={
                "validation_group_count": 2000,
                "validation_count_bucket": "sufficient",
                "target_case": "event_semantic_target",
            },
            policy_context={
                "src_alert_count": 0,
                "dst_alert_count": 2,
                "src_node_evidence_count": 0,
                "dst_node_evidence_count": 0,
            },
        )

        self.assertFalse(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "demoted_event")
        self.assertEqual(
            decision.get("policy_support_reason"),
            "v3c_read_strict_prior_alert",
        )
        self.assertTrue(decision["node_evidence"])

    def test_v3c_preserves_read_above_strict_margin(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3c"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.062,
            threshold=0.051,
            alert_row={
                "validation_group_count": 2000,
                "validation_count_bucket": "sufficient",
                "target_case": "event_semantic_target",
            },
            policy_context={
                "src_alert_count": 0,
                "dst_alert_count": 2,
                "src_node_evidence_count": 0,
                "dst_node_evidence_count": 0,
            },
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "event_alert")
        self.assertFalse(decision["node_evidence"])

    def test_v3c_demotes_read_with_four_prior_evidence_events(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3c"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.061,
            threshold=0.051,
            alert_row={
                "validation_group_count": 2000,
                "validation_count_bucket": "sufficient",
                "target_case": "event_semantic_target",
            },
            policy_context={
                "src_alert_count": 0,
                "dst_alert_count": 0,
                "src_node_evidence_count": 0,
                "dst_node_evidence_count": 4,
            },
        )

        self.assertFalse(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "demoted_event")
        self.assertEqual(
            decision.get("policy_support_reason"),
            "v3c_read_strict_prior_evidence",
        )
        self.assertTrue(decision["node_evidence"])

    def test_v3c_preserves_low_support_read_even_with_prior_support(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3c"),
            row=_row(action_id=4, src_type_id=1, dst_type_id=0),
            raw_alert=True,
            score=0.061,
            threshold=0.051,
            alert_row={
                "validation_group_count": 2000,
                "validation_count_bucket": "low",
                "target_case": "event_semantic_target",
            },
            policy_context={
                "src_alert_count": 0,
                "dst_alert_count": 2,
                "src_node_evidence_count": 0,
                "dst_node_evidence_count": 4,
            },
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "event_alert")
        self.assertFalse(decision["node_evidence"])

    def test_v3c_keeps_v3b_write_prior_evidence_behavior(self) -> None:
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="clearscope_v31_fp_guard_v3c"),
            row=_row(action_id=9, src_type_id=0, dst_type_id=1),
            raw_alert=True,
            score=0.086,
            threshold=0.057,
            alert_row={
                "validation_group_count": 2000,
                "validation_count_bucket": "sufficient",
                "target_case": "event_semantic_target",
            },
            policy_context={
                "src_alert_count": 0,
                "dst_alert_count": 0,
                "src_node_evidence_count": 2,
                "dst_node_evidence_count": 0,
            },
        )

        self.assertFalse(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "demoted_event")
        self.assertTrue(decision["node_evidence"])


if __name__ == "__main__":
    unittest.main()
