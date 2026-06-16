import unittest

import numpy as np

from scripts.pipeline.config.runtime_config import ACTION_TYPE_ALERT_POLICIES, SlimConfig
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


class CadetsE3V3PolicySmokeTests(unittest.TestCase):
    def test_policy_is_registered(self):
        self.assertIn("cadets_e4_v3_policy_smoke_v1", ACTION_TYPE_ALERT_POLICIES)

    def test_restores_process_netflow_connect_as_alert(self):
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="cadets_e4_v3_policy_smoke_v1"),
            row=_row(action_id=1, src_type_id=0, dst_type_id=2),
            raw_alert=True,
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "event_alert")
        self.assertEqual(decision["alert_priority"], "high")
        self.assertFalse(decision["node_evidence"])

    def test_preserves_netflow_process_recvfrom_as_high_priority(self):
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="cadets_e4_v3_policy_smoke_v1"),
            row=_row(action_id=5, src_type_id=2, dst_type_id=0),
            raw_alert=True,
        )

        self.assertTrue(decision["final_alert"])
        self.assertEqual(decision["alert_priority"], "high")

    def test_demotes_high_fp_process_file_groups(self):
        groups = [
            (1, 0, 1),
            (8, 0, 1),
            (7, 0, 1),
            (6, 1, 0),
        ]
        for action_id, src_type_id, dst_type_id in groups:
            with self.subTest(group=(action_id, src_type_id, dst_type_id)):
                decision = _action_type_alert_policy_decision(
                    config=SlimConfig(action_type_alert_policy="cadets_e4_v3_policy_smoke_v1"),
                    row=_row(action_id=action_id, src_type_id=src_type_id, dst_type_id=dst_type_id),
                    raw_alert=True,
                )
                self.assertFalse(decision["final_alert"])
                self.assertEqual(decision["alert_decision"], "demoted_event")
                self.assertTrue(decision["node_evidence"])

    def test_non_alert_stays_not_alert(self):
        decision = _action_type_alert_policy_decision(
            config=SlimConfig(action_type_alert_policy="cadets_e4_v3_policy_smoke_v1"),
            row=_row(action_id=1, src_type_id=0, dst_type_id=2),
            raw_alert=False,
        )

        self.assertFalse(decision["final_alert"])
        self.assertEqual(decision["alert_decision"], "not_alert")


if __name__ == "__main__":
    unittest.main()
