import unittest

from scripts.pipeline.config.runtime_config import (
    ENTITY_TYPES,
    EVENT_SEMANTIC_TARGET,
    EVENT_SEMANTIC_TARGET_ID,
    OnlineNodeCoverageTracker,
)
from scripts.pipeline.outputs.conditional_reports import (
    _ConditionalGroupStreamingSummary,
    _TargetCaseStreamingSummary,
    _phase3g_compact_gt_labels_for_eval,
    _phase3g_label_for_alert_row,
    _phase3g_load_validation_group_scores,
)


class ConditionalReportsStreamingSummaryTests(unittest.TestCase):
    def test_target_case_summary_can_observe_scores(self) -> None:
        summary = _TargetCaseStreamingSummary(bins=100)

        summary.observe(EVENT_SEMANTIC_TARGET, 0.5)
        payload = summary.summary({EVENT_SEMANTIC_TARGET: 0.4})

        self.assertEqual(payload[EVENT_SEMANTIC_TARGET]["count"], 1)
        self.assertEqual(payload[EVENT_SEMANTIC_TARGET]["test_above_threshold_count"], 1)
        self.assertEqual(payload[EVENT_SEMANTIC_TARGET]["target_case"], EVENT_SEMANTIC_TARGET)

    def test_group_summary_rows_include_entity_type_names(self) -> None:
        summary = _ConditionalGroupStreamingSummary(bins=100)

        summary.observe(
            EVENT_SEMANTIC_TARGET_ID,
            action_id=0,
            src_type_id=ENTITY_TYPES["process"],
            dst_type_id=ENTITY_TYPES["file"],
            score=0.5,
            alert=True,
        )
        rows = summary.rows({"threshold": 0.4, "group_thresholds": {}})

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_case"], EVENT_SEMANTIC_TARGET)
        self.assertEqual(rows[0]["src_type_name"], "process")
        self.assertEqual(rows[0]["dst_type_name"], "file")
        self.assertEqual(rows[0]["alert_count"], 1)

    def test_missing_validation_group_key_path_returns_empty_scores(self) -> None:
        grouped = _phase3g_load_validation_group_scores(
            {
                "count": 1,
                "validation_conditional_scores": "",
                "validation_conditional_group_keys": "",
            },
        )

        self.assertEqual(grouped, {})

    def test_online_node_coverage_alert_count_returns_observed_count(self) -> None:
        tracker = OnlineNodeCoverageTracker()

        self.assertEqual(tracker.alert_count(42), 0)

        tracker.observe(
            node_idx=42,
            node_type="process",
            event_id=100,
            event_score=0.7,
            side="src",
        )
        tracker.observe(
            node_idx=42,
            node_type="process",
            event_id=101,
            event_score=0.8,
            side="dst",
        )

        self.assertEqual(tracker.alert_count(42), 2)

    def test_compact_gt_labels_map_canonical_netflow_nodes(self) -> None:
        labels, summary = _phase3g_compact_gt_labels_for_eval(
            abnormal_db_node_ids={100, 101, 200, 999},
            idx_to_db_node_id={7: 9001, 8: 200, 9: 300},
            original_to_canonical_netflow={100: 9001, 101: 9001},
            node_id_to_idx={9001: 7, 200: 8, 300: 9},
        )

        self.assertEqual(labels, {7: "malicious", 8: "malicious"})
        self.assertEqual(summary["original_gt_total"], 4)
        self.assertEqual(summary["original_gt_compact_mapped"], 3)
        self.assertEqual(summary["unique_compact_gt_total"], 2)
        self.assertEqual(summary["canonical_netflow_original_gt_count"], 2)
        self.assertEqual(summary["canonical_netflow_unique_compact_count"], 1)

    def test_alert_label_uses_compact_canonical_gt_label(self) -> None:
        label, src_bad, dst_bad = _phase3g_label_for_alert_row(
            {"info_src": "7", "info_dst": "8"},
            idx_to_db_node_id={7: 9001, 8: 3000},
            abnormal_db_node_ids={100},
            compact_node_labels={7: "malicious"},
        )

        self.assertEqual(label, 1)
        self.assertTrue(src_bad)
        self.assertFalse(dst_bad)


if __name__ == "__main__":
    unittest.main()
