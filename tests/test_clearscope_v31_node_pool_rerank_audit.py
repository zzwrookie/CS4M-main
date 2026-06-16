"""Tests for ClearScope v31 post-inference node pool reranking helpers."""

from __future__ import annotations

import unittest

from tmp.clearscope_v31_node_pool_rerank_audit import (
    NodeSupport,
    compute_support_score,
    rerank_node_rows,
    topk_metrics,
)


class ClearScopeV31NodePoolRerankAuditTests(unittest.TestCase):
    """Validate runtime-derived node support scoring."""

    def test_file_bidirectional_support_lifts_multi_event_node(self) -> None:
        weak = NodeSupport(
            residual_max=0.24,
            alert_event_count=1,
            process_process_only=True,
        )
        supported = NodeSupport(
            residual_max=0.20,
            alert_event_count=8,
            file_cache_read=True,
            file_cache_write=True,
            process_file_bidirectional=True,
        )

        self.assertGreater(compute_support_score(supported), compute_support_score(weak))

    def test_process_process_only_penalty_lowers_same_residual_node(self) -> None:
        base = NodeSupport(residual_max=0.3, alert_event_count=4)
        process_only = NodeSupport(
            residual_max=0.3,
            alert_event_count=4,
            process_process_only=True,
        )

        self.assertLess(compute_support_score(process_only), compute_support_score(base))

    def test_rerank_rows_orders_by_support_score(self) -> None:
        rows = [
            {
                "node_id": "10",
                "node_score": "0.4",
                "residual_max": "0.4",
                "candidate_event_count": "1",
            },
            {
                "node_id": "20",
                "node_score": "0.2",
                "residual_max": "0.2",
                "candidate_event_count": "8",
            },
        ]
        supports = {
            10: NodeSupport(residual_max=0.4, alert_event_count=1, process_process_only=True),
            20: NodeSupport(
                residual_max=0.2,
                alert_event_count=8,
                file_cache_read=True,
                file_cache_write=True,
                process_file_bidirectional=True,
            ),
        }

        reranked = rerank_node_rows(rows, supports)

        self.assertEqual([row["node_id"] for row in reranked], [20, 10])
        self.assertEqual([row["rerank_rank"] for row in reranked], [1, 2])

    def test_topk_metrics_can_use_pipeline_node_labels(self) -> None:
        rows = [
            {"node_id": 10, "node_label": "benign", "is_correct_node_alert": False},
            {"node_id": 20, "node_label": "malicious", "is_correct_node_alert": True},
        ]

        metrics = topk_metrics(rows, set(), malicious_total=41)

        self.assertEqual(metrics["malicious_nodes_total"], 41)
        self.assertEqual(metrics["covered_malicious_count"], 1)
        self.assertEqual(metrics["top100_tp"], 1)


if __name__ == "__main__":
    unittest.main()
