"""Tests for ClearScope v31 FP/TP token audit helpers."""

from __future__ import annotations

import unittest

from tmp.clearscope_v31_fp_tp_token_audit import (
    aggregate_token_rows,
    event_label,
    raw_node_id,
    token_oov_summary,
)


class ClearScopeV31FPTPTokenAuditTests(unittest.TestCase):
    """Validate token-level TP/FP aggregation helpers."""

    def test_aggregate_token_rows_counts_tp_fp_and_examples(self) -> None:
        rows = [
            {
                "token_tuple": "file fennec cache",
                "event_label": "malicious",
                "score": 0.9,
                "example": "/cache/a",
            },
            {
                "token_tuple": "file fennec cache",
                "event_label": "benign",
                "score": 0.8,
                "example": "/cache/b",
            },
        ]

        [result] = aggregate_token_rows(rows)

        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["event_count"], 2)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertIn("/cache/a", result["example_paths"])

    def test_token_oov_summary_splits_known_and_oov_tokens(self) -> None:
        summary = token_oov_summary("file fennec unknown", {"file", "fennec"})

        self.assertEqual(summary["known_tokens"], "file fennec")
        self.assertEqual(summary["oov_tokens"], "unknown")
        self.assertAlmostEqual(summary["oov_ratio"], 1.0 / 3.0)

    def test_raw_node_id_maps_embedding_index_to_database_id(self) -> None:
        self.assertEqual(raw_node_id(7, {7: 42}), 42)
        self.assertEqual(raw_node_id(9, {}), 9)

    def test_event_label_uses_post_eval_node_coverage_labels(self) -> None:
        labels = {10: "malicious"}

        self.assertEqual(event_label({"src_idx": "10", "dst_idx": "11"}, labels), "malicious")
        self.assertEqual(event_label({"src_idx": "12", "dst_idx": "11"}, labels), "benign")


if __name__ == "__main__":
    unittest.main()
