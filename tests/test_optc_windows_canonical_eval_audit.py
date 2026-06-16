"""Tests for OpTC canonical-aware evaluation audit helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.tools.audit_optc_windows_canonical_eval import (
    GtNode,
    build_canonical_gt_mapping,
    summarize_gt_hits,
    write_counter_csv,
)


class OptcWindowsCanonicalEvalAuditTests(unittest.TestCase):
    """Validate canonical GT mapping and hit summaries."""

    def test_maps_netflow_gt_original_id_to_canonical_compact_idx(self) -> None:
        gt_nodes = [
            GtNode("uuid-a", "netflow", 100, "{'netflow': '1.1.1.1:443->142.20.1.2:1'}"),
            GtNode("uuid-b", "netflow", 101, "{'netflow': '142.20.1.2:1->1.1.1.1:443'}"),
            GtNode("uuid-c", "process", 200, "{'subject': 'cmd.exe'}"),
        ]
        original_to_canonical = {100: 9001, 101: 9001}
        node_id_to_idx = {9001: 7, 200: 8}

        mapped = build_canonical_gt_mapping(gt_nodes, original_to_canonical, node_id_to_idx)

        self.assertEqual([row.compact_node_idx for row in mapped], [7, 7, 8])
        self.assertEqual([row.lookup_node_id for row in mapped], [9001, 9001, 200])
        self.assertEqual([row.mapping_status for row in mapped], ["mapped", "mapped", "mapped"])

    def test_summarizes_hits_by_original_kind(self) -> None:
        gt_nodes = [
            GtNode("uuid-a", "netflow", 100, "{'netflow': '1.1.1.1:443->142.20.1.2:1'}"),
            GtNode("uuid-b", "netflow", 101, "{'netflow': '142.20.1.2:1->1.1.1.1:443'}"),
            GtNode("uuid-c", "process", 200, "{'subject': 'cmd.exe'}"),
            GtNode("uuid-d", "file", 300, "{'file': 'x.dll'}"),
        ]
        mapped = build_canonical_gt_mapping(
            gt_nodes,
            {100: 9001, 101: 9001},
            {9001: 7, 200: 8, 300: 9},
        )

        summary, hit_rows, missed_rows = summarize_gt_hits(mapped, {7, 8})

        self.assertEqual(summary["original_gt_total"], 4)
        self.assertEqual(summary["original_gt_hit"], 3)
        self.assertEqual(summary["unique_compact_gt_total"], 3)
        self.assertEqual(summary["unique_compact_gt_hit"], 2)
        self.assertEqual(summary["by_kind"]["netflow"]["hit"], 2)
        self.assertEqual(summary["by_kind"]["process"]["hit"], 1)
        self.assertEqual(summary["by_kind"]["file"]["missed"], 1)
        self.assertEqual({row.original_node_id for row in hit_rows}, {100, 101, 200})
        self.assertEqual([row.original_node_id for row in missed_rows], [300])

    def test_write_counter_csv_includes_zero_total_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "counter.csv"
            write_counter_csv(path, ["kind"], {})
            self.assertEqual(path.read_text(encoding="utf-8"), "kind,count,ratio\n")


if __name__ == "__main__":
    unittest.main()
