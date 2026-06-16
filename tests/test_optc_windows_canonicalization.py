"""Tests for OpTC Windows netflow node canonicalization helpers."""

from __future__ import annotations

import unittest

from cs4m.semantics.optc_windows import OPTC_NETFLOW_CANONICALIZATION_V1_3
from scripts.pipeline.io.event_artifacts import (
    _phase3e_build_optc_canonical_artifact_inputs,
    _phase3e_fetch_used_node_lookup_joined,
    _phase3e_optc_canonical_node_id,
)


class OptcWindowsCanonicalizationTests(unittest.TestCase):
    """Validate OpTC-only netflow canonical node identity behavior."""

    def test_process_and_file_nodes_keep_original_ids(self) -> None:
        self.assertEqual(
            _phase3e_optc_canonical_node_id(
                42,
                "process",
                {},
                OPTC_NETFLOW_CANONICALIZATION_V1_3,
            ),
            42,
        )
        self.assertEqual(
            _phase3e_optc_canonical_node_id(
                43,
                "file",
                {},
                OPTC_NETFLOW_CANONICALIZATION_V1_3,
            ),
            43,
        )

    def test_disabled_canonicalization_keeps_netflow_original_id(self) -> None:
        node_maps = {
            "netflow_meta": {
                10: {"src_addr": "10.0.0.5", "dst_addr": "142.20.1.7", "dst_port": "443"},
            },
        }

        self.assertEqual(_phase3e_optc_canonical_node_id(10, "netflow", node_maps, "none"), 10)

    def test_joined_lookup_uses_temp_table_and_returns_legacy_shape(self) -> None:
        class Cursor:
            def __init__(self) -> None:
                self.statements = []
                self._rows = []

            def execute(self, sql, params=None) -> None:
                self.statements.append(str(sql))
                lowered = str(sql).lower()
                if "from subject_node_table" in lowered:
                    self._rows = [(1, "cmd.exe", "cmd.exe /c whoami")]
                elif "from file_node_table" in lowered:
                    self._rows = [(2, "C:\\Users\\Public\\payload.exe")]
                elif "from netflow_node_table" in lowered:
                    self._rows = [(3, "10.0.0.5", "49152", "142.20.1.7", "443")]
                else:
                    self._rows = []

            def fetchall(self):
                return list(self._rows)

            def executemany(self, sql, rows) -> None:
                self.statements.append(str(sql))

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        class Connection:
            def __init__(self) -> None:
                self.cursor_obj = Cursor()
                self.inserted_rows = []

            def cursor(self):
                return self.cursor_obj

        conn = Connection()

        indexid2summary, meta_by_kind, elapsed = _phase3e_fetch_used_node_lookup_joined(
            conn=conn,
            used_node_ids={1, 2, 3},
        )

        sql_text = "\n".join(conn.cursor_obj.statements).lower()
        self.assertIn("create temp table temp_phase3e_used_node_ids", sql_text)
        self.assertIn("join temp_phase3e_used_node_ids u on u.index_id = n.index_id", sql_text)
        self.assertEqual(indexid2summary[1], ("process", "cmd.exe cmd.exe /c whoami"))
        self.assertEqual(indexid2summary[2], ("file", "C:\\Users\\Public\\payload.exe"))
        self.assertEqual(indexid2summary[3], ("netflow", "10.0.0.5 49152 142.20.1.7 443"))
        self.assertEqual(meta_by_kind["netflow_meta"][3]["dst_port"], "443")
        self.assertGreaterEqual(elapsed, 0.0)

    def test_joined_lookup_empty_used_nodes_returns_empty_shape(self) -> None:
        class Connection:
            def cursor(self):
                raise AssertionError("empty lookup should not open a cursor")

        indexid2summary, meta_by_kind, elapsed = _phase3e_fetch_used_node_lookup_joined(
            conn=Connection(),
            used_node_ids=set(),
        )

        self.assertEqual(indexid2summary, {})
        self.assertEqual(meta_by_kind["process_meta"], {})
        self.assertEqual(meta_by_kind["file_meta"], {})
        self.assertEqual(meta_by_kind["netflow_meta"], {})
        self.assertEqual(elapsed, 0.0)

    def test_v13_bidirectional_flows_share_remote_endpoint_canonical_id(self) -> None:
        node_maps = {
            "netflow_meta": {
                10: {
                    "src_addr": "142.20.57.246",
                    "src_port": "53886",
                    "dst_addr": "202.6.172.98",
                    "dst_port": "443",
                },
                11: {
                    "src_addr": "202.6.172.98",
                    "src_port": "443",
                    "dst_addr": "142.20.57.246",
                    "dst_port": "53374",
                },
            },
        }

        first = _phase3e_optc_canonical_node_id(
            10,
            "netflow",
            node_maps,
            OPTC_NETFLOW_CANONICALIZATION_V1_3,
        )
        second = _phase3e_optc_canonical_node_id(
            11,
            "netflow",
            node_maps,
            OPTC_NETFLOW_CANONICALIZATION_V1_3,
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, 10)

    def test_v13_different_remote_ips_do_not_share_canonical_id(self) -> None:
        node_maps = {
            "netflow_meta": {
                10: {
                    "src_addr": "142.20.57.246",
                    "src_port": "53886",
                    "dst_addr": "202.6.172.98",
                    "dst_port": "443",
                },
                11: {
                    "src_addr": "142.20.57.246",
                    "src_port": "53887",
                    "dst_addr": "203.0.113.99",
                    "dst_port": "443",
                },
            },
        }

        self.assertNotEqual(
            _phase3e_optc_canonical_node_id(
                10,
                "netflow",
                node_maps,
                OPTC_NETFLOW_CANONICALIZATION_V1_3,
            ),
            _phase3e_optc_canonical_node_id(
                11,
                "netflow",
                node_maps,
                OPTC_NETFLOW_CANONICALIZATION_V1_3,
            ),
        )

    def test_v13_artifact_inputs_report_hybrid_identity(self) -> None:
        class Config:
            dataset = "OPTC_501"
            semantic_mode = "optc_windows_v1_3_detail"
            max_tokens_per_node = 8
            optc_netflow_node_canonicalization = OPTC_NETFLOW_CANONICALIZATION_V1_3
            theia_netflow_policy = "scope_port"
            progress_interval_events = 100000

        inputs = _phase3e_build_optc_canonical_artifact_inputs(
            used_node_ids={1, 10, 11},
            split_used_nodes={"train": {1, 10, 11}, "validation": set(), "test": set()},
            indexid2summary={
                1: ("process", "cmd.exe"),
                10: ("netflow", "142.20.57.246 53886 202.6.172.98 443"),
                11: ("netflow", "202.6.172.98 443 142.20.57.246 53374"),
            },
            meta_by_kind={
                "process_meta": {1: {"path": "cmd.exe", "cmd": "cmd.exe"}},
                "netflow_meta": {
                    10: {
                        "src_addr": "142.20.57.246",
                        "src_port": "53886",
                        "dst_addr": "202.6.172.98",
                        "dst_port": "443",
                    },
                    11: {
                        "src_addr": "202.6.172.98",
                        "src_port": "443",
                        "dst_addr": "142.20.57.246",
                        "dst_port": "53374",
                    },
                },
            },
            config=Config(),
            process_cfg=None,
        )

        self.assertEqual(inputs["summary"]["hybrid_identity_mode"], "canonical_state_original_eval")
        self.assertEqual(inputs["summary"]["canonical_netflow_node_count"], 1)
        self.assertIn(10, inputs["original_to_canonical"])
        self.assertIn(11, inputs["original_to_canonical"])


if __name__ == "__main__":
    unittest.main()
