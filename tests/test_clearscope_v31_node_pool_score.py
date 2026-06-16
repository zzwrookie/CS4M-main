"""Tests for ClearScope v31 support-based node pool scoring."""

from __future__ import annotations

import unittest

from scripts.pipeline.outputs.alert_output import _node_pool_score


class ClearScopeV31NodePoolScoreTests(unittest.TestCase):
    """Validate the runtime-only support score used for v31 node reranking."""

    def test_cache_read_write_scores_higher_than_read_only(self) -> None:
        read_only = {
            "residual_max": 0.09,
            "candidate_event_count": 10,
            "file_cache_read": True,
            "file_cache_write": False,
            "process_file_bidirectional": False,
            "process_process_only": False,
            "netflow_only": False,
        }
        read_write = dict(read_only, file_cache_write=True)

        self.assertGreater(
            _node_pool_score(read_write, "base_conf_v31_support"),
            _node_pool_score(read_only, "base_conf_v31_support"),
        )

    def test_process_process_only_penalty_lowers_score(self) -> None:
        normal = {
            "residual_max": 0.09,
            "candidate_event_count": 10,
            "file_cache_read": False,
            "file_cache_write": False,
            "process_file_bidirectional": False,
            "process_process_only": False,
            "netflow_only": False,
        }
        process_only = dict(normal, process_process_only=True)

        self.assertLess(
            _node_pool_score(process_only, "base_conf_v31_support"),
            _node_pool_score(normal, "base_conf_v31_support"),
        )

    def test_netflow_only_penalty_lowers_score(self) -> None:
        normal = {
            "residual_max": 0.09,
            "candidate_event_count": 10,
            "file_cache_read": False,
            "file_cache_write": False,
            "process_file_bidirectional": False,
            "process_process_only": False,
            "netflow_only": False,
        }
        netflow_only = dict(normal, netflow_only=True)

        self.assertLess(
            _node_pool_score(netflow_only, "base_conf_v31_support"),
            _node_pool_score(normal, "base_conf_v31_support"),
        )


if __name__ == "__main__":
    unittest.main()
