import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import psycopg2

from scripts.pipeline.io import cache_payloads


class ClearScopeE5GtLoaderTests(unittest.TestCase):
    def test_clearscope_e5_does_not_load_e3_pickle(self) -> None:
        cfg = SimpleNamespace(dataset="CLEARSCOPE_E5")

        def fake_exists(self: Path) -> bool:
            return str(self) == "ground_truth/E3-CLEARSCOPE/abnormal_nodes.pkl"

        with mock.patch.object(Path, "exists", fake_exists), mock.patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError("read E3 GT"),
        ), mock.patch.object(cache_payloads, "_cfg_for_dataset") as cfg_for_dataset, mock.patch.object(
            cache_payloads,
            "init_database_connection",
            side_effect=psycopg2.Error("db unavailable"),
        ):
            cfg_for_dataset.return_value = SimpleNamespace(
                database=SimpleNamespace(host="", user="", password="", port=0),
            )

            result = cache_payloads._phase3g_load_abnormal_db_nodes_for_eval(cfg)

        self.assertEqual(result, set())

    def test_clearscope_e3_keeps_e3_pickle_fast_path(self) -> None:
        cfg = SimpleNamespace(dataset="CLEARSCOPE_E3")
        payload = b"\x80\x04]\x94(K\x01K\x02e."

        def fake_exists(self: Path) -> bool:
            return str(self) == "ground_truth/E3-CLEARSCOPE/abnormal_nodes.pkl"

        with mock.patch.object(Path, "exists", fake_exists), mock.patch.object(
            Path,
            "read_bytes",
            return_value=payload,
        ):
            result = cache_payloads._phase3g_load_abnormal_db_nodes_for_eval(cfg)

        self.assertEqual(result, {1, 2})


if __name__ == "__main__":
    unittest.main()
