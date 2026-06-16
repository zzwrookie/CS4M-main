import unittest

from scripts.pipeline.io.cache_payloads import _phase3g_load_abnormal_db_nodes_for_eval
from scripts.pipeline.checks.preflight import _slim_split_override_days
from scripts.tools.backfill_optc_v13c_canonical_eval import build_optc_v13c_config


class OptcV13cBackfillToolTests(unittest.TestCase):
    def test_build_config_points_to_v13c_full_artifacts(self) -> None:
        config = build_optc_v13c_config("OPTC_201")

        self.assertEqual(config.dataset, "OPTC_201")
        self.assertEqual(config.semantic_mode, "optc_windows_v1_3c_detail")
        self.assertEqual(
            config.node_embedding_cache_dir,
            "outputs/cache/phase3e_optc_windows_v1_3c/"
            "OPTC_201_OPTC_WINDOWS_V1_3C_DETAIL_full/node",
        )
        self.assertEqual(
            config.action_embedding_cache_dir,
            "outputs/cache/phase3e_optc_windows_v1_3c/"
            "OPTC_201_OPTC_WINDOWS_V1_3C_DETAIL_full/action",
        )
        self.assertEqual(
            config.event_index_cache_dir,
            "outputs/cache/phase3e_optc_windows_v1_3c/"
            "OPTC_201_OPTC_WINDOWS_V1_3C_DETAIL_full/event_index",
        )
        self.assertEqual(
            config.result_root,
            "outputs/results/tflr_light/optc_windows_v1_3c_full",
        )
        self.assertEqual(config.out_tag, "OPTC_201_OPTC_WINDOWS_V1_3C_DETAIL_FULL_INFER")
        self.assertEqual(config.optc_netflow_node_canonicalization, "remote_endpoint_v1_3")

    def test_optc_local_ground_truth_loader_reads_h201_nodes(self) -> None:
        config = build_optc_v13c_config("OPTC_201")

        nodes = _phase3g_load_abnormal_db_nodes_for_eval(config)

        self.assertEqual(len(nodes), 2905)
        self.assertIn(1622016, nodes)

    def test_optc_split_override_uses_new_unified_protocol(self) -> None:
        self.assertEqual(
            _slim_split_override_days("OPTC_201"),
            {"train": [19, 20, 21], "val": [22], "test": [23, 24, 25]},
        )


if __name__ == "__main__":
    unittest.main()
