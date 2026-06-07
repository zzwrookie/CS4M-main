import tempfile
import unittest
import json
from pathlib import Path

from cs4m.config.config import PROJECT_ROOT, get_runtime_required_args, get_yml_cfg
from cs4m.phase3g.compact_node_embeddings import load_compact_used_node_artifacts
from scripts.tools import causal_semantics_slim as slim


class ConfigPackageTests(unittest.TestCase):
    def test_runtime_yaml_loads_from_cs4m_config_package(self) -> None:
        args = get_runtime_required_args(args=["THEIA_E3"])

        cfg = get_yml_cfg(args)

        self.assertEqual("THEIA_E3", cfg.dataset.name)
        self.assertTrue(cfg.runtime.use_all_db_files)
        self.assertTrue(Path(PROJECT_ROOT, "cs4m", "config", "clad.yml").is_file())


class Phase3GConditionalMemmapOnlyTests(unittest.TestCase):
    def test_compact_embedding_cache_accepts_relocated_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "compact_embedding_meta.json").write_text(
                json.dumps(
                    {
                        "source_node_embedding_fingerprint": {
                            "path": "/old/repo/outputs/cache/node_embeddings.npy",
                            "num_bytes": 128,
                            "sha256": "abc123",
                        },
                        "source_event_index_fingerprints": {},
                    },
                )
                + "\n",
                encoding="utf-8",
            )

            paths = load_compact_used_node_artifacts(
                root,
                expected_source_node_embedding_fingerprint={
                    "path": "/new/repo/outputs/cache/node_embeddings.npy",
                    "num_bytes": 128,
                    "sha256": "abc123",
                },
                expected_source_event_index_fingerprints={},
            )

        self.assertEqual(root / "compact_embedding_meta.json", paths["compact_embedding_meta"])

    def test_word2vec_validation_uses_dataset_specific_semantic_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "THEIA_E3_word2vec.pkl"
            artifact.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "semantic_rules": {
                            "cadets": "cadets_freebsd_raw_detail_v2",
                            "clearscope": "raw_detail_v2_refined",
                            "theia": "theia_linux_raw_detail_v2",
                        },
                    },
                )
                + "\n",
                encoding="utf-8",
            )
            config = slim.SlimConfig(
                dataset="THEIA_E3",
                pretrained_residual_embedder_path=str(artifact),
                semantic_mode="theia_linux_raw_detail_v2",
            )

            slim.validate_word2vec_semantic_mode_matches_config(config)

    def test_cli_flag_sets_memmap_only_mode(self) -> None:
        args = slim.parse_args(
            [
                "--dataset",
                "THEIA_E3",
                "--pretrained_residual_embedder_path",
                "embedder.pkl",
                "--sspm_target_mode",
                "node_action_semantic_mean",
                "--sspm_score_head",
                "conditional_action_semantic",
                "--sspm_train_mode",
                "train_conditional_and_save",
                "--sspm_checkpoint_path",
                "phase3e.pkl",
                "--event_threshold_mode",
                slim.CONDITIONAL_GROUP_THRESHOLD_MODE,
                "--phase3g_build_conditional_memmap_only",
            ],
        )

        config = slim.config_from_args(args)

        self.assertTrue(config.phase3g_build_conditional_memmap_only)
        self.assertEqual("train_conditional_and_save", config.sspm_train_mode)
        self.assertEqual("conditional_action_semantic", config.sspm_score_head)

    def test_memmap_only_payload_marks_no_training_or_labels(self) -> None:
        config = slim.SlimConfig(
            dataset="THEIA_E3",
            out_tag="THEIA_E3_MEMMAP_ONLY",
            sspm_score_head="conditional_action_semantic",
            sspm_conditional_head_arch=slim.CONDITIONAL_HEAD_ARCH_SHARED_V1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = slim._phase3g_conditional_memmap_only_payload(
                config=config,
                output_dir=Path(temp_dir),
                memmap_meta={
                    "x_path": "X_conditional_s4d_complex_node.memmap",
                    "y_path": "Y_conditional_s4d_complex_node.memmap",
                    "target_case_path": "target_case_s4d_complex_node.memmap",
                },
                train_count=12,
                validation_count=3,
                elapsed_seconds=1.25,
            )

        self.assertEqual(12, payload["train_events_actual"])
        self.assertEqual(3, payload["validation_events_actual"])
        self.assertEqual(0, payload["test_events_actual"])
        self.assertEqual("", payload["phase3g"]["checkpoint_path"])
        self.assertFalse(payload["leakage_contract"]["contains_test_labels"])
        self.assertFalse(payload["leakage_contract"]["ground_truth_used"])


if __name__ == "__main__":
    unittest.main()
