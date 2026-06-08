import importlib
import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cs4m.utils.common import ENTITY_TYPES
from cs4m.embeddings.residual import ResidualEmbeddingConfig, build_residual_embedder
from cs4m.semantics import residual_tokens
from scripts.pipeline.config.runtime_config import SlimConfig
from scripts.pipeline.entrypoints.conditional_e4 import main
from scripts.pipeline.features.conditional_context import _phase3g_action_input_dim
from scripts.pipeline.io.event_artifacts import _phase3e_require_artifacts


REPO_ROOT = Path(__file__).resolve().parents[1]


class BestChainCleanupContractTests(unittest.TestCase):
    def test_active_lowrank_module_uses_best_chain_name(self) -> None:
        module = importlib.import_module("cs4m.models.cs4m_lowrank")

        self.assertTrue(hasattr(module, "SSPMLowRankModel"))

    def test_old_active_lowrank_module_path_is_removed(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("cs4m.models.lowrank")

    def test_residual_tokens_keep_token_parsing_without_hash_sketch(self) -> None:
        self.assertTrue(hasattr(residual_tokens, "residual_text_tokens"))
        self.assertFalse(hasattr(residual_tokens, "SemanticSketchSlim"))

    def test_hash_sketch_and_doc2vec_are_not_active_embedding_methods(self) -> None:
        for method in ("hash_sketch", "doc2vec"):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    build_residual_embedder(ResidualEmbeddingConfig(method=method))

    def test_historical_phase3g_head_module_is_not_active(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("cs4m.phase3g." + "action" + "_head")

    def test_e4_default_dry_run_uses_conditional_head_without_legacy_context(self) -> None:
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "DATASET": "CADETS_E3",
                "RUN_ONLY": "E4_NONE",
                "STAGE": "infer_ablation",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        env.pop("SSPM_SCORE_HEAD", None)
        env.pop("PHASE3E_X_CONTEXT_MEMMAP_ENABLED", None)

        result = subprocess.run(
            ["bash", "scripts/run/run_cadets_e3_phase3e_node_action_semantic_full.sh"],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        output = result.stdout
        self.assertIn("--sspm_score_head conditional_action_semantic", output)
        self.assertIn("--action_head_checkpoint_path", output)
        self.assertIn("--sspm_state_model s4d_complex_node", output)
        self.assertNotIn("--x" + "_context_cache_dir", output)
        self.assertNotIn("--x" + "_context_memmap_enabled", output)
        self.assertNotIn("--sspm_context_action_mode raw" + "_orthrus10", output)

    def test_phase3g_conditional_context_is_136_dim_for_latent64(self) -> None:
        node_embeddings = np.zeros((2, 64), dtype=np.float32)

        self.assertEqual(136, _phase3g_action_input_dim(node_embeddings))
        self.assertEqual(136, 2 * 64 + 2 * len(ENTITY_TYPES))

    def test_phase3g_artifact_requirement_excludes_legacy_context_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            node_dir = root / "nodes"
            action_dir = root / "actions"
            event_dir = root / "events"
            legacy_context_dir = root / ("x" + "_context")
            for directory in (node_dir, action_dir, event_dir, legacy_context_dir):
                directory.mkdir(parents=True)
            for path in (
                node_dir / "node_embeddings.npy",
                node_dir / "node_embedding_meta.json",
                action_dir / "action_embeddings.npy",
                action_dir / "action_embedding_meta.json",
                event_dir / "event_index_train.memmap",
                event_dir / "event_index_validation.memmap",
                event_dir / "event_index_test.memmap",
                event_dir / "event_index_meta.json",
            ):
                path.write_bytes(b"{}" if path.suffix == ".json" else b"")

            config = SlimConfig(
                node_embedding_cache_dir=str(node_dir),
                action_embedding_cache_dir=str(action_dir),
                event_index_cache_dir=str(event_dir),
            )
            paths = _phase3e_require_artifacts(config)

            self.assertIn("event_index_train", paths)
            self.assertFalse((legacy_context_dir / ("x" + "_context_meta.json")).exists())

    def test_active_cli_rejects_old_phase3e_precompute_or_lowrank_head_paths(self) -> None:
        rejected_argvs = (
            ["--phase3e" + "_precompute_only"],
            ["--sspm_score_head", "semantic" + "_residual", "--sspm_train_mode", "load_and_infer"],
            ["--sspm_train_mode", "train" + "_and_save"],
        )

        for argv in rejected_argvs:
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises((SystemExit, ValueError)):
                        main(argv)

    def test_active_cli_uses_clear_module_entrypoint(self) -> None:
        legacy_cli_path = REPO_ROOT / "legacy" / "tools" / "causal_semantics_slim.py"
        active_cli_path = (
            REPO_ROOT / "scripts" / "pipeline" / "entrypoints" / "conditional_e4.py"
        )

        self.assertFalse((REPO_ROOT / "scripts" / "tools" / "causal_semantics_slim.py").exists())
        self.assertFalse((REPO_ROOT / "scripts" / "pipeline" / "cli.py").exists())
        self.assertFalse(
            (
                REPO_ROOT
                / "scripts"
                / "pipeline"
                / ("causal" + "_semantics" + "_runtime.py")
            ).exists(),
        )
        self.assertTrue(legacy_cli_path.exists())

        line_count = len(active_cli_path.read_text(encoding="utf-8").splitlines())

        self.assertLess(line_count, 120)

    def test_pipeline_split_modules_are_importable_by_responsibility(self) -> None:
        expected_modules = {
            "scripts.pipeline.entrypoints.conditional_e4": "main",
            "scripts.pipeline.entrypoints.arguments": "config_from_args",
            "scripts.pipeline.config.runtime_config": "SlimConfig",
            "scripts.pipeline.checks.preflight": "stream_dataset_rows_slim",
            "scripts.pipeline.state.online_state_runtime": (
                "load_sspm_checkpoint_state_for_conditional"
            ),
            "scripts.pipeline.features.conditional_context": "_phase3g_action_input_dim",
            "scripts.pipeline.features.semantic_features": "residual_text",
            "scripts.pipeline.io.db_stream": "_stream_events",
            "scripts.pipeline.io.conditional_cache": "_phase3g_conditional_memmap_paths",
            "scripts.pipeline.io.event_artifacts": "_phase3e_require_artifacts",
            "scripts.pipeline.conditional.train": "run_phase3g_conditional_train_from_precompute",
            "scripts.pipeline.conditional.infer": (
                "run_phase3g_conditional_load_and_infer_from_precompute"
            ),
            "scripts.pipeline.outputs.alert_output": "_write_outputs",
            "scripts.pipeline.outputs.evaluation": "target_for_dataset",
            "scripts.pipeline.outputs.metrics_summary": "_compact_print_summary",
        }

        for module_name, expected_attr in expected_modules.items():
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)

                self.assertTrue(hasattr(module, expected_attr))

    def test_active_pipeline_is_organized_into_subpackages(self) -> None:
        pipeline_root = REPO_ROOT / "scripts" / "pipeline"
        root_python_files = {path.name for path in pipeline_root.glob("*.py")}

        self.assertEqual({"__init__.py"}, root_python_files)

        expected_packages = {
            "checks",
            "conditional",
            "config",
            "entrypoints",
            "features",
            "io",
            "outputs",
            "state",
        }
        for package_name in expected_packages:
            with self.subTest(package=package_name):
                self.assertTrue((pipeline_root / package_name / "__init__.py").exists())
        self.assertFalse((pipeline_root / "compatibility").exists())

        active_modules = [
            "checks/preflight.py",
            "conditional/infer.py",
            "conditional/train.py",
            "entrypoints/arguments.py",
            "features/conditional_context.py",
            "features/semantic_features.py",
            "io/conditional_cache.py",
            "io/event_artifacts.py",
            "outputs/alert_output.py",
            "outputs/conditional_reports.py",
            "outputs/metrics_summary.py",
            "state/online_state_runtime.py",
        ]
        for module_name in active_modules:
            with self.subTest(module=module_name):
                source = (pipeline_root / module_name).read_text(
                    encoding="utf-8",
                )

                self.assertNotIn(
                    "from scripts.pipeline.compatibility.runtime_exports import",
                    source,
                )
                self.assertNotIn("scripts.pipeline.compatibility", source)
                self.assertGreater(len(source.splitlines()), 30)

    def test_scripts_pipeline_chinese_explanation_doc_exists(self) -> None:
        doc_path = REPO_ROOT / "docs" / "SCRIPTS_PIPELINE_EXPLANATION_ZH.md"

        self.assertTrue(doc_path.exists())
        text = doc_path.read_text(encoding="utf-8")

        self.assertIn("scripts/pipeline/entrypoints/conditional_e4.py", text)
        self.assertIn("CADETS_E3 / THEIA_E3", text)
        self.assertIn("136", text)

    def test_phase3e_legacy_context_and_head_training_are_legacy_only(self) -> None:
        self.assertFalse((REPO_ROOT / "cs4m" / "phase3e" / "context_memmap.py").exists())
        self.assertFalse((REPO_ROOT / "cs4m" / "phase3e" / "head_training.py").exists())

        context_module = importlib.import_module("legacy.compatibility.phase3e_context_memmap")
        head_module = importlib.import_module("legacy.compatibility.phase3e_head_training")

        self.assertTrue(hasattr(context_module, "build_x_context_memmap"))
        self.assertTrue(hasattr(head_module, "train_lowrank_head_torch"))

    def test_non_best_runners_and_tools_are_outside_active_scripts(self) -> None:
        inactive_runner = REPO_ROOT / "scripts" / "run" / (
            "run_phase3g_cadets_e3_train_dual_head_v2_full.sh"
        )
        self.assertFalse(inactive_runner.exists())
        self.assertTrue(
            (
                REPO_ROOT
                / "legacy"
                / "runners"
                / "run_phase3g_cadets_e3_train_dual_head_v2_full.sh"
            ).exists(),
        )

        inactive_tools = {
            "smoke_raw_detail_tokens.py",
            "smoke_clearscope_refined_tokenizer_path.py",
            "smoke_clearscope_phase3g_refined_token_path.py",
            "train_residual_word2vec_models.py",
            "summarize_residual_ablation.py",
            "summarize_causal_semantics_slim_ablation.py",
        }
        active_tool_names = {
            path.name for path in (REPO_ROOT / "scripts" / "tools").glob("*.py")
        }

        self.assertFalse(inactive_tools.intersection(active_tool_names))


if __name__ == "__main__":
    unittest.main()
