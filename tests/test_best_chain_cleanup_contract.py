import importlib
import unittest

from cs4m.embeddings.residual import ResidualEmbeddingConfig, build_residual_embedder
from cs4m.semantics import residual_tokens


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

    def test_historical_action_head_is_not_active_phase3g_module(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("cs4m.phase3g.action_head")


if __name__ == "__main__":
    unittest.main()
