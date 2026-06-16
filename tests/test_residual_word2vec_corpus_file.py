import tempfile
import unittest
from pathlib import Path

from cs4m.embeddings.residual import ResidualEmbeddingConfig, Word2VecResidualEmbedder


class ResidualWord2VecCorpusFileTests(unittest.TestCase):
    def test_large_corpus_training_can_use_temporary_corpus_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ResidualEmbeddingConfig(
                latent_dim=8,
                word2vec_epochs=1,
                word2vec_workers=1,
                word2vec_corpus_file_dir=tmp_dir,
            )
            embedder = Word2VecResidualEmbedder(config)

            embedder.fit([["file", "cache"], ["process", "browser"]])

            self.assertEqual(embedder.stats()["train_sentence_count"], 2)
            self.assertIn("file", embedder.state_dict()["train_vocab"])
            self.assertEqual(list(Path(tmp_dir).glob("residual_word2vec_*.txt")), [])


if __name__ == "__main__":
    unittest.main()
