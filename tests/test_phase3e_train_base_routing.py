import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from scripts.pipeline.config.runtime_config import SlimConfig
from scripts.pipeline.io import event_artifacts


class Phase3ETrainBaseRoutingTests(unittest.TestCase):
    def test_train_base_does_not_delegate_to_conditional_head_training(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            config = SlimConfig(
                dataset="CLEARSCOPE_E3",
                out_tag="ROUTING_TEST",
                result_root=tmp_dir,
                sspm_target_mode="node_action_semantic_mean",
                sspm_score_head="conditional_action_semantic",
                sspm_train_mode="train_conditional_and_save",
                sspm_train_data_mode="phase3e_memmap",
                sspm_checkpoint_path=str(Path(tmp_dir) / "base.pkl"),
                pretrained_residual_embedder_path="word2vec.pkl",
            )

            with mock.patch.object(
                event_artifacts,
                "run_phase3g_conditional_train_from_precompute",
                side_effect=AssertionError("conditional head path was called"),
            ), mock.patch.object(
                event_artifacts,
                "_phase3e_run_train_base_checkpoint",
                return_value=Path(tmp_dir) / "base_eval.json",
            ) as base_train:
                result = event_artifacts.run_phase3e_train_base_from_precompute(config)

            self.assertEqual(result, Path(tmp_dir) / "base_eval.json")
            base_train.assert_called_once_with(config)


if __name__ == "__main__":
    unittest.main()
