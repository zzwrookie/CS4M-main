import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.pipeline.config.runtime_config import SlimConfig
from scripts.pipeline.features.conditional_context import _phase3g_conditional_fingerprint


class Phase3GConditionalValidationFingerprintTests(unittest.TestCase):
    def test_out_tag_does_not_change_validation_cache_fingerprint(self):
        config_a = SlimConfig(dataset="CADETS_E3", out_tag="RUN_A")
        config_b = SlimConfig(dataset="CADETS_E3", out_tag="RUN_B")
        head = SimpleNamespace(
            config=SimpleNamespace(
                target_case_head_mode="dual",
                input_dim=64,
                rank=8,
                output_dim=64,
            ),
            fingerprint=lambda: {
                "fingerprint_sha256": "head",
                "schema": "phase3g_conditional_head_v2",
            },
        )
        with patch(
            "scripts.pipeline.features.conditional_context._phase3e_event_index_array",
            side_effect=lambda array: array,
        ), patch(
            "scripts.pipeline.features.conditional_context.event_index_fingerprint",
            return_value={"fingerprint_sha256": "event"},
        ), patch(
            "scripts.pipeline.features.conditional_context._phase3g_file_fingerprint",
            return_value={"fingerprint_sha256": "file"},
        ):
            fingerprint_a = _phase3g_conditional_fingerprint(
                config=config_a,
                head=head,
                paths={
                    "node_embeddings": Path("node.npy"),
                    "action_embeddings": Path("action.npy"),
                },
                event_meta={"splits": {"validation": {"fingerprint": {"x": 1}}}},
                event_index=object(),
                split="validation",
            )
            fingerprint_b = _phase3g_conditional_fingerprint(
                config=config_b,
                head=head,
                paths={
                    "node_embeddings": Path("node.npy"),
                    "action_embeddings": Path("action.npy"),
                },
                event_meta={"splits": {"validation": {"fingerprint": {"x": 1}}}},
                event_index=object(),
                split="validation",
            )

        self.assertEqual(fingerprint_a, fingerprint_b)
        self.assertNotIn("run", fingerprint_a)
        self.assertNotIn("run_only", fingerprint_a)
        self.assertNotIn("RUN_ONLY", fingerprint_a)


if __name__ == "__main__":
    unittest.main()
