import unittest

from cs4m.semantics.cadets_freebsd import (
    CADETS_SEMANTIC_RULES_VERSION,
    CADETS_V3_SAFE_LEXICAL_SEMANTIC_MODE,
    cadets_semantic_mode_is_v3_safe_lexical,
    freebsd_file_natural_tokens,
    freebsd_file_natural_tokens_v3_safe_lexical,
    freebsd_netflow_natural_tokens,
    freebsd_netflow_natural_tokens_v3_safe_lexical,
    freebsd_process_natural_tokens_v3_safe_lexical,
)
from scripts.pipeline.config.runtime_config import SlimConfig
from scripts.pipeline.features.semantic_features import residual_text
from scripts.pipeline.io.cache_payloads import model_fingerprint_payload
from scripts.pipeline.io.event_artifacts import _phase3e_node_tokens_from_db_node


class CadetsFreeBsdV3SafeLexicalTests(unittest.TestCase):
    def test_v3_is_explicit_mode_and_v2_constant_stays_default(self):
        self.assertEqual(CADETS_SEMANTIC_RULES_VERSION, "cadets_freebsd_raw_detail_v2")
        self.assertTrue(
            cadets_semantic_mode_is_v3_safe_lexical(
                CADETS_V3_SAFE_LEXICAL_SEMANTIC_MODE
            )
        )

    def test_v3_keeps_missing_file_as_fixed_file_bucket(self):
        self.assertEqual(freebsd_file_natural_tokens(""), ("file",))
        self.assertEqual(freebsd_file_natural_tokens_v3_safe_lexical(""), ("file",))

    def test_v3_refines_process_other_without_gt_strings(self):
        self.assertEqual(
            freebsd_process_natural_tokens_v3_safe_lexical("/tmp/abc123x"),
            ("process", "process_other_payload_shape", "abc123x"),
        )

    def test_v3_keeps_home_lock_benign_and_splits_home_web_artifact(self):
        self.assertEqual(
            freebsd_file_natural_tokens_v3_safe_lexical("/home/george/Sent.lock"),
            ("file", "user_home_file", "sent_lock"),
        )
        self.assertEqual(
            freebsd_file_natural_tokens_v3_safe_lexical(
                "/usr/home/user/eraseme/index.html"
            ),
            ("file", "user_home_web_artifact_file", "eraseme_index_html"),
        )

    def test_v3_netflow_matches_v2(self):
        self.assertEqual(
            freebsd_netflow_natural_tokens_v3_safe_lexical(
                "8.8.8.8",
                "128.55.12.10",
            ),
            freebsd_netflow_natural_tokens("8.8.8.8", "128.55.12.10"),
        )

    def test_residual_text_uses_v3_only_for_explicit_semantic_mode(self):
        row = {
            "src_kind": "process",
            "src_process_cmd": "/tmp/abc123x",
            "dst_kind": "file",
            "dst_file_path": "/usr/home/user/eraseme/index.html",
            "object_type": "file",
            "action": "EVENT_WRITE",
        }

        current = residual_text(row, dataset="CADETS_E3")
        candidate = residual_text(
            row,
            dataset="CADETS_E3",
            semantic_mode=CADETS_V3_SAFE_LEXICAL_SEMANTIC_MODE,
        )

        self.assertIn("process_other", current)
        self.assertNotIn("process_other_payload_shape", current)
        self.assertIn("process_other_payload_shape", candidate)
        self.assertIn("user_home_web_artifact_file", candidate)

    def test_phase3e_node_token_routing_uses_v3_only_when_explicit(self):
        node_maps = {
            "process_meta": {10: {"cmd": "/tmp/abc123x"}},
            "file_meta": {11: {"path": ""}},
        }
        current_cfg = SlimConfig(dataset="CADETS_E3")
        v3_cfg = SlimConfig(
            dataset="CADETS_E3",
            semantic_mode=CADETS_V3_SAFE_LEXICAL_SEMANTIC_MODE,
        )

        self.assertEqual(
            _phase3e_node_tokens_from_db_node(
                node_id=10,
                node_kind="process",
                node_summary="",
                node_maps=node_maps,
                config=current_cfg,
                process_cfg=None,
            ),
            ("process", "process_other", "abc123x"),
        )
        self.assertEqual(
            _phase3e_node_tokens_from_db_node(
                node_id=10,
                node_kind="process",
                node_summary="",
                node_maps=node_maps,
                config=v3_cfg,
                process_cfg=None,
            ),
            ("process", "process_other_payload_shape", "abc123x"),
        )
        self.assertEqual(
            _phase3e_node_tokens_from_db_node(
                node_id=11,
                node_kind="file",
                node_summary="",
                node_maps=node_maps,
                config=v3_cfg,
                process_cfg=None,
            ),
            ("file",),
        )

    def test_cache_payload_records_selected_cadets_semantic_mode(self):
        cfg = SlimConfig(
            dataset="CADETS_E3",
            semantic_mode=CADETS_V3_SAFE_LEXICAL_SEMANTIC_MODE,
        )
        payload = model_fingerprint_payload(cfg)

        self.assertEqual(
            payload["cadets_semantic_rules_version"],
            CADETS_V3_SAFE_LEXICAL_SEMANTIC_MODE,
        )


if __name__ == "__main__":
    unittest.main()
