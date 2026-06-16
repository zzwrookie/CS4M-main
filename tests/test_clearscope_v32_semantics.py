"""Tests for ClearScope raw_detail_v32_discriminative semantics."""

from __future__ import annotations

import unittest

from cs4m.semantics.clearscope_android import (
    CLEARSCOPE_V31_SEMANTIC_MODE,
    CLEARSCOPE_V32_SEMANTIC_MODE,
    CLEARSCOPE_V32_CACHE_ONLY_SEMANTIC_MODE,
    android_file_detail_v31,
    android_file_detail_v32,
    clearscope_file_natural_tokens_v32_cache_only,
    clearscope_file_natural_tokens_v32,
    clearscope_residual_text_v32_cache_only,
    clearscope_residual_text_v32,
    normalize_clearscope_semantic_mode,
)


class ClearScopeV32SemanticsTests(unittest.TestCase):
    """Validate v32 tokenization while freezing v31 behavior."""

    def test_normalize_accepts_v32_aliases(self) -> None:
        self.assertEqual(
            normalize_clearscope_semantic_mode("raw_detail_v32_discriminative"),
            CLEARSCOPE_V32_SEMANTIC_MODE,
        )
        self.assertEqual(
            normalize_clearscope_semantic_mode(
                "clearscope_raw_detail_v32_discriminative",
            ),
            CLEARSCOPE_V32_SEMANTIC_MODE,
        )

    def test_normalize_accepts_v32_cache_only_aliases(self) -> None:
        self.assertEqual(
            normalize_clearscope_semantic_mode("raw_detail_v32_cache_only_discriminative"),
            CLEARSCOPE_V32_CACHE_ONLY_SEMANTIC_MODE,
        )
        self.assertEqual(
            normalize_clearscope_semantic_mode(
                "clearscope_raw_detail_v32_cache_only_discriminative",
            ),
            CLEARSCOPE_V32_CACHE_ONLY_SEMANTIC_MODE,
        )

    def test_v31_cache2_entry_detail_remains_unchanged(self) -> None:
        path = (
            "/data/data/org.mozilla.fennec_firefox_dev/cache/nz9885vc.default/"
            "cache2/entries/0A8BF759C7672A43A3452C951B80220C95C1576F"
        )

        self.assertEqual(
            android_file_detail_v31(path, "android_app_cache_file"),
            "fennec_firefox_dev_cache2_entries_hex40_body",
        )
        self.assertEqual(CLEARSCOPE_V31_SEMANTIC_MODE, "raw_detail_v31_discriminative")

    def test_v32_cache2_entry_uses_profile_shape_not_hash(self) -> None:
        path = (
            "/data/data/org.mozilla.fennec_firefox_dev/cache/nz9885vc.default/"
            "cache2/entries/0A8BF759C7672A43A3452C951B80220C95C1576F"
        )

        tokens = clearscope_file_natural_tokens_v32(path)

        self.assertEqual(
            tokens,
            (
                "file",
                "android_app_cache_file",
                "fennec_firefox_dev_cache2_entries_profile_default_hex40",
            ),
        )
        joined = " ".join(tokens)
        self.assertNotIn("0A8BF759C7672A43A3452C951B80220C95C1576F".lower(), joined)
        self.assertNotIn("0a8bf759", joined)

    def test_v32_cache2_entry_other_profile_shape(self) -> None:
        path = (
            "/data/data/org.mozilla.fennec_firefox_dev/cache/profile-release/"
            "cache2/entries/0A8BF759C7672A43A3452C951B80220C95C1576F"
        )

        self.assertEqual(
            android_file_detail_v32(path, "android_app_cache_file"),
            "fennec_firefox_dev_cache2_entries_profile_other_hex40",
        )

    def test_v32_socket_dnsproxyd_adds_event_context(self) -> None:
        row = {
            "action": "EVENT_READ",
            "src_kind": "file",
            "src_file_path": "/dev/socket/dnsproxyd",
            "dst_kind": "process",
            "dst_process_cmd": "org.mozilla.fennec_firefox_dev",
        }

        text = clearscope_residual_text_v32(row)

        self.assertIn("socket_dnsproxyd", text)
        self.assertIn("event_socket_dnsproxyd_to_fennec_firefox_dev", text)

    def test_v32_socket_dnsproxyd_process_to_file_adds_event_context(self) -> None:
        row = {
            "action": "EVENT_READ",
            "src_kind": "process",
            "src_process_cmd": "org.mozilla.fennec_firefox_dev",
            "dst_kind": "file",
            "dst_file_path": "/dev/socket/dnsproxyd",
        }

        text = clearscope_residual_text_v32(row)

        self.assertIn("socket_dnsproxyd", text)
        self.assertIn("event_socket_dnsproxyd_from_fennec_firefox_dev", text)

    def test_v32_cache_only_keeps_cache2_profile_token(self) -> None:
        path = (
            "/data/data/org.mozilla.fennec_firefox_dev/cache/"
            "nz9885vc.default/cache2/entries/"
            "0A8B9C7D6E5F4A3B2C1D0E9F8A7B6C5D4E3F2A1B"
        )

        tokens = clearscope_file_natural_tokens_v32_cache_only(path)

        self.assertEqual(tokens[0], "file")
        self.assertEqual(tokens[1], "android_app_cache_file")
        self.assertEqual(tokens[2], "fennec_firefox_dev_cache2_entries_profile_default_hex40")
        self.assertNotIn(
            "0a8b9c7d6e5f4a3b2c1d0e9f8a7b6c5d4e3f2a1b",
            " ".join(tokens),
        )

    def test_v32_cache_only_omits_socket_context(self) -> None:
        row = {
            "action": "EVENT_READ",
            "src_kind": "process",
            "src_process_cmd": "org.mozilla.fennec_firefox_dev",
            "dst_kind": "file",
            "dst_file_path": "/dev/socket/dnsproxyd",
        }

        text = clearscope_residual_text_v32_cache_only(row)

        self.assertIn("socket_dnsproxyd", text)
        self.assertNotIn("event_socket_dnsproxyd", text)

    def test_v32_non_socket_event_does_not_add_socket_context(self) -> None:
        row = {
            "action": "EVENT_READ",
            "src_kind": "file",
            "src_file_path": "/dev/socket/logdw",
            "dst_kind": "process",
            "dst_process_cmd": "org.mozilla.fennec_firefox_dev",
        }

        text = clearscope_residual_text_v32(row)

        self.assertNotIn("event_socket_dnsproxyd_to_fennec_firefox_dev", text)


if __name__ == "__main__":
    unittest.main()
