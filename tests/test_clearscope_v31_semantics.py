"""Tests for ClearScope raw_detail_v31_discriminative semantics."""

from __future__ import annotations

import unittest

from cs4m.semantics.clearscope_android import (
    CLEARSCOPE_V31_SEMANTIC_MODE,
    android_file_detail_v31,
    clearscope_file_natural_tokens_v3,
    clearscope_file_natural_tokens_v31,
    normalize_clearscope_semantic_mode,
)


HEX40 = "4BCE9F2B3270508CC941B753991F527FAABA2800"
BASE = "/data/data/org.mozilla.fennec_firefox_dev/cache/nz9885vc.default/cache2/entries"


class ClearScopeV31SemanticsTests(unittest.TestCase):
    """Validate v3.1 discriminative ClearScope token behavior."""

    def test_v3_cache_entry_behavior_is_unchanged(self) -> None:
        self.assertEqual(
            clearscope_file_natural_tokens_v3(f"{BASE}/{HEX40}"),
            (
                "file",
                "android_app_cache_file",
                "fennec_firefox_dev_cache2_entries_hexblob",
            ),
        )

    def test_v31_splits_hex40_body_from_metadata_without_exact_hash(self) -> None:
        body = clearscope_file_natural_tokens_v31(f"{BASE}/{HEX40}")
        meta = clearscope_file_natural_tokens_v31(f"{BASE}/{HEX40}.tc-md")

        self.assertEqual(
            body,
            (
                "file",
                "android_app_cache_file",
                "fennec_firefox_dev_cache2_entries_hex40_body",
            ),
        )
        self.assertEqual(
            meta,
            (
                "file",
                "android_app_cache_file",
                "fennec_firefox_dev_cache2_entries_hex40_tc_md",
            ),
        )
        self.assertNotIn(HEX40.lower(), " ".join(body).lower())
        self.assertNotIn(HEX40.lower(), " ".join(meta).lower())

    def test_v31_keeps_doomed_numeric_shape_not_exact_number(self) -> None:
        detail = android_file_detail_v31(
            "/data/data/org.mozilla.fennec_firefox_dev/cache/nz9885vc.default/"
            "cache2/doomed/1525576060",
        )

        self.assertEqual(detail, "fennec_firefox_dev_cache2_doomed_num")
        self.assertNotIn("1525576060", detail)

    def test_v31_aliases_normalize(self) -> None:
        self.assertEqual(
            normalize_clearscope_semantic_mode("clearscope_raw_detail_v31_discriminative"),
            CLEARSCOPE_V31_SEMANTIC_MODE,
        )


if __name__ == "__main__":
    unittest.main()
