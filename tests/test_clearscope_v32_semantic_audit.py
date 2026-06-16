"""Tests for ClearScope v32 semantic audit helpers."""

from __future__ import annotations

import unittest

from tmp.clearscope_v32_semantic_audit import (
    contains_exact_hex_material,
    summarize_replacements,
)


class ClearScopeV32SemanticAuditTests(unittest.TestCase):
    """Validate pure audit helper behavior."""

    def test_summarize_replacements_counts_changed_tokens(self) -> None:
        rows = [
            {"v31_token": "file a old", "v32_token": "file a new", "split": "train"},
            {"v31_token": "file a old", "v32_token": "file a new", "split": "test"},
            {"v31_token": "file b same", "v32_token": "file b same", "split": "test"},
        ]

        result = summarize_replacements(rows)

        self.assertEqual(result["total_rows"], 3)
        self.assertEqual(result["changed_rows"], 2)
        self.assertEqual(result["replacement_pairs"][("file a old", "file a new")], 2)
        self.assertEqual(result["v32_split_counts"][("file a new", "train")], 1)

    def test_contains_exact_hex_material_rejects_full_hash_and_prefix(self) -> None:
        self.assertTrue(
            contains_exact_hex_material(
                "file cache 0a8bf759c7672a43a3452c951b80220c95c1576f",
            ),
        )
        self.assertTrue(contains_exact_hex_material("file cache 0a8bf759"))
        self.assertFalse(
            contains_exact_hex_material(
                "file android_app_cache_file fennec_firefox_dev_cache2_entries_profile_default_hex40",
            ),
        )


if __name__ == "__main__":
    unittest.main()
