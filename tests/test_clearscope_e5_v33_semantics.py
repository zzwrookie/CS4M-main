"""Tests for ClearScope E5 v33 Android-safe semantics."""

from __future__ import annotations

import unittest

from cs4m.semantics.clearscope_android import (
    CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE,
    android_file_detail_v33_e5_android_safe,
    clearscope_file_natural_tokens_v31,
    clearscope_file_natural_tokens_v33_e5_android_safe,
    normalize_clearscope_semantic_mode,
)


class ClearScopeE5V33SemanticsTests(unittest.TestCase):
    """Validate v33 E5 Android-safe ClearScope token behavior."""

    def test_v33_mode_aliases_normalize(self) -> None:
        self.assertEqual(
            normalize_clearscope_semantic_mode("raw_detail_v33_e5_android_safe"),
            CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE,
        )
        self.assertEqual(
            normalize_clearscope_semantic_mode(
                "clearscope_raw_detail_v33_e5_android_safe"
            ),
            CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE,
        )

    def test_v33_sdcardfs_appid_keeps_general_package_family(self) -> None:
        self.assertEqual(
            clearscope_file_natural_tokens_v33_e5_android_safe(
                "/config/sdcardfs/com.android.providers.contacts/appid"
            ),
            (
                "file",
                "android_sdcardfs_appid",
                "com_android_providers_contacts_appid",
            ),
        )
        self.assertEqual(
            clearscope_file_natural_tokens_v33_e5_android_safe(
                "/config/sdcardfs/de.belu.appstarter/appid"
            ),
            ("file", "android_sdcardfs_appid", "de_belu_appstarter_appid"),
        )
        self.assertEqual(
            clearscope_file_natural_tokens_v33_e5_android_safe(
                "/config/sdcardfs/com.bloketech.lockwatch/appid"
            ),
            ("file", "android_sdcardfs_appid", "com_bloketech_lockwatch_appid"),
        )

    def test_v33_sdcardfs_package_directory_keeps_general_package_family(self) -> None:
        self.assertEqual(
            clearscope_file_natural_tokens_v33_e5_android_safe(
                "/config/sdcardfs/de.belu.appstarter"
            ),
            ("file", "android_sdcardfs_package", "de_belu_appstarter"),
        )

    def test_v33_device_allowlist(self) -> None:
        expected = {
            "/dev/pmsg0": "dev_pmsg0",
            "/dev/ashmem": "dev_ashmem",
            "/dev/ion": "dev_ion",
            "/dev/binder": "dev_binder",
            "/dev/hwbinder": "dev_hwbinder",
            "/dev/vndbinder": "dev_vndbinder",
            "/dev/null": "dev_null",
            "/dev/zero": "dev_zero",
        }
        for path, detail in expected.items():
            with self.subTest(path=path):
                self.assertEqual(
                    android_file_detail_v33_e5_android_safe(
                        path,
                        "android_device_file",
                    ),
                    detail,
                )

    def test_v33_unknown_device_stays_dev_other(self) -> None:
        for path in ("/dev/msm_g711tlaw", "/dev/urandom", "/dev/kgsl-3d0"):
            with self.subTest(path=path):
                self.assertEqual(
                    android_file_detail_v33_e5_android_safe(
                        path,
                        "android_device_file",
                    ),
                    "dev_other",
                )

    def test_v31_targeted_path_is_unchanged(self) -> None:
        self.assertEqual(
            clearscope_file_natural_tokens_v31(
                "/config/sdcardfs/de.belu.appstarter/appid"
            ),
            ("file", "file_other", "other"),
        )

    def test_v33_non_target_path_matches_v31(self) -> None:
        path = "/data/data/org.mozilla.fennec_vagrant/cache/x/cache2/doomed/12345"
        self.assertEqual(
            clearscope_file_natural_tokens_v33_e5_android_safe(path),
            clearscope_file_natural_tokens_v31(path),
        )


if __name__ == "__main__":
    unittest.main()
