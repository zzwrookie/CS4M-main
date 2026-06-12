"""Tests for ClearScope E5 v33 Android-safe semantics."""

from __future__ import annotations

import unittest

from cs4m.semantics.clearscope_android import (
    CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE,
    CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE,
    android_file_detail_v33_e5_android_safe,
    clearscope_file_natural_tokens_v31,
    clearscope_file_natural_tokens_v33b_e5_android_safe,
    clearscope_file_natural_tokens_v33_e5_android_safe,
    clearscope_residual_text_v33_e5_android_safe,
    normalize_clearscope_semantic_mode,
)
from scripts.pipeline.features.semantic_features import residual_text


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

    def test_v33b_mode_aliases_normalize(self) -> None:
        self.assertEqual(
            normalize_clearscope_semantic_mode("raw_detail_v33b_e5_android_safe"),
            CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE,
        )
        self.assertEqual(
            normalize_clearscope_semantic_mode(
                "clearscope_raw_detail_v33b_e5_android_safe"
            ),
            CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE,
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

    def test_v33b_accounting_paths_use_low_cardinality_tokens(self) -> None:
        expected = {
            "/acct": ("file", "android_accounting_root", "acct_root"),
            "/acct/uid_0": ("file", "android_accounting_uid", "acct_uid"),
            "/acct/uid_0/pid_549": (
                "file",
                "android_accounting_pid",
                "acct_pid",
            ),
            "/acct/uid_1000/pid_12345": (
                "file",
                "android_accounting_pid",
                "acct_pid",
            ),
            "/acct/uid_bad": ("file", "android_accounting_other", "acct_other"),
            "/acct/uid_0/other": (
                "file",
                "android_accounting_other",
                "acct_other",
            ),
        }
        for path, tokens in expected.items():
            with self.subTest(path=path):
                self.assertEqual(
                    clearscope_file_natural_tokens_v33b_e5_android_safe(path),
                    tokens,
                )

    def test_v33_does_not_route_accounting_paths(self) -> None:
        self.assertEqual(
            clearscope_file_natural_tokens_v33_e5_android_safe("/acct/uid_0/pid_549"),
            ("file", "file_other", "other"),
        )

    def test_v33b_preserves_v33_sdcardfs_and_device_behavior(self) -> None:
        self.assertEqual(
            clearscope_file_natural_tokens_v33b_e5_android_safe(
                "/config/sdcardfs/de.belu.appstarter/appid"
            ),
            ("file", "android_sdcardfs_appid", "de_belu_appstarter_appid"),
        )
        self.assertEqual(
            clearscope_file_natural_tokens_v33b_e5_android_safe("/dev/msm_g711tlaw"),
            ("file", "android_device_file", "dev_other"),
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

    def test_v33_residual_text_uses_sdcardfs_appid_detail(self) -> None:
        row = {
            "action": "EVENT_READ",
            "src_kind": "process",
            "src_process_cmd": "com.android.providers.contacts",
            "dst_kind": "file",
            "dst_file_path": "/config/sdcardfs/com.android.providers.contacts/appid",
        }

        text = clearscope_residual_text_v33_e5_android_safe(row)

        self.assertIn("android_sdcardfs_appid", text.split())
        self.assertIn("com_android_providers_contacts_appid", text.split())

    def test_pipeline_residual_text_routes_v33_mode(self) -> None:
        row = {
            "action": "EVENT_READ",
            "src_kind": "process",
            "src_process_cmd": "com.android.providers.contacts",
            "dst_kind": "file",
            "dst_file_path": "/config/sdcardfs/de.belu.appstarter/appid",
        }

        text = residual_text(
            row,
            dataset="CLEARSCOPE_E5",
            semantic_mode="raw_detail_v33_e5_android_safe",
        )

        self.assertIn("android_sdcardfs_appid", text.split())
        self.assertIn("de_belu_appstarter_appid", text.split())

    def test_pipeline_residual_text_default_does_not_route_v33_mode(self) -> None:
        row = {
            "action": "EVENT_READ",
            "src_kind": "process",
            "src_process_cmd": "com.android.providers.contacts",
            "dst_kind": "file",
            "dst_file_path": "/config/sdcardfs/de.belu.appstarter/appid",
        }

        text = residual_text(row, dataset="CLEARSCOPE_E5")

        self.assertIn("file_other", text.split())
        self.assertIn("other", text.split())
        self.assertNotIn("android_sdcardfs_appid", text.split())

    def test_v33_residual_text_keeps_netflow_fixed_coarse(self) -> None:
        row = {
            "action": "EVENT_CONNECT",
            "src_kind": "process",
            "src_process_cmd": "com.example.app",
            "dst_kind": "netflow",
            "dst_addr": "10.0.0.2",
            "dst_port": "443",
        }

        text = clearscope_residual_text_v33_e5_android_safe(row)

        self.assertIn("netflow", text.split())
        self.assertNotIn("10_0_0_2", text)
        self.assertNotIn("443", text.split())


if __name__ == "__main__":
    unittest.main()
