"""Tests for ClearScope v32 semantic mode routing."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

from scripts.pipeline.features.semantic_features import residual_text


class ClearScopeV32PipelineRoutingTests(unittest.TestCase):
    """Validate v32 routing through feature and runner entrypoints."""

    def test_residual_text_uses_v32_socket_context(self) -> None:
        row = {
            "action": "EVENT_READ",
            "src_kind": "file",
            "src_file_path": "/dev/socket/dnsproxyd",
            "dst_kind": "process",
            "dst_process_cmd": "org.mozilla.fennec_firefox_dev",
        }

        text = residual_text(
            row,
            dataset="CLEARSCOPE_E3",
            semantic_mode="raw_detail_v32_discriminative",
        )

        self.assertIn("event_socket_dnsproxyd_to_fennec_firefox_dev", text)

    def test_residual_text_uses_v32_socket_context_for_stream_direction(self) -> None:
        row = {
            "action": "EVENT_READ",
            "src_kind": "process",
            "src_process_cmd": "org.mozilla.fennec_firefox_dev",
            "dst_kind": "file",
            "dst_file_path": "/dev/socket/dnsproxyd",
        }

        text = residual_text(
            row,
            dataset="CLEARSCOPE_E3",
            semantic_mode="raw_detail_v32_discriminative",
        )

        self.assertIn("event_socket_dnsproxyd_from_fennec_firefox_dev", text)

    def test_residual_text_routes_v32_cache_only_without_socket_context(self) -> None:
        row = {
            "action": "EVENT_READ",
            "src_kind": "process",
            "src_process_cmd": "org.mozilla.fennec_firefox_dev",
            "dst_kind": "file",
            "dst_file_path": "/dev/socket/dnsproxyd",
        }

        text = residual_text(
            row,
            dataset="CLEARSCOPE_E3",
            semantic_mode="raw_detail_v32_cache_only_discriminative",
        )

        self.assertIn("socket_dnsproxyd", text)
        self.assertNotIn("event_socket_dnsproxyd", text)

    def test_runner_dry_run_accepts_v32_cache_only_mode(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "STAGE": "infer_full",
                "SEMANTIC_MODE": "raw_detail_v32_cache_only_discriminative",
                "PYTHON_BIN": "python3",
                "CLAD_DB_PASSWORD": "unused",
            },
        )

        result = subprocess.run(
            ["bash", "scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh"],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        combined = result.stdout + result.stderr
        self.assertIn("RAW_DETAIL_V32_CACHE_ONLY_DISCRIMINATIVE", combined)
        self.assertIn("raw_detail_v32_cache_only_discriminative", combined)
        self.assertIn("CLEARSCOPE_E3_v32_cache_only", combined)
        self.assertIn("CLEARSCOPE_E3_latent64_v32_cache_only", combined)

    def test_runner_dry_run_accepts_v32_mode(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "STAGE": "infer_full",
                "SEMANTIC_MODE": "raw_detail_v32_discriminative",
                "PYTHON_BIN": "python3",
                "CLAD_DB_PASSWORD": "unused",
            },
        )

        result = subprocess.run(
            ["bash", "scripts/run/run_clearscope_e3_v3_phase3e_phase3g_full.sh"],
            cwd=repo,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        combined = result.stdout + result.stderr
        self.assertIn("RAW_DETAIL_V32_DISCRIMINATIVE", combined)
        self.assertIn("raw_detail_v32_discriminative", combined)
        self.assertIn("CLEARSCOPE_E3_v32", combined)


if __name__ == "__main__":
    unittest.main()
