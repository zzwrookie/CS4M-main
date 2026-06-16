"""Tests for OpTC Windows semantic audit helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cs4m.semantics.optc_windows import (
    bucket_command_line_v1,
    bucket_file_basename_v1,
    bucket_file_family_v1,
    bucket_ip_scope_v1,
    bucket_process_path_v1,
    bucket_service_v1,
)
from scripts.tools.audit_optc_windows_semantics import (
    bucket_command_line,
    bucket_file_basename,
    bucket_file_family,
    bucket_process_role,
    bucket_registry_family,
    bucket_windows_port,
    is_candidate_column,
    parse_netflow,
    parse_ground_truth_csv,
    process_image_name,
    sanitize_token,
)


class OptcWindowsSemanticAuditTests(unittest.TestCase):
    """Validate label-free Windows audit token helpers."""

    def test_process_role_detects_powershell(self) -> None:
        image = "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        self.assertEqual(bucket_process_role(image), "powershell")

    def test_command_line_detects_encoded_command(self) -> None:
        cmd = "powershell.exe -NoP -EncodedCommand SQBFAFgA"
        self.assertEqual(bucket_command_line(cmd), "encoded_command")

    def test_file_family_detects_user_temp(self) -> None:
        path = r"C:\\Users\\alice\\AppData\\Local\\Temp\\dropper.exe"
        self.assertEqual(bucket_file_family(path), "user_temp")

    def test_file_basename_detects_driver(self) -> None:
        self.assertEqual(bucket_file_basename(r"C:\\Windows\\System32\\drivers\\x.sys"), "driver")

    def test_registry_family_detects_run_key(self) -> None:
        path = (
            r"HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion"
            r"\\Run\\Updater"
        )
        self.assertEqual(bucket_registry_family(path), "run_key")

    def test_port_bucket_detects_rdp(self) -> None:
        self.assertEqual(bucket_windows_port("3389"), "port_3389_rdp")

    def test_process_image_name_handles_device_path(self) -> None:
        image = r"/Device/HarddiskVolume1/Windows/System32/cmd.exe"
        self.assertEqual(process_image_name(image), "cmd.exe")

    def test_process_image_name_handles_program_files_path_with_spaces(self) -> None:
        image = r"/Device/HarddiskVolume1/Program Files (x86)/Google/Update/GoogleUpdate.exe"
        self.assertEqual(process_image_name(image), "googleupdate.exe")

    def test_sanitize_token_removes_forbidden_words(self) -> None:
        token = sanitize_token("process|windows|malicious|attack|label|gt")
        self.assertNotIn("malicious", token)
        self.assertNotIn("attack", token)
        self.assertNotIn("label", token)
        self.assertNotIn("gt", token)

    def test_sanitize_token_does_not_rewrite_normal_substrings(self) -> None:
        token = sanitize_token("file|windows|programdata|.gthr|other_file")
        self.assertEqual(token, "file|windows|programdata|.gthr|other_file")

    def test_parse_ground_truth_csv_reads_last_column_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gt.csv"
            path.write_text(
                "uuid,detail,node_idx\n"
                "abc,{'subject': 'cmd.exe'},42\n"
                "def,{'netflow': '1.1.1.1:1->2.2.2.2:443'},99\n",
                encoding="utf-8",
            )
            rows = parse_ground_truth_csv(path, "optc_test")

        self.assertEqual({row.entity_id for row in rows}, {"42", "99"})
        self.assertEqual(rows[0].dataset, "optc_test")

    def test_candidate_column_filter_includes_windows_relevant_names(self) -> None:
        self.assertTrue(is_candidate_column("event_table", "subject_uuid"))
        self.assertTrue(is_candidate_column("subject_node_table", "cmd_line"))
        self.assertTrue(is_candidate_column("file_node_table", "file_path"))
        self.assertTrue(is_candidate_column("netflow_node_table", "remote_ip"))
        self.assertFalse(is_candidate_column("pg_stat", "heap_blks_hit"))

    def test_v1_ip_scope_detects_multicast_and_link_local(self) -> None:
        self.assertEqual(bucket_ip_scope_v1("224.0.0.252"), "multicast")
        self.assertEqual(bucket_ip_scope_v1("ff02::1:3"), "multicast")
        self.assertEqual(bucket_ip_scope_v1("fe80::1"), "link_local")

    def test_v1_service_detects_windows_ports(self) -> None:
        self.assertEqual(bucket_service_v1("5355"), "llmnr")
        self.assertEqual(bucket_service_v1("3389"), "rdp")
        self.assertEqual(bucket_service_v1("443"), "https")

    def test_v1_file_family_detects_ntfs_metadata(self) -> None:
        self.assertEqual(bucket_file_family_v1(r"/Device/HarddiskVolume1/$Mft"), "ntfs_metadata")
        self.assertEqual(
            bucket_file_basename_v1(r"/Device/HarddiskVolume1/$Extend/$UsnJrnl:$J"),
            "ntfs_metadata",
        )

    def test_v1_process_path_detects_relative_or_bare(self) -> None:
        self.assertEqual(bucket_process_path_v1("PING.EXE"), "relative_or_bare")

    def test_v1_command_line_detects_remote_admin(self) -> None:
        cmd = r"C:/Windows/system32/cmd.exe /c qwinsta user /server 142.20.61.187"
        self.assertEqual(bucket_command_line_v1(cmd), "remote_admin_arg")

    def test_parse_netflow_handles_ipv6_endpoints(self) -> None:
        raw = "fe80::31e6:e252:6926:f81b:50669->ff02::1:3:5355"
        self.assertEqual(
            parse_netflow(raw),
            ("fe80::31e6:e252:6926:f81b", "50669", "ff02::1:3", "5355"),
        )


if __name__ == "__main__":
    unittest.main()
