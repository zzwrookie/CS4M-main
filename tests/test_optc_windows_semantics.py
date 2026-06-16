"""Tests for production OpTC Windows semantic extraction."""

from __future__ import annotations

import unittest

from cs4m.semantics.optc_windows import (
    OPTC_WINDOWS_V1_COARSE_SEMANTIC_MODE,
    OPTC_WINDOWS_V1_DETAIL_SEMANTIC_MODE,
    OPTC_WINDOWS_V1_3_DETAIL_SEMANTIC_MODE,
    OPTC_WINDOWS_V1_3B_DETAIL_SEMANTIC_MODE,
    OPTC_WINDOWS_V1_3C_DETAIL_SEMANTIC_MODE,
    bucket_safe_command_lex_v1_3b,
    bucket_safe_command_lex_v1_3c,
    bucket_command_line_v1,
    bucket_file_basename_v1,
    bucket_file_ext_class_v1,
    bucket_file_family_v1,
    bucket_ip_scope_v1,
    bucket_netflow_direction_v1,
    bucket_process_path_v1,
    bucket_process_role,
    bucket_remote_bucket_v1,
    bucket_service_v1,
    file_extension,
    optc_file_natural_tokens_v1,
    optc_netflow_canonical_key_v1_3,
    optc_netflow_natural_tokens_v1,
    optc_netflow_natural_tokens_v1_3,
    optc_process_natural_tokens_v1,
    optc_process_natural_tokens_v1_3,
    optc_process_natural_tokens_v1_3b,
    optc_process_natural_tokens_v1_3c,
    optc_residual_text_v1_coarse,
    optc_residual_text_v1_detail,
    optc_residual_text_v1_3_detail,
    optc_residual_text_v1_3b_detail,
    optc_residual_text_v1_3c_detail,
    process_image_name,
    sanitize_token,
)
from scripts.pipeline.features.semantic_features import residual_text
from scripts.pipeline.checks.preflight import _compact_temp_event_row
from scripts.pipeline.io.event_artifacts import _phase3e_node_tokens_from_db_node
from scripts.pipeline.io.db_stream import _cfg_for_dataset


class OptcWindowsSemanticsTests(unittest.TestCase):
    """Validate label-free production OpTC Windows semantic helpers."""

    def test_process_detail_detects_shell_and_relative_path(self) -> None:
        tokens = optc_process_natural_tokens_v1("PING.EXE", "", detail=True)
        self.assertEqual(tokens[:3], ("process", "windows", "shell"))
        self.assertIn("ping.exe", tokens)
        self.assertIn("relative_or_bare", tokens)

    def test_powershell_encoded_command_bucket(self) -> None:
        cmd = "powershell.exe -NoP -EncodedCommand SQBFAFgA"
        self.assertEqual(bucket_process_role("powershell.exe"), "powershell")
        self.assertEqual(bucket_command_line_v1(cmd), "encoded_command")

    def test_svchost_service_arg_bucket(self) -> None:
        cmd = r"C:\\Windows\\System32\\svchost.exe -k netsvcs"
        self.assertEqual(bucket_command_line_v1(cmd), "service_arg")

    def test_remote_admin_bucket(self) -> None:
        cmd = r"C:\\Windows\\System32\\cmd.exe /c qwinsta user /server 142.20.1.10"
        self.assertEqual(bucket_command_line_v1(cmd), "remote_admin_arg")

    def test_file_ntfs_metadata_and_directory_node(self) -> None:
        self.assertEqual(bucket_file_family_v1(r"/Device/HarddiskVolume1/$Mft"), "ntfs_metadata")
        self.assertEqual(
            bucket_file_basename_v1(r"/Device/HarddiskVolume1/$Extend/$UsnJrnl:$J"),
            "ntfs_metadata",
        )
        self.assertEqual(
            bucket_file_family_v1(r"/Device/HarddiskVolume1/ProgramData"),
            "directory_node",
        )
        self.assertEqual(
            bucket_file_ext_class_v1(r"/Device/HarddiskVolume1/ProgramData"),
            "directory",
        )
        self.assertEqual(
            bucket_file_basename_v1(r"/Device/HarddiskVolume1/Users/alice"),
            "directory_node",
        )

    def test_file_detail_keeps_system_dll_and_payload_bucket(self) -> None:
        system_tokens = optc_file_natural_tokens_v1(
            r"C:\\Windows\\System32\\kernel32.dll",
            detail=True,
        )
        self.assertEqual(
            system_tokens,
            ("file", "windows", "system32", ".dll", "known_system_dll"),
        )
        temp_tokens = optc_file_natural_tokens_v1(
            r"C:\\Windows\\Temp\\dropper.exe",
            detail=True,
        )
        self.assertEqual(
            temp_tokens,
            ("file", "windows", "windows_temp", ".exe", "payload_like"),
        )

    def test_netflow_multicast_llmnr_and_public_services(self) -> None:
        self.assertEqual(bucket_ip_scope_v1("224.0.0.252"), "multicast")
        self.assertEqual(bucket_ip_scope_v1("ff02::1:3"), "multicast")
        self.assertEqual(
            bucket_netflow_direction_v1("10.0.0.1", "224.0.0.252"),
            "multicast",
        )
        self.assertEqual(bucket_service_v1("5355"), "llmnr")
        self.assertEqual(bucket_service_v1("3389"), "rdp")
        self.assertEqual(bucket_service_v1("80"), "http")
        self.assertEqual(bucket_service_v1("443"), "https")
        self.assertEqual(bucket_remote_bucket_v1("142.20.1.7"), "public_142_20_x_x")

    def test_netflow_detail_tokens_do_not_keep_exact_ip(self) -> None:
        tokens = optc_netflow_natural_tokens_v1(
            src_addr="10.0.0.5",
            dst_addr="142.20.1.7",
            dst_port="443",
            detail=True,
        )
        self.assertEqual(
            tokens,
            ("netflow", "windows", "outbound", "public", "https", "public_142_20_x_x"),
        )
        self.assertNotIn("142.20.1.7", tokens)

    def test_leakage_guard_redacts_forbidden_and_long_values(self) -> None:
        token = sanitize_token("process|windows|malicious|attack|ground_truth|label|gt")
        for forbidden in ("malicious", "attack", "ground_truth", "label", "gt"):
            self.assertNotIn(forbidden, token.split("|"))
        self.assertIn("hashlike", sanitize_token("file|aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        self.assertIn("numlike", sanitize_token("file|1234567890123"))

    def test_residual_text_helpers_and_runtime_router(self) -> None:
        row = {
            "action": "EVENT_CONNECT",
            "object_type": "netflow",
            "src_kind": "process",
            "dst_kind": "netflow",
            "src_process_path": r"C:\\Windows\\System32\\cmd.exe",
            "src_process_cmd": r"cmd.exe /c qwinsta user /server 142.20.1.7",
            "src_addr": "10.0.0.5",
            "dst_addr": "142.20.1.7",
            "dst_port": "80",
        }
        direct = optc_residual_text_v1_detail(row)
        routed = residual_text(
            row,
            dataset="optc_051",
            semantic_mode=OPTC_WINDOWS_V1_DETAIL_SEMANTIC_MODE,
        )
        self.assertEqual(routed, direct)
        self.assertIn("remote_admin_arg", routed)
        self.assertIn("http", routed)
        coarse = residual_text(
            row,
            dataset="OPTC_051",
            semantic_mode=OPTC_WINDOWS_V1_COARSE_SEMANTIC_MODE,
        )
        self.assertEqual(coarse, optc_residual_text_v1_coarse(row))
        self.assertNotIn("cmd.exe", coarse)

    def test_phase3e_db_node_tokens_route_optc_modes(self) -> None:
        class Config:
            dataset = "optc_501"
            semantic_mode = OPTC_WINDOWS_V1_DETAIL_SEMANTIC_MODE
            max_tokens_per_node = 8
            theia_netflow_policy = "scope_port"

        node_maps = {
            "process_meta": {1: {"path": r"C:\\Windows\\System32\\cmd.exe", "cmd": "cmd.exe"}},
            "file_meta": {2: {"path": r"/Device/HarddiskVolume1/ProgramData"}},
            "netflow_meta": {
                3: {
                    "src_addr": "10.0.0.5",
                    "src_port": "49152",
                    "dst_addr": "224.0.0.252",
                    "dst_port": "5355",
                },
            },
        }
        process_tokens = _phase3e_node_tokens_from_db_node(
            node_id=1,
            node_kind="process",
            node_summary="",
            node_maps=node_maps,
            config=Config(),
            process_cfg=None,
        )
        file_tokens = _phase3e_node_tokens_from_db_node(
            node_id=2,
            node_kind="file",
            node_summary="",
            node_maps=node_maps,
            config=Config(),
            process_cfg=None,
        )
        netflow_tokens = _phase3e_node_tokens_from_db_node(
            node_id=3,
            node_kind="netflow",
            node_summary="",
            node_maps=node_maps,
            config=Config(),
            process_cfg=None,
        )
        self.assertIn("cmd.exe", process_tokens)
        self.assertIn("directory_node", file_tokens)
        self.assertIn("llmnr", netflow_tokens)

    def test_phase3e_db_node_tokens_route_optc_v13_netflow(self) -> None:
        class Config:
            dataset = "optc_501"
            semantic_mode = OPTC_WINDOWS_V1_3_DETAIL_SEMANTIC_MODE
            max_tokens_per_node = 8
            theia_netflow_policy = "scope_port"

        node_maps = {
            "netflow_meta": {
                3: {
                    "src_addr": "10.0.0.5",
                    "src_port": "49152",
                    "dst_addr": "142.20.1.7",
                    "dst_port": "443",
                },
            },
        }
        netflow_tokens = _phase3e_node_tokens_from_db_node(
            node_id=3,
            node_kind="netflow",
            node_summary="",
            node_maps=node_maps,
            config=Config(),
            process_cfg=None,
        )
        self.assertEqual(
            netflow_tokens,
            ("netflow", "windows", "internal", "https", "142_20_1_7"),
        )

    def test_basic_helpers_are_stable(self) -> None:
        self.assertEqual(process_image_name(r"C:\\Windows\\System32\\cmd.exe"), "cmd.exe")
        self.assertEqual(bucket_process_path_v1(r"C:\\Windows\\System32\\cmd.exe"), "system32")
        self.assertEqual(file_extension(r"C:\\Windows\\System32\\drivers\\x.sys"), ".sys")

    def test_optc_dataset_config_resolves_host_and_splits(self) -> None:
        cfg = _cfg_for_dataset("OPTC_051")
        self.assertEqual(cfg.dataset.db_name, "optc_051")
        self.assertEqual(cfg.dataset.db_name_all, "optc_051")
        self.assertEqual(cfg.dataset.year_month, "2019-09")
        self.assertEqual(
            list(cfg.dataset.train_splits),
            ["day_19", "day_20", "day_21"],
        )
        self.assertEqual(list(cfg.dataset.val_splits), ["day_22"])
        self.assertEqual(list(cfg.dataset.test_splits), ["day_23", "day_24", "day_25"])

    def test_optc_compact_row_maps_ecar_action_to_orthrus10(self) -> None:
        row = (
            4,
            "process",
            "process",
            "cmd.exe",
            "",
            "",
            r"C:\\Windows\\System32\\cmd.exe",
            "cmd.exe",
            "",
            "OPEN",
            10,
            "file",
            "file",
            r"C:\\Windows\\Temp\\a.txt",
            "",
            "",
            "",
            "",
            r"C:\\Windows\\Temp\\a.txt",
            "event-uuid",
            1568788800000000000,
            123,
        )
        compact = _compact_temp_event_row(row, 0, set(), optc_action_mode=True)
        assert compact is not None
        self.assertEqual(compact["action"], "EVENT_OPEN")
        self.assertEqual(compact["object_type"], "file")

    def test_process_v13_detail_excludes_path_and_parent_role(self) -> None:
        tokens = optc_process_natural_tokens_v1_3(
            r"C:\\Windows\\System32\\cmd.exe",
            r"cmd.exe /c qwinsta user /server 142.20.1.10",
            detail=True,
            parent_role="system_service",
        )

        self.assertEqual(tokens, ("process", "windows", "shell", "cmd.exe", "remote_admin_arg"))
        self.assertNotIn("system32", tokens)
        self.assertNotIn("system_service", tokens)

    def test_netflow_v13_normalizes_bidirectional_https_remote_endpoint(self) -> None:
        forward = optc_netflow_natural_tokens_v1_3(
            src_addr="142.20.57.246",
            src_port="53886",
            dst_addr="202.6.172.98",
            dst_port="443",
            detail=True,
        )
        reverse = optc_netflow_natural_tokens_v1_3(
            src_addr="202.6.172.98",
            src_port="443",
            dst_addr="142.20.57.246",
            dst_port="53374",
            detail=True,
        )

        self.assertEqual(forward, ("netflow", "windows", "external", "https", "202_6_172_98"))
        self.assertEqual(reverse, forward)
        self.assertNotIn("53886", forward)
        self.assertNotIn("53374", reverse)

    def test_netflow_v13_canonical_key_keeps_remote_ip_not_ephemeral_port(self) -> None:
        key = optc_netflow_canonical_key_v1_3(
            src_addr="142.20.57.246",
            src_port="53886",
            dst_addr="202.6.172.98",
            dst_port="443",
        )

        self.assertEqual(key, "netflow|windows|external|https|202_6_172_98")
        self.assertNotIn("53886", key)

    def test_residual_text_routes_optc_v13_detail(self) -> None:
        row = {
            "action": "EVENT_CONNECT",
            "object_type": "netflow",
            "src_kind": "process",
            "dst_kind": "netflow",
            "src_process_path": r"C:\\Windows\\System32\\cmd.exe",
            "src_process_cmd": "cmd.exe",
            "src_addr": "142.20.57.246",
            "src_port": "53886",
            "dst_addr": "202.6.172.98",
            "dst_port": "443",
        }

        direct = optc_residual_text_v1_3_detail(row)
        routed = residual_text(
            row,
            dataset="OPTC_501",
            semantic_mode=OPTC_WINDOWS_V1_3_DETAIL_SEMANTIC_MODE,
        )

        self.assertEqual(routed, direct)
        self.assertIn("202_6_172_98", routed)
        self.assertIn("cmd.exe", routed)
        self.assertNotIn("system32", routed)

    def test_process_v13b_role_small_fixes(self) -> None:
        self.assertEqual(bucket_process_role("Idle"), "system_process")
        self.assertEqual(bucket_process_role("MemCompression"), "system_process")
        self.assertEqual(bucket_process_role("ping"), "shell")
        self.assertEqual(bucket_process_role("chcp.com"), "shell")
        self.assertEqual(bucket_process_role("sc"), "lolbin")
        self.assertEqual(bucket_process_role("reg"), "lolbin")

    def test_safe_command_lexical_v13b_tool_buckets(self) -> None:
        cases = {
            "cmd.exe /c qwinsta user /server 142.20.1.10": "cmdtool|qwinsta",
            r"net use \\\\host\\c$ /user:a b": "cmdtool|net_use",
            "schtasks /s 142.20.1.10 /query": "cmdtool|schtasks_remote",
            "wmic process list": "cmdtool|wmic",
            "psexec \\\\host cmd": "cmdtool|psexec",
            "sc config OneSyncSvc start=disabled": "cmdtool|sc",
            "reg add HKCU\\\\Software\\\\Microsoft /f": "cmdtool|reg",
            "ping -n 2 127.0.0.1": "cmdtool|ping",
            "rundll32.exe shell32.dll,Control_RunDLL": "cmdtool|rundll32",
            "regsvr32.exe /s foo.dll": "cmdtool|regsvr32",
        }
        for cmd, expected in cases.items():
            self.assertEqual(bucket_safe_command_lex_v1_3b(cmd), expected)

    def test_safe_command_lexical_v13b_argument_buckets(self) -> None:
        cases = {
            "powershell -EncodedCommand SQBFAFgA": "cmdarg|encoded",
            "cmd /c curl http://example.test/a": "cmdarg|url",
            r"cmd /c C:\\Windows\\Temp\\payload.exe": "cmdarg|temp_path",
            "powershell ./x.ps1": "cmdarg|script_ext_ps1",
            "wscript.exe x.vbs": "cmdarg|script_ext_vbs",
            "cmd.exe /c x.bat": "cmdarg|script_ext_bat",
            "cmd.exe /c dir 142.20.1.10": "cmdarg|ip_literal",
            r"cmd.exe /c dir \\\\host\\share": "cmdarg|unc_path",
            "svchost.exe -k netsvcs": "cmdarg|service_arg",
            "reg query HKLM\\\\Software\\\\Microsoft": "cmdtool|reg",
            "plain harmless command": "cmdarg|other_safe",
        }
        for cmd, expected in cases.items():
            self.assertEqual(bucket_safe_command_lex_v1_3b(cmd), expected)

    def test_process_v13b_detail_adds_safe_command_lexical(self) -> None:
        tokens = optc_process_natural_tokens_v1_3b(
            "sc",
            "sc config OneSyncSvc start=disabled",
            detail=True,
        )

        self.assertEqual(
            tokens,
            ("process", "windows", "lolbin", "sc", "other_cmd", "cmdtool|sc"),
        )

    def test_residual_text_routes_optc_v13b_detail_and_keeps_v13_netflow(self) -> None:
        row = {
            "action": "EVENT_CONNECT",
            "object_type": "netflow",
            "src_kind": "process",
            "dst_kind": "netflow",
            "src_process_path": "sc",
            "src_process_cmd": "sc config OneSyncSvc start=disabled",
            "src_addr": "142.20.57.246",
            "src_port": "53886",
            "dst_addr": "202.6.172.98",
            "dst_port": "443",
        }

        direct = optc_residual_text_v1_3b_detail(row)
        routed = residual_text(
            row,
            dataset="OPTC_501",
            semantic_mode=OPTC_WINDOWS_V1_3B_DETAIL_SEMANTIC_MODE,
        )

        self.assertEqual(routed, direct)
        self.assertIn("cmdtool|sc", routed)
        self.assertIn("202_6_172_98", routed)

    def test_safe_command_lexical_v13c_second_batch_tool_buckets(self) -> None:
        cases = {
            r"C:\\Windows\\System32\\conhost.exe -ForceV1": "cmdhost|conhost_forcev1",
            "cmd.exe /c whoami": "cmdshell|cmd_c",
            "schtasks /query /tn foo": "cmdtool|schtasks_query",
            "schtasks /delete /tn foo /f": "cmdtool|schtasks_delete",
            "taskkill /f /im foo.exe": "cmdtool|taskkill",
            "tasklist /v": "cmdtool|tasklist",
            "netstat -n": "cmdtool|netstat",
            "gpupdate.exe /target:computer": "cmdtool|gpupdate",
            "bcdedit /set foo bar": "cmdtool|bcdedit",
            "shutdown.exe /r /t 0": "cmdtool|shutdown",
            r"C:\\Windows\\SysWOW64\\OneDriveSetup.exe /uninstall": "cmdtool|onedrive_setup",
            r"C:\\Windows\\servicing\\TrustedInstaller.exe": "cmdtool|trustedinstaller",
            r"C:\\Windows\\System32\\sppsvc.exe": "cmdtool|sppsvc",
            "SearchIndexer.exe /Embedding": "cmdtool|search_indexer",
            "BackgroundTaskHost.exe -ServerName:App.AppXabc.mca": (
                "cmdtool|background_task_host"
            ),
            "RuntimeBroker.exe -Embedding": "cmdtool|runtime_broker",
            "dllhost.exe /Processid:{12345678-1234-1234-1234-123456789abc}": (
                "cmdtool|dllhost"
            ),
            "wermgr.exe -upload": "cmdtool|wermgr",
            "WmiPrvSE.exe -Embedding": "cmdtool|wmiprvse",
            r"python.exe C:\\NCR\\kickoff\\kickoff.py": "cmdtool|python_script",
        }
        for cmd, expected in cases.items():
            self.assertEqual(bucket_safe_command_lex_v1_3c(cmd), expected)

    def test_safe_command_lexical_v13c_second_batch_argument_buckets(self) -> None:
        cases = {
            "foo.exe -Embedding": "cmdarg|embedding",
            "foo.exe -ServerName:App.AppXabc.mca": "cmdarg|servername_appx",
            "foo.exe /uninstall": "cmdarg|uninstall",
            "googleupdate.exe /ua /installsource scheduler": "cmdarg|update_task",
            "foo.exe /target:user": "cmdarg|group_policy",
            "mscorsvw.exe -StartupEvent a -NGenProcess b": "cmdarg|ngen",
            "ping -n 2 127.0.0.1": "cmdtool|ping",
            "plain harmless command": "cmdarg|other_safe",
        }
        for cmd, expected in cases.items():
            self.assertEqual(bucket_safe_command_lex_v1_3c(cmd), expected)

    def test_process_v13c_detail_adds_expanded_safe_command_lexical(self) -> None:
        tokens = optc_process_natural_tokens_v1_3c(
            r"C:\\Windows\\System32\\conhost.exe",
            r"C:\\Windows\\System32\\conhost.exe -ForceV1",
            detail=True,
        )

        self.assertEqual(
            tokens,
            (
                "process",
                "windows",
                "shell",
                "conhost.exe",
                "other_cmd",
                "cmdhost|conhost_forcev1",
            ),
        )

    def test_residual_text_routes_optc_v13c_detail_and_keeps_v13_netflow(self) -> None:
        row = {
            "action": "EVENT_CONNECT",
            "object_type": "netflow",
            "src_kind": "process",
            "dst_kind": "netflow",
            "src_process_path": r"C:\\Windows\\System32\\cmd.exe",
            "src_process_cmd": "cmd.exe /c whoami",
            "src_addr": "142.20.57.246",
            "src_port": "53886",
            "dst_addr": "202.6.172.98",
            "dst_port": "443",
        }

        direct = optc_residual_text_v1_3c_detail(row)
        routed = residual_text(
            row,
            dataset="OPTC_501",
            semantic_mode=OPTC_WINDOWS_V1_3C_DETAIL_SEMANTIC_MODE,
        )

        self.assertEqual(routed, direct)
        self.assertIn("cmdshell|cmd_c", routed)
        self.assertIn("202_6_172_98", routed)

    def test_phase3e_db_node_tokens_route_optc_v13c_process_and_netflow(self) -> None:
        class Config:
            dataset = "optc_501"
            semantic_mode = OPTC_WINDOWS_V1_3C_DETAIL_SEMANTIC_MODE
            max_tokens_per_node = 8
            theia_netflow_policy = "scope_port"

        node_maps = {
            "process_meta": {
                1: {
                    "path": r"C:\\Windows\\System32\\cmd.exe",
                    "cmd": "cmd.exe /c whoami",
                },
            },
            "netflow_meta": {
                3: {
                    "src_addr": "142.20.57.246",
                    "src_port": "53886",
                    "dst_addr": "202.6.172.98",
                    "dst_port": "443",
                },
            },
        }
        process_tokens = _phase3e_node_tokens_from_db_node(
            node_id=1,
            node_kind="process",
            node_summary="",
            node_maps=node_maps,
            config=Config(),
            process_cfg=None,
        )
        netflow_tokens = _phase3e_node_tokens_from_db_node(
            node_id=3,
            node_kind="netflow",
            node_summary="",
            node_maps=node_maps,
            config=Config(),
            process_cfg=None,
        )

        self.assertIn("cmdshell|cmd_c", process_tokens)
        self.assertEqual(
            netflow_tokens,
            ("netflow", "windows", "external", "https", "202_6_172_98"),
        )


if __name__ == "__main__":
    unittest.main()
