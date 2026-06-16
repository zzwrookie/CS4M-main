"""Label-free OpTC Windows semantic tokens.

This module contains runtime-safe Windows entity semantic extraction for OpTC.
Rules use only event/entity surface fields and must not depend on ground truth,
attack windows, malicious node lists, or test metrics.
"""

from __future__ import annotations

import hashlib
import ipaddress
import re
from collections.abc import Mapping


OPTC_WINDOWS_V1_DETAIL_SEMANTIC_MODE = "optc_windows_v1_detail"
OPTC_WINDOWS_V1_COARSE_SEMANTIC_MODE = "optc_windows_v1_coarse"
OPTC_WINDOWS_V1_3_DETAIL_SEMANTIC_MODE = "optc_windows_v1_3_detail"
OPTC_WINDOWS_V1_3_COARSE_SEMANTIC_MODE = "optc_windows_v1_3_coarse"
OPTC_WINDOWS_V1_3B_DETAIL_SEMANTIC_MODE = "optc_windows_v1_3b_detail"
OPTC_WINDOWS_V1_3B_COARSE_SEMANTIC_MODE = "optc_windows_v1_3b_coarse"
OPTC_WINDOWS_V1_3C_DETAIL_SEMANTIC_MODE = "optc_windows_v1_3c_detail"
OPTC_WINDOWS_V1_3C_COARSE_SEMANTIC_MODE = "optc_windows_v1_3c_coarse"
OPTC_NETFLOW_CANONICALIZATION_V1_3 = "remote_endpoint_v1_3"
FORBIDDEN_TOKEN_WORDS = ("malicious", "attack", "ground_truth", "label", "gt")


def is_optc_dataset(dataset: object) -> bool:
    """Return whether a dataset name denotes an OpTC Windows host database."""
    text = str(dataset or "").strip().lower()
    return text.startswith("optc") or text.startswith("op_tc")


def normalize_windows_path(value: object) -> str:
    """Return a lower-cased Windows-ish path with normalized separators."""
    text = str(value or "").strip().replace("/", "\\").replace('"', "").lower()
    return re.sub(r"\\+", r"\\", text)


def process_image_name(value: object) -> str:
    """Extract a stable process image basename without command arguments."""
    text = str(value or "").strip().replace('"', "")
    if not text:
        return "unknown_process"
    match = re.search(r"(?i)(?:^|[\\/])([^\\/\"\s]+\.exe)\b", text)
    if match:
        return sanitize_segment(match.group(1).lower())
    first = text.split()[0] if (" " in text and "\\" not in text and "/" not in text) else text
    name = first.replace("/", "\\").rsplit("\\", 1)[-1].lower()
    if not name:
        return "unknown_process"
    return sanitize_segment(name)


def sanitize_segment(value: object, max_len: int = 80) -> str:
    """Normalize one token segment and redact leakage or unstable identifiers."""
    text = str(value or "").strip().lower()
    if text in FORBIDDEN_TOKEN_WORDS:
        return "redacted"
    text = re.sub(r"[0-9a-f]{24,}", "hashlike", text)
    text = re.sub(r"\b\d{9,}\b", "numlike", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_.:+/|-]+", "_", text).strip("_")
    if text in FORBIDDEN_TOKEN_WORDS:
        return "redacted"
    if not text:
        return "unknown"
    return text[: max(int(max_len), 1)]


def sanitize_token(token: object) -> str:
    """Sanitize a pipe-delimited semantic token."""
    return "|".join(sanitize_segment(part) for part in str(token or "").split("|"))


def bucket_process_role(image: object) -> str:
    """Bucket a Windows process image into a label-free role."""
    name = process_image_name(image)
    if name in {"idle", "memcompression"}:
        return "system_process"
    if name in {"powershell.exe", "pwsh.exe"}:
        return "powershell"
    if name in {"cmd.exe", "conhost.exe", "ping.exe", "ping", "chcp.com"}:
        return "shell"
    if name in {"wscript.exe", "cscript.exe", "mshta.exe"}:
        return "script_host"
    if name in {"chrome.exe", "firefox.exe", "iexplore.exe", "edge.exe", "msedge.exe"}:
        return "browser"
    if name in {"winword.exe", "excel.exe", "powerpnt.exe", "outlook.exe"}:
        return "office"
    if name in {
        "rundll32.exe",
        "regsvr32.exe",
        "wmic.exe",
        "certutil.exe",
        "bitsadmin.exe",
        "msiexec.exe",
        "schtasks.exe",
        "sc",
        "reg",
    }:
        return "lolbin"
    if name in {"services.exe", "svchost.exe", "lsass.exe", "csrss.exe", "wininit.exe"}:
        return "system_service"
    if name in {"system", "system.exe", "smss.exe"}:
        return "system_process"
    if name in {"msmpeng.exe", "windefend.exe"}:
        return "security_tool"
    if name in {"python.exe", "pythonw.exe", "git.exe", "powershell_ise.exe"}:
        return "developer_tool"
    if name in {"setup.exe", "installer.exe", "googleupdate.exe"}:
        return "installer"
    if name.endswith(".exe"):
        return "user_app"
    return "unknown_process"


def bucket_process_path_v1(image: object) -> str:
    """Bucket process path for OpTC Windows v1."""
    raw = str(image or "").strip()
    text = normalize_windows_path(raw)
    if not text or text == "none":
        return "unknown_path"
    if "\\" not in text and "/" not in raw:
        return "relative_or_bare"
    if "\\windows\\system32\\" in text:
        return "system32"
    if "\\windows\\syswow64\\" in text:
        return "syswow64"
    if "\\windows\\temp\\" in text:
        return "windows_temp"
    if "\\appdata\\local\\temp\\" in text:
        return "user_temp"
    if "\\program files" in text:
        return "program_files"
    if "\\programdata\\" in text:
        return "programdata"
    if "\\users\\" in text:
        return "user_profile"
    return "device_path_other"


def bucket_command_line_v1(command_line: object) -> str:
    """Bucket Windows command line into a stable v1 shape."""
    text = str(command_line or "").strip().lower()
    norm = normalize_windows_path(text)
    if not text or text == "none":
        return "empty_cmd"
    if "-enc" in text or "encodedcommand" in text:
        return "encoded_command"
    if "downloadstring" in text or "downloadfile" in text:
        return "download_string"
    if "invoke-expression" in text or "iex " in text:
        return "invoke_expression"
    if "%temp%" in text or "\\appdata\\local\\temp\\" in norm or "\\windows\\temp\\" in norm:
        return "temp_exec_arg"
    if any(marker in text for marker in ("qwinsta", "psexec", "wmic", "net use")):
        return "remote_admin_arg"
    if "schtasks" in text and (" /s " in text or "\\\\" in text):
        return "remote_admin_arg"
    if "svchost.exe" in text and " -k " in text:
        return "service_arg"
    office_markers = ("winword.exe", "excel.exe", "powerpnt.exe", "outlook.exe")
    if any(marker in text for marker in office_markers):
        return "office_child_arg"
    if re.search(r"https?://", text):
        return "url_like"
    if re.search(r"\.(ps1|vbs|js|jse|bat|cmd|hta)(\s|\"|'|$)", text):
        return "script_path"
    if re.search(r"[a-z0-9+/]{80,}={0,2}", text):
        return "base64_like"
    return "other_cmd"


def bucket_safe_command_lex_v1_3b(command_line: object) -> str:
    """Return a safe lexical command token without exposing raw command text."""
    text = str(command_line or "").strip().lower()
    norm = normalize_windows_path(text)
    if not text or text == "none":
        return "cmdarg|other_safe"
    if "qwinsta" in text:
        return "cmdtool|qwinsta"
    if re.search(r"(?:^|\s)net\s+use(?:\s|$)", text):
        return "cmdtool|net_use"
    if "schtasks" in text and (" /s " in text or "\\\\" in text):
        return "cmdtool|schtasks_remote"
    if re.search(r"(?:^|[\s\\/])wmic(?:\.exe)?(?:\s|$)", text):
        return "cmdtool|wmic"
    if "psexec" in text:
        return "cmdtool|psexec"
    if re.search(r"(?:^|[\s\\/])sc(?:\.exe)?(?:\s|$)", text):
        return "cmdtool|sc"
    if re.search(r"(?:^|[\s\\/])reg(?:\.exe)?(?:\s|$)", text) or "/reg:" in text:
        return "cmdtool|reg"
    if re.search(r"(?:^|[\s\\/])ping(?:\.exe)?(?:\s|$)", text):
        return "cmdtool|ping"
    if "rundll32" in text:
        return "cmdtool|rundll32"
    if "regsvr32" in text:
        return "cmdtool|regsvr32"
    if "-enc" in text or "encodedcommand" in text:
        return "cmdarg|encoded"
    if re.search(r"https?://", text):
        return "cmdarg|url"
    if "%temp%" in text or "\\appdata\\local\\temp\\" in norm or "\\windows\\temp\\" in norm:
        return "cmdarg|temp_path"
    if re.search(r"\.ps1(?:\s|\"|'|$)", text):
        return "cmdarg|script_ext_ps1"
    if re.search(r"\.(?:vbs|js|jse|hta)(?:\s|\"|'|$)", text):
        return "cmdarg|script_ext_vbs"
    if re.search(r"\.(?:bat|cmd)(?:\s|\"|'|$)", text):
        return "cmdarg|script_ext_bat"
    if "svchost.exe" in text and " -k " in text:
        return "cmdarg|service_arg"
    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
        return "cmdarg|ip_literal"
    if "\\\\" in text:
        return "cmdarg|unc_path"
    if re.search(r"\\bhk(?:lm|cu|cr|u|cc)\\\\", norm) or "hkey_" in text:
        return "cmdarg|registry_key"
    return "cmdarg|other_safe"


def bucket_safe_command_lex_v1_3c(command_line: object) -> str:
    """Return the expanded OpTC v1.3c safe lexical command token."""
    text = str(command_line or "").strip().lower()
    norm = normalize_windows_path(text)
    if not text or text == "none":
        return "cmdarg|other_safe"
    if "conhost.exe" in norm and "forcev1" in text:
        return "cmdhost|conhost_forcev1"
    if re.search(r"(?:^|[\s\\/])cmd(?:\.exe)?(?:\s|$)", text) and re.search(
        r"(?:^|\s)/(?:c|k)(?:\s|$)",
        text,
    ):
        return "cmdshell|cmd_c"
    if "schtasks" in text and re.search(r"(?:^|\s)/(?:query)(?:\s|$)", text):
        return "cmdtool|schtasks_query"
    if "schtasks" in text and re.search(r"(?:^|\s)/(?:delete)(?:\s|$)", text):
        return "cmdtool|schtasks_delete"
    if re.search(r"(?:^|[\s\\/])taskkill(?:\.exe)?(?:\s|$)", text):
        return "cmdtool|taskkill"
    if re.search(r"(?:^|[\s\\/])tasklist(?:\.exe)?(?:\s|$)", text):
        return "cmdtool|tasklist"
    if re.search(r"(?:^|[\s\\/])netstat(?:\.exe)?(?:\s|$)", text):
        return "cmdtool|netstat"
    if re.search(r"(?:^|[\s\\/])gpupdate(?:\.exe)?(?:\s|$)", text):
        return "cmdtool|gpupdate"
    if re.search(r"(?:^|[\s\\/])bcdedit(?:\.exe)?(?:\s|$)", text):
        return "cmdtool|bcdedit"
    if re.search(r"(?:^|[\s\\/])shutdown(?:\.exe)?(?:\s|$)", text):
        return "cmdtool|shutdown"
    if "onedrivesetup.exe" in norm:
        return "cmdtool|onedrive_setup"
    if "trustedinstaller.exe" in norm:
        return "cmdtool|trustedinstaller"
    if "sppsvc.exe" in norm:
        return "cmdtool|sppsvc"
    if any(name in norm for name in ("searchindexer.exe", "searchfilterhost.exe")):
        return "cmdtool|search_indexer"
    if "backgroundtaskhost.exe" in norm:
        return "cmdtool|background_task_host"
    if "runtimebroker.exe" in norm:
        return "cmdtool|runtime_broker"
    if "dllhost.exe" in norm:
        return "cmdtool|dllhost"
    if "wermgr.exe" in norm:
        return "cmdtool|wermgr"
    if "wmiprvse.exe" in norm:
        return "cmdtool|wmiprvse"
    if re.search(r"(?:^|[\s\\/])pythonw?(?:\.exe)?(?:\s|$)", text) and re.search(
        r"\.py(?:\s|\"|'|$)",
        text,
    ):
        return "cmdtool|python_script"
    if "-embedding" in text or " /embedding" in text:
        return "cmdarg|embedding"
    if "-servername:" in text and "appx" in text:
        return "cmdarg|servername_appx"
    if re.search(r"(?:^|\s)/(?:uninstall|uninst)(?:\s|$)", text):
        return "cmdarg|uninstall"
    if "scheduler" in text or "update task" in text or "updatetask" in text:
        return "cmdarg|update_task"
    if "/target:computer" in text or "/target:user" in text or "gpupdate" in text:
        return "cmdarg|group_policy"
    if "ngen" in text or "ngenprocess" in text:
        return "cmdarg|ngen"
    return bucket_safe_command_lex_v1_3b(command_line)


def is_ntfs_metadata(path: object) -> bool:
    """Return whether a path points at NTFS metadata pseudo-files."""
    text = normalize_windows_path(path)
    return any(marker in text for marker in ("\\$mft", "\\$logfile", "\\$extend\\$usnjrnl"))


def file_extension(path: object) -> str:
    """Return a normalized file extension bucket."""
    name = normalize_windows_path(path).rsplit("\\", 1)[-1]
    match = re.search(r"(\.[a-z0-9_]{1,12})$", name)
    return match.group(1) if match else "no_ext"


def is_directory_like_path(path: object) -> bool:
    """Return whether a no-extension Windows path is likely a directory node."""
    text = normalize_windows_path(path)
    if not text or is_ntfs_metadata(text) or ":$" in text or file_extension(text) != "no_ext":
        return False
    parts = [part for part in text.split("\\") if part]
    if not parts:
        return False
    joined = "\\" + "\\".join(parts)
    directory_markers = (
        "\\programdata",
        "\\users",
        "\\windows",
        "\\program files",
        "\\appdata",
        "\\microsoft",
        "\\policies",
    )
    if any(marker in joined for marker in directory_markers):
        return True
    return parts[-1] in {"programdata", "users", "windows", "appdata", "microsoft", "policies"}


def bucket_file_family_v1(path: object) -> str:
    """Bucket file path family for OpTC Windows v1.1."""
    text = normalize_windows_path(path)
    if is_ntfs_metadata(text):
        return "ntfs_metadata"
    if is_directory_like_path(text):
        return "directory_node"
    if "\\windows\\system32\\drivers\\" in text:
        return "driver"
    if text.startswith("c:\\windows\\system32\\") or "\\windows\\system32\\" in text:
        return "system32"
    if text.startswith("c:\\windows\\syswow64\\") or "\\windows\\syswow64\\" in text:
        return "syswow64"
    if "\\windows\\temp\\" in text:
        return "windows_temp"
    if "\\appdata\\local\\temp\\" in text:
        return "user_temp"
    if "\\downloads\\" in text:
        return "downloads"
    if "\\appdata\\roaming\\" in text:
        return "appdata_roaming"
    if "\\programdata\\" in text:
        return "programdata"
    if "\\program files" in text:
        return "program_files"
    if "\\documents\\" in text or "\\desktop\\" in text:
        return "user_documents"
    if "\\cache\\" in text or "\\temporary internet files\\" in text:
        return "browser_cache"
    return "file_other"


def bucket_file_ext_class_v1(path: object) -> str:
    """Bucket file extension class for OpTC Windows coarse tokens."""
    if is_directory_like_path(path):
        return "directory"
    ext = file_extension(path)
    if ext == "no_ext":
        return "no_ext"
    if ext == ".exe":
        return "exe"
    if ext == ".dll":
        return "dll"
    if ext == ".sys":
        return "driver"
    if ext in {".ps1", ".vbs", ".js", ".jse", ".bat", ".cmd", ".hta"}:
        return "script"
    if ext in {".zip", ".rar", ".7z", ".cab"}:
        return "archive"
    if ext in {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf"}:
        return "document"
    if ext in {".log", ".evtx"}:
        return "log"
    if ext in {".ini", ".yml", ".yaml", ".xml", ".json", ".config"}:
        return "config"
    if ext == ".pf":
        return "prefetch"
    if ext == ".etl":
        return "etl"
    if ext == ".pyc":
        return "pyc"
    return "other_ext"


def bucket_file_basename_v1(path: object) -> str:
    """Bucket basename shape for OpTC Windows v1.1 detail tokens."""
    if is_ntfs_metadata(path):
        return "ntfs_metadata"
    if is_directory_like_path(path):
        return "directory_node"
    family = bucket_file_family_v1(path)
    ext = file_extension(path)
    name = normalize_windows_path(path).rsplit("\\", 1)[-1]
    if ext == ".dll":
        return "known_system_dll" if family in {"system32", "syswow64"} else "unknown_dll"
    if family in {"user_temp", "windows_temp"} and ext in {".exe", ".dll", "no_ext"}:
        return "payload_like"
    if ext == ".exe":
        if "setup" in name or "install" in name or "update" in name:
            return "installer_like"
        return "exe"
    if ext in {".ps1", ".vbs", ".js", ".jse", ".bat", ".cmd", ".hta"}:
        return "script"
    if ext == ".sys":
        return "driver"
    if ext in {".zip", ".rar", ".7z", ".cab"}:
        return "archive"
    if ext in {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf"}:
        return "document"
    if ext in {".log", ".evtx"}:
        return "log"
    if ext in {".ini", ".yml", ".yaml", ".json", ".config"}:
        return "policy_xml" if "\\policies\\" in normalize_windows_path(path) else "config"
    if ext == ".pf":
        return "prefetch"
    if "tmp" in name or "temp" in name:
        return "temp_like"
    stem = name[: -len(ext)] if ext != "no_ext" else name
    if re.fullmatch(r"[a-f0-9]{12,}", stem) or re.fullmatch(r"[a-z0-9]{16,}", stem):
        return "random_like"
    return "other_file"


def bucket_ip_scope_v1(ip_value: object) -> str:
    """Bucket IP scope for OpTC Windows netflow."""
    text = str(ip_value or "").strip()
    if not text:
        return "unknown"
    try:
        ip_obj = ipaddress.ip_address(text)
    except ValueError:
        return "unknown"
    if ip_obj.is_loopback:
        return "loopback"
    if ip_obj.is_multicast:
        return "multicast"
    if ip_obj.version == 6 and ip_obj.is_link_local:
        return "link_local"
    if ip_obj.version == 4 and text.startswith("169.254."):
        return "link_local"
    if ip_obj.is_private:
        return "private"
    return "public"


def bucket_optc_remote_scope_v1_3(ip_value: object) -> str:
    """Return OpTC v1.3 remote scope with 142.20/16 treated as internal."""
    scope = bucket_ip_scope_v1(ip_value)
    text = str(ip_value or "").strip()
    if scope in {"loopback", "multicast", "link_local", "private", "unknown"}:
        return scope
    try:
        ip_obj = ipaddress.ip_address(text)
    except ValueError:
        return "unknown"
    if ip_obj.version == 4 and text.startswith("142.20."):
        return "internal"
    return "external"


def bucket_service_v1(port: object) -> str:
    """Bucket Windows-relevant destination service ports."""
    text = str(port or "").strip()
    if not text.isdigit():
        return "unknown_port"
    value = int(text)
    common = {
        53: "dns",
        80: "http",
        135: "rpc",
        139: "netbios",
        443: "https",
        445: "smb",
        3389: "rdp",
        5355: "llmnr",
        5353: "mdns",
    }
    if value in common:
        return common[value]
    if 1 <= value <= 1023:
        return "system"
    if 1024 <= value <= 49151:
        return "registered"
    if 49152 <= value <= 65535:
        return "ephemeral"
    return "unknown_port"


def bucket_netflow_direction_v1(src_ip: object, dst_ip: object) -> str:
    """Infer direction from source and destination endpoint scopes."""
    src_scope = bucket_ip_scope_v1(src_ip)
    dst_scope = bucket_ip_scope_v1(dst_ip)
    if src_scope == "loopback" or dst_scope == "loopback":
        return "loopback"
    if dst_scope == "multicast":
        return "multicast"
    if src_scope == "link_local" or dst_scope == "link_local":
        return "link_local"
    if src_scope == "private" and dst_scope == "public":
        return "outbound"
    if src_scope == "public" and dst_scope == "private":
        return "inbound"
    if src_scope == "private" and dst_scope == "private":
        return "internal"
    return "unknown_direction"


def bucket_remote_bucket_v1(ip_value: object) -> str:
    """Bucket remote endpoint without preserving exact IP."""
    scope = bucket_ip_scope_v1(ip_value)
    text = str(ip_value or "").strip()
    if scope == "loopback":
        return "loopback"
    if scope == "multicast":
        return "multicast_group"
    if scope == "link_local":
        return "link_local_group"
    try:
        ip_obj = ipaddress.ip_address(text)
    except ValueError:
        return "unknown_remote"
    if ip_obj.version == 4:
        parts = text.split(".")
        if len(parts) >= 2:
            return f"{scope}_{parts[0]}_{parts[1]}_x_x"
    if ip_obj.version == 6:
        return f"{scope}_ipv6_group"
    return "unknown_remote"


def normalize_remote_ip_token_v1_3(ip_value: object) -> str:
    """Return exact remote IP as a stable token segment for OpTC v1.3."""
    text = str(ip_value or "").strip()
    if not text:
        return "unknown_remote"
    try:
        ip_obj = ipaddress.ip_address(text)
    except ValueError:
        return "unknown_remote"
    if ip_obj.version == 4:
        return sanitize_segment(text.replace(".", "_"))
    return sanitize_segment(text.replace(":", "_"))


def _optc_v13_port_int(value: object) -> int | None:
    text = str(value or "").strip()
    if not text.isdigit():
        return None
    port = int(text)
    if 0 <= port <= 65535:
        return port
    return None


def _optc_v13_is_ephemeral_port(value: object) -> bool:
    port = _optc_v13_port_int(value)
    return bool(port is not None and 49152 <= port <= 65535)


def _optc_v13_service_rank(port: object) -> int:
    service = bucket_service_v1(port)
    if service in {
        "dns",
        "http",
        "https",
        "rdp",
        "smb",
        "rpc",
        "netbios",
        "llmnr",
        "mdns",
    }:
        return 0
    if service in {"system", "registered"}:
        return 1
    if service == "ephemeral":
        return 3
    return 2


def optc_remote_endpoint_v1_3(
    *,
    src_addr: object = "",
    src_port: object = "",
    dst_addr: object = "",
    dst_port: object = "",
) -> tuple[str, str, str]:
    """Choose the OpTC v1.3 remote endpoint as scope, service, and exact IP token."""
    endpoints = (
        {
            "addr": str(src_addr or "").strip(),
            "port": src_port,
            "scope": bucket_optc_remote_scope_v1_3(src_addr),
            "side": "src",
        },
        {
            "addr": str(dst_addr or "").strip(),
            "port": dst_port,
            "scope": bucket_optc_remote_scope_v1_3(dst_addr),
            "side": "dst",
        },
    )
    special_scopes = {"multicast", "link_local", "loopback"}
    for endpoint in endpoints:
        if endpoint["scope"] in special_scopes:
            return (
                str(endpoint["scope"]),
                bucket_service_v1(endpoint["port"]),
                normalize_remote_ip_token_v1_3(endpoint["addr"]),
            )
    candidates = []
    for endpoint in endpoints:
        scope = str(endpoint["scope"])
        if scope == "unknown" or not endpoint["addr"]:
            continue
        rank = 0 if scope == "external" else 1
        rank += _optc_v13_service_rank(endpoint["port"])
        if _optc_v13_is_ephemeral_port(endpoint["port"]):
            rank += 4
        if endpoint["side"] == "dst":
            rank -= 1
        candidates.append((rank, endpoint))
    if not candidates:
        return ("unknown", "unknown_port", "unknown_remote")
    endpoint = sorted(candidates, key=lambda item: item[0])[0][1]
    service_port = endpoint["port"]
    other_port = dst_port if endpoint["side"] == "src" else src_port
    if bucket_service_v1(service_port) == "ephemeral" and bucket_service_v1(other_port) != "ephemeral":
        service_port = other_port
    return (
        str(endpoint["scope"]),
        bucket_service_v1(service_port),
        normalize_remote_ip_token_v1_3(endpoint["addr"]),
    )


def optc_process_natural_tokens_v1(
    path: object = "",
    cmd: object = "",
    *,
    detail: bool = True,
    parent_role: object = "parent_unknown",
) -> tuple[str, ...]:
    """Return OpTC Windows process tokens for detail or coarse mode."""
    image = process_image_name(path or cmd)
    role = bucket_process_role(image)
    if not detail:
        return ("process", "windows", role)
    token = (
        "process",
        "windows",
        role,
        image,
        bucket_process_path_v1(path or cmd),
        bucket_command_line_v1(cmd),
        sanitize_segment(parent_role),
    )
    return tuple(sanitize_segment(part) for part in token)


def optc_process_natural_tokens_v1_3(
    path: object = "",
    cmd: object = "",
    *,
    detail: bool = True,
    parent_role: object = "parent_unknown",
) -> tuple[str, ...]:
    """Return OpTC Windows v1.3 process tokens without path or parent role."""
    del parent_role
    image = process_image_name(path or cmd)
    role = bucket_process_role(image)
    if not detail:
        return ("process", "windows", role)
    token = ("process", "windows", role, image, bucket_command_line_v1(cmd))
    return tuple(sanitize_segment(part) for part in token)


def optc_process_natural_tokens_v1_3b(
    path: object = "",
    cmd: object = "",
    *,
    detail: bool = True,
    parent_role: object = "parent_unknown",
) -> tuple[str, ...]:
    """Return OpTC Windows v1.3b process tokens with safe command lexical bucket."""
    del parent_role
    image = process_image_name(path or cmd)
    role = bucket_process_role(image)
    if not detail:
        return ("process", "windows", role)
    token = (
        "process",
        "windows",
        role,
        image,
        bucket_command_line_v1(cmd),
        bucket_safe_command_lex_v1_3b(cmd),
    )
    return tuple(sanitize_segment(part) for part in token)


def optc_process_natural_tokens_v1_3c(
    path: object = "",
    cmd: object = "",
    *,
    detail: bool = True,
    parent_role: object = "parent_unknown",
) -> tuple[str, ...]:
    """Return OpTC Windows v1.3c process tokens with expanded safe lexical bucket."""
    del parent_role
    image = process_image_name(path or cmd)
    role = bucket_process_role(image)
    if not detail:
        return ("process", "windows", role)
    token = (
        "process",
        "windows",
        role,
        image,
        bucket_command_line_v1(cmd),
        bucket_safe_command_lex_v1_3c(cmd),
    )
    return tuple(sanitize_segment(part) for part in token)


def optc_file_natural_tokens_v1(path: object = "", *, detail: bool = True) -> tuple[str, ...]:
    """Return OpTC Windows file tokens for detail or coarse mode."""
    if not detail:
        return (
            "file",
            "windows",
            bucket_file_family_v1(path),
            bucket_file_ext_class_v1(path),
        )
    return (
        "file",
        "windows",
        bucket_file_family_v1(path),
        file_extension(path),
        bucket_file_basename_v1(path),
    )


def optc_netflow_natural_tokens_v1(
    src_addr: object = "",
    dst_addr: object = "",
    dst_port: object = "",
    *,
    detail: bool = True,
) -> tuple[str, ...]:
    """Return OpTC Windows netflow tokens for detail or coarse mode."""
    base = (
        "netflow",
        "windows",
        bucket_netflow_direction_v1(src_addr, dst_addr),
        bucket_ip_scope_v1(dst_addr),
        bucket_service_v1(dst_port),
    )
    if not detail:
        return base
    return (*base, bucket_remote_bucket_v1(dst_addr))


def optc_netflow_natural_tokens_v1_3(
    src_addr: object = "",
    dst_addr: object = "",
    dst_port: object = "",
    *,
    src_port: object = "",
    detail: bool = True,
) -> tuple[str, ...]:
    """Return OpTC Windows v1.3 CADETS/THEIA-style remote endpoint tokens."""
    scope, service, remote_ip = optc_remote_endpoint_v1_3(
        src_addr=src_addr,
        src_port=src_port,
        dst_addr=dst_addr,
        dst_port=dst_port,
    )
    base = ("netflow", "windows", scope, service)
    if not detail:
        return tuple(sanitize_segment(part) for part in base)
    return tuple(sanitize_segment(part) for part in (*base, remote_ip))


def _optc_netflow_canonical_id(canonical_key: object) -> int:
    """Return a deterministic positive node id for an OpTC netflow canonical key."""
    key = sanitize_token(canonical_key)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return int(9_000_000_000_000_000_000 + (value % 900_000_000_000_000_000))


def optc_netflow_canonical_key_v1_3(
    *,
    src_addr: object = "",
    src_port: object = "",
    dst_addr: object = "",
    dst_port: object = "",
) -> str:
    """Return the stable v1.3 remote endpoint canonical key for a netflow node."""
    return "|".join(
        optc_netflow_natural_tokens_v1_3(
            src_addr=src_addr,
            src_port=src_port,
            dst_addr=dst_addr,
            dst_port=dst_port,
            detail=True,
        ),
    )


def optc_netflow_canonical_id_v1_3(canonical_key: object) -> int:
    """Return a deterministic positive node id for a v1.3 netflow canonical key."""
    return _optc_netflow_canonical_id(canonical_key)


def optc_registry_natural_tokens_v1(
    registry_path: object = "",
    *,
    detail: bool = True,
) -> tuple[str, ...]:
    """Return registry tokens only when a registry path source exists."""
    text = normalize_windows_path(registry_path)
    if not text:
        return ()
    family = "registry_other"
    if "\\currentversion\\run" in text:
        family = "run_key"
    elif "\\currentcontrolset\\services\\" in text:
        family = "service_key"
    elif "\\software\\classes\\" in text:
        family = "com_classes"
    elif "\\image file execution options\\" in text:
        family = "ifeo"
    elif "\\winlogon\\" in text:
        family = "winlogon"
    elif "\\wmi\\" in text or "\\wbem\\" in text:
        family = "wmi"
    if not detail:
        return ("registry", "windows", family)
    return ("registry", "windows", family, "hive_unknown", "value_unknown")


def _kind_for_side(row: Mapping[str, object], side: str) -> str:
    object_kind = sanitize_segment(row.get("object_type", "unknown"))
    default = "process" if side == "src" else object_kind
    return sanitize_segment(row.get(f"{side}_kind", default))


def _node_tokens_for_row_side(
    row: Mapping[str, object],
    side: str,
    *,
    detail: bool,
    netflow_version: str = "v1",
) -> tuple[str, ...]:
    kind = _kind_for_side(row, side)
    if kind == "process":
        if netflow_version == "v1_3c":
            return optc_process_natural_tokens_v1_3c(
                row.get(f"{side}_process_path", ""),
                row.get(f"{side}_process_cmd", ""),
                detail=detail,
            )
        if netflow_version == "v1_3b":
            return optc_process_natural_tokens_v1_3b(
                row.get(f"{side}_process_path", ""),
                row.get(f"{side}_process_cmd", ""),
                detail=detail,
            )
        if netflow_version == "v1_3":
            return optc_process_natural_tokens_v1_3(
                row.get(f"{side}_process_path", ""),
                row.get(f"{side}_process_cmd", ""),
                detail=detail,
            )
        return optc_process_natural_tokens_v1(
            row.get(f"{side}_process_path", ""),
            row.get(f"{side}_process_cmd", ""),
            detail=detail,
        )
    if kind == "file":
        return optc_file_natural_tokens_v1(row.get(f"{side}_file_path", ""), detail=detail)
    if kind == "netflow":
        if netflow_version in {"v1_3", "v1_3b", "v1_3c"}:
            return optc_netflow_natural_tokens_v1_3(
                row.get("src_addr", row.get("local_ip", "")),
                row.get(f"{side}_addr", row.get("dst_addr", row.get("remote_ip", ""))),
                row.get(f"{side}_port", row.get("dst_port", row.get("remote_port", ""))),
                src_port=row.get("src_port", row.get("local_port", "")),
                detail=detail,
            )
        return optc_netflow_natural_tokens_v1(
            row.get("src_addr", row.get("local_ip", "")),
            row.get(f"{side}_addr", row.get("dst_addr", row.get("remote_ip", ""))),
            row.get(f"{side}_port", row.get("dst_port", row.get("remote_port", ""))),
            detail=detail,
        )
    if kind == "registry":
        return optc_registry_natural_tokens_v1(row.get(f"{side}_registry_path", ""), detail=detail)
    return (kind or "unknown",)


def optc_residual_text_v1_detail(row: Mapping[str, object]) -> str:
    """Return residual text for OpTC Windows v1 detail mode."""
    return _optc_residual_text_v1(row, detail=True)


def optc_residual_text_v1_coarse(row: Mapping[str, object]) -> str:
    """Return residual text for OpTC Windows v1 coarse mode."""
    return _optc_residual_text_v1(row, detail=False)


def optc_residual_text_v1_3_detail(row: Mapping[str, object]) -> str:
    """Return residual text for OpTC Windows v1.3 detail mode."""
    return _optc_residual_text_v1(row, detail=True, netflow_version="v1_3")


def optc_residual_text_v1_3_coarse(row: Mapping[str, object]) -> str:
    """Return residual text for OpTC Windows v1.3 coarse mode."""
    return _optc_residual_text_v1(row, detail=False, netflow_version="v1_3")


def optc_residual_text_v1_3b_detail(row: Mapping[str, object]) -> str:
    """Return residual text for OpTC Windows v1.3b detail mode."""
    return _optc_residual_text_v1(row, detail=True, netflow_version="v1_3b")


def optc_residual_text_v1_3b_coarse(row: Mapping[str, object]) -> str:
    """Return residual text for OpTC Windows v1.3b coarse mode."""
    return _optc_residual_text_v1(row, detail=False, netflow_version="v1_3b")


def optc_residual_text_v1_3c_detail(row: Mapping[str, object]) -> str:
    """Return residual text for OpTC Windows v1.3c detail mode."""
    return _optc_residual_text_v1(row, detail=True, netflow_version="v1_3c")


def optc_residual_text_v1_3c_coarse(row: Mapping[str, object]) -> str:
    """Return residual text for OpTC Windows v1.3c coarse mode."""
    return _optc_residual_text_v1(row, detail=False, netflow_version="v1_3c")


def _optc_residual_text_v1(
    row: Mapping[str, object],
    *,
    detail: bool,
    netflow_version: str = "v1",
) -> str:
    action = sanitize_segment(row.get("action", "unknown"), max_len=80)
    tokens = (
        *_node_tokens_for_row_side(
            row,
            "src",
            detail=detail,
            netflow_version=netflow_version,
        ),
        action,
        *_node_tokens_for_row_side(
            row,
            "dst",
            detail=detail,
            netflow_version=netflow_version,
        ),
    )
    return " ".join(token for token in tokens if str(token).strip())
