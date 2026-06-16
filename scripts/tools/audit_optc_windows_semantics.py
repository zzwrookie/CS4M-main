"""Read-only OpTC Windows semantic detail audit.

This script inspects restored OpTC PostgreSQL databases and writes diagnostic
CSV/Markdown artifacts for Windows process, file, registry, and netflow semantic
design. It does not train models, run inference, modify tokenizer code, or create
persistent database objects.
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import psycopg2
import psycopg2.extensions
import psycopg2.extras

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cs4m.semantics import optc_windows as optc_sem


DEFAULT_DATABASES = ("optc_051", "optc_201", "optc_501")
DEFAULT_GT_PATHS = {
    "optc_051": Path("ground_truth/h051/node_h051_0925.csv"),
    "optc_201": Path("ground_truth/h201/node_h201_0923.csv"),
    "optc_501": Path("ground_truth/h501/node_h501_0924.csv"),
}
FORBIDDEN_TOKEN_WORDS = ("malicious", "attack", "ground_truth", "label", "gt")


@dataclass(frozen=True)
class GroundTruthRow:
    """One audit-only ground-truth row."""

    dataset: str
    entity_id: str
    raw_uuid: str
    raw_detail: str


@dataclass(frozen=True)
class AuditEntity:
    """Label-free entity detail extracted for audit aggregation."""

    dataset: str
    entity_id: str
    entity_kind: str
    raw_detail: str
    semantic_token: str
    family: str
    subtype: str
    bucket: str
    example: str


def normalize_windows_path(value: str) -> str:
    """Return a lower-cased Windows-ish path with backslash separators."""
    text = str(value or "").strip().replace("/", "\\").lower()
    return re.sub(r"\\+", r"\\", text)


def process_image_name(value: str) -> str:
    """Extract the image basename from a Windows or device path."""
    text = str(value or "").strip().replace('"', "")
    if not text:
        return "unknown_process"
    match = re.search(r"(?i)(?:^|[\\/])([^\\/\"\s]+\.exe)\b", text)
    if match:
        return match.group(1).lower()
    first = text.split()[0] if " " in text and "\\" not in text and "/" not in text else text
    first = first.replace("/", "\\")
    name = first.rsplit("\\", 1)[-1].lower()
    return name or "unknown_process"


def bucket_process_role(image: str) -> str:
    """Bucket a Windows process image into a label-free role."""
    name = process_image_name(image)
    if name in {"powershell.exe", "pwsh.exe"}:
        return "powershell"
    if name in {"cmd.exe", "conhost.exe"}:
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
    }:
        return "lolbin"
    if name in {"services.exe", "svchost.exe", "lsass.exe", "csrss.exe", "wininit.exe"}:
        return "system_service"
    if name in {"system", "system.exe", "smss.exe"}:
        return "system_process"
    if name.endswith(".exe"):
        return "user_app"
    return "other_process"


def bucket_process_path(image: str) -> str:
    """Bucket a process path."""
    text = normalize_windows_path(image)
    if "\\windows\\system32\\" in text:
        return "system32"
    if "\\windows\\syswow64\\" in text:
        return "syswow64"
    if "\\appdata\\local\\temp\\" in text or "\\windows\\temp\\" in text:
        return "user_temp"
    if "\\program files" in text:
        return "program_files"
    if "\\users\\" in text:
        return "user_profile"
    if text in {"", "none"}:
        return "unknown_path"
    return "process_path_other"


def bucket_command_line(command_line: str) -> str:
    """Bucket Windows command-line detail into suspicious/shape patterns."""
    text = str(command_line or "").strip().lower()
    if not text or text == "none":
        return "empty_cmd"
    if "-enc" in text or "encodedcommand" in text:
        return "encoded_command"
    if "downloadstring" in text or "downloadfile" in text:
        return "download_string"
    if "invoke-expression" in text or "iex " in text:
        return "invoke_expression"
    if re.search(r"https?://", text):
        return "url_like"
    if re.search(r"\.(ps1|vbs|js|jse|bat|cmd|hta)(\s|\"|'|$)", text):
        return "script_path"
    if re.search(r"[a-z0-9+/]{80,}={0,2}", text):
        return "base64_like"
    return "other_cmd"


def bucket_process_path_v1(image: str) -> str:
    """Bucket process path for v1 audit tokens."""
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


def bucket_command_line_v1(command_line: str) -> str:
    """Bucket command line for v1 audit tokens."""
    text = str(command_line or "").strip().lower()
    if not text or text == "none":
        return "empty_cmd"
    if "-enc" in text or "encodedcommand" in text:
        return "encoded_command"
    if "downloadstring" in text or "downloadfile" in text:
        return "download_string"
    if "invoke-expression" in text or "iex " in text:
        return "invoke_expression"
    if re.search(r"https?://", text):
        return "url_like"
    if (
        "%temp%" in text
        or "\\appdata\\local\\temp\\" in normalize_windows_path(text)
        or "\\windows\\temp\\" in normalize_windows_path(text)
    ):
        return "temp_exec_arg"
    if any(marker in text for marker in ("qwinsta", "psexec", "wmic", "net use")):
        return "remote_admin_arg"
    if "schtasks" in text and (" /s " in text or "\\\\") in text:
        return "remote_admin_arg"
    if "svchost.exe" in text and " -k " in text:
        return "service_arg"
    if any(marker in text for marker in ("winword.exe", "excel.exe", "powerpnt.exe", "outlook.exe")):
        return "office_child_arg"
    if re.search(r"\.(ps1|vbs|js|jse|bat|cmd|hta)(\s|\"|'|$)", text):
        return "script_path"
    if re.search(r"[a-z0-9+/]{80,}={0,2}", text):
        return "base64_like"
    return "other_cmd"


def bucket_file_family(path: str) -> str:
    """Bucket a Windows file path into a coarse family."""
    text = normalize_windows_path(path)
    if text.startswith("c:\\windows\\system32\\") or "\\windows\\system32\\" in text:
        if "\\drivers\\" in text:
            return "driver"
        return "system32"
    if text.startswith("c:\\windows\\syswow64\\") or "\\windows\\syswow64\\" in text:
        return "syswow64"
    if "\\appdata\\local\\temp\\" in text or "\\windows\\temp\\" in text:
        return "user_temp"
    if "\\appdata\\roaming\\" in text:
        return "appdata_roaming"
    if "\\downloads\\" in text:
        return "downloads"
    if text.startswith("c:\\programdata\\") or "\\programdata\\" in text:
        return "programdata"
    if "\\documents\\" in text or "\\desktop\\" in text:
        return "user_documents"
    if text.startswith("c:\\users\\") or "\\users\\" in text:
        return "user_profile"
    return "file_other"


def is_ntfs_metadata(path: str) -> bool:
    """Return whether a file path points at NTFS metadata pseudo-files."""
    text = normalize_windows_path(path)
    return any(marker in text for marker in ("\\$mft", "\\$logfile", "\\$extend\\$usnjrnl"))


def bucket_file_family_v1(path: str) -> str:
    """Bucket file path family for v1 audit tokens."""
    text = normalize_windows_path(path)
    if is_ntfs_metadata(text):
        return "ntfs_metadata"
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


def bucket_file_ext_class_v1(path: str) -> str:
    """Bucket file extension class for v1 coarse tokens."""
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


def file_extension(path: str) -> str:
    """Return a normalized file extension bucket."""
    name = normalize_windows_path(path).rsplit("\\", 1)[-1]
    match = re.search(r"(\.[a-z0-9_]{1,12})$", name)
    return match.group(1) if match else "no_ext"


def bucket_file_basename(path: str) -> str:
    """Bucket a file basename shape."""
    name = normalize_windows_path(path).rsplit("\\", 1)[-1]
    ext = file_extension(name)
    if ext == ".dll":
        return "dll"
    if ext == ".exe":
        return "exe"
    if ext in {".ps1", ".vbs", ".js", ".jse", ".bat", ".cmd", ".hta"}:
        return "script"
    if ext == ".sys":
        return "driver"
    if ext in {".zip", ".rar", ".7z", ".cab"}:
        return "archive"
    if ext in {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf"}:
        return "document"
    if "tmp" in name or "temp" in name:
        return "temp_like"
    stem = name[: -len(ext)] if ext != "no_ext" else name
    if re.fullmatch(r"[a-f0-9]{12,}", stem) or re.fullmatch(r"[a-z0-9]{16,}", stem):
        return "random_like"
    return "other_file"


def bucket_file_basename_v1(path: str) -> str:
    """Bucket basename for v1 detail tokens."""
    if is_ntfs_metadata(path):
        return "ntfs_metadata"
    family = bucket_file_family_v1(path)
    ext = file_extension(path)
    name = normalize_windows_path(path).rsplit("\\", 1)[-1]
    if ext == ".dll":
        if family in {"system32", "syswow64"}:
            return "known_system_dll"
        return "unknown_dll"
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
    if ext in {".ini", ".yml", ".yaml", ".xml", ".json", ".config"}:
        return "config"
    if ext == ".pf":
        return "prefetch"
    if "tmp" in name or "temp" in name:
        return "temp_like"
    stem = name[: -len(ext)] if ext != "no_ext" else name
    if re.fullmatch(r"[a-f0-9]{12,}", stem) or re.fullmatch(r"[a-z0-9]{16,}", stem):
        return "random_like"
    if family in {"user_temp", "windows_temp"} and ext in {".exe", ".dll", "no_ext"}:
        return "payload_like"
    return "other_file"


def bucket_registry_family(path: str) -> str:
    """Bucket Windows registry path family."""
    text = normalize_windows_path(path)
    if "\\currentversion\\run" in text:
        return "run_key"
    if "\\currentcontrolset\\services\\" in text:
        return "service_key"
    if "\\software\\classes\\" in text:
        return "com_classes"
    if "\\image file execution options\\" in text:
        return "ifeo"
    if "\\winlogon\\" in text:
        return "winlogon"
    if "\\wmi\\" in text or "\\wbem\\" in text:
        return "wmi"
    return "registry_other"


def registry_hive(path: str) -> str:
    """Bucket registry hive."""
    text = normalize_windows_path(path)
    if text.startswith("hkey_local_machine") or text.startswith("hklm"):
        return "hklm"
    if text.startswith("hkey_current_user") or text.startswith("hkcu"):
        return "hkcu"
    if text.startswith("hkey_classes_root") or text.startswith("hkcr"):
        return "hkcr"
    if text.startswith("hkey_users") or text.startswith("hku"):
        return "hku"
    return "hive_unknown"


def ip_scope(ip_value: str) -> str:
    """Return a coarse IP scope."""
    text = str(ip_value or "").strip()
    if not text:
        return "unknown"
    try:
        ip_obj = ipaddress.ip_address(text)
    except ValueError:
        return "unknown"
    if ip_obj.is_loopback:
        return "loopback"
    if ip_obj.is_private:
        return "private"
    return "public_or_external"


def bucket_ip_scope_v1(ip_value: str) -> str:
    """Bucket IP scope for v1 audit tokens."""
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


def bucket_windows_port(port: str) -> str:
    """Bucket Windows-relevant destination port values."""
    text = str(port or "").strip()
    if not text.isdigit():
        return "port_unknown"
    value = int(text)
    common = {
        53: "port_53_dns",
        80: "port_80_http",
        135: "port_135_rpc",
        139: "port_139_netbios",
        443: "port_443_https",
        445: "port_445_smb",
        3389: "port_3389_rdp",
    }
    if value in common:
        return common[value]
    if 1 <= value <= 1023:
        return "port_system"
    if 1024 <= value <= 49151:
        return "port_registered"
    if 49152 <= value <= 65535:
        return "port_ephemeral"
    return "port_unknown"


def bucket_service_v1(port: str) -> str:
    """Bucket Windows-relevant service ports for v1."""
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


def bucket_netflow_direction_v1(src_ip: str, dst_ip: str) -> str:
    """Infer v1 direction from endpoint scopes."""
    src_scope = bucket_ip_scope_v1(src_ip)
    dst_scope = bucket_ip_scope_v1(dst_ip)
    if src_scope == "loopback" or dst_scope == "loopback":
        return "loopback"
    if src_scope == "link_local" or dst_scope == "link_local":
        return "link_local"
    if dst_scope == "multicast":
        return "multicast"
    if src_scope == "private" and dst_scope == "public":
        return "outbound"
    if src_scope == "public" and dst_scope == "private":
        return "inbound"
    if src_scope == "private" and dst_scope == "private":
        return "internal"
    return "unknown_direction"


def bucket_remote_bucket_v1(ip_value: str) -> str:
    """Bucket remote endpoint without exact IP for v1."""
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
            prefix = f"{parts[0]}_{parts[1]}_x_x"
            return f"{scope}_{prefix}"
    if ip_obj.version == 6:
        return f"{scope}_ipv6_group"
    return "unknown_remote"


def endpoint_bucket(ip_value: str) -> str:
    """Return a fallback endpoint bucket without exact IP."""
    scope = ip_scope(ip_value)
    if scope != "public_or_external":
        return scope
    parts = str(ip_value or "").split(".")
    if len(parts) >= 2 and all(part.isdigit() for part in parts[:2]):
        return f"public_{parts[0]}_{parts[1]}_x_x"
    return "public_external"


def sanitize_token(token: str) -> str:
    """Remove forbidden audit/leakage words from a candidate token."""
    parts = []
    for part in str(token or "").lower().split("|"):
        if part in FORBIDDEN_TOKEN_WORDS:
            part = "redacted"
        part = re.sub(r"[0-9a-f]{24,}", "hashlike", part)
        part = re.sub(r"\b\d{9,}\b", "numlike", part)
        parts.append(part)
    return "|".join(parts)


def process_token(image: str, command_line: str) -> str:
    """Build a temporary process audit token."""
    name = process_image_name(image or command_line)
    token = "|".join(
        [
            "process",
            "windows",
            bucket_process_role(name),
            name,
            bucket_process_path(image or command_line),
            bucket_command_line(command_line),
            "parent_unknown",
        ],
    )
    return sanitize_token(token)


def process_token_v1(image: str, command_line: str) -> str:
    """Build a v1 temporary process audit token."""
    name = process_image_name(image or command_line)
    token = "|".join(
        [
            "process",
            "windows",
            bucket_process_role(name),
            name,
            bucket_process_path_v1(image or command_line),
            bucket_command_line_v1(command_line),
            "parent_unknown",
        ],
    )
    return sanitize_token(token)


def file_token(path: str) -> str:
    """Build a temporary file audit token."""
    token = "|".join(
        [
            "file",
            "windows",
            bucket_file_family(path),
            file_extension(path),
            bucket_file_basename(path),
        ],
    )
    return sanitize_token(token)


def file_token_v1(path: str) -> str:
    """Build a v1 temporary file audit token."""
    token = "|".join(
        [
            "file",
            "windows",
            bucket_file_family_v1(path),
            file_extension(path),
            bucket_file_basename_v1(path),
        ],
    )
    return sanitize_token(token)


def registry_token(path: str) -> str:
    """Build a temporary registry audit token."""
    token = "|".join(
        [
            "registry",
            "windows",
            bucket_registry_family(path),
            registry_hive(path),
        ],
    )
    return sanitize_token(token)


def registry_token_v1(path: str) -> str:
    """Build a v1 temporary registry audit token."""
    return registry_token(path)


def netflow_token(src_ip: str, dst_ip: str, dst_port: str) -> str:
    """Build a temporary netflow audit token."""
    direction = "outbound" if ip_scope(src_ip) == "private" and ip_scope(dst_ip) != "private" else "unknown_direction"
    token = "|".join(
        [
            "netflow",
            "windows",
            direction,
            ip_scope(dst_ip),
            bucket_windows_port(dst_port),
            endpoint_bucket(dst_ip),
        ],
    )
    return sanitize_token(token)


def netflow_token_v1(src_ip: str, dst_ip: str, dst_port: str) -> str:
    """Build a v1 temporary netflow audit token."""
    token = "|".join(
        [
            "netflow",
            "windows",
            bucket_netflow_direction_v1(src_ip, dst_ip),
            bucket_ip_scope_v1(dst_ip),
            bucket_service_v1(dst_port),
            bucket_remote_bucket_v1(dst_ip),
        ],
    )
    return sanitize_token(token)


def entity_v1(entity: AuditEntity) -> AuditEntity:
    """Return a v1-tokenized copy of an audit entity."""
    if entity.entity_kind == "process":
        tokens = optc_sem.optc_process_natural_tokens_v1(
            entity.raw_detail,
            entity.raw_detail,
            detail=True,
        )
        token = optc_sem.sanitize_token("|".join(tokens))
        return AuditEntity(
            dataset=entity.dataset,
            entity_id=entity.entity_id,
            entity_kind=entity.entity_kind,
            raw_detail=entity.raw_detail,
            semantic_token=token,
            family=optc_sem.bucket_process_role(optc_sem.process_image_name(entity.raw_detail)),
            subtype=optc_sem.process_image_name(entity.raw_detail),
            bucket=optc_sem.bucket_command_line_v1(entity.raw_detail),
            example=entity.example,
        )
    if entity.entity_kind == "file":
        tokens = optc_sem.optc_file_natural_tokens_v1(entity.raw_detail, detail=True)
        token = optc_sem.sanitize_token("|".join(tokens))
        return AuditEntity(
            dataset=entity.dataset,
            entity_id=entity.entity_id,
            entity_kind=entity.entity_kind,
            raw_detail=entity.raw_detail,
            semantic_token=token,
            family=optc_sem.bucket_file_family_v1(entity.raw_detail),
            subtype=optc_sem.bucket_file_ext_class_v1(entity.raw_detail),
            bucket=optc_sem.bucket_file_basename_v1(entity.raw_detail),
            example=entity.example,
        )
    if entity.entity_kind == "netflow":
        src_ip, src_port, dst_ip, dst_port = parse_netflow(entity.raw_detail)
        del src_port
        tokens = optc_sem.optc_netflow_natural_tokens_v1(
            src_ip,
            dst_ip,
            dst_port,
            detail=True,
        )
        token = optc_sem.sanitize_token("|".join(tokens))
        return AuditEntity(
            dataset=entity.dataset,
            entity_id=entity.entity_id,
            entity_kind=entity.entity_kind,
            raw_detail=entity.raw_detail,
            semantic_token=token,
            family=optc_sem.bucket_ip_scope_v1(dst_ip),
            subtype=optc_sem.bucket_service_v1(dst_port),
            bucket=optc_sem.bucket_remote_bucket_v1(dst_ip),
            example=entity.example,
        )
    return AuditEntity(
        dataset=entity.dataset,
        entity_id=entity.entity_id,
        entity_kind=entity.entity_kind,
        raw_detail=entity.raw_detail,
        semantic_token=registry_token_v1(entity.raw_detail),
        family=entity.family,
        subtype=entity.subtype,
        bucket=entity.bucket,
        example=entity.example,
    )


def parse_ground_truth_csv(path: Path, dataset: str) -> list[GroundTruthRow]:
    """Parse OpTC GT CSV rows, using the final column as node index."""
    rows: list[GroundTruthRow] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[-1].lower() in {"node_idx", "index", "entity_id"}:
                continue
            rows.append(
                GroundTruthRow(
                    dataset=dataset,
                    entity_id=str(row[-1]).strip(),
                    raw_uuid=str(row[0]).strip() if row else "",
                    raw_detail=str(row[1]).strip() if len(row) > 1 else "",
                ),
            )
    return rows


def is_candidate_column(table_name: str, column_name: str) -> bool:
    """Return whether a schema column is relevant for OpTC/Windows audit."""
    text = f"{table_name} {column_name}".lower()
    needles = (
        "optc",
        "ecar",
        "event",
        "node",
        "subject",
        "object",
        "actor",
        "process",
        "cmd",
        "image",
        "path",
        "file",
        "registry",
        "ip",
        "addr",
        "port",
        "hostname",
        "action",
        "type",
        "uuid",
        "id",
        "label",
        "dataset",
    )
    return any(needle in text for needle in needles)


def connect(args: argparse.Namespace, database: str) -> psycopg2.extensions.connection:
    """Open a PostgreSQL connection."""
    return psycopg2.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password or os.environ.get("PGPASSWORD"),
        dbname=database,
    )


def query_dicts(
    conn: psycopg2.extensions.connection,
    sql: str,
    params: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    """Run a read-only query and return dictionaries."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params or ())
        return [dict(row) for row in cur.fetchall()]


def iter_dicts(
    conn: psycopg2.extensions.connection,
    sql: str,
    params: Sequence[object] | None = None,
    *,
    fetch_size: int = 50000,
) -> Iterable[dict[str, object]]:
    """Stream a read-only query as dictionaries."""
    cursor_name = f"audit_cursor_{abs(hash(sql))}"
    with conn.cursor(name=cursor_name, cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.itersize = fetch_size
        cur.execute(sql, params or ())
        while True:
            rows = cur.fetchmany(fetch_size)
            if not rows:
                break
            for row in rows:
                yield dict(row)


def discover_schema(
    conn: psycopg2.extensions.connection,
    dataset: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Discover schema tables, columns, and candidate columns."""
    tables = query_dicts(
        conn,
        """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog','information_schema')
        ORDER BY table_schema, table_name
        """,
    )
    columns = query_dicts(
        conn,
        """
        SELECT table_schema, table_name, column_name, data_type, ordinal_position
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog','information_schema')
        ORDER BY table_schema, table_name, ordinal_position
        """,
    )
    for row in tables + columns:
        row["dataset"] = dataset
    candidates = [row for row in columns if is_candidate_column(str(row["table_name"]), str(row["column_name"]))]
    return tables, columns, candidates


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """Write rows to CSV with stable headers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def table_exists(tables: Sequence[Mapping[str, object]], name: str) -> bool:
    """Return whether a public table exists in discovered schema rows."""
    return any(str(row.get("table_name")) == name for row in tables)


def limit_clause(limit: int) -> str:
    """Return a SQL LIMIT clause for optional entity caps."""
    return "" if limit <= 0 else f" LIMIT {int(limit)}"


def read_process_rows(conn: psycopg2.extensions.connection, dataset: str, limit: int) -> list[AuditEntity]:
    """Read process/subject node details."""
    sql = (
        "SELECT index_id::text AS entity_id, COALESCE(node_uuid::text, '') AS uuid, "
        "COALESCE(path::text, '') AS path, COALESCE(cmd::text, '') AS cmd_line "
        "FROM subject_node_table ORDER BY index_id"
        + limit_clause(limit)
    )
    entities: list[AuditEntity] = []
    for row in iter_dicts(conn, sql):
        path = str(row.get("path") or "")
        cmd = str(row.get("cmd_line") or "")
        raw = path if cmd in {"", "None", "none"} else f"{path} {cmd}".strip()
        token = process_token(path, cmd)
        entities.append(
            AuditEntity(
                dataset=dataset,
                entity_id=str(row["entity_id"]),
                entity_kind="process",
                raw_detail=raw,
                semantic_token=token,
                family=bucket_process_role(path or cmd),
                subtype=process_image_name(path or cmd),
                bucket=bucket_command_line(cmd),
                example=raw,
            ),
        )
    return entities


def read_file_rows(conn: psycopg2.extensions.connection, dataset: str, limit: int) -> list[AuditEntity]:
    """Read file node details."""
    sql = (
        "SELECT index_id::text AS entity_id, COALESCE(node_uuid::text, '') AS uuid, "
        "COALESCE(path::text, '') AS path FROM file_node_table ORDER BY index_id"
        + limit_clause(limit)
    )
    entities: list[AuditEntity] = []
    for row in iter_dicts(conn, sql):
        path = str(row.get("path") or "")
        entities.append(
            AuditEntity(
                dataset=dataset,
                entity_id=str(row["entity_id"]),
                entity_kind="file",
                raw_detail=path,
                semantic_token=file_token(path),
                family=bucket_file_family(path),
                subtype=file_extension(path),
                bucket=bucket_file_basename(path),
                example=path,
            ),
        )
    return entities


NETFLOW_RE = re.compile(r"(?P<src>[^:\s]+):(?P<src_port>\d+)->(?P<dst>[^:\s]+):(?P<dst_port>\d+)")


def split_endpoint(value: str) -> tuple[str, str]:
    """Split an IPv4/IPv6 endpoint string at the final port separator."""
    text = str(value or "").strip()
    if ":" not in text:
        return text, ""
    host, port = text.rsplit(":", 1)
    if not port.isdigit():
        return text, ""
    return host, port


def parse_netflow(value: str) -> tuple[str, str, str, str]:
    """Parse OpTC netflow detail into src/dst IP and ports."""
    text = str(value or "")
    if "->" in text:
        src_raw, dst_raw = text.split("->", 1)
        src, src_port = split_endpoint(src_raw)
        dst, dst_port = split_endpoint(dst_raw)
        if src and dst and src_port and dst_port:
            return src, src_port, dst, dst_port
    match = NETFLOW_RE.search(text)
    if not match:
        return "", "", "", ""
    return (
        match.group("src"),
        match.group("src_port"),
        match.group("dst"),
        match.group("dst_port"),
    )


def read_netflow_rows(conn: psycopg2.extensions.connection, dataset: str, limit: int) -> list[AuditEntity]:
    """Read netflow node details."""
    sql = (
        "SELECT index_id::text AS entity_id, COALESCE(node_uuid::text, '') AS uuid, "
        "COALESCE(dst_addr::text, '') AS remote_ip, COALESCE(src_addr::text, '') AS local_ip, "
        "COALESCE(dst_port::text, '') AS remote_port, COALESCE(src_port::text, '') AS local_port "
        "FROM netflow_node_table ORDER BY index_id"
        + limit_clause(limit)
    )
    entities: list[AuditEntity] = []
    for row in iter_dicts(conn, sql):
        src_ip = str(row.get("local_ip") or "")
        dst_ip = str(row.get("remote_ip") or "")
        src_port = str(row.get("local_port") or "")
        dst_port = str(row.get("remote_port") or "")
        raw = f"{src_ip}:{src_port}->{dst_ip}:{dst_port}"
        entities.append(
            AuditEntity(
                dataset=dataset,
                entity_id=str(row["entity_id"]),
                entity_kind="netflow",
                raw_detail=raw,
                semantic_token=netflow_token(src_ip, dst_ip, dst_port),
                family=ip_scope(dst_ip),
                subtype=bucket_windows_port(dst_port),
                bucket=endpoint_bucket(dst_ip),
                example=raw,
            ),
        )
    return entities


def read_registry_rows(conn: psycopg2.extensions.connection, dataset: str, limit: int) -> list[AuditEntity]:
    """Read registry details when a source exists; restored dumps have none."""
    _ = conn, limit
    return [
        AuditEntity(
            dataset=dataset,
            entity_id="",
            entity_kind="registry",
            raw_detail="missing_registry_source",
            semantic_token=registry_token(""),
            family="missing_registry_source",
            subtype="hive_unknown",
            bucket="missing_registry_source",
            example="No dedicated registry table discovered in restored dump.",
        ),
    ]


def aggregate_distribution(entities: Sequence[AuditEntity], kind: str) -> list[dict[str, object]]:
    """Aggregate distribution rows for one entity kind."""
    grouped: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for entity in entities:
        if entity.entity_kind != kind:
            continue
        key = (entity.dataset, entity.family, entity.subtype, entity.bucket, entity.semantic_token)
        row = grouped.setdefault(
            key,
            {
                "dataset": entity.dataset,
                "entity_kind": kind,
                "family": entity.family,
                "subtype": entity.subtype,
                "bucket": entity.bucket,
                "semantic_token": entity.semantic_token,
                "node_count": 0,
                "example": entity.example,
            },
        )
        row["node_count"] = int(row["node_count"]) + 1
    return sorted(grouped.values(), key=lambda row: (-int(row["node_count"]), str(row["dataset"]), str(row["family"])))


def update_distribution(
    grouped: dict[tuple[str, str, str, str, str, str], dict[str, object]],
    entity: AuditEntity,
) -> None:
    """Update per-dataset and combined distribution counters."""
    for dataset in (entity.dataset, "combined"):
        key = (
            dataset,
            entity.entity_kind,
            entity.family,
            entity.subtype,
            entity.bucket,
            entity.semantic_token,
        )
        row = grouped.setdefault(
            key,
            {
                "dataset": dataset,
                "entity_kind": entity.entity_kind,
                "family": entity.family,
                "subtype": entity.subtype,
                "bucket": entity.bucket,
                "semantic_token": entity.semantic_token,
                "node_count": 0,
                "example": entity.example,
            },
        )
        row["node_count"] = int(row["node_count"]) + 1


def update_collision(
    groups: dict[tuple[str, str, str], dict[str, object]],
    entity: AuditEntity,
    gt_ids: Mapping[str, set[str]],
    focus_rows: list[dict[str, object]],
    mapped_gt: set[tuple[str, str]],
    gt_raw_by_key: Mapping[tuple[str, str], GroundTruthRow],
) -> None:
    """Update label-aware diagnostics after tokenization."""
    is_gt = entity.entity_id in gt_ids.get(entity.dataset, set())
    key = (entity.dataset, entity.entity_kind, entity.semantic_token)
    row = groups.setdefault(
        key,
        {
            "dataset": entity.dataset,
            "entity_kind": entity.entity_kind,
            "semantic_token": entity.semantic_token,
            "total_count": 0,
            "gt_count": 0,
            "benign_count": 0,
            "example_gt_raw_detail": "",
            "example_benign_raw_detail": "",
        },
    )
    row["total_count"] = int(row["total_count"]) + 1
    if is_gt:
        row["gt_count"] = int(row["gt_count"]) + 1
        mapped_gt.add((entity.dataset, entity.entity_id))
        gt_row = gt_raw_by_key.get((entity.dataset, entity.entity_id))
        if not row["example_gt_raw_detail"]:
            row["example_gt_raw_detail"] = entity.raw_detail
        focus_rows.append(
            {
                "dataset": entity.dataset,
                "entity_id": entity.entity_id,
                "entity_kind": entity.entity_kind,
                "semantic_token": entity.semantic_token,
                "entity_raw_detail": entity.raw_detail,
                "gt_raw_detail": gt_row.raw_detail if gt_row is not None else "",
            },
        )
    else:
        row["benign_count"] = int(row["benign_count"]) + 1
        if not row["example_benign_raw_detail"]:
            row["example_benign_raw_detail"] = entity.raw_detail


def finalize_collision_rows(
    groups: Mapping[tuple[str, str, str], Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return sorted collision rows that contain both GT and benign entities."""
    rows: list[dict[str, object]] = []
    for group in groups.values():
        gt_count = int(group["gt_count"])
        benign_count = int(group["benign_count"])
        total = int(group["total_count"])
        if gt_count == 0 or benign_count == 0:
            continue
        row = dict(group)
        row["gt_ratio"] = f"{gt_count / total:.6f}" if total else "0.000000"
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -int(row["gt_count"]),
            -int(row["benign_count"]),
            -int(row["total_count"]),
            str(row["semantic_token"]),
        ),
    )
    return rows


def rows_by_kind(
    grouped: Mapping[tuple[str, str, str, str, str, str], Mapping[str, object]],
    kind: str,
) -> list[dict[str, object]]:
    """Extract sorted distribution rows for one kind."""
    rows = [dict(row) for row in grouped.values() if row.get("entity_kind") == kind]
    rows.sort(key=lambda row: (-int(row["node_count"]), str(row["dataset"]), str(row["family"])))
    return rows


def combined_entities(entities: Sequence[AuditEntity]) -> list[AuditEntity]:
    """Return combined-copy entities for aggregate outputs."""
    return [
        AuditEntity(
            dataset="combined",
            entity_id=entity.entity_id,
            entity_kind=entity.entity_kind,
            raw_detail=entity.raw_detail,
            semantic_token=entity.semantic_token,
            family=entity.family,
            subtype=entity.subtype,
            bucket=entity.bucket,
            example=entity.example,
        )
        for entity in entities
    ]


def build_collision_rows(
    entities: Sequence[AuditEntity],
    gt_rows_by_dataset: Mapping[str, Sequence[GroundTruthRow]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Build label-aware diagnostics after label-free tokenization."""
    gt_ids = {
        dataset: {row.entity_id for row in rows}
        for dataset, rows in gt_rows_by_dataset.items()
    }
    gt_raw_by_key = {
        (dataset, row.entity_id): row.raw_detail
        for dataset, rows in gt_rows_by_dataset.items()
        for row in rows
    }
    entity_by_key = {(entity.dataset, entity.entity_id): entity for entity in entities}
    groups: dict[tuple[str, str, str], dict[str, object]] = {}
    for entity in entities:
        key = (entity.dataset, entity.entity_kind, entity.semantic_token)
        is_gt = entity.entity_id in gt_ids.get(entity.dataset, set())
        row = groups.setdefault(
            key,
            {
                "dataset": entity.dataset,
                "entity_kind": entity.entity_kind,
                "semantic_token": entity.semantic_token,
                "total_count": 0,
                "gt_count": 0,
                "benign_count": 0,
                "example_gt_raw_detail": "",
                "example_benign_raw_detail": "",
            },
        )
        row["total_count"] = int(row["total_count"]) + 1
        if is_gt:
            row["gt_count"] = int(row["gt_count"]) + 1
            if not row["example_gt_raw_detail"]:
                row["example_gt_raw_detail"] = entity.raw_detail
        else:
            row["benign_count"] = int(row["benign_count"]) + 1
            if not row["example_benign_raw_detail"]:
                row["example_benign_raw_detail"] = entity.raw_detail
    collision_rows = []
    for row in groups.values():
        total = int(row["total_count"])
        gt_count = int(row["gt_count"])
        benign_count = int(row["benign_count"])
        if gt_count == 0 or benign_count == 0:
            continue
        out = dict(row)
        out["gt_ratio"] = f"{gt_count / total:.6f}" if total else "0.000000"
        collision_rows.append(out)
    collision_rows.sort(
        key=lambda row: (
            -int(row["gt_count"]),
            -int(row["benign_count"]),
            -int(row["total_count"]),
            str(row["semantic_token"]),
        ),
    )

    focus_rows: list[dict[str, object]] = []
    missing_rows: list[dict[str, object]] = []
    for dataset, rows in gt_rows_by_dataset.items():
        for gt_row in rows:
            entity = entity_by_key.get((dataset, gt_row.entity_id))
            if entity is None:
                missing_rows.append(
                    {
                        "dataset": dataset,
                        "entity_id": gt_row.entity_id,
                        "raw_uuid": gt_row.raw_uuid,
                        "gt_raw_detail": gt_row.raw_detail,
                    },
                )
                continue
            focus_rows.append(
                {
                    "dataset": dataset,
                    "entity_id": gt_row.entity_id,
                    "entity_kind": entity.entity_kind,
                    "semantic_token": entity.semantic_token,
                    "entity_raw_detail": entity.raw_detail,
                    "gt_raw_detail": gt_raw_by_key.get((dataset, gt_row.entity_id), ""),
                },
            )
    return collision_rows, focus_rows, missing_rows


def write_schema_outputs(
    out_dir: Path,
    table_rows: Sequence[Mapping[str, object]],
    column_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> None:
    """Write schema CSV outputs."""
    write_csv(out_dir / "schema_tables.csv", table_rows, ["dataset", "table_schema", "table_name", "table_type"])
    write_csv(
        out_dir / "schema_columns.csv",
        column_rows,
        ["dataset", "table_schema", "table_name", "column_name", "data_type", "ordinal_position"],
    )
    write_csv(
        out_dir / "candidate_columns.csv",
        candidate_rows,
        ["dataset", "table_schema", "table_name", "column_name", "data_type", "ordinal_position"],
    )


def summarize_top(rows: Sequence[Mapping[str, object]], limit: int = 8) -> str:
    """Format top distribution rows for markdown."""
    lines = []
    for row in rows[:limit]:
        lines.append(
            f"- `{row.get('dataset')}` `{row.get('family')}` `{row.get('subtype')}` "
            f"`{row.get('bucket')}`: {row.get('node_count')}",
        )
    return "\n".join(lines) if lines else "- No rows."


def write_reports(
    out_dir: Path,
    table_rows: Sequence[Mapping[str, object]],
    process_rows: Sequence[Mapping[str, object]],
    file_rows: Sequence[Mapping[str, object]],
    registry_rows: Sequence[Mapping[str, object]],
    netflow_rows: Sequence[Mapping[str, object]],
    collision_rows: Sequence[Mapping[str, object]],
    missing_rows: Sequence[Mapping[str, object]],
) -> None:
    """Write audit and extractor design markdown reports."""
    table_summary = Counter(str(row.get("table_name")) for row in table_rows)
    report = [
        "# OPTC Windows Semantic Audit Report",
        "",
        "## Scope",
        "",
        "This is a read-only audit over `optc_051`, `optc_201`, and `optc_501` on PostgreSQL 5433. Action types were not modified.",
        "",
        "## Real Table Structure",
        "",
        "Observed table names:",
    ]
    for name, count in sorted(table_summary.items()):
        report.append(f"- `{name}` observed in {count} database(s)")
    report.extend(
        [
            "",
            "Expected entity mapping from restored dumps:",
            "",
            "- event table: `event_table`",
            "- process/entity table: `subject_node_table`",
            "- file/entity table: `file_node_table`",
            "- netflow/entity table: `netflow_node_table`",
            "- registry source: no dedicated registry table observed; emitted as `missing_registry_source`",
            "- ground truth source: CSV files under `ground_truth/h051`, `ground_truth/h201`, and `ground_truth/h501`",
            "- vocab / embedding / selected tables: not observed in restored OpTC dumps unless listed in `schema_tables.csv`",
            "",
            "## Process Distribution",
            "",
            summarize_top(process_rows),
            "",
            "## File Distribution",
            "",
            summarize_top(file_rows),
            "",
            "## Registry Distribution",
            "",
            summarize_top(registry_rows),
            "",
            "## Netflow Distribution",
            "",
            summarize_top(netflow_rows),
            "",
            "## GT / Benign Collision Summary",
            "",
        ],
    )
    if collision_rows:
        for row in collision_rows[:10]:
            report.append(
                f"- `{row.get('dataset')}` `{row.get('entity_kind')}` `{row.get('semantic_token')}`: "
                f"GT={row.get('gt_count')} benign={row.get('benign_count')} total={row.get('total_count')}"
            )
    else:
        report.append("- No GT/benign same-token collisions found in audited entities.")
    report.extend(
        [
            "",
            "## Priorities",
            "",
            "Windows process and netflow detail should be prioritized first because GT examples include subject/process and flow nodes, while registry is not represented by a dedicated restored table.",
            "",
            "Event-local context is likely needed beyond node-level detail for high-frequency Windows processes such as `svchost.exe`, `cmd.exe`, `powershell.exe`, and for common public service ports such as 80, 443, and 3389.",
            "",
            "## Safety",
            "",
            "- Action remains the current 10 event types and was not redesigned.",
            "- Ground truth was used only for post-tokenization diagnostics.",
            "- No persistent database objects were created by the script.",
            f"- Missing GT mappings: {len(missing_rows)}",
        ],
    )
    (out_dir / "OPTC_Windows_Semantic_Audit_Report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    design = [
        "# OPTC Windows Semantic Extractor Design",
        "",
        "## Design Boundary",
        "",
        "This design changes only Windows entity/detail semantics. It does not change action types, alert policy, thresholds, Word2Vec training, or Phase3G inference.",
        "",
        "## Process Token",
        "",
        "`process|windows|<role>|<image_name>|<path_bucket>|<cmdline_bucket>|<parent_role>`",
        "",
        "Recommended role buckets: `shell`, `powershell`, `script_host`, `browser`, `office`, `lolbin`, `system_service`, `system_process`, `user_app`, `other_process`.",
        "",
        "Command-line buckets should include `encoded_command`, `download_string`, `invoke_expression`, `script_path`, `url_like`, `base64_like`, `empty_cmd`, and `other_cmd`.",
        "",
        "## File Token",
        "",
        "`file|windows|<family>|<extension>|<basename_bucket>`",
        "",
        "Recommended family buckets: `system32`, `syswow64`, `user_temp`, `downloads`, `appdata_roaming`, `programdata`, `user_documents`, `driver`, and `file_other`.",
        "",
        "## Registry Token",
        "",
        "`registry|windows|<family>|<hive_bucket>`",
        "",
        "Recommended families: `run_key`, `service_key`, `com_classes`, `ifeo`, `winlogon`, `wmi`, and `registry_other`. The restored dumps did not expose a dedicated registry table, so implementation should first identify registry-bearing event/object fields.",
        "",
        "## Netflow Token",
        "",
        "`netflow|windows|<direction>|<ip_scope>|<port_bucket>|<subnet_or_domain_bucket>`",
        "",
        "Main model tokens should use private/public/loopback scope and service-aware port buckets. Exact IP/domain should be debug-only with fallback buckets.",
        "",
        "## Leakage Guard",
        "",
        "The production extractor must not use `malicious`, `attack`, `ground_truth`, `gt`, `label`, attack windows, or long random identifiers as semantic tokens.",
        "",
        "## Need for Event-Local Context",
        "",
        "Node-level Windows detail is not sufficient for common binaries and ports. Event-local context should be designed next using action, subject/object type, parent process, command-line bucket, path bucket, and endpoint direction, while keeping action definitions unchanged.",
    ]
    (out_dir / "OPTC_Windows_Semantic_Extractor_Design.md").write_text("\n".join(design) + "\n", encoding="utf-8")


def metric_value(rows: Sequence[Mapping[str, object]], token: str) -> int:
    """Return node_count for a distribution token across combined rows."""
    total = 0
    for row in rows:
        if row.get("dataset") == "combined" and row.get("semantic_token") == token:
            total += int(row.get("node_count", 0) or 0)
    return total


def write_v1_reports(
    out_dir: Path,
    process_rows_v1: Sequence[Mapping[str, object]],
    file_rows_v1: Sequence[Mapping[str, object]],
    registry_rows_v1: Sequence[Mapping[str, object]],
    netflow_rows_v1: Sequence[Mapping[str, object]],
    collision_rows_v1: Sequence[Mapping[str, object]],
    missing_rows: Sequence[Mapping[str, object]],
    comparison_rows: Sequence[Mapping[str, object]],
) -> None:
    """Write v1 audit and extractor design reports."""
    report = [
        "# OPTC Windows Semantic v1 Audit Report",
        "",
        "## Scope",
        "",
        "This v1 refinement is read-only and keeps the current 10 action types unchanged.",
        "",
        "## Top v1 Process Distribution",
        "",
        summarize_top(process_rows_v1),
        "",
        "## Top v1 File Distribution",
        "",
        summarize_top(file_rows_v1),
        "",
        "## Top v1 Registry Distribution",
        "",
        summarize_top(registry_rows_v1),
        "",
        "## Top v1 Netflow Distribution",
        "",
        summarize_top(netflow_rows_v1),
        "",
        "## Top v1 GT / Benign Collisions",
        "",
    ]
    for row in collision_rows_v1[:10]:
        report.append(
            f"- `{row.get('dataset')}` `{row.get('entity_kind')}` `{row.get('semantic_token')}`: "
            f"GT={row.get('gt_count')} benign={row.get('benign_count')} total={row.get('total_count')}"
        )
    report.extend(["", "## v0 / v1 Comparison", ""])
    for row in comparison_rows:
        report.append(
            f"- `{row.get('metric')}`: v0={row.get('v0_value')} v1={row.get('v1_value')} "
            f"note={row.get('note')}"
        )
    report.extend(
        [
            "",
            "## Decision Notes",
            "",
            "- File NTFS metadata is separated from generic no-extension file buckets.",
            "- Netflow multicast/link-local/service buckets make LLMNR, HTTP, HTTPS, and RDP behavior explicit.",
            "- Registry remains no-source because no dedicated registry table is present in restored dumps.",
            f"- GT missing mappings: {len(missing_rows)}",
            "- Next stage can implement production `optc_windows_v1_detail` / `optc_windows_v1_coarse` only after reviewing this evidence.",
        ],
    )
    (out_dir / "OPTC_Windows_Semantic_v1_Audit_Report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    design = [
        "# OPTC Windows Semantic v1 Extractor Design",
        "",
        "## Boundary",
        "",
        "This is the proposed production extractor shape, not an implemented runtime tokenizer. Action definitions remain unchanged.",
        "",
        "## Coarse Tokens",
        "",
        "- `process|windows|<role>`",
        "- `file|windows|<family>|<ext_class>`",
        "- `netflow|windows|<direction>|<ip_scope>|<service_bucket>`",
        "- `registry|windows|<family>` once registry source fields are found",
        "",
        "## Detail Tokens",
        "",
        "- `process|windows|<role>|<image_name>|<path_bucket>|<cmd_bucket>|<parent_role>`",
        "- `file|windows|<family>|<extension>|<basename_bucket>`",
        "- `netflow|windows|<direction>|<ip_scope>|<service_bucket>|<remote_bucket>`",
        "- `registry|windows|<family>|<hive_bucket>|<value_bucket>` once registry source fields are found",
        "",
        "## Required Production Tests",
        "",
        "- `$Mft`, `$LogFile`, and `$Extend/$UsnJrnl` map to `ntfs_metadata`.",
        "- `224.0.0.252` and `ff02::1:3` map to multicast LLMNR buckets.",
        "- `80`, `443`, and `3389` map to `http`, `https`, and `rdp`.",
        "- Exact IPs and random strings do not become primary model tokens.",
        "- Forbidden words are absent from all primary tokens.",
    ]
    (out_dir / "OPTC_Windows_Semantic_v1_Extractor_Design.md").write_text("\n".join(design) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5433)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default=os.environ.get("PGPASSWORD", ""))
    parser.add_argument("--databases", nargs="+", default=list(DEFAULT_DATABASES))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-entities-per-kind", type=int, default=0)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    """Run the read-only OpTC Windows semantic audit."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"outputs/diagnostics/optc_windows_semantic_audit_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_tables: list[dict[str, object]] = []
    all_columns: list[dict[str, object]] = []
    all_candidates: list[dict[str, object]] = []
    distribution_groups: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}
    distribution_groups_v1: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}
    collision_groups: dict[tuple[str, str, str], dict[str, object]] = {}
    collision_groups_v1: dict[tuple[str, str, str], dict[str, object]] = {}
    focus_rows: list[dict[str, object]] = []
    focus_rows_v1: list[dict[str, object]] = []
    gt_rows_by_dataset = {
        database: parse_ground_truth_csv(DEFAULT_GT_PATHS[database], database)
        for database in args.databases
        if database in DEFAULT_GT_PATHS
    }
    gt_ids = {
        dataset: {row.entity_id for row in rows}
        for dataset, rows in gt_rows_by_dataset.items()
    }
    gt_raw_by_key = {
        (dataset, row.entity_id): row
        for dataset, rows in gt_rows_by_dataset.items()
        for row in rows
    }
    mapped_gt: set[tuple[str, str]] = set()

    for database in args.databases:
        with connect(args, database) as conn:
            conn.set_session(readonly=True, autocommit=False)
            tables, columns, candidates = discover_schema(conn, database)
            all_tables.extend(tables)
            all_columns.extend(columns)
            all_candidates.extend(candidates)
            if not table_exists(tables, "subject_node_table"):
                raise RuntimeError(f"{database}: missing subject_node_table")
            if not table_exists(tables, "file_node_table"):
                raise RuntimeError(f"{database}: missing file_node_table")
            if not table_exists(tables, "netflow_node_table"):
                raise RuntimeError(f"{database}: missing netflow_node_table")
            for reader in (read_process_rows, read_file_rows, read_netflow_rows, read_registry_rows):
                for entity in reader(conn, database, args.max_entities_per_kind):
                    update_distribution(distribution_groups, entity)
                    if entity.entity_kind != "registry":
                        update_collision(
                            collision_groups,
                            entity,
                            gt_ids,
                            focus_rows,
                            mapped_gt,
                            gt_raw_by_key,
                        )
                    v1_entity = entity_v1(entity)
                    update_distribution(distribution_groups_v1, v1_entity)
                    if v1_entity.entity_kind != "registry":
                        update_collision(
                            collision_groups_v1,
                            v1_entity,
                            gt_ids,
                            focus_rows_v1,
                            mapped_gt,
                            gt_raw_by_key,
                        )

    write_schema_outputs(out_dir, all_tables, all_columns, all_candidates)

    process_rows = rows_by_kind(distribution_groups, "process")
    file_rows = rows_by_kind(distribution_groups, "file")
    registry_rows = rows_by_kind(distribution_groups, "registry")
    netflow_rows = rows_by_kind(distribution_groups, "netflow")
    process_rows_v1 = rows_by_kind(distribution_groups_v1, "process")
    file_rows_v1 = rows_by_kind(distribution_groups_v1, "file")
    registry_rows_v1 = rows_by_kind(distribution_groups_v1, "registry")
    netflow_rows_v1 = rows_by_kind(distribution_groups_v1, "netflow")
    distribution_fields = [
        "dataset",
        "entity_kind",
        "family",
        "subtype",
        "bucket",
        "semantic_token",
        "node_count",
        "example",
    ]
    write_csv(out_dir / "process_distribution.csv", process_rows, distribution_fields)
    write_csv(out_dir / "file_distribution.csv", file_rows, distribution_fields)
    write_csv(out_dir / "registry_distribution.csv", registry_rows, distribution_fields)
    write_csv(out_dir / "netflow_distribution.csv", netflow_rows, distribution_fields)
    write_csv(out_dir / "process_distribution_v1.csv", process_rows_v1, distribution_fields)
    write_csv(out_dir / "file_distribution_v1.csv", file_rows_v1, distribution_fields)
    write_csv(out_dir / "registry_distribution_v1.csv", registry_rows_v1, distribution_fields)
    write_csv(out_dir / "netflow_distribution_v1.csv", netflow_rows_v1, distribution_fields)

    collision_rows = finalize_collision_rows(collision_groups)
    collision_rows_v1 = finalize_collision_rows(collision_groups_v1)
    missing_rows = [
        {
            "dataset": dataset,
            "entity_id": row.entity_id,
            "raw_uuid": row.raw_uuid,
            "gt_raw_detail": row.raw_detail,
        }
        for dataset, rows in gt_rows_by_dataset.items()
        for row in rows
        if (dataset, row.entity_id) not in mapped_gt
    ]
    write_csv(
        out_dir / "semantic_collision_top.csv",
        collision_rows,
        [
            "dataset",
            "entity_kind",
            "semantic_token",
            "total_count",
            "gt_count",
            "benign_count",
            "gt_ratio",
            "example_gt_raw_detail",
            "example_benign_raw_detail",
        ],
    )
    write_csv(
        out_dir / "semantic_collision_top_v1.csv",
        collision_rows_v1,
        [
            "dataset",
            "entity_kind",
            "semantic_token",
            "total_count",
            "gt_count",
            "benign_count",
            "gt_ratio",
            "example_gt_raw_detail",
            "example_benign_raw_detail",
        ],
    )
    write_csv(
        out_dir / "gt_focus_context.csv",
        focus_rows,
        ["dataset", "entity_id", "entity_kind", "semantic_token", "entity_raw_detail", "gt_raw_detail"],
    )
    write_csv(
        out_dir / "gt_focus_context_v1.csv",
        focus_rows_v1,
        ["dataset", "entity_id", "entity_kind", "semantic_token", "entity_raw_detail", "gt_raw_detail"],
    )
    write_csv(
        out_dir / "gt_missing_mapping.csv",
        missing_rows,
        ["dataset", "entity_id", "raw_uuid", "gt_raw_detail"],
    )
    write_reports(out_dir, all_tables, process_rows, file_rows, registry_rows, netflow_rows, collision_rows, missing_rows)
    comparison_rows = [
        {
            "metric": "combined_file_other_no_ext_other_file",
            "v0_value": metric_value(file_rows, "file|windows|file_other|no_ext|other_file"),
            "v1_value": metric_value(file_rows_v1, "file|windows|file_other|no_ext|other_file"),
            "note": "should drop after ntfs_metadata split",
        },
        {
            "metric": "combined_ntfs_metadata",
            "v0_value": 0,
            "v1_value": sum(
                int(row.get("node_count", 0) or 0)
                for row in file_rows_v1
                if row.get("dataset") == "combined" and row.get("family") == "ntfs_metadata"
            ),
            "note": "new v1 file family",
        },
        {
            "metric": "combined_v1_llmnr_multicast",
            "v0_value": 0,
            "v1_value": sum(
                int(row.get("node_count", 0) or 0)
                for row in netflow_rows_v1
                if row.get("dataset") == "combined"
                and row.get("family") == "multicast"
                and row.get("subtype") == "llmnr"
            ),
            "note": "LLMNR should be explicit instead of generic public external",
        },
        {
            "metric": "collision_rows",
            "v0_value": len(collision_rows),
            "v1_value": len(collision_rows_v1),
            "note": "row count may rise when v1 separates interpretable buckets",
        },
        {
            "metric": "gt_missing_mapping",
            "v0_value": len(missing_rows),
            "v1_value": len(missing_rows),
            "note": "must remain zero",
        },
    ]
    write_csv(
        out_dir / "semantic_v0_v1_comparison.csv",
        comparison_rows,
        ["metric", "v0_value", "v1_value", "note"],
    )
    write_v1_reports(
        out_dir,
        process_rows_v1,
        file_rows_v1,
        registry_rows_v1,
        netflow_rows_v1,
        collision_rows_v1,
        missing_rows,
        comparison_rows,
    )
    print(out_dir)
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
