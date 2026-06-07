from __future__ import annotations

import ipaddress
import re
import shlex
from pathlib import PurePosixPath


CADETS_SEMANTIC_RULES_VERSION = "cadets_freebsd_raw_detail_v2"

INTERNAL_ENV_CIDR = ipaddress.IPv4Network("128.55.12.0/24")
PRIVATE_CIDRS = (
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("172.16.0.0/12"),
)
ZERO_CIDR = ipaddress.IPv4Network("0.0.0.0/8")
MISSING_DETAIL_VALUES = {"", "none", "null", "na", "n/a", "unknown", "path_none"}
CONTEXT_GENERIC_BASENAMES = {
    "index",
    "index_html",
    "index_shtml",
    "minion",
    "proc",
    "files",
    "file",
    "img",
    "images",
    "template",
    "cache",
    "log",
    "logs",
    "fd",
    "task",
    "stat",
    "cmdline",
    "maps",
    "environ",
    "status",
    "mounts",
    "mountinfo",
    "main_cpp",
    "main_qml",
    "build_gn",
    "qmldir",
    "include",
    "readme",
    "readme_md",
    "init___py",
    "manifest_json",
    "doc",
    "data",
    "src",
    "scripts",
    "css",
    "lib",
    "resources",
    "community",
    "video",
    "episode",
    "cast",
    "lc_messages",
}
PAYLOAD_LIKE_BASENAMES = {
    "main",
    "test",
    "tmux_1002",
    "vugefal",
    "peja72ma",
    "minions",
    "xim",
    "gtcache",
    "pass_mgr",
    "clean",
    "profile",
    "wdev",
    "xdev",
    "memtrace_so",
}
SHELL_INTERPRETERS = {
    "python",
    "python2",
    "python2_7",
    "python3",
    "perl",
    "ruby",
    "bash",
    "sh",
    "dash",
    "zsh",
    "csh",
    "tcsh",
    "php",
    "node",
}
SHELL_CONTROL_TOKENS = {"-c", "-e", "-lc", "-l", "&", "&&", "||", "|"}


def is_cadets_dataset(dataset: object) -> bool:
    """Return true when dataset uses the CADETS FreeBSD semantic policy."""
    return str(dataset or "").upper().startswith("CADETS_")


def normalize_token(value: object, max_len: int = 80) -> str:
    """Normalize one FreeBSD semantic fragment to a bounded token."""
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    return (text or "unknown")[: max(int(max_len), 1)]


def _split_command(cmd: object) -> list[str]:
    raw = str(cmd or "").strip()
    if not raw or raw.lower() in MISSING_DETAIL_VALUES:
        return []
    try:
        return shlex.split(raw)
    except ValueError:
        return raw.split()


def _parent_basename(path: object) -> str:
    raw = str(path or "").strip().rstrip("/")
    if "/" not in raw:
        return ""
    parent = raw.rsplit("/", 1)[0].rstrip("/")
    if not parent:
        return ""
    return normalize_token(PurePosixPath(parent).name, max_len=60)


def _is_numeric_pid_like(token: str) -> bool:
    return bool(re.fullmatch(r"[0-9]+", str(token or "")))


def _proc_file_detail(path: str) -> str:
    match = re.match(r"^/proc/[0-9]+/(.*)$", path)
    rest = match.group(1) if match else path.removeprefix("/proc/")
    parts = [part for part in rest.split("/") if part]
    if not parts:
        return "proc"
    return normalize_token(parts[-1], max_len=60)


def _path_detail_token(path: object) -> str:
    raw = _valid_file_path(path)
    if not raw:
        return ""
    lowered = raw.lower().rstrip("/")
    if lowered == "/":
        return "root"
    if re.match(r"^/proc/[0-9]+(/|$)", lowered):
        return _proc_file_detail(lowered)
    basename = _basename_token(raw)
    if not basename:
        return ""
    if basename in PAYLOAD_LIKE_BASENAMES:
        return basename
    parent = _parent_basename(raw)
    if basename in CONTEXT_GENERIC_BASENAMES and parent and not _is_numeric_pid_like(parent):
        return normalize_token(f"{parent}_{basename}", max_len=60)
    return basename


def _meaningful_command_arg(parts: list[str]) -> str:
    for raw_arg in parts[1:]:
        arg = str(raw_arg or "").strip()
        if not arg or arg in SHELL_CONTROL_TOKENS:
            continue
        if arg.startswith("-") or "=" in arg:
            continue
        if arg.startswith((">", "2>", "&>")) or arg.endswith(("&", ";")):
            continue
        if "/" in arg or "." in arg:
            token = _path_detail_token(arg) or normalize_token(PurePosixPath(arg).name, max_len=60)
            if token:
                return token
    for raw_arg in parts[1:]:
        arg = str(raw_arg or "").strip()
        if not arg or arg in SHELL_CONTROL_TOKENS or arg.startswith("-"):
            continue
        token = normalize_token(arg, max_len=60)
        if token and token != "unknown":
            return token
    return ""


def _process_detail_token(cmd: object) -> str:
    parts = _split_command(cmd)
    if not parts:
        return ""
    executable = normalize_token(parts[0].rsplit("/", 1)[-1].lstrip("-"), max_len=60)
    if executable in SHELL_INTERPRETERS:
        arg_token = _meaningful_command_arg(parts)
        if arg_token:
            return arg_token
    return executable


def _command_token(cmd: object) -> str:
    parts = _split_command(cmd)
    if not parts:
        return ""
    executable = parts[0].rsplit("/", 1)[-1].lstrip("-")
    return normalize_token(executable, max_len=60)


def classify_freebsd_process_nll(cmd: object) -> str:
    """Return the CADETS FreeBSD process coarse label."""
    token = _command_token(cmd)
    if not token:
        return "unknown_process"
    if token in {"nginx", "httpd", "apache"}:
        return "web_service"
    if token in {
        "imapd",
        "sendmail",
        "mail",
        "mail_local",
        "local",
        "smtpd",
        "smtp",
        "proxymap",
        "cleanup",
        "trivial_rewrite",
        "anvil",
        "pickup",
        "bounce",
        "postmap",
        "qmgr",
        "master",
        "ipop3d",
    }:
        return "mail_service"
    if token.startswith("sshd") or token == "ssh":
        return "ssh_service"
    if token in {"cron", "crond", "atrun"}:
        return "scheduler_service"
    if token in {"sh", "bash", "csh", "tcsh", "zsh"}:
        return "shell"
    if token.startswith("python") or token in {"perl", "ruby", "php_fpm"}:
        return "script_interpreter"
    if token in {
        "lsof",
        "head",
        "tail",
        "dd",
        "mv",
        "cp",
        "rm",
        "cat",
        "chmod",
        "chown",
        "ln",
        "unlink",
        "mkdir",
        "find",
        "ls",
        "mktemp",
        "stat",
        "touch",
    }:
        return "file_util"
    if token in {
        "grep",
        "egrep",
        "awk",
        "nawk",
        "sed",
        "sort",
        "wc",
        "cmp",
        "tee",
        "tr",
        "test",
        "uniq",
        "expr",
        "basename",
        "cut",
        "diff",
        "which",
    }:
        return "text_util"
    if token in {"top", "vmstat", "ps", "netstat", "sockstat", "procstat"}:
        return "system_monitor"
    if token in {"date", "sleep", "uname", "id", "whoami", "uptime", "hostname"}:
        return "core_util"
    if token in {"mlock"}:
        return "memory_util"
    if token in {
        "sysctl",
        "adjkerntz",
        "kenv",
        "dmesg",
        "newsyslog",
        "devd",
        "ntpd",
        "syslogd",
        "init",
        "kldstat",
        "lsvfs",
    }:
        return "system_admin"
    if token in {"route", "ifconfig", "dhclient", "ping", "inetd"}:
        return "network_util"
    if token in {"pfctl", "ipfw", "ipfstat"}:
        return "firewall_util"
    if token in {"wget", "links"}:
        return "network_client"
    if token in {"pkg"}:
        return "package_manager"
    if token in {"sudo", "su"}:
        return "privilege_util"
    if token in {"mount", "df"}:
        return "filesystem_admin"
    if token in {"kill", "pwait"}:
        return "process_control"
    if token in {"bzcat", "bzip2", "xz"}:
        return "compression_util"
    if token in {"resizewin", "fortune", "alpine", "screen", "less", "msgs", "tty"}:
        return "user_interactive"
    return "process_other"


def freebsd_process_detail(cmd: object, label: str | None = None) -> str:
    """Return the CADETS FreeBSD process residual detail token."""
    actual_label = label or classify_freebsd_process_nll(cmd)
    if actual_label == "unknown_process":
        return "unknown"
    return _process_detail_token(cmd) or normalize_token(actual_label, max_len=60)


def freebsd_process_nll_role(cmd: object) -> str:
    """Return the CADETS FreeBSD process NLL role."""
    return f"process|freebsd|{classify_freebsd_process_nll(cmd)}"


def freebsd_process_natural_tokens(cmd: object) -> tuple[str, ...]:
    """Return natural CADETS FreeBSD process residual tokens."""
    label = classify_freebsd_process_nll(cmd)
    return ("process", label, freebsd_process_detail(cmd, label))


def freebsd_file_nll_role() -> str:
    """Return the CADETS FreeBSD file NLL role."""
    return "file|freebsd|file"


def _valid_file_path(path: object) -> str:
    raw = str(path or "").strip()
    if raw.lower() in MISSING_DETAIL_VALUES:
        return ""
    return raw


def _basename_token(path: object) -> str:
    return normalize_token(PurePosixPath(str(path)).name, max_len=60)


def classify_freebsd_file_nll(path: object) -> str:
    """Return the CADETS FreeBSD file coarse label."""
    raw = _valid_file_path(path)
    if not raw:
        return "file"
    lowered = raw.lower()
    if raw == "/":
        return "filesystem_root"
    if raw in {
        "/bin",
        "/boot",
        "/dev",
        "/etc",
        "/home",
        "/lib",
        "/proc",
        "/root",
        "/sbin",
        "/tmp",
        "/usr",
        "/var",
    }:
        return "freebsd_root_dir"
    if lowered.startswith("/tmp/"):
        return "tmp_file"
    if lowered.startswith("/var/log/"):
        return "system_log_file"
    if lowered.startswith("/var/"):
        return "var_state_file"
    if lowered.startswith("/etc/"):
        return "system_config_file"
    if lowered.startswith("/dev/"):
        return "device_file"
    if lowered.startswith("/usr/ports/"):
        return "ports_tree_file"
    if lowered.startswith("/usr/local/"):
        return "usr_local_file"
    if lowered.startswith("/usr/bin/"):
        return "system_binary_file"
    if lowered.startswith("/usr/sbin/"):
        return "system_sbin_file"
    if lowered.startswith("/usr/lib/"):
        return "system_library_file"
    if lowered.startswith("/usr/share/"):
        return "system_share_file"
    if lowered.startswith("/usr/home/") or lowered.startswith("/home/"):
        return "user_home_file"
    if lowered.startswith("/usr/"):
        return "usr_file"
    if lowered.startswith("/bin/"):
        return "core_binary_file"
    if lowered.startswith("/sbin/"):
        return "system_sbin_file"
    if lowered.startswith("/lib/"):
        return "core_library_file"
    if lowered.startswith("/data/ufs/"):
        return "ufs_mirror_file"
    if lowered.startswith("/data/update/"):
        return "update_build_file"
    if lowered.startswith("/data/"):
        return "data_file"
    if lowered.startswith("www.") or lowered.startswith("http"):
        return "web_mirror_file"
    if lowered.endswith(".so"):
        return "native_library_file"
    return "file_other"


def freebsd_file_detail(path: object, label: str | None = None) -> str:
    """Return one coarse CADETS FreeBSD file detail token."""
    raw = _valid_file_path(path)
    if not raw:
        return "unknown"
    actual_label = label or classify_freebsd_file_nll(raw)
    lowered = raw.lower()
    name = PurePosixPath(raw).name.lower()
    detail = _path_detail_token(raw)
    if actual_label == "filesystem_root":
        return "root"
    if actual_label == "freebsd_root_dir":
        root_name = raw.strip("/").split("/", 1)[0]
        return normalize_token(f"{root_name}_root", max_len=60) if root_name else "root"
    if actual_label == "tmp_file":
        return detail or ("hidden_tmp" if name.startswith(".") else "tmp")
    if actual_label == "system_log_file":
        return detail or "system_log"
    if actual_label == "system_config_file":
        return detail or "config_file"
    if actual_label == "device_file":
        return detail or "device"
    binary_labels = {
        "core_binary_file",
        "system_binary_file",
        "system_sbin_file",
        "core_library_file",
        "system_library_file",
        "native_library_file",
    }
    if actual_label in binary_labels:
        return detail or _basename_token(raw) or "binary"
    if actual_label == "system_share_file":
        return detail or "usr_share"
    if actual_label == "usr_local_file":
        return detail or "usr_local"
    if actual_label == "ports_tree_file":
        return detail or "ports_tree"
    if actual_label == "user_home_file":
        return detail or "home_file"
    if actual_label == "web_mirror_file":
        return detail or "web_mirror"
    if actual_label == "update_build_file":
        return detail or "update_build"
    if actual_label == "ufs_mirror_file":
        return detail or "ufs_mirror"
    if actual_label == "data_file":
        return detail or "data_file"
    if actual_label == "var_state_file":
        return detail or "var_state"
    if actual_label == "file_other":
        return detail or "other"
    return detail or "other"


def freebsd_file_natural_tokens(path: object = "") -> tuple[str, ...]:
    """Return natural CADETS FreeBSD file residual tokens."""
    raw = _valid_file_path(path)
    if not raw:
        return ("file",)
    label = classify_freebsd_file_nll(raw)
    return ("file", label, freebsd_file_detail(raw, label))


def _parse_ip(value: object) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        return None


def freebsd_ip_scope(value: object) -> str:
    """Return the CADETS FreeBSD netflow IP scope token."""
    addr = _parse_ip(value)
    if addr is None:
        return "ip_unknown_or_zero"
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.is_loopback:
            return "ip_loopback"
        return "ip_public_or_external"
    if addr in ZERO_CIDR:
        return "ip_unknown_or_zero"
    if addr.is_loopback:
        return "ip_loopback"
    if addr in INTERNAL_ENV_CIDR:
        return "ip_internal_env"
    if any(addr in cidr for cidr in PRIVATE_CIDRS):
        return "ip_private"
    return "ip_public_or_external"


def freebsd_port_bucket(value: object) -> str:
    """Return the CADETS FreeBSD destination port bucket."""
    text = str(value or "").strip()
    if not text:
        return "port_unknown_or_zero"
    try:
        port = int(text, 10)
    except ValueError:
        return "port_unknown_or_zero"
    if port <= 0:
        return "port_unknown_or_zero"
    if port <= 1023:
        return "port_system"
    if port <= 49151:
        return "port_registered"
    if port <= 65535:
        return "port_ephemeral"
    return "port_invalid"


def freebsd_exact_ip_token(value: object) -> str:
    """Return exact CADETS FreeBSD IP residual token without a port token."""
    addr = _parse_ip(value)
    if addr is None:
        return "unknown_ip"
    text = str(addr).lower().replace(".", "_").replace(":", "_")
    token = re.sub(r"[^a-z0-9_]+", "_", text)
    return (token or "unknown_ip")[:80]


def freebsd_netflow_nll_role(dst_addr: object, dst_port: object) -> str:
    """Return the CADETS FreeBSD netflow NLL role."""
    return f"net|{freebsd_ip_scope(dst_addr)}|{freebsd_port_bucket(dst_port)}"


def freebsd_endpoint_detail_token(value: object, prefix: str = "") -> str:
    """Return normalized CADETS endpoint detail with an optional direction prefix."""
    addr = _parse_ip(value)
    if addr is not None:
        if isinstance(addr, ipaddress.IPv4Address) and addr in ZERO_CIDR:
            return ""
        text = str(addr).lower().replace(".", "_").replace(":", "_")
    else:
        raw = str(value or "").strip().lower()
        text = "local" if raw in {"local", "localhost"} else ""
    if not text:
        return ""
    return normalize_token(f"{prefix}{text}", max_len=80)


def freebsd_netflow_natural_tokens(dst_addr: object, src_addr: object = "") -> tuple[str, ...]:
    """Return natural CADETS FreeBSD netflow residual tokens."""
    if freebsd_ip_scope(dst_addr) != "ip_unknown_or_zero":
        return ("netflow", freebsd_ip_scope(dst_addr), freebsd_exact_ip_token(dst_addr))
    fallback = freebsd_endpoint_detail_token(src_addr, prefix="src_")
    if fallback:
        return ("netflow", fallback)
    return ("netflow",)
