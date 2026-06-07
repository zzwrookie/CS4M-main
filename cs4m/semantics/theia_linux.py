from __future__ import annotations

import ipaddress
import re
import shlex


THEIA_SEMANTIC_RULES_VERSION = "theia_linux_raw_detail_v2"

INTERNAL_ENV_CIDR = ipaddress.IPv4Network("128.55.12.0/24")
PRIVATE_CIDRS = (
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("172.16.0.0/12"),
)
ZERO_CIDR = ipaddress.IPv4Network("0.0.0.0/8")
HOST_PSEUDO_PROCESS_RE = re.compile(r"^128\.55\.12\.\d+-m$", re.IGNORECASE)
REPLAY_SNAPSHOT_SEGMENT_RE = r"f[0-9]+(?:\.bak(?:\.1)?)?"
REPLAY_LOGDB_PATH_RE = re.compile(
    rf"^/data/{REPLAY_SNAPSHOT_SEGMENT_RE}/replay_logdb(?:/|$)"
)
REPLAY_CACHE_PATH_RE = re.compile(
    rf"^/data/{REPLAY_SNAPSHOT_SEGMENT_RE}/replay_cache(?:/|[^/].*|$)"
)
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
SHELL_CONTROL_TOKENS = {
    "-c",
    "-e",
    "-lc",
    "-l",
    "&",
    "&&",
    "||",
    "|",
}


def is_theia_dataset(dataset: object) -> bool:
    """Return true when dataset uses the THEIA Linux semantic policy."""
    return str(dataset or "").upper().startswith("THEIA_")


def normalize_token(value: object, max_len: int = 80) -> str:
    """Normalize one Linux semantic fragment to a bounded token."""
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    return (text or "unknown")[: max(int(max_len), 1)]


def _is_missing_detail(value: object) -> bool:
    return str(value or "").strip().lower() in MISSING_DETAIL_VALUES


def _split_command(cmd: object) -> list[str]:
    raw = str(cmd or "").strip()
    if not raw or raw.lower() in MISSING_DETAIL_VALUES:
        return []
    try:
        return shlex.split(raw)
    except ValueError:
        return raw.split()


def _raw_basename(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw.rstrip("/").rsplit("/", 1)[-1]


def _normalized_basename(value: object) -> str:
    return normalize_token(_raw_basename(value), max_len=60) if str(value or "").strip() else ""


def _parent_basename(path: object) -> str:
    raw = str(path or "").strip().rstrip("/")
    if "/" not in raw:
        return ""
    parent = raw.rsplit("/", 1)[0].rstrip("/")
    if not parent:
        return ""
    return normalize_token(parent.rsplit("/", 1)[-1], max_len=60)


def _is_numeric_pid_like(token: str) -> bool:
    return bool(re.fullmatch(r"[0-9]+", str(token or "")))


def _path_detail_token(path: object) -> str:
    raw = str(path or "").strip()
    if _is_missing_detail(raw):
        return ""
    lowered = raw.lower().rstrip("/")
    if lowered == "0":
        return "swapper"
    if lowered == "/":
        return "root"
    if re.match(r"^/proc/[0-9]+(/|$)", lowered):
        return _proc_file_detail(lowered)
    basename = _normalized_basename(raw)
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
            token = _path_detail_token(arg) or _normalized_basename(arg)
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


def _process_detail_token(path: object, cmd: object) -> str:
    parts = _split_command(cmd)
    executable = ""
    if parts:
        executable = normalize_token(parts[0].rsplit("/", 1)[-1].lstrip("-"), max_len=60)
    if executable in SHELL_INTERPRETERS:
        arg_token = _meaningful_command_arg(parts)
        if arg_token:
            return arg_token
    if executable:
        return executable
    return _path_detail_token(path)


def _first_command_token(cmd: object) -> str:
    parts = _split_command(cmd)
    if not parts:
        return ""
    executable = parts[0].rsplit("/", 1)[-1].lstrip("-")
    return normalize_token(executable, max_len=60)


def _path_basename(path: object) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    if raw == "0":
        return "swapper"
    return _normalized_basename(raw)


def _combined_text(path: object, cmd: object) -> str:
    return f"{cmd or ''} {path or ''}".lower()


def classify_linux_process_nll(path: object, cmd: object) -> str:
    """Return the THEIA Linux process coarse label."""
    raw_cmd = str(cmd or "").strip()
    raw_path = str(path or "").strip()
    token = _first_command_token(raw_cmd) or _path_basename(raw_path)
    text = _combined_text(raw_path, raw_cmd)
    if not token and not raw_path:
        return "unknown_process"
    if token in {"0", "swapper"} or "dir_swapper" in text or "swapper" in text:
        return "kernel_thread"
    if HOST_PSEUDO_PROCESS_RE.fullmatch(raw_path) and raw_cmd.lower() in {"", "n/a"}:
        return "host_pseudo_process"
    if token.startswith(("kworker", "kthreadd", "flush", "swapper")):
        return "kernel_thread"
    if token in {"cron", "anacron", "run_parts"}:
        return "scheduler_service"
    if token.startswith("dhclient"):
        return "network_service"
    if token in {"sudo", "logrotate", "mount", "umount", "sysctl"}:
        return "admin_tool"
    if token in {"dbus_daemon", "cupsd", "init", "systemd", "console_kit_daemon"}:
        return "system_daemon"
    if token in {"bash", "dash", "sh", "zsh", "csh", "tcsh"}:
        return "shell"
    if token in {"python", "python2", "python2_7", "python3", "perl", "ruby"}:
        return "interpreter"
    if token in {"firefox", "mozilla", "plugin_container"} or "firefox" in text:
        return "browser"
    if token in {"thunderbird"}:
        return "mail_client"
    if token in {"sshd", "ssh", "scp", "sftp"}:
        return "ssh_service"
    if token in {"postgres", "postmaster", "psql"} or "postgresql" in text:
        return "database_process"
    if token in {"apt", "apt_get", "apt_config", "dpkg", "pkg", "yum", "rpm"}:
        return "package_manager"
    if token in {"pulseaudio", "pulseaudio_kde", "speech_dispatcher"}:
        return "audio_service"
    if token in {"update_notifier", "update_manager"}:
        return "update_notifier"
    if token in {"fluxbox", "gnome_terminal", "unity_2d_shell", "unity_2d_panel"}:
        return "desktop_env"
    if token in {"stat", "uname", "grep", "sed", "sort", "cut", "basename", "dirname"}:
        return "core_util"
    if token in {"awk", "gawk", "ls", "sync", "top", "ps", "cat", "readlink"}:
        return "core_util"
    if token in {"gtcache", "chrome", "chromium"}:
        return "browser_helper"
    if raw_path.startswith("/usr/bin/"):
        return "system_user_bin"
    if raw_path.startswith(("/usr/lib/", "/usr/sbin/", "/sbin/")):
        return "system_helper"
    if raw_path.startswith(("/bin/", "/lib/")):
        return "core_binary"
    return "process_other"


def linux_process_detail(path: object, cmd: object, label: str | None = None) -> str:
    """Return the THEIA Linux process residual detail token."""
    actual_label = label or classify_linux_process_nll(path, cmd)
    raw_path = str(path or "").strip()
    token = _process_detail_token(raw_path, cmd)
    if actual_label == "unknown_process":
        return "unknown"
    if actual_label == "host_pseudo_process":
        return "internal_host"
    if actual_label == "kernel_thread":
        if token in {"0", "swapper"} or "swapper" in str(path or "").lower():
            return "swapper"
        if token.startswith("kworker"):
            return "kworker"
        if token.startswith("flush"):
            return "flush"
        return token or "kernel_thread"
    aliases = {
        "dhclient3": "dhclient",
        "run_parts": "run_parts",
        "apt_get": "apt_get",
        "apt_config": "apt_config",
        "dbus_daemon": "dbus_daemon",
        "unity_2d_shell": "unity_2d",
        "unity_2d_panel": "unity_2d",
        "plugin_container": "plugin_container",
    }
    return aliases.get(token, token or normalize_token(actual_label, max_len=60))


def linux_process_nll_role(path: object, cmd: object) -> str:
    """Return the THEIA Linux process NLL role."""
    return f"process|linux|{classify_linux_process_nll(path, cmd)}"


def linux_process_natural_tokens(path: object, cmd: object) -> tuple[str, ...]:
    """Return natural THEIA Linux process residual tokens."""
    label = classify_linux_process_nll(path, cmd)
    detail = linux_process_detail(path, cmd, label)
    return ("process", label, detail)


def classify_linux_file_nll(path: object) -> str:
    """Return the THEIA Linux file coarse label."""
    raw = str(path or "").strip()
    if not raw:
        return "unknown_file"
    p = re.sub(r"/+", "/", raw.lower())
    if len(p) > 1:
        p = p.rstrip("/")
    if p == "/":
        return "filesystem_root"
    if p in {
        "/bin",
        "/boot",
        "/cdrom",
        "/data",
        "/dev",
        "/etc",
        "/lib",
        "/lost+found",
        "/proc",
        "/root",
        "/sbin",
        "/tmp",
        "/usr",
        "/var",
        "/home",
        "/run",
    }:
        return "linux_root_dir"
    if re.match(r"^/proc/[0-9]+/cmdline$", p):
        return "proc_cmdline"
    if re.match(r"^/proc/[0-9]+(/|$)", p):
        return "proc_file"
    if p.startswith("/proc/"):
        return "proc_file"
    if re.match(r"^/[0-9]+(/|$)", p):
        return "pseudo_proc_file"
    if "/.cache/mozilla/" in p or "/cache2/entries/" in p:
        return "browser_cache_file"
    if p.startswith("/home/admin/.mozilla/"):
        return "browser_profile_file"
    if p.startswith("/home/admin/downloads/"):
        return "user_download_file"
    if p.startswith("/home/admin/qt") or p.startswith("/home/theia/"):
        return "user_source_or_dev_file"
    if p.startswith("/home/admin/") or p.startswith("/home/"):
        return "user_home_file"
    if p.startswith("/run/shm/"):
        return "shared_memory_file"
    if p.startswith("/run/"):
        return "runtime_state_file"
    if p.startswith("/tmp/"):
        return "tmp_file"
    if REPLAY_LOGDB_PATH_RE.match(p):
        return "replay_log_file"
    if REPLAY_CACHE_PATH_RE.match(p):
        return "replay_cache_file"
    if p.startswith("/data/replay_logdb/"):
        return "replay_log_file"
    if p.startswith("/var/lib/postgresql/"):
        return "database_state_file"
    if p.startswith("/var/cache/"):
        return "system_cache_file"
    if p.startswith("/var/log/"):
        return "system_log_file"
    if p.startswith("/var/"):
        return "var_state_file"
    if p.startswith("/etc/"):
        return "system_config_file"
    if p.endswith(".so"):
        return "native_library_file"
    if p.startswith("/usr/bin/"):
        return "system_binary_file"
    if p.startswith("/usr/lib/"):
        return "system_library_file"
    if p.startswith("/usr/share/"):
        return "system_share_file"
    if p.startswith("/usr/"):
        return "usr_file"
    if p.startswith("/bin/"):
        return "core_binary_file"
    if p.startswith("/lib/"):
        return "core_library_file"
    if p.startswith("/sbin/"):
        return "system_sbin_file"
    if p.startswith("/dev/"):
        return "device_file"
    if p.startswith("/sys/"):
        return "sysfs_file"
    if p.startswith("/root/"):
        return "root_home_file"
    if p.startswith("/boot/"):
        return "boot_file"
    if not p.startswith("/"):
        return "relative_or_special_file"
    return "file_other"


def linux_file_detail(path: object, label: str | None = None) -> str:
    """Return the THEIA Linux file residual detail token."""
    actual_label = label or classify_linux_file_nll(path)
    raw = str(path or "").strip().lower()
    basename = _raw_basename(raw).lower() if raw else ""
    detail = _path_detail_token(raw)
    if actual_label == "unknown_file":
        return "unknown"
    if actual_label == "filesystem_root":
        return "root"
    if actual_label == "linux_root_dir":
        root = raw.strip("/").split("/", 1)[0] if raw.strip("/") else "root"
        return normalize_token(f"{root}_root", max_len=60)
    if actual_label == "proc_cmdline":
        return "cmdline"
    if actual_label == "browser_cache_file":
        return detail or "browser_cache"
    if actual_label == "browser_profile_file":
        return detail or "profile_file"
    if actual_label == "shared_memory_file":
        return detail or "shm_other"
    if actual_label == "proc_file":
        return detail or "proc"
    if actual_label == "replay_log_file":
        return detail or "replay_record"
    if actual_label == "pseudo_proc_file":
        return detail or "pseudo_proc"
    if actual_label == "user_download_file":
        return detail or "download_file"
    if actual_label == "user_source_or_dev_file":
        return detail or "dev_source"
    if actual_label == "user_home_file":
        return detail or "home_file"
    if actual_label == "runtime_state_file":
        return detail or "runtime_state"
    if actual_label == "tmp_file":
        return detail or "tmp"
    if actual_label == "database_state_file":
        return detail or "postgres_state"
    if actual_label == "system_cache_file":
        return detail or "system_cache"
    if actual_label == "system_log_file":
        return detail or "system_log"
    if actual_label == "var_state_file":
        return detail or "var_state"
    if actual_label == "system_config_file":
        return detail or "config_file"
    if actual_label == "native_library_file":
        return detail or "native_lib"
    if actual_label == "system_binary_file":
        return detail or normalize_token(basename or "system_binary", max_len=60)
    if actual_label == "system_library_file":
        return detail or "usr_lib"
    if actual_label == "system_share_file":
        return detail or "usr_share"
    if actual_label == "usr_file":
        return detail or "usr_file"
    if actual_label == "core_binary_file":
        return detail or normalize_token(basename or "core_binary", max_len=60)
    if actual_label == "core_library_file":
        return detail or "core_lib"
    if actual_label == "system_sbin_file":
        return detail or normalize_token(basename or "sbin_file", max_len=60)
    if actual_label == "device_file":
        return detail or "device"
    if actual_label == "sysfs_file":
        return detail or "sysfs"
    if actual_label == "root_home_file":
        return detail or "root_home"
    if actual_label == "boot_file":
        return detail or "boot_file"
    if actual_label == "relative_or_special_file":
        return detail or "relative_or_special"
    return detail or "other"


def _proc_file_detail(path: str) -> str:
    match = re.match(r"^/proc/[0-9]+/(.*)$", path)
    rest = match.group(1) if match else path.removeprefix("/proc/")
    parts = [part for part in rest.split("/") if part]
    if not parts:
        return "proc"
    if parts[0] == "fd":
        return "fd"
    if parts[0] == "task" and len(parts) >= 3:
        return "task_" + normalize_token(parts[2], max_len=40)
    return normalize_token(parts[0], max_len=40)


def linux_file_nll_role(path: object) -> str:
    """Return the THEIA Linux file NLL role."""
    return f"file|linux|{classify_linux_file_nll(path)}"


def linux_file_natural_tokens(path: object) -> tuple[str, ...]:
    """Return natural THEIA Linux file residual tokens."""
    label = classify_linux_file_nll(path)
    return ("file", label, linux_file_detail(path, label))


def _parse_ipv4(value: object) -> ipaddress.IPv4Address | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return ipaddress.IPv4Address(text)
    except ipaddress.AddressValueError:
        return None


def linux_ip_scope(value: object) -> str:
    """Return the THEIA Linux netflow IP scope token."""
    addr = _parse_ipv4(value)
    if addr is None:
        return "ip_unknown_or_zero"
    if addr in ZERO_CIDR:
        return "ip_unknown_or_zero"
    if addr.is_loopback:
        return "ip_loopback"
    if addr in INTERNAL_ENV_CIDR:
        return "ip_internal_env"
    if any(addr in cidr for cidr in PRIVATE_CIDRS):
        return "ip_private"
    return "ip_public_or_external"


def linux_port_bucket(value: object) -> str:
    """Return the THEIA Linux destination port bucket."""
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


def linux_exact_ip_token(value: object) -> str:
    """Return exact THEIA Linux IP residual token without a port token."""
    addr = _parse_ipv4(value)
    if addr is None:
        return "unknown_ip"
    return "_".join(str(addr).split("."))


def linux_netflow_nll_role(dst_addr: object, dst_port: object) -> str:
    """Return the THEIA Linux netflow NLL role."""
    return f"net|{linux_ip_scope(dst_addr)}|{linux_port_bucket(dst_port)}"


def linux_endpoint_detail_token(value: object, prefix: str = "") -> str:
    """Return normalized THEIA endpoint detail with an optional direction prefix."""
    addr = _parse_ipv4(value)
    if addr is not None and addr not in ZERO_CIDR:
        token = "_".join(str(addr).split("."))
    else:
        text = str(value or "").strip().lower()
        if text in {"local", "localhost"}:
            token = "local"
        else:
            token = ""
    if not token:
        return ""
    return normalize_token(f"{prefix}{token}", max_len=80)


def linux_netflow_natural_tokens(dst_addr: object, src_addr: object = "") -> tuple[str, ...]:
    """Return natural THEIA Linux netflow residual tokens."""
    if linux_ip_scope(dst_addr) != "ip_unknown_or_zero":
        return ("netflow", linux_ip_scope(dst_addr), linux_exact_ip_token(dst_addr))
    fallback = linux_endpoint_detail_token(src_addr, prefix="src_")
    if fallback:
        return ("netflow", fallback)
    return ("netflow",)


def linux_netflow_detail_or_fixed_natural_tokens(
    dst_addr: object,
    src_addr: object = "",
) -> tuple[str, ...]:
    """Return THEIA netflow detail tokens when a concrete remote IP is present."""
    return linux_netflow_natural_tokens(dst_addr, src_addr)


def linux_netflow_fixed_nll_role() -> str:
    """Return the fixed THEIA Linux netflow NLL role."""
    return "net|linux|netflow"


def linux_netflow_fixed_natural_tokens() -> tuple[str, ...]:
    """Return fixed THEIA Linux netflow residual tokens."""
    return ("netflow",)
