from __future__ import annotations

import ipaddress
import re


CLEARSCOPE_LEGACY_SEMANTIC_MODE = "clearscope_android_semantics_v1"
CLEARSCOPE_SEMANTIC_RULES_VERSION = CLEARSCOPE_LEGACY_SEMANTIC_MODE
CLEARSCOPE_REFINED_SEMANTIC_MODE = "raw_detail_v2_refined"
CLEARSCOPE_REFINED_SEMANTIC_ALIASES = {
    "",
    "default",
    "raw_detail_v2_refined",
    "clearscope_raw_detail_v2_refined",
}
CLEARSCOPE_LEGACY_SEMANTIC_ALIASES = {
    "legacy",
    "clearscope_android_semantics_v1",
}
UNKNOWN_ENDPOINT_VALUES = {
    "",
    "0",
    "0.0.0.0",
    "::",
    ":::",
    "na",
    "n/a",
    "none",
    "null",
    "unknown",
}


def is_clearscope_dataset(dataset: object) -> bool:
    """Return true when dataset uses the ClearScope Android semantic policy."""
    return str(dataset or "").upper().startswith("CLEARSCOPE_")


def normalize_clearscope_semantic_mode(semantic_mode: object = "") -> str:
    """Return the canonical ClearScope semantic mode, defaulting to refined."""
    text = str(semantic_mode or "").strip().lower()
    if text in CLEARSCOPE_REFINED_SEMANTIC_ALIASES:
        return CLEARSCOPE_REFINED_SEMANTIC_MODE
    if text in CLEARSCOPE_LEGACY_SEMANTIC_ALIASES:
        return CLEARSCOPE_LEGACY_SEMANTIC_MODE
    return CLEARSCOPE_REFINED_SEMANTIC_MODE


def clearscope_semantic_mode_is_legacy(semantic_mode: object = "") -> bool:
    """Return true only when ClearScope legacy semantics were explicitly requested."""
    return normalize_clearscope_semantic_mode(semantic_mode) == CLEARSCOPE_LEGACY_SEMANTIC_MODE


def normalize_token(value: object, max_len: int = 80) -> str:
    """Normalize a ClearScope semantic fragment to one bounded token."""
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    return (text or "unknown")[: max(int(max_len), 1)]


def normalize_refined_token(value: object, max_len: int = 80) -> str:
    """Normalize one refined raw-detail token and collapse repeated separators."""
    text = normalize_token(value, max_len=max_len)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "unknown")[: max(int(max_len), 1)]


def classify_android_cmd_nll(cmd: object) -> str:
    """Return the ClearScope Android process coarse label."""
    text = str(cmd or "").strip()
    if not text:
        return ""
    if "." in text:
        pieces = text.split(".")
        if any(piece == "" for piece in pieces):
            return ""
        if any(re.fullmatch(r"[A-Za-z0-9_]+", piece) is None for piece in pieces):
            return ""
    elif re.fullmatch(r"[A-Za-z0-9_:-]+", text) is None:
        return ""
    if text.startswith("com.android.providers."):
        return "providers"
    if text.startswith("com.android."):
        return "com_android"
    if text.startswith("android.process."):
        return "android_process"
    if text.startswith("org.mozilla."):
        return "org_mozilla"
    if text.startswith("com.google."):
        return "com_google"
    if text.startswith("com."):
        return "com_other"
    if "." in text:
        return "package_other"
    return "native_process"


def _android_cmd_base_and_suffix(cmd: object) -> tuple[str, str]:
    text = str(cmd or "").strip()
    if ":" not in text:
        return text, ""
    base, suffix = text.split(":", 1)
    return base.strip(), suffix.strip()


def classify_android_cmd_nll_refined(cmd: object) -> str:
    """Return the refined ClearScope Android process coarse label."""
    base_pkg, _suffix = _android_cmd_base_and_suffix(cmd)
    text = base_pkg.strip()
    if not text:
        return "unknown"
    if text.startswith("com.android.providers."):
        return "providers"
    if text.startswith("com.android."):
        return "com_android"
    if text.startswith("android.process."):
        return "android_process"
    if text.startswith("org.mozilla."):
        return "org_mozilla"
    if text.startswith("com.google."):
        return "com_google"
    if text.startswith("com."):
        return "com_other"
    if "." in text:
        return "package_other"
    return "native_process"


def android_cmd_detail(cmd: object, label: str) -> str:
    """Return the ClearScope residual process detail token."""
    text = str(cmd or "").strip()
    lowered = text.lower()
    prefixes = {
        "providers": "com.android.providers.",
        "com_android": "com.android.",
        "android_process": "android.process.",
        "org_mozilla": "org.mozilla.",
        "com_google": "com.google.",
        "com_other": "com.",
    }
    if label in prefixes and lowered.startswith(prefixes[label]):
        return normalize_token(lowered[len(prefixes[label]):], max_len=60)
    if label == "package_other":
        return normalize_token(lowered.replace(".", "_"), max_len=60)
    return normalize_token(lowered.split()[0] if lowered else "", max_len=60)


def android_cmd_detail_refined(cmd: object, label: str | None = None) -> str:
    """Return the refined ClearScope Android process detail token."""
    base_pkg, suffix = _android_cmd_base_and_suffix(cmd)
    actual_label = label or classify_android_cmd_nll_refined(cmd)
    lowered = base_pkg.lower()
    prefixes = {
        "providers": "com.android.providers.",
        "com_android": "com.android.",
        "android_process": "android.process.",
        "org_mozilla": "org.mozilla.",
        "com_google": "com.google.",
        "com_other": "com.",
    }
    if actual_label in prefixes and lowered.startswith(prefixes[actual_label]):
        base_detail = lowered[len(prefixes[actual_label]) :]
    elif actual_label == "package_other":
        base_detail = lowered
    else:
        base_detail = lowered.split()[0] if lowered else ""
    parts = [base_detail, suffix.lower()]
    return normalize_refined_token("_".join(part for part in parts if part), max_len=60)


def android_process_nll_role(cmd: object) -> str:
    """Return the ClearScope process NLL role."""
    label = classify_android_cmd_nll(cmd)
    if not label:
        return "process|android|unknown"
    return f"process|android|{label}"


def android_process_residual_tokens(cmd: object) -> tuple[str, ...]:
    """Return ClearScope process residual tokens."""
    label = classify_android_cmd_nll(cmd)
    if not label:
        return ("android_cmd_label:unknown", "android_cmd_detail:unknown")
    detail = android_cmd_detail(cmd, label)
    return (f"android_cmd_label:{label}", f"android_cmd_detail:{detail}")


def android_process_natural_tokens(cmd: object) -> tuple[str, ...]:
    """Return natural ClearScope process residual tokens."""
    label = classify_android_cmd_nll(cmd)
    if not label:
        return ("process", "unknown", "unknown")
    return ("process", label, android_cmd_detail(cmd, label))


def android_process_natural_tokens_refined(cmd: object) -> tuple[str, ...]:
    """Return refined natural ClearScope process residual tokens."""
    label = classify_android_cmd_nll_refined(cmd)
    return ("process", label, android_cmd_detail_refined(cmd, label))


def classify_android_file_nll(path: object) -> str:
    """Return the ClearScope Android file coarse label."""
    raw = str(path or "").strip()
    if not raw:
        return "unknown_file"
    p = raw.lower()
    if p.endswith(".apk"):
        return "android_apk_file"
    if p.endswith(".so"):
        return "native_library_file"
    if re.match(r"^/data/data/[^/]+/cache(/|$)", p):
        return "android_app_cache_file"
    if re.match(r"^/data/data/[^/]+(/|$)", p):
        return "android_app_private_file"
    if re.match(r"^/storage/emulated/0/android/data/[^/]+/cache(/|$)", p):
        return "android_external_app_cache_file"
    if p.startswith("/storage/emulated/0/dcim/") or p.startswith("/data/media/0/dcim/"):
        return "android_user_media_file"
    if p.startswith("/storage/emulated/0/"):
        return "android_user_storage_file"
    if p.startswith("/data/local/tmp/"):
        return "android_tmp_file"
    if p.startswith("/data/app/"):
        return "android_app_install_file"
    if p.startswith("/data/dalvik-cache/"):
        return "android_dalvik_cache_file"
    if p.startswith("/data/misc/"):
        return "android_misc_file"
    if p.startswith("/data/backup/"):
        return "android_backup_file"
    if p.startswith("/data/tombstones"):
        return "android_tombstone_file"
    if p.startswith("/data/system/recent_tasks/"):
        return "android_recent_task_file"
    if p.startswith("/data/system/recent_images/"):
        return "android_recent_image_file"
    if p.startswith("/data/system/dropbox/"):
        return "android_system_dropbox_file"
    if p.startswith("/data/system/procstats/"):
        return "android_procstats_file"
    if p.startswith("/data/system/usagestats/"):
        return "android_usagestats_file"
    if p.startswith("/data/system/users/"):
        return "android_user_state_file"
    if p.startswith("/data/system/"):
        return "android_system_state_file"
    if re.match(r"^/proc/[0-9]+/cmdline$", p):
        return "proc_cmdline"
    if re.match(r"^/proc/[0-9]+(/|$)", p):
        return "proc_file"
    if p.startswith("/proc/"):
        return "proc_file"
    if p == "/system" or p.startswith("/system/"):
        return "android_system_file"
    if p == "/vendor" or p.startswith("/vendor/"):
        return "android_vendor_file"
    if p.startswith("/dev/socket/"):
        return "android_socket_file"
    if p.startswith("/dev/"):
        return "android_device_file"
    if p.startswith("/sys/"):
        return "android_sysfs_file"
    if p.startswith("/cache/"):
        return "android_cache_file"
    if p == "/ashmem":
        return "android_ashmem_file"
    if p == "/seapp_contexts":
        return "android_policy_file"
    if p == "/":
        return "filesystem_root"
    roots = {
        "/data",
        "/data/data",
        "/data/system",
        "/storage",
        "/storage/emulated",
        "/storage/emulated/0",
        "/cache",
    }
    if p in roots:
        return "android_root_dir"
    return "file_other"


def _package_family(path: str) -> str:
    match = re.search(r"/data/data/([^/]+)", path.lower())
    if match is None:
        match = re.search(r"/android/data/([^/]+)", path.lower())
    package = match.group(1) if match else ""
    if package.startswith("org.mozilla."):
        return "org_mozilla"
    if package.startswith("com.android.email"):
        return "com_android_email"
    if package.startswith("com.android."):
        rest = package[len("com.android."):]
        return "com_android_" + normalize_token(rest, max_len=40)
    if package.startswith("com.google."):
        rest = package[len("com.google."):]
        return "com_google_" + normalize_token(rest, max_len=40)
    if package.startswith("com."):
        rest = package[len("com."):]
        return "com_" + normalize_token(rest, max_len=40)
    return normalize_token(package, max_len=40) if package else "app"


def _package_name_from_path(path: str) -> str:
    lowered = path.lower()
    match = re.search(r"/data/data/([^/]+)", lowered)
    if match is None:
        match = re.search(r"/android/data/([^/]+)", lowered)
    if match is None:
        match = re.search(r"/data/app/([^/]+)", lowered)
    package = match.group(1) if match else ""
    return package.rsplit("-", 1)[0] if package else ""


def _refined_package_family_from_package(package: str) -> str:
    pkg = str(package or "").strip().lower()
    if not pkg:
        return "app"
    if pkg == "org.mozilla.fennec_firefox_dev":
        return "fennec_firefox_dev"
    if pkg.startswith("org.mozilla."):
        rest = pkg[len("org.mozilla.") :]
        return "org_mozilla_" + normalize_refined_token(rest, max_len=40)
    if pkg.startswith("com.android."):
        rest = pkg[len("com.android.") :]
        return "com_android_" + normalize_refined_token(rest, max_len=40)
    if pkg.startswith("com.google."):
        rest = pkg[len("com.google.") :]
        return "com_google_" + normalize_refined_token(rest, max_len=40)
    if pkg.startswith("com."):
        rest = pkg[len("com.") :]
        return "com_" + normalize_refined_token(rest, max_len=40)
    return normalize_refined_token(pkg, max_len=50)


def _refined_package_family(path: str) -> str:
    return _refined_package_family_from_package(_package_name_from_path(path))


def android_file_detail(path: object, label: str) -> str:
    """Return the ClearScope residual file detail token."""
    raw = str(path or "").strip()
    p = raw.lower()
    basename = p.rstrip("/").rsplit("/", 1)[-1] if p else ""
    family = _package_family(p)
    if label == "unknown_file":
        return "unknown"
    if label == "proc_cmdline":
        return "cmdline"
    if label == "proc_file":
        return _proc_detail(p)
    if label == "android_app_cache_file":
        return f"{family}_cache" if family != "app" else "app_cache"
    if label == "android_app_private_file":
        if "/files/body/" in p:
            return "email_body"
        if p.endswith("/shared_files") or "/shared_files/" in p:
            return "shared_files"
        if "csb.tracee." in p:
            return "tracee_file"
        return "app_private"
    if label == "android_external_app_cache_file":
        if basename.endswith(".eml"):
            return "email_eml"
        return f"{family}_cache" if family != "app" else "external_cache"
    if label == "android_user_media_file":
        if basename.endswith(".tc-md"):
            return "camera_metadata"
        if basename.endswith((".jpg", ".jpeg", ".png")):
            return "camera_image"
        return "user_media"
    if label == "android_user_storage_file":
        return _user_storage_detail(p, family)
    if label == "android_tmp_file":
        return "tmp_file"
    if label == "android_app_install_file":
        return _app_install_detail(p)
    if label == "android_dalvik_cache_file":
        if "boot.oat" in p:
            return "boot_oat"
        if "data@app@" in p or basename.endswith(".dex"):
            return "app_dex_cache"
        return "dalvik_cache"
    if label == "native_library_file":
        if p.startswith("/system/lib/hw/"):
            return "hardware_lib"
        if p.startswith("/system/lib/"):
            if "webviewchromium" in p:
                return "webview_lib"
            return "system_lib"
        if "/data/app/" in p:
            return "app_lib"
        return "native_lib"
    fixed = {
        "android_system_file": _prefix_detail(p, "system"),
        "android_vendor_file": _prefix_detail(p, "vendor"),
        "android_system_dropbox_file": _dropbox_detail(p),
        "android_recent_task_file": "task_backup" if basename.endswith(".bak") else "task_xml",
        "android_recent_image_file": "thumbnail",
        "android_procstats_file": "state_bin" if basename.endswith(".bin") else "procstats",
        "android_usagestats_file": "daily",
        "android_user_state_file": _user_state_detail(basename),
        "android_system_state_file": _system_state_detail(p, basename),
        "android_misc_file": _misc_detail(p),
        "android_backup_file": _backup_detail(p),
        "android_tombstone_file": "tombstone_dir" if p == "/data/tombstones" else basename,
        "android_socket_file": "fwmarkd" if basename == "fwmarkd" else "socket",
        "android_device_file": _device_detail(basename),
        "android_sysfs_file": _sysfs_detail(p),
        "android_cache_file": "cache_root" if p == "/cache" else "cache_file",
        "android_apk_file": "apk",
        "android_policy_file": "seapp_contexts",
        "android_ashmem_file": "ashmem",
        "filesystem_root": "root",
        "android_root_dir": _root_dir_detail(p),
        "file_other": "other",
    }
    return normalize_token(fixed.get(label, label), max_len=60)


def android_file_detail_refined(path: object, label: str | None = None) -> str:
    """Return the refined ClearScope Android file detail token."""
    raw = str(path or "").strip()
    p = raw.lower()
    actual_label = label or classify_android_file_nll(p)
    basename = p.rstrip("/").rsplit("/", 1)[-1] if p else ""
    if actual_label == "unknown_file":
        return "unknown"
    if actual_label == "proc_cmdline":
        return "cmdline"
    if actual_label == "proc_file":
        return _proc_detail_refined(p)
    if actual_label == "android_device_file":
        return _device_detail_refined(p)
    if actual_label == "android_socket_file":
        return _socket_detail_refined(p)
    if actual_label == "android_apk_file":
        return _apk_detail_refined(p)
    if actual_label == "android_system_file":
        return _system_file_detail_refined(p)
    if actual_label == "android_app_private_file":
        return _app_private_detail_refined(p)
    if actual_label == "android_app_cache_file":
        return _app_cache_detail_refined(p)
    if actual_label == "android_external_app_cache_file":
        if basename.endswith(".eml"):
            return "email_eml"
    if actual_label == "android_user_media_file":
        if basename.endswith(".tc-md"):
            return "camera_metadata"
        if basename.endswith((".jpg", ".jpeg", ".png")):
            return "camera_image"
    if actual_label == "android_recent_image_file":
        return "thumbnail"
    return android_file_detail(path, actual_label)


def clearscope_file_nll_role(path: object) -> str:
    """Return the ClearScope file NLL role."""
    return f"file|android|{classify_android_file_nll(path)}"


def clearscope_file_residual_text(path: object) -> str:
    """Return ClearScope file residual text with coarse label plus one detail."""
    label = classify_android_file_nll(path)
    detail = android_file_detail(path, label)
    return f"android_file_label:{label} android_file_detail:{detail}"


def clearscope_file_natural_tokens(path: object) -> tuple[str, ...]:
    """Return natural ClearScope file residual tokens."""
    label = classify_android_file_nll(path)
    return ("file", label, android_file_detail(path, label))


def clearscope_file_natural_tokens_refined(path: object) -> tuple[str, ...]:
    """Return refined natural ClearScope file residual tokens."""
    label = classify_android_file_nll(path)
    return ("file", label, android_file_detail_refined(path, label))


def clearscope_netflow_nll_role() -> str:
    """Return the fixed ClearScope netflow NLL role."""
    return "net|android|netflow"


def clearscope_netflow_residual_text() -> str:
    """Return the fixed ClearScope netflow residual semantic text."""
    return "netflow"


def clearscope_netflow_natural_tokens() -> tuple[str, ...]:
    """Return natural ClearScope netflow residual tokens."""
    return ("netflow",)


def clearscope_netflow_natural_tokens_refined(
    *,
    src_addr: object = "",
    dst_addr: object = "",
    keep_src_detail_for_audit: bool = False,
) -> tuple[str, ...]:
    """Return refined ClearScope netflow tokens, fixed by default."""
    del dst_addr
    if not bool(keep_src_detail_for_audit):
        return ("netflow",)
    src_token = _netflow_endpoint_detail_token(src_addr)
    return ("netflow", src_token)


def clearscope_residual_tokens_refined(
    row: dict[str, object],
    *,
    keep_netflow_src_detail_for_audit: bool = False,
) -> tuple[str, ...]:
    """Return refined ClearScope residual sentence tokens for one event row."""
    action = _refined_action_token(row.get("action", "unknown"))
    tokens = [
        *_clearscope_refined_node_tokens(row, "src", keep_netflow_src_detail_for_audit),
        "event",
        action,
        *_clearscope_refined_node_tokens(row, "dst", keep_netflow_src_detail_for_audit),
    ]
    return tuple(str(token) for token in tokens if str(token).strip())


def clearscope_residual_text_refined(
    row: dict[str, object],
    *,
    keep_netflow_src_detail_for_audit: bool = False,
) -> str:
    """Return refined ClearScope residual sentence text for one event row."""
    return " ".join(
        clearscope_residual_tokens_refined(
            row,
            keep_netflow_src_detail_for_audit=keep_netflow_src_detail_for_audit,
        ),
    )


def _proc_detail(path: str) -> str:
    match = re.match(r"^/proc/[0-9]+/(.*)$", path)
    rest = match.group(1) if match else path.removeprefix("/proc/")
    parts = [part for part in rest.split("/") if part]
    if not parts:
        return "proc"
    if parts[0] == "task" and len(parts) >= 3:
        return "task_" + normalize_token(parts[2], max_len=30)
    if parts[0] == "fd":
        return "fd"
    return normalize_token(parts[0], max_len=30)


def _proc_detail_refined(path: str) -> str:
    match = re.match(r"^/proc/[0-9]+/(.*)$", path)
    rest = match.group(1) if match else path.removeprefix("/proc/")
    parts = [part for part in rest.split("/") if part]
    if not parts:
        return "proc"
    if parts[0] == "net":
        return _proc_net_detail_refined(parts[1:])
    if parts[0] == "task" and len(parts) >= 3:
        return "task_" + normalize_refined_token(parts[2], max_len=40)
    if parts[0] == "fd":
        return "fd"
    return normalize_refined_token(parts[0], max_len=40)


def _proc_net_detail_refined(parts: list[str]) -> str:
    if not parts:
        return "net_other"
    first = normalize_refined_token(parts[0], max_len=40)
    if first == "xt_qtaguid":
        second = normalize_refined_token(parts[1], max_len=40) if len(parts) >= 2 else ""
        if second == "ctrl":
            return "net_xt_qtaguid_ctrl"
        if second.startswith("iface_stat"):
            return "net_xt_qtaguid_iface_stat"
        return "net_xt_qtaguid"
    if first in {"arp", "tcp", "udp", "unix"}:
        return f"net_{first}"
    return "net_other"


def _device_detail_refined(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    basename = normalize_refined_token(parts[-1] if parts else "", max_len=40)
    if len(parts) >= 2 and parts[1] == "graphics":
        return "dev_graphics"
    if len(parts) >= 2 and parts[1] == "input":
        return "dev_input"
    if len(parts) >= 2 and parts[1] == "log":
        return "dev_log"
    if basename in {"ion", "ashmem", "pmsg0", "binder", "urandom"}:
        return f"dev_{basename}"
    if basename.startswith("kgsl"):
        return "dev_kgsl"
    return "dev_other"


def _socket_detail_refined(path: str) -> str:
    basename = normalize_refined_token(path.rstrip("/").rsplit("/", 1)[-1], max_len=40)
    known = {
        "logdw",
        "lmkd",
        "dnsproxyd",
        "netd",
        "zygote",
        "vold",
        "installd",
        "rild",
        "fwmarkd",
    }
    if basename in known:
        return f"socket_{basename}"
    return "socket_other"


def _apk_detail_refined(path: str) -> str:
    system_app = re.match(r"^/system/app/([^/]+)/", path)
    if system_app is not None:
        app = normalize_refined_token(system_app.group(1), max_len=45)
        return f"system_app_{app}_apk"
    system_priv = re.match(r"^/system/priv-app/([^/]+)/", path)
    if system_priv is not None:
        app = normalize_refined_token(system_priv.group(1), max_len=45)
        return f"system_priv_app_{app}_apk"
    if re.match(r"^/data/app/vmdl[0-9]+\.tmp/base\.apk$", path):
        return "data_app_vmdl_base_apk"
    if re.match(r"^/data/app/com\.metasploit\.stage-[^/]+/base\.apk$", path):
        return "data_app_metasploit_stage_apk"
    if path.startswith(("/tmp/", "/data/local/tmp/")):
        basename = path.rstrip("/").rsplit("/", 1)[-1]
        if "mozilla" in basename or "fennec" in basename:
            return "tmp_org_mozilla_apk"
        return "tmp_apk"
    return "apk_other"


def _system_file_detail_refined(path: str) -> str:
    if path.startswith("/system/media/audio/ui/"):
        return "system_media_ui"
    if path.startswith("/system/media/audio/notifications/"):
        return "system_media_notifications"
    if path.startswith("/system/fonts/"):
        return "system_fonts"
    if path.startswith("/system/etc/"):
        return "system_etc"
    if path.startswith("/system/framework/"):
        return "system_framework"
    if path.startswith("/system/bin/"):
        return "system_bin"
    if path.startswith("/system/lib/"):
        return "system_lib"
    return "system_file"


def _app_private_detail_refined(path: str) -> str:
    family = _refined_package_family(path)
    if "/files/body/" in path:
        basename = path.rstrip("/").rsplit("/", 1)[-1]
        if basename.endswith(".txt"):
            return "email_body_txt"
        if basename.endswith(".html"):
            return "email_body_html"
        return "email_body"
    if "/app_webview/" in path or path.rstrip("/").endswith("/app_webview"):
        return f"{family}_app_webview"
    if "/files/mozilla/" in path or re.search(r"/data/data/[^/]+/mozilla/", path):
        return f"{family}_mozilla_profile"
    if path.rstrip("/").endswith("/shared_files") or "/shared_files/" in path:
        return f"{family}_shared_files"
    if "csb.tracee." in path:
        return f"{family}_tracee_file"
    match = re.match(r"^/data/data/[^/]+/([^/]+)", path)
    subdir = normalize_refined_token(match.group(1), max_len=40) if match else "app_private"
    if subdir in {"databases", "shared_prefs", "files", "code_cache", "lib"}:
        return f"{family}_{subdir}"
    return f"{family}_app_private"


def _app_cache_detail_refined(path: str) -> str:
    family = _refined_package_family(path)
    if family == "fennec_firefox_dev":
        if "/cache2/doomed/" in path:
            return "fennec_firefox_dev_doomed_num"
        if "/cache2/entries/" in path:
            return "fennec_firefox_dev_entries_hexblob"
        if "safebrowsing" in path:
            return "fennec_firefox_dev_safebrowsing_cache"
        return "fennec_firefox_dev_cache"
    if family == "com_android_camera2" and "image_manager_disk_cache" in path:
        return "camera2_image_manager_disk_cache_mixedid"
    if family == "com_android_browser":
        return "com_android_browser_cache"
    return f"{family}_cache" if family != "app" else "app_cache"


def _netflow_endpoint_detail_token(value: object) -> str:
    text = str(value or "").strip()
    lowered = text.lower()
    if lowered in UNKNOWN_ENDPOINT_VALUES or lowered.startswith("<unnamed fd:"):
        return "remote_unknown"
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return "remote_unknown"
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    if address.is_unspecified:
        return "remote_unknown"
    token = str(address).replace(".", "_").replace(":", "_")
    token = normalize_refined_token(token, max_len=80)
    return f"src_{token}" if token != "unknown" else "remote_unknown"


def _refined_action_token(action: object) -> str:
    text = normalize_refined_token(action, max_len=60)
    if text.startswith("event_"):
        return text
    return f"event_{text}" if text else "event_unknown"


def _clearscope_refined_node_tokens(
    row: dict[str, object],
    side: str,
    keep_netflow_src_detail_for_audit: bool,
) -> tuple[str, ...]:
    kind = normalize_refined_token(row.get(f"{side}_kind", ""), max_len=30)
    if kind == "process":
        return android_process_natural_tokens_refined(row.get(f"{side}_process_cmd", ""))
    if kind == "file":
        return clearscope_file_natural_tokens_refined(row.get(f"{side}_file_path", ""))
    if kind == "netflow":
        return clearscope_netflow_natural_tokens_refined(
            src_addr=row.get("src_addr", ""),
            dst_addr=row.get(f"{side}_addr", row.get("dst_addr", "")),
            keep_src_detail_for_audit=keep_netflow_src_detail_for_audit,
        )
    return (kind or "unknown",)


def _user_storage_detail(path: str, family: str) -> str:
    if path.endswith("contacts.vcf.tc-md"):
        return "contacts_metadata"
    if "/files/download" in path and family == "org_mozilla":
        return "org_mozilla_download"
    if "/files/temp" in path and family == "org_mozilla":
        return "org_mozilla_temp"
    if path.rstrip("/").endswith("/alarms"):
        return "alarms_dir"
    if path.rstrip("/").endswith("/podcasts"):
        return "podcasts_dir"
    return "user_storage"


def _app_install_detail(path: str) -> str:
    if "com.metasploit.stage" in path:
        return "metasploit_stage"
    if "com.ta5.android.blue" in path:
        return "ta5_android_blue"
    if re.search(r"/vmdl[0-9]+\.tmp/lib/?$", path):
        return "vmdl_tmp_lib"
    if re.search(r"/vmdl[0-9]+\.tmp", path):
        return "vmdl_tmp"
    if "org.mozilla.fennec_firefox_dev" in path and path.rstrip("/").endswith("/lib"):
        return "org_mozilla_lib"
    return "app_install"


def _prefix_detail(path: str, prefix: str) -> str:
    if f"/{prefix}/bin/" in path:
        return f"{prefix}_bin"
    if f"/{prefix}/etc/" in path:
        return f"{prefix}_etc"
    if f"/{prefix}/framework/" in path:
        return f"{prefix}_framework"
    if f"/{prefix}/firmware/" in path:
        return f"{prefix}_firmware"
    if f"/{prefix}/lib/" in path:
        return f"{prefix}_lib"
    return f"{prefix}_file"


def _dropbox_detail(path: str) -> str:
    basename = path.rstrip("/").rsplit("/", 1)[-1]
    if basename.startswith("system_app_anr@"):
        return "system_app_anr"
    if basename.startswith("system_server_lowmem@"):
        return "system_server_lowmem"
    if basename.startswith("drop") and basename.endswith(".tmp"):
        return "dropbox_tmp"
    return "dropbox_log"


def _user_state_detail(basename: str) -> str:
    mapping = {
        "settings_system.xml": "settings_system",
        "settings_secure.xml": "settings_secure",
        "settings_global.xml.bak": "settings_global_backup",
        "accounts.db": "accounts_db",
        "runtime-permissions.xml": "runtime_permissions",
        "package-restrictions.xml": "package_restrictions",
        "wallpaper_info.xml": "wallpaper_info",
    }
    return mapping.get(basename, "user_state")


def _system_state_detail(path: str, basename: str) -> str:
    if basename == "packages.xml":
        return "packages_xml"
    if "/sync/accounts.xml" in path:
        return "sync_state"
    if basename == "locksettings.db":
        return "locksettings_db"
    return "system_state"


def _misc_detail(path: str) -> str:
    if "/shared_relro/" in path:
        return "shared_relro"
    if "/wifi/" in path:
        return "wifi_config"
    if "/bluedroid/" in path:
        return "bluetooth_config"
    return "misc_state"


def _backup_detail(path: str) -> str:
    if "/pending/journal-" in path:
        return "backup_journal"
    if path.rstrip("/").endswith("/pending"):
        return "backup_pending"
    if path.rstrip("/").endswith("/fb-schedule"):
        return "backup_schedule"
    return "backup_state"


def _device_detail(basename: str) -> str:
    if basename == "null":
        return "null_device"
    if basename == "urandom":
        return "urandom"
    return "device"


def _sysfs_detail(path: str) -> str:
    if "/cpufreq/" in path:
        return "cpu_cpufreq"
    if path.startswith("/sys/class/"):
        return "sys_class"
    return "sysfs"


def _root_dir_detail(path: str) -> str:
    return {
        "/data": "data_root",
        "/data/data": "data_data_root",
        "/data/system": "data_system_root",
        "/storage": "storage_root",
        "/storage/emulated": "storage_root",
        "/storage/emulated/0": "emulated_storage_root",
        "/cache": "cache_root",
    }.get(path, "root_dir")
