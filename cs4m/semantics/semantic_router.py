from __future__ import annotations

import fnmatch
import os
import re
import shlex
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import yaml

from cs4m.semantics.clearscope_android import (
    android_process_nll_role,
    android_process_residual_tokens,
    classify_android_cmd_nll,
)
from cs4m.semantics.cadets_freebsd import (
    classify_freebsd_process_nll,
    freebsd_process_natural_tokens,
    freebsd_process_nll_role,
)
from cs4m.semantics.theia_linux import (
    classify_linux_process_nll,
    linux_process_natural_tokens,
    linux_process_nll_role,
)


PROCESS_SEMANTIC_RULES_VERSION = "v1"
PROCESS_SEMANTIC_COARSE_NLL_RULES_VERSION = "v2_coarse_nll"
PROCESS_SEMANTIC_PROCESS_KIND_NLL_RULES_VERSION = "v3_process_kind_nll"
DEFAULT_PROCESS_SEMANTICS_CONFIG = "configs/common/process_semantics.yaml"
_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:/-]+")
_LEGACY_TOKEN_RE = re.compile(r"[A-Za-z0-9_.-]+")
_ANDROID_PACKAGE_SEGMENT_RE = re.compile(r"[a-z0-9_:]+")
_SUPPORTED_V1_PROFILES = {"android", "linux", "freebsd", "windows"}
_AUDIT_COUNTER_LIMIT = 10000
_AUDIT_OTHER_KEY = "__other__"


@dataclass(frozen=True)
class ProcessSemanticConfig:
    """Configuration for process semantic normalization."""

    rules_version: str = PROCESS_SEMANTIC_RULES_VERSION
    dataset_os_profile: Mapping[str, str] = field(default_factory=dict)
    process_nll_max_tokens: int = 6
    process_residual_max_tokens: int = 8
    path_parent_depth: int = 1
    package_namespace_segments: int = 2
    package_component_index: int = 2
    semantic_sketch_latent_dim: int = 64
    semantic_sketch_max_tokens: int = 48
    file_nll_max_tokens: int = 8
    audit_sample_limit: int = 5000
    audit_target_limit_per_pattern: int = 50
    audit_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessSemanticResult:
    """Normalized process role and residual text."""

    rules_version: str
    profile: str
    nll_role: str
    residual_text: str
    nll_tokens: tuple[str, ...]
    residual_tokens: tuple[str, ...]
    source_field: str
    path: str
    cmd: str
    used_unknown: bool


class ProcessSemanticAudit:
    """Collect label-free process semantic diagnostics by split."""

    def __init__(
        self,
        dataset: str,
        config: ProcessSemanticConfig,
        labels_used: bool = False,
    ) -> None:
        self.dataset = str(dataset)
        self.config = config
        self.labels_used = bool(labels_used)
        self._split_counts: dict[str, int] = defaultdict(int)
        self._split_unknown: dict[str, int] = defaultdict(int)
        self._role_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._residual_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._token_count_sum: dict[str, int] = defaultdict(int)
        self._rows: list[dict[str, object]] = []
        self._target_counts: Counter[str] = Counter()

    def observe(
        self,
        split: str,
        event_index: int,
        node_id: int,
        path: object,
        cmd: object,
        semantic: ProcessSemanticResult,
    ) -> None:
        """Record one label-free process semantic observation."""
        split_key = str(split)
        self._split_counts[split_key] += 1
        self._increment_limited_counter(self._role_counts[split_key], semantic.nll_role)
        self._increment_limited_counter(
            self._residual_counts[split_key],
            semantic.residual_text,
        )
        self._token_count_sum[split_key] += len(semantic.residual_tokens)
        if semantic.used_unknown:
            self._split_unknown[split_key] += 1
        if self._should_keep_row(path, cmd):
            self._rows.append(
                {
                    "split": split_key,
                    "dataset": self.dataset,
                    "rules_version": semantic.rules_version,
                    "profile": semantic.profile,
                    "event_index": int(event_index),
                    "node_id": int(node_id),
                    "path": str(path or ""),
                    "cmd": str(cmd or ""),
                    "nll_role": semantic.nll_role,
                    "residual_text": semantic.residual_text,
                    "residual_token_count": len(semantic.residual_tokens),
                    "used_unknown": bool(semantic.used_unknown),
                },
            )

    def csv_rows(self) -> list[dict[str, object]]:
        """Return sampled label-free audit rows for CSV output."""
        return list(self._rows)

    def to_json_summary(
        self,
        train_available: bool = True,
        validation_available: bool = True,
        test_available: bool = True,
        skipped_reason: str = "",
    ) -> dict[str, object]:
        """Return a JSON-serializable label-free summary by split."""
        split_names = ("train", "validation", "test", "total")
        return {
            "dataset": self.dataset,
            "rules_version": self.config.rules_version,
            "labels_used": self.labels_used,
            "train_audit_available": bool(train_available),
            "validation_audit_available": bool(validation_available),
            "test_audit_available": bool(test_available),
            "train_validation_audit_skipped_reason": str(skipped_reason),
            "splits": {name: self._summary_for_split(name) for name in split_names},
        }

    def _should_keep_row(self, path: object, cmd: object) -> bool:
        if len(self._rows) < int(self.config.audit_sample_limit):
            return True
        text = f"{path or ''} {cmd or ''}".lower()
        for pattern in self.config.audit_targets:
            key = str(pattern).lower()
            if key and key in text:
                if self._target_counts[key] < int(self.config.audit_target_limit_per_pattern):
                    self._target_counts[key] += 1
                    return True
        return False

    def _increment_limited_counter(self, counter: Counter[str], key: str) -> None:
        if key in counter or len(counter) < _AUDIT_COUNTER_LIMIT:
            counter[key] += 1
            return
        counter[_AUDIT_OTHER_KEY] += 1

    def _summary_for_split(self, split: str) -> dict[str, object]:
        if split == "total":
            count = sum(self._split_counts.values())
            unknown = sum(self._split_unknown.values())
            role_counter: Counter[str] = Counter()
            residual_counter: Counter[str] = Counter()
            token_sum = sum(self._token_count_sum.values())
            for counter in self._role_counts.values():
                role_counter.update(counter)
            for counter in self._residual_counts.values():
                residual_counter.update(counter)
        else:
            count = self._split_counts[split]
            unknown = self._split_unknown[split]
            role_counter = self._role_counts[split]
            residual_counter = self._residual_counts[split]
            token_sum = self._token_count_sum[split]
        mean_tokens = float(token_sum / count) if count else 0.0
        return {
            "event_count": int(count),
            "unknown_count": int(unknown),
            "nll_role_cardinality": int(len(role_counter)),
            "residual_signature_cardinality": int(len(residual_counter)),
            "residual_token_count_mean": mean_tokens,
            "top_nll_roles": role_counter.most_common(20),
            "top_residual_signatures": residual_counter.most_common(20),
        }


def load_process_semantic_config(path: str | os.PathLike[str]) -> ProcessSemanticConfig:
    """Load process semantic settings from a YAML mapping."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError("process semantics config must be a mapping")

    profile_value = data.get("dataset_os_profile", {})
    if profile_value is None:
        profile_value = {}
    if not isinstance(profile_value, Mapping):
        raise ValueError("dataset_os_profile must be a mapping")

    audit_targets_value = data.get("process_semantic_audit_targets", [])
    if audit_targets_value is None:
        audit_targets_value = []
    if isinstance(audit_targets_value, str):
        audit_targets = (audit_targets_value,)
    else:
        audit_targets = tuple(str(value) for value in audit_targets_value)

    return ProcessSemanticConfig(
        rules_version=str(data.get("process_semantic_rules_version", "v1")).lower(),
        dataset_os_profile={
            str(key): str(value) for key, value in profile_value.items()
        },
        process_nll_max_tokens=_config_int(data, "process_nll_max_tokens", 6),
        process_residual_max_tokens=_config_int(
            data,
            "process_residual_max_tokens",
            8,
        ),
        path_parent_depth=_config_int(data, "path_parent_depth", 1),
        package_namespace_segments=_config_int(data, "package_namespace_segments", 2),
        package_component_index=_config_int(data, "package_component_index", 2),
        semantic_sketch_latent_dim=_config_int(data, "semantic_sketch_latent_dim", 64),
        semantic_sketch_max_tokens=_config_int(data, "semantic_sketch_max_tokens", 48),
        file_nll_max_tokens=_config_int(data, "file_nll_max_tokens", 8),
        audit_sample_limit=_config_int(
            data,
            "process_semantic_audit_sample_limit",
            5000,
        ),
        audit_target_limit_per_pattern=_config_int(
            data,
            "process_semantic_audit_target_limit_per_pattern",
            50,
        ),
        audit_targets=audit_targets,
    )


def resolve_dataset_profile(dataset: str, config: ProcessSemanticConfig) -> str:
    """Return the effective OS process profile for a dataset."""
    rules_version = str(config.rules_version).strip().lower()
    if rules_version == "legacy":
        return "legacy"
    if rules_version not in {
        PROCESS_SEMANTIC_RULES_VERSION,
        PROCESS_SEMANTIC_COARSE_NLL_RULES_VERSION,
        PROCESS_SEMANTIC_PROCESS_KIND_NLL_RULES_VERSION,
    }:
        raise ValueError(f"unsupported process semantic rules version: {rules_version}")

    dataset_text = str(dataset)
    for pattern, profile in config.dataset_os_profile.items():
        if fnmatch.fnmatchcase(dataset_text, str(pattern)):
            profile_text = str(profile).strip().lower()
            if profile_text not in _SUPPORTED_V1_PROFILES:
                raise ValueError(f"unsupported dataset_os_profile: {profile_text}")
            return profile_text
    raise ValueError(f"missing dataset_os_profile for dataset: {dataset_text}")


def normalize_piece(value: object, max_len: int = 80) -> str:
    """Normalize a process semantic fragment to a bounded lowercase token."""
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_.:/|-]+", "_", text).strip("_")
    return (text or "unknown")[: max(int(max_len), 1)]


def _normalize_process_token(value: object, max_len: int = 80) -> str:
    token = normalize_piece(value, max_len=max_len).replace(".", "_").strip("_")
    return token or "unknown"


def _normalize_android_package_segment(value: object, max_len: int = 60) -> str | None:
    raw_text = str(value or "").strip().lower()
    if not raw_text:
        return None

    full_token = normalize_piece(raw_text, max_len=max(len(raw_text), 1))
    if (
        full_token == "unknown"
        or full_token != raw_text
        or not _ANDROID_PACKAGE_SEGMENT_RE.fullmatch(full_token)
    ):
        return None
    return normalize_piece(raw_text, max_len=max_len)


def normalize_process_semantics(
    dataset: str,
    path: object,
    cmd: object,
    config: ProcessSemanticConfig,
    max_tokens_per_node: int,
) -> ProcessSemanticResult:
    """Normalize a process node into NLL role and residual text semantics."""
    profile = resolve_dataset_profile(dataset, config)
    if profile == "legacy":
        return _legacy_semantics(dataset, path, cmd, config, max_tokens_per_node)
    if profile == "android":
        return _android_semantics(cmd, path, config)
    if profile == "linux":
        return _linux_semantics(path, cmd, config)
    if profile == "freebsd":
        return _freebsd_semantics(cmd, path, config)
    raise ValueError(f"process semantic profile is not implemented: {profile}")


def _config_int(data: Mapping[str, object], key: str, default: int) -> int:
    try:
        return int(data.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _bounded(tokens: list[str], limit: int) -> tuple[str, ...]:
    return tuple(tokens[: max(int(limit), 0)])


def _unknown_result(
    profile: str,
    path: object,
    cmd: object,
    source_field: str,
    config: ProcessSemanticConfig,
) -> ProcessSemanticResult:
    rules_version = str(config.rules_version).strip().lower()
    if rules_version == PROCESS_SEMANTIC_PROCESS_KIND_NLL_RULES_VERSION:
        nll_tokens = ("process",)
    else:
        nll_tokens = ("process", "unknown")
    return ProcessSemanticResult(
        rules_version=rules_version,
        profile=profile,
        nll_role="|".join(nll_tokens),
        residual_text="process unknown",
        nll_tokens=nll_tokens,
        residual_tokens=("process", "unknown"),
        source_field=source_field,
        path=str(path or ""),
        cmd=str(cmd or ""),
        used_unknown=True,
    )


def _coarse_process_role(profile: str, tokens: tuple[str, ...]) -> str:
    """Map exact process tokens to a stable role for NLL only."""
    token_text = " ".join(tokens).lower()
    if "unknown" in token_text:
        return "unknown_low_information"
    if profile == "android":
        if "pkg_com_android" in token_text:
            if "comp_browser" in token_text or "comp_email" in token_text:
                return "android_user_app"
            return "android_system_component"
        if "pkg_org_mozilla" in token_text or "firefox" in token_text:
            return "browser_like"
        return "android_user_app"

    if any(name in token_text for name in ("firefox", "chrome", "browser")):
        return "browser_like"
    if any(name in token_text for name in ("thunderbird", "email", "imap", "smtp")):
        return "mail_like"
    if any(name in token_text for name in ("sshd", "nginx", "httpd", "postgres")):
        return "network_service"
    if any(name in token_text for name in ("bash", "dash", "sh", "python", "perl")):
        return "shell_or_interpreter"
    if any(name in token_text for name in ("pkg", "apt", "yum", "dpkg")):
        return "package_manager"
    if any(name in token_text for name in ("lsof", "top", "vmstat", "ps")):
        return "system_monitor"
    if any(name in token_text for name in ("fluxbox", "dbus", "system", "cron")):
        return "system_daemon"
    return "process_other"


def _with_nll_role_policy(result: ProcessSemanticResult) -> ProcessSemanticResult:
    rules_version = str(result.rules_version).strip().lower()
    if rules_version == PROCESS_SEMANTIC_PROCESS_KIND_NLL_RULES_VERSION:
        nll_tokens = ("process",)
        return ProcessSemanticResult(
            rules_version=result.rules_version,
            profile=result.profile,
            nll_role="process",
            residual_text=result.residual_text,
            nll_tokens=nll_tokens,
            residual_tokens=result.residual_tokens,
            source_field=result.source_field,
            path=result.path,
            cmd=result.cmd,
            used_unknown=result.used_unknown,
        )
    if rules_version != PROCESS_SEMANTIC_COARSE_NLL_RULES_VERSION:
        return result
    role = _coarse_process_role(result.profile, result.residual_tokens)
    nll_tokens = ("process", role)
    return ProcessSemanticResult(
        rules_version=result.rules_version,
        profile=result.profile,
        nll_role="|".join(nll_tokens),
        residual_text=result.residual_text,
        nll_tokens=nll_tokens,
        residual_tokens=result.residual_tokens,
        source_field=result.source_field,
        path=result.path,
        cmd=result.cmd,
        used_unknown=result.used_unknown,
    )


def _linux_semantics(
    path: object,
    cmd: object,
    config: ProcessSemanticConfig,
) -> ProcessSemanticResult:
    label = classify_linux_process_nll(path, cmd)
    if label == "unknown_process":
        return _unknown_result("linux", path, cmd, "cmd", config)

    nll_tokens = tuple(linux_process_nll_role(path, cmd).split("|"))
    residual_tokens = _bounded(
        list(linux_process_natural_tokens(path, cmd)),
        config.process_residual_max_tokens,
    )
    return _with_nll_role_policy(ProcessSemanticResult(
        rules_version=str(config.rules_version).strip().lower(),
        profile="linux",
        nll_role="|".join(nll_tokens),
        residual_text=" ".join(residual_tokens),
        nll_tokens=nll_tokens,
        residual_tokens=residual_tokens,
        source_field="cmd",
        path=str(path or ""),
        cmd=str(cmd or ""),
        used_unknown=False,
    ))


def _android_semantics(
    cmd: object,
    path: object,
    config: ProcessSemanticConfig,
) -> ProcessSemanticResult:
    raw_cmd = str(cmd or "").strip()
    label = classify_android_cmd_nll(raw_cmd)
    if not label:
        return _unknown_result("android", path, cmd, "cmd", config)

    nll_tokens_tuple = tuple(android_process_nll_role(raw_cmd).split("|"))
    residual_tokens_tuple = android_process_residual_tokens(raw_cmd)
    return _with_nll_role_policy(ProcessSemanticResult(
        rules_version=str(config.rules_version).strip().lower(),
        profile="android",
        nll_role="|".join(nll_tokens_tuple),
        residual_text=" ".join(residual_tokens_tuple),
        nll_tokens=nll_tokens_tuple,
        residual_tokens=residual_tokens_tuple,
        source_field="cmd",
        path=str(path or ""),
        cmd=raw_cmd,
        used_unknown=False,
    ))


def _freebsd_semantics(
    cmd: object,
    path: object,
    config: ProcessSemanticConfig,
) -> ProcessSemanticResult:
    raw_cmd = str(cmd or "").strip()
    label = classify_freebsd_process_nll(raw_cmd)
    if label == "unknown_process":
        return _unknown_result("freebsd", path, cmd, "cmd", config)

    nll_tokens = tuple(freebsd_process_nll_role(raw_cmd).split("|"))
    residual_tokens = _bounded(
        list(freebsd_process_natural_tokens(raw_cmd)),
        config.process_residual_max_tokens,
    )
    return _with_nll_role_policy(ProcessSemanticResult(
        rules_version=str(config.rules_version).strip().lower(),
        profile="freebsd",
        nll_role="|".join(nll_tokens),
        residual_text=" ".join(residual_tokens),
        nll_tokens=nll_tokens,
        residual_tokens=residual_tokens,
        source_field="cmd",
        path=str(path or ""),
        cmd=raw_cmd,
        used_unknown=False,
    ))


def _legacy_semantics(
    dataset: str,
    path: object,
    cmd: object,
    config: ProcessSemanticConfig,
    max_tokens_per_node: int,
) -> ProcessSemanticResult:
    del dataset
    tokens: list[str] = []
    for match in _LEGACY_TOKEN_RE.finditer(str(path or "") + " " + str(cmd or "")):
        token = normalize_piece(match.group(0), max_len=40)
        if token != "unknown":
            tokens.append(token)
        if len(tokens) >= int(max_tokens_per_node):
            break
    if not tokens:
        return _unknown_result("legacy", path, cmd, "legacy_summary", config)

    nll_tokens = tuple(["process", *tokens])
    return ProcessSemanticResult(
        rules_version="legacy",
        profile="legacy",
        nll_role="|".join(nll_tokens),
        residual_text=" ".join(nll_tokens),
        nll_tokens=nll_tokens,
        residual_tokens=nll_tokens,
        source_field="legacy_summary",
        path=str(path or ""),
        cmd=str(cmd or ""),
        used_unknown=False,
    )
