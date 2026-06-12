"""Runtime-visible semantic tokenization, residual text, and threshold helpers."""

from __future__ import annotations

from cs4m.semantics.clearscope_android import (
    CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE,
    CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE,
    clearscope_residual_text_v33b_e5_android_safe,
    clearscope_residual_text_v33_e5_android_safe,
)
from scripts.pipeline.config.runtime_config import *


def normalize_piece(value: object, max_len: int = 40) -> str:
    """Normalize a role/key fragment to a bounded lowercase token."""
    text = "" if value is None else str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_.:/|-]+", "_", text).strip("_")
    if not text:
        text = "unknown"
    limit = max(int(max_len), 1)
    return text[:limit]


def action_family(action: object) -> str:
    """Map raw event actions to a stable coarse action family."""
    text = normalize_piece(action, max_len=80).upper()
    if text in {"EVENT_READ"}:
        return "read"
    if text in {"EVENT_RECVFROM", "EVENT_RECVMSG", "EVENT_ACCEPT"}:
        return "recv"
    if text in {"EVENT_WRITE"}:
        return "write"
    if text in {"EVENT_SENDTO", "EVENT_SENDMSG", "EVENT_CONNECT"}:
        return "send"
    if text in {"EVENT_EXECUTE", "EVENT_CLONE"}:
        return "spawn"
    if text in {"EVENT_OPEN"}:
        return "open"
    if text in {"EVENT_FORK"}:
        return "fork"
    if text in {"EVENT_CLOSE"}:
        return "close"
    if text in {"EVENT_BOOT"}:
        return "boot"
    if text in {"EVENT_LSEEK"}:
        return "seek"
    if text in {"EVENT_CHANGE_PRINCIPAL"}:
        return "principal"
    if text in {"EVENT_MODIFY_PROCESS"}:
        return "modify"
    return "other"


RAW_ACTION_TOKENS = tuple(
    normalize_piece(action, max_len=80).lower()
    for action in sorted(ORTHRUS10_EVENT_TYPES)
)
RAW_ACTION_IDS = {token: index for index, token in enumerate(RAW_ACTION_TOKENS)}
RAW_ACTION_OTHER_ID = len(RAW_ACTION_IDS)


def raw_action_token(action: object) -> str:
    """Return the exact normalized event-action token used by slim residual semantics."""
    token = normalize_piece(action, max_len=80).lower()
    return token if token else "unknown"


def normalize_raw_action_token_for_sspm(action: object) -> str:
    """Return an uppercase raw action token for SSPM context/gates."""
    return str(action).strip().upper()


def raw_action_id(action: object) -> int:
    """Return a stable ID for a raw event action, with an explicit unknown bucket."""
    return int(RAW_ACTION_IDS.get(raw_action_token(action), RAW_ACTION_OTHER_ID))


def split_summary_tokens(summary: object, max_tokens: int) -> tuple[str, ...]:
    """Return normalized bounded summary tokens."""
    limit = max(int(max_tokens), 0)
    if limit <= 0 or summary is None:
        return ()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(str(summary).lower()):
        token = normalize_piece(match.group(0), max_len=40)
        if token != "unknown":
            tokens.append(token)
        if len(tokens) >= limit:
            break
    return tuple(tokens)


def role_head(
    kind: object,
    summary: object,
    max_tokens: int,
    dst_addr: object = "",
    dst_port: object = "",
) -> str:
    """Return a bounded role head, using coarse netflow scope and port for netflow NLL."""
    kind_token = normalize_piece(kind)
    if kind_token == "netflow":
        return netflow_nll_role(dst_addr, dst_port)
    tokens = split_summary_tokens(summary, max_tokens)
    if tokens:
        return "|".join((kind_token, *tokens))
    return kind_token


def _process_semantics_is_v1(config: ProcessSemanticConfig | None) -> bool:
    if config is None:
        return False
    return str(config.rules_version).strip().lower() in {
        "v1",
        "v2_coarse_nll",
        "v3_process_kind_nll",
    }


def _process_role_for_row(
    row: Mapping[str, Any],
    side: str,
    dataset: str,
    config: ProcessSemanticConfig | None,
    max_tokens_per_node: int,
) -> str:
    if not _process_semantics_is_v1(config):
        return role_head(
            "process",
            row.get(f"{side}_summary", row.get("process_name", "")),
            max_tokens_per_node,
        )
    assert config is not None
    result = normalize_process_semantics(
        dataset=dataset,
        path=row.get(f"{side}_process_path", ""),
        cmd=row.get(f"{side}_process_cmd", ""),
        config=config,
        max_tokens_per_node=max_tokens_per_node,
    )
    return result.nll_role


def _role_for_row_side(
    row: Mapping[str, Any],
    side: str,
    kind: str,
    dataset: str,
    process_semantic_config: ProcessSemanticConfig | None,
    max_tokens_per_node: int,
    summary_default: object = "",
    theia_netflow_policy: str = "scope_port",
) -> str:
    if kind == "process":
        return _process_role_for_row(
            row,
            side,
            dataset,
            process_semantic_config,
            max_tokens_per_node,
        )
    if is_clearscope_dataset(dataset) and kind == "netflow":
        return clearscope_netflow_nll_role()
    if is_clearscope_dataset(dataset) and kind == "file":
        return clearscope_file_nll_role(row.get(f"{side}_file_path", ""))
    if is_theia_dataset(dataset) and kind == "netflow":
        if str(theia_netflow_policy) == "fixed":
            return linux_netflow_fixed_nll_role()
        return linux_netflow_nll_role(
            row.get(f"{side}_addr", row.get("dst_addr", row.get("remote_ip", ""))),
            row.get(f"{side}_port", row.get("dst_port", row.get("remote_port", ""))),
        )
    if is_theia_dataset(dataset) and kind == "file":
        return linux_file_nll_role(row.get(f"{side}_file_path", ""))
    if is_cadets_dataset(dataset) and kind == "netflow":
        return freebsd_netflow_nll_role(
            row.get(f"{side}_addr", row.get("dst_addr", row.get("remote_ip", ""))),
            row.get(f"{side}_port", row.get("dst_port", row.get("remote_port", ""))),
        )
    if is_cadets_dataset(dataset) and kind == "file":
        return freebsd_file_nll_role()
    token_limit = max_tokens_per_node
    if kind == "file" and _process_semantics_is_v1(process_semantic_config):
        assert process_semantic_config is not None
        token_limit = process_semantic_config.file_nll_max_tokens
    return role_head(
        kind,
        row.get(f"{side}_summary", summary_default),
        token_limit,
        dst_addr=row.get("dst_addr", row.get("remote_ip", "")),
        dst_port=row.get("dst_port", row.get("remote_port", "")),
    )


def row_fields(
    row: Mapping[str, Any],
    max_tokens_per_node: int,
    dataset: str = "SYNTHETIC",
    process_semantic_config: ProcessSemanticConfig | None = None,
    theia_netflow_policy: str = "scope_port",
) -> dict[str, Any]:
    """Build the slim causal semantic fields for one event row."""
    action = row.get("action", "")
    object_type = row.get("object_type", "unknown")
    src_idx = int(row.get("src_idx", row.get("info_src", -1)))
    dst_idx = int(row.get("dst_idx", row.get("info_dst", -1)))
    info_src, info_dst, relation_id, info_src_type, info_dst_type = information_flow(
        src_idx,
        dst_idx,
        action,
        object_type,
    )
    relation_name = _relation_name(relation_id)
    object_kind = normalize_piece(object_type)
    src_kind = normalize_piece(row.get("src_kind", "process"))
    dst_kind = normalize_piece(row.get("dst_kind", object_kind))
    surface_src_type_name = "process" if src_kind == "unknown" else src_kind
    surface_dst_type_name = (
        object_kind
        if object_kind != "unknown"
        else ("unknown" if dst_kind == "unknown" else dst_kind)
    )
    src_type_name = ENTITY_TYPE_NAMES.get(int(info_src_type), "unknown")
    dst_type_name = ENTITY_TYPE_NAMES.get(int(info_dst_type), "unknown")
    text = row.get("text", "")
    src_role = _role_for_row_side(
        row,
        "src",
        src_kind,
        dataset,
        process_semantic_config,
        max_tokens_per_node,
        summary_default=row.get("process_name", row.get("src_name", "")),
        theia_netflow_policy=theia_netflow_policy,
    )
    dst_role = _role_for_row_side(
        row,
        "dst",
        (
            "process"
            if dst_kind == "process"
            else (object_kind if object_kind != "unknown" else dst_kind)
        ),
        dataset,
        process_semantic_config,
        max_tokens_per_node,
        summary_default=text,
        theia_netflow_policy=theia_netflow_policy,
    )
    action_key = raw_action_token(action)
    raw_action = normalize_raw_action_token_for_sspm(action)
    return {
        "action": raw_action,
        "raw_action": raw_action,
        "action_token": action_key,
        "action_id": raw_action_id(action),
        "object_type": object_kind,
        "src_role": src_role,
        "dst_role": dst_role,
        "src_type_name": src_type_name,
        "dst_type_name": dst_type_name,
        "surface_src_type_name": surface_src_type_name,
        "surface_dst_type_name": surface_dst_type_name,
        "relation": relation_name,
        "info_src": int(info_src),
        "info_dst": int(info_dst),
        "relation_id": int(relation_id),
        "info_src_type": int(info_src_type),
        "info_dst_type": int(info_dst_type),
        "action_given_src": (src_role, action_key),
        "dst_given_src_action": (src_role, action_key, dst_role),
        "relation_given_roles": (src_role, dst_role, relation_name),
    }


def _process_residual_text_for_row(
    row: Mapping[str, Any],
    side: str,
    dataset: str,
    process_semantic_config: ProcessSemanticConfig,
    max_tokens_per_node: int,
) -> str:
    return normalize_process_semantics(
        dataset=dataset,
        path=row.get(f"{side}_process_path", ""),
        cmd=row.get(f"{side}_process_cmd", ""),
        config=process_semantic_config,
        max_tokens_per_node=max_tokens_per_node,
    ).residual_text


def _clearscope_natural_node_tokens(row: Mapping[str, Any], side: str) -> tuple[str, ...]:
    kind = normalize_piece(row.get(f"{side}_kind", ""))
    if kind == "process":
        return android_process_natural_tokens(row.get(f"{side}_process_cmd", ""))
    if kind == "file":
        return clearscope_file_natural_tokens(row.get(f"{side}_file_path", ""))
    if kind == "netflow":
        return clearscope_netflow_natural_tokens()
    return (kind or "unknown",)


def _clearscope_natural_residual_text(row: Mapping[str, Any]) -> str:
    action = raw_action_token(row.get("action", "unknown"))
    tokens = [
        *_clearscope_natural_node_tokens(row, "src"),
        action,
        *_clearscope_natural_node_tokens(row, "dst"),
    ]
    return " ".join(str(token) for token in tokens if str(token).strip())


def _theia_natural_node_tokens(
    row: Mapping[str, Any],
    side: str,
    theia_netflow_policy: str = "scope_port",
) -> tuple[str, ...]:
    kind = normalize_piece(row.get(f"{side}_kind", ""))
    if kind == "process":
        return linux_process_natural_tokens(
            row.get(f"{side}_process_path", ""),
            row.get(f"{side}_process_cmd", ""),
        )
    if kind == "file":
        return linux_file_natural_tokens(row.get(f"{side}_file_path", ""))
    if kind == "netflow":
        if str(theia_netflow_policy) == "fixed":
            return linux_netflow_detail_or_fixed_natural_tokens(
                row.get(f"{side}_addr", row.get("dst_addr", row.get("remote_ip", ""))),
                row.get("src_addr", ""),
            )
        return linux_netflow_natural_tokens(
            row.get(f"{side}_addr", row.get("dst_addr", row.get("remote_ip", ""))),
            row.get("src_addr", ""),
        )
    return (kind or "unknown",)


def _theia_natural_residual_text(
    row: Mapping[str, Any],
    theia_netflow_policy: str = "scope_port",
) -> str:
    action = raw_action_token(row.get("action", "unknown"))
    tokens = [
        *_theia_natural_node_tokens(row, "src", theia_netflow_policy),
        action,
        *_theia_natural_node_tokens(row, "dst", theia_netflow_policy),
    ]
    return " ".join(str(token) for token in tokens if str(token).strip())


def _cadets_natural_node_tokens(row: Mapping[str, Any], side: str) -> tuple[str, ...]:
    kind = normalize_piece(row.get(f"{side}_kind", ""))
    if kind == "process":
        return freebsd_process_natural_tokens(row.get(f"{side}_process_cmd", ""))
    if kind == "file":
        return freebsd_file_natural_tokens(row.get(f"{side}_file_path", ""))
    if kind == "netflow":
        return freebsd_netflow_natural_tokens(
            row.get(f"{side}_addr", row.get("dst_addr", row.get("remote_ip", ""))),
            row.get("src_addr", ""),
        )
    return (kind or "unknown",)


def _cadets_natural_residual_text(row: Mapping[str, Any]) -> str:
    action = raw_action_token(row.get("action", "unknown"))
    tokens = [
        *_cadets_natural_node_tokens(row, "src"),
        action,
        *_cadets_natural_node_tokens(row, "dst"),
    ]
    return " ".join(str(token) for token in tokens if str(token).strip())


def _observe_process_semantic_audit(
    audit: ProcessSemanticAudit | None,
    split: str,
    row: Mapping[str, Any],
    dataset: str,
    process_cfg: ProcessSemanticConfig,
    max_tokens_per_node: int,
) -> None:
    if audit is None:
        return
    object_kind = normalize_piece(row.get("object_type", "unknown"))
    side_defaults = {"src": "process", "dst": object_kind}
    for side in ("src", "dst"):
        kind = normalize_piece(row.get(f"{side}_kind", side_defaults[side]))
        if kind != "process":
            continue
        result = normalize_process_semantics(
            dataset=dataset,
            path=row.get(f"{side}_process_path", ""),
            cmd=row.get(f"{side}_process_cmd", ""),
            config=process_cfg,
            max_tokens_per_node=max_tokens_per_node,
        )
        audit.observe(
            split=split,
            event_index=int(row.get("event_index", -1)),
            node_id=int(row.get(f"{side}_idx", -1)),
            path=row.get(f"{side}_process_path", ""),
            cmd=row.get(f"{side}_process_cmd", ""),
            semantic=result,
        )


def residual_text(
    row: Mapping[str, Any],
    dataset: str = "SYNTHETIC",
    process_semantic_config: ProcessSemanticConfig | None = None,
    max_tokens_per_node: int = 8,
    theia_netflow_policy: str = "scope_port",
    semantic_mode: str = CLEARSCOPE_REFINED_SEMANTIC_MODE,
) -> str:
    """Return residual text, using exact IP only for netflow rows."""
    if is_clearscope_dataset(dataset):
        clearscope_mode = normalize_clearscope_semantic_mode(semantic_mode)
        if clearscope_mode == CLEARSCOPE_LEGACY_SEMANTIC_MODE:
            return _clearscope_natural_residual_text(row)
        if clearscope_mode == CLEARSCOPE_REFINED_SEMANTIC_MODE:
            return clearscope_residual_text_refined(dict(row))
        if clearscope_mode == CLEARSCOPE_V32_CACHE_ONLY_SEMANTIC_MODE:
            return clearscope_residual_text_v32_cache_only(dict(row))
        if clearscope_mode == CLEARSCOPE_V32_SEMANTIC_MODE:
            return clearscope_residual_text_v32(dict(row))
        if clearscope_mode == CLEARSCOPE_V33B_E5_ANDROID_SAFE_SEMANTIC_MODE:
            return clearscope_residual_text_v33b_e5_android_safe(dict(row))
        if clearscope_mode == CLEARSCOPE_V33_E5_ANDROID_SAFE_SEMANTIC_MODE:
            return clearscope_residual_text_v33_e5_android_safe(dict(row))
        if clearscope_mode == CLEARSCOPE_V31_SEMANTIC_MODE:
            return clearscope_residual_text_v31(dict(row))
        if clearscope_mode == CLEARSCOPE_V3_SEMANTIC_MODE:
            return clearscope_residual_text_v3(dict(row))
        return clearscope_residual_text_refined(dict(row))
    if is_theia_dataset(dataset):
        return _theia_natural_residual_text(row, theia_netflow_policy)
    if is_cadets_dataset(dataset):
        return _cadets_natural_residual_text(row)
    object_kind = normalize_piece(row.get("object_type", "unknown"))
    src_kind = normalize_piece(row.get("src_kind", "process"))
    dst_kind = normalize_piece(row.get("dst_kind", object_kind))
    parts = [
        f"action:{normalize_piece(row.get('action', 'unknown'))}",
        f"object:{object_kind}",
    ]
    if src_kind == "process" and _process_semantics_is_v1(process_semantic_config):
        assert process_semantic_config is not None
        parts.append(
            _process_residual_text_for_row(
                row,
                "src",
                dataset,
                process_semantic_config,
                max_tokens_per_node,
            ),
        )
    else:
        parts.extend(
            [
                f"a_{src_kind}",
                str(row.get("src_summary", row.get("process_name", ""))),
            ],
        )
    if object_kind == "netflow":
        netflow_text = (
            clearscope_netflow_residual_text()
            if is_clearscope_dataset(dataset)
            else _netflow_residual_text(row.get("dst_addr", row.get("remote_ip", "")))
        )
        parts.extend(
            [
                "o_netflow",
                netflow_text,
            ],
        )
        return " ".join(part for part in parts if str(part).strip())
    if is_clearscope_dataset(dataset) and (object_kind == "file" or dst_kind == "file"):
        parts.append(clearscope_file_residual_text(row.get("dst_file_path", "")))
        return " ".join(str(part) for part in parts if str(part).strip())
    if (
        (object_kind == "process" or dst_kind == "process")
        and _process_semantics_is_v1(process_semantic_config)
    ):
        assert process_semantic_config is not None
        parts.append(
            _process_residual_text_for_row(
                row,
                "dst",
                dataset,
                process_semantic_config,
                max_tokens_per_node,
            ),
        )
        return " ".join(str(part) for part in parts if str(part).strip())
    parts.append(str(row.get("dst_summary", row.get("dst_name", row.get("text", "")))))
    return " ".join(str(part) for part in parts if str(part).strip())


def empirical_tail_score(value: float, sorted_scores: Sequence[float]) -> float:
    """Return empirical validation-tail surprisal using (ge_count + 1) smoothing."""
    scores = _checked_sorted_scores(sorted_scores)
    val = float(value)
    if not math.isfinite(val):
        raise ValueError("value must be finite")
    first_ge = bisect.bisect_left(scores, val)
    ge_count = len(scores) - first_ge
    probability = (ge_count + 1.0) / (len(scores) + 1.0)
    return float(-math.log(probability))


def _empirical_tail_score_compact(value: float, sorted_scores: np.ndarray) -> float:
    val = float(value)
    if not math.isfinite(val):
        raise ValueError("value must be finite")
    count = int(sorted_scores.size)
    if count <= 0:
        raise ValueError("validation scores must not be empty")
    first_ge = int(np.searchsorted(sorted_scores, val, side="left"))
    ge_count = count - first_ge
    probability = (ge_count + 1.0) / (count + 1.0)
    return float(-math.log(probability))


def event_threshold_from_budget(
    sorted_scores: Sequence[float],
    expected_budget: float,
    horizon_events: int,
) -> float:
    """Return a validation empirical-tail threshold for an expected alert budget."""
    scores = _checked_sorted_scores(sorted_scores)
    horizon = int(horizon_events)
    if horizon <= 0:
        raise ValueError("horizon_events must be positive")
    budget = float(expected_budget)
    if not math.isfinite(budget) or budget <= 0.0:
        return float("inf")
    target_rate = min(max(budget / float(horizon), 0.0), 1.0)
    if target_rate >= 1.0:
        return float(scores[0])
    quantile = 1.0 - target_rate
    return float(np.quantile(np.asarray(scores, dtype=np.float64), quantile, method="higher"))


def event_threshold_from_quantile(
    sorted_scores: Sequence[float],
    quantile: float,
) -> float:
    """Return a validation threshold at a fixed empirical quantile."""
    scores = _checked_sorted_scores(sorted_scores)
    q = float(quantile)
    if not math.isfinite(q):
        raise ValueError("event threshold quantile must be finite")
    q = min(max(q, 0.0), 1.0)
    return float(np.quantile(np.asarray(scores, dtype=np.float64), q, method="higher"))


def event_threshold_from_mode(
    sorted_scores: Sequence[float],
    expected_budget: float,
    horizon_events: int,
    threshold_mode: str,
    fixed_quantile: float,
) -> float:
    """Return the validation-only event threshold for the requested ablation mode."""
    mode = str(threshold_mode or "budget").strip().lower()
    if mode in {"budget", "adaptive_rate"}:
        return event_threshold_from_budget(sorted_scores, expected_budget, horizon_events)
    if mode == "quantile":
        return event_threshold_from_quantile(sorted_scores, fixed_quantile)
    if mode in {"max", "validation_max"}:
        scores = _checked_sorted_scores(sorted_scores)
        return float(scores[-1])
    raise ValueError(f"unknown event_threshold_mode: {threshold_mode}")


def derive_event_calibration(
    validation_residual_scores: Sequence[float],
    event_score_mode: str,
    expected_budget: float,
    horizon_events: int,
    residual_sorted: Sequence[float] | None = None,
    threshold_mode: str = "budget",
    fixed_quantile: float = 0.999,
) -> EventCalibration:
    """Derive residual-only event calibration from raw validation residual scores."""
    _validate_active_event_score_mode(event_score_mode)
    residual_raw = np.asarray(validation_residual_scores, dtype=np.float32)
    if residual_raw.size == 0:
        raise ValueError("validation residual score array must not be empty")
    res_sorted = (
        _compact_sorted_scores(residual_sorted)
        if residual_sorted is not None
        else _compact_sorted_scores(residual_raw)
    )
    event_values = np.empty((int(residual_raw.size),), dtype=np.float32)
    for index in range(int(residual_raw.size)):
        residual_tail = _empirical_tail_score_compact(float(residual_raw[index]), res_sorted)
        event_values[index] = np.float32(
            _event_score_from_residual_tail(residual_tail, event_score_mode),
        )
    event_sorted = _compact_sorted_scores(event_values)
    threshold = event_threshold_from_mode(
        event_sorted,
        expected_budget,
        horizon_events,
        threshold_mode,
        fixed_quantile,
    )
    summary: dict[str, float | int] = {
        "count": int(event_values.size),
        "min": float(np.min(event_values)),
        "max": float(np.max(event_values)),
        "mean": float(np.mean(event_values)),
        "std": float(np.std(event_values)),
        "threshold": float(threshold),
        "final_threshold": float(threshold),
        "validation_max_score": float(event_sorted[-1]),
        "threshold_mode": str(threshold_mode),
        "threshold_quantile": float(fixed_quantile),
    }
    return EventCalibration(
        res_sorted,
        event_values,
        event_sorted,
        float(threshold),
        summary,
    )


def _score_distribution_summary(
    values: Sequence[float] | np.ndarray,
    threshold: float,
    prefix: str = "score",
) -> dict[str, float | int]:
    """Return finite score summary fields for validation/test diagnostics."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p99": 0.0,
            f"{prefix}_p999": 0.0,
            f"{prefix}_p9995": 0.0,
            f"{prefix}_p9999": 0.0,
            f"{prefix}_max": 0.0,
            "above_threshold_count": 0,
        }
    return {
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_p99": float(np.quantile(finite, 0.99, method="higher")),
        f"{prefix}_p999": float(np.quantile(finite, 0.999, method="higher")),
        f"{prefix}_p9995": float(np.quantile(finite, 0.9995, method="higher")),
        f"{prefix}_p9999": float(np.quantile(finite, 0.9999, method="higher")),
        f"{prefix}_max": float(np.max(finite)),
        "above_threshold_count": int(np.count_nonzero(finite >= float(threshold))),
    }


def _score_summary_from_event_calibration(
    calibration: EventCalibration,
) -> dict[str, float | int | str]:
    """Return validation score summary using the selected event-score distribution."""
    summary = _score_distribution_summary(
        calibration.event_scores,
        float(calibration.threshold),
    )
    summary.update(
        {
            "count": int(calibration.event_scores.size),
            "final_threshold": float(calibration.threshold),
            "threshold_mode": str(calibration.summary.get("threshold_mode", "")),
            "threshold_quantile": float(calibration.summary.get("threshold_quantile", 0.0)),
            "validation_max_score": float(calibration.summary.get("validation_max_score", 0.0)),
        },
    )
    return summary


def _phase3e_test_score_summary(
    event_scores: Sequence[float],
    threshold: float,
    event_alert_count: int | None = None,
) -> dict[str, float | int]:
    """Return Phase3E test event-score diagnostics."""
    summary = _score_distribution_summary(event_scores, threshold)
    summary["count"] = int(len(event_scores))
    summary["final_threshold"] = float(threshold)
    summary["test_above_threshold_count"] = int(summary.pop("above_threshold_count"))
    if event_alert_count is not None:
        summary["event_alert_count"] = int(event_alert_count)
    return summary


class _StreamingScoreSummary:
    """Bounded-memory score summary for long online test streams."""

    def __init__(self, threshold: float, bins: int = 20000) -> None:
        self.threshold = float(threshold)
        self.bins = max(int(bins), 100)
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0
        self.min_score = float("inf")
        self.max_score = float("-inf")
        self.above_threshold_count = 0
        self._hist_counts = np.zeros((self.bins,), dtype=np.int64)
        self._hist_min = 0.0
        self._hist_max = 1.0
        self._hist_overflow_rebuilds = 0

    def observe(self, value: float) -> None:
        score = float(value)
        if not math.isfinite(score):
            return
        self.count += 1
        self.sum += score
        self.sum_sq += score * score
        self.min_score = min(self.min_score, score)
        self.max_score = max(self.max_score, score)
        if score >= self.threshold:
            self.above_threshold_count += 1
        self._observe_histogram(score)

    def _observe_histogram(self, score: float) -> None:
        if score < self._hist_min or score > self._hist_max:
            self._expand_histogram(score)
        if self._hist_max <= self._hist_min:
            index = 0
        else:
            relative = (score - self._hist_min) / max(self._hist_max - self._hist_min, 1e-12)
            index = int(relative * (self.bins - 1))
        index = min(max(index, 0), self.bins - 1)
        self._hist_counts[index] += 1

    def _expand_histogram(self, score: float) -> None:
        old_counts = self._hist_counts
        old_min = float(self._hist_min)
        old_max = float(self._hist_max)
        new_min = min(old_min, float(score), 0.0)
        new_max = max(old_max, float(score), 1.0)
        if new_max <= new_min:
            new_max = new_min + 1.0
        rebuilt = np.zeros_like(old_counts)
        old_total = int(old_counts.sum())
        if old_total > 0:
            old_width = (old_max - old_min) / float(self.bins)
            centers = old_min + (np.arange(self.bins, dtype=np.float64) + 0.5) * old_width
            relative = (centers - new_min) / max(new_max - new_min, 1e-12)
            indices = np.floor(relative * self.bins).astype(np.int64)
            indices = np.clip(indices, 0, self.bins - 1)
            np.add.at(rebuilt, indices, old_counts)
        self._hist_counts = rebuilt
        self._hist_min = float(new_min)
        self._hist_max = float(new_max)
        self._hist_overflow_rebuilds += 1

    def _hist_quantile(self, quantile: float) -> float:
        if self.count <= 0:
            return 0.0
        q = min(max(float(quantile), 0.0), 1.0)
        rank = max(int(math.ceil(q * self.count)), 1)
        cumulative = np.cumsum(self._hist_counts)
        index = int(np.searchsorted(cumulative, rank, side="left"))
        index = min(max(index, 0), self.bins - 1)
        width = (self._hist_max - self._hist_min) / float(self.bins)
        estimate = self._hist_min + (float(index) + 1.0) * width
        return float(min(max(estimate, self.min_score), self.max_score))

    def summary(self, event_alert_count: int | None = None) -> dict[str, float | int]:
        if self.count <= 0:
            payload = _phase3e_test_score_summary([], self.threshold, event_alert_count)
            payload["test_score_quantile_method"] = "streaming_histogram"
            payload["score_quantile_method"] = "streaming_histogram"
            return payload
        mean = float(self.sum / self.count)
        variance = max(float(self.sum_sq / self.count) - mean * mean, 0.0)
        payload: dict[str, float | int] = {
            "score_min": float(self.min_score),
            "score_mean": mean,
            "score_std": float(math.sqrt(variance)),
            "score_p99": self._hist_quantile(0.99),
            "score_p999": self._hist_quantile(0.999),
            "score_p9995": self._hist_quantile(0.9995),
            "score_p9999": self._hist_quantile(0.9999),
            "score_max": float(self.max_score),
            "count": int(self.count),
            "final_threshold": float(self.threshold),
            "test_above_threshold_count": int(self.above_threshold_count),
            "test_score_quantile_method": "streaming_histogram",
            "score_quantile_method": "streaming_histogram",
            "score_histogram_bins": int(self.bins),
            "score_histogram_rebuilds": int(self._hist_overflow_rebuilds),
        }
        if event_alert_count is not None:
            payload["event_alert_count"] = int(event_alert_count)
        return payload


# Explicit cross-module imports; replaces the migration namespace bridge.
from scripts.pipeline.io.cache_payloads import (
    _checked_sorted_scores,
    _compact_sorted_scores,
    _relation_name,
)
from scripts.pipeline.outputs.metrics_summary import (
    _event_score_from_residual_tail,
    _validate_active_event_score_mode,
)
