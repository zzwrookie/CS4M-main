import os
import ipaddress
import re
import sys
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE", str(REPO_ROOT / "dataset_checkpoint.json"))

from cs4m.config.config import (  # type: ignore  # noqa: E402
    ROOT_GROUND_TRUTH_DIR,
    get_runtime_required_args,
    get_yml_cfg,
)
from cs4m.config.provnet_utils import (  # type: ignore  # noqa: E402
    PROCESS_NODE_TYPE,
    datetime_to_ns_time_US,
    fetch_process_node_rows,
    get_indexid2msg,
    init_database_connection,
    log,
)

# Typing aliases for clarity
NetflowRow = Tuple[str, str, str, str, str, str, int]
ProcessRow = Tuple[str, str, str, str, int]
FileRow = Tuple[str, str, str, int]
EventRow = Tuple[str, str, str, str, str, str, int]

# Keep exactly aligned with Orthrus rel2id semantic space (10 edge/event types).
ORTHRUS10_EVENT_TYPES: Set[str] = {
    "EVENT_CONNECT",
    "EVENT_EXECUTE",
    "EVENT_OPEN",
    "EVENT_READ",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG",
    "EVENT_SENDMSG",
    "EVENT_SENDTO",
    "EVENT_WRITE",
    "EVENT_CLONE",
}


def parse_split_days(split_names: Iterable[str]) -> List[int]:
    # 从配置里的日级 split 标识（例如 day_XX）提取数字部分，用作按天切分
    days = []
    for name in split_names:
        digits = "".join(filter(str.isdigit, str(name)))
        if digits:
            days.append(int(digits))
    return days


def get_dataset_splits(cfg, split_name: str) -> List[str]:
    new_attr = f"{split_name}_splits"
    old_attr = f"{split_name}_files"
    values = getattr(cfg.dataset, new_attr, None)
    if values is None:
        values = getattr(cfg.dataset, old_attr, [])
    return list(values)


def get_ground_truth_paths(cfg) -> List[str]:
    values = getattr(cfg.dataset, "ground_truth_paths", None)
    if values is None:
        values = getattr(cfg.dataset, "ground_truth_relative_path", [])
    return list(values)


def get_day_range(cfg) -> Tuple[int, int]:
    # Prefer current naming (`day_range`); fallback to legacy (`start_end_day_range`).
    day_range = getattr(cfg.dataset, "day_range", None)
    if day_range is None:
        day_range = getattr(cfg.dataset, "start_end_day_range")
    start_day, end_day = day_range
    return int(start_day), int(end_day)


def use_event_type_filter(cfg) -> bool:
    """
    Runtime switch for Orthrus-compatible 10-type event filtering.
    Priority:
      1) ENV EVENT_TYPE_FILTER (0/1 true/false)
      2) cfg.runtime.event_type_filter
    """
    env_raw = os.getenv("EVENT_TYPE_FILTER", "").strip().lower()
    if env_raw in {"1", "true", "yes", "on"}:
        return True
    if env_raw in {"0", "false", "no", "off"}:
        return False
    runtime_cfg = getattr(cfg, "runtime", object())
    return bool(getattr(runtime_cfg, "event_type_filter", True))


def fetch_node_tables(cur) -> Tuple[List[NetflowRow], List[ProcessRow], List[FileRow]]:
    # 直接从数据库读各类节点表
    cur.execute(
        """
        select node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port, index_id
        from netflow_node_table
        """
    )
    netflow_nodes = cur.fetchall()

    process_nodes, process_table_name = fetch_process_node_rows(cur)
    log(f"[INFO] using process node table: {process_table_name}")

    cur.execute(
        """
        select node_uuid, hash_id, path, index_id
        from file_node_table
        """
    )
    file_nodes = cur.fetchall()

    return netflow_nodes, process_nodes, file_nodes


def build_hash_uuid_index_map(
    netflow_nodes: Sequence[NetflowRow],
    process_nodes: Sequence[ProcessRow],
    file_nodes: Sequence[FileRow],
) -> Dict[str, Tuple[str, int]]:
    # 构建 hash_id -> (uuid, index_id) 映射，供时间窗节点导出使用
    mapping: Dict[str, Tuple[str, int]] = {}

    for uuid, h, *_rest, index_id in netflow_nodes:
        mapping[str(h)] = (str(uuid), int(index_id))

    for uuid, h, *_rest, index_id in process_nodes:
        mapping[str(h)] = (str(uuid), int(index_id))

    for uuid, h, *_rest, index_id in file_nodes:
        mapping[str(h)] = (str(uuid), int(index_id))

    return mapping


def build_uuid_index_map(
    netflow_nodes: Sequence[NetflowRow],
    process_nodes: Sequence[ProcessRow],
    file_nodes: Sequence[FileRow],
) -> Dict[str, int]:
    # 构建 uuid -> index_id 映射，供 ground truth 对齐使用
    uuid2index: Dict[str, int] = {}

    for uuid, *_rest, index_id in netflow_nodes:
        uuid2index[str(uuid)] = int(index_id)
    for uuid, *_rest, index_id in process_nodes:
        uuid2index[str(uuid)] = int(index_id)
    for uuid, *_rest, index_id in file_nodes:
        uuid2index[str(uuid)] = int(index_id)
    return uuid2index


def build_hash_to_type(
    netflow_nodes: Sequence[NetflowRow],
    process_nodes: Sequence[ProcessRow],
    file_nodes: Sequence[FileRow],
) -> Dict[str, str]:
    # 构建 hash_id -> 节点类型映射，便于过滤 actor 仅保留 process
    mapping: Dict[str, str] = {}
    for _uuid, h, *_rest in netflow_nodes:
        mapping[str(h)] = "netflow"
    for _uuid, h, *_rest in process_nodes:
        mapping[str(h)] = PROCESS_NODE_TYPE
    for _uuid, h, *_rest in file_nodes:
        mapping[str(h)] = "file"
    return mapping


def normalize_process_event_row(row_dict: Dict[str, Any], hash2type: Dict[str, str]) -> Optional[Dict[str, Any]]:
    actor_hash = str(row_dict.get("actorID"))
    obj_hash = str(row_dict.get("objectorID"))
    actor_type = hash2type.get(actor_hash)
    obj_type = hash2type.get(obj_hash)

    if actor_type != PROCESS_NODE_TYPE and obj_type != PROCESS_NODE_TYPE:
        return None
    if actor_type == PROCESS_NODE_TYPE:
        return dict(row_dict)

    normalized = dict(row_dict)
    normalized["actorID"] = obj_hash
    normalized["actor_msg"] = row_dict.get("objector_msg")
    normalized["objectorID"] = actor_hash
    normalized["objector_msg"] = row_dict.get("actor_msg")
    return normalized


def tokenize_msg(msg: str) -> List[str]:
    text = str(msg).strip()
    if not text:
        return ["NA"]
    return text.split()


NODE_TYPE_TOKENS = {
    "process": "PROC",
    "file": "FILE",
    "netflow": "NET",
    "unknown": "UNK",
}

EVENT_PAYLOAD_STYLE = os.getenv("CLAD_EVENT_PAYLOAD_STYLE", "semantic_compact").strip().lower()
PROCESS_SUMMARY_MAX_TOKENS = max(3, int(os.getenv("CLAD_PROCESS_SUMMARY_MAX_TOKENS", "8")))
FILE_SUMMARY_MAX_TOKENS = max(2, int(os.getenv("CLAD_FILE_SUMMARY_MAX_TOKENS", "4")))
NETFLOW_SUMMARY_MAX_TOKENS = max(1, int(os.getenv("CLAD_NETFLOW_SUMMARY_MAX_TOKENS", "2")))
UNKNOWN_SUMMARY_MAX_TOKENS = max(1, int(os.getenv("CLAD_UNKNOWN_SUMMARY_MAX_TOKENS", "4")))

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    flags=re.IGNORECASE,
)
_HEXISH_RE = re.compile(r"^[0-9a-f]{12,}$", flags=re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_LIST_SPLIT_RE = re.compile(r"[|;,]+")
_SHELL_OPERATOR_TOKENS = {
    ">": "REDIR_OUT",
    ">>": "REDIR_APPEND",
    "<": "REDIR_IN",
    "|": "PIPE",
    "||": "OR",
    "&&": "AND",
    ";": "SEQ",
}


def split_node_message(msg: str) -> Tuple[str, List[str]]:
    raw_tokens = tokenize_msg(msg)
    if not raw_tokens:
        return "unknown", []

    first = str(raw_tokens[0]).strip().lower()
    if first in NODE_TYPE_TOKENS:
        node_type = first
        payload_tokens = raw_tokens[1:]
    else:
        node_type = "unknown"
        payload_tokens = raw_tokens

    return node_type, payload_tokens


def _normalize_semantic_piece(value: str, fallback: str = "na", max_len: int = 32) -> str:
    text = str(value).strip().lower()
    if not text:
        return fallback
    if _UUID_RE.fullmatch(text):
        return "uuid"
    if _HEXISH_RE.fullmatch(text):
        return "hex"
    if text.isdigit():
        return "num"
    text = _NON_ALNUM_RE.sub("_", text).strip("_")
    if not text:
        return fallback
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text or fallback


def _dedupe_keep_order(tokens: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for token in tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _bucket_count_token(prefix: str, count: int) -> str:
    count = max(0, int(count))
    if count <= 1:
        label = "1"
    elif count <= 2:
        label = "2"
    elif count <= 4:
        label = "3_4"
    elif count <= 8:
        label = "5_8"
    elif count <= 16:
        label = "9_16"
    elif count <= 32:
        label = "17_32"
    elif count <= 64:
        label = "33_64"
    elif count <= 128:
        label = "65_128"
    else:
        label = "129p"
    return f"{prefix}_{label}"


def _looks_like_path(token: str) -> bool:
    text = str(token).strip()
    return "/" in text or "\\" in text


def _looks_like_flag(token: str) -> bool:
    text = str(token).strip()
    return text.startswith("-") and text not in {"-", "--"}


def _normalize_path_parts(path_value: str) -> Tuple[str, List[str]]:
    raw = str(path_value).strip()
    if not raw:
        return "na", []
    norm = raw.replace("\\", "/")
    norm = re.sub(r"/+", "/", norm).strip()
    if not norm:
        return "na", []

    root = "rel"
    if norm.startswith("/"):
        stripped = norm[1:]
        segments = [seg for seg in stripped.split("/") if seg]
        if segments:
            root = _normalize_semantic_piece(segments[0], fallback="root")
    else:
        segments = [seg for seg in norm.split("/") if seg]

    return root, segments


def _basename_signature(path_value: str) -> str:
    _root, segments = _normalize_path_parts(path_value)
    if not segments:
        return "path"
    name = segments[-1]
    lower = name.lower()
    if _UUID_RE.fullmatch(lower):
        return "uuid"
    base_part = lower
    ext_piece = ""
    if "." in lower:
        parts = [part for part in lower.split(".") if part]
        if parts:
            base_part = parts[0]
            for suffix in parts[1:]:
                if any(ch.isalpha() for ch in suffix):
                    ext_piece = suffix
                    break

    base_norm = _normalize_semantic_piece(base_part, fallback="path", max_len=24)
    if ext_piece:
        ext_norm = _normalize_semantic_piece(ext_piece, fallback="x", max_len=12)
        return f"{base_norm}_{ext_norm}"
    return base_norm


def _summarize_path_tokens(path_value: str, max_tokens: int) -> List[str]:
    root, segments = _normalize_path_parts(path_value)
    if not segments:
        return ["PATH_na"][:max_tokens]

    basename = segments[-1]
    parent = segments[-2] if len(segments) >= 2 else ""
    tokens = [f"ROOT_{root}"]
    if parent:
        tokens.append(f"DIR_{_normalize_semantic_piece(parent, fallback='dir', max_len=20)}")
    tokens.append(f"PATH_{_basename_signature(basename)}")
    return _dedupe_keep_order(tokens)[:max_tokens]


def _list_value_count(value: str) -> int:
    text = str(value).strip()
    if not text:
        return 0
    parts = [part for part in _LIST_SPLIT_RE.split(text) if part]
    return len(parts) if parts else 1


def _value_signature(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "na"
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered
    if text.isdigit():
        return _bucket_count_token("N", int(text))
    if _looks_like_path(text):
        return _basename_signature(text)
    if "|" in text or ";" in text:
        return _bucket_count_token("L", _list_value_count(text))
    if ":" in text and not text.startswith("http"):
        pieces = [piece for piece in text.split(":") if piece]
        if len(pieces) > 2:
            return _bucket_count_token("L", len(pieces))
    try:
        ipaddress.ip_address(text)
        return f"IP_{text.replace('.', '_').replace(':', '_')}"
    except ValueError:
        pass
    return _normalize_semantic_piece(text, fallback="val", max_len=24)


def _summarize_assignment_token(token: str) -> str:
    key, value = str(token).split("=", 1)
    key_norm = _normalize_semantic_piece(key, fallback="key", max_len=20)
    value_norm = _value_signature(value)
    prefix = "ENV" if str(key).isupper() else "KV"
    return f"{prefix}_{key_norm}_{value_norm}"


def _summarize_process_payload(payload_tokens: Sequence[str]) -> List[str]:
    if not payload_tokens:
        return ["PROC_na"]

    exe_token = str(payload_tokens[0]).strip()
    path_tokens = _summarize_path_tokens(exe_token, max_tokens=2)
    exe_name = _basename_signature(exe_token)
    summary = [f"EXE_{exe_name}"]
    for token in path_tokens:
        if token.startswith("DIR_"):
            summary.append(token)
            break

    tail = [str(tok).strip() for tok in payload_tokens[1:] if str(tok).strip()]
    while tail:
        head = tail[0]
        if head == exe_token or _normalize_semantic_piece(head, fallback="x", max_len=24) == exe_name:
            tail = tail[1:]
            continue
        break

    priority_tokens: List[str] = []
    fallback_tokens: List[str] = []
    idx = 0
    while idx < len(tail):
        token = tail[idx]
        if token in _SHELL_OPERATOR_TOKENS:
            priority_tokens.append(_SHELL_OPERATOR_TOKENS[token])
            idx += 1
            continue

        if _looks_like_flag(token):
            flag = _normalize_semantic_piece(token.lstrip("-"), fallback="flag", max_len=20)
            if idx + 1 < len(tail):
                next_token = tail[idx + 1]
                if next_token not in _SHELL_OPERATOR_TOKENS and not _looks_like_flag(next_token):
                    priority_tokens.append(f"FLAG_{flag}_{_value_signature(next_token)}")
                    idx += 2
                    continue
            priority_tokens.append(f"FLAG_{flag}")
            idx += 1
            continue

        if "=" in token and not token.startswith("=") and not token.endswith("="):
            priority_tokens.append(_summarize_assignment_token(token))
            idx += 1
            continue

        if _looks_like_path(token):
            fallback_tokens.append(f"ARG_{_basename_signature(token)}")
            idx += 1
            continue

        fallback_tokens.append(f"ARG_{_normalize_semantic_piece(token, fallback='arg', max_len=20)}")
        idx += 1

    detail_budget = max(0, PROCESS_SUMMARY_MAX_TOKENS - len(summary))
    detail_tokens = _dedupe_keep_order(priority_tokens)
    if len(detail_tokens) < detail_budget:
        detail_tokens.extend(
            token
            for token in _dedupe_keep_order(fallback_tokens)
            if token not in set(detail_tokens)
        )
    return _dedupe_keep_order(summary + detail_tokens)[:PROCESS_SUMMARY_MAX_TOKENS]


def _summarize_file_payload(payload_tokens: Sequence[str]) -> List[str]:
    if not payload_tokens:
        return ["PATH_na"]
    path_value = " ".join(map(str, payload_tokens)).strip()
    return _summarize_path_tokens(path_value, max_tokens=FILE_SUMMARY_MAX_TOKENS)


def _summarize_netflow_payload(payload_tokens: Sequence[str]) -> List[str]:
    text = " ".join(map(str, payload_tokens)).strip()
    if not text or text.upper() == "NA":
        return ["NET_na"]

    ip_text = text
    port_text = ""
    if ":" in text and text.count(":") == 1:
        ip_candidate, port_candidate = text.rsplit(":", 1)
        if port_candidate.isdigit():
            ip_text, port_text = ip_candidate, port_candidate

    ip_token = f"IP_{ip_text.replace('.', '_').replace(':', '_')}"
    tokens = [ip_token]
    if port_text:
        tokens.append(f"PORT_{port_text}")
    return _dedupe_keep_order(tokens)[:NETFLOW_SUMMARY_MAX_TOKENS]


def _summarize_unknown_payload(payload_tokens: Sequence[str]) -> List[str]:
    tokens = [f"RAW_{_normalize_semantic_piece(token, fallback='na', max_len=24)}" for token in payload_tokens]
    cleaned = [token for token in _dedupe_keep_order(tokens) if token != "RAW_na"]
    return cleaned[:UNKNOWN_SUMMARY_MAX_TOKENS] or ["RAW_na"]


def summarize_node_payload(node_type: str, payload_tokens: Sequence[str]) -> List[str]:
    if EVENT_PAYLOAD_STYLE in {"raw", "legacy"}:
        return list(payload_tokens)

    kind = str(node_type).strip().lower()
    if kind == "process":
        return _summarize_process_payload(payload_tokens)
    if kind == "file":
        return _summarize_file_payload(payload_tokens)
    if kind == "netflow":
        return _summarize_netflow_payload(payload_tokens)
    return _summarize_unknown_payload(payload_tokens)


def encode_event_text(row: Dict[str, Any]) -> str:
    actor_type, actor_tokens = split_node_message(str(row["actor_msg"]))
    object_type, object_tokens = split_node_message(str(row["objector_msg"]))
    action_tok = str(row["action"]).strip() or "EVENT_UNKNOWN"
    actor_summary = summarize_node_payload(actor_type, actor_tokens)
    object_summary = summarize_node_payload(object_type, object_tokens)
    return " ".join(
        [
            f"A_{NODE_TYPE_TOKENS.get(actor_type, NODE_TYPE_TOKENS['unknown'])}",
            *actor_summary,
            action_tok,
            f"O_{NODE_TYPE_TOKENS.get(object_type, NODE_TYPE_TOKENS['unknown'])}",
            *object_summary,
        ]
    )


def _normalize_optional_index(value: Optional[int]) -> int:
    if value is None:
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def compute_event_label(
    src_idx: Optional[int],
    dst_idx: Optional[int],
    abnormal_nodes: Set[int],
) -> int:
    src_bad = src_idx is not None and int(src_idx) in abnormal_nodes
    dst_bad = dst_idx is not None and int(dst_idx) in abnormal_nodes
    if src_bad and dst_bad:
        return 2
    if src_bad or dst_bad:
        return 1
    return 0


def build_process_event_texts(
    df: pd.DataFrame,
    hash2type: Dict[str, str],
) -> List[str]:
    events: List[str] = []

    for row in df.itertuples(index=False):
        normalized = normalize_process_event_row(row._asdict(), hash2type)
        if normalized is None:
            continue
        events.append(encode_event_text(normalized))

    return events


def split_train_val_by_process(
    df: pd.DataFrame,
    hash2type: Dict[str, str],
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    When val_splits is empty, carve a validation subset from train days without leaking
    the same owning process into both train and val.
    """
    if df.empty or float(val_ratio) <= 0.0:
        return df.copy(), pd.DataFrame(columns=df.columns)

    owner_hashes: List[str] = []
    for row in df.itertuples(index=False):
        actor_hash = str(getattr(row, "actorID"))
        object_hash = str(getattr(row, "objectorID"))
        actor_type = hash2type.get(actor_hash)
        object_type = hash2type.get(object_hash)
        if actor_type == PROCESS_NODE_TYPE:
            owner_hashes.append(actor_hash)
        elif object_type == PROCESS_NODE_TYPE:
            owner_hashes.append(object_hash)
        else:
            owner_hashes.append("")

    owner_series = pd.Series(owner_hashes, index=df.index, dtype="string")
    valid_owner_mask = owner_series != ""
    unique_processes = owner_series[valid_owner_mask].dropna().unique().tolist()
    if len(unique_processes) < 2:
        return df.copy(), pd.DataFrame(columns=df.columns)

    rng = np.random.default_rng(int(seed))
    unique_processes = np.asarray(unique_processes, dtype=object)
    rng.shuffle(unique_processes)
    num_val_processes = int(np.ceil(len(unique_processes) * float(val_ratio)))
    num_val_processes = max(1, min(num_val_processes, len(unique_processes) - 1))
    val_processes = set(unique_processes[:num_val_processes].tolist())

    val_mask = owner_series.isin(val_processes)
    val_df = df[val_mask].copy()
    train_df = df[~val_mask].copy()
    return train_df, val_df


def build_process_event_records(
    df: pd.DataFrame,
    hash2type: Dict[str, str],
    hash2uuid_index: Dict[str, Tuple[str, int]],
    abnormal_nodes: Set[int],
) -> Dict[str, Any]:
    records: Dict[str, List[Any]] = {
        "event_index": [],
        "timestamp_ns": [],
        "process_idx": [],
        "action": [],
        "object_type": [],
        "src_idx": [],
        "dst_idx": [],
        "text": [],
        "label": [],
    }

    next_event_index = 0
    for row in df.itertuples(index=False):
        normalized = normalize_process_event_row(row._asdict(), hash2type)
        if normalized is None:
            continue

        actor_hash = str(normalized.get("actorID"))
        obj_hash = str(normalized.get("objectorID"))
        actor_idx = hash2uuid_index.get(actor_hash, (None, None))[1]
        obj_idx = hash2uuid_index.get(obj_hash, (None, None))[1]
        process_idx = actor_idx if actor_idx is not None else -1
        event_label = compute_event_label(actor_idx, obj_idx, abnormal_nodes)

        records["event_index"].append(next_event_index)
        records["timestamp_ns"].append(int(normalized.get("timestamp", -1)))
        records["process_idx"].append(int(process_idx))
        records["action"].append(str(normalized.get("action", "")))
        records["object_type"].append(str(hash2type.get(obj_hash, "unknown")))
        records["src_idx"].append(_normalize_optional_index(actor_idx))
        records["dst_idx"].append(_normalize_optional_index(obj_idx))
        records["text"].append(encode_event_text(normalized))
        records["label"].append(int(event_label))
        next_event_index += 1

    return {
        "schema_version": 2,
        "event_index": records["event_index"],
        "timestamp_ns": records["timestamp_ns"],
        "process_idx": records["process_idx"],
        "action": records["action"],
        "object_type": records["object_type"],
        "src_idx": records["src_idx"],
        "dst_idx": records["dst_idx"],
        "text": records["text"],
        "label": records["label"],
    }

def load_checkpoint(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"processed_days": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"[WARN] failed to load checkpoint {path}: {e}, start fresh")
        return {"processed_days": []}


def save_checkpoint(path: str, processed_days: List[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"processed_days": sorted(processed_days)}, f)


def resolve_gt_path(rel_path: str) -> str:
    # Ground Truth 位置存在 darpa 子目录时做降级查找
    direct = os.path.join(ROOT_GROUND_TRUTH_DIR, rel_path)
    darpa = os.path.join(ROOT_GROUND_TRUTH_DIR, "darpa", rel_path)
    if os.path.exists(direct):
        return direct
    return darpa


def _dataset_gt_dir(cfg) -> str:
    # prefer folder inferred from first ground_truth_paths entry (e.g., "E3-CADETS/xxx.csv")
    rel_list = get_ground_truth_paths(cfg)
    if rel_list:
        first = str(rel_list[0])
        head = first.split("/", 1)[0]
        if head:
            return head
    return str(cfg.dataset.name).replace("_", "-")


def load_ground_truth_indices(cfg, uuid2index: Dict[str, int], output_dir: str) -> List[int]:
    # 读取 ground truth CSV，根据 uuid 查数据库映射到 index_id，保存 abnormal_nodes.pkl
    gt_indices = set()

    for rel_path in get_ground_truth_paths(cfg):
        abs_path = resolve_gt_path(rel_path)
        if not os.path.exists(abs_path):
            log(f"[WARN] ground truth file not found: {abs_path}")
            continue

        # 简单探测分隔符，尽量保持原格式
        detected_sep = None
        with open(abs_path, "r", encoding="utf-8") as fr:
            first_line = fr.readline()
            if "\t" in first_line:
                detected_sep = "\t"
            elif "," in first_line:
                detected_sep = ","

        df_gt = pd.read_csv(
            abs_path,
            sep=detected_sep,
            header=None,
            names=["uuid", "info", "old_index"],
            engine="python",
        )

        # 依据 uuid 映射到当前数据库的 index_id
        new_indices = []
        for uid in df_gt.iloc[:, 0]:
            uid = str(uid).strip()
            idx = uuid2index.get(uid, df_gt.loc[df_gt["uuid"] == uid, "old_index"].iloc[0])
            if idx is not None:
                gt_indices.add(idx)
            else:
                log(f"[WARN] uuid {uid} from {abs_path} not found in DB indices")
            new_indices.append(idx if idx is not None else "")

        # 回写带新 index 的 ground truth 文件，保持原分隔符
        df_gt["new_index"] = new_indices
        # 用新 index 替换 old_index 位置（第三列）
        out_df = df_gt[["uuid", "info", "new_index"]]
        out_df.to_csv(
            abs_path,
            sep=detected_sep or ",",
            header=False,
            index=False,
            encoding="utf-8",
        )
        log(f"[INFO] updated ground truth with new index_id -> {abs_path}")

    abnormal_nodes = sorted(gt_indices)
    if abnormal_nodes:
        gt_dir = os.path.join(ROOT_GROUND_TRUTH_DIR, _dataset_gt_dir(cfg))
        os.makedirs(gt_dir, exist_ok=True)
        path = os.path.join(gt_dir, "abnormal_nodes.pkl")
        with open(path, "wb") as f:
            pickle.dump(abnormal_nodes, f)
        log(f"Saved {len(abnormal_nodes)} abnormal nodes to {path}")

    return abnormal_nodes


def fetch_events(cur) -> List[EventRow]:
    # 从数据库事件表顺序取出所有事件
    cur.execute(
        """
        select src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec
        from event_table
        order by timestamp_rec
        """
    )
    return cur.fetchall()


def fetch_events_by_day(cur, year_month: str, day: int) -> List[EventRow]:
    # 按天分批读取事件，减少一次性内存占用
    start_str = f"{year_month}-{day:02d} 00:00:00"
    end_str = f"{year_month}-{day + 1:02d} 00:00:00"
    start_ns = datetime_to_ns_time_US(start_str)
    end_ns = datetime_to_ns_time_US(end_str)

    cur.execute(
        """
        select src_node, src_index_id, operation, dst_node, dst_index_id, event_uuid, timestamp_rec
        from event_table
        where timestamp_rec >= %s and timestamp_rec < %s
        order by timestamp_rec
        """,
        (start_ns, end_ns),
    )
    return cur.fetchall()


def build_dataset_df(events: Sequence[EventRow], indexid2msg: Dict[int, List[str]]) -> Tuple[pd.DataFrame, int]:
    # 将事件表转成可读 DataFrame（附带节点文本），返回跳过数量
    records = []
    skipped = 0

    for src_hash, src_idx, op, dst_hash, dst_idx, event_uuid, ts in events:
        try:
            src_idx_int = int(src_idx)
            dst_idx_int = int(dst_idx)
        except (TypeError, ValueError):
            skipped += 1
            continue

        src_desc = indexid2msg.get(src_idx_int)
        dst_desc = indexid2msg.get(dst_idx_int)
        if src_desc is None or dst_desc is None:
            skipped += 1
            continue

        records.append(
            [
                src_hash,
                " ".join(map(str, src_desc)),
                op,
                dst_hash,
                " ".join(map(str, dst_desc)),
                event_uuid,
                int(ts),
            ]
        )

    df = pd.DataFrame(
        records,
        columns=[
            "actorID",
            "actor_msg",
            "action",
            "objectorID",
            "objector_msg",
            "EventID",
            "timestamp",
        ],
    )
    df.sort_values(by="timestamp", ascending=True, inplace=True)
    return df, skipped


def build_split_payload(
    split_name: str,
    texts: List[str],
) -> Dict[str, Any]:
    if split_name == "test":
        return {
            "all": texts,
            "stream_mode": "all_no_gt_split",
        }
    return {
        "benign": texts,
    }


def write_split_pickle(
    output_dir: str,
    kind: str,
    split_name: str,
    payload: Dict[str, Any],
) -> str:
    split_dir_name = {
        "train": f"training_{kind}",
        "val": f"validating_{kind}",
        "test": f"testing_{kind}",
    }[split_name]
    file_name = {
        "train": f"data_train_{kind}s.pkl",
        "val": f"data_val_{kind}s.pkl",
        "test": f"data_test_{kind}s.pkl",
    }[split_name]
    split_dir = os.path.join(output_dir, kind, split_dir_name)
    os.makedirs(split_dir, exist_ok=True)
    path = os.path.join(split_dir, file_name)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def write_named_pickle(output_dir: str, relative_dir: str, file_name: str, payload: Dict[str, Any]) -> str:
    target_dir = os.path.join(output_dir, relative_dir)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, file_name)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path
def main(cfg=None):
    if cfg is None:
        # 优先读取真实 CLI；仅在无 CLI 参数时，才回落到环境变量 DATASET / 缺省数据集。
        cli_args = sys.argv[1:] if len(sys.argv) > 1 else [os.getenv("DATASET", "CADETS_E3")]
        args = get_runtime_required_args(args=cli_args)
        cfg = get_yml_cfg(args)

    output_dir = os.getenv("OUTPUT_DIR")
    if output_dir:
        output_dir = os.path.abspath(output_dir)
    else:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILE)
    checkpoint = load_checkpoint(checkpoint_path)
    processed_days = set(checkpoint.get("processed_days", []))

    cur, connect = init_database_connection(cfg)
    netflow_nodes, process_nodes, file_nodes = fetch_node_tables(cur)
    indexid2msg = get_indexid2msg(cur)
    hash2uuid_index = build_hash_uuid_index_map(netflow_nodes, process_nodes, file_nodes)
    uuid2index = build_uuid_index_map(netflow_nodes, process_nodes, file_nodes)
    hash2type = build_hash_to_type(netflow_nodes, process_nodes, file_nodes)

    year_month = cfg.dataset.year_month
    start_day, end_day = get_day_range(cfg)

    dataset_csv_path = os.path.join(output_dir, "dataset_csv.csv")
    if os.path.exists(dataset_csv_path) and not processed_days:
        os.remove(dataset_csv_path)

    header_written = os.path.exists(dataset_csv_path)
    total_skipped = 0
    total_rows = 0

    for day in range(start_day, end_day):
        if day in processed_days:
            log(f"[INFO] Day {day} already processed, skip (checkpoint).")
            continue

        events = fetch_events_by_day(cur, year_month, day)
        if not events:
            log(f"[INFO] Day {day} has no events, skip.")
            continue

        df_day, skipped = build_dataset_df(events, indexid2msg)
        total_skipped += skipped

        df_day["datetime"] = pd.to_datetime(df_day["timestamp"], unit="ns")
        df_day["day"] = day

        total_rows += len(df_day)
        df_day.to_csv(
            dataset_csv_path,
            mode="a",
            header=not header_written,
            index=False,
            encoding="utf-8",
        )
        header_written = True
        log(f"[INFO] Day {day} written {len(df_day)} rows (skipped {skipped}).")
        processed_days.add(day)
        save_checkpoint(checkpoint_path, list(processed_days))

    if total_skipped:
        log(f"[WARN] total skipped events due to missing index/msg: {total_skipped}")
    log(f"[INFO] Saved dataset CSV with {total_rows} rows to {dataset_csv_path}")

    # 重新加载全量 CSV 做后续时间窗导出和数据切分
    df = pd.read_csv(dataset_csv_path)
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ns")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"])
    if "day" not in df.columns:
        df["day"] = df["datetime"].dt.day

    event_type_filter = use_event_type_filter(cfg)
    if event_type_filter:
        total_before = int(len(df))
        uniq_before = int(df["action"].nunique(dropna=True))
        df = df[df["action"].isin(ORTHRUS10_EVENT_TYPES)].copy()
        total_after = int(len(df))
        uniq_after = int(df["action"].nunique(dropna=True))
        log(
            "[INFO] event_type_filter=ON "
            f"(events: {total_before} -> {total_after}, dropped={total_before - total_after}; "
            f"action_types: {uniq_before} -> {uniq_after})"
        )
    else:
        log("[INFO] event_type_filter=OFF (use full event stream)")

    _abnormal_nodes = set(load_ground_truth_indices(cfg, uuid2index, output_dir))

    train_days = parse_split_days(get_dataset_splits(cfg, "train"))
    val_splits = get_dataset_splits(cfg, "val")
    val_days = parse_split_days(val_splits) if val_splits else []
    test_days = parse_split_days(get_dataset_splits(cfg, "test"))
    val_ratio_from_train = float(getattr(cfg.dataset, "val_ratio_from_train", 0.0) or 0.0)
    val_split_seed = int(getattr(cfg.dataset, "val_split_seed", 42) or 42)

    # 按天切分后，仅导出当前主线需要的 event 视图与结构化事件视图。
    train_df_full = df[df["day"].isin(train_days)].copy()
    val_df_full = df[df["day"].isin(val_days)].copy() if val_days else pd.DataFrame(columns=df.columns)
    test_df_full = df[df["day"].isin(test_days)].copy()
    if val_df_full.empty and val_ratio_from_train > 0.0:
        train_df_full, val_df_full = split_train_val_by_process(
            df=train_df_full,
            hash2type=hash2type,
            val_ratio=val_ratio_from_train,
            seed=val_split_seed,
        )
        log(
            "[INFO] carved val from train days by process "
            f"(val_ratio={val_ratio_from_train:.3f}, seed={val_split_seed}) "
            f"-> train={len(train_df_full)} val={len(val_df_full)}"
        )
    log(
        "[INFO] split events after filtering "
        f"train={len(train_df_full)} val={len(val_df_full)} test={len(test_df_full)}"
    )

    split_frames = [
        ("train", train_df_full),
        ("test", test_df_full),
    ]
    if not val_df_full.empty:
        split_frames.append(("val", val_df_full))

    for split_name, split_df in split_frames:
        event_texts = build_process_event_texts(
            split_df,
            hash2type=hash2type,
        )
        event_payload = build_split_payload(split_name, event_texts)
        event_path = write_split_pickle(output_dir, "event", split_name, event_payload)
        log(f"Saved {split_name} events to {event_path} (total {len(event_texts)})")

        structured_events = build_process_event_records(
            split_df,
            hash2type=hash2type,
            hash2uuid_index=hash2uuid_index,
            abnormal_nodes=_abnormal_nodes,
        )
        structured_events["split"] = split_name
        structured_event_path = write_named_pickle(
            output_dir=output_dir,
            relative_dir=os.path.join("event", f"{ {'train': 'training', 'val': 'validating', 'test': 'testing'}[split_name] }_event"),
            file_name={
                "train": "data_train_events_structured.pkl",
                "val": "data_val_events_structured.pkl",
                "test": "data_test_events_structured.pkl",
            }[split_name],
            payload=structured_events,
        )
        label_counts = {
            "benign": int(sum(1 for x in structured_events["label"] if int(x) == 0)),
            "suspect": int(sum(1 for x in structured_events["label"] if int(x) == 1)),
            "malicious": int(sum(1 for x in structured_events["label"] if int(x) == 2)),
        }
        log(
            f"Saved {split_name} structured events to {structured_event_path} "
            f"(total {len(structured_events['event_index'])}, labels={label_counts})"
        )

    connect.close()
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)


if __name__ == "__main__":
    main()
