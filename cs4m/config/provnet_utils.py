import hashlib
import os
import re
import time
from datetime import datetime
from time import mktime

try:
    from zoneinfo import ZoneInfo
except ModuleNotFoundError:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None

try:
    import psycopg2
    import psycopg2.extras as ex
except ModuleNotFoundError:  # pragma: no cover - optional until DB utilities are actually run
    psycopg2 = None
    ex = None

try:
    import pytz
except ModuleNotFoundError:  # pragma: no cover - optional until timezone helpers are used
    pytz = None


def datetime_to_ns_time_US(date: str) -> int:
    """
    将美东时区的时间字符串转换为纳秒时间戳。
    形如 'YYYY-mm-dd HH:MM:SS' -> int 纳秒
    """
    time_array = time.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = datetime.fromtimestamp(mktime(time_array))
    if pytz is not None:
        timestamp = pytz.timezone("US/Eastern").localize(dt).timestamp()
    elif ZoneInfo is not None:
        timestamp = dt.replace(tzinfo=ZoneInfo("US/Eastern")).timestamp()
    else:
        raise ModuleNotFoundError("pytz or zoneinfo is required for timezone conversion utilities.")
    return int(timestamp * 1_000_000_000)


def stringtomd5(text: str) -> str:
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()


def get_all_filelist(root: str):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Raw data directory does not exist: {root}")
    files = [name for name in os.listdir(root) if os.path.isfile(os.path.join(root, name))]
    files.sort()
    return files


# Read/receive-style relations are stored as object -> process to preserve actor/object semantics.
edge_reversed = {"EVENT_ACCEPT", "EVENT_READ", "EVENT_RECVFROM", "EVENT_RECVMSG"}
PROCESS_NODE_TYPE = "process"
SUBJECT_NODE_TABLE = os.getenv("CLAD_SUBJECT_NODE_TABLE", "subject_node_table")
PATH_NONE_TOKEN = "PATH_None"
CMD_NONE_TOKEN = "CMD_None"
IP_NONE_TOKEN = "IP_None"
PORT_NONE_TOKEN = "PORT_None"
EVENT_NETFLOW_USE_PORT = os.getenv("CLAD_EVENT_NETFLOW_USE_PORT", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _use_all_db_files(cfg) -> bool:
    runtime = getattr(cfg, "runtime", None)
    if runtime is not None and hasattr(runtime, "use_all_db_files"):
        return bool(runtime.use_all_db_files)

    preprocessing = getattr(cfg, "preprocessing", None)
    if preprocessing is not None:
        build_graphs = getattr(preprocessing, "build_graphs", None)
        if build_graphs is not None and hasattr(build_graphs, "use_all_files"):
            return bool(build_graphs.use_all_files)

    return False


def init_database_connection(cfg):
    """
    按运行期配置选择数据库名称并建立连接，返回 cursor 和 connection。
    """
    if psycopg2 is None:
        raise ModuleNotFoundError("psycopg2-binary is required for PostgreSQL access.")

    if _use_all_db_files(cfg):
        database_name = getattr(cfg.dataset, "db_name_all", None)
        if database_name is None:
            database_name = getattr(cfg.dataset, "database_all_file")
    else:
        database_name = getattr(cfg.dataset, "db_name", None)
        if database_name is None:
            database_name = getattr(cfg.dataset, "database")

    if cfg.database.host is not None:
        connect = psycopg2.connect(
            database=database_name,
            host=cfg.database.host,
            user=cfg.database.user,
            password=cfg.database.password,
            port=cfg.database.port,
        )
    else:
        connect = psycopg2.connect(
            database=database_name,
            user=cfg.database.user,
            password=cfg.database.password,
            port=cfg.database.port,
        )
    cur = connect.cursor()
    return cur, connect


def log(msg: str, *args) -> None:
    """
    简单的控制台日志格式：'YYYY-mm-dd HH:MM:SS - msg'
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {msg}", *args)


def _clean_text_part(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "none":
        return ""
    return text


def _normalize_path_token(path_value) -> str:
    path = _clean_text_part(path_value)
    if not path:
        return ""

    norm = path.replace("\\", "/").strip()
    norm = re.sub(r"/+", "/", norm)
    return norm


def _normalize_cmd_text(cmd_value) -> str:
    cmd = _clean_text_part(cmd_value)
    if not cmd:
        return ""
    return re.sub(r"\s+", " ", cmd).strip()


def _normalize_ip_token(value) -> str:
    token = _clean_text_part(value)
    return token or ""


def _normalize_port_token(value) -> str:
    token = _clean_text_part(value)
    return token or ""


def _join_message_parts(*parts) -> str:
    cleaned = []
    for part in parts:
        token = _clean_text_part(part)
        if token:
            cleaned.append(token)
    return " ".join(cleaned)


def _format_process_token(path_value, cmd_value) -> str:
    # Stay close to Orthrus: preserve full path / full cmd when present,
    # but skip missing fields instead of emitting PATH_None/CMD_None noise.
    return _join_message_parts(_normalize_path_token(path_value), _normalize_cmd_text(cmd_value))


def _format_file_token(path_value) -> str:
    return _normalize_path_token(path_value)


def _format_netflow_token(src_addr, src_port, dst_addr, dst_port) -> str:
    # Orthrus uses remote_ip by default, and only adds remote_port when enabled.
    remote_ip = _normalize_ip_token(dst_addr)
    remote_port = _normalize_port_token(dst_port)
    if remote_ip and EVENT_NETFLOW_USE_PORT and remote_port:
        return f"{remote_ip}:{remote_port}"
    return remote_ip


def fetch_process_node_rows(cur):
    """
    读取进程节点表。
    """
    query = """
        select node_uuid, hash_id, path, cmd, index_id
        from {table_name}
    """
    cur.execute(query.format(table_name=SUBJECT_NODE_TABLE))
    return cur.fetchall(), SUBJECT_NODE_TABLE


def get_indexid2msg(cur):
    """
    构造 {index_id: [type, msg]} 映射，供数据导出使用。
    """
    indexid2msg = {}

    cur.execute("select * from netflow_node_table;")
    records = cur.fetchall()
    log(f"Number of netflow nodes: {len(records)}")
    for row in records:
        # row: [node_uuid, hash_id, src_addr, src_port, dst_addr, dst_port, index_id]
        src_addr = row[2]
        src_port = row[3]
        dst_addr = row[4]
        dst_port = row[5]
        index_id = row[-1]
        indexid2msg[index_id] = ["netflow", _format_netflow_token(src_addr, src_port, dst_addr, dst_port)]

    records, process_table_name = fetch_process_node_rows(cur)
    log(f"Number of process nodes: {len(records)} (table={process_table_name})")
    for row in records:
        path = row[2]
        cmd = row[3]
        index_id = row[-1]
        indexid2msg[index_id] = [PROCESS_NODE_TYPE, _format_process_token(path, cmd)]

    cur.execute("select * from file_node_table;")
    records = cur.fetchall()
    log(f"Number of file nodes: {len(records)}")
    for row in records:
        path = _format_file_token(row[2])
        index_id = row[-1]
        indexid2msg[index_id] = ["file", path]

    return indexid2msg
