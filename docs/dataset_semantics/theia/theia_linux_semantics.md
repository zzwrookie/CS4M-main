# THEIA Linux 语义规则说明

本文解释 [cs4m/semantics/theia_linux.py](/mnt/h/CS4M-main/cs4m/semantics/theia_linux.py) 中每个函数的作用、输入输出和示例。该模块只根据运行时可见字段生成语义标签与 residual tokens，例如 process path、command、file path、IP、port；它不读取 ground truth，不使用攻击窗口，也不做训练或评估。

## 1. 模块目标

THEIA 数据集来自 Linux 环境。原始 provenance event 中的实体字段通常是低层字符串，例如：

```text
process path = /bin/bash
command      = bash -c /tmp/vugefal
file path    = /proc/123/task/456/status
dst_addr     = 128.55.12.4
dst_port     = 443
```

`theia_linux.py` 的目标是把这些字符串转成两类稳定语义：

- NLL role：粗粒度类别，用于按语义角色建模或分组，例如 `process|linux|shell`、`file|linux|proc_file`、`net|ip_internal_env|port_system`。
- natural tokens：细粒度 residual tokens，用于 Word2Vec/residual embedding，例如 `("process", "shell", "vugefal")`。

下游可以把这些 tokens 查表成向量。假设词表和嵌入矩阵为：

```text
vocab:
  process -> 0
  shell   -> 1
  vugefal -> 2

E =
[[1.0, 0.0, 0.0],
 [0.0, 1.0, 0.0],
 [0.2, 0.8, 0.1]]
```

对于 `("process", "shell", "vugefal")`，一个简单 mean embedding 是：

```text
x = (E[0] + E[1] + E[2]) / 3
  = ([1.0, 0.0, 0.0] + [0.0, 1.0, 0.0] + [0.2, 0.8, 0.1]) / 3
  = [0.4, 0.6, 0.0333]
```

这类矩阵计算发生在下游 embedding/score 模块中；`theia_linux.py` 本身负责稳定地产生可查表的类别和 token。

## 2. 常量和规则表

`THEIA_SEMANTIC_RULES_VERSION = "theia_linux_raw_detail_v2"` 标识当前 THEIA Linux 规则版本。恢复或比较 artifact 时，这个版本可以帮助判断 token 规则是否一致。

`INTERNAL_ENV_CIDR = 128.55.12.0/24` 是 THEIA 环境内部网段。`PRIVATE_CIDRS` 包含 `10.0.0.0/8`、`192.168.0.0/16`、`172.16.0.0/12`。`ZERO_CIDR = 0.0.0.0/8` 用来识别未知、占位或零地址。

`HOST_PSEUDO_PROCESS_RE` 识别形如 `128.55.12.10-m` 的 host pseudo process。`REPLAY_LOGDB_PATH_RE` 和 `REPLAY_CACHE_PATH_RE` 识别 `/data/f.../replay_logdb` 与 `/data/f.../replay_cache` 这类重放数据路径。

`MISSING_DETAIL_VALUES` 定义缺失值集合，例如空字符串、`none`、`unknown`、`path_none`。`CONTEXT_GENERIC_BASENAMES` 收集过于泛化的 basename，例如 `index`、`cache`、`log`、`fd`、`cmdline`；这些名字单独作为 detail 信息太弱，通常需要结合父目录。`PAYLOAD_LIKE_BASENAMES` 收集更像 payload 或关键程序名的 basename，例如 `vugefal`、`pass_mgr`、`memtrace_so`。`SHELL_INTERPRETERS` 和 `SHELL_CONTROL_TOKENS` 帮助从 shell/interpreter command 中提取真正有意义的参数。

## 3. 通用辅助函数

### `is_theia_dataset(dataset)`

判断数据集名是否采用 THEIA Linux 语义策略。逻辑是把输入转成大写字符串，并检查是否以 `THEIA_` 开头。

示例：

```text
is_theia_dataset("THEIA_E3") -> True
is_theia_dataset("CADETS_E3") -> False
```

### `normalize_token(value, max_len=80)`

把任意对象规整成可用 token：

- 转字符串、去首尾空白、转小写；
- 连续空白替换成 `_`；
- 非 `a-z0-9_` 字符替换成 `_`；
- 去掉首尾 `_`；
- 空结果变成 `unknown`；
- 最多保留 `max_len` 个字符。

示例：

```text
normalize_token("A/B c++", 10) -> "a_b_c"
normalize_token("", 80)        -> "unknown"
```

矩阵示例：归一化后的 token 作为词表 key。若 `normalize_token("A/B c++") = "a_b_c"`，且词表中 `a_b_c -> 5`，则 one-hot 向量可以写成：

```text
one_hot("a_b_c") = [0, 0, 0, 0, 0, 1]
```

若 embedding 矩阵 `E` 有 6 行，则该 token 的向量为 `E[5]`。

### `_is_missing_detail(value)`

判断 detail 字段是否属于缺失值集合。它是内部 helper，用于避免把 `none`、`unknown` 等字符串当作真实语义。

示例：

```text
_is_missing_detail("none")     -> True
_is_missing_detail("/bin/sh")  -> False
```

### `_split_command(cmd)`

把 command line 切成参数列表。优先使用 `shlex.split`，这样可以保留引号语义；如果 command 存在不完整引号导致解析失败，则退回到简单空白切分。缺失 command 返回空列表。

示例：

```text
_split_command('/bin/bash -c "cat /etc/passwd"')
  -> ["/bin/bash", "-c", "cat /etc/passwd"]

_split_command("none")
  -> []
```

### `_raw_basename(value)`

返回路径或字符串的最后一段，不做 token 归一化。它会去掉末尾 `/`。

示例：

```text
_raw_basename("/usr/bin/python3") -> "python3"
_raw_basename("/tmp/a/")          -> "a"
```

### `_normalized_basename(value)`

先取 `_raw_basename`，再用 `normalize_token(..., max_len=60)` 归一化。

示例：

```text
_normalized_basename("/home/admin/index.html") -> "index_html"
```

### `_parent_basename(path)`

返回路径父目录的 basename，并归一化。没有父目录时返回空字符串。

示例：

```text
_parent_basename("/home/admin/index.html") -> "admin"
_parent_basename("bash")                   -> ""
```

### `_is_numeric_pid_like(token)`

判断 token 是否全是数字。它主要用于区分 `/proc/123/...` 里的 PID 与普通目录名。

示例：

```text
_is_numeric_pid_like("123") -> True
_is_numeric_pid_like("fd")  -> False
```

### `_path_detail_token(path)`

从 path 中提取细粒度 detail token。主要规则：

- 缺失值返回空字符串；
- `"0"` 代表 `swapper`；
- `"/"` 代表 `root`；
- `/proc/<pid>/...` 交给 `_proc_file_detail`；
- basename 如果属于 `PAYLOAD_LIKE_BASENAMES`，直接使用；
- basename 如果过于泛化且父目录有意义，则返回 `父目录_basename`；
- 否则返回归一化 basename。

示例：

```text
_path_detail_token("/proc/123/fd/4")      -> "fd"
_path_detail_token("/home/admin/index.html") -> "admin_index_html"
_path_detail_token("/tmp/vugefal")        -> "vugefal"
_path_detail_token("0")                   -> "swapper"
```

### `_meaningful_command_arg(parts)`

从 command 参数列表中找出最有语义价值的参数。它会跳过 shell 控制符、选项、赋值、重定向等，优先选择包含 `/` 或 `.` 的路径/文件型参数，并通过 `_path_detail_token` 规整；如果没有这类参数，再选择普通非选项参数。

示例：

```text
parts = ["bash", "-c", "/tmp/vugefal", "--debug"]
_meaningful_command_arg(parts) -> "vugefal"
```

这个函数用于避免把 `bash`、`python` 这类解释器本身当作全部语义，而是尽量提取被执行脚本或 payload。

### `_process_detail_token(path, cmd)`

提取 process 的细粒度 detail。流程是：

1. 用 `_split_command` 切 command；
2. 如果 command 的第一个 token 是 shell/interpreter，则尝试从后续参数提取 `_meaningful_command_arg`；
3. 如果不能提取参数，则使用 executable；
4. 如果 command 不可用，则退回 `_path_detail_token(path)`。

示例：

```text
_process_detail_token("/bin/bash", "bash -c /tmp/vugefal")
  -> "vugefal"

_process_detail_token("/usr/bin/python", "python script.py")
  -> "script_py"
```

### `_first_command_token(cmd)`

返回 command 的第一个 executable token，并去掉路径和前导 `-`，再归一化。

示例：

```text
_first_command_token("/usr/bin/python3 -m http.server") -> "python3"
_first_command_token("-bash")                           -> "bash"
```

### `_path_basename(path)`

返回 path 的归一化 basename。特殊地，路径 `"0"` 返回 `swapper`。

示例：

```text
_path_basename("/usr/bin/ssh") -> "ssh"
_path_basename("0")            -> "swapper"
```

### `_combined_text(path, cmd)`

把 command 和 path 拼接成小写文本，供粗粒度分类时做包含匹配。

示例：

```text
_combined_text("/usr/lib/firefox/firefox", "firefox")
  -> "firefox /usr/lib/firefox/firefox"
```

## 4. Process 语义函数

### `classify_linux_process_nll(path, cmd)`

返回 process 的粗粒度类别。它优先看 command 的第一个 token；如果 command 不可用，就看 path basename；同时也会检查完整 command/path 文本。

主要类别包括：

```text
unknown_process, kernel_thread, host_pseudo_process,
scheduler_service, network_service, admin_tool, system_daemon,
shell, interpreter, browser, mail_client, ssh_service,
database_process, package_manager, audio_service, update_notifier,
desktop_env, core_util, browser_helper, system_user_bin,
system_helper, core_binary, process_other
```

示例：

```text
classify_linux_process_nll("/bin/bash", "bash -c /tmp/vugefal")
  -> "shell"

classify_linux_process_nll("/usr/lib/postgresql/9.1/bin/postgres", "postgres")
  -> "database_process"

classify_linux_process_nll("128.55.12.2-m", "n/a")
  -> "host_pseudo_process"
```

NLL 矩阵示例：该函数返回的 label 可以作为分类目标。假设类别顺序为：

```text
[kernel_thread, shell, browser, process_other]
```

对于 label `shell`，目标 one-hot 为：

```text
y = [0, 1, 0, 0]
```

如果模型对四类给出的概率是：

```text
p = [0.05, 0.80, 0.10, 0.05]
```

则粗粒度 NLL 为：

```text
NLL = -sum(y_i * log(p_i)) = -log(0.80) = 0.2231
```

`theia_linux.py` 只产生 `shell` 这个目标标签，概率矩阵和 NLL 计算在下游模型中完成。

### `linux_process_detail(path, cmd, label=None)`

返回 process 的 residual detail token。若调用方没有传 `label`，函数会先调用 `classify_linux_process_nll`。

特殊规则：

- `unknown_process` 返回 `unknown`；
- `host_pseudo_process` 返回 `internal_host`；
- `kernel_thread` 会把 swapper、kworker、flush 等统一成稳定 token；
- 一些常见别名会被保留或折叠，例如 `dhclient3 -> dhclient`、`unity_2d_shell -> unity_2d`。

示例：

```text
linux_process_detail("/bin/bash", "bash -c /tmp/vugefal", "shell")
  -> "vugefal"

linux_process_detail("0", "", "kernel_thread")
  -> "swapper"
```

### `linux_process_nll_role(path, cmd)`

返回带命名空间的 process NLL role，格式是：

```text
process|linux|<coarse_label>
```

示例：

```text
linux_process_nll_role("/bin/bash", "bash -c /tmp/vugefal")
  -> "process|linux|shell"
```

### `linux_process_natural_tokens(path, cmd)`

返回 process residual token tuple：

```text
("process", coarse_label, detail)
```

示例：

```text
linux_process_natural_tokens("/bin/bash", "bash -c /tmp/vugefal")
  -> ("process", "shell", "vugefal")
```

矩阵示例：若 `("process", "shell", "vugefal")` 对应行号 `[0, 1, 2]`，embedding 矩阵为：

```text
E =
[[1.0, 0.0],
 [0.0, 1.0],
 [0.5, 0.5]]
```

则 token mean embedding 为：

```text
x = (E[0] + E[1] + E[2]) / 3
  = ([1.0, 0.0] + [0.0, 1.0] + [0.5, 0.5]) / 3
  = [0.5, 0.5]
```

## 5. File 语义函数

### `classify_linux_file_nll(path)`

返回 Linux 文件路径的粗粒度类别。它会先把路径小写、压缩重复 `/`，再按 Linux 目录结构和 THEIA replay 路径规则分类。

主要类别包括：

```text
unknown_file, filesystem_root, linux_root_dir,
proc_cmdline, proc_file, pseudo_proc_file,
browser_cache_file, browser_profile_file,
user_download_file, user_source_or_dev_file, user_home_file,
shared_memory_file, runtime_state_file, tmp_file,
replay_log_file, replay_cache_file, database_state_file,
system_cache_file, system_log_file, var_state_file,
system_config_file, native_library_file,
system_binary_file, system_library_file, system_share_file,
usr_file, core_binary_file, core_library_file, system_sbin_file,
device_file, sysfs_file, root_home_file, boot_file,
relative_or_special_file, file_other
```

示例：

```text
classify_linux_file_nll("/proc/123/cmdline")
  -> "proc_cmdline"

classify_linux_file_nll("/home/admin/.mozilla/firefox/x/cache2/entries/ABCD")
  -> "browser_cache_file"

classify_linux_file_nll("/var/lib/postgresql/9.1/main/base/1/123")
  -> "database_state_file"

classify_linux_file_nll("relative.txt")
  -> "relative_or_special_file"
```

### `linux_file_detail(path, label=None)`

返回 file 的 residual detail token。若未传入 `label`，会先调用 `classify_linux_file_nll`。它通常使用 `_path_detail_token` 提取 basename 或 proc detail，并按类别设置 fallback。

示例：

```text
linux_file_detail("/home/admin/.mozilla/firefox/x/cache2/entries/ABCD",
                  "browser_cache_file")
  -> "abcd"

linux_file_detail("/proc/123/task/456/status", "proc_file")
  -> "task_status"

linux_file_detail("/", "filesystem_root")
  -> "root"
```

### `_proc_file_detail(path)`

解析 `/proc/<pid>/...` 路径的细节：

- `/proc/<pid>` 返回 `proc`；
- `/proc/<pid>/fd/...` 返回 `fd`；
- `/proc/<pid>/task/<tid>/<name>` 返回 `task_<name>`；
- 其他 `/proc/<pid>/<name>` 返回 `<name>`。

示例：

```text
_proc_file_detail("/proc/123/fd/4")
  -> "fd"

_proc_file_detail("/proc/123/task/456/status")
  -> "task_status"

_proc_file_detail("/proc/123/maps")
  -> "maps"
```

### `linux_file_nll_role(path)`

返回带命名空间的 file NLL role，格式是：

```text
file|linux|<coarse_label>
```

示例：

```text
linux_file_nll_role("/proc/123/task/456/status")
  -> "file|linux|proc_file"
```

### `linux_file_natural_tokens(path)`

返回 file residual token tuple：

```text
("file", coarse_label, detail)
```

示例：

```text
linux_file_natural_tokens("/proc/123/task/456/status")
  -> ("file", "proc_file", "task_status")
```

矩阵示例：假设词表和 embedding 为：

```text
file        -> 0
proc_file   -> 1
task_status -> 2

E =
[[0.6, 0.0, 0.0],
 [0.0, 0.7, 0.0],
 [0.1, 0.1, 0.8]]
```

则 `("file", "proc_file", "task_status")` 的 mean embedding 为：

```text
x = ([0.6, 0.0, 0.0] + [0.0, 0.7, 0.0] + [0.1, 0.1, 0.8]) / 3
  = [0.2333, 0.2667, 0.2667]
```

## 6. Netflow 语义函数

### `_parse_ipv4(value)`

把输入解析为 `ipaddress.IPv4Address`。空值或非法 IPv4 返回 `None`。

示例：

```text
_parse_ipv4("128.55.12.4") -> IPv4Address("128.55.12.4")
_parse_ipv4("not-ip")      -> None
```

### `linux_ip_scope(value)`

返回 IP 地址范围 token：

- 非法、空值或 `0.0.0.0/8`：`ip_unknown_or_zero`；
- loopback：`ip_loopback`；
- `128.55.12.0/24`：`ip_internal_env`；
- RFC1918 私有地址：`ip_private`；
- 其他：`ip_public_or_external`。

示例：

```text
linux_ip_scope("0.0.0.1")    -> "ip_unknown_or_zero"
linux_ip_scope("127.0.0.1")  -> "ip_loopback"
linux_ip_scope("128.55.12.4")-> "ip_internal_env"
linux_ip_scope("10.1.2.3")   -> "ip_private"
linux_ip_scope("8.8.8.8")    -> "ip_public_or_external"
```

### `linux_port_bucket(value)`

把端口分桶：

- 空、非法或 `<=0`：`port_unknown_or_zero`；
- `1..1023`：`port_system`；
- `1024..49151`：`port_registered`；
- `49152..65535`：`port_ephemeral`；
- 大于 `65535`：`port_invalid`。

示例：

```text
linux_port_bucket("")      -> "port_unknown_or_zero"
linux_port_bucket(22)      -> "port_system"
linux_port_bucket(5432)    -> "port_registered"
linux_port_bucket(55000)   -> "port_ephemeral"
linux_port_bucket(70000)   -> "port_invalid"
```

### `linux_exact_ip_token(value)`

返回精确 IP residual token，把点号替换成下划线。非法 IP 返回 `unknown_ip`。

示例：

```text
linux_exact_ip_token("128.55.12.4") -> "128_55_12_4"
linux_exact_ip_token("bad")         -> "unknown_ip"
```

### `linux_netflow_nll_role(dst_addr, dst_port)`

返回 netflow 粗粒度 role，格式是：

```text
net|<ip_scope>|<port_bucket>
```

示例：

```text
linux_netflow_nll_role("128.55.12.4", 443)
  -> "net|ip_internal_env|port_system"
```

矩阵示例：假设 IP scope 有 5 类，port bucket 有 5 类。可以把二者拼接成一个 10 维 multi-hot：

```text
scope order = [unknown, loopback, internal_env, private, public]
port order  = [unknown, system, registered, ephemeral, invalid]

ip_internal_env -> [0, 0, 1, 0, 0]
port_system     -> [0, 1, 0, 0, 0]

x_net_role = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
```

如果下游使用联合类别，也可以把 `net|ip_internal_env|port_system` 当作单个 class id。

### `linux_endpoint_detail_token(value, prefix="")`

返回端点 detail token。合法且非 `0.0.0.0/8` 的 IPv4 会被转成下划线形式；`local` 或 `localhost` 转成 `local`；其他情况返回空字符串。`prefix` 用于标记方向，例如 `src_`。

示例：

```text
linux_endpoint_detail_token("128.55.12.4")
  -> "128_55_12_4"

linux_endpoint_detail_token("10.0.0.2", prefix="src_")
  -> "src_10_0_0_2"

linux_endpoint_detail_token("localhost")
  -> "local"
```

### `linux_netflow_natural_tokens(dst_addr, src_addr="")`

返回 netflow residual tokens：

- 如果 dst IP 具体可用，返回 `("netflow", ip_scope, exact_dst_ip)`；
- 如果 dst IP 不可用，但 src IP/detail 可用，返回 `("netflow", src_detail)`；
- 都不可用时返回 `("netflow",)`。

示例：

```text
linux_netflow_natural_tokens("128.55.12.4", "10.0.0.2")
  -> ("netflow", "ip_internal_env", "128_55_12_4")

linux_netflow_natural_tokens("0.0.0.0", "10.0.0.2")
  -> ("netflow", "src_10_0_0_2")
```

### `linux_netflow_detail_or_fixed_natural_tokens(dst_addr, src_addr="")`

当前实现直接调用 `linux_netflow_natural_tokens`。这个函数保留了一个语义入口：当存在具体远端 IP 时返回 detail tokens，否则退回固定/简化 token 策略。

示例：

```text
linux_netflow_detail_or_fixed_natural_tokens("128.55.12.4", "10.0.0.2")
  -> ("netflow", "ip_internal_env", "128_55_12_4")
```

### `linux_netflow_fixed_nll_role()`

返回固定 netflow role：

```text
linux_netflow_fixed_nll_role() -> "net|linux|netflow"
```

它适合不希望按 IP scope 和 port bucket 拆分 netflow 粗类别的配置。

### `linux_netflow_fixed_natural_tokens()`

返回固定 residual tokens：

```text
linux_netflow_fixed_natural_tokens() -> ("netflow",)
```

## 7. 端到端示例

### 进程读取文件事件

输入：

```text
process path = /bin/bash
command      = bash -c /tmp/vugefal
file path    = /proc/123/task/456/status
```

语义输出：

```text
linux_process_nll_role(...)
  = process|linux|shell

linux_process_natural_tokens(...)
  = ("process", "shell", "vugefal")

linux_file_nll_role(...)
  = file|linux|proc_file

linux_file_natural_tokens(...)
  = ("file", "proc_file", "task_status")
```

下游矩阵示意：

```text
process token ids = [0, 1, 2]
file token ids    = [3, 4, 5]

E =
[[1.0, 0.0],
 [0.0, 1.0],
 [0.5, 0.5],
 [0.8, 0.1],
 [0.1, 0.8],
 [0.2, 0.2]]

x_process = mean(E[[0, 1, 2]], axis=0) = [0.5, 0.5]
x_file    = mean(E[[3, 4, 5]], axis=0) = [0.3667, 0.3667]
```

如果事件动作 embedding 为：

```text
x_action = [0.2, 0.9]
```

一个简单事件上下文可以拼接为：

```text
X_event = [x_process, x_action, x_file]
        = [0.5, 0.5, 0.2, 0.9, 0.3667, 0.3667]
```

### 网络事件

输入：

```text
dst_addr = 128.55.12.4
dst_port = 443
src_addr = 10.0.0.2
```

语义输出：

```text
linux_netflow_nll_role(dst_addr, dst_port)
  = net|ip_internal_env|port_system

linux_netflow_natural_tokens(dst_addr, src_addr)
  = ("netflow", "ip_internal_env", "128_55_12_4")
```

若 token embedding 为：

```text
netflow         -> [0.3, 0.0, 0.1]
ip_internal_env -> [0.0, 0.7, 0.1]
128_55_12_4     -> [0.4, 0.2, 0.8]
```

则：

```text
x_net = ([0.3, 0.0, 0.1] + [0.0, 0.7, 0.1] + [0.4, 0.2, 0.8]) / 3
      = [0.2333, 0.3000, 0.3333]
```

## 8. 使用边界

这些规则只使用运行时可见字段，不依赖测试标签、攻击窗口、恶意实体列表或测试集排序。它们适合在线推理前生成稳定语义表示。若修改本文件的分类规则，应同步考虑已经生成的 Word2Vec、Phase3E cache、Phase3G checkpoint 是否与新 token 规则兼容；否则会影响可复现性。
