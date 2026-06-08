# CS4M 模块说明

本文用中文解释当前 `cs4m/` 代码结构。它面向在新服务器上复现 CADETS_E3、
THEIA_E3，以及维护 CLEARSCOPE 相关语义工具的开发者。

## 1. 总体目标

CS4M 是一个基于 provenance event stream 的在线威胁检测项目。核心约束是：

- 事件按到达顺序或时间顺序评分；
- 评分时只能使用当前和过去的信息；
- 训练、阈值、模型选择、在线告警不能使用测试标签或攻击窗口；
- ground truth 只能在评分完成后用于评估。

当前主链路可以简化成：

```text
原始数据库事件
  -> 数据集/OS 语义适配
  -> residual tokens / Word2Vec 嵌入
  -> Phase3E 事件索引、节点/动作表
  -> Phase3E 在线状态运行时读写 h_src/h_dst
  -> Phase3G conditional head 产生最终 event score
  -> 在线 event alerts
```

## 2. Phase3E 与 Phase3G 的区别

Phase3E 负责把事件流转换成可复用的表示，并在推理时维护在线状态。它关注“表示”和
“状态运行时”：

- 事件索引 memmap；
- 节点 embedding 表；
- 动作 embedding 表；
- `h_src` / `h_dst` 的读写；
- S4D/EMA 状态更新规则；
- 旧版 `X_context` 和基础低秩 head 已移动到 `legacy/compatibility/`。

Phase3G 负责在 Phase3E 产物之上构建更细的 scoring head。它关注“当前事件是否异常”：

- conditional head 根据事件类型、动作、端点类型选择目标；
- compact node embeddings 降低只访问部分节点时的内存；
- validation cache 支持无标签阈值计算。

一个小例子：

```text
事件: process P 读取 file F
Phase3E:
  src_embedding(P) = [0.2, 0.1]
  action_embedding(READ) = [0.0, 0.4]
  dst_embedding(F) = [0.3, 0.2]
  读取旧状态 h_src_old / h_dst_old
  先构造 scoring context，再更新 h_src / h_dst

Phase3G:
  src_repr = mean(src_embedding, h_src_old)
  dst_repr = mean(dst_embedding, h_dst_old)
  x_G = concat(src_repr, dst_repr, src_type, dst_type)
  conditional head 预测 target = f(x_G)
  true target 可能是 mean(src, action, dst)
  score = distance(predicted target, true target)
```

当前 E4/S4D/update_gate=none 最佳默认路径不再生成 146 维 Phase3E `X_context.memmap`，
也不再用 Phase3E 低秩 head 的 `c1/c2/bias` 作为最终事件分数。Phase3E 仍然保留在线状态机：
读取 `h_src_old` / `h_dst_old`、按 context-before-update 顺序打分、再按 S4D/EMA 规则更新状态。
最终分数来自 Phase3G conditional head。

维度例子：

```text
latent_dim = 64
ENTITY_TYPES = [process, file, flow, other]  # 4 维 one-hot

x_G = [src_repr, dst_repr, src_type, dst_type]
dim(x_G) = 64 + 64 + 4 + 4 = 136
```

旧的 146 维 Phase3E `X_context = [h_src, h_dst, action_one_hot, src_type, dst_type]`
和 Phase3E 低秩 head 只保留在 `legacy/compatibility/` 中。当前 active CLI 不再暴露
Phase3E precompute 或 `--x_context_*` 参数，因此它们不是当前 E4 最佳默认路径。
E5/update_gate=quantile 可能仍依赖旧校准或 checkpoint 字段，需要单独验证。

## 3. `cs4m/config/`

### `config.py`

负责项目根目录、数据集默认配置、运行时 YAML 读取等。CADETS、THEIA、
CLEARSCOPE 的 split 和默认参数最终会通过这里进入 runner。

### `provnet_utils.py`

负责 PostgreSQL 连接、时间转换、节点类型常量、日志辅助等。数据库连接参数必须来自：

```text
CLAD_DB_HOST, CLAD_DB_PORT, CLAD_DB_USER, CLAD_DB_PASSWORD
```

不要把密码写进代码或文档。

## 4. `cs4m/semantics/`

语义模块把原始 process/file/netflow 字段转成稳定 token 或 role。它们不是重复代码，
而是数据集与操作系统差异的适配层。

### `cadets_freebsd.py`

CADETS 数据集来自 FreeBSD 环境。FreeBSD 的路径、进程名和事件类型与 Linux/Android 不同，
所以这里保留独立规则。

例子：

```text
/usr/local/bin/python -> usr local bin python
/etc/passwd           -> etc passwd
```

### `theia_linux.py`

THEIA 数据集偏 Linux 语义，包括 Linux 文件路径、进程路径、netflow 端口/地址 token。

例子：

```text
/bin/bash -c wget http://x -> bin bash wget http
10.0.0.2:443              -> netflow scope port
```

### `clearscope_android.py`

CLEARSCOPE 来自 Android 环境。Android 包名、组件名、设备路径有自己的结构。

例子：

```text
com.example.app/.MainActivity -> com example app mainactivity
/dev/ion                     -> dev ion
```

### `semantic_router.py`

统一语义调度器。它根据数据集配置选择 FreeBSD、Linux、Android 适配器，并把 process、
file、netflow 的语义处理集中起来。因此它不是单纯的 “process semantics”。

示例：

```python
profile = resolve_dataset_profile("THEIA_E3")
result = normalize_process_semantics(command="/bin/bash -c curl x", profile=profile)
```

### `residual_tokens.py`

把事件文本转换为 residual semantic tokens。它服务当前 SSPM 主链路，但不再在 active
代码里做 hash sketch 或 Doc2Vec。当前 CADETS/THEIA 最佳链路使用 residual Word2Vec：
token 先由数据集/OS 语义规则产生，再由预训练 Word2Vec 模型池化成向量。

例子：

```text
text = "/bin/bash -c curl"
tokens = ["bin", "bash", "-c", "curl"]
Word2Vec:
  bash -> [0.2, 0.1]
  curl -> [0.4, 0.0]
mean -> [0.3, 0.05]
```

## 5. `cs4m/embeddings/`

### `residual.py`

负责 residual token 的 embedding 构建和加载。主链路通常使用预训练 Word2Vec 模型，把
token 序列池化成固定维度向量。

例子：

```text
token vectors:
  bash = [0.2, 0.1]
  curl = [0.4, 0.0]
mean = [0.3, 0.05]
L2 normalize -> residual embedding
```

CADETS/THEIA/CLEARSCOPE 都可以使用该模块，但 token 由各自 OS adapter 产生。

## 6. `cs4m/phase3e/`

### `event_index.py`

定义事件索引 memmap 的 dtype、指纹和打开方式。它保存事件 id、时间戳、src/dst node、
action id、type id 等。Phase3E 和 Phase3G 都依赖它保证缓存与原始事件一致。

### `word2vec_adapter.py`

封装 Word2Vec token adapter 和 fingerprint。它确保当前语义模式和 Word2Vec 元数据匹配，
避免用 THEIA 语义去读 ClearScope 模型这类错误。

### `node_action_tables.py`

构建节点 embedding 表和动作 embedding 表。

矩阵例子：

```text
node table:
  node_id 0 -> [0.1, 0.2]
  node_id 1 -> [0.3, 0.4]

action table:
  READ  -> [0.0, 0.5]
  WRITE -> [0.6, 0.1]
```

一个事件 `(node 0, READ, node 1)` 的 action-semantic target 可以由这三部分组合。

### `legacy/compatibility/phase3e_context_memmap.py`

构建旧版 Phase3E `X_context`。这是历史 Phase3E 低秩 head 的输入矩阵，每一行对应一个事件的
146 维上下文。

当前 E4 最佳默认路径不再生成该 memmap；active CLI 不再提供旧 X_context 参数。
如果旧 checkpoint/cache 依赖 `action_one_hot` 或 `raw_orthrus10`，它属于历史兼容风险，
而不是 Phase3G final scoring 的核心输入。

### `legacy/compatibility/phase3e_head_training.py`

Torch 低秩 head 训练辅助。它服务旧 Phase3E `X_context`/`c1/c2/bias` 路径；当前 E4 最佳默认
路径不训练 Phase3E 低秩 head，也不把它作为最终评分头。

## 7. `cs4m/phase3g/`

### `compact_node_embeddings.py`

大数据集可能有很多节点，但某一次验证或推理只会访问其中一部分。compact node embedding
把“用到的节点”重排成紧凑数组，减少内存和随机访问成本。

例子：

```text
原始 node ids: [10, 99, 5000]
本次用到: [99, 10]

compact map:
  99 -> 0
  10 -> 1

原 embedding[99] 变成 compact_embedding[0]
原 embedding[10] 变成 compact_embedding[1]
```

### `conditional_head.py`

当前 CADETS/THEIA 最佳链路使用 conditional semantic scoring。它根据 target case 选择
预测目标：

- 普通事件：预测 event/action semantic target；
- 两端都冷启动等特殊情况：使用对应 fallback target；
- CADETS v2 支持 dual head；
- THEIA 当前最佳链使用 shared lowrank head。

小例子：

```text
x_G = [src_repr, dst_repr, src_type_one_hot, dst_type_one_hot]
W1 = [[1, 0], [0, 1], [1, 1], [0, 1]]
W2 = [[0.5, 0.0], [0.0, 0.5]]

hidden = x_G @ W1
pred = hidden @ W2
distance(pred, target) -> event score
```

阈值来自 validation score 的无标签统计，不使用 test labels。

## 8. `cs4m/models/`

### `cs4m_lowrank.py`

当前 SSPM 状态运行时和兼容低秩模型。它维护流式 node state。对于当前 E4 最佳路径，
`cs4m_lowrank.py` 的关键职责是在线读取/更新 `h_src` 和 `h_dst`；最终事件分数由
`phase3g/conditional_head.py` 计算，而不是由 Phase3E 的 `c1/c2/bias` 计算。

低秩结构：

```text
context_dim = 4
rank = 2
target_dim = 3

X  shape: [batch, 4]
W1 shape: [4, 2]
W2 shape: [2, 3]

prediction = X @ W1 @ W2
```

这样参数量从 `4 * 3 = 12` 变为 `4 * 2 + 2 * 3 = 14`。这个低秩预测头现在主要保留给
Phase3E 兼容路径；E4 最佳默认路径使用同一个模型对象的状态机，但用 Phase3G conditional
head 的低秩矩阵做最终预测。

## 9. `cs4m/scoring/`

### `target_builder.py`

构建不同 scoring target。例如：

```text
event_action_semantic = mean(src_embedding, action_embedding, dst_embedding)
node_pair_no_action   = mean(src_embedding, dst_embedding)
```

当前最佳 CADETS/THEIA 链路使用 event/action semantic 相关目标；experimental no-action head
已经放到 `legacy/experiments/`。

### `calibration.py`

残差校准和 update-gate calibration。它只应该使用训练/验证阶段允许的信息，不能用测试标签。

### `simple_gates.py`

固定动作 gate、旧版 raw ORTHRUS10 action context 兼容工具，以及简单的更新权重逻辑。
当前 E4 Phase3G final scoring 不把 `action_one_hot` 作为 `x_G` 的组成部分；它只可能出现在
旧 Phase3E `X_context`/低秩 head 兼容路径里。

## 10. `cs4m/state/`

### `state_models.py`

实现在线状态动力学，例如 EMA、real-diag、S4D complex node。状态表示“节点在过去事件后的
记忆”。

### `state_memory.py`

实现 bounded/probationary LRU state memory。它防止超大图把内存无限撑大。

### `online_state_merging.py`

实现在线 state merging。相似状态可以合并，减少状态数量。合并必须在线进行，不能使用未来信息。

## 11. `cs4m/utils/`

### `common.py`

共享 hash、类型映射、信息流方向、稳健统计等。

### `profiling.py`

RSS 和运行时 profiling 辅助。

### `residual_embed_cache.py`

SQLite residual embedding cache。用于避免重复计算相同 token 序列的 embedding。

## 12. 数据集适配器为什么不是重复代码

同一个概念在不同系统里表现不同：

| 数据集 | OS/环境 | 例子 | 为什么需要独立规则 |
|---|---|---|---|
| CADETS | FreeBSD | `/usr/local/bin` | FreeBSD 路径和审计事件格式不同 |
| THEIA | Linux | `/proc`, `/bin/bash`, netflow | Linux 进程和网络字段不同 |
| CLEARSCOPE | Android | package/activity, `/dev/ion` | Android 包名和设备路径不同 |

统一方法要求“流式、无标签泄漏、可解释、可复现”的 pipeline 一致，不要求所有数据集用完全相同
的字符串规则。

## 13. CADETS/THEIA/CLEARSCOPE 使用位置

| 模块组 | CADETS_E3 | THEIA_E3 | CLEARSCOPE |
|---|---|---|---|
| `semantics/*` | `cadets_freebsd.py` | `theia_linux.py` | `clearscope_android.py` |
| `embeddings/residual.py` | 使用 | 使用 | 训练/烟测使用 |
| `phase3e/*` | E4 状态运行时缓存 | E4 状态运行时缓存 | 可复用 |
| `models/cs4m_lowrank.py` | E4 在线状态运行时 | E4 在线状态运行时 | 可复用 |
| `phase3g/conditional_head.py` | dual-head v2/Q09995 | shared lowrank v1 | 可复用 |
| `scoring/target_builder.py` | event-action target | event-action target | 可复用 |

## 14. 历史基线位置

CSSM/旧 low-rank baseline 已移到：

```text
legacy/baselines/
legacy/tools/
```

它们可用于对照或历史诊断，但不是当前 CADETS/THEIA 最佳复现链的 active modules。

此外，历史 action-predict head、hash-sketch residual embedder、Doc2Vec residual embedder 已移动到：

```text
legacy/experiments/phase3g_action_head.py
legacy/experiments/residual_hash_doc2vec.py
legacy/diagnostics/export_residual_semantic_trace.py
```

当前 CADETS/THEIA 最佳链路集中在 `cs4m/phase3g/conditional_head.py`，不再把这些历史实现
作为 active main-chain 模块。

## 15. `scripts/` 当前组织

active runner 现在调用 `python3 -m scripts.pipeline.entrypoints.conditional_e4`。
`scripts/tools/causal_semantics_slim.py` 已移到 `legacy/tools/`，只作为历史入口保留。
active 实现已经按真实职责拆到 `scripts/pipeline/` 子包：

```text
scripts/pipeline/entrypoints/conditional_e4.py  # active shell runner 调用的 Python 入口
scripts/pipeline/entrypoints/arguments.py       # 参数、配置校验、顶层调度
scripts/pipeline/config/runtime_config.py       # 共享常量、SlimConfig、轻量运行时类
scripts/pipeline/checks/preflight.py            # split/DB stream 与 residual preflight
scripts/pipeline/state/online_state_runtime.py  # Phase3E 状态运行时与 h_src/h_dst 更新
scripts/pipeline/features/conditional_context.py# 136-dim x_G 与 conditional score stream
scripts/pipeline/features/semantic_features.py  # 数据集语义文本、residual text、校准
scripts/pipeline/conditional/train.py           # conditional head / memmap 构建入口
scripts/pipeline/conditional/infer.py           # Phase3G conditional 推理入口
scripts/pipeline/io/conditional_cache.py        # memmap/cache 路径与元数据
scripts/pipeline/io/event_artifacts.py          # active Phase3E artifact 与旧路径保护
scripts/pipeline/io/db_stream.py                # 数据库事件流 helper
scripts/pipeline/io/cache_payloads.py           # cache payload 读写
scripts/pipeline/outputs/alert_output.py        # online_event_alerts 与评估输出
scripts/pipeline/outputs/conditional_reports.py # group/coverage/RSS 报告
scripts/pipeline/outputs/evaluation.py          # 评估辅助，不参与在线阈值选择
scripts/pipeline/outputs/metrics_summary.py     # 简洁运行摘要
scripts/pipeline/compatibility/                 # 临时 namespace bridge，仅供旧 import 调用
```

更详细的中文说明、每个 active Python 文件和 shell runner 的用途、以及小型数值例子见：
`docs/SCRIPTS_PIPELINE_EXPLANATION_ZH.md`。

Active `scripts/run/` 只保留当前 E4 主链路相关 runner：

```text
run_cadets_e3_e4_conditional_v2_q09995.sh
run_cadets_e3_phase3e_node_action_semantic_full.sh
run_phase3g_theia_e3_e4_theia_v1_lazy100k_pair_only_full.sh
build_theia_e3_phase3g_conditional_memmaps.sh
```

旧的 E2/E4/E5 多消融 runner、训练 runner、smoke、summarizer 和诊断脚本已经移动到
`legacy/runners/`、`legacy/tools/` 或 `legacy/diagnostics/`。这表示它们仍可作为历史记录
检查，但不再混在 active mainline 脚本里。
