# scripts/ 与 scripts/pipeline/ 活跃路径说明

本文说明当前活跃的 `scripts/` Python 文件和 shell runner。这里的“活跃”只指
CADETS_E3 / THEIA_E3 当前 E4 conditional best path：

- Phase3E online state runtime。
- `h_src` / `h_dst` 读出后再更新。
- S4D/EMA 状态更新。
- context-before-update。
- Phase3G `conditional_action_semantic` scoring。
- 136 维 `x_G = [src_repr, dst_repr, src_type, dst_type]`。
- CADETS_E3 E4 conditional runner。
- THEIA_E3 E4 / `theia_v1` conditional runner。
- THEIA conditional memmap builder。
- 必需的 preflight、cache、alert output、eval helper。

旧 E2/E5 消融、旧 Phase3E `X_context` 生成、Phase3E low-rank head final scoring、
旧 `semantic_residual` scoring、Doc2Vec/hash-sketch residual、历史 action-predict head、
smoke/summarize/export/diagnostic 工具属于 `legacy/`，不属于本文的 active mainline。

## 1. active shell runner

### `scripts/run/run_cadets_e3_e4_conditional_v2_q09995.sh`

CADETS_E3 best path 的薄 wrapper。它固定 E4 NONE 配置、Q09995 threshold family 和
conditional head 参数，然后调用 base runner。它不直接做训练或推理逻辑，只负责把环境变量
收束为当前 CADETS_E3 E4 best-path 参数。

在 dry-run 中，它应展开到：

```text
python3 -m scripts.pipeline.entrypoints.conditional_e4 ...
--dataset CADETS_E3
--sspm_score_head conditional_action_semantic
--sspm_state_model s4d_complex_node
--event_threshold_quantile 0.9995
```

### `scripts/run/run_cadets_e3_phase3e_node_action_semantic_full.sh`

CADETS/THEIA 共用的 active base runner。它根据 `DATASET`、`RUN_ONLY`、`STAGE` 等环境变量
拼接 Python CLI 参数，并调用：

```text
python3 -m scripts.pipeline.entrypoints.conditional_e4
```

当前 active runner 不再调用旧 `scripts/tools/causal_semantics_slim.py` wrapper。

### `scripts/run/run_phase3g_theia_e3_e4_theia_v1_lazy100k_pair_only_full.sh`

THEIA_E3 best path wrapper。它使用 E4 conditional head、`theia_v1` alert policy、lazy100k
pair-only 配置，并依赖 THEIA conditional memmap/cache。它仍走同一个 Python entrypoint：

```text
python3 -m scripts.pipeline.entrypoints.conditional_e4
```

### `scripts/run/build_theia_e3_phase3g_conditional_memmaps.sh`

THEIA_E3 conditional memmap builder。它只构建 conditional train/validation memmap，不做完整
训练、完整推理或评估。输出目标是供 THEIA E4 runner 读取的 conditional cache，例如：

```text
X_conditional_s4d_complex_node.memmap
Y_conditional_s4d_complex_node.memmap
target_case_s4d_complex_node.memmap
```

## 2. `scripts/pipeline/` 子包总览

`scripts/pipeline/` 已按职责拆成子包，根目录只保留 `__init__.py`：

```text
scripts/pipeline/
  entrypoints/    # Python CLI 入口与参数调度
  config/         # SlimConfig、常量、轻量运行时类
  checks/         # preflight、split、DB stream 前置检查
  io/             # DB、memmap、cache、artifact I/O
  state/          # Phase3E online state runtime
  features/       # 语义特征与 conditional context
  conditional/    # Phase3G conditional train/infer
  outputs/        # alert、report、metrics、eval 输出
  compatibility/  # 临时旧 import bridge，不是 active runner 入口
```

## 3. entrypoints

### `scripts/pipeline/entrypoints/conditional_e4.py`

active shell runner 调用的唯一 Python module entrypoint。它导出 `main`、`parse_args`、
`config_from_args` 和 `SlimConfig`，但真实参数解析和调度在 `entrypoints/arguments.py`。

示例：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m scripts.pipeline.entrypoints.conditional_e4 --help
```

### `scripts/pipeline/entrypoints/arguments.py`

负责 CLI 参数、`SlimConfig` 构造、active-only 配置校验和顶层分派。

active 边界在这里强制执行：

- `--sspm_score_head` 只能走 `conditional_action_semantic`。
- 旧 Phase3E precompute-only 分支会被拒绝。
- 旧 Phase3E low-rank `train_and_save` 分支会被拒绝。
- E4 conditional train、conditional memmap-only、load-and-infer 会进入对应 active module。

例子：dry-run 展开的参数含义可以简化为：

```text
--dataset CADETS_E3
--sspm_train_mode load_and_infer
--sspm_score_head conditional_action_semantic
--sspm_state_model s4d_complex_node
```

这表示使用已有 Phase3E state checkpoint 和 Phase3G conditional head 做在线推理。

## 4. config 与 checks

### `scripts/pipeline/config/runtime_config.py`

集中保存 active pipeline 共享对象：

- `SlimConfig` dataclass。
- conditional head threshold 常量。
- cache schema/version 常量。
- adaptive online threshold controller 等轻量运行时类。
- active modules 共享的第三方/项目内 imports。

`SlimConfig` 是 shell runner 参数进入 Python 后的主配置对象。例如：

```text
dataset = "THEIA_E3"
latent_dim = 64
sspm_score_head = "conditional_action_semantic"
sspm_conditional_head_arch = "shared_lowrank_v1"
```

### `scripts/pipeline/checks/preflight.py`

负责 active run 进入主逻辑前的检查和数据流准备：

- 解析 split day。
- 建立 node map。
- 只读 ground-truth index helper，供 post-inference evaluation 使用。
- DB stream 状态发布和 fallback 信息。
- residual embedding / latent dim / semantic config 前置准备。

这里的 DB helper 按时间顺序 stream events。在线推理仍必须先产生 alert，再使用 ground truth 做
后处理评估。

## 5. io

### `scripts/pipeline/io/db_stream.py`

低层 DB event stream helper。它按 day、`timestamp_rec`、`_id` 排序取事件，并生成统一 row：

```text
event_index, timestamp, src_index_id, dst_index_id, action, src_summary, dst_summary
```

例如某天两条事件时间为 10 和 12，则输出顺序固定为 10 在前、12 在后，状态更新也只能按这个
顺序进行。

### `scripts/pipeline/io/event_artifacts.py`

解析 active Phase3E artifact 路径：

- node embedding cache。
- action embedding cache。
- event index memmap。
- Phase3E checkpoint。

它只解析 active node/action/event-index artifact；旧 `X_context` 生成和读取逻辑已移动到
`legacy/compatibility/phase3e_context_memmap.py`。

### `scripts/pipeline/io/conditional_cache.py`

负责 Phase3G conditional memmap/cache 路径和 metadata：

```text
X_conditional_s4d_complex_node.memmap
Y_conditional_s4d_complex_node.memmap
target_case_s4d_complex_node.memmap
conditional_train_meta.json
```

小例子：假设有 3 个 train events，`x_G` 维度为 8，则 `X_conditional` 可理解为：

```text
row0 = [0.1, 0.0, 0.4, 0.2, 1, 0, 0, 1]
row1 = [0.2, 0.1, 0.5, 0.3, 1, 0, 1, 0]
row2 = [0.0, 0.7, 0.1, 0.8, 0, 1, 1, 0]
```

实际 best path 中 latent dim 为 64，类型 one-hot 为 4 维，所以每行是 136 维。

### `scripts/pipeline/io/cache_payloads.py`

读写 active run 的 cache payload、fingerprint 和恢复信息。用途是复现实验输入，不改变 alert
或 threshold 语义。

## 6. state

### `scripts/pipeline/state/online_state_runtime.py`

Phase3E online state runtime。它负责：

- 加载 SSPM/Phase3E checkpoint 状态。
- 根据事件读出 `h_src` 和 `h_dst`。
- 在状态更新前构造 conditional context。
- 执行 S4D/EMA read-update。
- 校验 residual Word2Vec semantic metadata 与数据集语义规则一致。

小型 EMA 例子：

```text
h_old = [0.2, 0.0]
z_event = [0.6, 0.4]
alpha = 0.5
h_new = (1 - alpha) * h_old + alpha * z_event
      = 0.5 * [0.2, 0.0] + 0.5 * [0.6, 0.4]
      = [0.4, 0.2]
```

active best path 的关键点是 context-before-update：如果当前事件是 `A -> B`，Phase3G scoring
先使用更新前的 `h_A_old`、`h_B_old`，然后 Phase3E 才把这条事件写回状态。

## 7. features

### `scripts/pipeline/features/semantic_features.py`

负责把不同数据集的 process/file/netflow/action 信息转成 residual text 和 action family：

- CADETS_E3 使用 FreeBSD 语义。
- THEIA_E3 使用 Linux 语义。
- ClearScope 语义保留在 core semantics，但不属于当前 CADETS/THEIA best validation 范围。

例子：`/usr/bin/python script.py` 这样的 process summary 会被规整成进程/路径相关 token，
再进入 residual Word2Vec adapter。

### `scripts/pipeline/features/conditional_context.py`

构造 Phase3G conditional input `x_G`，并支持 conditional score stream。

真实 best path 维度：

```text
latent_dim = 64
len(entity_types) = 4
x_G = [src_repr, dst_repr, src_type, dst_type]
dim(x_G) = 64 + 64 + 4 + 4 = 136
```

小型数值例子：假设 latent dim 为 2，类型 one-hot 为 2 维：

```text
e_src = [0.2, 0.6]
h_src_old = [0.4, 0.2]
src_repr = mean(e_src, h_src_old) = [0.3, 0.4]

e_dst = [0.8, 0.0]
h_dst_old = [0.2, 0.4]
dst_repr = mean(e_dst, h_dst_old) = [0.5, 0.2]

src_type = [1, 0]
dst_type = [0, 1]
x_G = [0.3, 0.4, 0.5, 0.2, 1, 0, 0, 1]
```

如果某个 endpoint 没有在线 state，则该端会退到 embedding/type 可见信息；active path 不允许整条
Phase3G 退化成只看静态 embedding。

## 8. conditional

### `scripts/pipeline/conditional/train.py`

Phase3G conditional train path 和 THEIA memmap-only builder 所在模块：

- 从 Phase3E artifact 和 event index 生成 conditional train/validation memmap。
- 训练或保存 conditional head checkpoint。
- memmap-only 模式只写 memmap/cache payload，不做 full training/full inference。

小例子：`target_case` 标记 conditional target 类型：

```text
event0: process -> file write, target_case = EVENT_SEMANTIC_TARGET
event1: process -> process spawn, target_case = ACTION_SRC_DST
```

### `scripts/pipeline/conditional/infer.py`

Phase3G conditional load-and-infer path：

- 加载 Phase3E online state checkpoint。
- 加载 conditional head checkpoint。
- 按时间顺序 stream test events。
- 每条事件先构造 `x_G`，再打分，再更新 state。
- 写出 online alert/eval payload 所需的 stream outputs。

conditional head 打分的小例子：

```text
x_G = [0.3, 0.4, 0.5, 0.2]
W = [[1.0, 0.0, 0.5, 0.0],
     [0.0, 1.0, 0.0, 0.5]]
pred = W x_G = [0.55, 0.50]
target = [0.2, 0.6]
distance = ||pred - target||_2 = sqrt(0.35^2 + (-0.10)^2)
```

真实模型使用 conditional head 内部的 low-rank/shared/dual architecture 和 group threshold，
但在线语义仍是：当前事件的 runtime-visible context 进入 head，输出 score，再按当前阈值/分组
策略决定 alert。

## 9. outputs

### `scripts/pipeline/outputs/alert_output.py`

负责 active 输出：

- `online_event_alerts.csv`。
- raw/evaluated alert rows。
- strict/relaxed node alert 支持文件。
- post-stream eval payload。
- memory/profile summary。

ground truth 只在 stream 完成后进入 evaluated output，不参与在线 threshold、ranking 或 alert
生成。

### `scripts/pipeline/outputs/conditional_reports.py`

生成 conditional group threshold、coverage、RSS/backfill 等报告，帮助解释 conditional head
在不同 target/action/type group 下的表现。

### `scripts/pipeline/outputs/evaluation.py`

提供 post-inference evaluation helper，例如不同 dataset 的 target budget/label 解析。它不参与
在线打分或阈值选择。

### `scripts/pipeline/outputs/metrics_summary.py`

打印简洁运行摘要和 profile 信息，帮助 dry-run 或 bounded run 检查配置、artifact、计数、耗时。

## 10. compatibility 边界

`scripts/pipeline/compatibility/` 已从 active mainline 删除。active modules 之间使用显式
provider import，而不是把所有拆分模块重新绑定到一个 monolith 风格的全局命名空间。

旧 import caller 如果仍需要聚合导出，使用 `legacy/compatibility/pipeline_runtime_exports.py`；
旧 namespace bridge 只保留在 `legacy/compatibility/pipeline_namespace_bridge.py`，用于历史工具
compile-check，不属于 CADETS_E3 / THEIA_E3 E4 active runner。

active shell runner 只调用 `scripts.pipeline.entrypoints.conditional_e4`。测试也直接 import
responsibility modules。

## 11. active-vs-legacy 边界

active `scripts/` 只保留当前 CADETS_E3 / THEIA_E3 E4 conditional best path 必需文件。旧 runner、
旧工具、诊断和历史实验位于：

```text
legacy/runners/
legacy/tools/
legacy/diagnostics/
legacy/experiments/
legacy/compatibility/
```

这些文件可用于历史对照，但不应被 active E4 shell runner 或 tests 当作主路径依赖。
