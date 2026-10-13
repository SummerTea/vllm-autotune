# program.md — 调优循环细则

## 目标

在 SLO 约束下最大化 goodput；每次实验只改 `config.yaml`，其余文件只读。满足任一退出条件即进 Phase 4，不要无限循环。

- **远端约束**：所有远端操作只写 `target.yaml` 确认的 `work_dir`；默认不 pull 镜像、不安装环境。

## 指标定义

- **goodput** = 满足 SLO 的成功请求吞吐，来自 `vllm bench serve --goodput ttft:<slo.ttft_p99_ms> tpot:<slo.tpot_p99_ms>` 输出的 `request_goodput`（单位 **req/s**，只统计同时满足全部 SLO 的成功请求）。工具细节见 `docs/vllm-facts.md`。
- SLO 取值：`target.yaml` 的 `metrics.slo`；缺省用 `harness.yaml` 的 `slo` 占位。
- workload 与参考并发点固定：`target.yaml` 的 `workload`（**逐键覆盖**：target 未填的键回退 `harness.yaml` 默认）；缺省整块用 `harness.yaml` 的 `workload` 与 `bench.ref_concurrency`。
- **参考并发点映射**：`vllm bench serve --request-rate inf --max-concurrency <ref_concurrency>`；覆盖路径：`target.yaml` 的 `workload.ref_concurrency` 覆盖 `harness.yaml` 的 `bench.ref_concurrency`。
- 辅助列 `output_tok_s`（输出 token 吞吐，tok/s）仅用于诊断，不参与 keep/discard 判定。

## 决策规则

- **keep**：goodput 相比当前最佳提升 > `bench.noise_band_pct`（默认 3%），且 SLO 达标 → 更新当前最佳。
- **discard**：提升 ≤ 噪声带（含变差）→ 回退该配置。
- **crash**：启动失败 / OOM / 超时（`startup_timeout_s` / `trial_timeout_s`）→ 记录 crash，按崩溃处理（见「崩溃处理」）。

## git 约定

- 每次 trial：改 `config.yaml` → `git commit`（同一 run 分支，线性历史）→ `bash run.sh` 启动（秒回）→ 轮询 `bash run.sh --poll` 至退出 0（done）或 1（crash）；3 = 仍在运行 → 记录。
- **不回退提交**：discard 的配置仍留在 git 历史里可复盘；下一个 trial 前只需把工作区恢复到当前最佳：`git checkout <best_commit> -- config.yaml`。
- keep 即更新 `results.tsv` 中的 `best` 标记（记录其 commit）。
- `results.tsv` 与 `run.log` 保持 untracked，永不提交。

## results.tsv 列定义

tab 分隔，每行一个实验：

```
commit  goodput  output_tok_s  ttft_p99_ms  tpot_p99_ms  peak_vram_gb  dur_s  status  description
```

- `commit`：本次 config.yaml 的 git commit（短 hash）。
- `goodput`：SLO 约束下的成功请求吞吐（**req/s**，vLLM 的 `request_goodput`），用于 keep/discard 判定。
- `output_tok_s`：纯输出吞吐（tok/s），辅助诊断。
- `ttft_p99_ms` / `tpot_p99_ms`：实测 P99 时延，用于核对 SLO。
- `peak_vram_gb`：bench 期间 `nvidia-smi` 采样的峰值显存（GB）。
- `dur_s`：本次 trial 的墙钟耗时（秒），用于 deadline 感知调度的历史中位数估算。
- `status`：`keep` / `discard` / `crash` / `best`（当前最佳行）。
- `description`：本次改了什么（自由文本，便于复盘）。

## 退出机制（三层 + 调度细节）

### 主退出（硬保证）
- `target.yaml` 的 `budget`：`max_hours` 与 `stop_at` 二选一，同时填写时以 `max_hours` 优先；`max_hours` 为运行满 N 小时，`stop_at` 为启动后下一个该时刻（跨天自动理解，按编排机本地时钟）。

### 平台期
- 连续 `exit.plateau_trials` 次（默认 8 次）无超过噪声带的改进 → 平台期退出；crash 行不推进也不重置平台期连击（只计入崩溃连击）。

### 崩溃连击
- 连续 `exit.crash_streak` 次（默认 3 次）crash → 停下并报告，不自动重试结构性改动。

### deadline 感知调度
- 每轮开头检查：剩余预算是否 ≥ 预估单 trial 耗时；仅当 剩余 < 预估耗时 才停止开新 trial（相等仍可开，超时由「在途宽限」兜底）。预估值取 `results.tsv` 历史 `dur_s` 的中位数；无历史时用 `model_profile.json` 的启动耗时 + 一次 bench 粗估，并以 `startup_timeout_s` 作下限兜底。来不及 → 停止开新 trial，进 Phase 4。

### 在途宽限
- deadline 到时正在跑的 trial 让它跑完，再进 Phase 4。

### 可断点续跑
- 状态在 `results.tsv` + git 分支上；重启后先读它们，从当前最佳处继续，不重跑已完成的实验。
- 已 commit 但 `results.tsv` 无对应行的 trial（半途被杀）→ 重启后重跑一次。

## 搜索空间与探测驱动剪枝

依据 `env.json` / `model_profile.json` 动态剪枝，避免试已知无效的配置：

- **节点内互联**：NVLink / NVSwitch 全互联 → `tensor-parallel-size`（tp）可到 GPU 数；仅 PCIe → tp ≤ 4，优先并行数据（dp）。
- **RDMA**：有 RoCE / IB → NCCL 相关 env、NIXL / PD 分离进入搜索空间；无 → 移出搜索空间（NCCL env 名见 `docs/vllm-facts.md` §6；NIXL / PD 相关 env 需 agent 自行查证后使用）。
- **合法 tp 集合**：由 `num_attention_heads` / `num_key_value_heads` 的约数决定，非法 tp 值不试。
- **系统级参数**：governor / hugepages / 锁频等只探测 + 建议（写进 report.md），不自动修改（v1 不提供自动修改开关）。
- **布尔参数约定**：`true` → 裸 flag（`--enable-x`）；`false` → 省略（保持 vLLM 默认）；要关闭默认开启的开关，写 `no-<flag>: true`（原样透传为 `--no-<flag>`）。
- **运行时与镜像（docker-only）**：`runtime.image` 是合法搜索维度（如 CUDA 12/13 变体镜像）；镜像须在主机**预先准备**（模板不自动拉取，保持主机零新增），先固定镜像再调参。
- **执行主机**：`run.sh` / `bench.py` 仅作用于唯一 `role: head` 的主机；缺失或重复即报错。

## 成本梯度

- 先穷尽廉价的调度 / 内存类旋钮（`max-num-seqs`、`max-num-batched-tokens`、`gpu-memory-utilization`、KV 缓存相关等，flag 名称见 `config.yaml` 示例与 vLLM 文档）。
- 再做结构性改动：tp / 量化 / 投机解码会重新加载权重甚至重新下载，成本高，只在廉价旋钮耗尽后考虑。

## 崩溃处理

- 明显的 typo / 缺依赖：可修一次并重试该配置。
- 构思本身有问题（如参数组合不合法）：记 crash 跳过，不反复重试。

## 收尾

满足任一退出条件（主退出 / 平台期 / 崩溃连击）→ 进 Phase 4，运行 `python report.py` 产出 `report.md`。循环结束后不要再开新 trial。
