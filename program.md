# program.md — 调优循环细则

## 目标

在 SLO 约束下最大化 goodput；每次实验只改 `config.yaml`，其余文件只读。满足任一退出条件即进 Phase 4，不要无限循环。

- **远端约束**：所有远端操作只写 `target.yaml` 确认的 `work_dir`；默认不 pull 镜像、不安装环境。

## 运行期自主性契约

Phase 0 输入齐全且 Phase 1.5 无阻断项后进入禁问窗口：自主执行并不断逼近目标，运行期不再向人类提问。遭遇本文档未覆盖、且既有规则无法兜底的阻断时，停止循环并把诊断与未决问题写入 report.md，而不是实时求助。每个自主决策（SLO 校准 / workload 降档 / 显存与并行降级 / 共租处置 / 换机换模型）都必须在 results.tsv.description 留痕；SLO 校准由 report.py 在报告中以「运行期 SLO 校准」标注呈现。

## 指标定义

- **goodput** = 满足 SLO 的成功请求吞吐，来自 `vllm bench serve --goodput ttft:<slo.ttft_p99_ms> tpot:<slo.tpot_p99_ms>` 输出的 `request_goodput`（单位 **req/s**，只统计同时满足全部 SLO 的成功请求）。工具细节见 `docs/vllm-facts.md`。
- **σ（goodput_std）** = 本次 trial `bench.repeats`（默认 3）次测量的**样本标准差**（`statistics.stdev`；n<2 时退化为 0.0），由 `bench.py` 写入 `result.json` 的 `stats.goodput_std` 与 `results.tsv` 的 `goodput_std` 列。显著性口径：keep 阈值 = `max(best × noise_band_pct/100, noise_k_sigma × σ)`（统一单位 req/s，见「决策规则」）。
- **非独占主机（共租）**：`bench.py` 在起 serve 前检测 `nvidia-smi --query-compute-apps`；检测到其他 GPU 进程 → `result.json` 记录 `contention`（含 pid 列表）并在 status 告警；`bench.require_idle=true` 时直接判 crash（主机非空闲）。共租会污染吞吐/时延测量，建议尽量独占主机；无 nvidia-smi 的主机（如开发机 Mac）自动跳过检测。
- SLO 取值：`config.yaml` 的 `slo`（运行期校准的有效 SLO）> `target.yaml` 的 `metrics.slo`（参考）> `harness.yaml` 的 `slo`（占位）；逐键覆盖。
- workload 与参考并发点固定：`target.yaml` 的 `workload`（**逐键覆盖**：target 未填的键回退 `harness.yaml` 默认）；缺省整块用 `harness.yaml` 的 `workload` 与 `bench.ref_concurrency`。
- **参考并发点映射**：`vllm bench serve --request-rate inf --max-concurrency <ref_concurrency>`；覆盖路径：`target.yaml` 的 `workload.ref_concurrency` 覆盖 `harness.yaml` 的 `bench.ref_concurrency`。
- 辅助列 `output_tok_s`（输出 token 吞吐，tok/s）仅用于诊断，不参与 keep/discard 判定。

## SLO 可行性与自主放宽

- **参考 SLO**：target.yaml 声明值，仅用于 report.md 对标，运行期不改其语义。
- **有效 SLO**：运行期算 goodput 与 keep/discard 用的 SLO，默认等于参考 SLO（「决策规则」中的 SLO 均指有效 SLO）。
- **落点**：校准经 `config.yaml` 的 `slo:` 写入（`ttft_p99_ms` / `tpot_p99_ms`，优先级高于 `target.metrics.slo`）；`target.yaml` 的 SLO 只作参考、不改其语义。bench.py / report.py 均按此优先级解析。
- **触发**：任一 trial goodput == 0（无请求同时满足参考 SLO）→ 触发一次性校准。
- **口径**：有效 SLO = 该次实测 p99 × 1.10，向上取整到整数 ms（ttft / tpot 各自独立）；该次实测行重记为基线。
- **上限**：校准至多一次，之后锁定，禁止逐轮放宽；放宽后仍恒 0 → 按「SLO 不可达退出」提前收尾。`config.yaml` 的 `slo:` **仅用于该一次性校准**，不得用于其它目的逐轮写入。
- **留痕**：results.tsv.description 写 `slo-calibrated: ttft<ref>→<eff> tpot<ref>→<eff>`；report.py 在报告中出「运行期 SLO 校准」标注。

## 决策规则

本节 SLO 均指有效 SLO（见「SLO 可行性与自主放宽」）。

- **keep**：`(cand − best) > max(best × noise_band_pct/100, noise_k_sigma × σ)`——`best` = 当前最佳 goodput（req/s）、`σ` = 本次 trial 的 `goodput_std`（req/s）、`noise_band_pct` 为百分比（如 3.0 → 3%，乘 `best` 得绝对 req/s），两侧同为单位 req/s——且 SLO 达标 → 更新当前最佳。σ 不可用（n<2）时退化为仅 `best × noise_band_pct/100`。
- **discard**：提升 ≤ 噪声阈值（含变差）→ 回退该配置。
- **crash**：启动失败 / OOM / 超时（`startup_timeout_s` / `trial_timeout_s`）→ 记录 crash，按崩溃处理（见「崩溃处理」）。

## git 约定

- 每次 trial：改 `config.yaml` → `git commit`（同一 run 分支，线性历史）→ `bash run.sh` 启动（秒回）→ 轮询 `bash run.sh --poll` 至退出 0（done）或 1（crash）；3 = 仍在运行 → 记录。
- **不回退提交**：discard 的配置仍留在 git 历史里可复盘；下一个 trial 前只需把工作区恢复到当前最佳：`git checkout <best_commit> -- config.yaml`。
- keep 即更新 `results.tsv` 中的 `best` 标记（记录其 commit）。
- `results.tsv` 与 `run.log` 保持 untracked，永不提交。

## results.tsv 列定义

tab 分隔，每行一个实验：

```
commit  goodput  goodput_std  output_tok_s  ttft_p99_ms  tpot_p99_ms  peak_vram_gb  dur_s  status  description
```

- `commit`：本次 config.yaml 的 git commit（短 hash）。
- `goodput`：SLO 约束下的成功请求吞吐（**req/s**，vLLM 的 `request_goodput`），用于 keep/discard 判定。
- `goodput_std`：本次 trial repeats 的 goodput 样本标准差（σ；n<2 时为 0），显著性判定的噪声基数（`noise_k_sigma`×σ，与 `best × noise_band_pct/100` 取较大者）。
- `output_tok_s`：纯输出吞吐（tok/s），辅助诊断。
- `ttft_p99_ms` / `tpot_p99_ms`：实测 P99 时延，用于核对 SLO。
- `peak_vram_gb`：bench 期间 `nvidia-smi` 采样的峰值显存（GB）。
- `dur_s`：本次 trial 的墙钟耗时（秒），用于 deadline 感知调度的历史中位数估算。
- `status`：`keep` / `discard` / `crash` / `best`（当前最佳行）/ `confirm`（收尾对 best 的回归确认行；**不参与** best 选择，只作核对）。
- `description`：本次改了什么（自由文本，便于复盘）。

## 退出机制（四层 + 调度细节）

### 主退出（硬保证）
- `target.yaml` 的 `budget`：`max_hours` 与 `stop_at` 二选一，同时填写时以 `max_hours` 优先；`max_hours` 为运行满 N 小时，`stop_at` 为启动后下一个该时刻（跨天自动理解，按编排机本地时钟）。

### 平台期
- 连续 `exit.plateau_trials` 次（默认 8 次）无超过噪声带的改进 → 平台期退出；crash 行不推进也不重置平台期连击（只计入崩溃连击）。

### 崩溃连击
- 连续 `exit.crash_streak` 次（默认 3 次）crash → 停下并报告，不自动重试结构性改动。

### SLO 不可达退出
- 基线 goodput==0 且 SLO 校准后连续 2 次仍 goodput==0 → 判定 SLO 对本机/模型不可达，停止并进 Phase 4；report.md 标注不可达并给出「建议放宽至实测 p99」的结论。

### deadline 感知调度
- 每轮开头检查：剩余预算是否 ≥ 预估单 trial 耗时；仅当 剩余 < 预估耗时 才停止开新 trial（相等仍可开，超时由「在途宽限」兜底）。预估值取 `results.tsv` 历史 `dur_s` 的中位数；无历史时用 `model_profile.json` 的启动耗时 + 一次 bench 粗估，并以 `startup_timeout_s` 作下限兜底。来不及 → 停止开新 trial，进 Phase 4。
- 当预估单 trial 耗时 > 剩余预算约一半时，先按「workload 降档」缩小 num_prompts 再开新 trial；降档开销计入预估。

### 在途宽限
- deadline 到时正在跑的 trial 让它跑完，再进 Phase 4。

### 循环调度与断点续跑
- 每轮固定顺序：读任务板视图 → deadline 检查 → 生成/选择候选 →（可选）分析/评估子 agent → 写 config.yaml → commit → run.sh → poll → 记录 → keep/discard → 更新视图。
- 门禁：见 AGENTS.md「循环调度与上下文预算」的门表；无子 agent 时验证门降级为主循环二次独立计算。
- 状态在 results.tsv + git 分支上；重启后先重建任务板视图（results.tsv 末行 + git HEAD + 预算剩余 + 在途状态），从当前最佳处继续，不重跑已完成的实验。
- 已 commit 但 results.tsv 无对应行的 trial（半途被杀）→ 重启后重跑一次。

## 搜索空间与探测驱动剪枝

依据 `env.json` / `model_profile.json` 动态剪枝，避免试已知无效的配置：

- **节点内互联**：NVLink / NVSwitch 全互联 → `tensor-parallel-size`（tp）可到 GPU 数；仅 PCIe → tp ≤ 4，优先并行数据（dp）。
- **tp 优先序**：tp ≤ 4（PCIe）是**性能建议**；若单节点装不下模型（依据 `model_profile.json` 的权重量估算）则必须跨节点——此时以"可装下"为先，跨 NUMA 损耗须实测。
- **NUMA 亲和**：tp 组应限定在**同一 NUMA 节点内**的 GPU（依据 `env.json` 的 `numa_groups`，如 8 卡两节点 → tp=4 选 `[0,1,2,3]` 或 `[4,5,6,7]`）；tp=8 跨两节点（走 SYS/跨 NUMA）仅在同节点方案不可行时才试，跨 NUMA 损耗需实测确认。GPU 选择经 `config.yaml` 的 `serve.tensor-parallel-size` + `env.CUDA_VISIBLE_DEVICES`（如 `"0,1,2,3"`，限定本 trial 容器可见的 GPU，随 `-e` 注入、零冲突）——`tp` 按可见 GPU 数设置。**主机级**绑定（`numactl --cpunodebind/--membind`）→ **只建议**（写进 `report.md`），不自动；**容器级** `--cpuset-cpus/--cpuset-mems`（经 `runtime.extra_docker_args`）→ **agent 可 opt-in**（仅影响本 trial 容器，零破坏，不改变主机）。
- **RDMA**：有 RoCE / IB → NCCL 相关 env、NIXL / PD 分离进入搜索空间；无 → 移出搜索空间（NCCL env 名见 `docs/vllm-facts.md` §6；NIXL / PD 相关 env 需 agent 自行查证后使用）。
- **合法 tp 集合**：由 `num_attention_heads` / `num_key_value_heads` 的约数决定，非法 tp 值不试；例外：`num_key_value_heads`≤1（MLA / MQA）时 vLLM 在 tp>kv_heads 时复制 KV heads，合法 tp 只受 `num_attention_heads` 约数约束。
- **系统级参数**：governor / hugepages / 锁频 / **主机级 NUMA 亲和绑定**等只探测 + 建议（写进 report.md），不自动修改（v1 不提供自动修改开关）。
- **布尔参数约定**：`true` → 裸 flag（`--enable-x`）；`false` → 省略（保持 vLLM 默认）；要关闭默认开启的开关，写 `no-<flag>: true`（原样透传为 `--no-<flag>`）。
- **前缀缓存可测性**：`workload.prefix_hit_pct`（0-100，仅 random 数据集）让 `enable-prefix-caching` 成为可测维度——random 数据默认无共享前缀，prefix caching 测不出收益；`>0` 时 bench 按 `prefix_len = isl×pct/100` 拆共享前缀、`adjusted = isl−prefix_len`（总 prompt ≈ isl）。repeats 之间会 `POST /reset_prefix_cache` 保证每次测量是冷缓存独立（首 repeat 起服即冷缓存）。
- **运行时与镜像（docker-only）**：`runtime.image` 是合法搜索维度（如 CUDA 12/13 变体镜像）；镜像须在主机**预先准备**（模板不自动拉取，保持主机零新增），先固定镜像再调参。
- **执行主机**：`run.sh` / `bench.py` 仅作用于唯一 `role: head` 的主机；缺失或重复即报错。

## 成本梯度

- 廉价起服/内存适配预检用 `run.sh --detect-gmu` → `bash run.sh --poll` 循环：远端 `bench.py --detect-gmu` 用 `--load-format dummy` 从高到低探测安全 `gpu-memory-utilization`，结果回传本机 `gmu_probe.json`（`gmu_suggested` 建议用进后续 config）。dummy 生成**同 shape/dtype** 的张量，显存占用与稠密模型调度有代表性；**MoE 模型不适用**（router 依赖门控权重，dummy 下各 expert 负载/显存/吞吐失真）——dummy 仅用于起服/内存适配探测，**吞吐 keep/discard 实验一律用真实权重**。
- 先穷尽廉价的调度 / 内存类旋钮（`max-num-seqs`、`max-num-batched-tokens`、`gpu-memory-utilization`、KV 缓存相关等，flag 名称见 `config.yaml` 示例与 vLLM 文档）。
- 再做结构性改动：tp / 量化 / 投机解码会重新加载权重甚至重新下载，成本高，只在廉价旋钮耗尽后考虑。

## workload 降档（保持可比性）

isl / osl / ref_concurrency 定义负载身份，单次 run 内绝不改动（会破坏可比性）。预算紧张时只允许下调 num_prompts（缩短单 trial 墙钟，不改变 goodput 语义）；降档后所有后续 trial 同口径，必要时重测基线，并在 results.tsv.description 记 `num_prompts:<原>→<新>`。若必须改 isl/osl/ref_concurrency，则结束当前 run、开新 run 重新基线。

## 环境失效与共租处置

1. **显存不足 / 被外部占用致起服失败**（`Free memory ... less than desired`）：判为可自愈 crash，按序尝试——(a) 用 env.CUDA_VISIBLE_DEVICES 缩到空闲 GPU 子集并同步下调 tp；(b) 下调 gpu-memory-utilization；(c) 若 target.yaml 预声明了 fallback 主机/模型则切换；(d) 仍不行记 crash，继续下一配置。
2. **contention（检测到外部 GPU 进程）**：(a) 切到空闲的声明候选主机/GPU 子集 →(b) 有限时间内重探后重试（参考上限 15min，可按主机调整）→(c) 持续共租则带 contention 告警继续，并在该轮标注「共租，仅供横向参考」；不求助人类。
3. **主机整体不可达 / 模型路径消失**：停止循环，诊断写入 report.md。

## 崩溃处理

- **serve flag 预校验**：`bench.py` 在起 serve 前对照镜像的 `vllm serve --help=all`（按 image 缓存到 out-dir；bare `--help` 在现代 vLLM 只给配置组索引，不可用则退回并最终 fail-open 跳过）校验 `config.yaml.serve` 的 flag 名——未知 flag 直接判 crash（fail fast，typo 不浪费一次模型加载）；镜像无法提供 help 时跳过校验（best-effort）。
- 明显的 typo / 缺依赖：可修一次并重试该配置。
- 构思本身有问题（如参数组合不合法）：记 crash 跳过，不反复重试。

## 子 agent 分派

- **固定分派点**：Phase 1 —— 探索子 agent 核验环境事实与踩坑查证；Phase 3 —— 周期性（每 N 轮，N 取 3–5，可由 harness 配置）一次分析子 agent 做趋势 / 瓶颈体检；Phase 4 前 —— 验证子 agent 独立复核 best 回归与结论。
- **触发**：需要跨文件 / 跨知识检索，或结论将被 keep / 采纳时派；纯机械执行（改一个已定参数、跑一次 run.sh）不派。
- **评估可执行化**：评估子 agent 优先直接运行 scripts/mock_loop.py 的纯函数，不靠心算。
- **成本护栏**：同一 Phase 内子 agent 总预算设上限；验证角色不得复用产生结论者的会话。
- **降级**：harness 不支持子 agent 时，以上角色由主循环按同等摘要纪律内联执行。

## 收尾

满足任一退出条件（主退出 / 平台期 / 崩溃连击 / SLO 不可达退出）→ 进 Phase 4：

1. **确认 best 回归**：`git checkout <best_commit> -- config.yaml` → `bash run.sh` → 轮询 `bash run.sh --poll` 至退出 0（done）→ 在 `results.tsv` 追加一行 `status=confirm`（description 如 `confirm best`）。目的：排除夜间爬山产生的「噪声最优」。
2. 运行 `python report.py` 产出 `report.md`（含「确认（best 回归）」判定节）。

循环结束后不要再开新 trial。
