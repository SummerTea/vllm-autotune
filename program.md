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

搜索空间与剪枝由**四类环境维度**（主机 / NUMA / 网络 / 存储）共同决定，综合 `env.json` / `env_report.md` / `model_profile.json` 的具体事实，**不是单一指标**（如仅凭「有无 RDMA」）。每条剪枝/纳入都须引用上述文件的具体事实；环境事实变化（NIC/RDMA 状态、拓扑、时钟等）→ 重新评估相关维度；环境相关决策（含「放弃某维度」）写进 `results.tsv.description` 或 `report.md`。

### 主机（GPU / 拓扑 / 时钟 / 系统参数）
- **探测依据**：`env.json`——`gpu.count` / `gpu.topo`（NVLink/NVSwitch vs 仅 PCIe）、`gpu.gpus`（时钟 / 持久模式）、`cpu.governor`、`mem.thp`。
- **剪枝/纳入**：NVLink/NVSwitch 全互联 → tp 可到 GPU 数；**仅 PCIe → tp ≤ 4 仅是性能建议**（优先并行数据 dp）——**容量优先**：若 `model_profile.json` 的 `tp.min_tp_for_weights` > 4，则必须超出该建议（跨 NUMA 判定见「NUMA 与划分」节，不因该建议排除可行的 tp）。合法 tp 集合由 `num_attention_heads` / `num_key_value_heads` 的约数决定，非法 tp 值不试（例外：`num_key_value_heads`≤1（MLA / MQA）时 vLLM 复制 KV heads，合法 tp 只受 attention heads 约数约束）。governor / hugepages / 锁频 / 主机级 NUMA 亲和绑定等只探测 + 建议（写进 `report.md`），不自动修改（v1 不提供自动修改开关）。
- **留痕**：系统级建议写 `report.md`。

### NUMA 与划分（placement）
- **探测依据**：`env.json`——`numa_groups`（`{numa_node, gpus}`）、`gpu.topo` / `interconnect`（域内/跨域连接层级：NV#/PXB/NODE/SYS 与全互联判定）、`net.rdma.devices[].gpu_links` 与 `speed_gbps`（NIC↔GPU 亲和与速率）+ `model_profile.tp`（合法 tp 约数与 `min_tp_for_weights`）。
- **候选枚举口径**：先按 `min_tp_for_weights` 与合法 tp 约数过滤**可行性**——若 `min_tp_for_weights` > 单 NUMA 节点卡数 → **必然跨 NUMA**（既定事实，不再反复质疑），跨 NUMA 损耗须实测，可选项：跨 NUMA 通信优化（NCCL env，见「网络」）；装得下才谈划分。再枚举划分 = **GPU 子集**（`env.CUDA_VISIBLE_DEVICES`，如 `0-3` / `4-7` / `0-7`，`tp` 按可见 GPU 数设置）× **并行方案**（tp / pp / dp，flag 见 `docs/vllm-facts.md` §4）× **容器级 `--cpuset-cpus/--cpuset-mems`**（经 `runtime.extra_docker_args`，仅影响本 trial 容器，零破坏）；每个候选须能引用实测互联数据（域内/跨域连接码、NIC↔GPU 亲和、速率）。
- **评估口径**：用「域内 PCIe 带宽 / 跨域 RoCE 带宽（若有 RDMA）」与每步通信字节量做**预估**（带宽 / 时延哪个受限），**不能只看拓扑层级**——tp 组限同 NUMA 仍是**性能建议**（如 8 卡两节点 → tp=4 选 `[0,1,2,3]` 或 `[4,5,6,7]`），但候选须**测量 + 预估**后再选（互联成本因候选而异：域内 PCIe P2P vs 跨域 RoCE）；划分变小会牺牲容量/并发，须与 goodput / SLO 一起权衡。
- **互联形态覆盖（鲁棒）**：互联评估须对全部形态给出口径，**任一类缺失 → 降级并记录（不得空转或依赖某项存在）**；判定以 probe 实测为输入（`gpu.topo` 连接码、`interconnect`、`net.rdma.kind` / `devices[].speed_gbps`）：
  - **NVLink/NVSwitch**：域内高带宽，tp 可放大；跨节点走 IB/RoCE（NCCL env 见「网络」）。
  - **PCIe-only + IB/RoCE**：跨域可选走网络——配方 `NCCL_P2P_LEVEL=PXB` + `NCCL_SHM_DISABLE=1`（见「网络」事实）。
  - **PCIe-only 且无 RDMA**：跨 NUMA 只能 PCIe P2P / SHM / TCP，**无 NET 选项** → 倾向 tp 落在同一 PCIe 域/NUMA、或 dp；把「无网络可用」记为显式约束（写 `results.tsv.description`）。
- **决策与留痕**：所选划分 + 依据写 `results.tsv.description` / `report.md`；**主机级 NUMA 划分**（BIOS NPS / SNC / sub-NUMA clustering 等）→ 只探测 + 建议（写 `report.md` 系统级建议节），不自动改；主机级绑定（`numactl`）→ 只建议、不自动。

### 网络（三分：InfiniBand / RoCE / 无）
- **探测依据**：`env.json` 的 `net.rdma`——`ib_devices` 非空即有 RDMA（如 `mlx5_0`），IB / RoCE 按设备 link_layer 区分（RoCE=Ethernet，IB=InfiniBand）；**不要仅凭 `ibstat_present` 判无**（ibstat 缺席 ≠ 无 RDMA）。结合 NIC↔GPU 亲和（同 NUMA / PCIe 拓扑）与链路速率（来自 probe）。
- **剪枝/纳入**：三分 InfiniBand / RoCE / 无——「无」→ NCCL/RDMA 相关 env 移出搜索空间；「IB / RoCE 存在」→ NCCL/RDMA env **进入搜索空间**（如 `NCCL_P2P_LEVEL`、`NCCL_IB_HCA`、`NCCL_IB_GID_INDEX`、`NCCL_NET_GDR_LEVEL`、`NCCL_SHM_DISABLE`，名单见 `docs/vllm-facts.md` §6），并按 NIC↔GPU 亲和选 HCA 与 GDR 等级。**NCCL 事实**（口径见 vllm-facts §6）：`NCCL_P2P_LEVEL` 是距离 cutoff（官方文档口径 LOC/NVL/PIX/PXB/PHB/SYS，无 NODE），Intel x86 默认 PXB——**PHB 及以上**默认关 P2P、回落 SHM（主机内存/UPI 中转），**不走网络**（注：`PHB` 是同 NUMA 穿 CPU，`SYS` 才跨 NUMA）；NCCL 仅当 P2P 与 SHM 双双不可用才把该对标 PATH_NET（走 RoCE/IB）——「域内 PCIe P2P + 跨域 RoCE」的配方是 `NCCL_P2P_LEVEL=PXB` + `NCCL_SHM_DISABLE=1`（+ `NCCL_IB_HCA="=<dev>:1"` 精确匹配）。NIXL / PD 分离相关 env 需 agent 自行查证后使用。
- **留痕**：放弃 RDMA 维度 / HCA 与 GDR 选择写 `results.tsv.description` / `report.md`；RDMA 状态变化 → 重新评估。

### 存储 / 加载
- **探测依据**：`env.json` 的 `disk.avail_bytes`（模型盘可用空间）+ 校准后 `model_profile.json` 的 `boot.measured_seconds`（真实加载耗时）与权重量。
- **剪枝/纳入**：加载耗时占单 trial 墙钟比例高（如 >40%）时，把 `--load-format`、权重预热/缓存、`--max-model-len`（KV/加载权衡）等纳入候选；模型盘可用空间不足时优先加载/量化权衡。
- **留痕**：加载相关决策写 `results.tsv.description` / `report.md`。

### 其它维度（沿用）
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
2. **contention（检测到外部 GPU 进程）**：(a) 切到空闲的声明候选主机/GPU 子集 →(b) 有限时间内重探后重试（参考上限 15min，可按主机调整；只读占用探测用 `bash run.sh --check-gpus`，退出码 0=空闲 / 1=有外部占用）→(c) 持续共租则带 contention 告警继续，并在该轮标注「共租，仅供横向参考」；不求助人类。
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
