# 路线图（roadmap.md）

> 将 `docs/external-resources.md` 的外部调研落成可执行计划。两条轨道：
> **v1.1（低成本增强）** 在现架构上叠加证据/验收层；**v2（架构扩展）** 需要重新设计执行拓扑。
> 每条目给出：目标 / 集成方式 / 成本 / 依据（引用 external-resources.md 小节）。

## 现状与取舍

- v1 保持**单节点（head 独占）+ docker-only 运行时**：跨机涉及 NCCL/RDMA 拓扑与故障域，v2 才做；单节点上「serve 参数 + 进程级 env」已可覆盖大部分收益。vLLM 环境复杂，运行时统一收敛为 docker 镜像（此前曾支持本机安装式，已移除）。
- 零破坏契约不变：默认不 pull 镜像、不安装环境；v1.1 的监控/基准均为可选旁路，缺失不影响默认循环。
- 真机已验证：**单节点多卡**（docker 运行时）跑通 probe → bench → report 全链路与校准。
- 已随仓库交付工程级 skills（`.agents/skills/`：vllm-bench-serve / vllm-deploy-docker / vllm-prefix-cache-bench，见 `skills-lock.json`）。
- **NUMA 采集/约束已实现**：`probe.py` 产出 `env.json` 的 `numa_groups`（每 GPU `numa_node`）+ 检查清单「NUMA 放置」建议（warn on 多 NUMA）+ `program.md` 剪枝规则（tp 组限同 NUMA 节点）；系统级绑定仍**只建议、不自动**（零破坏契约不变）。

## v1.1 — 低成本增强

### (a) 监控证据层（GPU 瓶颈证据）
- **目标**：让 keep/discard 具备硬证据——SM 占用、NVLink/PCIe 吞吐、Xid 错误，不再只靠 goodput 反推。
- **集成方式**：head 上 `dcgm-exporter`(9400) + 单实例 Prometheus（仅 work_dir 内、零全局安装）；agent 经 `prometheus/prometheus-mcp` 跑 PromQL（`DCGM_FI_PROF_SM_ACTIVE`、`DCGM_FI_PROF_PCIE_TX/RX_BYTES`、`vllm:kv_cache_usage_perc` 等）。非致命：缺失时跳过证据记录。
- **成本**：低-中。
- **依据**：external-resources.md §1（prometheus-mcp 行）、§2（DCGM/dcgm-exporter）、§3（vllm 原生指标）、Top-8 #3。

### (b) 第二基准 / 验收（GuideLLM）
- **目标**：独立于 `vllm bench serve` 的 SLO 验收，防单工具系统性偏差。
- **集成方式**：收尾/门禁用 GuideLLM `sweep` profile + json 输出，`--constraint max_error_rate` 校准；结果写入 report.md 的验收小节。
- **成本**：低。
- **依据**：external-resources.md §4（GuideLLM 行）、Top-8 #5。

### (c) 拓扑实测（静态 → 实测带宽）— ✅ 已实现
- **现状**：`probe.py` 已实测 **GPU↔GPU P2P 带宽**（容器内 torch 测例，同/跨 NUMA）与 **跨 NUMA RoCE/IB 带宽**（`ib_write_bw`），写入 **`env.json` 的 `hosts.<name>.interconnect`**（`gpu_p2p_bw_gbps` / `rdma_bw_gbps`；不可得 → null + note，不阻塞）。同时补齐 **NIC↔GPU 亲和与速率**（`net.rdma.devices[].gpu_links/speed_gbps/numa_node`）与**模型盘顺序读带宽**（`disk.read_bw_gbps`）。`program.md` 的 NUMA/placement 与网络剪枝以这些**实测值**为输入。
- **仍未做**：`nvbandwidth` / `nccl-tests all_reduce_perf`（非必须；当前以容器 torch 测例 + `ib_write_bw` 覆盖）。
- **依据**：external-resources.md §7（nvbandwidth/nccl-tests/perftest）、Top-8 #6。

### (d) 精度 gate（lm-eval 小子集）
- **目标**：量化/kv-cache-dtype 等改动不伤质量——nightly 小子集守卫。
- **集成方式**：夜间（或低成本时段）对当前 best 跑 lm-eval 小子集；回归则记录并回退建议。
- **成本**：中（需模型推理时间，小子集可控）。
- **依据**：external-resources.md §4（lm-evaluation-harness 行）。

### (e) 可选大脑（vllm-tuner）
- **目标**：把循环的"下一个候选"交给 HPO 大脑（Optuna NSGA-II），替代纯启发式。
- **集成方式**：`vllm-tuner`（PyPI）作候选生成器，benchmark 提供方用 vLLM built-in；先小冒烟再接入（新项目 v0.1.0，需核 license/维护）。
- **成本**：中。
- **依据**：external-resources.md §6（vllm-tuner 行）、Top-8 #4。

### (f) opt-in 自动 NUMA/cpuset 绑定（**明确未做**）
- **现状**：NUMA 采集/约束已实现（`probe.py` `numa_groups` + 「NUMA 放置」检查清单建议 + `program.md` 剪枝规则）；系统级绑定仍只建议、不自动。
- **目标**：把「NUMA 亲和」从只建议升级为可 opt-in 的自动绑定——容器 `--cpuset-cpus/--cpuset-mems` + `env.CUDA_VISIBLE_DEVICES` 限定可见 GPU、必要时宿主机 `numactl` 绑定（改动前记录原值、失败自动回滚）。**注意**：`build_docker_run_cmd` 已固定 `--gpus all`，若将来改用 `--gpus device=` 限卡，须先在构建器做 `--gpus` 去重（或统一走 `CUDA_VISIBLE_DEVICES`）。
- **成本**：低-中；属 v2「系统级参数白名单自动调」的前置子项。

## v1.2 — 真机演练新增待办

> 来源：一次全流程真机演练。**前两项属设计决策，需人工拍板后再实现**；其余为低成本增强。

- **(g) 高方差 workload 的噪声模型**：keep 阈值 `max(best×3%, k×σ)` 在 σ 接近 best 量级时几乎无法 keep（实测 σ 可达 best 的约一半），循环会退化为「平台期退出」。候选手段：**丢弃预热轮**（首轮冷启动污染可显著放大 ttft_p99 的组间波动）、**σ 上限**、**自适应 repeats**。注意：改 `harness.yaml` 语义会影响跨 run 可比性，需与历史行区分标注。
- **(h) `fallbacks`（换机 / 换模型）**：`target.example.yaml` 已留注释位，但**当前未实现**（`run.sh` 硬绑唯一 `role:head`）。需要：target schema（候选主机/模型）+ `run.sh` 的选主/切换与失败回退机制 + 留痕。
- **(i) `loop-orchestration` skill**：把「主循环唯一写者 / 子 agent 只读认知分工（探索·分析·验证·评估）/ 摘要即弃」的纪律做成 `.agents/skills/` 工程级 skill，使不支持子 agent 的 harness 也有一致降级口径。
- **(j) `scripts/mock_loop.py --score`**：让「评估」角色直接跑纯函数打分，而非 LLM 心算 keep/discard。
- **(k) probe 的宿主回退**：docker-only 下若某次容器实测抖动失败，`gpu_p2p_bw` 会「未采集」（当前无宿主回退）——可加退避重试，或在 note 中写明成因（无 `runtime.image` / docker 不可用 / GPU 忙）。

## v2 — 架构扩展

### 跨机执行（PP/DP across nodes）
- **目标**：单节点装不下的模型 → 多机 tensor/pipeline 并行。
- **集成方式**：扩展 hosts/worker 语义（当前 worker 仅采集探测），run.sh/bench.py 支持跨机 NCCL；依赖拓扑实测（v1.1-c）确认互联天花板。
- **成本**：高。
- **依据**：external-resources.md §7（nccl-tests 跨节点）、§6（NCCL perf tuning 文档）。

### PD 分离（NIXL）
- **目标**：prefill/decode 解耦，单机 KV 复用率低时提升吞吐。
- **集成方式**：需 RDMA（RoCE/IB）+ NIXL/PD 相关 env 进入搜索空间（见 program.md 剪枝规则）；先在有 RDMA 的节点上验证。
- **成本**：高。
- **依据**：external-resources.md §6（Dynamo disagg 需 RDMA）。

### NVIDIA Dynamo AIConfigurator（预测初值）
- **目标**：用离线 ~30s 预测给 TP/PP/workers 提供初值，压缩搜索起点。
- **集成方式**：`aiconfigurator cli default --ttft … --tpot … --backend vllm` 产出初值 → 写入 config.yaml 作为首轮基线候选；预测式与实测互补。
- **成本**：中-高。
- **依据**：external-resources.md §6（Dynamo + AIConfigurator 行）、Top-8 #8。

### 系统级参数白名单自动调
- **目标**：governor/hugepages/NUMA 亲和等系统参数从"只建议"升级为白名单自动调（仅 opt-in，仍限 work_dir 与可回滚项）。
- **集成方式**：借鉴 gpu-host-tuning skill 清单（NUMA/PCIe/IRQ/NCCL env）建白名单；改动前记录原值、失败自动回滚。
- **成本**：中。
- **依据**：external-resources.md §1（air-gapped/skills 的 gpu-host-tuning 行）。

## 暂缓（明确暂不做）

- **并行实验（单机多 GPU 子集并发跑多配置）**：暂不做（决定见 loop 加固会话）。当前为单 trial 串行 + 多层退出；并行需唯一容器名/端口、每 lane 独立产物与聚合，收益大但改动面广，后续再评估。
