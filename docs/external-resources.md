# 外部资源参考：vLLM / GPU 调优辅助 skills、MCP 与工具

> 调研时间 2026-10；vLLM 相关事实以 v0.31.0 源码为准。**未实测安装运行**；新项目（v0.1.0 等）集成前请再核实 license/维护状态。
> 用途：本模板（vllm-autotune）的循环工程可借用的外部 skills / MCP / CLI / 库。

## 结论先行（三件套）

1. **`vllm-project/vllm-skills`**（官方 agent skills，Apache-2.0）——含 `vllm-bench-serve`、`vllm-deploy-docker` 等，Claude Code/opencode 可 `plugin marketplace add` 直接装。
2. **vLLM 自带 `benchmarks/auto_tune/auto_tune.sh`**——与本项目 loop 几乎同构（`vllm bench serve --goodput e2el:<ms>` + 网格搜索 + profile 落盘），是官方同构先例。
3. **Optuna 系 `vllm-tuner`（PyPI v0.1.0）**——多目标（NSGA-II）自动调参，benchmark 提供方可选 vLLM built-in，自带 OOM/KV/preemption telemetry 解析。

---

## 1. Agent Skills / MCP Servers（GPU & LLM-serving 运维）

| 名称 | URL | 类型 | 用途 | 契合 |
|---|---|---|---|---|
| **vllm-skills（官方）** | github.com/vllm-project/vllm-skills | agent-skill（SKILL.md / plugin marketplace） | `vllm-deploy-docker`、`vllm-bench-serve`、`vllm-bench-random-synthetic`、`vllm-prefix-cache-bench` | **高**（与 loop 同构） |
| **NVIDIA/skills** | github.com/NVIDIA/skills | agent-skill 目录（NVIDIA 官方，每日同步+安全扫描） | CUDA/NeMo/**Dynamo**/Jetson 等 | 中 |
| **air-gapped/skills** | github.com/air-gapped/skills | Claude Code plugin（58 skill） | `vllm-benchmarking`（goodput SLO、TTFT/TPOT/ITL 百分位）、`vllm-configuration`、`gpu-host-tuning`（NUMA/PCIe/IRQ/NCCL env） | **高** |
| **prometheus/prometheus-mcp（官方）** | github.com/prometheus/prometheus-mcp | MCP-server | agent 跑 PromQL，读 vllm:/dcgm 指标 | **高** |
| **grafana/mcp-grafana（官方）** | github.com/grafana/mcp-grafana | MCP-server | 经 Grafana 查 Prometheus/Loki | 中高 |
| **pmady/gpu-mcp-server** | github.com/pmady/gpu-mcp-server | MCP-server（NVML） | list/query GPU 指标、进程、NVLink 吞吐 | 中高 |
| **jaimenbell/vllm-ops-mcp** | github.com/jaimenbell/vllm-ops-mcp | MCP-server | 健康探测 + 从 `/proc/<pid>/cmdline` 反读运行参数 | 中 |
| **Jwrede/llmprobe** | github.com/Jwrede/llmprobe | CLI+MCP | 端点合成监控（TTFT/吞吐/错误率），Prometheus 输出 | 中 |
| **rhobs/obs-mcp** | github.com/rhobs/obs-mcp | MCP-server | Prometheus/Thanos/Alertmanager 查询（Red Hat） | 中 |

> 注：vLLM v0.31 自带的 `vllm/entrypoints/mcp/` 是**给模型提供工具调用**的 Harmony server，非运维 MCP。

## 2. GPU 监控 / 遥测

| 名称 | 关键点 | 契合 |
|---|---|---|
| nvidia-smi（query/dmon/pmon） | `--query-gpu=... --format=csv,noheader` 机器可读；`dmon` 连续采样；`pmon` 进程级 | 高（零依赖） |
| DCGM + dcgmi + dcgm-exporter | 数据中心级：Xid/ECC/**PROF_ 字段**（SM/DRAM 占用、PCIe/NVLink 吞吐）；exporter:9400 `/metrics` | 高 |
| nvitop / gpustat | TUI + JSON（`-1`/`--json`） | 中 |
| node-exporter + GPU operator | 主机级 + DCGM 导出 | 中高 |

## 3. vLLM 原生指标与工具（v0.31）

**Prometheus `/metrics` 关键指标名**（`vllm/v1/metrics/loggers.py`）：

```
vllm:num_requests_running / vllm:num_requests_waiting
vllm:kv_cache_usage_perc                      # 旧名 gpu_cache_usage_perc（已漂移）
vllm:prefix_cache_queries / vllm:prefix_cache_hits
vllm:num_preemptions
vllm:generation_tokens / vllm:prompt_tokens
vllm:request_success                          # 旧脚本常写 request_success_total（不适用）
vllm:time_to_first_token_seconds              # TTFT
vllm:inter_token_latency_seconds              # ITL
vllm:request_time_per_output_token_seconds    # TPOT（多次改名）
vllm:e2e_request_latency_seconds              # E2E
vllm:request_queue_time_seconds               # 排队（HPA 常用）
# 公共 label：model_name, engine(=DP rank)
```

- **漂移警告**：`request_success(_total)`、TPOT 名、`gpu_cache_usage_perc→kv_cache_usage_perc` 均有历史改名，做 PromQL 前先 `curl /metrics` 核对。
- **VLLM_* 观测/调优**：`VLLM_LOGGING_LEVEL`、`VLLM_LOG_STATS_INTERVAL`(10s)、`VLLM_ENGINE_READY_TIMEOUT_S`、`VLLM_WORKER_MULTIPROC_METHOD`(nsys 时用 spawn)、`VLLM_NCCL_SO_PATH`。**`VLLM_TORCH_PROFILER_DIR` 已废除** → 用 `--profiler-config` 或 `/start_profile` REST。
- **启动日志校准行**：`GPU KV cache size: N tokens, Maximum concurrency for N tokens per request: X.XXx`。
- **`vllm bench` 子命令**：`serve / throughput / latency / startup / sweep / mm_processor`。

## 4. 负载生成器对比

| 名称 | SLO/goodput | 并发扫描 | 契合 |
|---|---|---|---|
| **GuideLLM**（vLLM 官方组织） | ✅ SLO 驱动；TTFT/ITL/E2E 分布 | ✅ `sweep` profile | **高**（第二基准/验收） |
| NVIDIA GenAI-Perf（Triton）/ AIPerf | 无 goodput SLO；AIPerf 支持 trace 回放 | ✅ | 中高 |
| LLMPerf（ray） | ❌ | 部分 | 中 |
| **evalscope perf**（ModelScope） | 无 SLO；指标与 `vllm bench serve` 逐项对齐 | ✅ | 中（中文生态） |
| **lm-evaluation-harness** | 质量（非性能） | — | 中（"调参不伤质量" nightly gate） |
| locust / k6 | ❌ | 有 | 低 |

## 5. Profiling

- **vLLM 官方 profiling 文档**（docs.vllm.ai/en/stable/contributing/profiling）：nsys 推荐 `--trace-fork-before-exec=true --cuda-graph-trace=node --capture-range=cudaProfilerApi`；torch profiler 经 `--profiler-config`。
- Nsight Systems `nsys`（低开销，`nsys stats`/`export --type=sqlite` 机器可读）；Nsight Compute `ncu`（kernel 级）。
- vLLM `auto_tune` 内建 profiler：最佳配置自动存 `.xplane.pb`。

## 6. 自动调参框架 / 先例

| 名称 | 机制 | 与本 loop 关系 |
|---|---|---|
| **vLLM `benchmarks/auto_tune/auto_tune.sh`** | bash 网格 + `--goodput e2el:<ms>` SLO 搜索 + profile | **官方原版**，可直接移植骨架；坑：`pkill -f vllm` 会误杀、`--load-format dummy` |
| **vllm-tuner（PyPI v0.1.0）** | Optuna（TPE/NSGA-II），benchmark=GuideLLM/HTTP/vLLM built-in，NVML 监控，telemetry 解析 | 几乎为本 loop 定制；新，需冒烟 |
| **Red Hat Auto-Tune vLLM** | GuideLLM + Optuna NSGA-II 多目标，300 trial → Pareto | 业界验证的多目标范式 |
| InseeFrLab/auto-tuning-vllm | Optuna + GuideLLM，本地/Ray | 轻量可自托 |
| **NVIDIA Dynamo + AIConfigurator** | 离线 ~30s 预测 TP/PP/workers（`--ttft/--tpot --backend vllm`） | 宿主机已跑 dynamo，高相关；预测式与实测互补；disagg 需 RDMA |
| Optuna / Ray Tune | 通用 HPO | loop 大脑可选 |
| Triton Model Analyzer / llm-d / KServe | Triton 专属 / 疑似停维护 / 平台层 | 低 |

## 7. 拓扑 / 互联工具

| 名称 | 要点 |
|---|---|
| **nvbandwidth**（官方） | `--format json`；测 NVLink/PCIe 每对带宽 → 划 TP 域 |
| **nccl-tests** | `all_reduce_perf -b 8 -e 128M -f 2 -g 8`，读 busbw；节点内 vs 跨节点天花板 |
| **perftest（ib_write_bw）** | `-a -d mlx5_0 -F`；`--use_cuda` 测 GPUDirect；400G≈392Gb/s |
| nvidia-smi topo / -p2p / nvlink -s | NV#/PIX/PHB/SYS/NODE → TP/PP 域划分（零安装） |
| NCCL perf tuning 文档 | RoCE 排障：`iommu=pt`+清 ACS+不设 `NCCL_IB_SPLIT_DATA_ON_QPS`（216→456 GB/s） |

## 8. 可复用 agent "skill" 生态

- **anthropics/skills + skills.sh**（`npx skills add <org>/<repo>`）；已收录 **NVIDIA/skills**、**vllm-project/vllm-skills**、**air-gapped/skills**、**sickn33/agentic-awesome-skills**（含 `gpu_health.sh` 三重探活）、**EvilFreelancer/gpu-server-setup**。
- 社区分发：skills.rest / ClawHub / Hermes Hub（质量参差，需审）。
- 建议：把 §2/§3 的指标采集封装成 **opencode skill（SKILL.md + 脚本）**，比 MCP 更贴合本地 loop。

---

## Top-8 集成清单（按性价比）

| # | 资源 | 集成方式 | 成本 |
|---|---|---|---|
| 1 | vLLM `auto_tune.sh` | 移植为 loop 决策/执行骨架（保留 `--goodput e2el:` 语义） | 低-中 |
| 2 | vllm-project/vllm-skills | 直接装 `vllm-bench-serve` / `vllm-deploy-docker` | 低 |
| 3 | prometheus-mcp + dcgm-exporter | exporter(9400) → Prometheus → agent 查 `vllm:`/`DCGM_FI_PROF_*` | 低-中 |
| 4 | vllm-tuner（Optuna） | `pip install vllm-tuner`，benchmark=vLLM built-in | 中 |
| 5 | GuideLLM | 第二基准/验收（`sweep` + json，`--constraint max_error_rate`） | 低 |
| 6 | nvbandwidth + nccl-tests + ib_write_bw | 开跑前一次性测，写入 config 元数据 | 低 |
| 7 | vllmstat（或 vllm-htop） | 后台 TUI + `--once --json` 快照 | 低 |
| 8 | NVIDIA Dynamo AIConfigurator | `aiconfigurator cli default --ttft … --tpot … --backend vllm` 生成初值 | 中-高 |
