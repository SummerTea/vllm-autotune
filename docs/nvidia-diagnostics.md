# 外部资源参考：NVIDIA GPU 诊断与监控（供调优 Agent 使用）

> 调研时间 2026-10。**未实机验证**命令与字段；DCGM `dcgmi diag` 的 JSON schema 随版本变化，落地前请在目标驱动/DCGM 版本上冒烟。
> 用途：vllm-autotune 的 (a) 基准前体检 (b) 运行中监控 (c) 故障诊断（Xid/ECC/热节流/NVLink/OOM/掉线）。

## 1. Agent Skills（skills.sh / Claude Skills / opencode）

| 名称 | URL | 用途 | 契合 |
|---|---|---|---|
| **NVIDIA/skills（官方目录）** | github.com/NVIDIA/skills | 官方 skills（Dynamo/Jetson 诊断/DOCA…）；`npx skills add nvidia/skills --skill <name>` | 中高（**CLI 需 ≥ v1.5.16**） |
| **sickn33/agentic-awesome-skills** | github.com/sickn33/agentic-awesome-skills | `gpu_health.sh`：`nvidia-smi dmon` 采样 + dmesg Xid（48/79）+ 节流判据，可直接移植为 pre-run gate | **高** |
| **air-gapped/skills** | github.com/air-gapped/skills | `nvidia-datacenter-bringup`、`gpu-host-tuning`（PCIe/IRQ/NCCL env/BIOS 审计） | **高** |
| **vllm-project/vllm-skills** | github.com/vllm-project/vllm-skills | vLLM 运维/基准（本仓库已装 3 个），**无硬件诊断** | 低-中 |
| EvilFreelancer/gpu-server-setup | github.com/EvilFreelancer/gpu-server-setup | 裸机→驱动/CUDA/Docker→vLLM 的分阶段验证门 | 中 |
| NVIDIA/OSMO skills | github.com/NVIDIA/OSMO/tree/main/skills | OSMO 平台作业/失败诊断（走平台才有用） | 中 |

## 2. MCP Servers

| 名称 | URL | 用途 | 契合 |
|---|---|---|---|
| **ArangoGutierrez/k8s-gpu-mcp-server** | github.com/ArangoGutierrez/k8s-gpu-mcp-server | `get_gpu_health`、**`analyze_xid_errors`**、`get_nvlink_topology`、**`get_gpu_timeline`（30min flight recorder）** | 高（K8s；裸机需 `--mode=read-only` + root/CAP_SYSLOG） |
| **pmady/gpu-mcp-server** | github.com/pmady/gpu-mcp-server | NVML stdio：`get_gpu_metrics`/`get_gpu_processes`/`gpu_summary` | 中高（裸金属即用；无 Xid/ECC） |
| **prometheus/prometheus-mcp（官方）** | github.com/prometheus/prometheus-mcp | PromQL + runbook skills；配 dcgm-exporter 查 `DCGM_FI_*` | 高 |

> NVIDIA **无官方 DCGM/NVML MCP**；现有实现即上述社区项目。

## 3. NVIDIA 官方工具

| 名称 | 要点 | 契合 |
|---|---|---|
| **DCGM + dcgmi** | `dcgmi diag`（`--run 1/2/3/4`、`--json`、`--entity-id gpu:N`）、`dcgmi health`（被动 watch：ECC/热/PCIe replay/功耗/InfoROM/NVLink）；**open-core——诊断模块专有**；非数据中心卡仅 level 1 | **高** |
| **dcgm-exporter** | :9400 `/metrics`；`DCGM_FI_PROF_*`、`DCGM_FI_DEV_ECC_*`、`DCGM_FI_DEV_XID_ERRORS`(230)、`DCGM_FI_DEV_NVLINK_*` | 高 |
| **nvidia-smi** | `--query-gpu=… --format=csv`、`dmon -s pucvmet`、`nvlink -s/-e/-c`、`-q -d ECC/ROW_REMAP/NVLINK` | 高（零依赖兜底） |
| **NVSM** | DGX/CGX/RTX 整机专用常驻健康引擎（`nvsm dump health`） | 低（仅 NVIDIA 整机） |
| **nvidia-bug-report.sh** | 全量支持包（需 root） | 中（事故取证） |
| **gpu-burn**（BSD-2） | 多卡满载压力测试（`-m`/`-tc`/`-l`） | 高（可疑结果后确认） |
| **nvbandwidth** | `--format json`、`--pair`；低带宽=NVLink 红旗 | 高（体检 + tp 决策） |
| **XID 目录（官方）** | docs.nvidia.com/deploy/xid-errors（码→成因/处置） | 高（归因知识库） |

## 4. 三个阶段：查什么 / 怎么查

### (a) 基准前体检（probe.py gate，失败即跳过该卡）
```bash
nvidia-smi --query-gpu=index,name,persistence_mode,ecc.mode,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.max.sm,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
nvidia-smi --query-gpu=ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total,retired_pages.single_bit,retired_pages.double_bit,retired_pages.pending_retirement --format=csv,noheader,nounits
dcgmi diag --run 1 --entity-id gpu:0 --json        # 需 root/DCGM；非数据中心卡仅 level1
nvidia-smi nvlink -e ; nvidia-smi -q -d NVLINK
journalctl -k --since "boot" | grep -iE "NVRM|Xid"   # 既往 Xid
```
关键：Persistence=Enabled、ECC=Enabled、`retired_pages.*`、throttle reasons 全 0。

### (b) 运行中监控（bench.py sampler，与 bench 同周期）
```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu,throttle_reasons.hw_thermal_slowdown,throttle_reasons.sw_thermal_slowdown,throttle_reasons.hw_power_brake_slowdown,throttle_reasons.sw_power_cap --format=csv,noheader,nounits -l 2
```
任一 throttle reason=1 且 GPU 忙 → **该轮结果不可信，丢弃**。或走 dcgm-exporter → prometheus-mcp。

### (c) 故障后诊断（report.py / 事故分支）
```bash
nvidia-smi -q ; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
journalctl -k --since "-30min" | grep -iE "NVRM: Xid|GPU has fallen"   # 对照官方 XID 目录
dcgmi diag --run 2 --entity-id gpu:N --json
sudo nvidia-bug-report.sh
```

## 5. 故障签名 → 检出工具

| 签名 | 含义 | 检出 | 处置 |
|---|---|---|---|
| **Xid 79** | GPU 掉线 | `journalctl -k \| grep "Xid 79"`；卡从 nvidia-smi 消失 | 停用该卡 |
| **Xid 48** | 不可纠正 ECC(DBE) | dmesg；`ecc.errors.uncorrected.*`；DCGM 311 | reset/reboot；报修 |
| **Xid 63/64** | 行重映射/页退役 | `-q -d ROW_REMAP`；`retired_pages.*` | 63 可续，64 reset |
| **Xid 13/31** | 非法访问/软件错 | dmesg | 查代码，非硬件 |
| **Xid 74** | NVLink 错误 | `nvidia-smi nvlink -e`；CRC/replay | 同链路>2 次报修 |
| **热节流** | thermal/power brake/cap | `throttle_reasons.*`=1 + 温度 | 结果作废重测 |
| **静默 ECC DBE** | 无 Xid | 周期巡检 `ecc.errors.*`（pre-run 必查） | 记录/报修 |
| **NVLink 低带宽** | 链路半速 | `nvbandwidth --pair --format json` | 限 TP 在 NVLink 簇内 |
| **OOM** | vLLM 退出/日志 | vllm 日志 `CUDA out of memory` | 属配置样本（非硬件） |

## Top-7 集成建议

| # | 资源 | 检出 | 集成位置 | 成本 |
|---|---|---|---|---|
| 1 | `dcgmi diag --run 1` + `dcgmi health` | 部署/持久/退役页/内存/PCIe | probe.py **pre-run gate**（`--json`，失败跳过该卡） | 低（需 DCGM+root；数据中心卡） |
| 2 | `nvidia-smi --query-gpu=…` CSV | ECC/退役页/节流/时钟/功耗 | pre-run gate + bench.py **during-run sampler** | 低（零依赖） |
| 3 | dmesg/journalctl Xid + 官方 XID 目录 | 全 Xid | report.py **post-crash 归因** | 低 |
| 4 | dcgm-exporter(:9400) + prometheus-mcp | 持续遥测（SM/DRAM/PCIe/NVLink、Xid 230） | during-run 证据层 | 中 |
| 5 | k8s-gpu-mcp-server / pmady-gpu-mcp-server | 健康评分/Xid 解析/NVLink/flight recorder | MCP 接入 | 低-中 |
| 6 | gpu-burn + nvbandwidth | 满载热/功耗；NVLink 带宽 | 失败分支确认 | 低-中 |
| 7 | sickn33 `gpu_health.sh`（移植） | dmon+Xid+节流三合一 | pre-run 脚本并入 probe | 低 |

## 版本/权限/弃用标注

- **`nvidia-smi` 不能直接读 Xid**（Xid 在 dmesg/kmsg；DCGM `DCGM_FI_DEV_XID_ERRORS`=230 是唯一数值入口）；`persistence` 只在 `-q` 输出。
- DCGM 为 **open-core**：诊断模块专有；缺 `nvvs` 时报 `NVVS_BIN_PATH`。
- `dcgmi diag --json` schema 随 DCGM 版本变化——解析需按版本适配。
- NVVS 独立工具已弃用（统一走 `dcgmi diag`）；NVSM 仅 NVIDIA 整机。
- `npx skills` CLI 需 **≥ v1.5.16** 才在 Claude Code 生效；sickn33/air-gapped 为社区项目，采用前复核 license。
