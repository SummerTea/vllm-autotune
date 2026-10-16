---
name: host-explore
description: Read-only host exploration for vllm-autotune (Phase 0.5). Runs host-only `probe.py` and hands back a bounded digest plus a self-contained `host_report.html`. Use when the user gives a host (an ssh target) and wants a rough evaluation of a machine — GPU, network, available vLLM images, model artifacts, current occupancy — before issuing a tuning instruction.
---

# Host Explore（Phase 0.5 主机探索）

给定一台主机（只有 ssh 目标），**只读**采集并产出一份人类可读的 `host_report.html`，同时向主循环回传**≤15 行摘要**。目标：人类看报告「大概了解」该主机 → 再下发调优指令（model / SLO / budget / work_dir）。

## 何时用

- 用户只给了主机，没给 model / SLO / budget / work_dir。
- 需要评估「这台机器能不能跑、适合跑什么、现在是否空闲」。
- **不适用**：已有完整 target.yaml 的 Phase 1 正式探测（那是主循环跑 `python probe.py` 的全量产物）。

## 输入

- `target.yaml`：**只需 `hosts` 段**（`name` / `ssh` / `role: head`）。缺 model / SLO / budget / work_dir 是正常的，**不要为此追问人类**。
- `.env`：`SSH_PORT` / `SSH_KEY`（`probe.py` 会自动加载）。勿在命令行硬编码私钥。

## 步骤

1. 确认 `target.yaml` 为 host-only（只有 `hosts`）。
2. 运行（你是本阶段唯一执行者；对远端只读）：
   ```bash
   python probe.py          # 或 uv run python probe.py
   ```
   产物：`env.json` / `env_report.md` / `model_profile.json` / `search_space.json` / **`host_report.html`**。
3. **读 `env.json`（这落在你自己的上下文里，不是主循环的）**，按「摘要契约」归纳。
4. 回传：≤15 行摘要 + `host_report.html` 的绝对路径。

## 摘要契约（≤15 行，原文不回传）

按序覆盖，缺项写 `—`：

1. 主机名 + ssh + role
2. GPU：`{count}× {name}`、单卡显存 GiB / 合计 GiB
3. 互联：`gpu.topo.summary`（`nvlink-full` / `pcie` / …）→ **tp 建议**（`pcie` ⇒ 限同一 `numa_groups` 组内）
4. 网络：NIC 数 + RDMA 类型/速率 + 实测带宽（`interconnect.gpu_p2p_bw_gbps` / `rdma_bw_gbps`）
5. 可用 vLLM 镜像：`docker.vllm_images`，各带 `docker.images[].size`
6. 模型介质：`model_inventory` 里**能否装下**（`size_bytes ≤ Σ 显存`）的模型，按大小降序，标出量化 `quant_method`
7. 当前占用：`docker.containers` / `ports.listening` / `users.logged_in` 的数量；**`ports.vllm_port_conflict` 为真必须显式告警**
8. 检查清单：`pass/warn/fail` 计数；**fail 项一句话**
9. 结论：能跑什么 + 阻塞项 + `host_report.html` 路径

## 硬约束

- **只读远端**：只跑 `probe.py`。不改主机状态、不 pull 镜像、不起 trial。
- **可写范围仅限探测生成物**：`env.json` / `env_report.md` / `model_profile.json` / `search_space.json` / `host_report.html`。**绝不写** `config.yaml` / `target.yaml` / `results.tsv` / report.md / git。
- **原文不回传**：`env.json`（数百行）留在你的上下文；主循环只收摘要 + 报告路径。人类直接打开 `host_report.html`。
- **不追问**：model / SLO / budget / work_dir 缺失属预期，由人类看过报告后下发。

## 降级

harness 不支持子 agent 时，主循环内联执行，但保持同一摘要纪律：只跑 `probe.py`、只把 ≤15 行摘要与报告路径留在主上下文，**不把 `env.json` 读进来**。
