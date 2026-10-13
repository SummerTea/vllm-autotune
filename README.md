# vllm-autotune

用外部编码 agent 在真实 GPU 机器上自动循环调优 vLLM 服务参数：你只提供目标与预算，agent 自己探测环境、跑实验、保留改进、按时收尾。

## 灵感来源

本项目受 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) 启发：人类负责编排（写好 `program.md` 与 `AGENTS.md`），编码 agent 负责执行。**你不改代码，只改 `config.yaml`**；agent 自动跑实验，用实测 goodput 决定 keep / discard，并把每个配置留在 git 历史里（线性 commit）供你复盘。

## 核心思想

- **单点可变**：整个仓库对 agent 只有一个可编辑文件 `config.yaml`，其余均为只读 harness。改动面小，实验结果可审计。
- **可比性**：固定 workload 与参考并发点（见 `harness.yaml`），所有实验在同一输入分布下比较，噪声可统计。
- **goodput 标量化**：单个标量 = 在 SLO（ttft_p99 / tpot_p99）约束下成功请求的吞吐（单位 req/s，即 vLLM 的 `request_goodput`；输出 token 吞吐作为辅助诊断列）。调优目标明确，keep/discard 判定机械可执行。
- **三层退出**：deadline 主退出 + 平台期 + 崩溃连击，过夜运行自动收尾，不留僵尸进程。

## 如何使用（fork 指南）

1. Fork / clone 本仓库。
2. `cp target.example.yaml target.yaml`，填写主机、模型、SLO、预算。
3. `cp .env.example .env`，填写 SSH 选项（端口、私钥路径、远端目录）。
4. 在编排机（本仓库目录）运行 `bash setup_remote.sh`（一次性，经 ssh 对 head 主机执行并 pin vLLM 版本）。
5. 把编码 agent（Claude Code / Codex / opencode）指向本仓库。
6. 对它说：「读 AGENTS.md，开始」——之后它自主循环，你睡觉。

## 仓库结构

| 文件 | 说明 | 可编辑性 |
|---|---|---|
| `config.yaml` | ★ 一次实验的全部输入（env / model / serve 参数） | **agent 可编辑（唯一）** |
| `target.yaml` | 由 `target.example.yaml` 复制而来；主机/模型/SLO/预算（本地文件，已 gitignore，勿提交） | 人类填写（启动前） |
| `target.example.yaml` | target 的模板 | 只读 |
| `harness.yaml` | 全局默认值与循环控制参数（噪声带、超时、退出阈值） | 只读 |
| `.env` | 由 `.env.example` 复制而来；SSH 传输选项 | 人类填写（启动前） |
| `.env.example` | .env 模板 | 只读 |
| `AGENTS.md` | agent 的入口：权限边界 + 四阶段流程 | 只读 |
| `program.md` | 调优循环细则：决策规则、退出机制、剪枝策略 | 只读 |
| `README.md` | 本文件 | 只读 |
| `probe.py` | 多机环境探测（并行）→ env.json / env_report.md / model_profile.json | 只读 |
| `bench.py` | 远端起服 + `vllm bench serve --goodput` 度量 + 僵尸清理 | 只读 |
| `report.py` | 收尾汇总 → report.md | 只读（后续阶段交付） |
| `run.sh` | 两段式执行：rsync + 远端 nohup 启动 + 轮询状态 | 只读 |
| `setup_remote.sh` | head 主机一次性环境准备（pin vLLM 版本） | 只读 |
| `docs/vllm-facts.md` | 模板依赖的 vLLM 行为事实（版本相关，升级需重新核对） | 只读 |
| `pyproject.toml` / `LICENSE` / `.gitignore` | 工程元数据 | 只读 |

## Design choices

- **固定 workload 与参考并发**：所有 trial 用同一合成 workload 与参考并发点（`harness.yaml`），保证配置之间可比。
- **goodput 标量化**：以 `vllm bench serve --goodput` 的输出为唯一指标，SLO 来自 `target.yaml`（缺省用 `harness.yaml` 占位）。
- **三层退出**：deadline 主退出（硬保证）+ 平台期（连续 N 次无 > 噪声带的改进）+ 崩溃连击（连续 N 次 crash）。每轮开头做 deadline 感知调度，来不及就不开新 trial；在途 trial 宽限跑完。
- **系统参数只建议**：governor / hugepages / 锁频等系统级参数只探测并写入报告建议，不自动修改；v1 不提供自动修改开关。
- **多机采集 + 单机执行**：v1 只在 `role=head` 的主机上跑实验，其余主机仅做采集探测；跨机调度留到 v2。

## 支持矩阵

- **已支持**：NVIDIA GPU + Mellanox（RoCE / IB）网络。
- **未支持**：AMD / ROCm（欢迎 fork 扩展）。

## 首次使用检查清单

> 模板交付前只在 mock 级验证过，真机首次使用务必按顺序冒烟：

1. 跑 `python probe.py`，检查生成的 `env_report.md`（系统参数清单 pass / warn / fail）。
2. 用初始 `config.yaml`（基线）跑 `bash run.sh` 并轮询 `bash run.sh --poll` 至退出 0，确认 bench 出数正常。
3. 跑一次短预算循环，确认 keep / discard 判定与三层退出逻辑正确。

> 注：`probe.py` / `bench.py` / `run.sh` / `setup_remote.sh` 均已交付（清单第 1、2 条可执行）；仅 `report.py` 留待后续阶段交付，届时第 3 条可执行。

## 依赖

- 远端：vLLM（版本由 `setup_remote.sh` pin 住）+ NVIDIA 驱动 + uv。
- 本机：Python 3.10+。

## License

MIT，见 [LICENSE](LICENSE)。
