# vllm-autotune

用外部编码 agent 在真实 GPU 机器上自动循环调优 vLLM 服务参数：你只提供目标与预算，agent 自己探测环境、跑实验、保留改进、按时收尾。

## 灵感来源

本项目受 [karpathy/autoresearch](https://github.com/karpathy/autoresearch) 启发：人类负责编排（写好 `program.md` 与 `AGENTS.md`），编码 agent 负责执行。**你不改代码，只改 `config.yaml`**；agent 自动跑实验，用实测 goodput 决定 keep / discard，并把每个配置留在 git 历史里（线性 commit）供你复盘。

## 核心思想

- **单点可变**：整个仓库对 agent 只有一个可编辑文件 `config.yaml`，其余均为只读 harness。改动面小，实验结果可审计。
- **可比性**：固定 workload 与参考并发点（见 `harness.yaml`），所有实验在同一输入分布下比较，噪声可统计。
- **goodput 标量化**：单个标量 = 在 SLO（ttft_p99 / tpot_p99）约束下成功请求的吞吐（单位 req/s，即 vLLM 的 `request_goodput`；输出 token 吞吐作为辅助诊断列）。调优目标明确，keep/discard 判定机械可执行。
- **多层退出**：deadline 主退出 + 平台期 + 崩溃连击 + SLO 不可达退出，过夜运行自动收尾，不留僵尸进程。

## 如何使用（fork 指南）

1. Fork / clone 本仓库。
2. `cp target.example.yaml target.yaml`，填写主机、模型、SLO、预算。
3. `cp .env.example .env`，填写 SSH 选项（端口、私钥路径、远端目录）。
4. 在编排机（本仓库目录）运行 `bash setup_remote.sh`（一次性校验：docker 可用 + 镜像已就绪；零 pull、零安装）。
5. 把编码 agent（Claude Code / Codex / opencode）指向本仓库。
6. 对它说：「读 AGENTS.md，开始」——之后它自主循环、运行期自行处置环境波动，不再打断你；你睡觉。

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
| `bench.py` | 远端起服 + `vllm bench serve --goodput` 度量 + 僵尸清理 + 可选 `--detect-gmu` 预检 | 只读 |
| `report.py` | 收尾汇总 → report.md（基线 vs 最佳 + 全部 trial + 系统级建议） | 只读 |
| `run.sh` | 两段式执行：rsync + 远端 nohup 启动 + 轮询状态 | 只读 |
| `setup_remote.sh` | 主机环境校验（docker-only：校验 docker 可用 + 镜像就绪；零 pull、零安装） | 只读 |
| `scripts/mock_loop.py` | 循环规则的可执行规格/演示（REFERENCE，供离线核对 program.md 规则） | 只读 |
| `docs/vllm-facts.md` | 模板依赖的 vLLM 行为事实（版本相关，升级需重新核对） | 只读 |
| `docs/external-resources.md` | 外部 skills / MCP / 工具调研（vLLM 调优生态） | 只读 |
| `docs/roadmap.md` | 后续路线图（v1.1 证据/验收层、v2 架构扩展） | 只读 |
| `.agents/skills/` | 工程级 agent skills（vllm-bench-serve / vllm-deploy-docker / vllm-prefix-cache-bench，版本由 skills-lock.json 锁定） | 只读 |
| `skills-lock.json` | skills 来源与哈希锁定（防漂移） | 只读 |
| `pyproject.toml` / `LICENSE` / `.gitignore` | 工程元数据 | 只读 |

## Design choices

- **固定 workload 与参考并发**：所有 trial 用同一合成 workload 与参考并发点（`harness.yaml`），保证配置之间可比。
- **goodput 标量化**：以 `vllm bench serve --goodput` 的输出为唯一指标，SLO 来自 `target.yaml`（缺省用 `harness.yaml` 占位）。
- **多层退出**：deadline 主退出（硬保证）+ 平台期（连续 N 次无 > 噪声带的改进）+ 崩溃连击（连续 N 次 crash）+ SLO 不可达退出（校准后仍 goodput=0）。每轮开头做 deadline 感知调度，来不及就不开新 trial；在途 trial 宽限跑完。
- **系统参数只建议**：governor / hugepages / 锁频等系统级参数只探测并写入报告建议，不自动修改；v1 不提供自动修改开关。
- **NUMA 感知的 tp 选择**：`probe.py` 产出 `numa_groups`（每 GPU 的 `numa_node`）与「NUMA 放置」检查清单建议——tp 组限同一 NUMA 节点内的 GPU；**主机级**绑定（`numactl`）只建议、不自动；**容器级** `--cpuset-cpus/--cpuset-mems`（经 `runtime.extra_docker_args`）为 opt-in 逃生口，限定可见 GPU 用 `env.CUDA_VISIBLE_DEVICES`。
- **环境综合驱动**：搜索空间与剪枝由主机 / NUMA / 网络（IB/RoCE）/ 存储四类环境事实共同决定（依据 `env.json` / `env_report.md` / `model_profile.json`），不以单一指标做剪枝；**NUMA 划分 / GPU 子集 × 并行方案（placement）纳入评估**（主机级 BIOS 划分只建议）。
- **前缀缓存可测性**：`workload.prefix_hit_pct`（0-100，仅 random 数据集）按目标命中率拆共享前缀，让 `enable-prefix-caching` 成为可测维度；repeats 之间自动 `POST /reset_prefix_cache`，保证每次测量冷缓存独立。
- **多机采集 + 单机执行**：v1 只在 `role=head` 的主机上跑实验，其余主机仅做采集探测；跨机调度留到 v2。
- **运行时：docker-only**：vLLM 环境复杂，统一用镜像（如 `vllm/vllm-openai:v0.31.0`）——`config.yaml` 的 `runtime.image` 指定（逐键覆盖 `harness.yaml` 默认），**不支持**本机安装式 vLLM；`image` 本身是合法的搜索维度（如 CUDA 12/13 变体镜像）；模型目录经 `-v host:container:ro` 只读挂载进容器，serve/bench 都在容器内跑、本机无需安装 vllm。
- **运行期自主**：Phase 0 之后 agent 不再打断人类；SLO 校准、workload 降档、共租 / 失效处置均按 program.md 自主执行并留痕。
- **编排**：主循环串行 + 子 agent 只读认知分工（摘要即弃）；results.tsv + git 为唯一真源，子 agent 会话不承载状态；不并发 trial（并行实验见 docs/roadmap.md 暂缓）。

## 支持矩阵

- **已支持**：NVIDIA GPU + Mellanox（RoCE / IB）网络。
- **未支持**：AMD / ROCm（欢迎 fork 扩展）。

## 首次使用检查清单

> 模板交付前只在 mock 级验证过，真机首次使用务必按顺序冒烟：

1. 跑 `python probe.py`，检查生成的 `env_report.md`（系统参数清单 pass / warn / fail）。
2. 用初始 `config.yaml`（基线）跑 `bash run.sh` 并轮询 `bash run.sh --poll` 至退出 0，确认 bench 出数正常。
2.5 可选预检（共享主机防 OOM）：`bash run.sh --detect-gmu`，然后循环 `bash run.sh --poll` 至退出 0，结果回传本机 `gmu_probe.json`（`gmu_suggested` 为安全 `gpu-memory-utilization` 起点）。
3. 跑一次短预算循环，确认 keep / discard 判定与多层退出逻辑正确。
> 若 harness 不支持子 agent，主循环以内联摘要纪律等价运行（功能不缺失）。

> 注：全部脚本均已交付（清单 1/2/3 条均可执行）。离线排练：`uv run pytest -q` 无需 GPU/网络即可跑通 probe → bench → report 的 mock 全链路与循环规则核对。

## 主机零破坏契约

- 模板在每台主机上**只写** `target.yaml` 里用户确认的 `work_dir`（专属目录，Phase 0 必须与用户确认；`run.sh`/`setup_remote.sh` 的所有远端写入都局限于此）。

## 推荐 skills（工程级 `.agents/`）

仓库已内置 3 个工程级 agent skills（来源 `vllm-project/vllm-skills`，由 `skills-lock.json` 锁定版本）：

| skill | 用途 |
|---|---|
| `vllm-bench-serve` | `vllm bench serve` 基准运行（与模板度量一致） |
| `vllm-deploy-docker` | vLLM 的 docker 部署/参数核对（与 docker 运行时互补） |
| `vllm-prefix-cache-bench` | prefix-caching 专项压测（`enable-prefix-caching` 调优证据） |

添加更多：`npx skills add vllm-project/vllm-skills`（或 NVIDIA/skills 等，见 `docs/external-resources.md`）。注意 CLI 会为其他 harness 创建 gitignore 掉的软链接。

## 如何解读 report.md

- 「基线 vs 最佳」：goodput 提升 % =（最佳 − 基线）/ 基线；若标注「⚠ 无满足 SLO 的 trial」，则最佳为回退选择（全部已测量行中最高），需人工复核 SLO 是否过严。
- 若出现「运行期 SLO 校准」标注：说明参考 SLO 不可达，agent 已按实测 p99 校准出有效 SLO（概览标注「参考→有效」ttft/tpot）；best 为有效 SLO 下的最优，校准留痕见 `results.tsv.description`。
- 「全部 trial 表」：按 goodput 降序，SLO 达标列一眼看出哪些配置可用。
- 「最佳配置」：内嵌 `git show <best_commit>:config.yaml` 的快照（git 不可用时提示手动查看）。
- 「系统级建议」：来自 `env_report.md` 的 warn/fail 项，均为人工可执行建议（v1 不自动改）。

## 更多文档

- `docs/vllm-facts.md` — vLLM 行为事实（bench CLI / 输出字段 / 启动校准行）。
- `docs/external-resources.md` — 外部 skills / MCP / 工具调研（vLLM 调优生态）。
- `docs/roadmap.md` — 后续路线图（v1.1：监控证据层 / GuideLLM 验收 / nvbandwidth 拓扑实测 / lm-eval 精度 gate / vllm-tuner 大脑；v2：跨机执行 / PD 分离 / Dynamo AIConfigurator / 系统参数白名单）。

## 依赖

- 远端：NVIDIA 驱动 + docker 运行时 + 预先就绪的 vLLM 镜像（`runtime.image`，默认不 pull）。
- 远端运行 `bench.py` 的 python（`harness.yaml` 的 `remote.vllm_python`）须已装 **pyyaml**（零破坏不安装；系统 python3 通常自带）。
- 本机：Python 3.10+。

## License

MIT，见 [LICENSE](LICENSE)。
