# AGENTS.md — 编码 agent 入口

## 项目目的

本仓库是一个可 fork 的模板：由外部编码 agent 在真实 GPU 主机上（经 SSH）自主循环调优 vLLM 服务参数，目标是最大化 goodput（SLO 约束下的成功请求吞吐）。人类只负责填写 `target.yaml` 与启动预算；agent 负责探测环境、跑实验、按实测指标 keep / discard、并在 deadline 前自动收尾。

## 你的权限边界

- **只编辑 `config.yaml`**。它是一次实验的全部输入：`env`（进程级环境变量）、`model`（可选覆盖）、`serve`（vLLM serve 参数）。
- 除 `config.yaml` 外，本仓库所有文件均为只读（含 `README.md`、`docs/`、`scripts/`、`.env`、`pyproject.toml`、`target.example.yaml` 等）。
- **提问窗口仅限 Phase 0**：只有 Phase 0 发现 target.yaml 缺项（主机 / 模型 / SLO / 预算 / work_dir）时才向人类提问，不要编造。进入 Phase 1 后至 Phase 4 结束为**禁问窗口**：不再就运行期问题（SLO 口径 / workload / 预算剩余 / GPU 占用 / 主机或模型失效）打断人类——一切按 program.md 的自主处置规则决策，并把依据写入 results.tsv 与 report.md。

## 阶段流程

### Phase 0 — 信息采集
- **读**：`target.yaml`、`harness.yaml`。
- **跑**：无需命令（确认输入齐全即可）。
- **产出**：确认后的运行参数清单；缺项则向人类提问后继续。**必含**：主机/模型/SLO/预算，以及每个主机的 `work_dir`（模板在该主机上唯一允许写入的目录）——缺失时向用户询问确认，**不得自行假设或写入其他路径**。

### Phase 0.5 — 主机探索（只给主机时）
- **读**：`target.yaml`（只需 `hosts` 段；**无需 model / SLO / budget / work_dir**）。
- **派**：**「主机探索」子 agent**——Phase 0.5 的唯一执行者（只读远端）。主循环**不亲自读** `env.json`。
- **子 agent 跑**：`python probe.py`（host-only；SSH 选项取自 `.env`）→ `env.json` / `env_report.md` / `model_profile.json` / `search_space.json` / **`host_report.html`**。
- **子 agent 回传（≤15 行；原文不进主上下文）**：GPU 型号×数×显存 + 有无 NVLink（→ tp 建议）/ 网络与实测带宽 / 可用 vLLM 镜像（名 + size）/ **可服务模型清单（路径·大小·量化·能否装下）** / 当前占用（容器·监听端口·**端口冲突**·登录用户）/ 结论与阻塞项 + `host_report.html` 路径。
- **产出**：**`host_report.html`**（五段 + 结论），供人类「大概了解」该主机。
- **用途**：人类据此评估并**下发调优指令**（补 model / SLO / budget / work_dir），随后才进入 Phase 1。此阶段即 Phase 0 提问窗口的证据面。

### Phase 1 — 环境探测
- **读**：`target.yaml`（hosts / model）。
- **跑**：`python probe.py`（多机并行采集）。
- **产出**：`env.json`（原始采集）+ `env_report.md`（系统参数检查清单 pass / warn / fail）+ `model_profile.json`（权重规模估算）+ `search_space.json`（机器可读候选池：tp / placement / NCCL env / cheap knobs + `pruned` 剪枝清单，均带 reason）+ `host_report.html`（主机探索报告，人类可读）。
- **判断依据**：后续搜索空间与剪枝综合**主机 / NUMA / 网络 / 存储**四类环境事实（依据 `env.json` / `env_report.md` / `model_profile.json`），不是单一指标；**NUMA 划分与 GPU 子集 / 并行方案（placement）属于可评估维度**，主机级（BIOS）NUMA 划分只建议、不自动改。

### Phase 1.5 — 环境就绪门禁
- **读**：`env_report.md`。
- **跑**：无（先按 program.md「环境失效与共租处置」尝试自愈：更换可见 GPU、下调 gmu/tp）。
- **产出**：不影响测量有效性的系统级 fail 记 waived（依据写入 report.md）后继续；仍阻断且超出 agent 权限的，**停止循环并把诊断写入 report.md，不等待人类实时介入**；仅 warn → 继续 Phase 2。

### Phase 2 — 基线
- **读**：`config.yaml`（初始即为基线态）。
- **跑**：`bash run.sh` 启动基线 trial，轮询 `bash run.sh --poll` 直至退出 0（done）/ 1（crash）。
- **产出**：基线行的 goodput / 时延指标写入 `results.tsv`；用 vLLM 启动日志中的实测值校准 `model_profile.json`——写入 `boot.measured_seconds` 与 `kv.measured_capacity_tokens`（实测），并置 `calibrated: true`。
- 基线出数后立即判定 SLO 可行性：`goodput>0` → 继续；`goodput=0` → 先执行 program.md「SLO 可行性与自主放宽」的一次性校准，再进 Phase 3。

### Phase 3 — 调优循环
- **读**：`program.md`（循环细则与退出机制）、`env.json` / `model_profile.json`（剪枝依据）、`results.tsv`（历史与当前最佳）（含 program.md 的「环境失效与共租处置」「子 agent 分派」「循环调度与断点续跑」）。
- **跑**：每轮「改 `config.yaml` → commit → `bash run.sh` 启动（秒回）→ 轮询 `bash run.sh --poll`（退出码 3=仍在运行，0=done，1=crash）→ 记录结果 → keep / discard」。
- **产出**：更新 `results.tsv`；每轮一个 config.yaml 的 git commit。

### Phase 4 — 收尾
- **读**：`results.tsv`、`env_report.md`。
- **跑**：`python report.py`。
- **产出**：`report.md`——基线 vs 最佳对比（含 goodput 提升 %）、全部 trial 表（SLO 达标标注）、最佳配置（config.yaml 快照）、系统级建议项（不自动改）、适用范围（v1 单节点结论，跨机不外推）；运行期校准过 SLO 时报告出「运行期 SLO 校准」标注，自主处置留痕在 `results.tsv.description`。

## 结果与日志约定

- `results.tsv` 与 `run.log` 为生成物，已在 `.gitignore`，**不入库**。
- 每个实验在 git 历史里留 `config.yaml` 的线性 commit，保证「每个配置都可回溯」；discard 不回退提交，只把工作区恢复到当前最佳（`git checkout <best_commit> -- config.yaml`）。
- 重启后可断点续跑：先读 `results.tsv` 与 git 分支状态，再决定从哪个实验继续。

## 远程执行约定

- `run.sh` 为两段式启动/轮询：`bash run.sh` 先 rsync 同步 harness 到远端，再远端 `nohup` 启动并秒回；随后循环 `bash run.sh --poll` 直到退出 0（done）或 1（crash）；退出码 3 = 仍在运行，继续轮询。
- `run.sh --detect-gmu`：可选 gmu 预检（Phase 1/2 之间，共享主机防 OOM）——与启动 trial 同机制（rsync → 远端 nohup），远端跑 `bench.py --detect-gmu`；随后同样循环 `bash run.sh --poll`，结果回传本机 `gmu_probe.json`（`gmu_suggested` 建议写进 `config.yaml` 的 `serve.gpu-memory-utilization`）。
- 不假定 head 主机独占：起服前后按 program.md「环境失效与共租处置」检测外部占用并自主应对（缩 GPU 子集 / 有限重探 / 带告警继续）；bench.py 负责清理僵尸 vllm 进程。
- 运行时为 **docker-only**：镜像由 `config.yaml` 的 `runtime.image` 指定（逐键覆盖 `harness.yaml` 默认，kind 键如存在则忽略）；serve/bench 都在容器内跑、模型路径经只读挂载注入，本机无需安装 vllm。
- **零破坏**：所有远端操作只写 `target.yaml` 确认的 `work_dir`；默认不 pull 镜像、不安装环境。

## 循环细则

调优循环的决策规则、搜索空间剪枝与退出机制，见 **[program.md](program.md)**。

工程级 skills（`host-explore` / `vllm-bench-serve` / `vllm-deploy-docker` / `vllm-prefix-cache-bench`）在 `.agents/skills/`；外部调研与后续路线图见 `docs/external-resources.md` / `docs/roadmap.md`。

## 编排与子 agent 约定

主循环是唯一的写者与唯一的串行执行者；子 agent 只做只读认知分工，产出「摘要即弃」，不承载状态（**唯一例外**：Phase 0.5「主机探索」子 agent，见下）。

- **分工而非并发**：主循环串行执行调优；子 agent 只承担探索 / 分析 / 验证 / 评估，**绝不起 trial**（单 GPU、单容器名、单端口；并行实验见 docs/roadmap.md「暂缓」）。
- **角色边界**：主机探索（Phase 0.5：只读远端跑 `probe.py` → ≤15 行摘要 + `host_report.html` 路径，**原文不回传**）；探索（只读源码/文档/外部知识 → 有界摘要 + 引用位置）；分析（解析 run.log/result.json/env.json/results.tsv → 根因/趋势，附 commit 或「文件:行」证据）；验证（对将被采纳的结论做对抗性复核，独立于产生结论者）；评估（按 program.md 决策规则打分，优先直接调用 scripts/mock_loop.py）。
- **唯一写者原则**：只有主循环写 config.yaml 与生成物（results.tsv/report.md/git commit）；子 agent 只读 repo 与远端，经返回值回传，不写任何文件。**窄例外**：Phase 0.5「主机探索」子 agent 可写其探测生成物（`env.json` / `env_report.md` / `model_profile.json` / `search_space.json` / `host_report.html`），但**绝不写** `config.yaml` / `target.yaml` / `results.tsv` / report.md / git。
- **并行与顺序**：只读且不占 GPU 的子 agent 可并行，上限 ≤3；任何触发 run.sh 的动作必须串行。
- **应派**：Phase 0.5 主机探索（跑 probe + 归纳 `env.json`，主循环零接触原文）；跨文件 / 跨知识检索；结论将被 keep / 采纳时的对抗性复核。
- **不该派**：单文件小读取；结果立刻要用且便宜；会重复加载模型或重跑 trial（单 trial 成本高，通常远超一次只读检索）。harness 不支持子 agent 时，在主循环内联同等的摘要纪律（优雅降级）。
- **会话回收**：常驻 explorer（环境/事实，跨 trial 复用）+ 一次性 analyst（每 trial）；回传 ≤N 行摘要 + 指针，原文不进主上下文。

## 循环调度与上下文预算

- **门禁与检查者**：Phase 0.5 主机探索子 agent 回传摘要 + `host_report.html` 落盘；Phase 0→1 自检（target 必含 host/model/SLO/budget/work_dir）；1→1.5 自检 + 探索子 agent 核验；1.5→2 自检（waived 需留痕）；2→3 自检（基线出数 + SLO 可行性判定完成）；3→4 验证子 agent 复核 best 回归（无子 agent 时主循环二次独立计算）。
- **任务板 = 派生视图，不新增 authored 文件**：任务板 = results.tsv 末行 + git 分支 HEAD + budget 剩余 + 在途 trial 状态（bench.pid/status.json）；意图性备注写进 results.tsv.description 与 commit message。
- **后台任务**：run.sh 起 serve/bench 即后台任务，同一时刻至多一个；生命周期 start → poll（3=运行中 / 0=done / 1=crash）；crash 按 program.md 处置。
- **断点续跑**：状态必须在 results.tsv + git，不得依赖会话记忆；重启后先重建任务板视图再继续。
- **上下文预算**：主上下文只保留任务板视图 + 当前 best + 最近一行 + 未决项；run.log/大输出一律经子 agent 摘要后入；接近上限时把摘要固化进 results.tsv.description 与 report 草稿。

## Cloned Dependency Source

`.slim/clonedeps/repos/` 下有只读依赖源码可供查阅，请勿编辑这些克隆。

- `.slim/clonedeps/repos/vllm-project__vllm/` — `vllm-project/vllm` @ `v0.31.0`（sparse 仅 `benchmarks/auto_tune`）；官方 vLLM serve 参数自动调优脚本（网格 + `--goodput` SLO 搜索），是本模板循环的同构先例，用于借鉴其 gmu 自动探测、`--load-format dummy` 快速扫描、`--random-prefix-len` 前缀缓存工作负载等做法。
- `.slim/clonedeps/repos/NVIDIA__nccl/` — `NVIDIA/nccl` @ `v2.32.3-1`；NCCL 源码，用于核实 `NCCL_*` 环境变量语义与多 GPU 通信调优项，作为 `search_space.json` 中 NCCL env 候选的依据。
- `.slim/clonedeps/repos/vllm-project__guidellm/` — `vllm-project/guidellm` @ `v0.8.1`；第二基准/验收层（roadmap v1.1(b)）的源码依据：`sweep` profile 格式与 `--constraint max_error_rate` 语义。

结构化清单见 `.slim/clonedeps.json`（无需为基本仓库列表阅读它）。
