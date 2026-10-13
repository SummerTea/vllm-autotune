# AGENTS.md — 编码 agent 入口

## 项目目的

本仓库是一个可 fork 的模板：由外部编码 agent 在真实 GPU 主机上（经 SSH）自主循环调优 vLLM 服务参数，目标是最大化 goodput（SLO 约束下的成功请求吞吐）。人类只负责填写 `target.yaml` 与启动预算；agent 负责探测环境、跑实验、按实测指标 keep / discard、并在 deadline 前自动收尾。

## 你的权限边界

- **只编辑 `config.yaml`**。它是一次实验的全部输入：`env`（进程级环境变量）、`model`（可选覆盖）、`serve`（vLLM serve 参数）。
- 除 `config.yaml` 外，本仓库所有文件均为只读（含 `README.md`、`docs/`、`scripts/`、`.env`、`pyproject.toml`、`target.example.yaml` 等）。
- 背景信息缺项时（如 target.yaml 缺主机 / 模型 / SLO / 预算）**向人类提问，不要编造**。

## 四阶段流程

### Phase 0 — 信息采集
- **读**：`target.yaml`、`harness.yaml`。
- **跑**：无需命令（确认输入齐全即可）。
- **产出**：确认后的运行参数清单；缺项则向人类提问后继续。**必含**：主机/模型/SLO/预算，以及每个主机的 `work_dir`（模板在该主机上唯一允许写入的目录）——缺失时向用户询问确认，**不得自行假设或写入其他路径**。

### Phase 1 — 环境探测
- **读**：`target.yaml`（hosts / model）。
- **跑**：`python probe.py`（多机并行采集）。
- **产出**：`env.json`（原始采集）+ `env_report.md`（系统参数检查清单 pass / warn / fail）+ `model_profile.json`（权重规模估算）。

### Phase 1.5 — 环境就绪门禁
- **读**：`env_report.md`。
- **跑**：无（把 fail 项的修复命令交给人类执行，等修好再继续）。
- **产出**：门禁结论——有 fail 项 → 交人类修复后重跑 Phase 1；仅 warn → 可继续 Phase 2。

### Phase 2 — 基线
- **读**：`config.yaml`（初始即为基线态）。
- **跑**：`bash run.sh` 启动基线 trial，轮询 `bash run.sh --poll` 直至退出 0（done）/ 1（crash）。
- **产出**：基线行的 goodput / 时延指标写入 `results.tsv`；用 vLLM 启动日志中的实测值校准 `model_profile.json`——写入 `boot.measured_seconds` 与 `kv.measured_capacity_tokens`（实测），并置 `calibrated: true`。

### Phase 3 — 调优循环
- **读**：`program.md`（循环细则与退出机制）、`env.json` / `model_profile.json`（剪枝依据）、`results.tsv`（历史与当前最佳）。
- **跑**：每轮「改 `config.yaml` → commit → `bash run.sh` 启动（秒回）→ 轮询 `bash run.sh --poll`（退出码 3=仍在运行，0=done，1=crash）→ 记录结果 → keep / discard」。
- **产出**：更新 `results.tsv`；每轮一个 config.yaml 的 git commit。

### Phase 4 — 收尾
- **读**：`results.tsv`、`env_report.md`。
- **跑**：`python report.py`。
- **产出**：`report.md`——基线 vs 最佳对比（含 goodput 提升 %）、全部 trial 表（SLO 达标标注）、最佳配置（config.yaml 快照）、系统级建议项（不自动改）、适用范围（v1 单节点结论，跨机不外推）。

## 结果与日志约定

- `results.tsv` 与 `run.log` 为生成物，已在 `.gitignore`，**不入库**。
- 每个实验在 git 历史里留 `config.yaml` 的线性 commit，保证「每个配置都可回溯」；discard 不回退提交，只把工作区恢复到当前最佳（`git checkout <best_commit> -- config.yaml`）。
- 重启后可断点续跑：先读 `results.tsv` 与 git 分支状态，再决定从哪个实验继续。

## 远程执行约定

- `run.sh` 为两段式启动/轮询：`bash run.sh` 先 rsync 同步 harness 到远端，再远端 `nohup` 启动并秒回；随后循环 `bash run.sh --poll` 直到退出 0（done）或 1（crash）；退出码 3 = 仍在运行，继续轮询。
- 假定 head 主机由本次实验独占；`bench.py` 负责清理僵尸 vllm 进程。
- 运行时为 **docker-only**：镜像由 `config.yaml` 的 `runtime.image` 指定（逐键覆盖 `harness.yaml` 默认，kind 键如存在则忽略）；serve/bench 都在容器内跑、模型路径经只读挂载注入，本机无需安装 vllm。
- **零破坏**：所有远端操作只写 `target.yaml` 确认的 `work_dir`；默认不 pull 镜像、不安装环境。

## 循环细则

调优循环的决策规则、搜索空间剪枝与三层退出机制，见 **[program.md](program.md)**。

工程级 skills（`vllm-bench-serve` / `vllm-deploy-docker` / `vllm-prefix-cache-bench`）在 `.agents/skills/`；外部调研与后续路线图见 `docs/external-resources.md` / `docs/roadmap.md`。

## Cloned Dependency Source

`.slim/clonedeps/repos/` 下有只读依赖源码可供查阅，请勿编辑这些克隆。

- `.slim/clonedeps/repos/vllm-project__vllm/` — `vllm-project/vllm` @ `v0.31.0`（sparse 仅 `benchmarks/auto_tune`）；官方 vLLM serve 参数自动调优脚本（网格 + `--goodput` SLO 搜索），是本模板循环的同构先例，用于借鉴其 gmu 自动探测、`--load-format dummy` 快速扫描、`--random-prefix-len` 前缀缓存工作负载等做法。

结构化清单见 `.slim/clonedeps.json`（无需为基本仓库列表阅读它）。
