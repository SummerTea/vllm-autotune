# AGENTS.md — 编码 agent 入口

## 项目目的

本仓库是一个可 fork 的模板：由外部编码 agent 在真实 GPU 主机上（经 SSH）自主循环调优 vLLM 服务参数，目标是最大化 goodput（SLO 约束下的成功请求吞吐）。人类只负责填写 `target.yaml` 与启动预算；agent 负责探测环境、跑实验、按实测指标 keep / discard、并在 deadline 前自动收尾。

## 你的权限边界

- **只编辑 `config.yaml`**。它是一次实验的全部输入：`env`（进程级环境变量）、`model`（可选覆盖）、`serve`（vLLM serve 参数）。
- 除 `config.yaml` 外，本仓库所有文件均为只读（含 `README.md`、`docs/`、`.env`、`pyproject.toml`、`target.example.yaml` 等）。
- 背景信息缺项时（如 target.yaml 缺主机 / 模型 / SLO / 预算）**向人类提问，不要编造**。

## 四阶段流程

### Phase 0 — 信息采集
- **读**：`target.yaml`、`harness.yaml`。
- **跑**：无需命令（确认输入齐全即可）。
- **产出**：确认后的运行参数清单；缺项则向人类提问后继续。

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
- **跑**：`bash run.sh` 一次。
- **产出**：基线行的 goodput / 时延指标写入 `results.tsv`；用 vLLM 启动日志中的实测值（如权重加载耗时、显存占用）校准 `model_profile.json`。

### Phase 3 — 调优循环
- **读**：`program.md`（循环细则与退出机制）、`env.json` / `model_profile.json`（剪枝依据）、`results.tsv`（历史与当前最佳）。
- **跑**：每轮「改 `config.yaml` → commit → `bash run.sh` → 记录结果 → keep / discard」。
- **产出**：更新 `results.tsv`；每轮一个 config.yaml 的 git commit。

### Phase 4 — 收尾
- **读**：`results.tsv`、`env_report.md`。
- **跑**：`python report.py`。
- **产出**：`report.md`——推荐配置 vs 基线对比、SLO 是否达标、系统级建议项（不自动改）。

## 结果与日志约定

- `results.tsv` 与 `run.log` 为生成物，已在 `.gitignore`，**不入库**。
- 每个实验在 git 历史里留 `config.yaml` 的线性 commit，保证「每个配置都可回溯」；discard 不回退提交，只把工作区恢复到当前最佳（`git checkout <best_commit> -- config.yaml`）。
- 重启后可断点续跑：先读 `results.tsv` 与 git 分支状态，再决定从哪个实验继续。

## 远程执行约定

- `run.sh` 为两段式：先 rsync 同步 harness 到远端，再远端 `nohup` 启动 + 本机轮询状态。
- 假定 head 主机由本次实验独占；`bench.py` 负责清理僵尸 vllm 进程。

## 循环细则

调优循环的决策规则、搜索空间剪枝与三层退出机制，见 **[program.md](program.md)**。
