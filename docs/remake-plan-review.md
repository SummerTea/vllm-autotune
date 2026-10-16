# 对「GPT 5-Sprint 改造计划」的评估

> 评估对象：会话中 GPT 提出的「立即落地改造计划」（Loop Controller / Workload-Concurrency Sweep / Executable Search Space / GuideLLM 验收 / 硬件与质量证据）。
> 方法：对计划的事实前提做 `file:line` 核验（3 个只读 lane）+ 外部最佳实践调研 + 一轮对抗性复核（@oracle）。
> 结论中的成本数据取自某次真机演练的账本快照（`results.tsv`——本地生成物，不入库）；未运行真实 GPU trial。

---

## 结论速览

| 维度 | 判定 | 一句话 |
|---|---|---|
| **方向** | ✅ 正确 | 先补可靠性、再补搜索能力、最后补证据，顺序合理；保留 docker-only/零-pull/单 trial/账本契约正确 |
| **事实前提** | ⚠️ 3 处需修正 | `ref_concurrency` 优先级被误述；`budget` 是「已声明未执行」而非缺失；`config.workload` 键当前**不存在**（静默 no-op） |
| **鲁棒性** | ⚠️ 有新增风险 | 引入 `loop_state.json` 第二真源、controller 新失败面、多并发成本未量化 |
| **agent 自主规划** | ❌ 计划如写会**收紧**自主 | controller 若校验/裁决候选 + 硬编码 `concurrency_points` → 策略回落到脚本，与目标相反 |
| **可执行性** | ⚠️ 有条件可行 | 4 处前提修正 + 1 处边界定义后可落地 |

**总判**：计划**方向对、细节需要修正**。它不是「死板」，但其 Sprint 1/2 的写法会**把策略权从 agent 移回脚本**，这恰是用户最担心的问题。真正的缺口不是「agent 记不住账」，而是「**agent 与 controller 的权责边界从未定义**」。

---

## 一、被评估计划（5 Sprint）

1. **Loop Controller**：新增 `scripts/loop_controller.py`——候选校验、自动记账、crash 恢复、预算/deadline **强制**、重启恢复、已测配置去重；新增 `loop_state.json`。
2. **Workload/Concurrency Sweep**：`config.workload` 可覆盖（含 `num_prompts` 降档）；多并发点 `concurrency_points=[16,64,128,256]`；记录 error_rate/attainment；产出 sustainable best。
3. **Executable Search Space**：`search_space.json` v2（cheap-knob 带值、placement 带 PP/DP、候选带 cost/risk/priority）。
4. **GuideLLM 验收层**：最佳配置用 GuideLLM 二次验收。
5. **硬件与质量证据**：GPU/NCCL 证据 + 质量 gate。

---

## 二、A 段：事实前提错误（material，须先改）

### A1. `ref_concurrency` 优先级被误述

- 计划把「agent 改不了并发/负载」当作既定事实。
- 实测代码：`bench.py:1028` `ref_conc = workload.get("ref_concurrency") or bench_cfg.get("ref_concurrency", DEFAULT)`，而 `workload = harness.workload ⊕ target.workload`（`bench.py:1026`）、`bench_cfg = harness.bench ⊕ config.bench`（`bench.py:1004`）。
- 因此优先级为：**`target.workload` > `harness.workload` > `config.bench` > `harness.bench` > 默认**（注意 `config.bench` **覆盖** `harness.bench`，见 `bench.py:1004` `bench_cfg = {**harness.bench, **config.bench}`）。当前若用户在 `target.yaml`（本地文件，不入库）声明了 `ref_concurrency`，会**短路** `config.bench.ref_concurrency`。
- **含义**：今天 agent 经 `config.yaml` 确实改不了参考并发点。但这里是**文档承诺 vs 代码禁用**的张力：`program.md:42` 明文把 `config.bench.ref_concurrency` 列为覆盖 `harness.bench` 的通道，而 `bench.py:1028` 的优先级让 `target.workload` 把它短路——既非「缺少通道」，也非纯粹设计意图，而是**两处语义未对齐**。
- **对计划的修正**：Sprint 2 引入 `config.workload` 时，必须**明确定义覆盖优先级**并对齐 `program.md:42`，同时保持「负载身份不可变」（`program.md:155`）约束——否则会把一处刻意的设计误当 bug 去「修」。

### A2. `budget` 不是「缺失」，而是「已声明、未执行」

- 计划称「预算检查下沉到代码」，暗示现状没有预算。
- 实测：`target.yaml`（本地文件，不入库）已声明 `budget`（`max_hours` 优先 / `stop_at`）；`program.md:82` 定义了语义；但**没有任何代码读取它**（`scripts/mock_loop.py:118-126` 的 `should_exit` 是纯函数且**未接线** `run.sh`）。
- **含义**：controller 不是「接一根现成的线」，而是**首次引入预算执行语义**——改造面比计划宣称的大，且它将成为唯一读取 budget 的执行者。
- **对计划的修正**：把「下沉预算」重述为「**新增预算执行语义**」，并明确 budget 的真源仍是 `target.yaml`（人类侧），controller 只读不写。

### A3. `config.workload` 键当前**不存在** → 静默 no-op

- 全仓 `.py` 无 `config.get("workload")`（已 grep 确认空）。
- **风险**：agent 今天若在 `config.yaml` 写 `workload:`，会被**静默忽略**——与「未知 flag 起服前 fail-fast」（`bench.py:1082-1085`）不同，`config.yaml` 的**顶层未知键没有校验**。
- **对计划的修正**：Sprint 2 增加 `config.workload` 的同时，必须加**顶层键 fail-loud 校验**（只允许 `env/model/serve/runtime/bench/slo/workload`），否则新增通道会放大静默拼写错误。

---

## 三、B 段：自主性自相矛盾（最关键）

### B1. controller 与「唯一写者」契约冲突，且边界未定义

- 现契约：`AGENTS.md:73`「只有主循环写 config.yaml 与生成物」；`README.md:65`「主循环串行 + results.tsv + git 为唯一真源」。
- 计划的 controller 承担**自动记账（写生成物）**与**执行 keep/discard**，这已与「只有主循环写生成物」（`AGENTS.md:73`）冲突；计划文本**未明说**它是否也写 `config.yaml`/commit——若也写，则进一步与「agent 是主循环」冲突；若不写，则它必须把决策回传 agent，本身即需定义接口。
- 更实质的问题：计划说「controller 校验候选」却**未定义它可否拒绝候选**。若它按规则拒绝，**策略权就从 agent 转移到脚本**——这正是用户担心的「脚本太死板」。
- **必须显式定义权责边界**：
  - **controller 只负责「安全 / 执行 / 状态 / 证据」**：语法与 flag 校验、资源可行性（显存/并行度）、预算与 deadline、trial 身份与记账、crash 自愈、状态恢复。**不得**决定「试哪个维度、试哪个值、何时停」——那是 agent 的策略权。
  - **agent 负责「策略」**：选维度、选候选、定顺序、判断是否换方向。
  - controller 的「拒绝」只允许是**安全拒绝**（非法/不可执行/超预算），且必须回传结构化理由供 agent 重新规划，而不是替 agent 决定。

### B2. 硬编码 `concurrency_points` 是脚本化策略

- 计划把 `[16,64,128,256]` 写进配置 → 这是**固定的扫描策略**，与「释放 agent 自主规划」相反。
- **修正**：把并发点作为 **agent 可选的维度**（controller 只提供成本估算与可行性），默认档位可复用但应可由 agent 覆盖；或至少在 `program.md` 明确「这是初始档位、非强制」。
- **同类问题**：「选哪个并发点作为 sustainable best」本身也是**策略判断**（不同业务负载取不同点），应由 agent 决策、controller 只提供各点指标与成本，不应固化为 report 的默认规则。

---

## 四、C 段：鲁棒性缺口

### C1. `loop_state.json` 是第二真源

- 契约真源是 `results.tsv` + git（`README.md:65`）。新增 `loop_state.json` 若无约束，会与账本分歧——正是本仓已经打过的那类「陈旧产物污染」bug。
- **修正**：`loop_state.json` 必须**严格派生/可重建**（由 `results.tsv` + git HEAD + `.trial_id` + 远端 status.json 重算），**永不作为权威真源**；写入即可被删除并由重算复原。计划需把这条写成硬约束 + 恢复测试。

### C2. 多并发 sweep 成本未量化（实测反证）

基于某次真机演练账本（`results.tsv`）的实测数据（**示例，非模板保证值**）：

| 观测 | 数值 |
|---|---|
| 基线 trial `dur_s` / load | 764s / 446.8s |
| 确认 trial `dur_s` / load | 637.3s / 319.6s |
| 单次 bench 均摊（`(dur−load)/4`，warmup1+repeats3） | ≈ 79s @ `num_prompts=400`, `concurrency=128` |
| 单 trial 总耗时区间 | 636–937s（≈10.6–15.6min） |
| 2h 预算可容纳 trial 数 | ≈ 8–11 |

若 Sprint 2 采纳 4 点 sweep（复用同一已加载 server，以下为**乐观下界**，假定各点等时）：
`4 点 × 4 次 × 79s ≈ 1264s` + load 320–450s → **≈26–29min / trial → 2h ≈ 4–4.5 个 trial**（7200/1714≈4.2、7200/1584≈4.5）。
低并发点并行度低，单点墙钟通常**更慢**（400 prompts 在 concurrency=16 时的批数约为 128 时的 8 倍），故实际预算更紧。

**含义**：多并发 sweep 会把搜索覆盖从现状 ~8–11 个配置（636–937s/trial）压到 **~4–4.5 个**，**削弱「逼近最佳实践」的能力**。
**修正**：计划必须（i）复用同一 serve 实例跨点测量；（ii）对低并发点下调 `num_prompts`（并明确其**改变 `workload_hash` → 换上下文**，见 C5）；（iii）加**成本闸门**：预估 sweep 耗时 > 剩余预算某比例时自动降级为单点。

### C3. GuideLLM 与「零-pull/零-装」契约冲突未解

- 契约：docker-only、`--pull=never`、不装环境（`AGENTS.md:59`；`bench.py:322` 强制 `--pull=never`）。
- 实测：guidellm **不在**运行时（仅见 `docs/roadmap.md`、`docs/external-resources.md`、`README.md:114`、`AGENTS.md:92`、只读 clonedeps）。
- 外部调研：GuideLLM 可经官方 `Containerfile.vllm`（以 vllm 镜像为 base）**构建期装入**，运行期不联网；但**无官方文档**保证出厂 vllm 镜像内零安装可用。
- **更优解**：官方镜像内**已有** `vllm bench sweep serve_workload`（官方称 `max_concurrency` 比 `request_rate` 更可靠），可承载扫描/验收语义而**不引入新镜像**。
- **修正（条件式）**：Sprint 4 应写成——「**若冒烟确认被 pin 的 `vllm/vllm-openai:v0.31.0` 内含 `vllm bench sweep`，则优先用它**承载扫描/验收；**否则兜底**：在 `bench.py` 内自循环并发点（复用同一已加载 server），不引入新镜像。GuideLLM 仅作可选外部验收（需人类预置镜像）。」⚠ 该前提**未核验**，故**不可无条件「优先」**。

### C4. 解析器仅 4 字段，error_rate/attainment 是净新增

- `parse_bench_stdout`（`bench.py:127-136`）与 `parse_bench_json`（`bench.py:143-150`）各只取 4 个字段；`result.json`（`bench.py:1180-1207`）**无** failed/error_rate/attainment/over-saturation。
- 原始 JSON **含** `completed` / `failed`（`docs/vllm-facts.md:45`）→ error_rate 技术上可得，但**必须扩解析器**，并把 crash/timeout 一并计入失败统计口径。

### C5. 上下文切换语义必须与 `num_prompts` 一起改

- `workload_hash` 含 `num_prompts`（`record.py:59-65` ← `bench.py:222-228`）→ 降档即换上下文。
- 现实现是**设计好的代价**：降档需「重测基线」（`program.md:153-155`），best 只在基线上下文内选（`report.py:571-581`）。
- 计划若想在 `concurrency_points` / `num_prompts` 变化后仍跨上下文比较，**必须同改** `record.py` + `report.py`，否则 best 选择会在切换点**静默断裂**。

### C6. SLO 校准触发变更需同步多处

- 计划提出把触发条件从「任一 trial goodput==0」（`program.md:30`）改为「基线或多个独立候选」。方向正确，但「至多一次」目前**仅文档约束、代码不强制**（`bench.py`/`report.py` 无强制）。
- 变更须同步 `program.md` + `report.py:549-550` 渲染 + `scripts/mock_loop.py` 同口径 + 测试，否则出现「手册与实现不一致」。

### C7. 「已测去重」的键不含 workload → 会误跳合法重测

- 计划 Sprint 1 含「已测配置去重」。但 `config_fingerprint = sha256(config.yaml 内容)`（`bench.py:87-90`）**不含 workload**（当前 `config.workload` 不存在；也不含 `target.workload`）。
- 一旦 Sprint 2 让并发点 / `num_prompts` 成为可变维度，仅按 `config_hash` 去重会把「同 config、不同 workload」的合法重测误判为「已测」而**跳过**。
- **修正**：去重键必须扩为 `(config_hash, workload_hash)`（或再加 `effective_slo`）。

### C8. 测量语义：`--request-rate inf` 是满饱和测量

- `bench.py:428` 固定 `--request-rate inf` + `--max-concurrency`，即**把并发打满**。「sustainable best」因此是**满饱和点**的产出，未必映射生产负载；判断「可持续」须靠 attainment / over-saturation（当前 result.json 无这些字段）。
- 计划的 Sprint 2 若只加 `concurrency_points` 而不加达标率/过饱和判据，得到的仍是「压满时的好配置」。

### C9. `num_prompts` 降档会放大 P99 噪声

- `bench.py:427` 的 `--num-prompts` 降档省钱，但 P99 由尾部样本估计——400 请求时 P99 仅约由 4 个样本决定，降档后更少，直接放大 SLO 判定噪声。
- 计划把「降档」当纯成本手段，未评估其对判定可信度的影响。

---

## 五、D 段：计划做对的部分

1. **诊断准确**：确实没有 controller、没有预算执行、单负载点、`search_space` 只有名字没有值、contention 不进 best（`report.py:158/165-166` 无 contention 过滤，实测确认）、环境指纹不进上下文（`report.py:117-123`）。
2. **顺序合理**：先可靠性、再能力、后证据。
3. **契约保留正确**：docker-only、零-pull、单 trial、`results.tsv`+git。
4. **自建 controller 的判断合理**：外部调研确认现成工具（vllm-tuner/GuideLLM/AIConfigurator/Red Hat Auto-Tune）都假设自由装环境，**受约束 controller 无同构现成品 → 非重复造轮子**。
5. **正确把 `vllm bench sweep`/GuideLLM 的 attainment/CI/over-saturation 视为目标语义**。

## 六、E 段：计划遗漏的部分

1. **agent ↔ controller 权责边界**（最重要，见 B1）——不定义则必然退化为「脚本裁决策略」。
2. **已有 run 的迁移**：旧格式账本（列集与新表头不一致时）与新 17 列头的兼容、在途远端 trial 的接管——计划只把它列为风险，未列为交付项。
3. **controller 状态机的测试策略**：计划只写「现有测试必须通过」，未要求状态机专项测试（crash 中断/半途 commit/重复入账/预算边界）。
4. **顶层未知 config 键的 fail-loud**（见 A3）。
5. **成本闸门**（见 C2）。
6. **`vllm bench sweep` 的候选资格**（见 C3）。
7. **并发点：策略 vs 维度**（见 B2）。
8. **去重键语义**：`config_hash` 不含 workload → 须扩为 `(config_hash, workload_hash)`（见 C7）。
9. **满饱和测量语义**：`--request-rate inf` 与 sustainable best 的关系、以及 `num_prompts` 降档对 P99 噪声的影响（见 C8/C9）。
10. **更简替代**：优先接线 `scripts/mock_loop.py` 纯函数，而非直接新建有状态 controller（见 Q2）。

---

## 七、直接回答用户的三个问题

### 1. 这个改造设计是否有问题？

**有，但方向对。** 关键问题不是「脚本太死板」，而是**计划用一个新的 controller 把策略判断搬出 agent**（B1/B2），并把几处「设计好的机制」误当缺陷去修（A1/A2）。此外有 3 处事实前提错误、2 处未量化的成本/契约冲突（C2/C3）、1 处第二真源风险（C1）。

### 2. 是否具备鲁棒性？

**计划意图上追求鲁棒，实现上引入新风险。** 新增的 `loop_state.json`、controller 状态机、多并发成本、未解的 GuideLLM 安装都是新失败面。按本文修正（状态严格派生、fail-loud、成本闸门、复用 serve 实例、优先镜像内 sweep）后可达鲁棒；按现状直接实施会引入新的不一致风险。

### 3. 是否具备 agent 自主规划能力？

**按计划原样实施会削弱自主规划能力。** controller 越权校验候选 + 硬编码 `concurrency_points` = 把策略写回脚本。要真正提升自主性，必须确立：
- **controller = 安全/执行/状态/证据**（只做安全拒绝，回传理由）；
- **agent = 策略**（维度、候选、顺序、停止/换向）；
- 所有约束以**结构化、可读、可覆盖**的形式暴露给 agent，而非写死。

**边界需再细分**：「预算/deadline 强制」与「crash 重试/降档」本身含策略成分——controller 应只执行**已由 agent 声明或规则表固定**的动作，并把「重试/降档/换向」作为选项回传 agent，而非自行决定。

**更简的替代**：仓库已有 `scripts/mock_loop.py` 的纯函数（`decide()` `:50-78`、`should_exit()` `:118-131` 含 deadline 感知三层退出、`LoopState` `:81-115`），而 `run.sh` 全文**未引用**它们。补「预算/退出/决策」缺口的**最小解**可能是**把这些纯函数接线进主循环**，而非新建有状态的 `loop_controller.py`——第二真源、状态机 bug 这些新失败面**只在有状态方案里才存在**。建议先评估接线方案，再决定是否升级为 controller。

---

## 八、修正后的最小路线（bounded）

在原 5 Sprint 上做**最小修正**（不推翻）：

| Sprint | 原计划 | 修正 |
|---|---|---|
| 1 | Loop Controller | 显式定义**权责边界**（controller 只安全拒绝）；`loop_state.json` **严格派生可重建**、非权威真源；迁移旧 `results.tsv`+在途 trial 列为交付项；**状态机专项测试** |
| 2 | Workload/Concurrency Sweep | 定义 `config.workload` **覆盖优先级**并保持负载身份不可变；加**顶层键 fail-loud**；并发点作为**agent 维度**（非硬编码）；**复用 serve 实例跨点**；**成本闸门**；同步上下文切换语义 |
| 3 | Search Space v2 | 保持；补「候选 cost/risk」用于 agent 排序而非脚本裁决 |
| 4 | GuideLLM 验收 | 改为**条件式**：冒烟确认镜像内 `vllm bench sweep` 存在则优先用它；否则在 `bench.py` 内自循环并发点（复用已加载 server）。GuideLLM 仅可选（人类预置镜像） |
| 5 | 硬件质量证据 | 保持为 roadmap 级，不阻塞前 4 Sprint |

**替代（更简，先评估）**：把 `scripts/mock_loop.py` 的纯函数（`decide()`/`should_exit()`/`LoopState`，`run.sh` 当前未引用）接线进主循环——不引入第二真源与状态机 bug；仅在确有需要时再升级为有状态 controller。

**先决动作**：补一个「agent/controller 权责边界」契约段（写入 `program.md`/`AGENTS.md`），否则 Sprint 1 一定跑偏。

---

## 九、适用范围与残留风险

- 本评估基于某次真机演练的账本（`results.tsv`）实测成本；**未运行真实 GPU trial**，多并发 sweep 的实际墙钟与低并发点退化程度仍属**估算**。
- 未核验：`vllm bench sweep` 在 `vllm/vllm-openai:v0.31.0` 中是否存在（C3 前提）；本地无 `bench_results/*.json`，原始 JSON 键取自 `docs/vllm-facts.md:45`。
- 外部工具的成熟度（vllm-tuner 等）未实测。
