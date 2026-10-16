# 借鉴：vLLM 官方 `benchmarks/auto_tune`

> 源码：`.slim/clonedeps/repos/vllm-project__vllm/benchmarks/auto_tune/`（`vllm-project/vllm` @ `v0.31.0`，sparse 检出）。
> 这是官方"网格搜 `max-num-seqs × max-num-batched-tokens` + `--goodput e2el:` SLO 约束"的调优脚本，与本模板循环同构。下面记录**值得借鉴**与**应规避**之处。

## 高价值借鉴（建议纳入）

### A. 预检自动探测最大 `gpu-memory-utilization` ★★★
- 来源：`auto_tune.sh:252-272`（从 0.98 起每步 -0.01，直到服务器能起不 OOM，之后固定用该值做网格）。
- 价值：**直接解决共享主机的痛点**——每卡可用显存常远低于标称，固定高 `gmu` 会被拒；自动下降探测可落到实际可用值。
- 适配（**已实现**）：`bench.py --detect-gmu`（`run.sh --detect-gmu` → `bash run.sh --poll`）——从高值下降直到 serve 就绪，产出 `gmu_probe.json` 的 `gmu_suggested` 与 status.json（可直接轮询）；建议值用进后续 config 的 `serve.gpu-memory-utilization`。可保留为搜索维度，只是给一个"安全上限"起点。

### B. `--load-format dummy` 快速扫描 ★★★（已实现 `--detect-gmu`）
- 来源：`auto_tune.sh:95`。
- 价值：**用随机权重启动**，省去真实权重加载（真实加载常达分钟级），把"起服/内存适配"预检从分钟级降到秒级。
- 适配：`bench.py --detect-gmu` 用 dummy 快速起服探测安全 gmu。**适用范围**：dummy 生成同 shape/dtype 的张量，显存占用与稠密模型调度有代表性；**MoE 模型不适用**——router 依赖门控权重，dummy 下各 expert 负载/显存/吞吐失真，**仅可用于起服/内存适配探测，不得用于吞吐 keep/discard**（吞吐/精度实验用真实权重）。

### C. 前缀缓存感知工作负载 `--random-prefix-len` ★★★
- 来源：`auto_tune.sh:165-166,180`（`prefix_len = INPUT_LEN × MIN_CACHE_HIT_PCT/100`，`adjusted_input_len = INPUT_LEN - prefix_len`）。
- 价值：**闭合我们此前标注的缺口**——random 数据集没有共享前缀，测不出 prefix caching 收益。用固定前缀 + 目标命中率，就能让 `enable-prefix-caching` 成为可测维度。
- 适配：`harness.workload` 增 `prefix_hit_pct`；`bench.py` 的 bench 参数加 `--random-prefix-len`（并相应减少 `--input-len`）。

### D. 运行间 `/reset_prefix_cache` ★★
- 来源：`auto_tune.sh:197`（每次降速率前 `POST /reset_prefix_cache` + sleep）。
- 价值：清前缀缓存，保证每次测量独立、不被上一轮污染。
- 适配：`bench.py` 的 repeats/多配置测量之间，向 serve 发 `POST /reset_prefix_cache`。

### E. 延迟约束下的 request-rate 下降搜索 ★★
- 来源：`auto_tune.sh:162-224`（先 `--request-rate inf`；P99 E2EL 超限则逐 1 递减 request-rate 直到满足）。
- 价值：比我们固定 `ref_concurrency` 更贴近"满足 SLO 的最高可持续吞吐"。
- 适配：作为 ref_concurrency 模式的补充（可选 search 策略）。

### F. 仅对最佳配置做 profiling ★★
- 来源：`auto_tune.sh:281-321`（网格后对 best 起服并 `--profiler-config '{"profiler":"torch","torch_profiler_dir":…}'` + `vllm bench serve --profile`）。
- 价值：与 roadmap 的"收尾 profiling"一致，只在 best 上抓一次。
- 适配：Phase 4 / report 后对 best commit 起服抓 trace。

### G. 显式 `--percentile-metrics ttft,tpot,itl,e2el` ★
- 来源：`auto_tune.sh:177,209,310`。
- 价值：字段更全（我们目前只取 ttft/tpot p99）。
- 适配：`bench.py` 的 bench 参数补 `--percentile-metrics`。

## 应规避（我们已做得更稳）
- **`pkill -if "vllm serve"` 自杀陷阱**：其 README 明示脚本路径不能含 `vllm`，否则 pkill 杀掉脚本自身（`auto_tune.sh:82,148,243,320`）。我们用**唯一容器名 + `docker rm -f`**，已消除该类脆弱（Phase 7 还删掉了 docker-only 下的死 pkill）。
- **全局 `pip install -q datasets`**（`auto_tune.sh:56`）：污染主机环境。我们零破坏 + docker-only。
- **硬编码端口 8004 / `--host $HOSTNAME`**：我们参数化端口、用 `127.0.0.1`。
- **无 zero-pull 保护**：我们在 Phase 7 给 `docker run` 加了 `--pull=never` + 镜像预检。

## 结论
A/B/C 三项价值最高且与我们真机教训直接对应（gmu 争用、加载耗时、prefix caching 缺口），建议纳入 v1.1；D/E/F/G 为增量增强。整体上我们的循环在**零破坏、docker 化、可审计、退出机制**上已超出该脚本；可借其"廉价快速扫描"的工程化技巧。
