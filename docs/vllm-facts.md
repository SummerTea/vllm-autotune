# vLLM 事实参考（v0.31.0）

本文件记录本模板所依赖的 vLLM 行为事实，**取自 v0.31.0**（2026-10-05 发布的最新稳定版）源码与官方文档，部分条目另经 nightly（`0.31.1rc1.dev*`）实机核对（见各条标注）。模板实现（`bench.py` / `probe.py` / `setup_remote.sh`）以下列事实为准；升级 vLLM 版本时请重新核对。

> ⚠️ 版本漂移警告：v0.31 为快速迭代版本，落地前请对 `vllm serve --help=all`（bare `--help` 只给配置组索引）/ `vllm bench serve --help` 输出与启动日志行做一次冒烟核对。

## 1. `vllm bench serve` CLI

```bash
vllm bench serve \
  --model <model> \
  --dataset-name random \
  --input-len 1024 --output-len 128 \
  --request-rate inf \
  --num-prompts 1000
```

| 标志 | 默认 | 语义 |
|---|---|---|
| `--model` | None | 缺省从服务端 `/v1/models` 取第一个 |
| `--dataset-name` | `random` | choices 含 sharegpt / hf / custom / prefix_repetition / spec_bench 等；`random` 用 tokenizer 合成可复现文本 |
| `--input-len` / `--output-len` | None（random 默认 1024/128） | 映射到数据集专属参数 |
| `--num-prompts` | 1000 | 请求数 |
| `--request-rate` | `inf` | req/s；`inf` = 全部在 t=0 发出 |
| `--max-concurrency` | None | 在途请求上限 |
| `--burstiness` | 1.0 | 仅 request-rate≠inf 生效 |
| `--save-result` / `--result-dir` | off / cwd | 结果 JSON（字段见 §2） |
| `--seed` | 0 | 数据集/采样种子 |
| `--random-input-len` / `--random-prefix-len` | — | random 数据集：`--random-input-len`=非前缀余量、`--random-prefix-len`=前置共享前缀；**总 prompt = 两者之和**（`datasets.py` RandomDataset 确证）。用于模拟目标前缀缓存命中率 |
| `--percentile-metrics` | `ttft` | 逗号分隔；合法值 `ttft,tpot,itl,e2el`（控制输出哪些百分位） |
| `--trust-remote-code` / `--tokenizer` | off / None | — |

### `--goodput`（关键）

```bash
--goodput ttft:500 tpot:50 e2el:2500   # nargs="+", "KEY:VALUE" 空格分隔，单位毫秒
```

- 合法 KEY 仅 `ttft` / `tpot` / `e2el`，值 ≥ 0。
- 计算：只统计**成功**请求；要求**所有**指定 SLO 同时满足；`request_goodput = good_completed / dur_s`（**req/s**），打印为 `Request goodput (req/s):`。
- 引入：PR #9338（2024-10-20），随 v0.6.3 发布线面世；源头为 DistServe 论文 (arXiv:2401.09670)。

## 2. bench 输出字段（stdout / `--save-result` JSON）

顶层键：`duration, completed, failed, total_input_tokens, total_output_tokens, request_throughput, request_goodput, output_throughput, total_token_throughput, ...`（无 `--goodput` 时 `request_goodput` 为 null）。

百分位：`mean_<m>_ms, median_<m>_ms, std_<m>_ms, p99_<m>_ms`，其中 `m ∈ {ttft, tpot, itl, e2el}`。

> 注意：**不存在** `p99_e2e_latency_ms`，实际键名是 `p99_e2el_ms`。

stdout 表格头：`Successful requests:` / `Failed requests:` / `Request throughput (req/s):` / `Output token throughput (tok/s):` / `Mean TTFT (ms):` 等。

## 3. 启动日志（校准用 ground truth）

单条日志（`vllm/v1/core/kv_cache_utils.py`）：

```
GPU KV cache size: 1,234,567 tokens, Maximum concurrency for 204,800 tokens per request: 5.00x
```

其他可校准行：

```
Auto-fit max_model_len: reduced from %d to %d to fit in available GPU memory (%s GiB available for KV cache)
Chunked prefill is enabled with max_num_batched_tokens=%d.
```

- `--kv-cache-memory` **不存在**；正确标志是 **`--kv-cache-memory-bytes`**（每 GPU 字节数，覆盖 `gpu-memory-utilization`，支持 `1G`/`2M` 后缀）。启动日志不回显字节值，只打印解析后的 token 数——复现时以 "GPU KV cache size: … tokens" 行为校准对照。
- v0.6~v0.8 时代这两句是两条独立日志；v0.31 合并为一条且按 KV cache 容量计算。

## 4. 可调 serve 参数（规范名 + v0.31 默认 + 漂移）

| 标志 | 默认 | 说明 |
|---|---|---|
| `--max-num-seqs` | None（硬件相关，DEFAULT 128；H100/H200 serve=1024，A100=256） | 模板应显式传值 |
| `--max-num-batched-tokens` | None（硬件相关，DEFAULT 2048；H100/H200=8192，A100=2048，Blackwell≥160GB=16384） | v0.8 起自适应；支持 `2K/4K/32K` 后缀 |
| `--gpu-memory-utilization` | **0.92** | 别名 `--device-memory-utilization` |
| `--kv-cache-dtype` | `auto` | 合法值随版本（auto/fp8/e5m2/…） |
| `--tensor-parallel-size` / `--pipeline-parallel-size` / `--data-parallel-size` | 1/1/1 | tp 切分权重；dp 每副本持**完整权重**（显存按副本数倍增），可与 tp 组合（总 GPU = tp × dp）；pp 切层、默认 1（pp>1 支持依模型/版本而定，使用前以 `--help=all` 与启动日志核对） |
| `--enable-expert-parallel` | False | 需 TP>1 且模型 opt-in |
| `--enable-prefix-caching` | True（CLI 层 None=按模型自动） | — |
| `--enable-chunked-prefill` | None=自动（V1 尽力开启） | — |
| `--max-cudagraph-capture-size` | None → 上限 512（数据中心 Blackwell 1024） | — |
| `--speculative-config` | JSON，如 `'{"method":"ngram","n":4}'`；别名 `-sc` | 支持 eagle3/draft_model/mtp 等 |
| `--api-server-count` | None=自动 | 见 cli_serve.py 分支 |
| `--async-scheduling` | None=默认开启 | pooling/不兼容 spec 时关闭 |
| `--stream-interval` | 1 | — |
| `--quantization` | None=自动识别；别名 `-q` | gptq/awq/fp8/mxfp4/compressed-tensors… |
| `--load-format` | `auto` | — |
| `--enforce-eager` | False | 关闭 torch.compile + CUDA graph |
| `--kv-cache-memory-bytes` | None | 覆盖 GMU；`--kv-cache-memory` 不存在 |
| `-O0..-O3`（`--optimization-level`） | **O2** | O3 当前等同 O2 |

### flag 预校验（bench.py 起服前）

- `bench.py` 用镜像内 **`vllm serve --help=all`**（`docker run --entrypoint vllm <image> serve --help=all`，stdout+stderr 都抓，按 image 缓存到 out-dir）提取合法 flag 集；`config.yaml.serve` 里不在该集的键在起服前 fail-fast crash。
- **bare `vllm serve --help` 只打印配置组索引**（如 `--help=ModelConfig`，约 41 行），不列任何真实 flag——不可用于预校验；完整列表须 `--help=all`（约 2384 行）。
- 嵌套点号子参数（如 `--attention_config.indexer_kv_dtype`）是 vLLM `FlexibleArgumentParser` 的真实语法（vLLM 官方部署样例/文档即用此形态——以官方文档为准，非某一模型专属；`--help=<dotted>` 查不到它只是帮助机制不枚举子字段）；模板对含 `.` 的键**原样透传**（不 `_`→`-`），并只校验其顶层组（`--attention-config`）——因为 `--help=all` 只列顶层组选项、不列子字段。
  - 代价（已知边界）：**子字段拼写错误不会被预校验捕获**（fail-fast 只覆盖顶层组名），会由 vLLM 起服时报错或按未知键忽略。
- 跳过条件：镜像无法提供可用 help（rc≠0，或文本里没有任何真实 flag token）→ 跳过预校验（best-effort，不阻塞 trial）。

## 5. 健康 / 就绪

- `GET /health` → **200** 健康 / **503** `EngineDeadError`。
- 轮询：`curl -s -o /dev/null -w "%{http_code}" http://<host>:<port>/health`，或等 `/v1/models` 可用。
- 其他：`/load`、`/version`、`/metrics`、`/kv_event_sources`。
- `POST /reset_prefix_cache`：清空前缀缓存（bench.py 在 repeats 之间调用，保证每次测量为冷缓存独立）。

## 6. NCCL 环境变量（多 GPU / 跨节点）

| 变量 | 作用 | 典型值 |
|---|---|---|
| `NCCL_IB_HCA` | 指定 RDMA HCA（**值前缀 `=` 表示精确匹配**，如 `NCCL_IB_HCA="=mlx5_0:1"`；不加 `=` 为前缀匹配） | `mlx5_0,mlx5_1`；`^` 排除 |
| `NCCL_SOCKET_IFNAME` | 指定 IP 接口前缀 | `eth0,ens3`；`^eth` 排除 |
| `NCCL_IB_GID_INDEX` | RoCE GID 索引 | 默认 0；RoCEv2 常用 3 |
| `NCCL_NET_GDR_LEVEL` | GPU Direct RDMA 等级 | 0 禁用；2/3 直通/中间缓冲 |
| `NCCL_P2P_LEVEL` | P2P 距离 cutoff（**官方文档口径**合法值 `LOC`/`NVL`/`PIX`/`PXB`/`PHB`/`SYS`，**无 `NODE`**；源码另接受内部值） | cutoff 达 `PXB` 时：同 PCIe switch（PIX/PXB）走 P2P；**`PHB` 及以上**默认关 P2P → 回落 SHM（主机内存/UPI 中转），**不走 NET**。注：`PHB` = **同 NUMA**（穿 CPU host bridge），`SYS` 才是跨 NUMA |
| `NCCL_P2P_DISABLE` | 禁用 GPUDirect P2P | 1。注意：`P2P_DISABLE` 会覆盖 / 使 `NCCL_P2P_LEVEL` 失效（不再按距离 cutoff 判 P2P，一律回落 SHM/NET）；**单节点通常为负优化**（砍掉节点内最快路径），search_space.json 对单节点默认置空 |
| `NCCL_SHM_DISABLE` | 禁用 SHM 传输（该 pair 回落到 NET/网络） | 1 |
| `NCCL_DEBUG` | 调试输出 | VERSION / WARN（默认）/ INFO / TRACE |
| `NCCL_IB_DISABLE` | 禁用 IB/RoCE，回退 socket | 1 |

- NCCL 传输选择（事实）：每对 GPU 先按距离 cutoff 判 P2P；P2P 不可用且同节点时回落 SHM（共享内存/主机中转）；**P2P 与 SHM 双双不可用才把该对标 `PATH_NET`**（走 RoCE/IB）。因此「域内 PCIe P2P + 跨域 RoCE」的组合是 `NCCL_P2P_LEVEL=PXB` + `NCCL_SHM_DISABLE=1`（+ `NCCL_IB_HCA` 用**值前缀 `=`** 精确匹配，如 `="=mlx5_0:1"`）。

## 7. 固定版本与安装

- 最新稳定版：**v0.31.0**（2026-10-05）。
- `uv pip install vllm==0.31.0`（PyPI 默认 CUDA 13.0 wheel；CUDA 12.9 环境用 `vllm==0.31.0+cu129`）。
- Docker：`vllm/vllm-openai:v0.31.0`（CUDA 13.0）/ `:v0.31.0-cu129` / `vllm/vllm-openai-rocm:v0.31.0`。
- ⚠️ v0.31 默认 CUDA 13.0；生产 CUDA 12.x 栈请选 `-cu129`。v0.29 起 `python -m vllm.entrypoints.openai.api_server` 已废弃，一律 `vllm serve`。

## 8. 拓扑 / 带宽检查命令

- `nvidia-smi topo -m`：GPU×GPU 矩阵；`NV#`(NVLink)、`NVSW`(NVSwitch)、`PIX`(同 PCIe switch)、`PHB`(同 host bridge)、`NODE`(跨 NUMA)、`SYS`(最慢)。多卡 NVLink 全互联应全为 `NV#`（H100=NV12）。
- `nvidia-smi -q -d CLOCK,POWER,THERMAL,UTILIZATION`：`Persistence Mode`（关 → 时钟掉，需 `nvidia-smi -pm 1`）、`Power Draw/Limit`、`Clocks`、`Temperature`、`Utilization`、`Memory Usage`、`GPU UUID`。
- 跨节点带宽：perftest `ib_write_bw -a -d mlx5_0 -F`（对照 200G/400G 线速）。
- GPU 集合带宽：`nccl-tests` → `all_reduce_perf -b 8 -e 1G -f 2 -g 8 -n 50 -w 20`（读 `busbw` 列；跨节点加 `-N` 与正确 `NCCL_IB_HCA`）。

## 存疑点

- `--goodput` 落在 v0.6.3 还是 v0.6.3.post1 未逐 tag 确证；"v0.6.3 发布线引入" 是可靠下限。
- quantization / kv-cache-dtype 的合法值表随版本扩展，需以 `vllm serve --help=all` 实际输出为准。
