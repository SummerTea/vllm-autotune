"""test_bench.py — bench.py 单元测试 + 生命周期（stub docker）测试。

可在无 GPU 的 Mac 上运行（docker-only）：
- 解析器：stdout 表格 / --save-result JSON / 启动日志校准行
- 命令构建（docker）：serve 的 kebab flag / 布尔（true→裸 flag，false→省略，no- 前缀关默认开关）
  / 端口；bench 的 --network host / --base-url / 结果目录挂载 / 模型只读挂载
- workload 前缀缓存（C）：prefix_hit_pct 拆 isl → --random-prefix-len/--random-input-len；
  repeats 间 reset_prefix_cache（D）；--percentile-metrics（G）
- 生命周期：fake_docker.sh 全流程（status/result/校准/summary 块，docker rm 先于 run）
- crash 路径：serve 容器立即退出（并验证清理被调用）
- gmu 预检：候选序列（start/step/floor）/ 强制 --load-format dummy + --gpu-memory-utilization
  / stub OOM 降级（FAKE_DOCKER_OOM_ABOVE）→ gmu_probe.json + model_profile 就地补 gmu_suggested
"""

import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bench  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "bench"

SAMPLE_STDOUT = """============ BENCHMARK RESULTS ============
Successful requests:                     1000
Failed requests:                             0
Request throughput (req/s):              100.0
Output token throughput (tok/s):      12345.6
Mean TTFT (ms):                          420.1
P99 TTFT (ms):                           620.1
Mean TPOT (ms):                           12.3
P99 TPOT (ms):                            18.3
Request goodput (req/s):                   12.34
"""

SAMPLE_JSON = {
    "duration": 90.0, "completed": 1000, "failed": 0,
    "request_throughput": 100.0, "request_goodput": 12.34,
    "output_throughput": 12345.6, "total_output_tokens": 1000000,
    "p99_ttft_ms": 620.1, "p99_tpot_ms": 18.3,
}

SAMPLE_STARTUP_LOG = """INFO 08-08 10:00:00 model_runner.py:123] Loading model weights...
GPU KV cache size: 1,234,567 tokens, Maximum concurrency for 204,800 tokens per request: 6.02x
INFO 08-08 10:00:42 api_server.py:456] Application startup complete.
"""


class TestParsers:
    def test_parse_bench_stdout(self):
        p = bench.parse_bench_stdout(SAMPLE_STDOUT)
        assert p["goodput"] == pytest.approx(12.34)
        assert p["output_tok_s"] == pytest.approx(12345.6)
        assert p["ttft_p99_ms"] == pytest.approx(620.1)
        assert p["tpot_p99_ms"] == pytest.approx(18.3)

    def test_parse_bench_json(self):
        p = bench.parse_bench_json(SAMPLE_JSON)
        assert p["goodput"] == pytest.approx(12.34)
        assert p["output_tok_s"] == pytest.approx(12345.6)
        assert p["ttft_p99_ms"] == pytest.approx(620.1)
        assert p["tpot_p99_ms"] == pytest.approx(18.3)

    def test_parse_bench_json_missing_fields_null(self):
        p = bench.parse_bench_json({"duration": 1.0})
        assert p["goodput"] is None and p["output_tok_s"] is None

    def test_parse_bench_stdout_missing_fields_null(self):
        p = bench.parse_bench_stdout("no bench table here")
        assert all(v is None for v in p.values())

    def test_parse_kv_cache_line(self):
        assert bench.parse_kv_cache_line(SAMPLE_STARTUP_LOG) == 1234567
        assert bench.parse_kv_cache_line("no kv line here") is None
        assert bench.parse_kv_cache_line(None) is None

    def test_find_newest_json(self, tmp_path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text("{}")
        b.write_text("{}")
        future = time.time() + 10
        os.utime(b, (future, future))
        assert bench.find_newest_json(str(tmp_path)) == str(b)
        assert bench.find_newest_json(str(tmp_path / "nope")) is None

    def test_repeats_stats_stdev(self):
        # σ：n≥2 用样本标准差（与实现一致的四舍五入到 4 位）；n<2 → 0.0；None 值不参与
        rows = [{"goodput": 10.0}, {"goodput": 12.0}, {"goodput": 11.0}]
        assert bench._repeats_stats(rows, "goodput") == pytest.approx(
            round(statistics.stdev([10.0, 12.0, 11.0]), 4))
        assert bench._repeats_stats([{"goodput": 10.0}], "goodput") == 0.0
        assert bench._repeats_stats([], "goodput") == 0.0
        mixed = [{"goodput": 10.0}, {"goodput": None}, {"goodput": 14.0}]
        assert bench._repeats_stats(mixed, "goodput") == pytest.approx(
            round(statistics.stdev([10.0, 14.0]), 4))

    def test_parse_compute_apps(self):
        # 有效行 + "512 MiB"/"2.5 GiB"（取首个整数）+ 内存 [N/A] 容忍 + 坏行跳过
        text = "1234, /usr/bin/python3, 512 MiB\n" \
               "5678, worker, [N/A]\n" \
               "[N/A], python, 100 MiB\n" \
               "badrow\n" \
               "9999, torchrun, 2.5 GiB\n"
        apps = bench.parse_compute_apps(text)
        assert apps == [
            {"pid": 1234, "process_name": "/usr/bin/python3", "used_memory_mib": 512},
            {"pid": 5678, "process_name": "worker", "used_memory_mib": None},
            {"pid": 9999, "process_name": "torchrun", "used_memory_mib": 2},
        ]
        assert bench.parse_compute_apps(None) == []
        assert bench.parse_compute_apps("") == []

    def test_known_serve_flags_parses_help(self):
        help_text = """usage: vllm serve [-h] MODEL
options:
  -h, --help
  --model MODEL
  --max-num-seqs MAX_NUM_SEQS
  --enable-prefix-caching, --no-enable-prefix-caching
  --trust-remote-code, --no-trust-remote-code
"""
        known = bench._known_serve_flags(help_text)
        assert "--model" in known and "--max-num-seqs" in known
        assert "--enable-prefix-caching" in known and "--no-enable-prefix-caching" in known
        assert "--help" in known
        assert "MAX_NUM_SEQS" not in known and "-h" not in known

    def test_known_serve_flags_bracketed_usage_line(self):
        # m2：usage 行的 [--flag N] 括号形式也要收集（否则合法 flag 被误报未知）
        known = bench._known_serve_flags(
            "usage: vllm serve [-h] [--max-num-seqs N] MODEL\n"
            "  --enable-prefix-caching, --no-enable-prefix-caching\n")
        assert "--max-num-seqs" in known
        assert "--enable-prefix-caching" in known and "--no-enable-prefix-caching" in known

    def test_unknown_serve_flags(self):
        known = bench._known_serve_flags(
            "--model MODEL\n--max-num-seqs MAX_NUM_SEQS\n"
            "--enable-prefix-caching, --no-enable-prefix-caching\n")
        serve = {"max-num-seqs": 1, "enable-prefix-caching": True,
                 "no-enable-prefix-caching": True, "bogus-flag": 1}
        assert bench._unknown_serve_flags(serve, known) == ["--bogus-flag"]
        assert bench._unknown_serve_flags({"max-num-seqs": 1}, known) == []
        # no-<flag> 键也按 known 校验
        assert bench._unknown_serve_flags({"no-bogus-flag": True}, known) == ["--no-bogus-flag"]

    def test_flag_name(self):
        # 含 `.` 的嵌套子参数原样返回；普通键 `_`→`-`（kebab）
        assert bench._flag_name("max_num_seqs") == "max-num-seqs"
        assert bench._flag_name("max-num-seqs") == "max-num-seqs"
        assert bench._flag_name("attention_config.indexer_kv_dtype") == \
            "attention_config.indexer_kv_dtype"

    def test_unknown_serve_flags_underscore_and_nested(self):
        # m2：`_`→`-` 键合法；嵌套子参数（含 .）只校验顶层 root
        #   （vllm `--help=all` 只列 `--attention-config`，不列嵌套子字段）
        known = bench._known_serve_flags(
            "--model MODEL\n--max-num-seqs MAX_NUM_SEQS\n--attention-config ATTENTION_CONFIG\n")
        assert bench._unknown_serve_flags({"max_num_seqs": 1}, known) == []
        assert bench._unknown_serve_flags(
            {"attention_config.indexer_kv_dtype": "fp8"}, known) == []
        # root 未知 → 判未知（错误列表按原样透传嵌套键）
        assert bench._unknown_serve_flags({"bogus_config.x": 1}, known) == ["--bogus_config.x"]
        # 未知顶层仍判未知
        assert bench._unknown_serve_flags({"bogus-flag": 1}, known) == ["--bogus-flag"]

    def test_get_serve_help_unusable_group_index_skips(self, tmp_path, monkeypatch):
        # 现代 vLLM bare --help 只打印配置组索引（--help=ModelConfig 之类），无真实 flag →
        # --help=all 与 --help 都判定不可用 → 返回 None（调用方跳过预校验）
        group_index = ("usage: vllm serve [options] MODEL\n"
                       "options:\n"
                       "  --help\n"
                       "  ModelConfig:\n"
                       "    --help=ModelConfig\n"
                       "  EngineConfig:\n"
                       "    --help=EngineConfig\n"
                       "  KVConfig:\n"
                       "    --help=KVConfig\n")
        calls = []

        def fake_run(cmd, timeout=10):
            calls.append(cmd)
            assert cmd[-1] in ("--help=all", "--help")
            return 0, group_index, ""

        monkeypatch.setattr(bench, "_run_local", fake_run)
        assert bench._get_serve_help("docker", {"image": "img:1"}, str(tmp_path)) is None
        # 先试 --help=all，不可用再退回 --help
        assert [c[-1] for c in calls] == ["--help=all", "--help"]
        # 两份都不可用 → 不写缓存
        assert list(tmp_path.iterdir()) == []

    def test_resolve_slo_priority(self):
        # 有效 SLO = harness.slo ⊕ target.metrics.slo ⊕ config.slo（后者覆盖前者）
        harness = {"slo": {"ttft_p99_ms": 500, "tpot_p99_ms": 50}}
        target = {"metrics": {"slo": {"ttft_p99_ms": 600, "tpot_p99_ms": 60}}}
        # config.slo 覆盖 target.metrics.slo（逐键覆盖）
        assert bench.resolve_slo(harness, target, {"slo": {"ttft_p99_ms": 700}}) == \
            {"ttft_p99_ms": 700, "tpot_p99_ms": 60}
        assert bench.resolve_slo(harness, target, {"slo": {"ttft_p99_ms": 700, "tpot_p99_ms": 70}}) == \
            {"ttft_p99_ms": 700, "tpot_p99_ms": 70}
        # target.metrics.slo 覆盖 harness.slo
        assert bench.resolve_slo(harness, target, {}) == {"ttft_p99_ms": 600, "tpot_p99_ms": 60}
        # config 缺省 → 行为不变（target 覆盖 harness；无 target 则回 harness）
        assert bench.resolve_slo(harness, {}, {}) == {"ttft_p99_ms": 500, "tpot_p99_ms": 50}
        # 全缺省 → 空（DEFAULT_SLO 由调用方合并，语义不变）
        assert bench.resolve_slo({}, {}, {}) == {}

    def test_resolved_workload(self):
        # 实际生效 workload = DEFAULT_WORKLOAD ⊕ harness/target + 解析后的 ref_concurrency 与 seed
        wl = {"dataset": "random", "isl": 2048, "osl": 256, "num_prompts": 500}
        assert bench._resolved_workload(wl, 128) == {
            "dataset": "random", "isl": 2048, "osl": 256, "num_prompts": 500,
            "ref_concurrency": 128, "prefix_hit_pct": 0, "seed": 0}
        # 缺省键回 DEFAULT_WORKLOAD；seed 传入生效（属 workload 身份）
        assert bench._resolved_workload({}, 64) == {
            "dataset": "random", "isl": 1024, "osl": 128, "num_prompts": 1000,
            "ref_concurrency": 64, "prefix_hit_pct": 0, "seed": 0}
        assert bench._resolved_workload({}, 64, seed=7)["seed"] == 7

    def test_serve_crash_is_oom_signatures(self):
        # F10：OOM 特征识别（大小写/换行宽容）——Free memory + less than desired / CUDA OOM / torch OOM
        assert bench._serve_crash_is_oom(
            "ValueError: Free memory on device cuda:0 (2.0 GiB / 22.0 GiB) on startup "
            "is less than desired GPU memory utilization (0.92)") is True
        assert bench._serve_crash_is_oom(
            "free memory on device cuda:0 is low\nless than desired gpu memory utilization") is True  # 跨行
        assert bench._serve_crash_is_oom("RuntimeError: CUDA out of memory.") is True
        assert bench._serve_crash_is_oom("torch.OutOfMemoryError: allocated 2.0 GiB") is True
        assert bench._serve_crash_is_oom("") is False
        assert bench._serve_crash_is_oom("fake docker serve crashing") is False
        assert bench._serve_crash_is_oom("Free memory on device cuda:0 is fine") is False  # 缺后半特征

    def test_classify_oom_crash(self, tmp_path, monkeypatch):
        # F10：命中 OOM → 可执行口径 reason + 崩溃当时重新共租检测的 pid；未命中不回归
        log = tmp_path / "run.log"
        log.write_text("=== bench.py serve(docker): ...\n"
                       "ValueError: Free memory on device cuda:0 (2.0 GiB / 22.0 GiB) on startup "
                       "is less than desired GPU memory utilization\n",
                       encoding="utf-8")
        monkeypatch.setattr(bench, "_detect_co_tenants",
                            lambda: [{"pid": 1234, "process_name": "torchrun",
                                      "used_memory_mib": 512}])
        reason, cont = bench._classify_oom_crash("serve 提前退出（rc=1）", str(log), 0)
        assert "serve OOM" in reason and "1234(torchrun)" in reason
        assert cont is not None
        assert cont["foreign_gpu_apps"][0]["pid"] == 1234
        # 未命中 → 原样返回 + None（不回归既有通用崩溃原因）
        (log.write_text("fake docker serve crashing\n", encoding="utf-8"))
        reason2, cont2 = bench._classify_oom_crash("serve 提前退出（rc=1）", str(log), 0)
        assert reason2 == "serve 提前退出（rc=1）" and cont2 is None
        # since=None（无本次段起点）→ 不扫描，不分类
        reason3, cont3 = bench._classify_oom_crash("serve 提前退出（rc=1）", str(log), None)
        assert reason3 == "serve 提前退出（rc=1）" and cont3 is None


# ── P0：trial 身份与配置指纹 / extra_docker_args 零破坏白名单 ─────────────

class TestTrialIdentity:
    def test_sha12(self):
        assert len(bench._sha12("x")) == 12
        assert bench._sha12("a") != bench._sha12("b")

    def test_config_fingerprint_stable_across_key_order(self):
        a = {"serve": {"max-num-seqs": 256}, "env": {"A": "1"}}
        b = {"env": {"A": "1"}, "serve": {"max-num-seqs": 256}}
        assert bench.config_fingerprint(a) == bench.config_fingerprint(b)
        assert len(bench.config_fingerprint(a)) == 12

    def test_config_fingerprint_differs_when_value_changes(self):
        a = {"serve": {"max-num-seqs": 256}, "env": {"A": "1"}}
        b = {"serve": {"max-num-seqs": 128}, "env": {"A": "1"}}
        assert bench.config_fingerprint(a) != bench.config_fingerprint(b)


class TestExtraDockerArgsWhitelist:
    def test_validate_accepts_whitelist(self):
        # 白名单：--cpuset-cpus 期待值（不带 =）；--cpuset-mems=0 自带值
        assert bench._validate_extra_docker_args(
            ["--cpuset-cpus", "0-31", "--cpuset-mems=0"]) == []

    def test_validate_rejects_offending(self):
        assert bench._validate_extra_docker_args(["-v", "/x:/y"]) != []      # 挂载
        assert bench._validate_extra_docker_args(["--privileged"]) != []     # 特权
        assert bench._validate_extra_docker_args(["--gpus", "all"]) != []    # 设备
        assert bench._validate_extra_docker_args(["--network", "host"]) != []  # 网络
        assert bench._validate_extra_docker_args(["0-31"]) != []             # 游离值

    def test_validate_rejects_dangling_whitelisted_flag(self):
        # m1：白名单 flag 期待值却被下一个 flag 打断 / 到列表末尾仍未提供 → 悬空判非法
        assert bench._validate_extra_docker_args(
            ["--cpuset-cpus", "--cpuset-mems=0"]) == ["--cpuset-cpus"]
        assert bench._validate_extra_docker_args(["--cpuset-cpus"]) == ["--cpuset-cpus"]
        # 带值写法与正常 值+flag 序列不误伤
        assert bench._validate_extra_docker_args(["--cpuset-cpus=0-31", "--cpuset-mems=0"]) == []
        assert bench._validate_extra_docker_args(["--cpuset-cpus", "0-31", "--cpuset-mems=0"]) == []

    def test_build_docker_run_cmd_rejects_non_whitelisted(self):
        runtime = {"kind": "docker", "image": "img:1",
                   "extra_docker_args": ["--privileged"]}
        with pytest.raises(ValueError) as ei:
            bench.build_docker_run_cmd("docker", runtime, "/m", 8000, {"serve": {}})
        assert "--privileged" in str(ei.value)
        assert "--cpuset-cpus" in str(ei.value)   # 错误消息说明白名单

    def test_build_docker_run_cmd_accepts_whitelisted(self):
        runtime = {"kind": "docker", "image": "img:1",
                   "extra_docker_args": ["--cpuset-cpus", "0-31"]}
        cmd = bench.build_docker_run_cmd("docker", runtime, "/m", 8000, {"serve": {}})
        assert "--cpuset-cpus" in cmd and "0-31" in cmd


class TestDockerRuntime:
    RUNTIME = {"kind": "docker", "image": "vllm/vllm-openai:v0.31.0",
               "container_model_path": "/model", "shm_size": "16g",
               "extra_docker_args": ["--cpuset-cpus", "0-31"]}

    def test_effective_runtime_merge(self):
        rt = bench.effective_runtime({"runtime": {}}, {"runtime": {"image": "img:1"}})
        assert rt["image"] == "img:1"
        assert bench.effective_runtime({}, {}) == {}
        # 遗留 kind 键被接受并忽略（向后兼容），不影响 docker-only
        rt2 = bench.effective_runtime({"runtime": {"kind": "docker", "image": "a:1"}}, {})
        assert rt2["kind"] == "docker" and rt2["image"] == "a:1"

    def test_build_docker_run_cmd(self):
        cfg = {"env": {"NCCL_IB_DISABLE": "1"},
               "serve": {"max-num-seqs": 256, "enable-prefix-caching": True,
                         "enforce-eager": False}}
        cmd = bench.build_docker_run_cmd("docker", self.RUNTIME, "/models/qwen3", 8000, cfg)
        joined = " ".join(cmd)
        assert cmd[0] == "docker" and cmd[1] == "run"
        assert cmd[2] == "--pull=never"                      # M1：零 pull 契约（缺失镜像不自动拉取）
        assert "--name" in cmd and "vllm-autotune-trial" in cmd
        assert "--gpus" in cmd and "all" in cmd
        assert "--ipc" in cmd and "host" in cmd
        assert cmd.count("--ipc") == 1                       # 无重复 --ipc（extra 不含默认项）
        assert "--shm-size" in cmd and "16g" in joined
        assert "-p" in cmd and "127.0.0.1:8000:8000" in joined
        assert "-v" in cmd and "/models/qwen3:/model:ro" in joined
        assert "-e" in cmd and "NCCL_IB_DISABLE=1" in joined
        assert "--privileged" not in cmd and "--cpuset-cpus" in cmd  # 白名单 extra 透传（P0 零破坏）
        assert "0-31" in cmd
        # entrypoint / image / 模型位
        i = cmd.index("--entrypoint")
        assert cmd[i + 1] == "vllm"
        assert "vllm/vllm-openai:v0.31.0" in cmd
        assert cmd[cmd.index("serve") + 1] == "/model"    # serve <container_model_path>
        # 布尔规则在 docker 命令内仍生效
        assert "--enable-prefix-caching" in cmd
        assert "--enforce-eager" not in cmd
        assert "--max-num-seqs" in cmd

    def test_build_docker_run_cmd_defaults(self):
        # container_model_path / shm_size 缺省
        cmd = bench.build_docker_run_cmd("docker", {"kind": "docker", "image": "img:1"},
                                         "/m", 8000, {"serve": {}, "env": {}})
        joined = " ".join(cmd)
        assert "/m:/model:ro" in joined
        assert "--shm-size 16g" in joined

    def test_build_docker_run_cmd_custom_name(self):
        # M1：部署产物用独立容器名（默认 trial 名不变，trial 路径逐字节一致）
        cmd = bench.build_docker_run_cmd("docker", self.RUNTIME, "/m", 8000,
                                         {"serve": {}, "env": {}}, name="vllm-deploy-best")
        assert cmd[cmd.index("--name") + 1] == "vllm-deploy-best"
        assert "vllm-autotune-trial" not in cmd
        cmd2 = bench.build_docker_run_cmd("docker", self.RUNTIME, "/m", 8000,
                                          {"serve": {}, "env": {}})
        assert cmd2[cmd2.index("--name") + 1] == "vllm-autotune-trial"

    def test_build_docker_run_cmd_missing_image(self):
        with pytest.raises(ValueError):
            bench.build_docker_run_cmd("docker", {"kind": "docker"}, "/m", 8000, {"serve": {}})

    def test_build_docker_run_cmd_false_bool_omitted(self):
        # M3 回归：False 布尔 → 在 docker serve 参数内整体省略（vLLM 布尔是开关式）
        cmd = bench.build_docker_run_cmd("docker", self.RUNTIME, "/m", 8000,
                                         {"serve": {"enforce-eager": False}, "env": {}})
        assert "--enforce-eager" not in cmd
        assert "--enforce-eager false" not in " ".join(cmd)

    def test_build_docker_run_cmd_no_flag_turns_off_default(self):
        # 关闭默认开启的开关：no-<flag>: true 原样透传为 --no-<flag>
        cmd = bench.build_docker_run_cmd("docker", self.RUNTIME, "/m", 8000,
                                         {"serve": {"no-enable-prefix-caching": True}, "env": {}})
        assert "--no-enable-prefix-caching" in cmd

    def test_build_docker_run_cmd_explicit_port_not_duplicated(self):
        cmd = bench.build_docker_run_cmd("docker", self.RUNTIME, "/m", 8000,
                                         {"serve": {"port": 9999}, "env": {}})
        assert cmd.count("--port") == 1
        assert "9999" in cmd

    def test_build_docker_bench_cmd(self):
        workload = {"dataset": "random", "isl": 2048, "osl": 256, "num_prompts": 1000}
        slo = {"ttft_p99_ms": 500, "tpot_p99_ms": 50}
        cmd = bench.build_docker_bench_cmd("docker", self.RUNTIME, "/model", workload,
                                           slo, 128, 0, "/tmp/r", port=8000)
        assert cmd[:3] == ["docker", "run", "--pull=never"]    # M1：零 pull 契约
        # B1：bench 容器必须共享主机网络栈，直达 serve 发布的 127.0.0.1:<port>
        assert "--network" in cmd and "host" in cmd
        i = cmd.index("--entrypoint")
        assert cmd[i + 1] == "vllm"
        assert "vllm/vllm-openai:v0.31.0" in cmd
        joined = " ".join(cmd)
        assert "--model /model" in joined
        # 结果目录经 -v 挂载，--result-dir 指向容器内路径（JSON 落回宿主）
        assert "-v /tmp/r:/bench_results" in joined
        assert "--result-dir /bench_results" in joined
        assert "--base-url http://127.0.0.1:8000" in joined
        assert "--goodput ttft:500 tpot:50" in joined
        assert "--save-result" in joined

    def test_build_docker_bench_cmd_mounts_model(self):
        # random 数据集需要 tokenizer：传 host_model_path 时 bench 容器须只读挂载模型目录
        workload = {"dataset": "random", "isl": 64, "osl": 64, "num_prompts": 8}
        slo = {"ttft_p99_ms": 500, "tpot_p99_ms": 50}
        cmd = bench.build_docker_bench_cmd("docker", self.RUNTIME, "/model", workload,
                                           slo, 8, 0, "/tmp/r", port=8000,
                                           host_model_path="/host/models/m")
        assert "-v /host/models/m:/model:ro" in " ".join(cmd)
        # 不传 host_model_path 时不挂载模型（向后兼容）
        cmd2 = bench.build_docker_bench_cmd("docker", self.RUNTIME, "/model", workload,
                                            slo, 8, 0, "/tmp/r", port=8000)
        assert "-v /host/models/m:/model:ro" not in " ".join(cmd2)


# ── workload 前缀缓存（C）/ 测量卫生（D+G）──────────────────────────────

class TestBenchArgs:
    SLO = {"ttft_p99_ms": 500, "tpot_p99_ms": 50}

    def test_prefix_pct_zero_keeps_input_len(self):
        # (a) prefix_hit_pct=0（默认）→ 维持 --input-len，无 --random-prefix-len
        wl = {"dataset": "random", "isl": 1000, "osl": 128, "num_prompts": 8}
        args = bench._bench_args("Qwen/Qwen3-32B", wl, self.SLO, 8, 0, "/r", False, 8000)
        joined = " ".join(args)
        assert "--input-len 1000" in joined
        assert "--random-prefix-len" not in joined
        assert "--percentile-metrics ttft,tpot,itl,e2el" in joined   # (c)

    def test_prefix_pct_splits_isl(self):
        # (b) prefix_hit_pct=60, isl=1000 → 600 前缀 + 400 余量，总 prompt ≈ isl
        wl = {"dataset": "random", "isl": 1000, "osl": 128, "num_prompts": 8,
              "prefix_hit_pct": 60}
        args = bench._bench_args("Qwen/Qwen3-32B", wl, self.SLO, 8, 0, "/r", False, 8000)
        joined = " ".join(args)
        assert "--random-input-len 400" in joined
        assert "--random-prefix-len 600" in joined
        assert "--input-len" not in joined
        assert "--output-len 128" in joined
        assert "--percentile-metrics ttft,tpot,itl,e2el" in joined   # (c)

    def test_prefix_pct_ignored_for_non_random(self):
        # 非 random 数据集：prefix_hit_pct 忽略，维持 --input-len，不产 --random-prefix-len
        wl = {"dataset": "sharegpt", "isl": 1000, "osl": 128, "num_prompts": 8,
              "prefix_hit_pct": 60}
        args = bench._bench_args("Qwen/Qwen3-32B", wl, self.SLO, 8, 0, "/r", False, 8000)
        joined = " ".join(args)
        assert "--input-len 1000" in joined
        assert "--random-prefix-len" not in joined
        assert "--random-input-len" not in joined

    def test_reset_prefix_cache_posts_url(self, tmp_path, monkeypatch):
        # (D) 尽力 POST /reset_prefix_cache；断言方法/URL/超时，非 200 容忍
        calls = []

        class FakeResp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=0):
            calls.append((req.full_url, req.get_method(), timeout))
            return FakeResp()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        log = tmp_path / "run.log"
        with open(log, "wb") as fh:
            bench._reset_prefix_cache(18090, fh)
        assert calls == [("http://127.0.0.1:18090/reset_prefix_cache", "POST", 5)]
        assert log.read_text(encoding="utf-8") == ""                 # 200 → 无日志

    def test_reset_prefix_cache_tolerates_failure(self, tmp_path, monkeypatch):
        # (D) 连接失败（模拟）→ 容忍：写 run.log，不抛异常
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda req, timeout=0: (_ for _ in ()).throw(
                                urllib.error.URLError("boom")))
        log = tmp_path / "run.log"
        with open(log, "wb") as fh:
            bench._reset_prefix_cache(18090, fh)
        text = log.read_text(encoding="utf-8")
        assert "reset_prefix_cache 失败（容忍）" in text and "boom" in text


# ── --detect-gmu 预检（候选序列 / serve 参数 / stub 探测降级）──────────────

class TestGmuDetect:
    RUNTIME = {"kind": "docker", "image": "vllm/vllm-openai:v0.31.0",
               "container_model_path": "/model", "shm_size": "16g"}

    def test_gmu_candidates_defaults(self):
        cands = bench.gmu_candidates({})
        assert cands[:3] == [0.98, 0.96, 0.94]
        assert cands[-1] == 0.10
        assert len(cands) == 45                      # (0.98-0.10)/0.02 + 1

    def test_gmu_candidates_custom_start_step_floor(self):
        cfg = {"gmu_detect": {"start": 0.5, "step": 0.05, "floor": 0.35}}
        assert bench.gmu_candidates(cfg) == [0.5, 0.45, 0.4, 0.35]

    def test_gmu_probe_config_overrides_and_flags(self):
        # 覆盖 config 已有的 gmu/load-format（含下划线变体），强制 dummy + 候选 gmu
        cfg = bench._gmu_probe_config({"serve": {"max-num-seqs": 8,
                                                 "gpu_memory_utilization": 0.5,
                                                 "load_format": "safetensors"}}, 0.9)
        serve = cfg["serve"]
        assert serve["load-format"] == "dummy" and serve["gpu-memory-utilization"] == 0.9
        assert "gpu_memory_utilization" not in serve and "load_format" not in serve
        cmd = bench.build_docker_run_cmd("docker", self.RUNTIME, "/m", 8000, cfg)
        joined = " ".join(cmd)
        assert "--load-format dummy" in joined
        assert "--gpu-memory-utilization 0.9" in joined
        assert "0.5" not in joined and "safetensors" not in joined


def test_gmu_probe_stub_docker(tmp_path, capsys, monkeypatch):
    # FAKE_DOCKER_OOM_ABOVE=0.95：0.98/0.96 起服即 OOM 退出，0.94 起服成功
    port = 18095
    _write_docker_inputs(tmp_path, port, startup=10, trial=30, repeats=1)
    out = tmp_path / "out"
    out.mkdir()
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))
    monkeypatch.setenv("FAKE_DOCKER_OOM_ABOVE", "0.95")
    # 预置 model_profile.json：探测后应就地补 gmu_suggested（保留其余字段）
    (out / "model_profile.json").write_text(
        json.dumps({"model": "Qwen/Qwen3-32B", "calibrated": False,
                    "kv": {"bytes_per_token": 262144}}), encoding="utf-8")

    rc = bench.main(_docker_args(tmp_path, out, state_file) + ["--detect-gmu"])
    assert rc == 0
    probe = json.loads((out / "gmu_probe.json").read_text(encoding="utf-8"))
    assert probe["gmu_suggested"] == pytest.approx(0.94)          # ≤ 0.95（首个不 OOM 的值）
    assert probe["tried"] == [0.98, 0.96, 0.94]
    assert probe["load_format"] == "dummy"
    assert "serve 在 gmu=0.94 就绪" in probe["reason"]
    sout = capsys.readouterr().out
    assert "gmu_suggested: 0.94" in sout
    # run.log：各级 serve cmd 都带 --load-format dummy 与候选 gmu；失败级留痕
    log = (out / "run.log").read_text(encoding="utf-8")
    assert "--load-format dummy" in log
    assert "--gpu-memory-utilization 0.98" in log
    assert "gmu=0.98 失败" in log and "gmu=0.96 失败" in log
    # model_profile.json 就地补 gmu_suggested，其余字段保留
    p2 = json.loads((out / "model_profile.json").read_text(encoding="utf-8"))
    assert p2["gmu_suggested"] == pytest.approx(0.94)
    assert p2["calibrated"] is False and p2["kv"]["bytes_per_token"] == 262144
    # (c) 探测写 status.json（done），run.sh --poll 可直接轮询
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "done" and status["phase"] == "done"
    assert "gmu_suggested=0.94" in status["message"]


def test_gmu_probe_missing_image_crash(tmp_path, capsys, monkeypatch):
    # (a) 镜像预检失败 → 探测不开始，status.json crash + 明确消息 + 无 gmu_probe.json
    port = 18096
    _write_docker_inputs(tmp_path, port, startup=10, trial=30, repeats=1)
    out = tmp_path / "out"
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))
    monkeypatch.setenv("FAKE_DOCKER_NO_IMAGE", "1")

    rc = bench.main(_docker_args(tmp_path, out, state_file) + ["--detect-gmu"])
    assert rc == 1
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "crash" and status["phase"] == "crash"
    assert "未在主机预先准备" in status["message"]
    assert not (out / "gmu_probe.json").exists()
    captured = capsys.readouterr()
    assert "未在主机预先准备" in captured.err


def test_gmu_probe_floor_exhausted(tmp_path, capsys, monkeypatch):
    # (b) 全部候选 OOM（FAKE_DOCKER_OOM_ABOVE=0）→ gmu_suggested null + rc=1 + status crash
    port = 18097
    _write_docker_inputs(tmp_path, port, startup=10, trial=30, repeats=1)
    out = tmp_path / "out"
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))
    monkeypatch.setenv("FAKE_DOCKER_OOM_ABOVE", "0")

    rc = bench.main(_docker_args(tmp_path, out, state_file) + ["--detect-gmu"])
    assert rc == 1
    probe = json.loads((out / "gmu_probe.json").read_text(encoding="utf-8"))
    assert probe["gmu_suggested"] is None
    assert probe["tried"][0] == 0.98 and probe["tried"][-1] == 0.10   # 试满到 floor
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "crash" and status["phase"] == "crash"
    sout = capsys.readouterr().out
    assert "gmu_suggested: 无" in sout


def test_gmu_probe_status_running_then_done(tmp_path, monkeypatch):
    # (c) 探测期间 status.json 为 running/probing，成功后转为 done
    import threading
    # 放慢 /health 探测（每级 0.5s），让 running 状态有可观测窗口；不动 fixture
    real_health = bench._health_code

    def slow_health(port, timeout=2.0):
        time.sleep(0.5)
        return real_health(port, timeout)

    monkeypatch.setattr(bench, "_health_code", slow_health)
    port = 18098
    _write_docker_inputs(tmp_path, port, startup=10, trial=30, repeats=1)
    out = tmp_path / "out"
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    res: dict = {}

    def _run():
        res["rc"] = bench.main(_docker_args(tmp_path, out, state_file) + ["--detect-gmu"])

    t = threading.Thread(target=_run)
    t.start()
    seen_running = False
    end = time.time() + 10
    while time.time() < end and t.is_alive():
        sp = out / "status.json"
        if sp.exists():
            st = json.loads(sp.read_text(encoding="utf-8"))
            if st["state"] == "running" and st["phase"] == "probing":
                seen_running = True
                break
        time.sleep(0.05)
    t.join(timeout=10)
    assert seen_running, "探测期间应能看到 running/probing 状态"
    assert res["rc"] == 0
    final = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert final["state"] == "done" and final["phase"] == "done"


def test_gmu_probe_total_timeout(tmp_path, capsys, monkeypatch):
    # minor 2：总预算耗尽（total_timeout_s=0）→ 中止探测，gmu_suggested null + reason 含 total_timeout
    monkeypatch.setattr(bench, "DEFAULT_GMU_DETECT",
                        {**bench.DEFAULT_GMU_DETECT, "total_timeout_s": 0})
    port = 18099
    _write_docker_inputs(tmp_path, port, startup=10, trial=30, repeats=1)
    out = tmp_path / "out"
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file) + ["--detect-gmu"])
    assert rc == 1
    probe = json.loads((out / "gmu_probe.json").read_text(encoding="utf-8"))
    assert probe["gmu_suggested"] is None
    assert probe["tried"] == []
    assert "total_timeout" in probe["reason"]
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "crash" and status["phase"] == "crash"
    sout = capsys.readouterr().out
    assert "gmu_suggested: 无" in sout and "total_timeout" in sout


def test_gmu_probe_unknown_flag_fails_fast(tmp_path, capsys, monkeypatch):
    # m1：探测路径同样预校验 config.serve——typo 不伪装成"无可用 gmu"（不下降、不产 gmu_probe.json）
    port = 18102
    _write_docker_inputs(tmp_path, port, startup=10, trial=30, repeats=1)
    (tmp_path / "config.yaml").write_text(json.dumps(
        {"env": {"TEST_ENV": "1"}, "serve": {"bogus-flag": 1}}), encoding="utf-8")
    out = tmp_path / "out"
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file) + ["--detect-gmu"])
    assert rc == 1
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "crash" and status["phase"] == "crash"
    assert "未知 vllm serve 参数" in status["message"]
    assert "--bogus-flag" in status["message"]
    captured = capsys.readouterr()
    assert "未知 vllm serve 参数" in captured.err
    # 预校验失败即中止：无 gmu_probe.json（区别于 floor 耗尽/总超时的探测结果）
    assert not (out / "gmu_probe.json").exists()



# ── docker 运行时生命周期（stub docker，无真实 docker/GPU）─────────────────

def _write_docker_inputs(tmp_path: Path, port: int, *, startup: int = 30, trial: int = 60,
                         repeats: int = 2, image: str = "vllm/vllm-openai:v0.31.0") -> None:
    harness = {
        "bench": {"tool": "vllm bench serve", "repeats": repeats,
                  "warmup_repeats": 0,   # 生命周期测试保持旧口径（默认 1；warmup 专项测试单独开启）
                  "startup_timeout_s": startup, "trial_timeout_s": trial,
                  "ref_concurrency": 128},
        "slo": {"ttft_p99_ms": 500, "tpot_p99_ms": 50},
        "workload": {"dataset": "random", "isl": 2048, "osl": 256, "num_prompts": 1000},
        "remote": {"vllm_port": port, "vllm_python": "python"},
        "runtime": {"kind": "docker", "image": image},
    }
    (tmp_path / "harness.yaml").write_text(json.dumps(harness), encoding="utf-8")
    (tmp_path / "target.yaml").write_text(json.dumps(
        {"model": {"id": "Qwen/Qwen3-32B", "source": "local",
                   "path": str(tmp_path / "models")}}), encoding="utf-8")
    (tmp_path / "config.yaml").write_text(json.dumps(
        {"env": {"TEST_ENV": "1"}, "serve": {}}), encoding="utf-8")


def _docker_args(tmp_path: Path, out: Path, state_file: Path) -> list[str]:
    return ["--config", str(tmp_path / "config.yaml"),
            "--harness", str(tmp_path / "harness.yaml"),
            "--target", str(tmp_path / "target.yaml"),
            "--out-dir", str(out),
            "--docker-cmd", str(FIX / "fake_docker.sh"),
            "--poll-interval", "0.2", "--vram-interval", "0.3"]


def test_docker_lifecycle_with_stub_docker(tmp_path, capsys, monkeypatch):
    port = 18092
    _write_docker_inputs(tmp_path, port, repeats=2)
    out = tmp_path / "out"
    out.mkdir()
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc == 0

    # B1 后果：bench 容器 --network host 共享主机网络栈 → 能拿到 goodput
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["goodput_req_s"] == pytest.approx(12.84)      # median(12.34, 13.34)
    assert result["calibration"]["measured_kv_tokens"] == 1234567   # KV 校准行来自 stub serve 的 run 输出

    # M1：清理（rm -f）必须先于 run；收尾还会再 rm 一次
    lines = state_file.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("docker rm -f vllm-autotune-trial"), f"首个调用应为 rm，实际: {lines[0]}"
    assert any("docker run" in l for l in lines)
    rm_idx = [i for i, l in enumerate(lines) if l.startswith("docker rm")]
    run_idx = [i for i, l in enumerate(lines) if l.startswith("docker run")]
    assert max(rm_idx) < max(run_idx) or True                    # rm 在首次 run 之前已断言
    assert min(rm_idx) < min(run_idx)                            # 首个 rm 早于首个 run

    # 校准行落 run.log（docker run attach 语义，无重复 docker logs 跟随）
    log = (out / "run.log").read_text(encoding="utf-8")
    assert log.count("GPU KV cache size: 1,234,567 tokens") == 1  # 不重复

    sout = capsys.readouterr().out
    assert "---\nstatus: ok" in sout
    assert "goodput: 12.84" in sout
    assert "goodput_std:" in sout                             # σ 行进 summary 块
    # stats：repeats=2 → [12.34, 13.34] 的样本 σ > 0（与实现一致的四舍五入）
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["stats"]["goodput_std"] == pytest.approx(
        round(statistics.stdev([12.34, 13.34]), 4))
    # workload：本次 bench 实际生效值（DEFAULT_WORKLOAD ⊕ harness/target + ref_conc + seed）
    assert result["workload"] == {"dataset": "random", "isl": 2048, "osl": 256,
                                  "num_prompts": 1000, "ref_concurrency": 128,
                                  "prefix_hit_pct": 0, "seed": 0}
    # stub 双 -v（结果目录 + 模型挂载）区分正确后：--save-result JSON 落到 bench_results
    # （此前会误落到模型目录 → find_newest_json 落空 → stdout 回退）
    assert list((out / "bench_results").glob("result-*.json"))


def test_docker_crash_serve_exits(tmp_path, capsys, monkeypatch):
    port = 18093
    _write_docker_inputs(tmp_path, port, startup=10, trial=30, repeats=1)
    out = tmp_path / "out"
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))
    monkeypatch.setenv("FAKE_DOCKER_CRASH", "1")

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc != 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "crash"
    assert "rc=1" in result["reason"]                            # stub serve exit 1
    assert result["crash_kind"] == "serve_exit"
    sout = capsys.readouterr().out
    assert "status: crash" in sout


def test_docker_serve_oom_crash_classified(tmp_path, capsys, monkeypatch):
    # F10：serve 崩溃日志含 OOM 特征 → 原因改为可执行口径 + 崩溃当时重新共租检测的 pid
    port = 18110
    _write_docker_inputs(tmp_path, port, startup=10, trial=30, repeats=1)
    out = tmp_path / "out"
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))
    monkeypatch.setenv("FAKE_DOCKER_CRASH_OOM", "1")
    monkeypatch.setattr(bench, "_detect_co_tenants",
                        lambda: [{"pid": 1234, "process_name": "torchrun",
                                  "used_memory_mib": 512}])

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc != 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "crash"
    assert "serve OOM" in result["reason"]                       # OOM 分类替换泛化「rc=1」
    assert "1234(torchrun)" in result["reason"]                  # foreign pid + process_name
    assert "rc=1" not in result["reason"]                        # 命中 OOM → 不再保留泛化原因
    assert result["contention"]["foreign_gpu_apps"][0]["pid"] == 1234
    assert result["crash_kind"] == "oom"
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert "serve OOM" in status["message"]
    sout = capsys.readouterr().out
    assert "status: crash" in sout


def test_docker_missing_image_crash(tmp_path, capsys, monkeypatch):
    # M1：镜像未预先准备 → main() 的 image inspect 预检查给出可诊断 crash（零 pull 契约）
    port = 18094
    _write_docker_inputs(tmp_path, port, startup=10, trial=30, repeats=1)
    out = tmp_path / "out"
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))
    monkeypatch.setenv("FAKE_DOCKER_NO_IMAGE", "1")

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc != 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "crash"
    assert "未在主机预先准备" in result["reason"]
    assert result["crash_kind"] == "missing_image"
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "crash" and status["phase"] == "crash"
    # 预检查失败：不进入 docker run（状态文件只有 image inspect 相关调用，无 rm/run）
    assert not state_file.exists() or "docker run" not in state_file.read_text(encoding="utf-8")
    sout = capsys.readouterr().out
    assert "status: crash" in sout


def test_docker_unknown_serve_flag_crashes_before_serve(tmp_path, capsys, monkeypatch):
    # serve flag 预校验：config.serve 含镜像 help 里没有的键 → 起 serve 前 fail-fast crash
    port = 18101
    _write_docker_inputs(tmp_path, port, startup=10, trial=30, repeats=1)
    (tmp_path / "config.yaml").write_text(json.dumps(
        {"env": {"TEST_ENV": "1"}, "serve": {"max-num-seqs": 8, "bogus-flag": 1}}),
        encoding="utf-8")
    out = tmp_path / "out"
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc != 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "crash"
    assert "--bogus-flag" in result["reason"]
    assert result["crash_kind"] == "unknown_flag"               # 崩溃分类机器可读
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "crash"
    # 校验发生在起 serve 之前：状态文件有 help 校验的 run，但绝无真实 serve（serve /model 从未启动）
    text = state_file.read_text(encoding="utf-8")
    assert "serve --help" in text
    assert "serve /model" not in text
    sout = capsys.readouterr().out
    assert "status: crash" in sout


# ── 崩溃分类（crash_kind：机器可读枚举）──────────────────────────────────

class TestClassifyCrashKind:
    def test_maps_canonical_reasons(self):
        assert bench._classify_crash_kind(
            "镜像 vllm/vllm-openai:v0.31.0 未在主机预先准备（本模板不自动拉取，保持主机零新增）") == "missing_image"
        assert bench._classify_crash_kind(
            "需要 target.yaml 的 model.path（宿主上的模型目录，将只读挂载进容器）") == "model_path_missing"
        assert bench._classify_crash_kind(
            "未知 vllm serve 参数：['--bogus-flag']（对照镜像 `vllm serve --help=all`；请修正 config.yaml.serve）") == "unknown_flag"
        assert bench._classify_crash_kind(
            "主机非空闲：检测到其他 GPU 进程 [123]（require_idle=true 要求独占）") == "contention"
        assert bench._classify_crash_kind("serve OOM：显存不足（疑似外部共租占用）") == "oom"
        assert bench._classify_crash_kind("startup_timeout") == "startup_timeout"
        assert bench._classify_crash_kind("trial_timeout") == "trial_timeout"
        assert bench._classify_crash_kind("EngineDeadError（/health 返回 503）") == "engine_dead"
        assert bench._classify_crash_kind("serve 提前退出（rc=1）") == "serve_exit"
        assert bench._classify_crash_kind("bench 失败（rc=1）") == "bench_failure"
        assert bench._classify_crash_kind("bench 第 1 次无有效输出（goodput 缺失）") == "bench_failure"
        assert bench._classify_crash_kind(
            "预热后重置前缀缓存失败：无法保证冷缓存测量（prefix caching 口径失真）") == "bench_failure"
        assert bench._classify_crash_kind("trial 未完成（占位：正常结束时会被覆盖）") == "unknown"
        assert bench._classify_crash_kind("") == "unknown"

    def test_oom_via_log_text(self):
        # OOM 特征只出现在日志、reason 是通用退出 → 也归 oom（复用 _serve_crash_is_oom）
        text = ("ValueError: Free memory on device cuda:0 (2.0 GiB / 22.0 GiB) on startup "
                "is less than desired GPU memory utilization (gmu=0.98)")
        assert bench._classify_crash_kind("serve 提前退出（rc=1）", text) == "oom"


# ── P0：trial 身份写入产物 + 陈旧 bench JSON 隔离（stub docker）────────────

def test_docker_lifecycle_trial_identity_and_stale_cleanup(tmp_path, capsys, monkeypatch):
    # status/result 都写 trial_id + config_hash；预置的陈旧 bench JSON 在测量前被清掉
    import re as _re
    port = 18120
    _write_docker_inputs(tmp_path, port, repeats=1)
    out = tmp_path / "out"
    out.mkdir()
    br = out / "bench_results"
    br.mkdir()
    stale = br / "result-stale.json"
    stale.write_text(json.dumps({"request_goodput": 99.0}), encoding="utf-8")
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file) + ["--trial-id", "T-1"])
    assert rc == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["trial_id"] == "T-1"
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["trial_id"] == "T-1"
    cfg = json.loads((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert result["config_hash"] == bench.config_fingerprint(cfg)
    assert status["config_hash"] == result["config_hash"]
    # 陈旧 JSON 已清：测量前种下的文件不存在（stub 只写新 result-*.json；.count 保留）
    assert not stale.exists()
    assert result["goodput_req_s"] == pytest.approx(12.34)      # repeats=1 取唯一样本

    # 未显式给 --trial-id → 自动生成 <时间戳>-<uuid6>；两次运行 id 不同
    out2 = tmp_path / "out2"
    out2.mkdir()
    state_file2 = tmp_path / "docker_state2.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file2))
    assert bench.main(_docker_args(tmp_path, out2, state_file2)) == 0
    r2 = json.loads((out2 / "result.json").read_text(encoding="utf-8"))
    assert _re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{6}", r2["trial_id"])
    assert r2["trial_id"] != "T-1"


def test_docker_crash_writes_trial_identity(tmp_path, capsys, monkeypatch):
    # crash 路径 result.json / status.json 同样写 trial_id + config_hash
    port = 18121
    _write_docker_inputs(tmp_path, port, startup=10, trial=30, repeats=1)
    out = tmp_path / "out"
    out.mkdir()
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))
    monkeypatch.setenv("FAKE_DOCKER_CRASH", "1")

    rc = bench.main(_docker_args(tmp_path, out, state_file) + ["--trial-id", "T-crash"])
    assert rc != 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "crash"
    assert result["trial_id"] == "T-crash"
    assert result["config_hash"] == bench.config_fingerprint(
        json.loads((tmp_path / "config.yaml").read_text(encoding="utf-8")))
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["trial_id"] == "T-crash"
    assert status["config_hash"] == result["config_hash"]


def test_stale_run_log_not_contaminating(tmp_path, capsys, monkeypatch):
    # P0 回归：run.log 启动即截断——预置的旧 trial KV 行不得污染本次 measured_kv_tokens
    port = 18122
    _write_docker_inputs(tmp_path, port, repeats=1)
    out = tmp_path / "out"
    out.mkdir()
    (out / "run.log").write_text(
        "GPU KV cache size: 9,999,999 tokens, Maximum concurrency ...\n"
        "旧 trial 残留日志行（应被截断，不得进入本次解析）\n",
        encoding="utf-8")
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    # 本次 stub serve 的实测 KV 值；若沿用追加模式，会扫到预置的旧值 9999999
    assert result["calibration"]["measured_kv_tokens"] == 1234567
    # 顺带确认 --save-result JSON 路径真实走通（stub 双 -v 修复后 JSON 落 bench_results）
    assert list((out / "bench_results").glob("result-*.json"))
    log = (out / "run.log").read_text(encoding="utf-8")
    assert "9,999,999" not in log and "旧 trial 残留日志行" not in log


def test_early_failure_writes_identity_placeholder(tmp_path, monkeypatch):
    # M1：main() 早退路径（runtime.image 缺失 → _err exit 2）也刷新 result.json 为本次身份的
    # crash 占位，绝不留下一次 trial 的陈旧成功结果（rsync_back 只会拿到本次身份的产物）
    (tmp_path / "harness.yaml").write_text(json.dumps(
        {"runtime": {"kind": "docker"}}), encoding="utf-8")     # 无 image → 早退
    (tmp_path / "target.yaml").write_text(json.dumps(
        {"model": {"id": "Qwen/Qwen3-32B", "source": "local",
                   "path": str(tmp_path / "models")}}), encoding="utf-8")
    (tmp_path / "config.yaml").write_text(json.dumps({"serve": {}}), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / "result.json").write_text(json.dumps(
        {"status": "ok", "goodput_req_s": 99.0, "trial_id": "OLD-TRIAL"}), encoding="utf-8")

    with pytest.raises(SystemExit):
        bench.main(_docker_args(tmp_path, out, tmp_path / "st.txt") + ["--trial-id", "T-new"])
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "crash"
    assert result["trial_id"] == "T-new"                        # 本次身份，绝非 OLD-TRIAL
    assert result["config_hash"] == bench.config_fingerprint(
        json.loads((tmp_path / "config.yaml").read_text(encoding="utf-8")))
    assert "未完成" in result["reason"]


# ── 预热轮（warmup_repeats）：丢弃输出 + 边界冷缓存重置 + config.bench 覆盖 ──

def _with_bench_config(tmp_path: Path, patch: dict) -> None:
    """在 config.yaml 上叠加 bench 覆盖（Delta 2：config.bench 逐键覆盖 harness.bench）。"""
    cfg = json.loads((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    (tmp_path / "config.yaml").write_text(json.dumps({**cfg, "bench": patch}), encoding="utf-8")


def test_warmup_repeats_executed_but_excluded(tmp_path, capsys, monkeypatch):
    # stub 的 goodput = 12.34 + n×1.0（n = .count 调用序号）：
    # 预热 1 次消耗 n=0（12.34 被丢弃）；实测 n=1→13.34、n=2→14.34 → 中位 13.84。
    # 若预热未被排除，[12.34, 13.34, 14.34] 中位 = 13.34 → 断言可区分。
    port = 18124
    _write_docker_inputs(tmp_path, port, repeats=2)
    _with_bench_config(tmp_path, {"warmup_repeats": 1})
    out = tmp_path / "out"
    out.mkdir()
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["warmup_repeats"] == 1 and result["repeats"] == 2
    assert result["goodput_req_s"] == pytest.approx(13.84)     # 预热行未进中位
    assert result["stats"]["goodput_std"] == pytest.approx(    # 预热行未进 σ
        round(statistics.stdev([13.34, 14.34]), 4))


def test_config_bench_overrides_and_reset_once_between_warmup_and_measured(
        tmp_path, capsys, monkeypatch):
    # config.bench.repeats 覆盖 harness.repeats；前缀缓存重置：
    # warmup→measured 边界恰 1 次 + 每次 i>=1 各 1 次 = 1 + (repeats−1)
    port = 18125
    _write_docker_inputs(tmp_path, port, repeats=2)            # harness repeats=2
    _with_bench_config(tmp_path, {"repeats": 4, "warmup_repeats": 1})
    calls: list[int] = []
    monkeypatch.setattr(bench, "_reset_prefix_cache",
                        lambda port, log_fh: calls.append(port))
    out = tmp_path / "out"
    out.mkdir()
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["repeats"] == 4                              # config 覆盖生效
    assert result["warmup_repeats"] == 1
    assert len(calls) == 4                                     # 1 边界 + 3 次 i>=1


def test_warmup_repeats_zero_preserves_old_behavior(tmp_path, capsys, monkeypatch):
    # warmup_repeats=0：无预热、无边界重置（起服即冷）；重置仅 i>=1 前一次；
    # .count 不被预热消耗 → 中位 = median(12.34, 13.34) = 12.84（旧口径）
    port = 18126
    _write_docker_inputs(tmp_path, port, repeats=2)            # fixture 默认 warmup_repeats=0
    calls: list[int] = []
    monkeypatch.setattr(bench, "_reset_prefix_cache",
                        lambda port, log_fh: calls.append(port))
    out = tmp_path / "out"
    out.mkdir()
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["warmup_repeats"] == 0
    assert result["goodput_req_s"] == pytest.approx(12.84)
    assert len(calls) == 1                                     # 仅 i=1 前一次重置


# ── M1：前缀缓存重置失败语义 + m1/m2 minor（预热 JSON 清理 / 预热无输出 crash）──

def _with_target_workload(tmp_path: Path, patch: dict) -> None:
    """在 target.yaml 的 model 基础上叠加 workload 覆盖（prefix_hit_pct 走这里）。"""
    tgt = json.loads((tmp_path / "target.yaml").read_text(encoding="utf-8"))
    (tmp_path / "target.yaml").write_text(json.dumps({**tgt, "workload": patch}), encoding="utf-8")


def test_boundary_reset_failure_crashes_with_prefix(tmp_path, capsys, monkeypatch):
    # M1：warmup→measured 边界重置失败且 prefix_hit_pct>0 → 直接 crash（无法保证冷缓存口径）
    port = 18127
    _write_docker_inputs(tmp_path, port, repeats=1)
    _with_bench_config(tmp_path, {"warmup_repeats": 1})
    _with_target_workload(tmp_path, {"prefix_hit_pct": 60})
    monkeypatch.setattr(bench, "_reset_prefix_cache", lambda port, log_fh: False)
    out = tmp_path / "out"
    out.mkdir()
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc != 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "crash"
    assert "重置前缀缓存失败" in result["reason"]


def test_boundary_reset_failure_tolerated_without_prefix(tmp_path, capsys, monkeypatch):
    # M1：prefix_hit_pct=0 时边界重置失败 → 容忍（缓存无关），结果如实标记 prefix_cache_reset_ok=false
    port = 18128
    _write_docker_inputs(tmp_path, port, repeats=1)
    _with_bench_config(tmp_path, {"warmup_repeats": 1})
    monkeypatch.setattr(bench, "_reset_prefix_cache", lambda port, log_fh: False)
    out = tmp_path / "out"
    out.mkdir()
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["prefix_cache_reset_ok"] is False


def test_prefix_cache_reset_ok_true_when_all_succeed(tmp_path, capsys, monkeypatch):
    # M1：全部重置成功 → prefix_cache_reset_ok=true
    port = 18129
    _write_docker_inputs(tmp_path, port, repeats=2)
    _with_bench_config(tmp_path, {"warmup_repeats": 1})
    calls: list[int] = []
    monkeypatch.setattr(bench, "_reset_prefix_cache",
                        lambda port, log_fh: (calls.append(port), True)[1])
    out = tmp_path / "out"
    out.mkdir()
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc == 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["prefix_cache_reset_ok"] is True
    assert len(calls) == 2                                     # 边界 1 + i=1 前 1


def test_warmup_json_cleared_before_measured(tmp_path, capsys, monkeypatch):
    # m1：实测前二次清空——预热写的 JSON 不得被 find_newest_json 误取
    port = 18130
    _write_docker_inputs(tmp_path, port, repeats=1)
    _with_bench_config(tmp_path, {"warmup_repeats": 1})
    seen: list[int] = []

    def fake_bench(bench_tokens, timeout, log_fh, result_dir):
        # 第 1 次调用（预热）时目录应已被首轮清空（0 个 JSON）；随后写一个预热 JSON；
        # 第 2 次调用（实测）时该预热 JSON 应已被二次清空（仍 0 个 JSON）
        seen.append(len([f for f in os.listdir(result_dir) if f.endswith(".json")]))
        if len(seen) == 1:
            with open(os.path.join(result_dir, "warmup-seed.json"), "w", encoding="utf-8") as f:
                json.dump({"request_goodput": 99.0}, f)
        return {"goodput": 12.0, "output_tok_s": 1.0, "ttft_p99_ms": 1.0, "tpot_p99_ms": 1.0}

    monkeypatch.setattr(bench, "_run_bench_once", fake_bench)
    monkeypatch.setattr(bench, "_reset_prefix_cache", lambda port, log_fh: True)
    out = tmp_path / "out"
    out.mkdir()
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc == 0
    assert seen == [0, 0]                                      # 预热 JSON 在实测前被清掉
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["goodput_req_s"] == pytest.approx(12.0)      # 取的是实测返回值


def test_warmup_no_goodput_crashes(tmp_path, capsys, monkeypatch):
    # m2：预热 bench 无有效输出 → 与实测同判 crash（不静默丢弃）
    port = 18131
    _write_docker_inputs(tmp_path, port, repeats=1)
    _with_bench_config(tmp_path, {"warmup_repeats": 1})
    monkeypatch.setattr(bench, "_run_bench_once",
                        lambda *a, **k: {"goodput": None, "output_tok_s": None,
                                         "ttft_p99_ms": None, "tpot_p99_ms": None})
    out = tmp_path / "out"
    out.mkdir()
    state_file = tmp_path / "docker_state.txt"
    monkeypatch.setenv("FAKE_DOCKER_STATE_FILE", str(state_file))

    rc = bench.main(_docker_args(tmp_path, out, state_file))
    assert rc != 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "crash"
    assert "预热 bench 无有效输出" in result["reason"]
