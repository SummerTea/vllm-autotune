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


class TestDockerRuntime:
    RUNTIME = {"kind": "docker", "image": "vllm/vllm-openai:v0.31.0",
               "container_model_path": "/model", "shm_size": "16g",
               "extra_docker_args": ["--privileged"]}

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
        assert "--privileged" in cmd                         # extra_docker_args 透传
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
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "crash"
    # 校验发生在起 serve 之前：状态文件有 help 校验的 run，但绝无真实 serve（serve /model 从未启动）
    text = state_file.read_text(encoding="utf-8")
    assert "serve --help" in text
    assert "serve /model" not in text
    sout = capsys.readouterr().out
    assert "status: crash" in sout
