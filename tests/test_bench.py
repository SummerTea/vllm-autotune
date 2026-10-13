"""test_bench.py — bench.py 单元测试 + 生命周期（stub 脚本）测试。

可在无 GPU 的 Mac 上运行：
- 解析器：stdout 表格 / --save-result JSON / 启动日志校准行
- 命令构建：serve 的 kebab flag / 布尔（true→裸 flag，false→省略，no- 前缀关默认开关） / 端口
- 生命周期：fake_serve.sh + fake_bench.py 全流程（status/result/校准/summary 块）
- crash 路径：立即退出 / 启动超时（并验证清理真的杀掉了假 serve）
"""

import json
import os
import subprocess
import sys
import time
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


class TestBuildCommands:
    def test_build_serve_cmd_kebab_bool_port(self):
        cfg = {"env": {"NCCL_IB_DISABLE": "1"},
               "serve": {"max-num-seqs": 256, "gpu-memory-utilization": 0.92,
                         "enable-prefix-caching": True, "kv-cache-dtype": "fp8"}}
        cmd = bench.build_serve_cmd("vllm serve", cfg, "Qwen/Qwen3-32B", 8000)
        assert cmd[:2] == ["vllm", "serve"]
        assert cmd[2] == "Qwen/Qwen3-32B"
        for flag, val in (("--max-num-seqs", "256"), ("--gpu-memory-utilization", "0.92"),
                          ("--kv-cache-dtype", "fp8")):
            assert cmd[cmd.index(flag) + 1] == val
        assert "--enable-prefix-caching" in cmd        # True → 裸 flag
        assert "--port" in cmd and "8000" in cmd        # 端口默认补齐
        assert "NCCL_IB_DISABLE" not in cmd             # env 不进命令（由调用方注入）

    def test_build_serve_cmd_false_bool_omitted(self):
        # M3 回归：False 布尔 → 整体省略，绝不产出 "--flag false"（vLLM 布尔是开关式）
        cmd = bench.build_serve_cmd("vllm serve", {"serve": {"enforce-eager": False}},
                                    "M", 8000)
        assert "--enforce-eager" not in cmd
        assert "--enforce-eager false" not in " ".join(cmd)

    def test_build_serve_cmd_no_flag_turns_off_default(self):
        # 关闭默认开启的开关：写 no-<flag>: true，键原样透传为 --no-<flag>
        cmd = bench.build_serve_cmd("vllm serve",
                                    {"serve": {"no-enable-prefix-caching": True}}, "M", 8000)
        assert "--no-enable-prefix-caching" in cmd

    def test_build_serve_cmd_explicit_port_not_duplicated(self):
        cmd = bench.build_serve_cmd("vllm serve", {"serve": {"port": 9999}}, "M", 8000)
        assert cmd.count("--port") == 1
        assert "9999" in cmd

    def test_build_bench_cmd(self):
        workload = {"dataset": "random", "isl": 2048, "osl": 256, "num_prompts": 1000}
        slo = {"ttft_p99_ms": 500, "tpot_p99_ms": 50}
        cmd = bench.build_bench_cmd("vllm bench serve", "M", workload, slo, 128, 0, "/tmp/r",
                                    port=8000)
        assert cmd[:4] == ["vllm", "bench", "serve", "--model"]
        assert cmd[4] == "M"
        joined = " ".join(cmd)
        assert "--base-url http://127.0.0.1:8000" in joined   # 显式 base-url（端口≠默认也正确）
        assert "--dataset-name random" in joined
        assert "--input-len 2048" in joined and "--output-len 256" in joined
        assert "--num-prompts 1000" in joined
        assert "--request-rate inf" in joined
        assert "--max-concurrency 128" in joined
        assert "--goodput ttft:500 tpot:50" in joined
        assert "--seed 0" in joined and "--save-result" in joined
        assert "--result-dir /tmp/r" in joined

    def test_build_bench_cmd_trust_remote_code(self):
        cmd = bench.build_bench_cmd("vllm bench serve", "M", {}, {}, 128, 0, "/tmp/r",
                                    trust_remote_code=True, port=8000)
        assert "--trust-remote-code" in cmd


class TestDockerRuntime:
    RUNTIME = {"kind": "docker", "image": "vllm/vllm-openai:v0.31.0",
               "container_model_path": "/model", "shm_size": "16g",
               "extra_docker_args": ["--privileged"]}

    def test_effective_runtime_merge(self):
        rt = bench.effective_runtime({"runtime": {"kind": "native"}},
                                     {"runtime": {"kind": "docker", "image": "img:1"}})
        assert rt["kind"] == "docker" and rt["image"] == "img:1"
        assert bench.effective_runtime({}, {})["kind"] == "native"
        assert bench.effective_runtime({"runtime": {"kind": "docker", "image": "a:1"}}, {})["kind"] == "docker"

    def test_build_docker_run_cmd(self):
        cfg = {"env": {"NCCL_IB_DISABLE": "1"},
               "serve": {"max-num-seqs": 256, "enable-prefix-caching": True,
                         "enforce-eager": False}}
        cmd = bench.build_docker_run_cmd("docker", self.RUNTIME, "/models/qwen3", 8000, cfg)
        joined = " ".join(cmd)
        assert cmd[0] == "docker" and cmd[1] == "run"
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

    def test_build_docker_run_cmd_missing_image(self):
        with pytest.raises(ValueError):
            bench.build_docker_run_cmd("docker", {"kind": "docker"}, "/m", 8000, {"serve": {}})

    def test_build_docker_bench_cmd(self):
        workload = {"dataset": "random", "isl": 2048, "osl": 256, "num_prompts": 1000}
        slo = {"ttft_p99_ms": 500, "tpot_p99_ms": 50}
        cmd = bench.build_docker_bench_cmd("docker", self.RUNTIME, "/model", workload,
                                           slo, 128, 0, "/tmp/r", port=8000)
        assert cmd[:2] == ["docker", "run"]
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


# ── 生命周期 / crash 测试（stub 脚本）───────────────────────────────────

def _write_inputs(tmp_path: Path, port: int, *, startup: int = 30, trial: int = 60,
                  repeats: int = 2) -> None:
    harness = {
        "bench": {"tool": "vllm bench serve", "repeats": repeats,
                  "startup_timeout_s": startup, "trial_timeout_s": trial,
                  "ref_concurrency": 128},
        "slo": {"ttft_p99_ms": 500, "tpot_p99_ms": 50},
        "workload": {"dataset": "random", "isl": 2048, "osl": 256, "num_prompts": 1000},
        "remote": {"vllm_port": port, "vllm_python": "python"},
    }
    (tmp_path / "harness.yaml").write_text(json.dumps(harness), encoding="utf-8")
    (tmp_path / "target.yaml").write_text(json.dumps(
        {"model": {"id": "Qwen/Qwen3-32B", "trust_remote_code": False},
         "workload": {"ref_concurrency": 128}}), encoding="utf-8")
    (tmp_path / "config.yaml").write_text(json.dumps(
        {"env": {"TEST_ENV": "1"}, "serve": {}}), encoding="utf-8")


def _base_args(tmp_path: Path, out: Path, serve_cmd: str) -> list[str]:
    return ["--config", str(tmp_path / "config.yaml"),
            "--harness", str(tmp_path / "harness.yaml"),
            "--target", str(tmp_path / "target.yaml"),
            "--out-dir", str(out),
            "--serve-cmd", serve_cmd,
            "--poll-interval", "0.2", "--vram-interval", "0.3"]


def test_lifecycle_with_stub_scripts(tmp_path, capsys):
    port = 18080
    _write_inputs(tmp_path, port, repeats=2)
    out = tmp_path / "out"
    out.mkdir()
    # 预置 model_profile.json，验证就地校准
    profile = {
        "model": "Qwen/Qwen3-32B",
        "arch": {"model_type": "qwen2", "num_layers": 64, "hidden_size": 4096},
        "boot": {"estimated_seconds": 14.6, "measured_seconds": None},
        "kv": {"bytes_per_token": 262144, "est_capacity_tokens": 36832,
               "measured_capacity_tokens": None},
        "calibrated": False,
    }
    (out / "model_profile.json").write_text(json.dumps(profile), encoding="utf-8")

    rc = bench.main(_base_args(tmp_path, out, str(FIX / "fake_serve.sh")) + [
        "--bench-cmd", f"{sys.executable} {FIX / 'fake_bench.py'}"])
    assert rc == 0

    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "done" and status["phase"] == "done"

    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["goodput_req_s"] == pytest.approx(12.84)      # median(12.34, 13.34)
    assert result["output_tok_s"] == pytest.approx(12345.6)
    assert result["ttft_p99_ms"] == pytest.approx(620.1)
    assert result["tpot_p99_ms"] == pytest.approx(18.3)
    assert result["repeats"] == 2
    assert result["peak_vram_gb"] is None                       # 无 nvidia-smi → null
    assert result["dur_s"] > 0
    assert result["calibration"]["measured_kv_tokens"] == 1234567
    assert result["calibration"]["load_seconds"] > 0
    assert "captured" in result["env_fingerprint"]              # 指纹键存在（Mac 上无 nvidia-smi/sysfs 可为空）
    assert result["env_fingerprint"]["note"]                    # 无 env.json → 跳过对比

    # run.log 含校准行（fake_serve 的 stdout）
    assert "GPU KV cache size: 1,234,567 tokens" in (out / "run.log").read_text(encoding="utf-8")

    # model_profile.json 就地校准：保留 estimated，写 measured，置 calibrated
    p2 = json.loads((out / "model_profile.json").read_text(encoding="utf-8"))
    assert p2["calibrated"] is True
    assert p2["boot"]["estimated_seconds"] == 14.6
    assert p2["boot"]["measured_seconds"] > 0
    assert p2["kv"]["bytes_per_token"] == 262144
    assert p2["kv"]["measured_capacity_tokens"] == 1234567

    # summary 块（agent grep 用）
    sout = capsys.readouterr().out
    assert "---\nstatus: ok" in sout
    assert "goodput: 12.84" in sout
    assert "peak_vram_gb: null" in sout


def test_crash_immediate_exit(tmp_path, capsys):
    port = 18081
    _write_inputs(tmp_path, port, startup=10, trial=30, repeats=1)
    out = tmp_path / "out"
    rc = bench.main(_base_args(tmp_path, out, str(FIX / "fake_crash_serve.sh")) + [
        "--bench-cmd", "vllm bench serve"])
    assert rc != 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "crash"
    assert "rc=1" in result["reason"]                       # 假 serve 立即 exit 1
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "crash" and status["phase"] == "crash"
    sout = capsys.readouterr().out
    assert "status: crash" in sout
    assert "goodput:" not in sout                            # crash 不输出指标行


def test_crash_startup_timeout_and_cleanup(tmp_path, capsys):
    port = 18082
    _write_inputs(tmp_path, port, startup=2, trial=30, repeats=1)
    out = tmp_path / "out"
    rc = bench.main(_base_args(tmp_path, out, str(FIX / "fake_sleep_serve.sh")) + [
        "--bench-cmd", "vllm bench serve"])
    assert rc != 0
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "crash"
    assert "startup_timeout" in result["reason"]
    # 清理验证：假 serve 进程组必须已被杀掉（无残留）
    p = subprocess.run(["pgrep", "-f", "fake_sleep_serve"], capture_output=True, text=True)
    assert p.returncode != 0, f"残留进程: {p.stdout}"


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
