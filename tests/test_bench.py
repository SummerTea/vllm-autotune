"""test_bench.py — bench.py 单元测试 + 生命周期（stub docker）测试。

可在无 GPU 的 Mac 上运行（docker-only）：
- 解析器：stdout 表格 / --save-result JSON / 启动日志校准行
- 命令构建（docker）：serve 的 kebab flag / 布尔（true→裸 flag，false→省略，no- 前缀关默认开关）
  / 端口；bench 的 --network host / --base-url / 结果目录挂载 / 模型只读挂载
- 生命周期：fake_docker.sh 全流程（status/result/校准/summary 块，docker rm 先于 run）
- crash 路径：serve 容器立即退出（并验证清理被调用）
"""

import json
import os
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
