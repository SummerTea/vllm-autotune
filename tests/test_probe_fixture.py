"""test_probe_fixture.py — 端到端（离线 fixture 模式）集成测试。

在临时目录生成 target.yaml + harness.yaml，以 --fixtures tests/fixtures
运行 probe.py，断言三个产物（env.json / env_report.md / model_profile.json）
落盘且关键结构正确。
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "probe.py"
FIXTURES = REPO / "tests" / "fixtures"
HEAD = FIXTURES / "head"

MODEL = {"id": "Qwen/Qwen3-32B", "source": "local", "path": "/models/qwen3-32b"}

TARGET = {
    "hosts": [
        {"name": "head", "ssh": "user@10.0.0.1", "role": "head"},
    ],
    "model": MODEL,
}

TARGET_MULTI = {
    "hosts": [
        {"name": "head", "ssh": "user@10.0.0.1", "role": "head"},
        {"name": "worker1", "ssh": "user@10.0.0.2", "role": "worker"},
    ],
    "model": MODEL,
}


def _run_probe(tmp_path: Path, target: dict = TARGET) -> subprocess.CompletedProcess:
    target_p = tmp_path / "target.yaml"
    harness = tmp_path / "harness.yaml"
    target_p.write_text(yaml.safe_dump(target), encoding="utf-8")
    shutil.copy(REPO / "harness.yaml", harness)
    out = tmp_path / "out"
    cmd = [sys.executable, str(PROBE), "--fixtures", str(FIXTURES),
           "--target", str(target_p), "--harness", str(harness), "--out-dir", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_fixture_probe_produces_artifacts(tmp_path):
    r = _run_probe(tmp_path)
    assert r.returncode == 0, f"probe 失败:\n{r.stdout}\n{r.stderr}"

    env_p = tmp_path / "out" / "env.json"
    report_p = tmp_path / "out" / "env_report.md"
    profile_p = tmp_path / "out" / "model_profile.json"
    for p in (env_p, report_p, profile_p):
        assert p.exists(), f"缺产物: {p.name}"

    env = json.loads(env_p.read_text(encoding="utf-8"))
    assert env["mode"] == "fixtures"
    assert env["generated_at"]
    head = env["hosts"]["head"]
    assert head["role"] == "head"                       # head 主机被识别
    assert head["hostname"] == "head"                   # 来自 fixture hostname.txt
    assert head["vllm_version"] == "0.31.0"
    assert head["gpu"]["count"] == 2
    assert head["gpu"]["topo"]["summary"] == "nvlink-full"
    assert head["cpu"]["governor"] == "performance"
    assert isinstance(head["checklist"], list) and head["checklist"]  # 检查清单存在
    assert any(i["status"] == "pass" for i in head["checklist"])
    assert env["interconnect"]["covered"] is False      # 单主机 → 不执行跨机检查

    profile = json.loads(profile_p.read_text(encoding="utf-8"))
    assert profile["model"] == "Qwen/Qwen3-32B"
    assert profile["calibrated"] is False
    assert profile["weights"]["source"] == "safetensors-index"
    assert profile["weights"]["bytes"] == 29200000000   # 非 null
    assert 1 in profile["tp"]["valid_divisors"] and 8 in profile["tp"]["valid_divisors"]
    assert profile["kv"]["bytes_per_token"] == 262144
    assert profile["roofline"]["decode_tps_upper_bound"] > 0

    report = report_p.read_text(encoding="utf-8")
    assert "系统参数检查清单" in report
    assert "下一步" in report


def test_fixture_report_has_head_summary(tmp_path):
    r = _run_probe(tmp_path)
    assert r.returncode == 0
    assert "head (head)" in r.stdout                     # 汇总打印含主机名与角色
    assert "model_profile.json" in r.stdout


def test_missing_target_yaml_gives_clear_error(tmp_path):
    # 无 target.yaml 时必须给出复制指引而非晦涩报错；cwd 用空目录与仓库状态解耦
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(PROBE), "--out-dir", str(out)],
                       capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode != 0
    assert "target.example.yaml" in r.stderr


class TestValidateTarget:
    def test_requires_exactly_one_head(self):
        with pytest.raises(SystemExit):  # 无 head
            probe.validate_target({"hosts": [
                {"name": "w1", "ssh": "u@h", "role": "worker"}]})
        with pytest.raises(SystemExit):  # 重复 head
            probe.validate_target({"hosts": [
                {"name": "h1", "ssh": "u@h", "role": "head"},
                {"name": "h2", "ssh": "u@h", "role": "head"}]})

    def test_duplicate_names_rejected(self):
        with pytest.raises(SystemExit):  # 重名会在 plan/raw 字典中被静默覆盖
            probe.validate_target({"hosts": [
                {"name": "dup", "ssh": "u@h", "role": "head"},
                {"name": "dup", "ssh": "u@h2", "role": "worker"}]})

    def test_single_head_ok(self):
        h = probe.validate_target({"hosts": [
            {"name": "h1", "ssh": "u@h", "role": "head"},
            {"name": "w1", "ssh": "u@w", "role": "worker"}]})
        assert h["name"] == "h1"


def test_multihost_interconnect_emitted(tmp_path):
    # 两台主机 → 必须输出 interconnect 块，covered 反映 fixture 是否存在
    r = _run_probe(tmp_path, TARGET_MULTI)
    assert r.returncode == 0, f"probe 失败:\n{r.stdout}\n{r.stderr}"

    env = json.loads((tmp_path / "out" / "env.json").read_text(encoding="utf-8"))
    assert "head" in env["hosts"] and "worker1" in env["hosts"]
    assert env["hosts"]["worker1"]["hostname"] == "worker1"  # worker fixture 生效

    ic = env["interconnect"]
    assert "checks" in ic
    assert ic["covered"] is True                 # ping_worker1.txt fixture 存在
    assert any("ping worker1" in c["check"] and c["result"] for c in ic["checks"])
    assert ic["reason"] is None
