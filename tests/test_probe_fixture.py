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
    assert head["gpu"]["count"] == 8
    assert head["gpu"]["topo"]["summary"] == "pcie"     # 新 fixture：全 PCIe 矩阵（PXB/NODE/SYS，无 NV）
    assert head["gpu"]["gpus"][0]["numa_node"] == 0     # NUMA 放置合并进 GPU 事实
    assert head["numa_groups"] == [
        {"numa_node": 0, "gpus": [0, 1, 2, 3]},
        {"numa_node": 1, "gpus": [4, 5, 6, 7]},
    ]
    # 8 卡与 topo/lscpu 一致（GPU0-7 → NUMA0/1，2 节点）——fixture 自洽
    assert head["cpu"]["numa_nodes"] == 2
    assert head["cpu"]["governor"] == "performance"
    assert head["docker"]["version"]                    # docker 检测（fixture docker_version.txt）
    assert any("vllm/vllm-openai" in i for i in (head["docker"]["vllm_images"] or []))
    assert isinstance(head["checklist"], list) and head["checklist"]  # 检查清单存在
    assert any(i["status"] == "pass" for i in head["checklist"])
    # NUMA 放置：2-NUMA 且每节点 >1 GPU → warn（仅建议，模板不自动绑定）
    numa_item = next(i for i in head["checklist"] if i["name"] == "NUMA 放置")
    assert numa_item["status"] == "warn"
    assert "numa_groups" in numa_item["detail"]
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
    assert "NUMA 分组" in report                       # 关键事实含 numa_groups
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


# ── docker-only 模型探测（_collect_plan 命令串选择）─────────────────────

def _head_plan(harness: dict | None = None, config: dict | None = None) -> dict:
    """构造单 head 主机（source=local）的采集计划条目 dict（key → entry）。"""
    target = {"hosts": [{"name": "head", "ssh": "user@10.0.0.1", "role": "head"}],
              "model": {"id": "m", "source": "local", "path": "/models/m"}}
    plan = probe._collect_plan(target, harness or {"runtime": {}}, config=config)
    return {e["key"]: e for e in plan["head"]}


def test_collect_plan_local_with_image_uses_docker_cmds():
    """docker-only：source=local + runtime.image 存在 → 模型探测与 vllm_version 走镜像内执行。"""
    harness = {"runtime": {"image": "vllm/vllm-openai:nightly"}}
    entries = _head_plan(harness, config={})
    for key in ("model_config_json", "model_safetensors_index_json", "df_model"):
        cmd = entries[key]["cmd"]
        assert "docker run" in cmd and "vllm/vllm-openai:nightly" in cmd
    # 镜像版本命令选择
    assert "docker run" in entries["vllm_version"]["cmd"]
    assert "vllm/vllm-openai:nightly" in entries["vllm_version"]["cmd"]
    # fixture 文件名保持不变（离线模式按 key→<key>.txt 读取）
    assert entries["model_config_json"]["fixture"] == "config.json"
    assert entries["model_safetensors_index_json"]["fixture"] == "safetensors.index.json"
    assert entries["df_model"]["fixture"] == "df_model.txt"


def test_collect_plan_local_without_image_uses_host_cmds():
    """无镜像（source=local）→ 保持宿主 cat/df/pip 命令，向后兼容。"""
    entries = _head_plan(config={})
    assert "docker run" not in entries["model_config_json"]["cmd"]
    assert "cat /models/m/config.json" in entries["model_config_json"]["cmd"]
    assert "docker run" not in entries["model_safetensors_index_json"]["cmd"]
    assert "*.safetensors.index.json" in entries["model_safetensors_index_json"]["cmd"]
    assert "docker run" not in entries["df_model"]["cmd"]
    assert "df -B1 --output=avail /models" in entries["df_model"]["cmd"]  # 宿主版取 dirname
    assert "docker run" not in entries["vllm_version"]["cmd"]
    assert "pip show vllm" in entries["vllm_version"]["cmd"]


def test_collect_plan_config_none_missing_config_yaml_tolerant(tmp_path, monkeypatch):
    """config=None 且仓库根无 config.yaml（测试在 tmp 目录跑）→ 容错为 {}，不抛错。"""
    monkeypatch.chdir(tmp_path)  # tmp 目录没有 config.yaml
    entries = _head_plan(config=None)
    assert "docker run" not in entries["model_config_json"]["cmd"]
    assert "cat /models/m/config.json" in entries["model_config_json"]["cmd"]


def test_collect_plan_config_runtime_overrides_harness():
    """逐键合并：config.yaml.runtime.image 覆盖 harness.runtime.image。"""
    harness = {"runtime": {"image": "img-a"}}
    config = {"runtime": {"image": "img-b"}}
    entries = _head_plan(harness, config=config)
    assert "img-b" in entries["model_config_json"]["cmd"]
    assert "img-a" not in entries["model_config_json"]["cmd"]


def test_collect_plan_hf_source_unchanged_with_image():
    """source=hf 即使有镜像也走宿主 HF 缓存命令（行为不变）。"""
    target = {"hosts": [{"name": "head", "ssh": "user@10.0.0.1", "role": "head"}],
              "model": {"id": "Qwen/Qwen3-32B", "source": "hf"}}
    plan = probe._collect_plan(target, {"runtime": {"image": "img"}}, config={})
    entries = {e["key"]: e for e in plan["head"]}
    assert "docker run" not in entries["model_config_json"]["cmd"]
    assert "huggingface/hub" in entries["model_config_json"]["cmd"]


def test_checklist_gpu_persistence_mode_is_warn_not_fail():
    """GPU 持久模式为「只建议、不自动改」的系统级旋钮 → fail 降级 warn，不阻断门禁。"""
    facts = {
        "cpu": {"governor": "performance"},
        "gpu": {"gpus": [{"index": 0, "persistence_mode": "Disabled",
                          "clocks_sm": 1800, "clocks_max_sm": 2000}]},
        "model_src": {"config_json": None, "note": None},
        "disk": {"avail_bytes": None},
        "mem": {"thp": "madvise", "swappiness": 10},
        "net": {"rdma": {"ibstat_present": False, "ib_devices": None}, "interfaces": []},
        "numa_groups": [{"numa_node": 0, "gpus": [0]}],
        "docker": {"version": "20.10"},
    }
    items = probe._system_checklist(facts, is_head=True, weights_bytes=None)
    pm = next(i for i in items if i["name"] == "GPU 持久模式")
    assert pm["status"] == "warn"
    assert pm["fix"] == "sudo nvidia-smi -pm 1"  # detail/fix 文案保留
