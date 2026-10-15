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


def _run_probe(tmp_path: Path, target: dict = TARGET,
               config: dict | None = None) -> subprocess.CompletedProcess:
    target_p = tmp_path / "target.yaml"
    harness = tmp_path / "harness.yaml"
    target_p.write_text(yaml.safe_dump(target), encoding="utf-8")
    shutil.copy(REPO / "harness.yaml", harness)
    # config.yaml 由测试自带并以其为 cwd：probe 从 cwd 读取 config.yaml（决定 runtime.image →
    # 是否采集 docker 路径的键）。若不自带，测试会隐式耦合「仓库实例的 config.yaml」——
    # 模板仓库 config.yaml 无 runtime 时会导致 docker 相关键「未采集」。
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config if config is not None else {"runtime": {"image": "vllm/vllm-openai:v0.31.0"}}),
        encoding="utf-8")
    out = tmp_path / "out"
    cmd = [sys.executable, str(PROBE), "--fixtures", str(FIXTURES),
           "--target", str(target_p), "--harness", str(harness), "--out-dir", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_path)


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
    # RDMA：fixture 无 ibstat，但有 sysfs mlx5_* + link_layer Ethernet → 必须判 RoCE（防假阴性回归）
    rdma = head["net"]["rdma"]
    assert rdma["kind"] == "roce"
    assert rdma["ibstat_present"] is False              # 兼容既有字段保留
    assert len(rdma["devices"]) == 2
    assert rdma["devices"][0] == {
        "name": "mlx5_0", "netdev": "enp24s0f0np0", "link_layer": "Ethernet",
        "state": "ACTIVE", "physical_state": "LINK_UP", "speed_gbps": 400,
        "numa_node": 0,  # 来自 /sys/class/infiniband/<dev>/device/numa_node
        "gpu_links": {"0": "PXB", "1": "PXB", "2": "PXB", "3": "NODE",
                      "4": "PXB", "5": "PXB", "6": "PXB", "7": "NODE"},
    }
    assert rdma["devices"][1]["numa_node"] == 1         # sysfs numa_node 生效
    assert head["disk"]["read_bw_gbps"] == pytest.approx(2.3 * 8, abs=0.01)  # dd 真机逐字输出 → Gbps
    assert head["disk"]["model_path"] == "/models/qwen3-32b"
    assert isinstance(head["checklist"], list) and head["checklist"]  # 检查清单存在
    assert any(i["status"] == "pass" for i in head["checklist"])
    # NUMA 放置：2-NUMA 且每节点 >1 GPU → warn（仅建议，模板不自动绑定）
    numa_item = next(i for i in head["checklist"] if i["name"] == "NUMA 放置")
    assert numa_item["status"] == "warn"
    assert "numa_groups" in numa_item["detail"]
    assert env["interconnect"]["covered"] is False      # 单主机 → 不执行跨机检查
    # 顶层 interconnect 只保留「跨机互联检查」语义（covered/reason/checks），不再承载带宽
    assert set(env["interconnect"]) == {"covered", "reason", "checks"}
    # 本机互联带宽 = host 级事实（hosts.head.interconnect；fixture 逐字样例 → 解析出数值）
    hic = head["interconnect"]
    assert hic["gpu_p2p_bw_gbps"] == {"intra_numa": 43.35, "cross_numa": 33.87}
    assert hic["rdma_bw_gbps"] == {"bw_gbps": 341.7}

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
    assert "RDMA：RoCE(Ethernet) ×2（400 Gb/s）" in report  # 报告显示 RoCE 而非「无」（假阴性回归）
    assert "互联带宽：" in report                      # 主机小节读到互联带宽（host 级事实）
    assert "GPU P2P 带宽（容器实测）" not in report    # 跨机小节不再承载带宽数值
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
    # worker1 无任何 RDMA fixture → kind=none（无 RDMA 判 none 回归）
    assert env["hosts"]["worker1"]["net"]["rdma"]["kind"] == "none"
    # 本机互联带宽：每台主机都产出该键（host 级事实）；worker 无 fixture → null + note
    for name in ("head", "worker1"):
        hic = env["hosts"][name]["interconnect"]
        assert set(hic) == {"gpu_p2p_bw_gbps", "rdma_bw_gbps"}
    wic = env["hosts"]["worker1"]["interconnect"]
    assert wic["gpu_p2p_bw_gbps"]["intra_numa"] is None
    assert "note" in wic["gpu_p2p_bw_gbps"]
    assert wic["rdma_bw_gbps"]["bw_gbps"] is None
    assert "note" in wic["rdma_bw_gbps"]

    ic = env["interconnect"]
    assert "checks" in ic
    assert ic["covered"] is True                 # ping_worker1.txt fixture 存在
    assert any("ping worker1" in c["check"] and c["result"] for c in ic["checks"])
    assert ic["reason"] is None
    # 顶层 interconnect 只承载跨机检查语义，不再放带宽数值
    assert set(ic) == {"covered", "reason", "checks"}


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
    # 互联实测：有镜像 → gpu_p2p_bw 走容器测例；rdma_bw 恒为 head 采集项
    assert "docker run" in entries["gpu_p2p_bw"]["cmd"]
    assert entries["gpu_p2p_bw"]["fixture"] == "gpu_p2p_bw.txt"
    assert "rdma_bw" in entries and entries["rdma_bw"]["fixture"] == "rdma_bw.txt"


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
    # 无镜像 → 无 gpu_p2p_bw 键（不跑容器测例）；rdma_bw 仍在（脚本自含守卫）
    assert "gpu_p2p_bw" not in entries
    assert "rdma_bw" in entries


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


def test_collect_plan_interconnect_cmds_for_all_hosts():
    """本机互联带宽是 host 级事实：每台主机（含 worker）都产 gpu_p2p_bw/rdma_bw 采集项；
    无镜像 → 无 gpu_p2p_bw（容器测例不跑），rdma_bw 恒在。"""
    target = {"hosts": [
        {"name": "head", "ssh": "user@10.0.0.1", "role": "head"},
        {"name": "worker1", "ssh": "user@10.0.0.2", "role": "worker"},
    ], "model": {"id": "m", "source": "local", "path": "/models/m"}}
    plan = probe._collect_plan(target, {"runtime": {}}, config={})
    for name in ("head", "worker1"):
        keys = {e["key"] for e in plan[name]}
        assert "rdma_bw" in keys                      # 脚本自含守卫，恒采集
        assert "gpu_p2p_bw" not in keys               # 无镜像 → 不跑容器测例
    plan2 = probe._collect_plan(target, {"runtime": {"image": "img"}}, config={})
    for name in ("head", "worker1"):
        keys = {e["key"] for e in plan2[name]}
        assert "gpu_p2p_bw" in keys and "rdma_bw" in keys


def test_checklist_gpu_persistence_mode_is_warn_not_fail():
    """GPU 持久模式为「只建议、不自动改」的系统级旋钮 → fail 降级 warn，不阻断门禁。"""
    facts = {
        "cpu": {"governor": "performance"},
        "gpu": {"gpus": [{"index": 0, "persistence_mode": "Disabled",
                          "clocks_sm": 1800, "clocks_max_sm": 2000}]},
        "model_src": {"config_json": None, "note": None},
        "disk": {"avail_bytes": None},
        "mem": {"thp": "madvise", "swappiness": 10},
        "net": {"rdma": {"kind": "none", "devices": [], "ibstat_present": False, "ib_devices": None},
                "interfaces": []},
        "numa_groups": [{"numa_node": 0, "gpus": [0]}],
        "docker": {"version": "20.10"},
    }
    items = probe._system_checklist(facts, is_head=True, weights_bytes=None)
    pm = next(i for i in items if i["name"] == "GPU 持久模式")
    assert pm["status"] == "warn"
    assert pm["fix"] == "sudo nvidia-smi -pm 1"  # detail/fix 文案保留


# ── RDMA 检测（sysfs 主判据，缺 ibstat 不假阴性）────────────────────────

# 真机逐字输出（node-user02@10.23.69.131）——必须能用这套字符串解析出全部字段
REAL_RDMA_LINK = (
    "link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev enp24s0f0np0 \n"
    "link mlx5_1/1 state ACTIVE physical_state LINK_UP netdev enp24s0f1np1 \n"
)
REAL_RDMA_PORTS = "mlx5_0 Ethernet 400 Gb/sec (4X NDR)\nmlx5_1 Ethernet 400 Gb/sec (4X NDR)\n"
REAL_RDMA_NUMA = "mlx5_0 0\nmlx5_1 1\n"
REAL_DD_OUT = "536870912 bytes (537 MB, 512 MiB) copied, 0.238186 s, 2.3 GB/s\n"

_ROCE_RAW = {
    "rdma_devices": {"ok": True, "value": "mlx5_0\nmlx5_1\n"},
    "rdma_link": {"ok": True, "value": REAL_RDMA_LINK},
    "rdma_ports": {"ok": True, "value": REAL_RDMA_PORTS},
    "rdma_numa": {"ok": True, "value": REAL_RDMA_NUMA},
    "gpu_smi": {"ok": True, "value": (
        "0, NVIDIA A100-SXM4-40GB, 42234101760, 535.104.05, Enabled, 810, 1410, 28.63, 300.00\n"
        "1, NVIDIA A100-SXM4-40GB, 42234101760, 535.104.05, Enabled, 810, 1410, 28.63, 300.00\n")},
    "gpu_topo": {"ok": True, "value": (HEAD / "gpu_topo.txt").read_text(encoding="utf-8")},
}


def test_rdma_roce_detected_without_ibstat():
    """防假阴性回归：无 ibstat、有 sysfs mlx5_* + link_layer Ethernet → kind=roce。"""
    facts = probe.build_host_facts({"name": "head", "role": "head"}, _ROCE_RAW)
    rdma = facts["net"]["rdma"]
    assert rdma["kind"] == "roce"
    assert rdma["ibstat_present"] is False          # 兼容字段：无 ibstat 仍判 RoCE
    d0 = rdma["devices"][0]
    assert d0["netdev"] == "enp24s0f0np0"
    assert d0["state"] == "ACTIVE"                  # 真机 rdma link show 格式
    assert d0["speed_gbps"] == 400                  # rate "400 Gb/sec (4X NDR)" → 400
    assert d0["numa_node"] == 0                     # sysfs device/numa_node
    assert d0["gpu_links"]["0"] == "PXB"            # NIC↔GPU 拓扑级别（topo NIC 行）
    assert probe.format_rdma(rdma) == "RoCE(Ethernet) ×2（400 Gb/s）"


def test_rdma_real_outputs_parse_all_fields():
    """关键教训回归：真机逐字输出必须解析出 netdev/state/speed_gbps/numa_node/read_bw 全非 None。"""
    assert probe.parse_rdma_link(REAL_RDMA_LINK)["mlx5_0"]["netdev"] == "enp24s0f0np0"
    assert probe.parse_rdma_link(REAL_RDMA_LINK)["mlx5_0"]["state"] == "ACTIVE"
    assert probe.parse_rdma_ports(REAL_RDMA_PORTS)["mlx5_0"]["rate"] == "400 Gb/sec (4X NDR)"
    assert probe.parse_rate_gbps("400 Gb/sec (4X NDR)") == 400
    assert probe.parse_rate_gbps("100 Gb/sec (4X EDR)") == 100
    assert probe.parse_rate_gbps("25 Gb/sec") == 25
    assert probe.parse_rate_gbps("200 Gb/sec (2X HDR)") == 200
    assert probe.parse_rdma_numa(REAL_RDMA_NUMA) == {"mlx5_0": 0, "mlx5_1": 1}
    assert probe.parse_dd_bw(REAL_DD_OUT) == pytest.approx(2.3 * 8, abs=0.01)


def test_rdma_none_when_no_devices():
    """无任何 RDMA 设备 → kind=none（报告「无」）。"""
    raw = dict(_ROCE_RAW)
    raw.update({"rdma_devices": {"ok": False, "value": None, "note": "fixture 缺失"},
                "rdma_link": {"ok": False, "value": None, "note": "fixture 缺失"},
                "rdma_ports": {"ok": False, "value": None, "note": "fixture 缺失"},
                "rdma_numa": {"ok": False, "value": None, "note": "fixture 缺失"}})
    facts = probe.build_host_facts({"name": "head", "role": "head"}, raw)
    assert facts["net"]["rdma"]["kind"] == "none"
    assert facts["net"]["rdma"]["devices"] == []
    assert probe.format_rdma(facts["net"]["rdma"]) == "无"


def test_rdma_mtu_check_applies_to_roce():
    """RoCE 也要过 MTU 检查（原 ibstat 判据会漏）：接口 MTU 1500 < 9000 → warn。"""
    facts = {
        "cpu": {"governor": "performance"},
        "gpu": {"gpus": [{"index": 0, "persistence_mode": "Enabled",
                          "clocks_sm": 1800, "clocks_max_sm": 2000}]},
        "model_src": {"config_json": {}, "note": None},
        "disk": {"avail_bytes": None},
        "mem": {"thp": "madvise", "swappiness": 10},
        "net": {"rdma": {"kind": "roce", "devices": [{"speed_gbps": 400}]},
                "interfaces": [{"name": "enp177s0np0", "mtu": 1500}]},
        "numa_groups": [{"numa_node": 0, "gpus": [0]}],
        "docker": {"version": "20.10"},
    }
    items = probe._system_checklist(facts, is_head=True, weights_bytes=None)
    mtu = next(i for i in items if i["name"] == "RDMA 网卡 MTU")
    assert mtu["status"] == "warn"
    assert "enp177s0np0" in mtu["detail"]


# ── 互联带宽实测（gpu_p2p_bw / rdma_bw 解析）────────────────────────────

_P2P_FIXTURE = "P2P_INTRA 43.35 P2P_CROSS 33.87\n"
_RDMA_FIXTURE = (
    "---------------------------------------------------------------------------------------\n"
    "                    RDMA_Write BW Test\n"
    "---------------------------------------------------------------------------------------\n"
    " #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]\n"
    " 65536      0               352.9              341.7               0.000651\n"
    "---------------------------------------------------------------------------------------\n"
)


def test_interconnect_bw_parsed_from_verbatim_samples():
    """fixture 逐字样例 → 解析出 GPU P2P 与 RDMA 带宽数值。"""
    p2p = probe.parse_gpu_p2p_bw({"ok": True, "value": _P2P_FIXTURE, "note": None})
    assert p2p == {"intra_numa": 43.35, "cross_numa": 33.87}
    rdma = probe.parse_rdma_bw({"ok": True, "value": _RDMA_FIXTURE, "note": None})
    assert rdma == {"bw_gbps": 341.7}              # BW average[Gb/sec] 第 4 列


def test_interconnect_bw_unavailable_gives_null_and_note():
    """不可得（fixture 缺失/采集失败）→ null + note，绝不抛错。"""
    rec = {"ok": False, "value": None, "note": "fixture 缺失"}
    assert probe.parse_gpu_p2p_bw(rec) == {"intra_numa": None, "cross_numa": None,
                                           "note": "gpu_p2p_bw: fixture 缺失"}
    assert probe.parse_rdma_bw(rec) == {"bw_gbps": None, "note": "rdma_bw: fixture 缺失"}
    assert probe.parse_gpu_p2p_bw(None)["intra_numa"] is None     # rec 缺省也容错
    assert probe.parse_rdma_bw(None)["bw_gbps"] is None


def test_interconnect_bw_skip_marker_note():
    """RDMA_SKIP（无跨 NUMA 对 / 无 ib_write_bw）→ null + 具体 note。"""
    r = probe.parse_rdma_bw({"ok": True, "value": "RDMA_SKIP no cross-numa pair\n", "note": None})
    assert r["bw_gbps"] is None
    assert "no cross-numa pair" in r["note"]


def test_p2p_script_warmup_before_timing():
    """关键教训回归：真机 intra 被低估（无预热）→ 脚本必须在计时前有 ≥2 次不计时预热 copy。"""
    s = probe._P2P_SCRIPT
    warm = s.index("for _ in range(2): y.copy_(x)")     # 预热循环
    t0 = s.index("t0=time.perf_counter()")             # 计时起点
    timed = s.index("for _ in range(8): y.copy_(x)")   # 计时循环
    assert warm < t0 < timed                           # 先预热、再计时


def test_rdma_cmd_client_single_ip_with_hca():
    """关键教训回归：真机 rdma_bw=None（client 多传 IP + 未指定 HCA）→
    client 只传一个 server IP，且两侧都用 -d <hca> 指定跨 NUMA 的卡。"""
    cmd = probe._RDMA_BW_CMD
    assert '"$sip"' in cmd
    assert '"$cip"' not in cmd                          # 不再有 client IP 参数
    client_lines = [ln.strip() for ln in cmd.splitlines()
                    if ln.strip().startswith("ib_write_bw") and "nohup" not in ln]
    server_lines = [ln.strip() for ln in cmd.splitlines() if ln.strip().startswith("nohup")]
    assert client_lines and server_lines
    assert '-d "$ca"' in client_lines[0]                # client 指定选中的 HCA
    assert client_lines[0].count('"$sip"') == 1         # 只传一个 server IP
    assert '-d "$sa"' in server_lines[0]                # server 指定选中的 HCA
    # 纪律保留：nohup+重定向、timeout 兜底、按 PID 精确 kill、禁 pkill -f
    assert "nohup" in cmd and ">/tmp/probe_ibw_srv.log 2>&1 </dev/null &" in cmd
    assert "timeout 15" in cmd
    assert 'kill "$spid"' in cmd
    assert "pkill" not in cmd
