"""test_host_report.py — 主机探索 HTML 渲染（host_report.render_host_report）与新增解析器。

host_report.html 是「先提供主机 → agent 自主探索 → 出报告给人看」的交付物，
约束：自包含离线（无 <script> / 无外链）、所有动态文本转义、缺失字段不抛异常。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import host_report  # noqa: E402
import probe  # noqa: E402

TARGET = {"hosts": [{"name": "head", "ssh": "user@10.0.0.1", "role": "head"}], "model": None}
HARNESS = {"remote": {"vllm_port": 8000}}


def _facts(**over) -> dict:
    f = {
        "role": "head", "hostname": "n1", "python3_version": "Python 3.12.3", "vllm_version": None,
        "gpu": {"count": 2, "gpus": [
            {"index": 0, "name": "NVIDIA A100", "memory_total": 81920, "driver_version": "595",
             "persistence_mode": "Enabled", "clocks_sm": 1000, "clocks_max_sm": 1410, "numa_node": 0},
            {"index": 1, "name": "NVIDIA A100", "memory_total": 81920, "driver_version": "595",
             "persistence_mode": "Enabled", "clocks_sm": 1000, "clocks_max_sm": 1410, "numa_node": 0},
        ], "topo": {"summary": "nvlink-full", "raw": "GPU0 X NV2\n"}},
        "numa_groups": [{"numa_node": 0, "gpus": [0, 1]}],
        "cpu": {"model_name": "Xeon", "cores": 64, "numa_nodes": 1, "governor": "performance"},
        "mem": {"thp": "madvise", "swappiness": 10},
        "net": {"interfaces": [{"name": "eth0", "mtu": 9000}],
                "rdma": {"kind": "roce", "devices": [{"name": "mlx5_0", "state": "ACTIVE",
                         "netdev": "eth0", "speed_gbps": 400, "numa_node": 0}]}},
        "disk": {"model_path": None, "avail_bytes": 1_000_000_000_000, "read_bw_gbps": 2.3},
        "docker": {"version": "Docker 29", "vllm_images": ["vllm/vllm-openai:v0.31.0"],
                   "images": [{"repo": "vllm/vllm-openai:v0.31.0", "size": "32.3GB",
                               "created": "12 days ago"}],
                   "containers": [{"name": "c1", "image": "x", "status": "Up", "ports": "8000"}]},
        "ports": {"listening": [{"addr": "0.0.0.0:8000", "port": 8000}], "vllm_port_conflict": True},
        "users": {"logged_in": ["u pts/0"]},
        "model_inventory": [{"dir": "/m/big", "size_bytes": 900 * 1024**3, "n_shards": 8,
                             "model_type": "qwen2", "architectures": "Qwen2ForCausalLM",
                             "quant_method": "fp8", "torch_dtype": "bfloat16", "layers": 40,
                             "heads": 64, "kv_heads": 8}],
        "checklist": [{"name": "X", "status": "pass", "detail": "ok", "fix": "—"}],
        "interconnect": {"gpu_p2p_bw_gbps": {"intra_numa": 40.0, "cross_numa": 30.0},
                         "rdma_bw_gbps": {"bw_gbps": 280.0}},
    }
    f.update(over)
    return f


def _render(facts: dict) -> str:
    return host_report.render_host_report(TARGET, facts, HARNESS, {"weights": {"source": "unavailable"}})


def test_render_is_self_contained():
    html = _render({"head": _facts()})
    assert "<html" in html.lower()
    assert "<script" not in html
    assert "http://" not in html and "https://" not in html   # 无外链（可离线打开）


def test_render_escapes_dynamic_text():
    html = _render({"head": _facts(hostname="a<b>&c")})
    assert "a&lt;b&gt;&amp;c" in html
    assert "a<b>&c" not in html


def test_render_contains_key_sections():
    html = _render({"head": _facts()})
    assert "NVIDIA A100" in html
    assert "vllm/vllm-openai:v0.31.0" in html
    assert "/m/big" in html
    assert "8000" in html


def test_render_survives_empty_facts():
    html = _render({"head": {"role": "head"}})
    assert "<html" in html.lower()


def test_new_parsers():
    inv = probe.parse_model_inventory("/m/a\t100\t2\tmt\tArch\tfp8\tbfloat16\t40\t64\t1\n")
    assert inv == [{"dir": "/m/a", "size_bytes": 100, "n_shards": 2, "model_type": "mt",
                    "architectures": "Arch", "quant_method": "fp8", "torch_dtype": "bfloat16",
                    "layers": 40, "heads": 64, "kv_heads": 1}]
    assert probe.parse_model_inventory("坏行") == []
    assert probe.parse_model_inventory(None) == []

    ports = probe.parse_listening_ports(
        "State  Recv-Q Send-Q Local Address:Port  Peer Address:Port\n"
        "LISTEN 0      128          0.0.0.0:8000         0.0.0.0:*\n")
    assert ports == [{"addr": "0.0.0.0:8000", "port": 8000}]

    imgs = probe.parse_docker_images_detail("vllm/x:1\t32GB\t12 days ago\nbare:1\n")
    assert imgs == [{"repo": "vllm/x:1", "size": "32GB", "created": "12 days ago"},
                    {"repo": "bare:1", "size": None, "created": None}]

    assert probe.parse_docker_ps("c1\timg\tUp\t8000\n")[0]["name"] == "c1"
    assert probe.parse_who("u pts/0\n\n") == ["u pts/0"]
