#!/usr/bin/env python3
"""probe.py — Phase 1 环境探测（只读 harness 工具）。

由编码 agent 在编排机执行：读取 target.yaml + harness.yaml，
对所有主机经 SSH 并行采集只读事实，产出：
  env.json          — 结构化原始事实（供 program.md 剪枝规则消费）
  env_report.md     — 人类可读报告，含系统参数检查清单（pass/warn/fail + 修复命令）
  model_profile.json— 模型权重/配置估算（供搜索空间剪枝与成本估算）

设计约束：
- 对系统只读：所有远端命令均为查询类，绝不修改主机状态。
- 容错：单个命令失败/缺工具 → 记录 null + note，绝不中断整个探测。
- --fixtures DIR：离线模式，从 DIR/<hostname>/<filename> 读取预录输出，
  与在线模式采集计划 1:1 映射，可在无 GPU 的笔记本上运行。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import math
import os
import re
import shlex
import subprocess
import sys
from typing import NoReturn

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("缺少 pyyaml：请用 `uv run python probe.py` 或 `pip install pyyaml`")

# ── 系统参数检查清单阈值（模块常量，便于调整）──────────────────────────
GOVERNOR_WANTED = "performance"          # CPU 频率调节器：非 performance 视为 fail
THP_OK = {"always", "madvise"}           # THP 合法模式；其他（如 never）视为 warn
SWAPPINESS_WARN_ABOVE = 10               # vm.swappiness 超过该值视为 warn
MTU_WARN_BELOW = 9000                    # RDMA 网卡 MTU 低于该值视为 warn
GPU_SMCLOCK_WARN_FRACTION = 0.80         # SM 时钟 < 该比例 × max 视为 warn（负载时复查）
DISK_FREE_FACTOR = 1.2                   # 磁盘剩余需 ≥ 权重 × 该系数
DEFAULT_GMU = 0.92                       # vLLM 默认 gpu-memory-utilization（docs §4）
KV_DTYPE_BYTES = 2                       # KV cache 默认 bf16（2 字节/元素）
BOOT_DISK_BW_GBPS = 2.0                  # 权重加载粗估带宽（磁盘→显存，含解码开销）

# 已知 GPU 的 HBM 带宽（GB/s）粗表；未知型号 → None（roofline 置 null）
HBM_BW_GBPS = {
    "H200": 4800, "H100": 3350, "H800": 3350, "B200": 8000, "B300": 8000,
    "A100": 1555, "A800": 2039, "A6000": 768, "A40": 696, "A10": 600,
    "A30": 933, "L40": 864, "L40S": 864, "L4": 252, "RTX 4090": 1008,
    "RTX 6000 Ada": 960, "V100": 900, "P100": 732,
}

# 不做 KV 估算的架构族（MLA / mamba 类；只能靠 Phase 2 基线校准）
UNSUPPORTED_KV_TYPES = {
    "mamba", "mamba2", "falcon_mamba", "recurrent_gemma", "rwkv", "griffin",
}

DTYPE_BYTES = {  # torch_dtype → 每参数字节数（估算用）
    "float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5,
    "float8_e4m3fn": 1, "float8_e5m2": 1, "uint8": 1,
}


# ── 输入与校验 ──────────────────────────────────────────────────────────

def _err(msg: str) -> NoReturn:
    print(f"probe.py: 错误: {msg}", file=sys.stderr)
    sys.exit(1)


def load_yaml(path: str, required: bool = True) -> dict:
    """加载 yaml；target.yaml 缺失时给出明确的复制指引。"""
    if not os.path.exists(path):
        if required and path == "target.yaml":
            _err("未找到 target.yaml。请先 `cp target.example.yaml target.yaml` 并填写主机/模型/SLO/预算。")
        if required:
            _err(f"未找到 {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def validate_target(target: dict) -> dict:
    """校验 target.yaml：hosts 非空、name 唯一、有且仅有一个 role=head（program.md:74）。"""
    hosts = target.get("hosts") or []
    if not hosts:
        _err("target.yaml 的 hosts 为空。")
    names = [h.get("name") for h in hosts]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        _err(f"主机 name 必须唯一（重复：{', '.join(sorted(str(d) for d in dupes))}）；重名会在采集计划/结果中被静默覆盖。")
    heads = [h for h in hosts if h.get("role") == "head"]
    if len(heads) != 1:
        names = ", ".join(h.get("name", "?") for h in heads) or "（无）"
        _err(f"必须恰好一个 role=head 的主机（当前 {len(heads)} 个：{names}）；缺失或重复都不可执行。")
    return heads[0]


def read_gmu_default() -> float:
    """从 config.yaml 取 gpu-memory-utilization（若有），否则用 vLLM 默认 0.92。"""
    try:
        cfg = load_yaml("config.yaml", required=False)
        gmu = (cfg.get("serve") or {}).get("gpu-memory-utilization")
        return float(gmu) if gmu is not None else DEFAULT_GMU
    except (OSError, TypeError, ValueError):
        return DEFAULT_GMU


# ── SSH 与采集计划 ──────────────────────────────────────────────────────

def _ssh(host: dict, cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    """在远端执行命令；尊重环境变量 SSH_PORT / SSH_KEY（见 .env.example）。"""
    base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    port = os.environ.get("SSH_PORT")
    key = os.environ.get("SSH_KEY")
    if port:
        base += ["-p", str(port)]
    if key:
        base += ["-i", key]
    base += [host["ssh"], cmd]
    try:
        p = subprocess.run(base, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "ssh 超时"
    except OSError as e:
        return -1, "", str(e)


def _cmd(key: str, cmd: str, fixture: str | None = None) -> dict:
    """采集计划条目：key 与 cmd 一一对应；fixture 文件名缺省为 <key>.txt。"""
    return {"key": key, "cmd": cmd, "fixture": fixture or f"{key}.txt"}


def _collect_plan(target: dict, harness: dict) -> dict[str, list]:
    """每台主机的采集计划：list of (key, cmd, fixture 文件名)。"""
    remote = harness.get("remote") or {}
    vllm_python = remote.get("vllm_python", "python")
    model = target.get("model") or {}
    plan: dict[str, list] = {}
    for h in target.get("hosts", []):
        name = h["name"]
        cmds = [
            _cmd("gpu_smi", "nvidia-smi --query-gpu=index,name,memory.total,driver_version,"
                            "persistence_mode,clocks.sm,clocks.max.sm,power.draw,power.limit "
                            "--format=csv,noheader"),
            _cmd("gpu_topo", "nvidia-smi topo -m"),
            _cmd("lscpu", "lscpu"),
            _cmd("governor", "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || true"),
            _cmd("numactl", "command -v numactl >/dev/null 2>&1 && numactl -H || true"),
            _cmd("thp", "cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true"),
            _cmd("swappiness", "sysctl -n vm.swappiness 2>/dev/null || true"),
            _cmd("ibstat", "command -v ibstat >/dev/null 2>&1 && ibstat || true"),
            # 用 -o 格式（每行一条 + mtu），避免 -br 别名在不同系统上的解析差异
            _cmd("ip_link", "ip -o link show 2>/dev/null || true"),
            _cmd("ibv_devinfo", "command -v ibv_devinfo >/dev/null 2>&1 && ibv_devinfo -l || true"),
            _cmd("docker_version", "docker --version 2>/dev/null || true"),
            _cmd("docker_images", "docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null || true"),
            # 用 pip show（不 import）避免在主机上写入 torch/vllm 缓存
            _cmd("vllm_version", f"{shlex.quote(vllm_python)} -m pip show vllm 2>/dev/null || true"),
            _cmd("hostname", "hostname"),
            _cmd("python3_version", "python3 --version 2>&1 || true"),
        ]
        if h.get("role") == "head":
            src = model.get("source", "hf")
            if src == "local" and model.get("path"):
                mp = model["path"]
                cmds += [
                    _cmd("model_config_json", f"cat {shlex.quote(mp)}/config.json 2>/dev/null || true", "config.json"),
                    _cmd("model_safetensors_index_json",
                         f"cat {shlex.quote(mp)}/*.safetensors.index.json 2>/dev/null || true",
                         "safetensors.index.json"),
                    _cmd("df_model", f"df -B1 --output=avail {shlex.quote(os.path.dirname(mp))} 2>/dev/null | tail -1 || true"),
                ]
            else:  # source=hf：尝试本地 HF 缓存（~/.cache/huggingface/hub/models--<id>/snapshots/*）
                hid = model.get("id", "").split("@")[0].replace("/", "--")
                glob = f"~/.cache/huggingface/hub/models--{shlex.quote(hid)}/snapshots/*"
                cmds += [
                    _cmd("model_config_json",
                         f"for f in {glob}/config.json; do [ -f \"$f\" ] && cat \"$f\" && exit 0; done; exit 0",
                         "config.json"),
                    _cmd("model_safetensors_index_json",
                         f"for f in {glob}/*.safetensors.index.json; do [ -f \"$f\" ] && cat \"$f\" && exit 0; done; exit 0",
                         "safetensors.index.json"),
                ]
        plan[name] = cmds
    return plan


def _read_fixture(fixture_dir: str | None, hostname: str, filename: str) -> tuple[bool, str | None]:
    if not fixture_dir:
        return False, None
    path = os.path.join(fixture_dir, hostname, filename)
    try:
        with open(path, encoding="utf-8") as f:
            return True, f.read()
    except OSError:
        return False, None


def collect_host(host: dict, entries: list, mode: str, fixture_dir: str | None,
                 timeout: int = 30) -> dict:
    """采集一台主机的全部事实；单条失败只记 null，不中断。"""
    out: dict = {}
    for e in entries:
        key = e["key"]
        if mode == "fixtures":
            ok, text = _read_fixture(fixture_dir, host["name"], e["fixture"])
            out[key] = {"ok": ok, "value": text, "note": None if ok else "fixture 缺失"}
        else:
            rc, stdout, stderr = _ssh(host, e["cmd"], timeout)
            if rc != 0 or not stdout.strip():
                note = f"rc={rc}"
                if stderr.strip():
                    note += f"; {stderr.strip()[:200]}"
                out[key] = {"ok": False, "value": None, "note": note}
            else:
                out[key] = {"ok": True, "value": stdout, "note": None}
    return out


# ── 输出解析（容错：None/坏文本 → None，绝不抛异常）─────────────────────

def _pip_show_version(text: str | None) -> str | None:
    """从 `python -m pip show vllm` 输出提取 Version: 行；缺失/未安装 → None。"""
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("Version:"):
            v = line.split(":", 1)[1].strip()
            return v or None
    return None


def _first_int(text: str | None, pattern: str = r"(\d+)") -> int | None:
    if not text:
        return None
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def _parse_float_cell(cell: str) -> float | None:
    """解析 nvidia-smi CSV 数值单元格；坏值（如 'N/A'、'1.2.3'）→ None，不炸整行。"""
    if not re.match(r"[\d.]+", cell):
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def parse_gpu_smi(text: str | None) -> list[dict] | None:
    """nvidia-smi CSV → GPU 列表；单行解析失败跳过（不炸整个探测）。"""
    if not text:
        return None
    gpus = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        gpus.append({
            "index": _first_int(parts[0]),
            "name": parts[1],
            "memory_total": _first_int(parts[2]),
            "driver_version": parts[3],
            "persistence_mode": parts[4],
            "clocks_sm": _first_int(parts[5]),
            "clocks_max_sm": _first_int(parts[6]),
            "power_draw": _parse_float_cell(parts[7]),
            "power_limit": _parse_float_cell(parts[8]),
        })
    return gpus or None


def parse_lscpu(text: str | None) -> dict:
    if not text:
        return {}
    d: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            d[k.strip()] = v.strip()
    return d


def parse_topo(text: str | None, gpu_count: int) -> tuple[str | None, str | None]:
    """nvidia-smi topo -m → (矩阵原文, 摘要)。摘要供 program.md 剪枝：
    nvlink-full / pcie / mixed / unknown。"""
    if not text:
        return None, None
    links = []
    for line in text.splitlines():
        toks = line.split()
        if len(toks) > gpu_count and toks[0].startswith("GPU"):
            cells = toks[1 : gpu_count + 1]
            # 跳过列标签头行（含 GPU1/NIC0 等列名；数据行只含 X/NV*/PIX/PHB/NODE/SYS 等连接码）
            if any(re.match(r"^(GPU|NIC)\d+$", c) for c in cells):
                continue
            try:
                gi = int(toks[0][3:])  # 行号（GPU0/GPU1…）
            except ValueError:
                gi = None
            for j, cell in enumerate(cells):
                if gi is None or j != gi:  # 跳过对角线自连接 "X"
                    links.append(cell)
    if not links:
        return text, None
    if all(l.startswith("NV") for l in links):
        summary = "nvlink-full"
    elif not any(l.startswith("NV") for l in links):
        # 仅有 PIX/PHB/NODE/SYS、无任何 NV 链接 → 视为 PCIe 拓扑（program.md:70 剪枝依据）
        summary = "pcie"
    else:
        summary = "mixed"  # NV 与非 NV 混连（部分直连部分走 PCIe）
    return text, summary


def parse_ip_link(text: str | None) -> list[dict]:
    """解析 `ip -o link show`（每行: `N: name: <flags> mtu M …`）；缺 mtu/坏行跳过。"""
    if not text:
        return []
    ifaces = []
    for line in text.splitlines():
        m = re.match(r"\d+:\s+(\S+):.*\bmtu (\d+)", line)
        if m:
            ifaces.append({"name": m.group(1), "mtu": int(m.group(2))})
    return ifaces


def parse_thp(text: str | None) -> str | None:
    """/sys/kernel/mm/transparent_hugepage/enabled → 当前模式（[] 括起的那个）。"""
    if not text:
        return None
    m = re.search(r"\[(\w+)\]", text)
    return m.group(1) if m else None


# ── 纯估算函数（单元测试目标）──────────────────────────────────────────

_NESTED_KEYS = (
    "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
    "head_dim", "vocab_size", "intermediate_size", "max_position_embeddings",
    "torch_dtype", "model_type", "kv_lora_rank", "linear_attn",
    "num_local_experts", "num_experts", "shared_expert",
)


def normalize_config(config: dict) -> dict:
    """规范化模型 config：新式 HF 配置（如 glm5_next / Glm5NextForConditionalGeneration）
    把文本模型参数嵌在 text_config 下——顶层缺关键字段时从 text_config 补齐；
    head_dim=0 → 置 None（调用方以 hidden_size//num_attention_heads 兜底）。
    对真正未知的 schema 不做编造：仍保持原样，由上层以 None + note 处理。"""
    if not isinstance(config, dict):
        return config
    c = dict(config)
    tc = c.get("text_config")
    if isinstance(tc, dict):
        for k in _NESTED_KEYS:
            if (k not in c or c[k] is None) and k in tc:
                c[k] = tc[k]
    if c.get("head_dim") == 0:
        c["head_dim"] = None
    return c


def effective_head_dim(config: dict) -> int | None:
    """head_dim：显式值 > 0 用之；否则 hidden_size // num_attention_heads 兜底。"""
    hd = config.get("head_dim")
    if hd:
        return int(hd)
    hidden = config.get("hidden_size")
    heads = config.get("num_attention_heads")
    if hidden and heads:
        return int(hidden // heads)
    return None


def dtype_bytes(config: dict) -> float:
    """每参数字节数：量化方法（gptq/awq 4bit）优先——真实量化模型的 torch_dtype
    仍是 float16，先看 quantization_config 才能避免权重估算 4 倍虚高；否则按
    torch_dtype；缺省 bf16。"""
    qc = config.get("quantization_config") or {}
    qm = str(qc.get("quant_method") or "")
    if qm in {"gptq", "awq", "bitsandbytes", "hqq"}:
        return 0.5
    td = str(config.get("torch_dtype") or "").lower()
    if td in DTYPE_BYTES:
        return DTYPE_BYTES[td]
    return 2  # bf16 兜底


def estimate_params(config: dict) -> int | None:
    """参数总量粗估（含 GQA / 稠密 MoE）；不可估返回 None。
    近似：仅统计 注意力 + MLP + embedding，忽略 norm/rope 等小头。"""
    hidden = config.get("hidden_size")
    layers = config.get("num_hidden_layers")
    heads = config.get("num_attention_heads")
    vocab = config.get("vocab_size")
    if not (hidden and layers and heads and vocab):
        return None
    head_dim = config.get("head_dim") or hidden // heads
    kv_heads = config.get("num_key_value_heads") or heads
    attn_per_layer = 2 * hidden * heads * head_dim + 2 * hidden * kv_heads * head_dim  # q+o, k+v
    inter = config.get("intermediate_size")
    if inter:
        experts = config.get("num_local_experts") or config.get("num_experts")
        if experts:  # MoE：专家 FFN 每层 × experts（+ 共享专家，若存在）
            mlp_per_layer = experts * 3 * hidden * inter
            if "shared_expert" in config:
                mlp_per_layer += 3 * hidden * inter
        else:
            mlp_per_layer = 3 * hidden * inter  # 门控 MLP：gate/up/down
    else:
        mlp_per_layer = 2 * hidden * hidden     # 非门控近似
    return (attn_per_layer + mlp_per_layer) * layers + vocab * hidden


def weight_bytes(index: dict | None, config: dict) -> tuple[int | None, str]:
    """权重字节数：优先 safetensors index 的 metadata.total_size；
    否则 params × dtype_bytes（estimate）；都不可得 → (None, 'unavailable')。"""
    if index and index.get("metadata", {}).get("total_size"):
        try:
            return int(index["metadata"]["total_size"]), "safetensors-index"
        except (TypeError, ValueError):
            pass
    params = estimate_params(config)
    if params:
        return int(params * dtype_bytes(config)), "estimate"
    return None, "unavailable"


def _kv_unsupported(config: dict) -> bool:
    if any(k in config for k in ("kv_lora_rank", "linear_attn")):
        return True  # MLA / mamba-hybrid
    return config.get("model_type") in UNSUPPORTED_KV_TYPES


def kv_bytes_per_token(config: dict, kv_dtype_bytes: int = KV_DTYPE_BYTES) -> int | None:
    """每 token 的 KV cache 字节数 = layers × kv_heads × head_dim × 2 × dtype。
    GQA 取 num_key_value_heads；head_dim 缺省 hidden/heads。
    MLA / mamba 类架构不支持估算 → None（由调用方写 note，靠基线校准）。"""
    if _kv_unsupported(config):
        return None
    layers = config.get("num_hidden_layers")
    heads = config.get("num_attention_heads")
    hidden = config.get("hidden_size")
    if not (layers and heads and hidden):
        return None
    kv_heads = config.get("num_key_value_heads") or heads
    head_dim = config.get("head_dim") or hidden // heads
    return layers * kv_heads * head_dim * 2 * kv_dtype_bytes


def valid_tp_divisors(config: dict, gpu_mem_bytes: int, gmu: float = DEFAULT_GMU,
                      weights_bytes: int | None = None) -> dict:
    """合法 tp 集合：同时整除 num_attention_heads 与 num_key_value_heads（program.md:72），
    且单卡权重 ≤ gpu_mem × gmu。min_tp_for_weights 为满足显存的最小 tp。"""
    heads = config.get("num_attention_heads")
    if not heads:
        return {"valid_divisors": [], "min_tp_for_weights": None}
    kv_heads = config.get("num_key_value_heads") or heads
    wb = weights_bytes
    if wb is None:
        wb, _ = weight_bytes(None, config)
    divs = [t for t in range(1, heads + 1) if heads % t == 0 and kv_heads % t == 0]
    budget = gpu_mem_bytes * gmu
    if wb and budget:
        # 最小 tp 必须是合法约数（否则与 valid_divisors 矛盾）；装不下 → None
        need = math.ceil(wb / budget)
        min_tp = min([t for t in divs if t >= need], default=None)
        divs = [t for t in divs if wb / t <= budget]
    else:
        min_tp = 1 if divs else None  # 无法核算显存时假设 tp=1 即可装下
    return {"valid_divisors": divs, "min_tp_for_weights": min_tp}


def roofline_decode_tps(weight_bytes_: int | None, hbm_bw_gbps: float | None) -> float | None:
    """batch-1 decode 上界（tps）= 带宽 / 权重字节数。注意是天花板：实测算例
    还需乘模型利用系数（如 0.3~0.5），只用于横向比较与预算。"""
    if not weight_bytes_ or not hbm_bw_gbps:
        return None
    return hbm_bw_gbps * 1e9 / weight_bytes_


def estimate_boot_seconds(weight_bytes_: int | None, disk_bw_gbps: float = BOOT_DISK_BW_GBPS) -> float | None:
    """权重加载耗时粗估 = 字节数 / 磁盘带宽。假设：整份权重顺序读一遍、
    不含解压/pin 内存/编译开销；实际启动以 Phase 2 基线校准（calibrated）。"""
    if not weight_bytes_:
        return None
    return weight_bytes_ / (disk_bw_gbps * 1e9)


def hbm_bw_gbps_for(gpu_name: str | None) -> float | None:
    if not gpu_name:
        return None
    for key, bw in HBM_BW_GBPS.items():
        if key in gpu_name:
            return float(bw)
    return None


# ── 主机事实组装 ────────────────────────────────────────────────────────

def build_host_facts(host: dict, raw: dict) -> dict:
    """把采集原始输出解析为结构化事实（env.json 的 hosts.<name>）。"""
    gpus = parse_gpu_smi(raw.get("gpu_smi", {}).get("value"))
    lscpu = parse_lscpu(raw.get("lscpu", {}).get("value"))
    topo_raw, topo_summary = parse_topo(raw.get("gpu_topo", {}).get("value"), len(gpus or []))
    ifaces = parse_ip_link(raw.get("ip_link", {}).get("value"))

    numa = _first_int(lscpu.get("NUMA node(s)")) if lscpu.get("NUMA node(s)") is not None else None
    if numa is None:
        numa = _first_int(raw.get("numactl", {}).get("value"), r"available: (\d+) nodes")

    model_src: dict = {
        "config_json": None,
        "safetensors_index": None,
        "note": None,
    }
    for key, fld in (("model_config_json", "config_json"),
                     ("model_safetensors_index_json", "safetensors_index")):
        rec = raw.get(key, {})
        if rec.get("ok") and rec.get("value"):
            try:
                model_src[fld] = json.loads(rec["value"])
            except ValueError:
                model_src["note"] = f"{key} 内容非合法 JSON"
        elif rec.get("ok") is False:
            model_src["note"] = f"{key}: {rec.get('note')}"

    df_avail = None
    rec = raw.get("df_model", {})
    if rec.get("ok") and rec.get("value"):
        df_avail = _first_int(rec["value"])

    ib_ok = raw.get("ibstat", {}).get("ok")
    ib_devs_raw = raw.get("ibv_devinfo", {}).get("value")
    ib_devs = [l.strip() for l in ib_devs_raw.splitlines() if l.strip()] if ib_devs_raw else None

    docker_ver = (raw.get("docker_version", {}).get("value") or "").strip() or None
    docker_imgs_raw = raw.get("docker_images", {}).get("value") or ""
    docker_vllm = [l.strip() for l in docker_imgs_raw.splitlines() if "vllm" in l.lower()] or None

    return {
        "role": host.get("role"),
        "hostname": (raw.get("hostname", {}).get("value") or "").strip() or None,
        "python3_version": (raw.get("python3_version", {}).get("value") or "").strip() or None,
        "vllm_version": _pip_show_version(raw.get("vllm_version", {}).get("value")),
        "gpu": {
            "count": len(gpus or []),
            "gpus": gpus,
            "topo": {"raw": topo_raw, "summary": topo_summary},
        },
        "cpu": {
            "model_name": lscpu.get("Model name"),
            "cores": _first_int(lscpu.get("CPU(s)")),
            "numa_nodes": numa,
            "governor": (raw.get("governor", {}).get("value") or "").strip() or None,
        },
        "mem": {
            "thp": parse_thp(raw.get("thp", {}).get("value")),
            "swappiness": _first_int(raw.get("swappiness", {}).get("value")),
        },
        "net": {
            "interfaces": ifaces,
            "rdma": {
                "ibstat_present": bool(ib_ok),
                "ib_devices": ib_devs,
            },
        },
        "disk": {"model_path": None, "avail_bytes": df_avail},
        "docker": {"version": docker_ver, "vllm_images": docker_vllm},
        "model_src": model_src,
    }


def _system_checklist(facts: dict, is_head: bool, weights_bytes: int | None) -> list[dict]:
    """系统参数检查清单：每项 pass/warn/fail + 修复命令。阈值见模块常量。"""
    items: list[dict] = []

    gov = facts["cpu"]["governor"]
    if gov is None:
        items.append({"name": "CPU 频率调节器", "status": "warn", "detail": "无法读取 scaling_governor",
                      "fix": "检查 cpufreq 驱动是否加载（intel_pstate/amd_pstate 等）"})
    elif gov != GOVERNOR_WANTED:
        items.append({"name": "CPU 频率调节器", "status": "fail", "detail": f"当前 {gov}，需 {GOVERNOR_WANTED}",
                      "fix": "sudo cpupower frequency-set -g performance（或写入 /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor）"})
    else:
        items.append({"name": "CPU 频率调节器", "status": "pass", "detail": f"governor={gov}", "fix": "—"})

    gpus = facts["gpu"]["gpus"] or []
    if not gpus:
        items.append({"name": "GPU 持久模式", "status": "warn", "detail": "未检测到 nvidia-smi 输出（无 GPU 或驱动缺失）",
                      "fix": "安装 NVIDIA 驱动并重跑 probe.py"})
    else:
        bad = [g["index"] for g in gpus if g.get("persistence_mode") != "Enabled"]
        if bad:
            items.append({"name": "GPU 持久模式", "status": "fail",
                          "detail": f"GPU {bad} 未开启持久模式",
                          "fix": "sudo nvidia-smi -pm 1"})
        else:
            items.append({"name": "GPU 持久模式", "status": "pass", "detail": "全部 Enabled", "fix": "—"})

    if is_head:
        cfg = facts["model_src"]["config_json"]
        if cfg is None:
            items.append({"name": "模型权重路径", "status": "fail",
                          "detail": f"config.json 不可读（{facts['model_src'].get('note') or '路径缺失'}）",
                          "fix": "核对 target.yaml 的 model.path / 重新下载 HF 模型后重跑 Phase 1"})
        else:
            items.append({"name": "模型权重路径", "status": "pass", "detail": "config.json 可读", "fix": "—"})
        avail = facts["disk"]["avail_bytes"]
        if avail is not None and weights_bytes is not None:
            need = weights_bytes * DISK_FREE_FACTOR
            if avail < need:
                items.append({"name": "磁盘剩余空间", "status": "fail",
                              f"detail": f"可用 {avail/1e9:.1f} GB < 权重×1.2 = {need/1e9:.1f} GB",
                              "fix": "扩容或清理磁盘后重跑"})
            else:
                items.append({"name": "磁盘剩余空间", "status": "pass",
                              "detail": f"可用 {avail/1e9:.1f} GB ≥ {need/1e9:.1f} GB", "fix": "—"})
        else:
            items.append({"name": "磁盘剩余空间", "status": "warn",
                          "detail": "无法核算（仅本地权重路径检查）", "fix": "—"})

    thp = facts["mem"]["thp"]
    if thp is None:
        items.append({"name": "THP（透明大页）", "status": "warn", "detail": "无法读取 enabled",
                      "fix": "echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"})
    elif thp not in THP_OK:
        items.append({"name": "THP（透明大页）", "status": "warn", "detail": f"当前 {thp}，建议 madvise/always",
                      "fix": "echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"})
    else:
        items.append({"name": "THP（透明大页）", "status": "pass", "detail": f"当前 {thp}", "fix": "—"})

    sw = facts["mem"]["swappiness"]
    if sw is None:
        items.append({"name": "vm.swappiness", "status": "warn", "detail": "无法读取",
                      "fix": "sudo sysctl -w vm.swappiness=10"})
    elif sw > SWAPPINESS_WARN_ABOVE:
        items.append({"name": "vm.swappiness", "status": "warn", "detail": f"当前 {sw} > {SWAPPINESS_WARN_ABOVE}",
                      "fix": "sudo sysctl -w vm.swappiness=10"})
    else:
        items.append({"name": "vm.swappiness", "status": "pass", "detail": f"当前 {sw}", "fix": "—"})

    rdma = facts["net"]["rdma"]
    if rdma["ibstat_present"] or (rdma["ib_devices"] and rdma["ib_devices"][0] != ""):
        low = [i["name"] for i in facts["net"]["interfaces"] if i.get("mtu", 0) < MTU_WARN_BELOW]
        if low:
            items.append({"name": "RDMA 网卡 MTU", "status": "warn",
                          "detail": f"接口 {low} MTU < {MTU_WARN_BELOW}",
                          "fix": f"sudo ip link set {low[0]} mtu 9000"})
        else:
            items.append({"name": "RDMA 网卡 MTU", "status": "pass", "detail": "MTU ≥ 9000 或未检测到接口", "fix": "—"})
    else:
        items.append({"name": "RDMA 网卡 MTU", "status": "pass", "detail": "未启用 RDMA/IB", "fix": "—"})

    dv = (facts.get("docker") or {}).get("version")
    if dv:
        items.append({"name": "docker 可用", "status": "pass", "detail": f"docker: {dv}", "fix": "—"})
    else:
        items.append({"name": "docker 可用", "status": "warn",
                      "detail": "docker 不可用（本模板为 docker-only，serve/bench 都需在主机上跑 docker）",
                      "fix": "安装 docker + nvidia-container-toolkit"})

    for g in gpus:
        sm, smmax = g.get("clocks_sm"), g.get("clocks_max_sm")
        if sm is not None and smmax and sm < smmax * GPU_SMCLOCK_WARN_FRACTION:
            items.append({"name": "GPU SM 时钟", "status": "warn",
                          "detail": f"GPU{g.get('index')}: {sm}/{smmax} MHz（< 80%）；空闲时钟偏低属正常",
                          "fix": "无（负载时用 nvidia-smi -q -d CLOCK 复查）"})
    if gpus and not any(i["name"] == "GPU SM 时钟" for i in items):
        items.append({"name": "GPU SM 时钟", "status": "pass", "detail": "≥ 80% max（或无需复查）", "fix": "—"})

    return items


# ── 模型画像 ────────────────────────────────────────────────────────────

def build_model_profile(model: dict, head_facts: dict, gmu: float) -> dict:
    """组装 model_profile.json（见 probe 需求 §4 字段契约）。"""
    src = head_facts["model_src"]
    config = normalize_config(src["config_json"])
    gpus = head_facts["gpu"]["gpus"] or []
    gpu_mem = min((g.get("memory_total") or 0 for g in gpus), default=0)

    if config is None:
        return {
            "model": model.get("id"),
            "arch": None,
            "weights": {"bytes": None, "dtype": None, "source": "unavailable"},
            "tp": {"valid_divisors": [], "min_tp_for_weights": None},
            "kv": {"bytes_per_token": None, "max_model_len": None, "est_capacity_tokens": None,
                   "measured_capacity_tokens": None,
                   "note": "config.json 不可读；需修复后重跑 Phase 1"},
            "roofline": {"hbm_bw_gbps": None, "decode_tps_upper_bound": None},
            "boot": {"estimated_seconds": None, "measured_seconds": None},
            "calibrated": False,
        }

    wb, wsrc = weight_bytes(src["safetensors_index"], config)
    arch = {
        "model_type": config.get("model_type"),
        "num_layers": config.get("num_hidden_layers"),
        "hidden_size": config.get("hidden_size"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "head_dim": effective_head_dim(config),  # head_dim=0 → hidden/heads 兜底
        "vocab_size": config.get("vocab_size"),
    }
    tp = valid_tp_divisors(config, gpu_mem, gmu, wb) if gpu_mem else {
        "valid_divisors": [], "min_tp_for_weights": None}
    kv_bytes = kv_bytes_per_token(config)
    max_len = config.get("max_position_embeddings")
    est_tokens = None
    kv_note = None
    if kv_bytes is None:
        kv_note = "arch not covered; rely on baseline-boot calibration"
    elif tp["min_tp_for_weights"] is None:
        kv_note = "权重在合法 tp 集合内无法装下（无法估算 KV 容量）"
    elif gpu_mem and wb:
        kv_budget = gpu_mem * gmu - wb / tp["min_tp_for_weights"]
        est_tokens = max(0, int(kv_budget // kv_bytes))
    gpu_name = gpus[0]["name"] if gpus else None
    bw = hbm_bw_gbps_for(gpu_name)
    return {
        "model": model.get("id"),
        "arch": arch,
        "weights": {"bytes": wb, "dtype": config.get("torch_dtype"), "source": wsrc},
        "tp": tp,
        # measured_capacity_tokens 由 Phase 2 基线用启动日志
        # "GPU KV cache size: N tokens"（docs/vllm-facts.md §3）写入实测值
        "kv": {"bytes_per_token": kv_bytes, "max_model_len": max_len,
               "est_capacity_tokens": est_tokens, "measured_capacity_tokens": None,
               **({"note": kv_note} if kv_note else {})},
        "roofline": {"hbm_bw_gbps": bw, "decode_tps_upper_bound": roofline_decode_tps(wb, bw)},
        # measured_seconds 由 Phase 2 基线用实测启动耗时写入
        "boot": {"estimated_seconds": estimate_boot_seconds(wb), "measured_seconds": None},
        "calibrated": False,
    }


# ── 跨机互联（仅 >1 台主机时尽力而为）──────────────────────────────────

def interconnect_check(target: dict, mode: str, fixture_dir: str | None) -> dict:
    hosts = target.get("hosts", [])
    if len(hosts) <= 1:
        return {"covered": False, "reason": "单主机，无需跨机检查", "checks": []}
    head = next(h for h in hosts if h.get("role") == "head")
    checks: list[dict] = []
    others = [h for h in hosts if h is not head]
    for o in others:
        ip = o["ssh"].rsplit("@", 1)[-1]
        if mode == "fixtures":
            ok, txt = _read_fixture(fixture_dir, head["name"], f"ping_{o['name']}.txt")
            result = txt.strip() if ok and txt else None
            checks.append({"check": f"ping {o['name']}", "result": result,
                           "note": None if ok else "fixture 缺失"})
        else:
            rc, out, _err2 = _ssh(head, f"ping -c 1 -W 2 {ip} 2>&1 || true", timeout=15)
            checks.append({"check": f"ping {o['name']}", "result": out.strip(), "note": None})
    for tool in ("ib_write_bw", "all_reduce_perf"):
        if mode == "fixtures":
            ok, txt = _read_fixture(fixture_dir, head["name"], f"{tool}.txt")
            result = txt.strip() if ok and txt else None
            checks.append({"check": tool, "result": result,
                           "note": None if ok else "未安装（fixture 缺失）"})
        else:
            rc, out, _err2 = _ssh(head, f"command -v {tool} 2>/dev/null || true", timeout=15)
            checks.append({"check": tool, "result": out.strip(), "note": None})
    covered = any(c.get("result") for c in checks)
    return {"covered": covered,
            "reason": None if covered else "跨机检查未执行或工具缺失（见 checks）",
            "checks": checks}


# ── 报告渲染（env_report.md）───────────────────────────────────────────

def render_env_report(target: dict, facts: dict, interconnect: dict,
                      model_profile: dict, gmu: float) -> str:
    head_name = next(h["name"] for h in target["hosts"] if h.get("role") == "head")
    total = {"pass": 0, "warn": 0, "fail": 0}
    lines = ["# 环境探测报告（env_report.md）", ""]
    lines.append(f"- 生成时间：{datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 主机数：{len(target['hosts'])}（head={head_name}；v1 只在 head 上跑实验）")
    lines.append(f"- 模型：`{model_profile.get('model')}`（gpu-memory-utilization 默认 {gmu}）")
    lines.append("")
    for name, f in facts.items():
        ck = f["checklist"]
        for it in ck:
            total[it["status"]] += 1
        lines.append(f"## {name}（role={f['role']}）")
        lines.append("")
        lines.append("### 系统参数检查清单")
        lines.append("")
        lines.append("| 检查项 | 状态 | 详情 | 修复命令 |")
        lines.append("|---|---|---|---|")
        for it in ck:
            lines.append(f"| {it['name']} | {it['status']} | {it['detail']} | {it['fix']} |")
        lines.append("")
        lines.append("### 关键事实")
        lines.append("")
        gpus = f["gpu"]["gpus"] or []
        gpu_desc = ", ".join(f"{g.get('name')}" for g in gpus) or "（无 GPU）"
        lines.append(f"- GPU：{f['gpu']['count']} 张 — {gpu_desc}；拓扑摘要 `{f['gpu']['topo']['summary']}`")
        lines.append(f"- CPU：{f['cpu'].get('model_name') or '—'}，{f['cpu'].get('cores')} 核，"
                     f"NUMA {f['cpu'].get('numa_nodes')}，governor `{f['cpu'].get('governor')}`")
        lines.append(f"- 内存：THP=`{f['mem'].get('thp')}`，swappiness=`{f['mem'].get('swappiness')}`")
        lines.append(f"- vLLM：`{f.get('vllm_version')}`（{f.get('python3_version')}，hostname={f.get('hostname')}）")
        lines.append(f"- 网络接口：{', '.join(i['name'] for i in f['net']['interfaces']) or '—'}"
                     f"；RDMA/IB：{'有' if f['net']['rdma']['ibstat_present'] else '无'}")
        dk = f.get("docker") or {}
        lines.append(f"- docker：{dk.get('version') or '不可用'}；本地 vllm 镜像："
                     f"{', '.join(dk.get('vllm_images') or []) or '无'}")
        lines.append("")
    lines.append("## 跨机互联")
    lines.append("")
    lines.append(f"- covered：{interconnect['covered']}；{interconnect.get('reason') or '（见 checks）'}")
    for c in interconnect.get("checks", []):
        lines.append(f"  - {c['check']}: `{c.get('result') or '—'}` {c.get('note') or ''}".rstrip())
    lines.append("")
    lines.append(f"## 汇总：pass={total['pass']} warn={total['warn']} fail={total['fail']}")
    lines.append("")
    lines.append("## 下一步（Phase 1.5 门禁）")
    lines.append("")
    if total["fail"]:
        lines.append("有 fail 项 → 按上表修复命令交给人类执行，修好后再跑 `python probe.py` 复查。")
    else:
        lines.append("无 fail 项（仅 warn 可继续）→ 进入 Phase 2 基线实验。")
    lines.append("")
    lines.append(f"模型画像见 `model_profile.json`（calibrated=false，启动耗时将在 Phase 2 用实测校准）。")
    return "\n".join(lines)


# ── CLI 入口 ────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 1 环境探测（只读）")
    ap.add_argument("--target", default="target.yaml")
    ap.add_argument("--harness", default="harness.yaml")
    ap.add_argument("--fixtures", default=None,
                    help="离线模式：从 DIR/<hostname>/<key>.txt 读取预录输出（无需 SSH/GPU）")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--ssh-timeout", type=int, default=30)
    args = ap.parse_args(argv)

    target = load_yaml(args.target)
    harness = load_yaml(args.harness)
    head = validate_target(target)
    mode = "fixtures" if args.fixtures else "live"

    plan = _collect_plan(target, harness)
    os.makedirs(args.out_dir, exist_ok=True)

    # 并行采集所有主机
    raw: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(target["hosts"]))) as ex:
        futs = {ex.submit(collect_host, h, plan[h["name"]], mode, args.fixtures,
                          args.ssh_timeout): h for h in target["hosts"]}
        for fut in concurrent.futures.as_completed(futs):
            h = futs[fut]
            raw[h["name"]] = fut.result()

    facts = {h["name"]: build_host_facts(h, raw[h["name"]]) for h in target["hosts"]}

    gmu = read_gmu_default()
    model_profile = build_model_profile(target.get("model") or {}, facts[head["name"]], gmu)
    wb = model_profile["weights"]["bytes"]
    for h in target["hosts"]:
        facts[h["name"]]["checklist"] = _system_checklist(
            facts[h["name"]], is_head=(h is head), weights_bytes=wb)
    interconnect = interconnect_check(target, mode, args.fixtures)

    env = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "probe": "probe.py (Phase 1)",
        "mode": mode,
        "remote": harness.get("remote") or {},
        "target": {"hosts": target.get("hosts"), "model": target.get("model")},
        "hosts": facts,
        "interconnect": interconnect,
    }

    env_path = os.path.join(args.out_dir, "env.json")
    profile_path = os.path.join(args.out_dir, "model_profile.json")
    report_path = os.path.join(args.out_dir, "env_report.md")
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(model_profile, f, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_env_report(target, facts, interconnect, model_profile, gmu))

    print(f"探测完成：{len(target['hosts'])} 台主机（并行采集，mode={mode}）")
    for name, f in facts.items():
        ck = f["checklist"]
        n = {s: sum(1 for i in ck if i["status"] == s) for s in ("pass", "warn", "fail")}
        print(f"  {name} ({f['role']}): pass={n['pass']} warn={n['warn']} fail={n['fail']}")
    print(f"模型配置：{model_profile.get('model')} → weights.source={model_profile['weights']['source']}")
    print(f"产物：env.json / env_report.md / model_profile.json（{args.out_dir}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
