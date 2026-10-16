#!/usr/bin/env python3
"""probe.py — Phase 1 环境探测（只读 harness 工具）。

由编码 agent 在编排机执行：读取 target.yaml + harness.yaml，
对所有主机经 SSH 并行采集只读事实，产出：
  env.json          — 结构化原始事实（供 program.md 剪枝规则消费）
  env_report.md     — 人类可读报告，含系统参数检查清单（pass/warn/fail + 修复命令）
  model_profile.json— 模型权重/配置估算（供搜索空间剪枝与成本估算）
  host_report.html  — 主机探索报告（GPU/网络/镜像/模型介质/占用），供人类评估该主机

只给 host（未声明 model）也可运行：即「主机探索」阶段——产出 host_report.html 供人类
评估后再下发调优指令（target.yaml 补 model/SLO/budget/work_dir）。

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
MODEL_SCAN_DEPTH = 4                     # 主机探索：模型介质扫描最大目录深度（护栏，防深层递归）
MODEL_SCAN_TIMEOUT_S = 45                # 主机探索：模型介质扫描的远端超时（护栏）
DEFAULT_MODEL_SCAN_DIRS = [              # 主机探索：模型介质扫描默认候选（可被 target.model_scan_dirs 覆盖）
    "/data/model", "/data/models", "/data/deploy", "/nfsdata/models", "~/models",
]

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

def _load_dotenv(path: str = ".env") -> None:
    """把 .env 载入 os.environ（不覆盖已存在的键；`~` 展开），与 run.sh 的 .env 语义一致。

    _ssh 读 SSH_PORT / SSH_KEY，此前须外部 export 才生效——host-only intake 首次即踩坑
    （静默全字段 None）。此处补齐，使 `python probe.py` 与 `bash run.sh` 共用同一份 .env。"""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and key not in os.environ:
            os.environ[key] = os.path.expanduser(val)


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


def _runtime_image(harness: dict, config: dict | None) -> str | None:
    """运行时镜像 = harness.runtime 与 config.yaml.runtime 逐键合并（config 覆盖）。
    config 缺省（None）时容错读仓库根 config.yaml：缺失/非法 → {}（测试可能在无 config.yaml 的目录跑）。"""
    if config is None:
        try:
            config = load_yaml("config.yaml", required=False)
        except Exception:
            config = {}
    runtime = {**(harness.get("runtime") or {}), **((config.get("runtime") or {}))}
    img = runtime.get("image")
    return str(img) if img else None


def _docker_cmd(img: str, mp: str, inner: str) -> str:
    """在镜像内执行采集命令串：模型路径只读挂到容器内同一绝对路径，--entrypoint sh -c。
    整串经 SSH 远端 shell 执行，故内层命令用 shlex.quote 整体引用（远程不再二次解释）。"""
    return (f"docker run --rm --pull=never -v {shlex.quote(mp)}:{shlex.quote(mp)}:ro "
            f"--entrypoint sh {shlex.quote(img)} -c {shlex.quote(inner)}")


def _read_bw_cmd(mp: str) -> str:
    """模型目录顺序读带宽实测：对最大文件做限时只读顺序读（dd of=/dev/null），
    先试 iflag=direct，失败回退不 direct；bs=1M count=1024（约 1GiB，数秒内）。
    dd 的速率汇总写在 **stderr**，须 2>&1 并入 stdout 才能被远端捕获解析（真机教训）；
    无文件/失败静默（|| true 容错，只读零破坏）。"""
    q = shlex.quote(mp)
    inner = (f"f=$(find {q} -type f -printf '%s %p\\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-); "
             f'if [ -n "$f" ]; then dd if="$f" of=/dev/null bs=1M count=1024 iflag=direct 2>&1 '
             f'|| dd if="$f" of=/dev/null bs=1M count=1024 2>&1 || true; fi')
    return inner + " || true"


# 容器内 GPU P2P copy 短时测例：128MiB 块，8 次取均值；同 NUMA 对 0→1、跨 NUMA 对 0→4
#（启发式；少于 5 卡则跨 NUMA 取 0→n-1）。每个 pair 先 ≥2 次不计时预热 copy 并同步，
# 再计时（真机教训：无预热会把 CUDA/P2P 首次初始化开销算进第一个被测对）。输出行
# `P2P_INTRA <GB/s> P2P_CROSS <GB/s>`。
_P2P_SCRIPT = (
    "import torch,time\n"
    "n=torch.cuda.device_count()\n"
    "if n<2:\n"
    " print('P2P_SKIP n<2')\n"
    " raise SystemExit(0)\n"
    "def bw(i,j):\n"
    " x=torch.empty(128*1024*1024,dtype=torch.uint8,device='cuda:%d'%i)\n"
    " y=torch.empty(128*1024*1024,dtype=torch.uint8,device='cuda:%d'%j)\n"
    " for _ in range(2): y.copy_(x)\n"
    " torch.cuda.synchronize()\n"
    " t0=time.perf_counter()\n"
    " for _ in range(8): y.copy_(x)\n"
    " torch.cuda.synchronize()\n"
    " return 0.134217728/((time.perf_counter()-t0)/8)\n"
    "same=bw(0,1)\n"
    "cp=4 if n>=5 else n-1\n"
    "cross=bw(0,cp)\n"
    "print('P2P_INTRA %.3f P2P_CROSS %.3f'%(same,cross))\n"
)


def _docker_p2p_cmd(img: str) -> str:
    """容器内 torch GPU P2P copy 短时实测（同/跨 NUMA 启发式对；短时、只读、零安装）。"""
    return (f"docker run --rm --pull=never --gpus all --ipc host --entrypoint python3 "
            f"{shlex.quote(img)} -c {shlex.quote(_P2P_SCRIPT)} 2>/dev/null || true")


# 单 SSH 调用内的 RDMA 带宽实测（ib_write_bw）：按 sysfs numa_node 挑两张跨 NUMA 的 NIC，
# 两侧都用 -d <hca> 指定各自选中的卡（真机教训：不指定会落到同一默认 HCA，跨 NUMA 意图落空）；
# client **只传 server IP**（ib_write_bw 一收到 IP 即进 client 模式：**server 绝不能带 IP**，否则两侧都去连别人；
# 真机教训）。两侧 `-d <hca>` 钉住跨 NUMA 选中的卡；`--report_gbits` 让输出列为 `BW average[Gb/sec]`（默认是 MB/sec）。
# server 必须 nohup+重定向+</dev/null 后台（否则 ssh 挂到进程结束），用 timeout 15 兜底
#（防孤儿挂死），测完按 PID 精确 kill（禁用 pkill -f——同串会杀自身 shell）。
_RDMA_BW_CMD = """\
out=""; a=""; b=""
for d in /sys/class/infiniband/*; do
  [ -d "$d" ] || continue
  n=${d##*/}
  nn=$(cat "$d/device/numa_node" 2>/dev/null || true)
  nd=$(ls "$d/device/net" 2>/dev/null | head -1)
  [ -n "$nn" ] || continue
  [ -n "$nd" ] || continue
  if [ -z "$out" ]; then out="$n $nn $nd"; continue; fi
  set -- $out
  if [ "$2" != "$nn" ]; then a="$1 $3"; b="$n $nd"; break; fi
done
if [ -z "$a" ] || [ -z "$b" ]; then echo "RDMA_SKIP no cross-numa pair"; exit 0; fi
set -- $a; sa=$1; snd=$2
set -- $b; ca=$1
sip=$(ip -4 -o addr show dev "$snd" 2>/dev/null | awk '{print $4}'); sip=${sip%/*}
if [ -z "$sip" ]; then echo "RDMA_SKIP no ip on netdev"; exit 0; fi
port=18515
if command -v ib_write_bw >/dev/null 2>&1; then
  nohup timeout 15 ib_write_bw -d "$sa" -D 3 -p "$port" --report_gbits >/tmp/probe_ibw_srv.log 2>&1 </dev/null &
  spid=$!
  sleep 2
  ib_write_bw -d "$ca" -D 3 -p "$port" --report_gbits "$sip" 2>&1 || true
  kill "$spid" 2>/dev/null || true
fi
exit 0
"""


def _model_inventory_cmd(dirs: list[str]) -> str:
    """模型介质清单：远端扫描 dirs 下的 config.json 目录，TSV 一行一模型（非阻塞）。

    字段：dir \\t size_bytes \\t n_shards \\t model_type \\t architectures \\t quant_method
          \\t torch_dtype \\t layers \\t heads \\t kv_heads
    用 os.walk 限深度 + safetensors 元数据求和（不做 du 全遍历）；缺 python3 → 空输出。"""
    dirs_literal = "[" + ", ".join(repr(d) for d in dirs) + "]"
    script = f"""import json, os, glob
dirs, depth = {dirs_literal}, {MODEL_SCAN_DEPTH}
found = []
for d in dirs:
    base = os.path.expanduser(d).rstrip("/")
    if not base or not os.path.isdir(base):
        continue
    for root, subdirs, files in os.walk(base):
        rel = os.path.relpath(root, base)
        if rel != "." and rel.count(os.sep) + 1 >= depth:
            subdirs[:] = []
        if "config.json" in files:
            found.append(root)
for m in found:
    try:
        shards = [f for f in glob.glob(os.path.join(m, "*.safetensors")) if os.path.isfile(f)]
        size = sum(os.path.getsize(f) for f in shards)
    except OSError:
        size, shards = 0, []
    vals = ["", "", "", "", "", "", ""]
    try:
        with open(os.path.join(m, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        arch = cfg.get("architectures") or []
        q = cfg.get("quant_method") or (cfg.get("quantization_config") or {{}}).get("quant_method") or ""
        vals = [str(cfg.get("model_type") or ""), str(arch[0] if arch else ""),
                str(q), str(cfg.get("torch_dtype") or ""), str(cfg.get("num_hidden_layers") or ""),
                str(cfg.get("num_attention_heads") or ""), str(cfg.get("num_key_value_heads") or "")]
    except Exception:
        pass
    print("\\t".join([m, str(size), str(len(shards)), *vals]))
"""
    return f"timeout {MODEL_SCAN_TIMEOUT_S} python3 - <<'PY' 2>/dev/null || true\n{script}PY"


def _collect_plan(target: dict, harness: dict, config: dict | None = None) -> dict[str, list]:
    """每台主机的采集计划：list of (key, cmd, fixture 文件名)。
    docker-only：source=local + runtime.image 可解析时，模型探测改为在镜像内执行
    （宿主 SSH 用户可能不可读模型目录，而容器内 root 可读）。"""
    remote = harness.get("remote") or {}
    vllm_python = remote.get("vllm_python", "python")
    model = target.get("model") or {}
    image = _runtime_image(harness, config)
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
            # RDMA 主判据（sysfs + rdma 工具；ibstat/ibv_devinfo 仅补充，缺工具不判假阴性）
            _cmd("rdma_devices", "ls /sys/class/infiniband 2>/dev/null || true"),
            _cmd("rdma_link", "rdma link show 2>/dev/null || true"),
            _cmd("rdma_ports",
                 "for d in /sys/class/infiniband/*/ports/1; do [ -f \"$d/link_layer\" ] || continue; "
                 "n=${d%/ports/1}; n=${n##*/}; echo \"$n $(cat \"$d/link_layer\" 2>/dev/null) "
                 "$(cat \"$d/rate\" 2>/dev/null)\"; done 2>/dev/null || true"),
            _cmd("rdma_numa",
                 "for d in /sys/class/infiniband/*/device/numa_node; do [ -f \"$d\" ] || continue; "
                 "n=${d%/device/numa_node}; n=${n##*/}; echo \"$n $(cat \"$d\" 2>/dev/null)\"; "
                 "done 2>/dev/null || true"),
            _cmd("docker_version", "docker --version 2>/dev/null || true"),
            # 镜像含 size/created：评估磁盘余量与镜像新旧（host 报告「可用 vLLM 镜像」段）
            _cmd("docker_images",
                 "docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}'"
                 " 2>/dev/null || true"),
            # 当前占用 / 共租（host 报告）：运行中容器、监听端口、登录用户
            _cmd("docker_ps",
                 "docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || true"),
            _cmd("listening_ports", "ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null || true"),
            _cmd("logged_in_users", "who 2>/dev/null || true"),
            # 镜像可解析 → 镜像内 vllm --version（docker-only 主机无宿主 pip 版 vllm）；
            # 否则用 pip show（不 import）避免在主机上写入 torch/vllm 缓存
            _cmd("vllm_version",
                 (f"docker run --rm --pull=never --entrypoint vllm {shlex.quote(image)} --version 2>/dev/null || true"
                  if image else f"{shlex.quote(vllm_python)} -m pip show vllm 2>/dev/null || true")),
            _cmd("hostname", "hostname"),
            _cmd("python3_version", "python3 --version 2>&1 || true"),
        ]
        if h.get("role") == "head":
            # 模型介质清单（主机探索）：扫描候选目录，与是否声明 model 无关
            cmds += [_cmd("model_inventory",
                          _model_inventory_cmd(target.get("model_scan_dirs") or DEFAULT_MODEL_SCAN_DIRS),
                          "model_inventory.tsv")]
            src = model.get("source", "hf")
            if src == "local" and model.get("path"):
                mp = model["path"]
                if image:  # docker-only：宿主 SSH 用户不可读模型目录 → 镜像内 root 只读挂载探测
                    cmds += [
                        _cmd("model_config_json",
                             _docker_cmd(image, mp, f"cat {shlex.quote(mp)}/config.json") + " 2>/dev/null || true",
                             "config.json"),
                        _cmd("model_safetensors_index_json",
                             _docker_cmd(image, mp, f"cat {shlex.quote(mp)}/*.safetensors.index.json")
                             + " 2>/dev/null || true",
                             "safetensors.index.json"),
                        _cmd("df_model",
                             _docker_cmd(image, mp, f"df -B1 --output=avail {shlex.quote(mp)}")
                             + " 2>/dev/null | tail -1 || true"),
                        _cmd("read_bw_model",
                             _docker_cmd(image, mp, _read_bw_cmd(mp)) + " 2>/dev/null || true",
                             "read_bw_model.txt"),
                    ]
                else:
                    cmds += [
                        _cmd("model_config_json", f"cat {shlex.quote(mp)}/config.json 2>/dev/null || true", "config.json"),
                        _cmd("model_safetensors_index_json",
                             f"cat {shlex.quote(mp)}/*.safetensors.index.json 2>/dev/null || true",
                             "safetensors.index.json"),
                        _cmd("df_model", f"df -B1 --output=avail {shlex.quote(os.path.dirname(mp))} 2>/dev/null | tail -1 || true"),
                        _cmd("read_bw_model", _read_bw_cmd(mp), "read_bw_model.txt"),
                    ]
            elif model.get("id"):  # source=hf：尝试本地 HF 缓存（~/.cache/huggingface/hub/models--<id>/snapshots/*）
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
        # 本机互联带宽实测（每台主机都尽量采集；不可得 → null+note，不阻塞）：GPU P2P 需镜像
        #（容器测例）；RDMA 脚本自含选卡/起停。属 host 级事实，见 env.json hosts.<name>.interconnect
        if image:
            cmds += [_cmd("gpu_p2p_bw", _docker_p2p_cmd(image), "gpu_p2p_bw.txt")]
        cmds += [_cmd("rdma_bw", _RDMA_BW_CMD, "rdma_bw.txt")]
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


def parse_topo_affinity(text: str | None) -> tuple[dict[int, int | None], dict[int, str | None]]:
    """nvidia-smi topo -m 的 CPU Affinity / NUMA Affinity / GPU NUMA ID 三列 →
    ({gpu_index: numa_node}, {gpu_index: cpu_affinity})。
    假定每行末 3 个空白分隔 token 为 CPU Affinity / NUMA Affinity / GPU NUMA ID；
    若驱动省略尾列则退化为无 NUMA（该信息静默丢弃）。
    NUMA 列 N/A 或非数字 → None。跳过：token<4 的行、列标签头行（首 token 也是 GPU<d>
    但 NUMA Affinity 列是单词而非数字/N/A）。"""
    numa: dict[int, int | None] = {}
    cpu: dict[int, str | None] = {}
    if not text:
        return numa, cpu
    for line in text.splitlines():
        toks = line.split()
        if len(toks) < 4 or not re.match(r"GPU(\d+)$", toks[0]):
            continue
        numa_aff = toks[-2]
        if numa_aff != "N/A" and not numa_aff.isdigit():
            continue  # 头行等非数据行（NUMA Affinity 列不是数字/N/A）
        gi = int(toks[0][3:])
        cpu_aff = toks[-3]
        numa[gi] = int(numa_aff) if numa_aff.isdigit() else None
        cpu[gi] = cpu_aff if cpu_aff != "N/A" else None
    return numa, cpu


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


def parse_rdma_devices(text: str | None) -> list[str]:
    """`ls /sys/class/infiniband` → RDMA 设备名列表（mlx5_0 等）；空/缺 → []。"""
    if not text:
        return []
    return [d.strip() for d in text.splitlines() if d.strip()]


def parse_rdma_link(text: str | None) -> dict[str, dict]:
    """`rdma link show` → {dev: {netdev, state, physical_state}}。
    真机行格式（行尾有尾随空格，netdev 仅 linked 时出现）:
    `link mlx5_0/1 state ACTIVE physical_state LINK_UP netdev enp24s0f0np0 `"""
    out: dict[str, dict] = {}
    if not text:
        return out
    for line in text.splitlines():
        m = re.match(r"^link (\S+)/(\d+) state (\S+) physical_state (\S+)(?:\s+netdev (\S+))?\s*$", line)
        if not m:
            continue
        out[m.group(1)] = {"netdev": m.group(5), "state": m.group(3), "physical_state": m.group(4)}
    return out


def parse_rdma_ports(text: str | None) -> dict[str, dict]:
    """sysfs ports 循环输出（每行 `<dev> <link_layer> <rate 全文>`）→ {dev: {link_layer, rate}}。
    rate 为多 token（如 `400 Gb/sec (4X NDR)`），拼接保留原文。"""
    out: dict[str, dict] = {}
    if not text:
        return out
    for line in text.splitlines():
        toks = line.split()
        if len(toks) >= 2:
            out[toks[0]] = {"link_layer": toks[1], "rate": " ".join(toks[2:]) or None}
    return out


def parse_rdma_numa(text: str | None) -> dict[str, int | None]:
    """sysfs numa 循环输出（每行 `<dev> <numa_node>`）→ {dev: numa_node}。
    值可能为 -1（无 NUMA 归属）→ None。"""
    out: dict[str, int | None] = {}
    if not text:
        return out
    for line in text.splitlines():
        toks = line.split()
        if len(toks) >= 2 and toks[1].lstrip("-").isdigit():
            out[toks[0]] = int(toks[1]) if int(toks[1]) >= 0 else None
    return out


def parse_rate_gbps(rate: str | None) -> int | None:
    """sysfs `/ports/1/rate` → Gbps。真机格式 `400 Gb/sec (4X NDR)`，
    兼容 `100 Gb/sec (4X EDR)` / `25 Gb/sec` / `200 Gb/sec (2X HDR)` 等变体。"""
    if not rate:
        return None
    m = re.search(r"(\d+)\s*Gb", rate)
    return int(m.group(1)) if m else None


def parse_topo_nics(text: str | None, gpu_count: int) -> list[dict]:
    """nvidia-smi topo -m 的 NIC 行 → [{name, gpu_links: {gpu_index: 连接码}, numa_node}]。
    行格式与 GPU 行一致：首 token NIC<n>，随后 gpu_count 个 GPU 连接码，
    末 3 列 CPU Affinity / NUMA Affinity / GPU NUMA ID（NUMA Affinity 非数字 → None）。"""
    nics: list[dict] = []
    if not text or not gpu_count:
        return nics
    for line in text.splitlines():
        toks = line.split()
        if len(toks) < gpu_count + 4 or not re.match(r"^NIC(\d+)$", toks[0]):
            continue
        gpu_links = {str(j): toks[1 + j] for j in range(gpu_count)}
        numa_aff = toks[-2]
        nics.append({"name": toks[0], "index": int(toks[0][3:]), "gpu_links": gpu_links,
                     "numa_node": int(numa_aff) if numa_aff.isdigit() else None})
    return nics


def parse_dd_bw(text: str | None) -> float | None:
    """`dd if=<f> of=/dev/null` 尾行 `X bytes copied, T s, S MB/s` → 顺序读带宽 Gbps。
    容错：无速率行/坏行 → None（探测不中断）。"""
    if not text:
        return None
    for line in reversed(text.splitlines()):
        m = re.search(r"copied, ([\d.]+) s, ([\d.]+) (MB|GB)/s", line)
        if m:
            v = float(m.group(2))
            return v * 8 / (1000 if m.group(3) == "MB" else 1)
    return None


def parse_gpu_p2p_bw(rec: dict | None) -> dict:
    """容器内 torch p2p copy 输出 → {intra_numa, cross_numa}（GB/s）。
    行格式 `P2P_INTRA 43.35 P2P_CROSS 33.87`；不可得 → None + note（绝不抛错）。"""
    if not rec or not rec.get("ok") or not rec.get("value"):
        note = f"gpu_p2p_bw: {rec.get('note') or '未采集'}" if rec else "gpu_p2p_bw: 未采集"
        return {"intra_numa": None, "cross_numa": None, "note": note}
    out: dict = {"intra_numa": None, "cross_numa": None}
    for m in re.finditer(r"P2P_INTRA\s+([\d.]+)|P2P_CROSS\s+([\d.]+)", rec["value"]):
        if m.group(1):
            out["intra_numa"] = float(m.group(1))
        if m.group(2):
            out["cross_numa"] = float(m.group(2))
    if out["intra_numa"] is None or out["cross_numa"] is None:
        out["note"] = "gpu_p2p_bw: 输出缺 P2P_INTRA/P2P_CROSS 行"
    return out


def parse_rdma_bw(rec: dict | None) -> dict:
    """ib_write_bw client 输出 → {bw_gbps}（BW average[Gb/sec]）。
    数据行形如 ` 65536 0 352.9 341.7 0.000651`（第 4 列为平均带宽）；
    不可得/RDMA_SKIP → None + note（绝不抛错）。"""
    if not rec or not rec.get("ok") or not rec.get("value"):
        note = f"rdma_bw: {rec.get('note') or '未采集'}" if rec else "rdma_bw: 未采集"
        return {"bw_gbps": None, "note": note}
    v = None
    for line in rec["value"].splitlines():
        m = re.match(r"^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s", line)
        if m:
            v = float(m.group(4))
    if v is None:
        skip = re.search(r"RDMA_SKIP\s*(.*)", rec["value"])
        note = f"rdma_bw: {skip.group(1).strip()}" if skip else "rdma_bw: 无 BW average 数据行"
        return {"bw_gbps": None, "note": note}
    return {"bw_gbps": v}


def parse_thp(text: str | None) -> str | None:
    """/sys/kernel/mm/transparent_hugepage/enabled → 当前模式（[] 括起的那个）。"""
    if not text:
        return None
    m = re.search(r"\[(\w+)\]", text)
    return m.group(1) if m else None


def _int_or_none(x) -> int | None:
    try:
        return int(str(x).strip())
    except (TypeError, ValueError):
        return None


def _str_or_none(x) -> str | None:
    s = (x or "").strip() if isinstance(x, str) else (str(x).strip() if x is not None else "")
    return s or None


def parse_docker_images_detail(text: str | None) -> list[dict]:
    """`docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}'`
    → [{repo, size, created}]。仅有名字的行也能解析（size/created 为 None）。"""
    out: list[dict] = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        p = line.split("\t")
        repo = p[0].strip()
        if not repo:
            continue
        out.append({"repo": repo,
                    "size": p[1].strip() if len(p) > 1 and p[1].strip() else None,
                    "created": p[2].strip() if len(p) > 2 and p[2].strip() else None})
    return out


def parse_docker_ps(text: str | None) -> list[dict]:
    """`docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'` → [{name,image,status,ports}]。"""
    out: list[dict] = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        p = line.split("\t")
        out.append({"name": p[0].strip(),
                    "image": _str_or_none(p[1] if len(p) > 1 else None),
                    "status": _str_or_none(p[2] if len(p) > 2 else None),
                    "ports": _str_or_none(p[3] if len(p) > 3 else None)})
    return out


def parse_listening_ports(text: str | None) -> list[dict]:
    """`ss -ltn` → [{addr, port}]（去重、按端口排序）。Local Address:Port 是第 4 列。"""
    out: list[dict] = []
    seen: set[int] = set()
    for line in (text or "").splitlines():
        cols = line.split()
        if len(cols) < 4 or not line.lstrip().startswith("LISTEN"):
            continue
        m = re.search(r":(\d+)$", cols[3])
        if not m:
            continue
        port = int(m.group(1))
        if port in seen:
            continue
        seen.add(port)
        out.append({"addr": cols[3], "port": port})
    return sorted(out, key=lambda d: d["port"])


def parse_who(text: str | None) -> list[str]:
    """`who` → 非空行列表（登录会话）。"""
    return [l.strip() for l in (text or "").splitlines() if l.strip()]


def parse_model_inventory(text: str | None) -> list[dict]:
    """model_inventory TSV → [{dir,size_bytes,n_shards,model_type,architectures,
    quant_method,torch_dtype,layers,heads,kv_heads}]，按大小降序。坏行跳过。"""
    out: list[dict] = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        p = line.split("\t")
        if len(p) < 3 or not p[0].strip():
            continue
        out.append({
            "dir": p[0].strip(),
            "size_bytes": _int_or_none(p[1] if len(p) > 1 else None),
            "n_shards": _int_or_none(p[2] if len(p) > 2 else None),
            "model_type": _str_or_none(p[3] if len(p) > 3 else None),
            "architectures": _str_or_none(p[4] if len(p) > 4 else None),
            "quant_method": _str_or_none(p[5] if len(p) > 5 else None),
            "torch_dtype": _str_or_none(p[6] if len(p) > 6 else None),
            "layers": _int_or_none(p[7] if len(p) > 7 else None),
            "heads": _int_or_none(p[8] if len(p) > 8 else None),
            "kv_heads": _int_or_none(p[9] if len(p) > 9 else None),
        })
    return sorted(out, key=lambda d: -(d["size_bytes"] or 0))


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
    """合法 tp 集合：tp 须整除 num_attention_heads，且 kv_heads>1（GQA）时还须同时整除
    num_key_value_heads。例外：kv_heads<=1（MLA 的 num_key_value_heads=1、MQA）时 vLLM
    会在 tp>kv_heads 时复制 KV heads，故合法 tp 只受 heads 约数约束（program.md:72）。
    且单卡权重 ≤ gpu_mem × gmu。min_tp_for_weights 为满足显存的最小 tp。"""
    heads = config.get("num_attention_heads")
    if not heads:
        return {"valid_divisors": [], "min_tp_for_weights": None}
    kv_heads = config.get("num_key_value_heads") or heads
    wb = weights_bytes
    if wb is None:
        wb, _ = weight_bytes(None, config)
    if kv_heads <= 1:  # MLA（num_key_value_heads=1）/ MQA：只受 heads 约束
        divs = [t for t in range(1, heads + 1) if heads % t == 0]
    else:              # GQA：同时整除 heads 与 kv_heads
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
    numa_map, cpu_map = parse_topo_affinity(raw.get("gpu_topo", {}).get("value"))
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

    read_bw = None
    rec = raw.get("read_bw_model", {})
    if rec.get("ok") and rec.get("value"):
        read_bw = parse_dd_bw(rec["value"])

    # RDMA 主判据：sysfs 设备名 + rdma link（dev↔netdev/state）+ sysfs link_layer/rate/numa_node；
    # ibstat/ibv_devinfo 仅补充（缺 ibstat 不再误判「无 RDMA」）
    rdma_devs = parse_rdma_devices(raw.get("rdma_devices", {}).get("value"))
    rdma_links = parse_rdma_link(raw.get("rdma_link", {}).get("value"))
    rdma_ports = parse_rdma_ports(raw.get("rdma_ports", {}).get("value"))
    rdma_numa_map = parse_rdma_numa(raw.get("rdma_numa", {}).get("value"))
    ib_ok = raw.get("ibstat", {}).get("ok")
    ib_devs_raw = raw.get("ibv_devinfo", {}).get("value")
    ib_devs = [l.strip() for l in ib_devs_raw.splitlines() if l.strip()] if ib_devs_raw else None
    devices = []
    for d in rdma_devs:
        port = rdma_ports.get(d, {})
        link = rdma_links.get(d, {})
        devices.append({
            "name": d,
            "netdev": link.get("netdev"),
            "link_layer": port.get("link_layer"),
            "state": link.get("state"),
            "physical_state": link.get("physical_state"),
            "speed_gbps": parse_rate_gbps(port.get("rate")),
            "numa_node": rdma_numa_map.get(d),
            "gpu_links": {},
        })
    if devices:
        # 补充 NIC↔GPU 拓扑级别（topo NIC 行）；numa 以 sysfs 为准，topo 仅兜底
        nics = parse_topo_nics(raw.get("gpu_topo", {}).get("value"), len(gpus or []))
        for i, d in enumerate(devices):
            if i < len(nics):
                if d["numa_node"] is None:
                    d["numa_node"] = nics[i]["numa_node"]
                d["gpu_links"] = nics[i]["gpu_links"]
    if not devices:
        kind = "none"
    elif any(d["link_layer"] == "Ethernet" for d in devices):
        kind = "roce"
    else:
        kind = "infiniband"
    rdma = {
        "kind": kind,
        "devices": devices,
        "ibstat_present": bool(ib_ok),   # 兼容既有字段
        "ib_devices": ib_devs,           # 兼容既有字段
    }

    docker_ver = (raw.get("docker_version", {}).get("value") or "").strip() or None
    docker_images = parse_docker_images_detail(raw.get("docker_images", {}).get("value"))
    docker_vllm = [i["repo"] for i in docker_images if "vllm" in i["repo"].lower()] or None

    # NUMA 放置：numa_node 合并进每张 GPU；numa_groups 按 topo 的 GPU 索引分组
    #（真实主机上 topo 与 gpu_smi 的 GPU 集合一致；None=N/A 的组放最后）
    for g in gpus or []:
        gi = g.get("index")
        if gi is not None:
            g["numa_node"] = numa_map.get(gi)
    numa_groups: list[dict] = []
    if numa_map:
        by_node: dict[int | None, list[int]] = {}
        for gi, nn in sorted(numa_map.items()):
            by_node.setdefault(nn, []).append(gi)
        numa_groups = [{"numa_node": k, "gpus": v}
                       for k, v in sorted(by_node.items(),
                                          key=lambda kv: (kv[0] is None,
                                                          kv[0] if kv[0] is not None else 0))]

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
        "numa_groups": numa_groups,
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
            "rdma": rdma,
        },
        "disk": {"model_path": None, "avail_bytes": df_avail, "read_bw_gbps": read_bw},
        "docker": {
            "version": docker_ver,
            "images": docker_images,       # 全部镜像（repo/size/created）
            "vllm_images": docker_vllm,    # vllm 过滤后的镜像名（env_report.md 用）
            "containers": parse_docker_ps(raw.get("docker_ps", {}).get("value")),
        },
        # 主机探索（host 报告）用：当前占用 / 共租 + 模型介质清单
        "ports": {"listening": parse_listening_ports(raw.get("listening_ports", {}).get("value"))},
        "users": {"logged_in": parse_who(raw.get("logged_in_users", {}).get("value"))},
        "model_inventory": parse_model_inventory(raw.get("model_inventory", {}).get("value")),
        "model_src": model_src,
    }


def format_rdma(rdma: dict) -> str:
    """RDMA 描述（env_report 关键事实行）：
    无 → 「无」；RoCE → 「RoCE(Ethernet) ×N（速率 Gb/s）」；InfiniBand → 「InfiniBand ×N（速率 Gb/s）」。"""
    kind = rdma.get("kind")
    if kind not in ("infiniband", "roce"):
        return "无"
    devs = rdma.get("devices") or []
    speeds = sorted({d.get("speed_gbps") for d in devs if d.get("speed_gbps")})
    sp = f"（{'/'.join(str(s) for s in speeds)} Gb/s）" if speeds else ""
    label = "RoCE(Ethernet)" if kind == "roce" else "InfiniBand"
    return f"{label} ×{len(devs)}{sp}"


def _system_checklist(facts: dict, is_head: bool, weights_bytes: int | None,
                      model_declared: bool = True) -> list[dict]:
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
            # 模板「只建议、不自动改」的系统级旋钮；不影响测量有效性（只影响驱动初始化/启动时延），不阻断门禁
            items.append({"name": "GPU 持久模式", "status": "warn",
                          "detail": f"GPU {bad} 未开启持久模式",
                          "fix": "sudo nvidia-smi -pm 1"})
        else:
            items.append({"name": "GPU 持久模式", "status": "pass", "detail": "全部 Enabled", "fix": "—"})

    if is_head:
        cfg = facts["model_src"]["config_json"]
        if not model_declared:
            # host-only intake（只给主机、未声明 model）：模型路径尚不适用，不应判 fail
            items.append({"name": "模型权重路径", "status": "warn",
                          "detail": "未声明 model（主机探索阶段）；待下发调优指令后核对",
                          "fix": "在 target.yaml 填 model 后重跑 probe.py"})
        elif cfg is None:
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
    if rdma.get("kind") not in ("infiniband", "roce"):
        items.append({"name": "RDMA 网卡 MTU", "status": "pass",
                      "detail": "未启用 RDMA（无 InfiniBand/RoCE）", "fix": "—"})
    else:  # InfiniBand 与 RoCE 都检查（RoCE 走以太网接口，也可能低于 MTU 阈值）
        low = [i["name"] for i in facts["net"]["interfaces"] if i.get("mtu", 0) < MTU_WARN_BELOW]
        if low:
            items.append({"name": "RDMA 网卡 MTU", "status": "warn",
                          "detail": f"接口 {low} MTU < {MTU_WARN_BELOW}",
                          "fix": f"sudo ip link set {low[0]} mtu 9000"})
        else:
            items.append({"name": "RDMA 网卡 MTU", "status": "pass", "detail": "MTU ≥ 9000 或未检测到接口", "fix": "—"})

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

    # NUMA 放置（advisory，仅建议——本模板从不自动绑定系统参数，人工应用）
    ngroups = facts.get("numa_groups") or []
    real = [g for g in ngroups if g["numa_node"] is not None]
    if not real:
        items.append({"name": "NUMA 放置", "status": "pass",
                      "detail": "无有效 NUMA Affinity 数据（topo 未提供或全为 N/A）", "fix": "—"})
    else:
        desc = ", ".join(f"{g['numa_node']}:{g['gpus']}" for g in real)
        if len(real) <= 1:
            items.append({"name": "NUMA 放置", "status": "pass",
                          "detail": f"单 NUMA 节点（numa_groups={{{desc}}}）", "fix": "—"})
        elif any(len(g["gpus"]) > 1 for g in real):
            items.append({
                "name": "NUMA 放置", "status": "warn",
                "detail": (f"多 NUMA（{desc}）：tp 组应限制在同一 NUMA 节点内的 GPU（依据 numa_groups）；"
                           "worker/engine 进程的 CPU 亲和应 align 到该节点"
                           "（numactl --cpunodebind/--membind；系统级绑定仅建议，人工应用）"),
                "fix": "numactl --cpunodebind=<node> --membind=<node> 绑定到目标 NUMA 节点（人工应用，模板不自动改）",
            })
        else:
            items.append({"name": "NUMA 放置", "status": "pass",
                          "detail": f"多 NUMA 但每节点仅 1 GPU（numa_groups={{{desc}}}）", "fix": "—"})

    return items


# ── 模型画像 ────────────────────────────────────────────────────────────

def build_model_profile(model: dict, head_facts: dict, gmu: float) -> dict:
    """组装 model_profile.json（见 probe 需求 §4 字段契约）。"""
    src = head_facts["model_src"]
    config = normalize_config(src["config_json"])
    gpus = head_facts["gpu"]["gpus"] or []
    # parse_gpu_smi 的 memory.total 单位是 MiB → 转字节再用于显存估算（与权重字节一致）
    gpu_mem_bytes = min((g.get("memory_total") or 0 for g in gpus), default=0) * 1024 * 1024

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
    tp = valid_tp_divisors(config, gpu_mem_bytes, gmu, wb) if gpu_mem_bytes else {
        "valid_divisors": [], "min_tp_for_weights": None}
    kv_bytes = kv_bytes_per_token(config)
    max_len = config.get("max_position_embeddings")
    est_tokens = None
    kv_note = None
    if kv_bytes is None:
        kv_note = "arch not covered; rely on baseline-boot calibration"
    elif tp["min_tp_for_weights"] is None:
        kv_note = "权重在合法 tp 集合内无法装下（无法估算 KV 容量）"
    elif gpu_mem_bytes and wb:
        kv_budget = gpu_mem_bytes * gmu - wb / tp["min_tp_for_weights"]
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


# ── 搜索空间（机器可读候选池：search_space.json）────────────────────────

def _gpu_set_str(gpus: list[int]) -> str:
    """GPU 索引子集 → CUDA_VISIBLE_DEVICES 串：连续区间 `0-3`，否则逗号 `0,2,4`。"""
    if not gpus:
        return ""
    if gpus == list(range(gpus[0], gpus[0] + len(gpus))):
        return f"{gpus[0]}-{gpus[-1]}"
    return ",".join(str(g) for g in gpus)


def build_search_space(model_profile: dict, head_facts: dict,
                       gmu_suggested: float | None = None,
                       host_count: int = 1) -> dict:
    """从 Phase 1 采集事实派生「机器可读候选池」（写 <out-dir>/search_space.json）。

    只产出事实可推导的项（合法 tp 约数 ∩ 显存 ∩ 主机 GPU 数上限、NUMA 分组、RDMA 判据），
    不猜魔法值；每条候选/剪枝都带 reason（引用 program.md 剪枝规则）。gating 规则：
    - tp：valid_divisors 按 min_tp_for_weights 过滤且**受主机 GPU 数上限**（tp ≤ gpu_count
      才可执行；min_tp 缺失 → 装不下，记 pruned；上限滤空 → 本机无法起服，记 pruned）；
    - placement：每个 NUMA 节点整组 + 全 GPU 候选（仅 tp 在（已截断的）合法集内）；
    - NCCL_P2P_LEVEL 仅 tp>1 且拓扑 pcie/mixed；单节点：P2P/SHM/IB 一律排除（节点内不走 NET）；
    - 多节点 + RDMA：NCCL_IB_HCA / NCCL_NET_GDR_LEVEL 启用；多节点 PCIe+RoCE 附 SHM_DISABLE 配方。
    """
    tp_info = model_profile.get("tp") or {}
    divisors = [int(t) for t in (tp_info.get("valid_divisors") or [])]
    min_tp = tp_info.get("min_tp_for_weights")
    gpu_count = int((head_facts.get("gpu") or {}).get("count") or 0)
    topo_summary = ((head_facts.get("gpu") or {}).get("topo") or {}).get("summary")
    numa_groups = [g["gpus"] for g in (head_facts.get("numa_groups") or []) if g.get("gpus")]
    rdma = (head_facts.get("net") or {}).get("rdma") or {}
    rdma_kind = rdma.get("kind")
    single_node = int(host_count or 1) == 1

    tp_values: list[int] = []
    pruned: list[dict] = []
    if not divisors:
        pruned.append({"item": "tensor-parallel-size",
                       "reason": "无合法 tp 约数（num_attention_heads 缺失或未采集）"})
    elif min_tp is None:
        pruned.append({"item": "tensor-parallel-size",
                       "reason": "min_tp_for_weights=None：合法 tp 集合内装不下权重（显存不足）"})
    else:
        too_small = [t for t in divisors if t < min_tp]
        too_many = [t for t in divisors if gpu_count > 0 and t > gpu_count]
        feasible = [t for t in divisors if t >= min_tp]
        if gpu_count > 0:                      # 主机 GPU 数上限（不可执行的 tp 不产出）
            feasible = [t for t in feasible if t <= gpu_count]
        tp_values = feasible
        if too_small:
            pruned.append({"item": f"tensor-parallel-size < {min_tp}（{too_small}）",
                           "reason": f"单卡权重装不下：min_tp_for_weights={min_tp}"})
        if too_many:
            pruned.append({"item": f"tensor-parallel-size > {gpu_count}（{too_many}）",
                           "reason": f"超出本主机 GPU 数（gpu.count={gpu_count}）：不可执行"})
        if not tp_values:
            pruned.append({"item": "tensor-parallel-size",
                           "reason": f"模型无法在本主机 GPU 数上起服：min_tp_for_weights={min_tp} "
                                      f"> 可见 GPU 数 {gpu_count}（合法约数 {divisors}）"})

    placement: list[dict] = []
    for grp in numa_groups:
        if len(grp) in tp_values:
            placement.append({"cuda_visible_devices": _gpu_set_str(grp), "tp": len(grp),
                              "reason": f"同一 NUMA 节点内的 GPU 子集 {grp}（numa_groups）"})
    if gpu_count in tp_values:
        cross = "；跨 NUMA" if len(numa_groups) > 1 else ""
        placement.append({"cuda_visible_devices": _gpu_set_str(list(range(gpu_count))),
                          "tp": gpu_count,
                          "reason": f"全部 {gpu_count} 卡：最大并行度{cross}"})
    # M2：单 NUMA 组覆盖全部 GPU 时节点候选与全卡候选相同 → 按 (devices, tp) 去重（保持首见顺序）
    placement_dedup: list[dict] = []
    seen_pairs: set[tuple[str, int]] = set()
    for p in placement:
        key = (p["cuda_visible_devices"], p["tp"])
        if key not in seen_pairs:
            seen_pairs.add(key)
            placement_dedup.append(p)
    placement = placement_dedup

    env: dict[str, dict] = {}
    # B1：P2P_LEVEL 的门控是「存在 tp>1 的多卡候选」，不是「候选个数 >1」
    if topo_summary in ("pcie", "mixed") and any(t > 1 for t in tp_values):
        env["NCCL_P2P_LEVEL"] = {
            "values": ["PXB", "SYS"],
            "requires": "tp>1 且拓扑存在 P2P 梯度（pcie/mixed）",
            "reason": "PCIe 域内 P2P cutoff 候选（docs/vllm-facts.md §6：PXB=同 PCIe switch、"
                      "SYS=跨 NUMA；PHB 及以上默认关 P2P 回落 SHM）"}
    elif topo_summary == "nvlink-full":
        env["NCCL_P2P_LEVEL"] = {
            "values": [],
            "reason": "拓扑 nvlink-full：NVLink 全互联，P2P 已是最快路径，无需距离 cutoff"}
    elif topo_summary is None:
        env["NCCL_P2P_LEVEL"] = {
            "values": [],
            "reason": "拓扑未采集：P2P cutoff 待定（先补全 gpu_topo 再评估），不得假设单卡"}
    else:
        env["NCCL_P2P_LEVEL"] = {
            "values": [],
            "reason": "单卡或无需 NCCL cutoff"}
    if single_node:
        # 单节点多 NUMA PCIe 门控：节点内 GPU-GPU 不走 NET；禁 P2P/SHM 反而砍掉节点内最快路径
        env["NCCL_P2P_DISABLE"] = {"values": [], "reason": "单节点：砍掉节点内最快路径（P2P），通常负优化"}
        env["NCCL_SHM_DISABLE"] = {"values": [], "reason": "单节点：无 NET 语义，强制绕主机内存回环，通常负优化"}
        env["NCCL_IB_HCA"] = {"values": [], "reason": "单节点：节点内 GPU-GPU 不走 NET（IB/RoCE 仅跨节点用）"}
    elif rdma_kind in ("roce", "infiniband"):
        hcas = [f"={d.get('name')}:1" for d in (rdma.get("devices") or [])
                if d.get("state") == "ACTIVE"]
        env["NCCL_IB_HCA"] = {"values": hcas, "requires": "跨节点多机 + RDMA 已就绪",
                              "reason": f"跨节点走 RDMA（{rdma_kind}）：HCA 精确匹配候选（docs §6 `=` 前缀）"}
        env["NCCL_NET_GDR_LEVEL"] = {"values": ["2", "3"], "requires": "跨节点多机 + RDMA",
                                      "reason": "GPU Direct RDMA 等级候选（docs §6：2/3 直通/中间缓冲；0 禁用）"}
        if topo_summary in ("pcie", "mixed"):
            env["NCCL_SHM_DISABLE"] = {"values": ["1"], "requires": "跨节点 + PCIe 域内 P2P",
                                        "reason": "跨节点 PCIe+RoCE 配方（program.md §网络）："
                                                  "域内 P2P 直连 + 跨域走 RoCE，防回落主机内存/SHM"}
    else:
        env["NCCL_IB_HCA"] = {"values": [], "reason": f"无 RDMA（kind={rdma_kind}）：跨节点网络不可用"}
        env["NCCL_NET_GDR_LEVEL"] = {"values": [], "reason": "无 RDMA：GDR 无意义"}

    gmu_block: dict = {}
    if gmu_suggested is not None:
        gmu_block["gpu-memory-utilization"] = {
            "values": [round(gmu_suggested, 4)],
            "reason": "gmu 预检建议值（--detect-gmu / --load-format dummy 实测最大可用）"}

    return {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": "head",
        "gates": {"topology": topo_summary, "single_node": single_node,
                  "numa_groups": numa_groups, "rdma_kind": rdma_kind},
        "serve": {
            "tensor-parallel-size": {
                "values": tp_values,
                "reason": (f"合法 tp 约数 ∩ 装得下 ∩ 主机 GPU 数上限"
                           f"（min_tp_for_weights={min_tp}, gpu_count={gpu_count}）"),
                "phase": "structural"},
            **gmu_block,
        },
        "placement": placement,
        "env": env,
        "cheap_knobs": [
            {"name": "max-num-seqs", "phase": "cheap"},
            {"name": "max-num-batched-tokens", "phase": "cheap"},
            {"name": "gpu-memory-utilization", "phase": "cheap"},
            {"name": "kv-cache-dtype", "phase": "cheap"},
            {"name": "enable-prefix-caching", "phase": "cheap"},
            {"name": "tensor-parallel-size", "phase": "structural"},
            {"name": "quantization", "phase": "structural"},
        ],
        "pruned": pruned,
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
                      model_profile: dict, gmu: float,
                      search_space: dict | None = None) -> str:
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
        ng = f.get("numa_groups") or []
        if ng:
            ng_desc = ", ".join(f"{g['numa_node']}:{g['gpus']}" for g in ng)
            lines.append(f"- NUMA 分组：{{{ng_desc}}}")
        lines.append(f"- CPU：{f['cpu'].get('model_name') or '—'}，{f['cpu'].get('cores')} 核，"
                     f"NUMA {f['cpu'].get('numa_nodes')}，governor `{f['cpu'].get('governor')}`")
        lines.append(f"- 内存：THP=`{f['mem'].get('thp')}`，swappiness=`{f['mem'].get('swappiness')}`")
        lines.append(f"- vLLM：`{f.get('vllm_version')}`（{f.get('python3_version')}，hostname={f.get('hostname')}）")
        lines.append(f"- 网络接口：{', '.join(i['name'] for i in f['net']['interfaces']) or '—'}"
                     f"；RDMA：{format_rdma(f['net']['rdma'])}")
        dk = f.get("docker") or {}
        lines.append(f"- docker：{dk.get('version') or '不可用'}；本地 vllm 镜像："
                     f"{', '.join(dk.get('vllm_images') or []) or '无'}")
        icb = f.get("interconnect") or {}
        p2p = icb.get("gpu_p2p_bw_gbps") or {}
        rdma = icb.get("rdma_bw_gbps") or {}
        lines.append(f"- 互联带宽：GPU P2P intra_numa={p2p.get('intra_numa')} GB/s / "
                     f"cross_numa={p2p.get('cross_numa')} GB/s；RDMA={rdma.get('bw_gbps')} Gb/s"
                     f"（容器/ib_write_bw 实测；不可得见 env.json note）")
        lines.append("")
    lines.append("## 跨机互联")
    lines.append("")
    lines.append(f"- covered：{interconnect['covered']}；{interconnect.get('reason') or '（见 checks）'}")
    for c in interconnect.get("checks", []):
        lines.append(f"  - {c['check']}: `{c.get('result') or '—'}` {c.get('note') or ''}".rstrip())
    lines.append("")
    lines.append(f"## 汇总：pass={total['pass']} warn={total['warn']} fail={total['fail']}")
    lines.append("")
    lines.append("## 搜索空间（初值）")
    lines.append("")
    if not search_space:
        lines.append("未生成 search_space.json（事实缺失，需修复后重跑 Phase 1）。")
    else:
        lines.append("- 机器可读候选池见 `search_space.json`（本目录）；agent 据此选候选并引用各条目的 `reason`，"
                     "`pruned` 条目在事实不变前不得重试。")
        ss_tp = (search_space.get("serve") or {}).get("tensor-parallel-size") or {}
        lines.append(f"- serve.tensor-parallel-size 候选：`{ss_tp.get('values') or '[]'}`（{ss_tp.get('reason')}）")
        pl = search_space.get("placement") or []
        if pl:
            desc = "、".join(f"`{p['cuda_visible_devices']}`(tp={p['tp']})" for p in pl)
            lines.append(f"- placement 候选：{desc}")
        env_ss = search_space.get("env") or {}
        nccl = [f"`{k}`: {v.get('values')}" for k, v in env_ss.items() if v.get("values")]
        if nccl:
            lines.append(f"- env（NCCL）候选：{'；'.join(nccl)}")
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
    _load_dotenv()   # 与 run.sh 一致：SSH_PORT / SSH_KEY 来自 .env

    target = load_yaml(args.target)
    harness = load_yaml(args.harness)
    config = load_yaml("config.yaml", required=False)
    head = validate_target(target)
    mode = "fixtures" if args.fixtures else "live"

    plan = _collect_plan(target, harness, config)
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
    # disk.model_path：head 上模型目录（build_host_facts 无 target 上下文，此处补）
    mp = (target.get("model") or {}).get("path")
    if mp:
        facts[head["name"]]["disk"]["model_path"] = mp

    gmu = read_gmu_default()
    model_profile = build_model_profile(target.get("model") or {}, facts[head["name"]], gmu)
    # gmu 建议值：out-dir 已有 gmu_probe.json 时并入搜索空间（Phase 1.5/2 探测产物）
    gmu_suggested = None
    gmu_probe_path = os.path.join(args.out_dir, "gmu_probe.json")
    if os.path.exists(gmu_probe_path):
        try:
            with open(gmu_probe_path, encoding="utf-8") as f:
                gmu_suggested = (json.load(f)).get("gmu_suggested")
        except (OSError, ValueError):
            gmu_suggested = None
    search_space = build_search_space(model_profile, facts[head["name"]],
                                      gmu_suggested=gmu_suggested,
                                      host_count=len(target["hosts"]))
    wb = model_profile["weights"]["bytes"]
    model_declared = bool((target.get("model") or {}).get("path")
                          or (target.get("model") or {}).get("id"))
    # 端口冲突：与 harness.remote.vllm_port 比对（build_host_facts 无 harness 上下文，此处补）
    vllm_port = (harness.get("remote") or {}).get("vllm_port")
    for h in target["hosts"]:
        f = facts[h["name"]]
        f["checklist"] = _system_checklist(
            f, is_head=(h is head), weights_bytes=wb, model_declared=model_declared)
        listening = (f.get("ports") or {}).get("listening") or []
        f["ports"]["vllm_port_conflict"] = bool(vllm_port) and any(
            p.get("port") == vllm_port for p in listening)
    interconnect = interconnect_check(target, mode, args.fixtures)
    # 本机互联带宽：host 级事实（hosts.<name>.interconnect）——fixtures 模式从 <key>.txt 读、
    # live 模式由 plan 命令短时实测；每台主机都产出该键，不可得 → null + note，绝不阻塞。
    # 顶层 interconnect 只保留「跨机互联检查」（covered/reason/checks），不再承载带宽数值。
    for h in target["hosts"]:
        facts[h["name"]]["interconnect"] = {
            "gpu_p2p_bw_gbps": parse_gpu_p2p_bw(raw[h["name"]].get("gpu_p2p_bw")),
            "rdma_bw_gbps": parse_rdma_bw(raw[h["name"]].get("rdma_bw")),
        }

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
    search_space_path = os.path.join(args.out_dir, "search_space.json")
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(model_profile, f, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_env_report(target, facts, interconnect, model_profile, gmu,
                                  search_space=search_space))
    with open(search_space_path, "w", encoding="utf-8") as f:
        json.dump(search_space, f, ensure_ascii=False, indent=2)

    # 主机探索报告（人类评估用）：渲染失败不阻塞探测（附加产物）
    host_report_path = os.path.join(args.out_dir, "host_report.html")
    try:
        import host_report
        html = host_report.render_host_report(target, facts, harness, model_profile)
    except Exception as e:  # noqa: BLE001 —— 附加产物，任何渲染问题都不应中断 Phase 1
        html = None
        print(f"提示：host_report.html 跳过渲染（{e}）", file=sys.stderr)
    if html is not None:
        with open(host_report_path, "w", encoding="utf-8") as f:
            f.write(html)

    print(f"探测完成：{len(target['hosts'])} 台主机（并行采集，mode={mode}）")
    for name, f in facts.items():
        ck = f["checklist"]
        n = {s: sum(1 for i in ck if i["status"] == s) for s in ("pass", "warn", "fail")}
        print(f"  {name} ({f['role']}): pass={n['pass']} warn={n['warn']} fail={n['fail']}")
    print(f"模型配置：{model_profile.get('model')} → weights.source={model_profile['weights']['source']}")
    arts = "env.json / env_report.md / model_profile.json / search_space.json"
    if os.path.exists(host_report_path):
        arts += " / host_report.html"
    print(f"产物：{arts}（{args.out_dir}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
