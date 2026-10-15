#!/usr/bin/env python3
"""bench.py — 单次实验的度量 harness（docker-only，在 head GPU 主机上运行）。

run.sh 把本文件 rsync 到远端 head 主机后执行一次：按 config.yaml 起 docker 容器跑
vllm serve，轮询 /health 至就绪，跑 harness.bench.repeats 次 `vllm bench serve
--goodput`（bench 客户端也在容器内），取中位数，写 status.json / result.json /
run.log，并用启动日志实测值就地校准 model_profile.json，最后 docker rm -f 清理。

对仓库只读（绝不改 config.yaml）；只写 --out-dir。

  冻结接口（run.sh / 编码 agent 依赖，勿改）：
    CLI: python bench.py --config config.yaml --harness harness.yaml --target target.yaml --out-dir .
        测试钩子：--docker-cmd（指向 stub docker 客户端）；
        轮询间隔 --poll-interval、显存采样间隔 --vram-interval 亦可调参（保证测试可跑）。
        --detect-gmu：可选预检——用 --load-format dummy 从高到低探测最大可用
                     gpu-memory-utilization，写 gmu_probe.json 后退出（不跑 trial）。
  产物:
    status.json   {"state":"running|done|crash","phase":"loading|benching|done|crash",
                   "started_at":ISO,"updated_at":ISO,"message":str}
    result.json   成功 {"status":"ok", goodput_req_s, output_tok_s, ttft_p99_ms, tpot_p99_ms,
                        stats:{goodput_std,...}（repeats 样本 σ）, peak_vram_gb, dur_s,
                        repeats, env_fingerprint, calibration,
                        contention?（检测到共租 GPU 进程时）}
                   失败 {"status":"crash","reason":str, ...}
    run.log       vllm serve 容器的 stdout/stderr（另附 bench 每次调用的标注与输出，便于排障）
    model_profile.json 就地校准 boot.measured_seconds / kv.measured_capacity_tokens / calibrated=true
    gmu_probe.json 仅 --detect-gmu：{"gmu_suggested":X|None,"tried":[..],"load_format":"dummy","reason":str}
  stdout 末尾固定输出 "---\nstatus: ...\n..." 块，供 agent grep。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import NoReturn

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("缺少 pyyaml：远端运行 bench.py 的 python 需已安装 pyyaml"
             "（零破坏默认不装；如用系统 python3 通常自带）；本机可用 `uv run python bench.py`")

# 默认值（与 harness.yaml / docs/vllm-facts.md 保持一致）
DEFAULT_SLO = {"ttft_p99_ms": 500, "tpot_p99_ms": 50}
DEFAULT_WORKLOAD = {"dataset": "random", "isl": 1024, "osl": 128, "num_prompts": 1000,
                    "prefix_hit_pct": 0}
DEFAULT_REF_CONCURRENCY = 128
DEFAULT_SEED = 0
HEALTH_URL = "http://127.0.0.1:{port}/health"
KV_LINE_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
# --detect-gmu 预检（借鉴 vLLM auto_tune.sh:252-272 + :95，见 docs/auto-tune-notes.md A/B）
DEFAULT_GMU_DETECT = {"start": 0.98, "step": 0.02, "floor": 0.10,
                      "ready_timeout_s": 120, "total_timeout_s": 900}


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def _err(msg: str) -> NoReturn:
    print(f"bench.py: 错误: {msg}", file=sys.stderr)
    sys.exit(2)


def load_yaml(path: str, required: bool = True) -> dict:
    if not os.path.exists(path):
        if required:
            _err(f"未找到 {path}。")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run_local(cmd: list[str], timeout: float = 10) -> tuple[int, str, str]:
    """本地执行（bench.py 运行在 head 主机上）；尽力而为，不抛异常。"""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, "", str(e)


# ── 解析器（单元测试目标）───────────────────────────────────────────────

def _field(text: str, *labels: str) -> float | None:
    """在 vllm bench stdout 表格中按规范化标签取数值（找不到 → None）。"""
    for line in text.splitlines():
        s = line.strip()
        if ":" not in s:
            continue
        label, _, val = s.partition(":")
        key = re.sub(r"[^a-z0-9]", "", label.lower())
        if any(w in key for w in labels):
            m = re.search(r"[-+]?\d+(?:\.\d+)?", val)
            return float(m.group(0)) if m else None
    return None


def parse_bench_stdout(text: str | None) -> dict:
    """从 stdout 表格解析（docs §2 字段名）；缺失字段 → None，绝不抛异常。"""
    if not text:
        return {"goodput": None, "output_tok_s": None, "ttft_p99_ms": None, "tpot_p99_ms": None}
    return {
        "goodput": _field(text, "requestgoodput"),
        "output_tok_s": _field(text, "outputtokenthroughput"),
        "ttft_p99_ms": _field(text, "p99ttft"),
        "tpot_p99_ms": _field(text, "p99tpot"),
    }


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def parse_bench_json(data: dict) -> dict:
    """从 --save-result JSON 解析（docs §2 键名 p99_<m>_ms 等）。"""
    return {
        "goodput": _num(data.get("request_goodput")),
        "output_tok_s": _num(data.get("output_throughput")),
        "ttft_p99_ms": _num(data.get("p99_ttft_ms")),
        "tpot_p99_ms": _num(data.get("p99_tpot_ms")),
    }


def parse_kv_cache_line(text: str | None) -> int | None:
    """从启动日志提取 "GPU KV cache size: <N> tokens"（docs §3 校准行）。"""
    if not text:
        return None
    m = KV_LINE_RE.search(text)
    return int(m.group(1).replace(",", "")) if m else None


def find_newest_json(d: str) -> str | None:
    """result-dir 中最新（mtime）的 JSON 文件。"""
    try:
        cands = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json")]
    except OSError:
        return None
    return max(cands, key=os.path.getmtime) if cands else None


def parse_compute_apps(text: str | None) -> list[dict]:
    """nvidia-smi --query-compute-apps=pid,process_name,used_memory →
    [{"pid":int, "process_name":str|None, "used_memory_mib":int|None}]。
    used_memory 形如 "512 MiB"（取首个整数）；[N/A]/空/坏行跳过（尽力而为）。"""
    if not text:
        return []
    apps = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0].isdigit():
            continue  # pid 列必须是数字；[N/A]/表头/坏行跳过
        m = re.search(r"\d+", parts[2])
        apps.append({
            "pid": int(parts[0]),
            "process_name": parts[1] if parts[1] not in ("", "[N/A]", "N/A") else None,
            "used_memory_mib": int(m.group(0)) if m else None,
        })
    return apps


def _repeats_stats(results: list[dict], key: str) -> float:
    """repeats 样本标准差：n≥2 用 statistics.stdev；n<2 → 0.0（退化）。"""
    vals = [r[key] for r in results if r[key] is not None]
    return round(statistics.stdev(vals), 4) if len(vals) >= 2 else 0.0


# ── 命令构建（单元测试目标）─────────────────────────────────────────────

def effective_runtime(harness: dict, config: dict) -> dict:
    """运行时 = harness.runtime 与 config.runtime 逐键合并（config 覆盖）。
    docker-only：若仍有 kind 键则接受并忽略（向后兼容）。"""
    return {**(harness.get("runtime") or {}), **((config.get("runtime") or {}))}


def container_model_path(runtime: dict) -> str:
    """容器内模型挂载点，默认 /model。"""
    return str(runtime.get("container_model_path") or "/model")


def docker_clean_stale(docker_cmd: str) -> None:
    """幂等清理同名残留容器（硬杀遗留 trial 会留下 vllm-autotune-trial），
    失败忽略——否则 docker run --name 会因同名冲突而失败。"""
    _run_local([docker_cmd, "rm", "-f", "vllm-autotune-trial"])


def _serve_flags(config: dict, port: int) -> list[str]:
    """serve 参数 token（不含模型位）：kebab flag + 布尔约定 + --port 兜底。
    布尔：true → 裸 flag；false → 省略（保持默认）；no-<flag>: true 原样透传。"""
    args = []
    for k, v in (config.get("serve") or {}).items():
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            if v:
                args.append(flag)
            # false → 不产出任何 token（省略，保持默认）
        else:
            args += [flag, str(v)]
    if "--port" not in args:
        args += ["--port", str(port)]
    return args


def _known_serve_flags(help_text: str) -> set[str]:
    """从 `vllm serve --help` 提取全部选项名。每个空白 token 先规范化：
    剥首部 `[`/`(` 与尾部 `]`/`)`/`,`（覆盖 usage 行 `[--max-num-seqs N]` 等），再取 `--` 开头的。
    自动覆盖 BooleanOptionalAction 的 `--enable-x, --no-enable-x` 对。"""
    flags = set()
    for tok in help_text.split():
        norm = tok.lstrip("[(").rstrip("]),")
        if norm.startswith("--") and len(norm) > 2:
            flags.add(norm)
    return flags


def _unknown_serve_flags(serve: dict, known: set[str]) -> list[str]:
    """config.serve 键 → `--<kebab>`（下划线→连字符）后不在 help 选项集里的列表（排序稳定）。
    no-<flag> 键同规则：`--no-enable-x` 有效当且仅当它在 known 里。"""
    return sorted(
        "--" + str(k).replace("_", "-")
        for k in serve
        if "--" + str(k).replace("_", "-") not in known
    )


def _get_serve_help(docker_cmd: str, runtime: dict, cache_dir: str) -> str | None:
    """镜像内 `vllm serve --help` 文本：docker run --pull=never --entrypoint vllm <image> serve --help
    （stdout+stderr 都抓；部分 CLI 打到 stderr）。按 image 缓存到 <cache_dir>/.serve_help_<hash>.txt，
    命中即复用不重跑。失败/空 → None（调用方跳过校验，不阻塞 trial）。"""
    image = str(runtime.get("image"))
    cache_path = os.path.join(cache_dir, f".serve_help_{hashlib.sha256(image.encode()).hexdigest()[:12]}.txt")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            if text.strip():
                return text
        except OSError:
            pass
    rc, out, err = _run_local(
        [docker_cmd, "run", "--rm", "--pull=never", "--entrypoint", "vllm", image, "serve", "--help"],
        timeout=60)
    text = (out or "") + (err or "")
    if rc == 0 and text.strip():
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass
        return text
    return None


def build_docker_run_cmd(docker_cmd: str, runtime: dict, model_path: str, port: int,
                         config: dict, name: str = "vllm-autotune-trial") -> list[str]:
    """docker 运行时 serve 命令（frozen 设计）：
    docker run --pull=never --rm --name <name> --gpus all --ipc=host --shm-size=<shm|16g>
      -p 127.0.0.1:<port>:<port> -v <model_path>:<container_path>:ro [-e K=V ...]
      [extra_docker_args...] --entrypoint vllm <image> serve <container_path> <serve flags…>
    name：容器名，默认 trial 名（调用方逐字节一致）；report.py 部署产物传 vllm-deploy-best，
    避免被后续调优轮次的 docker rm -f vllm-autotune-trial 误杀。"""
    image = runtime.get("image")
    if not image:
        raise ValueError("docker 运行时缺少 runtime.image（写进 config.yaml 或 harness.yaml）")
    cpath = container_model_path(runtime)
    cmd = [docker_cmd, "run", "--pull=never", "--rm", "--name", name,
           "--gpus", "all", "--ipc", "host", "--shm-size", str(runtime.get("shm_size") or "16g"),
           # --ipc host 共享宿主机 /dev/shm，--shm-size 实际不生效（保留两者，仅说明）
           "-p", f"127.0.0.1:{port}:{port}",
           "-v", f"{model_path}:{cpath}:ro"]
    for k, v in (config.get("env") or {}).items():
        cmd += ["-e", f"{k}={v}"]
    for a in (runtime.get("extra_docker_args") or []):
        cmd.append(str(a))
    cmd += ["--entrypoint", "vllm", image, "serve", cpath]
    return cmd + _serve_flags(config, port)


def _bench_args(model: str, workload: dict, slo: dict, ref_conc: int, seed: int,
                result_dir: str, trust_remote_code: bool, port: int) -> list[str]:
    """bench serve 参数（docs §1 CLI + --goodput + --save-result + 显式 --base-url
    + --percentile-metrics）。
    prefix_hit_pct（仅 random 数据集）：把 isl 拆成共享前缀 + 余量（prefix_len=isl×pct/100，
    adjusted=isl−prefix_len），让 enable-prefix-caching 成为可测维度；非 random 数据集忽略该维度。"""
    wl = {**DEFAULT_WORKLOAD, **workload}
    sl = {**DEFAULT_SLO, **slo}
    dataset = wl.get("dataset", "random")
    isl = int(wl.get("isl", 1024))
    osl = int(wl.get("osl", 128))
    prefix_pct = max(0, min(100, int(wl.get("prefix_hit_pct", 0) or 0)))  # 钳制 0-100，防 >100 产生负 --random-input-len
    if dataset == "random" and prefix_pct > 0:
        # 共享前缀 + 余量：总 prompt ≈ isl；不带 --input-len（避免与 --random-input-len 冲突）
        prefix_len = isl * prefix_pct // 100
        adjusted = isl - prefix_len
        len_args = ["--random-input-len", str(adjusted), "--random-prefix-len", str(prefix_len)]
    else:
        # prefix_hit_pct=0 或非 random 数据集（无共享前缀语义，忽略该维度）→ 维持原样
        len_args = ["--input-len", str(isl)]
    args = [
        "--model", model,
        "--base-url", f"http://127.0.0.1:{port}",   # 显式指向 serve 端口（≠8000 时也正确）
        "--dataset-name", str(dataset),
        *len_args,
        "--output-len", str(osl),
        "--num-prompts", str(wl.get("num_prompts", 1000)),
        "--request-rate", "inf",
        "--max-concurrency", str(ref_conc),
        "--goodput", f"ttft:{sl['ttft_p99_ms']}", f"tpot:{sl['tpot_p99_ms']}",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--seed", str(seed),
        "--save-result", "--result-dir", result_dir,
    ]
    if trust_remote_code:
        args.append("--trust-remote-code")
    return args


def build_docker_bench_cmd(docker_cmd: str, runtime: dict, model: str, workload: dict,
                           slo: dict, ref_conc: int, seed: int, host_result_dir: str,
                           trust_remote_code: bool = False, port: int = 8000,
                           container_result_dir: str = "/bench_results",
                           host_model_path: str | None = None) -> list[str]:
    """docker 模式 bench 客户端也在容器内跑（本机无需 vllm CLI）：
    docker run --pull=never --rm --network host -v <host_result_dir>:<container_result_dir>
      [-v <host_model_path>:<cpath>:ro] --entrypoint vllm <image> bench serve <参数…>
    --network host 让容器共享主机网络栈，可直达 serve 发布的 127.0.0.1:<port>；
    --result-dir 指向容器内挂载点（--save-result JSON 经 -v 落回宿主目录）。
    host_model_path：random 数据集需要 tokenizer，故必须把模型（只读）挂进 bench 容器，
    才能让 --model <容器模型路径> 在容器内可解析。"""
    image = runtime.get("image")
    if not image:
        raise ValueError("docker 运行时缺少 runtime.image")
    base = [docker_cmd, "run", "--pull=never", "--rm", "--network", "host",
            "-v", f"{host_result_dir}:{container_result_dir}"]
    if host_model_path:
        base += ["-v", f"{host_model_path}:{container_model_path(runtime)}:ro"]
    base += ["--entrypoint", "vllm", image, "bench", "serve"]
    return base + _bench_args(model, workload, slo, ref_conc, seed, container_result_dir,
                              trust_remote_code, port)


def _start_serve_and_wait(docker_cmd: str, runtime: dict, model_path: str, port: int,
                          config: dict, log_fh, log_path: str, deadline: float,
                          startup_timeout_s: float, poll_interval: float,
                          kv_sink: list[int | None], procs: list[subprocess.Popen]) -> None:
    """docker 起 serve（attach → log_fh）并轮询 /health 至就绪；失败 raise TrialCrash。
    普通 trial 与 --detect-gmu 探测共用。KV 校准行写入 kv_sink[0]；serve 进程追加到
    procs（由调用方负责清理——探测模式每级尝试后须清理）。"""
    docker_clean_stale(docker_cmd)  # 先清残留同名容器（幂等），再 run
    serve_tokens = build_docker_run_cmd(docker_cmd, runtime, model_path, port, config)
    log_fh.write(f"=== bench.py {_now_iso()} serve(docker): {shlex.join(serve_tokens)}\n".encode())
    log_fh.flush()
    # docker run 前台/attach 模式：容器 stdout 即 serve 日志，直接落入 run.log（含 KV 校准行）
    procs.append(subprocess.Popen(serve_tokens, stdout=log_fh, stderr=subprocess.STDOUT,
                                  env=os.environ, start_new_session=True))
    # 就绪轮询：/health 200；503(EngineDeadError)/进程退出/超时 → crash
    ready_deadline = time.monotonic() + min(startup_timeout_s, max(0.0, deadline - time.monotonic()))
    ready = False
    while time.monotonic() < ready_deadline:
        if procs[0].poll() is not None:
            raise TrialCrash(f"serve 提前退出（rc={procs[0].poll()}）")
        code = _health_code(port)
        if code == 200:
            ready = True
            break
        if code == 503:
            raise TrialCrash("EngineDeadError（/health 返回 503）")
        kv = _scan_kv(log_path)
        if kv:
            kv_sink[0] = kv
        time.sleep(poll_interval)
    if not ready:
        if time.monotonic() >= deadline:
            raise TrialCrash("trial_timeout")
        raise TrialCrash("startup_timeout")
    kv = _scan_kv(log_path)  # 日志可能晚刷新，就绪后再扫一次
    if kv:
        kv_sink[0] = kv


# ── --detect-gmu 预检（最大可用 gpu-memory-utilization，--load-format dummy）────────

def gmu_candidates(bench_cfg: dict) -> list[float]:
    """候选 gpu-memory-utilization：从 start 按 step 递减到 >= floor。"""
    d = {**DEFAULT_GMU_DETECT, **((bench_cfg.get("gmu_detect") or {}))}
    start, step, floor = float(d["start"]), float(d["step"]), float(d["floor"])
    cands = []
    v = start
    while v >= floor - 1e-9:
        cands.append(round(v, 4))
        v -= step
    return cands


def _gmu_probe_config(config: dict, gmu: float) -> dict:
    """探测用 serve 配置：强制 --load-format dummy + --gpu-memory-utilization <gmu>
    （覆盖 config 中已有的 gmu/load-format 写法，含下划线变体）。"""
    serve = dict(config.get("serve") or {})
    for k in [k for k in serve if k.replace("_", "-") in ("gpu-memory-utilization", "load-format")]:
        del serve[k]
    serve["gpu-memory-utilization"] = gmu
    serve["load-format"] = "dummy"
    return {**config, "serve": serve}


def _set_profile_gmu_suggested(out_dir: str, gmu: float | None) -> None:
    """out_dir 已有 model_profile.json 时，就地补 gmu_suggested（保留其余字段）。"""
    prof_path = os.path.join(out_dir, "model_profile.json")
    if not os.path.exists(prof_path):
        return
    try:
        with open(prof_path, encoding="utf-8") as f:
            profile = json.load(f)
    except (OSError, ValueError):
        return
    profile["gmu_suggested"] = gmu
    with open(prof_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def _run_gmu_probe(docker_cmd: str, runtime: dict, config: dict, target: dict, port: int,
                   bench_cfg: dict, out_dir: str, poll_interval: float) -> int:
    """--detect-gmu：从高到低起服（--load-format dummy）探测最大可用 gpu-memory-utilization。
    产出 gmu_probe.json 与 status.json（running→done/crash，run.sh --poll 可直接轮询）；
    out_dir 已有 model_profile.json 则就地补 gmu_suggested。
    返回 0=找到建议值 / 1=失败（floor 耗尽 / total_timeout / 预检不通过）。"""
    os.makedirs(out_dir, exist_ok=True)
    started_at = _now_iso()
    host_model_path = (target.get("model") or {}).get("path")
    if not host_model_path:
        msg = "需要 target.yaml 的 model.path（宿主上的模型目录，将只读挂载进容器）"
        print(f"bench.py: 错误: {msg}", file=sys.stderr)
        _write_status(out_dir, "crash", "crash", f"gmu 探测失败：{msg}", started_at)
        return 1
    img = str(runtime.get("image"))  # main 已保证 image 非空
    if _run_local([docker_cmd, "image", "inspect", img])[0] != 0:
        msg = (f"镜像 {img} 未在主机预先准备（本模板不自动拉取，保持主机零新增；"
               "请人工预先 docker pull 或改用已存在的镜像）")
        print(f"bench.py: 错误: {msg}", file=sys.stderr)
        _write_status(out_dir, "crash", "crash", f"gmu 探测失败：{msg}", started_at)
        return 1
    detect = {**DEFAULT_GMU_DETECT, **((bench_cfg.get("gmu_detect") or {}))}
    ready_timeout_s = float(detect["ready_timeout_s"])
    total_timeout_s = float(detect["total_timeout_s"])
    cands = gmu_candidates(bench_cfg)
    _write_status(out_dir, "running", "probing",
                  f"gmu 探测（--load-format dummy；{len(cands)} 级候选，预算 {total_timeout_s:.0f}s）",
                  started_at)
    log_path = os.path.join(out_dir, "run.log")
    tried: list[float] = []
    suggested: float | None = None
    reason = ""
    timed_out = False
    total_deadline = time.monotonic() + total_timeout_s
    with open(log_path, "ab") as log_fh:
        # m1：探测路径也预校验 config.serve 的 flag 名（与 trial 同口径）——typo 不伪装成"无可用 gmu"
        help_text = _get_serve_help(docker_cmd, runtime, out_dir)
        if help_text is None:
            log_fh.write(f"=== bench.py {_now_iso()} serve --help 不可用，跳过 flag 预校验\n".encode())
            log_fh.flush()
        else:
            unknown = _unknown_serve_flags(config.get("serve") or {}, _known_serve_flags(help_text))
            if unknown:
                reason = (f"未知 vllm serve 参数：{unknown}"
                          "（对照镜像 `vllm serve --help`；请修正 config.yaml.serve）")
                log_fh.write(f"=== bench.py {_now_iso()} flag 预校验失败: {reason}\n".encode())
                log_fh.flush()
                msg = f"gmu 探测失败：{reason}"
                _write_status(out_dir, "crash", "crash", msg, started_at)
                print(f"bench.py: 错误: {reason}", file=sys.stderr)
                return 1
        for gmu in cands:
            if time.monotonic() >= total_deadline:
                timed_out = True
                reason = f"total_timeout（预算 {total_timeout_s:.0f}s 耗尽，尚未有候选成功）"
                log_fh.write(f"=== bench.py {_now_iso()} total_timeout 中止探测\n".encode())
                log_fh.flush()
                break
            tried.append(gmu)
            procs: list[subprocess.Popen] = []
            try:
                _start_serve_and_wait(docker_cmd, runtime, host_model_path, port,
                                      _gmu_probe_config(config, gmu), log_fh, log_path,
                                      float("inf"), ready_timeout_s, poll_interval,
                                      [None], procs)
                suggested = gmu
                reason = f"serve 在 gmu={gmu} 就绪（--load-format dummy）"
                break
            except TrialCrash as c:
                reason = c.reason
                log_fh.write(f"=== bench.py {_now_iso()} gmu={gmu} 失败: {c.reason}\n".encode())
                log_fh.flush()
            finally:
                _cleanup(procs, docker_cmd)  # 每级尝试后必清理（rm -f 幂等）
    with open(os.path.join(out_dir, "gmu_probe.json"), "w", encoding="utf-8") as f:
        json.dump({"gmu_suggested": suggested, "tried": tried, "load_format": "dummy",
                   "reason": reason}, f, ensure_ascii=False, indent=2)
    _set_profile_gmu_suggested(out_dir, suggested)
    if suggested is not None:
        _write_status(out_dir, "done", "done", f"gmu_suggested={suggested}", started_at)
        print(f"gmu_suggested: {suggested}")
        return 0
    _write_status(out_dir, "crash", "crash", f"gmu 探测失败：{reason}", started_at)
    if timed_out:
        print(f"gmu_suggested: 无（{reason}；已尝试 {len(tried)} 级，未到 floor）")
    else:
        print(f"gmu_suggested: 无（floor={detect['floor']} 以下仍无法起服；已尝试 {len(tried)} 级）")
    return 1


# ── 环境指纹与校准 ──────────────────────────────────────────────────────

def _capture_fingerprint() -> dict:
    """启动前采集驱动/时钟/governor（尽力而为；缺失字段直接省略）。"""
    fp: dict = {}
    rc, out, _ = _run_local(
        ["nvidia-smi", "--query-gpu=driver_version,clocks.sm", "--format=csv,noheader"])
    if rc == 0 and out.strip():
        parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
        if len(parts) >= 2:
            fp["driver_version"] = parts[0] or None
            fp["clocks_sm"] = parts[1] or None
    rc, out, _ = _run_local(["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"])
    if rc == 0 and out.strip():
        fp["governor"] = out.strip().splitlines()[0].strip()
    return fp


def _fingerprint_diff(fp: dict, env: dict | None) -> tuple[dict | None, str | None]:
    """与 env.json 对比指纹；返回 (diff|None, note|None)。"""
    if not env or not env.get("hosts") or "head" not in env["hosts"]:
        return None, "out-dir 无 env.json（未跑 probe.py），跳过对比"
    head = env["hosts"]["head"]
    diff: dict = {}
    ref = head.get("cpu", {}).get("governor")
    if ref and fp.get("governor") and fp["governor"] != ref:
        diff["governor"] = f"{ref} → {fp['governor']}"
    gpus = head.get("gpu", {}).get("gpus") or []
    if gpus:
        ref = gpus[0].get("driver_version")
        if ref and fp.get("driver_version") and fp["driver_version"] != ref:
            diff["driver_version"] = f"{ref} → {fp['driver_version']}"
        ref = gpus[0].get("clocks_sm")
        if ref is not None and fp.get("clocks_sm") and str(ref) != fp["clocks_sm"]:
            diff["clocks_sm"] = f"{ref} → {fp['clocks_sm']}"
    return (diff or None), None


def _calibrate_profile(out_dir: str, kv_tokens: int | None, load_seconds: float | None) -> dict:
    """就地校准 model_profile.json（保留 estimated_*，写 measured_*，置 calibrated）。"""
    p = os.path.join(out_dir, "model_profile.json")
    if not os.path.exists(p):
        return {"measured_kv_tokens": kv_tokens, "load_seconds": load_seconds,
                "profile_updated": False}
    with open(p, encoding="utf-8") as f:
        profile = json.load(f)
    profile["boot"] = {**(profile.get("boot") or {}), "measured_seconds": load_seconds}
    profile["kv"] = {**(profile.get("kv") or {}), "measured_capacity_tokens": kv_tokens}
    profile["calibrated"] = True
    with open(p, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
    return {"measured_kv_tokens": kv_tokens, "load_seconds": load_seconds,
            "profile_updated": True}


# ── 状态与健康 ──────────────────────────────────────────────────────────

def _write_status(out_dir: str, state: str, phase: str, message: str, started_at: str) -> None:
    data = {"state": state, "phase": phase, "started_at": started_at,
            "updated_at": _now_iso(), "message": message}
    with open(os.path.join(out_dir, "status.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _health_code(port: int, timeout: float = 2.0) -> int | None:
    """GET /health；200 就绪；503 = EngineDeadError；连接失败/异常 → None。"""
    try:
        with urllib.request.urlopen(HEALTH_URL.format(port=port), timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _reset_prefix_cache(port: int, log_fh) -> None:
    """尽力 POST /reset_prefix_cache（repeats≥2 前清缓存，保证每次测量是冷缓存独立）。
    非 200 / 异常一律容忍：只写 run.log，绝不因重置失败中止 trial。"""
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/reset_prefix_cache", method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            status = r.status
    except (urllib.error.URLError, OSError, ValueError) as e:
        log_fh.write(f"=== bench.py {_now_iso()} reset_prefix_cache 失败（容忍）: {e}\n".encode())
        log_fh.flush()
        return
    if status != 200:
        log_fh.write(f"=== bench.py {_now_iso()} reset_prefix_cache 返回 {status}（容忍，非 200）\n".encode())
        log_fh.flush()


def _scan_kv(log_path: str) -> int | None:
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            return parse_kv_cache_line(f.read())
    except OSError:
        return None


# ── 显存采样与清理 ──────────────────────────────────────────────────────

def _vram_sampler(stop: threading.Event, interval: float, peak_mib: list[int]) -> None:
    """后台采样 nvidia-smi 显存峰值（MiB）；连失败 3 次（如无 GPU）→ 退出。"""
    fails = 0
    while not stop.wait(interval):
        rc, out, _ = _run_local(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        if rc != 0 or not out.strip():
            fails += 1
            if fails >= 3:
                return
            continue
        fails = 0
        vals = [int(x) for x in re.findall(r"\d+", out)]
        if vals:
            peak_mib.append(max(vals))


def _detect_co_tenants() -> list[dict]:
    """trial 开始前探测共租：nvidia-smi compute-apps 有输出 → 非独占主机。
    无 nvidia-smi（如开发机 Mac）/ 空输出 → []（优雅跳过）。"""
    rc, out, _ = _run_local(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"])
    if rc != 0 or not out.strip():
        return []
    return parse_compute_apps(out)


def _wait_mem_freed(timeout: float = 30) -> None:
    """等待 nvidia-smi 显存释放；无 nvidia-smi → 立即返回（尽力而为）。"""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        rc, out, _ = _run_local(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        if rc != 0 or not out.strip():
            return
        vals = [int(x) for x in re.findall(r"\d+", out)]
        if not vals or max(vals) < 1024:  # 全部 < 1 GiB 视为已释放
            return
        time.sleep(2)


def _cleanup(procs: list[subprocess.Popen] | None, docker_cmd: str | None = None) -> None:
    """终止 docker 客户端进程组 + docker rm -f（docker 模式）+ 等显存释放。
    绝不让僵尸存活，绝不抛异常。"""
    for proc in procs or []:
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
    if docker_cmd:
        _run_local([docker_cmd, "rm", "-f", "vllm-autotune-trial"])  # 幂等；容器随 run 一起终止
    _wait_mem_freed(timeout=30)


class TrialCrash(Exception):
    """trial 失败（启动/超时/bench 出错），reason 写入 result.json。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason


# ── bench 单次运行 ──────────────────────────────────────────────────────

def _run_bench_once(bench_tokens: list[str], timeout: float, log_fh, out_dir: str) -> dict:
    """跑一次 bench：优先解析 --save-result JSON，回退 stdout 表格；失败抛 TrialCrash。"""
    result_dir = os.path.join(out_dir, "bench_results")
    os.makedirs(result_dir, exist_ok=True)
    cmd = bench_tokens
    marker = f"\n=== bench run {_now_iso()} cmd: {shlex.join(cmd)}\n"
    log_fh.write(marker.encode())
    log_fh.flush()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise TrialCrash("trial_timeout")
    log_fh.write((p.stdout or "").encode("utf-8", "replace"))
    if p.stderr:
        log_fh.write((f"\n--- bench stderr ---\n{p.stderr}").encode("utf-8", "replace"))
    log_fh.flush()
    if p.returncode != 0:
        raise TrialCrash(f"bench 失败（rc={p.returncode}）")
    parsed = None
    jpath = find_newest_json(result_dir)
    if jpath:
        try:
            with open(jpath, encoding="utf-8") as f:
                parsed = parse_bench_json(json.load(f))
        except (OSError, ValueError):
            parsed = None
    if not parsed or parsed["goodput"] is None:
        parsed = parse_bench_stdout(p.stdout)
    return parsed


# ── CLI 入口 ────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    """summary 数字格式：去掉多余小数尾零；None → 'null'。"""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _print_summary(result: dict) -> None:
    """末尾固定输出块（agent grep 用）；crash 只输出 status。"""
    print("---")
    print(f"status: {result['status']}")
    if result["status"] == "ok":
        print(f"goodput: {_fmt(result['goodput_req_s'])}")
        print(f"goodput_std: {_fmt((result.get('stats') or {}).get('goodput_std'))}")
        print(f"output_tok_s: {_fmt(result['output_tok_s'])}")
        print(f"ttft_p99_ms: {_fmt(result['ttft_p99_ms'])}")
        print(f"tpot_p99_ms: {_fmt(result['tpot_p99_ms'])}")
        print(f"peak_vram_gb: {_fmt(result['peak_vram_gb'])}")
        print(f"dur_s: {_fmt(result['dur_s'])}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="单次实验度量 harness（在 head GPU 主机运行，只读仓库）")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--harness", default="harness.yaml")
    ap.add_argument("--target", default="target.yaml")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--docker-cmd", default="docker", help="docker 客户端命令（测试可指向 stub）")
    ap.add_argument("--detect-gmu", action="store_true",
                    help="预检模式：--load-format dummy 从高到低探测最大可用 "
                         "gpu-memory-utilization，写 gmu_probe.json 后退出（不跑 trial）")
    ap.add_argument("--poll-interval", type=float, default=2.0, help="/health 轮询间隔（秒）")
    ap.add_argument("--vram-interval", type=float, default=2.0, help="显存采样间隔（秒）")
    args = ap.parse_args(argv)

    config = load_yaml(args.config)
    harness = load_yaml(args.harness)
    target = load_yaml(args.target)

    # docker-only：运行时 = harness.runtime 与 config.runtime 逐键合并（config 覆盖）
    runtime = effective_runtime(harness, config)
    if not runtime.get("image"):
        _err("需要 docker 镜像：在 config.yaml 或 harness.yaml 的 runtime.image 指定"
             "（如 vllm/vllm-openai:v0.31.0）")

    bench_cfg = harness.get("bench") or {}
    remote = harness.get("remote") or {}
    port = int(remote.get("vllm_port", 8000))
    out_dir = args.out_dir

    # 预检模式：探测最大可用 gpu-memory-utilization（--load-format dummy）后退出，不跑 trial
    if args.detect_gmu:
        return _run_gmu_probe(args.docker_cmd, runtime, config, target, port,
                              bench_cfg, out_dir, args.poll_interval)

    model = config.get("model") or (target.get("model") or {}).get("id")
    if not model:
        _err("config.yaml 与 target.yaml 都未提供模型 id。")
    assert model is not None  # _err 已 exit
    repeats = int(bench_cfg.get("repeats", 3))
    startup_timeout_s = float(bench_cfg.get("startup_timeout_s", 1200))
    trial_timeout_s = float(bench_cfg.get("trial_timeout_s", 2400))
    seed = int(bench_cfg.get("seed", DEFAULT_SEED))
    trust_remote_code = bool((target.get("model") or {}).get("trust_remote_code", False))

    # workload / slo：target 逐键覆盖 harness（program.md:11）
    workload = {**harness.get("workload", {}), **(target.get("workload") or {})}
    slo = {**harness.get("slo", {}), **((target.get("metrics") or {}).get("slo") or {})}
    ref_conc = int(workload.get("ref_concurrency") or bench_cfg.get("ref_concurrency", DEFAULT_REF_CONCURRENCY))

    os.makedirs(out_dir, exist_ok=True)
    started_at = _now_iso()
    start_mono = time.monotonic()
    deadline = start_mono + trial_timeout_s
    _write_status(out_dir, "running", "loading", "启动 vllm serve（等待 /health 就绪）", started_at)

    fp = _capture_fingerprint()
    env_path = os.path.join(out_dir, "env.json")
    env = None
    if os.path.exists(env_path):
        try:
            with open(env_path, encoding="utf-8") as f:
                env = json.load(f)
        except (OSError, ValueError):
            env = None
    diff, note = _fingerprint_diff(fp, env)
    fingerprint = {"captured": fp, "diff": diff, **({"note": note} if note else {})}

    procs: list[subprocess.Popen] = []
    log_fh = None
    kv_tokens: int | None = None
    load_seconds: float | None = None
    result: dict | None = None
    # 启动 helper 在轮询中把 KV 校准行写入 kv_sink[0]；serve 死在启动期也要读回（crash 路径保留）
    kv_sink: list[int | None] = [None]

    try:
        log_path = os.path.join(out_dir, "run.log")
        log_fh = open(log_path, "ab")
        # docker：serve/bench 都在容器内跑；本机只跑 docker 客户端
        host_model_path = (target.get("model") or {}).get("path")
        if not host_model_path:
            raise TrialCrash("需要 target.yaml 的 model.path（宿主上的模型目录，将只读挂载进容器）")
        # M1：零 pull 契约——先确认镜像已预先准备（docker run 另带 --pull=never 兜底，给出可诊断信息）
        img = str(runtime.get("image"))  # 前面 _err 已保证 image 非空
        if _run_local([args.docker_cmd, "image", "inspect", img])[0] != 0:
            raise TrialCrash(f"镜像 {img} 未在主机预先准备（本模板不自动拉取，保持主机零新增；"
                             "请人工预先 docker pull 或改用已存在的镜像）")
        # 先清残留同名容器（幂等）：硬杀遗留的同名容器不算外来共租，须在共租检测前清理
        docker_clean_stale(args.docker_cmd)
        # serve flag 预校验：对照镜像 `vllm serve --help`（按 image 缓存），未知名直接 fail-fast
        help_text = _get_serve_help(args.docker_cmd, runtime, out_dir)
        if help_text is None:
            log_fh.write(f"=== bench.py {_now_iso()} serve --help 不可用，跳过 flag 预校验\n".encode())
            log_fh.flush()
        else:
            unknown = _unknown_serve_flags(config.get("serve") or {}, _known_serve_flags(help_text))
            if unknown:
                raise TrialCrash(f"未知 vllm serve 参数：{unknown}"
                                 "（对照镜像 `vllm serve --help`；请修正 config.yaml.serve）")
        # 共租检测（起 serve 之前）：非独占主机上其他 GPU 进程会污染测量
        foreign_apps = _detect_co_tenants()
        contention = None
        if foreign_apps:
            contention = {"foreign_gpu_apps": foreign_apps, "checked_at": _now_iso()}
            if bool(bench_cfg.get("require_idle", False)):
                raise TrialCrash("主机非空闲：检测到其他 GPU 进程 "
                                 f"{[a['pid'] for a in foreign_apps]}（require_idle=true 要求独占）")
        # 起 serve（attach → run.log）并轮询 /health 就绪；与 --detect-gmu 共用同一 helper
        _start_serve_and_wait(args.docker_cmd, runtime, host_model_path, port, config,
                              log_fh, log_path, deadline, startup_timeout_s,
                              args.poll_interval, kv_sink, procs)
        model = container_model_path(runtime)  # docker 下 serve/bench 的模型位都用容器挂载点
        kv_tokens = kv_sink[0]
        load_seconds = round(time.monotonic() - start_mono, 1)
        msg = f"就绪（load={load_seconds}s，runtime=docker），开始 bench（repeats={repeats}）"
        if contention:
            msg += (f"；⚠ 检测到其他 GPU 进程（共租，测量可能受扰）："
                    f"{[a['pid'] for a in contention['foreign_gpu_apps']]}")
        if diff:
            msg += f"；环境指纹变化：{diff}"
        _write_status(out_dir, "running", "benching", msg, started_at)

        # 显存峰值采样线程（整个 bench 期间）
        peak_mib: list[int] = []
        stop = threading.Event()
        vram_thread = threading.Thread(target=_vram_sampler,
                                       args=(stop, args.vram_interval, peak_mib), daemon=True)
        vram_thread.start()
        try:
            bench_tokens = build_docker_bench_cmd(args.docker_cmd, runtime, model, workload, slo,
                                                  ref_conc, seed, os.path.join(out_dir, "bench_results"),
                                                  trust_remote_code, port,
                                                  host_model_path=host_model_path)
            results = []
            for i in range(repeats):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TrialCrash("trial_timeout")
                if i >= 1:
                    # D：同实例的后续 repeat 前清前缀缓存 → 每次测量冷缓存独立（首 repeat 起服即冷）
                    _reset_prefix_cache(port, log_fh)
                parsed = _run_bench_once(bench_tokens, remaining, log_fh, out_dir)
                if parsed["goodput"] is None:
                    raise TrialCrash(f"bench 第 {i + 1} 次无有效输出（goodput 缺失）")
                results.append(parsed)
            med = {k: (statistics.median([r[k] for r in results if r[k] is not None])
                      if any(r[k] is not None for r in results) else None)
                   for k in ("goodput", "output_tok_s", "ttft_p99_ms", "tpot_p99_ms")}
            # σ：repeats 样本标准差（n<2 → 0.0），keep 判定用（noise_k_sigma×σ 显著性口径）
            stats = {k: _repeats_stats(results, k)
                     for k in ("goodput", "output_tok_s", "ttft_p99_ms", "tpot_p99_ms")}
        finally:
            stop.set()
            vram_thread.join(timeout=5)

        kv = _scan_kv(log_path)  # 日志可能晚刷新，最后再扫一次
        if kv:
            kv_tokens = kv
        peak_gb = round(max(peak_mib) / 1024, 1) if peak_mib else None
        dur_s = round(time.monotonic() - start_mono, 1)
        calibration = _calibrate_profile(out_dir, kv_tokens, load_seconds)
        result = {
            "status": "ok",
            "goodput_req_s": med["goodput"],
            "output_tok_s": med["output_tok_s"],
            "ttft_p99_ms": med["ttft_p99_ms"],
            "tpot_p99_ms": med["tpot_p99_ms"],
            "stats": {
                "goodput_std": stats["goodput"],
                "output_tok_s_std": stats["output_tok_s"],
                "ttft_p99_std": stats["ttft_p99_ms"],
                "tpot_p99_std": stats["tpot_p99_ms"],
            },
            "peak_vram_gb": peak_gb,
            "dur_s": dur_s,
            "repeats": repeats,
            "env_fingerprint": fingerprint,
            "calibration": calibration,
            "started_at": started_at,
            **({"contention": contention} if contention else {}),
        }
    except TrialCrash as c:
        kv_tokens = kv_sink[0]  # serve 死在启动期也可能已扫到 KV 行，保留进 calibration
        result = {"status": "crash", "reason": c.reason, "started_at": started_at,
                  "calibration": {"measured_kv_tokens": kv_tokens, "load_seconds": load_seconds}}
    except Exception as e:  # 兜底：任何异常都记为 crash，绝不让进程挂起
        kv_tokens = kv_sink[0]
        result = {"status": "crash", "reason": f"{type(e).__name__}: {e}",
                  "started_at": started_at,
                  "calibration": {"measured_kv_tokens": kv_tokens, "load_seconds": load_seconds}}
    finally:
        _cleanup(procs, args.docker_cmd)
        if log_fh is not None:
            try:
                log_fh.close()
            except OSError:
                pass

    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    if result["status"] == "ok":
        _write_status(out_dir, "done", "done", f"完成 dur_s={result['dur_s']}", started_at)
    else:
        _write_status(out_dir, "crash", "crash", f"崩溃：{result['reason']}", started_at)
    _print_summary(result)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
