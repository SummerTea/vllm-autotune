#!/usr/bin/env python3
"""bench.py — 单次实验的度量 harness（在 head GPU 主机上运行）。

run.sh 把本文件 rsync 到远端 head 主机后执行一次：按 config.yaml 起 vllm serve，
轮询 /health 至就绪，跑 harness.bench.repeats 次 `vllm bench serve --goodput`，
取中位数，写 status.json / result.json / run.log，并用启动日志实测值就地校准
model_profile.json，最后清理 server 进程（不留僵尸）。

对仓库只读（绝不改 config.yaml）；只写 --out-dir。

冻结接口（run.sh / 编码 agent 依赖，勿改）：
  CLI: python bench.py --config config.yaml --harness harness.yaml --target target.yaml --out-dir .
       可选测试钩子：--serve-cmd / --bench-cmd（指向 stub 脚本）、--python（前置解释器）；
       轮询间隔 --poll-interval、显存采样间隔 --vram-interval 亦可调参（保证测试可跑）。
  产物:
    status.json   {"state":"running|done|crash","phase":"loading|benching|done|crash",
                   "started_at":ISO,"updated_at":ISO,"message":str}
    result.json   成功 {"status":"ok", goodput_req_s, output_tok_s, ttft_p99_ms, tpot_p99_ms,
                        peak_vram_gb, dur_s, repeats, env_fingerprint, calibration}
                   失败 {"status":"crash","reason":str, ...}
    run.log       vllm server 的 stdout/stderr（另附 bench 每次调用的标注与输出，便于排障）
    model_profile.json 就地校准 boot.measured_seconds / kv.measured_capacity_tokens / calibrated=true
  stdout 末尾固定输出 "---\nstatus: ...\n..." 块，供 agent grep。
"""

from __future__ import annotations

import argparse
import datetime
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
    sys.exit("缺少 pyyaml：请用 `uv run python bench.py` 或 `pip install pyyaml`")

# 默认值（与 harness.yaml / docs/vllm-facts.md 保持一致）
DEFAULT_SLO = {"ttft_p99_ms": 500, "tpot_p99_ms": 50}
DEFAULT_WORKLOAD = {"dataset": "random", "isl": 1024, "osl": 128, "num_prompts": 1000}
DEFAULT_REF_CONCURRENCY = 128
DEFAULT_SEED = 0
HEALTH_URL = "http://127.0.0.1:{port}/health"
KV_LINE_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")


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


# ── 命令构建（单元测试目标）─────────────────────────────────────────────

def build_serve_cmd(serve_cmd: str, config: dict, model: str, port: int,
                    python: str | None = None) -> list[str]:
    """vllm serve 命令：`vllm serve <model> <serve 参数…>`。
    serve 键 kebab-case 转 `--flag`；布尔 False → `--flag false`；缺 --port 则补默认。"""
    toks = shlex.split(serve_cmd) or ["vllm", "serve"]
    args = [model]
    for k, v in (config.get("serve") or {}).items():
        flag = "--" + str(k).replace("_", "-")
        if isinstance(v, bool):
            if v:
                args.append(flag)
            else:
                args += [flag, "false"]
        else:
            args += [flag, str(v)]
    if "--port" not in args:
        args += ["--port", str(port)]
    full = toks + args
    if python:
        full = [python] + full
    return full


def build_bench_cmd(bench_cmd: str, model: str, workload: dict, slo: dict,
                    ref_conc: int, seed: int, result_dir: str,
                    trust_remote_code: bool = False, python: str | None = None) -> list[str]:
    """vllm bench serve 命令（docs §1 CLI + --goodput + --save-result）。"""
    toks = shlex.split(bench_cmd) or ["vllm", "bench", "serve"]
    wl = {**DEFAULT_WORKLOAD, **workload}
    sl = {**DEFAULT_SLO, **slo}
    args = [
        "--model", model,
        "--dataset-name", str(wl.get("dataset", "random")),
        "--input-len", str(wl.get("isl", 1024)),
        "--output-len", str(wl.get("osl", 128)),
        "--num-prompts", str(wl.get("num_prompts", 1000)),
        "--request-rate", "inf",
        "--max-concurrency", str(ref_conc),
        "--goodput", f"ttft:{sl['ttft_p99_ms']}", f"tpot:{sl['tpot_p99_ms']}",
        "--seed", str(seed),
        "--save-result", "--result-dir", result_dir,
    ]
    if trust_remote_code:
        args.append("--trust-remote-code")
    full = toks + args
    if python:
        full = [python] + full
    return full


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


def _cleanup(proc: subprocess.Popen | None) -> None:
    """终止 server 进程组 + pkill 兜底 + 等显存释放；绝不让僵尸存活，绝不抛异常。"""
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
    _run_local(["pkill", "-f", "vllm serve"])  # 兜底（尽力而为）
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
    ap.add_argument("--serve-cmd", default="vllm serve", help="serve 命令（测试可指向 stub 脚本）")
    ap.add_argument("--bench-cmd", default="vllm bench serve", help="bench 命令（测试可指向 stub 脚本）")
    ap.add_argument("--python", default=None, help="可选：serve/bench 命令前追加的解释器")
    ap.add_argument("--poll-interval", type=float, default=2.0, help="/health 轮询间隔（秒）")
    ap.add_argument("--vram-interval", type=float, default=2.0, help="显存采样间隔（秒）")
    args = ap.parse_args(argv)

    config = load_yaml(args.config)
    harness = load_yaml(args.harness)
    target = load_yaml(args.target)

    bench_cfg = harness.get("bench") or {}
    remote = harness.get("remote") or {}
    model = config.get("model") or (target.get("model") or {}).get("id")
    if not model:
        _err("config.yaml 与 target.yaml 都未提供模型 id。")
    assert model is not None  # _err 已 exit
    port = int(remote.get("vllm_port", 8000))
    repeats = int(bench_cfg.get("repeats", 2))
    startup_timeout_s = float(bench_cfg.get("startup_timeout_s", 1200))
    trial_timeout_s = float(bench_cfg.get("trial_timeout_s", 2400))
    seed = int(bench_cfg.get("seed", DEFAULT_SEED))
    trust_remote_code = bool((target.get("model") or {}).get("trust_remote_code", False))

    # workload / slo：target 逐键覆盖 harness（program.md:11）
    workload = {**harness.get("workload", {}), **(target.get("workload") or {})}
    slo = {**harness.get("slo", {}), **((target.get("metrics") or {}).get("slo") or {})}
    ref_conc = int(workload.get("ref_concurrency") or bench_cfg.get("ref_concurrency", DEFAULT_REF_CONCURRENCY))

    out_dir = args.out_dir
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

    proc: subprocess.Popen | None = None
    log_fh = None
    kv_tokens: int | None = None
    load_seconds: float | None = None
    result: dict | None = None

    try:
        serve_tokens = build_serve_cmd(args.serve_cmd, config, model, port, args.python)
        env_all = {**os.environ, **{str(k): str(v) for k, v in (config.get("env") or {}).items()}}
        log_path = os.path.join(out_dir, "run.log")
        log_fh = open(log_path, "ab")
        log_fh.write(f"=== bench.py {_now_iso()} serve: {shlex.join(serve_tokens)}\n".encode())
        log_fh.flush()
        proc = subprocess.Popen(serve_tokens, stdout=log_fh, stderr=subprocess.STDOUT,
                                env=env_all, start_new_session=True)

        # 就绪轮询：/health 200；503(EngineDeadError)/进程退出/超时 → crash
        budget = max(0.0, deadline - time.monotonic())
        ready_deadline = time.monotonic() + min(startup_timeout_s, budget)
        ready = False
        while time.monotonic() < ready_deadline:
            if proc.poll() is not None:
                raise TrialCrash(f"serve 提前退出（rc={proc.poll()}）")
            code = _health_code(port)
            if code == 200:
                ready = True
                break
            if code == 503:
                raise TrialCrash("EngineDeadError（/health 返回 503）")
            kv = _scan_kv(log_path)
            if kv:
                kv_tokens = kv
            time.sleep(args.poll_interval)
        if not ready:
            if time.monotonic() >= deadline:
                raise TrialCrash("trial_timeout")
            raise TrialCrash("startup_timeout")
        load_seconds = round(time.monotonic() - start_mono, 1)
        kv = _scan_kv(log_path)
        if kv:
            kv_tokens = kv
        msg = f"就绪（load={load_seconds}s），开始 bench（repeats={repeats}）"
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
            bench_tokens = build_bench_cmd(args.bench_cmd, model, workload, slo, ref_conc, seed,
                                           os.path.join(out_dir, "bench_results"),
                                           trust_remote_code, args.python)
            results = []
            for i in range(repeats):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TrialCrash("trial_timeout")
                parsed = _run_bench_once(bench_tokens, remaining, log_fh, out_dir)
                if parsed["goodput"] is None:
                    raise TrialCrash(f"bench 第 {i + 1} 次无有效输出（goodput 缺失）")
                results.append(parsed)
            med = {k: (statistics.median([r[k] for r in results if r[k] is not None])
                      if any(r[k] is not None for r in results) else None)
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
            "peak_vram_gb": peak_gb,
            "dur_s": dur_s,
            "repeats": repeats,
            "env_fingerprint": fingerprint,
            "calibration": calibration,
            "started_at": started_at,
        }
    except TrialCrash as c:
        result = {"status": "crash", "reason": c.reason, "started_at": started_at,
                  "calibration": {"measured_kv_tokens": kv_tokens, "load_seconds": load_seconds}}
    except Exception as e:  # 兜底：任何异常都记为 crash，绝不让进程挂起
        result = {"status": "crash", "reason": f"{type(e).__name__}: {e}",
                  "started_at": started_at,
                  "calibration": {"measured_kv_tokens": kv_tokens, "load_seconds": load_seconds}}
    finally:
        _cleanup(proc)
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
