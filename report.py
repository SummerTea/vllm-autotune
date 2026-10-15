#!/usr/bin/env python3
"""report.py — Phase 4 收尾：results.tsv → report.md（只读 harness，唯一写出的文件是报告）。

由编码 agent 在满足任一退出条件后运行：解析 results.tsv，找出基线（首行）与
最佳（keep/best 且 SLO 达标的最高 goodput；无达标行则回退最高 goodput 并标注），
渲染中文 report.md（概览 / 基线 vs 最佳 / 全部 trial / 最佳配置 / 部署 / 系统级建议 / 适用范围），
并在 --out 同目录生成可执行的 deploy-best.sh（复用 bench.py 的 docker 命令构建器）。

只读性：绝不修改 results.tsv / config.yaml / env_report.md / model_profile.json；
仅写 --out 指定的 report.md 与同目录 deploy-best.sh（git show 仅读取历史）。
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from typing import NoReturn

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("缺少 pyyaml：请用 `uv run python report.py` 或 `pip install pyyaml`")

# 结果账本列（单一来源；scripts/record.py 从此导入）。列序固定，但 parse_tsv 按名字对齐：
# 旧 10 列表与新 16 列表都接受——核心列必须齐全，扩展列可选。
HEADER = ["commit", "trial_id", "config_hash", "workload_hash", "slo", "repeats",
          "warmup_repeats", "contention",
          "goodput", "goodput_std", "output_tok_s", "ttft_p99_ms", "tpot_p99_ms",
          "peak_vram_gb", "dur_s", "status", "description"]
# 数值列（coerce 用）：纯诊断列除外——contention 保持字符串/None，不参与数值解析
NUM_COLS = {"goodput", "goodput_std", "output_tok_s", "ttft_p99_ms", "tpot_p99_ms",
            "peak_vram_gb", "dur_s", "repeats", "warmup_repeats"}
# 任何表头都必须包含的核心列（旧表与扩展表的交集）
CORE_COLS = ["commit", "goodput", "goodput_std", "output_tok_s", "ttft_p99_ms", "tpot_p99_ms",
             "peak_vram_gb", "dur_s", "status", "description"]
DEFAULT_SLO = {"ttft_p99_ms": 500, "tpot_p99_ms": 50}


def _err(msg: str) -> NoReturn:
    print(f"report.py: 错误: {msg}", file=sys.stderr)
    sys.exit(1)


def _fmt(v) -> str:
    """数字格式化：None → '—'；整数原样；浮点去尾零。"""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    return f"{v:.2f}".rstrip("0").rstrip(".")


# ── 纯函数（单元测试目标）──────────────────────────────────────────────

def _coerce(col: str, v: str):
    if col in NUM_COLS:
        if v in ("", "-", "None", "null", "n/a"):
            return None
        try:
            return float(v)
        except ValueError:
            return None
    return v


def parse_tsv(text: str) -> list[dict]:
    """解析 results.tsv：按名字对齐——表头必须包含全部核心列（CORE_COLS），列序无关；
    扩展列（trial_id/config_hash/workload_hash/slo/repeats/contention）可选，缺失 → None。
    兼容旧 10 列表与新的 16 列表；数值列转 float（坏值/空 → None）。"""
    lines = [l for l in text.splitlines() if l.strip() != ""]
    if not lines:
        return []
    header = [c.strip() for c in lines[0].split("\t")]
    missing = [c for c in CORE_COLS if c not in header]
    if missing:
        _err(f"results.tsv 表头必须包含核心列：{', '.join(missing)}（缺少；实际表头: {header}）。"
             "旧 10 列表可继续用；新列见 report.HEADER（scripts/record.py 按此生成）")
    idx = {h: i for i, h in enumerate(header)}
    rows = []
    for line in lines[1:]:
        cells = line.split("\t")
        row = {}
        for h in HEADER:
            if h in idx:
                i = idx[h]
                row[h] = _coerce(h, cells[i].strip() if i < len(cells) else "")
            else:
                row[h] = _coerce(h, "")   # 扩展列缺失 → None（数值）/ 空串（字符串）
        rows.append(row)
    return rows


def slo_ok(row: dict, slo: dict | None = None) -> bool:
    """SLO 达标：ttft_p99 ≤ slo.ttft_p99_ms 且 tpot_p99 ≤ slo.tpot_p99_ms；缺值 → False。"""
    sl = {**DEFAULT_SLO, **(slo or {})}
    t, p = row.get("ttft_p99_ms"), row.get("tpot_p99_ms")
    if t is None or p is None:
        return False
    return t <= sl["ttft_p99_ms"] and p <= sl["tpot_p99_ms"]


def pick_baseline(rows: list[dict]) -> dict | None:
    """基线 = 首个数据行（与文件顺序一致，即 Phase 2 的第一个 trial）。"""
    return rows[0] if rows else None


def context_key(row: dict) -> tuple | None:
    """可比性上下文 = (workload_hash, slo)。两者皆空（旧表/未知）→ None（legacy，
    best 选择退化为旧行为——全部行）；其一存在 → 以该元组为上下文。"""
    w, s = row.get("workload_hash"), row.get("slo")
    if not w and not s:
        return None
    return (w or "", s or "")


def _ctx_rows(rows: list[dict], baseline_ctx: tuple | None) -> list[dict]:
    """best 选择候选：基线上下文已知时只留同上下文行（防止跨 workload/SLO 混比）；
    未知（legacy）→ 全部行（旧行为）。"""
    if baseline_ctx is None:
        return rows
    return [r for r in rows if context_key(r) == baseline_ctx]


def _protocol_mismatch_line(rows: list[dict], baseline_ctx: tuple | None) -> str | None:
    """基线上下文内出现多种 (repeats, warmup_repeats) 测量协议时的 ⚠ 文案（单一协议/无基线 → None）。
    协议 = (repeats, warmup_repeats)；任一缺失（None/空）按 0 归一（= 旧口径 / 未知），
    渲染值一致用 0。"""
    if baseline_ctx is None:
        return None
    bctx = [r for r in rows if context_key(r) == baseline_ctx]
    protocols = sorted({(r.get("repeats") or 0, r.get("warmup_repeats") or 0) for r in bctx})
    if len(protocols) <= 1:
        return None
    protos = "、".join(f"(repeats={_fmt(p[0])}, warmup={_fmt(p[1])})" for p in protocols)
    return (f"- ⚠ 基线上下文内存在多种测量协议 {protos}：不同 (repeats, warmup_repeats) 协议下的行"
            "严格不可比（账本记录协议；跨协议慎做 keep 判定）。")


def pick_best(rows: list[dict], slo: dict | None = None,
              baseline_ctx: tuple | None = None) -> tuple[dict | None, bool]:
    """最佳：状态恰为 {keep, best} 且 SLO 达标的最高 goodput 行——confirm 行绝不参与
    best 选择（只作回归核对，见「确认（best 回归）」节）；只在与基线同 (workload_hash, slo)
    上下文内选（baseline_ctx=None 的旧表 → 全部行）。
    无达标行时：先取 {keep, best} 中最高 goodput（即便不达标）；仅当 keep/best 行
    不存在时，才回退全部已测量（非 crash、非 confirm）行的最高 goodput。
    两种情况均置 flagged=True（报告中提示 ⚠ 人工复核）。"""
    cand = _ctx_rows(rows, baseline_ctx)
    eligible = [r for r in cand if r.get("status") in ("keep", "best")]  # 明确排除 confirm
    met = [r for r in eligible if slo_ok(r, slo)]
    if met:
        return max(met, key=lambda r: r.get("goodput") or 0), False
    if eligible:
        return max(eligible, key=lambda r: r.get("goodput") or 0), True
    # 回退：已测量行中最高 goodput；confirm/crash 均不参与 best 选择
    measured = [r for r in cand
                if r.get("goodput") is not None and r.get("status") not in ("crash", "confirm")]
    if measured:
        return max(measured, key=lambda r: r.get("goodput") or 0), True
    return None, True


def improvement_pct(base: dict | None, best: dict | None) -> float | None:
    """goodput 相对基线的提升百分比；缺数据/基数为 0 → None。"""
    if base is None or best is None:
        return None
    b, t = base.get("goodput"), best.get("goodput")
    if b is None or t is None or b == 0:
        return None
    return round((t - b) / b * 100, 1)


def extract_system_advice(env_report_md: str | None) -> list[dict]:
    """从 env_report.md 提取 warn/fail 检查项（markdown 表格行；尽力而为）。"""
    if not env_report_md:
        return []
    items = []
    for line in env_report_md.splitlines():
        m = re.match(r"\|\s*(.+?)\s*\|\s*(warn|fail)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|", line)
        if m:
            items.append({"name": m.group(1), "status": m.group(2),
                          "detail": m.group(3), "fix": m.group(4)})
    return items


def git_show_config(commit: str | None) -> str | None:
    """尽力取 config.yaml 快照（git show）；git 不可用/失败 → None。"""
    if not commit:
        return None
    try:
        p = subprocess.run(["git", "show", f"{commit}:config.yaml"],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout if p.returncode == 0 else None


# ── 部署产物（best 配置 → 可直接运行的 docker 命令）────────────────────

def _deploy_port(cmd: list[str]) -> int:
    """serve 端口从命令里取（build_docker_run_cmd 必带 --port）；缺省 8000。"""
    try:
        return int(cmd[cmd.index("--port") + 1])
    except (ValueError, IndexError):
        return 8000


def build_deploy_cmd(best_config_text: str, target: dict, harness: dict,
                     best: dict, slo: dict) -> list[str] | None:
    """由 best commit 的 config.yaml 快照构建独立部署的 docker run 命令——
    复用 bench.build_docker_run_cmd + effective_runtime（不重复实现）。
    部署容器名 vllm-deploy-best（与 trial 的 vllm-autotune-trial 不同，后续调优轮次的
    docker rm -f 不会误杀部署）。
    有效 serve 端口 = best 配置 serve.port（若有），否则 harness.remote.vllm_port（缺省 8000），
    使 -p 发布端口与容器内 serve 端口一致。
    任一前提缺失（YAML 解析失败 / bench 不可导入 / runtime.image 缺 /
    target.model.path 缺）→ None（降级跳过，绝不 crash）。"""
    try:
        import bench  # 复用 bench.py 的 docker 命令构建器
        config = yaml.safe_load(best_config_text) or {}
        runtime = bench.effective_runtime(harness, config)
        model_path = (target.get("model") or {}).get("path")
        if not model_path:
            return None
        serve_port = int(((config.get("serve") or {}).get("port"))
                         or ((harness.get("remote") or {}).get("vllm_port")) or 8000)
        return bench.build_docker_run_cmd("docker", runtime, model_path, serve_port, config,
                                          name="vllm-deploy-best")
    except Exception:
        return None  # 降级：构建失败不阻塞报告


def write_deploy_artifact(out_dir: str, cmd: list[str], best: dict, slo: dict) -> str:
    """写 <out_dir>/deploy-best.sh（可执行）；返回文件路径。"""
    os.makedirs(out_dir, exist_ok=True)
    g, sig = best.get("goodput"), best.get("goodput_std")
    header = [
        "#!/usr/bin/env bash",
        "# 部署 best 配置（report.py 生成；仅 serve 单实例，不含 bench/循环）",
        f"# 最佳 commit: {best.get('commit') or '?'}  goodput: {_fmt(g)}"
        + (f"（σ={_fmt(sig)}）" if sig is not None else "")
        + f"  SLO: ttft≤{slo.get('ttft_p99_ms')}ms / tpot≤{slo.get('tpot_p99_ms')}ms",
        f"# 生成时间: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "# 容器名 vllm-deploy-best：与 trial 的 vllm-autotune-trial 不同，后续调优轮次的清理不会误杀本部署。",
        f"# 镜像须在主机预先准备（模板不自动拉取）；模型目录只读挂载进容器；serve 端口 {_deploy_port(cmd)}。",
        "set -euo pipefail",
        "exec " + shlex.join(cmd),
        "",
    ]
    path = os.path.join(out_dir, "deploy-best.sh")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header))
    os.chmod(path, 0o755)
    return path


# ── 渲染（纯函数，测试目标）────────────────────────────────────────────

def _ref_eligible(rows: list[dict], baseline_ctx: tuple | None) -> list[dict]:
    """参考 SLO 对标口径（F1）：基线上下文内的 keep/best 候选行——discard 不是候选 best，
    confirm 只作回归核对不参与；与 best_under_slo / 可达性判定完全同口径。"""
    cand = _ctx_rows(rows, baseline_ctx)
    return [r for r in cand if r.get("status") in ("keep", "best")]


def _any_row_satisfies(rows: list[dict], slo: dict | None, baseline_ctx: tuple | None) -> bool:
    """参考 SLO 可达性：仅在基线上下文内、keep/best 行中判定（与 best_under_slo 同口径，
    杜绝「参考 SLO 可达」与「参考 SLO 下无达标行」并存的矛盾）。"""
    return any(slo_ok(r, slo) for r in _ref_eligible(rows, baseline_ctx))


def best_under_slo(rows: list[dict], slo: dict | None, baseline_ctx: tuple | None) -> dict | None:
    """给定 SLO（参考）下、基线上下文内的最优 keep/best 行；无达标行 → None。"""
    met = [r for r in _ref_eligible(rows, baseline_ctx) if slo_ok(r, slo)]
    return max(met, key=lambda r: r.get("goodput") or 0) if met else None


def render(rows: list[dict], slo: dict | None, baseline: dict | None, best: dict | None,
           best_flagged: bool, improvement: float | None, *, model: str = "（未知）",
           workload: dict | None = None, config_snapshot: str | None = None,
           system_advice: list[dict] | None = None,
           deploy_cmd: list[str] | None = None,
           noise_band_pct: float = 3.0, noise_k_sigma: float = 2.0,
           slo_calibrated: dict | None = None,
           reference_slo: dict | None = None, baseline_ctx: tuple | None = None,
           comparability: list | None = None) -> str:
    sl = {**DEFAULT_SLO, **(slo or {})}
    ref = {**DEFAULT_SLO, **(reference_slo or {})}
    wl = workload or {}
    counts = Counter(r.get("status") for r in rows)
    lines = ["# 调优报告（report.md）", ""]

    lines += ["## 概览", ""]
    lines.append(f"- 模型：`{model}`")
    lines.append(f"- workload：dataset=`{wl.get('dataset', 'random')}`，"
                 f"isl=`{wl.get('isl', 1024)}`，osl=`{wl.get('osl', 128)}`，"
                 f"num_prompts=`{wl.get('num_prompts', 1000)}`"
                 + (f"，ref_concurrency=`{wl.get('ref_concurrency')}`" if wl.get("ref_concurrency") else ""))
    lines.append("- 各 trial 实际生效 workload 以 `result.json` 的 `workload` 字段为准（跨 workload 的行不可混比）。")
    if reference_slo:
        # 双口径：参考 SLO（target 声明，不改语义）vs 有效 SLO（config.slo 运行期校准，判定用）
        lines.append(f"- 参考 SLO（target 声明，不改语义）：ttft_p99 ≤ {ref['ttft_p99_ms']} ms；"
                     f"tpot_p99 ≤ {ref['tpot_p99_ms']} ms")
        lines.append(f"- 有效 SLO（goodput/keep 判定用）：ttft_p99 ≤ {sl['ttft_p99_ms']} ms；"
                     f"tpot_p99 ≤ {sl['tpot_p99_ms']} ms"
                     + ("（与参考一致）" if ref == sl else "（运行期校准，见下）"))
        if _any_row_satisfies(rows, ref, baseline_ctx):
            lines.append("- 参考 SLO：**可达**（基线上下文内存在满足参考 SLO 的 keep/best trial）")
        else:
            # 「不可达」仅指无 keep/best 候选达标；若存在非 keep/best 的已测量行达标（如被 discard），
            # 说明时延本身可达 → 软化建议，避免误导为「放宽 SLO」。
            measured_ref = [r for r in _ctx_rows(rows, baseline_ctx)
                            if r.get("status") not in ("crash", "confirm") and slo_ok(r, ref)]
            if measured_ref:
                lines.append("- 参考 SLO：**不可达**（无 keep/best trial 满足参考 SLO；但存在已测量行达标却未通过 keep 判定）"
                             "——建议人工复核该配置（是否仅被噪声带拦下），而非直接放宽 SLO。")
            else:
                lines.append("- 参考 SLO：**不可达**（基线上下文内无 keep/best trial 满足参考 SLO；best 为有效 SLO 下的最优；"
                             "建议放宽至实测 p99 或复核硬件/模型适配）")
    else:
        lines.append(f"- SLO：ttft_p99 ≤ {sl['ttft_p99_ms']} ms；tpot_p99 ≤ {sl['tpot_p99_ms']} ms")
    if slo_calibrated:
        ref_marker = {**DEFAULT_SLO, **slo_calibrated}
        lines.append(f"- ⚠ **运行期 SLO 校准**：参考 ttft {ref_marker['ttft_p99_ms']}→{sl['ttft_p99_ms']} ms、"
                     f"tpot {ref_marker['tpot_p99_ms']}→{sl['tpot_p99_ms']} ms 为有效 SLO"
                     "（本报告 SLO 达标列与 best 判定均按有效 SLO）。")
    lines.append(f"- trial 数：{len(rows)}（keep={counts.get('keep', 0)}，"
                 f"discard={counts.get('discard', 0)}，crash={counts.get('crash', 0)}，"
                 f"best={counts.get('best', 0)}，confirm={counts.get('confirm', 0)}）")
    # M2b：基线上下文内协议（repeats/warmup_repeats）不一致 → ⚠（挂在可比性节，无节则挂概览）
    proto_line = _protocol_mismatch_line(rows, baseline_ctx)
    show_ctx_section = bool(comparability) and (
        len(comparability) > 1 or (baseline_ctx is None and len(comparability) >= 1))
    if proto_line is not None and not show_ctx_section:
        lines.append(proto_line)
    lines.append(f"- 生成时间：{datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")

    # F4：legacy 基线（无上下文）+ ≥1 个非空上下文 → 也要出可比性节（否则静默混比）；
    # 基线有上下文时仅在 >1 个不同上下文（含基线外的）才出节。
    if show_ctx_section and comparability:
        lines += ["## 可比性", ""]
        lines.append("账本包含多个不同的 `(workload_hash, slo)` 上下文——跨上下文行不可混比：")
        lines.append("")
        lines.append("| workload_hash | slo | trial 数 |")
        lines.append("|---|---|---|")
        for (w, s), n in comparability:
            lines.append(f"| {w or '—'} | {s or '—'} | {n} |")
        lines.append("")
        if baseline_ctx is not None:
            lines.append(f"- 仅基线上下文（`{baseline_ctx[0] or '—'}` / `{baseline_ctx[1] or '—'}`）"
                         "参与 best 选择；其余上下文仅供诊断（不混比）。")
        else:
            lines.append("- 基线为旧表行（无上下文）：best 选择按旧行为覆盖全部行，但各上下文不可混比"
                         "——建议用 scripts/record.py 重建账本。")
        if proto_line is not None:
            lines.append(proto_line)   # M2b：基线上下文内协议不一致的 ⚠ 挂在可比性节内
        lines.append("")

    lines += ["## 基线 vs 最佳", ""]
    lines.append("| 指标 | 基线 | 最佳 |")
    lines.append("|---|---|---|")
    for col, label in (("goodput", "goodput（req/s）"), ("output_tok_s", "output_tok_s（tok/s）"),
                       ("ttft_p99_ms", "ttft_p99（ms）"), ("tpot_p99_ms", "tpot_p99（ms）")):
        lines.append(f"| {label} | {_fmt(baseline.get(col) if baseline else None)} | "
                     f"{_fmt(best.get(col) if best else None)} |")
    lines.append("")
    if best is not None:
        lines.append(f"- goodput 提升：**{_fmt(improvement)}%**"
                     f"（基线 {_fmt(baseline.get('goodput') if baseline else None)} → "
                     f"最佳 {_fmt(best.get('goodput'))}，commit={best.get('commit')}）"
                     + ("，均按**有效 SLO**" if reference_slo else ""))
        if best_flagged:
            lines.append("- ⚠ **无满足 SLO 的 trial**：最佳为「全部已测量行中最高 goodput」的回退选择，"
                         "需人工复核（SLO 是否过严 / 配置是否需改进）。")
        if reference_slo:
            # 参考 SLO 对标：基线上下文内的参考 SLO 最优行（存在才列，否则明示无）
            b_ref = best_under_slo(rows, ref, baseline_ctx)
            if b_ref is not None:
                lines.append(f"- 参考 SLO 下的最佳：commit={b_ref.get('commit')}，"
                             f"goodput={_fmt(b_ref.get('goodput'))} req/s"
                             f"（ttft_p99 {_fmt(b_ref.get('ttft_p99_ms'))} ms ≤ {ref['ttft_p99_ms']}、"
                             f"tpot_p99 {_fmt(b_ref.get('tpot_p99_ms'))} ms ≤ {ref['tpot_p99_ms']}）"
                             + ("；与有效 SLO best 相同" if b_ref is best else "。"))
            else:
                lines.append("- 参考 SLO 下无达标行（参考 SLO 不可达；best 为有效 SLO 下的最优）。")
    else:
        # F5：行存在但基线上下文内无候选 → 明确文案；仅当表真空/全 crash 才用原文案
        if not rows or all(r.get("status") == "crash" for r in rows):
            lines.append("- 无任何可比较的 trial（results.tsv 为空或全部 crash）。")
        else:
            lines.append("- 基线上下文内无可比较的 trial（行存在，但基线上下文内无候选；"
                         "其余上下文/legacy 行不参与 best 选择）。")
    lines.append("")

    lines += ["## 全部 trial 表", ""]
    lines.append("| commit | goodput | goodput_std | output_tok_s | ttft_p99 | tpot_p99 | peak_vram | dur_s | SLO 达标 | status | description |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    def _gp(r: dict) -> float:
        g = r.get("goodput")
        return g if isinstance(g, (int, float)) else -1.0

    ordered = sorted(rows, key=_gp, reverse=True)
    for r in ordered:
        ok = slo_ok(r, sl)
        lines.append(f"| {r.get('commit') or '—'} | {_fmt(r.get('goodput'))} | "
                     f"{_fmt(r.get('goodput_std'))} | {_fmt(r.get('output_tok_s'))} | "
                     f"{_fmt(r.get('ttft_p99_ms'))} | {_fmt(r.get('tpot_p99_ms'))} | "
                     f"{_fmt(r.get('peak_vram_gb'))} | {_fmt(r.get('dur_s'))} | "
                     f"{'✅' if ok else '❌'} | {r.get('status') or '—'} | "
                     f"{r.get('description') or ''} |")
    lines.append("")

    best_commit = best.get("commit") if best else None
    lines += [f"## 最佳配置（commit={best_commit or '（无）'}，config.yaml 内容）", ""]
    if config_snapshot:
        lines += ["```yaml", config_snapshot.rstrip("\n"), "```"]
    else:
        lines += [f"`git show {best_commit}:config.yaml` 未能自动读取（git 不可用或 commit 不在仓库）；"
                  "请手动执行该命令查看最佳配置。" if best_commit else
                  "无最佳配置可展示（results.tsv 为空或全部 crash）。"]
    lines.append("")

    lines += ["## 部署（best 配置）", ""]
    if deploy_cmd:
        lines += ["```bash", shlex.join(deploy_cmd), "```", ""]
        lines.append("- 运行：`bash deploy-best.sh`（report.py 生成于 `--out` 同目录，可直接执行）。")
        lines.append("- 镜像须在主机**预先准备**（模板不自动拉取，零新增）；模型目录经 `-v host:container:ro` 只读挂载。")
        lines.append(f"- serve 就绪后访问 `http://127.0.0.1:{_deploy_port(deploy_cmd)}`（`/health` 返回 200 即可用）。")
    else:
        lines.append("未生成部署命令（无 best 配置 / `git show` 失败 / 缺 `target.yaml.model.path`）——"
                     "最佳配置见上节，可手动起服。")
    lines.append("")

    lines += ["## 确认（best 回归）", ""]
    confirm_rows = [r for r in rows if r.get("status") == "confirm"]
    latest = confirm_rows[-1] if confirm_rows else None
    if latest is not None and best is not None and best.get("goodput"):
        bg, cg = best["goodput"], latest.get("goodput")
        # 复用 keep 规则公式（合并双侧不确定度的标准误）：阈值 =
        # max(best × noise_band_pct/100, k × sqrt(σ_best²/n_best + σ_confirm²/n_confirm))；
        # n = 该行 repeats（min 1，缺失 → 1）；n<2 → 该侧 σ 项贡献 0（stdev 未定义 → σ=0）
        bs = best.get("goodput_std") or 0.0
        cs = latest.get("goodput_std") or 0.0
        bn = max(1, int(best.get("repeats") or 1))
        cn = max(1, int(latest.get("repeats") or 1))

        def _se_term(sig: float, n: int) -> float:
            return sig if n >= 2 else 0.0

        threshold = max(bg * noise_band_pct / 100.0,
                        noise_k_sigma * math.sqrt(_se_term(bs, bn) ** 2 / bn
                                                  + _se_term(cs, cn) ** 2 / cn))
        delta = abs((cg or 0.0) - bg)
        if delta <= threshold:
            verdict = "确认（噪声带内一致）"
        else:
            verdict = f"未确认（差异 {_fmt(delta)} 超噪声带 {_fmt(threshold)}）"
        lines += [
            "| 指标 | best | confirm（最新） |",
            "|---|---|---|",
            f"| goodput（req/s） | {_fmt(bg)} | {_fmt(cg)} |",
            f"| goodput_std（σ） | {_fmt(bs)} | {_fmt(cs)} |",
            f"| repeats | {_fmt(best.get('repeats'))} | {_fmt(latest.get('repeats'))} |",
            f"| Δ（|confirm−best|） | — | {_fmt(delta)} |",
            "",
        ]
        lines.append(f"- 判定：**{verdict}**（阈值 = max(best × {noise_band_pct}/100, "
                     f"{noise_k_sigma} × sqrt(σ_best²/n_best + σ_confirm²/n_confirm)) = "
                     f"{_fmt(threshold)} req/s；σ 为 best / confirm 行 goodput_std，"
                     f"n 为各行 repeats）。")
    elif latest is not None:
        lines.append("存在 confirm 行，但无 best（keep/best）行可核对——best 回退选择见「基线 vs 最佳」。")
    else:
        lines.append("未做确认 run（建议收尾对 best 再跑一次，results.tsv 记 `status=confirm`）。")
    lines.append("")

    lines += ["## 系统级建议（附录）", ""]
    advice = system_advice or []
    if advice:
        lines.append("以下 warn/fail 项提取自 `env_report.md`（v1 只建议，不自动修改）：")
        lines.append("")
        lines.append("| 检查项 | 状态 | 详情 | 修复命令 |")
        lines.append("|---|---|---|---|")
        for it in advice:
            lines.append(f"| {it['name']} | {it['status']} | {it['detail']} | {it['fix']} |")
    else:
        lines.append("未提供 `env_report.md`，或其中无 warn/fail 项。")
    lines.append("")

    lines += ["## 适用范围", ""]
    lines.append("本报告结论仅适用于 v1 单节点（head 主机独占）的实测环境；跨机/集群配置**不外推**。")
    lines.append("")
    return "\n".join(lines)


# ── CLI 入口 ────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 4 收尾：results.tsv → report.md（只读 harness）")
    ap.add_argument("--results", default="results.tsv")
    ap.add_argument("--harness", default="harness.yaml")
    ap.add_argument("--target", default="target.yaml")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--env-report", default="env_report.md")
    ap.add_argument("--model-profile", default="model_profile.json")
    ap.add_argument("--out", default="report.md")
    args = ap.parse_args(argv)

    if not os.path.exists(args.results):
        _err(f"未找到 {args.results}（先运行 probe/bench 循环后再收尾）。")
    with open(args.results, encoding="utf-8") as f:
        rows = parse_tsv(f.read())

    harness = {}
    if os.path.exists(args.harness):
        with open(args.harness, encoding="utf-8") as f:
            harness = yaml.safe_load(f) or {}
    target = {}
    if os.path.exists(args.target):
        with open(args.target, encoding="utf-8") as f:
            target = yaml.safe_load(f) or {}
    config = {}
    if os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    # 有效 SLO：harness.slo ⊕ target.metrics.slo ⊕ config.slo（与 bench.py 同口径，共用 resolve_slo）
    try:
        from bench import resolve_slo
    except ImportError:  # pragma: no cover — 降级：bench 不可导入时内联同表达式
        def resolve_slo(harness: dict, target: dict, config: dict) -> dict:
            return {**harness.get("slo", {}),
                    **((target.get("metrics") or {}).get("slo") or {}),
                    **(config.get("slo") or {})}
    target_slo = {**harness.get("slo", {}), **((target.get("metrics") or {}).get("slo") or {})}
    config_slo = config.get("slo") or {}
    slo = resolve_slo(harness, target, config)
    # 双口径：参考 SLO（target 声明，不改语义）vs 有效 SLO（config.slo 运行期校准，判定用）
    reference_slo = target_slo
    # 校准标注：config.slo 非空且与参考（target 口径）不同 → 报告出「运行期 SLO 校准」
    slo_calibrated = target_slo if (config_slo and config_slo != target_slo) else None
    workload = {**harness.get("workload", {}), **(target.get("workload") or {})}
    # 确认回归的噪声参数（harness.bench；缺省与 harness.yaml 一致）
    bench_cfg = harness.get("bench") or {}
    noise_band_pct = float(bench_cfg.get("noise_band_pct", 3.0))
    noise_k_sigma = float(bench_cfg.get("noise_k_sigma", 2.0))

    model = "（未知）"
    if os.path.exists(args.model_profile):
        try:
            with open(args.model_profile, encoding="utf-8") as f:
                model = (json.load(f).get("model")) or model
        except (OSError, ValueError):
            pass

    env_report = None
    if os.path.exists(args.env_report):
        with open(args.env_report, encoding="utf-8") as f:
            env_report = f.read()

    baseline = pick_baseline(rows)
    # 可比性：基线上下文 (workload_hash, slo)；best 只在其内选（旧表基线无上下文 → 旧行为全行）
    baseline_ctx = context_key(baseline) if baseline else None
    comparability = sorted((k, n) for k, n in
                           Counter(context_key(r) for r in rows).items() if k is not None)
    # F4：legacy 基线（无上下文）+ ≥1 个非空上下文 → 警告（否则静默混比）
    if baseline_ctx is None and len(comparability) >= 1:
        # 旧表（基线无上下文）但行含非空上下文：best 覆盖全行，跨上下文不可混比
        print("警告: results.tsv 基线无 (workload_hash, slo) 上下文（旧表），但存在含上下文的行；"
              "best 选择按旧行为覆盖全部行，legacy 行上下文未知，跨上下文不可混比。"
              "建议用 scripts/record.py 重建账本。", file=sys.stderr)
    best, flagged = pick_best(rows, slo, baseline_ctx)
    imp = improvement_pct(baseline, best)
    snapshot = git_show_config(best.get("commit") if best else None)
    # 部署产物：best commit 的 config.yaml 快照 → 独立 serve 的 docker run（复用 bench.py 构建器）
    deploy_cmd = build_deploy_cmd(snapshot, target, harness, best, slo) if (best is not None and snapshot) else None
    md = render(rows, slo, baseline, best, flagged, imp, model=model, workload=workload,
                config_snapshot=snapshot, system_advice=extract_system_advice(env_report),
                deploy_cmd=deploy_cmd, noise_band_pct=noise_band_pct, noise_k_sigma=noise_k_sigma,
                slo_calibrated=slo_calibrated,
                reference_slo=reference_slo, baseline_ctx=baseline_ctx, comparability=comparability)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)

    if deploy_cmd:
        assert best is not None  # deploy_cmd 只在 best 存在时构建
        deploy_dir = os.path.dirname(args.out) or "."
        write_deploy_artifact(deploy_dir, deploy_cmd, best, slo)
        print(f"deploy-best.sh → {deploy_dir}/deploy-best.sh")

    print(f"trial 数: {len(rows)}（keep={sum(1 for r in rows if r['status'] == 'keep')}，"
          f"discard={sum(1 for r in rows if r['status'] == 'discard')}，"
          f"crash={sum(1 for r in rows if r['status'] == 'crash')}，"
          f"confirm={sum(1 for r in rows if r['status'] == 'confirm')}）")
    if best is not None:
        print(f"best: {best.get('commit')} goodput={_fmt(best.get('goodput'))}"
              f"（提升 {_fmt(imp)}%）"
              + ("；无 SLO 达标行，回退选择" if flagged else ""))
    else:
        print("无最佳行（results.tsv 为空或全部 crash）")
    print(f"report.md → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
