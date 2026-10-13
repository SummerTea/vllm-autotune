#!/usr/bin/env python3
"""report.py — Phase 4 收尾：results.tsv → report.md（只读 harness，唯一写出的文件是报告）。

由编码 agent 在满足任一退出条件后运行：解析 results.tsv，找出基线（首行）与
最佳（keep/best 且 SLO 达标的最高 goodput；无达标行则回退最高 goodput 并标注），
渲染中文 report.md（概览 / 基线 vs 最佳 / 全部 trial / 最佳配置 / 系统级建议 / 适用范围）。

只读性：绝不修改 results.tsv / config.yaml / env_report.md / model_profile.json；
仅写 --out 指定的 report.md（git show 仅读取历史）。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from collections import Counter
from typing import NoReturn

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("缺少 pyyaml：请用 `uv run python report.py` 或 `pip install pyyaml`")

HEADER = ["commit", "goodput", "output_tok_s", "ttft_p99_ms", "tpot_p99_ms",
          "peak_vram_gb", "dur_s", "status", "description"]
NUM_COLS = set(HEADER) - {"commit", "status", "description"}
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
    """解析 results.tsv：表头必须恰为 HEADER；数值列转 float（坏值/空 → None）。"""
    lines = [l for l in text.splitlines() if l.strip() != ""]
    if not lines:
        return []
    header = [c.strip() for c in lines[0].split("\t")]
    if header != HEADER:
        sep = "\t"
        _err(f"results.tsv 表头必须恰为 {sep.join(HEADER)}（实际: {header}）")
    rows = []
    for line in lines[1:]:
        cells = [c for c in line.split("\t")]
        if len(cells) < len(HEADER):
            cells += [""] * (len(HEADER) - len(cells))
        else:
            cells = cells[: len(HEADER)]
        rows.append({h: _coerce(h, c.strip()) for h, c in zip(HEADER, cells)})
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


def pick_best(rows: list[dict], slo: dict | None = None) -> tuple[dict | None, bool]:
    """最佳：状态 ∈ {keep, best} 且 SLO 达标的最高 goodput 行。
    无达标行时：先取 {keep, best} 中最高 goodput（即便不达标）；仅当 keep/best 行
    不存在时，才回退全部已测量（非 crash）行的最高 goodput。
    两种情况均置 flagged=True（报告中提示 ⚠ 人工复核）。"""
    eligible = [r for r in rows if r.get("status") in ("keep", "best")]
    met = [r for r in eligible if slo_ok(r, slo)]
    if met:
        return max(met, key=lambda r: r.get("goodput") or 0), False
    if eligible:
        return max(eligible, key=lambda r: r.get("goodput") or 0), True
    measured = [r for r in rows if r.get("goodput") is not None and r.get("status") != "crash"]
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


# ── 渲染（纯函数，测试目标）────────────────────────────────────────────

def render(rows: list[dict], slo: dict | None, baseline: dict | None, best: dict | None,
           best_flagged: bool, improvement: float | None, *, model: str = "（未知）",
           workload: dict | None = None, config_snapshot: str | None = None,
           system_advice: list[dict] | None = None) -> str:
    sl = {**DEFAULT_SLO, **(slo or {})}
    wl = workload or {}
    counts = Counter(r.get("status") for r in rows)
    lines = ["# 调优报告（report.md）", ""]

    lines += ["## 概览", ""]
    lines.append(f"- 模型：`{model}`")
    lines.append(f"- workload：dataset=`{wl.get('dataset', 'random')}`，"
                 f"isl=`{wl.get('isl', 1024)}`，osl=`{wl.get('osl', 128)}`，"
                 f"num_prompts=`{wl.get('num_prompts', 1000)}`"
                 + (f"，ref_concurrency=`{wl.get('ref_concurrency')}`" if wl.get("ref_concurrency") else ""))
    lines.append(f"- SLO：ttft_p99 ≤ {sl['ttft_p99_ms']} ms；tpot_p99 ≤ {sl['tpot_p99_ms']} ms")
    lines.append(f"- trial 数：{len(rows)}（keep={counts.get('keep', 0)}，"
                 f"discard={counts.get('discard', 0)}，crash={counts.get('crash', 0)}，"
                 f"best={counts.get('best', 0)}）")
    lines.append(f"- 生成时间：{datetime.datetime.now().isoformat(timespec='seconds')}")
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
                     f"最佳 {_fmt(best.get('goodput'))}，commit={best.get('commit')}）")
        if best_flagged:
            lines.append("- ⚠ **无满足 SLO 的 trial**：最佳为「全部已测量行中最高 goodput」的回退选择，"
                         "需人工复核（SLO 是否过严 / 配置是否需改进）。")
    else:
        lines.append("- 无任何可比较的 trial（results.tsv 为空或全部 crash）。")
    lines.append("")

    lines += ["## 全部 trial 表", ""]
    lines.append("| commit | goodput | output_tok_s | ttft_p99 | tpot_p99 | peak_vram | dur_s | SLO 达标 | status | description |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    def _gp(r: dict) -> float:
        g = r.get("goodput")
        return g if isinstance(g, (int, float)) else -1.0

    ordered = sorted(rows, key=_gp, reverse=True)
    for r in ordered:
        ok = slo_ok(r, sl)
        lines.append(f"| {r.get('commit') or '—'} | {_fmt(r.get('goodput'))} | "
                     f"{_fmt(r.get('output_tok_s'))} | {_fmt(r.get('ttft_p99_ms'))} | "
                     f"{_fmt(r.get('tpot_p99_ms'))} | {_fmt(r.get('peak_vram_gb'))} | "
                     f"{_fmt(r.get('dur_s'))} | {'✅' if ok else '❌'} | {r.get('status') or '—'} | "
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
    slo = {**harness.get("slo", {}), **((target.get("metrics") or {}).get("slo") or {})}
    workload = {**harness.get("workload", {}), **(target.get("workload") or {})}

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
    best, flagged = pick_best(rows, slo)
    imp = improvement_pct(baseline, best)
    snapshot = git_show_config(best.get("commit") if best else None)
    md = render(rows, slo, baseline, best, flagged, imp, model=model, workload=workload,
                config_snapshot=snapshot, system_advice=extract_system_advice(env_report))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"trial 数: {len(rows)}（keep={sum(1 for r in rows if r['status'] == 'keep')}，"
          f"discard={sum(1 for r in rows if r['status'] == 'discard')}，"
          f"crash={sum(1 for r in rows if r['status'] == 'crash')}）")
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
