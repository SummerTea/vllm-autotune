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

HEADER = ["commit", "goodput", "goodput_std", "output_tok_s", "ttft_p99_ms", "tpot_p99_ms",
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
    """最佳：状态恰为 {keep, best} 且 SLO 达标的最高 goodput 行——confirm 行绝不参与
    best 选择（只作回归核对，见「确认（best 回归）」节）。
    无达标行时：先取 {keep, best} 中最高 goodput（即便不达标）；仅当 keep/best 行
    不存在时，才回退全部已测量（非 crash、非 confirm）行的最高 goodput。
    两种情况均置 flagged=True（报告中提示 ⚠ 人工复核）。"""
    eligible = [r for r in rows if r.get("status") in ("keep", "best")]  # 明确排除 confirm
    met = [r for r in eligible if slo_ok(r, slo)]
    if met:
        return max(met, key=lambda r: r.get("goodput") or 0), False
    if eligible:
        return max(eligible, key=lambda r: r.get("goodput") or 0), True
    # 回退：已测量行中最高 goodput；confirm/crash 均不参与 best 选择
    measured = [r for r in rows
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

def render(rows: list[dict], slo: dict | None, baseline: dict | None, best: dict | None,
           best_flagged: bool, improvement: float | None, *, model: str = "（未知）",
           workload: dict | None = None, config_snapshot: str | None = None,
           system_advice: list[dict] | None = None,
           deploy_cmd: list[str] | None = None,
           noise_band_pct: float = 3.0, noise_k_sigma: float = 2.0,
           slo_calibrated: dict | None = None) -> str:
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
    if slo_calibrated:
        ref = {**DEFAULT_SLO, **slo_calibrated}
        lines.append(f"- ⚠ **运行期 SLO 校准**：参考 ttft {ref['ttft_p99_ms']}→{sl['ttft_p99_ms']} ms、"
                     f"tpot {ref['tpot_p99_ms']}→{sl['tpot_p99_ms']} ms 为有效 SLO"
                     "（本报告 SLO 达标列与 best 判定均按有效 SLO）。")
    lines.append(f"- trial 数：{len(rows)}（keep={counts.get('keep', 0)}，"
                 f"discard={counts.get('discard', 0)}，crash={counts.get('crash', 0)}，"
                 f"best={counts.get('best', 0)}，confirm={counts.get('confirm', 0)}）")
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
        bs = best.get("goodput_std") or 0.0
        # 复用 keep 规则公式：阈值 = max(best × noise_band_pct/100, noise_k_sigma × σ)
        threshold = max(bg * noise_band_pct / 100.0, noise_k_sigma * bs)
        delta = abs((cg or 0.0) - bg)
        if delta <= threshold:
            verdict = "确认（噪声带内一致）"
        else:
            verdict = f"未确认（差异 {_fmt(delta)} 超噪声带 {_fmt(threshold)}）"
        lines += [
            "| 指标 | best | confirm（最新） |",
            "|---|---|---|",
            f"| goodput（req/s） | {_fmt(bg)} | {_fmt(cg)} |",
            f"| goodput_std（σ） | {_fmt(bs)} | {_fmt(latest.get('goodput_std'))} |",
            f"| Δ（|confirm−best|） | — | {_fmt(delta)} |",
            "",
        ]
        lines.append(f"- 判定：**{verdict}**（阈值 = max(best × {noise_band_pct}/100, "
                     f"{noise_k_sigma} × σ) = {_fmt(threshold)} req/s；σ=best 行的 goodput_std）。")
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
    best, flagged = pick_best(rows, slo)
    imp = improvement_pct(baseline, best)
    snapshot = git_show_config(best.get("commit") if best else None)
    # 部署产物：best commit 的 config.yaml 快照 → 独立 serve 的 docker run（复用 bench.py 构建器）
    deploy_cmd = build_deploy_cmd(snapshot, target, harness, best, slo) if (best is not None and snapshot) else None
    md = render(rows, slo, baseline, best, flagged, imp, model=model, workload=workload,
                config_snapshot=snapshot, system_advice=extract_system_advice(env_report),
                deploy_cmd=deploy_cmd, noise_band_pct=noise_band_pct, noise_k_sigma=noise_k_sigma,
                slo_calibrated=slo_calibrated)

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
