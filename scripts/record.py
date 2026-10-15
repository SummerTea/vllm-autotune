#!/usr/bin/env python3
"""scripts/record.py — 从 result.json 机械追加一行 results.tsv（Phase 2 双口径记账）。

手写 results.tsv 行在加入身份/上下文列（trial_id/config_hash/workload_hash/slo 等）后
极易错列；本工具把"本次 trial 产物 → 账本行"变成纯机械步骤：

  python scripts/record.py --status keep --desc "max-num-seqs 256" [--commit <hash>]

- 字段全部来自 --result-json（默认 ./result.json，run.sh --poll done 后回传本机）。
- 表头（HEADER）单一来源 = report.py（report.HEADER）；已有 results.tsv 表头必须与之
  完全一致，否则报错提示迁移（不静默追加进错列表）。
- 行内数值/身份/上下文列均从 result.json 提取；crash 行数值列留空。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# 单一来源：report.py 的 HEADER（脚本在 scripts/ 下，先补仓库根到 sys.path）
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from report import HEADER  # noqa: E402


def _err(msg: str) -> None:
    print(f"record.py: 错误: {msg}", file=sys.stderr)
    sys.exit(1)


def load_result(path: str) -> dict:
    """读 result.json；缺失/解析失败 → 明确报错退出（不产出任何账本行）。"""
    if not os.path.exists(path):
        _err(f"未找到 {path}（先 `bash run.sh --poll` 至 done 回传 result.json 再记录）")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        _err(f"解析 {path} 失败: {e}")
    raise AssertionError("unreachable: _err 已退出")


def current_commit() -> str:
    """git rev-parse --short HEAD（best-effort；git 不可用/失败 → 空）。"""
    try:
        p = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def workload_hash(result: dict) -> str:
    """workload_hash = sha256(result.workload 规范化 JSON)[:12]；workload 缺失 → 空。"""
    wl = result.get("workload")
    if not wl:
        return ""
    blob = json.dumps(wl, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def format_slo(slo: dict | None) -> str:
    """slo 单元格：`ttft<A>/tpot<B>`——整数值不带小数点（4000 与 4000.0 同格，防伪上下文分裂）；
    任一键缺失/None → 整格空（不产出 ttftNone/...）。"""
    if not slo:
        return ""
    t, p = slo.get("ttft_p99_ms"), slo.get("tpot_p99_ms")
    if t is None or p is None:
        return ""

    def _norm(v) -> str:
        try:
            f = float(v)
            return str(int(f)) if f.is_integer() else str(v)
        except (TypeError, ValueError):
            return str(v)   # 非数值 → 原样（防御）

    return f"ttft{_norm(t)}/tpot{_norm(p)}"


def resolve_slo_effective(result: dict, harness_path: str, target_path: str,
                          config_path: str) -> dict | None:
    """slo 单元格来源：result.json["slo"]（已含有效 SLO）优先；缺失时回退
    bench.resolve_slo(harness, target, config)（与 bench/report 同口径）。"""
    slo = result.get("slo")
    if slo:
        return slo if isinstance(slo, dict) else None
    try:
        import yaml  # noqa: PLC0415
        from bench import resolve_slo  # noqa: PLC0415
        def _load(p: str) -> dict:
            return yaml.safe_load(open(p, encoding="utf-8")) or {} if os.path.exists(p) else {}
        return resolve_slo(_load(harness_path), _load(target_path), _load(config_path)) or None
    except Exception:  # 回退失败 → None（单元格留空）
        return None


def build_row(result: dict, *, commit: str, desc: str, status: str, slo: dict | None,
              wl_hash: str) -> list[str]:
    """按 HEADER 顺序拼一行 cell（crash 行数值列留空；身份/上下文列保留）。"""
    ok = result.get("status") == "ok"

    def metric(v) -> str:
        """数值 cell：非 ok 行（crash/占位）或值缺失 → 空串。"""
        return "" if (not ok or v is None) else str(v)

    return [
        commit,
        str(result.get("trial_id") or ""),
        str(result.get("config_hash") or ""),
        wl_hash,
        format_slo(slo),
        metric(result.get("repeats")),
        "1" if "contention" in result else "0",
        metric(result.get("goodput_req_s")),
        metric((result.get("stats") or {}).get("goodput_std")),
        metric(result.get("output_tok_s")),
        metric(result.get("ttft_p99_ms")),
        metric(result.get("tpot_p99_ms")),
        metric(result.get("peak_vram_gb")),
        metric(result.get("dur_s")),
        status,
        desc,
    ]


def check_existing_header(path: str) -> None:
    """已有 results.tsv 的表头必须与 HEADER 完全一致；缺失文件/空文件 → 将写表头。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f.read().splitlines() if l.strip() != ""]
    if not lines:
        return
    actual = lines[0].split("\t")
    if actual != HEADER:
        _err(f"{path} 表头与当前 {len(HEADER)} 列不一致（实际 {len(actual)} 列）。"
             f"请迁移表头为: {chr(9).join(HEADER)}（可用 scripts/record.py 重建账本）")


def check_identity(result: dict) -> None:
    """身份列不可静默为空（F6）：ok 行必须有 trial_id 与 config_hash；
    crash 行两者皆空也报错（无法入账）。"""
    tid, ch = result.get("trial_id"), result.get("config_hash")
    if result.get("status") == "ok":
        if not tid:
            _err("result.json 缺 trial_id（ok 行必须带身份列；请确认 bench.py 已写入身份）")
        if not ch:
            _err("result.json 缺 config_hash（ok 行必须带身份列；请确认 bench.py 已写入指纹）")
    elif not tid and not ch:
        _err("result.json 既无 trial_id 也无 config_hash（无法入账；请确认 bench.py 已写入身份）")


def check_desc(desc: str) -> None:
    """--desc 必须是单行（F2）：含 tab/换行会写坏 tab 分隔账本。"""
    if any(c in desc for c in ("\t", "\n", "\r")):
        _err("--desc 不得包含 tab / 换行（账本为单行 tab 分隔）；请改用单行描述")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="从 result.json 追加一行 results.tsv（机械记账，防手写错列）")
    ap.add_argument("--status", required=True,
                    choices=["keep", "discard", "crash", "best", "confirm"])
    ap.add_argument("--desc", required=True, help="本行说明（自由文本，便于复盘）")
    ap.add_argument("--commit", default=None, help="config.yaml 的 git 短 hash（缺省取 HEAD）")
    ap.add_argument("--results", default="results.tsv")
    ap.add_argument("--result-json", default="result.json")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--harness", default="harness.yaml")
    ap.add_argument("--target", default="target.yaml")
    args = ap.parse_args(argv)

    result = load_result(args.result_json)
    check_identity(result)
    check_desc(args.desc)
    commit = args.commit if args.commit is not None else current_commit()
    wl_hash = workload_hash(result)
    slo = resolve_slo_effective(result, args.harness, args.target, args.config)
    row = build_row(result, commit=commit, desc=args.desc, status=args.status,
                    slo=slo, wl_hash=wl_hash)

    check_existing_header(args.results)
    with open(args.results, "a", encoding="utf-8") as f:
        if os.path.getsize(args.results) == 0:
            f.write("\t".join(HEADER) + "\n")   # 首次创建：先写表头
        f.write("\t".join(row) + "\n")

    print("\t".join(row))   # 透明：把写入的行原样回显
    return 0


if __name__ == "__main__":
    sys.exit(main())
