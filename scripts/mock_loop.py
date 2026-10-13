#!/usr/bin/env python3
"""mock_loop.py — program.md 循环规则的可执行规格 / 演示（REFERENCE，非运行时组件）。

⚠ 本脚本不是调优循环的运行时实现：真实循环由编码 agent 按 program.md 驱动
（改 config.yaml → commit → run.sh 启动/轮询 → 记录 results.tsv → keep/discard）。
本脚本只是把「决策规则 + 三层退出 + deadline 感知」翻译成纯函数，并跑一段固定剧本
离线核对行为（无 GPU / 无网络）。阈值优先取自仓库根 harness.yaml，缺失时回退内置默认。
"""

from __future__ import annotations

from pathlib import Path

NOISE_BAND_PCT = 3.0   # 内置默认（与 harness.yaml 当前值一致）
PLATEAU_TRIALS = 8     # harness.yaml exit.plateau_trials
CRASH_STREAK = 3       # harness.yaml exit.crash_streak

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def _load_thresholds():
    """阈值优先取自仓库根 harness.yaml（bench.noise_band_pct / exit.plateau_trials /
    exit.crash_streak），缺失/不可读时回退内置默认——脚本独立可跑，阈值不会静默漂移。"""
    repo_root = Path(__file__).resolve().parent.parent
    try:
        if yaml is not None:
            with open(repo_root / "harness.yaml", encoding="utf-8") as f:
                h = yaml.safe_load(f) or {}
            b = h.get("bench") or {}
            e = h.get("exit") or {}
            return (float(b.get("noise_band_pct", NOISE_BAND_PCT)),
                    int(e.get("plateau_trials", PLATEAU_TRIALS)),
                    int(e.get("crash_streak", CRASH_STREAK)))
    except (OSError, ValueError, TypeError):
        pass
    return NOISE_BAND_PCT, PLATEAU_TRIALS, CRASH_STREAK


# 阈值来自 harness.yaml（可用时）；否则用内置默认
NOISE_BAND_PCT, PLATEAU_TRIALS, CRASH_STREAK = _load_thresholds()


def decide(trial_goodput, best_goodput, noise_band_pct=NOISE_BAND_PCT, slo_ok=True):
    """决策规则（program.md §决策规则）：
    - keep   ：goodput 相比当前最佳提升 > 噪声带，且 SLO 达标；
    - discard：提升 ≤ 噪声带（含变差）或 SLO 不达标；
    - crash  ：启动失败 / OOM / 超时（调用方传 trial_goodput=None）。
    首个有效 trial（best 尚为 None）即成为当前最佳。"""
    if trial_goodput is None:
        return "crash"
    if not slo_ok:
        return "discard"
    if best_goodput is None:
        return "keep" if trial_goodput > 0 else "discard"
    if trial_goodput > best_goodput * (1 + noise_band_pct / 100.0):
        return "keep"
    return "discard"


class LoopState:
    """小型状态机：plateau_count / crash_count / best_goodput / best_commit。

    - crash 行**不计入**平台期序列（plateau_count 不前进；只计入崩溃连击）；
    - keep 清零崩溃连击并重置平台期；discard 清零崩溃连击、推进平台期。"""

    def __init__(self, plateau_trials=PLATEAU_TRIALS, crash_streak=CRASH_STREAK):
        self.plateau_trials = plateau_trials
        self.crash_streak = crash_streak
        self.plateau_count = 0
        self.crash_count = 0
        self.best_goodput = None
        self.best_commit = None

    def step(self, trial_goodput, slo_ok, noise_band_pct=NOISE_BAND_PCT, commit=None):
        d = decide(trial_goodput, self.best_goodput, noise_band_pct, slo_ok)
        if d == "crash":
            self.crash_count += 1            # plateau_count 不前进（program.md §平台期）
        elif d == "keep":
            self.crash_count = 0
            self.plateau_count = 0
            self.best_goodput = trial_goodput
            self.best_commit = commit
        else:  # discard
            self.crash_count = 0
            self.plateau_count += 1
        return d


def should_exit(state, budget_remaining_s=None, median_trial_s=None):
    """三层退出（program.md §退出机制），返回退出原因或 None（继续）：
    - deadline 感知（主退出）：剩余预算 < 预估单 trial 耗时（历史 dur_s 中位数）→ 'budget'；
    - 平台期：连续 plateau_trials 次无超过噪声带的改进 → 'plateau'；
    - 崩溃连击：连续 crash_streak 次 crash → 'crash_streak'。
    预算缺省（None）时跳过 deadline 检查。"""
    if (budget_remaining_s is not None and median_trial_s is not None
            and budget_remaining_s < median_trial_s):
        return "budget"
    if state.plateau_count >= state.plateau_trials:
        return "plateau"
    if state.crash_count >= state.crash_streak:
        return "crash_streak"
    return None


# ── 演示剧本（固定序列，无随机）─────────────────────────────────────────

_SCENARIO = [
    # (commit, goodput, slo_ok)
    ("a1b2c3", 10.0, True),   # 基线：首个有效 trial → keep
    ("d4e5f6", 10.2, True),   # +2.0% ≤ 3.0% → discard
    ("a7b8c9", 10.9, True),   # +9.0% > 3.0% → keep
    ("d0e1f2", 10.9, True),   # 持平 → discard
    ("a3b4c5", 11.0, True),   # +0.9% → discard
    ("d6e7f8", 10.8, True),   # 变差 → discard
    ("a9b0c1", 11.0, True),   # +0.9% → discard
    ("d2e3f4", 11.1, True),   # +1.8% → discard
    ("a5b6c7", None, True),   # crash（不计入平台期）
    ("d8e9f0", 11.1, True),   # 持平 → discard
    ("a1b2d3", 11.1, True),   # 持平 → discard
    ("c4d5e6", 11.2, True),   # +2.7% → discard → 平台期计数到 8
]


def _pct_str(g, best):
    if g is None or best is None:
        return "—"
    return f"{ (g - best) / best * 100:+.1f}%"


def main() -> int:
    print("== mock_loop.py：program.md 循环规则的可执行规格（REFERENCE）==")
    print("   真实循环由编码 agent 驱动；此处仅离线核对规则行为。")
    print(f"   阈值：noise_band_pct={NOISE_BAND_PCT} plateau_trials={PLATEAU_TRIALS} "
          f"crash_streak={CRASH_STREAK}（取自仓库根 harness.yaml，缺失时回退内置默认）\n")

    state = LoopState()
    for i, (commit, gp, slo_ok) in enumerate(_SCENARIO, 1):
        d = state.step(gp, slo_ok, commit=commit)
        print(f"[#{i:02d}] commit={commit} goodput={gp} slo_ok={slo_ok} "
              f"→ {d}（best={state.best_goodput}，相对 best {_pct_str(gp, state.best_goodput)}，"
              f"plateau={state.plateau_count}/{state.plateau_trials}，"
              f"crash_streak={state.crash_count}/{state.crash_streak}）")

    print("\n== 退出判定 ==")
    r1 = should_exit(state, budget_remaining_s=3600, median_trial_s=600)
    print(f"预算 3600s / 中位 600s → {r1 or '继续'}")
    r2 = should_exit(state, budget_remaining_s=300, median_trial_s=600)
    print(f"预算 300s / 中位 600s → {r2 or '继续'}（deadline 感知：来不及开新 trial）")
    r3 = should_exit(state, budget_remaining_s=None, median_trial_s=None)
    print(f"无预算约束 → {r3 or '继续'}")
    print(f"\n本剧本最终退出层：{r3 or r2}（演示用；真实运行以 target.yaml 预算与实测 dur_s 为准）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
