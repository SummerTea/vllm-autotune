"""test_mock_loop.py — program.md 循环规则（mock_loop.py 可执行规格）的单元测试。

覆盖：keep/discard 噪声带判定、平台期退出、crash 不计入平台期、崩溃连击、
deadline 感知退出。全部离线。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scripts.mock_loop as ml  # noqa: E402


class TestDecide:
    def test_keep_when_beyond_noise_band(self):
        assert ml.decide(10.5, 10.0, noise_band_pct=3.0) == "keep"   # +5% > 3%

    def test_discard_within_noise_band(self):
        assert ml.decide(10.2, 10.0, noise_band_pct=3.0) == "discard"  # +2% ≤ 3%

    def test_discard_on_worse(self):
        assert ml.decide(9.5, 10.0, noise_band_pct=3.0) == "discard"

    def test_discard_when_slo_not_met(self):
        assert ml.decide(12.0, 10.0, noise_band_pct=3.0, slo_ok=False) == "discard"

    def test_crash_on_none_goodput(self):
        assert ml.decide(None, 10.0, noise_band_pct=3.0) == "crash"

    def test_first_trial_becomes_best(self):
        assert ml.decide(10.0, None, noise_band_pct=3.0) == "keep"

    def test_sigma_decides_when_larger_than_percent_band(self):
        # n=3 时 σ 项 = k×sqrt(σ²/3)：σ=0.25 → 2×0.1443=0.2887 < 百分比带 0.3 → 百分比主导
        assert ml.decide(10.4, 10.0, sigma=0.25, n_cand=3, noise_band_pct=3.0) == "keep"
        # σ=0.5 → 2×0.2887=0.5774 > 0.3 → σ 主导：Δ=0.4 不够、Δ=0.7 才够
        assert ml.decide(10.4, 10.0, sigma=0.5, n_cand=3, noise_band_pct=3.0) == "discard"
        assert ml.decide(10.7, 10.0, sigma=0.5, n_cand=3, noise_band_pct=3.0) == "keep"
        # n<2 → σ 项贡献 0 → 纯百分比带（退化不变）
        assert ml.decide(10.4, 10.0, sigma=0.5, n_cand=1, noise_band_pct=3.0) == "keep"

    def test_combined_se_uses_both_uncertainties(self):
        # 合并标准误：k×sqrt(σ_cand²/n_cand + σ_best²/n_best)。
        # σ=0.25(n=3) 双侧 → sqrt(0.0625/3+0.0625/3)=0.2041，k×=0.4082：Δ=0.4 不够、Δ=0.5 才够
        assert ml.decide(10.4, 10.0, sigma=0.25, best_sigma=0.25, n_cand=3, n_best=3,
                         noise_band_pct=3.0) == "discard"
        assert ml.decide(10.5, 10.0, sigma=0.25, best_sigma=0.25, n_cand=3, n_best=3,
                         noise_band_pct=3.0) == "keep"
        # 同 σ 更大 n → SE 更小 → keep；更小 n → SE 更大 → discard
        assert ml.decide(10.4, 10.0, sigma=0.5, n_cand=8, noise_band_pct=3.0) == "keep"
        assert ml.decide(10.4, 10.0, sigma=0.5, n_cand=3, noise_band_pct=3.0) == "discard"
        # best_sigma=0 → 退化为单侧 SE（不回归）
        assert ml.decide(10.5, 10.0, sigma=0.25, n_cand=3, noise_band_pct=3.0) == "keep"
        # 双侧 σ=0 → 纯百分比带
        assert ml.decide(10.4, 10.0, sigma=0.0, n_cand=3, best_sigma=0.0, n_best=3,
                         noise_band_pct=3.0) == "keep"

    def test_loop_state_tracks_best_sigma_and_n(self):
        # keep 后 best_sigma / best_n 记录该行 σ 与 repeats，并传入后续 decide
        st = ml.LoopState(plateau_trials=8, crash_streak=3)
        st.step(10.0, True, sigma=0.5, n=3, commit="a")       # keep（基线）→ best_sigma=0.5, best_n=3
        assert st.best_sigma == 0.5 and st.best_n == 3
        # 候选 σ=0.5(n=3) vs best σ=0.5(n=3)：k×sqrt(0.25/3+0.25/3)=0.8165 → Δ=0.4 不够
        assert st.step(10.4, True, sigma=0.5, n=3, commit="b") == "discard"
        assert st.best_sigma == 0.5                            # discard 不改 best
        # 超过合并 SE → keep，并更新 best_n
        assert st.step(10.9, True, sigma=0.5, n=3, commit="c") == "keep"
        assert st.best_sigma == 0.5 and st.best_n == 3


class TestLoopState:
    def test_keep_updates_best_and_resets_counters(self):
        st = ml.LoopState(plateau_trials=8, crash_streak=3)
        st.step(10.0, True, commit="aaa")      # keep（基线）
        st.step(10.2, True, commit="bbb")      # discard → plateau=1
        st.step(10.1, True, commit="ccc")      # discard → plateau=2
        st.step(11.5, True, commit="ddd")      # keep → 重置
        assert st.best_goodput == 11.5 and st.best_commit == "ddd"
        assert st.plateau_count == 0 and st.crash_count == 0

    def test_plateau_exit_after_n_consecutive(self):
        st = ml.LoopState(plateau_trials=3, crash_streak=3)
        st.step(10.0, True, commit="a")        # keep（基线）
        for _ in range(2):
            assert ml.should_exit(st) is None  # 尚未满 3 次
            st.step(10.0, True, commit="x")    # discard
        st.step(10.0, True, commit="y")        # 第 3 次 discard
        assert ml.should_exit(st) == "plateau"

    def test_crash_does_not_advance_plateau(self):
        st = ml.LoopState(plateau_trials=3, crash_streak=3)
        st.step(10.0, True, commit="a")        # keep（基线）
        st.step(10.0, True, commit="b")        # discard → plateau=1
        st.step(None, True, commit="c")        # crash：plateau 不前进
        assert st.plateau_count == 1
        assert st.crash_count == 1
        st.step(10.0, True, commit="d")        # discard → plateau=2
        assert st.plateau_count == 2

    def test_crash_streak_exit(self):
        st = ml.LoopState(plateau_trials=8, crash_streak=3)
        st.step(10.0, True, commit="a")        # keep（基线）
        st.step(None, True, commit="b")
        st.step(None, True, commit="c")
        assert ml.should_exit(st) is None      # 2 次未满
        st.step(None, True, commit="d")        # 第 3 次 crash
        assert ml.should_exit(st) == "crash_streak"

    def test_non_crash_resets_crash_streak(self):
        st = ml.LoopState(plateau_trials=8, crash_streak=3)
        st.step(None, True, commit="a")        # crash ×1
        st.step(None, True, commit="b")        # crash ×2
        st.step(10.0, True, commit="c")        # discard → 清零
        assert st.crash_count == 0
        assert ml.should_exit(st) is None


class TestShouldExit:
    def test_budget_deadline_aware(self):
        st = ml.LoopState()
        assert ml.should_exit(st, budget_remaining_s=300, median_trial_s=600) == "budget"

    def test_budget_exact_equal_continues(self):
        st = ml.LoopState()
        assert ml.should_exit(st, budget_remaining_s=600, median_trial_s=600) is None

    def test_budget_none_skips_deadline_check(self):
        st = ml.LoopState(plateau_trials=8, crash_streak=3)
        st.step(None, True, commit="a")        # crash ×1（未达连击）
        assert ml.should_exit(st, budget_remaining_s=None, median_trial_s=None) is None

    def test_priority_budget_over_plateau(self):
        st = ml.LoopState(plateau_trials=1, crash_streak=3)
        st.step(10.0, True, commit="a")        # keep
        st.step(10.0, True, commit="b")        # discard → plateau=1（满足平台期）
        assert ml.should_exit(st, budget_remaining_s=300, median_trial_s=600) == "budget"
