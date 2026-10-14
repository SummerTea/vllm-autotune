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
        # 提升 +4%（> 3% 百分比带）但 σ 大：k×σ = 2×0.25 = 0.5（abs）> best×4% = 0.4 → σ 主导
        assert ml.decide(10.4, 10.0, sigma=0.25, noise_band_pct=3.0) == "discard"
        # 同 σ 下超过 k×σ（Δ=0.6 > 0.5）才 keep
        assert ml.decide(10.6, 10.0, sigma=0.25, noise_band_pct=3.0) == "keep"
        # 百分比带主导时（σ=0 → 纯百分比）行为不变
        assert ml.decide(10.4, 10.0, sigma=0.0, noise_band_pct=3.0) == "keep"


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
