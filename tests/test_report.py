"""test_report.py — report.py 纯函数单元测试 + 渲染冒烟。

口径与 report.py 内实现一致；全部在无 GPU/网络的本机运行。
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import report  # noqa: E402

SLO = {"ttft_p99_ms": 500, "tpot_p99_ms": 50}

TSV = (
    "commit\tgoodput\tgoodput_std\toutput_tok_s\tttft_p99_ms\ttpot_p99_ms\tpeak_vram_gb\tdur_s\tstatus\tdescription\n"
    "aaaa111\t10.0\t0.2\t9000.0\t420.1\t18.3\t71.2\t912.4\tkeep\t基线（mock）\n"
    "bbbb222\t10.2\t0.3\t9200.0\t430.2\t19.1\t71.5\t910.0\tdiscard\tmax-num-seqs 256\n"
    "cccc333\t11.0\t0.5\t9800.0\t450.0\t20.0\t72.0\t905.0\tkeep\tgpu-memory-utilization 0.92\n"
    "dddd444\t\t\t\t\t\t\t\tcrash\tOOM\n"
)


def _rows():
    return report.parse_tsv(TSV)


class TestParseTsv:
    def test_parses_header_and_rows(self):
        rows = _rows()
        assert len(rows) == 4
        assert rows[0]["commit"] == "aaaa111"
        assert rows[0]["status"] == "keep"
        assert rows[0]["description"] == "基线（mock）"

    def test_numeric_coercion(self):
        rows = _rows()
        assert rows[0]["goodput"] == 10.0
        assert rows[0]["goodput_std"] == 0.2          # 新列 numeric
        assert rows[0]["dur_s"] == 912.4
        assert rows[1]["goodput"] == 10.2

    def test_crash_row_empties_become_none(self):
        crash = _rows()[3]
        assert crash["status"] == "crash"
        assert crash["goodput"] is None
        assert crash["ttft_p99_ms"] is None
        assert crash["dur_s"] is None

    def test_bad_header_rejected(self):
        with pytest.raises(SystemExit):
            report.parse_tsv("commit\tgoodput\tstatus\nx\t1\ty\n")


class TestSloOk:
    def test_within_slo(self):
        assert report.slo_ok(_rows()[0], SLO) is True

    def test_over_ttft(self):
        row = dict(_rows()[0], ttft_p99_ms=999.0)
        assert report.slo_ok(row, SLO) is False

    def test_over_tpot(self):
        row = dict(_rows()[0], tpot_p99_ms=99.0)
        assert report.slo_ok(row, SLO) is False

    def test_missing_latency_is_not_ok(self):
        assert report.slo_ok(_rows()[3], SLO) is False  # crash 行无时延

    def test_default_slo_when_none(self):
        assert report.slo_ok(_rows()[0], None) is True  # 缺省 500/50


class TestPickBaseline:
    def test_first_data_row(self):
        b = report.pick_baseline(_rows())
        assert b is not None
        assert b["commit"] == "aaaa111"

    def test_empty_rows_none(self):
        assert report.pick_baseline([]) is None


class TestPickBest:
    def test_picks_highest_slo_met_keep_row(self):
        # cccc333 goodput=11.0 SLO 达标且为 keep → 最佳
        best, flagged = report.pick_best(_rows(), SLO)
        assert best is not None
        assert best["commit"] == "cccc333"
        assert flagged is False

    def test_skips_non_slo_rows(self):
        # 构造：goodput=20 但 ttft 超 SLO 的 keep 行，不应胜过 SLO 达标的 11.0
        rows = _rows() + [dict(_rows()[2], commit="zzzz999", goodput=20.0, ttft_p99_ms=900.0)]
        best, _ = report.pick_best(rows, SLO)
        assert best is not None
        assert best["commit"] == "cccc333"

    def test_fallback_when_none_meet_slo(self):
        # 全部 keep/best 行超 SLO → 回退全部已测量行的最高 goodput，并标记 flagged
        rows = [dict(_rows()[0], ttft_p99_ms=999.0, goodput=10.0),
                dict(_rows()[2], ttft_p99_ms=999.0, goodput=11.0),
                _rows()[3]]  # crash 行不参与
        best, flagged = report.pick_best(rows, SLO)
        assert best is not None
        assert best["commit"] == "cccc333"
        assert flagged is True

    def test_crash_rows_never_selected(self):
        rows = _rows()
        best, _ = report.pick_best(rows, SLO)
        assert best is not None
        assert best["status"] != "crash"

    def test_fallback_prefers_keep_over_higher_discard(self):
        # 无 SLO 达标行时：{keep,best} 行优先，即使 discard 行的 goodput 更高
        rows = [dict(_rows()[2], ttft_p99_ms=999.0, goodput=11.0),  # keep 但超 SLO
                dict(_rows()[1], goodput=99.0)]                     # discard 更高 goodput
        best, flagged = report.pick_best(rows, SLO)
        assert best is not None
        assert best["status"] == "keep"          # keep/best 行胜出
        assert best["goodput"] == 11.0
        assert flagged is True

    def test_no_measured_rows(self):
        rows = [_rows()[3]]  # 只有 crash 行
        best, flagged = report.pick_best(rows, SLO)
        assert best is None
        assert flagged is True


class TestImprovementPct:
    def test_positive(self):
        base = {"goodput": 10.0}
        best = {"goodput": 12.5}
        assert report.improvement_pct(base, best) == pytest.approx(25.0)

    def test_zero_base_none(self):
        assert report.improvement_pct({"goodput": 0.0}, {"goodput": 1.0}) is None

    def test_missing_goodput_none(self):
        assert report.improvement_pct({"goodput": None}, {"goodput": 1.0}) is None
        assert report.improvement_pct(None, {"goodput": 1.0}) is None


class TestExtractSystemAdvice:
    def test_extracts_warn_rows_only(self):
        md = (
            "# 环境探测报告（env_report.md）\n"
            "| 检查项 | 状态 | 详情 | 修复命令 |\n"
            "|---|---|---|---|\n"
            "| CPU 频率调节器 | pass | governor=performance | — |\n"
            "| GPU SM 时钟 | warn | GPU0: 810/1410 MHz | 无（负载时复查） |\n"
            "| 磁盘剩余空间 | fail | 可用 10 GB | 扩容后重跑 |\n"
        )
        advice = report.extract_system_advice(md)
        assert len(advice) == 2
        assert advice[0]["name"] == "GPU SM 时钟" and advice[0]["status"] == "warn"
        assert advice[1]["name"] == "磁盘剩余空间" and advice[1]["status"] == "fail"

    def test_none_input(self):
        assert report.extract_system_advice(None) == []


class TestRender:
    def test_smoke_sections(self):
        rows = _rows()
        baseline = report.pick_baseline(rows)
        best, flagged = report.pick_best(rows, SLO)
        md = report.render(rows, SLO, baseline, best, flagged,
                           report.improvement_pct(baseline, best),
                           model="Qwen/Qwen3-32B",
                           workload={"dataset": "random", "isl": 2048, "osl": 256,
                                     "num_prompts": 1000},
                           config_snapshot="serve:\n  max-num-seqs: 256\n",
                           system_advice=[{"name": "GPU SM 时钟", "status": "warn",
                                           "detail": "空闲偏低", "fix": "负载时复查"}])
        for sec in ("# 调优报告", "## 概览", "## 基线 vs 最佳", "## 全部 trial 表",
                    "## 最佳配置", "## 系统级建议", "## 适用范围"):
            assert sec in md
        assert "goodput 提升" in md      # 提升行
        assert "commit=cccc333" in md    # 最佳 commit
        assert "max-num-seqs: 256" in md   # 配置快照内嵌
        assert "GPU SM 时钟" in md   # 系统级建议

    def test_render_fallback_note(self):
        rows = [dict(_rows()[0], ttft_p99_ms=999.0),
                dict(_rows()[2], ttft_p99_ms=999.0)]
        baseline = report.pick_baseline(rows)
        best, flagged = report.pick_best(rows, SLO)
        md = report.render(rows, SLO, baseline, best, flagged,
                           report.improvement_pct(baseline, best))
        assert "无满足 SLO 的 trial" in md

    def test_render_empty_rows(self):
        md = report.render([], SLO, None, None, True, None)
        assert "无任何可比较的 trial" in md
