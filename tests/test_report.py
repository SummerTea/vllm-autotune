"""test_report.py — report.py 纯函数单元测试 + 渲染冒烟。

口径与 report.py 内实现一致；全部在无 GPU/网络的本机运行。
"""

import os
import sys
from pathlib import Path

import pytest
import yaml

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


NEW_TSV = (
    "commit\ttrial_id\tconfig_hash\tworkload_hash\tslo\trepeats\tcontention\tgoodput\tgoodput_std\toutput_tok_s\tttft_p99_ms\ttpot_p99_ms\tpeak_vram_gb\tdur_s\tstatus\tdescription\n"
    "aaaa111\tt1\th1\tw1\tttft500/tpot50\t3\t0\t10.0\t0.2\t9000.0\t420.1\t18.3\t71.2\t912.4\tkeep\t基线（mock）\n"
    "bbbb222\tt2\th2\tw1\tttft500/tpot50\t3\t0\t10.2\t0.3\t9200.0\t430.2\t19.1\t71.5\t910.0\tdiscard\tmax-num-seqs 256\n"
    "cccc333\tt3\th3\tw1\tttft500/tpot50\t3\t1\t11.0\t0.5\t9800.0\t450.0\t20.0\t72.0\t905.0\tkeep\tgpu-memory-utilization 0.92\n"
)


class TestParseTsvNewColumns:
    def test_new_16col_header_parses(self):
        rows = report.parse_tsv(NEW_TSV)
        assert len(rows) == 3
        r0 = rows[0]
        assert r0["trial_id"] == "t1" and r0["config_hash"] == "h1"
        assert r0["workload_hash"] == "w1" and r0["slo"] == "ttft500/tpot50"
        assert r0["repeats"] == 3.0                       # 新列数值 coerce
        assert r0["contention"] == "0"                    # contention 保持字符串
        assert rows[2]["contention"] == "1"
        assert r0["goodput"] == 10.0                      # 核心列照常

    def test_column_order_does_not_matter(self):
        # 名字对齐：核心列齐全但列序打乱也能解析出相同字段（扩展列任意位置）
        shuffled = ("commit\tdescription\tgoodput\tstatus\tgoodput_std\tttft_p99_ms"
                    "\ttpot_p99_ms\toutput_tok_s\tpeak_vram_gb\tdur_s\ttrial_id\tcontention\n"
                    "x\t说明\t1.0\tkeep\t0.1\t420.0\t18.0\t9000.0\t71.0\t900.0\tt9\t1\n")
        rows = report.parse_tsv(shuffled)
        assert rows[0]["goodput"] == 1.0
        assert rows[0]["trial_id"] == "t9"
        assert rows[0]["contention"] == "1"
        assert rows[0]["description"] == "说明"
        assert not rows[0]["workload_hash"]           # 扩展列缺失 → 空（未知上下文）
        assert rows[0]["repeats"] is None             # 数值扩展列缺失 → None


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
        assert "跨 workload 的行不可混比" in md   # 概览 workload 来源备注

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


# ── 部署产物（best 配置 → deploy-best.sh）──────────────────────────────

def _write_mini(tmp: Path, results: str, *, harness: dict | None = None,
                target: dict | None = None) -> None:
    """最小 target/harness/results（deploy 测试用）。"""
    (tmp / "results.tsv").write_text(results, encoding="utf-8")
    (tmp / "harness.yaml").write_text(yaml.safe_dump(harness or {
        "slo": {"ttft_p99_ms": 500, "tpot_p99_ms": 50},
        "workload": {"dataset": "random", "isl": 2048, "osl": 256, "num_prompts": 1000},
        "remote": {"vllm_port": 8000},
        "runtime": {"image": "vllm/vllm-openai:v0.31.0"},
    }), encoding="utf-8")
    (tmp / "target.yaml").write_text(yaml.safe_dump(target or {
        "hosts": [{"name": "head", "ssh": "u@h", "role": "head"}],
        "model": {"id": "Qwen/Qwen3-32B", "source": "local", "path": "/models/qwen3-32b"},
    }), encoding="utf-8")


def _run_main(tmp: Path) -> int:
    return report.main(["--results", str(tmp / "results.tsv"),
                        "--harness", str(tmp / "harness.yaml"),
                        "--target", str(tmp / "target.yaml"),
                        "--config", str(tmp / "config.yaml"),
                        "--out", str(tmp / "report.md")])


class TestDeploy:
    def test_deploy_artifact_written(self, tmp_path, monkeypatch):
        monkeypatch.setattr(report, "git_show_config", lambda commit: (
            "serve:\n  max-num-seqs: 256\n  enable-prefix-caching: true\n"))
        _write_mini(tmp_path, TSV)          # best = cccc333（keep 且 SLO 达标）
        assert _run_main(tmp_path) == 0

        script = tmp_path / "deploy-best.sh"
        assert script.exists()
        text = script.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash")
        assert "set -euo pipefail" in text
        assert "vllm/vllm-openai:v0.31.0" in text        # 镜像
        assert "--entrypoint vllm" in text
        assert "serve" in text
        assert "--max-num-seqs 256" in text              # best 配置的 serve flag
        assert "--enable-prefix-caching" in text
        assert "/models/qwen3-32b:/model:ro" in text     # 模型只读挂载
        assert "exec docker run" in text
        assert os.access(script, os.X_OK)                # 可执行
        assert "cccc333" in text                          # 头部最佳 commit
        # M1：部署容器名独立于 trial——exec 命令行用 vllm-deploy-best，绝不带 trial 名
        exec_line = next(l for l in text.splitlines() if l.startswith("exec docker"))
        assert "vllm-deploy-best" in exec_line
        assert "vllm-autotune-trial" not in exec_line
        assert "vllm-deploy-best" in text                 # 头部注释说明独立容器名

        md = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "## 部署（best 配置）" in md
        assert "bash deploy-best.sh" in md
        assert "http://127.0.0.1:8000" in md

    def test_build_deploy_cmd_reuses_bench_builder(self):
        # 直接单测构建器复用（bench.build_docker_run_cmd + effective_runtime）
        config_text = "serve:\n  max-num-seqs: 256\n"
        harness = {"remote": {"vllm_port": 8123}, "runtime": {"image": "img:1"}}
        target = {"model": {"path": "/m"}}
        cmd = report.build_deploy_cmd(config_text, target, harness,
                                      {"commit": "x"}, {"ttft_p99_ms": 500, "tpot_p99_ms": 50})
        assert cmd is not None
        assert "img:1" in cmd
        assert "--max-num-seqs 256" in " ".join(cmd)
        assert "-p 127.0.0.1:8123:8123" in " ".join(cmd)
        # 降级：缺 model.path → None（不 crash）
        assert report.build_deploy_cmd(config_text, {}, harness, {}, {}) is None

    def test_deploy_skipped_when_git_show_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(report, "git_show_config", lambda commit: None)
        _write_mini(tmp_path, TSV)
        assert _run_main(tmp_path) == 0
        assert not (tmp_path / "deploy-best.sh").exists()
        md = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "未生成部署命令" in md

    def test_deploy_skipped_when_all_crash(self, tmp_path, monkeypatch):
        crash_tsv = ("commit\tgoodput\tgoodput_std\toutput_tok_s\tttft_p99_ms\ttpot_p99_ms"
                     "\tpeak_vram_gb\tdur_s\tstatus\tdescription\n"
                     "d111\t\t\t\t\t\t\t\tcrash\tOOM\n")
        _write_mini(tmp_path, crash_tsv)
        assert _run_main(tmp_path) == 0
        assert not (tmp_path / "deploy-best.sh").exists()
        assert "未生成部署命令" in (tmp_path / "report.md").read_text(encoding="utf-8")

    def test_deploy_port_consistent_when_config_sets_port(self, tmp_path, monkeypatch):
        # minor：best 配置 serve.port=9999 → -p 发布端口与容器内 --port 一致（报告 URL 同端口）
        monkeypatch.setattr(report, "git_show_config", lambda c: (
            "serve:\n  port: 9999\n  max-num-seqs: 256\n"))
        _write_mini(tmp_path, TSV)          # harness remote.vllm_port=8000（应被 9999 覆盖）
        assert _run_main(tmp_path) == 0
        text = (tmp_path / "deploy-best.sh").read_text(encoding="utf-8")
        assert "-p 127.0.0.1:9999:9999" in text
        assert "--port 9999" in text
        assert "-p 127.0.0.1:8000" not in text
        md = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "http://127.0.0.1:9999" in md


# ── 运行期 SLO 校准（config.slo = 有效 SLO 通道，与 bench 同优先级）─────────

class TestSloCalibration:
    def test_calibration_marker_when_config_slo_differs(self, tmp_path, monkeypatch):
        # config.slo（有效 SLO）覆盖 harness.slo；报告出「运行期 SLO 校准」标注，
        # 部署产物 SLO 头也用有效 SLO（report 与 bench 共用 resolve_slo → 口径一致）
        monkeypatch.setattr(report, "git_show_config", lambda c: (
            "serve:\n  max-num-seqs: 256\n"))
        _write_mini(tmp_path, TSV)          # harness.slo=500/50，target 无 metrics.slo
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(
            {"slo": {"ttft_p99_ms": 550, "tpot_p99_ms": 55}}), encoding="utf-8")
        assert _run_main(tmp_path) == 0
        md = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "运行期 SLO 校准" in md
        assert "500→550 ms" in md and "50→55 ms" in md
        text = (tmp_path / "deploy-best.sh").read_text(encoding="utf-8")
        assert "ttft≤550ms" in text          # 有效 SLO 生效

    def test_config_slo_beats_target_metrics(self, tmp_path, monkeypatch):
        # 优先级：config.slo > target.metrics.slo > harness.slo（逐键覆盖）
        monkeypatch.setattr(report, "git_show_config", lambda c: (
            "serve:\n  max-num-seqs: 256\n"))
        (tmp_path / "harness.yaml").write_text(yaml.safe_dump(
            {"slo": {"ttft_p99_ms": 500, "tpot_p99_ms": 50},
             "remote": {"vllm_port": 8000},
             "runtime": {"image": "img:1"},
             "workload": {"dataset": "random", "isl": 2048, "osl": 256, "num_prompts": 1000}}),
            encoding="utf-8")
        (tmp_path / "target.yaml").write_text(yaml.safe_dump(
            {"metrics": {"slo": {"ttft_p99_ms": 600, "tpot_p99_ms": 60}}}), encoding="utf-8")
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(
            {"slo": {"ttft_p99_ms": 700, "tpot_p99_ms": 70}}), encoding="utf-8")
        (tmp_path / "results.tsv").write_text(TSV, encoding="utf-8")
        assert _run_main(tmp_path) == 0
        md = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "600→700 ms" in md and "60→70 ms" in md     # 参考=target(600/60)，有效=config(700/70)
        assert "ttft_p99 ≤ 700 ms" in md

    def test_no_marker_when_config_has_no_slo(self, tmp_path, monkeypatch):
        # config 缺省（无 slo 键）→ 行为不变：无校准标注，按参考 SLO 判定
        monkeypatch.setattr(report, "git_show_config", lambda c: (
            "serve:\n  max-num-seqs: 256\n"))
        _write_mini(tmp_path, TSV)
        (tmp_path / "config.yaml").write_text(yaml.safe_dump({"serve": {}}), encoding="utf-8")
        assert _run_main(tmp_path) == 0
        md = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "运行期 SLO 校准" not in md
        assert "ttft_p99 ≤ 500 ms" in md      # 参考 SLO 500/50（harness）
        text = (tmp_path / "deploy-best.sh").read_text(encoding="utf-8")
        assert "ttft≤500ms" in text


# ── 确认（best 回归）────────────────────────────────────────────────────

CONFIRM_TSV = TSV + (
    "eeee555\t10.9\t0.4\t9700.0\t455.0\t20.5\t72.0\t900.0\tconfirm\tconfirm best\n")


class TestConfirm:
    def test_pick_best_ignores_confirm(self):
        # confirm 行绝不参与 best 选择：best 仍为 keep 的 cccc333
        rows = report.parse_tsv(CONFIRM_TSV)
        best, flagged = report.pick_best(rows, SLO)
        assert best is not None
        assert best["status"] == "keep" and best["commit"] == "cccc333"
        assert best["goodput"] == 11.0
        assert flagged is False

    def test_confirm_within_band(self):
        # confirm 10.9 vs best 11.0：Δ=0.1 ≤ max(11×3%, 2×0.5)=1.0 → 确认
        rows = report.parse_tsv(CONFIRM_TSV)
        best, _ = report.pick_best(rows, SLO)
        md = report.render(rows, SLO, None, best, False, None,
                           noise_band_pct=3.0, noise_k_sigma=2.0)
        assert "## 确认（best 回归）" in md
        assert "确认（噪声带内一致）" in md
        assert "confirm（最新）" in md

    def test_confirm_outside_band(self):
        # 构造 Δ=2.0 > 1.0 → 未确认
        tsv = CONFIRM_TSV.replace("10.9\t0.4", "9.0\t0.4")
        rows = report.parse_tsv(tsv)
        best, _ = report.pick_best(rows, SLO)
        md = report.render(rows, SLO, None, best, False, None,
                           noise_band_pct=3.0, noise_k_sigma=2.0)
        assert "未确认（差异" in md

    def test_confirm_missing_shows_note(self):
        rows = _rows()                     # 无 confirm 行
        best, _ = report.pick_best(rows, SLO)
        md = report.render(rows, SLO, None, best, False, None,
                           noise_band_pct=3.0, noise_k_sigma=2.0)
        assert "未做确认 run" in md


# ── Phase 2：可比性上下文 + 参考/有效 SLO 双口径 ─────────────────────────

MIXED_TSV = NEW_TSV + (
    "dddd444\tt4\th4\tw2\tttft4000/tpot80\t3\t0\t50.0\t1.0\t50000.0\t300.0\t20.0\t75.0\t900.0\tkeep\t另一上下文（w2/slo2）高 goodput\n"
)


class TestComparability:
    def test_context_key(self):
        assert report.context_key({"workload_hash": "w1", "slo": "ttft500/tpot50"}) == \
            ("w1", "ttft500/tpot50")
        # 两者皆空 → None（旧表/未知）
        assert report.context_key({"workload_hash": "", "slo": None}) is None
        assert report.context_key({}) is None
        # 其一存在 → 有上下文
        assert report.context_key({"workload_hash": "w1"}) == ("w1", "")

    def test_best_restricted_to_baseline_context(self):
        # 另一上下文（w2/slo2）的 50.0 不得参与 best 选择——best 仍是基线上下文内 11.0
        rows = report.parse_tsv(MIXED_TSV)
        baseline = report.pick_baseline(rows)
        ctx = report.context_key(baseline)
        assert ctx == ("w1", "ttft500/tpot50")
        best, flagged = report.pick_best(rows, SLO, ctx)
        assert best is not None
        assert best["commit"] == "cccc333" and best["goodput"] == 11.0
        assert flagged is False
        # 不传上下文（legacy）→ 全部行参与（旧行为），50.0 胜出
        best_legacy, _ = report.pick_best(rows, SLO)
        assert best_legacy["commit"] == "dddd444"

    def test_render_comparability_note(self):
        rows = report.parse_tsv(MIXED_TSV)
        baseline = report.pick_baseline(rows)
        ctx = report.context_key(baseline)
        best, flagged = report.pick_best(rows, SLO, ctx)
        comparability = sorted((k, n) for k, n in
                               report.Counter(report.context_key(r) for r in rows).items()
                               if k is not None)
        md = report.render(rows, SLO, baseline, best, flagged,
                           report.improvement_pct(baseline, best),
                           reference_slo=SLO, baseline_ctx=ctx, comparability=comparability)
        assert "## 可比性" in md
        assert "w2" in md and "ttft4000/tpot80" in md          # 逐项列出各上下文
        assert "仅基线上下文" in md                              # 只有基线上下文参与 best
        assert "参考 SLO 下的最佳" in md                          # 参考 SLO 可达 → 附对标行

    def test_f1_reachability_scoped_to_baseline_context(self):
        # F1：基线上下文（w1）无 keep/best 行满足参考 SLO，另一上下文（w2）有 → 不得报可达
        tsv = NEW_TSV.replace("420.1\t18.3", "620.1\t18.3") \
                     .replace("430.2\t19.1", "630.2\t19.1") \
                     .replace("450.0\t20.0", "650.0\t20.0") + (
            "eeee555\tt5\th5\tw2\tttft4000/tpot80\t3\t0\t5.0\t1.0\t1000.0\t300.0\t20.0\t75.0\t900.0\tkeep\t参考 SLO 达标但在另一上下文\n")
        rows = report.parse_tsv(tsv)
        baseline = report.pick_baseline(rows)
        ctx = report.context_key(baseline)
        assert ctx == ("w1", "ttft500/tpot50")
        best, _ = report.pick_best(rows, SLO, ctx)
        comparability = sorted((k, n) for k, n in
                               report.Counter(report.context_key(r) for r in rows).items()
                               if k is not None)
        md = report.render(rows, SLO, baseline, best, False,
                           report.improvement_pct(baseline, best),
                           reference_slo=SLO, baseline_ctx=ctx, comparability=comparability)
        assert "参考 SLO：**可达**" not in md                    # 另一上下文达标不算
        assert "参考 SLO：**不可达**" in md
        assert "参考 SLO 下无达标行" in md                        # 与可达性同口径，不矛盾

    def test_f1_discard_only_ref_met_not_reachable(self):
        # F1：仅 discard 行满足参考 SLO（keep/best 均不达标）→ 不可达（best 口径需 keep/best）
        tsv = NEW_TSV.replace("420.1\t18.3", "620.1\t18.3") \
                     .replace("450.0\t20.0", "650.0\t20.0")   # bbbb222(discard) 保持达标
        rows = report.parse_tsv(tsv)
        baseline = report.pick_baseline(rows)
        ctx = report.context_key(baseline)
        best, flagged = report.pick_best(rows, SLO, ctx)
        md = report.render(rows, SLO, baseline, best, flagged,
                           report.improvement_pct(baseline, best),
                           reference_slo=SLO, baseline_ctx=ctx)
        assert "参考 SLO：**不可达**" in md
        assert "参考 SLO 下的最佳" not in md                      # discard 不是候选 best
        assert "参考 SLO 下无达标行" in md
        # 存在非 keep/best 的已测量行达标 → 软化建议（复核配置），而非误导为「放宽 SLO」
        assert "人工复核该配置" in md
        assert "建议放宽至实测 p99" not in md

    def test_f4_legacy_baseline_one_context_warns(self, tmp_path, monkeypatch, capsys):
        # F4：legacy 基线（无上下文）+ 恰一个非空上下文 → 出可比性节 + stderr 警告
        monkeypatch.setattr(report, "git_show_config", lambda c: None)
        tsv = ("commit\ttrial_id\tconfig_hash\tworkload_hash\tslo\trepeats\tcontention"
               "\tgoodput\tgoodput_std\toutput_tok_s\tttft_p99_ms\ttpot_p99_ms"
               "\tpeak_vram_gb\tdur_s\tstatus\tdescription\n"
               "aaaa111\t\t\t\t\t\t\t10.0\t0.2\t9000.0\t420.1\t18.3\t71.2\t912.4\tkeep\t基线（无上下文）\n"
               "bbbb222\tt2\th2\tw1\tttft500/tpot50\t3\t0\t10.2\t0.3\t9200.0\t430.2\t19.1\t71.5\t910.0\tkeep\t有上下文\n")
        _write_mini(tmp_path, tsv)
        assert _run_main(tmp_path) == 0
        md = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "## 可比性" in md
        assert "best 选择按旧行为覆盖全部行" in md
        err = capsys.readouterr().err
        assert "legacy 行上下文未知" in err                        # main() 警告

    def test_f5_wording_no_candidates_vs_all_crash(self):
        # F5：行存在但基线上下文内无候选 → 明确文案；全 crash → 原文案
        tsv = ("commit\ttrial_id\tconfig_hash\tworkload_hash\tslo\trepeats\tcontention"
               "\tgoodput\tgoodput_std\toutput_tok_s\tttft_p99_ms\ttpot_p99_ms"
               "\tpeak_vram_gb\tdur_s\tstatus\tdescription\n"
               "a1\tt\th\tw1\tttft500/tpot50\t3\t0\t\t\t\t\t\t\t\tcrash\tOOM in baseline ctx\n"
               "b2\tt\th\tw2\tttft4000/tpot80\t3\t0\t50.0\t1.0\t50000.0\t300.0\t20.0\t75.0\t900.0\tkeep\t另一上下文\n")
        rows = report.parse_tsv(tsv)
        baseline = report.pick_baseline(rows)
        ctx = report.context_key(baseline)
        best, flagged = report.pick_best(rows, SLO, ctx)
        assert best is None
        md = report.render(rows, SLO, baseline, best, flagged, None, baseline_ctx=ctx)
        assert "基线上下文内无可比较的 trial" in md
        assert "无任何可比较的 trial（results.tsv 为空或全部 crash）" not in md   # 旧结论行不再出现
        # 全 crash → 保留原文案
        all_crash = [dict(r, status="crash") for r in report.parse_tsv(NEW_TSV)]
        md2 = report.render(all_crash, SLO, all_crash[0], None, True, None,
                            baseline_ctx=report.context_key(all_crash[0]))
        assert "无任何可比较的 trial（results.tsv 为空或全部 crash）" in md2


class TestDualSlo:
    def test_reference_reachable_explicit_line(self):
        # 参考 SLO 可达：概览出「可达」，基线 vs 最佳附参考 SLO 下最佳行
        rows = report.parse_tsv(NEW_TSV)
        baseline = report.pick_baseline(rows)
        ctx = report.context_key(baseline)
        best, flagged = report.pick_best(rows, SLO, ctx)
        md = report.render(rows, SLO, baseline, best, flagged,
                           report.improvement_pct(baseline, best),
                           reference_slo=SLO, baseline_ctx=ctx)
        assert "参考 SLO（target 声明，不改语义）：ttft_p99 ≤ 500 ms" in md
        assert "有效 SLO（goodput/keep 判定用）：ttft_p99 ≤ 500 ms" in md
        assert "参考 SLO：**可达**" in md
        assert "参考 SLO 下的最佳" in md
        # 参考 = 有效 → 标注与参考一致，且无校准标注
        assert "（与参考一致）" in md
        assert "运行期 SLO 校准" not in md

    def test_reference_unreachable_explicit_line(self):
        # 参考 SLO 不可达：概览出「不可达」+ 说明；基线 vs 最佳明示参考 SLO 下无达标行
        rows = report.parse_tsv(NEW_TSV)
        baseline = report.pick_baseline(rows)
        ctx = report.context_key(baseline)
        best, flagged = report.pick_best(rows, SLO, ctx)
        md = report.render(rows, SLO, baseline, best, flagged,
                           report.improvement_pct(baseline, best),
                           reference_slo={"ttft_p99_ms": 400, "tpot_p99_ms": 30},
                           baseline_ctx=ctx)
        assert "参考 SLO：**不可达**" in md
        assert "建议放宽至实测 p99 或复核硬件/模型适配" in md
        assert "参考 SLO 下无达标行" in md
        # 有效 SLO（500/50）仍是判定口径 → best 照常
        assert best["commit"] == "cccc333"

    def test_effective_slo_reported_when_calibrated(self):
        # 双口径 + 校准：有效 SLO 与参考不同 → 概览双列 + 校准标注
        rows = report.parse_tsv(NEW_TSV)
        baseline = report.pick_baseline(rows)
        ctx = report.context_key(baseline)
        eff = {"ttft_p99_ms": 550, "tpot_p99_ms": 55}
        best, _ = report.pick_best(rows, eff, ctx)
        md = report.render(rows, eff, baseline, best, False,
                           report.improvement_pct(baseline, best),
                           reference_slo=SLO, baseline_ctx=ctx,
                           slo_calibrated=SLO)
        assert "参考 SLO（target 声明，不改语义）：ttft_p99 ≤ 500 ms" in md
        assert "有效 SLO（goodput/keep 判定用）：ttft_p99 ≤ 550 ms" in md
        assert "（运行期校准，见下）" in md
        assert "运行期 SLO 校准" in md
