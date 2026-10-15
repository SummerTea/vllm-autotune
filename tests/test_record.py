"""test_record.py — scripts/record.py 单元测试（行拼接 / 表头迁移 / 上下文列）。

record.main 写账本用真实模块（tmp_path 内），行拼接待测函数直接调用。
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import record  # noqa: E402
from report import HEADER  # noqa: E402

OK_RESULT = {
    "status": "ok",
    "goodput_req_s": 12.84,
    "output_tok_s": 12345.6,
    "ttft_p99_ms": 620.1,
    "tpot_p99_ms": 18.3,
    "stats": {"goodput_std": 0.42},
    "peak_vram_gb": 71.2,
    "dur_s": 912.4,
    "repeats": 2,
    "workload": {"dataset": "random", "isl": 2048, "osl": 256, "num_prompts": 1000,
                 "ref_concurrency": 128, "prefix_hit_pct": 0},
    "slo": {"ttft_p99_ms": 500, "tpot_p99_ms": 50},
    "trial_id": "20261015T103000-abc123",
    "config_hash": "feedfacebeef",
    "started_at": "2026-10-15T10:30:00",
}


class TestBuildRow:
    def test_ok_row_full_identity(self):
        # ok 行：全部列按 HEADER 顺序、身份/上下文/数值齐全；warmup_repeats 缺省 → 0（协议列）
        row = record.build_row(OK_RESULT, commit="abc1234", desc="基线",
                               status="keep", slo=OK_RESULT["slo"],
                               wl_hash=record.workload_hash(OK_RESULT))
        assert len(row) == len(HEADER)
        assert row == [
            "abc1234", "20261015T103000-abc123", "feedfacebeef",
            record.workload_hash(OK_RESULT), "ttft500/tpot50",
            "2", "0", "0", "12.84", "0.42", "12345.6", "620.1", "18.3", "71.2", "912.4",
            "keep", "基线",
        ]

    def test_ok_row_warmup_repeats_present(self):
        # M2a：ok 行带 warmup_repeats → 协议列如实写入
        with_wu = dict(OK_RESULT, warmup_repeats=1)
        row = record.build_row(with_wu, commit="c", desc="", status="keep",
                               slo=OK_RESULT["slo"], wl_hash="w")
        assert row[6] == "1"
        assert row[5] == "2"                      # repeats 在其前

    def test_crash_row_blank_metrics(self):
        # crash 行：数值列留空，身份/上下文列保留
        crash = {"status": "crash", "reason": "OOM",
                 "trial_id": "T-crash", "config_hash": "h1",
                 "workload": OK_RESULT["workload"], "slo": OK_RESULT["slo"]}
        row = record.build_row(crash, commit="deadbeef", desc="OOM", status="crash",
                               slo=crash["slo"], wl_hash=record.workload_hash(crash))
        assert len(row) == len(HEADER)
        # 数值列全空：repeats(5)/warmup(6)/goodput(8)/goodput_std(9)/…/dur(14)
        for i in (5, 6, 8, 9, 10, 11, 12, 13, 14):
            assert row[i] == ""
        assert row[1] == "T-crash" and row[2] == "h1"          # 身份列保留
        assert row[4] == "ttft500/tpot50"                      # slo 列保留
        assert row[16] == "OOM"

    def test_contention_flag(self):
        with_cont = dict(OK_RESULT, contention={"foreign_gpu_apps": []})
        row = record.build_row(with_cont, commit="c", desc="", status="keep",
                               slo=OK_RESULT["slo"], wl_hash="w")
        assert row[7] == "1"
        row2 = record.build_row(OK_RESULT, commit="c", desc="", status="keep",
                                slo=OK_RESULT["slo"], wl_hash="w")
        assert row2[7] == "0"


class TestWorkloadHash:
    def test_stable_and_short(self):
        h = record.workload_hash(OK_RESULT)
        assert len(h) == 12
        assert record.workload_hash(OK_RESULT) == h

    def test_missing_workload_empty(self):
        assert record.workload_hash({"status": "crash"}) == ""

    def test_seed_changes_hash(self):
        # M2c：seed 属 workload 身份——seed 变化 → workload_hash 变化
        wl0 = {**OK_RESULT["workload"], "seed": 0}
        wl1 = {**OK_RESULT["workload"], "seed": 1}
        assert record.workload_hash({"workload": wl0}) != record.workload_hash({"workload": wl1})


class TestFormatSlo:
    def test_format(self):
        assert record.format_slo({"ttft_p99_ms": 500, "tpot_p99_ms": 50}) == "ttft500/tpot50"

    def test_empty(self):
        assert record.format_slo(None) == ""
        assert record.format_slo({}) == ""

    def test_integral_rendering_unified(self):
        # F3：4000 与 4000.0 必须渲染成同一单元格（否则同一 SLO 被拆成两个伪上下文）
        assert record.format_slo({"ttft_p99_ms": 4000.0, "tpot_p99_ms": 80.0}) == "ttft4000/tpot80"
        assert record.format_slo({"ttft_p99_ms": 4000, "tpot_p99_ms": 80}) == "ttft4000/tpot80"
        assert record.format_slo({"ttft_p99_ms": 4000.0, "tpot_p99_ms": 80}) == "ttft4000/tpot80"

    def test_non_integral_preserved(self):
        assert record.format_slo({"ttft_p99_ms": 0.5, "tpot_p99_ms": 50}) == "ttft0.5/tpot50"

    def test_missing_key_empties_whole_cell(self):
        # F3：任一键缺失 → 整格空，绝不产出 ttftNone/...
        assert record.format_slo({"ttft_p99_ms": None, "tpot_p99_ms": 50}) == ""
        assert record.format_slo({"ttft_p99_ms": 500}) == ""


def _write_result_json(tmp_path: Path, result: dict) -> Path:
    p = tmp_path / "result.json"
    p.write_text(json.dumps(result), encoding="utf-8")
    return p


class TestMain:
    def test_header_written_when_absent(self, tmp_path, capsys):
        _write_result_json(tmp_path, OK_RESULT)
        out = tmp_path / "results.tsv"
        rc = record.main(["--status", "keep", "--desc", "基线",
                          "--commit", "abc1234",
                          "--result-json", str(tmp_path / "result.json"),
                          "--results", str(out)])
        assert rc == 0
        text = out.read_text(encoding="utf-8")
        lines = text.splitlines()
        assert lines[0].split("\t") == HEADER        # 首建写表头
        assert len(lines) == 2                        # 表头 + 1 行
        assert lines[1].startswith("abc1234\t20261015T103000-abc123")
        # 透明回显：stdout 打印写入的行
        assert capsys.readouterr().out.split("\t")[0] == "abc1234"

    def test_mismatched_existing_header_errors(self, tmp_path, capsys):
        _write_result_json(tmp_path, OK_RESULT)
        out = tmp_path / "results.tsv"
        out.write_text("commit\tgoodput\tstatus\nx\t1\ty\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            record.main(["--status", "keep", "--desc", "d", "--commit", "c",
                         "--result-json", str(tmp_path / "result.json"),
                         "--results", str(out)])
        assert "请迁移表头" in capsys.readouterr().err

    def test_missing_result_json_errors(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            record.main(["--status", "keep", "--desc", "d", "--commit", "c",
                         "--result-json", str(tmp_path / "nope.json"),
                         "--results", str(tmp_path / "results.tsv")])
        assert "未找到" in capsys.readouterr().err

    def test_desc_with_newline_rejected(self, tmp_path, capsys):
        # F2：--desc 含换行/tab → 拒绝且不修改账本（防止嵌入幻影行）
        _write_result_json(tmp_path, OK_RESULT)
        out = tmp_path / "results.tsv"
        out.write_text("\t".join(HEADER) + "\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            record.main(["--status", "keep", "--desc", "a\nb", "--commit", "c",
                         "--result-json", str(tmp_path / "result.json"), "--results", str(out)])
        assert "单行" in capsys.readouterr().err
        assert out.read_text(encoding="utf-8") == "\t".join(HEADER) + "\n"   # 文件未被修改

    def test_desc_with_tab_rejected(self, tmp_path, capsys):
        _write_result_json(tmp_path, OK_RESULT)
        out = tmp_path / "results.tsv"
        with pytest.raises(SystemExit):
            record.main(["--status", "keep", "--desc", "a\tb", "--commit", "c",
                         "--result-json", str(tmp_path / "result.json"), "--results", str(out)])
        assert "单行" in capsys.readouterr().err
        assert not out.exists()                        # 未创建文件

    def test_ok_row_empty_identity_rejected(self, tmp_path, capsys):
        # F6：ok 行 trial_id / config_hash 任一为空 → 拒绝入账
        _write_result_json(tmp_path, {k: v for k, v in OK_RESULT.items() if k != "trial_id"})
        out = tmp_path / "results.tsv"
        with pytest.raises(SystemExit):
            record.main(["--status", "keep", "--desc", "d", "--commit", "c",
                         "--result-json", str(tmp_path / "result.json"), "--results", str(out)])
        assert "trial_id" in capsys.readouterr().err

    def test_crash_row_both_identity_empty_rejected(self, tmp_path, capsys):
        # F6：crash 行两者皆空 → 拒绝（无法入账）
        _write_result_json(tmp_path, {"status": "crash", "reason": "OOM"})
        out = tmp_path / "results.tsv"
        with pytest.raises(SystemExit):
            record.main(["--status", "crash", "--desc", "OOM", "--commit", "c",
                         "--result-json", str(tmp_path / "result.json"), "--results", str(out)])
        assert "trial_id" in capsys.readouterr().err

    def test_crash_row_partial_identity_allowed(self, tmp_path):
        # F6：crash 行至少一个身份列存在 → 允许（保留可用身份，缺列留空）
        _write_result_json(tmp_path, {"status": "crash", "reason": "OOM", "trial_id": "T1"})
        out = tmp_path / "results.tsv"
        rc = record.main(["--status", "crash", "--desc", "OOM", "--commit", "c",
                          "--result-json", str(tmp_path / "result.json"), "--results", str(out)])
        assert rc == 0
        row = out.read_text(encoding="utf-8").splitlines()[1].split("\t")
        assert row[1] == "T1" and row[2] == ""
