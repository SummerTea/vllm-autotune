"""test_end_to_end_mock.py — probe → bench → report 全链路 mock 集成测试。

在临时 cwd 中用真实模块跑通：probe.py（fixture 模式）→ bench.py（stub docker）
→ 手写 results.tsv → report.py。无 GPU / 无网络；可在 Mac 上运行。
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROBE = REPO / "probe.py"
BENCH = REPO / "bench.py"
REPORT = REPO / "report.py"
FIXTURES = REPO / "tests" / "fixtures"
HEAD_FIX = FIXTURES / "head"
BENCH_FIX = FIXTURES / "bench"

PORT = 18090


def _write_yaml(tmp: Path, name: str, data: dict) -> Path:
    p = tmp / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


def test_full_pipeline(tmp_path):
    # 准备：fixture 目录作为本地模型路径；target/harness/config 齐全
    model_dir = tmp_path / "model"
    shutil.copytree(HEAD_FIX, model_dir)
    _write_yaml(tmp_path, "target.yaml", {
        "hosts": [{"name": "head", "ssh": "user@10.0.0.1", "role": "head"}],
        "model": {"id": "Qwen/Qwen3-32B", "source": "local", "path": str(model_dir),
                  "trust_remote_code": False},
    })
    _write_yaml(tmp_path, "harness.yaml", {
        "bench": {"tool": "vllm bench serve", "repeats": 1, "noise_band_pct": 3.0,
                  "startup_timeout_s": 60, "trial_timeout_s": 120, "ref_concurrency": 128},
        "slo": {"ttft_p99_ms": 500, "tpot_p99_ms": 50},
        "workload": {"dataset": "random", "isl": 2048, "osl": 256, "num_prompts": 1000},
        "exit": {"plateau_trials": 8, "crash_streak": 3},
        "remote": {"vllm_port": PORT, "vllm_python": "python"},
        "runtime": {"kind": "docker", "image": "vllm/vllm-openai:v0.31.0"},
    })
    _write_yaml(tmp_path, "config.yaml", {"env": {}, "serve": {}})

    # (a) probe.py --fixtures（离线采集）
    r = _run([sys.executable, str(PROBE), "--fixtures", str(FIXTURES), "--out-dir", "."],
             cwd=tmp_path)
    assert r.returncode == 0, f"probe 失败:\n{r.stdout}\n{r.stderr}"
    for f in ("env.json", "env_report.md", "model_profile.json"):
        assert (tmp_path / f).exists(), f"probe 缺产物 {f}"

    # (b) bench.py（stub docker：serve/bench 都在 fake_docker.sh 容器内跑）
    r = _run([sys.executable, str(BENCH),
              "--config", str(tmp_path / "config.yaml"),
              "--harness", str(tmp_path / "harness.yaml"),
              "--target", str(tmp_path / "target.yaml"),
              "--out-dir", ".",
              "--docker-cmd", str(BENCH_FIX / "fake_docker.sh"),
              "--poll-interval", "0.2", "--vram-interval", "0.3"],
             cwd=tmp_path)
    assert r.returncode == 0, f"bench 失败:\n{r.stdout}\n{r.stderr}"
    assert "status: ok" in r.stdout
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert status["state"] == "done"

    # (c) 把本次 trial 追加进 results.tsv（status=keep，SLO 不达标 → 回退路径可被 report 展示）
    header = "commit\tgoodput\tgoodput_std\toutput_tok_s\tttft_p99_ms\ttpot_p99_ms\tpeak_vram_gb\tdur_s\tstatus\tdescription"
    row = "\t".join([
        "deadbeef",
        str(result["goodput_req_s"]),
        str((result.get("stats") or {}).get("goodput_std")),
        str(result["output_tok_s"]),
        str(result["ttft_p99_ms"]),
        str(result["tpot_p99_ms"]),
        str(result["peak_vram_gb"] or ""),
        str(result["dur_s"]),
        "keep",
        "e2e-mock baseline",
    ])
    (tmp_path / "results.tsv").write_text(header + "\n" + row + "\n", encoding="utf-8")

    # (d) report.py（缺省读取本 cwd 的 results.tsv/harness/target/env_report）
    r = _run([sys.executable, str(REPORT), "--out", "report.md"], cwd=tmp_path)
    assert r.returncode == 0, f"report 失败:\n{r.stdout}\n{r.stderr}"
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    for sec in ("# 调优报告", "## 概览", "## 基线 vs 最佳", "## 全部 trial 表",
                "## 最佳配置", "## 部署（best 配置）", "## 确认（best 回归）",
                "## 系统级建议", "## 适用范围"):
        assert sec in md, f"report.md 缺小节 {sec}"
    assert "Qwen/Qwen3-32B" in md                 # 模型来自 model_profile.json
    assert "无满足 SLO 的 trial" in md            # ttft 620.1 > 500 → 回退标注
    assert "请手动执行该命令" in md               # tmp 非 git 仓库 → git show 失败提示
    assert "deadbeef" in md                       # 最佳 commit 出现在报告
