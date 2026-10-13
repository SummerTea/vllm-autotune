"""test_estimates.py — 估算纯函数的单元测试（合成 GQA 配置）。

口径与 probe.py 内实现一致；用合成 Qwen2 式 GQA 配置验证：
tp 合法性（同时整除 heads 与 kv_heads）、kv 每 token 字节数、roofline 单调性。
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe  # noqa: E402

# 合成 GQA 配置（32 attn heads / 8 kv heads / 64 层 / hidden 4096 / head_dim 128）
GQA_CONFIG = {
    "model_type": "qwen2",
    "num_hidden_layers": 64,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 152064,
    "intermediate_size": 14336,
    "max_position_embeddings": 32768,
    "torch_dtype": "bfloat16",
}

GPU_MEM = 42234101760  # 40 GiB（与 fixture 的 A100-40GB 一致）
GMU = 0.92


def _load_fixture_json(name: str) -> dict:
    p = Path(__file__).parent / "fixtures" / "head" / name
    return json.loads(p.read_text(encoding="utf-8"))


class TestDtypeBytes:
    def test_quantization_wins_over_torch_dtype(self):
        # 真实 GPTQ 模型 torch_dtype=float16，但按量化应估 0.5 B/参数（否则权重 4 倍虚高）
        cfg = {"torch_dtype": "float16", "quantization_config": {"quant_method": "gptq"}}
        assert probe.dtype_bytes(cfg) == 0.5

    def test_quantization_applies_to_weight_bytes(self):
        base = dict(GQA_CONFIG, torch_dtype="float16")
        q = dict(base, quantization_config={"quant_method": "awq"})
        wb_plain, _ = probe.weight_bytes(None, base)
        wb_quant, _ = probe.weight_bytes(None, q)
        assert wb_plain is not None and wb_quant is not None
        assert wb_quant == pytest.approx(wb_plain / 4)

    def test_torch_dtype_fallback(self):
        assert probe.dtype_bytes({"torch_dtype": "float32"}) == 4
        assert probe.dtype_bytes({}) == 2  # bf16 兜底


class TestParseGpuSmi:
    def test_malformed_row_skipped_others_parsed(self):
        # 坏单元格（power_draw='1.2.3' 匹配数字正则但非法）不得炸掉整行
        text = (
            "0, NVIDIA A100-SXM4-40GB, 42234101760, 535.104.05, Enabled, 810, 1410, 1.2.3, 300.00\n"
            "1, NVIDIA A100-SXM4-40GB, 42234101760, 535.104.05, Enabled, 810, 1410, 28.63, 300.00\n"
        )
        gpus = probe.parse_gpu_smi(text)
        assert gpus is not None
        assert len(gpus) == 2
        assert gpus[0]["power_draw"] is None
        assert gpus[1]["power_draw"] == 28.63
        assert gpus[1]["power_limit"] == 300.0

    def test_n_a_cells_become_none(self):
        text = "0, NVIDIA A100-SXM4-40GB, 42234101760, 535.104.05, Disabled, N/A, N/A, N/A, N/A\n"
        gpus = probe.parse_gpu_smi(text)
        assert gpus is not None
        assert gpus[0]["clocks_sm"] is None
        assert gpus[0]["power_draw"] is None


class TestWeightBytes:
    def test_prefers_safetensors_index_total_size(self):
        index = _load_fixture_json("safetensors.index.json")
        wb, source = probe.weight_bytes(index, GQA_CONFIG)
        assert wb == 29200000000
        assert source == "safetensors-index"

    def test_falls_back_to_params_estimate(self):
        wb, source = probe.weight_bytes(None, GQA_CONFIG)
        # 估算 ≈ 14.58B params × 2B（bf16）；与 index 的 29.2GB 同一量级
        assert source == "estimate"
        assert wb and 25e9 < wb < 35e9

    def test_dtype_scaling(self):
        cfg32 = dict(GQA_CONFIG, torch_dtype="float32")
        wb16, _ = probe.weight_bytes(None, GQA_CONFIG)
        wb32, _ = probe.weight_bytes(None, cfg32)
        assert wb16 is not None and wb32 is not None
        assert wb32 == 2 * wb16

    def test_unknown_arch_returns_unavailable(self):
        wb, source = probe.weight_bytes(None, {"foo": 1})
        assert wb is None
        assert source == "unavailable"


class TestValidTpDivisors:
    def test_gqa_divisibility(self):
        tp = probe.valid_tp_divisors(GQA_CONFIG, GPU_MEM, GMU)
        assert 8 in tp["valid_divisors"]     # 32%8==0 且 8%8==0
        assert 3 not in tp["valid_divisors"]  # 32%3 != 0
        assert 16 not in tp["valid_divisors"]  # 8%16 != 0
        assert tp["min_tp_for_weights"] == 1

    def test_memory_fit_filter(self):
        # 单卡 20GiB 时 tp=8 装不下（29.2GB/8=3.65GB 装得下；tp=4 → 7.3GB 装得下）
        small = 20 * 2**30
        tp = probe.valid_tp_divisors(GQA_CONFIG, small, GMU)
        assert all(wb_is_ok(t, small, GMU) for t in tp["valid_divisors"])

    def test_min_tp_for_weights_scales_with_gpu_mem(self):
        tp_small = probe.valid_tp_divisors(GQA_CONFIG, 20 * 2**30, GMU)
        tp_large = probe.valid_tp_divisors(GQA_CONFIG, 40 * 2**30, GMU)
        assert tp_small["min_tp_for_weights"] >= tp_large["min_tp_for_weights"]

    def test_min_tp_lands_on_non_divisor_gives_none(self):
        # F2 回归：ceil(wb/budget)=11，不是 32/8 的合法约数 → min_tp 必须为 None
        tp = probe.valid_tp_divisors({"num_attention_heads": 32, "num_key_value_heads": 8},
                                     40 * 1024**3, 0.92, 400 * 1024**3)
        assert tp["min_tp_for_weights"] is None
        assert tp["valid_divisors"] == []

    def test_min_tp_rounded_up_to_valid_divisor(self):
        # F2 回归：ceil(wb/budget)=6（非约数）→ 必须取下一个合法约数 8
        tp = probe.valid_tp_divisors({"num_attention_heads": 32, "num_key_value_heads": 8},
                                     40 * 1024**3, 0.92, 200 * 1024**3)
        assert tp["min_tp_for_weights"] == 8
        assert tp["min_tp_for_weights"] in tp["valid_divisors"]

    def test_no_contradiction_between_min_tp_and_divisors(self):
        for wb in (100 * 1024**3, 200 * 1024**3, 400 * 1024**3):
            tp = probe.valid_tp_divisors({"num_attention_heads": 32, "num_key_value_heads": 8},
                                         40 * 1024**3, 0.92, wb)
            mtp = tp["min_tp_for_weights"]
            assert mtp is None or mtp in tp["valid_divisors"]


def wb_is_ok(t: int, gpu_mem: int, gmu: float) -> bool:
    wb, _ = probe.weight_bytes(None, GQA_CONFIG)
    return wb is not None and wb / t <= gpu_mem * gmu


class TestKvBytes:
    def test_gqa_kv_bytes_per_token(self):
        # 64 层 × 8 kv heads × 128 head_dim × 2(K+V) × 2B(bf16) = 262144
        assert probe.kv_bytes_per_token(GQA_CONFIG) == 262144

    def test_mla_unsupported_returns_none(self):
        cfg = dict(GQA_CONFIG, kv_lora_rank=512)
        assert probe.kv_bytes_per_token(cfg) is None

    def test_mamba_unsupported_returns_none(self):
        cfg = dict(GQA_CONFIG, model_type="mamba2")
        assert probe.kv_bytes_per_token(cfg) is None

    def test_head_dim_defaults_to_hidden_over_heads(self):
        cfg = dict(GQA_CONFIG)
        cfg.pop("head_dim")
        assert probe.kv_bytes_per_token(cfg) == 262144  # head_dim=128 兜底


class TestRooflineAndBoot:
    def test_roofline_monotonic(self):
        wb = 29200000000
        slow = probe.roofline_decode_tps(wb, 500.0)
        fast = probe.roofline_decode_tps(wb, 1000.0)
        assert slow is not None and fast is not None
        assert fast > slow
        assert probe.roofline_decode_tps(wb, 1555.0) == pytest.approx(1555e9 / wb)

    def test_roofline_none_without_inputs(self):
        assert probe.roofline_decode_tps(None, 1000.0) is None
        assert probe.roofline_decode_tps(1, None) is None

    def test_boot_seconds_scales_with_size(self):
        b1 = probe.estimate_boot_seconds(10_000_000_000)
        b2 = probe.estimate_boot_seconds(20_000_000_000)
        assert b1 is not None and b2 is not None
        assert b2 == pytest.approx(2 * b1)
        assert probe.estimate_boot_seconds(None) is None
