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


class TestParseIpLink:
    def test_ip_o_link_show(self):
        # `ip -o link show` 格式：N: name: <flags> mtu M …
        text = (
            "1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN mode DEFAULT "
            "group default qlen 1000\\    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00\n"
            "2: ens5: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc mq state UP mode DEFAULT "
            "group default qlen 1000\\    link/ether 02:00:00:00:00:01 brd ff:ff:ff:ff:ff:ff\n"
        )
        ifaces = probe.parse_ip_link(text)
        assert len(ifaces) == 2
        assert ifaces[0]["name"] == "lo" and ifaces[0]["mtu"] == 65536
        assert ifaces[1]["name"] == "ens5" and ifaces[1]["mtu"] == 1500

    def test_bad_lines_skipped(self):
        assert probe.parse_ip_link("no mtu here\n") == []
        assert probe.parse_ip_link(None) == []


class TestNestedTextConfig:
    # 新式 HF 配置：文本模型参数嵌在 text_config 下（glm5_next / Glm5NextForConditionalGeneration）
    NESTED = {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": 64,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 0,                     # 0 → 用 hidden/heads 兜底
            "vocab_size": 152064,
            "intermediate_size": 14336,
            "max_position_embeddings": 32768,
        },
    }

    def test_normalize_pulls_from_text_config(self):
        c = probe.normalize_config(self.NESTED)
        assert c["hidden_size"] == 4096
        assert c["num_hidden_layers"] == 64
        assert c["num_key_value_heads"] == 8
        assert c["model_type"] == "glm5_next"

    def test_head_dim_zero_normalized(self):
        c = probe.normalize_config(self.NESTED)
        assert c["head_dim"] is None           # 0 → None（kv 估算以 hidden//heads 兜底）
        assert probe.effective_head_dim(c) == 128

    def test_kv_bytes_non_null_after_normalize(self):
        c = probe.normalize_config(self.NESTED)
        assert probe.kv_bytes_per_token(c) == 262144   # 64×8×128×2×2

    def test_arch_fields_from_text_config_via_profile(self):
        head_facts = {
            "model_src": {"config_json": dict(self.NESTED), "safetensors_index": None},
            "gpu": {"gpus": [{"name": "NVIDIA A100-SXM4-40GB", "memory_total": 42234101760}]},
        }
        prof = probe.build_model_profile({"id": "glm5-next"}, head_facts, 0.92)
        assert prof["arch"]["num_layers"] == 64
        assert prof["arch"]["hidden_size"] == 4096
        assert prof["arch"]["num_key_value_heads"] == 8
        assert prof["arch"]["head_dim"] == 128     # head_dim=0 → hidden/heads
        assert prof["kv"]["bytes_per_token"] == 262144

    def test_plain_config_untouched(self):
        c = probe.normalize_config(dict(GQA_CONFIG))
        assert c["hidden_size"] == GQA_CONFIG["hidden_size"]
        assert c["head_dim"] == GQA_CONFIG["head_dim"]


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
