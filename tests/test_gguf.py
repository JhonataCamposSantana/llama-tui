import unittest

from unittest.mock import patch

from llama_tui.gguf import (
    GGML_TYPE_NAME,
    GGML_TYPE_SIZE,
    _UNKNOWN_GGML_TYPES_SEEN,
    _estimated_tensor_payload_bytes,
    cache_type_bytes,
    estimate_kv_bytes_per_token,
    extra_arg_value,
    has_extra_flag,
    selected_cache_type,
    set_model_extra_arg,
    strip_extra_args,
    unknown_ggml_types_seen,
)
from llama_tui.models import ModelConfig


def _model(extra_args=None):
    return ModelConfig(
        id='m', name='M', path='/m.gguf', alias='m',
        extra_args=list(extra_args or []),
    )


class ExtraArgTests(unittest.TestCase):
    def test_value_from_separate_token(self):
        self.assertEqual(extra_arg_value(['--ctx-size', '4096'], '--ctx-size'), '4096')

    def test_value_from_equals_form(self):
        self.assertEqual(extra_arg_value(['--ctx-size=8192'], '--ctx-size'), '8192')

    def test_value_missing(self):
        self.assertIsNone(extra_arg_value(['--foo'], '--ctx-size'))

    def test_has_extra_flag(self):
        self.assertTrue(has_extra_flag(['--no-mmap'], '--no-mmap'))
        self.assertTrue(has_extra_flag(['--ctx-size=8192'], '--ctx-size'))
        self.assertFalse(has_extra_flag(['--foo'], '--no-mmap'))

    def test_strip_extra_args_removes_flag_and_value(self):
        self.assertEqual(
            strip_extra_args(['--ctx-size', '4096', '--keep'], '--ctx-size'),
            ['--keep'],
        )
        self.assertEqual(
            strip_extra_args(['--ctx-size=8192', '--keep'], '--ctx-size'),
            ['--keep'],
        )


class ModelExtraArgTests(unittest.TestCase):
    def test_set_model_extra_arg_replaces(self):
        model = _model(['--ctx-size', '4096', '--threads', '6'])
        set_model_extra_arg(model, '--ctx-size', '131072')
        self.assertEqual(extra_arg_value(model.extra_args, '--ctx-size'), '131072')
        self.assertEqual(extra_arg_value(model.extra_args, '--threads'), '6')
        self.assertEqual(model.extra_args.count('--ctx-size'), 1)

    def test_selected_cache_type_turboquant_aware_default(self):
        # 2026-05-26: a turboquant-eligible model (native/padded,
        # head_dim>=128) actually launches with q8_0/turbo4 by default
        # (see AppConfig.turboquant_served_kv_preset). The estimator must
        # agree -- using f16 for both made kv_per_token 2-4x too high
        # and pushed every ctx cap below what would actually fit.
        model = _model()
        model.turboquant_status = 'native'
        model.turboquant_head_dim = 256
        self.assertEqual(selected_cache_type(model, 'k'), 'q8_0')
        self.assertEqual(selected_cache_type(model, 'v'), 'turbo4')

        small_head = _model()
        small_head.turboquant_status = 'incompatible'
        small_head.turboquant_head_dim = 64
        self.assertEqual(selected_cache_type(small_head, 'k'), 'f16')
        self.assertEqual(selected_cache_type(small_head, 'v'), 'f16')

    def test_selected_cache_type_per_model_override_beats_default(self):
        # Per-model kv_key_mode/kv_value_mode pin (e.g. q4_0/q4_0 for an
        # MXFP4 GGUF that tolerates it) must reflect in the estimator
        # too, not just the served launch.
        model = _model()
        model.kv_key_mode = 'q4_0'
        model.kv_value_mode = 'q4_0'
        self.assertEqual(selected_cache_type(model, 'k'), 'q4_0')
        self.assertEqual(selected_cache_type(model, 'v'), 'q4_0')

        # Explicit -ctk/-ctv still wins over per-model pin.
        explicit = _model(['-ctk', 'q5_1', '-ctv', 'q5_1'])
        explicit.kv_key_mode = 'q4_0'
        explicit.kv_value_mode = 'q4_0'
        self.assertEqual(selected_cache_type(explicit, 'k'), 'q5_1')
        self.assertEqual(selected_cache_type(explicit, 'v'), 'q5_1')

    def test_selected_cache_type_default_and_override(self):
        self.assertEqual(selected_cache_type(_model(), 'k'), 'f16')
        self.assertEqual(
            selected_cache_type(_model(['-ctk', 'q8_0']), 'k'), 'q8_0'
        )
        self.assertEqual(
            selected_cache_type(_model(['-ctv', 'q4_0']), 'v'), 'q4_0'
        )


class GgmlTypeTableTests(unittest.TestCase):
    """Guards against the historical "shifted" GGML_TYPE_SIZE table that
    inflated weight estimates by 30-50% for K-quant and Q8 models. See audit
    finding #3."""

    def test_legacy_q_quant_sizes_match_block_math(self):
        # Legacy Q-quants are 32 elements per block.
        self.assertEqual(GGML_TYPE_SIZE[2], 18 / 32)   # Q4_0
        self.assertEqual(GGML_TYPE_SIZE[6], 22 / 32)   # Q5_0
        self.assertEqual(GGML_TYPE_SIZE[8], 34 / 32)   # Q8_0

    def test_k_quant_sizes_use_256_element_blocks(self):
        self.assertAlmostEqual(GGML_TYPE_SIZE[10], 84 / 256)   # Q2_K
        self.assertAlmostEqual(GGML_TYPE_SIZE[12], 144 / 256)  # Q4_K
        self.assertAlmostEqual(GGML_TYPE_SIZE[14], 210 / 256)  # Q6_K

    def test_names_and_sizes_are_aligned(self):
        self.assertEqual(GGML_TYPE_NAME[0], 'F32')
        self.assertEqual(GGML_TYPE_NAME[1], 'F16')
        self.assertEqual(GGML_TYPE_NAME[8], 'Q8_0')
        self.assertEqual(GGML_TYPE_NAME[12], 'Q4_K')
        self.assertEqual(GGML_TYPE_NAME[30], 'BF16')
        self.assertEqual(GGML_TYPE_NAME[39], 'MXFP4')
        self.assertEqual(set(GGML_TYPE_SIZE.keys()), set(GGML_TYPE_NAME.keys()))

    def test_unknown_ggml_type_is_recorded_with_conservative_size(self):
        _UNKNOWN_GGML_TYPES_SEEN.discard(9999)
        baseline = unknown_ggml_types_seen()
        size = _estimated_tensor_payload_bytes(
            {'type': 9999, 'dimensions': [16, 16]}
        )
        self.assertEqual(size, int(16 * 16 * 2.0))
        self.assertIn(9999, unknown_ggml_types_seen())
        _UNKNOWN_GGML_TYPES_SEEN.discard(9999)
        self.assertEqual(unknown_ggml_types_seen(), baseline)


class EstimateKvBytesTests(unittest.TestCase):
    def test_mla_branch_uses_latent_plus_rope(self):
        # DeepSeek-V3 shape: 61 layers, kv_lora_rank=512, rope_dim=64.
        # Expected per token: 61 * (512 + 64) * 2 bytes * 1.08 overhead ≈ 75.8 KB.
        metadata = {
            'general.architecture': 'deepseek2',
            'deepseek2.block_count': 61,
            'deepseek2.attention.kv_lora_rank': 512,
            'deepseek2.rope.dimension_count': 64,
        }
        model = ModelConfig(id='m', name='M', path='/m.gguf', alias='m')
        with patch('llama_tui.gguf.read_gguf_metadata', return_value=metadata):
            kv_bytes = estimate_kv_bytes_per_token(model)
        expected = int(61 * (512 + 64) * 2 * 1.08)
        self.assertEqual(kv_bytes, expected)
        # Sanity: the naive head_count*head_dim path would be much larger.
        # DeepSeek-V3 has 128 heads × 192 key_length+128 value_length ≈ 5x bigger.
        self.assertLess(kv_bytes, 100_000)

    def test_sliding_window_halves_dense_estimate(self):
        # Gemma-2 9B shape: 42 layers, 16 heads, 8 KV heads, head_dim 256,
        # sliding_window=4096. Expected: dense × 0.5 × 1.08 overhead.
        metadata = {
            'general.architecture': 'gemma2',
            'gemma2.block_count': 42,
            'gemma2.attention.head_count': 16,
            'gemma2.attention.head_count_kv': 8,
            'gemma2.embedding_length': 3584,
            'gemma2.attention.key_length': 256,
            'gemma2.attention.value_length': 256,
            'gemma2.attention.sliding_window': 4096,
        }
        model = ModelConfig(id='m', name='M', path='/m.gguf', alias='m')
        with patch('llama_tui.gguf.read_gguf_metadata', return_value=metadata):
            kv_bytes = estimate_kv_bytes_per_token(model)
        dense = 42 * 8 * (256 * 2 + 256 * 2)
        expected = int(dense * 0.5 * 1.08)
        self.assertEqual(kv_bytes, expected)

    def test_sliding_window_absent_keeps_dense_estimate(self):
        # Identical shape without the sliding-window key returns the dense
        # value — the branch only triggers when the metadata signals it.
        metadata = {
            'general.architecture': 'llama',
            'llama.block_count': 42,
            'llama.attention.head_count': 16,
            'llama.attention.head_count_kv': 8,
            'llama.embedding_length': 3584,
            'llama.attention.key_length': 256,
            'llama.attention.value_length': 256,
        }
        model = ModelConfig(id='m', name='M', path='/m.gguf', alias='m')
        with patch('llama_tui.gguf.read_gguf_metadata', return_value=metadata):
            kv_bytes = estimate_kv_bytes_per_token(model)
        dense = 42 * 8 * (256 * 2 + 256 * 2)
        expected = int(dense * 1.08)
        self.assertEqual(kv_bytes, expected)

    def test_hybrid_ssm_arch_scales_down_dense_estimate(self):
        # 2026-05-26: Nemotron-H interleaves SSM/Mamba layers (no KV) with
        # attention layers. Treating every layer as attention returned
        # 920 KiB/token for iq4-xs (Nemotron-3 30B-A3B MoE IQ4_XS), capping
        # ctx at 2,561 vs the empirically-working 32k. Apply the hybrid
        # attention-layer fraction.
        metadata = {
            'general.architecture': 'nemotron_h_moe',
            'nemotron_h_moe.block_count': 52,
            'nemotron_h_moe.attention.head_count': 32,
            'nemotron_h_moe.attention.head_count_kv': 8,
            'nemotron_h_moe.embedding_length': 4096,
            'nemotron_h_moe.attention.key_length': 128,
            'nemotron_h_moe.attention.value_length': 128,
        }
        model = ModelConfig(id='m', name='M', path='/m.gguf', alias='m')
        with patch('llama_tui.gguf.read_gguf_metadata', return_value=metadata):
            kv_bytes = estimate_kv_bytes_per_token(model)
        dense = 52 * 8 * (128 * 2 + 128 * 2)  # f16 K + f16 V
        expected = int(dense * 0.20 * 1.08)   # nemotron_h_moe fraction
        self.assertEqual(kv_bytes, expected)
        # Sanity: result is dense/5-ish, well below the bare-dense number.
        self.assertLess(kv_bytes, int(dense * 0.30))

    def test_non_hybrid_arch_unaffected(self):
        # An arch not in _HYBRID_SSM_ATTN_FRACTION_BY_ARCH keeps the
        # dense estimate (no accidental regression).
        metadata = {
            'general.architecture': 'llama',
            'llama.block_count': 32,
            'llama.attention.head_count': 32,
            'llama.attention.head_count_kv': 8,
            'llama.embedding_length': 4096,
            'llama.attention.key_length': 128,
            'llama.attention.value_length': 128,
        }
        model = ModelConfig(id='m', name='M', path='/m.gguf', alias='m')
        with patch('llama_tui.gguf.read_gguf_metadata', return_value=metadata):
            kv_bytes = estimate_kv_bytes_per_token(model)
        expected = int(32 * 8 * (128 * 2 + 128 * 2) * 1.08)
        self.assertEqual(kv_bytes, expected)

    def test_mla_falls_back_to_dense_when_lora_rank_missing(self):
        # GQA shape: 32 layers, 32 heads, 8 KV heads, 128 head dim.
        metadata = {
            'general.architecture': 'llama',
            'llama.block_count': 32,
            'llama.attention.head_count': 32,
            'llama.attention.head_count_kv': 8,
            'llama.embedding_length': 4096,
            'llama.attention.key_length': 128,
            'llama.attention.value_length': 128,
        }
        model = ModelConfig(id='m', name='M', path='/m.gguf', alias='m')
        with patch('llama_tui.gguf.read_gguf_metadata', return_value=metadata):
            kv_bytes = estimate_kv_bytes_per_token(model)
        expected = int(32 * 8 * (128 * 2 + 128 * 2) * 1.08)
        self.assertEqual(kv_bytes, expected)


class CacheTypeBytesTests(unittest.TestCase):
    def test_known_types(self):
        self.assertEqual(cache_type_bytes('f16'), 2.0)
        self.assertEqual(cache_type_bytes('q8_0'), 1.0625)
        self.assertEqual(cache_type_bytes('q4_0'), 0.5625)

    def test_unknown_type_defaults_to_f16(self):
        self.assertEqual(cache_type_bytes('made-up'), 2.0)
        self.assertEqual(cache_type_bytes(''), 2.0)


if __name__ == '__main__':
    unittest.main()
