import unittest

from llama_tui.gguf import (
    GGML_TYPE_NAME,
    GGML_TYPE_SIZE,
    _UNKNOWN_GGML_TYPES_SEEN,
    _estimated_tensor_payload_bytes,
    cache_type_bytes,
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
