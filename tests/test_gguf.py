import unittest

from llama_tui.gguf import (
    cache_type_bytes,
    extra_arg_value,
    has_extra_flag,
    selected_cache_type,
    set_model_extra_arg,
    strip_extra_args,
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
