import unittest
from unittest.mock import patch

from llama_tui.launch_profiles import (
    RAW_SPEED_OUTPUT,
    SERVE_DEFAULT_FAST_OUTPUT_CAP,
    SERVE_DEFAULT_FULL_OUTPUT_CAP,
    benchmark_launch_metadata,
    benchmark_profile_request_fields,
    build_benchmark_launch_profile,
)
from llama_tui.models import ModelConfig
from llama_tui.runtime_profiles import RuntimeProfile


class LaunchProfileTests(unittest.TestCase):
    def test_raw_speed_defaults_are_deterministic_and_bounded(self):
        model = ModelConfig(id='tiny', name='Tiny', path='tiny.gguf', alias='tiny', port=18080, output=4096, temp=0.8)

        profile = build_benchmark_launch_profile(model, purpose='raw_speed')

        self.assertEqual(profile.name, 'raw_speed')
        self.assertEqual(profile.output, RAW_SPEED_OUTPUT)
        self.assertEqual(profile.measurement_output, RAW_SPEED_OUTPUT)
        self.assertEqual(profile.temp, 0.0)
        self.assertIsNone(profile.top_p)
        self.assertIsNone(profile.top_k)
        self.assertEqual(profile.repeat_penalty, 1.0)
        self.assertEqual(profile.presence_penalty, 0.0)
        self.assertFalse(profile.no_context_shift)

    def test_serve_default_uses_model_sampling_and_depth_caps(self):
        model = ModelConfig(
            id='tiny',
            name='Tiny',
            path='tiny.gguf',
            alias='tiny',
            port=18080,
            output=4096,
            temp=0.44,
            top_p=0.91,
            top_k=23,
            repeat_penalty=1.07,
            presence_penalty=0.2,
            no_context_shift=True,
        )

        fast = build_benchmark_launch_profile(model, purpose='serve_default', depth='fast')
        full = build_benchmark_launch_profile(model, purpose='serve_default', depth='full')

        self.assertEqual(fast.measurement_output, SERVE_DEFAULT_FAST_OUTPUT_CAP)
        self.assertEqual(full.measurement_output, SERVE_DEFAULT_FULL_OUTPUT_CAP)
        self.assertEqual(full.output, 4096)
        self.assertEqual(full.temp, 0.44)
        self.assertEqual(full.top_p, 0.91)
        self.assertEqual(full.top_k, 23)
        self.assertEqual(full.repeat_penalty, 1.07)
        self.assertEqual(full.presence_penalty, 0.2)
        self.assertTrue(full.no_context_shift)

    def test_config_overrides_adjust_sampling_output_and_extra_args(self):
        model = ModelConfig(
            id='tiny',
            name='Tiny',
            path='tiny.gguf',
            alias='tiny',
            port=18080,
            output=4096,
            launch_overrides={
                'measurement_output': 300,
                'top_p': 0.7,
                'top_k': 12,
                'min_p': 0.05,
                'seed': 42,
                'samplers': 'top_k;top_p',
                'cache_reuse': 128,
                'fit_target': '0.85',
                'extra_args': ['--poll', '500'],
            },
        )

        profile = build_benchmark_launch_profile(model, purpose='serve_default')

        self.assertEqual(profile.measurement_output, 300)
        self.assertEqual(profile.top_p, 0.7)
        self.assertEqual(profile.top_k, 12)
        self.assertEqual(profile.min_p, 0.05)
        self.assertEqual(profile.seed, 42)
        self.assertEqual(profile.samplers, 'top_k;top_p')
        self.assertEqual(profile.cache_reuse, 128)
        self.assertEqual(profile.fit_target, '0.85')
        self.assertEqual(profile.extra_args, ('--poll', '500'))

    def test_preserve_thinking_override_template_family_and_default(self):
        manual_on = ModelConfig(
            id='manual',
            name='Manual',
            path='manual.gguf',
            alias='manual',
            port=18080,
            preserve_thinking='on',
        )
        manual_off = ModelConfig(
            id='manual-off',
            name='QwQ Reasoning',
            path='manual-off.gguf',
            alias='manual-off',
            port=18081,
            preserve_thinking='off',
        )
        template_hint = ModelConfig(id='tpl', name='Plain', path='tpl.gguf', alias='tpl', port=18082)
        family_hint = ModelConfig(id='qwq', name='QwQ 32B', path='qwq.gguf', alias='qwq', port=18083)
        uncertain = ModelConfig(id='plain', name='Plain', path='plain.gguf', alias='plain', port=18084)

        with patch('llama_tui.launch_profiles.read_gguf_template_metadata', return_value={}):
            self.assertTrue(build_benchmark_launch_profile(manual_on).preserve_thinking)
            self.assertFalse(build_benchmark_launch_profile(manual_off).preserve_thinking)
            self.assertTrue(build_benchmark_launch_profile(family_hint).preserve_thinking)
            self.assertFalse(build_benchmark_launch_profile(uncertain).preserve_thinking)

        with patch(
            'llama_tui.launch_profiles.read_gguf_template_metadata',
            return_value={'tokenizer.chat_template': '{{ preserve_thinking }}'},
        ):
            profile = build_benchmark_launch_profile(template_hint)

        self.assertTrue(profile.preserve_thinking)
        self.assertEqual(profile.chat_template_kwargs, {'preserve_thinking': True})

    def test_request_fields_and_metadata_reflect_profile(self):
        model = ModelConfig(id='tiny', name='Tiny', path='tiny.gguf', alias='tiny', port=18080, output=4096)
        runtime = RuntimeProfile(
            engine_id='llama.cpp',
            name='q8',
            ctx_size=8192,
            parallel=1,
            gpu_layers=99,
            kv_preset='q8_0/q8_0',
            flash_attn='on',
            fit=True,
            fit_context=4096,
        )

        profile = build_benchmark_launch_profile(model, runtime, purpose='serve_default', depth='fast')
        payload = benchmark_profile_request_fields(profile)
        metadata = benchmark_launch_metadata(profile, unsupported_flags=['--chat-template-kwargs'])

        self.assertEqual(payload['max_tokens'], SERVE_DEFAULT_FAST_OUTPUT_CAP)
        self.assertEqual(payload['top_p'], 0.95)
        self.assertEqual(metadata['benchmark_profile'], 'serve_default')
        self.assertEqual(metadata['ctx'], 8192)
        self.assertEqual(metadata['kv_key'], 'q8_0')
        self.assertEqual(metadata['kv_value'], 'q8_0')
        self.assertTrue(metadata['fit'])
        self.assertEqual(metadata['fit_context'], 4096)
        self.assertEqual(metadata['unsupported_launch_flags'], ['--chat-template-kwargs'])


if __name__ == '__main__':
    unittest.main()
