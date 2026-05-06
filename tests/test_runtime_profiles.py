import inspect
import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from llama_tui.app import AppConfig
from llama_tui.benchmark import (
    active_engine_runtime_profiles,
    adaptive_record_from_candidate,
    benchmark_adaptive_candidate,
    benchmark_all_models_runner,
    benchmark_completion,
    benchmark_exhaustive_profiles,
    benchmark_fast_profiles,
    benchmark_raw_speed_profile,
    benchmark_run_summary,
    benchmark_runtime_profile_with_retry,
    classify_benchmark_failure,
    launch_with_failsafe,
    measured_profile_runtime_profile,
    memory_guardrail_admission,
    model_and_runtime_profile_from_measured_profile,
    runtime_record_context,
    runtime_profile_memory_disable_key,
    runtime_profile_memory_skip_reason,
    select_measured_profiles,
)
from llama_tui.hardware import HardwareProfile
from llama_tui.launch_profiles import build_benchmark_launch_profile
from llama_tui.main import (
    build_cli_parser,
    ensure_engine_session_lock,
    engine_session_path,
    last_engine_session_stop_count,
    release_engine_session_lock,
    validate_buun_kv_args,
    validate_tq3_kv_args,
    validate_turboquant_kv_args,
)
from llama_tui.models import ModelConfig
from llama_tui.runtime_profiles import (
    EngineCapabilities,
    RuntimeProfile,
    TQ3_KV_MODES,
    default_engine_capabilities,
    make_runtime_profile,
    parse_engine_capabilities,
)


def turboquant_no_fit_capabilities() -> EngineCapabilities:
    return replace(
        default_engine_capabilities('turboquant'),
        supports_fit=False,
        supports_fit_ctx=False,
    )


class RuntimeProfileTests(unittest.TestCase):
    def test_buun_profile_defaults_to_symmetric_turbo4(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('BUUN_LLAMA_SERVER_BIN', None)
            profile = make_runtime_profile('buun', 'llama-server')

        self.assertEqual(profile.server_command, 'buun-llama-server')
        self.assertEqual(profile.llama_extra_args(), ['--flash-attn', 'on', '-ctk', 'turbo4', '-ctv', 'turbo4'])
        self.assertIn('key=turbo4 value=turbo4', profile.header_indicator())

    def test_buun_profile_respects_explicit_server_override(self):
        with patch.dict(os.environ, {'BUUN_LLAMA_SERVER_BIN': '/opt/buun/bin/llama-server'}):
            profile = make_runtime_profile('buun', 'llama-server')

        self.assertEqual(profile.server_command, '/opt/buun/bin/llama-server')

    def test_buun_profile_uses_kv_shorthand_for_both_sides(self):
        profile = make_runtime_profile('buun', 'llama-server', kv_mode='turbo3_tcq')

        self.assertEqual(profile.llama_extra_args(), ['--flash-attn', 'on', '-ctk', 'turbo3_tcq', '-ctv', 'turbo3_tcq'])
        self.assertIn('key=turbo3_tcq value=turbo3_tcq', profile.header_indicator())

    def test_buun_profile_allows_asymmetric_kv_pair(self):
        profile = make_runtime_profile(
            'buun',
            'llama-server',
            kv_mode='turbo4',
            kv_key_mode='turbo3_tcq',
            kv_value_mode='turbo2_tcq',
        )

        self.assertEqual(profile.llama_extra_args(), ['--flash-attn', 'on', '-ctk', 'turbo3_tcq', '-ctv', 'turbo2_tcq'])
        self.assertIn('key=turbo3_tcq value=turbo2_tcq', profile.header_indicator())

    def test_turboquant_profile_defaults_to_q8_baseline(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('TURBOQUANT_LLAMA_SERVER_BIN', None)
            profile = make_runtime_profile('turboquant', 'llama-server')

        self.assertEqual(profile.engine_id, 'turboquant')
        self.assertTrue(profile.supports_turbo_kv)
        self.assertEqual(profile.turboquant_kv_pair(), ('q8_0', 'q8_0'))
        self.assertEqual(profile.llama_extra_args(), ['--flash-attn', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0'])
        self.assertIn('TurboQuant+', profile.header_indicator())

    def test_turboquant_profile_respects_env_override_and_manual_kv(self):
        with patch.dict(os.environ, {'TURBOQUANT_LLAMA_SERVER_BIN': '/opt/tqp/bin/llama-server'}):
            profile = make_runtime_profile(
                'turboquant',
                'llama-server',
                kv_key_mode='q8_0',
                kv_value_mode='turbo4',
            )

        self.assertEqual(profile.server_command, '/opt/tqp/bin/llama-server')
        self.assertEqual(profile.turboquant_kv_pair(), ('q8_0', 'turbo4'))
        self.assertIn('key=q8_0 value=turbo4', profile.header_indicator())

    def test_tq3_profile_defaults_to_q8_and_respects_env_override(self):
        with patch.dict(os.environ, {'TQ3_LLAMA_SERVER_BIN': '/opt/tq3/bin/llama-server'}):
            profile = make_runtime_profile('tq3', 'llama-server')

        self.assertEqual(profile.engine_id, 'tq3')
        self.assertEqual(profile.server_command, '/opt/tq3/bin/llama-server')
        self.assertEqual(profile.tq3_kv_pair(), ('q8_0', 'q8_0'))
        self.assertEqual(profile.llama_extra_args(), ['--flash-attn', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0'])
        self.assertIn('llama.cpp-tq3', profile.header_indicator())

    def test_capability_parser_detects_buun_flash_value_and_ngl(self):
        caps = parse_engine_capabilities(
            'usage: llama-server --flash-attn on|off|auto -ctk MODE -ctv MODE --parallel N -ngl N -fit on -fitc N --no-warmup',
            engine_id='buun',
        )

        self.assertEqual(caps.flash_attn_syntax, 'value')
        self.assertTrue(caps.supports_ctk_ctv)
        self.assertTrue(caps.supports_parallel)
        self.assertTrue(caps.supports_fit)
        self.assertTrue(caps.supports_fit_ctx)
        self.assertTrue(caps.supports_no_warmup)
        self.assertEqual(caps.gpu_layers_flag, '-ngl')

    def test_capability_parser_detects_turboquant_cache_types(self):
        caps = parse_engine_capabilities(
            'KV cache data type for K\nallowed values: f16 q8_0 q4_0 turbo2 turbo3 turbo4\n'
            'KV cache data type for V\nallowed values: f16 q8_0 q4_0 turbo2 turbo3 turbo4\n'
            '--flash-attn on|off|auto -ctk TYPE -ctv TYPE -ngl N --parallel N',
            engine_id='turboquant',
        )

        self.assertTrue(caps.supports_ctk_ctv)
        self.assertIn('turbo4', caps.supported_kv_modes)
        self.assertEqual(caps.gpu_layers_flag, '-ngl')

    def test_capability_parser_detects_tq3_cache_type(self):
        caps = parse_engine_capabilities(
            'KV cache data type for K\nallowed values: q8_0 tq3_0\n'
            'KV cache data type for V\nallowed values: q8_0 tq3_0\n'
            '--flash-attn on|off|auto -ctk TYPE -ctv TYPE -ngl N --parallel N',
            engine_id='tq3',
        )

        self.assertTrue(caps.supports_ctk_ctv)
        self.assertEqual(caps.supported_kv_modes, TQ3_KV_MODES)
        self.assertEqual(caps.gpu_layers_flag, '-ngl')

    def test_capability_parser_detects_llama_cache_type_flags(self):
        caps = parse_engine_capabilities(
            'usage: llama-server --flash-attn on|off|auto --cache-type-k TYPE --cache-type-v TYPE --n-gpu-layers N --parallel N',
            engine_id='llama.cpp',
        )

        self.assertEqual(caps.flash_attn_syntax, 'value')
        self.assertTrue(caps.supports_cache_type_kv)
        self.assertEqual(caps.gpu_layers_flag, '--n-gpu-layers')

    def test_capability_parser_detects_benchmark_relevant_flags(self):
        caps = parse_engine_capabilities(
            'usage: llama-server --chat-template-kwargs JSON --reasoning auto --reasoning-budget N '
            '--context-shift --no-context-shift --cache-prompt --cache-reuse N --fit-target R '
            '--top-p P --top-k K --min-p P --repeat-penalty N --presence-penalty N --samplers LIST --seed SEED',
            engine_id='llama.cpp',
        )

        self.assertTrue(caps.supports_chat_template_kwargs)
        self.assertTrue(caps.supports_reasoning)
        self.assertTrue(caps.supports_reasoning_budget)
        self.assertTrue(caps.supports_context_shift)
        self.assertTrue(caps.supports_cache_prompt)
        self.assertTrue(caps.supports_cache_reuse)
        self.assertTrue(caps.supports_fit_target)
        self.assertTrue(caps.supports_top_p)
        self.assertTrue(caps.supports_top_k)
        self.assertTrue(caps.supports_min_p)
        self.assertTrue(caps.supports_repeat_penalty)
        self.assertTrue(caps.supports_presence_penalty)
        self.assertTrue(caps.supports_samplers)
        self.assertTrue(caps.supports_seed)

    def test_buun_command_uses_value_flash_and_strips_generic_cache_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='qwen',
                name='Qwen',
                path='/models/qwen.gguf',
                alias='qwen36-buun',
                port=18080,
                turboquant_status='native',
                turboquant_key_dim=256,
                turboquant_value_dim=256,
                extra_args=['-fa', '-ctk', 'turbo4', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0'],
            )
            caps = EngineCapabilities(
                flash_attn_syntax='value',
                flash_attn_flag='--flash-attn',
                supports_ctk_ctv=True,
                supports_cache_type_kv=True,
                supports_parallel=True,
                gpu_layers_flag='-ngl',
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertIn('--flash-attn', cmd)
        self.assertIn('on', cmd)
        self.assertIn('-ctk', cmd)
        self.assertIn('turbo4', cmd)
        self.assertEqual(cmd[0], 'buun-llama-server')
        self.assertNotIn('-fa', cmd)
        self.assertNotIn('--cache-type-k', cmd)
        self.assertNotIn('--cache-type-v', cmd)
        self.assertEqual(cmd[cmd.index('--flash-attn') + 1], 'on')

    def test_command_builder_adds_supported_benchmark_profile_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='tiny',
                name='Tiny Reasoning',
                path='/models/tiny.gguf',
                alias='tiny',
                port=18080,
                temp=0.33,
                top_p=0.8,
                top_k=24,
                repeat_penalty=1.08,
                presence_penalty=0.1,
                no_context_shift=True,
                preserve_thinking='on',
                extra_args=['--top-p', '0.1', '--no-context-shift', '--seed', '999'],
                launch_overrides={
                    'min_p': 0.05,
                    'seed': 123,
                    'samplers': 'top_k;top_p',
                    'reasoning': 'auto',
                    'reasoning_budget': 256,
                    'cache_prompt': True,
                    'cache_reuse': 64,
                    'fit_target': '0.85',
                },
            )
            caps = EngineCapabilities(
                flash_attn_syntax='value',
                flash_attn_flag='--flash-attn',
                supports_cache_type_kv=True,
                supports_parallel=True,
                supports_context_shift=True,
                supports_chat_template_kwargs=True,
                supports_reasoning=True,
                supports_reasoning_budget=True,
                supports_cache_prompt=True,
                supports_cache_reuse=True,
                supports_fit_target=True,
                supports_top_p=True,
                supports_top_k=True,
                supports_min_p=True,
                supports_repeat_penalty=True,
                supports_presence_penalty=True,
                supports_samplers=True,
                supports_seed=True,
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertEqual(cmd[cmd.index('--temp') + 1], '0.33')
        self.assertEqual(cmd.count('--top-p'), 1)
        self.assertEqual(cmd.count('--no-context-shift'), 1)
        self.assertEqual(cmd.count('--seed'), 1)
        self.assertEqual(cmd[cmd.index('--top-p') + 1], '0.8')
        self.assertEqual(cmd[cmd.index('--top-k') + 1], '24')
        self.assertEqual(cmd[cmd.index('--min-p') + 1], '0.05')
        self.assertEqual(cmd[cmd.index('--repeat-penalty') + 1], '1.08')
        self.assertEqual(cmd[cmd.index('--presence-penalty') + 1], '0.1')
        self.assertIn('--no-context-shift', cmd)
        self.assertIn('--cache-prompt', cmd)
        self.assertEqual(cmd[cmd.index('--cache-reuse') + 1], '64')
        self.assertEqual(cmd[cmd.index('--reasoning') + 1], 'auto')
        self.assertEqual(cmd[cmd.index('--reasoning-budget') + 1], '256')
        self.assertEqual(cmd[cmd.index('-fitt') + 1], '0.85')
        self.assertIn('--chat-template-kwargs', cmd)
        self.assertIn('"preserve_thinking": true', cmd[cmd.index('--chat-template-kwargs') + 1])

    def test_command_builder_omits_unsupported_profile_flags_and_strips_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='tiny',
                name='Tiny',
                path='/models/tiny.gguf',
                alias='tiny',
                port=18080,
                top_p=0.8,
                no_context_shift=True,
                preserve_thinking='on',
                extra_args=['--top-p', '0.1', '--no-context-shift'],
            )
            caps = EngineCapabilities(
                flash_attn_syntax='unsupported',
                supports_cache_type_kv=True,
                supports_parallel=True,
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertIn('--temp', cmd)
        self.assertNotIn('--top-p', cmd)
        self.assertNotIn('--no-context-shift', cmd)
        self.assertNotIn('--chat-template-kwargs', cmd)

    def test_vllm_command_is_not_polluted_with_llama_server_profile_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            app.vllm_command = 'vllm'
            model = ModelConfig(
                id='tiny',
                name='Tiny',
                path='/models/tiny',
                alias='tiny',
                port=18080,
                runtime='vllm',
                top_p=0.8,
                no_context_shift=True,
                preserve_thinking='on',
            )
            profile = build_benchmark_launch_profile(model, purpose='serve_default')

            cmd = app.build_command(model, benchmark_profile=profile)

        self.assertEqual(cmd[:2], ['vllm', 'serve'])
        self.assertNotIn('--temp', cmd)
        self.assertNotIn('--top-p', cmd)
        self.assertNotIn('--no-context-shift', cmd)
        self.assertNotIn('--chat-template-kwargs', cmd)

    def test_turboquant_command_uses_short_cache_flags_for_manual_safe_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile(
                    'turboquant',
                    'llama-server',
                    kv_key_mode='q8_0',
                    kv_value_mode='turbo4',
                ),
            )
            model = ModelConfig(
                id='qwen',
                name='Qwen Q8_0',
                path='/models/qwen.Q8_0.gguf',
                alias='qwen-tq',
                port=18080,
                turboquant_status='native',
                turboquant_key_dim=128,
                turboquant_value_dim=128,
            )

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('turboquant')):
                cmd = app.build_command(model)

        self.assertEqual(cmd[0], app.runtime_profile.server_command)
        self.assertEqual(cmd[cmd.index('-ctk') + 1], 'q8_0')
        self.assertEqual(cmd[cmd.index('-ctv') + 1], 'turbo4')
        self.assertNotIn('--cache-type-k', cmd)

    def test_tq3_command_uses_short_cache_flags_and_q8_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('tq3', 'llama-server'),
            )
            model = ModelConfig(
                id='qwen-tq3',
                name='Qwen TQ3_4S',
                path='/models/qwen.TQ3_4S.gguf',
                alias='qwen-tq3',
                port=18080,
                tq3_status='native',
                tq3_weight_format='TQ3_4S',
            )

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('tq3')):
                cmd = app.build_command(model)

        self.assertEqual(cmd[0], app.runtime_profile.server_command)
        self.assertEqual(cmd[cmd.index('-ctk') + 1], 'q8_0')
        self.assertEqual(cmd[cmd.index('-ctv') + 1], 'q8_0')
        self.assertNotIn('--cache-type-k', cmd)

    def test_turboquant_head_dim_64_forces_q8_baseline_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile(
                    'turboquant',
                    'llama-server',
                    kv_key_mode='q8_0',
                    kv_value_mode='turbo4',
                ),
            )
            model = ModelConfig(
                id='oss',
                name='GPT OSS',
                path='/models/gpt-oss.gguf',
                alias='oss-tq',
                port=18080,
                turboquant_status='incompatible',
                turboquant_head_dim=64,
                turboquant_key_dim=64,
                turboquant_value_dim=64,
            )

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('turboquant')):
                cmd = app.build_command(model)

        self.assertEqual(cmd[cmd.index('-ctk') + 1], 'q8_0')
        self.assertEqual(cmd[cmd.index('-ctv') + 1], 'q8_0')
        self.assertIn('head_dim=64', app.turboquant_session_advisory(model))

    def test_turboquant_binary_warning_for_vanilla_help(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(id='m', name='M', path='/models/m.gguf', alias='m', port=18080)
            caps = EngineCapabilities(
                supports_ctk_ctv=True,
                supports_cache_type_kv=True,
                supported_kv_modes=('f16', 'q8_0', 'q4_0'),
                help_text='allowed values: f16 q8_0 q4_0',
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                warning = app.turboquant_binary_warning(model)

        self.assertIn('does not advertise turbo cache types', warning)

    def test_buun_command_omits_turbokv_for_incompatible_model_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='gpt-oss',
                name='GPT OSS',
                path='/models/gpt-oss.gguf',
                alias='gpt-oss-buun',
                port=18080,
                turboquant_status='incompatible',
                turboquant_key_dim=64,
                turboquant_value_dim=64,
            )
            caps = default_engine_capabilities('buun')

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertEqual(cmd[0], 'buun-llama-server')
        self.assertNotIn('-ctk', cmd)
        self.assertNotIn('-ctv', cmd)
        self.assertIn('--flash-attn', cmd)

    def test_buun_try_out_launch_path_uses_buun_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / 'model.gguf'
            model_path.write_bytes(b'GGUF')
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop('BUUN_LLAMA_SERVER_BIN', None)
                app = AppConfig(
                    root / 'models.json',
                    runtime_profile=make_runtime_profile('buun', 'llama-server'),
                )
            model = ModelConfig(
                id='m',
                name='M',
                path=str(model_path),
                alias='m',
                port=18080,
                ctx=4096,
                ctx_min=2048,
                ctx_max=4096,
                turboquant_status='native',
                turboquant_key_dim=128,
                turboquant_value_dim=128,
            )
            hardware = HardwareProfile(
                cpu_logical=8,
                cpu_physical=4,
                memory_total=64 * 1024**3,
                memory_available=48 * 1024**3,
            )
            commands = []

            class FakeProcess:
                pid = 4242

            def fake_popen(command, *args, **kwargs):
                commands.append(command)
                return FakeProcess()

            with patch.object(app, 'hardware_profile', return_value=hardware), \
                patch.object(app, 'command_exists', return_value=True), \
                patch.object(app, 'runtime_command_ready', return_value=(True, '')), \
                patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('buun')), \
                patch.object(app, 'enrich_model_turboquant', return_value=False), \
                patch.object(app, 'get_pid', return_value=None), \
                patch.object(app, 'wait_until_ready', return_value=(True, 'ready')), \
                patch.object(app, 'logfile', side_effect=lambda model_id: root / f'{model_id}.log'), \
                patch.object(app, 'pidfile', side_effect=lambda model_id: root / f'{model_id}.pid'), \
                patch.object(app, 'pid_metadata_file', side_effect=lambda model_id: root / f'{model_id}.pid.json'), \
                patch('llama_tui.optimize.process_pressure_score', return_value=0.0), \
                patch('llama_tui.app.subprocess.Popen', side_effect=fake_popen):
                ok, _msg = launch_with_failsafe(app, model, 'best', 'auto')

        self.assertTrue(ok)
        self.assertTrue(commands)
        cmd = commands[0]
        self.assertEqual(cmd[0], 'buun-llama-server')
        self.assertIn('-ctk', cmd)
        self.assertIn('-ctv', cmd)
        self.assertEqual(cmd[cmd.index('-ctk') + 1], 'turbo4')
        self.assertEqual(cmd[cmd.index('-ctv') + 1], 'turbo4')

    def test_measured_buun_profile_replays_fit_runtime_metadata(self):
        fingerprint = {
            'engine_id': 'buun',
            'runtime_profile': 'fit_context_growth_sweep_32768_turbo4_turbo4',
            'gpu_layers': None,
            'kv_preset': 'turbo4/turbo4',
            'flash_attn': 'on',
            'batch_size': 128,
            'ubatch_size': 64,
            'fit': True,
            'fit_context': 4096,
            'no_warmup': True,
        }
        model = ModelConfig(
            id='m',
            name='M',
            path='/models/m.gguf',
            alias='m',
            port=18200,
            measured_profiles={
                'opencode_ready': {
                    'status': 'ok',
                    'ctx': 32768,
                    'ctx_per_slot': 32768,
                    'parallel': 1,
                    'ngl': 999,
                    'tokens_per_sec': 20.0,
                    'engine': 'buun',
                    'runtime_profile': 'fit_context_growth_sweep_32768_turbo4_turbo4',
                    'kv_preset': 'turbo4/turbo4',
                    'runtime_fit': True,
                    'fit_context': 4096,
                    'runtime_no_warmup': True,
                    'gpu_layers_mode': 'fit',
                    'batch_size': 128,
                    'ubatch_size': 64,
                    'config_fingerprint': json.dumps(fingerprint),
                }
            },
        )

        candidate, runtime_profile = model_and_runtime_profile_from_measured_profile(model, 'opencode_ready')
        direct_runtime = measured_profile_runtime_profile(model, 'opencode_ready')

        self.assertIsNotNone(candidate)
        self.assertIsNotNone(runtime_profile)
        self.assertIsNotNone(direct_runtime)
        self.assertEqual(candidate.ctx, 32768)
        self.assertTrue(runtime_profile.fit)
        self.assertIsNone(runtime_profile.gpu_layers)
        self.assertEqual(runtime_profile.fit_context, 4096)
        self.assertTrue(runtime_profile.no_warmup)
        self.assertEqual(runtime_profile.kv_preset, 'turbo4/turbo4')
        self.assertEqual(runtime_profile.batch_size, 128)
        self.assertEqual(runtime_profile.ubatch_size, 64)

    def test_launch_with_failsafe_starts_measured_profile_with_runtime_replay(self):
        model = ModelConfig(
            id='m',
            name='M',
            path='/models/m.gguf',
            alias='m',
            port=18200,
            measured_profiles={
                'auto': {
                    'status': 'ok',
                    'ctx': 32768,
                    'ctx_per_slot': 32768,
                    'parallel': 1,
                    'ngl': 999,
                    'tokens_per_sec': 20.0,
                    'engine': 'buun',
                    'runtime_profile': 'fit_context_growth_sweep_32768_turbo4_turbo4',
                    'kv_preset': 'turbo4/turbo4',
                    'runtime_fit': True,
                    'fit_context': 4096,
                    'runtime_no_warmup': True,
                    'gpu_layers_mode': 'fit',
                    'batch_size': 128,
                    'ubatch_size': 64,
                }
            },
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.started_runtime_profiles = []
                self.saved = []

            def hardware_profile(self, refresh=False):
                return HardwareProfile(memory_available=32 * 1024**3)

            def add_or_update(self, saved):
                self.saved.append(saved)

            def start(self, _model, runtime_profile=None, benchmark_profile=None):
                self.started_runtime_profiles.append(runtime_profile)
                return True, 'started'

            def wait_until_ready(self, _model, timeout=120, cancel_token=None):
                return True, 'ready'

            def stop(self, _model, managed_only=True):
                return True, 'stopped'

        app = FakeApp()
        ok, msg = launch_with_failsafe(app, model, 'best', 'auto')

        self.assertTrue(ok, msg)
        self.assertTrue(app.started_runtime_profiles)
        runtime_profile = app.started_runtime_profiles[0]
        self.assertIsNotNone(runtime_profile)
        self.assertTrue(runtime_profile.fit)
        self.assertIsNone(runtime_profile.gpu_layers)
        self.assertEqual(runtime_profile.fit_context, 4096)

    def test_runtime_artifacts_are_scoped_by_active_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = ModelConfig(id='m', name='M', path='/models/m.gguf', alias='m', port=18080)
            with patch('llama_tui.app.CACHE_DIR', root):
                llama_app = AppConfig(root / 'llama.json')
                llama_app.models = [model]
                buun_app = AppConfig(
                    root / 'buun.json',
                    runtime_profile=make_runtime_profile('buun', 'llama-server'),
                )
                buun_app.models = [model]
                turboquant_app = AppConfig(
                    root / 'turboquant.json',
                    runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
                )
                turboquant_app.models = [model]

                llama_log = llama_app.logfile(model.id)
                buun_log = buun_app.logfile(model.id)
                turboquant_log = turboquant_app.logfile(model.id)
                llama_pid = llama_app.pidfile(model.id)
                buun_pid = buun_app.pidfile(model.id)
                turboquant_pid = turboquant_app.pidfile(model.id)
                legacy_log = buun_app.legacy_logfile(model.id)

        self.assertNotEqual(llama_log, buun_log)
        self.assertNotEqual(llama_log, turboquant_log)
        self.assertNotEqual(llama_pid, buun_pid)
        self.assertNotEqual(llama_pid, turboquant_pid)
        self.assertEqual(llama_log, root / 'runtime' / 'llama.cpp' / 'm.log')
        self.assertEqual(buun_log, root / 'runtime' / 'buun' / 'm.log')
        self.assertEqual(turboquant_log, root / 'runtime' / 'turboquant' / 'm.log')
        self.assertEqual(llama_pid, root / 'runtime' / 'llama.cpp' / 'm.pid')
        self.assertEqual(buun_pid, root / 'runtime' / 'buun' / 'm.pid')
        self.assertEqual(turboquant_pid, root / 'runtime' / 'turboquant' / 'm.pid')
        self.assertEqual(legacy_log, root / 'm.log')

    def test_llama_command_can_use_supported_q8_cache_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='dense',
                name='Dense',
                path='/models/dense.gguf',
                alias='dense',
                port=18080,
                extra_args=['--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0'],
            )
            caps = EngineCapabilities(
                flash_attn_syntax='value',
                flash_attn_flag='--flash-attn',
                supports_cache_type_kv=True,
                supports_parallel=True,
                gpu_layers_flag='--n-gpu-layers',
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertIn('--flash-attn', cmd)
        self.assertIn('on', cmd)
        self.assertIn('--cache-type-k', cmd)
        self.assertIn('--cache-type-v', cmd)
        self.assertNotIn('-ctk', cmd)

    def test_continue_tool_export_forces_jinja_for_llamacpp_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='dense',
                name='Dense',
                path='/models/dense.gguf',
                alias='dense',
                port=18080,
                jinja=False,
            )
            caps = default_engine_capabilities('llama.cpp')

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertIn('--jinja', cmd)

    def test_continue_tool_export_forces_jinja_for_buun_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='qwen',
                name='Qwen',
                path='/models/qwen.gguf',
                alias='qwen',
                port=18080,
                jinja=False,
            )
            caps = default_engine_capabilities('buun')

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertIn('--jinja', cmd)

    def test_continue_tool_export_does_not_add_jinja_to_vllm_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='vllm',
                name='vLLM',
                path='org/model',
                alias='vllm-model',
                port=18080,
                runtime='vllm',
                jinja=False,
                extra_args=['--trust-remote-code'],
            )

            cmd = app.build_command(model)

        self.assertNotIn('--jinja', cmd)
        self.assertIn('--trust-remote-code', cmd)

    def test_continue_tool_export_preserves_template_override_without_chatml_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='dense',
                name='Dense',
                path='/models/dense.gguf',
                alias='dense',
                port=18080,
                jinja=False,
                extra_args=['--chat-template-file', '/models/tool-template.jinja'],
            )
            caps = default_engine_capabilities('llama.cpp')

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertIn('--jinja', cmd)
        self.assertIn('--chat-template-file', cmd)
        self.assertIn('/models/tool-template.jinja', cmd)
        self.assertNotIn('chatml', cmd)

    def test_runtime_profile_command_accepts_known_working_buun_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'buun-llama-server'),
            )
            model = ModelConfig(id='qwen', name='Qwen', path='$MODEL', alias='qwen36-buun', port=18080)
            profile = RuntimeProfile(
                engine_id='buun',
                name='kv_compression_probe',
                ctx_size=8192,
                gpu_layers=20,
                parallel=1,
                kv_preset='turbo4/turbo4',
                flash_attn='on',
            )
            caps = EngineCapabilities(
                flash_attn_syntax='value',
                flash_attn_flag='--flash-attn',
                supports_ctk_ctv=True,
                supports_parallel=True,
                gpu_layers_flag='-ngl',
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model, runtime_profile=profile)

        self.assertEqual(cmd[0], 'buun-llama-server')
        self.assertIn('--ctx-size', cmd)
        self.assertIn('8192', cmd)
        self.assertIn('-ngl', cmd)
        self.assertIn('20', cmd)
        self.assertIn('--parallel', cmd)
        self.assertIn('--flash-attn', cmd)
        self.assertIn('-ctk', cmd)
        self.assertIn('-ctv', cmd)

    def test_buun_fit_runtime_profile_omits_fixed_ngl_and_disables_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'buun-llama-server'),
            )
            model = ModelConfig(id='gemma', name='Gemma', path='$MODEL', alias='gemma-buun', port=18080)
            profile = RuntimeProfile(
                engine_id='buun',
                name='fit_turbokv_probe',
                ctx_size=8192,
                gpu_layers=None,
                parallel=1,
                kv_preset='turbo4/turbo4',
                flash_attn='on',
                fit=True,
                fit_context=4096,
                no_warmup=True,
            )
            caps = default_engine_capabilities('buun')

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model, runtime_profile=profile)

        self.assertEqual(cmd[0], 'buun-llama-server')
        self.assertNotIn('-ngl', cmd)
        self.assertIn('-fit', cmd)
        self.assertEqual(cmd[cmd.index('-fit') + 1], 'on')
        self.assertIn('-fitc', cmd)
        self.assertEqual(cmd[cmd.index('-fitc') + 1], '4096')
        self.assertIn('--no-warmup', cmd)
        self.assertIn('-ctk', cmd)
        self.assertIn('-ctv', cmd)

    def test_turboquant_fit_runtime_profile_omits_fixed_ngl(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(id='dense', name='Dense', path='$MODEL', alias='dense-tq', port=18080)
            profile = RuntimeProfile(
                engine_id='turboquant',
                name='fit_weight_discovery_q8_0_turbo4',
                ctx_size=8192,
                gpu_layers=None,
                parallel=1,
                kv_preset='q8_0/turbo4',
                flash_attn='on',
                fit=True,
                fit_context=4096,
                fit_discovery_phase='weight_fit',
            )

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('turboquant')):
                cmd = app.build_command(model, runtime_profile=profile)

        self.assertEqual(cmd[0], app.runtime_profile.server_command)
        self.assertNotIn('-ngl', cmd)
        self.assertNotIn('--n-gpu-layers', cmd)
        self.assertIn('-fit', cmd)
        self.assertEqual(cmd[cmd.index('-fit') + 1], 'on')
        self.assertIn('-fitc', cmd)
        self.assertEqual(cmd[cmd.index('-fitc') + 1], '4096')
        self.assertEqual(cmd[cmd.index('-ctk') + 1], 'q8_0')
        self.assertEqual(cmd[cmd.index('-ctv') + 1], 'turbo4')

    def test_failure_classification_names_actionable_startup_errors(self):
        cases = {
            'unknown value for --flash-attn: -ctk': 'CLI_INVALID',
            'cudaMalloc failed: out of memory while loading tensors': 'CUDA_OOM_WEIGHTS',
            'cudaMalloc failed: out of memory allocating KV cache': 'CUDA_OOM_KV',
            'K cache type turbo4 with block size 128 does not divide': 'KV_MODE_INCOMPATIBLE',
            'failed to fit params to free device memory, n_gpu_layers already set by user to 21': 'FIXED_GPU_LAYERS_FIT_FAILED',
            'llama_params_fit_impl: projected to use 9879 MiB of device memory vs. 7665 MiB of free device memory; cannot meet free memory target of 1024 MiB': 'MEMORY_FIT_FAILED',
            'failed to allocate buffer for kv cache; failed to create context': 'CUDA_OOM_KV',
            'ggml-cpu/ops.cpp:4443: fatal error in ggml_compute_forward_scale': 'BUUN_CPU_WARMUP_ABORT',
            'failed to load model': 'MODEL_LOAD_FAILED',
            'server timed out': 'SERVER_TIMEOUT',
            'request timed out': 'API_TIMEOUT',
            'connection refused': 'PORT_UNREACHABLE',
            'chat template error': 'CHAT_TEMPLATE_ERROR',
        }
        mixed_buun_fit_oom = (
            'llama_params_fit: failed to fit params to free device memory: '
            'n_gpu_layers already set by user to 21, abort\n'
            'ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory\n'
            'llama_model_load: failed to load model'
        )
        cases[mixed_buun_fit_oom] = 'FIXED_GPU_LAYERS_FIT_FAILED'
        observed_fixed_fit_failure = (
            'llama_params_fit: failed to fit params to free device memory: '
            'n_gpu_layers already set by user to 18, abort\n'
            'ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory\n'
            'llama_model_load_from_file_impl: failed to load model'
        )
        cases[observed_fixed_fit_failure] = 'FIXED_GPU_LAYERS_FIT_FAILED'
        observed_weight_oom = (
            'ggml_backend_cuda_buffer_type_alloc_buffer: allocating 512.00 MiB on device 0: '
            'cudaMalloc failed: out of memory\n'
            'alloc_tensor_range: failed to allocate CUDA0 buffer\n'
            'llama_model_load: error loading model: unable to allocate CUDA0 buffer\n'
            'llama_model_load_from_file_impl: failed to load model'
        )
        cases[observed_weight_oom] = 'CUDA_OOM_WEIGHTS'
        for text, expected in cases.items():
            with self.subTest(text=text):
                default = 'API_TIMEOUT' if text == 'request timed out' else 'SERVER_TIMEOUT'
                self.assertEqual(classify_benchmark_failure(text, default)['failure_category'], expected)

        classified = classify_benchmark_failure(observed_weight_oom)
        self.assertIn('cudaMalloc failed', classified['failure_excerpt'])
        self.assertEqual(classified['failure_category'], 'CUDA_OOM_WEIGHTS')

    def test_memory_guardrail_admission_skips_critical_memory(self):
        model = ModelConfig(id='m', name='M', path='/models/model.gguf', alias='m', port=18200, ctx=65536)
        profile = HardwareProfile(
            memory_total=16 * 1024**3,
            memory_available=512 * 1024**2,
            gpu_memory_total=8 * 1024**3,
            gpu_memory_free=7 * 1024**3,
        )

        decision = memory_guardrail_admission(profile, model, estimated_safe_ctx=32768)

        self.assertTrue(decision.should_skip)
        self.assertEqual(decision.status, 'memory_guardrail_skipped')
        self.assertIn('RAM available', decision.reason)

    def test_runtime_memory_pruning_skips_same_or_larger_risky_shapes(self):
        failed_profile = RuntimeProfile(
            engine_id='llama.cpp',
            name='context_growth_sweep_32768',
            ctx_size=32768,
            gpu_layers=30,
            parallel=1,
            kv_preset='q8_0/q8_0',
        )
        failed = {'failure_category': 'CUDA_OOM_KV'}
        key = runtime_profile_memory_disable_key(failed, failed_profile)
        larger_same_shape = RuntimeProfile(
            engine_id='llama.cpp',
            name='context_growth_sweep_65536',
            ctx_size=65536,
            gpu_layers=30,
            parallel=1,
            kv_preset='q8_0/q8_0',
        )
        safer_default = RuntimeProfile(
            engine_id='llama.cpp',
            name='context_growth_sweep_65536_default',
            ctx_size=65536,
            gpu_layers=30,
            parallel=1,
            kv_preset='default',
        )

        self.assertTrue(runtime_profile_memory_skip_reason(larger_same_shape, {key}))
        self.assertEqual(runtime_profile_memory_skip_reason(safer_default, {key}), '')

    def test_adaptive_candidate_watchdog_stops_managed_candidate_on_critical_memory(self):
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=8192, ctx_min=2048, ctx_max=8192)

        class FakeApp:
            def __init__(self):
                self.samples = 0
                self.stops = 0

            def hardware_profile(self, refresh=False):
                self.samples += 1
                if self.samples == 1:
                    return HardwareProfile(
                        memory_total=16 * 1024**3,
                        memory_available=12 * 1024**3,
                        gpu_memory_total=8 * 1024**3,
                        gpu_memory_free=7 * 1024**3,
                    )
                return HardwareProfile(
                    memory_total=16 * 1024**3,
                    memory_available=512 * 1024**2,
                    gpu_memory_total=8 * 1024**3,
                    gpu_memory_free=7 * 1024**3,
                )

            def build_command(self, _model, runtime_profile=None, benchmark_profile=None):
                return ['llama-server']

            def start(self, _model, runtime_profile=None, benchmark_profile=None):
                return True, 'started'

            def wait_until_ready(self, _model, timeout=180, cancel_token=None):
                time.sleep(1.2)
                return False, 'not ready'

            def stop(self, _model, managed_only=True):
                self.stops += 1
                return True, 'stopped'

        app = FakeApp()
        record, measured = benchmark_adaptive_candidate(app, model, 'long_context', None, None)

        self.assertIsNone(measured)
        self.assertEqual(record['failure_category'], 'MEMORY_GUARDRAIL')
        self.assertEqual(record['memory_guardrail_status'], 'memory_guardrail_stopped')
        self.assertGreaterEqual(app.stops, 1)

    def test_buun_heavy_moe_profiles_use_fit_only_turbokv_from_traits(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='qwen',
                name='Qwen3.6 35B A3B',
                path='/models/qwen.gguf',
                alias='qwen',
                port=18080,
                architecture='qwen35moe',
                architecture_type='moe',
                expert_count=256,
                expert_used_count=8,
                turboquant_status='native',
                turboquant_key_dim=512,
                turboquant_value_dim=512,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('buun')):
                with patch('llama_tui.benchmark.model_file_size', return_value=int(11.44 * 1024**3)):
                    profiles = active_engine_runtime_profiles(app, model, hardware)

        fit_probe = profiles[0]
        self.assertEqual(fit_probe.name, 'fit_turbokv_probe')
        self.assertEqual(fit_probe.ctx_size, 8192)
        self.assertIsNone(fit_probe.gpu_layers)
        self.assertTrue(fit_probe.fit)
        self.assertTrue(fit_probe.no_warmup)
        self.assertEqual(fit_probe.kv_preset, 'turbo4/turbo4')
        self.assertTrue(all(item.fit for item in profiles))
        self.assertTrue(all(item.gpu_layers is None for item in profiles))
        self.assertFalse(any(item.name == 'partial_gpu_probe' for item in profiles))
        self.assertFalse(any(item.name.startswith('gpu_layer_sweep') for item in profiles))
        turbo_probe = next(item for item in profiles if item.name == 'fit_kv_compression_probe_turbo3_tcq_turbo3_tcq')
        self.assertIsNone(turbo_probe.gpu_layers)
        self.assertIn(32768, {item.ctx_size for item in profiles})

    def test_buun_dense_profiles_include_fit_context_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='dense',
                name='Dense',
                path='/models/dense.gguf',
                alias='dense',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                turboquant_status='native',
                turboquant_key_dim=128,
                turboquant_value_dim=128,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('buun')):
                with patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                    profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        growth = [item for item in profiles if item.name.startswith('fit_context_growth_sweep_')]
        self.assertTrue(any(item.ctx_size >= 16384 for item in growth))
        self.assertTrue(all(item.fit and item.gpu_layers is None for item in growth))
        self.assertTrue(any(item.name == 'partial_gpu_probe' for item in profiles))

    def test_buun_dense_health_based_growth_can_exceed_32k(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='dense',
                name='Dense',
                path='/models/dense.gguf',
                alias='dense',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                turboquant_status='native',
                turboquant_key_dim=128,
                turboquant_value_dim=128,
                ctx_max=131072,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('buun')), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3), \
                patch('llama_tui.benchmark.current_process_pressure_payload', return_value={'process_pressure_score': 0.1}):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        growth_contexts = {item.ctx_size for item in profiles if item.name.startswith('fit_context_growth_sweep_')}
        self.assertGreater(max(growth_contexts), 32768)
        self.assertIn(131072, growth_contexts)

    def test_buun_moe_health_based_growth_can_exceed_32k(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='moe',
                name='MoE',
                path='/models/moe.gguf',
                alias='moe',
                port=18080,
                architecture_type='moe',
                expert_count=64,
                expert_used_count=8,
                turboquant_status='native',
                turboquant_key_dim=256,
                turboquant_value_dim=256,
                ctx_max=131072,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('buun')), \
                patch('llama_tui.benchmark.model_file_size', return_value=12 * 1024**3), \
                patch('llama_tui.benchmark.current_process_pressure_payload', return_value={'process_pressure_score': 0.1}):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        growth_contexts = {item.ctx_size for item in profiles if item.name.startswith('fit_context_growth_sweep_')}
        self.assertGreater(max(growth_contexts), 32768)
        self.assertIn(131072, growth_contexts)

    def test_buun_context_growth_caps_at_safe_estimate_under_pressure(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='dense',
                name='Dense',
                path='/models/dense.gguf',
                alias='dense',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                turboquant_status='native',
                turboquant_key_dim=128,
                turboquant_value_dim=128,
                ctx_max=131072,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('buun')), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3), \
                patch('llama_tui.benchmark.current_process_pressure_payload', return_value={'process_pressure_score': 0.7}), \
                patch('llama_tui.benchmark.candidate_safe_context_estimate', return_value=65536):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        growth_contexts = {item.ctx_size for item in profiles if item.name.startswith('fit_context_growth_sweep_')}
        self.assertIn(65536, growth_contexts)
        self.assertLessEqual(max(growth_contexts), 65536)

    def test_buun_fast_profiles_use_curated_turbokv_ladder(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='moe',
                name='MoE',
                path='/models/model.gguf',
                alias='moe',
                port=18080,
                architecture_type='moe',
                expert_count=64,
                expert_used_count=8,
                turboquant_status='native',
                turboquant_key_dim=256,
                turboquant_value_dim=256,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('buun')):
                with patch('llama_tui.benchmark.model_file_size', return_value=12 * 1024**3):
                    profiles = active_engine_runtime_profiles(app, model, hardware, depth='fast')

        presets = {item.kv_preset for item in profiles}
        self.assertIn('turbo4/turbo4', presets)
        self.assertIn('turbo3_tcq/turbo3_tcq', presets)
        self.assertIn('turbo3_tcq/turbo2_tcq', presets)
        self.assertNotIn('turbo2_tcq/turbo2_tcq', presets)
        self.assertNotIn('turbo3/turbo3', presets)
        self.assertNotIn('turbo2/turbo2', presets)
        self.assertTrue(all(item.benchmark_depth == 'fast' for item in profiles))
        self.assertTrue(all(item.fit for item in profiles))
        self.assertTrue(all(item.gpu_layers is None for item in profiles))
        self.assertTrue(any(item.name.startswith('fit_context_growth_sweep_16384') for item in profiles))
        names = [item.name for item in profiles]
        self.assertLess(names.index('fit_turbokv_probe'), names.index('fit_default_probe'))

    def test_buun_fit_profiles_include_default_fallback_after_turbokv(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='moe',
                name='MoE',
                path='/models/model.gguf',
                alias='moe',
                port=18080,
                architecture_type='moe',
                expert_count=64,
                expert_used_count=8,
                turboquant_status='native',
                turboquant_key_dim=192,
                turboquant_value_dim=192,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('buun')):
                with patch('llama_tui.benchmark.model_file_size', return_value=12 * 1024**3):
                    profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        names = [item.name for item in profiles]
        self.assertIn('fit_turbokv_probe', names)
        self.assertIn('fit_default_probe', names)
        self.assertTrue(any(item.name.startswith('fit_context_growth_sweep_') and item.kv_preset == 'default' for item in profiles))

    def test_buun_fit_context_growth_command_omits_fixed_ngl(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='moe',
                name='MoE',
                path='/models/model.gguf',
                alias='moe',
                port=18080,
                architecture_type='moe',
                expert_count=64,
                expert_used_count=8,
                turboquant_status='native',
                turboquant_key_dim=256,
                turboquant_value_dim=256,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('buun')):
                with patch('llama_tui.benchmark.model_file_size', return_value=12 * 1024**3):
                    profiles = active_engine_runtime_profiles(app, model, hardware, depth='fast')
                profile = next(item for item in profiles if item.name.startswith('fit_context_growth_sweep_16384'))
                cmd = app.build_command(model, runtime_profile=profile)

        self.assertNotIn('-ngl', cmd)
        self.assertNotIn('--n-gpu-layers', cmd)
        self.assertIn('-fit', cmd)
        self.assertEqual(cmd[cmd.index('-fit') + 1], 'on')
        self.assertIn('-fitc', cmd)
        self.assertIn('--no-warmup', cmd)

    def test_buun_full_profiles_include_all_curated_turbokv_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='moe',
                name='MoE',
                path='/models/model.gguf',
                alias='moe',
                port=18080,
                architecture_type='moe',
                expert_count=64,
                expert_used_count=8,
                turboquant_status='native',
                turboquant_key_dim=256,
                turboquant_value_dim=256,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('buun')):
                with patch('llama_tui.benchmark.model_file_size', return_value=12 * 1024**3):
                    profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        presets = {item.kv_preset for item in profiles}
        self.assertIn('turbo4/turbo4', presets)
        self.assertIn('turbo3_tcq/turbo3_tcq', presets)
        self.assertIn('turbo3_tcq/turbo2_tcq', presets)
        self.assertIn('turbo2_tcq/turbo2_tcq', presets)
        self.assertIn('turbo3/turbo3', presets)
        self.assertIn('turbo2/turbo2', presets)
        self.assertTrue(all(item.benchmark_depth == 'full' for item in profiles))
        self.assertTrue(all(item.fit for item in profiles))
        self.assertTrue(all(item.gpu_layers is None for item in profiles))
        self.assertTrue(any(item.name.startswith('fit_context_growth_sweep_32768') for item in profiles))

    def test_buun_incompatible_turboquant_uses_fit_default_without_turbokv(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='gpt-oss',
                name='GPT OSS',
                path='/models/gpt-oss.gguf',
                alias='gpt-oss',
                port=18080,
                architecture_type='moe',
                expert_count=32,
                expert_used_count=4,
                turboquant_status='incompatible',
                turboquant_key_dim=64,
                turboquant_value_dim=64,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('buun')):
                with patch('llama_tui.benchmark.model_file_size', return_value=12 * 1024**3):
                    profiles = active_engine_runtime_profiles(app, model, hardware, depth='fast')

        self.assertEqual(profiles[0].name, 'fit_default_probe')
        self.assertTrue(profiles[0].fit)
        self.assertIsNone(profiles[0].gpu_layers)
        self.assertTrue(all(item.fit for item in profiles))
        self.assertTrue(all(item.gpu_layers is None for item in profiles))
        self.assertFalse(any('turbo' in item.kv_preset for item in profiles))

    def test_turboquant_head_dim_64_profiles_use_baseline_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(
                id='oss',
                name='GPT OSS Q4_K_M',
                path='/models/gpt-oss-Q4_K_M.gguf',
                alias='oss',
                port=18080,
                turboquant_status='incompatible',
                turboquant_head_dim=64,
                turboquant_key_dim=64,
                turboquant_value_dim=64,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=turboquant_no_fit_capabilities()), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        self.assertTrue(profiles)
        self.assertTrue(all(item.engine_id == 'turboquant' for item in profiles))
        self.assertEqual({item.kv_preset for item in profiles}, {'q8_0/q8_0'})

    def test_turboquant_unknown_head_dim_avoids_auto_turbo(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(
                id='unknown',
                name='Unknown Q4_K_M',
                path='/models/unknown-Q4_K_M.gguf',
                alias='unknown',
                port=18080,
                turboquant_status='unknown',
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=turboquant_no_fit_capabilities()), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        self.assertEqual({item.kv_preset for item in profiles}, {'q8_0/q8_0'})

    def test_turboquant_fit_profiles_discover_weight_fit_before_context_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(
                id='dense',
                name='Dense Q8_0',
                path='/models/dense-Q8_0.gguf',
                alias='dense',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                turboquant_status='native',
                turboquant_head_dim=128,
                turboquant_key_dim=128,
                turboquant_value_dim=128,
                ctx_max=262144,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('turboquant')), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3), \
                patch('llama_tui.benchmark.current_process_pressure_payload', return_value={}):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        self.assertTrue(profiles)
        self.assertEqual(profiles[0].fit_discovery_phase, 'weight_fit')
        self.assertTrue(profiles[0].fit)
        self.assertIsNone(profiles[0].gpu_layers)
        self.assertEqual(profiles[0].ctx_size, 8192)
        self.assertEqual(profiles[0].kv_preset, 'q8_0/turbo4')
        growth = profiles[1:]
        self.assertTrue(growth)
        self.assertTrue(all(item.fit_discovery_phase == 'context_growth' for item in growth))
        self.assertTrue(all(item.fit and item.gpu_layers is None for item in growth))
        self.assertFalse(any(item.ctx_size == 262144 for item in growth))
        growth_presets = [item.kv_preset for item in growth]
        self.assertIn('q8_0/turbo4', growth_presets)
        self.assertIn('q8_0/turbo3', growth_presets)
        self.assertLess(
            max(index for index, preset in enumerate(growth_presets) if preset == 'q8_0/turbo4'),
            min(index for index, preset in enumerate(growth_presets) if preset == 'q8_0/turbo3'),
        )

    def test_turboquant_q8_profiles_include_safe_balanced_and_symmetric(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(
                id='dense',
                name='Dense Q8_0',
                path='/models/dense-Q8_0.gguf',
                alias='dense',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                turboquant_status='native',
                turboquant_head_dim=128,
                turboquant_key_dim=128,
                turboquant_value_dim=128,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=turboquant_no_fit_capabilities()), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        presets = {item.kv_preset for item in profiles}
        self.assertIn('q8_0/q8_0', presets)
        self.assertIn('q8_0/turbo4', presets)
        self.assertIn('q8_0/turbo3', presets)
        self.assertIn('q8_0/turbo2', presets)
        self.assertIn('turbo4/turbo4', presets)
        self.assertIn('turbo3/turbo3', presets)

    def test_turboquant_low_bit_profiles_keep_symmetric_manual_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(
                id='sensitive',
                name='Sensitive Q4_K_M',
                path='/models/sensitive-Q4_K_M.gguf',
                alias='sensitive',
                port=18080,
                architecture='unknown',
                architecture_type='dense',
                turboquant_status='native',
                turboquant_head_dim=128,
                turboquant_key_dim=128,
                turboquant_value_dim=128,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=turboquant_no_fit_capabilities()), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        presets = {item.kv_preset for item in profiles}
        self.assertIn('q8_0/turbo4', presets)
        self.assertIn('q8_0/turbo3', presets)
        self.assertNotIn('turbo4/turbo4', presets)
        self.assertNotIn('turbo3/turbo3', presets)

    def test_turboquant_validated_low_bit_family_allows_symmetric(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(
                id='mistral',
                name='Mistral Q4_K_M',
                path='/models/mistral-Q4_K_M.gguf',
                alias='mistral',
                port=18080,
                architecture='mistral',
                model_family='mistral',
                architecture_type='dense',
                turboquant_status='native',
                turboquant_head_dim=128,
                turboquant_key_dim=128,
                turboquant_value_dim=128,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=turboquant_no_fit_capabilities()), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        self.assertIn('turbo3/turbo3', {item.kv_preset for item in profiles})

    def test_turboquant_runtime_record_context_includes_cache_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile(
                    'turboquant',
                    'llama-server',
                    kv_key_mode='q8_0',
                    kv_value_mode='turbo4',
                ),
            )
            model = ModelConfig(
                id='dense',
                name='Dense Q8_0',
                path='/models/dense-Q8_0.gguf',
                alias='dense',
                port=18080,
                architecture='llama',
                model_family='llama',
                turboquant_status='native',
                turboquant_head_dim=128,
                turboquant_key_dim=128,
                turboquant_value_dim=128,
            )
            caps = default_engine_capabilities('turboquant')

            with patch.object(app, 'engine_capabilities', return_value=caps):
                context = runtime_record_context(app, model)

        self.assertEqual(context['engine'], 'turboquant')
        self.assertEqual(context['ctk'], 'q8_0')
        self.assertEqual(context['ctv'], 'turbo4')
        self.assertEqual(context['detected_head_dim'], 128)
        self.assertEqual(context['model_quant'], 'Q8_0')
        self.assertEqual(context['model_family'], 'llama')
        self.assertEqual(context['binary_path'], app.runtime_profile.server_command)
        self.assertIn('turbo4', context['help_supported_cache_types'])

    def test_runtime_record_context_includes_launch_profile_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='dense',
                name='Dense',
                path='/models/dense.gguf',
                alias='dense',
                port=18080,
                output=4096,
            )
            profile = build_benchmark_launch_profile(model, purpose='serve_default', depth='fast')

            with patch.object(app, 'engine_capabilities', return_value=EngineCapabilities()):
                context = runtime_record_context(app, model, benchmark_profile=profile)

        self.assertEqual(context['benchmark_profile'], 'serve_default')
        self.assertEqual(context['benchmark_purpose'], 'serve_default')
        self.assertEqual(context['measurement_output'], 512)
        self.assertEqual(context['ctx'], model.ctx)
        self.assertEqual(context['temp'], model.temp)

    def test_benchmark_completion_payload_uses_launch_profile_sampling(self):
        model = ModelConfig(
            id='tiny',
            name='Tiny',
            path='/models/tiny.gguf',
            alias='tiny',
            port=18080,
            output=4096,
            temp=0.31,
            top_p=0.82,
            top_k=17,
            repeat_penalty=1.04,
            presence_penalty=0.2,
            launch_overrides={'min_p': 0.03, 'seed': 99},
        )
        profile = build_benchmark_launch_profile(model, purpose='serve_default', depth='fast')

        with patch(
            'llama_tui.benchmark.post_json',
            return_value={
                'usage': {'completion_tokens': 12, 'prompt_tokens': 8},
                'choices': [{'message': {'content': 'done'}}],
            },
        ) as post:
            ok, result = benchmark_completion(model, launch_profile=profile)

        payload = post.call_args.args[1]
        self.assertTrue(ok)
        self.assertGreater(result['tokens_per_sec'], 0)
        self.assertEqual(payload['temperature'], 0.31)
        self.assertEqual(payload['max_tokens'], 512)
        self.assertEqual(payload['top_p'], 0.82)
        self.assertEqual(payload['top_k'], 17)
        self.assertEqual(payload['repeat_penalty'], 1.04)
        self.assertEqual(payload['presence_penalty'], 0.2)
        self.assertEqual(payload['min_p'], 0.03)
        self.assertEqual(payload['seed'], 99)

    def test_raw_speed_benchmark_records_history_without_updating_measured_profiles(self):
        original = ModelConfig(
            id='tiny',
            name='Tiny',
            path='/models/tiny.gguf',
            alias='tiny',
            port=18080,
            measured_profiles={'auto': {'tokens_per_sec': 9.0}},
            last_benchmark_tokens_per_sec=9.0,
        )

        class FakeApp:
            models = [original]

            def __init__(self):
                self.saved = None

            def hardware_profile(self, refresh=False):
                return HardwareProfile(memory_available=16 * 1024**3)

            def runtime_profile_from_model(self, *_args, **_kwargs):
                return None

            def engine_capabilities(self):
                return EngineCapabilities()

            def add_or_update(self, saved):
                self.saved = saved

        record = {
            'status': 'ok',
            'objective': 'raw_speed',
            'tokens_per_sec': 33.0,
            'benchmark_profile': 'raw_speed',
            'benchmark_purpose': 'raw_speed',
            'measurement_output': 512,
            'engine': 'llama.cpp',
        }
        app = FakeApp()

        with patch('llama_tui.benchmark.benchmark_preflight_cleanup', return_value=(True, 'clean')), \
            patch('llama_tui.benchmark.benchmark_adaptive_candidate', return_value=(record, dict(record))) as runner:
            ok, msg = benchmark_raw_speed_profile(app, original)

        self.assertTrue(ok)
        self.assertIn('raw speed benchmark saved', msg)
        self.assertEqual(app.saved.measured_profiles, original.measured_profiles)
        self.assertEqual(app.saved.last_benchmark_tokens_per_sec, original.last_benchmark_tokens_per_sec)
        self.assertEqual(app.saved.benchmark_runs[0]['kind'], 'raw_speed')
        self.assertEqual(app.saved.benchmark_runs[0]['benchmark_profiles'], ['raw_speed'])
        self.assertEqual(runner.call_args.kwargs['benchmark_purpose'], 'raw_speed')

    def test_fit_discovery_metadata_persists_on_measured_profiles(self):
        candidate = ModelConfig(id='dense', name='Dense', path='/models/dense.gguf', alias='dense', port=18080, ctx=8192)
        record = adaptive_record_from_candidate(
            candidate,
            'long_context',
            'ok',
            engine='turboquant',
            runtime_profile='fit_weight_discovery_q8_0_turbo4',
            kv_preset='q8_0/turbo4',
            runtime_fit=True,
            fit_context=4096,
            fit_discovery_phase='weight_fit',
            viable_ngl=28,
            viable_ngl_source='offloaded_layers',
            fit_selected_ngl=28,
            fit_selected_ngl_source='offloaded_layers',
            fit_log_excerpt='offloaded 28/33 layers to GPU',
            runtime_log_path='/tmp/runtime/turboquant/dense.log',
        )
        model = ModelConfig(
            id='dense',
            name='Dense',
            path='/models/dense.gguf',
            alias='dense',
            port=18080,
            measured_profiles={'opencode_ready': record},
        )

        profile = measured_profile_runtime_profile(model, 'opencode_ready')

        self.assertEqual(record['fit_discovery_phase'], 'weight_fit')
        self.assertEqual(record['viable_ngl'], 28)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.fit_discovery_phase, 'weight_fit')
        self.assertEqual(profile.viable_ngl, 28)
        self.assertEqual(profile.viable_ngl_source, 'offloaded_layers')
        self.assertEqual(profile.fit_selected_ngl, 28)
        self.assertEqual(profile.fit_log_excerpt, 'offloaded 28/33 layers to GPU')

    def test_benchmark_run_summary_mentions_saved_winners_despite_failures(self):
        ok = {'status': 'ok', 'tokens_per_sec': 24.0, 'ctx': 8192, 'ctx_per_slot': 8192}
        failed = {'status': 'not ready', 'failure_category': 'CUDA_OOM_WEIGHTS'}

        summary = benchmark_run_summary({'auto': ok}, [failed, ok])

        self.assertIn('auto=8192 ctx', summary)
        self.assertIn('1 candidate failure(s), winners saved', summary)

    def test_buun_profiles_filter_unsupported_help_kv_modes(self):
        caps = EngineCapabilities(
            flash_attn_syntax='value',
            flash_attn_flag='--flash-attn',
            supports_ctk_ctv=True,
            supports_parallel=True,
            gpu_layers_flag='-ngl',
            supported_kv_modes=('turbo4', 'turbo3_tcq'),
        )
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='moe',
                name='MoE',
                path='/models/model.gguf',
                alias='moe',
                port=18080,
                architecture_type='moe',
                expert_count=64,
                expert_used_count=8,
                turboquant_status='native',
                turboquant_key_dim=256,
                turboquant_value_dim=256,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=caps):
                with patch('llama_tui.benchmark.model_file_size', return_value=12 * 1024**3):
                    profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        presets = {item.kv_preset for item in profiles}
        self.assertIn('turbo4/turbo4', presets)
        self.assertIn('turbo3_tcq/turbo3_tcq', presets)
        self.assertNotIn('turbo3_tcq/turbo2_tcq', presets)
        self.assertNotIn('turbo2_tcq/turbo2_tcq', presets)

    def test_capability_parser_extracts_turbo_kv_modes_from_allowed_values(self):
        caps = parse_engine_capabilities(
            'usage: llama-server --flash-attn on|off|auto -ctk TYPE -ctv TYPE\n'
            'allowed values: f32, f16, turbo4, turbo3_tcq, turbo2_tcq, turbo3, turbo2\n'
            '--parallel N -ngl N',
            engine_id='buun',
        )

        self.assertIn('turbo4', caps.supported_kv_modes)
        self.assertIn('turbo3_tcq', caps.supported_kv_modes)
        self.assertIn('turbo2_tcq', caps.supported_kv_modes)

    def test_buun_wrapped_help_values_enable_turbokv_planner(self):
        help_text = (
            '-ctk,  --cache-type-k TYPE              KV cache data type for K\n'
            '                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1,\n'
            '                                        turbo2, turbo3, turbo4, turbo3_tcq, turbo2_tcq\n'
            '                                        (default: f16)\n'
            '-ctv,  --cache-type-v TYPE              KV cache data type for V\n'
            '                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1,\n'
            '                                        turbo2, turbo3, turbo4, turbo3_tcq, turbo2_tcq\n'
            '                                        (default: f16)\n'
            '--flash-attn on|off|auto --parallel N -ngl N'
        )
        caps = parse_engine_capabilities(help_text, engine_id='buun')
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='moe',
                name='MoE',
                path='/models/model.gguf',
                alias='moe',
                port=18080,
                architecture_type='moe',
                expert_count=64,
                expert_used_count=8,
                turboquant_status='native',
                turboquant_key_dim=256,
                turboquant_value_dim=256,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=caps):
                with patch('llama_tui.benchmark.model_file_size', return_value=12 * 1024**3):
                    profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        presets = {item.kv_preset for item in profiles}
        self.assertIn('turbo4/turbo4', presets)
        self.assertIn('turbo3_tcq/turbo3_tcq', presets)
        self.assertIn('turbo3_tcq/turbo2_tcq', presets)
        self.assertIn('turbo2_tcq/turbo2_tcq', presets)
        self.assertFalse(any(item.kv_preset == 'q8_0/q8_0' for item in profiles))

    def test_capability_parser_falls_back_to_known_buun_modes_when_help_omits_values(self):
        caps = parse_engine_capabilities(
            'usage: llama-server --flash-attn on|off|auto -ctk MODE -ctv MODE --parallel N -ngl N',
            engine_id='buun',
        )

        self.assertEqual(caps.supported_kv_modes, ('turbo4', 'turbo3_tcq', 'turbo2_tcq', 'turbo3', 'turbo2'))

    def test_buun_turbo_command_never_adds_generic_cache_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='m',
                name='M',
                path='/models/model.gguf',
                alias='m',
                port=18080,
                extra_args=['--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0'],
            )
            profile = RuntimeProfile(
                engine_id='buun',
                ctx_size=8192,
                gpu_layers=20,
                parallel=1,
                kv_preset='turbo3_tcq/turbo2_tcq',
                flash_attn='on',
                name='kv_compression_probe_turbo3_tcq_turbo2_tcq',
            )
            caps = default_engine_capabilities('buun')

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model, runtime_profile=profile)

        self.assertIn('-ctk', cmd)
        self.assertIn('-ctv', cmd)
        self.assertIn('turbo3_tcq', cmd)
        self.assertIn('turbo2_tcq', cmd)
        self.assertNotIn('--cache-type-k', cmd)
        self.assertNotIn('--cache-type-v', cmd)

    def test_turbokv_scoring_prefers_safe_mode_when_context_and_speed_match(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx_max=32768)
        safe = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=16384, parallel=1)
        aggressive = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=16384, parallel=1)
        measured = [
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'long_context', 'model': safe, 'tokens_per_sec': 30.0, 'ctx_per_slot': 16384, 'parallel': 1, 'kv_preset': 'turbo4/turbo4', 'kv_score_penalty': 0.0},
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'long_context', 'model': aggressive, 'tokens_per_sec': 30.0, 'ctx_per_slot': 16384, 'parallel': 1, 'kv_preset': 'turbo2_tcq/turbo2_tcq', 'kv_score_penalty': 0.10},
        ]

        winners = select_measured_profiles(model, measured, profile)

        self.assertEqual(winners['long_context']['kv_preset'], 'turbo4/turbo4')
        self.assertEqual(winners['auto']['kv_preset'], 'turbo4/turbo4')

    def test_turbokv_scoring_allows_more_aggressive_mode_for_material_context_gain(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx_max=32768)
        safe = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=8192, parallel=1)
        aggressive = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=32768, parallel=1)
        measured = [
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'long_context', 'model': safe, 'tokens_per_sec': 35.0, 'ctx_per_slot': 8192, 'parallel': 1, 'kv_preset': 'turbo4/turbo4', 'kv_score_penalty': 0.0},
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'long_context', 'model': aggressive, 'tokens_per_sec': 30.0, 'ctx_per_slot': 32768, 'parallel': 1, 'kv_preset': 'turbo3_tcq/turbo2_tcq', 'kv_score_penalty': 0.06},
        ]

        winners = select_measured_profiles(model, measured, profile)

        self.assertEqual(winners['long_context']['kv_preset'], 'turbo3_tcq/turbo2_tcq')
        self.assertEqual(winners['opencode_ready']['kv_preset'], 'turbo3_tcq/turbo2_tcq')

    def test_runtime_profile_runners_use_expected_depth_and_attempts(self):
        model = ModelConfig(id='m', name='M', path='/models/model.gguf', alias='m', port=18200)
        hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        runtime_profile = RuntimeProfile(
            engine_id='buun',
            name='kv_compression_probe',
            ctx_size=8192,
            gpu_layers=20,
            parallel=1,
            kv_preset='turbo4/turbo4',
            benchmark_depth='fast',
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model):
                return None

            def hardware_profile(self, refresh=False):
                return hardware

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, model):
                self.saved.append(model)

        def fake_runtime_benchmark(_app, base_model, profile, objective, _progress, _cancel_token, completed, total, **kwargs):
            candidate = ModelConfig(id=base_model.id, name=base_model.name, path=base_model.path, alias=base_model.alias, port=base_model.port, ctx=profile.ctx_size, parallel=profile.parallel, ngl=profile.gpu_layers)
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=25.0,
                seconds=1.0,
                runtime_profile=profile.name,
                kv_preset=profile.kv_preset,
                benchmark_depth=kwargs.get('benchmark_depth', ''),
            )
            record['variant'] = profile.name
            measured = dict(record)
            measured['model'] = candidate
            return True, False, [record], [measured], completed + 1

        app = FakeApp()
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[runtime_profile]):
            with patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark) as runner:
                ok, msg = benchmark_fast_profiles(app, model)

        self.assertTrue(ok, msg)
        self.assertEqual(runner.call_args.kwargs['max_attempts'], 1)
        self.assertEqual(runner.call_args.kwargs['benchmark_depth'], 'fast')

        app = FakeApp()
        runtime_profile = RuntimeProfile(
            engine_id='buun',
            name='kv_compression_probe',
            ctx_size=8192,
            gpu_layers=20,
            parallel=1,
            kv_preset='turbo4/turbo4',
            benchmark_depth='full',
        )
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[runtime_profile]):
            with patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark) as runner:
                ok, msg = benchmark_exhaustive_profiles(app, model)

        self.assertTrue(ok, msg)
        self.assertEqual(runner.call_args.kwargs['max_attempts'], 2)
        self.assertEqual(runner.call_args.kwargs['benchmark_depth'], 'full')

    def test_runtime_profile_retry_does_not_repeat_buun_fit_failures(self):
        model = ModelConfig(id='m', name='M', path='/models/model.gguf', alias='m', port=18200)
        runtime_profile = RuntimeProfile(
            engine_id='buun',
            name='partial_gpu_probe',
            ctx_size=8192,
            gpu_layers=21,
            parallel=1,
            kv_preset='turbo4/turbo4',
        )
        failed = adaptive_record_from_candidate(
            model,
            'long_context',
            'not ready',
            detail='failed to fit params to free device memory: n_gpu_layers already set by user to 21',
            failure_category='BUUN_FIT_FAILED',
            failure_reason='failed to fit params to free device memory: n_gpu_layers already set by user to 21',
        )

        class FakeApp:
            def build_command(self, _model, runtime_profile=None, benchmark_profile=None):
                return ['buun-llama-server']

        with patch('llama_tui.benchmark.benchmark_adaptive_candidate', return_value=(failed, None)) as runner:
            ok, broke, records, measured, completed = benchmark_runtime_profile_with_retry(
                FakeApp(),
                model,
                runtime_profile,
                'long_context',
                None,
                None,
                0,
                2,
                max_attempts=2,
            )

        self.assertFalse(ok)
        self.assertTrue(broke)
        self.assertEqual(completed, 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(measured, [])
        self.assertEqual(runner.call_count, 1)

    def test_failed_fast_runtime_profile_run_preserves_previous_working_winners(self):
        profile = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        old_auto = {
            'status': 'ok',
            'tokens_per_sec': 22.0,
            'seconds': 1.0,
            'ctx': 8192,
            'ctx_per_slot': 8192,
            'parallel': 1,
        }
        model = ModelConfig(
            id='m',
            name='M',
            path='/models/model.gguf',
            alias='m',
            port=18200,
            measured_profiles={
                'auto': dict(old_auto),
                'fast_chat': dict(old_auto),
                'long_context': dict(old_auto),
                'opencode_ready': dict(old_auto),
            },
            default_benchmark_status='done',
            last_benchmark_tokens_per_sec=22.0,
        )
        runtime_profile = RuntimeProfile(
            engine_id='buun',
            name='fit_turbokv_probe',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='turbo4/turbo4',
            kv_family='turbo',
            fit=True,
            fit_context=4096,
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self, stored):
                self.models = [stored]
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model, discover=True, managed_only=False):
                return None

            def get_model(self, model_id):
                return next((item for item in self.models if item.id == model_id), None)

            def hardware_profile(self, refresh=False):
                return profile

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, saved):
                self.models[0] = saved
                self.saved.append(saved)

        failed = adaptive_record_from_candidate(
            model,
            'long_context',
            'not ready',
            detail='K cache type turbo4 with block size 128 does not divide n_embd_head_k=192',
            failure_category='KV_MODE_INCOMPATIBLE',
            failure_reason='K cache type turbo4 with block size 128 does not divide n_embd_head_k=192',
        )

        def fake_runtime_benchmark(_app, _base_model, _profile, _objective, _progress, _cancel_token, completed, _total, **_kwargs):
            return False, True, [dict(failed)], [], completed + 1

        app = FakeApp(model)
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[runtime_profile]), \
             patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
            ok, msg = benchmark_fast_profiles(app, model)

        self.assertFalse(ok)
        self.assertIn('kept previous working measured profiles', msg)
        self.assertEqual(app.saved[-1].default_benchmark_status, 'done')
        self.assertEqual(app.saved[-1].measured_profiles['auto']['tokens_per_sec'], 22.0)
        self.assertEqual(app.saved[-1].last_benchmark_tokens_per_sec, 22.0)

    def test_fast_runner_skips_remaining_turbokv_after_shape_incompatibility(self):
        model = ModelConfig(id='m', name='M', path='/models/model.gguf', alias='m', port=18200)
        hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        turbo4 = RuntimeProfile(
            engine_id='buun',
            name='fit_turbokv_probe',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='turbo4/turbo4',
            kv_family='turbo',
            fit=True,
            fit_context=4096,
        )
        turbo3 = RuntimeProfile(
            engine_id='buun',
            name='fit_kv_compression_probe_turbo3',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='turbo3_tcq/turbo3_tcq',
            kv_family='turbo',
            fit=True,
            fit_context=4096,
        )
        default = RuntimeProfile(
            engine_id='buun',
            name='fit_default_probe',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='default',
            kv_family='default',
            fit=True,
            fit_context=4096,
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model, discover=True, managed_only=False):
                return None

            def hardware_profile(self, refresh=False):
                return hardware

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, saved):
                self.saved.append(saved)

        calls = []

        def fake_runtime_benchmark(_app, base_model, profile, objective, _progress, _cancel_token, completed, _total, **kwargs):
            calls.append(profile.name)
            candidate = ModelConfig(id=base_model.id, name=base_model.name, path=base_model.path, alias=base_model.alias, port=base_model.port, ctx=profile.ctx_size, parallel=profile.parallel)
            if profile.kv_family == 'turbo':
                failed = adaptive_record_from_candidate(
                    candidate,
                    objective,
                    'not ready',
                    detail='K cache type turbo4 with block size 128 does not divide n_embd_head_k=192',
                    failure_category='KV_MODE_INCOMPATIBLE',
                    failure_reason='K cache type turbo4 with block size 128 does not divide n_embd_head_k=192',
                )
                return False, True, [failed], [], completed + 1
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=18.0,
                seconds=1.0,
                engine=profile.engine_id,
                runtime_profile=profile.name,
                kv_preset=profile.kv_preset,
                runtime_fit=profile.fit,
                fit_context=profile.fit_context,
            )
            measured = dict(record)
            measured['model'] = candidate
            return True, False, [record], [measured], completed + 1

        events = []
        app = FakeApp()
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[turbo4, turbo3, default]), \
             patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
            ok, msg = benchmark_fast_profiles(app, model, progress=events.append)

        self.assertTrue(ok, msg)
        self.assertEqual(calls, ['fit_turbokv_probe', 'fit_default_probe'])
        self.assertTrue(any('disabled family turbo' in str(event) for event in events))

    def test_fast_runner_skips_fixed_buun_profiles_after_fit_success(self):
        model = ModelConfig(id='m', name='M', path='/models/model.gguf', alias='m', port=18200)
        hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        fit_probe = RuntimeProfile(
            engine_id='buun',
            name='fit_turbokv_probe',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='turbo4/turbo4',
            fit=True,
            fit_context=4096,
            no_warmup=True,
            benchmark_depth='fast',
        )
        fixed_probe = RuntimeProfile(
            engine_id='buun',
            name='partial_gpu_probe',
            ctx_size=8192,
            gpu_layers=21,
            parallel=1,
            kv_preset='turbo4/turbo4',
            benchmark_depth='fast',
        )
        fit_growth = RuntimeProfile(
            engine_id='buun',
            name='fit_context_growth_sweep_16384_turbo4_turbo4',
            ctx_size=16384,
            gpu_layers=None,
            parallel=1,
            kv_preset='turbo4/turbo4',
            fit=True,
            fit_context=4096,
            no_warmup=True,
            benchmark_depth='fast',
        )
        fit_growth_high = RuntimeProfile(
            engine_id='buun',
            name='fit_context_growth_sweep_49152_turbo4_turbo4',
            ctx_size=49152,
            gpu_layers=None,
            parallel=1,
            kv_preset='turbo4/turbo4',
            fit=True,
            fit_context=4096,
            no_warmup=True,
            benchmark_depth='fast',
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model):
                return None

            def hardware_profile(self, refresh=False):
                return hardware

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, model):
                self.saved.append(model)

        calls = []

        def fake_runtime_benchmark(_app, base_model, profile, objective, _progress, _cancel_token, completed, total, **kwargs):
            calls.append(profile.name)
            candidate = ModelConfig(
                id=base_model.id,
                name=base_model.name,
                path=base_model.path,
                alias=base_model.alias,
                port=base_model.port,
                ctx=profile.ctx_size,
                parallel=profile.parallel,
                ngl=profile.gpu_layers if profile.gpu_layers is not None else base_model.ngl,
            )
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=25.0,
                seconds=1.0,
                engine=profile.engine_id,
                runtime_profile=profile.name,
                kv_preset=profile.kv_preset,
                benchmark_depth=kwargs.get('benchmark_depth', ''),
                runtime_fit=profile.fit,
                fit_context=profile.fit_context,
                runtime_no_warmup=profile.no_warmup,
                gpu_layers_mode='fit' if profile.gpu_layers is None else 'fixed',
            )
            measured = dict(record)
            measured['model'] = candidate
            return True, False, [record], [measured], completed + 1

        app = FakeApp()
        events = []
        profiles = [fit_probe, fixed_probe, fit_growth, fit_growth_high]
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=profiles):
            with patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
                ok, msg = benchmark_fast_profiles(app, model, progress=events.append)

        self.assertTrue(ok, msg)
        self.assertEqual(calls, [
            'fit_turbokv_probe',
            'fit_context_growth_sweep_16384_turbo4_turbo4',
            'fit_context_growth_sweep_49152_turbo4_turbo4',
        ])
        self.assertTrue(any('skipping fixed-NGL fallback probes' in str(item) for item in events))
        self.assertTrue(any('skipped 1 fixed-NGL profile' in str(item) for item in events))
        self.assertIn('buun fit profile', app.saved[-1].benchmark_runs[0]['summary'])

    def test_fast_runner_carries_weight_fit_ceiling_into_turboquant_context_growth(self):
        model = ModelConfig(id='m', name='M', path='/models/model-Q8_0.gguf', alias='m', port=18200)
        hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        weight_fit = RuntimeProfile(
            engine_id='turboquant',
            name='fit_weight_discovery_q8_0_turbo4',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='q8_0/turbo4',
            fit=True,
            fit_context=4096,
            benchmark_depth='fast',
            fit_discovery_phase='weight_fit',
        )
        context_growth = RuntimeProfile(
            engine_id='turboquant',
            name='fit_context_growth_sweep_16384_q8_0_turbo4',
            ctx_size=16384,
            gpu_layers=None,
            parallel=1,
            kv_preset='q8_0/turbo4',
            fit=True,
            fit_context=4096,
            benchmark_depth='fast',
            fit_discovery_phase='context_growth',
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model):
                return None

            def hardware_profile(self, refresh=False):
                return hardware

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, model):
                self.saved.append(model)

        seen_viable = []

        def fake_runtime_benchmark(_app, base_model, profile, objective, _progress, _cancel_token, completed, total, **kwargs):
            seen_viable.append((profile.name, profile.viable_ngl, profile.viable_ngl_source))
            candidate = ModelConfig(
                id=base_model.id,
                name=base_model.name,
                path=base_model.path,
                alias=base_model.alias,
                port=base_model.port,
                ctx=profile.ctx_size,
                parallel=profile.parallel,
            )
            viable_ngl = 28 if profile.fit_discovery_phase == 'weight_fit' else profile.viable_ngl
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=20.0,
                seconds=1.0,
                engine=profile.engine_id,
                runtime_profile=profile.name,
                kv_preset=profile.kv_preset,
                benchmark_depth=kwargs.get('benchmark_depth', ''),
                runtime_fit=profile.fit,
                fit_context=profile.fit_context,
                gpu_layers_mode='fit',
                fit_discovery_phase=profile.fit_discovery_phase,
                viable_ngl=viable_ngl,
                viable_ngl_source='offloaded_layers',
                fit_selected_ngl=viable_ngl,
                fit_selected_ngl_source='offloaded_layers',
                fit_log_excerpt='offloaded 28/33 layers to GPU',
            )
            measured = dict(record)
            measured['model'] = candidate
            return True, False, [record], [measured], completed + 1

        app = FakeApp()
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[weight_fit, context_growth]), \
             patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
            ok, msg = benchmark_fast_profiles(app, model, progress=lambda _msg: None)

        self.assertTrue(ok, msg)
        self.assertEqual(seen_viable[0], ('fit_weight_discovery_q8_0_turbo4', 0, ''))
        self.assertEqual(seen_viable[1], ('fit_context_growth_sweep_16384_q8_0_turbo4', 28, 'offloaded_layers'))

    def test_fast_runner_prunes_fixed_gpu_layer_profiles_after_fit_oom(self):
        model = ModelConfig(id='m', name='M', path='/models/model.gguf', alias='m', port=18200)
        hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        fixed_a = RuntimeProfile(
            engine_id='buun',
            name='gpu_layer_sweep_ngl26',
            ctx_size=8192,
            gpu_layers=26,
            parallel=1,
            kv_preset='turbo4/turbo4',
            benchmark_depth='fast',
        )
        fixed_b = RuntimeProfile(
            engine_id='buun',
            name='gpu_layer_sweep_ngl30',
            ctx_size=8192,
            gpu_layers=30,
            parallel=1,
            kv_preset='turbo4/turbo4',
            benchmark_depth='fast',
        )
        fit_fallback = RuntimeProfile(
            engine_id='buun',
            name='fit_default_probe',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='default',
            fit=True,
            fit_context=4096,
            no_warmup=True,
            benchmark_depth='fast',
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model):
                return None

            def hardware_profile(self, refresh=False):
                return hardware

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, model):
                self.saved.append(model)

        calls = []

        def fake_runtime_benchmark(_app, base_model, profile, objective, _progress, _cancel_token, completed, total, **kwargs):
            calls.append(profile.name)
            candidate = ModelConfig(
                id=base_model.id,
                name=base_model.name,
                path=base_model.path,
                alias=base_model.alias,
                port=base_model.port,
                ctx=profile.ctx_size,
                parallel=profile.parallel,
                ngl=profile.gpu_layers if profile.gpu_layers is not None else base_model.ngl,
            )
            if profile.gpu_layers is not None:
                failed = adaptive_record_from_candidate(
                    candidate,
                    objective,
                    'not ready',
                    detail='failed to fit params to free device memory: n_gpu_layers already set by user to 26',
                    failure_category='FIXED_GPU_LAYERS_FIT_FAILED',
                    failure_reason='fixed GPU layers cannot fit',
                    runtime_profile=profile.name,
                    gpu_layers_mode='fixed',
                )
                return False, True, [failed], [], completed + 1
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=20.0,
                seconds=1.0,
                engine=profile.engine_id,
                runtime_profile=profile.name,
                kv_preset=profile.kv_preset,
                runtime_fit=profile.fit,
                fit_context=profile.fit_context,
                gpu_layers_mode='fit',
            )
            measured = dict(record)
            measured['model'] = candidate
            return True, False, [record], [measured], completed + 1

        events = []
        app = FakeApp()
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[fixed_a, fixed_b, fit_fallback]), \
             patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
            ok, msg = benchmark_fast_profiles(app, model, progress=events.append)

        self.assertTrue(ok, msg)
        self.assertEqual(calls, ['gpu_layer_sweep_ngl26', 'fit_default_probe'])
        self.assertTrue(any('fixed GPU-layer profiles were already rejected' in str(item) for item in events))

    def test_fast_runner_prunes_turboquant_fixed_profiles_after_cuda_weight_oom(self):
        model = ModelConfig(id='m', name='M', path='/models/model-Q8_0.gguf', alias='m', port=18200)
        hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        fixed_a = RuntimeProfile(
            engine_id='turboquant',
            name='gpu_layer_sweep_ngl26',
            ctx_size=8192,
            gpu_layers=26,
            parallel=1,
            kv_preset='q8_0/turbo4',
            benchmark_depth='fast',
        )
        fixed_b = RuntimeProfile(
            engine_id='turboquant',
            name='gpu_layer_sweep_ngl30',
            ctx_size=8192,
            gpu_layers=30,
            parallel=1,
            kv_preset='q8_0/turbo4',
            benchmark_depth='fast',
        )
        fit_fallback = RuntimeProfile(
            engine_id='turboquant',
            name='fit_weight_discovery_q8_0_turbo4',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='q8_0/turbo4',
            fit=True,
            fit_context=4096,
            benchmark_depth='fast',
            fit_discovery_phase='weight_fit',
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model):
                return None

            def hardware_profile(self, refresh=False):
                return hardware

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, model):
                self.saved.append(model)

        calls = []

        def fake_runtime_benchmark(_app, base_model, profile, objective, _progress, _cancel_token, completed, total, **kwargs):
            calls.append(profile.name)
            candidate = ModelConfig(
                id=base_model.id,
                name=base_model.name,
                path=base_model.path,
                alias=base_model.alias,
                port=base_model.port,
                ctx=profile.ctx_size,
                parallel=profile.parallel,
                ngl=profile.gpu_layers if profile.gpu_layers is not None else base_model.ngl,
            )
            if profile.gpu_layers is not None:
                failed = adaptive_record_from_candidate(
                    candidate,
                    objective,
                    'not ready',
                    detail='alloc_tensor_range: failed to allocate CUDA0 buffer',
                    failure_category='CUDA_OOM_WEIGHTS',
                    failure_reason='failed to allocate CUDA0 buffer',
                    runtime_profile=profile.name,
                    gpu_layers_mode='fixed',
                )
                return False, True, [failed], [], completed + 1
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=20.0,
                seconds=1.0,
                engine=profile.engine_id,
                runtime_profile=profile.name,
                kv_preset=profile.kv_preset,
                runtime_fit=profile.fit,
                fit_context=profile.fit_context,
                gpu_layers_mode='fit',
                fit_discovery_phase=profile.fit_discovery_phase,
                viable_ngl=24,
                viable_ngl_source='offloaded_layers',
            )
            measured = dict(record)
            measured['model'] = candidate
            return True, False, [record], [measured], completed + 1

        events = []
        app = FakeApp()
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[fixed_a, fixed_b, fit_fallback]), \
             patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
            ok, msg = benchmark_fast_profiles(app, model, progress=events.append)

        self.assertTrue(ok, msg)
        self.assertEqual(calls, ['gpu_layer_sweep_ngl26', 'fit_weight_discovery_q8_0_turbo4'])
        self.assertTrue(any('fixed GPU-layer profiles were already rejected' in str(item) for item in events))

    def test_deep_benchmark_all_continues_using_adaptive_runner(self):
        self.assertEqual(benchmark_all_models_runner.__name__, 'benchmark_all_models_runner')

    def test_llama_dense_gguf_profiles_use_generic_non_turbo_kv(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='dense',
                name='Dense GGUF',
                path='/models/dense.gguf',
                alias='dense',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch('llama_tui.benchmark.model_file_size', return_value=5 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware)

        self.assertTrue(any(item.name == 'cpu_probe' for item in profiles))
        self.assertTrue(any(item.name == 'partial_gpu_probe' for item in profiles))
        self.assertTrue(any(item.kv_preset == 'q8_0/q8_0' for item in profiles))
        self.assertFalse(any('turbo' in item.kv_preset for item in profiles))

    def test_tq3_runtime_profiles_require_native_weight_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('tq3', 'llama-server'),
            )
            native = ModelConfig(
                id='native',
                name='Native TQ3',
                path='/models/native.TQ3_4S.gguf',
                alias='native',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                tq3_status='native',
                tq3_weight_format='TQ3_4S',
                ctx_max=32768,
            )
            regular = ModelConfig(
                id='regular',
                name='Regular GGUF',
                path='/models/regular.Q4_K_M.gguf',
                alias='regular',
                port=18081,
                architecture='llama',
                architecture_type='dense',
                tq3_status='not_native',
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('tq3')):
                with patch('llama_tui.benchmark.model_file_size', return_value=5 * 1024**3):
                    native_profiles = active_engine_runtime_profiles(app, native, hardware)
                    regular_profiles = active_engine_runtime_profiles(app, regular, hardware)

        self.assertTrue(native_profiles)
        self.assertTrue(all(item.engine_id == 'tq3' for item in native_profiles))
        self.assertTrue(any(item.kv_preset == 'q8_0/q8_0' for item in native_profiles))
        self.assertEqual(regular_profiles, [])

    def test_small_moe_gguf_can_plan_full_gpu_without_model_specific_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('buun', 'llama-server'),
            )
            model = ModelConfig(
                id='mix',
                name='Small MoE',
                path='/models/small-moe.gguf',
                alias='mix',
                port=18080,
                architecture='mixtral',
                architecture_type='moe',
                expert_count=8,
                expert_used_count=2,
                turboquant_status='native',
                turboquant_key_dim=128,
                turboquant_value_dim=128,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('buun')):
                with patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                    profiles = active_engine_runtime_profiles(app, model, hardware)

        self.assertTrue(any(item.name == 'gpu_layer_sweep_full' and item.gpu_layers == 999 for item in profiles))
        self.assertTrue(any(item.kv_preset == 'turbo4/turbo4' for item in profiles))

    def test_large_dense_gguf_uses_partial_gpu_and_q8_without_turbo(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='large-dense',
                name='Large Dense',
                path='/models/large-dense.gguf',
                alias='large',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=10 * 1024**3, gpu_memory_free=8 * 1024**3)

            with patch('llama_tui.benchmark.model_file_size', return_value=14 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware)

        partial = next(item for item in profiles if item.name == 'partial_gpu_probe')
        self.assertGreater(partial.gpu_layers, 0)
        self.assertLess(partial.gpu_layers, 999)
        self.assertTrue(any(item.kv_preset == 'q8_0/q8_0' for item in profiles))
        self.assertFalse(any('turbo' in item.kv_preset for item in profiles))

    def test_cpu_only_gguf_profiles_do_not_request_gpu_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='cpu',
                name='CPU Model',
                path='/models/cpu.gguf',
                alias='cpu',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                ctx_max=16384,
            )
            hardware = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=32 * 1024**3, memory_available=24 * 1024**3)

            with patch('llama_tui.benchmark.model_file_size', return_value=6 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware)

        self.assertTrue(profiles)
        self.assertTrue(all(item.gpu_layers == 0 for item in profiles))

    def test_runtime_profile_planner_does_not_branch_on_qwen_names(self):
        source = inspect.getsource(active_engine_runtime_profiles).lower()

        self.assertNotIn('qwen', source)

    def test_invalid_buun_kv_modes_fail_clearly(self):
        for flag in ('--kv', '--kv-key', '--kv-value'):
            with self.subTest(flag=flag):
                args = build_cli_parser().parse_args(['--engine', 'buun', flag, 'bad-mode'])

                with self.assertRaises(SystemExit) as ctx:
                    validate_buun_kv_args(args)

                self.assertIn(f'Unsupported {flag} "bad-mode"', str(ctx.exception))

    def test_invalid_turboquant_kv_modes_fail_clearly(self):
        for flag in ('--kv', '--kv-key', '--kv-value'):
            with self.subTest(flag=flag):
                args = build_cli_parser().parse_args(['--engine', 'turboquant', flag, 'turbo3_tcq'])

                with self.assertRaises(SystemExit) as ctx:
                    validate_turboquant_kv_args(args)

                self.assertIn(f'Unsupported {flag} "turbo3_tcq"', str(ctx.exception))

    def test_invalid_tq3_kv_modes_fail_clearly(self):
        for flag in ('--kv', '--kv-key', '--kv-value'):
            with self.subTest(flag=flag):
                args = build_cli_parser().parse_args(['--engine', 'tq3', flag, 'turbo4'])

                with self.assertRaises(SystemExit) as ctx:
                    validate_tq3_kv_args(args)

                self.assertIn(f'Unsupported {flag} "turbo4"', str(ctx.exception))

    def test_cli_parser_accepts_kill_existing(self):
        args = build_cli_parser().parse_args(['--kill-existing'])

        self.assertTrue(args.kill_existing)

    def test_cli_help_documents_engines_examples_and_env_vars(self):
        help_text = build_cli_parser().format_help()

        self.assertIn('examples:', help_text)
        self.assertIn('llama-tui --engine turboquant --kv-key q8_0 --kv-value turbo4', help_text)
        self.assertIn('llama-tui --engine tq3', help_text)
        self.assertIn('llama-tui --engine buun --kill-existing', help_text)
        self.assertIn('supported runtimes: llama.cpp, turboquant, tq3, buun, vLLM saved model entries', help_text)
        self.assertIn('config path:', help_text)
        self.assertIn('TURBOQUANT_LLAMA_SERVER_BIN', help_text)
        self.assertIn('TQ3_LLAMA_SERVER_BIN', help_text)
        self.assertIn('BUUN_LLAMA_SERVER_BIN', help_text)
        self.assertIn('VLLM_COMMAND', help_text)
        self.assertIn('--help exits before curses starts', help_text)


class EngineSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_session(self, pid: int, engine: str):
        path = self.cache_dir / 'runtime_engine_sessions' / f'{pid}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"pid": {pid}, "engine": "{engine}"}}\n', encoding='utf-8')
        return path

    def test_stale_engine_sessions_are_pruned(self):
        stale = self.write_session(11111, 'llama.cpp')

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), patch('llama_tui.main.pid_is_alive', return_value=False):
            path = ensure_engine_session_lock('llama.cpp')

        self.assertFalse(stale.exists())
        self.assertTrue(path.exists())

    def test_same_engine_sessions_are_allowed(self):
        self.write_session(11111, 'llama.cpp')

        def alive(pid):
            return pid in {11111, os.getpid()}

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), patch('llama_tui.main.pid_is_alive', side_effect=alive):
            expected = engine_session_path()
            path = ensure_engine_session_lock('llama.cpp')

        self.assertEqual(path, expected)
        self.assertTrue(path.exists())

    def test_releasing_current_session_preserves_other_same_engine_sessions(self):
        old = self.write_session(11111, 'llama.cpp')

        def alive(pid):
            return pid in {11111, os.getpid()}

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), patch('llama_tui.main.pid_is_alive', side_effect=alive):
            path = ensure_engine_session_lock('llama.cpp')
            release_engine_session_lock(path)

        self.assertTrue(old.exists())
        self.assertFalse(path.exists())

    def test_different_live_engine_blocks_startup(self):
        session_path = self.write_session(11111, 'buun')

        def alive(pid):
            return pid == 11111

        process = {
            'pid': 11111,
            'command': 'python3 /home/jcampos/.local/bin/llama-tui',
            'cwd': '/home/jcampos/.cache/llmfit/models',
            'status': 'S (sleeping)',
            'state': 'S',
        }
        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), \
             patch('llama_tui.main.pid_is_alive', side_effect=alive), \
             patch('llama_tui.main.describe_pid', return_value=process):
            with self.assertRaises(SystemExit) as ctx:
                ensure_engine_session_lock('llama.cpp')

        message = str(ctx.exception)
        self.assertIn('Engine switch blocked', message)
        self.assertIn('PID 11111', message)
        self.assertIn('engine "buun"', message)
        self.assertIn('command: python3 /home/jcampos/.local/bin/llama-tui', message)
        self.assertIn('cwd: /home/jcampos/.cache/llmfit/models', message)
        self.assertIn(f'session: {session_path}', message)
        self.assertIn('--kill-existing', message)

    def test_interactive_prompt_accepts_kill_and_acquires_lock(self):
        blocker = self.write_session(11111, 'buun')
        terminated = []
        prompts = []

        def alive(pid):
            if pid == 11111:
                return pid not in terminated
            return pid == os.getpid()

        def terminate(pid):
            terminated.append(pid)
            return True, f'terminated PID {pid}'

        def prompt(engine, sessions):
            prompts.append((engine, sessions))
            return True

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), \
             patch('llama_tui.main.pid_is_alive', side_effect=alive), \
             patch('llama_tui.main.terminate_pid_group', side_effect=terminate):
            path = ensure_engine_session_lock('llama.cpp', interactive=True, prompt_fn=prompt)

        self.assertEqual(terminated, [11111])
        self.assertEqual(prompts[0][0], 'llama.cpp')
        self.assertEqual(prompts[0][1][0]['pid'], 11111)
        self.assertFalse(blocker.exists())
        self.assertTrue(path.exists())

    def test_interactive_prompt_declines_kill_and_exits_cleanly(self):
        blocker = self.write_session(11111, 'buun')

        def alive(pid):
            return pid == 11111

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), \
             patch('llama_tui.main.pid_is_alive', side_effect=alive), \
             patch('llama_tui.main.terminate_pid_group') as terminate:
            with self.assertRaises(SystemExit) as ctx:
                ensure_engine_session_lock('llama.cpp', interactive=True, prompt_fn=lambda _engine, _sessions: False)

        self.assertIn('Startup canceled', str(ctx.exception))
        self.assertFalse(terminate.called)
        self.assertTrue(blocker.exists())

    def test_kill_existing_terminates_blockers_removes_sessions_and_acquires_lock(self):
        blocker = self.write_session(11111, 'buun')
        terminated = []

        def alive(pid):
            if pid == 11111:
                return pid not in terminated
            return pid == os.getpid()

        def terminate(pid):
            terminated.append(pid)
            return True, f'terminated PID {pid}'

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), \
             patch('llama_tui.main.pid_is_alive', side_effect=alive), \
             patch('llama_tui.main.terminate_pid_group', side_effect=terminate):
            path = ensure_engine_session_lock('llama.cpp', kill_existing=True)

        self.assertEqual(terminated, [11111])
        self.assertFalse(blocker.exists())
        self.assertTrue(path.exists())
        self.assertEqual(last_engine_session_stop_count(), 1)

    def test_zombie_engine_session_is_pruned(self):
        stale = self.write_session(11111, 'buun')

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), \
             patch('llama_tui.main.os.kill', return_value=None), \
             patch('llama_tui.main.pid_state', return_value='Z'):
            path = ensure_engine_session_lock('llama.cpp')

        self.assertFalse(stale.exists())
        self.assertTrue(path.exists())


if __name__ == '__main__':
    unittest.main()
