import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_tui.app import AppConfig
from llama_tui.benchmark import (
    active_engine_runtime_profiles,
    adaptive_record_from_candidate,
    benchmark_all_models_runner,
    benchmark_exhaustive_profiles,
    benchmark_fast_profiles,
    classify_benchmark_failure,
    launch_with_failsafe,
    select_measured_profiles,
)
from llama_tui.hardware import HardwareProfile
from llama_tui.main import (
    build_cli_parser,
    ensure_engine_session_lock,
    engine_session_path,
    release_engine_session_lock,
    validate_buun_kv_args,
)
from llama_tui.models import ModelConfig
from llama_tui.runtime_profiles import (
    EngineCapabilities,
    RuntimeProfile,
    default_engine_capabilities,
    make_runtime_profile,
    parse_engine_capabilities,
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

    def test_capability_parser_detects_buun_flash_value_and_ngl(self):
        caps = parse_engine_capabilities(
            'usage: llama-server --flash-attn on|off|auto -ctk MODE -ctv MODE --parallel N -ngl N',
            engine_id='buun',
        )

        self.assertEqual(caps.flash_attn_syntax, 'value')
        self.assertTrue(caps.supports_ctk_ctv)
        self.assertTrue(caps.supports_parallel)
        self.assertEqual(caps.gpu_layers_flag, '-ngl')

    def test_capability_parser_detects_llama_cache_type_flags(self):
        caps = parse_engine_capabilities(
            'usage: llama-server --flash-attn on|off|auto --cache-type-k TYPE --cache-type-v TYPE --n-gpu-layers N --parallel N',
            engine_id='llama.cpp',
        )

        self.assertEqual(caps.flash_attn_syntax, 'value')
        self.assertTrue(caps.supports_cache_type_kv)
        self.assertEqual(caps.gpu_layers_flag, '--n-gpu-layers')

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

    def test_failure_classification_names_actionable_startup_errors(self):
        cases = {
            'unknown value for --flash-attn: -ctk': 'CLI_INVALID',
            'cudaMalloc failed: out of memory while loading tensors': 'CUDA_OOM_WEIGHTS',
            'cudaMalloc failed: out of memory allocating KV cache': 'CUDA_OOM_KV',
            'K cache type turbo4 with block size 128 does not divide': 'KV_MODE_INCOMPATIBLE',
            'failed to load model': 'MODEL_LOAD_FAILED',
            'server timed out': 'SERVER_TIMEOUT',
            'request timed out': 'API_TIMEOUT',
            'connection refused': 'PORT_UNREACHABLE',
            'chat template error': 'CHAT_TEMPLATE_ERROR',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                default = 'API_TIMEOUT' if text == 'request timed out' else 'SERVER_TIMEOUT'
                self.assertEqual(classify_benchmark_failure(text, default)['failure_category'], expected)

    def test_buun_heavy_moe_profiles_include_partial_offload_turbokv_from_traits(self):
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
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('buun')):
                with patch('llama_tui.benchmark.model_file_size', return_value=int(11.44 * 1024**3)):
                    profiles = active_engine_runtime_profiles(app, model, hardware)

        turbo_probe = next(item for item in profiles if item.name == 'kv_compression_probe')
        self.assertEqual(turbo_probe.ctx_size, 8192)
        self.assertEqual(turbo_probe.gpu_layers, 20)
        self.assertEqual(turbo_probe.parallel, 1)
        self.assertEqual(turbo_probe.kv_preset, 'turbo4/turbo4')
        self.assertIn(20, {item.gpu_layers for item in profiles if item.name.startswith('gpu_layer_sweep')})
        self.assertIn(32768, {item.ctx_size for item in profiles})

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
        self.write_session(11111, 'buun')

        def alive(pid):
            return pid == 11111

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), patch('llama_tui.main.pid_is_alive', side_effect=alive):
            with self.assertRaises(SystemExit) as ctx:
                ensure_engine_session_lock('llama.cpp')

        self.assertIn('Engine switch blocked', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
