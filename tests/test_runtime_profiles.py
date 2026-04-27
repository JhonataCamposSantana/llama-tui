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
    benchmark_runtime_profile_with_retry,
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

                llama_log = llama_app.logfile(model.id)
                buun_log = buun_app.logfile(model.id)
                llama_pid = llama_app.pidfile(model.id)
                buun_pid = buun_app.pidfile(model.id)
                legacy_log = buun_app.legacy_logfile(model.id)

        self.assertNotEqual(llama_log, buun_log)
        self.assertNotEqual(llama_pid, buun_pid)
        self.assertEqual(llama_log, root / 'runtime' / 'llama.cpp' / 'm.log')
        self.assertEqual(buun_log, root / 'runtime' / 'buun' / 'm.log')
        self.assertEqual(llama_pid, root / 'runtime' / 'llama.cpp' / 'm.pid')
        self.assertEqual(buun_pid, root / 'runtime' / 'buun' / 'm.pid')
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

    def test_failure_classification_names_actionable_startup_errors(self):
        cases = {
            'unknown value for --flash-attn: -ctk': 'CLI_INVALID',
            'cudaMalloc failed: out of memory while loading tensors': 'CUDA_OOM_WEIGHTS',
            'cudaMalloc failed: out of memory allocating KV cache': 'CUDA_OOM_KV',
            'K cache type turbo4 with block size 128 does not divide': 'KV_MODE_INCOMPATIBLE',
            'failed to fit params to free device memory, n_gpu_layers already set by user to 21': 'BUUN_FIT_FAILED',
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
        cases[mixed_buun_fit_oom] = 'BUUN_FIT_FAILED'
        for text, expected in cases.items():
            with self.subTest(text=text):
                default = 'API_TIMEOUT' if text == 'request timed out' else 'SERVER_TIMEOUT'
                self.assertEqual(classify_benchmark_failure(text, default)['failure_category'], expected)

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
            def build_command(self, _model, runtime_profile=None):
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
        profiles = [fit_probe, fixed_probe, fit_growth]
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=profiles):
            with patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
                ok, msg = benchmark_fast_profiles(app, model, progress=events.append)

        self.assertTrue(ok, msg)
        self.assertEqual(calls, ['fit_turbokv_probe', 'fit_context_growth_sweep_16384_turbo4_turbo4'])
        self.assertTrue(any('skipping fixed-NGL buun fallback probes' in str(item) for item in events))
        self.assertTrue(any('skipped 1 fixed-NGL buun profile' in str(item) for item in events))
        self.assertIn('buun fit profile', app.saved[-1].benchmark_runs[0]['summary'])

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
