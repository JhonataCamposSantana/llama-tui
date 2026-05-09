import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llama_tui.app import AppConfig
from llama_tui.hermes_benchmark import (
    benchmark_hermes_workflow,
    build_hermes_run_command,
    hermes_candidate_models,
    hermes_cli_preflight,
    isolated_hermes_env,
    run_hermes_floor_probe,
    run_hermes_task,
    write_temp_hermes_config,
)
from llama_tui.models import ModelConfig
from llama_tui.opencode_benchmark import WorkflowTask
from llama_tui.runtime_profiles import RuntimeProfile, make_runtime_profile


class HermesIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp.name) / 'models.json'
        self.app = AppConfig(self.config_path)
        self.app.hermes.home_root = str(Path(self.tmp.name) / 'hermes-home')
        self.model = ModelConfig(
            id='tiny',
            name='Tiny Model',
            path=__file__,
            alias='tiny-local',
            port=18080,
            ctx=8192,
            output=1024,
        )
        self.app.add_or_update(self.model)

    def tearDown(self):
        self.tmp.cleanup()

    def test_hermes_config_generation_is_isolated(self):
        ok, msg = self.app.generate_hermes_config(self.model)
        config_path = self.app.hermes_config_path(self.model)

        self.assertTrue(ok)
        self.assertIn(str(config_path), msg)
        self.assertTrue(config_path.exists())
        self.assertEqual(config_path.name, 'config.yaml')
        self.assertIn('provider: "custom"', config_path.read_text())
        self.assertIn('base_url: "http://127.0.0.1:18080/v1"', config_path.read_text())
        self.assertIn('context_length: 8192', config_path.read_text())
        self.assertTrue(str(config_path).startswith(str(Path(self.tmp.name))))

        config_path.write_text('old hermes config', encoding='utf-8')
        ok, msg = self.app.generate_hermes_config(self.model)
        backups = list((config_path.parent / 'backups').glob('config.*.yaml'))

        self.assertTrue(ok, msg)
        self.assertTrue(backups)
        self.assertEqual(backups[0].read_text(encoding='utf-8'), 'old hermes config')

    def test_hermes_settings_round_trip_context_policy(self):
        self.app.hermes.min_context_tokens = 65536
        self.app.hermes.allow_experimental_context_override = True
        self.app.hermes.experimental_context_override_tokens = 70000
        self.app.save()

        loaded = AppConfig(self.config_path)

        self.assertEqual(loaded.hermes.min_context_tokens, 65536)
        self.assertTrue(loaded.hermes.allow_experimental_context_override)
        self.assertEqual(loaded.hermes.experimental_context_override_tokens, 70000)

    def test_hermes_settings_normalize_legacy_default_floor(self):
        self.config_path.write_text(json.dumps({
            'hermes': {
                'home_root': str(Path(self.tmp.name) / 'hermes-home'),
                'min_context_tokens': 64000,
            },
            'models': [],
        }), encoding='utf-8')

        loaded = AppConfig(self.config_path)

        self.assertEqual(loaded.hermes.min_context_tokens, 65536)

    def test_temp_hermes_config_uses_experimental_context_override(self):
        self.app.hermes.allow_experimental_context_override = True
        self.app.hermes.experimental_context_override_tokens = 70000
        home = Path(self.tmp.name) / 'bench-home'

        config_path = write_temp_hermes_config(self.app, self.model, home)
        text = config_path.read_text()

        self.assertIn('context_length: 70000', text)

    def test_hermes_shell_command_includes_local_endpoint_and_keep_open_footer(self):
        command = self.app.build_hermes_shell_command(self.model, Path('/tmp/project'))

        self.assertIn('HERMES_HOME=', command)
        self.assertIn('OPENAI_BASE_URL=http://127.0.0.1:18080/v1', command)
        self.assertIn('hermes chat', command)
        self.assertNotIn('--provider custom', command)
        self.assertNotIn('--base-url http://127.0.0.1:18080/v1', command)
        self.assertIn('-m tiny-local', command)
        self.assertIn('Hermes exited with status', command)
        self.assertIn('Press Enter to close', command)

    def test_hermes_run_command_is_headless_query(self):
        command = build_hermes_run_command(self.app, self.model, Path('/tmp/project'), 'fix it')

        self.assertEqual(command[:2], ['hermes', 'chat'])
        self.assertNotIn('--provider', command)
        self.assertNotIn('--base-url', command)
        self.assertIn('-m', command)
        self.assertIn('tiny-local', command)
        self.assertIn('-t', command)
        self.assertIn('--max-turns', command)
        self.assertIn('--yolo', command)
        self.assertIn('--quiet', command)
        self.assertIn('-q', command)
        self.assertIn('fix it', command)

    def test_isolated_hermes_env_uses_temp_home(self):
        home = Path(self.tmp.name) / 'bench-home'
        env = isolated_hermes_env(self.app, self.model, home)

        self.assertEqual(env['HERMES_HOME'], str(home))
        self.assertEqual(env['HOME'], str(home))
        self.assertEqual(env['HERMES_INFERENCE_PROVIDER'], 'custom')
        self.assertEqual(env['OPENAI_BASE_URL'], 'http://127.0.0.1:18080/v1')
        self.assertEqual(env['OPENAI_API_KEY'], 'no-key-required')
        self.assertEqual(env['HERMES_YOLO_MODE'], '1')

    def test_hermes_preflight_missing_command_fails_cleanly(self):
        self.app.hermes.command = '/tmp/definitely-missing-hermes'

        ok, msg = hermes_cli_preflight(self.app)

        self.assertFalse(ok)
        self.assertIn('Hermes command not found', msg)

    def test_run_hermes_task_records_success_with_fake_process(self):
        task = WorkflowTask(
            name='fake',
            prompt='fix tests',
            files={'thing.py': 'VALUE = 1\n', 'test_thing.py': 'import unittest\n'},
        )
        fake_run = {
            'returncode': 0,
            'timed_out': False,
            'no_output_timeout': False,
            'idle_output_timeout': False,
            'aborted': False,
            'elapsed': 1.25,
            'first_output': 0.2,
            'stdout': ['python -m unittest -q'],
            'stderr': [],
            'json_event_tail': [],
            'raw_event_tail': ['python -m unittest -q'],
            'min_ram_available': 2 * 1024**3,
            'min_gpu_memory_free': 1024**3,
        }
        with patch.object(self.app, 'append_log'):
            with patch('llama_tui.hermes_benchmark.run_process_with_metrics', return_value=fake_run):
                with patch('llama_tui.hermes_benchmark.verify_fixture', return_value=(True, 'OK')):
                    sample = run_hermes_task(self.app, self.model, task)

        self.assertTrue(sample['ok'])
        self.assertEqual(sample['status'], 'tests passed')
        self.assertEqual(sample['exit_code'], 0)
        self.assertTrue(sample['unittest_command_seen'])
        self.assertTrue(sample['config_path'].endswith('config.yaml'))

    def test_run_hermes_task_uses_adaptive_timeout_policy(self):
        task = WorkflowTask(
            name='fake',
            prompt='fix tests',
            files={'thing.py': 'VALUE = 1\n', 'test_thing.py': 'import unittest\n'},
        )
        self.model.architecture_type = 'moe'
        fake_run = {
            'returncode': 0,
            'timed_out': False,
            'no_output_timeout': False,
            'idle_output_timeout': False,
            'aborted': False,
            'elapsed': 1.25,
            'first_output': 0.2,
            'first_meaningful_output': 0.2,
            'first_process_output': 0.1,
            'startup_output_seen': False,
            'resolved_no_output_timeout': 180,
            'resolved_idle_output_timeout': 240,
            'stdout': ['python -m unittest -q'],
            'stderr': [],
            'json_event_tail': [],
            'raw_event_tail': ['python -m unittest -q'],
            'min_ram_available': 2 * 1024**3,
            'min_gpu_memory_free': 1024**3,
        }
        with patch.object(self.app, 'append_log'):
            with patch('llama_tui.hermes_benchmark.current_process_pressure_payload', return_value={'process_pressure_level': 'high'}):
                with patch('llama_tui.hermes_benchmark.run_process_with_metrics', return_value=fake_run) as run:
                    with patch('llama_tui.hermes_benchmark.verify_fixture', return_value=(True, 'OK')):
                        sample = run_hermes_task(self.app, self.model, task)

        self.assertTrue(sample['ok'])
        self.assertEqual(run.call_args.kwargs['no_output_timeout'], 180)
        self.assertEqual(run.call_args.kwargs['idle_output_timeout'], 240)
        self.assertEqual(sample['timeout_policy'], 'moe+high_pressure')

    def test_run_hermes_task_records_failure_tails_with_fake_process(self):
        task = WorkflowTask(
            name='fake',
            prompt='fix tests',
            files={'thing.py': 'VALUE = 1\n', 'test_thing.py': 'import unittest\n'},
        )
        fake_run = {
            'returncode': 2,
            'timed_out': False,
            'no_output_timeout': False,
            'idle_output_timeout': False,
            'aborted': False,
            'elapsed': 0.25,
            'first_output': 0.05,
            'stdout': ['usage: hermes chat'],
            'stderr': ['hermes: error: bad flag'],
            'json_event_tail': [],
            'raw_event_tail': ['hermes: error: bad flag'],
            'min_ram_available': 2 * 1024**3,
            'min_gpu_memory_free': 1024**3,
        }
        with patch.object(self.app, 'append_log') as append_log:
            with patch('llama_tui.hermes_benchmark.run_process_with_metrics', return_value=fake_run):
                with patch('llama_tui.hermes_benchmark.verify_fixture', return_value=(False, 'tests failed')):
                    sample = run_hermes_task(self.app, self.model, task)

        self.assertFalse(sample['ok'])
        self.assertEqual(sample['status'], 'hermes command failed')
        self.assertEqual(sample['exit_code'], 2)
        self.assertIn('hermes: error: bad flag', sample['stderr_tail'])
        self.assertTrue(sample['config_path'].endswith('config.yaml'))
        self.assertTrue(any('Hermes stderr tail' in str(call.args[1]) for call in append_log.call_args_list))

    def test_hermes_benchmark_persists_score_and_history_with_fakes(self):
        self.model.measured_profiles = {
            'auto': {
                'status': 'ok',
                'ctx': 8192,
                'ctx_per_slot': 8192,
                'parallel': 1,
                'threads': 6,
                'ngl': 0,
                'output': 1024,
                'tokens_per_sec': 24.0,
                'extra_args': [],
            },
            'hermes_floor': {
                'status': 'ok',
                'ctx': 65536,
                'ctx_per_slot': 65536,
                'parallel': 1,
                'threads': 6,
                'ngl': 0,
                'output': 1024,
                'tokens_per_sec': 12.0,
                'extra_args': [],
            },
        }
        sample = {
            'task': 'fake',
            'command_preview': 'hermes chat -q fix',
            'ok': True,
            'tests_ok': True,
            'status': 'tests passed',
            'exit_code': 0,
            'timed_out': False,
            'no_output_timeout': False,
            'idle_output_timeout': False,
            'aborted': False,
            'elapsed': 1.0,
            'first_output': 0.1,
            'min_ram_available': 2 * 1024**3,
            'min_gpu_memory_free': 1024**3,
            'stdout_tail': ['python -m unittest -q'],
            'stderr_tail': [],
            'json_event_tail': [],
            'raw_event_tail': ['python -m unittest -q'],
            'unittest_command_seen': True,
            'context_required': 0,
            'detail': 'OK',
        }
        profile = SimpleNamespace(short_summary=lambda: 'fake hardware')
        with patch('llama_tui.hermes_benchmark.hermes_cli_preflight', return_value=(True, 'Hermes CLI ready')):
            with patch.object(self.app, 'health', return_value=('STOPPED', '')):
                with patch.object(self.app, 'get_pid', return_value=None):
                    with patch.object(self.app, 'hardware_profile', return_value=profile):
                        with patch.object(self.app, 'start', return_value=(True, 'started')):
                            with patch.object(self.app, 'wait_until_ready', return_value=(True, 'ready')):
                                with patch.object(self.app, 'stop', return_value=(True, 'stopped')):
                                    with patch('llama_tui.hermes_benchmark.run_hermes_task', return_value=sample):
                                        ok, msg = benchmark_hermes_workflow(self.app, self.model)

        saved = self.app.get_model(self.model.id)
        self.assertTrue(ok)
        self.assertIn('Hermes workflow winner', msg)
        self.assertGreater(saved.last_hermes_benchmark_score, 0.0)
        self.assertEqual(saved.benchmark_runs[0]['kind'], 'hermes')
        self.assertIn('auto', saved.measured_profiles)
        self.assertIn('hermes_floor', saved.measured_profiles)
        self.assertIn('hermes_cache_ram', saved.measured_profiles)
        self.assertEqual(saved.measured_profiles['hermes_cache_ram']['cache_ram_mib'], 0)
        self.assertIn('hermes_cache_ram', saved.benchmark_runs[0]['winners'])
        self.assertEqual(saved.cache_ram, self.model.cache_ram)
        self.assertNotIn('hermes_measured_hermes_floor_0', saved.measured_profiles)

    def test_hermes_benchmark_skips_candidates_below_context_floor(self):
        self.model.measured_profiles = {
            'opencode_ready': {
                'status': 'ok',
                'ctx': 35968,
                'ctx_per_slot': 35968,
                'parallel': 1,
                'threads': 6,
                'ngl': 0,
                'output': 1024,
                'tokens_per_sec': 12.0,
                'extra_args': [],
            },
        }
        profile = SimpleNamespace(short_summary=lambda: 'fake hardware')
        with patch('llama_tui.hermes_benchmark.hermes_cli_preflight', return_value=(True, 'Hermes CLI ready')):
            with patch.object(self.app, 'health', return_value=('STOPPED', '')):
                with patch.object(self.app, 'get_pid', return_value=None):
                    with patch.object(self.app, 'hardware_profile', return_value=profile):
                        with patch('llama_tui.hermes_benchmark.run_hermes_floor_probe', return_value=(False, 'probe failed')):
                            with patch.object(self.app, 'start') as start:
                                ok, msg = benchmark_hermes_workflow(self.app, self.model)

        saved = self.app.get_model(self.model.id)
        self.assertFalse(ok)
        self.assertIn('not Hermes-ready', msg)
        start.assert_not_called()
        self.assertEqual(saved.benchmark_runs[0]['status'], 'failed')
        row = saved.last_hermes_benchmark_results[0]
        self.assertEqual(row['status'], 'not Hermes-ready')
        self.assertEqual(row['required_context'], 65536)
        self.assertEqual(row['actual_ctx_per_slot'], 35968)

    def test_hermes_benchmark_experimental_override_allows_low_context_candidate(self):
        self.app.hermes.allow_experimental_context_override = True
        self.app.hermes.experimental_context_override_tokens = 70000
        self.model.measured_profiles = {
            'auto': {
                'status': 'ok',
                'ctx': 35968,
                'ctx_per_slot': 35968,
                'parallel': 1,
                'threads': 6,
                'ngl': 0,
                'output': 1024,
                'tokens_per_sec': 12.0,
                'extra_args': [],
            },
        }
        sample = {
            'task': 'fake',
            'command_preview': 'hermes chat -q fix',
            'ok': True,
            'tests_ok': True,
            'status': 'tests passed',
            'exit_code': 0,
            'timed_out': False,
            'no_output_timeout': False,
            'idle_output_timeout': False,
            'aborted': False,
            'elapsed': 1.0,
            'first_output': 0.1,
            'min_ram_available': 2 * 1024**3,
            'min_gpu_memory_free': 1024**3,
            'stdout_tail': ['python -m unittest -q'],
            'stderr_tail': [],
            'json_event_tail': [],
            'raw_event_tail': ['python -m unittest -q'],
            'unittest_command_seen': True,
            'context_required': 0,
            'detail': 'OK',
            'configured_context_length': 70000,
            'actual_ctx_per_slot': 35968,
            'required_context': 65536,
            'experimental_context_override': True,
        }
        profile = SimpleNamespace(short_summary=lambda: 'fake hardware')
        with patch('llama_tui.hermes_benchmark.hermes_cli_preflight', return_value=(True, 'Hermes CLI ready')):
            with patch.object(self.app, 'health', return_value=('STOPPED', '')):
                with patch.object(self.app, 'get_pid', return_value=None):
                    with patch.object(self.app, 'hardware_profile', return_value=profile):
                        with patch.object(self.app, 'start', return_value=(True, 'started')) as start:
                            with patch.object(self.app, 'wait_until_ready', return_value=(True, 'ready')):
                                with patch.object(self.app, 'stop', return_value=(True, 'stopped')):
                                    with patch('llama_tui.hermes_benchmark.run_hermes_task', return_value=sample):
                                        ok, msg = benchmark_hermes_workflow(self.app, self.model)

        saved = self.app.get_model(self.model.id)
        self.assertTrue(ok)
        self.assertIn('Hermes workflow winner', msg)
        self.assertEqual(start.call_count, 4)
        self.assertEqual([call.args[0].cache_ram for call in start.call_args_list], [0, 512, 1024, 2048])
        row = saved.last_hermes_benchmark_results[0]
        self.assertTrue(row['experimental_context_override'])
        self.assertEqual(row['configured_context_length'], 70000)
        self.assertEqual(row['actual_ctx_per_slot'], 35968)

    def test_hermes_benchmark_with_zero_passes_is_failed_not_winner(self):
        self.model.measured_profiles = {
            'hermes_floor': {
                'status': 'ok',
                'ctx': 65536,
                'ctx_per_slot': 65536,
                'parallel': 1,
                'threads': 6,
                'ngl': 0,
                'output': 1024,
                'tokens_per_sec': 12.0,
                'extra_args': [],
            },
        }
        sample = {
            'task': 'fake',
            'command_preview': 'hermes chat -q fix',
            'ok': False,
            'tests_ok': False,
            'status': 'hermes command failed',
            'exit_code': 2,
            'timed_out': False,
            'no_output_timeout': False,
            'idle_output_timeout': False,
            'aborted': False,
            'elapsed': 0.2,
            'first_output': 0.1,
            'min_ram_available': 2 * 1024**3,
            'min_gpu_memory_free': 1024**3,
            'stdout_tail': [],
            'stderr_tail': ['bad flag'],
            'json_event_tail': [],
            'raw_event_tail': ['bad flag'],
            'unittest_command_seen': False,
            'context_required': 0,
            'detail': 'bad flag',
        }
        profile = SimpleNamespace(short_summary=lambda: 'fake hardware')
        with patch('llama_tui.hermes_benchmark.hermes_cli_preflight', return_value=(True, 'Hermes CLI ready')):
            with patch.object(self.app, 'health', return_value=('STOPPED', '')):
                with patch.object(self.app, 'get_pid', return_value=None):
                    with patch.object(self.app, 'hardware_profile', return_value=profile):
                        with patch.object(self.app, 'start', return_value=(True, 'started')):
                            with patch.object(self.app, 'wait_until_ready', return_value=(True, 'ready')):
                                with patch.object(self.app, 'stop', return_value=(True, 'stopped')):
                                    with patch('llama_tui.hermes_benchmark.run_hermes_task', return_value=sample):
                                        ok, msg = benchmark_hermes_workflow(self.app, self.model)

        saved = self.app.get_model(self.model.id)
        self.assertFalse(ok)
        self.assertIn('no candidate passed', msg)
        self.assertEqual(saved.last_hermes_benchmark_score, 0.0)
        self.assertEqual(saved.benchmark_runs[0]['kind'], 'hermes')
        self.assertEqual(saved.benchmark_runs[0]['status'], 'failed')

    def test_hermes_partial_pass_does_not_become_winner(self):
        self.model.measured_profiles = {
            'hermes_floor': {
                'status': 'ok',
                'ctx': 65536,
                'ctx_per_slot': 65536,
                'parallel': 1,
                'threads': 6,
                'ngl': 0,
                'output': 1024,
                'tokens_per_sec': 12.0,
                'extra_args': [],
            },
        }
        ok_sample = {
            'task': 'fix_calc',
            'command_preview': 'hermes chat -q fix',
            'ok': True,
            'tests_ok': True,
            'status': 'tests passed',
            'exit_code': 0,
            'timed_out': False,
            'no_output_timeout': False,
            'idle_output_timeout': False,
            'aborted': False,
            'elapsed': 1.0,
            'first_output': 0.1,
            'first_meaningful_output': 0.1,
            'first_process_output': 0.05,
            'min_ram_available': 2 * 1024**3,
            'min_gpu_memory_free': 1024**3,
            'stdout_tail': ['python -m unittest -q'],
            'stderr_tail': [],
            'json_event_tail': [],
            'raw_event_tail': ['python -m unittest -q'],
            'unittest_command_seen': True,
            'context_required': 0,
            'detail': 'OK',
        }
        timeout_sample = dict(ok_sample)
        timeout_sample.update({
            'task': 'add_slugify',
            'ok': False,
            'tests_ok': False,
            'status': 'hermes no output timeout',
            'timed_out': True,
            'no_output_timeout': True,
            'unittest_command_seen': False,
            'detail': 'no meaningful Hermes output for 180s',
        })
        profile = SimpleNamespace(short_summary=lambda: 'fake hardware')
        with patch('llama_tui.hermes_benchmark.hermes_cli_preflight', return_value=(True, 'Hermes CLI ready')):
            with patch.object(self.app, 'health', return_value=('STOPPED', '')):
                with patch.object(self.app, 'get_pid', return_value=None):
                    with patch.object(self.app, 'hardware_profile', return_value=profile):
                        with patch.object(self.app, 'start', return_value=(True, 'started')):
                            with patch.object(self.app, 'wait_until_ready', return_value=(True, 'ready')):
                                with patch.object(self.app, 'stop', return_value=(True, 'stopped')):
                                    with patch('llama_tui.hermes_benchmark.run_hermes_task', side_effect=[ok_sample, timeout_sample] * 4):
                                        ok, msg = benchmark_hermes_workflow(self.app, self.model)

        saved = self.app.get_model(self.model.id)
        self.assertFalse(ok)
        self.assertIn('no candidate completed all tasks', msg)
        self.assertEqual(saved.last_hermes_benchmark_score, 0.0)
        self.assertEqual(saved.last_hermes_benchmark_results[0]['passed'], 1)
        self.assertEqual(saved.last_hermes_benchmark_results[0]['tasks'], 2)

    def test_hermes_candidates_skip_under_floor_and_run_all_viable(self):
        self.model.measured_profiles = {
            'opencode_ready': {'status': 'ok', 'ctx': 32768, 'ctx_per_slot': 32768, 'parallel': 1, 'tokens_per_sec': 20.0},
            'auto': {'status': 'ok', 'ctx': 8192, 'ctx_per_slot': 8192, 'parallel': 1, 'tokens_per_sec': 30.0},
            'hermes_floor': {'status': 'ok', 'ctx': 65536, 'ctx_per_slot': 65536, 'parallel': 1, 'tokens_per_sec': 10.0},
            'long_context': {'status': 'ok', 'ctx': 98304, 'ctx_per_slot': 98304, 'parallel': 1, 'tokens_per_sec': 11.0},
        }

        candidates = hermes_candidate_models(self.app, self.model)

        self.assertEqual([candidate.ctx for _label, _tier, candidate, _detail, _runtime in candidates], [65536, 98304])

    def test_hermes_candidates_reuse_turboquant_history_runtime_profile(self):
        app = AppConfig(self.config_path, runtime_profile=make_runtime_profile('turboquant', 'llama-server'))
        model = ModelConfig(
            id='tq',
            name='TurboQuant Model',
            path=__file__,
            alias='tq',
            port=18080,
            ctx=8192,
            benchmark_runs=[{
                'kind': 'server_fast',
                'records': [{
                    'status': 'ok',
                    'measurement_type': 'full',
                    'objective': 'opencode_ready',
                    'engine': 'turboquant',
                    'ctx': 65536,
                    'ctx_per_slot': 65536,
                    'parallel': 1,
                    'ngl': 0,
                    'tokens_per_sec': 9.0,
                    'runtime_profile': 'fit_context_growth_sweep_65536_q8_0_turbo4',
                    'kv_preset': 'q8_0/turbo4',
                    'runtime_fit': True,
                    'fit_context': 4096,
                    'runtime_no_warmup': True,
                    'gpu_layers_mode': 'fit',
                }],
            }],
        )

        candidates = hermes_candidate_models(app, model)

        self.assertEqual(len(candidates), 1)
        _label, tier, candidate, _detail, runtime_profile = candidates[0]
        self.assertEqual(tier, 'history')
        self.assertEqual(candidate.ctx, 65536)
        self.assertIsNotNone(runtime_profile)
        self.assertEqual(runtime_profile.engine_id, 'turboquant')
        self.assertTrue(runtime_profile.fit)
        self.assertEqual(runtime_profile.kv_preset, 'q8_0/turbo4')

    def test_hermes_benchmark_triggers_floor_probe_when_missing_exact_floor(self):
        self.model.measured_profiles = {
            'long_context': {
                'status': 'ok',
                'ctx': 98304,
                'ctx_per_slot': 98304,
                'parallel': 1,
                'threads': 6,
                'ngl': 0,
                'output': 1024,
                'tokens_per_sec': 12.0,
                'extra_args': [],
            },
        }
        self.app.add_or_update(self.model)
        sample = {
            'task': 'fake',
            'command_preview': 'hermes chat -q fix',
            'ok': True,
            'tests_ok': True,
            'status': 'tests passed',
            'exit_code': 0,
            'timed_out': False,
            'no_output_timeout': False,
            'idle_output_timeout': False,
            'aborted': False,
            'elapsed': 1.0,
            'first_output': 0.1,
            'min_ram_available': 2 * 1024**3,
            'min_gpu_memory_free': 1024**3,
            'stdout_tail': ['python -m unittest -q'],
            'stderr_tail': [],
            'json_event_tail': [],
            'raw_event_tail': ['python -m unittest -q'],
            'unittest_command_seen': True,
            'context_required': 0,
            'detail': 'OK',
        }

        def fake_probe(app, model, _profile, _required, *_args, **_kwargs):
            saved = app.get_model(model.id) or model
            saved.benchmark_runs = [{
                'kind': 'server_fast',
                'records': [{
                    'status': 'ok',
                    'measurement_type': 'full',
                    'objective': 'opencode_ready',
                    'engine': 'llama.cpp',
                    'ctx': 65536,
                    'ctx_per_slot': 65536,
                    'parallel': 1,
                    'threads': 6,
                    'ngl': 0,
                    'output': 1024,
                    'tokens_per_sec': 10.0,
                }],
            }]
            app.add_or_update(saved)
            return True, 'probe ok'

        profile = SimpleNamespace(short_summary=lambda: 'fake hardware')
        with patch('llama_tui.hermes_benchmark.hermes_cli_preflight', return_value=(True, 'Hermes CLI ready')):
            with patch.object(self.app, 'health', return_value=('STOPPED', '')):
                with patch.object(self.app, 'get_pid', return_value=None):
                    with patch.object(self.app, 'hardware_profile', return_value=profile):
                        with patch('llama_tui.hermes_benchmark.run_hermes_floor_probe', side_effect=fake_probe) as probe:
                            with patch.object(self.app, 'start', return_value=(True, 'started')) as start:
                                with patch.object(self.app, 'wait_until_ready', return_value=(True, 'ready')):
                                    with patch.object(self.app, 'stop', return_value=(True, 'stopped')):
                                        with patch('llama_tui.hermes_benchmark.run_hermes_task', return_value=sample):
                                            ok, msg = benchmark_hermes_workflow(self.app, self.model)

        self.assertTrue(ok, msg)
        probe.assert_called_once()
        self.assertEqual(start.call_count, 5)
        started_ctxs = [call.args[0].ctx for call in start.call_args_list]
        self.assertEqual(started_ctxs, [65536, 65536, 65536, 65536, 98304])
        self.assertEqual([call.args[0].cache_ram for call in start.call_args_list[:4]], [0, 512, 1024, 2048])

    def test_hermes_floor_probe_persists_reusable_floor_profile(self):
        runtime_profile = RuntimeProfile(
            engine_id='llama.cpp',
            ctx_size=65536,
            gpu_layers=0,
            parallel=1,
            name='serve_65536',
        )
        measured = {
            'status': 'ok',
            'measurement_type': 'full',
            'objective': 'opencode_ready',
            'engine': 'llama.cpp',
            'ctx': 65536,
            'ctx_per_slot': 65536,
            'parallel': 1,
            'tokens_per_sec': 10.0,
            'runtime_profile': 'serve_65536',
        }
        profile = SimpleNamespace(short_summary=lambda: 'fake hardware')

        with patch('llama_tui.hermes_benchmark.select_hermes_floor_probe_profiles', return_value=[runtime_profile]):
            with patch(
                'llama_tui.hermes_benchmark.benchmark_runtime_profile_with_retry',
                return_value=(True, False, [measured], [measured], 1),
            ):
                ok, msg = run_hermes_floor_probe(self.app, self.model, profile, 65536)

        saved = self.app.get_model(self.model.id)
        self.assertTrue(ok, msg)
        self.assertEqual(saved.measured_profiles['hermes_floor']['ctx_per_slot'], 65536)
        self.assertEqual(saved.benchmark_runs[0]['kind'], 'server_fast')
        self.assertEqual(saved.benchmark_runs[0]['winners']['hermes_floor']['ctx_per_slot'], 65536)


if __name__ == '__main__':
    unittest.main()
