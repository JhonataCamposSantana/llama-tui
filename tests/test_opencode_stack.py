import os
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llama_tui.app import AppConfig, render_terminal_template, terminal_command_for_launcher
from llama_tui.benchmark import (
    adaptive_context_search,
    adaptive_record_from_candidate,
    annotate_spectrum_records,
    apply_measured_profile,
    benchmark_all_models_deep,
    benchmark_config_fingerprint,
    benchmark_profile_is_fresh,
    benchmark_exhaustive_candidate_with_retry,
    benchmark_exhaustive_profiles,
    benchmark_fast_profiles,
    break_refinement_contexts,
    build_benchmark_run,
    context_knee_refinement_contexts,
    exhaustive_context_ladder,
    exhaustive_parallel_values,
    fast_benchmark_contexts,
    fast_benchmark_parallel_values,
    machine_best_summary,
    machine_benchmark_rows,
    model_from_measured_profile,
    parallel_refinement_values,
    parse_context_requirement,
    safe_bootstrap_candidate_models,
    select_adaptive_candidate_mix,
    select_measured_profiles,
    smart_break_refinement_contexts,
    smart_fast_contexts,
    smart_measurement_contexts,
    smart_should_continue_optional,
    smart_should_try_q8,
    upsert_benchmark_run,
)
from llama_tui.control import CancelToken
from llama_tui.discovery import detected_model_from_path
from llama_tui.hardware import HardwareProfile
from llama_tui.models import ModelConfig
from llama_tui.opencode_benchmark import (
    build_opencode_run_command,
    compact_sample_details,
    ctx_per_slot,
    detected_unittest_command,
    isolated_opencode_env,
    opencode_candidate_models,
    run_process_with_metrics,
    sample_timeout_type,
    score_opencode_samples,
)


def process_active(pid: int) -> bool:
    stat_path = Path(f'/proc/{pid}/stat')
    try:
        stat = stat_path.read_text()
        end = stat.rfind(')')
        state = stat[end + 2:].split()[0] if end != -1 else ''
        if state in ('Z', 'X'):
            return False
    except Exception:
        pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def measured_profile(tokens_per_sec: float, ctx: int, ram_gib: int = 8):
    return {
        'status': 'ok',
        'tokens_per_sec': tokens_per_sec,
        'seconds': 1.0,
        'ctx': ctx,
        'ctx_per_slot': ctx,
        'parallel': 1,
        'ram_available': ram_gib * 1024**3,
        'benchmarked_at': '2026-04-23T00:00:00',
    }


def benchmarked_model(model_id: str, tokens_per_sec: float = 50.0, ctx: int = 8192, status: str = 'done'):
    model = ModelConfig(id=model_id, name=model_id.title(), path=f'{model_id}.gguf', alias=model_id, port=18000 + len(model_id))
    model.default_benchmark_status = status
    model.benchmark_fingerprint = f'fp-{model_id}'
    auto = measured_profile(tokens_per_sec, ctx)
    model.measured_profiles = {
        'auto': dict(auto),
        'fast_chat': dict(auto, tokens_per_sec=tokens_per_sec + 10.0),
        'long_context': dict(auto, ctx=ctx, ctx_per_slot=ctx),
        'opencode_ready': dict(auto, ctx=ctx, ctx_per_slot=ctx),
    }
    return model


class FakeBenchmarkApp:
    def __init__(self, models):
        self.models = list(models)
        self.health_by_id = {model.id: ('STOPPED', 'stopped') for model in self.models}
        self.managed_running = set()
        self.unmanaged_running = set()
        self.stop_calls = []
        self.saved = []

    def model_fingerprint(self, model):
        return f'fp-{model.id}'

    def get_model(self, model_id):
        return next((model for model in self.models if model.id == model_id), None)

    def add_or_update(self, model):
        for idx, existing in enumerate(self.models):
            if existing.id == model.id:
                self.models[idx] = model
                self.saved.append(model.id)
                return
        self.models.append(model)
        self.saved.append(model.id)

    def health(self, model):
        if model.id in self.managed_running or model.id in self.unmanaged_running:
            return 'READY', 'responding'
        return self.health_by_id.get(model.id, ('STOPPED', 'stopped'))

    def get_pid(self, model, discover=True, managed_only=False):
        if model.id in self.managed_running:
            return 1000 + len(model.id)
        if managed_only or not discover:
            return None
        if model.id in self.unmanaged_running:
            return 2000 + len(model.id)
        return None

    def stop(self, model, managed_only=False):
        self.stop_calls.append((model.id, managed_only))
        self.managed_running.discard(model.id)
        return True, 'stopped'


class DeepBenchmarkAllTests(unittest.TestCase):
    def fake_runner(self, calls):
        def runner(app, model, progress=None, cancel_token=None):
            calls.append(model.id)
            saved = ModelConfig(**asdict(model))
            saved.default_benchmark_status = 'done'
            saved.benchmark_fingerprint = app.model_fingerprint(saved)
            auto = measured_profile(60.0 + len(calls), 8192 + len(calls) * 1024)
            saved.measured_profiles = {
                'auto': dict(auto),
                'fast_chat': dict(auto, tokens_per_sec=auto['tokens_per_sec'] + 5.0),
                'long_context': dict(auto),
                'opencode_ready': dict(auto),
            }
            app.add_or_update(saved)
            if progress:
                progress({'event': 'benchmark_started', 'message': 'inner start', 'completed': 0, 'total': 1})
                progress({'event': 'benchmark_done', 'message': 'inner done', 'completed': 1, 'total': 1})
            return True, 'ok'
        return runner

    def test_deep_benchmark_all_skips_fresh_and_disabled_by_default(self):
        fresh = benchmarked_model('fresh')
        pending = benchmarked_model('pending', status='pending')
        stale = benchmarked_model('stale')
        stale.benchmark_fingerprint = 'old'
        failed = benchmarked_model('failed', status='failed')
        disabled = benchmarked_model('disabled')
        disabled.enabled = False
        app = FakeBenchmarkApp([fresh, pending, stale, failed, disabled])
        calls = []
        events = []

        ok, msg = benchmark_all_models_deep(
            app,
            progress=events.append,
            benchmark_runner=self.fake_runner(calls),
            start_runner=lambda *_args, **_kwargs: (True, 'ready'),
        )

        self.assertTrue(ok, msg)
        self.assertEqual(calls, ['pending', 'stale', 'failed'])
        self.assertIn('3 benchmarked', msg)
        skipped_messages = '\n'.join(str(event.get('message', '')) for event in events if isinstance(event, dict))
        self.assertIn('fresh skipped', skipped_messages)
        self.assertIn('disabled skipped', skipped_messages)

    def test_deep_benchmark_all_force_refreshes_fresh_models(self):
        fresh = benchmarked_model('fresh')
        disabled = benchmarked_model('disabled')
        disabled.enabled = False
        app = FakeBenchmarkApp([fresh, disabled])
        calls = []

        ok, msg = benchmark_all_models_deep(
            app,
            force=True,
            benchmark_runner=self.fake_runner(calls),
            start_runner=lambda *_args, **_kwargs: (True, 'ready'),
        )

        self.assertTrue(ok, msg)
        self.assertEqual(calls, ['fresh'])

    def test_deep_benchmark_all_skips_unmanaged_running_model(self):
        model = benchmarked_model('external', status='pending')
        app = FakeBenchmarkApp([model])
        app.unmanaged_running.add('external')
        calls = []

        ok, msg = benchmark_all_models_deep(
            app,
            benchmark_runner=self.fake_runner(calls),
            start_runner=lambda *_args, **_kwargs: (True, 'ready'),
        )

        self.assertTrue(ok, msg)
        self.assertEqual(calls, [])
        self.assertIn('1 skipped', msg)

    def test_deep_benchmark_all_stops_and_restores_managed_running_model(self):
        model = benchmarked_model('managed', status='pending')
        app = FakeBenchmarkApp([model])
        app.managed_running.add('managed')
        calls = []
        restore_calls = []

        def restarter(app, model, progress=None, cancel_token=None):
            restore_calls.append(model.id)
            if progress:
                progress('ready')
            return True, 'ready'

        ok, msg = benchmark_all_models_deep(
            app,
            benchmark_runner=self.fake_runner(calls),
            start_runner=restarter,
        )

        self.assertTrue(ok, msg)
        self.assertEqual(calls, ['managed'])
        self.assertEqual(restore_calls, ['managed'])
        self.assertEqual(app.stop_calls, [('managed', True)])
        self.assertIn('1 restored', msg)

    def test_machine_best_summary_uses_fresh_profiles_only(self):
        fast = benchmarked_model('fast', tokens_per_sec=100.0, ctx=8192)
        balanced = benchmarked_model('balanced', tokens_per_sec=75.0, ctx=64000)
        balanced.measured_profiles['auto']['ram_available'] = 16 * 1024**3
        balanced.measured_profiles['long_context']['ctx_per_slot'] = 64000
        balanced.measured_profiles['long_context']['ctx'] = 64000
        balanced.measured_profiles['opencode_ready']['ctx_per_slot'] = 64000
        balanced.measured_profiles['opencode_ready']['ctx'] = 64000
        stale = benchmarked_model('stale', tokens_per_sec=500.0, ctx=131072)
        stale.benchmark_fingerprint = 'old'
        app = FakeBenchmarkApp([fast, balanced, stale])

        rows = machine_benchmark_rows(app)
        summary = machine_best_summary(app)

        self.assertEqual({row['model_id'] for row in rows}, {'fast', 'balanced'})
        self.assertTrue(benchmark_profile_is_fresh(app, fast))
        self.assertFalse(benchmark_profile_is_fresh(app, stale))
        self.assertEqual(summary['categories']['fastest_chat']['model_id'], 'fast')
        self.assertEqual(summary['categories']['longest_context']['model_id'], 'balanced')
        self.assertEqual(summary['categories']['opencode_ready']['model_id'], 'balanced')
        self.assertEqual(summary['machine_pick']['model_id'], 'balanced')


class OpencodeStackHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp.name) / 'models.json'
        self.app = AppConfig(self.config_path)
        self.model = ModelConfig(
            id='tiny',
            name='Tiny Model',
            path=__file__,
            alias='tiny-local',
            port=18080,
            runtime='vllm',
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_workspace_validation(self):
        ok, path, reason = self.app.validate_workspace_path(self.tmp.name)
        self.assertTrue(ok)
        self.assertEqual(path, Path(self.tmp.name).resolve())
        self.assertEqual(reason, '')

        ok, path, reason = self.app.validate_workspace_path(Path(self.tmp.name) / 'missing')
        self.assertFalse(ok)
        self.assertIsNone(path)
        self.assertIn('does not exist', reason)

    def test_opencode_model_ref(self):
        self.assertEqual(self.app.opencode_provider_key(self.model), 'local-tiny')
        self.assertEqual(self.app.opencode_model_ref(self.model), 'local-tiny/tiny-local')

    def test_default_terminal_command_shape(self):
        command = terminal_command_for_launcher('xterm', 'OpenCode Tiny', Path('/tmp/project'), 'echo hello')
        self.assertEqual(command[:4], ['xterm', '-T', 'OpenCode Tiny', '-e'])
        self.assertIn('cd /tmp/project && echo hello', command)

    def test_extra_terminal_launchers(self):
        ptyxis = terminal_command_for_launcher('ptyxis', 'OpenCode Tiny', Path('/tmp/project'), 'echo hello')
        self.assertEqual(ptyxis[:3], ['ptyxis', '--working-directory', '/tmp/project'])
        self.assertIn('bash', ptyxis)

        xdg = terminal_command_for_launcher('xdg-terminal-exec', 'OpenCode Tiny', Path('/tmp/project'), 'echo hello')
        self.assertEqual(xdg[:3], ['xdg-terminal-exec', 'bash', '-lc'])
        self.assertIn('cd /tmp/project && echo hello', xdg)

        ghostty = terminal_command_for_launcher('ghostty', 'OpenCode Tiny', Path('/tmp/project'), 'echo hello')
        self.assertEqual(ghostty[:3], ['ghostty', '--title', 'OpenCode Tiny'])
        self.assertIn('--working-directory', ghostty)

    def test_terminal_template_quotes_command(self):
        command = render_terminal_template(
            'xterm -T {title} -e bash -lc {cmd}',
            'OpenCode Tiny',
            Path('/tmp/project'),
            'echo hello && sleep 1',
        )
        self.assertEqual(command, ['xterm', '-T', 'OpenCode Tiny', '-e', 'bash', '-lc', 'echo hello && sleep 1'])

    def test_custom_terminal_command_wins(self):
        self.app.opencode.terminal_command = 'xterm -T {title} -e bash -lc {cmd}'
        with patch.object(self.app, 'detect_terminal_launcher', return_value='/usr/bin/konsole'):
            ok, command, label = self.app.build_terminal_command('OpenCode Tiny', Path('/tmp/project'), 'echo hello')

        self.assertTrue(ok)
        self.assertEqual(label, 'custom')
        self.assertEqual(command[:4], ['xterm', '-T', 'OpenCode Tiny', '-e'])

    def test_local_terminal_route(self):
        with patch.object(self.app, 'detect_terminal_launcher', return_value='/usr/bin/konsole'):
            ok, command, label = self.app.build_terminal_command('OpenCode Tiny', Path('/tmp/project'), 'echo hello')

        self.assertTrue(ok)
        self.assertEqual(label, 'local:konsole')
        self.assertEqual(command[:2], ['/usr/bin/konsole', '--workdir'])

    def test_host_terminal_route_reenters_container(self):
        with patch.object(self.app, 'detect_terminal_launcher', return_value=None):
            with patch.object(self.app, 'detect_host_terminal_launcher', return_value=('/usr/bin/host-spawn', 'konsole')):
                with patch('llama_tui.app.container_environment_detected', return_value=True):
                    with patch('llama_tui.app.current_container_name', return_value='my-distrobox'):
                        ok, command, label = self.app.build_terminal_command('OpenCode Tiny', Path('/tmp/project'), 'echo hello')

        self.assertTrue(ok)
        self.assertEqual(label, 'host:host-spawn/konsole')
        self.assertEqual(command[:3], ['/usr/bin/host-spawn', '-no-pty', 'konsole'])
        self.assertIn('distrobox enter my-distrobox', ' '.join(command))

    def test_host_terminal_failure_is_explicit(self):
        with patch.object(self.app, 'detect_terminal_launcher', return_value=None):
            with patch.object(self.app, 'detect_host_terminal_launcher', return_value=('/usr/bin/host-spawn', None)):
                with patch('llama_tui.app.container_environment_detected', return_value=True):
                    ok, command, message = self.app.build_terminal_command('OpenCode Tiny', Path('/tmp/project'), 'echo hello')

        self.assertFalse(ok)
        self.assertEqual(command, [])
        self.assertIn('No terminal launcher was visible', message)
        self.assertIn('Host bridge host-spawn is available', message)
        self.assertIn('opencode.terminal_command', message)

    def test_opencode_shell_command_uses_config_and_model(self):
        self.app.opencode.path = '/tmp/opencode.json'
        command = self.app.build_opencode_shell_command(self.model, Path('/tmp/project'))
        self.assertIn('OPENCODE_CONFIG=/tmp/opencode.json', command)
        self.assertIn('OPENCODE_DISABLE_AUTOUPDATE=true', command)
        self.assertIn('OPENCODE_DISABLE_PRUNE=true', command)
        self.assertIn('OPENCODE_DISABLE_MODELS_FETCH=true', command)
        self.assertIn('opencode /tmp/project', command)
        self.assertIn("--model local-tiny/tiny-local", command)
        self.assertIn('cd /tmp/project', command)
        self.assertIn('OpenCode exited with status', command)
        self.assertIn('Press Enter to close', command)

    def test_opencode_run_command_shape(self):
        command = build_opencode_run_command(self.app, self.model, Path('/tmp/project'), 'fix it')
        self.assertEqual(command[:3], ['opencode', 'run', '--pure'])
        self.assertIn('--model', command)
        self.assertIn('local-tiny/tiny-local', command)
        self.assertIn('--agent', command)
        self.assertIn('build', command)
        self.assertIn('--format', command)
        self.assertIn('json', command)
        self.assertIn('--dir', command)
        self.assertIn('--dangerously-skip-permissions', command)
        self.assertIn('--print-logs', command)
        self.assertIn('--log-level', command)
        self.assertNotIn('xterm', command)

    def test_opencode_sample_details_keep_headless_process_diagnostics(self):
        sample = {
            'task': 'fix_calc',
            'command_preview': 'opencode run --pure --dir /tmp/work fix',
            'status': 'opencode no output timeout',
            'ok': False,
            'tests_ok': False,
            'exit_code': -9,
            'timed_out': True,
            'no_output_timeout': True,
            'idle_output_timeout': False,
            'unittest_command_seen': False,
            'context_required': 9616,
            'stdout_tail': ['stdout line'],
            'stderr_tail': ['stderr line'],
        }

        details = compact_sample_details([sample])

        self.assertEqual(sample_timeout_type(sample), 'no_output')
        self.assertEqual(details[0]['command_preview'], sample['command_preview'])
        self.assertEqual(details[0]['exit_code'], -9)
        self.assertEqual(details[0]['timeout_type'], 'no_output')
        self.assertEqual(details[0]['context_required'], 9616)
        self.assertEqual(details[0]['stderr_tail'], ['stderr line'])

    def test_isolated_opencode_env(self):
        home = Path(self.tmp.name) / 'home'
        config = home / '.config' / 'opencode' / 'opencode.json'
        env = isolated_opencode_env(home, config)
        self.assertEqual(env['HOME'], str(home))
        self.assertEqual(env['XDG_CONFIG_HOME'], str(home / '.config'))
        self.assertEqual(env['OPENCODE_CONFIG'], str(config))
        self.assertEqual(env['OPENCODE_DISABLE_AUTOUPDATE'], 'true')
        self.assertEqual(env['OPENCODE_DISABLE_PRUNE'], 'true')
        self.assertEqual(env['OPENCODE_DISABLE_MODELS_FETCH'], 'true')

    def test_persistence_fields_round_trip(self):
        model = ModelConfig(
            id='tiny',
            name='Tiny Model',
            path='example/tiny-model',
            alias='tiny-local',
            port=18080,
            runtime='vllm',
            benchmark_fingerprint='abc123',
            default_benchmark_status='done',
            default_benchmark_at='2026-04-14T12:00:00',
        )
        self.app.add_or_update(model)

        loaded = AppConfig(self.config_path).get_model('tiny')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.benchmark_fingerprint, 'abc123')
        self.assertEqual(loaded.default_benchmark_status, 'done')
        self.assertEqual(loaded.default_benchmark_at, '2026-04-14T12:00:00')
        self.assertEqual(loaded.benchmark_runs, [])

    def test_benchmark_runs_round_trip_and_trim(self):
        model = ModelConfig(
            id='tiny',
            name='Tiny Model',
            path=__file__,
            alias='tiny-local',
            port=18080,
            runtime='vllm',
        )
        for idx in range(12):
            upsert_benchmark_run(
                model,
                build_benchmark_run(
                    f'run-{idx}',
                    'server',
                    'done',
                    [],
                    {},
                    f'2026-04-14T12:00:{idx:02d}',
                ),
            )
        self.app.add_or_update(model)

        loaded = AppConfig(self.config_path).get_model('tiny')

        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.benchmark_runs), 10)
        self.assertEqual(loaded.benchmark_runs[0]['id'], 'run-11')
        self.assertEqual(loaded.benchmark_runs[-1]['id'], 'run-2')

    def test_measured_profiles_round_trip_and_apply(self):
        model = ModelConfig(
            id='tiny',
            name='Tiny Model',
            path='example/tiny-model',
            alias='tiny-local',
            port=18080,
            runtime='vllm',
            measured_profiles={
                'auto': {
                    'status': 'ok',
                    'ctx': 12345,
                    'parallel': 3,
                    'threads': 7,
                    'ngl': 99,
                    'output': 1024,
                    'cache_ram': 0,
                    'extra_args': ['--batch-size', '512'],
                    'tokens_per_sec': 42.0,
                }
            },
        )
        self.app.add_or_update(model)

        loaded = AppConfig(self.config_path).get_model('tiny')
        self.assertIsNotNone(loaded)
        self.assertIn('auto', loaded.measured_profiles)
        ok, msg = apply_measured_profile(loaded, 'auto')
        self.assertTrue(ok)
        self.assertIn('42.00 tok/s', msg)
        self.assertEqual(loaded.ctx, 12345)
        self.assertEqual(loaded.parallel, 3)
        self.assertEqual(loaded.extra_args, ['--batch-size', '512'])

    def test_measured_mode_launch_profile_uses_saved_values(self):
        model = ModelConfig(
            id='tiny',
            name='Tiny Model',
            path=__file__,
            alias='tiny-local',
            port=18080,
            ctx=999999,
            parallel=7,
            optimize_mode='measured_auto',
        )

        ok, profile, msg = self.app.safe_launch_profile(model)

        self.assertTrue(ok)
        self.assertEqual(profile['ctx'], 999999)
        self.assertEqual(profile['parallel'], 7)
        self.assertIn('measured profile', msg)


class OpencodeWorkflowScoreTests(unittest.TestCase):
    def test_successful_fast_samples_score_above_failures(self):
        good = score_opencode_samples([
            {
                'ok': True,
                'elapsed': 10.0,
                'first_output': 2.0,
                'min_ram_available': 4 * 1024**3,
                'min_gpu_memory_free': 2 * 1024**3,
            },
            {
                'ok': True,
                'elapsed': 12.0,
                'first_output': 3.0,
                'min_ram_available': 3 * 1024**3,
                'min_gpu_memory_free': 2 * 1024**3,
            },
        ])
        bad = score_opencode_samples([
            {
                'ok': False,
                'elapsed': 80.0,
                'first_output': 30.0,
                'min_ram_available': 700 * 1024**2,
                'min_gpu_memory_free': 200 * 1024**2,
            }
        ])
        self.assertGreater(good, bad)

    def test_opencode_candidates_prefer_measured_profiles(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(
            id='bigctx',
            name='Big Context',
            path=__file__,
            alias='bigctx',
            port=18180,
            ctx_max=32768,
            parallel=8,
            measured_profiles={
                'opencode_ready': {
                    'status': 'ok',
                    'ctx': 20000,
                    'parallel': 1,
                    'threads': 6,
                    'ngl': 0,
                    'output': 2048,
                    'extra_args': [],
                    'tokens_per_sec': 12.0,
                }
            },
        )
        candidates = opencode_candidate_models(model, profile)

        self.assertTrue(candidates)
        self.assertEqual(candidates[0][0], 'opencode_ready')
        self.assertEqual(ctx_per_slot(candidates[0][2]), 20000)

    def test_opencode_candidates_are_dynamic_for_tiny_context(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(id='smallctx', name='Small Context', path=__file__, alias='smallctx', port=18181, ctx_max=4096, parallel=1)

        candidates = opencode_candidate_models(model, profile)
        self.assertTrue(candidates)
        self.assertTrue(all(ctx_per_slot(candidate) <= 4096 for _preset, _tier, candidate, _msg in candidates))

    def test_detects_visible_unittest_command(self):
        self.assertTrue(detected_unittest_command(['running python -m unittest -q now']))
        self.assertTrue(detected_unittest_command(['python3 -m unittest']))
        self.assertFalse(detected_unittest_command(['pytest -q']))

    def test_context_requirement_parser(self):
        self.assertEqual(parse_context_requirement('request (9616 tokens) exceeds the available context size'), 9616)
        self.assertEqual(parse_context_requirement('needs about 14000 ctx'), 14000)
        self.assertEqual(parse_context_requirement('all good'), 0)

    def test_adaptive_context_search_binary_refines_failure_band(self):
        probed = []

        def probe(ctx):
            probed.append(ctx)
            return ctx <= 10000

        successes, failures = adaptive_context_search(2048, 20000, probe, max_probes=10)

        self.assertTrue(successes)
        self.assertTrue(failures)
        self.assertLessEqual(max(successes), 10000)
        self.assertGreater(min(failures), max(successes))
        self.assertGreater(len(probed), 3)

    def test_exhaustive_context_ladder_uses_tiered_steps(self):
        self.assertEqual(exhaustive_context_ladder(2048, 8192), [2048, 4096, 6144, 8192])
        self.assertEqual(
            exhaustive_context_ladder(2048, 81920),
            [
                2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384,
                20480, 24576, 28672, 32768, 36864, 40960, 45056, 49152,
                53248, 57344, 61440, 65536, 73728, 81920,
            ],
        )
        self.assertEqual(exhaustive_context_ladder(3000, 7600), [3000, 5048, 7096, 7600])

    def test_break_and_knee_context_refinement_helpers(self):
        self.assertEqual(break_refinement_contexts(20480, 32768, {20480, 32768}), [22528, 24576, 26624, 28672, 30720])
        self.assertEqual(smart_break_refinement_contexts(20480, 32768, {20480, 32768}), [22528, 26624, 30720])
        records = [
            {'status': 'ok', 'ctx': 2048, 'ctx_per_slot': 2048, 'tokens_per_sec': 80.0},
            {'status': 'ok', 'ctx': 8192, 'ctx_per_slot': 8192, 'tokens_per_sec': 40.0},
        ]
        self.assertEqual(context_knee_refinement_contexts(records, {2048, 8192}, 8192), [5120])

    def test_smart_measurement_contexts_focus_frontier_and_floors(self):
        contexts = smart_measurement_contexts(
            [2048, 4096, 8192, 16384, 32768],
            [65536],
            2048,
            131072,
            chat_floor=5000,
            opencode_floor=20000,
        )

        self.assertIn(2048, contexts)
        self.assertIn(8192, contexts)
        self.assertIn(32768, contexts)
        self.assertLessEqual(len(contexts), 5)
        self.assertEqual(smart_fast_contexts([2048, 4096, 8192, 16384], 5000), [8192, 16384])

    def test_smart_fingerprint_ignores_objective_but_tracks_runtime_settings(self):
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200)
        candidate = ModelConfig(**asdict(model))
        first = benchmark_config_fingerprint(candidate)
        candidate.ctx = 4096
        second = benchmark_config_fingerprint(candidate)

        self.assertNotEqual(first, second)
        long_candidate = ModelConfig(**asdict(model))
        fast_candidate = ModelConfig(**asdict(model))
        long_candidate.ctx = fast_candidate.ctx = 4096
        long_candidate.parallel = fast_candidate.parallel = 1
        self.assertEqual(benchmark_config_fingerprint(long_candidate), benchmark_config_fingerprint(fast_candidate))

    def test_smart_optional_budget_continues_only_until_winners_exist(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200)
        fast_model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=4096, parallel=2)
        long_model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=32768, parallel=1)
        measured = [
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'fast_chat', 'model': fast_model, 'tokens_per_sec': 80.0, 'ctx_per_slot': 2048, 'parallel': 2, 'ram_available': 8 * 1024**3},
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'long_context', 'model': long_model, 'tokens_per_sec': 20.0, 'ctx_per_slot': 32768, 'parallel': 1, 'ram_available': 8 * 1024**3},
        ]

        self.assertFalse(smart_should_continue_optional(0.0, measured, model, profile, now=3600.0))
        self.assertTrue(smart_should_continue_optional(0.0, measured[:1], model, profile, now=3600.0))

    def test_smart_q8_gate_requires_llamacpp_gpu_and_meaningful_gain(self):
        profile = HardwareProfile(
            cpu_logical=8,
            cpu_physical=4,
            memory_total=64 * 1024**3,
            memory_available=48 * 1024**3,
            gpu_memory_total=8 * 1024**3,
            gpu_memory_free=6 * 1024**3,
        )
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, runtime='llama.cpp', ctx_max=65536)

        self.assertTrue(smart_should_try_q8(model, profile, default_best_ctx=32768, default_break_ctx=65536))
        with patch('llama_tui.benchmark.candidate_safe_context_estimate', side_effect=[10000, 11499]):
            self.assertFalse(smart_should_try_q8(model, profile, default_best_ctx=32768, default_break_ctx=0))
        with patch('llama_tui.benchmark.candidate_safe_context_estimate', side_effect=[10000, 11600]):
            self.assertTrue(smart_should_try_q8(model, profile, default_best_ctx=32768, default_break_ctx=0))
        model.runtime = 'vllm'
        self.assertFalse(smart_should_try_q8(model, profile, default_best_ctx=32768, default_break_ctx=65536))

    def test_exhaustive_parallel_values_are_power_of_two_and_refined(self):
        profile = HardwareProfile(cpu_logical=20, cpu_physical=10, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        self.assertEqual(exhaustive_parallel_values(profile), [1, 2, 4, 8, 16])
        self.assertEqual(parallel_refinement_values(profile, 4, {1, 2, 4, 8, 16}), [3, 5])

    def test_fast_benchmark_planners_are_shallow_and_limited(self):
        profile = HardwareProfile(cpu_logical=20, cpu_physical=10, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx_min=2048, ctx_max=20000)

        with patch('llama_tui.benchmark.candidate_safe_context_estimate', return_value=12288):
            contexts = fast_benchmark_contexts(model, profile)

        self.assertEqual(fast_benchmark_parallel_values(profile), [1, 2, 4])
        self.assertEqual(fast_benchmark_parallel_values(HardwareProfile(cpu_logical=2, cpu_physical=1)), [1, 2])
        self.assertIn(2048, contexts)
        self.assertIn(8192, contexts)
        self.assertIn(16384, contexts)
        self.assertIn(12288, contexts)
        self.assertLessEqual(max(contexts), 20000)

    def test_exhaustive_candidate_retries_and_marks_break_point(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx_max=8192)

        class FakeApp:
            def __init__(self):
                self.starts = 0

            def start(self, _candidate):
                self.starts += 1
                return False, 'boom'

            def stop(self, _candidate, managed_only=True):
                return True, 'stopped'

            def hardware_profile(self, refresh=False):
                return profile

        ok, broke, records, measured, completed = benchmark_exhaustive_candidate_with_retry(
            FakeApp(),
            model,
            profile,
            'long_context',
            4096,
            1,
            'default',
            None,
            None,
            0,
            10,
        )

        self.assertFalse(ok)
        self.assertTrue(broke)
        self.assertEqual(len(records), 2)
        self.assertEqual(measured, [])
        self.assertEqual(completed, 2)
        self.assertFalse(records[0]['break_point'])
        self.assertTrue(records[1]['break_point'])

    def test_smart_profiles_reuse_long_context_for_opencode_ready(self):
        profile = HardwareProfile(cpu_logical=2, cpu_physical=1, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(
            id='m',
            name='M',
            path=__file__,
            alias='m',
            port=18200,
            runtime='vllm',
            ctx_min=2048,
            ctx_max=6144,
            output=256,
        )
        calls = []

        class FakeApp:
            opencode = SimpleNamespace(path='')

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model):
                return None

            def hardware_profile(self, refresh=False):
                return profile

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, model):
                self.saved.append(model)

        def fake_probe(_app, candidate, objective, _progress, _cancel_token):
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'probe ok',
                tokens_per_sec=10.0,
                seconds=0.1,
                ram_available=8 * 1024**3,
            )
            return record, True

        def fake_benchmark(_app, candidate, objective, _progress, _cancel_token):
            calls.append((objective, candidate.ctx, candidate.parallel))
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=100.0 / max(1, candidate.parallel),
                seconds=1.0,
            )
            measured = dict(record)
            measured['model'] = ModelConfig(**asdict(candidate))
            return record, measured

        app = FakeApp()
        with patch('llama_tui.benchmark.benchmark_frontier_probe_candidate', side_effect=fake_probe), \
             patch('llama_tui.benchmark.benchmark_adaptive_candidate', side_effect=fake_benchmark):
            ok, msg = benchmark_exhaustive_profiles(app, model)

        self.assertTrue(ok, msg)
        self.assertEqual(
            [ctx for objective, ctx, _parallel in calls if objective == 'long_context'],
            [2048, 4096, 5632, 6144],
        )
        self.assertEqual([ctx for objective, ctx, _parallel in calls if objective == 'opencode_ready'], [])
        self.assertTrue(any(objective == 'fast_chat' for objective, _ctx, _parallel in calls))
        saved = app.saved[-1]
        self.assertEqual(saved.benchmark_runs[0]['kind'], 'server')
        self.assertEqual(saved.measured_profiles['opencode_ready'].get('reused_from'), 'long_context')
        self.assertIn('auto/smart-bounded', saved.last_benchmark_profile)

    def test_fast_benchmark_profiles_persist_server_fast_history(self):
        profile = HardwareProfile(cpu_logical=4, cpu_physical=2, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(
            id='m',
            name='M',
            path=__file__,
            alias='m',
            port=18200,
            runtime='vllm',
            ctx_min=2048,
            ctx_max=8192,
            output=256,
        )

        class FakeApp:
            opencode = SimpleNamespace(path='')

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model):
                return None

            def hardware_profile(self, refresh=False):
                return profile

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, model):
                self.saved.append(model)

        def fake_benchmark(_app, candidate, objective, _progress, _cancel_token):
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=100.0 / max(1, candidate.parallel),
                seconds=1.0,
                ram_available=8 * 1024**3,
            )
            measured = dict(record)
            measured['model'] = ModelConfig(**asdict(candidate))
            return record, measured

        app = FakeApp()
        with patch('llama_tui.benchmark.benchmark_adaptive_candidate', side_effect=fake_benchmark):
            ok, msg = benchmark_fast_profiles(app, model)

        self.assertTrue(ok, msg)
        saved = app.saved[-1]
        self.assertEqual(saved.benchmark_runs[0]['kind'], 'server_fast')
        self.assertIn('auto', saved.measured_profiles)
        self.assertIn('auto/fast', saved.last_benchmark_profile)
        self.assertGreater(saved.last_benchmark_tokens_per_sec, 0.0)

    def test_fast_benchmark_stops_higher_contexts_after_confirmed_break(self):
        profile = HardwareProfile(cpu_logical=4, cpu_physical=2, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(
            id='m',
            name='M',
            path=__file__,
            alias='m',
            port=18200,
            runtime='vllm',
            ctx_min=2048,
            ctx_max=16384,
            output=256,
        )
        calls = []

        class FakeApp:
            opencode = SimpleNamespace(path='')

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model):
                return None

            def hardware_profile(self, refresh=False):
                return profile

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, model):
                self.saved.append(model)

        def fake_benchmark(_app, candidate, objective, _progress, _cancel_token):
            calls.append((objective, candidate.ctx, candidate.parallel))
            if objective == 'long_context' and candidate.ctx > 2048:
                return adaptive_record_from_candidate(candidate, objective, 'start failed', detail='oom'), None
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=50.0,
                seconds=1.0,
                ram_available=8 * 1024**3,
            )
            measured = dict(record)
            measured['model'] = ModelConfig(**asdict(candidate))
            return record, measured

        app = FakeApp()
        with patch('llama_tui.benchmark.benchmark_adaptive_candidate', side_effect=fake_benchmark):
            ok, msg = benchmark_fast_profiles(app, model)

        self.assertTrue(ok, msg)
        self.assertEqual([ctx for objective, ctx, _parallel in calls if objective == 'long_context'], [2048, 8192, 8192])
        self.assertNotIn(('long_context', 16384, 1), calls)

    def test_adaptive_candidate_mix_keeps_each_spectrum_objective(self):
        items = []
        for ctx in (2048, 4096, 8192, 16384):
            items.append(('fast_chat', ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=ctx, parallel=2), f'fast/{ctx}'))
        items.append(('long_context', ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=32768, parallel=1), 'long/32768'))
        items.append(('opencode_ready', ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=24000, parallel=1), 'opencode/24000'))

        selected = select_adaptive_candidate_mix(items, limit=4)
        objectives = {item[0] for item in selected}

        self.assertIn('fast_chat', objectives)
        self.assertIn('long_context', objectives)
        self.assertIn('opencode_ready', objectives)

    def test_select_measured_profiles_picks_distinct_winners(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200)
        fast_model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=8192, parallel=2)
        long_model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=32000, parallel=1)
        measured = [
            {'status': 'ok', 'objective': 'fast_chat', 'model': fast_model, 'tokens_per_sec': 60.0, 'seconds': 2.0, 'ctx_per_slot': 4096, 'parallel': 2, 'ram_available': 8 * 1024**3},
            {'status': 'ok', 'objective': 'long_context', 'model': long_model, 'tokens_per_sec': 20.0, 'seconds': 3.0, 'ctx_per_slot': 32000, 'parallel': 1, 'ram_available': 8 * 1024**3},
        ]

        winners = select_measured_profiles(model, measured, profile)

        self.assertEqual(winners['fast_chat']['tokens_per_sec'], 60.0)
        self.assertEqual(winners['long_context']['ctx_per_slot'], 32000)
        self.assertEqual(winners['opencode_ready']['ctx_per_slot'], 32000)
        self.assertIn('auto', winners)
        self.assertIn('selection_score', winners['auto'])
        self.assertIn('quality score', winners['auto']['selection_reason'])

    def test_select_measured_profiles_ignores_probe_rows_for_winners(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200)
        probe_model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=65536, parallel=1)
        full_model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=8192, parallel=1)
        measured = [
            {'status': 'ok', 'measurement_type': 'probe', 'objective': 'long_context', 'model': probe_model, 'tokens_per_sec': 999.0, 'ctx_per_slot': 65536, 'parallel': 1},
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'long_context', 'model': full_model, 'tokens_per_sec': 20.0, 'ctx_per_slot': 8192, 'parallel': 1},
        ]

        winners = select_measured_profiles(model, measured, profile)

        self.assertEqual(winners['long_context']['ctx_per_slot'], 8192)
        self.assertEqual(winners['opencode_ready'].get('reused_from'), 'long_context')

    def test_select_measured_profiles_prefers_rows_meeting_opencode_floor(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(
            id='m',
            name='M',
            path=__file__,
            alias='m',
            port=18200,
            output=256,
            last_opencode_benchmark_results=[{'detail': 'request (12000 tokens) exceeds context'}],
        )
        small_model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=8192, parallel=1)
        floor_model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=16000, parallel=1)
        measured = [
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'long_context', 'model': small_model, 'tokens_per_sec': 50.0, 'ctx_per_slot': 8192, 'parallel': 1},
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'long_context', 'model': floor_model, 'tokens_per_sec': 30.0, 'ctx_per_slot': 16000, 'parallel': 1},
        ]

        winners = select_measured_profiles(model, measured, profile)

        self.assertEqual(winners['opencode_ready']['ctx_per_slot'], 16000)
        self.assertIn('OpenCode floor 12000', winners['opencode_ready']['selection_reason'])

    def test_select_measured_profiles_auto_uses_quality_score(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, output=256)
        fast_model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=4096, parallel=1)
        balanced_model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=32000, parallel=1)
        measured = [
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'fast_chat', 'model': fast_model, 'tokens_per_sec': 100.0, 'ctx_per_slot': 4096, 'parallel': 1, 'ram_available': 0},
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'long_context', 'model': balanced_model, 'tokens_per_sec': 70.0, 'ctx_per_slot': 32000, 'parallel': 1, 'ram_available': 16 * 1024**3},
        ]

        winners = select_measured_profiles(model, measured, profile)

        self.assertEqual(winners['fast_chat']['tokens_per_sec'], 100.0)
        self.assertEqual(winners['auto']['ctx_per_slot'], 32000)
        self.assertGreater(winners['auto']['selection_score'], 0.0)

    def test_annotates_spectrum_tradeoff_rows(self):
        records = [
            {'status': 'ok', 'objective': 'fast_chat', 'tokens_per_sec': 80.0, 'ctx': 4096, 'ctx_per_slot': 2048, 'parallel': 2},
            {'status': 'ok', 'objective': 'long_context', 'tokens_per_sec': 20.0, 'ctx': 32768, 'ctx_per_slot': 32768, 'parallel': 1},
            {'status': 'ok', 'objective': 'auto', 'tokens_per_sec': 50.0, 'ctx': 12000, 'ctx_per_slot': 12000, 'parallel': 1},
        ]
        winners = {
            'fast_chat': {'tokens_per_sec': 80.0, 'ctx': 4096, 'parallel': 2},
            'long_context': {'tokens_per_sec': 20.0, 'ctx': 32768, 'parallel': 1},
            'opencode_ready': {'tokens_per_sec': 20.0, 'ctx': 32768, 'parallel': 1},
            'auto': {'tokens_per_sec': 50.0, 'ctx': 12000, 'parallel': 1},
        }

        annotate_spectrum_records(records, winners)
        labels = '\n'.join(str(row.get('spectrum_label', '')) for row in records)

        self.assertIn('Possible', labels)
        self.assertIn('Fastest', labels)
        self.assertIn('Ideal', labels)
        self.assertIn('Highest Context', labels)
        self.assertIn('OpenCode-ready', labels)
        self.assertIn('Winner', labels)

    def test_annotates_failed_and_breakpoint_rows(self):
        records = [
            {'status': 'ok', 'objective': 'fast_chat', 'tokens_per_sec': 80.0, 'ctx': 4096, 'ctx_per_slot': 4096, 'parallel': 1},
            {'status': 'start failed', 'objective': 'long_context', 'tokens_per_sec': 0.0, 'ctx': 6144, 'ctx_per_slot': 6144, 'parallel': 1, 'break_point': True},
        ]

        annotate_spectrum_records(records, {
            'fast_chat': {'tokens_per_sec': 80.0, 'ctx': 4096, 'parallel': 1},
            'long_context': {'tokens_per_sec': 80.0, 'ctx': 4096, 'parallel': 1},
            'opencode_ready': {'tokens_per_sec': 80.0, 'ctx': 4096, 'parallel': 1},
            'auto': {'tokens_per_sec': 80.0, 'ctx': 4096, 'parallel': 1},
        })

        self.assertIn('Failed', records[1]['spectrum_label'])
        self.assertIn('Break Point', records[1]['spectrum_label'])

    def test_model_from_measured_profile(self):
        model = ModelConfig(
            id='m',
            name='M',
            path=__file__,
            alias='m',
            port=18201,
            measured_profiles={'fast_chat': {'status': 'ok', 'ctx': 8192, 'parallel': 2, 'threads': 4, 'ngl': 0, 'output': 1024, 'extra_args': []}},
        )
        candidate = model_from_measured_profile(model, 'fast_chat')
        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.ctx, 8192)
        self.assertEqual(candidate.parallel, 2)


class DiscoveryDefaultsTests(unittest.TestCase):
    def test_detected_model_uses_generic_safe_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'Qwen3-30B-A3B-Instruct-Q4_K_M.gguf'
            path.write_bytes(b'not real gguf metadata')
            model = detected_model_from_path(path, [])

        self.assertEqual(model.id, 'qwen3-30b-a3b-instruct-q4-k-m')
        self.assertEqual(model.ctx, 2048)
        self.assertEqual(model.ctx_min, 2048)
        self.assertEqual(model.ctx_max, 131072)
        self.assertEqual(model.ngl, 0)
        self.assertEqual(model.temp, 0.7)
        self.assertEqual(model.output, 2048)
        self.assertEqual(model.optimize_tier, 'safe')
        self.assertEqual(model.memory_reserve_percent, 40)
        self.assertEqual(model.default_benchmark_status, 'pending')

    def test_personal_paths_are_not_embedded_in_constants(self):
        text = Path('llama_tui/constants.py').read_text()
        self.assertNotIn('/var/home/jcampos', text)

    def test_safe_bootstrap_candidates_ignore_model_name(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=16 * 1024**3, memory_available=8 * 1024**3)
        first = ModelConfig(id='a', name='Qwen Opus Coder Gemma', path=__file__, alias='a', port=18100)
        second = ModelConfig(id='b', name='Plain Local Model', path=__file__, alias='b', port=18101)
        first_candidates = safe_bootstrap_candidate_models(first, profile)
        second_candidates = safe_bootstrap_candidate_models(second, profile)

        first_shape = [(preset, tier, c.ctx, c.ngl, c.parallel, c.memory_reserve_percent, c.extra_args) for preset, tier, c, _ in first_candidates]
        second_shape = [(preset, tier, c.ctx, c.ngl, c.parallel, c.memory_reserve_percent, c.extra_args) for preset, tier, c, _ in second_candidates]
        self.assertEqual(first_shape, second_shape)


class ProcessLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = AppConfig(Path(self.tmp.name) / 'models.json')

    def tearDown(self):
        self.tmp.cleanup()

    def child_command(self, pidfile: Path):
        code = (
            'import pathlib, subprocess, sys, time; '
            'child=subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]); '
            'pathlib.Path(sys.argv[1]).write_text(str(child.pid)); '
            'time.sleep(60)'
        )
        return [sys.executable, '-c', code, str(pidfile)]

    def test_timeout_kills_process_group_children(self):
        pidfile = Path(self.tmp.name) / 'child.pid'
        result = run_process_with_metrics(
            self.child_command(pidfile),
            Path(self.tmp.name),
            os.environ.copy(),
            timeout=0.2,
            app=self.app,
        )
        child_pid = int(pidfile.read_text())
        time.sleep(0.3)
        self.assertTrue(result['timed_out'])
        self.assertFalse(process_active(child_pid))

    def test_no_output_timeout_kills_process(self):
        result = run_process_with_metrics(
            [sys.executable, '-c', 'import time; time.sleep(60)'],
            Path(self.tmp.name),
            os.environ.copy(),
            timeout=10,
            app=self.app,
            no_output_timeout=0.2,
            idle_output_timeout=0,
        )
        self.assertTrue(result['timed_out'])
        self.assertTrue(result['no_output_timeout'])
        self.assertEqual(result['stdout'], [])
        self.assertEqual(result['stderr'], [])

    def test_stdout_stderr_tails_are_persisted(self):
        code = 'import sys; print("{\\"event\\":\\"ok\\"}"); print("python -m unittest -q", file=sys.stderr)'
        result = run_process_with_metrics(
            [sys.executable, '-c', code],
            Path(self.tmp.name),
            os.environ.copy(),
            timeout=10,
            app=self.app,
        )
        self.assertEqual(result['returncode'], 0)
        self.assertIn('{"event":"ok"}', result['stdout'])
        self.assertIn('python -m unittest -q', result['stderr'])
        self.assertTrue(result['json_event_tail'])

    def test_abort_kills_process_group_children(self):
        pidfile = Path(self.tmp.name) / 'child-abort.pid'
        token = CancelToken()
        timer = threading.Timer(0.2, token.cancel)
        timer.start()
        try:
            result = run_process_with_metrics(
                self.child_command(pidfile),
                Path(self.tmp.name),
                os.environ.copy(),
                timeout=10,
                app=self.app,
                cancel_token=token,
            )
        finally:
            timer.cancel()
        child_pid = int(pidfile.read_text())
        time.sleep(0.3)
        self.assertTrue(result['aborted'])
        self.assertFalse(process_active(child_pid))


if __name__ == '__main__':
    unittest.main()
