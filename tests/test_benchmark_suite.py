import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from llama_tui.benchmark import (
    adaptive_record_from_candidate,
    apply_full_suite_profile_recommendation,
    apply_full_suite_recommendations,
    active_engine_runtime_profiles,
    benchmark_full_suite,
    build_runtime_overlay_from_moe_recommendation,
    full_suite_recommended_profile_key,
    model_for_runtime_profile,
    runtime_record_context,
    runtime_profile_moe_placement_mode,
    runtime_profile_with_overlay,
    suite_run_recommended_profile_key,
)
from llama_tui.hardware import HardwareProfile
from llama_tui.models import ModelConfig
from llama_tui.runtime_profiles import EngineCapabilities, RuntimeProfile, make_runtime_profile, runtime_profile_extra_args


GIB = 1024 ** 3


def suite_model(**overrides) -> ModelConfig:
    values = {
        'id': 'suite-moe',
        'name': 'Suite MoE',
        'path': '/models/suite-moe.gguf',
        'alias': 'suite-moe',
        'port': 18080,
        'runtime': 'llama.cpp',
        'ctx': 32768,
        'ctx_min': 8192,
        'ctx_max': 65536,
        'parallel': 1,
        'ngl': 33,
        'threads': 8,
        'output': 2048,
        'architecture_type': 'moe',
        'expert_count': 64,
        'expert_used_count': 8,
        'extra_args': ['--n-cpu-moe', '99', '--custom-flag'],
    }
    values.update(overrides)
    return ModelConfig(**values)


class SuiteFakeApp:
    def __init__(self, model: ModelConfig, engine: str = 'llama.cpp', supports_fit: bool = False):
        self.models = [ModelConfig(**asdict(model))]
        self.engine = engine
        self.supports_fit = supports_fit
        self.saved = []
        self.logs = []
        self.hardware = HardwareProfile(
            cpu_logical=16,
            cpu_physical=8,
            memory_total=32 * GIB,
            memory_available=24 * GIB,
            gpu_name='Suite GPU',
            gpu_memory_total=12 * GIB,
            gpu_memory_free=10 * GIB,
        )
        self.opencode = type('OpenCode', (), {'path': ''})()
        self.continue_settings = type('Continue', (), {'path': ''})()
        self.runtime_profile = make_runtime_profile(engine, 'llama-server')

    def get_model(self, model_id):
        return next((model for model in self.models if model.id == model_id), None)

    def add_or_update(self, model):
        stored = ModelConfig(**asdict(model))
        for idx, existing in enumerate(self.models):
            if existing.id == stored.id:
                self.models[idx] = stored
                self.saved.append(stored)
                return
        self.models.append(stored)
        self.saved.append(stored)

    def active_engine_key_for_model(self, _model):
        return self.engine

    def runtime_profile_from_model(self, model, ctx_value, parallel_value, ngl_value, runtime_profile=None):
        if runtime_profile is not None:
            return runtime_profile
        return RuntimeProfile(
            engine_id=self.engine,
            name='manual',
            ctx_size=max(1, int(ctx_value or 1)),
            gpu_layers=int(ngl_value or 0),
            parallel=max(1, int(parallel_value or 1)),
            placement_strategy=getattr(model, 'moe_placement_strategy', '') or '',
            cpu_moe=bool(getattr(model, 'cpu_moe', False)),
            n_cpu_moe=max(0, int(getattr(model, 'n_cpu_moe', 0) or 0)),
            tensor_overrides=tuple(getattr(model, 'tensor_overrides', []) or []),
        )

    def engine_capabilities(self):
        return EngineCapabilities(
            supports_cpu_moe=True,
            supports_n_cpu_moe=True,
            supports_override_tensor=True,
            supports_ctk_ctv=True,
            supports_fit=self.supports_fit,
            supports_fit_ctx=self.supports_fit,
            supports_no_warmup=self.supports_fit,
            cpu_moe_flag='-cmoe',
            n_cpu_moe_flag='-ncmoe',
            override_tensor_flag='-ot',
            gpu_layers_flag='-ngl',
            supported_kv_modes=('q8_0', 'turbo4', 'turbo3', 'turbo2'),
        )

    def build_command(
        self,
        model,
        ctx_override=None,
        parallel_override=None,
        ngl_override=None,
        runtime_profile=None,
        benchmark_profile=None,
    ):
        ctx_value = int(ctx_override if ctx_override is not None else getattr(model, 'ctx', 0) or 0)
        parallel_value = int(parallel_override if parallel_override is not None else getattr(model, 'parallel', 1) or 1)
        ngl_value = int(ngl_override if ngl_override is not None else getattr(model, 'ngl', 0) or 0)
        profile = self.runtime_profile_from_model(
            model,
            ctx_value,
            parallel_value,
            ngl_value,
            runtime_profile=runtime_profile,
        )
        cmd = ['llama-server', '-m', str(getattr(model, 'path', '') or '')]
        if profile.gpu_layers is not None:
            cmd += ['-ngl', str(int(profile.gpu_layers))]
        cmd += runtime_profile_extra_args(
            self.runtime_profile,
            profile,
            self.engine_capabilities(),
            existing_args=list(getattr(model, 'extra_args', []) or []),
        )
        return cmd

    def hardware_profile(self, refresh=False):
        return self.hardware

    def health(self, _model):
        return 'STOPPED', ''

    def get_pid(self, _model, discover=True, managed_only=False):
        return None

    def stop(self, _model, managed_only=False):
        return True, 'stopped'

    def active_engine_model_compatibility(self, _model):
        return True, ''

    def model_fingerprint(self, model):
        return f'fp-{model.id}'

    def sync_generated_configs(self, reason):
        return f'synced {reason}'

    def append_log(self, model_id, text):
        self.logs.append((model_id, text))


def append_run(model: ModelConfig, kind: str, status: str, record=None, winners=None):
    run = {
        'id': f'{kind}-run',
        'kind': kind,
        'status': status,
        'records': [dict(record)] if record else [],
        'winners': {key: dict(value) for key, value in (winners or {}).items()},
        'summary': f'{kind} {status}',
    }
    model.benchmark_runs = [run] + list(getattr(model, 'benchmark_runs', []) or [])


def suite_stage_runners(order, seen, fail_moe=False):
    def moe_runner(app, model, progress=None, cancel_token=None, depth='full'):
        order.append('moe')
        saved = ModelConfig(**asdict(model))
        if fail_moe:
            append_run(saved, 'moe_tuning', 'failed')
            app.add_or_update(saved)
            return False, 'MoE placement failed with CUDA_OOM_WEIGHTS'
        profile = {
            'status': 'ok',
            'kind': 'moe_tuning',
            'measured_candidate_name': 'n_cpu_moe_30',
            'placement_strategy': 'measured:moe_tuning:n_cpu_moe_30',
            'cpu_moe': False,
            'n_cpu_moe': 30,
            'tensor_overrides': [],
            'tokens_per_sec': 25.87,
            'tuning_run_id': 'moe-run',
            'ngl_required_for_moe_tuning': False,
        }
        saved.measured_profiles = dict(getattr(saved, 'measured_profiles', {}) or {})
        saved.measured_profiles['moe_placement'] = profile
        append_run(saved, 'moe_tuning', 'done', winners={'moe_placement': profile})
        app.add_or_update(saved)
        return True, 'MoE placement winner n_cpu_moe_30'

    def smart_runner(app, model, progress=None, cancel_token=None):
        order.append('smart')
        seen['smart_n_cpu_moe'] = model.n_cpu_moe
        saved = ModelConfig(**asdict(model))
        saved.ctx = 12345
        saved.parallel = 4
        saved.n_cpu_moe = 30
        record = adaptive_record_from_candidate(
            model,
            'opencode_ready',
            'ok',
            tokens_per_sec=22.0,
            seconds=1.0,
            command='llama-server -ncmoe 30',
        )
        record['effective_server_args'] = ['llama-server', '-ncmoe', '30']
        record['effective_server_command'] = 'llama-server -ncmoe 30'
        auto = dict(record, status='ok', ctx=model.ctx, ctx_per_slot=model.ctx, parallel=model.parallel)
        opencode = dict(auto, objective='opencode_ready')
        saved.measured_profiles = dict(getattr(saved, 'measured_profiles', {}) or {})
        saved.measured_profiles.update({'auto': auto, 'fast_chat': auto, 'opencode_ready': opencode})
        append_run(saved, 'server', 'done', record=record, winners={'auto': auto, 'opencode_ready': opencode})
        app.add_or_update(saved)
        return True, 'smart benchmark saved opencode_ready'

    def hermes_runner(app, model, progress=None, cancel_token=None):
        order.append('hermes')
        seen['hermes_n_cpu_moe'] = model.n_cpu_moe
        saved = ModelConfig(**asdict(model))
        saved.last_hermes_benchmark_score = 1.0
        record = {'status': 'ok', 'profile_used': full_suite_recommended_profile_key(model)}
        append_run(saved, 'hermes', 'done', record=record)
        app.add_or_update(saved)
        return True, 'Hermes passed'

    def opencode_runner(app, model, progress=None, cancel_token=None):
        order.append('opencode')
        seen['opencode_n_cpu_moe'] = model.n_cpu_moe
        saved = ModelConfig(**asdict(model))
        saved.last_opencode_benchmark_score = 1.0
        record = {'status': 'ok', 'profile_used': full_suite_recommended_profile_key(model)}
        append_run(saved, 'opencode', 'done', record=record)
        app.add_or_update(saved)
        return True, 'OpenCode passed'

    return moe_runner, smart_runner, hermes_runner, opencode_runner


class FullSuiteBackendTests(unittest.TestCase):
    def run_suite_with_measured_moe_profile(self, moe_profile, initial_model=None):
        model = initial_model or suite_model()
        app = SuiteFakeApp(model)
        order = []
        seen = {'progress': []}

        def moe_runner(app, model, progress=None, cancel_token=None, depth='full'):
            order.append('moe')
            saved = ModelConfig(**asdict(model))
            saved.measured_profiles = dict(getattr(saved, 'measured_profiles', {}) or {})
            saved.measured_profiles['moe_placement'] = dict(moe_profile)
            append_run(saved, 'moe_tuning', 'done', winners={'moe_placement': moe_profile})
            app.add_or_update(saved)
            return True, f'MoE placement winner {moe_profile["measured_candidate_name"]}'

        def smart_runner(app, model, progress=None, cancel_token=None):
            order.append('smart')
            runtime_profiles = active_engine_runtime_profiles(
                app,
                model,
                app.hardware_profile(refresh=True),
                depth='fast',
            )
            self.assertTrue(runtime_profiles)
            runtime_profile = runtime_profiles[0]
            candidate = model_for_runtime_profile(model, runtime_profile)
            command = app.build_command(candidate, runtime_profile=runtime_profile)
            seen['smart_runtime_profile'] = runtime_profile
            seen['smart_command'] = command
            saved = ModelConfig(**asdict(model))
            record = adaptive_record_from_candidate(
                candidate,
                'opencode_ready',
                'ok',
                tokens_per_sec=22.0,
                seconds=1.0,
                runtime_profile=runtime_profile.name,
                command=' '.join(command),
                effective_server_args=command,
                effective_server_command=' '.join(command),
            )
            auto = dict(record, status='ok', ctx=candidate.ctx, ctx_per_slot=candidate.ctx, parallel=candidate.parallel)
            opencode = dict(auto, objective='opencode_ready')
            saved.measured_profiles = dict(getattr(saved, 'measured_profiles', {}) or {})
            saved.measured_profiles.update({'auto': auto, 'fast_chat': auto, 'opencode_ready': opencode})
            append_run(saved, 'server', 'done', record=record, winners={'auto': auto, 'opencode_ready': opencode})
            app.add_or_update(saved)
            return True, 'smart benchmark saved opencode_ready'

        runners = suite_stage_runners(order, seen)
        with tempfile.TemporaryDirectory() as tmp, patch('llama_tui.benchmark.CACHE_DIR', Path(tmp)):
            ok, msg = benchmark_full_suite(
                app,
                model,
                progress=seen['progress'].append,
                moe_runner=moe_runner,
                smart_runner=smart_runner,
                hermes_runner=runners[2],
                opencode_runner=runners[3],
            )
        return ok, msg, app, seen

    def test_full_suite_orders_stages_uses_overlay_and_restores_config(self):
        model = suite_model()
        app = SuiteFakeApp(model)
        order = []
        seen = {}
        runners = suite_stage_runners(order, seen)

        with tempfile.TemporaryDirectory() as tmp, patch('llama_tui.benchmark.CACHE_DIR', Path(tmp)):
            ok, msg = benchmark_full_suite(
                app,
                model,
                moe_runner=runners[0],
                smart_runner=runners[1],
                hermes_runner=runners[2],
                opencode_runner=runners[3],
            )

        saved = app.get_model(model.id)
        suite_run = next(run for run in saved.benchmark_runs if run['kind'] == 'full_suite')
        server_run = next(run for run in saved.benchmark_runs if run['kind'] == 'server')

        self.assertTrue(ok, msg)
        self.assertEqual(order, ['moe', 'smart', 'hermes', 'opencode'])
        self.assertEqual(seen['smart_n_cpu_moe'], 30)
        self.assertEqual(seen['hermes_n_cpu_moe'], 30)
        self.assertEqual(seen['opencode_n_cpu_moe'], 30)
        self.assertEqual(saved.ctx, model.ctx)
        self.assertEqual(saved.parallel, model.parallel)
        self.assertEqual(saved.n_cpu_moe, 0)
        self.assertEqual(saved.extra_args, model.extra_args)
        self.assertIn('moe_placement', saved.measured_profiles)
        self.assertIn('opencode_ready', saved.measured_profiles)
        self.assertEqual(suite_run['status'], 'done')
        self.assertEqual(suite_run['recommendations']['moe_placement'], 'n_cpu_moe_30')
        self.assertEqual(suite_run['recommendations']['default_profile'], 'opencode_ready')
        self.assertTrue(suite_run['uses_suite_overlay'])
        self.assertEqual(server_run['suite_run_id'], suite_run['suite_run_id'])
        self.assertTrue(server_run['uses_suite_overlay'])
        self.assertEqual(server_run['records'][0]['moe_overlay_source'], 'current_suite')
        self.assertEqual(server_run['records'][0]['moe_overlay_flags'], '-ncmoe 30')
        self.assertEqual(server_run['records'][0]['effective_server_args'], ['llama-server', '-ncmoe', '30'])

    def test_full_suite_smart_candidates_include_ncmoe_overlay(self):
        ok, msg, app, seen = self.run_suite_with_measured_moe_profile({
            'status': 'ok',
            'kind': 'moe_tuning',
            'measured_candidate_name': 'n_cpu_moe_40',
            'placement_strategy': 'measured:moe_tuning:n_cpu_moe_40',
            'cpu_moe': False,
            'n_cpu_moe': 40,
            'tensor_overrides': [],
            'tokens_per_sec': 25.0,
            'tuning_run_id': 'moe-run',
            'ngl_required_for_moe_tuning': False,
        })

        saved = app.get_model('suite-moe')

        self.assertTrue(ok, msg)
        self.assertEqual(saved.n_cpu_moe, 0)
        self.assertEqual(seen['smart_runtime_profile'].n_cpu_moe, 40)
        self.assertIn('+moe_ncpu40', seen['smart_runtime_profile'].name)
        self.assertIn('-ncmoe', seen['smart_command'])
        self.assertEqual(seen['smart_command'][seen['smart_command'].index('-ncmoe') + 1], '40')
        progress_text = '\n'.join(str(item) for item in seen['progress'])
        self.assertIn('Full Suite MoE winner: source=current_suite winner=n_cpu_moe_40', progress_text)
        self.assertIn('Full Suite MoE overlay: source=current_suite winner=n_cpu_moe_40', progress_text)
        self.assertIn('flags="-ncmoe 40"', progress_text)

    def test_full_suite_smart_candidates_include_cpu_moe_overlay(self):
        ok, msg, app, seen = self.run_suite_with_measured_moe_profile({
            'status': 'ok',
            'kind': 'moe_tuning',
            'measured_candidate_name': 'cpu_moe_all',
            'placement_strategy': 'measured:moe_tuning:cpu_moe_all',
            'cpu_moe': True,
            'n_cpu_moe': 0,
            'tensor_overrides': [],
            'tokens_per_sec': 18.0,
            'tuning_run_id': 'moe-run',
            'ngl_required_for_moe_tuning': False,
        })

        saved = app.get_model('suite-moe')

        self.assertTrue(ok, msg)
        self.assertFalse(saved.cpu_moe)
        self.assertTrue(seen['smart_runtime_profile'].cpu_moe)
        self.assertIn('+moe_cpu', seen['smart_runtime_profile'].name)
        self.assertIn('-cmoe', seen['smart_command'])
        self.assertNotIn('-ncmoe', seen['smart_command'])

    def test_current_suite_moe_winner_takes_precedence_over_stored_recommendation(self):
        initial = suite_model(
            measured_profiles={
                'moe_placement': {
                    'status': 'ok',
                    'kind': 'moe_tuning',
                    'measured_candidate_name': 'n_cpu_moe_12',
                    'placement_strategy': 'measured:moe_tuning:n_cpu_moe_12',
                    'cpu_moe': False,
                    'n_cpu_moe': 12,
                    'tensor_overrides': [],
                }
            }
        )

        ok, msg, _app, seen = self.run_suite_with_measured_moe_profile(
            {
                'status': 'ok',
                'kind': 'moe_tuning',
                'measured_candidate_name': 'cpu_moe_all',
                'placement_strategy': 'measured:moe_tuning:cpu_moe_all',
                'cpu_moe': True,
                'n_cpu_moe': 0,
                'tensor_overrides': [],
                'tokens_per_sec': 18.0,
                'tuning_run_id': 'current-suite-moe',
                'ngl_required_for_moe_tuning': False,
            },
            initial_model=initial,
        )

        self.assertTrue(ok, msg)
        self.assertTrue(seen['smart_runtime_profile'].cpu_moe)
        self.assertEqual(seen['smart_runtime_profile'].n_cpu_moe, 0)
        self.assertIn('-cmoe', seen['smart_command'])
        self.assertNotIn('-ncmoe', seen['smart_command'])
        progress_text = '\n'.join(str(item) for item in seen['progress'])
        self.assertIn('source=current_suite winner=cpu_moe_all', progress_text)
        self.assertNotIn('winner=n_cpu_moe_12', progress_text)

    def test_standalone_smart_does_not_use_unapplied_moe_recommendation(self):
        model = suite_model(
            measured_profiles={
                'moe_placement': {
                    'status': 'ok',
                    'measured_candidate_name': 'n_cpu_moe_37',
                    'placement_strategy': 'measured:moe_tuning:n_cpu_moe_37',
                    'cpu_moe': False,
                    'n_cpu_moe': 37,
                    'tensor_overrides': [],
                }
            },
            cpu_moe=False,
            n_cpu_moe=0,
            moe_placement_strategy='',
            tensor_overrides=[],
        )
        app = SuiteFakeApp(model)

        with patch('llama_tui.moe_placement.gguf_layer_count', return_value=0):
            runtime_profiles = active_engine_runtime_profiles(
                app,
                model,
                app.hardware_profile(refresh=True),
                depth='fast',
            )

        self.assertTrue(runtime_profiles)
        self.assertEqual(runtime_profiles[0].n_cpu_moe, 0)
        self.assertNotIn('+moe_ncpu37', runtime_profiles[0].name)
        self.assertFalse(any(profile.n_cpu_moe == 37 for profile in runtime_profiles))
        command = app.build_command(
            model_for_runtime_profile(model, runtime_profiles[0]),
            runtime_profile=runtime_profiles[0],
        )
        self.assertNotIn('-ncmoe', command)
        self.assertNotIn('--n-cpu-moe', command)

    def test_applied_moe_recommendation_is_used_by_normal_launch_and_smart_profiles(self):
        model = suite_model(
            n_cpu_moe=30,
            moe_placement_strategy='measured:moe_tuning:n_cpu_moe_30',
            tensor_overrides=[],
        )
        app = SuiteFakeApp(model)

        launch_command = app.build_command(model)
        runtime_profiles = active_engine_runtime_profiles(
            app,
            model,
            app.hardware_profile(refresh=True),
            depth='fast',
        )
        smart_command = app.build_command(
            model_for_runtime_profile(model, runtime_profiles[0]),
            runtime_profile=runtime_profiles[0],
        )

        self.assertIn('-ncmoe', launch_command)
        self.assertEqual(launch_command[launch_command.index('-ncmoe') + 1], '30')
        self.assertEqual(runtime_profiles[0].n_cpu_moe, 30)
        self.assertIn('+moe_ncpu30', runtime_profiles[0].name)
        self.assertIn('-ncmoe', smart_command)
        self.assertEqual(smart_command[smart_command.index('-ncmoe') + 1], '30')

    def test_profile_level_moe_placement_overrides_global_for_matching_context(self):
        model = suite_model(
            ctx_max=65536,
            n_cpu_moe=30,
            moe_placement_strategy='measured:moe_tuning:n_cpu_moe_30',
            measured_profiles={
                'moe_placement': {
                    'status': 'ok',
                    'measured_candidate_name': 'n_cpu_moe_30',
                    'cpu_moe': False,
                    'n_cpu_moe': 30,
                    'tensor_overrides': [],
                    'profile_moe_placements': {
                        'fast_chat': {
                            'strategy': 'n_cpu_moe_30',
                            'cpu_moe': False,
                            'n_cpu_moe': 30,
                            'tensor_overrides': [],
                        },
                        'auto': {
                            'strategy': 'n_cpu_moe_38',
                            'cpu_moe': False,
                            'n_cpu_moe': 38,
                            'tensor_overrides': [],
                        },
                    },
                }
            },
        )
        app = SuiteFakeApp(model)

        runtime_profiles = active_engine_runtime_profiles(
            app,
            model,
            app.hardware_profile(refresh=True),
            depth='fast',
        )
        auto_context = next(item for item in runtime_profiles if int(item.ctx_size or 0) > 32768)
        command = app.build_command(
            model_for_runtime_profile(model, auto_context),
            runtime_profile=auto_context,
        )

        self.assertEqual(auto_context.n_cpu_moe, 38)
        self.assertIn('+moe_ncpu38', auto_context.name)
        self.assertIn('-ncmoe', command)
        self.assertEqual(command[command.index('-ncmoe') + 1], '38')

    def test_fit_assisted_moe_candidates_are_labeled_separately_after_locked_probe(self):
        model = suite_model(
            runtime='llama.cpp',
            n_cpu_moe=40,
            moe_placement_strategy='measured:moe_tuning:n_cpu_moe_40',
            turboquant_status='native',
            turboquant_head_dim=128,
            turboquant_key_dim=128,
            turboquant_value_dim=128,
        )
        app = SuiteFakeApp(model, engine='turboquant', supports_fit=True)

        runtime_profiles = active_engine_runtime_profiles(
            app,
            model,
            app.hardware_profile(refresh=True),
            depth='fast',
        )

        locked = next(item for item in runtime_profiles if item.name.startswith('moe_locked_probe'))
        fit_assisted = next(item for item in runtime_profiles if item.fit and '+fit_assisted' in item.name)
        locked_command = app.build_command(model_for_runtime_profile(model, locked), runtime_profile=locked)
        fit_command = app.build_command(model_for_runtime_profile(model, fit_assisted), runtime_profile=fit_assisted)
        locked_context = runtime_record_context(
            app,
            model_for_runtime_profile(model, locked),
            runtime_profile=locked,
        )
        fit_context = runtime_record_context(
            app,
            model_for_runtime_profile(model, fit_assisted),
            runtime_profile=fit_assisted,
        )

        self.assertLess(runtime_profiles.index(locked), runtime_profiles.index(fit_assisted))
        self.assertFalse(locked.fit)
        self.assertTrue(fit_assisted.fit)
        self.assertIn('+moe_ncpu40', locked.name)
        self.assertIn('+moe_ncpu40+fit_assisted', fit_assisted.name)
        self.assertEqual(runtime_profile_moe_placement_mode(locked), 'locked')
        self.assertEqual(runtime_profile_moe_placement_mode(fit_assisted), 'fit_assisted')
        self.assertIn('-ncmoe', locked_command)
        self.assertIn('-ncmoe', fit_command)
        self.assertIn('-fit', fit_command)
        self.assertEqual(locked_context['moe_placement_mode'], 'locked')
        self.assertFalse(locked_context['fit_assisted_moe_placement'])
        self.assertEqual(fit_context['moe_placement_mode'], 'fit_assisted')
        self.assertTrue(fit_context['fit_assisted_moe_placement'])

    def test_dense_model_skips_moe_stage(self):
        model = suite_model(architecture_type='dense', expert_count=0, expert_used_count=0)
        app = SuiteFakeApp(model)
        order = []
        seen = {}

        def forbidden_moe(*_args, **_kwargs):
            raise AssertionError('dense model should not run MoE tuning')

        runners = suite_stage_runners(order, seen)
        with tempfile.TemporaryDirectory() as tmp, patch('llama_tui.benchmark.CACHE_DIR', Path(tmp)):
            ok, msg = benchmark_full_suite(
                app,
                model,
                moe_runner=forbidden_moe,
                smart_runner=runners[1],
                hermes_runner=runners[2],
                opencode_runner=runners[3],
            )

        saved = app.get_model(model.id)
        suite_run = next(run for run in saved.benchmark_runs if run['kind'] == 'full_suite')

        self.assertTrue(ok, msg)
        self.assertEqual(order, ['smart', 'hermes', 'opencode'])
        self.assertEqual(suite_run['stages']['moe_placement']['status'], 'skipped')
        self.assertIn('not detected as MoE', suite_run['stages']['moe_placement']['detail'])

    def test_non_llama_engine_skips_moe_stage(self):
        model = suite_model(runtime='vllm')
        app = SuiteFakeApp(model, engine='vllm')
        order = []
        seen = {}

        def forbidden_moe(*_args, **_kwargs):
            raise AssertionError('vLLM should not run MoE tuning')

        runners = suite_stage_runners(order, seen)
        with tempfile.TemporaryDirectory() as tmp, patch('llama_tui.benchmark.CACHE_DIR', Path(tmp)):
            ok, msg = benchmark_full_suite(
                app,
                model,
                moe_runner=forbidden_moe,
                smart_runner=runners[1],
                hermes_runner=runners[2],
                opencode_runner=runners[3],
            )

        saved = app.get_model(model.id)
        suite_run = next(run for run in saved.benchmark_runs if run['kind'] == 'full_suite')

        self.assertTrue(ok, msg)
        self.assertEqual(order, ['smart', 'hermes', 'opencode'])
        self.assertEqual(suite_run['stages']['moe_placement']['status'], 'skipped')
        self.assertIn('not eligible', suite_run['stages']['moe_placement']['detail'])

    def test_failed_moe_stage_continues_without_overlay_warning(self):
        model = suite_model()
        app = SuiteFakeApp(model)
        order = []
        seen = {}
        runners = suite_stage_runners(order, seen, fail_moe=True)

        with tempfile.TemporaryDirectory() as tmp, patch('llama_tui.benchmark.CACHE_DIR', Path(tmp)):
            ok, msg = benchmark_full_suite(
                app,
                model,
                moe_runner=runners[0],
                smart_runner=runners[1],
                hermes_runner=runners[2],
                opencode_runner=runners[3],
            )

        saved = app.get_model(model.id)
        suite_run = next(run for run in saved.benchmark_runs if run['kind'] == 'full_suite')

        self.assertTrue(ok, msg)
        self.assertEqual(order, ['moe', 'smart', 'hermes', 'opencode'])
        self.assertEqual(seen['smart_n_cpu_moe'], 0)
        self.assertEqual(suite_run['stages']['moe_placement']['status'], 'failed')
        self.assertTrue(any('MoE placement failed' in warning for warning in suite_run['warnings']))

    def test_overlay_helpers_apply_only_moe_runtime_fields(self):
        model = suite_model(
            measured_profiles={
                'moe_placement': {
                    'status': 'ok',
                    'measured_candidate_name': 'n_cpu_moe_24',
                    'n_cpu_moe': 24,
                    'cpu_moe': False,
                    'tensor_overrides': ['blk.*.ffn=CPU'],
                    'ngl_required_for_moe_tuning': True,
                    'ngl': 41,
                }
            }
        )
        overlay = build_runtime_overlay_from_moe_recommendation(model)
        runtime_profile = RuntimeProfile(
            engine_id='llama.cpp',
            name='manual',
            ctx_size=32768,
            gpu_layers=33,
            parallel=2,
            batch_size=512,
        )

        overlaid = runtime_profile_with_overlay(runtime_profile, overlay)

        self.assertEqual(overlaid.ctx_size, runtime_profile.ctx_size)
        self.assertEqual(overlaid.parallel, runtime_profile.parallel)
        self.assertEqual(overlaid.batch_size, runtime_profile.batch_size)
        self.assertEqual(overlaid.gpu_layers, 41)
        self.assertEqual(overlaid.n_cpu_moe, 24)
        self.assertEqual(overlaid.tensor_overrides, ('blk.*.ffn=CPU',))

    def test_apply_profile_only_preserves_moe_fields(self):
        model = suite_model(
            n_cpu_moe=0,
            moe_placement_strategy='',
            measured_profiles={
                'moe_placement': {
                    'status': 'ok',
                    'measured_candidate_name': 'n_cpu_moe_30',
                    'n_cpu_moe': 30,
                    'tensor_overrides': [],
                },
                'opencode_ready': {
                    'status': 'ok',
                    'ctx': 16384,
                    'ctx_per_slot': 16384,
                    'parallel': 1,
                    'threads': 4,
                    'ngl': 22,
                    'output': 1024,
                    'tokens_per_sec': 20.0,
                    'n_cpu_moe': 30,
                    'placement_strategy': 'measured:moe_tuning:n_cpu_moe_30',
                    'tensor_overrides': [],
                },
            },
        )
        run = {'recommendations': {'default_profile': 'opencode_ready'}}

        ok, msg = apply_full_suite_profile_recommendation(model, run)

        self.assertTrue(ok, msg)
        self.assertEqual(model.ctx, 16384)
        self.assertEqual(model.parallel, 1)
        self.assertEqual(model.n_cpu_moe, 0)
        self.assertEqual(model.moe_placement_strategy, '')
        self.assertEqual(model.tensor_overrides, [])
        self.assertEqual(suite_run_recommended_profile_key(model, run), 'opencode_ready')

    def test_apply_all_applies_profile_and_moe_then_syncs(self):
        model = suite_model(
            measured_profiles={
                'moe_placement': {
                    'status': 'ok',
                    'measured_candidate_name': 'n_cpu_moe_30',
                    'cpu_moe': False,
                    'n_cpu_moe': 30,
                    'tensor_overrides': [],
                    'tokens_per_sec': 25.0,
                },
                'opencode_ready': {
                    'status': 'ok',
                    'ctx': 16384,
                    'ctx_per_slot': 16384,
                    'parallel': 1,
                    'threads': 4,
                    'ngl': 22,
                    'output': 1024,
                    'tokens_per_sec': 20.0,
                    'n_cpu_moe': 30,
                    'tensor_overrides': [],
                },
            },
        )
        app = SuiteFakeApp(model)
        run = {'recommendations': {'default_profile': 'opencode_ready', 'moe_placement': 'n_cpu_moe_30'}}

        ok, msg = apply_full_suite_recommendations(app, model, run)

        self.assertTrue(ok, msg)
        self.assertIn('Applied Full Suite recommendations', msg)
        self.assertEqual(model.ctx, 16384)
        self.assertEqual(model.n_cpu_moe, 30)
        self.assertEqual(model.moe_placement_strategy, 'measured:moe_tuning:n_cpu_moe_30')
        self.assertTrue(app.saved)
        self.assertTrue(any('Applied full suite recommendations' in text for _model_id, text in app.logs))


if __name__ == '__main__':
    unittest.main()
