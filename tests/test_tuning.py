import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from llama_tui.app import AppConfig
from llama_tui.benchmark import (
    adaptive_record_from_candidate,
    apply_measured_profile,
    apply_moe_recommendation,
    benchmark_moe_placement_tuning,
    measured_profile_runtime_profile,
    model_for_runtime_profile,
)
from llama_tui.hardware import HardwareProfile
from llama_tui.models import ModelConfig
from llama_tui.runtime_profiles import EngineCapabilities, RuntimeProfile
from llama_tui.tuning import (
    TuningObjective,
    generate_moe_tuning_candidates,
    generate_n_cpu_moe_ladder,
    moe_tuning_eligibility_reason,
    select_measured_tuning_winner,
)


GIB = 1024 ** 3


def moe_model(**overrides) -> ModelConfig:
    values = {
        'id': 'moe',
        'name': 'MoE',
        'path': '/models/moe.gguf',
        'alias': 'moe',
        'port': 18080,
        'runtime': 'llama.cpp',
        'ctx': 32768,
        'ctx_min': 8192,
        'ctx_max': 65536,
        'ngl': 33,
        'parallel': 1,
        'threads': 8,
        'output': 4096,
        'architecture_type': 'moe',
        'expert_count': 64,
        'expert_used_count': 8,
    }
    values.update(overrides)
    return ModelConfig(**values)


def tuning_caps() -> EngineCapabilities:
    return EngineCapabilities(
        supports_cpu_moe=True,
        supports_n_cpu_moe=True,
        supports_override_tensor=True,
        gpu_layers_flag='-ngl',
        cpu_moe_flag='-cmoe',
        n_cpu_moe_flag='-ncmoe',
        override_tensor_flag='-ot',
    )


def tuning_hardware() -> HardwareProfile:
    return HardwareProfile(
        cpu_logical=16,
        cpu_physical=8,
        memory_total=32 * GIB,
        memory_available=20 * GIB,
        gpu_name='Test GPU',
        gpu_memory_total=12 * GIB,
        gpu_memory_free=10 * GIB,
    )


class TuningHelperTests(unittest.TestCase):
    def test_dynamic_ladder_clamps_and_dedupes(self):
        self.assertEqual(generate_n_cpu_moe_ladder(0), [])
        values = generate_n_cpu_moe_ladder(4)
        self.assertEqual(values, sorted(set(values), reverse=True))
        self.assertEqual(values[0], 4)
        self.assertEqual(values[-1], 0)
        self.assertTrue(all(0 <= item <= 4 for item in values))

    def test_winner_selection_ignores_unmeasured_candidates(self):
        objective = TuningObjective(
            purpose='moe_placement',
            target_context=32768,
            min_context=8192,
            target_vram_headroom_bytes=1024 ** 3,
            max_trials=8,
            depth='fast',
        )
        winner = select_measured_tuning_winner(
            [
                {'status': 'planned', 'measured_candidate_name': 'n_cpu_moe_20', 'tokens_per_sec': 999.0},
                {'status': 'ok', 'measured_candidate_name': 'n_cpu_moe_24', 'tokens_per_sec': 41.0, 'ctx': 32768},
            ],
            objective,
        )

        self.assertEqual(winner['measured_candidate_name'], 'n_cpu_moe_24')

    def test_unknown_layer_count_uses_conservative_fallback(self):
        baseline = RuntimeProfile(
            engine_id='llama.cpp',
            name='manual',
            ctx_size=32768,
            gpu_layers=33,
            parallel=2,
            kv_preset='q8_0/q8_0',
            flash_attn='on',
            batch_size=512,
            ubatch_size=128,
            placement_strategy='n_cpu_moe_8',
            n_cpu_moe=8,
            tensor_overrides=('saved=CPU',),
        )

        candidates = generate_moe_tuning_candidates(
            baseline,
            tuning_caps(),
            tuning_hardware(),
            layer_count=0,
            depth='full',
        )
        names = [candidate.name for candidate in candidates]
        baseline_candidate = candidates[0].runtime_profile

        self.assertEqual(names[0], 'baseline_current')
        self.assertNotIn('n_cpu_moe_1', names)
        self.assertFalse(any(name.startswith('n_cpu_moe_') for name in names if name != 'baseline_current'))
        self.assertIn('cpu_moe_all', names)
        self.assertIn('experts_cpu_override', names)
        self.assertEqual(baseline_candidate.ctx_size, baseline.ctx_size)
        self.assertEqual(baseline_candidate.parallel, baseline.parallel)
        self.assertEqual(baseline_candidate.kv_preset, baseline.kv_preset)
        self.assertEqual(baseline_candidate.n_cpu_moe, 8)
        self.assertEqual(baseline_candidate.tensor_overrides, ('saved=CPU',))

    def test_no_moe_capabilities_returns_baseline_only(self):
        baseline = RuntimeProfile(
            engine_id='llama.cpp',
            name='manual',
            ctx_size=32768,
            gpu_layers=33,
            parallel=1,
        )

        candidates = generate_moe_tuning_candidates(
            baseline,
            EngineCapabilities(),
            tuning_hardware(),
            layer_count=32,
            depth='full',
        )

        self.assertEqual([candidate.name for candidate in candidates], ['baseline_current'])

    def test_dense_and_non_llama_family_skip(self):
        dense = moe_model(architecture_type='dense', expert_count=0, expert_used_count=0)

        self.assertIn(
            'not detected as MoE',
            moe_tuning_eligibility_reason(dense, tuning_hardware(), tuning_caps(), 'llama.cpp'),
        )
        self.assertIn(
            'not eligible',
            moe_tuning_eligibility_reason(moe_model(), tuning_hardware(), tuning_caps(), 'vllm'),
        )


class MoeTuningRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.app = AppConfig(self.root / 'models.json')
        self.model = moe_model()
        self.app.add_or_update(self.model)

    def tearDown(self):
        self.tmp.cleanup()

    def fake_runtime_benchmark(self, calls, tps_by_name=None, failures=None):
        tps_by_name = dict(tps_by_name or {})
        failures = dict(failures or {})

        def fake(app, base_model, runtime_profile, objective, progress, cancel_token, completed, total, **kwargs):
            name = runtime_profile.name
            calls.append(name)
            candidate = model_for_runtime_profile(base_model, runtime_profile)
            failure = failures.get(name, '')
            if failure:
                record = adaptive_record_from_candidate(
                    candidate,
                    objective,
                    'start failed',
                    tokens_per_sec=0.0,
                    seconds=1.0,
                    gpu_memory_free=400 * 1024 ** 2,
                    gpu_memory_total=12 * GIB,
                    runtime_profile=name,
                    benchmark_profile='moe_tuning',
                    benchmark_purpose='moe_tuning',
                    failure_category=failure,
                    detail=failure,
                )
                return False, True, [record], [], completed + 1
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=tps_by_name.get(name, 10.0),
                seconds=1.0,
                gpu_memory_free=2 * GIB,
                gpu_memory_total=12 * GIB,
                generated_tokens=128,
                runtime_profile=name,
                benchmark_profile='moe_tuning',
                benchmark_purpose='moe_tuning',
            )
            return True, False, [record], [dict(record)], completed + 1

        return fake

    def run_tuning(self, fake_benchmark, depth='fast', layer_count=32):
        with patch.object(self.app, 'hardware_profile', return_value=tuning_hardware()), \
             patch.object(self.app, 'engine_capabilities', return_value=tuning_caps()), \
             patch('llama_tui.benchmark.benchmark_preflight_cleanup', return_value=(True, 'ok')), \
             patch('llama_tui.benchmark._moe_tuning_layer_count', return_value=layer_count), \
             patch('llama_tui.benchmark.CACHE_DIR', self.root / 'cache'), \
             patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_benchmark):
            return benchmark_moe_placement_tuning(self.app, self.model, depth=depth)

    def test_refinement_candidates_are_benchmarked_before_selection(self):
        calls = []
        ok, msg = self.run_tuning(self.fake_runtime_benchmark(
            calls,
            {
                'baseline_current': 7.0,
                'cpu_moe_all': 8.0,
                'n_cpu_moe_32': 9.0,
                'n_cpu_moe_27': 14.0,
                'n_cpu_moe_24': 12.0,
                'n_cpu_moe_21': 11.0,
                'n_cpu_moe_28': 22.0,
                'n_cpu_moe_26': 13.0,
            },
        ))

        saved = self.app.get_model('moe')
        profile = saved.measured_profiles['moe_placement']
        latest_run = saved.benchmark_runs[0]
        record_names = [row.get('measured_candidate_name') for row in latest_run['records']]

        self.assertTrue(ok, msg)
        self.assertIn('n_cpu_moe_28', calls)
        self.assertIn('n_cpu_moe_26', calls)
        self.assertEqual(profile['measured_candidate_name'], 'n_cpu_moe_28')
        self.assertIn('n_cpu_moe_28', record_names)
        self.assertTrue(profile['effective_server_args'])
        self.assertIn('-ncmoe', profile['effective_server_args'])
        self.assertIn('effective_server_args_preview', profile)
        self.assertEqual(latest_run['kind'], 'moe_tuning')

    def test_early_stop_reason_is_persisted(self):
        calls = []
        ok, msg = self.run_tuning(self.fake_runtime_benchmark(
            calls,
            {
                'baseline_current': 7.0,
                'cpu_moe_all': 8.0,
                'n_cpu_moe_32': 11.0,
            },
            failures={'n_cpu_moe_27': 'CUDA_OOM_WEIGHTS'},
        ))

        saved = self.app.get_model('moe')
        profile = saved.measured_profiles['moe_placement']
        latest_run = saved.benchmark_runs[0]

        self.assertTrue(ok, msg)
        self.assertEqual(profile['measured_candidate_name'], 'n_cpu_moe_32')
        self.assertIn('unsafe', profile['early_stop_reason'])
        self.assertEqual(profile['early_stop_reason'], latest_run['early_stop_reason'])
        self.assertNotIn('n_cpu_moe_24', calls)

    def test_full_tuning_validates_moe_placement_across_context_buckets(self):
        calls = []
        self.model.ctx_max = 131072

        ok, msg = self.run_tuning(self.fake_runtime_benchmark(
            calls,
            {
                'n_cpu_moe_34': 30.0,
                'n_cpu_moe_36': 35.0,
                'fast_chat_n_cpu_moe_34': 55.0,
                'auto_n_cpu_moe_38': 50.0,
                'hermes_ready_cpu_moe_all': 45.0,
                'long_context_cpu_moe_all': 44.0,
            },
        ), depth='full', layer_count=40)

        saved = self.app.get_model('moe')
        profile = saved.measured_profiles['moe_placement']
        placements = profile['profile_moe_placements']
        latest_run = saved.benchmark_runs[0]
        context_rows = [row for row in latest_run['records'] if row.get('context_validation')]

        self.assertTrue(ok, msg)
        self.assertIn('fast_chat', placements)
        self.assertIn('auto', placements)
        self.assertIn('hermes_ready', placements)
        self.assertIn('long_context', placements)
        self.assertEqual(placements['fast_chat']['strategy'], 'n_cpu_moe_34')
        self.assertEqual(placements['auto']['strategy'], 'n_cpu_moe_38')
        self.assertEqual(placements['hermes_ready']['strategy'], 'cpu_moe_all')
        self.assertEqual(placements['long_context']['strategy'], 'cpu_moe_all')
        self.assertIn('auto_n_cpu_moe_38', calls)
        self.assertTrue(any(row.get('context_validation_winner') for row in context_rows))
        self.assertEqual(latest_run['profile_moe_placements']['auto']['n_cpu_moe'], 38)


class ApplyMoeRecommendationTests(unittest.TestCase):
    def test_apply_mutates_only_moe_fields_without_ngl_requirement(self):
        model = moe_model(
            ctx=65536,
            parallel=4,
            ngl=17,
            temp=0.33,
            top_p=0.81,
            top_k=31,
            cache_ram=2048,
            threads=6,
            extra_args=['-ncmoe', '8', '--foo', 'bar'],
            measured_profiles={
                'moe_placement': {
                    'status': 'ok',
                    'measured_candidate_name': 'n_cpu_moe_24',
                    'tuning_run_id': 'run-1',
                    'cpu_moe': False,
                    'n_cpu_moe': 24,
                    'tensor_overrides': [],
                    'ngl': 999,
                    'ngl_required_for_moe_tuning': False,
                    'tokens_per_sec': 41.2,
                }
            },
        )
        before = asdict(model)

        ok, msg = apply_moe_recommendation(model)

        self.assertTrue(ok, msg)
        self.assertEqual(model.moe_placement_strategy, 'measured:moe_tuning:n_cpu_moe_24')
        self.assertEqual(model.n_cpu_moe, 24)
        self.assertEqual(model.ngl, before['ngl'])
        for field in (
            'ctx', 'parallel', 'temp', 'top_p', 'top_k', 'cache_ram',
            'threads', 'output', 'extra_args',
        ):
            self.assertEqual(getattr(model, field), before[field])
        stored = model.measured_profiles['moe_placement']
        self.assertEqual(stored['applied_from'], 'moe_placement')
        self.assertEqual(stored['tuning_run_id'], 'run-1')
        self.assertEqual(stored['measured_candidate_name'], 'n_cpu_moe_24')
        self.assertIn('applied_at', stored)

    def test_apply_updates_ngl_only_when_explicitly_required(self):
        model = moe_model(
            ngl=17,
            measured_profiles={
                'moe_placement': {
                    'status': 'ok',
                    'measured_candidate_name': 'cpu_moe_all',
                    'cpu_moe': True,
                    'n_cpu_moe': 0,
                    'tensor_overrides': [],
                    'ngl': 999,
                    'ngl_required_for_moe_tuning': True,
                }
            },
        )

        ok, _msg = apply_moe_recommendation(model)

        self.assertTrue(ok)
        self.assertTrue(model.cpu_moe)
        self.assertEqual(model.ngl, 999)

    def test_extra_args_survive_apply_and_effective_command_strips_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = moe_model(
                extra_args=['-ncmoe', '8', '--override-tensor', 'old=CPU', '--foo', 'bar'],
                measured_profiles={
                    'moe_placement': {
                        'status': 'ok',
                        'measured_candidate_name': 'n_cpu_moe_24',
                        'cpu_moe': False,
                        'n_cpu_moe': 24,
                        'tensor_overrides': ['new=CPU'],
                        'tokens_per_sec': 40.0,
                    }
                },
            )
            original_extra = list(model.extra_args)

            ok, _msg = apply_moe_recommendation(model)
            self.assertTrue(ok)
            self.assertEqual(model.extra_args, original_extra)

            with patch.object(app, 'engine_capabilities', return_value=tuning_caps()):
                cmd = app.build_command(model)

            self.assertEqual(model.extra_args, original_extra)
            self.assertIn('--foo', cmd)
            self.assertIn('bar', cmd)
            self.assertIn('-ncmoe', cmd)
            self.assertEqual(cmd[cmd.index('-ncmoe') + 1], '24')
            self.assertIn('-ot', cmd)
            self.assertIn('new=CPU', cmd)
            self.assertNotIn('old=CPU', cmd)
            self.assertEqual(cmd.count('-ncmoe'), 1)

    def test_profile_level_moe_placement_takes_precedence_over_flat_fields(self):
        model = moe_model(
            measured_profiles={
                'long_context': {
                    'status': 'ok',
                    'ctx': 65536,
                    'ctx_per_slot': 65536,
                    'parallel': 1,
                    'threads': 8,
                    'ngl': 33,
                    'tokens_per_sec': 20.0,
                    'placement_strategy': 'n_cpu_moe_30',
                    'n_cpu_moe': 30,
                    'cpu_moe': False,
                    'tensor_overrides': [],
                    'moe_placement': {
                        'strategy': 'n_cpu_moe_38',
                        'n_cpu_moe': 38,
                        'cpu_moe': False,
                        'tensor_overrides': [],
                    },
                }
            },
        )

        runtime_profile = measured_profile_runtime_profile(model, 'long_context')
        ok, msg = apply_measured_profile(model, 'long_context')

        self.assertIsNotNone(runtime_profile)
        self.assertEqual(runtime_profile.n_cpu_moe, 38)
        self.assertTrue(ok, msg)
        self.assertEqual(model.n_cpu_moe, 38)
        self.assertEqual(model.moe_placement_strategy, 'n_cpu_moe_38')

    def test_flat_moe_fields_remain_backward_compatible_when_profile_level_missing(self):
        model = moe_model(
            measured_profiles={
                'fast_chat': {
                    'status': 'ok',
                    'ctx': 8192,
                    'ctx_per_slot': 8192,
                    'parallel': 1,
                    'threads': 8,
                    'ngl': 33,
                    'tokens_per_sec': 40.0,
                    'placement_strategy': 'n_cpu_moe_30',
                    'n_cpu_moe': 30,
                    'cpu_moe': False,
                    'tensor_overrides': [],
                }
            },
        )

        runtime_profile = measured_profile_runtime_profile(model, 'fast_chat')
        ok, msg = apply_measured_profile(model, 'fast_chat')

        self.assertIsNotNone(runtime_profile)
        self.assertEqual(runtime_profile.n_cpu_moe, 30)
        self.assertTrue(ok, msg)
        self.assertEqual(model.n_cpu_moe, 30)
        self.assertEqual(model.moe_placement_strategy, 'n_cpu_moe_30')


if __name__ == '__main__':
    unittest.main()
