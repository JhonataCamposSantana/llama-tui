import inspect
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from llama_tui.benchmark import (
    active_engine_runtime_profiles,
    benchmark_best_optimization,
    benchmark_fast_profiles,
    benchmark_full_suite,
    parse_mtp_acceptance_metrics,
)
from llama_tui.benchmark_strategies import select_benchmark_strategy
from llama_tui.hardware import HardwareProfile
from llama_tui.model_compat import detect_model_runtime_features
from llama_tui.models import ModelConfig
from llama_tui.runtime_profiles import default_engine_capabilities, make_runtime_profile
from llama_tui.app import AppConfig


class BenchmarkStrategyTests(unittest.TestCase):
    def test_feature_detection_feeds_strategy_selection(self):
        hardware = HardwareProfile(gpu_memory_total=8 * 1024 ** 3, gpu_memory_free=6 * 1024 ** 3)
        model = ModelConfig(
            id='tq3',
            name='Generic MoE TQ3',
            path='/models/generic-moe.TQ3_4S.gguf',
            alias='tq3',
            architecture_type='moe',
            expert_count=128,
            tq3_status='native',
            tq3_weight_format='TQ3_4S',
            ctx_max=131072,
        )

        features = detect_model_runtime_features(model, hardware_profile=hardware, model_size_bytes=18 * 1024 ** 3)

        self.assertIn('moe', features)
        self.assertIn('tq3_native', features)
        self.assertIn('long_context', features)
        self.assertIn('small_vram_risk', features)

    def test_strategy_selection_does_not_branch_on_model_family_names(self):
        source = inspect.getsource(select_benchmark_strategy).lower()

        for token in ('qwen', 'gemma', 'deepseek', 'mistral', 'mixtral'):
            self.assertNotIn(token, source)

    def test_tq3_strategy_is_small_first_pass(self):
        model = ModelConfig(
            id='tq3',
            name='TQ3 Native',
            path='/models/model.TQ3_4S.gguf',
            alias='tq3',
            tq3_status='native',
            tq3_weight_format='TQ3_4S',
        )
        strategy = select_benchmark_strategy('tq3', model, depth='fast')

        self.assertEqual(strategy.id, 'tq3_native_probe')
        self.assertLessEqual(strategy.max_candidates, 6)
        self.assertEqual(strategy.retry_policy, 'tq3_terminal_timeout')
        self.assertIn('pp_baseline', [phase.id for phase in strategy.phases])
        self.assertIn('tg_baseline', [phase.id for phase in strategy.phases])

    def test_tq3_fast_runtime_profiles_are_capped_to_strategy(self):
        model = ModelConfig(
            id='tq3-moe',
            name='TQ3 MoE Native',
            path='/models/model.TQ3_4S.gguf',
            alias='tq3-moe',
            architecture_type='moe',
            expert_count=128,
            expert_used_count=8,
            tq3_status='native',
            tq3_weight_format='TQ3_4S',
            ctx_min=2048,
            ctx_max=131072,
        )
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('tq3', 'llama-server'),
            )
            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('tq3')), \
                 patch('llama_tui.benchmark.model_file_size', return_value=20 * 1024 ** 3):
                profiles = active_engine_runtime_profiles(
                    app,
                    model,
                    HardwareProfile(gpu_memory_total=8 * 1024 ** 3, gpu_memory_free=6 * 1024 ** 3),
                    depth='fast',
                )

        self.assertTrue(profiles)
        self.assertLessEqual(len(profiles), 6)
        self.assertTrue(all(item.benchmark_strategy_id == 'tq3_native_probe' for item in profiles))
        self.assertIn('server_sanity', {item.benchmark_phase for item in profiles})

    def test_mtp_strategy_tracks_acceptance_metrics(self):
        model = ModelConfig(
            id='mtp',
            name='Generic MTP-capable GGUF',
            path='/models/generic-mtp-model.gguf',
            alias='mtp',
            supports_mtp='yes',
        )
        caps = replace(default_engine_capabilities('llama.cpp-mtp'), supports_spec_type=True, supports_mtp=True, mtp_spec_type='mtp', supports_spec_draft_n_max=True)
        strategy = select_benchmark_strategy('llama.cpp-mtp', model, capabilities=caps, depth='fast')

        self.assertEqual(strategy.id, 'mtp_acceptance_matrix')
        self.assertIn('accept_rate', strategy.metric_groups)
        self.assertEqual(strategy.max_candidates, 20)

    def test_mtp_engine_selects_acceptance_matrix_from_generic_cache_provenance(self):
        caps = replace(default_engine_capabilities('llama.cpp-mtp'), supports_spec_type=True, supports_mtp=True, mtp_spec_type='mtp', supports_spec_draft_n_max=True)
        model = ModelConfig(
            id='generic',
            name='Generic Model',
            path='/cache/hub/models--owner--generic-MTP-GGUF/snapshots/abc/model.gguf',
            alias='generic',
        )

        strategy = select_benchmark_strategy(
            engine_id='llama.cpp-mtp',
            model=model,
            capabilities=caps,
            objective='quick_sanity',
        )

        self.assertEqual(strategy.id, 'mtp_acceptance_matrix')
        self.assertFalse(strategy.blocked_reason)

    def test_mtp_engine_blocks_uncertain_model_instead_of_full_suite(self):
        caps = replace(default_engine_capabilities('llama.cpp-mtp'), supports_spec_type=True, supports_mtp=True, mtp_spec_type='mtp', supports_spec_draft_n_max=True)
        model = ModelConfig(
            id='generic',
            name='Generic Model',
            path='/models/generic.gguf',
            alias='generic',
        )

        strategy = select_benchmark_strategy(
            engine_id='llama.cpp-mtp',
            model=model,
            capabilities=caps,
            objective='quick_sanity',
        )

        self.assertEqual(strategy.id, 'mtp_acceptance_matrix')
        self.assertTrue(strategy.blocked_reason)
        self.assertEqual(strategy.max_candidates, 0)
        self.assertEqual(strategy.phases, ())

    def test_mtp_strategy_blocks_when_binary_lacks_spec_flags(self):
        model = ModelConfig(
            id='mtp',
            name='Generic MTP-capable GGUF',
            path='/models/generic-mtp-model.gguf',
            alias='mtp',
            supports_mtp='yes',
        )
        strategy = select_benchmark_strategy('llama.cpp-mtp', model, capabilities=default_engine_capabilities('llama.cpp-mtp'))

        self.assertEqual(strategy.id, 'mtp_acceptance_matrix')
        self.assertEqual(strategy.max_candidates, 0)
        self.assertTrue(strategy.blocked_reason)
        self.assertEqual(strategy.phases, ())

    def test_turboquant_strategy_separates_kv_matrix(self):
        model = ModelConfig(
            id='tq',
            name='TurboQuant native',
            path='/models/model.gguf',
            alias='tq',
            turboquant_status='native',
            turboquant_head_dim=128,
        )
        strategy = select_benchmark_strategy('turboquant', model, depth='full')

        self.assertEqual(strategy.id, 'turboquant_kv_sweep')
        self.assertIn('kv_compression', [phase.id for phase in strategy.phases])
        self.assertIn('quality_risk', strategy.metric_groups)

    def test_vllm_strategy_does_not_use_llama_bench(self):
        model = ModelConfig(id='hf', name='HF', path='owner/model', alias='hf', runtime='vllm')
        strategy = select_benchmark_strategy('vllm', model)

        self.assertEqual(strategy.id, 'vllm_serving_latency')
        self.assertTrue(all(phase.runner == 'vllm_bench' for phase in strategy.phases))

    def test_runtime_profiles_carry_strategy_metadata(self):
        model = ModelConfig(
            id='mtp',
            name='Generic MTP-capable GGUF',
            path='/models/generic-mtp-model.gguf',
            alias='mtp',
            runtime='llama.cpp-mtp',
            supports_mtp='yes',
            ctx_min=4096,
            ctx_max=32768,
        )
        app = AppConfig(
            Path('/tmp/llama-tui-test-strategy.json'),
            runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
        )
        caps = replace(default_engine_capabilities('llama.cpp-mtp'), supports_spec_type=True, supports_mtp=True, mtp_spec_type='mtp', supports_spec_draft_n_max=True)
        app.engine_capabilities = lambda: caps

        profiles = active_engine_runtime_profiles(app, model, HardwareProfile(gpu_memory_total=8 * 1024 ** 3, gpu_memory_free=6 * 1024 ** 3), depth='fast')

        self.assertGreaterEqual(len(profiles), 4)
        self.assertTrue(all(item.benchmark_strategy_id == 'mtp_acceptance_matrix' for item in profiles))
        self.assertIn('draft_n1', {item.benchmark_phase for item in profiles})

    def test_parse_mtp_acceptance_metrics(self):
        metrics = parse_mtp_acceptance_metrics('draft_tokens: 120 accepted_tokens: 90 acceptance_rate: 75%')

        self.assertEqual(metrics['draft_tokens'], 120)
        self.assertEqual(metrics['accepted_tokens'], 90)
        self.assertEqual(metrics['accept_rate'], 0.75)

    def test_blocked_mtp_fast_and_smart_benchmarks_do_not_run_generic_fallback(self):
        caps = replace(default_engine_capabilities('llama.cpp-mtp'), supports_spec_type=True, supports_mtp=True, mtp_spec_type='mtp', supports_spec_draft_n_max=True)
        for runner in (benchmark_fast_profiles, benchmark_best_optimization):
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory() as tmp:
                app = AppConfig(
                    Path(tmp) / 'models.json',
                    runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
                )
                model = ModelConfig(
                    id='generic',
                    name='Generic Model',
                    path='/models/generic.gguf',
                    alias='generic',
                    port=18080,
                )
                progress = []
                with patch.object(app, 'hardware_profile', return_value=HardwareProfile(gpu_memory_total=8 * 1024 ** 3, gpu_memory_free=6 * 1024 ** 3)), \
                     patch.object(app, 'engine_capabilities', return_value=caps), \
                     patch('llama_tui.benchmark.benchmark_preflight_cleanup', return_value=(True, 'ok')), \
                     patch('llama_tui.benchmark.active_engine_runtime_profiles') as planner:
                    ok, msg = runner(app, model, progress=progress.append)

                self.assertFalse(ok)
                self.assertIn('Benchmark strategy blocked: mtp_acceptance_matrix', msg)
                planner.assert_not_called()
                progress_text = '\n'.join(str(item) for item in progress)
                self.assertIn('Detected features:', progress_text)
                self.assertIn('Detection sources:', progress_text)
                self.assertIn('Benchmark strategy blocked: mtp_acceptance_matrix', progress_text)
                self.assertIn('supports_mtp=yes', progress_text)

    def test_blocked_mtp_full_suite_does_not_start_generic_full_suite(self):
        caps = replace(default_engine_capabilities('llama.cpp-mtp'), supports_spec_type=True, supports_mtp=True, mtp_spec_type='mtp', supports_spec_draft_n_max=True)
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='generic',
                name='Generic Model',
                path='/models/generic.gguf',
                alias='generic',
                port=18080,
            )
            progress = []

            def forbidden(*_args, **_kwargs):
                raise AssertionError('generic full suite runner should not be called')

            with patch.object(app, 'hardware_profile', return_value=HardwareProfile(gpu_memory_total=8 * 1024 ** 3, gpu_memory_free=6 * 1024 ** 3)), \
                 patch.object(app, 'engine_capabilities', return_value=caps):
                ok, msg = benchmark_full_suite(
                    app,
                    model,
                    progress=progress.append,
                    moe_runner=forbidden,
                    smart_runner=forbidden,
                    hermes_runner=forbidden,
                    opencode_runner=forbidden,
                )

        self.assertFalse(ok)
        self.assertIn('Benchmark strategy blocked: mtp_acceptance_matrix', msg)
        progress_text = '\n'.join(str(item) for item in progress)
        self.assertNotIn('Full Suite Benchmark started', progress_text)
        self.assertIn('Benchmark strategy blocked: mtp_acceptance_matrix', progress_text)


if __name__ == '__main__':
    unittest.main()
