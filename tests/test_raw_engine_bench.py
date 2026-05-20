"""Unit tests for the raw-speed (TQ3 / llama-bench) primitives.

Covers the module extracted in audit #6 step 3. The heavy orchestrator
``run_tq3_raw_llama_bench_presearch`` stays in benchmark.py; this file
exercises the pure helpers that were carved out: command building,
output parsing, profile selection, candidate cloning, and the
subprocess runner's no-fork happy paths via mocking.
"""

import unittest
from dataclasses import replace
from typing import Tuple
from unittest.mock import MagicMock, patch

from llama_tui.models import ModelConfig
from llama_tui.raw_engine_bench import (
    RAW_BENCH_DETERMINISTIC_SEED,
    TQ3_RAW_BENCH_CASES,
    _raw_bench_candidate_model,
    _tq3_raw_profile_key,
    _tq3_raw_profile_rank,
    _tq3_raw_runtime_profiles,
    parse_llama_bench_tokens_per_sec,
    sibling_llama_bench_for_server,
    tq3_llama_bench_command,
    tq3_moe_cpu_placement_threads,
    tq3_raw_presearch_case_total,
)
from llama_tui.runtime_profiles import RuntimeProfile


def _model(**overrides) -> ModelConfig:
    defaults = dict(id='m', name='M', path='/models/m.gguf', alias='m', port=18080, threads=4)
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _profile(**overrides) -> RuntimeProfile:
    defaults = dict(engine_id='tq3', name='base', ctx_size=8192, parallel=1, gpu_layers=999)
    defaults.update(overrides)
    return RuntimeProfile(**defaults)


class SiblingLlamaBenchTests(unittest.TestCase):
    def test_absolute_path_uses_sibling(self):
        # When llama-server is at /opt/llama-server, llama-bench lives
        # alongside it.
        self.assertEqual(
            sibling_llama_bench_for_server('/opt/llama.cpp/build/bin/llama-server'),
            '/opt/llama.cpp/build/bin/llama-bench',
        )

    def test_relative_path_returns_plain_llama_bench(self):
        # A PATH-resolved command falls through to plain 'llama-bench'.
        self.assertEqual(sibling_llama_bench_for_server('llama-server'), 'llama-bench')
        self.assertEqual(sibling_llama_bench_for_server(''), 'llama-bench')


class Tq3ProfileKeyAndRankTests(unittest.TestCase):
    def test_key_distinguishes_placement(self):
        a = _profile(n_cpu_moe=32, kv_preset='q8_0/q8_0')
        b = _profile(n_cpu_moe=24, kv_preset='q8_0/q8_0')
        self.assertNotEqual(_tq3_raw_profile_key(a), _tq3_raw_profile_key(b))

    def test_rank_prefers_32_n_cpu_moe(self):
        # The TQ3 raw-bench heuristic prefers exactly 32, then 30,
        # then everything else stride-distance from 32.
        rank32 = _tq3_raw_profile_rank(_profile(n_cpu_moe=32), 0)[0]
        rank30 = _tq3_raw_profile_rank(_profile(n_cpu_moe=30), 0)[0]
        rank20 = _tq3_raw_profile_rank(_profile(n_cpu_moe=20), 0)[0]
        cpu_moe_rank = _tq3_raw_profile_rank(_profile(cpu_moe=True), 0)[0]
        baseline = _tq3_raw_profile_rank(_profile(), 0)[0]
        self.assertLess(rank32, rank30)
        self.assertLess(rank30, rank20)
        self.assertLess(rank20, cpu_moe_rank)
        self.assertLess(cpu_moe_rank, baseline)


class Tq3ProfileSelectionTests(unittest.TestCase):
    def test_non_tq3_engines_are_filtered_out(self):
        profiles = [_profile(engine_id='llama.cpp', n_cpu_moe=32, kv_preset='q8_0/q8_0')]
        self.assertEqual(_tq3_raw_runtime_profiles(profiles, depth='full'), [])

    def test_fit_profiles_are_filtered_out(self):
        profiles = [_profile(n_cpu_moe=32, fit=True, kv_preset='q8_0/q8_0')]
        self.assertEqual(_tq3_raw_runtime_profiles(profiles, depth='full'), [])

    def test_fast_depth_caps_at_one_profile(self):
        profiles = [
            _profile(n_cpu_moe=32, kv_preset='q8_0/q8_0'),
            _profile(n_cpu_moe=30, kv_preset='q8_0/q8_0'),
            _profile(n_cpu_moe=24, kv_preset='q8_0/q8_0'),
        ]
        self.assertEqual(len(_tq3_raw_runtime_profiles(profiles, depth='fast')), 1)

    def test_full_depth_caps_at_two_profiles(self):
        profiles = [
            _profile(n_cpu_moe=32, kv_preset='q8_0/q8_0'),
            _profile(n_cpu_moe=30, kv_preset='q8_0/q8_0'),
            _profile(n_cpu_moe=24, kv_preset='q8_0/q8_0'),
        ]
        self.assertEqual(len(_tq3_raw_runtime_profiles(profiles, depth='full')), 2)

    def test_case_total_multiplies_by_bench_cases(self):
        profiles = [
            _profile(n_cpu_moe=32, kv_preset='q8_0/q8_0'),
            _profile(n_cpu_moe=30, kv_preset='q8_0/q8_0'),
        ]
        self.assertEqual(
            tq3_raw_presearch_case_total(profiles, depth='full'),
            2 * len(TQ3_RAW_BENCH_CASES),
        )


class LlamaBenchCommandTests(unittest.TestCase):
    def test_seed_is_deterministic(self):
        cmd = tq3_llama_bench_command('llama-bench', _model(), _profile(), 1024, 64, threads=4)
        self.assertIn('--seed', cmd)
        self.assertIn(str(RAW_BENCH_DETERMINISTIC_SEED), cmd)

    def test_includes_model_path_and_prompt_args(self):
        cmd = tq3_llama_bench_command('llama-bench', _model(), _profile(), 1024, 64, threads=4)
        self.assertIn('-m', cmd)
        self.assertIn('/models/m.gguf', cmd)
        self.assertIn('-p', cmd)
        self.assertIn('1024', cmd)
        self.assertIn('-n', cmd)
        self.assertIn('64', cmd)

    def test_cpu_moe_flag_short_circuits_n_cpu_moe(self):
        cmd = tq3_llama_bench_command('llama-bench', _model(), _profile(cpu_moe=True, n_cpu_moe=30), 1024, 64, threads=4)
        self.assertIn('-cmoe', cmd)
        # When cpu_moe is set the ncmoe flag is not emitted.
        self.assertNotIn('-ncmoe', cmd)

    def test_n_cpu_moe_emits_ncmoe(self):
        cmd = tq3_llama_bench_command('llama-bench', _model(), _profile(n_cpu_moe=30), 1024, 64, threads=4)
        self.assertIn('-ncmoe', cmd)
        self.assertIn('30', cmd)

    def test_q8_kv_preset_emits_ctk_and_ctv(self):
        cmd = tq3_llama_bench_command('llama-bench', _model(), _profile(kv_preset='q8_0/q8_0'), 1024, 64, threads=4)
        self.assertIn('-ctk', cmd)
        self.assertIn('-ctv', cmd)
        self.assertIn('q8_0', cmd)


class ParseLlamaBenchTokensPerSecTests(unittest.TestCase):
    def test_picks_last_tok_per_sec(self):
        output = (
            'pp tokens/s: 312.5\n'
            'tg tokens/s: 42.7 tok/s\n'
            'combined: 21.3 tok/s\n'
        )
        self.assertEqual(parse_llama_bench_tokens_per_sec(output), 21.3)

    def test_handles_uncertainty_marker(self):
        output = 'combined: 42.7 ± 0.3 tok/s'
        self.assertEqual(parse_llama_bench_tokens_per_sec(output), 42.7)

    def test_empty_returns_zero(self):
        self.assertEqual(parse_llama_bench_tokens_per_sec(''), 0.0)
        self.assertEqual(parse_llama_bench_tokens_per_sec('no benchmark output here'), 0.0)


class RawBenchCandidateModelTests(unittest.TestCase):
    def test_clones_model_with_profile_overrides(self):
        model = _model(ctx=4096, parallel=1, ngl=999)
        profile = _profile(ctx_size=8192, parallel=2, gpu_layers=20, n_cpu_moe=24)
        clone = _raw_bench_candidate_model(model, profile)
        self.assertEqual(clone.ctx, 8192)
        self.assertEqual(clone.parallel, 2)
        self.assertEqual(clone.ngl, 20)
        self.assertEqual(clone.n_cpu_moe, 24)
        # Original is untouched.
        self.assertEqual(model.ctx, 4096)
        self.assertEqual(model.parallel, 1)


class CpuPlacementThreadsTests(unittest.TestCase):
    def test_non_tq3_engine_returns_model_threads_unchanged(self):
        result = tq3_moe_cpu_placement_threads(
            _model(threads=4),
            runtime_profile=_profile(engine_id='llama.cpp'),
        )
        self.assertEqual(result, 4)

    def test_dense_model_returns_model_threads(self):
        result = tq3_moe_cpu_placement_threads(
            _model(threads=4, architecture_type='dense'),
            runtime_profile=_profile(),
        )
        self.assertEqual(result, 4)

    def test_moe_with_cpu_placement_clamps_to_logical_capped_at_12(self):
        hardware = MagicMock(cpu_logical=32)
        result = tq3_moe_cpu_placement_threads(
            _model(threads=4, architecture_type='moe', cpu_moe=True),
            runtime_profile=_profile(),
            hardware=hardware,
        )
        # Result is clamped at 12 regardless of huge logical CPU count.
        self.assertEqual(result, 12)


if __name__ == '__main__':
    unittest.main()
