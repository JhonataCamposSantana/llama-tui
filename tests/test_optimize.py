import unittest
from unittest.mock import patch

from llama_tui.hardware import HardwareProfile
from llama_tui.models import ModelConfig
from llama_tui.optimize import (
    choose_threads_for_profile,
    effective_gpu_reserve_percent,
    estimate_safe_context_for_profile,
    gpu_reserve_percent_for_tier,
    kv_cache_uses_gpu,
    model_is_moe,
    model_uses_cpu_execution,
)


def _model(**kwargs):
    base = dict(id='m', name='M', path='/m.gguf', alias='m')
    base.update(kwargs)
    return ModelConfig(**base)


class ModelIsMoeTests(unittest.TestCase):
    def test_moe_architecture(self):
        self.assertTrue(model_is_moe(_model(architecture_type='moe')))
        self.assertTrue(model_is_moe(_model(architecture_type='MoE')))

    def test_dense_architecture(self):
        self.assertFalse(model_is_moe(_model()))
        self.assertFalse(model_is_moe(_model(architecture_type='dense')))


class CpuExecutionTests(unittest.TestCase):
    def test_zero_gpu_layers_uses_cpu(self):
        self.assertTrue(model_uses_cpu_execution(_model(ngl=0)))

    def test_gpu_layers_without_profile_uses_gpu(self):
        self.assertFalse(model_uses_cpu_execution(_model(ngl=99)))

    def test_gpu_layers_but_no_usable_gpu_uses_cpu(self):
        no_gpu = HardwareProfile(gpu_memory_free=0)
        self.assertTrue(model_uses_cpu_execution(_model(ngl=99), no_gpu))


class GpuReserveTests(unittest.TestCase):
    def test_reserve_percent_is_a_sane_int(self):
        for tier in ('aggressive', 'moderate', 'conservative', 'unknown-tier'):
            reserve = gpu_reserve_percent_for_tier(tier)
            self.assertIsInstance(reserve, int)
            self.assertGreater(reserve, 0)

    def test_effective_reserve_never_exceeds_tier_reserve(self):
        tier = 'moderate'
        cap = gpu_reserve_percent_for_tier(tier)
        self.assertLessEqual(effective_gpu_reserve_percent(95, tier), cap)
        self.assertLessEqual(effective_gpu_reserve_percent(1, tier), cap)


class KvCacheUsesGpuTests(unittest.TestCase):
    # Regression for the 2026-05-26 cpu_moe ctx-cap bug. The legacy version
    # returned False when model.ngl==0, which made estimate_gpu_context_for_profile
    # short-circuit to None and allowed estimate_safe_context_for_profile to
    # return values up to ctx_max purely on RAM budget -- letting the opencode
    # ladder test ctx=131072 on an 8GB GPU and OOM there.

    _HW = HardwareProfile(cpu_physical=8, cpu_logical=12, gpu_memory_total=8 * 1024**3, gpu_memory_free=6 * 1024**3)

    def test_ngl_zero_with_usable_gpu_still_uses_gpu_kv(self):
        # A persisted MoE winner often ends up with ngl=0 -- but llama.cpp
        # keeps the KV on GPU regardless of layer offload.
        self.assertTrue(kv_cache_uses_gpu(_model(ngl=0), self._HW))

    def test_explicit_no_kv_offload_disables_gpu_kv(self):
        # Only --no-kv-offload genuinely puts KV on CPU.
        self.assertFalse(kv_cache_uses_gpu(_model(ngl=99, extra_args=['--no-kv-offload']), self._HW))
        self.assertFalse(kv_cache_uses_gpu(_model(ngl=99, extra_args=['-nkvo']), self._HW))

    def test_no_gpu_returns_false(self):
        no_gpu = HardwareProfile(gpu_memory_free=0)
        self.assertFalse(kv_cache_uses_gpu(_model(ngl=99), no_gpu))


class SafeContextEstimatorTests(unittest.TestCase):
    # Verifies that an MoE that can't fit on a small GPU gets ctx capped
    # well below ctx_max -- the cpu_moe-aware refinement frees enough VRAM
    # budget for the KV cache but not so much that the runtime OOMs.

    _SMALL_GPU = HardwareProfile(
        cpu_physical=8, cpu_logical=12,
        memory_total=int(23.2 * 1024**3), memory_available=int(18.6 * 1024**3),
        gpu_memory_total=int(8.0 * 1024**3), gpu_memory_free=int(6.7 * 1024**3),
    )

    def test_moe_doesnt_fit_gpu_caps_ctx_below_ctx_max(self):
        # Persisted ngl=0 (cpu_moe winner) on a model that doesn't fit GPU.
        # With model_file_size mocked to 16 GiB (qwen3-Q3_K_S-like) the cap
        # must be well below ctx_max=131072 (the user's OOM point).
        moe = _model(
            architecture_type='moe', expert_count=256, expert_used_count=8,
            ngl=0, ctx_min=2048, ctx_max=131072,
        )
        with patch('llama_tui.optimize.model_file_size', return_value=16 * 1024**3), \
             patch('llama_tui.optimize.process_pressure_score', return_value=0.1):
            safe = estimate_safe_context_for_profile(moe, self._SMALL_GPU, 25, 1, 2048, 131072)
        self.assertGreaterEqual(safe, 2048)
        self.assertLess(safe, 131072,
            f'Expected MoE-on-small-VRAM to cap ctx well below 131072 to avoid OOM, got {safe}.')
        # Also bound it from below so the cpu_moe-aware fraction isn't
        # over-tightened -- the user still needs room for ~5K-token workflows.
        self.assertGreaterEqual(safe, 16384,
            f'cpu_moe-aware fraction may be too aggressive; cap dropped to {safe}.')

    def test_dense_model_unaffected_by_cpu_moe_branch(self):
        # The cpu_moe-aware refinement only fires for MoE; dense models
        # should still get the regular partial-offload fraction.
        from llama_tui.optimize import estimate_gpu_weight_bytes
        dense = _model(architecture_type='dense', ngl=0, ctx_max=131072)
        with patch('llama_tui.optimize.model_file_size', return_value=16 * 1024**3):
            weights = estimate_gpu_weight_bytes(dense, self._SMALL_GPU, 'moderate', 5 * 1024**3)
        # Regular MoE partial-offload fraction 0.48 would have given ~2.4 GiB
        # of weights on a 5 GiB usable_gpu; dense uses 0.58 (~2.9 GiB). We
        # just need to confirm the cpu_moe path was NOT taken (which would
        # have capped weights below the dense partial-offload value).
        # The new constant for dense isn't lower, so weights should be at
        # least the equivalent of the dense partial-offload fraction.
        self.assertGreater(weights, int(5 * 1024**3 * 0.40),
            'Dense model should not be subject to the MoE cpu_moe refinement.')


class ChooseThreadsForProfileTests(unittest.TestCase):
    # 8 physical / 12 logical, with a usable (but too-small-for-the-model) GPU.
    _HW = HardwareProfile(cpu_physical=8, cpu_logical=12, gpu_memory_total=8 * 1024**3, gpu_memory_free=6 * 1024**3)

    def _moe(self):
        # No real GGUF file -> model_file_size==0 -> model_likely_fits_gpu False,
        # i.e. experts will run on CPU.
        return _model(architecture_type='moe', expert_count=256, expert_used_count=8, threads=6)

    def test_moe_on_cpu_uses_physical_cores_under_low_pressure(self):
        with patch('llama_tui.optimize.process_pressure_score', return_value=0.1):
            for tier in ('safe', 'moderate', 'extreme'):
                self.assertEqual(
                    choose_threads_for_profile(self._moe(), self._HW, tier), 8,
                    f'tier={tier} should give physical cores for MoE-on-CPU',
                )

    def test_moe_on_cpu_backs_off_under_pressure(self):
        with patch('llama_tui.optimize.process_pressure_score', return_value=0.6):
            self.assertEqual(choose_threads_for_profile(self._moe(), self._HW, 'safe'), 6)

    def test_dense_model_still_respects_safe_tier_reduction(self):
        dense = _model(threads=6)
        with patch('llama_tui.optimize.process_pressure_score', return_value=0.1):
            # Dense + safe tier keeps the physical-2 headroom reduction (6).
            self.assertEqual(choose_threads_for_profile(dense, self._HW, 'safe'), 6)

    def test_moe_on_hybrid_cpu_uses_performance_cores(self):
        # 4 P-cores + 4 E-cores = 8 physical / 12 logical.
        hybrid = HardwareProfile(
            cpu_physical=8, cpu_logical=12, cpu_performance=4,
            gpu_memory_total=8 * 1024**3, gpu_memory_free=6 * 1024**3,
        )
        with patch('llama_tui.optimize.process_pressure_score', return_value=0.1):
            for tier in ('safe', 'moderate', 'extreme'):
                self.assertEqual(
                    choose_threads_for_profile(self._moe(), hybrid, tier), 4,
                    f'tier={tier} should give P-core count on a hybrid CPU',
                )

    def test_moe_on_hybrid_cpu_backs_off_under_pressure(self):
        hybrid = HardwareProfile(
            cpu_physical=8, cpu_logical=12, cpu_performance=4,
            gpu_memory_total=8 * 1024**3, gpu_memory_free=6 * 1024**3,
        )
        with patch('llama_tui.optimize.process_pressure_score', return_value=0.6):
            self.assertEqual(choose_threads_for_profile(self._moe(), hybrid, 'safe'), 2)


if __name__ == '__main__':
    unittest.main()
