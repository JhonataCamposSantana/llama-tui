import unittest
from unittest.mock import patch

from llama_tui.hardware import HardwareProfile
from llama_tui.models import ModelConfig
from llama_tui.optimize import (
    choose_threads_for_profile,
    effective_gpu_reserve_percent,
    gpu_reserve_percent_for_tier,
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

    def test_vllm_runtime_never_cpu(self):
        self.assertFalse(model_uses_cpu_execution(_model(ngl=0, runtime='vllm')))


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


if __name__ == '__main__':
    unittest.main()
