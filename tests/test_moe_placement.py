import unittest
from unittest.mock import patch

from llama_tui.hardware import HardwareProfile
from llama_tui.models import ModelConfig
from llama_tui.moe_placement import generate_moe_placement_candidates
from llama_tui.runtime_profiles import EngineCapabilities


class MoePlacementTests(unittest.TestCase):
    def model(self, architecture_type: str = 'moe') -> ModelConfig:
        return ModelConfig(
            id='model',
            name='Model',
            path='/models/model.gguf',
            alias='model',
            port=18080,
            architecture_type=architecture_type,
            expert_count=64 if architecture_type == 'moe' else 0,
            expert_used_count=8 if architecture_type == 'moe' else 0,
        )

    def test_dense_model_returns_no_candidates(self):
        caps = EngineCapabilities(supports_cpu_moe=True, supports_n_cpu_moe=True)
        candidates = generate_moe_placement_candidates(
            self.model('dense'),
            HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3),
            caps,
            'llama.cpp',
            'full',
        )

        self.assertEqual(candidates, [])

    def test_small_gpu_starts_with_conservative_partial_ngl_ladder(self):
        caps = EngineCapabilities(
            supports_cpu_moe=True,
            supports_n_cpu_moe=True,
            supports_override_tensor=True,
        )

        with patch('llama_tui.moe_placement.gguf_layer_count', return_value=40):
            candidates = generate_moe_placement_candidates(
                self.model(),
                HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3),
                caps,
                'llama.cpp',
                'full',
            )

        names = [item.name for item in candidates]
        partial_names = [name for name in names if name.startswith('partial_ngl_')]
        partial_values = [int(name.rsplit('_', 1)[1]) for name in partial_names]
        # New parametrized ladder (audit #18): small absolute floors plus
        # 1/4, 1/3, 1/2 fractions of layer_count. For 40 layers this
        # produces 8, 10, 12, 13, 16, 20. The exact set is verified in
        # test_partial_ngl_ladder_parametrizes_on_layer_count below.
        self.assertIn(8, partial_values)
        self.assertIn(20, partial_values)
        self.assertEqual(partial_values, sorted(partial_values))
        self.assertIn('cpu_moe_all', names)
        self.assertIn('n_cpu_moe_40', names)
        self.assertIn('experts_cpu_override', names)
        self.assertNotIn('full_gpu', names)
        self.assertLessEqual(len(candidates), 12)

    def test_partial_ngl_ladder_parametrizes_on_layer_count(self):
        # Audit #18: the ladder used to be a hardcoded (8, 12, 13, 16, 20)
        # tuned for ~32-layer models. Modern MoE shapes are bigger — verify
        # the parametrized version produces meaningfully different ladders
        # for representative architectures.
        from llama_tui.moe_placement import _partial_ngl_ladder

        # 40 layers (e.g. Llama-3 8B): small floors + 10, 13, 20.
        self.assertEqual(_partial_ngl_ladder(40), [8, 10, 12, 13, 16, 20])
        # 48 layers (Qwen3-30B-A3B): bigger fractions for the same floors.
        self.assertEqual(_partial_ngl_ladder(48), [8, 12, 16, 24])
        # 61 layers (DeepSeek-V3): 15, 20, 30 fractions.
        self.assertEqual(_partial_ngl_ladder(61), [8, 12, 15, 16, 20, 30])
        # Unknown layer count falls back to the conservative legacy set.
        self.assertEqual(_partial_ngl_ladder(0), [8, 12, 13, 16, 20])

    def test_unsupported_capabilities_suppress_cpu_moe_candidates(self):
        caps = EngineCapabilities()

        with patch('llama_tui.moe_placement.gguf_layer_count', return_value=40):
            candidates = generate_moe_placement_candidates(
                self.model(),
                HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3),
                caps,
                'llama.cpp',
                'fast',
            )

        names = {item.name for item in candidates}
        self.assertIn('partial_ngl_8', names)
        self.assertNotIn('cpu_moe_all', names)
        self.assertFalse(any(name.startswith('n_cpu_moe_') for name in names))

    def test_tq3_small_gpu_prioritizes_ncmoe_before_cmoe_and_partial(self):
        caps = EngineCapabilities(supports_cpu_moe=True, supports_n_cpu_moe=True)

        with patch('llama_tui.moe_placement.gguf_layer_count', return_value=40):
            candidates = generate_moe_placement_candidates(
                self.model(),
                HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3),
                caps,
                'tq3',
                'full',
            )

        self.assertEqual(
            [item.name for item in candidates[:6]],
            ['n_cpu_moe_32', 'n_cpu_moe_30', 'n_cpu_moe_36', 'n_cpu_moe_40', 'cpu_moe_all', 'baseline_ngl'],
        )
        self.assertTrue(all((item.gpu_layers == 999) for item in candidates[:5]))
        self.assertIsNone(candidates[-1].gpu_layers)

    def test_non_llama_family_engine_returns_no_candidates(self):
        caps = EngineCapabilities(supports_cpu_moe=True, supports_n_cpu_moe=True)
        candidates = generate_moe_placement_candidates(
            self.model(),
            HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3),
            caps,
            'vllm',
            'full',
        )

        self.assertEqual(candidates, [])


if __name__ == '__main__':
    unittest.main()
