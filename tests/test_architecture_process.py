import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_tui.benchmark import adaptive_record_from_candidate, architecture_payload, process_pressure_payload
from llama_tui.gguf import (
    ArchitectureInfo,
    apply_turboquant_info,
    architecture_label,
    detect_architecture_info,
    detect_turboquant_info,
    estimate_layer_weight_bytes_from_tensor_descriptors,
    turboquant_detail,
    turboquant_short,
)
from llama_tui.hardware import ProcessPressureSnapshot, benchmark_current_process_pressure, process_pressure_label
from llama_tui.models import ModelConfig
from llama_tui.optimize import apply_optimization_preset, choose_gpu_layers_for_profile
from llama_tui.hardware import HardwareProfile
from llama_tui.opencode_benchmark import benchmark_record_context
from llama_tui.hermes_benchmark import hermes_benchmark_record_context


class ArchitectureDetectionTests(unittest.TestCase):
    def detect_with_metadata(self, metadata, path='model.gguf', tensors=None):
        with patch('llama_tui.gguf.read_gguf_metadata', return_value=metadata), \
             patch('llama_tui.gguf.read_gguf_tensor_descriptors', return_value=tensors or []):
            return detect_architecture_info(ModelConfig(id='m', name='Model', path=path, alias='m', port=18080))

    def detect_turboquant_with_metadata(self, metadata, filename='model.gguf', runtime='llama.cpp'):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / filename
            path.write_bytes(b'GGUF')
            model = ModelConfig(id='m', name='Model', path=str(path), alias='m', port=18080, runtime=runtime)
            with patch('llama_tui.gguf.read_gguf_metadata', return_value=metadata):
                return detect_turboquant_info(model)

    def test_dense_metadata(self):
        info = self.detect_with_metadata({
            'general.architecture': 'llama',
            'llama.block_count': 32,
            'llama.embedding_length': 4096,
        })

        self.assertEqual(info.architecture_type, 'dense')
        self.assertGreaterEqual(info.confidence, 0.8)

    def test_qwen_moe_metadata(self):
        info = self.detect_with_metadata({
            'general.architecture': 'qwen2moe',
            'qwen2moe.expert_count': 60,
            'qwen2moe.expert_used_count': 4,
        })

        self.assertEqual(info.architecture_type, 'moe')
        self.assertEqual(info.expert_count, 60)
        self.assertEqual(info.expert_used_count, 4)
        self.assertAlmostEqual(info.active_expert_ratio, 4 / 60)

    def test_qwen36_moe_metadata_and_benchmark_payload(self):
        info = self.detect_with_metadata({
            'general.architecture': 'qwen35moe',
            'qwen35moe.expert_count': 256,
            'qwen35moe.expert_used_count': 8,
            'qwen35moe.attention.key_length': 256,
            'qwen35moe.attention.value_length': 256,
            'qwen35moe.attention.head_count': 64,
            'qwen35moe.attention.head_count_kv': 4,
            'qwen35moe.context_length': 32768,
        })

        model = ModelConfig(
            id='qwen36',
            name='Qwen3.6 35B A3B',
            path='qwen36.gguf',
            alias='qwen36',
            port=18080,
            architecture=info.architecture,
            architecture_type=info.architecture_type,
            expert_count=info.expert_count,
            expert_used_count=info.expert_used_count,
            active_expert_ratio=info.active_expert_ratio,
            classification_source=info.source,
            classification_confidence=info.confidence,
        )
        metadata = {
            'general.architecture': 'qwen35moe',
            'qwen35moe.expert_count': 256,
            'qwen35moe.expert_used_count': 8,
            'qwen35moe.attention.key_length': 256,
            'qwen35moe.attention.value_length': 256,
            'qwen35moe.attention.head_count': 64,
            'qwen35moe.attention.head_count_kv': 4,
            'qwen35moe.context_length': 32768,
        }
        with patch('llama_tui.benchmark.read_gguf_metadata', return_value=metadata), \
             patch('llama_tui.benchmark.model_file_size', return_value=int(11.44 * 1024**3)):
            payload = architecture_payload(model)

        self.assertEqual(info.architecture, 'qwen35moe')
        self.assertEqual(info.architecture_type, 'moe')
        self.assertEqual(info.expert_count, 256)
        self.assertEqual(info.expert_used_count, 8)
        self.assertEqual(payload['attention_key_length'], 256)
        self.assertEqual(payload['attention_value_length'], 256)
        self.assertEqual(payload['attention_head_count'], 64)
        self.assertEqual(payload['attention_head_count_kv'], 4)
        self.assertEqual(payload['native_context_length'], 32768)
        self.assertGreater(payload['model_file_size'], 11 * 1024**3)
        self.assertEqual(info.confidence, 1.0)

    def test_mixtral_style_metadata_label(self):
        info = self.detect_with_metadata({
            'general.architecture': 'llama',
            'llama.expert_count': 8,
            'llama.expert_used_count': 2,
        })
        model = ModelConfig(
            id='m',
            name='Model',
            path='model.gguf',
            alias='m',
            port=18080,
            architecture_type=info.architecture_type,
            expert_count=info.expert_count,
            expert_used_count=info.expert_used_count,
        )

        self.assertEqual(info.architecture_type, 'moe')
        self.assertEqual(architecture_label(model), 'MoE 8x2')

    def test_filename_fallback_for_active_parameter_name(self):
        info = self.detect_with_metadata({}, path='Qwen-30B-A3B-Q4.gguf')

        self.assertEqual(info.architecture_type, 'moe')
        self.assertEqual(info.source, 'filename_heuristic')
        self.assertGreaterEqual(info.confidence, 0.55)
        self.assertLess(info.confidence, 0.8)

    def test_unknown_without_metadata_or_filename_signal(self):
        info = self.detect_with_metadata({}, path='ordinary-model.gguf')

        self.assertEqual(info.architecture_type, 'unknown')
        self.assertEqual(info.confidence, 0.0)

    def test_no_false_positive_for_remote_substring(self):
        info = self.detect_with_metadata({}, path='remote-model.gguf')

        self.assertEqual(info.architecture_type, 'unknown')

    def test_tensor_name_fallback_and_layer_byte_estimate(self):
        tensors = [
            {'name': 'blk.0.attn_q.weight', 'dimensions': [4, 4], 'type': 0, 'offset': 0},
            {'name': 'blk.0.ffn_gate_exps.weight', 'dimensions': [4, 4], 'type': 0, 'offset': 64},
            {'name': 'blk.1.ffn_down.weight', 'dimensions': [4, 4], 'type': 0, 'offset': 192},
        ]
        info = self.detect_with_metadata({'general.architecture': ''}, tensors=tensors)

        self.assertEqual(info.architecture_type, 'moe')
        self.assertEqual(info.source, 'tensor_names')
        with patch('llama_tui.gguf.read_gguf_tensor_descriptors', return_value=tensors):
            self.assertEqual(estimate_layer_weight_bytes_from_tensor_descriptors('fake.gguf'), [192, 64])

    def test_turboquant_native_head_dim(self):
        info = self.detect_turboquant_with_metadata({
            'general.architecture': 'llama',
            'llama.attention.key_length': 128,
            'llama.attention.value_length': 128,
        })
        model = apply_turboquant_info(
            ModelConfig(id='m', name='Model', path='model.gguf', alias='m', port=18080),
            info,
        )

        self.assertEqual(info.status, 'native')
        self.assertEqual(info.head_dim, 128)
        self.assertEqual(info.key_dim, 128)
        self.assertEqual(turboquant_short(model), 'NAT')
        self.assertIn('key=128 value=128', turboquant_detail(model))

    def test_turboquant_padded_head_dims(self):
        for dim in (96, 64):
            with self.subTest(dim=dim):
                info = self.detect_turboquant_with_metadata({
                    'general.architecture': 'llama',
                    'llama.attention.key_length': dim,
                    'llama.attention.value_length': dim,
                })

                self.assertEqual(info.status, 'padded')
                self.assertEqual(info.head_dim, dim)
                self.assertIn('zero-padding', info.reason)

    def test_turboquant_falls_back_to_embedding_over_head_count(self):
        info = self.detect_turboquant_with_metadata({
            'general.architecture': 'llama',
            'llama.embedding_length': 4096,
            'llama.attention.head_count': 32,
        })

        self.assertEqual(info.status, 'native')
        self.assertEqual(info.key_dim, 128)
        self.assertEqual(info.value_dim, 128)
        self.assertEqual(info.source, 'gguf_metadata_fallback')

    def test_turboquant_missing_metadata_is_unknown(self):
        info = self.detect_turboquant_with_metadata({})

        self.assertEqual(info.status, 'unknown')
        self.assertIn('metadata', info.reason)

    def test_turboquant_vllm_and_non_gguf_are_not_applicable(self):
        vllm = detect_turboquant_info(ModelConfig(id='m', name='Model', path='org/model', alias='m', port=18080, runtime='vllm'))
        self.assertEqual(vllm.status, 'not_applicable')

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.txt'
            path.write_text('not gguf', encoding='utf-8')
            non_gguf = detect_turboquant_info(ModelConfig(id='m', name='Model', path=str(path), alias='m', port=18080))

        self.assertEqual(non_gguf.status, 'not_applicable')


class ProcessPressureTests(unittest.TestCase):
    def test_process_pressure_reads_fake_proc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'loadavg').write_text('4.00 2.00 1.00 3/120 999\n', encoding='utf-8')
            proc = root / '100'
            proc.mkdir()
            stat_tail = ' '.join(['0'] * 30)
            proc.joinpath('stat').write_text(f'100 (firefox) S {stat_tail}\n', encoding='utf-8')
            proc.joinpath('cmdline').write_text('firefox\0--profile\0dev', encoding='utf-8')
            proc.joinpath('statm').write_text('1000 500 0 0 0 0 0\n', encoding='utf-8')

            snapshot = benchmark_current_process_pressure(root)

        self.assertEqual(snapshot.runnable_processes, 3)
        self.assertEqual(snapshot.process_count, 1)
        self.assertEqual(snapshot.known_processes.get('browser'), 1)
        self.assertIn('pressure=', process_pressure_label(snapshot))

    def test_process_pressure_payload_is_flat(self):
        snapshot = ProcessPressureSnapshot(
            pressure_level='medium',
            pressure_score=0.5,
            detail='pressure=medium',
            process_count=7,
            known_processes={'ide': 1},
        )
        payload = process_pressure_payload(snapshot)

        self.assertEqual(payload['process_pressure_level'], 'medium')
        self.assertEqual(payload['process_known'], {'ide': 1})


class OptimizerProcessPressureTests(unittest.TestCase):
    def test_high_pressure_lowers_moe_gpu_layers_and_parallel(self):
        model = ModelConfig(
            id='moe',
            name='MoE',
            path='moe.gguf',
            alias='moe',
            port=18080,
            architecture_type='moe',
            expert_count=8,
            expert_used_count=2,
            ctx_min=2048,
            ctx_max=32768,
        )
        profile = HardwareProfile(
            cpu_logical=16,
            cpu_physical=8,
            memory_total=64 * 1024**3,
            memory_available=48 * 1024**3,
            gpu_memory_total=8 * 1024**3,
            gpu_memory_free=6 * 1024**3,
        )
        low = ProcessPressureSnapshot(pressure_score=0.1, pressure_level='low')
        high = ProcessPressureSnapshot(pressure_score=0.9, pressure_level='high')
        patches = [
            patch('llama_tui.optimize.model_file_size', return_value=12 * 1024**3),
            patch('llama_tui.optimize.gguf_layer_count', return_value=12),
            patch('llama_tui.optimize.estimate_layer_weight_bytes_from_tensor_descriptors', return_value=[512 * 1024**2] * 12),
        ]
        with patches[0], patches[1], patches[2]:
            with patch('llama_tui.optimize.benchmark_current_process_pressure', return_value=low):
                low_layers = choose_gpu_layers_for_profile(model, profile, 'moderate')
            with patch('llama_tui.optimize.benchmark_current_process_pressure', return_value=high):
                high_layers = choose_gpu_layers_for_profile(model, profile, 'moderate')
                candidate = ModelConfig(**model.__dict__)
                apply_optimization_preset(candidate, 'tokens_per_sec', tier='extreme', profile=profile)

        self.assertLess(high_layers, low_layers)
        self.assertEqual(candidate.parallel, 1)


class BenchmarkPayloadTests(unittest.TestCase):
    def test_adaptive_record_includes_architecture_and_process_pressure(self):
        model = ModelConfig(
            id='moe',
            name='MoE',
            path='moe.gguf',
            alias='moe',
            port=18080,
            architecture_type='moe',
            expert_count=8,
            expert_used_count=2,
            classification_source='gguf_metadata',
        )
        pressure = {'process_pressure_level': 'medium', 'process_pressure_score': 0.5}

        record = adaptive_record_from_candidate(
            model,
            'fast_chat',
            'ok',
            tokens_per_sec=42.0,
            process_snapshots={'after_generation': pressure},
        )

        self.assertEqual(record['architecture_label'], 'MoE 8x2')
        self.assertEqual(record['process_pressure_level'], 'medium')

    def test_workflow_benchmark_context_helpers_include_shared_payload(self):
        model = ModelConfig(
            id='dense',
            name='Dense',
            path='dense.gguf',
            alias='dense',
            port=18080,
            architecture_type='dense',
            classification_source='gguf_metadata',
        )
        pressure = {'process_pressure_level': 'low', 'process_pressure_score': 0.1}
        with patch('llama_tui.opencode_benchmark.current_process_pressure_payload', return_value=pressure), \
             patch('llama_tui.hermes_benchmark.current_process_pressure_payload', return_value=pressure):
            opencode_payload = benchmark_record_context(model)
            hermes_payload = hermes_benchmark_record_context(model)

        self.assertEqual(opencode_payload['architecture_label'], 'Dense')
        self.assertEqual(hermes_payload['process_pressure_level'], 'low')
