import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_tui.app import AppConfig
from llama_tui.hardware import HardwareProfile
from llama_tui.models import ModelConfig


def large_profile() -> HardwareProfile:
    return HardwareProfile(
        cpu_logical=16,
        cpu_physical=8,
        memory_total=512 * 1024**3,
        memory_available=480 * 1024**3,
        gpu_name='test gpu',
        gpu_memory_total=128 * 1024**3,
        gpu_memory_free=120 * 1024**3,
    )


class ModelVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.app = AppConfig(self.root / 'models.json')

    def tearDown(self):
        self.tmp.cleanup()

    def model(self, **overrides) -> ModelConfig:
        payload = {
            'id': 'm',
            'name': 'Model',
            'path': 'org/model',
            'alias': 'model',
            'port': 18080,
            'runtime': 'vllm',
            'ctx': 8192,
            'ctx_min': 2048,
            'ctx_max': 131072,
        }
        payload.update(overrides)
        return ModelConfig(**payload)

    def test_static_diagnostics_reports_missing_invalid_magic_and_truncated_files(self):
        missing = ModelConfig(
            id='missing',
            name='Missing',
            path=str(self.root / 'missing.gguf'),
            alias='missing',
            port=18080,
        )
        bad_suffix_path = self.root / 'model.txt'
        bad_suffix_path.write_text('not gguf', encoding='utf-8')
        bad_suffix = ModelConfig(
            id='suffix',
            name='Suffix',
            path=str(bad_suffix_path),
            alias='suffix',
            port=18081,
        )
        bad_magic_path = self.root / 'bad.gguf'
        bad_magic_path.write_bytes(b'NOPE' + (b'\0' * 64))
        bad_magic = ModelConfig(
            id='magic',
            name='Magic',
            path=str(bad_magic_path),
            alias='magic',
            port=18082,
        )
        truncated_path = self.root / 'truncated.gguf'
        truncated_path.write_bytes(b'GGUF')
        truncated = ModelConfig(
            id='truncated',
            name='Truncated',
            path=str(truncated_path),
            alias='truncated',
            port=18083,
        )

        self.assertEqual(self.app.static_model_diagnostics(missing)['status'], 'failed')
        self.assertIn('missing', self.app.static_model_diagnostics(missing)['reason'])
        self.assertIn('not a GGUF', self.app.static_model_diagnostics(bad_suffix)['reason'])
        self.assertIn('bad GGUF magic', self.app.static_model_diagnostics(bad_magic)['reason'])
        self.assertIn('truncated', self.app.static_model_diagnostics(truncated)['reason'])

    def test_fresh_benchmark_fingerprint_passes_verification(self):
        model = self.model()
        model.default_benchmark_status = 'done'
        model.benchmark_fingerprint = self.app.model_fingerprint(model)
        model.measured_profiles = {
            'auto': {
                'status': 'ok',
                'tokens_per_sec': 42.0,
                'ctx': 8192,
                'ctx_per_slot': 8192,
                'parallel': 1,
            }
        }

        with patch.object(self.app, 'hardware_profile', return_value=large_profile()):
            result = self.app.verify_model(model)

        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['fresh_benchmark'])
        self.assertEqual(self.app.models[0].verification_status, 'passed')

    def test_missing_or_stale_benchmark_proof_needs_benchmark(self):
        missing = self.model(id='missing', alias='missing')
        stale = self.model(id='stale', alias='stale')
        stale.default_benchmark_status = 'done'
        stale.benchmark_fingerprint = 'old-fingerprint'
        stale.measured_profiles = {
            'auto': {
                'status': 'ok',
                'tokens_per_sec': 42.0,
                'ctx_per_slot': 8192,
                'parallel': 1,
            }
        }

        with patch.object(self.app, 'hardware_profile', return_value=large_profile()):
            missing_result = self.app.verify_model(missing, save=False)
            stale_result = self.app.verify_model(stale, save=False)

        self.assertEqual(missing_result['status'], 'needs_benchmark')
        self.assertEqual(stale_result['status'], 'needs_benchmark')

    def test_cap_diagnosis_names_ctx_max_native_hardware_and_parallel_limits(self):
        with patch.object(self.app, 'hardware_profile', return_value=large_profile()):
            ctx_max = self.app.model_cap_diagnosis(self.model(ctx=8192, ctx_max=4096))
            parallel = self.app.model_cap_diagnosis(self.model(ctx=4096, parallel=4))
            proof_model = self.model(ctx=65536, parallel=4)
            proof_model.measured_profiles = {
                'auto': {'status': 'ok', 'ctx_per_slot': 4096, 'tokens_per_sec': 20.0}
            }
            proof = self.app.model_cap_diagnosis(proof_model)

        self.assertEqual(ctx_max['limiting_factor'], 'user_ctx_max')
        self.assertEqual(parallel['limiting_factor'], 'parallel_split')
        self.assertEqual(parallel['ctx_per_slot'], 1024)
        self.assertEqual(proof['limiting_factor'], 'benchmark_proof')
        self.assertEqual(proof['effective_limit'], 4096)

        native_model = self.model(runtime='llama.cpp', path=str(self.root / 'native.gguf'), ctx=8192)
        with patch.object(self.app, 'hardware_profile', return_value=large_profile()):
            with patch.object(self.app, 'static_model_diagnostics', return_value={'native_context': 4096, 'status': 'passed'}):
                native = self.app.model_cap_diagnosis(native_model)
        self.assertEqual(native['limiting_factor'], 'model_native_context')

        tiny_profile = HardwareProfile(
            cpu_logical=2,
            cpu_physical=1,
            memory_total=1024**3,
            memory_available=256 * 1024**2,
        )
        with patch.object(self.app, 'hardware_profile', return_value=tiny_profile):
            hardware = self.app.model_cap_diagnosis(self.model(ctx=65536, ctx_max=131072))
        self.assertEqual(hardware['limiting_factor'], 'hardware_safe_context')

    def test_benchmark_proof_model_ids_selects_only_enabled_stale_or_missing_models(self):
        fresh = self.model(id='fresh', alias='fresh')
        fresh.default_benchmark_status = 'done'
        fresh.benchmark_fingerprint = self.app.model_fingerprint(fresh)
        fresh.measured_profiles = {
            'auto': {'status': 'ok', 'tokens_per_sec': 10.0, 'ctx_per_slot': 4096, 'parallel': 1}
        }
        missing = self.model(id='missing', alias='missing')
        disabled = self.model(id='disabled', alias='disabled', enabled=False)
        self.app.models = [fresh, missing, disabled]

        self.assertEqual(self.app.benchmark_proof_model_ids(force=False), ['missing'])
        self.assertEqual(self.app.benchmark_proof_model_ids(force=True), ['fresh', 'missing'])


if __name__ == '__main__':
    unittest.main()
