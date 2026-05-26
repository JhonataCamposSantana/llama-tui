import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_tui.app import AppConfig
from llama_tui.benchmark import benchmark_preflight_cleanup
from llama_tui.gguf import TurboQuantInfo
from llama_tui.hardware import HardwareProfile
from llama_tui.models import ModelConfig
from llama_tui.runtime_profiles import make_runtime_profile


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
            'runtime': 'llama.cpp',
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

    def test_turboquant_metadata_is_enriched_and_persisted(self):
        model_path = self.root / 'model.gguf'
        model_path.write_bytes(b'GGUF')
        model = ModelConfig(
            id='tq',
            name='TurboQuant',
            path=str(model_path),
            alias='tq',
            port=18080,
        )
        detected = TurboQuantInfo(
            status='native',
            head_dim=128,
            key_dim=128,
            value_dim=128,
            source='gguf_metadata',
            reason='key/value head dims are multiples of 128',
        )

        with patch('llama_tui.app.detect_turboquant_info', return_value=detected):
            self.app.add_or_update(model)
            reloaded = AppConfig(self.root / 'models.json').get_model('tq')

        self.assertIsNotNone(reloaded)
        self.assertEqual(reloaded.turboquant_status, 'native')
        self.assertEqual(reloaded.turboquant_key_dim, 128)
        self.assertEqual(reloaded.turboquant_value_dim, 128)
        self.assertEqual(reloaded.turboquant_source, 'gguf_metadata')

    def turbo_app(self, **profile_kwargs) -> AppConfig:
        return AppConfig(
            self.root / 'models.json',
            runtime_profile=make_runtime_profile('turboquant', 'llama-server', **profile_kwargs),
        )

    def turbo_model(self, **overrides) -> ModelConfig:
        model = self.model(runtime='llama.cpp', **overrides)
        model.turboquant_status = overrides.get('turboquant_status', 'native')
        model.turboquant_key_dim = overrides.get('turboquant_key_dim', 128)
        model.turboquant_value_dim = overrides.get('turboquant_value_dim', 128)
        return model

    def test_turboquant_default_serves_turbo4_for_compatible_models(self):
        # Regression: a plain `--engine turboquant` session used to serve
        # -ctv q8_0 (zero turbo compression). Compatible (native/padded,
        # head_dim>=128) models now default the value cache to the
        # benchmark-validated turbo4 (2026-05-26 A/B: turbo4 was the safer
        # choice -- ~neutral on dense models and equal-or-better on MoE
        # vs turbo3, with less aggressive quantization).
        app = self.turbo_app()
        native = self.turbo_model(turboquant_status='native', turboquant_key_dim=256, turboquant_value_dim=256)
        padded = self.turbo_model(turboquant_status='padded', turboquant_key_dim=192, turboquant_value_dim=192)

        self.assertEqual(app.turboquant_served_kv_preset(native), 'q8_0/turbo4')
        self.assertEqual(app.turboquant_served_kv_preset(padded), 'q8_0/turbo4')

    def test_turboquant_incompatible_or_unknown_falls_back_to_q8(self):
        app = self.turbo_app()
        incompatible = self.turbo_model(turboquant_status='incompatible', turboquant_key_dim=64, turboquant_value_dim=64)
        unknown = self.turbo_model(turboquant_status='unknown', turboquant_key_dim=0, turboquant_value_dim=0)

        self.assertEqual(app.turboquant_served_kv_preset(incompatible), 'q8_0/q8_0')
        self.assertEqual(app.turboquant_served_kv_preset(unknown), 'q8_0/q8_0')

    def test_turboquant_explicit_value_is_respected_but_downgraded_when_unfit(self):
        explicit = self.turbo_app(kv_value_mode='turbo4')
        native = self.turbo_model(turboquant_status='native', turboquant_key_dim=128, turboquant_value_dim=128)
        unfit = self.turbo_model(turboquant_status='incompatible', turboquant_key_dim=64, turboquant_value_dim=64)

        self.assertEqual(explicit.turboquant_served_kv_preset(native), 'q8_0/turbo4')
        # An explicit turbo value still can't run on a sub-128 head dim.
        self.assertEqual(explicit.turboquant_served_kv_preset(unfit), 'q8_0/q8_0')

        opted_out = self.turbo_app(kv_value_mode='q8_0')
        self.assertEqual(opted_out.turboquant_served_kv_preset(native), 'q8_0/q8_0')

    def test_turboquant_benchmark_winner_overrides_default(self):
        app = self.turbo_app()
        native = self.turbo_model(turboquant_status='native', turboquant_key_dim=128, turboquant_value_dim=128)
        native.measured_profiles = {'auto': {'status': 'ok', 'kv_preset': 'q8_0/turbo2', 'tokens_per_sec': 12.0}}
        self.assertEqual(app.turboquant_served_kv_preset(native), 'q8_0/turbo2')

        # A winner where turbo lost (q8_0/q8_0) is honoured, not overridden.
        lost = self.turbo_model(turboquant_status='native', turboquant_key_dim=128, turboquant_value_dim=128)
        lost.measured_profiles = {'auto': {'status': 'ok', 'kv_preset': 'q8_0/q8_0', 'tokens_per_sec': 30.0}}
        self.assertEqual(app.turboquant_served_kv_preset(lost), 'q8_0/q8_0')

        # A stale llama.cpp record (f16) never leaks into a turbo launch.
        stale = self.turbo_model(turboquant_status='native', turboquant_key_dim=128, turboquant_value_dim=128)
        stale.measured_profiles = {'auto': {'status': 'ok', 'kv_preset': 'f16/f16', 'tokens_per_sec': 30.0}}
        self.assertEqual(app.turboquant_served_kv_preset(stale), 'q8_0/turbo4')

    def test_turboquant_per_model_override_beats_default_and_winner(self):
        # A user pin on the model (e.g. q4_0/q4_0 for an MXFP4 weight that
        # tolerates it -- 2026-05-26 bench) overrides the validated default
        # AND any persisted benchmark winner, but yields to an explicit
        # session --kv-value passed on the CLI.
        app = self.turbo_app()
        native = self.turbo_model(turboquant_status='native', turboquant_key_dim=128, turboquant_value_dim=128)
        native.kv_key_mode = 'q4_0'
        native.kv_value_mode = 'q4_0'
        native.measured_profiles = {'auto': {'status': 'ok', 'kv_preset': 'q8_0/turbo2', 'tokens_per_sec': 12.0}}
        self.assertEqual(app.turboquant_served_kv_preset(native), 'q4_0/q4_0')

        # An explicit session flag still wins over the per-model pin.
        explicit_session = self.turbo_app(kv_value_mode='turbo3')
        self.assertEqual(explicit_session.turboquant_served_kv_preset(native), 'q8_0/turbo3')

        # Per-model V only -- K falls through to the session default (q8_0).
        v_only = self.turbo_model(turboquant_status='native', turboquant_key_dim=128, turboquant_value_dim=128)
        v_only.kv_value_mode = 'q5_1'
        self.assertEqual(app.turboquant_served_kv_preset(v_only), 'q8_0/q5_1')

        # A turbo pin on a head_dim=64 model still gets downgraded -- the
        # model can't run turbo blocks no matter who set the preference.
        unfit = self.turbo_model(turboquant_status='incompatible', turboquant_key_dim=64, turboquant_value_dim=64)
        unfit.kv_value_mode = 'turbo3'
        self.assertEqual(app.turboquant_served_kv_preset(unfit), 'q8_0/q8_0')

    def test_tq3_native_gguf_is_unsupported_on_every_engine(self):
        # TQ3 engine removed (2026-05): a TQ3-native GGUF (detected by name)
        # can no longer be launched by any remaining engine.
        model_path = self.root / 'model.TQ3_4S.gguf'
        model_path.write_bytes(b'GGUF' + (b'\0' * 64))
        model = ModelConfig(
            id='tq3',
            name='TQ3',
            path=str(model_path),
            alias='tq3',
            port=18080,
        )

        ok, msg = self.app.start(model)

        self.assertFalse(ok)
        self.assertIn('TQ3-native GGUFs', msg)

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

        with patch.object(self.app, 'hardware_profile', return_value=large_profile()), \
                patch.object(self.app, 'static_model_diagnostics',
                             return_value={'status': 'passed', 'native_context': 4096, 'reason': 'GGUF metadata parsed'}):
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

        with patch.object(self.app, 'hardware_profile', return_value=large_profile()), \
                patch.object(self.app, 'static_model_diagnostics',
                             return_value={'status': 'passed', 'native_context': 4096, 'reason': 'GGUF metadata parsed'}):
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
