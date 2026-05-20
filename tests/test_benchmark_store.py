"""Unit tests for the per-engine benchmark payload helpers.

Covers the module extracted in audit #9 step 1. Everything here is
pure (no subprocess, no filesystem), so the tests are fast and use
real ``ModelConfig`` instances without mocking.
"""

import unittest

from llama_tui.benchmark_store import (
    ENGINE_BENCHMARK_FIELDS,
    apply_benchmark_payload,
    benchmark_payload_for_model,
    canonical_legacy_engine_key,
    copy_benchmark_value,
    default_engine_benchmark_payload,
    has_benchmark_payload,
)
from llama_tui.models import ModelConfig


def _model(**overrides) -> ModelConfig:
    return ModelConfig(
        id='m', name='M', path='/m.gguf', alias='m', port=18080,
        **overrides,
    )


class DefaultPayloadTests(unittest.TestCase):
    def test_default_payload_has_every_field(self):
        payload = default_engine_benchmark_payload()
        self.assertEqual(set(payload.keys()), set(ENGINE_BENCHMARK_FIELDS))

    def test_default_payload_uses_safe_types(self):
        # Lists / dicts must be fresh empties so callers can mutate
        # without leaking state into the global default.
        first = default_engine_benchmark_payload()
        second = default_engine_benchmark_payload()
        first['measured_profiles']['x'] = 1
        self.assertEqual(second['measured_profiles'], {})
        self.assertEqual(first['last_benchmark_tokens_per_sec'], 0.0)
        self.assertEqual(first['last_benchmark_seconds'], 0.0)
        self.assertEqual(first['last_benchmark_results'], [])


class CopyBenchmarkValueTests(unittest.TestCase):
    def test_scalar_passthrough(self):
        self.assertEqual(copy_benchmark_value(0.0), 0.0)
        self.assertEqual(copy_benchmark_value(''), '')
        self.assertEqual(copy_benchmark_value(42), 42)

    def test_list_is_duplicated(self):
        source = [{'a': 1}, {'a': 2}]
        copied = copy_benchmark_value(source)
        self.assertEqual(copied, source)
        copied[0]['a'] = 99
        self.assertEqual(source[0]['a'], 1)

    def test_dict_is_duplicated_one_level(self):
        source = {'k': {'a': 1}}
        copied = copy_benchmark_value(source)
        self.assertEqual(copied, source)
        copied['k']['a'] = 99
        self.assertEqual(source['k']['a'], 1)


class PayloadSnapshotAndApplyTests(unittest.TestCase):
    def test_snapshot_captures_all_engine_fields(self):
        model = _model(
            last_benchmark_tokens_per_sec=42.0,
            last_benchmark_profile='fast_chat',
            measured_profiles={'fast_chat': {'status': 'ok'}},
        )
        snapshot = benchmark_payload_for_model(model)
        self.assertEqual(snapshot['last_benchmark_tokens_per_sec'], 42.0)
        self.assertEqual(snapshot['last_benchmark_profile'], 'fast_chat')
        self.assertEqual(snapshot['measured_profiles'], {'fast_chat': {'status': 'ok'}})

    def test_apply_writes_payload_to_model(self):
        model = _model()
        apply_benchmark_payload(model, {
            'last_benchmark_tokens_per_sec': 99.5,
            'last_benchmark_profile': 'long_context',
            'measured_profiles': {'long_context': {'status': 'ok'}},
        })
        self.assertEqual(model.last_benchmark_tokens_per_sec, 99.5)
        self.assertEqual(model.last_benchmark_profile, 'long_context')
        self.assertEqual(model.measured_profiles, {'long_context': {'status': 'ok'}})

    def test_apply_with_missing_fields_uses_defaults(self):
        model = _model(
            last_benchmark_tokens_per_sec=42.0,
            measured_profiles={'x': {'status': 'ok'}},
        )
        apply_benchmark_payload(model, {})  # empty payload
        # Defaults clobber existing fields — that's the contract.
        self.assertEqual(model.last_benchmark_tokens_per_sec, 0.0)
        self.assertEqual(model.measured_profiles, {})

    def test_apply_coerces_wrong_typed_list_to_empty(self):
        model = _model()
        apply_benchmark_payload(model, {'last_benchmark_results': 'not a list'})
        self.assertEqual(model.last_benchmark_results, [])

    def test_apply_coerces_wrong_typed_dict_to_empty(self):
        model = _model()
        apply_benchmark_payload(model, {'measured_profiles': 'not a dict'})
        self.assertEqual(model.measured_profiles, {})


class HasBenchmarkPayloadTests(unittest.TestCase):
    def test_fresh_model_has_no_payload(self):
        self.assertFalse(has_benchmark_payload(_model()))

    def test_recorded_tokens_per_sec_signals_payload(self):
        self.assertTrue(has_benchmark_payload(_model(last_benchmark_tokens_per_sec=20.0)))

    def test_status_string_signals_payload(self):
        self.assertTrue(has_benchmark_payload(_model(default_benchmark_status='done')))

    def test_measured_profile_signals_payload(self):
        self.assertTrue(has_benchmark_payload(_model(measured_profiles={'k': {}})))

    def test_benchmark_runs_signal_payload(self):
        self.assertTrue(has_benchmark_payload(_model(benchmark_runs=[{}])))


class CanonicalLegacyEngineKeyTests(unittest.TestCase):
    def test_vllm_runtime_returns_vllm(self):
        self.assertEqual(canonical_legacy_engine_key(_model(runtime='vllm')), 'vllm')

    def test_llama_cpp_runtime_returns_llama_cpp(self):
        self.assertEqual(canonical_legacy_engine_key(_model(runtime='llama.cpp')), 'llama.cpp')

    def test_unknown_runtime_collapses_to_llama_cpp(self):
        # Pre-multi-engine builds only differentiated vLLM vs llama.cpp.
        self.assertEqual(canonical_legacy_engine_key(_model(runtime='buun')), 'llama.cpp')
        self.assertEqual(canonical_legacy_engine_key(_model(runtime='')), 'llama.cpp')


if __name__ == '__main__':
    unittest.main()
