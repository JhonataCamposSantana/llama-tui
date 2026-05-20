"""Unit tests for the MoE-tuning pure helpers.

Covers the module extracted in audit #6 step 6. The heavy orchestrator
``benchmark_moe_placement_tuning`` is exercised indirectly via
``tests/test_tuning``; this file gives the small pure helpers their own
focused coverage.
"""

import unittest
from dataclasses import replace
from types import SimpleNamespace

from llama_tui.models import ModelConfig
from llama_tui.moe_tuning import (
    _context_validation_n_cpu_moe_values,
    _model_has_nextn_or_recurrent_features,
    _model_mtp_acceptance_records,
    _moe_tuning_layer_count,
    _moe_tuning_mtp_acceptance_required_reason,
    _moe_tuning_mtp_aware,
    _moe_tuning_mtp_blocked_reason,
    _moe_tuning_mtp_required,
    _moe_tuning_warnings,
    moe_context_bucket_specs,
)


def _model(**overrides) -> ModelConfig:
    defaults = dict(id='m', name='M', path='/m.gguf', alias='m', port=18080)
    defaults.update(overrides)
    return ModelConfig(**defaults)


class LayerCountTests(unittest.TestCase):
    def test_default_model_returns_zero(self):
        # Model with no real GGUF path → 0 (safe fallback when gguf
        # parsing fails).
        self.assertEqual(_moe_tuning_layer_count(_model(path='/nope.gguf')), 0)


class MtpAcceptanceRecordsTests(unittest.TestCase):
    def test_no_records_returns_empty(self):
        self.assertEqual(_model_mtp_acceptance_records(_model()), [])

    def test_picks_up_measured_profile_entry(self):
        model = _model(measured_profiles={
            'mtp_acceptance': {'accept_rate': 0.8, 'mtp_enabled': True}
        })
        records = _model_mtp_acceptance_records(model)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['accept_rate'], 0.8)

    def test_picks_up_benchmark_run_records(self):
        model = _model(benchmark_runs=[
            {
                'benchmark_strategy_id': 'mtp_acceptance_matrix',
                'records': [{'mtp_enabled': True, 'accept_rate': 0.7}],
            },
            # Unrelated strategy must be ignored.
            {
                'benchmark_strategy_id': 'fast_chat',
                'records': [{'tokens_per_sec': 50.0}],
            },
        ])
        records = _model_mtp_acceptance_records(model)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['accept_rate'], 0.7)


class MtpAwareEligibilityTests(unittest.TestCase):
    def _caps(self, **overrides):
        return SimpleNamespace(
            supports_spec_type=overrides.get('supports_spec_type', True),
            supports_mtp=overrides.get('supports_mtp', True),
            supports_spec_draft_n_max=overrides.get('supports_spec_draft_n_max', True),
            spec_type_values=overrides.get('spec_type_values', ('draft-mtp',)),
            mtp_spec_type_value=overrides.get('mtp_spec_type_value', 'draft-mtp'),
            mtp_spec_type='draft-mtp',
        )

    def test_non_mtp_engine_is_never_aware(self):
        # The function only reports True for the legacy llama.cpp-mtp
        # engine alias. Plain 'llama.cpp' returns False even with a
        # fully-capable binary.
        self.assertFalse(_moe_tuning_mtp_aware('llama.cpp', _model(), self._caps()))

    def test_missing_binary_capability_blocks(self):
        # Engine = mtp alias but binary doesn't advertise --spec-type:
        # eligibility is False.
        self.assertFalse(_moe_tuning_mtp_aware('llama.cpp-mtp', _model(), self._caps(supports_spec_type=False)))


class MtpBlockedReasonTests(unittest.TestCase):
    def test_reports_missing_flag(self):
        caps = SimpleNamespace(
            supports_spec_type=False,
            supports_mtp=False,
            supports_spec_draft_n_max=False,
            spec_type_values=(),
            mtp_spec_type_value='',
        )
        reason = _moe_tuning_mtp_blocked_reason(caps)
        self.assertIn('--spec-type', reason)
        self.assertIn('--spec-draft-n-max', reason)

    def test_reports_when_capabilities_object_is_none(self):
        # None caps must not raise — defensive code path.
        reason = _moe_tuning_mtp_blocked_reason(None)
        self.assertTrue(reason)


class MtpAcceptanceReasonTests(unittest.TestCase):
    def test_reason_string_is_actionable(self):
        reason = _moe_tuning_mtp_acceptance_required_reason(_model())
        self.assertIn('MTP Optimizer', reason)
        self.assertIn('draft_n', reason)


class NextnRecurrentSniffTests(unittest.TestCase):
    def test_default_model_is_not_recurrent(self):
        self.assertFalse(_model_has_nextn_or_recurrent_features(_model()))

    def test_filename_marker_is_detected(self):
        self.assertTrue(_model_has_nextn_or_recurrent_features(_model(path='/models/qwen3-nextn-30b.gguf')))
        self.assertTrue(_model_has_nextn_or_recurrent_features(_model(architecture='recurrent-state')))
        self.assertTrue(_model_has_nextn_or_recurrent_features(_model(model_family='mamba-ssm')))


class ContextBucketSpecsTests(unittest.TestCase):
    def test_full_depth_returns_all_four_buckets(self):
        specs = moe_context_bucket_specs(_model(), depth='full')
        keys = {spec['profile_key'] for spec in specs}
        self.assertEqual(keys, {'fast_chat', 'auto', 'hermes_ready', 'long_context'})

    def test_non_full_depth_returns_two_buckets(self):
        specs = moe_context_bucket_specs(_model(), depth='fast')
        self.assertEqual(len(specs), 2)

    def test_specs_clamped_to_model_ctx_window(self):
        # ctx_max=8192 must clamp every bucket at or below 8192.
        specs = moe_context_bucket_specs(_model(ctx_max=8192), depth='full')
        for spec in specs:
            self.assertLessEqual(int(spec['ctx']), 8192)


class ValidationLadderTests(unittest.TestCase):
    def test_returns_empty_for_invalid_inputs(self):
        self.assertEqual(_context_validation_n_cpu_moe_values(0, 40, 'full'), [])
        self.assertEqual(_context_validation_n_cpu_moe_values(20, 0, 'full'), [])

    def test_full_depth_has_five_step_deltas(self):
        result = _context_validation_n_cpu_moe_values(20, 40, 'full')
        self.assertIn(20, result)  # the centre value
        # Full ladder uses -4, -2, 0, 2, 4 deltas → 5 distinct values
        # when not clamped against bounds.
        self.assertEqual(len(result), 5)

    def test_fast_depth_uses_smaller_ladder(self):
        result = _context_validation_n_cpu_moe_values(20, 40, 'fast')
        self.assertLessEqual(len(result), 3)


class WarningsTests(unittest.TestCase):
    def test_layer_count_zero_warns(self):
        warnings = _moe_tuning_warnings([], layer_count=0, early_stop_text='')
        self.assertTrue(any('layer count' in w.lower() for w in warnings))

    def test_oom_failures_listed(self):
        records = [
            {'status': 'failed', 'failure_category': 'CUDA_OOM_KV', 'runtime_profile': 'p1'},
            {'status': 'failed', 'failure_category': 'MEMORY_GUARDRAIL', 'runtime_profile': 'p2'},
        ]
        warnings = _moe_tuning_warnings(records, layer_count=40, early_stop_text='')
        self.assertTrue(any('OOM' in w or 'unsafe' in w for w in warnings))

    def test_early_stop_text_propagates(self):
        warnings = _moe_tuning_warnings([], layer_count=40, early_stop_text='budget exhausted')
        self.assertIn('budget exhausted', warnings)

    def test_dedupe_collapses_repeats(self):
        # Same warning surfaces twice through different sources — dedup
        # should leave only one copy.
        warnings = _moe_tuning_warnings([], layer_count=40, early_stop_text='retry hit cap')
        self.assertEqual(warnings.count('retry hit cap'), 1)


if __name__ == '__main__':
    unittest.main()
