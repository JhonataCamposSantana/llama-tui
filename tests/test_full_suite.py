"""Unit tests for the full-suite pure helpers.

Covers the module extracted in audit #6 step 7. The heavy orchestrator
``benchmark_full_suite`` stays in benchmark.py; this file exercises
the tight pure subset that was carved out.
"""

import unittest
from dataclasses import replace

from llama_tui.full_suite import (
    SUITE_BENCHMARK_STATE_FIELDS,
    _latest_benchmark_run,
    _mtp_acceptance_stage_status,
    _suite_restore_config_fields,
    full_suite_summary_text,
)
from llama_tui.models import ModelConfig


def _model(**overrides) -> ModelConfig:
    defaults = dict(id='m', name='M', path='/m.gguf', alias='m', port=18080)
    defaults.update(overrides)
    return ModelConfig(**defaults)


class SuiteRestoreConfigFieldsTests(unittest.TestCase):
    def test_benchmark_state_is_preserved(self):
        original = _model(threads=4)
        current = _model(
            threads=8,  # changed during the suite
            last_benchmark_tokens_per_sec=42.0,  # benchmark state — keep
            measured_profiles={'auto': {'status': 'ok'}},
        )
        restored = _suite_restore_config_fields(current, original)
        # Pre-suite config is back…
        self.assertEqual(restored.threads, 4)
        # …but the freshly-measured benchmark numbers survive.
        self.assertEqual(restored.last_benchmark_tokens_per_sec, 42.0)
        self.assertEqual(restored.measured_profiles, {'auto': {'status': 'ok'}})

    def test_prior_profiles_merge_into_restored_model(self):
        # When the caller supplies prior_profiles, it must overlay onto
        # the current model's measured_profiles (not replace them).
        original = _model()
        current = _model(measured_profiles={'auto': {'status': 'ok'}})
        prior = {'fast_chat': {'status': 'ok'}}
        restored = _suite_restore_config_fields(current, original, prior_profiles=prior)
        self.assertIn('auto', restored.measured_profiles)
        self.assertIn('fast_chat', restored.measured_profiles)

    def test_state_field_set_matches_modelconfig(self):
        # Every name in SUITE_BENCHMARK_STATE_FIELDS must be a real
        # ModelConfig field, otherwise the restore loop silently skips
        # the wrong attributes.
        valid = set(getattr(ModelConfig, '__dataclass_fields__', {}))
        for field in SUITE_BENCHMARK_STATE_FIELDS:
            self.assertIn(field, valid, f'{field!r} is not a ModelConfig field')


class LatestBenchmarkRunTests(unittest.TestCase):
    def test_returns_first_matching_kind(self):
        model = _model(benchmark_runs=[
            {'kind': 'fast_chat', 'records': [], 'rank': 1},
            {'kind': 'long_context', 'records': [], 'rank': 2},
        ])
        run = _latest_benchmark_run(model, kind='long_context')
        self.assertEqual(run.get('rank'), 2)

    def test_strategy_id_narrows_search(self):
        model = _model(benchmark_runs=[
            {'kind': 'auto', 'benchmark_strategy_id': 'safe_baseline', 'rank': 1},
            {'kind': 'auto', 'benchmark_strategy_id': 'aggressive', 'rank': 2},
        ])
        run = _latest_benchmark_run(model, kind='auto', strategy_id='aggressive')
        self.assertEqual(run.get('rank'), 2)

    def test_missing_returns_empty_dict(self):
        model = _model(benchmark_runs=[{'kind': 'fast_chat', 'records': []}])
        self.assertEqual(_latest_benchmark_run(model, kind='nope'), {})


class MtpAcceptanceStageStatusTests(unittest.TestCase):
    def test_no_best_record_is_failed(self):
        self.assertEqual(_mtp_acceptance_stage_status({}, {}), 'failed')

    def test_done_run_status_with_best_is_done(self):
        self.assertEqual(
            _mtp_acceptance_stage_status({'status': 'done'}, {'accept_rate': 0.8}),
            'done',
        )

    def test_partial_with_best_is_usable(self):
        self.assertEqual(
            _mtp_acceptance_stage_status({'status': 'partial'}, {'accept_rate': 0.8}),
            'usable',
        )


class FullSuiteSummaryTextTests(unittest.TestCase):
    def test_mtp_path_when_mtp_acceptance_stage_present(self):
        records = [
            {'stage': 'mtp_acceptance', 'status': 'done'},
            {'stage': 'moe_placement', 'status': 'done'},
            {'stage': 'summary', 'status': 'done'},
        ]
        summary = full_suite_summary_text(records)
        self.assertIn('MTP Full Suite', summary)
        self.assertIn('mtp_acceptance=done', summary)

    def test_mtp_path_includes_best_draft_n_when_recommended(self):
        records = [{'stage': 'mtp_acceptance', 'status': 'done'}]
        summary = full_suite_summary_text(
            records,
            recommendations={'mtp_acceptance': {'draft_n': 3}},
        )
        self.assertIn('best_draft_n=3', summary)

    def test_full_path_when_no_mtp_stage(self):
        records = [
            {'stage': 'model_benchmark', 'status': 'done'},
            {'stage': 'hermes', 'status': 'done'},
            {'stage': 'opencode', 'status': 'failed'},
        ]
        summary = full_suite_summary_text(records)
        self.assertIn('Full Suite Benchmark complete', summary)
        self.assertIn('hermes=done', summary)
        self.assertIn('opencode=failed', summary)

    def test_warnings_counted_in_summary(self):
        records = [{'stage': 'model_benchmark', 'status': 'done'}]
        summary = full_suite_summary_text(records, warnings=['low VRAM', 'thermal cap'])
        self.assertIn('2 warning(s)', summary)

    def test_empty_records_still_returns_a_string(self):
        # Defensive: an empty suite still gets a meaningful summary.
        self.assertTrue(full_suite_summary_text([]))


if __name__ == '__main__':
    unittest.main()
