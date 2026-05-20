"""Unit tests for the stderr-classification helpers.

Covers the module extracted in audit #6 step 1. The big
``classify_benchmark_failure`` function is exercised heavily by
``tests/test_runtime_profiles`` via its actionable-startup-errors
suite; this file focuses on the smaller helpers and reinforces a few
category mappings directly so a future refactor (audit #11
declarative-table conversion) has a fixed ground truth.
"""

import unittest

from llama_tui.failure_classification import (
    FAILURE_CATEGORIES,
    FAILURE_EXCERPT_MARKERS,
    benchmark_failure_excerpt,
    benchmark_failure_summary,
    classify_benchmark_failure,
)


class FailureExcerptTests(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        self.assertEqual(benchmark_failure_excerpt(''), '')
        self.assertEqual(benchmark_failure_excerpt(None), '')

    def test_picks_marker_matching_lines_first(self):
        text = (
            'some preamble\n'
            'ggml_assert: condition failed at /path/llama.cpp:123\n'
            'continuing\n'
        )
        excerpt = benchmark_failure_excerpt(text)
        self.assertIn('ggml_assert', excerpt.lower())

    def test_fallback_returns_last_three_lines(self):
        text = '\n'.join(f'line {n}' for n in range(10))
        excerpt = benchmark_failure_excerpt(text)
        self.assertIn('line 9', excerpt)
        self.assertIn('line 8', excerpt)

    def test_truncates_long_excerpt(self):
        long_marker_line = 'ggml_assert: ' + 'x' * 1000
        excerpt = benchmark_failure_excerpt(long_marker_line, limit=120)
        self.assertLessEqual(len(excerpt), 120)
        self.assertTrue(excerpt.endswith('...'))


class ClassifyBenchmarkFailureTests(unittest.TestCase):
    def test_default_category_used_when_no_signals(self):
        result = classify_benchmark_failure('something inscrutable happened')
        self.assertEqual(result['failure_category'], 'SERVER_TIMEOUT')

    def test_custom_default_category_is_validated(self):
        result = classify_benchmark_failure('inscrutable', default_category='API_TIMEOUT')
        # No signal in the text → falls back to the (validated) default.
        self.assertEqual(result['failure_category'], 'API_TIMEOUT')

    def test_invalid_default_collapses_to_server_timeout(self):
        result = classify_benchmark_failure('inscrutable', default_category='MADE_UP')
        self.assertEqual(result['failure_category'], 'SERVER_TIMEOUT')

    def test_ggml_assert_is_engine_runtime_crash(self):
        text = 'ggml_assert: tensor->ne[0] == K at ggml-cuda.cu:42'
        result = classify_benchmark_failure(text)
        self.assertEqual(result['failure_category'], 'ENGINE_RUNTIME_CRASH')
        self.assertTrue(result['suggested_fix'])

    def test_cuda_oom_weights(self):
        text = 'cudaMalloc failed: out of memory while loading model tensors'
        result = classify_benchmark_failure(text)
        self.assertEqual(result['failure_category'], 'CUDA_OOM_WEIGHTS')

    def test_cuda_oom_kv(self):
        text = 'failed to allocate buffer for kv cache: out of memory'
        result = classify_benchmark_failure(text)
        self.assertEqual(result['failure_category'], 'CUDA_OOM_KV')

    def test_cli_invalid_for_unknown_arg(self):
        text = 'error: unknown argument: --bogus-flag'
        result = classify_benchmark_failure(text)
        self.assertEqual(result['failure_category'], 'CLI_INVALID')

    def test_engine_binary_missing(self):
        text = 'ENGINE_BINARY_MISSING: llama.cpp-tq3 server not found: TQ3_LLAMA_SERVER_BIN unset'
        result = classify_benchmark_failure(text)
        self.assertEqual(result['failure_category'], 'ENGINE_BINARY_MISSING')

    def test_kv_mode_incompatible_unsupported_cache_type(self):
        text = 'unsupported cache type: turbo3 (head dim does not divide)'
        result = classify_benchmark_failure(text)
        self.assertEqual(result['failure_category'], 'KV_MODE_INCOMPATIBLE')

    def test_chat_template_error(self):
        text = 'failed to apply chat template: jinja error in render'
        result = classify_benchmark_failure(text)
        self.assertEqual(result['failure_category'], 'CHAT_TEMPLATE_ERROR')

    def test_model_load_failed(self):
        text = 'failed to load model from /models/m.gguf'
        result = classify_benchmark_failure(text)
        self.assertEqual(result['failure_category'], 'MODEL_LOAD_FAILED')

    def test_port_unreachable(self):
        text = 'urllib.error.URLError: connection refused'
        result = classify_benchmark_failure(text)
        self.assertEqual(result['failure_category'], 'PORT_UNREACHABLE')

    def test_timeout_categorisation_respects_default(self):
        text = 'request timed out after 240s'
        result = classify_benchmark_failure(text, default_category='API_TIMEOUT')
        self.assertEqual(result['failure_category'], 'API_TIMEOUT')
        result = classify_benchmark_failure(text, default_category='RAW_ENGINE_TIMEOUT')
        self.assertEqual(result['failure_category'], 'RAW_ENGINE_TIMEOUT')

    def test_record_always_contains_four_keys(self):
        result = classify_benchmark_failure('anything')
        self.assertEqual(
            set(result.keys()),
            {'failure_category', 'failure_reason', 'suggested_fix', 'failure_excerpt'},
        )


class FailureSummaryTests(unittest.TestCase):
    def test_uses_fallback_when_no_records(self):
        self.assertEqual(benchmark_failure_summary([], 'nothing happened'), 'nothing happened')

    def test_uses_fallback_when_no_categorised_record(self):
        records = [{'detail': 'just noise'}]
        self.assertEqual(benchmark_failure_summary(records, 'fallback msg'), 'fallback msg')

    def test_first_categorised_record_wins(self):
        records = [
            {'failure_category': 'CUDA_OOM_KV', 'failure_reason': 'kv cache OOM'},
            {'failure_category': 'ENGINE_RUNTIME_CRASH', 'failure_reason': 'should not surface'},
        ]
        summary = benchmark_failure_summary(records, 'fallback msg')
        self.assertIn('CUDA_OOM_KV', summary)
        self.assertIn('kv cache OOM', summary)
        self.assertNotIn('should not surface', summary)


class CategoryAndMarkerInvariantsTests(unittest.TestCase):
    def test_categories_are_unique(self):
        # FAILURE_CATEGORIES is a fixed schema used in records; any
        # accidental duplicate would mask classification bugs.
        self.assertEqual(len(FAILURE_CATEGORIES), len(set(FAILURE_CATEGORIES)))

    def test_markers_are_lowercase(self):
        # The excerpt selector matches against ``line.lower()``, so an
        # accidental uppercase marker would silently fail to match.
        for marker in FAILURE_EXCERPT_MARKERS:
            self.assertEqual(marker, marker.lower(), f'{marker!r} is not lowercase')


if __name__ == '__main__':
    unittest.main()
