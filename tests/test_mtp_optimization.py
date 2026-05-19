import unittest
from dataclasses import replace

from llama_tui.benchmark import (
    _draft_mtp_session_totals,
    generate_context_moe_validation_candidates,
    mtp_long_context_probe_request_timeout,
    parse_mtp_acceptance_metrics,
)
from llama_tui.benchmark_mtp import (
    MTP_WORKLOAD_OUTPUT_CAPS,
    annotate_mtp_optimizer_records,
    best_mtp_acceptance_record,
    weighted_mtp_accept_rate,
    _record_decode_tps,
)
from llama_tui.runtime_profiles import RuntimeProfile, default_engine_capabilities


# A real llama.cpp MTP statistics block: counters are cumulative within a
# server session and reset (#gen drafts drops) when a new server starts.
_DRAFT_MTP_LOG = (
    'statistics draft-mtp: #calls(b,g,a) = 2 146 146, #gen drafts = 146, '
    '#acc drafts = 122, #gen tokens = 146, #acc tokens = 122\n'
    'statistics draft-mtp: #calls(b,g,a) = 4 369 369, #gen drafts = 369, '
    '#acc drafts = 313, #gen tokens = 369, #acc tokens = 313\n'
    'statistics draft-mtp: #calls(b,g,a) = 1 7 7, #gen drafts = 7, '
    '#acc drafts = 7, #gen tokens = 7, #acc tokens = 7\n'
)


class Fix1TimeoutPolicyTests(unittest.TestCase):
    def test_long_context_probe_timeout_caps_raised(self):
        self.assertEqual(mtp_long_context_probe_request_timeout(131072), 180)
        self.assertEqual(mtp_long_context_probe_request_timeout(65536), 180)
        self.assertEqual(mtp_long_context_probe_request_timeout(32768), 120)
        self.assertEqual(mtp_long_context_probe_request_timeout(8192), 75)

    def test_acceptance_probe_output_caps_lowered(self):
        self.assertEqual(MTP_WORKLOAD_OUTPUT_CAPS['decode_heavy']['full'], 160)
        self.assertEqual(MTP_WORKLOAD_OUTPUT_CAPS['prompt_heavy']['full'], 112)


class Fix2AcceptanceParsingTests(unittest.TestCase):
    def test_session_totals_split_on_counter_reset(self):
        sessions = _draft_mtp_session_totals(_DRAFT_MTP_LOG)
        self.assertEqual(sessions, [(369, 313), (7, 7)])

    def test_parse_uses_most_recent_session(self):
        parsed = parse_mtp_acceptance_metrics(_DRAFT_MTP_LOG)
        self.assertEqual(parsed['draft_tokens'], 7)
        self.assertEqual(parsed['accepted_tokens'], 7)
        self.assertEqual(parsed['mtp_acceptance_sessions'], 2)

    def test_parse_single_session_real_format(self):
        parsed = parse_mtp_acceptance_metrics(
            'statistics draft-mtp: #gen drafts = 1153, #acc drafts = 906\n'
        )
        self.assertEqual(parsed['draft_tokens'], 1153)
        self.assertEqual(parsed['accept_rate'], round(906 / 1153, 4))

    def test_weighted_accept_rate_prefers_reliable_samples(self):
        rate, total, samples, reliability = weighted_mtp_accept_rate([
            {'mtp_enabled': True, 'draft_tokens': 1153, 'accepted_tokens': 906},
            {'mtp_enabled': True, 'draft_tokens': 7, 'accepted_tokens': 7},
        ])
        self.assertEqual(samples, 1)
        self.assertEqual(total, 1153)
        self.assertEqual(rate, round(906 / 1153, 4))
        self.assertEqual(reliability, 'reliable')

    def test_weighted_accept_rate_falls_back_to_sparse_samples(self):
        rate, total, samples, reliability = weighted_mtp_accept_rate([
            {'mtp_enabled': True, 'draft_tokens': 7, 'accepted_tokens': 7},
        ])
        self.assertEqual(samples, 1)
        self.assertEqual(total, 7)
        self.assertEqual(rate, 1.0)
        self.assertEqual(reliability, 'sparse')

    def test_weighted_accept_rate_none_when_no_samples(self):
        rate, total, samples, reliability = weighted_mtp_accept_rate([
            {'mtp_enabled': False, 'draft_tokens': 100, 'accepted_tokens': 80},
            {'mtp_enabled': True, 'draft_tokens': 0, 'accepted_tokens': 0},
        ])
        self.assertEqual((rate, total, samples), (0.0, 0, 0))
        self.assertEqual(reliability, 'none')


class Fix4PartialWinnerTests(unittest.TestCase):
    def test_ok_record_beats_tiny_partial(self):
        best = best_mtp_acceptance_record([
            {'status': 'ok', 'mtp_enabled': True, 'accept_rate': 0.82,
             'draft_tokens': 1153, 'tokens_per_sec': 29.0},
            {'status': 'partial', 'mtp_enabled': True, 'accept_rate': 1.0,
             'draft_tokens': 7, 'tokens_per_sec': 31.0},
        ])
        self.assertEqual(best['accept_rate'], 0.82)

    def test_partial_with_metrics_is_eligible_when_only_partials(self):
        best = best_mtp_acceptance_record([
            {'status': 'partial', 'mtp_enabled': True, 'accept_rate': 0.81,
             'draft_tokens': 369, 'tokens_per_sec': 28.0, 'mtp_draft_n_max': 1},
        ])
        self.assertTrue(best)
        self.assertEqual(best['accept_rate'], 0.81)

    def test_failed_risk_level_still_excluded(self):
        self.assertEqual(
            best_mtp_acceptance_record([
                {'status': 'partial', 'mtp_enabled': True, 'accept_rate': 0.9,
                 'draft_tokens': 369, 'mtp_risk_level': 'failed'},
            ]),
            {},
        )


class Fix3ContextValidationNglTests(unittest.TestCase):
    def _caps(self):
        return replace(
            default_engine_capabilities('llama.cpp'),
            supports_cpu_moe=True,
            supports_n_cpu_moe=True,
        )

    def _baseline(self):
        return RuntimeProfile('llama.cpp', 8192, None, 1, fit=True)

    def test_fit_winner_keeps_context_validation_fit_assisted(self):
        winner = {
            'cpu_moe': True, 'n_cpu_moe': 0, 'fit': True, 'ngl': 999,
            'gpu_layers_mode': 'fit', 'moe_placement_mode': 'fit_only',
        }
        candidates = generate_context_moe_validation_candidates(
            self._baseline(), winner, self._caps(), 41, 131072, 'long_context', 'full',
        )
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertIsNone(candidate.runtime_profile.gpu_layers)
            self.assertTrue(candidate.runtime_profile.fit)

    def test_non_fit_winner_keeps_fixed_ngl(self):
        winner = {'cpu_moe': True, 'n_cpu_moe': 0, 'fit': False, 'ngl': 40}
        candidates = generate_context_moe_validation_candidates(
            self._baseline(), winner, self._caps(), 41, 131072, 'long_context', 'full',
        )
        self.assertTrue(candidates)
        self.assertTrue(
            any(candidate.runtime_profile.gpu_layers == 40 for candidate in candidates)
        )
        self.assertFalse(
            any(candidate.runtime_profile.gpu_layers == 999 for candidate in candidates)
        )


class Fix5ServerDecodeTests(unittest.TestCase):
    def test_record_decode_tps_prefers_server_truth(self):
        self.assertEqual(
            _record_decode_tps({
                'server_decode_tokens_per_sec': 29.0,
                'decode_tokens_per_sec': 11.5,
                'tokens_per_sec': 11.5,
            }),
            29.0,
        )
        self.assertEqual(
            _record_decode_tps({'decode_tokens_per_sec': 18.0, 'tokens_per_sec': 9.0}),
            18.0,
        )
        self.assertEqual(_record_decode_tps({'tokens_per_sec': 7.0}), 7.0)

    def test_decode_gain_uses_server_truth(self):
        records = [
            {'status': 'ok', 'mtp_enabled': False, 'benchmark_phase': 'baseline_no_mtp',
             'tokens_per_sec': 10.0, 'server_decode_tokens_per_sec': 20.0},
            {'status': 'ok', 'mtp_enabled': True, 'benchmark_phase': 'fit_q8_draft_n1',
             'tokens_per_sec': 12.0, 'server_decode_tokens_per_sec': 30.0,
             'accept_rate': 0.82, 'draft_tokens': 369},
        ]
        annotated = annotate_mtp_optimizer_records(records, 'draft-mtp')
        mtp_record = next(item for item in annotated if item.get('mtp_enabled'))
        # 30/20 server-truth = 1.5, not the conflated 12/10 = 1.2.
        self.assertEqual(mtp_record['decode_gain_vs_baseline'], 1.5)


if __name__ == '__main__':
    unittest.main()
