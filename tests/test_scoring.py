"""Unit tests for the per-objective benchmark scorers.

Covers the module extracted in audit #6 step 4. Pure functions over
measurement-record dicts plus a ``ModelConfig``, so the tests use
hand-built records that exercise the curves and tradeoff weights
directly.
"""

import unittest

from llama_tui.models import ModelConfig
from llama_tui.scoring import (
    _ctx_curve,
    _record_headroom_score,
    _record_kv_quality_penalty,
    _record_stability_score,
    _tps_curve,
    score_auto,
    score_fast_chat,
    score_long_context,
    score_opencode_ready,
)


def _model(**overrides) -> ModelConfig:
    return ModelConfig(id='m', name='M', path='/m.gguf', alias='m', port=18080, **overrides)


def _dense() -> ModelConfig:
    return _model(architecture_type='dense')


def _moe() -> ModelConfig:
    return _model(architecture_type='moe')


class HeadroomScoreTests(unittest.TestCase):
    def test_zero_ram_falls_back_to_baseline(self):
        # No ram_available info → baseline 0.35.
        self.assertAlmostEqual(_record_headroom_score({}), 0.35)

    def test_ram_only_caps_at_one(self):
        score = _record_headroom_score({'ram_available': 64 * 1024**3})
        self.assertEqual(score, 1.0)

    def test_vram_lifts_score_when_higher_than_ram(self):
        # 8 GiB VRAM dominates a low-RAM record because VRAM caps
        # at 2 GiB → 1.0.
        score = _record_headroom_score({'ram_available': 0, 'gpu_memory_free': 8 * 1024**3})
        self.assertEqual(score, 1.0)

    def test_pressure_dampens_score(self):
        base = _record_headroom_score({'ram_available': 64 * 1024**3})
        stressed = _record_headroom_score({
            'ram_available': 64 * 1024**3,
            'process_pressure_score': 1.0,
        })
        self.assertLess(stressed, base)
        # Pressure can't push the score below 0.45 of itself.
        self.assertGreaterEqual(stressed, base * 0.45 - 1e-9)


class StabilityScoreTests(unittest.TestCase):
    def test_clean_ok_record_is_full_score(self):
        self.assertEqual(_record_stability_score({'status': 'ok'}), 1.0)

    def test_retry_attempt_penalises(self):
        self.assertLess(_record_stability_score({'status': 'ok', 'retry_attempt': 2}), 1.0)

    def test_non_ok_status_heavy_penalty(self):
        score = _record_stability_score({'status': 'failed'})
        self.assertLess(score, 0.7)

    def test_slow_ready_seconds_penalises(self):
        slow = _record_stability_score({'status': 'ok', 'ready_seconds': 120.0})
        self.assertLess(slow, 1.0)


class CurveTests(unittest.TestCase):
    def test_tps_curve_zero_for_no_tps(self):
        self.assertEqual(_tps_curve({}), 0.0)

    def test_tps_curve_monotonic_and_bounded(self):
        low = _tps_curve({'tokens_per_sec': 10.0})
        high = _tps_curve({'tokens_per_sec': 100.0})
        self.assertLess(low, high)
        self.assertLess(high, 1.0)

    def test_tps_curve_prefers_server_truth_decode(self):
        # decode_tokens_per_sec (server-truth) overrides client-side
        # tokens_per_sec when present.
        self.assertEqual(
            _tps_curve({'decode_tokens_per_sec': 30.0, 'tokens_per_sec': 5.0}),
            _tps_curve({'tokens_per_sec': 30.0}),
        )

    def test_ctx_curve_normalises_to_cap(self):
        self.assertEqual(_ctx_curve({'ctx': 8192}, 8192), 1.0)
        self.assertEqual(_ctx_curve({'ctx': 4096}, 8192), 0.5)
        self.assertEqual(_ctx_curve({'ctx': 0}, 8192), 0.0)

    def test_ctx_curve_prefers_ctx_per_slot(self):
        self.assertEqual(
            _ctx_curve({'ctx_per_slot': 4096, 'ctx': 999999}, 8192),
            0.5,
        )


class KvQualityPenaltyTests(unittest.TestCase):
    def test_explicit_penalty_passes_through(self):
        self.assertEqual(_record_kv_quality_penalty({'kv_score_penalty': 0.3}), 0.3)

    def test_negative_explicit_penalty_clamped_to_zero(self):
        self.assertEqual(_record_kv_quality_penalty({'kv_score_penalty': -1.0}), 0.0)

    def test_unknown_kv_preset_has_no_penalty(self):
        self.assertEqual(_record_kv_quality_penalty({'kv_preset': 'totally_made_up'}), 0.0)


class ScoreFastChatTests(unittest.TestCase):
    def test_tps_dominates_fast_chat(self):
        a = score_fast_chat({'tokens_per_sec': 50.0, 'status': 'ok'}, _dense())
        b = score_fast_chat({'tokens_per_sec': 100.0, 'status': 'ok'}, _dense())
        self.assertGreater(b, a)

    def test_moe_gets_higher_cap_for_ctx(self):
        record = {'tokens_per_sec': 50.0, 'status': 'ok', 'ctx_per_slot': 16384}
        moe_score = score_fast_chat(record, _moe())
        dense_score = score_fast_chat(record, _dense())
        # MoE doubles the ctx cap (16k vs 8k) so a 16k-ctx record gets
        # full ctx-curve credit; dense already capped at 8k so its ctx
        # contribution saturates at the same value. The non-ctx terms
        # are identical, so MoE >= dense.
        self.assertGreaterEqual(moe_score, dense_score)


class ScoreLongContextTests(unittest.TestCase):
    def test_long_context_prefers_higher_ctx(self):
        a = score_long_context({'ctx': 8192, 'tokens_per_sec': 20.0, 'status': 'ok'}, _dense())
        b = score_long_context({'ctx': 65536, 'tokens_per_sec': 20.0, 'status': 'ok'}, _dense())
        self.assertGreater(b, a)


class ScoreOpencodeReadyTests(unittest.TestCase):
    def test_moe_below_16k_ctx_penalised(self):
        # MoE models below 16k ctx get their score multiplied by 0.35.
        score_low = score_opencode_ready({'ctx': 8192, 'tokens_per_sec': 50.0, 'status': 'ok'}, _moe())
        score_mid = score_opencode_ready({'ctx': 16384, 'tokens_per_sec': 50.0, 'status': 'ok'}, _moe())
        self.assertLess(score_low, score_mid * 0.5)

    def test_moe_above_32k_ctx_gets_bonus(self):
        score_mid = score_opencode_ready({'ctx': 16384, 'tokens_per_sec': 50.0, 'status': 'ok'}, _moe())
        score_high = score_opencode_ready({'ctx': 32768, 'tokens_per_sec': 50.0, 'status': 'ok'}, _moe())
        self.assertGreater(score_high, score_mid)


class ScoreAutoTests(unittest.TestCase):
    def test_auto_blends_sub_scores_for_moe_partial_ngl(self):
        # MoE with non-full GPU layers triggers the blended path.
        score = score_auto({
            'ctx': 32768, 'tokens_per_sec': 40.0, 'status': 'ok', 'ngl': 20,
        }, _moe())
        self.assertGreater(score, 0.0)

    def test_auto_uses_simple_blend_for_dense_or_full_ngl(self):
        score = score_auto({
            'ctx': 32768, 'tokens_per_sec': 40.0, 'status': 'ok', 'ngl': 999,
        }, _dense())
        self.assertGreater(score, 0.0)


if __name__ == '__main__':
    unittest.main()
