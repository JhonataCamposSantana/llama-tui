import curses
import tempfile
import unittest
from pathlib import Path

from llama_tui.app import AppConfig
from llama_tui.models import ModelConfig
from llama_tui.ui_benchmark import benchmark_plan_summary_lines
from llama_tui.ui_components import truncate, wrap_card_lines
from llama_tui.ui_models import (
    build_model_row_summary,
    format_engine_badge,
    format_model_state,
    mtp_status_from_measured,
    mtp_status_short,
    status_symbol,
)
from llama_tui.ui_theme import (
    health_style_name,
    kind_style,
    mtp_style_name,
    state_chip_style,
    style,
)


FAKE_COLORS = {
    'default': 0, 'accent': 1, 'success': 2, 'warning': 3, 'error': 4,
    'muted': 5, 'selection': 6, 'banner': 7, 'panel': 8,
    'chip_ready': 9, 'chip_loading': 10, 'chip_stopped': 11,
}


class UiThemeTests(unittest.TestCase):
    def test_semantic_style_maps_to_palette_key(self):
        self.assertEqual(style(FAKE_COLORS, 'muted') & ~curses.A_BOLD, FAKE_COLORS['muted'])
        self.assertEqual(style(FAKE_COLORS, 'warning') & ~curses.A_BOLD, FAKE_COLORS['warning'])
        # success/error/active are bold by default
        self.assertTrue(style(FAKE_COLORS, 'success') & curses.A_BOLD)

    def test_health_style_name_mapping(self):
        self.assertEqual(health_style_name('OK'), 'success')
        self.assertEqual(health_style_name('STALE'), 'warning')
        self.assertEqual(health_style_name('WARN'), 'warning')
        self.assertEqual(health_style_name('FAIL'), 'error')
        self.assertEqual(health_style_name('???'), 'muted')

    def test_mtp_style_name_mapping(self):
        self.assertEqual(mtp_style_name('ready'), 'success')
        self.assertEqual(mtp_style_name('usable'), 'success')
        self.assertEqual(mtp_style_name('capable'), 'active')
        self.assertEqual(mtp_style_name('risky'), 'warning')
        self.assertEqual(mtp_style_name('blocked'), 'error')
        self.assertEqual(mtp_style_name('unsupported'), 'muted')
        self.assertEqual(mtp_style_name('off'), 'muted')

    def test_status_chip_style_for_run_states(self):
        self.assertEqual(state_chip_style(FAKE_COLORS, 'running'), FAKE_COLORS['chip_ready'])
        self.assertEqual(state_chip_style(FAKE_COLORS, 'starting'), FAKE_COLORS['chip_loading'])
        self.assertEqual(state_chip_style(FAKE_COLORS, 'stopped'), FAKE_COLORS['chip_stopped'])
        # unknown state falls back to the stopped chip
        self.assertEqual(state_chip_style(FAKE_COLORS, 'bogus'), FAKE_COLORS['chip_stopped'])

    def test_kind_style_heading_is_bold_accent(self):
        self.assertTrue(kind_style(FAKE_COLORS, 'heading') & curses.A_BOLD)


class UiComponentsTests(unittest.TestCase):
    def test_truncate_hard_cuts(self):
        self.assertEqual(truncate('hello world', 5), 'hello')
        self.assertEqual(truncate('hi', 0), '')
        self.assertEqual(truncate('hi', 10), 'hi')

    def test_wrap_card_lines_wraps_to_width(self):
        wrapped = wrap_card_lines(['abcdefgh'], 3)
        self.assertEqual(wrapped, [('abc', 'normal'), ('def', 'normal'), ('gh', 'normal')])

    def test_wrap_card_lines_preserves_kind_and_short_lines(self):
        wrapped = wrap_card_lines([('short', 'warning'), 'plain'], 20)
        self.assertEqual(wrapped, [('short', 'warning'), ('plain', 'normal')])

    def test_wrap_card_lines_keeps_blank_lines(self):
        self.assertEqual(wrap_card_lines([''], 10), [('', 'normal')])


class UiModelFormatTests(unittest.TestCase):
    def test_format_engine_badge(self):
        self.assertEqual(format_engine_badge('llama.cpp'), 'llama.cpp')
        self.assertEqual(format_engine_badge('llama.cpp-mtp'), 'llama.cpp MTP')
        self.assertEqual(format_engine_badge('llama.cpp-mtp', narrow=True), 'MTP')
        self.assertEqual(format_engine_badge('turboquant', narrow=True), 'TQ')

    def test_format_model_state_and_status_symbol(self):
        self.assertEqual(format_model_state('READY'), 'running')
        self.assertEqual(format_model_state('STARTING'), 'starting')
        self.assertEqual(format_model_state('STOPPED'), 'stopped')
        self.assertEqual(status_symbol('READY'), '●')
        self.assertEqual(status_symbol('STOPPED'), '○')

    def test_mtp_status_from_measured_mapping(self):
        self.assertEqual(mtp_status_from_measured({'status': 'ok', 'mtp_risk_level': 'good'}), 'usable')
        self.assertEqual(mtp_status_from_measured({'status': 'ok', 'mtp_risk_level': 'risky'}), 'risky')
        self.assertEqual(mtp_status_from_measured({'status': 'failed'}), 'blocked')
        self.assertEqual(mtp_status_from_measured({}), '')


class UiModelRowTests(unittest.TestCase):
    def _app_model(self, **overrides):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        app = AppConfig(Path(tmp.name) / 'models.json')
        fields = dict(id='m', name='Model', path='/models/m.gguf', alias='m', port=18080)
        fields.update(overrides)
        return app, ModelConfig(**fields)

    def test_build_model_row_summary_has_expected_keys(self):
        app, model = self._app_model()
        summary = build_model_row_summary(app, model, 'STOPPED')
        for key in ('display_name', 'state', 'pick', 'ctx', 'tokens_per_sec', 'engine', 'health', 'mtp'):
            self.assertIn(key, summary)
        self.assertEqual(summary['state'], 'stopped')

    def test_mtp_status_short_off_for_non_mtp_model(self):
        app, model = self._app_model(supports_mtp='auto')
        self.assertEqual(mtp_status_short(app, model), 'off')

    def test_mtp_status_short_unsupported_when_disabled(self):
        app, model = self._app_model(supports_mtp='no')
        self.assertEqual(mtp_status_short(app, model), 'unsupported')

    def test_mtp_status_short_uses_measured_risk(self):
        app, model = self._app_model(supports_mtp='yes')
        model.measured_profiles = {'mtp_acceptance': {'status': 'ok', 'mtp_risk_level': 'risky'}}
        self.assertEqual(mtp_status_short(app, model), 'risky')


class BenchmarkPlanLineTests(unittest.TestCase):
    def test_plan_summary_lines_structure(self):
        lines = benchmark_plan_summary_lines(
            engine='llama.cpp',
            binary='/opt/llama-server',
            capability_summary=['--spec-type (draft-mtp)', 'fit'],
            candidate_names=['mtp_baseline', 'mtp_fit_q8_draftq8_nommap_draft1_128k', 'context_growth_sweep_8192'],
            skipped=[('mtp_acceptance_matrix', 'binary lacks --spec-type')],
            strategy_id='mtp_acceptance_matrix',
        )
        text = '\n'.join(line for line, _kind in lines)
        self.assertEqual(lines[0], ('Benchmark Plan', 'heading'))
        self.assertIn('Engine: llama.cpp', text)
        self.assertIn('Binary: /opt/llama-server', text)
        self.assertIn('Strategy: mtp_acceptance_matrix', text)
        self.assertIn('Generated candidates: 3', text)
        self.assertIn('mtp_fit_q8_draftq8_nommap_draft1_128k', text)
        self.assertIn('Skipped', text)
        self.assertIn('binary lacks --spec-type', text)

    def test_plan_summary_lines_handle_empty_candidates(self):
        lines = benchmark_plan_summary_lines('llama.cpp', '', [], [], [], '')
        text = '\n'.join(line for line, _kind in lines)
        self.assertIn('Generated candidates: 0', text)
        self.assertIn('none detected', text)


if __name__ == '__main__':
    unittest.main()
