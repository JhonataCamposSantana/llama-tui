import curses
import tempfile
import threading
import time
import unittest
from pathlib import Path

from llama_tui.app import AppConfig
from llama_tui.control import CancelToken
from llama_tui.models import ModelConfig
from llama_tui.ui import shutdown_workers
from llama_tui.ui_action_runner import ActionRunner
from llama_tui.ui_benchmark import benchmark_plan_summary_lines
from llama_tui.ui_components import (
    kind_status_prefix,
    kind_status_symbol,
    truncate,
    wrap_card_lines,
)
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


class KindStatusSymbolTests(unittest.TestCase):
    def test_symbol_per_known_kind(self):
        self.assertEqual(kind_status_symbol('success'), '✓')
        self.assertEqual(kind_status_symbol('error'), '✗')
        self.assertEqual(kind_status_symbol('warning'), '⚠')
        self.assertEqual(kind_status_symbol('muted'), '·')

    def test_no_symbol_for_heading_or_normal(self):
        self.assertEqual(kind_status_symbol('heading'), '')
        self.assertEqual(kind_status_symbol('normal'), '')
        self.assertEqual(kind_status_symbol(''), '')
        self.assertEqual(kind_status_symbol(None), '')

    def test_prefix_prepends_glyph_and_space(self):
        self.assertEqual(kind_status_prefix('--spec-type included: yes', 'success'), '✓ --spec-type included: yes')
        self.assertEqual(kind_status_prefix('--spec-type included: no', 'error'), '✗ --spec-type included: no')

    def test_prefix_passes_through_for_headings(self):
        self.assertEqual(kind_status_prefix('MTP Capability Probe', 'heading'), 'MTP Capability Probe')
        self.assertEqual(kind_status_prefix('plain text', 'normal'), 'plain text')


class ShutdownWorkersTests(unittest.TestCase):
    def test_cancels_tokens_and_joins_live_threads(self):
        token = CancelToken()
        stopper = threading.Event()

        def runner():
            while not stopper.is_set() and not token.is_cancelled():
                time.sleep(0.01)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        try:
            shutdown_workers(token, thread, join_timeout=1.0)
            self.assertTrue(token.is_cancelled())
            self.assertFalse(thread.is_alive())
        finally:
            stopper.set()
            thread.join(timeout=1.0)

    def test_none_entries_are_tolerated(self):
        # Should not raise when no tokens or threads are active.
        shutdown_workers(None, None, join_timeout=0.1)


class ActionRunnerTests(unittest.TestCase):
    def test_default_is_idle(self):
        runner = ActionRunner()
        self.assertFalse(runner.is_running())
        self.assertIsNone(runner.thread)
        self.assertIsNone(runner.token)

    def test_is_running_tracks_thread_lifecycle(self):
        runner = ActionRunner()
        stopper = threading.Event()

        def loop():
            while not stopper.is_set():
                time.sleep(0.01)

        runner.token = CancelToken()
        runner.thread = threading.Thread(target=loop, daemon=True)
        runner.thread.start()
        try:
            self.assertTrue(runner.is_running())
        finally:
            stopper.set()
            runner.thread.join(timeout=1.0)
        self.assertFalse(runner.is_running())

    def test_cancel_is_safe_when_token_none(self):
        # No exception, no side effect — used on the shutdown path.
        ActionRunner().cancel('shutdown')

    def test_cancel_calls_token(self):
        runner = ActionRunner(token=CancelToken())
        runner.cancel('user requested abort')
        self.assertTrue(runner.token.is_cancelled())

    def test_reset_clears_both_slots(self):
        runner = ActionRunner(thread=threading.Thread(target=lambda: None), token=CancelToken())
        runner.reset()
        self.assertIsNone(runner.thread)
        self.assertIsNone(runner.token)


if __name__ == '__main__':
    unittest.main()
