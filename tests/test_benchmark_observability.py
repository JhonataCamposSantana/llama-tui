import types
import unittest
import urllib.error
from unittest.mock import patch

from llama_tui.benchmark import percentile_summary
from llama_tui.hardware import probe_nvidia_gpu_thermal
from llama_tui.models import ModelConfig
from llama_tui.server_metrics import (
    engine_supports_metrics,
    parse_prometheus_metrics,
    reset_scrape_error_log,
    scrape_llama_server_metrics,
)
from llama_tui.ui import (
    new_benchmark_run_state,
    reduce_benchmark_event,
    refresh_benchmark_live,
)
from llama_tui.ui_benchmark import (
    benchmark_leaderboard_lines,
    build_benchmark_cockpit_items,
    build_engine_leaderboard,
)
from llama_tui.ui_components import gauge_bar, sparkline


class _FakeResponse:
    def __init__(self, body: str, status: int = 200):
        self._body = body
        self.status = status

    def read(self):
        return self._body.encode('utf-8')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _model(store=None):
    return ModelConfig(
        id='m',
        name='Model',
        path='/models/model.gguf',
        alias='m',
        engine_benchmark_store=dict(store or {}),
    )


class ServerMetricsTests(unittest.TestCase):
    def test_engine_supports_metrics(self):
        self.assertTrue(engine_supports_metrics('llama.cpp'))
        self.assertTrue(engine_supports_metrics('llama.cpp-mtp'))
        self.assertTrue(engine_supports_metrics('TURBOQUANT'))
        self.assertFalse(engine_supports_metrics('vllm'))
        self.assertFalse(engine_supports_metrics(''))

    def test_parse_prometheus_metrics_keeps_known_drops_labels_and_comments(self):
        text = (
            '# HELP llamacpp:predicted_tokens_seconds rate\n'
            '# TYPE llamacpp:predicted_tokens_seconds gauge\n'
            'llamacpp:predicted_tokens_seconds 42.5\n'
            'llamacpp:prompt_tokens_seconds 310.0\n'
            'llamacpp:kv_cache_usage_ratio{slot="0"} 0.5\n'
            'unrelated_metric 99\n'
        )
        parsed = parse_prometheus_metrics(text)
        self.assertEqual(parsed['decode_tokens_per_sec'], 42.5)
        self.assertEqual(parsed['prompt_tokens_per_sec'], 310.0)
        self.assertEqual(parsed['kv_cache_usage_ratio'], 0.5)
        self.assertNotIn('unrelated_metric', parsed)

    def test_scrape_returns_none_for_invalid_port(self):
        self.assertIsNone(scrape_llama_server_metrics('127.0.0.1', 0))
        self.assertIsNone(scrape_llama_server_metrics('127.0.0.1', 'bad'))

    def test_scrape_parses_endpoint_body(self):
        body = 'llamacpp:predicted_tokens_seconds 27.0\n'
        with patch('llama_tui.server_metrics.urllib.request.urlopen', return_value=_FakeResponse(body)):
            result = scrape_llama_server_metrics('0.0.0.0', 8080)
        self.assertEqual(result, {'decode_tokens_per_sec': 27.0})

    def test_scrape_degrades_gracefully_on_url_error(self):
        with patch(
            'llama_tui.server_metrics.urllib.request.urlopen',
            side_effect=urllib.error.URLError('refused'),
        ):
            self.assertIsNone(scrape_llama_server_metrics('127.0.0.1', 8080))

    def test_scrape_error_callback_fires_once_per_host_port_error_class(self):
        reset_scrape_error_log()
        seen = []

        def on_error(cls, message):
            seen.append((cls, message))

        with patch(
            'llama_tui.server_metrics.urllib.request.urlopen',
            side_effect=urllib.error.URLError('refused'),
        ):
            scrape_llama_server_metrics('127.0.0.1', 8181, on_error=on_error)
            scrape_llama_server_metrics('127.0.0.1', 8181, on_error=on_error)
            scrape_llama_server_metrics('127.0.0.1', 8182, on_error=on_error)
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][0], 'URLError')


class BenchmarkBudgetEnvOverrideTests(unittest.TestCase):
    def test_env_int_override_replaces_default_and_clamps_minimum(self):
        from llama_tui.benchmark import _env_int_override
        import os
        prev = os.environ.get('LLAMA_TUI_FAKE_BUDGET')
        try:
            os.environ['LLAMA_TUI_FAKE_BUDGET'] = '120'
            self.assertEqual(_env_int_override('LLAMA_TUI_FAKE_BUDGET', 600, minimum=60), 120)
            os.environ['LLAMA_TUI_FAKE_BUDGET'] = '30'
            self.assertEqual(_env_int_override('LLAMA_TUI_FAKE_BUDGET', 600, minimum=60), 60)
            os.environ['LLAMA_TUI_FAKE_BUDGET'] = ''
            self.assertEqual(_env_int_override('LLAMA_TUI_FAKE_BUDGET', 600, minimum=60), 600)
            os.environ['LLAMA_TUI_FAKE_BUDGET'] = 'forever'
            self.assertEqual(_env_int_override('LLAMA_TUI_FAKE_BUDGET', 600, minimum=60), 600)
        finally:
            if prev is None:
                os.environ.pop('LLAMA_TUI_FAKE_BUDGET', None)
            else:
                os.environ['LLAMA_TUI_FAKE_BUDGET'] = prev


class ThermalProbeTests(unittest.TestCase):
    def test_returns_zero_when_nvidia_smi_missing(self):
        with patch('llama_tui.hardware.shutil.which', return_value=None):
            self.assertEqual(probe_nvidia_gpu_thermal(), (0, False))

    def test_parses_temperature_without_throttle(self):
        fake = types.SimpleNamespace(returncode=0, stdout='71, Not Active, Not Active\n')
        with patch('llama_tui.hardware.shutil.which', return_value='/usr/bin/nvidia-smi'), \
                patch('llama_tui.hardware.subprocess.run', return_value=fake):
            self.assertEqual(probe_nvidia_gpu_thermal(), (71, False))

    def test_detects_active_thermal_throttle(self):
        fake = types.SimpleNamespace(returncode=0, stdout='86, Active, Not Active\n')
        with patch('llama_tui.hardware.shutil.which', return_value='/usr/bin/nvidia-smi'), \
                patch('llama_tui.hardware.subprocess.run', return_value=fake):
            self.assertEqual(probe_nvidia_gpu_thermal(), (86, True))

    def test_returns_zero_when_query_fails(self):
        fake = types.SimpleNamespace(returncode=2, stdout='')
        with patch('llama_tui.hardware.shutil.which', return_value='/usr/bin/nvidia-smi'), \
                patch('llama_tui.hardware.subprocess.run', return_value=fake):
            self.assertEqual(probe_nvidia_gpu_thermal(), (0, False))


class PercentileSummaryTests(unittest.TestCase):
    def test_empty_sample(self):
        self.assertEqual(
            percentile_summary([]),
            {'min': 0.0, 'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0},
        )

    def test_single_sample_collapses_to_value(self):
        summary = percentile_summary([12.5])
        self.assertEqual(set(summary.values()), {12.5})

    def test_ordered_percentiles(self):
        summary = percentile_summary([30.0, 10.0, 20.0])
        self.assertEqual(summary['min'], 10.0)
        self.assertEqual(summary['p50'], 20.0)
        self.assertEqual(summary['max'], 30.0)


class SparklineGaugeTests(unittest.TestCase):
    def test_sparkline_empty_pads_to_width(self):
        self.assertEqual(sparkline([], 5), '     ')
        self.assertEqual(sparkline([1, 2, 3], 0), '')

    def test_sparkline_flat_series_is_uniform_band(self):
        result = sparkline([5, 5, 5], 10)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(set(result)), 1)

    def test_sparkline_ascending_uses_full_range(self):
        result = sparkline([1, 2, 3, 4, 5, 6, 7, 8], 8)
        self.assertEqual(result[0], '▁')
        self.assertEqual(result[-1], '█')

    def test_gauge_bar_fill_and_clamp(self):
        self.assertEqual(gauge_bar(0.5, 10), '[' + '|' * 5 + ' ' * 5 + ']')
        self.assertEqual(gauge_bar(0.0, 4), '[    ]')
        self.assertEqual(gauge_bar(2.0, 4), '[||||]')
        self.assertEqual(gauge_bar(-1.0, 4), '[    ]')
        self.assertEqual(gauge_bar(0.5, 0), '')


class EngineLeaderboardTests(unittest.TestCase):
    def _store(self):
        return {
            'llama.cpp': {
                'last_benchmark_tokens_per_sec': 20.0,
                'measured_profiles': {
                    'auto': {'tokens_per_sec': 20.0, 'ctx': 32768, 'peak_vram_used': 5 * 1024 ** 3},
                },
            },
            'llama.cpp-mtp': {
                'last_benchmark_tokens_per_sec': 31.25,
                'measured_profiles': {
                    'auto': {'tokens_per_sec': 31.25, 'ctx': 131072, 'peak_vram_used': 7 * 1024 ** 3},
                    'mtp_acceptance': {'accept_rate': 0.767},
                },
            },
            'vllm': {},
        }

    def test_leaderboard_sorts_fastest_first_and_data_before_empty(self):
        rows = build_engine_leaderboard(_model(self._store()), 'tps')
        self.assertEqual(rows[0]['engine'], 'llama.cpp-mtp')
        self.assertEqual(rows[1]['engine'], 'llama.cpp')
        self.assertFalse(rows[-1]['has_data'])
        self.assertEqual(rows[-1]['engine'], 'vllm')

    def test_leaderboard_lines_mark_winner_and_have_heading(self):
        lines = benchmark_leaderboard_lines(_model(self._store()))
        self.assertEqual(lines[0][1], 'heading')
        self.assertEqual(lines[1][1], 'success')
        self.assertIn('llama.cpp-mtp', lines[1][0])

    def test_leaderboard_lines_empty_store(self):
        self.assertEqual(
            benchmark_leaderboard_lines(_model()),
            [('no per-engine benchmark data yet', 'muted')],
        )


class CockpitTests(unittest.TestCase):
    def test_empty_live_waits_for_samples(self):
        self.assertEqual(
            build_benchmark_cockpit_items(new_benchmark_run_state()),
            [('waiting for live samples...', 'muted')],
        )

    def test_cockpit_renders_tps_and_throttle(self):
        state = {
            'live': {
                'tps': [20.0, 25.0, 31.25],
                'vram_used': [7 * 1024 ** 3],
                'vram_total': [8 * 1024 ** 3],
                'gpu_temp': [88],
                'thermal_throttled': True,
            }
        }
        items = build_benchmark_cockpit_items(state, width=80)
        self.assertIn('tok/s', items[0][0])
        temp_line = next(text for text, _kind in items if 'temp' in text)
        self.assertIn('THROTTLE', temp_line)
        self.assertTrue(any(kind == 'error' for _text, kind in items))


class BenchmarkLiveStateTests(unittest.TestCase):
    def _record(self, tps=31.25, throttled=False):
        return {
            'tokens_per_sec': tps,
            'gpu_memory_total': 8 * 1024 ** 3,
            'peak_vram_used': 6 * 1024 ** 3,
            'gpu_temp_peak': 78,
            'thermal_throttled': throttled,
        }

    def test_new_state_has_empty_live_buffers(self):
        live = new_benchmark_run_state()['live']
        self.assertEqual(live['tps'], [])
        self.assertEqual(live['vram_used'], [])
        self.assertFalse(live['thermal_throttled'])

    def test_reduce_populates_live_buffers_from_result(self):
        state = {}
        reduce_benchmark_event(state, {'event': 'benchmark_started', 'model_id': 'm'})
        reduce_benchmark_event(state, {'event': 'benchmark_result', 'record': self._record()})
        live = state['live']
        self.assertEqual(live['tps'], [31.25])
        self.assertEqual(live['vram_used'], [6 * 1024 ** 3])
        self.assertEqual(live['vram_total'], [8 * 1024 ** 3])
        self.assertEqual(live['gpu_temp'], [78])

    def test_failed_candidate_adds_no_tps_sample(self):
        state = {}
        reduce_benchmark_event(state, {'event': 'benchmark_started', 'model_id': 'm'})
        reduce_benchmark_event(state, {'event': 'benchmark_result', 'record': self._record(tps=0.0)})
        self.assertEqual(state['live']['tps'], [])

    def test_refresh_is_idempotent(self):
        state = {'records': [self._record(), self._record(tps=26.9, throttled=True)]}
        refresh_benchmark_live(state)
        first = dict(state['live'])
        refresh_benchmark_live(state)
        self.assertEqual(state['live'], first)
        self.assertEqual(state['live']['tps'], [31.25, 26.9])
        self.assertTrue(state['live']['thermal_throttled'])


if __name__ == '__main__':
    unittest.main()
