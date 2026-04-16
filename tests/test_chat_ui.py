import unittest
from pathlib import Path
from unittest.mock import patch

from llama_tui.chat import build_chat_payload, parse_openai_sse_lines, stream_chat_completion
from llama_tui.control import CancelToken, CancelledError
from llama_tui.models import ModelConfig
from llama_tui.ui import (
    apply_quit_policy,
    adjust_scroll_offset,
    benchmark_ranking_items,
    build_benchmark_progress_items,
    build_error_items,
    build_header_config_items,
    build_header_dashboard_items,
    build_log_items,
    header_dashboard_layout,
    header_dashboard_title,
    benchmark_elapsed_text,
    benchmark_progress_fraction,
    benchmark_record_display_items,
    benchmark_ranking_rows,
    benchmark_row_text,
    benchmark_run_line,
    benchmark_runs_for_model,
    benchmark_wiki_lines,
    benchmark_command_lines,
    build_try_live_stat_lines,
    clamp_scroll,
    finish_try_live_metrics,
    new_try_live_metrics,
    new_benchmark_run_state,
    profile_label,
    progress_bar_text,
    reduce_benchmark_event,
    reset_try_live_metrics,
    cycle_right_tab,
    default_right_tab,
    normalize_right_tab,
    right_tab_label,
    right_tab_scroll_key,
    right_tabs_for_view,
    should_prompt_quit_keepalive,
    simple_profile_action,
    scrollable_pane_max_scroll,
    scrollable_pane_view,
    scrollable_pane_wrapped_lines,
    stop_try_model,
    try_live_metric_snapshot,
    try_input_max_scroll,
    try_input_view,
    try_input_wrapped_lines,
    update_try_live_metrics,
    wrap_display_item_lines,
)


class ChatPayloadTests(unittest.TestCase):
    def test_chat_payload_uses_selected_model_settings(self):
        model = ModelConfig(
            id='tiny',
            name='Tiny',
            path='tiny.gguf',
            alias='tiny-local',
            port=18080,
            temp=0.42,
            output=321,
        )
        messages = [{'role': 'user', 'content': 'hello'}]
        payload = build_chat_payload(model, messages, stream=True)

        self.assertEqual(payload['model'], 'tiny-local')
        self.assertEqual(payload['messages'], messages)
        self.assertEqual(payload['temperature'], 0.42)
        self.assertEqual(payload['max_tokens'], 321)
        self.assertTrue(payload['stream'])

    def test_sse_parser_handles_chunks_done_and_noise(self):
        lines = [
            'event: message\n',
            'data: {"choices":[{"delta":{"content":"hel"}}]}\n',
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
            'data: {"choices":[{"delta":{}}]}\n',
            'data: not-json\n',
            'data: [DONE]\n',
        ]
        self.assertEqual(
            list(parse_openai_sse_lines(lines)),
            [('chunk', 'hel'), ('chunk', 'lo'), ('done', '')],
        )

    def test_stream_chat_completion_obeys_cancel_token(self):
        model = ModelConfig(id='tiny', name='Tiny', path='tiny.gguf', alias='tiny-local', port=18080)
        token = CancelToken()

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"content":"first"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"second"}}]}\n',
                ])

        with patch('llama_tui.chat.request.urlopen', return_value=FakeResponse()):
            stream = stream_chat_completion(model, [{'role': 'user', 'content': 'hi'}], cancel_token=token)
            self.assertEqual(next(stream), 'first')
            token.cancel('test cancel')
            with self.assertRaises(CancelledError):
                next(stream)


class ProfileUiTests(unittest.TestCase):
    def test_simple_profile_mapping(self):
        self.assertEqual(simple_profile_action('auto_profile')[:2], ('best', 'auto'))
        self.assertEqual(simple_profile_action('balanced_chat')[:2], ('tokens_per_sec', 'moderate'))
        self.assertEqual(simple_profile_action('fast_chat')[:2], ('tokens_per_sec', 'extreme'))
        self.assertEqual(simple_profile_action('long_context')[:2], ('max_context', 'moderate'))

    def test_profile_labels_are_friendly(self):
        self.assertEqual(profile_label('max_context'), 'Long Context')
        self.assertEqual(profile_label('tokens_per_sec'), 'Fast Chat')
        self.assertEqual(profile_label('tokens_per_sec_q8_kv'), 'Fast Chat q8 KV')
        self.assertEqual(profile_label('measured_auto'), 'Measured Auto')

    def test_try_exit_helper_always_calls_stop(self):
        model = ModelConfig(id='tiny', name='Tiny', path='tiny.gguf', alias='tiny-local', port=18080)

        class FakeApp:
            def __init__(self):
                self.stopped_model = None

            def stop(self, stopped):
                self.stopped_model = stopped
                return True, 'stopped'

        app = FakeApp()
        ok, msg = stop_try_model(app, model)

        self.assertTrue(ok)
        self.assertEqual(msg, 'stopped')
        self.assertIs(app.stopped_model, model)

    def test_quit_keepalive_policy_helpers(self):
        class FakeApp:
            def __init__(self):
                self.left_running = False

            def leave_managed_processes_running(self):
                self.left_running = True

        self.assertTrue(should_prompt_quit_keepalive(True, False))
        self.assertFalse(should_prompt_quit_keepalive(False, False))
        self.assertFalse(should_prompt_quit_keepalive(True, True))

        app = FakeApp()
        should_quit, msg = apply_quit_policy(app, 'leave')
        self.assertTrue(should_quit)
        self.assertTrue(app.left_running)
        self.assertIn('Leaving', msg)

        should_quit, msg = apply_quit_policy(app, 'cancel')
        self.assertFalse(should_quit)
        self.assertIn('cancelled', msg.lower())

    def test_benchmark_wiki_lines_include_expected_topics(self):
        text = '\n'.join(benchmark_wiki_lines(48))

        self.assertIn('What is a benchmark?', text)
        self.assertIn('ctx', text)
        self.assertIn('parallel', text)
        self.assertIn('smart bounded', text)
        self.assertIn('Fast benchmark: F', text)
        self.assertIn('OpenCode benchmark: O', text)
        self.assertIn('headless', text)
        self.assertIn('Fast Chat', text)
        self.assertIn('Long Context', text)
        self.assertIn('Auto', text)
        self.assertIn('Break Point', text)

    def test_clamp_scroll_bounds(self):
        self.assertEqual(clamp_scroll(-10, 100, 10), 0)
        self.assertEqual(clamp_scroll(999, 100, 10), 90)
        self.assertEqual(clamp_scroll(5, 8, 10), 0)

    def test_scrollable_pane_defaults_to_newest_lines(self):
        lines = [f'line {idx}' for idx in range(8)]

        visible, scroll, has_older, has_newer = scrollable_pane_view(lines, width=20, rows=3, scroll=0)

        self.assertEqual(scroll, 0)
        self.assertEqual(visible[:3], ['line 5', 'line 6', 'line 7'])
        self.assertTrue(has_older)
        self.assertFalse(has_newer)

    def test_scrollable_pane_scrolls_back_and_forward(self):
        lines = [f'line {idx}' for idx in range(8)]
        total = len(scrollable_pane_wrapped_lines(lines, width=20))

        scroll = adjust_scroll_offset(0, 'page_older', total, 3)
        visible, scroll, has_older, has_newer = scrollable_pane_view(lines, width=20, rows=3, scroll=scroll)

        self.assertEqual(visible[:3], ['line 2', 'line 3', 'line 4'])
        self.assertTrue(has_older)
        self.assertTrue(has_newer)

        oldest_scroll = adjust_scroll_offset(scroll, 'oldest', total, 3)
        oldest_visible, oldest_scroll, has_older, has_newer = scrollable_pane_view(lines, width=20, rows=3, scroll=oldest_scroll)

        self.assertEqual(oldest_visible[:3], ['line 0', 'line 1', 'line 2'])
        self.assertFalse(has_older)
        self.assertTrue(has_newer)

        scroll = adjust_scroll_offset(scroll, 'newest', total, 3)
        visible, scroll, _has_older, has_newer = scrollable_pane_view(lines, width=20, rows=3, scroll=scroll)

        self.assertEqual(scroll, 0)
        self.assertEqual(visible[:3], ['line 5', 'line 6', 'line 7'])
        self.assertFalse(has_newer)

    def test_scrollable_pane_wraps_long_lines_for_bounds(self):
        lines = ['alpha beta gamma delta', 'tail']

        wrapped = scrollable_pane_wrapped_lines(lines, width=10)
        max_scroll = scrollable_pane_max_scroll(lines, width=10, rows=2)

        self.assertGreater(len(wrapped), len(lines))
        self.assertEqual(max_scroll, len(wrapped) - 2)

    def test_wrapped_item_lines_indent_continuations(self):
        wrapped = wrap_display_item_lines('alpha beta gamma delta', width=12)

        self.assertGreater(len(wrapped), 1)
        self.assertTrue(wrapped[1].startswith('  '))

    def test_right_tabs_by_view_and_defaults(self):
        self.assertEqual(right_tabs_for_view('detail'), ['summary', 'logs', 'errors', 'command', 'benchmarks'])
        self.assertEqual(right_tabs_for_view('benchmark'), ['progress', 'results', 'commands', 'logs', 'errors'])
        self.assertEqual(right_tabs_for_view('try'), ['profile', 'logs', 'errors', 'stats', 'command'])
        self.assertEqual(right_tabs_for_view('results'), ['run_summary', 'rankings', 'failures'])

        self.assertEqual(default_right_tab('detail'), 'summary')
        self.assertEqual(default_right_tab('benchmark'), 'progress')
        self.assertEqual(normalize_right_tab('detail', 'missing'), 'summary')

    def test_right_tab_cycling_and_scroll_keys(self):
        self.assertEqual(cycle_right_tab('benchmark', 'progress', 1), 'results')
        self.assertEqual(cycle_right_tab('benchmark', 'progress', -1), 'errors')
        self.assertEqual(cycle_right_tab('benchmark', 'missing', 1), 'results')
        self.assertEqual(right_tab_scroll_key('benchmark', 'commands'), 'benchmark:commands')
        self.assertEqual(right_tab_scroll_key('benchmark', 'missing'), 'benchmark:progress')
        self.assertEqual(right_tab_label('errors', 3), 'Errors 3')
        self.assertEqual(right_tab_label('logs', 3), 'Logs')

    def test_header_dashboard_layout_is_responsive(self):
        enabled, left_w, right_x, right_w = header_dashboard_layout(150)

        self.assertTrue(enabled)
        self.assertGreaterEqual(left_w, 76)
        self.assertGreater(right_x, left_w)
        self.assertLess(right_x, 150)
        self.assertLessEqual(right_x + right_w, 150)

        small_enabled, _left_w, _right_x, _right_w = header_dashboard_layout(110)
        self.assertFalse(small_enabled)

    def test_header_dashboard_content_builder(self):
        model = ModelConfig(id='tiny', name='Tiny', path='tiny.gguf', alias='tiny', port=18080)
        state = new_benchmark_run_state(model.id, 'hermes', 'Hermes workflow', now=10.0)
        state.update({
            'active': True,
            'phase': 'Hermes readiness',
            'candidate': 'opencode_ready/measured',
            'completed': 1,
            'total': 4,
        })
        items = build_header_dashboard_items(
            {
                'tiny': ('READY', 'ok'),
                'other': ('ERROR', 'boom'),
                'loading': ('STARTING', ''),
            },
            model,
            ('READY', 'ok'),
            'benchmark',
            state,
            True,
            'Hermes workflow',
            'cpu=8 ram=12/32GiB',
            ['Hermes not ready, needs 64000 ctx/slot'],
            width=80,
        )
        text = '\n'.join(line for line, _kind in items)

        self.assertEqual(header_dashboard_title('benchmark'), 'Benchmark Status')
        self.assertIn('READY:1', text)
        self.assertIn('LOADING:1', text)
        self.assertIn('ERROR:1', text)
        self.assertIn('active: tiny READY', text)
        self.assertNotIn('action:', text)
        self.assertIn('view: Benchmark', text)
        self.assertIn('bench: hermes Hermes readiness 1/4 25%', text)
        self.assertIn('hardware: cpu=8', text)
        self.assertIn('last error: Hermes not ready', text)

    def test_header_config_content_builder_mentions_lm_studio_roots(self):
        class FakeApp:
            config_path = '/tmp/models.json'
            llama_server = '/bin/llama-server'
            vllm_command = 'vllm'
            hf_cache_root = '/hf'
            llmfit_cache_root = '/llmfit'
            llm_models_cache_root = '/models'

            class OpenCode:
                path = '/tmp/opencode.json'

            class Hermes:
                command = 'hermes'

            opencode = OpenCode()
            hermes = Hermes()

            def lm_studio_roots(self):
                return [Path('/lmstudio/models'), Path('/lmstudio/hub/models')]

        items = build_header_config_items(FakeApp(), 'Ready.', width=120)
        text = '\n'.join(line for line, _kind in items)

        self.assertIn('config: /tmp/models.json', text)
        self.assertIn('llama-server: /bin/llama-server', text)
        self.assertIn('opencode: /tmp/opencode.json', text)
        self.assertIn('hermes: hermes', text)
        self.assertIn('lm-studio=/lmstudio/models, /lmstudio/hub/models', text)
        self.assertIn('message: Ready.', text)

    def test_log_and_error_tab_builders_are_separate(self):
        logs = build_log_items(['server line'], log_attr=2, muted_attr=4)
        errors = build_error_items(['boom'], error_attr=1, muted_attr=4)
        empty_errors = build_error_items([], error_attr=1, muted_attr=4)

        self.assertEqual(logs, [('server line', 2)])
        self.assertEqual(errors, [('boom', 1)])
        self.assertIn(('No errors captured for this model/run.', 4), empty_errors)

    def test_benchmark_progress_content_builder_includes_runtime_fields(self):
        model = ModelConfig(id='tiny', name='Tiny', path='tiny.gguf', alias='tiny', port=18080)
        state = new_benchmark_run_state(model.id, 'server', 'bench', now=10.0)
        state.update({
            'status': 'running',
            'phase': 'measure',
            'candidate': 'ctx=4096',
            'completed': 1,
            'total': 4,
            'records': [{
                'preset': 'opencode_ready',
                'score': 0.0,
                'seconds': 0.0,
                'ctx': 4096,
                'ctx_per_slot': 4096,
                'parallel': 1,
                'status': 'not Hermes-ready',
                'required_context': 64000,
                'actual_ctx_per_slot': 4096,
                'detail': 'not Hermes-ready: needs 64000 ctx/slot; candidate has 4096',
            }],
        })

        items = build_benchmark_progress_items(model, state, 'READY', 'ok', 123, width=40)
        text = '\n'.join(line for line, _attr in items)

        self.assertIn('model: tiny', text)
        self.assertIn('status: running / server READY', text)
        self.assertIn('phase: measure', text)
        self.assertIn('candidate: ctx=4096', text)
        self.assertIn('pid: 123', text)
        self.assertIn('progress:', text)
        self.assertIn('profile:', text)
        self.assertIn('runtime:', text)
        self.assertIn('latest result:', text)
        self.assertIn('Hermes readiness: needs 64000', text)


class TryLiveStatsTests(unittest.TestCase):
    def test_live_metrics_waiting_before_chunks(self):
        metrics = new_try_live_metrics()
        reset_try_live_metrics(metrics, now=10.0)

        snapshot = try_live_metric_snapshot(metrics, now=12.0)

        self.assertEqual(snapshot['tokens'], 0.0)
        self.assertEqual(snapshot['seconds'], 2.0)
        self.assertEqual(snapshot['tokens_per_sec'], 0.0)
        self.assertEqual(snapshot['active'], 1.0)

    def test_live_metrics_track_streaming_and_completed_speed(self):
        metrics = new_try_live_metrics()
        reset_try_live_metrics(metrics, now=10.0)
        update_try_live_metrics(metrics, 'hello world', now=12.0)

        streaming = try_live_metric_snapshot(metrics, now=12.0)
        self.assertEqual(streaming['tokens'], 2.0)
        self.assertEqual(streaming['tokens_per_sec'], 1.0)
        self.assertEqual(streaming['active'], 1.0)

        finish_try_live_metrics(metrics, now=14.0)
        finished = try_live_metric_snapshot(metrics, now=20.0)
        self.assertEqual(finished['tokens'], 2.0)
        self.assertEqual(finished['seconds'], 4.0)
        self.assertEqual(finished['tokens_per_sec'], 0.5)
        self.assertEqual(finished['active'], 0.0)

    def test_try_live_stat_lines_include_benchmark_and_runtime_info(self):
        metrics = new_try_live_metrics()
        reset_try_live_metrics(metrics, now=1.0)
        update_try_live_metrics(metrics, 'hello world', now=2.0)
        model = ModelConfig(
            id='tiny',
            name='Tiny Model',
            path='tiny.gguf',
            alias='tiny-local',
            port=18080,
            ctx=4096,
            output=512,
            last_benchmark_tokens_per_sec=12.34,
            last_benchmark_profile='tokens_per_sec/moderate',
        )

        lines = build_try_live_stat_lines(model, 'responding', 1234, metrics, now=3.0)
        joined = '\n'.join(lines)

        self.assertIn('model: Tiny Model', joined)
        self.assertIn('profile:', joined)
        self.assertIn('benchmark: 12.34 tok/s tokens_per_sec/moderate', joined)
        self.assertIn('live:', joined)
        self.assertIn('status: responding pid=1234', joined)
        self.assertIn('ctx/output: 4096/512', joined)

    def test_try_live_stat_lines_handle_missing_benchmark(self):
        metrics = new_try_live_metrics()
        model = ModelConfig(id='tiny', name='', path='tiny.gguf', alias='tiny-local', port=18080)

        lines = build_try_live_stat_lines(model, 'ready', None, metrics, now=1.0)
        joined = '\n'.join(lines)

        self.assertIn('model: tiny', joined)
        self.assertIn('benchmark: not run', joined)
        self.assertIn('last: waiting / 0 tok / 0.0s', joined)


class TryInputViewportTests(unittest.TestCase):
    def test_try_input_wraps_with_prompt_prefix(self):
        lines = try_input_wrapped_lines('hello world from llama tui', width=10)

        self.assertGreater(len(lines), 1)
        self.assertTrue(lines[0].startswith('> '))

    def test_try_input_view_clamps_scroll_and_reports_overflow(self):
        text = 'one two three four five six seven eight nine ten'
        max_scroll = try_input_max_scroll(text, width=8, rows=3)

        visible, scroll, has_more_above, has_more_below = try_input_view(text, width=8, rows=3, scroll=999)

        self.assertGreater(max_scroll, 0)
        self.assertEqual(scroll, max_scroll)
        self.assertEqual(len(visible), 3)
        self.assertTrue(has_more_above)
        self.assertFalse(has_more_below)


class BenchmarkDashboardTests(unittest.TestCase):
    def test_progress_bar_and_fraction_are_clamped(self):
        self.assertEqual(benchmark_progress_fraction(5, 10), 0.5)
        self.assertEqual(benchmark_progress_fraction(20, 10), 1.0)
        self.assertEqual(benchmark_progress_fraction(1, 0), 0.0)
        self.assertEqual(progress_bar_text(2, 4, 8), '[####----]')

    def test_reducer_tracks_results_and_done_state(self):
        state = new_benchmark_run_state(now=10.0)
        reduce_benchmark_event(
            state,
            {
                'event': 'benchmark_started',
                'model_id': 'tiny',
                'run_kind': 'server',
                'message': 'started',
                'completed': 0,
                'total': 2,
            },
            now=10.0,
        )
        reduce_benchmark_event(
            state,
            {
                'event': 'benchmark_result',
                'message': 'candidate ok',
                'completed': 1,
                'total': 2,
                'record': {
                    'spectrum_label': 'Fastest',
                    'tokens_per_sec': 42.0,
                    'seconds': 3.0,
                    'ctx': 4096,
                    'ctx_per_slot': 2048,
                    'parallel': 2,
                    'status': 'ok',
                },
            },
            now=13.0,
        )
        reduce_benchmark_event(state, {'event': 'benchmark_done', 'message': 'done'}, now=15.0)

        self.assertFalse(state['active'])
        self.assertEqual(state['status'], 'done')
        self.assertEqual(state['ended_at'], 15.0)
        self.assertEqual(len(state['records']), 1)
        self.assertIn('candidate ok', state['feed'])
        self.assertEqual(benchmark_elapsed_text(state, now=15.0), '00:05')
        self.assertEqual(benchmark_elapsed_text(state, now=99.0), '00:05')

    def test_reducer_tracks_benchmark_commands_separately(self):
        state = new_benchmark_run_state(now=10.0)
        reduce_benchmark_event(
            state,
            {
                'event': 'benchmark_started',
                'model_id': 'tiny',
                'run_kind': 'server',
                'message': 'started',
            },
            now=10.0,
        )

        self.assertEqual(state['commands'], [])
        self.assertEqual(state['current_command'], '')

        reduce_benchmark_event(
            state,
            {
                'event': 'benchmark_candidate',
                'message': 'running candidate',
                'command': 'llama-server --ctx-size 4096',
            },
            now=11.0,
        )

        self.assertEqual(state['current_command'], 'llama-server --ctx-size 4096')
        self.assertEqual(state['commands'], ['llama-server --ctx-size 4096'])
        self.assertIn('running candidate', state['feed'])

        reduce_benchmark_event(
            state,
            {
                'event': 'benchmark_phase',
                'message': 'opencode run --pure --dir /tmp/work',
                'command': 'opencode run --pure --dir /tmp/work',
            },
            now=12.0,
        )

        self.assertEqual(state['current_command'], 'opencode run --pure --dir /tmp/work')
        self.assertNotIn('opencode run --pure --dir /tmp/work', state['feed'])

        for idx in range(20):
            reduce_benchmark_event(state, {'event': 'benchmark_candidate', 'command_preview': f'cmd-{idx}'}, now=13.0 + idx)

        self.assertEqual(len(state['commands']), 12)
        self.assertEqual(state['commands'][0], 'cmd-8')
        self.assertEqual(state['commands'][-1], 'cmd-19')

    def test_benchmark_command_lines_format_current_and_history(self):
        empty = benchmark_command_lines(new_benchmark_run_state(), width=40, max_rows=3)
        self.assertEqual(empty, [('waiting for first command...', 'muted')])

        state = new_benchmark_run_state()
        state['current_command'] = 'llama-server --ctx-size 4096 --very-long-flag value'
        state['commands'] = ['cmd-1', 'cmd-2', state['current_command']]

        lines = benchmark_command_lines(state, width=32, max_rows=3)

        self.assertEqual(lines[0][1], 'current')
        self.assertTrue(lines[0][0].startswith('current: llama-server'))
        self.assertGreater(len(lines[0][0]), 32)
        self.assertIn(('recent: cmd-2', 'muted'), lines)

    def test_benchmark_row_text_includes_tradeoff_fields(self):
        row = benchmark_row_text({
            'spectrum_label': 'Highest Context',
            'tokens_per_sec': 12.5,
            'seconds': 6.25,
            'ctx': 32768,
            'ctx_per_slot': 32768,
            'parallel': 1,
            'status': 'ok',
            'scan_level': 'knee_refine',
            'exit_code': 0,
            'context_required': 9616,
        })

        self.assertIn('Highest Context', row)
        self.assertIn('12.50', row)
        self.assertIn('ctx=32768', row)
        self.assertIn('slot=32768', row)
        self.assertIn('ok', row)
        self.assertIn('knee_refine', row)
        self.assertIn('needs~9616tok', row)

    def test_benchmark_record_display_items_include_sample_details(self):
        items = benchmark_record_display_items({
            'preset': 'auto',
            'score': 10.0,
            'seconds': 1.0,
            'ctx': 2048,
            'ctx_per_slot': 2048,
            'parallel': 1,
            'status': 'hermes command failed',
            'detail': 'bad flag',
            'required_context': 64000,
            'configured_context_length': 70000,
            'actual_ctx_per_slot': 2048,
            'experimental_context_override': True,
            'samples': [{
                'task': 'fix_calc',
                'status': 'hermes command failed',
                'exit_code': 2,
                'timeout_type': '',
                'unittest_command_seen': False,
                'command_preview': 'hermes chat -q fix',
                'config_path': '/tmp/hermes/config.yaml',
                'stderr_tail': ['bad flag'],
                'stdout_tail': [],
            }],
        })
        text = '\n'.join(line for line, _attr in items)

        self.assertIn('detail: bad flag', text)
        self.assertIn('context: required=64000 configured=70000 actual_slot=2048 experimental override', text)
        self.assertIn('command: hermes chat -q fix', text)
        self.assertIn('config: /tmp/hermes/config.yaml', text)
        self.assertIn('stderr: bad flag', text)

    def test_benchmark_runs_for_model_falls_back_to_legacy_rows(self):
        model = ModelConfig(
            id='tiny',
            name='Tiny',
            path='tiny.gguf',
            alias='tiny',
            port=18080,
            last_benchmark_profile='auto/exhaustive 12 tok/s',
            last_benchmark_results=[{'status': 'ok', 'ctx': 2048}],
        )

        runs = benchmark_runs_for_model(model)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]['id'], 'legacy-latest')
        self.assertIn('auto/exhaustive', runs[0]['summary'])

    def test_results_run_line_and_ranking_rows(self):
        run = {
            'id': 'server-1',
            'status': 'done',
            'summary': 'fast=12.00 tok/s',
            'winners': {
                'fast_chat': {'ctx': 4096, 'ctx_per_slot': 2048, 'parallel': 2, 'tokens_per_sec': 40.0},
                'long_context': {'ctx': 32768, 'ctx_per_slot': 32768, 'parallel': 1, 'tokens_per_sec': 12.0},
            },
            'records': [
                {'status': 'ok', 'objective': 'fast_chat', 'ctx': 4096, 'ctx_per_slot': 2048, 'parallel': 2, 'tokens_per_sec': 40.0},
                {'status': 'ok', 'objective': 'auto', 'ctx': 8192, 'ctx_per_slot': 8192, 'parallel': 1, 'tokens_per_sec': 20.0},
                {
                    'status': 'start failed',
                    'objective': 'long_context',
                    'variant': 'default',
                    'ctx': 34816,
                    'parallel': 1,
                    'break_point': True,
                    'detail': 'oom',
                },
            ],
        }

        self.assertIn('server-1', benchmark_run_line(run, 0, selected=True))
        lines = '\n'.join(benchmark_ranking_rows(run))

        self.assertIn('#01 [Winner, Fastest]', lines)
        self.assertIn('40.00 tok/s', lines)
        self.assertIn('[Break Point]', lines)
        self.assertLess(lines.index('[Winner, Fastest]'), lines.index('[Break Point]'))

        items = benchmark_ranking_items(run, success_attr=1, warning_attr=2, error_attr=3, heading_attr=4)
        self.assertEqual(items[0], ('ranked candidates:', 4))
        self.assertTrue(any('[Break Point]' in line and attr == 3 for line, attr in items))

    def test_agent_ranking_rows_sort_passed_tasks_before_failures(self):
        run = {
            'kind': 'hermes',
            'winners': {
                'hermes': {'status': 'tests passed', 'score': 120.0, 'ctx': 8192, 'ctx_per_slot': 8192, 'parallel': 1},
            },
            'records': [
                {'runtime': 'hermes', 'status': 'hermes command failed', 'score': 0.0, 'ctx': 4096, 'ctx_per_slot': 4096, 'parallel': 1, 'detail': 'bad flag'},
                {'runtime': 'hermes', 'status': 'tests passed', 'score': 120.0, 'ctx': 8192, 'ctx_per_slot': 8192, 'parallel': 1},
            ],
        }

        lines = benchmark_ranking_rows(run)
        text = '\n'.join(lines)

        self.assertIn('#01 [Winner, Passed] tests passed 120.00 score', text)
        self.assertIn('#02 [Failed] hermes command failed', text)


if __name__ == '__main__':
    unittest.main()
