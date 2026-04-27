import curses
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llama_tui.chat import build_chat_payload, parse_openai_sse_lines, stream_chat_completion, stream_chat_events
from llama_tui.control import CancelToken, CancelledError
from llama_tui.models import ModelConfig
from llama_tui.benchmark import machine_best_summary
from llama_tui.ui import (
    apply_quit_policy,
    adjust_scroll_offset,
    benchmark_freshness_label,
    benchmark_ranking_items,
    body_content_bottom,
    body_content_rows,
    body_pane_layout,
    body_pane_height,
    active_engine_detail_line,
    browser_models,
    build_error_source_lines,
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
    browser_model_line,
    BROWSER_HEADER,
    benchmark_wiki_lines,
    benchmark_command_lines,
    build_try_live_stat_lines,
    build_try_transcript_items,
    clamp_scroll,
    config_doctor_items,
    deep_benchmark_all_options,
    finish_try_live_metrics,
    machine_category_items,
    machine_gap_items,
    machine_ranking_items,
    model_sort_key,
    new_try_live_metrics,
    new_benchmark_run_state,
    launch_options_for_stopped_model,
    parse_bool_text,
    parse_browser_filter_answers,
    parse_model_form_answers,
    parse_settings_form_answers,
    profile_label,
    progress_bar_text,
    reduce_benchmark_event,
    reset_try_live_metrics,
    cycle_right_tab,
    sort_mode_label,
    default_right_tab,
    normalize_right_tab,
    right_scroll_action_for_view,
    right_tab_label,
    right_tab_key_direction,
    right_tab_scroll_key,
    right_tabs_for_view,
    should_prompt_quit_keepalive,
    should_stop_try_model,
    simple_profile_action,
    scrollable_pane_max_scroll,
    scrollable_pane_view,
    scrollable_pane_wrapped_lines,
    stop_try_model,
    try_live_metric_snapshot,
    try_input_max_scroll,
    try_input_row_count,
    try_input_view,
    try_transcript_scroll_action,
    try_input_wrapped_lines,
    turboquant_detail_line,
    turboquant_status_kind,
    runtime_engine_source_line,
    update_try_live_metrics,
    visible_selection_window,
    wrap_display_item_lines,
)
from llama_tui.textutil import is_error_message


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

    def test_sse_parser_emits_reasoning_chunks(self):
        lines = [
            'data: {"choices":[{"delta":{"reasoning":"plan","reasoning_content":" more","content":"answer"}}]}\n',
            'data: [DONE]\n',
        ]

        self.assertEqual(
            list(parse_openai_sse_lines(lines)),
            [('reasoning', 'plan more'), ('chunk', 'answer'), ('done', '')],
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

    def test_stream_chat_events_preserves_reasoning_and_chunks(self):
        model = ModelConfig(id='tiny', name='Tiny', path='tiny.gguf', alias='tiny-local', port=18080)

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"reasoning":"think"}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"answer"}}]}\n',
                ])

        with patch('llama_tui.chat.request.urlopen', return_value=FakeResponse()):
            events = list(stream_chat_events(model, [{'role': 'user', 'content': 'hi'}]))

        self.assertEqual(events, [('reasoning', 'think'), ('chunk', 'answer')])


class BrowserAndFormTests(unittest.TestCase):
    def test_parse_bool_text_accepts_common_values(self):
        self.assertTrue(parse_bool_text('true'))
        self.assertTrue(parse_bool_text('YES'))
        self.assertFalse(parse_bool_text('0'))
        with self.assertRaises(ValueError):
            parse_bool_text('maybe')

    def test_parse_model_form_answers_reports_field_errors(self):
        initial = ModelConfig(id='tiny', name='Tiny', path='tiny.gguf', alias='tiny', port=18080)
        model, errors = parse_model_form_answers({
            'id': '',
            'name': 'Tiny',
            'path': '',
            'alias': 'tiny',
            'runtime': 'wrong',
            'optimize_mode': 'max_context_safe',
            'optimize_tier': 'wild',
            'port': 'oops',
            'host': '',
            'ctx': '8192',
            'ctx_min': '2048',
            'ctx_max': '1024',
            'threads': '4',
            'ngl': '10',
            'temp': '0.7',
            'parallel': '1',
            'memory_reserve_percent': '25',
            'cache_ram': '0',
            'output': '512',
            'enabled': 'true',
            'flash_attn': 'true',
            'jinja': 'true',
            'favorite': 'false',
            'extra_args': '',
        }, initial=initial)

        self.assertIsNone(model)
        self.assertIn('id', errors)
        self.assertIn('path', errors)
        self.assertIn('runtime', errors)
        self.assertIn('optimize_tier', errors)
        self.assertIn('port', errors)
        self.assertIn('ctx_max', errors)
        self.assertIn('host', errors)

    def test_parse_settings_form_answers_validates_preferences(self):
        parsed, errors = parse_settings_form_answers({
            'llama_server': '/bin/llama-server',
            'vllm_command': 'vllm',
            'hf_cache_root': '/hf',
            'llm_models_cache_root': '/models',
            'llmfit_cache_root': '/llmfit',
            'lm_studio_model_roots': '/lm',
            'opencode_path': '/tmp/opencode.json',
            'opencode_backup_dir': '/tmp/backups',
            'continue_path': '/tmp/config.yaml',
            'continue_backup_dir': '/tmp/backups',
            'default_model_id': 'main',
            'small_model_id': 'small',
            'build_model_id': 'build',
            'plan_model_id': 'plan',
            'instructions': './a.md, ./b.md',
            'build_prompt': 'build',
            'plan_prompt': 'plan',
            'timeout': '1000',
            'chunk_timeout': '100',
            'terminal_command': '',
            'last_workspace_path': '/tmp/project',
            'hermes_command': 'hermes',
            'hermes_home_root': '/tmp/hermes',
            'hermes_default_model_id': 'main',
            'hermes_code_model_id': 'code',
            'hermes_toolsets': 'terminal, file',
            'hermes_max_turns': '20',
            'hermes_quiet': 'not-bool',
            'hermes_min_context_tokens': '64000',
            'hermes_allow_experimental_context_override': 'false',
            'hermes_experimental_context_override_tokens': '0',
            'hermes_terminal_command': '',
            'hermes_last_workspace_path': '/tmp/project',
            'preferred_sort': 'recent',
            'detail_density': 'dense',
        })

        self.assertIsNone(parsed)
        self.assertIn('hermes_quiet', errors)
        self.assertIn('detail_density', errors)

    def test_parse_settings_form_answers_round_trips_continue_roles(self):
        parsed, errors = parse_settings_form_answers({
            'llama_server': '/bin/llama-server',
            'vllm_command': 'vllm',
            'hf_cache_root': '/hf',
            'llm_models_cache_root': '/models',
            'llmfit_cache_root': '/llmfit',
            'lm_studio_model_roots': '/lm',
            'opencode_path': '/tmp/opencode.json',
            'opencode_backup_dir': '/tmp/backups',
            'default_model_id': 'main',
            'small_model_id': 'small',
            'build_model_id': 'build',
            'plan_model_id': 'plan',
            'instructions': './a.md, ./b.md',
            'build_prompt': 'build',
            'plan_prompt': 'plan',
            'timeout': '1000',
            'chunk_timeout': '100',
            'terminal_command': '',
            'last_workspace_path': '/tmp/project',
            'continue_path': '/tmp/config.yaml',
            'continue_backup_dir': '/tmp/backups',
            'continue_default_model_id': 'continue-main',
            'continue_edit_model_id': 'continue-edit',
            'continue_autocomplete_model_id': 'continue-small',
            'continue_merge_mode': 'preserve_sections',
            'hermes_command': 'hermes',
            'hermes_home_root': '/tmp/hermes',
            'hermes_default_model_id': 'main',
            'hermes_code_model_id': 'code',
            'hermes_toolsets': 'terminal, file',
            'hermes_max_turns': '20',
            'hermes_quiet': 'true',
            'hermes_min_context_tokens': '64000',
            'hermes_allow_experimental_context_override': 'false',
            'hermes_experimental_context_override_tokens': '0',
            'hermes_terminal_command': '',
            'hermes_last_workspace_path': '/tmp/project',
            'preferred_sort': 'recent',
            'detail_density': 'advanced',
        })

        self.assertFalse(errors)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['continue']['default_model_id'], 'continue-main')
        self.assertEqual(parsed['continue']['edit_model_id'], 'continue-edit')
        self.assertEqual(parsed['continue']['autocomplete_model_id'], 'continue-small')
        self.assertEqual(parsed['continue']['merge_mode'], 'preserve_sections')

    def test_browser_models_filters_and_sorts_by_user_preferences(self):
        alpha = ModelConfig(id='alpha', name='Alpha', path='alpha.gguf', alias='alpha', port=18080, runtime='llama.cpp')
        beta = ModelConfig(id='beta', name='Beta', path='beta.gguf', alias='beta', port=18081, runtime='vllm')
        gamma = ModelConfig(id='gamma', name='Gamma', path='gamma.gguf', alias='gamma', port=18082, runtime='llama.cpp')
        alpha.favorite = True
        alpha.last_used_at = '2026-04-23T10:00:00'
        gamma.last_used_at = '2026-04-23T12:00:00'
        beta.source = 'lm-studio'
        beta.ctx = 65536
        beta.parallel = 2
        alpha.default_benchmark_status = 'done'
        alpha.benchmark_fingerprint = 'fp-alpha'
        alpha.measured_profiles = {'auto': {'status': 'ok', 'tokens_per_sec': 60.0, 'ctx': 8192, 'ctx_per_slot': 8192, 'parallel': 1}}

        class FakeApp:
            models = [alpha, beta, gamma]

            def model_fingerprint(self, model):
                return f'fp-{model.id}'

        statuses = {
            'alpha': ('READY', ''),
            'beta': ('STOPPED', ''),
            'gamma': ('ERROR', ''),
        }

        filtered = browser_models(FakeApp(), statuses, search='beta', runtime_filter='vllm', source_filter='lm-studio', status_filter='all', sort_mode='name')
        self.assertEqual([model.id for model in filtered], ['beta'])

        favorites = browser_models(FakeApp(), statuses, sort_mode='favorites')
        self.assertEqual(favorites[0].id, 'alpha')

        recents = browser_models(FakeApp(), statuses, sort_mode='recent')
        self.assertEqual(recents[0].id, 'gamma')

        self.assertEqual(benchmark_freshness_label(FakeApp(), alpha), 'fresh')
        self.assertEqual(benchmark_freshness_label(FakeApp(), beta), 'missing')
        self.assertEqual(sort_mode_label('benchmark'), 'Best Benchmark')
        self.assertLess(model_sort_key(beta, 'context'), model_sort_key(alpha, 'context'))

    def test_browser_models_filters_by_tags(self):
        alpha = ModelConfig(id='alpha', name='Alpha', path='alpha.gguf', alias='alpha', port=18080, runtime='llama.cpp')
        beta = ModelConfig(id='beta', name='Beta', path='beta.gguf', alias='beta', port=18081, runtime='vllm')
        alpha.tags = ['coding', 'fast-chat']
        beta.tags = ['autocomplete']

        class FakeApp:
            models = [alpha, beta]

            def model_fingerprint(self, model):
                return f'fp-{model.id}'

        statuses = {'alpha': ('STOPPED', ''), 'beta': ('STOPPED', '')}

        filtered = browser_models(FakeApp(), statuses, tag_filter='coding')
        self.assertEqual([model.id for model in filtered], ['alpha'])

    def test_parse_browser_filter_answers_rejects_unknown_values(self):
        parsed, errors = parse_browser_filter_answers({
            'runtime_filter': 'weird',
            'source_filter': 'manual',
            'status_filter': 'READY',
            'tag_filter': 'coding',
        })

        self.assertIsNone(parsed)
        self.assertIn('runtime_filter', errors)

    def test_parse_browser_filter_answers_accepts_tag_filter(self):
        parsed, errors = parse_browser_filter_answers({
            'runtime_filter': 'all',
            'source_filter': 'manual',
            'status_filter': 'READY',
            'tag_filter': 'coding',
        })

        self.assertFalse(errors)
        self.assertEqual(parsed, ('all', 'manual', 'READY', 'coding'))

    def test_config_doctor_items_reports_verification_counts(self):
        passed = ModelConfig(id='passed', name='Passed', path='org/model', alias='passed', port=18080, runtime='vllm')
        pending = ModelConfig(id='pending', name='Pending', path='org/model2', alias='pending', port=18081, runtime='vllm')
        passed.verification_status = 'passed'
        pending.verification_status = 'needs_benchmark'
        passed.turboquant_status = 'padded'
        passed.turboquant_key_dim = 96
        passed.turboquant_value_dim = 96
        passed.turboquant_source = 'gguf_metadata'
        passed.turboquant_reason = 'buun zero-padding handles non-128 head dims'
        passed.verification_results = {
            'cap': {
                'limiting_factor': 'parallel_split',
                'configured_ctx': 8192,
                'ctx_per_slot': 4096,
                'estimated_safe_context': 65536,
                'measured_max_context': 4096,
            }
        }

        class FakeApp:
            llama_server = '/bin/sh'
            vllm_command = 'vllm'
            opencode = SimpleNamespace(path='/tmp/opencode.json')
            continue_settings = SimpleNamespace(path='/tmp/config.yaml', merge_mode='preserve_sections')
            hermes = SimpleNamespace(command='hermes', home_root='/tmp/hermes')
            models = [passed, pending]

            def command_exists(self, command):
                return command in ('/bin/sh', 'code')

            def detect_terminal_launcher(self):
                return '/usr/bin/xterm'

            def benchmark_proof_model_ids(self, force=False):
                return ['pending']

        rows = config_doctor_items(FakeApp(), active_model=passed)
        text = '\n'.join(row for row, _kind in rows)

        self.assertIn('model verification: needs_benchmark:1 passed:1', text)
        self.assertIn('benchmark proof needed: 1 model(s)', text)
        self.assertIn('cap: factor=parallel_split', text)
        self.assertIn('turboquant: padded key=96 value=96', text)

    def test_turboquant_browser_detail_and_buun_warning_labels(self):
        model = ModelConfig(id='tq', name='TurboQuant Model', path='model.gguf', alias='tq', port=18080)
        model.turboquant_status = 'unknown'
        model.turboquant_source = 'gguf_metadata'
        model.turboquant_reason = 'GGUF metadata missing or unreadable'

        class FakeApp:
            def role_badges(self, _model_id):
                return '-'

        line = browser_model_line(FakeApp(), model, 'STOPPED', '', 120)

        self.assertIn(' TQ ', BROWSER_HEADER)
        self.assertIn(' UNK ', line)
        self.assertIn('turboquant: unknown from gguf_metadata', turboquant_detail_line(model))
        self.assertEqual(turboquant_status_kind(model, buun_session=False), 'muted')
        self.assertEqual(turboquant_status_kind(model, buun_session=True), 'warning')

        model.turboquant_status = 'native'
        model.turboquant_key_dim = 128
        model.turboquant_value_dim = 128
        line = browser_model_line(FakeApp(), model, 'STOPPED', '', 120)
        self.assertIn(' NAT ', line)
        self.assertEqual(turboquant_status_kind(model, buun_session=True), 'success')

    def test_active_engine_labels_distinguish_buun_from_model_runtime(self):
        model = ModelConfig(id='gemma', name='Gemma', path='gemma.gguf', alias='gemma', port=18080)
        model.turboquant_status = 'native'

        class Profile:
            def buun_kv_pair(self):
                return 'turbo4', 'turbo4'

        class FakeApp:
            runtime_profile = Profile()

            def role_badges(self, _model_id):
                return '-'

            def active_engine_key_for_model(self, _model):
                return 'buun'

            def active_runtime_binary_for_model(self, _model):
                return 'buun-llama-server'

        line = browser_model_line(FakeApp(), model, 'STOPPED', '', 120)

        self.assertIn(' ENGINE ', BROWSER_HEADER)
        self.assertIn(' buun ', line)
        self.assertIn('model runtime/active engine', runtime_engine_source_line(FakeApp(), model))
        self.assertIn('llama.cpp / buun', runtime_engine_source_line(FakeApp(), model))
        self.assertIn('binary: buun-llama-server', active_engine_detail_line(FakeApp(), model))
        self.assertIn('key=turbo4 value=turbo4', active_engine_detail_line(FakeApp(), model))

    def test_machine_best_summary_includes_explanatory_reason(self):
        model = ModelConfig(id='balanced', name='Balanced', path='balanced.gguf', alias='balanced', port=18080)
        model.default_benchmark_status = 'done'
        model.benchmark_fingerprint = 'fp-balanced'
        model.measured_profiles = {
            'auto': {
                'status': 'ok',
                'tokens_per_sec': 75.0,
                'ctx': 32768,
                'ctx_per_slot': 32768,
                'parallel': 1,
                'ram_available': 8 * 1024**3,
                'gpu_memory_free': 4 * 1024**3,
                'benchmarked_at': '2026-04-23T12:00:00',
            },
            'fast_chat': {'status': 'ok', 'tokens_per_sec': 82.0, 'ctx': 16384, 'ctx_per_slot': 16384, 'parallel': 1},
            'long_context': {'status': 'ok', 'tokens_per_sec': 50.0, 'ctx': 65536, 'ctx_per_slot': 65536, 'parallel': 1},
            'opencode_ready': {'status': 'ok', 'tokens_per_sec': 70.0, 'ctx': 65536, 'ctx_per_slot': 65536, 'parallel': 1},
        }

        class FakeApp:
            models = [model]

            def model_fingerprint(self, _model):
                return 'fp-balanced'

        summary = machine_best_summary(FakeApp())
        reason = str(summary['machine_pick']['reason'])

        self.assertIn('auto 75.00 tok/s', reason)
        self.assertIn('ctx/slot 32768', reason)
        self.assertIn('headroom', reason)


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

    def test_try_exit_stop_decision_only_matches_launched_model(self):
        model = ModelConfig(id='tiny', name='Tiny', path='tiny.gguf', alias='tiny-local', port=18080)

        self.assertTrue(should_stop_try_model('tiny', model))
        self.assertFalse(should_stop_try_model('', model))
        self.assertFalse(should_stop_try_model('other', model))
        self.assertFalse(should_stop_try_model('tiny', None))

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
        self.assertEqual(right_tabs_for_view('machine_results'), ['overview', 'rankings', 'failures'])

        self.assertEqual(default_right_tab('detail'), 'summary')
        self.assertEqual(default_right_tab('benchmark'), 'progress')
        self.assertEqual(default_right_tab('machine_results'), 'overview')
        self.assertEqual(normalize_right_tab('detail', 'missing'), 'summary')

    def test_right_tab_cycling_and_scroll_keys(self):
        self.assertEqual(cycle_right_tab('benchmark', 'progress', 1), 'results')
        self.assertEqual(cycle_right_tab('benchmark', 'progress', -1), 'errors')
        self.assertEqual(cycle_right_tab('benchmark', 'missing', 1), 'results')
        self.assertEqual(right_tab_scroll_key('benchmark', 'commands'), 'benchmark:commands')
        self.assertEqual(right_tab_scroll_key('benchmark', 'missing'), 'benchmark:progress')
        self.assertEqual(right_tab_label('errors', 3), 'Errors 3')
        self.assertEqual(right_tab_label('logs', 3), 'Logs')
        self.assertEqual(try_transcript_scroll_action(16), 'older')
        self.assertEqual(try_transcript_scroll_action(14), 'newer')
        self.assertEqual(try_transcript_scroll_action(ord('x')), '')

    def test_try_transcript_builder_includes_reasoning(self):
        model = ModelConfig(id='tiny', name='Tiny', path='tiny.gguf', alias='tiny-local', port=18080)
        items = build_try_transcript_items(
            model,
            [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'reasoning': 'Checking options', 'content': 'Hi there'},
            ],
            'ready',
            40,
            user_attr=1,
            assistant_attr=2,
            muted_attr=3,
        )

        lines = [line for line, _attr in items]
        self.assertIn('you> Hello', lines)
        self.assertIn('tiny-local> [reasoning] Checking options', lines)
        self.assertIn('tiny-local> Hi there', lines)

    def test_try_transcript_builder_shows_reasoning_only_notice_without_placeholder(self):
        model = ModelConfig(id='tiny', name='Tiny', path='tiny.gguf', alias='tiny-local', port=18080)
        items = build_try_transcript_items(
            model,
            [
                {'role': 'assistant', 'reasoning': 'Checking options', 'content': '', 'final_notice': '[no final answer returned]'},
            ],
            'ready',
            80,
            user_attr=1,
            assistant_attr=2,
            muted_attr=3,
        )

        self.assertIn(('tiny-local> [reasoning] Checking options', 3), items)
        self.assertIn(('tiny-local> [no final answer returned]', 3), items)
        self.assertFalse(any(line == 'tiny-local> ...' for line, _attr in items))

    def test_try_transcript_builder_preserves_empty_response_fallback(self):
        model = ModelConfig(id='tiny', name='Tiny', path='tiny.gguf', alias='tiny-local', port=18080)
        items = build_try_transcript_items(
            model,
            [
                {'role': 'assistant', 'reasoning': '', 'content': '(no content returned)'},
            ],
            'ready',
            80,
            user_attr=1,
            assistant_attr=2,
            muted_attr=3,
        )

        self.assertIn(('tiny-local> (no content returned)', 2), items)

    def test_header_dashboard_layout_is_responsive(self):
        enabled, left_w, right_x, right_w = header_dashboard_layout(150)

        self.assertTrue(enabled)
        self.assertGreaterEqual(left_w, 76)
        self.assertGreater(right_x, left_w)
        self.assertLess(right_x, 150)
        self.assertLessEqual(right_x + right_w, 150)

        small_enabled, _left_w, _right_x, _right_w = header_dashboard_layout(110)
        self.assertFalse(small_enabled)

    def test_body_pane_layout_keeps_panes_inside_supported_widths(self):
        for width in (88, 100, 124, 150):
            with self.subTest(width=width):
                left_w, right_x, right_w = body_pane_layout(width)

                self.assertGreaterEqual(left_w, 1)
                self.assertGreater(right_x, left_w)
                self.assertGreaterEqual(right_w, 1)
                self.assertLessEqual(right_x + right_w, width)

    def test_body_vertical_layout_stays_above_footer(self):
        box_top = 11
        for height in range(18, 25):
            with self.subTest(height=height):
                pane_h = body_pane_height(height, box_top)
                rows = body_content_rows(height, box_top)
                bottom = body_content_bottom(height, box_top)
                footer_top = height - 2
                model_rows = max(0, rows - 1)
                input_rows = try_input_row_count(rows)

                self.assertLess(box_top + pane_h, footer_top)
                self.assertLess(bottom, footer_top)
                if model_rows:
                    self.assertLessEqual(box_top + 3 + model_rows - 1, bottom)
                if input_rows:
                    input_y = bottom - input_rows
                    self.assertLessEqual(input_y + input_rows, bottom)

    def test_selection_window_keeps_selected_row_visible(self):
        self.assertEqual(visible_selection_window(10, 0, 4), (0, 4))
        self.assertEqual(visible_selection_window(10, 5, 4), (3, 7))
        self.assertEqual(visible_selection_window(10, 9, 4), (6, 10))

        selected = 0
        for _step in range(9):
            selected += 1
            start, end = visible_selection_window(10, selected, 4)
            self.assertLessEqual(start, selected)
            self.assertLess(selected, end)

    def test_results_run_window_uses_real_indices(self):
        runs = [
            {'id': f'run-{idx}', 'status': 'done', 'summary': f'summary {idx}'}
            for idx in range(10)
        ]
        selected = 8
        start, end = visible_selection_window(len(runs), selected, 4)
        lines = [
            benchmark_run_line(runs[idx], idx, selected=(idx == selected))
            for idx in range(start, end)
        ]

        self.assertTrue(any(line.startswith('> 09') and 'run-8' in line for line in lines))
        self.assertFalse(any('run-0' in line for line in lines))

    def test_right_tab_key_routing_restores_tab_switching(self):
        self.assertEqual(right_tab_key_direction(9), 1)
        self.assertEqual(right_tab_key_direction(ord(']')), 1)
        self.assertEqual(right_tab_key_direction(getattr(curses, 'KEY_BTAB', -999)), -1)
        self.assertEqual(right_tab_key_direction(ord('[')), -1)
        self.assertEqual(right_tab_key_direction(getattr(curses, 'KEY_F6', 270)), 0)

    def test_right_scroll_key_routing_is_view_specific(self):
        self.assertEqual(right_scroll_action_for_view('detail', ord('j')), 'newer')
        self.assertEqual(right_scroll_action_for_view('benchmark', ord('k')), 'older')
        self.assertEqual(right_scroll_action_for_view('results', curses.KEY_NPAGE), 'page_newer')
        self.assertEqual(right_scroll_action_for_view('try', curses.KEY_PPAGE), 'page_older')
        self.assertEqual(right_scroll_action_for_view('machine_results', curses.KEY_END), 'newest')
        self.assertEqual(right_scroll_action_for_view('results', ord('j')), '')
        self.assertEqual(right_scroll_action_for_view('try', curses.KEY_UP), '')
        self.assertEqual(right_scroll_action_for_view('list', curses.KEY_NPAGE), '')

    def test_deep_benchmark_all_menu_options(self):
        options = deep_benchmark_all_options()

        self.assertEqual(options[0], ('1', 'Safer adaptive batch for missing/stale/failed models', 'missing'))
        self.assertEqual(options[1], ('2', 'Safer adaptive batch force refresh for every model', 'force'))
        self.assertEqual(options[-1], ('q', 'Cancel', 'cancel'))

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

            def runtime_indicator(self):
                return 'Engine: llama.cpp | KV: - | Context: model default'

            def lm_studio_roots(self):
                return [Path('/lmstudio/models'), Path('/lmstudio/hub/models')]

        items = build_header_config_items(FakeApp(), 'Ready.', width=120)
        text = '\n'.join(line for line, _kind in items)

        self.assertIn('config: /tmp/models.json', text)
        self.assertIn('llama-server: /bin/llama-server', text)
        self.assertIn('Engine: llama.cpp', text)
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

    def test_ui_labels_do_not_become_error_history(self):
        self.assertFalse(is_error_message('Right tab: Errors 3.'))
        self.assertFalse(is_error_message('No errors captured for this model/run.'))
        self.assertFalse(is_error_message('Focus: Right pane.'))
        self.assertFalse(is_error_message('0 errors in this run'))
        self.assertFalse(is_error_message('error-free launch'))
        self.assertFalse(is_error_message('completed without error'))
        self.assertTrue(is_error_message('error: chat failed'))
        self.assertTrue(is_error_message('0 errors reported, but server failed later'))
        self.assertTrue(is_error_message('error-free until it crashed'))
        self.assertEqual(build_error_source_lines([], benchmark_errors=[], benchmark_mode=False), [])
        self.assertEqual(
            build_error_source_lines(['boom'], status_error='tiny: status ERROR (oom)'),
            ['boom', 'tiny: status ERROR (oom)'],
        )

    def test_stopped_launch_menu_starts_without_benchmark_first(self):
        model = ModelConfig(id='tiny', name='Tiny', path='tiny.gguf', alias='tiny', port=18080)
        options = launch_options_for_stopped_model(model)

        self.assertEqual(options[0], ('1', 'Start server now', 'keep'))
        self.assertIn(('2', 'Auto profile', 'auto_profile'), options)

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

    def test_reducer_preserves_deep_all_batch_counters(self):
        state = new_benchmark_run_state(now=10.0)
        reduce_benchmark_event(
            state,
            {
                'event': 'benchmark_started',
                'run_kind': 'server_all',
                'model_id': '',
                'message': 'deep benchmark all started',
                'completed': 0,
                'total': 3,
                'batch_skipped': 0,
                'batch_failed': 0,
                'batch_restored': 0,
            },
            now=10.0,
        )
        reduce_benchmark_event(
            state,
            {
                'event': 'benchmark_phase',
                'run_kind': 'server_all',
                'model_id': 'second',
                'message': '[2/3] second: candidate ok',
                'completed': 1,
                'total': 3,
                'batch_skipped': 1,
                'batch_failed': 0,
                'batch_restored': 1,
            },
            now=12.0,
        )

        self.assertTrue(state['active'])
        self.assertEqual(state['run_kind'], 'server_all')
        self.assertEqual(state['model_id'], 'second')
        self.assertEqual(state['completed'], 1)
        self.assertEqual(state['total'], 3)
        self.assertEqual(state['batch_skipped'], 1)
        self.assertEqual(state['batch_restored'], 1)

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
            'process_pressure_level': 'medium',
        })

        self.assertIn('Highest Context', row)
        self.assertIn('12.50', row)
        self.assertIn('ctx=32768', row)
        self.assertIn('slot=32768', row)
        self.assertIn('ok', row)
        self.assertIn('knee_refine', row)
        self.assertIn('needs~9616tok', row)
        self.assertIn('pressure=medium', row)

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
            'architecture_label': 'MoE 8x2',
            'classification_source': 'gguf_metadata',
            'process_pressure_detail': 'pressure=medium apps=ide:1',
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
        self.assertIn('architecture: MoE 8x2 from gguf_metadata', text)
        self.assertIn('process pressure: pressure=medium apps=ide:1', text)
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

        self.assertIn('Rank Role', lines)
        self.assertIn('Winner, Fastest', lines)
        self.assertIn('40.00', lines)
        self.assertIn('Break Point', lines)
        self.assertLess(lines.index('Winner, Fastest'), lines.index('Break Point'))

        items = benchmark_ranking_items(run, success_attr=1, warning_attr=2, error_attr=3, heading_attr=4)
        self.assertTrue(items[0][0].startswith('Rank Role'))
        self.assertEqual(items[0][1], 4)
        self.assertTrue(any('Break Point' in line and attr == 3 for line, attr in items))

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

        self.assertIn('Rank Role', text)
        self.assertIn('Winner, Passed', text)
        self.assertIn('tests passed', text)
        self.assertIn('120.00', text)
        self.assertIn('Failed', text)
        self.assertIn('hermes command', text)

    def test_server_ranking_table_stays_within_narrow_widths(self):
        run = {
            'kind': 'server',
            'winners': {
                'fast_chat': {'ctx': 4096, 'ctx_per_slot': 2048, 'parallel': 2, 'tokens_per_sec': 40.0},
            },
            'records': [
                {
                    'status': 'ok',
                    'objective': 'fast_chat',
                    'ctx': 4096,
                    'ctx_per_slot': 2048,
                    'parallel': 2,
                    'tokens_per_sec': 40.0,
                    'selection_reason': 'fastest stable full measurement',
                    'detail': 'long detail that must truncate cleanly',
                },
            ],
        }

        for width in range(24, 141):
            with self.subTest(width=width):
                lines = [line for line, _attr in benchmark_ranking_items(run, width=width)]

                self.assertTrue(lines[0].startswith('Rank'))
                self.assertTrue(any('40.00' in line or '40.0' in line for line in lines))
                self.assertTrue(all(len(line) <= width for line in lines))

    def test_agent_ranking_table_stays_within_narrow_widths(self):
        run = {
            'kind': 'opencode',
            'records': [
                {
                    'runtime': 'opencode',
                    'status': 'tests passed',
                    'score': 120.0,
                    'ctx': 8192,
                    'ctx_per_slot': 8192,
                    'parallel': 1,
                    'passed': 3,
                    'tasks': 3,
                    'detail': 'all tasks passed with a long truncation detail',
                },
            ],
        }

        for width in range(24, 141):
            with self.subTest(width=width):
                lines = [line for line, _attr in benchmark_ranking_items(run, width=width)]

                self.assertTrue(lines[0].startswith('Rank'))
                self.assertTrue(any('120.00' in line or '120.' in line for line in lines))
                self.assertTrue(all(len(line) <= width for line in lines))

    def test_machine_ranking_table_and_overview_items(self):
        summary = {
            'rows': [
                {
                    'model_id': 'balanced',
                    'machine_score': 91.25,
                    'auto_tokens_per_sec': 70.0,
                    'fast_tokens_per_sec': 80.0,
                    'auto_ctx_per_slot': 32000,
                    'long_ctx_per_slot': 64000,
                    'opencode_ctx_per_slot': 64000,
                    'machine_reason': '50% speed, 30% ctx/slot, 12% headroom, 8% stability',
                },
                {
                    'model_id': 'speedy',
                    'machine_score': 88.5,
                    'auto_tokens_per_sec': 100.0,
                    'fast_tokens_per_sec': 140.0,
                    'auto_ctx_per_slot': 4096,
                    'long_ctx_per_slot': 8192,
                    'opencode_ctx_per_slot': 8192,
                    'machine_reason': 'fast but smaller context',
                },
            ],
            'categories': {
                'machine_pick': {'label': 'Machine Pick', 'model_id': 'balanced', 'metric': '91.25', 'reason': 'weighted score'},
                'fastest_chat': {'label': 'Fastest Chat', 'model_id': 'speedy', 'metric': '140.00 tok/s', 'reason': 'highest measured Fast Chat throughput'},
                'longest_context': {'label': 'Longest Context', 'model_id': 'balanced', 'metric': '64000 ctx/slot', 'reason': 'largest measured Long Context ctx/slot'},
                'opencode_ready': {'label': 'OpenCode-ready', 'model_id': 'balanced', 'metric': '64000 ctx/slot', 'reason': 'meets observed OpenCode floor 32000'},
            },
            'machine_pick': {'model_id': 'balanced'},
        }

        overview = '\n'.join(line for line, _attr in machine_category_items(summary))

        self.assertIn('Machine Pick: balanced', overview)
        self.assertIn('Fastest Chat: speedy', overview)

        for width in range(24, 121):
            with self.subTest(width=width):
                lines = [line for line, _attr in machine_ranking_items(summary, width=width)]

                self.assertTrue(lines[0].startswith('Rank'))
                self.assertTrue(any('balanced' in line for line in lines))
                self.assertTrue(all(len(line) <= width for line in lines))

    def test_machine_gap_items_report_non_fresh_models(self):
        fresh = ModelConfig(id='fresh', name='Fresh', path='fresh.gguf', alias='fresh', port=18080)
        fresh.default_benchmark_status = 'done'
        fresh.benchmark_fingerprint = 'fp-fresh'
        fresh.measured_profiles = {
            'auto': {'status': 'ok', 'tokens_per_sec': 50.0, 'ctx': 8192, 'ctx_per_slot': 8192, 'parallel': 1},
        }
        stale = ModelConfig(id='stale', name='Stale', path='stale.gguf', alias='stale', port=18081)
        stale.default_benchmark_status = 'done'
        stale.benchmark_fingerprint = 'old'
        stale.measured_profiles = {
            'auto': {'status': 'ok', 'tokens_per_sec': 45.0, 'ctx': 8192, 'ctx_per_slot': 8192, 'parallel': 1},
        }

        class FakeApp:
            models = [fresh, stale]

            def model_fingerprint(self, model):
                return f'fp-{model.id}'

        summary = {'rows': [{'model_id': 'fresh'}]}
        text = '\n'.join(line for line, _attr in machine_gap_items(FakeApp(), summary))

        self.assertIn('stale', text)
        self.assertNotIn('fresh:', text)


if __name__ == '__main__':
    unittest.main()
