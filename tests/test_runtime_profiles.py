import inspect
import json
import os
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from llama_tui.app import AppConfig
from llama_tui.benchmark_strategies import select_benchmark_strategy
from llama_tui.benchmark import (
    BenchmarkDeadline,
    active_engine_runtime_profiles,
    annotate_mtp_optimizer_records,
    adaptive_record_from_candidate,
    benchmark_adaptive_candidate,
    benchmark_all_models_runner,
    benchmark_completion,
    benchmark_completion_suite,
    benchmark_exhaustive_profiles,
    benchmark_fast_profiles,
    benchmark_max_context_probe,
    benchmark_mtp_optimizer_workloads,
    benchmark_preflight_cleanup,
    build_profile_frontier,
    benchmark_raw_speed_profile,
    benchmark_run_summary,
    benchmark_mtp_acceptance_matrix_after_preflight,
    benchmark_runtime_profile_with_retry,
    close_stale_running_benchmark_runs,
    classify_benchmark_failure,
    expand_workflow_cache_ram_candidates,
    fill_missing_adaptive_profiles,
    launch_with_failsafe,
    measured_profile_runtime_profile,
    memory_guardrail_admission,
    model_and_runtime_profile_from_measured_profile,
    model_for_runtime_profile,
    mtp_long_context_probe_request_timeout,
    max_context_probe_runtime_profiles,
    mtp_optimizer_profile_recommendations,
    parse_mtp_runtime_diagnostics,
    runtime_record_context,
    runtime_profile_memory_disable_key,
    runtime_profile_memory_skip_reason,
    select_measured_profiles,
    select_max_context_probe_profiles,
    workflow_cache_ram_profile_from_record,
    workflow_cache_ram_selection_key,
)
from llama_tui.control import CancelToken, CancelledError
from llama_tui.hardware import HardwareProfile
from llama_tui.launch_profiles import build_benchmark_launch_profile
from llama_tui.main import (
    build_cli_parser,
    ensure_engine_session_lock,
    engine_session_path,
    last_engine_session_stop_count,
    mtp_engine_deprecation_notice,
    release_engine_session_lock,
    validate_turboquant_kv_args,
)
from llama_tui.models import ModelConfig
from llama_tui.runtime_profiles import (
    EngineCapabilities,
    RuntimeProfile,
    build_mtp_args,
    default_engine_capabilities,
    detect_engine_capabilities,
    make_runtime_profile,
    parse_engine_capabilities,
    parse_spec_type_values,
    resolve_llama_cpp_mtp_binary,
)


def turboquant_no_fit_capabilities() -> EngineCapabilities:
    return replace(
        default_engine_capabilities('turboquant'),
        supports_fit=False,
        supports_fit_ctx=False,
    )


class RuntimeProfileTests(unittest.TestCase):
    def test_turboquant_profile_defaults_to_q8_baseline(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('TURBOQUANT_LLAMA_SERVER_BIN', None)
            profile = make_runtime_profile('turboquant', 'llama-server')

        self.assertEqual(profile.engine_id, 'turboquant')
        self.assertTrue(profile.supports_turbo_kv)
        self.assertEqual(profile.turboquant_kv_pair(), ('q8_0', 'q8_0'))
        self.assertEqual(profile.llama_extra_args(), ['--flash-attn', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0'])
        self.assertIn('TurboQuant+', profile.header_indicator())

    def test_turboquant_profile_respects_env_override_and_manual_kv(self):
        with patch.dict(os.environ, {'TURBOQUANT_LLAMA_SERVER_BIN': '/opt/tqp/bin/llama-server'}):
            profile = make_runtime_profile(
                'turboquant',
                'llama-server',
                kv_key_mode='q8_0',
                kv_value_mode='turbo4',
            )

        self.assertEqual(profile.server_command, '/opt/tqp/bin/llama-server')
        self.assertEqual(profile.turboquant_kv_pair(), ('q8_0', 'turbo4'))
        self.assertIn('key=q8_0 value=turbo4', profile.header_indicator())

    def test_legacy_mtp_alias_resolves_to_llama_cpp_with_mtp_binary(self):
        # Audit #7: 'llama.cpp-mtp' is now just an alias for llama.cpp
        # that opts into LLAMA_CPP_MTP_PATH for binary discovery. The
        # resulting profile reports engine_id='llama.cpp' so the rest
        # of the codebase has one fewer special case to handle.
        with patch.dict(os.environ, {'LLAMA_CPP_MTP_PATH': '/opt/mtp/bin'}):
            profile = make_runtime_profile('llama.cpp-mtp', 'llama-server')

        self.assertEqual(profile.engine_id, 'llama.cpp')
        self.assertEqual(profile.display_name, 'llama.cpp')
        self.assertEqual(profile.server_command, '/opt/mtp/bin/llama-server')
        self.assertFalse(profile.experimental)

    def test_mtp_env_directory_searches_common_build_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'llama.cpp-mtp'
            binary = root / 'build' / 'bin' / 'llama-server'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\n', encoding='utf-8')
            binary.chmod(0o755)

            with patch.dict(os.environ, {'LLAMA_CPP_MTP_PATH': str(root)}):
                profile = make_runtime_profile('llama.cpp-mtp', 'llama-server')
                resolved = resolve_llama_cpp_mtp_binary()

        self.assertEqual(profile.server_command, str(binary))
        self.assertEqual(resolved.command, str(binary))
        self.assertTrue(resolved.exists)
        self.assertTrue(resolved.executable)
        self.assertEqual(resolved.source, 'env:LLAMA_CPP_MTP_PATH')

    def test_mtp_env_prefers_build_mtp_layout_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'llama.cpp-mtp'
            binary = root / 'build-mtp' / 'bin' / 'llama-server'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\n', encoding='utf-8')
            binary.chmod(0o755)

            with patch.dict(os.environ, {'LLAMA_CPP_MTP_PATH': str(root)}):
                resolved = resolve_llama_cpp_mtp_binary()

        self.assertEqual(resolved.command, str(binary))
        self.assertTrue(resolved.exists)
        self.assertTrue(resolved.executable)

    def test_mtp_env_accepts_direct_binary_and_reports_non_executable(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'llama-server'
            binary.write_text('#!/bin/sh\n', encoding='utf-8')
            binary.chmod(0o644)

            with patch.dict(os.environ, {'LLAMA_CPP_MTP_PATH': str(binary)}):
                resolved = resolve_llama_cpp_mtp_binary()

        self.assertEqual(resolved.command, str(binary))
        self.assertTrue(resolved.exists)
        self.assertFalse(resolved.executable)

    def test_mtp_default_search_checks_home_and_path_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            binary = home / 'llama.cpp-mtp' / 'build-mtp' / 'bin' / 'llama-server'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\n', encoding='utf-8')
            binary.chmod(0o755)

            with patch.dict(os.environ, {'LLAMA_CPP_MTP_PATH': ''}, clear=False), \
                 patch('llama_tui.runtime_profiles.Path.home', return_value=home):
                resolved = resolve_llama_cpp_mtp_binary()

        self.assertEqual(resolved.command, str(binary))
        self.assertEqual(resolved.source, 'default')
        self.assertTrue(resolved.executable)

    def test_mtp_env_missing_direct_binary_reports_missing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'llama-server'

            with patch.dict(os.environ, {'LLAMA_CPP_MTP_PATH': str(binary)}):
                resolved = resolve_llama_cpp_mtp_binary()

        self.assertEqual(resolved.command, str(binary))
        self.assertFalse(resolved.exists)
        self.assertFalse(resolved.executable)

    def test_capability_parser_detects_mtp_speculative_flags(self):
        caps = parse_engine_capabilities(
            'usage: llama-server --spec-type mtp --spec-draft-n-max N --parallel N '
            '-fit on -fitt N -fitc N -ctk TYPE -ctv TYPE '
            '--spec-draft-type-k TYPE --spec-draft-type-v TYPE --cache-ram N --no-mmap --no-warmup',
            engine_id='llama.cpp-mtp',
        )

        self.assertTrue(caps.supports_spec_type)
        self.assertTrue(caps.supports_mtp)
        self.assertIn('mtp', caps.spec_type_values)
        self.assertEqual(caps.mtp_spec_type, 'mtp')
        self.assertEqual(caps.mtp_spec_type_value, 'mtp')
        self.assertTrue(caps.supports_spec_draft_n_max)
        self.assertTrue(caps.supports_spec_draft_type_kv)
        self.assertTrue(caps.supports_fit)
        self.assertTrue(caps.supports_fit_ctx)
        self.assertTrue(caps.supports_fit_target)
        self.assertTrue(caps.supports_ctk_ctv)
        self.assertTrue(caps.supports_no_mmap)
        self.assertTrue(caps.supports_cache_ram)
        self.assertTrue(caps.supports_parallel)

    def test_capability_parser_detects_draft_mtp_spec_type_dialect(self):
        caps = parse_engine_capabilities(
            '--spec-type none,draft-simple,draft-eagle3,draft-mtp,ngram-simple\n'
            '--spec-draft-n-max N\n',
            engine_id='llama.cpp-mtp',
        )

        self.assertTrue(caps.supports_spec_type)
        self.assertTrue(caps.supports_mtp)
        self.assertIn('draft-mtp', caps.spec_type_values)
        self.assertEqual(caps.mtp_spec_type, 'draft-mtp')
        self.assertEqual(caps.mtp_spec_type_value, 'draft-mtp')
        self.assertTrue(caps.supports_spec_draft_n_max)

    def test_capability_parser_handles_realistic_mtp_spec_type_formats(self):
        cases = [
            ('--spec-type [none|mtp|ngram-simple]\n--spec-draft-n-max N', 'mtp'),
            ('--spec-type VALUE\n  allowed values: none, mtp, draft-mtp, ngram-simple\n--spec-draft-n-max N', 'draft-mtp'),
            ('--spec-type VALUE\n  allowed values: none, draft-mtp\n--spec-draft-n-max N', 'draft-mtp'),
            ('--spec-type VALUE\n  allowed values: none, mtp\n--spec-draft-n-max N', 'mtp'),
            ('--spec-type <none|draft-mtp|ngram-simple>\n--spec-draft-n-max N', 'draft-mtp'),
        ]

        for help_text, expected in cases:
            with self.subTest(expected=expected, help_text=help_text):
                caps = parse_engine_capabilities(help_text, engine_id='llama.cpp-mtp')
                self.assertIn(expected, parse_spec_type_values(help_text))
                self.assertEqual(caps.mtp_spec_type_value, expected)
                self.assertTrue(caps.supports_mtp)
                self.assertTrue(caps.supports_spec_draft_n_max)

    def test_capability_parser_reports_spec_type_without_mtp_values(self):
        caps = parse_engine_capabilities(
            '--spec-type VALUE\n  allowed values: none, ngram-map-k\n--spec-draft-n-max N',
            engine_id='llama.cpp-mtp',
        )

        self.assertTrue(caps.supports_spec_type)
        self.assertFalse(caps.supports_mtp)
        self.assertEqual(caps.mtp_spec_type_value, '')
        self.assertTrue(caps.supports_spec_draft_n_max)

    def test_detect_engine_capabilities_reads_help_from_stderr_even_when_nonzero(self):
        class Result:
            stdout = ''
            stderr = '--spec-type VALUE\nallowed values: none, draft-mtp\n--spec-draft-n-max N'
            returncode = 1

        detect_engine_capabilities.cache_clear()
        try:
            with patch('llama_tui.runtime_profiles.subprocess.run', return_value=Result()):
                caps = detect_engine_capabilities('/opt/mtp/bin/llama-server', 'llama.cpp-mtp')
        finally:
            detect_engine_capabilities.cache_clear()

        self.assertTrue(caps.supports_mtp)
        self.assertEqual(caps.mtp_spec_type_value, 'draft-mtp')

    def test_detect_engine_capabilities_falls_back_to_short_help(self):
        class EmptyResult:
            stdout = ''
            stderr = ''
            returncode = 1

        class HelpResult:
            stdout = '--spec-type VALUE\nallowed values: none, mtp\n--spec-draft-n-max N'
            stderr = ''
            returncode = 0

        calls = []

        def fake_run(args, **_kwargs):
            calls.append(args[-1])
            return EmptyResult() if args[-1] == '--help' else HelpResult()

        detect_engine_capabilities.cache_clear()
        try:
            with patch('llama_tui.runtime_profiles.subprocess.run', side_effect=fake_run):
                caps = detect_engine_capabilities('/opt/mtp/bin/llama-server', 'llama.cpp-mtp')
        finally:
            detect_engine_capabilities.cache_clear()

        self.assertEqual(calls, ['--help', '-h'])
        self.assertTrue(caps.supports_mtp)
        self.assertEqual(caps.mtp_spec_type_value, 'mtp')

    def test_capability_parser_does_not_treat_spec_draft_flags_as_mtp_value(self):
        caps = parse_engine_capabilities(
            '--spec-type none,draft-simple --spec-draft-n-max N --spec-draft-model FILE',
            engine_id='llama.cpp-mtp',
        )

        self.assertTrue(caps.supports_spec_type)
        self.assertFalse(caps.supports_mtp)
        self.assertNotIn('mtp', caps.spec_type_values)
        self.assertNotIn('draft-mtp', caps.spec_type_values)
        self.assertEqual(caps.mtp_spec_type, '')
        self.assertEqual(caps.mtp_spec_type_value, '')

    def test_build_mtp_args_uses_selected_spec_type_and_clamps_draft(self):
        model = ModelConfig(
            id='mtp',
            name='MTP',
            path='/models/native-mtp.gguf',
            alias='mtp',
            port=18080,
            mtp_draft_n_max=1,
        )
        profile = RuntimeProfile(
            engine_id='llama.cpp-mtp',
            ctx_size=2048,
            gpu_layers=13,
            parallel=1,
            mtp_enabled=True,
            mtp_draft_n_max=9,
        )
        caps = replace(
            default_engine_capabilities('llama.cpp-mtp'),
            supports_spec_type=True,
            supports_mtp=True,
            mtp_spec_type='draft-mtp',
            supports_spec_draft_n_max=True,
        )

        args, diagnostics = build_mtp_args(model, profile, caps)

        self.assertEqual(args, ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '3'])
        self.assertTrue(diagnostics.enabled)
        self.assertEqual(diagnostics.selected_spec_type, 'draft-mtp')
        self.assertEqual(diagnostics.draft_n_max, 3)
        self.assertEqual(diagnostics.added_flags, tuple(args))
        self.assertEqual(diagnostics.blocked_reason, '')

    def test_build_mtp_args_reports_missing_capabilities_without_flags(self):
        profile = RuntimeProfile(
            engine_id='llama.cpp-mtp',
            ctx_size=2048,
            gpu_layers=13,
            parallel=1,
            mtp_enabled=True,
            mtp_draft_n_max=2,
        )
        caps = replace(
            default_engine_capabilities('llama.cpp-mtp'),
            supports_spec_type=True,
            supports_mtp=False,
            spec_type_values=('none', 'ngram-simple'),
            supports_spec_draft_n_max=True,
        )

        args, diagnostics = build_mtp_args(None, profile, caps)

        self.assertEqual(args, [])
        self.assertTrue(diagnostics.enabled)
        self.assertEqual(diagnostics.blocked_reason, 'missing mtp/draft-mtp value')
        self.assertIn('--spec-type', diagnostics.skipped_flags)

    def test_build_mtp_args_emits_q8_draft_kv_flags(self):
        profile = RuntimeProfile(
            engine_id='llama.cpp-mtp',
            ctx_size=131072,
            gpu_layers=None,
            parallel=1,
            mtp_enabled=True,
            mtp_draft_n_max=1,
            mtp_draft_kv_preset='q8_0/q8_0',
        )
        caps = replace(
            default_engine_capabilities('llama.cpp-mtp'),
            supports_spec_type=True,
            supports_mtp=True,
            mtp_spec_type='draft-mtp',
            supports_spec_draft_n_max=True,
            supports_spec_draft_type_kv=True,
        )

        args, diagnostics = build_mtp_args(None, profile, caps)

        self.assertEqual(args[args.index('--spec-type') + 1], 'draft-mtp')
        self.assertEqual(args[args.index('--spec-draft-n-max') + 1], '1')
        self.assertEqual(args[args.index('--spec-draft-type-k') + 1], 'q8_0')
        self.assertEqual(args[args.index('--spec-draft-type-v') + 1], 'q8_0')
        self.assertEqual(diagnostics.blocked_reason, '')

    def test_build_mtp_args_works_for_any_mtp_capable_engine(self):
        # MTP is a binary capability: build_mtp_args must emit speculative flags
        # for a plain llama.cpp engine when the binary advertises MTP support.
        profile = RuntimeProfile(
            engine_id='llama.cpp',
            ctx_size=2048,
            gpu_layers=13,
            parallel=1,
            mtp_enabled=True,
            mtp_draft_n_max=2,
        )
        caps = replace(
            default_engine_capabilities('llama.cpp'),
            supports_spec_type=True,
            supports_mtp=True,
            spec_type_values=('none', 'draft-mtp'),
            mtp_spec_type='draft-mtp',
            mtp_spec_type_value='draft-mtp',
            supports_spec_draft_n_max=True,
        )

        args, diagnostics = build_mtp_args(None, profile, caps)

        self.assertEqual(args, ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '2'])
        self.assertTrue(diagnostics.enabled)
        self.assertEqual(diagnostics.selected_spec_type, 'draft-mtp')

    def test_build_mtp_args_skips_when_mtp_disabled(self):
        profile = RuntimeProfile(
            engine_id='llama.cpp',
            ctx_size=2048,
            gpu_layers=13,
            parallel=1,
            mtp_enabled=False,
        )
        caps = replace(
            default_engine_capabilities('llama.cpp'),
            supports_spec_type=True,
            supports_mtp=True,
            mtp_spec_type='draft-mtp',
            supports_spec_draft_n_max=True,
        )

        args, diagnostics = build_mtp_args(None, profile, caps)

        self.assertEqual(args, [])
        self.assertFalse(diagnostics.enabled)
        self.assertIn('mtp_disabled', diagnostics.skipped_flags)

    def test_capability_parser_detects_turboquant_cache_types(self):
        caps = parse_engine_capabilities(
            'KV cache data type for K\nallowed values: f16 q8_0 q4_0 turbo2 turbo3 turbo4\n'
            'KV cache data type for V\nallowed values: f16 q8_0 q4_0 turbo2 turbo3 turbo4\n'
            '--flash-attn on|off|auto -ctk TYPE -ctv TYPE -ngl N --parallel N',
            engine_id='turboquant',
        )

        self.assertTrue(caps.supports_ctk_ctv)
        self.assertIn('turbo4', caps.supported_kv_modes)
        self.assertEqual(caps.gpu_layers_flag, '-ngl')

    def test_capability_parser_detects_llama_cache_type_flags(self):
        caps = parse_engine_capabilities(
            'usage: llama-server --flash-attn on|off|auto --cache-type-k TYPE --cache-type-v TYPE --n-gpu-layers N --parallel N',
            engine_id='llama.cpp',
        )

        self.assertEqual(caps.flash_attn_syntax, 'value')
        self.assertTrue(caps.supports_cache_type_kv)
        self.assertEqual(caps.gpu_layers_flag, '--n-gpu-layers')

    def test_capability_parser_detects_benchmark_relevant_flags(self):
        caps = parse_engine_capabilities(
            'usage: llama-server --chat-template-kwargs JSON --reasoning auto --reasoning-budget N '
            '--reasoning-format deepseek --context-shift --no-context-shift --cache-prompt --cache-reuse N --fit-target R '
            '--top-p P --top-k K --min-p P --repeat-penalty N --presence-penalty N --samplers LIST --seed SEED',
            engine_id='llama.cpp',
        )

        self.assertTrue(caps.supports_chat_template_kwargs)
        self.assertTrue(caps.supports_reasoning)
        self.assertTrue(caps.supports_reasoning_budget)
        self.assertTrue(caps.supports_reasoning_format)
        self.assertTrue(caps.supports_context_shift)
        self.assertTrue(caps.supports_cache_prompt)
        self.assertTrue(caps.supports_cache_reuse)
        self.assertTrue(caps.supports_fit_target)
        self.assertTrue(caps.supports_top_p)
        self.assertTrue(caps.supports_top_k)
        self.assertTrue(caps.supports_min_p)
        self.assertTrue(caps.supports_repeat_penalty)
        self.assertTrue(caps.supports_presence_penalty)
        self.assertTrue(caps.supports_samplers)
        self.assertTrue(caps.supports_seed)

    def test_capability_parser_detects_moe_placement_flags(self):
        caps = parse_engine_capabilities(
            'usage: llama-server --flash-attn on|off|auto -cmoe --cpu-moe -ncmoe N '
            '--n-cpu-moe N -ot TENSOR=CPU --override-tensors TENSOR=CPU -ngl N',
            engine_id='llama.cpp',
        )

        self.assertTrue(caps.supports_cpu_moe)
        self.assertTrue(caps.supports_n_cpu_moe)
        self.assertTrue(caps.supports_override_tensor)
        self.assertEqual(caps.cpu_moe_flag, '-cmoe')
        self.assertEqual(caps.n_cpu_moe_flag, '-ncmoe')
        self.assertEqual(caps.override_tensor_flag, '-ot')

    def test_command_builder_adds_supported_benchmark_profile_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='tiny',
                name='Tiny Reasoning',
                path='/models/tiny.gguf',
                alias='tiny',
                port=18080,
                temp=0.33,
                top_p=0.8,
                top_k=24,
                repeat_penalty=1.08,
                presence_penalty=0.1,
                no_context_shift=True,
                preserve_thinking='on',
                extra_args=['--top-p', '0.1', '--no-context-shift', '--seed', '999'],
                launch_overrides={
                    'min_p': 0.05,
                    'seed': 123,
                    'samplers': 'top_k;top_p',
                    'reasoning': 'auto',
                    'reasoning_budget': 256,
                    'reasoning_format': 'deepseek',
                    'cache_prompt': True,
                    'cache_reuse': 64,
                    'fit_target': '0.85',
                },
            )
            caps = EngineCapabilities(
                flash_attn_syntax='value',
                flash_attn_flag='--flash-attn',
                supports_cache_type_kv=True,
                supports_parallel=True,
                supports_context_shift=True,
                supports_chat_template_kwargs=True,
                supports_reasoning=True,
                supports_reasoning_budget=True,
                supports_reasoning_format=True,
                supports_cache_prompt=True,
                supports_cache_reuse=True,
                supports_fit_target=True,
                supports_top_p=True,
                supports_top_k=True,
                supports_min_p=True,
                supports_repeat_penalty=True,
                supports_presence_penalty=True,
                supports_samplers=True,
                supports_seed=True,
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertEqual(cmd[cmd.index('--temp') + 1], '0.33')
        self.assertEqual(cmd.count('--top-p'), 1)
        self.assertEqual(cmd.count('--no-context-shift'), 1)
        self.assertEqual(cmd.count('--seed'), 1)
        self.assertEqual(cmd[cmd.index('--top-p') + 1], '0.8')
        self.assertEqual(cmd[cmd.index('--top-k') + 1], '24')
        self.assertEqual(cmd[cmd.index('--min-p') + 1], '0.05')
        self.assertEqual(cmd[cmd.index('--repeat-penalty') + 1], '1.08')
        self.assertEqual(cmd[cmd.index('--presence-penalty') + 1], '0.1')
        self.assertIn('--no-context-shift', cmd)
        self.assertIn('--cache-prompt', cmd)
        self.assertEqual(cmd[cmd.index('--cache-reuse') + 1], '64')
        self.assertEqual(cmd[cmd.index('--reasoning') + 1], 'auto')
        self.assertEqual(cmd[cmd.index('--reasoning-budget') + 1], '256')
        self.assertEqual(cmd[cmd.index('--reasoning-format') + 1], 'deepseek')
        self.assertEqual(cmd[cmd.index('-fitt') + 1], '0.85')
        self.assertIn('--chat-template-kwargs', cmd)
        self.assertIn('"preserve_thinking": true', cmd[cmd.index('--chat-template-kwargs') + 1])

    def test_command_builder_omits_unsupported_profile_flags_and_strips_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='tiny',
                name='Tiny',
                path='/models/tiny.gguf',
                alias='tiny',
                port=18080,
                top_p=0.8,
                no_context_shift=True,
                preserve_thinking='on',
                extra_args=['--top-p', '0.1', '--no-context-shift'],
            )
            caps = EngineCapabilities(
                flash_attn_syntax='unsupported',
                supports_cache_type_kv=True,
                supports_parallel=True,
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertIn('--temp', cmd)
        self.assertNotIn('--top-p', cmd)
        self.assertNotIn('--no-context-shift', cmd)
        self.assertNotIn('--chat-template-kwargs', cmd)

    def test_turboquant_command_uses_short_cache_flags_for_manual_safe_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile(
                    'turboquant',
                    'llama-server',
                    kv_key_mode='q8_0',
                    kv_value_mode='turbo4',
                ),
            )
            model = ModelConfig(
                id='qwen',
                name='Qwen Q8_0',
                path='/models/qwen.Q8_0.gguf',
                alias='qwen-tq',
                port=18080,
                turboquant_status='native',
                turboquant_key_dim=128,
                turboquant_value_dim=128,
            )

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('turboquant')):
                cmd = app.build_command(model)

        self.assertEqual(cmd[0], app.runtime_profile.server_command)
        self.assertEqual(cmd[cmd.index('-ctk') + 1], 'q8_0')
        self.assertEqual(cmd[cmd.index('-ctv') + 1], 'turbo4')
        self.assertNotIn('--cache-type-k', cmd)

    def test_mtp_command_emits_spec_flags_and_forces_single_parallel(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='mtp',
                name='MTP',
                path='/models/mtp.gguf',
                alias='mtp',
                port=18080,
                parallel=4,
                supports_mtp='yes',
                mtp_enabled=True,
                mtp_draft_n_max=2,
                extra_args=['--spec-type', 'none', '--spec-draft-n-max', '99'],
            )
            caps = replace(
                default_engine_capabilities('llama.cpp-mtp'),
                supports_spec_type=True,
                supports_mtp=True,
                mtp_spec_type='mtp',
                supports_spec_draft_n_max=True,
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertEqual(cmd[0], app.runtime_profile.server_command)
        self.assertEqual(cmd[cmd.index('--parallel') + 1], '1')
        self.assertEqual(cmd[cmd.index('--spec-type') + 1], 'mtp')
        self.assertEqual(cmd[cmd.index('--spec-draft-n-max') + 1], '2')
        self.assertEqual(cmd.count('--spec-type'), 1)
        self.assertEqual(cmd.count('--spec-draft-n-max'), 1)

    def test_mtp_command_uses_advertised_draft_mtp_spec_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='mtp',
                name='MTP',
                path='/models/mtp.gguf',
                alias='mtp',
                port=18080,
                supports_mtp='yes',
                mtp_enabled=True,
                mtp_draft_n_max=2,
            )
            caps = replace(
                default_engine_capabilities('llama.cpp-mtp'),
                supports_spec_type=True,
                supports_mtp=True,
                mtp_spec_type='draft-mtp',
                supports_spec_draft_n_max=True,
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)
                context = runtime_record_context(app, model)

        self.assertEqual(cmd[cmd.index('--spec-type') + 1], 'draft-mtp')
        self.assertEqual(cmd[cmd.index('--spec-draft-n-max') + 1], '2')
        self.assertEqual(context['spec_type'], 'draft-mtp')

    def test_mtp_launch_validation_blocks_mmproj(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='vision',
                name='Vision',
                path='/models/vision.gguf',
                alias='vision',
                port=18080,
                supports_mtp='yes',
                mtp_enabled=True,
                extra_args=['--mmproj', '/models/mmproj.gguf'],
            )
            caps = replace(
                default_engine_capabilities('llama.cpp-mtp'),
                supports_spec_type=True,
                supports_mtp=True,
                mtp_spec_type='mtp',
                supports_spec_draft_n_max=True,
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                ok, msg = app.validate_mtp_launch(model)

        self.assertFalse(ok)
        self.assertIn('MTP + mmproj/vision', msg)

    def test_mtp_launch_validation_blocks_detected_vision_feature(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='vision',
                name='Vision Model',
                path='/models/native-mtp-vision.gguf',
                alias='vision',
                port=18080,
                supports_mtp='yes',
                mtp_enabled=True,
            )
            caps = replace(
                default_engine_capabilities('llama.cpp-mtp'),
                supports_spec_type=True,
                supports_mtp=True,
                mtp_spec_type='mtp',
                supports_spec_draft_n_max=True,
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                ok, msg = app.validate_mtp_launch(model)

        self.assertFalse(ok)
        self.assertIn('MTP + mmproj/vision', msg)

    def test_mtp_launch_validation_requires_selected_spec_type_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='mtp',
                name='MTP',
                path='/models/native-mtp.gguf',
                alias='mtp',
                port=18080,
                supports_mtp='yes',
                mtp_enabled=True,
            )
            caps = replace(
                default_engine_capabilities('llama.cpp-mtp'),
                help_text='--spec-type none,ngram-simple\n--spec-draft-n-max N',
                supports_spec_type=True,
                supports_mtp=True,
                spec_type_values=('none', 'ngram-simple'),
                mtp_spec_type='',
                mtp_spec_type_value='',
                supports_spec_draft_n_max=True,
            )

            with patch.object(app, 'engine_capabilities', return_value=caps), \
                 patch.object(app, 'mtp_binary_warning', return_value='missing mtp/draft-mtp value'):
                ok, msg = app.validate_mtp_launch(model)

        self.assertFalse(ok)
        self.assertIn('missing mtp/draft-mtp value', msg)

    def test_runtime_profile_emits_moe_placement_flags_and_strips_stale_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='moe',
                name='MoE',
                path='/models/moe.gguf',
                alias='moe',
                port=18080,
                extra_args=['-ncmoe', '8', '--override-tensor', 'old=CPU'],
            )
            profile = RuntimeProfile(
                engine_id='llama.cpp',
                name='n_cpu_moe_32',
                ctx_size=8192,
                gpu_layers=999,
                parallel=1,
                n_cpu_moe=32,
                tensor_overrides=('blk.*ffn=CPU',),
                placement_strategy='n_cpu_moe_32',
            )
            caps = replace(
                default_engine_capabilities('llama.cpp'),
                supports_n_cpu_moe=True,
                supports_override_tensor=True,
                n_cpu_moe_flag='-ncmoe',
                override_tensor_flag='-ot',
                gpu_layers_flag='-ngl',
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model, runtime_profile=profile)

        self.assertIn('-ncmoe', cmd)
        self.assertEqual(cmd[cmd.index('-ncmoe') + 1], '32')
        self.assertIn('-ot', cmd)
        self.assertEqual(cmd[cmd.index('-ot') + 1], 'blk.*ffn=CPU')
        self.assertNotIn('8', cmd)
        self.assertNotIn('old=CPU', cmd)

    def test_cpu_moe_wins_over_n_cpu_moe(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(id='moe', name='MoE', path='/models/moe.gguf', alias='moe', port=18080)
            profile = RuntimeProfile(
                engine_id='llama.cpp',
                name='cpu_moe_all',
                ctx_size=8192,
                gpu_layers=999,
                parallel=1,
                cpu_moe=True,
                n_cpu_moe=32,
                placement_strategy='cpu_moe_all',
            )
            caps = replace(
                default_engine_capabilities('llama.cpp'),
                supports_cpu_moe=True,
                supports_n_cpu_moe=True,
                cpu_moe_flag='-cmoe',
                n_cpu_moe_flag='-ncmoe',
                gpu_layers_flag='-ngl',
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model, runtime_profile=profile)

        self.assertIn('-cmoe', cmd)
        self.assertNotIn('-ncmoe', cmd)

    def test_model_config_round_trips_moe_placement_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'models.json'
            app = AppConfig(config_path)
            app.models = [
                ModelConfig(
                    id='moe',
                    name='MoE',
                    path='/models/moe.gguf',
                    alias='moe',
                    port=18080,
                    moe_placement_strategy='n_cpu_moe_32',
                    cpu_moe=False,
                    n_cpu_moe=32,
                    tensor_overrides=['blk.*ffn=CPU'],
                )
            ]
            app.save()

            loaded = AppConfig(config_path)

        self.assertEqual(loaded.models[0].moe_placement_strategy, 'n_cpu_moe_32')
        self.assertFalse(loaded.models[0].cpu_moe)
        self.assertEqual(loaded.models[0].n_cpu_moe, 32)
        self.assertEqual(loaded.models[0].tensor_overrides, ['blk.*ffn=CPU'])

    def test_model_config_missing_moe_fields_loads_safe_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / 'models.json'
            config_path.write_text(json.dumps({
                'models': [{
                    'id': 'm',
                    'name': 'M',
                    'path': '/models/m.gguf',
                    'alias': 'm',
                    'port': 18080,
                }],
            }), encoding='utf-8')

            loaded = AppConfig(config_path)

        self.assertEqual(loaded.models[0].moe_placement_strategy, '')
        self.assertFalse(loaded.models[0].cpu_moe)
        self.assertEqual(loaded.models[0].n_cpu_moe, 0)
        self.assertEqual(loaded.models[0].tensor_overrides, [])

    def test_turboquant_head_dim_64_forces_q8_baseline_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile(
                    'turboquant',
                    'llama-server',
                    kv_key_mode='q8_0',
                    kv_value_mode='turbo4',
                ),
            )
            model = ModelConfig(
                id='oss',
                name='GPT OSS',
                path='/models/gpt-oss.gguf',
                alias='oss-tq',
                port=18080,
                turboquant_status='incompatible',
                turboquant_head_dim=64,
                turboquant_key_dim=64,
                turboquant_value_dim=64,
            )

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('turboquant')):
                cmd = app.build_command(model)

        self.assertEqual(cmd[cmd.index('-ctk') + 1], 'q8_0')
        self.assertEqual(cmd[cmd.index('-ctv') + 1], 'q8_0')
        self.assertIn('head_dim=64', app.turboquant_session_advisory(model))

    def test_turboquant_binary_warning_for_vanilla_help(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(id='m', name='M', path='/models/m.gguf', alias='m', port=18080)
            caps = EngineCapabilities(
                supports_ctk_ctv=True,
                supports_cache_type_kv=True,
                supported_kv_modes=('f16', 'q8_0', 'q4_0'),
                help_text='allowed values: f16 q8_0 q4_0',
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                warning = app.turboquant_binary_warning(model)

        self.assertIn('does not advertise turbo cache types', warning)

    def test_launch_with_failsafe_starts_measured_profile_with_runtime_replay(self):
        model = ModelConfig(
            id='m',
            name='M',
            path='/models/m.gguf',
            alias='m',
            port=18200,
            measured_profiles={
                'auto': {
                    'status': 'ok',
                    'ctx': 32768,
                    'ctx_per_slot': 32768,
                    'parallel': 1,
                    'ngl': 999,
                    'tokens_per_sec': 20.0,
                    'engine': 'turboquant',
                    'runtime_profile': 'fit_context_growth_sweep_32768_turbo4_turbo4',
                    'kv_preset': 'turbo4/turbo4',
                    'runtime_fit': True,
                    'fit_context': 4096,
                    'runtime_no_warmup': True,
                    'gpu_layers_mode': 'fit',
                    'batch_size': 128,
                    'ubatch_size': 64,
                }
            },
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.started_runtime_profiles = []
                self.saved = []

            def hardware_profile(self, refresh=False):
                return HardwareProfile(memory_available=32 * 1024**3)

            def add_or_update(self, saved):
                self.saved.append(saved)

            def start(self, _model, runtime_profile=None, benchmark_profile=None):
                self.started_runtime_profiles.append(runtime_profile)
                return True, 'started'

            def wait_until_ready(self, _model, timeout=120, cancel_token=None):
                return True, 'ready'

            def stop(self, _model, managed_only=True):
                return True, 'stopped'

        app = FakeApp()
        ok, msg = launch_with_failsafe(app, model, 'best', 'auto')

        self.assertTrue(ok, msg)
        self.assertTrue(app.started_runtime_profiles)
        runtime_profile = app.started_runtime_profiles[0]
        self.assertIsNotNone(runtime_profile)
        self.assertTrue(runtime_profile.fit)
        self.assertIsNone(runtime_profile.gpu_layers)
        self.assertEqual(runtime_profile.fit_context, 4096)

    def test_runtime_artifacts_are_scoped_by_active_engine(self):
        # Audit #7: the legacy 'llama.cpp-mtp' engine collapses into
        # 'llama.cpp', so artefacts for a session started with that
        # alias now share llama.cpp's runtime directory. The other
        # forks keep their own scopes.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = ModelConfig(id='m', name='M', path='/models/m.gguf', alias='m', port=18080)
            with patch('llama_tui.constants.CACHE_DIR', root), \
                    patch('llama_tui.app.CACHE_DIR', root):
                llama_app = AppConfig(root / 'llama.json')
                llama_app.models = [model]
                turboquant_app = AppConfig(
                    root / 'turboquant.json',
                    runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
                )
                turboquant_app.models = [model]
                mtp_app = AppConfig(
                    root / 'mtp.json',
                    runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
                )
                mtp_app.models = [model]

                llama_log = llama_app.logfile(model.id)
                turboquant_log = turboquant_app.logfile(model.id)
                mtp_log = mtp_app.logfile(model.id)
                llama_pid = llama_app.pidfile(model.id)
                turboquant_pid = turboquant_app.pidfile(model.id)
                mtp_pid = mtp_app.pidfile(model.id)

        self.assertNotEqual(llama_log, turboquant_log)
        self.assertEqual(llama_log, root / 'runtime' / 'llama.cpp' / 'm.log')
        self.assertEqual(turboquant_log, root / 'runtime' / 'turboquant' / 'm.log')
        # MTP alias now resolves under the llama.cpp runtime dir.
        self.assertEqual(mtp_log, root / 'runtime' / 'llama.cpp' / 'm.log')
        self.assertEqual(mtp_pid, root / 'runtime' / 'llama.cpp' / 'm.pid')
        self.assertEqual(llama_pid, root / 'runtime' / 'llama.cpp' / 'm.pid')
        self.assertEqual(turboquant_pid, root / 'runtime' / 'turboquant' / 'm.pid')

    def test_llama_command_can_use_supported_q8_cache_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='dense',
                name='Dense',
                path='/models/dense.gguf',
                alias='dense',
                port=18080,
                extra_args=['--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0'],
            )
            caps = EngineCapabilities(
                flash_attn_syntax='value',
                flash_attn_flag='--flash-attn',
                supports_cache_type_kv=True,
                supports_parallel=True,
                gpu_layers_flag='--n-gpu-layers',
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertIn('--flash-attn', cmd)
        self.assertIn('on', cmd)
        self.assertIn('--cache-type-k', cmd)
        self.assertIn('--cache-type-v', cmd)
        self.assertNotIn('-ctk', cmd)

    def test_continue_tool_export_forces_jinja_for_llamacpp_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='dense',
                name='Dense',
                path='/models/dense.gguf',
                alias='dense',
                port=18080,
                jinja=False,
            )
            caps = default_engine_capabilities('llama.cpp')

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertIn('--jinja', cmd)

    def test_continue_tool_export_preserves_template_override_without_chatml_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='dense',
                name='Dense',
                path='/models/dense.gguf',
                alias='dense',
                port=18080,
                jinja=False,
                extra_args=['--chat-template-file', '/models/tool-template.jinja'],
            )
            caps = default_engine_capabilities('llama.cpp')

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model)

        self.assertIn('--jinja', cmd)
        self.assertIn('--chat-template-file', cmd)
        self.assertIn('/models/tool-template.jinja', cmd)
        self.assertNotIn('chatml', cmd)

    def test_turboquant_fit_runtime_profile_omits_fixed_ngl(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(id='dense', name='Dense', path='$MODEL', alias='dense-tq', port=18080)
            profile = RuntimeProfile(
                engine_id='turboquant',
                name='fit_weight_discovery_q8_0_turbo4',
                ctx_size=8192,
                gpu_layers=None,
                parallel=1,
                kv_preset='q8_0/turbo4',
                flash_attn='on',
                fit=True,
                fit_context=4096,
                fit_discovery_phase='weight_fit',
            )

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('turboquant')):
                cmd = app.build_command(model, runtime_profile=profile)

        self.assertEqual(cmd[0], app.runtime_profile.server_command)
        self.assertNotIn('-ngl', cmd)
        self.assertNotIn('--n-gpu-layers', cmd)
        self.assertIn('-fit', cmd)
        self.assertEqual(cmd[cmd.index('-fit') + 1], 'on')
        self.assertIn('-fitc', cmd)
        self.assertEqual(cmd[cmd.index('-fitc') + 1], '4096')
        self.assertEqual(cmd[cmd.index('-ctk') + 1], 'q8_0')
        self.assertEqual(cmd[cmd.index('-ctv') + 1], 'turbo4')

    def test_mtp_fit_q8_nommap_runtime_profile_emits_manual_winner_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(id='mtp', name='MTP', path='$MODEL', alias='mtp', port=18080, threads=6)
            profile = RuntimeProfile(
                engine_id='llama.cpp-mtp',
                name='mtp_fit_q8_draftq8_nommap_draft1_128k',
                ctx_size=131072,
                gpu_layers=None,
                parallel=1,
                kv_preset='q8_0/q8_0',
                flash_attn='on',
                batch_size=128,
                ubatch_size=64,
                fit=True,
                fit_context=4096,
                fit_target=1024,
                cache_ram=0,
                no_mmap=True,
                no_warmup=True,
                mtp_enabled=True,
                mtp_draft_n_max=1,
                mtp_draft_kv_preset='q8_0/q8_0',
            )
            caps = replace(
                default_engine_capabilities('llama.cpp-mtp'),
                supports_spec_type=True,
                supports_mtp=True,
                mtp_spec_type='draft-mtp',
                supports_spec_draft_n_max=True,
                supports_spec_draft_type_kv=True,
                supports_ctk_ctv=True,
                supports_fit=True,
                supports_fit_ctx=True,
                supports_fit_target=True,
                supports_no_mmap=True,
                supports_no_warmup=True,
            )

            with patch.object(app, 'engine_capabilities', return_value=caps):
                cmd = app.build_command(model, runtime_profile=profile)

        self.assertIn('--ctx-size', cmd)
        self.assertEqual(cmd[cmd.index('--ctx-size') + 1], '131072')
        self.assertNotIn('-ngl', cmd)
        self.assertNotIn('--n-gpu-layers', cmd)
        self.assertEqual(cmd[cmd.index('-fit') + 1], 'on')
        self.assertEqual(cmd[cmd.index('-fitt') + 1], '1024')
        self.assertEqual(cmd[cmd.index('-fitc') + 1], '4096')
        self.assertEqual(cmd[cmd.index('-ctk') + 1], 'q8_0')
        self.assertEqual(cmd[cmd.index('-ctv') + 1], 'q8_0')
        self.assertEqual(cmd[cmd.index('--spec-type') + 1], 'draft-mtp')
        self.assertEqual(cmd[cmd.index('--spec-draft-n-max') + 1], '1')
        self.assertEqual(cmd[cmd.index('--spec-draft-type-k') + 1], 'q8_0')
        self.assertEqual(cmd[cmd.index('--spec-draft-type-v') + 1], 'q8_0')
        self.assertEqual(cmd[cmd.index('--cache-ram') + 1], '0')
        self.assertIn('--no-mmap', cmd)
        self.assertIn('--no-warmup', cmd)
        self.assertEqual(cmd[cmd.index('--batch-size') + 1], '128')
        self.assertEqual(cmd[cmd.index('--ubatch-size') + 1], '64')

    def test_failure_classification_names_actionable_startup_errors(self):
        cases = {
            'unknown value for --flash-attn: -ctk': 'CLI_INVALID',
            'cudaMalloc failed: out of memory while loading tensors': 'CUDA_OOM_WEIGHTS',
            'cudaMalloc failed: out of memory allocating KV cache': 'CUDA_OOM_KV',
            'K cache type turbo4 with block size 128 does not divide': 'KV_MODE_INCOMPATIBLE',
            'failed to fit params to free device memory, n_gpu_layers already set by user to 21': 'FIXED_GPU_LAYERS_BLOCKED_FIT',
            'llama_params_fit_impl: projected to use 9879 MiB of device memory vs. 7665 MiB of free device memory; cannot meet free memory target of 1024 MiB': 'MEMORY_FIT_FAILED',
            'failed to allocate buffer for kv cache; failed to create context': 'CUDA_OOM_KV',
            'llama-memory-recurrent.cpp:173: GGML_ASSERT(rollback >= 1 && rollback <= n_rs_seq) failed': 'ENGINE_RUNTIME_CRASH',
            'error while handling argument "--spec-type": unknown speculative type: mtp': 'CLI_INVALID',
            'failed to load model': 'MODEL_LOAD_FAILED',
            'server timed out': 'SERVER_TIMEOUT',
            'request timed out': 'API_TIMEOUT',
            'connection refused': 'PORT_UNREACHABLE',
            'chat template error': 'CHAT_TEMPLATE_ERROR',
        }
        mixed_fixed_fit_oom = (
            'llama_params_fit: failed to fit params to free device memory: '
            'n_gpu_layers already set by user to 21, abort\n'
            'ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory\n'
            'llama_model_load: failed to load model'
        )
        cases[mixed_fixed_fit_oom] = 'FIXED_GPU_LAYERS_BLOCKED_FIT'
        observed_fixed_fit_failure = (
            'llama_params_fit: failed to fit params to free device memory: '
            'n_gpu_layers already set by user to 18, abort\n'
            'ggml_backend_cuda_buffer_type_alloc_buffer: cudaMalloc failed: out of memory\n'
            'llama_model_load_from_file_impl: failed to load model'
        )
        cases[observed_fixed_fit_failure] = 'FIXED_GPU_LAYERS_BLOCKED_FIT'
        observed_weight_oom = (
            'ggml_backend_cuda_buffer_type_alloc_buffer: allocating 512.00 MiB on device 0: '
            'cudaMalloc failed: out of memory\n'
            'alloc_tensor_range: failed to allocate CUDA0 buffer\n'
            'llama_model_load: error loading model: unable to allocate CUDA0 buffer\n'
            'llama_model_load_from_file_impl: failed to load model'
        )
        cases[observed_weight_oom] = 'CUDA_OOM_WEIGHTS'
        for text, expected in cases.items():
            with self.subTest(text=text):
                default = 'API_TIMEOUT' if text == 'request timed out' else 'SERVER_TIMEOUT'
                self.assertEqual(classify_benchmark_failure(text, default)['failure_category'], expected)

        classified = classify_benchmark_failure(observed_weight_oom)
        self.assertIn('cudaMalloc failed', classified['failure_excerpt'])
        self.assertEqual(classified['failure_category'], 'CUDA_OOM_WEIGHTS')

    def test_mtp_runtime_diagnostics_classify_fixed_ngl_and_mmap_slowpath(self):
        fixed = parse_mtp_runtime_diagnostics(
            'failed to fit params to free device memory: n_gpu_layers already set by user'
        )
        mmap = parse_mtp_runtime_diagnostics(
            'tensor overrides to CPU are used with mmap enabled - consider using --no-mmap for better performance\n'
            'llama_model_load: CPU_Mapped model buffer size = 7523.81 MiB\n'
            'llama_model_load: CUDA_Host model buffer size = 128.50 MiB\n'
            'llama_model_load: CUDA0 model buffer size = 5148.50 MiB\n'
            'llama_print_timings: prompt eval time = 1200.00 ms / 512 tokens ( 2.34 ms per token, 426.67 tokens per second)\n'
            'llama_print_timings: eval time = 4096.00 ms / 128 tokens ( 32.00 ms per token, 31.25 tokens per second)\n'
            'draft acceptance rate = 76.71%'
        )

        self.assertEqual(fixed['failure_category'], 'FIXED_GPU_LAYERS_BLOCKED_FIT')
        self.assertIn('fit-assisted', fixed['suggested_fix'])
        self.assertEqual(mmap['warning'], 'MMAP_CPU_OVERRIDE_SLOWPATH')
        self.assertIn('--no-mmap', mmap['recommendation'])
        self.assertEqual(mmap['cpu_model_buffer_type'], 'CUDA_Host')
        self.assertEqual(mmap['cpu_mapped_model_buffer_mib'], 7523.81)
        self.assertEqual(mmap['cuda_host_model_buffer_mib'], 128.50)
        self.assertEqual(mmap['cuda_model_buffer_mib'], 5148.50)
        self.assertEqual(mmap['prompt_eval_tokens_per_sec'], 426.67)
        self.assertEqual(mmap['eval_tokens_per_sec'], 31.25)

    def test_runtime_profile_retry_does_not_repeat_recurrent_engine_crash(self):
        crash = 'llama-memory-recurrent.cpp:173: GGML_ASSERT(rollback >= 1 && rollback <= n_rs_seq) failed'
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='mtp',
                name='MTP',
                path='/models/mtp.gguf',
                alias='mtp',
                runtime='llama.cpp-mtp',
                supports_mtp='yes',
            )
            runtime_profile = RuntimeProfile(
                engine_id='llama.cpp-mtp',
                name='mtp_baseline',
                ctx_size=8192,
                gpu_layers=13,
                parallel=1,
                kv_preset='default',
                flash_attn='on',
                benchmark_strategy_id='mtp_acceptance_matrix',
                benchmark_phase='baseline_no_mtp',
            )
            caps = replace(
                default_engine_capabilities('llama.cpp-mtp'),
                supports_spec_type=True,
                supports_mtp=True,
                mtp_spec_type='draft-mtp',
                supports_spec_draft_n_max=True,
            )
            attempts = []

            def fake_candidate(_app, candidate, objective, *_args, **_kwargs):
                attempts.append(candidate.id)
                record = adaptive_record_from_candidate(candidate, objective, 'not ready', detail=crash)
                record.update(classify_benchmark_failure(crash))
                record['startup_result'] = 'FAILED'
                return record, None

            with patch.object(app, 'engine_capabilities', return_value=caps), \
                patch('llama_tui.benchmark.benchmark_adaptive_candidate', side_effect=fake_candidate):
                ok, broke, records, measured, completed = benchmark_runtime_profile_with_retry(
                    app,
                    model,
                    runtime_profile,
                    'long_context',
                    progress=None,
                    cancel_token=None,
                    completed=0,
                    total=1,
                    max_attempts=2,
                    benchmark_depth='fast',
                )

        self.assertFalse(ok)
        self.assertTrue(broke)
        self.assertEqual(len(records), 1)
        self.assertEqual(completed, 1)
        self.assertEqual(measured, [])
        self.assertEqual(attempts, ['mtp'])
        self.assertEqual(records[0]['failure_category'], 'ENGINE_RUNTIME_CRASH')

    def test_mtp_acceptance_runner_skips_recurrent_baseline_and_runs_drafts(self):
        caps = replace(
            default_engine_capabilities('llama.cpp-mtp'),
            supports_spec_type=True,
            supports_mtp=True,
            supports_no_warmup=True,
            supports_fit=True,
            supports_fit_ctx=True,
            supports_fit_target=True,
            supports_ctk_ctv=True,
            supports_spec_draft_type_kv=True,
            supports_no_mmap=True,
            spec_type_values=('none', 'draft-mtp'),
            mtp_spec_type='draft-mtp',
            mtp_spec_type_value='draft-mtp',
            supports_spec_draft_n_max=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='mtp',
                name='Qwen NextN MTP',
                path='/cache/hub/models--owner--Qwen-NextN-MTP-GGUF/snapshots/abc/model.gguf',
                alias='mtp',
                port=18080,
                runtime='llama.cpp-mtp',
                ctx_min=2048,
                ctx_max=32768,
                supports_mtp='yes',
                output=1024,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=6 * 1024**3)
            strategy = select_benchmark_strategy(
                'llama.cpp-mtp',
                model,
                hardware=hardware,
                capabilities=caps,
                objective='quick_sanity',
                depth='fast',
                model_size_bytes=4 * 1024**3,
            )
            phases = []
            purposes = []
            drafts = []
            progress = []

            def fake_runtime_profile_with_retry(
                _app,
                base_model,
                runtime_profile,
                objective,
                _progress,
                _cancel_token,
                completed,
                _total,
                **kwargs,
            ):
                self.assertEqual(kwargs.get('benchmark_purpose'), 'mtp_acceptance')
                self.assertEqual(kwargs.get('max_attempts'), 1)
                phases.append(runtime_profile.benchmark_phase)
                purposes.append(kwargs.get('benchmark_purpose'))
                candidate = model_for_runtime_profile(base_model, runtime_profile)
                draft = runtime_profile.mtp_draft_n_max
                drafts.append(draft)
                record = adaptive_record_from_candidate(
                    candidate,
                    objective,
                    'ok',
                    tokens_per_sec=10.0 + draft,
                    seconds=1.0,
                    engine='llama.cpp-mtp',
                    benchmark_phase=f'draft_n{draft}',
                    benchmark_strategy_id='mtp_acceptance_matrix',
                    benchmark_purpose='mtp_acceptance',
                    mtp_enabled=True,
                    mtp_draft_n_max=draft,
                    spec_type='draft-mtp',
                    measurement_output=128,
                    runtime_no_warmup=True,
                )
                record['accept_rate'] = 0.5 + (draft / 10.0)
                record['draft_tokens'] = 100 * draft
                record['accepted_tokens'] = 50 * draft
                return True, False, [record], [dict(record)], completed + 1

            with patch.object(app, 'engine_capabilities', return_value=caps), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3), \
                patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_profile_with_retry):
                ok, msg = benchmark_mtp_acceptance_matrix_after_preflight(
                    app,
                    model,
                    hardware,
                    strategy,
                    'run-mtp',
                    'server_fast',
                    '2026-05-14T00:00:00',
                    progress=progress.append,
                    depth='fast',
                )

        self.assertTrue(ok)
        self.assertIn('partial', msg)
        self.assertTrue(phases)
        self.assertTrue(all(phase.startswith('fit_q8_draftq8_draft_n') for phase in phases))
        self.assertEqual({1, 2, 3}, set(drafts))
        self.assertEqual(set(purposes), {'mtp_acceptance'})
        saved = app.models[0]
        self.assertEqual(saved.default_benchmark_status, 'partial')
        self.assertEqual(saved.last_benchmark_results[0]['status'], 'skipped_runtime_assert')
        self.assertEqual(saved.last_benchmark_results[0]['failure_category'], 'BASELINE_NOT_SUPPORTED_FOR_RECURRENT_NEXTN')
        self.assertTrue(all(item.get('measurement_output') == 128 for item in saved.last_benchmark_results if item.get('mtp_enabled')))
        self.assertIn('mtp_acceptance', saved.measured_profiles)
        self.assertIn('mtp_baseline_no_spec', saved.measured_profiles)
        self.assertIn('mtp_fast_chat', saved.measured_profiles)
        self.assertIn('mtp_safe', saved.measured_profiles)
        self.assertEqual(saved.measured_profiles['mtp_acceptance']['mtp_risk_level'], 'good')
        self.assertEqual(saved.measured_profiles['mtp_acceptance']['kind'], 'mtp_optimizer')
        self.assertEqual(saved.measured_profiles['mtp_acceptance']['mtp_spec_type'], 'draft-mtp')
        self.assertEqual(saved.measured_profiles['mtp_baseline_no_spec']['status'], 'skipped_runtime_assert')
        self.assertTrue(any('MTP spec type selected from binary: draft-mtp' in str(item) for item in progress))
        self.assertTrue(any('Baseline no-MTP skipped before launch: recurrent/NextN model requires MTP' in str(item) for item in progress))
        self.assertFalse(any('smart bounded' in str(item) or 'long_context not ready' in str(item) for item in progress))

    def test_mtp_acceptance_runner_filters_all_nextn_baseline_variants_before_launch(self):
        caps = replace(
            default_engine_capabilities('llama.cpp-mtp'),
            supports_spec_type=True,
            supports_mtp=True,
            supports_no_warmup=True,
            spec_type_values=('none', 'draft-mtp'),
            mtp_spec_type='draft-mtp',
            mtp_spec_type_value='draft-mtp',
            supports_spec_draft_n_max=True,
        )
        baseline_names = (
            'mtp_baseline',
            'mtp_baseline_partial_ngl_13',
            'mtp_baseline_cpu_moe_all',
            'mtp_baseline_experts_cpu_override',
        )
        baseline_profiles = [
            RuntimeProfile(
                engine_id='llama.cpp-mtp',
                name=name,
                ctx_size=8192,
                gpu_layers=13,
                parallel=1,
                kv_preset='default',
                benchmark_strategy_id='mtp_acceptance_matrix',
                benchmark_phase='baseline_no_mtp',
            )
            for name in baseline_names
        ]
        draft_profile = RuntimeProfile(
            engine_id='llama.cpp-mtp',
            name='mtp_fit_q8_draftq8_nommap_draft1_128k',
            ctx_size=131072,
            gpu_layers=None,
            parallel=1,
            kv_preset='q8_0/q8_0',
            fit=True,
            fit_context=4096,
            fit_target=1024,
            no_mmap=True,
            no_warmup=True,
            mtp_enabled=True,
            mtp_draft_n_max=1,
            mtp_draft_kv_preset='q8_0/q8_0',
            mtp_spec_type='draft-mtp',
            benchmark_strategy_id='mtp_acceptance_matrix',
            benchmark_phase='fit_q8_draftq8_draft_n1',
        )
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='mtp',
                name='Qwen NextN MTP',
                path='/models/qwen-nextn-mtp.gguf',
                alias='mtp',
                port=18080,
                runtime='llama.cpp-mtp',
                ctx_min=8192,
                ctx_max=131072,
                supports_mtp='yes',
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=6 * 1024**3)
            strategy = select_benchmark_strategy(
                'llama.cpp-mtp',
                model,
                hardware=hardware,
                capabilities=caps,
                objective='quick_sanity',
                depth='fast',
                model_size_bytes=4 * 1024**3,
            )
            launched = []

            def fake_runtime_profile_with_retry(
                _app,
                base_model,
                runtime_profile,
                objective,
                _progress,
                _cancel_token,
                completed,
                _total,
                **kwargs,
            ):
                launched.append(runtime_profile.name)
                self.assertEqual(objective, 'mtp_long_context_probe')
                self.assertEqual(kwargs.get('benchmark_purpose'), 'mtp_acceptance')
                candidate = model_for_runtime_profile(base_model, runtime_profile)
                record = adaptive_record_from_candidate(
                    candidate,
                    objective,
                    'ok',
                    tokens_per_sec=24.0,
                    seconds=1.0,
                    engine='llama.cpp-mtp',
                    benchmark_phase=runtime_profile.benchmark_phase,
                    benchmark_strategy_id='mtp_acceptance_matrix',
                    benchmark_purpose='mtp_acceptance',
                    mtp_enabled=True,
                    mtp_draft_n_max=1,
                    mtp_draft_kv_preset='q8_0/q8_0',
                    spec_type='draft-mtp',
                    runtime_no_warmup=True,
                    no_mmap=True,
                    runtime_fit=True,
                    fit_target=1024,
                )
                record['accept_rate'] = 0.82
                record['draft_tokens'] = 100
                record['accepted_tokens'] = 82
                return True, False, [record], [dict(record)], completed + 1

            with patch.object(app, 'engine_capabilities', return_value=caps), \
                patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=baseline_profiles + [draft_profile]), \
                patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_profile_with_retry):
                ok, msg = benchmark_mtp_acceptance_matrix_after_preflight(
                    app,
                    model,
                    hardware,
                    strategy,
                    'run-mtp',
                    'server_fast',
                    '2026-05-14T00:00:00',
                    depth='fast',
                )

        self.assertTrue(ok)
        self.assertIn('partial', msg)
        self.assertEqual(launched, ['mtp_fit_q8_draftq8_nommap_draft1_128k'])
        self.assertFalse(any(name.startswith('mtp_baseline') for name in launched))
        saved = app.models[0]
        self.assertEqual(saved.last_benchmark_results[0]['status'], 'skipped_runtime_assert')
        self.assertEqual(saved.last_benchmark_results[0]['failure_category'], 'BASELINE_NOT_SUPPORTED_FOR_RECURRENT_NEXTN')
        self.assertIn('mtp_acceptance', saved.measured_profiles)

    def test_mtp_acceptance_fit_timeout_continues_to_remaining_candidates(self):
        caps = replace(
            default_engine_capabilities('llama.cpp-mtp'),
            supports_spec_type=True,
            supports_mtp=True,
            supports_no_warmup=True,
            spec_type_values=('none', 'draft-mtp'),
            mtp_spec_type='draft-mtp',
            mtp_spec_type_value='draft-mtp',
            supports_spec_draft_n_max=True,
        )
        profiles = [
            RuntimeProfile(
                engine_id='llama.cpp-mtp',
                name='mtp_fit_q8_draftq8_nommap_draft1_128k',
                ctx_size=131072,
                gpu_layers=None,
                parallel=1,
                kv_preset='q8_0/q8_0',
                fit=True,
                fit_context=4096,
                fit_target=1024,
                no_mmap=True,
                no_warmup=True,
                mtp_enabled=True,
                mtp_draft_n_max=1,
                mtp_draft_kv_preset='q8_0/q8_0',
                mtp_spec_type='draft-mtp',
                benchmark_strategy_id='mtp_acceptance_matrix',
                benchmark_phase='fit_q8_draftq8_draft_n1',
            ),
            RuntimeProfile(
                engine_id='llama.cpp-mtp',
                name='mtp_fit_q8_draftq8_nommap_draft2_8k',
                ctx_size=8192,
                gpu_layers=None,
                parallel=1,
                kv_preset='q8_0/q8_0',
                fit=True,
                fit_context=4096,
                fit_target=1024,
                no_mmap=True,
                no_warmup=True,
                mtp_enabled=True,
                mtp_draft_n_max=2,
                mtp_draft_kv_preset='q8_0/q8_0',
                mtp_spec_type='draft-mtp',
                benchmark_strategy_id='mtp_acceptance_matrix',
                benchmark_phase='fit_q8_draftq8_draft_n2',
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='mtp',
                name='Qwen NextN MTP',
                path='/models/qwen-nextn-mtp.gguf',
                alias='mtp',
                port=18080,
                runtime='llama.cpp-mtp',
                ctx_min=8192,
                ctx_max=131072,
                supports_mtp='yes',
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=6 * 1024**3)
            strategy = select_benchmark_strategy(
                'llama.cpp-mtp',
                model,
                hardware=hardware,
                capabilities=caps,
                objective='quick_sanity',
                depth='fast',
                model_size_bytes=4 * 1024**3,
            )
            objectives = {}

            def fake_runtime_profile_with_retry(
                _app,
                base_model,
                runtime_profile,
                objective,
                _progress,
                _cancel_token,
                completed,
                _total,
                **_kwargs,
            ):
                objectives[runtime_profile.name] = objective
                candidate = model_for_runtime_profile(base_model, runtime_profile)
                if runtime_profile.name.endswith('128k'):
                    record = adaptive_record_from_candidate(
                        candidate,
                        objective,
                        'benchmark failed',
                        detail='request timed out',
                        engine='llama.cpp-mtp',
                        benchmark_phase=runtime_profile.benchmark_phase,
                        benchmark_strategy_id='mtp_acceptance_matrix',
                        benchmark_purpose='mtp_acceptance',
                        mtp_enabled=True,
                        mtp_draft_n_max=1,
                        mtp_draft_kv_preset='q8_0/q8_0',
                        spec_type='draft-mtp',
                    )
                    record.update(classify_benchmark_failure('request timed out', default_category='API_TIMEOUT'))
                    return False, True, [record], [], completed + 1
                record = adaptive_record_from_candidate(
                    candidate,
                    objective,
                    'ok',
                    tokens_per_sec=30.0,
                    seconds=1.0,
                    engine='llama.cpp-mtp',
                    benchmark_phase=runtime_profile.benchmark_phase,
                    benchmark_strategy_id='mtp_acceptance_matrix',
                    benchmark_purpose='mtp_acceptance',
                    mtp_enabled=True,
                    mtp_draft_n_max=2,
                    mtp_draft_kv_preset='q8_0/q8_0',
                    spec_type='draft-mtp',
                    runtime_fit=True,
                    no_mmap=True,
                )
                record['accept_rate'] = 0.84
                record['draft_tokens'] = 100
                record['accepted_tokens'] = 84
                return True, False, [record], [dict(record)], completed + 1

            with patch.object(app, 'engine_capabilities', return_value=caps), \
                patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=profiles), \
                patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_profile_with_retry):
                ok, msg = benchmark_mtp_acceptance_matrix_after_preflight(
                    app,
                    model,
                    hardware,
                    strategy,
                    'run-mtp',
                    'server_fast',
                    '2026-05-14T00:00:00',
                    depth='fast',
                )

        self.assertTrue(ok)
        self.assertNotIn('baseline failed', msg.lower())
        self.assertEqual(objectives['mtp_fit_q8_draftq8_nommap_draft1_128k'], 'mtp_long_context_probe')
        self.assertEqual(objectives['mtp_fit_q8_draftq8_nommap_draft2_8k'], 'mtp_long_context_probe')
        self.assertEqual(mtp_long_context_probe_request_timeout(131072), 600)
        self.assertEqual(mtp_long_context_probe_request_timeout(32768), 360)
        self.assertEqual(mtp_long_context_probe_request_timeout(8192), 240)
        saved = app.models[0]
        self.assertEqual(saved.default_benchmark_status, 'partial')
        self.assertEqual(saved.measured_profiles['mtp_acceptance']['mtp_draft_n_max'], 2)

    def test_mtp_acceptance_all_fit_timeouts_report_mtp_candidates_timed_out(self):
        caps = replace(
            default_engine_capabilities('llama.cpp-mtp'),
            supports_spec_type=True,
            supports_mtp=True,
            supports_no_warmup=True,
            spec_type_values=('none', 'draft-mtp'),
            mtp_spec_type='draft-mtp',
            mtp_spec_type_value='draft-mtp',
            supports_spec_draft_n_max=True,
        )
        profile_item = RuntimeProfile(
            engine_id='llama.cpp-mtp',
            name='mtp_fit_q8_draftq8_nommap_draft1_128k',
            ctx_size=131072,
            gpu_layers=None,
            parallel=1,
            kv_preset='q8_0/q8_0',
            fit=True,
            fit_context=4096,
            fit_target=1024,
            no_mmap=True,
            no_warmup=True,
            mtp_enabled=True,
            mtp_draft_n_max=1,
            mtp_draft_kv_preset='q8_0/q8_0',
            mtp_spec_type='draft-mtp',
            benchmark_strategy_id='mtp_acceptance_matrix',
            benchmark_phase='fit_q8_draftq8_draft_n1',
        )
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='mtp',
                name='Qwen NextN MTP',
                path='/models/qwen-nextn-mtp.gguf',
                alias='mtp',
                port=18080,
                runtime='llama.cpp-mtp',
                ctx_min=8192,
                ctx_max=131072,
                supports_mtp='yes',
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=6 * 1024**3)
            strategy = select_benchmark_strategy(
                'llama.cpp-mtp',
                model,
                hardware=hardware,
                capabilities=caps,
                objective='quick_sanity',
                depth='fast',
                model_size_bytes=4 * 1024**3,
            )

            def fake_runtime_profile_with_retry(
                _app,
                base_model,
                runtime_profile,
                objective,
                _progress,
                _cancel_token,
                completed,
                _total,
                **_kwargs,
            ):
                candidate = model_for_runtime_profile(base_model, runtime_profile)
                record = adaptive_record_from_candidate(
                    candidate,
                    objective,
                    'benchmark failed',
                    detail='request timed out',
                    engine='llama.cpp-mtp',
                    benchmark_phase=runtime_profile.benchmark_phase,
                    benchmark_strategy_id='mtp_acceptance_matrix',
                    benchmark_purpose='mtp_acceptance',
                    mtp_enabled=True,
                    mtp_draft_n_max=1,
                    mtp_draft_kv_preset='q8_0/q8_0',
                    spec_type='draft-mtp',
                )
                record.update(classify_benchmark_failure('request timed out', default_category='API_TIMEOUT'))
                return False, True, [record], [], completed + 1

            with patch.object(app, 'engine_capabilities', return_value=caps), \
                patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[profile_item]), \
                patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_profile_with_retry):
                ok, msg = benchmark_mtp_acceptance_matrix_after_preflight(
                    app,
                    model,
                    hardware,
                    strategy,
                    'run-mtp',
                    'server_fast',
                    '2026-05-14T00:00:00',
                    depth='fast',
                )

        self.assertFalse(ok)
        self.assertIn('MTP candidates timed out', msg)

    def test_mtp_optimizer_annotations_compare_candidates_to_baseline(self):
        records = [
            {
                'status': 'ok',
                'benchmark_phase': 'baseline_no_mtp',
                'mtp_enabled': False,
                'tokens_per_sec': 10.0,
                'prompt_workload_tokens_per_sec': 100.0,
                'seconds': 10.0,
                'peak_vram_used': 1000,
            },
            {
                'status': 'ok',
                'benchmark_phase': 'draft_n1',
                'mtp_enabled': True,
                'mtp_draft_n_max': 1,
                'tokens_per_sec': 20.0,
                'prompt_workload_tokens_per_sec': 90.0,
                'seconds': 8.0,
                'peak_vram_used': 1300,
                'accept_rate': 0.55,
            },
            {
                'status': 'ok',
                'benchmark_phase': 'draft_n2',
                'mtp_enabled': True,
                'mtp_draft_n_max': 2,
                'tokens_per_sec': 12.0,
                'prompt_workload_tokens_per_sec': 80.0,
                'seconds': 8.0,
                'peak_vram_used': 1200,
                'accept_rate': 0.78,
            },
            {
                'status': 'ok',
                'benchmark_phase': 'draft_n3',
                'mtp_enabled': True,
                'mtp_draft_n_max': 3,
                'tokens_per_sec': 16.0,
                'prompt_workload_tokens_per_sec': 40.0,
                'seconds': 9.0,
                'peak_vram_used': 1400,
                'accept_rate': 0.82,
            },
        ]

        annotated = annotate_mtp_optimizer_records(records, 'draft-mtp')
        draft1 = next(item for item in annotated if item['benchmark_phase'] == 'draft_n1')
        draft2 = next(item for item in annotated if item['benchmark_phase'] == 'draft_n2')
        draft3 = next(item for item in annotated if item['benchmark_phase'] == 'draft_n3')
        recommendations = mtp_optimizer_profile_recommendations(annotated)

        self.assertEqual(draft1['mtp_risk_level'], 'risky')
        self.assertEqual(draft2['decode_gain_vs_baseline'], 1.2)
        self.assertEqual(draft2['prefill_cost_vs_baseline'], 0.2)
        self.assertEqual(draft2['total_wall_gain_vs_baseline'], 1.25)
        self.assertEqual(draft2['memory_delta_vs_baseline'], 200)
        self.assertEqual(draft2['mtp_risk_level'], 'good')
        self.assertEqual(draft3['mtp_risk_level'], 'failed')
        self.assertEqual(recommendations['mtp_fast_chat']['benchmark_phase'], 'draft_n1')
        self.assertEqual(recommendations['mtp_safe']['benchmark_phase'], 'draft_n2')
        self.assertEqual(recommendations['mtp_opencode_ready']['benchmark_phase'], 'draft_n2')
        self.assertIn('mtp_baseline_no_spec', recommendations)

    def test_mtp_optimizer_keeps_prompt_regression_out_of_opencode_profile(self):
        records = [
            {
                'status': 'ok',
                'benchmark_phase': 'baseline_no_mtp',
                'mtp_enabled': False,
                'tokens_per_sec': 10.0,
                'prompt_workload_tokens_per_sec': 100.0,
                'seconds': 10.0,
            },
            {
                'status': 'ok',
                'benchmark_phase': 'draft_n1',
                'mtp_enabled': True,
                'mtp_draft_n_max': 1,
                'tokens_per_sec': 12.0,
                'prompt_workload_tokens_per_sec': 90.0,
                'seconds': 8.0,
                'accept_rate': 0.78,
            },
            {
                'status': 'ok',
                'benchmark_phase': 'draft_n3',
                'mtp_enabled': True,
                'mtp_draft_n_max': 3,
                'tokens_per_sec': 30.0,
                'prompt_workload_tokens_per_sec': 60.0,
                'seconds': 6.0,
                'accept_rate': 0.84,
            },
        ]

        annotated = annotate_mtp_optimizer_records(records, 'draft-mtp')
        recommendations = mtp_optimizer_profile_recommendations(annotated)

        self.assertEqual(recommendations['mtp_fast_chat']['benchmark_phase'], 'draft_n3')
        self.assertEqual(recommendations['mtp_safe']['benchmark_phase'], 'draft_n1')
        self.assertEqual(recommendations['mtp_opencode_ready']['benchmark_phase'], 'draft_n1')

    def test_mtp_optimizer_does_not_save_opencode_without_ok_baseline(self):
        records = [
            {
                'status': 'skipped_runtime_assert',
                'benchmark_phase': 'baseline_no_mtp',
                'mtp_enabled': False,
            },
            {
                'status': 'ok',
                'benchmark_phase': 'draft_n2',
                'mtp_enabled': True,
                'mtp_draft_n_max': 2,
                'tokens_per_sec': 20.0,
                'prompt_workload_tokens_per_sec': 80.0,
                'seconds': 5.0,
                'accept_rate': 0.82,
            },
        ]

        annotated = annotate_mtp_optimizer_records(records, 'draft-mtp')
        recommendations = mtp_optimizer_profile_recommendations(annotated)

        self.assertIn('mtp_fast_chat', recommendations)
        self.assertIn('mtp_safe', recommendations)
        self.assertNotIn('mtp_opencode_ready', recommendations)

    def test_mtp_optimizer_does_not_promote_decode_regression(self):
        records = [
            {
                'status': 'ok',
                'benchmark_phase': 'baseline_no_mtp',
                'mtp_enabled': False,
                'tokens_per_sec': 20.0,
                'prompt_workload_tokens_per_sec': 100.0,
                'seconds': 10.0,
            },
            {
                'status': 'ok',
                'benchmark_phase': 'draft_n2',
                'mtp_enabled': True,
                'mtp_draft_n_max': 2,
                'tokens_per_sec': 19.0,
                'prompt_workload_tokens_per_sec': 100.0,
                'seconds': 8.0,
                'accept_rate': 0.80,
            },
        ]

        annotated = annotate_mtp_optimizer_records(records, 'draft-mtp')
        recommendations = mtp_optimizer_profile_recommendations(annotated)

        draft = next(item for item in annotated if item['benchmark_phase'] == 'draft_n2')
        self.assertEqual(draft['decode_gain_vs_baseline'], 0.95)
        self.assertNotIn('mtp_fast_chat', recommendations)
        self.assertNotIn('mtp_safe', recommendations)
        self.assertNotIn('mtp_opencode_ready', recommendations)

    def test_mtp_optimizer_low_acceptance_promotes_no_mtp_winners(self):
        records = [
            {
                'status': 'ok',
                'benchmark_phase': 'baseline_no_mtp',
                'mtp_enabled': False,
                'tokens_per_sec': 10.0,
                'prompt_workload_tokens_per_sec': 100.0,
                'seconds': 10.0,
            },
            {
                'status': 'ok',
                'benchmark_phase': 'draft_n1',
                'mtp_enabled': True,
                'mtp_draft_n_max': 1,
                'tokens_per_sec': 30.0,
                'prompt_workload_tokens_per_sec': 100.0,
                'seconds': 5.0,
                'accept_rate': 0.55,
            },
        ]

        annotated = annotate_mtp_optimizer_records(records, 'draft-mtp')
        recommendations = mtp_optimizer_profile_recommendations(annotated)

        draft = next(item for item in annotated if item['benchmark_phase'] == 'draft_n1')
        self.assertEqual(draft['mtp_risk_level'], 'risky')
        self.assertIn('mtp_fast_chat', recommendations)
        self.assertNotIn('mtp_safe', recommendations)

    def test_mtp_optimizer_prompt_regression_blocks_safe_and_opencode_profiles(self):
        records = [
            {
                'status': 'ok',
                'benchmark_phase': 'baseline_no_mtp',
                'mtp_enabled': False,
                'tokens_per_sec': 10.0,
                'prompt_workload_tokens_per_sec': 100.0,
                'seconds': 10.0,
            },
            {
                'status': 'ok',
                'benchmark_phase': 'draft_n3',
                'mtp_enabled': True,
                'mtp_draft_n_max': 3,
                'tokens_per_sec': 20.0,
                'prompt_workload_tokens_per_sec': 65.0,
                'seconds': 5.0,
                'accept_rate': 0.84,
            },
        ]

        annotated = annotate_mtp_optimizer_records(records, 'draft-mtp')
        recommendations = mtp_optimizer_profile_recommendations(annotated)

        draft = next(item for item in annotated if item['benchmark_phase'] == 'draft_n3')
        self.assertEqual(draft['prefill_cost_vs_baseline'], 0.35)
        self.assertEqual(recommendations['mtp_fast_chat']['benchmark_phase'], 'draft_n3')
        self.assertNotIn('mtp_safe', recommendations)
        self.assertNotIn('mtp_opencode_ready', recommendations)

    def test_mtp_optimizer_selects_manual_fit_q8_nommap_draft1_winner(self):
        records = [
            {
                'status': 'ok',
                'benchmark_phase': 'fit_q8_draftq8_draft_n1',
                'runtime_profile': 'mtp_fit_q8_draftq8_nommap_draft1_128k',
                'ctx': 131072,
                'mtp_enabled': True,
                'mtp_draft_n_max': 1,
                'kv_preset': 'q8_0/q8_0',
                'mtp_draft_kv_preset': 'q8_0/q8_0',
                'no_mmap': True,
                'tokens_per_sec': 31.25,
                'prompt_workload_tokens_per_sec': 100.0,
                'seconds': 5.0,
                'accept_rate': 0.7671,
            },
            {
                'status': 'ok',
                'benchmark_phase': 'fit_q8_draftq8_draft_n2',
                'runtime_profile': 'mtp_fit_q8_draftq8_mmap_draft2_128k',
                'ctx': 131072,
                'mtp_enabled': True,
                'mtp_draft_n_max': 2,
                'kv_preset': 'q8_0/q8_0',
                'mtp_draft_kv_preset': 'q8_0/q8_0',
                'tokens_per_sec': 26.91,
                'prompt_workload_tokens_per_sec': 100.0,
                'seconds': 5.0,
                'accept_rate': 0.6453,
            },
            {
                'status': 'ok',
                'benchmark_phase': 'fit_q8_draftq8_draft_n3',
                'runtime_profile': 'mtp_fit_q8_draftq8_mmap_draft3_128k',
                'ctx': 131072,
                'mtp_enabled': True,
                'mtp_draft_n_max': 3,
                'kv_preset': 'q8_0/q8_0',
                'mtp_draft_kv_preset': 'q8_0/q8_0',
                'tokens_per_sec': 23.27,
                'prompt_workload_tokens_per_sec': 100.0,
                'seconds': 5.0,
                'accept_rate': 0.527,
            },
        ]

        annotated = annotate_mtp_optimizer_records(records, 'draft-mtp')
        recommendations = mtp_optimizer_profile_recommendations(annotated)
        draft3 = next(item for item in annotated if item['mtp_draft_n_max'] == 3)

        self.assertEqual(draft3['mtp_risk_level'], 'failed')
        self.assertEqual(recommendations['mtp_safe']['mtp_draft_n_max'], 1)
        self.assertEqual(recommendations['mtp_long_context']['runtime_profile'], 'mtp_fit_q8_draftq8_nommap_draft1_128k')
        self.assertEqual(recommendations['mtp_fast_chat']['mtp_draft_n_max'], 1)

    def test_memory_guardrail_admission_skips_critical_memory(self):
        model = ModelConfig(id='m', name='M', path='/models/model.gguf', alias='m', port=18200, ctx=65536)
        profile = HardwareProfile(
            memory_total=16 * 1024**3,
            memory_available=512 * 1024**2,
            gpu_memory_total=8 * 1024**3,
            gpu_memory_free=7 * 1024**3,
        )

        decision = memory_guardrail_admission(profile, model, estimated_safe_ctx=32768)

        self.assertTrue(decision.should_skip)
        self.assertEqual(decision.status, 'memory_guardrail_skipped')
        self.assertIn('RAM available', decision.reason)

    def test_runtime_memory_pruning_skips_same_or_larger_risky_shapes(self):
        failed_profile = RuntimeProfile(
            engine_id='llama.cpp',
            name='context_growth_sweep_32768',
            ctx_size=32768,
            gpu_layers=30,
            parallel=1,
            kv_preset='q8_0/q8_0',
        )
        failed = {'failure_category': 'CUDA_OOM_KV'}
        key = runtime_profile_memory_disable_key(failed, failed_profile)
        larger_same_shape = RuntimeProfile(
            engine_id='llama.cpp',
            name='context_growth_sweep_65536',
            ctx_size=65536,
            gpu_layers=30,
            parallel=1,
            kv_preset='q8_0/q8_0',
        )
        safer_default = RuntimeProfile(
            engine_id='llama.cpp',
            name='context_growth_sweep_65536_default',
            ctx_size=65536,
            gpu_layers=30,
            parallel=1,
            kv_preset='default',
        )

        self.assertTrue(runtime_profile_memory_skip_reason(larger_same_shape, {key}))
        self.assertEqual(runtime_profile_memory_skip_reason(safer_default, {key}), '')

    def test_adaptive_candidate_watchdog_stops_managed_candidate_on_critical_memory(self):
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=8192, ctx_min=2048, ctx_max=8192)

        class FakeApp:
            def __init__(self):
                self.samples = 0
                self.stops = 0

            def hardware_profile(self, refresh=False):
                self.samples += 1
                if self.samples == 1:
                    return HardwareProfile(
                        memory_total=16 * 1024**3,
                        memory_available=12 * 1024**3,
                        gpu_memory_total=8 * 1024**3,
                        gpu_memory_free=7 * 1024**3,
                    )
                return HardwareProfile(
                    memory_total=16 * 1024**3,
                    memory_available=512 * 1024**2,
                    gpu_memory_total=8 * 1024**3,
                    gpu_memory_free=7 * 1024**3,
                )

            def build_command(self, _model, runtime_profile=None, benchmark_profile=None):
                return ['llama-server']

            def start(self, _model, runtime_profile=None, benchmark_profile=None):
                return True, 'started'

            def wait_until_ready(self, _model, timeout=180, cancel_token=None):
                time.sleep(1.2)
                return False, 'not ready'

            def stop(self, _model, managed_only=True):
                self.stops += 1
                return True, 'stopped'

        app = FakeApp()
        record, measured = benchmark_adaptive_candidate(app, model, 'long_context', None, None)

        self.assertIsNone(measured)
        self.assertEqual(record['failure_category'], 'MEMORY_GUARDRAIL')
        self.assertEqual(record['memory_guardrail_status'], 'memory_guardrail_stopped')
        self.assertGreaterEqual(app.stops, 1)

    def test_turboquant_head_dim_64_profiles_use_baseline_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(
                id='oss',
                name='GPT OSS Q4_K_M',
                path='/models/gpt-oss-Q4_K_M.gguf',
                alias='oss',
                port=18080,
                turboquant_status='incompatible',
                turboquant_head_dim=64,
                turboquant_key_dim=64,
                turboquant_value_dim=64,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=turboquant_no_fit_capabilities()), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        self.assertTrue(profiles)
        self.assertTrue(all(item.engine_id == 'turboquant' for item in profiles))
        self.assertEqual({item.kv_preset for item in profiles}, {'q8_0/q8_0'})

    def test_turboquant_unknown_head_dim_avoids_auto_turbo(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(
                id='unknown',
                name='Unknown Q4_K_M',
                path='/models/unknown-Q4_K_M.gguf',
                alias='unknown',
                port=18080,
                turboquant_status='unknown',
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=turboquant_no_fit_capabilities()), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        self.assertEqual({item.kv_preset for item in profiles}, {'q8_0/q8_0'})

    def test_turboquant_fit_profiles_discover_weight_fit_before_context_growth(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(
                id='dense',
                name='Dense Q8_0',
                path='/models/dense-Q8_0.gguf',
                alias='dense',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                turboquant_status='native',
                turboquant_head_dim=128,
                turboquant_key_dim=128,
                turboquant_value_dim=128,
                ctx_max=262144,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('turboquant')), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3), \
                patch('llama_tui.benchmark.current_process_pressure_payload', return_value={}):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        self.assertTrue(profiles)
        self.assertEqual(profiles[0].fit_discovery_phase, 'weight_fit')
        self.assertTrue(profiles[0].fit)
        self.assertIsNone(profiles[0].gpu_layers)
        self.assertEqual(profiles[0].ctx_size, 8192)
        self.assertEqual(profiles[0].kv_preset, 'q8_0/turbo4')
        growth = profiles[1:]
        self.assertTrue(growth)
        self.assertTrue(all(item.fit_discovery_phase == 'context_growth' for item in growth))
        self.assertTrue(all(item.fit and item.gpu_layers is None for item in growth))
        self.assertFalse(any(item.ctx_size == 262144 for item in growth))
        growth_presets = [item.kv_preset for item in growth]
        self.assertIn('q8_0/turbo4', growth_presets)
        self.assertIn('q8_0/turbo3', growth_presets)
        self.assertLess(
            max(index for index, preset in enumerate(growth_presets) if preset == 'q8_0/turbo4'),
            min(index for index, preset in enumerate(growth_presets) if preset == 'q8_0/turbo3'),
        )

    def test_turboquant_q8_profiles_include_safe_balanced_and_symmetric(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(
                id='dense',
                name='Dense Q8_0',
                path='/models/dense-Q8_0.gguf',
                alias='dense',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                turboquant_status='native',
                turboquant_head_dim=128,
                turboquant_key_dim=128,
                turboquant_value_dim=128,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=turboquant_no_fit_capabilities()), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        presets = {item.kv_preset for item in profiles}
        self.assertIn('q8_0/q8_0', presets)
        self.assertIn('q8_0/turbo4', presets)
        self.assertIn('q8_0/turbo3', presets)
        self.assertIn('q8_0/turbo2', presets)
        self.assertIn('turbo4/turbo4', presets)
        self.assertIn('turbo3/turbo3', presets)

    def test_turboquant_low_bit_profiles_keep_symmetric_manual_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(
                id='sensitive',
                name='Sensitive Q4_K_M',
                path='/models/sensitive-Q4_K_M.gguf',
                alias='sensitive',
                port=18080,
                architecture='unknown',
                architecture_type='dense',
                turboquant_status='native',
                turboquant_head_dim=128,
                turboquant_key_dim=128,
                turboquant_value_dim=128,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=turboquant_no_fit_capabilities()), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        presets = {item.kv_preset for item in profiles}
        self.assertIn('q8_0/turbo4', presets)
        self.assertIn('q8_0/turbo3', presets)
        self.assertNotIn('turbo4/turbo4', presets)
        self.assertNotIn('turbo3/turbo3', presets)

    def test_turboquant_low_bit_family_name_does_not_enable_symmetric(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('turboquant', 'llama-server'),
            )
            model = ModelConfig(
                id='renamed',
                name='Renamed Q4_K_M',
                path='/models/renamed-Q4_K_M.gguf',
                alias='renamed',
                port=18080,
                architecture='custom',
                model_family='custom',
                architecture_type='dense',
                turboquant_status='native',
                turboquant_head_dim=128,
                turboquant_key_dim=128,
                turboquant_value_dim=128,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=turboquant_no_fit_capabilities()), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        self.assertNotIn('turbo3/turbo3', {item.kv_preset for item in profiles})

    def test_turboquant_runtime_record_context_includes_cache_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile(
                    'turboquant',
                    'llama-server',
                    kv_key_mode='q8_0',
                    kv_value_mode='turbo4',
                ),
            )
            model = ModelConfig(
                id='dense',
                name='Dense Q8_0',
                path='/models/dense-Q8_0.gguf',
                alias='dense',
                port=18080,
                architecture='llama',
                model_family='llama',
                turboquant_status='native',
                turboquant_head_dim=128,
                turboquant_key_dim=128,
                turboquant_value_dim=128,
            )
            caps = default_engine_capabilities('turboquant')

            with patch.object(app, 'engine_capabilities', return_value=caps):
                context = runtime_record_context(app, model)

        self.assertEqual(context['engine'], 'turboquant')
        self.assertEqual(context['ctk'], 'q8_0')
        self.assertEqual(context['ctv'], 'turbo4')
        self.assertEqual(context['detected_head_dim'], 128)
        self.assertEqual(context['model_quant'], 'Q8_0')
        self.assertEqual(context['model_family'], 'llama')
        self.assertEqual(context['binary_path'], app.runtime_profile.server_command)
        self.assertIn('turbo4', context['help_supported_cache_types'])

    def test_runtime_record_context_includes_launch_profile_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='dense',
                name='Dense',
                path='/models/dense.gguf',
                alias='dense',
                port=18080,
                output=4096,
            )
            profile = build_benchmark_launch_profile(model, purpose='serve_default', depth='fast')

            with patch.object(app, 'engine_capabilities', return_value=EngineCapabilities()):
                context = runtime_record_context(app, model, benchmark_profile=profile)

        self.assertEqual(context['benchmark_profile'], 'serve_default')
        self.assertEqual(context['benchmark_purpose'], 'serve_default')
        self.assertEqual(context['measurement_output'], 512)
        self.assertEqual(context['ctx'], model.ctx)
        self.assertEqual(context['temp'], model.temp)

    def test_benchmark_completion_payload_uses_launch_profile_sampling(self):
        model = ModelConfig(
            id='tiny',
            name='Tiny',
            path='/models/tiny.gguf',
            alias='tiny',
            port=18080,
            output=4096,
            temp=0.31,
            top_p=0.82,
            top_k=17,
            repeat_penalty=1.04,
            presence_penalty=0.2,
            launch_overrides={'min_p': 0.03, 'seed': 99},
        )
        profile = build_benchmark_launch_profile(model, purpose='serve_default', depth='fast')

        with patch(
            'llama_tui.benchmark.post_json',
            return_value={
                'usage': {'completion_tokens': 12, 'prompt_tokens': 8},
                'choices': [{'message': {'content': 'done'}}],
            },
        ) as post:
            ok, result = benchmark_completion(model, launch_profile=profile)

        payload = post.call_args.args[1]
        self.assertTrue(ok)
        self.assertGreater(result['tokens_per_sec'], 0)
        self.assertEqual(payload['temperature'], 0.31)
        self.assertEqual(payload['max_tokens'], 512)
        self.assertEqual(payload['top_p'], 0.82)
        self.assertEqual(payload['top_k'], 17)
        self.assertEqual(payload['repeat_penalty'], 1.04)
        self.assertEqual(payload['presence_penalty'], 0.2)
        self.assertEqual(payload['min_p'], 0.03)
        self.assertEqual(payload['seed'], 99)

    def test_benchmark_completion_suite_accepts_custom_prompts(self):
        model = ModelConfig(id='tiny', name='Tiny', path='/models/tiny.gguf', alias='tiny', port=18080)
        prompts = ['decode prompt', 'prompt-heavy prompt']
        seen = []

        def fake_completion(_model, **kwargs):
            seen.append(kwargs.get('prompt'))
            return True, {
                'elapsed': 0.5,
                'completion_tokens': 10,
                'prompt_tokens': 20,
                'tokens_per_sec': 20.0,
                'text': f'ok {len(seen)}',
            }

        with patch('llama_tui.benchmark.benchmark_completion', side_effect=fake_completion):
            ok, result = benchmark_completion_suite(model, prompts=prompts, max_tokens=32)

        self.assertTrue(ok)
        self.assertEqual(seen, prompts)
        self.assertEqual(result['sample_count'], 2)
        self.assertEqual(result['completion_tokens'], 20)
        self.assertEqual(result['prompt_tokens_per_sec'], 40.0)
        self.assertEqual(result['texts'], ['ok 1', 'ok 2'])

    def test_mtp_optimizer_workloads_measure_decode_and_prompt_heavy(self):
        model = ModelConfig(id='mtp', name='MTP', path='/models/native-mtp.gguf', alias='mtp', port=18080)
        profile = build_benchmark_launch_profile(model, purpose='mtp_acceptance', depth='fast')
        calls = []

        def fake_suite(_model, **kwargs):
            max_tokens = kwargs.get('max_tokens')
            prompts = tuple(kwargs.get('prompts') or ())
            calls.append((max_tokens, len(prompts)))
            if max_tokens == 128:
                return True, {
                    'elapsed': 2.0,
                    'completion_tokens': 64,
                    'prompt_tokens': 40,
                    'tokens_per_sec': 32.0,
                    'sample_count': 2,
                    'sample_tokens_per_sec': [31.0, 33.0],
                    'texts': ['decode ok'],
                    'error': '',
                }
            return True, {
                'elapsed': 4.0,
                'completion_tokens': 24,
                'prompt_tokens': 480,
                'tokens_per_sec': 6.0,
                'sample_count': 1,
                'sample_tokens_per_sec': [6.0],
                'texts': ['prompt ok'],
                'error': '',
            }

        with patch('llama_tui.benchmark.benchmark_completion_suite', side_effect=fake_suite):
            ok, result = benchmark_mtp_optimizer_workloads(model, profile, depth='fast')

        self.assertTrue(ok)
        self.assertEqual([item[0] for item in calls], [128, 96])
        self.assertEqual(set(result['mtp_workloads']), {'decode_heavy', 'prompt_heavy'})
        self.assertEqual(result['tokens_per_sec'], 33.0)
        self.assertEqual(result['mtp_workloads']['decode_heavy']['first_sample_tokens_per_sec'], 31.0)
        self.assertEqual(result['mtp_workloads']['decode_heavy']['steady_sample_tokens_per_sec'], 33.0)
        self.assertEqual(result['prompt_workload_tokens_per_sec'], 120.0)
        self.assertNotIn('prompt_tokens_per_sec', result)
        self.assertEqual(result['mtp_workloads']['prompt_heavy']['prompt_workload_tokens_per_sec'], 120.0)
        self.assertNotIn('prompt_tokens_per_sec', result['mtp_workloads']['prompt_heavy'])
        self.assertEqual(result['elapsed'], 6.0)
        self.assertEqual(result['completion_tokens'], 88)

    def test_mtp_optimizer_workloads_reject_empty_output(self):
        model = ModelConfig(id='mtp', name='MTP', path='/models/native-mtp.gguf', alias='mtp', port=18080)
        profile = build_benchmark_launch_profile(model, purpose='mtp_acceptance', depth='fast')

        def fake_suite(_model, **_kwargs):
            return True, {
                'elapsed': 1.0,
                'completion_tokens': 0,
                'prompt_tokens': 50,
                'tokens_per_sec': 0.0,
                'sample_count': 1,
                'texts': [''],
                'error': '',
            }

        with patch('llama_tui.benchmark.benchmark_completion_suite', side_effect=fake_suite):
            ok, result = benchmark_mtp_optimizer_workloads(model, profile, depth='fast')

        self.assertFalse(ok)
        self.assertIn('empty benchmark output', result['error'])
        self.assertIn('decode_heavy', result['mtp_workloads'])

    def test_adaptive_mtp_acceptance_record_persists_workload_metrics(self):
        model = ModelConfig(
            id='mtp',
            name='MTP',
            path='/models/native-mtp.gguf',
            alias='mtp',
            port=18080,
            runtime='llama.cpp-mtp',
            supports_mtp='yes',
        )
        runtime_profile = RuntimeProfile(
            engine_id='llama.cpp-mtp',
            name='mtp_draft_n2',
            ctx_size=2048,
            gpu_layers=13,
            parallel=1,
            kv_preset='default',
            mtp_enabled=True,
            mtp_draft_n_max=2,
            benchmark_phase='draft_n2',
            benchmark_strategy_id='mtp_acceptance_matrix',
        )

        class FakeApp:
            def __init__(self):
                self.stops = 0

            def hardware_profile(self, refresh=False):
                return HardwareProfile(
                    memory_total=16 * 1024**3,
                    memory_available=12 * 1024**3,
                    gpu_memory_total=8 * 1024**3,
                    gpu_memory_free=6 * 1024**3,
                )

            def engine_capabilities(self):
                return default_engine_capabilities('llama.cpp-mtp')

            def build_command(self, _model, runtime_profile=None, benchmark_profile=None):
                return ['llama-server', '--spec-type', 'draft-mtp', '--spec-draft-n-max', '2']

            def start(self, _model, runtime_profile=None, benchmark_profile=None):
                return True, 'started'

            def wait_until_ready(self, _model, timeout=180, cancel_token=None):
                return True, 'ready'

            def stop(self, _model, managed_only=True):
                self.stops += 1
                return True, 'stopped'

        def fake_warmup(*_args, **_kwargs):
            return True, {
                'elapsed': 0.1,
                'completion_tokens': 1,
                'prompt_tokens': 1,
                'tokens_per_sec': 10.0,
                'text': 'warm',
            }

        def fake_suite(_model, **kwargs):
            max_tokens = kwargs.get('max_tokens')
            if max_tokens == 128:
                return True, {
                    'elapsed': 2.0,
                    'completion_tokens': 64,
                    'prompt_tokens': 40,
                    'tokens_per_sec': 32.0,
                    'sample_count': 2,
                    'texts': ['decode ok'],
                    'error': '',
                }
            return True, {
                'elapsed': 4.0,
                'completion_tokens': 24,
                'prompt_tokens': 480,
                'tokens_per_sec': 6.0,
                'sample_count': 1,
                'texts': ['prompt ok'],
                'error': '',
            }

        app = FakeApp()
        with patch('llama_tui.benchmark.benchmark_completion', side_effect=fake_warmup), \
            patch('llama_tui.benchmark.benchmark_completion_suite', side_effect=fake_suite), \
            patch('llama_tui.benchmark.start_memory_guardrail_watchdog', return_value=(None, None)):
            record, measured = benchmark_adaptive_candidate(
                app,
                model,
                'quick_sanity',
                None,
                None,
                runtime_profile=runtime_profile,
                benchmark_purpose='mtp_acceptance',
                benchmark_depth='fast',
            )

        self.assertEqual(record['status'], 'ok')
        self.assertIsNotNone(measured)
        self.assertEqual(record['tokens_per_sec'], 32.0)
        self.assertEqual(record['prompt_tokens_per_sec'], 0.0)
        self.assertEqual(record['prompt_workload_tokens_per_sec'], 120.0)
        self.assertEqual(record['seconds'], 6.0)
        self.assertEqual(set(record['mtp_workloads']), {'decode_heavy', 'prompt_heavy'})
        self.assertNotIn('prompt_tokens_per_sec', record['mtp_workloads']['prompt_heavy'])
        self.assertEqual(record['mtp_workloads']['prompt_heavy']['completion_tokens'], 24)
        self.assertGreaterEqual(app.stops, 1)

    def test_raw_speed_benchmark_records_history_without_updating_measured_profiles(self):
        original = ModelConfig(
            id='tiny',
            name='Tiny',
            path='/models/tiny.gguf',
            alias='tiny',
            port=18080,
            measured_profiles={'auto': {'tokens_per_sec': 9.0}},
            last_benchmark_tokens_per_sec=9.0,
        )

        class FakeApp:
            models = [original]

            def __init__(self):
                self.saved = None

            def hardware_profile(self, refresh=False):
                return HardwareProfile(memory_available=16 * 1024**3)

            def runtime_profile_from_model(self, *_args, **_kwargs):
                return None

            def engine_capabilities(self):
                return EngineCapabilities()

            def add_or_update(self, saved):
                self.saved = saved

        record = {
            'status': 'ok',
            'objective': 'raw_speed',
            'tokens_per_sec': 33.0,
            'benchmark_profile': 'raw_speed',
            'benchmark_purpose': 'raw_speed',
            'measurement_output': 512,
            'engine': 'llama.cpp',
        }
        app = FakeApp()

        with patch('llama_tui.benchmark.benchmark_preflight_cleanup', return_value=(True, 'clean')), \
            patch('llama_tui.benchmark.benchmark_adaptive_candidate', return_value=(record, dict(record))) as runner:
            ok, msg = benchmark_raw_speed_profile(app, original)

        self.assertTrue(ok)
        self.assertIn('raw speed benchmark saved', msg)
        self.assertEqual(app.saved.measured_profiles, original.measured_profiles)
        self.assertEqual(app.saved.last_benchmark_tokens_per_sec, original.last_benchmark_tokens_per_sec)
        self.assertEqual(app.saved.benchmark_runs[0]['kind'], 'raw_speed')
        self.assertEqual(app.saved.benchmark_runs[0]['benchmark_profiles'], ['raw_speed'])
        self.assertEqual(runner.call_args.kwargs['benchmark_purpose'], 'raw_speed')

    def test_fit_discovery_metadata_persists_on_measured_profiles(self):
        candidate = ModelConfig(id='dense', name='Dense', path='/models/dense.gguf', alias='dense', port=18080, ctx=8192)
        record = adaptive_record_from_candidate(
            candidate,
            'long_context',
            'ok',
            engine='turboquant',
            runtime_profile='fit_weight_discovery_q8_0_turbo4',
            kv_preset='q8_0/turbo4',
            runtime_fit=True,
            fit_context=4096,
            fit_discovery_phase='weight_fit',
            viable_ngl=28,
            viable_ngl_source='offloaded_layers',
            fit_selected_ngl=28,
            fit_selected_ngl_source='offloaded_layers',
            fit_log_excerpt='offloaded 28/33 layers to GPU',
            runtime_log_path='/tmp/runtime/turboquant/dense.log',
        )
        model = ModelConfig(
            id='dense',
            name='Dense',
            path='/models/dense.gguf',
            alias='dense',
            port=18080,
            measured_profiles={'opencode_ready': record},
        )

        profile = measured_profile_runtime_profile(model, 'opencode_ready')

        self.assertEqual(record['fit_discovery_phase'], 'weight_fit')
        self.assertEqual(record['viable_ngl'], 28)
        self.assertIsNotNone(profile)
        self.assertEqual(profile.fit_discovery_phase, 'weight_fit')
        self.assertEqual(profile.viable_ngl, 28)
        self.assertEqual(profile.viable_ngl_source, 'offloaded_layers')
        self.assertEqual(profile.fit_selected_ngl, 28)
        self.assertEqual(profile.fit_log_excerpt, 'offloaded 28/33 layers to GPU')

    def test_benchmark_run_summary_mentions_saved_winners_despite_failures(self):
        ok = {'status': 'ok', 'tokens_per_sec': 24.0, 'ctx': 8192, 'ctx_per_slot': 8192}
        failed = {'status': 'not ready', 'failure_category': 'CUDA_OOM_WEIGHTS'}

        summary = benchmark_run_summary({'auto': ok}, [failed, ok])

        self.assertIn('auto=8192 ctx', summary)
        self.assertIn('1 candidate failure(s), winners saved', summary)

    def test_capability_parser_extracts_turbo_kv_modes_from_allowed_values(self):
        caps = parse_engine_capabilities(
            'usage: llama-server --flash-attn on|off|auto -ctk TYPE -ctv TYPE\n'
            'allowed values: f32, f16, q8_0, turbo4, turbo3, turbo2\n'
            '--parallel N -ngl N',
            engine_id='turboquant',
        )

        self.assertIn('turbo4', caps.supported_kv_modes)
        self.assertIn('turbo3', caps.supported_kv_modes)
        self.assertIn('turbo2', caps.supported_kv_modes)

    def test_turbokv_scoring_prefers_safe_mode_when_context_and_speed_match(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx_max=32768)
        safe = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=16384, parallel=1)
        aggressive = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=16384, parallel=1)
        measured = [
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'long_context', 'model': safe, 'tokens_per_sec': 30.0, 'ctx_per_slot': 16384, 'parallel': 1, 'kv_preset': 'turbo4/turbo4', 'kv_score_penalty': 0.0},
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'long_context', 'model': aggressive, 'tokens_per_sec': 30.0, 'ctx_per_slot': 16384, 'parallel': 1, 'kv_preset': 'turbo2_tcq/turbo2_tcq', 'kv_score_penalty': 0.10},
        ]

        winners = select_measured_profiles(model, measured, profile)

        self.assertEqual(winners['long_context']['kv_preset'], 'turbo4/turbo4')
        self.assertEqual(winners['auto']['kv_preset'], 'turbo4/turbo4')

    def test_turbokv_scoring_allows_more_aggressive_mode_for_material_context_gain(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=64 * 1024**3, memory_available=48 * 1024**3)
        model = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx_max=32768)
        safe = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=8192, parallel=1)
        aggressive = ModelConfig(id='m', name='M', path=__file__, alias='m', port=18200, ctx=32768, parallel=1)
        measured = [
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'long_context', 'model': safe, 'tokens_per_sec': 35.0, 'ctx_per_slot': 8192, 'parallel': 1, 'kv_preset': 'turbo4/turbo4', 'kv_score_penalty': 0.0},
            {'status': 'ok', 'measurement_type': 'full', 'objective': 'long_context', 'model': aggressive, 'tokens_per_sec': 30.0, 'ctx_per_slot': 32768, 'parallel': 1, 'kv_preset': 'turbo3_tcq/turbo2_tcq', 'kv_score_penalty': 0.06},
        ]

        winners = select_measured_profiles(model, measured, profile)

        self.assertEqual(winners['long_context']['kv_preset'], 'turbo3_tcq/turbo2_tcq')
        self.assertEqual(winners['opencode_ready']['kv_preset'], 'turbo3_tcq/turbo2_tcq')

    def test_profile_frontier_keeps_distinct_usage_winners(self):
        profile = HardwareProfile(
            cpu_logical=8,
            cpu_physical=4,
            memory_total=64 * 1024**3,
            memory_available=48 * 1024**3,
            gpu_memory_total=8 * 1024**3,
            gpu_memory_free=6 * 1024**3,
        )
        model = ModelConfig(
            id='m',
            name='MoE',
            path='/models/moe.gguf',
            alias='m',
            port=18200,
            architecture_type='moe',
            ctx_max=131072,
        )

        def measured_row(ctx, objective, tps, runtime_profile, *, gpu_free=2 * 1024**3, n_cpu_moe=0, cpu_moe=False, fit=False):
            candidate = ModelConfig(
                id='m',
                name='MoE',
                path='/models/moe.gguf',
                alias='m',
                port=18200,
                architecture_type='moe',
                ctx=ctx,
                parallel=1,
                n_cpu_moe=n_cpu_moe,
                cpu_moe=cpu_moe,
            )
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=tps,
                seconds=1.0,
                ram_available=32 * 1024**3,
                gpu_memory_free=gpu_free,
                engine='llama.cpp',
                runtime_profile=runtime_profile,
                kv_preset='q8_0/q8_0',
                runtime_fit=fit,
                fit_context=4096 if fit else 0,
                placement_strategy='cpu_moe_all' if cpu_moe else f'n_cpu_moe_{n_cpu_moe}' if n_cpu_moe else '',
                cpu_moe=cpu_moe,
                n_cpu_moe=n_cpu_moe,
            )
            record['model'] = candidate
            record['measurement_type'] = 'full'
            return record

        measured = [
            measured_row(8192, 'fast_chat', 60.0, 'fast_chat_ncpu30', n_cpu_moe=30),
            measured_row(32768, 'long_context', 36.0, 'balanced_ncpu34', n_cpu_moe=34),
            measured_row(65536, 'long_context', 22.0, 'hermes_ncpu38', n_cpu_moe=38),
            measured_row(131072, 'long_context', 9.0, 'max_fit_cpu', gpu_free=256 * 1024**2, cpu_moe=True, fit=True),
        ]
        winners = select_measured_profiles(model, measured, profile)

        frontier = build_profile_frontier(model, measured, winners, profile, generated_at='2026-05-09T00:00:00')

        categories = frontier['categories']
        self.assertEqual(categories['fastest_usable']['tokens_per_sec'], 60.0)
        self.assertEqual(categories['best_balanced']['source_profile_key'], 'auto')
        self.assertEqual(categories['hermes_ready']['ctx_per_slot'], 65536)
        self.assertEqual(categories['highest_stable_context']['ctx_per_slot'], 65536)
        self.assertEqual(categories['max_context_experimental']['ctx_per_slot'], 131072)
        self.assertEqual(categories['max_context_experimental']['status'], 'experimental')
        self.assertFalse(categories['max_context_experimental']['default_eligible'])
        self.assertEqual(categories['max_context_experimental']['fit_mode'], 'fit_assisted')
        self.assertEqual(categories['max_context_experimental']['moe_placement']['strategy'], 'cpu_moe_all')
        self.assertIn('profile_classes', frontier)
        self.assertEqual(frontier['stable_profile'], frontier['profile_classes']['stable_profile']['profile_id'])
        self.assertEqual(frontier['adaptive_profile'], 'long_context_fit_assisted_moe_cpu')
        self.assertEqual(frontier['profile_classes']['adaptive_profile']['profile_class'], 'adaptive')
        self.assertTrue(frontier['pareto'])
        self.assertTrue(all('stability_rating' in item for item in frontier['pareto']))

    def test_profile_frontier_stores_stable_and_fit_assisted_profile_classes(self):
        profile = HardwareProfile(
            cpu_logical=8,
            cpu_physical=4,
            memory_total=64 * 1024**3,
            memory_available=48 * 1024**3,
            gpu_memory_total=8 * 1024**3,
            gpu_memory_free=6 * 1024**3,
        )
        model = ModelConfig(
            id='m',
            name='MoE',
            path='/models/moe.gguf',
            alias='m',
            port=18200,
            architecture_type='moe',
            ctx_max=65536,
        )

        def measured_row(ctx, tps, runtime_profile, fit=False):
            candidate = ModelConfig(
                id='m',
                name='MoE',
                path='/models/moe.gguf',
                alias='m',
                port=18200,
                architecture_type='moe',
                ctx=ctx,
                parallel=1,
                n_cpu_moe=34,
            )
            record = adaptive_record_from_candidate(
                candidate,
                'long_context',
                'ok',
                tokens_per_sec=tps,
                seconds=1.0,
                ram_available=32 * 1024**3,
                gpu_memory_free=2 * 1024**3,
                engine='llama.cpp',
                runtime_profile=runtime_profile,
                kv_preset='q8_0/q8_0',
                runtime_fit=fit,
                fit_context=4096 if fit else 0,
                placement_strategy='n_cpu_moe_34',
                n_cpu_moe=34,
            )
            record['model'] = candidate
            record['measurement_type'] = 'full'
            return record

        measured = [
            measured_row(32768, 31.0, 'long_context_locked_moe_ncpu34'),
            measured_row(65536, 23.0, 'long_context_fit_assisted_moe_ncpu34', fit=True),
        ]

        frontier = build_profile_frontier(model, measured, {}, profile, generated_at='2026-05-09T00:00:00')

        stable = frontier['profile_classes']['stable_profile']
        adaptive = frontier['profile_classes']['adaptive_profile']
        self.assertEqual(frontier['stable_profile'], 'long_context_locked_moe_ncpu34')
        self.assertEqual(frontier['adaptive_profile'], 'long_context_fit_assisted_moe_ncpu34')
        self.assertEqual(stable['profile_class'], 'stable')
        self.assertEqual(stable['fit_mode'], 'locked_moe')
        self.assertTrue(stable['default_eligible'])
        self.assertEqual(adaptive['profile_class'], 'adaptive')
        self.assertEqual(adaptive['fit_mode'], 'fit_assisted')
        self.assertFalse(adaptive['default_eligible'])
        self.assertIn('Fit-assisted', frontier['mode_explanations']['adaptive'])

    def test_max_context_probe_generates_bounded_turboquant_targets(self):
        model = ModelConfig(
            id='m',
            name='TurboQuant',
            path='/models/model-Q8_0.gguf',
            alias='m',
            port=18200,
            runtime='llama.cpp',
            ctx=8192,
            ctx_min=8192,
            ctx_max=262144,
            ngl=999,
            turboquant_head_dim=128,
            turboquant_key_dim=128,
            turboquant_value_dim=128,
        )
        hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

        class FakeApp:
            def active_engine_key_for_model(self, _model):
                return 'turboquant'

            def engine_capabilities(self):
                return default_engine_capabilities('turboquant')

            def runtime_profile_from_model(self, _model, ctx, parallel, ngl):
                return RuntimeProfile(
                    engine_id='turboquant',
                    name='base',
                    ctx_size=ctx,
                    gpu_layers=ngl,
                    parallel=parallel,
                    kv_preset='q8_0/q8_0',
                )

        profiles = max_context_probe_runtime_profiles(FakeApp(), model, hardware)

        self.assertLessEqual(len(profiles), 15)
        self.assertEqual(profiles[0].ctx_size, 131072)
        self.assertEqual(
            [item.kv_preset for item in profiles[:3]],
            ['q8_0/turbo4', 'q8_0/turbo3', 'q8_0/turbo2'],
        )
        self.assertTrue(all(item.name.startswith('max_context_probe_') for item in profiles))
        self.assertTrue(all(item.benchmark_depth == 'full' for item in profiles))

    def test_max_context_probe_selection_separates_safe_from_experimental(self):
        hardware = HardwareProfile(
            gpu_memory_total=8 * 1024**3,
            gpu_memory_free=7 * 1024**3,
            memory_total=64 * 1024**3,
            memory_available=48 * 1024**3,
        )
        model = ModelConfig(id='m', name='M', path='/models/model.gguf', alias='m', port=18200, ctx_max=262144)

        def measured_row(ctx, headroom, tps):
            candidate = ModelConfig(
                id='m',
                name='M',
                path='/models/model.gguf',
                alias='m',
                port=18200,
                ctx=ctx,
                parallel=1,
                ngl=999,
            )
            record = adaptive_record_from_candidate(
                candidate,
                'max_context_probe',
                'ok',
                tokens_per_sec=tps,
                seconds=1.0,
                ram_available=32 * 1024**3,
                gpu_memory_free=headroom,
                engine='turboquant',
                runtime_profile=f'max_context_probe_{ctx}_q8_0_turbo4',
                benchmark_purpose='max_context_probe',
                kv_preset='q8_0/turbo4',
            )
            record['model'] = candidate
            record['measurement_type'] = 'full'
            return record

        measured = [
            measured_row(131072, 2 * 1024**3, 20.0),
            measured_row(196608, 256 * 1024**2, 12.0),
        ]

        winners = select_max_context_probe_profiles(model, measured, hardware)

        self.assertEqual(winners['max_context_safe']['ctx_per_slot'], 131072)
        self.assertEqual(winners['max_context_safe']['safety_class'], 'safe')
        self.assertTrue(winners['max_context_safe']['default_eligible'])
        self.assertEqual(winners['max_context_experimental']['ctx_per_slot'], 196608)
        self.assertEqual(winners['max_context_experimental']['safety_class'], 'experimental')
        self.assertFalse(winners['max_context_experimental']['default_eligible'])

    def test_max_context_probe_persists_winners_without_applying_config(self):
        model = ModelConfig(
            id='m',
            name='M',
            path='/models/model.gguf',
            alias='m',
            port=18200,
            ctx=8192,
            ctx_max=196608,
            ngl=999,
        )
        hardware = HardwareProfile(
            gpu_memory_total=8 * 1024**3,
            gpu_memory_free=7 * 1024**3,
            memory_total=64 * 1024**3,
            memory_available=48 * 1024**3,
        )
        profiles = [
            RuntimeProfile(
                engine_id='turboquant',
                name='max_context_probe_131072_q8_0_turbo4',
                ctx_size=131072,
                gpu_layers=999,
                parallel=1,
                kv_preset='q8_0/turbo4',
                benchmark_depth='full',
            ),
            RuntimeProfile(
                engine_id='turboquant',
                name='max_context_probe_196608_q8_0_turbo3',
                ctx_size=196608,
                gpu_layers=999,
                parallel=1,
                kv_preset='q8_0/turbo3',
                benchmark_depth='full',
            ),
        ]

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.models = [model]
                self.saved = []

            def active_engine_key_for_model(self, _model):
                return 'turboquant'

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model, discover=True, managed_only=False):
                return None

            def hardware_profile(self, refresh=False):
                return hardware

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, saved):
                self.saved.append(saved)

        def fake_runtime_benchmark(_app, base_model, profile, objective, _progress, _cancel_token, completed, total, **kwargs):
            candidate = model_for_runtime_profile(base_model, profile)
            headroom = 2 * 1024**3 if profile.ctx_size == 131072 else 256 * 1024**2
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=18.0 if profile.ctx_size == 131072 else 10.0,
                seconds=1.0,
                ram_available=32 * 1024**3,
                gpu_memory_free=headroom,
                engine=profile.engine_id,
                runtime_profile=profile.name,
                benchmark_purpose=kwargs.get('benchmark_purpose', ''),
                kv_preset=profile.kv_preset,
                benchmark_depth=kwargs.get('benchmark_depth', ''),
            )
            measured = dict(record)
            measured['model'] = candidate
            measured['measurement_type'] = 'full'
            return True, False, [record], [measured], completed + 1

        app = FakeApp()
        with patch('llama_tui.benchmark.max_context_probe_runtime_profiles', return_value=profiles), \
             patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
            ok, msg = benchmark_max_context_probe(app, model)

        self.assertTrue(ok, msg)
        saved = app.saved[-1]
        self.assertEqual(saved.ctx, 8192)
        self.assertIn('max_context_safe', saved.measured_profiles)
        self.assertIn('max_context_experimental', saved.measured_profiles)
        self.assertEqual(saved.measured_profiles['max_context_safe']['ctx_per_slot'], 131072)
        self.assertEqual(saved.measured_profiles['max_context_experimental']['ctx_per_slot'], 196608)
        self.assertFalse(saved.measured_profiles['max_context_experimental']['default_eligible'])
        self.assertEqual(saved.benchmark_runs[0]['kind'], 'max_context_probe')
        self.assertIn('max_context_experimental', saved.benchmark_runs[0]['winners'])

    def test_workflow_cache_ram_candidate_expansion_is_bounded_and_non_destructive(self):
        model = ModelConfig(
            id='m',
            name='M',
            path='/models/model.gguf',
            alias='m',
            port=18200,
            ctx=32768,
            parallel=1,
            cache_ram=0,
        )
        hardware = HardwareProfile(memory_total=32 * 1024**3, memory_available=16 * 1024**3)
        candidates = [('opencode_ready', 'measured', model, 'measured opencode_ready')]

        expanded = expand_workflow_cache_ram_candidates(candidates, hardware)

        self.assertEqual([item[2].cache_ram for item in expanded], [0, 512, 1024, 2048])
        self.assertEqual(model.cache_ram, 0)
        self.assertEqual(expanded[0][0], 'opencode_ready')
        self.assertIn('cache_ram_512', expanded[1][0])
        self.assertIn('cache_ram=2048 MiB', expanded[-1][3])

    def test_workflow_cache_ram_selection_uses_time_stability_and_overhead(self):
        base = {
            'score': 1.0,
            'passed': 2,
            'tasks': 2,
            'status': 'ok',
            'ctx': 32768,
            'ctx_per_slot': 32768,
        }
        faster = dict(base, seconds=90.0, cache_ram=512, cache_ram_mib=512, preset='auto', tier='cache_ram_512')
        slower = dict(base, seconds=140.0, cache_ram=0, cache_ram_mib=0, preset='auto', tier='cache_ram_0')
        unstable = dict(base, seconds=20.0, cache_ram=0, cache_ram_mib=0, timeout_type='idle', preset='auto', tier='timeout')

        winner = max([slower, faster, unstable], key=workflow_cache_ram_selection_key)
        profile = workflow_cache_ram_profile_from_record('opencode', winner)

        self.assertIs(winner, faster)
        self.assertEqual(profile['kind'], 'workflow_cache_ram')
        self.assertEqual(profile['workflow'], 'opencode')
        self.assertEqual(profile['cache_ram_mib'], 512)
        self.assertEqual(profile['status'], 'ok')
        self.assertIn('completion time', profile['selection_reason'])

    def test_runtime_profile_explicit_threads_override_wins(self):
        # The thread-sweep phase sets runtime_profile.threads directly; that
        # explicit value must take precedence over the model's persisted threads.
        model = ModelConfig(
            id='m',
            name='MoE',
            path='/models/moe.gguf',
            alias='m',
            port=18201,
            architecture_type='moe',
            expert_count=256,
            threads=8,
        )
        runtime_profile = RuntimeProfile(
            engine_id='llama.cpp',
            name='experts_cpu_override_threads12',
            ctx_size=2048,
            gpu_layers=999,
            parallel=1,
            tensor_overrides=('.*ffn_.*_exps.*=CPU',),
            threads=12,
        )

        candidate = model_for_runtime_profile(model, runtime_profile)

        self.assertEqual(candidate.threads, 12)

    def test_benchmark_deadline_caps_candidate_timeouts(self):
        deadline = BenchmarkDeadline.from_end(time.monotonic() + 5.0)

        self.assertLessEqual(deadline.cap_timeout(240), 5.0)
        self.assertGreater(deadline.cap_timeout(240), 0.0)

        expired = BenchmarkDeadline.from_end(time.monotonic() - 1.0)
        self.assertEqual(expired.cap_timeout(240), 0.0)

    def test_missing_adaptive_profiles_are_explicitly_skipped(self):
        winners = {
            'fast_chat': {'status': 'ok', 'tokens_per_sec': 20.0},
            'auto': {'status': 'ok', 'tokens_per_sec': 18.0},
        }

        filled = fill_missing_adaptive_profiles(
            winners,
            'No valid candidate measured before benchmark budget expired',
        )

        self.assertEqual(filled['long_context']['status'], 'skipped_budget')
        self.assertIn('budget expired', filled['long_context']['reason'])
        self.assertEqual(filled['opencode_ready']['status'], 'skipped_budget')

    def test_stale_running_benchmark_runs_are_closed_before_new_run(self):
        model = ModelConfig(id='m', name='M', path='/models/model.gguf', alias='m', port=18200)
        model.default_benchmark_status = 'running'
        model.benchmark_runs = [
            {
                'id': 'server-older',
                'kind': 'server',
                'status': 'running',
                'started_at': '2026-05-06T13:48:10',
                'records': [],
                'winners': {},
                'summary': 'raw pre-search',
            }
        ]

        changed = close_stale_running_benchmark_runs(model, '2026-05-06T14:10:00')

        self.assertTrue(changed)
        self.assertEqual(model.default_benchmark_status, 'aborted')
        self.assertEqual(model.benchmark_runs[0]['status'], 'aborted')
        self.assertEqual(model.benchmark_runs[0]['stale_reason'], 'aborted_stale_previous_run')
        self.assertIn('aborted_stale_previous_run', model.benchmark_runs[0]['summary'])

    def test_runtime_profile_runners_use_expected_depth_and_attempts(self):
        model = ModelConfig(id='m', name='M', path='/models/model.gguf', alias='m', port=18200)
        hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        runtime_profile = RuntimeProfile(
            engine_id='turboquant',
            name='kv_compression_probe',
            ctx_size=8192,
            gpu_layers=20,
            parallel=1,
            kv_preset='turbo4/turbo4',
            benchmark_depth='fast',
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model):
                return None

            def hardware_profile(self, refresh=False):
                return hardware

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, model):
                self.saved.append(model)

        def fake_runtime_benchmark(_app, base_model, profile, objective, _progress, _cancel_token, completed, total, **kwargs):
            candidate = ModelConfig(id=base_model.id, name=base_model.name, path=base_model.path, alias=base_model.alias, port=base_model.port, ctx=profile.ctx_size, parallel=profile.parallel, ngl=profile.gpu_layers)
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=25.0,
                seconds=1.0,
                runtime_profile=profile.name,
                kv_preset=profile.kv_preset,
                benchmark_depth=kwargs.get('benchmark_depth', ''),
            )
            record['variant'] = profile.name
            measured = dict(record)
            measured['model'] = candidate
            return True, False, [record], [measured], completed + 1

        app = FakeApp()
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[runtime_profile]):
            with patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark) as runner:
                ok, msg = benchmark_fast_profiles(app, model)

        self.assertTrue(ok, msg)
        self.assertEqual(runner.call_args.kwargs['max_attempts'], 1)
        self.assertEqual(runner.call_args.kwargs['benchmark_depth'], 'fast')

        app = FakeApp()
        runtime_profile = RuntimeProfile(
            engine_id='turboquant',
            name='kv_compression_probe',
            ctx_size=8192,
            gpu_layers=20,
            parallel=1,
            kv_preset='turbo4/turbo4',
            benchmark_depth='full',
        )
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[runtime_profile]):
            with patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark) as runner:
                ok, msg = benchmark_exhaustive_profiles(app, model)

        self.assertTrue(ok, msg)
        self.assertEqual(runner.call_args.kwargs['max_attempts'], 2)
        self.assertEqual(runner.call_args.kwargs['benchmark_depth'], 'full')

    def test_fast_benchmark_persists_profile_frontier(self):
        model = ModelConfig(id='m', name='M', path='/models/model.gguf', alias='m', port=18200, ctx_max=65536)
        hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        fast_profile = RuntimeProfile(
            engine_id='llama.cpp',
            name='fast_chat_probe',
            ctx_size=8192,
            gpu_layers=999,
            parallel=1,
            kv_preset='q8_0/q8_0',
            benchmark_depth='fast',
        )
        long_profile = RuntimeProfile(
            engine_id='llama.cpp',
            name='long_context_probe',
            ctx_size=65536,
            gpu_layers=999,
            parallel=1,
            kv_preset='q8_0/q8_0',
            benchmark_depth='fast',
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model, discover=True, managed_only=False):
                return None

            def hardware_profile(self, refresh=False):
                return hardware

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, saved):
                self.saved.append(saved)

        def fake_runtime_benchmark(_app, base_model, profile, objective, _progress, _cancel_token, completed, _total, **kwargs):
            candidate = ModelConfig(
                id=base_model.id,
                name=base_model.name,
                path=base_model.path,
                alias=base_model.alias,
                port=base_model.port,
                ctx=profile.ctx_size,
                parallel=profile.parallel,
                ngl=profile.gpu_layers or 0,
            )
            tps = 50.0 if profile.ctx_size <= 8192 else 20.0
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=tps,
                seconds=1.0,
                ram_available=24 * 1024**3,
                gpu_memory_free=4 * 1024**3,
                engine=profile.engine_id,
                runtime_profile=profile.name,
                kv_preset=profile.kv_preset,
                benchmark_depth=kwargs.get('benchmark_depth', ''),
            )
            measured = dict(record)
            measured['model'] = candidate
            return True, False, [record], [measured], completed + 1

        app = FakeApp()
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[fast_profile, long_profile]), \
             patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
            ok, msg = benchmark_fast_profiles(app, model)

        self.assertTrue(ok, msg)
        saved = app.saved[-1]
        self.assertIn('profile_frontier', saved.measured_profiles)
        self.assertIn('profile_frontier', saved.benchmark_runs[0]['winners'])
        frontier = saved.measured_profiles['profile_frontier']
        self.assertEqual(frontier['categories']['fastest_usable']['runtime_profile'], 'fast_chat_probe')
        self.assertEqual(frontier['categories']['highest_stable_context']['ctx_per_slot'], 65536)

    def test_failed_fast_runtime_profile_run_preserves_previous_working_winners(self):
        profile = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        old_auto = {
            'status': 'ok',
            'tokens_per_sec': 22.0,
            'seconds': 1.0,
            'ctx': 8192,
            'ctx_per_slot': 8192,
            'parallel': 1,
        }
        model = ModelConfig(
            id='m',
            name='M',
            path='/models/model.gguf',
            alias='m',
            port=18200,
            measured_profiles={
                'auto': dict(old_auto),
                'fast_chat': dict(old_auto),
                'long_context': dict(old_auto),
                'opencode_ready': dict(old_auto),
            },
            default_benchmark_status='done',
            last_benchmark_tokens_per_sec=22.0,
        )
        runtime_profile = RuntimeProfile(
            engine_id='turboquant',
            name='fit_turbokv_probe',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='turbo4/turbo4',
            kv_family='turbo',
            fit=True,
            fit_context=4096,
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self, stored):
                self.models = [stored]
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model, discover=True, managed_only=False):
                return None

            def get_model(self, model_id):
                return next((item for item in self.models if item.id == model_id), None)

            def hardware_profile(self, refresh=False):
                return profile

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, saved):
                self.models[0] = saved
                self.saved.append(saved)

        failed = adaptive_record_from_candidate(
            model,
            'long_context',
            'not ready',
            detail='K cache type turbo4 with block size 128 does not divide n_embd_head_k=192',
            failure_category='KV_MODE_INCOMPATIBLE',
            failure_reason='K cache type turbo4 with block size 128 does not divide n_embd_head_k=192',
        )

        def fake_runtime_benchmark(_app, _base_model, _profile, _objective, _progress, _cancel_token, completed, _total, **_kwargs):
            return False, True, [dict(failed)], [], completed + 1

        app = FakeApp(model)
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[runtime_profile]), \
             patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
            ok, msg = benchmark_fast_profiles(app, model)

        self.assertFalse(ok)
        self.assertIn('kept previous working measured profiles', msg)
        self.assertEqual(app.saved[-1].default_benchmark_status, 'done')
        self.assertEqual(app.saved[-1].measured_profiles['auto']['tokens_per_sec'], 22.0)
        self.assertEqual(app.saved[-1].last_benchmark_tokens_per_sec, 22.0)

    def test_fast_runner_skips_remaining_turbokv_after_shape_incompatibility(self):
        model = ModelConfig(id='m', name='M', path='/models/model.gguf', alias='m', port=18200)
        hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        turbo4 = RuntimeProfile(
            engine_id='turboquant',
            name='fit_turbokv_probe',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='turbo4/turbo4',
            kv_family='turbo',
            fit=True,
            fit_context=4096,
        )
        turbo3 = RuntimeProfile(
            engine_id='turboquant',
            name='fit_kv_compression_probe_turbo3',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='turbo3_tcq/turbo3_tcq',
            kv_family='turbo',
            fit=True,
            fit_context=4096,
        )
        default = RuntimeProfile(
            engine_id='turboquant',
            name='fit_default_probe',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='default',
            kv_family='default',
            fit=True,
            fit_context=4096,
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model, discover=True, managed_only=False):
                return None

            def hardware_profile(self, refresh=False):
                return hardware

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, saved):
                self.saved.append(saved)

        calls = []

        def fake_runtime_benchmark(_app, base_model, profile, objective, _progress, _cancel_token, completed, _total, **kwargs):
            calls.append(profile.name)
            candidate = ModelConfig(id=base_model.id, name=base_model.name, path=base_model.path, alias=base_model.alias, port=base_model.port, ctx=profile.ctx_size, parallel=profile.parallel)
            if profile.kv_family == 'turbo':
                failed = adaptive_record_from_candidate(
                    candidate,
                    objective,
                    'not ready',
                    detail='K cache type turbo4 with block size 128 does not divide n_embd_head_k=192',
                    failure_category='KV_MODE_INCOMPATIBLE',
                    failure_reason='K cache type turbo4 with block size 128 does not divide n_embd_head_k=192',
                )
                return False, True, [failed], [], completed + 1
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=18.0,
                seconds=1.0,
                engine=profile.engine_id,
                runtime_profile=profile.name,
                kv_preset=profile.kv_preset,
                runtime_fit=profile.fit,
                fit_context=profile.fit_context,
            )
            measured = dict(record)
            measured['model'] = candidate
            return True, False, [record], [measured], completed + 1

        events = []
        app = FakeApp()
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[turbo4, turbo3, default]), \
             patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
            ok, msg = benchmark_fast_profiles(app, model, progress=events.append)

        self.assertTrue(ok, msg)
        self.assertEqual(calls, ['fit_turbokv_probe', 'fit_default_probe'])
        self.assertTrue(any('disabled family turbo' in str(event) for event in events))

    def test_fast_runner_carries_weight_fit_ceiling_into_turboquant_context_growth(self):
        model = ModelConfig(id='m', name='M', path='/models/model-Q8_0.gguf', alias='m', port=18200)
        hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        weight_fit = RuntimeProfile(
            engine_id='turboquant',
            name='fit_weight_discovery_q8_0_turbo4',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='q8_0/turbo4',
            fit=True,
            fit_context=4096,
            benchmark_depth='fast',
            fit_discovery_phase='weight_fit',
        )
        context_growth = RuntimeProfile(
            engine_id='turboquant',
            name='fit_context_growth_sweep_16384_q8_0_turbo4',
            ctx_size=16384,
            gpu_layers=None,
            parallel=1,
            kv_preset='q8_0/turbo4',
            fit=True,
            fit_context=4096,
            benchmark_depth='fast',
            fit_discovery_phase='context_growth',
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model):
                return None

            def hardware_profile(self, refresh=False):
                return hardware

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, model):
                self.saved.append(model)

        seen_viable = []

        def fake_runtime_benchmark(_app, base_model, profile, objective, _progress, _cancel_token, completed, total, **kwargs):
            seen_viable.append((profile.name, profile.viable_ngl, profile.viable_ngl_source))
            candidate = ModelConfig(
                id=base_model.id,
                name=base_model.name,
                path=base_model.path,
                alias=base_model.alias,
                port=base_model.port,
                ctx=profile.ctx_size,
                parallel=profile.parallel,
            )
            viable_ngl = 28 if profile.fit_discovery_phase == 'weight_fit' else profile.viable_ngl
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=20.0,
                seconds=1.0,
                engine=profile.engine_id,
                runtime_profile=profile.name,
                kv_preset=profile.kv_preset,
                benchmark_depth=kwargs.get('benchmark_depth', ''),
                runtime_fit=profile.fit,
                fit_context=profile.fit_context,
                gpu_layers_mode='fit',
                fit_discovery_phase=profile.fit_discovery_phase,
                viable_ngl=viable_ngl,
                viable_ngl_source='offloaded_layers',
                fit_selected_ngl=viable_ngl,
                fit_selected_ngl_source='offloaded_layers',
                fit_log_excerpt='offloaded 28/33 layers to GPU',
            )
            measured = dict(record)
            measured['model'] = candidate
            return True, False, [record], [measured], completed + 1

        app = FakeApp()
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[weight_fit, context_growth]), \
             patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
            ok, msg = benchmark_fast_profiles(app, model, progress=lambda _msg: None)

        self.assertTrue(ok, msg)
        self.assertEqual(seen_viable[0], ('fit_weight_discovery_q8_0_turbo4', 0, ''))
        self.assertEqual(seen_viable[1], ('fit_context_growth_sweep_16384_q8_0_turbo4', 28, 'offloaded_layers'))

    def test_fast_runner_prunes_fixed_gpu_layer_profiles_after_fit_oom(self):
        model = ModelConfig(id='m', name='M', path='/models/model.gguf', alias='m', port=18200)
        hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        fixed_a = RuntimeProfile(
            engine_id='turboquant',
            name='gpu_layer_sweep_ngl26',
            ctx_size=8192,
            gpu_layers=26,
            parallel=1,
            kv_preset='turbo4/turbo4',
            benchmark_depth='fast',
        )
        fixed_b = RuntimeProfile(
            engine_id='turboquant',
            name='gpu_layer_sweep_ngl30',
            ctx_size=8192,
            gpu_layers=30,
            parallel=1,
            kv_preset='turbo4/turbo4',
            benchmark_depth='fast',
        )
        fit_fallback = RuntimeProfile(
            engine_id='turboquant',
            name='fit_default_probe',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='default',
            fit=True,
            fit_context=4096,
            no_warmup=True,
            benchmark_depth='fast',
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model):
                return None

            def hardware_profile(self, refresh=False):
                return hardware

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, model):
                self.saved.append(model)

        calls = []

        def fake_runtime_benchmark(_app, base_model, profile, objective, _progress, _cancel_token, completed, total, **kwargs):
            calls.append(profile.name)
            candidate = ModelConfig(
                id=base_model.id,
                name=base_model.name,
                path=base_model.path,
                alias=base_model.alias,
                port=base_model.port,
                ctx=profile.ctx_size,
                parallel=profile.parallel,
                ngl=profile.gpu_layers if profile.gpu_layers is not None else base_model.ngl,
            )
            if profile.gpu_layers is not None:
                failed = adaptive_record_from_candidate(
                    candidate,
                    objective,
                    'not ready',
                    detail='failed to fit params to free device memory: n_gpu_layers already set by user to 26',
                    failure_category='FIXED_GPU_LAYERS_FIT_FAILED',
                    failure_reason='fixed GPU layers cannot fit',
                    runtime_profile=profile.name,
                    gpu_layers_mode='fixed',
                )
                return False, True, [failed], [], completed + 1
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=20.0,
                seconds=1.0,
                engine=profile.engine_id,
                runtime_profile=profile.name,
                kv_preset=profile.kv_preset,
                runtime_fit=profile.fit,
                fit_context=profile.fit_context,
                gpu_layers_mode='fit',
            )
            measured = dict(record)
            measured['model'] = candidate
            return True, False, [record], [measured], completed + 1

        events = []
        app = FakeApp()
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[fixed_a, fixed_b, fit_fallback]), \
             patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
            ok, msg = benchmark_fast_profiles(app, model, progress=events.append)

        self.assertTrue(ok, msg)
        self.assertEqual(calls, ['gpu_layer_sweep_ngl26', 'fit_default_probe'])
        self.assertTrue(any('fixed GPU-layer profiles were already rejected' in str(item) for item in events))

    def test_fast_runner_prunes_turboquant_fixed_profiles_after_cuda_weight_oom(self):
        model = ModelConfig(id='m', name='M', path='/models/model-Q8_0.gguf', alias='m', port=18200)
        hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)
        fixed_a = RuntimeProfile(
            engine_id='turboquant',
            name='gpu_layer_sweep_ngl26',
            ctx_size=8192,
            gpu_layers=26,
            parallel=1,
            kv_preset='q8_0/turbo4',
            benchmark_depth='fast',
        )
        fixed_b = RuntimeProfile(
            engine_id='turboquant',
            name='gpu_layer_sweep_ngl30',
            ctx_size=8192,
            gpu_layers=30,
            parallel=1,
            kv_preset='q8_0/turbo4',
            benchmark_depth='fast',
        )
        fit_fallback = RuntimeProfile(
            engine_id='turboquant',
            name='fit_weight_discovery_q8_0_turbo4',
            ctx_size=8192,
            gpu_layers=None,
            parallel=1,
            kv_preset='q8_0/turbo4',
            fit=True,
            fit_context=4096,
            benchmark_depth='fast',
            fit_discovery_phase='weight_fit',
        )

        class FakeApp:
            opencode = type('OpenCode', (), {'path': ''})()

            def __init__(self):
                self.saved = []

            def health(self, _model):
                return 'STOPPED', ''

            def get_pid(self, _model):
                return None

            def hardware_profile(self, refresh=False):
                return hardware

            def model_fingerprint(self, _model):
                return 'fingerprint'

            def add_or_update(self, model):
                self.saved.append(model)

        calls = []

        def fake_runtime_benchmark(_app, base_model, profile, objective, _progress, _cancel_token, completed, total, **kwargs):
            calls.append(profile.name)
            candidate = ModelConfig(
                id=base_model.id,
                name=base_model.name,
                path=base_model.path,
                alias=base_model.alias,
                port=base_model.port,
                ctx=profile.ctx_size,
                parallel=profile.parallel,
                ngl=profile.gpu_layers if profile.gpu_layers is not None else base_model.ngl,
            )
            if profile.gpu_layers is not None:
                failed = adaptive_record_from_candidate(
                    candidate,
                    objective,
                    'not ready',
                    detail='alloc_tensor_range: failed to allocate CUDA0 buffer',
                    failure_category='CUDA_OOM_WEIGHTS',
                    failure_reason='failed to allocate CUDA0 buffer',
                    runtime_profile=profile.name,
                    gpu_layers_mode='fixed',
                )
                return False, True, [failed], [], completed + 1
            record = adaptive_record_from_candidate(
                candidate,
                objective,
                'ok',
                tokens_per_sec=20.0,
                seconds=1.0,
                engine=profile.engine_id,
                runtime_profile=profile.name,
                kv_preset=profile.kv_preset,
                runtime_fit=profile.fit,
                fit_context=profile.fit_context,
                gpu_layers_mode='fit',
                fit_discovery_phase=profile.fit_discovery_phase,
                viable_ngl=24,
                viable_ngl_source='offloaded_layers',
            )
            measured = dict(record)
            measured['model'] = candidate
            return True, False, [record], [measured], completed + 1

        events = []
        app = FakeApp()
        with patch('llama_tui.benchmark.active_engine_runtime_profiles', return_value=[fixed_a, fixed_b, fit_fallback]), \
             patch('llama_tui.benchmark.benchmark_runtime_profile_with_retry', side_effect=fake_runtime_benchmark):
            ok, msg = benchmark_fast_profiles(app, model, progress=events.append)

        self.assertTrue(ok, msg)
        self.assertEqual(calls, ['gpu_layer_sweep_ngl26', 'fit_weight_discovery_q8_0_turbo4'])
        self.assertTrue(any('fixed GPU-layer profiles were already rejected' in str(item) for item in events))

    def test_deep_benchmark_all_continues_using_adaptive_runner(self):
        self.assertEqual(benchmark_all_models_runner.__name__, 'benchmark_all_models_runner')

    def test_llama_dense_gguf_profiles_use_generic_non_turbo_kv(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='dense',
                name='Dense GGUF',
                path='/models/dense.gguf',
                alias='dense',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch('llama_tui.benchmark.model_file_size', return_value=5 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware)

        self.assertTrue(any(item.name == 'cpu_probe' for item in profiles))
        self.assertTrue(any(item.name == 'partial_gpu_probe' for item in profiles))
        self.assertTrue(any(item.kv_preset == 'q8_0/q8_0' for item in profiles))
        self.assertFalse(any('turbo' in item.kv_preset for item in profiles))

    def test_mtp_runtime_profiles_drop_legacy_fixed_ngl_profiles(self):
        # When the binary cannot do the fit/q8/no-mmap family the legacy
        # fixed-NGL MTP profiles are dropped entirely; only a fit-assisted
        # no-MTP baseline remains for a non-recurrent model.
        caps = replace(
            default_engine_capabilities('llama.cpp-mtp'),
            supports_spec_type=True,
            supports_mtp=True,
            supports_no_warmup=True,
            supports_fit=True,
            mtp_spec_type='draft-mtp',
            mtp_spec_type_value='draft-mtp',
            spec_type_values=('none', 'draft-mtp'),
            supports_spec_draft_n_max=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='mtp',
                name='Generic native-mtp',
                path='/models/generic-native-mtp.gguf',
                alias='mtp',
                port=18080,
                ctx_min=4096,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=6 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=caps), \
                patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='fast')

        names = [item.name for item in profiles]
        self.assertIn('mtp_baseline', names)
        # No legacy fixed-NGL MTP candidates without the full fit family.
        self.assertFalse(any(item.mtp_enabled for item in profiles))
        self.assertFalse(any(name.startswith('mtp_draft_n') for name in names))
        baseline = next(item for item in profiles if item.benchmark_phase == 'baseline_no_mtp')
        self.assertIsNone(baseline.gpu_layers)
        self.assertFalse(baseline.mtp_enabled)
        self.assertTrue(all(item.parallel == 1 for item in profiles))

    def test_mtp_fit_profile_family_includes_long_context_q8_nommap_without_ngl(self):
        caps = replace(
            default_engine_capabilities('llama.cpp-mtp'),
            supports_spec_type=True,
            supports_mtp=True,
            supports_no_warmup=True,
            supports_fit=True,
            supports_fit_ctx=True,
            supports_fit_target=True,
            supports_ctk_ctv=True,
            supports_spec_draft_n_max=True,
            supports_spec_draft_type_kv=True,
            supports_no_mmap=True,
            mtp_spec_type='draft-mtp',
            mtp_spec_type_value='draft-mtp',
        )
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='mtp',
                name='Qwen NextN MTP',
                path='/models/Qwen3.6-35B-A3B-MTP-GGUF.gguf',
                alias='mtp',
                port=18080,
                ctx_min=8192,
                ctx_max=131072,
                supports_mtp='yes',
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=6 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=caps), \
                patch('llama_tui.benchmark.model_file_size', return_value=35 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='fast')

        names = [item.name for item in profiles]
        primary = next(item for item in profiles if item.name == 'mtp_fit_q8_draftq8_nommap_draft1_128k')
        self.assertEqual(primary.ctx_size, 131072)
        self.assertIsNone(primary.gpu_layers)
        self.assertTrue(primary.fit)
        self.assertEqual(primary.fit_context, 4096)
        self.assertEqual(primary.fit_target, 1024)
        self.assertEqual(primary.kv_preset, 'q8_0/q8_0')
        self.assertEqual(primary.mtp_draft_kv_preset, 'q8_0/q8_0')
        self.assertTrue(primary.no_mmap)
        self.assertTrue(primary.no_warmup)
        self.assertEqual(primary.cache_ram, 0)
        self.assertEqual(primary.batch_size, 128)
        self.assertEqual(primary.ubatch_size, 64)
        self.assertEqual(primary.parallel, 1)
        self.assertEqual({8192, 32768, 65536, 131072}, {item.ctx_size for item in profiles if item.name.startswith('mtp_fit_q8_draftq8_nommap') and item.fit_target == 1024})
        for draft in (1, 2, 3):
            self.assertTrue(any(item.mtp_draft_n_max == draft and item.fit and item.kv_preset == 'q8_0/q8_0' for item in profiles))
        self.assertNotIn('mtp_baseline', names)
        # Legacy fixed-NGL MTP profiles are no longer generated.
        self.assertFalse(any(name.startswith('mtp_draft_n') for name in names))
        self.assertFalse(any((item.benchmark_phase or '').startswith('draft_n') for item in profiles))

    def test_mtp_runtime_profiles_emit_selected_draft_mtp_spec_type_for_every_mtp_profile(self):
        caps = replace(
            default_engine_capabilities('llama.cpp-mtp'),
            supports_spec_type=True,
            supports_mtp=True,
            supports_no_warmup=True,
            supports_fit=True,
            supports_fit_ctx=True,
            supports_fit_target=True,
            supports_ctk_ctv=True,
            supports_spec_draft_n_max=True,
            supports_spec_draft_type_kv=True,
            supports_no_mmap=True,
            spec_type_values=('none', 'draft-mtp'),
            mtp_spec_type='draft-mtp',
            mtp_spec_type_value='draft-mtp',
        )
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(
                Path(tmp) / 'models.json',
                runtime_profile=make_runtime_profile('llama.cpp-mtp', 'llama-server'),
            )
            model = ModelConfig(
                id='mtp',
                name='Generic native-mtp',
                path='/models/generic-native-mtp.gguf',
                alias='mtp',
                port=18080,
                runtime='llama.cpp-mtp',
                ctx_min=8192,
                ctx_max=131072,
                supports_mtp='yes',
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=6 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=caps), \
                patch('llama_tui.benchmark.model_file_size', return_value=16 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='fast')
                mtp_profiles = [item for item in profiles if item.mtp_enabled]
                self.assertTrue(mtp_profiles)
                for runtime_profile in mtp_profiles:
                    candidate = model_for_runtime_profile(model, runtime_profile)
                    launch_profile = build_benchmark_launch_profile(
                        candidate,
                        runtime_profile,
                        caps,
                        purpose='mtp_acceptance',
                        depth='fast',
                    )
                    command = app.build_command(
                        candidate,
                        runtime_profile=runtime_profile,
                        benchmark_profile=launch_profile,
                    )
                    spec_positions = [idx for idx, value in enumerate(command) if value == '--spec-type']
                    self.assertEqual(len(spec_positions), 1, runtime_profile.name)
                    self.assertEqual(command[spec_positions[0] + 1], 'draft-mtp', runtime_profile.name)
                    self.assertNotIn('--spec-type mtp', ' '.join(command), runtime_profile.name)

    def test_moe_runtime_profiles_include_bounded_placement_candidates(self):
        caps = replace(
            default_engine_capabilities('llama.cpp'),
            supports_cpu_moe=True,
            supports_n_cpu_moe=True,
            supports_override_tensor=True,
            gpu_layers_flag='-ngl',
        )
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='moe',
                name='MoE',
                path='/models/moe.gguf',
                alias='moe',
                port=18080,
                architecture_type='moe',
                expert_count=64,
                expert_used_count=8,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=8 * 1024**3, gpu_memory_free=7 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=caps), \
                patch('llama_tui.benchmark.model_file_size', return_value=12 * 1024**3), \
                patch('llama_tui.moe_placement.gguf_layer_count', return_value=40):
                profiles = active_engine_runtime_profiles(app, model, hardware, depth='fast')

        self.assertTrue(any(item.placement_strategy == 'partial_ngl_8' for item in profiles))
        self.assertTrue(any(item.cpu_moe for item in profiles))
        self.assertTrue(any(tuple(item.tensor_overrides or ()) for item in profiles))
        self.assertFalse(any(item.placement_strategy == 'full_gpu' for item in profiles))
        self.assertLessEqual(len({item.placement_strategy for item in profiles if item.placement_strategy}), 8)

    def test_small_moe_gguf_can_plan_full_gpu_without_model_specific_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='mix',
                name='Small MoE',
                path='/models/small-moe.gguf',
                alias='mix',
                port=18080,
                architecture='mixtral',
                architecture_type='moe',
                expert_count=8,
                expert_used_count=2,
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=16 * 1024**3, gpu_memory_free=12 * 1024**3)

            with patch.object(app, 'engine_capabilities', return_value=default_engine_capabilities('llama.cpp')):
                with patch('llama_tui.benchmark.model_file_size', return_value=4 * 1024**3):
                    profiles = active_engine_runtime_profiles(app, model, hardware, depth='full')

        self.assertTrue(any(item.name == 'gpu_layer_sweep_full' and item.gpu_layers == 999 for item in profiles))
        self.assertTrue(any(item.gpu_layers == 999 for item in profiles))

    def test_large_dense_gguf_uses_partial_gpu_and_q8_without_turbo(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='large-dense',
                name='Large Dense',
                path='/models/large-dense.gguf',
                alias='large',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                ctx_max=32768,
            )
            hardware = HardwareProfile(gpu_memory_total=10 * 1024**3, gpu_memory_free=8 * 1024**3)

            with patch('llama_tui.benchmark.model_file_size', return_value=14 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware)

        partial = next(item for item in profiles if item.name == 'partial_gpu_probe')
        self.assertGreater(partial.gpu_layers, 0)
        self.assertLess(partial.gpu_layers, 999)
        self.assertTrue(any(item.kv_preset == 'q8_0/q8_0' for item in profiles))
        self.assertFalse(any('turbo' in item.kv_preset for item in profiles))

    def test_cpu_only_gguf_profiles_do_not_request_gpu_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = AppConfig(Path(tmp) / 'models.json')
            model = ModelConfig(
                id='cpu',
                name='CPU Model',
                path='/models/cpu.gguf',
                alias='cpu',
                port=18080,
                architecture='llama',
                architecture_type='dense',
                ctx_max=16384,
            )
            hardware = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=32 * 1024**3, memory_available=24 * 1024**3)

            with patch('llama_tui.benchmark.model_file_size', return_value=6 * 1024**3):
                profiles = active_engine_runtime_profiles(app, model, hardware)

        self.assertTrue(profiles)
        self.assertTrue(all(item.gpu_layers == 0 for item in profiles))

    def test_runtime_profile_planner_does_not_branch_on_qwen_names(self):
        source = inspect.getsource(active_engine_runtime_profiles).lower()

        self.assertNotIn('qwen', source)

    def test_invalid_turboquant_kv_modes_fail_clearly(self):
        for flag in ('--kv', '--kv-key', '--kv-value'):
            with self.subTest(flag=flag):
                args = build_cli_parser().parse_args(['--engine', 'turboquant', flag, 'turbo3_tcq'])

                with self.assertRaises(SystemExit) as ctx:
                    validate_turboquant_kv_args(args)

                self.assertIn(f'Unsupported {flag} "turbo3_tcq"', str(ctx.exception))

    def test_cli_parser_accepts_kill_existing(self):
        args = build_cli_parser().parse_args(['--kill-existing'])

        self.assertTrue(args.kill_existing)

    def test_cli_help_documents_engines_examples_and_env_vars(self):
        help_text = build_cli_parser().format_help()

        self.assertIn('examples:', help_text)
        self.assertIn('llama-tui --engine turboquant --kv-key q8_0 --kv-value turbo4', help_text)
        self.assertIn('llama-tui --engine llama.cpp-mtp', help_text)
        self.assertIn('supported runtimes: llama.cpp, llama.cpp-mtp, turboquant', help_text)
        self.assertNotIn('vLLM', help_text)
        self.assertNotIn('Buun', help_text)
        self.assertNotIn('tq3', help_text)
        self.assertIn('config path:', help_text)
        self.assertIn('LLAMA_CPP_MTP_PATH', help_text)
        self.assertIn('TURBOQUANT_LLAMA_SERVER_BIN', help_text)
        self.assertNotIn('TQ3_LLAMA_SERVER_BIN', help_text)
        self.assertNotIn('BUUN_LLAMA_SERVER_BIN', help_text)
        self.assertNotIn('VLLM_COMMAND', help_text)
        self.assertIn('--help exits before curses starts', help_text)


class EngineSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_session(self, pid: int, engine: str):
        path = self.cache_dir / 'runtime_engine_sessions' / f'{pid}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"pid": {pid}, "engine": "{engine}"}}\n', encoding='utf-8')
        return path

    def test_stale_engine_sessions_are_pruned(self):
        stale = self.write_session(11111, 'llama.cpp')

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), patch('llama_tui.main.pid_is_alive', return_value=False):
            path = ensure_engine_session_lock('llama.cpp')

        self.assertFalse(stale.exists())
        self.assertTrue(path.exists())

    def test_same_engine_sessions_are_allowed(self):
        self.write_session(11111, 'llama.cpp')

        def alive(pid):
            return pid in {11111, os.getpid()}

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), patch('llama_tui.main.pid_is_alive', side_effect=alive):
            expected = engine_session_path()
            path = ensure_engine_session_lock('llama.cpp')

        self.assertEqual(path, expected)
        self.assertTrue(path.exists())

    def test_releasing_current_session_preserves_other_same_engine_sessions(self):
        old = self.write_session(11111, 'llama.cpp')

        def alive(pid):
            return pid in {11111, os.getpid()}

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), patch('llama_tui.main.pid_is_alive', side_effect=alive):
            path = ensure_engine_session_lock('llama.cpp')
            release_engine_session_lock(path)

        self.assertTrue(old.exists())
        self.assertFalse(path.exists())

    def test_different_live_engine_blocks_startup(self):
        session_path = self.write_session(11111, 'turboquant')

        def alive(pid):
            return pid == 11111

        process = {
            'pid': 11111,
            'command': 'python3 /home/jcampos/.local/bin/llama-tui',
            'cwd': '/home/jcampos/.cache/llmfit/models',
            'status': 'S (sleeping)',
            'state': 'S',
        }
        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), \
             patch('llama_tui.main.pid_is_alive', side_effect=alive), \
             patch('llama_tui.main.describe_pid', return_value=process):
            with self.assertRaises(SystemExit) as ctx:
                ensure_engine_session_lock('llama.cpp')

        message = str(ctx.exception)
        self.assertIn('Engine switch blocked', message)
        self.assertIn('PID 11111', message)
        self.assertIn('engine "turboquant"', message)
        self.assertIn('command: python3 /home/jcampos/.local/bin/llama-tui', message)
        self.assertIn('cwd: /home/jcampos/.cache/llmfit/models', message)
        self.assertIn(f'session: {session_path}', message)
        self.assertIn('--kill-existing', message)

    def test_interactive_prompt_accepts_kill_and_acquires_lock(self):
        blocker = self.write_session(11111, 'turboquant')
        terminated = []
        prompts = []

        def alive(pid):
            if pid == 11111:
                return pid not in terminated
            return pid == os.getpid()

        def terminate(pid):
            terminated.append(pid)
            return True, f'terminated PID {pid}'

        def prompt(engine, sessions):
            prompts.append((engine, sessions))
            return True

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), \
             patch('llama_tui.main.pid_is_alive', side_effect=alive), \
             patch('llama_tui.main.terminate_pid_group', side_effect=terminate):
            path = ensure_engine_session_lock('llama.cpp', interactive=True, prompt_fn=prompt)

        self.assertEqual(terminated, [11111])
        self.assertEqual(prompts[0][0], 'llama.cpp')
        self.assertEqual(prompts[0][1][0]['pid'], 11111)
        self.assertFalse(blocker.exists())
        self.assertTrue(path.exists())

    def test_interactive_prompt_declines_kill_and_exits_cleanly(self):
        blocker = self.write_session(11111, 'turboquant')

        def alive(pid):
            return pid == 11111

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), \
             patch('llama_tui.main.pid_is_alive', side_effect=alive), \
             patch('llama_tui.main.terminate_pid_group') as terminate:
            with self.assertRaises(SystemExit) as ctx:
                ensure_engine_session_lock('llama.cpp', interactive=True, prompt_fn=lambda _engine, _sessions: False)

        self.assertIn('Startup canceled', str(ctx.exception))
        self.assertFalse(terminate.called)
        self.assertTrue(blocker.exists())

    def test_kill_existing_terminates_blockers_removes_sessions_and_acquires_lock(self):
        blocker = self.write_session(11111, 'turboquant')
        terminated = []

        def alive(pid):
            if pid == 11111:
                return pid not in terminated
            return pid == os.getpid()

        def terminate(pid):
            terminated.append(pid)
            return True, f'terminated PID {pid}'

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), \
             patch('llama_tui.main.pid_is_alive', side_effect=alive), \
             patch('llama_tui.main.terminate_pid_group', side_effect=terminate):
            path = ensure_engine_session_lock('llama.cpp', kill_existing=True)

        self.assertEqual(terminated, [11111])
        self.assertFalse(blocker.exists())
        self.assertTrue(path.exists())
        self.assertEqual(last_engine_session_stop_count(), 1)

    def test_zombie_engine_session_is_pruned(self):
        stale = self.write_session(11111, 'turboquant')

        with patch('llama_tui.main.CACHE_DIR', self.cache_dir), \
             patch('llama_tui.main.os.kill', return_value=None), \
             patch('llama_tui.main.pid_state', return_value='Z'):
            path = ensure_engine_session_lock('llama.cpp')

        self.assertFalse(stale.exists())
        self.assertTrue(path.exists())


class MtpEngineDeprecationTests(unittest.TestCase):
    def test_notice_fires_only_for_legacy_alias(self):
        notice = mtp_engine_deprecation_notice('llama.cpp-mtp')
        self.assertIn('deprecation', notice.lower())
        self.assertIn('--engine llama.cpp', notice)
        self.assertIn('capability', notice)

    def test_empty_for_other_engines(self):
        for engine in ('llama.cpp', 'turboquant', ''):
            self.assertEqual(mtp_engine_deprecation_notice(engine), '')

    def test_case_and_whitespace_tolerant(self):
        self.assertNotEqual(mtp_engine_deprecation_notice('  LLAMA.CPP-MTP  '), '')


if __name__ == '__main__':
    unittest.main()
