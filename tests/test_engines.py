import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llama_tui.engines import (
    ENGINE_BUUN,
    ENGINE_LLAMA_CPP,
    ENGINE_LLAMA_CPP_MTP,
    ENGINE_TQ3,
    ENGINE_TURBOQUANT,
    ENGINE_VLLM,
    command_exists,
    get_engine_definitions,
    get_engine_health,
    mtp_binary_warning,
    resolve_engine_install,
    tq3_binary_warning,
    turboquant_binary_warning,
)
from llama_tui.runtime_profiles import EngineCapabilities


class EngineRegistryTests(unittest.TestCase):
    def test_builtin_engine_definitions_cover_supported_engines(self):
        definitions = get_engine_definitions()

        self.assertIn(ENGINE_LLAMA_CPP, definitions)
        self.assertIn(ENGINE_LLAMA_CPP_MTP, definitions)
        self.assertIn(ENGINE_TURBOQUANT, definitions)
        self.assertIn(ENGINE_TQ3, definitions)
        self.assertIn(ENGINE_BUUN, definitions)
        self.assertIn(ENGINE_VLLM, definitions)
        self.assertTrue(definitions[ENGINE_LLAMA_CPP].supports_gguf)
        self.assertEqual(definitions[ENGINE_LLAMA_CPP_MTP].display_name, 'llama.cpp MTP')
        self.assertIn('Experimental', definitions[ENGINE_LLAMA_CPP_MTP].notes)
        self.assertTrue(definitions[ENGINE_VLLM].supports_hf_ref)

    def test_resolve_engine_install_preserves_config_and_env_behavior(self):
        config = SimpleNamespace(llama_server='/opt/llama-server', vllm_command='python -m vllm.entrypoints.openai.api_server')

        llama_install = resolve_engine_install(config, ENGINE_LLAMA_CPP)
        vllm_install = resolve_engine_install(config, ENGINE_VLLM)
        with patch.dict('os.environ', {
            'TURBOQUANT_LLAMA_SERVER_BIN': '/opt/tq/llama-server',
            'TQ3_LLAMA_SERVER_BIN': '/opt/tq3/llama-server',
            'LLAMA_CPP_MTP_PATH': '/opt/mtp/bin',
        }):
            turbo_install = resolve_engine_install(config, ENGINE_TURBOQUANT)
            tq3_install = resolve_engine_install(config, ENGINE_TQ3)
            mtp_install = resolve_engine_install(config, ENGINE_LLAMA_CPP_MTP)

        self.assertEqual(llama_install.resolved_command, '/opt/llama-server')
        self.assertEqual(llama_install.source, 'config:llama_server')
        self.assertEqual(vllm_install.resolved_command, 'python -m vllm.entrypoints.openai.api_server')
        self.assertEqual(vllm_install.source, 'config:vllm_command')
        self.assertEqual(turbo_install.resolved_command, '/opt/tq/llama-server')
        self.assertEqual(turbo_install.source, 'env:TURBOQUANT_LLAMA_SERVER_BIN')
        self.assertEqual(tq3_install.resolved_command, '/opt/tq3/llama-server')
        self.assertEqual(tq3_install.source, 'env:TQ3_LLAMA_SERVER_BIN')
        self.assertEqual(mtp_install.resolved_command, '/opt/mtp/bin/llama-server')
        self.assertEqual(mtp_install.source, 'env:LLAMA_CPP_MTP_PATH')

    def test_mtp_engine_install_uses_shared_discovery_for_defaults_and_path(self):
        config = SimpleNamespace(llama_server='/opt/llama-server', vllm_command='vllm')
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            binary = home / 'src' / 'llama.cpp' / 'build-mtp' / 'bin' / 'llama-server'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\n', encoding='utf-8')
            binary.chmod(0o755)

            with patch.dict(os.environ, {'LLAMA_CPP_MTP_PATH': ''}, clear=False), \
                 patch('llama_tui.runtime_profiles.Path.home', return_value=home):
                install = resolve_engine_install(config, ENGINE_LLAMA_CPP_MTP)

        self.assertEqual(install.resolved_command, str(binary))
        self.assertEqual(install.source, 'default')
        self.assertTrue(install.exists)
        self.assertTrue(install.executable)

        with patch.dict(os.environ, {'LLAMA_CPP_MTP_PATH': ''}, clear=False), \
             patch('llama_tui.runtime_profiles.Path.home', return_value=Path('/definitely/missing')), \
             patch('llama_tui.runtime_profiles.shutil.which', return_value='/usr/local/bin/llama-server-mtp'):
            path_install = resolve_engine_install(config, ENGINE_LLAMA_CPP_MTP)

        self.assertEqual(path_install.resolved_command, 'llama-server-mtp')
        self.assertEqual(path_install.source, 'PATH')
        self.assertTrue(path_install.exists)
        self.assertTrue(path_install.executable)

    def test_mtp_env_path_takes_precedence_over_saved_runtime_profile(self):
        runtime_profile = SimpleNamespace(
            engine_id=ENGINE_LLAMA_CPP_MTP,
            server_command='/old/llama.cpp-mtp/build/bin/llama-server',
        )
        config = SimpleNamespace(
            llama_server='/opt/llama-server',
            vllm_command='vllm',
            runtime_profile=runtime_profile,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'llama.cpp-mtp'
            binary = root / 'build-mtp' / 'bin' / 'llama-server'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\n', encoding='utf-8')
            binary.chmod(0o755)

            with patch.dict(os.environ, {'LLAMA_CPP_MTP_PATH': str(root)}, clear=False):
                install = resolve_engine_install(config, ENGINE_LLAMA_CPP_MTP)

        self.assertEqual(install.resolved_command, str(binary))
        self.assertEqual(install.source, 'env:LLAMA_CPP_MTP_PATH')
        self.assertTrue(install.exists)
        self.assertTrue(install.executable)

    def test_missing_binary_health_is_fail_not_exception(self):
        config = SimpleNamespace(llama_server='/definitely/missing/llama-server', vllm_command='vllm')

        health = get_engine_health(config, ENGINE_LLAMA_CPP)

        self.assertEqual(health.status, 'FAIL')
        self.assertIn('binary missing', health.summary)

    def test_unknown_capabilities_are_warn_not_ok(self):
        config = SimpleNamespace(llama_server='/bin/sh', vllm_command='vllm')

        with patch('llama_tui.engines.detect_engine_capabilities', return_value=EngineCapabilities()):
            health = get_engine_health(config, ENGINE_LLAMA_CPP)

        self.assertEqual(health.status, 'WARN')
        self.assertIn('capabilities unknown', health.summary)

    def test_vllm_command_resolution_uses_path_lookup_for_simple_command(self):
        config = SimpleNamespace(llama_server='/opt/llama-server', vllm_command='vllm')

        with patch('llama_tui.runtime_profiles.shutil.which', return_value='/usr/bin/vllm'):
            install = resolve_engine_install(config, ENGINE_VLLM)
            exists = command_exists('vllm')

        self.assertTrue(install.exists)
        self.assertTrue(exists)

    def test_turboquant_binary_warning_preserves_existing_message(self):
        caps = EngineCapabilities(help_text='--cache-type-k allowed values: f16 q8_0 q4_0')

        warning = turboquant_binary_warning('/work/llama.cpp/build/bin/llama-server', caps)

        self.assertIn('does not advertise turbo cache types', warning)
        self.assertIn('vanilla llama.cpp', warning)
        self.assertEqual(
            turboquant_binary_warning('/work/llama-cpp-turboquant/build/bin/llama-server', EngineCapabilities(help_text='allowed values: q8_0 turbo4')),
            '',
        )

    def test_tq3_binary_warning_requires_tq3_cache_type(self):
        caps = EngineCapabilities(help_text='--cache-type-k allowed values: f16 q8_0 q4_0')

        warning = tq3_binary_warning('/work/llama.cpp/build/bin/llama-server', caps)

        self.assertIn('does not advertise tq3_0 cache type', warning)
        self.assertIn('vanilla llama.cpp', warning)
        self.assertEqual(
            tq3_binary_warning('/work/llama.cpp-tq3/build/bin/llama-server', EngineCapabilities(help_text='allowed values: q8_0 tq3_0')),
            '',
        )

    def test_mtp_binary_warning_requires_speculative_flags(self):
        caps = EngineCapabilities(help_text='--ctx-size N')

        warning = mtp_binary_warning(
            '/work/llama.cpp/build/bin/llama-server',
            caps,
            source='default',
            exists=True,
            executable=True,
        )

        self.assertIn('--spec-type mtp', warning)
        self.assertIn('stable llama.cpp', warning)
        self.assertIn('source=default', warning)
        self.assertIn('executable=yes', warning)
        self.assertEqual(
            mtp_binary_warning(
                '/work/llama.cpp-mtp/build/bin/llama-server',
                EngineCapabilities(help_text='--spec-type mtp\n--spec-draft-n-max N', supports_spec_type=True, supports_mtp=True, supports_spec_draft_n_max=True),
            ),
            '',
        )

    def test_mtp_engine_health_reports_missing_flags(self):
        config = SimpleNamespace(llama_server='/opt/llama-server', vllm_command='vllm')

        with patch('llama_tui.engines.resolve_engine_install') as resolve_install, \
             patch('llama_tui.engines.detect_engine_capabilities', return_value=EngineCapabilities(help_text='--help')):
            install = resolve_engine_install(config, ENGINE_LLAMA_CPP_MTP)
            resolve_install.return_value = type(install)(
                id=ENGINE_LLAMA_CPP_MTP,
                resolved_command='/work/llama.cpp-mtp/build/bin/llama-server',
                source='default',
                exists=True,
                executable=True,
                resolved_path='/work/llama.cpp-mtp/build/bin/llama-server',
                checked_paths=['/work/llama.cpp-mtp/build/bin/llama-server'],
            )
            health = get_engine_health(config, ENGINE_LLAMA_CPP_MTP)

        self.assertEqual(health.status, 'MTP_FLAGS_NOT_FOUND')
        self.assertIn('--spec-type mtp', health.summary)
        self.assertIn('source=default', health.summary)

    def test_mtp_engine_health_reports_non_executable_binary(self):
        config = SimpleNamespace(llama_server='/opt/llama-server', vllm_command='vllm')

        with patch('llama_tui.engines.resolve_engine_install') as resolve_install:
            install = resolve_engine_install(config, ENGINE_LLAMA_CPP_MTP)
            resolve_install.return_value = type(install)(
                id=ENGINE_LLAMA_CPP_MTP,
                resolved_command='/work/llama.cpp-mtp/build/bin/llama-server',
                source='default',
                exists=True,
                executable=False,
                resolved_path='/work/llama.cpp-mtp/build/bin/llama-server',
                checked_paths=['/work/llama.cpp-mtp/build/bin/llama-server'],
            )
            health = get_engine_health(config, ENGINE_LLAMA_CPP_MTP)

        self.assertEqual(health.status, 'BUILD_REQUIRED')
        self.assertIn('executable=no', health.summary)


if __name__ == '__main__':
    unittest.main()
