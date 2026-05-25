import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from llama_tui.engines import (
    ENGINE_LLAMA_CPP,
    ENGINE_LLAMA_CPP_MTP,
    ENGINE_TURBOQUANT,
    get_engine_definitions,
    get_engine_health,
    mtp_binary_warning,
    normalize_engine_id,
    resolve_engine_install,
    turboquant_binary_warning,
)
from llama_tui.runtime_profiles import EngineCapabilities


class EngineRegistryTests(unittest.TestCase):
    def test_builtin_engine_definitions_cover_supported_engines(self):
        # Audit #7: ENGINE_LLAMA_CPP_MTP collapsed into ENGINE_LLAMA_CPP;
        # MTP is now a binary capability resolved via --help, not a
        # separate engine entry.
        definitions = get_engine_definitions()

        self.assertIn(ENGINE_LLAMA_CPP, definitions)
        self.assertNotIn(ENGINE_LLAMA_CPP_MTP, definitions)
        self.assertIn(ENGINE_TURBOQUANT, definitions)
        # vLLM, Buun and TQ3 engines removed (2026-05): no longer registered.
        self.assertNotIn('vllm', definitions)
        self.assertNotIn('buun', definitions)
        self.assertNotIn('tq3', definitions)
        self.assertTrue(definitions[ENGINE_LLAMA_CPP].supports_gguf)
        # The MTP fork's binary discovery paths now live inside the
        # llama.cpp default_paths so users who built the fork to a
        # non-standard location are still found.
        llama_defaults = definitions[ENGINE_LLAMA_CPP].default_paths
        self.assertTrue(any('llama.cpp-mtp' in path or 'llama-server-mtp' in path for path in llama_defaults))

    def test_resolve_engine_install_preserves_config_and_env_behavior(self):
        config = SimpleNamespace(llama_server='/opt/llama-server', vllm_command='python -m vllm.entrypoints.openai.api_server')

        llama_install = resolve_engine_install(config, ENGINE_LLAMA_CPP)
        with patch.dict('os.environ', {
            'TURBOQUANT_LLAMA_SERVER_BIN': '/opt/tq/llama-server',
        }):
            turbo_install = resolve_engine_install(config, ENGINE_TURBOQUANT)

        self.assertEqual(llama_install.resolved_command, '/opt/llama-server')
        self.assertEqual(llama_install.source, 'config:llama_server')
        self.assertEqual(turbo_install.resolved_command, '/opt/tq/llama-server')
        self.assertEqual(turbo_install.source, 'env:TURBOQUANT_LLAMA_SERVER_BIN')

    def test_legacy_llama_cpp_mtp_path_falls_back_for_llama_cpp_engine(self):
        # Audit #7: ENGINE_LLAMA_CPP_MTP is gone, but the legacy
        # LLAMA_CPP_MTP_PATH env var still resolves to a binary for
        # the plain llama.cpp engine when LLAMA_SERVER is unset.
        config = SimpleNamespace(llama_server='', vllm_command='vllm')
        with patch.dict('os.environ', {'LLAMA_CPP_MTP_PATH': '/opt/mtp/bin', 'LLAMA_SERVER': ''}, clear=False):
            install = resolve_engine_install(config, ENGINE_LLAMA_CPP)
        self.assertEqual(install.resolved_command, '/opt/mtp/bin/llama-server')
        self.assertEqual(install.source, 'env:LLAMA_CPP_MTP_PATH')

    def test_legacy_mtp_engine_alias_normalises_to_llama_cpp(self):
        # Audit #7: passing the legacy 'llama.cpp-mtp' engine identifier
        # to resolve_engine_install now resolves to the llama.cpp engine
        # so all the binary-discovery logic lives in one place.
        config = SimpleNamespace(llama_server='/opt/llama-server', vllm_command='vllm')
        install = resolve_engine_install(config, ENGINE_LLAMA_CPP_MTP)
        self.assertEqual(install.id, ENGINE_LLAMA_CPP)
        self.assertEqual(install.resolved_command, '/opt/llama-server')

    def test_legacy_mtp_env_var_resolves_via_llama_cpp_engine(self):
        # When LLAMA_CPP_MTP_PATH is set and no other binary is
        # configured, llama.cpp resolution falls back to the MTP fork
        # binary directory — this keeps existing fork users working
        # without having to migrate to LLAMA_SERVER.
        config = SimpleNamespace(llama_server='', vllm_command='vllm')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'llama.cpp-mtp'
            binary = root / 'build-mtp' / 'bin' / 'llama-server'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/sh\n', encoding='utf-8')
            binary.chmod(0o755)
            with patch.dict(os.environ, {'LLAMA_CPP_MTP_PATH': str(root), 'LLAMA_SERVER': ''}, clear=False):
                install = resolve_engine_install(config, ENGINE_LLAMA_CPP)
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

    def test_legacy_removed_engine_aliases_normalise_to_llama_cpp(self):
        # vLLM, Buun and TQ3 engines removed (2026-05): legacy identifiers
        # degrade to llama.cpp so old models.json/benchmark records load.
        self.assertEqual(normalize_engine_id('vllm'), ENGINE_LLAMA_CPP)
        self.assertEqual(normalize_engine_id('buun'), ENGINE_LLAMA_CPP)
        self.assertEqual(normalize_engine_id('tq3'), ENGINE_LLAMA_CPP)
        self.assertEqual(normalize_engine_id('llama.cpp-tq3'), ENGINE_LLAMA_CPP)

    def test_turboquant_binary_warning_preserves_existing_message(self):
        caps = EngineCapabilities(help_text='--cache-type-k allowed values: f16 q8_0 q4_0')

        warning = turboquant_binary_warning('/work/llama.cpp/build/bin/llama-server', caps)

        self.assertIn('does not advertise turbo cache types', warning)
        self.assertIn('vanilla llama.cpp', warning)
        self.assertEqual(
            turboquant_binary_warning('/work/llama-cpp-turboquant/build/bin/llama-server', EngineCapabilities(help_text='allowed values: q8_0 turbo4')),
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

        self.assertIn('--spec-type mtp/draft-mtp', warning)
        self.assertIn('stable llama.cpp', warning)
        self.assertIn('source=default', warning)
        self.assertIn('executable=yes', warning)
        self.assertEqual(
            mtp_binary_warning(
                '/work/llama.cpp-mtp/build/bin/llama-server',
                EngineCapabilities(help_text='--spec-type mtp\n--spec-draft-n-max N', supports_spec_type=True, supports_mtp=True, mtp_spec_type='mtp', supports_spec_draft_n_max=True),
            ),
            '',
        )

    # Audit #7: the previous test_mtp_engine_health_* cases asserted a
    # dedicated MTP_FLAGS_NOT_FOUND / BUILD_REQUIRED engine_health
    # status for ENGINE_LLAMA_CPP_MTP. With MTP collapsed into the
    # llama.cpp engine, the equivalent diagnostics are surfaced
    # per-model via AppConfig.mtp_binary_warning + validate_mtp_launch
    # (covered in test_runtime_profiles), not as engine-level health.


if __name__ == '__main__':
    unittest.main()
