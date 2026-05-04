import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llama_tui.engines import (
    ENGINE_BUUN,
    ENGINE_LLAMA_CPP,
    ENGINE_TURBOQUANT,
    ENGINE_VLLM,
    command_exists,
    get_engine_definitions,
    get_engine_health,
    resolve_engine_install,
    turboquant_binary_warning,
)
from llama_tui.runtime_profiles import EngineCapabilities


class EngineRegistryTests(unittest.TestCase):
    def test_builtin_engine_definitions_cover_supported_engines(self):
        definitions = get_engine_definitions()

        self.assertIn(ENGINE_LLAMA_CPP, definitions)
        self.assertIn(ENGINE_TURBOQUANT, definitions)
        self.assertIn(ENGINE_BUUN, definitions)
        self.assertIn(ENGINE_VLLM, definitions)
        self.assertTrue(definitions[ENGINE_LLAMA_CPP].supports_gguf)
        self.assertTrue(definitions[ENGINE_VLLM].supports_hf_ref)

    def test_resolve_engine_install_preserves_config_and_env_behavior(self):
        config = SimpleNamespace(llama_server='/opt/llama-server', vllm_command='python -m vllm.entrypoints.openai.api_server')

        llama_install = resolve_engine_install(config, ENGINE_LLAMA_CPP)
        vllm_install = resolve_engine_install(config, ENGINE_VLLM)
        with patch.dict('os.environ', {'TURBOQUANT_LLAMA_SERVER_BIN': '/opt/tq/llama-server'}):
            turbo_install = resolve_engine_install(config, ENGINE_TURBOQUANT)

        self.assertEqual(llama_install.resolved_command, '/opt/llama-server')
        self.assertEqual(llama_install.source, 'config:llama_server')
        self.assertEqual(vllm_install.resolved_command, 'python -m vllm.entrypoints.openai.api_server')
        self.assertEqual(vllm_install.source, 'config:vllm_command')
        self.assertEqual(turbo_install.resolved_command, '/opt/tq/llama-server')
        self.assertEqual(turbo_install.source, 'env:TURBOQUANT_LLAMA_SERVER_BIN')

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

        with patch('llama_tui.engines.shutil.which', return_value='/usr/bin/vllm'):
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


if __name__ == '__main__':
    unittest.main()
