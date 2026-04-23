import tempfile
import unittest
from pathlib import Path

from llama_tui.app import AppConfig
from llama_tui.models import ModelConfig


def block_for_model(text: str, name: str) -> str:
    marker = f'  - name: "{name}"'
    start = text.index(marker)
    next_start = text.find('\n  - name:', start + len(marker))
    return text[start:] if next_start == -1 else text[start:next_start]


class ContinueIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config_path = self.root / 'models.json'
        self.app = AppConfig(self.config_path)
        self.app.continue_settings.path = str(self.root / '.continue' / 'config.yaml')
        self.app.continue_settings.backup_dir = str(self.root / 'backups')

        self.main_model = ModelConfig(
            id='main',
            name='Main Model',
            path=__file__,
            alias='main-local',
            port=18080,
            ctx=16384,
            output=2048,
        )
        self.build_model = ModelConfig(
            id='build',
            name='Build Model',
            path=__file__,
            alias='build-local',
            port=18081,
            ctx=32768,
            output=4096,
        )
        self.small_model = ModelConfig(
            id='small',
            name='Small Model',
            path=__file__,
            alias='small-local',
            port=18082,
            ctx=8192,
            output=1024,
        )
        self.extra_model = ModelConfig(
            id='extra',
            name='Extra Model',
            path=__file__,
            alias='extra-local',
            port=18083,
            ctx=12288,
            output=1536,
        )
        for model in (self.main_model, self.build_model, self.small_model, self.extra_model):
            self.app.add_or_update(model)

    def tearDown(self):
        self.tmp.cleanup()

    def test_continue_settings_round_trip(self):
        self.app.continue_settings.default_model_id = 'main'
        self.app.continue_settings.edit_model_id = 'build'
        self.app.continue_settings.autocomplete_model_id = 'small'
        self.app.continue_settings.merge_mode = 'managed_file'
        self.app.save()

        loaded = AppConfig(self.config_path)

        self.assertEqual(loaded.continue_settings.path, str(self.root / '.continue' / 'config.yaml'))
        self.assertEqual(loaded.continue_settings.backup_dir, str(self.root / 'backups'))
        self.assertEqual(loaded.continue_settings.default_model_id, 'main')
        self.assertEqual(loaded.continue_settings.edit_model_id, 'build')
        self.assertEqual(loaded.continue_settings.autocomplete_model_id, 'small')
        self.assertEqual(loaded.continue_settings.merge_mode, 'managed_file')

    def test_generate_continue_config_uses_existing_role_assignments_and_backup(self):
        self.app.opencode.default_model_id = 'main'
        self.app.opencode.build_model_id = 'build'
        self.app.opencode.small_model_id = 'small'
        self.app.continue_settings.merge_mode = 'managed_file'
        target = Path(self.app.continue_settings.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('name: old\n', encoding='utf-8')

        ok, msg = self.app.generate_continue_config()

        self.assertTrue(ok)
        self.assertIn(str(target), msg)
        backups = list(Path(self.app.continue_settings.backup_dir).glob('config.*.yaml'))
        self.assertTrue(backups)

        text = target.read_text(encoding='utf-8')
        self.assertIn('name: "llama-tui Local Models"', text)
        self.assertIn('schema: "v1"', text)

        main_block = block_for_model(text, 'Main Model')
        self.assertIn('model: "main-local"', main_block)
        self.assertIn('apiBase: "http://127.0.0.1:18080/v1"', main_block)
        self.assertIn('roles:\n      - chat', main_block)
        self.assertNotIn('      - autocomplete', main_block)

        build_block = block_for_model(text, 'Build Model')
        self.assertIn('      - edit', build_block)
        self.assertIn('      - apply', build_block)
        self.assertNotIn('      - autocomplete', build_block)

        small_block = block_for_model(text, 'Small Model')
        self.assertIn('      - autocomplete', small_block)
        self.assertIn('autocompleteOptions:', small_block)
        self.assertIn('maxPromptTokens: 2048', small_block)

        extra_block = block_for_model(text, 'Extra Model')
        self.assertIn('roles:\n      - chat', extra_block)

    def test_generate_continue_config_falls_back_to_first_and_second_enabled_models(self):
        ok, msg = self.app.generate_continue_config()

        self.assertTrue(ok, msg)
        text = Path(self.app.continue_settings.path).read_text(encoding='utf-8')
        main_block = block_for_model(text, 'Main Model')
        second_block = block_for_model(text, 'Build Model')

        self.assertIn('      - chat', main_block)
        self.assertIn('      - edit', main_block)
        self.assertIn('      - apply', main_block)
        self.assertIn('      - autocomplete', second_block)

    def test_preserves_existing_sections_and_unmarked_models(self):
        target = Path(self.app.continue_settings.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            '\n'.join([
                'name: "Existing Continue"',
                'version: "1.0.0"',
                'schema: "v1"',
                'rules:',
                '  - "keep this rule"',
                'context:',
                '  - provider: "code"',
                'prompts:',
                '  - name: "saved prompt"',
                'mcpServers:',
                '  local:',
                '    command: "keep"',
                'docs:',
                '  - name: "docs"',
                'data:',
                '  - name: "data"',
                'models:',
                '  - name: "User Model"',
                '    provider: "openai"',
                '    model: "user-model"',
            ]) + '\n',
            encoding='utf-8',
        )

        ok, msg = self.app.generate_continue_config()

        self.assertTrue(ok, msg)
        text = target.read_text(encoding='utf-8')
        self.assertIn('rules:\n  - "keep this rule"', text)
        self.assertIn('context:\n  - provider: "code"', text)
        self.assertIn('prompts:', text)
        self.assertIn('mcpServers:', text)
        self.assertIn('docs:', text)
        self.assertIn('data:', text)
        self.assertIn('  # BEGIN llama-tui managed models', text)
        self.assertIn('  # END llama-tui managed models', text)
        self.assertIn('  - name: "User Model"', text)
        self.assertIn('  - name: "Main Model"', text)

    def test_existing_managed_marker_block_is_replaced_once(self):
        target = Path(self.app.continue_settings.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            '\n'.join([
                'name: "Existing Continue"',
                'version: "1.0.0"',
                'schema: "v1"',
                'models:',
                '  # BEGIN llama-tui managed models',
                '  - name: "Stale Model"',
                '    provider: "openai"',
                '    model: "stale"',
                '  # END llama-tui managed models',
                '  - name: "User Model"',
                '    provider: "openai"',
                '    model: "user-model"',
            ]) + '\n',
            encoding='utf-8',
        )

        ok, msg = self.app.generate_continue_config()
        self.assertTrue(ok, msg)
        ok, msg = self.app.generate_continue_config()

        self.assertTrue(ok, msg)
        text = target.read_text(encoding='utf-8')
        self.assertEqual(text.count('  # BEGIN llama-tui managed models'), 1)
        self.assertEqual(text.count('  # END llama-tui managed models'), 1)
        self.assertNotIn('Stale Model', text)
        self.assertIn('  - name: "User Model"', text)
        self.assertIn('  - name: "Main Model"', text)

    def test_continue_roles_override_opencode_fallback_roles(self):
        self.app.opencode.default_model_id = 'main'
        self.app.opencode.build_model_id = 'build'
        self.app.opencode.small_model_id = 'small'
        self.app.continue_settings.default_model_id = 'extra'
        self.app.continue_settings.edit_model_id = 'build'
        self.app.continue_settings.autocomplete_model_id = 'small'

        ok, msg = self.app.generate_continue_config()

        self.assertTrue(ok, msg)
        text = Path(self.app.continue_settings.path).read_text(encoding='utf-8')
        extra_block = block_for_model(text, 'Extra Model')
        build_block = block_for_model(text, 'Build Model')
        small_block = block_for_model(text, 'Small Model')
        main_block = block_for_model(text, 'Main Model')

        self.assertIn('      - chat', extra_block)
        self.assertIn('      - edit', build_block)
        self.assertIn('      - apply', build_block)
        self.assertIn('      - autocomplete', small_block)
        self.assertNotIn('      - edit', main_block)
        self.assertNotIn('      - apply', main_block)
        self.assertNotIn('      - autocomplete', main_block)
        self.assertLess(text.index('  - name: "Extra Model"'), text.index('  - name: "Main Model"'))

    def test_context_length_uses_per_slot_context(self):
        self.build_model.parallel = 4
        self.app.add_or_update(self.build_model)

        ok, msg = self.app.generate_continue_config()

        self.assertTrue(ok, msg)
        text = Path(self.app.continue_settings.path).read_text(encoding='utf-8')
        build_block = block_for_model(text, 'Build Model')
        self.assertIn('contextLength: 8192', build_block)
