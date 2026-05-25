"""Unit tests for the config-file serialisation + archive helpers.

Covers the module extracted in audit #9 step 4. These are pure
filesystem helpers, so the tests run against ``tempfile`` roots and
do not touch the user's real config.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from llama_tui.config_io import (
    archive_broken_config_file,
    serialize_app_state,
    write_config_dict,
)


def _settings(**kwargs) -> SimpleNamespace:
    """Build a stand-in for the dataclass settings AppConfig holds."""
    return SimpleNamespace(**kwargs)


def _fake_app(models, **overrides) -> SimpleNamespace:
    """Build a minimal duck-typed AppConfig stand-in for the serialiser."""
    defaults = dict(
        llama_server='llama-server',
        hf_cache_root='',
        llmfit_cache_root='',
        llm_models_cache_root='',
        lm_studio_model_roots=[],
        opencode=_settings(path='', backup_dir='', timeout=600000),
        continue_settings=_settings(path='~/.continue/config.yaml'),
        hermes=_settings(command='hermes', home_root=''),
        ui=_settings(preferred_sort='port'),
        models=models,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class SerializeAppStateTests(unittest.TestCase):
    def test_top_level_keys_match_expected_schema(self):
        # serialize_app_state uses ``asdict`` on settings objects;
        # SimpleNamespace doesn't satisfy that, so the test imports the
        # real dataclasses for the settings slots.
        from llama_tui.models import (
            ContinueSettings,
            HermesSettings,
            OpencodeSettings,
            UiSettings,
        )
        app = SimpleNamespace(
            llama_server='ls',
            hf_cache_root='hf',
            llmfit_cache_root='lf',
            llm_models_cache_root='llm',
            lm_studio_model_roots=['/root'],
            opencode=OpencodeSettings(),
            continue_settings=ContinueSettings(),
            hermes=HermesSettings(),
            ui=UiSettings(),
            models=[],
        )
        data = serialize_app_state(app)
        self.assertEqual(
            sorted(data.keys()),
            sorted([
                'llama_server', 'hf_cache_root',
                'llmfit_cache_root', 'llm_models_cache_root',
                'lm_studio_model_roots',
                'opencode', 'continue', 'hermes', 'ui', 'models',
            ]),
        )
        self.assertEqual(data['llama_server'], 'ls')
        self.assertEqual(data['models'], [])

    def test_continue_settings_serialise_under_continue_key(self):
        # 'continue' is a Python keyword so the dataclass attribute is
        # named continue_settings; the JSON key must still be 'continue'
        # for backwards compatibility with existing models.json files.
        from llama_tui.models import (
            ContinueSettings,
            HermesSettings,
            OpencodeSettings,
            UiSettings,
        )
        app = SimpleNamespace(
            llama_server='', hf_cache_root='',
            llmfit_cache_root='', llm_models_cache_root='',
            lm_studio_model_roots=[],
            opencode=OpencodeSettings(),
            continue_settings=ContinueSettings(path='~/.foo/config.yaml'),
            hermes=HermesSettings(),
            ui=UiSettings(),
            models=[],
        )
        data = serialize_app_state(app)
        self.assertIn('continue', data)
        self.assertEqual(data['continue']['path'], '~/.foo/config.yaml')


class WriteConfigDictTests(unittest.TestCase):
    def test_writes_pretty_json_with_trailing_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a' / 'b' / 'models.json'
            write_config_dict(path, {'llama_server': 'ls', 'models': []})
            content = path.read_text(encoding='utf-8')
        self.assertTrue(content.endswith('\n'))
        # Pretty JSON: indented with 2 spaces.
        self.assertIn('  ', content)
        # Round-trip back through json.loads succeeds.
        self.assertEqual(json.loads(content), {'llama_server': 'ls', 'models': []})

    def test_creates_missing_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'deep' / 'nested' / 'config.json'
            write_config_dict(path, {})
            self.assertTrue(path.exists())


class ArchiveBrokenConfigFileTests(unittest.TestCase):
    def test_copies_existing_file_with_timestamped_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'models.json'
            config.write_text('{broken}')
            backup_dir = root / 'backups'
            backup = archive_broken_config_file(config, backup_dir)
        self.assertIsNotNone(backup)
        self.assertTrue(backup.name.startswith('models.broken.'))
        self.assertTrue(backup.name.endswith('.json'))

    def test_missing_source_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(
                archive_broken_config_file(Path(tmp) / 'nope.json', Path(tmp) / 'backups')
            )

    def test_creates_backup_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'config.json'
            config.write_text('{}')
            backup_dir = root / 'sub' / 'backups'
            backup = archive_broken_config_file(config, backup_dir)
            self.assertIsNotNone(backup)
            self.assertTrue(backup_dir.exists())


if __name__ == '__main__':
    unittest.main()
