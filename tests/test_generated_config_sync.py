import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_tui.app import AppConfig
from llama_tui.models import ModelConfig


class GeneratedConfigSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._patchers = [
            patch('llama_tui.app.CONFIG_DIR', self.root / 'config-root'),
            patch('llama_tui.app.DATA_DIR', self.root / 'data-root'),
            patch('llama_tui.app.CACHE_DIR', self.root / 'cache-root'),
        ]
        for patcher in self._patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.app = AppConfig(self.root / 'models.json')
        self.app.opencode.path = str(self.root / 'opencode.json')
        self.app.opencode.backup_dir = str(self.root / 'backups')
        self.app.continue_settings.path = str(self.root / '.continue' / 'config.yaml')
        self.app.continue_settings.backup_dir = str(self.root / 'backups')
        self.app.hermes.home_root = str(self.root / 'hermes')
        self.main_model = self.model('main', 'Main Model', 'main-local', 18080)
        self.stale_model = self.model('stale', 'Stale Model', 'stale-local', 18081)
        self.app.add_or_update(self.main_model)
        self.app.add_or_update(self.stale_model)

    def tearDown(self):
        self.tmp.cleanup()

    def model(self, model_id: str, name: str, alias: str, port: int, path: str = __file__) -> ModelConfig:
        return ModelConfig(
            id=model_id,
            name=name,
            path=path,
            alias=alias,
            port=port,
            ctx=8192,
            output=1024,
        )

    def generate_all(self):
        self.app.generate_opencode()
        self.app.generate_continue_config()
        self.app.generate_hermes_config(self.stale_model)

    def opencode_provider_keys(self):
        return json.loads(Path(self.app.opencode.path).read_text(encoding='utf-8')).get('provider', {}).keys()

    def continue_text(self) -> str:
        return Path(self.app.continue_settings.path).read_text(encoding='utf-8')

    def test_delete_syncs_configs_and_removes_generated_hermes_home(self):
        self.generate_all()
        self.app.continue_settings.default_model_id = 'stale'
        self.app.continue_settings.edit_model_id = 'stale'
        self.app.continue_settings.autocomplete_model_id = 'stale'
        stale_home = self.app.hermes_home_for_model(self.stale_model)

        with patch.object(self.app, 'stop', return_value=(True, 'stopped')):
            ok, msg = self.app.delete('stale', sync_exports=True)

        self.assertTrue(ok, msg)
        self.assertNotIn('local-stale', self.opencode_provider_keys())
        self.assertNotIn('Stale Model', self.continue_text())
        self.assertNotIn('stale-local', self.continue_text())
        self.assertIn('Main Model', self.continue_text())
        self.assertFalse(stale_home.exists())
        self.assertTrue(list((self.root / 'hermes' / 'backups' / 'stale').glob('config.*.yaml')))
        self.assertEqual(self.app.continue_settings.default_model_id, '')
        self.assertEqual(self.app.continue_settings.edit_model_id, '')
        self.assertEqual(self.app.continue_settings.autocomplete_model_id, '')

    def test_delete_aborts_when_stop_fails(self):
        with patch.object(self.app, 'stop', return_value=(False, 'running but unmanaged; could not find PID')):
            ok, msg = self.app.delete('stale', sync_exports=True)

        self.assertFalse(ok)
        self.assertIn('not deleted', msg)
        self.assertIsNotNone(self.app.get_model('stale'))

    def test_continue_export_write_failure_returns_error(self):
        with patch('llama_tui.app.write_text_atomic', side_effect=OSError('read-only')):
            ok, msg = self.app.generate_continue_config()

        self.assertFalse(ok)
        self.assertIn('Continue export failed', msg)
        self.assertIn('read-only', msg)

    def test_prune_last_model_writes_empty_exports(self):
        missing = self.root / 'missing.gguf'
        model = self.model('missing', 'Missing Model', 'missing-local', 18082, path=str(missing))
        model.source = 'manual'
        self.app.models = [model]
        self.app.save()
        self.app.generate_opencode()
        self.app.generate_continue_config()
        self.app.generate_hermes_config(model)

        with patch.object(self.app, 'stop', return_value=(True, 'stopped')):
            count, removed = self.app.prune_missing_models(sync_exports=True)

        self.assertEqual(count, 1)
        self.assertEqual(removed, ['missing'])
        opencode = json.loads(Path(self.app.opencode.path).read_text(encoding='utf-8'))
        self.assertEqual(opencode.get('provider'), {})
        self.assertEqual(opencode.get('model'), '')
        text = self.continue_text()
        self.assertEqual(text.count('  # BEGIN llama-tui managed models'), 1)
        self.assertEqual(text.count('  # END llama-tui managed models'), 1)
        self.assertNotIn('Missing Model', text)
        self.assertFalse((self.root / 'hermes' / 'missing').exists())

    def test_hermes_cleanup_leaves_non_generated_home(self):
        self.app.generate_opencode()
        self.app.generate_continue_config()
        home = self.app.hermes_home_for_model(self.stale_model)
        home.mkdir(parents=True)
        (home / 'config.yaml').write_text('user maintained hermes config\n', encoding='utf-8')

        with patch.object(self.app, 'stop', return_value=(True, 'stopped')):
            ok, msg = self.app.delete('stale', sync_exports=True)

        self.assertTrue(ok, msg)
        self.assertTrue(home.exists())
        self.assertEqual((home / 'config.yaml').read_text(encoding='utf-8'), 'user maintained hermes config\n')


if __name__ == '__main__':
    unittest.main()
