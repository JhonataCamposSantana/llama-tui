import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_tui.app import AppConfig
from llama_tui.constants import default_lm_studio_home, default_lm_studio_model_roots
from llama_tui.models import ModelConfig


class LmStudioDiscoveryTests(unittest.TestCase):
    def test_lm_studio_home_uses_env_then_pointer_then_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with patch.dict('os.environ', {'LM_STUDIO_HOME': str(home / 'env-home')}, clear=False):
                self.assertEqual(default_lm_studio_home(), home / 'env-home')

            pointer_home = home / 'pointed-home'
            pointer = home / '.lmstudio-home-pointer'
            pointer.write_text(str(pointer_home), encoding='utf-8')
            with patch.dict('os.environ', {}, clear=True):
                with patch('pathlib.Path.home', return_value=home):
                    self.assertEqual(default_lm_studio_home(), pointer_home)

            pointer.unlink()
            with patch.dict('os.environ', {}, clear=True):
                with patch('pathlib.Path.home', return_value=home):
                    self.assertEqual(default_lm_studio_home(), home / '.lmstudio')

    def test_lm_studio_default_model_roots_are_user_model_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / 'lmstudio'
            with patch.dict('os.environ', {'LM_STUDIO_HOME': str(home)}, clear=False):
                roots = default_lm_studio_model_roots()

        self.assertEqual(roots, [home / 'models', home / 'hub' / 'models'])

    def test_lm_studio_detection_adds_user_models_but_skips_internal_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'models.json'
            lm_home = root / 'lmstudio'
            user_model = lm_home / 'models' / 'org' / 'model' / 'chat.Q4_K_M.gguf'
            hub_model = lm_home / 'hub' / 'models' / 'org' / 'other' / 'coder.Q5_K_M.gguf'
            internal_model = lm_home / '.internal' / 'bundled-models' / 'org' / 'embed' / 'embed.Q4_K_M.gguf'
            user_model.parent.mkdir(parents=True)
            hub_model.parent.mkdir(parents=True)
            internal_model.parent.mkdir(parents=True)
            user_model.write_bytes(b'not a real gguf, but a discoverable file')
            hub_model.write_bytes(b'not a real gguf, but a discoverable file')
            internal_model.write_bytes(b'not a real gguf, but a skipped bundled file')

            app = AppConfig(config)
            app.hf_cache_root = str(root / 'missing-hf')
            app.llmfit_cache_root = str(root / 'missing-llmfit')
            app.llm_models_cache_root = str(root / 'missing-local')
            app.lm_studio_model_roots = f'{lm_home / "models"}, {lm_home / "hub" / "models"}'

            count, messages = app.detect_models()

        self.assertEqual(count, 2)
        self.assertTrue(any('added:' in message for message in messages))
        self.assertEqual({model.source for model in app.models}, {'lm-studio'})
        self.assertIn(str(user_model), {model.path for model in app.models})
        self.assertIn(str(hub_model), {model.path for model in app.models})
        self.assertNotIn(str(internal_model), {model.path for model in app.models})

    def test_old_config_loads_with_default_lm_studio_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'models.json'
            config.write_text(json.dumps({
                'llama_server': '/bin/llama-server',
                'models': [],
            }), encoding='utf-8')

            app = AppConfig(config)

        self.assertTrue(getattr(app, 'lm_studio_model_roots', ''))
        self.assertIn('models', app.lm_studio_model_roots)

    def test_load_ignores_unknown_future_settings_and_model_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'models.json'
            config.write_text(json.dumps({
                'continue': {
                    'path': str(root / 'config.yaml'),
                    'merge_mode': 'preserve_sections',
                    'future_continue_key': 'keep loader tolerant',
                },
                'opencode': {
                    'path': str(root / 'opencode.json'),
                    'future_opencode_key': 'ignored',
                },
                'models': [
                    {
                        'id': 'good',
                        'name': 'Good',
                        'path': str(root / 'good.gguf'),
                        'alias': 'good',
                        'port': 18080,
                        'tags': ['coding'],
                        'verification_status': 'passed',
                        'future_model_key': {'ignored': True},
                    },
                ],
            }), encoding='utf-8')

            app = AppConfig(config)

        self.assertEqual(app.continue_settings.path, str(root / 'config.yaml'))
        self.assertEqual(app.opencode.path, str(root / 'opencode.json'))
        self.assertEqual([model.id for model in app.models], ['good'])
        self.assertEqual(app.models[0].tags, ['coding'])
        self.assertEqual(app.models[0].verification_status, 'passed')

    def test_load_preserves_manual_models_when_paths_are_temporarily_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'models.json'
            app = AppConfig(config)
            missing_path = root / 'detached-drive' / 'portable.gguf'
            app.models = [ModelConfig(
                id='portable',
                name='Portable',
                path=str(missing_path),
                alias='portable',
                port=19001,
                source='manual',
            )]
            app.save()

            loaded = AppConfig(config)

        self.assertEqual(len(loaded.models), 1)
        self.assertEqual(loaded.models[0].id, 'portable')
        self.assertEqual(loaded.models[0].path, str(missing_path))

    def test_prune_removes_missing_lm_studio_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'models.json'
            lm_models = root / 'lmstudio' / 'models'
            app = AppConfig(config)
            app.hf_cache_root = str(root / 'missing-hf')
            app.llmfit_cache_root = str(root / 'missing-llmfit')
            app.llm_models_cache_root = str(root / 'missing-local')
            app.lm_studio_model_roots = str(lm_models)
            model_path = lm_models / 'org' / 'gone.Q4_K_M.gguf'
            app.models = [ModelConfig(
                id='gone',
                name='Gone',
                path=str(model_path),
                alias='gone',
                port=19000,
                source='lm-studio',
            )]
            app.save()

            with patch.object(app, 'stop', return_value=(True, 'stopped')):
                with patch.object(app, '_clear_pid_tracking'):
                    removed_count, removed = app.prune_missing_models()

        self.assertEqual(removed_count, 1)
        self.assertEqual(removed, ['gone'])
        self.assertEqual(app.models, [])

    def test_load_recovers_from_malformed_json_and_archives_broken_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'models.json'
            config.write_text('{"models": [', encoding='utf-8')
            config_dir = root / 'config'
            data_dir = root / 'data'
            cache_dir = root / 'cache'

            with patch('llama_tui.app.CONFIG_DIR', config_dir):
                with patch('llama_tui.app.DATA_DIR', data_dir):
                    with patch('llama_tui.app.CACHE_DIR', cache_dir):
                        app = AppConfig(config)
                        saved_text = config.read_text(encoding='utf-8')
                        backups = list((config_dir / 'backups').glob('models.broken.*.json'))

        self.assertEqual(app.models, [])
        self.assertTrue(app.load_warnings)
        self.assertIn('Config recovery', app.load_warnings[0])
        self.assertEqual(json.loads(saved_text)['models'], [])
        self.assertTrue(backups)

    def test_load_keeps_valid_models_and_reports_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'models.json'
            config.write_text(json.dumps({
                'models': [
                    {
                        'id': 'good',
                        'name': 'Good',
                        'path': str(root / 'good.gguf'),
                        'alias': 'good',
                        'port': 18080,
                    },
                    {
                        'id': 'bad',
                        'name': 'Bad',
                        'path': str(root / 'bad.gguf'),
                        'alias': 'bad',
                        'port': 'oops',
                    },
                ],
            }), encoding='utf-8')

            app = AppConfig(config)

        self.assertEqual([model.id for model in app.models], ['good'])
        self.assertTrue(any('skipped model row 2' in warning for warning in app.load_warnings))
        self.assertEqual(app.models[0].sort_rank, 1)

    def test_workspace_presets_remember_recent_unique_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'models.json'
            app = AppConfig(config)

            app.remember_workspace_preset('opencode', str(root / 'one'))
            app.remember_workspace_preset('opencode', str(root / 'two'))
            app.remember_workspace_preset('opencode', str(root / 'one'))

        self.assertEqual(
            app.workspace_presets('opencode'),
            [str(root / 'one'), str(root / 'two')],
        )


if __name__ == '__main__':
    unittest.main()
