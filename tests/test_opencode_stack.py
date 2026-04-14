import tempfile
import unittest
from pathlib import Path

from llama_tui.app import AppConfig, render_terminal_template, terminal_command_for_launcher
from llama_tui.models import ModelConfig
from llama_tui.opencode_benchmark import score_opencode_samples


class OpencodeStackHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = Path(self.tmp.name) / 'models.json'
        self.app = AppConfig(self.config_path)
        self.model = ModelConfig(
            id='tiny',
            name='Tiny Model',
            path=__file__,
            alias='tiny-local',
            port=18080,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_workspace_validation(self):
        ok, path, reason = self.app.validate_workspace_path(self.tmp.name)
        self.assertTrue(ok)
        self.assertEqual(path, Path(self.tmp.name).resolve())
        self.assertEqual(reason, '')

        ok, path, reason = self.app.validate_workspace_path(Path(self.tmp.name) / 'missing')
        self.assertFalse(ok)
        self.assertIsNone(path)
        self.assertIn('does not exist', reason)

    def test_opencode_model_ref(self):
        self.assertEqual(self.app.opencode_provider_key(self.model), 'local-tiny')
        self.assertEqual(self.app.opencode_model_ref(self.model), 'local-tiny/tiny-local')

    def test_default_terminal_command_shape(self):
        command = terminal_command_for_launcher('xterm', 'OpenCode Tiny', Path('/tmp/project'), 'echo hello')
        self.assertEqual(command[:4], ['xterm', '-T', 'OpenCode Tiny', '-e'])
        self.assertIn('cd /tmp/project && echo hello', command)

    def test_terminal_template_quotes_command(self):
        command = render_terminal_template(
            'xterm -T {title} -e bash -lc {cmd}',
            'OpenCode Tiny',
            Path('/tmp/project'),
            'echo hello && sleep 1',
        )
        self.assertEqual(command, ['xterm', '-T', 'OpenCode Tiny', '-e', 'bash', '-lc', 'echo hello && sleep 1'])

    def test_opencode_shell_command_uses_config_and_model(self):
        self.app.opencode.path = '/tmp/opencode.json'
        command = self.app.build_opencode_shell_command(self.model, Path('/tmp/project'))
        self.assertIn('OPENCODE_CONFIG=/tmp/opencode.json', command)
        self.assertIn("--model local-tiny/tiny-local", command)
        self.assertIn('cd /tmp/project', command)


class OpencodeWorkflowScoreTests(unittest.TestCase):
    def test_successful_fast_samples_score_above_failures(self):
        good = score_opencode_samples([
            {
                'ok': True,
                'elapsed': 10.0,
                'first_output': 2.0,
                'min_ram_available': 4 * 1024**3,
                'min_gpu_memory_free': 2 * 1024**3,
            },
            {
                'ok': True,
                'elapsed': 12.0,
                'first_output': 3.0,
                'min_ram_available': 3 * 1024**3,
                'min_gpu_memory_free': 2 * 1024**3,
            },
        ])
        bad = score_opencode_samples([
            {
                'ok': False,
                'elapsed': 80.0,
                'first_output': 30.0,
                'min_ram_available': 700 * 1024**2,
                'min_gpu_memory_free': 200 * 1024**2,
            }
        ])
        self.assertGreater(good, bad)


if __name__ == '__main__':
    unittest.main()
