import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from llama_tui.app import AppConfig, render_terminal_template, terminal_command_for_launcher
from llama_tui.benchmark import safe_bootstrap_candidate_models
from llama_tui.control import CancelToken
from llama_tui.discovery import detected_model_from_path
from llama_tui.hardware import HardwareProfile
from llama_tui.models import ModelConfig
from llama_tui.opencode_benchmark import build_opencode_run_command, run_process_with_metrics, score_opencode_samples


def process_active(pid: int) -> bool:
    stat_path = Path(f'/proc/{pid}/stat')
    try:
        stat = stat_path.read_text()
        end = stat.rfind(')')
        state = stat[end + 2:].split()[0] if end != -1 else ''
        if state in ('Z', 'X'):
            return False
    except Exception:
        pass
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


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

    def test_opencode_run_command_shape(self):
        command = build_opencode_run_command(self.app, self.model, Path('/tmp/project'), 'fix it')
        self.assertEqual(command[:3], ['opencode', 'run', '--pure'])
        self.assertIn('--model', command)
        self.assertIn('local-tiny/tiny-local', command)
        self.assertIn('--agent', command)
        self.assertIn('build', command)
        self.assertIn('--format', command)
        self.assertIn('json', command)
        self.assertIn('--dir', command)

    def test_persistence_fields_round_trip(self):
        model = ModelConfig(
            id='tiny',
            name='Tiny Model',
            path='example/tiny-model',
            alias='tiny-local',
            port=18080,
            runtime='vllm',
            benchmark_fingerprint='abc123',
            default_benchmark_status='done',
            default_benchmark_at='2026-04-14T12:00:00',
        )
        self.app.add_or_update(model)

        loaded = AppConfig(self.config_path).get_model('tiny')
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.benchmark_fingerprint, 'abc123')
        self.assertEqual(loaded.default_benchmark_status, 'done')
        self.assertEqual(loaded.default_benchmark_at, '2026-04-14T12:00:00')


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


class DiscoveryDefaultsTests(unittest.TestCase):
    def test_detected_model_uses_generic_safe_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'Qwen3-30B-A3B-Instruct-Q4_K_M.gguf'
            path.write_bytes(b'not real gguf metadata')
            model = detected_model_from_path(path, [])

        self.assertEqual(model.id, 'qwen3-30b-a3b-instruct-q4-k-m')
        self.assertEqual(model.ctx, 2048)
        self.assertEqual(model.ctx_min, 2048)
        self.assertEqual(model.ctx_max, 131072)
        self.assertEqual(model.ngl, 0)
        self.assertEqual(model.temp, 0.7)
        self.assertEqual(model.output, 2048)
        self.assertEqual(model.optimize_tier, 'safe')
        self.assertEqual(model.memory_reserve_percent, 40)
        self.assertEqual(model.default_benchmark_status, 'pending')

    def test_personal_paths_are_not_embedded_in_constants(self):
        text = Path('llama_tui/constants.py').read_text()
        self.assertNotIn('/var/home/jcampos', text)

    def test_safe_bootstrap_candidates_ignore_model_name(self):
        profile = HardwareProfile(cpu_logical=8, cpu_physical=4, memory_total=16 * 1024**3, memory_available=8 * 1024**3)
        first = ModelConfig(id='a', name='Qwen Opus Coder Gemma', path=__file__, alias='a', port=18100)
        second = ModelConfig(id='b', name='Plain Local Model', path=__file__, alias='b', port=18101)
        first_candidates = safe_bootstrap_candidate_models(first, profile)
        second_candidates = safe_bootstrap_candidate_models(second, profile)

        first_shape = [(preset, tier, c.ctx, c.ngl, c.parallel, c.memory_reserve_percent, c.extra_args) for preset, tier, c, _ in first_candidates]
        second_shape = [(preset, tier, c.ctx, c.ngl, c.parallel, c.memory_reserve_percent, c.extra_args) for preset, tier, c, _ in second_candidates]
        self.assertEqual(first_shape, second_shape)


class ProcessLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = AppConfig(Path(self.tmp.name) / 'models.json')

    def tearDown(self):
        self.tmp.cleanup()

    def child_command(self, pidfile: Path):
        code = (
            'import pathlib, subprocess, sys, time; '
            'child=subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]); '
            'pathlib.Path(sys.argv[1]).write_text(str(child.pid)); '
            'time.sleep(60)'
        )
        return [sys.executable, '-c', code, str(pidfile)]

    def test_timeout_kills_process_group_children(self):
        pidfile = Path(self.tmp.name) / 'child.pid'
        result = run_process_with_metrics(
            self.child_command(pidfile),
            Path(self.tmp.name),
            os.environ.copy(),
            timeout=0.2,
            app=self.app,
        )
        child_pid = int(pidfile.read_text())
        time.sleep(0.3)
        self.assertTrue(result['timed_out'])
        self.assertFalse(process_active(child_pid))

    def test_abort_kills_process_group_children(self):
        pidfile = Path(self.tmp.name) / 'child-abort.pid'
        token = CancelToken()
        timer = threading.Timer(0.2, token.cancel)
        timer.start()
        try:
            result = run_process_with_metrics(
                self.child_command(pidfile),
                Path(self.tmp.name),
                os.environ.copy(),
                timeout=10,
                app=self.app,
                cancel_token=token,
            )
        finally:
            timer.cancel()
        child_pid = int(pidfile.read_text())
        time.sleep(0.3)
        self.assertTrue(result['aborted'])
        self.assertFalse(process_active(child_pid))


if __name__ == '__main__':
    unittest.main()
