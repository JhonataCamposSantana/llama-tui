"""Integration tests for AppConfig.cleanup_managed_processes.

Covers the shutdown-walk used by main.py's atexit handler. Without these,
audit finding #5 (daemon-thread / cleanup-race) lacked a direct safety net
and the runtime cache layout drift could go undetected by the suite.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_tui.app import AppConfig


class CleanupManagedProcessesTests(unittest.TestCase):
    def _make_app(self, cache_dir: Path, config_dir: Path, data_dir: Path) -> AppConfig:
        config_path = config_dir / 'models.json'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text('{"models": []}', encoding='utf-8')
        return AppConfig(config_path)

    def _write_pidfile(self, runtime_root: Path, engine: str, model_id: str, pid: int, pgid: int):
        engine_dir = runtime_root / engine
        engine_dir.mkdir(parents=True, exist_ok=True)
        pidfile = engine_dir / f'{model_id}.pid'
        pidfile.write_text(str(pid))
        meta = engine_dir / f'{model_id}.pid.json'
        meta.write_text(json.dumps({'pid': pid, 'pgid': pgid, 'command': 'llama-server', 'cwd': str(engine_dir)}))
        return pidfile, meta

    def test_dead_pids_are_unlinked_without_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / 'cache'
            cache_dir.mkdir(parents=True)
            runtime_root = cache_dir / 'runtime'
            pidfile, meta = self._write_pidfile(runtime_root, 'llama.cpp', 'sample-model', 9001, 9001)
            with patch('llama_tui.app.CACHE_DIR', cache_dir), \
                    patch('llama_tui.app.CONFIG_DIR', root / 'config'), \
                    patch('llama_tui.app.DATA_DIR', root / 'data'):
                app = self._make_app(cache_dir, root / 'config', root / 'data')
                with patch.object(app, '_pid_alive', return_value=False), \
                        patch.object(app, '_pid_looks_like_any_runtime', return_value=False), \
                        patch.object(app, '_process_group_pids', return_value=[]):
                    msgs = app.cleanup_managed_processes()
        self.assertFalse(pidfile.exists())
        self.assertFalse(meta.exists())
        # No 'stop' message accumulated for a dead PID — only stop_all() ran.
        self.assertTrue(all('sample-model' not in msg or 'managed_only' in msg or msg.endswith('skipped') for msg in msgs) or msgs == [])

    def test_alive_managed_pid_is_stopped_and_cleaned_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / 'cache'
            cache_dir.mkdir(parents=True)
            runtime_root = cache_dir / 'runtime'
            pidfile, meta = self._write_pidfile(runtime_root, 'llama.cpp', 'live-model', 9100, 9100)
            stop_calls = []

            def fake_stop_pid(model_id, pid, use_group):
                stop_calls.append((model_id, pid, use_group))
                return True, f'stopped pid {pid}'

            with patch('llama_tui.app.CACHE_DIR', cache_dir), \
                    patch('llama_tui.app.CONFIG_DIR', root / 'config'), \
                    patch('llama_tui.app.DATA_DIR', root / 'data'):
                app = self._make_app(cache_dir, root / 'config', root / 'data')
                with patch.object(app, '_pid_alive', return_value=True), \
                        patch.object(app, '_pid_looks_like_any_runtime', return_value=True), \
                        patch.object(app, '_process_group_pids', return_value=[9100]), \
                        patch.object(app, '_stop_pid', side_effect=fake_stop_pid):
                    msgs = app.cleanup_managed_processes()

        self.assertFalse(pidfile.exists())
        self.assertFalse(meta.exists())
        self.assertEqual(stop_calls, [('live-model', 9100, True)])
        self.assertTrue(any('live-model' in msg and 'stopped pid 9100' in msg for msg in msgs))

    def test_idempotent_on_second_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / 'cache'
            cache_dir.mkdir(parents=True)
            (cache_dir / 'runtime').mkdir()
            with patch('llama_tui.app.CACHE_DIR', cache_dir), \
                    patch('llama_tui.app.CONFIG_DIR', root / 'config'), \
                    patch('llama_tui.app.DATA_DIR', root / 'data'):
                app = self._make_app(cache_dir, root / 'config', root / 'data')
                first = app.cleanup_managed_processes()
                second = app.cleanup_managed_processes()
        # First call runs cleanup; second is a no-op so msgs is empty.
        self.assertEqual(second, [])

    def test_legacy_top_level_pidfile_is_unlinked_when_dead(self):
        # Older versions wrote pidfiles directly at CACHE_DIR/*.pid rather
        # than under runtime/{engine}/. The cleanup walk must still find
        # and remove those when the PID is dead.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / 'cache'
            cache_dir.mkdir(parents=True)
            legacy = cache_dir / 'legacy-model.pid'
            legacy.write_text('5555')
            (cache_dir / 'legacy-model.pid.json').write_text(json.dumps({'pid': 5555}))
            with patch('llama_tui.app.CACHE_DIR', cache_dir), \
                    patch('llama_tui.app.CONFIG_DIR', root / 'config'), \
                    patch('llama_tui.app.DATA_DIR', root / 'data'):
                app = self._make_app(cache_dir, root / 'config', root / 'data')
                with patch.object(app, '_pid_alive', return_value=False), \
                        patch.object(app, '_pid_looks_like_any_runtime', return_value=False), \
                        patch.object(app, '_process_group_pids', return_value=[]):
                    app.cleanup_managed_processes()
        self.assertFalse(legacy.exists())
        self.assertFalse((cache_dir / 'legacy-model.pid.json').exists())


if __name__ == '__main__':
    unittest.main()
