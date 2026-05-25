"""Unit tests for the per-engine runtime path helpers.

Covers the module extracted in audit #9 step 2. All assertions patch
``llama_tui.constants.CACHE_DIR`` to a temp root so the tests never
touch the user's real ``~/.cache/llama-tui``.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_tui.runtime_paths import (
    legacy_logfile,
    legacy_pid_metadata_file,
    legacy_pidfile,
    runtime_artifact_dir,
    runtime_artifact_path,
    runtime_logfile,
    runtime_pid_metadata_file,
    runtime_pidfile,
)


class RuntimePathTests(unittest.TestCase):
    def _patched(self, root: Path):
        return patch('llama_tui.constants.CACHE_DIR', root)

    def test_artifact_dir_is_engine_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._patched(root):
                self.assertEqual(runtime_artifact_dir('llama.cpp'), root / 'runtime' / 'llama.cpp')
                self.assertEqual(runtime_artifact_dir('turboquant'), root / 'runtime' / 'turboquant')
                self.assertEqual(runtime_artifact_dir('turboquant'), root / 'runtime' / 'turboquant')
                self.assertEqual(runtime_artifact_dir('llama.cpp-mtp'), root / 'runtime' / 'llama.cpp-mtp')

    def test_artifact_dir_slugifies_unsafe_characters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._patched(root):
                # Spaces and forward slashes get replaced with underscores.
                self.assertEqual(runtime_artifact_dir('my engine'), root / 'runtime' / 'my_engine')
                self.assertEqual(runtime_artifact_dir('a/b'), root / 'runtime' / 'a_b')

    def test_artifact_dir_empty_engine_uses_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._patched(root):
                # Empty / falsy engine keys fall back to llama.cpp.
                self.assertEqual(runtime_artifact_dir(''), root / 'runtime' / 'llama.cpp')
                self.assertEqual(runtime_artifact_dir(None), root / 'runtime' / 'llama.cpp')

    def test_artifact_path_joins_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._patched(root):
                self.assertEqual(
                    runtime_artifact_path('m', '.pid', 'llama.cpp'),
                    root / 'runtime' / 'llama.cpp' / 'm.pid',
                )
                self.assertEqual(
                    runtime_artifact_path('m', '.log', 'turboquant'),
                    root / 'runtime' / 'turboquant' / 'm.log',
                )

    def test_per_engine_pidfile_log_and_metadata_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self._patched(root):
                self.assertEqual(runtime_pidfile('m', 'turboquant'), root / 'runtime' / 'turboquant' / 'm.pid')
                self.assertEqual(runtime_pid_metadata_file('m', 'turboquant'), root / 'runtime' / 'turboquant' / 'm.pid.json')
                self.assertEqual(runtime_logfile('m', 'turboquant'), root / 'runtime' / 'turboquant' / 'm.log')


class LegacyPathTests(unittest.TestCase):
    def test_legacy_paths_use_flat_cache_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('llama_tui.constants.CACHE_DIR', root):
                self.assertEqual(legacy_pidfile('m'), root / 'm.pid')
                self.assertEqual(legacy_pid_metadata_file('m'), root / 'm.pid.json')
                self.assertEqual(legacy_logfile('m'), root / 'm.log')


if __name__ == '__main__':
    unittest.main()
