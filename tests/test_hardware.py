import tempfile
import unittest
from pathlib import Path

from llama_tui.hardware import (
    _compact_cmdline,
    _known_process_bucket,
    _read_loadavg,
    _read_process_stat,
    bytes_to_gib,
)


class BytesToGibTests(unittest.TestCase):
    def test_conversion(self):
        self.assertEqual(bytes_to_gib(1024 ** 3), 1.0)
        self.assertEqual(bytes_to_gib(0), 0.0)


class ReadProcessStatTests(unittest.TestCase):
    def test_parses_comm_state_and_cpu_ticks(self):
        # Fields after ')': state, then utime at index 11 and stime at index 12.
        rest = ['R'] + ['0'] * 10 + ['100', '50', '0']
        comm, state, ticks = _read_process_stat(f'1234 (python3) {" ".join(rest)}')
        self.assertEqual(comm, 'python3')
        self.assertEqual(state, 'R')
        self.assertEqual(ticks, 150)

    def test_comm_with_spaces_and_parens(self):
        rest = ['S'] + ['0'] * 12
        comm, state, _ticks = _read_process_stat(f'9 (my proc) {" ".join(rest)}')
        self.assertEqual(comm, 'my proc')
        self.assertEqual(state, 'S')

    def test_malformed_returns_zeros(self):
        self.assertEqual(_read_process_stat('garbage'), ('', '', 0))


class CmdlineAndBucketTests(unittest.TestCase):
    def test_compact_cmdline_replaces_nuls(self):
        self.assertEqual(_compact_cmdline('a\x00b\x00c', 'fb'), 'a b c')

    def test_compact_cmdline_falls_back(self):
        self.assertEqual(_compact_cmdline('', 'fallback'), 'fallback')

    def test_known_buckets(self):
        self.assertEqual(_known_process_bucket('brave', ''), 'browser')
        self.assertEqual(_known_process_bucket('llama-server', ''), 'llama')
        self.assertEqual(_known_process_bucket('', 'docker run'), 'container')

    def test_unknown_bucket(self):
        self.assertEqual(_known_process_bucket('randomthing', ''), '')


class ReadLoadavgTests(unittest.TestCase):
    def test_parses_loadavg_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'loadavg').write_text('1.5 2.0 3.0 2/100 12345\n')
            self.assertEqual(_read_loadavg(Path(tmp)), (1.5, 2.0, 3.0, 2, 100))

    def test_missing_file_returns_zeros(self):
        self.assertEqual(_read_loadavg(Path('/no/such/dir')), (0.0, 0.0, 0.0, 0, 0))


if __name__ == '__main__':
    unittest.main()
