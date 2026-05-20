import tempfile
import unittest
from pathlib import Path

from llama_tui.hardware import (
    _compact_cmdline,
    _known_process_bucket,
    _read_loadavg,
    _read_process_stat,
    bytes_to_gib,
    clamp_memory_to_cgroup,
    read_cgroup_memory_limits,
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


class CgroupMemoryTests(unittest.TestCase):
    def _write_cgroup(self, root: Path, max_value: str, current_value: str = '0'):
        (root / 'memory.max').write_text(max_value)
        (root / 'memory.current').write_text(current_value)

    def test_read_cgroup_max_and_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_cgroup(root, '4294967296', '1073741824')
            limits = read_cgroup_memory_limits(root)
            self.assertEqual(limits['max'], 4294967296)
            self.assertEqual(limits['current'], 1073741824)

    def test_max_literal_means_unlimited(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_cgroup(root, 'max', '0')
            limits = read_cgroup_memory_limits(root)
            self.assertNotIn('max', limits)
            self.assertEqual(limits['current'], 0)

    def test_missing_cgroup_returns_empty(self):
        self.assertEqual(read_cgroup_memory_limits(Path('/no/such/cgroup/root')), {})

    def test_clamp_caps_host_memory_to_cgroup_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_cgroup(root, '4294967296', '1073741824')  # 4 GiB cap, 1 GiB used
            host_total = 64 * 1024**3
            host_avail = 50 * 1024**3
            total, available = clamp_memory_to_cgroup(host_total, host_avail, root)
            self.assertEqual(total, 4294967296)
            self.assertEqual(available, 4294967296 - 1073741824)

    def test_clamp_passthrough_when_no_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_cgroup(root, 'max', '0')
            total, available = clamp_memory_to_cgroup(64 * 1024**3, 50 * 1024**3, root)
            self.assertEqual(total, 64 * 1024**3)
            self.assertEqual(available, 50 * 1024**3)


if __name__ == '__main__':
    unittest.main()
