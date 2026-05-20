import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from llama_tui.hardware import (
    APPLE_METAL_WORKING_SET_FRACTION,
    _compact_cmdline,
    _known_process_bucket,
    _read_loadavg,
    _read_process_stat,
    bytes_to_gib,
    clamp_memory_to_cgroup,
    probe_amd_rocm_gpu,
    probe_apple_silicon_gpu,
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


class ProbeAmdRocmGpuTests(unittest.TestCase):
    _SAMPLE_CSV = (
        'device,Card series,VRAM Total Memory (B),VRAM Total Used Memory (B)\n'
        'card0,Radeon RX 7900 XTX,25757220864,2147483648\n'
    )

    def test_returns_empty_when_rocm_smi_missing(self):
        with patch('llama_tui.hardware.shutil.which', return_value=None):
            self.assertEqual(probe_amd_rocm_gpu(), ('', 0, 0, ''))

    def test_parses_csv_into_total_and_free(self):
        fake = types.SimpleNamespace(returncode=0, stdout=self._SAMPLE_CSV, stderr='')
        with patch('llama_tui.hardware.shutil.which', return_value='/usr/bin/rocm-smi'), \
                patch('llama_tui.hardware.subprocess.run', return_value=fake):
            name, total, free, error = probe_amd_rocm_gpu()
        self.assertEqual(name, 'Radeon RX 7900 XTX')
        self.assertEqual(total, 25757220864)
        self.assertEqual(free, 25757220864 - 2147483648)
        self.assertEqual(error, '')

    def test_returns_error_message_on_nonzero_exit(self):
        fake = types.SimpleNamespace(returncode=1, stdout='', stderr='rocm-smi: no devices\n')
        with patch('llama_tui.hardware.shutil.which', return_value='/usr/bin/rocm-smi'), \
                patch('llama_tui.hardware.subprocess.run', return_value=fake):
            name, total, free, error = probe_amd_rocm_gpu()
        self.assertEqual((name, total, free), ('', 0, 0))
        self.assertIn('no devices', error)

    def test_subprocess_exception_is_surfaced_as_error(self):
        with patch('llama_tui.hardware.shutil.which', return_value='/usr/bin/rocm-smi'), \
                patch('llama_tui.hardware.subprocess.run', side_effect=OSError('boom')):
            name, total, free, error = probe_amd_rocm_gpu()
        self.assertEqual((name, total, free), ('', 0, 0))
        self.assertIn('boom', error)


class ProbeAppleSiliconGpuTests(unittest.TestCase):
    _VM_STAT_OUTPUT = (
        'Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
        'Pages free:                               65536.\n'
        'Pages active:                           1000000.\n'
        'Pages speculative:                         4096.\n'
    )

    def _which_factory(self):
        # Route shutil.which by command name so sysctl and vm_stat resolve to
        # distinct paths; the subprocess dispatcher uses cmd[0] to pick the
        # canned response.
        def fake_which(name):
            if name == 'sysctl':
                return '/usr/sbin/sysctl'
            if name == 'vm_stat':
                return '/usr/bin/vm_stat'
            return None
        return fake_which

    def _sysctl_factory(self, mapping):
        def fake_run(cmd, *args, **kwargs):
            if cmd[0].endswith('sysctl'):
                key = cmd[-1]
                value = mapping.get(key, '')
                return types.SimpleNamespace(returncode=0, stdout=str(value) + '\n', stderr='')
            if cmd[0].endswith('vm_stat'):
                return types.SimpleNamespace(returncode=0, stdout=self._VM_STAT_OUTPUT, stderr='')
            return types.SimpleNamespace(returncode=1, stdout='', stderr='unhandled')
        return fake_run

    def test_returns_empty_on_non_darwin(self):
        with patch('llama_tui.hardware.sys.platform', 'linux'):
            self.assertEqual(probe_apple_silicon_gpu(), ('', 0, 0, ''))

    def test_apple_silicon_reports_metal_working_set_fraction(self):
        total_ram = 32 * 1024**3
        mapping = {
            'hw.memsize': total_ram,
            'machdep.cpu.brand_string': 'Apple M2 Pro',
        }
        free_pages = (65536 + 4096) * 16384  # vm_stat free+speculative * 16k page
        with patch('llama_tui.hardware.sys.platform', 'darwin'), \
                patch('llama_tui.hardware.shutil.which', side_effect=self._which_factory()), \
                patch('llama_tui.hardware.subprocess.run', side_effect=self._sysctl_factory(mapping)):
            name, gpu_total, gpu_free, error = probe_apple_silicon_gpu()
        expected_total = int(total_ram * APPLE_METAL_WORKING_SET_FRACTION)
        expected_free = int(min(free_pages, total_ram) * APPLE_METAL_WORKING_SET_FRACTION)
        self.assertEqual(name, 'Apple M2 Pro')
        self.assertEqual(gpu_total, expected_total)
        self.assertEqual(gpu_free, expected_free)
        self.assertEqual(error, '')

    def test_apple_silicon_env_override_replaces_fraction(self):
        total_ram = 16 * 1024**3
        mapping = {
            'hw.memsize': total_ram,
            'machdep.cpu.brand_string': 'Apple M1',
        }
        prev = os.environ.get('LLAMA_TUI_APPLE_METAL_FRACTION')
        try:
            os.environ['LLAMA_TUI_APPLE_METAL_FRACTION'] = '0.5'
            with patch('llama_tui.hardware.sys.platform', 'darwin'), \
                    patch('llama_tui.hardware.shutil.which', side_effect=self._which_factory()), \
                    patch('llama_tui.hardware.subprocess.run', side_effect=self._sysctl_factory(mapping)):
                _name, gpu_total, _gpu_free, _error = probe_apple_silicon_gpu()
            self.assertEqual(gpu_total, total_ram // 2)
        finally:
            if prev is None:
                os.environ.pop('LLAMA_TUI_APPLE_METAL_FRACTION', None)
            else:
                os.environ['LLAMA_TUI_APPLE_METAL_FRACTION'] = prev

    def test_apple_silicon_returns_empty_when_memsize_unknown(self):
        mapping = {'hw.memsize': ''}
        with patch('llama_tui.hardware.sys.platform', 'darwin'), \
                patch('llama_tui.hardware.shutil.which', side_effect=self._which_factory()), \
                patch('llama_tui.hardware.subprocess.run', side_effect=self._sysctl_factory(mapping)):
            self.assertEqual(probe_apple_silicon_gpu(), ('', 0, 0, ''))


if __name__ == '__main__':
    unittest.main()
