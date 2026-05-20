"""Unit tests for the pure /proc + signal primitives.

These exercise the helpers extracted from AppConfig in audit finding #9
step 5. They run in the current process so they exercise real /proc
state — the test process itself is always present, so ``pid_alive`` /
``pid_cmdline_parts`` / ``proc_state`` for ``os.getpid()`` are reliable
fixtures.
"""

import os
import signal
import time
import unittest
from unittest.mock import patch

from llama_tui.process_supervisor import (
    pid_alive,
    pid_cmdline_parts,
    proc_state,
    process_group_pids,
    reap_pid,
    send_signal,
)


class ProcStateTests(unittest.TestCase):
    def test_current_process_state_is_letter(self):
        state = proc_state(os.getpid())
        # The test process is at least Running ('R') or Sleeping ('S')
        # while the assertion runs.
        self.assertIsNotNone(state)
        self.assertIn(state, ('R', 'S', 'D'))

    def test_nonexistent_pid_returns_none(self):
        # PID 2**30 is well above any real process.
        self.assertIsNone(proc_state(2**30))


class PidAliveTests(unittest.TestCase):
    def test_current_pid_is_alive(self):
        self.assertTrue(pid_alive(os.getpid()))

    def test_nonexistent_pid_is_not_alive(self):
        self.assertFalse(pid_alive(2**30))

    def test_parent_pid_is_alive(self):
        # The parent of the test process is the unittest runner —
        # always present and we have permission to signal it.
        self.assertTrue(pid_alive(os.getppid()))


class PidCmdlineTests(unittest.TestCase):
    def test_current_process_cmdline_contains_python(self):
        parts = pid_cmdline_parts(os.getpid())
        self.assertTrue(parts)
        # The test runner is invoked as python -m unittest ..., so
        # the argv list must include at least one part referencing
        # python or unittest.
        joined = ' '.join(parts).lower()
        self.assertTrue(
            'python' in joined or 'unittest' in joined,
            f'unexpected cmdline: {parts}',
        )

    def test_nonexistent_pid_returns_empty_list(self):
        self.assertEqual(pid_cmdline_parts(2**30), [])


class ProcessGroupPidsTests(unittest.TestCase):
    def test_current_process_group_contains_us(self):
        my_pgid = os.getpgid(os.getpid())
        pids = process_group_pids(my_pgid)
        self.assertIn(os.getpid(), pids)

    def test_nonexistent_group_returns_empty(self):
        # A pgid that no real process has will yield nothing.
        self.assertEqual(process_group_pids(2**30), [])


class ReapPidTests(unittest.TestCase):
    def test_reap_unknown_pid_does_not_raise(self):
        # We do not own PID 2**30, so waitpid should raise
        # ChildProcessError under the hood — the helper swallows it.
        reap_pid(2**30)

    def test_reap_actual_child(self):
        # Fork a tiny child that exits immediately and confirm reap_pid
        # collects it cleanly so it never lingers as a zombie.
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        time.sleep(0.05)  # give the child a chance to exit
        reap_pid(pid)
        # After reaping, the kernel no longer reports the PID via
        # /proc/{pid}/stat — pid_alive should now be False.
        # (Linux may briefly keep it as a zombie if reap_pid raced;
        # the helper's waitpid loop handles that.)
        self.assertFalse(pid_alive(pid, include_zombie=False))


class SendSignalTests(unittest.TestCase):
    def test_send_signal_uses_kill_when_group_disabled(self):
        kills = []

        def fake_kill(pid, sig):
            kills.append((pid, sig))

        with patch('llama_tui.process_supervisor.os.kill', side_effect=fake_kill):
            pgid, used_group = send_signal(12345, signal.SIGTERM, use_group=False)
        self.assertEqual(kills, [(12345, signal.SIGTERM)])
        self.assertIsNone(pgid)
        self.assertFalse(used_group)

    def test_send_signal_falls_back_to_kill_when_pgid_lookup_fails(self):
        kills = []

        def fake_kill(pid, sig):
            kills.append((pid, sig))

        with patch('llama_tui.process_supervisor.os.getpgid', side_effect=OSError('no such process')), \
                patch('llama_tui.process_supervisor.os.kill', side_effect=fake_kill):
            pgid, used_group = send_signal(12345, signal.SIGTERM, use_group=True)
        self.assertEqual(kills, [(12345, signal.SIGTERM)])
        self.assertIsNone(pgid)
        self.assertFalse(used_group)

    def test_send_signal_uses_killpg_when_pgid_is_foreign(self):
        killpgs = []

        def fake_killpg(pgid, sig):
            killpgs.append((pgid, sig))

        with patch('llama_tui.process_supervisor.os.getpgid', return_value=9999), \
                patch('llama_tui.process_supervisor.os.getpgrp', return_value=1234), \
                patch('llama_tui.process_supervisor.os.killpg', side_effect=fake_killpg):
            pgid, used_group = send_signal(12345, signal.SIGTERM, use_group=True)
        self.assertEqual(killpgs, [(9999, signal.SIGTERM)])
        self.assertEqual(pgid, 9999)
        self.assertTrue(used_group)


if __name__ == '__main__':
    unittest.main()
