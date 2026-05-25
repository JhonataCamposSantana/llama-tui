"""Pure /proc + signal helpers for managed-process supervision.

Fifth extraction of audit finding #9 (split ``AppConfig``). Owns the
PID-probe and signal primitives that don't need any AppConfig state:

  - ``proc_state(pid)`` reads the state letter (``R``, ``S``, ``Z``…)
    from ``/proc/{pid}/stat``.
  - ``pid_alive(pid, include_zombie=False)`` checks
    ``os.kill(pid, 0)`` and optionally treats zombies/dead as alive.
  - ``pid_cmdline_parts(pid)`` reads ``/proc/{pid}/cmdline`` and splits
    on NULs.
  - ``process_group_pids(pgid)`` enumerates ``/proc`` for live processes
    in the given group.
  - ``reap_pid(pid)`` ``waitpid(WNOHANG)`` until the kernel reports
    EAGAIN, so a finished child stays out of the zombie table.
  - ``send_signal(pid, sig, use_group)`` delivers a signal either to the
    process group (when the target is the group leader of a foreign
    group) or to the bare PID; returns ``(pgid, used_group)``.

The fancier wrappers that combine these with AppConfig state
(``_pid_looks_like_runtime``, ``_pid_matches_model``,
``terminate_process_group``, the start/stop orchestrators) stay in
AppConfig because they need ``llama_server`` / ``runtime_profile``
to decide what counts as "our" process.
"""

import os
import signal
from pathlib import Path
from typing import List, Optional, Tuple


def proc_state(pid: int) -> Optional[str]:
    """Return the state letter from ``/proc/{pid}/stat`` or ``None``.

    Returns ``None`` when the file is missing (process has exited),
    unreadable, or has a malformed schema. Callers should treat the
    None case as "no information"; ``pid_alive`` is the canonical
    check for "is this PID still running".
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        end = stat.rfind(")")
        if end == -1:
            return None
        rest = stat[end + 2:].split()
        return rest[0] if rest else None
    except OSError:
        return None


def pid_alive(pid: int, include_zombie: bool = False) -> bool:
    """True when ``pid`` exists and is not a zombie/dead.

    ``include_zombie=True`` keeps zombies in the "alive" set, which is
    what the stop path wants — a zombie still needs reaping before its
    pidfile is recyclable.
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    state = proc_state(pid)
    if not include_zombie and state in ('Z', 'X'):
        return False
    return True


def pid_cmdline_parts(pid: int) -> List[str]:
    """Return the NUL-split ``/proc/{pid}/cmdline`` argv list.

    Empty list when the process has exited or the file is unreadable.
    Embedded NULs are dropped so the result is a clean argv-style list.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return [p.decode(errors='ignore') for p in raw.split(b'\x00') if p]
    except OSError:
        return []


def process_group_pids(pgid: int) -> List[int]:
    """Return PIDs whose group ID matches ``pgid``, excluding zombies/dead."""
    pids: List[int] = []
    try:
        proc_root = Path('/proc')
        entries = proc_root.iterdir()
    except OSError:
        return pids
    for proc_dir in entries:
        if not proc_dir.name.isdigit():
            continue
        try:
            pid = int(proc_dir.name)
            if os.getpgid(pid) != pgid:
                continue
            if proc_state(pid) in ('Z', 'X'):
                continue
            pids.append(pid)
        except (OSError, ValueError):
            continue
    return pids


def reap_pid(pid: int) -> None:
    """``waitpid(WNOHANG)`` loop so a finished child stays out of the
    zombie table.

    Safe to call when the caller does not actually own the PID — the
    ``ChildProcessError`` / ``OSError`` are swallowed.
    """
    try:
        while True:
            reaped, _status = os.waitpid(pid, os.WNOHANG)
            if reaped == 0:
                break
            if reaped == pid:
                break
    except ChildProcessError:
        pass
    except OSError:
        pass


def send_signal(pid: int, sig, use_group: bool) -> Tuple[Optional[int], bool]:
    """Deliver ``sig`` to ``pid`` or its group; return ``(pgid, used_group)``.

    When ``use_group`` is True and the target is the leader of a
    different process group from this one, ``killpg`` is used; otherwise
    ``os.kill`` delivers to the bare PID. ``pgid`` is the resolved group
    id (``None`` if it could not be read).
    """
    pgid: Optional[int] = None
    if use_group:
        try:
            pgid = os.getpgid(pid)
        except OSError:
            pgid = None
        if pgid is not None and pgid != os.getpgrp():
            os.killpg(pgid, sig)
            return pgid, True
    os.kill(pid, sig)
    return pgid, False
